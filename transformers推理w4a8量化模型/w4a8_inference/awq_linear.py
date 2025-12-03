from __future__ import annotations

import torch
import torch.nn as nn


def _unpack_awq_qweight_to_int8(
    qweight_int32: torch.Tensor,
    n_bits: int = 4,
) -> torch.Tensor:
    """Unpack packed int32 AWQ weights into an int8 weight matrix.

    权重打包格式与 AutoAWQ / vLLM 的 W4A8 一致：
    - qweight: [K, N_pack]，每个 int32 里打 8 个 4bit 权重。
    - 使用 order_map = [0, 2, 4, 6, 1, 3, 5, 7] 进行列重排。

    返回的 int8 矩阵形状为 [K, N_full]，数值范围 [-8, 7]，可直接作为
    W8A8-int8 算子的权重输入（只是值域更小）。
    """

    if qweight_int32.dtype != torch.int32:
        qweight_int32 = qweight_int32.to(torch.int32)

    pack_num = 32 // n_bits  # 8 values per int32 word
    k_dim, n_pack = qweight_int32.shape
    n_full = n_pack * pack_num

    device = qweight_int32.device
    weight_int8 = torch.empty((k_dim, n_full), dtype=torch.int8, device=device)

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    mask = (1 << n_bits) - 1
    sign_bit = 1 << (n_bits - 1)

    cols = torch.arange(n_pack, device=device)
    dst_base = cols * pack_num

    for i in range(pack_num):
        shift = i * n_bits
        vals = (qweight_int32 >> shift) & mask  # [K, N_pack]
        signed = vals - ((vals & sign_bit) << 1)
        dst_cols = dst_base + order_map[i]      # [N_pack]
        weight_int8[:, dst_cols] = signed.to(torch.int8)

    return weight_int8


class AwqW4A8Linear(nn.Module):
    """Linear 层：权重来自 packed W4A8，解包为 int8，用 W8A8-int8 路径做运算。

    实现思路：
    - set_awq_params 阶段：
      * 从 packed 4bit 权重 + per-channel scales 解码出 int8 权重矩阵 [K, N]；
      * 只保留 int8 权重和 float scale（不做提前反量化到 float）；
    - forward 阶段：
      * 对激活做 per-token 对称 int8 量化；
      * 用 int8×int8 的 GEMM（PyTorch matmul 实现 W8A8 算子形式）；
      * 再乘以 per-token / per-channel scale 还原为浮点结果。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        use_triton: bool = True,  # Default to using Triton
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.out_dtype = dtype
        self.use_triton = use_triton

        # W8A8-int8 路径用到的量：int8 权重 + float scale
        self.register_buffer(
            "qweight_int8",
            torch.empty(0, 0, dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "w_scales",
            torch.empty(0, dtype=torch.float32),
            persistent=True,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def set_awq_params(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        """从 packed 4bit 权重和 scale 生成 int8 权重矩阵 + per-channel scale。

        Args:
            qweight: [K, N_pack] int32 packed 4-bit weights.
            scales: [1, N] 或 [N] per-output-channel scales（float）。
        """

        if qweight.ndim != 2:
            raise ValueError(f"qweight must be 2D, got shape {tuple(qweight.shape)}")

        # 先搬到 CPU 做一次性解包，避免在 GPU 上做大量 bit 运算
        target_device = qweight.device
        q_cpu = qweight.to("cpu", dtype=torch.int32)
        s_cpu = scales.to("cpu", dtype=torch.float32)

        k_dim, n_pack = q_cpu.shape
        pack_num = 8  # 32 // 4
        n_full = n_pack * pack_num

        if s_cpu.ndim == 2:
            s_cpu = s_cpu.view(-1)
        if s_cpu.numel() != n_full:
            raise ValueError(
                f"scales length {s_cpu.numel()} != unpacked out_features {n_full}"
            )

        if self.in_features != k_dim:
            raise ValueError(
                f"AwqW4A8Linear.in_features={self.in_features} does not match "
                f"qweight K dim={k_dim}"
            )
        if self.out_features != n_full:
            raise ValueError(
                f"AwqW4A8Linear.out_features={self.out_features} does not match "
                f"unpacked N dim={n_full} from qweight"
            )

        # CPU 上解包为 int8 权重矩阵 [K, N]
        weight_int8_cpu = _unpack_awq_qweight_to_int8(q_cpu)  # [K, N]

        # 搬回目标 device，后续 int8×int8 GEMM 直接用它
        self.qweight_int8 = weight_int8_cpu.to(target_device, dtype=torch.int8).contiguous()
        self.w_scales = s_cpu.to(target_device, dtype=torch.float32).view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.qweight_int8.numel() == 0 or self.w_scales.numel() == 0:
            raise RuntimeError(
                "AwqW4A8Linear has empty qweight_int8/w_scales. "
                "Make sure set_awq_params() was called with checkpoint tensors."
            )

        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.size(-1)}"
            )

        orig_shape = x.shape
        device = x.device

        if self.use_triton:
            try:
                from .awq_triton_int8 import awq_int8_gemm_triton
            except ImportError:
                raise ImportError(
                    "Triton is not installed. Please install it with `pip install triton`."
                )
            # Triton path handles its own reshaping and quantization
            out_fp32, act_scales, w_scales = awq_int8_gemm_triton(
                x, self.qweight_int8, self.w_scales
            )
            # Rescale output
            out_fp32 = out_fp32 * act_scales.view(-1, 1) * w_scales.view(1, -1)
        else:
            # PyTorch fallback path
            x_2d = x.reshape(-1, self.in_features)  # [M, K]

            # 1) Per-token symmetric int8 quantization for activations
            max_abs = x_2d.abs().max(dim=-1, keepdim=True).values
            act_scales = (max_abs / 127.0).clamp(min=1e-8)
            x_q = torch.round(x_2d / act_scales).clamp(-127, 127).to(torch.int8)

            # 2) W8A8-int8 GEMM: int8*int8 -> float32
            out_fp32 = x_q.to(torch.float32) @ self.qweight_int8.to(torch.float32)

            # 3) Rescale with per-token and per-channel scales
            act_scales_1d = act_scales.view(-1)
            w_scales = self.w_scales.to(device=device, dtype=out_fp32.dtype)
            out_fp32 = out_fp32 * act_scales_1d.view(-1, 1) * w_scales.view(1, -1)

        # 4) Add bias, reshape, and cast
        if self.bias is not None:
            out_fp32 = out_fp32 + self.bias.to(device=device, dtype=out_fp32.dtype)

        out = out_fp32.reshape(*orig_shape[:-1], self.out_features)
        return out.to(x.dtype if x.is_floating_point() else self.out_dtype)
