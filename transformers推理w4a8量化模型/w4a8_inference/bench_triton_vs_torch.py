from __future__ import annotations

import time

import torch
from safetensors import safe_open

from .awq_linear import _unpack_awq_qweight_to_int8
from .awq_triton_int8 import (
    per_token_quant_int8_triton,
    int8_gemm_triton,
    awq_int8_gemm_triton,
)


MODEL_DIR = "/nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed"
CKPT_SHARD = f"{MODEL_DIR}/model-00001-of-00007.safetensors"


def load_one_layer_int8_weight(layer_name: str = "model.layers.0.mlp.down_proj"):
    qkey = f"{layer_name}.qweight"
    skey = f"{layer_name}.scales"
    with safe_open(CKPT_SHARD, framework="pt", device="cpu") as f:
        qweight = f.get_tensor(qkey)
        scales = f.get_tensor(skey)

    qweight = qweight.to(torch.int32)
    scales = scales.to(torch.float32)
    w_int8 = _unpack_awq_qweight_to_int8(qweight)
    w_scales = scales.view(-1).to(torch.float32)
    return w_int8.cuda(), w_scales.cuda()


def time_fn(fn, name: str, iters: int = 50) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000 / iters
    print(f"[BENCH] {name:<25}: {ms:.4f} ms / call")
    return ms


def compare_tensors(t1, t2, name: str):
    diff = (t1.to(torch.float32) - t2.to(torch.float32)).abs()
    print(
        f"[CHECK] {name:<25}: max_diff={diff.max().item():.6f}, "
        f"mean_diff={diff.mean().item():.6f}"
    )


def bench_once(M: int = 128, layer_name: str = "model.layers.0.mlp.down_proj") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Benchmark requires CUDA GPU.")

    print(f"--- Loading weights for {layer_name} ---")
    w_int8, w_scales = load_one_layer_int8_weight(layer_name)
    K, N = w_int8.shape
    print(f"Weight shape: K={K}, N={N}, Batch size M={M}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # --- 1. Per-token Quantization Benchmark ---
    print("\n--- 1. Per-token Quantization Benchmark ---")

    def quant_torch_path():
        max_abs = x.abs().amax(dim=-1, keepdim=True)
        scales = (max_abs / 127.0).clamp(min=1e-8)
        x_q = torch.floor(x / scales + 0.5).clamp(-127, 127).to(torch.int8)
        return x_q, scales.view(-1)

    def quant_triton_path():
        return per_token_quant_int8_triton(x)

    # Warmup
    for _ in range(5):
        quant_torch_path()
        quant_triton_path()
    
    torch_quant_ms = time_fn(quant_torch_path, "quant_torch_path")
    triton_quant_ms = time_fn(quant_triton_path, "quant_triton_path")
    print(f"Speedup: {torch_quant_ms / triton_quant_ms:.3f}x")

    x_q_torch, scales_torch = quant_torch_path()
    x_q_triton, scales_triton = quant_triton_path()
    compare_tensors(x_q_torch, x_q_triton, "Quantized Activations (x_q)")
    compare_tensors(scales_torch, scales_triton, "Activation Scales")

    # --- 2. W8A8 GEMM Benchmark ---
    print("\n--- 2. W8A8 GEMM Benchmark ---")
    x_q = x_q_torch  # Use the same quantized input for both

    def gemm_torch_path():
        return x_q.to(torch.float32) @ w_int8.to(torch.float32)

    def gemm_triton_path():
        return int8_gemm_triton(x_q, w_int8)

    for _ in range(5):
        gemm_torch_path()
        gemm_triton_path()

    torch_gemm_ms = time_fn(gemm_torch_path, "gemm_torch_path")
    triton_gemm_ms = time_fn(gemm_triton_path, "gemm_triton_path")
    print(f"Speedup: {torch_gemm_ms / triton_gemm_ms:.3f}x")

    c_torch = gemm_torch_path()
    c_triton = gemm_triton_path()
    compare_tensors(c_torch, c_triton, "GEMM Output (C)")

    # --- 3. End-to-End Benchmark ---
    print("\n--- 3. End-to-End Benchmark ---")

    def e2e_torch_path():
        x_q, scales = quant_torch_path()
        c = x_q.to(torch.float32) @ w_int8.to(torch.float32)
        return c * scales.view(-1, 1) * w_scales.view(1, -1)

    def e2e_triton_path():
        c, act_s, w_s = awq_int8_gemm_triton(x, w_int8, w_scales)
        return c * act_s.view(-1, 1) * w_s.view(1, -1)

    for _ in range(5):
        e2e_torch_path()
        e2e_triton_path()

    torch_e2e_ms = time_fn(e2e_torch_path, "e2e_torch_path")
    triton_e2e_ms = time_fn(e2e_triton_path, "e2e_triton_path")
    print(f"Speedup: {torch_e2e_ms / triton_e2e_ms:.3f}x")

    y_torch = e2e_torch_path()
    y_triton = e2e_triton_path()
    compare_tensors(y_torch, y_triton, "Final Output (Y)")


if __name__ == "__main__":
    bench_once(M=128)
