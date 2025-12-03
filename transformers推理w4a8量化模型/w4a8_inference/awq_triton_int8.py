from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - 允许在无triton环境下导入文件
    triton = None
    tl = None

#================================================================
# 1. Per-token Quantization Kernel
#================================================================
@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,           # float tensor [M, K]
    output_ptr,      # int8 tensor [M, K]
    scales_ptr,      # float tensor [M]
    M: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for per-token symmetric INT8 quantization (fused).
    Each program operates on a single row (token).
    """
    pid_m = tl.program_id(axis=0)  # Current row

    # 1. Find max absolute value for the current row
    row_start_ptr = x_ptr + pid_m * stride_xm
    max_abs = 0.0
    for k_offset in range(0, K, BLOCK_K):
        offs = k_offset + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_vals = tl.load(row_start_ptr + offs * stride_xk, mask=mask, other=0.0)
        max_abs = tl.maximum(max_abs, tl.max(tl.abs(x_vals)))

    # 2. Calculate and store scale
    scale = max_abs / 127.0
    scale = tl.where(scale == 0, 1e-8, scale) # Avoid division by zero
    tl.store(scales_ptr + pid_m, scale)

    # 3. Quantize the entire row and store
    inv_scale = 1.0 / scale
    for k_offset in range(0, K, BLOCK_K):
        offs = k_offset + tl.arange(0, BLOCK_K)
        mask = offs < K
        
        # Load float values
        x_vals = tl.load(row_start_ptr + offs * stride_xk, mask=mask, other=0.0)
        
        # Quantize, round, clamp
        quantized_vals = x_vals * inv_scale
        # A robust way to round to nearest int, avoiding version-specific functions
        quantized_vals = tl.floor(quantized_vals + 0.5)
        quantized_vals = tl.clamp(quantized_vals, -127, 127)
        
        # Store as int8
        output_row_start_ptr = output_ptr + pid_m * stride_outm
        tl.store(output_row_start_ptr + offs * stride_outk, quantized_vals.to(tl.int8), mask=mask)


def per_token_quant_int8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for the Triton per-token INT8 quantization kernel.
    
    Args:
        x: Input float tensor of shape [M, K]
        
    Returns:
        A tuple of (quantized_x_int8, scales_float32)
    """
    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")
    
    M, K = x.shape
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device."

    output_int8 = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    grid = (M,)
    
    # Heuristic for block size
    BLOCK_K = 1024 if K > 4096 else 512

    _per_token_quant_int8_kernel[grid](
        x,
        output_int8,
        scales,
        M=M,
        K=K,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_outm=output_int8.stride(0),
        stride_outk=output_int8.stride(1),
        BLOCK_K=BLOCK_K,
    )
    
    return output_int8, scales

#================================================================
# 2. Int8 GEMM Kernel
#================================================================
@triton.jit
def _int8_gemm_kernel(
    a_ptr,  # int8 [M, K]
    b_ptr,  # int8 [K, N]
    c_ptr,  # float32 [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton W8A8 GEMM a_int8 * b_int8 -> c_float32

    - Accumulator is int32, converted to float32 before storing.
    """

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Use int32 accumulator for int8 dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Correctly mask for the K dimension
        k_mask = (offs_k[None, :] + k * BLOCK_K) < K

        # Load int8 input tiles, applying masks
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0)
        b = tl.load(b_ptrs, mask=k_mask.T & (offs_n[None, :] < N), other=0)

        # Perform int8 dot product, accumulating in int32
        acc = tl.dot(a, b, acc)

        # Advance pointers for the next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert accumulator to float32 before storing
    acc_fp32 = acc.to(tl.float32)

    # Store the result tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc_fp32, mask=c_mask)


def int8_gemm_triton(
    activations_int8: torch.Tensor,
    weight_int8: torch.Tensor,
) -> torch.Tensor:
    """使用 Triton 实现的 int8xint8 GEMM（无 scale）。

    Args:
        activations_int8: [M, K] int8，已经量化好的激活
        weight_int8: [K, N] int8，解包后的权重

    Returns:
        float32 [M, N]，是纯整数乘加的结果，尚未乘 scale。
    """

    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")

    assert activations_int8.dtype == torch.int8
    assert weight_int8.dtype == torch.int8
    assert activations_int8.is_cuda and weight_int8.is_cuda

    M, K = activations_int8.shape
    K2, N = weight_int8.shape
    if K != K2:
        raise ValueError(f"K mismatch: {K} vs {K2}")

    a = activations_int8.contiguous()
    b = weight_int8.contiguous()

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int8_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c

#================================================================
# 3. Fused Quantization + GEMM function
#================================================================
def awq_int8_gemm_triton(
    x_fp: torch.Tensor,
    qweight_int8: torch.Tensor,
    w_scales: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AWQ 风格的 W8A8-int8 运算：per-token 激活量化 + Triton GEMM。

    现在使用 Triton 进行激活量化。
    """

    if x_fp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"x_fp must be floating, got {x_fp.dtype}")

    orig_shape = x_fp.shape
    M = int(torch.prod(torch.tensor(orig_shape[:-1])).item())
    K = orig_shape[-1]

    x_2d = x_fp.reshape(M, K)

    # Use Triton for per-token symmetric int8 quantization
    x_q, act_scales = per_token_quant_int8_triton(x_2d)

    # Use Triton for GEMM
    c_fp32 = int8_gemm_triton(x_q, qweight_int8)

    return c_fp32, act_scales, w_scales.view(-1)
