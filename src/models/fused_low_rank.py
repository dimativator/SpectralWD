"""Inference-only Triton kernel for a factorized linear projection.

The kernel evaluates ``(x @ B.T) @ A.T + bias`` without materializing the
``x @ B.T`` intermediate in global memory.  Unsupported inputs fall back to
PyTorch's two-GEMM implementation so enabling the kernel does not change the
training path or CPU behavior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _persistent_fused_low_rank_kernel(
        x_ptr,
        b_ptr,
        a_ptr,
        bias_ptr,
        output_ptr,
        m_size,
        n_size,
        k_size,
        rank_size,
        stride_xm,
        stride_xk,
        stride_br,
        stride_bk,
        stride_an,
        stride_ar,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PADDED_RANK: tl.constexpr,
    ):
        """Compute the rank intermediate once, then consume all output tiles."""
        pid_m = tl.program_id(axis=0)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_r = tl.arange(0, PADDED_RANK)
        intermediate = tl.zeros((BLOCK_M, PADDED_RANK), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            offsets_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr
                + offsets_m[:, None] * stride_xm
                + offsets_k[None, :] * stride_xk,
                mask=(offsets_m[:, None] < m_size)
                & (offsets_k[None, :] < k_size),
                other=0.0,
            )
            b = tl.load(
                b_ptr
                + offsets_r[:, None] * stride_br
                + offsets_k[None, :] * stride_bk,
                mask=(offsets_r[:, None] < rank_size)
                & (offsets_k[None, :] < k_size),
                other=0.0,
            )
            if USE_BF16:
                x = x.to(tl.bfloat16)
                b = b.to(tl.bfloat16)
            else:
                x = x.to(tl.float16)
                b = b.to(tl.float16)
            intermediate += tl.dot(x, tl.trans(b))

        if USE_BF16:
            intermediate = intermediate.to(tl.bfloat16)
        else:
            intermediate = intermediate.to(tl.float16)

        for n_block in range(0, tl.cdiv(n_size, BLOCK_N)):
            offsets_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            a = tl.load(
                a_ptr
                + offsets_n[:, None] * stride_an
                + offsets_r[None, :] * stride_ar,
                mask=(offsets_n[:, None] < n_size)
                & (offsets_r[None, :] < rank_size),
                other=0.0,
            )
            if USE_BF16:
                a = a.to(tl.bfloat16)
            else:
                a = a.to(tl.float16)
            accumulator = tl.dot(intermediate, tl.trans(a))
            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + offsets_n,
                    mask=offsets_n < n_size,
                    other=0.0,
                )
                accumulator += bias[None, :]
            tl.store(
                output_ptr
                + offsets_m[:, None] * stride_om
                + offsets_n[None, :] * stride_on,
                accumulator,
                mask=(offsets_m[:, None] < m_size)
                & (offsets_n[None, :] < n_size),
            )

    @triton.jit
    def _fused_low_rank_kernel(
        x_ptr,
        b_ptr,
        a_ptr,
        bias_ptr,
        output_ptr,
        m_size,
        n_size,
        k_size,
        rank_size,
        stride_xm,
        stride_xk,
        stride_br,
        stride_bk,
        stride_an,
        stride_ar,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """Back-to-back GEMM tile.

        A program owns one output tile.  Its rank-space intermediate remains
        in registers between the two ``tl.dot`` operations.  The first GEMM is
        consequently recomputed for each output-column tile.  This prototype
        is intended to measure when avoiding HBM traffic and a launch offsets
        that recomputation; it is not assumed to win for every shape.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for rank_block in range(0, tl.cdiv(rank_size, BLOCK_R)):
            offsets_r = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
            intermediate = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

            for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
                offsets_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr
                    + offsets_m[:, None] * stride_xm
                    + offsets_k[None, :] * stride_xk,
                    mask=(offsets_m[:, None] < m_size)
                    & (offsets_k[None, :] < k_size),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + offsets_r[:, None] * stride_br
                    + offsets_k[None, :] * stride_bk,
                    mask=(offsets_r[:, None] < rank_size)
                    & (offsets_k[None, :] < k_size),
                    other=0.0,
                )
                if USE_BF16:
                    x = x.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)
                else:
                    x = x.to(tl.float16)
                    b = b.to(tl.float16)
                intermediate += tl.dot(x, tl.trans(b))

            a = tl.load(
                a_ptr
                + offsets_n[:, None] * stride_an
                + offsets_r[None, :] * stride_ar,
                mask=(offsets_n[:, None] < n_size)
                & (offsets_r[None, :] < rank_size),
                other=0.0,
            )
            if USE_BF16:
                intermediate = intermediate.to(tl.bfloat16)
                a = a.to(tl.bfloat16)
            else:
                intermediate = intermediate.to(tl.float16)
                a = a.to(tl.float16)
            accumulator += tl.dot(intermediate, tl.trans(a))

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offsets_n,
                mask=offsets_n < n_size,
                other=0.0,
            )
            accumulator += bias[None, :]

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_om
            + offsets_n[None, :] * stride_on,
            accumulator,
            mask=(offsets_m[:, None] < m_size)
            & (offsets_n[None, :] < n_size),
        )


def triton_low_rank_available() -> bool:
    """Return whether the optional Triton dependency can be imported."""
    return triton is not None


def _torch_low_rank_linear(
    x: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(F.linear(x, b_weight), a_weight, bias)


def _can_use_triton(
    x: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if triton is None or torch.is_grad_enabled():
        return False
    if not (x.is_cuda and b_weight.is_cuda and a_weight.is_cuda):
        return False
    if x.device != b_weight.device or x.device != a_weight.device:
        return False
    if bias is not None and (not bias.is_cuda or bias.device != x.device):
        return False
    if x.ndim < 2 or not x.is_contiguous():
        return False
    if not (b_weight.is_contiguous() and a_weight.is_contiguous()):
        return False
    if bias is not None and not bias.is_contiguous():
        return False
    if b_weight.ndim != 2 or a_weight.ndim != 2:
        return False
    rank, in_features = b_weight.shape
    out_features, a_rank = a_weight.shape
    if x.shape[-1] != in_features or rank != a_rank:
        return False
    if bias is not None and bias.numel() != out_features:
        return False

    compute_dtype = (
        torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
    )
    return compute_dtype in (torch.bfloat16, torch.float16)


def fused_low_rank_linear(
    x: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a factorized linear projection with an optional fused kernel.

    The custom kernel is inference-only.  Grad-enabled execution, CPU tensors,
    non-contiguous inputs, FP32 execution, or a missing Triton installation use
    the exact existing PyTorch expression as a safe fallback.
    """
    if not _can_use_triton(x, b_weight, a_weight, bias):
        return _torch_low_rank_linear(x, b_weight, a_weight, bias)

    input_2d = x.view(-1, x.shape[-1])
    m_size, k_size = input_2d.shape
    rank_size = b_weight.shape[0]
    n_size = a_weight.shape[0]
    compute_dtype = (
        torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
    )
    output = torch.empty(
        (m_size, n_size), device=x.device, dtype=compute_dtype
    )

    block_m = 16
    block_n = 32
    block_k = 32
    block_r = 32
    bias_ptr = bias if bias is not None else a_weight
    common_args = (
        input_2d,
        b_weight,
        a_weight,
        bias_ptr,
        output,
        m_size,
        n_size,
        k_size,
        rank_size,
        input_2d.stride(0),
        input_2d.stride(1),
        b_weight.stride(0),
        b_weight.stride(1),
        a_weight.stride(0),
        a_weight.stride(1),
        output.stride(0),
        output.stride(1),
    )
    if rank_size <= 1024:
        padded_rank = max(16, triton.next_power_of_2(rank_size))
        grid = (triton.cdiv(m_size, block_m),)
        _persistent_fused_low_rank_kernel[grid](
            *common_args,
            HAS_BIAS=bias is not None,
            USE_BF16=compute_dtype == torch.bfloat16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            PADDED_RANK=padded_rank,
            num_warps=8 if padded_rank >= 256 else 4,
            num_stages=1,
        )
    else:
        grid = (
            triton.cdiv(m_size, block_m),
            triton.cdiv(n_size, block_n),
        )
        _fused_low_rank_kernel[grid](
            *common_args,
            HAS_BIAS=bias is not None,
            USE_BF16=compute_dtype == torch.bfloat16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_R=block_r,
            num_warps=4,
            num_stages=3,
        )
    return output.view(*x.shape[:-1], n_size)
