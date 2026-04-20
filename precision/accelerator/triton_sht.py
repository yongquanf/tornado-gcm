# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Triton SHT Legendre summation kernels.

Provides high-performance Legendre summation for spherical harmonic
transforms, with FP64 basis × FP32/BF16 coefficients and configurable
accumulator precision.

Kernels:
  - _sht_legendre_kernel: Triton JIT kernel (GPU, sm_80+)
  - sht_legendre_torch: pure-PyTorch reference (any device)
  - sht_legendre: auto-dispatch (Triton if available, else PyTorch)

Einsum pattern ported:
  inverse: "mjl,...ml->...mj"  (P[m,j,l] × x[...,m,l] → out[...,m,j])
  forward: "mjl,...mj->...ml"  (P[m,j,l] × x[...,m,j] → out[...,m,l])
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Triton kernel (optional — requires triton + NVIDIA GPU)
# ═══════════════════════════════════════════════════════════════════════════

_HAS_TRITON = False

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _sht_legendre_fwd_kernel(
        basis_ptr,   # P[M, J, L]  — FP64 Legendre basis
        x_ptr,       # x[..., M, L]  — input coefficients
        out_ptr,     # out[..., M, J] — output
        M: tl.constexpr,
        J: tl.constexpr,
        L: tl.constexpr,
        stride_basis_m: tl.constexpr,
        stride_basis_j: tl.constexpr,
        stride_x_m: tl.constexpr,
        stride_out_m: tl.constexpr,
        BLOCK_L: tl.constexpr,
        USE_FP64_ACC: tl.constexpr,
    ):
        """Legendre inverse transform kernel.

        Computes out[batch, m, j] = sum_l basis[m, j, l] * x[batch, m, l]
        for one (batch, m, j) tile.
        """
        pid_batch_m = tl.program_id(0)
        pid_j = tl.program_id(1)

        batch_idx = pid_batch_m // M
        m_idx = pid_batch_m % M
        j_idx = pid_j

        # Accumulator
        if USE_FP64_ACC:
            acc = tl.zeros([], dtype=tl.float64)
        else:
            acc = tl.zeros([], dtype=tl.float32)

        for l_start in range(0, L, BLOCK_L):
            l_offsets = l_start + tl.arange(0, BLOCK_L)
            mask = l_offsets < L

            # Load basis[m, j, l] — always FP64
            basis_offset = (
                m_idx * stride_basis_m + j_idx * stride_basis_j + l_offsets
            )
            p_vals = tl.load(basis_ptr + basis_offset, mask=mask, other=0.0)

            # Load x[batch, m, l]
            x_offset = batch_idx * (M * L) + m_idx * stride_x_m + l_offsets
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Cast to accumulator precision
            if USE_FP64_ACC:
                x_vals = x_vals.to(tl.float64)
            else:
                p_vals = p_vals.to(tl.float32)

            acc += tl.sum(p_vals * x_vals)

        # Store result
        out_offset = batch_idx * (M * J) + m_idx * stride_out_m + j_idx
        tl.store(out_ptr + out_offset, acc.to(tl.float32))

    @triton.jit
    def _sht_legendre_bwd_kernel(
        basis_ptr,   # P[M, J, L] — FP64 Legendre basis
        x_ptr,       # x[..., M, J] — input nodal values
        out_ptr,     # out[..., M, L] — output coefficients
        M: tl.constexpr,
        J: tl.constexpr,
        L: tl.constexpr,
        stride_basis_m: tl.constexpr,
        stride_basis_j: tl.constexpr,
        stride_x_m: tl.constexpr,
        stride_out_m: tl.constexpr,
        BLOCK_J: tl.constexpr,
        USE_FP64_ACC: tl.constexpr,
    ):
        """Legendre forward transform kernel.

        Computes out[batch, m, l] = sum_j basis[m, j, l] * x[batch, m, j]
        """
        pid_batch_m = tl.program_id(0)
        pid_l = tl.program_id(1)

        batch_idx = pid_batch_m // M
        m_idx = pid_batch_m % M
        l_idx = pid_l

        if USE_FP64_ACC:
            acc = tl.zeros([], dtype=tl.float64)
        else:
            acc = tl.zeros([], dtype=tl.float32)

        for j_start in range(0, J, BLOCK_J):
            j_offsets = j_start + tl.arange(0, BLOCK_J)
            mask = j_offsets < J

            basis_offset = (
                m_idx * stride_basis_m + j_offsets * stride_basis_j + l_idx
            )
            p_vals = tl.load(basis_ptr + basis_offset, mask=mask, other=0.0)

            x_offset = batch_idx * (M * J) + m_idx * stride_x_m + j_offsets
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            if USE_FP64_ACC:
                x_vals = x_vals.to(tl.float64)
            else:
                p_vals = p_vals.to(tl.float32)

            acc += tl.sum(p_vals * x_vals)

        out_offset = batch_idx * (M * L) + m_idx * stride_out_m + l_idx
        tl.store(out_ptr + out_offset, acc.to(tl.float32))

except (ImportError, Exception):
    pass  # Triton not available


# ═══════════════════════════════════════════════════════════════════════════
# Pure PyTorch reference implementations (any device)
# ═══════════════════════════════════════════════════════════════════════════


def sht_legendre_inverse_torch(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64: bool = True,
) -> torch.Tensor:
    """Legendre inverse transform: out[...,m,j] = sum_l P[m,j,l] * x[...,m,l].

    Args:
        basis: Legendre polynomial matrix, shape (M, J, L), typically FP64.
        x: modal coefficients, shape (..., M, L).
        use_fp64: compute in FP64 for numerical stability.

    Returns:
        Nodal values, shape (..., M, J), FP32.
    """
    if use_fp64:
        b64 = basis.to(torch.float64)
        x64 = x.to(torch.float64)
        result = torch.einsum("mjl,...ml->...mj", b64, x64)
        return result.to(torch.float32)
    return torch.einsum("mjl,...ml->...mj", basis.to(x.dtype), x)


def sht_legendre_forward_torch(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64: bool = True,
) -> torch.Tensor:
    """Legendre forward transform: out[...,m,l] = sum_j P[m,j,l] * x[...,m,j].

    Args:
        basis: Legendre polynomial matrix, shape (M, J, L), typically FP64.
        x: nodal values, shape (..., M, J).
        use_fp64: compute in FP64 for numerical stability.

    Returns:
        Modal coefficients, shape (..., M, L), FP32.
    """
    if use_fp64:
        b64 = basis.to(torch.float64)
        x64 = x.to(torch.float64)
        result = torch.einsum("mjl,...mj->...ml", b64, x64)
        return result.to(torch.float32)
    return torch.einsum("mjl,...mj->...ml", basis.to(x.dtype), x)


# ═══════════════════════════════════════════════════════════════════════════
# Triton wrappers (dispatch to kernels)
# ═══════════════════════════════════════════════════════════════════════════


def sht_legendre_inverse_triton(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64_acc: bool = True,
    block_l: int = 64,
) -> torch.Tensor:
    """Triton-accelerated Legendre inverse transform.

    Falls back to PyTorch if Triton is not available.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return sht_legendre_inverse_torch(basis, x, use_fp64=use_fp64_acc)

    # Flatten batch dims
    orig_shape = x.shape
    M, L = orig_shape[-2], orig_shape[-1]
    J = basis.shape[1]
    batch_size = x[..., 0, 0].numel()
    x_flat = x.reshape(batch_size, M, L).contiguous()
    basis_c = basis.contiguous()

    out = torch.empty(batch_size, M, J, dtype=torch.float32, device=x.device)

    grid = (batch_size * M, J)
    _sht_legendre_fwd_kernel[grid](
        basis_c, x_flat, out,
        M=M, J=J, L=L,
        stride_basis_m=basis_c.stride(0),
        stride_basis_j=basis_c.stride(1),
        stride_x_m=x_flat.stride(1),
        stride_out_m=out.stride(1),
        BLOCK_L=block_l,
        USE_FP64_ACC=use_fp64_acc,
    )

    out_shape = list(orig_shape[:-1]) + [J]
    out_shape[-2] = M
    return out.reshape(out_shape)


def sht_legendre_forward_triton(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64_acc: bool = True,
    block_j: int = 32,
) -> torch.Tensor:
    """Triton-accelerated Legendre forward transform.

    Falls back to PyTorch if Triton is not available.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return sht_legendre_forward_torch(basis, x, use_fp64=use_fp64_acc)

    orig_shape = x.shape
    M, J = orig_shape[-2], orig_shape[-1]
    L = basis.shape[2]
    batch_size = x[..., 0, 0].numel()
    x_flat = x.reshape(batch_size, M, J).contiguous()
    basis_c = basis.contiguous()

    out = torch.empty(batch_size, M, L, dtype=torch.float32, device=x.device)

    grid = (batch_size * M, L)
    _sht_legendre_bwd_kernel[grid](
        basis_c, x_flat, out,
        M=M, J=J, L=L,
        stride_basis_m=basis_c.stride(0),
        stride_basis_j=basis_c.stride(1),
        stride_x_m=x_flat.stride(1),
        stride_out_m=out.stride(1),
        BLOCK_J=block_j,
        USE_FP64_ACC=use_fp64_acc,
    )

    out_shape = list(orig_shape[:-1]) + [L]
    out_shape[-2] = M
    return out.reshape(out_shape)


# ═══════════════════════════════════════════════════════════════════════════
# Auto-dispatch API
# ═══════════════════════════════════════════════════════════════════════════


def sht_legendre_inverse(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64: bool = True,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Legendre inverse transform with auto backend selection.

    Args:
        basis: (M, J, L) FP64 Legendre basis.
        x: (..., M, L) modal coefficients.
        use_fp64: use FP64 accumulator.
        backend: 'triton', 'torch', or None (auto).

    Returns:
        (..., M, J) nodal values in FP32.
    """
    if backend == "triton" or (backend is None and _HAS_TRITON and x.is_cuda):
        return sht_legendre_inverse_triton(basis, x, use_fp64_acc=use_fp64)
    return sht_legendre_inverse_torch(basis, x, use_fp64=use_fp64)


def sht_legendre_forward(
    basis: torch.Tensor,
    x: torch.Tensor,
    use_fp64: bool = True,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Legendre forward transform with auto backend selection.

    Args:
        basis: (M, J, L) FP64 Legendre basis.
        x: (..., M, J) nodal values.
        use_fp64: use FP64 accumulator.
        backend: 'triton', 'torch', or None (auto).

    Returns:
        (..., M, L) modal coefficients in FP32.
    """
    if backend == "triton" or (backend is None and _HAS_TRITON and x.is_cuda):
        return sht_legendre_forward_triton(basis, x, use_fp64_acc=use_fp64)
    return sht_legendre_forward_torch(basis, x, use_fp64=use_fp64)


def has_triton() -> bool:
    """Check if Triton backend is available."""
    return _HAS_TRITON
