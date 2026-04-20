"""Triton reduction kernels for conservation fixers.

Provides high-performance FP64-accumulator reductions for:
  - Weighted sphere quadrature: integrate nodal fields over the sphere
  - Spectral norm: Parseval-identity L2 norm in modal space
  - Kahan-compensated global summation

Kernels:
  - _sphere_reduce_kernel: Triton JIT 2D weighted reduction (GPU)
  - sphere_integrate_torch: pure-PyTorch reference
  - sphere_integrate: auto-dispatch
  - spectral_norm_torch / spectral_norm: modal L2 norm
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_HAS_TRITON = False

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _sphere_reduce_kernel(
        z_ptr,        # z[B, X, Y]  — nodal field
        w_ptr,        # w[Y]        — quadrature weights (lat only)
        out_ptr,      # out[B]      — integrated values
        B: tl.constexpr,
        X: tl.constexpr,   # n_longitude
        Y: tl.constexpr,   # n_latitude
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        """2D weighted reduction: out[b] = sum_{x,y} w[y] * z[b, x, y].

        Each program handles one batch element.
        """
        pid_b = tl.program_id(0)
        acc = tl.zeros([], dtype=tl.float64)

        for x_start in range(0, X, BLOCK_X):
            x_off = x_start + tl.arange(0, BLOCK_X)
            x_mask = x_off < X

            for y_start in range(0, Y, BLOCK_Y):
                y_off = y_start + tl.arange(0, BLOCK_Y)
                y_mask = y_off < Y

                # Load weights (broadcast along x)
                w_vals = tl.load(w_ptr + y_off, mask=y_mask, other=0.0)

                # Load z block: shape (BLOCK_X, BLOCK_Y)
                # z layout: [B, X, Y] — contiguous
                z_offset = pid_b * (X * Y) + x_off[:, None] * Y + y_off[None, :]
                mask_2d = x_mask[:, None] & y_mask[None, :]
                z_vals = tl.load(z_ptr + z_offset, mask=mask_2d, other=0.0)

                # FP64 accumulation
                z_f64 = z_vals.to(tl.float64)
                w_f64 = w_vals.to(tl.float64)
                weighted = z_f64 * w_f64[None, :]
                acc += tl.sum(weighted)

        tl.store(out_ptr + pid_b, acc)

    @triton.jit
    def _spectral_norm_kernel(
        x_ptr,        # x[B, M, L] — modal field
        out_ptr,      # out[B] — norms
        B: tl.constexpr,
        M: tl.constexpr,
        L: tl.constexpr,
        BLOCK_ML: tl.constexpr,
    ):
        """Spectral L2 norm: out[b] = sum_{m,l} x[b,m,l]^2."""
        pid_b = tl.program_id(0)
        acc = tl.zeros([], dtype=tl.float64)
        ML = M * L

        for ml_start in range(0, ML, BLOCK_ML):
            ml_off = ml_start + tl.arange(0, BLOCK_ML)
            mask = ml_off < ML
            vals = tl.load(x_ptr + pid_b * ML + ml_off, mask=mask, other=0.0)
            v64 = vals.to(tl.float64)
            acc += tl.sum(v64 * v64)

        tl.store(out_ptr + pid_b, acc)

except (ImportError, Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Pure-PyTorch reference implementations
# ═══════════════════════════════════════════════════════════════════════════


def sphere_integrate_torch(
    z: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Integrate nodal field over the sphere using quadrature weights.

    Mirrors RealSphericalHarmonics.integrate():
        result = sum_{x,y} w[y] * z[..., x, y]

    Args:
        z: nodal field, shape (..., n_lon, n_lat).
        weights: quadrature weights, shape (n_lat,). Should include
                 radius² and Fourier factor.

    Returns:
        Integrated scalar per batch element, shape (...), FP64.
    """
    w64 = weights.to(torch.float64)
    z64 = z.to(torch.float64)
    return torch.einsum("y,...xy->...", w64, z64)


def spectral_norm_torch(
    x: torch.Tensor,
) -> torch.Tensor:
    """Spectral L2 norm: sum(x**2) in FP64.

    Args:
        x: modal field, shape (..., M, L).

    Returns:
        Sum of squares, shape (...), FP64.
    """
    x64 = x.to(torch.float64)
    return (x64 ** 2).sum(dim=(-2, -1))


class KahanAccumulator:
    """Kahan-compensated FP64 summation for conservation diagnostics.

    Tracks running sum with compensation for floating-point rounding.
    """

    def __init__(self, shape: tuple, device: torch.device = torch.device("cpu")):
        self.sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.compensation = torch.zeros(shape, dtype=torch.float64, device=device)

    def add(self, x: torch.Tensor) -> None:
        """Add a value with Kahan compensation."""
        x64 = x.to(torch.float64).to(self.sum.device)
        y = x64 - self.compensation
        t = self.sum + y
        self.compensation = (t - self.sum) - y
        self.sum = t

    def value(self) -> torch.Tensor:
        """Return current accumulated value."""
        return self.sum

    def reset(self) -> None:
        """Reset accumulator to zero."""
        self.sum.zero_()
        self.compensation.zero_()


# ═══════════════════════════════════════════════════════════════════════════
# Triton wrappers
# ═══════════════════════════════════════════════════════════════════════════


def sphere_integrate_triton(
    z: torch.Tensor,
    weights: torch.Tensor,
    block_x: int = 64,
    block_y: int = 32,
) -> torch.Tensor:
    """Triton-accelerated sphere integration."""
    if not _HAS_TRITON or not z.is_cuda:
        return sphere_integrate_torch(z, weights)

    orig_shape = z.shape
    X, Y = orig_shape[-2], orig_shape[-1]
    B = z[..., 0, 0].numel()
    z_flat = z.reshape(B, X, Y).contiguous()
    w_c = weights.contiguous()

    out = torch.empty(B, dtype=torch.float64, device=z.device)

    _sphere_reduce_kernel[(B,)](
        z_flat, w_c, out,
        B=B, X=X, Y=Y,
        BLOCK_X=block_x, BLOCK_Y=block_y,
    )

    out_shape = list(orig_shape[:-2])
    return out.reshape(out_shape) if out_shape else out.squeeze()


def spectral_norm_triton(
    x: torch.Tensor,
    block_ml: int = 256,
) -> torch.Tensor:
    """Triton-accelerated spectral norm."""
    if not _HAS_TRITON or not x.is_cuda:
        return spectral_norm_torch(x)

    orig_shape = x.shape
    M, L = orig_shape[-2], orig_shape[-1]
    B = x[..., 0, 0].numel()
    x_flat = x.reshape(B, M * L).contiguous()

    out = torch.empty(B, dtype=torch.float64, device=x.device)

    _spectral_norm_kernel[(B,)](
        x_flat, out,
        B=B, M=M, L=L,
        BLOCK_ML=block_ml,
    )

    out_shape = list(orig_shape[:-2])
    return out.reshape(out_shape) if out_shape else out.squeeze()


# ═══════════════════════════════════════════════════════════════════════════
# Auto-dispatch API
# ═══════════════════════════════════════════════════════════════════════════


def sphere_integrate(
    z: torch.Tensor,
    weights: torch.Tensor,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Integrate nodal field over sphere with auto backend selection.

    Args:
        z: (..., n_lon, n_lat) nodal field.
        weights: (n_lat,) quadrature weights.
        backend: 'triton', 'torch', or None (auto).

    Returns:
        (...) integrated scalar field, FP64.
    """
    if backend == "triton" or (backend is None and _HAS_TRITON and z.is_cuda):
        return sphere_integrate_triton(z, weights)
    return sphere_integrate_torch(z, weights)


def spectral_norm(
    x: torch.Tensor,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Spectral L2 norm with auto backend selection.

    Args:
        x: (..., M, L) modal field.
        backend: 'triton', 'torch', or None (auto).

    Returns:
        (...) sum of squares, FP64.
    """
    if backend == "triton" or (backend is None and _HAS_TRITON and x.is_cuda):
        return spectral_norm_triton(x)
    return spectral_norm_torch(x)


def has_triton() -> bool:
    """Check if Triton backend is available."""
    return _HAS_TRITON
