"""Real Fourier basis evaluation and derivative coefficients.

Z0 (Numerical Foundation): Basis matrices computed in float64.
Runtime derivatives computed in PyTorch.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Basis construction (pure NumPy, float64)
# ═══════════════════════════════════════════════════════════════════════════

def real_basis(wavenumbers: int, nodes: int) -> np.ndarray:
    """Real-valued Fourier basis matrix.

    Args:
        wavenumbers: Number of wavenumbers.
        nodes: Number of equally spaced nodes in [0, 2π). Must be >= wavenumbers.

    Returns:
        Matrix F of shape (nodes, 2*wavenumbers - 1) such that
            F[i, 0] = 1 / √(2π)
            F[i, 2j-1] = cos(j x_i) / √π   for 1 ≤ j < wavenumbers
            F[i, 2j]   = sin(j x_i) / √π   for 1 ≤ j < wavenumbers
        with x_i = i * 2π / nodes.
    """
    if nodes < wavenumbers:
        raise ValueError(
            f"`real_basis` requires nodes >= wavenumbers; "
            f"got nodes = {nodes} and wavenumbers = {wavenumbers}."
        )
    dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)
    cos = np.real(dft[:, 1:])
    sin = -np.imag(dft[:, 1:])
    f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1::2] = cos
    f[:, 2::2] = sin
    return f


def real_basis_with_zero_imag(wavenumbers: int, nodes: int) -> np.ndarray:
    """Real basis with an explicit zero imaginary part column."""
    if nodes < wavenumbers:
        raise ValueError(
            f"`real_basis` requires nodes >= wavenumbers; "
            f"got nodes = {nodes} and wavenumbers = {wavenumbers}."
        )
    dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)
    cos = np.real(dft[:, 1:])
    sin = -np.imag(dft[:, 1:])
    f = np.empty(shape=[nodes, 2 * wavenumbers], dtype=np.float64)
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1] = 0
    f[:, 2::2] = cos
    f[:, 3::2] = sin
    return f


# ═══════════════════════════════════════════════════════════════════════════
# Derivative operators (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

def _shift(u: torch.Tensor, offset: int, axis: int) -> torch.Tensor:
    """Circular shift tensor along axis by offset positions."""
    n = u.shape[axis]
    idx = torch.arange(n, device=u.device)
    idx = (idx - offset) % n
    return u.index_select(axis if axis >= 0 else u.ndim + axis, idx)


def real_basis_derivative(
    u: torch.Tensor, /, axis: int = -1
) -> torch.Tensor:
    """Derivative of a signal in the real Fourier basis.

    Args:
        u: Signal to differentiate, in the real Fourier basis.
            Shape along ``axis`` must be odd (2*wavenumbers - 1).
        axis: Axis along which to differentiate (must be negative).

    Returns:
        Derivative of u along axis:
            u_x[..., 0]      = 0
            u_x[..., 2j-1]   =  j * u[..., 2j]
            u_x[..., 2j]     = -j * u[..., 2j-1]
    """
    if u.shape[axis] % 2 != 1:
        raise ValueError(f"u.shape along axis={axis} must be odd, got {u.shape[axis]}")
    if axis >= 0:
        raise ValueError("axis must be negative")

    ndim_after = -1 - axis
    shape = (-1,) + (1,) * ndim_after
    i = torch.arange(u.shape[axis], device=u.device).reshape(shape)
    j = (i + 1) // 2
    u_down = _shift(u, -1, axis)
    u_up = _shift(u, +1, axis)
    return j * torch.where(i % 2 == 1, u_down, -u_up)


def real_basis_derivative_with_zero_imag(
    u: torch.Tensor, axis: int = -1, frequency_offset: int = 0
) -> torch.Tensor:
    """Derivative along a real basis with zero imaginary part."""
    if u.shape[axis] % 2:
        raise ValueError(f"u.shape along axis={axis} must be even, got {u.shape[axis]}")
    if axis >= 0:
        raise ValueError("axis must be negative")

    ndim_after = -1 - axis
    shape = (-1,) + (1,) * ndim_after
    i = torch.arange(u.shape[axis], device=u.device).reshape(shape)
    j = frequency_offset + i // 2
    u_down = _shift(u, -1, axis)
    u_up = _shift(u, +1, axis)
    return j * torch.where((i + 1) % 2 == 1, u_down, -u_up)


# ═══════════════════════════════════════════════════════════════════════════
# Quadrature
# ═══════════════════════════════════════════════════════════════════════════

def quadrature_nodes(nodes: int) -> tuple[np.ndarray, float]:
    """Trapezoidal quadrature nodes and weight for Fourier integration.

    Returns:
        (nodes_array, weight) where nodes_array has shape (nodes,) in [0, 2π)
        and weight is a scalar 2π/nodes.
    """
    xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    weight = 2 * np.pi / nodes
    return xs, weight
