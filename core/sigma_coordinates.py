# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Sigma coordinate system based on normalized pressure.

Z0 (Numerical Foundation): Coordinate arithmetic uses float64 (_with_f64_math).
See https://en.wikipedia.org/wiki/Sigma_coordinate_system
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _with_f64_math(f):
    """Decorator: execute f with float64 inputs, cast result back."""
    def wrapper(x):
        return f(x.astype(np.float64)).astype(x.dtype)
    return wrapper


def _slice_shape_along_axis(
    x: torch.Tensor, axis: int, slice_width: int = 1
) -> list[int]:
    shape = list(x.shape)
    shape[axis] = slice_width
    return shape


# ═══════════════════════════════════════════════════════════════════════════
# SigmaCoordinates
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
    """Discrete vertical sigma coordinate system.

    Layers are indexed from the top of the atmosphere (σ=0) to the surface (σ=1).

    Attributes:
        boundaries: Sigma values at layer boundaries. Increasing array from 0 to 1.
            For n layers, has length n+1.
    """

    boundaries: np.ndarray

    def __init__(self, boundaries):
        boundaries = np.asarray(boundaries, dtype=np.float64)
        if not (np.isclose(boundaries[0], 0) and np.isclose(boundaries[-1], 1)):
            raise ValueError(
                f"Expected boundaries[0]=0, boundaries[-1]=1, "
                f"got boundaries={boundaries}"
            )
        if not all(np.diff(boundaries) > 0):
            raise ValueError(
                f"Expected boundaries to be monotonically increasing, "
                f"got boundaries={boundaries}"
            )
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def internal_boundaries(self) -> np.ndarray:
        return self.boundaries[1:-1]

    @property
    def centers(self) -> np.ndarray:
        return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

    @property
    def layer_thickness(self) -> np.ndarray:
        return _with_f64_math(np.diff)(self.boundaries)

    @property
    def center_to_center(self) -> np.ndarray:
        return _with_f64_math(np.diff)(self.centers)

    @property
    def layers(self) -> int:
        return len(self.boundaries) - 1

    @classmethod
    def equidistant(cls, layers: int, dtype=np.float32) -> SigmaCoordinates:
        boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
        return cls(boundaries)

    @classmethod
    def from_centers(cls, centers) -> SigmaCoordinates:
        """Create SigmaCoordinates from layer centers."""
        def centers_to_boundaries(centers):
            layers = len(centers)
            bounds_to_centers = 0.5 * (
                np.eye(layers) + np.eye(layers, k=-1)
            )
            unpadded_bounds = np.linalg.solve(bounds_to_centers, centers)
            return np.pad(unpadded_bounds, [(1, 0)])

        boundaries = _with_f64_math(centers_to_boundaries)(
            np.asarray(centers)
        )
        return cls(boundaries)

    def asdict(self):
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))

    def __eq__(self, other):
        return isinstance(other, SigmaCoordinates) and np.array_equal(
            self.centers, other.centers
        )


# ═══════════════════════════════════════════════════════════════════════════
# Vertical calculus operations (PyTorch tensors)
# ═══════════════════════════════════════════════════════════════════════════

def _diff(x: torch.Tensor, axis: int = -3) -> torch.Tensor:
    """Compute differences along axis: x[i+1] - x[i]."""
    return x.narrow(axis if axis >= 0 else x.ndim + axis, 1, x.shape[axis] - 1) - \
           x.narrow(axis if axis >= 0 else x.ndim + axis, 0, x.shape[axis] - 1)


def _dot_cumsum_prefix_like_jax(xds: torch.Tensor, axis: int) -> torch.Tensor:
    """Prefix sum along ``axis`` matching dinosaur ``jax_numpy_utils._single_device_dot_cumsum`` (forward).

    JAX accumulates ``x * dσ`` with an upper-triangular matmul cumsum
    (``jax_numpy_utils._single_device_dot_cumsum``, not ``jnp.cumsum``). For equidistant σ and
    typical dycore sizes this differs slightly from ``torch.cumsum`` and biases
    ``σ̇`` / vertical advection in PT–JAX lockstep probes.

    The matmul prefix is evaluated in **float64** then cast back to ``xds.dtype`` so PT
    lockstep probes sit closer to JAX's mixed-precision einsum (``bfloat16``, ``highest``)
    than a pure float32 ``@``.
    """
    dtype_orig = xds.dtype
    pos = axis if axis >= 0 else xds.ndim + axis
    n = xds.shape[pos]
    if n <= 1:
        return xds
    x64 = xds.to(torch.float64)
    u = torch.triu(torch.ones((n, n), dtype=torch.float64, device=xds.device))
    x_last = x64.movedim(pos, -1)
    lead = x_last.shape[:-1]
    flat = x_last.reshape(-1, n)
    out64 = flat @ u
    out_last = out64.reshape(*lead, n).movedim(-1, pos).to(dtype_orig)
    return out_last


def centered_difference(
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
) -> torch.Tensor:
    """Derivative of x with respect to sigma (centered finite difference).

    Returns values at internal_boundaries (one fewer than layers).
    """
    if coordinates.layers != x.shape[axis]:
        raise ValueError(
            f"x.shape[axis] must equal coordinates.layers; "
            f"got {x.shape[axis]} and {coordinates.layers}."
        )
    dx = _diff(x, axis)
    inv_dsigma = torch.tensor(
        1 / coordinates.center_to_center,
        dtype=x.dtype, device=x.device,
    )
    # Broadcast inv_dsigma along the vertical axis
    shape = [1] * x.ndim
    pos = axis if axis >= 0 else x.ndim + axis
    shape[pos] = -1
    inv_dsigma = inv_dsigma.reshape(shape)
    return dx * inv_dsigma


def cumulative_sigma_integral(
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
) -> torch.Tensor:
    """Cumulative integral of x with respect to sigma using midpoint rule."""
    if coordinates.layers != x.shape[axis]:
        raise ValueError(
            f"x.shape[axis] must equal coordinates.layers; "
            f"got {x.shape[axis]} and {coordinates.layers}."
        )
    dsigma = torch.tensor(
        coordinates.layer_thickness,
        dtype=x.dtype, device=x.device,
    )
    shape = [1] * x.ndim
    pos = axis if axis >= 0 else x.ndim + axis
    shape[pos] = -1
    dsigma = dsigma.reshape(shape)
    xds = x * dsigma
    if downward:
        return _dot_cumsum_prefix_like_jax(xds, pos)
    return torch.flip(
        _dot_cumsum_prefix_like_jax(torch.flip(xds, dims=[pos]), pos),
        dims=[pos],
    )


# Lockstep / regression probes read this to confirm PT uses the JAX-aligned prefix
# accumulation (matmul upper-triangular), not plain ``torch.cumsum``.
CUMULATIVE_SIGMA_INTEGRAL_BACKEND = "dot_matmul_prefix_like_jax_f64acc"


def sigma_integral(
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> torch.Tensor:
    """Full integral of x with respect to sigma."""
    if coordinates.layers != x.shape[axis]:
        raise ValueError(
            f"x.shape[axis] must equal coordinates.layers; "
            f"got {x.shape[axis]} and {coordinates.layers}."
        )
    dsigma = torch.tensor(
        coordinates.layer_thickness,
        dtype=x.dtype, device=x.device,
    )
    shape = [1] * x.ndim
    pos = axis if axis >= 0 else x.ndim + axis
    shape[pos] = -1
    dsigma = dsigma.reshape(shape)
    xds = x * dsigma
    return xds.sum(dim=pos, keepdim=keepdims)


def cumulative_log_sigma_integral(
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
) -> torch.Tensor:
    """Cumulative integral of x with respect to log(sigma) using trapezoid rule."""
    if coordinates.layers != x.shape[axis]:
        raise ValueError(
            f"x.shape[axis] must equal coordinates.layers; "
            f"got {x.shape[axis]} and {coordinates.layers}."
        )
    pos = axis if axis >= 0 else x.ndim + axis
    x_last = x.narrow(pos, x.shape[pos] - 1, 1)
    x_interp = (x.narrow(pos, 1, x.shape[pos] - 1) +
                x.narrow(pos, 0, x.shape[pos] - 1)) / 2
    integrand = torch.cat([x_interp, x_last], dim=pos)

    log_sigma = np.log(coordinates.centers)
    dlog_sigma = np.diff(log_sigma, append=0)
    dlog_sigma_t = torch.tensor(dlog_sigma, dtype=x.dtype, device=x.device)
    shape = [1] * x.ndim
    shape[pos] = -1
    dlog_sigma_t = dlog_sigma_t.reshape(shape)
    xds = integrand * dlog_sigma_t

    if downward:
        return torch.cumsum(xds, dim=pos)
    else:
        return torch.flip(
            torch.cumsum(torch.flip(xds, dims=[pos]), dim=pos),
            dims=[pos],
        )


def centered_vertical_advection(
    w: torch.Tensor,
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    w_boundary_values: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    dx_dsigma_boundary_values: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Vertical advection using 2nd order centered finite differences.

    Computes -(w * ∂x/∂σ)[n] at layer centers.
    """
    pos = axis if axis >= 0 else x.ndim + axis

    if w_boundary_values is None:
        w_slc_shape = _slice_shape_along_axis(w, pos)
        w_boundary_values = (
            torch.zeros(w_slc_shape, dtype=w.dtype, device=w.device),
            torch.zeros(w_slc_shape, dtype=w.dtype, device=w.device),
        )
    if dx_dsigma_boundary_values is None:
        x_slc_shape = _slice_shape_along_axis(x, pos)
        dx_dsigma_boundary_values = (
            torch.zeros(x_slc_shape, dtype=x.dtype, device=x.device),
            torch.zeros(x_slc_shape, dtype=x.dtype, device=x.device),
        )

    w_top, w_bot = w_boundary_values
    w = torch.cat([w_top, w, w_bot], dim=pos)

    x_diff = centered_difference(x, coordinates, axis)
    xd_top, xd_bot = dx_dsigma_boundary_values
    x_diff = torch.cat([xd_top, x_diff, xd_bot], dim=pos)

    w_times_xd = w * x_diff
    return -0.5 * (
        w_times_xd.narrow(pos, 1, w_times_xd.shape[pos] - 1)
        + w_times_xd.narrow(pos, 0, w_times_xd.shape[pos] - 1)
    )


def upwind_vertical_advection(
    w: torch.Tensor,
    x: torch.Tensor,
    coordinates: SigmaCoordinates,
    axis: int = -3,
) -> torch.Tensor:
    """Vertical advection using 1st order upwinding."""
    pos = axis if axis >= 0 else x.ndim + axis

    w_slc_shape = _slice_shape_along_axis(w, pos)
    zero_w = torch.zeros(w_slc_shape, dtype=w.dtype, device=w.device)

    x_slc_shape = _slice_shape_along_axis(x, pos)
    zero_x = torch.zeros(x_slc_shape, dtype=x.dtype, device=x.device)

    x_diff = centered_difference(x, coordinates, axis)

    w_up = torch.cat([zero_w, w], dim=pos)
    w_down = torch.cat([w, zero_w], dim=pos)

    x_diff_up = torch.cat([zero_x, x_diff], dim=pos)
    x_diff_down = torch.cat([x_diff, zero_x], dim=pos)

    return -(
        torch.clamp(w_up, min=0) * x_diff_up
        + torch.clamp(w_down, max=0) * x_diff_down
    )
