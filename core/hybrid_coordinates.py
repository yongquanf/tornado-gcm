# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Hybrid sigma-pressure vertical coordinate system — PyTorch implementation.

Zone: Z0 (FP64) for all coordinate arithmetic — these are structural
constants and numerical foundations.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.core import sigma_coordinates


class _HybridBuffers(nn.Module):
    """Internal nn.Module holding HybridCoordinates' static tensors as buffers."""

    def __init__(self):
        super().__init__()

    def _init(self, _a_bnd, _b_bnd, _p_thick, _s_thick, _c2c):
        self.register_buffer("_a_boundaries_torch", torch.from_numpy(_a_bnd.copy()).to(torch.float64))
        self.register_buffer("_b_boundaries_torch", torch.from_numpy(_b_bnd.copy()).to(torch.float64))
        self.register_buffer("_pressure_thickness_torch", torch.from_numpy(_p_thick.copy()).to(torch.float64))
        self.register_buffer("_sigma_thickness_torch", torch.from_numpy(_s_thick.copy()).to(torch.float64))
        self.register_buffer("_center_to_center_torch", torch.from_numpy(_c2c.copy()).to(torch.float64))


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
    """Hybrid sigma-pressure vertical coordinate system.

    Pressure at a level: P = a + b * P_surface.

    Attributes:
        a_boundaries: pressure offset coefficients (top → surface).
        b_boundaries: sigma slope coefficients (top → surface).
    """

    a_boundaries: np.ndarray
    b_boundaries: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "a_boundaries", np.asarray(self.a_boundaries))
        object.__setattr__(self, "b_boundaries", np.asarray(self.b_boundaries))
        if len(self.a_boundaries) != len(self.b_boundaries):
            raise ValueError(
                "a_boundaries and b_boundaries must have the same length; "
                f"got {len(self.a_boundaries)} and {len(self.b_boundaries)}."
            )
        _p_thick = np.diff(self.a_boundaries)
        _s_thick = np.diff(self.b_boundaries)
        _b_centers = (self.b_boundaries[1:] + self.b_boundaries[:-1]) / 2
        _c2c = np.diff(_b_centers)
        _buf = _HybridBuffers()
        _buf._init(self.a_boundaries, self.b_boundaries, _p_thick, _s_thick, _c2c)
        object.__setattr__(self, "_buf", _buf)

    def to(self, device: torch.device | str) -> None:
        """Move all static tensors to *device* (in-place)."""
        self._buf.to(device=torch.device(device))

    @classmethod
    def from_coefficients(
        cls,
        a_coeffs: list[float] | np.ndarray,
        b_coeffs: list[float] | np.ndarray,
        p0: float = 1000.0,
    ) -> HybridCoordinates:
        """Create from dimensionless A, B coefficients and reference pressure.

        P(k) = A(k) * p0 + B(k) * Ps
        """
        a_boundaries = np.array(a_coeffs) * p0
        return cls(a_boundaries=a_boundaries, b_boundaries=np.array(b_coeffs))

    @classmethod
    def analytic_levels(
        cls,
        n_levels: int,
        p_top: float = 0.0,
        p0: float = 1000.0,
        sigma_exponent: float = 3.0,
        stretch_exponent: float = 2.0,
    ) -> HybridCoordinates:
        """Generate analytically smooth hybrid coordinates via power-law."""
        k = np.linspace(0, 1, n_levels + 1)
        eta = k**stretch_exponent
        p_profile = p_top + eta * (p0 - p_top)
        b_boundaries = eta**sigma_exponent
        a_boundaries = (p_profile / p0) - b_boundaries
        a_boundaries[0] = p_top / p0
        b_boundaries[0] = 0.0
        a_boundaries[-1] = 0.0
        b_boundaries[-1] = 1.0
        return cls.from_coefficients(a_boundaries, b_boundaries, p0)

    @classmethod
    def from_sigma_levels(
        cls, sigma_levels: sigma_coordinates.SigmaCoordinates
    ) -> HybridCoordinates:
        """Create hybrid coordinates equivalent to sigma coordinates."""
        b_boundaries = np.array(sigma_levels.boundaries)
        a_boundaries = np.zeros_like(b_boundaries)
        return cls(a_boundaries=a_boundaries, b_boundaries=b_boundaries)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def layers(self) -> int:
        return len(self.a_boundaries) - 1

    @property
    def pressure_thickness(self) -> np.ndarray:
        return np.diff(self.a_boundaries)

    @property
    def sigma_thickness(self) -> np.ndarray:
        return np.diff(self.b_boundaries)

    @property
    def a_centers(self) -> np.ndarray:
        return (self.a_boundaries[1:] + self.a_boundaries[:-1]) / 2

    @property
    def b_centers(self) -> np.ndarray:
        return (self.b_boundaries[1:] + self.b_boundaries[:-1]) / 2

    @property
    def center_to_center(self) -> np.ndarray:
        return np.diff(self.b_centers)

    def get_eta(self, p_surface: float) -> np.ndarray:
        return self.a_centers / p_surface + self.b_centers

    def pressure_boundaries(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        """Boundaries of each layer in pressure units."""
        a = self._buf._a_boundaries_torch.to(dtype=surface_pressure.dtype)
        b = self._buf._b_boundaries_torch.to(dtype=surface_pressure.dtype)
        # a: (layers+1,), surface_pressure: (...,)
        ndim = surface_pressure.dim()
        for _ in range(ndim):
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        return a + b * surface_pressure.unsqueeze(0)

    def pressure_centers(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        """Centers of each layer in pressure units."""
        boundaries = self.pressure_boundaries(surface_pressure)
        return (boundaries[1:] + boundaries[:-1]) / 2

    def layer_thickness(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        """Thickness of each layer in pressure units."""
        p_thick = self._buf._pressure_thickness_torch.to(dtype=surface_pressure.dtype)
        s_thick = self._buf._sigma_thickness_torch.to(dtype=surface_pressure.dtype)
        ndim = surface_pressure.dim()
        for _ in range(ndim):
            p_thick = p_thick.unsqueeze(-1)
            s_thick = s_thick.unsqueeze(-1)
        return p_thick + s_thick * surface_pressure.unsqueeze(0)

    def get_sigma_boundaries(self, surface_pressure: float) -> np.ndarray:
        return self.a_boundaries / surface_pressure + self.b_boundaries

    def get_sigma_centers(self, surface_pressure: float) -> np.ndarray:
        boundaries = self.get_sigma_boundaries(surface_pressure)
        return (boundaries[1:] + boundaries[:-1]) / 2

    def __hash__(self):
        return hash(
            (tuple(self.a_boundaries.tolist()),
             tuple(self.b_boundaries.tolist()))
        )

    def __eq__(self, other):
        return (
            isinstance(other, HybridCoordinates)
            and np.array_equal(self.a_boundaries, other.a_boundaries)
            and np.array_equal(self.b_boundaries, other.b_boundaries)
        )


# Register HybridCoordinates as a pytree node so Dynamo traces its tensor
# children with correct FakeTensor device propagation.
def __hybrid_flatten(hc: HybridCoordinates):
    children = (hc._buf,)
    context = {
        "a_boundaries": hc.a_boundaries,
        "b_boundaries": hc.b_boundaries,
    }
    return children, context


def __hybrid_unflatten(children, context):
    (buf,) = children
    hc = HybridCoordinates(
        a_boundaries=context["a_boundaries"],
        b_boundaries=context["b_boundaries"],
    )
    object.__setattr__(hc, "_buf", buf)
    return hc


from torch.utils._pytree import register_pytree_node as _reg_pytree_hybrid
_reg_pytree_hybrid(HybridCoordinates, __hybrid_flatten, __hybrid_unflatten)


# ═══════════════════════════════════════════════════════════════════════════
# Vertical calculus on hybrid coordinates
# ═══════════════════════════════════════════════════════════════════════════

def cumulative_integral_over_pressure(
    x: torch.Tensor,
    surface_pressure: torch.Tensor,
    coordinates: HybridCoordinates,
    axis: int = -3,
    downward: bool = True,
) -> torch.Tensor:
    """Cumulative integral of x w.r.t. pressure."""
    a_thick = coordinates._buf._pressure_thickness_torch.to(device=x
        .device, dtype=x
        .dtype).reshape(-1, 1, 1)
    b_thick = coordinates._buf._sigma_thickness_torch.to(device=x
        .device, dtype=x
        .dtype).reshape(-1, 1, 1)
    dp = a_thick + b_thick * surface_pressure
    xdp = x * dp
    if downward:
        return torch.cumsum(xdp, dim=axis)
    else:
        return torch.flip(torch.cumsum(torch.flip(xdp, [axis]), dim=axis), [axis])


def integral_over_pressure(
    x: torch.Tensor,
    surface_pressure: torch.Tensor,
    coordinates: HybridCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> torch.Tensor:
    """Definite integral of x over pressure."""
    a_thick = coordinates._buf._pressure_thickness_torch.to(device=x
        .device, dtype=x
        .dtype).reshape(-1, 1, 1)
    b_thick = coordinates._buf._sigma_thickness_torch.to(device=x
        .device, dtype=x
        .dtype).reshape(-1, 1, 1)
    x_da = (x * a_thick).sum(dim=axis, keepdim=keepdims)
    x_db = (x * b_thick).sum(dim=axis, keepdim=keepdims)
    return x_da + surface_pressure * x_db


def centered_difference(
    x: torch.Tensor,
    coordinates: HybridCoordinates,
    axis: int = -3,
) -> torch.Tensor:
    """∂x/∂η at layer boundaries via centered differences."""
    dx = torch.diff(x, dim=axis)
    inv_deta = 1.0 / coordinates._buf._center_to_center_torch.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
    return dx * inv_deta


def centered_vertical_advection(
    w: torch.Tensor,
    x: torch.Tensor,
    coordinates: HybridCoordinates,
    axis: int = -3,
) -> torch.Tensor:
    """Vertical advection using 2nd-order finite differences."""
    slc_shape = list(w.shape)
    slc_shape[axis] = 1
    zero = torch.zeros(slc_shape, dtype=w.dtype, device=w.device)
    w = torch.cat([zero, w, zero], dim=axis)
    x_diff = centered_difference(x, coordinates, axis)
    zero_x = torch.zeros_like(x_diff.narrow(axis, 0, 1))
    x_diff = torch.cat([zero_x, x_diff, zero_x], dim=axis)
    w_times_xd = w * x_diff
    return -0.5 * (
        w_times_xd.narrow(axis, 1, w_times_xd.shape[axis] - 1)
        + w_times_xd.narrow(axis, 0, w_times_xd.shape[axis] - 1)
    )
