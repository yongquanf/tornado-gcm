"""Hybrid sigma-pressure vertical coordinate system — PyTorch implementation.

Zone: Z0 (FP64) for all coordinate arithmetic — these are structural
constants and numerical foundations.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

from pytorch_src.core import sigma_coordinates


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
        a = torch.tensor(self.a_boundaries, dtype=surface_pressure.dtype,
                         device=surface_pressure.device)
        b = torch.tensor(self.b_boundaries, dtype=surface_pressure.dtype,
                         device=surface_pressure.device)
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
        p_thick = torch.tensor(
            self.pressure_thickness, dtype=surface_pressure.dtype,
            device=surface_pressure.device,
        )
        s_thick = torch.tensor(
            self.sigma_thickness, dtype=surface_pressure.dtype,
            device=surface_pressure.device,
        )
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
    a_thick = torch.tensor(
        coordinates.pressure_thickness, dtype=x.dtype, device=x.device,
    ).reshape(-1, 1, 1)
    b_thick = torch.tensor(
        coordinates.sigma_thickness, dtype=x.dtype, device=x.device,
    ).reshape(-1, 1, 1)
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
    a_thick = torch.tensor(
        coordinates.pressure_thickness, dtype=x.dtype, device=x.device,
    ).reshape(-1, 1, 1)
    b_thick = torch.tensor(
        coordinates.sigma_thickness, dtype=x.dtype, device=x.device,
    ).reshape(-1, 1, 1)
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
    inv_deta = torch.tensor(
        1.0 / coordinates.center_to_center, dtype=x.dtype, device=x.device,
    ).reshape(-1, 1, 1)
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
