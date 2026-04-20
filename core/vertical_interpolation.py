# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Vertical interpolation between sigma and pressure levels — PyTorch.

Provides linear interpolation with constant extrapolation plus conservative
regridding between hybrid and sigma coordinates.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import torch

from tornado_gcm.core import hybrid_coordinates as hybrid_mod
from tornado_gcm.core import sigma_coordinates


# ═══════════════════════════════════════════════════════════════════════════
# PressureCoordinates
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class PressureCoordinates:
    """Vertical coordinates with pressure levels.

    Attributes:
        centers: monotonically increasing pressure levels (top → surface).
    """

    centers: np.ndarray

    def __init__(self, centers):
        object.__setattr__(self, "centers", np.asarray(centers))
        if not all(np.diff(self.centers) > 0):
            raise ValueError(
                f"centers must be monotonically increasing, got {self.centers}"
            )

    @property
    def layers(self) -> int:
        return len(self.centers)

    def asdict(self) -> Dict[str, Any]:
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))

    def __eq__(self, other):
        return isinstance(other, PressureCoordinates) and np.array_equal(
            self.centers, other.centers
        )


# ═══════════════════════════════════════════════════════════════════════════
# Interpolation primitives
# ═══════════════════════════════════════════════════════════════════════════

def interp_1d(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
) -> torch.Tensor:
    """1D linear interpolation with constant extrapolation (like np.interp).

    Args:
        x: query points (1D).
        xp: source coordinates (1D, sorted ascending).
        fp: source values (1D, same length as xp).

    Returns:
        Interpolated values at x.
    """
    # searchsorted gives insertion indices
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    w = (x - x0) / (x1 - x0)
    result = f0 + w * (f1 - f0)
    # Constant extrapolation at boundaries
    result = torch.where(x <= xp[0], fp[0], result)
    result = torch.where(x >= xp[-1], fp[-1], result)
    return result


def linear_interp_with_linear_extrap(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
) -> torch.Tensor:
    """1D linear interpolation with linear extrapolation at both ends."""
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    w = (x - x0) / (x1 - x0)
    return f0 + w * (f1 - f0)


# ═══════════════════════════════════════════════════════════════════════════
# 3D field interpolation
# ═══════════════════════════════════════════════════════════════════════════

def _interp_column(
    target_levels: torch.Tensor,
    source_levels: torch.Tensor,
    column: torch.Tensor,
) -> torch.Tensor:
    """Interpolate a single column (source_levels,) → (target_levels,)."""
    return interp_1d(target_levels, source_levels, column)


def interp_pressure_to_sigma(
    fields: dict[str, torch.Tensor],
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Interpolate 3D fields from pressure to sigma levels.

    Args:
        fields: dict of tensors with shape (..., n_pressure, lon, lat).
        pressure_coords: source pressure levels.
        sigma_coords: target sigma levels.
        surface_pressure: surface pressure (..., lon, lat).

    Returns:
        dict of tensors on sigma levels.
    """
    device = surface_pressure.device
    dtype = surface_pressure.dtype
    source = torch.tensor(pressure_coords.centers, dtype=dtype, device=device)
    sigma_c = torch.tensor(sigma_coords.centers, dtype=dtype, device=device)

    def regrid(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3 or x.shape[-3] != pressure_coords.layers:
            return x
        # Vectorized column interpolation
        # target pressures: (n_sigma, lon, lat)
        target = sigma_c[:, None, None] * surface_pressure
        # Flatten spatial dims for batch interpolation
        spatial_shape = x.shape[-2:]
        n_pressure = x.shape[-3]
        n_sigma = len(sigma_c)

        x_flat = x.reshape(-1, n_pressure)  # (n_cols, n_pressure)
        target_flat = target.reshape(n_sigma, -1).T  # (n_cols, n_sigma)

        # Batch interpolation: for each column, interp from source→target
        # source: (n_pressure,), x_flat[col]: (n_pressure,) → target_flat[col]: (n_sigma,)
        idx = torch.searchsorted(source.contiguous(), target_flat.contiguous())
        idx = idx.clamp(1, n_pressure - 1)
        x0 = source[idx - 1]  # (n_cols, n_sigma)
        x1 = source[idx]
        f0 = x_flat.gather(1, idx - 1)  # (n_cols, n_sigma)
        f1 = x_flat.gather(1, idx)
        w = (target_flat - x0) / (x1 - x0 + 1e-30)
        result_flat = f0 + w * (f1 - f0)
        # Constant extrapolation
        result_flat = torch.where(target_flat <= source[0], f0[:, :1].expand_as(result_flat), result_flat)
        result_flat = torch.where(target_flat >= source[-1], f1[:, -1:].expand_as(result_flat), result_flat)
        # Reshape: (n_cols, n_sigma) → (..., n_sigma, lon, lat)
        result = result_flat.T.reshape(n_sigma, *spatial_shape)
        return result

    return {k: regrid(v) for k, v in fields.items()}


def interp_sigma_to_pressure(
    fields: dict[str, torch.Tensor],
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Interpolate 3D fields from sigma to pressure levels."""
    device = surface_pressure.device
    dtype = surface_pressure.dtype
    sigma_c = torch.tensor(sigma_coords.centers, dtype=dtype, device=device)
    target = torch.tensor(pressure_coords.centers, dtype=dtype, device=device)

    def regrid(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3 or x.shape[-3] != len(sigma_c):
            return x
        n_sigma = len(sigma_c)
        n_pressure = pressure_coords.layers
        spatial_shape = x.shape[-2:]

        # Source pressures per column: sigma * sp, shape (n_sigma, lon, lat)
        src_p = sigma_c[:, None, None] * surface_pressure
        # Flatten spatial dims
        x_flat = x.reshape(n_sigma, -1).T  # (n_cols, n_sigma)
        src_flat = src_p.reshape(n_sigma, -1).T  # (n_cols, n_sigma)
        target_expanded = target.unsqueeze(0).expand(x_flat.shape[0], -1)  # (n_cols, n_pressure)

        # Batch interpolation
        idx = torch.searchsorted(src_flat.contiguous(), target_expanded.contiguous())
        idx = idx.clamp(1, n_sigma - 1)
        x0 = src_flat.gather(1, idx - 1)
        x1 = src_flat.gather(1, idx)
        f0 = x_flat.gather(1, idx - 1)
        f1 = x_flat.gather(1, idx)
        w = (target_expanded - x0) / (x1 - x0 + 1e-30)
        result_flat = f0 + w * (f1 - f0)
        # Constant extrapolation
        result_flat = torch.where(target_expanded <= src_flat[:, :1], x_flat[:, :1].expand_as(result_flat), result_flat)
        result_flat = torch.where(target_expanded >= src_flat[:, -1:], x_flat[:, -1:].expand_as(result_flat), result_flat)
        result = result_flat.T.reshape(n_pressure, *spatial_shape)
        return result

    return {k: regrid(v) for k, v in fields.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Conservative regridding
# ═══════════════════════════════════════════════════════════════════════════

def conservative_regrid_weights(
    source_bounds: torch.Tensor,
    target_bounds: torch.Tensor,
) -> torch.Tensor:
    """Weight matrix for conservative regridding.

    Args:
        source_bounds: (n_source+1,) strictly increasing.
        target_bounds: (n_target+1,) strictly increasing.

    Returns:
        (n_target, n_source) weight matrix. Rows sum to 1.
    """
    upper = torch.minimum(
        target_bounds[1:, None], source_bounds[None, 1:]
    )
    lower = torch.maximum(
        target_bounds[:-1, None], source_bounds[None, :-1]
    )
    overlap = torch.clamp(upper - lower, min=0)
    row_sums = overlap.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    return overlap / row_sums


def get_surface_pressure(
    pressure_levels: PressureCoordinates,
    geopotential: torch.Tensor,
    orography: torch.Tensor,
    gravity_acceleration: float,
) -> torch.Tensor:
    """Calculate surface pressure from geopotential on pressure levels.

    Args:
        pressure_levels: pressure level coordinates.
        geopotential: (..., level, x, y).
        orography: (1, x, y).
        gravity_acceleration: g.

    Returns:
        Surface pressure (..., 1, x, y).
    """
    device = geopotential.device
    dtype = geopotential.dtype
    levels = torch.tensor(
        pressure_levels.centers, dtype=dtype, device=device
    )
    relative_height = orography * gravity_acceleration - geopotential
    # Vectorized: interpolate all columns simultaneously
    # relative_height: (..., level, x, y), levels: (level,)
    # We need to find where relative_height crosses zero along the level axis.
    # Reshape for batch interpolation: flatten spatial dims.
    spatial_shape = geopotential.shape[-2:]  # (x, y)
    batch_shape = geopotential.shape[:-3]
    n_levels = geopotential.shape[-3]
    # Flatten to (..., level, n_spatial)
    rh_flat = relative_height.reshape(*batch_shape, n_levels, -1)
    n_spatial = rh_flat.shape[-1]

    # For each spatial column, find the pressure where rh crosses zero
    # using vectorized linear interpolation with linear extrapolation.
    target = torch.zeros(1, dtype=dtype, device=device)
    # rh_flat: (..., level, n_spatial); levels: (level,)
    # We want: for each column, interp target=0 over (rh, levels)
    # Transpose to (..., n_spatial, level) for searchsorted
    rh_cols = rh_flat.movedim(-2, -1)  # (..., n_spatial, level)
    # Sort each column's rh (normally already sorted-ish)
    # Use searchsorted on each column
    target_expanded = target.expand(*batch_shape, n_spatial)  # (..., n_spatial)
    # Vectorized column interpolation: rh_cols[..., k] are the x-coords,
    # levels are the y-coords. We want y where x=0.
    idx = torch.searchsorted(rh_cols.contiguous(), target_expanded.unsqueeze(-1).contiguous())
    idx = idx.squeeze(-1).clamp(1, n_levels - 1)  # (..., n_spatial)
    # Gather neighboring rh and level values
    idx_m1 = (idx - 1).unsqueeze(-1)  # (..., n_spatial, 1)
    idx_0 = idx.unsqueeze(-1)
    rh0 = rh_cols.gather(-1, idx_m1).squeeze(-1)  # (..., n_spatial)
    rh1 = rh_cols.gather(-1, idx_0).squeeze(-1)
    p0 = levels[idx - 1]  # (..., n_spatial)
    p1 = levels[idx]
    w = (0.0 - rh0) / (rh1 - rh0 + 1e-30)
    result_flat = p0 + w * (p1 - p0)
    # Extrapolation: clamp to boundary values
    result_flat = torch.where(
        target_expanded <= rh_cols[..., 0], levels[0], result_flat)
    result_flat = torch.where(
        target_expanded >= rh_cols[..., -1], levels[-1], result_flat)
    # Reshape back
    result = result_flat.reshape(*batch_shape, 1, *spatial_shape)
    return result
