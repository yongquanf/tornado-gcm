# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Neural-level vertical interpolators — PyTorch implementation.

Provides:
  - LinearOnPressure: linear vertical interpolation on pressure
  - ConservativeOnPressure: conservative vertical regridding
  - get_surface_pressure: estimate surface pressure from geopotential
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import torch

from tornado_gcm.core import vertical_interpolation as vi
from tornado_gcm.core import sigma_coordinates as sigma_mod
from tornado_gcm.neural.coordinates import (
    SigmaLevels,
    HybridLevels,
    PressureLevels,
)
from tornado_gcm.units import SimUnits


def _source_pressures(
    levels: Union[SigmaLevels, HybridLevels, PressureLevels],
    surface_pressure: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pressure at level centers for the given coordinate type."""
    device = surface_pressure.device if surface_pressure is not None else torch.device("cpu")
    dtype = surface_pressure.dtype if surface_pressure is not None else torch.float64

    if isinstance(levels, PressureLevels):
        return torch.tensor(levels.levels, dtype=dtype, device=device)
    elif isinstance(levels, SigmaLevels):
        sigma_c = torch.tensor(levels.centers, dtype=dtype, device=device)
        # sigma_c: (n_levels,), surface_pressure: (..., lon, lat) or scalar
        if surface_pressure is not None:
            return sigma_c.reshape(-1, 1, 1) * surface_pressure.unsqueeze(-3)
        return sigma_c
    elif isinstance(levels, HybridLevels):
        if surface_pressure is not None:
            return levels.pressure_at_level(surface_pressure)
        hybrid = levels.hybrid
        a_c = 0.5 * (hybrid.a_boundaries[:-1] + hybrid.a_boundaries[1:])
        return torch.tensor(a_c, dtype=dtype, device=device)
    else:
        raise TypeError(f"Unsupported level type: {type(levels)}")


class LinearOnPressure:
    """Linear vertical interpolation between any two coordinate types.

    Source fields are interpolated column-wise from source pressure levels
    to target pressure levels using linear interpolation.

    Attributes:
        target_levels: destination vertical coordinate.
        extrapolation: 'linear' or 'constant'.
    """

    def __init__(
        self,
        target_levels: Union[SigmaLevels, PressureLevels, HybridLevels],
        extrapolation: Literal["linear", "constant"] = "linear",
    ):
        self.target_levels = target_levels
        self.extrapolation = extrapolation

    def _interp_fn(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
    ) -> torch.Tensor:
        if self.extrapolation == "linear":
            return vi.linear_interp_with_linear_extrap(x, xp, fp)
        else:
            return vi.interp_1d(x, xp, fp)

    def interpolate_field(
        self,
        field: torch.Tensor,
        source_levels: Union[SigmaLevels, HybridLevels, PressureLevels],
        surface_pressure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate a single 3D field from source to target levels.

        Args:
            field: (..., n_source, lon, lat)
            source_levels: source vertical coordinate.
            surface_pressure: (..., lon, lat) surface pressure.

        Returns:
            (..., n_target, lon, lat) interpolated field.
        """
        src_p = _source_pressures(source_levels, surface_pressure)
        tgt_p = _source_pressures(self.target_levels, surface_pressure)

        n_target = (
            self.target_levels.n_levels
            if isinstance(self.target_levels, PressureLevels)
            else self.target_levels.layers
        )

        out_shape = field.shape[:-3] + (n_target,) + field.shape[-2:]
        result = torch.zeros(out_shape, dtype=field.dtype, device=field.device)

        # Column-wise interpolation
        for i in range(field.shape[-2]):
            for j in range(field.shape[-1]):
                col = field[..., :, i, j]  # (..., n_source)

                # Build per-column source/target pressures
                if src_p.ndim == 1:
                    sp = src_p
                elif src_p.ndim == 3:
                    sp = src_p[:, i, j]
                else:
                    sp = src_p[..., :, i, j].squeeze()

                if tgt_p.ndim == 1:
                    tp = tgt_p
                elif tgt_p.ndim == 3:
                    tp = tgt_p[:, i, j]
                else:
                    tp = tgt_p[..., :, i, j].squeeze()

                result[..., :, i, j] = self._interp_fn(tp, sp, col)

        return result

    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        source_levels: Union[SigmaLevels, HybridLevels, PressureLevels],
    ) -> Dict[str, torch.Tensor]:
        """Interpolate all fields in dict from source to target levels.

        Expects 'surface_pressure' key if source/target is sigma/hybrid.
        """
        surface_pressure = inputs.get("surface_pressure")
        out = {}
        for key, val in inputs.items():
            if key == "surface_pressure":
                continue
            if val.ndim >= 3:
                out[key] = self.interpolate_field(
                    val, source_levels, surface_pressure,
                )
            else:
                out[key] = val
        return out


class ConservativeOnPressure:
    """Conservative vertical regridding between sigma/pressure coordinates.

    Uses overlap-based weights to ensure conservation of vertically
    integrated quantities.

    Attributes:
        target_levels: destination vertical coordinate (usually SigmaLevels).
    """

    def __init__(
        self,
        target_levels: SigmaLevels,
    ):
        self.target_levels = target_levels

    def regrid_field(
        self,
        field: torch.Tensor,
        source_levels: Union[SigmaLevels, PressureLevels],
        surface_pressure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Conservative regridding for a single field.

        Args:
            field: (..., n_source, lon, lat)
            source_levels: source vertical coordinate.
            surface_pressure: surface pressure for sigma coordinates.

        Returns:
            (..., n_target, lon, lat)
        """
        device = field.device
        dtype = field.dtype

        # Build source and target boundaries
        if isinstance(source_levels, SigmaLevels):
            src_bounds = torch.tensor(
                source_levels.sigma.boundaries, dtype=dtype, device=device,
            )
            if surface_pressure is not None:
                src_bounds_p = src_bounds * surface_pressure.mean()  # approx
            else:
                src_bounds_p = src_bounds
        elif isinstance(source_levels, PressureLevels):
            # Create boundaries from midpoints
            p = source_levels.levels
            bounds = np.zeros(len(p) + 1)
            bounds[1:-1] = 0.5 * (p[:-1] + p[1:])
            bounds[0] = max(0, p[0] - 0.5 * (p[1] - p[0]))
            bounds[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
            src_bounds_p = torch.tensor(bounds, dtype=dtype, device=device)
        else:
            raise TypeError(f"Unsupported source: {type(source_levels)}")

        tgt_bounds = torch.tensor(
            self.target_levels.sigma.boundaries, dtype=dtype, device=device,
        )
        if surface_pressure is not None:
            tgt_bounds_p = tgt_bounds * surface_pressure.mean()
        else:
            tgt_bounds_p = tgt_bounds

        weights = vi.conservative_regrid_weights(src_bounds_p, tgt_bounds_p)
        # weights: (n_target, n_source)
        # field: (..., n_source, lon, lat)
        return torch.einsum("ts,...slm->...tlm", weights, field)

    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        source_levels: Union[SigmaLevels, PressureLevels],
    ) -> Dict[str, torch.Tensor]:
        """Conservative regridding of all fields."""
        surface_pressure = inputs.get("surface_pressure")
        out = {}
        for key, val in inputs.items():
            if key == "surface_pressure":
                continue
            if val.ndim >= 3:
                out[key] = self.regrid_field(
                    val, source_levels, surface_pressure,
                )
            else:
                out[key] = val
        return out


def get_surface_pressure(
    geopotential: torch.Tensor,
    geopotential_at_surface: torch.Tensor,
    pressure_levels: PressureLevels,
) -> torch.Tensor:
    """Estimate surface pressure from geopotential on pressure levels.

    Finds the pressure level where (Φ_surface - Φ) = 0 by linear interpolation.

    Args:
        geopotential: (..., n_levels, lon, lat)
        geopotential_at_surface: (..., 1, lon, lat)
        pressure_levels: pressure level coordinates.

    Returns:
        (..., 1, lon, lat) estimated surface pressure.
    """
    device = geopotential.device
    dtype = geopotential.dtype
    p_levels = torch.tensor(pressure_levels.levels, dtype=dtype, device=device)

    relative_height = geopotential_at_surface - geopotential
    # shape: (..., n_levels, lon, lat)

    result_shape = geopotential.shape[:-3] + (1,) + geopotential.shape[-2:]
    result = torch.zeros(result_shape, dtype=dtype, device=device)

    for i in range(geopotential.shape[-2]):
        for j in range(geopotential.shape[-1]):
            rh = relative_height[..., :, i, j]  # (..., n_levels)
            sp = vi.linear_interp_with_linear_extrap(
                torch.zeros(1, dtype=dtype, device=device),
                rh.squeeze(),
                p_levels,
            )
            result[..., 0, i, j] = sp

    return result
