"""Feature transforms — PyTorch implementation.

Provides:
  - ShiftAndNormalize / InverseShiftAndNormalize
  - NondimensionalizeTransform
  - TendencyTransform (level-scale + clip)
  - ToModalWithDivCurl
  - SequentialTransform
  - SoftClip / HardClip
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from pytorch_src.core import spherical_harmonic


# ═══════════════════════════════════════════════════════════════════════════
# ShiftAndNormalize
# ═══════════════════════════════════════════════════════════════════════════

class ShiftAndNormalize(nn.Module):
    """Input standardization: (x - shift) / scale.

    Args:
        shifts: dict mapping field name → shift value (scalar or tensor).
        scales: dict mapping field name → scale value (scalar or tensor).
        global_scale: optional global multiplier applied to all scales.
    """

    def __init__(
        self,
        shifts: Dict[str, float],
        scales: Dict[str, float],
        global_scale: Optional[float] = None,
    ):
        super().__init__()
        self.shifts = dict(shifts)
        self.scales = dict(scales)
        if global_scale is not None:
            self.scales = {k: v * global_scale for k, v in self.scales.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if key in self.shifts and key in self.scales:
                shift = self.shifts[key]
                scale = self.scales[key]
                if isinstance(shift, torch.Tensor):
                    shift = shift.to(val.device, val.dtype)
                if isinstance(scale, torch.Tensor):
                    scale = scale.to(val.device, val.dtype)
                out[key] = (val - shift) / scale
            else:
                out[key] = val
        return out


class InverseShiftAndNormalize(nn.Module):
    """Reverse standardization: x * scale + shift.

    Args:
        shifts: dict mapping field name → shift value.
        scales: dict mapping field name → scale value.
        global_scale: optional global multiplier applied to all scales.
    """

    def __init__(
        self,
        shifts: Dict[str, float],
        scales: Dict[str, float],
        global_scale: Optional[float] = None,
    ):
        super().__init__()
        self.shifts = dict(shifts)
        self.scales = dict(scales)
        if global_scale is not None:
            self.scales = {k: v * global_scale for k, v in self.scales.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if key in self.shifts and key in self.scales:
                shift = self.shifts[key]
                scale = self.scales[key]
                if isinstance(shift, torch.Tensor):
                    shift = shift.to(val.device, val.dtype)
                if isinstance(scale, torch.Tensor):
                    scale = scale.to(val.device, val.dtype)
                out[key] = val * scale + shift
            else:
                out[key] = val
        return out


# ═══════════════════════════════════════════════════════════════════════════
# NondimensionalizeTransform
# ═══════════════════════════════════════════════════════════════════════════

class NondimensionalizeTransform(nn.Module):
    """Convert dimensional inputs to nondimensional form.

    Uses SimUnits to scale each field according to its physical unit.

    Args:
        inputs_to_units_mapping: dict of {field_name: unit_string}
            e.g. {'temperature': 'K', 'u_component_of_wind': 'm/s'}
        physics_specs: SimUnits instance for nondimensionalization.
    """

    def __init__(
        self,
        inputs_to_units_mapping: Dict[str, str],
        physics_specs,
    ):
        super().__init__()
        self.mapping = dict(inputs_to_units_mapping)
        self.physics_specs = physics_specs

        # Precompute scale factors
        self._scale_factors: Dict[str, float] = {}
        for key, unit_str in self.mapping.items():
            from pytorch_src import scales
            q = 1.0 * scales.units(unit_str)
            self._scale_factors[key] = self.physics_specs.nondimensionalize(q)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if key in self._scale_factors:
                out[key] = val * self._scale_factors[key]
            else:
                out[key] = val
        return out


# ═══════════════════════════════════════════════════════════════════════════
# ToModalWithDivCurl
# ═══════════════════════════════════════════════════════════════════════════

class ToModalWithDivCurl(nn.Module):
    """Convert nodal (u, v, ...) to modal (divergence, vorticity, ...).

    Pops u and v from the dict, multiplies by sec(lat), transforms to
    modal space, then computes div and curl to get divergence/vorticity.
    All other fields are transformed to modal via standard SHT.

    Args:
        grid: spherical harmonic grid.
        u_key: key for zonal wind component.
        v_key: key for meridional wind component.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        u_key: str = "u_component_of_wind",
        v_key: str = "v_component_of_wind",
    ):
        super().__init__()
        self.grid = grid
        self.u_key = u_key
        self.v_key = v_key

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(inputs)
        u = out.pop(self.u_key, None)
        v = out.pop(self.v_key, None)

        grid = self.grid
        sec2 = torch.tensor(
            grid.sec2_lat, dtype=u.dtype if u is not None else torch.float32,
            device=next(iter(out.values())).device if out else torch.device("cpu"),
        )

        if u is not None and v is not None:
            # u_sec = u * sec(lat), v_sec = v * sec(lat)
            u_modal = grid.to_modal(u * sec2)
            v_modal = grid.to_modal(v * sec2)
            out["divergence"] = grid.div_cos_lat((u_modal, v_modal))
            out["vorticity"] = grid.curl_cos_lat((u_modal, v_modal))

        # Transform remaining nodal fields to modal
        result = {}
        for key, val in out.items():
            try:
                result[key] = grid.to_modal(val)
            except Exception:
                result[key] = val

        return result


# ═══════════════════════════════════════════════════════════════════════════
# TendencyTransform
# ═══════════════════════════════════════════════════════════════════════════

class TendencyTransform(nn.Module):
    """Post-process NN-predicted tendencies with per-level scaling and clipping.

    Args:
        level_scales: optional dict of {field_name: 1D tensor of per-level scales}.
        clip_min: optional dict of {field_name: min_value}.
        clip_max: optional dict of {field_name: max_value}.
    """

    def __init__(
        self,
        level_scales: Optional[Dict[str, torch.Tensor]] = None,
        clip_min: Optional[Dict[str, float]] = None,
        clip_max: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.level_scales = level_scales or {}
        self.clip_min = clip_min or {}
        self.clip_max = clip_max or {}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if key in self.level_scales:
                scale = self.level_scales[key]
                if isinstance(scale, torch.Tensor):
                    scale = scale.to(val.device, val.dtype)
                    shape = [1] * val.ndim
                    shape[-3] = -1
                    scale = scale.reshape(shape)
                val = val * scale
            if key in self.clip_min:
                val = torch.clamp(val, min=self.clip_min[key])
            if key in self.clip_max:
                val = torch.clamp(val, max=self.clip_max[key])
            out[key] = val
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Clipping transforms
# ═══════════════════════════════════════════════════════════════════════════

class SoftClip(nn.Module):
    """Smooth clipping via softplus, matching JAX NeuralGCM implementation.

    Maps values from (-inf, inf) to (-max_value, max_value) with smooth
    boundaries controlled by hinge_softness. Values outside the range are
    mapped into intervals of width approximately ``log(2) * hinge_softness``
    on the interior of each boundary.

    Args:
        max_value: symmetric clipping range (-max_value, max_value).
        hinge_softness: controls smoothness at boundaries.
    """

    def __init__(self, max_value: float = 16.0, hinge_softness: float = 1.0):
        super().__init__()
        if max_value < 0 or hinge_softness < 0:
            raise ValueError(
                f"max_value and hinge_softness must be positive, "
                f"got {max_value=}, {hinge_softness=}"
            )
        self.max_value = max_value
        self.hinge_softness = hinge_softness
        low = -max_value
        high = max_value
        self._low = low
        self._high = high
        self._range = high - low
        # Precompute softplus(range/hinge) for the normalization denominator
        self._sp_range = hinge_softness * torch.nn.functional.softplus(
            torch.tensor((high - low) / hinge_softness)
        ).item()

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hinge_softness
        return h * torch.nn.functional.softplus(x / h)

    def _clip_fn(self, x: torch.Tensor) -> torch.Tensor:
        inner = self._softplus(x - self._low)
        outer = self._softplus(self._range - inner)
        return -outer * self._range / self._sp_range + self._high

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self._clip_fn(v) for k, v in inputs.items()}


class InverseLevelScale(nn.Module):
    """Per-level scaling of selected feature keys.

    Multiplies values by ``1 / scales[level]`` for specified keys,
    matching JAX ``InverseLevelScale``.

    Args:
        keys_to_scale: feature keys to apply level scaling to.
        scales: 1D array of per-level scale factors (length = n_levels).
    """

    def __init__(
        self,
        keys_to_scale: Sequence[str],
        scales: Sequence[float],
    ):
        super().__init__()
        self.keys_to_scale = set(keys_to_scale)
        # Store inverse scales as buffer
        inv_scales = 1.0 / torch.tensor(scales, dtype=torch.float32)
        self.register_buffer("_inv_scales", inv_scales, persistent=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if key in self.keys_to_scale:
                inv_s = self._inv_scales.to(device=val.device, dtype=val.dtype)
                # Reshape to (n_levels, 1, 1) for broadcasting
                shape = [1] * val.ndim
                shape[0] = -1
                inv_s = inv_s.reshape(*shape)
                out[key] = val * inv_s
            else:
                out[key] = val
        return out


class HardClip(nn.Module):
    """Hard clipping using torch.clamp."""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: torch.clamp(v, min=self.min_val, max=self.max_val)
                for k, v in inputs.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Sequential
# ═══════════════════════════════════════════════════════════════════════════

class SequentialTransform(nn.Module):
    """Apply multiple transforms sequentially."""

    def __init__(self, transforms: Sequence[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            inputs = t(inputs)
        return inputs


# ═══════════════════════════════════════════════════════════════════════════
# Identity / Empty transforms
# ═══════════════════════════════════════════════════════════════════════════

class IdentityTransform(nn.Module):
    """Pass-through transform that returns inputs unchanged."""

    def __init__(self, grid=None, **kwargs):
        super().__init__()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return inputs


class EmptyTransform(nn.Module):
    """Transform that returns an empty dict (no features)."""

    def __init__(self, grid=None, **kwargs):
        super().__init__()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# ToModalDiffOperators — gradient ∂/∂lon, ∂/∂lat, ∇²
# ═══════════════════════════════════════════════════════════════════════════

class ToModalDiffOperators(nn.Module):
    """Compute spectral gradient and Laplacian features of modal input fields.

    For each input key (name, cos_lat_order):
      - name_dlon: cos(lat)*∂f/∂lon  (cos_lat_order + 1)
      - name_dlat: cos(lat)*∂f/∂lat  (cos_lat_order + 1)
      - name_del2: ∇²f              (cos_lat_order unchanged)

    The cos_lat_order tracking prevents accidental accumulation of
    cos(lat) factors when features are later converted to nodal space.

    Args:
        grid: spherical harmonic Grid providing cos_lat_grad and laplacian.
    """

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def forward(
        self,
        inputs: Dict[tuple[str, int], torch.Tensor],
    ) -> Dict[tuple[str, int], torch.Tensor]:
        """Compute diff-operator features.

        Args:
            inputs: dict mapping (name, cos_lat_order) → modal tensor.

        Returns:
            dict mapping (name_dlon/dlat/del2, new_order) → modal tensor.
        """
        features: Dict[tuple[str, int], torch.Tensor] = {}
        for key, value in inputs.items():
            name, cos_lat_order = key
            d_dlon, d_dlat = self.grid.cos_lat_grad(value, clip=False)
            laplacian_value = self.grid.laplacian(value)
            features[(name + "_dlon", cos_lat_order + 1)] = d_dlon
            features[(name + "_dlat", cos_lat_order + 1)] = d_dlat
            features[(name + "_del2", cos_lat_order)] = laplacian_value
        return features


# ═══════════════════════════════════════════════════════════════════════════
# ModalToNodal / NodalToModal transforms
# ═══════════════════════════════════════════════════════════════════════════

class ModalToNodalTransform(nn.Module):
    """Transform that converts modal inputs to nodal representation."""

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.grid.to_nodal(v) for k, v in inputs.items()}


class NodalToModalTransform(nn.Module):
    """Transform that converts nodal inputs to modal representation."""

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.grid.to_modal(v) for k, v in inputs.items()}


# ═══════════════════════════════════════════════════════════════════════════
# ClipTransform — wavenumber clipping
# ═══════════════════════════════════════════════════════════════════════════

class ClipTransform(nn.Module):
    """Clip highest total wavenumbers from modal fields.

    Args:
        grid: spherical harmonic Grid with clip_wavenumbers method.
        wavenumbers_to_clip: number of highest wavenumbers to zero out.
    """

    def __init__(self, grid: spherical_harmonic.Grid, wavenumbers_to_clip: int = 1):
        super().__init__()
        self.grid = grid
        self.wavenumbers_to_clip = wavenumbers_to_clip

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: self.grid.clip_wavenumbers(v, n=self.wavenumbers_to_clip)
            for k, v in inputs.items()
        }
