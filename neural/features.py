# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Feature extraction modules for NeuralGCM — PyTorch implementation.

Provides:
  - CombinedFeatures: aggregates multiple feature modules
  - RadiationFeatures: solar radiation flux
  - ForcingFeatures: external forcing data (SST, sea ice, etc.)
  - PressureFeatures: per-level pressure from log_surface_pressure
  - OrographyFeatures: surface orography
  - LatitudeFeatures: cos/sin latitude
  - RandomnessFeatures: random fields for stochastic parameterization
  - VelocityAndPrognostics: u,v + prognostic variables + optional gradients
  - MemoryVelocityAndValues: time-lagged features from memory state
"""

from __future__ import annotations

from collections import namedtuple
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.core import spherical_harmonic


# ═══════════════════════════════════════════════════════════════════════════
# cos_lat factor tracking
# ═══════════════════════════════════════════════════════════════════════════

KeyWithCosLatFactor = namedtuple("KeyWithCosLatFactor", ["name", "factor_order"])
"""Named key annotating how many cos(lat) factors multiply this field.

Usage:
  KeyWithCosLatFactor("u", 1)        → cos(lat)*u
  KeyWithCosLatFactor("vorticity", 0) → vorticity (no factor)
  KeyWithCosLatFactor("u_dlon", 2)   → cos²(lat)·∂u/∂lon

When converting to nodal space the appropriate sec(lat)^order is applied
so the result is the bare (un-weighted) physical field.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

OROGRAPHY_KEY = "orography"
REFERENCE_DATETIME_KEY = "reference_datetime"


# ═══════════════════════════════════════════════════════════════════════════
# Feature Protocol
# ═══════════════════════════════════════════════════════════════════════════

class FeatureModule(nn.Module):
    """Base class for feature extraction modules."""

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory: Optional[Dict[str, torch.Tensor]] = None,
        diagnostics: Optional[Dict[str, torch.Tensor]] = None,
        randomness: Optional[Dict[str, torch.Tensor]] = None,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# RadiationFeatures
# ═══════════════════════════════════════════════════════════════════════════

class RadiationFeatures(FeatureModule):
    """Compute normalized solar radiation flux matching dinosaur/radiation.py.

    Implements JAX SolarRadiation.normalized exactly:
    1. Converts nondimensional sim_time to orbital/synodic phases using
       reference_datetime and the physical time scale (1/(2*OMEGA)).
    2. Applies equation-of-time correction to the hour angle.
    3. Multiplies by normalized direct solar irradiance (TSI/scale + variation/scale).
    4. Clips negative values (nighttime) to zero.
    """

    # Constants matching dinosaur/radiation.py
    _DAYS_PER_YEAR = 365.25
    _MINUTES_PER_DAY = 1440
    _SPRING_EQUINOX = 79.0 * 2.0 * np.pi / 365.25        # rad
    _PERIHELION = 3.0 * 2.0 * np.pi / 365.25              # rad
    _EARTH_AXIS_INCLINATION = 23.45 * np.pi / 180.0        # rad
    _TSI = 1361.0        # W/m²
    _TSI_VARIATION = 47.0  # W/m²
    _TSI_SCALE = 1361.0 + 47.0  # 1408.0  — normalization denominator
    # Physical time scale: 1 nondim unit = 1/(2*OMEGA) seconds
    _OMEGA = 7.292e-5    # rad/s

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        reference_datetime_str: str = "1979-01-01T00",
    ):
        import datetime as _dt
        super().__init__()
        self.grid = grid

        # Parse reference datetime (matches JAX datetime_to_orbital_time)
        ref = _dt.datetime.fromisoformat(reference_datetime_str)
        days_in_year = _dt.datetime(ref.year, 12, 31).timetuple().tm_yday
        full_days = ref.timetuple().tm_yday - 1
        frac_day = (60 * ref.hour + ref.minute) / self._MINUTES_PER_DAY
        frac_year = (full_days + frac_day) / days_in_year
        self._ref_orbital: float = 2.0 * np.pi * frac_year
        self._ref_synodic: float = 2.0 * np.pi * frac_day

        # Nondim rates: 2π per year/day in nondim time units
        time_scale_s = 1.0 / (2.0 * self._OMEGA)   # seconds per nondim unit
        self._orbital_rate: float = 2.0 * np.pi / (self._DAYS_PER_YEAR * 86400.0 / time_scale_s)
        self._synodic_rate: float = 2.0 * np.pi / (86400.0 / time_scale_s)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        ref = next(iter(inputs.values()))
        device = ref.device
        lon_t = self.grid._buf._lon_torch.to(device=ref.device, dtype=ref.dtype).to(dtype=torch.float32)
        sin_lat_t = self.grid._buf._sin_lat_torch.to(device=ref.device, dtype=ref.dtype).to(dtype=torch.float32)
        cos_lat_t = torch.sqrt(1.0 - sin_lat_t ** 2)

        sim_time_t = inputs.get("sim_time")
        if sim_time_t is not None:
            # Pure-torch path: keep everything as tensor scalars for
            # torch.export / Dynamo compatibility (no data-dependent
            # Python float() calls).  Matches JAX math exactly.
            st = sim_time_t.mean()  # tensor scalar (0-d)
            two_pi = 2.0 * torch.pi
            ref_orbital = torch.tensor(self._ref_orbital, device=device, dtype=torch.float64)
            ref_synodic = torch.tensor(self._ref_synodic, device=device, dtype=torch.float64)
            orbital_rate = torch.tensor(self._orbital_rate, device=device, dtype=torch.float64)
            synodic_rate = torch.tensor(self._synodic_rate, device=device, dtype=torch.float64)

            orbital = ref_orbital + orbital_rate * st
            synodic = ref_synodic + synodic_rate * st
            # Single-step reduction to shrink magnitude (matches JAX)
            orbital = orbital - torch.floor(orbital / two_pi) * two_pi
            synodic = synodic - torch.floor(synodic / two_pi) * two_pi

            # Solar declination (torch trig)
            spring_eq = torch.tensor(self._SPRING_EQUINOX, device=device, dtype=torch.float64)
            axis_incl = torch.tensor(self._EARTH_AXIS_INCLINATION, device=device, dtype=torch.float64)
            sin_dec = torch.sin(axis_incl * torch.sin(orbital - spring_eq))
            cos_dec = torch.cos(axis_incl * torch.sin(orbital - spring_eq))

            # Equation of time (torch trig)
            b = orbital - spring_eq
            eot_min = 9.87 * torch.sin(2.0 * b) - 7.53 * torch.cos(b) - 1.5 * torch.sin(b)
            eot = two_pi * eot_min / self._MINUTES_PER_DAY

            # Hour angle: matches JAX get_hour_angle
            hour_angle = synodic + eot + lon_t.to(torch.float64) - torch.pi  # (lon,)

            # CSZA = cos(lat)·cos(dec)·cos(h) + sin(lat)·sin(dec)
            csza = (
                cos_lat_t.unsqueeze(0).to(torch.float64) * cos_dec
                * torch.cos(hour_angle).unsqueeze(1)
                + sin_lat_t.unsqueeze(0).to(torch.float64) * sin_dec
            )  # (lon, lat)

            # Normalized direct solar irradiance
            perihelion = torch.tensor(self._PERIHELION, device=device, dtype=torch.float64)
            tsi = torch.tensor(self._TSI, device=device, dtype=torch.float64)
            tsi_var = torch.tensor(self._TSI_VARIATION, device=device, dtype=torch.float64)
            tsi_scale = torch.tensor(self._TSI_SCALE, device=device, dtype=torch.float64)
            flux = (tsi + tsi_var * torch.cos(orbital - perihelion)) / tsi_scale

            # Daytime only: clamp(csza * flux, min=0)
            radiation = torch.clamp(csza * flux, min=0.0).unsqueeze(0).to(torch.float32)  # (1, lon, lat)
        else:
            # Fallback: time-mean approximation ≈ cos(lat)
            radiation = cos_lat_t.unsqueeze(0).unsqueeze(0).expand(
                1, self.grid.nodal_shape[-2], -1
            )

        return {"radiation": radiation}


# ═══════════════════════════════════════════════════════════════════════════
# ForcingFeatures
# ═══════════════════════════════════════════════════════════════════════════

class ForcingFeatures(FeatureModule):
    """Extract external forcing fields as features.

    Passes through selected keys from the forcing dict to the feature dict.
    """

    def __init__(self, forcing_to_include: Sequence[str] = ()):
        super().__init__()
        self.forcing_to_include = list(forcing_to_include)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if forcing is None:
            return {}
        out = {}
        for key in self.forcing_to_include:
            if key in forcing:
                val = forcing[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(val)
                if val.ndim == 2:
                    val = val.unsqueeze(0)                     # (H,W) → (1,H,W)
                elif val.ndim == 4:
                    val = val[0]                                # (B,1,H,W) → (1,H,W)
                out[key] = val
        return out


# ═══════════════════════════════════════════════════════════════════════════
# PressureFeatures
# ═══════════════════════════════════════════════════════════════════════════

class PressureFeatures(FeatureModule):
    """Compute per-level pressure field from log_surface_pressure.

    pressure[k] = sigma[k] * exp(log_surface_pressure)
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        sigma_centers: np.ndarray,
    ):
        super().__init__()
        self.grid = grid
        self.register_buffer(
            "_sigma_centers",
            torch.tensor(sigma_centers, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        lnps = inputs.get("log_surface_pressure")
        if lnps is None:
            return {}
        # Modal → nodal
        nodal_lnps = self.grid.to_nodal(lnps)  # (1, lon, lat)
        surface_p = torch.exp(nodal_lnps)  # (1, lon, lat)
        sigma = self._sigma_centers.to(
            device=surface_p.device, dtype=surface_p.dtype,
        ).reshape(-1, 1, 1)
        pressure = sigma * surface_p  # (n_levels, lon, lat)
        return {"pressure": pressure}


# ═══════════════════════════════════════════════════════════════════════════
# OrographyFeatures
# ═══════════════════════════════════════════════════════════════════════════

class OrographyFeatures(FeatureModule):
    """Provide surface orography as a feature.

    Expects orography stored in the module at construction time.
    """

    def __init__(self, nodal_orography: torch.Tensor):
        super().__init__()
        # Shape: (lon, lat) or (1, lon, lat)
        if nodal_orography.ndim == 2:
            nodal_orography = nodal_orography.unsqueeze(0)
        self.register_buffer("_orography", nodal_orography, persistent=False)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        return {OROGRAPHY_KEY: self._orography.to(
            device=next(iter(inputs.values())).device,
        )}


# ═══════════════════════════════════════════════════════════════════════════
# LatitudeFeatures
# ═══════════════════════════════════════════════════════════════════════════

class LatitudeFeatures(FeatureModule):
    """Provide cos(lat) and sin(lat) as features."""

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        ref = next(iter(inputs.values()))
        device = ref.device
        sin_lat_t = self.grid._buf._sin_lat_torch.to(device=ref.device, dtype=ref.dtype).to(dtype=torch.float32)
        cos_lat_t = torch.sqrt(1.0 - sin_lat_t ** 2)

        n_lon = self.grid.nodal_shape[-2]
        # Broadcast to (1, lon, lat)
        sin_lat_field = sin_lat_t.unsqueeze(0).unsqueeze(0).expand(1, n_lon, -1)
        cos_lat_field = cos_lat_t.unsqueeze(0).unsqueeze(0).expand(1, n_lon, -1)

        return {
            "sin_latitude": sin_lat_field,
            "cos_latitude": cos_lat_field,
        }


# ═══════════════════════════════════════════════════════════════════════════
# RandomnessFeatures
# ═══════════════════════════════════════════════════════════════════════════

class RandomnessFeatures(FeatureModule):
    """Inject random fields as features for stochastic parameterization.

    If randomness dict contains 2D fields, they are expanded to (1, lon, lat).
    """

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None,
        randomness: Optional[Dict[str, torch.Tensor]] = None,
        forcing=None,
    ) -> Dict[str, torch.Tensor]:
        if randomness is None:
            return {}
        out = {}
        for key, val in randomness.items():
            if val.ndim == 2:
                val = val.unsqueeze(0)
            out[key] = val
        return out


# ═══════════════════════════════════════════════════════════════════════════
# VelocityAndPrognostics
# ═══════════════════════════════════════════════════════════════════════════

class VelocityAndPrognostics(FeatureModule):
    """Extract u,v velocity and all prognostic fields as nodal features.

    From modal vorticity+divergence, computes cos(lat)*u, cos(lat)*v,
    converts all prognostics to nodal space, and optionally computes
    spatial gradient features (∂/∂lon, ∂/∂lat, ∇²).

    Gradient features are produced by *compute_gradients_fn* which operates
    on ``{KeyWithCosLatFactor: modal_tensor}`` dicts and returns a new dict
    of gradient features.  All features are converted to nodal space with
    appropriate sec(lat)^order weighting.

    Args:
        grid: spherical harmonic Grid.
        fields_to_include: if provided, only include these prognostic keys.
        compute_gradients_fn: optional nn.Module operating on
            ``dict[KeyWithCosLatFactor, Tensor]`` → ``dict[KeyWithCosLatFactor, Tensor]``.
            Typically a :class:`ToModalDiffOperators` instance.
        features_transform_fn: optional final transform on the nodal features dict.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        fields_to_include: Optional[Sequence[str]] = None,
        compute_gradients_fn: Optional[nn.Module] = None,
        features_transform_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.grid = grid
        self.fields_to_include = fields_to_include
        self.compute_gradients_fn = compute_gradients_fn
        self.features_transform_fn = features_transform_fn

    def _extract_features(
        self,
        inputs: Dict[str, torch.Tensor],
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Build nodal feature dict from modal state inputs."""
        grid = self.grid

        # --- modal features keyed with cos_lat tracking ---
        modal_features: Dict[tuple, torch.Tensor] = {}

        # Compute cos_lat*u, cos_lat*v from vorticity + divergence
        if "vorticity" in inputs and "divergence" in inputs:
            cos_lat_uv = spherical_harmonic.get_cos_lat_vector(
                inputs["vorticity"], inputs["divergence"], grid, clip=True,
            )
            modal_features[KeyWithCosLatFactor(prefix + "u", 1)] = cos_lat_uv[0]
            modal_features[KeyWithCosLatFactor(prefix + "v", 1)] = cos_lat_uv[1]

        # Other prognostic fields (order 0)
        skip_keys = {"sim_time", "u_component_of_wind", "v_component_of_wind"}
        for key, val in inputs.items():
            if key in skip_keys:
                continue
            # vorticity/divergence already used for u,v — but if requested,
            # also add them as separate features (consistent with JAX)
            if key in ("vorticity", "divergence"):
                if self.fields_to_include is not None and key in self.fields_to_include:
                    modal_features[KeyWithCosLatFactor(prefix + key, 0)] = val
                continue
            if key == "tracers" and isinstance(val, dict):
                for tk, tv in val.items():
                    if self.fields_to_include is None or tk in self.fields_to_include:
                        modal_features[KeyWithCosLatFactor(prefix + tk, 0)] = tv
                continue
            if self.fields_to_include is not None and key not in self.fields_to_include:
                continue
            modal_features[KeyWithCosLatFactor(prefix + key, 0)] = val

        # --- optional gradient features ---
        if self.compute_gradients_fn is not None:
            diff_features = self.compute_gradients_fn(modal_features)
        else:
            diff_features = {}

        # --- convert all to nodal with sec(lat)^order weighting ---
        _ref = next(iter(inputs.values()))
        _cos_lat = grid._buf._cos_lat_torch.to(device=_ref.device, dtype=_ref.dtype).to(dtype=torch.float32)
        sec_lat = 1.0 / _cos_lat
        sec2_lat = grid._buf._sec2_lat_torch.to(device=_ref.device, dtype=_ref.dtype).to(dtype=torch.float32)
        sec_scales = {0: None, 1: sec_lat, 2: sec2_lat}

        nodal_features: Dict[str, torch.Tensor] = {}
        for key, val in {**modal_features, **diff_features}.items():
            name, order = key
            nodal_val = grid.to_nodal(val)
            scale = sec_scales.get(order)
            if scale is not None:
                nodal_val = nodal_val * scale
            nodal_features[name] = nodal_val

        # --- optional final transform ---
        if self.features_transform_fn is not None:
            nodal_features = self.features_transform_fn(nodal_features)

        return nodal_features

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        return self._extract_features(inputs)


# ═══════════════════════════════════════════════════════════════════════════
# MemoryVelocityAndValues
# ═══════════════════════════════════════════════════════════════════════════

class MemoryVelocityAndValues(VelocityAndPrognostics):
    """Like VelocityAndPrognostics, but operates on the *memory* state.

    Used for time-difference parameterization: the memory typically holds
    the state from the previous outer step.  Keys are prefixed with
    ``memory_`` to distinguish from current-state features.
    """

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory: Optional[Dict[str, torch.Tensor]] = None,
        diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        if memory is None:
            return {}
        return self._extract_features(memory, prefix="memory_")


# ═══════════════════════════════════════════════════════════════════════════
# CombinedFeatures
# ═══════════════════════════════════════════════════════════════════════════

class CombinedFeatures(FeatureModule):
    """Aggregate multiple feature modules into a single feature dict.

    Each sub-module produces a dict of features. All dicts are merged;
    duplicate keys raise an error unless excluded.
    """

    def __init__(
        self,
        feature_modules: Sequence[FeatureModule],
        features_to_exclude: Sequence[str] = (),
    ):
        super().__init__()
        self.feature_modules = nn.ModuleList(feature_modules)
        self.features_to_exclude = set(features_to_exclude)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        combined: Dict[str, torch.Tensor] = {}
        for mod in self.feature_modules:
            features = mod(
                inputs,
                memory=memory,
                diagnostics=diagnostics,
                randomness=randomness,
                forcing=forcing,
            )
            for key, val in features.items():
                if key in self.features_to_exclude:
                    continue
                if key in combined:
                    raise ValueError(f"Duplicate feature key: {key}")
                combined[key] = val
        return combined


# ═══════════════════════════════════════════════════════════════════════════
# FloatDataFeatures
# ═══════════════════════════════════════════════════════════════════════════

class FloatDataFeatures(FeatureModule):
    """Provide precomputed nodal covariate fields as features.

    Stores static nodal fields (e.g. geopotential_at_surface, land_sea_mask)
    and optionally computes spatial gradient features (∂/∂lon, ∂/∂lat, ∇²)
    via spectral differentiation.

    Args:
        grid: spherical harmonic Grid.
        covariates: dict mapping field name → nodal tensor (1, lon, lat).
        compute_gradients_fn: optional gradient module operating on
            ``dict[KeyWithCosLatFactor, Tensor]``.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        covariates: Dict[str, torch.Tensor],
        compute_gradients_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.grid = grid
        self.compute_gradients_fn = compute_gradients_fn
        # Store covariates as buffers
        for key, val in covariates.items():
            if val.ndim == 2:
                val = val.unsqueeze(0)
            self.register_buffer(f"_cov_{key}", val, persistent=False)
        self._cov_keys = list(covariates.keys())

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        device = next(iter(inputs.values())).device
        features: Dict[str, torch.Tensor] = {}
        for key in self._cov_keys:
            features[key] = getattr(self, f"_cov_{key}")

        if self.compute_gradients_fn is not None:
            grid = self.grid
            # Convert to modal, compute gradients, convert back to nodal
            modal_features = {
                KeyWithCosLatFactor(k, 0): grid.to_modal(v)
                for k, v in features.items()
            }
            modal_grads = self.compute_gradients_fn(modal_features)
            _ref_mg = next(iter(modal_features.values()))
            sec_lat = (1.0 / grid._buf._cos_lat_torch.to(device=_ref_mg.device, dtype=_ref_mg.dtype)).to(dtype=torch.float32)
            sec2_lat = grid._buf._sec2_lat_torch.to(device=_ref_mg.device, dtype=_ref_mg.dtype).to(dtype=torch.float32)
            sec_scales = {0: None, 1: sec_lat, 2: sec2_lat}
            for key, val in modal_grads.items():
                name, order = key
                nodal_val = grid.to_nodal(val)
                scale = sec_scales.get(order)
                if scale is not None:
                    nodal_val = nodal_val * scale
                features[name] = nodal_val

        return features


# ═══════════════════════════════════════════════════════════════════════════
# LearnedPositionalFeatures
# ═══════════════════════════════════════════════════════════════════════════

class LearnedPositionalFeatures(FeatureModule):
    """Provide learned positional features at surface nodal locations.

    The parameter has shape (latent_size, lon, lat) and is returned
    unmodified as ``{'learned_positional_features': param}``.

    Args:
        positional_param: nn.Parameter of shape (latent_size, lon, lat).
    """

    def __init__(self, positional_param: nn.Parameter):
        super().__init__()
        self.positional_param = positional_param

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        return {"learned_positional_features": self.positional_param}


# ═══════════════════════════════════════════════════════════════════════════
# EmbeddingVolumeFeatures
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingVolumeFeatures(FeatureModule):
    """Volume embedding via VerticalConvTower on state fields.

    Extracts selected volume fields + pressure, stacks into
    (n_channels, n_levels, lon, lat), applies VerticalConvTower,
    then splits output channels into named CNN features.

    For the deterministic_2_8_deg model:
      Input: 9 channels (u, v, div, vor, temp_var, sh, cl, ci, pressure)
      Output: 32 CNN1D features, each (n_levels, lon, lat)

    Args:
        grid: spherical harmonic Grid.
        conv_tower: VerticalConvTower module.
        sigma_centers: sigma level centers for PressureFeatures.
        n_output: number of output CNN1D features.
        feature_name: prefix for output feature keys (default: 'CNN1D').
        volume_fields: state field names to include.
        embedding_scales: optional ShiftAndNormalize scales dict for
            the embedding model inputs.
        embedding_shifts: optional ShiftAndNormalize shifts dict for
            the embedding model inputs.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        conv_tower: nn.Module,
        sigma_centers: np.ndarray,
        n_output: int = 32,
        feature_name: str = "CNN1D",
        volume_fields: Sequence[str] = (
            "divergence", "vorticity", "u", "v",
            "temperature_variation", "specific_humidity",
            "specific_cloud_liquid_water_content",
            "specific_cloud_ice_water_content",
        ),
        embedding_scales: Optional[Dict[str, float]] = None,
        embedding_shifts: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.grid = grid
        self.conv_tower = conv_tower
        self.n_output = n_output
        self.feature_name = feature_name
        self.volume_fields = list(volume_fields)
        self.register_buffer(
            "_sigma_centers",
            torch.tensor(sigma_centers, dtype=torch.float32),
            persistent=False,
        )
        self.embedding_scales = embedding_scales
        self.embedding_shifts = embedding_shifts

    def _extract_volume_features(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract volume fields in nodal space for the conv tower."""
        grid = self.grid
        features: Dict[str, torch.Tensor] = {}

        # Compute u, v from vorticity + divergence
        if "vorticity" in inputs and "divergence" in inputs:
            cos_lat_uv = spherical_harmonic.get_cos_lat_vector(
                inputs["vorticity"], inputs["divergence"], grid, clip=True,
            )
            sec_lat = (1.0 / grid._buf._cos_lat_torch.to(device=cos_lat_uv[0].device, dtype=torch.float32))
            features["u"] = grid.to_nodal(cos_lat_uv[0]) * sec_lat
            features["v"] = grid.to_nodal(cos_lat_uv[1]) * sec_lat

        # Other state fields
        skip_keys = {"vorticity", "divergence", "sim_time"}
        for key, val in inputs.items():
            if key in skip_keys or key in features:
                continue
            if key == "tracers" and isinstance(val, dict):
                for tk, tv in val.items():
                    if tk in self.volume_fields:
                        features[tk] = grid.to_nodal(tv)
                continue
            if key in self.volume_fields:
                features[key] = grid.to_nodal(val)

        # Also explicitly add vorticity/divergence if requested
        for fld in ("vorticity", "divergence"):
            if fld in self.volume_fields and fld not in features:
                features[fld] = grid.to_nodal(inputs[fld])

        # Add pressure
        lnps = inputs.get("log_surface_pressure")
        if lnps is not None:
            nodal_lnps = grid.to_nodal(lnps)
            surface_p = torch.exp(nodal_lnps)
            sigma = self._sigma_centers.to(
                device=surface_p.device, dtype=surface_p.dtype,
            ).reshape(-1, 1, 1)
            features["pressure"] = sigma * surface_p

        # Synthetic / minimal states often omit cloud tracers; JAX conv_tower is
        # fixed to len(volume_fields)+pressure channels in sorted-key order.
        tmpl = None
        for prefer in (
            "specific_humidity", "temperature_variation", "u", "vorticity",
            "divergence",
        ):
            if prefer in features:
                tmpl = features[prefer]
                break
        if tmpl is None and features:
            tmpl = next(iter(features.values()))
        if tmpl is not None:
            for name in self.volume_fields:
                if name not in features:
                    features[name] = torch.zeros_like(tmpl)

        return features

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None, forcing=None,
    ) -> Dict[str, torch.Tensor]:
        features = self._extract_volume_features(inputs)

        # Apply optional normalization
        if self.embedding_scales is not None:
            for key in list(features.keys()):
                if key in self.embedding_scales and key in self.embedding_shifts:
                    shift = self.embedding_shifts[key]
                    scale = self.embedding_scales[key]
                    features[key] = (features[key] - shift) / scale

        # Stack in sorted key order → (n_channels, n_levels, lon, lat)
        sorted_keys = sorted(features.keys())
        stacked = torch.stack([features[k] for k in sorted_keys], dim=0)

        # Run conv tower: (n_channels, n_levels, lon, lat) → (n_output, n_levels, lon, lat)
        result = self.conv_tower(stacked)

        # Split into named features, assigning in alphabetical key order.
        # JAX's pytree_utils.unstack_to_pytree uses jax.tree_util.tree_flatten
        # which sorts dict keys alphabetically, so result[k] maps to the k-th
        # alphabetically sorted key name (not the k-th numeric key).
        # e.g., for 32 outputs: result[2] → "CNN1D_10" (3rd alphabetically),
        #   NOT "CNN1D_2" (3rd numerically).
        key_names = sorted(
            [f"{self.feature_name}_{i}" for i in range(self.n_output)]
        )
        out: Dict[str, torch.Tensor] = {}
        for k, key in enumerate(key_names):
            out[key] = result[k]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# EmbeddingSurfaceFeatures
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingSurfaceFeatures(FeatureModule):
    """Surface embedding via weighted land/sea/sea-ice EPD towers.

    Implements JAX NodalLandSeaIceEmbedding:
      1. Extract bottom-level state features for land/sea-ice models
      2. Extract SST for sea model
      3. Run each through its surface EPD
      4. Weight by land/sea/ice fractions and combine
      5. Return as 'surface_embedding' features

    Args:
        grid: spherical harmonic Grid.
        land_epd: ForwardTower wrapping the land surface EPD.
        sea_epd: ForwardTower wrapping the sea surface EPD.
        sea_ice_epd: ForwardTower wrapping the sea-ice surface EPD.
        land_sea_mask: nodal land fraction (lon, lat) or (1, lon, lat).
        land_norm: (scales, shifts) dict pair for land model features.
        sea_norm: (scales, shifts) dict pair for sea model features.
        sea_ice_norm: (scales, shifts) dict pair for sea-ice model features.
        output_size: number of output embedding channels (default 8).
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        land_epd: Optional[nn.Module],
        sea_epd: nn.Module,
        sea_ice_epd: Optional[nn.Module],
        land_sea_mask: torch.Tensor,
        land_norm: tuple[Dict[str, float], Dict[str, float]],
        sea_norm: tuple[Dict[str, float], Dict[str, float]],
        sea_ice_norm: tuple[Dict[str, float], Dict[str, float]],
        output_size: int = 8,
        land_bias: Optional[nn.Parameter] = None,
        sea_ice_bias: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.grid = grid
        self.land_epd = land_epd
        self.sea_epd = sea_epd
        self.sea_ice_epd = sea_ice_epd
        self.output_size = output_size
        if land_sea_mask.ndim == 2:
            land_sea_mask = land_sea_mask.unsqueeze(0)
        self.register_buffer("_land_mask", land_sea_mask, persistent=False)
        self.land_scales, self.land_shifts = land_norm
        self.sea_scales, self.sea_shifts = sea_norm
        self.sea_ice_scales, self.sea_ice_shifts = sea_ice_norm
        # "Bias-only" mode (JAX det 1.4°/0.7°): per-surface constant params
        # (output_size, 1, 1) broadcast to (output_size, lon, lat) instead of
        # running a learned Epd on real state features. Land / sea_ice may
        # independently be either an Epd tower or a bias parameter — exactly
        # one of ``{land_epd, land_bias}`` (resp. sea_ice) must be provided.
        if (land_epd is None) == (land_bias is None):
            raise ValueError(
                "EmbeddingSurfaceFeatures: exactly one of {land_epd, "
                "land_bias} must be provided."
            )
        if (sea_ice_epd is None) == (sea_ice_bias is None):
            raise ValueError(
                "EmbeddingSurfaceFeatures: exactly one of {sea_ice_epd, "
                "sea_ice_bias} must be provided."
            )
        # Store bias params as Parameters (matches how ``LearnedPositional-
        # Features`` shares ``physics_positional`` with JaxNeuralComponents).
        # None values are simply not registered.
        if land_bias is not None:
            self.land_bias = land_bias
        if sea_ice_bias is not None:
            self.sea_ice_bias = sea_ice_bias

    def _extract_bottom_level_features(
        self,
        inputs: Dict[str, torch.Tensor],
        fields: Sequence[str] = ("u", "v", "temperature_variation", "specific_humidity"),
    ) -> Dict[str, torch.Tensor]:
        """Extract fields at the bottom sigma level (closest to surface)."""
        grid = self.grid
        features: Dict[str, torch.Tensor] = {}

        # Compute u, v from vorticity + divergence
        if "vorticity" in inputs and "divergence" in inputs:
            cos_lat_uv = spherical_harmonic.get_cos_lat_vector(
                inputs["vorticity"], inputs["divergence"], grid, clip=True,
            )
            sec_lat = (1.0 / grid._buf._cos_lat_torch.to(device=cos_lat_uv[0].device, dtype=torch.float32))
            if "u" in fields:
                u_nodal = grid.to_nodal(cos_lat_uv[0]) * sec_lat
                features["u"] = u_nodal[-1:, :, :]  # bottom level
            if "v" in fields:
                v_nodal = grid.to_nodal(cos_lat_uv[1]) * sec_lat
                features["v"] = v_nodal[-1:, :, :]

        # Other fields
        for key, val in inputs.items():
            if key in ("vorticity", "divergence", "sim_time"):
                continue
            if key == "tracers" and isinstance(val, dict):
                for tk, tv in val.items():
                    if tk in fields:
                        nodal = grid.to_nodal(tv)
                        features[tk] = nodal[-1:, :, :]
                continue
            if key in fields:
                nodal = grid.to_nodal(val)
                features[key] = nodal[-1:, :, :]

        return features

    def _normalize(
        self,
        features: Dict[str, torch.Tensor],
        scales: Dict[str, float],
        shifts: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in features.items():
            if key in scales and key in shifts:
                out[key] = (val - shifts[key]) / scales[key]
            else:
                out[key] = val
        return out

    def _pack_sorted(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack features in sorted key order → (n_features, ..., lon, lat)."""
        parts = []
        for k in sorted(features.keys()):
            t = features[k]
            if t.ndim == 2:
                t = t.unsqueeze(0)                     # (H,W) → (1,H,W)
            elif t.ndim == 4:
                t = t[0]                                # (B,1,H,W) → (1,H,W)
            parts.append(t)
        return torch.cat(parts, dim=0)

    @staticmethod
    def _to_torch(val, device: torch.device) -> torch.Tensor:
        """Convert array-like to torch tensor on *device* (no-op if already)."""
        if isinstance(val, torch.Tensor):
            return val.to(device=device)
        return torch.as_tensor(val, device=device)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        memory=None, diagnostics=None, randomness=None,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(iter(inputs.values())).device

        # Convert any numpy arrays in forcing to torch tensors (HPS / ERA5 paths).
        if forcing is not None:
            forcing = {k: self._to_torch(v, device) for k, v in forcing.items()}

        # --- Land features (4): specific_humidity, temperature_variation, u, v ---
        # In bias-only mode (land_epd is None) we skip feature extraction
        # entirely and use the per-surface constant bias broadcast across the
        # nodal grid (matches JAX ``NodalLandSeaIceEmbedding`` bias-only fn).
        if self.land_epd is not None:
            land_feat = self._extract_bottom_level_features(inputs)
            land_feat = self._normalize(land_feat, self.land_scales, self.land_shifts)
            land_packed = self._pack_sorted(land_feat)
            land_out = self.land_epd(land_packed)  # (8, lon, lat)
        else:
            # ``land_bias`` shape (output_size, 1, 1) broadcasts across lon/lat
            # when multiplied by land_weight (1, lon, lat) below.
            land_out = self.land_bias.to(device=device)

        # --- Sea features (1): sea_surface_temperature ---
        sea_feat: Dict[str, torch.Tensor] = {}
        if forcing is not None and "sea_surface_temperature" in forcing:
            sst = forcing["sea_surface_temperature"]
            if sst.ndim == 2:
                sst = sst.unsqueeze(0)
            sea_feat["sea_surface_temperature"] = sst.to(device=device)
        sea_feat = self._normalize(sea_feat, self.sea_scales, self.sea_shifts)
        # Fallback synthetic SST field: use the grid nodal shape (from
        # ``_land_mask``) rather than ``land_out.shape[1:]`` because in
        # bias-only mode ``land_out`` is (output_size, 1, 1).
        _nodal_shape = self._land_mask.shape[-2:]
        sea_packed = self._pack_sorted(sea_feat) if sea_feat else torch.zeros(
            1, *_nodal_shape, device=device, dtype=land_out.dtype,
        )
        sea_out = self.sea_epd(sea_packed)

        # --- Sea ice features (5): sea_ice_cover, specific_humidity, temp_var, u, v ---
        # In bias-only mode (sea_ice_epd is None), skip feature extraction and
        # use the per-surface constant bias broadcast across the nodal grid.
        if self.sea_ice_epd is not None:
            ice_feat = self._extract_bottom_level_features(inputs)
            if forcing is not None and "sea_ice_cover" in forcing:
                sic = forcing["sea_ice_cover"]
                if sic.ndim == 2:
                    sic = sic.unsqueeze(0)
                ice_feat["sea_ice_cover"] = sic.to(device=device)
            elif "sea_ice_cover" not in ice_feat:
                # Synthetic / offline IC often has no forcings: sea_ice_epd still expects 5 channels.
                ref = next(iter(ice_feat.values()), None)
                if ref is not None:
                    ice_feat["sea_ice_cover"] = torch.zeros_like(ref)
                else:
                    ice_feat["sea_ice_cover"] = torch.zeros(
                        1, *_nodal_shape, device=device, dtype=land_out.dtype,
                    )
            ice_feat = self._normalize(ice_feat, self.sea_ice_scales, self.sea_ice_shifts)
            ice_packed = self._pack_sorted(ice_feat)
            ice_out = self.sea_ice_epd(ice_packed)
        else:
            ice_out = self.sea_ice_bias.to(device=device)

        # --- Combine with land/sea/ice fractions ---
        land_frac = self._land_mask.to(dtype=land_out.dtype)
        sea_frac = 1.0 - land_frac
        sea_ice_frac = torch.zeros_like(land_frac)
        if forcing is not None and "sea_ice_cover" in forcing:
            sic = forcing["sea_ice_cover"]
            if sic.ndim == 2:
                sic = sic.unsqueeze(0)
            sea_ice_frac = sic.to(device=device, dtype=land_out.dtype)

        land_weight = land_frac
        sea_ice_weight = sea_ice_frac * sea_frac
        sea_weight = (1.0 - sea_ice_frac) * sea_frac

        surface_embedding = (
            land_weight * land_out
            + sea_weight * sea_out
            + sea_ice_weight * ice_out
        )

        return {"surface_embedding": surface_embedding}


# ═══════════════════════════════════════════════════════════════════════════
# Feature packing utility
# ═══════════════════════════════════════════════════════════════════════════

def pack_features(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack feature dict into (C, lon, lat) tensor in sorted key order.

    This matches JAX's pytree_utils.pack_pytree which uses tree_flatten
    (alphabetical key ordering for dicts).
    """
    parts = []
    for key in sorted(features.keys()):
        val = features[key]
        if val.ndim == 2:
            val = val.unsqueeze(0)                     # (H,W) → (1,H,W)
        elif val.ndim == 4:
            val = val[0]                                # (B,1,H,W) → (1,H,W)
        parts.append(val)
    return torch.cat(parts, dim=0)
