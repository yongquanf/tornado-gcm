"""Encoder/Decoder modules — PyTorch implementation.

Provides:
  - WeatherbenchToPrimitiveEncoder: physics-based encoding
  - LearnedWeatherbenchToPrimitiveEncoder: physics + NN correction
  - PrimitiveToWeatherbenchDecoder: physics-based decoding
  - LearnedPrimitiveToWeatherbenchDecoder: physics + NN correction
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from pytorch_src.core import primitive_equations as pe
from pytorch_src.core import spherical_harmonic
from pytorch_src.core import vertical_interpolation as vi
from pytorch_src.core import coordinate_systems
from pytorch_src.units import SimUnits
from pytorch_src.neural import field_utils


# ═══════════════════════════════════════════════════════════════════════════
# WeatherbenchToPrimitiveEncoder (physics-only)
# ═══════════════════════════════════════════════════════════════════════════

class WeatherbenchToPrimitiveEncoder(nn.Module):
    """Encode WeatherBench data to primitive equation spectral state.

    Pipeline:
      1. Compute surface pressure from geopotential
      2. Vertically interpolate from pressure → sigma levels
      3. Compute vorticity/divergence from u,v
      4. Linearize temperature: T' = T - T_ref
      5. Return modal State

    Args:
        coords: model coordinate system (horizontal + vertical).
        physics_specs: SimUnits for physical constants.
        reference_temperatures: reference T profile (n_levels,).
        input_pressure_levels: pressure levels of input data (Pa).
        orography: nodal orography field (1, lon, lat).
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        input_pressure_levels: np.ndarray,
        orography: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.coords = coords
        self.physics_specs = physics_specs
        self.ref_temps = reference_temperatures
        self.input_pressure = vi.PressureCoordinates(input_pressure_levels)
        if orography is not None:
            self.register_buffer("_orography", orography, persistent=False)
        else:
            self._orography = None

    def weatherbench_to_primitive(
        self,
        wb_dict: Dict[str, torch.Tensor],
    ) -> pe.State:
        """Convert WeatherBench dict to primitive equations State.

        Expected keys in wb_dict:
          - u_component_of_wind: (n_pressure, lon, lat)
          - v_component_of_wind: (n_pressure, lon, lat)
          - temperature: (n_pressure, lon, lat)
          - geopotential: (n_pressure, lon, lat)
          - specific_humidity: (n_pressure, lon, lat) [optional]
          - geopotential_at_surface: (1, lon, lat) [optional, else uses orography * g]
        """
        grid = self.coords.horizontal
        vertical = self.coords.vertical
        g = self.physics_specs.g
        device = wb_dict["temperature"].device
        dtype = wb_dict["temperature"].dtype

        # 1. Surface pressure from geopotential
        geo = wb_dict["geopotential"]
        if "geopotential_at_surface" in wb_dict:
            geo_surface = wb_dict["geopotential_at_surface"]
        elif self._orography is not None:
            geo_surface = self._orography.to(device=device, dtype=dtype) * g
        else:
            # Use lowest pressure level as estimate
            geo_surface = geo[-1:] * 0

        sp = vi.get_surface_pressure(
            self.input_pressure, geo, geo_surface, g,
        )

        # 2. Vertical interpolation: pressure → sigma
        fields_to_interp = {
            "u": wb_dict["u_component_of_wind"],
            "v": wb_dict["v_component_of_wind"],
            "t": wb_dict["temperature"],
        }
        if "specific_humidity" in wb_dict:
            fields_to_interp["q"] = wb_dict["specific_humidity"]

        sigma_fields = vi.interp_pressure_to_sigma(
            fields_to_interp, self.input_pressure, vertical, sp.squeeze(-3),
        )

        # 3. Compute vorticity/divergence
        u_sigma = sigma_fields["u"]
        v_sigma = sigma_fields["v"]
        vor_modal, div_modal = spherical_harmonic.uv_nodal_to_vor_div_modal(
            grid, u_sigma, v_sigma,
        )

        # 4. Linearize temperature
        ref_t = torch.tensor(
            self.ref_temps, dtype=dtype, device=device,
        ).reshape(-1, 1, 1)
        t_var = sigma_fields["t"] - ref_t
        t_var_modal = grid.to_modal(t_var)

        # 5. Log surface pressure
        lnps = torch.log(sp.squeeze(-3)).unsqueeze(0)  # (1, lon, lat)
        lnps_modal = grid.to_modal(lnps)

        # Tracers
        tracers = {}
        if "q" in sigma_fields:
            tracers["specific_humidity"] = grid.to_modal(sigma_fields["q"])

        return pe.State(
            vorticity=vor_modal,
            divergence=div_modal,
            temperature_variation=t_var_modal,
            log_surface_pressure=lnps_modal,
            tracers=tracers,
        )

    def forward(
        self, wb_dict: Dict[str, torch.Tensor],
    ) -> pe.State:
        return self.weatherbench_to_primitive(wb_dict)


# ═══════════════════════════════════════════════════════════════════════════
# LearnedWeatherbenchToPrimitiveEncoder
# ═══════════════════════════════════════════════════════════════════════════

class LearnedWeatherbenchToPrimitiveEncoder(WeatherbenchToPrimitiveEncoder):
    """Physics encoder + neural network correction.

    Pipeline:
      1. Physics-based encoding (parent class)
      2. Extract features from data space and model space
      3. NN predicts nodal corrections
      4. Add corrections to physics-encoded state
      5. Optional perturbation

    Args:
        features_fn: feature extraction module.
        correction_net: NN that maps features → correction dict.
        correction_transform: post-processing for corrections.
        prediction_mask: dict of {key: bool} controlling which fields are corrected.
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        input_pressure_levels: np.ndarray,
        features_fn: nn.Module,
        correction_net: nn.Module,
        correction_transform: Optional[nn.Module] = None,
        prediction_mask: Optional[Dict[str, bool]] = None,
        orography: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            coords, physics_specs, reference_temperatures,
            input_pressure_levels, orography,
        )
        self.features_fn = features_fn
        self.correction_net = correction_net
        self.correction_transform = correction_transform
        self.prediction_mask = prediction_mask or {}

    def forward(
        self,
        wb_dict: Dict[str, torch.Tensor],
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> pe.State:
        # 1. Physics-based encoding
        pe_state = self.weatherbench_to_primitive(wb_dict)

        # 2. Extract features
        grid = self.coords.horizontal
        state_dict = {
            "vorticity": pe_state.vorticity,
            "divergence": pe_state.divergence,
            "temperature_variation": pe_state.temperature_variation,
            "log_surface_pressure": pe_state.log_surface_pressure,
        }
        state_dict.update(pe_state.tracers)

        features = self.features_fn(state_dict, forcing=forcing)

        # 3. NN correction
        corrections = self.correction_net(features)
        if self.correction_transform is not None:
            corrections = self.correction_transform(corrections)

        # 4. Apply corrections (mask controls which fields)
        def _apply_correction(
            field: torch.Tensor, key: str,
        ) -> torch.Tensor:
            if key not in corrections:
                return field
            if key in self.prediction_mask and not self.prediction_mask[key]:
                return field
            corr = corrections[key]
            return field + grid.to_modal(corr) if corr.shape != field.shape else field + corr

        return pe.State(
            vorticity=_apply_correction(pe_state.vorticity, "vorticity"),
            divergence=_apply_correction(pe_state.divergence, "divergence"),
            temperature_variation=_apply_correction(
                pe_state.temperature_variation, "temperature_variation"),
            log_surface_pressure=_apply_correction(
                pe_state.log_surface_pressure, "log_surface_pressure"),
            tracers={k: _apply_correction(v, k) for k, v in pe_state.tracers.items()},
            sim_time=pe_state.sim_time,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PrimitiveToWeatherbenchDecoder (physics-only)
# ═══════════════════════════════════════════════════════════════════════════

class PrimitiveToWeatherbenchDecoder(nn.Module):
    """Decode primitive equation state to WeatherBench dict.

    Pipeline:
      1. Recover u,v from vorticity/divergence
      2. Recover absolute temperature T = T' + T_ref
      3. Compute geopotential
      4. Vertically interpolate sigma → pressure levels
      5. Return WeatherBench-format dict

    Args:
        coords: model coordinate system.
        physics_specs: SimUnits.
        reference_temperatures: reference T profile.
        output_pressure_levels: target pressure levels for output (Pa).
        orography: nodal orography.
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        output_pressure_levels: np.ndarray,
        orography: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.coords = coords
        self.physics_specs = physics_specs
        self.ref_temps = reference_temperatures
        self.output_pressure = vi.PressureCoordinates(output_pressure_levels)
        if orography is not None:
            self.register_buffer("_orography", orography, persistent=False)
        else:
            self._orography = None

    def primitive_to_weatherbench(
        self, state: pe.State,
    ) -> Dict[str, torch.Tensor]:
        """Convert primitive equations State to WeatherBench dict."""
        grid = self.coords.horizontal
        vertical = self.coords.vertical
        g = self.physics_specs.g
        R = self.physics_specs.R
        device = state.vorticity.device
        dtype = state.vorticity.dtype

        # 1. Recover u, v from vorticity/divergence
        cos_uv = spherical_harmonic.get_cos_lat_vector(
            state.vorticity, state.divergence, grid, clip=False,
        )
        sec2 = torch.tensor(grid.sec2_lat, dtype=dtype, device=device)
        u = grid.to_nodal(cos_uv[0]) * sec2
        v = grid.to_nodal(cos_uv[1]) * sec2

        # 2. Absolute temperature
        ref_t = torch.tensor(
            self.ref_temps, dtype=dtype, device=device,
        ).reshape(-1, 1, 1)
        t_nodal = grid.to_nodal(state.temperature_variation) + ref_t

        # 3. Geopotential (simplified: Φ = Φ_s + R*T_virt * ln(P_s/P))
        sp = torch.exp(grid.to_nodal(state.log_surface_pressure))
        if self._orography is not None:
            geo_surface = self._orography.to(device=device, dtype=dtype) * g
        else:
            geo_surface = torch.zeros(1, *sp.shape[-2:], dtype=dtype, device=device)

        from pytorch_src.core.primitive_equations import get_geopotential_on_sigma
        q = None
        if "specific_humidity" in state.tracers:
            q = grid.to_nodal(state.tracers["specific_humidity"])
        geo = get_geopotential_on_sigma(
            t_nodal,
            grid.to_nodal(self._orography.to(dtype=dtype)) if self._orography is not None
            else torch.zeros_like(t_nodal[:1]),
            vertical, g, R,
            specific_humidity=q,
            water_vapor_gas_constant=self.physics_specs.R_vapor if q is not None else None,
        )

        # 4. Interpolate sigma → pressure
        fields = {"u_component_of_wind": u, "v_component_of_wind": v,
                  "temperature": t_nodal, "geopotential": geo}
        for k, v_t in state.tracers.items():
            fields[k] = grid.to_nodal(v_t)

        result = vi.interp_sigma_to_pressure(
            fields, self.output_pressure, vertical, sp.squeeze(0),
        )
        result["surface_pressure"] = sp

        return result

    def forward(self, state: pe.State) -> Dict[str, torch.Tensor]:
        return self.primitive_to_weatherbench(state)


# ═══════════════════════════════════════════════════════════════════════════
# Utility: add_prefix to dict keys
# ═══════════════════════════════════════════════════════════════════════════

def _add_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Prepend ``prefix`` to every key in dict ``d``."""
    return {prefix + k: v for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════
# LearnedPrimitiveToWeatherbenchDecoder
# ═══════════════════════════════════════════════════════════════════════════

class LearnedPrimitiveToWeatherbenchDecoder(PrimitiveToWeatherbenchDecoder):
    """Physics decoder + neural network correction (full JAX-aligned).

    Matches the JAX ``LearnedPrimitiveToWeatherbenchDecoder`` pipeline:
      1. Optional stochastic perturbation of model state
      2. Physics-based decoding (parent ``primitive_to_weatherbench``)
      3. Optional diagnostics appended to decoded dict
      4. Dual feature extraction:
         - ``model_features_fn``: from model-coordinate prognostics
         - ``data_features_fn``: from output-coordinate WB fields
      5. NN correction via ``correction_net``
      6. Additive corrections masked by ``prediction_mask``
      7. Optional output ``transform_fn``

    Args:
        model_features_fn: feature extraction from model-coordinate state.
        data_features_fn: feature extraction from output-coordinate WB fields.
            If ``None``, only model features are used (deterministic default).
        correction_net: NN mapping concatenated features → correction dict.
        correction_transform: optional post-processing on corrections.
        prediction_mask: ``{field: bool}`` controlling which outputs get
            corrections.  ``False`` → skip correction for that field.
        randomness_fn: optional stochastic field module (e.g.
            ``GaussianRandomField``).  Provides ``unconditional_sample()``
            whose ``nodal_value`` is fed to features_fn & perturbation_fn.
        perturbation_fn: optional perturbation module applied to model state
            *before* physics decoding (stochastic models).
        diagnostic_fn: optional diagnostic module producing extra fields
            (e.g. evaporation rate) appended under the ``diagnostics`` key.
        transform_fn: optional final transform on the output dict (e.g.
            re-dimensionalize).
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        output_pressure_levels: np.ndarray,
        model_features_fn: nn.Module,
        correction_net: nn.Module,
        data_features_fn: Optional[nn.Module] = None,
        correction_transform: Optional[nn.Module] = None,
        prediction_mask: Optional[Dict[str, bool]] = None,
        randomness_fn: Optional[nn.Module] = None,
        perturbation_fn: Optional[nn.Module] = None,
        diagnostic_fn: Optional[nn.Module] = None,
        transform_fn: Optional[nn.Module] = None,
        orography: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            coords, physics_specs, reference_temperatures,
            output_pressure_levels, orography,
        )
        self.model_features_fn = model_features_fn
        self.data_features_fn = data_features_fn
        self.correction_net = correction_net
        self.correction_transform = correction_transform
        self.prediction_mask = prediction_mask or {}
        self.randomness_fn = randomness_fn
        self.perturbation_fn = perturbation_fn
        self.diagnostic_fn = diagnostic_fn
        self.transform_fn = transform_fn

    def forward(
        self,
        state: pe.State,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
        randomness=None,
    ) -> Dict[str, torch.Tensor]:
        # ── 0. Optional randomness ──────────────────────────────────────
        randomness_nodal = None
        if self.randomness_fn is not None:
            rng_state = self.randomness_fn.unconditional_sample(randomness)
            randomness_nodal = rng_state.nodal_value

        # ── 1. Optional perturbation ────────────────────────────────────
        if self.perturbation_fn is not None and randomness_nodal is not None:
            state = self.perturbation_fn(
                inputs=state, state=None, randomness=randomness_nodal,
            )

        # ── 2. Optional diagnostics ─────────────────────────────────────
        decoder_diagnostics = None
        if self.diagnostic_fn is not None:
            decoder_diagnostics = self.diagnostic_fn(state)

        # ── 3. Physics-based decoding ───────────────────────────────────
        wb_dict = self.primitive_to_weatherbench(state)

        # Append diagnostics to decoded dict
        if decoder_diagnostics is not None:
            wb_dict["diagnostics"] = decoder_diagnostics

        # ── 4. Dual feature extraction ──────────────────────────────────
        state_dict = {
            "vorticity": state.vorticity,
            "divergence": state.divergence,
            "temperature_variation": state.temperature_variation,
            "log_surface_pressure": state.log_surface_pressure,
        }
        state_dict.update(state.tracers)

        model_kwargs = {"forcing": forcing}
        if randomness_nodal is not None:
            model_kwargs["randomness"] = randomness_nodal
        model_features = self.model_features_fn(state_dict, **model_kwargs)

        if self.data_features_fn is not None:
            # Convert WB nodal fields to modal for data feature extraction
            grid = self.coords.horizontal
            wb_modal = {k: grid.to_modal(v) for k, v in wb_dict.items()
                        if isinstance(v, torch.Tensor) and v.dim() >= 2}
            data_features = self.data_features_fn(wb_modal, forcing=forcing)
            # Merge with prefix
            all_features = _add_prefix(data_features, "data_")
            all_features.update(_add_prefix(model_features, "model_"))
        else:
            all_features = model_features

        # ── 5. NN corrections ───────────────────────────────────────────
        corrections = self.correction_net(all_features)
        if self.correction_transform is not None:
            corrections = self.correction_transform(corrections)

        # ── 6. Apply corrections (masked) ───────────────────────────────
        for key in wb_dict:
            if key in corrections:
                if key in self.prediction_mask and not self.prediction_mask[key]:
                    continue
                corr = corrections[key]
                if corr is not None:
                    wb_dict[key] = wb_dict[key] + corr

        # ── 7. Optional output transform ────────────────────────────────
        if self.transform_fn is not None:
            wb_dict = self.transform_fn(wb_dict)

        return wb_dict
