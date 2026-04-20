# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Neural-level PDE equation wrappers — PyTorch implementation.

Provides:
  - get_reference_temperature: vertical reference temperature profile
  - get_temperature_linearization_transform / delinearization_transform
  - PrimitiveEquations: wraps core PrimitiveEquationsSigma with dict state I/O
  - HeldSuarezForcing: Rayleigh friction + Newtonian cooling
  - ComposedODE, SimTimeEquation
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch

from tornado_gcm.core import coordinate_systems
from tornado_gcm.core import primitive_equations as pe
from tornado_gcm.core import sigma_coordinates as sigma_mod
from tornado_gcm.core import time_integration
from tornado_gcm.core import spherical_harmonic
from tornado_gcm.units import SimUnits
from tornado_gcm.neural.coordinates import SigmaLevels, HybridLevels, PressureLevels
from tornado_gcm.neural import field_utils


# ═══════════════════════════════════════════════════════════════════════════
# Reference temperature
# ═══════════════════════════════════════════════════════════════════════════

# Reference values from arco-ERA5 1990-1998 area-weighted mean
REF_PRESSURE = np.array([
    100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000,
    10000, 15000, 20000, 25000, 30000,
], dtype=np.float64)

REF_TEMPERATURE = np.array([
    260.7, 218.4, 211.1, 209.8, 215.1, 218.1, 219.3, 221.8, 230.2,
    237.7, 245.9, 257.2, 264.7, 270.5, 278.0,
], dtype=np.float64)

P_SURF_REF = 101_325.0  # Pa


def get_reference_temperature(
    model_levels: Union[SigmaLevels, HybridLevels, PressureLevels],
    p_surf_ref: float = P_SURF_REF,
) -> np.ndarray:
    """Compute reference temperature profile for given vertical levels.

    Uses log-linear interpolation from ERA5-derived reference values.

    Args:
        model_levels: vertical coordinate description.
        p_surf_ref: reference surface pressure (Pa).

    Returns:
        1D numpy array of reference temperatures at level centers.
    """
    ref_log_p = np.log(REF_PRESSURE)
    ref_t = REF_TEMPERATURE

    if isinstance(model_levels, PressureLevels):
        target_log_p = np.log(model_levels.levels)
    elif isinstance(model_levels, SigmaLevels):
        target_log_p = np.log(model_levels.centers * p_surf_ref)
    elif isinstance(model_levels, HybridLevels):
        hybrid = model_levels.hybrid
        a_c = 0.5 * (hybrid.a_boundaries[:-1] + hybrid.a_boundaries[1:])
        b_c = 0.5 * (hybrid.b_boundaries[:-1] + hybrid.b_boundaries[1:])
        target_log_p = np.log(a_c + b_c * p_surf_ref)
    else:
        raise TypeError(f"Unsupported level type: {type(model_levels)}")

    return np.interp(target_log_p, ref_log_p, ref_t)


# ═══════════════════════════════════════════════════════════════════════════
# Temperature linearization transform helpers
# ═══════════════════════════════════════════════════════════════════════════

def temperature_linearize(
    state_dict: Dict[str, torch.Tensor],
    ref_temperatures: torch.Tensor,
    abs_key: str = "temperature",
    del_key: str = "temperature_variation",
) -> Dict[str, torch.Tensor]:
    """Convert absolute temperature to temperature deviation (T - T_ref)."""
    out = dict(state_dict)
    if abs_key in out:
        t_abs = out.pop(abs_key)
        out[del_key] = field_utils.extract_1d_field_perturbation(
            t_abs, ref_temperatures, axis=-3,
        )
    return out


def temperature_delinearize(
    state_dict: Dict[str, torch.Tensor],
    ref_temperatures: torch.Tensor,
    abs_key: str = "temperature",
    del_key: str = "temperature_variation",
) -> Dict[str, torch.Tensor]:
    """Convert temperature deviation back to absolute temperature (T' + T_ref)."""
    out = dict(state_dict)
    if del_key in out:
        t_var = out.pop(del_key)
        out[abs_key] = field_utils.reconstruct_1d_field_from_ref_values(
            t_var, ref_temperatures, axis=-3,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PrimitiveEquations wrapper (dict state ↔ State conversion)
# ═══════════════════════════════════════════════════════════════════════════

# Standard key names
VORTICITY_KEY = "vorticity"
DIVERGENCE_KEY = "divergence"
TEMPERATURE_KEY = "temperature"
TEMPERATURE_VAR_KEY = "temperature_variation"
LOG_SP_KEY = "log_surface_pressure"
SIM_TIME_KEY = "sim_time"


def _dict_to_state(
    inputs: Dict[str, torch.Tensor],
    tracer_names: Sequence[str] = (),
) -> pe.State:
    """Convert dict representation to core State."""
    tracers = {k: inputs[k] for k in tracer_names if k in inputs}
    sim_time = inputs.get(SIM_TIME_KEY)
    return pe.State(
        vorticity=inputs[VORTICITY_KEY],
        divergence=inputs[DIVERGENCE_KEY],
        temperature_variation=inputs[TEMPERATURE_VAR_KEY],
        log_surface_pressure=inputs[LOG_SP_KEY],
        tracers=tracers,
        sim_time=float(sim_time) if sim_time is not None else None,
    )


def _state_to_dict(
    state: pe.State,
    include_tracers: bool = True,
    is_tendency: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert core State to dict representation."""
    out = {
        VORTICITY_KEY: state.vorticity,
        DIVERGENCE_KEY: state.divergence,
        TEMPERATURE_VAR_KEY: state.temperature_variation,
        LOG_SP_KEY: state.log_surface_pressure,
    }
    if include_tracers:
        out.update(state.tracers)
    return out


class PrimitiveEquations(time_integration.ImplicitExplicitODE):
    """Neural-level PDE wrapper around core PrimitiveEquationsSigma.

    Operates on dict[str, Tensor] states rather than State dataclasses.
    Handles temperature linearization/delinearization automatically.
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        orography: torch.Tensor,
        tracer_names: Sequence[str] = (),
        include_vertical_advection: bool = True,
        humidity_key: Optional[str] = None,
        cloud_keys: Optional[tuple[str, ...]] = None,
    ):
        self.tracer_names = list(tracer_names)
        self.ref_temps = reference_temperatures
        self._ref_temps_tensor: Optional[torch.Tensor] = None

        self._core_eq = pe.PrimitiveEquationsSigma(
            reference_temperature=reference_temperatures,
            orography=orography,
            coords=coords,
            physics_specs=physics_specs,
            include_vertical_advection=include_vertical_advection,
            humidity_key=humidity_key,
            cloud_keys=cloud_keys,
        )

    def _get_ref_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._ref_temps_tensor is None or self._ref_temps_tensor.device != device:
            self._ref_temps_tensor = torch.tensor(
                self.ref_temps, dtype=dtype, device=device,
            )
        return self._ref_temps_tensor.to(dtype=dtype)

    def _to_state(self, inputs: Dict[str, torch.Tensor]) -> pe.State:
        ref = self._get_ref_tensor(
            inputs[VORTICITY_KEY].device, inputs[VORTICITY_KEY].dtype,
        )
        linearized = temperature_linearize(inputs, ref)
        return _dict_to_state(linearized, self.tracer_names)

    def _from_state(
        self, state: pe.State, is_tendency: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out = _state_to_dict(state, is_tendency=is_tendency)
        if not is_tendency:
            ref = self._get_ref_tensor(
                state.vorticity.device, state.vorticity.dtype,
            )
            out = temperature_delinearize(out, ref)
        return out

    def explicit_terms(self, state):
        if isinstance(state, dict):
            s = self._to_state(state)
            result = self._core_eq.explicit_terms(s)
            return self._from_state(result, is_tendency=True)
        return self._core_eq.explicit_terms(state)

    def implicit_terms(self, state):
        if isinstance(state, dict):
            s = self._to_state(state)
            result = self._core_eq.implicit_terms(s)
            return self._from_state(result, is_tendency=True)
        return self._core_eq.implicit_terms(state)

    def implicit_inverse(self, state, step_size):
        if isinstance(state, dict):
            s = self._to_state(state)
            result = self._core_eq.implicit_inverse(s, step_size)
            return self._from_state(result, is_tendency=False)
        return self._core_eq.implicit_inverse(state, step_size)


class SimTimeEquation(time_integration.ExplicitODE):
    """Trivial ODE for simulation time: dt/dt = 1."""

    def explicit_terms(self, state):
        if isinstance(state, dict):
            out = {k: torch.zeros_like(v) for k, v in state.items()}
            if SIM_TIME_KEY in out:
                out[SIM_TIME_KEY] = torch.ones_like(out[SIM_TIME_KEY])
            return out
        return state


class HeldSuarezForcing(time_integration.ExplicitODE):
    """Held-Suarez forcing: Rayleigh friction + Newtonian cooling.

    This is an explicit-only ODE that provides idealized forcing terms
    for testing the primitive equations without realistic physics.
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        sigma_b: float = 0.7,
        kf: float = 1.0,     # nondimensional Rayleigh friction rate
        ka: float = 1.0 / 40,  # nondimensional tropospheric cooling rate
        ks: float = 1.0 / 4,   # nondimensional stratospheric cooling rate
        min_t: float = 200.0,
        max_t: float = 315.0,
        d_ty: float = 60.0,    # meridional temperature difference
        d_thz: float = 10.0,   # vertical temperature difference
    ):
        self.coords = coords
        self.physics_specs = physics_specs
        self.ref_temps = reference_temperatures
        self.sigma_b = sigma_b
        self.kf = kf
        self.ka = ka
        self.ks = ks
        self.min_t = min_t
        self.max_t = max_t
        self.d_ty = d_ty
        self.d_thz = d_thz

    def explicit_terms(self, state: pe.State) -> pe.State:
        grid = self.coords.horizontal
        sigma = self.coords.vertical
        kappa = self.physics_specs.kappa

        # σ levels
        sigma_c = torch.tensor(
            sigma.centers, dtype=state.vorticity.dtype,
            device=state.vorticity.device,
        ).reshape(-1, 1, 1)

        # Rayleigh friction coefficient: kf * max(0, (σ-σ_b)/(1-σ_b))
        kv = self.kf * torch.clamp(
            (sigma_c - self.sigma_b) / (1.0 - self.sigma_b), min=0.0,
        )

        # Newtonian cooling coefficient
        _, sin_lat = grid.nodal_axes
        sin_lat_t = torch.tensor(sin_lat, dtype=state.vorticity.dtype,
                                 device=state.vorticity.device)
        cos2_lat = 1.0 - sin_lat_t ** 2

        kt = self.ka + (self.ks - self.ka) * torch.clamp(
            (sigma_c - self.sigma_b) / (1.0 - self.sigma_b), min=0.0,
        ) * cos2_lat

        # Equilibrium temperature
        T_eq = (self.max_t - self.d_ty * sin_lat_t ** 2
                - self.d_thz * torch.log(sigma_c / 1.0) * cos2_lat) * sigma_c ** kappa
        T_eq = torch.clamp(T_eq, min=self.min_t)

        # Temperature tendency: -kt * (T - T_eq)
        nodal_T = grid.to_nodal(state.temperature_variation)
        T_ref = torch.tensor(
            self.ref_temps, dtype=nodal_T.dtype, device=nodal_T.device,
        ).reshape(-1, 1, 1)
        T_full = nodal_T + T_ref
        temp_tendency = -kt * (T_full - T_eq)

        # Friction tendency on vorticity/divergence
        nodal_vor = grid.to_nodal(state.vorticity)
        nodal_div = grid.to_nodal(state.divergence)
        vor_tendency = grid.to_modal(-kv * nodal_vor)
        div_tendency = grid.to_modal(-kv * nodal_div)

        return pe.State(
            vorticity=vor_tendency,
            divergence=div_tendency,
            temperature_variation=grid.to_modal(temp_tendency),
            log_surface_pressure=torch.zeros_like(state.log_surface_pressure),
            tracers={k: torch.zeros_like(v) for k, v in state.tracers.items()},
            sim_time=state.sim_time,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Moist Primitive Equations wrappers
# ═══════════════════════════════════════════════════════════════════════════

class MoistPrimitiveEquations(PrimitiveEquations):
    """Primitive equations with moisture: enables humidity_key='specific_humidity'.

    Delegates to core PrimitiveEquationsSigma with humidity_key set, which
    activates the virtual-temperature divergence/vorticity corrections
    (divergence_tendency_due_to_humidity, vorticity_tendency_due_to_humidity).
    """

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        orography: torch.Tensor,
        tracer_names: Sequence[str] = ("specific_humidity",),
        include_vertical_advection: bool = True,
    ):
        super().__init__(
            coords=coords,
            physics_specs=physics_specs,
            reference_temperatures=reference_temperatures,
            orography=orography,
            tracer_names=tracer_names,
            include_vertical_advection=include_vertical_advection,
            humidity_key="specific_humidity",
            cloud_keys=self.CLOUD_KEYS,
        )


class MoistPrimitiveEquationsWithCloudMoisture(PrimitiveEquations):
    """Primitive equations with moisture and cloud water/ice tracers.

    Enables humidity_key and adds cloud water content tracers.
    """

    CLOUD_KEYS = (
        "specific_cloud_liquid_water_content",
        "specific_cloud_ice_water_content",
    )

    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: SimUnits,
        reference_temperatures: np.ndarray,
        orography: torch.Tensor,
        tracer_names: Sequence[str] = (
            "specific_humidity",
            "specific_cloud_liquid_water_content",
            "specific_cloud_ice_water_content",
        ),
        include_vertical_advection: bool = True,
    ):
        super().__init__(
            coords=coords,
            physics_specs=physics_specs,
            reference_temperatures=reference_temperatures,
            orography=orography,
            tracer_names=tracer_names,
            include_vertical_advection=include_vertical_advection,
            humidity_key="specific_humidity",
        )
