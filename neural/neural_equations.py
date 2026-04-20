"""Neural equation wrappers — DirectNeuralEquations / DivCurlNeuralEquations.

These wrap features → NN mapping → tendency transforms → modal conversion
into ExplicitODE interfaces for composing with dynamics core.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn

from pytorch_src.core import primitive_equations as pe
from pytorch_src.core import spherical_harmonic
from pytorch_src.core import time_integration


class DirectNeuralEquations(time_integration.ExplicitODE, nn.Module):
    """Predict modal tendencies directly from NN output.

    Pipeline:
      1. features_fn(state) → nodal feature dict
      2. mapping(features) → nodal tendency dict
      3. tendency_transform(tendencies) → post-processed
      4. to_modal → modal tendencies
      5. filter (optional)

    Args:
        grid: spherical harmonic Grid.
        features_fn: callable producing feature dict from state.
        mapping: nn.Module mapping features → tendency dict.
        tendency_transform: optional post-processing transform.
        filter_fn: optional spectral filter.
        prediction_keys: which tendency keys to output (others → 0).
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        features_fn: nn.Module,
        mapping: nn.Module,
        tendency_transform: Optional[nn.Module] = None,
        filter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        prediction_keys: Optional[Sequence[str]] = None,
    ):
        nn.Module.__init__(self)
        self.grid = grid
        self.features_fn = features_fn
        self.mapping = mapping
        self.tendency_transform = tendency_transform
        self.filter_fn = filter_fn
        self.prediction_keys = set(prediction_keys) if prediction_keys else None

    def explicit_terms(self, state) -> pe.State:
        # Convert State → dict if needed
        if isinstance(state, pe.State):
            inputs = {
                "vorticity": state.vorticity,
                "divergence": state.divergence,
                "temperature_variation": state.temperature_variation,
                "log_surface_pressure": state.log_surface_pressure,
            }
            inputs.update(state.tracers)
        else:
            inputs = state

        # 1. Extract features
        features = self.features_fn(inputs)

        # 2. NN mapping
        tendencies = self.mapping(features)

        # 3. Post-process
        if self.tendency_transform is not None:
            tendencies = self.tendency_transform(tendencies)

        # 4. To modal
        grid = self.grid
        modal_tendencies = {}
        for key, val in tendencies.items():
            try:
                modal_tendencies[key] = grid.to_modal(val)
            except Exception:
                modal_tendencies[key] = val

        # 5. Filter
        if self.filter_fn is not None:
            modal_tendencies = {
                k: self.filter_fn(v) for k, v in modal_tendencies.items()
            }

        # Build output State
        def _get(key, shape_ref):
            if self.prediction_keys and key not in self.prediction_keys:
                return torch.zeros_like(shape_ref)
            return modal_tendencies.get(key, torch.zeros_like(shape_ref))

        if isinstance(state, pe.State):
            return pe.State(
                vorticity=_get("vorticity", state.vorticity),
                divergence=_get("divergence", state.divergence),
                temperature_variation=_get("temperature_variation", state.temperature_variation),
                log_surface_pressure=_get("log_surface_pressure", state.log_surface_pressure),
                tracers={k: _get(k, v) for k, v in state.tracers.items()},
                sim_time=state.sim_time,
            )
        return modal_tendencies


class DivCurlNeuralEquations(time_integration.ExplicitODE, nn.Module):
    """Predict u,v tendencies → convert to divergence/vorticity via div/curl.

    Similar to DirectNeuralEquations but converts wind tendencies to
    vorticity/divergence tendencies through spectral div/curl operators.

    Args:
        grid: spherical harmonic Grid.
        features_fn: callable producing feature dict from state.
        mapping: nn.Module mapping features → tendency dict (outputs u,v keys).
        tendency_transform: optional post-processing transform.
        filter_fn: optional spectral filter.
        u_key: key for zonal wind tendency.
        v_key: key for meridional wind tendency.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        features_fn: nn.Module,
        mapping: nn.Module,
        tendency_transform: Optional[nn.Module] = None,
        filter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        u_key: str = "u_component_of_wind",
        v_key: str = "v_component_of_wind",
    ):
        nn.Module.__init__(self)
        self.grid = grid
        self.features_fn = features_fn
        self.mapping = mapping
        self.tendency_transform = tendency_transform
        self.filter_fn = filter_fn
        self.u_key = u_key
        self.v_key = v_key

    def explicit_terms(self, state) -> pe.State:
        if isinstance(state, pe.State):
            inputs = {
                "vorticity": state.vorticity,
                "divergence": state.divergence,
                "temperature_variation": state.temperature_variation,
                "log_surface_pressure": state.log_surface_pressure,
            }
            inputs.update(state.tracers)
        else:
            inputs = state

        # 1. Extract features
        features = self.features_fn(inputs)

        # 2. NN mapping → produces u, v tendencies
        tendencies = self.mapping(features)

        # 3. Post-process
        if self.tendency_transform is not None:
            tendencies = self.tendency_transform(tendencies)

        # 4. Convert u,v → div/curl in modal space
        grid = self.grid
        u = tendencies.pop(self.u_key, None)
        v = tendencies.pop(self.v_key, None)

        modal_tendencies = {}
        if u is not None and v is not None:
            sec2 = torch.tensor(
                grid.sec2_lat, dtype=u.dtype, device=u.device,
            )
            u_modal = grid.to_modal(u * sec2)
            v_modal = grid.to_modal(v * sec2)
            modal_tendencies["divergence"] = grid.div_cos_lat((u_modal, v_modal))
            modal_tendencies["vorticity"] = grid.curl_cos_lat((u_modal, v_modal))

        # Transform remaining fields to modal
        for key, val in tendencies.items():
            try:
                modal_tendencies[key] = grid.to_modal(val)
            except Exception:
                modal_tendencies[key] = val

        # 5. Filter
        if self.filter_fn is not None:
            modal_tendencies = {
                k: self.filter_fn(v) for k, v in modal_tendencies.items()
            }

        # Build output State
        if isinstance(state, pe.State):
            return pe.State(
                vorticity=modal_tendencies.get("vorticity", torch.zeros_like(state.vorticity)),
                divergence=modal_tendencies.get("divergence", torch.zeros_like(state.divergence)),
                temperature_variation=modal_tendencies.get(
                    "temperature_variation", torch.zeros_like(state.temperature_variation)),
                log_surface_pressure=modal_tendencies.get(
                    "log_surface_pressure", torch.zeros_like(state.log_surface_pressure)),
                tracers={k: modal_tendencies.get(k, torch.zeros_like(v))
                         for k, v in state.tracers.items()},
                sim_time=state.sim_time,
            )
        return modal_tendencies
