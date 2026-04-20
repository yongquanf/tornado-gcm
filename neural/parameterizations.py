"""Neural parameterizations — Z3 (BF16) → Z1 boundary management.

Implements:
  - ModalNeuralDivCurlParameterization: NN tendencies filtered to spectral space
  - SimpleParameterization: lightweight parameterization for testing
  - NodeToModalParameterization: nodal NN output projected to modal space
  - MixedPrecisionBackward integration (Algorithm 6) for precise BF16/FP32 control
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from pytorch_src.core import primitive_equations
from pytorch_src.core.spherical_harmonic import Grid
from pytorch_src.precision.policy import PrecisionZone
from pytorch_src.precision.zone_cast import zone_cast


class SimpleParameterization(nn.Module):
    """Minimal parameterization: MLP maps state features to tendencies.

    Input: flattened modal state → Output: modal tendency with same shape.
    Designed for testing and simple experiments.
    """

    def __init__(
        self,
        modal_shape: tuple[int, int],
        n_levels: int,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.modal_shape = modal_shape
        self.n_levels = n_levels
        # 4 fields: vorticity, divergence, temperature, log_sp
        n_fields = 4
        flat_size = n_levels * modal_shape[0] * modal_shape[1]
        sp_flat = modal_shape[0] * modal_shape[1]
        self.input_size = n_fields * flat_size - (n_fields - 1) * flat_size + n_fields * flat_size
        # Simpler: just use one flat representation
        self.input_size = (3 * n_levels + 1) * modal_shape[0] * modal_shape[1]
        self.output_size = self.input_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.output_size),
        )
        # Initialize output layer to near-zero so initial tendencies are small
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
        memory: object | None = None,
    ) -> primitive_equations.State:
        # ``memory`` is used by full NeuralGCM parameterizations; ignored here.
        _ = memory
        M, L = self.modal_shape
        nz = self.n_levels

        # Flatten state to vector
        v = state.vorticity.reshape(-1)
        d = state.divergence.reshape(-1)
        t = state.temperature_variation.reshape(-1)
        sp = state.log_surface_pressure.reshape(-1)
        parts = [v, d, t, sp]

        # Append forcing features if available
        if forcings is not None:
            for key in sorted(forcings.keys()):
                parts.append(forcings[key].reshape(-1).to(v.dtype))

        x = torch.cat(parts)

        # Pad or truncate to match expected input size
        if x.shape[0] < self.input_size:
            x = torch.nn.functional.pad(x, (0, self.input_size - x.shape[0]))
        elif x.shape[0] > self.input_size:
            x = x[:self.input_size]

        # Forward pass
        y = self.net(x)

        # Unpack to state fields
        sz_vol = nz * M * L
        sz_sp = M * L
        idx = 0
        vor = y[idx : idx + sz_vol].reshape(nz, M, L)
        idx += sz_vol
        div = y[idx : idx + sz_vol].reshape(nz, M, L)
        idx += sz_vol
        temp = y[idx : idx + sz_vol].reshape(nz, M, L)
        idx += sz_vol
        lsp = y[idx : idx + sz_sp].reshape(1, M, L)

        return primitive_equations.State(
            vorticity=vor,
            divergence=div,
            temperature_variation=temp,
            log_surface_pressure=lsp,
            tracers={k: torch.zeros_like(v_) for k, v_ in state.tracers.items()},
            sim_time=state.sim_time,
        )


class ModalNeuralDivCurlParameterization(nn.Module):
    """Neural parameterization with spectral filtering at Z3→Z1 boundary.

    Architecture:
      1. Modal state → nodal via SH inverse transform
      2. Neural network processes nodal fields (Z3 / BF16)
      3. Nodal tendencies → modal via SH forward transform
      4. Spectral filter to remove aliased modes
      5. Cast to Z1 dtype

    This ensures NN-produced tendencies are spectrally consistent with
    the dynamics core, preventing numerical instability from unfiltered
    high-frequency content.
    """

    def __init__(
        self,
        grid: Grid,
        neural_net: nn.Module,
        n_levels: int,
        filter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_mixed_precision_backward: bool = False,
        fwd_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.grid = grid
        self.neural_net = neural_net
        self.n_levels = n_levels
        self.filter_fn = filter_fn
        self.use_mixed_precision_backward = use_mixed_precision_backward
        self.fwd_dtype = fwd_dtype

    def _state_to_nodal_features(
        self, state: primitive_equations.State
    ) -> torch.Tensor:
        """Convert modal state to nodal feature tensor for NN input.

        Returns: (n_features, lon, lat) where n_features = 3*n_levels + 1
        """
        g = self.grid
        nodal_v = g.to_nodal(state.vorticity)       # (nz, lon, lat)
        nodal_d = g.to_nodal(state.divergence)
        nodal_t = g.to_nodal(state.temperature_variation)
        nodal_sp = g.to_nodal(state.log_surface_pressure)  # (1, lon, lat)
        return torch.cat([nodal_v, nodal_d, nodal_t, nodal_sp], dim=0)

    def _nodal_output_to_state(
        self, y: torch.Tensor, state: primitive_equations.State
    ) -> primitive_equations.State:
        """Convert nodal NN output back to modal State.

        Args:
            y: (n_features, lon, lat) nodal tendency
            state: reference state for shape/tracers
        """
        g = self.grid
        nz = self.n_levels
        idx = 0

        vor_nodal = y[idx : idx + nz]
        idx += nz
        div_nodal = y[idx : idx + nz]
        idx += nz
        temp_nodal = y[idx : idx + nz]
        idx += nz
        sp_nodal = y[idx : idx + 1]

        # Transform to modal
        vor = g.to_modal(vor_nodal)
        div = g.to_modal(div_nodal)
        temp = g.to_modal(temp_nodal)
        sp = g.to_modal(sp_nodal)

        # Apply spectral filter if specified
        if self.filter_fn is not None:
            vor = self.filter_fn(vor)
            div = self.filter_fn(div)
            temp = self.filter_fn(temp)
            sp = self.filter_fn(sp)

        return primitive_equations.State(
            vorticity=vor,
            divergence=div,
            temperature_variation=temp,
            log_surface_pressure=sp,
            tracers={k: torch.zeros_like(v) for k, v in state.tracers.items()},
            sim_time=state.sim_time,
        )

    def forward(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
        memory: object | None = None,
    ) -> primitive_equations.State:
        _ = memory
        # State → nodal features
        x_nodal = self._state_to_nodal_features(state)

        # Append forcing features if available
        if forcings is not None:
            forcing_parts = []
            for key in sorted(forcings.keys()):
                val = forcings[key].to(x_nodal.dtype)
                if val.ndim == 2:
                    val = val.unsqueeze(0)
                forcing_parts.append(val)
            if forcing_parts:
                x_nodal = torch.cat([x_nodal] + forcing_parts, dim=0)

        # Neural network forward (Z3) with optional mixed-precision backward
        if self.use_mixed_precision_backward and self.training:
            from pytorch_src.training.optimization import MixedPrecisionBackward
            y_nodal = MixedPrecisionBackward.apply(
                self.neural_net, x_nodal, self.fwd_dtype
            )
        else:
            y_nodal = self.neural_net(x_nodal)

        # Nodal → modal with spectral filtering
        return self._nodal_output_to_state(y_nodal, state)


class JaxAlignedDivCurlParameterization(nn.Module):
    """JAX-aligned physics parameterization with full 2365-feature pipeline.

    Replaces ModalNeuralDivCurlParameterization's simple 97-feature
    _state_to_nodal_features() with the complete CombinedFeatures pipeline
    matching JAX DivCurlNeuralParameterization.

    Forward flow:
      1. Convert State to dict
      2. CombinedFeatures → 2365-entry feature dict
      3. ShiftAndNormalize → InverseLevelScale → SoftClip transforms
      4. Pack features (sorted key order) → (2365, lon, lat)
      5. ForwardTower(physics_epd) → (192, lon, lat)
      6. Unpack → {u(32), v(32), temp(32), sh(32), cl(32), ci(32)}
      7. ToModalWithDivCurl → modal tendency State

    Args:
        grid: spherical harmonic Grid.
        neural_net: ForwardTower wrapping the physics EPD.
        feature_extractor: CombinedFeatures module producing 2365 features.
        feature_transform: SequentialTransform (normalize + scale + clip).
        n_levels: number of sigma levels (32).
        filter_fn: optional spectral filter.
        output_keys: ordered output field keys (sorted alphabetically).
        output_sizes: number of levels per output key.
    """

    # Default output fields (alphabetically sorted, matching 192 = 6×32)
    DEFAULT_OUTPUT_KEYS = [
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "temperature_variation",
        "u",
        "v",
    ]

    # Must match ``advance_fields`` tracer subset in ``jax_checkpoint_loader``:
    # missing keys shrink the packed EPD width (e.g. synthetic ICs often only
    # have ``specific_humidity`` → 320 channels short vs 2365).
    _EPD_TRACER_KEYS: tuple[str, ...] = (
        "specific_humidity",
        "specific_cloud_liquid_water_content",
        "specific_cloud_ice_water_content",
    )

    def __init__(
        self,
        grid: Grid,
        neural_net: nn.Module,
        feature_extractor: nn.Module,
        feature_transform: nn.Module,
        tendency_transform: Optional[nn.Module] = None,
        n_levels: int = 32,
        filter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_keys: Optional[list[str]] = None,
        output_sizes: Optional[list[int]] = None,
    ):
        super().__init__()
        self.grid = grid
        self.neural_net = neural_net
        self.feature_extractor = feature_extractor
        self.feature_transform = feature_transform
        self.tendency_transform = tendency_transform
        self.n_levels = n_levels
        self.filter_fn = filter_fn
        self.output_keys = output_keys or self.DEFAULT_OUTPUT_KEYS
        self.output_sizes = output_sizes or [n_levels] * len(self.output_keys)

    def _state_to_dict(
        self, state: primitive_equations.State,
    ) -> dict[str, torch.Tensor]:
        """Convert primitive_equations.State to flat dict for feature modules."""
        d: dict[str, torch.Tensor] = {
            "vorticity": state.vorticity,
            "divergence": state.divergence,
            "temperature_variation": state.temperature_variation,
            "log_surface_pressure": state.log_surface_pressure,
        }
        tracers = dict(state.tracers) if state.tracers else {}
        tmpl = state.temperature_variation
        for tk in self._EPD_TRACER_KEYS:
            if tk not in tracers:
                tracers[tk] = torch.zeros_like(tmpl)
        d["tracers"] = tracers
        if state.sim_time is not None:
            device = state.vorticity.device
            d["sim_time"] = torch.tensor(
                [state.sim_time], device=device, dtype=torch.float32,
            )
        return d

    def _memory_to_dict(
        self, memory,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Convert memory state to flat dict if available."""
        if memory is None:
            return None
        if isinstance(memory, dict):
            return memory
        if isinstance(memory, primitive_equations.State):
            return self._state_to_dict(memory)
        # Callers may pass a State-like object that is not our dataclass
        # (e.g. another module's wrapper); avoid silently dropping memory.
        if hasattr(memory, "vorticity") and hasattr(memory, "divergence"):
            st = primitive_equations.State(
                vorticity=memory.vorticity,
                divergence=memory.divergence,
                temperature_variation=memory.temperature_variation,
                log_surface_pressure=memory.log_surface_pressure,
                tracers=dict(getattr(memory, "tracers", None) or {}),
                sim_time=getattr(memory, "sim_time", None),
            )
            return self._state_to_dict(st)
        return None

    def _memory_dict_matches_inputs(
        self,
        mem: dict[str, torch.Tensor],
        ref: dict[str, torch.Tensor],
    ) -> bool:
        """Require full prognostic / tracer coverage so ``memory_*`` width matches JAX."""
        if not mem:
            return False
        for k in (
            "vorticity",
            "divergence",
            "temperature_variation",
            "log_surface_pressure",
        ):
            if k not in mem:
                return False
        r_tr = ref.get("tracers")
        m_tr = mem.get("tracers")
        if r_tr:
            if not isinstance(m_tr, dict):
                return False
            if set(r_tr.keys()) - set(m_tr.keys()):
                return False
        return True

    def _unpack_output(
        self, y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split EPD output tensor into named fields."""
        out = {}
        idx = 0
        for key, sz in zip(self.output_keys, self.output_sizes):
            out[key] = y[idx:idx + sz]
            idx += sz
        return out

    def _nodal_tendencies_to_modal_state(
        self,
        tendencies: dict[str, torch.Tensor],
        state: primitive_equations.State,
    ) -> primitive_equations.State:
        """Convert nodal u,v,temp,tracer tendencies to modal State.

        u,v → divergence,vorticity via spectral div/curl.
        Other fields → direct SH transform.
        """
        g = self.grid

        # Extract u, v tendencies (nodal) and compute modal div/curl
        u_nodal = tendencies.get("u")
        v_nodal = tendencies.get("v")

        if u_nodal is not None and v_nodal is not None:
            # u,v are physical velocities in nodal space
            # For spectral div/curl we need cos(lat)*u, cos(lat)*v
            cos_lat = torch.tensor(
                g.cos_lat, dtype=u_nodal.dtype, device=u_nodal.device,
            )
            cu = u_nodal * cos_lat
            cv = v_nodal * cos_lat

            # Nodal → modal
            cu_modal = g.to_modal(cu)
            cv_modal = g.to_modal(cv)

            # Divergence = div(cos_lat * v_field)
            vor = g.curl_cos_lat((cu_modal, cv_modal))
            div = g.div_cos_lat((cu_modal, cv_modal))
        else:
            vor = torch.zeros_like(state.vorticity)
            div = torch.zeros_like(state.divergence)

        # Temperature variation tendency
        temp_nodal = tendencies.get("temperature_variation")
        if temp_nodal is not None:
            temp = g.to_modal(temp_nodal)
        else:
            temp = torch.zeros_like(state.temperature_variation)

        # Log surface pressure: NOT predicted by physics NN
        lsp = torch.zeros_like(state.log_surface_pressure)

        # Tracer tendencies
        tracers = {}
        for tk in state.tracers:
            if tk in tendencies:
                tracers[tk] = g.to_modal(tendencies[tk])
            else:
                tracers[tk] = torch.zeros_like(state.tracers[tk])

        # Apply spectral filter
        if self.filter_fn is not None:
            vor = self.filter_fn(vor)
            div = self.filter_fn(div)
            temp = self.filter_fn(temp)
            for tk in tracers:
                tracers[tk] = self.filter_fn(tracers[tk])

        return primitive_equations.State(
            vorticity=vor,
            divergence=div,
            temperature_variation=temp,
            log_surface_pressure=lsp,
            tracers=tracers,
            sim_time=state.sim_time,
        )

    def forward(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
        memory=None,
    ) -> primitive_equations.State:
        # 1. Convert state/memory to dict
        inputs = self._state_to_dict(state)
        memory_dict = self._memory_to_dict(memory)
        # JAX-trained EPD expects ``memory_*`` channels; lag-0 = inputs when
        # memory is missing or a partial dict (would shrink packed width).
        if memory_dict is None or not self._memory_dict_matches_inputs(
            memory_dict, inputs
        ):
            memory_dict = inputs

        # 2. Extract features via CombinedFeatures → 2365-entry dict
        features = self.feature_extractor(
            inputs,
            memory=memory_dict,
            forcing=forcings,
        )

        # 3. Apply feature transforms (normalize + level scale + clip)
        features = self.feature_transform(features)

        # 4. Pack features in sorted key order → (2365, lon, lat)
        from pytorch_src.neural.features import pack_features
        x = pack_features(features)

        # 5. Physics EPD → (192, lon, lat)
        y = self.neural_net(x)

        # 6. Unpack output → {field_name: (n_levels, lon, lat)}
        tendencies = self._unpack_output(y)

        # 6.5 Apply JAX-aligned tendency output transforms.
        if self.tendency_transform is not None:
            tendencies = self.tendency_transform(tendencies)

        # 7. Convert nodal tendencies to modal State
        return self._nodal_tendencies_to_modal_state(tendencies, state)
