# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Atmospheric model assembly — PyTorch implementation.

Provides AtmosphericModel: assembles dynamics core, neural parameterization,
encoder/decoder, stochastic physics, and conservation fixers into a complete
model that operates on dict states.

Architecture:
  AtmosphericModel is a *facade* that owns the encoder, decoder, and
  stochastic step while delegating the core dynamics–physics coupling to
  an internal ``NeuralGCMModel`` (from api.py).  This avoids duplicating
  the IMEX-SIL3 inner loop and keeps the coupling logic in one place.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from tornado_gcm.core import coordinate_systems
from tornado_gcm.core import filtering
from tornado_gcm.core import primitive_equations as pe
from tornado_gcm.core import time_integration
from tornado_gcm.model.api import ModelConfig, NeuralGCMModel, VectorizedModel  # noqa: F401
from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.zone_cast import zone_cast
from tornado_gcm.typing import ModelState, RandomnessState
from tornado_gcm.units import SimUnits


@dataclasses.dataclass
class AtmosphereConfig:
    """Configuration for an atmospheric model.

    Attributes:
        coords: coordinate system.
        physics_specs: physical constants and units.
        dt: nondimensional time step (outer physics step).
        inner_steps: substeps between outer saves.
        dycore_substeps: dynamics inner steps per physics step.
            The inner time step is ``dt / dycore_substeps``.
            Paper default: 8 (30 min / 8 = 3.75 min).
        filter_attenuation: exponential filter attenuation.
        filter_order: exponential filter sharpness.
        num_substeps: physics sub-steps per inner step.
    """
    coords: coordinate_systems.CoordinateSystem
    physics_specs: SimUnits
    dt: float = 1.0
    inner_steps: int = 1
    dycore_substeps: int = 8
    filter_attenuation: float = 16.0
    filter_order: int = 18
    num_substeps: int = 1
    precision_policy: PrecisionPolicy = dataclasses.field(
        default_factory=PrecisionPolicy,
    )


class AtmosphericModel(nn.Module):
    """Complete atmospheric model assembling all components.

    Delegates the core dynamics–physics coupling step to an internal
    ``NeuralGCMModel`` from api.py, ensuring that the IMEX-SIL3 inner
    loop, spectral filtering, and conservation fix logic is maintained
    in exactly one place.

    AtmosphericModel adds:
      - encoder (WeatherBench → primitive state)
      - decoder (primitive state → WeatherBench)
      - stochastic physics sub-stepping
      - encode / decode / assimilate / observe convenience methods

    The model follows the PZHA precision architecture:
      Z0: coordinate precomputation (FP64)
      Z1: dynamics core (TF32)
      Z2: conservation fixers (FP64)
      Z3: neural networks (BF16)
    """

    def __init__(
        self,
        config: AtmosphereConfig,
        dynamics_equation: Optional[time_integration.ImplicitExplicitODE] = None,
        neural_equation: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        stochastic_step: Optional[nn.Module] = None,
        conservation_fixer: Optional[nn.Module] = None,
        diagnostics_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # ── Build internal NeuralGCMModel as step engine ──────────────
        # Convert AtmosphereConfig → ModelConfig for the core model.
        core_config = ModelConfig(
            coords=config.coords,
            dt=config.dt,
            inner_steps=config.inner_steps,
            dycore_substeps=config.dycore_substeps,
            physics_specs=None,  # dynamics_equation passed directly
            precision_policy=config.precision_policy,
            filter_attenuation=config.filter_attenuation,
            filter_order=config.filter_order,
            use_dycore=False,  # We supply dycore_equation below
        )
        self._core = NeuralGCMModel(
            config=core_config,
            parameterization=None,  # tendencies come via neural_equation
            conservation_fixer=conservation_fixer,
            encoder=encoder,
            decoder=decoder,
            stochastic_step=stochastic_step,
            diagnostics_fn=diagnostics_fn,
        )
        # Inject pre-built dynamics equation (if provided) directly,
        # bypassing NeuralGCMModel's own physics_specs-based builder.
        if dynamics_equation is not None:
            self._core._dycore_equation = dynamics_equation
            self._core._inner_dt = config.dt / config.dycore_substeps

        # Wire neural_equation as parameterization wrapper.
        # NeuralGCMModel expects a parameterization(state, forcings=...)
        # returning a State tendency.  neural_equation exposes
        # .explicit_terms(state) → State.  Wrap it.
        if neural_equation is not None:
            self._neural_equation = neural_equation
            self._core.parameterization = _ExplicitODEWrapper(neural_equation)
        else:
            self._neural_equation = None

        self.policy = config.precision_policy

    # ── Encode / Decode ───────────────────────────────────────────────

    def encode(
        self,
        inputs: Dict[str, torch.Tensor],
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> pe.State:
        """Encode input data to model state."""
        if self.encoder is not None:
            try:
                return self.encoder(inputs, forcing=forcing)
            except TypeError:
                return self.encoder(inputs)
        raise ValueError("No encoder configured")

    def decode(
        self,
        state: pe.State,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode model state to output data."""
        if self.decoder is not None:
            try:
                return self.decoder(state, forcing=forcing)
            except TypeError:
                return self.decoder(state)
        raise ValueError("No decoder configured")

    # ── Step (delegated) ──────────────────────────────────────────────

    def step(
        self,
        state: pe.State,
        forcings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> pe.State:
        """Single model time step — delegated to internal NeuralGCMModel.

        See ``NeuralGCMModel.step`` for the full algorithm.
        """
        return self._core.step(state, forcings=forcings)

    def step_model_state(
        self,
        model_state: ModelState[pe.State],
        forcings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ModelState[pe.State]:
        """Full ModelState step — delegated to internal NeuralGCMModel."""
        return self._core.step_model_state(model_state, forcings=forcings)

    def initialize_model_state(
        self,
        state: pe.State,
        **kwargs,
    ) -> ModelState[pe.State]:
        """Create initial ModelState."""
        return self._core.initialize_model_state(state, **kwargs)

    # ── Forward (rollout) ─────────────────────────────────────────────

    def forward(
        self,
        initial_state: pe.State | ModelState,
        outer_steps: int,
        inner_steps: Optional[int] = None,
        forcings: Optional[Dict[str, torch.Tensor]] = None,
        post_process_fn: Optional[Callable] = None,
        checkpoint_outer_steps: bool = False,
    ) -> tuple[pe.State | ModelState, list]:
        """Run model for outer_steps * inner_steps.

        Delegates to ``NeuralGCMModel.forward``.

        Returns:
            (final_state, trajectory) where trajectory has outer_steps entries.
        """
        return self._core(
            initial_state,
            outer_steps,
            inner_steps=inner_steps,
            forcings=forcings,
            post_process_fn=post_process_fn,
            checkpoint_outer_steps=checkpoint_outer_steps,
        )

    # ── Assimilate / Observe ──────────────────────────────────────────

    def assimilate(
        self,
        inputs: Dict[str, torch.Tensor],
        forcings: Optional[Dict[str, torch.Tensor]] = None,
        state: Optional[pe.State] = None,
        alpha: float = 1.0,
    ) -> pe.State:
        """Assimilate observations into model state.

        When an encoder is available, encodes ``inputs`` and optionally
        blends with an existing ``state``.
        """
        if self.encoder is not None:
            encoded = self.encode(inputs, forcing=forcings)
            if state is not None and alpha < 1.0:
                return state * (1.0 - alpha) + encoded * alpha
            return encoded
        if state is not None and isinstance(inputs, pe.State):
            return state * (1.0 - alpha) + inputs * alpha
        raise ValueError(
            "No encoder configured and inputs are not a State."
        )

    def observe(
        self,
        state: pe.State,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate observations from model state (alias for decode)."""
        return self.decode(state, forcing)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Wrap ExplicitODE as a parameterization callable
# ═══════════════════════════════════════════════════════════════════════════

class _ExplicitODEWrapper(nn.Module):
    """Adapt an ``ExplicitODE`` nn.Module to the parameterization interface.

    ``NeuralGCMModel`` expects ``parameterization(state, forcings=...)``
    returning a State tendency.  ``neural_equation`` in AtmosphericModel
    exposes ``.explicit_terms(state)`` instead.  This thin wrapper bridges
    the two.
    """

    def __init__(self, equation: nn.Module):
        super().__init__()
        self.equation = equation

    def forward(
        self,
        state: pe.State,
        forcings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> pe.State:
        return self.equation.explicit_terms(state)
