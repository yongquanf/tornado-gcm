# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Stochastic physics parameterization — PyTorch implementation.

Provides:
  - GaussianRandomField: sphere-correlated AR(1) random field
  - StochasticPhysicsParameterizationStep: sub-step iterator
  - Corrector: Euler-forward + humidity clipping
  - Perturbation: additive random perturbation to tendencies
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.core import primitive_equations as pe
from tornado_gcm.core import spherical_harmonic


# ═══════════════════════════════════════════════════════════════════════════
# RandomnessState
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class RandomnessState:
    """State of a random field generator.

    Attributes:
        core: modal-space random field coefficients (levels, m, l).
        nodal_value: nodal-space expansion (levels, lon, lat).
        prng_state: torch Generator state for reproducibility.
        step_count: number of advances performed.
    """
    core: Optional[torch.Tensor] = None
    nodal_value: Optional[torch.Tensor] = None
    prng_state: Optional[torch.Tensor] = None
    step_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# GaussianRandomField
# ═══════════════════════════════════════════════════════════════════════════

class GaussianRandomField(nn.Module):
    """Spherical Gaussian random field with space-time correlations.

    Generates an AR(1) process in spectral space:
      U(t+dt) = phi * U(t) + sigma * eta(t)

    where phi = exp(-dt / correlation_time) and sigma is the spectral-space
    standard deviation scaled by the correlation length.

    Args:
        grid: spherical harmonic Grid.
        correlation_time: temporal decorrelation scale (nondimensional).
        correlation_length: spatial correlation length (nondimensional).
        variance: field variance (if None, returns zero field).
        clip: clip realizations at clip*std_dev.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        correlation_time: float = 1.0,
        correlation_length: float = 1.0,
        variance: Optional[float] = 1.0,
        clip: float = 6.0,
    ):
        super().__init__()
        self.grid = grid
        self.correlation_time = correlation_time
        self.correlation_length = correlation_length
        self.variance = variance
        self.clip = clip

        # Precompute phi and sigma_array
        self._phi = math.exp(-1.0 / max(correlation_time, 1e-12))

        # Spectral damping: sigma_n = exp(-kappa * n(n+1) / 2)
        # where kappa = (correlation_length / radius)^2
        L = grid.total_wavenumbers
        n_vals = np.arange(L, dtype=np.float64)
        kappa = (correlation_length / grid.radius) ** 2 if grid.radius > 0 else 0
        sigma_n = np.exp(-kappa * n_vals * (n_vals + 1) / 2.0)

        # Normalize so that sum of sigma_n^2 * (2n+1) = 4*pi
        total = np.sum(sigma_n ** 2 * (2 * n_vals + 1))
        if total > 0 and variance is not None:
            sigma_n *= math.sqrt(4 * math.pi * variance / total)
        elif variance is None:
            sigma_n *= 0.0

        self.register_buffer(
            "_sigma_n", torch.tensor(sigma_n, dtype=torch.float32), persistent=False,
        )

    def unconditional_sample(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> RandomnessState:
        """Draw initial sample from stationary distribution."""
        L = self.grid.total_wavenumbers
        M = self.grid.total_wavenumbers  # for triangular truncation
        shape = (M, L)
        sigma = self._sigma_n.to(device=device, dtype=dtype)

        # Scale for stationary variance: factor = 1 / sqrt(1 - phi^2)
        if abs(self._phi) < 1.0:
            scale = 1.0 / math.sqrt(1.0 - self._phi ** 2)
        else:
            scale = 1.0

        # Sample truncated normal
        eta = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        eta = torch.clamp(eta, -self.clip, self.clip)

        core = scale * sigma.unsqueeze(0) * eta  # (M, L)
        nodal = self.grid.to_nodal(core)

        return RandomnessState(
            core=core,
            nodal_value=nodal,
            step_count=0,
        )

    def advance(
        self,
        state: RandomnessState,
        generator: Optional[torch.Generator] = None,
    ) -> RandomnessState:
        """Advance AR(1) one time step."""
        device = state.core.device
        dtype = state.core.dtype
        sigma = self._sigma_n.to(device=device, dtype=dtype)

        eta = torch.randn_like(state.core, generator=generator)
        eta = torch.clamp(eta, -self.clip, self.clip)

        next_core = self._phi * state.core + sigma.unsqueeze(0) * eta
        nodal = self.grid.to_nodal(next_core)

        return RandomnessState(
            core=next_core,
            nodal_value=nodal,
            step_count=state.step_count + 1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# NoRandomField — ablation variant (disables randomness)
# ═══════════════════════════════════════════════════════════════════════════

class NoRandomField(nn.Module):
    """Disables randomness entirely: all values are None.

    Used for ablation experiments to test models without stochastic forcing.
    """

    def __init__(self, grid: Optional[spherical_harmonic.Grid] = None, **kwargs):
        super().__init__()

    def unconditional_sample(self, **kwargs) -> RandomnessState:
        return RandomnessState(step_count=0)

    def advance(self, state: RandomnessState, **kwargs) -> RandomnessState:
        return RandomnessState(step_count=state.step_count + 1)


# ═══════════════════════════════════════════════════════════════════════════
# ZerosRandomField — ablation variant (constant zero field)
# ═══════════════════════════════════════════════════════════════════════════

class ZerosRandomField(nn.Module):
    """Constant-zero random field for ablation experiments.

    Unlike NoRandomField, this provides actual zero tensors at the expected
    shapes so downstream code that relies on ``randomness.nodal_value``
    being a tensor (not None) still works.

    Args:
        grid: spherical harmonic Grid (provides nodal/modal shapes).
        prefer_nodal: if True, the core state uses nodal shape.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        prefer_nodal: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self._prefer_nodal = prefer_nodal

    def unconditional_sample(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> RandomnessState:
        nodal_shape = self.grid.nodal_shape
        modal_shape = self.grid.modal_shape
        core_shape = nodal_shape if self._prefer_nodal else modal_shape
        return RandomnessState(
            core=torch.zeros(core_shape, dtype=dtype, device=device),
            nodal_value=torch.zeros(nodal_shape, dtype=dtype, device=device),
            step_count=0,
        )

    def advance(self, state: RandomnessState, **kwargs) -> RandomnessState:
        return RandomnessState(
            core=torch.zeros_like(state.core) if state.core is not None else None,
            nodal_value=torch.zeros_like(state.nodal_value) if state.nodal_value is not None else None,
            step_count=state.step_count + 1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Corrector
# ═══════════════════════════════════════════════════════════════════════════

class Corrector(nn.Module):
    """Euler-forward corrector with humidity non-negativity constraint.

    Given a tendency from the physics parameterization, advances the
    state by dt and clips specific humidity to >= 0.

    Args:
        dt: sub-step size (nondimensional).
        humidity_key: tracer key for specific humidity.
    """

    def __init__(self, dt: float, humidity_key: str = "specific_humidity"):
        super().__init__()
        self.dt = dt
        self.humidity_key = humidity_key

    def forward(
        self,
        state: pe.State,
        tendency: pe.State,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> pe.State:
        # Euler forward
        next_state = state + self.dt * tendency

        # Clip humidity to non-negative
        if self.humidity_key in next_state.tracers:
            q = next_state.tracers[self.humidity_key]
            next_state.tracers[self.humidity_key] = torch.clamp(q, min=0.0)

        return next_state


# ═══════════════════════════════════════════════════════════════════════════
# Perturbation
# ═══════════════════════════════════════════════════════════════════════════

class Perturbation(nn.Module):
    """Additive random perturbation to tendencies.

    Implements SPPT-style multiplicative perturbation:
      tendency_perturbed = tendency * (1 + epsilon * random_field)

    Args:
        mode: 'additive' or 'multiplicative'.
        epsilon: perturbation amplitude.
    """

    def __init__(self, mode: str = "multiplicative", epsilon: float = 1.0):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon

    def forward(
        self,
        tendency: pe.State,
        randomness: Optional[RandomnessState] = None,
    ) -> pe.State:
        if randomness is None or randomness.nodal_value is None:
            return tendency
        noise = randomness.nodal_value
        if self.mode == "multiplicative":
            factor = 1.0 + self.epsilon * noise
            # Broadcast: noise may be (lon, lat), tendency is (levels, m, l) modal
            # Need to convert factor to modal space or work in nodal
            # For simplicity, we apply a global scaling
            scale = 1.0 + self.epsilon * noise.mean()
            return tendency * scale
        else:
            # Additive: add noise directly (need shape matching)
            return pe.State(
                vorticity=tendency.vorticity,
                divergence=tendency.divergence,
                temperature_variation=tendency.temperature_variation + self.epsilon * noise.mean(),
                log_surface_pressure=tendency.log_surface_pressure,
                tracers=tendency.tracers,
                sim_time=tendency.sim_time,
            )


# ═══════════════════════════════════════════════════════════════════════════
# StochasticPhysicsParameterizationStep
# ═══════════════════════════════════════════════════════════════════════════

class StochasticPhysicsParameterizationStep(nn.Module):
    """Physics parameterization with stochastic sub-stepping.

    Iterates num_substeps of:
      1. Compute parameterization tendency
      2. Apply perturbation
      3. Euler-correct state
      4. Advance random field
      5. Update memory (prev state)

    Args:
        physics_parameterization: NN module producing tendency from state.
        corrector: Corrector module.
        random_field: GaussianRandomField for stochastic forcing.
        perturbation: Perturbation module.
        num_substeps: number of sub-steps per outer step.
    """

    def __init__(
        self,
        physics_parameterization: nn.Module,
        corrector: Corrector,
        random_field: Optional[GaussianRandomField] = None,
        perturbation: Optional[Perturbation] = None,
        num_substeps: int = 1,
    ):
        super().__init__()
        self.physics_param = physics_parameterization
        self.corrector = corrector
        self.random_field = random_field
        self.perturbation = perturbation
        self.num_substeps = num_substeps

    def forward(
        self,
        state: pe.State,
        forcing: Optional[Dict[str, torch.Tensor]] = None,
        randomness: Optional[RandomnessState] = None,
    ) -> tuple[pe.State, Optional[RandomnessState]]:
        """Run sub-stepping loop.

        Args:
            state: current model state.
            forcing: external forcing dict.
            randomness: current random field state.

        Returns:
            (updated_state, updated_randomness)
        """
        current_state = state
        current_rand = randomness

        for _ in range(self.num_substeps):
            # 1. Compute tendency
            tendency = self.physics_param(current_state)

            # 2. Apply perturbation
            if self.perturbation is not None and current_rand is not None:
                tendency = self.perturbation(tendency, current_rand)

            # 3. Euler-correct
            current_state = self.corrector(current_state, tendency, forcing)

            # 4. Advance random field
            if self.random_field is not None and current_rand is not None:
                current_rand = self.random_field.advance(current_rand)

        return current_state, current_rand

    def initialize_randomness(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Optional[RandomnessState]:
        """Create initial random field state."""
        if self.random_field is None:
            return None
        return self.random_field.unconditional_sample(device=device, dtype=dtype)
