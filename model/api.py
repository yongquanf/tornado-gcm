"""NeuralGCM model API — PyTorch implementation.

Provides the main model class that combines:
  - Spectral dynamics core (Z0/Z1)
  - Neural parameterizations (Z3)
  - Conservation fixers (Z2)
  - PZHA precision policy

Coupling architecture (see paper Fig. 1 / Extended Data Fig. 1):
  Outer loop: every ``physics_timestep`` (e.g. 30 min)
    1. Learned_phy(X_sigma) → constant physics tendencies (Z3)
  Inner loop: ``dycore_substeps`` times (e.g. 8 × 3.75 min = 30 min)
    2. Dyn_core.explicit_terms(X_sigma) + physics tendencies  (Z1)
    3. IMEX-RK3-SIL3 integration step
    4. Spectral filter
  After inner loop:
    5. Conservation fix (Z2)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from torch.profiler import record_function

from pytorch_src.core import coordinate_systems
from pytorch_src.core import filtering
from pytorch_src.core import primitive_equations
from pytorch_src.core import spherical_harmonic
from pytorch_src.core import time_integration
from pytorch_src.precision.monitor import PrecisionMonitor
from pytorch_src.precision.policy import PrecisionPolicy, PrecisionZone
from pytorch_src.precision.zone_cast import zone_cast
from pytorch_src.typing import ModelState, RandomnessState


logger = logging.getLogger(__name__)

# Type alias: forcing_fn(forcing_data, sim_time) → dict[str, Tensor]
# Matches NeuralGCM legacy forcing_fn(forcing_data, sim_time) protocol.
ForcingFn = Callable[
    [dict[str, torch.Tensor], float], dict[str, torch.Tensor]
]


def _physics_memory_ok_for_epd(
    memory: primitive_equations.State | dict[str, Any] | None,
    state: primitive_equations.State,
) -> bool:
    """True only when ``memory`` can supply the full ``memory_*`` feature block.

    Partial dicts (e.g. forcings-only blobs) are not ``None`` but would shrink
    the packed EPD width below the trained 2365 channels.
    """
    if memory is None:
        return False
    if isinstance(memory, primitive_equations.State):
        return True
    if isinstance(memory, dict):
        if not memory:
            return False
        required = (
            "vorticity",
            "divergence",
            "temperature_variation",
            "log_surface_pressure",
        )
        if not all(k in memory for k in required):
            return False
        if state.tracers:
            m_tr = memory.get("tracers")
            if not isinstance(m_tr, dict):
                return False
            if set(state.tracers.keys()) - set(m_tr.keys()):
                return False
        return True
    return True


@dataclasses.dataclass
class ModelConfig:
    """Configuration for NeuralGCM model.

    Attributes:
        coords: coordinate system (horizontal + vertical).
        dt: time step size (nondimensional) for the *outer* physics step.
        inner_steps: number of outer steps saved per ``forward()`` call.
        dycore_substeps: number of dynamics inner steps per outer physics
            step.  The inner time step is ``dt / dycore_substeps``.
            Paper default: 8 (30 min / 8 = 3.75 min).
        physics_specs: optional SimUnits for PrimitiveEquationsSigma.
        reference_temperature: T_ref profile (n_levels,).  Required when
            ``physics_specs`` is given.
        orography: modal orography tensor.  If ``None``, a zero field is used.
        precision_policy: PZHA precision configuration.
        filter_attenuation: exponential filter attenuation.
        filter_order: exponential filter order.
        diffusion_tau: horizontal diffusion timescale.
        use_dycore: if True *and* ``physics_specs`` is provided, build a
            full dynamics core and couple it with the neural parameterization
            via IMEX-RK3-SIL3.  When False, the legacy simple-Euler path is
            used (fast tests / lightweight runs without physics_specs).
    """

    coords: coordinate_systems.CoordinateSystem
    dt: float = 1.0
    inner_steps: int = 1
    dycore_substeps: int = 8
    physics_specs: Any = None
    reference_temperature: Optional[np.ndarray] = None
    orography: Optional[torch.Tensor] = None
    precision_policy: PrecisionPolicy = dataclasses.field(
        default_factory=PrecisionPolicy
    )
    filter_attenuation: float = 16.0
    filter_order: int = 18
    filter_cutoff: float = 0.0
    stability_filter_attenuation: Optional[float] = None
    stability_filter_order: int = 10
    stability_filter_cutoff: float = 0.4
    diffusion_tau: float = 0.01
    use_dycore: bool = True
    checkpoint_inner_steps: bool = False
    # Match JAX MoistPrimitiveEquationsWithCloudMoisture default behavior.
    # Set to None to force a dry dycore.
    humidity_key: str | None = "specific_humidity"
    # Optional cloud condensate tracer keys used in virtual-temperature adjustment.
    cloud_keys: tuple[str, ...] | None = None


class NeuralGCMModel(nn.Module):
    """Main NeuralGCM model combining dynamics and neural parameterizations.

    Coupling architecture
    =====================
    ``outer_step(state)``  (one physics time step, e.g. 30 min):

        1. ``Learned_phy(state)`` → constant physics tendency   [Z3]
        2. ``compose_equations(dycore, physics_tendency)``       [Z1]
        3. for i in range(dycore_substeps):                      [Z1]
              state = imex_rk_sil3_step(combined, inner_dt)(state)
              state = spectral_filter(state)
        4. ``conservation_fix(state, state_before)``             [Z2]

    When no dynamics core is configured (``use_dycore=False`` or no
    ``physics_specs``), falls back to a simple Euler step:
        state' = state + dt * NN_tendency(state)

    PZHA zone boundaries:
      - Z3 → Z1: neural outputs cast to Z1 before adding to dynamics
      - Z1 → Z2: after inner loop, state cast to FP64 for conservation fix
      - Z2 → Z1: fixed state cast back for next step
    """

    def __init__(
        self,
        config: ModelConfig,
        parameterization: nn.Module | None = None,
        conservation_fixer: nn.Module | None = None,
        monitor: PrecisionMonitor | None = None,
        forcing_fn: ForcingFn | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        stochastic_step: nn.Module | None = None,
        perturbation_fn: nn.Module | None = None,
        random_field: nn.Module | None = None,
        diagnostics_fn: Callable | None = None,
    ):
        super().__init__()
        self.config = config
        self.parameterization = parameterization
        self.conservation_fixer = conservation_fixer
        self.monitor = monitor
        self.forcing_fn = forcing_fn
        self._fixer_cadence: int = 1   # apply fixer every N steps (1=every step)
        self._step_count: int = 0
        self._sanitize_inner_enabled: bool = True
        self._sanitize_inner_clamp_abs: float = 1.0e12
        self.encoder = encoder
        self.decoder = decoder
        self.stochastic_step = stochastic_step
        self.perturbation_fn = perturbation_fn
        self.random_field = random_field
        self.diagnostics_fn = diagnostics_fn

        # Build spectral filters (dycore + optional stability)
        grid = config.coords.horizontal
        self.exp_filter = filtering.exponential_filter(
            grid,
            attenuation=config.filter_attenuation,
            order=config.filter_order,
            cutoff=config.filter_cutoff,
        )
        self.stability_filter: Callable | None = None
        if config.stability_filter_attenuation is not None:
            self.stability_filter = filtering.exponential_filter(
                grid,
                attenuation=config.stability_filter_attenuation,
                order=config.stability_filter_order,
                cutoff=config.stability_filter_cutoff,
            )
        self.policy = config.precision_policy

        # ── Nanoscope integration ────────────────────────────────────
        # Set via attach_nanoscope() / detach_nanoscope() at runtime.
        self._nanoscope_ctx: Any | None = None  # NanoscopeContext or None

        # ── O9-4 Diagnostic caching ──────────────────────────────────
        self._cached_diagnostics: dict[str, Any] = {}
        self._diagnostics_enabled: bool = False

        # ── Build dynamics core (if configured) ───────────────────────
        self._dycore_equation: time_integration.ImplicitExplicitODE | None = None
        self._inner_step_fn: time_integration.TimeStepFn | None = None

        if config.use_dycore and config.physics_specs is not None:
            ref_temp = config.reference_temperature
            if ref_temp is None:
                ref_temp = np.full(config.coords.vertical.layers, 250.0)

            orography = config.orography
            if orography is None:
                modal_shape = config.coords.modal_shape
                orography = torch.zeros(modal_shape[1:], dtype=torch.float32)

            dycore = primitive_equations.PrimitiveEquationsSigma(
                reference_temperature=ref_temp,
                orography=orography,
                coords=config.coords,
                physics_specs=config.physics_specs,
                humidity_key=config.humidity_key,
                cloud_keys=config.cloud_keys,
            )
            self._dycore_equation = dycore.as_imex_ode()

            # Build inner step function (IMEX-RK3-SIL3 + filter)
            inner_dt = config.dt / config.dycore_substeps
            raw_step = time_integration.imex_rk_sil3(
                self._dycore_equation,  # placeholder; replaced each outer step
                inner_dt,
            )
            # We cache inner_dt; the actual step_fn is rebuilt per outer step
            # because physics tendencies change each outer step.
            self._inner_dt = inner_dt

    # ── Physics tendency (Z3 → Z1) ───────────────────────────────────

    def _apply_parameterization(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
        memory: primitive_equations.State | dict[str, torch.Tensor] | None = None,
    ) -> primitive_equations.State:
        """Apply neural parameterization in Z3 precision."""
        if self.parameterization is None:
            return state.tree_map(torch.zeros_like)

        z3_dtype = self.policy.compute_dtype(PrecisionZone.Z3_NEURAL_NETWORK)
        z1_dtype = self.policy.compute_dtype(PrecisionZone.Z1_DYNAMICS_CORE)

        # Match input dtype to NN weight dtype to avoid mat1/mat2 mismatch.
        # When Z3 policy says BF16 but weights are FP32 (e.g. no weight
        # casting applied), use the actual weight dtype for the forward pass.
        param_dtype = next(self.parameterization.parameters()).dtype
        effective_z3 = param_dtype if param_dtype != z3_dtype else z3_dtype
        z3_state = state.tree_map(lambda t: t.to(effective_z3))

        # Memory can carry dycore-upcast tensors (e.g., float64 from implicit
        # paths). Keep it aligned with NN weight/input dtype to avoid Linear
        # matmul dtype mismatches in feature pipelines.
        # JAX-aligned parameterizations pack ``memory_*`` features; when callers
        # omit ``memory`` (e.g. HPS / single-step probes), use lag-0 = current
        # state so the physics EPD input width matches trained weights (2365).
        # Also reject empty / partial memory dicts (truthy but not a full state).
        z3_memory = memory if _physics_memory_ok_for_epd(memory, state) else z3_state
        if hasattr(z3_memory, "tree_map"):
            z3_memory = z3_memory.tree_map(lambda t: t.to(effective_z3))
        elif isinstance(z3_memory, dict):
            z3_memory = {
                k: (v.to(effective_z3) if isinstance(v, torch.Tensor) else v)
                for k, v in z3_memory.items()
            }

        if self.monitor is not None:
            self.monitor.audit_zone_transfer(
                "Z1_to_Z3", state.vorticity, z3_state.vorticity
            )

        tendency = self.parameterization(z3_state, forcings=forcings, memory=z3_memory)

        tendency_z1 = tendency.tree_map(lambda t: t.to(z1_dtype))

        if self.monitor is not None:
            self.monitor.audit_zone_transfer(
                "Z3_to_Z1", tendency.vorticity, tendency_z1.vorticity
            )

        return tendency_z1

    # ── Conservation fix (Z1 → Z2 → Z1) ──────────────────────────────

    def _apply_conservation_fix(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        """Apply conservation fixer in Z2 precision.

        Two paths:
          - LiteConservationFixer (bypass_upcast=True): operates on FP32
            state directly using sum(dtype=fp64) — no full-tensor upcast.
          - F64ConservationFixer (legacy): full state upcast FP32→FP64,
            fixer in FP64, downcast back.
        """
        if self.conservation_fixer is None:
            return state

        # Z2-Lite path: skip full-tensor upcast/downcast
        if getattr(self.conservation_fixer, "bypass_upcast", False):
            if self.monitor is not None:
                self.monitor.audit_zone_transfer(
                    "Z2_lite", state.vorticity, state.vorticity
                )
            fixed = self.conservation_fixer(state, state_before)
            if self._diagnostics_enabled:
                self._cache_fixer_diagnostics(state, state_before, fixed)
            return fixed

        # Legacy full-upcast path
        z2_dtype = self.policy.compute_dtype(PrecisionZone.Z2_CONSERVATION_FIXER)
        z1_dtype = self.policy.compute_dtype(PrecisionZone.Z1_DYNAMICS_CORE)

        # Z1 → Z2: upcast inputs for fixer
        state_z2 = state.tree_map(lambda t: t.to(z2_dtype))
        before_z2 = state_before.tree_map(lambda t: t.to(z2_dtype))

        if self.monitor is not None:
            self.monitor.audit_zone_transfer(
                "Z1_to_Z2", state.vorticity, state_z2.vorticity
            )

        fixed_z2 = self.conservation_fixer(state_z2, before_z2)

        # Z2 → Z1: downcast result back to dynamics precision
        fixed = fixed_z2.tree_map(lambda t: t.to(z1_dtype))

        if self.monitor is not None:
            self.monitor.audit_zone_transfer(
                "Z2_to_Z1", fixed_z2.vorticity, fixed.vorticity
            )

        # O9-4: Cache fixer diagnostics for deferred output
        if self._diagnostics_enabled:
            self._cache_fixer_diagnostics(state, state_before, fixed)

        return fixed

    # ── O9-4 Diagnostic caching ──────────────────────────────────────

    def set_precision_policy(self, policy: "PrecisionPolicy") -> None:
        """Replace the active precision policy at runtime."""
        self.policy = policy

    def set_fixer(self, fixer: nn.Module | None) -> None:
        """Replace (or disable) the conservation fixer at runtime."""
        self.conservation_fixer = fixer

    def set_fixer_cadence(self, cadence: int) -> None:
        """Set fixer cadence: apply conservation fix every *cadence* steps."""
        if cadence < 1:
            raise ValueError(f"fixer_cadence must be >= 1, got {cadence}")
        self._fixer_cadence = cadence

    def enable_diagnostics(self, enabled: bool = True) -> None:
        """Enable/disable O9-4 diagnostic caching."""
        self._diagnostics_enabled = enabled
        if not enabled:
            self._cached_diagnostics.clear()

    def _cache_fixer_diagnostics(
        self,
        state_post: primitive_equations.State,
        state_pre: primitive_equations.State,
        state_fixed: primitive_equations.State,
    ) -> None:
        """Cache fixer input/output diagnostics for deferred analysis."""
        with torch.no_grad():
            # Conservation drift: energy-like quantity before vs after fix
            vort_drift = (
                state_fixed.vorticity.double() - state_post.vorticity.double()
            ).abs().max().item()
            div_drift = (
                state_fixed.divergence.double() - state_post.divergence.double()
            ).abs().max().item()
            self._cached_diagnostics["last_fixer"] = {
                "vorticity_drift_max": vort_drift,
                "divergence_drift_max": div_drift,
            }

    def get_diagnostics(self) -> dict[str, Any]:
        """Return and clear cached diagnostics (O9-4)."""
        diags = dict(self._cached_diagnostics)
        self._cached_diagnostics.clear()
        return diags

    # ── Spectral filter ───────────────────────────────────────────────

    def _filter_state(
        self, state: primitive_equations.State,
    ) -> primitive_equations.State:
        """Apply spectral filtering."""
        return state.tree_map(self.exp_filter)

    def _sanitize_inner_state(
        self, state: primitive_equations.State,
    ) -> primitive_equations.State:
        """Numerical safeguard after each IMEX inner substep.

        Prevents explosive modal values from overflowing into NaN/Inf during
        subsequent nonlinear tendency evaluations.
        """
        if not self._sanitize_inner_enabled:
            return state

        clamp_abs = self._sanitize_inner_clamp_abs

        def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(t):
                return t
            t = torch.nan_to_num(t, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
            return torch.clamp(t, min=-clamp_abs, max=clamp_abs)

        return state.tree_map(_sanitize_tensor)

    def set_inner_state_safeguard(
        self,
        enabled: bool,
        clamp_abs: float = 1.0e12,
    ) -> None:
        """Enable/disable post-inner-step sanitization.

        This safeguard is useful for robust training runs, but it can mask
        true instability signatures in numerical equivalence diagnostics.
        """
        self._sanitize_inner_enabled = bool(enabled)
        self._sanitize_inner_clamp_abs = float(clamp_abs)

    # ── Nanoscope integration ─────────────────────────────────────────

    def attach_nanoscope(self, ctx: Any) -> None:
        """Attach a ``NanoscopeContext`` to monitor this model's dynamics.

        After calling this, each :meth:`step` call will push state into the
        Nanoscope ring buffer after every inner substep and will propagate any
        ``NanoscopeShutterException`` raised by the instrumented dynamics.

        Args:
            ctx: A ``nanoscope.NanoscopeContext`` instance.  Must already be
                attached (i.e. ``ctx.attach()`` has been called) before passing
                it here, or the model can be used inside a ``with ctx:`` block.
        """
        self._nanoscope_ctx = ctx

    def detach_nanoscope(self) -> None:
        """Detach the current ``NanoscopeContext`` from this model."""
        self._nanoscope_ctx = None

    # ── Inner step builders ───────────────────────────────────────────

    def _build_combined_equation(
        self,
        physics_tendency: primitive_equations.State,
    ) -> time_integration.ImplicitExplicitODE:
        """Build a composed IMEX equation: dycore + constant physics.

        The physics tendency is wrapped as an ``ExplicitODE`` whose
        ``explicit_terms`` always returns the *same constant* tendency
        (computed once per outer step by the neural parameterization).
        The dynamics core supplies the state-dependent explicit *and*
        implicit terms.

        ``compose_equations`` sums all explicit terms (physics constant +
        dycore explicit) while keeping implicit terms from the dycore only.
        """
        physics_eq = time_integration.ExplicitODE.from_functions(
            lambda _state: physics_tendency,
        )
        return time_integration.compose_equations(
            [self._dycore_equation, physics_eq],
        )

    def _build_inner_step_fn(
        self,
        physics_tendency: primitive_equations.State,
    ) -> time_integration.TimeStepFn:
        """Build filtered IMEX inner step for current physics tendency.

        Returns a function ``state → state`` that performs one inner step:
          1. IMEX-RK3-SIL3 advance with ``inner_dt``
          2. Spectral filtering (applied via ``State.tree_map``)
        """
        combined_eq = self._build_combined_equation(physics_tendency)

        # Build step_hook for Nanoscope intermediate-stage observation
        ns_ctx = self._nanoscope_ctx
        step_hook = None
        if ns_ctx is not None:
            def step_hook(stage_index: int, y_stage: "primitive_equations.State") -> None:  # noqa: F821
                try:
                    from nanoscope.neuralgcm_adapter import NeuralGCMStateAdapter
                    flat = NeuralGCMStateAdapter.flatten(y_stage)
                    ns_ctx.snapshot_manager.push_state(
                        step=self._step_count,
                        t=sim_time + stage_index * self._inner_dt,
                        y=flat,
                    )
                except Exception as exc:
                    logger.debug("[Nanoscope] step_hook push_state failed: %s", exc)

        raw_step = time_integration.imex_rk_sil3(
            combined_eq, self._inner_dt, step_hook=step_hook
        )

        # Wrap the tensor-level filters into State-level filters
        # so they work with the time_integration step_with_filters API.
        # JAX applies two sequential filters per inner substep:
        #   1. dycore/ExponentialFilter  (tau=120min, order=3, cutoff=0)
        #   2. stability/ExponentialFilter (tau=4min, order=10, cutoff=0.4)
        dycore_tensor_filter = self.exp_filter
        stability_tensor_filter = self.stability_filter

        def dycore_state_filter(s: primitive_equations.State) -> primitive_equations.State:
            with record_function("Z0_spectral"):
                return s.tree_map(dycore_tensor_filter)

        filters = [time_integration.runge_kutta_step_filter(dycore_state_filter)]

        if stability_tensor_filter is not None:
            def stability_state_filter(s: primitive_equations.State) -> primitive_equations.State:
                with record_function("Z0_stability"):
                    return s.tree_map(stability_tensor_filter)
            filters.append(time_integration.runge_kutta_step_filter(stability_state_filter))

        return time_integration.step_with_filters(raw_step, filters)

    # ── Outer step (coupled) ──────────────────────────────────────────

    def step(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
        memory: primitive_equations.State | dict[str, torch.Tensor] | None = None,
    ) -> primitive_equations.State:
        """One outer physics time step with dynamics–physics coupling.

        When a dynamics core is configured:

        .. code-block:: text

            Trend_phy = Learned_phy(X_sigma(t))          # once
            for _ in range(dycore_substeps):              # 8× by default
                X_sigma = IMEX_SIL3(dycore + Trend_phy, inner_dt)(X_sigma)
                X_sigma = filter(X_sigma)
            X_sigma = conservation_fix(X_sigma)

        Without a dynamics core (legacy / test mode):

        .. code-block:: text

            X_sigma = X_sigma + dt * Learned_phy(X_sigma)
            X_sigma = filter(X_sigma)
            X_sigma = conservation_fix(X_sigma)
        """
        state_before = state

        # 0. Resolve time-dependent forcings via forcing_fn if available
        if self.forcing_fn is not None and forcings is not None:
            sim_time = state.sim_time if state.sim_time is not None else 0.0
            forcings = self.forcing_fn(forcings, sim_time)

        # 1. Neural parameterization tendency (Z3 → Z1)
        with record_function("Z3_neural_net"):
            physics_tendency = self._apply_parameterization(
                state,
                forcings=forcings,
                memory=memory,
            )

        if self._dycore_equation is not None:
            # ── Coupled IMEX path ─────────────────────────────────────
            # Build inner step function with current (constant) physics
            inner_step = self._build_inner_step_fn(physics_tendency)

            # Inner loop: dycore_substeps iterations
            # With checkpoint_inner_steps, each substep recomputes forward
            # during backward pass → O(1) memory instead of O(N).
            sim_time = float(state.sim_time) if state.sim_time is not None else 0.0
            try:
                with record_function("Z1_dycore"):
                    for substep in range(self.config.dycore_substeps):
                        # Push state to Nanoscope ring buffer before each substep
                        ns_ctx = self._nanoscope_ctx
                        if ns_ctx is not None:
                            try:
                                from nanoscope.neuralgcm_adapter import NeuralGCMStateAdapter
                                flat = NeuralGCMStateAdapter.flatten(state)
                                ns_ctx.snapshot_manager.push_state(
                                    step=self._step_count,
                                    t=sim_time + substep * self._inner_dt,
                                    y=flat,
                                )
                            except Exception:
                                pass
                        if self.config.checkpoint_inner_steps and self.training:
                            state = torch_checkpoint.checkpoint(
                                inner_step, state, use_reentrant=False,
                            )
                        else:
                            state = inner_step(state)
                        state = self._sanitize_inner_state(state)
            except Exception as exc:
                # Re-raise NanoscopeShutterException with enriched context
                try:
                    from nanoscope.snapshot import NanoscopeShutterException
                    if isinstance(exc, NanoscopeShutterException):
                        raise NanoscopeShutterException(
                            step=exc.step,
                            fault_op=exc.fault_op,
                            snapshot_path=exc.snapshot_path,
                            t=exc.t,
                        ) from exc
                except ImportError:
                    pass
                raise
        else:
            # ── Legacy simple-Euler path (no dycore) ──────────────────
            with record_function("Z1_dycore"):
                dt = self.config.dt
                state = state + dt * physics_tendency
            with record_function("Z0_spectral"):
                state = self._filter_state(state)

        # 3. Conservation fix (Z2 = FP64), respecting fixer_cadence
        self._step_count += 1
        if self._step_count % self._fixer_cadence == 0:
            with record_function("Z2_physics_bridge"):
                state = self._apply_conservation_fix(state, state_before)

        # 4. Advance sim_time (needed for forcing interpolation & cycle features)
        if state.sim_time is not None:
            state = dataclasses.replace(state, sim_time=state.sim_time + self.config.dt)

        # Precision audit step
        if self.monitor is not None:
            self.monitor.step()

        return state

    # ── ModelState step (memory / diagnostics / randomness) ───────────

    def step_model_state(
        self,
        model_state: ModelState[primitive_equations.State],
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> ModelState[primitive_equations.State]:
        """One outer step operating on the full ModelState.

        Extends :meth:`step` with the stochastic-physics pipeline:

        1. Feed ``memory``, ``diagnostics``, ``randomness`` into features →
           parameterization (if supported).
        2. Core dynamics step (delegated to :meth:`step`).
        3. Apply perturbation to physics tendency (optional).
        4. Stochastic physics sub-stepping (optional).
        5. Advance randomness, update memory = previous state.
        6. Compute new diagnostics from tendency + state.

        Compatible with bare ``State`` callers: if ``model_state`` is a
        ``State`` (not a ``ModelState``), wraps it first.

        Args:
            model_state: full model state with memory/diagnostics/randomness.
            forcings: optional forcing dict.

        Returns:
            Updated ``ModelState`` with new state, memory, diagnostics,
            and advanced randomness.
        """
        state = model_state.state
        memory = model_state.memory
        diagnostics = model_state.diagnostics
        randomness = model_state.randomness

        # 1. Core step (physics tendency + IMEX inner loop + conservation fix)
        next_state = self.step(state, forcings=forcings, memory=memory)

        # 2. Stochastic physics sub-stepping (after dycore, before next step)
        next_randomness = randomness
        if self.stochastic_step is not None:
            rand_input = randomness if randomness.core is not None else None
            next_state, next_randomness_raw = self.stochastic_step(
                next_state, forcing=forcings, randomness=rand_input,
            )
            if next_randomness_raw is not None:
                next_randomness = next_randomness_raw
        elif self.random_field is not None and randomness.core is not None:
            next_randomness = self.random_field.advance(randomness)

        # 3. Memory = previous state snapshot (for lagged NN features next step)
        next_memory = (
            state.tree_map(torch.clone) if memory is not None else None
        )

        # 4. Diagnostics (if diagnostics_fn provided)
        next_diagnostics: dict = {}
        if self.diagnostics_fn is not None:
            next_diagnostics = self.diagnostics_fn(
                ModelState(
                    state=next_state,
                    memory=next_memory,
                    diagnostics=diagnostics,
                    randomness=next_randomness,
                ),
                physics_tendencies=None,
                forcing=forcings,
            )
        elif isinstance(diagnostics, dict):
            next_diagnostics = diagnostics

        return ModelState(
            state=next_state,
            memory=next_memory,
            diagnostics=next_diagnostics,
            randomness=next_randomness,
        )

    def initialize_model_state(
        self,
        state: primitive_equations.State,
        with_memory: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ModelState[primitive_equations.State]:
        """Create an initial ModelState from a bare State.

        Initializes memory (deep tensor copy of state), empty diagnostics, and
        randomness (from ``random_field`` if available).

        Memory must not alias ``state`` tensors: ``step_model_state`` threads
        both into the parameterization; sharing storage can diverge from JAX
        (where memory is a separate carry) and cause subtle in-place hazards.
        """
        memory = state.tree_map(torch.clone) if with_memory else None
        diagnostics: dict = {}
        randomness = RandomnessState()
        if self.random_field is not None:
            dev = device or state.vorticity.device
            randomness_result = self.random_field.unconditional_sample(
                device=dev, dtype=dtype,
            )
            # Adapt from stochastic.RandomnessState to typing.RandomnessState
            randomness = RandomnessState(
                core=randomness_result.core,
                nodal_value=randomness_result.nodal_value,
            )
        return ModelState(
            state=state,
            memory=memory,
            diagnostics=diagnostics,
            randomness=randomness,
        )

    def forward(
        self,
        initial_state: primitive_equations.State | ModelState,
        outer_steps: int,
        inner_steps: int | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
        post_process_fn: Callable[[primitive_equations.State], Any] | None = None,
        checkpoint_outer_steps: bool = False,
    ) -> tuple[primitive_equations.State | ModelState, list]:
        """Run model for outer_steps * inner_steps physics time steps.

        Each physics time step calls ``step()`` which internally runs
        ``dycore_substeps`` IMEX inner steps.

        Total simulation time =
            outer_steps × inner_steps × dt

        Total dynamics integrations =
            outer_steps × inner_steps × dycore_substeps

        Supports two calling modes:

        1. **Bare State** — ``initial_state`` is a ``State``.  Uses
           ``self.step()`` directly (deterministic, no memory/diagnostics).
        2. **ModelState** — ``initial_state`` is a ``ModelState``.  Uses
           ``self.step_model_state()`` which threads memory, diagnostics,
           and randomness through every step.

        Args:
            initial_state: initial State or ModelState.
            outer_steps: number of saved trajectory snapshots.
            inner_steps: physics sub-steps between saves
                (default: ``config.inner_steps``).
            forcings: optional dict of forcing tensors (SST, sea ice, etc.).
            post_process_fn: optional function applied to each saved state.
                When provided, the trajectory stores only the processed
                output (e.g. decoded observations) instead of full State
                objects, significantly reducing memory for long rollouts.
            checkpoint_outer_steps: if True and training, wrap each
                outer save interval with ``torch.utils.checkpoint`` to
                reduce peak memory at the cost of recomputation.

        Returns:
            ``(final_state, trajectory)`` where ``trajectory`` has length
            ``outer_steps``.  Type matches ``initial_state``.
        """
        if inner_steps is None:
            inner_steps = self.config.inner_steps

        # Apply TF32 setting
        self.policy.apply_tf32_setting()

        # Determine mode: ModelState or bare State
        use_model_state = isinstance(initial_state, ModelState)
        current = initial_state
        trajectory: list = []

        def _extract_state(s):
            return s.state if isinstance(s, ModelState) else s

        def _inner_block(s):
            for _ in range(inner_steps):
                if use_model_state:
                    s = self.step_model_state(s, forcings)
                else:
                    s = self.step(s, forcings)
            return s

        for _ in range(outer_steps):
            if checkpoint_outer_steps and self.training:
                current = torch_checkpoint.checkpoint(
                    _inner_block, current, use_reentrant=False,
                )
            else:
                current = _inner_block(current)

            save_state = _extract_state(current)
            if post_process_fn is not None:
                trajectory.append(post_process_fn(save_state))
            else:
                trajectory.append(save_state)

        return current, trajectory

    @torch.no_grad()
    def forward_streaming(
        self,
        initial_state: primitive_equations.State | ModelState,
        outer_steps: int,
        inner_steps: int | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
        post_process_fn: Callable[[primitive_equations.State], Any] | None = None,
    ):
        """Yield each trajectory state without accumulating in GPU memory.

        Generator equivalent of :meth:`forward` for memory-efficient
        inference.  Each outer step is yielded immediately so the caller
        can offload to CPU, compute metrics, or discard — keeping GPU
        peak memory at O(1) states instead of O(outer_steps).

        If *post_process_fn* is provided it is applied **before** yield
        (matching NeuralGCM's in-scan ``process_observations_fn``
        pattern) so that intermediate full-size States can be GC'd.

        Yields:
            ``post_process_fn(state)`` if provided, otherwise the bare
            ``State`` (on the model device).

        Returns:
            Final ``State`` or ``ModelState`` via ``generator.value``
            after ``StopIteration``.
        """
        if inner_steps is None:
            inner_steps = self.config.inner_steps

        self.policy.apply_tf32_setting()

        use_model_state = isinstance(initial_state, ModelState)
        current = initial_state

        def _extract_state(s):
            return s.state if isinstance(s, ModelState) else s

        for _ in range(outer_steps):
            for _ in range(inner_steps):
                if use_model_state:
                    current = self.step_model_state(current, forcings)
                else:
                    current = self.step(current, forcings)

            save_state = _extract_state(current)
            if post_process_fn is not None:
                yield post_process_fn(save_state)
            else:
                yield save_state

        return current

    @torch.no_grad()
    def forward_chunked(
        self,
        initial_state: primitive_equations.State | ModelState,
        chunk_size: int,
        inner_steps: int | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
        post_process_fn: Callable[[primitive_equations.State], Any] | None = None,
    ) -> tuple[primitive_equations.State | ModelState, list]:
        """Run *chunk_size* outer steps and return trajectory.

        Semantic wrapper around :meth:`forward` intended for chunked
        inference loops::

            while remaining > 0:
                n = min(chunk_size, remaining)
                state, chunk = model.forward_chunked(state, n, ...)
                offload(chunk)
                remaining -= n
        """
        return self.forward(
            initial_state, chunk_size, inner_steps,
            forcings=forcings, post_process_fn=post_process_fn,
        )

    def forward_nested(
        self,
        initial_state: primitive_equations.State | ModelState,
        forcings: list[dict[str, torch.Tensor]],
        frequencies: dict[str, int | float],
        dt: int | float = 1,
        collect_every: int = 1,
        checkpoint_level: int = -1,
        post_process_fn: Callable[[primitive_equations.State], Any] | None = None,
    ) -> tuple[primitive_equations.State | ModelState, list]:
        """Run model with multi-frequency nested forcing.

        This is the PyTorch equivalent of JAX's ``_recursive_scan`` for
        handling forcing data at different temporal frequencies (e.g. 6h
        atmospheric + 24h ocean SST).

        Args:
            initial_state: initial State or ModelState.
            forcings: flat list of forcing dicts, one per model step.
            frequencies: mapping from variable group to interval in dt
                units. E.g. ``{'atmos': 6, 'ocean': 24}``.
            dt: model timestep unit (default 1).
            collect_every: save trajectory every N model steps.
            checkpoint_level: nesting level to gradient-checkpoint
                (-1 = none, 0 = innermost, 1 = next level, ...).
            post_process_fn: optional transform applied to saved states.

        Returns:
            ``(final_state, trajectory)``
        """
        from pytorch_src.core.scan_utils import nested_rollout

        self.policy.apply_tf32_setting()
        use_model_state = isinstance(initial_state, ModelState)

        def _step(state, step_forcings):
            if use_model_state:
                return self.step_model_state(state, step_forcings)
            return self.step(state, step_forcings)

        def _extract(s):
            return s.state if isinstance(s, ModelState) else s

        final, raw_traj = nested_rollout(
            step_fn=_step,
            initial_state=initial_state,
            forcings=forcings,
            frequencies=frequencies,
            dt=dt,
            checkpoint_level=checkpoint_level,
            collect_every=collect_every,
        )

        if post_process_fn is not None:
            trajectory = [post_process_fn(_extract(s)) for s in raw_traj]
        else:
            trajectory = raw_traj

        return final, trajectory


class InferenceModel(nn.Module):
    """Inference wrapper with autoregressive rollout.

    Provides a simplified API for running the model in evaluation mode
    with optional post-processing of outputs.

    Follows the NeuralGCM state-machine interface:
      - ``assimilate``: encode raw observations into model state (+ init
        memory/diagnostics/randomness)
      - ``forward``:    autoregressive rollout (ModelState or bare State)
      - ``observe``:    decode model state to physical-space observations
    """

    def __init__(
        self,
        model: NeuralGCMModel,
        post_process_fn: Callable | None = None,
    ):
        super().__init__()
        self.model = model
        self.post_process_fn = post_process_fn or (lambda x: x)

    @torch.no_grad()
    def forward(
        self,
        initial_state: primitive_equations.State | ModelState,
        outer_steps: int,
        inner_steps: int | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list:
        """Run inference and return post-processed trajectory."""
        self.model.eval()
        _, trajectory = self.model(
            initial_state, outer_steps, inner_steps, forcings=forcings,
        )
        return [self.post_process_fn(s) for s in trajectory]

    def assimilate(
        self,
        inputs: dict[str, torch.Tensor],
        forcings: dict[str, torch.Tensor] | None = None,
        state: primitive_equations.State | None = None,
        alpha: float = 1.0,
        return_model_state: bool = False,
    ) -> primitive_equations.State | ModelState:
        """Encode raw observations into model state.

        When the model has an encoder, ``inputs`` (a dict of raw data fields
        such as pressure-level u, v, T, z, q) are encoded into a model-space
        ``State``.  If ``state`` is also provided, the result is blended::

            new_state = (1 - alpha) * state + alpha * encoded

        This matches the NeuralGCM legacy ``encode + finalize_state`` pattern.

        When ``return_model_state=True``, wraps the result in a
        ``ModelState`` with initialized memory, empty diagnostics, and
        randomness (if a random field is configured).  This is the
        recommended path for stochastic ensemble forecasting.

        When no encoder is available, falls back to simple linear nudging
        between ``state`` and ``inputs`` (treated as field tensors).

        Args:
            inputs: raw observation dict (pressure-level variables) or
                pre-encoded State when no encoder is available.
            forcings: optional forcing data (SST, sea ice, etc.).
            state: optional existing model state to blend with.
            alpha: blending factor (0 = keep state, 1 = replace with encoded).
            return_model_state: if True, returns a full ``ModelState`` with
                memory, diagnostics, and randomness initialized.

        Returns:
            Model-space State (or ModelState) ready for ``forward()``.
        """
        encoder = self.model.encoder
        if encoder is not None:
            # Full encoder path: raw data → model state
            try:
                encoded = encoder(inputs, forcings=forcings)
            except TypeError:
                encoded = encoder(inputs)
            if state is not None and alpha < 1.0:
                result_state = state * (1.0 - alpha) + encoded * alpha
            else:
                result_state = encoded
        elif state is not None and isinstance(inputs, primitive_equations.State):
            # Fallback: treat inputs as a pre-encoded State and do nudging
            result_state = state * (1.0 - alpha) + inputs * alpha
        else:
            raise ValueError(
                "No encoder configured and inputs are not a State. "
                "Provide an encoder or pass a pre-encoded State as inputs."
            )

        if return_model_state:
            return self.model.initialize_model_state(result_state)
        return result_state

    def observe(
        self,
        state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode model state to physical-space observations.

        When the model has a decoder, produces physical-space outputs
        (u, v, T, geopotential, etc.) matching the NeuralGCM
        ``decode`` interface.

        Without a decoder, returns raw spectral-space state fields.
        """
        decoder = self.model.decoder
        if decoder is not None:
            try:
                return decoder(state, forcings=forcings)
            except TypeError:
                return decoder(state)

        # Fallback: raw state fields
        result: dict[str, torch.Tensor] = {
            "vorticity": state.vorticity,
            "divergence": state.divergence,
            "temperature_variation": state.temperature_variation,
            "log_surface_pressure": state.log_surface_pressure,
        }
        if state.tracers:
            result.update(state.tracers)
        return result


class VectorizedModel(nn.Module):
    """Batch processing wrapper for ensemble forecasting.

    Accepts a batched ``State`` where every tensor has a leading batch
    dimension (dim 0).  Each ensemble member is either:

    * **vmap mode** (``use_vmap=True``, default when ``torch.vmap`` is
      available): uses ``torch.vmap`` over a flattened tensor representation
      of State so that all members execute in a single vectorised call.
      This mirrors the JAX ``jax.vmap``-based ``VectorizedModel``.

    * **loop mode** (``use_vmap=False`` or fallback): unbatches, loops
      over members, re-stacks.

    Args:
        model: the base NeuralGCMModel (or AtmosphericModel).
        use_vmap: attempt torch.vmap vectorisation.  Falls back to loop
            mode if vmap is unavailable or raises an error.
    """

    def __init__(self, model: NeuralGCMModel, use_vmap: bool = True):
        super().__init__()
        self.model = model
        self.use_vmap = use_vmap
        self._vmap_available = hasattr(torch, "vmap")

    # ── State ↔ flat-tensor helpers ───────────────────────────────────

    @staticmethod
    def _state_to_tensors(
        state: primitive_equations.State,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Flatten State to an ordered list of tensors + tracer keys."""
        tensors = [
            state.vorticity,
            state.divergence,
            state.temperature_variation,
            state.log_surface_pressure,
        ]
        tracer_keys = sorted(state.tracers.keys())
        for k in tracer_keys:
            tensors.append(state.tracers[k])
        return tensors, tracer_keys

    @staticmethod
    def _tensors_to_state(
        tensors: list[torch.Tensor],
        tracer_keys: list[str],
        sim_time: float | None = None,
    ) -> primitive_equations.State:
        """Reconstruct State from ordered tensor list."""
        tracers = {k: tensors[4 + i] for i, k in enumerate(tracer_keys)}
        return primitive_equations.State(
            vorticity=tensors[0],
            divergence=tensors[1],
            temperature_variation=tensors[2],
            log_surface_pressure=tensors[3],
            tracers=tracers,
            sim_time=sim_time,
        )

    @staticmethod
    def _unbatch_state(
        state: primitive_equations.State,
    ) -> list[primitive_equations.State]:
        """Split a batched State (leading dim 0) into a list of States."""
        batch_size = state.vorticity.shape[0]
        states = []
        for i in range(batch_size):
            states.append(primitive_equations.State(
                vorticity=state.vorticity[i],
                divergence=state.divergence[i],
                temperature_variation=state.temperature_variation[i],
                log_surface_pressure=state.log_surface_pressure[i],
                tracers={k: v[i] for k, v in state.tracers.items()},
                sim_time=state.sim_time,
            ))
        return states

    @staticmethod
    def _batch_states(
        states: list[primitive_equations.State],
    ) -> primitive_equations.State:
        """Stack a list of States into a single batched State."""
        return primitive_equations.State(
            vorticity=torch.stack([s.vorticity for s in states]),
            divergence=torch.stack([s.divergence for s in states]),
            temperature_variation=torch.stack(
                [s.temperature_variation for s in states]
            ),
            log_surface_pressure=torch.stack(
                [s.log_surface_pressure for s in states]
            ),
            tracers={
                k: torch.stack([s.tracers[k] for s in states])
                for k in states[0].tracers
            },
            sim_time=states[0].sim_time,
        )

    # ── vmap-based step ───────────────────────────────────────────────

    def _vmap_step(
        self,
        batched_state: primitive_equations.State,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> primitive_equations.State:
        """Vectorised step using torch.vmap over flattened tensors."""
        tensors, tracer_keys = self._state_to_tensors(batched_state)
        sim_time = batched_state.sim_time

        def _single_step(*flat_tensors):
            s = self._tensors_to_state(
                list(flat_tensors), tracer_keys, sim_time=sim_time,
            )
            out = self.model.step(s, forcings=forcings)
            out_tensors, _ = self._state_to_tensors(out)
            return tuple(out_tensors)

        vmapped = torch.vmap(_single_step, in_dims=tuple(0 for _ in tensors))
        results = vmapped(*tensors)

        updated_sim_time = sim_time
        # Try to read sim_time from a single member
        probe = self._tensors_to_state(
            [r[0] for r in results], tracer_keys, sim_time=sim_time,
        )
        dummy = self.model.step(
            self._unbatch_state(batched_state)[0], forcings=forcings,
        )
        updated_sim_time = dummy.sim_time

        return self._tensors_to_state(
            list(results), tracer_keys, sim_time=updated_sim_time,
        )

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        initial_states: primitive_equations.State | list[primitive_equations.State],
        outer_steps: int,
        inner_steps: int | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> tuple[primitive_equations.State, list[primitive_equations.State]]:
        """Run model for each member in the batch.

        Args:
            initial_states: batched State with leading batch dim **or**
                list of individual States.
            outer_steps: number of saved output steps.
            inner_steps: substeps between saves.
            forcings: optional forcing data (shared across members).

        Returns:
            ``(batched_final_state, batched_trajectory)`` — each
            trajectory entry is a batched State with leading batch dim.
        """
        if inner_steps is None:
            inner_steps = self.model.config.inner_steps

        # Try vmap path
        if self.use_vmap and self._vmap_available:
            # Normalise to batched State
            if isinstance(initial_states, list):
                initial_states = self._batch_states(initial_states)

            state = initial_states
            trajectory: list[primitive_equations.State] = []
            try:
                for _ in range(outer_steps):
                    for _ in range(inner_steps):
                        state = self._vmap_step(state, forcings)
                    trajectory.append(state)
                return state, trajectory
            except Exception:
                # Fallback to loop on vmap failure
                pass

        # ── Loop fallback ─────────────────────────────────────────────
        if isinstance(initial_states, primitive_equations.State):
            member_states = self._unbatch_state(initial_states)
        else:
            member_states = initial_states

        finals: list[primitive_equations.State] = []
        trajs: list[list] = []
        for s in member_states:
            final, traj = self.model(
                s, outer_steps, inner_steps, forcings=forcings,
            )
            finals.append(final)
            trajs.append(traj)

        # Re-stack: batch each trajectory step
        batched_final = self._batch_states(finals)
        n_steps = len(trajs[0])
        batched_traj = [
            self._batch_states([trajs[m][t] for m in range(len(finals))])
            for t in range(n_steps)
        ]
        return batched_final, batched_traj
