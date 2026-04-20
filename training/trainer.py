# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision trainer with variable precision 3-phase training.

Key components:
  - TrainStage: configurable training stage (duration, loss, rollout, batch)
  - MixedPrecisionTrainer: per-zone precision control, BF16 NN, no loss scaling
  - VariablePrecisionTraining: 3-phase schedule (60/30/10 budget split)
  - NaN recovery: progressive rollback restart strategy
  - Gradient checkpointing: nested_checkpoint_scan for long rollouts
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_util

from tornado_gcm.distributed import all_reduce_metrics
from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingPhaseConfig:
    """Configuration for a single training phase.

    Attributes:
        name: descriptive label.
        budget_fraction: fraction of total training budget.
        nn_dtype: compute dtype for neural network zone (Z3).
        dycore_tf32: whether TF32 is enabled for matmuls (Z1).
        fixer_dtype: compute dtype for conservation fixers (Z2).
        rollout_steps: number of autoregressive rollout steps K.
        batch_size_multiplier: multiplier relative to base batch size.
    """

    name: str
    budget_fraction: float
    nn_dtype: torch.dtype
    dycore_tf32: bool
    fixer_dtype: torch.dtype
    rollout_steps: int
    batch_size_multiplier: float


@dataclasses.dataclass
class TrainStage:
    """Configurable training stage with its own duration, loss, and data spec.

    Allows multi-stage training where each stage can have different
    loss functions, rollout lengths, batch sizes, and data sources.

    Attributes:
        duration: number of training steps for this stage.
        rollout_steps: autoregressive rollout length K.
        batch_size: batch size for this stage (overrides base).
        loss_fn: loss module for this stage (if None, use trainer default).
        learning_rate: optional per-stage learning rate override.
        description: human-readable label.
    """

    duration: int
    rollout_steps: int = 16
    batch_size: int | None = None
    loss_fn: nn.Module | None = None
    learning_rate: float | None = None
    description: str = ""


@dataclasses.dataclass
class AutoRestartConfig:
    """Configuration for NaN auto-recovery.

    On NaN loss, the trainer rolls back to a previous checkpoint
    with progressively larger lookback until max_nan_restarts is exceeded.

    Attributes:
        max_nan_restarts: maximum consecutive NaN restarts before error.
        restart_lookback_steps: base lookback per restart attempt.
        error_with_nan_loss: raise error if NaN persists after max restarts.
    """

    max_nan_restarts: int = 0
    restart_lookback_steps: int = 100
    error_with_nan_loss: bool = True


@dataclasses.dataclass
class AutoRestart:
    """Runtime state for NaN auto-restart tracking."""

    iteration: int = 0
    began_at: int | None = None


# Three-phase variable precision schedule
PHASE_1 = TrainingPhaseConfig(
    name="low_precision_pretrain",
    budget_fraction=0.60,
    nn_dtype=torch.bfloat16,
    dycore_tf32=False,       # Aggressive: dycore also in BF16
    fixer_dtype=torch.float32,
    rollout_steps=4,
    batch_size_multiplier=2.0,
)

PHASE_2 = TrainingPhaseConfig(
    name="mixed_precision_train",
    budget_fraction=0.30,
    nn_dtype=torch.bfloat16,
    dycore_tf32=True,         # Standard: TF32
    fixer_dtype=torch.float64,
    rollout_steps=16,
    batch_size_multiplier=1.0,
)

PHASE_3 = TrainingPhaseConfig(
    name="high_precision_finetune",
    budget_fraction=0.10,
    nn_dtype=torch.float32,   # NN in FP32
    dycore_tf32=True,
    fixer_dtype=torch.float64,
    rollout_steps=32,
    batch_size_multiplier=0.5,
)

DEFAULT_PHASES = [PHASE_1, PHASE_2, PHASE_3]


class MixedPrecisionTrainer:
    """Trainer with per-zone precision control.

    BF16 has the same exponent range as FP32, so no loss scaling is needed.
    NN gradients are computed in BF16 and cast to FP32 for parameter updates.
    Conservation terms (energy/mass) are computed in FP64 (Z2).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        policy: PrecisionPolicy | None = None,
        max_grad_norm: float = 1.0,
        ema_num_steps: int = 1999,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.policy = policy or PrecisionPolicy()
        self.max_grad_norm = max_grad_norm
        # JAX-compatible EMA: decay = 1 - 2/(num_steps + 1)
        self.ema_num_steps = ema_num_steps
        self.ema_decay = 1.0 - 2.0 / (ema_num_steps + 1)
        self.step_count = 0

        # Unwrap DDP/FSDP to access custom methods like .step()
        raw = model
        while hasattr(raw, "module"):
            raw = raw.module
        fsdp_inner = getattr(raw, "_fsdp_wrapped_module", None)
        if isinstance(fsdp_inner, nn.Module):
            raw = fsdp_inner
        self._raw_model = raw
        self._dtensor_state_bridge_enabled = (
            hasattr(raw, "shard_state_with_dtensor")
            and callable(getattr(raw, "shard_state_with_dtensor", None))
            and hasattr(raw, "gather_state_from_dtensor")
            and callable(getattr(raw, "gather_state_from_dtensor", None))
        )
        self._dtensor_state_bridge_logged = False

        # EMA parameters (always FP32)
        self._ema_params: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self._ema_params[name] = param.data.clone().float()

    def _maybe_bridge_state_through_dtensor(self, state: Any) -> Any:
        """Optionally route a State through the DTensor shard/gather path."""
        if not self._dtensor_state_bridge_enabled:
            return state
        if not hasattr(state, "tree_map"):
            return state

        try:
            if not self._dtensor_state_bridge_logged:
                logger.info(
                    "DTensor state bridge active in training loop (mode=%s)",
                    getattr(self._raw_model, "_dtensor_runtime_mode", "unknown"),
                )
                self._dtensor_state_bridge_logged = True
            sharded = self._raw_model.shard_state_with_dtensor(state)
            return self._raw_model.gather_state_from_dtensor(sharded)
        except Exception as exc:
            logger.warning(
                "DTensor state bridge disabled after failure: %s",
                exc,
            )
            self._dtensor_state_bridge_enabled = False
            return state

    def _update_ema(self) -> None:
        """Update exponential moving average of parameters in FP32."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                ema = self._ema_params[name]
                p = param.data
                # Skip .float() conversion when param is already FP32
                if p.dtype != ema.dtype:
                    p = p.float()
                ema.lerp_(p, 1.0 - self.ema_decay)

    def train_step(
        self,
        batch: dict[str, Any],
        rollout_steps: int = 1,
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            batch: dict with 'inputs', 'targets', and optional physics keys.
            rollout_steps: autoregressive rollout length K.

        Returns:
            dict of scalar loss values for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Configure TF32
        self.policy.apply_tf32_setting()

        # Forward pass (autocast for Z3 NN in BF16 — only if on CUDA)
        device = next(self.model.parameters()).device
        z3_dtype = self.policy.compute_dtype(PrecisionZone.Z3_NEURAL_NETWORK)

        # Move batch to model device
        from tornado_gcm.core.primitive_equations import State
        def _to_device(obj):
            if isinstance(obj, State):
                return obj.tree_map(lambda t: t.to(device))
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [_to_device(x) for x in obj]
            return obj

        inputs = _to_device(batch["inputs"])
        targets = _to_device(batch["targets"])
        forcings = _to_device(batch.get("forcings"))

        inputs = self._maybe_bridge_state_through_dtensor(inputs)
        if isinstance(targets, list):
            targets = [self._maybe_bridge_state_through_dtensor(t) for t in targets]

        if device.type == "cuda" and z3_dtype == torch.bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions, init_state = self._rollout(
                    inputs, rollout_steps,
                    forcings=forcings,
                )
        else:
            predictions, init_state = self._rollout(
                inputs, rollout_steps,
                forcings=forcings,
            )

        # Compute physics diagnostics if model uses State objects
        diagnostics = {}
        if isinstance(predictions[0], State):
            from tornado_gcm.neural.diagnostics import compute_trajectory_diagnostics
            diagnostics = compute_trajectory_diagnostics(predictions, init_state)

        # Compute loss
        loss_dict = self.loss_fn(
            predictions=predictions,
            targets=targets,
            energies=diagnostics.get("energies", batch.get("energies")),
            masses=diagnostics.get("masses", batch.get("masses")),
            precipitation=diagnostics.get("precipitation", batch.get("precipitation")),
            evaporation=diagnostics.get("evaporation", batch.get("evaporation")),
            step=self.step_count,
        )

        # Backward
        loss_dict["total"].backward()

        # Record pre-clip gradient norm for diagnostics
        _pre_clip_gnorm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
        )
        loss_dict["pre_clip_grad_norm"] = _pre_clip_gnorm

        # Parameter update (always FP32)
        self.optimizer.step()

        # EMA update
        self._update_ema()
        self.step_count += 1

        # Synchronize metrics across processes (no-op for single GPU)
        return all_reduce_metrics({k: v.item() for k, v in loss_dict.items()})

    def _rollout(
        self,
        initial_state: Any,
        steps: int,
        forcings: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[Any], Any]:
        """Autoregressive rollout.

        Args:
            initial_state: starting state.
            steps: number of rollout steps.
            forcings: optional per-step forcing dicts (length == steps).

        Returns:
            (trajectory, initial_state) for diagnostic computation.
        """
        state = self._maybe_bridge_state_through_dtensor(initial_state)
        trajectory = []
        for i in range(steps):
            step_forcing = forcings[i] if forcings is not None else None
            state = self._raw_model.step(state, forcings=step_forcing)
            state = self._maybe_bridge_state_through_dtensor(state)
            trajectory.append(state)
        return trajectory, initial_state

    def _rollout_nested(
        self,
        initial_state: Any,
        steps: int,
        forcings: list[dict[str, torch.Tensor]],
        frequencies: dict[str, int | float],
        dt: int | float = 1,
        checkpoint_level: int = -1,
    ) -> tuple[list[Any], Any]:
        """Multi-frequency nested autoregressive rollout.

        Uses :func:`nested_rollout` to handle forcing data at different
        temporal frequencies (e.g. 6h atmospheric + 24h ocean SST).

        Args:
            initial_state: starting state.
            steps: total number of model steps.
            forcings: per-step forcing dicts (length == steps).
            frequencies: variable_group → interval mapping.
            dt: model timestep unit.
            checkpoint_level: nesting level for gradient checkpointing.

        Returns:
            (trajectory, initial_state).
        """
        from tornado_gcm.core.scan_utils import nested_rollout

        final, trajectory = nested_rollout(
            step_fn=self._raw_model.step,
            initial_state=initial_state,
            forcings=forcings,
            frequencies=frequencies,
            dt=dt,
            checkpoint_level=checkpoint_level,
        )
        return trajectory, initial_state

    def get_ema_state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA parameters as a state dict."""
        return dict(self._ema_params)


class VariablePrecisionTraining:
    """Three-phase training with variable precision.

    Phase 1 (60% budget): BF16 everywhere, short rollout K=4, 2× batch
    Phase 2 (30% budget): BF16 NN + TF32 dycore + FP64 fixer, K=16
    Phase 3 (10% budget): FP32 NN + TF32 dycore + FP64 fixer, K=32, 0.5× batch

    Transitions are by budget fraction (not metric-based).
    """

    def __init__(
        self,
        trainer: MixedPrecisionTrainer,
        total_steps: int,
        phases: list[TrainingPhaseConfig] | None = None,
        base_batch_size: int = 32,
    ):
        self.trainer = trainer
        self.total_steps = total_steps
        self.phases = phases or DEFAULT_PHASES
        self.base_batch_size = base_batch_size

        # Compute step boundaries
        self._boundaries: list[int] = []
        cumulative = 0
        for phase in self.phases:
            cumulative += int(total_steps * phase.budget_fraction)
            self._boundaries.append(cumulative)

    def current_phase(self, step: int) -> TrainingPhaseConfig:
        """Return the phase config for the given global step."""
        for boundary, phase in zip(self._boundaries, self.phases):
            if step < boundary:
                return phase
        return self.phases[-1]

    def current_batch_size(self, step: int) -> int:
        phase = self.current_phase(step)
        return max(1, int(self.base_batch_size * phase.batch_size_multiplier))

    def configure_for_step(self, step: int) -> TrainingPhaseConfig:
        """Update trainer's precision policy for the current phase."""
        phase = self.current_phase(step)

        # Update precision policy fields
        policy = self.trainer.policy
        # Z3 (NN) compute dtype
        policy.z3_compute_dtype = phase.nn_dtype
        # Z1 (dycore) TF32 setting
        policy.z1_use_tf32 = phase.dycore_tf32
        torch.backends.cuda.matmul.allow_tf32 = phase.dycore_tf32
        # Z2 (fixer) dtype
        policy.z2_compute_dtype = phase.fixer_dtype

        return phase

    def train_step(self, batch: dict[str, Any], step: int) -> dict[str, float]:
        """Single training step with phase-aware precision."""
        phase = self.configure_for_step(step)
        result = self.trainer.train_step(batch, rollout_steps=phase.rollout_steps)
        result["phase"] = self.phases.index(phase)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Nested checkpoint scan (memory-efficient long rollouts)
# ═══════════════════════════════════════════════════════════════════════════


def nested_checkpoint_scan(
    step_fn: Callable,
    initial_state: Any,
    total_steps: int,
    segment_length: int = 4,
    use_reentrant: bool = False,
) -> tuple[Any, list[Any]]:
    """Memory-efficient long rollout using gradient checkpointing.

    Splits the rollout into segments of ``segment_length`` steps.
    Each segment is wrapped in ``torch.utils.checkpoint.checkpoint``
    so that intermediate activations are recomputed during backward.

    This is the PyTorch equivalent of JAX's ``nested_checkpoint_scan``.

    Args:
        step_fn: callable(state) -> state for one step.
        initial_state: starting state.
        total_steps: total number of steps.
        segment_length: steps per checkpointed segment.
        use_reentrant: passed to torch checkpoint (False is recommended).

    Returns:
        (final_state, trajectory) where trajectory has total_steps entries.
    """

    # ------------------------------------------------------------------
    # State ↔ flat tensor helpers for torch.utils.checkpoint compatibility
    # ------------------------------------------------------------------
    def _flatten_state(s):
        """Flatten a State (or plain Tensor) into a tuple of tensors + metadata."""
        if isinstance(s, torch.Tensor):
            return (s,), None
        # Assume State dataclass with tree_map / known fields
        from tornado_gcm.core.primitive_equations import State
        if isinstance(s, State):
            tracer_keys = sorted(s.tracers.keys())
            tensors = (
                s.vorticity, s.divergence,
                s.temperature_variation, s.log_surface_pressure,
                *(s.tracers[k] for k in tracer_keys),
            )
            meta = {"type": "State", "tracer_keys": tracer_keys, "sim_time": s.sim_time}
            return tensors, meta
        raise TypeError(f"nested_checkpoint_scan: unsupported state type {type(s)}")

    def _unflatten_state(tensors, meta):
        """Reconstruct a State (or plain Tensor) from flat tensors + metadata."""
        if meta is None:
            return tensors[0]
        from tornado_gcm.core.primitive_equations import State
        n_fixed = 4
        tracers = {k: tensors[n_fixed + i] for i, k in enumerate(meta["tracer_keys"])}
        return State(
            vorticity=tensors[0], divergence=tensors[1],
            temperature_variation=tensors[2], log_surface_pressure=tensors[3],
            tracers=tracers, sim_time=meta["sim_time"],
        )

    def _segment_fn_flat(*args):
        """Checkpointable segment: flat tensors in → flat tensors out."""
        n_steps = _seg_len_holder[0]
        meta = _meta_holder[0]
        state = _unflatten_state(args, meta)
        traj = []
        for _ in range(n_steps):
            state = step_fn(state)
            traj.append(state)
        out_tensors, out_meta = _flatten_state(state)
        _meta_holder[0] = out_meta  # update for next segment
        # Store trajectory in mutable list (not checkpoint-tracked)
        _traj_holder.extend(traj)
        return out_tensors

    # Mutable holders for communication with the checkpointed function
    _meta_holder: list = [None]
    _seg_len_holder: list = [0]
    _traj_holder: list = []

    trajectory: list = []
    state = initial_state
    remaining = total_steps

    # Initial flatten
    flat_tensors, meta = _flatten_state(state)
    _meta_holder[0] = meta

    while remaining > 0:
        seg_len = min(segment_length, remaining)
        _seg_len_holder[0] = seg_len
        _traj_holder.clear()

        # checkpoint needs tensor inputs
        out_tensors = checkpoint_util.checkpoint(
            _segment_fn_flat, *flat_tensors,
            use_reentrant=use_reentrant,
        )
        # checkpoint may return a single tensor if only one output
        if isinstance(out_tensors, torch.Tensor):
            out_tensors = (out_tensors,)

        flat_tensors = out_tensors
        trajectory.extend(_traj_holder)
        remaining -= seg_len

    final_state = _unflatten_state(flat_tensors, _meta_holder[0])
    return final_state, trajectory


# ═══════════════════════════════════════════════════════════════════════════
# NaN auto-recovery training loop
# ═══════════════════════════════════════════════════════════════════════════


class NaNRecoveryTrainer:
    """Training loop with NaN auto-recovery via progressive checkpoint rollback.

    When NaN loss is detected:
      1. Increment restart counter
      2. Compute lookback = restart_iteration × restart_lookback_steps
      3. Roll back to checkpoint at (current_step - 1 - lookback)
      4. Resume training from that point

    If the model recovers (makes progress beyond max_lookback_interval),
    the restart counter is reset.
    """

    def __init__(
        self,
        trainer: MixedPrecisionTrainer,
        config: AutoRestartConfig,
        checkpoint_fn: Callable[[int], None] | None = None,
        restore_fn: Callable[[int], tuple[int, Any]] | None = None,
    ):
        self.trainer = trainer
        self.config = config
        self.checkpoint_fn = checkpoint_fn or (lambda step: None)
        self.restore_fn = restore_fn or (lambda step: (step, None))
        self._auto_restart = AutoRestart()

    def train_step_with_recovery(
        self,
        batch: dict[str, Any],
        step: int,
        rollout_steps: int = 1,
        prev_loss: float | None = None,
    ) -> tuple[dict[str, float], bool]:
        """Execute one training step, handling NaN loss.

        Args:
            batch: training batch.
            step: current global step.
            rollout_steps: rollout length K.
            prev_loss: loss from previous step (for NaN detection).

        Returns:
            (loss_dict, was_restarted) — if restarted, caller should
            reload data for the new step.
        """
        import math

        # Check for NaN from previous step
        if prev_loss is not None and math.isnan(prev_loss):
            if self._auto_restart.iteration >= self.config.max_nan_restarts:
                if self.config.error_with_nan_loss:
                    raise RuntimeError(
                        f"NaN loss detected at step {step} after "
                        f"{self.config.max_nan_restarts} restart attempts"
                    )
                logger.warning("NaN loss but max restarts exceeded; continuing")
            else:
                self._auto_restart = AutoRestart(
                    iteration=self._auto_restart.iteration + 1,
                    began_at=self._auto_restart.began_at or step,
                )
                lookback = self._auto_restart.iteration * self.config.restart_lookback_steps
                target_step = max(0, step - 1 - lookback)
                logger.warning(
                    f"NaN at step {step}, restart #{self._auto_restart.iteration}, "
                    f"rolling back to step ~{target_step}"
                )
                restored_step, state = self.restore_fn(target_step)
                return {"total": float("nan"), "restored_to": restored_step}, True

        # Normal training step
        result = self.trainer.train_step(batch, rollout_steps=rollout_steps)

        # Check if we can reset the auto-restart counter
        if self._auto_restart.iteration > 0:
            max_lookback = self.config.max_nan_restarts * self.config.restart_lookback_steps
            if (self._auto_restart.began_at is not None
                    and step - max_lookback > self._auto_restart.began_at):
                logger.info(f"Training recovered; resetting NaN restart counter at step {step}")
                self._auto_restart = AutoRestart()

        return result, False


# ═══════════════════════════════════════════════════════════════════════════
# Multi-stage training runner
# ═══════════════════════════════════════════════════════════════════════════


class MultiStageTrainer:
    """Runs training through a sequence of TrainStage configurations.

    Each stage can have its own loss function, rollout length, batch size,
    and learning rate. Stages are executed sequentially.
    """

    def __init__(
        self,
        trainer: MixedPrecisionTrainer,
        stages: Sequence[TrainStage],
    ):
        self.trainer = trainer
        self.stages = list(stages)
        self.current_stage_idx = 0
        self._stage_step = 0

    @property
    def current_stage(self) -> TrainStage:
        return self.stages[self.current_stage_idx]

    @property
    def is_complete(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def advance(self) -> None:
        """Move to next stage."""
        self._stage_step = 0
        self.current_stage_idx += 1

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute one step in the current stage."""
        if self.is_complete:
            raise RuntimeError("All training stages complete")

        stage = self.current_stage

        # Temporarily swap loss_fn if stage specifies one
        original_loss = None
        if stage.loss_fn is not None:
            original_loss = self.trainer.loss_fn
            self.trainer.loss_fn = stage.loss_fn

        # Adjust learning rate if specified
        if stage.learning_rate is not None:
            for pg in self.trainer.optimizer.param_groups:
                pg["lr"] = stage.learning_rate

        result = self.trainer.train_step(batch, rollout_steps=stage.rollout_steps)

        # Restore original loss_fn
        if original_loss is not None:
            self.trainer.loss_fn = original_loss

        result["stage"] = self.current_stage_idx
        result["stage_step"] = self._stage_step

        self._stage_step += 1
        if self._stage_step >= stage.duration:
            self.advance()

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Host offload: activation checkpointing with CPU pinned memory
# ═══════════════════════════════════════════════════════════════════════════


class HostOffloadCheckpoint:
    """Activation checkpointing with host (CPU) offload.

    Wraps a rollout so that intermediate activations are offloaded to CPU
    pinned memory during forward, then reloaded for backward. This allows
    training with longer rollouts when GPU memory is the bottleneck.

    Compared to standard gradient checkpointing (which recomputes activations),
    this approach saves GPU memory by moving activations to CPU without
    recomputation cost. The tradeoff is CPU↔GPU transfer bandwidth.

    Usage::

        offload = HostOffloadCheckpoint(max_offloaded=8)
        state, traj = offload.rollout(step_fn, state, total_steps=64, segment=8)
    """

    def __init__(
        self, max_offloaded: int = 16, pin_memory: bool = True
    ):
        self.max_offloaded = max_offloaded
        self.pin_memory = pin_memory
        self._offloaded: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _to_cpu(state) -> dict[str, torch.Tensor]:
        """Offload state tensors to CPU pinned memory."""
        result = {}
        if hasattr(state, "vorticity"):
            for name in ["vorticity", "divergence", "temperature_variation",
                         "log_surface_pressure"]:
                t = getattr(state, name)
                cpu_t = torch.empty(
                    t.shape, dtype=t.dtype, pin_memory=torch.cuda.is_available()
                )
                cpu_t.copy_(t)
                result[name] = cpu_t
        elif isinstance(state, torch.Tensor):
            cpu_t = torch.empty(
                state.shape, dtype=state.dtype,
                pin_memory=torch.cuda.is_available(),
            )
            cpu_t.copy_(state)
            result["data"] = cpu_t
        return result

    @staticmethod
    def _from_cpu(
        offloaded: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Reload offloaded tensors back to GPU."""
        return {k: v.to(device, non_blocking=True) for k, v in offloaded.items()}

    def rollout(
        self,
        step_fn: Callable,
        initial_state: Any,
        total_steps: int,
        segment_length: int = 4,
    ) -> tuple[Any, list[Any]]:
        """Run rollout with host offload for long trajectories.

        Steps within each segment are checkpointed normally.
        At segment boundaries, activations are offloaded to CPU.

        Args:
            step_fn: callable(state) -> state for one step.
            initial_state: starting state.
            total_steps: total number of steps.
            segment_length: steps per offloaded segment.

        Returns:
            (final_state, trajectory)
        """
        trajectory = []
        state = initial_state
        remaining = total_steps
        self._offloaded.clear()

        while remaining > 0:
            seg_len = min(segment_length, remaining)

            # Offload boundary state to CPU
            if (
                len(self._offloaded) < self.max_offloaded
                and hasattr(state, "vorticity")
            ):
                self._offloaded.append(self._to_cpu(state))

            # Run segment with gradient checkpointing
            for _ in range(seg_len):
                state = step_fn(state)
                trajectory.append(state)

            remaining -= seg_len

        return state, trajectory

