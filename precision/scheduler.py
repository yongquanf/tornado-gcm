"""PrecisionScheduler: adaptive precision strategy dispatch.

Implements three scheduling modes:
  - milestone: fixed phase transitions (preserves existing 3-phase behaviour)
  - loss_aware: advance phase when loss plateaus (O9-3)
  - stability_aware: escalate precision on NaN / gradient explosion
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import torch

from pytorch_src.precision.policy import PrecisionPolicy
from pytorch_src.precision.sda import SchedulerConfig

if TYPE_CHECKING:
    from pytorch_src.precision.profiler import ProfilerMetrics

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Training phase presets (mirrored from PrecisionPolicy.for_training_phase)
# ═══════════════════════════════════════════════════════════════════════════

PHASE_CONFIGS: list[dict] = [
    # Phase 1: aggressive low-precision
    dict(
        z3_compute_dtype=torch.bfloat16,
        z1_use_tf32=True,
        z2_compute_dtype=torch.float32,
    ),
    # Phase 2: standard PZHA
    dict(
        z3_compute_dtype=torch.bfloat16,
        z1_use_tf32=True,
        z2_compute_dtype=torch.float64,
    ),
    # Phase 3: high-precision fine-tune
    dict(
        z3_compute_dtype=torch.float32,
        z3_param_dtype=torch.float32,
        z1_use_tf32=True,
        z2_compute_dtype=torch.float64,
    ),
]

# Phase rollout steps (default progressive schedule)
PHASE_ROLLOUTS: list[int] = [4, 16, 32]

# Phase budget fractions  (60/30/10)
PHASE_BUDGETS: tuple[float, ...] = (0.60, 0.30, 0.10)


@dataclasses.dataclass
class SchedulerDecision:
    """Output of PrecisionScheduler.decide()."""

    policy_changed: bool = False
    new_policy: Optional[PrecisionPolicy] = None
    rollout_changed: bool = False
    new_rollout: Optional[int] = None
    reason: str = ""


class PrecisionScheduler:
    """Adaptive precision scheduler.

    Manages phase transitions across 3 training phases.  Supports three
    modes (set via ``SchedulerConfig.mode``):

    - ``milestone``: phase transitions at fixed step fractions (60%/90%).
    - ``loss_aware``: transitions when loss plateau is detected (O9-3).
    - ``stability_aware``: escalates precision on NaN or gradient explosion.

    Args:
        config: SchedulerConfig with mode and thresholds.
        base_policy: starting PrecisionPolicy (used to derive phase variants).
        total_steps: total training steps (needed for milestone mode).
    """

    def __init__(
        self,
        config: SchedulerConfig,
        base_policy: PrecisionPolicy,
        total_steps: int = 100_000,
    ) -> None:
        self.config = config
        self.base_policy = base_policy
        self.total_steps = total_steps

        # Phase state
        self.current_phase: int = 0  # 0-indexed: Phase 1/2/3
        self.max_phase: int = len(PHASE_CONFIGS) - 1

        # Pre-build phase policies
        self._phase_policies = self._build_phase_policies(base_policy)
        self._phase_rollouts = list(PHASE_ROLLOUTS)

        # Loss-aware tracking
        self._loss_window: deque[float] = deque(maxlen=200)
        self._plateau_count: int = 0

        # Stability tracking
        self._nan_count: int = 0
        self._last_switch_step: int = 0

        # Warmup tracking: after a phase switch, delay rollout change
        self._warmup_remaining: int = 0
        self._pending_rollout: int | None = None

        # Rollout ramp state: linearly interpolate K over ramp_steps
        self._ramp_remaining: int = 0
        self._ramp_start_rollout: int = PHASE_ROLLOUTS[0]
        self._ramp_target_rollout: int = PHASE_ROLLOUTS[0]
        self._ramp_total: int = 0

        # Last decision (for external inspection)
        self.last_decision = SchedulerDecision()

    def _build_phase_policies(
        self, base: PrecisionPolicy
    ) -> list[PrecisionPolicy]:
        """Build per-phase PrecisionPolicy from base + PHASE_CONFIGS."""
        policies = []
        for cfg in PHASE_CONFIGS:
            policies.append(dataclasses.replace(base, **cfg))
        return policies

    def decide(
        self,
        metrics: "ProfilerMetrics",
        step: int,
    ) -> SchedulerDecision:
        """Make a scheduling decision based on current metrics.

        Args:
            metrics: ProfilerMetrics from SDAProfiler.collect().
            step: current global training step.

        Returns:
            SchedulerDecision indicating any policy/rollout changes.
        """
        if self.config.mode == "milestone":
            decision = self._milestone_decide(step)
        elif self.config.mode == "loss_aware":
            decision = self._loss_aware_decide(metrics, step)
        elif self.config.mode == "stability_aware":
            decision = self._stability_aware_decide(metrics, step)
        else:
            decision = SchedulerDecision(reason=f"unknown mode: {self.config.mode}")

        self.last_decision = decision
        return decision

    # ── Milestone mode ──

    def _milestone_decide(self, step: int) -> SchedulerDecision:
        """Fixed fraction-based phase transitions (60% / 90%)."""
        frac = step / max(self.total_steps, 1)
        cumulative = 0.0
        target_phase = 0
        for i, budget in enumerate(PHASE_BUDGETS):
            cumulative += budget
            if frac < cumulative:
                target_phase = i
                break
        else:
            target_phase = self.max_phase

        if target_phase != self.current_phase and target_phase > self.current_phase:
            return self._advance_to(target_phase, step, reason="milestone")
        return SchedulerDecision()

    # ── Loss-aware mode (O9-3) ──

    def _loss_aware_decide(
        self, metrics: "ProfilerMetrics", step: int
    ) -> SchedulerDecision:
        """Advance phase when loss plateaus."""
        # Accumulate loss history
        for loss_val in metrics.recent_losses:
            self._loss_window.append(loss_val)

        # Also check NaN → immediate precision escalation
        if metrics.nan_count > self._nan_count:
            self._nan_count = metrics.nan_count
            if (
                metrics.nan_count >= self.config.nan_trigger_count
                and self.current_phase < self.max_phase
            ):
                return self._advance_to(
                    self.current_phase + 1, step,
                    reason=f"NaN detected ({metrics.nan_count}x)",
                )

        # Need enough history for plateau detection
        n = len(self._loss_window)
        if n < 20:
            return SchedulerDecision()

        # Compute relative loss change over recent window
        recent = list(self._loss_window)
        half = n // 2
        old_mean = sum(recent[:half]) / half
        new_mean = sum(recent[half:]) / (n - half)

        if old_mean == 0:
            return SchedulerDecision()

        rel_change = abs(new_mean - old_mean) / (abs(old_mean) + 1e-30)

        if rel_change < self.config.loss_plateau_threshold:
            self._plateau_count += 1
        else:
            self._plateau_count = max(0, self._plateau_count - 1)

        if (
            self._plateau_count >= self.config.loss_plateau_patience
            and self.current_phase < self.max_phase
            and (step - self._last_switch_step) > self.config.check_interval * 3
        ):
            self._plateau_count = 0
            self._loss_window.clear()
            return self._advance_to(
                self.current_phase + 1, step,
                reason=f"loss plateau (rel_change={rel_change:.4f})",
            )

        return SchedulerDecision()

    # ── Stability-aware mode ──

    def _stability_aware_decide(
        self, metrics: "ProfilerMetrics", step: int
    ) -> SchedulerDecision:
        """Escalate precision on NaN, gradient explosion, or spectral tail."""
        reasons = []

        # NaN detection → immediate escalation
        if metrics.nan_count >= self.config.nan_trigger_count:
            reasons.append(f"NaN×{metrics.nan_count}")

        # Gradient variance check
        if metrics.gradient_stats:
            max_grad = max(
                (gs.max_abs for gs in metrics.gradient_stats.values()), default=0.0
            )
            if max_grad > self.config.grad_var_threshold:
                reasons.append(f"grad_max={max_grad:.1e}")

        # Spectral tail check
        if metrics.spectral_energy_tail > self.config.spectral_tail_threshold:
            reasons.append(f"spectral_tail={metrics.spectral_energy_tail:.3f}")

        if (
            reasons
            and self.current_phase < self.max_phase
            and (step - self._last_switch_step) > self.config.check_interval * 2
        ):
            return self._advance_to(
                self.current_phase + 1, step,
                reason="stability: " + ", ".join(reasons),
            )

        return SchedulerDecision()

    # ── Phase advancement ──

    def _advance_to(
        self, target_phase: int, step: int, *, reason: str
    ) -> SchedulerDecision:
        """Advance to a target phase, returning the decision.

        The precision policy changes immediately, but the rollout increase
        is deferred for ``warmup_steps_after_switch`` steps (if configured)
        to avoid a compounded loss discontinuity from simultaneous policy
        + rollout changes.  The deferred rollout is applied by
        :meth:`tick_warmup`.
        """
        target_phase = min(target_phase, self.max_phase)
        if target_phase <= self.current_phase:
            return SchedulerDecision()

        old_phase = self.current_phase
        self.current_phase = target_phase
        self._last_switch_step = step

        new_policy = self._phase_policies[target_phase]
        new_rollout = self._phase_rollouts[target_phase]

        warmup = self.config.warmup_steps_after_switch
        if warmup > 0:
            # Defer rollout change: keep old rollout during warmup
            self._warmup_remaining = warmup
            self._pending_rollout = new_rollout
            logger.info(
                "Scheduler: phase %d → %d at step %d [%s], "
                "policy changed now, rollout → %d deferred by %d warmup steps",
                old_phase + 1, target_phase + 1, step, reason,
                new_rollout, warmup,
            )
            return SchedulerDecision(
                policy_changed=True,
                new_policy=new_policy,
                rollout_changed=False,
                new_rollout=None,
                reason=reason,
            )

        logger.info(
            "Scheduler: phase %d → %d at step %d [%s], rollout → %d",
            old_phase + 1, target_phase + 1, step, reason, new_rollout,
        )

        return SchedulerDecision(
            policy_changed=True,
            new_policy=new_policy,
            rollout_changed=True,
            new_rollout=new_rollout,
            reason=reason,
        )

    # ── Inspection ──

    def tick_warmup(self) -> SchedulerDecision:
        """Called each step to apply deferred rollout changes.

        Handles two mechanisms in sequence:

        1. **Warmup hold**: After a phase transition, the old rollout is
           kept for ``warmup_steps_after_switch`` steps while the new
           precision policy stabilises.
        2. **Rollout ramp**: Once warmup expires (or if warmup=0), the
           rollout linearly interpolates from the old value to the target
           over ``rollout_ramp_steps`` steps.  If ramp=0, the jump is
           immediate.

        Returns:
            SchedulerDecision with rollout_changed=True when rollout
            updates, or an empty decision otherwise.
        """
        # Phase 1: warmup hold (precision changed, rollout not yet)
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            if self._warmup_remaining == 0 and self._pending_rollout is not None:
                target = self._pending_rollout
                self._pending_rollout = None
                ramp = self.config.rollout_ramp_steps
                if ramp > 0:
                    # Start ramping from current rollout to target
                    self._ramp_start_rollout = self._phase_rollouts[
                        max(0, self.current_phase - 1)
                    ]
                    self._ramp_target_rollout = target
                    self._ramp_total = ramp
                    self._ramp_remaining = ramp
                    # First ramp step
                    return self._ramp_tick()
                logger.info(
                    "Scheduler: warmup complete, rollout → %d", target,
                )
                return SchedulerDecision(
                    rollout_changed=True,
                    new_rollout=target,
                    reason="deferred rollout after warmup",
                )
            return SchedulerDecision()

        # Phase 2: rollout ramp (linear interpolation)
        if self._ramp_remaining > 0:
            return self._ramp_tick()

        return SchedulerDecision()

    def _ramp_tick(self) -> SchedulerDecision:
        """Emit the next interpolated rollout during a ramp."""
        self._ramp_remaining -= 1
        progress = 1.0 - (self._ramp_remaining / max(self._ramp_total, 1))
        k = round(
            self._ramp_start_rollout
            + progress * (self._ramp_target_rollout - self._ramp_start_rollout)
        )
        k = max(1, k)
        if self._ramp_remaining == 0:
            k = self._ramp_target_rollout
            logger.info("Scheduler: rollout ramp complete, K=%d", k)
        return SchedulerDecision(
            rollout_changed=True,
            new_rollout=k,
            reason=f"rollout ramp ({progress:.0%})",
        )

    @property
    def current_policy(self) -> PrecisionPolicy:
        """The policy for the current phase."""
        return self._phase_policies[self.current_phase]

    @property
    def current_rollout(self) -> int:
        """The rollout length for the current phase."""
        return self._phase_rollouts[self.current_phase]
