# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Learning rate and rollout schedules.

Learning rate schedules aligned with JAX NeuralGCM (optax-based):
  - CosineDecaySchedule        (cosine decay + linear warmup)
  - ExponentialDecaySchedule    (≡ optax.exponential_decay)
  - WarmupExponentialDecaySchedule (≡ optax.warmup_exponential_decay_schedule)
  - PiecewiseConstantSchedule   (≡ optax.piecewise_constant_schedule)
  - PiecewiseConstantByRatesSchedule (≡ piecewise_constant_schedule_specified_by_rates)
  - DelayedConstantSchedule     (≡ delayed_constant_schedule)
  - JoinedSchedule              (≡ optax.join_schedules)

Rollout schedule:
  - ProgressiveRolloutSchedule  (PZHA progressive rollout)
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Learning rate schedule protocol
# ---------------------------------------------------------------------------
# All LR schedules are callables: __call__(step: int) -> float


@dataclasses.dataclass
class ProgressiveRolloutSchedule:
    """Progressive rollout length schedule based on training progress.

    At the beginning of training, short rollouts reduce memory and allow
    training on shorter contiguous data windows. The rollout length
    increases as the model improves.

    If the required data window (K × dt) exceeds available data,
    the rollout is truncated to fit.

    Attributes:
        milestones: sorted fractions of total steps where rollout increases.
        rollout_lengths: rollout K at each milestone (len = len(milestones) + 1).
        total_steps: total training steps.
    """

    total_steps: int
    milestones: tuple[float, ...] = (0.20, 0.50, 0.80)
    rollout_lengths: tuple[int, ...] = (4, 8, 16, 32)

    def __post_init__(self):
        if len(self.rollout_lengths) != len(self.milestones) + 1:
            raise ValueError(
                f"rollout_lengths ({len(self.rollout_lengths)}) must be "
                f"len(milestones)+1 ({len(self.milestones) + 1})"
            )

    def get_rollout(
        self,
        step: int,
        max_available: int | None = None,
    ) -> int:
        """Return the rollout length for the given step.

        Args:
            step: current training step.
            max_available: maximum rollout based on available data window.
                If provided, truncates to fit.

        Returns:
            rollout length K.
        """
        frac = step / max(self.total_steps, 1)
        idx = bisect.bisect_right(self.milestones, frac)
        k = self.rollout_lengths[idx]

        if max_available is not None:
            k = min(k, max_available)
        return max(k, 1)


# ---------------------------------------------------------------------------
# LR Schedules
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CosineDecaySchedule:
    """Cosine-decay learning rate with linear warmup.

    lr(t) =
      base_lr * t / warmup_steps,                          if t < warmup_steps
      min_lr + 0.5*(base_lr - min_lr)*(1 + cos(π*progress)), otherwise

    where progress = (t - warmup_steps) / (total_steps - warmup_steps).
    """

    base_lr: float = 1e-3
    warmup_steps: int = 1000
    total_steps: int = 100_000
    min_lr: float = 1e-6

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )


# Keep old name as alias for backward compatibility.
LearningRateSchedule = CosineDecaySchedule


@dataclasses.dataclass
class ExponentialDecaySchedule:
    """Exponential decay schedule (≡ optax.exponential_decay).

    lr(t) =
      init_value,                                                  if t < transition_begin
      max(init_value * decay_rate ^ ((t - transition_begin) / transition_steps), end_value),
                                                                   otherwise

    When staircase=True, the exponent is floored.
    """

    init_value: float = 1e-4
    transition_steps: int = 100_000
    decay_rate: float = 0.1
    transition_begin: int = 0
    staircase: bool = False
    end_value: float = 0.0

    def __call__(self, step: int) -> float:
        count = max(step - self.transition_begin, 0)
        if self.staircase:
            exponent = count // max(self.transition_steps, 1)
        else:
            exponent = count / max(self.transition_steps, 1)
        lr = self.init_value * (self.decay_rate ** exponent)
        return max(lr, self.end_value)


@dataclasses.dataclass
class WarmupExponentialDecaySchedule:
    """Linear warmup followed by exponential decay
    (≡ optax.warmup_exponential_decay_schedule).

    lr(t) =
      init_value + t/warmup_steps * (peak_value - init_value),   if t < warmup_steps
      peak_value,                    if warmup_steps <= t < warmup_steps + transition_begin
      max(peak_value * decay_rate ^ ((t - warmup_steps - transition_begin) / transition_steps),
          end_value),                otherwise

    When staircase=True, the exponent is floored.
    """

    init_value: float = 0.0
    peak_value: float = 1e-4
    warmup_steps: int = 1000
    transition_steps: int = 100_000
    decay_rate: float = 0.1
    transition_begin: int = 0
    staircase: bool = False
    end_value: float = 0.0

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            frac = step / max(self.warmup_steps, 1)
            return self.init_value + frac * (self.peak_value - self.init_value)
        # Exponential decay phase (count starts from end of warmup)
        count = max(step - self.warmup_steps - self.transition_begin, 0)
        if step < self.warmup_steps + self.transition_begin:
            return self.peak_value
        if self.staircase:
            exponent = count // max(self.transition_steps, 1)
        else:
            exponent = count / max(self.transition_steps, 1)
        lr = self.peak_value * (self.decay_rate ** exponent)
        return max(lr, self.end_value)


@dataclasses.dataclass
class PiecewiseConstantSchedule:
    """Piecewise constant schedule with multiplicative scales
    (≡ optax.piecewise_constant_schedule).

    lr(t) = init_value * ∏{b_i <= t} scales[b_i]

    boundaries_and_scales is a dict mapping step boundary → scale factor.
    """

    init_value: float = 1e-4
    boundaries_and_scales: dict[int, float] = dataclasses.field(
        default_factory=dict
    )

    def __call__(self, step: int) -> float:
        lr = self.init_value
        for boundary in sorted(self.boundaries_and_scales):
            if step >= boundary:
                lr *= self.boundaries_and_scales[boundary]
            else:
                break
        return lr


@dataclasses.dataclass
class PiecewiseConstantByRatesSchedule:
    """Piecewise constant schedule with absolute rates
    (≡ piecewise_constant_schedule_specified_by_rates).

    Segments defined by boundaries; each segment uses rates[i] directly.
    len(rates) == len(boundaries) + 1.
    """

    rates: Sequence[float] = (0.0, 1e-4)
    boundaries: Sequence[int] = (1000,)

    def __post_init__(self):
        if len(self.rates) != len(self.boundaries) + 1:
            raise ValueError(
                f"len(rates)={len(self.rates)} must be "
                f"len(boundaries)+1={len(self.boundaries) + 1}"
            )

    def __call__(self, step: int) -> float:
        idx = bisect.bisect_right(list(self.boundaries), step)
        return float(self.rates[idx])


@dataclasses.dataclass
class DelayedConstantSchedule:
    """Zero until turn_on_step, then constant rate
    (≡ delayed_constant_schedule).

    lr(t) = 0 if t < turn_on_step else rate.
    """

    turn_on_step: int = 0
    rate: float = 1e-4

    def __call__(self, step: int) -> float:
        return self.rate if step >= self.turn_on_step else 0.0


@dataclasses.dataclass
class JoinedSchedule:
    """Joins multiple schedules at given boundaries
    (≡ optax.join_schedules).

    Each sub-schedule receives a *relative* step count starting from 0
    at the beginning of its segment.
    len(schedules) == len(boundaries) + 1.
    """

    schedules: Sequence = dataclasses.field(default_factory=list)
    boundaries: Sequence[int] = dataclasses.field(default_factory=list)

    def __call__(self, step: int) -> float:
        idx = bisect.bisect_right(list(self.boundaries), step)
        offset = 0 if idx == 0 else self.boundaries[idx - 1]
        return float(self.schedules[idx](step - offset))
