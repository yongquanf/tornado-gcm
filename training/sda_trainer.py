# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""SDA-enhanced trainer with automated precision scheduling.

Extends MixedPrecisionTrainer with:
  - SDA control-plane integration (auto phase transitions)
  - Per-step detection (profiler) + scheduling (scheduler)
  - O9-2: per-layer backward precision hooks
  - O9-3: loss-aware rollout adaptation (via scheduler)
  - High-level fit() interface
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
import torch.nn as nn

from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.sda import SDAConfig, SDAController, SDAReport

logger = logging.getLogger(__name__)


class SDATrainer:
    """SDA-enhanced trainer with automated mixed-precision management.

    Wraps a MixedPrecisionTrainer, adding the SDA control loop:
    each train_step runs  profiler.collect → scheduler.decide → hot_swap.

    O9-2 (per-layer backward precision) is activated via
    ``configure_backward_precision()``.

    Args:
        model: the model to train.
        optimizer: optimizer (typically multi_adam with per-group LRs).
        loss_fn: loss module (e.g. PhysicsConstrainedLoss).
        sda_config: SDAConfig with all precision/profiler/scheduler settings.
        max_grad_norm: gradient clipping norm.
        ema_num_steps: EMA window size; decay = 1 - 2/(num_steps+1).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        sda_config: SDAConfig | None = None,
        max_grad_norm: float = 1.0,
        ema_num_steps: int = 1999,
    ) -> None:
        from tornado_gcm.training.trainer import MixedPrecisionTrainer

        self.sda_config = sda_config or SDAConfig()
        self.controller = SDAController(self.sda_config)

        # Wrap actual trainer
        self._trainer = MixedPrecisionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            policy=self.controller.policy,
            max_grad_norm=max_grad_norm,
            ema_num_steps=ema_num_steps,
        )

        self._rollout_steps: int = self.controller.scheduler.current_rollout
        self._backward_hooks: list[torch.utils.hooks.RemovableHook] = []

        # Apply initial policy
        self.controller.apply()

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def policy(self) -> PrecisionPolicy:
        return self.controller.policy

    @property
    def step_count(self) -> int:
        return self._trainer.step_count

    # ── Core train_step ──

    def train_step(
        self,
        batch: dict[str, Any],
        rollout_steps: int | None = None,
    ) -> dict[str, float]:
        """Enhanced train_step: SDA scheduling + profiling.

        1. Controller.step() — profiler collect → scheduler decide → hot_swap
        2. If scheduler changed rollout, override rollout_steps
        3. Delegate to MixedPrecisionTrainer.train_step()
        4. Record loss, audit gradients (if sample step)

        Args:
            batch: training batch dict.
            rollout_steps: override rollout length (None = use scheduler's).

        Returns:
            dict of scalar loss values.
        """
        # 1. SDA control loop
        self.controller.step()

        # Check for rollout/policy changes from scheduler
        decision = self.controller.scheduler.last_decision
        if decision.rollout_changed and decision.new_rollout is not None:
            self._rollout_steps = decision.new_rollout
        if decision.policy_changed and decision.new_policy is not None:
            self._trainer.policy = self.controller.policy

        # Tick warmup: apply deferred rollout change after warmup period
        warmup_decision = self.controller.scheduler.tick_warmup()
        if warmup_decision.rollout_changed and warmup_decision.new_rollout is not None:
            self._rollout_steps = warmup_decision.new_rollout

        # 2. Determine rollout
        k = rollout_steps if rollout_steps is not None else self._rollout_steps

        # 3. Forward/backward/update
        result = self._trainer.train_step(batch, rollout_steps=k)

        # 4. Profile this step
        loss_val = result.get("total", 0.0)
        self.controller.profiler.record_loss(loss_val)

        if self.controller.profiler.should_sample():
            self.controller.profiler.audit_gradient_stats(
                self.model.named_parameters()
            )

        return result

    # ── High-level fit() ──

    def fit(
        self,
        data_iter: Iterator[dict[str, Any]],
        total_steps: int,
        callbacks: Sequence[Callable] | None = None,
        log_interval: int = 100,
    ) -> SDAReport:
        """Train for ``total_steps``, with SDA auto-management.

        Args:
            data_iter: infinite iterator yielding batch dicts.
            total_steps: number of training steps.
            callbacks: optional list of callables(step, loss_dict).
            log_interval: logging frequency.

        Returns:
            SDAReport summarizing the session.
        """
        # Set total_steps on scheduler for milestone mode
        self.controller.scheduler.total_steps = total_steps

        for step in range(total_steps):
            batch = next(data_iter)
            loss_dict = self.train_step(batch)

            if callbacks:
                for cb in callbacks:
                    cb(step, loss_dict)

            if step % log_interval == 0:
                total_loss = loss_dict.get("total", 0.0)
                phase = self.controller.scheduler.current_phase + 1
                k = self._rollout_steps
                logger.info(
                    "step=%d loss=%.6f phase=%d rollout=%d",
                    step, total_loss, phase, k,
                )

        return self.controller.report()

    # ── O9-2: Per-layer backward precision hooks ──

    def configure_backward_precision(
        self,
        final_layer_keywords: Sequence[str] = ("output_layer", "head", "final"),
        high_precision_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Register backward hooks for high-precision gradients in final NN layers.

        This implements O9-2: the last 2-3 layers of the NN parameterization
        compute gradients in FP32 even when the forward pass uses BF16.

        Args:
            final_layer_keywords: module name substrings identifying final layers.
            high_precision_dtype: dtype for backward computation.
        """
        # Remove any existing hooks
        self.remove_backward_hooks()

        for name, module in self.model.named_modules():
            lower = name.lower()
            if any(kw in lower for kw in final_layer_keywords):

                def _hook(
                    mod: nn.Module,
                    grad_input: tuple,
                    grad_output: tuple,
                    dtype: torch.dtype = high_precision_dtype,
                ) -> tuple | None:
                    # Cast gradient outputs to high precision
                    new_grads = tuple(
                        g.to(dtype) if isinstance(g, torch.Tensor) and g is not None else g
                        for g in grad_input
                    )
                    return new_grads

                hook = module.register_full_backward_hook(_hook)
                self._backward_hooks.append(hook)
                logger.debug("O9-2 backward hook on: %s", name)

    def remove_backward_hooks(self) -> None:
        """Remove all O9-2 backward precision hooks."""
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks.clear()

    def swap_policy(self, phase: "TrainingPhaseConfig") -> None:
        """Hot-swap precision policy to match a TrainingPhaseConfig."""
        new_policy = PrecisionPolicy(
            z1_use_tf32=phase.dycore_tf32,
            z2_compute_dtype=phase.fixer_dtype,
            z3_compute_dtype=phase.nn_dtype,
            z3_param_dtype=phase.nn_dtype,
        )
        self.controller.hot_swap(new_policy, reason=f"swap_policy({phase.name})")
        self._trainer.policy = new_policy
        if hasattr(self._trainer.model, "set_precision_policy"):
            self._trainer.model.set_precision_policy(new_policy)
        self._rollout_steps = phase.rollout_steps

    # ── EMA / state dict ──

    def get_ema_state_dict(self) -> dict[str, torch.Tensor]:
        return self._trainer.get_ema_state_dict()
