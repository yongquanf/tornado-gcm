"""Optimizer utilities for NeuralGCM training.

Implements:
  - multi_adam: per-parameter-group Adam with different learning rates
  - StopGradientTrajectory: gradient-cutting at specified outer steps
  - MixedPrecisionBackward: torch.autograd.Function for BF16 fwd / FP32 bwd
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn


def _normalize_param_name(name: str) -> str:
    """Strip common distributed wrapper prefixes from parameter names."""
    prefixes = (
        "module.",
        "_fsdp_wrapped_module.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def multi_adam(
    model: nn.Module,
    top_level_keys: Sequence[str] = (),
    learning_rates: Sequence[float] = (),
    default_learning_rate: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    raise_if_keys_not_found: bool = True,
) -> torch.optim.Optimizer:
    """Create Adam optimizer with per-parameter-group learning rates.

    Parameters are matched to groups by their top-level module name.
    Keys starting with 'REGEX_' are treated as regex patterns.

    Args:
        model: the model whose parameters to optimize.
        top_level_keys: parameter group identifiers (module name prefixes or REGEX_*).
        learning_rates: per-group learning rates (same length as top_level_keys).
        default_learning_rate: LR for parameters not matching any key.
        b1, b2, eps: Adam hyperparameters.
        weight_decay: L2 regularization.
        raise_if_keys_not_found: error if a key matches no parameters.

    Returns:
        torch.optim.AdamW optimizer with configured parameter groups.
    """
    if len(top_level_keys) != len(learning_rates):
        raise ValueError(
            f"top_level_keys ({len(top_level_keys)}) and "
            f"learning_rates ({len(learning_rates)}) must have same length"
        )

    # Build matcher functions
    matchers = []
    for key in top_level_keys:
        if key.startswith("REGEX_"):
            pattern = re.compile(key[6:])
            matchers.append(
                lambda name, p=pattern: p.search(name) is not None or p.search(_normalize_param_name(name)) is not None
            )
        else:
            matchers.append(
                lambda name, k=key: (
                    name.startswith(k + ".")
                    or name == k
                    or _normalize_param_name(name).startswith(k + ".")
                    or _normalize_param_name(name) == k
                )
            )

    # Assign parameters to groups
    groups: list[list[torch.nn.Parameter]] = [[] for _ in top_level_keys]
    default_group: list[torch.nn.Parameter] = []
    matched_keys = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assigned = False
        for i, matcher in enumerate(matchers):
            if matcher(name):
                groups[i].append(param)
                matched_keys.add(i)
                assigned = True
                break
        if not assigned:
            default_group.append(param)

    if raise_if_keys_not_found:
        for i, key in enumerate(top_level_keys):
            if i not in matched_keys:
                raise ValueError(f"Parameter group key '{key}' matched no parameters")

    param_groups = []
    for i, (key, lr) in enumerate(zip(top_level_keys, learning_rates)):
        if groups[i]:
            param_groups.append({
                "params": groups[i],
                "lr": lr,
                "name": key,
            })
    if default_group:
        param_groups.append({
            "params": default_group,
            "lr": default_learning_rate,
            "name": "default",
        })

    return torch.optim.AdamW(
        param_groups,
        lr=default_learning_rate,
        betas=(b1, b2),
        eps=eps,
        weight_decay=weight_decay,
    )


class StopGradientTrajectory:
    """Segments a rollout into legs with stop-gradient cuts between them.

    At each specified outer step, ``torch.detach()`` is applied to the state,
    preventing gradient flow across that boundary. This reduces memory usage
    and enables training with very long rollouts.

    Usage::

        sgt = StopGradientTrajectory(stop_gradient_outer_steps=[4, 12])
        trajectory = sgt.rollout(model, initial_state, total_outer_steps=20)
    """

    def __init__(self, stop_gradient_outer_steps: Sequence[int] = ()):
        self.stops = sorted(stop_gradient_outer_steps)

    def rollout(
        self,
        step_fn,
        initial_state,
        total_outer_steps: int,
        inner_steps: int = 1,
    ) -> list:
        """Run rollout with stop-gradient cuts.

        Args:
            step_fn: callable(state) -> state for one inner step.
            initial_state: starting state.
            total_outer_steps: total outer steps to produce.
            inner_steps: inner steps per outer step.

        Returns:
            list of states at each outer step.
        """
        boundaries = [0] + [s for s in self.stops if 0 < s < total_outer_steps] + [total_outer_steps]
        trajectory = []
        state = initial_state

        for leg_idx in range(len(boundaries) - 1):
            start = boundaries[leg_idx]
            end = boundaries[leg_idx + 1]

            # Detach at boundary (except the very first)
            if leg_idx > 0:
                state = _detach_state(state)

            for outer in range(start, end):
                for _ in range(inner_steps):
                    state = step_fn(state)
                trajectory.append(state)

        return trajectory


def _detach_state(state):
    """Detach all tensors in a state (supports State dataclass, dict, or tensor)."""
    from pytorch_src.core.primitive_equations import State
    if isinstance(state, State):
        return state.tree_map(lambda t: t.detach().requires_grad_(t.requires_grad))
    elif isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    elif isinstance(state, torch.Tensor):
        return state.detach().requires_grad_(state.requires_grad)
    return state


class MixedPrecisionBackward(torch.autograd.Function):
    """Custom autograd function: BF16 forward, FP32 backward (Algorithm 6).

    Wraps a module's forward pass so that:
      - Forward is computed in BF16 (or specified low-precision dtype)
      - Gradients are accumulated in FP32 for numerical stability

    Supports two modes via the ``mode`` argument:
      - ``"recompute"`` (default): saves FP32 input, recomputes the forward
        pass in FP32 during backward. Highest gradient accuracy, but 2×
        forward compute cost.
      - ``"stash"``: saves the FP32 input and computes backward through the
        BF16 forward graph without recomputation. ~1.0× forward cost, with
        slightly less accurate gradients due to BF16 activations.

    Usage::

        output = MixedPrecisionBackward.apply(module, input_tensor, torch.bfloat16, "recompute")
    """

    @staticmethod
    def forward(ctx, module, input_tensor, fwd_dtype=torch.bfloat16, mode="recompute"):
        ctx.module = module
        ctx.fwd_dtype = fwd_dtype
        ctx.mode = mode
        with torch.cuda.amp.autocast(enabled=False):
            if mode == "stash":
                # Stash mode: run BF16 forward with grad tracking so the
                # autograd graph is retained for backward.
                input_low = input_tensor.to(fwd_dtype).requires_grad_(True)
                output = module(input_low)
                ctx.save_for_backward(input_tensor, input_low, output)
                return output.float()
            else:
                # Recompute mode (original): only save input
                input_low = input_tensor.to(fwd_dtype)
                ctx.save_for_backward(input_tensor)
                output = module(input_low)
                return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        mode = ctx.mode
        if mode == "stash":
            input_tensor, input_low, output = ctx.saved_tensors
            # Backward through the BF16 graph, accumulate grads in FP32
            grad_out_fp32 = grad_output.float()
            output.backward(grad_out_fp32.to(output.dtype))
            # Compute input gradient via chain rule on the saved FP32 input
            input_grad = input_low.grad
            if input_grad is not None:
                input_grad = input_grad.float()
            return None, input_grad, None, None
        else:
            # Recompute mode: recompute forward in FP32 for accurate gradients
            (input_tensor,) = ctx.saved_tensors
            with torch.enable_grad():
                input_fp32 = input_tensor.float().requires_grad_(True)
                output = ctx.module(input_fp32)
                output.backward(grad_output.float())
            return None, input_fp32.grad, None, None
