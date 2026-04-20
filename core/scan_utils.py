# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Utilities for nested multi-frequency scan (unrolling).

Implements the PyTorch equivalent of JAX's ``_recursive_scan`` and
``scan_utils`` for handling forcing data at multiple temporal frequencies.

Example: 6h atmospheric forcing + 24h ocean SST with dt=1h

    nested_steps = compute_nested_steps(
        frequencies={'atmos': 6, 'ocean': 24}, dt=1
    )
    # → (6, 4)  meaning 6×dt inner, 4×6h outer

    nested_forcings = nest_forcings(
        forcings={'atmos': atmos_6h, 'ocean': ocean_24h},
        frequencies={'atmos': 6, 'ocean': 24},
        dt=1,
    )
    # → [({}, steps=6), ({'atmos': ...}, steps=4), ({'ocean': ...}, steps=1)]

    trajectory = nested_scan(
        step_fn, state, nested_steps, nested_forcings, ...
    )
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint_util


# ═══════════════════════════════════════════════════════════════════════════════
# Step computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_nested_steps(
    frequencies: dict[str, int | float],
    dt: int | float = 1,
    total_steps: int | None = None,
) -> tuple[int, ...]:
    """Compute nested scan step counts from forcing frequencies.

    Sorts forcing groups by frequency (highest first = innermost loop),
    then computes how many steps each nesting level needs.

    Args:
        frequencies: mapping from group name to forcing interval (in units
            of model timesteps). E.g. ``{'atmos': 6, 'ocean': 24}`` means
            atmospheric forcing every 6 dt, ocean every 24 dt.
        dt: model timestep unit (default 1, meaning frequencies are already
            in units of dt).
        total_steps: optional total number of model steps. If provided,
            the outermost level is computed so that
            ``prod(nested_steps) == total_steps``.
            If None, only the *ratios* between frequency levels are returned
            (one fewer entry than with total_steps).

    Returns:
        Tuple of step counts from innermost to outermost.
        ``prod(nested_steps)`` equals ``total_steps`` when provided.

    Example:
        >>> compute_nested_steps({'atmos': 6, 'ocean': 24}, dt=1, total_steps=72)
        (6, 4, 3)
        # 6 dt-steps per atmos interval, 4 atmos intervals per ocean interval,
        # 3 ocean intervals to cover 72 steps total.

    Raises:
        ValueError: if frequencies are not congruent (higher must divide lower).
    """
    if not frequencies:
        if total_steps is not None:
            return (total_steps,)
        return ()

    # Sort by interval (smallest = highest frequency = innermost)
    sorted_items = sorted(frequencies.items(), key=lambda kv: kv[1])
    intervals = [int(v / dt) for _, v in sorted_items]

    # Build step counts: dt → first interval, first → second, ...
    steps: list[int] = []

    # Innermost: how many dt steps per smallest forcing interval
    prev = 1  # dt
    for interval in intervals:
        n, rem = divmod(interval, prev)
        if rem != 0:
            raise ValueError(
                f"Forcing intervals are not congruent: {interval} is not "
                f"divisible by {prev}. All intervals must be nested multiples."
            )
        steps.append(n)
        prev = interval

    # Outermost: remaining steps to fill total
    if total_steps is not None:
        inner_product = math.prod(steps)
        n_outer, rem = divmod(total_steps, inner_product)
        if rem != 0:
            raise ValueError(
                f"total_steps={total_steps} is not divisible by the product "
                f"of nested intervals ({inner_product}). "
                f"Intervals: {intervals}, dt={dt}."
            )
        steps.append(n_outer)

    return tuple(steps)


def frequency_groups(
    frequencies: dict[str, int | float],
    dt: int | float = 1,
) -> list[tuple[int, list[str]]]:
    """Group forcing variable names by their interval, sorted innermost first.

    Returns:
        List of (interval_in_dt, [var_names]) sorted by ascending interval.
    """
    groups: dict[int, list[str]] = {}
    for name, freq in frequencies.items():
        key = int(freq / dt)
        groups.setdefault(key, []).append(name)
    return sorted(groups.items(), key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Forcing nesting
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class NestedForcingLevel:
    """Forcing data for one nesting level.

    Attributes:
        steps: number of iterations at this level.
        var_names: variable names that update at this frequency.
        data: list of forcing dicts, one per step at this level.
              ``data[i]`` contains only the variables that change at this
              frequency. Length == ``steps``.
        stride: global step index stride for this level. E.g. if inner
                loop is 6 steps, the next level has stride=6.
    """
    steps: int
    var_names: list[str]
    data: list[dict[str, torch.Tensor]]
    stride: int


def nest_forcings(
    forcings: list[dict[str, torch.Tensor]],
    frequencies: dict[str, int | float],
    dt: int | float = 1,
    total_steps: int | None = None,
) -> list[NestedForcingLevel]:
    """Partition flat per-step forcing sequence into nested levels.

    Args:
        forcings: list of forcing dicts, one per global model step.
            Length must equal ``total_steps`` (or ``prod(nested_steps)``
            computed from frequencies).
        frequencies: mapping variable_group_name → interval in dt units.
        dt: model timestep.
        total_steps: total model steps (inferred from len(forcings) if None).

    Returns:
        List of ``NestedForcingLevel`` from innermost to outermost.
        The innermost level (index 0) has ``var_names=[]`` and empty data
        (no forcing changes at every dt — those are the pure model steps).

    Example:
        With dt=1, frequencies={'atmos': 6, 'ocean': 24}, total_steps=72:

        Level 0: steps=6, vars=[], data=[] (pure dt steps, no new forcing)
        Level 1: steps=4, vars=['atmos'], data=[atmos at 0h,6h,12h,18h]
        Level 2: steps=3, vars=['ocean'], data=[ocean at 0h,24h,48h]
    """
    if total_steps is None:
        total_steps = len(forcings)

    nested_steps = compute_nested_steps(frequencies, dt, total_steps)
    groups = frequency_groups(frequencies, dt)

    # Level 0: innermost dt steps (no forcing variables change here)
    levels: list[NestedForcingLevel] = [
        NestedForcingLevel(
            steps=nested_steps[0],
            var_names=[],
            data=[],
            stride=1,
        )
    ]

    # Levels 1..N-1: one per frequency group
    stride = nested_steps[0]
    for level_idx, (interval, var_names) in enumerate(groups, start=1):
        n_steps = nested_steps[level_idx] if level_idx < len(nested_steps) else 1
        # Extract forcing data at this frequency's cadence
        level_data: list[dict[str, torch.Tensor]] = []
        for step_i in range(n_steps):
            global_idx = step_i * stride
            if global_idx < len(forcings):
                src = forcings[global_idx]
                level_data.append({k: src[k] for k in var_names if k in src})
            else:
                level_data.append({})
        levels.append(NestedForcingLevel(
            steps=n_steps,
            var_names=var_names,
            data=level_data,
            stride=stride,
        ))
        stride *= n_steps

    # If total_steps requires an extra outermost level beyond frequency groups
    if len(nested_steps) > len(levels):
        remaining = nested_steps[len(levels)]
        levels.append(NestedForcingLevel(
            steps=remaining,
            var_names=[],
            data=[],
            stride=stride,
        ))

    return levels


# ═══════════════════════════════════════════════════════════════════════════════
# Nested scan execution
# ═══════════════════════════════════════════════════════════════════════════════


def nested_scan(
    step_fn: Callable[[Any, dict[str, torch.Tensor] | None], Any],
    initial_state: Any,
    levels: list[NestedForcingLevel],
    checkpoint_level: int = -1,
    collect_every: int = 1,
) -> tuple[Any, list[Any]]:
    """Execute nested multi-frequency scan.

    Recursively unrolls the model with forcing data injected at different
    temporal frequencies. This is the PyTorch equivalent of JAX's
    ``_recursive_scan``.

    Args:
        step_fn: ``step_fn(state, forcings) -> state``. Single model step.
            ``forcings`` is a dict containing all forcing variables merged
            from the current nesting context.
        initial_state: starting state.
        levels: nested forcing levels from ``nest_forcings()``.
            ``levels[0]`` is innermost (dt-level), ``levels[-1]`` is outermost.
        checkpoint_level: if >= 0, wrap iterations at this nesting level
            with ``torch.utils.checkpoint``. Level 0 = innermost.
            Default -1 = no checkpointing.
        collect_every: collect trajectory every N innermost steps.
            Default 1 = collect every step.

    Returns:
        ``(final_state, trajectory)`` where trajectory length depends on
        ``collect_every``.
    """
    trajectory: list[Any] = []
    # Mutable counter for global step index
    step_counter = [0]
    # Current merged forcing context (updated at each level)
    current_forcing: dict[str, torch.Tensor] = {}

    def _run_level(state: Any, level_idx: int) -> Any:
        """Recursively execute nesting level ``level_idx`` (outermost first)."""
        level = levels[level_idx]

        for i in range(level.steps):
            # Inject this level's forcing data into context
            if level.data and i < len(level.data):
                current_forcing.update(level.data[i])

            if level_idx > 0:
                # Recurse into inner level
                if checkpoint_level == level_idx:
                    state = checkpoint_util.checkpoint(
                        _run_level, state, level_idx - 1,
                        use_reentrant=False,
                    )
                else:
                    state = _run_level(state, level_idx - 1)
            else:
                # Innermost level: execute model step
                merged = dict(current_forcing) if current_forcing else None
                state = step_fn(state, merged)

                if collect_every == 1 or step_counter[0] % collect_every == 0:
                    trajectory.append(state)
                step_counter[0] += 1

        return state

    # Start from outermost level
    outermost = len(levels) - 1
    final_state = _run_level(initial_state, outermost)

    return final_state, trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: single-call nested rollout
# ═══════════════════════════════════════════════════════════════════════════════


def nested_rollout(
    step_fn: Callable[[Any, dict[str, torch.Tensor] | None], Any],
    initial_state: Any,
    forcings: list[dict[str, torch.Tensor]],
    frequencies: dict[str, int | float],
    dt: int | float = 1,
    checkpoint_level: int = -1,
    collect_every: int = 1,
) -> tuple[Any, list[Any]]:
    """Convenience wrapper: partition forcings + execute nested scan.

    Args:
        step_fn: ``step_fn(state, forcings) -> state``.
        initial_state: starting state.
        forcings: flat list of forcing dicts, one per model step.
        frequencies: variable_group → interval mapping.
        dt: model timestep.
        checkpoint_level: nesting level to checkpoint (-1 = none).
        collect_every: trajectory collection cadence.

    Returns:
        ``(final_state, trajectory)``.

    Example:
        >>> # 6h atmos + 24h ocean, model dt=1h, 72h rollout
        >>> forcings = [{'u': u_t, 'sst': sst_t} for t in range(72)]
        >>> final, traj = nested_rollout(
        ...     model.step, state, forcings,
        ...     frequencies={'u': 6, 'sst': 24},
        ...     dt=1, collect_every=6,
        ... )
    """
    total_steps = len(forcings)
    levels = nest_forcings(forcings, frequencies, dt, total_steps)
    return nested_scan(
        step_fn, initial_state, levels,
        checkpoint_level=checkpoint_level,
        collect_every=collect_every,
    )
