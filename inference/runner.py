# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision inference runner.

Implements MPInference from PZHA:
  - NN params stored in BF16 (pre-converted offline)
  - Encoder (Assimilate) in BF16 → output FP32
  - Dycore in TF32 (Z1)
  - NN parameterization in BF16 (Z3) → output FP32
  - Conservation fixer in FP64 (Z2) → output FP32
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn

from tornado_gcm.model.api import NeuralGCMModel
from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.monitor import PrecisionMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: offload State / Tensor / arbitrary object to target device
# Ref: NeuralGCM REF-C device_put_to_cpu() async transfer pattern
# ---------------------------------------------------------------------------

def _offload(
    obj: Any,
    device: str | torch.device = "cpu",
    async_transfer: bool = True,
) -> Any:
    """Move *obj* to *device*, handling State, Tensor, or passthrough.

    When *async_transfer* is True and target is CPU, uses
    ``non_blocking=True`` for overlapped D2H copy (requires pinned
    memory on the source tensor for true async behaviour).
    """
    non_blocking = async_transfer and str(device) == "cpu"
    if hasattr(obj, "tree_map"):
        return obj.tree_map(lambda t: t.to(device, non_blocking=non_blocking))
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    return obj  # e.g. numpy array from post_process_fn


def _has_nan(obj: Any) -> bool:
    """Check whether *obj* (State, Tensor, or other) contains NaN."""
    if hasattr(obj, "tree_map"):
        # State-like: check all tensor fields
        for name in ("vorticity", "divergence", "temperature_variation",
                     "log_surface_pressure"):
            t = getattr(obj, name, None)
            if t is not None and torch.isnan(t).any():
                return True
        return False
    if isinstance(obj, torch.Tensor):
        return torch.isnan(obj).any().item()
    return False


def auto_estimate_chunk_size(
    model: NeuralGCMModel,
    sample_state: Any,
    headroom: float = 0.6,
) -> int:
    """Estimate safe chunk_size from free GPU memory.

    Strategy (aligned with NeuralGCM REF-C steps_per_unroll tuning):
      1. Measure single-State GPU bytes.
      2. Query ``torch.cuda.mem_get_info`` for free memory.
      3. ``chunk = int(free * headroom) // state_bytes``.
      4. Round down to nearest power-of-2, clamped to [1, 256].

    Returns 64 on CPU (no memory pressure).
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return 64

    if device.type != "cuda":
        return 64

    # Measure state size via its tensor fields
    state = sample_state.state if hasattr(sample_state, "state") else sample_state
    state_bytes = 0
    for name in ("vorticity", "divergence", "temperature_variation",
                 "log_surface_pressure"):
        t = getattr(state, name, None)
        if t is not None:
            state_bytes += t.nelement() * t.element_size()

    if state_bytes == 0:
        return 64

    free, _total = torch.cuda.mem_get_info(device)
    usable = int(free * headroom)
    raw = max(1, usable // state_bytes)

    # Round down to power of 2
    chunk = 1
    while chunk * 2 <= raw and chunk * 2 <= 256:
        chunk *= 2
    return chunk


class MPInference:
    """Mixed-precision inference engine.

    Follows PZHA Algorithm 2: each step executes
      1. Encode (BF16)  →  cast FP32
      2. Dycore advance (TF32)
      3. NN parameterization (BF16) → cast FP32
      4. Conservation fix (FP64) → cast FP32

    Args:
        model: trained NeuralGCMModel.
        policy: precision policy (default: standard PZHA).
        monitor: optional precision monitor for auditing.
    """

    def __init__(
        self,
        model: NeuralGCMModel,
        policy: PrecisionPolicy | None = None,
        monitor: PrecisionMonitor | None = None,
    ):
        self.model = model
        self.policy = policy or model.policy
        self.monitor = monitor

        # Set eval mode
        self.model.eval()

        # Apply TF32 setting for Z1
        self.policy.apply_tf32_setting()

    @torch.no_grad()
    def run(
        self,
        initial_state: Any,
        outer_steps: int,
        inner_steps: int | None = None,
        post_process_fn: Callable | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list[Any]:
        """Run autoregressive rollout.

        Args:
            initial_state: initial model state.
            outer_steps: number of saved output steps.
            inner_steps: substeps between saves.
            post_process_fn: optional function to transform each output.
            forcings: optional dict of forcing tensors (SST, sea_ice, etc.).

        Returns:
            list of (optionally post-processed) states.
        """
        _, trajectory = self.model(
            initial_state, outer_steps, inner_steps, forcings=forcings,
        )

        if post_process_fn is not None:
            trajectory = [post_process_fn(s) for s in trajectory]

        return trajectory

    @torch.no_grad()
    def run_chunked(
        self,
        initial_state: Any,
        total_steps: int,
        chunk_size: int = 100,
        inner_steps: int = 1,
        post_process_fn: Callable | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list[Any]:
        """Run inference in chunks to limit memory.

        Executes chunk_size outer steps at a time, yielding outputs
        progressively. Useful for long forecasts.

        Returns:
            list of all output states.
        """
        results = []
        state = initial_state
        remaining = total_steps

        while remaining > 0:
            n = min(chunk_size, remaining)
            state, chunk_traj = self.model(
                state, n, inner_steps, forcings=forcings,
            )
            if post_process_fn is not None:
                chunk_traj = [post_process_fn(s) for s in chunk_traj]
            results.extend(chunk_traj)
            remaining -= n

        return results

    @torch.no_grad()
    def run_streaming(
        self,
        initial_state: Any,
        outer_steps: int,
        inner_steps: int = 1,
        forcings: dict[str, torch.Tensor] | None = None,
        *,
        chunk_size: int = 1,
        keep_steps: set[int] | None = None,
        on_step_fn: Callable[[int, Any], None] | None = None,
        post_process_fn: Callable | None = None,
        offload_device: str | torch.device = "cpu",
        async_offload: bool = True,
        nan_guard: bool = False,
        checkpoint_interval: int = 0,
    ) -> tuple[Any, list[tuple[int, Any]]]:
        """Stream inference with chunk-wise GPU→CPU offload.

        Aligned with NeuralGCM REF-C InferenceRunner pipeline:
          1. GPU unroll *chunk_size* steps (= steps_per_unroll).
          2. Optional *on_step_fn* callback (on GPU, before offload).
          3. Filter by *keep_steps*.
          4. Offload kept states to CPU (async if available).
          5. Release GPU references.

        Args:
            chunk_size: outer steps per GPU unroll.  1 = generator path
                (lowest memory); 32-128 recommended for throughput.
            keep_steps: 1-indexed outer step indices to retain.
                ``None`` keeps all steps.
            on_step_fn: ``fn(global_step, state_on_gpu)`` called per step.
            post_process_fn: applied to each state before offload.
            offload_device: target device for saved states.
            async_offload: use ``non_blocking=True`` for D2H copies.
            nan_guard: if True, check each step output for NaN and
                revert to the last checkpoint if detected (REF-C §4b).
            checkpoint_interval: save a CPU checkpoint every N outer
                steps for NaN recovery.  0 = disabled.

        Returns:
            ``(final_state_on_gpu, saved)`` where
            ``saved = [(step_idx, offloaded_state), ...]``.
        """
        saved: list[tuple[int, Any]] = []
        state = initial_state
        global_step = 0
        remaining = outer_steps

        # NaN guard checkpoint state (on CPU) — REF-C §4b
        _ckpt_state: Any | None = None
        _ckpt_step: int = 0
        _nan_retries: int = 0
        _max_nan_retries: int = 3

        while remaining > 0:
            n = min(chunk_size, remaining)

            # ── Checkpoint save (REF-C §4b) ──
            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                _ckpt_state = _offload(state, "cpu", False)
                _ckpt_step = global_step

            if chunk_size == 1:
                # Generator path — lowest GPU memory.
                # Manually consume so we can capture the return value
                # (for-loop swallows StopIteration).
                gen = self.model.forward_streaming(
                    state, n, inner_steps, forcings=forcings,
                    post_process_fn=post_process_fn,
                )
                while True:
                    try:
                        output = next(gen)
                    except StopIteration as e:
                        if e.value is not None:
                            state = e.value
                        break
                    global_step += 1

                    # ── NaN guard ──
                    if nan_guard and _has_nan(output):
                        logger.warning(
                            "NaN detected at step %d", global_step)
                        _nan_retries += 1
                        if _ckpt_state is not None and _nan_retries <= _max_nan_retries:
                            device = next(self.model.parameters()).device
                            state = _offload(_ckpt_state, device, False)
                            remaining = outer_steps - _ckpt_step
                            global_step = _ckpt_step
                            # Discard saved states after checkpoint
                            saved = [(s, v) for s, v in saved
                                     if s <= _ckpt_step]
                            break
                        raise RuntimeError(
                            f"NaN at step {global_step} "
                            f"(retries={_nan_retries}, no viable checkpoint)"
                        )

                    if on_step_fn is not None:
                        on_step_fn(global_step, output)
                    if keep_steps is None or global_step in keep_steps:
                        saved.append((
                            global_step,
                            _offload(output, offload_device, async_offload),
                        ))
                else:
                    # Normal completion (while-else): update remaining
                    remaining -= n
                    continue
                # break from inner while → retry from checkpoint
                continue
            else:
                # Chunk path — batched unroll (REF-C steps_per_unroll)
                state, chunk_traj = self.model.forward_chunked(
                    state, n, inner_steps, forcings=forcings,
                    post_process_fn=post_process_fn,
                )
                nan_reverted = False
                for _i, output in enumerate(chunk_traj):
                    global_step += 1

                    # ── NaN guard ──
                    if nan_guard and _has_nan(output):
                        logger.warning(
                            "NaN detected at step %d", global_step)
                        _nan_retries += 1
                        if _ckpt_state is not None and _nan_retries <= _max_nan_retries:
                            device = next(self.model.parameters()).device
                            state = _offload(_ckpt_state, device, False)
                            remaining = outer_steps - _ckpt_step
                            global_step = _ckpt_step
                            saved = [(s, v) for s, v in saved
                                     if s <= _ckpt_step]
                            nan_reverted = True
                            break
                        raise RuntimeError(
                            f"NaN at step {global_step} "
                            f"(retries={_nan_retries}, no viable checkpoint)"
                        )

                    if on_step_fn is not None:
                        on_step_fn(global_step, output)
                    if keep_steps is None or global_step in keep_steps:
                        saved.append((
                            global_step,
                            _offload(output, offload_device, async_offload),
                        ))
                del chunk_traj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if nan_reverted:
                    continue  # retry from checkpoint

                remaining -= n

        return state, saved

    def audit(self) -> dict[str, Any]:
        """Return precision audit summary if monitor is attached."""
        if self.monitor is None:
            return {}
        return self.monitor.summary()
