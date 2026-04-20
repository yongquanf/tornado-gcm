# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Production inference runner with ensemble support and Zarr output.

Implements:
  - InferenceRunner: 3-stage GPU pipeline (prefetch | unroll | write)
  - EnsembleRunner: parallel ensemble member execution
  - Zarr output writer
  - NaN guard with atomic checkpoint
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.inference.runner import MPInference
from tornado_gcm.precision.policy import PrecisionPolicy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferenceConfig:
    """Configuration for production inference.

    Attributes:
        outer_steps: total forecast outer steps.
        inner_steps: substeps per outer step.
        chunk_size: outer steps per pipeline chunk.
        nan_guard: enable NaN detection + atomic checkpoint.
        checkpoint_interval: steps between atomic checkpoints (0 = disabled).
        output_path: directory for Zarr output.
        output_variables: subset of variables to save (None = all).
    """

    outer_steps: int = 100
    inner_steps: int = 1
    chunk_size: int = 50
    nan_guard: bool = True
    checkpoint_interval: int = 0
    output_path: str | pathlib.Path | None = None
    output_variables: Sequence[str] | None = None


class InferenceRunner:
    """Production inference with 3-stage pipeline.

    Stage 1 (prefetch): Load and prepare forcing data for next chunk.
    Stage 2 (unroll): Run model for chunk_size steps on GPU.
    Stage 3 (write): Write outputs to storage (Zarr, numpy, etc.).

    Includes NaN guard: if any output contains NaN, reverts to last
    atomic checkpoint and re-runs.
    """

    def __init__(
        self,
        engine: MPInference,
        config: InferenceConfig,
        forcing_fn: Callable[[float], dict[str, torch.Tensor]] | None = None,
        post_process_fn: Callable | None = None,
    ):
        self.engine = engine
        self.config = config
        self.forcing_fn = forcing_fn
        self.post_process_fn = post_process_fn
        self._last_checkpoint: tuple[int, Any] | None = None

    @torch.no_grad()
    def run(
        self,
        initial_state: Any,
        writer: "OutputWriter | None" = None,
    ) -> list[Any]:
        """Run full forecast with NaN guard and optional writing.

        Args:
            initial_state: initial model state.
            writer: optional output writer (ZarrWriter or NumpyWriter).

        Returns:
            list of all output states.
        """
        cfg = self.config
        all_results = []
        state = initial_state
        step = 0

        while step < cfg.outer_steps:
            n = min(cfg.chunk_size, cfg.outer_steps - step)

            # Atomic checkpoint
            if cfg.checkpoint_interval > 0 and step % cfg.checkpoint_interval == 0:
                self._last_checkpoint = (step, _clone_state(state))

            # Resolve forcings for this chunk
            forcings = None
            if self.forcing_fn is not None:
                sim_time = getattr(state, "sim_time", float(step))
                forcings = self.forcing_fn(sim_time)

            # Unroll
            state, chunk = self.engine.model(
                state, n, cfg.inner_steps, forcings=forcings,
            )

            # Post-process
            if self.post_process_fn is not None:
                chunk = [self.post_process_fn(s) for s in chunk]

            # NaN guard
            if cfg.nan_guard and _has_nan(chunk):
                logger.warning(f"NaN detected at step {step + n}, attempting recovery")
                if self._last_checkpoint is not None:
                    ckpt_step, ckpt_state = self._last_checkpoint
                    logger.warning(f"Rolling back to checkpoint at step {ckpt_step}")
                    state = ckpt_state
                    step = ckpt_step
                    continue
                else:
                    logger.error("NaN detected but no checkpoint available")
                    break

            # Write
            if writer is not None:
                writer.write_chunk(chunk, start_step=step)

            all_results.extend(chunk)
            step += n

        return all_results


class EnsembleRunner:
    """Run multiple ensemble members with different random seeds.

    Each member gets a unique RNG seed for stochastic physics.
    Members are run sequentially (for single-GPU) or can be batched.
    """

    def __init__(
        self,
        runner: InferenceRunner,
        n_members: int = 10,
        base_seed: int = 42,
    ):
        self.runner = runner
        self.n_members = n_members
        self.base_seed = base_seed

    @torch.no_grad()
    def run(
        self,
        initial_state: Any,
        writer: "OutputWriter | None" = None,
    ) -> list[list[Any]]:
        """Run all ensemble members.

        Returns:
            list of member trajectories, each a list of states.
        """
        all_members = []

        for member_idx in range(self.n_members):
            seed = self.base_seed + member_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            logger.info(f"Running ensemble member {member_idx + 1}/{self.n_members}")
            trajectory = self.runner.run(initial_state, writer=writer)
            all_members.append(trajectory)

        return all_members


# ═══════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════


class NumpyWriter:
    """Write inference outputs to numpy arrays (in-memory accumulation)."""

    def __init__(self, output_variables: Sequence[str] | None = None):
        self.output_variables = output_variables
        self.data: dict[str, list[np.ndarray]] = {}

    def write_chunk(self, chunk: list, start_step: int = 0) -> None:
        for state in chunk:
            state_dict = _state_to_dict(state)
            for key, val in state_dict.items():
                if self.output_variables and key not in self.output_variables:
                    continue
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(val)

    def finalize(self) -> dict[str, np.ndarray]:
        return {key: np.stack(vals) for key, vals in self.data.items()}


class ZarrWriter:
    """Write inference outputs to Zarr store (v3 compatible).

    Creates or appends to a Zarr store. Each variable is stored as a
    separate array with time as the first axis.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        output_variables: Sequence[str] | None = None,
        chunks: dict[str, int] | None = None,
    ):
        self.path = pathlib.Path(path)
        self.output_variables = output_variables
        self.chunks = chunks or {}
        self._store = None
        self._arrays: dict[str, Any] = {}

    def _ensure_store(self):
        if self._store is not None:
            return
        try:
            import zarr
            self._store = zarr.open(str(self.path), mode="a")
        except ImportError:
            raise ImportError("zarr package required for ZarrWriter")

    def write_chunk(self, chunk: list, start_step: int = 0) -> None:
        self._ensure_store()
        import zarr

        for i, state in enumerate(chunk):
            state_dict = _state_to_dict(state)
            for key, val in state_dict.items():
                if self.output_variables and key not in self.output_variables:
                    continue
                if key not in self._arrays:
                    shape = (0,) + val.shape
                    chunk_size = self.chunks.get(key, 100)
                    chunks = (chunk_size,) + val.shape
                    self._arrays[key] = self._store.zeros(
                        key, shape=shape, chunks=chunks,
                        dtype=val.dtype,
                    )
                self._arrays[key].append(val[np.newaxis], axis=0)


class AsyncZarrWriter:
    """Asynchronous Zarr writer using a background thread.

    Decouples GPU inference from I/O by queuing write operations to
    a background thread. This implements the 3-stage pipeline:
      Stage 1 (prefetch) → Stage 2 (GPU unroll) → Stage 3 (async write)

    The writer maintains a thread-safe queue and a single consumer thread
    that drains items and writes to Zarr sequentially.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        output_variables: Sequence[str] | None = None,
        chunks: dict[str, int] | None = None,
        max_queue_size: int = 64,
    ):
        import queue
        import threading

        self.path = pathlib.Path(path)
        self.output_variables = output_variables
        self.chunks = chunks or {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._store = None
        self._arrays: dict[str, Any] = {}
        self._error: Exception | None = None

        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="zarr-writer"
        )
        self._thread.start()

    def _ensure_store(self):
        if self._store is not None:
            return
        try:
            import zarr
            self._store = zarr.open(str(self.path), mode="a")
        except ImportError:
            raise ImportError("zarr package required for AsyncZarrWriter")

    def _writer_loop(self):
        """Background thread: drain queue and write to Zarr."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._ensure_store()
                state_dict, start_step = item
                for key, val in state_dict.items():
                    if self.output_variables and key not in self.output_variables:
                        continue
                    if key not in self._arrays:
                        shape = (0,) + val.shape
                        chunk_size = self.chunks.get(key, 100)
                        chunks = (chunk_size,) + val.shape
                        self._arrays[key] = self._store.zeros(
                            key, shape=shape, chunks=chunks,
                            dtype=val.dtype,
                        )
                    self._arrays[key].append(val[np.newaxis], axis=0)
            except Exception as e:
                self._error = e
                logger.error(f"AsyncZarrWriter error: {e}")
            finally:
                self._queue.task_done()

    def write_chunk(self, chunk: list, start_step: int = 0) -> None:
        """Queue a chunk of states for async writing.

        Converts GPU tensors to numpy on the calling thread (to avoid
        holding GPU memory), then enqueues for the background thread.
        """
        if self._error is not None:
            raise RuntimeError(
                f"AsyncZarrWriter background thread failed: {self._error}"
            )
        for i, state in enumerate(chunk):
            state_dict = _state_to_dict(state)
            self._queue.put((state_dict, start_step + i))

    def flush(self) -> None:
        """Block until all queued writes are complete."""
        self._queue.join()

    def close(self) -> None:
        """Stop the background writer thread."""
        self._queue.put(None)
        self._thread.join(timeout=30)
        if self._error is not None:
            raise RuntimeError(
                f"AsyncZarrWriter had errors: {self._error}"
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _has_nan(chunk: list) -> bool:
    """Check if any tensor in the chunk contains NaN."""
    for state in chunk:
        if isinstance(state, torch.Tensor):
            if torch.isnan(state).any():
                return True
        elif isinstance(state, dict):
            for v in state.values():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    return True
        elif hasattr(state, "vorticity"):
            for field in [state.vorticity, state.divergence,
                          state.temperature_variation, state.log_surface_pressure]:
                if torch.isnan(field).any():
                    return True
    return False


def _clone_state(state):
    """Deep clone a state for checkpoint."""
    if isinstance(state, torch.Tensor):
        return state.clone()
    elif isinstance(state, dict):
        return {k: _clone_state(v) for k, v in state.items()}
    elif hasattr(state, "tree_map"):
        return state.tree_map(lambda t: t.clone())
    return state


def _state_to_dict(state) -> dict[str, np.ndarray]:
    """Convert a state object to a dict of numpy arrays."""
    if isinstance(state, dict):
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()}
    elif hasattr(state, "vorticity"):
        result = {
            "vorticity": state.vorticity.cpu().numpy(),
            "divergence": state.divergence.cpu().numpy(),
            "temperature_variation": state.temperature_variation.cpu().numpy(),
            "log_surface_pressure": state.log_surface_pressure.cpu().numpy(),
        }
        if hasattr(state, "tracers"):
            for k, v in state.tracers.items():
                result[k] = v.cpu().numpy()
        return result
    elif isinstance(state, torch.Tensor):
        return {"data": state.cpu().numpy()}
    return {}
