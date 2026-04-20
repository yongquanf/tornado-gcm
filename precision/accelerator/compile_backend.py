# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""torch.compile backend configuration for SDA framework.

Handles:
  1. State dataclass registration as pytree (for torch.compile tracing)
  2. torch._dynamo configuration (suppress graph breaks, backends)
  3. TF32 / cuDNN settings
  4. Benchmark utilities for compile vs eager comparison

Usage:
    from tornado_gcm.precision.accelerator.compile_backend import (
        configure_dynamo, register_state_pytree, benchmark_compile,
    )
    register_state_pytree()        # call once before compile
    configure_dynamo(verbose=False)
    result = benchmark_compile(model, sample_input)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

_PYTREE_REGISTERED = False


def register_state_pytree() -> None:
    """Register State dataclass as a pytree node for torch.compile.

    Must be called once before any torch.compile invocation that
    processes State objects. Safe to call multiple times (idempotent).
    """
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return

    try:
        from torch.utils._pytree import register_pytree_node
    except ImportError:
        logger.debug("torch.utils._pytree not available, skipping registration")
        return

    try:
        from tornado_gcm.core.primitive_equations import State
    except ImportError:
        logger.debug("State not importable, skipping pytree registration")
        return

    def _state_flatten(state: State) -> tuple[list[Any], dict[str, Any]]:
        """Flatten State into (children, context)."""
        children = [
            state.vorticity,
            state.divergence,
            state.temperature_variation,
            state.log_surface_pressure,
        ]
        # Tracers as ordered items
        tracer_keys = sorted(state.tracers.keys()) if state.tracers else []
        children.extend(state.tracers[k] for k in tracer_keys)
        context = {
            "tracer_keys": tracer_keys,
            "sim_time": state.sim_time,
        }
        return children, context

    def _state_unflatten(children: list[Any], context: dict[str, Any]) -> State:
        """Reconstruct State from (children, context)."""
        tracer_keys = context["tracer_keys"]
        n_base = 4  # vorticity, divergence, temperature_variation, log_surface_pressure
        tracers = {k: children[n_base + i] for i, k in enumerate(tracer_keys)}
        return State(
            vorticity=children[0],
            divergence=children[1],
            temperature_variation=children[2],
            log_surface_pressure=children[3],
            tracers=tracers,
            sim_time=context["sim_time"],
        )

    register_pytree_node(State, _state_flatten, _state_unflatten)
    _PYTREE_REGISTERED = True
    logger.info("Registered State as pytree node for torch.compile")


def configure_dynamo(
    verbose: bool = False,
    suppress_errors: bool = True,
    cache_size_limit: int = 64,
) -> None:
    """Configure torch._dynamo settings for NeuralGCM compilation.

    Args:
        verbose: enable dynamo verbose logging.
        suppress_errors: fall back to eager on errors (recommended).
        cache_size_limit: max cached compiled graphs.
    """
    try:
        import torch._dynamo as dynamo

        dynamo.config.verbose = verbose
        dynamo.config.suppress_errors = suppress_errors
        dynamo.config.cache_size_limit = cache_size_limit
        logger.info(
            "Dynamo configured: verbose=%s suppress_errors=%s cache=%d",
            verbose, suppress_errors, cache_size_limit,
        )
    except ImportError:
        logger.warning("torch._dynamo not available, skipping configuration")


def configure_matmul_precision(
    allow_tf32: bool = True,
    allow_fp16_accumulation: bool = False,
    deterministic: bool = False,
) -> None:
    """Configure CUDA matmul and cuDNN precision settings.

    Args:
        allow_tf32: enable TF32 for matmul operations.
        allow_fp16_accumulation: allow FP16 accumulation (unsafe for GCM).
        deterministic: enable deterministic algorithms (slower).
    """
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        allow_fp16_accumulation
    )
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(
        "Matmul precision: tf32=%s fp16_accum=%s deterministic=%s",
        allow_tf32, allow_fp16_accumulation, deterministic,
    )


def benchmark_compile(
    fn: Callable,
    sample_args: tuple = (),
    sample_kwargs: Optional[dict] = None,
    compile_mode: str = "reduce-overhead",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> dict[str, float]:
    """Benchmark torch.compile vs eager execution.

    Args:
        fn: function to benchmark.
        sample_args: sample positional arguments.
        sample_kwargs: sample keyword arguments.
        compile_mode: torch.compile mode.
        warmup_iters: warmup iterations before timing.
        bench_iters: timed iterations.

    Returns:
        Dict with 'eager_ms', 'compiled_ms', 'speedup' keys.
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    # Eager benchmark
    for _ in range(warmup_iters):
        fn(*sample_args, **sample_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        fn(*sample_args, **sample_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / bench_iters * 1000

    # Compiled benchmark
    try:
        compiled_fn = torch.compile(fn, mode=compile_mode)

        for _ in range(warmup_iters):
            compiled_fn(*sample_args, **sample_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            compiled_fn(*sample_args, **sample_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compiled_ms = (time.perf_counter() - t0) / bench_iters * 1000
    except Exception as e:
        logger.warning("Compile benchmark failed: %s", e)
        compiled_ms = float("inf")

    speedup = eager_ms / compiled_ms if compiled_ms > 0 else 0.0

    return {
        "eager_ms": round(eager_ms, 3),
        "compiled_ms": round(compiled_ms, 3),
        "speedup": round(speedup, 3),
    }
