# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark utilities for SDA framework.

Provides timing, memory, and storage benchmarks comparing different
precision configurations and backends.

Usage:
    from tornado_gcm.precision.benchmarks import TimingBenchmark
    bench = TimingBenchmark()
    bench.time_fn("eager_step", model.step, state, forcings)
    print(bench.report())
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)


class TimingBenchmark:
    """GPU/CPU timing benchmark with proper synchronisation.

    Args:
        warmup: number of warmup iterations.
        iterations: number of timed iterations.
        use_cuda_events: use CUDA events for GPU timing
            (more accurate than wall-clock on GPU).
    """

    def __init__(
        self,
        warmup: int = 3,
        iterations: int = 10,
        use_cuda_events: bool = True,
    ) -> None:
        self._warmup = warmup
        self._iterations = iterations
        self._use_cuda = use_cuda_events and torch.cuda.is_available()
        self._results: list[BenchmarkResult] = []

    def time_fn(
        self,
        name: str,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Time a function call.

        Args:
            name: benchmark label.
            fn: callable to benchmark.
            *args, **kwargs: arguments to fn.

        Returns:
            BenchmarkResult with timing statistics.
        """
        # Warmup
        for _ in range(self._warmup):
            fn(*args, **kwargs)

        if self._use_cuda:
            torch.cuda.synchronize()

        times_ms: list[float] = []

        for _ in range(self._iterations):
            if self._use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                fn(*args, **kwargs)
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                fn(*args, **kwargs)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000)

        import statistics

        result = BenchmarkResult(
            name=name,
            mean_ms=round(statistics.mean(times_ms), 3),
            std_ms=round(statistics.stdev(times_ms), 3) if len(times_ms) > 1 else 0.0,
            min_ms=round(min(times_ms), 3),
            max_ms=round(max(times_ms), 3),
            iterations=self._iterations,
        )
        self._results.append(result)
        return result

    def report(self) -> str:
        """Generate a formatted benchmark report."""
        lines = ["=" * 70, "Performance Benchmark Report", "=" * 70]
        for r in self._results:
            lines.append(
                f"  {r.name:<30s}  "
                f"mean={r.mean_ms:8.3f}ms  "
                f"std={r.std_ms:7.3f}ms  "
                f"min={r.min_ms:8.3f}ms  "
                f"max={r.max_ms:8.3f}ms  "
                f"(n={r.iterations})"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)


class MemoryBenchmark:
    """GPU memory usage benchmark.

    Measures peak memory allocation for a function call.
    """

    @staticmethod
    def measure(
        fn: Callable,
        *args: Any,
        device: str = "cuda",
        **kwargs: Any,
    ) -> dict[str, float]:
        """Measure peak memory for a function call.

        Args:
            fn: callable to measure.
            *args, **kwargs: arguments to fn.
            device: device to measure ('cuda' or 'cpu').

        Returns:
            Dict with memory stats in MB.
        """
        if device == "cuda" and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            before = torch.cuda.memory_allocated() / (1024 ** 2)
            fn(*args, **kwargs)
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

            return {
                "before_mb": round(before, 2),
                "after_mb": round(after, 2),
                "peak_mb": round(peak, 2),
                "delta_mb": round(after - before, 2),
            }
        else:
            # CPU: use tracemalloc
            import tracemalloc

            tracemalloc.start()
            fn(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                "current_mb": round(current / (1024 ** 2), 2),
                "peak_mb": round(peak / (1024 ** 2), 2),
            }


class StorageBenchmark:
    """I/O throughput benchmark for SDAStorageManager."""

    @staticmethod
    def measure_write_throughput(
        manager: Any,
        variables: dict[str, torch.Tensor],
        output_dir: str,
        iterations: int = 5,
    ) -> dict[str, float]:
        """Measure write throughput in MB/s.

        Args:
            manager: SDAStorageManager instance.
            variables: test variable dict.
            output_dir: temporary output directory.
            iterations: number of write iterations.

        Returns:
            Dict with throughput stats.
        """
        total_bytes = sum(
            t.nelement() * t.element_size() for t in variables.values()
        )

        times_ms: list[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            manager.write(variables, output_dir, step=i)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        import statistics

        mean_ms = statistics.mean(times_ms)
        throughput_mbs = (total_bytes / (1024 ** 2)) / (mean_ms / 1000) if mean_ms > 0 else 0

        return {
            "mean_write_ms": round(mean_ms, 3),
            "throughput_mb_s": round(throughput_mbs, 2),
            "total_data_mb": round(total_bytes / (1024 ** 2), 2),
        }
