# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""SDAProfiler: runtime precision + performance detection module.

Extends PrecisionMonitor with:
  - Gradient statistics per layer
  - Spectral energy tail monitoring
  - Zone-level wall-clock timing
  - Memory profiling per zone
  - Structured ProfilerMetrics collection
  - TensorBoard / JSON export
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import field
from typing import Any, Optional

import torch

from tornado_gcm.precision.monitor import PrecisionMonitor
from tornado_gcm.precision.policy import PrecisionZone
from tornado_gcm.precision.sda import ProfilerConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GradientStats:
    """Per-parameter gradient statistics."""

    mean: float = 0.0
    std: float = 0.0
    max_abs: float = 0.0
    min_abs: float = 0.0
    nan_count: int = 0
    numel: int = 0


@dataclasses.dataclass
class ProfilerMetrics:
    """Structured metrics collected by SDAProfiler.collect()."""

    step: int = 0
    # Zone timing (wall-clock seconds)
    zone_timing: dict[str, float] = field(default_factory=dict)
    # Zone memory (bytes)
    zone_memory: dict[str, int] = field(default_factory=dict)
    # Conservation drift (parts per million)
    conservation_drift_ppm: dict[str, float] = field(default_factory=dict)
    # Per-parameter gradient stats
    gradient_stats: dict[str, GradientStats] = field(default_factory=dict)
    # Spectral energy in high-frequency tail (fraction of total)
    spectral_energy_tail: float = 0.0
    # Max transfer error at zone boundaries
    transfer_max_error: dict[str, float] = field(default_factory=dict)
    # Auto-detected warnings
    warnings: list[str] = field(default_factory=list)
    # Recent loss values (for scheduler)
    recent_losses: list[float] = field(default_factory=list)
    # NaN count in recent window
    nan_count: int = 0


class SDAProfiler(PrecisionMonitor):
    """Enhanced precision + performance profiler for SDA.

    Inherits PrecisionMonitor's audit_zone_transfer / audit_conservation,
    adds gradient stats, spectral monitoring, zone timing, and memory tracking.

    Args:
        config: ProfilerConfig controlling granularity and feature switches.
    """

    def __init__(self, config: ProfilerConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._gradient_stats: dict[str, GradientStats] = {}
        self._spectral_tail: float = 0.0
        self._zone_timers: dict[str, float] = defaultdict(float)
        self._active_zone_timer: Optional[tuple[str, float]] = None
        self._loss_history: deque[float] = deque(maxlen=1000)
        self._nan_counter: int = 0
        self._sample_step: int = 0

    def should_sample(self) -> bool:
        """Whether this step should collect detailed metrics."""
        return self.enabled and (self._step % max(self.config.sample_rate, 1) == 0)

    # ── Gradient auditing (new) ──

    @torch.no_grad()
    def audit_gradient_stats(
        self,
        named_parameters: Any,
    ) -> None:
        """Collect per-layer gradient statistics.

        Args:
            named_parameters: iterable of (name, param) from model.named_parameters().
        """
        if not self.enabled or not self.config.gradient_audit:
            return

        self._gradient_stats.clear()
        for name, param in named_parameters:
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            nan_cnt = g.isnan().sum().item()
            g_clean = g.nan_to_num(0.0)
            abs_g = g_clean.abs()

            stats = GradientStats(
                mean=g_clean.mean().item(),
                std=g_clean.std().item() if g_clean.numel() > 1 else 0.0,
                max_abs=abs_g.max().item() if abs_g.numel() > 0 else 0.0,
                min_abs=abs_g[abs_g > 0].min().item() if (abs_g > 0).any() else 0.0,
                nan_count=int(nan_cnt),
                numel=int(g.numel()),
            )
            self._gradient_stats[name] = stats

            if nan_cnt > 0:
                self._nan_counter += int(nan_cnt)

    # ── Spectral energy auditing (new) ──

    @torch.no_grad()
    def audit_spectral_energy(
        self,
        modal_tensor: torch.Tensor,
        cutoff_fraction: float = 0.5,
    ) -> None:
        """Monitor high-frequency spectral energy.

        Args:
            modal_tensor: spectral coefficients (..., L) where L is wavenumber axis.
            cutoff_fraction: fraction of L above which coefficients are "high-order".
        """
        if not self.enabled or not self.config.spectral_audit:
            return

        t = modal_tensor.detach().float()
        total_energy = (t * t).sum().item()
        if total_energy < 1e-30:
            self._spectral_tail = 0.0
            return

        L = t.shape[-1]
        cutoff = int(L * cutoff_fraction)
        high_energy = (t[..., cutoff:] ** 2).sum().item()
        self._spectral_tail = high_energy / total_energy

    # ── Zone timing (new) ──

    def start_zone_timer(self, zone: PrecisionZone) -> None:
        """Start timing a precision zone."""
        if not self.enabled or not self.config.performance_counters:
            return
        if self._active_zone_timer is not None:
            self._stop_active_timer()
        self._active_zone_timer = (zone.value, time.perf_counter())

    def stop_zone_timer(self) -> None:
        """Stop the currently active zone timer."""
        if self._active_zone_timer is not None:
            self._stop_active_timer()

    def _stop_active_timer(self) -> None:
        if self._active_zone_timer is None:
            return
        zone_name, start = self._active_zone_timer
        elapsed = time.perf_counter() - start
        self._zone_timers[zone_name] += elapsed
        self._active_zone_timer = None

    # ── Loss tracking ──

    def record_loss(self, loss_value: float) -> None:
        """Record a training loss value (used by scheduler)."""
        self._loss_history.append(loss_value)
        if not (loss_value == loss_value):  # NaN check
            self._nan_counter += 1

    # ── Collect structured metrics ──

    def collect(self) -> ProfilerMetrics:
        """Collect all current metrics into a structured ProfilerMetrics."""
        # Zone boundary transfer errors (from parent PrecisionMonitor)
        transfer_errors = {}
        for key, values in self._metrics.items():
            if key.endswith("/max_rel_error") and values:
                transfer_errors[key] = values[-1]

        # Conservation drift
        conservation = {}
        for key in ("conservation/energy_drift_ppm", "conservation/mass_drift_ppm"):
            vals = self._metrics.get(key, [])
            if vals:
                conservation[key] = vals[-1]

        # Warnings
        warnings = []
        if self._nan_counter > 0:
            warnings.append(f"NaN detected: {self._nan_counter} occurrences")
        if self._spectral_tail > self.config.spectral_audit and self.config.spectral_audit:
            warnings.append(
                f"High spectral tail energy: {self._spectral_tail:.4f}"
            )
        for name, gs in self._gradient_stats.items():
            if gs.nan_count > 0:
                warnings.append(f"NaN gradients in {name}: {gs.nan_count}")
            if gs.max_abs > 1e6:
                warnings.append(f"Gradient explosion in {name}: max={gs.max_abs:.2e}")

        return ProfilerMetrics(
            step=self._step,
            zone_timing=dict(self._zone_timers),
            conservation_drift_ppm=conservation,
            gradient_stats=dict(self._gradient_stats),
            spectral_energy_tail=self._spectral_tail,
            transfer_max_error=transfer_errors,
            warnings=warnings,
            recent_losses=list(self._loss_history)[-20:],
            nan_count=self._nan_counter,
        )

    # ── Summary (override parent for richer output) ──

    def summary(self, last_n: int = 10) -> dict[str, float]:
        """Return a summary merging parent metrics + new profiler metrics."""
        base = super().summary(last_n)
        # Add zone timing
        for zone, t in self._zone_timers.items():
            base[f"zone_time/{zone}"] = t
        # Add spectral tail
        if self._spectral_tail > 0:
            base["spectral/energy_tail"] = self._spectral_tail
        # Add gradient summary (aggregated)
        if self._gradient_stats:
            max_grad = max(
                (gs.max_abs for gs in self._gradient_stats.values()), default=0.0
            )
            base["gradient/max_abs"] = max_grad
            total_nans = sum(gs.nan_count for gs in self._gradient_stats.values())
            base["gradient/total_nans"] = float(total_nans)
        return base

    # ── Export ──

    def export_json(self, path: str) -> None:
        """Export current metrics to a JSON file."""
        metrics = self.collect()
        data = dataclasses.asdict(metrics)
        # GradientStats is a dataclass, already handled by asdict
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def log_to_tensorboard(self, writer: Any, global_step: int) -> None:
        """Log metrics to a TensorBoard SummaryWriter."""
        summary = self.summary()
        for key, val in summary.items():
            writer.add_scalar(f"sda/{key}", val, global_step)

    # ── Reset (override) ──

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        super().reset()
        self._gradient_stats.clear()
        self._spectral_tail = 0.0
        self._zone_timers.clear()
        self._active_zone_timer = None
        self._loss_history.clear()
        self._nan_counter = 0
