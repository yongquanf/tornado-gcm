# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""PrecisionMonitor: runtime precision diagnostics for PZHA.

Tracks zone boundary precision loss, conservation accuracy, and NaN rates.
Logs metrics for precision tuning and debugging.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PrecisionMonitor:
    """Runtime precision diagnostics for PZHA zones.

    Accumulates metrics at zone boundaries and conservation fixers.
    Call ``summary()`` periodically to inspect precision health.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._step = 0

    def step(self) -> None:
        """Increment the internal step counter."""
        self._step += 1

    @torch.no_grad()
    def audit_zone_transfer(
        self,
        name: str,
        x_high: torch.Tensor,
        x_low: torch.Tensor,
    ) -> None:
        """Audit precision loss at a zone boundary (downcast).

        Records max relative error and NaN rate.
        """
        if not self.enabled:
            return

        x_roundtrip = x_low.to(x_high.dtype)
        abs_high = x_high.abs()
        rel_error = (
            (x_high - x_roundtrip).abs() / (abs_high + 1e-30)
        ).max().item()
        nan_rate = x_low.isnan().float().mean().item()

        self._metrics[f"{name}/max_rel_error"].append(rel_error)
        self._metrics[f"{name}/nan_rate"].append(nan_rate)

        if abs_high.numel() > 0:
            dyn_range = (
                torch.log2(abs_high.max() + 1e-30)
                - torch.log2(abs_high[abs_high > 0].min() + 1e-30)
            ).item() if (abs_high > 0).any() else 0.0
            self._metrics[f"{name}/dynamic_range_bits"].append(dyn_range)

    @torch.no_grad()
    def audit_conservation(
        self,
        energy_before: torch.Tensor,
        energy_after: torch.Tensor,
        mass_before: torch.Tensor,
        mass_after: torch.Tensor,
    ) -> None:
        """Audit conservation fixer effectiveness.

        Records energy/mass drift in parts per million.
        """
        if not self.enabled:
            return

        e_drift_ppm = (
            1e6
            * (energy_after - energy_before).abs()
            / (energy_before.abs() + 1e-30)
        ).item()
        m_drift_ppm = (
            1e6
            * (mass_after - mass_before).abs()
            / (mass_before.abs() + 1e-30)
        ).item()

        self._metrics["conservation/energy_drift_ppm"].append(e_drift_ppm)
        self._metrics["conservation/mass_drift_ppm"].append(m_drift_ppm)

    def summary(self, last_n: int = 10) -> dict[str, float]:
        """Return a summary of recent metrics (mean of last_n entries)."""
        result = {}
        for key, values in self._metrics.items():
            recent = values[-last_n:]
            if recent:
                result[key] = sum(recent) / len(recent)
        return result

    def log_summary(self, last_n: int = 10) -> None:
        """Log a human-readable summary of precision metrics."""
        summary = self.summary(last_n)
        if not summary:
            return
        lines = [f"[PrecisionMonitor step={self._step}]"]
        for key, val in sorted(summary.items()):
            lines.append(f"  {key}: {val:.6g}")
        logger.info("\n".join(lines))

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._metrics.clear()
        self._step = 0
