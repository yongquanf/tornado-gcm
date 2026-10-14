# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""HiFPTuner/Precimonious-style Automatic Mixed-Precision Tuner for NeuralGCM.

Implements:
- Community detection (spectral clustering) for module grouping (HiFPTuner)
- Delta-debugging style precision search (Precimonious)
- Frozen static policy generation for comparison with Tornado-GCM

Usage:
    from tornado_gcm.precision.community_tuner import HiFPTuner
    tuner = HiFPTuner(model, init_state)
    policy, metadata = tuner.tune(n_tune_steps=20)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import torch

from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.sensitivity import SensitivityProfiler, SensitivityMap
from tornado_gcm.precision.zone_discovery import ModuleGraph, ZoneDiscovery

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HiFPTunerConfig:
    """Configuration for HiFPTuner-style tuning."""
    # Community detection
    use_community_detection: bool = True
    max_communities: int = 8
    alpha_affinity: float = 0.3

    # Precision search
    tune_tolerance: float = 1e-3
    candidate_precisions: tuple[torch.dtype, ...] = (
        torch.bfloat16, torch.float32, torch.float64)

    # Profiling
    n_tune_steps: int = 20


@dataclasses.dataclass
class HiFPTunerResult:
    """Result from HiFPTuner-style tuning."""
    tuned_policy: PrecisionPolicy
    community_structure: dict[int, list[str]]  # community_id -> module names
    search_trace: list[dict]
    tuning_time: float
    policy_per_zone: dict[str, str]  # zone -> dtype name


def _set_zone_dtype(policy: PrecisionPolicy, zone: PrecisionZone, dtype: torch.dtype) -> PrecisionPolicy:
    """Set compute dtype for a specific zone."""
    if zone == PrecisionZone.Z0_NUMERICAL_FOUNDATION:
        return dataclasses.replace(policy, z0_dtype=dtype)
    elif zone == PrecisionZone.Z1_DYNAMICS_CORE:
        return dataclasses.replace(policy, z1_compute_dtype=dtype)
    elif zone == PrecisionZone.Z2_CONSERVATION_FIXER:
        return dataclasses.replace(policy, z2_compute_dtype=dtype)
    elif zone == PrecisionZone.Z3_NEURAL_NETWORK:
        return dataclasses.replace(policy, z3_compute_dtype=dtype)
    return policy


def _dtype_name(dtype: torch.dtype) -> str:
    """Get human-readable dtype name."""
    mapping = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16",
        torch.float32: "FP32",
        torch.float64: "FP64",
    }
    return mapping.get(dtype, str(dtype))


def _compute_rmse_vs_gold(state, gold_state) -> float:
    """Compute RMSE between state and gold state."""
    def flatten_prognostics(s):
        parts = []
        for name in ["vorticity", "divergence", "temperature_variation",
                    "log_surface_pressure"]:
            t = getattr(s, name, None)
            if t is not None:
                parts.append(t.detach().double().reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(1, dtype=torch.float64)

    pred = flatten_prognostics(state)
    tgt = flatten_prognostics(gold_state)
    n = min(pred.numel(), tgt.numel())
    if n == 0:
        return 0.0
    return float(torch.sqrt(torch.mean((pred[:n] - tgt[:n]) ** 2)))


class HiFPTuner:
    """HiFPTuner-style automatic precision tuner for NeuralGCM.

    Follows HiFPTuner's approach:
    1. Profile module-level sensitivity
    2. Use spectral clustering to find communities
    3. Greedily lower precision per-community
    4. Freeze configuration for full rollout

    Also implements Precimonious-style delta-debugging search.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        init_state,
        config: Optional[HiFPTunerConfig] = None,
    ):
        self.model = model
        self.init_state = init_state
        self.config = config or HiFPTunerConfig()
        self._sensitivity_map: Optional[SensitivityMap] = None
        self._module_graph: Optional[ModuleGraph] = None

    def tune(
        self,
        n_tune_steps: int = 20,
        tolerance: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> tuple[PrecisionPolicy, HiFPTunerResult]:
        """Run full HiFPTuner-style tuning pipeline.

        Args:
            n_tune_steps: number of steps for tuning window
            tolerance: maximum acceptable drift
            device: torch device

        Returns:
            (PrecisionPolicy, HiFPTunerResult)
        """
        import time
        start_time = time.perf_counter()

        if device is None:
            device = next(self.model.parameters()).device

        # Step 1: Profile sensitivity
        logger.info("HiFPTuner: Step 1 - Profiling module sensitivity...")
        self._profile_sensitivity(n_tune_steps, device)

        # Step 2: Discover communities
        logger.info("HiFPTuner: Step 2 - Discovering communities...")
        communities = self._discover_communities()

        # Step 3: Delta-debugging precision search
        logger.info("HiFPTuner: Step 3 - Delta-debugging precision search...")
        policy, search_trace = self._delta_debug_search(
            n_tune_steps, tolerance, device)

        elapsed = time.perf_counter() - start_time
        logger.info("HiFPTuner: Tuning complete (%.2f s)", elapsed)

        result = HiFPTunerResult(
            tuned_policy=policy,
            community_structure=communities,
            search_trace=search_trace,
            tuning_time=elapsed,
            policy_per_zone={
                z.value: _dtype_name(policy.compute_dtype(z))
                for z in PrecisionZone
            }
        )

        return policy, result

    def _profile_sensitivity(
        self, n_steps: int, device: torch.device
    ):
        """Profile module-level sensitivity."""
        profiler = SensitivityProfiler(
            self.model, self.init_state, device=device)
        self._sensitivity_map = profiler.run_hierarchical(
            self.init_state, n_steps=n_steps)

        self._module_graph = ModuleGraph.for_neuralgcm()

    def _discover_communities(self) -> dict[int, list[str]]:
        """Discover module communities via spectral clustering."""
        if self._sensitivity_map is None:
            raise ValueError("Must profile sensitivity first")

        discovery = ZoneDiscovery(
            self._sensitivity_map,
            self._module_graph,
            alpha=self.config.alpha_affinity,
        )
        result = discovery.discover(max_zones=self.config.max_communities)

        return {
            za.zone_id: za.members
            for za in result.zones
        }

    def _delta_debug_search(
        self, n_tune_steps: int, tolerance: float, device: torch.device
    ) -> tuple[PrecisionPolicy, list[dict]]:
        """Precimonious-style delta-debugging precision search.

        Starts from full precision, greedily lowers per community/zone while accuracy holds.
        """
        # Start from full FP64
        policy = PrecisionPolicy(
            z0_dtype=torch.float64,
            z1_compute_dtype=torch.float64,
            z1_storage_dtype=torch.float64,
            z2_compute_dtype=torch.float64,
            z2_storage_dtype=torch.float64,
            z3_compute_dtype=torch.float64,
            z3_param_dtype=torch.float64,
        )

        search_trace = []

        # Get gold reference (full FP64)
        gold_result = self._rollout_with_policy(policy, n_tune_steps, device)
        gold_final = gold_result["final_state"]

        # Tune each tunable zone
        tunable_zones = [
            PrecisionZone.Z1_DYNAMICS_CORE,
            PrecisionZone.Z2_CONSERVATION_FIXER,
            PrecisionZone.Z3_NEURAL_NETWORK,
        ]

        for zone in tunable_zones:
            found_valid = False

            for dtype in self.config.candidate_precisions:
                trial_policy = _set_zone_dtype(policy, zone, dtype)
                result = self._rollout_with_policy(trial_policy, n_tune_steps, device)
                rmse = _compute_rmse_vs_gold(result["final_state"], gold_final)

                passed = (
                    torch.isfinite(torch.tensor(rmse)) and
                    rmse <= tolerance
                )

                search_trace.append({
                    "zone": zone.value,
                    "dtype": _dtype_name(dtype),
                    "rmse": float(rmse),
                    "passed": passed,
                })

                if passed:
                    policy = trial_policy
                    found_valid = True
                    logger.info(
                        "  Zone %s: set to %s (RMSE=%.3e)",
                        zone.value, _dtype_name(dtype), rmse)
                    break

            if not found_valid:
                logger.info(
                    "  Zone %s: kept at FP64 (no lower precision passed)",
                    zone.value)

        return policy, search_trace

    def _rollout_with_policy(
        self, policy: PrecisionPolicy, n_steps: int, device: torch.device
    ) -> dict:
        """Rollout model with given policy and return metrics."""
        self.model.set_precision_policy(policy)
        policy.apply_tf32_setting()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # We need to import this here to avoid circular imports
        from tornado_gcm.scripts.experiments.exp_utils import run_forecast_streaming

        saved = run_forecast_streaming(
            self.model, self.init_state, n_steps, inner_steps=1, keep_steps={n_steps})
        final = saved[-1] if saved else self.init_state

        return {"final_state": final}


class PrecimoniousTuner:
    """Precimonious-style delta-debugging tuner (without community detection).

    Simplified version that tunes per variable/module without grouping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        init_state,
    ):
        self.model = model
        self.init_state = init_state

    def tune(
        self,
        n_tune_steps: int = 20,
        tolerance: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> tuple[PrecisionPolicy, dict]:
        """Precimonious-style tuning: greedily lower precision without community detection.

        More aggressive per-component tuning without grouping.
        """
        import time
        start_time = time.perf_counter()

        if device is None:
            device = next(self.model.parameters()).device

        policy = PrecisionPolicy(
            z0_dtype=torch.float64,
            z1_compute_dtype=torch.float64,
            z1_storage_dtype=torch.float64,
            z2_compute_dtype=torch.float64,
            z2_storage_dtype=torch.float64,
            z3_compute_dtype=torch.float64,
            z3_param_dtype=torch.float64,
        )

        search_trace = []
        tunable_zones = [
            PrecisionZone.Z1_DYNAMICS_CORE,
            PrecisionZone.Z2_CONSERVATION_FIXER,
            PrecisionZone.Z3_NEURAL_NETWORK,
        ]

        # Gold reference
        gold_result = self._simple_rollout(policy, n_tune_steps, device)
        gold_final = gold_result["final_state"]

        for zone in tunable_zones:
            for dtype in [torch.bfloat16, torch.float32, torch.float64]:
                trial_policy = _set_zone_dtype(policy, zone, dtype)
                result = self._simple_rollout(trial_policy, n_tune_steps, device)
                rmse = _compute_rmse_vs_gold(result["final_state"], gold_final)
                passed = torch.isfinite(torch.tensor(rmse)) and rmse <= tolerance

                search_trace.append({
                    "zone": zone.value,
                    "dtype": _dtype_name(dtype),
                    "rmse": float(rmse),
                    "passed": passed,
                })

                if passed:
                    policy = trial_policy
                    break

        elapsed = time.perf_counter() - start_time

        metadata = {
            "search_trace": search_trace,
            "tuning_time": elapsed,
        }

        return policy, metadata

    def _simple_rollout(self, policy, n_steps, device):
        self.model.set_precision_policy(policy)
        policy.apply_tf32_setting()

        if device.type == "cuda":
            torch.cuda.synchronize()

        from tornado_gcm.scripts.experiments.exp_utils import run_forecast_streaming

        saved = run_forecast_streaming(
            self.model, self.init_state, n_steps, inner_steps=1, keep_steps={n_steps})

        return {"final_state": saved[-1] if saved else self.init_state}
