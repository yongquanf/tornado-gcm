"""HPS Level-1: Hierarchical Sensitivity Profiler.

Measures per-module precision sensitivity via controlled perturbation:
  Phase 1 (Coarse): perturb each top-level module to BF16, measure drift.
  Phase 2 (Refine): for high-sensitivity modules, perturb sub-modules.

The result is a sensitivity map: module_name → float (drift magnitude),
which feeds into ZoneDiscovery for automatic zone partitioning.

Key design choices:
  - Physics-based sensitivity (conservation drift), NOT loss-based.
    This is the core differentiator vs HAWQ/AMP.
  - Black-box perturbation: no Hessian, no backward pass needed.
  - Hierarchical: coarse pass first, refine only sensitive modules.
  - Supports user-defined validator for custom physics metrics.

Usage:
    profiler = SensitivityProfiler(model, baseline_policy, validator=my_validator)
    smap = profiler.run_hierarchical(init_state, n_steps=200)
    # smap: {"parameterization": 0.001, "conservation_fixer": 0.95, ...}
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import torch
import torch.nn as nn

from pytorch_src.precision.policy import PrecisionPolicy, PrecisionZone
from pytorch_src.precision.monitor import PrecisionMonitor

if TYPE_CHECKING:
    from pytorch_src.core.primitive_equations import State

logger = logging.getLogger(__name__)

# ─── Validator protocol (user-extensible) ────────────────────────────────────

# A validator takes a State and returns a dict of named scalar metrics.
# Users can supply any callable matching this signature.
# For NeuralGCM: conservation quantities (energy, mass).
# For PDE solvers: residual norms.
# For climate models: radiation balance.
ValidatorFn = Callable[["State"], dict[str, float]]


class PhysicsValidator(Protocol):
    """Protocol for user-defined physics validators.

    Any callable ``(State) -> dict[str, float]`` satisfies this protocol.
    Implement this to customise HPS for non-NeuralGCM models.

    Example::

        class RadiationValidator:
            def __call__(self, state: State) -> dict[str, float]:
                return {"TOA_imbalance": compute_toa(state)}

        profiler = SensitivityProfiler(model, policy,
                                       validator=RadiationValidator())
    """

    def __call__(self, state: State) -> dict[str, float]: ...


def default_conservation_validator(state: State) -> dict[str, float]:
    """Default physics validator: conservation quantities.

    Measures total energy proxy (|vorticity|² + |divergence|²) and
    total mass proxy (sum of log_surface_pressure).
    """
    with torch.no_grad():
        vort = state.vorticity.detach().double()
        div = state.divergence.detach().double()
        lsp = state.log_surface_pressure.detach().double()

        energy_proxy = (vort * vort).sum().item() + (div * div).sum().item()
        mass_proxy = lsp.sum().item()

    return {
        "energy_proxy": energy_proxy,
        "mass_proxy": mass_proxy,
    }


def _hps_baseline_metrics_usable(metrics: dict[str, float]) -> bool:
    """True iff at least one reported metric is finite (baseline run is meaningful)."""
    for v in metrics.values():
        try:
            if math.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False


# ─── Sensitivity Result ─────────────────────────────────────────────────────


@dataclasses.dataclass
class ModuleSensitivity:
    """Sensitivity measurement for a single module."""

    module_name: str
    # Drift magnitude when this module is downcast to test_dtype.
    drift: float
    # Per-metric drifts (from validator).
    metric_drifts: dict[str, float] = dataclasses.field(default_factory=dict)
    # The dtype used for perturbation test.
    test_dtype: torch.dtype = torch.bfloat16
    # Wall-clock time for this measurement (seconds).
    elapsed_s: float = 0.0
    # Number of sub-modules (if refined).
    n_children_profiled: int = 0


@dataclasses.dataclass
class SensitivityMap:
    """Complete hierarchical sensitivity map."""

    # Module name → sensitivity measurement.
    modules: dict[str, ModuleSensitivity] = dataclasses.field(default_factory=dict)
    # Baseline metrics (all modules at baseline precision).
    baseline_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    # Configuration.
    n_steps: int = 0
    refine_threshold: float = 0.0
    total_elapsed_s: float = 0.0

    @property
    def sensitivity_ranking(self) -> list[tuple[str, float]]:
        """Return modules sorted by drift (highest first; non-finite last)."""
        items = [(name, ms.drift) for name, ms in self.modules.items()]

        def _rank_key(item: tuple[str, float]) -> tuple:
            try:
                d = float(item[1])
            except (TypeError, ValueError):
                return (2, 0.0)
            if not math.isfinite(d):
                return (1, 0.0)
            return (0, -d)

        return sorted(items, key=_rank_key)

    def get_zone_recommendation(self) -> dict[str, str]:
        """Simple zone recommendation based on sensitivity thresholds.

        Returns dict of module_name → recommended precision level:
          "critical"  (drift > 0.1):  FP64 required
          "moderate"  (drift > 0.001): FP32/TF32 safe
          "tolerant"  (drift <= 0.001): BF16 safe
          "invalid_baseline"  (drift not finite): unstable rollout — fix IC/forcing
        """
        rec = {}
        for name, ms in self.modules.items():
            try:
                d = float(ms.drift)
            except (TypeError, ValueError):
                rec[name] = "invalid_baseline"
                continue
            if not math.isfinite(d):
                rec[name] = "invalid_baseline"
            elif d > 0.1:
                rec[name] = "critical"
            elif d > 0.001:
                rec[name] = "moderate"
            else:
                rec[name] = "tolerant"
        return rec

    def summary_str(self) -> str:
        """Human-readable summary."""
        lines = [
            f"SensitivityMap: {len(self.modules)} modules, "
            f"{self.n_steps} steps, {self.total_elapsed_s:.1f}s total",
            f"Baseline metrics: {self.baseline_metrics}",
            "",
            "Ranking (highest sensitivity first):",
        ]
        for rank, (name, drift) in enumerate(self.sensitivity_ranking, 1):
            ms = self.modules[name]
            rec = self.get_zone_recommendation().get(name, "?")
            children_str = (
                f" ({ms.n_children_profiled} sub-modules)" if ms.n_children_profiled else ""
            )
            lines.append(f"  {rank}. {name}: drift={drift:.6e} [{rec}]{children_str}")
        return "\n".join(lines)


# ─── Sensitivity Profiler ────────────────────────────────────────────────────


class SensitivityProfiler:
    """Hierarchical module-level sensitivity profiler for HPS Level-1.

    Measures how precision-sensitive each module is by temporarily
    downcasting one module at a time and measuring the resulting
    conservation drift.

    Args:
        model: NeuralGCMModel (or any nn.Module with a step() method).
        baseline_policy: the precision policy to use as the baseline.
        validator: optional user-defined function State → dict[str, float].
            Default: conservation quantities (energy proxy, mass proxy).
        step_fn: optional custom stepping function. If None, uses model.step().
            Signature: (model, state, forcings) → state.
        device: torch device for profiling.
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_policy: PrecisionPolicy,
        validator: ValidatorFn | None = None,
        step_fn: Callable | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.baseline_policy = baseline_policy
        self.validator = validator or default_conservation_validator
        self.step_fn = step_fn
        self.device = device or next(model.parameters()).device

    # ── Core perturbation test ────────────────────────────────────────

    def _run_forward(
        self,
        state: State,
        n_steps: int,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> State:
        """Run model forward for n_steps, return final state."""
        step_fn = self.step_fn
        model = self.model
        model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                if step_fn is not None:
                    state = step_fn(model, state, forcings)
                else:
                    state = model.step(state, forcings=forcings)
        return state

    def _measure_drift(
        self,
        baseline_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Compute normalized drift between baseline and test metrics.

        Returns:
            (aggregate_drift, per_metric_drifts)
        """
        metric_drifts: dict[str, float] = {}
        for key in baseline_metrics:
            b_raw = baseline_metrics[key]
            t_raw = test_metrics.get(key, float("nan"))
            try:
                b = float(b_raw)
                t = float(t_raw)
            except (TypeError, ValueError):
                metric_drifts[key] = float("nan")
                continue
            if not math.isfinite(b):
                logger.debug(
                    "Drift skip: baseline metric %r is non-finite (%r)",
                    key,
                    b_raw,
                )
                metric_drifts[key] = float("nan")
                continue
            if not math.isfinite(t):
                metric_drifts[key] = float("inf")
                continue
            ref = abs(b) + 1e-30
            metric_drifts[key] = abs(t - b) / ref

        finite = [v for v in metric_drifts.values() if math.isfinite(v)]
        aggregate = max(finite) if finite else float("nan")
        if not finite and metric_drifts:
            logger.info(
                "Drift: no finite per-metric drifts (baseline or test metrics "
                "non-finite — HPS comparison skipped)."
            )
        return aggregate, metric_drifts

    # ── Module dtype patching ─────────────────────────────────────────

    def _get_top_level_modules(self) -> list[tuple[str, nn.Module]]:
        """Get the model's top-level (direct child) modules.

        Filters out trivial modules (no parameters, or internal bookkeeping).
        """
        modules = []
        for name, mod in self.model.named_children():
            # Skip non-parametric config-like attributes.
            if isinstance(mod, nn.Module):
                modules.append((name, mod))
        return modules

    def _get_submodules(self, parent_name: str) -> list[tuple[str, nn.Module]]:
        """Get direct children of a named module."""
        parent = dict(self.model.named_modules()).get(parent_name)
        if parent is None:
            return []
        return [
            (f"{parent_name}.{name}", mod)
            for name, mod in parent.named_children()
            if isinstance(mod, nn.Module)
        ]

    @torch.no_grad()
    def _downcast_module(
        self,
        module_name: str,
        target_dtype: torch.dtype,
    ) -> dict[str, torch.dtype]:
        """Temporarily downcast a module's parameters and buffers.

        Returns a dict of original dtypes for restoration.
        """
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            return {}

        original_dtypes: dict[str, torch.dtype] = {}

        for pname, param in module.named_parameters():
            full_name = f"{module_name}.{pname}"
            original_dtypes[full_name] = param.dtype
            param.data = param.data.to(target_dtype)

        for bname, buf in module.named_buffers():
            if buf is not None:
                full_name = f"{module_name}.{bname}"
                original_dtypes[full_name] = buf.dtype
                buf.data = buf.data.to(target_dtype)

        return original_dtypes

    @torch.no_grad()
    def _restore_module(
        self,
        module_name: str,
        original_dtypes: dict[str, torch.dtype],
    ) -> None:
        """Restore a module's parameters and buffers to original dtypes."""
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            return

        for pname, param in module.named_parameters():
            full_name = f"{module_name}.{pname}"
            if full_name in original_dtypes:
                param.data = param.data.to(original_dtypes[full_name])

        for bname, buf in module.named_buffers():
            if buf is not None:
                full_name = f"{module_name}.{bname}"
                if full_name in original_dtypes:
                    buf.data = buf.data.to(original_dtypes[full_name])

    # ── Profile a single module ───────────────────────────────────────

    def profile_module(
        self,
        module_name: str,
        test_dtype: torch.dtype,
        init_state: State,
        baseline_metrics: dict[str, float],
        n_steps: int,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> ModuleSensitivity:
        """Profile a single module by downcasting and measuring drift.

        Args:
            module_name: name of the module to perturb.
            test_dtype: dtype to temporarily cast the module to (e.g. BF16).
            init_state: initial model state (must be on self.device).
            baseline_metrics: metrics from the unperturbed baseline run.
            n_steps: number of forward steps.
            forcings: optional forcing data.

        Returns:
            ModuleSensitivity with drift measurement.
        """
        t0 = time.perf_counter()

        # Downcast the target module.
        original_dtypes = self._downcast_module(module_name, test_dtype)

        if not original_dtypes:
            # Module has no parameters/buffers — sensitivity = 0.
            return ModuleSensitivity(
                module_name=module_name,
                drift=0.0,
                test_dtype=test_dtype,
                elapsed_s=time.perf_counter() - t0,
            )

        try:
            # Run forward with perturbed module.
            final_state = self._run_forward(init_state, n_steps, forcings)
            test_metrics = self.validator(final_state)

            drift, metric_drifts = self._measure_drift(baseline_metrics, test_metrics)

            return ModuleSensitivity(
                module_name=module_name,
                drift=drift,
                metric_drifts=metric_drifts,
                test_dtype=test_dtype,
                elapsed_s=time.perf_counter() - t0,
            )

        except RuntimeError as e:
            # If downcast causes a runtime error (e.g., dtype mismatch in matmul),
            # that module is critically sensitive — infinite drift.
            logger.warning(
                "Module %s failed at %s: %s. Marking as critical.",
                module_name, test_dtype, e,
            )
            return ModuleSensitivity(
                module_name=module_name,
                drift=float("inf"),
                metric_drifts={},
                test_dtype=test_dtype,
                elapsed_s=time.perf_counter() - t0,
            )

        finally:
            # Always restore original dtypes.
            self._restore_module(module_name, original_dtypes)

    # ── Baseline run ──────────────────────────────────────────────────

    def run_baseline(
        self,
        init_state: State,
        n_steps: int,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Run the model with baseline policy and return validator metrics."""
        final_state = self._run_forward(init_state, n_steps, forcings)
        return self.validator(final_state)

    # ── Phase 1: Coarse profiling ─────────────────────────────────────

    def run_coarse(
        self,
        init_state: State,
        n_steps: int = 200,
        test_dtype: torch.dtype = torch.bfloat16,
        forcings: dict[str, torch.Tensor] | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> SensitivityMap:
        """Phase 1: Profile all top-level modules (coarse pass).

        Args:
            init_state: initial state on self.device.
            n_steps: forward steps per module test.
            test_dtype: dtype to perturb each module to.
            forcings: optional forcing data.
            baseline_metrics: if provided, skip baseline run.

        Returns:
            SensitivityMap with coarse-level sensitivity for each module.
        """
        t_total = time.perf_counter()

        # Baseline run (if not provided).
        if baseline_metrics is None:
            logger.info("Running baseline (%d steps)...", n_steps)
            baseline_metrics = self.run_baseline(init_state, n_steps, forcings)
        logger.info("Baseline metrics: %s", baseline_metrics)

        top_modules = self._get_top_level_modules()
        logger.info(
            "Phase 1 (coarse): profiling %d top-level modules, %d steps each",
            len(top_modules), n_steps,
        )

        smap = SensitivityMap(
            baseline_metrics=baseline_metrics,
            n_steps=n_steps,
        )

        for name, mod in top_modules:
            n_params = sum(p.numel() for p in mod.parameters())
            logger.info("  Profiling module '%s' (%d params) at %s ...", name, n_params, test_dtype)

            ms = self.profile_module(
                module_name=name,
                test_dtype=test_dtype,
                init_state=init_state,
                baseline_metrics=baseline_metrics,
                n_steps=n_steps,
                forcings=forcings,
            )
            smap.modules[name] = ms
            logger.info("    → drift=%.6e (%.1fs)", ms.drift, ms.elapsed_s)

        smap.total_elapsed_s = time.perf_counter() - t_total
        return smap

    # ── Phase 2: Refine high-sensitivity modules ──────────────────────

    def run_refine(
        self,
        coarse_map: SensitivityMap,
        init_state: State,
        refine_threshold: float = 0.01,
        n_steps: int | None = None,
        test_dtype: torch.dtype = torch.bfloat16,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> SensitivityMap:
        """Phase 2: Refine modules whose coarse drift > refine_threshold.

        For each high-sensitivity module, profile its direct children.
        The resulting map merges coarse (for low-sensitivity modules) and
        refined (for high-sensitivity modules) results.

        Args:
            coarse_map: Phase 1 output.
            init_state: initial state.
            refine_threshold: modules with drift above this are refined.
            n_steps: per-submodule test steps (default: same as coarse).
            test_dtype: perturbation dtype.
            forcings: optional forcing data.

        Returns:
            Merged SensitivityMap with refined entries where applicable.
        """
        if n_steps is None:
            n_steps = coarse_map.n_steps

        t_total = time.perf_counter()
        refined = SensitivityMap(
            baseline_metrics=coarse_map.baseline_metrics,
            n_steps=n_steps,
            refine_threshold=refine_threshold,
        )

        modules_to_refine = []
        for name, ms in coarse_map.modules.items():
            if ms.drift > refine_threshold:
                modules_to_refine.append(name)
            else:
                # Keep coarse result for low-sensitivity modules.
                refined.modules[name] = ms

        if not modules_to_refine:
            logger.info("Phase 2: no modules above threshold %.4f, skipping.", refine_threshold)
            refined.total_elapsed_s = time.perf_counter() - t_total
            return coarse_map  # Nothing to refine.

        logger.info(
            "Phase 2 (refine): %d modules above threshold %.4f: %s",
            len(modules_to_refine), refine_threshold, modules_to_refine,
        )

        for parent_name in modules_to_refine:
            children = self._get_submodules(parent_name)
            if not children:
                # No sub-modules to refine; keep the coarse result.
                refined.modules[parent_name] = coarse_map.modules[parent_name]
                continue

            logger.info(
                "  Refining '%s' → %d sub-modules",
                parent_name, len(children),
            )

            n_profiled = 0
            for child_name, child_mod in children:
                n_params = sum(p.numel() for p in child_mod.parameters())
                if n_params == 0:
                    continue
                logger.info("    Profiling '%s' (%d params)...", child_name, n_params)

                ms = self.profile_module(
                    module_name=child_name,
                    test_dtype=test_dtype,
                    init_state=init_state,
                    baseline_metrics=coarse_map.baseline_metrics,
                    n_steps=n_steps,
                    forcings=forcings,
                )
                refined.modules[child_name] = ms
                n_profiled += 1
                logger.info("      → drift=%.6e (%.1fs)", ms.drift, ms.elapsed_s)

            # Also keep parent entry with n_children_profiled metadata.
            parent_ms = dataclasses.replace(
                coarse_map.modules[parent_name],
                n_children_profiled=n_profiled,
            )
            refined.modules[parent_name] = parent_ms

        refined.total_elapsed_s = time.perf_counter() - t_total
        return refined

    # ── Full hierarchical profiling ───────────────────────────────────

    def run_hierarchical(
        self,
        init_state: State,
        n_steps: int = 200,
        refine_threshold: float = 0.01,
        test_dtype: torch.dtype = torch.bfloat16,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> SensitivityMap:
        """Run full 2-phase hierarchical sensitivity profiling.

        Phase 1: Coarse (all top-level modules).
        Phase 2: Refine (sub-modules of high-sensitivity modules).

        Args:
            init_state: initial model state.
            n_steps: forward steps per perturbation test.
            refine_threshold: drift threshold for Phase 2 refinement.
            test_dtype: dtype to perturb modules to (default BF16).
            forcings: optional forcing data.

        Returns:
            SensitivityMap with hierarchical sensitivity measurements.
        """
        logger.info("═" * 60)
        logger.info("HPS Level-1: Hierarchical Sensitivity Profiling")
        logger.info("  n_steps=%d, refine_threshold=%.4f, test_dtype=%s",
                     n_steps, refine_threshold, test_dtype)
        logger.info("═" * 60)

        # Phase 1: Coarse.
        coarse_map = self.run_coarse(
            init_state, n_steps, test_dtype, forcings,
        )
        logger.info("Phase 1 complete: %s", coarse_map.summary_str())

        # Phase 2: Refine.
        full_map = self.run_refine(
            coarse_map, init_state, refine_threshold, n_steps, test_dtype, forcings,
        )
        logger.info("Phase 2 complete: %s", full_map.summary_str())

        return full_map

    # ── Multi-dtype profiling ─────────────────────────────────────────

    def profile_module_multi_dtype(
        self,
        module_name: str,
        test_dtypes: list[torch.dtype],
        init_state: State,
        baseline_metrics: dict[str, float],
        n_steps: int,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list[ModuleSensitivity]:
        """Profile a module at multiple perturbation dtypes.

        Useful for determining the exact precision threshold:
          BF16 → FP32 → TF32 → FP64 sensitivity curve.
        """
        results = []
        for dtype in test_dtypes:
            ms = self.profile_module(
                module_name=module_name,
                test_dtype=dtype,
                init_state=init_state,
                baseline_metrics=baseline_metrics,
                n_steps=n_steps,
                forcings=forcings,
            )
            results.append(ms)
        return results

    # ── Zone-level policy perturbation ────────────────────────────────

    # Standard zone perturbation targets for NeuralGCM.
    # Each entry: (zone_name, {policy_field: True}, recast_params?)
    # When recast_params is True, model params are also cast to test_dtype.
    _ZONE_TARGETS: list[tuple[str, dict[str, bool], bool]] = [
        ("Z1_dynamics", {"z1_compute_dtype": True, "z1_storage_dtype": True}, False),
        ("Z2_fixer", {"z2_compute_dtype": True, "z2_storage_dtype": True}, False),
        ("Z3_neural", {"z3_compute_dtype": True, "z3_param_dtype": True}, True),
    ]

    def _save_param_dtypes(self) -> dict[str, torch.dtype]:
        """Save all parameter/buffer dtypes for later restoration."""
        saved: dict[str, torch.dtype] = {}
        for name, p in self.model.named_parameters():
            saved[f"param:{name}"] = p.dtype
        for name, b in self.model.named_buffers():
            if b is not None:
                saved[f"buffer:{name}"] = b.dtype
        return saved

    def _restore_param_dtypes(self, saved: dict[str, torch.dtype]) -> None:
        """Restore parameters/buffers to saved dtypes."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                key = f"param:{name}"
                if key in saved and p.dtype != saved[key]:
                    p.data = p.data.to(saved[key])
            for name, b in self.model.named_buffers():
                if b is not None:
                    key = f"buffer:{name}"
                    if key in saved and b.dtype != saved[key]:
                        b.data = b.data.to(saved[key])

    @torch.no_grad()
    def _recast_all_params(self, dtype: torch.dtype) -> None:
        """Cast all model parameters to the given dtype."""
        for p in self.model.parameters():
            p.data = p.data.to(dtype)

    def run_zone_perturbation(
        self,
        init_state: State,
        n_steps: int = 200,
        test_dtype: torch.dtype = torch.bfloat16,
        use_fp64_baseline: bool = True,
        forcings: dict[str, torch.Tensor] | None = None,
        zone_targets: list[tuple[str, dict[str, bool], bool]] | None = None,
    ) -> SensitivityMap:
        """Profile precision sensitivity at the *zone level* via policy swap.

        For NeuralGCM, the 4-zone precision is controlled by PrecisionPolicy,
        not by nn.Module parameters.  Module-level perturbation misses
        sensitivity in functional components (dycore, filter).

        This method creates an all-FP64 baseline, then lowers one zone at a
        time to *test_dtype* (default BF16) and measures conservation drift.

        Expected result for NeuralGCM T21:
          zone:Z2_fixer >> zone:Z1_dynamics > zone:Z3_neural
        which validates PZHA's precision assignment (Z2=FP64, Z1=FP32, Z3=BF16).

        Args:
            init_state: initial model state.
            n_steps: forward steps per test configuration.
            test_dtype: the lower-precision dtype to test (default: BF16).
            use_fp64_baseline: if True, baseline uses all-FP64.
                If False, uses the model's current policy as baseline.
            forcings: optional forcing data.
            zone_targets: custom [(name, {field: True}, recast_params)].
                Defaults to _ZONE_TARGETS (Z1, Z2, Z3).

        Returns:
            SensitivityMap with zone entries named "zone:Z1_dynamics", etc.
        """
        if not hasattr(self.model, "set_precision_policy"):
            raise AttributeError(
                "Model does not support set_precision_policy(). "
                "Zone perturbation requires a model with PrecisionPolicy."
            )

        t_total = time.perf_counter()
        targets = zone_targets or self._ZONE_TARGETS

        original_policy = self.model.policy
        saved_dtypes = self._save_param_dtypes()

        logger.info("═" * 60)
        logger.info("HPS Level-1z: Zone-level Sensitivity Profiling")
        logger.info("  n_steps=%d, test_dtype=%s, baseline=%s",
                     n_steps, test_dtype,
                     "all-FP64" if use_fp64_baseline else "current policy")
        logger.info("═" * 60)

        try:
            # ── Build baseline policy ──
            if use_fp64_baseline:
                baseline_policy = dataclasses.replace(
                    original_policy,
                    z0_dtype=torch.float64,
                    z1_compute_dtype=torch.float64,
                    z1_use_tf32=False,
                    z1_storage_dtype=torch.float64,
                    z2_compute_dtype=torch.float64,
                    z2_storage_dtype=torch.float64,
                    z3_compute_dtype=torch.float64,
                    z3_param_dtype=torch.float64,
                    z3_master_dtype=torch.float64,
                )
                self._recast_all_params(torch.float64)
            else:
                baseline_policy = original_policy

            # ── Baseline run ──
            self.model.set_precision_policy(baseline_policy)
            bl_z1 = baseline_policy.compute_dtype(PrecisionZone.Z1_DYNAMICS_CORE)
            bl_state = init_state.tree_map(lambda t: t.to(bl_z1))

            logger.info("Running zone baseline (%d steps, z1=%s)...", n_steps, bl_z1)
            baseline_final = self._run_forward(
                bl_state.tree_map(lambda t: t.clone()), n_steps, forcings,
            )
            baseline_metrics = self.validator(baseline_final)
            logger.info("Baseline metrics: %s", baseline_metrics)

            # ── Per-zone perturbation ──
            smap = SensitivityMap(
                baseline_metrics=baseline_metrics,
                n_steps=n_steps,
                refine_threshold=0.0,
            )

            if not _hps_baseline_metrics_usable(baseline_metrics):
                spec = tuple(init_state.vorticity.shape[-2:])
                logger.error(
                    "Skipping zone perturbation passes: baseline metrics are all "
                    "non-finite (rollout divergent or validator rejected). "
                    "Use a stable IC + matching ERA5 cache (modal %s), real "
                    "forcings, shorter --n-steps-list, or unset NEURALGCM_ERA5_OFFLINE.",
                    spec,
                )
                for zone_name, _, _ in targets:
                    smap.modules[f"zone:{zone_name}"] = ModuleSensitivity(
                        module_name=f"zone:{zone_name}",
                        drift=float("nan"),
                        metric_drifts={},
                        test_dtype=test_dtype,
                        elapsed_s=0.0,
                    )
                smap.total_elapsed_s = time.perf_counter() - t_total
                logger.info("Zone perturbation complete:\n%s", smap.summary_str())
                return smap

            for zone_name, field_overrides, needs_param_recast in targets:
                logger.info("  Perturbing zone '%s' → %s ...", zone_name, test_dtype)
                t0 = time.perf_counter()

                # Build perturbed policy: replace flagged fields with test_dtype.
                policy_kwargs = {
                    fname: test_dtype for fname, flag in field_overrides.items() if flag
                }
                perturbed_policy = dataclasses.replace(baseline_policy, **policy_kwargs)

                if needs_param_recast:
                    self._recast_all_params(test_dtype)
                self.model.set_precision_policy(perturbed_policy)

                # The working dtype is z1, which may have changed.
                z1_test = perturbed_policy.compute_dtype(PrecisionZone.Z1_DYNAMICS_CORE)
                test_init = init_state.tree_map(lambda t: t.to(z1_test))

                try:
                    test_final = self._run_forward(
                        test_init.tree_map(lambda t: t.clone()), n_steps, forcings,
                    )
                    test_metrics = self.validator(test_final)
                    drift, metric_drifts = self._measure_drift(
                        baseline_metrics, test_metrics,
                    )
                    ms = ModuleSensitivity(
                        module_name=f"zone:{zone_name}",
                        drift=drift,
                        metric_drifts=metric_drifts,
                        test_dtype=test_dtype,
                        elapsed_s=time.perf_counter() - t0,
                    )
                except RuntimeError as e:
                    logger.warning(
                        "Zone '%s' failed at %s: %s. Marking as critical.",
                        zone_name, test_dtype, e,
                    )
                    ms = ModuleSensitivity(
                        module_name=f"zone:{zone_name}",
                        drift=float("inf"),
                        test_dtype=test_dtype,
                        elapsed_s=time.perf_counter() - t0,
                    )

                smap.modules[f"zone:{zone_name}"] = ms
                logger.info("    → drift=%.6e (%.1fs)", ms.drift, ms.elapsed_s)

                # Restore baseline precision for next zone test.
                if needs_param_recast:
                    if use_fp64_baseline:
                        self._recast_all_params(torch.float64)
                    else:
                        self._restore_param_dtypes(saved_dtypes)
                self.model.set_precision_policy(baseline_policy)

            smap.total_elapsed_s = time.perf_counter() - t_total
            logger.info("Zone perturbation complete:\n%s", smap.summary_str())
            return smap

        finally:
            # Always restore original model state.
            self.model.set_precision_policy(original_policy)
            self._restore_param_dtypes(saved_dtypes)
