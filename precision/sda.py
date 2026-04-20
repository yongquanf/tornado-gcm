# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""SDA (Software-Defined Abstraction) mixed-precision framework.

Control-plane / data-plane architecture for automated precision management.

T1.1: Configuration dataclasses (ProfilerConfig, SchedulerConfig, etc.)
T1.5: SDAController — central orchestrator
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import field
from typing import TYPE_CHECKING, Any, Optional

import torch

from tornado_gcm.precision.policy import DEFAULT_POLICY, PrecisionPolicy, PrecisionZone

if TYPE_CHECKING:
    from tornado_gcm.precision.profiler import ProfilerMetrics, SDAProfiler
    from tornado_gcm.precision.scheduler import PrecisionScheduler, SchedulerDecision

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-module configuration dataclasses (T1.1)
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ProfilerConfig:
    """Configuration for SDAProfiler (precision + performance detection).

    Attributes:
        enabled: master switch for all profiling.
        granularity: profiling granularity — 'zone', 'module', or 'op'.
        sample_rate: collect metrics every N training steps.
        performance_counters: enable GPU kernel timing via torch.profiler.
        numerical_audit: enable precision-loss auditing at zone boundaries.
        gradient_audit: enable per-layer gradient statistics.
        spectral_audit: enable spectral energy tail monitoring.
        export_format: metric export format — 'tensorboard', 'json', 'wandb'.
    """

    enabled: bool = True
    granularity: str = "zone"
    sample_rate: int = 10
    performance_counters: bool = False
    numerical_audit: bool = True
    gradient_audit: bool = True
    spectral_audit: bool = False
    export_format: str = "tensorboard"


@dataclasses.dataclass
class SchedulerConfig:
    """Configuration for PrecisionScheduler (adaptive precision strategy).

    Attributes:
        mode: scheduling mode — 'milestone', 'loss_aware', 'stability_aware'.
        check_interval: run scheduling logic every N steps.
        loss_plateau_threshold: relative loss change below this → plateau.
        loss_plateau_patience: consecutive plateau detections before phase advance.
        nan_trigger_count: NaN occurrences that trigger precision escalation.
        grad_var_threshold: gradient variance threshold for stability mode.
        spectral_tail_threshold: high-frequency energy ratio threshold.
        warmup_steps_after_switch: gradual warm-up steps after hot-swap.
    """

    mode: str = "milestone"
    check_interval: int = 100
    loss_plateau_threshold: float = 0.01
    loss_plateau_patience: int = 3
    nan_trigger_count: int = 2
    grad_var_threshold: float = 100.0
    spectral_tail_threshold: float = 0.1
    warmup_steps_after_switch: int = 5
    rollout_ramp_steps: int = 0


@dataclasses.dataclass
class AcceleratorConfig:
    """Configuration for accelerator backend (Triton / torch.compile).

    Attributes:
        backend: backend selection — 'auto', 'triton', 'torch_compile', 'eager'.
        enable_fused_zone_cast: fuse multi-step dtype conversions (O9-1).
        enable_triton_sht: use Triton SHT kernels (experimental).
        enable_triton_fixers: use Triton conservation kernels (experimental).
        compile_mode: torch.compile mode — 'default', 'reduce-overhead', 'max-autotune'.
        compile_dynamic: allow dynamic shapes in torch.compile.
    """

    backend: str = "eager"
    enable_fused_zone_cast: bool = False
    enable_triton_sht: bool = False
    enable_triton_fixers: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False


@dataclasses.dataclass
class StorageConfig:
    """Configuration for mixed-precision storage.

    Attributes:
        mode: storage mode — 'exact', 'lossy', 'mixed'.
        exact_dtype: dtype for exact (lossless) variables.
        lossy_dtype: dtype for lossy-compressed variables.
        critical_variables: variables that always use exact storage.
        non_critical_variables: variables that can use lossy storage.
        compressor: compression algorithm — 'zstd', 'lz4', 'none'.
        compression_level: compression strength (1–9).
    """

    mode: str = "mixed"
    exact_dtype: torch.dtype = torch.float32
    lossy_dtype: torch.dtype = torch.bfloat16
    critical_variables: list[str] = field(
        default_factory=lambda: ["vorticity", "divergence", "log_surface_pressure"]
    )
    non_critical_variables: list[str] = field(
        default_factory=lambda: ["tracers"]
    )
    compressor: str = "zstd"
    compression_level: int = 3


@dataclasses.dataclass
class SparseConfig:
    """Configuration for sparse tensor mixed-precision.

    Attributes:
        enabled: master switch (experimental, default off).
        mode: sparsity strategy — 'structural', 'adaptive', 'mixed'.
        adaptive_threshold: absolute value below which coefficients are zeroed.
        high_order_cutoff: fraction of L above which coefficients count as high-order.
        high_order_dtype: dtype for high-order spectral coefficients.
        low_order_dtype: dtype for low-order spectral coefficients.
    """

    enabled: bool = False
    mode: str = "structural"
    adaptive_threshold: float = 1e-6
    high_order_cutoff: float = 0.5
    high_order_dtype: torch.dtype = torch.bfloat16
    low_order_dtype: torch.dtype = torch.float32


# ═══════════════════════════════════════════════════════════════════════════
# SDAConfig: top-level aggregate configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class SDAConfig:
    """Top-level SDA configuration — aggregates all sub-module configs.

    Load from YAML/JSON via ``SDAConfig.from_yaml(path)`` or construct directly.
    """

    policy: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SDAConfig:
        """Construct SDAConfig from a nested dict (e.g. parsed YAML)."""
        policy_d = d.get("policy", {})
        # Convert string dtype names to torch.dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        for k, v in list(policy_d.items()):
            if isinstance(v, str) and v in dtype_map:
                policy_d[k] = dtype_map[v]

        return cls(
            policy=PrecisionPolicy(**policy_d) if policy_d else PrecisionPolicy(),
            profiler=ProfilerConfig(**d.get("profiler", {})),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            accelerator=AcceleratorConfig(**d.get("accelerator", {})),
            storage=StorageConfig(**{
                k: v for k, v in d.get("storage", {}).items()
                if k not in ("exact_dtype", "lossy_dtype")
            }),
            sparse=SparseConfig(**{
                k: v for k, v in d.get("sparse", {}).items()
                if k not in ("high_order_dtype", "low_order_dtype")
            }),
        )

    @classmethod
    def from_yaml(cls, path: str) -> SDAConfig:
        """Load SDAConfig from a YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dict (for YAML export)."""
        dtype_name = {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.bfloat16: "bfloat16",
        }

        def _dc_to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                for fld in dataclasses.fields(obj):
                    val = getattr(obj, fld.name)
                    if isinstance(val, torch.dtype):
                        val = dtype_name.get(val, str(val))
                    elif dataclasses.is_dataclass(val):
                        val = _dc_to_dict(val)
                    result[fld.name] = val
                return result
            return obj

        return _dc_to_dict(self)


# ═══════════════════════════════════════════════════════════════════════════
# CompiledPolicy: runtime-optimized dtype lookup
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class CompiledPolicy:
    """Pre-compiled zone → dtype mapping for fast runtime lookup.

    Created by PolicyEngine.compile() — avoids dict lookup on every cast.
    """

    z0_compute: torch.dtype
    z0_storage: torch.dtype
    z1_compute: torch.dtype
    z1_storage: torch.dtype
    z2_compute: torch.dtype
    z2_storage: torch.dtype
    z3_compute: torch.dtype
    z3_storage: torch.dtype
    tf32_enabled: bool

    def compute_dtype(self, zone: PrecisionZone) -> torch.dtype:
        return {
            PrecisionZone.Z0_NUMERICAL_FOUNDATION: self.z0_compute,
            PrecisionZone.Z1_DYNAMICS_CORE: self.z1_compute,
            PrecisionZone.Z2_CONSERVATION_FIXER: self.z2_compute,
            PrecisionZone.Z3_NEURAL_NETWORK: self.z3_compute,
        }[zone]

    def storage_dtype(self, zone: PrecisionZone) -> torch.dtype:
        return {
            PrecisionZone.Z0_NUMERICAL_FOUNDATION: self.z0_storage,
            PrecisionZone.Z1_DYNAMICS_CORE: self.z1_storage,
            PrecisionZone.Z2_CONSERVATION_FIXER: self.z2_storage,
            PrecisionZone.Z3_NEURAL_NETWORK: self.z3_storage,
        }[zone]


# ═══════════════════════════════════════════════════════════════════════════
# SDAReport: summary of an SDA session
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class SDAReport:
    """Summary report produced by SDAController.report()."""

    total_steps: int = 0
    phase_transitions: list[dict[str, Any]] = field(default_factory=list)
    final_policy: Optional[PrecisionPolicy] = None
    profiler_summary: dict[str, float] = field(default_factory=dict)
    scheduler_decisions: int = 0
    warnings: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SDAController: control-plane orchestrator (T1.5)
# ═══════════════════════════════════════════════════════════════════════════


class SDAController:
    """Central SDA control-plane orchestrator.

    Coordinates PolicyEngine, SDAProfiler, and PrecisionScheduler.
    Call ``apply()`` once at init, then ``step()`` every training step.

    Args:
        config: SDAConfig containing all sub-module configurations.
    """

    def __init__(self, config: SDAConfig) -> None:
        from tornado_gcm.precision.profiler import SDAProfiler
        from tornado_gcm.precision.scheduler import PrecisionScheduler

        self.config = config
        self._policy = config.policy
        self._compiled: Optional[CompiledPolicy] = None
        self._step_count = 0
        self._start_time = 0.0
        self._phase_transitions: list[dict[str, Any]] = []
        self._decision_count = 0

        # Sub-components (control-plane)
        self.profiler = SDAProfiler(config.profiler)
        self.scheduler = PrecisionScheduler(config.scheduler, config.policy)

        # Sub-components (data-plane, lazy-init)
        self._accelerator: Optional[Any] = None  # AcceleratorRegistry
        self._storage: Optional[Any] = None       # SDAStorageManager
        self._fused_kernel: Optional[Any] = None   # FusedAdvanceKernel

    @property
    def policy(self) -> PrecisionPolicy:
        """Current active precision policy."""
        return self._policy

    @property
    def compiled(self) -> CompiledPolicy:
        """Current compiled policy for fast lookup."""
        if self._compiled is None:
            self._compiled = self.compile(self._policy)
        return self._compiled

    # ── lifecycle ──

    def apply(self) -> None:
        """Initialize runtime: compile policy, start profiler, configure TF32."""
        self._compiled = self.compile(self._policy)
        self._policy.apply_tf32_setting()
        self._init_accelerator()
        self._init_storage()
        self._start_time = time.monotonic()
        logger.info(
            "SDA applied: Z0=%s Z1=%s(tf32=%s) Z2=%s Z3=%s",
            self._policy.z0_dtype,
            self._policy.z1_compute_dtype,
            self._policy.z1_use_tf32,
            self._policy.z2_compute_dtype,
            self._policy.z3_compute_dtype,
        )

    def _init_accelerator(self) -> None:
        """Lazily initialise the accelerator registry from config."""
        if self.config.accelerator.backend == "eager":
            return
        try:
            from tornado_gcm.precision.accelerator import (
                AcceleratorRegistry,
                configure_dynamo,
                register_state_pytree,
            )
            self._accelerator = AcceleratorRegistry(
                default_backend=self.config.accelerator.backend,
            )
            if self.config.accelerator.backend in ("torch_compile", "auto"):
                register_state_pytree()
                configure_dynamo(suppress_errors=True)
            logger.info("Accelerator backend: %s", self.config.accelerator.backend)
        except ImportError:
            logger.debug("Accelerator package not available, using eager")

    def _init_storage(self) -> None:
        """Lazily initialise the storage manager from config."""
        try:
            from tornado_gcm.precision.storage import SDAStorageManager
            self._storage = SDAStorageManager(self.config.storage)
        except ImportError:
            logger.debug("Storage package not available")

    def get_fused_kernel(self, model: Any) -> Any:
        """Get or create a FusedAdvanceKernel for the given model (O9-1).

        Returns the kernel if accelerator backend supports compilation,
        otherwise returns None (caller should use model.step directly).
        """
        if not self.config.accelerator.enable_fused_zone_cast:
            return None
        if self._fused_kernel is not None:
            return self._fused_kernel
        try:
            from tornado_gcm.precision.accelerator import FusedAdvanceKernel
            self._fused_kernel = FusedAdvanceKernel(
                model,
                self._policy,
                compile_mode=self.config.accelerator.compile_mode,
            )
            return self._fused_kernel
        except ImportError:
            return None

    @property
    def storage(self) -> Optional[Any]:
        """SDAStorageManager instance (or None if not initialised)."""
        return self._storage

    @property
    def accelerator(self) -> Optional[Any]:
        """AcceleratorRegistry instance (or None if eager)."""
        return self._accelerator

    def step(self) -> None:
        """Per-step callback: collect metrics → schedule → hot-swap if needed."""
        self._step_count += 1
        self.profiler.step()

        if self._step_count % self.config.scheduler.check_interval != 0:
            return

        metrics = self.profiler.collect()
        decision = self.scheduler.decide(metrics, self._step_count)

        if decision.policy_changed and decision.new_policy is not None:
            self.hot_swap(decision.new_policy, reason=decision.reason)
            self._decision_count += 1

        if decision.rollout_changed:
            self._decision_count += 1

    def hot_swap(
        self,
        new_policy: PrecisionPolicy,
        *,
        reason: str = "",
    ) -> None:
        """Hot-swap precision policy at step boundary (no training restart)."""
        old_summary = (
            f"Z1={self._policy.z1_compute_dtype}, "
            f"Z3={self._policy.z3_compute_dtype}"
        )
        self._policy = new_policy
        self._compiled = self.compile(new_policy)
        new_policy.apply_tf32_setting()
        # Reset fused kernel (compiled graph is policy-specific)
        if self._fused_kernel is not None:
            self._fused_kernel.reset()
            self._fused_kernel = None
        new_summary = (
            f"Z1={new_policy.z1_compute_dtype}, Z3={new_policy.z3_compute_dtype}"
        )
        self._phase_transitions.append({
            "step": self._step_count,
            "from": old_summary,
            "to": new_summary,
            "reason": reason,
        })
        logger.info(
            "SDA hot-swap at step %d: %s → %s [%s]",
            self._step_count, old_summary, new_summary, reason,
        )

    def report(self) -> SDAReport:
        """Generate a session summary report."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return SDAReport(
            total_steps=self._step_count,
            phase_transitions=list(self._phase_transitions),
            final_policy=self._policy,
            profiler_summary=self.profiler.summary(),
            scheduler_decisions=self._decision_count,
            wall_time_seconds=elapsed,
        )

    # ── policy compilation ──

    @staticmethod
    def compile(policy: PrecisionPolicy) -> CompiledPolicy:
        """Compile a PrecisionPolicy into fast-lookup CompiledPolicy."""
        return CompiledPolicy(
            z0_compute=policy.z0_dtype,
            z0_storage=policy.z0_dtype,
            z1_compute=policy.z1_compute_dtype,
            z1_storage=policy.z1_storage_dtype,
            z2_compute=policy.z2_compute_dtype,
            z2_storage=policy.z2_storage_dtype,
            z3_compute=policy.z3_compute_dtype,
            z3_storage=policy.z3_param_dtype,
            tf32_enabled=policy.z1_use_tf32,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: default SDA config
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SDA_CONFIG = SDAConfig()
