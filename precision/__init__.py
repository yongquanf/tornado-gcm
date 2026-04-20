"""PZHA (Precision-Zoned Hybrid Architecture) precision management.

SDA (Software-Defined Abstraction) extensions:
  - SDAConfig / SDAController: control-plane orchestration
  - SDAProfiler: precision + performance detection
  - PrecisionScheduler: adaptive strategy dispatch
  - PolicyEngine: runtime policy compilation
"""

from pytorch_src.precision.policy import (
    DEFAULT_POLICY,
    PolicyEngine,
    PrecisionPolicy,
    PrecisionZone,
)
from pytorch_src.precision.zone_cast import zone_cast, zone_boundary, f64_math, einsum_highest
from pytorch_src.precision.monitor import PrecisionMonitor
from pytorch_src.precision.sda import (
    SDAConfig,
    SDAController,
    SDAReport,
    CompiledPolicy,
    ProfilerConfig,
    SchedulerConfig,
    AcceleratorConfig,
    StorageConfig,
    SparseConfig,
    DEFAULT_SDA_CONFIG,
)
from pytorch_src.precision.profiler import SDAProfiler, ProfilerMetrics, GradientStats
from pytorch_src.precision.scheduler import PrecisionScheduler, SchedulerDecision
from pytorch_src.precision.sensitivity import (
    SensitivityProfiler,
    SensitivityMap,
    ModuleSensitivity,
    ValidatorFn,
    default_conservation_validator,
)
from pytorch_src.precision.zone_discovery import (
    ZoneDiscovery,
    ModuleGraph,
    DiscoveryResult,
    ZoneAssignment,
    compare_with_pzha,
)

__all__ = [
    # Core PZHA
    "PrecisionPolicy",
    "PrecisionZone",
    "DEFAULT_POLICY",
    "zone_cast",
    "zone_boundary",
    "f64_math",
    "einsum_highest",
    "PrecisionMonitor",
    # SDA framework
    "SDAConfig",
    "SDAController",
    "SDAReport",
    "CompiledPolicy",
    "PolicyEngine",
    # SDA configs
    "ProfilerConfig",
    "SchedulerConfig",
    "AcceleratorConfig",
    "StorageConfig",
    "SparseConfig",
    "DEFAULT_SDA_CONFIG",
    # Profiler
    "SDAProfiler",
    "ProfilerMetrics",
    "GradientStats",
    # Scheduler
    "PrecisionScheduler",
    "SchedulerDecision",
    # HPS Level-1: Sensitivity
    "SensitivityProfiler",
    "SensitivityMap",
    "ModuleSensitivity",
    "ValidatorFn",
    "default_conservation_validator",
    # HPS Level-2: Zone Discovery
    "ZoneDiscovery",
    "ModuleGraph",
    "DiscoveryResult",
    "ZoneAssignment",
    "compare_with_pzha",
]
