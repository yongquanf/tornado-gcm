# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""HPS Level-2: Automatic Zone Discovery via Sensitivity Clustering.

Takes a SensitivityMap from Level-1 and produces an optimal zone partition:
  - Builds an affinity matrix from sensitivity similarity + module adjacency.
  - Uses spectral clustering (or agglomerative as fallback) to group modules.
  - Eigengap heuristic auto-selects the number of zones K.
  - Each zone gets a recommended precision based on its sensitivity centroid.

Theoretical backing:
  - Zone discovery = optimal quantization of sensitivity space with
    regularization for cast cost (Tanaka, 思辨 §Round 3).
  - Approximation ratio: CostRatio(Ẑ)/CostRatio(Z*) ≤ 1 + O(1/K).

Usage:
    from tornado_gcm.precision.sensitivity import SensitivityProfiler
    from tornado_gcm.precision.zone_discovery import ZoneDiscovery

    smap = profiler.run_hierarchical(init_state, n_steps=200)
    discovery = ZoneDiscovery(smap, module_graph)
    result = discovery.discover(max_zones=8)
    print(result.summary_str())
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import torch

from collections import Counter, defaultdict

from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.sensitivity import SensitivityMap

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DAG-based discovery (task_decomposition_v2.plan Phases A-F)
# ═══════════════════════════════════════════════════════════════════════════════

# Op → dtype constraint table (Phase B)
_DAG_SKIP_OPS = frozenset({
    "aten._assert_tensor_metadata.default",
    "aten.arange.default", "aten.remainder.Scalar",
    "aten.zeros.default", "aten.zeros_like.default",
    "aten.eq.Scalar", "aten.floor_divide.default",
    "aten.clone.default", "aten.copy_.default",
})

_DAG_VIEW_OPS = frozenset({
    "aten.unsqueeze.default", "aten.permute.default",
    "aten.reshape.default", "aten.select.int",
    "aten.slice.Tensor", "aten.movedim.int",
    "aten.narrow.default", "aten.expand.default",
})

_DAG_ZONE_BARRIER_OPS = frozenset({
    "aten.to.dtype", "aten.to.device",
})

_DAG_DTYPE_CONSTRAINTS: dict[str, frozenset[str]] = {
    # === Former per-op advisory constraints — downgraded to comments (U2a/b).
    # Measurement-driven assignment is the authority; per-op rules are kept
    # only as a historical record of conservative defaults, NOT enforced.
    #
    # All ops allow {BF16, FP32, FP64} by default.  Only the 3 hard ops in
    # _DAG_HARD_HIGH_PRECISION_OPS are restricted to {FP32, FP64} because
    # BF16 can produce SEMANTIC errors (wrong indices, flipped comparisons),
    # not just precision loss.
    #
    # Original advisory table (no longer enforced):
    #   "aten.einsum": frozenset({"BF16", "FP32", "FP64"}),
    #   "aten.gelu":   frozenset({"FP32", "FP64"}),   ← advisory, now relaxed
    #   "aten.pow":    frozenset({"FP32", "FP64"}),   ← advisory, now relaxed
    #   "aten.sqrt":   frozenset({"FP32", "FP64"}),   ← advisory, now relaxed
    #   "aten.exp":    frozenset({"FP32", "FP64"}),   ← advisory, now relaxed
    #   ...
    #   "aten.index_select": frozenset({"FP32", "FP64"}),  ← HARD, still enforced
    #   "aten.gt":          frozenset({"FP32", "FP64"}),  ← HARD, still enforced
    #   "aten.lt":          frozenset({"FP32", "FP64"}),  ← HARD, still enforced
    #   "aten.ne":          frozenset({"FP32", "FP64"}),  ← HARD, still enforced
}

# Ops that genuinely require ≥FP32 for correctness, NOT just numerical comfort.
# These are NEVER relaxed to BF16: index gathering needs exact indices, and
# comparisons produce boolean masks whose outcome can flip for near-equal
# BF16 operands.  This is the ONLY authoritative per-op constraint table.
_DAG_HARD_HIGH_PRECISION_OPS = frozenset({
    "aten.index_select",
    "aten.gt", "aten.lt", "aten.ne",
})

# Op → approximate BF16 sensitivity (Phase C fallback for unmeasured groups).
# These are initial estimates only; measured drift overrides them.
_DAG_OP_SENSITIVITY: dict[str, float] = {
    "aten.einsum": 1e-4, "aten.matmul": 1e-4, "aten.bmm": 1e-4,
    "aten.linear": 5e-5, "aten.gelu": 1e-3, "aten.pow": 1e-4,
    "aten.sqrt": 1e-5, "aten.exp": 1e-3, "aten.log": 1e-4,
    "aten.reciprocal": 1e-5, "aten.index_select": 1e-6,
    "aten.narrow": 1e-6, "aten.mul": 1e-7, "aten.add": 1e-7,
    "aten.sub": 1e-7, "aten.neg": 1e-7, "aten.cat": 1e-7,
    "aten.stack": 1e-7, "aten.ones": 1e-7, "aten.zeros": 1e-7,
    "aten.triu": 1e-7, "aten.sum": 1e-6, "aten.where": 1e-6,
    "aten.clamp": 1e-6, "aten.sign": 1e-7, "aten.div": 1e-6,
    "aten.gt": 1e-7, "aten.lt": 1e-7, "aten.ne": 1e-7, "aten.eq": 1e-7,
}


def _dag_get_allowed_dtypes(op_target: str) -> frozenset[str]:
    """Return the allowed dtype set for *op_target*.

    U2a/b: measurement-driven assignment.  All ops allow {BF16, FP32, FP64}
    by default.  Only ops in ``_DAG_HARD_HIGH_PRECISION_OPS`` are restricted
    to {FP32, FP64} (semantic correctness, not numerical comfort).
    """
    if any(op_target.startswith(p) for p in _DAG_HARD_HIGH_PRECISION_OPS):
        return frozenset({"FP32", "FP64"})
    return frozenset({"BF16", "FP32", "FP64"})


def _dag_get_op_sensitivity(op_target: str) -> float:
    for prefix, s in _DAG_OP_SENSITIVITY.items():
        if op_target.startswith(prefix):
            return s
    return 1e-7


_DAG_DTYPE_RANK = {
    torch.bfloat16: 0, torch.float16: 0, torch.float32: 1, torch.float64: 2,
}


def _dag_node_module_path(node):
    """Deepest nn.Module path that produced *node*, from its ``nn_module_stack``
    FX metadata.  Aligns with ``model.named_modules()`` keys.  Returns None for
    nodes with no module attribution (graph-level / functional-only ops)."""
    meta = getattr(node, "meta", None)
    if not meta:
        return None
    stk = meta.get("nn_module_stack")
    if not stk:
        return None
    paths = []
    for v in stk.values():
        p = v[0] if isinstance(v, (tuple, list)) and v else None
        if isinstance(p, str) and p:
            paths.append(p)
    return paths[-1] if paths else None


def _dag_leaf_matches(group_modpath: str, leaf_path: str) -> bool:
    """Whether a DAG group attributed to *group_modpath* should contribute its
    dtype to leaf module *leaf_path*.  Functional ops attribute to the parent
    module, so a leaf nested under the group's module also matches."""
    return (
        group_modpath == leaf_path
        or leaf_path.startswith(group_modpath + ".")
        or group_modpath.startswith(leaf_path + ".")
    )


def _dag_resolve_leaf_dtypes(root_dtype, group_to_modules, leaf_paths,
                             leaf_to_subzone, sz_resolved, budget):
    """Per-leaf dtype with NO sub-zone collapse.

    Each leaf module aggregates only the dtypes of the DAG groups whose
    ``nn_module_stack`` path matches it (MIN for unconstrained budget, MAX for
    constrained).  Two leaves in the same sub-zone may therefore differ.
    Leaves with no matching group fall back to the sub-zone aggregate.
    """
    mod_dtypes = defaultdict(list)
    for root, dt in root_dtype.items():
        for mp in group_to_modules.get(root, ()):  # set of module paths
            mod_dtypes[mp].append(dt)

    use_min = budget == float("inf")
    rank = lambda dt: _DAG_DTYPE_RANK.get(dt, 1)
    resolved = {}
    for lp in leaf_paths:
        dts = []
        for mp, ds in mod_dtypes.items():
            if _dag_leaf_matches(mp, lp):
                dts.extend(ds)
        if dts:
            resolved[lp] = min(dts, key=rank) if use_min else max(dts, key=rank)
        else:
            sz = leaf_to_subzone.get(lp, "Z1_DYNAMICS")
            fallback = sz_resolved.get(sz)
            if fallback is not None:
                resolved[lp] = fallback
    return resolved


def _dag_distribute_zone_sensitivity(group_complexity, group_subzone, zone_drift):
    """Distribute each sub-zone's MEASURED BF16 drift across its fine groups by
    op-complexity share.

    Replaces the op-type sensitivity heuristic: a group's sensitivity is the
    measured zone-level drift weighted by how much of the sub-zone's total
    compute (objective op counts) that group performs.  No hand-tuned per-op
    weights — only measured drift × op-count ratios.

    Args:
        group_complexity: {root: float op-count}
        group_subzone:    {root: sub-zone label}
        zone_drift:       {sub-zone label: measured BF16 drift}
    Returns:
        {root: sensitivity}.  Groups in sub-zones absent from *zone_drift* get
        0.0 (the caller may override with a heuristic fallback).
    """
    sz_groups = defaultdict(list)
    for root, sz in group_subzone.items():
        sz_groups[sz].append(root)
    out = {}
    for sz, roots in sz_groups.items():
        drift = zone_drift.get(sz, 0.0)
        total = sum(group_complexity.get(r, 0.0) for r in roots)
        n = len(roots)
        for r in roots:
            if total > 0:
                out[r] = drift * (group_complexity.get(r, 0.0) / total)
            else:
                # No complexity signal (e.g. placeholder-only) → split evenly
                # so the measured drift is not silently dropped.
                out[r] = drift / n if n else 0.0
    return out


def _dag_relax_bf16(allowed, group_ops, group_subzones, zone_drift, budget):
    """Measurement-gated BF16 relaxation (B4, now mostly a no-op under U2a/b).

    U2a/b makes all non-hard ops BF16-eligible by default.  The only ops that
    reach here without BF16 are the 3 hard ops in ``_DAG_HARD_HIGH_PRECISION_OPS``,
    and they are NEVER relaxed — their restriction is semantic, not numerical.
    Returns *allowed* unchanged in all remaining cases.
    """
    if "BF16" in allowed:
        return allowed
    # Hard ops (index_select/gt/lt/ne): BF16 can produce wrong answers,
    # not just imprecise ones.  Never relax.
    return allowed


def _estimate_group_sensitivities(strategy, group_roots, group_op_counts,
                                  group_subzone=None, zone_drift=None,
                                  D_real=None, predictor=None,
                                  group_edges=None, group_to_modules=None):
    """Configurable per-group BF16 sensitivity estimator (P1).

    strategy:
      ``"op_heuristic"`` — max per-op heuristic sensitivity (legacy baseline).
      ``"calibrated"``   — measurement-anchored: ``s_i = D_real · c_i/Σc_i`` with
                           ``c_i = Σ op counts`` (objective complexity).  The
                           total is pinned to the REAL all-bf16 drift ``D_real``
                           (``Σ s_i = D_real``), so the budget comparison is
                           physically meaningful.  Requires ``D_real``.
      ``"gnn"``          — learned nonlinear surrogate (``predictor`` callable).
                           G17 fix: passes *group_edges* and *group_to_modules*
                           so the GNN predictor uses structural features
                           (degree/depth/has_param/has_fixer) consistent with
                           training.

    Returns ``{root: sensitivity}``.
    """
    if strategy == "op_heuristic":
        out = {}
        for r in group_roots:
            ops = group_op_counts.get(r, {})
            out[r] = (max(_dag_get_op_sensitivity(op) for op in ops)
                      if ops else 1e-7)
        return out
    if strategy == "calibrated":
        comp = {r: float(sum(group_op_counts.get(r, {}).values()))
                for r in group_roots}
        # Per-zone calibrated: when zone_drift is available (e.g. from per-zone
        # D_real measurements), distribute each zone's measured drift by op-count
        # within that zone.  This fixes the 500,000× underestimation of global
        # B formula — dycore gets D_z1 (~1.6e-4), NN gets D_z3 (~7.7e-8).
        if zone_drift and len(zone_drift) > 0 and group_subzone:
            return _dag_distribute_zone_sensitivity(comp, group_subzone, zone_drift)
        # Global calibrated fallback: single D_real distributes uniformly
        if D_real is None:
            raise ValueError("calibrated strategy requires D_real")
        total = sum(comp.values())
        if total <= 0:
            n = len(group_roots) or 1
            return {r: float(D_real) / n for r in group_roots}
        return {r: float(D_real) * comp[r] / total for r in group_roots}
    if strategy == "gnn":
        if predictor is None:
            raise ValueError("gnn strategy requires a trained predictor")
        return predictor(group_roots, group_op_counts, group_subzone, zone_drift,
                         edges=group_edges, group_to_modules=group_to_modules)
    raise ValueError(f"unknown sensitivity strategy {strategy!r}")


# Simple Union-Find for DAG analysis
class _DAGUnionFind:
    def __init__(self):
        self._parent = {}
        self._constraints = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            ca = self._constraints.get(ra, frozenset({"BF16", "FP32", "FP64"}))
            cb = self._constraints.get(rb, frozenset({"BF16", "FP32", "FP64"}))
            self._parent[ra] = rb
            # C4: use UNION (most permissive), not intersection.
            # A hard-constrained op (index_select → {FP32, FP64}) should
            # not block BF16 for the rest of the group.  Hard ops never
            # include BF16, so union cannot accidentally add it to a
            # hard-only group.
            self._constraints[rb] = ca | cb

    def union_all(self, items):
        items = list(items)
        for i in range(1, len(items)):
            self.union(items[0], items[i])

    def set_constraint(self, tid, allowed):
        root = self.find(tid)
        cur = self._constraints.get(root, frozenset({"BF16", "FP32", "FP64"}))
        # C4: union (most permissive) — see union() above.
        self._constraints[root] = cur | allowed

    def get_constraint(self, tid):
        return self._constraints.get(self.find(tid), frozenset({"BF16", "FP32", "FP64"}))

    def groups(self):
        groups = defaultdict(list)
        for tid in self._parent:
            groups[self.find(tid)].append(tid)
        return dict(groups)


# ─── Module Graph ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ModuleGraph:
    """Directed graph of module data dependencies.

    Attributes:
        nodes: list of module names (matching SensitivityMap keys).
        edges: list of (src, dst) edges indicating data flow.
    """

    nodes: list[str]
    edges: list[tuple[str, str]]

    @classmethod
    def from_execution_order(cls, module_names: list[str]) -> "ModuleGraph":
        """Build a simple chain graph from execution order.

        Assumes data flows sequentially: module[i] → module[i+1].
        This is a reasonable default for NeuralGCM's outer-step pipeline:
          parameterization → dycore → fixer → filter.
        """
        edges = []
        for i in range(len(module_names) - 1):
            edges.append((module_names[i], module_names[i + 1]))
        return cls(nodes=list(module_names), edges=edges)

    @classmethod
    def for_neuralgcm(cls) -> "ModuleGraph":
        """Default module graph for NeuralGCM's physics step.

        Execution order from model.step():
          1. parameterization (Z3: neural tendency)
          2. _dycore_equation (Z1: IMEX integration)
          3. exp_filter (Z0: spectral filtering)
          4. conservation_fixer (Z2: energy/mass fix)

        Additional edges for data dependencies:
          - parameterization outputs feed into dycore.
          - Fixer reads both current state and state_before (from dycore).
        """
        nodes = [
            "parameterization",
            "_dycore_equation",
            "exp_filter",
            "conservation_fixer",
        ]
        edges = [
            ("parameterization", "_dycore_equation"),   # tendency → dycore
            ("_dycore_equation", "exp_filter"),          # state → filter
            ("exp_filter", "conservation_fixer"),        # filtered → fixer
            ("_dycore_equation", "conservation_fixer"),  # state_before → fixer
        ]
        return cls(nodes=nodes, edges=edges)


# ─── Zone Assignment Result ──────────────────────────────────────────────────


@dataclasses.dataclass
class ZoneAssignment:
    """A single discovered zone."""

    zone_id: int
    # Module names assigned to this zone.
    members: list[str]
    # Sensitivity centroid (mean drift of members).
    sensitivity_centroid: float
    # Recommended precision level.
    recommended_precision: str  # "BF16", "FP32", "TF32", "FP64"
    # Recommended dtype.
    recommended_dtype: torch.dtype = torch.float32


@dataclasses.dataclass
class DiscoveryResult:
    """Complete zone discovery output."""

    zones: list[ZoneAssignment]
    # Number of zones K (auto-selected or user-specified).
    k: int = 0
    # Eigengap values (for diagnostics).
    eigengaps: list[float] = dataclasses.field(default_factory=list)
    # Module → zone_id mapping.
    module_zone_map: dict[str, int] = dataclasses.field(default_factory=dict)
    # Total cast cost under this partition.
    estimated_cast_cost: float = 0.0

    def to_policy(self, base_policy: PrecisionPolicy | None = None) -> PrecisionPolicy:
        """Convert discovery result to a PrecisionPolicy.

        Maps discovered zones back to PZHA Z0-Z3 by matching module names
        to the known zone semantics. Falls back to sensitivity-based
        assignment for unknown modules.
        """
        if base_policy is None:
            base_policy = PrecisionPolicy()

        # Build module → recommended_dtype map.
        dtype_map = {}
        for za in self.zones:
            for member in za.members:
                dtype_map[member] = za.recommended_dtype

        # Map known NeuralGCM modules to PZHA zones.
        z1_dtype = dtype_map.get("_dycore_equation", base_policy.z1_compute_dtype)
        z2_dtype = dtype_map.get("conservation_fixer", base_policy.z2_compute_dtype)
        z3_dtype = dtype_map.get("parameterization", base_policy.z3_compute_dtype)

        # Z0 is always FP64 (basis arrays).
        z0_dtype = torch.float64

        return dataclasses.replace(
            base_policy,
            z0_dtype=z0_dtype,
            z1_compute_dtype=z1_dtype,
            z2_compute_dtype=z2_dtype,
            z3_compute_dtype=z3_dtype,
        )

    def summary_str(self) -> str:
        lines = [
            f"DiscoveryResult: K={self.k} zones",
            f"Eigengaps: {[f'{g:.4f}' for g in self.eigengaps[:8]]}",
            "",
        ]
        for za in sorted(self.zones, key=lambda z: z.sensitivity_centroid, reverse=True):
            members_str = ", ".join(za.members)
            lines.append(
                f"  Zone {za.zone_id}: "
                f"{za.recommended_precision} (centroid={za.sensitivity_centroid:.6e})"
                f"  members=[{members_str}]"
            )
        return "\n".join(lines)

    def to_dag_policy(
        self, base_policy: PrecisionPolicy | None = None,
        per_leaf: bool = False,
    ) -> PrecisionPolicy:
        """Generate PrecisionPolicy with DAG-discovered sub-zone dtypes.

        Unlike ``to_policy()`` which maps everything to 4 coarse PZHA zones,
        this method populates the sub-zone fields (z1_sht_dtype, z1_imex_dtype,
        z1_diagnostic_dtype) and the per-module ``_module_dtype_map`` for
        runtime enforcement via ``get_effective_dtype(zone, module_path=...)``.

        Resolution order at runtime:
          1. ``_module_dtype_map[module_path]`` — per-group optimal (Phase E direct)
          2. ``_subzone_dtype[subzone]`` — sub-zone override
          3. ``compute_dtype(zone)`` — coarse zone

        Args:
            per_leaf: If True, populate per-leaf ``_module_dtype_map`` entries
                (enables per-module mixed dtypes, requires U5 rewrite to avoid
                cross-boundary dtype mismatch crashes).  Default False for
                safe sub-zone-uniform enforcement.
        """
        if base_policy is None:
            base_policy = PrecisionPolicy()

        sub_dtypes: dict[str, torch.dtype] = getattr(
            self, "_dag_sub_zone_dtypes", {},
        )
        # Per-group module dtype map (Phase E direct → NO Phase F, NO MAX collapse)
        _module_dtype_map: dict[str, torch.dtype] = getattr(
            self, "_dag_module_dtype_map", {},
        )

        # U5 guard: when per_leaf=False, remove per-leaf entries to avoid
        # cross-boundary dtype mismatch crashes.  Per-leaf mixed dtypes
        # require FX graph rewriting (U5) to insert tensor_cast() at
        # module boundaries.  Sub-zone entries are preserved for uniform
        # sub-zone enforcement via _subzone_dtype.
        if not per_leaf:
            _module_dtype_map = {
                k: v for k, v in _module_dtype_map.items()
                if not k.startswith("leaf:")
            }

        # When _module_dtype_map is available, derive subzone hints from it
        # rather than using aggregated MAX. This preserves per-module granularity.
        # The _module_dtype_map is the authoritative source; subzone fields are
        # only backward-compat hints.
        # Prefer the budget-consistent per-leaf-derived sub-zone dtype
        # (``_module_dtype_map["subzone:X"]``, B1 MIN/budget-aware) over the
        # legacy MAX aggregation in ``_dag_sub_zone_dtypes``, which forces the
        # most precise member's dtype on the whole sub-zone (made the dycore
        # collapse to FP64 even when budget/measurement permits BF16).
        sz_hints: dict[str, torch.dtype] = {}
        for key, dt in _module_dtype_map.items():
            if isinstance(key, str) and key.startswith("subzone:"):
                sz_hints[key[len("subzone:"):]] = dt
        for sz, dt in sub_dtypes.items():
            sz_hints.setdefault(sz, dt)

        z1_sht = sz_hints.get("Z1_SHT")
        z1_imex = sz_hints.get("Z1_IMEX")
        z1_diag = sz_hints.get("Z1_DIAGNOSTIC")
        z1_dyn = sz_hints.get("Z1_DYNAMICS")

        # If no sub-zone assignment, fall back to Z1 coarse dtype
        z1_default = z1_dyn or base_policy.z1_compute_dtype

        # Build _subzone_dtype for runtime enforcement via zone_cast()
        _subzone_dtype: dict[str, torch.dtype] = {}
        for sz, dt in sz_hints.items():
            if dt is not None:
                _subzone_dtype[sz] = dt

        return dataclasses.replace(
            base_policy,
            z0_dtype=sz_hints.get("Z0_BASIS", base_policy.z0_dtype),
            z1_compute_dtype=z1_default,
            z1_sht_dtype=z1_sht,
            z1_imex_dtype=z1_imex,
            z1_diagnostic_dtype=z1_diag,
            z2_compute_dtype=sz_hints.get("Z2_FIXER", base_policy.z2_compute_dtype),
            z3_compute_dtype=sz_hints.get("Z3_NEURAL", base_policy.z3_compute_dtype),
            _subzone_dtype=_subzone_dtype,
            _module_dtype_map=_module_dtype_map,
        )

    def to_sda_config(self) -> dict:
        """Generate SDA-compatible precision configuration from DAG result.

        Returns a dict suitable for SDAController/SDAConfig consumption:
          - ``subzone_dtypes``: {subzone_label: dtype_name}
          - ``fine_groups``: per-group op composition + assignment
          - ``module_mapping``: DAG subzone → module path patterns
          - ``runtime_overrides``: ready-to-use _subzone_dtype dict
        """
        sub_dtypes = getattr(self, "_dag_sub_zone_dtypes", {})
        fine_groups = getattr(self, "_dag_fine_groups", {})
        n_assigned = sum(
            1 for g in fine_groups.values() if g.get("assigned_level"))

        # Module → subzone mapping (from _MODULE_TO_SUBZONE rules)
        from tornado_gcm.precision.zone_discovery import _MODULE_TO_SUBZONE
        module_map: dict[str, list[str]] = {}
        for pattern, sz in _MODULE_TO_SUBZONE:
            module_map.setdefault(sz, []).append(pattern)

        return {
            "dag_version": "task_decomposition_v2",
            "n_fine_groups": len(fine_groups),
            "n_assigned_fine_groups": n_assigned,
            "n_subzones": len(sub_dtypes),
            "subzone_dtypes": {
                sz: str(dt).split(".")[-1] for sz, dt in sub_dtypes.items()
            },
            "coarse_zones": {
                "Z0": str(sub_dtypes.get("Z0_BASIS", torch.float64)).split(".")[-1],
                "Z1": str(sub_dtypes.get("Z1_DYNAMICS", torch.float32)).split(".")[-1],
                "Z2": str(sub_dtypes.get("Z2_FIXER", torch.float64)).split(".")[-1],
                "Z3": str(sub_dtypes.get("Z3_NEURAL", torch.bfloat16)).split(".")[-1],
            },
            "runtime_overrides": {
                sz: str(dt).split(".")[-1] for sz, dt in sub_dtypes.items()
            },
            "module_patterns": {
                sz: patterns[:5] for sz, patterns in sorted(module_map.items())
            },
            "fine_group_summary": {
                "total": len(fine_groups),
                "assigned": n_assigned,
                "subzone_distribution": {
                    sz: sum(1 for g in fine_groups.values()
                            if g.get("subzone") == sz)
                    for sz in sorted(set(g.get("subzone", "?")
                                        for g in fine_groups.values()))
                },
                "top_groups_by_size": sorted(
                    [
                        {"size": g["size"], "ops": g["ops"],
                         "subzone": g["subzone"], "assigned": g["assigned_level"]}
                        for g in fine_groups.values()
                    ],
                    key=lambda x: x["size"], reverse=True,
                )[:10],
            },
        }

    def dag_summary_str(self) -> str:
        """Summary emphasizing sub-zone granularity (for DAG-discovered results)."""
        sub_dtypes = getattr(self, "_dag_sub_zone_dtypes", {})
        lines = [
            f"DiscoveryResult (DAG): K={self.k} zones → {len(sub_dtypes)} sub-zones",
            f"Sub-zone dtypes:",
        ]
        for sz, dt in sorted(sub_dtypes.items()):
            lines.append(f"  {sz:20s} → {str(dt).split('.')[-1]}")
        lines.append(f"\nPer-cluster assignment:")
        cluster_info = getattr(self, "_dag_cluster_assignment", {})
        for i, (cr, (dt, name, sz)) in enumerate(
            sorted(cluster_info.items(), key=lambda x: x[1][2])
        ):
            lines.append(f"  Cluster {i}: {name:4s} → {sz}")
        return "\n".join(lines)


# ─── Prior-Driven Precision Assignment ───────────────────────────────────────

# Precision levels are a hardware PRIOR — fixed, ordered, not discovered from data.
# Each level has: (torch.dtype, name, roundoff_u, relative_compute_cost)
#
# roundoff_u: effective unit roundoff (machine epsilon equivalent).
#   Drift scales as s_i · u(dtype), where s_i = drift_BF16 / u(BF16).
# cost: relative compute cost (FP32 = 1.0 baseline).
#   BF16 ≈ 0.5× (2× throughput on tensor cores).
#   FP64 ≈ 4× on consumer GPUs, 2× on H100.
#
# Ordered cheapest → most expensive. Zone IDs map to indices in this list.
_PRIOR_PRECISION_LEVELS: list[tuple[torch.dtype, str, float, float]] = [
    (torch.bfloat16, "BF16", 2.0 ** -8,  0.5),
    (torch.float32, "FP32", 2.0 ** -24, 1.0),
    (torch.float64, "FP64", 2.0 ** -53, 4.0),
]

# Backward-compatible aliases used by _feasibility_gated_dtypes and legacy code.
_DTYPE_ROUNDOFF: list[tuple[torch.dtype, str, float]] = [
    (dt, name, u) for dt, name, u, _cost in _PRIOR_PRECISION_LEVELS
]
_U_BF16 = _PRIOR_PRECISION_LEVELS[0][2]  # profiling probe roundoff


def _recommend_precision(centroid: float) -> tuple[torch.dtype, str]:
    """Map sensitivity centroid → cheapest feasible dtype (centroid-based fallback)."""
    for dt, name, u, _cost in _PRIOR_PRECISION_LEVELS:  # cheapest first
        if centroid * (_U_BF16 / u) <= _U_BF16:
            return dt, name
    return _PRIOR_PRECISION_LEVELS[-1][0], _PRIOR_PRECISION_LEVELS[-1][1]


def assign_precision_greedy(
    sensitivities: list[float],
    budget: float,
    levels: list[tuple[torch.dtype, str, float, float]] | None = None,
) -> tuple[list[int], float, float]:
    """Greedy optimal precision assignment (provably optimal for separable cost).

    Given n module sensitivities s_i (measured as BF16 drift) and a fixed set of
    hardware precision levels with cost c_k and roundoff u_k, find the assignment
    π: {0..n-1} → {0..K-1} minimizing Σ c_{π(i)} subject to Σ s_i · u(π(i)) / u_BF16 ≤ budget.

    Algorithm:
      1. Start all modules at cheapest precision (index 0).
      2. While total_drift > budget, upgrade the module with the best
         drift_reduction / cost_increase ratio (fractional knapsack greedy).

    Returns:
        (assignment list, total_cost, total_drift)
    """
    if levels is None:
        levels = _PRIOR_PRECISION_LEVELS

    n = len(sensitivities)
    K = len(levels)
    u_bf16 = levels[0][2]

    # assignment[i] = index into levels (0 = cheapest, K-1 = most expensive)
    assignment = [0] * n

    # Compute current total drift (all at BF16 = u_bf16).
    total_drift = sum(abs(s) for s in sensitivities)
    total_cost = sum(levels[0][3] for _ in range(n))

    # Greedy upgrade loop.
    while total_drift > budget:
        best_i = -1
        best_ratio = -1.0

        for i in range(n):
            k = assignment[i]
            if k + 1 >= K:
                continue  # already at max precision

            # Drift reduction from upgrading this module one level.
            current_u = levels[k][2]
            next_u = levels[k + 1][2]
            drift_reduction = abs(sensitivities[i]) * (current_u - next_u) / u_bf16

            # Cost increase from upgrading.
            cost_increase = levels[k + 1][3] - levels[k][3]

            if cost_increase <= 0:
                # Same-cost upgrade (e.g. BF16→TF32 on same HW): always do it.
                ratio = float("inf")
            else:
                ratio = drift_reduction / cost_increase

            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i

        if best_i < 0:
            break  # no module can be upgraded further

        k_old = assignment[best_i]
        assignment[best_i] += 1
        k_new = assignment[best_i]

        total_drift -= abs(sensitivities[best_i]) * (
            levels[k_old][2] - levels[k_new][2]
        ) / u_bf16
        total_cost += levels[k_new][3] - levels[k_old][3]

    return assignment, total_cost, total_drift


def _pareto_knee(points):
    """Index of the knee on a (cost, drift) frontier — the point of maximum
    perpendicular distance from the line joining the two endpoints (kneedle-style
    max-distance method).  Returns None for empty input, 0 for <3 points.
    """
    import math
    if not points:
        return None
    if len(points) < 3:
        return 0
    x0, y0 = points[0]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy) or 1.0
    best_i, best_d = 0, -1.0
    for i, (x, y) in enumerate(points):
        d = abs(dy * (x - x0) - dx * (y - y0)) / denom
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def pareto_budget_sweep(sensitivities, budgets=None, levels=None):
    """Sweep the drift budget and return the cost-vs-drift Pareto frontier plus
    the auto-selected knee point — replacing a single hand-picked budget.

    Operates on whatever per-unit *sensitivities* list is passed (per-group /
    per-leaf, NOT a subzone-MAX collapse), running the provably-optimal greedy
    assignment at each budget.  Cheap: no model execution, just the greedy.

    Returns a dict: {"frontier": [{budget,cost,drift,assignment,n_bf16,n_fp32,
    n_fp64}], "knee_index": int|None, "knee": <frontier entry>|None}.
    """
    if levels is None:
        levels = _PRIOR_PRECISION_LEVELS
    if budgets is None:
        # Log-spaced sweep across realistic drift budgets.
        budgets = [10.0 ** e for e in range(-7, -1)]
    frontier = []
    for b in budgets:
        assignment, cost, drift = assign_precision_greedy(sensitivities, b, levels)
        names = Counter(levels[k][1] for k in assignment)
        frontier.append({
            "budget": b, "cost": cost, "drift": drift, "assignment": assignment,
            "n_bf16": names.get("BF16", 0), "n_fp32": names.get("FP32", 0),
            "n_fp64": names.get("FP64", 0),
        })
    knee_index = _pareto_knee([(p["cost"], p["drift"]) for p in frontier])
    return {
        "frontier": frontier,
        "knee_index": knee_index,
        "knee": frontier[knee_index] if knee_index is not None else None,
    }


# ─── Zone Discovery ─────────────────────────────────────────────────────────


class ZoneDiscovery:
    """Automatic zone boundary discovery via sensitivity clustering.

    Algorithm:
      1. Build affinity matrix from sensitivity similarity + graph adjacency.
      2. Compute graph Laplacian.
      3. Eigengap heuristic selects K.
      4. Spectral clustering (or fallback to agglomerative) produces K zones.
      5. Each zone gets a precision recommendation.

    Args:
        sensitivity_map: SensitivityMap from Level-1 profiling.
        module_graph: ModuleGraph describing module data dependencies.
        alpha: weight of adjacency bonus in affinity (0 = ignore graph, 1 = strong).
        sigma: RBF kernel bandwidth for sensitivity similarity. If None,
            uses the median distance heuristic.
    """

    def __init__(
        self,
        sensitivity_map: SensitivityMap,
        module_graph: ModuleGraph | None = None,
        alpha: float = 0.3,
        sigma: float | None = None,
    ):
        self.smap = sensitivity_map
        self.graph = module_graph
        self.alpha = alpha
        self.sigma = sigma

        # Extract module names and sensitivities.
        self._names: list[str] = []
        self._sensitivities: list[float] = []
        for name, ms in sensitivity_map.modules.items():
            # Only include modules that have meaningful sensitivity.
            # Skip parent modules that were refined (their children are in the map).
            if ms.n_children_profiled > 0:
                continue
            self._names.append(name)
            self._sensitivities.append(ms.drift)

    @property
    def n_modules(self) -> int:
        return len(self._names)

    # ── Affinity matrix ───────────────────────────────────────────────

    def _build_affinity_matrix(self) -> np.ndarray:
        """Build affinity matrix from sensitivity similarity + adjacency.

        A[i,j] = exp(-|s_i - s_j|² / (2σ²)) + α * adjacent(i,j)
        """
        n = self.n_modules
        s = np.array(self._sensitivities, dtype=np.float64)

        # Pairwise sensitivity distance.
        diff = s[:, None] - s[None, :]  # (n, n)
        dist_sq = diff ** 2

        # Sigma: median distance heuristic.
        sigma = self.sigma
        if sigma is None:
            # Use median of non-zero distances.
            triu_dist = dist_sq[np.triu_indices(n, k=1)]
            sigma = float(np.sqrt(np.median(triu_dist) + 1e-30))
            sigma = max(sigma, 1e-10)

        # RBF affinity.
        A = np.exp(-dist_sq / (2 * sigma ** 2))

        # Quarantine zero-sensitivity modules: set their affinity with
        # all other modules to a negligible value (not zero, to keep the
        # Laplacian well-conditioned). A module with drift exactly 0 is
        # trivially BF16-safe and structurally distinct — the ε-gate
        # already handles this, but spectral clustering must also respect
        # it to avoid collapsing them into a shared zone (K*=1).
        zero_mask = np.abs(s) < 1e-15
        if zero_mask.any():
            tiny = 1e-12
            for zi in np.where(zero_mask)[0]:
                for j in range(n):
                    if zi != j:
                        A[zi, j] = tiny
                        A[j, zi] = tiny

        # Adjacency bonus.
        if self.graph is not None:
            name_to_idx = {name: i for i, name in enumerate(self._names)}
            for src, dst in self.graph.edges:
                # Match edges to our filtered module list.
                # Edges may reference modules not in our list (e.g., refined parents).
                i = name_to_idx.get(src)
                j = name_to_idx.get(dst)
                if i is not None and j is not None:
                    A[i, j] += self.alpha
                    A[j, i] += self.alpha

        # Zero diagonal.
        np.fill_diagonal(A, 0.0)
        return A

    # ── Eigengap heuristic ────────────────────────────────────────────

    def _compute_eigengaps(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalized Laplacian eigenvalues and eigengaps.

        Returns:
            (eigenvalues sorted ascending, gaps between consecutive eigenvalues)
        """
        n = A.shape[0]
        D = np.diag(A.sum(axis=1))
        L = D - A  # Unnormalized Laplacian.

        # Normalized Laplacian: D^{-1/2} L D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-30))
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt

        # Eigendecomposition.
        eigenvalues = np.linalg.eigvalsh(L_norm)
        eigenvalues = np.sort(eigenvalues)

        # Gaps.
        gaps = np.diff(eigenvalues)
        return eigenvalues, gaps

    def _select_k(
        self,
        eigengaps: np.ndarray,
        max_zones: int,
    ) -> int:
        """Select K using the eigengap heuristic.

        K = argmax_{1 ≤ k ≤ max_zones} gap[k-1].

        gap[i] = λ_{i+2} − λ_{i+1}, so index i corresponds to K = i+1. We now
        include gap[0] (K=1) as a legitimate candidate: a homogeneous model whose
        first nonzero eigenvalue already produces the largest gap collapses to a
        single zone (uniform precision) instead of being forced into an artificial
        ≥2-way split. For a model with genuine block structure (e.g. sensitive
        dynamics/fixer vs. tolerant neural net) the dominant gap stays at K≥2, so
        the documented K*=2 partition is preserved.
        """
        n = len(eigengaps)
        if n < 1:
            return min(1, self.n_modules)

        # Search range: K ∈ [1, min(max_zones, n_modules)].
        # We evaluate ALL possible K values, not just those with eigengaps.
        # n = len(eigengaps) = n_modules - 1, so upper ≤ n_modules - 1.
        # But K = n_modules (each module separate) is always a valid candidate.
        upper = min(max_zones, self.n_modules)
        if upper < 1:
            return 1

        # Extend eigengaps to cover K = n_modules if needed.
        # eigengaps[i] = λ_{i+1} - λ_i for i = 0..n_modules-2.
        # K = i + 1 needs eigengaps[i]. K = n_modules has no eigengap
        # (all modules separate); we treat it as a candidate with
        # gap = mean of all lower gaps (neutral reference).
        padded_gaps = list(eigengaps)
        if upper > len(padded_gaps):
            mean_gap = float(np.mean(padded_gaps)) if padded_gaps else 1.0
            while len(padded_gaps) < upper:
                padded_gaps.append(mean_gap)

        search_range = padded_gaps[:upper]
        if len(search_range) == 0:
            return 1

        # Find the max eigengap.
        best_idx = int(np.argmax(search_range))

        # Tie-breaking: if multiple K have gaps within 5% of the max,
        # prefer the LARGEST K. More zones → finer granularity → less
        # over-pay. The ε-gate can always merge zones that end up with
        # the same precision, but under-splitting forces modules with
        # different sensitivity needs into a shared (costly) precision.
        max_gap = search_range[best_idx]
        near_ties = [i for i, g in enumerate(search_range)
                     if g > 0 and max_gap / g < 1.05]
        if near_ties:
            best_idx = max(near_ties)

        k = best_idx + 1
        return k

    # ── ε-gated zone floor (feasibility-driven precision) ─────────────

    def _feasibility_gated_dtypes(
        self,
        labels: np.ndarray,
        k: int,
        budget: float,
    ) -> dict[int, tuple[torch.dtype, str]]:
        """Pick the LOWEST dtype per zone whose total drift stays within budget.

        Mirrors the upgraded HPS ``assign_coarse_zone`` rule: instead of mapping a
        centroid through fixed thresholds (which carries no feasibility guarantee),
        every zone is lowered greedily while the *global* aggregate drift
        Σ_v s_v·u(π_v) ≤ ε remains satisfied. Module drifts are measured at the
        BF16 probe, so s_v = drift_v / u(BF16) and the drift under dtype d scales as
        s_v·u(d). Tolerant (low-centroid) zones are lowered first so the budget is
        spent where it costs the least precision, keeping sensitive zones high.

        Returns a mapping zone_id → (dtype, name) that is feasible by construction
        (falls back to all-FP64 for any zone if even that is infeasible).
        """
        n = self.n_modules
        drift_bf16 = np.asarray(self._sensitivities, dtype=np.float64)
        s_unit = drift_bf16 / _U_BF16  # drift per unit roundoff (≈ reference s_v)

        members_idx: dict[int, list[int]] = {}
        centroids: dict[int, float] = {}
        for zid in range(k):
            idx = [i for i in range(n) if labels[i] == zid]
            if not idx:
                continue
            members_idx[zid] = idx
            centroids[zid] = float(np.mean(drift_bf16[idx]))

        fp64_u = _DTYPE_ROUNDOFF[2][2]
        zone_u: dict[int, float] = {zid: fp64_u for zid in members_idx}
        chosen: dict[int, tuple[torch.dtype, str]] = {
            zid: (_DTYPE_ROUNDOFF[2][0], _DTYPE_ROUNDOFF[2][1]) for zid in members_idx
        }

        def total_drift() -> float:
            return sum(float(np.sum(s_unit[idx])) * zone_u[zid]
                       for zid, idx in members_idx.items())

        # Lower tolerant zones first (ascending centroid).
        for zid in sorted(members_idx, key=lambda z: centroids[z]):
            for dt, name, u in _DTYPE_ROUNDOFF:  # BF16 → FP32 → FP64
                zone_u[zid] = u
                chosen[zid] = (dt, name)
                if total_drift() <= budget:
                    break
        return chosen

    # ── Clustering ────────────────────────────────────────────────────

    def _spectral_cluster(
        self, A: np.ndarray, k: int,
    ) -> np.ndarray:
        """Spectral clustering on affinity matrix.

        Returns array of cluster labels (0..k-1) for each module.
        Uses scikit-learn if available, falls back to simple k-means on
        eigenvectors.
        """
        n = A.shape[0]
        if k <= 1:
            # Single zone: every module shares one (uniform) precision.
            return np.zeros(n, dtype=int)
        if k >= n:
            # Each module is its own zone.
            return np.arange(n)

        try:
            from sklearn.cluster import SpectralClustering

            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )
            labels = sc.fit_predict(A)
            return labels

        except ImportError:
            logger.warning(
                "scikit-learn not available; using fallback eigenvector k-means."
            )
            return self._fallback_cluster(A, k)

    def _fallback_cluster(self, A: np.ndarray, k: int) -> np.ndarray:
        """Fallback clustering without scikit-learn.

        Uses eigenvectors of the Laplacian + simple greedy assignment.
        """
        n = A.shape[0]
        D = np.diag(A.sum(axis=1))
        L = D - A
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-30))
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        # Take first k eigenvectors (smallest eigenvalues).
        V = eigenvectors[:, :k]  # (n, k)

        # Row-normalize.
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-30
        V = V / norms

        # Simple greedy k-means (10 iterations).
        rng = np.random.RandomState(42)
        centers = V[rng.choice(n, size=k, replace=False)]

        for _ in range(10):
            # Assign.
            dists = np.linalg.norm(V[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            # Update centers.
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = V[mask].mean(axis=0)

        return labels

    # ── Cast cost estimation ──────────────────────────────────────────

    def _estimate_cast_cost(
        self,
        labels: np.ndarray,
        A: np.ndarray,
    ) -> float:
        """Estimate total cast cost for a partition.

        Cast cost = sum of affinities across zone boundaries.
        Lower is better (fewer/cheaper casts).
        """
        n = len(labels)
        cost = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j] and self.graph is not None:
                    name_to_idx = {name: idx for idx, name in enumerate(self._names)}
                    for src, dst in self.graph.edges:
                        si = name_to_idx.get(src)
                        sj = name_to_idx.get(dst)
                        if (si == i and sj == j) or (si == j and sj == i):
                            cost += 1.0
        return cost

    # ── Main discovery ────────────────────────────────────────────────

    def discover(
        self,
        max_zones: int = 8,
        force_k: int | None = None,
        budget: float | None = None,
    ) -> DiscoveryResult:
        """Run automatic zone discovery (spectral clustering path).

        Args:
            max_zones: maximum number of zones to consider.
            force_k: if set, force exactly this many zones (skip eigengap).
            budget: optional drift budget ε for ε-gated zone dtypes.

        Returns:
            DiscoveryResult with zone assignments and diagnostics.
        """
        n = self.n_modules
        logger.info("═" * 60)
        logger.info("HPS Level-2: Zone Discovery (spectral)")
        logger.info("  %d modules, max_zones=%d", n, max_zones)
        logger.info("═" * 60)

        if n == 0:
            logger.warning("No modules to cluster.")
            return DiscoveryResult(zones=[], k=0)

        if n <= 2:
            return self._trivial_assignment()

        A = self._build_affinity_matrix()
        eigenvalues, gaps = self._compute_eigengaps(A)
        eigengaps_list = gaps.tolist()

        if force_k is not None:
            k = min(force_k, n)
        else:
            k = self._select_k(gaps, max_zones)
        logger.info("  Selected K=%d (eigengaps: %s)", k, [f"{g:.4f}" for g in eigengaps_list[:8]])

        labels = self._spectral_cluster(A, k)
        gated = (
            self._feasibility_gated_dtypes(labels, k, budget)
            if budget is not None
            else None
        )
        return self._build_result(labels, eigengaps_list, gated, A)

    def discover_prior(
        self,
        budget: float | None = None,
        levels: list[tuple[torch.dtype, str, float, float]] | None = None,
    ) -> DiscoveryResult:
        """Run prior-driven precision assignment (no spectral clustering).

        K is fixed by hardware: |{BF16, FP32, FP64}|. Modules are assigned
        to the cheapest precision level that satisfies the global drift budget
        via greedy optimal assignment (provably optimal for separable cost).

        Args:
            budget: global drift budget ε. Modules are upgraded from cheapest
                precision until Σ s_i·u(π_i)/u_BF16 ≤ budget.
            levels: precision levels ordered cheapest→most expensive.
                Defaults to _PRIOR_PRECISION_LEVELS.

        Returns:
            DiscoveryResult where each zone = one precision level.
        """
        if levels is None:
            levels = _PRIOR_PRECISION_LEVELS

        n = self.n_modules
        logger.info("═" * 60)
        logger.info("HPS Level-2: Prior-Driven Assignment")
        logger.info("  %d modules, budget=%s, %d hardware levels",
                     n, f"{budget:.2e}" if budget else "None", len(levels))
        logger.info("═" * 60)

        if n == 0:
            return DiscoveryResult(zones=[], k=0)

        if budget is None:
            # Default budget: total drift if all modules were at cheapest precision,
            # scaled by 1e-3 (allow 0.1% drift).
            budget = sum(abs(s) for s in self._sensitivities) * 1e-3
            budget = max(budget, 1e-12)

        # Greedy optimal assignment.
        assignment, total_cost, total_drift = assign_precision_greedy(
            self._sensitivities, budget, levels,
        )

        # Group modules by precision level → zones.
        # zone_id = index into levels (0 = cheapest = BF16).
        K = len(levels)
        zone_members: dict[int, list[str]] = {k: [] for k in range(K)}
        for i, k in enumerate(assignment):
            zone_members[k].append(self._names[i])

        zones = []
        module_zone_map = {}
        for k in range(K):
            if not zone_members[k]:
                continue
            _, name, u, cost = levels[k]
            centroid = float(np.mean([
                self._sensitivities[i] for i in range(n) if assignment[i] == k
            ])) if zone_members[k] else 0.0

            za = ZoneAssignment(
                zone_id=k,
                members=zone_members[k],
                sensitivity_centroid=centroid,
                recommended_precision=name,
                recommended_dtype=levels[k][0],
            )
            zones.append(za)
            for m in zone_members[k]:
                module_zone_map[m] = k

        result = DiscoveryResult(
            zones=zones,
            k=len(zones),
            eigengaps=[],
            module_zone_map=module_zone_map,
            estimated_cast_cost=float(total_cost),
        )
        logger.info(result.summary_str())
        return result

    @staticmethod
    def discover_dag(
        model: "torch.nn.Module",
        sample_state,
        budget: float = 1e-3,
        target_groups: int = 10,
        sensitivities: dict[str, float] | None = None,
        coarsen: bool = True,
        strategy: str = "op_heuristic",
        D_real: float | None = None,
        predictor=None,
        **export_kwargs,
    ) -> "DiscoveryResult":
        """DAG-based precision zone discovery via torch.export (Phases A-F).

        Two operational modes:

        **Heuristic** (``coarsen=True``, default): Phases A→B→C→D→E→F.
          Coarsens ~2000 fine groups to ``target_groups`` clusters before
          assignment. Fast, practical, ~8-zone output.

        **Optimal** (``coarsen=False``): Phases A→B→C→E→F (skip D).
          Assigns each fine group individually via constraint-aware greedy,
          then merges groups with identical assigned dtypes. Provably tighter
          cost bound — no coarsening approximation loss.

        Args:
            model: NeuralGCMModel (or any nn.Module with State inputs).
            sample_state: example State for tracing.
            budget: global drift budget ε (default 1e-3 = 0.1%).
            target_groups: target number of coarsened clusters (ignored when
                ``coarsen=False``).
            sensitivities: optional dict mapping sub-zone labels to measured
                BF16 drift. When provided, replaces op-type heuristics in
                Phase C. Obtain via ``profile_dag_sensitivity(model, state)``.
            coarsen: If True (default), run hierarchical coarsening (Phase D).
                If False, skip coarsening — assign each fine group individually
                then merge by assigned dtype (optimal, tighter bound).
            **export_kwargs: passed to torch.export.export().

        Returns:
            DiscoveryResult with per-cluster dtype assignments.
        """
        import time

        logger.info("═" * 60)
        logger.info("HPS Level-2: DAG-based Discovery (Phases A-F)")
        logger.info("  budget=%.2e  target_groups=%d", budget, target_groups)
        logger.info("═" * 60)

        # Phase A: torch.export
        logger.info("Phase A: torch.export → FX graph...")
        t0 = time.perf_counter()
        ep = torch.export.export(
            model, (sample_state,),
            kwargs=export_kwargs or {"outer_steps": 1},
            strict=False,
        )
        graph = ep.graph
        logger.info("  %d nodes in %.1fs", len(list(graph.nodes)), time.perf_counter() - t0)

        # Phase B: Union-Find + dtype constraints
        logger.info("Phase B: Union-Find tensor grouping...")
        t0 = time.perf_counter()
        uf = _DAGUnionFind()
        node_to_tid = {}
        tid_counter = [0]

        def _new_tid():
            tid_counter[0] += 1
            return tid_counter[0]

        # Assign tensor IDs
        for node in graph.nodes:
            if node.op == "placeholder":
                node_to_tid[node.name] = _new_tid()
            elif node.op == "call_function":
                target_str = str(node.target)
                if target_str in _DAG_VIEW_OPS:
                    # View op: output = first input's tensor ID
                    first_arg = None
                    for arg in node.args:
                        if isinstance(arg, torch.fx.Node):
                            first_arg = arg; break
                        elif isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], torch.fx.Node):
                            first_arg = arg[0]; break
                    if first_arg is not None and first_arg.name in node_to_tid:
                        node_to_tid[node.name] = node_to_tid[first_arg.name]
                    else:
                        node_to_tid[node.name] = _new_tid()
                else:
                    node_to_tid[node.name] = _new_tid()

        # Union-Find on compute ops
        group_op_counts = defaultdict(Counter)
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if target_str in _DAG_SKIP_OPS or target_str in _DAG_VIEW_OPS:
                continue
            if target_str in _DAG_ZONE_BARRIER_OPS:
                continue  # precision zone boundary

            tids_in_op = set()
            out_tid = node_to_tid.get(node.name)
            if out_tid is not None:
                tids_in_op.add(out_tid)
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    in_tid = node_to_tid.get(arg.name)
                    if in_tid is not None:
                        tids_in_op.add(in_tid)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, torch.fx.Node):
                            in_tid = node_to_tid.get(a.name)
                            if in_tid is not None:
                                tids_in_op.add(in_tid)

            if len(tids_in_op) >= 2:
                uf.union_all(tids_in_op)
                allowed = _dag_get_allowed_dtypes(target_str)
                for tid in tids_in_op:
                    uf.set_constraint(tid, allowed)
                root = uf.find(out_tid) if out_tid is not None else None
                if root is not None:
                    group_op_counts[root][target_str] += 1

        groups = uf.groups()
        logger.info("  %d fine-grained tensor groups in %.3fs",
                     len(groups), time.perf_counter() - t0)

        # Map each group root → set of nn.Module paths that produced it
        # (from nn_module_stack), for per-leaf dtype assignment (no collapse).
        _group_to_modules = defaultdict(set)
        _group_edges = set()   # Q1: real inter-group dataflow edges (src→dst roots)
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            out_tid = node_to_tid.get(node.name)
            if out_tid is None:
                continue
            dst = uf.find(out_tid)
            mp = _dag_node_module_path(node)
            if mp:
                _group_to_modules[dst].add(mp)
            # Inter-group dataflow: each input tensor's group feeds this node's
            # output group.  Edges across group boundaries are the real DAG edges
            # the GNN propagates drift along (without these it degenerates to MLP).
            for arg in node.args:
                _args = arg if isinstance(arg, (list, tuple)) else (arg,)
                for a in _args:
                    if isinstance(a, torch.fx.Node):
                        in_tid = node_to_tid.get(a.name)
                        if in_tid is not None:
                            src = uf.find(in_tid)
                            if src != dst:
                                _group_edges.add((src, dst))

        # C3: Propagate module paths along dataflow edges (bidirectional).
        # Dycore ops (einsum, SHT) lack nn_module_stack metadata.  They
        # inherit module attribution from their dataflow neighbours so that
        # every group has a meaningful module path for leaf mapping.
        _changed = True
        for _ in range(20):      # bounded fixed-point iteration
            if not _changed:
                break
            _changed = False
            for src, dst in _group_edges:
                src_mp = _group_to_modules.get(src)
                dst_mp = _group_to_modules.get(dst)
                if src_mp and dst_mp is not None:
                    new_dst = src_mp - dst_mp
                    if new_dst:
                        _group_to_modules[dst] |= new_dst
                        _changed = True
                if dst_mp and src_mp is not None:
                    new_src = dst_mp - src_mp
                    if new_src:
                        _group_to_modules[src] |= new_src
                        _changed = True

        # C1 fix: post-C3 dark-group path inheritance.  After propagation
        # any group still without a module path lives in a connected
        # component where NO node has a path.  Inherit the nearest lit
        # neighbour's path WITHOUT rewiring edges — the original DAG
        # topology must be preserved for the GNN leaf graph.
        _dark = {r for r in groups if not _group_to_modules.get(r)}
        if _dark:
            _adj = defaultdict(set)
            for s, d in _group_edges:
                _adj[s].add(d); _adj[d].add(s)
            from collections import deque as _deque
            for _seed in sorted(_dark):
                if _group_to_modules.get(_seed):
                    continue
                _q = _deque([(_seed, 0)])
                _seen = {_seed}
                _best, _best_dist = None, 10 ** 9
                while _q:
                    _u, _d = _q.popleft()
                    if _d >= _best_dist:
                        continue
                    if _group_to_modules.get(_u):
                        _best, _best_dist = _u, _d
                        break
                    for _v in _adj.get(_u, ()):
                        if _v not in _seen:
                            _seen.add(_v)
                            _q.append((_v, _d + 1))
                if _best is not None:
                    # Inherit paths only — do NOT rewire edges.
                    _group_to_modules[_seed] = _group_to_modules[_best].copy()
            _still_dark = sum(1 for r in groups if not _group_to_modules.get(r))
            logger.info(
                "  C1 path inheritance: %d dark → %d still dark (%.1f%%), "
                "edges preserved",
                len(_dark), _still_dark,
                100 * _still_dark / max(len(groups), 1))

        # Phase C: Sensitivity estimation
        # Priority: measured sensitivities > op-type heuristics
        # Method-scope so Phase E can consult them for measurement-gated
        # constraint relaxation (B4); empty when no measurement is available.
        zone_drift: dict = {}
        group_subzone: dict = {}
        _phase_c_done = False

        # ── Unconditional sub-zone classification (was previously duplicated
        # inside the "calibrated" and "sensitivities-provided" branches, but
        # NOT run for strategy="op_heuristic" or strategy="gnn", causing all
        # groups to collapse to Z1_DYNAMICS at the reporting stage).  Running
        # it once here guarantees a valid group_subzone[root] for every root,
        # regardless of strategy; downstream branches may still override.
        # Classification order matches the historical duplicates:
        #   Z3: NN parameterization leaves
        #   Z2: fixer / conservation / kahan
        #   Z0: coords / sigma / grid / legendre / fourier / basis
        #   Z1_IMEX: any hard high-precision op (index/comparison)
        #   Z0_BASIS: no ops (constant tensor)
        #   Z1_DYNAMICS: otherwise
        for _root in groups:
            _mod_paths = _group_to_modules.get(_root, set())
            _ops = group_op_counts.get(_root, {})
            _zone = None
            for _mp in _mod_paths:
                if _mp.startswith("parameterization"):
                    _zone = "Z3_NEURAL"
                elif ("fixer" in _mp or "conservation" in _mp
                      or "kahan" in _mp.lower()):
                    _zone = "Z2_FIXER"
                elif any(_mp.startswith(_p) for _p in
                         ("coords", "sigma", "grid",
                          "legendre", "fourier", "basis")):
                    _zone = "Z0_BASIS"
                if _zone is not None:
                    break
            if _zone is None:
                _has_index_cmp = any(
                    _op.startswith(_p) for _op in _ops
                    for _p in _DAG_HARD_HIGH_PRECISION_OPS)
                if _has_index_cmp:
                    _zone = "Z1_IMEX"
                elif not _ops:
                    _zone = "Z0_BASIS"
                else:
                    _zone = "Z1_DYNAMICS"
            group_subzone[_root] = _zone

        if strategy == "calibrated":
            if D_real is None:
                raise ValueError(
                    "strategy='calibrated' requires D_real "
                    "(one real all-bf16 drift measurement)")
            # Per-zone calibrated: when per-zone measured drift dict is provided
            # (e.g. D_z1=1.61e-04, D_z3=7.72e-08), distribute each zone's drift
            # within that zone by op-count.  Falls back to global D_real for zones
            # without per-zone data.
            if sensitivities is not None and len(sensitivities) > 0:
                logger.info("Phase C: calibrated (per-zone) — %d zone anchors "
                            "from real measurement", len(sensitivities))
                for sz, dv in sorted(sensitivities.items()):
                    logger.info("  %-20s D=%.3e", sz, dv)
                zone_drift = dict(sensitivities)
                # Build group_subzone + group_complexity for per-zone distribution
                group_subzone = {}
                group_complexity = {}
                for root in groups:
                    mod_paths = _group_to_modules.get(root, set())
                    ops = group_op_counts.get(root, {})
                    zone = None
                    for mp in mod_paths:
                        if mp.startswith("parameterization"):
                            zone = "Z3_NEURAL"
                        elif "fixer" in mp or "conservation" in mp or "kahan" in mp.lower():
                            zone = "Z2_FIXER"
                        elif any(mp.startswith(p) for p in
                                 ("coords", "sigma", "grid", "legendre", "fourier", "basis")):
                            zone = "Z0_BASIS"
                    if zone is None:
                        has_index_cmp = any(
                            op.startswith(p) for op in ops
                            for p in _DAG_HARD_HIGH_PRECISION_OPS)
                        zone = "Z1_IMEX" if has_index_cmp else ("Z0_BASIS" if not ops else "Z1_DYNAMICS")
                    group_subzone[root] = zone
                    group_complexity[root] = float(sum(ops.values())) if ops else 0.0
                # Distribute per-zone calibrated: zone_drift[zone] × complexity/share
                sensitivities = _dag_distribute_zone_sensitivity(
                    group_complexity, group_subzone, zone_drift)
                # Fallback to op-heuristic for zones without measured drift
                for root in groups:
                    if sensitivities.get(root, 0.0) <= 0.0:
                        ops = group_op_counts.get(root, {})
                        sensitivities[root] = (
                            max(_dag_get_op_sensitivity(op) for op in ops)
                            if ops else 1e-7)
                # Mark that Phase C has fully populated sensitivities +
                # zone_drift/group_subzone — skip the elif below
                _phase_c_done = True
            else:
                logger.info("Phase C: calibrated — Σs_i anchored to real "
                            "D_real=%.3e × op-complexity (no op-heuristic pollution)",
                            D_real)
                sensitivities = _estimate_group_sensitivities(
                    "calibrated", list(groups.keys()), group_op_counts, D_real=D_real,
                    group_edges=_group_edges, group_to_modules=_group_to_modules)
                _phase_c_done = False
        elif strategy == "gnn":
            if predictor is None:
                raise ValueError("strategy='gnn' requires a trained predictor")
            logger.info("Phase C: GNN learned surrogate — per-group "
                         "sensitivities from nonlinear DAG predictor")
            sensitivities = _estimate_group_sensitivities(
                "gnn", list(groups.keys()), group_op_counts, predictor=predictor,
                group_edges=_group_edges, group_to_modules=_group_to_modules)
        elif sensitivities is not None and not _phase_c_done:
            logger.info("Phase C: measured zone drift × op-complexity "
                         "distribution (%d sub-zones)", len(sensitivities))
            zone_drift = dict(sensitivities)  # sub-zone label → measured drift
            # U2a/b: measurement-driven subzone classification.
            # Classify each group by which coarse precision zone its
            # connected leaf modules belong to — NOT by op-type heuristics.
            # Priority: (1) module-path→leaf→zone, (2) structural role
            # (index/comparison → Z1, empty → Z0), (3) default Z1.
            group_subzone = {}
            group_complexity = {}
            for root in groups:
                mod_paths = _group_to_modules.get(root, set())
                ops = group_op_counts.get(root, {})
                # Determine coarse zone from module paths:
                # - Z3: NN parameterization leaves (parameterization.*)
                # - Z2: conservation fixer modules
                # - Z0: basis/coordinate modules
                # - Z1: everything else (dycore default)
                zone = None
                for mp in mod_paths:
                    if mp.startswith("parameterization"):
                        zone = "Z3_NEURAL"
                    elif "fixer" in mp or "conservation" in mp or "kahan" in mp.lower():
                        zone = "Z2_FIXER"
                    elif any(mp.startswith(p) for p in
                             ("coords", "sigma", "grid", "legendre", "fourier", "basis")):
                        zone = "Z0_BASIS"
                if zone is None:
                    # Structural role from ops (only hard constraints + emptiness
                    # — no op-type sensitivity heuristics).
                    has_index_cmp = any(
                        op.startswith(p) for op in ops
                        for p in _DAG_HARD_HIGH_PRECISION_OPS)
                    if has_index_cmp:
                        zone = "Z1_IMEX"
                    elif not ops:
                        zone = "Z0_BASIS"
                    else:
                        zone = "Z1_DYNAMICS"
                group_subzone[root] = zone
                group_complexity[root] = float(sum(ops.values())) if ops else 0.0

            # Distribute measured drift across each sub-zone's groups by their
            # complexity share (objective op counts, no hand-tuned weights).
            _group_sensitivities = _dag_distribute_zone_sensitivity(
                group_complexity, group_subzone, zone_drift)
            # Sub-zones with no measured drift → op-heuristic fallback per group
            # (only used when measurement is unavailable for that subzone).
            for root in groups:
                if group_subzone[root] not in zone_drift:
                    ops = group_op_counts.get(root, {})
                    _group_sensitivities[root] = (
                        max(_dag_get_op_sensitivity(op) for op in ops)
                        if ops else 1e-7)
            sensitivities = _group_sensitivities
        else:
            # Op-type heuristic (fast, no model runs needed)
            _group_sensitivities = {}
            for root in groups:
                ops = group_op_counts.get(root, {})
                if ops:
                    max_s = max(_dag_get_op_sensitivity(op) for op in ops)
                else:
                    max_s = 1e-7
                _group_sensitivities[root] = max_s
            sensitivities = _group_sensitivities

        group_roots = list(groups.keys())
        group_sens = {r: sensitivities.get(r, 1e-7) for r in group_roots}

        if coarsen:
            # ── Phase D: Hierarchical coarsening (heuristic) ─────────────
            logger.info("Phase D: Coarsening %d → ≤%d...", len(groups), target_groups)
            t0 = time.perf_counter()

            clusters = {r: [r] for r in group_roots}
            cluster_sens = dict(group_sens)
            active = set(group_roots)
            iteration = 0
            max_iter = len(group_roots) * 2

            while len(active) > target_groups and iteration < max_iter:
                iteration += 1
                active_list = sorted(active, key=lambda r: cluster_sens[r])
                best_diff = float("inf")
                best_pair = None
                for i in range(len(active_list) - 1):
                    diff = abs(cluster_sens[active_list[i]] - cluster_sens[active_list[i + 1]])
                    if diff < best_diff:
                        best_diff = diff
                        best_pair = (active_list[i], active_list[i + 1])
                if best_pair is None:
                    break
                a, b = best_pair
                clusters[a].extend(clusters.pop(b))
                cluster_sens[a] = max(cluster_sens[a], cluster_sens[b])
                active.discard(b)

            logger.info("  %d clusters in %.3fs", len(active), time.perf_counter() - t0)
            assignment_roots = list(active)
            assignment_sensitivities = [cluster_sens[r] for r in assignment_roots]
        else:
            # ── Phase D skipped: per-group direct assignment (optimal) ──
            logger.info("Phase D: SKIPPED (coarsen=False → per-group optimal assignment)")
            logger.info("  %d fine groups → direct Phase E assignment", len(group_roots))
            # Each fine group is its own "cluster"
            clusters = {r: [r] for r in group_roots}
            assignment_roots = list(group_roots)
            assignment_sensitivities = [group_sens[r] for r in assignment_roots]

        # Phase E: Constraint-aware submodular assignment
        logger.info("Phase E: Submodular assignment (budget=%.2e, %d units)...",
                     budget, len(assignment_roots))
        cluster_roots = assignment_roots
        cluster_sensitivities = assignment_sensitivities

        # Per-cluster allowed levels
        name_to_idx = {"BF16": 0, "FP32": 1, "FP64": 2,
                        "torch.bfloat16": 0, "torch.float32": 1, "torch.float64": 2}
        filtered_indices = []
        for cr in cluster_roots:
            allowed = frozenset({"BF16", "FP32", "FP64"})
            for fine_r in clusters[cr]:
                allowed = allowed & uf.get_constraint(fine_r)
            # B4: measurement-gated relaxation (see _dag_relax_bf16).  Only
            # widens the option set; the budget-constrained greedy below still
            # upgrades away from BF16 if needed, so the budget is never violated.
            _grp_ops = [op for fine_r in clusters[cr]
                        for op in group_op_counts.get(fine_r, {})]
            _grp_szs = [group_subzone.get(fine_r) for fine_r in clusters[cr]]
            allowed = _dag_relax_bf16(allowed, _grp_ops, _grp_szs, zone_drift, budget)
            # Calibrated (B): D_real is the MEASURED all-bf16 drift.  If it is
            # within budget, all-bf16 is provably safe, so admit BF16 for every
            # group without a hard high-precision op (index/comparison).  This is
            # the same measurement-gated principle as B4, anchored to the single
            # real all-bf16 drift instead of per-zone estimates.
            if (strategy == "calibrated" and D_real is not None
                    and D_real <= budget and "BF16" not in allowed
                    and not any(op in _DAG_HARD_HIGH_PRECISION_OPS
                                for op in _grp_ops)):
                allowed = allowed | {"BF16"}
            indices = sorted({name_to_idx.get(a) for a in allowed if name_to_idx.get(a) is not None})
            if not indices:
                indices = [0, 1, 2]
            filtered_indices.append(indices)

        # Constraint-aware greedy: start at cheapest allowed, upgrade best ratio
        assignment = [filtered_indices[i][0] for i in range(len(cluster_roots))]
        u_bf16 = _PRIOR_PRECISION_LEVELS[0][2]
        total_drift = sum(
            abs(cluster_sensitivities[i]) * _PRIOR_PRECISION_LEVELS[assignment[i]][2] / u_bf16
            for i in range(len(cluster_roots))
        )
        total_cost = sum(_PRIOR_PRECISION_LEVELS[assignment[i]][3] for i in range(len(cluster_roots)))

        while total_drift > budget:
            best_i = -1
            best_ratio = -1.0
            for i in range(len(cluster_roots)):
                current = assignment[i]
                # Find next allowed level above current
                next_level = None
                for li in filtered_indices[i]:
                    if li > current:
                        next_level = li; break
                if next_level is None:
                    continue
                cur_u = _PRIOR_PRECISION_LEVELS[current][2]
                next_u = _PRIOR_PRECISION_LEVELS[next_level][2]
                drift_red = abs(cluster_sensitivities[i]) * (cur_u - next_u) / u_bf16
                cost_inc = _PRIOR_PRECISION_LEVELS[next_level][3] - _PRIOR_PRECISION_LEVELS[current][3]
                ratio = float("inf") if cost_inc <= 0 else drift_red / cost_inc
                if ratio > best_ratio:
                    best_ratio = ratio; best_i = i
            if best_i < 0:
                break
            old = assignment[best_i]
            for li in filtered_indices[best_i]:
                if li > old:
                    next_level = li; break
            assignment[best_i] = next_level
            total_drift -= abs(cluster_sensitivities[best_i]) * (
                _PRIOR_PRECISION_LEVELS[old][2] - _PRIOR_PRECISION_LEVELS[next_level][2]
            ) / u_bf16
            total_cost += _PRIOR_PRECISION_LEVELS[next_level][3] - _PRIOR_PRECISION_LEVELS[old][3]

        logger.info("  total_cost=%.1f  total_drift=%.2e", total_cost, total_drift)

        # Save per-fine-group assignment before merging (coarsen=False path)
        _fine_assignment_pre_merge = None
        _fine_roots_pre_merge = None
        if not coarsen:
            _fine_assignment_pre_merge = list(assignment)
            _fine_roots_pre_merge = list(cluster_roots)

        # Post-assignment: when coarsen=False, merge fine groups by (dtype, subzone)
        # NOT just by dtype — preserves sub-zone identity within same dtype level.
        if not coarsen and len(cluster_roots) > 20:
            # U2a/b: inherit subzone from Phase C measurement-driven
            # classification, not from op-type heuristics.
            _premerge_subzones = {}
            for root in cluster_roots:
                _premerge_subzones[root] = group_subzone.get(
                    root, "Z1_DYNAMICS")

            # Merge by (dtype_level, subzone) — preserves sub-zone identity
            dtype_sz_to_roots: dict[tuple[int, str], list[int]] = defaultdict(list)
            for i, cr in enumerate(cluster_roots):
                sz = _premerge_subzones.get(cr, "Z1_DYNAMICS")
                dtype_sz_to_roots[(assignment[i], sz)].append(cr)
            n_merged = len(dtype_sz_to_roots)
            logger.info("  Optimal merge: %d fine groups → %d (dtype,subzone) zones",
                         len(cluster_roots), n_merged)
            # Rebuild clusters by (dtype, subzone)
            new_clusters = {}
            new_roots = []
            new_sensitivities = []
            new_filtered = []
            new_assignment = []
            for (level_idx, sz), fine_roots in dtype_sz_to_roots.items():
                merged_root = fine_roots[0]
                new_clusters[merged_root] = []
                for fr in fine_roots:
                    new_clusters[merged_root].extend(clusters.get(fr, [fr]))
                new_roots.append(merged_root)
                new_sensitivities.append(max(
                    cluster_sensitivities[cluster_roots.index(fr)] for fr in fine_roots))
                new_filtered.append(filtered_indices[cluster_roots.index(fine_roots[0])])
                new_assignment.append(level_idx)
                # Store subzone for this merged root
                _premerge_subzones[merged_root] = sz
            # Replace with merged
            clusters = new_clusters
            cluster_roots = new_roots
            cluster_sensitivities = new_sensitivities
            filtered_indices = new_filtered
            assignment = new_assignment
            # Replace with merged
            clusters = new_clusters
            cluster_roots = new_roots
            cluster_sensitivities = new_sensitivities
            filtered_indices = new_filtered
            assignment = new_assignment
            total_cost = sum(_PRIOR_PRECISION_LEVELS[assignment[i]][3]
                           for i in range(len(cluster_roots)))
            total_drift = sum(
                abs(cluster_sensitivities[i]) * _PRIOR_PRECISION_LEVELS[assignment[i]][2] / u_bf16
                for i in range(len(cluster_roots)))
            logger.info("  Merged: cost=%.1f drift=%.2e %d zones",
                         total_cost, total_drift, len(cluster_roots))

        # Phase F: Build DiscoveryResult with sub-zone classification
        # Classify each cluster into a sub-zone based on dominating op types
        sub_zone_map: dict[int, str] = {}  # cluster_root → sub_zone label
        for i, cr in enumerate(cluster_roots):
            ops = group_op_counts.get(cr, {})
            op_str = " ".join(f"{op}:{c}" for op, c in ops.most_common(5))

            # U2a/b: inherit subzone from Phase C measurement-driven classification.
            sub_zone_map[cr] = group_subzone.get(cr, "Z1_DYNAMICS")

        # Build ZoneAssignment with sub_zone labels
        zones = []
        module_zone_map = {}
        for i, cr in enumerate(cluster_roots):
            level_idx = assignment[i]
            dt, name, u, cost = _PRIOR_PRECISION_LEVELS[level_idx]
            centroid = cluster_sensitivities[i]
            sz = sub_zone_map.get(cr, "Z1_DYNAMICS")
            n_fine = len(clusters[cr])
            members = [f"{sz}_{cr % 10000:04d}"]
            za = ZoneAssignment(
                zone_id=i,
                members=members,
                sensitivity_centroid=centroid,
                recommended_precision=name,
                recommended_dtype=dt,
            )
            zones.append(za)
            for m in members:
                module_zone_map[m] = i

        # Aggregate: compute per-sub-zone recommended dtype
        sub_zone_dtypes: dict[str, torch.dtype] = {}
        for i, cr in enumerate(cluster_roots):
            sz = sub_zone_map.get(cr, "Z1_DYNAMICS")
            dt = _PRIOR_PRECISION_LEVELS[assignment[i]][0]
            if sz not in sub_zone_dtypes:
                sub_zone_dtypes[sz] = dt
            else:
                # Pick the highest precision among clusters in this sub-zone
                existing = sub_zone_dtypes[sz]
                dt_rank = {torch.bfloat16: 0, torch.float32: 1, torch.float64: 2}
                if dt_rank.get(dt, 0) > dt_rank.get(existing, 0):
                    sub_zone_dtypes[sz] = dt

        logger.info(
            "  Sub-zone dtypes: %s",
            {k: str(v).split(".")[-1] for k, v in sub_zone_dtypes.items()},
        )

        result = DiscoveryResult(
            zones=zones, k=len(zones), eigengaps=[],
            module_zone_map=module_zone_map,
            estimated_cast_cost=float(total_cost),
        )
        # ── Build group → module path mapping (bypasses Phase F) ─────
        # Use leaf module enumeration for fine-grained path resolution.
        # Stack traces alone give only ~7 files; leaf modules give 50-200
        # natural paths with semantic meaning.  _group_to_modules was already
        # populated from nn_module_stack right after grouping (above).
        _module_to_dtype: dict[str, torch.dtype] = {}

        # Helper: classify a group's op composition to a subzone label
        # U2a/b unified classification: leaf→controlling group→subzone.: leaf→controlling group→subzone.
        # No keyword heuristics — the subzone is inherited from the DAG
        # group that controls the leaf.  Falls back to Z1_DYNAMICS when
        # no DAG group maps to the leaf.
        _leaf_paths = []
        for mod_name, mod in model.named_modules():
            if len(list(mod.children())) == 0:
                _leaf_paths.append(mod_name)

        _leaf_to_subzone: dict[str, str] = {}
        for lp in _leaf_paths:
            _sz = None
            for root in group_roots:
                mod_paths = _group_to_modules.get(root, set())
                for mp in mod_paths:
                    if lp == mp or lp.startswith(mp + ".") or mp.startswith(lp + "."):
                        _sz = group_subzone.get(root)
                        if _sz:
                            break
                if _sz:
                    break
            _leaf_to_subzone[lp] = _sz or "Z1_DYNAMICS"

        # Group subzones: inherit from Phase C measurement-driven
        # classification, falling back to Z1_DYNAMICS.
        _group_subzones: dict[int, str] = {}
        for root in group_roots:
            _group_subzones[root] = group_subzone.get(root, "Z1_DYNAMICS")

        logger.info("  Leaf modules: %d paths mapped to %d subzones",
                     len(_leaf_paths), len(set(_leaf_to_subzone.values())))

        # Local dtype rank helper (higher = more precise)
        _local_rank = {torch.bfloat16: 0, torch.float16: 0, torch.float32: 1, torch.float64: 2}
        def _dtype_rank(dt):
            return _local_rank.get(dt, 1)

        # Populate _module_dtype_map from Phase E assignments
        # Map each fine group's assigned dtype to its subzone, then
        # propagate to all leaf modules in that subzone via MIN/MAX rule.
        if not coarsen and _fine_assignment_pre_merge is not None:
            # First pass: accumulate per-subzone dtypes from fine groups
            _sz_dtypes: dict[str, list[torch.dtype]] = defaultdict(list)
            for i, root in enumerate(_fine_roots_pre_merge):
                lvl_idx = _fine_assignment_pre_merge[i]
                dt = _PRIOR_PRECISION_LEVELS[lvl_idx][0]
                sz = _group_subzones.get(root, "Z1_DYNAMICS")
                _sz_dtypes[sz].append(dt)

            # Second pass: compute per-subzone dtype (MIN for unconstrained, MAX for budget)
            _sz_resolved: dict[str, torch.dtype] = {}
            for sz in set(_leaf_to_subzone.values()):
                dtypes = _sz_dtypes.get(sz)
                if not dtypes:
                    # Subzone has leaf modules but no DAG groups —
                    # this indicates a gap in the DAG's op-type classification.
                    logger.warning(
                        "  Subzone %s has %d leaf modules but 0 DAG groups. "
                        "Using coarse Z1 default.",
                        sz, sum(1 for lp in _leaf_paths
                                if _leaf_to_subzone.get(lp) == sz))
                    # Fall back to conservative: FP64 for budget, BF16 for unconstrained
                    dtypes = [torch.float64] if budget != float("inf") else [torch.bfloat16]
                if budget == float("inf"):
                    _sz_resolved[sz] = min(dtypes, key=lambda dt: _dtype_rank(dt))
                else:
                    _sz_resolved[sz] = max(dtypes, key=lambda dt: _dtype_rank(dt))

            # Third pass: per-leaf dtype with NO sub-zone collapse.  Each leaf
            # aggregates only the DAG groups whose nn_module_stack path matches
            # it; leaves with no match fall back to the sub-zone aggregate.
            _root_dtype = {
                root: _PRIOR_PRECISION_LEVELS[_fine_assignment_pre_merge[i]][0]
                for i, root in enumerate(_fine_roots_pre_merge)
            }
            _leaf_resolved = _dag_resolve_leaf_dtypes(
                _root_dtype, _group_to_modules, _leaf_paths,
                _leaf_to_subzone, _sz_resolved, budget)
            for lp in _leaf_paths:
                if lp in _leaf_resolved:
                    _module_to_dtype[f"leaf:{lp}"] = _leaf_resolved[lp]
                    sz = _leaf_to_subzone.get(lp, "Z1_DYNAMICS")
                    if sz in _sz_resolved:
                        _module_to_dtype.setdefault(
                            f"subzone:{sz}", _sz_resolved[sz])
            logger.info(
                "  Per-leaf assignment: %d leaves → %d distinct dtypes "
                "(no sub-zone collapse)",
                len(_leaf_resolved), len(set(_leaf_resolved.values())))

        # Local dtype rank helper (higher = more precise)
        _local_rank = {torch.bfloat16: 0, torch.float16: 0, torch.float32: 1, torch.float64: 2}
        def _dtype_rank(dt):
            return _local_rank.get(dt, 1)

        # Populate _module_dtype_map from Phase E assignments

        # ── DAG-specific metadata ────────────────────────────────────

        # Sub-zone dtypes: {"Z1_SHT": torch.float64, ...}
        result._dag_sub_zone_dtypes = sub_zone_dtypes

        # Per-group module dtype map (Phase E direct → runtime, bypasses Phase F)
        result._dag_module_dtype_map = dict(_module_to_dtype)

        # Real inter-group dataflow edges (Q1) — the DAG edges the GNN
        # propagates drift along; without these the GNN degenerates to an MLP.
        result._dag_group_edges = sorted(_group_edges)
        # Group → leaf-module-path map (Q4) — to project group edges to the
        # measurable leaf granularity for the GNN bake-off.
        result._dag_group_to_modules = {
            r: sorted(ms) for r, ms in _group_to_modules.items()}

        # Per-cluster assignment: cluster_root → (dtype, name, subzone_label)
        result._dag_cluster_assignment = {
            cr: (_PRIOR_PRECISION_LEVELS[assignment[i]][0],
                 _PRIOR_PRECISION_LEVELS[assignment[i]][1],
                 sub_zone_map.get(cr, "Z1_DYNAMICS"))
            for i, cr in enumerate(cluster_roots)
        }

        # Fine groups: each original Union-Find group with its composition
        result._dag_fine_groups = {
            root: {
                "size": len(groups[root]),
                "ops": dict(group_op_counts.get(root, {}).most_common(5)),
                "sensitivity": sensitivities.get(root, 1e-7),
                "allowed_dtypes": set(uf.get_constraint(root)),
                "subzone": sub_zone_map.get(root, "Z1_DYNAMICS"),
                "assigned_level": None,  # filled below for coarsen=False
            }
            for root in group_roots
        }

        # Per-fine-group dtype assignment (coarsen=False path)
        if not coarsen and _fine_assignment_pre_merge is not None:
            for i, root in enumerate(_fine_roots_pre_merge):
                if root in result._dag_fine_groups:
                    lvl_idx = _fine_assignment_pre_merge[i]
                    result._dag_fine_groups[root]["assigned_level"] = (
                        _PRIOR_PRECISION_LEVELS[lvl_idx][1])
            result._dag_fine_group_count = len(_fine_roots_pre_merge)

        logger.info(result.summary_str())
        return result

    def _build_result(
        self,
        labels: np.ndarray,
        eigengaps_list: list[float],
        gated: dict | None,
        A: np.ndarray,
    ) -> DiscoveryResult:
        """Build DiscoveryResult from spectral clustering labels."""
        n = self.n_modules
        k = len(set(labels))
        zones = []
        module_zone_map = {}
        for zone_id in sorted(set(labels)):
            members = [self._names[i] for i in range(n) if labels[i] == zone_id]
            if not members:
                continue
            centroid = float(np.mean([
                self._sensitivities[i] for i in range(n) if labels[i] == zone_id
            ]))
            if gated is not None and zone_id in gated:
                rec_dtype, rec_name = gated[zone_id]
            else:
                rec_dtype, rec_name = _recommend_precision(centroid)
            za = ZoneAssignment(
                zone_id=zone_id,
                members=members,
                sensitivity_centroid=centroid,
                recommended_precision=rec_name,
                recommended_dtype=rec_dtype,
            )
            zones.append(za)
            for m in members:
                module_zone_map[m] = zone_id
        cast_cost = self._estimate_cast_cost(labels, A)
        return DiscoveryResult(
            zones=zones,
            k=len(zones),
            eigengaps=eigengaps_list,
            module_zone_map=module_zone_map,
            estimated_cast_cost=cast_cost,
        )

    def _trivial_assignment(self) -> DiscoveryResult:
        """Assign each module to its own zone (for n ≤ 2)."""
        zones = []
        module_zone_map = {}
        for i, (name, s) in enumerate(zip(self._names, self._sensitivities)):
            rec_dtype, rec_name = _recommend_precision(s)
            zones.append(ZoneAssignment(
                zone_id=i,
                members=[name],
                sensitivity_centroid=s,
                recommended_precision=rec_name,
                recommended_dtype=rec_dtype,
            ))
            module_zone_map[name] = i
        return DiscoveryResult(
            zones=zones, k=len(zones),
            module_zone_map=module_zone_map,
        )


# ─── Convenience: compare with PZHA ─────────────────────────────────────────


def compare_with_pzha(
    result: DiscoveryResult,
    sensitivity_map: SensitivityMap,
) -> str:
    """Compare auto-discovered zones with the manual PZHA 4-zone design.

    Returns a formatted comparison string.
    """
    pzha_mapping = {
        "parameterization": "Z3 (Neural)",
        "_dycore_equation": "Z1 (Dynamics)",
        "exp_filter": "Z0 (Basis)",
        "conservation_fixer": "Z2 (Fixer)",
    }

    lines = [
        "Comparison: HPS Auto-Discovery vs Manual PZHA",
        "=" * 50,
    ]

    for name in sorted(result.module_zone_map.keys()):
        zone_id = result.module_zone_map[name]
        za = next(z for z in result.zones if z.zone_id == zone_id)
        pzha_zone = pzha_mapping.get(name, "N/A")
        ms = sensitivity_map.modules.get(name)
        drift_str = f"{ms.drift:.6e}" if ms else "N/A"

        lines.append(
            f"  {name:30s}  PZHA={pzha_zone:15s}  "
            f"HPS=Zone{zone_id}({za.recommended_precision:4s})  "
            f"drift={drift_str}"
        )

    # Agreement measure.
    pzha_to_precision = {
        "Z3 (Neural)": "BF16",
        "Z1 (Dynamics)": "FP32",
        "Z0 (Basis)": "FP64",
        "Z2 (Fixer)": "FP64",
    }
    agree = 0
    total = 0
    for name in result.module_zone_map:
        if name in pzha_mapping:
            zone_id = result.module_zone_map[name]
            za = next(z for z in result.zones if z.zone_id == zone_id)
            expected = pzha_to_precision.get(pzha_mapping[name], "?")
            total += 1
            if za.recommended_precision == expected:
                agree += 1

    if total > 0:
        lines.append(f"\nAgreement with PZHA: {agree}/{total} ({100*agree/total:.0f}%)")

    return "\n".join(lines)


# ─── Monotonicity violation detection ────────────────────────────────────────


def detect_monotonicity_violations(
    pareto_results: list[dict],
    precision_order: list[str] | None = None,
) -> list[dict]:
    """Detect monotonicity violations in zone-level search results.

    Monotonicity: higher precision should always reduce drift.
    A violation occurs when promoting a zone to higher precision
    increases drift (e.g., due to mixed-precision boundary rounding).

    Args:
        pareto_results: list of dicts from zone_level_search, each with
            'zone_dtypes' (list of dtype strings) and 'drift' (float).
        precision_order: ordering of dtypes from low to high.
            Default: ['torch.bfloat16', 'torch.float32', 'torch.float64'].

    Returns:
        List of violation dicts with 'config_low', 'config_high',
        'drift_low', 'drift_high', 'zone_changed'.
    """
    if precision_order is None:
        precision_order = ["torch.bfloat16", "torch.float32", "torch.float64"]

    prec_rank = {p: i for i, p in enumerate(precision_order)}
    violations = []

    for i, r1 in enumerate(pareto_results):
        for j, r2 in enumerate(pareto_results):
            if i >= j:
                continue
            dtypes1 = r1["zone_dtypes"]
            dtypes2 = r2["zone_dtypes"]
            if len(dtypes1) != len(dtypes2):
                continue

            # Check if r2 dominates r1 in precision (all zones ≥).
            dominated = True
            changed_zone = -1
            n_changes = 0
            for k in range(len(dtypes1)):
                rank1 = prec_rank.get(dtypes1[k], -1)
                rank2 = prec_rank.get(dtypes2[k], -1)
                if rank2 < rank1:
                    dominated = False
                    break
                if rank2 > rank1:
                    changed_zone = k
                    n_changes += 1

            if dominated and n_changes == 1 and r2["drift"] > r1["drift"] * 1.05:
                violations.append({
                    "config_low": r1["config_id"],
                    "config_high": r2["config_id"],
                    "drift_low": r1["drift"],
                    "drift_high": r2["drift"],
                    "zone_changed": changed_zone,
                    "ratio": r2["drift"] / max(r1["drift"], 1e-30),
                })

    if violations:
        logger.warning(
            "Detected %d monotonicity violations (higher precision → higher drift).",
            len(violations),
        )
        for v in violations:
            logger.warning(
                "  %s → %s: drift %.3e → %.3e (%.1fx), zone %d changed",
                v["config_low"], v["config_high"],
                v["drift_low"], v["drift_high"], v["ratio"], v["zone_changed"],
            )

    return violations


# ─── Lattice-based pruning for Level-2 search ───────────────────────────────


def lattice_prune(
    configs: list[tuple[str, ...]],
    drift_cache: dict[tuple[str, ...], float],
    threshold: float,
    precision_order: list[str] | None = None,
) -> list[tuple[str, ...]]:
    """Prune infeasible configurations using the precision lattice.

    If config π violates the physics bound (drift > threshold),
    then any π' ⪯ π (lower or equal precision in all zones) also
    violates it.  This is the monotone pruning property.

    Non-monotone regions (detected by detect_monotonicity_violations)
    are handled by retaining configs within 2x of the monotone
    prediction.

    Args:
        configs: list of dtype tuples, one per zone.
        drift_cache: mapping from config → measured drift.
        threshold: physics drift threshold.
        precision_order: dtype ordering low → high.

    Returns:
        Pruned list of configs that cannot be ruled infeasible.
    """
    if precision_order is None:
        precision_order = ["torch.bfloat16", "torch.float32", "torch.float64"]

    prec_rank = {p: i for i, p in enumerate(precision_order)}

    def dominates(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        """True if a ⪰ b (a is higher or equal precision in all zones)."""
        return all(
            prec_rank.get(a[k], 0) >= prec_rank.get(b[k], 0)
            for k in range(len(a))
        )

    # Find infeasible configs.
    infeasible = {
        cfg for cfg, drift in drift_cache.items() if drift > threshold
    }

    pruned = []
    for cfg in configs:
        if cfg in drift_cache:
            # Already evaluated — keep if feasible.
            if drift_cache[cfg] <= threshold:
                pruned.append(cfg)
            continue

        # Check if any infeasible config dominates this one
        # (i.e., this one has even lower precision → also infeasible).
        should_prune = False
        for inf_cfg in infeasible:
            if dominates(inf_cfg, cfg):
                should_prune = True
                break

        if not should_prune:
            pruned.append(cfg)

    n_pruned = len(configs) - len(pruned)
    if n_pruned > 0:
        logger.info(
            "Lattice pruning: removed %d/%d configs (%d remain)",
            n_pruned, len(configs), len(pruned),
        )

    return pruned


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C (Upgraded): Real BF16 perturbation profiling for DAG clusters
# ═══════════════════════════════════════════════════════════════════════════════

# Module name → DAG sub-zone classification rules.
# Maps nn.Module leaf names to DAG sub-zone labels based on architectural role.
_MODULE_TO_SUBZONE: list[tuple[str, str]] = [
    # Grid/basis (Z0) — always FP64
    ("grid._buf", "Z0_BASIS"),
    ("coords.vertical._buf", "Z0_BASIS"),
    ("filter", "Z0_BASIS"),
    # SHT transforms (Z1_SHT)
    ("spherical_harmonics", "Z1_SHT"),
    ("inverse_transform", "Z1_SHT"),
    ("transform", "Z1_SHT"),
    ("to_nodal", "Z1_SHT"),
    ("to_modal", "Z1_SHT"),
    ("legendre", "Z1_SHT"),
    ("fourier", "Z1_SHT"),
    ("sht", "Z1_SHT"),
    # IMEX solver (Z1_IMEX)
    ("imex", "Z1_IMEX"),
    ("implicit", "Z1_IMEX"),
    ("explicit", "Z1_IMEX"),
    ("linalg", "Z1_IMEX"),
    ("solve", "Z1_IMEX"),
    ("integrator", "Z1_IMEX"),
    ("stepper", "Z1_IMEX"),
    ("dycore_pe", "Z1_IMEX"),
    # Diagnostic computation (Z1_DIAGNOSTIC)
    ("diagnostic", "Z1_DIAGNOSTIC"),
    ("geopotential", "Z1_DIAGNOSTIC"),
    ("advection", "Z1_DIAGNOSTIC"),
    ("sigma_coordinates", "Z1_DIAGNOSTIC"),
    # Conservation fixers (Z2)
    ("fixer", "Z2_FIXER"),
    ("conservation", "Z2_FIXER"),
    ("energy", "Z2_FIXER"),
    ("mass", "Z2_FIXER"),
    ("kahan", "Z2_FIXER"),
    # Neural network (Z3)
    ("parameterization", "Z3_NEURAL"),
    ("neural_net", "Z3_NEURAL"),
    ("mlp", "Z3_NEURAL"),
    ("epd", "Z3_NEURAL"),
    ("tower", "Z3_NEURAL"),
    ("conv", "Z3_NEURAL"),
    ("linear", "Z3_NEURAL"),
    ("gelu", "Z3_NEURAL"),
    ("transformer", "Z3_NEURAL"),
    ("attention", "Z3_NEURAL"),
    ("embed", "Z3_NEURAL"),
    ("feature", "Z3_NEURAL"),
    ("encode", "Z3_NEURAL"),
    ("decode", "Z3_NEURAL"),
    ("process_block", "Z3_NEURAL"),
    # General dynamics (fallback)
    ("_dycore", "Z1_DYNAMICS"),
    ("primitive", "Z1_DYNAMICS"),
    ("einsum", "Z1_DYNAMICS"),
]


def _classify_module_to_subzone(module_name: str) -> str:
    """Map an nn.Module leaf name to a DAG sub-zone label."""
    lower = module_name.lower()
    for pattern, subzone in _MODULE_TO_SUBZONE:
        if pattern in lower:
            return subzone
    return "Z1_DYNAMICS"  # default fallback


def _validate_e_ref(e_ref: float) -> float:
    """Validate the baseline all-FP64 reference energy before computing relative
    drifts.  A non-finite or zero baseline makes every downstream drift
    meaningless, so fail loudly rather than return a poisoned sensitivity dict."""
    import math
    if not math.isfinite(e_ref) or e_ref == 0.0:
        raise ValueError(
            f"profile_dag_sensitivity: invalid baseline energy e_ref={e_ref!r} "
            "(must be finite and non-zero); cannot compute relative drift.")
    return e_ref


def _accept_drift(drift_val):
    """Return a finite drift value, or None if the measurement is invalid
    (None/NaN/inf) and must be EXCLUDED from sub-zone aggregation.

    A non-finite drift means the probe failed, NOT that the module is
    critically sensitive — masking it as 1.0 would poison the sub-zone's
    aggregate and force the whole zone to FP64.
    """
    import math
    if drift_val is None:
        return None
    try:
        v = float(drift_val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _common_module_prefix(names):
    """Longest dotted module-path prefix shared by all *names*.

    Used to find the NN-block root (e.g. "parameterization") from its leaf
    paths, so the whole coherent block can be profiled at BF16 in one shot.
    Returns None when there is no common prefix.
    """
    if not names:
        return None
    parts = [n.split(".") for n in names]
    common = []
    for tup in zip(*parts):
        if len(set(tup)) == 1:
            common.append(tup[0])
        else:
            break
    return ".".join(common) if common else None


def _robust_subzone_sensitivity(drifts):
    """Robust aggregate of a sub-zone's per-module BF16 drifts.

    MAX lets a single outlier (or measurement glitch) poison the whole
    sub-zone → FP64.  Use the 95th percentile instead: it tracks the sensitive
    end of the distribution while ignoring a lone extreme.  For small samples
    (<5) p95 is unreliable, so fall back to MAX (the safe conservative choice).
    Returns None when there is no finite drift (caller applies critical fallback).
    """
    import math
    finite = []
    for d in drifts:
        if d is None:
            continue
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            finite.append(v)
    if not finite:
        return None
    if len(finite) < 5:
        return max(finite)
    return float(np.percentile(finite, 95))


def profile_dag_sensitivity(
    model: "torch.nn.Module",
    state,
    n_steps: int = 10,
    device: str = "cpu",
    test_dtype: "torch.dtype | None" = None,
) -> dict[str, float]:
    """Measure real BF16 drift for each DAG sub-zone via module perturbation.

    Profiles individual nn.Module leaves at BF16, then aggregates by DAG
    sub-zone classification. This replaces the op-type heuristic (Phase C)
    with actual measured sensitivities.

    Args:
        model: NeuralGCMModel (already built, weights loaded).
        state: Initial State for profiling rollout.
        n_steps: Number of forecast steps per perturbation test.
        device: Torch device string.
        test_dtype: Probe dtype (default: torch.bfloat16).

    Returns:
        Dict mapping sub-zone label → measured relative energy drift.
        Keys: Z0_BASIS, Z1_SHT, Z1_IMEX, Z1_DIAGNOSTIC, Z1_DYNAMICS,
              Z2_FIXER, Z3_NEURAL.
    """
    import torch as _torch

    if test_dtype is None:
        test_dtype = _torch.bfloat16

    from tornado_gcm.precision.sensitivity import SensitivityProfiler
    from tornado_gcm.neural.fixers import compute_total_energy

    logger.info("═" * 60)
    logger.info("Phase C (upgraded): Real BF16 perturbation profiling")
    logger.info("  n_steps=%d  test_dtype=%s  device=%s",
                 n_steps, str(test_dtype).split(".")[-1], device)
    logger.info("═" * 60)

    model.eval()
    if device != "cpu":
        model = model.to(device)
        state = state.tree_map(
            lambda t: t.to(device) if _torch.is_tensor(t) else t)

    # Baseline: all-FP64 reference run
    logger.info("  Baseline (all-FP64, %d steps)...", n_steps)
    with _torch.no_grad():
        s = state
        for _ in range(n_steps):
            s = model.step(s)
    e_ref = float(compute_total_energy(s).cpu())
    e_ref = _validate_e_ref(e_ref)

    # Collect leaf modules, skipping NN modules (Z3 is BF16-safe by design;
    # profiling them causes dtype mismatch when params are BF16 but inputs FP32)
    leaf_modules = []
    nn_keywords = ("neural_net", "linear", "conv", "gelu", "mlp", "tower",
                   "epd", "attention", "embed", "encode", "decode",
                   "process_block", "feature_extractor", "parameterization")
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and (list(mod.parameters()) or list(mod.buffers())):
            if not any(kw in name.lower() for kw in nn_keywords):
                leaf_modules.append(name)

    logger.info("  Profiling %d non-NN leaf modules at %s...", len(leaf_modules),
                 str(test_dtype).split(".")[-1])

    # Profile each leaf module
    from tornado_gcm.precision.policy import PrecisionPolicy
    # Use an energy validator that matches the baseline metric key ("energy").
    # The default conservation validator returns different keys, so every
    # per-metric drift fell to NaN ("no finite per-metric drifts") — the real
    # root cause of profile_dag_sensitivity producing NaN for every module.
    _energy_validator = lambda _s: {"energy": float(compute_total_energy(_s).cpu())}
    profiler = SensitivityProfiler(model, PrecisionPolicy(), validator=_energy_validator)
    baseline_metrics = {"energy": e_ref}
    per_module_drift: dict[str, float] = {}

    for i, mod_name in enumerate(leaf_modules):
        try:
            ms = profiler.profile_module(
                module_name=mod_name,
                test_dtype=test_dtype,
                init_state=state,
                baseline_metrics=baseline_metrics,
                n_steps=n_steps,
            )
            drift_val = _accept_drift(ms.drift)
            if drift_val is None:
                logger.warning(
                    "  Module %s: invalid drift %r — excluded from aggregation",
                    mod_name, ms.drift)
            else:
                per_module_drift[mod_name] = drift_val
        except Exception as e:
            logger.warning(
                "  Module %s profiling failed: %s — excluded from aggregation",
                mod_name, e)

        if (i + 1) % 20 == 0:
            logger.info("  Progress: %d/%d modules profiled", i + 1, len(leaf_modules))

    # NN (Z3) measured at ZONE level: downcast the whole NN block together and
    # measure real drift.  Per-module NN profiling is unreliable (chained
    # mat1×mat2 dtype mismatch), but the Z3 boundary auto-matches input dtype to
    # weight dtype, so the coherent all-NN-BF16 block runs cleanly.  Replaces the
    # previous hardcoded 1e-7 (the "zero hardcoded" goal).
    nn_leaves = [
        name for name, mod in model.named_modules()
        if len(list(mod.children())) == 0
        and any(kw in name.lower() for kw in nn_keywords)
        and (list(mod.parameters()) or list(mod.buffers()))
    ]
    nn_root = _common_module_prefix(nn_leaves)
    z3_drift = None
    if nn_root is not None:
        try:
            z3_ms = profiler.profile_module(
                module_name=nn_root, test_dtype=test_dtype,
                init_state=state, baseline_metrics=baseline_metrics,
                n_steps=n_steps)
            z3_drift = _accept_drift(z3_ms.drift)
            logger.info("  Z3 NN zone (%s) measured BF16 drift: %s",
                         nn_root, z3_drift)
        except Exception as e:
            logger.warning("  Z3 NN zone profiling failed: %s — excluded", e)
    if z3_drift is not None:
        for name in nn_leaves:
            per_module_drift[name] = z3_drift
    # else: NN modules excluded → Z3 sub-zone falls back to conservative

    # Aggregate by DAG sub-zone
    subzone_drifts: dict[str, list[float]] = {}
    for mod_name, drift in per_module_drift.items():
        sz = _classify_module_to_subzone(mod_name)
        subzone_drifts.setdefault(sz, []).append(drift)

    # Per sub-zone: robust p95 aggregate (a single outlier/glitch must not
    # poison the whole zone → FP64).  Empty → conservative critical.
    subzone_sensitivities: dict[str, float] = {}
    for sz, drifts in sorted(subzone_drifts.items()):
        agg = _robust_subzone_sensitivity(drifts)
        subzone_sensitivities[sz] = agg if agg is not None else 1.0

    logger.info("  Measured sub-zone sensitivities (BF16 drift):")
    for sz in sorted(subzone_sensitivities):
        n_mods = len(subzone_drifts.get(sz, []))
        s_val = subzone_sensitivities[sz]
        logger.info("    %-20s: %8.2e  (%d modules)", sz, s_val, n_mods)

    return subzone_sensitivities


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement-Driven DAG (zero hardcoded — model as oracle)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_dag_measured(
    model: "torch.nn.Module",
    sample_state,
    budget: float = 1e-4,
    n_steps: int = 5,
    max_measurements: int = 30,
    **export_kwargs,
) -> "DiscoveryResult":
    """Measurement-driven DAG discovery — real model as oracle.

    Phase A+B: torch.export → Union-Find → subzone structure.
    Phase C+D: SKIPPED (no sensitivity estimation, no coarsening).
    Phase E: greedy search using actual model executions as oracle.

    Each candidate dtype assignment is evaluated by running the model
    and measuring real energy drift. Zero hardcoded sensitivity values.

    Args:
        model: NeuralGCMModel.
        sample_state: initial State.
        budget: global drift budget ε.
        n_steps: forecast steps per measurement.
        max_measurements: limit on expensive model runs.
        **export_kwargs: passed to torch.export.

    Returns:
        DiscoveryResult with measurement-driven assignments.
    """
    import copy, time as _time
    from tornado_gcm.neural.fixers import compute_total_energy
    from tornado_gcm.precision.policy import PrecisionPolicy

    logger.info("═" * 60)
    logger.info("DAG Measurement-Driven Discovery (model as oracle)")
    logger.info("  budget=%.2e  n_steps=%d  max_measurements=%d",
                 budget, n_steps, max_measurements)
    logger.info("═" * 60)

    # ═══ Phase A+B: Get subzone structure ═══
    logger.info("Phase A+B: torch.export → Union-Find → subzones...")
    t0 = _time.perf_counter()
    r_base = ZoneDiscovery.discover_dag(
        model, sample_state, budget=budget, coarsen=False,
        sensitivities=None, **export_kwargs)
    sub_dtypes = getattr(r_base, "_dag_sub_zone_dtypes", {})
    export_time = _time.perf_counter() - t0
    logger.info("  %d subzones identified in %.1fs", len(sub_dtypes), export_time)

    if not sub_dtypes:
        logger.warning("No subzones found — returning base result")
        return r_base

    # ═══ Oracle: measure actual drift for a given subzone assignment ═══
    _n_measurements = [0]

    def _measure(policy_assignment: dict[str, torch.dtype]) -> float:
        """Run model with subzone policy, return energy drift vs all-FP64."""
        _n_measurements[0] += 1
        bp = PrecisionPolicy(
            z0_dtype=torch.float64,
            z1_compute_dtype=torch.float32,
            z2_compute_dtype=torch.float64,
            z3_compute_dtype=torch.bfloat16,
            _subzone_dtype=dict(policy_assignment),
        )
        # Baseline all-FP64
        fp64 = copy.deepcopy(bp)
        fp64.z0_dtype = fp64.z1_compute_dtype = fp64.z2_compute_dtype = fp64.z3_compute_dtype = torch.float64
        fp64._subzone_dtype = {sz: torch.float64 for sz in policy_assignment}

        orig_policy = model.policy
        model.set_precision_policy(fp64)
        s_ref = sample_state
        with torch.no_grad():
            for _ in range(n_steps):
                s_ref = model.step(s_ref)
        e_ref = float(compute_total_energy(s_ref).cpu())

        model.set_precision_policy(bp)
        s_test = sample_state
        with torch.no_grad():
            for _ in range(n_steps):
                s_test = model.step(s_test)
        e_test = float(compute_total_energy(s_test).cpu())

        model.set_precision_policy(orig_policy)
        drift = abs(e_test - e_ref) / max(abs(e_ref), 1e-30)
        return drift

    # ═══ Greedy search with real measurements ═══
    levels = [(torch.bfloat16, "BF16", 0.5), (torch.float32, "FP32", 1.0), (torch.float64, "FP64", 4.0)]
    sz_list = sorted(sub_dtypes.keys())
    assignment = {sz: 0 for sz in sz_list}  # index into levels: 0=BF16

    def _build_assign():
        return {sz: levels[assignment[sz]][0] for sz in sz_list}

    current_drift = _measure(_build_assign())
    logger.info("  Initial (all-BF16): drift=%.4e, measurement #%d",
                 current_drift, _n_measurements[0])

    while current_drift > budget and _n_measurements[0] < max_measurements:
        best_sz = None
        best_ratio = -1.0
        for sz in sz_list:
            if assignment[sz] >= len(levels) - 1:
                continue  # already at max (FP64)
            # Try upgrading this subzone
            trial_assign = dict(assignment)
            trial_assign[sz] += 1
            trial_drift = _measure(
                {s: levels[trial_assign[s]][0] for s in sz_list})
            improvement = current_drift - trial_drift
            cost_inc = levels[trial_assign[sz]][2] - levels[assignment[sz]][2]
            ratio = improvement / max(cost_inc, 1e-9)
            logger.debug("    %s: drift %.4e → %.4e, ratio=%.2e",
                          sz, current_drift, trial_drift, ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_sz = sz
            # Restore assignment for next trial
            if _n_measurements[0] >= max_measurements:
                break

        if best_sz is None:
            logger.info("  No feasible upgrade — all subzones at max precision")
            break

        assignment[best_sz] += 1
        current_drift = _measure(_build_assign())
        logger.info("  Upgraded %s → %s: drift=%.4e, measurement #%d",
                     best_sz, levels[assignment[best_sz]][1],
                     current_drift, _n_measurements[0])

    logger.info("  Converged: drift=%.4e, %d measurements, %d subzones",
                 current_drift, _n_measurements[0], len(sz_list))

    # ═══ Build result ═══
    final_assign = _build_assign()
    final_sub = {sz: dt for sz, dt in final_assign.items()}

    zones = []
    for i, sz in enumerate(sz_list):
        dt = final_assign[sz]
        name = {torch.bfloat16: "BF16", torch.float32: "FP32", torch.float64: "FP64"}[dt]
        za = ZoneAssignment(
            zone_id=i, members=[f"measured:{sz}"],
            sensitivity_centroid=0.0,  # not estimated — measured directly
            recommended_precision=name,
            recommended_dtype=dt,
        )
        zones.append(za)

    result = DiscoveryResult(
        zones=zones, k=len(zones), eigengaps=[],
        module_zone_map={f"measured:{sz}": i for i, sz in enumerate(sz_list)},
        estimated_cast_cost=float(
            sum(levels[assignment[sz]][2] for sz in sz_list)),
    )
    result._dag_sub_zone_dtypes = final_sub
    result._dag_measurement_count = _n_measurements[0]
    result._dag_export_time = export_time
    logger.info(result.summary_str())
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Unified HPS entry point — DAG-based is the default
# ═══════════════════════════════════════════════════════════════════════════════

def discover_zones(
    model: "torch.nn.Module | None" = None,
    state = None,
    sensitivity_map: "SensitivityMap | None" = None,
    module_graph: "ModuleGraph | None" = None,
    method: str = "dag",
    budget: float = 1e-4,
    target_groups: int = 8,
    max_zones: int = 8,
    force_k: int | None = None,
    profile: bool = False,         # default: op-heuristic (fast, no model runs)
    profile_n_steps: int = 2,       # fast 2-step profiling
    sensitivities: dict[str, float] | None = None,
    coarsen: bool = False,         # default: per-group direct (PhaseE→runtime, optimal)
    oracle: str = "heuristic",     # "heuristic" | "measured" | "unconstrained"
    oracle_n_steps: int = 5,       # forecast steps per oracle measurement
    oracle_max_measurements: int = 30,
    **export_kwargs,
) -> "DiscoveryResult":
    """Unified HPS zone discovery — DAG-based tensor-level analysis by default.

    Three methods are available, selected by the ``method`` parameter:

    ==============  ====================================================
    method          Description
    ==============  ====================================================
    ``"dag"``       DAG-based (default): torch.export → FX graph →
                    Union-Find + dtype constraints → coarsening →
                    submodular assignment. Requires ``model`` + ``state``.
                    Produces sub-zone granularity (Z1_SHT, Z1_IMEX, Z1_DIAG).
    ``"spectral"``  Spectral clustering on module sensitivity map.
                    Requires ``sensitivity_map``. Auto-selects K via eigengap.
    ``"prior"``     Prior-driven: hardware-fixed precision levels with
                    budget-gated greedy assignment. Requires ``sensitivity_map``.
    ==============  ====================================================

    Args:
        model: NeuralGCMModel for DAG-based discovery.
        state: Sample State for torch.export tracing.
        sensitivity_map: SensitivityMap for spectral/prior methods.
        module_graph: ModuleGraph for spectral clustering adjacency.
        method: ``"dag"`` (default), ``"spectral"``, or ``"prior"``.
        budget: Global drift budget ε for prior/DAG methods.
        target_groups: Target number of coarsened clusters (DAG only).
        max_zones: Maximum zones for spectral clustering.
        force_k: Force exactly K zones (spectral only; skips eigengap).
        profile: If True, run real BF16 perturbation profiling (Phase C
            upgrade). Adds ~N_modules × n_steps × step_time.
        profile_n_steps: Number of forecast steps per profiling run.
        **export_kwargs: Passed to torch.export.export() (DAG only).

    Returns:
        DiscoveryResult with zone assignments.
        Use ``result.to_policy()`` (4-zone) or ``result.to_dag_policy()``
        (sub-zone granularity, DAG only).

    Examples::

        # Default: DAG-based with op-type heuristic sensitivities (fast)
        result = discover_zones(model, state)
        policy = result.to_dag_policy()

        # DAG-based with real BF16 profiling (accurate, slower)
        result = discover_zones(model, state, profile=True, profile_n_steps=20)
        policy = result.to_dag_policy()

        # Spectral clustering on pre-profiled sensitivities
        result = discover_zones(sensitivity_map=smap, method="spectral")
        policy = result.to_policy()
    """
    method = method.lower()
    if method not in ("dag", "spectral", "prior"):
        raise ValueError(
            f"Unknown method '{method}'. Expected 'dag', 'spectral', or 'prior'."
        )

    if method == "dag":
        if model is None or state is None:
            raise ValueError(
                "DAG-based discovery requires model + state. "
                "Use method='spectral' or method='prior' with a sensitivity_map, "
                "or pass a model and sample state."
            )
        # Unconstrained oracle: no budget, cheapest-allowed per group.
        # Each group gets its individually optimal dtype (BF16 where safe,
        # FP32/FP64 where required by op constraints). No aggregation loss.
        if oracle == "unconstrained":
            return ZoneDiscovery.discover_dag(
                model, state,
                budget=float("inf"),         # never upgrade
                target_groups=target_groups,
                sensitivities=None,           # no sensitivity needed
                coarsen=False,                # direct per-group
                **export_kwargs,
            )
        # Measurement-driven oracle: zero hardcoded, model as ground truth
        if oracle == "measured":
            return discover_dag_measured(
                model, state, budget=budget,
                n_steps=oracle_n_steps,
                max_measurements=oracle_max_measurements,
                **export_kwargs,
            )
        # Phase C upgrade: real profiling (if requested and no explicit sensitivities given)
        if sensitivities is None and profile:
            sensitivities = profile_dag_sensitivity(
                model, state, n_steps=profile_n_steps,
                device=str(next(model.parameters()).device),
            )
        return ZoneDiscovery.discover_dag(
            model, state, budget=budget,
            target_groups=target_groups,
            sensitivities=sensitivities,
            coarsen=coarsen,
            **export_kwargs,
        )

    # Spectral or prior: require sensitivity_map
    if sensitivity_map is None:
        raise ValueError(
            f"Method '{method}' requires a sensitivity_map. "
            f"Run SensitivityProfiler first, or use method='dag' with model+state."
        )

    if module_graph is None:
        module_graph = ModuleGraph.for_neuralgcm()

    zd = ZoneDiscovery(sensitivity_map, module_graph)

    if method == "spectral":
        return zd.discover(max_zones=max_zones, force_k=force_k, budget=budget)

    # prior
    return zd.discover_prior(budget=budget)
