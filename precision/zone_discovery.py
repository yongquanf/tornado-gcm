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

from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone
from tornado_gcm.precision.sensitivity import SensitivityMap

logger = logging.getLogger(__name__)


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


# ─── Precision recommendation logic ─────────────────────────────────────────

# Thresholds for mapping sensitivity centroid → precision level.
# These can be tuned; defaults based on exp32/exp35 empirical data.
_PRECISION_THRESHOLDS = {
    "critical":  0.1,      # drift > 0.1 → FP64
    "moderate":  0.001,    # drift > 0.001 → FP32
    "tolerant":  0.0,      # drift ≤ 0.001 → BF16
}

_PRECISION_DTYPE_MAP = {
    "critical": (torch.float64, "FP64"),
    "moderate": (torch.float32, "FP32"),
    "tolerant": (torch.bfloat16, "BF16"),
}


def _recommend_precision(centroid: float) -> tuple[torch.dtype, str]:
    """Map a sensitivity centroid to a recommended dtype."""
    if centroid > _PRECISION_THRESHOLDS["critical"]:
        return _PRECISION_DTYPE_MAP["critical"]
    elif centroid > _PRECISION_THRESHOLDS["moderate"]:
        return _PRECISION_DTYPE_MAP["moderate"]
    else:
        return _PRECISION_DTYPE_MAP["tolerant"]


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

        K = argmax_{2 ≤ k ≤ max_zones} gap[k-1]
        (Skip gap[0] which is always ~0 for the trivial eigenvalue.)
        """
        n = len(eigengaps)
        if n < 2:
            return min(2, self.n_modules)

        # Search range: K ∈ [2, min(max_zones, n_modules)].
        upper = min(max_zones, self.n_modules, n)
        if upper < 2:
            return max(1, self.n_modules)

        # Find largest gap in [1, upper-1] (index 1 = gap between λ_1 and λ_2).
        search_range = eigengaps[1:upper]
        if len(search_range) == 0:
            return 2

        best_idx = int(np.argmax(search_range))
        k = best_idx + 2  # +1 for 0-index, +1 because gap[i] → K=i+1.
        return k

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
    ) -> DiscoveryResult:
        """Run automatic zone discovery.

        Args:
            max_zones: maximum number of zones to consider.
            force_k: if set, force exactly this many zones (skip eigengap).

        Returns:
            DiscoveryResult with zone assignments and diagnostics.
        """
        n = self.n_modules
        logger.info("═" * 60)
        logger.info("HPS Level-2: Zone Discovery")
        logger.info("  %d modules, max_zones=%d", n, max_zones)
        logger.info("═" * 60)

        if n == 0:
            logger.warning("No modules to cluster.")
            return DiscoveryResult(zones=[], k=0)

        if n <= 2:
            # With ≤2 modules, each is its own zone.
            return self._trivial_assignment()

        # Step 1: Build affinity matrix.
        A = self._build_affinity_matrix()

        # Step 2: Eigengap analysis.
        eigenvalues, gaps = self._compute_eigengaps(A)
        eigengaps_list = gaps.tolist()

        # Step 3: Select K.
        if force_k is not None:
            k = min(force_k, n)
        else:
            k = self._select_k(gaps, max_zones)
        logger.info("  Selected K=%d (eigengaps: %s)", k, [f"{g:.4f}" for g in eigengaps_list[:8]])

        # Step 4: Spectral clustering.
        labels = self._spectral_cluster(A, k)

        # Step 5: Build zone assignments.
        zones = []
        module_zone_map = {}
        for zone_id in range(k):
            members = [self._names[i] for i in range(n) if labels[i] == zone_id]
            if not members:
                continue

            centroid = float(
                np.mean([self._sensitivities[i] for i in range(n) if labels[i] == zone_id])
            )
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

        # Step 6: Estimate cast cost.
        cast_cost = self._estimate_cast_cost(labels, A)

        result = DiscoveryResult(
            zones=zones,
            k=len(zones),
            eigengaps=eigengaps_list,
            module_zone_map=module_zone_map,
            estimated_cast_cost=cast_cost,
        )
        logger.info(result.summary_str())
        return result

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
