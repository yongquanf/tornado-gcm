"""HPS Visualization: sensitivity heatmaps + zone partition diagrams.

Produces:
  1. Sensitivity bar chart (module drift ranking).
  2. Zone partition diagram (color-coded zones with recommendations).
  3. Eigengap plot (K selection diagnostic).
  4. PZHA comparison table.
  5. Multi-resolution K scaling plot (if data available).

All plots use matplotlib. Exports PNG + PDF to output directory.

Usage:
    from pytorch_src.precision.hps_visualization import plot_hps_results
    plot_hps_results(smap, discovery, output_dir)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np

from pytorch_src.precision.sensitivity import SensitivityMap
from pytorch_src.precision.zone_discovery import DiscoveryResult

logger = logging.getLogger(__name__)

# Zone color palette (colorblind-friendly, from Okabe-Ito).
ZONE_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#999999",  # grey
]

PRECISION_COLORS = {
    "BF16": "#56B4E9",
    "FP32": "#E69F00",
    "TF32": "#F0E442",
    "FP64": "#D55E00",
}


def _try_import_matplotlib():
    """Import matplotlib, return (plt, True) or (None, False)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for headless servers.
        import matplotlib.pyplot as plt
        return plt, True
    except ImportError:
        logger.warning("matplotlib not installed; skipping HPS visualization.")
        return None, False


def plot_sensitivity_bar(
    smap: SensitivityMap,
    output_path: pathlib.Path,
    title: str = "Module Precision Sensitivity (HPS Level-1)",
) -> None:
    """Bar chart of per-module sensitivity (drift when downcast to BF16)."""
    plt, ok = _try_import_matplotlib()
    if not ok:
        return

    ranking = smap.sensitivity_ranking
    names = [name for name, _ in ranking]
    drifts = [drift for _, drift in ranking]

    # Color by recommendation.
    rec = smap.get_zone_recommendation()
    colors = []
    for name in names:
        level = rec.get(name, "tolerant")
        if level == "critical":
            colors.append(PRECISION_COLORS["FP64"])
        elif level == "moderate":
            colors.append(PRECISION_COLORS["FP32"])
        else:
            colors.append(PRECISION_COLORS["BF16"])

    # Replace zero-drift with a small floor for log scale.
    plot_drifts = [max(d, 1e-15) for d in drifts]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), plot_drifts, color=colors, edgecolor="black", linewidth=0.5)

    # Shorten long labels for readability.
    short_names = [n.replace("zone:", "").replace("conservation_fixer", "cons_fixer") for n in names]
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative Drift (max over metrics)")
    ax.set_title(title)
    ax.set_yscale("log")

    # Threshold lines.
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Critical (FP64)")
    ax.axhline(y=0.001, color="orange", linestyle="--", alpha=0.5, label="Moderate (FP32)")
    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sensitivity bar chart → %s", output_path)


def plot_zone_partition(
    result: DiscoveryResult,
    smap: SensitivityMap,
    output_path: pathlib.Path,
    title: str = "Auto-Discovered Zone Partition (HPS Level-2)",
) -> None:
    """Diagram showing discovered zones with module assignments."""
    plt, ok = _try_import_matplotlib()
    if not ok:
        return

    from matplotlib.patches import FancyBboxPatch

    n_zones = result.k
    fig, ax = plt.subplots(figsize=(12, max(4, n_zones * 1.5)))

    y_pos = 0
    for za in sorted(result.zones, key=lambda z: z.sensitivity_centroid, reverse=True):
        color = ZONE_COLORS[za.zone_id % len(ZONE_COLORS)]

        # Zone header.
        ax.add_patch(FancyBboxPatch(
            (0.05, y_pos), 0.9, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.3, edgecolor="black",
        ))

        ax.text(
            0.1, y_pos + 0.55,
            f"Zone {za.zone_id}: {za.recommended_precision} "
            f"(centroid={za.sensitivity_centroid:.2e})",
            fontsize=11, fontweight="bold", va="center",
        )

        members_str = ", ".join(za.members)
        ax.text(
            0.1, y_pos + 0.25,
            f"Members: {members_str}",
            fontsize=9, va="center", color="#333333",
        )

        y_pos += 1.0

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, y_pos + 0.2)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Zone partition diagram → %s", output_path)


def plot_eigengap(
    result: DiscoveryResult,
    output_path: pathlib.Path,
    title: str = "Eigengap Analysis (K Selection)",
) -> None:
    """Plot eigengap values to diagnose K selection."""
    plt, ok = _try_import_matplotlib()
    if not ok:
        return

    gaps = result.eigengaps
    if not gaps or len(gaps) < 2:
        logger.info("Not enough eigengaps to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(gaps) + 1))
    ax.bar(x, gaps, color="#56B4E9", edgecolor="black", linewidth=0.5)

    # Mark selected K.
    if result.k > 1 and result.k - 1 < len(gaps):
        ax.bar(result.k - 1, gaps[result.k - 1], color="#D55E00",
               edgecolor="black", linewidth=0.5, label=f"K={result.k}")
        ax.legend()

    ax.set_xlabel("Gap index (k → k+1)")
    ax.set_ylabel("Eigenvalue gap")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Eigengap plot → %s", output_path)


def plot_multi_resolution_scaling(
    resolution_data: dict[str, tuple[int, int]],  # {res: (L_max, K)}
    output_path: pathlib.Path,
    title: str = "Zone Count K vs Spectral Resolution L_max",
) -> None:
    """Plot K(L_max) to test the K ∝ log(L_max) hypothesis."""
    plt, ok = _try_import_matplotlib()
    if not ok:
        return

    resolutions = sorted(resolution_data.keys())
    l_vals = [resolution_data[r][0] for r in resolutions]
    k_vals = [resolution_data[r][1] for r in resolutions]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(l_vals, k_vals, s=100, c="#0072B2", zorder=5, edgecolors="black")
    for res, l, k in zip(resolutions, l_vals, k_vals):
        ax.annotate(f"T{l} ({res}°)", (l, k), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    # Fit log curve if enough data points.
    if len(l_vals) >= 3:
        log_l = np.log(np.array(l_vals, dtype=float))
        k_arr = np.array(k_vals, dtype=float)
        # K = a * log(L_max) + b
        A = np.vstack([log_l, np.ones(len(log_l))]).T
        result = np.linalg.lstsq(A, k_arr, rcond=None)
        a, b = result[0]
        l_fit = np.linspace(min(l_vals) * 0.8, max(l_vals) * 1.2, 100)
        k_fit = a * np.log(l_fit) + b
        ax.plot(l_fit, k_fit, "--", color="red", alpha=0.6,
                label=f"K ≈ {a:.2f}·log(L) + {b:.2f}")
        ax.legend()

    ax.set_xlabel("L_max (max wavenumber)")
    ax.set_ylabel("K (discovered zones)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Multi-resolution scaling plot → %s", output_path)


# ─── Convenience: plot all ────────────────────────────────────────────────────


def plot_hps_results(
    smap: SensitivityMap,
    result: DiscoveryResult,
    output_dir: pathlib.Path,
    resolution_data: Optional[dict[str, tuple[int, int]]] = None,
) -> None:
    """Generate all HPS visualizations.

    Args:
        smap: Level-1 sensitivity map.
        result: Level-2 discovery result.
        output_dir: output directory.
        resolution_data: optional multi-resolution data for K scaling plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_sensitivity_bar(smap, output_dir / "sensitivity_bar.png")
    plot_zone_partition(result, smap, output_dir / "zone_partition.png")
    plot_eigengap(result, output_dir / "eigengap.png")

    if resolution_data:
        plot_multi_resolution_scaling(
            resolution_data, output_dir / "k_scaling.png",
        )

    logger.info("All HPS visualizations exported to %s", output_dir)
