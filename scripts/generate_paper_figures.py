#!/usr/bin/env python3
"""Generate paper figures for DFAH-Bench NeurIPS 2026 submission.

Produces three figures from existing result CSVs:
  1. DAR vs TAR scatter (the money plot)
  2. DCB horizontal bar chart
  3. Task-level DAR-TAR gap heatmap

Usage:
    python scripts/generate_paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "neurips2026" / "figures"

MODEL_CSV = RESULTS_DIR / "dfah_model_level.csv"
TASK_CSV = RESULTS_DIR / "dfah_task_level.csv"

# Display-friendly model names
MODEL_NAMES = {
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "qwen3.5_latest": "Qwen 3.5",
    "granite3.3_latest": "Granite 3.3",
    "mistral_7b": "Mistral 7B",
    "gpt-oss_20b": "GPT-OSS 20B",
    "gemma4_latest": "Gemma 4",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-20250514": "Claude Opus",
    "claude-sonnet-4-20250514": "Claude Sonnet",
    "deepseek-r1_8b": "DeepSeek-R1",
}

# Tier assignments
TIERS = {
    "qwen2.5_7b-instruct": "Pattern Matcher",
    "qwen3.5_latest": "Pattern Matcher",
    "granite3.3_latest": "Pattern Matcher",
    "mistral_7b": "Pattern Matcher",
    "gemma4_latest": "Pattern Matcher",
    "gpt-oss_20b": "Stable Executor",
    "gemini-2.0-flash": "Stable Executor",
    "gemini-2.5-pro": "Trajectory Diverger",
    "claude-opus-4-20250514": "Trajectory Diverger",
    "claude-sonnet-4-20250514": "Trajectory Diverger",
    "deepseek-r1_8b": "Pattern Matcher",
}

TIER_COLORS = {
    "Pattern Matcher": "#4e79a7",
    "Stable Executor": "#59a14f",
    "Trajectory Diverger": "#e15759",
}

TIER_MARKERS = {
    "Pattern Matcher": "s",
    "Stable Executor": "D",
    "Trajectory Diverger": "o",
}

BENCHMARK_LABELS = {
    "compliance": "Compliance",
    "dataops": "DataOps",
    "portfolio": "Portfolio",
}


def fig1_dar_tar_scatter():
    """DAR vs TAR scatter plot — the money plot showing the gap."""
    df = pd.read_csv(MODEL_CSV)
    # Only models with trajectory data
    plot_df = df[df["mean_tar"].notna()].copy()
    plot_df["name"] = plot_df["model"].map(MODEL_NAMES)
    plot_df["tier"] = plot_df["model"].map(TIERS)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Diagonal line (DAR = TAR)
    ax.plot([0.84, 1.005], [0.84, 1.005], "k--", alpha=0.25, linewidth=0.7,
            zorder=1)

    # Subtle shaded region below diagonal
    ax.fill_between([0.84, 1.005], [0.84, 1.005], [0.72, 0.72],
                    alpha=0.03, color="#e15759", zorder=0)

    # Plot points by tier
    for tier in ["Pattern Matcher", "Stable Executor", "Trajectory Diverger"]:
        mask = plot_df["tier"] == tier
        tdf = plot_df[mask]
        ax.scatter(
            tdf["mean_dar"], tdf["mean_tar"],
            c=TIER_COLORS[tier], marker=TIER_MARKERS[tier],
            s=70, label=tier, edgecolors="white", linewidths=0.4, zorder=3,
        )

    # Hand-tuned label positions for each model
    label_cfg = {
        "Qwen 3.5":      dict(xytext=(0.975, 1.008), ha="right"),
        "Gemma 4":        dict(xytext=(1.008, 0.983), ha="left"),
        "Qwen 2.5 7B":   dict(xytext=(1.008, 1.005), ha="left"),
        "GPT-OSS 20B":   dict(xytext=(0.938, 0.963), ha="right"),
        "Gemini Flash":   dict(xytext=(0.963, 0.878), ha="left"),
        "Gemini Pro":     dict(xytext=(0.848, 0.728), ha="center"),
        "Claude Opus":    dict(xytext=(0.892, 0.724), ha="center"),
        "Claude Sonnet":  dict(xytext=(0.955, 0.750), ha="center"),
    }
    for _, row in plot_df.iterrows():
        name = row["name"]
        cfg = label_cfg.get(name, {})
        xytext = cfg.get("xytext", (row["mean_dar"] + 0.005,
                                     row["mean_tar"] - 0.015))
        ha = cfg.get("ha", "left")
        ax.annotate(
            name, (row["mean_dar"], row["mean_tar"]),
            xytext=xytext,
            fontsize=6.5, alpha=0.85, ha=ha,
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.35,
                            linewidth=0.5, shrinkA=3, shrinkB=2),
        )

    ax.set_xlabel("Decision Agreement Rate (DAR)", fontsize=9)
    ax.set_ylabel("Trajectory Agreement Rate (TAR)", fontsize=9)
    ax.set_title("Outcome Stability vs. Trajectory Stability",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85,
              edgecolor="0.8", handletextpad=0.4, borderpad=0.4,
              labelspacing=0.3)
    ax.set_xlim(0.84, 1.015)
    ax.set_ylim(0.72, 1.015)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.12, linewidth=0.5)

    # Subtle gap annotation
    ax.text(0.965, 0.815, "DAR > TAR\n(hidden instability)",
            fontsize=6.5, color="#e15759", alpha=0.55,
            ha="center", style="italic")

    fig.tight_layout(pad=0.5)
    out = FIGURES_DIR / "fig1_dar_tar_scatter.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 1: {out}")


def fig2_dcb_bar():
    """Across-case DCB horizontal bar chart with GT baseline reference line.

    Reads from the across-case DCB CSV (paper-accurate definition:
    concentration of per-case modal decisions across a task's 50 cases).
    """
    dcb_csv = RESULTS_DIR / "dfah_dcb_across_case_model.csv"
    df = pd.read_csv(dcb_csv)
    df["name"] = df["model"].map(MODEL_NAMES)
    df["tier"] = df["model"].map(TIERS)
    df = df.sort_values("mean_dcb_across_case", ascending=True)

    # Ground-truth baseline (task-averaged across the 3 benchmarks)
    GT_DCB = 0.115

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    colors = [TIER_COLORS[t] for t in df["tier"]]
    bars = ax.barh(df["name"], df["mean_dcb_across_case"],
                   color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df["mean_dcb_across_case"]):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, alpha=0.85)

    # GT reference line
    ax.axvline(GT_DCB, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               zorder=0)
    ax.text(GT_DCB + 0.005, -0.6, f"GT baseline\n({GT_DCB:.3f})",
            fontsize=6.5, alpha=0.7, ha="left", va="top")

    ax.set_xlabel("Decision Concentration Bias (across-case, task-averaged)",
                  fontsize=10)
    ax.set_title("Decision Concentration by Model", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, max(df["mean_dcb_across_case"].max() + 0.12, 0.55))
    ax.grid(True, axis="x", alpha=0.15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS[t], label=t)
        for t in ["Pattern Matcher", "Stable Executor", "Trajectory Diverger"]
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_dcb_bar.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2: {out}")


def fig3_gap_heatmap():
    """Task-level DAR-TAR gap heatmap for models with trajectory data."""
    df = pd.read_csv(TASK_CSV)
    # Only rows with trajectory data
    df = df[df["mean_tar"].notna()].copy()
    df["name"] = df["model"].map(MODEL_NAMES)

    # Models with trajectory data, ordered by overall gap
    model_order_ids = [
        "qwen3.5_latest",
        "qwen2.5_7b-instruct",
        "gemma4_latest",
        "gpt-oss_20b",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ]
    model_order = [MODEL_NAMES[m] for m in model_order_ids]
    bench_order = ["compliance", "portfolio", "dataops"]
    bench_labels = [BENCHMARK_LABELS[b] for b in bench_order]

    # Build matrix
    matrix = np.full((len(model_order), len(bench_order)), np.nan)
    for _, row in df.iterrows():
        name = row["name"]
        bench = row["benchmark"]
        if name in model_order and bench in bench_order:
            i = model_order.index(name)
            j = bench_order.index(bench)
            gap = row["mean_dar_tar_gap"] if pd.notna(row["mean_dar_tar_gap"]) else np.nan
            matrix[i, j] = gap

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # Custom colormap: green (low gap) -> yellow -> red (high gap)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=-0.02, vmax=0.30)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Cell text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "---", ha="center", va="center", fontsize=8, color="gray")
            else:
                color = "white" if val > 0.15 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(bench_labels)))
    ax.set_xticklabels(bench_labels, fontsize=9)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=9)
    ax.set_title("DAR−TAR Gap by Task and Model", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("DAR − TAR Gap", fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_gap_heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3: {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating DFAH-Bench paper figures...")
    fig1_dar_tar_scatter()
    fig2_dcb_bar()
    fig3_gap_heatmap()
    print("\nDone. All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
