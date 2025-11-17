#!/usr/bin/env python3
# One composite figure per provider/model from results/aggregate.csv
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
from scipy import stats

# --- Load ---
root = Path(".")
candidates = [root/"results"/"aggregate.csv", root/"aggregate.csv"]
for p in candidates:
    if p.exists():
        agg = pd.read_csv(p)
        break
else:
    raise FileNotFoundError("aggregate.csv not found in ./results/ or ./")

figs = root/"figs"
figs.mkdir(exist_ok=True)

# Helper: safe subset
def sub(df, **kwargs):
    out = df.copy()
    for k, v in kwargs.items():
        out = out[out[k] == v]
    return out

# Tasks to plot in fixed order
TASKS = ["rag", "summary", "sql"]

# --- Plot per (provider, model) ---
for (provider, model), dfpm in agg.groupby(["provider","model"]):
    temps = sorted(dfpm["temp"].dropna().unique().tolist())
    concs = sorted(dfpm["concurrency"].dropna().unique().tolist())

    # Prepare 2x2 composite
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    axes = axes.ravel()

    # Panels A/B/C: drift vs concurrency per task, lines = temp
    for i, task in enumerate(TASKS):
        ax = axes[i]
        has_any = False
        for t in temps:
            dft = sub(dfpm, task=task, temp=t).sort_values("concurrency")
            if dft.empty:
                continue
            has_any = True
            ax.plot(dft["concurrency"], dft["mean_drift"], marker="o", label=f"T={t}")
        ax.set_title(f"{task.upper()} — mean drift")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Normalized Levenshtein")
        ax.grid(True, alpha=0.3)
        if has_any:
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # Panel D: latency & throughput (avg over tasks) vs concurrency, lines = temp
    ax = axes[3]
    for t in temps:
        dft = (dfpm[dfpm["temp"] == t]
               .groupby(["temp","concurrency"], as_index=False)
               .agg(mean_latency_s=("mean_latency_s","mean")))
        dft = dft.sort_values("concurrency")
        if dft.empty:
            continue
        ax.plot(dft["concurrency"], dft["mean_latency_s"], marker="o", label=f"latency (T={t})")

    ax.set_title("Latency & Throughput (avg over tasks)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Mean latency (s)")
    ax.grid(True, alpha=0.3)

    # Twin axis: throughput = concurrency / latency
    ax2 = ax.twinx()
    for t in temps:
        dft = (dfpm[dfpm["temp"] == t]
               .groupby(["temp","concurrency"], as_index=False)
               .agg(mean_latency_s=("mean_latency_s","mean")))
        dft = dft.sort_values("concurrency")
        if dft.empty:
            continue
        thr = dft["concurrency"] / dft["mean_latency_s"]
        ax2.plot(dft["concurrency"], thr, marker="x", linestyle="--", label=f"throughput (T={t})")
    ax2.set_ylabel("Throughput (QPS)")

    # Combined legend for panel D
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines or lines2:
        ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    fig.suptitle(f"Drift & Performance — {provider} / {model}", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])

    model_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', model)
    out = figs / f"figure2_{provider}_{model_safe}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

print("[ok] wrote a single composite figure per provider/model to figs/figure2_*.png")

# --- New plots ---

# Drift surface heatmaps at concurrency=4
for (provider, model), dfpm in agg.groupby(["provider","model"]):
    # Filter to concurrency=4
    df_surface = dfpm[dfpm["concurrency"] == 4]

    if df_surface.empty:
        continue

    # Check if we have temperature and top_p data
    if "top_p" not in df_surface.columns:
        continue

    temps = sorted(df_surface["temp"].dropna().unique())
    top_ps = sorted(df_surface["top_p"].dropna().unique())

    if len(temps) < 2 or len(top_ps) < 2:
        continue  # Need at least 2x2 grid

    # Create small multiples over tasks
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4*n_tasks, 4), dpi=150)
    if n_tasks == 1:
        axes = [axes]

    for i, task in enumerate(TASKS):
        ax = axes[i]
        task_data = df_surface[df_surface["task"] == task]

        if task_data.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{task.upper()}")
            continue

        # Create heatmap matrix
        heatmap_data = np.full((len(temps), len(top_ps)), np.nan)

        for j, temp in enumerate(temps):
            for k, top_p in enumerate(top_ps):
                subset = task_data[(task_data["temp"] == temp) & (task_data["top_p"] == top_p)]
                if not subset.empty:
                    heatmap_data[j, k] = subset["mean_drift"].iloc[0]

        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(top_ps)))
        ax.set_yticks(range(len(temps)))
        ax.set_xticklabels([f"{p:.1f}" for p in top_ps])
        ax.set_yticklabels([f"{t:.1f}" for t in temps])
        ax.set_xlabel("top_p")
        ax.set_ylabel("temperature")
        ax.set_title(f"{task.upper()}")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.6)

    model_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', model)
    fig.suptitle(f"Drift Surface (conc=4) — {provider} / {model}", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.94])

    out = figs / f"figure3_drift_surface_{provider}_{model_safe}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

print("[ok] wrote drift surface heatmaps to figs/figure3_*.png")

# Seed-sweep bar plots for SUMMARY at T=0.0, concurrency=16
for (provider, model), dfpm in agg.groupby(["provider","model"]):
    # Filter to SUMMARY task, T=0.0, concurrency=16
    df_seed = dfpm[
        (dfpm["task"] == "summary") &
        (dfpm["temp"] == 0.0) &
        (dfpm["concurrency"] == 16)
    ]

    if df_seed.empty or "seed" not in df_seed.columns:
        continue

    seeds = sorted(df_seed["seed"].dropna().unique())
    if len(seeds) < 2:
        continue

    # Extract identical_pct values
    identical_pcts = []
    seed_labels = []

    for seed in seeds:
        seed_data = df_seed[df_seed["seed"] == seed]
        if not seed_data.empty:
            identical_pcts.append(seed_data["pct_identical"].iloc[0])
            seed_labels.append(str(int(seed)))

    if not identical_pcts:
        continue

    # Calculate 95% Wilson confidence intervals (approximation)
    n = 100  # Assuming percentage out of 100 trials
    ci_lower = []
    ci_upper = []

    for pct in identical_pcts:
        p = pct / 100.0
        # Wilson interval approximation
        z = 1.96  # 95% CI
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom

        ci_lower.append(max(0, (center - margin) * 100))
        ci_upper.append(min(100, (center + margin) * 100))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    x_pos = np.arange(len(seed_labels))
    bars = ax.bar(x_pos, identical_pcts, alpha=0.7)

    # Add error bars
    errors = [np.array(identical_pcts) - np.array(ci_lower),
              np.array(ci_upper) - np.array(identical_pcts)]
    ax.errorbar(x_pos, identical_pcts, yerr=errors, fmt='none', capsize=5, color='black')

    ax.set_xlabel("Seed")
    ax.set_ylabel("Identical %")
    ax.set_title(f"Seed Sweep (SUMMARY, T=0.0, conc=16) — {provider} / {model}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed_labels)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    model_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', model)
    fig.tight_layout()

    out = figs / f"figure4_seed_sweep_{provider}_{model_safe}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

print("[ok] wrote seed sweep bar plots to figs/figure4_*.png")
