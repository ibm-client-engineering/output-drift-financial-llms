#!/usr/bin/env python3
"""Generate historical workshop figures from results/aggregate.csv."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = ["rag", "summary", "sql"]


def _subset(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    return selected


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)


def _load_aggregate(root: Path) -> pd.DataFrame:
    for candidate in (root / "results" / "aggregate.csv", root / "aggregate.csv"):
        if candidate.exists():
            aggregate = pd.read_csv(candidate)
            break
    else:
        raise FileNotFoundError("aggregate.csv not found in ./results/ or ./")

    if "pct_identical" not in aggregate.columns and "identity_rate" in aggregate.columns:
        aggregate["pct_identical"] = aggregate["identity_rate"]
    return aggregate


def _plot_provider_composites(aggregate: pd.DataFrame, figures: Path) -> None:
    for (provider, model), model_rows in aggregate.groupby(["provider", "model"]):
        temperatures = sorted(model_rows["temp"].dropna().unique().tolist())
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        axes = axes.ravel()

        for index, task in enumerate(TASKS):
            axis = axes[index]
            has_data = False
            for temperature in temperatures:
                task_rows = _subset(
                    model_rows,
                    task=task,
                    temp=temperature,
                ).sort_values("concurrency")
                if task_rows.empty:
                    continue
                has_data = True
                axis.plot(
                    task_rows["concurrency"],
                    task_rows["mean_drift"],
                    marker="o",
                    label=f"T={temperature}",
                )
            axis.set_title(f"{task.upper()} — mean drift")
            axis.set_xlabel("Concurrency")
            axis.set_ylabel("Normalized Levenshtein")
            axis.grid(True, alpha=0.3)
            if has_data:
                axis.legend(frameon=False)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

        latency_axis = axes[3]
        for temperature in temperatures:
            temperature_rows = (
                model_rows[model_rows["temp"] == temperature]
                .groupby(["temp", "concurrency"], as_index=False)
                .agg(mean_latency_s=("mean_latency_s", "mean"))
                .sort_values("concurrency")
            )
            if temperature_rows.empty:
                continue
            latency_axis.plot(
                temperature_rows["concurrency"],
                temperature_rows["mean_latency_s"],
                marker="o",
                label=f"latency (T={temperature})",
            )

        latency_axis.set_title("Latency & Throughput (avg over tasks)")
        latency_axis.set_xlabel("Concurrency")
        latency_axis.set_ylabel("Mean latency (s)")
        latency_axis.grid(True, alpha=0.3)

        throughput_axis = latency_axis.twinx()
        for temperature in temperatures:
            temperature_rows = (
                model_rows[model_rows["temp"] == temperature]
                .groupby(["temp", "concurrency"], as_index=False)
                .agg(mean_latency_s=("mean_latency_s", "mean"))
                .sort_values("concurrency")
            )
            if temperature_rows.empty:
                continue
            throughput = temperature_rows["concurrency"] / temperature_rows["mean_latency_s"]
            throughput_axis.plot(
                temperature_rows["concurrency"],
                throughput,
                marker="x",
                linestyle="--",
                label=f"throughput (T={temperature})",
            )
        throughput_axis.set_ylabel("Throughput (QPS)")

        lines, labels = latency_axis.get_legend_handles_labels()
        throughput_lines, throughput_labels = throughput_axis.get_legend_handles_labels()
        if lines or throughput_lines:
            throughput_axis.legend(
                lines + throughput_lines,
                labels + throughput_labels,
                frameon=False,
                loc="best",
            )

        figure.suptitle(
            f"Drift & Performance — {provider} / {model}",
            y=0.98,
            fontsize=12,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.96])
        output = figures / f"figure2_{provider}_{_safe_model_name(model)}.png"
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)

    print("[ok] wrote a single composite figure per provider/model to figs/figure2_*.png")


def _plot_drift_surfaces(aggregate: pd.DataFrame, figures: Path) -> None:
    for (provider, model), model_rows in aggregate.groupby(["provider", "model"]):
        surface_rows = model_rows[model_rows["concurrency"] == 4]
        if surface_rows.empty or "top_p" not in surface_rows.columns:
            continue

        temperatures = sorted(surface_rows["temp"].dropna().unique())
        top_ps = sorted(surface_rows["top_p"].dropna().unique())
        if len(temperatures) < 2 or len(top_ps) < 2:
            continue

        figure, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 4), dpi=150)
        if len(TASKS) == 1:
            axes = [axes]

        for index, task in enumerate(TASKS):
            axis = axes[index]
            task_rows = surface_rows[surface_rows["task"] == task]
            if task_rows.empty:
                axis.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(task.upper())
                continue

            heatmap = np.full((len(temperatures), len(top_ps)), np.nan)
            for row, temperature in enumerate(temperatures):
                for column, top_p in enumerate(top_ps):
                    selected = task_rows[
                        (task_rows["temp"] == temperature) & (task_rows["top_p"] == top_p)
                    ]
                    if not selected.empty:
                        heatmap[row, column] = selected["mean_drift"].iloc[0]

            image = axis.imshow(heatmap, cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(top_ps)))
            axis.set_yticks(range(len(temperatures)))
            axis.set_xticklabels([f"{value:.1f}" for value in top_ps])
            axis.set_yticklabels([f"{value:.1f}" for value in temperatures])
            axis.set_xlabel("top_p")
            axis.set_ylabel("temperature")
            axis.set_title(task.upper())
            plt.colorbar(image, ax=axis, shrink=0.6)

        figure.suptitle(f"Drift Surface (conc=4) — {provider} / {model}", y=0.98)
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        output = figures / f"figure3_drift_surface_{provider}_{_safe_model_name(model)}.png"
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)

    print("[ok] wrote drift surface heatmaps to figs/figure3_*.png")


def _plot_seed_sweeps(aggregate: pd.DataFrame, figures: Path) -> None:
    for (provider, model), model_rows in aggregate.groupby(["provider", "model"]):
        seed_rows = model_rows[
            (model_rows["task"] == "summary")
            & (model_rows["temp"] == 0.0)
            & (model_rows["concurrency"] == 16)
        ]
        if seed_rows.empty or "seed" not in seed_rows.columns:
            continue

        seeds = sorted(seed_rows["seed"].dropna().unique())
        if len(seeds) < 2:
            continue

        percentages: list[float] = []
        sample_sizes: list[int] = []
        labels: list[str] = []
        for seed in seeds:
            selected = seed_rows[seed_rows["seed"] == seed]
            if not selected.empty:
                percentages.append(selected["pct_identical"].iloc[0])
                sample_sizes.append(int(selected["runs"].iloc[0]))
                labels.append(str(int(seed)))
        if not percentages:
            continue

        lower_bounds = []
        upper_bounds = []
        for percentage, sample_size in zip(percentages, sample_sizes, strict=True):
            if sample_size <= 0:
                raise ValueError("Seed-sweep aggregate contains a non-positive run count")
            proportion = percentage / 100.0
            z_score = 1.96
            denominator = 1 + z_score**2 / sample_size
            center = (proportion + z_score**2 / (2 * sample_size)) / denominator
            margin = (
                z_score
                * np.sqrt(
                    proportion * (1 - proportion) / sample_size
                    + z_score**2 / (4 * sample_size**2)
                )
                / denominator
            )
            lower_bounds.append(max(0, (center - margin) * 100))
            upper_bounds.append(min(100, (center + margin) * 100))

        figure, axis = plt.subplots(figsize=(8, 6), dpi=150)
        positions = np.arange(len(labels))
        axis.bar(positions, percentages, alpha=0.7)
        errors = [
            np.array(percentages) - np.array(lower_bounds),
            np.array(upper_bounds) - np.array(percentages),
        ]
        axis.errorbar(
            positions,
            percentages,
            yerr=errors,
            fmt="none",
            capsize=5,
            color="black",
        )
        axis.set_xlabel("Seed")
        axis.set_ylabel("Identical %")
        axis.set_title(f"Seed Sweep (SUMMARY, T=0.0, conc=16) — {provider} / {model}")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 105)
        figure.tight_layout()

        output = figures / f"figure4_seed_sweep_{provider}_{_safe_model_name(model)}.png"
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)

    print("[ok] wrote seed sweep bar plots to figs/figure4_*.png")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate historical output-drift workshop figures"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _parse_args(argv)
    root = Path(".")
    aggregate = _load_aggregate(root)
    figures = root / "figs"
    figures.mkdir(exist_ok=True)

    _plot_provider_composites(aggregate, figures)
    _plot_drift_surfaces(aggregate, figures)
    _plot_seed_sweeps(aggregate, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
