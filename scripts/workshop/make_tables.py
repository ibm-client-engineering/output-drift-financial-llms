#!/usr/bin/env python3
"""Generate the historical workshop tables and composite table figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLS_BASE = [
    "task",
    "provider",
    "model",
    "concurrency",
    "runs",
    "pct_identical",
    "mean_drift",
    "factual_drift_rate",
    "schema_violation_rate",
    "decision_flip_rate",
    "mean_latency_s",
]
COLS_ALL = [
    "task",
    "provider",
    "model",
    "temp",
    "concurrency",
    "runs",
    "pct_identical",
    "mean_drift",
    "factual_drift_rate",
    "schema_violation_rate",
    "decision_flip_rate",
    "mean_latency_s",
]
COLS_CROSS_PROVIDER = [
    "task",
    "provider",
    "model",
    "temp",
    "pct_identical",
    "mean_drift",
    "mean_latency_s",
]
RENAME = {
    "task": "Task",
    "provider": "Prov.",
    "model": "Model",
    "temp": "Temp",
    "concurrency": "Conc.",
    "runs": "Runs",
    "pct_identical": "Identical (%)",
    "mean_drift": "Mean drift",
    "factual_drift_rate": "Fact. rate",
    "schema_violation_rate": "Schema rate",
    "decision_flip_rate": "Flip rate",
    "mean_latency_s": "Lat. (s)",
}
COLFMT_BASE = "p{1.3cm}p{1.2cm}p{3.0cm}" + "r" * 8
COLFMT_ALL = "p{1.3cm}p{1.2cm}p{3.0cm}r" + "r" * 8
COLFMT_CROSS = "p{1.3cm}p{1.2cm}p{3.0cm}r" + "r" * 3


def _repository_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repository_root()


def _prepare(frame: pd.DataFrame, columns: list[str], sort_columns: list[str]) -> pd.DataFrame:
    return frame[columns].copy().sort_values(sort_columns).rename(columns=RENAME)


def _load_frames(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(source)
    if "pct_identical" not in frame.columns and "identity_rate" in frame.columns:
        frame["pct_identical"] = frame["identity_rate"]

    baseline = _prepare(frame[frame["temp"] == 0.0], COLS_BASE, ["task", "concurrency"])
    all_rows = _prepare(frame, COLS_ALL, ["task", "temp", "concurrency"])
    cross_provider = frame[frame["concurrency"].isin([1, 16]) & frame["temp"].isin([0.0, 0.2])]
    cross_provider = _prepare(
        cross_provider,
        COLS_CROSS_PROVIDER,
        ["task", "provider", "model", "temp"],
    )
    return frame, baseline, all_rows, cross_provider


def _to_tex(frame: pd.DataFrame, column_format: str) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        float_format="%.3f",
        column_format=column_format,
        longtable=False,
    )


def _extract_latex_rows(tex_content: str) -> str:
    """Return data rows from either booktabs or classic tabular output."""
    lines = tex_content.splitlines()
    stripped = [line.strip() for line in lines]

    if "\\midrule" in stripped:
        start = stripped.index("\\midrule") + 1
    else:
        hlines = [index for index, line in enumerate(stripped) if line == "\\hline"]
        if not hlines:
            raise ValueError("LaTeX table has no header boundary")
        start = hlines[1] + 1 if len(hlines) > 1 else hlines[0] + 1

    end = len(lines)
    for index in range(start, len(lines)):
        if stripped[index] in {"\\bottomrule", "\\hline", "\\end{tabular}"}:
            end = index
            break

    body = [line for line in lines[start:end] if line.strip()]
    if not body:
        raise ValueError("LaTeX table contains no data rows")
    return "\n".join(body) + "\n"


def _format_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if formatted[column].dtype.kind == "f":
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    return formatted


def _style_table(table, frame: pd.DataFrame) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.3, 2.2)
    for index in range(len(frame.columns)):
        table[(0, index)].set_facecolor("#40466e")
        table[(0, index)].set_text_props(weight="bold", color="white", wrap=True)
    for row in range(len(frame)):
        for column in range(len(frame.columns)):
            table[(row + 1, column)].set_text_props(wrap=True)


def _render_two_tables_png(
    top: pd.DataFrame,
    top_title: str,
    bottom: pd.DataFrame,
    bottom_title: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(20, 16), dpi=150)
    grid = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.4)

    top_axis = figure.add_subplot(grid[0])
    top_axis.axis("off")
    top_axis.set_title(top_title, fontsize=16, pad=25, weight="bold")
    top_table = top_axis.table(
        cellText=top.values,
        colLabels=top.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    _style_table(top_table, top)

    bottom_axis = figure.add_subplot(grid[1])
    bottom_axis.axis("off")
    bottom_axis.set_title(bottom_title, fontsize=16, pad=25, weight="bold")
    bottom_table = bottom_axis.table(
        cellText=bottom.values,
        colLabels=bottom.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    _style_table(bottom_table, bottom)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=150, pad_inches=0.5)
    plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-only",
        action="store_true",
        help="Output rows-only version without a tabular wrapper",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source = Path("results/aggregate.csv")
    output_directory = Path("tables")
    output_directory.mkdir(exist_ok=True)
    frame, baseline, all_rows, cross_provider = _load_frames(source)

    if args.rows_only:
        outputs = {
            "table_1_baseline_rows.tex": _extract_latex_rows(_to_tex(baseline, COLFMT_BASE)),
            "table_2_all_rows.tex": _extract_latex_rows(_to_tex(all_rows, COLFMT_ALL)),
            "table_3_cross_provider_rows.tex": _extract_latex_rows(
                _to_tex(cross_provider, COLFMT_CROSS)
            ),
        }
        for filename, content in outputs.items():
            (output_directory / filename).write_text(content, encoding="utf-8")
        print("[ok] wrote rows-only tables")
    else:
        outputs = {
            "table_1_baseline.tex": _to_tex(baseline, COLFMT_BASE),
            "table_2_all.tex": _to_tex(all_rows, COLFMT_ALL),
            "table_3_cross_provider.tex": _to_tex(cross_provider, COLFMT_CROSS),
        }
        for filename, content in outputs.items():
            (output_directory / filename).write_text(content, encoding="utf-8")
        print("[ok] wrote workshop LaTeX tables")

    baseline_original = (
        frame[frame["temp"] == 0.0][COLS_BASE]
        .sort_values(["task", "concurrency"])
        .reset_index(drop=True)
    )
    all_rows_original = (
        frame[COLS_ALL].sort_values(["task", "temp", "concurrency"]).reset_index(drop=True)
    )
    figure_path = REPO_ROOT / "figs" / "figure1_tables.png"
    _render_two_tables_png(
        _format_frame(baseline_original),
        "Table 1 (baseline): temp=0.0",
        _format_frame(all_rows_original),
        "Table 2 (all): includes temp ∈ {0.0, 0.2}",
        figure_path,
    )
    print("[ok] wrote figs/figure1_tables.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
