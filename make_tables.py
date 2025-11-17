#!/usr/bin/env python3
# Writes LaTeX tables with wrapped text columns and short headers.
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import argparse

src = Path("results/aggregate.csv")
outdir = Path("tables"); outdir.mkdir(exist_ok=True)

df = pd.read_csv(src)

# Keep columns and give short, print-friendly headers
COLS_BASE = [
    "task","provider","model","concurrency","runs",
    "pct_identical","mean_drift","factual_drift_rate",
    "schema_violation_rate","decision_flip_rate","mean_latency_s"
]
COLS_ALL = [
    "task","provider","model","temp","concurrency","runs",
    "pct_identical","mean_drift","factual_drift_rate",
    "schema_violation_rate","decision_flip_rate","mean_latency_s"
]
COLS_CROSS_PROVIDER = [
    "task","provider","model","temp","pct_identical","mean_drift","mean_latency_s"
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

# Format + sort
def prep(df, cols, sort_cols):
    out = df[cols].copy()
    out = out.sort_values(sort_cols).rename(columns=RENAME)
    return out

baseline = prep(df[df["temp"]==0.0], COLS_BASE, ["task","concurrency"])
all_rows = prep(df, COLS_ALL, ["task","temp","concurrency"])

# Cross-provider table: filter concurrency in {1,16} and temps in {0.0,0.2}
cross_provider = df[(df["concurrency"].isin([1, 16])) & (df["temp"].isin([0.0, 0.2]))]
cross_provider = prep(cross_provider, COLS_CROSS_PROVIDER, ["task","provider","model","temp"])

# Column formats:
# p{..} for the 3 text columns to allow wrapping; r for numbers.
# Choose slightly wider model column since it carries "qwen2.5:7b-instruct".
colfmt_base = "p{1.3cm}p{1.2cm}p{3.0cm}" + "r"*8
colfmt_all  = "p{1.3cm}p{1.2cm}p{3.0cm}r" + "r"*8  # (+temp)

def to_tex(df, colfmt):
    # Use escape=True so underscores are handled, shorter float format for readability
    tex = df.to_latex(
        index=False,
        escape=True,
        float_format="%.3f",
        column_format=colfmt,
        longtable=False
    )
    # to_latex emits \begin{tabular} ... keep it; we'll control width from main .tex with adjustbox
    return tex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-only", action="store_true",
                        help="Output rows-only version (no tabular wrapper)")
    args = parser.parse_args()

    if args.rows_only:
        # Output rows-only version without tabular wrapper
        # Extract just the body rows by parsing the LaTeX output
        baseline_tex = to_tex(baseline, colfmt_base)
        all_rows_tex = to_tex(all_rows, colfmt_all)
        colfmt_cross = "p{1.3cm}p{1.2cm}p{3.0cm}r" + "r"*3
        cross_provider_tex = to_tex(cross_provider, colfmt_cross)

        # Extract body rows (skip header and tabular wrapper)
        def extract_rows(tex_content):
            lines = tex_content.split('\n')
            body_lines = []
            in_body = False
            for line in lines:
                if line.strip().startswith('\\hline') and not in_body:
                    in_body = True
                    continue
                elif line.strip().startswith('\\end{tabular}'):
                    break
                elif in_body and not line.strip().startswith('\\hline'):
                    body_lines.append(line)
            return '\n'.join(body_lines)

        (outdir/"table_1_baseline_rows.tex").write_text(extract_rows(baseline_tex))
        (outdir/"table_2_all_rows.tex").write_text(extract_rows(all_rows_tex))
        (outdir/"table_3_cross_provider_rows.tex").write_text(extract_rows(cross_provider_tex))
        print("[ok] wrote rows-only tables")
    else:
        # Regular tabular output
        (outdir/"table_1_baseline.tex").write_text(to_tex(baseline, colfmt_base))
        (outdir/"table_2_all.tex").write_text(to_tex(all_rows, colfmt_all))

        # Cross-provider table with appropriate column format
        colfmt_cross = "p{1.3cm}p{1.2cm}p{3.0cm}r" + "r"*3
        (outdir/"table_3_cross_provider.tex").write_text(to_tex(cross_provider, colfmt_cross))

        print("[ok] wrote tables/table_1_baseline.tex, tables/table_2_all.tex, and tables/table_3_cross_provider.tex")

# ---- Composite Figure 1 (PNG with both tables stacked) ----
HERE = Path(__file__).parent
os.makedirs(HERE/"figs", exist_ok=True)

def _format_df(d):
    # Show blanks (—not NaN text) for NAs; 3 decimals like LaTeX output
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind in "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return d

# For figure generation, use original column names
baseline_orig = df[df["temp"]==0.0][COLS_BASE].sort_values(["task","concurrency"]).reset_index(drop=True)
all_rows_orig = df[COLS_ALL].sort_values(["task","temp","concurrency"]).reset_index(drop=True)

bshow = _format_df(baseline_orig)
ashow = _format_df(all_rows_orig)

def render_two_tables_png(df_top, title_top, df_bottom, title_bottom, outpath):
    # Improved matplotlib table render with better spacing and text wrapping
    fig = plt.figure(figsize=(20, 16), dpi=150)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.4)

    # Top table
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.set_title(title_top, fontsize=16, pad=25, weight='bold')

    table1 = ax1.table(cellText=df_top.values,
                       colLabels=df_top.columns.tolist(),
                       loc="center",
                       cellLoc="center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1.3, 2.2)  # More vertical space

    # Style header row with better text wrapping
    for i in range(len(df_top.columns)):
        table1[(0, i)].set_facecolor('#40466e')
        table1[(0, i)].set_text_props(weight='bold', color='white', wrap=True)

    # Enable text wrapping for all cells
    for i in range(len(df_top)):
        for j in range(len(df_top.columns)):
            table1[(i+1, j)].set_text_props(wrap=True)

    # Bottom table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title(title_bottom, fontsize=16, pad=25, weight='bold')

    table2 = ax2.table(cellText=df_bottom.values,
                       colLabels=df_bottom.columns.tolist(),
                       loc="center",
                       cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.3, 2.2)  # More vertical space

    # Style header row with better text wrapping
    for i in range(len(df_bottom.columns)):
        table2[(0, i)].set_facecolor('#40466e')
        table2[(0, i)].set_text_props(weight='bold', color='white', wrap=True)

    # Enable text wrapping for all cells
    for i in range(len(df_bottom)):
        for j in range(len(df_bottom.columns)):
            table2[(i+1, j)].set_text_props(wrap=True)

    fig.savefig(outpath, bbox_inches="tight", dpi=150, pad_inches=0.5)
    plt.close(fig)

# Generate the composite figure
render_two_tables_png(
    bshow, "Table 1 (baseline): temp=0.0",
    ashow, "Table 2 (all): includes temp ∈ {0.0, 0.2}",
    HERE/"figs"/"figure1_tables.png"
)
print("[ok] wrote figs/figure1_tables.png")

if __name__ == "__main__":
    main()