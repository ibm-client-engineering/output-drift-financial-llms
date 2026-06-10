#!/usr/bin/env python3
# MIT License — DFAH-Bench
"""Reproduce every published DFAH-Bench paper number from raw run logs.

This is the one-command verification entry point behind `make reproduce-paper`.
It re-runs the full replay-analysis pipeline against the raw episode logs and
FAILS LOUDLY (non-zero exit) if anything diverges from the committed reference
results or from the numbers printed in the paper.

What it checks, in order:

  1. Episode accounting — 8,129 raw episodes -> 8,127 analyzed across
     1,338 case groups and 30 benchmark-model configurations.
  2. Pipeline regeneration — every compute_*.py script re-runs from raw logs.
  3. Reference diff — regenerated CSVs must match the committed reference
     CSVs (numeric-aware comparison, exact for counts, tiny float tolerance
     for cross-platform last-ULP differences).
  4. Paper-claim assertions — the headline numbers quoted in the paper text
     and tables are asserted against the regenerated CSVs directly, so a
     silently edited reference CSV cannot mask drift.
  5. Subsampling robustness — C(8,3) = 56 subsets, Spearman rho = 1.0
     (stdout assertion).

On failure the committed reference CSVs are restored and the divergent
regenerated CSVs are preserved under build/repro/regenerated/ for inspection.

Usage:
    python3 scripts/reproduce_paper.py
    python3 scripts/reproduce_paper.py --skip-bootstrap   # fast mode (~CI)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
RESULTS = REPO_ROOT / "results"
RUN_LOGS = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
BUILD = REPO_ROOT / "build" / "repro"
REFERENCE = BUILD / "reference"
REGENERATED = BUILD / "regenerated"

# Pipeline stages, in dependency order. Each re-runs from raw logs (or from
# CSVs produced by an earlier stage) and rewrites its results/ outputs.
PIPELINE = [
    "compute_dfah_metrics.py",
    "compute_dcb_across_case.py",
    "compute_dfah_accuracy.py",
    "compute_kappa.py",
    "compute_gt_baselines.py",
    "compute_tool_call_counts.py",
    "compute_task_gap_cis.py",
    "compute_bootstrap_cis.py",          # B=10,000, seed=42 (slowest stage)
    "compute_accuracy_metric_correlations.py",
]

BOOTSTRAP_STAGES = {"compute_bootstrap_cis.py", "compute_task_gap_cis.py"}

# Outputs introduced after the reference snapshot was first cut. They are
# diffed when a reference copy exists and adopted otherwise.
NEW_OUTPUTS = {"dfah_kill_criterion_by_model.csv"}

# Tolerance for float comparison: zero would be ideal (same machine, same
# libs, fixed seeds) but cross-platform BLAS/accumulation can shift the last
# ULP. Counts and integers are compared exactly.
RTOL = 1e-9
ATOL = 1e-12


class Failure(Exception):
    pass


_failures: list[str] = []
_passes: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        _passes.append(label)
        print(f"  PASS  {label}")
    else:
        _failures.append(f"{label}  {detail}")
        print(f"  FAIL  {label}  {detail}")


def approx(a: float, b: float, places: int = 3) -> bool:
    return round(float(a), places) == round(float(b), places)


# ---------------------------------------------------------------------------
# Stage 1 — episode accounting from raw logs
# ---------------------------------------------------------------------------

def check_episode_accounting() -> None:
    print("\n[1/5] Episode accounting from raw logs")
    light = [p for p in RUN_LOGS.rglob("case_*_run_*.json")
             if "_full" not in p.name]
    full = list(RUN_LOGS.rglob("*_full.json"))
    check("raw light episode logs == 8,129", len(light) == 8129,
          f"found {len(light)}")
    check("full-log variants == 4,697", len(full) == 4697,
          f"found {len(full)}")


# ---------------------------------------------------------------------------
# Stage 2 — regenerate the pipeline
# ---------------------------------------------------------------------------

def snapshot_reference() -> None:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for csv in sorted(RESULTS.glob("*.csv")):
        shutil.copy2(csv, REFERENCE / csv.name)


def run_pipeline(skip_bootstrap: bool) -> None:
    print("\n[2/5] Regenerating pipeline from raw logs")
    env = dict(os.environ, PYTHONHASHSEED="0")
    for script in PIPELINE:
        if skip_bootstrap and script in BOOTSTRAP_STAGES:
            print(f"  SKIP  {script} (--skip-bootstrap)")
            continue
        print(f"  RUN   {script}")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / script)],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-15:])
            raise Failure(f"{script} exited {proc.returncode}:\n{tail}")


# ---------------------------------------------------------------------------
# Stage 3 — diff regenerated CSVs against the committed reference
# ---------------------------------------------------------------------------

def _frames_match(ref: pd.DataFrame, new: pd.DataFrame) -> tuple[bool, str]:
    if list(ref.columns) != list(new.columns):
        return False, f"columns differ: {list(ref.columns)} vs {list(new.columns)}"
    if len(ref) != len(new):
        return False, f"row count differs: {len(ref)} vs {len(new)}"
    for col in ref.columns:
        r, n = ref[col], new[col]
        r_float = (pd.api.types.is_numeric_dtype(r)
                   and not pd.api.types.is_bool_dtype(r))
        n_float = (pd.api.types.is_numeric_dtype(n)
                   and not pd.api.types.is_bool_dtype(n))
        if r_float and n_float:
            rf, nf = r.astype(float), n.astype(float)
            both_nan = rf.isna() & nf.isna()
            close = ((rf - nf).abs() <= (ATOL + RTOL * nf.abs())) | both_nan
            if not close.all():
                i = int(close.idxmin())
                return False, f"col '{col}' row {i}: {r[i]!r} vs {n[i]!r}"
        else:
            # Bool and string columns: exact comparison (compare as strings
            # so bool dtype vs "True"/"False" object dtype still matches).
            rs = r.astype(str).fillna("<NA>")
            ns = n.astype(str).fillna("<NA>")
            if not (rs == ns).all():
                neq = rs != ns
                i = int(neq.idxmax())
                return False, f"col '{col}' row {i}: {r[i]!r} vs {n[i]!r}"
    return True, ""


def diff_against_reference(skip_bootstrap: bool) -> None:
    print("\n[3/5] Diffing regenerated CSVs against committed reference")
    bootstrap_outputs = {"dfah_model_cis.csv", "dfah_task_gap_cis.csv"}
    for ref_csv in sorted(REFERENCE.glob("*.csv")):
        name = ref_csv.name
        if skip_bootstrap and name in bootstrap_outputs:
            print(f"  SKIP  {name} (--skip-bootstrap)")
            continue
        regen = RESULTS / name
        if not regen.exists():
            check(f"{name} regenerated", False, "missing after pipeline run")
            continue
        ref_df = pd.read_csv(ref_csv)
        new_df = pd.read_csv(regen)
        ok, detail = _frames_match(ref_df, new_df)
        check(f"{name} matches reference", ok, detail)
    for name in sorted(NEW_OUTPUTS):
        regen = RESULTS / name
        check(f"{name} produced", regen.exists(),
              "expected new output missing")


# ---------------------------------------------------------------------------
# Stage 4 — paper-claim assertions (independent of the reference CSVs)
# ---------------------------------------------------------------------------

def check_paper_claims(skip_bootstrap: bool) -> None:
    print("\n[4/5] Asserting paper-quoted numbers against regenerated CSVs")

    case = pd.read_csv(RESULTS / "dfah_case_level.csv")
    model = pd.read_csv(RESULTS / "dfah_model_level.csv").set_index("model")
    kill = pd.read_csv(RESULTS / "dfah_kill_criterion.csv").set_index("criterion")

    # --- Episode accounting (paper §5, footnote) ---
    check("analyzed episodes == 8,127", int(case["n_runs"].sum()) == 8127,
          f"got {int(case['n_runs'].sum())}")
    check("case groups == 1,338", len(case) == 1338, f"got {len(case)}")
    n_configs = case.groupby(["benchmark", "model"]).ngroups
    check("benchmark-model configs == 30", n_configs == 30, f"got {n_configs}")
    n_dist = sorted(case["n_runs"].unique().tolist())
    check("N in {3..8} replays per group", set(n_dist) <= set(range(3, 9)),
          f"got {n_dist}")

    # --- Table 1 (model-level, task-averaged) ---
    sonnet = model.loc["claude-sonnet-4-20250514"]
    check("Sonnet DAR == 0.947", approx(sonnet["mean_dar"], 0.947))
    check("Sonnet TAR == 0.767", approx(sonnet["mean_tar"], 0.767))
    check("Sonnet gap == 0.180", approx(sonnet["mean_dar_tar_gap"], 0.180))
    check("Sonnet ECD == 0.250", approx(sonnet["mean_ecd"], 0.250))

    opus = model.loc["claude-opus-4-20250514"]
    check("Opus DAR == 0.902", approx(opus["mean_dar"], 0.902))
    check("Opus TAR == 0.742", approx(opus["mean_tar"], 0.742))
    check("Opus gap == 0.160", approx(opus["mean_dar_tar_gap"], 0.160))

    qwen = model.loc["qwen2.5_7b-instruct"]
    check("Qwen2.5 DAR == 0.998", approx(qwen["mean_dar"], 0.998))
    check("Qwen2.5 TAR == 0.998", approx(qwen["mean_tar"], 0.998))
    check("Qwen2.5 gap == 0.000", approx(qwen["mean_dar_tar_gap"], 0.0))

    # --- Cross-case DCB (Table 1 DCB column) ---
    dcb = pd.read_csv(RESULTS / "dfah_dcb_across_case_model.csv").set_index("model")
    check("Sonnet DCB == 0.408",
          approx(dcb.loc["claude-sonnet-4-20250514", "mean_dcb_across_case"], 0.408))
    check("Qwen2.5 DCB == 0.352",
          approx(dcb.loc["qwen2.5_7b-instruct", "mean_dcb_across_case"], 0.352))
    check("Gemma4 DCB == 0.111",
          approx(dcb.loc["gemma4_latest", "mean_dcb_across_case"], 0.111))

    # --- Accuracy column ---
    acc = pd.read_csv(RESULTS / "dfah_model_accuracy.csv").set_index("model")
    check("Sonnet Acc == 36.7%",
          approx(acc.loc["claude-sonnet-4-20250514", "task_weighted_accuracy_pct"], 36.7, 1))
    check("Qwen2.5 Acc == 33.3%",
          approx(acc.loc["qwen2.5_7b-instruct", "task_weighted_accuracy_pct"], 33.3, 1))
    check("Gemma4 Acc == 56.0%",
          approx(acc.loc["gemma4_latest", "task_weighted_accuracy_pct"], 56.0, 1))

    # --- Kappa column ---
    kappa = pd.read_csv(RESULTS / "dfah_kappa_model_level.csv").set_index("model")
    check("Sonnet kappa == 0.558",
          approx(kappa.loc["claude-sonnet-4-20250514", "mean_kappa"], 0.558),
          f"got {kappa.loc['claude-sonnet-4-20250514', 'mean_kappa']}")
    check("Qwen2.5 kappa == 0.992",
          approx(kappa.loc["qwen2.5_7b-instruct", "mean_kappa"], 0.992),
          f"got {kappa.loc['qwen2.5_7b-instruct', 'mean_kappa']}")

    # --- Kill criterion (§4.4 / §5) ---
    check("kill: eligible == 912",
          int(kill.loc["moderate_tar_lt_0.9", "n_eligible"]) == 912)
    check("kill: TAR<0.9 == 199 (21.8%)",
          int(kill.loc["moderate_tar_lt_0.9", "n_divergent"]) == 199
          and approx(kill.loc["moderate_tar_lt_0.9", "pct"], 21.8, 1))
    check("kill: TAR<0.7 == 177 (19.4%)",
          int(kill.loc["strong_tar_lt_0.7", "n_divergent"]) == 177
          and approx(kill.loc["strong_tar_lt_0.7", "pct"], 19.4, 1))

    by_model_path = RESULTS / "dfah_kill_criterion_by_model.csv"
    if by_model_path.exists():
        km = pd.read_csv(by_model_path).set_index("model")
        check("kill: Sonnet diverger rate == 55.6%",
              approx(km.loc["claude-sonnet-4-20250514", "pct_any_variation"], 55.6, 1),
              f"got {km.loc['claude-sonnet-4-20250514', 'pct_any_variation']:.1f}")
        check("kill: Gemini 2.5 Pro diverger rate == 56.6%",
              approx(km.loc["gemini-2.5-pro", "pct_any_variation"], 56.6, 1),
              f"got {km.loc['gemini-2.5-pro', 'pct_any_variation']:.1f}")

    # --- Bootstrap CIs (Table 5 / Appendix; B=10,000, seed=42) ---
    if not skip_bootstrap:
        cis = pd.read_csv(RESULTS / "dfah_model_cis.csv")
        srow = cis[(cis["model_id"] == "claude-sonnet-4-20250514")
                   & (cis["metric"] == "Gap")].iloc[0]
        check("Sonnet gap CI == [0.142, 0.218]",
              approx(srow["ci_lo"], 0.142) and approx(srow["ci_hi"], 0.218),
              f"got [{srow['ci_lo']:.3f}, {srow['ci_hi']:.3f}]")

        task = pd.read_csv(RESULTS / "dfah_task_gap_cis.csv")
        drow = task[(task["benchmark"] == "dataops")
                    & (task["model"] == "claude-sonnet-4-20250514")].iloc[0]
        check("DataOps Sonnet gap 0.273 CI [0.207, 0.340]",
              approx(drow["gap"], 0.273) and approx(drow["gap_ci_lo"], 0.207)
              and approx(drow["gap_ci_hi"], 0.340),
              f"got {drow['gap']:.3f} [{drow['gap_ci_lo']:.3f}, {drow['gap_ci_hi']:.3f}]")


# ---------------------------------------------------------------------------
# Stage 5 — subsampling robustness (stdout assertions)
# ---------------------------------------------------------------------------

def check_subsampling(skip: bool) -> None:
    print("\n[5/5] Subsampling robustness (C(8,3) = 56 subsets, rho = 1.0)")
    if skip:
        print("  SKIP  n3_subsampling_sensitivity.py (--skip-subsampling)")
        return
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "n3_subsampling_sensitivity.py")],
        cwd=REPO_ROOT, env=dict(os.environ, PYTHONHASHSEED="0"),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        check("n3 subsampling runs", False,
              "\n".join(proc.stderr.splitlines()[-10:]))
        return
    out = proc.stdout
    check("enumerates 56 subsets", bool(re.search(r"\b56\b", out)))
    m = re.search(r"Total complete N=8 case groups subsampled:\s*(\d+)", out)
    check("568 case groups subsampled", bool(m) and m.group(1) == "568",
          f"got {m.group(1) if m else 'no total line in output'}")
    rho = re.findall(r"rho\s*[=:]\s*(1\.0+|0\.\d+)", out, flags=re.IGNORECASE)
    check("Spearman rho == 1.0", any(v.startswith("1.0") for v in rho),
          f"found rho values {rho[:5]}")


# ---------------------------------------------------------------------------

def restore_reference() -> None:
    for ref_csv in REFERENCE.glob("*.csv"):
        shutil.copy2(ref_csv, RESULTS / ref_csv.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="skip the B=10,000 bootstrap stages (fast mode)")
    parser.add_argument("--skip-subsampling", action="store_true",
                        help="skip the C(8,3) subsampling stage")
    args = parser.parse_args()

    print("=" * 72)
    print("DFAH-Bench paper reproduction")
    print(f"  repo root: {REPO_ROOT}")
    print(f"  run logs:  {RUN_LOGS}")
    print("=" * 72)

    if not RUN_LOGS.is_dir():
        print(f"FATAL: run logs not found at {RUN_LOGS}")
        return 2

    snapshot_reference()
    REGENERATED.mkdir(parents=True, exist_ok=True)

    try:
        check_episode_accounting()
        run_pipeline(args.skip_bootstrap)
        diff_against_reference(args.skip_bootstrap)
        check_paper_claims(args.skip_bootstrap)
        check_subsampling(args.skip_subsampling)
    except Failure as exc:
        _failures.append(str(exc))
        print(f"\nFATAL: {exc}")
    except Exception as exc:  # any crash must still restore the reference
        import traceback
        traceback.print_exc()
        _failures.append(f"unhandled {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print(f"RESULT: {len(_passes)} passed, {len(_failures)} failed")
    if _failures:
        print("\nFailed checks:")
        for f in _failures:
            print(f"  - {f}")
        # Preserve divergent outputs for inspection, then restore reference.
        for csv in RESULTS.glob("*.csv"):
            shutil.copy2(csv, REGENERATED / csv.name)
        restore_reference()
        print(f"\nReference CSVs restored. Divergent outputs kept in "
              f"{REGENERATED}")
        print("=" * 72)
        return 1

    print("All published numbers regenerate from the raw episode logs.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
