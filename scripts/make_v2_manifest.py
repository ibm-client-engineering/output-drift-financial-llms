#!/usr/bin/env python3
"""Build or verify the corrected arXiv v2 artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = REPO_ROOT / "results" / "v2"
MANIFEST_PATH = V2_DIR / "manifest.json"
FIXTURE = V2_DIR / "fixtures" / "retrospective_episodes.jsonl"
RETROSPECTIVE_FILES = (
    "analysis_case_level.csv",
    "analysis_channel_eligibility.csv",
    "analysis_cluster_cis.csv",
    "analysis_corpus_lineage.csv",
    "analysis_fallback_bound.csv",
    "analysis_first_replay.csv",
    "analysis_flash_leave_one_case_out.csv",
    "analysis_n3_subsample.csv",
    "analysis_permutation.csv",
    "analysis_rq2_decomposition.csv",
    "analysis_rq2_tool_set_changes.csv",
    "analysis_shadow_flag_load.csv",
    "analysis_summary.csv",
    "analysis_task_level.csv",
    "analysis_zero_tool_configuration_exclusions.csv",
    "analysis_zero_tool_inclusion_sensitivity.csv",
)
EXTENSION_FILES = (
    "analysis_local_reconciliation_extension.csv",
    "analysis_local_reconciliation_extension.csv.sha256",
    "analysis_prospective_components.csv",
    "analysis_prospective_components.csv.sha256",
    "analysis_v6_diagnostic.csv",
    "analysis_v6_diagnostic.csv.sha256",
)
CODE_FILES = (
    "scripts/build_v2_analysis.py",
    "scripts/export_v2_fixture.py",
    "scripts/make_v2_manifest.py",
    "scripts/reproduce_paper_v2.py",
)
EXPECTED_LINEAGE = (
    ("raw_replay_ledger", 8129, 1340),
    ("archived_v1_after_singleton_removal", 8127, 1338),
    ("remove_analyzed_portfolio_fixture", 5515, 889),
    ("remove_deepseek_compliance_pilot", 5501, 887),
    ("retain_configurations_with_observed_tool_use", 4157, 719),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            record["data_rows"] = sum(1 for _ in csv.DictReader(handle))
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            record["data_rows"] = sum(1 for line in handle if line.strip())
    return record


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def retrospective_claims() -> dict[str, object]:
    lineage_rows = read_csv(V2_DIR / "retrospective" / "analysis_corpus_lineage.csv")
    lineage = tuple(
        (row["stage"], int(row["episodes"]), int(row["groups"])) for row in lineage_rows
    )
    if lineage != EXPECTED_LINEAGE:
        raise AssertionError(f"unexpected corrected-v2 lineage: {lineage!r}")

    fixture_rows: list[dict[str, Any]] = []
    with FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            fixture_rows.append(json.loads(line))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fixture_rows:
        grouped[(row["task"], row["model"], row["case_id"])].append(row)
    primary_groups = {
        key: rows
        for key, rows in grouped.items()
        if len(rows) >= 2
        and key[0] in {"compliance", "dataops"}
        and key[1] not in {"deepseek-r1_8b", "granite3.3_latest", "mistral_7b"}
        and all(row["decision"] and row["tool_names"] is not None for row in rows)
    }
    primary_rows = [row for rows in primary_groups.values() for row in rows]
    empty_groups = sum(
        any(not row["tool_names"] for row in rows) for rows in primary_groups.values()
    )
    empty_episodes = sum(not row["tool_names"] for row in primary_rows)
    if (
        len(primary_groups),
        len(primary_rows),
        empty_groups,
        empty_episodes,
    ) != (719, 4157, 9, 25):
        raise AssertionError("primary retrospective denominator changed")

    summary = read_csv(V2_DIR / "retrospective" / "analysis_summary.csv")
    opus = next(row for row in summary if row["model"] == "claude-opus-4-20250514")
    if opus["model_name"] != "Claude Opus 4":
        raise AssertionError("Claude Opus 4 identity regression")

    return {
        "scope_wording": ("episodes from configurations with observed tool use"),
        "primary_episodes": len(primary_rows),
        "primary_case_groups": len(primary_groups),
        "primary_tasks": ["compliance", "dataops"],
        "empty_tool_sequence_episodes_retained": empty_episodes,
        "case_groups_with_an_empty_tool_sequence": empty_groups,
        "nonempty_tool_sequence_episodes": len(primary_rows) - empty_episodes,
        "exact_model_identity": {"claude-opus-4-20250514": "Claude Opus 4"},
        "lineage": [
            {"stage": stage, "episodes": episodes, "case_groups": groups}
            for stage, episodes, groups in EXPECTED_LINEAGE
        ],
    }


def extension_claims() -> dict[str, object]:
    for csv_path in sorted((V2_DIR / "extensions").glob("*.csv")):
        sidecar = csv_path.with_suffix(csv_path.suffix + ".sha256")
        if sidecar.read_text(encoding="utf-8").strip() != sha256(csv_path):
            raise AssertionError(f"extension sidecar mismatch: {csv_path.name}")
    diagnostic = read_csv(V2_DIR / "extensions" / "analysis_v6_diagnostic.csv")
    components = read_csv(V2_DIR / "extensions" / "analysis_prospective_components.csv")
    local = read_csv(
        V2_DIR / "extensions" / "analysis_local_reconciliation_extension.csv"
    )
    if {row["publication_schema_version"] for row in diagnostic} != {
        "dfah-prospective-api-diagnostic-aggregate-v1"
    }:
        raise AssertionError("prospective diagnostic schema changed")
    if {row["batch_id"] for row in diagnostic} != {"prospective-api-20260723-01"}:
        raise AssertionError("prospective diagnostic batch ID changed")
    if {row["schema_version"] for row in components} != {
        "dfah-prospective-component-analysis-v1"
    }:
        raise AssertionError("prospective component schema changed")
    if {row["batch_id"] for row in components} != {"prospective-api-20260723-01"}:
        raise AssertionError("prospective component batch ID changed")
    if {row["publication_schema_version"] for row in local} != {
        "dfah-local-reconciliation-aggregate-v1"
    }:
        raise AssertionError("local reconciliation schema changed")
    if sum(int(row["eligible_episodes"]) for row in diagnostic) != 570:
        raise AssertionError("prospective diagnostic episode count changed")
    if sum(int(row["eligible_groups"]) for row in diagnostic) != 190:
        raise AssertionError("prospective diagnostic group count changed")
    if any(row["global_publication_gate_passed"] != "False" for row in diagnostic):
        raise AssertionError("prospective diagnostic gate unexpectedly passed")
    if sum(int(row["eligible_episodes"]) for row in components) != 570:
        raise AssertionError("prospective component denominator changed")
    if sum(int(row["eligible_episodes"]) for row in local) != 792:
        raise AssertionError("local reconciliation denominator changed")
    if sum(int(row["eligible_groups"]) for row in local) != 99:
        raise AssertionError("local reconciliation group count changed")
    return {
        "publication_mode": "aggregate_only",
        "raw_provider_captures_in_release": False,
        "regeneration": (
            "integrity and schema validation only; raw provider captures "
            "remain approval-gated"
        ),
        "v6_diagnostic": {
            "eligible_episodes": 570,
            "eligible_case_groups": 190,
            "global_publication_gate_passed": False,
        },
        "prospective_components": {
            "eligible_episodes": 570,
            "eligible_case_groups": 190,
        },
        "local_reconciliation": {
            "eligible_episodes": 792,
            "eligible_case_groups": 99,
        },
    }


def build_manifest() -> dict[str, object]:
    artifacts = [artifact_record(FIXTURE)]
    artifacts.extend(
        artifact_record(V2_DIR / "retrospective" / name) for name in RETROSPECTIVE_FILES
    )
    artifacts.extend(
        artifact_record(V2_DIR / "extensions" / name) for name in EXTENSION_FILES
    )
    artifacts.extend(artifact_record(REPO_ROOT / name) for name in CODE_FILES)
    return {
        "schema_version": "dfah-arxiv-v2-public-manifest-v1",
        "release": "arXiv v2 corrected analysis",
        "retrospective": retrospective_claims(),
        "prospective_extensions": extension_claims(),
        "archived_v1": {
            "location": "results/*.csv",
            "status": (
                "historical machine outputs retained for lineage; not the "
                "corrected-v2 default"
            ),
            "episodes": 8127,
            "case_groups": 1338,
        },
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = build_manifest()
    if args.write:
        MANIFEST_PATH.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {MANIFEST_PATH}")
        return
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"missing manifest: {MANIFEST_PATH}; use --write")
    actual = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("v2 manifest is stale; regenerate with --write")
    print("Corrected-v2 manifest: PASS")


if __name__ == "__main__":
    main()
