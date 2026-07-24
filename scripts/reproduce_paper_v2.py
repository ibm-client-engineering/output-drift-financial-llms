#!/usr/bin/env python3
"""Regenerate and validate the corrected arXiv v2 public artifacts offline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = REPO_ROOT / "results" / "v2"
RETROSPECTIVE_DIR = V2_DIR / "retrospective"
EXTENSION_DIR = V2_DIR / "extensions"
FIXTURE = V2_DIR / "fixtures" / "retrospective_episodes.jsonl"
BUILDER = REPO_ROOT / "scripts" / "build_v2_analysis.py"
MANIFEST_CHECKER = REPO_ROOT / "scripts" / "make_v2_manifest.py"
PRIVATE_PATTERN = re.compile(
    r"(?:/Users/[^/\s]+|/home/[^/\s]+|/private/(?:tmp|var)/[^\s,\"']+|"
    r"/tmp/[^\s,\"']+|[A-Za-z]:\\\\Users\\\\[^\\\\\s]+)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{12,}|bearer\s+[a-z0-9._-]{12,}|"
    r"(?:api[_-]?key|password|secret|credential)\s*[:=])"
)


def validate_public_boundary() -> None:
    """Fail if internal venue/review-system identifiers enter public text."""
    venue_token = "ic" + "aif"
    review_token = "cm" + "t"
    excluded_parts = {".git", "__pycache__", "site"}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lowered = text.lower()
        if venue_token in lowered or re.search(rf"\b{review_token}\b", lowered):
            raise AssertionError(
                f"non-public conference identifier in {path.relative_to(REPO_ROOT)}"
            )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _numeric_equal(left: str, right: str) -> bool:
    try:
        left_number = Decimal(left)
        right_number = Decimal(right)
    except InvalidOperation:
        return False
    if left_number.is_nan() and right_number.is_nan():
        return True
    delta = abs(left_number - right_number)
    scale = max(abs(left_number), abs(right_number), Decimal(1))
    return delta <= max(Decimal("1e-12"), Decimal("1e-10") * scale)


def compare_csv(expected_path: Path, generated_path: Path) -> None:
    expected_header, expected_rows = read_csv(expected_path)
    generated_header, generated_rows = read_csv(generated_path)
    if expected_header != generated_header:
        raise AssertionError(f"column mismatch: {expected_path.name}")
    if len(expected_rows) != len(generated_rows):
        raise AssertionError(f"row-count mismatch: {expected_path.name}")
    for row_number, (expected, generated) in enumerate(
        zip(expected_rows, generated_rows),  # noqa: B905 - lengths checked above
        start=2,
    ):
        for column in expected_header:
            if expected[column] == generated[column]:
                continue
            if _numeric_equal(expected[column], generated[column]):
                continue
            raise AssertionError(
                f"{expected_path.name}:{row_number}:{column}: "
                f"{expected[column]!r} != {generated[column]!r}"
            )


def validate_fixture_privacy() -> None:
    allowed = {
        "task",
        "model",
        "case_id",
        "replay",
        "decision",
        "tool_names",
    }
    count = 0
    with FIXTURE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if set(row) != allowed:
                raise AssertionError(f"fixture row {line_number} has non-public fields")
            serialized = json.dumps(row, sort_keys=True)
            if PRIVATE_PATTERN.search(serialized) or SECRET_PATTERN.search(serialized):
                raise AssertionError(f"fixture row {line_number} failed privacy scan")
            count += 1
    if count != 8129:
        raise AssertionError(f"fixture count changed: {count}")


def validate_retrospective_claims() -> None:
    _, case_rows = read_csv(RETROSPECTIVE_DIR / "analysis_case_level.csv")
    if len(case_rows) != 719:
        raise AssertionError("corrected case-group count changed")
    if sum(int(row["n_runs"]) for row in case_rows) != 4157:
        raise AssertionError("corrected episode count changed")
    if {row["task"] for row in case_rows} != {"compliance", "dataops"}:
        raise AssertionError("corrected task scope changed")
    if any("accuracy" in column.lower() or "ecd" in column.lower() for column in case_rows[0]):
        raise AssertionError("withdrawn metric leaked into corrected artifacts")

    _, summary = read_csv(RETROSPECTIVE_DIR / "analysis_summary.csv")
    opus = next(row for row in summary if row["model"] == "claude-opus-4-20250514")
    if opus["model_name"] != "Claude Opus 4":
        raise AssertionError("Claude Opus 4 identity changed")

    _, exclusions = read_csv(
        RETROSPECTIVE_DIR / "analysis_zero_tool_configuration_exclusions.csv"
    )
    if (
        sum(int(row["case_groups"]) for row in exclusions),
        sum(int(row["episodes"]) for row in exclusions),
        sum(int(row["observed_tool_calls"]) for row in exclusions),
    ) != (168, 1344, 0):
        raise AssertionError("zero-tool configuration exclusion changed")

    _, decomposition = read_csv(RETROSPECTIVE_DIR / "analysis_rq2_decomposition.csv")
    pooled = next(row for row in decomposition if row["scope"] == "__all__")
    observed = tuple(
        int(pooled[column])
        for column in (
            "unanimous_groups",
            "sequence_varying",
            "reorder_only",
            "multiplicity_changed_same_set",
            "tool_set_changed",
        )
    )
    if observed != (627, 122, 17, 58, 47):
        raise AssertionError(f"RQ2 decomposition changed: {observed!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_extensions() -> None:
    for csv_path in sorted(EXTENSION_DIR.glob("*.csv")):
        public_text = csv_path.read_text(encoding="utf-8")
        if PRIVATE_PATTERN.search(public_text) or SECRET_PATTERN.search(public_text):
            raise AssertionError(f"aggregate failed privacy scan: {csv_path.name}")
        sidecar = csv_path.with_suffix(csv_path.suffix + ".sha256")
        expected = sidecar.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AssertionError(f"invalid digest sidecar: {sidecar.name}")
        if sha256(csv_path) != expected:
            raise AssertionError(f"aggregate digest mismatch: {csv_path.name}")

    _, diagnostic = read_csv(EXTENSION_DIR / "analysis_v6_diagnostic.csv")
    if (
        sum(int(row["eligible_episodes"]) for row in diagnostic),
        sum(int(row["eligible_groups"]) for row in diagnostic),
    ) != (570, 190):
        raise AssertionError("prospective diagnostic denominator changed")
    if any(row["global_publication_gate_passed"] != "False" for row in diagnostic):
        raise AssertionError("prospective publication gate unexpectedly passed")

    _, components = read_csv(EXTENSION_DIR / "analysis_prospective_components.csv")
    expected = {
        "gpt-5.6-terra": (
            0.951304347826087,
            0.5150724637681159,
            0.5434782608695652,
        ),
        "claude-sonnet-5": (
            0.9415151515151514,
            0.45000000000000007,
            0.5690909090909092,
        ),
    }
    for row in components:
        observed = (
            float(row["decision_agreement"]),
            float(row["tar_name_canonical_arguments"]),
            float(row["tar_result_only"]),
        )
        if row["model"] not in expected or not all(
            math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(  # noqa: B905 - both tuples have three metrics
                observed, expected[row["model"]]
            )
        ):
            raise AssertionError(f"prospective component metric changed: {row['model']}")


def main() -> None:
    validate_public_boundary()
    validate_fixture_privacy()
    with tempfile.TemporaryDirectory(prefix="dfah-v2-reproduce-") as temp:
        generated_dir = Path(temp)
        subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--fixture",
                str(FIXTURE),
                "--output-dir",
                str(generated_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        expected_files = sorted(path.name for path in RETROSPECTIVE_DIR.glob("*.csv"))
        generated_files = sorted(path.name for path in generated_dir.glob("*.csv"))
        if generated_files != expected_files:
            raise AssertionError("corrected-v2 artifact inventory changed")
        for name in expected_files:
            compare_csv(
                RETROSPECTIVE_DIR / name,
                generated_dir / name,
            )

    validate_retrospective_claims()
    validate_extensions()
    subprocess.run(
        [sys.executable, str(MANIFEST_CHECKER), "--check"],
        cwd=REPO_ROOT,
        check=True,
    )
    print(
        "Corrected arXiv v2 reproduction: PASS "
        "(4,157 episodes from configurations with observed tool use; "
        "719 groups)"
    )
    print(
        "Prospective extensions: aggregate hashes/schema/claims PASS; "
        "raw provider captures are approval-gated and are not regenerated "
        "by this public target."
    )


if __name__ == "__main__":
    main()
