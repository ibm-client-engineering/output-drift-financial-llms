"""Fail-closed checks for the corrected public arXiv v2 artifact release."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = REPO_ROOT / "results" / "v2"
FIXTURE = V2_DIR / "fixtures" / "retrospective_episodes.jsonl"


def _load_reproducer():
    spec = importlib.util.spec_from_file_location(
        "reproduce_paper_v2",
        REPO_ROOT / "scripts" / "reproduce_paper_v2.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_fixture_has_only_approved_observable_channels() -> None:
    allowed = {
        "task",
        "model",
        "case_id",
        "replay",
        "decision",
        "tool_names",
    }
    with FIXTURE.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 8129
    assert all(set(row) == allowed for row in rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"], row["case_id"])].append(row)
    primary = {
        key: episodes
        for key, episodes in grouped.items()
        if len(episodes) >= 2
        and key[0] in {"compliance", "dataops"}
        and key[1] not in {"deepseek-r1_8b", "granite3.3_latest", "mistral_7b"}
    }
    assert len(primary) == 719
    assert sum(map(len, primary.values())) == 4157
    assert (
        sum(not episode["tool_names"] for episodes in primary.values() for episode in episodes)
        == 25
    )


def test_corrected_model_identity_and_scope() -> None:
    with (V2_DIR / "retrospective" / "analysis_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    opus = next(row for row in rows if row["model"] == "claude-opus-4-20250514")
    assert opus["model_name"] == "Claude Opus 4"

    manifest = json.loads((V2_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["retrospective"]["scope_wording"] == (
        "episodes from configurations with observed tool use"
    )
    assert manifest["retrospective"]["primary_episodes"] == 4157
    assert manifest["retrospective"]["primary_case_groups"] == 719


def test_extension_sidecars_and_manifest_are_current() -> None:
    for artifact in sorted((V2_DIR / "extensions").glob("*.csv")):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
        assert sidecar.read_text(encoding="utf-8").strip() == digest
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_v2_manifest.py"),
            "--check",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_public_boundary_has_no_internal_conference_identifiers() -> None:
    _load_reproducer().validate_public_boundary()


def test_announced_venue_is_allowed_on_curated_page(tmp_path) -> None:
    module = _load_reproducer()
    module.REPO_ROOT = tmp_path
    (tmp_path / "README.md").write_text("Accepted poster at ACM " + "IC" + "AIF 2026")
    module.validate_public_boundary()


def test_venue_details_remain_blocked_outside_curated_pages(tmp_path) -> None:
    module = _load_reproducer()
    module.REPO_ROOT = tmp_path
    (tmp_path / "private-notes.md").write_text("IC" + "AIF review correspondence")
    with pytest.raises(AssertionError, match="non-public conference identifier"):
        module.validate_public_boundary()


def test_private_review_system_is_blocked_even_on_curated_page(tmp_path) -> None:
    module = _load_reproducer()
    module.REPO_ROOT = tmp_path
    (tmp_path / "README.md").write_text("CM" + "T reviewer record")
    with pytest.raises(AssertionError, match="non-public conference identifier"):
        module.validate_public_boundary()


def test_manuscript_source_is_blocked_without_venue_identifier(tmp_path) -> None:
    module = _load_reproducer()
    module.REPO_ROOT = tmp_path
    (tmp_path / "main.tex").write_text("Manuscript source")
    with pytest.raises(AssertionError, match="manuscript or internal source"):
        module.validate_public_boundary()
