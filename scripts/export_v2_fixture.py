#!/usr/bin/env python3
"""Export the public, channel-minimal replay fixture used by arXiv v2.

The historical replay directory contains synthetic benchmark logs with fields
that are not needed to reproduce the corrected analysis.  This exporter keeps
only the grouping keys and the two audited observable channels: the normalized
decision and the ordered tool-name sequence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "v2" / "fixtures" / "retrospective_episodes.jsonl"
)
ALLOWED_TASKS = {"compliance", "dataops", "portfolio"}


def _run_index(source: Path, row: dict[str, Any]) -> int:
    value = row.get("run_id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"_run_(\d+)\.json$", source.name)
    if match is None:
        raise ValueError(f"cannot recover replay index from {source.name}")
    return int(match.group(1))


def _sanitize(source: Path) -> dict[str, object]:
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    task = raw.get("benchmark")
    model = raw.get("model") or source.parent.name
    case_id = raw.get("case_id")
    if task not in ALLOWED_TASKS:
        raise ValueError(f"unexpected benchmark in {source}: {task!r}")
    if not isinstance(model, str) or not model:
        raise ValueError(f"missing model in {source}")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"missing synthetic case ID in {source}")

    raw_decision = raw.get("decision_output")
    decision = raw_decision.strip().lower() if isinstance(raw_decision, str) else None
    raw_tools = raw.get("tool_sequence")
    tool_names: list[str] | None
    if isinstance(raw_tools, list) and all(
        isinstance(name, str) and name for name in raw_tools
    ):
        tool_names = raw_tools
    else:
        tool_names = None

    return {
        "task": task,
        "model": model,
        "case_id": case_id,
        "replay": _run_index(source, raw),
        "decision": decision,
        "tool_names": tool_names,
    }


def export_fixture(source_dir: Path, output: Path) -> int:
    rows = [
        _sanitize(source)
        for source in source_dir.rglob("case_*_run_*.json")
        if "_full" not in source.name
    ]
    rows.sort(
        key=lambda row: (
            str(row["task"]),
            str(row["model"]),
            str(row["case_id"]),
            int(row["replay"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_fixture(args.source_dir, args.output)
    print(f"Wrote {count:,} sanitized synthetic episodes to {args.output}")


if __name__ == "__main__":
    main()
