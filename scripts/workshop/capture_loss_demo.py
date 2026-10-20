"""Small, offline teaching example of agreement surviving capture loss.

These three constructed replays are not observations from either workshop.
Run with Python's standard library: python3 scripts/workshop/capture_loss_demo.py
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy


def agreement(paths: list[list[dict[str, object]]]) -> float:
    """Modal exact-path agreement for a nonempty group of observed records."""
    if not paths:
        raise ValueError("At least one observed record is required")
    keys = [
        json.dumps(path, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for path in paths
    ]
    return max(Counter(keys).values()) / len(keys)


def compare_records(
    recorded: list[list[dict[str, object]] | None],
    dispatched: list[list[dict[str, object]]],
) -> dict[str, int | float | None]:
    """Compare paired observations without converting missing records to [].

    Exact equality checks this example's supplied reference only. It does not
    attest the reference's origin or cover effects outside the recorded calls.
    """
    if not recorded or len(recorded) != len(dispatched):
        raise ValueError("Records and references need the same nonzero denominator")
    available = [path for path in recorded if path is not None]
    return {
        "records": len(recorded),
        "missing_records": len(recorded) - len(available),
        "recorded_path_agreement": (
            agreement(available) if len(available) == len(recorded) else None
        ),
        "reference_mismatches": sum(
            record != dispatched[index]
            for index, record in enumerate(recorded)
            if record is not None
        ),
    }


def demonstration() -> dict[str, object]:
    dispatched = [
        [
            {"name": "read_balance", "arguments": {"account": "demo"}, "result": 2500},
            {"name": "read_policy", "arguments": {"policy": "demo"}, "result": 100},
        ]
        for _ in range(3)
    ]
    views = {
        "unchanged": deepcopy(dispatched),
        "drop_first": [path[1:] for path in dispatched],
        "drop_all": [[] for _ in dispatched],
        "missing_record": [dispatched[0], None, dispatched[2]],
    }
    return {
        "data": "three constructed teaching replays; no study data or model calls",
        "views": {name: compare_records(paths, dispatched) for name, paths in views.items()},
        "observed_empty_reference": compare_records([[], [], []], [[], [], []]),
    }


if __name__ == "__main__":
    print(json.dumps(demonstration(), indent=2))
