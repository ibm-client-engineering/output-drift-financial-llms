#!/usr/bin/env python3
"""Run the deterministic agent that ships with DFAH."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make this example runnable from a source checkout before installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfah import Replay
from dfah.demo import (
    toy_agent,
    toy_agent_calls,
    toy_suite,
    toy_tool_calls,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dfah-quickstart-") as directory:
        replay = Replay(
            suite=toy_suite,
            replays=3,
            seed=42,
            out=Path(directory).resolve(),
        )
        first = replay.run(toy_agent)
        first_agent_calls = toy_agent_calls["count"]
        first_tool_calls = toy_tool_calls["count"]
        second = replay.run(toy_agent)

    assert first.status.value == "complete"
    assert second.status.value == "complete"
    assert first_agent_calls == first_tool_calls == 6
    assert toy_agent_calls["count"] == first_agent_calls
    assert toy_tool_calls["count"] == first_tool_calls
    assert first.metrics_available
    assert first.dar is not None and first.tar is not None and first.gap is not None
    assert first.flags_per_100_cases is not None
    print(f"DAR={first.dar:.3f} TARseq={first.tar.seq:.3f} gap={first.gap:.3f}")
    print(f"flags/100={first.flags_per_100_cases:.1f}")
    print("Second run reused committed episodes without invoking the agent or tool.")


if __name__ == "__main__":
    main()
