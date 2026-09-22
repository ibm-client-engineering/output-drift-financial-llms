"""Lower-trust artifact inputs must fail without special-file reads or expansion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dfah import ArtifactError, Replay, Report
from dfah._canonical import read_regular_bytes
from dfah.demo import make_toy_agent
from dfah.store import files as store_module


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file boundary")
@pytest.mark.parametrize(
    ("loader", "suffix"),
    [
        ("Suite.load(path)", ".json"),
        ("Suite.load(path)", ".yaml"),
        ("GatePolicy.load(path)", ".json"),
        ("GatePolicy.load(path)", ".yaml"),
        ("GateRecord.from_json(path)", ".json"),
        ("Report.from_json(path, allow_unverified=True)", ".json"),
        ("Report.from_json(path.parent.parent)", ".json"),
    ],
)
def test_artifact_loaders_reject_fifo_without_blocking(tmp_path, loader, suffix):
    reports = tmp_path / "run" / "reports"
    reports.mkdir(parents=True)
    path = reports / f"input{suffix}"
    os.mkfifo(path)
    source = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "from dfah import Suite, GatePolicy, Report",
            "from dfah.gate import GateRecord",
            "path = Path(sys.argv[1])",
            "try:",
            f"    {loader}",
            "except OSError:",
            "    print('rejected_nonregular_input')",
            "else:",
            "    raise AssertionError('special file was accepted')",
        ]
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", source, str(path)],
        env=env,
        timeout=5,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rejected_nonregular_input" in result.stdout


def test_regular_reader_rejects_final_symlink_and_preserves_selected_file(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b'{"synthetic":true}')
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        read_regular_bytes(link)
    assert read_regular_bytes(target) == b'{"synthetic":true}'


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file boundary")
def test_regular_reader_rejects_fifo_swapped_after_metadata_check(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    original_open = os.open

    def swap_before_open(source, flags, *args, **kwargs):
        assert Path(source) == path
        assert flags & os.O_NONBLOCK
        path.unlink()
        os.mkfifo(path)
        return original_open(source, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(OSError, match="input changed"):
        read_regular_bytes(path)


def test_report_and_resume_check_plan_commitment_before_schedule_expansion(
    tmp_path, monkeypatch
):
    candidate, suite, _tools, calls, _tool_calls = make_toy_agent()
    run = tmp_path / "run"
    replay = Replay(suite=suite, replays=2, out=run)
    replay.run(candidate)
    before_calls = dict(calls)
    path = run / "run-plan.json"
    payload = json.loads(path.read_bytes())
    payload["replays"] = 1_000_000_000
    payload["episodes_planned"] = len(payload["case_tasks"]) * payload["replays"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    def bounded_range(stop):
        # The regression must fail before allocating a maliciously large schedule.
        assert stop < 1000, "unverified replay count reached schedule expansion"
        return range(stop)

    monkeypatch.setattr(store_module, "range", bounded_range, raising=False)
    with pytest.raises(ArtifactError, match="run-plan commitment"):
        Report.from_json(run)
    with pytest.raises(ArtifactError, match="different replay design"):
        replay.run(candidate)
    assert calls == before_calls
