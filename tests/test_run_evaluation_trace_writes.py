"""Fail-closed tests for legacy workshop trace writes."""

from __future__ import annotations

import json

import pytest

import run_evaluation


def test_write_trace_refuses_to_mix_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_evaluation, "TRACES_DIR", tmp_path)
    filename = "trace_mock_model_t0.0_tp1.0_s42_strFalse_c1.jsonl"
    first = [{"run": 1}]
    second = [{"run": 2}]

    run_evaluation.write_trace(filename, first)

    with pytest.raises(FileExistsError, match="Trace already exists"):
        run_evaluation.write_trace(filename, second)

    assert (tmp_path / filename).read_text(encoding="utf-8") == (
        json.dumps(first[0]) + "\n"
    )


def test_write_trace_overwrite_replaces_complete_condition(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(run_evaluation, "TRACES_DIR", tmp_path)
    filename = "trace_mock_model_t0.0_tp1.0_s42_strFalse_c1.jsonl"

    run_evaluation.write_trace(filename, [{"run": 1}])
    run_evaluation.write_trace(filename, [{"run": 2}], overwrite=True)

    assert (tmp_path / filename).read_text(encoding="utf-8") == (
        json.dumps({"run": 2}) + "\n"
    )


def test_trace_filename_is_stable() -> None:
    assert run_evaluation.trace_filename_for_condition(
        "mock", "vendor/model", 0.0, 1.0, 42, False, 4
    ) == "trace_mock_vendor_model_t0.0_tp1.0_s42_strFalse_c4.jsonl"
