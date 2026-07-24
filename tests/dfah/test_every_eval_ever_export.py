from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dfah import ArtifactError, Replay, Report, export_every_eval_ever
from dfah._canonical import sha256
from dfah.cli import app
from dfah.demo import toy_agent, toy_suite
from dfah.exporters.every_eval_ever import _aggregate_metrics, _generation_config


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    report = Replay(suite=toy_suite, replays=2, seed=42, out=run_dir).run(toy_agent)
    assert report.artifacts_verified
    return run_dir


def test_export_is_verified_hash_only_and_eee_shaped(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    result = export_every_eval_ever(run_dir, tmp_path / "export")

    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["schema_version"] == "0.2.2"
    assert aggregate["evaluation_id"] == result.evaluation_id
    assert str(uuid.UUID(result.evaluation_id)) == result.evaluation_id
    assert aggregate["source_metadata"]["source_organization_name"] == "unspecified"
    assert aggregate["eval_library"]["name"] == "dfah-bench"
    assert {row["metric_config"]["metric_id"] for row in aggregate["evaluation_results"]} == {
        "dfah.dar",
        "dfah.tar_seq",
        "dfah.tar_bag",
        "dfah.tar_set",
        "dfah.tar_strong",
        "dfah.delta_dt",
        "dfah.eligible_fraction",
        "dfah.flags_per_100",
    }
    for row in aggregate["evaluation_results"]:
        details = row["generation_config"]["additional_details"]
        assert "dfah_request_parameters" not in details
        assert len(details["dfah_request_parameters_sha256"]) == 64

    assert result.instances_path is not None
    payload = result.instances_path.read_bytes()
    assert result.instances_sha256 == hashlib.sha256(payload).hexdigest()
    records = [
        json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()
    ]
    assert result.instance_count == len(records) == 4
    assert all(row["schema_version"] == "instance_level_eval_0.2.2" for row in records)
    assert all(row["interaction_type"] == "agentic" for row in records)
    assert all(row["evaluation"]["is_correct"] is True for row in records)
    assert all(
        row["metadata"]["dfah_evaluation_semantics"]
        == "capture_eligibility_not_decision_correctness"
        for row in records
    )
    assert {row["input"]["raw"] for row in records} == {
        "[DFAH input withheld; artifact_case_id=CASE-001]",
        "[DFAH input withheld; artifact_case_id=CASE-002]",
    }
    for row in records:
        assert row["output"] is None
        assert row["messages"][0]["reasoning_trace"] is None
        for call in row["messages"][0]["tool_calls"]:
            assert set(call["arguments"]) == {
                "dfah_arguments_sha256",
                "dfah_output_sha256",
                "dfah_execution_state",
                "dfah_result_state",
            }
            assert len(call["arguments"]["dfah_arguments_sha256"]) == 64
            assert len(call["arguments"]["dfah_output_sha256"]) == 64


def test_export_can_be_aggregate_only(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    result = export_every_eval_ever(
        run_dir,
        tmp_path / "aggregate",
        include_instances=False,
    )
    assert result.instances_path is None
    assert result.instances_sha256 is None
    assert result.instance_count == 0
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert "detailed_evaluation_results" not in aggregate


def test_export_passes_optional_upstream_validation(tmp_path: Path) -> None:
    pytest.importorskip(
        "every_eval_ever",
        reason="official EEE validator is exercised in the Python 3.12 package job",
    )
    run_dir = _build_run(tmp_path)
    result = export_every_eval_ever(
        run_dir,
        tmp_path / "validated",
        validate=True,
    )
    assert result.officially_validated


def test_export_rejects_a_detached_unverified_report(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    report_path = next((run_dir / "reports").glob("*.json"))
    detached = tmp_path / "detached-report.json"
    detached.write_bytes(report_path.read_bytes())

    with pytest.raises(ArtifactError, match="requires a DFAH run directory"):
        export_every_eval_ever(detached, tmp_path / "export")


def test_aggregate_export_rejects_unavailable_metrics(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    report = Report.from_json(run_dir)
    unavailable = report.model_copy(
        update={
            "observed_groups": 0,
            "case_reports": (),
            "dar": None,
            "tar": None,
            "gap": None,
        }
    )

    with pytest.raises(ArtifactError, match="at least one eligible replay group"):
        _aggregate_metrics(unavailable)


def test_generation_config_hashes_arbitrary_request_parameters(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    report = Report.from_json(run_dir)
    parameters = {
        "temperature": 0.0,
        "deployment": "private-deployment-name",
    }
    manifest = report.manifest.model_copy(update={"request_parameters": parameters})
    altered = report.model_copy(update={"manifest": manifest})

    rendered = _generation_config(altered)
    serialized = json.dumps(rendered, sort_keys=True)
    details = rendered["additional_details"]
    assert "private-deployment-name" not in serialized
    assert "dfah_request_parameters" not in details
    assert details["dfah_request_parameters_sha256"] == sha256(parameters)


def test_cli_exports_a_verified_run_without_uploading(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    destination = tmp_path / "cli-export"
    result = CliRunner().invoke(
        app,
        [
            "export",
            str(run_dir),
            "--format",
            "every-eval-ever",
            "--out",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    exported_id = next(
        line.split("=", 1)[1].split()[0]
        for line in result.output.splitlines()
        if line.startswith("evaluation_id=")
    )
    assert str(uuid.UUID(exported_id)) == exported_id
    assert "validated=not-requested" in result.output
    assert len(list(destination.glob("*.json"))) == 1
    assert len(list(destination.glob("*.jsonl"))) == 1


def test_cli_rejects_an_unknown_export_format(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    result = CliRunner().invoke(
        app,
        ["export", str(run_dir), "--format", "not-a-format"],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "unsupported export format" in str(result.exception)
