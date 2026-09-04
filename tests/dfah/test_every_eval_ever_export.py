from __future__ import annotations

import hashlib
import importlib
import json
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dfah import (
    AgentResult,
    ArtifactError,
    ChannelState,
    Replay,
    Report,
    Suite,
    Trajectory,
    WireRequest,
    agent,
    build_manifest,
    export_every_eval_ever,
)
from dfah._canonical import canonical_bytes, sha256
from dfah.cli import app
from dfah.demo import toy_agent, toy_suite
from dfah.exporters.every_eval_ever import _aggregate_metrics, _generation_config


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    report = Replay(suite=toy_suite, replays=2, seed=42, out=run_dir).run(toy_agent)
    assert report.artifacts_verified
    return run_dir


def _eligibility_agent(*, failure: str | None = None):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="eee-eligibility",
        adapter="tests.eee-eligibility",
        implementation_hash="a" * 64,
        request_parameters=parameters,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        payload = {"model": "eee-eligibility", "case_id": case.case_id, **parameters}
        output = "DECISION: ESCALATE"
        if case.case_id == suite.cases[0].case_id:
            if failure == "wire_mismatch":
                payload["nonce"] = context.replay_index
            elif failure == "parse_failure" and context.replay_index == 0:
                output = "No accepted decision marker"
        return AgentResult(
            output_text=output,
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="eee-eligibility",
                payload=payload,
                parameters=parameters,
                adapter="tests.eee-eligibility",
            ),
            cost_usd=1.0,
        )

    return candidate, suite


def _read_export(result):
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert result.instances_path is not None
    records = [
        json.loads(line)
        for line in result.instances_path.read_text(encoding="utf-8").splitlines()
    ]
    metrics = {row["evaluation_result_id"]: row for row in aggregate["evaluation_results"]}
    return aggregate, metrics, records


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
        == "replay_group_retention_not_decision_correctness"
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


@pytest.mark.parametrize("profile", ["toy", "wire_mismatch", "partial"])
def test_export_passes_optional_upstream_validation(tmp_path: Path, profile: str) -> None:
    pytest.importorskip(
        "every_eval_ever",
        reason="official EEE validator is exercised in the Python 3.12 package job",
    )
    if profile == "toy":
        run_dir = _build_run(tmp_path)
    else:
        candidate, suite = _eligibility_agent(
            failure="wire_mismatch" if profile == "wire_mismatch" else None
        )
        run_dir = tmp_path / "run"
        Replay(
            suite=suite,
            replays=2,
            seed=42,
            out=run_dir,
            budget_usd=3 if profile == "partial" else None,
            estimated_max_episode_cost_usd=1,
        ).run(candidate)
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


@pytest.mark.parametrize("failure", ["wire_mismatch", "parse_failure"])
def test_instance_scores_match_replay_group_retention(tmp_path, failure):
    candidate, suite = _eligibility_agent(failure=failure)
    run_dir = tmp_path / "mixed-run"
    report = Replay(suite=suite, replays=2, seed=42, out=run_dir).run(candidate)
    result = export_every_eval_ever(run_dir, tmp_path / "export")
    _aggregate, metrics, records = _read_export(result)

    assert report.observed_groups == 1
    assert metrics["dfah.eligible_fraction"]["score_details"]["score"] == 0.5
    assert sum(row["evaluation"]["score"] for row in records) == report.episodes_eligible == 2
    for row in records:
        retained = row["sample_id"].startswith(suite.cases[1].case_id + "/")
        assert row["evaluation"]["score"] == float(retained)
        assert row["evaluation"]["is_correct"] is retained
        assert row["metadata"]["dfah_replay_group_eligible"] == str(retained).lower()
        assert row["evaluation_result_id"] == "dfah.eligible_fraction"
    if failure == "wire_mismatch":
        assert all(
            row["metadata"]["dfah_episode_capture_eligible"] == "true" for row in records
        )

    for metric_id, row in metrics.items():
        assert row["metric_config"]["metric_parameters"]["task_weighted"] is (
            metric_id not in {"dfah.eligible_fraction", "dfah.flags_per_100"}
        )
    assert (
        "planned episodes"
        in metrics["dfah.eligible_fraction"]["metric_config"]["additional_details"][
            "dfah_denominator"
        ]
    )


def test_partial_export_preserves_the_planned_episode_denominator(tmp_path):
    candidate, suite = _eligibility_agent()
    run_dir = tmp_path / "partial-run"
    report = Replay(
        suite=suite,
        replays=2,
        seed=42,
        out=run_dir,
        budget_usd=3,
        estimated_max_episode_cost_usd=1,
    ).run(candidate)
    result = export_every_eval_ever(run_dir, tmp_path / "export")
    aggregate, metrics, records = _read_export(result)
    assert report.episodes_completed == len(records) == 3
    assert report.episodes_eligible == 2
    retained = sum(row["evaluation"]["score"] for row in records)
    assert (
        retained / report.episodes_planned
        == metrics["dfah.eligible_fraction"]["score_details"]["score"]
        == 0.5
    )
    details = aggregate["detailed_evaluation_results"]["additional_details"]
    assert details["dfah_planned_episodes"] == "4"
    assert details["dfah_uncommitted_episodes"] == "1"
    assert (
        "uncommitted planned episodes are omitted and contribute zero"
        in details["dfah_instance_aggregation"]
    )


@pytest.mark.parametrize("resume_after_snapshot", [False, True])
def test_export_binds_instances_to_one_snapshot_during_resume(
    tmp_path, monkeypatch, resume_after_snapshot
):
    candidate, suite = _eligibility_agent()
    run_dir = tmp_path / "resume-run"
    first = Replay(
        suite=suite,
        replays=2,
        seed=42,
        out=run_dir,
        budget_usd=3,
        estimated_max_episode_cost_usd=1,
    ).run(candidate)
    assert first.episodes_completed == 3
    assert first.observed_groups == 1
    exporter = importlib.import_module("dfah.exporters.every_eval_ever")
    hook = "_verified_episode_snapshot" if resume_after_snapshot else "_resolve_verified_run"
    original = getattr(exporter, hook)

    def read_then_resume(*args):
        snapshot = original(*args)
        resumed = Replay(
            suite=suite,
            replays=2,
            seed=42,
            out=run_dir,
            budget_usd=4,
            estimated_max_episode_cost_usd=1,
        ).run(candidate)
        assert resumed.episodes_completed == 4
        assert resumed.episode_artifact_root_sha256 != first.episode_artifact_root_sha256
        return snapshot

    monkeypatch.setattr(exporter, hook, read_then_resume)
    destination = tmp_path / "export"
    if resume_after_snapshot:
        result = export_every_eval_ever(run_dir, destination)
        aggregate, _metrics, records = _read_export(result)
        assert len(records) == 3
        assert (
            aggregate["source_metadata"]["additional_details"][
                "dfah_episode_artifact_root_sha256"
            ]
            == first.episode_artifact_root_sha256
        )
    else:
        with pytest.raises(ArtifactError, match="run changed after report verification"):
            export_every_eval_ever(run_dir, destination)
        assert not destination.exists()


def test_export_rejects_changed_episode_payload_after_verification(tmp_path, monkeypatch):
    run_dir = _build_run(tmp_path)
    exporter = importlib.import_module("dfah.exporters.every_eval_ever")
    original = exporter._resolve_verified_run

    def verify_then_change_payload(source):
        verified = original(source)
        episode_path = next((run_dir / "episodes").glob("*.json"))
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["usage"]["input_tokens"] += 1
        payload = canonical_bytes(episode) + b"\n"
        episode_path.write_bytes(payload)
        episode_path.with_suffix(".json.sha256").write_text(
            hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
        )
        return verified

    monkeypatch.setattr(exporter, "_resolve_verified_run", verify_then_change_payload)
    destination = tmp_path / "export"
    with pytest.raises(ArtifactError, match="run changed after report verification"):
        export_every_eval_ever(run_dir, destination)
    assert not destination.exists()


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
