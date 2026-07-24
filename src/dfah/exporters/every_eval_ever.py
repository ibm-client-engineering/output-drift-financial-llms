"""Privacy-safe export to the Every Eval Ever v0.2.2 interchange schema.

Every Eval Ever (EEE) is a reporting schema, not a replay estimand.  DFAH's
replay groups, channel eligibility, suite version, and path abstractions are
therefore retained as namespaced string metadata while standard EEE fields
carry the aggregate scores and agentic tool-call records.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from .._canonical import atomic_private_write, canonical_bytes, sha256
from ..exceptions import ArtifactError, ConfigurationError, OptionalDependencyError
from ..metrics import evaluate_episode
from ..models import Episode, Report
from ..store import FileStore

EEE_SCHEMA_VERSION = "0.2.2"
EEE_INSTANCE_SCHEMA_VERSION = "instance_level_eval_0.2.2"
EvaluatorRelationship = Literal["first_party", "third_party", "collaborative", "other"]


@dataclass(frozen=True, slots=True)
class EveryEvalEverExportResult:
    """Paths and commitments produced by one EEE export."""

    aggregate_path: Path
    instances_path: Path | None
    evaluation_id: str
    instance_count: int
    instances_sha256: str | None
    officially_validated: bool


def _json_string(value: object) -> str:
    return canonical_bytes(value, redact=True).decode("utf-8")


def _epoch(value: datetime) -> str:
    rendered = f"{value.timestamp():.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _resolve_verified_run(source: str | Path) -> tuple[Report, Path]:
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return Report.from_json(candidate), candidate
    if candidate.is_file() and candidate.parent.name == "reports":
        return Report.from_json(candidate), candidate.parent.parent
    raise ArtifactError(
        "EEE export requires a DFAH run directory or a report inside RUN/reports"
    )


def _source_data(report: Report) -> dict[str, object]:
    return {
        "dataset_name": report.suite_id,
        "source_type": "other",
        "additional_details": {
            "dfah_suite_version": report.suite_version,
            "dfah_fixture_sha256": report.manifest.fixture_hash,
            "dfah_tool_schema_sha256": report.manifest.tool_schema_hash,
            "dfah_case_population": str(report.suite_cases_total),
            "dfah_cases_selected": str(report.cases_selected),
        },
    }


def _generation_config(report: Report) -> dict[str, object]:
    return {
        "generation_args": {
            "temperature": report.manifest.temperature,
            "top_p": report.manifest.top_p,
            "agentic_eval_config": {
                "additional_details": {
                    "dfah_tool_schema_sha256": report.manifest.tool_schema_hash,
                }
            },
        },
        "additional_details": {
            "dfah_seed": (
                str(report.manifest.seed)
                if report.manifest.seed is not None
                else "provider_unavailable"
            ),
            # Standard decoding controls are represented above.  Commit to all
            # remaining request settings without copying deployment-specific
            # names or values into an interchange record.
            "dfah_request_parameters_sha256": sha256(dict(report.manifest.request_parameters)),
            "dfah_wire_attestation": ("manifest derived from post-normalization payload echo"),
            "dfah_replay_design": _json_string(
                {
                    "replays_requested": report.replays_requested,
                    "schedule_seed": report.schedule_seed,
                    "sample_rate": report.sample_rate,
                    "episodes_planned": report.episodes_planned,
                    "episodes_eligible": report.episodes_eligible,
                    "eligible_groups": report.observed_groups,
                    "required_channels": ["decision", "trajectory"],
                }
            ),
        },
    }


def _metric(
    report: Report,
    *,
    metric_id: str,
    name: str,
    description: str,
    score: float,
    lower_is_better: bool,
    minimum: float,
    maximum: float,
    unit: str,
) -> dict[str, object]:
    return {
        "evaluation_result_id": metric_id,
        "evaluation_name": f"{report.suite_id}: {name}",
        "source_data": _source_data(report),
        "evaluation_timestamp": report.created_at.isoformat(),
        "metric_config": {
            "evaluation_description": description,
            "metric_id": metric_id,
            "metric_name": name,
            "metric_kind": "replay_agreement",
            "metric_unit": unit,
            "metric_parameters": {
                "replays": report.replays_requested,
                "task_weighted": True,
            },
            "lower_is_better": lower_is_better,
            "score_type": "continuous",
            "min_score": minimum,
            "max_score": maximum,
            "additional_details": {
                "dfah_denominator": (
                    "same eligible repeated case groups for decision and trajectory"
                ),
                "dfah_not_accuracy": "true",
            },
        },
        "score_details": {
            "score": score,
            "details": {
                "eligible_groups": str(report.observed_groups),
                "eligible_episodes": str(report.episodes_eligible),
                "planned_episodes": str(report.episodes_planned),
                "suite_version": report.suite_version,
            },
        },
        "generation_config": _generation_config(report),
    }


def _aggregate_metrics(report: Report) -> list[dict[str, object]]:
    if (
        not report.metrics_available
        or report.dar is None
        or report.tar is None
        or report.gap is None
        or report.flags_per_100_cases is None
    ):
        raise ArtifactError(
            "Every Eval Ever aggregate export requires at least one eligible "
            "replay group; unavailable DFAH metrics are never serialized as scores"
        )
    dar = report.dar
    tar = report.tar
    gap = report.gap
    flags_per_100_cases = report.flags_per_100_cases
    common = [
        (
            "dfah.dar",
            "Decision agreement",
            "Modal agreement of closed decisions across eligible replay groups.",
            dar,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.tar_seq",
            "Tool-path sequence agreement",
            "Modal agreement of ordered tool-name sequences.",
            tar.seq,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.tar_bag",
            "Tool-path multiset agreement",
            "Modal agreement after ignoring tool order but retaining multiplicity.",
            tar.bag,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.tar_set",
            "Tool-set agreement",
            "Modal agreement after ignoring tool order and multiplicity.",
            tar.set,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.tar_strong",
            "Argument-aware tool-path agreement",
            "Modal agreement of tool names, argument hashes, and result hashes.",
            tar.strong,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.delta_dt",
            "Decision-path gap",
            "Signed DAR minus sequence-level TAR; diagnostic, not a quality ranking.",
            gap,
            False,
            -1.0,
            1.0,
            "proportion_points",
        ),
        (
            "dfah.eligible_fraction",
            "Eligible episode fraction",
            "Planned episodes retained in comparable groups with required channels.",
            report.eligible_fraction,
            False,
            0.0,
            1.0,
            "proportion",
        ),
        (
            "dfah.flags_per_100",
            "Unanimous-decision path flags per 100 groups",
            "Review load from unanimous decisions with argument-aware path variation.",
            flags_per_100_cases,
            True,
            0.0,
            100.0,
            "cases_per_100",
        ),
    ]
    return [
        _metric(
            report,
            metric_id=metric_id,
            name=name,
            description=description,
            score=score,
            lower_is_better=lower_is_better,
            minimum=minimum,
            maximum=maximum,
            unit=unit,
        )
        for (
            metric_id,
            name,
            description,
            score,
            lower_is_better,
            minimum,
            maximum,
            unit,
        ) in common
    ]


def _model_id(report: Report) -> str:
    prefix = f"{report.manifest.provider}/"
    return (
        report.manifest.model
        if report.manifest.model.startswith(prefix)
        else prefix + report.manifest.model
    )


def _instance_record(
    report: Report,
    episode: Episode,
    *,
    evaluation_id: str,
) -> dict[str, object]:
    eligibility = evaluate_episode(episode)
    raw_input = f"[DFAH input withheld; artifact_case_id={episode.case_id}]"
    references: list[str] = []
    sample_hash = sha256({"raw": raw_input, "reference": references})
    tool_calls = [
        {
            "id": call.call_id or f"{episode.episode_id}-tool-{index}",
            "name": call.name,
            "arguments": {
                "dfah_arguments_sha256": call.argument_hash,
                "dfah_output_sha256": call.output_hash or "unavailable",
                "dfah_execution_state": call.execution_state.value,
                "dfah_result_state": call.result_state.value,
            },
        }
        for index, call in enumerate(episode.trajectory.tool_calls)
    ]
    extracted = (
        episode.decision.label
        if episode.decision is not None
        else f"[no decision: {episode.status.value}]"
    )
    metadata = {
        "dfah_export_profile": "privacy_safe_hash_only",
        "dfah_input_redacted": "true",
        "dfah_evaluation_semantics": "capture_eligibility_not_decision_correctness",
        "dfah_report_id": report.report_id,
        "dfah_manifest_sha256": report.manifest.hash,
        "dfah_replay_group_id": f"{report.manifest.hash}:{episode.case_id}",
        "dfah_suite_version": episode.suite_version,
        "dfah_task": episode.task,
        "dfah_replay_index": str(episode.replay_index),
        "dfah_episode_status": episode.status.value,
        "dfah_episode_eligible": str(eligibility.eligible).lower(),
        "dfah_eligibility_reasons": _json_string(eligibility.reasons),
        "dfah_decision_channel": eligibility.decision.value,
        "dfah_trajectory_channel": eligibility.trajectory.value,
        "dfah_evidence_channel": eligibility.evidence.value,
        "dfah_parse_strategy": episode.parse.strategy,
        "dfah_parse_accepted": str(episode.parse.accepted).lower(),
        "dfah_parse_fallback": str(episode.parse.fallback).lower(),
        "dfah_wire_payload_sha256": (
            episode.wire_request.payload_hash
            if episode.wire_request is not None
            else "unavailable"
        ),
    }
    return {
        "schema_version": EEE_INSTANCE_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "model_id": _model_id(report),
        "evaluation_name": f"{report.suite_id}: DFAH episode eligibility",
        "evaluation_result_id": "dfah.eligible_fraction",
        "sample_id": f"{episode.case_id}/replay-{episode.replay_index}",
        "sample_hash": sample_hash,
        "interaction_type": "agentic",
        "input": {
            "raw": raw_input,
            "formatted": None,
            "reference": references,
            "choices": None,
        },
        "output": None,
        "messages": [
            {
                "turn_idx": 0,
                "role": "assistant",
                "content": None,
                "reasoning_trace": None,
                "tool_calls": tool_calls,
                "tool_call_id": None,
            }
        ],
        "answer_attribution": [
            {
                "turn_idx": 0,
                "source": "dfah.parse_provenance",
                "extracted_value": extracted,
                "extraction_method": episode.parse.strategy,
                "is_terminal": True,
            }
        ],
        "evaluation": {
            "score": 1.0 if eligibility.eligible else 0.0,
            "is_correct": eligibility.eligible,
            "num_turns": 1,
            "tool_calls_count": len(tool_calls),
        },
        "token_usage": {
            "input_tokens": episode.usage.input_tokens,
            "output_tokens": episode.usage.output_tokens,
            "total_tokens": episode.usage.total_tokens,
            "input_tokens_cache_write": None,
            "input_tokens_cache_read": episode.usage.cached_input_tokens,
            "reasoning_tokens": episode.usage.reasoning_tokens,
        },
        "performance": {
            "latency_ms": episode.latency_ms,
            "time_to_first_token_ms": None,
            "generation_time_ms": None,
            "additional_details": {
                "cost_usd": f"{episode.cost_usd:.12g}",
            },
        },
        "error": episode.error.kind if episode.error is not None else None,
        "metadata": metadata,
    }


def _official_validate(aggregate_path: Path, instances_path: Path | None) -> None:
    try:
        from every_eval_ever.eval_types import EvaluationLog
        from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog
    except ImportError as exc:
        raise OptionalDependencyError(
            "official Every Eval Ever validation requires "
            "Python 3.12+ and `python -m pip install 'dfah-bench[eee]'`"
        ) from exc

    EvaluationLog.model_validate_json(aggregate_path.read_bytes())
    if instances_path is not None:
        for line_number, line in enumerate(
            instances_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                InstanceLevelEvaluationLog.model_validate_json(line)
            except Exception as exc:
                raise ArtifactError(
                    f"EEE instance validation failed at line {line_number}"
                ) from exc


def validate_every_eval_ever(
    aggregate_path: str | Path,
    instances_path: str | Path | None = None,
) -> None:
    """Validate an export with the optional upstream EEE Pydantic models."""

    aggregate = Path(aggregate_path)
    instances = Path(instances_path) if instances_path is not None else None
    _official_validate(aggregate, instances)


def export_every_eval_ever(
    run_path: str | Path,
    out: str | Path,
    *,
    source_organization_name: str = "unspecified",
    evaluator_relationship: EvaluatorRelationship = "other",
    include_instances: bool = True,
    validate: bool = False,
) -> EveryEvalEverExportResult:
    """Export one artifact-verified DFAH run to EEE aggregate JSON and JSONL.

    The default profile excludes captured content: it emits artifact case
    identifiers, decision labels, model/provider/adapter identifiers, tool
    names, and argument/result equality hashes.  Those identifiers and labels
    can still reveal deployment or business context and should be reviewed
    before sharing.  The exporter never emits prompts, raw case inputs, raw
    tool arguments, raw tool results, reasoning traces, endpoints, or
    provider-native usage payloads.
    """

    if not source_organization_name.strip():
        raise ConfigurationError("source_organization_name cannot be empty")
    if evaluator_relationship not in {
        "first_party",
        "third_party",
        "collaborative",
        "other",
    }:
        raise ConfigurationError("unsupported Every Eval Ever evaluator relationship")

    report, run_dir = _resolve_verified_run(run_path)
    with FileStore(run_dir, create=False) as store:
        episodes = store.list(manifest_hash=report.manifest.hash)
    aggregate_metrics = _aggregate_metrics(report)

    destination = Path(out).expanduser()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink() or not destination.is_dir():
        raise ArtifactError("EEE export destination must be a real directory")

    timestamp = _epoch(report.created_at)
    evaluation_id = str(uuid.uuid4())
    stem = evaluation_id
    instances_path: Path | None = None
    instances_sha256: str | None = None
    instance_count = 0
    detailed: dict[str, object] | None = None
    if include_instances:
        records = [
            _instance_record(report, episode, evaluation_id=evaluation_id)
            for episode in episodes
        ]
        payload = b"".join(canonical_bytes(record, redact=True) + b"\n" for record in records)
        instances_path = destination / f"{stem}.every-eval-ever.jsonl"
        atomic_private_write(instances_path, payload)
        instances_sha256 = hashlib.sha256(payload).hexdigest()
        instance_count = len(records)
        detailed = {
            "format": "jsonl",
            "file_path": instances_path.name,
            "hash_algorithm": "sha256",
            "checksum": instances_sha256,
            "total_rows": instance_count,
            "additional_details": {
                "dfah_export_profile": "privacy_safe_hash_only",
                "dfah_interaction_semantics": "one record per replay episode",
            },
        }

    aggregate: dict[str, object] = {
        "schema_version": EEE_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_timestamp": report.created_at.isoformat(),
        "retrieved_timestamp": timestamp,
        "source_metadata": {
            "source_name": "DFAH artifact-verified replay run",
            "source_type": "evaluation_run",
            "source_organization_name": source_organization_name,
            "evaluator_relationship": evaluator_relationship,
            "additional_details": {
                "dfah_export_profile": "privacy_safe_hash_only",
                "dfah_report_id": report.report_id,
                "dfah_manifest_sha256": report.manifest.hash,
                "dfah_episode_artifact_root_sha256": (report.episode_artifact_root_sha256),
            },
        },
        "model_info": {
            "name": report.manifest.model,
            "id": _model_id(report),
            "developer": report.manifest.provider,
            "inference_platform": report.manifest.provider,
            "additional_details": {
                "dfah_adapter": report.manifest.adapter,
                "dfah_adapter_version": report.manifest.adapter_version or "unknown",
            },
        },
        "eval_library": {
            "name": "dfah-bench",
            "version": report.manifest.library_version,
            "additional_details": {
                "dfah_suite_id": report.suite_id,
                "dfah_suite_version": report.suite_version,
                "dfah_run_plan_sha256": report.run_plan_sha256,
            },
        },
        "evaluation_results": aggregate_metrics,
    }
    if detailed is not None:
        aggregate["detailed_evaluation_results"] = detailed

    aggregate_path = destination / f"{stem}.every-eval-ever.json"
    atomic_private_write(aggregate_path, canonical_bytes(aggregate, redact=True) + b"\n")
    if validate:
        _official_validate(aggregate_path, instances_path)
    return EveryEvalEverExportResult(
        aggregate_path=aggregate_path,
        instances_path=instances_path,
        evaluation_id=evaluation_id,
        instance_count=instance_count,
        instances_sha256=instances_sha256,
        officially_validated=validate,
    )
