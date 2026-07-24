"""Stable, typed records used by the DFAH public API and artifact format."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ._canonical import atomic_private_write, canonical_bytes, contains_secret, sha256
from ._frozen import FrozenJson, FrozenJsonMap
from .exceptions import EligibilityError

SCHEMA_VERSION: Literal["1.0"] = "1.0"
SEMVER_PATTERN = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


class Record(BaseModel):
    """Strict immutable base for persisted records."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False
    )
    schema_version: Literal["1.0"] = SCHEMA_VERSION


class ChannelState(str, Enum):
    """Availability and activity for one observable channel."""

    UNAVAILABLE = "unavailable"
    OBSERVED_EMPTY = "observed_empty"
    OBSERVED_NONEMPTY = "observed_nonempty"
    MALFORMED = "malformed"

    @property
    def observed(self) -> bool:
        """Whether the channel was validly captured, including an empty capture."""

        return self in {self.OBSERVED_EMPTY, self.OBSERVED_NONEMPTY}


class EpisodeStatus(str, Enum):
    """Terminal scientific status for an episode."""

    VALID = "valid"
    PARSE_FAILURE = "parse_failure"
    REFUSAL = "refusal"
    TOOL_ERROR = "tool_error"
    PROVIDER_ERROR = "provider_error"
    BUDGET_STOP = "budget_stop"
    INTERRUPTED = "interrupted"
    CONTRACT_ERROR = "contract_error"


class DispatchState(str, Enum):
    """Durable operational boundary reached by an episode attempt."""

    PLANNED = "planned"
    RESERVED = "reserved"
    DISPATCHING = "dispatching"
    RESPONDED = "responded"
    UNKNOWN_AFTER_DISPATCH = "unknown_after_dispatch"
    CAPTURED = "captured"
    COMPLETE = "complete"


class ReplayMode(str, Enum):
    """How replay results affect the production workflow."""

    SHADOW = "shadow"
    BLOCKING = "blocking"


class PathVariationKind(str, Enum):
    """Most consequential observable difference within a replay group."""

    NONE = "none"
    ARGUMENT_OR_RESULT = "argument_or_result"
    ORDER_ONLY = "order_only"
    MULTIPLICITY = "multiplicity"
    TOOL_SET = "tool_set"


class ToolExecutionState(str, Enum):
    """Whether an observed tool request actually reached its implementation."""

    REQUESTED = "requested"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ERROR = "error"


class ParseProvenance(Record):
    """How a decision label was—or was not—extracted from model output.

    ``fallback=True`` means an unambiguous label could not be extracted and a
    fallback would have been required. DFAH never substitutes one: ``label``
    remains absent and the episode is ineligible.
    """

    strategy: Literal["strict_marker", "typed_output", "none"]
    raw_span: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback: bool = False
    accepted: bool = False

    @model_validator(mode="after")
    def _coherent(self) -> ParseProvenance:
        if self.accepted and (self.strategy == "none" or self.fallback):
            raise ValueError("an accepted parse cannot use strategy=none or fallback=True")
        if not self.accepted and self.strategy != "none":
            raise ValueError("an unaccepted parse must use strategy=none")
        return self


class Decision(Record):
    """One final decision emitted by an agent."""

    label: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ToolCall(Record):
    """One observed tool invocation, including canonicalizable arguments."""

    name: str = Field(min_length=1)
    arguments: FrozenJsonMap = Field(default_factory=dict)
    arguments_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    arguments_redacted: bool = False
    call_id: str | None = None
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    result_state: ChannelState = ChannelState.UNAVAILABLE
    execution_state: ToolExecutionState = ToolExecutionState.EXECUTED
    latency_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _execution_matches_capture(self) -> ToolCall:
        if self.arguments_redacted and self.arguments_hash is None:
            raise ValueError("redacted tool arguments require a retained arguments_hash")
        if self.arguments_redacted and self.arguments:
            raise ValueError("redacted tool arguments must not retain raw argument values")
        if (
            self.arguments_hash is not None
            and not self.arguments_redacted
            and self.arguments_hash != sha256(dict(self.arguments))
        ):
            raise ValueError("arguments_hash differs from the captured arguments")
        if (
            self.execution_state
            in {
                ToolExecutionState.REQUESTED,
                ToolExecutionState.REJECTED,
            }
            and self.output_hash is not None
        ):
            raise ValueError("an unexecuted tool call cannot have an output hash")
        if (
            self.result_state.observed
            and self.execution_state is ToolExecutionState.EXECUTED
            and self.output_hash is None
        ):
            raise ValueError("an observed executed tool result requires an output hash")
        if (
            self.execution_state is ToolExecutionState.EXECUTED
            and not self.result_state.observed
        ):
            raise ValueError("an executed tool call requires an observed result channel")
        return self

    @property
    def argument_hash(self) -> str:
        """Canonical SHA-256 of the argument object."""

        return self.arguments_hash or sha256(dict(self.arguments))


class Trajectory(Record):
    """Ordered observable tool path for one episode."""

    state: ChannelState
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def _state_matches_calls(self) -> Trajectory:
        if self.state is ChannelState.OBSERVED_EMPTY and self.tool_calls:
            raise ValueError("observed_empty trajectory cannot contain tool calls")
        if self.state is ChannelState.OBSERVED_NONEMPTY and not self.tool_calls:
            raise ValueError("observed_nonempty trajectory must contain tool calls")
        if not self.state.observed and self.tool_calls:
            raise ValueError("unavailable or malformed trajectory cannot contain trusted calls")
        return self


class Usage(Record):
    """Normalized usage retained alongside provider-native usage."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    provider_usage: FrozenJsonMap = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens, excluding cache-accounting duplicates."""

        return self.input_tokens + self.output_tokens


class WireRequest(Record):
    """Sanitized description of the request that the adapter actually sent."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None
    parameters: FrozenJsonMap = Field(default_factory=dict)
    parameters_attested: bool = False
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter: str = Field(min_length=1)
    adapter_version: str | None = None

    @model_validator(mode="after")
    def _no_credentials(self) -> WireRequest:
        if contains_secret(self):
            raise ValueError("wire request echo contains a credential-shaped value")
        return self

    @classmethod
    def from_payload(
        cls,
        *,
        provider: str,
        model: str,
        payload: Mapping[str, FrozenJson],
        parameters: Mapping[str, FrozenJson],
        parameter_paths: Mapping[str, tuple[str, ...]] | None = None,
        adapter: str,
        endpoint: str | None = None,
        adapter_version: str | None = None,
    ) -> WireRequest:
        """Build an echo from, and attest parameters inside, the wire payload.

        Parameters are top-level by default. Adapters whose providers nest
        request settings can supply a field path for each declared parameter.
        """

        declared = dict(parameters)
        paths = dict(parameter_paths or {})
        if set(paths) - set(declared):
            raise ValueError("parameter_paths contains an undeclared parameter")
        for name, expected in declared.items():
            path = paths.get(name, (name,))
            if not path or any(not part for part in path):
                raise ValueError("wire parameter paths must be nonempty")
            observed: object = payload
            for part in path:
                if not isinstance(observed, Mapping) or part not in observed:
                    raise ValueError("declared request parameter is absent from wire payload")
                observed = observed[part]
            if observed != expected:
                raise ValueError("declared request parameter differs from wire payload")

        return cls(
            provider=provider,
            model=model,
            endpoint=endpoint,
            parameters=declared,
            parameters_attested=True,
            payload_hash=sha256(dict(payload)),
            adapter=adapter,
            adapter_version=adapter_version,
        )


class Manifest(Record):
    """Pinned, human-readable execution contract for one comparable run."""

    suite_id: str = Field(min_length=1)
    suite_version: str = Field(pattern=SEMVER_PATTERN)
    decision_ontology: tuple[str, ...] = Field(min_length=2)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None
    temperature: float | None = None
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None
    tool_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_parameters: FrozenJsonMap = Field(default_factory=dict)
    adapter: str = Field(min_length=1)
    adapter_version: str | None = None
    library_version: str = Field(min_length=1)
    git_sha: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    implementation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    capture_raw_content: bool = False

    @model_validator(mode="after")
    def _no_credentials(self) -> Manifest:
        if contains_secret(self):
            raise ValueError("manifest contains a credential-shaped value")
        normalized = tuple(label.strip().lower() for label in self.decision_ontology)
        if any(not label for label in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("manifest decision ontology must contain unique labels")
        if normalized != self.decision_ontology or any(
            re.fullmatch(r"[a-z][a-z0-9_]*", label) is None for label in normalized
        ):
            raise ValueError(
                "manifest decision ontology must use canonical parser-compatible labels"
            )
        if self.git_sha is None and self.implementation_hash is None:
            raise ValueError("manifest requires a Git revision or reviewed implementation hash")
        return self

    @property
    def hash(self) -> str:
        """Content hash used in durable episode identity."""

        return sha256(self)

    def validate_wire_request(self, request: WireRequest) -> None:
        """Fail if recorded settings do not match the request actually sent."""

        if not request.parameters_attested:
            raise ValueError("wire request parameters were not attested to the payload")

        if (self.provider, self.model, self.adapter) != (
            request.provider,
            request.model,
            request.adapter,
        ):
            raise ValueError("wire request provider/model/adapter differs from manifest")
        if self.adapter_version != request.adapter_version:
            raise ValueError("wire request adapter_version differs from manifest")
        if self.endpoint != request.endpoint:
            raise ValueError("wire request endpoint differs from manifest")
        if dict(self.request_parameters) != dict(request.parameters):
            raise ValueError("wire request parameters differ from manifest")
        for name in ("temperature", "top_p", "seed"):
            expected = getattr(self, name)
            actual = request.parameters.get(name)
            if expected != actual:
                raise ValueError(
                    f"manifest {name}={expected!r} differs from wire payload {actual!r}"
                )


class Case(Record):
    """One stable input unit in a versioned suite."""

    case_id: str = Field(min_length=1)
    artifact_case_id: str | None = Field(default=None, min_length=1)
    input: FrozenJson
    task: str = Field(default="default", min_length=1)
    metadata: FrozenJsonMap = Field(default_factory=dict)

    @property
    def effective_case_id(self) -> str:
        """Identifier retained in artifacts; defaults to ``case_id`` for compatibility."""

        return self.artifact_case_id or self.case_id


class Eligibility(Record):
    """Required-channel eligibility for an episode or replay group."""

    eligible: bool
    decision: ChannelState
    trajectory: ChannelState
    evidence: ChannelState = ChannelState.UNAVAILABLE
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reasons_match_status(self) -> Eligibility:
        if self.eligible and self.reasons:
            raise ValueError("eligible records cannot have exclusion reasons")
        if not self.eligible and not self.reasons:
            raise ValueError("ineligible records must name at least one reason")
        return self


class EligibilityReasonCount(Record):
    """Number of ineligible replay groups containing one privacy-safe reason."""

    reason: str = Field(min_length=1, max_length=200, pattern=r"^[a-z0-9_]+$")
    groups: int = Field(ge=1)


class EpisodeError(Record):
    """Sanitized terminal error retained instead of dropping an episode."""

    kind: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool = False
    dispatch_state: DispatchState


class Episode(Record):
    """One immutable replay attempt and all channels required to score it."""

    episode_id: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_id: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    replay_index: int = Field(ge=0)
    status: EpisodeStatus
    dispatch_state: DispatchState = DispatchState.COMPLETE
    decision: Decision | None = None
    parse: ParseProvenance
    trajectory: Trajectory
    wire_request: WireRequest | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    started_at: datetime
    ended_at: datetime
    error: EpisodeError | None = None

    @model_validator(mode="after")
    def _terminal_contract(self) -> Episode:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.status is EpisodeStatus.VALID:
            if self.decision is None or not self.parse.accepted:
                raise ValueError("valid episode requires an accepted decision")
            if not self.trajectory.state.observed:
                raise ValueError("valid episode requires an observed trajectory channel")
            if self.wire_request is None:
                raise ValueError("valid episode requires a wire-request echo")
            if self.error is not None:
                raise ValueError("valid episode cannot contain an error")
        elif self.decision is not None and not self.parse.accepted:
            raise ValueError("an unaccepted parse cannot retain a decision")
        if contains_secret(self):
            raise ValueError("episode contains a credential-shaped value")
        return self

    @property
    def key(self) -> tuple[str, str, int]:
        """Idempotent durable key: manifest, case, replay index."""

        return self.manifest_hash, self.case_id, self.replay_index


class TARReport(Record):
    """Trajectory agreement under increasingly coarse path abstractions."""

    seq: float = Field(ge=0.0, le=1.0)
    bag: float = Field(ge=0.0, le=1.0)
    set: float = Field(ge=0.0, le=1.0)
    strong: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coarsening_order(self) -> TARReport:
        if self.set + 1e-12 < self.bag or self.bag + 1e-12 < self.seq:
            raise ValueError("TAR coarsening invariant requires set >= bag >= seq")
        if self.strong > self.seq + 1e-12:
            raise ValueError("TAR strong cannot exceed sequence agreement")
        return self


class CaseReport(Record):
    """Paired decision and path measurements for one replay group."""

    case_id: str
    task: str
    replay_count: int = Field(ge=1)
    decision_denominator: int = Field(ge=1)
    trajectory_denominator: int = Field(ge=1)
    dar: float = Field(ge=0.0, le=1.0)
    tar: TARReport
    gap: float = Field(ge=-1.0, le=1.0)
    eligibility: Eligibility
    unanimous_with_sequence_change: bool = False
    unanimous_with_path_change: bool = False

    @model_validator(mode="after")
    def _shared_denominator(self) -> CaseReport:
        if self.decision_denominator != self.trajectory_denominator:
            raise EligibilityError("DAR and TAR must use the identical retained denominator")
        if self.decision_denominator != self.replay_count:
            raise EligibilityError("case report denominator differs from replay_count")
        if abs(self.gap - (self.dar - self.tar.seq)) > 1e-12:
            raise ValueError("gap must equal DAR - TAR_seq")
        if self.unanimous_with_sequence_change and not self.unanimous_with_path_change:
            raise ValueError("a sequence change is also a strong path change")
        return self

    @property
    def path_variation_kind(self) -> PathVariationKind:
        """Classify a flag for practical review triage, without claiming materiality."""

        if self.tar.strong == 1.0:
            return PathVariationKind.NONE
        if self.tar.set < 1.0:
            return PathVariationKind.TOOL_SET
        if self.tar.bag < 1.0:
            return PathVariationKind.MULTIPLICITY
        if self.tar.seq < 1.0:
            return PathVariationKind.ORDER_ONLY
        return PathVariationKind.ARGUMENT_OR_RESULT


class ReportStatus(str, Enum):
    """Completion state for a replay report."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INELIGIBLE = "ineligible"


class ToolCallDigest(Record):
    """Privacy-safe tool identity used in case-level explanations."""

    name: str
    call_id: str | None = None
    arguments_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_state: ToolExecutionState
    result_state: ChannelState


class EpisodeDigest(Record):
    """One replay row for operational case triage."""

    replay_index: int = Field(ge=0)
    status: EpisodeStatus
    decision: str | None = None
    tool_sequence: tuple[str, ...]
    tool_calls: tuple[ToolCallDigest, ...]


class CaseExplanation(Record):
    """Verified, content-safe replay differences for one case."""

    suite_id: str
    suite_version: str
    case_id: str
    task: str
    episodes: tuple[EpisodeDigest, ...]


class Report(Record):
    """Top-level typed replay result."""

    report_id: str
    suite_id: str
    suite_version: str
    manifest: Manifest
    status: ReportStatus
    replays_requested: int = Field(ge=2)
    schedule_seed: int
    sample_rate: float = Field(gt=0.0, le=1.0)
    suite_cases_total: int = Field(gt=0)
    cases_selected: int = Field(gt=0)
    mode: ReplayMode = ReplayMode.SHADOW
    episodes_planned: int = Field(gt=0)
    episodes_completed: int = Field(ge=0)
    episodes_eligible: int = Field(ge=0)
    case_reports: tuple[CaseReport, ...]
    dar: float | None = Field(default=None, ge=0.0, le=1.0)
    tar: TARReport | None = None
    gap: float | None = Field(default=None, ge=-1.0, le=1.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    budget_exceeded: bool = False
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    flagged_groups: int = Field(default=0, ge=0)
    observed_groups: int = Field(default=0, ge=0)
    ineligible_groups: int = Field(default=0, ge=0)
    ineligibility_reasons: tuple[EligibilityReasonCount, ...] = ()
    partial_reason: str | None = None
    run_dir: str | None = None
    episode_artifact_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_artifact_count: int = Field(ge=0)
    run_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    _artifacts_verified: bool = PrivateAttr(default=False)
    _verification_seal: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _aggregate_contract(self) -> Report:
        if (self.suite_id, self.suite_version) != (
            self.manifest.suite_id,
            self.manifest.suite_version,
        ):
            raise ValueError("report suite differs from its manifest")
        if self.episodes_completed > self.episodes_planned:
            raise ValueError("completed episodes exceed the planned schedule")
        if self.cases_selected > self.suite_cases_total:
            raise ValueError("selected case count exceeds the suite population")
        if self.cases_selected * self.replays_requested != self.episodes_planned:
            raise ValueError("planned episodes differ from selected cases times replays")
        if self.episodes_eligible > self.episodes_completed:
            raise ValueError("eligible episodes exceed committed episodes")
        if self.episode_artifact_count != self.episodes_completed:
            raise ValueError("artifact count differs from committed episode count")
        if self.observed_groups != len(self.case_reports):
            raise ValueError("observed_groups differs from case_reports")
        if self.observed_groups > self.cases_selected:
            raise ValueError("observed groups exceed the selected case count")
        if self.observed_groups + self.ineligible_groups != self.cases_selected:
            raise ValueError("eligible and ineligible groups must cover selected cases")
        reasons = tuple(reason.reason for reason in self.ineligibility_reasons)
        if len(reasons) != len(set(reasons)):
            raise ValueError("ineligibility reason counts must be unique")
        if reasons != tuple(sorted(reasons)):
            raise ValueError("ineligibility reason counts must use canonical order")
        if any(reason.groups > self.ineligible_groups for reason in self.ineligibility_reasons):
            raise ValueError("ineligibility reason count exceeds ineligible groups")
        if self.ineligible_groups and not self.ineligibility_reasons:
            raise ValueError("ineligible groups must retain at least one reason count")
        if not self.ineligible_groups and self.ineligibility_reasons:
            raise ValueError("eligible population cannot retain ineligibility reasons")
        expected_flags = sum(row.unanimous_with_path_change for row in self.case_reports)
        if self.flagged_groups != expected_flags:
            raise ValueError("flagged_groups differs from case_reports")
        identities = {(row.task, row.case_id) for row in self.case_reports}
        if len(identities) != len(self.case_reports):
            raise ValueError("case_reports contain duplicate task/case identities")
        if any(row.replay_count != self.replays_requested for row in self.case_reports):
            raise ValueError("eligible case report has the wrong replay count")
        expected_eligible = sum(row.replay_count for row in self.case_reports)
        if self.episodes_eligible != expected_eligible:
            raise ValueError("eligible episode count differs from case_reports")
        if self.case_reports:
            from .metrics.agreement import task_weighted

            if self.dar is None or self.tar is None or self.gap is None:
                raise ValueError("observed groups require available aggregate metrics")
            expected = task_weighted(self.case_reports)
            values = (
                (self.dar, expected.dar, "DAR"),
                (self.tar.seq, expected.tar.seq, "TAR.seq"),
                (self.tar.bag, expected.tar.bag, "TAR.bag"),
                (self.tar.set, expected.tar.set, "TAR.set"),
                (self.tar.strong, expected.tar.strong, "TAR.strong"),
                (self.gap, expected.gap, "gap"),
            )
            for observed, regenerated, name in values:
                if abs(observed - regenerated) > 1e-12:
                    raise ValueError(f"report {name} differs from case-level regeneration")
        elif self.dar is not None or self.tar is not None or self.gap is not None:
            raise ValueError("empty report must use unavailable aggregate metrics")
        if self.status is ReportStatus.COMPLETE:
            if not (self.episodes_completed == self.episodes_eligible == self.episodes_planned):
                raise ValueError("complete report does not cover the full eligible schedule")
            if self.budget_exceeded or self.partial_reason is not None:
                raise ValueError("complete report cannot carry a partial condition")
        elif self.partial_reason is None:
            raise ValueError("non-complete report must explain why it is partial")
        return self

    @property
    def metrics_available(self) -> bool:
        """Whether at least one replay group supports aggregate metrics."""

        return self.observed_groups > 0

    @property
    def unanimous_with_path_change_rate(self) -> float | None:
        """Fraction of observed groups with unanimous decisions and path variation."""

        if not self.observed_groups:
            return None
        return self.flagged_groups / self.observed_groups

    @property
    def flags_per_100_cases(self) -> float | None:
        """Operational review load per 100 observed case groups."""

        rate = self.unanimous_with_path_change_rate
        return None if rate is None else 100.0 * rate

    @property
    def sequence_flags_per_100_cases(self) -> float | None:
        """Unanimous groups with tool-name sequence changes per 100 groups."""

        if not self.observed_groups:
            return None
        flags = sum(row.unanimous_with_sequence_change for row in self.case_reports)
        return 100.0 * flags / self.observed_groups

    @property
    def review_queue_breakdown(self) -> Mapping[PathVariationKind, int]:
        """Flag counts by structural kind; these categories are not severity labels."""

        return {
            kind: sum(
                row.unanimous_with_path_change and row.path_variation_kind is kind
                for row in self.case_reports
            )
            for kind in PathVariationKind
            if kind is not PathVariationKind.NONE
        }

    @property
    def eligible_fraction(self) -> float:
        """Fraction of planned episodes retained in eligible replay groups."""

        return self.episodes_eligible / self.episodes_planned if self.episodes_planned else 0.0

    @property
    def case_sampling_fraction(self) -> float:
        """Selected suite cases divided by the full versioned suite population."""

        return self.cases_selected / self.suite_cases_total

    def compare(
        self,
        other: Report,
        *,
        allow_cross_manifest: bool = False,
        allow_cross_version: bool = False,
        allow_partial: bool = False,
        allow_unverified: bool = False,
    ) -> Mapping[str, float]:
        """Compare two reports only across explicitly compatible contracts."""

        from .exceptions import ManifestMismatchError, SuiteVersionMismatchError

        if not allow_unverified and not (self.artifacts_verified and other.artifacts_verified):
            raise ManifestMismatchError("report comparison requires artifact-verified reports")
        if not allow_partial and (
            self.status is not ReportStatus.COMPLETE
            or other.status is not ReportStatus.COMPLETE
        ):
            raise ManifestMismatchError("report comparison requires complete reports")
        if not self.metrics_available or not other.metrics_available:
            raise EligibilityError("report comparison requires available aggregate metrics")

        if self.suite_id != other.suite_id:
            raise ManifestMismatchError("report suite IDs differ")
        versions_differ = self.suite_version != other.suite_version
        if versions_differ and not allow_cross_version:
            raise SuiteVersionMismatchError("suite versions differ")
        if not versions_differ and (
            self.manifest.decision_ontology,
            self.manifest.fixture_hash,
            self.manifest.tool_schema_hash,
        ) != (
            other.manifest.decision_ontology,
            other.manifest.fixture_hash,
            other.manifest.tool_schema_hash,
        ):
            raise ManifestMismatchError(
                "same-version reports have different suite fixtures, tools, or ontology"
            )
        if (
            self.replays_requested,
            self.schedule_seed,
            self.sample_rate,
            self.suite_cases_total,
            self.cases_selected,
            self.episodes_planned,
            self.episodes_eligible,
            self.observed_groups,
            tuple(sorted((row.task, row.case_id) for row in self.case_reports)),
        ) != (
            other.replays_requested,
            other.schedule_seed,
            other.sample_rate,
            other.suite_cases_total,
            other.cases_selected,
            other.episodes_planned,
            other.episodes_eligible,
            other.observed_groups,
            tuple(sorted((row.task, row.case_id) for row in other.case_reports)),
        ):
            raise ManifestMismatchError("report replay populations or designs differ")
        execution_contract = (
            "provider",
            "model",
            "endpoint",
            "temperature",
            "top_p",
            "seed",
            "request_parameters",
            "adapter",
            "adapter_version",
            "library_version",
            "git_sha",
            "implementation_hash",
            "capture_raw_content",
        )
        if (
            any(
                getattr(self.manifest, field) != getattr(other.manifest, field)
                for field in execution_contract
            )
            and not allow_cross_manifest
        ):
            raise ManifestMismatchError("report execution manifests differ")
        assert self.dar is not None and self.tar is not None and self.gap is not None
        assert other.dar is not None and other.tar is not None and other.gap is not None
        self_flags = self.flags_per_100_cases
        other_flags = other.flags_per_100_cases
        assert self_flags is not None and other_flags is not None
        return {
            "dar": self.dar - other.dar,
            "tar_seq": self.tar.seq - other.tar.seq,
            "gap": self.gap - other.gap,
            "flags_per_100_cases": self_flags - other_flags,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Return strict JSON and optionally write an artifact-safe copy."""

        text = canonical_bytes(self, redact=True).decode("utf-8") + "\n"
        if path is not None:
            atomic_private_write(path, text.encode("utf-8"))
        return text

    def to_html(self, path: str | Path) -> None:
        """Render a standalone HTML report."""

        from .report import render_html

        atomic_private_write(path, render_html(self).encode("utf-8"))

    def explain_case(self, run_dir: str | Path, case_id: str) -> CaseExplanation:
        """Return verified decision/path/hash deltas without raw prompts or values."""

        from .exceptions import ArtifactError
        from .store import FileStore

        self.verify_artifacts(run_dir)
        rows = tuple(
            episode
            for episode in FileStore(run_dir, create=False).iter_episodes(
                manifest_hash=self.manifest.hash
            )
            if episode.case_id == case_id
        )
        if not rows:
            raise ArtifactError(f"case {case_id!r} is not present in the verified run")
        return CaseExplanation(
            suite_id=self.suite_id,
            suite_version=self.suite_version,
            case_id=case_id,
            task=rows[0].task,
            episodes=tuple(
                EpisodeDigest(
                    replay_index=episode.replay_index,
                    status=episode.status,
                    decision=(episode.decision.label if episode.decision is not None else None),
                    tool_sequence=tuple(call.name for call in episode.trajectory.tool_calls),
                    tool_calls=tuple(
                        ToolCallDigest(
                            name=call.name,
                            call_id=call.call_id,
                            arguments_hash=call.argument_hash,
                            output_hash=call.output_hash,
                            execution_state=call.execution_state,
                            result_state=call.result_state,
                        )
                        for call in episode.trajectory.tool_calls
                    ),
                )
                for episode in sorted(rows, key=lambda row: row.replay_index)
            ),
        )

    @property
    def artifacts_verified(self) -> bool:
        """Whether this in-memory report was checked against committed episodes."""

        return (
            self._artifacts_verified
            and self._verification_seal is not None
            and self._verification_seal == sha256(self)
        )

    def verify_artifacts(self, run_dir: str | Path) -> Report:
        """Bind aggregate claims to the verified append-only episode store."""

        from .exceptions import ArtifactError
        from .metrics.agreement import (
            case_reports_from_episodes,
            ineligibility_summary_from_episodes,
        )
        from .store import FileStore

        store = FileStore(run_dir, create=False)
        plan = store.read_plan()
        if plan.hash != self.run_plan_sha256:
            raise ArtifactError("report run-plan commitment does not match the artifact store")
        if (
            plan.manifest_hash,
            plan.suite_id,
            plan.suite_version,
            plan.fixture_hash,
            plan.tool_schema_hash,
            plan.replays,
            plan.seed,
            plan.sample_rate,
            plan.suite_cases_total,
            plan.episodes_planned,
        ) != (
            self.manifest.hash,
            self.suite_id,
            self.suite_version,
            self.manifest.fixture_hash,
            self.manifest.tool_schema_hash,
            self.replays_requested,
            self.schedule_seed,
            self.sample_rate,
            self.suite_cases_total,
            self.episodes_planned,
        ):
            raise ArtifactError("report metadata differs from the immutable run plan")
        if len(plan.case_tasks) != self.cases_selected:
            raise ArtifactError("report selected-case count differs from the run plan")
        expected = {
            (case_id, replay_index): str(task)
            for case_id, task in plan.case_tasks.items()
            for replay_index in range(plan.replays)
        }
        store.assert_schedule_inventory(
            manifest_hash=self.manifest.hash,
            suite_id=self.suite_id,
            suite_version=self.suite_version,
            expected=expected,
        )
        episodes = store.list(manifest_hash=self.manifest.hash)
        identities: set[tuple[str, int]] = set()
        for episode in episodes:
            identity = (episode.case_id, episode.replay_index)
            expected_task = expected.get(identity)
            if identity not in expected or expected_task != episode.task:
                raise ArtifactError("committed episode falls outside the immutable run plan")
            if identity in identities:
                raise ArtifactError("artifact store contains duplicate episode identities")
            identities.add(identity)
            try:
                start = store.read_start(*episode.key)
                dispatch = store.read_dispatch(*episode.key)
            except ArtifactError as exc:
                raise ArtifactError(
                    "committed episode lacks a readable start or dispatch boundary"
                ) from exc
            if (
                start.manifest_hash,
                start.suite_id,
                start.suite_version,
                start.case_id,
                start.task,
                start.replay_index,
            ) != (
                episode.manifest_hash,
                episode.suite_id,
                episode.suite_version,
                episode.case_id,
                episode.task,
                episode.replay_index,
            ) or dispatch.key != episode.key:
                raise ArtifactError(
                    "episode metadata differs from its durable start/dispatch boundary"
                )
            if (
                episode.manifest_hash,
                episode.suite_id,
                episode.suite_version,
            ) != (self.manifest.hash, self.suite_id, self.suite_version):
                raise ArtifactError("committed episode metadata differs from the report")
            expected_episode_id = sha256(
                {
                    "manifest_hash": self.manifest.hash,
                    "case_id": episode.case_id,
                    "replay_index": episode.replay_index,
                }
            )[:32]
            if episode.episode_id != expected_episode_id:
                raise ArtifactError("committed episode ID differs from its durable identity")
            if episode.wire_request is not None:
                try:
                    self.manifest.validate_wire_request(episode.wire_request)
                except ValueError as exc:
                    raise ArtifactError(
                        "committed episode wire request differs from the manifest"
                    ) from exc
            if episode.status is EpisodeStatus.VALID and (
                episode.decision is None
                or episode.decision.label.strip().lower() not in self.manifest.decision_ontology
            ):
                raise ArtifactError("valid episode decision is outside the manifest ontology")
        root, count = store.commitment(manifest_hash=self.manifest.hash)
        if (root, count) != (
            self.episode_artifact_root_sha256,
            self.episode_artifact_count,
        ):
            raise ArtifactError("report episode commitment does not match the artifact store")
        regenerated = case_reports_from_episodes(
            episodes, required_replays=self.replays_requested
        )
        if regenerated != self.case_reports:
            raise ArtifactError("report case metrics differ from committed episodes")
        ineligible_groups, ineligibility_reasons = ineligibility_summary_from_episodes(
            episodes,
            required_replays=self.replays_requested,
            expected_groups=tuple(
                (str(task), case_id) for case_id, task in plan.case_tasks.items()
            ),
        )
        if (ineligible_groups, ineligibility_reasons) != (
            self.ineligible_groups,
            self.ineligibility_reasons,
        ):
            raise ArtifactError("report eligibility diagnostics differ from committed episodes")
        observed_cost = sum(episode.cost_usd for episode in episodes)
        observed_latency = sum(episode.latency_ms for episode in episodes)
        if abs(observed_cost - self.total_cost_usd) > 1e-12:
            raise ArtifactError("report cost differs from committed episodes")
        if abs(observed_latency - self.total_latency_ms) > 1e-9:
            raise ArtifactError("report latency differs from committed episodes")
        self._artifacts_verified = True
        self._verification_seal = sha256(self)
        return self

    @classmethod
    def from_json(cls, path: str | Path, *, allow_unverified: bool = False) -> Report:
        """Load a report and verify it against its committed episode store."""

        source = Path(path)
        run_dir: Path | None = None
        if source.is_dir():
            run_dir = source
            candidates = list((source / "reports").glob("*.json"))
            if not candidates:
                candidates = list(source.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"no DFAH report JSON found under {source}")
            source = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
        elif source.parent.name == "reports":
            run_dir = source.parent.parent
        report = cls.model_validate_json(source.read_bytes())
        if run_dir is not None:
            return report.verify_artifacts(run_dir)
        if not allow_unverified:
            from .exceptions import ArtifactError

            raise ArtifactError(
                "standalone report has no episode store; pass allow_unverified=True only "
                "for non-gating inspection"
            )
        return report


class ConformanceStatus(str, Enum):
    """Outcome of one conformance assertion."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class ConformanceCheck(Record):
    """One agent-integration conformance assertion."""

    name: str
    status: ConformanceStatus
    detail: str

    @property
    def passed(self) -> bool:
        """Whether this check permits the overall preflight to continue."""

        return self.status is not ConformanceStatus.FAIL

    @property
    def skipped(self) -> bool:
        """Whether the check was not applicable to the observed integration."""

        return self.status is ConformanceStatus.SKIP


class ConformanceReport(Record):
    """Result returned by :func:`dfah.testing.check_agent`."""

    checks: tuple[ConformanceCheck, ...]
    cases_selected: int = Field(default=0, ge=0)
    episodes_planned: int = Field(default=0, ge=0)
    estimated_cost_ceiling_usd: float | None = Field(default=None, ge=0.0)

    @property
    def passed(self) -> bool:
        """Whether every conformance assertion passed."""

        return all(check.status is not ConformanceStatus.FAIL for check in self.checks)

    def raise_for_failures(self) -> None:
        """Raise one actionable error when any conformance assertion fails."""

        from .exceptions import AgentContractError

        failed = [
            f"{check.name}: {check.detail}"
            for check in self.checks
            if check.status is ConformanceStatus.FAIL
        ]
        if failed:
            raise AgentContractError("agent conformance failed: " + "; ".join(failed))
