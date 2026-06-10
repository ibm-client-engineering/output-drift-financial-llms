"""Canonical schema for DFAH-Bench replay episodes.

Defines the data contract for benchmark episodes with channel-aware
optional fields. The schema supports heterogeneous legacy traces — each
divergence channel is optional, and metrics only run on channels that
are actually present.

Divergence channels:
    TRAJECTORY — tool-call sequences (from tool_sequence in run logs)
    EVIDENCE_CONTACT — evidence subsets consulted (from tool outputs)
    RATIONALE — reasoning text (only when captured; unavailable in existing data)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Divergence channels
# ---------------------------------------------------------------------------

class DivergenceChannel(Enum):
    """Observable divergence channels for behavioral consistency measurement."""
    TRAJECTORY = "trajectory"
    EVIDENCE_CONTACT = "evidence_contact"
    RATIONALE = "rationale"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    """Provider-agnostic run metadata for reproducibility.

    Fields may be None when loading legacy traces that did not capture
    all metadata. Backend choice is experiment metadata, not benchmark logic.
    """
    backend_type: Optional[str] = None       # ollama, llama_cpp, api
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_digest_or_hash: Optional[str] = None
    quantization: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    system_prompt_version: Optional[str] = None
    tool_config_version: Optional[str] = None
    timestamp_utc: Optional[str] = None


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation within a replay episode."""
    name: str
    arguments: Optional[Dict[str, Any]] = None
    output_hash: Optional[str] = None


@dataclass(frozen=True)
class EvidenceContact:
    """A canonical evidence reference touched during execution.

    source_id: stable string identifier (document ID, chunk ID, filing ref,
               tool-return identifier, etc.)
    contact_type: category of evidence (e.g., "sanctions_check", "customer_profile",
                  "market_data", "risk_score")
    """
    source_id: str
    contact_type: str = ""


@dataclass(frozen=True)
class ReasoningTrace:
    """Optional reasoning text from an agent run.

    Only populated when full reasoning capture is enabled. Unavailable
    in the current checked-in raw corpus (0% reasoning-text coverage as of
    the 2026-04-06 channel audit).
    """
    steps: Optional[List[str]] = None
    raw_text: Optional[str] = None


@dataclass(frozen=True)
class Decision:
    """The agent's final decision."""
    label: str
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Replay episode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayEpisode:
    """A single replay run in the DFAH-Bench benchmark.

    Core identity fields (case_id, benchmark, run_id, decision) are always
    required. Observable execution channels are each optional.
    """
    # Core identity — always present
    case_id: str
    benchmark: str
    run_id: int
    decision: Decision
    runtime_seconds: float = 0.0

    # Run metadata — provider-agnostic
    metadata: RunMetadata = field(default_factory=RunMetadata)

    # Observable channels — each optional
    tool_calls: Optional[List[ToolCall]] = None
    evidence_contacts: Optional[List[EvidenceContact]] = None
    reasoning_trace: Optional[ReasoningTrace] = None

    # Raw source reference
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------

def available_channels(episode: ReplayEpisode) -> Set[DivergenceChannel]:
    """Determine which divergence channels are available for an episode."""
    channels: Set[DivergenceChannel] = set()

    if episode.tool_calls is not None and len(episode.tool_calls) > 0:
        channels.add(DivergenceChannel.TRAJECTORY)

    if episode.evidence_contacts is not None and len(episode.evidence_contacts) > 0:
        channels.add(DivergenceChannel.EVIDENCE_CONTACT)

    # Rationale requires meaningful text (>20 chars to filter noise)
    if (episode.reasoning_trace is not None
            and episode.reasoning_trace.raw_text is not None
            and len(episode.reasoning_trace.raw_text) > 20):
        channels.add(DivergenceChannel.RATIONALE)

    return channels


def episodes_common_channels(
    episodes: List[ReplayEpisode],
) -> Set[DivergenceChannel]:
    """Find divergence channels available across ALL episodes in a group."""
    if not episodes:
        return set()
    common = available_channels(episodes[0])
    for ep in episodes[1:]:
        common &= available_channels(ep)
    return common


# ---------------------------------------------------------------------------
# Loaders — map existing run log JSON to canonical schema
# ---------------------------------------------------------------------------

def _parse_run_log(data: Dict[str, Any], source_path: Optional[str] = None) -> ReplayEpisode:
    """Parse a single run log JSON dict into a ReplayEpisode."""
    # Core identity
    decision_label = data.get("decision_output", "")
    decision = Decision(label=decision_label.strip().lower() if decision_label else "")

    # Run metadata — fill from available fields
    metadata = RunMetadata(
        model_name=data.get("model"),
        temperature=data.get("temperature"),
        seed=data.get("seed"),
        timestamp_utc=data.get("timestamp"),
    )

    # Tool calls — from tool_sequence
    tool_calls = None
    tool_seq = data.get("tool_sequence", [])
    tool_hashes = data.get("tool_output_hashes", [])
    if tool_seq:
        tool_calls = []
        for i, name in enumerate(tool_seq):
            output_hash = tool_hashes[i] if i < len(tool_hashes) else None
            tool_calls.append(ToolCall(name=name, output_hash=output_hash))

    # Evidence contacts — derive from tool outputs when available
    evidence_contacts = None
    tool_outputs = data.get("tool_outputs", [])
    if tool_outputs:
        evidence_contacts = []
        for i, output in enumerate(tool_outputs):
            if not output:
                continue
            tool_name = tool_seq[i] if i < len(tool_seq) else f"tool_{i}"
            # Create a stable evidence contact ID from the tool output
            if isinstance(output, dict):
                for key, value in output.items():
                    source_id = f"{tool_name}.{key}={value}"
                    evidence_contacts.append(
                        EvidenceContact(source_id=source_id, contact_type=tool_name)
                    )
            else:
                source_id = f"{tool_name}.result={output}"
                evidence_contacts.append(
                    EvidenceContact(source_id=source_id, contact_type=tool_name)
                )
    elif tool_hashes:
        # Weaker signal: use tool output hashes as evidence contact IDs
        evidence_contacts = []
        for i, h in enumerate(tool_hashes):
            if h:
                tool_name = tool_seq[i] if i < len(tool_seq) else f"tool_{i}"
                evidence_contacts.append(
                    EvidenceContact(source_id=f"{tool_name}.hash={h}", contact_type=tool_name)
                )

    # Reasoning trace — not available in current data
    reasoning_trace = None
    for field_name in ("reasoning_text", "rationale", "final_content", "raw_response"):
        raw_text = data.get(field_name)
        if isinstance(raw_text, str) and len(raw_text) > 20:
            reasoning_trace = ReasoningTrace(raw_text=raw_text)
            break

    return ReplayEpisode(
        case_id=data.get("case_id", ""),
        benchmark=data.get("benchmark", ""),
        run_id=data.get("run_id", 0),
        decision=decision,
        runtime_seconds=data.get("runtime_seconds", 0.0),
        metadata=metadata,
        tool_calls=tool_calls,
        evidence_contacts=evidence_contacts if evidence_contacts else None,
        reasoning_trace=reasoning_trace,
        source_path=source_path,
    )


def load_episode(path: Path) -> ReplayEpisode:
    """Load a single replay episode from a run log JSON file.

    If a corresponding _full.json file exists, merge richer channel data
    from it (evidence contacts, reasoning text).
    """
    with open(path) as f:
        data = json.load(f)

    episode = _parse_run_log(data, source_path=str(path))

    # Try to merge from _full.json for richer data
    full_path = path.with_name(path.name.replace(".json", "_full.json"))
    if full_path.exists() and "_full" not in path.name:
        try:
            with open(full_path) as f:
                full_data = json.load(f)
            full_episode = _parse_run_log(full_data, source_path=str(full_path))
            # Merge richer channels from full log
            if episode.evidence_contacts is None and full_episode.evidence_contacts is not None:
                episode = ReplayEpisode(
                    case_id=episode.case_id,
                    benchmark=episode.benchmark,
                    run_id=episode.run_id,
                    decision=episode.decision,
                    runtime_seconds=episode.runtime_seconds,
                    metadata=episode.metadata,
                    tool_calls=episode.tool_calls,
                    evidence_contacts=full_episode.evidence_contacts,
                    reasoning_trace=full_episode.reasoning_trace or episode.reasoning_trace,
                    source_path=episode.source_path,
                )
        except (json.JSONDecodeError, OSError):
            pass

    return episode


def load_episodes(directory: Path) -> List[ReplayEpisode]:
    """Load all replay episodes from a directory of run log JSON files.

    Scans for case_*_run_*.json files (excluding _full.json variants).
    """
    episodes = []
    dir_path = Path(directory)
    for log_file in sorted(dir_path.glob("case_*_run_*.json")):
        if "_full" in log_file.name:
            continue
        try:
            episodes.append(load_episode(log_file))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return episodes
