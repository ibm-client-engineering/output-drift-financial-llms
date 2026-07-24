"""DFAH: replay stability for tool-using AI agents."""

from __future__ import annotations

from ._meta import package_version

__version__ = package_version()

from .agents import Agent, AgentResult, CallableAgent, RunContext, agent
from .exceptions import (
    AgentContractError,
    ArtifactError,
    BudgetExceededError,
    ConfigurationError,
    DFAHError,
    EligibilityError,
    GateViolationError,
    ManifestMismatchError,
    SuiteVersionMismatchError,
    ToolExecutionError,
)
from .gate import Gate, GatePolicy, GateResult, TaskGatePolicy
from .models import (
    Case,
    CaseExplanation,
    CaseReport,
    ChannelState,
    ConformanceReport,
    ConformanceStatus,
    Decision,
    DispatchState,
    Eligibility,
    EligibilityReasonCount,
    Episode,
    EpisodeStatus,
    Manifest,
    ParseProvenance,
    PathVariationKind,
    ReplayMode,
    Report,
    TARReport,
    ToolCall,
    ToolExecutionState,
    Trajectory,
    Usage,
    WireRequest,
)
from .replay import Replay, build_manifest
from .suite import Suite, ToolSpec
from .tools import ToolRegistry, ToolSession

__all__ = [
    "Agent",
    "AgentContractError",
    "AgentResult",
    "ArtifactError",
    "BudgetExceededError",
    "CallableAgent",
    "Case",
    "CaseExplanation",
    "CaseReport",
    "ChannelState",
    "ConfigurationError",
    "ConformanceReport",
    "ConformanceStatus",
    "DFAHError",
    "Decision",
    "DispatchState",
    "Eligibility",
    "EligibilityError",
    "EligibilityReasonCount",
    "Episode",
    "EpisodeStatus",
    "Gate",
    "GatePolicy",
    "GateResult",
    "GateViolationError",
    "Manifest",
    "ManifestMismatchError",
    "ParseProvenance",
    "PathVariationKind",
    "Replay",
    "ReplayMode",
    "Report",
    "RunContext",
    "Suite",
    "SuiteVersionMismatchError",
    "TARReport",
    "TaskGatePolicy",
    "ToolCall",
    "ToolExecutionError",
    "ToolExecutionState",
    "ToolRegistry",
    "ToolSession",
    "ToolSpec",
    "Trajectory",
    "Usage",
    "WireRequest",
    "__version__",
    "agent",
    "build_manifest",
]
