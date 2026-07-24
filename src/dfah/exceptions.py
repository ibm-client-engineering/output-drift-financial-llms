"""Typed exception hierarchy for DFAH."""

from __future__ import annotations


class DFAHError(Exception):
    """Base class for every expected DFAH failure."""


class ConfigurationError(DFAHError):
    """The suite, agent, manifest, or replay policy is invalid."""


class AgentContractError(DFAHError):
    """An agent did not satisfy the capture or request-echo contract."""


class ToolExecutionError(DFAHError):
    """A registered tool was missing, rejected its arguments, or raised."""


class EligibilityError(DFAHError):
    """A report or case group cannot satisfy its required-channel contract."""


class ManifestMismatchError(DFAHError):
    """Two artifacts were compared across incompatible manifests."""


class SuiteVersionMismatchError(DFAHError):
    """Two artifacts were compared across different suite versions."""


class ArtifactError(DFAHError):
    """An artifact is missing, malformed, conflicting, or cannot be persisted."""


class EpisodeConflictError(ArtifactError):
    """A durable episode key already exists with different content."""


class IncompleteEpisodeError(ArtifactError):
    """A prior attempt crossed dispatch without a terminal capture."""


class BudgetExceededError(DFAHError):
    """Starting another episode would exceed a declared budget."""


class GateViolationError(DFAHError):
    """A gate policy rejected a report."""


class OptionalDependencyError(DFAHError, ImportError):
    """An explicitly requested optional feature is not installed."""
