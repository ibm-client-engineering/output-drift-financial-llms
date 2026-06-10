"""Task taxonomy and closed decision ontologies for DFAH-Bench.

Decision categories per benchmark task derive from the existing benchmark
task enums in econometrics/benchmarks/*/task.py where possible. When those
modules are not importable (e.g., missing optional dependencies), we fall
back to hardcoded categories that mirror the enum values.

The ontology defines K for DCB — category count comes from benchmark task
truth, not arbitrary user-provided constants.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type


@dataclass(frozen=True)
class DecisionOntology:
    """A closed set of decision categories for a benchmark task."""
    name: str
    categories: List[str]
    description: str = ""

    @property
    def k(self) -> int:
        """Number of decision categories (K for DCB)."""
        return len(self.categories)

    def validate(self, decision: str) -> bool:
        """Check if a decision label belongs to this ontology."""
        return decision.lower().strip() in [c.lower() for c in self.categories]


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a benchmark task."""
    task_id: str
    name: str
    description: str
    ontology: DecisionOntology
    tool_count: int
    expected_tools: List[str]


def _load_enum_values(module_path: str, enum_name: str) -> Optional[List[str]]:
    """Try to import an enum from econometrics and extract its values.

    Uses importlib to avoid triggering heavy econometrics/__init__.py imports.
    Returns None if the import fails for any reason.
    """
    try:
        import importlib
        mod = importlib.import_module(module_path)
        enum_cls = getattr(mod, enum_name)
        return [member.value for member in enum_cls]
    except Exception:
        return None


def _build_compliance_ontology() -> DecisionOntology:
    values = _load_enum_values(
        "econometrics.benchmarks.compliance_triage.task", "TriageDecision"
    )
    categories = values or ["escalate", "dismiss", "investigate"]
    return DecisionOntology(
        name="compliance_triage",
        categories=categories,
        description="Triage a compliance alert: escalate, dismiss, or investigate.",
    )


def _build_portfolio_ontology() -> DecisionOntology:
    values = _load_enum_values(
        "econometrics.benchmarks.portfolio_constraint.task", "TradeDecision"
    )
    categories = values or ["approve", "reject", "modify"]
    return DecisionOntology(
        name="portfolio_constraint",
        categories=categories,
        description="Validate a trade against portfolio constraints: approve, reject, or modify.",
    )


def _build_dataops_ontology() -> DecisionOntology:
    values = _load_enum_values(
        "econometrics.benchmarks.dataops_exception.task", "ExceptionDecision"
    )
    categories = values or ["auto_fix", "escalate", "quarantine"]
    return DecisionOntology(
        name="dataops_exception",
        categories=categories,
        description="Handle a data quality exception: auto-fix, escalate, or quarantine.",
    )


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

COMPLIANCE_ONTOLOGY = _build_compliance_ontology()
PORTFOLIO_ONTOLOGY = _build_portfolio_ontology()
DATAOPS_ONTOLOGY = _build_dataops_ontology()

COMPLIANCE_SPEC = TaskSpec(
    task_id="compliance",
    name="Compliance Triage",
    description="Evaluate compliance alerts using sanctions, customer profiles, and risk scores.",
    ontology=COMPLIANCE_ONTOLOGY,
    tool_count=3,
    expected_tools=["check_sanctions", "get_customer_profile", "calculate_risk_score"],
)

PORTFOLIO_SPEC = TaskSpec(
    task_id="portfolio",
    name="Portfolio Constraint Checking",
    description="Validate proposed trades against position limits, sector caps, and regulatory constraints.",
    ontology=PORTFOLIO_ONTOLOGY,
    tool_count=5,
    expected_tools=[
        "get_current_holdings", "get_market_data", "check_position_limit",
        "calculate_sector_exposure", "get_regulatory_constraints",
    ],
)

DATAOPS_SPEC = TaskSpec(
    task_id="dataops",
    name="DataOps Exception Handling",
    description="Resolve data quality exceptions in a financial data pipeline.",
    ontology=DATAOPS_ONTOLOGY,
    tool_count=6,
    expected_tools=[
        "get_exception_details", "query_reference_data", "get_historical_fixes",
        "validate_fix", "apply_fix", "escalate_to_human",
    ],
)

TASK_REGISTRY: Dict[str, TaskSpec] = {
    "compliance": COMPLIANCE_SPEC,
    "portfolio": PORTFOLIO_SPEC,
    "dataops": DATAOPS_SPEC,
}


def register_task(spec: TaskSpec, overwrite: bool = False) -> None:
    """Register a new benchmark task with its decision ontology.

    This is the domain-extension entry point (paper §3.1): a new domain
    drops in by providing a TaskSpec — a closed decision ontology plus the
    expected tool surface — and every DFAH metric (DAR, TAR, ECD, DCB)
    operates on it without modification, because the metrics are defined
    over decisions, tool sequences, and evidence sets, not over
    finance-specific structures.

    Args:
        spec: The TaskSpec to register. spec.task_id becomes the benchmark
            key used by get_k / get_ontology / validate_decision.
        overwrite: Allow replacing an existing registration. Defaults to
            False so the three published financial ontologies cannot be
            silently redefined.

    Raises:
        ValueError: If the task_id is already registered and overwrite is
            False, or if the ontology has fewer than 2 categories.
    """
    if spec.ontology.k < 2:
        raise ValueError(
            f"Ontology for '{spec.task_id}' must define at least 2 decision "
            f"categories (got {spec.ontology.k}); K < 2 makes DCB degenerate."
        )
    if spec.task_id in TASK_REGISTRY and not overwrite:
        raise ValueError(
            f"Task '{spec.task_id}' is already registered; "
            f"pass overwrite=True to replace it."
        )
    TASK_REGISTRY[spec.task_id] = spec


def get_k(benchmark: str) -> int:
    """Return the number of decision categories for a benchmark task.

    Args:
        benchmark: Task identifier (e.g., "compliance", "portfolio", "dataops").

    Returns:
        K — the category count for DCB computation.

    Raises:
        KeyError: If benchmark is not in the registry.
    """
    return TASK_REGISTRY[benchmark].ontology.k


def get_ontology(benchmark: str) -> DecisionOntology:
    """Return the decision ontology for a benchmark task."""
    return TASK_REGISTRY[benchmark].ontology


def validate_decision(decision: str, benchmark: str) -> bool:
    """Check if a decision label belongs to the benchmark's ontology."""
    return TASK_REGISTRY[benchmark].ontology.validate(decision)
