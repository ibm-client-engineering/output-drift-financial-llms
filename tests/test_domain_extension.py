# MIT License — DFAH-Bench
"""Tests for the domain-extension mechanism (paper §3.1).

Proves that a new decision ontology drops into the task registry and every
metric resolves K / validates decisions through the standard path with zero
changes to metric code.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics.dcb import compute_dcb
from bench.spec.taxonomy import (
    TASK_REGISTRY,
    DecisionOntology,
    TaskSpec,
    get_k,
    get_ontology,
    register_task,
    validate_decision,
)


def _spec(task_id="zz_test_domain", categories=("alpha", "beta", "gamma")):
    return TaskSpec(
        task_id=task_id,
        name="Test Domain",
        description="synthetic",
        ontology=DecisionOntology(
            name=task_id, categories=list(categories), description="synthetic"
        ),
        tool_count=1,
        expected_tools=["mock_tool"],
    )


@pytest.fixture
def clean_registry():
    """Snapshot and restore the registry so tests cannot leak state."""
    before = dict(TASK_REGISTRY)
    yield
    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(before)


class TestRegisterTask:
    def test_unknown_benchmark_raises_before_registration(self, clean_registry):
        with pytest.raises(KeyError):
            get_k("zz_test_domain")

    def test_registration_resolves_k_through_standard_path(self, clean_registry):
        register_task(_spec())
        assert get_k("zz_test_domain") == 3
        assert get_ontology("zz_test_domain").categories == ["alpha", "beta", "gamma"]
        assert validate_decision("ALPHA", "zz_test_domain") is True
        assert validate_decision("delta", "zz_test_domain") is False

    def test_dcb_runs_unchanged_on_new_domain(self, clean_registry):
        register_task(_spec())
        # Uniform over 3 categories -> H = log(3) -> DCB = 0.
        result = compute_dcb(["alpha", "beta", "gamma"], benchmark="zz_test_domain")
        assert result.k_categories == 3
        assert result.dcb == pytest.approx(0.0, abs=1e-12)
        # Full collapse -> DCB = 1.
        result = compute_dcb(["alpha"] * 9, benchmark="zz_test_domain")
        assert result.dcb == pytest.approx(1.0)

    def test_published_ontologies_cannot_be_silently_replaced(self, clean_registry):
        with pytest.raises(ValueError, match="already registered"):
            register_task(_spec(task_id="compliance"))
        # Explicit overwrite is allowed (and explicit).
        register_task(_spec(task_id="compliance"), overwrite=True)
        assert get_k("compliance") == 3

    def test_degenerate_ontology_rejected(self, clean_registry):
        with pytest.raises(ValueError, match="at least 2"):
            register_task(_spec(categories=("only_one",)))

    def test_financial_ontologies_unchanged(self):
        # The three published ontologies stay exactly as the paper states.
        assert get_k("compliance") == 3
        assert get_k("portfolio") == 3
        assert get_k("dataops") == 3
