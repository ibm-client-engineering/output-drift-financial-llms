"""Tests for Decision Concentration Bias (DCB) metric."""

import math
import pytest
from bench.metrics.dcb import compute_dcb, DCBResult


class TestDCBBasic:
    """Core DCB computation tests with known expected values."""

    def test_degenerate_distribution_dcb_is_one(self):
        """All decisions the same → DCB = 1.0."""
        result = compute_dcb(["escalate"] * 50, k=3)
        assert result.dcb == 1.0
        assert result.entropy == 0.0
        assert result.n_decisions == 50
        assert result.k_categories == 3

    def test_uniform_distribution_dcb_near_zero(self):
        """Equal split across K categories → DCB ≈ 0."""
        decisions = ["a"] * 100 + ["b"] * 100 + ["c"] * 100
        result = compute_dcb(decisions, k=3)
        assert abs(result.dcb) < 0.01
        assert abs(result.entropy - math.log(3)) < 0.01

    def test_binary_80_20_split(self):
        """Known entropy for 80/20 binary split."""
        decisions = ["yes"] * 80 + ["no"] * 20
        result = compute_dcb(decisions, k=2)
        expected_entropy = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        expected_dcb = 1.0 - expected_entropy / math.log(2)
        assert abs(result.dcb - expected_dcb) < 1e-10
        assert abs(result.entropy - expected_entropy) < 1e-10

    def test_single_decision_single_category(self):
        """K=1 → DCB = 1.0 (degenerate)."""
        result = compute_dcb(["only"], k=1)
        assert result.dcb == 1.0
        assert result.k_categories == 1


class TestDCBEdgeCases:
    """Edge case handling."""

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_dcb([])

    def test_k_inferred_from_data(self):
        """When k is None and no benchmark, infer from unique decisions."""
        result = compute_dcb(["a", "b", "c", "a", "b"])
        assert result.k_categories == 3

    def test_k_from_benchmark(self):
        """K derived from taxonomy when benchmark is provided."""
        result = compute_dcb(["escalate"] * 10, benchmark="compliance")
        assert result.k_categories == 3  # compliance has 3 categories

    def test_k_explicit_overrides_benchmark(self):
        """Explicit k takes precedence over benchmark lookup."""
        result = compute_dcb(["escalate"] * 10, k=5, benchmark="compliance")
        assert result.k_categories == 5

    def test_unknown_benchmark_falls_back(self):
        """Unknown benchmark falls back to inferred K."""
        result = compute_dcb(["a", "b"], benchmark="nonexistent_task")
        assert result.k_categories == 2  # inferred from data

    def test_case_insensitive(self):
        """Decision labels are normalized to lowercase."""
        result = compute_dcb(["ESCALATE", "Escalate", "escalate"], k=3)
        assert result.n_decisions == 3
        assert len(result.distribution) == 1

    def test_distribution_sums_to_one(self):
        decisions = ["a"] * 40 + ["b"] * 30 + ["c"] * 30
        result = compute_dcb(decisions, k=3)
        total = sum(result.distribution.values())
        assert abs(total - 1.0) < 1e-10


class TestDCBResult:
    """Result type correctness."""

    def test_result_is_frozen(self):
        result = compute_dcb(["a", "b"], k=2)
        with pytest.raises(AttributeError):
            result.dcb = 0.5  # type: ignore

    def test_result_fields_complete(self):
        result = compute_dcb(["x"] * 5, k=3)
        assert isinstance(result, DCBResult)
        assert isinstance(result.dcb, float)
        assert isinstance(result.entropy, float)
        assert isinstance(result.max_entropy, float)
        assert isinstance(result.distribution, dict)
        assert isinstance(result.n_decisions, int)
        assert isinstance(result.k_categories, int)
