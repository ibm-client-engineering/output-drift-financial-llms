"""Tests for Evidence Contact Divergence (ECD) metric."""

import pytest
from bench.metrics.ecd import compute_ecd, ECDResult


class TestECDBasic:
    """Core ECD computation with known expected values."""

    def test_identical_evidence_ecd_zero(self):
        """All runs consult same evidence → ECD = 0."""
        evidence = [{"doc_a", "doc_b"}] * 5
        result = compute_ecd(evidence)
        assert result.ecd == 0.0
        assert result.n_runs == 5

    def test_disjoint_evidence_ecd_one(self):
        """Completely disjoint evidence → ECD = 1.0."""
        evidence = [{"a"}, {"b"}, {"c"}]
        result = compute_ecd(evidence)
        assert result.ecd == 1.0

    def test_50_percent_overlap(self):
        """Two runs with 50% overlap → Jaccard distance = 1/3 ≈ 0.333."""
        # A = {a, b}, B = {b, c} → |A∩B|=1, |A∪B|=3 → JD = 1-1/3 = 2/3
        evidence = [{"a", "b"}, {"b", "c"}]
        result = compute_ecd(evidence)
        assert abs(result.ecd - 2.0 / 3.0) < 1e-10

    def test_scde_compound_metric(self):
        """SCDE = DAR * ECD when decisions are provided."""
        evidence = [{"a"}, {"b"}, {"a"}]
        decisions = ["escalate", "escalate", "escalate"]
        result = compute_ecd(evidence, decisions=decisions)
        assert result.dar == 1.0  # all agree
        assert result.scde == result.dar * result.ecd

    def test_scde_with_mixed_decisions(self):
        """DAR < 1 with mixed decisions."""
        evidence = [{"a"}, {"b"}, {"c"}]
        decisions = ["escalate", "dismiss", "escalate"]
        result = compute_ecd(evidence, decisions=decisions)
        assert result.dar == 2.0 / 3.0  # 2 out of 3 agree on "escalate"
        assert abs(result.scde - result.dar * result.ecd) < 1e-10


class TestECDEdgeCases:
    """Edge case handling."""

    def test_n_equals_one(self):
        """Single run → ECD = 0 (no divergence measurable)."""
        result = compute_ecd([{"a", "b"}])
        assert result.ecd == 0.0
        assert result.n_runs == 1

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_ecd([])

    def test_empty_evidence_sets_warning(self):
        """Runs with no evidence contacts → warning flag."""
        result = compute_ecd([set(), set()])
        assert result.ecd == 0.0
        assert result.empty_evidence_warning is True

    def test_decisions_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            compute_ecd([{"a"}], decisions=["x", "y"])

    def test_deduplication(self):
        """Duplicate IDs within a run are deduplicated."""
        evidence = [["a", "b", "a", "b"], ["a", "b"]]  # lists with dupes
        result = compute_ecd(evidence)
        assert result.ecd == 0.0  # identical after dedup

    def test_list_input_accepted(self):
        """Lists are auto-converted to sets."""
        result = compute_ecd([["x", "y"], ["y", "z"]])
        assert result.n_runs == 2
        assert result.ecd > 0

    def test_union_and_intersection(self):
        evidence = [{"a", "b", "c"}, {"b", "c", "d"}]
        result = compute_ecd(evidence)
        assert result.union_contact_count == 4  # a, b, c, d
        assert result.intersection_contact_count == 2  # b, c

    def test_mean_contact_count(self):
        evidence = [{"a", "b"}, {"x", "y", "z"}]
        result = compute_ecd(evidence)
        assert result.mean_contact_count == 2.5


class TestECDResult:
    """Result type correctness."""

    def test_result_is_frozen(self):
        result = compute_ecd([{"a"}])
        with pytest.raises(AttributeError):
            result.ecd = 0.5  # type: ignore

    def test_no_decisions_dar_is_none(self):
        result = compute_ecd([{"a"}, {"b"}])
        assert result.dar is None
        assert result.scde is None
