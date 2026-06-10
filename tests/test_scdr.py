"""Tests for SCDR metric.

Uses pre-computed embeddings and mocks to avoid sentence-transformers
model downloads in CI.
"""

import numpy as np
import pytest
from bench.metrics.scdr import compute_scdr, SCDRResult, _compute_dar, _pairwise_cosine_distances


class TestSCDRWithPrecomputedEmbeddings:
    """SCDR tests using pre-computed embeddings (no model download)."""

    def test_n_equals_one(self):
        """Single run → SCDR = 0."""
        result = compute_scdr([("escalate", "check_sanctions")])
        assert result.scdr == 0.0
        assert result.dar == 1.0
        assert result.rds == 0.0
        assert result.n_runs == 1

    def test_identical_traces_rds_zero(self):
        """All identical traces → RDS = 0, SCDR = 0."""
        # Use pre-computed identical embeddings
        emb = np.array([[1.0, 0.0, 0.0]] * 5)
        runs = [("escalate", "same trace")] * 5
        result = compute_scdr(runs, embeddings=emb)
        assert result.rds == 0.0
        assert result.scdr == 0.0
        assert result.dar == 1.0
        assert result.mode == "precomputed"

    def test_orthogonal_traces_high_rds(self):
        """Orthogonal embeddings → RDS = 1.0."""
        emb = np.eye(3)  # 3 orthogonal vectors
        runs = [("escalate", f"trace_{i}") for i in range(3)]
        result = compute_scdr(runs, embeddings=emb)
        assert abs(result.rds - 1.0) < 1e-10
        assert result.dar == 1.0
        assert abs(result.scdr - 1.0) < 1e-10

    def test_mixed_decisions_lower_dar(self):
        """Mixed decisions → DAR < 1."""
        emb = np.eye(3)
        runs = [("escalate", "t1"), ("dismiss", "t2"), ("escalate", "t3")]
        result = compute_scdr(runs, embeddings=emb)
        assert abs(result.dar - 2.0 / 3.0) < 1e-10
        assert result.modal_decision == "escalate"

    def test_scdr_equals_dar_times_rds(self):
        """SCDR = DAR * RDS."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        runs = [("a", "t1"), ("a", "t2"), ("b", "t3")]
        result = compute_scdr(runs, embeddings=emb)
        assert abs(result.scdr - result.dar * result.rds) < 1e-10


class TestDARComputation:
    """Test DAR helper directly."""

    def test_all_agree(self):
        dar, modal, counts = _compute_dar(["a", "a", "a"])
        assert dar == 1.0
        assert modal == "a"

    def test_none_agree(self):
        dar, modal, counts = _compute_dar(["a", "b", "c"])
        assert abs(dar - 1.0 / 3.0) < 1e-10

    def test_case_insensitive(self):
        dar, modal, counts = _compute_dar(["ESCALATE", "Escalate", "escalate"])
        assert dar == 1.0


class TestPairwiseCosineDistance:
    """Test distance computation helper."""

    def test_single_vector(self):
        assert _pairwise_cosine_distances(np.array([[1.0, 0.0]])) == 0.0

    def test_identical_vectors(self):
        emb = np.array([[1.0, 0.0]] * 4)
        assert _pairwise_cosine_distances(emb) == 0.0

    def test_orthogonal_vectors(self):
        emb = np.eye(3)
        assert abs(_pairwise_cosine_distances(emb) - 1.0) < 1e-10


class TestSCDREdgeCases:
    """Edge case handling."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_scdr([])

    def test_mode_reported_in_result(self):
        emb = np.array([[1.0, 0.0]])
        result = compute_scdr([("a", "t")], embeddings=emb)
        # N=1 doesn't use embeddings, so mode stays as default
        assert result.mode in ("auto", "precomputed", "trajectory", "rationale")

    def test_result_is_frozen(self):
        result = compute_scdr([("a", "t")])
        with pytest.raises(AttributeError):
            result.dar = 0.5  # type: ignore
