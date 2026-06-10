"""Tests for bootstrap CI and significance utilities."""

import numpy as np
import pytest
from bench.stats.bootstrap import bootstrap_ci
from bench.stats.significance import permutation_test, spearman_correlation


class TestBootstrapCI:
    """Bootstrap confidence interval tests."""

    def test_output_shape(self):
        """Returns (point, lower, upper)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point, lo, hi = bootstrap_ci(np.mean, data, seed=42)
        assert isinstance(point, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_point_estimate_is_statistic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point, lo, hi = bootstrap_ci(np.mean, data, seed=42)
        assert abs(point - 3.0) < 1e-10

    def test_ci_contains_point(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point, lo, hi = bootstrap_ci(np.mean, data, seed=42)
        assert lo <= point <= hi

    def test_deterministic_seed(self):
        """Same seed → identical CIs."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r1 = bootstrap_ci(np.mean, data, seed=123)
        r2 = bootstrap_ci(np.mean, data, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds produce different bootstrap samples (with enough data)."""
        rng = np.random.default_rng(0)
        data = rng.normal(50, 10, size=100)
        r1 = bootstrap_ci(np.mean, data, seed=1)
        r2 = bootstrap_ci(np.mean, data, seed=2)
        assert r1[1] != r2[1] or r1[2] != r2[2]

    def test_narrow_ci_for_constant_data(self):
        data = np.array([5.0] * 100)
        point, lo, hi = bootstrap_ci(np.mean, data, seed=42)
        assert abs(lo - 5.0) < 1e-10
        assert abs(hi - 5.0) < 1e-10


class TestPermutationTest:
    """Permutation test correctness."""

    def test_identical_groups_high_p(self):
        """Identical distributions → p ≈ 1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        diff, p = permutation_test(np.mean, a, b, seed=42)
        assert abs(diff) < 1e-10
        assert p > 0.5

    def test_clearly_different_groups_low_p(self):
        """Very different distributions → p < 0.05."""
        a = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        diff, p = permutation_test(np.mean, a, b, seed=42)
        assert diff > 90  # large observed difference
        assert p < 0.05

    def test_deterministic_seed(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        r1 = permutation_test(np.mean, a, b, seed=99)
        r2 = permutation_test(np.mean, a, b, seed=99)
        assert r1 == r2


class TestSpearmanCorrelation:
    """Spearman rank correlation tests."""

    def test_perfect_positive(self):
        rho, p = spearman_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert abs(rho - 1.0) < 1e-10

    def test_perfect_negative(self):
        rho, p = spearman_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(rho - (-1.0)) < 1e-10

    def test_returns_tuple(self):
        result = spearman_correlation([1, 2, 3], [3, 2, 1])
        assert len(result) == 2
