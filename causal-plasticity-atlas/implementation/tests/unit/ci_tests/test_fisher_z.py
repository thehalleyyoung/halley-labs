"""Unit tests for cpa.ci_tests.fisher_z."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.ci_tests.fisher_z import CITestResult, FisherZTest, PartialCorrelation


# ── helpers ─────────────────────────────────────────────────────────

def _make_independent_data(n: int = 500, p: int = 3, seed: int = 42):
    """Generate p independent Gaussian variables."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def _make_chain_data(n: int = 500, seed: int = 42):
    """Generate chain X0 -> X1 -> X2 with known coefficients."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + rng.standard_normal(n) * 0.5
    x2 = 0.7 * x1 + rng.standard_normal(n) * 0.5
    return np.column_stack([x0, x1, x2])


def _make_confounded_data(n: int = 500, seed: int = 42):
    """X0 <- X2 -> X1: X0 and X1 are correlated via confounder X2."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(n)
    x0 = 0.9 * x2 + rng.standard_normal(n) * 0.3
    x1 = 0.9 * x2 + rng.standard_normal(n) * 0.3
    return np.column_stack([x0, x1, x2])


def _make_five_var_data(n: int = 1000, seed: int = 42):
    """X0 -> X1 -> X2; X3 -> X4; all others independent given parents."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.7 * x0 + rng.standard_normal(n) * 0.5
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.5
    x3 = rng.standard_normal(n)
    x4 = 0.8 * x3 + rng.standard_normal(n) * 0.4
    return np.column_stack([x0, x1, x2, x3, x4])


# ── PartialCorrelation tests ───────────────────────────────────────

class TestPartialCorrelation:
    """Tests for the PartialCorrelation helper class."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            PartialCorrelation(method="bogus")

    def test_zero_correlation_independent_data(self):
        data = _make_independent_data(n=1000, seed=99)
        pc = PartialCorrelation("matrix_inversion")
        r = pc.compute(data, 0, 1, set())
        assert abs(r) < 0.1

    def test_high_correlation_chain(self):
        data = _make_chain_data(n=2000, seed=7)
        pc = PartialCorrelation("matrix_inversion")
        r01 = pc.compute(data, 0, 1, set())
        assert abs(r01) > 0.5

    def test_partial_correlation_removes_confounder(self):
        data = _make_confounded_data(n=2000, seed=11)
        pc = PartialCorrelation("matrix_inversion")
        r_marginal = pc.compute(data, 0, 1, set())
        r_partial = pc.compute(data, 0, 1, {2})
        assert abs(r_marginal) > 0.3
        assert abs(r_partial) < 0.15

    def test_matrix_inversion_vs_recursive_agree(self):
        data = _make_chain_data(n=500, seed=42)
        pc_mi = PartialCorrelation("matrix_inversion")
        pc_re = PartialCorrelation("recursive")
        r_mi = pc_mi.compute(data, 0, 2, {1})
        r_re = pc_re.compute(data, 0, 2, {1})
        assert abs(r_mi - r_re) < 0.05

    def test_compute_from_cov_matches_compute(self):
        data = _make_chain_data(n=500, seed=42)
        cov = np.cov(data, rowvar=False, ddof=1)
        pc = PartialCorrelation("matrix_inversion")
        r_data = pc.compute(data, 0, 2, {1})
        r_cov = pc.compute_from_cov(cov, 0, 2, {1})
        assert abs(r_data - r_cov) < 1e-10

    def test_empty_conditioning_set_is_pearson(self):
        data = _make_chain_data(n=500, seed=42)
        pc = PartialCorrelation("matrix_inversion")
        r = pc.compute(data, 0, 1, set())
        pearson = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        assert abs(r - pearson) < 1e-10

    def test_single_conditioning_variable(self):
        data = _make_chain_data(n=1000, seed=42)
        pc = PartialCorrelation("matrix_inversion")
        r = pc.compute(data, 0, 2, {1})
        # X0 and X2 should be near-independent given X1 in a chain
        assert abs(r) < 0.15

    def test_data_validation_not_2d(self):
        pc = PartialCorrelation()
        with pytest.raises(ValueError, match="2-D"):
            pc.compute(np.ones(10), 0, 1, set())

    def test_index_out_of_range(self):
        data = _make_independent_data(n=50, p=3)
        pc = PartialCorrelation()
        with pytest.raises(IndexError):
            pc.compute(data, 0, 5, set())

    def test_too_few_samples(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = PartialCorrelation()
        with pytest.raises(ValueError, match="Not enough samples"):
            pc.compute(data, 0, 1, {2})

    def test_perfect_positive_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        data = np.column_stack([x, x, rng.standard_normal(100)])
        pc = PartialCorrelation()
        r = pc.compute(data, 0, 1, set())
        assert r > 0.99

    def test_perfect_negative_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        data = np.column_stack([x, -x, rng.standard_normal(100)])
        pc = PartialCorrelation()
        r = pc.compute(data, 0, 1, set())
        assert r < -0.99

    def test_recursive_with_multiple_conditioning(self):
        data = _make_five_var_data(n=1000, seed=42)
        pc = PartialCorrelation("recursive")
        r = pc.compute(data, 0, 2, {1, 3})
        assert abs(r) < 0.15

    def test_cov_not_square_raises(self):
        pc = PartialCorrelation()
        with pytest.raises(ValueError, match="square"):
            pc.compute_from_cov(np.ones((3, 4)), 0, 1, set())


# ── FisherZTest tests ──────────────────────────────────────────────

class TestFisherZTest:
    """Tests for the FisherZTest class."""

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            FisherZTest(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            FisherZTest(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            FisherZTest(alpha=-0.1)

    def test_independent_data_not_rejected(self):
        data = _make_independent_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        stat, pval = fz.test(data, 0, 1)
        assert pval > 0.05, "Independent data should not be rejected"

    def test_correlated_data_rejected(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        stat, pval = fz.test(data, 0, 1)
        assert pval < 0.05, "Correlated data should be rejected"

    def test_conditional_independence_chain(self):
        """In X0 -> X1 -> X2, X0 ⊥ X2 | X1."""
        data = _make_chain_data(n=2000, seed=42)
        fz = FisherZTest(alpha=0.05)
        stat, pval = fz.test(data, 0, 2, conditioning_set={1})
        assert pval > 0.05

    def test_marginal_dependence_chain(self):
        """In X0 -> X1 -> X2, X0 and X2 are marginally dependent."""
        data = _make_chain_data(n=2000, seed=42)
        fz = FisherZTest(alpha=0.05)
        stat, pval = fz.test(data, 0, 2)
        assert pval < 0.05

    def test_confounded_marginal_dependent(self):
        """X0 and X1 confounded by X2 should be marginally dependent."""
        data = _make_confounded_data(n=1000, seed=42)
        fz = FisherZTest(alpha=0.05)
        _, pval = fz.test(data, 0, 1)
        assert pval < 0.05

    def test_confounded_conditional_independent(self):
        """X0 ⊥ X1 | X2 when X2 is the confounder."""
        data = _make_confounded_data(n=5000, seed=42)
        fz = FisherZTest(alpha=0.01)
        _, pval = fz.test(data, 0, 1, conditioning_set={2})
        assert pval > 0.01

    def test_test_full_returns_ci_result(self):
        data = _make_independent_data(n=200, seed=42)
        fz = FisherZTest(alpha=0.05)
        result = fz.test_full(data, 0, 1)
        assert isinstance(result, CITestResult)
        assert result.method == "fisher_z"
        assert result.independent is True
        assert result.partial_corr is not None

    def test_test_full_correlated(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        result = fz.test_full(data, 0, 1)
        assert result.independent is False
        assert abs(result.partial_corr) > 0.3

    def test_fisher_z_transform_properties(self):
        # Zero correlation => zero statistic
        z = FisherZTest.fisher_z_transform(0.0, 100, 0)
        assert z == 0.0

        # Positive correlation => positive statistic
        z = FisherZTest.fisher_z_transform(0.5, 100, 0)
        assert z > 0

        # Negative correlation => negative statistic
        z = FisherZTest.fisher_z_transform(-0.5, 100, 0)
        assert z < 0

    def test_fisher_z_insufficient_dof(self):
        z = FisherZTest.fisher_z_transform(0.5, 5, 3)
        assert z == 0.0

    def test_fisher_z_near_perfect_correlation(self):
        z = FisherZTest.fisher_z_transform(0.9999, 100, 0)
        assert np.isfinite(z)
        assert z > 0

    def test_insufficient_dof_returns_no_rejection(self):
        # n=4, k=1 => n-k-3=0, triggers early return
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, 3))
        fz = FisherZTest(alpha=0.05)
        stat, pval = fz.test(data, 0, 2, conditioning_set={1})
        assert pval == 1.0
        assert stat == 0.0

    def test_index_out_of_range_raises(self):
        data = _make_independent_data(n=50, p=3)
        fz = FisherZTest(alpha=0.05)
        with pytest.raises(IndexError):
            fz.test(data, 0, 5)

    def test_not_2d_raises(self):
        fz = FisherZTest(alpha=0.05)
        with pytest.raises(ValueError, match="2-D"):
            fz.test(np.ones(10), 0, 1)

    def test_method_recursive(self):
        data = _make_chain_data(n=500, seed=42)
        fz_mi = FisherZTest(alpha=0.05, method="matrix_inversion")
        fz_re = FisherZTest(alpha=0.05, method="recursive")
        _, p_mi = fz_mi.test(data, 0, 2, conditioning_set={1})
        _, p_re = fz_re.test(data, 0, 2, conditioning_set={1})
        # Both should agree on independence decision
        assert (p_mi > 0.05) == (p_re > 0.05)

    def test_all_pairwise_independent(self):
        data = _make_independent_data(n=500, p=4, seed=42)
        fz = FisherZTest(alpha=0.05)
        pvals = fz.all_pairwise(data)
        assert pvals.shape == (4, 4)
        # Diagonal should be 1.0
        assert_array_almost_equal(np.diag(pvals), np.ones(4))
        # Off-diag should be > 0.05 for independent data (most of them)
        off_diag = pvals[np.triu_indices(4, k=1)]
        assert np.mean(off_diag > 0.05) > 0.5

    def test_all_pairwise_chain(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        pvals = fz.all_pairwise(data)
        assert pvals[0, 1] < 0.05  # adjacent in chain

    def test_stable_test_finds_separating_set(self):
        data = _make_chain_data(n=2000, seed=42)
        fz = FisherZTest(alpha=0.05)
        indep, sep_set = fz.stable_test(data, 0, 2, candidate_z=[1])
        assert indep is True
        assert 1 in sep_set

    def test_stable_test_no_separation(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        indep, sep_set = fz.stable_test(data, 0, 1, candidate_z=[2])
        assert indep is False

    def test_stable_test_max_order(self):
        data = _make_five_var_data(n=1000, seed=42)
        fz = FisherZTest(alpha=0.05)
        indep, _ = fz.stable_test(data, 0, 2, candidate_z=[1, 3, 4], max_order=1)
        assert indep is True

    def test_partial_correlation_symmetry(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        r01 = fz.partial_correlation(data, 0, 1, set())
        r10 = fz.partial_correlation(data, 1, 0, set())
        assert abs(r01 - r10) < 1e-10

    def test_partial_correlation_from_cov(self):
        data = _make_chain_data(n=500, seed=42)
        cov = np.cov(data, rowvar=False, ddof=1)
        fz = FisherZTest(alpha=0.05)
        r = fz.partial_correlation_from_cov(cov, 0, 1, set())
        r_data = fz.partial_correlation(data, 0, 1, set())
        assert abs(r - r_data) < 1e-10

    def test_larger_sample_size_more_power(self):
        """Larger n should yield smaller p-value for same effect."""
        fz = FisherZTest(alpha=0.05)
        data_small = _make_chain_data(n=50, seed=42)
        data_large = _make_chain_data(n=2000, seed=42)
        _, p_small = fz.test(data_small, 0, 1)
        _, p_large = fz.test(data_large, 0, 1)
        assert p_large <= p_small

    def test_five_var_independence_structure(self):
        data = _make_five_var_data(n=2000, seed=42)
        fz = FisherZTest(alpha=0.05)
        # X0 and X3 should be independent (no connection)
        _, p_03 = fz.test(data, 0, 3)
        assert p_03 > 0.05
        # X3 and X4 should be dependent
        _, p_34 = fz.test(data, 3, 4)
        assert p_34 < 0.05

    def test_conditioning_set_none_treated_as_empty(self):
        data = _make_chain_data(n=500, seed=42)
        fz = FisherZTest(alpha=0.05)
        s1, p1 = fz.test(data, 0, 1, conditioning_set=None)
        s2, p2 = fz.test(data, 0, 1, conditioning_set=set())
        assert s1 == s2
        assert p1 == p2
