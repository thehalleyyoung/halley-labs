"""Unit tests for cpa.scores.bge."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.scores.bge import BGeScore, _log_det_spd, _log_multivariate_gamma


# ── helpers ─────────────────────────────────────────────────────────

def _chain_data(n: int = 500, seed: int = 42):
    """X0 -> X1 -> X2."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + rng.standard_normal(n) * 0.3
    x2 = 0.7 * x1 + rng.standard_normal(n) * 0.3
    return np.column_stack([x0, x1, x2])


def _fork_data(n: int = 500, seed: int = 42):
    """X0 -> X1, X0 -> X2."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.9 * x0 + rng.standard_normal(n) * 0.2
    x2 = 0.6 * x0 + rng.standard_normal(n) * 0.4
    return np.column_stack([x0, x1, x2])


def _independent_data(n: int = 500, p: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def _five_var_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.7 * x0 + rng.standard_normal(n) * 0.5
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.5
    x3 = rng.standard_normal(n)
    x4 = 0.8 * x3 + rng.standard_normal(n) * 0.4
    return np.column_stack([x0, x1, x2, x3, x4])


# ── _log_multivariate_gamma tests ──────────────────────────────────

class TestLogMultivariateGamma:
    def test_p_zero(self):
        assert _log_multivariate_gamma(5.0, 0) == 0.0

    def test_p_one_matches_loggamma(self):
        from scipy.special import gammaln
        val = _log_multivariate_gamma(3.0, 1)
        expected = gammaln(3.0)
        assert abs(val - expected) < 1e-10

    def test_p_two(self):
        val = _log_multivariate_gamma(5.0, 2)
        assert np.isfinite(val)


# ── _log_det_spd tests ────────────────────────────────────────────

class TestLogDetSPD:
    def test_identity(self):
        ld = _log_det_spd(np.eye(5))
        assert abs(ld) < 1e-10

    def test_scaled_identity(self):
        ld = _log_det_spd(2.0 * np.eye(3))
        expected = 3 * math.log(2.0)
        assert abs(ld - expected) < 1e-10

    def test_positive_definite(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        M = A @ A.T + np.eye(4)
        ld = _log_det_spd(M)
        _, expected = np.linalg.slogdet(M)
        assert abs(ld - expected) < 1e-8


# ── BGeScore tests ─────────────────────────────────────────────────

class TestBGeScore:
    def test_true_parents_score_higher(self):
        data = _chain_data(n=500, seed=42)
        bge = BGeScore(data)
        s_true = bge.local_score(1, [0])
        s_false = bge.local_score(1, [2])
        assert s_true > s_false

    def test_no_parents_vs_true_parents(self):
        data = _chain_data(n=500, seed=42)
        bge = BGeScore(data)
        s_empty = bge.local_score(1, [])
        s_true = bge.local_score(1, [0])
        assert s_true > s_empty

    def test_score_dag_is_sum_of_local(self):
        data = _chain_data(n=200, seed=42)
        bge = BGeScore(data)
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        total = bge.score_dag(adj)
        local_sum = (bge.local_score(0, [])
                     + bge.local_score(1, [0])
                     + bge.local_score(2, [1]))
        assert abs(total - local_sum) < 1e-10

    def test_invalid_alpha_mu_raises(self):
        data = _chain_data(n=100, seed=42)
        with pytest.raises(ValueError, match="alpha_mu"):
            BGeScore(data, alpha_mu=-1.0)

    def test_invalid_alpha_w_raises(self):
        data = _chain_data(n=100, seed=42)
        with pytest.raises(ValueError, match="alpha_w"):
            BGeScore(data, alpha_w=3)  # needs > p+1 = 4

    def test_node_out_of_range_raises(self):
        data = _chain_data(n=100, seed=42)
        bge = BGeScore(data)
        with pytest.raises(ValueError, match="out of range"):
            bge.local_score(10, [0])

    def test_self_parent_raises(self):
        data = _chain_data(n=100, seed=42)
        bge = BGeScore(data)
        with pytest.raises(ValueError, match="own parent"):
            bge.local_score(1, [1])

    def test_score_finite(self):
        data = _chain_data(n=200, seed=42)
        bge = BGeScore(data)
        for node in range(3):
            for parents in [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]:
                if node in parents:
                    continue
                s = bge.local_score(node, parents)
                assert np.isfinite(s), f"Non-finite score for node={node}, pa={parents}"

    def test_prior_sensitivity_alpha_mu(self):
        """Different alpha_mu values should yield different scores."""
        data = _chain_data(n=200, seed=42)
        bge1 = BGeScore(data, alpha_mu=0.1)
        bge2 = BGeScore(data, alpha_mu=10.0)
        s1 = bge1.local_score(1, [0])
        s2 = bge2.local_score(1, [0])
        assert s1 != s2

    def test_score_ordering_true_vs_random(self):
        """True DAG structure should score higher than random structures."""
        data = _chain_data(n=500, seed=42)
        bge = BGeScore(data)
        # True DAG: 0->1->2
        adj_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        # Random wrong DAG: 2->0, 1->0
        adj_wrong = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        s_true = bge.score_dag(adj_true)
        s_wrong = bge.score_dag(adj_wrong)
        assert s_true > s_wrong

    def test_score_edge_addition(self):
        data = _chain_data(n=300, seed=42)
        bge = BGeScore(data)
        delta = bge.score_edge_addition(1, [], 0)
        assert delta > 0  # adding true parent should improve score

    def test_score_edge_removal(self):
        data = _chain_data(n=300, seed=42)
        bge = BGeScore(data)
        delta = bge.score_edge_removal(1, [0], 0)
        assert delta < 0  # removing true parent should hurt

    def test_posterior_parameters(self):
        data = _chain_data(n=200, seed=42)
        bge = BGeScore(data)
        params = bge.posterior_parameters(1, [0])
        assert "alpha_w_post" in params
        assert "mu_post" in params
        assert "T_n_family" in params
        assert params["alpha_w_post"] == bge.alpha_w + bge.n_samples

    def test_repr(self):
        data = _chain_data(n=100, seed=42)
        bge = BGeScore(data)
        r = repr(bge)
        assert "BGeScore" in r

    def test_large_sample_consistency_with_bic(self):
        """For large n, BGe and BIC should agree on structure ranking."""
        data = _chain_data(n=2000, seed=42)
        bge = BGeScore(data)
        from cpa.scores.bic import BICScore
        bic = BICScore(data)

        # True: parents of 1 = [0]
        # False: parents of 1 = [2]
        bge_true = bge.local_score(1, [0])
        bge_false = bge.local_score(1, [2])
        bic_true = bic.local_score(1, [0])
        bic_false = bic.local_score(1, [2])

        # Both should rank true parents higher
        assert (bge_true > bge_false) == (bic_true > bic_false)

    def test_five_var_structure(self):
        """Test correct structure ranking on 5-variable graph."""
        data = _five_var_data(n=500, seed=42)
        bge = BGeScore(data)
        # X4's true parent is X3
        s_true = bge.local_score(4, [3])
        s_false = bge.local_score(4, [0])
        s_empty = bge.local_score(4, [])
        assert s_true > s_false
        assert s_true > s_empty
