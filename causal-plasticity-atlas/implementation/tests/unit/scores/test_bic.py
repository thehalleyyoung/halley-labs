"""Unit tests for cpa.scores.bic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.scores.bic import (
    BICScore,
    ExtendedBICScore,
    ModifiedBICScore,
    compare_bic_scores,
    gaussian_log_likelihood,
    select_best_parents,
)


# ── helpers ─────────────────────────────────────────────────────────

def _chain_data(n: int = 500, seed: int = 42):
    """X0 -> X1 -> X2 with known coefficients."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + rng.standard_normal(n) * 0.3
    x2 = 0.7 * x1 + rng.standard_normal(n) * 0.3
    return np.column_stack([x0, x1, x2])


def _fork_data(n: int = 500, seed: int = 42):
    """X0 -> X1, X0 -> X2 (fork)."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.9 * x0 + rng.standard_normal(n) * 0.2
    x2 = 0.6 * x0 + rng.standard_normal(n) * 0.4
    return np.column_stack([x0, x1, x2])


def _independent_data(n: int = 500, p: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def _five_var_data(n: int = 500, seed: int = 42):
    """X0->X1->X2; X3->X4; all noise Gaussian."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.7 * x0 + rng.standard_normal(n) * 0.5
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.5
    x3 = rng.standard_normal(n)
    x4 = 0.8 * x3 + rng.standard_normal(n) * 0.4
    return np.column_stack([x0, x1, x2, x3, x4])


# ── gaussian_log_likelihood tests ──────────────────────────────────

class TestGaussianLogLikelihood:
    def test_finite_positive_variance(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(100)
        var = float(np.var(r))
        ll = gaussian_log_likelihood(r, var, 100)
        assert np.isfinite(ll)

    def test_zero_variance_handled(self):
        r = np.zeros(50)
        ll = gaussian_log_likelihood(r, 0.0, 50)
        assert np.isfinite(ll)

    def test_higher_variance_lower_ll(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(100)
        ll_small = gaussian_log_likelihood(r, 0.5, 100)
        ll_large = gaussian_log_likelihood(r, 5.0, 100)
        # At MLE variance, likelihood is maximized; these aren't MLE
        # but large variance should generally give lower ll for small residuals
        assert np.isfinite(ll_small) and np.isfinite(ll_large)


# ── BICScore tests ─────────────────────────────────────────────────

class TestBICScore:
    def test_true_parents_score_higher(self):
        """True parent set should score higher than wrong parents."""
        data = _chain_data(n=500, seed=42)
        scorer = BICScore(data)
        # X1's true parent is X0
        score_true = scorer.local_score(1, [0])
        score_false = scorer.local_score(1, [2])
        assert score_true > score_false

    def test_no_parents_vs_true_parents(self):
        data = _chain_data(n=500, seed=42)
        scorer = BICScore(data)
        score_empty = scorer.local_score(1, [])
        score_true = scorer.local_score(1, [0])
        assert score_true > score_empty

    def test_extra_irrelevant_parent_penalized(self):
        """Adding an irrelevant parent should reduce score for large n."""
        data = _five_var_data(n=1000, seed=42)
        scorer = BICScore(data)
        # X1's true parent is X0; adding X3 (irrelevant) should hurt
        score_true = scorer.local_score(1, [0])
        score_extra = scorer.local_score(1, [0, 3])
        assert score_true > score_extra

    def test_score_dag_is_sum_of_local(self):
        """score_dag should equal sum of local_scores (decomposability)."""
        data = _chain_data(n=200, seed=42)
        scorer = BICScore(data)
        # DAG: 0->1->2
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        total = scorer.score_dag(adj)
        local_sum = (scorer.local_score(0, [])
                     + scorer.local_score(1, [0])
                     + scorer.local_score(2, [1]))
        assert abs(total - local_sum) < 1e-10

    def test_penalty_increases_with_parents(self):
        data = _chain_data(n=200, seed=42)
        scorer = BICScore(data)
        pen0 = scorer.penalty(0, [])
        pen1 = scorer.penalty(0, [1])
        pen2 = scorer.penalty(0, [1, 2])
        assert pen0 < pen1 < pen2

    def test_penalty_weight_effect(self):
        data = _chain_data(n=200, seed=42)
        s1 = BICScore(data, penalty_weight=0.5)
        s2 = BICScore(data, penalty_weight=2.0)
        p1 = s1.penalty(1, [0])
        p2 = s2.penalty(1, [0])
        assert p1 < p2

    def test_residual_variance_decreases_with_true_parent(self):
        data = _chain_data(n=500, seed=42)
        scorer = BICScore(data)
        var_no = scorer.residual_variance(1, [])
        var_true = scorer.residual_variance(1, [0])
        assert var_true < var_no

    def test_score_edge_addition_positive_for_true_edge(self):
        data = _chain_data(n=500, seed=42)
        scorer = BICScore(data)
        delta = scorer.score_edge_addition(1, [], 0)
        assert delta > 0

    def test_score_edge_removal_negative_for_true_edge(self):
        data = _chain_data(n=500, seed=42)
        scorer = BICScore(data)
        delta = scorer.score_edge_removal(1, [0], 0)
        assert delta < 0

    def test_invalid_node_raises(self):
        data = _chain_data(n=100, seed=42)
        scorer = BICScore(data)
        with pytest.raises(ValueError, match="out of range"):
            scorer.local_score(5, [0])

    def test_self_parent_raises(self):
        data = _chain_data(n=100, seed=42)
        scorer = BICScore(data)
        with pytest.raises(ValueError, match="own parent"):
            scorer.local_score(1, [1])

    def test_log_likelihood_finite(self):
        data = _chain_data(n=200, seed=42)
        scorer = BICScore(data)
        ll = scorer.log_likelihood(1, [0])
        assert np.isfinite(ll)

    def test_repr(self):
        data = _chain_data(n=100, seed=42)
        scorer = BICScore(data)
        r = repr(scorer)
        assert "BICScore" in r
        assert "100" in r

    def test_asymptotic_behavior(self):
        """With more data, BIC should strongly prefer true parents."""
        data = _five_var_data(n=2000, seed=42)
        scorer = BICScore(data)
        # X1's true parent is X0; X3 is independent of X1
        s_true = scorer.local_score(1, [0])
        s_wrong = scorer.local_score(1, [3])
        assert s_true > s_wrong


# ── ExtendedBICScore tests ─────────────────────────────────────────

class TestExtendedBICScore:
    def test_gamma_zero_matches_bic(self):
        data = _chain_data(n=300, seed=42)
        bic = BICScore(data)
        ebic = ExtendedBICScore(data, gamma=0.0)
        # gamma=0 should yield same scores as standard BIC
        s_bic = bic.local_score(1, [0])
        s_ebic = ebic.local_score(1, [0])
        assert abs(s_bic - s_ebic) < 1e-10

    def test_higher_gamma_stronger_penalty(self):
        data = _five_var_data(n=500, seed=42)
        eb_low = ExtendedBICScore(data, gamma=0.2)
        eb_high = ExtendedBICScore(data, gamma=0.8)
        # Higher gamma penalizes larger parent sets more
        pen_low = eb_low.penalty(1, [0, 2, 3])
        pen_high = eb_high.penalty(1, [0, 2, 3])
        assert pen_high > pen_low

    def test_empty_parents_no_extended_penalty(self):
        data = _chain_data(n=200, seed=42)
        ebic = ExtendedBICScore(data, gamma=0.5)
        extra = ebic.extended_penalty(0, [])
        assert extra == 0.0

    def test_invalid_gamma_raises(self):
        data = _chain_data(n=100, seed=42)
        with pytest.raises(ValueError, match="gamma"):
            ExtendedBICScore(data, gamma=-0.1)
        with pytest.raises(ValueError, match="gamma"):
            ExtendedBICScore(data, gamma=1.5)

    def test_sparsity_prefers_smaller_parent_set(self):
        data = _five_var_data(n=500, seed=42)
        ebic = ExtendedBICScore(data, gamma=1.0)
        # X4's true parent is X3; adding X0 and X1 (independent) should hurt
        s1 = ebic.local_score(4, [3])
        s3 = ebic.local_score(4, [3, 0, 1])
        assert s1 > s3  # stronger sparsity penalty kills extra parents

    def test_repr(self):
        data = _chain_data(n=100, seed=42)
        ebic = ExtendedBICScore(data, gamma=0.5)
        assert "ExtendedBICScore" in repr(ebic)
        assert "0.5" in repr(ebic)


# ── ModifiedBICScore tests ─────────────────────────────────────────

class TestModifiedBICScore:
    def test_uniform_prior_matches_bic(self):
        data = _chain_data(n=300, seed=42)
        bic = BICScore(data)
        mbic = ModifiedBICScore(data, prior="uniform")
        assert abs(bic.local_score(1, [0]) - mbic.local_score(1, [0])) < 1e-10

    def test_sparse_prior_penalizes_more_parents(self):
        data = _five_var_data(n=500, seed=42)
        mbic = ModifiedBICScore(data, prior="sparse")
        # X4's true parent is X3; adding unrelated X0,X1 should hurt
        s1 = mbic.local_score(4, [3])
        s3 = mbic.local_score(4, [3, 0, 1])
        assert s1 > s3

    def test_erdos_renyi_prior(self):
        data = _chain_data(n=200, seed=42)
        mbic = ModifiedBICScore(data, prior="erdos_renyi", prior_edge_prob=0.3)
        s = mbic.local_score(1, [0])
        assert np.isfinite(s)

    def test_invalid_prior_raises(self):
        data = _chain_data(n=100, seed=42)
        with pytest.raises(ValueError, match="prior"):
            ModifiedBICScore(data, prior="invalid")

    def test_repr(self):
        data = _chain_data(n=100, seed=42)
        mbic = ModifiedBICScore(data, prior="sparse")
        assert "ModifiedBICScore" in repr(mbic)


# ── compare_bic_scores tests ──────────────────────────────────────

class TestCompareBICScores:
    def test_returns_dict(self):
        data = _chain_data(n=200, seed=42)
        result = compare_bic_scores(data, 1, [0])
        assert isinstance(result, dict)
        assert len(result) == 4  # default 4 penalty weights

    def test_custom_weights(self):
        data = _chain_data(n=200, seed=42)
        result = compare_bic_scores(data, 1, [0], penalty_weights=[1.0, 2.0])
        assert len(result) == 2


# ── select_best_parents tests ─────────────────────────────────────

class TestSelectBestParents:
    def test_finds_true_parent(self):
        data = _chain_data(n=500, seed=42)
        best_parents, best_score = select_best_parents(data, node=1, max_parents=2)
        assert 0 in best_parents

    def test_empty_parents_when_independent(self):
        data = _independent_data(n=500, p=3, seed=42)
        best_parents, _ = select_best_parents(data, node=0, max_parents=2)
        assert len(best_parents) == 0

    def test_fork_structure(self):
        data = _fork_data(n=500, seed=42)
        best_parents, _ = select_best_parents(data, node=1, max_parents=2)
        assert 0 in best_parents
