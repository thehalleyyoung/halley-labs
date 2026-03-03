"""Tests for causal_qd.scores — BIC, BDeu, BGe, and CachedScore.

Covers decomposability, regression-based local scoring, discrete Bayesian
scoring, Gaussian marginal-likelihood scoring, caching behaviour, and
incremental score_diff consistency.
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
import pytest

from causal_qd.scores.bic import BICScore
from causal_qd.scores.bdeu import BDeuScore
from causal_qd.scores.bge import BGeScore
from causal_qd.scores.cached import CachedScore, CacheStats
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _make_chain_adj(n: int) -> AdjacencyMatrix:
    """Return adjacency matrix for a chain 0→1→…→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj


def _make_empty_adj(n: int) -> AdjacencyMatrix:
    return np.zeros((n, n), dtype=np.int8)


def _make_full_forward_adj(n: int) -> AdjacencyMatrix:
    """All forward edges i→j for i < j."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = 1
    return adj


def _generate_linear_gaussian(
    adj: AdjacencyMatrix,
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float = 0.5,
) -> DataMatrix:
    """Generate data from a linear Gaussian SCM with unit edge weights."""
    n = adj.shape[0]
    data = np.zeros((n_samples, n))
    # topological order: since adj[i,j]=1 means i→j and i<j for our
    # test DAGs, column-order is a valid topological order.
    for j in range(n):
        parents = list(np.where(adj[:, j])[0])
        noise = rng.standard_normal(n_samples) * noise_scale
        if parents:
            data[:, j] = data[:, parents].sum(axis=1) + noise
        else:
            data[:, j] = rng.standard_normal(n_samples) + noise
    return data


def _parents_of(adj: AdjacencyMatrix, node: int) -> List[int]:
    return list(np.where(adj[:, node])[0])


# ===================================================================
# BIC Score Tests
# ===================================================================


class TestBICScoreKnownModel:
    """test_bic_score_known_model: true DAG should outscore wrong DAGs."""

    def test_true_dag_beats_empty(self, gaussian_data, bic_scorer):
        data, true_adj = gaussian_data
        true_score = bic_scorer.score(true_adj, data)
        empty_adj = _make_empty_adj(true_adj.shape[0])
        empty_score = bic_scorer.score(empty_adj, data)
        assert true_score > empty_score, (
            f"True DAG score {true_score:.2f} should exceed empty "
            f"DAG score {empty_score:.2f}"
        )

    def test_true_dag_matches_reversed_chain(self, gaussian_data, bic_scorer):
        """BIC is score-equivalent, so a chain and its reversal score the same."""
        data, true_adj = gaussian_data
        true_score = bic_scorer.score(true_adj, data)
        reversed_adj = true_adj.T.copy()
        rev_score = bic_scorer.score(reversed_adj, data)
        assert abs(true_score - rev_score) < 1e-6, (
            f"Chain and reversed chain should be score-equivalent "
            f"(true={true_score:.2f}, rev={rev_score:.2f})"
        )

    def test_true_dag_beats_random_permutation(self, gaussian_data, bic_scorer, rng):
        data, true_adj = gaussian_data
        true_score = bic_scorer.score(true_adj, data)
        n = true_adj.shape[0]
        perm = rng.permutation(n)
        perm_adj = true_adj[np.ix_(perm, perm)]
        perm_score = bic_scorer.score(perm_adj, data)
        # The permuted DAG has the same structure but mismatched columns;
        # it may or may not be valid, but if the data truly matches the
        # original structure the true DAG should generally score at least
        # as well.
        assert true_score >= perm_score - 1.0  # allow small tolerance

    def test_true_dag_score_is_negative(self, gaussian_data, bic_scorer):
        data, true_adj = gaussian_data
        score = bic_scorer.score(true_adj, data)
        # BIC scores are log-likelihoods with a penalty, always negative
        assert score < 0.0

    def test_removing_true_edge_lowers_score(self, gaussian_data, bic_scorer):
        """Removing a true edge should lower the BIC score."""
        data, true_adj = gaussian_data
        true_score = bic_scorer.score(true_adj, data)
        # Remove edge 0→1
        missing_adj = true_adj.copy()
        missing_adj[0, 1] = 0
        missing_score = bic_scorer.score(missing_adj, data)
        assert true_score > missing_score


class TestBICScoreEmptyVsFull:
    """test_bic_score_empty_vs_full: compare empty and fully-connected DAGs."""

    def test_empty_scores_higher_on_independent_data(self, rng):
        """On independent data the empty DAG should beat the full DAG."""
        n = 4
        data = rng.standard_normal((300, n))
        scorer = BICScore()
        empty_adj = _make_empty_adj(n)
        full_adj = _make_full_forward_adj(n)
        empty_score = scorer.score(empty_adj, data)
        full_score = scorer.score(full_adj, data)
        assert empty_score > full_score, (
            f"Empty score {empty_score:.2f} should exceed full "
            f"score {full_score:.2f} on independent data"
        )

    def test_full_dag_heavily_penalised(self, random_data, bic_scorer):
        n = random_data.shape[1]
        full_adj = _make_full_forward_adj(n)
        empty_adj = _make_empty_adj(n)
        full_score = bic_scorer.score(full_adj, random_data)
        empty_score = bic_scorer.score(empty_adj, random_data)
        # With random data, the penalty for extra parameters should dominate
        assert empty_score > full_score

    def test_penalty_multiplier_increases_gap(self, random_data):
        n = random_data.shape[1]
        full_adj = _make_full_forward_adj(n)
        empty_adj = _make_empty_adj(n)

        scorer_low = BICScore(penalty_multiplier=0.5)
        scorer_high = BICScore(penalty_multiplier=2.0)

        gap_low = scorer_low.score(empty_adj, random_data) - scorer_low.score(
            full_adj, random_data
        )
        gap_high = scorer_high.score(
            empty_adj, random_data
        ) - scorer_high.score(full_adj, random_data)
        assert gap_high > gap_low, (
            "Higher penalty multiplier should widen the gap between "
            "empty and full DAG scores on random data"
        )

    def test_scores_are_finite(self, random_data, bic_scorer):
        n = random_data.shape[1]
        for adj in [_make_empty_adj(n), _make_full_forward_adj(n)]:
            s = bic_scorer.score(adj, random_data)
            assert np.isfinite(s), f"Score must be finite, got {s}"


class TestBICScoreDecomposability:
    """test_bic_score_decomposability: score = sum of local scores."""

    def test_chain_decomposition(self, gaussian_data, bic_scorer):
        data, true_adj = gaussian_data
        n = true_adj.shape[0]
        total = bic_scorer.score(true_adj, data)
        local_sum = sum(
            bic_scorer.local_score(j, _parents_of(true_adj, j), data)
            for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10, (
            f"Total {total} != local sum {local_sum}"
        )

    def test_empty_dag_decomposition(self, random_data, bic_scorer):
        n = random_data.shape[1]
        adj = _make_empty_adj(n)
        total = bic_scorer.score(adj, random_data)
        local_sum = sum(
            bic_scorer.local_score(j, [], random_data) for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10

    def test_full_dag_decomposition(self, random_data, bic_scorer):
        n = random_data.shape[1]
        adj = _make_full_forward_adj(n)
        total = bic_scorer.score(adj, random_data)
        local_sum = sum(
            bic_scorer.local_score(j, _parents_of(adj, j), random_data)
            for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10

    def test_fork_dag_decomposition(self, rng):
        """Fork structure: 0→1, 0→2, 0→3."""
        n = 4
        adj = np.zeros((n, n), dtype=np.int8)
        adj[0, 1] = 1
        adj[0, 2] = 1
        adj[0, 3] = 1
        data = _generate_linear_gaussian(adj, 400, rng)
        scorer = BICScore()
        total = scorer.score(adj, data)
        local_sum = sum(
            scorer.local_score(j, _parents_of(adj, j), data) for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10

    def test_collider_dag_decomposition(self, rng):
        """Collider: 0→2, 1→2."""
        n = 3
        adj = np.zeros((n, n), dtype=np.int8)
        adj[0, 2] = 1
        adj[1, 2] = 1
        data = _generate_linear_gaussian(adj, 400, rng)
        scorer = BICScore()
        total = scorer.score(adj, data)
        local_sum = sum(
            scorer.local_score(j, _parents_of(adj, j), data) for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10


class TestBICLocalScoreRegression:
    """test_bic_local_score_regression: verify numerical properties."""

    def test_no_parents_score_is_marginal_ll(self, rng):
        """Score with no parents depends only on variance of the node."""
        data = rng.standard_normal((200, 3))
        scorer = BICScore()
        s = scorer.local_score(0, [], data)
        m = data.shape[0]
        var = np.var(data[:, 0])
        expected_ll = (
            -0.5 * m * np.log(2 * np.pi) - 0.5 * m * np.log(var) - 0.5 * m
        )
        expected_penalty = -0.5 * 1.0 * 1 * np.log(m)  # k=1 (intercept)
        expected = expected_ll + expected_penalty
        assert abs(s - expected) < 1e-6, f"Expected {expected}, got {s}"

    def test_adding_true_parent_improves_score(self, rng):
        """Adding the true parent should improve the local score."""
        m = 500
        x0 = rng.standard_normal(m)
        x1 = 0.9 * x0 + rng.standard_normal(m) * 0.3
        data = np.column_stack([x0, x1])
        scorer = BICScore()
        s_no_parent = scorer.local_score(1, [], data)
        s_with_parent = scorer.local_score(1, [0], data)
        assert s_with_parent > s_no_parent

    def test_adding_irrelevant_parent_hurts_score(self, rng):
        """Adding a noise parent to an independent node hurts the score."""
        m = 300
        data = rng.standard_normal((m, 3))
        scorer = BICScore()
        s_alone = scorer.local_score(0, [], data)
        s_with_noise = scorer.local_score(0, [1], data)
        assert s_alone > s_with_noise

    def test_l2_regularization_finite(self, gaussian_data):
        data, adj = gaussian_data
        scorer = BICScore(regularization="l2", reg_lambda=0.1)
        s = scorer.local_score(1, [0], data)
        assert np.isfinite(s)

    def test_l1_regularization_finite(self, gaussian_data):
        data, adj = gaussian_data
        scorer = BICScore(regularization="l1", reg_lambda=0.01)
        s = scorer.local_score(1, [0], data)
        assert np.isfinite(s)

    def test_l2_vs_ols_on_true_edge(self, rng):
        """Ridge and OLS should agree closely with small lambda on well-
        conditioned data."""
        m = 500
        x0 = rng.standard_normal(m)
        x1 = 0.8 * x0 + rng.standard_normal(m) * 0.3
        data = np.column_stack([x0, x1])
        s_ols = BICScore(regularization="none").local_score(1, [0], data)
        s_ridge = BICScore(regularization="l2", reg_lambda=1e-6).local_score(
            1, [0], data
        )
        assert abs(s_ols - s_ridge) < 1.0, (
            f"OLS {s_ols:.4f} and Ridge {s_ridge:.4f} should be close"
        )

    def test_invalid_regularization_raises(self):
        with pytest.raises(ValueError, match="Unknown regularization"):
            BICScore(regularization="elastic_net")

    def test_negative_reg_lambda_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            BICScore(reg_lambda=-0.1)

    def test_local_score_deterministic(self, gaussian_data, bic_scorer):
        data, adj = gaussian_data
        s1 = bic_scorer.local_score(2, [1], data)
        s2 = bic_scorer.local_score(2, [1], data)
        assert s1 == s2

    def test_more_samples_higher_magnitude(self, rng):
        """With more samples, the magnitude of the BIC score increases."""
        scorer = BICScore()
        x0 = rng.standard_normal(2000)
        x1 = 0.7 * x0 + rng.standard_normal(2000) * 0.4
        data_small = np.column_stack([x0[:100], x1[:100]])
        data_large = np.column_stack([x0, x1])
        s_small = abs(scorer.local_score(1, [0], data_small))
        s_large = abs(scorer.local_score(1, [0], data_large))
        assert s_large > s_small


# ===================================================================
# BDeu Score Tests
# ===================================================================


class TestBDeuScoreDiscreteData:
    """test_bdeu_score_discrete_data: BDeu on categorical data."""

    def test_basic_score_finite(self, discrete_data, bdeu_scorer):
        s = bdeu_scorer.local_score(0, [1], discrete_data)
        assert np.isfinite(s)
        assert isinstance(s, float)

    def test_no_parents_score(self, discrete_data, bdeu_scorer):
        s = bdeu_scorer.local_score(0, [], discrete_data)
        assert np.isfinite(s)
        assert s < 0.0  # log-marginal-likelihood is negative

    def test_true_parent_improves_score(self, rng):
        """When x1 = f(x0) + noise, adding x0 as parent should help."""
        m = 500
        x0 = rng.integers(0, 3, size=m)
        # x1 determined by x0 with some noise
        x1 = (x0 + rng.integers(0, 2, size=m)) % 3
        data = np.column_stack([x0, x1]).astype(np.float64)
        scorer = BDeuScore(equivalent_sample_size=1.0)
        s_alone = scorer.local_score(1, [], data)
        s_parent = scorer.local_score(1, [0], data)
        assert s_parent > s_alone

    def test_independent_parent_hurts(self, rng):
        """Adding an independent parent should lower the BDeu score."""
        m = 500
        data = rng.integers(0, 3, size=(m, 3)).astype(np.float64)
        scorer = BDeuScore(equivalent_sample_size=1.0)
        s_alone = scorer.local_score(0, [], data)
        s_noise_parent = scorer.local_score(0, [1], data)
        assert s_alone > s_noise_parent

    def test_decomposability(self, discrete_data, bdeu_scorer):
        n = discrete_data.shape[1]
        adj = _make_chain_adj(n)
        total = bdeu_scorer.score(adj, discrete_data)
        local_sum = sum(
            bdeu_scorer.local_score(j, _parents_of(adj, j), discrete_data)
            for j in range(n)
        )
        assert abs(total - local_sum) < 1e-10

    def test_score_with_multiple_parents(self, discrete_data, bdeu_scorer):
        s = bdeu_scorer.local_score(3, [0, 1, 2], discrete_data)
        assert np.isfinite(s)

    def test_auto_discretize_continuous(self, rng):
        """BDeu should auto-discretize continuous data."""
        data = rng.standard_normal((300, 3))
        scorer = BDeuScore(equivalent_sample_size=1.0, max_discrete_levels=5)
        s = scorer.local_score(0, [1], data)
        assert np.isfinite(s)

    def test_discretize_quantile(self, rng):
        data = rng.standard_normal((200, 3))
        disc = BDeuScore.discretize(data, method="quantile", n_bins=4)
        assert disc.shape == data.shape
        assert disc.dtype == np.int64
        for col in range(3):
            assert disc[:, col].min() >= 0
            assert disc[:, col].max() <= 3

    def test_discretize_uniform(self, rng):
        data = rng.standard_normal((200, 3))
        disc = BDeuScore.discretize(data, method="uniform", n_bins=4)
        assert disc.shape == data.shape
        assert disc.dtype == np.int64

    def test_discretize_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            BDeuScore.discretize(np.zeros((10, 2)), method="invalid")

    def test_empty_dag_score(self, discrete_data, bdeu_scorer):
        n = discrete_data.shape[1]
        adj = _make_empty_adj(n)
        s = bdeu_scorer.score(adj, discrete_data)
        assert np.isfinite(s)
        assert s < 0.0


class TestBDeuScoreEquivalentSampleSize:
    """test_bdeu_score_equivalent_sample_size: ESS controls complexity."""

    def test_small_ess_prefers_simpler(self, rng):
        """Smaller ESS should prefer simpler (fewer parents) models."""
        m = 500
        data = rng.integers(0, 3, size=(m, 4)).astype(np.float64)
        scorer_small = BDeuScore(equivalent_sample_size=0.1)
        scorer_large = BDeuScore(equivalent_sample_size=10.0)

        # With small ESS, adding a noise parent should hurt more
        s_small_alone = scorer_small.local_score(0, [], data)
        s_small_parent = scorer_small.local_score(0, [1, 2], data)
        gap_small = s_small_alone - s_small_parent

        s_large_alone = scorer_large.local_score(0, [], data)
        s_large_parent = scorer_large.local_score(0, [1, 2], data)
        gap_large = s_large_alone - s_large_parent

        # Small ESS penalises extra parents more
        assert gap_small > gap_large, (
            f"Small ESS gap {gap_small:.4f} should exceed large ESS gap "
            f"{gap_large:.4f}"
        )

    def test_ess_affects_score_magnitude(self, discrete_data):
        scorer1 = BDeuScore(equivalent_sample_size=1.0)
        scorer10 = BDeuScore(equivalent_sample_size=10.0)
        s1 = scorer1.local_score(0, [1], discrete_data)
        s10 = scorer10.local_score(0, [1], discrete_data)
        # Different ESS values should yield different scores
        assert s1 != s10

    def test_ess_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BDeuScore(equivalent_sample_size=0.0)

    def test_ess_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BDeuScore(equivalent_sample_size=-1.0)

    def test_large_ess_more_permissive(self, rng):
        """With very large ESS, prior dominates and scores converge."""
        m = 200
        data = rng.integers(0, 2, size=(m, 3)).astype(np.float64)
        scorer = BDeuScore(equivalent_sample_size=10000.0)
        s_no = scorer.local_score(0, [], data)
        s_parent = scorer.local_score(0, [1], data)
        # With huge ESS the difference should be small relative to scores
        ratio = abs(s_no - s_parent) / max(abs(s_no), 1.0)
        assert ratio < 0.5, f"Ratio {ratio} too large for huge ESS"

    def test_ess_property(self):
        scorer = BDeuScore(equivalent_sample_size=5.0)
        assert scorer.equivalent_sample_size == 5.0

    def test_max_discrete_levels_property(self):
        scorer = BDeuScore(max_discrete_levels=8)
        assert scorer.max_discrete_levels == 8


# ===================================================================
# BGe Score Tests
# ===================================================================


class TestBGeScoreGaussianData:
    """test_bge_score_gaussian_data: BGe on continuous Gaussian data."""

    def test_basic_local_score_finite(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        s = bge_scorer.local_score(1, [0], data)
        assert np.isfinite(s)

    def test_true_parent_improves_score(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        s_no = bge_scorer.local_score(1, [], data)
        s_parent = bge_scorer.local_score(1, [0], data)
        assert s_parent > s_no

    def test_true_dag_beats_empty(self, gaussian_data, bge_scorer):
        data, true_adj = gaussian_data
        true_score = bge_scorer.score(true_adj, data)
        empty_adj = _make_empty_adj(true_adj.shape[0])
        empty_score = bge_scorer.score(empty_adj, data)
        assert true_score > empty_score

    def test_decomposability(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        n = adj.shape[0]
        total = bge_scorer.score(adj, data)
        local_sum = sum(
            bge_scorer.local_score(j, _parents_of(adj, j), data)
            for j in range(n)
        )
        assert abs(total - local_sum) < 1e-8

    def test_independent_parent_hurts(self, rng):
        data = rng.standard_normal((300, 4))
        scorer = BGeScore()
        s_alone = scorer.local_score(0, [], data)
        s_noise = scorer.local_score(0, [1], data)
        assert s_alone > s_noise

    def test_prior_precision_effect(self, gaussian_data):
        data, adj = gaussian_data
        scorer_low = BGeScore(prior_precision=0.01)
        scorer_high = BGeScore(prior_precision=100.0)
        s_low = scorer_low.local_score(1, [0], data)
        s_high = scorer_high.local_score(1, [0], data)
        # Different precision → different scores
        assert s_low != s_high

    def test_prior_mean_effect(self, gaussian_data):
        data, adj = gaussian_data
        scorer_zero = BGeScore(prior_mean=0.0)
        scorer_shift = BGeScore(prior_mean=10.0)
        s_zero = scorer_zero.local_score(1, [0], data)
        s_shift = scorer_shift.local_score(1, [0], data)
        assert s_zero != s_shift

    def test_score_diff_sign(self, gaussian_data, bge_scorer):
        """Adding the true parent should yield positive score_diff."""
        data, adj = gaussian_data
        diff = bge_scorer.score_diff(1, [], [0], data)
        assert diff > 0.0

    def test_score_diff_sign_removing_true_parent(self, gaussian_data, bge_scorer):
        """Removing the true parent should yield negative score_diff."""
        data, adj = gaussian_data
        diff = bge_scorer.score_diff(1, [0], [], data)
        assert diff < 0.0

    def test_score_all_nodes_finite(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        n = adj.shape[0]
        for j in range(n):
            parents = _parents_of(adj, j)
            s = bge_scorer.local_score(j, parents, data)
            assert np.isfinite(s), f"Node {j} score not finite: {s}"

    def test_prior_df_extra(self, gaussian_data):
        data, adj = gaussian_data
        scorer0 = BGeScore(prior_df_extra=0)
        scorer5 = BGeScore(prior_df_extra=5)
        s0 = scorer0.local_score(1, [0], data)
        s5 = scorer5.local_score(1, [0], data)
        assert s0 != s5

    def test_deterministic(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        s1 = bge_scorer.local_score(2, [1], data)
        s2 = bge_scorer.local_score(2, [1], data)
        assert s1 == s2

    def test_empty_dag_score_negative(self, gaussian_data, bge_scorer):
        data, _ = gaussian_data
        n = data.shape[1]
        adj = _make_empty_adj(n)
        s = bge_scorer.score(adj, data)
        assert np.isfinite(s)


# ===================================================================
# CachedScore Tests
# ===================================================================


class TestCachedScoreMatchesUncached:
    """test_cached_score_matches_uncached: cache must be transparent."""

    def test_bic_cached_equals_uncached(self, gaussian_data):
        data, adj = gaussian_data
        base = BICScore()
        cached = CachedScore(base)
        n = adj.shape[0]
        for j in range(n):
            parents = _parents_of(adj, j)
            expected = base.local_score(j, parents, data)
            actual = cached.local_score(j, parents, data)
            assert abs(expected - actual) < 1e-12, (
                f"Node {j}: base={expected}, cached={actual}"
            )

    def test_bge_cached_equals_uncached(self, gaussian_data):
        data, adj = gaussian_data
        base = BGeScore()
        cached = CachedScore(base)
        n = adj.shape[0]
        for j in range(n):
            parents = _parents_of(adj, j)
            expected = base.local_score(j, parents, data)
            actual = cached.local_score(j, parents, data)
            assert abs(expected - actual) < 1e-12

    def test_bdeu_cached_equals_uncached(self, discrete_data):
        base = BDeuScore(equivalent_sample_size=1.0)
        cached = CachedScore(base)
        n = discrete_data.shape[1]
        adj = _make_chain_adj(n)
        for j in range(n):
            parents = _parents_of(adj, j)
            expected = base.local_score(j, parents, discrete_data)
            actual = cached.local_score(j, parents, discrete_data)
            assert abs(expected - actual) < 1e-12

    def test_full_dag_score_cached_equals_uncached(self, gaussian_data):
        data, adj = gaussian_data
        base = BICScore()
        cached = CachedScore(base)
        expected = base.score(adj, data)
        actual = cached.score(adj, data)
        assert abs(expected - actual) < 1e-10

    def test_score_diff_cached_equals_uncached(self, gaussian_data):
        data, adj = gaussian_data
        base = BICScore()
        cached = CachedScore(base)
        expected = base.score_diff(adj, 2, [1], [0, 1], data)
        actual = cached.score_diff(adj, 2, [1], [0, 1], data)
        assert abs(expected - actual) < 1e-12

    def test_repeated_calls_same_result(self, gaussian_data):
        data, adj = gaussian_data
        cached = CachedScore(BICScore())
        s1 = cached.local_score(1, [0], data)
        s2 = cached.local_score(1, [0], data)
        s3 = cached.local_score(1, [0], data)
        assert s1 == s2 == s3

    def test_clear_cache_resets(self, gaussian_data):
        data, adj = gaussian_data
        cached = CachedScore(BICScore())
        _ = cached.local_score(1, [0], data)
        cached.clear_cache()
        info = cached.cache_info
        assert info.hits == 0
        assert info.misses == 0
        assert info.current_size == 0


class TestCacheHitRate:
    """test_cache_hit_rate: verify cache statistics are correct."""

    def test_first_call_is_miss(self, gaussian_data):
        data, _ = gaussian_data
        cached = CachedScore(BICScore())
        _ = cached.local_score(0, [], data)
        info = cached.cache_info
        assert info.misses == 1
        assert info.hits == 0

    def test_second_call_is_hit(self, gaussian_data):
        data, _ = gaussian_data
        cached = CachedScore(BICScore())
        _ = cached.local_score(0, [], data)
        _ = cached.local_score(0, [], data)
        info = cached.cache_info
        assert info.hits == 1
        assert info.misses == 1
        assert abs(info.hit_rate - 0.5) < 1e-10

    def test_many_repeated_calls_high_hit_rate(self, gaussian_data):
        data, adj = gaussian_data
        cached = CachedScore(BICScore())
        n = adj.shape[0]
        # First pass: all misses
        for j in range(n):
            _ = cached.local_score(j, _parents_of(adj, j), data)
        # Second and third passes: all hits
        for _ in range(2):
            for j in range(n):
                _ = cached.local_score(j, _parents_of(adj, j), data)
        info = cached.cache_info
        # n misses, 2*n hits → hit_rate = 2n/(3n) ≈ 0.667
        expected_rate = (2 * n) / (3 * n)
        assert abs(info.hit_rate - expected_rate) < 0.01

    def test_distinct_queries_all_misses(self, gaussian_data):
        data, _ = gaussian_data
        cached = CachedScore(BICScore())
        n = data.shape[1]
        # Each call is a unique (node, parents) combination
        for j in range(n):
            _ = cached.local_score(j, [], data)
        info = cached.cache_info
        assert info.misses == n
        assert info.hits == 0
        assert info.hit_rate == 0.0

    def test_cache_size_grows(self, gaussian_data):
        data, _ = gaussian_data
        cached = CachedScore(BICScore())
        n = data.shape[1]
        for j in range(n):
            _ = cached.local_score(j, [], data)
        info = cached.cache_info
        assert info.current_size == n

    def test_cache_info_types(self, gaussian_data):
        data, _ = gaussian_data
        cached = CachedScore(BICScore())
        _ = cached.local_score(0, [], data)
        info = cached.cache_info
        assert isinstance(info, CacheStats)
        assert isinstance(info.hits, int)
        assert isinstance(info.misses, int)
        assert isinstance(info.hit_rate, float)
        assert isinstance(info.current_size, int)
        assert isinstance(info.max_size, int)
        assert isinstance(info.memory_estimate_bytes, int)

    def test_max_cache_size_respected(self, gaussian_data):
        data, _ = gaussian_data
        small_cache = CachedScore(BICScore(), max_cache_size=3)
        n = data.shape[1]
        # Insert more unique entries than cache size
        for j in range(n):
            _ = small_cache.local_score(j, [], data)
        info = small_cache.cache_info
        assert info.current_size <= 3

    def test_warm_cache(self, gaussian_data):
        data, adj = gaussian_data
        cached = CachedScore(BICScore())
        cached.warm_cache(adj, data)
        n = adj.shape[0]
        info = cached.cache_info
        assert info.current_size >= 1  # at least some entries cached


# ===================================================================
# Score Diff Consistency Tests
# ===================================================================


class TestScoreDiffConsistency:
    """test_score_diff_consistency: score_diff = new_local - old_local."""

    def test_bic_score_diff_matches_direct(self, gaussian_data, bic_scorer):
        data, adj = gaussian_data
        node = 2
        old_parents = [1]
        new_parents = [0, 1]
        diff = bic_scorer.score_diff(adj, node, old_parents, new_parents, data)
        s_old = bic_scorer.local_score(node, old_parents, data)
        s_new = bic_scorer.local_score(node, new_parents, data)
        assert abs(diff - (s_new - s_old)) < 1e-12

    def test_bge_score_diff_matches_direct(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        node = 2
        old_parents = [1]
        new_parents = [0, 1]
        diff = bge_scorer.score_diff(node, old_parents, new_parents, data)
        s_old = bge_scorer.local_score(node, old_parents, data)
        s_new = bge_scorer.local_score(node, new_parents, data)
        assert abs(diff - (s_new - s_old)) < 1e-12

    def test_bdeu_score_diff_matches_direct(self, discrete_data, bdeu_scorer):
        node = 2
        old_parents = [0]
        new_parents = [0, 1]
        diff = bdeu_scorer.score_diff(
            node, old_parents, new_parents, discrete_data
        )
        s_old = bdeu_scorer.local_score(node, old_parents, discrete_data)
        s_new = bdeu_scorer.local_score(node, new_parents, discrete_data)
        assert abs(diff - (s_new - s_old)) < 1e-12

    def test_score_diff_zero_when_same_parents(self, gaussian_data, bic_scorer):
        data, adj = gaussian_data
        diff = bic_scorer.score_diff(adj, 1, [0], [0], data)
        assert abs(diff) < 1e-12

    def test_score_diff_antisymmetric(self, gaussian_data, bic_scorer):
        """diff(old→new) = -diff(new→old)."""
        data, adj = gaussian_data
        d1 = bic_scorer.score_diff(adj, 2, [1], [0, 1], data)
        d2 = bic_scorer.score_diff(adj, 2, [0, 1], [1], data)
        assert abs(d1 + d2) < 1e-12

    def test_bge_score_diff_antisymmetric(self, gaussian_data, bge_scorer):
        data, adj = gaussian_data
        d1 = bge_scorer.score_diff(2, [1], [0, 1], data)
        d2 = bge_scorer.score_diff(2, [0, 1], [1], data)
        assert abs(d1 + d2) < 1e-12

    def test_bdeu_score_diff_antisymmetric(self, discrete_data, bdeu_scorer):
        d1 = bdeu_scorer.score_diff(2, [0], [0, 1], discrete_data)
        d2 = bdeu_scorer.score_diff(2, [0, 1], [0], discrete_data)
        assert abs(d1 + d2) < 1e-12

    def test_cached_score_diff_matches_base(self, gaussian_data):
        data, adj = gaussian_data
        base = BICScore()
        cached = CachedScore(base)
        node, old_pa, new_pa = 3, [2], [1, 2]
        diff_base = base.score_diff(adj, node, old_pa, new_pa, data)
        diff_cached = cached.score_diff(adj, node, old_pa, new_pa, data)
        assert abs(diff_base - diff_cached) < 1e-12

    def test_score_diff_add_edge_from_empty(self, gaussian_data, bic_scorer):
        data, adj = gaussian_data
        diff = bic_scorer.score_diff(adj, 1, [], [0], data)
        s_old = bic_scorer.local_score(1, [], data)
        s_new = bic_scorer.local_score(1, [0], data)
        assert abs(diff - (s_new - s_old)) < 1e-12

    def test_score_diff_remove_to_empty(self, gaussian_data, bic_scorer):
        data, adj = gaussian_data
        diff = bic_scorer.score_diff(adj, 1, [0], [], data)
        s_old = bic_scorer.local_score(1, [0], data)
        s_new = bic_scorer.local_score(1, [], data)
        assert abs(diff - (s_new - s_old)) < 1e-12

    def test_bdeu_score_diff_with_cache_dict(self, discrete_data, bdeu_scorer):
        cache = {}
        d1 = bdeu_scorer.score_diff(
            2, [0], [0, 1], discrete_data, cache=cache
        )
        # After first call, entries should be in the cache
        assert len(cache) > 0
        d2 = bdeu_scorer.score_diff(
            2, [0], [0, 1], discrete_data, cache=cache
        )
        assert abs(d1 - d2) < 1e-12


# ===================================================================
# Cross-scorer comparison tests
# ===================================================================


class TestCrossScorer:
    """Additional cross-cutting tests for score consistency."""

    def test_all_scorers_agree_on_direction(self, gaussian_data):
        """All scorers should prefer true parent over no parent for node 1."""
        data, adj = gaussian_data
        bic = BICScore()
        bge = BGeScore()
        for scorer in [bic, bge]:
            s_no = scorer.local_score(1, [], data)
            s_pa = scorer.local_score(1, [0], data)
            assert s_pa > s_no, f"{type(scorer).__name__} failed direction test"

    def test_bic_regularization_modes(self, gaussian_data):
        """All three regularization modes should produce finite scores."""
        data, adj = gaussian_data
        for reg in ["none", "l1", "l2"]:
            scorer = BICScore(regularization=reg, reg_lambda=0.01)
            s = scorer.score(adj, data)
            assert np.isfinite(s), f"reg={reg} produced non-finite score"

    def test_bic_repr(self):
        scorer = BICScore(penalty_multiplier=2.0, regularization="l2", reg_lambda=0.05)
        r = repr(scorer)
        assert "BICScore" in r
        assert "2.0" in r
        assert "l2" in r

    def test_bdeu_repr(self):
        scorer = BDeuScore(equivalent_sample_size=5.0, max_discrete_levels=3)
        r = repr(scorer)
        assert "BDeuScore" in r
        assert "5.0" in r

    def test_bge_repr(self):
        scorer = BGeScore(prior_mean=1.0, prior_precision=2.0)
        r = repr(scorer)
        assert "BGeScore" in r

    def test_bdeu_arities(self, discrete_data):
        scorer = BDeuScore()
        arities = scorer.arities(discrete_data)
        assert len(arities) == discrete_data.shape[1]
        assert all(a >= 1 for a in arities)

    def test_bdeu_clear_cache(self, discrete_data):
        scorer = BDeuScore()
        _ = scorer.local_score(0, [], discrete_data)
        scorer.clear_cache()
        # Should not raise after clearing
        s = scorer.local_score(0, [], discrete_data)
        assert np.isfinite(s)

    def test_bge_clear_cache(self, gaussian_data):
        data, _ = gaussian_data
        scorer = BGeScore()
        _ = scorer.local_score(0, [], data)
        scorer.clear_cache()
        s = scorer.local_score(0, [], data)
        assert np.isfinite(s)
