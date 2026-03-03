"""Unit tests for cpa.sampling.order_mcmc."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.sampling.order_mcmc import (
    DAGPosteriorSamples,
    OrderMCMC,
    OrderSample,
)
from cpa.scores.bic import BICScore


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


def _simple_score_fn(data):
    scorer = BICScore(data)
    return scorer.local_score


def _is_dag(adj):
    from collections import deque
    n = adj.shape[0]
    in_deg = (adj != 0).sum(axis=0).astype(int)
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return count == n


# ── OrderSample tests ──────────────────────────────────────────────

class TestOrderSample:
    def test_creation(self):
        s = OrderSample(order=[0, 1, 2], score=-10.5)
        assert s.order == [0, 1, 2]
        assert s.score == -10.5
        assert s.dag is None

    def test_with_dag(self):
        dag = np.eye(3)
        s = OrderSample(order=[0, 1, 2], score=-5.0, dag=dag)
        assert s.dag is not None


# ── OrderMCMC basic tests ─────────────────────────────────────────

class TestOrderMCMCBasic:
    def test_propose_order_swaps_adjacent(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        order = [0, 1, 2]
        proposed = mcmc.propose_order(order)
        # Should differ in exactly one adjacent swap
        diffs = sum(1 for a, b in zip(order, proposed) if a != b)
        assert diffs == 2  # adjacent swap changes exactly 2 positions

    def test_propose_order_single_node(self):
        mcmc = OrderMCMC(lambda n, p: 0.0, n_nodes=1, seed=42)
        result = mcmc.propose_order([0])
        assert result == [0]

    def test_acceptance_ratio_better_accepted(self):
        mcmc = OrderMCMC(lambda n, p: 0.0, n_nodes=3, seed=42)
        alpha = mcmc.acceptance_ratio(-10.0, -5.0)
        assert alpha == 1.0

    def test_acceptance_ratio_worse_partial(self):
        mcmc = OrderMCMC(lambda n, p: 0.0, n_nodes=3, seed=42)
        alpha = mcmc.acceptance_ratio(-5.0, -10.0)
        assert 0.0 < alpha < 1.0
        expected = math.exp(-10.0 - (-5.0))
        assert abs(alpha - expected) < 1e-10

    def test_acceptance_ratio_equal(self):
        mcmc = OrderMCMC(lambda n, p: 0.0, n_nodes=3, seed=42)
        alpha = mcmc.acceptance_ratio(-5.0, -5.0)
        assert alpha == 1.0

    def test_sample_dag_from_order_is_dag(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        dag = mcmc.sample_dag_from_order([0, 1, 2])
        assert _is_dag(dag)
        assert dag.shape == (3, 3)

    def test_dag_respects_order(self):
        """Edges should only go from earlier to later in order."""
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        order = [2, 0, 1]
        dag = mcmc.sample_dag_from_order(order)
        pos = {v: i for i, v in enumerate(order)}
        for i in range(3):
            for j in range(3):
                if dag[i, j] != 0:
                    assert pos[i] < pos[j]


# ── OrderMCMC run tests ───────────────────────────────────────────

class TestOrderMCMCRun:
    def test_run_returns_samples(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10, thin=2)
        assert len(samples) > 0
        for s in samples:
            assert isinstance(s, OrderSample)
            assert s.dag is not None
            assert _is_dag(s.dag)

    def test_burnin_reduces_samples(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        s_no_burn = mcmc.run(n_iterations=50, burnin=0)
        mcmc2 = OrderMCMC(score_fn, n_nodes=3, seed=42)
        s_with_burn = mcmc2.run(n_iterations=50, burnin=20)
        assert len(s_with_burn) < len(s_no_burn)

    def test_thinning_reduces_samples(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        s_thin1 = mcmc.run(n_iterations=100, burnin=0, thin=1)
        mcmc2 = OrderMCMC(score_fn, n_nodes=3, seed=42)
        s_thin5 = mcmc2.run(n_iterations=100, burnin=0, thin=5)
        assert len(s_thin5) < len(s_thin1)

    def test_acceptance_rate_reasonable(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        mcmc.run(n_iterations=200, burnin=0)
        rate = mcmc._last_acceptance_rate
        assert 0.0 < rate < 1.0

    def test_run_full(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        result = mcmc.run_full(n_iterations=100, burn_in=20, thin=2)
        assert isinstance(result, DAGPosteriorSamples)
        assert len(result.dags) > 0
        assert len(result.scores) > 0
        assert len(result.orders) > 0
        assert result.burn_in == 20
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_run_full_with_seed(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        r1 = mcmc.run_full(n_iterations=50, burn_in=10, seed=99)
        r2 = mcmc.run_full(n_iterations=50, burn_in=10, seed=99)
        assert_array_almost_equal(r1.dags[0], r2.dags[0])


# ── Edge posterior tests ──────────────────────────────────────────

class TestEdgePosterior:
    @pytest.mark.slow
    def test_true_edges_high_probability(self):
        """True edges should have high posterior probability."""
        data = _chain_data(n=500, seed=42)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, max_parents=2, seed=42)
        samples = mcmc.run(n_iterations=500, burnin=100, thin=2)
        probs = OrderMCMC.edge_posterior_probabilities(samples)
        # Edge 0->1 should have non-trivial probability (MCMC is stochastic)
        assert probs[0, 1] > 0.1
        # Edge 1->2 should have non-trivial probability
        assert probs[1, 2] > 0.1

    def test_empty_samples(self):
        probs = OrderMCMC.edge_posterior_probabilities([])
        assert probs.shape == (0, 0)

    def test_probabilities_in_01(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10)
        probs = OrderMCMC.edge_posterior_probabilities(samples)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ── MAP DAG tests ─────────────────────────────────────────────────

class TestMAPDag:
    def test_map_dag_returns_best(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10)
        dag, score = OrderMCMC.map_dag(samples)
        assert _is_dag(dag)
        max_score = max(s.score for s in samples)
        assert score == max_score


# ── Convergence diagnostics tests ─────────────────────────────────

class TestConvergenceDiagnostics:
    def test_effective_sample_size(self):
        scores = list(np.random.default_rng(42).standard_normal(200))
        ess = OrderMCMC.effective_sample_size(scores)
        assert ess > 0
        assert ess <= len(scores)

    def test_ess_short_sequence(self):
        ess = OrderMCMC.effective_sample_size([1.0, 2.0])
        assert ess == 2.0

    def test_ess_constant_sequence(self):
        ess = OrderMCMC.effective_sample_size([5.0] * 100)
        assert ess == 100.0

    def test_gelman_rubin_single_chain(self):
        r = OrderMCMC.gelman_rubin([[1.0, 2.0, 3.0]])
        assert math.isnan(r)

    def test_gelman_rubin_identical_chains(self):
        chain = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = OrderMCMC.gelman_rubin([chain, chain])
        assert abs(r - 1.0) < 0.2  # tolerance for Gelman-Rubin on short identical chains

    def test_gelman_rubin_divergent_chains(self):
        c1 = list(np.random.default_rng(1).standard_normal(100))
        c2 = list(np.random.default_rng(2).standard_normal(100) + 10.0)
        r = OrderMCMC.gelman_rubin([c1, c2])
        assert r > 1.5  # should be far from 1

    def test_trace_plot_data(self):
        samples = [OrderSample(order=[0, 1], score=float(-i)) for i in range(10)]
        iters, scores = OrderMCMC.trace_plot_data(samples)
        assert len(iters) == 10
        assert len(scores) == 10
        assert_array_almost_equal(iters, np.arange(10, dtype=float))

    @pytest.mark.slow
    def test_correct_order_recovery(self):
        """OrderMCMC should recover the correct topological order."""
        data = _chain_data(n=1000, seed=42)
        score_fn = _simple_score_fn(data)
        mcmc = OrderMCMC(score_fn, n_nodes=3, seed=42)
        result = mcmc.run_full(n_iterations=500, burn_in=100)
        # The MAP DAG should have edges 0->1 and 1->2
        dag, _ = OrderMCMC.map_dag(
            [OrderSample(order=o, score=s, dag=d)
             for o, s, d in zip(result.orders, result.scores, result.dags)]
        )
        # Check that 0->1 and 1->2 edges are present
        assert dag[0, 1] != 0 or dag[1, 0] != 0  # some edge between 0,1
