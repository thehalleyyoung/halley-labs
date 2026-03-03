"""Tests for ParallelTempering — Metropolis-coupled MCMC for Order MCMC."""
from __future__ import annotations

import numpy as np
import pytest

from causal_qd.sampling.parallel_tempering import ParallelTempering, TemperingResult
from causal_qd.scores.bic import BICScore


# ===================================================================
# Helpers
# ===================================================================

def _simple_data(n_samples: int = 300, n_vars: int = 3, seed: int = 42):
    """Generate data from a simple chain 0→1→2."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_samples, n_vars))
    data[:, 0] = rng.standard_normal(n_samples)
    for i in range(1, n_vars):
        data[:, i] = 0.8 * data[:, i - 1] + rng.standard_normal(n_samples) * 0.5
    return data


# ===================================================================
# Temperature ladder tests
# ===================================================================

class TestTemperatureLadder:
    def test_geometric_ladder(self):
        pt = ParallelTempering(BICScore(), n_chains=4, ladder="geometric", max_temp=10.0)
        assert len(pt.temperatures) == 4
        assert abs(pt.temperatures[0] - 1.0) < 1e-10
        assert abs(pt.temperatures[-1] - 10.0) < 1e-10
        # Geometric: each ratio should be constant
        ratios = [pt.temperatures[i + 1] / pt.temperatures[i] for i in range(3)]
        assert abs(ratios[0] - ratios[1]) < 1e-10

    def test_linear_ladder(self):
        pt = ParallelTempering(BICScore(), n_chains=4, ladder="linear", max_temp=10.0)
        assert len(pt.temperatures) == 4
        assert abs(pt.temperatures[0] - 1.0) < 1e-10
        assert abs(pt.temperatures[-1] - 10.0) < 1e-10
        # Linear: differences should be constant
        diffs = [pt.temperatures[i + 1] - pt.temperatures[i] for i in range(3)]
        assert abs(diffs[0] - diffs[1]) < 1e-10

    def test_cold_chain_is_t1(self):
        pt = ParallelTempering(BICScore(), n_chains=6, ladder="geometric")
        assert abs(pt.temperatures[0] - 1.0) < 1e-10

    def test_explicit_temperatures(self):
        temps = [1.0, 2.0, 5.0, 20.0]
        pt = ParallelTempering(BICScore(), temperatures=temps)
        assert pt.temperatures == temps
        assert pt.n_chains == 4

    def test_single_chain(self):
        pt = ParallelTempering(BICScore(), n_chains=1)
        assert len(pt.temperatures) == 1
        assert pt.temperatures[0] == 1.0


# ===================================================================
# Swap acceptance tests
# ===================================================================

class TestSwapAcceptance:
    def test_swap_always_accepted_when_hot_has_better_score(self):
        """If hot chain has better score, swap should always be accepted."""
        rng = np.random.default_rng(0)
        # β_cold > β_hot, score_hot > score_cold
        # log_alpha = (β_cold - β_hot)(score_hot - score_cold) > 0
        accepted = ParallelTempering._propose_swap(
            score_i=-100.0, score_j=-50.0,  # j has better score
            beta_i=1.0, beta_j=0.1,  # i is colder
            rng=rng,
        )
        assert accepted is True

    def test_swap_probabilistic_when_cold_has_better_score(self):
        """When cold chain has better score, swap may be rejected."""
        rng = np.random.default_rng(0)
        # β_cold > β_hot, score_cold > score_hot
        # log_alpha = (β_cold - β_hot)(score_hot - score_cold) < 0
        # Use small difference so some swaps are accepted
        n_accept = 0
        for i in range(1000):
            rng_i = np.random.default_rng(i)
            accepted = ParallelTempering._propose_swap(
                score_i=-50.0, score_j=-51.0,
                beta_i=1.0, beta_j=0.5,
                rng=rng_i,
            )
            n_accept += int(accepted)
        # Should accept some but not all
        assert 0 < n_accept < 1000


# ===================================================================
# Edge probability tests
# ===================================================================

class TestEdgeProbabilities:
    def test_edge_probabilities_shape(self):
        data = _simple_data(n_vars=3)
        pt = ParallelTempering(BICScore(), n_chains=2, max_temp=2.0)
        result = pt.run(data, n_samples=20, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        probs = result.edge_probabilities
        assert probs.shape == (3, 3)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_edge_probabilities_from_method(self):
        data = _simple_data(n_vars=3)
        pt = ParallelTempering(BICScore(), n_chains=2, max_temp=2.0)
        result = pt.run(data, n_samples=20, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        probs = pt.edge_probabilities(result)
        assert np.allclose(probs, result.edge_probabilities)

    def test_diagonal_zero(self):
        data = _simple_data(n_vars=3)
        pt = ParallelTempering(BICScore(), n_chains=2, max_temp=2.0)
        result = pt.run(data, n_samples=30, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        # No self-loops in DAGs
        assert np.allclose(np.diag(result.edge_probabilities), 0.0)


# ===================================================================
# Run and diagnostics tests
# ===================================================================

class TestRun:
    def test_run_produces_samples(self):
        data = _simple_data()
        pt = ParallelTempering(BICScore(), n_chains=2, max_temp=2.0)
        result = pt.run(data, n_samples=30, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        assert len(result.samples) == 30

    def test_result_has_swap_rates(self):
        data = _simple_data()
        pt = ParallelTempering(BICScore(), n_chains=3, max_temp=5.0)
        result = pt.run(data, n_samples=20, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        assert len(result.swap_acceptance_rates) == 2  # n_chains - 1

    def test_chain_log_scores(self):
        data = _simple_data()
        pt = ParallelTempering(BICScore(), n_chains=3, max_temp=5.0)
        result = pt.run(data, n_samples=20, burnin=10, max_parents=2,
                        rng=np.random.default_rng(42))
        assert len(result.chain_log_scores) == 3
        assert all(len(t) == 30 for t in result.chain_log_scores)  # burnin + n_samples

    def test_diagnostics_keys(self):
        data = _simple_data()
        pt = ParallelTempering(BICScore(), n_chains=3, max_temp=5.0)
        result = pt.run(data, n_samples=50, burnin=20, max_parents=2,
                        rng=np.random.default_rng(42))
        diag = result.diagnostics
        assert "r_hat" in diag
        assert "ess_per_chain" in diag
        assert "n_chains" in diag
        assert "temperatures" in diag

    def test_with_two_node_graph(self):
        """Minimal graph: 2 variables."""
        rng = np.random.default_rng(0)
        data = np.zeros((200, 2))
        data[:, 0] = rng.standard_normal(200)
        data[:, 1] = 0.9 * data[:, 0] + rng.standard_normal(200) * 0.3
        pt = ParallelTempering(BICScore(), n_chains=2, max_temp=2.0)
        result = pt.run(data, n_samples=30, burnin=10, max_parents=1,
                        rng=np.random.default_rng(42))
        assert result.edge_probabilities.shape == (2, 2)


# ===================================================================
# Temperature adaptation tests
# ===================================================================

class TestAdaptation:
    def test_adapt_increases_spacing_when_rate_too_high(self):
        temps = [1.0, 2.0, 4.0]
        rates = [0.8, 0.8]  # Too high
        new_temps = ParallelTempering._adapt_temperatures(temps, rates)
        # Spacing should increase
        assert new_temps[1] / new_temps[0] > temps[1] / temps[0]

    def test_adapt_decreases_spacing_when_rate_too_low(self):
        temps = [1.0, 2.0, 4.0]
        rates = [0.05, 0.05]  # Too low
        new_temps = ParallelTempering._adapt_temperatures(temps, rates)
        # Spacing should decrease
        assert new_temps[1] / new_temps[0] < temps[1] / temps[0]

    def test_adapt_keeps_cold_chain_at_one(self):
        temps = [1.0, 3.0, 9.0]
        rates = [0.5, 0.1]
        new_temps = ParallelTempering._adapt_temperatures(temps, rates)
        assert abs(new_temps[0] - 1.0) < 1e-10

    def test_single_chain_no_adaptation(self):
        temps = [1.0]
        new_temps = ParallelTempering._adapt_temperatures(temps, [])
        assert new_temps == [1.0]
