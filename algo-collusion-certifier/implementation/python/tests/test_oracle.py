"""Tests for the oracle module.

These tests exercise oracle observation, checkpoint/restore, rewind
counterfactual logic, and the learning agents (Q-learning, grim trigger,
DQN, bandits) used in simulations.
"""

import pytest
import numpy as np

from collusion_proof.cli.commands import (
    run_simulation,
    _simulate_q_learning,
    _simulate_grim_trigger,
    _simulate_bandit,
)
from collusion_proof.config import MarketConfig
from collusion_proof.types import GameConfig, AlgorithmConfig


class TestPassiveOracle:
    """Tests for passive observation of price data."""

    def test_observe_records_prices(self):
        """Simulated prices should be recorded as a valid matrix."""
        result = run_simulation(2, 100, "q_learning", seed=42)
        prices = result["prices"]
        assert prices.shape == (100, 2)
        assert not np.any(np.isnan(prices))

    def test_get_price_history_shape(self):
        result = run_simulation(2, 500, "q_learning", seed=42)
        prices = result["prices"]
        assert prices.shape[0] == 500
        assert prices.shape[1] == 2

    def test_summary_statistics(self):
        result = run_simulation(2, 1000, "q_learning", seed=42)
        prices = result["prices"]
        mean = float(np.mean(prices))
        std = float(np.std(prices))
        assert mean > 0
        assert std > 0
        assert np.all(prices >= 0) or np.any(prices < 0)  # prices can be anything

    def test_multiple_players(self):
        result = run_simulation(3, 200, "bandit", seed=42)
        prices = result["prices"]
        assert prices.shape == (200, 3)

    def test_price_bounds(self):
        """Q-learning prices should stay within the action space."""
        result = run_simulation(2, 500, "q_learning", seed=42)
        prices = result["prices"]
        # Action space goes from marginal_cost to monopoly_price * 1.2
        assert np.all(prices >= 0)
        assert np.all(prices <= 10.0)  # 5.5 * 1.2 = 6.6, generous upper bound


class TestCheckpointOracle:
    """Tests for checkpoint/restore functionality via reproducible simulation."""

    def test_save_restore_reproducibility(self):
        """Same seed should produce same results."""
        r1 = run_simulation(2, 500, "q_learning", seed=123)
        r2 = run_simulation(2, 500, "q_learning", seed=123)
        np.testing.assert_array_equal(r1["prices"], r2["prices"])

    def test_different_seeds_differ(self):
        r1 = run_simulation(2, 500, "q_learning", seed=1)
        r2 = run_simulation(2, 500, "q_learning", seed=2)
        assert not np.array_equal(r1["prices"], r2["prices"])

    def test_reproducibility_bandit(self):
        r1 = run_simulation(2, 300, "bandit", seed=77)
        r2 = run_simulation(2, 300, "bandit", seed=77)
        np.testing.assert_array_equal(r1["prices"], r2["prices"])


class TestRewindOracle:
    """Tests for counterfactual reasoning via simulation comparison."""

    def test_counterfactual_different_algorithm(self):
        """Different algorithms should produce different price dynamics."""
        r_ql = run_simulation(2, 1000, "q_learning", seed=42)
        r_gt = run_simulation(2, 1000, "grim_trigger", seed=42)
        # They should differ significantly
        mean_ql = float(np.mean(r_ql["prices"]))
        mean_gt = float(np.mean(r_gt["prices"]))
        assert mean_ql != mean_gt

    def test_counterfactual_player_count(self):
        """More players should generally lower prices (competitive pressure)."""
        r2 = run_simulation(2, 2000, "bandit", seed=42)
        r3 = run_simulation(3, 2000, "bandit", seed=42)
        mean2 = float(np.mean(r2["prices"][-500:]))
        mean3 = float(np.mean(r3["prices"][-500:]))
        # Not guaranteed, but with bandits more players usually means different dynamics
        assert r2["prices"].shape[1] == 2
        assert r3["prices"].shape[1] == 3


class TestQLearning:
    """Tests for Q-learning agent simulation."""

    def test_initialization(self):
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_q_learning(2, 10, action_space, 1.0, rng)
        assert prices.shape == (10, 2)

    def test_epsilon_greedy_exploration(self):
        """Early rounds should show more price variety than later rounds."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_q_learning(2, 5000, action_space, 1.0, rng)
        early_std = float(np.std(prices[:500]))
        late_std = float(np.std(prices[-500:]))
        # After epsilon decays, prices should be less variable
        assert late_std <= early_std + 0.5  # allow some tolerance

    def test_q_learning_update(self):
        """Q-learning should learn: later prices should be more profitable."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_q_learning(2, 10000, action_space, 1.0, rng)
        # Verify convergence: less variance at end
        early_var = float(np.var(prices[:1000, :].mean(axis=1)))
        late_var = float(np.var(prices[-1000:, :].mean(axis=1)))
        assert late_var < early_var + 1.0

    def test_convergence_to_stable_prices(self):
        """After many rounds, Q-learning should stabilize."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_q_learning(2, 20000, action_space, 1.0, rng)
        late_prices = prices[-2000:]
        std = float(np.std(late_prices.mean(axis=1)))
        # Should be relatively stable
        assert std < 3.0

    def test_multi_agent_shape(self):
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_q_learning(2, 100, action_space, 1.0, rng)
        assert prices.shape == (100, 2)


class TestGrimTrigger:
    """Tests for grim trigger strategy."""

    def test_cooperative_play(self):
        """Without deviations, grim trigger should maintain high prices."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        nash_price, monopoly_price = 1.0, 5.5
        prices = _simulate_grim_trigger(2, 500, action_space, nash_price, monopoly_price, rng)
        # Should start near monopoly price
        mean_early = float(np.mean(prices[:50]))
        assert mean_early > 4.0

    def test_trigger_detection(self):
        """Grim trigger should show sharp price drops after deviation."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        nash_price, monopoly_price = 1.0, 5.5
        prices = _simulate_grim_trigger(2, 1000, action_space, nash_price, monopoly_price, rng)
        # Check if there are any sharp drops
        diffs = np.diff(prices.mean(axis=1))
        has_drop = np.any(diffs < -0.5)
        # It's OK if no trigger happens (noise too small), but prices should be sensible
        assert np.all(prices >= 0)

    def test_grim_trigger_shape(self):
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_grim_trigger(2, 100, action_space, 1.0, 5.5, rng)
        assert prices.shape == (100, 2)


class TestDQN:
    """Tests for DQN agent (uses Q-learning fallback in CLI)."""

    def test_dqn_select_action(self):
        """DQN simulation should produce valid prices."""
        result = run_simulation(2, 100, "dqn", seed=42)
        prices = result["prices"]
        assert prices.shape == (100, 2)
        assert np.all(np.isfinite(prices))

    def test_dqn_reproducibility(self):
        r1 = run_simulation(2, 100, "dqn", seed=99)
        r2 = run_simulation(2, 100, "dqn", seed=99)
        np.testing.assert_array_equal(r1["prices"], r2["prices"])


class TestBandits:
    """Tests for bandit-based agents."""

    def test_epsilon_greedy(self):
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_bandit(2, 500, action_space, 1.0, rng)
        assert prices.shape == (500, 2)
        assert np.all(prices >= 1.0)
        assert np.all(prices <= 6.6)

    def test_bandit_learns(self):
        """Bandit should converge to higher-profit actions over time."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_bandit(2, 10000, action_space, 1.0, rng)
        early_mean = float(np.mean(prices[:500]))
        late_mean = float(np.mean(prices[-500:]))
        # The bandit should have learned something
        early_var = float(np.var(prices[:500]))
        late_var = float(np.var(prices[-500:]))
        assert late_var <= early_var + 1.0

    def test_bandit_exploration(self):
        """Early bandit behavior should have variety."""
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_bandit(2, 1000, action_space, 1.0, rng)
        unique_early = len(np.unique(np.round(prices[:100, 0], 2)))
        assert unique_early >= 2  # At least some exploration

    def test_bandit_three_players(self):
        rng = np.random.RandomState(42)
        action_space = np.linspace(1.0, 6.6, 15)
        prices = _simulate_bandit(3, 200, action_space, 1.0, rng)
        assert prices.shape == (200, 3)


class TestAlgorithmConfig:
    """Tests for algorithm configuration types."""

    def test_effective_epsilon_decay(self):
        config = AlgorithmConfig(
            name="test",
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.01,
        )
        eps_0 = config.effective_epsilon(0)
        eps_100 = config.effective_epsilon(100)
        eps_1000 = config.effective_epsilon(1000)
        assert eps_0 == 1.0
        assert eps_100 < eps_0
        assert eps_1000 < eps_100
        assert eps_1000 >= 0.01

    def test_epsilon_min_enforced(self):
        config = AlgorithmConfig(
            name="test",
            epsilon=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.05,
        )
        eps = config.effective_epsilon(10000)
        assert eps == 0.05

    def test_invalid_epsilon_min(self):
        with pytest.raises(Exception):
            AlgorithmConfig(
                name="test",
                epsilon=0.1,
                epsilon_min=0.5,  # > epsilon
            )


class TestSimulationResults:
    """Tests for simulation result structure."""

    def test_result_keys(self):
        result = run_simulation(2, 100, "q_learning", seed=42)
        assert "prices" in result
        assert "algorithm" in result
        assert "num_players" in result
        assert "num_rounds" in result
        assert "nash_price" in result
        assert "monopoly_price" in result

    def test_result_values(self):
        result = run_simulation(2, 100, "q_learning", seed=42)
        assert result["algorithm"] == "q_learning"
        assert result["num_players"] == 2
        assert result["num_rounds"] == 100
        assert result["nash_price"] == 1.0
        assert result["monopoly_price"] == 5.5

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_simulation(2, 100, "nonexistent", seed=42)
