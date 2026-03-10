"""Tests for usability_oracle.variational.free_energy.

Verifies the bounded-rational free-energy computation, softmax policy
derivation, soft value iteration, policy gradients, and optimal β
estimation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from usability_oracle.variational.free_energy import (
    compute_free_energy,
    compute_optimal_beta,
    compute_policy_gradient,
    compute_softmax_policy,
    compute_value_iteration,
)
from usability_oracle.variational.kl_divergence import compute_kl_divergence


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def equal_qvalues() -> np.ndarray:
    """Q-values that are all equal (3 actions)."""
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def distinct_qvalues() -> np.ndarray:
    """Distinct Q-values."""
    return np.array([3.0, 1.0, 0.5])


@pytest.fixture
def uniform_prior() -> np.ndarray:
    """Uniform prior over 3 actions."""
    return np.ones(3) / 3


@pytest.fixture
def simple_transition() -> np.ndarray:
    """2-state, 2-action deterministic transition tensor."""
    # T[s, a, s'] — shape (2, 2, 2)
    T = np.zeros((2, 2, 2))
    # State 0, action 0 -> state 1
    T[0, 0, 1] = 1.0
    # State 0, action 1 -> state 0 (stay)
    T[0, 1, 0] = 1.0
    # State 1, action 0 -> state 1 (stay at goal)
    T[1, 0, 1] = 1.0
    # State 1, action 1 -> state 0
    T[1, 1, 0] = 1.0
    return T


@pytest.fixture
def simple_reward() -> np.ndarray:
    """2-state, 2-action reward matrix (goal at state 1)."""
    R = np.array([
        [-1.0, -0.5],   # state 0: both actions costly
        [0.0, -2.0],    # state 1: staying is free, leaving is costly
    ])
    return R


@pytest.fixture
def stochastic_transition() -> np.ndarray:
    """3-state, 2-action stochastic transition tensor."""
    T = np.zeros((3, 2, 3))
    # State 0, action 0 -> state 1 (0.8) or state 2 (0.2)
    T[0, 0, 1] = 0.8
    T[0, 0, 2] = 0.2
    # State 0, action 1 -> state 0 (stay)
    T[0, 1, 0] = 1.0
    # State 1, action 0 -> state 2
    T[1, 0, 2] = 1.0
    # State 1, action 1 -> state 0
    T[1, 1, 0] = 1.0
    # State 2 (absorbing)
    T[2, 0, 2] = 1.0
    T[2, 1, 2] = 1.0
    return T


@pytest.fixture
def stochastic_reward() -> np.ndarray:
    """3-state, 2-action reward."""
    return np.array([
        [-1.0, -0.5],
        [-0.5, -1.5],
        [0.0, 0.0],  # absorbing goal
    ])


# =====================================================================
# Free energy computation
# =====================================================================

class TestComputeFreeEnergy:
    """Test the free energy F(π) = E_π[R] - (1/β) D_KL(π || p₀)."""

    def test_high_beta_recovers_optimal_policy(self, distinct_qvalues: np.ndarray) -> None:
        """With β → ∞, the softmax policy concentrates on the best action
        and free energy approaches max reward."""
        prior = np.ones(3) / 3
        beta = 1000.0
        policy = compute_softmax_policy(distinct_qvalues, beta, prior)
        fe = compute_free_energy(policy, distinct_qvalues, prior, beta)
        # Should be close to the max Q-value
        assert fe == pytest.approx(3.0, abs=0.1)

    def test_low_beta_recovers_prior(self) -> None:
        """With β → 0⁺, the KL penalty dominates and policy → prior."""
        q = np.array([10.0, 1.0, 0.1])
        prior = np.array([0.2, 0.5, 0.3])
        beta = 0.001
        policy = compute_softmax_policy(q, beta, prior)
        # Policy should be close to prior
        np.testing.assert_allclose(policy, prior, atol=0.05)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            compute_free_energy(
                np.array([0.5, 0.5]),
                np.array([1.0, 2.0, 3.0]),
                np.array([0.5, 0.5]),
                1.0,
            )


# =====================================================================
# Softmax policy
# =====================================================================

class TestSoftmaxPolicy:
    """Test bounded-rational softmax policy derivation."""

    def test_sums_to_one(self, distinct_qvalues: np.ndarray) -> None:
        """Softmax policy should sum to 1."""
        for beta in [0.1, 1.0, 10.0, 100.0]:
            policy = compute_softmax_policy(distinct_qvalues, beta)
            assert policy.sum() == pytest.approx(1.0, abs=1e-10)

    def test_equal_qvalues_gives_uniform(self, equal_qvalues: np.ndarray) -> None:
        """With equal Q-values, softmax gives uniform policy."""
        policy = compute_softmax_policy(equal_qvalues, beta=1.0)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(policy, expected, atol=1e-10)

    def test_equal_qvalues_with_prior_gives_prior(self) -> None:
        """Equal Q + non-uniform prior → prior is the softmax output."""
        q = np.array([1.0, 1.0, 1.0])
        prior = np.array([0.5, 0.3, 0.2])
        policy = compute_softmax_policy(q, beta=1.0, prior=prior)
        np.testing.assert_allclose(policy, prior, atol=1e-10)

    def test_higher_beta_more_peaked(self, distinct_qvalues: np.ndarray) -> None:
        """Higher β should produce more peaked policy."""
        policy_low = compute_softmax_policy(distinct_qvalues, beta=0.5)
        policy_high = compute_softmax_policy(distinct_qvalues, beta=10.0)
        # The best action should have higher probability with higher β
        assert policy_high[0] > policy_low[0]

    @pytest.mark.parametrize("beta", [0.01, 0.1, 1.0, 5.0, 50.0])
    def test_all_probabilities_non_negative(
        self, distinct_qvalues: np.ndarray, beta: float
    ) -> None:
        """All softmax probabilities should be non-negative."""
        policy = compute_softmax_policy(distinct_qvalues, beta)
        assert np.all(policy >= 0)

    def test_2d_qvalues(self) -> None:
        """Test softmax on 2D Q-value array (n_states × n_actions)."""
        q = np.array([[1.0, 2.0], [3.0, 1.0]])
        policy = compute_softmax_policy(q, beta=2.0)
        assert policy.shape == (2, 2)
        for s in range(2):
            assert policy[s].sum() == pytest.approx(1.0, abs=1e-10)


# =====================================================================
# Soft value iteration
# =====================================================================

class TestSoftValueIteration:
    """Test soft Bellman iteration with KL penalty."""

    def test_converges_deterministic_mdp(
        self, simple_transition: np.ndarray, simple_reward: np.ndarray
    ) -> None:
        """Soft value iteration should converge on a simple 2-state MDP."""
        V, policy, n_iter = compute_value_iteration(
            simple_transition, simple_reward, beta=5.0, gamma=0.9,
            max_iterations=200,
        )
        assert V.shape == (2,)
        assert policy.shape == (2, 2)
        # Goal state (state 1) should have higher value
        assert V[1] >= V[0]
        # Policy should sum to 1 for each state
        for s in range(2):
            assert policy[s].sum() == pytest.approx(1.0, abs=1e-10)

    def test_converges_stochastic_mdp(
        self, stochastic_transition: np.ndarray, stochastic_reward: np.ndarray
    ) -> None:
        """Soft VI converges on a 3-state stochastic MDP."""
        V, policy, n_iter = compute_value_iteration(
            stochastic_transition, stochastic_reward, beta=2.0, gamma=0.95,
            max_iterations=500,
        )
        assert V.shape == (3,)
        assert policy.shape == (3, 2)
        # Absorbing goal state should have highest value
        assert V[2] >= V[0]

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched reward shape should raise ValueError."""
        T = np.zeros((2, 2, 2))
        R_bad = np.zeros((3, 2))  # wrong number of states
        with pytest.raises(ValueError, match="reward"):
            compute_value_iteration(T, R_bad, beta=1.0)


# =====================================================================
# Policy gradient
# =====================================================================

class TestPolicyGradient:
    """Test gradient of free energy w.r.t. policy parameters."""

    def test_gradient_shape(self) -> None:
        """Gradient should have same shape as policy."""
        policy = np.array([0.4, 0.3, 0.3])
        reward = np.array([1.0, 2.0, 0.5])
        prior = np.ones(3) / 3
        grad = compute_policy_gradient(policy, reward, prior, beta=1.0)
        assert grad.shape == policy.shape

    def test_gradient_at_optimal_near_zero(self) -> None:
        """At the optimal policy, gradient should be near zero."""
        reward = np.array([2.0, 1.0])
        prior = np.array([0.5, 0.5])
        beta = 5.0
        # Compute optimal policy
        policy = compute_softmax_policy(reward, beta, prior)
        grad = compute_policy_gradient(policy, reward, prior, beta)
        # Gradient should be very small at the optimal
        assert np.max(np.abs(grad)) < 0.01

    @pytest.mark.parametrize("n_actions", [2, 5, 10])
    def test_gradient_sums_to_zero(self, n_actions: int) -> None:
        """The gradient w.r.t. θ in softmax parameterization sums to 0
        because probabilities must sum to 1."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(n_actions))
        reward = rng.standard_normal(n_actions)
        prior = np.ones(n_actions) / n_actions
        grad = compute_policy_gradient(probs, reward, prior, beta=2.0)
        assert abs(grad.sum()) < 1e-10


# =====================================================================
# Optimal beta estimation
# =====================================================================

class TestOptimalBeta:
    """Test bisection search for optimal β."""

    def test_returns_valid_beta(self) -> None:
        """compute_optimal_beta should return a finite non-negative β."""
        reward = np.array([3.0, 1.0, 0.5])
        target_mi = 0.5  # nats
        beta = compute_optimal_beta(reward, target_mi)
        assert beta >= 0.0
        assert np.isfinite(beta)

    def test_zero_target_returns_zero(self) -> None:
        """Target MI = 0 → β = 0 (prior policy)."""
        reward = np.array([1.0, 2.0])
        beta = compute_optimal_beta(reward, target_mutual_info=0.0)
        assert beta == 0.0

    def test_high_target_returns_large_beta(self) -> None:
        """Target MI close to max entropy → large β."""
        reward = np.array([1.0, 2.0, 3.0])
        max_mi = math.log(3)
        beta = compute_optimal_beta(reward, target_mutual_info=max_mi + 0.1)
        assert beta > 100.0

    def test_increasing_target_gives_increasing_beta(self) -> None:
        """Higher target MI → higher β."""
        reward = np.array([2.0, 1.0, 0.5])
        beta_low = compute_optimal_beta(reward, target_mutual_info=0.2)
        beta_high = compute_optimal_beta(reward, target_mutual_info=0.8)
        assert beta_high >= beta_low


# =====================================================================
# Blahut-Arimoto convergence (via FreeEnergyComputer)
# =====================================================================

class TestBlahutArimotoConvergence:
    """Test convergence of the Blahut-Arimoto-style alternating projection."""

    def test_convergence_simple_cost_matrix(self) -> None:
        """FreeEnergyComputer should converge for a simple cost matrix."""
        from usability_oracle.variational.free_energy import FreeEnergyComputer
        from usability_oracle.variational.types import VariationalConfig

        cost_matrix = {"s0": {"a0": 0.5, "a1": 1.0}, "s1": {"a0": 0.8, "a1": 0.3}}
        reference = {"s0": {"a0": 0.5, "a1": 0.5}, "s1": {"a0": 0.5, "a1": 0.5}}
        config = VariationalConfig(beta=2.0, max_iterations=200, tolerance=1e-6)
        computer = FreeEnergyComputer(config)
        result = computer.compute(cost_matrix, reference)
        # Should produce a valid policy
        for state in result.policy:
            probs = list(result.policy[state].values())
            assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(result.free_energy)


# =====================================================================
# Deterministic 2-state MDP
# =====================================================================

class TestDeterministic2StateMDP:
    """Integration-style test with a known deterministic 2-state MDP."""

    def test_optimal_action_selected(self) -> None:
        """In a deterministic MDP, high β → nearly deterministic policy."""
        T = np.zeros((2, 2, 2))
        T[0, 0, 1] = 1.0  # action 0: go to goal
        T[0, 1, 0] = 1.0  # action 1: stay
        T[1, 0, 1] = 1.0
        T[1, 1, 1] = 1.0
        R = np.array([[-1.0, -2.0], [0.0, 0.0]])
        V, policy, _ = compute_value_iteration(T, R, beta=20.0, gamma=0.95)
        # In state 0, action 0 (go to goal) should dominate
        assert policy[0, 0] > 0.9


# =====================================================================
# Stochastic 3-state MDP
# =====================================================================

class TestStochastic3StateMDP:
    """Test with a 3-state stochastic MDP."""

    def test_policy_reflects_stochasticity(self) -> None:
        """Policy should reflect transition probabilities."""
        T = np.zeros((3, 2, 3))
        T[0, 0, 1] = 0.9
        T[0, 0, 0] = 0.1
        T[0, 1, 2] = 0.5
        T[0, 1, 0] = 0.5
        T[1, 0, 2] = 1.0
        T[1, 1, 0] = 1.0
        T[2, 0, 2] = 1.0
        T[2, 1, 2] = 1.0
        R = np.array([[-1.0, -1.5], [-0.5, -2.0], [0.0, 0.0]])
        V, policy, _ = compute_value_iteration(T, R, beta=5.0, gamma=0.9)
        # Goal state 2 should have highest value
        assert V[2] >= V[0]
        assert V[2] >= V[1]
        # Policy is valid
        for s in range(3):
            assert policy[s].sum() == pytest.approx(1.0, abs=1e-10)
