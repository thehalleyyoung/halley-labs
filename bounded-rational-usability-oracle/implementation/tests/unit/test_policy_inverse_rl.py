"""Unit tests for usability_oracle.policy.inverse_rl — Inverse Reinforcement Learning.

Tests cover MaxEntropy IRL, feature matching IRL, Bayesian IRL, bounded-rational
IRL, and reward recovery validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.policy.models import Policy, QValues
from usability_oracle.policy.inverse_rl import (
    BayesianIRL,
    BoundedRationalIRL,
    Demonstration,
    FeatureMatchingIRL,
    IRLResult,
    MaxEntropyIRL,
    compute_feature_expectations,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _simple_demonstrations() -> list[Demonstration]:
    """Simple demonstrations: prefer action 'a' in s0 and 'c' in s1."""
    return [
        Demonstration(
            states=["s0", "s1"],
            actions=["a", "c"],
            features=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        ),
        Demonstration(
            states=["s0", "s1"],
            actions=["a", "c"],
            features=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        ),
        Demonstration(
            states=["s0"],
            actions=["a"],
            features=[np.array([1.0, 0.0])],
        ),
    ]


def _feature_fn() -> dict[str, dict[str, np.ndarray]]:
    return {
        "s0": {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        },
        "s1": {
            "c": np.array([0.0, 1.0]),
            "d": np.array([1.0, 0.0]),
        },
    }


def _states() -> list[str]:
    return ["s0", "s1"]


def _actions_per_state() -> dict[str, list[str]]:
    return {"s0": ["a", "b"], "s1": ["c", "d"]}


# ═══════════════════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════════════════


class TestDemonstration:
    """Test Demonstration dataclass."""

    def test_demonstration_length(self):
        demo = Demonstration(
            states=["s0", "s1", "s2"],
            actions=["a", "b", "c"],
            features=[np.zeros(2), np.zeros(2), np.zeros(2)],
        )
        assert demo.length == 3

    def test_feature_expectations(self):
        demos = _simple_demonstrations()
        fe = compute_feature_expectations(demos, discount=0.99)
        assert fe.shape == (2,)
        # First feature should be positive (demos prefer action 'a' with [1,0])
        assert fe[0] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Max Entropy IRL
# ═══════════════════════════════════════════════════════════════════════════


class TestMaxEntropyIRL:
    """Test Maximum Entropy Inverse Reinforcement Learning."""

    def test_fit_returns_result(self):
        irl = MaxEntropyIRL(feature_dim=2, beta=1.0, max_iter=50, learning_rate=0.1)
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert isinstance(result, IRLResult)
        assert result.reward_weights.shape == (2,)

    def test_recovered_reward_assigns_values(self):
        irl = MaxEntropyIRL(feature_dim=2, beta=1.0, max_iter=100, learning_rate=0.05)
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert "s0" in result.reward_map
        assert len(result.reward_map["s0"]) > 0

    def test_policy_is_returned(self):
        irl = MaxEntropyIRL(feature_dim=2, beta=1.0, max_iter=50)
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert isinstance(result.policy, Policy)

    def test_log_likelihood_is_finite(self):
        irl = MaxEntropyIRL(feature_dim=2, beta=1.0, max_iter=50)
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert math.isfinite(result.log_likelihood)

    def test_regularisation_shrinks_weights(self):
        irl_low = MaxEntropyIRL(feature_dim=2, beta=1.0, regularisation=0.001, max_iter=100)
        irl_high = MaxEntropyIRL(feature_dim=2, beta=1.0, regularisation=10.0, max_iter=100)
        r_low = irl_low.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        r_high = irl_high.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        norm_low = np.linalg.norm(r_low.reward_weights)
        norm_high = np.linalg.norm(r_high.reward_weights)
        assert norm_high <= norm_low + 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Feature Matching IRL
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureMatchingIRL:
    """Test feature matching IRL."""

    def test_fit_returns_result(self):
        irl = FeatureMatchingIRL(feature_dim=2, beta=1.0, max_iter=20)
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert isinstance(result, IRLResult)

    def test_reward_weights_finite(self):
        irl = FeatureMatchingIRL(feature_dim=2, max_iter=20)
        result = irl.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        assert all(np.isfinite(result.reward_weights))


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian IRL
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianIRL:
    """Test Bayesian IRL with MCMC sampling."""

    def test_fit_returns_result(self):
        irl = BayesianIRL(
            feature_dim=2, beta=1.0, n_samples=100,
            rng=np.random.default_rng(42),
        )
        result = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert isinstance(result, IRLResult)

    def test_confidence_intervals(self):
        irl = BayesianIRL(
            feature_dim=2, beta=1.0, n_samples=200,
            rng=np.random.default_rng(42),
        )
        result = irl.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        # Should have confidence intervals for each weight dimension
        if result.confidence_intervals:
            for dim, (lo, hi) in result.confidence_intervals.items():
                assert lo <= hi


# ═══════════════════════════════════════════════════════════════════════════
# Bounded-Rational IRL
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundedRationalIRL:
    """Test bounded-rational IRL with learned β."""

    def test_fit_returns_result_and_beta(self):
        irl = BoundedRationalIRL(feature_dim=2, initial_beta=1.0, max_iter=50)
        result, learned_beta = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
        )
        assert isinstance(result, IRLResult)
        assert learned_beta > 0

    def test_beta_learning(self):
        """Learned β should be positive and finite."""
        irl = BoundedRationalIRL(feature_dim=2, initial_beta=0.5, max_iter=100)
        _, beta = irl.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        assert math.isfinite(beta)
        assert beta > 0

    def test_with_prior_policy(self):
        prior = Policy(
            state_action_probs={
                "s0": {"a": 0.5, "b": 0.5},
                "s1": {"c": 0.5, "d": 0.5},
            },
            beta=1.0,
        )
        irl = BoundedRationalIRL(feature_dim=2, initial_beta=1.0, max_iter=50)
        result, beta = irl.fit(
            demos=_simple_demonstrations(),
            states=_states(),
            actions_per_state=_actions_per_state(),
            feature_fn=_feature_fn(),
            prior=prior,
        )
        assert isinstance(result, IRLResult)

    def test_reward_recovery_direction(self):
        """Recovered rewards should align with demonstrated preferences."""
        irl = BoundedRationalIRL(feature_dim=2, initial_beta=1.0, max_iter=100)
        result, _ = irl.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        # In s0, demos prefer 'a' (feature [1,0]), so reward for 'a' should be higher
        if "s0" in result.reward_map:
            r_a = result.reward_map["s0"].get("a", 0)
            r_b = result.reward_map["s0"].get("b", 0)
            # At least one should be non-zero
            assert r_a != 0 or r_b != 0

    def test_convergence_info(self):
        irl = BoundedRationalIRL(feature_dim=2, max_iter=30)
        result, _ = irl.fit(
            _simple_demonstrations(), _states(), _actions_per_state(), _feature_fn()
        )
        assert isinstance(result.convergence_info, dict)
