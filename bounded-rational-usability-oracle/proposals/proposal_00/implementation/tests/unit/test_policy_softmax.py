"""Unit tests for usability_oracle.policy.softmax — SoftmaxPolicy.

Validates softmax/Boltzmann policy construction from Q-values, the effect
of the rationality parameter β on policy determinism, information-theoretic
measures (KL divergence, mutual information), beta sweeps, effective
rationality estimation, and internal numerical stability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.models import Policy, QValues


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _simple_q() -> QValues:
    """Q-values for a two-state MDP: start has actions a (cost 1) and b (cost 3)."""
    return QValues(values={
        "start": {"a": 1.0, "b": 3.0},
    })


def _multi_state_q() -> QValues:
    """Q-values for a three-state MDP."""
    return QValues(values={
        "s0": {"left": 2.0, "right": 4.0},
        "s1": {"up": 1.0, "down": 5.0},
        "s2": {"go": 3.0},
    })


def _uniform_prior(q: QValues) -> Policy:
    """Uniform prior over the same action sets."""
    probs: dict[str, dict[str, float]] = {}
    for state, actions in q.values.items():
        n = len(actions)
        probs[state] = {a: 1.0 / n for a in actions}
    return Policy(state_action_probs=probs)


# ═══════════════════════════════════════════════════════════════════════════
# from_q_values tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFromQValues:
    """Tests for SoftmaxPolicy.from_q_values — constructing a policy from Q-values."""

    def test_returns_policy(self):
        """from_q_values should return a Policy instance."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=1.0)
        assert isinstance(policy, Policy)

    def test_probabilities_sum_to_one(self):
        """Action probabilities must sum to 1 for each state."""
        q = _multi_state_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=2.0)
        for state in q.values:
            probs = policy.state_action_probs[state]
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=1e-8)

    def test_high_beta_near_deterministic(self):
        """High β should produce a near-deterministic (greedy) policy
        concentrating mass on the lowest-cost action."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=100.0)
        probs = policy.state_action_probs["start"]
        # Action "a" has lower cost (1.0) → higher probability
        assert probs["a"] > 0.99

    def test_low_beta_near_uniform(self):
        """Low β should produce a near-uniform policy."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=0.001)
        probs = policy.state_action_probs["start"]
        assert abs(probs["a"] - probs["b"]) < 0.05

    def test_beta_zero_gives_prior(self):
        """β = 0 should yield the prior policy (uniform by default)."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=0.0)
        probs = policy.state_action_probs["start"]
        assert probs["a"] == pytest.approx(0.5, abs=0.05)
        assert probs["b"] == pytest.approx(0.5, abs=0.05)

    def test_with_explicit_prior(self):
        """A non-uniform prior should shift the resulting policy."""
        q = _simple_q()
        prior = Policy(state_action_probs={"start": {"a": 0.9, "b": 0.1}})
        policy = SoftmaxPolicy.from_q_values(q, beta=1.0, prior=prior)
        # Even with moderate β, the strong prior on "a" should amplify its probability
        assert policy.state_action_probs["start"]["a"] > 0.5

    def test_multi_state(self):
        """from_q_values should create distributions for all states."""
        q = _multi_state_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=1.0)
        assert "s0" in policy.state_action_probs
        assert "s1" in policy.state_action_probs
        assert "s2" in policy.state_action_probs


# ═══════════════════════════════════════════════════════════════════════════
# Internal _softmax tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSoftmaxInternal:
    """Tests for SoftmaxPolicy._softmax and _softmax_from_logits."""

    def test_softmax_sums_to_one(self):
        """_softmax output should sum to 1."""
        values = np.array([1.0, 2.0, 3.0])
        probs = SoftmaxPolicy._softmax(values, beta=1.0)
        assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-10)

    def test_softmax_high_beta(self):
        """High β should put most mass on the maximum value."""
        values = np.array([1.0, 5.0, 2.0])
        probs = SoftmaxPolicy._softmax(values, beta=50.0)
        assert probs[1] > 0.99

    def test_softmax_from_logits_stability(self):
        """_softmax_from_logits should handle large logits without overflow."""
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = SoftmaxPolicy._softmax_from_logits(logits)
        assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-10)
        assert np.all(np.isfinite(probs))

    def test_softmax_from_logits_negative(self):
        """_softmax_from_logits handles large negative logits."""
        logits = np.array([-1000.0, -999.0, -1001.0])
        probs = SoftmaxPolicy._softmax_from_logits(logits)
        assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-10)

    def test_log_partition(self):
        """_log_partition should compute log Σ exp(β·v_a)."""
        values = np.array([0.0, 0.0, 0.0])
        log_z = SoftmaxPolicy._log_partition(values, beta=1.0)
        assert log_z == pytest.approx(math.log(3.0), abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# KL divergence and mutual information tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKLDivergence:
    """Tests for SoftmaxPolicy.kl_divergence and mutual_information."""

    def test_kl_divergence_nonnegative(self):
        """KL divergence D_KL(π ‖ p₀) must be >= 0."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=2.0)
        prior = _uniform_prior(q)
        kl = SoftmaxPolicy.kl_divergence(policy, prior, "start")
        assert kl >= 0.0

    def test_kl_divergence_self_zero(self):
        """D_KL(p ‖ p) should be 0 (policy = prior)."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=0.0)
        prior = _uniform_prior(q)
        kl = SoftmaxPolicy.kl_divergence(policy, prior, "start")
        assert kl == pytest.approx(0.0, abs=0.05)

    def test_kl_increases_with_beta(self):
        """Higher β → more deviation from prior → higher KL."""
        q = _simple_q()
        prior = _uniform_prior(q)
        kl_low = SoftmaxPolicy.kl_divergence(
            SoftmaxPolicy.from_q_values(q, beta=0.5), prior, "start"
        )
        kl_high = SoftmaxPolicy.kl_divergence(
            SoftmaxPolicy.from_q_values(q, beta=10.0), prior, "start"
        )
        assert kl_high > kl_low

    def test_mutual_information_nonnegative(self):
        """Mutual information I(S; A) must be >= 0."""
        q = _multi_state_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=2.0)
        prior = _uniform_prior(q)
        mi = SoftmaxPolicy.mutual_information(policy, prior)
        assert mi >= 0.0

    def test_mutual_information_empty_policy(self):
        """MI should be 0 for an empty policy."""
        policy = Policy(state_action_probs={})
        prior = Policy(state_action_probs={})
        mi = SoftmaxPolicy.mutual_information(policy, prior)
        assert mi == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Beta sweep and effective rationality tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBetaSweep:
    """Tests for SoftmaxPolicy.beta_sweep and effective_rationality."""

    def test_beta_sweep_returns_list(self):
        """beta_sweep should return one Policy per β value."""
        q = _simple_q()
        betas = [0.1, 1.0, 10.0]
        policies = SoftmaxPolicy.beta_sweep(q, betas)
        assert len(policies) == 3
        assert all(isinstance(p, Policy) for p in policies)

    def test_beta_sweep_monotone_determinism(self):
        """Policies should become more deterministic as β increases."""
        q = _simple_q()
        betas = [0.1, 1.0, 5.0, 20.0]
        policies = SoftmaxPolicy.beta_sweep(q, betas)
        max_probs = [
            max(p.state_action_probs["start"].values()) for p in policies
        ]
        for i in range(len(max_probs) - 1):
            assert max_probs[i + 1] >= max_probs[i] - 1e-8

    def test_beta_sweep_with_prior(self):
        """beta_sweep accepts an optional prior."""
        q = _simple_q()
        prior = Policy(state_action_probs={"start": {"a": 0.8, "b": 0.2}})
        policies = SoftmaxPolicy.beta_sweep(q, [0.5, 5.0], prior=prior)
        assert len(policies) == 2

    def test_effective_rationality_positive(self):
        """effective_rationality should return a positive value for non-trivial policy."""
        q = _multi_state_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=5.0)
        prior = _uniform_prior(q)
        eff_beta = SoftmaxPolicy.effective_rationality(policy, prior)
        assert eff_beta > 0.0

    def test_effective_rationality_zero_for_prior(self):
        """effective_rationality should be ~0 when policy ≈ prior (β → 0)."""
        q = _simple_q()
        policy = SoftmaxPolicy.from_q_values(q, beta=0.0)
        prior = _uniform_prior(q)
        eff_beta = SoftmaxPolicy.effective_rationality(policy, prior)
        assert eff_beta == 0.0

    def test_effective_rationality_empty_policy(self):
        """Empty policy should yield 0 effective rationality."""
        policy = Policy(state_action_probs={})
        prior = Policy(state_action_probs={})
        eff_beta = SoftmaxPolicy.effective_rationality(policy, prior)
        assert eff_beta == 0.0
