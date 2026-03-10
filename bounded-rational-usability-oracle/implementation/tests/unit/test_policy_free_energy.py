"""Unit tests for usability_oracle.policy.free_energy — FreeEnergyComputer.

Validates free-energy computation, decomposition into expected cost and
information cost, optimal policy computation, rate-distortion curves,
and the fundamental relationship F = E[c] + (1/β) D_KL(π ‖ p₀).
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.policy.free_energy import (
    FreeEnergyComputer,
    FreeEnergyDecomposition,
)
from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.models import Policy, QValues
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.mdp.solver import ValueIterationSolver
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _uniform_prior(mdp: MDP) -> Policy:
    """Construct a uniform prior over all actions in each state."""
    probs: dict[str, dict[str, float]] = {}
    for sid in mdp.states:
        actions = mdp.get_actions(sid)
        if actions:
            p = 1.0 / len(actions)
            probs[sid] = {a: p for a in actions}
    return Policy(state_action_probs=probs)


def _make_policy_from_mdp(mdp: MDP, beta: float) -> Policy:
    """Create a softmax policy by solving the MDP and converting Q-values."""
    solver = ValueIterationSolver()
    values, _ = solver.solve(mdp)
    q_dict: dict[str, dict[str, float]] = {}
    for sid in mdp.states:
        state = mdp.states[sid]
        if state.is_terminal or state.is_goal:
            continue
        available = mdp.get_actions(sid)
        if not available:
            continue
        q_s: dict[str, float] = {}
        for aid in available:
            outcomes = mdp.get_transitions(sid, aid)
            q_sa = sum(p * (c + mdp.discount * values.get(t, 0.0))
                       for t, p, c in outcomes)
            q_s[aid] = q_sa
        q_dict[sid] = q_s
    q = QValues(values=q_dict)
    return SoftmaxPolicy.from_q_values(q, beta)


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyComputer.compute tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFreeEnergyCompute:
    """Tests for FreeEnergyComputer.compute() — total free energy F(π)."""

    def test_compute_returns_float(self):
        """compute() should return a float."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        result = fe.compute(policy, mdp, beta=2.0, prior=prior)
        assert isinstance(result, float)

    def test_compute_finite(self):
        """Free energy should be finite."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=5.0)
        fe = FreeEnergyComputer()
        result = fe.compute(policy, mdp, beta=5.0, prior=prior)
        assert math.isfinite(result)

    def test_free_energy_nonnegative_for_positive_costs(self):
        """For MDPs with strictly positive costs, free energy should be > 0."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        result = fe.compute(policy, mdp, beta=2.0, prior=prior)
        assert result >= 0.0

    def test_compute_on_cyclic_mdp(self):
        """compute() works on the cyclic MDP without error."""
        mdp = make_cyclic_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=1.0)
        fe = FreeEnergyComputer()
        result = fe.compute(policy, mdp, beta=1.0, prior=prior)
        assert math.isfinite(result)


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyComputer.decompose tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFreeEnergyDecompose:
    """Tests for FreeEnergyComputer.decompose() — F = E[c] + (1/β) D_KL."""

    def test_decompose_returns_type(self):
        """decompose() should return a FreeEnergyDecomposition instance."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=2.0, prior=prior)
        assert isinstance(decomp, FreeEnergyDecomposition)

    def test_decompose_sum_equality(self):
        """total_free_energy should equal expected_cost + information_cost."""
        mdp = make_choice_mdp(n_choices=4)
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=3.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=3.0, prior=prior)
        assert decomp.total_free_energy == pytest.approx(
            decomp.expected_cost + decomp.information_cost, rel=1e-6
        )

    def test_information_cost_nonnegative(self):
        """Information cost (1/β) D_KL ≥ 0."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=2.0, prior=prior)
        assert decomp.information_cost >= -1e-10

    def test_expected_cost_nonnegative(self):
        """Expected cost under a policy on a positive-cost MDP should be >= 0."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=2.0, prior=prior)
        assert decomp.expected_cost >= -1e-10

    def test_per_state_free_energy_populated(self):
        """Per-state decomposition dicts should be populated."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=2.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=2.0, prior=prior)
        assert len(decomp.per_state_free_energy) > 0
        assert len(decomp.per_state_expected_cost) > 0
        assert len(decomp.per_state_information_cost) > 0

    def test_beta_stored_in_decomposition(self):
        """The β value should be stored in the decomposition."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        policy = _make_policy_from_mdp(mdp, beta=7.0)
        fe = FreeEnergyComputer()
        decomp = fe.decompose(policy, mdp, beta=7.0, prior=prior)
        assert decomp.beta == 7.0

    def test_very_high_beta_low_information_cost(self):
        """At very high β, information cost (1/β)·D_KL → 0 since D_KL is bounded
        by log(|A|) while 1/β → 0."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()

        policy_high = _make_policy_from_mdp(mdp, beta=1000.0)
        decomp_high = fe.decompose(policy_high, mdp, beta=1000.0, prior=prior)

        # D_KL ≤ log(3) ≈ 1.1, so (1/1000)*1.1 ≈ 0.001
        assert decomp_high.information_cost < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyComputer.optimal_policy tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimalPolicy:
    """Tests for FreeEnergyComputer.optimal_policy() — free-energy minimiser."""

    def test_returns_policy(self):
        """optimal_policy() should return a Policy instance."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()
        policy = fe.optimal_policy(mdp, beta=2.0, prior=prior)
        assert isinstance(policy, Policy)

    def test_optimal_policy_probabilities_sum_to_one(self):
        """Probabilities in the optimal policy must sum to 1 per state."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()
        policy = fe.optimal_policy(mdp, beta=5.0, prior=prior)
        for state, dist in policy.state_action_probs.items():
            total = sum(dist.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_optimal_policy_has_lower_free_energy(self):
        """The optimal policy should have <= free energy of a random policy."""
        mdp = make_choice_mdp(n_choices=4)
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()

        optimal_pol = fe.optimal_policy(mdp, beta=5.0, prior=prior)
        fe_optimal = fe.compute(optimal_pol, mdp, beta=5.0, prior=prior)
        fe_prior = fe.compute(prior, mdp, beta=5.0, prior=prior)

        assert fe_optimal <= fe_prior + 1e-6

    def test_optimal_policy_custom_parameters(self):
        """optimal_policy accepts custom discount, epsilon, max_iter."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()
        policy = fe.optimal_policy(
            mdp, beta=2.0, prior=prior,
            discount=0.95, epsilon=1e-4, max_iter=100,
        )
        assert isinstance(policy, Policy)


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyComputer.rate_distortion_curve tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRateDistortionCurve:
    """Tests for FreeEnergyComputer.rate_distortion_curve()."""

    def test_returns_list_of_tuples(self):
        """rate_distortion_curve should return [(info_cost, expected_cost), ...]."""
        mdp = make_two_state_mdp()
        fe = FreeEnergyComputer()
        curve = fe.rate_distortion_curve(mdp, betas=[0.5, 2.0, 10.0])
        assert isinstance(curve, list)
        assert len(curve) == 3
        for point in curve:
            assert isinstance(point, tuple)
            assert len(point) == 2

    def test_curve_length_matches_betas(self):
        """One (rate, distortion) point per β."""
        mdp = make_choice_mdp(n_choices=3)
        fe = FreeEnergyComputer()
        betas = [0.1, 1.0, 5.0, 20.0]
        curve = fe.rate_distortion_curve(mdp, betas=betas)
        assert len(curve) == len(betas)

    def test_information_cost_nonnegative_on_curve(self):
        """Information cost in each curve point should be >= 0."""
        mdp = make_two_state_mdp()
        fe = FreeEnergyComputer()
        curve = fe.rate_distortion_curve(mdp, betas=[1.0, 5.0])
        for info_cost, _ in curve:
            assert info_cost >= -1e-10

    def test_curve_with_explicit_prior(self):
        """rate_distortion_curve should accept an explicit prior."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        fe = FreeEnergyComputer()
        curve = fe.rate_distortion_curve(mdp, betas=[1.0, 5.0], prior=prior)
        assert len(curve) == 2

    def test_curve_values_finite(self):
        """All rate-distortion curve values should be finite."""
        mdp = make_choice_mdp(n_choices=3)
        fe = FreeEnergyComputer()
        curve = fe.rate_distortion_curve(mdp, betas=[0.5, 2.0, 10.0])
        for info_cost, exp_cost in curve:
            assert math.isfinite(info_cost)
            assert math.isfinite(exp_cost)
