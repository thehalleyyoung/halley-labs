"""Unit tests for usability_oracle.information_theory.free_energy.

Tests cover free energy computation, optimal policy recovery, Bethe
approximation, annealing schedules, and phase transition detection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.information_theory.free_energy import (
    bethe_free_energy,
    bounded_rational_value,
    cosine_annealing,
    exponential_annealing,
    find_phase_transitions,
    free_energy_decomposition,
    free_energy_landscape,
    information_cost,
    linear_annealing,
    minimize_free_energy,
    optimal_free_energy,
    optimal_policy,
    variational_free_energy,
)
from usability_oracle.information_theory.mutual_information import kl_divergence


# ------------------------------------------------------------------ #
# Core free energy computation
# ------------------------------------------------------------------ #


class TestVariationalFreeEnergy:
    """Tests for F(π) = E_π[R] − (1/β) D_KL(π ‖ p₀)."""

    @pytest.fixture
    def uniform_prior(self) -> np.ndarray:
        return np.array([1 / 3, 1 / 3, 1 / 3])

    @pytest.fixture
    def rewards(self) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])

    def test_fe_at_prior_equals_expected_reward(
        self, uniform_prior: np.ndarray, rewards: np.ndarray,
    ) -> None:
        """F(p₀) = E_{p₀}[R] since D_KL(p₀‖p₀)=0."""
        fe = variational_free_energy(uniform_prior, uniform_prior, rewards, beta=5.0)
        expected = float(np.dot(uniform_prior, rewards))
        assert fe == pytest.approx(expected, rel=1e-9)

    def test_fe_decreases_with_kl_penalty(
        self, uniform_prior: np.ndarray, rewards: np.ndarray,
    ) -> None:
        """Policy far from prior has lower F (KL penalty)."""
        greedy = np.array([0.0, 0.0, 1.0])
        fe_prior = variational_free_energy(uniform_prior, uniform_prior, rewards, beta=1.0)
        fe_greedy = variational_free_energy(greedy, uniform_prior, rewards, beta=1.0)
        assert fe_prior > fe_greedy  # Prior has 0 KL cost

    def test_fe_high_beta_favours_best(
        self, uniform_prior: np.ndarray, rewards: np.ndarray,
    ) -> None:
        """At high β, F is dominated by expected reward."""
        greedy = np.array([0.0, 0.0, 1.0])
        fe = variational_free_energy(greedy, uniform_prior, rewards, beta=1000.0)
        expected_reward = 3.0
        kl_cost = kl_divergence(greedy, uniform_prior, base=math.e)
        assert fe == pytest.approx(expected_reward - kl_cost / 1000.0, rel=1e-4)

    def test_fe_zero_beta_ignores_kl(
        self, uniform_prior: np.ndarray, rewards: np.ndarray,
    ) -> None:
        """β=0 → F = E[R] (no rationality penalty)."""
        pi = np.array([0.1, 0.3, 0.6])
        fe = variational_free_energy(pi, uniform_prior, rewards, beta=0.0)
        assert fe == pytest.approx(float(np.dot(pi, rewards)), rel=1e-9)


class TestFreeEnergyDecomposition:
    """Tests for accuracy/complexity decomposition."""

    def test_decomposition_identity(self) -> None:
        """F = accuracy - complexity."""
        prior = np.array([0.5, 0.5])
        pi = np.array([0.3, 0.7])
        r = np.array([1.0, 2.0])
        dec = free_energy_decomposition(pi, prior, r, beta=2.0)
        assert dec["free_energy"] == pytest.approx(
            dec["accuracy"] - dec["complexity"], rel=1e-9,
        )

    def test_accuracy_equals_expected_reward(self) -> None:
        prior = np.array([0.5, 0.5])
        pi = np.array([0.3, 0.7])
        r = np.array([1.0, 2.0])
        dec = free_energy_decomposition(pi, prior, r, beta=2.0)
        assert dec["accuracy"] == pytest.approx(float(np.dot(pi, r)), rel=1e-9)

    def test_complexity_non_negative(self) -> None:
        prior = np.array([0.5, 0.5])
        pi = np.array([0.3, 0.7])
        r = np.array([1.0, 2.0])
        dec = free_energy_decomposition(pi, prior, r, beta=2.0)
        assert dec["complexity"] >= -1e-12


# ------------------------------------------------------------------ #
# Optimal policy recovery
# ------------------------------------------------------------------ #


class TestOptimalPolicy:
    """Tests for π*(a) ∝ p₀(a) exp(β R(a))."""

    def test_sums_to_one(self) -> None:
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        rewards = np.array([1.0, 2.0, 3.0])
        pi = optimal_policy(prior, rewards, beta=5.0)
        assert np.sum(pi) == pytest.approx(1.0, abs=1e-10)

    def test_all_non_negative(self) -> None:
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        rewards = np.array([1.0, 2.0, 3.0])
        pi = optimal_policy(prior, rewards, beta=5.0)
        assert np.all(pi >= 0)

    def test_high_beta_concentrates(self) -> None:
        """Large β → policy concentrates on best action."""
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        rewards = np.array([1.0, 2.0, 5.0])
        pi = optimal_policy(prior, rewards, beta=100.0)
        assert pi[2] > 0.99

    def test_low_beta_uniform(self) -> None:
        """β → 0 → policy = prior."""
        prior = np.array([0.2, 0.3, 0.5])
        rewards = np.array([10.0, 1.0, 0.1])
        pi = optimal_policy(prior, rewards, beta=1e-10)
        np.testing.assert_allclose(pi, prior, atol=1e-4)

    def test_zero_beta_returns_prior(self) -> None:
        prior = np.array([0.2, 0.3, 0.5])
        rewards = np.array([10.0, 1.0, 0.1])
        pi = optimal_policy(prior, rewards, beta=0.0)
        np.testing.assert_allclose(pi, prior, atol=1e-10)

    def test_equal_rewards_returns_prior(self) -> None:
        """When all rewards are equal, optimal policy = prior."""
        prior = np.array([0.2, 0.3, 0.5])
        rewards = np.array([1.0, 1.0, 1.0])
        pi = optimal_policy(prior, rewards, beta=5.0)
        np.testing.assert_allclose(pi, prior, atol=1e-8)

    def test_optimal_fe_consistency(self) -> None:
        """F at π* should equal the closed-form F*."""
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        rewards = np.array([0.0, 1.0, 2.0, 3.0])
        beta = 3.0
        pi_star = optimal_policy(prior, rewards, beta)
        fe_computed = variational_free_energy(pi_star, prior, rewards, beta, base=math.e)
        fe_optimal = optimal_free_energy(prior, rewards, beta)
        assert fe_computed == pytest.approx(fe_optimal, rel=1e-4)


class TestMinimizeFreeEnergy:
    """Tests for iterative free energy minimisation."""

    def test_converges_to_feasible(self) -> None:
        """Iterative minimiser returns a valid probability distribution."""
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        rewards = np.array([0.0, 1.0, 2.0, 4.0])
        beta = 2.0
        pi_found, fe = minimize_free_energy(
            prior,
            lambda pi: float(np.dot(pi, rewards)),
            beta,
        )
        assert np.sum(pi_found) == pytest.approx(1.0, abs=1e-8)
        assert np.all(pi_found >= 0)
        assert math.isfinite(fe)


# ------------------------------------------------------------------ #
# Bethe free energy approximation
# ------------------------------------------------------------------ #


class TestBetheFreeEnergy:
    """Tests for the Bethe free energy on a simple factor graph."""

    def test_single_node_single_edge(self) -> None:
        """Simple 2-node graph with one factor."""
        b_i = np.array([0.6, 0.4])
        b_edge = np.array([[0.3, 0.1], [0.3, 0.3]])
        psi_edge = np.array([[1.0, 0.5], [0.5, 1.0]])
        psi_node = np.array([1.0, 1.0])

        fe = bethe_free_energy(
            node_beliefs=[b_i],
            edge_beliefs=[b_edge],
            node_potentials=[psi_node],
            edge_potentials=[psi_edge],
            node_degrees=[1],
        )
        assert math.isfinite(fe)

    def test_uniform_beliefs(self) -> None:
        """Uniform beliefs should give finite result."""
        n = 3
        b_i = np.full(n, 1.0 / n)
        b_edge = np.full((n, n), 1.0 / (n * n))
        psi_edge = np.ones((n, n))
        psi_node = np.ones(n)

        fe = bethe_free_energy(
            node_beliefs=[b_i, b_i],
            edge_beliefs=[b_edge],
            node_potentials=[psi_node, psi_node],
            edge_potentials=[psi_edge],
            node_degrees=[1, 1],
        )
        assert math.isfinite(fe)

    def test_degree_zero_no_correction(self) -> None:
        """d_i = 1 means (d_i - 1) = 0: no variable entropy correction."""
        b_i = np.array([0.5, 0.5])
        b_edge = np.array([[0.25, 0.25], [0.25, 0.25]])
        psi_edge = np.ones((2, 2))

        fe = bethe_free_energy(
            node_beliefs=[b_i],
            edge_beliefs=[b_edge],
            node_potentials=[b_i],
            edge_potentials=[psi_edge],
            node_degrees=[1],
        )
        assert math.isfinite(fe)


# ------------------------------------------------------------------ #
# Annealing schedules
# ------------------------------------------------------------------ #


class TestAnnealingSchedules:
    """Tests for β annealing schedules."""

    def test_linear_endpoints(self) -> None:
        schedule = linear_annealing(0.1, 10.0, 50)
        assert schedule[0] == pytest.approx(0.1)
        assert schedule[-1] == pytest.approx(10.0)
        assert len(schedule) == 50

    def test_linear_monotone(self) -> None:
        schedule = linear_annealing(1.0, 100.0, 20)
        assert np.all(np.diff(schedule) >= 0)

    def test_exponential_endpoints(self) -> None:
        schedule = exponential_annealing(0.01, 100.0, 30)
        assert schedule[0] == pytest.approx(0.01, rel=1e-3)
        assert schedule[-1] == pytest.approx(100.0, rel=1e-3)
        assert len(schedule) == 30

    def test_exponential_monotone(self) -> None:
        schedule = exponential_annealing(0.1, 50.0, 20)
        assert np.all(np.diff(schedule) >= 0)

    def test_exponential_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            exponential_annealing(0.0, 10.0, 10)

    def test_cosine_endpoints(self) -> None:
        schedule = cosine_annealing(10.0, 0.1, 40)
        assert schedule[0] == pytest.approx(10.0, rel=1e-6)
        assert schedule[-1] == pytest.approx(0.1, rel=1e-6)
        assert len(schedule) == 40

    def test_cosine_smooth(self) -> None:
        """Cosine annealing should be smooth (no jumps > 2× step)."""
        schedule = cosine_annealing(10.0, 0.1, 100)
        diffs = np.abs(np.diff(schedule))
        assert np.max(diffs) < 0.5  # reasonable smoothness

    def test_all_schedules_positive(self) -> None:
        for sched in [
            linear_annealing(0.1, 10.0, 10),
            exponential_annealing(0.1, 10.0, 10),
            cosine_annealing(10.0, 0.1, 10),
        ]:
            assert np.all(sched >= 0)


# ------------------------------------------------------------------ #
# Phase transitions
# ------------------------------------------------------------------ #


class TestPhaseTransitions:
    """Tests for free energy landscape and phase transition detection."""

    def test_landscape_returns_points(self) -> None:
        prior = np.array([0.5, 0.5])
        rewards = np.array([0.0, 1.0])
        betas = np.linspace(0.1, 10.0, 20)
        points = free_energy_landscape(prior, rewards, betas)
        assert len(points) == 20

    def test_landscape_fe_increases_with_beta(self) -> None:
        """Free energy should generally increase with β."""
        prior = np.array([0.5, 0.5])
        rewards = np.array([0.0, 1.0])
        betas = np.linspace(0.1, 50.0, 50)
        points = free_energy_landscape(prior, rewards, betas)
        fes = [p.free_energy for p in points]
        # Should be broadly increasing (monotone for this simple case)
        assert fes[-1] > fes[0]

    def test_landscape_policy_valid(self) -> None:
        """Every policy in the landscape sums to 1."""
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        rewards = np.array([1.0, 2.0, 3.0])
        betas = [0.1, 1.0, 10.0]
        points = free_energy_landscape(prior, rewards, betas)
        for pt in points:
            assert np.sum(pt.policy) == pytest.approx(1.0, abs=1e-8)

    def test_find_phase_transitions_returns_list(self) -> None:
        prior = np.array([0.5, 0.5])
        rewards = np.array([0.0, 1.0])
        transitions = find_phase_transitions(prior, rewards)
        assert isinstance(transitions, list)

    def test_high_beta_policy_deterministic(self) -> None:
        """At very high β, optimal policy is nearly deterministic."""
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        rewards = np.array([0.0, 1.0, 2.0, 5.0])
        pi = optimal_policy(prior, rewards, beta=200.0)
        assert np.max(pi) > 0.99


# ------------------------------------------------------------------ #
# Bounded-rational helpers
# ------------------------------------------------------------------ #


class TestBoundedRationalHelpers:
    """Tests for bounded_rational_value and information_cost."""

    def test_bounded_rational_value_equals_optimal_fe(self) -> None:
        prior = np.array([0.5, 0.5])
        qvals = np.array([1.0, 2.0])
        beta = 3.0
        assert bounded_rational_value(prior, qvals, beta) == pytest.approx(
            optimal_free_energy(prior, qvals, beta), rel=1e-9,
        )

    def test_information_cost_zero_for_prior(self) -> None:
        """Cost = 0 when π = p₀."""
        prior = np.array([0.5, 0.5])
        assert information_cost(prior, prior, beta=5.0) == pytest.approx(0.0, abs=1e-12)

    def test_information_cost_positive(self) -> None:
        prior = np.array([0.5, 0.5])
        pi = np.array([0.9, 0.1])
        assert information_cost(pi, prior, beta=5.0) > 0

    def test_information_cost_zero_beta(self) -> None:
        prior = np.array([0.5, 0.5])
        pi = np.array([0.9, 0.1])
        assert information_cost(pi, prior, beta=0.0) == pytest.approx(0.0)
