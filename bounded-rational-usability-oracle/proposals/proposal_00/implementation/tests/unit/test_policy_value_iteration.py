"""Unit tests for usability_oracle.policy.value_iteration — SoftValueIteration.

Validates the entropy-regularised (soft) value iteration solver: convergence,
policy structure, free-energy computation, the effect of β on policy
determinism, state-value ordering along optimal paths, and the returned
PolicyResult bundle.
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.models import Policy, QValues, PolicyResult
from usability_oracle.mdp.models import MDP, State, Action, Transition
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
    make_large_chain_mdp,
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


def _make_three_state_mdp() -> MDP:
    """start --[a, cost=1]--> mid --[b, cost=2]--> goal."""
    states = {
        "start": State(state_id="start"),
        "mid": State(state_id="mid"),
        "goal": State(state_id="goal", is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="m"),
        "b": Action(action_id="b", action_type="click", target_node_id="g"),
    }
    transitions = [
        Transition(source="start", action="a", target="mid",
                   probability=1.0, cost=1.0),
        Transition(source="mid", action="b", target="goal",
                   probability=1.0, cost=2.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


# ═══════════════════════════════════════════════════════════════════════════
# SoftValueIteration.solve tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSoftValueIterationSolve:
    """Tests for SoftValueIteration.solve() — the main entry point."""

    def test_solve_returns_policy_result(self):
        """solve() should return a PolicyResult instance."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        solver = SoftValueIteration()
        result = solver.solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result, PolicyResult)

    def test_result_has_policy(self):
        """The result should contain a Policy object."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result.policy, Policy)

    def test_result_has_q_values(self):
        """The result should contain a QValues object."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result.q_values, QValues)

    def test_result_has_state_values(self):
        """state_values should be a dict mapping state IDs to floats."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result.state_values, dict)
        assert "start" in result.state_values

    def test_result_has_free_energy(self):
        """free_energy should be a finite float."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result.free_energy, float)
        assert math.isfinite(result.free_energy)

    def test_convergence_info_present(self):
        """convergence_info dict should include 'converged' and 'iterations'."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert "converged" in result.convergence_info
        assert "iterations" in result.convergence_info

    def test_convergence_achieved(self):
        """On a simple MDP with sufficient iterations, solver should converge."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert result.convergence_info["converged"] is True

    def test_convergence_within_max_iter(self):
        """Solver should converge within the specified max_iter."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(
            mdp, beta=2.0, prior=prior, max_iter=5000,
        )
        assert result.convergence_info["iterations"] <= 5000

    def test_policy_probabilities_sum_to_one(self):
        """Result policy probabilities must sum to 1 for each state."""
        mdp = make_choice_mdp(n_choices=4)
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=3.0, prior=prior)
        for state, dist in result.policy.state_action_probs.items():
            total = sum(dist.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"State {state}: probabilities sum to {total}"
            )

    def test_terminal_state_value_zero(self):
        """Terminal/goal state values should be 0."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert result.state_values["goal"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# β effect on policy
# ═══════════════════════════════════════════════════════════════════════════


class TestBetaEffect:
    """Tests that β controls the rationality / determinism of the policy."""

    def test_high_beta_more_deterministic(self):
        """Higher β → more deterministic policy (lower entropy)."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)

        r_low = SoftValueIteration().solve(mdp, beta=0.5, prior=prior)
        r_high = SoftValueIteration().solve(mdp, beta=50.0, prior=prior)

        h_low = r_low.policy.mean_entropy()
        h_high = r_high.policy.mean_entropy()
        assert h_high < h_low

    def test_low_beta_more_exploratory(self):
        """Lower β → more uniform / exploratory policy."""
        mdp = make_choice_mdp(n_choices=4)
        prior = _uniform_prior(mdp)

        r_low = SoftValueIteration().solve(mdp, beta=0.1, prior=prior)

        # With very low β, entropy should be close to max
        max_entropy = r_low.policy.max_entropy("start")
        actual_entropy = r_low.policy.entropy("start")
        if max_entropy > 0:
            ratio = actual_entropy / max_entropy
            assert ratio > 0.8  # at least 80% of max entropy

    def test_very_high_beta_selects_best_action(self):
        """β → ∞ should concentrate on the optimal (lowest-cost) action."""
        mdp = make_choice_mdp(n_choices=5)
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=200.0, prior=prior)
        # choice_0 has lowest cost
        greedy = result.policy.greedy_action("start")
        assert greedy == "choice_0"


# ═══════════════════════════════════════════════════════════════════════════
# State value ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestStateValueOrdering:
    """Tests for monotonicity of state values along the optimal path."""

    def test_values_decrease_along_chain(self):
        """In a linear chain, V(s_i) >= V(s_{i+1}) toward the goal."""
        mdp = make_large_chain_mdp(n=10)
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=5.0, prior=prior)
        for i in range(9):
            assert result.state_values[f"s{i}"] >= result.state_values[f"s{i+1}"] - 1e-6

    def test_three_state_ordering(self):
        """V(start) > V(mid) > V(goal)=0 in a three-state chain."""
        mdp = _make_three_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=5.0, prior=prior)
        assert result.state_values["start"] > result.state_values["mid"]
        assert result.state_values["mid"] > result.state_values["goal"]

    def test_goal_value_zero(self):
        """Goal states should always have value 0."""
        mdp = _make_three_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=5.0, prior=prior)
        assert result.state_values["goal"] == 0.0

    def test_all_values_finite(self):
        """All state values should be finite."""
        mdp = make_cyclic_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        for v in result.state_values.values():
            assert math.isfinite(v)


# ═══════════════════════════════════════════════════════════════════════════
# Free energy and Q-values
# ═══════════════════════════════════════════════════════════════════════════


class TestFreeEnergyAndQValues:
    """Tests for the free_energy and q_values fields of the result."""

    def test_free_energy_finite(self):
        """Result free_energy should be finite."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert math.isfinite(result.free_energy)

    def test_free_energy_positive_for_positive_costs(self):
        """Free energy should be > 0 for MDPs with positive costs."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert result.free_energy >= 0.0

    def test_q_values_populated(self):
        """Q-values should be populated for non-terminal states."""
        mdp = make_choice_mdp(n_choices=3)
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert "start" in result.q_values.values

    def test_q_values_best_action(self):
        """Q-values best_action should select the lowest-cost action."""
        mdp = make_choice_mdp(n_choices=5)
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=10.0, prior=prior)
        best = result.q_values.best_action("start")
        assert best == "choice_0"

    def test_cyclic_mdp_solves(self):
        """Soft VI should handle cyclic MDPs without errors."""
        mdp = make_cyclic_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=2.0, prior=prior)
        assert isinstance(result, PolicyResult)
        assert math.isfinite(result.free_energy)


# ═══════════════════════════════════════════════════════════════════════════
# Custom parameters
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomParameters:
    """Tests for custom discount, epsilon, and max_iter parameters."""

    def test_custom_discount(self):
        """Explicit discount should override mdp.discount."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        r1 = SoftValueIteration().solve(mdp, beta=2.0, prior=prior, discount=0.5)
        r2 = SoftValueIteration().solve(mdp, beta=2.0, prior=prior, discount=0.99)
        assert r1.convergence_info["discount"] == 0.5
        assert r2.convergence_info["discount"] == 0.99

    def test_custom_epsilon(self):
        """Custom epsilon affects convergence precision."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(
            mdp, beta=2.0, prior=prior, epsilon=1e-10,
        )
        assert result.convergence_info["residual"] < 1e-10

    def test_custom_max_iter(self):
        """max_iter limits the number of iterations."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(
            mdp, beta=2.0, prior=prior, max_iter=2,
        )
        assert result.convergence_info["iterations"] <= 2

    def test_beta_stored_in_convergence_info(self):
        """Beta should be stored in convergence_info."""
        mdp = make_two_state_mdp()
        prior = _uniform_prior(mdp)
        result = SoftValueIteration().solve(mdp, beta=7.5, prior=prior)
        assert result.convergence_info["beta"] == 7.5
