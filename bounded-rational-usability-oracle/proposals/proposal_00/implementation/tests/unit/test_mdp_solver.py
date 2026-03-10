"""Unit tests for usability_oracle.mdp.solver — ValueIterationSolver, PolicyIterationSolver.

Validates classical MDP solvers: convergence of value iteration and policy
iteration, optimality of extracted policies, agreement between the two
algorithms, and behaviour under custom parameters.
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.mdp.solver import ValueIterationSolver, PolicyIterationSolver
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


def _make_simple_three_state_mdp() -> MDP:
    """start --[a, cost=1]--> middle --[b, cost=2]--> goal."""
    states = {
        "start": State(state_id="start", is_terminal=False, is_goal=False),
        "middle": State(state_id="middle", is_terminal=False, is_goal=False),
        "goal": State(state_id="goal", is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="m"),
        "b": Action(action_id="b", action_type="click", target_node_id="g"),
    }
    transitions = [
        Transition(source="start", action="a", target="middle",
                   probability=1.0, cost=1.0),
        Transition(source="middle", action="b", target="goal",
                   probability=1.0, cost=2.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


# ═══════════════════════════════════════════════════════════════════════════
# ValueIterationSolver tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValueIterationSolver:
    """Tests for ValueIterationSolver — Bellman-backup based solver."""

    def test_solve_returns_tuple(self):
        """solve() returns a (values, policy) tuple."""
        solver = ValueIterationSolver()
        mdp = make_two_state_mdp()
        result = solver.solve(mdp)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_dict(self):
        """Values should be a dict mapping state IDs to floats."""
        solver = ValueIterationSolver()
        values, _ = solver.solve(make_two_state_mdp())
        assert isinstance(values, dict)
        for v in values.values():
            assert isinstance(v, float)

    def test_policy_dict(self):
        """Policy should map non-terminal state IDs to action IDs."""
        solver = ValueIterationSolver()
        _, policy = solver.solve(make_two_state_mdp())
        assert isinstance(policy, dict)
        assert "start" in policy

    def test_terminal_state_value_zero(self):
        """Terminal / goal states should have value 0."""
        solver = ValueIterationSolver()
        values, _ = solver.solve(make_two_state_mdp())
        assert values["goal"] == 0.0

    def test_three_state_value_ordering(self):
        """V(start) > V(middle) > V(goal) = 0 for a chain with positive costs."""
        solver = ValueIterationSolver()
        mdp = _make_simple_three_state_mdp()
        values, _ = solver.solve(mdp)
        assert values["start"] > values["middle"]
        assert values["middle"] > 0.0
        assert values["goal"] == 0.0

    def test_optimal_policy_selects_only_action(self):
        """In a two-state MDP the policy must select the only action."""
        solver = ValueIterationSolver()
        _, policy = solver.solve(make_two_state_mdp())
        assert policy["start"] == "go"

    def test_choice_mdp_selects_cheapest(self):
        """In a choice MDP the optimal policy should pick the lowest-cost action."""
        mdp = make_choice_mdp(n_choices=5)
        solver = ValueIterationSolver()
        _, policy = solver.solve(mdp)
        # choice_0 has cost 0.3, choice_1 has 0.4, …
        assert policy["start"] == "choice_0"

    def test_custom_epsilon(self):
        """Solver accepts a custom epsilon without error."""
        solver = ValueIterationSolver()
        values, policy = solver.solve(make_two_state_mdp(), epsilon=1e-10)
        assert "start" in values

    def test_custom_max_iter(self):
        """Solver respects max_iter (may not converge with max_iter=1)."""
        solver = ValueIterationSolver()
        values, _ = solver.solve(make_two_state_mdp(), max_iter=1)
        assert isinstance(values, dict)

    def test_custom_discount_override(self):
        """Explicit discount overrides mdp.discount."""
        solver = ValueIterationSolver()
        mdp = _make_simple_three_state_mdp()
        v1, _ = solver.solve(mdp, discount=0.5)
        v2, _ = solver.solve(mdp, discount=0.99)
        # Lower discount → future costs matter less → lower start value
        assert v1["start"] < v2["start"]

    def test_cyclic_mdp_converges(self):
        """Value iteration converges on the cyclic MDP."""
        solver = ValueIterationSolver()
        values, policy = solver.solve(make_cyclic_mdp())
        assert "s0" in values
        assert "s2" in policy

    def test_large_chain_values_monotone(self):
        """In a chain, values should be monotonically decreasing toward the goal."""
        solver = ValueIterationSolver()
        mdp = make_large_chain_mdp(n=10)
        values, _ = solver.solve(mdp)
        for i in range(9):
            assert values[f"s{i}"] >= values[f"s{i+1}"]

    def test_values_finite(self):
        """All values should be finite (no inf or nan)."""
        solver = ValueIterationSolver()
        values, _ = solver.solve(make_cyclic_mdp())
        for v in values.values():
            assert math.isfinite(v)

    def test_policy_covers_non_terminal(self):
        """Policy should have an entry for every non-terminal state with actions."""
        solver = ValueIterationSolver()
        mdp = make_cyclic_mdp()
        _, policy = solver.solve(mdp)
        for sid, state in mdp.states.items():
            if not state.is_terminal and not state.is_goal:
                if mdp.get_actions(sid):
                    assert sid in policy


# ═══════════════════════════════════════════════════════════════════════════
# PolicyIterationSolver tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyIterationSolver:
    """Tests for PolicyIterationSolver — Howard-style policy iteration."""

    def test_solve_returns_tuple(self):
        """solve() returns a (values, policy) tuple."""
        solver = PolicyIterationSolver()
        result = solver.solve(make_two_state_mdp())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_terminal_value_zero(self):
        """Terminal states should have value 0."""
        solver = PolicyIterationSolver()
        values, _ = solver.solve(make_two_state_mdp())
        assert values["goal"] == 0.0

    def test_agrees_with_vi(self):
        """Policy iteration should agree with value iteration within tolerance."""
        mdp = _make_simple_three_state_mdp()
        vi_values, vi_policy = ValueIterationSolver().solve(mdp)
        pi_values, pi_policy = PolicyIterationSolver().solve(mdp)
        for sid in mdp.states:
            assert vi_values[sid] == pytest.approx(pi_values[sid], abs=1e-4)

    def test_agrees_with_vi_cyclic(self):
        """PI and VI agree on the cyclic MDP."""
        mdp = make_cyclic_mdp()
        vi_values, _ = ValueIterationSolver().solve(mdp)
        pi_values, _ = PolicyIterationSolver().solve(mdp)
        for sid in mdp.states:
            assert vi_values[sid] == pytest.approx(pi_values[sid], abs=1e-4)

    def test_agrees_with_vi_choice(self):
        """PI and VI agree on the choice MDP."""
        mdp = make_choice_mdp(n_choices=5)
        vi_values, vi_policy = ValueIterationSolver().solve(mdp)
        pi_values, pi_policy = PolicyIterationSolver().solve(mdp)
        assert vi_policy.get("start") == pi_policy.get("start")

    def test_custom_epsilon(self):
        """Custom epsilon parameter is accepted."""
        solver = PolicyIterationSolver()
        values, _ = solver.solve(make_two_state_mdp(), epsilon=1e-12)
        assert "start" in values

    def test_custom_max_iter(self):
        """Custom max_iter parameter is accepted."""
        solver = PolicyIterationSolver()
        values, _ = solver.solve(make_two_state_mdp(), max_iter=5)
        assert isinstance(values, dict)

    def test_policy_selects_cheapest_action(self):
        """PI should also select the cheapest action in the choice MDP."""
        solver = PolicyIterationSolver()
        _, policy = solver.solve(make_choice_mdp(n_choices=4))
        assert policy["start"] == "choice_0"

    def test_large_chain_converges(self):
        """Policy iteration converges on a 20-state chain."""
        solver = PolicyIterationSolver()
        mdp = make_large_chain_mdp(n=20)
        values, policy = solver.solve(mdp)
        assert len(values) == 20
        assert values["s0"] > values["s19"]

    def test_values_finite(self):
        """All values from PI should be finite."""
        solver = PolicyIterationSolver()
        values, _ = solver.solve(make_cyclic_mdp())
        for v in values.values():
            assert math.isfinite(v)
