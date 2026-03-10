"""Integration tests for the Layer-2 (bounded-rational MDP) pipeline.

Layer 2 builds an MDP from an accessibility tree + task spec, solves it
with soft value iteration at a given rationality parameter β, samples
trajectories, computes total expected cost, and compares two MDPs at a
fixed β.  These tests exercise the full L2 path and verify numerical
correctness of the intermediate artefacts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.mdp.trajectory import TrajectorySampler, Trajectory, TrajectoryStats
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.models import Policy, PolicyResult, QValues
from usability_oracle.comparison.paired import PairedComparator
from usability_oracle.comparison.models import ComparisonResult, AlignmentResult, StateMapping
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec
from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer

from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _load_html(name: str) -> str:
    return (SAMPLE_HTML_DIR / f"{name}.html").read_text()


def _parse(html: str):
    tree = HTMLAccessibilityParser().parse(html)
    tree = AccessibilityNormalizer().normalize(tree)
    tree.root.properties['tabindex'] = '0'
    return tree


def _make_task() -> TaskSpec:
    """Minimal login task."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Submit", description="Submit",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="f1", name="Login", steps=steps,
                    success_criteria=["done"])
    return TaskSpec(spec_id="t1", name="Login", flows=[flow])


def _make_branching_mdp() -> MDP:
    """5-state MDP with stochastic branching at s0."""
    states = {}
    for i in range(5):
        sid = f"s{i}"
        states[sid] = State(
            state_id=sid,
            features={"x": float(i * 50), "y": float(i * 30)},
            label=sid,
            is_terminal=(i == 4),
            is_goal=(i == 4),
        )
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK,
                      target_node_id="n0", description=""),
        "a1": Action(action_id="a1", action_type=Action.CLICK,
                      target_node_id="n1", description=""),
        "a2": Action(action_id="a2", action_type=Action.CLICK,
                      target_node_id="n2", description=""),
        "a3": Action(action_id="a3", action_type=Action.CLICK,
                      target_node_id="n3", description=""),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1",
                   probability=0.7, cost=0.3),
        Transition(source="s0", action="a0", target="s2",
                   probability=0.3, cost=0.5),
        Transition(source="s1", action="a1", target="s3",
                   probability=1.0, cost=0.4),
        Transition(source="s2", action="a2", target="s3",
                   probability=1.0, cost=0.6),
        Transition(source="s3", action="a3", target="s4",
                   probability=1.0, cost=0.2),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"s4"}, discount=0.99,
    )


# ===================================================================
# Tests – MDP construction from HTML
# ===================================================================


class TestMDPBuild:
    """Build MDPs from parsed accessibility trees."""

    def test_build_from_simple_form(self) -> None:
        """MDP from simple_form should have states and transitions."""
        tree = _parse(_load_html("simple_form"))
        mdp = MDPBuilder().build(tree, _make_task())
        assert mdp.n_states > 0
        assert mdp.n_transitions > 0

    def test_build_initial_state_valid(self) -> None:
        """Initial state must exist in the state dict."""
        tree = _parse(_load_html("simple_form"))
        mdp = MDPBuilder().build(tree, _make_task())
        assert mdp.initial_state in mdp.states

    def test_build_goal_states_exist(self) -> None:
        """All goal states must be present in states dict."""
        tree = _parse(_load_html("simple_form"))
        mdp = MDPBuilder().build(tree, _make_task())
        for g in mdp.goal_states:
            assert g in mdp.states

    def test_build_validates(self) -> None:
        """Built MDP should pass its own validation."""
        tree = _parse(_load_html("simple_form"))
        mdp = MDPBuilder().build(tree, _make_task())
        errors = mdp.validate()
        assert len(errors) == 0, f"MDP validation errors: {errors}"


# ===================================================================
# Tests – Soft value iteration
# ===================================================================


class TestSoftValueIteration:
    """Solve MDPs with bounded-rational (soft) value iteration."""

    def test_solve_branching_mdp(self) -> None:
        """Soft VI on the branching MDP should converge."""
        mdp = _make_branching_mdp()
        prior = Policy(
            state_action_probs={
                "s0": {"a0": 1.0},
                "s1": {"a1": 1.0},
                "s2": {"a2": 1.0},
                "s3": {"a3": 1.0},
            },
            beta=1.0,
        )
        svi = SoftValueIteration()
        result: PolicyResult = svi.solve(mdp, beta=2.0, prior=prior)
        assert result.policy is not None
        assert result.q_values is not None

    def test_high_beta_approaches_optimal(self) -> None:
        """At high β the soft policy should approach optimal VI policy."""
        mdp = _make_branching_mdp()
        prior = Policy(
            state_action_probs={
                "s0": {"a0": 1.0},
                "s1": {"a1": 1.0},
                "s2": {"a2": 1.0},
                "s3": {"a3": 1.0},
            },
            beta=1.0,
        )
        svi = SoftValueIteration()
        result = svi.solve(mdp, beta=100.0, prior=prior)
        # At high β, the policy should be nearly deterministic
        for state in ["s0", "s1", "s2", "s3"]:
            probs = result.policy.action_probs(state)
            max_prob = max(probs.values()) if probs else 0
            assert max_prob > 0.8

    def test_low_beta_increases_entropy(self) -> None:
        """At low β the policy should have higher entropy than at high β."""
        mdp = _make_branching_mdp()
        prior = Policy(
            state_action_probs={
                "s0": {"a0": 1.0},
                "s1": {"a1": 1.0},
                "s2": {"a2": 1.0},
                "s3": {"a3": 1.0},
            },
            beta=1.0,
        )
        svi = SoftValueIteration()
        low = svi.solve(mdp, beta=0.5, prior=prior)
        high = svi.solve(mdp, beta=50.0, prior=prior)
        assert low.policy.mean_entropy() >= high.policy.mean_entropy()

    def test_state_values_non_negative(self) -> None:
        """All state values should be non-negative (cost domain)."""
        mdp = _make_branching_mdp()
        values, _ = ValueIterationSolver().solve(mdp)
        for v in values.values():
            assert v >= -1e-6


# ===================================================================
# Tests – Trajectory sampling
# ===================================================================


class TestTrajectorySampling:
    """Sample trajectories from solved MDPs."""

    def test_sample_from_branching_mdp(self) -> None:
        """Sampling trajectories must return a non-empty list."""
        mdp = _make_branching_mdp()
        _, policy = ValueIterationSolver().solve(mdp)
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        trajectories = sampler.sample(mdp, policy, n_trajectories=50)
        assert len(trajectories) == 50

    def test_trajectories_reach_goal(self) -> None:
        """Most trajectories should reach the goal state."""
        mdp = _make_branching_mdp()
        _, policy = ValueIterationSolver().solve(mdp)
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        trajectories = sampler.sample(mdp, policy, n_trajectories=100)
        reached = sum(1 for t in trajectories if t.reached_goal)
        assert reached > 50

    def test_trajectory_stats_mean_cost(self) -> None:
        """Mean trajectory cost should be finite and positive."""
        mdp = _make_branching_mdp()
        _, policy = ValueIterationSolver().solve(mdp)
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        trajectories = sampler.sample(mdp, policy, n_trajectories=200)
        stats: TrajectoryStats = TrajectorySampler.trajectory_statistics(
            trajectories,
        )
        assert math.isfinite(stats.mean_cost)
        assert stats.mean_cost >= 0

    def test_trajectory_step_costs_positive(self) -> None:
        """Every step cost within a trajectory should be non-negative."""
        mdp = _make_branching_mdp()
        _, policy = ValueIterationSolver().solve(mdp)
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        trajectories = sampler.sample(mdp, policy, n_trajectories=10)
        for traj in trajectories:
            for step in traj.steps:
                assert step.cost >= 0


# ===================================================================
# Tests – Paired comparison at fixed β
# ===================================================================


class TestLayer2Comparison:
    """Compare two MDPs at a fixed rationality parameter."""

    def test_identical_mdps_neutral(self) -> None:
        """Comparing identical MDPs must yield NEUTRAL / INCONCLUSIVE."""
        mdp = _make_branching_mdp()
        identity_map = AlignmentResult(
            mappings=[StateMapping(state_a=s, state_b=s) for s in mdp.states],
        )
        task = _make_task()
        result = PairedComparator(beta=2.0).compare(
            mdp_a=mdp, mdp_b=mdp, alignment=identity_map, task=task,
        )
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_comparison_returns_costs(self) -> None:
        """Paired comparison must populate cost_before and cost_after."""
        mdp = _make_branching_mdp()
        identity_map = AlignmentResult(
            mappings=[StateMapping(state_a=s, state_b=s) for s in mdp.states],
        )
        task = _make_task()
        result = PairedComparator(beta=2.0).compare(
            mdp_a=mdp, mdp_b=mdp, alignment=identity_map, task=task,
        )
        assert result.cost_before is not None
        assert result.cost_after is not None

    def test_comparison_effect_size_near_zero_for_identical(self) -> None:
        """Identical MDPs should produce near-zero effect size."""
        mdp = _make_branching_mdp()
        identity_map = AlignmentResult(
            mappings=[StateMapping(state_a=s, state_b=s) for s in mdp.states],
        )
        task = _make_task()
        result = PairedComparator(beta=2.0).compare(
            mdp_a=mdp, mdp_b=mdp, alignment=identity_map, task=task,
        )
        assert abs(result.effect_size) < 1.0

    def test_comparison_confidence_in_range(self) -> None:
        """Confidence should be between 0 and 1."""
        mdp = _make_branching_mdp()
        identity_map = AlignmentResult(
            mappings=[StateMapping(state_a=s, state_b=s) for s in mdp.states],
        )
        task = _make_task()
        result = PairedComparator(beta=2.0).compare(
            mdp_a=mdp, mdp_b=mdp, alignment=identity_map, task=task,
        )
        assert 0.0 <= result.confidence <= 1.0


class TestSoftmaxPolicyIntegration:
    """Verify SoftmaxPolicy helpers with solved MDPs."""

    def test_beta_sweep(self) -> None:
        """``beta_sweep`` must produce one policy per beta value."""
        mdp = _make_branching_mdp()
        _, raw_policy = ValueIterationSolver().solve(mdp)
        q = QValues(values={s: {a: 0.5 for a in mdp.get_actions(s)}
                            for s in mdp.states if not mdp.states[s].is_terminal})
        betas = [0.5, 1.0, 5.0, 10.0]
        policies = SoftmaxPolicy.beta_sweep(q, betas)
        assert len(policies) == len(betas)

    def test_softmax_from_q_values(self) -> None:
        """``from_q_values`` must return a valid Policy."""
        q = QValues(values={"s0": {"a0": 1.0}, "s1": {"a1": 0.5}})
        policy = SoftmaxPolicy.from_q_values(q, beta=2.0)
        assert policy.n_states() >= 1
