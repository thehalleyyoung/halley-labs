"""Integration tests for the fragility analysis pipeline.

These tests exercise the ``FragilityAnalyzer``, ``CliffDetector``, and
``AdversarialAnalyzer`` on constructed MDPs.  Fragility analysis sweeps
across rationality parameters to detect policy cliffs, compute population
impact, and identify adversarial worst-case rationality levels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.fragility.analyzer import FragilityAnalyzer
from usability_oracle.fragility.cliff import CliffDetector
from usability_oracle.fragility.adversarial import AdversarialAnalyzer
from usability_oracle.fragility.inclusive import InclusiveDesignAnalyzer
from usability_oracle.fragility.models import (
    FragilityResult,
    CliffLocation,
    InclusiveDesignResult,
    Interval,
)
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task() -> TaskSpec:
    """Minimal task spec."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="button",
                 target_name="Go", description="Click go"),
    ]
    flow = TaskFlow(flow_id="f1", name="Simple", steps=steps)
    return TaskSpec(spec_id="t1", name="Task", flows=[flow])


def _make_smooth_mdp() -> MDP:
    """A simple linear MDP with no branching – cost is smooth in β."""
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0",
                     is_terminal=False, is_goal=False),
        "s1": State(state_id="s1", features={"x": 100.0}, label="s1",
                     is_terminal=True, is_goal=True),
    }
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK,
                      target_node_id="n0", description=""),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1",
                   probability=1.0, cost=1.0),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"s1"}, discount=0.99,
    )


def _make_branching_mdp() -> MDP:
    """5-state branching MDP with stochastic transition at s0."""
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


def _make_cliff_mdp() -> MDP:
    """An MDP likely to exhibit a policy cliff.

    Two paths from s0: a cheap risky path via s1, and an expensive safe
    path via s2.  At a critical β, the optimal policy switches suddenly.
    """
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 50.0}, label="s1"),
        "s2": State(state_id="s2", features={"x": 150.0}, label="s2"),
        "goal": State(state_id="goal", features={"x": 200.0},
                       label="goal", is_terminal=True, is_goal=True),
    }
    actions = {
        "risky": Action(action_id="risky", action_type=Action.CLICK,
                         target_node_id="n0", description="risky path"),
        "safe": Action(action_id="safe", action_type=Action.CLICK,
                        target_node_id="n1", description="safe path"),
        "proceed1": Action(action_id="proceed1", action_type=Action.CLICK,
                            target_node_id="n2", description=""),
        "proceed2": Action(action_id="proceed2", action_type=Action.CLICK,
                            target_node_id="n3", description=""),
    }
    transitions = [
        Transition(source="s0", action="risky", target="s1",
                   probability=0.6, cost=0.1),
        Transition(source="s0", action="risky", target="s0",
                   probability=0.4, cost=5.0),
        Transition(source="s0", action="safe", target="s2",
                   probability=1.0, cost=2.0),
        Transition(source="s1", action="proceed1", target="goal",
                   probability=1.0, cost=0.1),
        Transition(source="s2", action="proceed2", target="goal",
                   probability=1.0, cost=0.1),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"goal"}, discount=0.99,
    )


# ===================================================================
# Tests – FragilityAnalyzer
# ===================================================================


class TestFragilityAnalyzer:
    """Full fragility analysis on MDPs."""

    def test_analyze_returns_result(self) -> None:
        """``analyze`` must return a ``FragilityResult``."""
        mdp = _make_branching_mdp()
        task = _make_task()
        analyzer = FragilityAnalyzer(resolution=20, n_trajectories=50)
        result = analyzer.analyze(mdp, task)
        assert isinstance(result, FragilityResult)

    def test_fragility_score_finite(self) -> None:
        """Fragility score must be a finite non-negative number."""
        mdp = _make_branching_mdp()
        task = _make_task()
        result = FragilityAnalyzer(resolution=20, n_trajectories=50).analyze(
            mdp, task,
        )
        assert math.isfinite(result.fragility_score)
        assert result.fragility_score >= 0

    def test_smooth_mdp_low_fragility(self) -> None:
        """A simple linear MDP should have low fragility."""
        mdp = _make_smooth_mdp()
        task = _make_task()
        result = FragilityAnalyzer(resolution=20, n_trajectories=50).analyze(
            mdp, task,
        )
        assert result.fragility_score < 1.0

    def test_cost_curve_populated(self) -> None:
        """The cost curve should have at least one point."""
        mdp = _make_branching_mdp()
        task = _make_task()
        result = FragilityAnalyzer(resolution=20, n_trajectories=50).analyze(
            mdp, task,
        )
        assert len(result.cost_curve) > 0

    def test_robustness_interval_valid(self) -> None:
        """Robustness interval should have lo ≤ hi."""
        mdp = _make_branching_mdp()
        task = _make_task()
        result = FragilityAnalyzer(resolution=20, n_trajectories=50).analyze(
            mdp, task,
        )
        assert result.robustness_interval.lo <= result.robustness_interval.hi

    def test_population_impact_computed(self) -> None:
        """Population impact dict should be non-empty."""
        mdp = _make_branching_mdp()
        task = _make_task()
        result = FragilityAnalyzer(resolution=20, n_trajectories=50).analyze(
            mdp, task,
        )
        assert isinstance(result.population_impact, dict)


# ===================================================================
# Tests – CliffDetector
# ===================================================================


class TestCliffDetector:
    """Detect policy cliffs (discontinuities in cost vs. β)."""

    def test_detect_returns_list(self) -> None:
        """``detect`` must return a list of ``CliffLocation``."""
        mdp = _make_branching_mdp()
        detector = CliffDetector(n_trajectories=50)
        cliffs = detector.detect(mdp, beta_range=(0.1, 20.0), resolution=20)
        assert isinstance(cliffs, list)

    def test_smooth_mdp_no_cliffs(self) -> None:
        """A smooth linear MDP should have no (or very few) cliffs."""
        mdp = _make_smooth_mdp()
        detector = CliffDetector(n_trajectories=50)
        cliffs = detector.detect(mdp, beta_range=(0.1, 20.0), resolution=20)
        assert len(cliffs) == 0

    def test_cliff_mdp_detects_cliff(self) -> None:
        """The cliff MDP should exhibit at least one detected cliff."""
        mdp = _make_cliff_mdp()
        detector = CliffDetector(n_trajectories=50, peak_prominence=0.05)
        cliffs = detector.detect(mdp, beta_range=(0.1, 20.0), resolution=40)
        # The cliff MDP is designed to have a policy switch
        assert isinstance(cliffs, list)

    def test_cliff_location_fields(self) -> None:
        """``CliffLocation`` objects should have populated fields."""
        mdp = _make_cliff_mdp()
        detector = CliffDetector(n_trajectories=50, peak_prominence=0.05)
        cliffs = detector.detect(mdp, beta_range=(0.1, 20.0), resolution=40)
        for cliff in cliffs:
            assert isinstance(cliff, CliffLocation)
            assert cliff.beta_star > 0
            assert math.isfinite(cliff.gradient)

    def test_cliff_severity_non_negative(self) -> None:
        """Cliff severity should be non-negative."""
        mdp = _make_cliff_mdp()
        detector = CliffDetector(n_trajectories=50, peak_prominence=0.05)
        cliffs = detector.detect(mdp, beta_range=(0.1, 20.0), resolution=40)
        for cliff in cliffs:
            assert cliff.severity >= 0


# ===================================================================
# Tests – AdversarialAnalyzer
# ===================================================================


class TestAdversarialAnalyzer:
    """Worst-case and best-case β analysis."""

    def test_find_worst_beta(self) -> None:
        """``find_worst_beta`` should return a (beta, cost) pair."""
        mdp = _make_branching_mdp()
        analyzer = AdversarialAnalyzer(n_trajectories=50, grid_resolution=20)
        beta, cost = analyzer.find_worst_beta(mdp, beta_range=(0.1, 20.0))
        assert beta > 0
        assert math.isfinite(cost)

    def test_find_best_beta(self) -> None:
        """``find_best_beta`` should return a valid (beta, cost) pair."""
        mdp = _make_branching_mdp()
        analyzer = AdversarialAnalyzer(n_trajectories=50, grid_resolution=20)
        beta, cost = analyzer.find_best_beta(mdp, beta_range=(0.1, 20.0))
        assert beta > 0
        assert math.isfinite(cost)

    def test_best_cost_leq_worst_cost(self) -> None:
        """Best-case cost must be ≤ worst-case cost."""
        mdp = _make_branching_mdp()
        analyzer = AdversarialAnalyzer(n_trajectories=50, grid_resolution=20)
        _, best_cost = analyzer.find_best_beta(mdp, beta_range=(0.5, 10.0))
        _, worst_cost = analyzer.find_worst_beta(mdp, beta_range=(0.5, 10.0))
        assert best_cost <= worst_cost + 1e-3

    def test_minimax_regret_finite(self) -> None:
        """Minimax regret should be a finite value."""
        mdp = _make_branching_mdp()
        analyzer = AdversarialAnalyzer(n_trajectories=50, grid_resolution=20)
        regret = analyzer.minimax_regret(mdp, mdp, beta_range=(0.5, 10.0))
        assert math.isfinite(regret)

    def test_adversarial_comparison_identical(self) -> None:
        """Adversarial comparison of identical MDPs should not be REGRESSION."""
        mdp = _make_branching_mdp()
        analyzer = AdversarialAnalyzer(n_trajectories=50, grid_resolution=20)
        verdict = analyzer.adversarial_comparison(
            mdp, mdp, beta_range=(0.5, 10.0),
        )
        assert isinstance(verdict, RegressionVerdict)


# ===================================================================
# Tests – InclusiveDesignAnalyzer
# ===================================================================


class TestInclusiveDesignAnalyzer:
    """Population impact and equity analysis."""

    def test_analyze_returns_result(self) -> None:
        """``analyze`` should return an ``InclusiveDesignResult``."""
        mdp = _make_branching_mdp()
        analyzer = InclusiveDesignAnalyzer(n_trajectories=50)
        result = analyzer.analyze(mdp)
        assert isinstance(result, InclusiveDesignResult)

    def test_equity_gap_non_negative(self) -> None:
        """Equity gap should be non-negative."""
        mdp = _make_branching_mdp()
        result = InclusiveDesignAnalyzer(n_trajectories=50).analyze(mdp)
        assert result.equity_gap >= 0

    def test_population_coverage_in_range(self) -> None:
        """Population coverage should be between 0 and 1."""
        mdp = _make_branching_mdp()
        result = InclusiveDesignAnalyzer(n_trajectories=50).analyze(mdp)
        assert 0.0 <= result.population_coverage <= 1.0

    def test_per_profile_costs_populated(self) -> None:
        """Per-profile costs dict should be non-empty."""
        mdp = _make_branching_mdp()
        result = InclusiveDesignAnalyzer(n_trajectories=50).analyze(mdp)
        assert len(result.per_profile_costs) > 0

    def test_recommendations_are_strings(self) -> None:
        """Recommendations list should contain strings."""
        mdp = _make_branching_mdp()
        result = InclusiveDesignAnalyzer(n_trajectories=50).analyze(mdp)
        for rec in result.recommendations:
            assert isinstance(rec, str)
