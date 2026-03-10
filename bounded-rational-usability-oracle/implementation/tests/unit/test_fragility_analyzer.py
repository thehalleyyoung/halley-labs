"""Unit tests for usability_oracle.fragility.analyzer.FragilityAnalyzer.

Tests the fragility analysis pipeline that evaluates how sensitive
a UI's usability cost is to changes in the user rationality parameter beta.
Covers cost-curve computation, fragility scoring, population impact,
robustness intervals, and resolution-dependent granularity.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from usability_oracle.fragility.analyzer import FragilityAnalyzer
from usability_oracle.fragility.models import FragilityResult, Interval
from usability_oracle.mdp.models import MDP, Action, State, Transition
from usability_oracle.taskspec.models import TaskFlow, TaskSpec, TaskStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_mdp() -> MDP:
    """Build a two-state MDP: start → goal with one action."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="goal", is_terminal=True, is_goal=True)
    a = Action(action_id="a_go", action_type="click", target_node_id="btn")
    t = Transition(source="s0", action="a_go", target="s1", probability=1.0, cost=1.0)
    mdp = MDP(
        states={"s0": s0, "s1": s1},
        actions={"a_go": a},
        transitions=[t],
        initial_state="s0",
        goal_states={"s1"},
        discount=0.99,
    )
    return mdp


def _make_branching_mdp() -> MDP:
    """Three-state MDP with two actions from the start state."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="mid")
    s2 = State(state_id="s2", label="goal", is_terminal=True, is_goal=True)
    a1 = Action(action_id="a1", action_type="click", target_node_id="n1")
    a2 = Action(action_id="a2", action_type="click", target_node_id="n2")
    a3 = Action(action_id="a3", action_type="click", target_node_id="n3")
    transitions = [
        Transition(source="s0", action="a1", target="s1", probability=1.0, cost=2.0),
        Transition(source="s0", action="a2", target="s2", probability=1.0, cost=5.0),
        Transition(source="s1", action="a3", target="s2", probability=1.0, cost=1.0),
    ]
    return MDP(
        states={"s0": s0, "s1": s1, "s2": s2},
        actions={"a1": a1, "a2": a2, "a3": a3},
        transitions=transitions,
        initial_state="s0",
        goal_states={"s2"},
        discount=0.99,
    )


def _make_simple_task() -> TaskSpec:
    """Build a minimal TaskSpec with one flow and one step."""
    step = TaskStep(
        step_id="step_click",
        action_type="click",
        target_role="button",
        target_name="Submit",
    )
    flow = TaskFlow(flow_id="f1", name="submit_flow", steps=[step])
    return TaskSpec(spec_id="ts1", name="submit_task", description="Click submit", flows=[flow])


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestFragilityAnalyzerConstruction:
    """Tests for FragilityAnalyzer.__init__ and defaults."""

    def test_default_construction(self):
        """FragilityAnalyzer can be created with default parameters."""
        analyzer = FragilityAnalyzer()
        assert analyzer is not None

    def test_custom_resolution(self):
        """Resolution parameter is stored and accessible."""
        analyzer = FragilityAnalyzer(resolution=50)
        assert analyzer.resolution == 50

    def test_custom_n_trajectories(self):
        """n_trajectories parameter is stored."""
        analyzer = FragilityAnalyzer(n_trajectories=200)
        assert analyzer.n_trajectories == 200

    def test_custom_population_betas(self):
        """Custom population betas override defaults."""
        custom = {"novice": 1.0, "expert": 10.0}
        analyzer = FragilityAnalyzer(population_betas=custom)
        assert analyzer.population_betas == custom

    def test_default_population_betas_are_dict(self):
        """DEFAULT_POPULATION_BETAS is a non-empty dictionary."""
        assert isinstance(FragilityAnalyzer.DEFAULT_POPULATION_BETAS, dict)
        assert len(FragilityAnalyzer.DEFAULT_POPULATION_BETAS) > 0

    def test_default_population_betas_keys(self):
        """Default population betas include known percentile keys."""
        betas = FragilityAnalyzer.DEFAULT_POPULATION_BETAS
        assert "p50_average" in betas
        assert "p05_impaired" in betas
        assert "p95_expert" in betas

    def test_default_population_betas_values_positive(self):
        """All default population beta values are positive floats."""
        for key, val in FragilityAnalyzer.DEFAULT_POPULATION_BETAS.items():
            assert val > 0, f"Beta for {key} must be positive, got {val}"

    def test_default_population_betas_ordered(self):
        """Impaired beta < novice beta < average < experienced < expert."""
        b = FragilityAnalyzer.DEFAULT_POPULATION_BETAS
        assert b["p05_impaired"] < b["p25_novice"]
        assert b["p25_novice"] < b["p50_average"]
        assert b["p50_average"] < b["p75_experienced"]
        assert b["p75_experienced"] < b["p95_expert"]


# ---------------------------------------------------------------------------
# Analyze method tests (mocked solver)
# ---------------------------------------------------------------------------

class TestFragilityAnalyzerAnalyze:
    """Tests for FragilityAnalyzer.analyze with mocked cost evaluation."""

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_analyze_returns_fragility_result(self, mock_cost):
        """analyze() returns a FragilityResult dataclass."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        mdp = _make_minimal_mdp()
        task = _make_simple_task()
        result = analyzer.analyze(mdp, task)
        assert isinstance(result, FragilityResult)

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_fragility_score_in_unit_interval(self, mock_cost):
        """Fragility score must lie in [0, 1]."""
        mock_cost.return_value = 2.5
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert 0.0 <= result.fragility_score <= 1.0

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_constant_cost_gives_low_fragility(self, mock_cost):
        """A flat cost curve (constant across beta) should produce low fragility."""
        mock_cost.return_value = 3.0
        analyzer = FragilityAnalyzer(resolution=20, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert result.fragility_score < 0.3

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_cost_curve_populated(self, mock_cost):
        """The returned cost_curve list is non-empty after analysis."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert len(result.cost_curve) > 0

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_cost_curve_length_matches_resolution(self, mock_cost):
        """Cost curve length should match the resolution parameter."""
        mock_cost.return_value = 1.0
        res = 15
        analyzer = FragilityAnalyzer(resolution=res, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert len(result.cost_curve) == res

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_cost_curve_beta_ordering(self, mock_cost):
        """Beta values in the cost curve should be monotonically increasing."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=20, n_trajectories=5)
        result = analyzer.analyze(
            _make_minimal_mdp(), _make_simple_task(), beta_range=(0.5, 10.0)
        )
        betas = [pt[0] for pt in result.cost_curve]
        assert betas == sorted(betas)

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_custom_beta_range(self, mock_cost):
        """Custom beta_range is respected; curve spans the given interval."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(
            _make_minimal_mdp(), _make_simple_task(), beta_range=(1.0, 5.0)
        )
        betas = [pt[0] for pt in result.cost_curve]
        assert betas[0] >= 1.0 - 0.01
        assert betas[-1] <= 5.0 + 0.01

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_population_impact_keys_match(self, mock_cost):
        """Population impact dict keys match the population_betas keys."""
        mock_cost.return_value = 2.0
        custom = {"low": 0.5, "mid": 3.0, "high": 12.0}
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5, population_betas=custom)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert set(result.population_impact.keys()) == set(custom.keys())

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_population_impact_values_nonnegative(self, mock_cost):
        """All population impact cost values must be >= 0."""
        mock_cost.return_value = 1.5
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        for profile, cost in result.population_impact.items():
            assert cost >= 0, f"Cost for {profile} is negative: {cost}"

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_robustness_interval_type(self, mock_cost):
        """robustness_interval should be an Interval instance."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert isinstance(result.robustness_interval, Interval)

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_robustness_interval_nonnegative_width(self, mock_cost):
        """Robustness interval width must be >= 0."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=10, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert result.robustness_interval.width >= 0

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_constant_curve_wide_robustness(self, mock_cost):
        """Flat cost curve ⇒ robustness interval should span most of beta range."""
        mock_cost.return_value = 4.0
        analyzer = FragilityAnalyzer(resolution=20, n_trajectories=5)
        result = analyzer.analyze(
            _make_minimal_mdp(), _make_simple_task(), beta_range=(0.1, 20.0)
        )
        assert result.robustness_interval.width > 5.0


# ---------------------------------------------------------------------------
# Resolution parameter tests
# ---------------------------------------------------------------------------

class TestResolutionAffectsGranularity:
    """Tests that resolution controls granularity of the cost curve."""

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_higher_resolution_more_points(self, mock_cost):
        """Higher resolution produces more cost-curve sample points."""
        mock_cost.return_value = 1.0
        lo = FragilityAnalyzer(resolution=10, n_trajectories=5)
        hi = FragilityAnalyzer(resolution=50, n_trajectories=5)
        mdp, task = _make_minimal_mdp(), _make_simple_task()
        r_lo = lo.analyze(mdp, task)
        r_hi = hi.analyze(mdp, task)
        assert len(r_hi.cost_curve) > len(r_lo.cost_curve)

    @patch("usability_oracle.fragility.analyzer._solve_and_cost")
    def test_resolution_one(self, mock_cost):
        """Resolution of 1 is too small for gradient computation; use 2."""
        mock_cost.return_value = 1.0
        analyzer = FragilityAnalyzer(resolution=2, n_trajectories=5)
        result = analyzer.analyze(_make_minimal_mdp(), _make_simple_task())
        assert len(result.cost_curve) == 2


# ---------------------------------------------------------------------------
# FragilityResult property tests
# ---------------------------------------------------------------------------

class TestFragilityResultProperties:
    """Tests for FragilityResult dataclass properties."""

    def test_is_fragile_true(self):
        """is_fragile is True when fragility_score > 0.5."""
        r = FragilityResult(fragility_score=0.8)
        assert r.is_fragile is True

    def test_is_fragile_false(self):
        """is_fragile is False when fragility_score <= 0.5."""
        r = FragilityResult(fragility_score=0.3)
        assert r.is_fragile is False

    def test_n_cliffs_zero(self):
        """n_cliffs is 0 when cliff_locations is empty."""
        r = FragilityResult()
        assert r.n_cliffs == 0

    def test_worst_cliff_none(self):
        """worst_cliff is None when no cliffs exist."""
        r = FragilityResult()
        assert r.worst_cliff is None

    def test_default_fragility_score_zero(self):
        """Default fragility_score is 0."""
        r = FragilityResult()
        assert r.fragility_score == 0.0


# ---------------------------------------------------------------------------
# Interval model tests
# ---------------------------------------------------------------------------

class TestIntervalModel:
    """Tests for the Interval dataclass used in robustness results."""

    def test_width(self):
        """Interval width is hi - lo."""
        iv = Interval(lo=1.0, hi=5.0)
        assert iv.width == pytest.approx(4.0)

    def test_mid(self):
        """Interval midpoint."""
        iv = Interval(lo=2.0, hi=8.0)
        assert iv.mid == pytest.approx(5.0)

    def test_contains_true(self):
        """contains returns True for a point inside."""
        iv = Interval(lo=0.0, hi=10.0)
        assert iv.contains(5.0)

    def test_contains_false(self):
        """contains returns False for a point outside."""
        iv = Interval(lo=0.0, hi=10.0)
        assert not iv.contains(11.0)

    def test_degenerate_interval(self):
        """Degenerate interval (lo == hi) has width 0."""
        iv = Interval(lo=3.0, hi=3.0)
        assert iv.width == 0.0
