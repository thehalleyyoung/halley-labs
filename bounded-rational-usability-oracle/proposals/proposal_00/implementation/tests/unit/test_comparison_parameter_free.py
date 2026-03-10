"""Unit tests for usability_oracle.comparison.parameter_free.

Tests ParameterFreeComparator and BetaInterval for parameter-free
regression detection across all β values in a specified range.

References
----------
- Ortega & Braun (2013). *Proc. R. Soc. A*, 469.
- Moore, Kearfott & Cloud (2009). *Intro to Interval Analysis*. SIAM.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from usability_oracle.comparison.parameter_free import (
    BetaInterval,
    ParameterFreeComparator,
)
from usability_oracle.comparison.models import (
    AlignmentResult,
    ComparisonResult,
    StateMapping,
)
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.core.errors import ComparisonError
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.taskspec.models import TaskSpec

from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
)
from tests.fixtures.sample_tasks import make_login_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_alignment(mdp: MDP) -> AlignmentResult:
    """Build an identity alignment mapping every state to itself."""
    mappings = [
        StateMapping(state_a=sid, state_b=sid, similarity=1.0)
        for sid in mdp.states
    ]
    return AlignmentResult(mappings=mappings, overall_similarity=1.0)


def _make_higher_cost_mdp() -> MDP:
    """Create a 2-state MDP with cost 5.0 (vs 1.0 baseline)."""
    states = {
        "start": State(state_id="start", features={"x": 0.0}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "mid": State(state_id="mid", features={"x": 0.5}, label="mid",
                     is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="g", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="go", target="mid",
                   probability=0.5, cost=4.0),
        Transition(source="start", action="go", target="goal",
                   probability=0.5, cost=6.0),
        Transition(source="mid", action="go", target="goal",
                   probability=1.0, cost=5.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


def _make_lower_cost_mdp() -> MDP:
    """Create a 2-state MDP with cost 0.1 (vs 1.0 baseline)."""
    states = {
        "start": State(state_id="start", features={"x": 0.0}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "mid": State(state_id="mid", features={"x": 0.5}, label="mid",
                     is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="g", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="go", target="mid",
                   probability=0.5, cost=0.05),
        Transition(source="start", action="go", target="goal",
                   probability=0.5, cost=0.30),
        Transition(source="mid", action="go", target="goal",
                   probability=1.0, cost=0.02),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


# ---------------------------------------------------------------------------
# Tests: BetaInterval
# ---------------------------------------------------------------------------


class TestBetaInterval:
    """Tests for BetaInterval — a closed interval [lo, hi] for β values."""

    def test_basic_fields(self):
        """BetaInterval should store lo and hi endpoints."""
        bi = BetaInterval(lo=0.1, hi=10.0)
        assert bi.lo == 0.1
        assert bi.hi == 10.0

    def test_mid(self):
        """mid should return (lo + hi) / 2."""
        bi = BetaInterval(lo=1.0, hi=5.0)
        assert bi.mid == pytest.approx(3.0)

    def test_width(self):
        """width should return hi - lo."""
        bi = BetaInterval(lo=2.0, hi=8.0)
        assert bi.width == pytest.approx(6.0)

    def test_contains_inside(self):
        """contains(x) should return True for x within [lo, hi] (closed)."""
        bi = BetaInterval(lo=1.0, hi=5.0)
        assert bi.contains(3.0) is True
        assert bi.contains(1.0) is True
        assert bi.contains(5.0) is True

    def test_contains_outside(self):
        """contains(x) should return False for x outside [lo, hi]."""
        bi = BetaInterval(lo=1.0, hi=5.0)
        assert bi.contains(0.5) is False
        assert bi.contains(5.5) is False

    def test_split(self):
        """split() should divide the interval into two halves at midpoint."""
        bi = BetaInterval(lo=1.0, hi=5.0)
        left, right = bi.split()

        assert left.lo == 1.0
        assert left.hi == pytest.approx(3.0)
        assert right.lo == pytest.approx(3.0)
        assert right.hi == 5.0

    def test_split_preserves_coverage(self):
        """Two sub-intervals from split() should cover the original."""
        bi = BetaInterval(lo=0.5, hi=10.0)
        left, right = bi.split()

        assert left.lo == bi.lo
        assert right.hi == bi.hi
        assert left.hi == right.lo

    def test_frozen_dataclass(self):
        """BetaInterval is frozen — attributes should be immutable."""
        bi = BetaInterval(lo=1.0, hi=5.0)
        with pytest.raises(AttributeError):
            bi.lo = 2.0  # type: ignore

    def test_zero_width_interval(self):
        """A degenerate interval with lo == hi should have width 0."""
        bi = BetaInterval(lo=3.0, hi=3.0)
        assert bi.width == 0.0
        assert bi.mid == 3.0
        assert bi.contains(3.0) is True


# ---------------------------------------------------------------------------
# Tests: ParameterFreeComparator initialization
# ---------------------------------------------------------------------------


class TestParameterFreeComparatorInit:
    """Tests for ParameterFreeComparator constructor."""

    def test_default_parameters(self):
        """Default ParameterFreeComparator: n_grid=20, α=0.05, tol=0.01."""
        pfc = ParameterFreeComparator()
        assert pfc.n_grid == 20
        assert pfc.n_trajectories == 200
        assert pfc.significance_level == 0.05
        assert pfc.bisection_tol == 0.01
        assert pfc.max_bisection_iters == 50

    def test_custom_grid(self):
        """ParameterFreeComparator should accept a custom grid resolution."""
        pfc = ParameterFreeComparator(n_grid=50)
        assert pfc.n_grid == 50

    def test_custom_bisection_tol(self):
        """Smaller bisection tolerance yields more precise crossover detection."""
        pfc = ParameterFreeComparator(bisection_tol=0.001)
        assert pfc.bisection_tol == 0.001


# ---------------------------------------------------------------------------
# Tests: compare() — identical MDPs (unanimous verdict)
# ---------------------------------------------------------------------------


class TestParameterFreeIdentical:
    """Tests for parameter-free comparison of identical MDPs."""

    def test_identical_mdps_result_type(self):
        """Comparing identical MDPs should return a ComparisonResult."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=5, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment, task=task)

        assert isinstance(result, ComparisonResult)

    def test_identical_mdps_verdict(self):
        """Identical MDPs should yield NEUTRAL or INCONCLUSIVE across β."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=5, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment, task=task)

        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_identical_mdps_confidence(self):
        """confidence should be 1 - α (default 0.95)."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=100, significance_level=0.05)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment, task=task)

        assert result.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Tests: compare() — regression detection across β
# ---------------------------------------------------------------------------


class TestParameterFreeRegression:
    """Tests for detecting regressions across the full β range."""

    def test_unanimous_regression(self):
        """A large cost increase should be REGRESSION at all β values."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = _identity_alignment(mdp_before)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=5, n_trajectories=200)
        result = pfc.compare(
            mdp_before, mdp_after, beta_range=(0.5, 5.0),
            alignment=alignment, task=task,
        )

        assert result.verdict == RegressionVerdict.REGRESSION

    def test_unanimous_improvement(self):
        """A large cost decrease yields NEUTRAL (one-sided regression test)."""
        mdp_before = _make_higher_cost_mdp()
        mdp_after = _make_lower_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="mid", state_b="mid", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            overall_similarity=1.0,
        )
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=5, n_trajectories=200)
        result = pfc.compare(
            mdp_before, mdp_after, beta_range=(0.5, 5.0),
            alignment=alignment, task=task,
        )

        assert result.verdict == RegressionVerdict.NEUTRAL
        assert result.delta_cost.mean_time < 0

    def test_parameter_free_flag(self):
        """Unanimous verdict + interval analysis → is_parameter_free may be True."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = _identity_alignment(mdp_before)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=5, n_trajectories=200)
        result = pfc.compare(
            mdp_before, mdp_after, beta_range=(0.5, 5.0),
            alignment=alignment, task=task,
        )

        # If the verdict is REGRESSION, is_parameter_free may be True
        if result.verdict == RegressionVerdict.REGRESSION:
            assert isinstance(result.is_parameter_free, bool)


# ---------------------------------------------------------------------------
# Tests: compare() — invalid β range
# ---------------------------------------------------------------------------


class TestParameterFreeValidation:
    """Tests for input validation in ParameterFreeComparator.compare()."""

    def test_invalid_beta_range_zero_lo(self):
        """β_lo must be > 0; range starting at 0 should raise ComparisonError."""
        mdp = make_two_state_mdp()
        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=50)

        with pytest.raises(ComparisonError):
            pfc.compare(mdp, mdp, beta_range=(0.0, 5.0))

    def test_invalid_beta_range_reversed(self):
        """Reversed range (5.0, 1.0) should raise ComparisonError."""
        mdp = make_two_state_mdp()
        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=50)

        with pytest.raises(ComparisonError):
            pfc.compare(mdp, mdp, beta_range=(5.0, 1.0))

    def test_negative_beta_lo(self):
        """Negative β_lo should raise ComparisonError."""
        mdp = make_two_state_mdp()
        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=50)

        with pytest.raises(ComparisonError):
            pfc.compare(mdp, mdp, beta_range=(-1.0, 5.0))


# ---------------------------------------------------------------------------
# Tests: grid resolution
# ---------------------------------------------------------------------------


class TestGridResolution:
    """Tests for the effect of grid resolution on comparison results."""

    def test_finer_grid_still_produces_result(self):
        """Increasing n_grid should still produce a valid ComparisonResult."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=10, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment, task=task)

        assert isinstance(result, ComparisonResult)

    def test_single_grid_point(self):
        """n_grid=1 should evaluate at a single β and still return a result."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=1, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(1.0, 5.0), alignment=alignment, task=task)

        assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# Tests: parameter sensitivity metadata
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    """Tests for the parameter_sensitivity field in the result."""

    def test_sensitivity_contains_beta_range(self):
        """Result's parameter_sensitivity should record the β range tested."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = _identity_alignment(mdp_before)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=100)
        result = pfc.compare(
            mdp_before, mdp_after, beta_range=(0.5, 5.0),
            alignment=alignment, task=task,
        )

        assert isinstance(result.parameter_sensitivity, dict)

    def test_description_nonempty(self):
        """The result should include a non-empty description."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()

        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment, task=task)

        assert isinstance(result.description, str)
        assert len(result.description) > 0


# ---------------------------------------------------------------------------
# Tests: default alignment and task
# ---------------------------------------------------------------------------


class TestParameterFreeDefaults:
    """Tests for default alignment and task parameters."""

    def test_none_alignment_uses_default(self):
        """alignment=None should use a default empty AlignmentResult."""
        mdp = make_two_state_mdp()
        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0))

        assert isinstance(result, ComparisonResult)

    def test_none_task_uses_default(self):
        """task=None should use a default empty TaskSpec."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        pfc = ParameterFreeComparator(n_grid=3, n_trajectories=100)
        result = pfc.compare(mdp, mdp, beta_range=(0.5, 5.0), alignment=alignment)

        assert isinstance(result, ComparisonResult)
