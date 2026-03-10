"""Unit tests for usability_oracle.fragility.cliff.CliffDetector.

Tests the cliff-detection pipeline that identifies discontinuities
(policy switches, state collapses, information cliffs) in the cost-vs-beta
curve of a bounded-rational usability MDP.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from usability_oracle.fragility.cliff import CliffDetector
from usability_oracle.fragility.models import CliffLocation
from usability_oracle.mdp.models import MDP, Action, State, Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_mdp() -> MDP:
    """Two-state MDP with a single deterministic transition."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="goal", is_terminal=True, is_goal=True)
    a = Action(action_id="a_go", action_type="click", target_node_id="btn")
    t = Transition(source="s0", action="a_go", target="s1", probability=1.0, cost=1.0)
    return MDP(
        states={"s0": s0, "s1": s1},
        actions={"a_go": a},
        transitions=[t],
        initial_state="s0",
        goal_states={"s1"},
        discount=0.99,
    )


def _make_choice_mdp() -> MDP:
    """MDP with a choice between cheap-risky and expensive-safe paths."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="risky_mid")
    s2 = State(state_id="s2", label="goal", is_terminal=True, is_goal=True)
    a_risky = Action(action_id="a_risky", action_type="click", target_node_id="r")
    a_safe = Action(action_id="a_safe", action_type="click", target_node_id="s")
    a_done = Action(action_id="a_done", action_type="click", target_node_id="d")
    transitions = [
        Transition(source="s0", action="a_risky", target="s1", probability=0.8, cost=0.5),
        Transition(source="s0", action="a_risky", target="s0", probability=0.2, cost=0.5),
        Transition(source="s0", action="a_safe", target="s2", probability=1.0, cost=5.0),
        Transition(source="s1", action="a_done", target="s2", probability=1.0, cost=0.5),
    ]
    return MDP(
        states={"s0": s0, "s1": s1, "s2": s2},
        actions={"a_risky": a_risky, "a_safe": a_safe, "a_done": a_done},
        transitions=transitions,
        initial_state="s0",
        goal_states={"s2"},
        discount=0.99,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCliffDetectorConstruction:
    """Tests for CliffDetector instantiation."""

    def test_default_construction(self):
        """CliffDetector can be created with no arguments."""
        det = CliffDetector()
        assert det is not None

    def test_custom_peak_prominence(self):
        """peak_prominence is stored correctly."""
        det = CliffDetector(peak_prominence=0.5)
        assert det.peak_prominence == 0.5

    def test_custom_bisection_tol(self):
        """bisection_tol is stored correctly."""
        det = CliffDetector(bisection_tol=0.001)
        assert det.bisection_tol == 0.001

    def test_class_constants(self):
        """CliffDetector exposes classification constants."""
        assert CliffDetector.POLICY_SWITCH == "policy_switch"
        assert CliffDetector.STATE_COLLAPSE == "state_collapse"
        assert CliffDetector.INFORMATION_CLIFF == "information_cliff"


# ---------------------------------------------------------------------------
# detect() with mocked solver
# ---------------------------------------------------------------------------

class TestCliffDetectorDetect:
    """Tests for CliffDetector.detect method."""

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_detect_returns_list(self, mock_cost):
        """detect() always returns a list."""
        mock_cost.return_value = 1.0
        det = CliffDetector(n_trajectories=5)
        result = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=20)
        assert isinstance(result, list)

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_no_cliffs_for_constant_cost(self, mock_cost):
        """Constant cost curve should produce zero cliffs."""
        mock_cost.return_value = 2.0
        det = CliffDetector(n_trajectories=5, peak_prominence=0.05)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=30)
        assert len(cliffs) == 0

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_cliff_at_discontinuity(self, mock_cost):
        """A step-function cost curve should trigger at least one cliff."""
        # Simulate a step: cost=10 for beta < 5, cost=1 for beta >= 5
        def step_cost(mdp, beta, n_trajectories=100):
            return 10.0 if beta < 5.0 else 1.0

        mock_cost.side_effect = step_cost
        det = CliffDetector(n_trajectories=5, peak_prominence=0.01)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=50)
        assert len(cliffs) >= 1

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_cliff_elements_are_cliff_location(self, mock_cost):
        """Each element returned by detect() is a CliffLocation."""
        def step_cost(mdp, beta, n_trajectories=100):
            return 10.0 if beta < 5.0 else 1.0

        mock_cost.side_effect = step_cost
        det = CliffDetector(n_trajectories=5, peak_prominence=0.01)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=50)
        for c in cliffs:
            assert isinstance(c, CliffLocation)

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_cliff_beta_star_in_range(self, mock_cost):
        """Detected cliff beta_star lies within the searched beta range."""
        def step_cost(mdp, beta, n_trajectories=100):
            return 8.0 if beta < 3.0 else 2.0

        mock_cost.side_effect = step_cost
        det = CliffDetector(n_trajectories=5, peak_prominence=0.01)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.5, 8.0), resolution=50)
        for c in cliffs:
            assert 0.5 <= c.beta_star <= 8.0


# ---------------------------------------------------------------------------
# CliffLocation properties
# ---------------------------------------------------------------------------

class TestCliffLocationProperties:
    """Tests for the CliffLocation dataclass attributes and properties."""

    def test_cost_jump(self):
        """cost_jump is the absolute difference between cost_after and cost_before."""
        cl = CliffLocation(beta_star=5.0, cost_before=10.0, cost_after=3.0)
        assert cl.cost_jump == pytest.approx(7.0)

    def test_relative_jump(self):
        """relative_jump is cost_jump / cost_before when cost_before > 0."""
        cl = CliffLocation(beta_star=5.0, cost_before=10.0, cost_after=3.0)
        assert cl.relative_jump == pytest.approx(0.7)

    def test_relative_jump_zero_baseline(self):
        """relative_jump handles cost_before == 0 gracefully."""
        cl = CliffLocation(beta_star=5.0, cost_before=0.0, cost_after=3.0)
        # Should not raise; implementation may return inf or 0
        rj = cl.relative_jump
        assert isinstance(rj, float)

    def test_default_cliff_type(self):
        """Default cliff_type is 'policy_switch'."""
        cl = CliffLocation()
        assert cl.cliff_type == "policy_switch"

    def test_affected_states_default_empty(self):
        """affected_states defaults to empty list."""
        cl = CliffLocation()
        assert cl.affected_states == []

    def test_gradient_stored(self):
        """gradient field is stored correctly."""
        cl = CliffLocation(gradient=42.0)
        assert cl.gradient == 42.0


# ---------------------------------------------------------------------------
# Cliff classification
# ---------------------------------------------------------------------------

class TestCliffClassification:
    """Tests for cliff type classification constants and semantics."""

    def test_policy_switch_is_string(self):
        """POLICY_SWITCH constant is a string."""
        assert isinstance(CliffDetector.POLICY_SWITCH, str)

    def test_state_collapse_is_string(self):
        """STATE_COLLAPSE constant is a string."""
        assert isinstance(CliffDetector.STATE_COLLAPSE, str)

    def test_information_cliff_is_string(self):
        """INFORMATION_CLIFF constant is a string."""
        assert isinstance(CliffDetector.INFORMATION_CLIFF, str)

    def test_all_types_distinct(self):
        """All three cliff classification types are distinct."""
        types = {
            CliffDetector.POLICY_SWITCH,
            CliffDetector.STATE_COLLAPSE,
            CliffDetector.INFORMATION_CLIFF,
        }
        assert len(types) == 3


# ---------------------------------------------------------------------------
# cliff_severity static method
# ---------------------------------------------------------------------------

class TestCliffSeverity:
    """Tests for CliffDetector._cliff_severity static method."""

    def test_severity_nonnegative(self):
        """Severity is always >= 0."""
        cl = CliffLocation(beta_star=5.0, cost_before=10.0, cost_after=3.0, gradient=5.0)
        sev = CliffDetector._cliff_severity(cl)
        assert sev >= 0.0

    def test_severity_zero_for_no_jump(self):
        """Severity is 0 (or near 0) when cost_before == cost_after."""
        cl = CliffLocation(beta_star=5.0, cost_before=3.0, cost_after=3.0, gradient=0.0)
        sev = CliffDetector._cliff_severity(cl)
        assert sev == pytest.approx(0.0, abs=1e-6)

    def test_severity_increases_with_jump(self):
        """Larger cost jump ⇒ higher severity."""
        cl_small = CliffLocation(beta_star=5.0, cost_before=10.0, cost_after=9.0, gradient=1.0)
        cl_big = CliffLocation(beta_star=5.0, cost_before=10.0, cost_after=1.0, gradient=9.0)
        assert CliffDetector._cliff_severity(cl_big) > CliffDetector._cliff_severity(cl_small)


# ---------------------------------------------------------------------------
# Peak prominence filtering
# ---------------------------------------------------------------------------

class TestPeakProminenceFiltering:
    """Tests that the peak_prominence parameter filters out noise."""

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_high_prominence_filters_small_bumps(self, mock_cost):
        """With high peak_prominence, small bumps are not reported as cliffs."""
        # Small sine-wave-like perturbation
        def noisy_cost(mdp, beta, n_trajectories=100):
            return 5.0 + 0.05 * math.sin(beta)

        mock_cost.side_effect = noisy_cost
        det = CliffDetector(n_trajectories=5, peak_prominence=0.5)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=40)
        assert len(cliffs) == 0

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    def test_low_prominence_may_detect_bumps(self, mock_cost):
        """With very low prominence and a sharp step, at least one cliff is found."""
        def sharp_step(mdp, beta, n_trajectories=100):
            return 10.0 if beta < 4.0 else 2.0

        mock_cost.side_effect = sharp_step
        det = CliffDetector(n_trajectories=5, peak_prominence=0.001)
        cliffs = det.detect(_make_minimal_mdp(), beta_range=(0.1, 10.0), resolution=50)
        assert len(cliffs) >= 1


# ---------------------------------------------------------------------------
# refine_cliff bisection
# ---------------------------------------------------------------------------

class TestRefineCliff:
    """Tests for _refine_cliff bisection refinement."""

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    @patch("usability_oracle.fragility.cliff._solve_policy")
    def test_refine_cliff_returns_cliff_location(self, mock_policy, mock_cost):
        """_refine_cliff returns a CliffLocation."""
        mock_cost.side_effect = lambda mdp, beta, n_trajectories=100: 10.0 if beta < 3.0 else 2.0
        policy_mock = MagicMock()
        policy_mock.action_probs.return_value = {"a_go": 1.0}
        policy_mock.entropy.return_value = 0.5
        mock_policy.return_value = policy_mock
        det = CliffDetector(n_trajectories=5, bisection_tol=0.01)
        result = det._refine_cliff(_make_minimal_mdp(), 3.0, 1.0)
        assert isinstance(result, CliffLocation)

    @patch("usability_oracle.fragility.cliff._solve_and_cost")
    @patch("usability_oracle.fragility.cliff._solve_policy")
    def test_refine_cliff_beta_near_true(self, mock_policy, mock_cost):
        """Refined beta_star is close to the true discontinuity at beta=3."""
        mock_cost.side_effect = lambda mdp, beta, n_trajectories=100: 10.0 if beta < 3.0 else 2.0
        policy_mock = MagicMock()
        policy_mock.action_probs.return_value = {"a_go": 1.0}
        policy_mock.entropy.return_value = 0.5
        mock_policy.return_value = policy_mock
        det = CliffDetector(n_trajectories=5, bisection_tol=0.01)
        result = det._refine_cliff(_make_minimal_mdp(), 3.0, 1.0)
        assert abs(result.beta_star - 3.0) < 1.5
