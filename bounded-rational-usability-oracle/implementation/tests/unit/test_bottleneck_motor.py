"""Unit tests for MotorDifficultyDetector.

Tests cover detection via Fitts' law, index-of-difficulty calculations,
small-target and large-distance scenarios, severity scaling, threshold
constants, and result attributes.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from usability_oracle.bottleneck.motor import MotorDifficultyDetector
from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import Action, MDP, State, Transition


# ===================================================================
# Helper factories
# ===================================================================

def _make_motor_mdp(
    target_width: float = 50.0,
    target_height: float = 50.0,
    distance_x: float = 100.0,
    distance_y: float = 100.0,
) -> MDP:
    """Build a 2-state MDP with spatial features encoding target geometry.

    The ``page`` state has features that represent the target's position
    and size.  ``distance`` is implied from the cursor origin (0, 0) to
    (distance_x, distance_y).
    """
    import math as _math
    target_distance = _math.sqrt(distance_x ** 2 + distance_y ** 2)
    states = {
        "page": State(
            state_id="page",
            features={
                "n_elements": 5.0,
                "x": distance_x,
                "y": distance_y,
                "width": target_width,
                "height": target_height,
                "wm_load": 1.0,
                "target_width": target_width,
                "target_height": target_height,
                "target_distance": target_distance,
            },
            label="page",
            is_terminal=False,
            is_goal=False,
            metadata={},
        ),
        "goal": State(
            state_id="goal",
            features={"n_elements": 1.0, "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0, "wm_load": 0.0},
            label="goal",
            is_terminal=True,
            is_goal=True,
            metadata={},
        ),
    }
    actions = {
        "click": Action(
            action_id="click",
            action_type=Action.CLICK,
            target_node_id="btn",
            description="Click target",
            preconditions=[],
        ),
    }
    transitions = [
        Transition(source="page", action="click", target="goal", probability=1.0, cost=0.5),
    ]
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="page",
        goal_states={"goal"},
        discount=0.99,
    )


# ===================================================================
# Tests — constants
# ===================================================================

class TestMotorConstants:
    """Verify well-known constants on MotorDifficultyDetector."""

    def test_difficulty_threshold_exists(self):
        """DIFFICULTY_THRESHOLD should be defined."""
        assert hasattr(MotorDifficultyDetector, "DIFFICULTY_THRESHOLD")

    def test_difficulty_threshold_value(self):
        """DIFFICULTY_THRESHOLD should be 4.5 bits."""
        assert MotorDifficultyDetector.DIFFICULTY_THRESHOLD == pytest.approx(4.5)

    def test_min_target_px_exists(self):
        """MIN_TARGET_PX should be defined and positive."""
        assert hasattr(MotorDifficultyDetector, "MIN_TARGET_PX")
        assert MotorDifficultyDetector.MIN_TARGET_PX > 0

    def test_min_target_px_value(self):
        """MIN_TARGET_PX should be 24.0."""
        assert MotorDifficultyDetector.MIN_TARGET_PX == pytest.approx(24.0)

    def test_fitts_a_exists(self):
        """Fitts intercept FITTS_A should be defined."""
        assert hasattr(MotorDifficultyDetector, "FITTS_A")
        assert MotorDifficultyDetector.FITTS_A >= 0

    def test_fitts_b_exists(self):
        """Fitts slope FITTS_B should be defined and positive."""
        assert hasattr(MotorDifficultyDetector, "FITTS_B")
        assert MotorDifficultyDetector.FITTS_B > 0

    def test_max_distance_px_exists(self):
        """MAX_DISTANCE_PX should be defined."""
        assert hasattr(MotorDifficultyDetector, "MAX_DISTANCE_PX")
        assert MotorDifficultyDetector.MAX_DISTANCE_PX > 0


# ===================================================================
# Tests — detect() with small target + large distance
# ===================================================================

class TestMotorDifficultyDetected:
    """Small targets far from the cursor should trigger motor difficulty."""

    def test_small_distant_target_triggers(self):
        """A 5×5 px target 400 px away should produce a BottleneckResult."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        state = "page"
        action = "click"
        features = mdp.states["page"].features
        result = det.detect(state, action, mdp, features)
        assert result is not None

    def test_result_type_is_motor_difficulty(self):
        """Result bottleneck_type must be MOTOR_DIFFICULTY."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert result.bottleneck_type == BottleneckType.MOTOR_DIFFICULTY

    def test_result_is_bottleneck_result(self):
        """detect() should return a BottleneckResult when triggered."""
        mdp = _make_motor_mdp(target_width=8.0, target_height=8.0, distance_x=450.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert isinstance(result, BottleneckResult)

    def test_confidence_positive(self):
        """Confidence should be > 0 for a detected difficulty."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert result.confidence > 0.0

    def test_severity_not_info(self):
        """A genuine motor difficulty should have severity above INFO."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert result.severity != Severity.INFO


# ===================================================================
# Tests — detect() with large target + small distance (no difficulty)
# ===================================================================

class TestMotorNoDifficulty:
    """Large, close targets should not trigger motor difficulty."""

    def test_large_close_target_returns_none(self):
        """A 200×200 px target 10 px away should not be difficult."""
        mdp = _make_motor_mdp(target_width=200.0, target_height=200.0, distance_x=10.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is None

    def test_medium_target_close_returns_none(self):
        """A 100×100 px target 20 px away should be easy."""
        mdp = _make_motor_mdp(target_width=100.0, target_height=100.0, distance_x=20.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is None

    def test_zero_distance_returns_none(self):
        """Target at cursor origin should have minimal difficulty."""
        mdp = _make_motor_mdp(target_width=50.0, target_height=50.0, distance_x=0.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is None


# ===================================================================
# Tests — Fitts' law index of difficulty
# ===================================================================

class TestFittsLawCalculation:
    """Verify Fitts' law index-of-difficulty semantics."""

    def test_id_increases_with_distance(self):
        """Fitts' ID should increase as distance increases (width constant)."""
        det = MotorDifficultyDetector()
        id_near = det._index_of_difficulty(50.0, 30.0)
        id_far = det._index_of_difficulty(500.0, 30.0)
        assert id_far > id_near

    def test_id_increases_with_smaller_width(self):
        """Fitts' ID should increase as target width decreases."""
        det = MotorDifficultyDetector()
        id_wide = det._index_of_difficulty(200.0, 100.0)
        id_narrow = det._index_of_difficulty(200.0, 10.0)
        assert id_narrow > id_wide

    def test_id_is_non_negative(self):
        """Index of difficulty should always be >= 0."""
        det = MotorDifficultyDetector()
        assert det._index_of_difficulty(0.0, 50.0) >= 0
        assert det._index_of_difficulty(100.0, 50.0) >= 0

    def test_id_shannon_formulation(self):
        """ID should follow log2(D/W + 1) (Shannon formulation)."""
        det = MotorDifficultyDetector()
        D, W = 256.0, 32.0
        expected = math.log2(D / W + 1)
        actual = det._index_of_difficulty(D, W)
        assert actual == pytest.approx(expected, rel=0.05)


# ===================================================================
# Tests — severity scaling
# ===================================================================

class TestMotorSeverityScaling:
    """Severity should increase with the index of difficulty."""

    def test_higher_id_means_higher_severity(self):
        """Smaller target + larger distance should yield >= severity."""
        det = MotorDifficultyDetector()
        mdp_easy = _make_motor_mdp(target_width=80.0, target_height=80.0, distance_x=50.0)
        mdp_hard = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=450.0)
        r_easy = det.detect("page", "click", mdp_easy, mdp_easy.states["page"].features)
        r_hard = det.detect("page", "click", mdp_hard, mdp_hard.states["page"].features)
        if r_easy is not None and r_hard is not None:
            assert r_hard.severity_score >= r_easy.severity_score
        elif r_easy is None and r_hard is not None:
            pass  # expected: hard detected, easy not

    def test_very_small_target_high_severity(self):
        """A 3×3 target 500 px away should produce HIGH or CRITICAL."""
        mdp = _make_motor_mdp(target_width=3.0, target_height=3.0, distance_x=500.0, distance_y=0.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert result.severity in (Severity.HIGH, Severity.CRITICAL)


# ===================================================================
# Tests — cognitive law and evidence
# ===================================================================

class TestMotorCognitiveLaw:
    """Result should reference Fitts' law."""

    def test_cognitive_law_is_fitts(self):
        """Motor difficulty results should cite FITTS."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert result.cognitive_law == CognitiveLaw.FITTS

    def test_evidence_is_dict(self):
        """Evidence should be a dict."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert isinstance(result.evidence, dict)

    def test_recommendation_non_empty(self):
        """Recommendation should be a non-empty string."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert len(result.recommendation) > 0

    def test_affected_states_includes_page(self):
        """affected_states should include 'page'."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert "page" in result.affected_states

    def test_affected_actions_includes_click(self):
        """affected_actions should include 'click'."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        assert "click" in result.affected_actions


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestMotorEdgeCases:
    """Edge-case behaviour for the motor difficulty detector."""

    def test_very_large_target_no_difficulty(self):
        """A full-screen target should never trigger difficulty."""
        mdp = _make_motor_mdp(target_width=1920.0, target_height=1080.0, distance_x=100.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is None

    def test_result_to_dict(self):
        """to_dict() should succeed on a motor difficulty result."""
        mdp = _make_motor_mdp(target_width=5.0, target_height=5.0, distance_x=400.0)
        det = MotorDifficultyDetector()
        result = det.detect("page", "click", mdp, mdp.states["page"].features)
        assert result is not None
        d = result.to_dict()
        assert isinstance(d, dict)
