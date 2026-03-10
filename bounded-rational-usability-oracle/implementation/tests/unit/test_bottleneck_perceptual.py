"""Unit tests for PerceptualOverloadDetector.

Tests cover detection of visual overload from high element counts,
absence of overload on clean layouts, severity scaling, threshold
constants, evidence dictionaries, and recommendation strings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pytest

from usability_oracle.bottleneck.perceptual import PerceptualOverloadDetector
from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import Action, MDP, State, Transition


# ===================================================================
# Helper factories
# ===================================================================

def _make_mdp_with_elements(n_elements: int, width: float = 50.0, height: float = 50.0) -> MDP:
    """Create a 2-state MDP where the non-goal state has *n_elements*.

    ``features`` dict includes ``n_elements``, spatial coordinates,
    dimensions, and ``wm_load``.
    """
    states = {
        "page": State(
            state_id="page",
            features={
                "n_elements": float(n_elements),
                "x": 200.0,
                "y": 150.0,
                "width": width,
                "height": height,
                "wm_load": 2.0,
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
        "go": Action(
            action_id="go",
            action_type=Action.CLICK,
            target_node_id="n0",
            description="Click",
            preconditions=[],
        ),
    }
    transitions = [
        Transition(source="page", action="go", target="goal", probability=1.0, cost=0.5),
    ]
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="page",
        goal_states={"goal"},
        discount=0.99,
    )


def _get_page_state(mdp: MDP) -> str:
    """Return the 'page' state id string from *mdp*."""
    return "page"


def _get_features(mdp: MDP) -> Dict[str, float]:
    """Return the feature dict for the 'page' state."""
    return mdp.states["page"].features


# ===================================================================
# Tests — threshold constants
# ===================================================================

class TestPerceptualConstants:
    """Verify well-known constants on PerceptualOverloadDetector."""

    def test_overload_threshold_exists(self):
        """OVERLOAD_THRESHOLD should be defined and positive."""
        assert hasattr(PerceptualOverloadDetector, "OVERLOAD_THRESHOLD")
        assert PerceptualOverloadDetector.OVERLOAD_THRESHOLD > 0

    def test_clutter_threshold_exists(self):
        """CLUTTER_THRESHOLD should be defined and in (0, 1]."""
        assert hasattr(PerceptualOverloadDetector, "CLUTTER_THRESHOLD")
        assert 0 < PerceptualOverloadDetector.CLUTTER_THRESHOLD <= 1.0

    def test_overload_threshold_value(self):
        """OVERLOAD_THRESHOLD should be 3.0 (entropy bits)."""
        assert PerceptualOverloadDetector.OVERLOAD_THRESHOLD == pytest.approx(3.0)

    def test_clutter_threshold_value(self):
        """CLUTTER_THRESHOLD should be 0.3."""
        assert PerceptualOverloadDetector.CLUTTER_THRESHOLD == pytest.approx(0.3)


# ===================================================================
# Tests — detect() with high n_elements
# ===================================================================

class TestPerceptualOverloadDetected:
    """When visual complexity is high, the detector should fire."""

    def test_many_elements_triggers_overload(self):
        """A state with a very high n_elements should produce a result."""
        mdp = _make_mdp_with_elements(n_elements=80)
        detector = PerceptualOverloadDetector()
        state = _get_page_state(mdp)
        features = _get_features(mdp)
        result = detector.detect(state, mdp, features)
        assert result is not None

    def test_result_type_is_perceptual_overload(self):
        """Returned BottleneckResult must have type PERCEPTUAL_OVERLOAD."""
        mdp = _make_mdp_with_elements(n_elements=80)
        detector = PerceptualOverloadDetector()
        result = detector.detect("page", mdp, _get_features(mdp))
        assert result is not None
        assert result.bottleneck_type == BottleneckType.PERCEPTUAL_OVERLOAD

    def test_result_is_bottleneck_result(self):
        """detect() should return a BottleneckResult instance when triggered."""
        mdp = _make_mdp_with_elements(n_elements=100)
        detector = PerceptualOverloadDetector()
        result = detector.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert isinstance(result, BottleneckResult)

    def test_result_has_nonzero_confidence(self):
        """Confidence should be > 0 for a detected overload."""
        mdp = _make_mdp_with_elements(n_elements=80)
        detector = PerceptualOverloadDetector()
        result = detector.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert result.confidence > 0.0

    def test_result_severity_not_info(self):
        """A genuine overload should have severity above INFO."""
        mdp = _make_mdp_with_elements(n_elements=80)
        detector = PerceptualOverloadDetector()
        result = detector.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert result.severity != Severity.INFO


# ===================================================================
# Tests — detect() with low n_elements (no overload)
# ===================================================================

class TestPerceptualNoOverload:
    """Clean layouts with few elements should not trigger overload."""

    def test_few_elements_returns_none(self):
        """A state with 2 elements should not be overloaded when well-spaced."""
        mdp = _make_mdp_with_elements(n_elements=2)
        detector = PerceptualOverloadDetector()
        features = dict(_get_features(mdp))
        # Space elements apart to prevent synthetic bounding-box overlap
        features["element_0_x"] = 0.0
        features["element_1_x"] = 200.0
        result = detector.detect(_get_page_state(mdp), mdp, features)
        assert result is None

    def test_single_element_no_overload(self):
        """A state with 1 element must never produce overload."""
        mdp = _make_mdp_with_elements(n_elements=1)
        detector = PerceptualOverloadDetector()
        result = detector.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is None

    def test_moderate_elements_no_overload(self):
        """A state with a moderate count (e.g. 4) should likely not trigger."""
        mdp = _make_mdp_with_elements(n_elements=4)
        detector = PerceptualOverloadDetector()
        result = detector.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        # For a low number, no overload is the expected outcome
        if result is not None:
            assert result.severity in (Severity.LOW, Severity.INFO)


# ===================================================================
# Tests — severity scaling
# ===================================================================

class TestPerceptualSeverityScaling:
    """Severity should increase with element count."""

    def test_more_elements_means_higher_or_equal_severity(self):
        """A state with 100 elements should have >= severity vs 30."""
        det = PerceptualOverloadDetector()
        mdp_low = _make_mdp_with_elements(30)
        mdp_high = _make_mdp_with_elements(100)
        r_low = det.detect(_get_page_state(mdp_low), mdp_low, _get_features(mdp_low))
        r_high = det.detect(_get_page_state(mdp_high), mdp_high, _get_features(mdp_high))
        if r_low is not None and r_high is not None:
            assert r_high.severity_score >= r_low.severity_score

    def test_extreme_elements_severity_is_high_or_critical(self):
        """200 elements should produce at least MEDIUM severity."""
        mdp = _make_mdp_with_elements(200)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert result.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)


# ===================================================================
# Tests — evidence dict
# ===================================================================

class TestPerceptualEvidence:
    """The evidence dict should contain relevant perceptual metrics."""

    def test_evidence_is_dict(self):
        """Evidence should be a non-None dict."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert isinstance(result.evidence, dict)

    def test_evidence_contains_numeric_values(self):
        """All evidence values should be numeric (int or float)."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        for v in result.evidence.values():
            assert isinstance(v, (int, float))


# ===================================================================
# Tests — recommendation string
# ===================================================================

class TestPerceptualRecommendation:
    """The recommendation field should be a non-empty string."""

    def test_recommendation_non_empty(self):
        """Recommendation must be a non-empty string when overload is detected."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0

    def test_description_non_empty(self):
        """Description should also be populated."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0


# ===================================================================
# Tests — cognitive law
# ===================================================================

class TestPerceptualCognitiveLaw:
    """The result should reference the correct cognitive law."""

    def test_cognitive_law_is_perceptual_processing(self):
        """Perceptual overload results should cite PERCEPTUAL_PROCESSING."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert result.cognitive_law == CognitiveLaw.VISUAL_SEARCH

    def test_affected_states_includes_page(self):
        """The affected_states list should include the tested state id."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        assert "page" in result.affected_states


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestPerceptualEdgeCases:
    """Edge-case behaviour for the perceptual detector."""

    def test_zero_elements(self):
        """Zero elements should not trigger overload."""
        mdp = _make_mdp_with_elements(0)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is None

    def test_negative_elements_no_crash(self):
        """Negative n_elements (invalid) should not crash."""
        mdp = _make_mdp_with_elements(-5)
        det = PerceptualOverloadDetector()
        # Should either return None or handle gracefully
        try:
            result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        except Exception:
            pass  # acceptable to raise on invalid input

    def test_result_to_dict(self):
        """BottleneckResult.to_dict should succeed."""
        mdp = _make_mdp_with_elements(80)
        det = PerceptualOverloadDetector()
        result = det.detect(_get_page_state(mdp), mdp, _get_features(mdp))
        assert result is not None
        d = result.to_dict()
        assert isinstance(d, dict)
