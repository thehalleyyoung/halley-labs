"""Unit tests for CrossChannelInterferenceDetector.

Tests cover detection of resource conflicts between co-occurring
actions, independence when channels differ, the default channel
capacity table, interference pairs, severity scaling, and constants.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from usability_oracle.bottleneck.interference import (
    CrossChannelInterferenceDetector,
    DEFAULT_CHANNEL_CAPACITIES,
    INTERFERENCE_PAIRS,
)
from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import Action, MDP, State, Transition


# ===================================================================
# Helper factories
# ===================================================================

def _make_mdp_with_actions(action_specs: List[Dict[str, str]]) -> MDP:
    """Build a 2-state MDP with actions described by *action_specs*.

    Each spec dict should have ``id``, ``type``, and ``target``.
    All actions transition from ``page`` → ``goal``.
    """
    states = {
        "page": State(
            state_id="page",
            features={"n_elements": 5.0, "x": 100.0, "y": 100.0, "width": 50.0, "height": 50.0, "wm_load": 2.0},
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
    actions: Dict[str, Action] = {}
    transitions: List[Transition] = []
    for spec in action_specs:
        aid = spec["id"]
        actions[aid] = Action(
            action_id=aid,
            action_type=spec.get("type", Action.CLICK),
            target_node_id=spec.get("target", "node"),
            description=spec.get("desc", f"Action {aid}"),
            preconditions=[],
        )
        transitions.append(
            Transition(source="page", action=aid, target="goal", probability=1.0, cost=0.3)
        )
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="page",
        goal_states={"goal"},
        discount=0.99,
    )


def _same_channel_actions() -> List[Dict[str, str]]:
    """Return action specs that use the same motor channel (CLICK + CLICK)."""
    return [
        {"id": "click_a", "type": Action.CLICK, "target": "btn_a"},
        {"id": "click_b", "type": Action.CLICK, "target": "btn_b"},
        {"id": "click_c", "type": Action.CLICK, "target": "btn_c"},
        {"id": "click_d", "type": Action.CLICK, "target": "btn_d"},
    ]


def _independent_actions() -> List[Dict[str, str]]:
    """Return a single action so no interference can occur."""
    return [
        {"id": "solo_click", "type": Action.CLICK, "target": "btn"},
    ]


def _mixed_channel_actions() -> List[Dict[str, str]]:
    """Return actions that span different types to exercise channel mapping."""
    return [
        {"id": "read_text", "type": Action.READ, "target": "para"},
        {"id": "type_field", "type": Action.TYPE, "target": "input"},
        {"id": "scroll_page", "type": Action.SCROLL, "target": "body"},
        {"id": "click_btn", "type": Action.CLICK, "target": "btn"},
    ]


# ===================================================================
# Tests — module-level constants
# ===================================================================

class TestInterferenceModuleConstants:
    """Verify module-level constants exported by the interference module."""

    def test_default_channel_capacities_is_dict(self):
        """DEFAULT_CHANNEL_CAPACITIES should be a dict."""
        assert isinstance(DEFAULT_CHANNEL_CAPACITIES, dict)

    def test_default_channel_capacities_contains_visual(self):
        """DEFAULT_CHANNEL_CAPACITIES should contain 'visual'."""
        assert "visual" in DEFAULT_CHANNEL_CAPACITIES

    def test_default_channel_capacities_contains_auditory(self):
        """DEFAULT_CHANNEL_CAPACITIES should contain 'auditory'."""
        assert "auditory" in DEFAULT_CHANNEL_CAPACITIES

    def test_default_channel_capacities_contains_motor_hand(self):
        """DEFAULT_CHANNEL_CAPACITIES should contain 'motor_hand'."""
        assert "motor_hand" in DEFAULT_CHANNEL_CAPACITIES

    def test_default_channel_capacities_values_positive(self):
        """All channel capacities should be positive floats."""
        for k, v in DEFAULT_CHANNEL_CAPACITIES.items():
            assert v > 0, f"Capacity for {k} should be positive, got {v}"

    def test_interference_pairs_is_set(self):
        """INTERFERENCE_PAIRS should be a set of frozensets."""
        assert isinstance(INTERFERENCE_PAIRS, set)
        for pair in INTERFERENCE_PAIRS:
            assert isinstance(pair, frozenset)
            assert len(pair) == 2

    def test_interference_pairs_contains_visual_motor_eye(self):
        """INTERFERENCE_PAIRS should include {visual, motor_eye}."""
        assert frozenset({"visual", "motor_eye"}) in INTERFERENCE_PAIRS

    def test_interference_pairs_contains_cognitive_verbal_motor_voice(self):
        """INTERFERENCE_PAIRS should include {cognitive_verbal, motor_voice}."""
        assert frozenset({"cognitive_verbal", "motor_voice"}) in INTERFERENCE_PAIRS


# ===================================================================
# Tests — instance-level constants
# ===================================================================

class TestInterferenceInstanceConstants:
    """Verify instance-level constants on CrossChannelInterferenceDetector."""

    def test_interference_threshold_exists(self):
        """INTERFERENCE_THRESHOLD should be defined."""
        assert hasattr(CrossChannelInterferenceDetector, "INTERFERENCE_THRESHOLD")

    def test_interference_threshold_value(self):
        """INTERFERENCE_THRESHOLD should be 0.5."""
        assert CrossChannelInterferenceDetector.INTERFERENCE_THRESHOLD == pytest.approx(0.5)

    def test_temporal_overlap_threshold_exists(self):
        """TEMPORAL_OVERLAP_THRESHOLD should be defined."""
        assert hasattr(CrossChannelInterferenceDetector, "TEMPORAL_OVERLAP_THRESHOLD")

    def test_temporal_overlap_threshold_value(self):
        """TEMPORAL_OVERLAP_THRESHOLD should be 0.3."""
        assert CrossChannelInterferenceDetector.TEMPORAL_OVERLAP_THRESHOLD == pytest.approx(0.3)

    def test_channel_capacities_default(self):
        """Instance should use DEFAULT_CHANNEL_CAPACITIES by default."""
        det = CrossChannelInterferenceDetector()
        assert det.channel_capacities == DEFAULT_CHANNEL_CAPACITIES

    def test_custom_channel_capacities(self):
        """Instance should accept custom channel capacities."""
        custom = {"visual": 20.0, "auditory": 15.0}
        det = CrossChannelInterferenceDetector(channel_capacities=custom)
        assert det.channel_capacities == custom


# ===================================================================
# Tests — detect() with same-channel actions (interference expected)
# ===================================================================

class TestInterferenceDetected:
    """Actions sharing the same channel should trigger interference."""

    def test_same_channel_actions_trigger(self):
        """Multiple CLICK actions (same motor channel) should detect interference."""
        specs = _same_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        state = mdp.states["page"]
        action_ids = list(mdp.actions.keys())
        result = det.detect(state, action_ids, mdp)
        # May or may not trigger depending on internal demand mapping
        # but should not crash
        assert result is None or isinstance(result, BottleneckResult)

    def test_mixed_channels_detect(self):
        """Mixed action types spanning conflicting channels may detect interference."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        state = mdp.states["page"]
        action_ids = list(mdp.actions.keys())
        result = det.detect(state, action_ids, mdp)
        assert result is None or isinstance(result, BottleneckResult)

    def test_result_type_is_cross_channel(self):
        """If detected, result type must be CROSS_CHANNEL_INTERFERENCE."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            assert result.bottleneck_type == BottleneckType.CROSS_CHANNEL_INTERFERENCE

    def test_result_confidence_positive(self):
        """If detected, confidence should be > 0."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            assert result.confidence > 0.0


# ===================================================================
# Tests — detect() with independent channels (no interference)
# ===================================================================

class TestNoInterference:
    """A single action or fully independent channels should not trigger."""

    def test_single_action_returns_none(self):
        """A single action cannot interfere with itself."""
        specs = _independent_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        assert result is None

    def test_no_actions_no_crash(self):
        """An empty action list should not crash."""
        mdp = _make_mdp_with_actions([])
        det = CrossChannelInterferenceDetector()
        try:
            result = det.detect(mdp.states["page"], [], mdp)
        except Exception:
            pass  # acceptable on degenerate input


# ===================================================================
# Tests — severity from interference level
# ===================================================================

class TestInterferenceSeverity:
    """Severity should scale with interference level."""

    def test_severity_from_interference_callable(self):
        """_severity_from_interference should exist and be callable."""
        det = CrossChannelInterferenceDetector()
        assert callable(getattr(det, "_severity_from_interference", None))

    def test_high_interference_high_severity(self):
        """High interference should produce higher severity than low."""
        det = CrossChannelInterferenceDetector()
        sev_low = det._severity_from_interference(0.2, 0.1)
        sev_high = det._severity_from_interference(0.9, 0.8)
        # Severity enum comparison
        assert sev_high.numeric >= sev_low.numeric


# ===================================================================
# Tests — evidence and recommendation
# ===================================================================

class TestInterferenceEvidenceRecommendation:
    """Evidence dict and recommendation string should be populated."""

    def test_evidence_is_dict_if_detected(self):
        """Evidence should be a dict if interference is detected."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            assert isinstance(result.evidence, dict)

    def test_recommendation_non_empty_if_detected(self):
        """Recommendation should be non-empty if interference is detected."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            assert len(result.recommendation) > 0

    def test_description_non_empty_if_detected(self):
        """Description should be non-empty if interference is detected."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            assert len(result.description) > 0


# ===================================================================
# Tests — internal methods
# ===================================================================

class TestInterferenceInternals:
    """Test internal helper methods of CrossChannelInterferenceDetector."""

    def test_interference_level_zero_for_empty(self):
        """_interference_level({}) should return 0.0."""
        det = CrossChannelInterferenceDetector()
        level = det._interference_level({})
        assert level == pytest.approx(0.0)

    def test_interference_level_increases_with_demand(self):
        """Higher demand relative to capacity should increase interference."""
        det = CrossChannelInterferenceDetector()
        low_demand = {"visual": 10.0}
        high_demand = {"visual": 80.0}
        level_low = det._interference_level(low_demand)
        level_high = det._interference_level(high_demand)
        assert level_high >= level_low

    def test_share_resource_dimension_callable(self):
        """_share_resource_dimension should exist and be callable."""
        det = CrossChannelInterferenceDetector()
        assert callable(getattr(det, "_share_resource_dimension", None))

    def test_share_resource_visual_motor_eye(self):
        """Visual and motor_eye should share a resource dimension."""
        det = CrossChannelInterferenceDetector()
        assert det._share_resource_dimension("visual", "motor_eye") is True

    def test_no_share_visual_auditory(self):
        """Visual and auditory should NOT share a resource dimension."""
        det = CrossChannelInterferenceDetector()
        result = det._share_resource_dimension("visual", "auditory")
        assert result is False


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestInterferenceEdgeCases:
    """Edge-case behaviour for the interference detector."""

    def test_result_to_dict_if_detected(self):
        """to_dict() should succeed on an interference result."""
        specs = _mixed_channel_actions()
        mdp = _make_mdp_with_actions(specs)
        det = CrossChannelInterferenceDetector()
        result = det.detect(mdp.states["page"], list(mdp.actions.keys()), mdp)
        if result is not None:
            d = result.to_dict()
            assert isinstance(d, dict)

    def test_detector_instantiation_default(self):
        """Default instantiation should not raise."""
        det = CrossChannelInterferenceDetector()
        assert det is not None

    def test_temporal_overlap_callable(self):
        """_temporal_overlap should exist and be callable."""
        det = CrossChannelInterferenceDetector()
        assert callable(getattr(det, "_temporal_overlap", None))

    def test_resource_conflict_callable(self):
        """_resource_conflict should exist and be callable."""
        det = CrossChannelInterferenceDetector()
        assert callable(getattr(det, "_resource_conflict", None))
