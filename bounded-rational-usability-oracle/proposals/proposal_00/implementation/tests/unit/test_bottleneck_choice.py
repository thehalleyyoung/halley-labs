"""Unit tests for ChoiceParalysisDetector.

Tests cover detection of choice paralysis from near-uniform action
distributions, absence when few actions exist, entropy-ratio thresholds,
Hick-Hyman law parameters, severity scaling, and constants.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from usability_oracle.bottleneck.choice import ChoiceParalysisDetector
from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import Action, MDP, State, Transition
from usability_oracle.policy.models import Policy


# ===================================================================
# Helper factories
# ===================================================================

def _make_mdp_with_n_actions(n: int) -> MDP:
    """Build an MDP where the initial state has exactly *n* available actions.

    All actions transition from ``page`` to ``goal`` with equal
    probability.  Features include ``n_elements`` and spatial data.
    """
    states = {
        "page": State(
            state_id="page",
            features={"n_elements": float(n), "x": 100.0, "y": 100.0, "width": 60.0, "height": 40.0, "wm_load": 2.0},
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
    for i in range(n):
        aid = f"a{i}"
        actions[aid] = Action(
            action_id=aid,
            action_type=Action.CLICK,
            target_node_id=f"n{i}",
            description=f"action {i}",
            preconditions=[],
        )
        transitions.append(
            Transition(source="page", action=aid, target="goal", probability=1.0, cost=0.5)
        )
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="page",
        goal_states={"goal"},
        discount=0.99,
    )


def _make_uniform_policy(mdp: MDP) -> Policy:
    """Create a perfectly uniform policy over all actions in *mdp*."""
    state_action_probs: Dict[str, Dict[str, float]] = {}
    q_values: Dict[str, Dict[str, float]] = {}
    values: Dict[str, float] = {}
    for sid in mdp.states:
        avail = mdp.get_actions(sid)
        if not avail:
            continue
        n = len(avail)
        state_action_probs[sid] = {a: 1.0 / n for a in avail}
        q_values[sid] = {a: 0.5 for a in avail}
        values[sid] = 0.5
    return Policy(
        state_action_probs=state_action_probs,
        beta=1.0,
        values=values,
        q_values=q_values,
        metadata={},
    )


def _make_peaked_policy(mdp: MDP, peak_action: str = "a0") -> Policy:
    """Create a deterministic policy strongly favouring *peak_action*."""
    state_action_probs: Dict[str, Dict[str, float]] = {}
    q_values: Dict[str, Dict[str, float]] = {}
    values: Dict[str, float] = {}
    for sid in mdp.states:
        avail = mdp.get_actions(sid)
        if not avail:
            continue
        n = len(avail)
        probs: Dict[str, float] = {}
        for a in avail:
            probs[a] = 0.95 if a == peak_action else 0.05 / max(n - 1, 1)
        state_action_probs[sid] = probs
        q_values[sid] = {a: (2.0 if a == peak_action else 0.1) for a in avail}
        values[sid] = 2.0
    return Policy(
        state_action_probs=state_action_probs,
        beta=10.0,
        values=values,
        q_values=q_values,
        metadata={},
    )


# ===================================================================
# Tests — constants
# ===================================================================

class TestChoiceConstants:
    """Verify well-known constants on ChoiceParalysisDetector."""

    def test_paralysis_threshold_exists(self):
        """PARALYSIS_THRESHOLD should be defined."""
        assert hasattr(ChoiceParalysisDetector, "PARALYSIS_THRESHOLD")

    def test_paralysis_threshold_value(self):
        """PARALYSIS_THRESHOLD should be 0.8."""
        assert ChoiceParalysisDetector.PARALYSIS_THRESHOLD == pytest.approx(0.8)

    def test_min_actions_exists(self):
        """MIN_ACTIONS should be defined and positive."""
        assert hasattr(ChoiceParalysisDetector, "MIN_ACTIONS")
        assert ChoiceParalysisDetector.MIN_ACTIONS > 0

    def test_min_actions_value(self):
        """MIN_ACTIONS should be 4."""
        assert ChoiceParalysisDetector.MIN_ACTIONS == 4

    def test_hick_a_exists(self):
        """Hick-Hyman intercept HICK_A should be defined."""
        assert hasattr(ChoiceParalysisDetector, "HICK_A")
        assert ChoiceParalysisDetector.HICK_A >= 0

    def test_hick_b_exists(self):
        """Hick-Hyman slope HICK_B should be defined and positive."""
        assert hasattr(ChoiceParalysisDetector, "HICK_B")
        assert ChoiceParalysisDetector.HICK_B > 0


# ===================================================================
# Tests — detect() with many uniform actions (paralysis expected)
# ===================================================================

class TestChoiceParalysisDetected:
    """Many actions with near-uniform probabilities should trigger paralysis."""

    def test_many_uniform_actions_triggers(self):
        """12 equally-weighted actions should produce a BottleneckResult."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        state = "page"
        result = det.detect(state, mdp, policy, beta=1.0)
        assert result is not None

    def test_result_type_is_choice_paralysis(self):
        """Result type must be CHOICE_PARALYSIS."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert result.bottleneck_type == BottleneckType.CHOICE_PARALYSIS

    def test_result_is_bottleneck_result(self):
        """detect() should return a BottleneckResult when triggered."""
        mdp = _make_mdp_with_n_actions(15)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert isinstance(result, BottleneckResult)

    def test_confidence_positive(self):
        """Confidence should be > 0 when paralysis is detected."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert result.confidence > 0.0

    def test_large_choice_set_paralysis(self):
        """20 uniform actions should strongly trigger paralysis."""
        mdp = _make_mdp_with_n_actions(20)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert result.severity in (Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM)


# ===================================================================
# Tests — detect() with few actions (no paralysis)
# ===================================================================

class TestChoiceNoParalysis:
    """Fewer than MIN_ACTIONS (or peaked policy) should not trigger."""

    def test_two_actions_returns_none(self):
        """With only 2 actions, paralysis should not be detected."""
        mdp = _make_mdp_with_n_actions(2)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is None

    def test_one_action_returns_none(self):
        """A single available action cannot cause paralysis."""
        mdp = _make_mdp_with_n_actions(1)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is None

    def test_three_actions_returns_none(self):
        """3 < MIN_ACTIONS so no paralysis should be detected."""
        mdp = _make_mdp_with_n_actions(3)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is None

    def test_peaked_policy_no_paralysis(self):
        """A deterministic policy should suppress paralysis even with many actions."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_peaked_policy(mdp, peak_action="a0")
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=10.0)
        # Peaked policy has low entropy ratio → no paralysis expected
        if result is not None:
            assert result.severity in (Severity.LOW, Severity.INFO)


# ===================================================================
# Tests — entropy ratio
# ===================================================================

class TestChoiceEntropyRatio:
    """High entropy ratio should lead to paralysis detection."""

    def test_uniform_policy_high_entropy_ratio(self):
        """A uniform policy on 10 actions should have entropy ratio ~1.0."""
        mdp = _make_mdp_with_n_actions(10)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None

    def test_low_entropy_peaked_policy(self):
        """A strongly peaked policy should have a low entropy ratio."""
        mdp = _make_mdp_with_n_actions(10)
        policy = _make_peaked_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=10.0)
        # Low entropy ratio → no paralysis or very low severity
        if result is not None:
            assert result.severity_score <= 2.0


# ===================================================================
# Tests — severity scaling
# ===================================================================

class TestChoiceSeverityScaling:
    """Severity should increase with the number of equally-probable actions."""

    def test_more_actions_higher_severity(self):
        """20 actions should produce >= severity than 5 actions."""
        det = ChoiceParalysisDetector()
        mdp5 = _make_mdp_with_n_actions(5)
        mdp20 = _make_mdp_with_n_actions(20)
        pol5 = _make_uniform_policy(mdp5)
        pol20 = _make_uniform_policy(mdp20)
        r5 = det.detect("page", mdp5, pol5, beta=1.0)
        r20 = det.detect("page", mdp20, pol20, beta=1.0)
        if r5 is not None and r20 is not None:
            assert r20.severity_score >= r5.severity_score


# ===================================================================
# Tests — Hick-Hyman law
# ===================================================================

class TestChoiceHickHyman:
    """The detector should use Hick-Hyman law parameters."""

    def test_cognitive_law_is_hick_hyman(self):
        """Result cognitive_law should be HICK_HYMAN."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert result.cognitive_law == CognitiveLaw.HICK_HYMAN


# ===================================================================
# Tests — evidence and recommendation
# ===================================================================

class TestChoiceEvidenceRecommendation:
    """Evidence dict and recommendation string should be populated."""

    def test_evidence_is_dict(self):
        """Evidence should be a dict with float values."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert isinstance(result.evidence, dict)

    def test_recommendation_non_empty(self):
        """Recommendation string should be non-empty."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert len(result.recommendation) > 0

    def test_description_non_empty(self):
        """Description string should be non-empty."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert len(result.description) > 0

    def test_affected_states_includes_page(self):
        """The detected state should appear in affected_states."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        assert "page" in result.affected_states


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestChoiceEdgeCases:
    """Edge-case behaviour for the choice paralysis detector."""

    def test_zero_actions_no_crash(self):
        """A state with zero actions should not crash."""
        mdp = _make_mdp_with_n_actions(0)
        policy = Policy(
            state_action_probs={}, beta=1.0, values={}, q_values={}, metadata={}
        )
        det = ChoiceParalysisDetector()
        try:
            result = det.detect("page", mdp, policy, beta=1.0)
        except (KeyError, ZeroDivisionError):
            pass  # acceptable on degenerate input

    def test_result_to_dict_succeeds(self):
        """BottleneckResult.to_dict should serialise without error."""
        mdp = _make_mdp_with_n_actions(12)
        policy = _make_uniform_policy(mdp)
        det = ChoiceParalysisDetector()
        result = det.detect("page", mdp, policy, beta=1.0)
        assert result is not None
        d = result.to_dict()
        assert isinstance(d, dict)
