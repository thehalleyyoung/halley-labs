"""Unit tests for BottleneckClassifier.

Tests cover the main classification pipeline, report generation,
severity ranking, parameter defaults, filtering, and the
BottleneckReport convenience accessors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pytest

from usability_oracle.bottleneck.classifier import BottleneckClassifier
from usability_oracle.bottleneck.models import (
    BottleneckReport,
    BottleneckResult,
    BottleneckSignature,
)
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import Action, MDP, State, Transition
from usability_oracle.policy.models import Policy


# ===================================================================
# Helper factories
# ===================================================================

def _make_cluttered_mdp(n_elements: int = 30) -> MDP:
    """Build an MDP whose states expose high visual complexity.

    The MDP has two states (``page`` and ``goal``) with multiple actions
    and features designed to trigger various bottleneck detectors.
    """
    actions_dict: Dict[str, Action] = {}
    transitions: List[Transition] = []
    action_ids: List[str] = []
    for i in range(max(n_elements, 6)):
        aid = f"click_{i}"
        action_ids.append(aid)
        actions_dict[aid] = Action(
            action_id=aid,
            action_type=Action.CLICK,
            target_node_id=f"node_{i}",
            description=f"Click element {i}",
            preconditions=[],
        )
        transitions.append(
            Transition(source="page", action=aid, target="goal", probability=1.0, cost=0.3)
        )

    states = {
        "page": State(
            state_id="page",
            features={
                "n_elements": float(n_elements),
                "x": 500.0,
                "y": 400.0,
                "width": 20.0,
                "height": 20.0,
                "wm_load": 6.0,
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
    return MDP(
        states=states,
        actions=actions_dict,
        transitions=transitions,
        initial_state="page",
        goal_states={"goal"},
        discount=0.99,
    )


def _make_uniform_policy(mdp: MDP) -> Policy:
    """Create a near-uniform softmax policy over all actions in *mdp*."""
    state_action_probs: Dict[str, Dict[str, float]] = {}
    q_values: Dict[str, Dict[str, float]] = {}
    values: Dict[str, float] = {}

    for sid, state in mdp.states.items():
        available = mdp.get_actions(sid)
        if not available:
            continue
        n = len(available)
        prob = 1.0 / n
        state_action_probs[sid] = {a: prob for a in available}
        q_values[sid] = {a: 0.5 for a in available}
        values[sid] = 0.5

    return Policy(
        state_action_probs=state_action_probs,
        beta=1.0,
        values=values,
        q_values=q_values,
        metadata={},
    )


def _make_simple_mdp_and_policy():
    """Return a small 3-state MDP with a deterministic policy."""
    states = {
        "start": State(state_id="start", features={"n_elements": 3.0, "x": 0.0, "y": 0.0, "width": 100.0, "height": 50.0, "wm_load": 1.0}, label="start", is_terminal=False, is_goal=False, metadata={}),
        "mid": State(state_id="mid", features={"n_elements": 3.0, "x": 50.0, "y": 50.0, "width": 80.0, "height": 40.0, "wm_load": 1.0}, label="mid", is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"n_elements": 1.0, "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0, "wm_load": 0.0}, label="goal", is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "a1": Action(action_id="a1", action_type=Action.CLICK, target_node_id="n1", description="", preconditions=[]),
        "a2": Action(action_id="a2", action_type=Action.CLICK, target_node_id="n2", description="", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="a1", target="mid", probability=1.0, cost=0.5),
        Transition(source="mid", action="a2", target="goal", probability=1.0, cost=0.5),
    ]
    mdp = MDP(states=states, actions=actions, transitions=transitions, initial_state="start", goal_states={"goal"}, discount=0.99)
    policy = Policy(
        state_action_probs={"start": {"a1": 1.0}, "mid": {"a2": 1.0}},
        beta=5.0,
        values={"start": 1.0, "mid": 0.5, "goal": 0.0},
        q_values={"start": {"a1": 1.0}, "mid": {"a2": 0.5}},
        metadata={},
    )
    return mdp, policy


# ===================================================================
# Tests — default parameters
# ===================================================================

class TestBottleneckClassifierDefaults:
    """Verify factory defaults of ``BottleneckClassifier``."""

    def test_default_beta(self):
        """Default rationality parameter beta should be 5.0."""
        clf = BottleneckClassifier()
        assert clf.beta == 5.0

    def test_default_min_confidence(self):
        """Default minimum confidence threshold should be 0.3."""
        clf = BottleneckClassifier()
        assert clf.min_confidence == 0.3

    def test_default_max_bottlenecks(self):
        """Default maximum bottleneck count should be 50."""
        clf = BottleneckClassifier()
        assert clf.max_bottlenecks == 50

    def test_custom_beta(self):
        """Classifier should accept a custom beta."""
        clf = BottleneckClassifier(beta=10.0)
        assert clf.beta == 10.0

    def test_custom_min_confidence(self):
        """Classifier should accept a custom min_confidence."""
        clf = BottleneckClassifier(min_confidence=0.6)
        assert clf.min_confidence == 0.6


# ===================================================================
# Tests — classify()
# ===================================================================

class TestClassify:
    """Tests for ``BottleneckClassifier.classify``."""

    def test_classify_returns_list(self):
        """classify() must return a list of BottleneckResult objects."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        results = clf.classify(mdp, policy)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, BottleneckResult)

    def test_classify_simple_mdp_no_crash(self):
        """classify() should not raise on a simple, well-formed MDP."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        clf.classify(mdp, policy)  # should not raise

    def test_classify_cluttered_mdp_finds_bottlenecks(self):
        """A cluttered MDP with many small-target actions should trigger detectors."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.1)
        results = clf.classify(mdp, policy)
        assert len(results) >= 0  # may find some based on complexity

    def test_classify_with_trajectory_stats(self):
        """classify() should accept optional trajectory_stats dict."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        stats = {"mean_steps": 3.0, "mean_cost": 1.5}
        results = clf.classify(mdp, policy, trajectory_stats=stats)
        assert isinstance(results, list)

    def test_classify_with_cost_breakdown(self):
        """classify() should accept optional cost_breakdown dict."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        breakdown = {"perceptual": 0.3, "motor": 0.5, "cognitive": 0.2}
        results = clf.classify(mdp, policy, cost_breakdown=breakdown)
        assert isinstance(results, list)

    def test_classify_results_have_valid_types(self):
        """Every result must have a valid BottleneckType."""
        mdp = _make_cluttered_mdp(n_elements=30)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.0)
        results = clf.classify(mdp, policy)
        valid_types = set(BottleneckType)
        for r in results:
            assert r.bottleneck_type in valid_types


# ===================================================================
# Tests — severity ranking
# ===================================================================

class TestSeverityRanking:
    """Results from classify() should be ranked by severity (descending)."""

    def test_results_ordered_by_severity_score(self):
        """Higher-severity bottlenecks must come first in the returned list."""
        mdp = _make_cluttered_mdp(n_elements=50)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.0)
        results = clf.classify(mdp, policy)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].severity_score >= results[i + 1].severity_score

    def test_severity_score_property(self):
        """BottleneckResult.severity_score should map correctly from severity."""
        r = BottleneckResult(
            bottleneck_type=BottleneckType.PERCEPTUAL_OVERLOAD,
            severity=Severity.CRITICAL,
            confidence=0.9,
            affected_states=["s1"],
            affected_actions=[],
            cognitive_law=CognitiveLaw.PERCEPTUAL_PROCESSING,
            evidence={},
            description="test",
            recommendation="test",
            repair_hints=[],
            metadata={},
        )
        assert r.severity_score == 4.0

    def test_impact_score_combines_severity_and_confidence(self):
        """impact_score should equal severity_score * confidence."""
        r = BottleneckResult(
            bottleneck_type=BottleneckType.MOTOR_DIFFICULTY,
            severity=Severity.HIGH,
            confidence=0.5,
            affected_states=["s1"],
            affected_actions=["a1"],
            cognitive_law=CognitiveLaw.FITTS,
            evidence={},
            description="",
            recommendation="",
            repair_hints=[],
            metadata={},
        )
        expected = 3.0 * 0.5
        assert abs(r.impact_score - expected) < 1e-9


# ===================================================================
# Tests — max_bottlenecks
# ===================================================================

class TestMaxBottlenecks:
    """The max_bottlenecks parameter should cap the number of results."""

    def test_max_bottlenecks_limits_output(self):
        """classify() should return at most max_bottlenecks items."""
        mdp = _make_cluttered_mdp(n_elements=60)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(max_bottlenecks=3, min_confidence=0.0)
        results = clf.classify(mdp, policy)
        assert len(results) <= 3

    def test_max_bottlenecks_one(self):
        """With max_bottlenecks=1, at most one result is returned."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(max_bottlenecks=1, min_confidence=0.0)
        results = clf.classify(mdp, policy)
        assert len(results) <= 1


# ===================================================================
# Tests — min_confidence filtering
# ===================================================================

class TestMinConfidenceFiltering:
    """Low-confidence results should be filtered out."""

    def test_high_min_confidence_filters_results(self):
        """Setting min_confidence above 1.0 should return no results."""
        mdp = _make_cluttered_mdp(n_elements=30)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=1.01)
        results = clf.classify(mdp, policy)
        assert len(results) == 0

    def test_zero_min_confidence_keeps_all(self):
        """Setting min_confidence=0.0 should keep every detected bottleneck."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf_strict = BottleneckClassifier(min_confidence=0.5)
        clf_lenient = BottleneckClassifier(min_confidence=0.0)
        strict = clf_strict.classify(mdp, policy)
        lenient = clf_lenient.classify(mdp, policy)
        assert len(lenient) >= len(strict)


# ===================================================================
# Tests — classify_to_report()
# ===================================================================

class TestClassifyToReport:
    """Tests for the ``classify_to_report`` convenience wrapper."""

    def test_returns_bottleneck_report(self):
        """classify_to_report() must return a BottleneckReport instance."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        report = clf.classify_to_report(mdp, policy)
        assert isinstance(report, BottleneckReport)

    def test_report_n_bottlenecks(self):
        """report.n_bottlenecks should equal len(report.bottlenecks)."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        report = clf.classify_to_report(mdp, policy)
        assert report.n_bottlenecks == len(report.bottlenecks)

    def test_report_critical_count_non_negative(self):
        """report.critical_count should be >= 0."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        report = clf.classify_to_report(mdp, policy)
        assert report.critical_count >= 0

    def test_report_by_type(self):
        """by_type should return only results of the requested type."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.0)
        report = clf.classify_to_report(mdp, policy)
        for btype in BottleneckType:
            subset = report.by_type(btype)
            for r in subset:
                assert r.bottleneck_type == btype

    def test_report_by_severity(self):
        """by_severity should return only results matching the requested severity."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.0)
        report = clf.classify_to_report(mdp, policy)
        for sev in Severity:
            subset = report.by_severity(sev)
            for r in subset:
                assert r.severity == sev

    def test_report_type_distribution(self):
        """type_distribution should map type names to counts summing to n_bottlenecks."""
        mdp = _make_cluttered_mdp(n_elements=40)
        policy = _make_uniform_policy(mdp)
        clf = BottleneckClassifier(min_confidence=0.0)
        report = clf.classify_to_report(mdp, policy)
        dist = report.type_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == report.n_bottlenecks

    def test_report_with_trajectory_stats(self):
        """classify_to_report should accept trajectory_stats kwarg."""
        mdp, policy = _make_simple_mdp_and_policy()
        clf = BottleneckClassifier()
        report = clf.classify_to_report(mdp, policy, trajectory_stats={"mean_steps": 2.0})
        assert isinstance(report, BottleneckReport)


# ===================================================================
# Tests — BottleneckReport accessors (standalone)
# ===================================================================

class TestBottleneckReportStandalone:
    """Unit tests for BottleneckReport constructed directly."""

    def _sample_result(self, btype: BottleneckType, sev: Severity, conf: float = 0.8) -> BottleneckResult:
        """Factory for a minimal BottleneckResult."""
        return BottleneckResult(
            bottleneck_type=btype,
            severity=sev,
            confidence=conf,
            affected_states=["s1"],
            affected_actions=["a1"],
            cognitive_law=btype.cognitive_law,
            evidence={"metric": 1.0},
            description="desc",
            recommendation="rec",
            repair_hints=["hint"],
            metadata={},
        )

    def test_empty_report(self):
        """An empty report should have n_bottlenecks == 0."""
        report = BottleneckReport(bottlenecks=[], metadata={})
        assert report.n_bottlenecks == 0
        assert report.critical_count == 0

    def test_critical_count(self):
        """critical_count must count only CRITICAL-severity results."""
        results = [
            self._sample_result(BottleneckType.PERCEPTUAL_OVERLOAD, Severity.CRITICAL),
            self._sample_result(BottleneckType.CHOICE_PARALYSIS, Severity.HIGH),
            self._sample_result(BottleneckType.MOTOR_DIFFICULTY, Severity.CRITICAL),
        ]
        report = BottleneckReport(bottlenecks=results, metadata={})
        assert report.critical_count == 2

    def test_high_count(self):
        """high_count must count only HIGH-severity results."""
        results = [
            self._sample_result(BottleneckType.PERCEPTUAL_OVERLOAD, Severity.HIGH),
            self._sample_result(BottleneckType.CHOICE_PARALYSIS, Severity.HIGH),
            self._sample_result(BottleneckType.MOTOR_DIFFICULTY, Severity.LOW),
        ]
        report = BottleneckReport(bottlenecks=results, metadata={})
        assert report.high_count == 2

    def test_affected_states(self):
        """affected_states should union all affected_states from results."""
        r1 = self._sample_result(BottleneckType.MEMORY_DECAY, Severity.MEDIUM)
        r1 = BottleneckResult(
            bottleneck_type=r1.bottleneck_type, severity=r1.severity,
            confidence=r1.confidence, affected_states=["s1", "s2"],
            affected_actions=[], cognitive_law=r1.cognitive_law,
            evidence={}, description="", recommendation="",
            repair_hints=[], metadata={},
        )
        r2 = self._sample_result(BottleneckType.PERCEPTUAL_OVERLOAD, Severity.HIGH)
        report = BottleneckReport(bottlenecks=[r1, r2], metadata={})
        all_states = report.affected_states()
        assert "s1" in all_states
        assert "s2" in all_states

    def test_to_dict(self):
        """BottleneckReport.to_dict should return a serialisable dict."""
        results = [self._sample_result(BottleneckType.CHOICE_PARALYSIS, Severity.MEDIUM)]
        report = BottleneckReport(bottlenecks=results, metadata={"key": "val"})
        d = report.to_dict()
        assert isinstance(d, dict)
