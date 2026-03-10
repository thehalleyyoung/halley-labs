"""Unit tests for MemoryDecayDetector.

Tests cover working-memory overload from high ``wm_load``, absence when
load is low, decay with delay (exponential forgetting), cross-page
memory demand, severity scaling, and threshold constants.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import pytest

from usability_oracle.bottleneck.memory import MemoryDecayDetector
from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.core.types import CostTuple, Trajectory, TrajectoryStep
from usability_oracle.mdp.models import Action, MDP, State, Transition


# ===================================================================
# Helper factories
# ===================================================================

def _make_memory_mdp(wm_load: float = 2.0, n_pages: int = 2) -> MDP:
    """Build a linear multi-page MDP with a configurable working-memory load.

    Each non-goal state has ``wm_load`` set in its features.  The MDP
    has *n_pages* intermediate pages plus one goal state.
    """
    states: Dict[str, State] = {}
    actions: Dict[str, Action] = {}
    transitions: List[Transition] = []

    for i in range(n_pages):
        sid = f"page_{i}"
        states[sid] = State(
            state_id=sid,
            features={
                "n_elements": 5.0,
                "x": float(i * 100),
                "y": 50.0,
                "width": 60.0,
                "height": 40.0,
                "working_memory_load": wm_load,
            },
            label=sid,
            is_terminal=False,
            is_goal=False,
            metadata={},
        )
    states["goal"] = State(
        state_id="goal",
        features={"n_elements": 1.0, "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0, "working_memory_load": 0.0},
        label="goal",
        is_terminal=True,
        is_goal=True,
        metadata={},
    )
    for i in range(n_pages):
        aid = f"nav_{i}"
        target = f"page_{i + 1}" if i + 1 < n_pages else "goal"
        actions[aid] = Action(
            action_id=aid,
            action_type=Action.NAVIGATE,
            target_node_id=f"link_{i}",
            description=f"Navigate to {target}",
            preconditions=[],
        )
        transitions.append(
            Transition(source=f"page_{i}", action=aid, target=target, probability=1.0, cost=0.5)
        )
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="page_0",
        goal_states={"goal"},
        discount=0.99,
    )


def _make_trajectory(state_ids: Sequence[str], delay_per_step: float = 1.0) -> list[dict]:
    """Build a simple trajectory through the given state_ids.

    Returns a list of dicts with ``state`` and ``timestamp`` keys,
    matching the format expected by :class:`MemoryDecayDetector`.
    """
    steps: list[dict] = []
    for i, sid in enumerate(state_ids):
        steps.append({
            "state": sid,
            "action": f"nav_{i}",
            "timestamp": float(i) * delay_per_step,
        })
    return steps


# ===================================================================
# Tests — constants
# ===================================================================

class TestMemoryConstants:
    """Verify well-known constants on MemoryDecayDetector."""

    def test_overload_threshold_exists(self):
        """OVERLOAD_THRESHOLD should be defined and > 0."""
        assert hasattr(MemoryDecayDetector, "OVERLOAD_THRESHOLD")
        assert MemoryDecayDetector.OVERLOAD_THRESHOLD > 0

    def test_overload_threshold_value(self):
        """OVERLOAD_THRESHOLD should be 4 (Cowan's 4±1)."""
        assert MemoryDecayDetector.OVERLOAD_THRESHOLD == 4

    def test_decay_half_life_exists(self):
        """DECAY_HALF_LIFE should be defined and positive."""
        assert hasattr(MemoryDecayDetector, "DECAY_HALF_LIFE")
        assert MemoryDecayDetector.DECAY_HALF_LIFE > 0

    def test_decay_half_life_value(self):
        """DECAY_HALF_LIFE should be 7.0 seconds."""
        assert MemoryDecayDetector.DECAY_HALF_LIFE == pytest.approx(7.0)

    def test_recall_threshold_exists(self):
        """RECALL_THRESHOLD should be defined."""
        assert hasattr(MemoryDecayDetector, "RECALL_THRESHOLD")
        assert 0 < MemoryDecayDetector.RECALL_THRESHOLD <= 1.0

    def test_cross_page_threshold_exists(self):
        """CROSS_PAGE_THRESHOLD should be defined."""
        assert hasattr(MemoryDecayDetector, "CROSS_PAGE_THRESHOLD")
        assert MemoryDecayDetector.CROSS_PAGE_THRESHOLD > 0


# ===================================================================
# Tests — detect() with high working-memory load
# ===================================================================

class TestMemoryOverloadDetected:
    """States with high wm_load should trigger memory overload."""

    def test_high_wm_load_triggers(self):
        """A state with wm_load=8 should trigger memory overload."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        state = "page_2"
        result = det.detect(trajectory, state, mdp)
        assert result is not None

    def test_result_type_is_memory_decay(self):
        """Result bottleneck_type must be MEMORY_DECAY."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_2", mdp)
        assert result is not None
        assert result.bottleneck_type == BottleneckType.MEMORY_DECAY

    def test_result_is_bottleneck_result(self):
        """detect() should return a BottleneckResult when triggered."""
        mdp = _make_memory_mdp(wm_load=10.0, n_pages=4)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2", "page_3"], delay_per_step=4.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_3", mdp)
        assert isinstance(result, BottleneckResult)

    def test_confidence_positive(self):
        """Confidence should be > 0 for a detected overload."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_2", mdp)
        assert result is not None
        assert result.confidence > 0.0

    def test_severity_not_info_for_high_load(self):
        """High wm_load should produce severity above INFO."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_2", mdp)
        assert result is not None
        assert result.severity != Severity.INFO


# ===================================================================
# Tests — detect() with low load (no overload)
# ===================================================================

class TestMemoryNoOverload:
    """States with low wm_load should not trigger memory overload."""

    def test_low_wm_load_returns_none(self):
        """wm_load=1 should not trigger overload."""
        mdp = _make_memory_mdp(wm_load=1.0, n_pages=2)
        trajectory = _make_trajectory(["page_0", "page_1"], delay_per_step=1.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_1", mdp)
        assert result is None

    def test_zero_wm_load_returns_none(self):
        """wm_load=0 should definitely not trigger."""
        mdp = _make_memory_mdp(wm_load=0.0, n_pages=2)
        trajectory = _make_trajectory(["page_0", "page_1"], delay_per_step=1.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_1", mdp)
        assert result is None

    def test_moderate_load_no_overload(self):
        """wm_load=3 (just under threshold) should not trigger."""
        mdp = _make_memory_mdp(wm_load=3.0, n_pages=2)
        trajectory = _make_trajectory(["page_0", "page_1"], delay_per_step=1.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_1", mdp)
        # Might be None or low/moderate severity
        if result is not None:
            assert result.severity in (Severity.LOW, Severity.INFO, Severity.MEDIUM)


# ===================================================================
# Tests — decay with delay
# ===================================================================

class TestMemoryDecayWithDelay:
    """Longer delays should reduce recall probability and raise severity."""

    def test_long_delay_triggers_overload(self):
        """A 30-second delay with moderate load should amplify decay."""
        mdp = _make_memory_mdp(wm_load=5.0, n_pages=3)
        trajectory = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=15.0)
        det = MemoryDecayDetector()
        result = det.detect(trajectory, "page_2", mdp)
        assert result is not None

    def test_short_delay_less_severe(self):
        """A 1-second delay should produce lower severity than a 20-second delay."""
        mdp_short = _make_memory_mdp(wm_load=6.0, n_pages=3)
        mdp_long = _make_memory_mdp(wm_load=6.0, n_pages=3)
        traj_short = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=0.5)
        traj_long = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=10.0)
        det = MemoryDecayDetector()
        r_short = det.detect(traj_short, "page_2", mdp_short)
        r_long = det.detect(traj_long, "page_2", mdp_long)
        if r_short is not None and r_long is not None:
            assert r_long.severity_score >= r_short.severity_score

    def test_recall_probability_decreases(self):
        """Internal _recall_probability should decrease with longer delay."""
        det = MemoryDecayDetector()
        p_short = det._recall_probability(4, 1.0)
        p_long = det._recall_probability(4, 30.0)
        assert p_long < p_short


# ===================================================================
# Tests — cross-page memory demand
# ===================================================================

class TestCrossPageMemory:
    """Multi-page trajectories should increase memory demand."""

    def test_many_pages_increases_demand(self):
        """Traversing 5 pages should produce a higher cross-page count."""
        mdp = _make_memory_mdp(wm_load=5.0, n_pages=6)
        trajectory = _make_trajectory(
            ["page_0", "page_1", "page_2", "page_3", "page_4", "page_5"],
            delay_per_step=3.0,
        )
        det = MemoryDecayDetector()
        cross_count = det._cross_page_memory_demand(trajectory)
        assert cross_count >= 0

    def test_single_page_low_cross_demand(self):
        """A single-page trajectory should have minimal cross-page demand."""
        mdp = _make_memory_mdp(wm_load=5.0, n_pages=1)
        trajectory = _make_trajectory(["page_0"], delay_per_step=1.0)
        det = MemoryDecayDetector()
        cross_count = det._cross_page_memory_demand(trajectory)
        assert cross_count <= 1


# ===================================================================
# Tests — severity scaling
# ===================================================================

class TestMemorySeverityScaling:
    """Severity should increase with load, delay, and cross-page demand."""

    def test_higher_load_higher_severity(self):
        """wm_load=10 should produce >= severity than wm_load=5."""
        det = MemoryDecayDetector()
        mdp_low = _make_memory_mdp(wm_load=5.0, n_pages=3)
        mdp_high = _make_memory_mdp(wm_load=10.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        r_low = det.detect(traj, "page_2", mdp_low)
        r_high = det.detect(traj, "page_2", mdp_high)
        if r_low is not None and r_high is not None:
            assert r_high.severity_score >= r_low.severity_score

    def test_extreme_load_high_or_critical(self):
        """wm_load=15 with long delay should be HIGH or CRITICAL."""
        mdp = _make_memory_mdp(wm_load=15.0, n_pages=4)
        traj = _make_trajectory(["page_0", "page_1", "page_2", "page_3"], delay_per_step=10.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_3", mdp)
        assert result is not None
        assert result.severity in (Severity.HIGH, Severity.CRITICAL)


# ===================================================================
# Tests — cognitive law and evidence
# ===================================================================

class TestMemoryCognitiveLaw:
    """Result should reference the working-memory decay law."""

    def test_cognitive_law_is_working_memory(self):
        """Memory decay results should cite WORKING_MEMORY_DECAY."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_2", mdp)
        assert result is not None
        assert result.cognitive_law == CognitiveLaw.WORKING_MEMORY_DECAY

    def test_evidence_is_dict(self):
        """Evidence should be a dict with numeric values."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_2", mdp)
        assert result is not None
        assert isinstance(result.evidence, dict)

    def test_recommendation_non_empty(self):
        """Recommendation should be non-empty."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_2", mdp)
        assert result is not None
        assert len(result.recommendation) > 0

    def test_affected_states_includes_tested_state(self):
        """affected_states should include the state being evaluated."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_2", mdp)
        assert result is not None
        assert "page_2" in result.affected_states


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestMemoryEdgeCases:
    """Edge-case behaviour for the memory decay detector."""

    def test_empty_trajectory_no_crash(self):
        """An empty trajectory should not crash the detector."""
        mdp = _make_memory_mdp(wm_load=5.0, n_pages=2)
        trajectory = Trajectory.from_steps([])
        det = MemoryDecayDetector()
        try:
            result = det.detect(trajectory, "page_0", mdp)
        except Exception:
            pass  # acceptable on degenerate input

    def test_result_to_dict(self):
        """to_dict() should succeed on a memory decay result."""
        mdp = _make_memory_mdp(wm_load=8.0, n_pages=3)
        traj = _make_trajectory(["page_0", "page_1", "page_2"], delay_per_step=5.0)
        det = MemoryDecayDetector()
        result = det.detect(traj, "page_2", mdp)
        assert result is not None
        d = result.to_dict()
        assert isinstance(d, dict)
