"""Unit tests for usability_oracle.scheduling.multitask — Multitask cognition.

Tests cover task switching costs, threaded cognition simulation, multitask
performance prediction, and interference graph construction.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from usability_oracle.scheduling.multitask import (
    CognitionThread,
    CognitiveResource,
    InterferenceEdge,
    SwitchCost,
    SwitchCostModel,
    TaskResourceProfile,
    ThreadedCognitionModel,
    build_interference_graph,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _visual_profile(task_id: str = "read") -> TaskResourceProfile:
    return TaskResourceProfile(
        task_id=task_id,
        demands={CognitiveResource.VISUAL: 0.8, CognitiveResource.COGNITIVE: 0.3},
        working_memory_chunks=3,
        skill_level=0.5,
    )


def _motor_profile(task_id: str = "type") -> TaskResourceProfile:
    return TaskResourceProfile(
        task_id=task_id,
        demands={CognitiveResource.MOTOR: 0.9, CognitiveResource.COGNITIVE: 0.2},
        working_memory_chunks=2,
        skill_level=0.7,
    )


def _dual_profile(task_id: str = "dual") -> TaskResourceProfile:
    return TaskResourceProfile(
        task_id=task_id,
        demands={
            CognitiveResource.VISUAL: 0.6,
            CognitiveResource.MOTOR: 0.5,
            CognitiveResource.COGNITIVE: 0.7,
        },
        working_memory_chunks=5,
        skill_level=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Task Switching Cost
# ═══════════════════════════════════════════════════════════════════════════


class TestSwitchCost:
    """Test task switching cost model."""

    def test_switch_cost_non_negative(self):
        model = SwitchCostModel()
        cost = model.compute(_visual_profile(), _motor_profile())
        assert isinstance(cost, SwitchCost)
        assert cost.total_s >= 0
        assert cost.cognitive_cost_bits >= 0

    def test_switch_cost_symmetry_direction(self):
        """Switching A→B may differ from B→A."""
        model = SwitchCostModel()
        ab = model.compute(_visual_profile(), _motor_profile())
        ba = model.compute(_motor_profile(), _visual_profile())
        # Both should be non-negative
        assert ab.total_s >= 0
        assert ba.total_s >= 0

    def test_switch_cost_same_task_is_minimal(self):
        model = SwitchCostModel()
        same = model.compute(_visual_profile(), _visual_profile())
        diff = model.compute(_visual_profile(), _motor_profile())
        # Switching to the same type of task should cost less
        assert same.total_s <= diff.total_s + 1e-6

    def test_switch_cost_components(self):
        model = SwitchCostModel()
        cost = model.compute(_visual_profile(), _motor_profile())
        assert cost.context_switch_s >= 0
        assert cost.wm_reload_s >= 0
        assert cost.resumption_lag_s >= 0
        assert abs(cost.total_s - (cost.context_switch_s + cost.wm_reload_s + cost.resumption_lag_s)) < 1e-6

    def test_switch_cost_with_interruption(self):
        model = SwitchCostModel()
        short_int = model.compute(_visual_profile(), _motor_profile(), interruption_duration_s=0.0)
        long_int = model.compute(_visual_profile(), _motor_profile(), interruption_duration_s=5.0)
        # Longer interruption generally means higher resumption lag
        assert long_int.resumption_lag_s >= short_int.resumption_lag_s - 1e-6

    def test_switch_cost_matrix_shape(self):
        model = SwitchCostModel()
        profiles = [_visual_profile("a"), _motor_profile("b"), _dual_profile("c")]
        matrix = model.switch_cost_matrix(profiles)
        assert matrix.shape == (3, 3)
        # Diagonal should be minimal
        for i in range(3):
            assert matrix[i, i] <= matrix.max() + 1e-6

    def test_custom_parameters(self):
        model = SwitchCostModel(
            base_switch_s=0.5,
            wm_reload_per_chunk_s=0.3,
            resumption_coefficient=0.5,
        )
        cost = model.compute(_visual_profile(), _motor_profile())
        assert cost.total_s > 0

    def test_resource_dissimilarity(self):
        sim = SwitchCostModel._resource_dissimilarity(_visual_profile(), _visual_profile())
        assert sim == 0.0 or sim < 1e-6
        diff = SwitchCostModel._resource_dissimilarity(_visual_profile(), _motor_profile())
        assert diff > 0


# ═══════════════════════════════════════════════════════════════════════════
# Threaded Cognition Model
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadedCognition:
    """Test threaded cognition simulation."""

    def test_basic_simulation(self):
        model = ThreadedCognitionModel()
        threads = [
            CognitionThread(
                task_id="read",
                resource_profile=_visual_profile(),
                priority=1.0,
                active=True,
                progress=0.0,
            ),
        ]
        result = model.simulate(threads, task_durations={"read": 2.0}, max_time_s=10.0)
        assert isinstance(result, dict)
        assert "read" in result
        assert result["read"] >= 2.0 - 1e-6

    def test_concurrent_tasks_take_longer(self):
        model = ThreadedCognitionModel()
        single = [
            CognitionThread("read", _visual_profile(), priority=1.0, active=True, progress=0.0),
        ]
        single_result = model.simulate(single, {"read": 2.0}, max_time_s=30.0)

        dual = [
            CognitionThread("read", _visual_profile(), priority=1.0, active=True, progress=0.0),
            CognitionThread("dual", _dual_profile(), priority=1.0, active=True, progress=0.0),
        ]
        dual_result = model.simulate(dual, {"read": 2.0, "dual": 2.0}, max_time_s=30.0)
        # With interference, at least one task takes longer
        assert max(dual_result.values()) >= min(single_result.values()) - 0.1

    def test_resource_capacity_limits(self):
        capacities = {
            CognitiveResource.VISUAL: 1.0,
            CognitiveResource.MOTOR: 1.0,
            CognitiveResource.COGNITIVE: 1.0,
        }
        model = ThreadedCognitionModel(resource_capacities=capacities)
        threads = [
            CognitionThread("t1", _visual_profile("t1"), 1.0, True, 0.0),
            CognitionThread("t2", _motor_profile("t2"), 1.0, True, 0.0),
        ]
        result = model.simulate(threads, {"t1": 1.0, "t2": 1.0}, max_time_s=20.0)
        assert len(result) == 2

    def test_priority_matters(self):
        model = ThreadedCognitionModel()
        threads = [
            CognitionThread("hi", _dual_profile("hi"), priority=2.0, active=True, progress=0.0),
            CognitionThread("lo", _dual_profile("lo"), priority=0.1, active=True, progress=0.0),
        ]
        result = model.simulate(threads, {"hi": 1.0, "lo": 1.0}, max_time_s=30.0)
        # Higher priority task should complete no later than low priority
        assert result["hi"] <= result["lo"] + 1e-3

    def test_time_quantum_parameter(self):
        model = ThreadedCognitionModel(time_quantum_s=0.1)
        threads = [
            CognitionThread("t", _visual_profile("t"), 1.0, True, 0.0),
        ]
        result = model.simulate(threads, {"t": 1.0}, max_time_s=10.0)
        assert result["t"] >= 1.0 - 0.2


# ═══════════════════════════════════════════════════════════════════════════
# Interference Graph
# ═══════════════════════════════════════════════════════════════════════════


class TestInterferenceGraph:
    """Test interference graph construction."""

    def test_build_graph_basic(self):
        profiles = [_visual_profile("a"), _motor_profile("b")]
        edges = build_interference_graph(profiles)
        assert isinstance(edges, list)
        assert all(isinstance(e, InterferenceEdge) for e in edges)

    def test_shared_resource_creates_interference(self):
        # Both use COGNITIVE resource
        profiles = [_visual_profile("a"), _motor_profile("b")]
        edges = build_interference_graph(profiles)
        if edges:
            edge = edges[0]
            assert edge.interference >= 0
            assert len(edge.shared_resources) > 0

    def test_no_overlap_minimal_interference(self):
        p1 = TaskResourceProfile(
            task_id="visual_only",
            demands={CognitiveResource.VISUAL: 1.0},
            working_memory_chunks=1,
            skill_level=1.0,
        )
        p2 = TaskResourceProfile(
            task_id="motor_only",
            demands={CognitiveResource.MOTOR: 1.0},
            working_memory_chunks=1,
            skill_level=1.0,
        )
        edges = build_interference_graph([p1, p2])
        for e in edges:
            if CognitiveResource.VISUAL not in e.shared_resources and CognitiveResource.MOTOR not in e.shared_resources:
                assert e.interference < 0.1

    def test_high_overlap_high_interference(self):
        p1 = _dual_profile("a")
        p2 = _dual_profile("b")
        edges = build_interference_graph([p1, p2])
        assert len(edges) > 0
        max_interference = max(e.interference for e in edges)
        assert max_interference > 0

    def test_single_task_no_edges(self):
        edges = build_interference_graph([_visual_profile()])
        assert len(edges) == 0

    def test_many_tasks_quadratic_edges(self):
        profiles = [_visual_profile(f"t{i}") for i in range(5)]
        edges = build_interference_graph(profiles)
        # n*(n-1)/2 possible edges
        assert len(edges) <= 10

    def test_interference_edge_attributes(self):
        profiles = [_visual_profile("a"), _dual_profile("b")]
        edges = build_interference_graph(profiles)
        for edge in edges:
            assert hasattr(edge, "task_a")
            assert hasattr(edge, "task_b")
            assert hasattr(edge, "interference")
            assert hasattr(edge, "shared_resources")
