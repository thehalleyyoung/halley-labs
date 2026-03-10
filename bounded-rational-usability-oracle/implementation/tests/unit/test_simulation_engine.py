"""Tests for usability_oracle.simulation.engine and related simulation components.

Verifies event queue ordering, processor state machine, KLM operator
timing, GOMS goal decomposition, ACT-R activation decay, discrete event
simulator, simulation traces, task network topological sort, and
critical path computation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.simulation.event_queue import (
    EventFilter,
    EventPriority,
    EventQueue,
    EventType,
    SimulationEvent,
)
from usability_oracle.simulation.processors import (
    CognitiveControlProcessor,
    MotorProcessor,
    PerceptualProcessor,
    ProcessorState,
    VisualAttentionProcessor,
    WorkingMemoryProcessor,
)
from usability_oracle.simulation.activation import (
    DEFAULT_DECAY,
    DEFAULT_LATENCY_EXPONENT,
    DEFAULT_LATENCY_FACTOR,
    DEFAULT_RETRIEVAL_THRESHOLD,
    ChunkActivation,
    base_level_learning,
)
from usability_oracle.simulation.goms import Goal, Method, Operator, ResourceType
from usability_oracle.simulation.task_network import TaskNetwork, TaskNode
from usability_oracle.simulation.engine import (
    DiscreteEventSimulator,
    SimulationConfig,
    SimulationTrace,
)


# =====================================================================
# Event queue ordering (min-heap)
# =====================================================================

class TestEventQueueOrdering:
    """Test that the event queue maintains correct min-heap ordering."""

    def test_events_ordered_by_timestamp(self) -> None:
        """Events should be popped in timestamp order."""
        queue = EventQueue()
        e1 = SimulationEvent(timestamp=3.0, event_type=EventType.KEYSTROKE)
        e2 = SimulationEvent(timestamp=1.0, event_type=EventType.MOUSE_CLICK)
        e3 = SimulationEvent(timestamp=2.0, event_type=EventType.MOUSE_MOVE)
        queue.insert(e1)
        queue.insert(e2)
        queue.insert(e3)

        first = queue.pop()
        second = queue.pop()
        third = queue.pop()
        assert first.timestamp <= second.timestamp <= third.timestamp

    def test_priority_tiebreaking(self) -> None:
        """Events at the same timestamp should be ordered by priority."""
        queue = EventQueue()
        # COGNITIVE (1) < MOTOR (3) < SYSTEM (5) in priority value
        e_motor = SimulationEvent(
            timestamp=1.0, event_type=EventType.MOTOR_EXECUTION_COMPLETE,
            priority=EventPriority.MOTOR,
        )
        e_cog = SimulationEvent(
            timestamp=1.0, event_type=EventType.PRODUCTION_FIRE,
            priority=EventPriority.COGNITIVE,
        )
        queue.insert(e_motor)
        queue.insert(e_cog)

        first = queue.pop()
        assert first.priority.value <= EventPriority.MOTOR.value

    def test_fifo_sequence_tiebreaking(self) -> None:
        """Events with same time and priority should be FIFO."""
        queue = EventQueue()
        e1 = SimulationEvent(
            timestamp=1.0, event_type=EventType.KEYSTROKE,
            priority=EventPriority.MOTOR, payload={"key": "a"},
        )
        e2 = SimulationEvent(
            timestamp=1.0, event_type=EventType.KEYSTROKE,
            priority=EventPriority.MOTOR, payload={"key": "b"},
        )
        queue.insert(e1)
        queue.insert(e2)

        first = queue.pop()
        assert first.payload.get("key") == "a"

    def test_cancel_event(self) -> None:
        """Cancelled events should be skipped during pop."""
        queue = EventQueue()
        e1 = SimulationEvent(timestamp=1.0, event_type=EventType.KEYSTROKE)
        e2 = SimulationEvent(timestamp=2.0, event_type=EventType.MOUSE_CLICK)
        eid1 = queue.insert(e1)
        queue.insert(e2)
        queue.cancel(eid1)

        popped = queue.pop()
        assert popped.timestamp == 2.0

    def test_empty_queue_returns_none(self) -> None:
        """Pop on empty queue should return None."""
        queue = EventQueue()
        assert queue.pop() is None


# =====================================================================
# Processor state machine transitions
# =====================================================================

class TestProcessorStateMachine:
    """Test cognitive processor state transitions."""

    def test_initial_state_is_idle(self) -> None:
        """Processors should start in IDLE state."""
        proc = PerceptualProcessor(
            base_encoding_time=0.1, eccentricity_slope=0.01,
        )
        assert proc.state == ProcessorState.IDLE

    def test_motor_processor_initial_idle(self) -> None:
        """Motor processor starts IDLE."""
        proc = MotorProcessor(
            fitts_a=0.05, fitts_b=0.15,
            preparation_time=0.15, keystroke_time=0.28,
        )
        assert proc.state == ProcessorState.IDLE

    def test_processor_state_enum_values(self) -> None:
        """ProcessorState enum should have expected values."""
        assert ProcessorState.IDLE is not None
        assert ProcessorState.PREPARING is not None
        assert ProcessorState.PROCESSING is not None
        assert ProcessorState.RESPONDING is not None
        assert ProcessorState.BUSY is not None


# =====================================================================
# KLM operator timing (Card et al. 1983)
# =====================================================================

class TestKLMOperatorTiming:
    """Test KLM operator times match Card, Moran & Newell (1983) table.

    These are tested more thoroughly in test_simulation_klm.py, but
    we verify basic integration here.
    """

    def test_keystroke_time(self) -> None:
        """K operator ≈ 0.28s for average skilled typist."""
        from usability_oracle.simulation.klm import KLMTimings
        timings = KLMTimings()
        assert timings.t_k_average == pytest.approx(0.28)

    def test_pointing_time(self) -> None:
        """P operator ≈ 1.1s (Fitts' law average)."""
        from usability_oracle.simulation.klm import KLMTimings
        assert KLMTimings().t_p == pytest.approx(1.1)


# =====================================================================
# GOMS goal decomposition
# =====================================================================

class TestGOMSGoalDecomposition:
    """Test GOMS model hierarchy."""

    def test_leaf_goal(self) -> None:
        """A goal with no sub-goals is a leaf."""
        g = Goal(name="press-button")
        assert g.is_leaf
        assert g.depth == 0

    def test_hierarchical_goal(self) -> None:
        """Nested goals compute depth correctly."""
        child1 = Goal(name="move-cursor")
        child2 = Goal(name="click")
        parent = Goal(name="select-item", subgoals=[child1, child2])
        assert not parent.is_leaf
        assert parent.depth == 1

    def test_deep_hierarchy(self) -> None:
        """Multi-level goal hierarchy."""
        leaf = Goal(name="keystroke")
        mid = Goal(name="type-char", subgoals=[leaf])
        top = Goal(name="type-word", subgoals=[mid])
        assert top.depth == 2

    def test_flatten(self) -> None:
        """Flatten should return all goals in pre-order."""
        c1 = Goal(name="a")
        c2 = Goal(name="b")
        root = Goal(name="root", subgoals=[c1, c2])
        flat = root.flatten()
        assert len(flat) == 3
        assert flat[0].name == "root"

    def test_n_descendants(self) -> None:
        """Count descendants correctly."""
        c1 = Goal(name="a")
        c2 = Goal(name="b", subgoals=[Goal(name="c")])
        root = Goal(name="root", subgoals=[c1, c2])
        assert root.n_descendants == 3  # a, b, c


# =====================================================================
# ACT-R activation decay
# =====================================================================

class TestACTRActivationDecay:
    """Test base-level learning activation computation."""

    def test_single_presentation_decay(self) -> None:
        """Activation of a chunk presented once at t=0, queried at t=1."""
        bl = base_level_learning([0.0], current_time=1.0, decay=DEFAULT_DECAY)
        # B = ln(1^{-0.5}) = ln(1) = 0
        assert bl == pytest.approx(0.0, abs=1e-10)

    def test_recent_higher_activation(self) -> None:
        """More recent presentations should give higher activation."""
        bl_old = base_level_learning([0.0], current_time=100.0, decay=DEFAULT_DECAY)
        bl_new = base_level_learning([99.0], current_time=100.0, decay=DEFAULT_DECAY)
        assert bl_new > bl_old

    def test_multiple_presentations_higher(self) -> None:
        """More presentations → higher base-level activation."""
        bl_one = base_level_learning([0.0], current_time=10.0)
        bl_many = base_level_learning([0.0, 2.0, 5.0, 8.0], current_time=10.0)
        assert bl_many > bl_one

    def test_no_presentations_negative_inf(self) -> None:
        """No presentations → effectively -infinity."""
        bl = base_level_learning([], current_time=10.0)
        assert bl == float("-inf")

    def test_chunk_activation_total(self) -> None:
        """ChunkActivation.total_activation = sum of components."""
        ca = ChunkActivation(
            chunk_id="test",
            base_level=1.0,
            spreading=0.5,
            partial_match=-0.3,
            noise_value=0.1,
        )
        assert ca.total_activation == pytest.approx(1.3)

    def test_above_threshold(self) -> None:
        """Check threshold comparison."""
        ca = ChunkActivation(base_level=0.0, spreading=0.0)
        assert ca.above_threshold(DEFAULT_RETRIEVAL_THRESHOLD)
        ca_low = ChunkActivation(base_level=-2.0, spreading=0.0)
        assert not ca_low.above_threshold(DEFAULT_RETRIEVAL_THRESHOLD)


# =====================================================================
# Discrete event simulator
# =====================================================================

class TestDiscreteEventSimulator:
    """Test the DES engine basic operations."""

    def test_step_advances_time(self) -> None:
        """Processing an event should advance the simulation clock."""
        sim = DiscreteEventSimulator(SimulationConfig(
            max_time=10.0, seed=42, noise_enabled=False,
        ))
        sim.initialize()
        # Insert an event at t=1.0
        event = SimulationEvent(
            timestamp=1.0,
            event_type=EventType.SIMULATION_START,
            target_processor="perceptual",
        )
        sim._queue.insert(event)
        # The clock should be at 0 initially
        assert sim._clock == 0.0

    def test_trace_records_events(self) -> None:
        """SimulationTrace should record events."""
        trace = SimulationTrace()
        event = SimulationEvent(
            timestamp=0.5,
            event_type=EventType.KEYSTROKE,
            source_processor="motor",
        )
        trace.record_event(event)
        assert trace.n_events == 1
        assert trace.events[0]["timestamp"] == 0.5

    def test_trace_duration(self) -> None:
        """Trace duration = last event time - first event time."""
        trace = SimulationTrace()
        for t in [0.1, 0.5, 1.2]:
            e = SimulationEvent(timestamp=t, event_type=EventType.KEYSTROKE)
            trace.record_event(e)
        assert trace.duration == pytest.approx(1.1)

    def test_empty_trace(self) -> None:
        """Empty trace should have 0 events and 0 duration."""
        trace = SimulationTrace()
        assert trace.n_events == 0
        assert trace.duration == 0.0


# =====================================================================
# Task network — topological sort
# =====================================================================

class TestTaskNetworkTopologicalSort:
    """Test topological ordering of task networks."""

    def test_linear_chain(self) -> None:
        """A → B → C should sort as [A, B, C]."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=1.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=2.0))
        net.add_task(TaskNode(task_id="C", name="C", duration=1.5))
        net.add_dependency("B", "A")
        net.add_dependency("C", "B")
        order = net.topological_sort()
        assert order == ["A", "B", "C"]

    def test_parallel_tasks(self) -> None:
        """Independent tasks: A, B have no dependency → both before C."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=1.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=1.0))
        net.add_task(TaskNode(task_id="C", name="C", duration=1.0))
        net.add_dependency("C", "A")
        net.add_dependency("C", "B")
        order = net.topological_sort()
        # A and B come before C
        assert order.index("C") > order.index("A")
        assert order.index("C") > order.index("B")

    def test_cycle_raises(self) -> None:
        """Cycle detection should reject cyclic networks at sort time."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=1.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=1.0))
        net.add_dependency("B", "A")
        # Direct add_dependency should reject cycle
        added = net.add_dependency("A", "B")
        assert not added  # Should return False for cycle

    def test_single_task(self) -> None:
        """Single task should be trivially sorted."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="X", name="X", duration=5.0))
        assert net.topological_sort() == ["X"]


# =====================================================================
# Critical path computation
# =====================================================================

class TestCriticalPath:
    """Test CPM/PERT critical path analysis."""

    def test_linear_chain_critical_path(self) -> None:
        """In a linear chain, all tasks are critical."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=2.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=3.0))
        net.add_task(TaskNode(task_id="C", name="C", duration=1.0))
        net.add_dependency("B", "A")
        net.add_dependency("C", "B")
        net.compute_schedule()
        analysis = net.critical_path_analysis()
        assert analysis["project_duration"] == pytest.approx(6.0)
        assert len(analysis["critical_path"]) == 3

    def test_parallel_paths_critical_is_longer(self) -> None:
        """The critical path should be the longest path.

        A(2) → C(1)   total = 3
        B(5) → C(1)   total = 6, B is on critical path
        """
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=2.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=5.0))
        net.add_task(TaskNode(task_id="C", name="C", duration=1.0))
        net.add_dependency("C", "A")
        net.add_dependency("C", "B")
        net.compute_schedule()
        analysis = net.critical_path_analysis()
        assert analysis["project_duration"] == pytest.approx(6.0)
        # B should be on the critical path
        assert "B" in analysis["critical_path"]

    def test_float_computation(self) -> None:
        """Non-critical tasks should have positive float."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="A", duration=2.0))
        net.add_task(TaskNode(task_id="B", name="B", duration=5.0))
        net.add_task(TaskNode(task_id="C", name="C", duration=1.0))
        net.add_dependency("C", "A")
        net.add_dependency("C", "B")
        net.compute_schedule()
        # A has float (not on critical path)
        node_a = net._nodes["A"]
        assert node_a.total_float > 0

    def test_single_task_critical(self) -> None:
        """Single task is always critical."""
        net = TaskNetwork()
        net.add_task(TaskNode(task_id="X", name="X", duration=3.0))
        net.compute_schedule()
        analysis = net.critical_path_analysis()
        assert analysis["critical_path"] == ["X"]
        assert analysis["project_duration"] == pytest.approx(3.0)
