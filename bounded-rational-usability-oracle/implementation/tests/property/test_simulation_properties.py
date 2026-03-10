"""Property-based tests for the simulation module.

Verifies properties of the event queue, KLM model, ACT-R activation dynamics,
and task network topological sort.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    lists,
    permutations,
    composite,
    sampled_from,
)

from usability_oracle.simulation import (
    EventQueue,
    SimulationEvent,
    EventType,
    EventPriority,
    KLMModel,
    KLMOperator,
    KLMStep,
    KLMTimings,
    TaskNetwork,
    TaskNode,
    TaskStatus,
    base_level_learning,
    retrieval_time,
    retrieval_probability,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_float = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)

_timestamp = floats(min_value=0.0, max_value=1e6,
                    allow_nan=False, allow_infinity=False)

_klm_operators = sampled_from(list(KLMOperator))

_event_types = sampled_from(list(EventType))

_priorities = sampled_from(list(EventPriority))


@composite
def _event_list(draw, min_size=2, max_size=20):
    """Generate a list of SimulationEvents with distinct timestamps."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    events = []
    for i in range(n):
        ts = draw(_timestamp)
        et = draw(_event_types)
        pr = draw(_priorities)
        events.append(SimulationEvent(
            event_id=i,
            timestamp=ts,
            event_type=et,
            priority=pr,
            source_processor='test',
        ))
    return events


@composite
def _klm_sequence(draw, min_size=1, max_size=10):
    """Generate a sequence of KLM steps with durations."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    steps = []
    for i in range(n):
        op = draw(_klm_operators)
        dur = draw(floats(min_value=0.01, max_value=5.0,
                          allow_nan=False, allow_infinity=False))
        steps.append(KLMStep(
            operator=op,
            description=f"step_{i}",
            duration=dur,
            element_id=f"e{i}",
        ))
    return steps


_ATOL = 1e-6

# ---------------------------------------------------------------------------
# Event queue always pops minimum timestamp
# ---------------------------------------------------------------------------


@given(_event_list(min_size=2, max_size=15))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_event_queue_pops_minimum_timestamp(events):
    """EventQueue.pop() always returns the event with smallest timestamp."""
    eq = EventQueue()
    for e in events:
        eq.insert(e)
    prev_ts = -1.0
    while not eq.is_empty:
        event = eq.pop()
        assert event.timestamp >= prev_ts - _ATOL, \
            f"Non-monotone pop: {event.timestamp} < {prev_ts}"
        prev_ts = event.timestamp


def test_event_queue_basic_ordering():
    """Basic test: events with timestamps 3, 1, 2 pop in order 1, 2, 3."""
    eq = EventQueue()
    for i, ts in enumerate([3.0, 1.0, 2.0]):
        eq.insert(SimulationEvent(
            event_id=i, timestamp=ts,
            event_type=EventType.TASK_START,
            priority=EventPriority.TASK,
            source_processor='test',
        ))
    assert eq.pop().timestamp == 1.0
    assert eq.pop().timestamp == 2.0
    assert eq.pop().timestamp == 3.0


def test_event_queue_size_tracking():
    """Queue size tracks inserts and pops correctly."""
    eq = EventQueue()
    assert eq.size == 0
    eq.insert(SimulationEvent(event_id=0, timestamp=1.0))
    eq.insert(SimulationEvent(event_id=1, timestamp=2.0))
    assert eq.size == 2
    eq.pop()
    assert eq.size == 1


# ---------------------------------------------------------------------------
# KLM task time ≥ 0 for any operator sequence
# ---------------------------------------------------------------------------


@given(_klm_sequence())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_klm_task_time_non_negative(steps):
    """KLM task time ≥ 0 for any operator sequence."""
    model = KLMModel(timings=KLMTimings(), auto_insert_mental=False)
    t = model.predict_task_time(steps)
    assert t >= -_ATOL, f"KLM task time is negative: {t}"


# ---------------------------------------------------------------------------
# KLM time is additive (sum of individual operator times)
# ---------------------------------------------------------------------------


@given(_klm_sequence(min_size=2, max_size=8))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_klm_time_is_additive(steps):
    """KLM task time equals sum of individual step durations (no mental prep)."""
    model = KLMModel(timings=KLMTimings(), auto_insert_mental=False)
    total = model.predict_task_time(steps)
    expected = sum(s.duration for s in steps)
    assert math.isclose(total, expected, abs_tol=1e-4), \
        f"KLM time {total} != sum of durations {expected}"


def test_klm_time_specific_sequence():
    """KLM time for K + P = 0.28 + 1.1 = 1.38."""
    model = KLMModel(timings=KLMTimings(), auto_insert_mental=False)
    steps = [
        KLMStep(operator=KLMOperator.K, duration=0.28, element_id='e1'),
        KLMStep(operator=KLMOperator.P, duration=1.1, element_id='e2'),
    ]
    t = model.predict_task_time(steps)
    assert math.isclose(t, 1.38, abs_tol=0.01)


def test_klm_empty_sequence_zero_time():
    """Empty operator sequence should yield 0 time."""
    model = KLMModel(timings=KLMTimings(), auto_insert_mental=False)
    t = model.predict_task_time([])
    assert math.isclose(t, 0.0, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# ACT-R activation decreases with time since presentation
# ---------------------------------------------------------------------------


def test_activation_decreases_with_time():
    """Base-level activation decreases as time since presentation increases."""
    recent = base_level_learning([9.0], current_time=10.0, decay=0.5)
    old = base_level_learning([1.0], current_time=10.0, decay=0.5)
    assert recent > old, \
        f"Recent presentation should have higher activation: {recent} vs {old}"


def test_activation_increases_with_practice():
    """More presentations increase base-level activation."""
    one = base_level_learning([5.0], current_time=10.0, decay=0.5)
    three = base_level_learning([3.0, 5.0, 8.0], current_time=10.0, decay=0.5)
    assert three > one, \
        f"More presentations should increase activation: {one} vs {three}"


@given(floats(min_value=0.1, max_value=5.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_retrieval_time_positive(activation):
    """Retrieval time is always positive."""
    rt = retrieval_time(activation)
    assert rt > 0, f"Retrieval time should be positive, got {rt}"


@given(floats(min_value=-2.0, max_value=5.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_retrieval_probability_in_unit_interval(activation):
    """Retrieval probability is in [0, 1]."""
    rp = retrieval_probability(activation)
    assert 0.0 <= rp <= 1.0 + _ATOL, \
        f"Retrieval probability out of range: {rp}"


def test_retrieval_time_decreases_with_activation():
    """Higher activation → faster retrieval (lower time)."""
    rt_low = retrieval_time(-0.5)
    rt_high = retrieval_time(1.0)
    assert rt_high < rt_low, \
        f"Higher activation should yield faster retrieval: {rt_low} vs {rt_high}"


# ---------------------------------------------------------------------------
# Task network topological sort respects all dependencies
# ---------------------------------------------------------------------------


def test_topological_sort_respects_dependencies():
    """Topological sort must output predecessors before successors."""
    tn = TaskNetwork()
    tn.add_task(TaskNode(task_id='a', name='A', duration=1.0))
    tn.add_task(TaskNode(task_id='b', name='B', duration=1.0))
    tn.add_task(TaskNode(task_id='c', name='C', duration=1.0))
    tn.add_task(TaskNode(task_id='d', name='D', duration=1.0))
    tn.add_dependency('b', 'a')  # a → b
    tn.add_dependency('c', 'a')  # a → c
    tn.add_dependency('d', 'b')  # b → d
    tn.add_dependency('d', 'c')  # c → d

    order = tn.topological_sort()
    pos = {tid: i for i, tid in enumerate(order)}
    assert pos['a'] < pos['b']
    assert pos['a'] < pos['c']
    assert pos['b'] < pos['d']
    assert pos['c'] < pos['d']


def test_topological_sort_includes_all_tasks():
    """Topological sort must include every task."""
    tn = TaskNetwork()
    for i in range(5):
        tn.add_task(TaskNode(task_id=f't{i}', name=f'T{i}', duration=1.0))
    for i in range(1, 5):
        tn.add_dependency(f't{i}', f't{i-1}')
    order = tn.topological_sort()
    assert len(order) == 5
    assert set(order) == {f't{i}' for i in range(5)}


def test_topological_sort_linear_chain():
    """Linear chain t0 → t1 → t2 → t3 must sort in order."""
    tn = TaskNetwork()
    for i in range(4):
        tn.add_task(TaskNode(task_id=f't{i}', name=f'T{i}', duration=1.0))
    for i in range(1, 4):
        tn.add_dependency(f't{i}', f't{i-1}')
    order = tn.topological_sort()
    assert order == ['t0', 't1', 't2', 't3']


def test_topological_sort_independent_tasks():
    """Independent tasks should all appear (order is implementation-defined)."""
    tn = TaskNetwork()
    for i in range(5):
        tn.add_task(TaskNode(task_id=f't{i}', name=f'T{i}', duration=1.0))
    order = tn.topological_sort()
    assert len(order) == 5
    assert set(order) == {f't{i}' for i in range(5)}


def test_task_network_diamond_dag():
    """Diamond DAG: a → {b, c} → d."""
    tn = TaskNetwork()
    tn.add_task(TaskNode(task_id='a', name='A', duration=1.0))
    tn.add_task(TaskNode(task_id='b', name='B', duration=2.0))
    tn.add_task(TaskNode(task_id='c', name='C', duration=1.5))
    tn.add_task(TaskNode(task_id='d', name='D', duration=0.5))
    tn.add_dependency('b', 'a')
    tn.add_dependency('c', 'a')
    tn.add_dependency('d', 'b')
    tn.add_dependency('d', 'c')
    order = tn.topological_sort()
    pos = {tid: i for i, tid in enumerate(order)}
    assert pos['a'] == 0
    assert pos['d'] == 3
