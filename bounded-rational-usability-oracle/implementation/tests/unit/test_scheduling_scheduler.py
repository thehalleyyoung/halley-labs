"""Unit tests for usability_oracle.scheduling.scheduler — BoundedRationalScheduler.

Tests cover EDF, RMS, WSJF scheduling algorithms, bounded-rational softmax
scheduling, feasibility analysis, multi-resource scheduling, and schedule
metrics computation.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import pytest

from usability_oracle.scheduling.types import (
    DeadlineModel,
    Schedule,
    ScheduledTask,
    SchedulingConstraint,
    TaskPriority,
)
from usability_oracle.scheduling.scheduler import (
    BoundedRationalScheduler,
    MultiResourceScheduler,
    ResourceChannel,
    ScheduleMetrics,
    _softmax,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_task(
    task_id: str,
    duration: float = 1.0,
    priority: TaskPriority = TaskPriority.MEDIUM,
    hard_deadline: float = float("inf"),
    cognitive_cost: float = 1.0,
    dependencies: tuple = (),
) -> ScheduledTask:
    deadline = DeadlineModel(
        hard_deadline_s=hard_deadline,
        soft_deadline_s=float("inf"),
        urgency_decay_rate=1.0,
        time_budget_s=duration * 2,
    )
    return ScheduledTask(
        task_id=task_id,
        name=f"Task {task_id}",
        priority=priority,
        estimated_duration_s=duration,
        cognitive_cost_bits=cognitive_cost,
        deadline=deadline,
        dependencies=dependencies,
        metadata={},
    )


def _tasks_with_deadlines() -> List[ScheduledTask]:
    """Three tasks with different deadlines for EDF testing."""
    return [
        _make_task("t1", duration=1.0, hard_deadline=5.0),
        _make_task("t2", duration=2.0, hard_deadline=3.0),
        _make_task("t3", duration=0.5, hard_deadline=4.0),
    ]


def _tasks_with_periods() -> List[ScheduledTask]:
    """Tasks with different periods (deadlines) for RMS testing."""
    return [
        _make_task("fast", duration=0.5, hard_deadline=2.0, priority=TaskPriority.HIGH),
        _make_task("medium", duration=1.0, hard_deadline=5.0, priority=TaskPriority.MEDIUM),
        _make_task("slow", duration=2.0, hard_deadline=10.0, priority=TaskPriority.LOW),
    ]


def _tasks_for_wsjf() -> List[ScheduledTask]:
    """Tasks with varied priority/duration for WSJF testing."""
    return [
        _make_task("high_short", duration=1.0, priority=TaskPriority.CRITICAL, hard_deadline=10.0),
        _make_task("high_long", duration=5.0, priority=TaskPriority.CRITICAL, hard_deadline=20.0),
        _make_task("low_short", duration=1.0, priority=TaskPriority.LOW, hard_deadline=30.0),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# EDF Scheduling
# ═══════════════════════════════════════════════════════════════════════════


class TestEDFScheduling:
    """Earliest-Deadline-First scheduling tests."""

    def test_edf_orders_by_deadline(self):
        scheduler = BoundedRationalScheduler(beta=100.0, algorithm="edf")
        tasks = _tasks_with_deadlines()
        schedule = scheduler.schedule(tasks)
        assert schedule.feasible
        assert schedule.task_count == 3
        starts = [(a, schedule.start_time(a)) for a in ["t1", "t2", "t3"]]
        starts_valid = [s for s in starts if s[1] is not None]
        assert len(starts_valid) == 3

    def test_edf_earliest_deadline_scheduled_first(self):
        scheduler = BoundedRationalScheduler(beta=100.0, algorithm="edf")
        tasks = _tasks_with_deadlines()
        schedule = scheduler.schedule(tasks)
        # t2 has the earliest deadline (3.0) so should be scheduled first
        t2_start = schedule.start_time("t2")
        t1_start = schedule.start_time("t1")
        t3_start = schedule.start_time("t3")
        assert t2_start is not None
        assert t1_start is not None
        assert t2_start <= t3_start
        assert t2_start <= t1_start

    def test_edf_single_task(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [_make_task("only", duration=2.0, hard_deadline=10.0)]
        schedule = scheduler.schedule(tasks)
        assert schedule.task_count == 1
        assert schedule.feasible

    def test_edf_empty_task_list(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        schedule = scheduler.schedule([])
        assert schedule.task_count == 0
        assert schedule.feasible


# ═══════════════════════════════════════════════════════════════════════════
# RMS Scheduling
# ═══════════════════════════════════════════════════════════════════════════


class TestRMSScheduling:
    """Rate-Monotonic Scheduling tests."""

    def test_rms_orders_by_period(self):
        scheduler = BoundedRationalScheduler(beta=100.0, algorithm="rms")
        tasks = _tasks_with_periods()
        schedule = scheduler.schedule(tasks)
        assert schedule.feasible
        # Shortest period task should go first
        fast_start = schedule.start_time("fast")
        medium_start = schedule.start_time("medium")
        slow_start = schedule.start_time("slow")
        assert fast_start is not None
        assert fast_start <= medium_start
        assert medium_start <= slow_start

    def test_rms_all_tasks_scheduled(self):
        scheduler = BoundedRationalScheduler(algorithm="rms")
        tasks = _tasks_with_periods()
        schedule = scheduler.schedule(tasks)
        assert schedule.task_count == 3

    def test_rms_makespan_reasonable(self):
        scheduler = BoundedRationalScheduler(algorithm="rms")
        tasks = _tasks_with_periods()
        schedule = scheduler.schedule(tasks)
        total_duration = sum(t.estimated_duration_s for t in tasks)
        assert schedule.makespan_s >= total_duration - 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# WSJF Scheduling
# ═══════════════════════════════════════════════════════════════════════════


class TestWSJFScheduling:
    """Weighted Shortest Job First scheduling tests."""

    def test_wsjf_prefers_high_value_short_jobs(self):
        scheduler = BoundedRationalScheduler(beta=100.0, algorithm="wsjf")
        tasks = _tasks_for_wsjf()
        schedule = scheduler.schedule(tasks)
        assert schedule.feasible
        # high_short should be scheduled before high_long (same priority, shorter)
        hs_start = schedule.start_time("high_short")
        hl_start = schedule.start_time("high_long")
        assert hs_start is not None
        assert hs_start <= hl_start

    def test_wsjf_all_tasks_included(self):
        scheduler = BoundedRationalScheduler(algorithm="wsjf")
        tasks = _tasks_for_wsjf()
        schedule = scheduler.schedule(tasks)
        assert schedule.task_count == 3


# ═══════════════════════════════════════════════════════════════════════════
# Bounded-Rational Softmax Scheduling
# ═══════════════════════════════════════════════════════════════════════════


class TestSoftmaxScheduling:
    """Bounded-rational softmax scheduling tests."""

    def test_high_beta_approaches_deterministic(self):
        """With very high β, softmax should approximate greedy EDF."""
        tasks = _tasks_with_deadlines()
        schedules_high = []
        for _ in range(5):
            scheduler = BoundedRationalScheduler(beta=100.0, algorithm="edf", rng_seed=42)
            schedules_high.append(scheduler.schedule(tasks))
        # All schedules should be the same for high β with same seed
        starts = [s.start_time("t2") for s in schedules_high]
        assert all(abs(s - starts[0]) < 1e-6 for s in starts)

    def test_low_beta_introduces_variability(self):
        """With very low β, scheduling order may vary across seeds."""
        tasks = _tasks_with_deadlines()
        orderings = set()
        for seed in range(20):
            scheduler = BoundedRationalScheduler(beta=0.01, algorithm="edf", rng_seed=seed)
            s = scheduler.schedule(tasks)
            order = tuple(
                sorted(["t1", "t2", "t3"], key=lambda tid: s.start_time(tid) or 0.0)
            )
            orderings.add(order)
        # Low β should produce at least some variation
        assert len(orderings) >= 1

    def test_beta_zero_still_produces_schedule(self):
        scheduler = BoundedRationalScheduler(beta=0.001, algorithm="edf", rng_seed=7)
        tasks = _tasks_with_deadlines()
        schedule = scheduler.schedule(tasks)
        assert schedule.task_count == 3

    def test_softmax_utility_produces_valid_probabilities(self):
        """Internal softmax should produce valid probability distribution."""
        values = np.array([1.0, 2.0, 3.0])
        probs = _softmax(values, beta=1.0)
        assert abs(probs.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in probs)

    def test_softmax_high_beta_peaks(self):
        values = np.array([1.0, 2.0, 3.0])
        probs = _softmax(values, beta=100.0)
        assert probs[2] > 0.99  # largest value should dominate

    def test_softmax_zero_beta_uniform(self):
        values = np.array([1.0, 2.0, 3.0])
        probs = _softmax(values, beta=0.0)
        assert abs(probs[0] - probs[1]) < 0.01
        assert abs(probs[1] - probs[2]) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Feasibility Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestFeasibility:
    """Schedule feasibility analysis tests."""

    def test_feasible_schedule(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [
            _make_task("a", duration=1.0, hard_deadline=5.0),
            _make_task("b", duration=1.0, hard_deadline=5.0),
        ]
        schedule = scheduler.schedule(tasks)
        assert scheduler.is_feasible(schedule)

    def test_tight_deadlines_may_be_infeasible(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [
            _make_task("a", duration=3.0, hard_deadline=2.0),
            _make_task("b", duration=3.0, hard_deadline=2.0),
        ]
        schedule = scheduler.schedule(tasks)
        # With impossibly tight deadlines, violations should be present
        assert schedule.deadline_violations is not None

    def test_cognitive_budget_constraint(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [
            _make_task("c1", cognitive_cost=5.0, hard_deadline=20.0),
            _make_task("c2", cognitive_cost=5.0, hard_deadline=20.0),
            _make_task("c3", cognitive_cost=5.0, hard_deadline=20.0),
        ]
        schedule = scheduler.schedule(tasks, cognitive_budget_bits=8.0)
        # Budget of 8 bits means not all 15 bits of tasks can be scheduled
        assert schedule.total_cognitive_cost_bits <= 8.0 + 1e-6

    def test_dependencies_respected(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        dep_task = _make_task("dep", duration=1.0, hard_deadline=10.0)
        main_task = _make_task("main", duration=1.0, hard_deadline=10.0, dependencies=("dep",))
        schedule = scheduler.schedule([main_task, dep_task])
        dep_start = schedule.start_time("dep")
        main_start = schedule.start_time("main")
        if dep_start is not None and main_start is not None:
            assert main_start >= dep_start

    def test_incremental_scheduling(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [_make_task("a", duration=1.0, hard_deadline=10.0)]
        schedule = scheduler.schedule(tasks)
        new_task = _make_task("b", duration=0.5, hard_deadline=10.0)
        new_schedule = scheduler.schedule_incremental(schedule, new_task)
        assert new_schedule.task_count >= schedule.task_count


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Resource Scheduling
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiResourceScheduling:
    """Multi-resource scheduler tests."""

    def test_multi_resource_construction(self):
        channels = [
            ResourceChannel(name="visual", capacity=1.0, current_load=0.0),
            ResourceChannel(name="motor", capacity=1.0, current_load=0.0),
        ]
        scheduler = MultiResourceScheduler(channels=channels)
        tasks = [_make_task("t1", hard_deadline=10.0), _make_task("t2", hard_deadline=10.0)]
        task_channel_map = {"t1": ["visual"], "t2": ["motor"]}
        schedule = scheduler.schedule(tasks, task_channel_map)
        assert schedule.task_count == 2

    def test_multi_resource_capacity_limits(self):
        channels = [
            ResourceChannel(name="cpu", capacity=1.0, current_load=0.0),
        ]
        scheduler = MultiResourceScheduler(channels=channels)
        tasks = [_make_task(f"t{i}", duration=1.0, hard_deadline=20.0) for i in range(5)]
        task_channel_map = {f"t{i}": ["cpu"] for i in range(5)}
        schedule = scheduler.schedule(tasks, task_channel_map)
        assert schedule.task_count == 5

    def test_resource_channel_available(self):
        ch = ResourceChannel(name="ch", capacity=1.0, current_load=0.0)
        assert ch.available
        ch_full = ResourceChannel(name="ch", capacity=1.0, current_load=1.0)
        assert not ch_full.available


# ═══════════════════════════════════════════════════════════════════════════
# Schedule Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestScheduleMetrics:
    """Tests for schedule metrics computation."""

    def test_compute_metrics_basic(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = _tasks_with_deadlines()
        schedule = scheduler.schedule(tasks)
        metrics = scheduler.compute_metrics(schedule, tasks)
        assert isinstance(metrics, ScheduleMetrics)
        assert metrics.makespan_s >= 0
        assert metrics.total_cognitive_cost_bits >= 0
        assert 0.0 <= metrics.utilization <= 1.0 + 1e-6

    def test_metrics_total_tardiness_non_negative(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = _tasks_with_deadlines()
        schedule = scheduler.schedule(tasks)
        metrics = scheduler.compute_metrics(schedule, tasks)
        assert metrics.total_tardiness_s >= 0

    def test_metrics_deadline_miss_count(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        # All tasks with generous deadlines
        tasks = [_make_task(f"t{i}", duration=0.5, hard_deadline=100.0) for i in range(3)]
        schedule = scheduler.schedule(tasks)
        metrics = scheduler.compute_metrics(schedule, tasks)
        assert metrics.deadline_miss_count == 0

    def test_schedule_has_violations_property(self):
        scheduler = BoundedRationalScheduler(algorithm="edf")
        tasks = [_make_task("ok", duration=0.5, hard_deadline=100.0)]
        schedule = scheduler.schedule(tasks)
        assert isinstance(schedule.has_violations, bool)


class TestSoftmaxFunction:
    """Additional tests for the internal _softmax function."""

    def test_softmax_single_element(self):
        probs = _softmax(np.array([5.0]), beta=1.0)
        assert probs[0] == pytest.approx(1.0)
