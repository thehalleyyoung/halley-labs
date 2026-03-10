"""Unit tests for usability_oracle.goms.simulation.

Tests cover discrete-event simulation, stochastic operator times, error
injection, trace generation, and regression comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    OperatorType,
)
from usability_oracle.goms.klm import KLMConfig, SkillLevel
from usability_oracle.goms.simulation import (
    GomsSimulator,
    SimEvent,
    SimTrace,
    SimulationConfig,
    WorkingMemoryTracker,
    compute_error_probability,
    sample_duration,
    schedule_parallel_execution,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _op(op_type: OperatorType, dur: float, **kwargs) -> GomsOperator:
    return GomsOperator(op_type=op_type, duration_s=dur, **kwargs)


def _method(mid: str, gid: str, ops: list[GomsOperator]) -> GomsMethod:
    return GomsMethod(method_id=mid, goal_id=gid, name=mid, operators=tuple(ops))


def _model(name: str, methods: list[GomsMethod]) -> GomsModel:
    goals = [GomsGoal(goal_id="g", description="goal")]
    return GomsModel(model_id="m", name=name, goals=tuple(goals), methods=tuple(methods))


# ------------------------------------------------------------------ #
# Discrete-event simulation
# ------------------------------------------------------------------ #


class TestDiscreteEventSimulation:
    """Tests for the main GomsSimulator."""

    def test_deterministic_simulation(self) -> None:
        """With stochastic=False, total time = sum of durations."""
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28), _op(OperatorType.K, 0.28)])
        model = _model("task", [m])
        traces = sim.simulate(model, [m], n_runs=1)
        assert len(traces) == 1
        assert traces[0].total_time_s == pytest.approx(0.56, rel=1e-2)

    def test_multiple_runs(self) -> None:
        """Multiple runs produce multiple traces."""
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)])
        model = _model("task", [m])
        traces = sim.simulate(model, [m], n_runs=5)
        assert len(traces) == 5

    def test_event_count_matches_operators(self) -> None:
        """Number of events = number of operators."""
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        ops = [_op(OperatorType.K, 0.28), _op(OperatorType.M, 1.35), _op(OperatorType.P, 1.10)]
        m = _method("m", "g", ops)
        model = _model("task", [m])
        traces = sim.simulate(model, [m])
        assert traces[0].event_count == 3

    def test_events_ordered_by_time(self) -> None:
        """Events should be in non-decreasing timestamp order."""
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [
            _op(OperatorType.M, 1.0),
            _op(OperatorType.K, 0.5),
            _op(OperatorType.P, 1.0),
        ])
        model = _model("task", [m])
        traces = sim.simulate(model, [m])
        events = traces[0].events
        for i in range(len(events) - 1):
            assert events[i].timestamp_s <= events[i + 1].timestamp_s


# ------------------------------------------------------------------ #
# Stochastic operator times
# ------------------------------------------------------------------ #


class TestStochasticOperatorTimes:
    """Tests for stochastic (log-normal) operator durations."""

    def test_stochastic_varies(self) -> None:
        """With stochastic=True, repeated runs should produce different times."""
        config = SimulationConfig(stochastic=True, inject_errors=False, seed=42)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)] * 10)
        model = _model("task", [m])
        traces = sim.simulate(model, [m], n_runs=10)
        times = [t.total_time_s for t in traces]
        # Should have some variation
        assert max(times) > min(times)

    def test_sample_duration_positive(self) -> None:
        """Sampled durations should be positive."""
        rng = np.random.default_rng(42)
        op = _op(OperatorType.K, 0.28)
        for _ in range(100):
            d = sample_duration(op, rng)
            assert d > 0

    def test_sample_duration_mean_close(self) -> None:
        """Mean of many samples should be close to nominal duration."""
        rng = np.random.default_rng(42)
        op = _op(OperatorType.K, 0.28)
        samples = [sample_duration(op, rng) for _ in range(1000)]
        assert np.mean(samples) == pytest.approx(0.28, rel=0.1)

    def test_zero_duration_operator(self) -> None:
        """Zero-duration operator should sample as 0."""
        rng = np.random.default_rng(42)
        op = _op(OperatorType.R, 0.0)
        assert sample_duration(op, rng) == 0.0


# ------------------------------------------------------------------ #
# Error injection
# ------------------------------------------------------------------ #


class TestErrorInjection:
    """Tests for error probability computation and injection."""

    def test_error_probability_positive(self) -> None:
        """Non-zero error probability for motor operators."""
        op = _op(OperatorType.K, 0.28)
        p = compute_error_probability(op, 0.0, SkillLevel.INTERMEDIATE)
        assert p > 0

    def test_novice_higher_error(self) -> None:
        """Novice has higher error rate than expert."""
        op = _op(OperatorType.K, 0.28)
        p_nov = compute_error_probability(op, 0.0, SkillLevel.NOVICE)
        p_exp = compute_error_probability(op, 0.0, SkillLevel.EXPERT)
        assert p_nov > p_exp

    def test_high_wm_increases_cognitive_error(self) -> None:
        """High WM load increases error for cognitive operators."""
        op = _op(OperatorType.M, 1.35)
        p_low = compute_error_probability(op, 1.0, SkillLevel.INTERMEDIATE)
        p_high = compute_error_probability(op, 6.0, SkillLevel.INTERMEDIATE)
        assert p_high >= p_low

    def test_system_response_zero_error(self) -> None:
        """System response operator has 0 error probability."""
        op = _op(OperatorType.R, 1.0)
        p = compute_error_probability(op, 0.0, SkillLevel.INTERMEDIATE)
        assert p == pytest.approx(0.0)

    def test_error_capped_at_one(self) -> None:
        """Error probability should never exceed 1.0."""
        op = _op(OperatorType.K, 0.28)
        p = compute_error_probability(op, 100.0, SkillLevel.NOVICE)
        assert p <= 1.0

    def test_errors_injected_in_simulation(self) -> None:
        """With inject_errors=True, some runs should have errors."""
        config = SimulationConfig(stochastic=True, inject_errors=True, seed=42)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)] * 30)
        model = _model("task", [m])
        traces = sim.simulate(model, [m], n_runs=20)
        total_errors = sum(t.error_count for t in traces)
        assert total_errors > 0  # at least some errors across 20 runs


# ------------------------------------------------------------------ #
# Trace generation
# ------------------------------------------------------------------ #


class TestTraceGeneration:
    """Tests for SimTrace properties and conversion."""

    def test_trace_has_id(self) -> None:
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)])
        model = _model("task", [m])
        traces = sim.simulate(model, [m])
        assert traces[0].trace_id.startswith("sim-")

    def test_trace_task_name(self) -> None:
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)])
        model = _model("my_task", [m])
        traces = sim.simulate(model, [m])
        assert traces[0].task_name == "my_task"

    def test_to_goms_trace(self) -> None:
        """SimTrace.to_goms_trace produces a valid GomsTrace."""
        sim_trace = SimTrace(trace_id="t", task_name="T", total_time_s=5.0)
        goals = [GomsGoal(goal_id="g", description="g")]
        methods = [_method("m", "g", [_op(OperatorType.K, 0.28)])]
        goms_trace = sim_trace.to_goms_trace(goals, methods)
        assert isinstance(goms_trace, GomsTrace)
        assert goms_trace.total_time_s == 5.0

    def test_peak_wm_tracked(self) -> None:
        """Peak working memory should be tracked."""
        config = SimulationConfig(stochastic=False, inject_errors=False)
        sim = GomsSimulator(config)
        m = _method("m", "g", [
            _op(OperatorType.M, 1.35),
            _op(OperatorType.M, 1.35),
            _op(OperatorType.K, 0.28),
        ])
        model = _model("task", [m])
        traces = sim.simulate(model, [m])
        assert traces[0].peak_wm_load >= 0


# ------------------------------------------------------------------ #
# Working memory tracker
# ------------------------------------------------------------------ #


class TestWorkingMemoryTracker:
    """Tests for the WM load tracking."""

    def test_initially_zero(self) -> None:
        wm = WorkingMemoryTracker()
        assert wm.current_load == 0.0

    def test_add_chunk_increases_load(self) -> None:
        wm = WorkingMemoryTracker()
        wm.add_chunk(0.0)
        assert wm.current_load > 0

    def test_decay_reduces_load(self) -> None:
        """Load decays over time."""
        wm = WorkingMemoryTracker(decay_half_life_s=1.0)
        wm.add_chunk(0.0)
        load_t0 = wm.current_load
        wm.advance_time(10.0)
        load_t10 = wm.current_load
        assert load_t10 < load_t0

    def test_reset_clears(self) -> None:
        wm = WorkingMemoryTracker()
        wm.add_chunk(0.0)
        wm.reset()
        assert wm.current_load == 0.0


# ------------------------------------------------------------------ #
# Parallel execution scheduling
# ------------------------------------------------------------------ #


class TestParallelScheduling:
    """Tests for schedule_parallel_execution."""

    def test_serial_for_intermediate(self) -> None:
        """Intermediate skill → purely serial schedule."""
        m = _method("m", "g", [
            _op(OperatorType.M, 1.0),
            _op(OperatorType.K, 0.5),
        ])
        schedule, total = schedule_parallel_execution([m], skill_level=SkillLevel.INTERMEDIATE)
        assert total == pytest.approx(1.5, rel=1e-9)

    def test_expert_may_overlap(self) -> None:
        """Expert skill can schedule cognitive and motor in parallel."""
        m = _method("m", "g", [
            _op(OperatorType.M, 1.0),
            _op(OperatorType.K, 0.5),
            _op(OperatorType.M, 1.0),
            _op(OperatorType.K, 0.5),
        ])
        _, total_serial = schedule_parallel_execution([m], skill_level=SkillLevel.INTERMEDIATE)
        _, total_expert = schedule_parallel_execution([m], skill_level=SkillLevel.EXPERT)
        assert total_expert <= total_serial


# ------------------------------------------------------------------ #
# Regression comparison
# ------------------------------------------------------------------ #


class TestRegressionComparison:
    """Tests for GomsSimulator.compare_versions."""

    def test_compare_identical(self) -> None:
        """Comparing identical models shows no regression."""
        config = SimulationConfig(stochastic=True, inject_errors=False, seed=42)
        sim = GomsSimulator(config)
        m = _method("m", "g", [_op(OperatorType.K, 0.28)] * 5)
        model = _model("task", [m])
        result = sim.compare_versions(model, [m], model, [m], n_runs=20)
        assert abs(result["time_delta_s"]) < 0.5
        assert result["regression_detected"] is False

    def test_compare_slower_detects_regression(self) -> None:
        """Slower new model should be detected as regression."""
        config = SimulationConfig(stochastic=False, inject_errors=False, seed=42)
        sim = GomsSimulator(config)
        m_old = _method("m", "g", [_op(OperatorType.K, 0.28)])
        m_new = _method("m", "g", [_op(OperatorType.K, 0.28)] * 20)
        model_old = _model("old", [m_old])
        model_new = _model("new", [m_new])
        result = sim.compare_versions(model_old, [m_old], model_new, [m_new], n_runs=10)
        assert result["time_delta_s"] > 0
        assert result["mean_time_new_s"] > result["mean_time_old_s"]
