"""Unit tests for usability_oracle.goms.analyzer.

Tests cover GOMS model construction, execution time prediction, critical
path analysis, method selection, and operator merging.
"""

from __future__ import annotations

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
from usability_oracle.goms.klm import KLMConfig, SkillLevel, make_keystroke, make_mental, make_pointing
from usability_oracle.goms.analyzer import (
    AnalyzerConfig,
    GomsAnalyzerImpl,
    compute_critical_path_time,
    estimate_error_probability,
    estimate_learning_time,
    merge_operators_expert,
    predict_execution_time,
)
from usability_oracle.core.types import Point2D


# ------------------------------------------------------------------ #
# Helpers — build small GOMS models
# ------------------------------------------------------------------ #


def _make_op(op_type: OperatorType, dur: float, **kwargs) -> GomsOperator:
    return GomsOperator(op_type=op_type, duration_s=dur, **kwargs)


def _make_method(
    method_id: str, goal_id: str, ops: list[GomsOperator], name: str = "",
) -> GomsMethod:
    return GomsMethod(
        method_id=method_id,
        goal_id=goal_id,
        name=name or method_id,
        operators=tuple(ops),
    )


def _simple_model() -> GomsModel:
    """A simple GOMS model with two leaf goals."""
    g_top = GomsGoal(goal_id="top", description="Top-level", subgoal_ids=("g1", "g2"))
    g1 = GomsGoal(goal_id="g1", description="Sub-goal 1", parent_id="top")
    g2 = GomsGoal(goal_id="g2", description="Sub-goal 2", parent_id="top")

    m1 = _make_method("m1", "g1", [
        _make_op(OperatorType.M, 1.35),
        _make_op(OperatorType.K, 0.28),
        _make_op(OperatorType.K, 0.28),
    ])
    m2 = _make_method("m2", "g2", [
        _make_op(OperatorType.M, 1.35),
        _make_op(OperatorType.P, 1.10),
        _make_op(OperatorType.B, 0.10),
    ])

    return GomsModel(
        model_id="test-model",
        name="Test Task",
        goals=(g_top, g1, g2),
        methods=(m1, m2),
        top_level_goal_id="top",
    )


# ------------------------------------------------------------------ #
# GOMS model construction
# ------------------------------------------------------------------ #


class TestGomsModelConstruction:
    """Tests for GomsModel properties and structure."""

    def test_goal_count(self) -> None:
        model = _simple_model()
        assert model.goal_count == 3

    def test_method_count(self) -> None:
        model = _simple_model()
        assert model.method_count == 2

    def test_methods_for_goal(self) -> None:
        model = _simple_model()
        methods = model.methods_for_goal("g1")
        assert len(methods) == 1
        assert methods[0].method_id == "m1"

    def test_methods_for_nonexistent_goal(self) -> None:
        model = _simple_model()
        assert model.methods_for_goal("nonexistent") == ()

    def test_leaf_goal(self) -> None:
        g = GomsGoal(goal_id="leaf", description="leaf")
        assert g.is_leaf

    def test_non_leaf_goal(self) -> None:
        g = GomsGoal(goal_id="parent", description="parent", subgoal_ids=("a",))
        assert not g.is_leaf

    def test_method_total_duration(self) -> None:
        m = _make_method("m", "g", [
            _make_op(OperatorType.K, 0.28),
            _make_op(OperatorType.K, 0.28),
        ])
        assert m.total_duration_s == pytest.approx(0.56, rel=1e-9)

    def test_method_motor_time(self) -> None:
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.35),
            _make_op(OperatorType.K, 0.28),
        ])
        assert m.motor_time_s == pytest.approx(0.28)
        assert m.cognitive_time_s == pytest.approx(1.35)


# ------------------------------------------------------------------ #
# Execution time prediction
# ------------------------------------------------------------------ #


class TestExecutionTimePrediction:
    """Tests for predict_execution_time."""

    def test_serial_sum(self) -> None:
        """Intermediate skill → serial sum of durations."""
        cfg = AnalyzerConfig(klm_config=KLMConfig(skill_level=SkillLevel.INTERMEDIATE))
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.35),
            _make_op(OperatorType.K, 0.28),
        ])
        t = predict_execution_time([m], config=cfg)
        assert t == pytest.approx(1.35 + 0.28, rel=1e-9)

    def test_expert_overlap_reduces_time(self) -> None:
        """Expert skill with overlap should produce less time."""
        cfg_inter = AnalyzerConfig(klm_config=KLMConfig(skill_level=SkillLevel.INTERMEDIATE))
        cfg_expert = AnalyzerConfig(
            klm_config=KLMConfig(skill_level=SkillLevel.EXPERT),
            expert_overlap=0.3,
        )
        m = _make_method("m", "g", [
            _make_op(OperatorType.K, 0.28),
            _make_op(OperatorType.M, 1.35),
        ])
        t_inter = predict_execution_time([m], config=cfg_inter)
        t_expert = predict_execution_time([m], config=cfg_expert)
        assert t_expert <= t_inter

    def test_multiple_methods_additive(self) -> None:
        cfg = AnalyzerConfig()
        m1 = _make_method("m1", "g1", [_make_op(OperatorType.K, 0.28)])
        m2 = _make_method("m2", "g2", [_make_op(OperatorType.K, 0.28)])
        t = predict_execution_time([m1, m2], config=cfg)
        assert t == pytest.approx(0.56, rel=1e-9)

    def test_empty_methods(self) -> None:
        assert predict_execution_time([]) == 0.0


# ------------------------------------------------------------------ #
# Critical path analysis
# ------------------------------------------------------------------ #


class TestCriticalPath:
    """Tests for compute_critical_path_time."""

    def test_single_method_serial(self) -> None:
        """Single method → critical path = sum of operators."""
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.0),
            _make_op(OperatorType.K, 0.5),
        ])
        cpt = compute_critical_path_time([m])
        assert cpt == pytest.approx(1.5, rel=1e-9)

    def test_empty_methods(self) -> None:
        assert compute_critical_path_time([]) == 0.0

    def test_independent_methods_parallel(self) -> None:
        """Two independent methods — critical path = max, not sum."""
        m1 = _make_method("m1", "g1", [_make_op(OperatorType.K, 1.0)])
        m2 = _make_method("m2", "g2", [_make_op(OperatorType.K, 2.0)])
        cpt = compute_critical_path_time([m1, m2])
        # Since they're independent (no shared targets), critical path
        # is max of individual times
        assert cpt == pytest.approx(2.0, rel=1e-9)

    def test_shared_target_serialises(self) -> None:
        """Methods sharing a target element must serialise."""
        m1 = _make_method("m1", "g1", [
            _make_op(OperatorType.K, 1.0, target_id="btn"),
        ])
        m2 = _make_method("m2", "g2", [
            _make_op(OperatorType.K, 2.0, target_id="btn"),
        ])
        cpt = compute_critical_path_time([m1, m2])
        assert cpt == pytest.approx(3.0, rel=1e-9)


# ------------------------------------------------------------------ #
# Method selection
# ------------------------------------------------------------------ #


class TestMethodSelection:
    """Tests for bounded-rational method selection via GomsAnalyzerImpl."""

    def test_single_method_always_selected(self) -> None:
        """With only one method, it is always selected."""
        model = GomsModel(
            model_id="m", name="T",
            goals=(
                GomsGoal(goal_id="top", description="top", subgoal_ids=("g1",)),
                GomsGoal(goal_id="g1", description="g1", parent_id="top"),
            ),
            methods=(_make_method("only", "g1", [_make_op(OperatorType.K, 0.28)]),),
            top_level_goal_id="top",
        )
        analyzer = GomsAnalyzerImpl()
        trace = analyzer.trace(model)
        assert len(trace.methods_selected) == 1

    def test_high_beta_picks_fastest(self) -> None:
        """High rationality β should strongly prefer the fastest method."""
        fast = _make_method("fast", "g1", [_make_op(OperatorType.K, 0.1)])
        slow = _make_method("slow", "g1", [
            _make_op(OperatorType.K, 0.5),
            _make_op(OperatorType.K, 0.5),
        ])
        model = GomsModel(
            model_id="m", name="T",
            goals=(
                GomsGoal(goal_id="top", description="top", subgoal_ids=("g1",)),
                GomsGoal(goal_id="g1", description="g1", parent_id="top"),
            ),
            methods=(fast, slow),
            top_level_goal_id="top",
        )
        config = AnalyzerConfig(rationality_beta=100.0)
        analyzer = GomsAnalyzerImpl(config=config)
        # Run multiple times — high β should almost always pick fast
        fast_count = 0
        for _ in range(10):
            analyzer._rng = np.random.default_rng(42)
            trace = analyzer.trace(model)
            if trace.methods_selected and trace.methods_selected[0].method_id == "fast":
                fast_count += 1
        assert fast_count >= 9

    def test_explicit_policy(self) -> None:
        """Explicit policy mapping overrides stochastic selection."""
        m1 = _make_method("m1", "g1", [_make_op(OperatorType.K, 10.0)])
        m2 = _make_method("m2", "g1", [_make_op(OperatorType.K, 0.1)])
        model = GomsModel(
            model_id="m", name="T",
            goals=(
                GomsGoal(goal_id="top", description="top", subgoal_ids=("g1",)),
                GomsGoal(goal_id="g1", description="g1", parent_id="top"),
            ),
            methods=(m1, m2),
            top_level_goal_id="top",
        )
        analyzer = GomsAnalyzerImpl()
        trace = analyzer.trace(model, selection_policy={"g1": "m1"})
        assert trace.methods_selected[0].method_id == "m1"


# ------------------------------------------------------------------ #
# Operator merging
# ------------------------------------------------------------------ #


class TestOperatorMerging:
    """Tests for merge_operators_expert."""

    def test_consecutive_k_merged(self) -> None:
        """Consecutive K operators merge into one chunked K."""
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.35),
            _make_op(OperatorType.K, 0.12),
            _make_op(OperatorType.K, 0.12),
            _make_op(OperatorType.K, 0.12),
        ])
        merged = merge_operators_expert([m])
        merged_ops = merged[0].operators
        k_ops = [op for op in merged_ops if op.op_type == OperatorType.K]
        # Should have fewer K ops (merged into one)
        assert len(k_ops) <= 1

    def test_merged_k_preserves_total_motor_time(self) -> None:
        """Total K-operator time should be preserved in merge."""
        m = _make_method("m", "g", [
            _make_op(OperatorType.K, 0.12),
            _make_op(OperatorType.K, 0.12),
            _make_op(OperatorType.K, 0.12),
        ])
        merged = merge_operators_expert([m])
        orig_k_time = sum(op.duration_s for op in m.operators if op.op_type == OperatorType.K)
        merged_k_time = sum(op.duration_s for op in merged[0].operators if op.op_type == OperatorType.K)
        assert merged_k_time == pytest.approx(orig_k_time, rel=1e-9)

    def test_single_k_not_merged(self) -> None:
        """A lone K operator should not change."""
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.35),
            _make_op(OperatorType.K, 0.12),
        ])
        merged = merge_operators_expert([m])
        k_ops = [op for op in merged[0].operators if op.op_type == OperatorType.K]
        assert len(k_ops) == 1

    def test_m_halved_before_chunk(self) -> None:
        """M operator before a chunked K run should be halved."""
        m = _make_method("m", "g", [
            _make_op(OperatorType.M, 1.0),
            _make_op(OperatorType.K, 0.12),
            _make_op(OperatorType.K, 0.12),
        ])
        merged = merge_operators_expert([m])
        m_ops = [op for op in merged[0].operators if op.op_type == OperatorType.M]
        if m_ops:
            assert m_ops[0].duration_s == pytest.approx(0.5, rel=1e-9)


# ------------------------------------------------------------------ #
# Error probability estimation
# ------------------------------------------------------------------ #


class TestErrorProbability:
    """Tests for estimate_error_probability."""

    def test_zero_for_empty(self) -> None:
        assert estimate_error_probability([]) == pytest.approx(0.0)

    def test_increases_with_operators(self) -> None:
        """More operators → higher error probability."""
        m_short = _make_method("s", "g", [_make_op(OperatorType.K, 0.28)])
        m_long = _make_method("l", "g", [_make_op(OperatorType.K, 0.28)] * 20)
        assert estimate_error_probability([m_long]) > estimate_error_probability([m_short])

    def test_probability_in_range(self) -> None:
        m = _make_method("m", "g", [_make_op(OperatorType.K, 0.28)] * 10)
        p = estimate_error_probability([m])
        assert 0.0 <= p <= 1.0

    def test_novice_higher_error(self) -> None:
        """Novice should have higher error than expert."""
        m = _make_method("m", "g", [_make_op(OperatorType.K, 0.28)] * 5)
        cfg_nov = AnalyzerConfig(klm_config=KLMConfig(skill_level=SkillLevel.NOVICE))
        cfg_exp = AnalyzerConfig(klm_config=KLMConfig(skill_level=SkillLevel.EXPERT))
        assert estimate_error_probability([m], config=cfg_nov) > estimate_error_probability([m], config=cfg_exp)


# ------------------------------------------------------------------ #
# Learning time
# ------------------------------------------------------------------ #


class TestLearningTime:
    """Tests for estimate_learning_time."""

    def test_positive(self) -> None:
        model = _simple_model()
        lt = estimate_learning_time(model)
        assert lt > 0

    def test_more_methods_more_learning(self) -> None:
        small = GomsModel(
            model_id="s", name="S",
            methods=(_make_method("m1", "g1", [_make_op(OperatorType.K, 0.28)]),),
        )
        big = GomsModel(
            model_id="b", name="B",
            methods=(
                _make_method("m1", "g1", [_make_op(OperatorType.K, 0.28)] * 5),
                _make_method("m2", "g2", [_make_op(OperatorType.K, 0.28)] * 5),
            ),
        )
        assert estimate_learning_time(big) > estimate_learning_time(small)


# ------------------------------------------------------------------ #
# Trace comparison
# ------------------------------------------------------------------ #


class TestTraceComparison:
    """Tests for GomsAnalyzerImpl.compare_traces."""

    def test_identical_traces_no_regression(self) -> None:
        t = GomsTrace(trace_id="a", task_name="T", total_time_s=5.0)
        analyzer = GomsAnalyzerImpl()
        result = analyzer.compare_traces(t, t)
        assert result["time_delta_s"] == pytest.approx(0.0)
        assert result["regression"] is False

    def test_slower_new_version_regression(self) -> None:
        t_old = GomsTrace(
            trace_id="a", task_name="T", total_time_s=3.0,
            metadata={"error_probability": 0.05},
        )
        t_new = GomsTrace(
            trace_id="b", task_name="T", total_time_s=5.0,
            metadata={"error_probability": 0.05},
        )
        analyzer = GomsAnalyzerImpl()
        result = analyzer.compare_traces(t_old, t_new)
        assert result["time_delta_s"] == pytest.approx(2.0)
        assert result["regression"] is True
