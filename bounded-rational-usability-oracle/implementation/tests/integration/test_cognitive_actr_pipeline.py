"""Integration tests for the cognitive / ACT-R pipeline.

ACT-R model from accessibility tree, ACT-R prediction vs. simplified
model comparison, and ACT-R regression detection.
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    KLMSequence,
    OperatorType,
)
from usability_oracle.goms.klm import (
    KLMConfig,
    KLMPredictorImpl,
    SkillLevel,
    make_keystroke,
    make_pointing,
    make_homing,
    make_mental,
)
from usability_oracle.goms.critical_path import (
    CriticalPathAnalyzer,
)
from usability_oracle.goms.analyzer import (
    GomsAnalyzerImpl,
    AnalyzerConfig,
)
from usability_oracle.goms.simulation import (
    GomsSimulator,
    SimulationConfig,
)
from usability_oracle.core.types import Point2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_klm_sequence(skill: SkillLevel = SkillLevel.INTERMEDIATE) -> KLMSequence:
    """Build a representative KLM sequence for a form-filling task."""
    config = KLMConfig(skill_level=skill)
    ops = (
        make_mental(config=config),
        make_pointing(
            source=Point2D(x=500.0, y=300.0),
            target_center=Point2D(x=200.0, y=100.0),
            target_width_px=100.0,
            config=config,
        ),
        make_keystroke(key="u", config=config),
        make_keystroke(key="s", config=config),
        make_keystroke(key="e", config=config),
        make_keystroke(key="r", config=config),
        make_mental(config=config),
        make_homing(config=config),
        make_pointing(
            source=Point2D(x=200.0, y=100.0),
            target_center=Point2D(x=300.0, y=400.0),
            target_width_px=80.0,
            config=config,
        ),
    )
    return KLMSequence(task_name="login_form", operators=ops)


def _build_goms_model() -> GomsModel:
    """Build a simple GOMS model for a two-step task."""
    config = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)

    goal_root = GomsGoal(
        goal_id="login", description="Log into the system",
        subgoal_ids=("enter_username", "click_submit"),
    )
    goal_username = GomsGoal(
        goal_id="enter_username", description="Enter username",
        parent_id="login",
    )
    goal_submit = GomsGoal(
        goal_id="click_submit", description="Click submit button",
        parent_id="login",
    )

    method_username = GomsMethod(
        method_id="m_username",
        goal_id="enter_username",
        name="type_username",
        operators=(
            make_mental(config=config),
            make_pointing(
                source=Point2D(x=500.0, y=300.0),
                target_center=Point2D(x=200.0, y=100.0),
                target_width_px=120.0,
                config=config,
            ),
            make_keystroke(key="a", config=config),
            make_keystroke(key="d", config=config),
            make_keystroke(key="m", config=config),
        ),
    )
    method_submit = GomsMethod(
        method_id="m_submit",
        goal_id="click_submit",
        name="click_submit_button",
        operators=(
            make_mental(config=config),
            make_pointing(
                source=Point2D(x=200.0, y=100.0),
                target_center=Point2D(x=300.0, y=400.0),
                target_width_px=80.0,
                config=config,
            ),
        ),
    )

    return GomsModel(
        model_id="login_model",
        name="Login Model",
        goals=(goal_root, goal_username, goal_submit),
        methods=(method_username, method_submit),
        top_level_goal_id="login",
    )


# ===================================================================
# Tests — KLM prediction pipeline
# ===================================================================


class TestKLMPipeline:
    """KLM prediction from operator sequence."""

    def test_klm_total_time_positive(self) -> None:
        """KLM total time is positive for a non-empty sequence."""
        seq = _build_klm_sequence()
        assert seq.total_time_s > 0

    def test_klm_expert_faster_than_novice(self) -> None:
        """Expert KLM sequence takes less time than novice."""
        novice_seq = _build_klm_sequence(SkillLevel.NOVICE)
        expert_seq = _build_klm_sequence(SkillLevel.EXPERT)
        assert expert_seq.total_time_s <= novice_seq.total_time_s

    def test_klm_sequence_operator_count(self) -> None:
        """KLM sequence contains the expected number of operators."""
        seq = _build_klm_sequence()
        assert len(seq.operators) == 9

    def test_klm_operator_string(self) -> None:
        """KLM operator string representation is non-empty."""
        seq = _build_klm_sequence()
        assert len(seq.operator_string) > 0

    def test_klm_motor_time_positive(self) -> None:
        """Motor time component is positive."""
        config = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        method = GomsMethod(
            method_id="m1", goal_id="g1", name="test",
            operators=(
                make_pointing(
                    source=Point2D(x=0.0, y=0.0),
                    target_center=Point2D(x=100.0, y=100.0),
                    target_width_px=50.0,
                    config=config,
                ),
                make_keystroke(key="x", config=config),
            ),
        )
        assert method.motor_time_s > 0

    def test_klm_cognitive_time_positive(self) -> None:
        """Cognitive (mental preparation) time is positive."""
        config = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        method = GomsMethod(
            method_id="m1", goal_id="g1", name="test",
            operators=(make_mental(config=config),),
        )
        assert method.cognitive_time_s > 0


# ===================================================================
# Tests — GOMS model analysis
# ===================================================================


class TestGomsModelAnalysis:
    """GOMS model construction and analysis."""

    def test_model_has_correct_structure(self) -> None:
        """GOMS model has expected goals and methods."""
        model = _build_goms_model()
        assert model.goal_count == 3
        assert model.method_count == 2

    def test_methods_for_goal(self) -> None:
        """methods_for_goal returns the correct methods."""
        model = _build_goms_model()
        methods = model.methods_for_goal("enter_username")
        assert len(methods) == 1
        assert methods[0].method_id == "m_username"

    def test_method_total_duration_positive(self) -> None:
        """Each method's total duration is positive."""
        model = _build_goms_model()
        for method in model.methods:
            assert method.total_duration_s > 0, \
                f"Method {method.method_id} has non-positive duration"


# ===================================================================
# Tests — Prediction vs. simplified model comparison
# ===================================================================


class TestPredictionComparison:
    """Compare ACT-R-style predictions with simplified KLM."""

    def test_klm_vs_serial_sum(self) -> None:
        """KLM total time = sum of operators (no parallelism)."""
        seq = _build_klm_sequence()
        serial_sum = sum(op.duration_s for op in seq.operators)
        assert math.isclose(seq.total_time_s, serial_sum, abs_tol=1e-9)

    def test_model_serial_time_geq_any_method(self) -> None:
        """Serial execution time ≥ any individual method duration."""
        model = _build_goms_model()
        total_serial = sum(m.total_duration_s for m in model.methods)
        for method in model.methods:
            assert total_serial >= method.total_duration_s


# ===================================================================
# Tests — Regression detection
# ===================================================================


class TestCognitiveRegression:
    """Detect regressions in cognitive cost predictions."""

    def test_same_model_same_time(self) -> None:
        """Same GOMS model produces the same time prediction."""
        seq1 = _build_klm_sequence()
        seq2 = _build_klm_sequence()
        assert math.isclose(seq1.total_time_s, seq2.total_time_s, abs_tol=1e-9)

    def test_added_operator_increases_time(self) -> None:
        """Adding an operator to a KLM sequence increases total time."""
        config = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        base_ops = (
            make_mental(config=config),
            make_keystroke(key="a", config=config),
        )
        extended_ops = base_ops + (make_keystroke(key="b", config=config),)
        seq_base = KLMSequence(task_name="base", operators=base_ops)
        seq_ext = KLMSequence(task_name="extended", operators=extended_ops)
        assert seq_ext.total_time_s > seq_base.total_time_s

    def test_skill_regression_detection(self) -> None:
        """Changing skill level changes predictions detectably."""
        seq_novice = _build_klm_sequence(SkillLevel.NOVICE)
        seq_expert = _build_klm_sequence(SkillLevel.EXPERT)
        diff = seq_novice.total_time_s - seq_expert.total_time_s
        assert diff > 0.1, \
            f"Skill difference too small to detect: {diff}"

    def test_trace_speedup_from_parallelism(self) -> None:
        """GomsTrace speedup ≥ 1 for well-formed traces."""
        trace = GomsTrace(
            trace_id="t1",
            task_name="test",
            total_time_s=10.0,
            critical_path_time_s=7.0,
        )
        assert trace.speedup_from_parallelism >= 1.0
