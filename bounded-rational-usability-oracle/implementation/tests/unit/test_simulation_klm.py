"""Tests for usability_oracle.simulation.klm — Keystroke-Level Model.

Verifies operator timings from Card, Moran & Newell (1983), mental-
operator insertion rules (Rules 0–4), expert vs novice differences,
task time prediction, and edge cases.
"""

from __future__ import annotations

import pytest

from usability_oracle.simulation.klm import (
    KLMModel,
    KLMOperator,
    KLMStep,
    KLMTimings,
    apply_heuristic_rules,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_timings() -> KLMTimings:
    """Default Card et al. (1983) timing parameters."""
    return KLMTimings()


@pytest.fixture
def default_model() -> KLMModel:
    """KLM model with default average-typist timings."""
    return KLMModel()


@pytest.fixture
def expert_model() -> KLMModel:
    """KLM model configured for expert users."""
    return KLMModel(skill_level="expert")


@pytest.fixture
def novice_model() -> KLMModel:
    """KLM model configured for novice users."""
    return KLMModel(skill_level="novice")


# =====================================================================
# K operator — keystroke
# =====================================================================

class TestKOperator:
    """Test keystroke operator timing."""

    def test_average_typist(self, default_timings: KLMTimings) -> None:
        """K operator = 0.28s for average skilled typist (55 wpm)."""
        assert default_timings.t_k_average == pytest.approx(0.28)

    def test_expert_typist(self, default_timings: KLMTimings) -> None:
        """K operator = 0.12s for expert typist (135 wpm)."""
        assert default_timings.t_k_expert == pytest.approx(0.12)

    def test_worst_case_typist(self, default_timings: KLMTimings) -> None:
        """K operator = 0.50s for worst-case (non-typist, 40 wpm)."""
        assert default_timings.t_k_worst == pytest.approx(0.50)

    def test_get_time_average(self, default_timings: KLMTimings) -> None:
        """get_time with 'average' skill returns t_k_average."""
        assert default_timings.get_time(KLMOperator.K, "average") == 0.28

    def test_get_time_expert(self, default_timings: KLMTimings) -> None:
        """get_time with 'expert' skill returns t_k_expert."""
        assert default_timings.get_time(KLMOperator.K, "expert") == 0.12


# =====================================================================
# P operator — pointing
# =====================================================================

class TestPOperator:
    """Test pointing operator timing."""

    def test_pointing_time(self, default_timings: KLMTimings) -> None:
        """P operator = 1.1s (Fitts' law average pointing time)."""
        assert default_timings.t_p == pytest.approx(1.1)

    def test_get_time_p(self, default_timings: KLMTimings) -> None:
        """get_time for P returns t_p."""
        assert default_timings.get_time(KLMOperator.P) == 1.1


# =====================================================================
# H operator — homing
# =====================================================================

class TestHOperator:
    """Test homing operator timing."""

    def test_homing_time(self, default_timings: KLMTimings) -> None:
        """H operator = 0.4s (keyboard ↔ mouse hand movement)."""
        assert default_timings.t_h == pytest.approx(0.4)


# =====================================================================
# M operator — mental preparation
# =====================================================================

class TestMOperator:
    """Test mental preparation operator timing."""

    def test_mental_preparation_time(self, default_timings: KLMTimings) -> None:
        """M operator = 1.35s."""
        assert default_timings.t_m == pytest.approx(1.35)


# =====================================================================
# Task time = sum of operators
# =====================================================================

class TestTaskTimePrediction:
    """Test that task time equals sum of operator durations."""

    def test_sum_of_operator_times(self, default_model: KLMModel) -> None:
        """T = Σ t_op for each operator in sequence."""
        ops = [
            KLMStep(operator=KLMOperator.M, duration=1.35),
            KLMStep(operator=KLMOperator.P, duration=1.1),
            KLMStep(operator=KLMOperator.K, duration=0.28),
        ]
        total = default_model.predict_task_time(ops)
        expected = 1.35 + 1.1 + 0.28
        assert total == pytest.approx(expected)

    def test_empty_sequence_returns_zero(self, default_model: KLMModel) -> None:
        """Empty operator sequence → T = 0."""
        assert default_model.predict_task_time([]) == 0.0

    def test_breakdown_sums_to_total(self, default_model: KLMModel) -> None:
        """Per-operator-type breakdown should sum to total."""
        ops = [
            KLMStep(operator=KLMOperator.M, duration=1.35),
            KLMStep(operator=KLMOperator.P, duration=1.1),
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.K, duration=0.28),
        ]
        breakdown = default_model.predict_with_breakdown(ops)
        assert "total" in breakdown
        non_total = sum(v for k, v in breakdown.items() if k != "total")
        assert breakdown["total"] == pytest.approx(non_total)

    @pytest.mark.parametrize("n_keys,expected_extra", [
        (1, 0.28),
        (5, 5 * 0.28),
        (10, 10 * 0.28),
    ])
    def test_multiple_keystrokes(
        self, default_model: KLMModel, n_keys: int, expected_extra: float
    ) -> None:
        """Multiple keystrokes should accumulate linearly."""
        ops = [KLMStep(operator=KLMOperator.K, duration=0.28) for _ in range(n_keys)]
        total = default_model.predict_task_time(ops)
        assert total == pytest.approx(expected_extra)


# =====================================================================
# Rule 0 — M inserted before K/P
# =====================================================================

class TestRule0MInsertion:
    """Test that M operators are inserted before K and P operators."""

    def test_m_inserted_before_k(self) -> None:
        """Rule 0: M should be inserted before K."""
        ops = [KLMStep(operator=KLMOperator.K, description="type a")]
        result = apply_heuristic_rules(ops)
        # First operator should be M
        assert result[0].operator == KLMOperator.M

    def test_m_inserted_before_p(self) -> None:
        """Rule 0: M should be inserted before P."""
        ops = [KLMStep(operator=KLMOperator.P, description="point to button")]
        result = apply_heuristic_rules(ops)
        assert result[0].operator == KLMOperator.M

    def test_no_m_before_h(self) -> None:
        """H (homing) should NOT get an M inserted before it."""
        ops = [KLMStep(operator=KLMOperator.H, description="home to mouse")]
        result = apply_heuristic_rules(ops)
        # Should just be H (no M inserted)
        assert len([s for s in result if s.operator == KLMOperator.M]) == 0


# =====================================================================
# Expert vs novice timing differences
# =====================================================================

class TestExpertVsNovice:
    """Test that expert and novice timings differ appropriately."""

    def test_expert_faster_keystrokes(
        self, expert_model: KLMModel, novice_model: KLMModel
    ) -> None:
        """Expert keystroke time < novice keystroke time."""
        ops = [KLMStep(operator=KLMOperator.K)]
        expert_time = expert_model.predict_task_time(ops)
        novice_time = novice_model.predict_task_time(ops)
        assert expert_time < novice_time

    def test_expert_time_is_012(self, expert_model: KLMModel) -> None:
        """Expert K = 0.12s."""
        ops = [KLMStep(operator=KLMOperator.K)]
        assert expert_model.predict_task_time(ops) == pytest.approx(0.12)

    def test_novice_time_is_050(self, novice_model: KLMModel) -> None:
        """Novice K = 0.50s."""
        ops = [KLMStep(operator=KLMOperator.K)]
        assert novice_model.predict_task_time(ops) == pytest.approx(0.50)

    def test_pointing_same_across_skills(
        self, expert_model: KLMModel, novice_model: KLMModel
    ) -> None:
        """Pointing time (P) is the same for all skill levels."""
        ops = [KLMStep(operator=KLMOperator.P)]
        assert expert_model.predict_task_time(ops) == novice_model.predict_task_time(ops)


# =====================================================================
# KLM operator properties
# =====================================================================

class TestKLMOperatorProperties:
    """Test operator classification properties."""

    @pytest.mark.parametrize("op,expected", [
        (KLMOperator.K, True),
        (KLMOperator.P, True),
        (KLMOperator.H, True),
        (KLMOperator.B, True),
        (KLMOperator.M, False),
        (KLMOperator.R, False),
    ])
    def test_is_physical(self, op: KLMOperator, expected: bool) -> None:
        """Physical operators: K, P, H, B."""
        assert op.is_physical == expected

    @pytest.mark.parametrize("op,expected", [
        (KLMOperator.M, True),
        (KLMOperator.W, True),
        (KLMOperator.K, False),
    ])
    def test_is_cognitive(self, op: KLMOperator, expected: bool) -> None:
        """Cognitive operators: M, W."""
        assert op.is_cognitive == expected

    def test_r_is_system(self) -> None:
        """R (system response) is a system operator."""
        assert KLMOperator.R.is_system


# =====================================================================
# KLMStep with_duration
# =====================================================================

class TestKLMStepWithDuration:
    """Test KLMStep.with_duration populates timing."""

    def test_with_duration_populates(self, default_timings: KLMTimings) -> None:
        """with_duration should set the duration from timings."""
        step = KLMStep(operator=KLMOperator.K)
        step_timed = step.with_duration(default_timings, skill="average")
        assert step_timed.duration == pytest.approx(0.28)

    def test_with_duration_expert(self, default_timings: KLMTimings) -> None:
        """with_duration for expert keystroke."""
        step = KLMStep(operator=KLMOperator.K)
        step_timed = step.with_duration(default_timings, skill="expert")
        assert step_timed.duration == pytest.approx(0.12)


# =====================================================================
# Complex task sequence
# =====================================================================

class TestComplexTaskSequence:
    """Test a realistic multi-operator task sequence."""

    def test_click_and_type_sequence(self, default_model: KLMModel) -> None:
        """Typical click-then-type: H + P + K*5 (typing 'hello')."""
        ops = [
            KLMStep(operator=KLMOperator.H, duration=0.4),
            KLMStep(operator=KLMOperator.P, duration=1.1),
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.K, duration=0.28),
        ]
        total = default_model.predict_task_time(ops)
        expected = 0.4 + 1.1 + 5 * 0.28
        assert total == pytest.approx(expected)

    def test_system_response_time(self, default_model: KLMModel) -> None:
        """R operator adds system response time."""
        ops = [
            KLMStep(operator=KLMOperator.K, duration=0.28),
            KLMStep(operator=KLMOperator.R, duration=2.0),  # 2s server wait
            KLMStep(operator=KLMOperator.K, duration=0.28),
        ]
        total = default_model.predict_task_time(ops)
        assert total == pytest.approx(0.28 + 2.0 + 0.28)
