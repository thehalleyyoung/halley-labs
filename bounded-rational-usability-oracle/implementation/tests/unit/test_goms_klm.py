"""Unit tests for usability_oracle.goms.klm.

Tests cover KLM operator times, M-operator placement rules, skill-level
calibration, sequence optimisation, and Fitts' law integration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.core.types import Point2D
from usability_oracle.goms.types import GomsOperator, KLMSequence, OperatorType
from usability_oracle.goms.klm import (
    KLMConfig,
    SkillLevel,
    apply_placement_rules,
    build_klm_sequence,
    compute_sequence_time,
    estimate_overlap,
    fitts_time,
    hick_hyman_time,
    make_button_press,
    make_homing,
    make_keystroke,
    make_mental,
    make_pointing,
    make_system_response,
    optimize_klm_sequence,
)


# ------------------------------------------------------------------ #
# KLM operator times
# ------------------------------------------------------------------ #


class TestKLMOperatorTimes:
    """Tests for default operator durations by type."""

    def test_keystroke_intermediate(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        k = make_keystroke("a", config=cfg)
        assert k.duration_s == pytest.approx(0.28)

    def test_keystroke_novice(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
        k = make_keystroke("a", config=cfg)
        assert k.duration_s == pytest.approx(0.40)

    def test_keystroke_expert(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
        k = make_keystroke("a", config=cfg)
        assert k.duration_s == pytest.approx(0.12)

    def test_homing_intermediate(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        h = make_homing(config=cfg)
        assert h.duration_s == pytest.approx(0.40)

    def test_mental_intermediate(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        m = make_mental(config=cfg)
        assert m.duration_s == pytest.approx(1.35)

    def test_mental_novice(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
        m = make_mental(config=cfg)
        assert m.duration_s == pytest.approx(1.50)

    def test_mental_expert(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
        m = make_mental(config=cfg)
        assert m.duration_s == pytest.approx(1.10)

    def test_button_press_intermediate(self) -> None:
        cfg = KLMConfig(skill_level=SkillLevel.INTERMEDIATE)
        b = make_button_press(config=cfg)
        assert b.duration_s == pytest.approx(0.10)

    def test_system_response_default(self) -> None:
        cfg = KLMConfig()
        r = make_system_response(config=cfg)
        assert r.duration_s == pytest.approx(cfg.system_response_s)

    def test_system_response_custom(self) -> None:
        r = make_system_response(latency_s=2.0)
        assert r.duration_s == pytest.approx(2.0)

    def test_operator_type_tags(self) -> None:
        """Each operator constructor uses the correct OperatorType."""
        assert make_keystroke().op_type == OperatorType.K
        assert make_homing().op_type == OperatorType.H
        assert make_mental().op_type == OperatorType.M
        assert make_button_press().op_type == OperatorType.B
        assert make_system_response().op_type == OperatorType.R


# ------------------------------------------------------------------ #
# M-operator placement rules
# ------------------------------------------------------------------ #


class TestMOperatorPlacement:
    """Tests for Card/Moran/Newell M-placement heuristics."""

    @pytest.fixture
    def cfg(self) -> KLMConfig:
        return KLMConfig(skill_level=SkillLevel.INTERMEDIATE)

    def test_rule0_m_before_k_and_p(self, cfg: KLMConfig) -> None:
        """Rule 0: M inserted before every K and P."""
        k = make_keystroke(config=cfg)
        ops = [k]
        result = apply_placement_rules(ops, config=cfg)
        assert result[0].op_type == OperatorType.M
        assert result[1].op_type == OperatorType.K

    def test_rule1_delete_m_in_string(self, cfg: KLMConfig) -> None:
        """Rule 1: Delete M before K continuing a keystroke string."""
        k1 = make_keystroke("a", config=cfg)
        k2 = make_keystroke("b", config=cfg)
        result = apply_placement_rules([k1, k2], config=cfg)
        # Should have M K K, not M K M K
        types = [op.op_type for op in result]
        assert types == [OperatorType.M, OperatorType.K, OperatorType.K]

    def test_rule4_delete_m_after_r(self, cfg: KLMConfig) -> None:
        """Rule 4: Delete M that directly follows R."""
        r = make_system_response(config=cfg)
        k = make_keystroke(config=cfg)
        result = apply_placement_rules([r, k], config=cfg)
        # After R, the M before K should be deleted
        types = [op.op_type for op in result]
        assert types[0] == OperatorType.R
        # Check M is removed after R
        assert types[1] == OperatorType.K

    def test_rule5_collapse_consecutive_m(self, cfg: KLMConfig) -> None:
        """Rule 5: Consecutive Ms collapse to one."""
        p = make_pointing(
            Point2D(0, 0), Point2D(100, 100), 50.0, config=cfg,
        )
        k = make_keystroke(config=cfg)
        result = apply_placement_rules([p, k], config=cfg)
        # Count M operators
        m_count = sum(1 for op in result if op.op_type == OperatorType.M)
        # After all rules, should not have two consecutive Ms
        for i in range(len(result) - 1):
            if result[i].op_type == OperatorType.M:
                assert result[i + 1].op_type != OperatorType.M

    def test_empty_input(self, cfg: KLMConfig) -> None:
        result = apply_placement_rules([], config=cfg)
        assert result == ()


# ------------------------------------------------------------------ #
# Skill level calibration
# ------------------------------------------------------------------ #


class TestSkillLevelCalibration:
    """Verify that skill level affects operator durations monotonically."""

    def test_keystroke_speed_ordering(self) -> None:
        """Expert < Intermediate < Novice for K duration."""
        configs = {
            level: KLMConfig(skill_level=level) for level in SkillLevel
        }
        assert configs[SkillLevel.EXPERT].k_duration < configs[SkillLevel.INTERMEDIATE].k_duration
        assert configs[SkillLevel.INTERMEDIATE].k_duration < configs[SkillLevel.NOVICE].k_duration

    def test_mental_speed_ordering(self) -> None:
        """Expert < Intermediate < Novice for M duration."""
        configs = {
            level: KLMConfig(skill_level=level) for level in SkillLevel
        }
        assert configs[SkillLevel.EXPERT].m_duration < configs[SkillLevel.INTERMEDIATE].m_duration
        assert configs[SkillLevel.INTERMEDIATE].m_duration < configs[SkillLevel.NOVICE].m_duration

    def test_homing_speed_ordering(self) -> None:
        configs = {
            level: KLMConfig(skill_level=level) for level in SkillLevel
        }
        assert configs[SkillLevel.EXPERT].h_duration < configs[SkillLevel.INTERMEDIATE].h_duration
        assert configs[SkillLevel.INTERMEDIATE].h_duration < configs[SkillLevel.NOVICE].h_duration

    def test_button_speed_ordering(self) -> None:
        configs = {
            level: KLMConfig(skill_level=level) for level in SkillLevel
        }
        assert configs[SkillLevel.EXPERT].b_duration < configs[SkillLevel.INTERMEDIATE].b_duration
        assert configs[SkillLevel.INTERMEDIATE].b_duration < configs[SkillLevel.NOVICE].b_duration

    def test_expert_total_time_less(self) -> None:
        """Same sequence is faster for experts than novices."""
        for level in [SkillLevel.NOVICE, SkillLevel.EXPERT]:
            cfg = KLMConfig(skill_level=level)
            k = make_keystroke(config=cfg)
            m = make_mental(config=cfg)
            ops = [m, k, k, k]
            time = compute_sequence_time(ops, config=cfg)
            if level == SkillLevel.NOVICE:
                novice_time = time
            else:
                expert_time = time
        assert expert_time < novice_time


# ------------------------------------------------------------------ #
# Sequence optimisation
# ------------------------------------------------------------------ #


class TestSequenceOptimisation:
    """Tests for KLM sequence building and optimisation."""

    def test_build_klm_sequence(self) -> None:
        cfg = KLMConfig()
        k = make_keystroke(config=cfg)
        seq = build_klm_sequence("test_task", [k, k, k], config=cfg)
        assert seq.task_name == "test_task"
        assert seq.mental_prep_placed is True
        assert seq.total_time_s > 0

    def test_build_without_m_placement(self) -> None:
        cfg = KLMConfig()
        k = make_keystroke(config=cfg)
        seq = build_klm_sequence("test_task", [k, k], config=cfg, apply_m_placement=False)
        assert seq.mental_prep_placed is False
        types = [op.op_type for op in seq.operators]
        assert OperatorType.M not in types

    def test_optimize_reduces_or_maintains(self) -> None:
        """Optimisation should not increase total time."""
        cfg = KLMConfig()
        k = make_keystroke(config=cfg)
        h = make_homing(config=cfg)
        seq = build_klm_sequence("task", [k, h, h, k], config=cfg)
        opt_seq, report = optimize_klm_sequence(seq, config=cfg)
        assert opt_seq.total_time_s <= seq.total_time_s + 1e-10

    def test_compute_sequence_time_sum(self) -> None:
        """Without overlap, time = sum of durations."""
        cfg = KLMConfig(overlap_fraction=0.0)
        k = make_keystroke(config=cfg)
        m = make_mental(config=cfg)
        ops = [m, k, k]
        total = compute_sequence_time(ops, config=cfg)
        expected = sum(op.duration_s for op in ops)
        assert total == pytest.approx(expected)

    def test_overlap_saves_time(self) -> None:
        """With overlap_fraction > 0, cognitive-motor overlap reduces time."""
        cfg = KLMConfig(overlap_fraction=0.3)
        k = make_keystroke(config=cfg)
        m = make_mental(config=cfg)
        ops = [k, m]  # motor followed by mental
        savings = estimate_overlap(ops, config=cfg)
        assert savings > 0

    def test_zero_overlap_no_savings(self) -> None:
        cfg = KLMConfig(overlap_fraction=0.0)
        k = make_keystroke(config=cfg)
        m = make_mental(config=cfg)
        assert estimate_overlap([k, m], config=cfg) == 0.0


# ------------------------------------------------------------------ #
# Fitts' law integration
# ------------------------------------------------------------------ #


class TestFittsLawIntegration:
    """Tests for Fitts' law pointing time computation."""

    def test_fitts_basic(self) -> None:
        """MT = a + b * log2(D/W + 1)."""
        cfg = KLMConfig()
        d_px = 200.0
        w_px = 50.0
        mt = fitts_time(d_px, w_px, cfg)
        d_mm = d_px * cfg.pixel_to_mm
        w_mm = w_px * cfg.pixel_to_mm
        expected = cfg.fitts_a + cfg.fitts_b * math.log2(d_mm / w_mm + 1.0)
        assert mt == pytest.approx(expected, rel=1e-9)

    def test_fitts_zero_distance(self) -> None:
        """Zero distance → MT = a + b * log2(1) = a."""
        cfg = KLMConfig()
        mt = fitts_time(0.0, 50.0, cfg)
        assert mt == pytest.approx(cfg.fitts_a, rel=1e-9)

    def test_fitts_large_target_fast(self) -> None:
        """Larger target → smaller ID → faster."""
        cfg = KLMConfig()
        mt_small = fitts_time(200.0, 10.0, cfg)
        mt_large = fitts_time(200.0, 100.0, cfg)
        assert mt_large < mt_small

    def test_fitts_far_target_slow(self) -> None:
        """Greater distance → larger ID → slower."""
        cfg = KLMConfig()
        mt_near = fitts_time(50.0, 50.0, cfg)
        mt_far = fitts_time(500.0, 50.0, cfg)
        assert mt_far > mt_near

    def test_make_pointing_uses_fitts(self) -> None:
        """P operator created via make_pointing applies Fitts' law."""
        cfg = KLMConfig()
        src = Point2D(0.0, 0.0)
        tgt = Point2D(200.0, 0.0)
        p = make_pointing(src, tgt, 50.0, config=cfg)
        assert p.op_type == OperatorType.P
        expected = fitts_time(200.0, 50.0, cfg)
        assert p.duration_s == pytest.approx(expected, rel=1e-9)

    def test_hick_hyman_time_monotone(self) -> None:
        """More choices → longer decision time."""
        cfg = KLMConfig()
        t2 = hick_hyman_time(2, cfg)
        t8 = hick_hyman_time(8, cfg)
        assert t8 > t2

    def test_hick_hyman_one_choice(self) -> None:
        cfg = KLMConfig()
        t = hick_hyman_time(1, cfg)
        expected = cfg.hick_a + cfg.hick_b * math.log2(2)
        assert t == pytest.approx(expected, rel=1e-9)

    def test_mental_with_choices_uses_hick(self) -> None:
        """M operator with n_choices > 0 uses Hick-Hyman duration."""
        cfg = KLMConfig()
        m = make_mental(config=cfg, n_choices=4)
        expected = hick_hyman_time(4, cfg)
        assert m.duration_s == pytest.approx(expected, rel=1e-9)
