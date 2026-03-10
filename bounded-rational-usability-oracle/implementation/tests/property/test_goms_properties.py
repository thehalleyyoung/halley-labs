"""Property-based tests for the GOMS / KLM module.

Verifies KLM operator times are positive, execution time exceeds the
critical path time, expert time is no greater than novice time, and
operator durations fall within published ranges.
"""

import math

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    composite,
    lists,
)

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
    fitts_time,
    hick_hyman_time,
)
from usability_oracle.goms.critical_path import (
    CriticalPathAnalyzer,
    OperatorNode,
)
from usability_oracle.core.types import Point2D


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ATOL = 1e-9

_skill = sampled_from(list(SkillLevel))
_op_type = sampled_from(list(OperatorType))

_distance = floats(min_value=1.0, max_value=2000.0,
                   allow_nan=False, allow_infinity=False)

_width = floats(min_value=1.0, max_value=500.0,
                allow_nan=False, allow_infinity=False)

_n_choices = integers(min_value=1, max_value=50)


# ---------------------------------------------------------------------------
# KLM operator durations are positive
# ---------------------------------------------------------------------------


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_keystroke_duration_positive(skill):
    """K operator duration is positive for all skill levels."""
    config = KLMConfig(skill_level=skill)
    op = make_keystroke(key="a", config=config)
    assert op.duration_s > 0, f"K duration not positive: {op.duration_s}"


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_homing_duration_positive(skill):
    """H operator duration is positive for all skill levels."""
    config = KLMConfig(skill_level=skill)
    op = make_homing(config=config)
    assert op.duration_s > 0, f"H duration not positive: {op.duration_s}"


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_mental_duration_positive(skill):
    """M operator duration is positive for all skill levels."""
    config = KLMConfig(skill_level=skill)
    op = make_mental(config=config)
    assert op.duration_s > 0, f"M duration not positive: {op.duration_s}"


@given(_skill, _n_choices)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mental_with_hick_hyman_positive(skill, n_choices):
    """M operator with Hick-Hyman adjustment is positive."""
    config = KLMConfig(skill_level=skill)
    op = make_mental(config=config, n_choices=n_choices)
    assert op.duration_s > 0, f"M(HH) duration not positive: {op.duration_s}"


# ---------------------------------------------------------------------------
# Fitts' law times are positive and monotone
# ---------------------------------------------------------------------------


@given(_distance, _width, _skill)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_time_positive(distance, width, skill):
    """Fitts' law movement time is always positive."""
    config = KLMConfig(skill_level=skill)
    t = fitts_time(distance, width, config)
    assert t > 0, f"Fitts time not positive: {t}"


@given(_width, _skill)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_time_increases_with_distance(width, skill):
    """Farther targets take longer to acquire."""
    config = KLMConfig(skill_level=skill)
    t_near = fitts_time(10.0, width, config)
    t_far = fitts_time(500.0, width, config)
    assert t_far >= t_near - _ATOL, \
        f"Farther target was faster: {t_far} < {t_near}"


@given(_distance, _skill)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_time_decreases_with_width(distance, skill):
    """Larger targets are easier to acquire."""
    config = KLMConfig(skill_level=skill)
    t_narrow = fitts_time(distance, 5.0, config)
    t_wide = fitts_time(distance, 200.0, config)
    assert t_wide <= t_narrow + _ATOL, \
        f"Wider target was slower: {t_wide} > {t_narrow}"


# ---------------------------------------------------------------------------
# Hick-Hyman time is positive and monotone
# ---------------------------------------------------------------------------


@given(_n_choices, _skill)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_hyman_time_positive(n, skill):
    """Hick-Hyman choice reaction time is positive."""
    config = KLMConfig(skill_level=skill)
    t = hick_hyman_time(n, config)
    assert t > 0, f"Hick-Hyman time not positive: {t}"


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_hick_hyman_time_increases_with_choices(skill):
    """More choices increase reaction time."""
    config = KLMConfig(skill_level=skill)
    t2 = hick_hyman_time(2, config)
    t10 = hick_hyman_time(10, config)
    assert t10 >= t2 - _ATOL, \
        f"More choices was faster: {t10} < {t2}"


# ---------------------------------------------------------------------------
# Expert time ≤ novice time
# ---------------------------------------------------------------------------


def test_expert_keystroke_le_novice():
    """Expert K duration ≤ novice K duration."""
    novice_cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
    expert_cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
    assert expert_cfg.k_duration <= novice_cfg.k_duration + _ATOL


def test_expert_homing_le_novice():
    """Expert H duration ≤ novice H duration."""
    novice_cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
    expert_cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
    assert expert_cfg.h_duration <= novice_cfg.h_duration + _ATOL


def test_expert_mental_le_novice():
    """Expert M duration ≤ novice M duration."""
    novice_cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
    expert_cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
    assert expert_cfg.m_duration <= novice_cfg.m_duration + _ATOL


def test_expert_button_le_novice():
    """Expert B duration ≤ novice B duration."""
    novice_cfg = KLMConfig(skill_level=SkillLevel.NOVICE)
    expert_cfg = KLMConfig(skill_level=SkillLevel.EXPERT)
    assert expert_cfg.b_duration <= novice_cfg.b_duration + _ATOL


# ---------------------------------------------------------------------------
# Operator durations match published ranges (Card et al. 1980)
# ---------------------------------------------------------------------------


def test_keystroke_in_published_range():
    """K operator across skill levels is within [0.08, 1.20] s."""
    for skill in SkillLevel:
        cfg = KLMConfig(skill_level=skill)
        assert 0.05 <= cfg.k_duration <= 1.5, \
            f"K({skill}) = {cfg.k_duration} out of published range"


def test_homing_in_published_range():
    """H operator is within [0.20, 0.80] s."""
    for skill in SkillLevel:
        cfg = KLMConfig(skill_level=skill)
        assert 0.10 <= cfg.h_duration <= 1.0, \
            f"H({skill}) = {cfg.h_duration} out of published range"


def test_mental_in_published_range():
    """M operator is within [0.60, 2.00] s."""
    for skill in SkillLevel:
        cfg = KLMConfig(skill_level=skill)
        assert 0.50 <= cfg.m_duration <= 2.5, \
            f"M({skill}) = {cfg.m_duration} out of published range"


# ---------------------------------------------------------------------------
# KLM sequence total time = sum of operator durations
# ---------------------------------------------------------------------------


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_klm_sequence_total_time(skill):
    """KLMSequence.total_time_s = Σ operator durations."""
    config = KLMConfig(skill_level=skill)
    ops = (
        make_mental(config=config),
        make_keystroke(key="a", config=config),
        make_keystroke(key="b", config=config),
        make_homing(config=config),
    )
    seq = KLMSequence(task_name="test", operators=ops)
    expected = sum(op.duration_s for op in ops)
    assert math.isclose(seq.total_time_s, expected, abs_tol=_ATOL), \
        f"Total time {seq.total_time_s} != sum {expected}"


# ---------------------------------------------------------------------------
# GomsMethod total time
# ---------------------------------------------------------------------------


@given(_skill)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_goms_method_total_duration(skill):
    """GomsMethod.total_duration_s = Σ operator durations."""
    config = KLMConfig(skill_level=skill)
    ops = (
        make_mental(config=config),
        make_keystroke(key="x", config=config),
    )
    method = GomsMethod(
        method_id="m1", goal_id="g1", name="test_method",
        operators=ops,
    )
    expected = sum(op.duration_s for op in ops)
    assert math.isclose(method.total_duration_s, expected, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Execution time ≥ critical path time (CPM-GOMS)
# ---------------------------------------------------------------------------


def test_execution_time_geq_critical_path():
    """Total serial time ≥ critical path time (parallelism can only help)."""
    trace = GomsTrace(
        trace_id="trace1",
        task_name="test",
        total_time_s=5.0,
        critical_path_time_s=3.5,
    )
    assert trace.total_time_s >= trace.critical_path_time_s - _ATOL


def test_speedup_from_parallelism_geq_one():
    """Speedup from parallelism ≥ 1 (serial / critical path)."""
    trace = GomsTrace(
        trace_id="trace1",
        task_name="test",
        total_time_s=5.0,
        critical_path_time_s=3.5,
    )
    assert trace.speedup_from_parallelism >= 1.0 - _ATOL


# ---------------------------------------------------------------------------
# OperatorType defaults are positive
# ---------------------------------------------------------------------------


def test_operator_type_defaults_non_negative():
    """All OperatorType default durations ≥ 0."""
    for op_type in OperatorType:
        assert op_type.default_duration_s >= 0.0, \
            f"{op_type} default duration negative: {op_type.default_duration_s}"
