"""Property-based tests for the WCAG module.

Verifies every violation has a valid criterion ID, conformance level
ordering A ⊂ AA ⊂ AAA, contrast ratio ≥ 1.0, and no false positives
on empty or trivial accessibility trees.
"""

import math
import re

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    composite,
    tuples,
)

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)
from usability_oracle.wcag.contrast import (
    Color,
    ContrastResult,
    check_contrast,
    contrast_ratio,
    relative_luminance,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ATOL = 1e-6

_level = sampled_from(list(ConformanceLevel))

_rgb = integers(min_value=0, max_value=255)


@composite
def _color(draw):
    """Generate a valid sRGB Color."""
    return Color(r=draw(_rgb), g=draw(_rgb), b=draw(_rgb))


@composite
def _success_criterion(draw):
    """Generate a plausible SuccessCriterion."""
    principle = draw(sampled_from(list(WCAGPrinciple)))
    level = draw(_level)
    p_num = principle.number
    gl_num = draw(integers(min_value=1, max_value=5))
    sc_num = draw(integers(min_value=1, max_value=10))
    sc_id = f"{p_num}.{gl_num}.{sc_num}"
    gl_id = f"{p_num}.{gl_num}"
    return SuccessCriterion(
        sc_id=sc_id,
        name=f"Test Criterion {sc_id}",
        level=level,
        principle=principle,
        guideline_id=gl_id,
    )


@composite
def _violation(draw):
    """Generate a WCAGViolation."""
    sc = draw(_success_criterion())
    impact = draw(sampled_from(list(ImpactLevel)))
    return WCAGViolation(
        criterion=sc,
        node_id=f"node_{draw(integers(min_value=1, max_value=100))}",
        impact=impact,
        message="Test violation",
    )


# ---------------------------------------------------------------------------
# Every violation has a valid criterion ID
# ---------------------------------------------------------------------------


@given(_violation())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_violation_has_valid_criterion_id(v):
    """Every violation's criterion has a dotted ID pattern (e.g. 1.4.3)."""
    assert re.match(r"^\d+\.\d+\.\d+$", v.sc_id), \
        f"Invalid criterion ID format: {v.sc_id}"


@given(_violation())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_violation_criterion_has_principle(v):
    """Every violation's criterion belongs to a WCAG principle."""
    assert v.criterion.principle in list(WCAGPrinciple), \
        f"Invalid principle: {v.criterion.principle}"


@given(_violation())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_violation_criterion_has_level(v):
    """Every violation's criterion has a conformance level."""
    assert v.criterion.level in list(ConformanceLevel), \
        f"Invalid level: {v.criterion.level}"


@given(_violation())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_violation_has_node_id(v):
    """Every violation references an element node."""
    assert len(v.node_id) > 0, "Violation has empty node_id"


# ---------------------------------------------------------------------------
# Conformance level ordering: A < AA < AAA
# ---------------------------------------------------------------------------


def test_conformance_level_a_lt_aa():
    """Level A < Level AA."""
    assert ConformanceLevel.A < ConformanceLevel.AA


def test_conformance_level_aa_lt_aaa():
    """Level AA < Level AAA."""
    assert ConformanceLevel.AA < ConformanceLevel.AAA


def test_conformance_level_a_lt_aaa():
    """Level A < Level AAA."""
    assert ConformanceLevel.A < ConformanceLevel.AAA


def test_conformance_level_a_le_a():
    """Level A ≤ Level A."""
    assert ConformanceLevel.A <= ConformanceLevel.A


def test_conformance_level_numeric_ordering():
    """Numeric values are strictly ordered: A=1 < AA=2 < AAA=3."""
    assert ConformanceLevel.A.numeric == 1
    assert ConformanceLevel.AA.numeric == 2
    assert ConformanceLevel.AAA.numeric == 3


# ---------------------------------------------------------------------------
# Guideline criteria_at_level reflects subset ordering
# ---------------------------------------------------------------------------


def test_criteria_at_level_subset_ordering():
    """criteria_at_level(A) ⊆ criteria_at_level(AA) ⊆ criteria_at_level(AAA)."""
    sc_a = SuccessCriterion(sc_id="1.1.1", name="A1", level=ConformanceLevel.A,
                            principle=WCAGPrinciple.PERCEIVABLE, guideline_id="1.1")
    sc_aa = SuccessCriterion(sc_id="1.4.3", name="AA1", level=ConformanceLevel.AA,
                             principle=WCAGPrinciple.PERCEIVABLE, guideline_id="1.4")
    sc_aaa = SuccessCriterion(sc_id="1.4.6", name="AAA1", level=ConformanceLevel.AAA,
                              principle=WCAGPrinciple.PERCEIVABLE, guideline_id="1.4")

    gl = WCAGGuideline(
        guideline_id="1.4", name="Distinguishable",
        principle=WCAGPrinciple.PERCEIVABLE,
        criteria=(sc_a, sc_aa, sc_aaa),
    )

    at_a = set(sc.sc_id for sc in gl.criteria_at_level(ConformanceLevel.A))
    at_aa = set(sc.sc_id for sc in gl.criteria_at_level(ConformanceLevel.AA))
    at_aaa = set(sc.sc_id for sc in gl.criteria_at_level(ConformanceLevel.AAA))

    assert at_a <= at_aa, f"A criteria not subset of AA: {at_a} vs {at_aa}"
    assert at_aa <= at_aaa, f"AA criteria not subset of AAA: {at_aa} vs {at_aaa}"


# ---------------------------------------------------------------------------
# Contrast ratio ≥ 1.0
# ---------------------------------------------------------------------------


@given(_color(), _color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_contrast_ratio_geq_one(fg, bg):
    """Contrast ratio ≥ 1.0 for any two colours."""
    cr = contrast_ratio(fg, bg)
    assert cr >= 1.0 - _ATOL, f"Contrast ratio < 1: {cr}"


@given(_color(), _color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_contrast_ratio_at_most_21(fg, bg):
    """Contrast ratio ≤ 21:1 (black vs white maximum)."""
    cr = contrast_ratio(fg, bg)
    assert cr <= 21.0 + _ATOL, f"Contrast ratio > 21: {cr}"


@given(_color(), _color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_contrast_ratio_symmetric(fg, bg):
    """Contrast ratio is symmetric: CR(a,b) = CR(b,a)."""
    cr_ab = contrast_ratio(fg, bg)
    cr_ba = contrast_ratio(bg, fg)
    assert math.isclose(cr_ab, cr_ba, abs_tol=_ATOL), \
        f"Contrast ratio not symmetric: {cr_ab} vs {cr_ba}"


@given(_color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_contrast_ratio_self_is_one(c):
    """Contrast of a colour with itself is 1.0."""
    cr = contrast_ratio(c, c)
    assert math.isclose(cr, 1.0, abs_tol=_ATOL), \
        f"Self-contrast = {cr} != 1.0"


def test_contrast_black_white():
    """Black on white has contrast ratio = 21:1."""
    black = Color(0, 0, 0)
    white = Color(255, 255, 255)
    cr = contrast_ratio(black, white)
    assert math.isclose(cr, 21.0, rel_tol=0.01), \
        f"Black/white contrast = {cr} != 21.0"


# ---------------------------------------------------------------------------
# Relative luminance ∈ [0, 1]
# ---------------------------------------------------------------------------


@given(_color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_relative_luminance_in_unit_interval(c):
    """Relative luminance L ∈ [0, 1]."""
    lum = relative_luminance(c)
    assert -_ATOL <= lum <= 1.0 + _ATOL, \
        f"Luminance out of range: {lum}"


def test_black_luminance_zero():
    """Black has luminance 0."""
    lum = relative_luminance(Color(0, 0, 0))
    assert math.isclose(lum, 0.0, abs_tol=_ATOL)


def test_white_luminance_one():
    """White has luminance 1."""
    lum = relative_luminance(Color(255, 255, 255))
    assert math.isclose(lum, 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# No false positives on empty WCAGResult
# ---------------------------------------------------------------------------


def test_empty_result_is_conformant():
    """A WCAGResult with no violations is conformant."""
    result = WCAGResult(
        violations=(),
        target_level=ConformanceLevel.AA,
        criteria_tested=10,
        criteria_passed=10,
    )
    assert result.is_conformant
    assert result.violation_count == 0
    assert result.conformance_ratio == 1.0


def test_empty_result_zero_violations_count():
    """criteria_failed = 0 when all pass."""
    result = WCAGResult(
        violations=(),
        target_level=ConformanceLevel.AA,
        criteria_tested=5,
        criteria_passed=5,
    )
    assert result.criteria_failed == 0


# ---------------------------------------------------------------------------
# WCAGResult conformance_ratio ∈ [0, 1]
# ---------------------------------------------------------------------------


@given(integers(min_value=1, max_value=100),
       integers(min_value=0, max_value=100))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_conformance_ratio_in_unit(tested, passed):
    """conformance_ratio ∈ [0, 1]."""
    assume(passed <= tested)
    result = WCAGResult(
        violations=(),
        target_level=ConformanceLevel.AA,
        criteria_tested=tested,
        criteria_passed=passed,
    )
    r = result.conformance_ratio
    assert 0.0 - _ATOL <= r <= 1.0 + _ATOL, \
        f"Conformance ratio out of range: {r}"


# ---------------------------------------------------------------------------
# Color parsing round-trip
# ---------------------------------------------------------------------------


@given(_color())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_color_hex_round_trip(c):
    """Color → hex → Color round-trips."""
    hex_str = c.to_hex()
    c2 = Color.from_hex(hex_str)
    assert c.r == c2.r and c.g == c2.g and c.b == c2.b, \
        f"Hex round-trip failed: {c} → {hex_str} → {c2}"


# ---------------------------------------------------------------------------
# SuccessCriterion serialization round-trip
# ---------------------------------------------------------------------------


@given(_success_criterion())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_success_criterion_round_trip(sc):
    """SuccessCriterion.to_dict/from_dict round-trips."""
    d = sc.to_dict()
    sc2 = SuccessCriterion.from_dict(d)
    assert sc.sc_id == sc2.sc_id
    assert sc.level == sc2.level
    assert sc.principle == sc2.principle
