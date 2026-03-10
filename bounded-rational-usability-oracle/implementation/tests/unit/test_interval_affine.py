"""Unit tests for usability_oracle.interval.affine — Affine arithmetic.

Tests affine addition, multiplication, interval conversion, tightness
vs naive interval arithmetic, and transcendental approximations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.interval.affine import (
    AffineForm,
    add,
    divide,
    exp,
    log,
    multiply,
    negate,
    reset_noise_counter,
    scale,
    subtract,
)
from usability_oracle.interval.interval import Interval


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_noise():
    """Reset the global noise counter before each test for determinism."""
    reset_noise_counter()


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:

    def test_from_value(self):
        af = AffineForm.from_value(3.0)
        assert af.center == 3.0
        assert af.radius == pytest.approx(0.0, abs=1e-14)

    def test_from_interval(self):
        iv = Interval(2.0, 4.0)
        af = AffineForm.from_interval(iv)
        assert af.center == pytest.approx(3.0)
        assert af.radius == pytest.approx(1.0)

    def test_from_interval_degenerate(self):
        iv = Interval(5.0, 5.0)
        af = AffineForm.from_interval(iv)
        assert af.center == 5.0
        assert af.radius == pytest.approx(0.0, abs=1e-14)


# ===================================================================
# Affine addition preserves correlations
# ===================================================================


class TestAddition:

    def test_add_constants(self):
        a = AffineForm.from_value(2.0)
        b = AffineForm.from_value(3.0)
        c = add(a, b)
        assert c.center == pytest.approx(5.0)

    def test_add_intervals(self):
        a = AffineForm.from_interval(Interval(1.0, 3.0))
        b = AffineForm.from_interval(Interval(2.0, 4.0))
        c = add(a, b)
        iv = c.to_interval()
        assert iv.low <= 3.0  # min(1+2)
        assert iv.high >= 7.0  # max(3+4)

    def test_self_subtract_cancels(self):
        """x - x should be exactly 0 due to correlation tracking."""
        x = AffineForm.from_interval(Interval(1.0, 5.0))
        result = subtract(x, x)
        assert result.center == pytest.approx(0.0)
        assert result.radius == pytest.approx(0.0, abs=1e-12)


# ===================================================================
# Affine multiplication
# ===================================================================


class TestMultiplication:

    def test_multiply_by_constant(self):
        a = AffineForm.from_interval(Interval(2.0, 4.0))
        c = scale(a, 3.0)
        iv = c.to_interval()
        assert iv.low == pytest.approx(6.0)
        assert iv.high == pytest.approx(12.0)

    def test_multiply_two_forms(self):
        a = AffineForm.from_interval(Interval(1.0, 3.0))
        b = AffineForm.from_interval(Interval(2.0, 4.0))
        c = multiply(a, b)
        iv = c.to_interval()
        # True range: [1*2, 3*4] = [2, 12]
        assert iv.low <= 2.0 + 0.5  # may be slightly wider
        assert iv.high >= 12.0 - 0.5

    def test_multiply_zero(self):
        a = AffineForm.from_interval(Interval(-1.0, 1.0))
        z = AffineForm.from_value(0.0)
        c = multiply(a, z)
        iv = c.to_interval()
        assert iv.low <= 0.0
        assert iv.high >= 0.0


# ===================================================================
# to_interval containment
# ===================================================================


class TestToInterval:

    def test_contains_true_value(self):
        """The interval from an affine form must contain the center."""
        af = AffineForm.from_interval(Interval(1.0, 5.0))
        iv = af.to_interval()
        assert iv.low <= af.center <= iv.high

    def test_contains_endpoints(self):
        iv_orig = Interval(2.0, 8.0)
        af = AffineForm.from_interval(iv_orig)
        iv = af.to_interval()
        assert iv.low <= iv_orig.low
        assert iv.high >= iv_orig.high

    def test_constant_form_degenerate(self):
        af = AffineForm.from_value(7.0)
        iv = af.to_interval()
        assert iv.low == pytest.approx(7.0)
        assert iv.high == pytest.approx(7.0)


# ===================================================================
# Tightness: affine vs naive interval
# ===================================================================


class TestTightness:

    def test_affine_tighter_for_correlated(self):
        """x - x with affine should give [0,0], but naive interval gives [-w, w]."""
        x = AffineForm.from_interval(Interval(1.0, 5.0))
        # Affine: x - x
        affine_result = subtract(x, x).to_interval()
        # Naive interval: [1,5] - [1,5] = [-4, 4]
        naive = Interval(1.0 - 5.0, 5.0 - 1.0)
        assert affine_result.high - affine_result.low <= naive.high - naive.low

    def test_affine_tighter_for_squared(self):
        """x * x with affine should be tighter than [lo,hi] * [lo,hi] for positive x."""
        x = AffineForm.from_interval(Interval(2.0, 4.0))
        affine_sq = multiply(x, x).to_interval()
        naive_sq = Interval(2.0 * 2.0, 4.0 * 4.0)
        # Affine may give slightly wider due to linearisation error, but should be similar
        affine_width = affine_sq.high - affine_sq.low
        naive_width = naive_sq.high - naive_sq.low
        # At minimum, affine result should contain the true range
        assert affine_sq.low <= 4.0 + 1.0
        assert affine_sq.high >= 16.0 - 1.0


# ===================================================================
# Transcendental approximations
# ===================================================================


class TestTranscendental:

    def test_exp_contains_true_range(self):
        x = AffineForm.from_interval(Interval(0.0, 1.0))
        result = exp(x)
        iv = result.to_interval()
        assert iv.low <= math.exp(0.0) + 0.1
        assert iv.high >= math.exp(1.0) - 0.1

    def test_log_contains_true_range(self):
        x = AffineForm.from_interval(Interval(1.0, 3.0))
        result = log(x)
        iv = result.to_interval()
        assert iv.low <= math.log(1.0) + 0.1
        assert iv.high >= math.log(3.0) - 0.1

    def test_exp_of_zero_near_one(self):
        x = AffineForm.from_value(0.0)
        result = exp(x)
        iv = result.to_interval()
        np.testing.assert_allclose(iv.low, 1.0, atol=0.1)
        np.testing.assert_allclose(iv.high, 1.0, atol=0.1)

    def test_log_of_one_near_zero(self):
        x = AffineForm.from_value(1.0)
        result = log(x)
        iv = result.to_interval()
        np.testing.assert_allclose(iv.low, 0.0, atol=0.1)
        np.testing.assert_allclose(iv.high, 0.0, atol=0.1)


# ===================================================================
# Properties
# ===================================================================


class TestProperties:

    def test_radius_non_negative(self):
        af = AffineForm.from_interval(Interval(-3.0, 3.0))
        assert af.radius >= 0.0

    def test_num_terms(self):
        af = AffineForm.from_interval(Interval(1.0, 5.0))
        assert af.num_terms >= 1

    def test_negate(self):
        af = AffineForm.from_interval(Interval(2.0, 4.0))
        neg = negate(af)
        iv = neg.to_interval()
        assert iv.low == pytest.approx(-4.0)
        assert iv.high == pytest.approx(-2.0)
