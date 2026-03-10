"""Unit tests for usability_oracle.cognitive.fitts.FittsLaw.

Tests cover point-estimate prediction, interval-valued prediction,
derived quantities (index of difficulty, throughput, effective width,
crossing time), batch/vectorised prediction, error-rate estimation,
edge cases, monotonicity properties, and default constant values.

References
----------
Fitts, P. M. (1954). The information capacity of the human motor system
    in controlling the amplitude of movement. *J Exp Psychol*, 47(6), 381-391.
MacKenzie, I. S. (1992). Fitts' law as a research and design tool in HCI.
    *Human-Computer Interaction*, 7(1), 91-139.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.interval.interval import Interval


# ------------------------------------------------------------------ #
# Default constants
# ------------------------------------------------------------------ #


class TestFittsLawDefaults:
    """Verify the published default parameter values."""

    def test_default_a_value(self) -> None:
        """DEFAULT_A should be 0.050 s (Card, Moran & Newell, 1983)."""
        assert FittsLaw.DEFAULT_A == pytest.approx(0.050)

    def test_default_b_value(self) -> None:
        """DEFAULT_B should be 0.150 s/bit (Card, Moran & Newell, 1983)."""
        assert FittsLaw.DEFAULT_B == pytest.approx(0.150)


# ------------------------------------------------------------------ #
# Core prediction — point estimates
# ------------------------------------------------------------------ #


class TestFittsPredict:
    """Tests for FittsLaw.predict() using the Shannon formulation."""

    def test_basic_calculation(self) -> None:
        """predict(distance, width) should return a + b * log2(1 + D/W).

        With D=200, W=40, defaults a=0.05, b=0.15:
          MT = 0.05 + 0.15 * log2(1 + 200/40) = 0.05 + 0.15 * log2(6)
        """
        expected = 0.05 + 0.15 * math.log2(1.0 + 200.0 / 40.0)
        result = FittsLaw.predict(distance=200, width=40)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_custom_a_b_parameters(self) -> None:
        """predict() should accept custom a and b parameters.

        With a=0.1, b=0.2, D=100, W=25:
          MT = 0.1 + 0.2 * log2(1 + 100/25) = 0.1 + 0.2 * log2(5)
        """
        expected = 0.1 + 0.2 * math.log2(1.0 + 100.0 / 25.0)
        result = FittsLaw.predict(distance=100, width=25, a=0.1, b=0.2)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_equal_distance_and_width(self) -> None:
        """When D == W, ID = log2(2) = 1 bit, so MT = a + b."""
        result = FittsLaw.predict(distance=50, width=50)
        expected = FittsLaw.DEFAULT_A + FittsLaw.DEFAULT_B * 1.0
        assert result == pytest.approx(expected, rel=1e-9)

    def test_predict_raises_on_zero_distance(self) -> None:
        """predict() must raise ValueError when distance <= 0."""
        with pytest.raises(ValueError, match="distance must be > 0"):
            FittsLaw.predict(distance=0, width=40)

    def test_predict_raises_on_negative_width(self) -> None:
        """predict() must raise ValueError when width < 0."""
        with pytest.raises(ValueError, match="width must be > 0"):
            FittsLaw.predict(distance=100, width=-5)

    def test_predict_returns_float(self) -> None:
        """predict() should always return a Python float."""
        result = FittsLaw.predict(distance=300, width=50)
        assert isinstance(result, float)


# ------------------------------------------------------------------ #
# Index of difficulty
# ------------------------------------------------------------------ #


class TestIndexOfDifficulty:
    """Tests for FittsLaw.index_of_difficulty()."""

    def test_basic_id(self) -> None:
        """ID = log2(1 + D/W).  D=200, W=40 → log2(6) ≈ 2.585 bits."""
        expected = math.log2(1.0 + 200.0 / 40.0)
        result = FittsLaw.index_of_difficulty(200, 40)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_id_raises_on_invalid_inputs(self) -> None:
        """index_of_difficulty() must reject non-positive distance or width."""
        with pytest.raises(ValueError):
            FittsLaw.index_of_difficulty(-10, 40)
        with pytest.raises(ValueError):
            FittsLaw.index_of_difficulty(100, 0)


# ------------------------------------------------------------------ #
# Throughput
# ------------------------------------------------------------------ #


class TestThroughput:
    """Tests for FittsLaw.throughput()."""

    def test_throughput_basic(self) -> None:
        """TP = ID / MT.  For D=200, W=40, MT=0.5:
        TP = log2(6) / 0.5 ≈ 5.17 bits/s."""
        expected = math.log2(1.0 + 200.0 / 40.0) / 0.5
        result = FittsLaw.throughput(200, 40, 0.5)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_throughput_raises_on_zero_mt(self) -> None:
        """throughput() must raise ValueError when movement_time <= 0."""
        with pytest.raises(ValueError, match="movement_time must be > 0"):
            FittsLaw.throughput(200, 40, 0.0)

    def test_throughput_positive_for_valid_inputs(self) -> None:
        """Throughput should always be positive for valid inputs."""
        tp = FittsLaw.throughput(100, 20, 0.3)
        assert tp > 0.0


# ------------------------------------------------------------------ #
# Crossing time (steering law)
# ------------------------------------------------------------------ #


class TestCrossingTime:
    """Tests for FittsLaw.crossing_time() (Accot & Zhai steering law)."""

    def test_crossing_time_basic(self) -> None:
        """T = a + b * (A/W).  A=500, W=50, defaults → 0.05 + 0.15*10 = 1.55."""
        expected = 0.05 + 0.15 * (500.0 / 50.0)
        result = FittsLaw.crossing_time(amplitude=500, tolerance=50)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_crossing_time_raises_on_zero_amplitude(self) -> None:
        """crossing_time() must raise ValueError for non-positive amplitude."""
        with pytest.raises(ValueError, match="amplitude must be > 0"):
            FittsLaw.crossing_time(amplitude=0, tolerance=10)

    def test_crossing_time_raises_on_zero_tolerance(self) -> None:
        """crossing_time() must raise ValueError for non-positive tolerance."""
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            FittsLaw.crossing_time(amplitude=100, tolerance=0)


# ------------------------------------------------------------------ #
# Batch prediction
# ------------------------------------------------------------------ #


class TestPredictBatch:
    """Tests for FittsLaw.predict_batch() with numpy arrays."""

    def test_batch_matches_scalar(self) -> None:
        """Vectorised results must match element-wise scalar predictions."""
        distances = np.array([100.0, 200.0, 300.0])
        widths = np.array([20.0, 40.0, 60.0])
        batch_result = FittsLaw.predict_batch(distances, widths)

        for i in range(len(distances)):
            scalar = FittsLaw.predict(distances[i], widths[i])
            assert batch_result[i] == pytest.approx(scalar, rel=1e-9)

    def test_batch_returns_ndarray(self) -> None:
        """predict_batch() should return a numpy ndarray."""
        result = FittsLaw.predict_batch(
            np.array([100.0]), np.array([20.0])
        )
        assert isinstance(result, np.ndarray)

    def test_batch_raises_on_nonpositive(self) -> None:
        """predict_batch() must reject arrays with non-positive elements."""
        with pytest.raises(ValueError, match="distances must be > 0"):
            FittsLaw.predict_batch(np.array([0.0, 100.0]), np.array([20.0, 20.0]))


# ------------------------------------------------------------------ #
# Error rate
# ------------------------------------------------------------------ #


class TestErrorRate:
    """Tests for FittsLaw.error_rate()."""

    def test_error_rate_in_unit_interval(self) -> None:
        """Error rate must lie in [0, 1] for valid inputs."""
        rate = FittsLaw.error_rate(200, 40)
        assert 0.0 <= rate <= 1.0

    def test_error_rate_decreases_with_wider_target(self) -> None:
        """Wider actual target → lower miss probability."""
        narrow = FittsLaw.error_rate(200, 40, actual_width=20)
        wide = FittsLaw.error_rate(200, 40, actual_width=80)
        assert wide < narrow

    def test_error_rate_raises_on_invalid(self) -> None:
        """error_rate() must reject non-positive distance or width."""
        with pytest.raises(ValueError):
            FittsLaw.error_rate(0, 40)
        with pytest.raises(ValueError):
            FittsLaw.error_rate(100, 0)


# ------------------------------------------------------------------ #
# Interval-valued prediction
# ------------------------------------------------------------------ #


class TestPredictInterval:
    """Tests for FittsLaw.predict_interval() with Interval inputs."""

    def test_interval_encloses_point(self) -> None:
        """Interval prediction must enclose the point-estimate prediction.

        Using degenerate (point) intervals for all parameters should yield
        a degenerate result matching the scalar prediction.
        """
        d, w = 200.0, 40.0
        point = FittsLaw.predict(d, w)
        ivl = FittsLaw.predict_interval(
            distance=Interval.from_value(d),
            width=Interval.from_value(w),
            a=Interval.from_value(FittsLaw.DEFAULT_A),
            b=Interval.from_value(FittsLaw.DEFAULT_B),
        )
        assert ivl.low == pytest.approx(point, rel=1e-6)
        assert ivl.high == pytest.approx(point, rel=1e-6)

    def test_interval_widens_with_uncertainty(self) -> None:
        """Wider input intervals should produce a wider output interval."""
        narrow = FittsLaw.predict_interval(
            Interval(190, 210), Interval(38, 42),
            Interval(0.049, 0.051), Interval(0.149, 0.151),
        )
        wide = FittsLaw.predict_interval(
            Interval(150, 250), Interval(30, 50),
            Interval(0.03, 0.07), Interval(0.12, 0.18),
        )
        assert wide.width > narrow.width

    def test_interval_returns_interval_type(self) -> None:
        """predict_interval() must return an Interval object."""
        result = FittsLaw.predict_interval(
            Interval(100, 200), Interval(20, 40),
            Interval(0.04, 0.06), Interval(0.14, 0.16),
        )
        assert isinstance(result, Interval)


# ------------------------------------------------------------------ #
# Monotonicity properties
# ------------------------------------------------------------------ #


class TestMonotonicity:
    """Monotonicity properties required by Fitts' law."""

    def test_increasing_distance_increases_time(self) -> None:
        """For fixed width, increasing distance must increase predicted MT.

        Fitts' law is monotonically increasing in distance for any fixed
        positive width, because log2(1 + D/W) is increasing in D.
        """
        w = 40
        times = [FittsLaw.predict(d, w) for d in [50, 100, 200, 400, 800]]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1], (
                f"MT should increase with distance: {times}"
            )

    def test_decreasing_width_increases_time(self) -> None:
        """For fixed distance, decreasing width must increase predicted MT.

        Fitts' law is monotonically decreasing in width for any fixed
        positive distance, because log2(1 + D/W) is decreasing in W.
        """
        d = 200
        times = [FittsLaw.predict(d, w) for w in [80, 40, 20, 10, 5]]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1], (
                f"MT should increase as width decreases: {times}"
            )


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Edge case behaviour for extreme parameter values."""

    def test_very_small_width(self) -> None:
        """A very small target width should yield a large movement time.

        With W → 0⁺ the index of difficulty → ∞, so MT → ∞.  We verify
        that a width of 0.1 px produces MT > 1 second.
        """
        mt = FittsLaw.predict(distance=500, width=0.1)
        assert mt > 1.0

    def test_very_large_distance(self) -> None:
        """A very large distance should yield a large movement time.

        D=100000, W=40 → ID ≈ 11.3 bits → MT ≈ 0.05 + 0.15*11.3 ≈ 1.74 s.
        """
        mt = FittsLaw.predict(distance=100_000, width=40)
        assert mt > 1.5

    def test_very_large_width_low_time(self) -> None:
        """A very large width with small distance gives near-minimum MT.

        D=10, W=10000 → ID ≈ log2(1.001) ≈ 0.0014 → MT ≈ a.
        """
        mt = FittsLaw.predict(distance=10, width=10_000)
        assert mt == pytest.approx(FittsLaw.DEFAULT_A, abs=0.01)

    def test_batch_with_single_element(self) -> None:
        """predict_batch() should work with single-element arrays."""
        result = FittsLaw.predict_batch(np.array([100.0]), np.array([20.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(FittsLaw.predict(100, 20), rel=1e-9)

    def test_crossing_time_narrow_tunnel(self) -> None:
        """A very narrow steering tunnel should produce a long traversal time."""
        t = FittsLaw.crossing_time(amplitude=1000, tolerance=1)
        assert t > 100.0
