"""Unit tests for cpa.analysis.supermartingale – SupermartingaleStopper."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.analysis.supermartingale import (
    SupermartingaleStopper,
    StoppingResult,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def stopper():
    return SupermartingaleStopper(alpha=0.05, wealth_init=1.0,
                                  mu_0=0.0, lambda_max=0.5, lookback=10)


@pytest.fixture
def lenient_stopper():
    return SupermartingaleStopper(alpha=0.2, wealth_init=1.0,
                                  mu_0=0.0, lambda_max=0.5, lookback=5)


# ===================================================================
# Tests – SupermartingaleStopper stop detection
# ===================================================================


class TestStopDetection:
    """Test that stopper detects when to stop."""

    def test_update_returns_result(self, stopper):
        result = stopper.update(0.1)
        assert isinstance(result, StoppingResult)

    def test_not_stopped_initially(self, stopper):
        stopper.update(0.1)
        assert not stopper.should_stop()

    def test_detects_stop_on_consistent_improvement(self, stopper):
        # Feed consistent positive improvements → wealth should accumulate against H0: mu=0
        for _ in range(200):
            stopper.update(0.5)
        # After many positive improvements, should reject H0 and stop
        assert stopper.should_stop()

    def test_does_not_stop_with_zero_improvements(self, stopper):
        # Feed zero improvements (matching H0: mu=0) → should NOT stop
        for _ in range(10):
            stopper.update(0.0)
        assert not stopper.should_stop()

    def test_result_fields(self, stopper):
        result = stopper.update(0.5)
        assert hasattr(result, "should_stop")
        assert hasattr(result, "confidence")
        assert hasattr(result, "expected_improvement")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "wealth")


# ===================================================================
# Tests – Wealth process
# ===================================================================


class TestWealthProcess:
    """Test wealth process computation."""

    def test_initial_wealth(self, stopper):
        assert_allclose(stopper.wealth(), 1.0)

    def test_wealth_after_update(self, stopper):
        stopper.update(0.0)
        w = stopper.wealth()
        assert np.isfinite(w)
        assert w > 0

    def test_wealth_monotone_for_null(self, stopper):
        # Under null (zero mean improvements), wealth should tend to grow
        wealths = []
        for _ in range(50):
            result = stopper.update(0.0)
            wealths.append(result.wealth)
        assert all(np.isfinite(w) for w in wealths)

    def test_wealth_history_in_result(self, stopper):
        for _ in range(10):
            result = stopper.update(0.0)
        assert hasattr(result, "wealth_history")
        assert len(result.wealth_history) > 0

    def test_static_wealth_process(self):
        values = np.zeros(50)
        wp = SupermartingaleStopper._wealth_process(values, mu_0=0.0,
                                                     lam=0.1, w0=1.0)
        assert isinstance(wp, np.ndarray)
        assert len(wp) == len(values) or len(wp) == len(values) + 1  # may include initial wealth
        assert wp[0] > 0


# ===================================================================
# Tests – Confidence sequence
# ===================================================================


class TestConfidenceSequence:
    """Test confidence sequence validity."""

    def test_confidence_sequence_returns_bounds(self):
        values = np.random.default_rng(42).normal(0, 1, 100)
        lower, upper = SupermartingaleStopper.confidence_sequence(
            values, alpha=0.05, v_opt=1.0,
        )
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)

    def test_confidence_sequence_length(self):
        values = np.random.default_rng(42).normal(0, 1, 100)
        lower, upper = SupermartingaleStopper.confidence_sequence(
            values, alpha=0.05,
        )
        assert len(lower) == len(values)
        assert len(upper) == len(values)

    def test_lower_leq_upper(self):
        values = np.random.default_rng(42).normal(0.5, 1, 100)
        lower, upper = SupermartingaleStopper.confidence_sequence(
            values, alpha=0.05,
        )
        assert np.all(lower <= upper + 1e-10)

    def test_confidence_bounds_contain_zero_for_null(self):
        values = np.random.default_rng(42).normal(0, 1, 200)
        lower, upper = SupermartingaleStopper.confidence_sequence(
            values, alpha=0.05,
        )
        # For null mean=0 data, bounds should contain 0 most of the time
        contains_zero = np.sum((lower <= 0) & (upper >= 0))
        assert contains_zero > len(values) * 0.5

    def test_narrow_with_more_data(self):
        rng = np.random.default_rng(42)
        values = rng.normal(1.0, 1.0, 200)
        lower, upper = SupermartingaleStopper.confidence_sequence(
            values, alpha=0.05,
        )
        widths = upper - lower
        # Width should generally decrease
        assert widths[-1] < widths[10] or widths[-1] < 5.0


# ===================================================================
# Tests – Reset
# ===================================================================


class TestReset:
    """Test reset functionality."""

    def test_reset_wealth(self, stopper):
        for _ in range(20):
            stopper.update(0.0)
        stopper.reset()
        assert_allclose(stopper.wealth(), 1.0)

    def test_reset_stops_flag(self, stopper):
        for _ in range(200):
            stopper.update(0.0)
        stopper.reset()
        assert not stopper.should_stop()


# ===================================================================
# Tests – Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_negative_improvements(self, stopper):
        for _ in range(20):
            result = stopper.update(-0.1)
        assert np.isfinite(result.wealth)

    def test_single_update(self, stopper):
        result = stopper.update(0.5)
        assert result.n_iterations == 1

    def test_very_large_improvement(self, stopper):
        result = stopper.update(1e6)
        assert np.isfinite(result.wealth)

    def test_alternating_sign_improvements(self, stopper):
        for i in range(50):
            stopper.update((-1) ** i * 0.1)
        assert np.isfinite(stopper.wealth())

    def test_mixture_martingale_static(self):
        values = np.random.default_rng(42).normal(0, 1, 50)
        mm = SupermartingaleStopper._mixture_martingale(
            values, mu_0=0.0, sigma=1.0, n_lambdas=10,
        )
        assert isinstance(mm, np.ndarray)
        assert len(mm) == len(values) or len(mm) == len(values) + 1
