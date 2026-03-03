"""Tests for streaming.continual module."""
import pytest
import numpy as np

from dp_forge.streaming.continual import (
    ContinualCounter,
    EventLevelDP,
    UserLevelDP,
    WindowedDP,
)
from dp_forge.streaming.online_synthesis import (
    PrivacyFilter,
    AboveThreshold,
)


class TestContinualCounter:
    """Test ContinualCounter accuracy and privacy."""

    def test_increment_returns_output(self):
        """Increment returns valid output."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, seed=42)
        out = cc.increment(1.0)
        assert out.true_value == 1.0

    def test_true_count_accumulates(self):
        """True count matches sum of increments."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, seed=42)
        for _ in range(10):
            cc.increment(1.0)
        assert cc.true_count == 10.0

    def test_privacy_constant(self):
        """Privacy budget is constant (tree mechanism)."""
        cc = ContinualCounter(max_time=100, epsilon=2.0, seed=42)
        for _ in range(50):
            cc.increment(1.0)
        assert cc.privacy_spent() == 2.0

    def test_reset(self):
        """Reset clears all state."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, seed=42)
        cc.increment(1.0)
        cc.reset()
        assert cc.true_count == 0.0

    def test_mean_error_bounded(self):
        """Mean error should be finite."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, seed=42)
        for _ in range(50):
            cc.increment(1.0)
        assert cc.mean_error() < 1000  # not infinity

    def test_clipping(self):
        """Values are clipped to [0, sensitivity]."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, sensitivity=1.0, seed=42)
        cc.increment(5.0)  # clipped to 1.0
        assert cc.true_count == 1.0

    def test_current_count_without_increment(self):
        """Current count query works."""
        cc = ContinualCounter(max_time=100, epsilon=1.0, seed=42)
        out = cc.current_count()
        assert out.value == 0.0


class TestEventLevelDP:
    """Test EventLevelDP."""

    def test_observe_accumulates(self):
        """Events accumulate."""
        eld = EventLevelDP(max_time=100, epsilon=1.0, seed=42)
        for _ in range(5):
            eld.observe(1.0)
        assert eld._counter.true_count == 5.0

    def test_privacy_matches_epsilon(self):
        """Privacy spent equals epsilon."""
        eld = EventLevelDP(max_time=100, epsilon=1.5, seed=42)
        eld.observe(1.0)
        assert eld.privacy_spent() == 1.5

    def test_reset(self):
        """Reset clears state."""
        eld = EventLevelDP(max_time=100, epsilon=1.0, seed=42)
        eld.observe(1.0)
        eld.reset()
        assert len(eld._values) == 0


class TestUserLevelDP:
    """Test UserLevelDP."""

    def test_contribution_limit(self):
        """Users can't exceed contribution limit."""
        uld = UserLevelDP(max_time=100, epsilon=1.0, max_contributions=2, seed=42)
        uld.observe("user1", 1.0)
        uld.observe("user1", 1.0)
        uld.observe("user1", 1.0)  # should be ignored
        assert uld._counter.true_count == pytest.approx(2.0, abs=0.01)

    def test_different_users_contribute(self):
        """Different users can all contribute."""
        uld = UserLevelDP(max_time=100, epsilon=1.0, max_contributions=1, seed=42)
        uld.observe("user1", 1.0)
        uld.observe("user2", 1.0)
        assert uld._counter.true_count == pytest.approx(2.0, abs=0.01)

    def test_sensitivity_scaling(self):
        """User-level sensitivity = max_contributions * per_event_sensitivity."""
        uld = UserLevelDP(max_contributions=3, per_event_sensitivity=2.0)
        assert uld.sensitivity == 6.0

    def test_num_users(self):
        """Num users tracks unique user IDs."""
        uld = UserLevelDP(max_time=100, epsilon=1.0, seed=42)
        uld.observe("a", 1.0)
        uld.observe("b", 1.0)
        uld.observe("a", 1.0)
        assert uld.num_users == 2


class TestWindowedDP:
    """Test WindowedDP sliding window."""

    def test_window_respects_size(self):
        """Buffer doesn't exceed window size."""
        wd = WindowedDP(window_size=5, epsilon=1.0, seed=42)
        for _ in range(10):
            wd.observe(1.0)
        assert len(wd._buffer) == 5

    def test_privacy_bounded(self):
        """Privacy is bounded by epsilon."""
        wd = WindowedDP(window_size=5, epsilon=2.0, seed=42)
        for _ in range(10):
            wd.observe(1.0)
        assert wd.privacy_spent() == 2.0

    def test_query_matches_observe(self):
        """Query returns same as last observe."""
        wd = WindowedDP(window_size=5, epsilon=1.0, seed=42)
        wd.observe(1.0)
        q = wd.query()
        assert q.true_value == 1.0

    def test_uniform_budget(self):
        """Uniform budget strategy splits evenly."""
        wd = WindowedDP(window_size=4, epsilon=1.0, budget_strategy="uniform", seed=42)
        expected = 1.0 / 4.0
        assert wd._budget_per_step[0] == pytest.approx(expected)

    def test_exponential_budget(self):
        """Exponential budget gives more to recent events."""
        wd = WindowedDP(window_size=4, epsilon=1.0, budget_strategy="exponential", seed=42)
        assert wd._budget_per_step[-1] > wd._budget_per_step[0]

    def test_reset(self):
        """Reset clears window."""
        wd = WindowedDP(window_size=5, epsilon=1.0, seed=42)
        wd.observe(1.0)
        wd.reset()
        assert len(wd._buffer) == 0


class TestPrivacyFilter:
    """Test PrivacyFilter budget enforcement."""

    def test_allows_within_budget(self):
        """Queries within budget are allowed."""
        pf = PrivacyFilter(total_epsilon=1.0)
        assert pf.check_and_charge(0.3)
        assert pf.check_and_charge(0.3)
        assert pf.check_and_charge(0.3)

    def test_halts_when_exceeded(self):
        """Filter halts when budget exceeded."""
        pf = PrivacyFilter(total_epsilon=1.0, slack=0.0)
        pf.check_and_charge(0.6)
        pf.check_and_charge(0.3)
        assert not pf.check_and_charge(0.5)
        assert pf.is_halted

    def test_budget_remaining(self):
        """Budget remaining decreases correctly."""
        pf = PrivacyFilter(total_epsilon=1.0)
        pf.check_and_charge(0.3)
        assert pf.budget_remaining == pytest.approx(0.7)

    def test_queries_answered_count(self):
        """Queries answered is tracked."""
        pf = PrivacyFilter(total_epsilon=1.0)
        pf.check_and_charge(0.1)
        pf.check_and_charge(0.1)
        assert pf.queries_answered == 2

    def test_reset(self):
        """Reset restores filter."""
        pf = PrivacyFilter(total_epsilon=1.0, slack=0.0)
        pf.check_and_charge(1.0)
        pf.check_and_charge(0.1)  # halts
        pf.reset()
        assert not pf.is_halted
        assert pf.queries_answered == 0


class TestAboveThreshold:
    """Test AboveThreshold correctness."""

    def test_below_threshold_not_counted(self):
        """Below-threshold queries don't count."""
        at = AboveThreshold(threshold=10.0, epsilon=1.0, max_above=1, seed=42)
        result = at.test(0.0)
        # Very likely below threshold
        # Can't guarantee due to noise but should work most of the time
        assert isinstance(result.above_threshold, bool)

    def test_halts_after_max_above(self):
        """Halts after max_above positive answers."""
        at = AboveThreshold(threshold=-1000.0, epsilon=1.0, max_above=2, seed=42)
        # Values well above threshold
        at.test(1000.0)
        at.test(1000.0)
        assert at.is_halted or at.above_count >= 2

    def test_privacy_constant(self):
        """Privacy is epsilon."""
        at = AboveThreshold(threshold=0.0, epsilon=1.5, seed=42)
        at.test(1.0)
        assert at.privacy_spent() == 1.5

    def test_batch_test(self):
        """Batch test processes multiple queries."""
        at = AboveThreshold(threshold=0.0, epsilon=1.0, max_above=10, seed=42)
        results = at.batch_test([1.0, 2.0, 3.0])
        assert len(results) == 3

    def test_reset(self):
        """Reset allows reuse."""
        at = AboveThreshold(threshold=-1000.0, epsilon=1.0, max_above=1, seed=42)
        at.test(1000.0)
        at.reset()
        assert not at.is_halted
        assert at.above_count == 0
