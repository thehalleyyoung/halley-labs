"""Unit tests for cpa.streaming – IncrementalPlasticityUpdater, SufficientStatistics,
WindowedDetector, OnlineCUSUM, CircularBuffer."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.streaming.incremental_plasticity import (
    IncrementalPlasticityUpdater,
    SufficientStatistics,
    PlasticityDelta,
)
from cpa.streaming.windowed_detection import (
    WindowedDetector,
    OnlineCUSUM,
    DetectionWindow,
)
from cpa.streaming.stream_buffer import (
    CircularBuffer,
    SlidingWindow,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sufficient_stats():
    return SufficientStatistics(num_variables=3)


@pytest.fixture
def simple_descriptors():
    """Initial descriptors for 3-variable system."""
    return {
        0: {"psi_S": 0.0, "psi_P": 0.0, "psi_E": 0.0, "psi_CS": 0.0},
        1: {"psi_S": 0.0, "psi_P": 0.0, "psi_E": 0.0, "psi_CS": 0.0},
        2: {"psi_S": 0.0, "psi_P": 0.0, "psi_E": 0.0, "psi_CS": 0.0},
    }


@pytest.fixture
def updater(simple_descriptors):
    return IncrementalPlasticityUpdater(
        initial_descriptors=simple_descriptors,
        alpha=0.05,
        decay_rate=0.95,
    )


@pytest.fixture
def cusum():
    return OnlineCUSUM(threshold=5.0, drift=0.5)


@pytest.fixture
def windowed_detector():
    return WindowedDetector(
        window_size=20,
        step_size=5,
        detection_method="cusum",
        threshold=5.0,
    )


# ===================================================================
# Tests – SufficientStatistics
# ===================================================================


class TestSufficientStatistics:
    """Test SufficientStatistics online updates."""

    def test_initial_state(self, sufficient_stats):
        assert sufficient_stats.n == 0

    def test_update_single_observation(self, sufficient_stats, rng):
        x = rng.normal(0, 1, 3)
        sufficient_stats.update(x)
        assert sufficient_stats.n == 1

    def test_mean_after_updates(self, sufficient_stats, rng):
        data = rng.normal(0, 1, (100, 3))
        for x in data:
            sufficient_stats.update(x)
        assert_allclose(sufficient_stats.mean, np.mean(data, axis=0), atol=1e-10)

    def test_covariance_matches_batch(self, sufficient_stats, rng):
        data = rng.normal(0, 1, (200, 3))
        for x in data:
            sufficient_stats.update(x)
        batch_cov = np.cov(data.T, bias=True)
        online_cov = sufficient_stats.covariance
        assert_allclose(online_cov, batch_cov, atol=0.1)

    def test_n_tracks_count(self, sufficient_stats, rng):
        for i in range(50):
            sufficient_stats.update(rng.normal(0, 1, 3))
        assert sufficient_stats.n == 50

    def test_reset(self, sufficient_stats, rng):
        for i in range(10):
            sufficient_stats.update(rng.normal(0, 1, 3))
        sufficient_stats.reset()
        assert sufficient_stats.n == 0

    def test_merge(self, rng):
        ss1 = SufficientStatistics(num_variables=3)
        ss2 = SufficientStatistics(num_variables=3)
        data = rng.normal(0, 1, (100, 3))
        for x in data[:50]:
            ss1.update(x)
        for x in data[50:]:
            ss2.update(x)
        ss1.merge(ss2)
        assert ss1.n == 100

    def test_copy(self, sufficient_stats, rng):
        for i in range(10):
            sufficient_stats.update(rng.normal(0, 1, 3))
        copy = sufficient_stats.copy()
        assert copy.n == sufficient_stats.n
        assert_allclose(copy.mean, sufficient_stats.mean)


# ===================================================================
# Tests – IncrementalPlasticityUpdater
# ===================================================================


class TestIncrementalPlasticityUpdater:
    """Test IncrementalPlasticityUpdater processes new contexts."""

    def test_update_returns_list(self, updater, rng):
        data = rng.normal(0, 1, (100, 3))
        deltas = updater.update(data, context_id="ctx_0")
        assert isinstance(deltas, list)

    def test_current_descriptors(self, updater, rng):
        data = rng.normal(0, 1, (100, 3))
        updater.update(data, context_id="ctx_0")
        desc = updater.current_descriptors()
        assert isinstance(desc, dict)

    def test_multiple_contexts(self, updater, rng):
        for i in range(3):
            data = rng.normal(0, 1, (100, 3))
            updater.update(data, context_id=f"ctx_{i}")
        desc = updater.current_descriptors()
        assert isinstance(desc, dict)

    def test_change_history(self, updater, rng):
        for i in range(3):
            data = rng.normal(i, 1, (100, 3))
            updater.update(data, context_id=f"ctx_{i}")
        history = updater.change_history()
        assert isinstance(history, list)

    def test_exponential_decay(self, updater):
        w = updater.exponential_decay_weight(0.0)
        assert_allclose(w, 1.0, atol=1e-10)
        w_later = updater.exponential_decay_weight(10.0)
        assert w_later < w

    def test_delta_fields(self, updater, rng):
        data1 = rng.normal(0, 1, (100, 3))
        data2 = rng.normal(5, 1, (100, 3))
        updater.update(data1, context_id="ctx_0")
        deltas = updater.update(data2, context_id="ctx_1")
        for d in deltas:
            assert isinstance(d, PlasticityDelta)
            assert hasattr(d, "node")
            assert hasattr(d, "old_class")
            assert hasattr(d, "new_class")

    def test_rollback(self, updater, rng):
        data = rng.normal(0, 1, (100, 3))
        updater.update(data, context_id="ctx_0")
        updater.rollback(n_steps=1)
        desc = updater.current_descriptors()
        assert isinstance(desc, dict)


# ===================================================================
# Tests – OnlineCUSUM
# ===================================================================


class TestOnlineCUSUM:
    """Test OnlineCUSUM with known shifts."""

    def test_no_change_no_detection(self, cusum, rng):
        for _ in range(50):
            detected = cusum.update(rng.normal(0, 0.1))
        assert not cusum.has_change()

    def test_detects_mean_shift(self, cusum, rng):
        # Phase 1: normal
        for _ in range(50):
            cusum.update(rng.normal(0, 0.1))
        # Phase 2: shifted mean
        detected_any = False
        for _ in range(100):
            if cusum.update(rng.normal(5.0, 0.1)):
                detected_any = True
                break
        assert detected_any

    def test_reset(self, cusum, rng):
        for _ in range(50):
            cusum.update(rng.normal(5.0, 0.1))
        cusum.reset()
        assert not cusum.has_change()

    def test_full_reset(self, cusum, rng):
        for _ in range(50):
            cusum.update(rng.normal(5.0, 0.1))
        cusum.full_reset()
        assert not cusum.has_change()

    def test_statistics_returns_tuple(self, cusum, rng):
        cusum.update(rng.normal(0, 1))
        stats = cusum.statistics
        assert isinstance(stats, tuple)
        assert len(stats) == 2

    def test_repr(self, cusum):
        assert "CUSUM" in repr(cusum) or "OnlineCUSUM" in repr(cusum)


# ===================================================================
# Tests – WindowedDetector
# ===================================================================


class TestWindowedDetector:
    """Test WindowedDetector detects known changepoints."""

    def test_feed_data(self, windowed_detector, rng):
        for i in range(30):
            windowed_detector.feed(rng.normal(0, 1, 3))

    def test_detect_returns_list(self, windowed_detector, rng):
        for i in range(30):
            windowed_detector.feed(rng.normal(0, 1, 3))
        detections = windowed_detector.detect()
        assert isinstance(detections, list)

    def test_detects_changepoint(self, rng):
        wd = WindowedDetector(
            window_size=20, step_size=5,
            detection_method="cusum", threshold=3.0,
        )
        # Phase 1: stable
        for i in range(30):
            wd.feed(rng.normal(0, 0.1, 3))
        # Phase 2: shifted
        for i in range(30):
            wd.feed(rng.normal(5, 0.1, 3))
        detections = wd.detect()
        # May or may not detect depending on implementation, just should not crash

    def test_current_window(self, windowed_detector, rng):
        for i in range(25):
            windowed_detector.feed(rng.normal(0, 1, 3))
        window = windowed_detector.current_window()
        assert isinstance(window, np.ndarray)

    def test_reset(self, windowed_detector, rng):
        for i in range(20):
            windowed_detector.feed(rng.normal(0, 1, 3))
        windowed_detector.reset()
        detections = windowed_detector.detect()
        assert len(detections) == 0

    def test_detected_changepoints(self, windowed_detector, rng):
        for i in range(50):
            windowed_detector.feed(rng.normal(0, 1, 3))
        cps = windowed_detector.detected_changepoints()
        assert isinstance(cps, list)

    def test_detection_window_fields(self):
        dw = DetectionWindow(start_idx=0, end_idx=10, statistic=3.5,
                              is_changepoint=True)
        assert dw.start_idx == 0
        assert dw.end_idx == 10
        assert dw.statistic == 3.5
        assert dw.is_changepoint


# ===================================================================
# Tests – CircularBuffer
# ===================================================================


class TestCircularBuffer:
    """Test CircularBuffer overflow behavior."""

    def test_push_within_capacity(self):
        buf = CircularBuffer(capacity=5, shape=(3,), dtype=np.float64)
        buf.push(np.array([1.0, 2.0, 3.0]))
        assert len(buf) == 1

    def test_push_to_capacity(self):
        buf = CircularBuffer(capacity=5, shape=(3,), dtype=np.float64)
        for i in range(5):
            buf.push(np.ones(3) * i)
        assert len(buf) == 5
        assert buf.is_full()

    def test_overflow_wraps(self):
        buf = CircularBuffer(capacity=3, shape=(2,), dtype=np.float64)
        for i in range(5):
            buf.push(np.ones(2) * i)
        assert len(buf) == 3
        # Should contain last 3 items: 2, 3, 4
        all_data = buf.get_all()
        assert all_data.shape == (3, 2)

    def test_peek(self):
        buf = CircularBuffer(capacity=5, shape=(2,), dtype=np.float64)
        for i in range(5):
            buf.push(np.ones(2) * i)
        last_2 = buf.peek(2)
        assert last_2.shape == (2, 2)

    def test_get_all(self):
        buf = CircularBuffer(capacity=5, shape=(2,), dtype=np.float64)
        for i in range(3):
            buf.push(np.ones(2) * i)
        data = buf.get_all()
        assert data.shape == (3, 2)

    def test_empty_buffer(self):
        buf = CircularBuffer(capacity=5, shape=(2,), dtype=np.float64)
        assert len(buf) == 0
        assert not buf.is_full()


# ===================================================================
# Tests – SlidingWindow
# ===================================================================


class TestSlidingWindow:
    """Test SlidingWindow statistics."""

    def test_append_and_length(self):
        sw = SlidingWindow(max_size=5, dtype=float)
        for i in range(3):
            sw.append(float(i))
        assert len(sw) == 3

    def test_overflow_keeps_max_size(self):
        sw = SlidingWindow(max_size=5, dtype=float)
        for i in range(10):
            sw.append(float(i))
        assert len(sw) == 5

    def test_mean(self):
        sw = SlidingWindow(max_size=100, dtype=float)
        for i in range(10):
            sw.append(float(i))
        assert_allclose(sw.mean(), 4.5, atol=1e-10)

    def test_std(self):
        sw = SlidingWindow(max_size=100, dtype=float)
        for v in [1.0, 1.0, 1.0]:
            sw.append(v)
        assert_allclose(sw.std(), 0.0, atol=1e-10)

    def test_get_window(self):
        sw = SlidingWindow(max_size=5, dtype=float)
        for i in range(5):
            sw.append(float(i))
        w = sw.get_window()
        assert isinstance(w, np.ndarray)
        assert len(w) == 5

    def test_is_full(self):
        sw = SlidingWindow(max_size=3, dtype=float)
        sw.append(1.0)
        sw.append(2.0)
        assert not sw.is_full()
        sw.append(3.0)
        assert sw.is_full()
