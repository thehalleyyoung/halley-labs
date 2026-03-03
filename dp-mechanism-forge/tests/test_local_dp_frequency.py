"""Tests for local_dp.frequency_oracle module."""
import math
import pytest
import numpy as np

from dp_forge.local_dp.frequency_oracle import (
    OLHEstimator,
    HadamardResponse,
    CMS,
    HeavyHitterDetector,
    FrequencyCalibrator,
)
from dp_forge.local_dp import FrequencyEstimate


class TestOLHEstimator:
    """Test OLHEstimator frequency estimation accuracy."""

    def test_encode_returns_report(self):
        """Encode returns a valid report."""
        olh = OLHEstimator(epsilon=2.0, domain_size=10, seed=42)
        r = olh.encode(3)
        assert r.domain_size == 10

    def test_optimal_g(self):
        """Default g = ceil(e^eps + 1)."""
        eps = 2.0
        olh = OLHEstimator(epsilon=eps, domain_size=100, seed=42)
        expected_g = max(2, int(math.ceil(math.exp(eps) + 1)))
        assert olh.g == expected_g

    def test_frequency_estimation(self):
        """OLH estimates frequency distribution."""
        eps = 4.0
        d = 5
        olh = OLHEstimator(epsilon=eps, domain_size=d, seed=42)
        n = 5000
        true_dist = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n, p=true_dist)
        reports = [olh.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = olh.estimate_all(reports)
        # Most frequent item should be detected
        assert np.argmax(est.frequencies) == 0

    def test_estimate_single(self):
        """Single-value estimate works."""
        olh = OLHEstimator(epsilon=3.0, domain_size=5, seed=42)
        n = 2000
        reports = [olh.encode(0, user_id=i) for i in range(n)]
        freq = olh.estimate_single(reports, 0)
        assert freq > 0.3

    def test_invalid_value_raises(self):
        """Invalid value raises error."""
        olh = OLHEstimator(epsilon=1.0, domain_size=5, seed=42)
        with pytest.raises(ValueError):
            olh.encode(10)


class TestHadamardResponse:
    """Test HadamardResponse communication efficiency."""

    def test_encode_returns_report(self):
        """Encode returns valid report with 2 elements."""
        hr = HadamardResponse(epsilon=2.0, domain_size=4, seed=42)
        r = hr.encode(2)
        enc = np.asarray(r.encoded_value)
        assert len(enc) == 2  # [coefficient_index, sign]

    def test_sign_in_valid_range(self):
        """Reported sign is +1 or -1."""
        hr = HadamardResponse(epsilon=2.0, domain_size=4, seed=42)
        r = hr.encode(1)
        enc = np.asarray(r.encoded_value)
        assert int(enc[1]) in (-1, 1)

    def test_estimate_all(self):
        """Estimation produces valid frequency vector."""
        hr = HadamardResponse(epsilon=3.0, domain_size=4, seed=42)
        n = 3000
        rng = np.random.default_rng(42)
        values = rng.choice(4, size=n, p=[0.5, 0.3, 0.15, 0.05])
        reports = [hr.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = hr.estimate_all(reports)
        assert est.frequencies.shape == (4,)
        assert np.sum(est.frequencies) == pytest.approx(1.0, abs=0.01)

    def test_communication_efficiency(self):
        """Each report is just 2 integers (much less than domain_size)."""
        hr = HadamardResponse(epsilon=2.0, domain_size=100, seed=42)
        r = hr.encode(50)
        enc = np.asarray(r.encoded_value)
        assert len(enc) == 2


class TestCMS:
    """Test CMS heavy hitter detection."""

    def test_encode_returns_report(self):
        """Encode returns valid CMS report."""
        cms = CMS(epsilon=2.0, domain_size=100, seed=42)
        r = cms.encode(42)
        enc = np.asarray(r.encoded_value)
        assert len(enc) == 2  # [hash_idx, bucket]

    def test_frequency_estimation(self):
        """CMS estimates frequencies."""
        d = 10
        cms = CMS(epsilon=4.0, domain_size=d, num_hashes=3, sketch_width=32, seed=42)
        n = 5000
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n, p=[0.5] + [0.5 / (d - 1)] * (d - 1))
        reports = [cms.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = cms.estimate_all(reports)
        assert est.frequencies.shape == (d,)
        # Most frequent item should be detected
        assert np.argmax(est.frequencies) == 0

    def test_invalid_value_raises(self):
        """Invalid value raises error."""
        cms = CMS(epsilon=1.0, domain_size=5, seed=42)
        with pytest.raises(ValueError):
            cms.encode(10)


class TestHeavyHitterDetector:
    """Test HeavyHitterDetector precision/recall."""

    def test_detect_heavy_hitters(self):
        """Detects items above threshold."""
        d = 10
        hhd = HeavyHitterDetector(epsilon=4.0, domain_size=d, threshold=0.05, seed=42)
        n = 5000
        rng = np.random.default_rng(42)
        # Value 0 is very frequent (50%)
        values = rng.choice(d, size=n, p=[0.5] + [0.5 / (d - 1)] * (d - 1))
        reports = [hhd.encode(int(v), user_id=i) for i, v in enumerate(values)]
        results = hhd.detect(reports)
        # Item 0 should be in the results
        detected_values = [v for v, _ in results]
        assert 0 in detected_values

    def test_top_k(self):
        """top_k returns k items."""
        d = 10
        hhd = HeavyHitterDetector(epsilon=4.0, domain_size=d, seed=42)
        n = 3000
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n)
        reports = [hhd.encode(int(v), user_id=i) for i, v in enumerate(values)]
        top = hhd.top_k(reports, k=3)
        assert len(top) == 3
        assert top[0][1] >= top[1][1]  # sorted by frequency

    def test_empty_reports(self):
        """Empty reports returns empty list."""
        hhd = HeavyHitterDetector(epsilon=1.0, domain_size=5, seed=42)
        results = hhd.detect([])
        assert len(results) == 0


class TestFrequencyCalibrator:
    """Test FrequencyCalibrator bias correction."""

    def test_simplex_projection(self):
        """Calibrated frequencies are a valid distribution."""
        fc = FrequencyCalibrator(domain_size=5)
        raw_freqs = np.array([0.5, -0.1, 0.3, 0.2, 0.3])
        est = FrequencyEstimate(frequencies=raw_freqs, num_reports=100)
        calibrated = fc.calibrate(est)
        assert np.all(calibrated.frequencies >= -1e-10)
        assert np.sum(calibrated.frequencies) == pytest.approx(1.0, abs=1e-6)

    def test_smoothing(self):
        """Laplace smoothing adds mass before projection."""
        fc = FrequencyCalibrator(domain_size=4, smoothing=0.1)
        raw = np.array([0.25, 0.25, 0.25, 0.25])
        est = FrequencyEstimate(frequencies=raw, num_reports=100)
        calibrated = fc.calibrate(est)
        # After smoothing and projection, should still be a valid distribution
        assert np.all(calibrated.frequencies >= 0)
        assert np.sum(calibrated.frequencies) == pytest.approx(1.0, abs=1e-6)

    def test_merge_estimates(self):
        """Merging estimates produces valid distribution."""
        fc = FrequencyCalibrator(domain_size=3)
        e1 = FrequencyEstimate(frequencies=np.array([0.5, 0.3, 0.2]), num_reports=100)
        e2 = FrequencyEstimate(frequencies=np.array([0.4, 0.4, 0.2]), num_reports=200)
        merged = fc.merge_estimates([e1, e2])
        assert np.sum(merged.frequencies) == pytest.approx(1.0, abs=1e-6)
        assert merged.num_reports == 300

    def test_merge_with_weights(self):
        """Weighted merge respects weights."""
        fc = FrequencyCalibrator(domain_size=3)
        e1 = FrequencyEstimate(frequencies=np.array([1.0, 0.0, 0.0]), num_reports=100)
        e2 = FrequencyEstimate(frequencies=np.array([0.0, 1.0, 0.0]), num_reports=100)
        merged = fc.merge_estimates([e1, e2], weights=[1.0, 1.0])
        # Should be roughly [0.5, 0.5, 0] after projection
        assert merged.frequencies[0] > 0
        assert merged.frequencies[1] > 0
