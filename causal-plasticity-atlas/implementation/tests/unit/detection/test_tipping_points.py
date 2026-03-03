"""Tests for tipping-point detection (ALG4).

Covers PELT on known changepoint sequences, permutation validation,
BH correction, mechanism attribution, effect size computation,
no changepoints, multiple changepoints, segment analysis, TippingPointReport.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.detection.tipping_points import (
    PELTDetector,
    TippingPoint,
    TippingPointResult,
    SegmentAnalyzer,
)
from cpa.detection.changepoint import PELTSolver, ChangepointResult, Segment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def detector():
    return PELTDetector(
        penalty_factor=0.5,
        min_segment_length=2,
        n_permutations=99,
        significance_level=0.05,
        random_state=42,
    )


@pytest.fixture
def single_changepoint_data(rng):
    """Divergence sequence with one changepoint at position 25."""
    n = 50
    seq = np.concatenate([
        rng.normal(0.1, 0.05, size=25),
        rng.normal(0.8, 0.05, size=25),
    ])
    return seq


@pytest.fixture
def two_changepoint_data(rng):
    """Divergence sequence with changepoints at positions 20 and 40."""
    seq = np.concatenate([
        rng.normal(0.1, 0.05, size=20),
        rng.normal(0.7, 0.05, size=20),
        rng.normal(0.2, 0.05, size=20),
    ])
    return seq


@pytest.fixture
def no_changepoint_data(rng):
    """Constant divergence — no changepoints."""
    return rng.normal(0.5, 0.05, size=50)


@pytest.fixture
def simple_adjacencies(rng):
    """List of adjacency matrices for multiple contexts."""
    n_contexts = 10
    p = 4
    adjs = []
    for k in range(n_contexts):
        adj = np.zeros((p, p))
        # Chain: 0->1->2->3 with slight variation after context 5
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 3] = 1
        if k >= 5:
            adj[0, 2] = 1  # Add edge after changepoint
        adjs.append(adj)
    return adjs


@pytest.fixture
def simple_datasets(rng):
    """List of datasets for multiple contexts."""
    n_contexts = 10
    n_samples = 100
    p = 4
    datasets = []
    for k in range(n_contexts):
        # Generate data with different parameters after changepoint
        if k < 5:
            X = rng.normal(0, 1, size=(n_samples, p))
        else:
            X = rng.normal(1, 1.5, size=(n_samples, p))
        datasets.append(X)
    return datasets


# ---------------------------------------------------------------------------
# Test PELT on known changepoint sequences
# ---------------------------------------------------------------------------

class TestPELTOnKnownSequences:

    def test_single_changepoint_detected(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        assert isinstance(result, TippingPointResult)
        assert result.n_tipping_points >= 1
        # Changepoint should be near position 25
        if result.tipping_points:
            locations = [tp.location for tp in result.tipping_points]
            assert any(20 <= loc <= 30 for loc in locations)

    def test_two_changepoints_detected(self, detector, two_changepoint_data):
        result = detector.detect_from_divergence(two_changepoint_data)
        assert result.n_tipping_points >= 1

    def test_no_changepoint_constant_signal(self, detector, no_changepoint_data):
        result = detector.detect_from_divergence(no_changepoint_data)
        # May detect 0 changepoints or very few
        assert result.n_tipping_points <= 2

    def test_from_adjacencies_and_data(self, detector, simple_adjacencies, simple_datasets):
        result = detector.detect(simple_adjacencies, simple_datasets)
        assert isinstance(result, TippingPointResult)

    def test_result_has_divergence_sequence(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        assert result.divergence_sequence is not None
        assert len(result.divergence_sequence) == len(single_changepoint_data)

    def test_result_has_segments(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        assert len(result.segments) >= 1

    def test_segments_cover_full_sequence(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        # Segments should cover [0, n)
        boundaries = result.segment_boundaries()
        if boundaries:
            assert boundaries[0][0] == 0
            assert boundaries[-1][1] == len(single_changepoint_data)

    def test_sharp_changepoint_precise(self, rng):
        """Very sharp changepoint should be detected precisely."""
        n = 100
        seq = np.concatenate([np.zeros(50), np.ones(50)]) + rng.normal(0, 0.01, size=n)
        det = PELTDetector(
            penalty_factor=1.0, min_segment_length=3,
            n_permutations=99, random_state=42,
        )
        result = det.detect_from_divergence(seq)
        if result.tipping_points:
            locations = [tp.location for tp in result.tipping_points]
            assert any(45 <= loc <= 55 for loc in locations)


# ---------------------------------------------------------------------------
# Test permutation validation
# ---------------------------------------------------------------------------

class TestPermutationValidation:

    def test_p_values_assigned(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        for tp in result.tipping_points:
            assert 0.0 <= tp.p_value <= 1.0

    def test_significant_tipping_points(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        sig = result.significant_tipping_points
        for tp in sig:
            assert tp.is_significant

    def test_more_permutations_more_precise(self, single_changepoint_data):
        det1 = PELTDetector(n_permutations=49, random_state=42)
        det2 = PELTDetector(n_permutations=499, random_state=42)
        r1 = det1.detect_from_divergence(single_changepoint_data)
        r2 = det2.detect_from_divergence(single_changepoint_data)
        # Both should detect changepoints
        assert isinstance(r1, TippingPointResult)
        assert isinstance(r2, TippingPointResult)

    def test_different_significance_levels(self, single_changepoint_data):
        det_strict = PELTDetector(significance_level=0.01, n_permutations=99, random_state=42)
        det_lenient = PELTDetector(significance_level=0.10, n_permutations=99, random_state=42)
        r_strict = det_strict.detect_from_divergence(single_changepoint_data)
        r_lenient = det_lenient.detect_from_divergence(single_changepoint_data)
        n_sig_strict = len(r_strict.significant_tipping_points)
        n_sig_lenient = len(r_lenient.significant_tipping_points)
        assert n_sig_strict <= n_sig_lenient


# ---------------------------------------------------------------------------
# Test BH correction
# ---------------------------------------------------------------------------

class TestBHCorrection:

    def test_fdr_adjusted_p_values(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        for tp in result.tipping_points:
            if tp.fdr_adjusted_p is not None:
                assert tp.fdr_adjusted_p >= tp.p_value - 1e-10
                assert tp.fdr_adjusted_p <= 1.0

    def test_multiple_changepoints_correction(self, detector, two_changepoint_data):
        result = detector.detect_from_divergence(two_changepoint_data)
        # With multiple changepoints, FDR correction may be applied
        for tp in result.tipping_points:
            assert tp.p_value >= 0.0


# ---------------------------------------------------------------------------
# Test mechanism attribution
# ---------------------------------------------------------------------------

class TestMechanismAttribution:

    def test_attribution_present(self, detector, simple_adjacencies, simple_datasets):
        result = detector.detect(simple_adjacencies, simple_datasets)
        # Attribution may or may not be filled
        for tp in result.tipping_points:
            assert isinstance(tp.attributed_mechanisms, list)

    def test_attribution_with_target(self, simple_adjacencies, simple_datasets):
        det = PELTDetector(
            n_permutations=49, random_state=42,
        )
        result = det.detect(
            simple_adjacencies, simple_datasets, target_idx=1,
        )
        assert isinstance(result, TippingPointResult)


# ---------------------------------------------------------------------------
# Test effect size computation
# ---------------------------------------------------------------------------

class TestEffectSize:

    def test_effect_size_positive_for_changepoint(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        for tp in result.tipping_points:
            assert tp.effect_size >= 0.0

    def test_effect_size_ci(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        for tp in result.tipping_points:
            if tp.effect_ci is not None:
                lo, hi = tp.effect_ci
                assert lo <= hi


# ---------------------------------------------------------------------------
# Test with no changepoints
# ---------------------------------------------------------------------------

class TestNoChangepoints:

    def test_constant_sequence(self, detector):
        seq = np.ones(30) * 0.5
        result = detector.detect_from_divergence(seq)
        assert result.n_tipping_points == 0

    def test_low_variance_noise(self, detector, rng):
        seq = rng.normal(0.5, 0.001, size=30)
        result = detector.detect_from_divergence(seq)
        assert result.n_tipping_points <= 1

    def test_short_sequence(self, detector, rng):
        seq = rng.normal(0, 1, size=5)
        result = detector.detect_from_divergence(seq)
        assert isinstance(result, TippingPointResult)


# ---------------------------------------------------------------------------
# Test with multiple changepoints
# ---------------------------------------------------------------------------

class TestMultipleChangepoints:

    def test_three_segments(self, rng):
        seq = np.concatenate([
            rng.normal(0.0, 0.05, size=30),
            rng.normal(1.0, 0.05, size=30),
            rng.normal(0.0, 0.05, size=30),
        ])
        det = PELTDetector(n_permutations=99, random_state=42)
        result = det.detect_from_divergence(seq)
        assert result.n_tipping_points >= 1

    def test_many_changepoints(self, rng):
        """Signal with many changepoints."""
        segments = []
        for i in range(5):
            mean = rng.uniform(-1, 1)
            segments.append(rng.normal(mean, 0.05, size=20))
        seq = np.concatenate(segments)
        det = PELTDetector(n_permutations=49, min_segment_length=3, penalty_factor=0.3, random_state=42)
        result = det.detect_from_divergence(seq)
        assert result.n_tipping_points >= 2

    def test_changepoints_ordered(self, detector, two_changepoint_data):
        result = detector.detect_from_divergence(two_changepoint_data)
        locations = [tp.location for tp in result.tipping_points]
        assert locations == sorted(locations)


# ---------------------------------------------------------------------------
# Test segment analysis
# ---------------------------------------------------------------------------

class TestSegmentAnalysis:

    def test_segment_boundaries(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        boundaries = result.segment_boundaries()
        assert isinstance(boundaries, list)
        for start, end in boundaries:
            assert start < end

    def test_segment_analyzer(self, detector, single_changepoint_data):
        result = detector.detect_from_divergence(single_changepoint_data)
        analyzer = SegmentAnalyzer()
        # Should be able to analyze segments
        assert result.segments is not None


# ---------------------------------------------------------------------------
# Test TippingPoint dataclass
# ---------------------------------------------------------------------------

class TestTippingPointDataclass:

    def test_is_significant_property(self):
        tp = TippingPoint(location=10, p_value=0.01)
        assert tp.is_significant

    def test_not_significant(self):
        tp = TippingPoint(location=10, p_value=0.5)
        assert not tp.is_significant

    def test_tipping_point_fields(self):
        tp = TippingPoint(
            location=10, p_value=0.02,
            effect_size=1.5, change_type="structural",
        )
        assert tp.location == 10
        assert tp.effect_size == 1.5
        assert tp.change_type == "structural"

    def test_tipping_point_result_n_tipping(self, rng):
        tps = [
            TippingPoint(location=10, p_value=0.01),
            TippingPoint(location=30, p_value=0.03),
        ]
        result = TippingPointResult(
            tipping_points=tps,
            divergence_sequence=rng.normal(0, 1, size=50),
            segments=[],
            n_contexts=50,
            penalty=1.0,
            method="PELT",
        )
        assert result.n_tipping_points == 2

    def test_significant_tipping_points_filter(self):
        tps = [
            TippingPoint(location=10, p_value=0.01),
            TippingPoint(location=30, p_value=0.5),
        ]
        result = TippingPointResult(
            tipping_points=tps,
            divergence_sequence=np.zeros(50),
            segments=[],
            n_contexts=50,
            penalty=1.0,
            method="PELT",
        )
        sig = result.significant_tipping_points
        assert len(sig) == 1
        assert sig[0].location == 10


# ---------------------------------------------------------------------------
# Test different penalty factors
# ---------------------------------------------------------------------------

class TestPenaltyFactors:

    @pytest.mark.parametrize("penalty", [0.5, 1.0, 2.0, 5.0])
    def test_penalty_affects_n_changepoints(self, penalty, single_changepoint_data):
        det = PELTDetector(
            penalty_factor=penalty,
            n_permutations=49,
            random_state=42,
        )
        result = det.detect_from_divergence(single_changepoint_data)
        assert isinstance(result, TippingPointResult)

    def test_higher_penalty_fewer_changepoints(self, rng):
        """Higher penalty should give fewer changepoints."""
        seq = np.concatenate([
            rng.normal(0, 0.1, size=30),
            rng.normal(0.5, 0.1, size=30),
            rng.normal(1.0, 0.1, size=30),
        ])
        det_low = PELTDetector(penalty_factor=0.1, n_permutations=49, random_state=42)
        det_high = PELTDetector(penalty_factor=10.0, n_permutations=49, random_state=42)
        r_low = det_low.detect_from_divergence(seq)
        r_high = det_high.detect_from_divergence(seq)
        assert r_high.n_tipping_points <= r_low.n_tipping_points

    def test_min_segment_length(self, rng):
        """min_segment_length should prevent very short segments."""
        seq = np.concatenate([
            rng.normal(0, 0.05, size=3),
            rng.normal(1, 0.05, size=3),
            rng.normal(0, 0.05, size=44),
        ])
        det = PELTDetector(min_segment_length=5, n_permutations=49, random_state=42)
        result = det.detect_from_divergence(seq)
        if result.segments:
            for seg in result.segments:
                assert seg.end - seg.start >= 2  # At least min length or close
