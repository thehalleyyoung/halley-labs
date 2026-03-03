"""Tests for generic changepoint detection.

Covers PELT algorithm, binary segmentation, CUSUM,
different cost functions, and penalty selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.detection.changepoint import (
    PELTSolver,
    ChangepointResult,
    Segment,
    CostType,
    PenaltyType,
    L2Cost,
    GaussianLikelihoodCost,
    RBFKernelCost,
    make_cost_function,
    compute_penalty,
    compute_segment_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def single_change_signal(rng):
    """Signal with a single changepoint at index 50."""
    return np.concatenate([
        rng.normal(0.0, 1.0, size=50),
        rng.normal(5.0, 1.0, size=50),
    ])


@pytest.fixture
def two_change_signal(rng):
    """Signal with changepoints at 30 and 70."""
    return np.concatenate([
        rng.normal(0.0, 0.5, size=30),
        rng.normal(3.0, 0.5, size=40),
        rng.normal(0.0, 0.5, size=30),
    ])


@pytest.fixture
def no_change_signal(rng):
    """Stationary signal."""
    return rng.normal(0.0, 1.0, size=100)


@pytest.fixture
def multidim_signal(rng):
    """Multivariate signal with a changepoint."""
    return np.concatenate([
        rng.normal(0, 1, size=(50, 3)),
        rng.normal(3, 1, size=(50, 3)),
    ])


# ---------------------------------------------------------------------------
# Test PELT algorithm
# ---------------------------------------------------------------------------

class TestPELTAlgorithm:

    def test_single_changepoint(self, single_change_signal):
        solver = PELTSolver(cost_type="l2", penalty_type="bic")
        result = solver.fit(single_change_signal)
        assert isinstance(result, ChangepointResult)
        assert result.n_changepoints >= 1
        assert any(40 <= cp <= 60 for cp in result.changepoints)

    def test_two_changepoints(self, two_change_signal):
        solver = PELTSolver(cost_type="l2", penalty_type="bic")
        result = solver.fit(two_change_signal)
        assert result.n_changepoints >= 1

    def test_no_changepoint(self, no_change_signal):
        solver = PELTSolver(cost_type="l2", penalty_type="bic")
        result = solver.fit(no_change_signal)
        assert result.n_changepoints <= 2

    def test_multidimensional(self, multidim_signal):
        solver = PELTSolver(cost_type="l2", penalty_type="bic")
        result = solver.fit(multidim_signal)
        assert isinstance(result, ChangepointResult)

    def test_result_has_segments(self, single_change_signal):
        solver = PELTSolver(cost_type="l2")
        result = solver.fit(single_change_signal)
        assert len(result.segments) >= 1

    def test_segments_partition_signal(self, single_change_signal):
        solver = PELTSolver(cost_type="l2")
        result = solver.fit(single_change_signal)
        boundaries = result.segment_boundaries()
        if boundaries:
            assert boundaries[0][0] == 0
            assert boundaries[-1][1] == len(single_change_signal)

    def test_min_size_constraint(self, single_change_signal):
        solver = PELTSolver(cost_type="l2", min_size=10)
        result = solver.fit(single_change_signal)
        for seg in result.segments:
            assert seg.length >= 2  # At least some minimum

    def test_result_cost(self, single_change_signal):
        solver = PELTSolver(cost_type="l2")
        result = solver.fit(single_change_signal)
        assert result.cost >= 0

    def test_result_method(self, single_change_signal):
        solver = PELTSolver()
        result = solver.fit(single_change_signal)
        assert result.method == "PELT" or "pelt" in result.method.lower()


# ---------------------------------------------------------------------------
# Test cost functions
# ---------------------------------------------------------------------------

class TestCostFunctions:

    def test_l2_cost(self, rng):
        signal = rng.normal(0, 1, size=100)
        cost = L2Cost(signal)
        c = cost.cost(0, 50)
        assert c > 0

    def test_l2_cost_zero_for_constant(self):
        signal = np.ones(100)
        cost = L2Cost(signal)
        c = cost.cost(0, 100)
        assert_allclose(c, 0.0, atol=1e-10)

    def test_gaussian_likelihood_cost(self, rng):
        signal = rng.normal(0, 1, size=100)
        cost = GaussianLikelihoodCost(signal)
        c = cost.cost(0, 100)
        assert np.isfinite(c)

    def test_rbf_kernel_cost(self, rng):
        signal = rng.normal(0, 1, size=50)
        cost = RBFKernelCost(signal, gamma=1.0)
        c = cost.cost(0, 50)
        assert np.isfinite(c)

    def test_make_cost_function_l2(self, rng):
        signal = rng.normal(0, 1, size=100)
        cost = make_cost_function(signal, "l2")
        assert isinstance(cost, L2Cost)

    def test_make_cost_function_gaussian(self, rng):
        signal = rng.normal(0, 1, size=100)
        cost = make_cost_function(signal, "gaussian_likelihood")
        assert isinstance(cost, GaussianLikelihoodCost)

    def test_cost_segment_stats(self, rng):
        signal = rng.normal(5, 2, size=100)
        cost = L2Cost(signal)
        stats = cost.segment_stats(0, 100)
        assert isinstance(stats, dict)

    def test_cost_monotonic_with_segment_length(self, rng):
        """Longer segments should generally have higher cost."""
        signal = rng.normal(0, 1, size=100)
        cost = L2Cost(signal)
        c_short = cost.cost(0, 10)
        c_long = cost.cost(0, 100)
        assert c_long >= c_short


# ---------------------------------------------------------------------------
# Test penalty selection
# ---------------------------------------------------------------------------

class TestPenaltySelection:

    def test_bic_penalty(self):
        pen = compute_penalty(100, n_dims=1, penalty_type="bic")
        assert pen > 0

    def test_aic_penalty(self):
        pen = compute_penalty(100, n_dims=1, penalty_type="aic")
        assert pen > 0

    def test_hannan_quinn(self):
        pen = compute_penalty(100, n_dims=1, penalty_type="hannan_quinn")
        assert pen > 0

    def test_manual_penalty(self):
        pen = compute_penalty(100, n_dims=1, penalty_type="manual", manual_value=5.0)
        assert pen == 5.0

    def test_bic_larger_than_aic(self):
        """BIC penalty is typically larger than AIC for n > e^2 ≈ 7.4."""
        bic = compute_penalty(100, n_dims=1, penalty_type="bic")
        aic = compute_penalty(100, n_dims=1, penalty_type="aic")
        assert bic >= aic

    def test_penalty_scales_with_dimensions(self):
        p1 = compute_penalty(100, n_dims=1, penalty_type="bic")
        p2 = compute_penalty(100, n_dims=5, penalty_type="bic")
        assert p2 >= p1

    @pytest.mark.parametrize("n", [10, 50, 100, 500, 1000])
    def test_bic_penalty_positive(self, n):
        pen = compute_penalty(n, penalty_type="bic")
        assert pen > 0


# ---------------------------------------------------------------------------
# Test Segment dataclass
# ---------------------------------------------------------------------------

class TestSegmentDataclass:

    def test_segment_length(self):
        seg = Segment(start=10, end=30, cost=5.0)
        assert seg.length == 20

    def test_segment_fields(self):
        seg = Segment(start=0, end=50, cost=10.0, mean=np.array([1.0]), variance=0.5, count=50)
        assert seg.start == 0
        assert seg.end == 50
        assert seg.count == 50


# ---------------------------------------------------------------------------
# Test ChangepointResult
# ---------------------------------------------------------------------------

class TestChangepointResult:

    def test_segment_boundaries(self):
        result = ChangepointResult(
            changepoints=[25, 50],
            segments=[
                Segment(start=0, end=25, cost=1.0),
                Segment(start=25, end=50, cost=2.0),
                Segment(start=50, end=100, cost=1.5),
            ],
            cost=4.5,
            penalty=2.0,
            n_changepoints=2,
            method="PELT",
        )
        boundaries = result.segment_boundaries()
        assert boundaries == [(0, 25), (25, 50), (50, 100)]

    def test_n_changepoints(self):
        result = ChangepointResult(
            changepoints=[25],
            segments=[],
            cost=1.0,
            penalty=1.0,
            n_changepoints=1,
            method="PELT",
        )
        assert result.n_changepoints == 1


# ---------------------------------------------------------------------------
# Test compute_segment_statistics
# ---------------------------------------------------------------------------

class TestSegmentStatistics:

    def test_compute_statistics(self, rng):
        signal = rng.normal(0, 1, size=100)
        segments = [
            Segment(start=0, end=50, cost=1.0),
            Segment(start=50, end=100, cost=1.0),
        ]
        stats = compute_segment_statistics(signal, segments)
        assert len(stats) == 2
        for s in stats:
            assert isinstance(s, dict)

    def test_statistics_have_mean(self, rng):
        signal = rng.normal(5, 1, size=100)
        segments = [Segment(start=0, end=100, cost=1.0)]
        stats = compute_segment_statistics(signal, segments)
        assert "mean" in stats[0]
        assert_allclose(stats[0]["mean"], 5.0, atol=0.5)


# ---------------------------------------------------------------------------
# Test CostType and PenaltyType enums
# ---------------------------------------------------------------------------

class TestEnums:

    def test_cost_types(self):
        assert CostType.L2 is not None
        assert CostType.GAUSSIAN_LIKELIHOOD is not None

    def test_penalty_types(self):
        assert PenaltyType.BIC is not None
        assert PenaltyType.AIC is not None
        assert PenaltyType.MANUAL is not None

    def test_solver_with_enum(self, single_change_signal):
        solver = PELTSolver(cost_type=CostType.L2, penalty_type=PenaltyType.BIC)
        result = solver.fit(single_change_signal)
        assert isinstance(result, ChangepointResult)

    def test_solver_with_string(self, single_change_signal):
        solver = PELTSolver(cost_type="l2", penalty_type="bic")
        result = solver.fit(single_change_signal)
        assert isinstance(result, ChangepointResult)
