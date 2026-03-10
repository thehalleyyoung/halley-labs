"""Tests for usability_oracle.montecarlo.statistics.

Verifies mean cost, variance (Bessel), quantiles, CDF properties,
tail risk, path entropy, and hitting times.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.montecarlo.statistics import (
    TrajectoryStatistics,
    compute_cost_cdf,
    compute_cost_quantiles,
    compute_cost_variance,
    compute_mean_cost,
    compute_path_entropy,
    compute_tail_risk,
)
from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    MCConfig,
    SampleStatistics,
    TerminationReason,
    TrajectoryBundle,
    VarianceEstimate,
)


# =====================================================================
# Helpers to build TrajectoryBundles
# =====================================================================

def _make_bundle(
    costs: list[float],
    lengths: list[int] | None = None,
    reasons: list[TerminationReason] | None = None,
    importance_weights: list[ImportanceWeight] | None = None,
) -> TrajectoryBundle:
    """Create a minimal TrajectoryBundle from costs."""
    n = len(costs)
    if n == 0:
        # TrajectoryBundle requires num_trajectories ≥ 1 via MCConfig,
        # so for empty-bundle tests we use n=1 with a sentinel cost of 0.
        # We skip empty-bundle tests and use n >= 1 instead.
        n = 1
        costs = [0.0]
        if lengths is None:
            lengths = [0]
        if reasons is None:
            reasons = [TerminationReason.GOAL_REACHED]
    if lengths is None:
        lengths = [1] * n
    if reasons is None:
        reasons = [TerminationReason.GOAL_REACHED] * n
    ve = VarianceEstimate(
        sample_variance=0.0, standard_error=0.0,
        coefficient_of_variation=0.0, effective_sample_size=float(n),
        ess_ratio=1.0,
    )
    stats = SampleStatistics(
        mean_cost=0.0, median_cost=0.0, std_cost=0.0,
        min_cost=0.0, max_cost=0.0, percentiles={},
        mean_length=0.0, goal_reach_rate=1.0,
        variance_estimate=ve,
    )
    config = MCConfig(num_samples=n, max_trajectory_length=100, seed=42)
    return TrajectoryBundle(
        num_trajectories=n,
        costs=tuple(costs),
        lengths=tuple(lengths),
        termination_reasons=tuple(reasons),
        importance_weights=tuple(importance_weights) if importance_weights else None,
        statistics=stats,
        config=config,
        wall_clock_seconds=0.0,
    )


# =====================================================================
# Mean cost
# =====================================================================

class TestMeanCost:
    """Test mean cost computation."""

    def test_known_trajectory_set(self) -> None:
        """Mean of [1.0, 2.0, 3.0, 4.0] = 2.5."""
        bundle = _make_bundle([1.0, 2.0, 3.0, 4.0])
        mean = compute_mean_cost(bundle)
        assert mean == pytest.approx(2.5)

    def test_single_trajectory(self) -> None:
        """Mean of a single trajectory is that trajectory's cost."""
        bundle = _make_bundle([7.5])
        assert compute_mean_cost(bundle) == pytest.approx(7.5)

    def test_single_element_bundle(self) -> None:
        """Single-element bundle mean is that element's cost."""
        bundle = _make_bundle([0.0])
        assert compute_mean_cost(bundle) == 0.0

    def test_with_importance_weights(self) -> None:
        """Weighted mean: Σ w_i * c_i."""
        costs = [10.0, 20.0]
        iw = [
            ImportanceWeight(sample_id=0, raw_weight=1.0, log_weight=0.0, normalised_weight=0.3),
            ImportanceWeight(sample_id=1, raw_weight=2.0, log_weight=0.69, normalised_weight=0.7),
        ]
        bundle = _make_bundle(costs, importance_weights=iw)
        expected = 0.3 * 10.0 + 0.7 * 20.0
        assert compute_mean_cost(bundle) == pytest.approx(expected)


# =====================================================================
# Cost variance
# =====================================================================

class TestCostVariance:
    """Test variance with Bessel correction."""

    def test_bessel_correction(self) -> None:
        """Variance of [2, 4, 4, 4, 5, 5, 7, 9] with Bessel correction."""
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        bundle = _make_bundle(data)
        var = compute_cost_variance(bundle)
        expected = np.var(data, ddof=1)
        assert var == pytest.approx(expected, rel=1e-8)

    def test_single_sample_zero_variance(self) -> None:
        """Single sample → variance = 0."""
        bundle = _make_bundle([5.0])
        assert compute_cost_variance(bundle) == 0.0

    def test_constant_values_zero_variance(self) -> None:
        """All same values → variance = 0."""
        bundle = _make_bundle([3.0, 3.0, 3.0, 3.0])
        assert compute_cost_variance(bundle) == pytest.approx(0.0, abs=1e-12)


# =====================================================================
# Cost quantiles
# =====================================================================

class TestCostQuantiles:
    """Test quantile estimation."""

    def test_50th_percentile_equals_median(self) -> None:
        """50th percentile should be the median."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bundle = _make_bundle(data)
        quantiles = compute_cost_quantiles(bundle, quantiles=(0.5,))
        assert quantiles[0.5] == pytest.approx(3.0)

    def test_edge_quantiles(self) -> None:
        """0th quantile = min, 100th = max."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        bundle = _make_bundle(data)
        q = compute_cost_quantiles(bundle, quantiles=(0.0, 1.0))
        assert q[0.0] == pytest.approx(10.0)
        assert q[1.0] == pytest.approx(50.0)

    def test_single_element_bundle(self) -> None:
        """Single-element bundle returns that element for all quantiles."""
        bundle = _make_bundle([5.0])
        q = compute_cost_quantiles(bundle, quantiles=(0.5,))
        assert q[0.5] == pytest.approx(5.0)


# =====================================================================
# Empirical CDF
# =====================================================================

class TestCostCDF:
    """Test empirical CDF computation."""

    def test_monotonically_non_decreasing(self) -> None:
        """CDF values should be monotonically non-decreasing."""
        data = [3.0, 1.0, 4.0, 1.5, 5.9, 2.6]
        bundle = _make_bundle(data)
        cost_vals, cdf_vals = compute_cost_cdf(bundle)
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i + 1] >= cdf_vals[i]

    def test_bounded_01(self) -> None:
        """CDF values should be in [0, 1]."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bundle = _make_bundle(data)
        _, cdf_vals = compute_cost_cdf(bundle)
        assert np.all(cdf_vals >= 0.0)
        assert np.all(cdf_vals <= 1.0)

    def test_cdf_ends_at_one(self) -> None:
        """Last CDF value should be 1.0."""
        data = [1.0, 2.0, 3.0]
        bundle = _make_bundle(data)
        _, cdf_vals = compute_cost_cdf(bundle)
        assert cdf_vals[-1] == pytest.approx(1.0)

    def test_single_element_cdf(self) -> None:
        """Single element → CDF is [1.0]."""
        bundle = _make_bundle([5.0])
        cost_vals, cdf_vals = compute_cost_cdf(bundle)
        assert len(cost_vals) == 1
        assert cdf_vals[-1] == pytest.approx(1.0)


# =====================================================================
# Tail risk
# =====================================================================

class TestTailRisk:
    """Test tail probability and CVaR."""

    def test_threshold_zero_is_one(self) -> None:
        """P(cost > 0) = 1.0 when all costs are positive."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bundle = _make_bundle(data)
        tail_prob, cvar = compute_tail_risk(bundle, threshold=0.0)
        assert tail_prob == pytest.approx(1.0)
        assert cvar == pytest.approx(3.0)  # mean of all

    def test_high_threshold_zero_tail(self) -> None:
        """P(cost > 100) = 0 when max cost is 5."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bundle = _make_bundle(data)
        tail_prob, cvar = compute_tail_risk(bundle, threshold=100.0)
        assert tail_prob == 0.0
        assert cvar == 0.0

    def test_partial_exceedance(self) -> None:
        """Correct tail probability for a specific threshold."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bundle = _make_bundle(data)
        tail_prob, cvar = compute_tail_risk(bundle, threshold=3.0)
        # 4.0 and 5.0 exceed 3.0 → P = 2/5 = 0.4
        assert tail_prob == pytest.approx(0.4)
        assert cvar == pytest.approx(4.5)  # mean of [4, 5]


# =====================================================================
# Path entropy
# =====================================================================

class TestPathEntropy:
    """Test Shannon entropy of empirical path distribution."""

    def test_non_negative(self) -> None:
        """Path entropy is always ≥ 0."""
        bundle = _make_bundle([1.0, 2.0, 3.0], lengths=[2, 3, 2])
        sequences = [("s0", "s1"), ("s0", "s1", "s2"), ("s0", "s1")]
        entropy = compute_path_entropy(bundle, state_sequences=sequences)
        assert entropy >= 0.0

    def test_single_unique_path_zero_entropy(self) -> None:
        """All identical paths → entropy = 0."""
        bundle = _make_bundle([1.0, 1.0, 1.0])
        sequences = [("s0", "s1"), ("s0", "s1"), ("s0", "s1")]
        entropy = compute_path_entropy(bundle, state_sequences=sequences)
        assert entropy == pytest.approx(0.0, abs=1e-12)

    def test_all_unique_paths_max_entropy(self) -> None:
        """All distinct paths → entropy = log(n)."""
        bundle = _make_bundle([1.0, 2.0, 3.0])
        sequences = [("s0",), ("s1",), ("s2",)]
        entropy = compute_path_entropy(bundle, state_sequences=sequences)
        assert entropy == pytest.approx(math.log(3), rel=1e-8)

    def test_length_based_fallback(self) -> None:
        """Without state sequences, uses length distribution."""
        bundle = _make_bundle([1.0, 2.0, 3.0], lengths=[2, 2, 3])
        entropy = compute_path_entropy(bundle)
        assert entropy >= 0.0


# =====================================================================
# Hitting time
# =====================================================================

class TestHittingTime:
    """Test first-passage time statistics."""

    def test_hitting_time_ge_1(self) -> None:
        """Hitting time uses 1-indexed steps, so ≥ 1."""
        bundle = _make_bundle([1.0, 1.0])
        seqs = [["s0", "s1", "goal"], ["s0", "goal"]]
        result = TrajectoryStatistics.compute_hitting_time(
            bundle, target_states={"goal"}, state_sequences=seqs
        )
        assert result["mean"] >= 1.0

    def test_mean_hitting_time(self) -> None:
        """Mean hitting time for known sequences."""
        bundle = _make_bundle([1.0, 1.0, 1.0])
        seqs = [
            ["s0", "s1", "goal"],   # hits at step 3
            ["s0", "goal"],          # hits at step 2
            ["s0", "s1", "s2", "goal"],  # hits at step 4
        ]
        result = TrajectoryStatistics.compute_hitting_time(
            bundle, target_states={"goal"}, state_sequences=seqs
        )
        # Steps are 1-indexed: 3, 2, 4
        assert result["mean"] == pytest.approx(3.0)
        assert result["median"] == pytest.approx(3.0)
        assert result["miss_rate"] == 0.0

    def test_miss_rate(self) -> None:
        """Trajectories that never hit target contribute to miss rate."""
        bundle = _make_bundle([1.0, 1.0])
        seqs = [
            ["s0", "s1"],         # never hits goal
            ["s0", "goal"],       # hits goal
        ]
        result = TrajectoryStatistics.compute_hitting_time(
            bundle, target_states={"goal"}, state_sequences=seqs
        )
        assert result["miss_rate"] == pytest.approx(0.5)

    def test_no_sequences_returns_defaults(self) -> None:
        """No state sequences → miss_rate = 1."""
        bundle = _make_bundle([1.0])
        result = TrajectoryStatistics.compute_hitting_time(
            bundle, target_states={"goal"}, state_sequences=None
        )
        assert result["miss_rate"] == 1.0
