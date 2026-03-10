"""Coverage verification for causal inference estimators.

Implements Monte Carlo coverage simulation, confidence interval coverage
rates, fragility score calibration, radius accuracy with ground truth,
p-value uniformity testing, bootstrap coverage calibration, and
sample-size requirements computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class CoverageResult:
    """Result of a coverage simulation."""
    nominal_level: float
    empirical_coverage: float
    n_simulations: int
    mean_width: float
    median_width: float
    coverage_se: float
    per_dgp: Optional[List[float]] = None


@dataclass
class CalibrationResult:
    """Calibration assessment result."""
    expected_levels: np.ndarray
    observed_rates: np.ndarray
    calibration_error: float
    max_deviation: float
    kolmogorov_smirnov_p: float = 0.0


@dataclass
class SampleSizeResult:
    """Sample size determination result."""
    target_coverage: float
    target_width: float
    required_n: int
    simulated_coverage: float
    simulated_width: float


# ===================================================================
# DGP protocol
# ===================================================================

DGPFn = Callable[[int, np.random.RandomState], Tuple[np.ndarray, float]]
EstimatorFn = Callable[[np.ndarray], Tuple[float, float, float, float]]


# ===================================================================
# 1.  Monte Carlo coverage simulation
# ===================================================================

def monte_carlo_coverage(
    dgp_fn: DGPFn,
    estimator_fn: EstimatorFn,
    *,
    n_simulations: int = 200,
    sample_size: int = 500,
    nominal_level: float = 0.95,
    seed: int = 42,
) -> CoverageResult:
    """Monte Carlo coverage simulation.

    Parameters
    ----------
    dgp_fn : callable
        ``(n, rng) -> (data_array, true_parameter)``
    estimator_fn : callable
        ``(data_array) -> (estimate, se, ci_lower, ci_upper)``
    n_simulations : int
        Number of DGPs / replications.
    sample_size : int
        Sample size per simulation.
    nominal_level : float
        Nominal coverage level (e.g., 0.95).

    Returns
    -------
    CoverageResult
    """
    rng = np.random.RandomState(seed)
    covers = np.zeros(n_simulations, dtype=bool)
    widths = np.zeros(n_simulations)

    for i in range(n_simulations):
        data, truth = dgp_fn(sample_size, rng)
        est, se, lo, hi = estimator_fn(data)
        covers[i] = lo <= truth <= hi
        widths[i] = hi - lo

    emp_cov = float(np.mean(covers))
    cov_se = math.sqrt(emp_cov * (1 - emp_cov) / n_simulations)

    return CoverageResult(
        nominal_level=nominal_level,
        empirical_coverage=emp_cov,
        n_simulations=n_simulations,
        mean_width=float(np.mean(widths)),
        median_width=float(np.median(widths)),
        coverage_se=cov_se,
    )


def multi_dgp_coverage(
    dgp_fns: List[DGPFn],
    estimator_fn: EstimatorFn,
    *,
    n_per_dgp: int = 100,
    sample_size: int = 500,
    nominal_level: float = 0.95,
    seed: int = 42,
) -> CoverageResult:
    """Coverage across multiple DGPs.

    Runs *n_per_dgp* simulations for each DGP in *dgp_fns* and
    reports aggregate and per-DGP coverage.
    """
    rng = np.random.RandomState(seed)
    all_covers: List[bool] = []
    all_widths: List[float] = []
    per_dgp_cov: List[float] = []

    for dgp_fn in dgp_fns:
        covers = np.zeros(n_per_dgp, dtype=bool)
        widths = np.zeros(n_per_dgp)
        for i in range(n_per_dgp):
            data, truth = dgp_fn(sample_size, rng)
            est, se, lo, hi = estimator_fn(data)
            covers[i] = lo <= truth <= hi
            widths[i] = hi - lo
        per_dgp_cov.append(float(np.mean(covers)))
        all_covers.extend(covers.tolist())
        all_widths.extend(widths.tolist())

    n_total = len(all_covers)
    emp_cov = float(np.mean(all_covers))
    cov_se = math.sqrt(emp_cov * (1 - emp_cov) / n_total)

    return CoverageResult(
        nominal_level=nominal_level,
        empirical_coverage=emp_cov,
        n_simulations=n_total,
        mean_width=float(np.mean(all_widths)),
        median_width=float(np.median(all_widths)),
        coverage_se=cov_se,
        per_dgp=per_dgp_cov,
    )


# ===================================================================
# 2.  CI coverage rate computation
# ===================================================================

def ci_coverage_rate(
    estimates: np.ndarray,
    ci_lowers: np.ndarray,
    ci_uppers: np.ndarray,
    true_value: float,
) -> float:
    """Compute the coverage rate from arrays of CIs."""
    covers = (ci_lowers <= true_value) & (true_value <= ci_uppers)
    return float(np.mean(covers))


def ci_width_statistics(
    ci_lowers: np.ndarray,
    ci_uppers: np.ndarray,
) -> Dict[str, float]:
    """Summary statistics for CI widths."""
    widths = ci_uppers - ci_lowers
    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "std_width": float(np.std(widths)),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
    }


def coverage_by_stratum(
    estimates: np.ndarray,
    ci_lowers: np.ndarray,
    ci_uppers: np.ndarray,
    true_values: np.ndarray,
    strata: np.ndarray,
) -> Dict[int, float]:
    """Coverage rate broken down by stratum."""
    result: Dict[int, float] = {}
    for s in np.unique(strata):
        mask = strata == s
        covers = (ci_lowers[mask] <= true_values[mask]) & (true_values[mask] <= ci_uppers[mask])
        result[int(s)] = float(np.mean(covers))
    return result


# ===================================================================
# 3.  Fragility score calibration assessment
# ===================================================================

def fragility_calibration(
    fragility_scores: np.ndarray,
    conclusion_flipped: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    """Assess calibration of fragility scores.

    If fragility = k, we expect that among instances with fragility k,
    roughly 1/2^k should have their conclusion flipped under random
    perturbation.  This function bins fragility scores and checks
    whether the observed flip rate aligns with the predicted rate.
    """
    bins = np.linspace(
        float(np.min(fragility_scores)) - 0.5,
        float(np.max(fragility_scores)) + 0.5,
        n_bins + 1,
    )
    expected: List[float] = []
    observed: List[float] = []

    for i in range(n_bins):
        mask = (fragility_scores >= bins[i]) & (fragility_scores < bins[i + 1])
        if mask.sum() == 0:
            continue
        mean_frag = float(np.mean(fragility_scores[mask]))
        exp_flip = 1.0 / max(2.0 ** mean_frag, 1.0)
        obs_flip = float(np.mean(conclusion_flipped[mask]))
        expected.append(exp_flip)
        observed.append(obs_flip)

    expected_arr = np.array(expected)
    observed_arr = np.array(observed)

    if len(expected) == 0:
        return CalibrationResult(
            expected_levels=expected_arr,
            observed_rates=observed_arr,
            calibration_error=0.0,
            max_deviation=0.0,
        )

    cal_error = float(np.mean((expected_arr - observed_arr) ** 2))
    max_dev = float(np.max(np.abs(expected_arr - observed_arr)))

    return CalibrationResult(
        expected_levels=expected_arr,
        observed_rates=observed_arr,
        calibration_error=cal_error,
        max_deviation=max_dev,
    )


# ===================================================================
# 4.  Radius accuracy with exact ground truth
# ===================================================================

def radius_accuracy(
    estimated_radii: np.ndarray,
    true_radii: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy metrics for estimated robustness radii.

    Returns MAE, RMSE, bias, and exact-match rate.
    """
    diff = estimated_radii - true_radii
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "exact_match_rate": float(np.mean(estimated_radii == true_radii)),
        "underestimate_rate": float(np.mean(estimated_radii < true_radii)),
        "overestimate_rate": float(np.mean(estimated_radii > true_radii)),
        "max_error": float(np.max(np.abs(diff))),
        "median_error": float(np.median(np.abs(diff))),
    }


def radius_by_graph_size(
    estimated_radii: np.ndarray,
    true_radii: np.ndarray,
    graph_sizes: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Radius accuracy broken down by graph size."""
    result: Dict[int, Dict[str, float]] = {}
    for size in np.unique(graph_sizes):
        mask = graph_sizes == size
        result[int(size)] = radius_accuracy(
            estimated_radii[mask], true_radii[mask]
        )
    return result


# ===================================================================
# 5.  P-value uniformity testing
# ===================================================================

def p_value_uniformity(
    p_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Test whether p-values are uniformly distributed under H₀.

    Uses Kolmogorov-Smirnov test against U[0,1].
    """
    n = len(p_values)
    ks_stat, ks_p = sp_stats.kstest(p_values, "uniform")

    expected_frac = alpha
    observed_reject = float(np.mean(p_values < alpha))

    chi2_bins = 10
    observed_counts, _ = np.histogram(p_values, bins=chi2_bins, range=(0, 1))
    expected_count = n / chi2_bins
    chi2_stat = float(np.sum((observed_counts - expected_count) ** 2 / expected_count))
    chi2_p = 1.0 - sp_stats.chi2.cdf(chi2_stat, df=chi2_bins - 1)

    return {
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "is_uniform_ks": float(ks_p) > alpha,
        "chi2_statistic": chi2_stat,
        "chi2_p_value": chi2_p,
        "is_uniform_chi2": chi2_p > alpha,
        "observed_rejection_rate": observed_reject,
        "expected_rejection_rate": expected_frac,
        "rejection_ratio": observed_reject / max(expected_frac, 1e-12),
    }


# ===================================================================
# 6.  Bootstrap coverage calibration
# ===================================================================

def bootstrap_coverage_calibration(
    data: np.ndarray,
    estimator_fn: Callable[[np.ndarray], Tuple[float, float]],
    true_value: float,
    *,
    n_outer: int = 200,
    n_inner: int = 500,
    nominal_levels: Optional[List[float]] = None,
    seed: int = 42,
) -> CalibrationResult:
    """Calibrate bootstrap confidence intervals.

    For each nominal level γ, computes the fraction of times the
    bootstrap CI at level γ contains the true value.  Good calibration
    means observed ≈ nominal.
    """
    if nominal_levels is None:
        nominal_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    rng = np.random.RandomState(seed)
    n = data.shape[0]

    coverage_counts = {lev: 0 for lev in nominal_levels}

    for _ in range(n_outer):
        idx = rng.choice(n, size=n, replace=True)
        sample = data[idx]

        boot_estimates = np.empty(n_inner)
        for b in range(n_inner):
            boot_idx = rng.choice(n, size=n, replace=True)
            est, _ = estimator_fn(sample[boot_idx])
            boot_estimates[b] = est

        for lev in nominal_levels:
            alpha = 1.0 - lev
            lo = np.percentile(boot_estimates, 100 * alpha / 2)
            hi = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
            if lo <= true_value <= hi:
                coverage_counts[lev] += 1

    expected = np.array(nominal_levels)
    observed = np.array([coverage_counts[lev] / n_outer for lev in nominal_levels])

    cal_error = float(np.mean((expected - observed) ** 2))
    max_dev = float(np.max(np.abs(expected - observed)))

    return CalibrationResult(
        expected_levels=expected,
        observed_rates=observed,
        calibration_error=cal_error,
        max_deviation=max_dev,
    )


# ===================================================================
# 7.  Sample size requirements
# ===================================================================

def sample_size_for_coverage(
    dgp_fn: DGPFn,
    estimator_fn: EstimatorFn,
    *,
    target_coverage: float = 0.95,
    target_width: float = 0.5,
    n_range: Tuple[int, int] = (50, 5000),
    n_simulations: int = 100,
    seed: int = 42,
    bisection_tol: int = 20,
) -> SampleSizeResult:
    """Determine the sample size needed for target coverage and width.

    Uses bisection search over *n_range*.
    """
    rng_base = np.random.RandomState(seed)

    def _evaluate(n_sample: int) -> Tuple[float, float]:
        rng = np.random.RandomState(seed)
        covers = 0
        total_width = 0.0
        for _ in range(n_simulations):
            data, truth = dgp_fn(n_sample, rng)
            est, se, lo, hi = estimator_fn(data)
            if lo <= truth <= hi:
                covers += 1
            total_width += (hi - lo)
        return covers / n_simulations, total_width / n_simulations

    lo_n, hi_n = n_range
    best_n = hi_n
    best_cov = 0.0
    best_width = float("inf")

    while hi_n - lo_n > bisection_tol:
        mid_n = (lo_n + hi_n) // 2
        cov, width = _evaluate(mid_n)
        if cov >= target_coverage and width <= target_width:
            best_n = mid_n
            best_cov = cov
            best_width = width
            hi_n = mid_n
        else:
            lo_n = mid_n

    if best_cov < target_coverage or best_width > target_width:
        cov, width = _evaluate(hi_n)
        best_n = hi_n
        best_cov = cov
        best_width = width

    return SampleSizeResult(
        target_coverage=target_coverage,
        target_width=target_width,
        required_n=best_n,
        simulated_coverage=best_cov,
        simulated_width=best_width,
    )


def power_analysis(
    dgp_fn: DGPFn,
    estimator_fn: EstimatorFn,
    *,
    sample_sizes: Optional[List[int]] = None,
    n_simulations: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[int, float]:
    """Power analysis: rejection rate as a function of sample size."""
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500, 1000, 2000]

    rng = np.random.RandomState(seed)
    result: Dict[int, float] = {}

    for n_sample in sample_sizes:
        rejections = 0
        for _ in range(n_simulations):
            data, truth = dgp_fn(n_sample, rng)
            est, se, lo, hi = estimator_fn(data)
            if lo > 0 or hi < 0:
                rejections += 1
        result[n_sample] = rejections / n_simulations

    return result
