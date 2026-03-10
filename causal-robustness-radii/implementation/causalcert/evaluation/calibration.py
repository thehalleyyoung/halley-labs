"""
Calibration assessment for CausalCert components.

Provides statistical tests and metrics for assessing whether CI tests,
fragility scores, robustness radii, and confidence intervals are
well-calibrated — i.e., whether stated probabilities match empirical
frequencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    CITestResult,
    FragilityScore,
    RobustnessRadius,
)


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CICalibrationResult:
    """Calibration assessment for CI test p-values.

    Attributes
    ----------
    nominal_alpha : float
        Significance level used.
    empirical_type_i_rate : float
        Observed false positive rate among truly independent pairs.
    n_tests : int
        Number of null tests evaluated.
    ks_statistic : float
        Kolmogorov-Smirnov statistic for uniformity of null p-values.
    ks_p_value : float
        KS test p-value.
    p_value_mean : float
        Mean of null p-values (expected ≈ 0.5).
    p_value_std : float
        Standard deviation of null p-values.
    binned_expected : list[float]
        Expected proportion in each calibration bin.
    binned_observed : list[float]
        Observed proportion in each calibration bin.
    """

    nominal_alpha: float = 0.05
    empirical_type_i_rate: float = 0.0
    n_tests: int = 0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    p_value_mean: float = 0.5
    p_value_std: float = 0.0
    binned_expected: list[float] = field(default_factory=list)
    binned_observed: list[float] = field(default_factory=list)


@dataclass(slots=True)
class FragilityCalibrationResult:
    """Calibration of fragility scores as predictors of actual fragility.

    Attributes
    ----------
    n_edges : int
        Number of edges evaluated.
    hosmer_lemeshow_stat : float
        Hosmer-Lemeshow goodness-of-fit statistic.
    hosmer_lemeshow_p : float
        p-value of the HL test.
    brier_score : float
        Brier score (mean squared error of predicted vs actual).
    binned_predicted : list[float]
        Mean predicted fragility in each calibration bin.
    binned_observed : list[float]
        Observed fragility rate in each bin.
    calibration_error : float
        Expected calibration error (ECE).
    """

    n_edges: int = 0
    hosmer_lemeshow_stat: float = 0.0
    hosmer_lemeshow_p: float = 1.0
    brier_score: float = 0.0
    binned_predicted: list[float] = field(default_factory=list)
    binned_observed: list[float] = field(default_factory=list)
    calibration_error: float = 0.0


@dataclass(slots=True)
class RadiusCalibrationResult:
    """Monte Carlo verification of reported robustness radius.

    Attributes
    ----------
    reported_radius : int
    mc_verified : bool
        ``True`` if MC samples confirm the radius.
    mc_lower : int
        Empirical lower bound from MC.
    mc_upper : int
        Empirical upper bound from MC.
    n_mc_samples : int
    fraction_overturned_at_radius : float
        Fraction of MC samples that overturn at the reported radius.
    fraction_survived_below : float
        Fraction that survived at distance radius-1.
    """

    reported_radius: int = 0
    mc_verified: bool = False
    mc_lower: int = 0
    mc_upper: int = 0
    n_mc_samples: int = 0
    fraction_overturned_at_radius: float = 0.0
    fraction_survived_below: float = 1.0


@dataclass(slots=True)
class CoverageCalibrationResult:
    """Coverage calibration: do CIs achieve nominal coverage?

    Attributes
    ----------
    nominal_level : float
    empirical_coverage : float
    n_intervals : int
    ks_statistic : float
        KS statistic for uniformity of centred residuals.
    ks_p_value : float
    mean_width : float
    median_width : float
    binned_nominal : list[float]
    binned_empirical : list[float]
    """

    nominal_level: float = 0.95
    empirical_coverage: float = 0.0
    n_intervals: int = 0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    mean_width: float = 0.0
    median_width: float = 0.0
    binned_nominal: list[float] = field(default_factory=list)
    binned_empirical: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CI test calibration
# ---------------------------------------------------------------------------


def ci_test_calibration(
    results: Sequence[CITestResult],
    true_independent: set[tuple[int, int, frozenset[int]]],
    *,
    n_bins: int = 10,
) -> CICalibrationResult:
    """Assess calibration of CI test p-values.

    Under the null hypothesis (true independence), p-values should be
    Uniform(0, 1).  This function computes the Type I error rate,
    the KS statistic for uniformity, and binned calibration data.

    Parameters
    ----------
    results : Sequence[CITestResult]
        All CI test results.
    true_independent : set[tuple[int, int, frozenset[int]]]
        ``(x, y, conditioning_set)`` triples that are truly independent.
    n_bins : int
        Number of bins for calibration plot data.

    Returns
    -------
    CICalibrationResult
    """
    null_pvals: list[float] = []
    alpha_val = 0.05

    for r in results:
        key = (r.x, r.y, r.conditioning_set)
        if key in true_independent:
            null_pvals.append(r.p_value)
            alpha_val = r.alpha

    cal = CICalibrationResult(nominal_alpha=alpha_val)

    if len(null_pvals) < 2:
        return cal

    arr = np.array(null_pvals)
    cal.n_tests = len(arr)
    cal.empirical_type_i_rate = float(np.mean(arr < alpha_val))
    cal.p_value_mean = float(np.mean(arr))
    cal.p_value_std = float(np.std(arr))

    # KS test against Uniform(0,1)
    ks_stat, ks_p = _ks_uniform(arr)
    cal.ks_statistic = ks_stat
    cal.ks_p_value = ks_p

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        expected_frac = hi - lo
        observed_frac = float(np.mean((arr >= lo) & (arr < hi)))
        cal.binned_expected.append(expected_frac)
        cal.binned_observed.append(observed_frac)

    return cal


# ---------------------------------------------------------------------------
# Fragility score calibration
# ---------------------------------------------------------------------------


def fragility_calibration(
    scores: Sequence[FragilityScore],
    true_fragile_edges: set[tuple[int, int]],
    *,
    n_bins: int = 10,
) -> FragilityCalibrationResult:
    """Assess whether fragility scores predict actual fragility.

    Computes the Hosmer-Lemeshow statistic, Brier score, and expected
    calibration error.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Predicted fragility scores.
    true_fragile_edges : set[tuple[int, int]]
        Ground-truth fragile edges.
    n_bins : int

    Returns
    -------
    FragilityCalibrationResult
    """
    cal = FragilityCalibrationResult()

    if not scores:
        return cal

    predicted = np.array([s.total_score for s in scores])
    actual = np.array([1.0 if s.edge in true_fragile_edges else 0.0
                       for s in scores])
    cal.n_edges = len(scores)

    # Brier score
    cal.brier_score = float(np.mean((predicted - actual) ** 2))

    # Binned calibration
    order = np.argsort(predicted)
    predicted = predicted[order]
    actual = actual[order]

    bin_size = max(1, len(predicted) // n_bins)
    hl_stat = 0.0
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_size
        hi = min((i + 1) * bin_size, len(predicted))
        if lo >= hi:
            break
        bin_pred = predicted[lo:hi]
        bin_actual = actual[lo:hi]
        mean_pred = float(np.mean(bin_pred))
        mean_obs = float(np.mean(bin_actual))
        cal.binned_predicted.append(mean_pred)
        cal.binned_observed.append(mean_obs)

        n_g = hi - lo
        # Hosmer-Lemeshow contribution
        if mean_pred > 0 and mean_pred < 1:
            expected_pos = n_g * mean_pred
            expected_neg = n_g * (1 - mean_pred)
            observed_pos = float(np.sum(bin_actual))
            observed_neg = n_g - observed_pos
            if expected_pos > 0:
                hl_stat += (observed_pos - expected_pos) ** 2 / expected_pos
            if expected_neg > 0:
                hl_stat += (observed_neg - expected_neg) ** 2 / expected_neg

        # ECE contribution
        ece += abs(mean_obs - mean_pred) * (n_g / len(predicted))

    cal.hosmer_lemeshow_stat = hl_stat
    cal.calibration_error = ece

    # HL p-value (chi-squared with n_bins - 2 degrees of freedom)
    df = max(1, min(len(cal.binned_predicted), n_bins) - 2)
    cal.hosmer_lemeshow_p = _chi2_survival(hl_stat, df)

    return cal


# ---------------------------------------------------------------------------
# Radius calibration
# ---------------------------------------------------------------------------


def radius_calibration(
    adj: AdjacencyMatrix,
    reported: RobustnessRadius,
    perturbation_fn: Callable[[AdjacencyMatrix, int], AdjacencyMatrix],
    check_fn: Callable[[AdjacencyMatrix], bool],
    *,
    n_mc: int = 1000,
    seed: int = 42,
) -> RadiusCalibrationResult:
    """Monte Carlo verification of a reported robustness radius.

    Draws *n_mc* random perturbations at each distance from 1 to
    ``reported.upper_bound + 1`` and checks whether the causal conclusion
    flips.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG.
    reported : RobustnessRadius
        The reported radius to verify.
    perturbation_fn : callable
        ``perturbation_fn(adj, k)`` → a randomly perturbed DAG at distance *k*.
    check_fn : callable
        ``check_fn(adj)`` → ``True`` if the conclusion still holds.
    n_mc : int
        Number of Monte Carlo samples per distance level.
    seed : int

    Returns
    -------
    RadiusCalibrationResult
    """
    adj = np.asarray(adj, dtype=np.int8)
    rng = np.random.RandomState(seed)
    cal = RadiusCalibrationResult(
        reported_radius=reported.upper_bound,
        n_mc_samples=n_mc,
    )

    max_d = reported.upper_bound + 1
    survived: dict[int, int] = defaultdict(int)
    overturned: dict[int, int] = defaultdict(int)

    for d in range(1, max_d + 1):
        for _ in range(n_mc):
            perturbed = perturbation_fn(adj, d)
            if check_fn(perturbed):
                survived[d] += 1
            else:
                overturned[d] += 1

    # All samples at distance < radius should survive (ideally)
    r = reported.upper_bound
    if r >= 1:
        below_survived = sum(survived.get(d, 0) for d in range(1, r))
        below_total = sum(survived.get(d, 0) + overturned.get(d, 0)
                          for d in range(1, r))
        cal.fraction_survived_below = (
            below_survived / below_total if below_total > 0 else 1.0
        )

    # At the reported radius, some should overturn
    at_radius_total = survived.get(r, 0) + overturned.get(r, 0)
    cal.fraction_overturned_at_radius = (
        overturned.get(r, 0) / at_radius_total if at_radius_total > 0 else 0.0
    )

    # Determine MC bounds
    mc_lower = max_d + 1
    for d in range(1, max_d + 1):
        if overturned.get(d, 0) > 0:
            mc_lower = d
            break

    mc_upper = mc_lower
    for d in range(mc_lower, max_d + 1):
        if overturned.get(d, 0) > 0:
            mc_upper = d

    cal.mc_lower = mc_lower
    cal.mc_upper = mc_upper
    cal.mc_verified = mc_lower == r

    return cal


# ---------------------------------------------------------------------------
# Coverage calibration
# ---------------------------------------------------------------------------


def coverage_calibration(
    true_values: Sequence[float],
    ci_lowers: Sequence[float],
    ci_uppers: Sequence[float],
    *,
    nominal_levels: Sequence[float] | None = None,
    n_bins: int = 10,
) -> CoverageCalibrationResult:
    """Assess whether confidence intervals achieve nominal coverage.

    Parameters
    ----------
    true_values : Sequence[float]
        Ground-truth values.
    ci_lowers, ci_uppers : Sequence[float]
        Lower and upper CI bounds.
    nominal_levels : Sequence[float] | None
        Nominal coverage levels to evaluate. Defaults to 0.95.
    n_bins : int

    Returns
    -------
    CoverageCalibrationResult
    """
    true_arr = np.asarray(true_values, dtype=float)
    lo_arr = np.asarray(ci_lowers, dtype=float)
    hi_arr = np.asarray(ci_uppers, dtype=float)
    n = len(true_arr)

    if nominal_levels is None:
        nominal_levels = [0.95]
    level = nominal_levels[0]

    cal = CoverageCalibrationResult(
        nominal_level=level,
        n_intervals=n,
    )

    if n == 0:
        return cal

    covered = (true_arr >= lo_arr) & (true_arr <= hi_arr)
    cal.empirical_coverage = float(np.mean(covered))

    widths = hi_arr - lo_arr
    cal.mean_width = float(np.mean(widths))
    cal.median_width = float(np.median(widths))

    # KS test: if CI is well-calibrated, the quantile of true_val within
    # each interval should be ~ Uniform
    midpoints = (lo_arr + hi_arr) / 2.0
    half_widths = widths / 2.0
    half_widths[half_widths == 0] = 1.0
    quantiles = (true_arr - midpoints) / half_widths  # in [-1, 1]
    # Transform to [0, 1]
    u = (quantiles + 1.0) / 2.0
    u = np.clip(u, 0.0, 1.0)
    ks_stat, ks_p = _ks_uniform(u)
    cal.ks_statistic = ks_stat
    cal.ks_p_value = ks_p

    # Multi-level calibration
    if len(nominal_levels) > 1:
        for lev in nominal_levels:
            # Shrink CI to lev fraction
            center = midpoints
            hw = half_widths * (lev / level) if level > 0 else half_widths
            cov_lev = float(np.mean(
                (true_arr >= center - hw) & (true_arr <= center + hw)
            ))
            cal.binned_nominal.append(lev)
            cal.binned_empirical.append(cov_lev)
    else:
        # Single-level: binned by predicted width
        order = np.argsort(widths)
        bin_size = max(1, n // n_bins)
        for i in range(n_bins):
            lo_idx = i * bin_size
            hi_idx = min((i + 1) * bin_size, n)
            if lo_idx >= hi_idx:
                break
            bin_covered = covered[order[lo_idx:hi_idx]]
            cal.binned_nominal.append(level)
            cal.binned_empirical.append(float(np.mean(bin_covered)))

    return cal


# ---------------------------------------------------------------------------
# Calibration plot data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CalibrationPlotData:
    """Data for a calibration plot.

    Attributes
    ----------
    predicted : list[float]
        Predicted probabilities / scores.
    observed : list[float]
        Observed frequencies.
    n_per_bin : list[int]
        Sample size in each bin.
    ece : float
        Expected calibration error.
    mce : float
        Maximum calibration error.
    """

    predicted: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    n_per_bin: list[int] = field(default_factory=list)
    ece: float = 0.0
    mce: float = 0.0


def calibration_plot_data(
    predicted: Sequence[float],
    actual: Sequence[int | float],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationPlotData:
    """Generate data for a calibration (reliability) plot.

    Parameters
    ----------
    predicted : Sequence[float]
        Predicted probabilities in [0, 1].
    actual : Sequence[int | float]
        Binary outcomes (0 or 1).
    n_bins : int
    strategy : str
        ``"uniform"`` for equally spaced bins, ``"quantile"`` for equal-count.

    Returns
    -------
    CalibrationPlotData
    """
    pred_arr = np.asarray(predicted, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    n = len(pred_arr)
    result = CalibrationPlotData()

    if n == 0:
        return result

    if strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(pred_arr, quantiles)
        bin_edges[-1] += 1e-10
    else:
        bin_edges = np.linspace(0.0, 1.0 + 1e-10, n_bins + 1)

    total_ece = 0.0
    max_ce = 0.0
    for i in range(n_bins):
        mask = (pred_arr >= bin_edges[i]) & (pred_arr < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(pred_arr[mask].mean())
        mean_obs = float(act_arr[mask].mean())
        result.predicted.append(mean_pred)
        result.observed.append(mean_obs)
        result.n_per_bin.append(count)
        ce = abs(mean_obs - mean_pred)
        total_ece += ce * (count / n)
        max_ce = max(max_ce, ce)

    result.ece = total_ece
    result.mce = max_ce
    return result


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def hosmer_lemeshow_test(
    predicted: Sequence[float],
    actual: Sequence[int | float],
    *,
    n_groups: int = 10,
) -> dict[str, float]:
    """Hosmer-Lemeshow goodness-of-fit test.

    Tests whether observed event rates match predicted probabilities
    across decile groups.

    Parameters
    ----------
    predicted : Sequence[float]
    actual : Sequence[int | float]
    n_groups : int

    Returns
    -------
    dict[str, float]
        ``statistic``, ``p_value``, ``df``.
    """
    pred_arr = np.asarray(predicted, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    n = len(pred_arr)

    if n < n_groups * 2:
        return {"statistic": 0.0, "p_value": 1.0, "df": max(1, n_groups - 2)}

    order = np.argsort(pred_arr)
    pred_sorted = pred_arr[order]
    act_sorted = act_arr[order]

    group_size = n // n_groups
    hl = 0.0
    for g in range(n_groups):
        lo = g * group_size
        hi = (g + 1) * group_size if g < n_groups - 1 else n
        n_g = hi - lo
        if n_g == 0:
            continue
        o_1 = float(act_sorted[lo:hi].sum())
        o_0 = n_g - o_1
        e_1 = float(pred_sorted[lo:hi].sum())
        e_0 = n_g - e_1
        if e_1 > 0:
            hl += (o_1 - e_1) ** 2 / e_1
        if e_0 > 0:
            hl += (o_0 - e_0) ** 2 / e_0

    df = max(1, n_groups - 2)
    p_val = _chi2_survival(hl, df)
    return {"statistic": hl, "p_value": p_val, "df": float(df)}


def kolmogorov_smirnov_test(
    sample: Sequence[float],
) -> dict[str, float]:
    """One-sample KS test against Uniform(0,1).

    Parameters
    ----------
    sample : Sequence[float]
        Values expected to be Uniform(0,1) under the null.

    Returns
    -------
    dict[str, float]
        ``statistic``, ``p_value``.
    """
    arr = np.asarray(sample, dtype=float)
    if len(arr) < 2:
        return {"statistic": 0.0, "p_value": 1.0}
    ks, p = _ks_uniform(arr)
    return {"statistic": ks, "p_value": p}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ks_uniform(arr: np.ndarray) -> tuple[float, float]:
    """KS test against Uniform(0,1). Returns (statistic, p_value)."""
    n = len(arr)
    if n < 2:
        return 0.0, 1.0
    sorted_vals = np.sort(arr)
    ecdf = np.arange(1, n + 1) / n
    d_plus = float(np.max(ecdf - sorted_vals))
    d_minus = float(np.max(sorted_vals - np.arange(0, n) / n))
    ks = max(d_plus, d_minus)

    # Approximate p-value via the asymptotic formula
    z = ks * math.sqrt(n)
    if z < 0.27:
        p = 1.0
    elif z < 1.0:
        p = 1.0 - 2.0 * sum(
            (-1) ** (k - 1) * math.exp(-2.0 * k * k * z * z)
            for k in range(1, 51)
        )
        p = max(0.0, min(1.0, p))
    else:
        p = 2.0 * math.exp(-2.0 * z * z)
        p = max(0.0, min(1.0, p))
    return ks, p


def _chi2_survival(x: float, df: int) -> float:
    """Approximate chi-squared survival function P(X > x).

    Uses the Wilson-Hilferty normal approximation.
    """
    if df <= 0 or x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
        2.0 / (9.0 * df)
    )
    # Standard normal CDF approximation
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, p))
