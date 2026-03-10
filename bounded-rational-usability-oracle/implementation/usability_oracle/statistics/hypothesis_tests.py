"""
usability_oracle.statistics.hypothesis_tests — Statistical hypothesis tests
for usability regression detection.

Provides paired and non-parametric tests for comparing pre/post UI
measurements:

- PairedTTest: paired t-test
- WilcoxonSignedRank: distribution-free signed-rank test
- PermutationTest: exact/approximate permutation test
- BrunnerMunzel: robust to unequal variances
- BootstrapTest: bootstrap-based hypothesis test
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

from usability_oracle.statistics.types import (
    AlternativeHypothesis,
    BootstrapResult,
    ConfidenceInterval,
    EffectSize,
    EffectSizeType,
    HypothesisTestResult,
    PowerAnalysisResult,
    TestType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_arrays(
    x: Sequence[float], y: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sequences to numpy arrays and validate."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("Each sample must have at least 2 observations.")
    return a, b


def _scipy_alternative(alt: AlternativeHypothesis) -> str:
    """Map our enum to scipy's ``alternative`` string."""
    return {
        AlternativeHypothesis.TWO_SIDED: "two-sided",
        AlternativeHypothesis.GREATER: "greater",
        AlternativeHypothesis.LESS: "less",
    }[alt]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (mean diff / SD of diffs)."""
    diff = b - a
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff)) / sd


def _pooled_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using pooled standard deviation (independent samples)."""
    n1, n2 = len(a), len(b)
    var1, var2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0.0:
        return 0.0
    return (float(np.mean(b)) - float(np.mean(a))) / pooled_sd


def _interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _make_effect_size(a: np.ndarray, b: np.ndarray, paired: bool = True) -> EffectSize:
    """Compute Cohen's d effect size with bootstrap CI."""
    d = _cohens_d(a, b) if paired else _pooled_cohens_d(a, b)
    # Quick bootstrap CI for effect size
    rng = np.random.default_rng(42)
    n = len(a)
    boot_ds = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if paired:
            boot_ds.append(_cohens_d(a[idx], b[idx]))
        else:
            idx_b = rng.integers(0, len(b), size=len(b))
            boot_ds.append(_pooled_cohens_d(a[idx], b[idx_b]))
    boot_ds_arr = np.array(boot_ds)
    lo, hi = float(np.percentile(boot_ds_arr, 2.5)), float(np.percentile(boot_ds_arr, 97.5))
    ci = ConfidenceInterval(lower=lo, upper=hi, level=0.95, point_estimate=d,
                            method="bootstrap percentile")
    return EffectSize(
        measure=EffectSizeType.COHENS_D,
        value=d,
        ci=ci,
        interpretation=_interpret_d(d),
    )


def _mean_diff_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, alt: AlternativeHypothesis,
) -> ConfidenceInterval:
    """Confidence interval for the mean difference (paired)."""
    diff = b.astype(np.float64) - a.astype(np.float64)
    n = len(diff)
    mean_d = float(np.mean(diff))
    se = float(np.std(diff, ddof=1)) / math.sqrt(n)
    df = n - 1
    if alt == AlternativeHypothesis.TWO_SIDED:
        t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df))
        lo, hi = mean_d - t_crit * se, mean_d + t_crit * se
    elif alt == AlternativeHypothesis.GREATER:
        t_crit = float(sp_stats.t.ppf(1 - alpha, df))
        lo, hi = mean_d - t_crit * se, float("inf")
    else:
        t_crit = float(sp_stats.t.ppf(1 - alpha, df))
        lo, hi = float("-inf"), mean_d + t_crit * se
    return ConfidenceInterval(
        lower=lo, upper=hi, level=1 - alpha, point_estimate=mean_d, method="t-distribution",
    )


def _power_paired_t(
    effect_size: float, alpha: float, power: float,
) -> PowerAnalysisResult:
    """Power analysis for paired t-test using non-central t distribution."""
    # Search for minimum n
    for n in range(2, 100_000):
        df = n - 1
        ncp = effect_size * math.sqrt(n)
        if AlternativeHypothesis.TWO_SIDED:
            t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df))
            pwr = 1.0 - float(sp_stats.nct.cdf(t_crit, df, ncp)) + float(
                sp_stats.nct.cdf(-t_crit, df, ncp)
            )
        if pwr >= power:
            return PowerAnalysisResult(
                target_effect_size=effect_size,
                alpha=alpha,
                power=power,
                required_sample_size=n,
                actual_power=pwr,
                test_type=TestType.WELCH_T,
            )
    # Fallback
    return PowerAnalysisResult(
        target_effect_size=effect_size, alpha=alpha, power=power,
        required_sample_size=100_000, actual_power=pwr,
        test_type=TestType.WELCH_T,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PairedTTest
# ═══════════════════════════════════════════════════════════════════════════

class PairedTTest:
    """Paired t-test for pre/post UI comparison.

    Tests H₀: μ_diff = 0 for matched pairs (same user, before vs. after).

    The test statistic is:
        t = (d̄) / (s_d / √n)
    where d̄ is the mean of paired differences and s_d is their SD.
    """

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> HypothesisTestResult:
        a, b = _to_arrays(sample_a, sample_b)
        if len(a) != len(b):
            raise ValueError("Paired test requires equal-length samples.")
        result = sp_stats.ttest_rel(b, a, alternative=_scipy_alternative(alternative))
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        es = _make_effect_size(a, b, paired=True)
        ci = _mean_diff_ci(a, b, alpha, alternative)
        return HypothesisTestResult(
            test_type=TestType.WELCH_T,
            statistic=statistic,
            p_value=p_value,
            alternative=alternative,
            alpha=alpha,
            reject_null=p_value < alpha,
            effect_size=es,
            ci=ci,
            sample_size_a=len(a),
            sample_size_b=len(b),
            degrees_of_freedom=float(len(a) - 1),
        )

    def power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80,
    ) -> PowerAnalysisResult:
        return _power_paired_t(effect_size, alpha, power)

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        return _bootstrap_mean_diff(sample_a, sample_b, num_resamples, confidence_level, seed)


# ═══════════════════════════════════════════════════════════════════════════
# WilcoxonSignedRank
# ═══════════════════════════════════════════════════════════════════════════

class WilcoxonSignedRank:
    """Wilcoxon signed-rank test (non-parametric paired test).

    Distribution-free alternative to the paired t-test. Tests whether
    the distribution of paired differences is symmetric about zero.

    Uses exact computation for n ≤ 25 and normal approximation otherwise.
    """

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> HypothesisTestResult:
        a, b = _to_arrays(sample_a, sample_b)
        if len(a) != len(b):
            raise ValueError("Paired test requires equal-length samples.")
        diff = b - a
        # Remove zero differences for Wilcoxon
        nonzero_mask = diff != 0
        if nonzero_mask.sum() < 2:
            # Degenerate — no differences
            es = EffectSize(
                measure=EffectSizeType.COHENS_D, value=0.0,
                ci=ConfidenceInterval(0.0, 0.0, 0.95, 0.0, "N/A"),
                interpretation="negligible",
            )
            ci = ConfidenceInterval(0.0, 0.0, 1 - alpha, 0.0, "N/A")
            return HypothesisTestResult(
                test_type=TestType.MANN_WHITNEY_U,
                statistic=0.0, p_value=1.0, alternative=alternative,
                alpha=alpha, reject_null=False, effect_size=es, ci=ci,
                sample_size_a=len(a), sample_size_b=len(b),
            )
        method = "exact" if nonzero_mask.sum() <= 25 else "approx"
        result = sp_stats.wilcoxon(
            b, a, alternative=_scipy_alternative(alternative), method=method,
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        es = _make_effect_size(a, b, paired=True)
        ci = _mean_diff_ci(a, b, alpha, alternative)
        return HypothesisTestResult(
            test_type=TestType.MANN_WHITNEY_U,
            statistic=statistic,
            p_value=p_value,
            alternative=alternative,
            alpha=alpha,
            reject_null=p_value < alpha,
            effect_size=es,
            ci=ci,
            sample_size_a=len(a),
            sample_size_b=len(b),
        )

    def power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80,
    ) -> PowerAnalysisResult:
        # Asymptotic relative efficiency of Wilcoxon ~ 0.955 vs t-test
        result = _power_paired_t(effect_size, alpha, power)
        adjusted_n = math.ceil(result.required_sample_size / 0.955)
        return PowerAnalysisResult(
            target_effect_size=effect_size, alpha=alpha, power=power,
            required_sample_size=adjusted_n, actual_power=result.actual_power,
            test_type=TestType.MANN_WHITNEY_U,
        )

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        return _bootstrap_mean_diff(sample_a, sample_b, num_resamples, confidence_level, seed)


# ═══════════════════════════════════════════════════════════════════════════
# PermutationTest
# ═══════════════════════════════════════════════════════════════════════════

class PermutationTest:
    """Exact/approximate permutation test for paired differences.

    For n ≤ 20 pairs the test is exact (2ⁿ permutations); otherwise
    an approximate Monte Carlo permutation test is used.

    Test statistic: mean of paired differences.
    """

    def __init__(self, n_permutations: int = 10_000) -> None:
        self.n_permutations = n_permutations

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> HypothesisTestResult:
        a, b = _to_arrays(sample_a, sample_b)
        if len(a) != len(b):
            raise ValueError("Paired test requires equal-length samples.")
        diff = b - a
        n = len(diff)
        observed = float(np.mean(diff))

        if n <= 20:
            # Exact permutation
            count = 0
            total = 2 ** n
            for mask_int in range(total):
                signs = np.array(
                    [1 if (mask_int >> i) & 1 else -1 for i in range(n)],
                    dtype=np.float64,
                )
                perm_mean = float(np.mean(signs * diff))
                if alternative == AlternativeHypothesis.TWO_SIDED:
                    if abs(perm_mean) >= abs(observed) - 1e-12:
                        count += 1
                elif alternative == AlternativeHypothesis.GREATER:
                    if perm_mean >= observed - 1e-12:
                        count += 1
                else:
                    if perm_mean <= observed + 1e-12:
                        count += 1
            p_value = count / total
        else:
            # Monte Carlo permutation
            rng = np.random.default_rng(0)
            count = 0
            for _ in range(self.n_permutations):
                signs = rng.choice([-1.0, 1.0], size=n)
                perm_mean = float(np.mean(signs * diff))
                if alternative == AlternativeHypothesis.TWO_SIDED:
                    if abs(perm_mean) >= abs(observed) - 1e-12:
                        count += 1
                elif alternative == AlternativeHypothesis.GREATER:
                    if perm_mean >= observed - 1e-12:
                        count += 1
                else:
                    if perm_mean <= observed + 1e-12:
                        count += 1
            p_value = (count + 1) / (self.n_permutations + 1)

        es = _make_effect_size(a, b, paired=True)
        ci = _mean_diff_ci(a, b, alpha, alternative)
        return HypothesisTestResult(
            test_type=TestType.PERMUTATION,
            statistic=observed,
            p_value=p_value,
            alternative=alternative,
            alpha=alpha,
            reject_null=p_value < alpha,
            effect_size=es,
            ci=ci,
            sample_size_a=len(a),
            sample_size_b=len(b),
        )

    def power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80,
    ) -> PowerAnalysisResult:
        return _power_paired_t(effect_size, alpha, power)

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        return _bootstrap_mean_diff(sample_a, sample_b, num_resamples, confidence_level, seed)


# ═══════════════════════════════════════════════════════════════════════════
# BrunnerMunzel
# ═══════════════════════════════════════════════════════════════════════════

class BrunnerMunzel:
    """Brunner-Munzel test — non-parametric, robust to unequal variances.

    Tests H₀: P(X < Y) + 0.5·P(X = Y) = 0.5.  Does not assume equal
    variances or equal shapes of the two distributions.

    Reference: Brunner, E. & Munzel, U. (2000).
    """

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> HypothesisTestResult:
        a, b = _to_arrays(sample_a, sample_b)
        result = sp_stats.brunnermunzel(
            a, b, alternative=_scipy_alternative(alternative),
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        # Clamp p-value to [0, 1]
        p_value = max(0.0, min(1.0, p_value))
        es = _make_effect_size(
            a, b[:len(a)] if len(b) >= len(a) else np.pad(b, (0, len(a) - len(b))),
            paired=False,
        )
        # CI based on Hodges-Lehmann estimator (approximate)
        diff_all = np.subtract.outer(b, a).ravel()
        mean_diff = float(np.median(diff_all))
        se = float(np.std(diff_all, ddof=1)) / math.sqrt(len(diff_all))
        z = float(sp_stats.norm.ppf(1 - alpha / 2))
        ci = ConfidenceInterval(
            lower=mean_diff - z * se, upper=mean_diff + z * se,
            level=1 - alpha, point_estimate=mean_diff,
            method="Hodges-Lehmann",
        )
        return HypothesisTestResult(
            test_type=TestType.MANN_WHITNEY_U,
            statistic=statistic,
            p_value=p_value,
            alternative=alternative,
            alpha=alpha,
            reject_null=p_value < alpha,
            effect_size=es,
            ci=ci,
            sample_size_a=len(a),
            sample_size_b=len(b),
        )

    def power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80,
    ) -> PowerAnalysisResult:
        result = _power_paired_t(effect_size, alpha, power)
        return PowerAnalysisResult(
            target_effect_size=effect_size, alpha=alpha, power=power,
            required_sample_size=result.required_sample_size,
            actual_power=result.actual_power,
            test_type=TestType.MANN_WHITNEY_U,
        )

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        return _bootstrap_mean_diff(sample_a, sample_b, num_resamples, confidence_level, seed)


# ═══════════════════════════════════════════════════════════════════════════
# BootstrapTest
# ═══════════════════════════════════════════════════════════════════════════

class BootstrapTest:
    """Bootstrap-based hypothesis test.

    Generates a null distribution by resampling under H₀ (pooled data)
    and computes a p-value from the observed mean difference.

    The test statistic is the difference in sample means: X̄_B − X̄_A.
    """

    def __init__(self, n_bootstrap: int = 10_000) -> None:
        self.n_bootstrap = n_bootstrap

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> HypothesisTestResult:
        a, b = _to_arrays(sample_a, sample_b)
        na, nb = len(a), len(b)
        observed_diff = float(np.mean(b)) - float(np.mean(a))

        # Generate null distribution by pooling and resampling
        pooled = np.concatenate([a, b])
        rng = np.random.default_rng(42)
        null_diffs = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            perm = rng.permutation(pooled)
            null_diffs[i] = float(np.mean(perm[:nb])) - float(np.mean(perm[nb:]))

        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff) - 1e-12))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = float(np.mean(null_diffs >= observed_diff - 1e-12))
        else:
            p_value = float(np.mean(null_diffs <= observed_diff + 1e-12))
        p_value = max(1.0 / (self.n_bootstrap + 1), p_value)

        paired = len(a) == len(b)
        es = _make_effect_size(a, b, paired=paired)

        # Bootstrap CI for mean difference
        boot_diffs = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            ia = rng.integers(0, na, size=na)
            ib = rng.integers(0, nb, size=nb)
            boot_diffs[i] = float(np.mean(b[ib])) - float(np.mean(a[ia]))
        half_alpha = alpha / 2 if alternative == AlternativeHypothesis.TWO_SIDED else alpha
        lo = float(np.percentile(boot_diffs, 100 * half_alpha))
        hi = float(np.percentile(boot_diffs, 100 * (1 - half_alpha)))
        if alternative == AlternativeHypothesis.GREATER:
            hi = float("inf")
        elif alternative == AlternativeHypothesis.LESS:
            lo = float("-inf")
        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1 - alpha,
            point_estimate=observed_diff, method="bootstrap percentile",
        )
        return HypothesisTestResult(
            test_type=TestType.BOOTSTRAP,
            statistic=observed_diff,
            p_value=p_value,
            alternative=alternative,
            alpha=alpha,
            reject_null=p_value < alpha,
            effect_size=es,
            ci=ci,
            sample_size_a=na,
            sample_size_b=nb,
        )

    def power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80,
    ) -> PowerAnalysisResult:
        return _power_paired_t(effect_size, alpha, power)

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        return _bootstrap_mean_diff(sample_a, sample_b, num_resamples, confidence_level, seed)


# ---------------------------------------------------------------------------
# Shared bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_mean_diff(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    num_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Bootstrap the mean-difference statistic."""
    a, b = _to_arrays(sample_a, sample_b)
    n = min(len(a), len(b))
    observed = float(np.mean(b[:n])) - float(np.mean(a[:n]))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(num_resamples)
    for i in range(num_resamples):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = float(np.mean(b[idx])) - float(np.mean(a[idx]))
    boot_stats.sort()
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    bias = float(np.mean(boot_stats)) - observed
    se = float(np.std(boot_stats, ddof=1))
    ci = ConfidenceInterval(
        lower=lo, upper=hi, level=confidence_level,
        point_estimate=observed, method="bootstrap percentile",
    )
    return BootstrapResult(
        statistic_name="mean_difference",
        observed_statistic=observed,
        bootstrap_distribution=tuple(boot_stats.tolist()),
        ci=ci,
        bias=bias,
        standard_error=se,
        num_resamples=num_resamples,
        seed=seed,
    )
