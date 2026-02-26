"""
Statistical analysis utilities for CoaCert-TLA evaluation.

Provides confidence intervals, effect sizes, and significance tests
for experiment results, using only the standard library (no scipy).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

@dataclass
class StatisticalSummary:
    """Summary statistics for a set of measurements.

    Computes: mean, median, std dev, 95 % confidence interval,
    min, max, and sample size.
    """
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
            "min": self.min_val,
            "max": self.max_val,
            "n": self.n,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compute_summary(values: Sequence[float]) -> StatisticalSummary:
    """Compute a StatisticalSummary from a sequence of measurements.

    Parameters
    ----------
    values : sequence of float
        Raw measurement values.

    Returns
    -------
    StatisticalSummary
    """
    s = StatisticalSummary()
    vals = list(values)
    s.values = vals
    s.n = len(vals)
    if s.n == 0:
        return s

    s.mean = statistics.mean(vals)
    s.median = statistics.median(vals)
    s.min_val = min(vals)
    s.max_val = max(vals)

    if s.n >= 2:
        s.std_dev = statistics.stdev(vals)
        se = s.std_dev / math.sqrt(s.n)
        # 95% CI: use t-critical for n-1 df
        t_crit = _t_critical_95(s.n - 1)
        s.ci_lower = s.mean - t_crit * se
        s.ci_upper = s.mean + t_crit * se
    else:
        s.ci_lower = s.mean
        s.ci_upper = s.mean

    return s


# ---------------------------------------------------------------------------
# Effect size (Cohen's d)
# ---------------------------------------------------------------------------

@dataclass
class EffectSizeResult:
    """Result of Cohen's d effect size computation."""
    cohens_d: float = 0.0
    magnitude: str = "negligible"  # negligible, small, medium, large

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohens_d": self.cohens_d,
            "magnitude": self.magnitude,
        }


def cohens_d(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> EffectSizeResult:
    """Compute Cohen's d effect size between two independent samples.

    d = (mean_a - mean_b) / pooled_std

    Magnitude thresholds (Cohen, 1988):
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| >= 0.8 → large

    Parameters
    ----------
    sample_a, sample_b : sequences of float

    Returns
    -------
    EffectSizeResult
    """
    a = list(sample_a)
    b = list(sample_b)
    n_a, n_b = len(a), len(b)
    result = EffectSizeResult()

    if n_a < 2 or n_b < 2:
        return result

    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)

    pooled_std = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled_std <= 0:
        return result

    result.cohens_d = (mean_a - mean_b) / pooled_std

    d_abs = abs(result.cohens_d)
    if d_abs < 0.2:
        result.magnitude = "negligible"
    elif d_abs < 0.5:
        result.magnitude = "small"
    elif d_abs < 0.8:
        result.magnitude = "medium"
    else:
        result.magnitude = "large"

    return result


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------

@dataclass
class TTestResult:
    """Result of Welch's t-test."""
    t_statistic: float = 0.0
    p_value: float = 1.0
    degrees_of_freedom: float = 0.0
    significant: bool = False
    alpha: float = 0.05
    mean_diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    effect_size: Optional[EffectSizeResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "significant": self.significant,
            "alpha": self.alpha,
            "mean_diff": self.mean_diff,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
        }
        if self.effect_size is not None:
            d["effect_size"] = self.effect_size.to_dict()
        return d


def welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
) -> TTestResult:
    """Perform Welch's t-test for independent samples.

    Tests H0: mean_a == mean_b vs H1: mean_a != mean_b (two-tailed).

    Parameters
    ----------
    sample_a, sample_b : sequences of float
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    TTestResult
    """
    a = list(sample_a)
    b = list(sample_b)
    n_a, n_b = len(a), len(b)
    result = TTestResult(alpha=alpha)

    if n_a < 2 or n_b < 2:
        return result

    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)

    se_sq = var_a / n_a + var_b / n_b
    if se_sq <= 0:
        return result
    se = math.sqrt(se_sq)

    result.t_statistic = (mean_a - mean_b) / se
    result.mean_diff = mean_a - mean_b

    # Welch-Satterthwaite degrees of freedom
    num = se_sq ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if den <= 0:
        return result
    result.degrees_of_freedom = num / den

    # Two-tailed p-value
    result.p_value = _t_pvalue(abs(result.t_statistic), result.degrees_of_freedom)
    result.significant = result.p_value < alpha

    # 95% CI for the difference
    t_crit = _t_critical_95(result.degrees_of_freedom)
    result.ci_lower = result.mean_diff - t_crit * se
    result.ci_upper = result.mean_diff + t_crit * se

    # Effect size
    result.effect_size = cohens_d(a, b)

    return result


# ---------------------------------------------------------------------------
# Internal t-distribution helpers (no scipy)
# ---------------------------------------------------------------------------

def _t_critical_95(df: float) -> float:
    """Approximate 95% two-tailed t-critical value."""
    if df > 1000:
        return 1.96
    if df <= 1:
        return 12.706
    # Approximation via Wilson-Hilferty transform
    z = 1.96  # z_{0.025}
    g1 = z ** 3 + z
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    t_val = z + g1 / (4 * df) + g2 / df ** 2
    return max(t_val, 1.96)


def _t_pvalue(t_abs: float, df: float) -> float:
    """Approximate two-tailed p-value for |t| with df degrees of freedom."""
    if df <= 0 or math.isnan(t_abs):
        return 1.0
    if df > 1000:
        return 2.0 * (1.0 - _normal_cdf(t_abs))

    x = df / (df + t_abs * t_abs)
    a, b = df / 2.0, 0.5
    ibeta = _regularized_beta(x, a, b)
    return max(0.0, min(1.0, ibeta))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - d * math.exp(-0.5 * x * x) * poly
    return cdf if x >= 0 else 1.0 - cdf


def _regularized_beta(
    x: float, a: float, b: float, max_iter: int = 200
) -> float:
    """Regularized incomplete beta function I_x(a,b) via Lentz CF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a, max_iter)

    ln_prefix = (
        a * math.log(max(x, 1e-300))
        + b * math.log(max(1.0 - x, 1e-300))
        - math.log(a)
        - _log_beta(a, b)
    )
    if ln_prefix < -500:
        return 0.0
    prefix = math.exp(ln_prefix)

    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(max_iter):
        if m == 0:
            a_m = 1.0
        elif m % 2 == 1:
            k = (m - 1) // 2 + 1
            a_m = k * (b - k) * x / ((a + 2 * k - 1) * (a + 2 * k))
        else:
            k = m // 2
            a_m = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1))

        d = 1.0 + a_m * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + a_m / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break

    return max(0.0, min(1.0, prefix * f))


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function B(a,b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
