"""Hypothesis testing framework for ML pipeline leakage auditing.

Provides a hierarchy of hypothesis tests specifically designed for
determining whether detected leakage is statistically significant,
comparing pipeline variants, and combining evidence across features
and pipeline stages.  All implementations use only the Python standard
library.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Mathematical helpers (duplicated from independence to keep modules standalone)
# ---------------------------------------------------------------------------

def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26 approximation."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * math.sqrt(2.0))))


def _normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Rational approximation of the normal inverse CDF."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return mu

    if p < 0.5:
        sign = -1.0
        p_work = p
    else:
        sign = 1.0
        p_work = 1.0 - p

    t = math.sqrt(-2.0 * math.log(p_work))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return mu + sign * z * sigma


def _t_cdf(t_val: float, df: int) -> float:
    """Approximate CDF of Student's t-distribution via normal for large df.

    For df >= 30 uses the normal approximation.  For smaller df uses the
    regularised incomplete beta function expansion.
    """
    if df >= 30:
        return _normal_cdf(t_val)
    # Series approximation for small df
    x = df / (df + t_val * t_val)
    return 1.0 - 0.5 * _regularised_beta(df / 2.0, 0.5, x) if t_val >= 0 \
        else 0.5 * _regularised_beta(df / 2.0, 0.5, x)


def _regularised_beta(a: float, b: float, x: float,
                      max_iter: int = 200, eps: float = 1e-12) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction."""
    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    # Use the symmetry relation when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_beta(b, a, 1.0 - x, max_iter, eps)

    log_prefix = (
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )

    # Lentz continued fraction
    f = 1e-30
    c = 1e-30
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    result = d

    for m in range(1, max_iter):
        # even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        result *= d * c

        # odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < eps:
            break

    return math.exp(log_prefix) * result / a


_LANCZOS_COEFFICIENTS: list[float] = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
]


def _log_gamma(x: float) -> float:
    if x <= 0:
        raise ValueError("log_gamma requires x > 0")
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for coeff in _LANCZOS_COEFFICIENTS:
        y += 1.0
        ser += coeff / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval for a parameter estimate.

    Attributes
    ----------
    lower : float
    upper : float
    level : float
        Confidence level (e.g. 0.95).
    method : str
        How the CI was computed (e.g. ``"t"`` or ``"bootstrap"``).
    """

    lower: float
    upper: float
    level: float
    method: str


@dataclass(frozen=True)
class EffectSizeResult:
    """Container for effect-size estimates."""

    cohens_d: float
    cliffs_delta: float
    interpretation: str


@dataclass(frozen=True)
class TestResult:
    """Comprehensive result of a hypothesis test.

    Collects the test statistic, *p*-value, decision, effect size,
    confidence interval, and (where applicable) post-hoc power.
    """

    test_statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    method: str
    effect_size: Optional[EffectSizeResult] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    power: Optional[float] = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Effect Size
# ---------------------------------------------------------------------------


class EffectSize:
    """Compute effect-size metrics for leakage magnitudes."""

    @staticmethod
    def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
        """Cohen's *d* (pooled standard deviation denominator).

        Returns the standardised mean difference.
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        m1 = math.fsum(group1) / n1
        m2 = math.fsum(group2) / n2
        ss1 = math.fsum((x - m1) ** 2 for x in group1)
        ss2 = math.fsum((x - m2) ** 2 for x in group2)
        pooled_var = (ss1 + ss2) / (n1 + n2 - 2)
        if pooled_var <= 0:
            return 0.0
        return (m1 - m2) / math.sqrt(pooled_var)

    @staticmethod
    def cliffs_delta(group1: Sequence[float], group2: Sequence[float]) -> float:
        """Cliff's delta: non-parametric effect size.

        Counts the proportion of all pairwise comparisons where the values
        in *group1* exceed those in *group2* minus the reverse.
        """
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        more = 0
        less = 0
        for a in group1:
            for b in group2:
                if a > b:
                    more += 1
                elif a < b:
                    less += 1
        return (more - less) / (n1 * n2)

    @staticmethod
    def interpret(d: float) -> str:
        """Interpret an absolute effect size |d| using Cohen's conventions."""
        ad = abs(d)
        if ad < 0.2:
            return "negligible"
        if ad < 0.5:
            return "small"
        if ad < 0.8:
            return "medium"
        return "large"

    @classmethod
    def compute(cls, group1: Sequence[float],
                group2: Sequence[float]) -> EffectSizeResult:
        """Compute both Cohen's *d* and Cliff's delta with interpretation."""
        d = cls.cohens_d(group1, group2)
        delta = cls.cliffs_delta(group1, group2)
        return EffectSizeResult(
            cohens_d=d,
            cliffs_delta=delta,
            interpretation=cls.interpret(d),
        )


# ---------------------------------------------------------------------------
# Confidence Interval computation
# ---------------------------------------------------------------------------


class ConfidenceIntervalEstimator:
    """Compute confidence intervals for leakage estimates."""

    @staticmethod
    def t_interval(data: Sequence[float],
                   confidence: float = 0.95) -> ConfidenceInterval:
        """*t*-based confidence interval for the population mean."""
        n = len(data)
        if n < 2:
            m = data[0] if n == 1 else 0.0
            return ConfidenceInterval(m, m, confidence, "t")
        m = math.fsum(data) / n
        se = math.sqrt(math.fsum((x - m) ** 2 for x in data) / (n * (n - 1)))
        # Use normal quantile as approximation for t (good for n >= 5)
        z = _normal_ppf(1.0 - (1.0 - confidence) / 2.0)
        return ConfidenceInterval(m - z * se, m + z * se, confidence, "t")

    @staticmethod
    def bootstrap_interval(data: Sequence[float], confidence: float = 0.95,
                           n_bootstrap: int = 1000,
                           seed: Optional[int] = None) -> ConfidenceInterval:
        """Percentile bootstrap confidence interval for the mean."""
        rng = random.Random(seed)
        n = len(data)
        means: list[float] = []
        for _ in range(n_bootstrap):
            sample = [data[rng.randrange(n)] for _ in range(n)]
            means.append(math.fsum(sample) / n)
        means.sort()
        lo_idx = int(math.floor((1.0 - confidence) / 2.0 * n_bootstrap))
        hi_idx = int(math.ceil((1.0 + confidence) / 2.0 * n_bootstrap)) - 1
        lo_idx = max(lo_idx, 0)
        hi_idx = min(hi_idx, n_bootstrap - 1)
        return ConfidenceInterval(means[lo_idx], means[hi_idx], confidence,
                                  "bootstrap")


# ---------------------------------------------------------------------------
# Power Analysis
# ---------------------------------------------------------------------------


class PowerAnalysis:
    """Statistical power computation for leakage detection.

    Computes the power of a one-sample *z*-test (H0: mu = 0) or determines
    the sample size needed to reach a target power.
    """

    @staticmethod
    def compute_power(effect_size: float, n: int, alpha: float = 0.05,
                      alternative: str = "two-sided") -> float:
        """Power of a one-sample *z*-test.

        Parameters
        ----------
        effect_size : Cohen's *d* (expected mean / std).
        n : sample size.
        alpha : significance level.
        alternative : ``"two-sided"``, ``"greater"``, ``"less"``.
        """
        se = 1.0 / math.sqrt(n) if n > 0 else float("inf")
        if alternative == "two-sided":
            z_crit = _normal_ppf(1.0 - alpha / 2.0)
            power = (
                1.0 - _normal_cdf(z_crit - effect_size / se)
                + _normal_cdf(-z_crit - effect_size / se)
            )
        elif alternative == "greater":
            z_crit = _normal_ppf(1.0 - alpha)
            power = 1.0 - _normal_cdf(z_crit - effect_size / se)
        else:
            z_crit = _normal_ppf(alpha)
            power = _normal_cdf(z_crit - effect_size / se)
        return max(min(power, 1.0), 0.0)

    @staticmethod
    def required_sample_size(effect_size: float, power: float = 0.8,
                             alpha: float = 0.05,
                             alternative: str = "two-sided") -> int:
        """Minimum sample size to achieve *power* for a one-sample z-test.

        Uses a binary search over sample sizes.
        """
        if effect_size == 0:
            return -1  # infinite samples needed
        lo, hi = 2, 10
        while PowerAnalysis.compute_power(effect_size, hi, alpha, alternative) < power:
            hi *= 2
            if hi > 1_000_000:
                return hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if PowerAnalysis.compute_power(effect_size, mid, alpha, alternative) >= power:
                hi = mid
            else:
                lo = mid
        return hi


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class HypothesisTest(ABC):
    """Abstract base class for hypothesis tests."""

    @abstractmethod
    def run(self, *args, **kwargs) -> TestResult:
        """Execute the test and return a :class:`TestResult`."""


# ---------------------------------------------------------------------------
# One-Sample Test (leakage differs from zero)
# ---------------------------------------------------------------------------


class OneSampleTest(HypothesisTest):
    """Test whether the mean of a sample significantly differs from a reference.

    Default reference value is 0, making this a "does leakage exist?" test.
    Uses a *t*-test approximation.
    """

    def __init__(self, alpha: float = 0.05, mu0: float = 0.0,
                 alternative: str = "two-sided") -> None:
        self.alpha = alpha
        self.mu0 = mu0
        self.alternative = alternative

    def run(self, data: Sequence[float]) -> TestResult:
        """Run the one-sample test on *data*."""
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 observations")
        mean = math.fsum(data) / n
        var = math.fsum((x - mean) ** 2 for x in data) / (n - 1)
        se = math.sqrt(var / n)
        if se == 0:
            t_stat = 0.0 if mean == self.mu0 else float("inf")
        else:
            t_stat = (mean - self.mu0) / se
        df = n - 1

        if self.alternative == "two-sided":
            p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
        elif self.alternative == "greater":
            p_value = 1.0 - _t_cdf(t_stat, df)
        else:
            p_value = _t_cdf(t_stat, df)

        p_value = max(min(p_value, 1.0), 0.0)
        ci = ConfidenceIntervalEstimator.t_interval(data, 1.0 - self.alpha)
        es_d = (mean - self.mu0) / math.sqrt(var) if var > 0 else 0.0
        es = EffectSizeResult(cohens_d=es_d, cliffs_delta=0.0,
                              interpretation=EffectSize.interpret(es_d))
        power = PowerAnalysis.compute_power(abs(es_d), n, self.alpha,
                                            self.alternative)

        return TestResult(
            test_statistic=t_stat, p_value=p_value,
            reject_null=p_value < self.alpha, alpha=self.alpha,
            method="OneSampleT", effect_size=es,
            confidence_interval=ci, power=power,
            details={"mean": mean, "se": se, "df": df, "mu0": self.mu0},
        )


# ---------------------------------------------------------------------------
# Two-Sample Test (compare two pipeline variants)
# ---------------------------------------------------------------------------


class TwoSampleHypothesisTest(HypothesisTest):
    """Welch's *t*-test comparing means of two independent groups.

    Typical use: compare leakage scores between two pipeline configurations.
    """

    def __init__(self, alpha: float = 0.05,
                 alternative: str = "two-sided") -> None:
        self.alpha = alpha
        self.alternative = alternative

    def run(self, group1: Sequence[float],
            group2: Sequence[float]) -> TestResult:
        """Run Welch's *t*-test."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            raise ValueError("Each group needs at least 2 observations")
        m1 = math.fsum(group1) / n1
        m2 = math.fsum(group2) / n2
        v1 = math.fsum((x - m1) ** 2 for x in group1) / (n1 - 1)
        v2 = math.fsum((x - m2) ** 2 for x in group2) / (n2 - 1)
        se = math.sqrt(v1 / n1 + v2 / n2)
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (m1 - m2) / se

        # Welch–Satterthwaite degrees of freedom
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = int(num / denom) if denom > 0 else n1 + n2 - 2

        if self.alternative == "two-sided":
            p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
        elif self.alternative == "greater":
            p_value = 1.0 - _t_cdf(t_stat, df)
        else:
            p_value = _t_cdf(t_stat, df)
        p_value = max(min(p_value, 1.0), 0.0)

        es = EffectSize.compute(group1, group2)
        ci = ConfidenceInterval(
            lower=m1 - m2 - _normal_ppf(1.0 - self.alpha / 2.0) * se,
            upper=m1 - m2 + _normal_ppf(1.0 - self.alpha / 2.0) * se,
            level=1.0 - self.alpha, method="welch-t",
        )

        return TestResult(
            test_statistic=t_stat, p_value=p_value,
            reject_null=p_value < self.alpha, alpha=self.alpha,
            method="WelchT", effect_size=es,
            confidence_interval=ci,
            details={"mean1": m1, "mean2": m2, "se": se, "df": df},
        )


# ---------------------------------------------------------------------------
# Paired Test (before / after remediation)
# ---------------------------------------------------------------------------


class PairedTest(HypothesisTest):
    """Paired *t*-test for before/after leakage comparisons.

    Computes the difference of each paired observation and applies a
    one-sample *t*-test on the differences.
    """

    def __init__(self, alpha: float = 0.05,
                 alternative: str = "two-sided") -> None:
        self.alpha = alpha
        self.alternative = alternative

    def run(self, before: Sequence[float],
            after: Sequence[float]) -> TestResult:
        """Run paired test on matched *before* / *after* observations."""
        if len(before) != len(after):
            raise ValueError("before and after must have equal length")
        diffs = [b - a for b, a in zip(before, after)]
        inner = OneSampleTest(alpha=self.alpha, mu0=0.0,
                              alternative=self.alternative)
        result = inner.run(diffs)
        return TestResult(
            test_statistic=result.test_statistic,
            p_value=result.p_value,
            reject_null=result.reject_null,
            alpha=result.alpha,
            method="PairedT",
            effect_size=result.effect_size,
            confidence_interval=result.confidence_interval,
            power=result.power,
            details={**result.details, "n_pairs": len(diffs)},
        )


# ---------------------------------------------------------------------------
# Leakage Significance Test
# ---------------------------------------------------------------------------


class LeakageSignificanceTest(HypothesisTest):
    """Test whether detected leakage is statistically significant.

    Wraps a one-sided one-sample test (H0: leakage ≤ threshold) with a
    practical significance threshold.
    """

    def __init__(self, alpha: float = 0.05,
                 threshold: float = 0.0) -> None:
        self.alpha = alpha
        self.threshold = threshold

    def run(self, leakage_scores: Sequence[float]) -> TestResult:
        """Test whether *leakage_scores* indicate significant leakage."""
        inner = OneSampleTest(alpha=self.alpha, mu0=self.threshold,
                              alternative="greater")
        result = inner.run(leakage_scores)
        mean_leakage = math.fsum(leakage_scores) / len(leakage_scores)
        return TestResult(
            test_statistic=result.test_statistic,
            p_value=result.p_value,
            reject_null=result.reject_null,
            alpha=result.alpha,
            method="LeakageSignificance",
            effect_size=result.effect_size,
            confidence_interval=result.confidence_interval,
            power=result.power,
            details={
                **result.details,
                "mean_leakage": mean_leakage,
                "threshold": self.threshold,
            },
        )


# ---------------------------------------------------------------------------
# Meta-Analysis
# ---------------------------------------------------------------------------


class MetaAnalysis:
    """Combine results from multiple independent tests (Stouffer's method).

    Useful when auditing multiple features or pipeline stages: each
    produces a *p*-value and we want an overall conclusion.
    """

    @staticmethod
    def stouffer(p_values: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> Tuple[float, float]:
        """Stouffer's weighted *Z* method.

        Returns ``(combined_z, combined_p)`` for a two-sided test.
        """
        k = len(p_values)
        if k == 0:
            return 0.0, 1.0
        if weights is None:
            weights = [1.0] * k

        z_scores = [_normal_ppf(1.0 - p) for p in p_values]
        wt = list(weights)
        combined_z = sum(w * z for w, z in zip(wt, z_scores)) / math.sqrt(
            sum(w * w for w in wt)
        )
        combined_p = 1.0 - _normal_cdf(combined_z)
        return combined_z, max(min(combined_p, 1.0), 0.0)

    @staticmethod
    def fisher(p_values: Sequence[float]) -> Tuple[float, float]:
        """Fisher's method for combining *p*-values.

        Test statistic: -2 * sum(log(p_i)), which under H0 follows a
        chi-squared distribution with 2k degrees of freedom.
        """
        k = len(p_values)
        if k == 0:
            return 0.0, 1.0
        stat = -2.0 * sum(math.log(max(p, 1e-300)) for p in p_values)
        # Chi-squared CDF with 2k df via regularised gamma
        from taintflow.stats.independence import _chi2_cdf
        p_combined = 1.0 - _chi2_cdf(stat, 2 * k)
        return stat, max(min(p_combined, 1.0), 0.0)

    @staticmethod
    def combine(p_values: Sequence[float], method: str = "stouffer",
                weights: Optional[Sequence[float]] = None,
                alpha: float = 0.05) -> dict:
        """Combine *p*-values and return a summary dict."""
        if method == "stouffer":
            stat, p = MetaAnalysis.stouffer(p_values, weights)
        elif method == "fisher":
            stat, p = MetaAnalysis.fisher(p_values)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return {
            "method": method,
            "statistic": stat,
            "combined_p_value": p,
            "reject_null": p < alpha,
            "alpha": alpha,
            "n_tests": len(p_values),
        }


# ---------------------------------------------------------------------------
# Sequential Test (early stopping)
# ---------------------------------------------------------------------------


class SequentialTest:
    """Sequential probability ratio test (SPRT) for early stopping.

    At each new observation the test computes the log-likelihood ratio
    between H1 (leakage = *effect_size*) and H0 (leakage = 0) under a
    normal model, and compares it against decision boundaries derived from
    alpha and beta.

    Useful when running expensive leakage simulations and wanting to stop
    as soon as sufficient evidence is gathered.
    """

    def __init__(self, effect_size: float = 0.5, alpha: float = 0.05,
                 beta: float = 0.20, sigma: float = 1.0) -> None:
        if effect_size <= 0:
            raise ValueError("effect_size must be positive")
        self.effect_size = effect_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self._upper = math.log((1.0 - beta) / alpha)
        self._lower = math.log(beta / (1.0 - alpha))
        self._log_lr = 0.0
        self._observations: list[float] = []
        self._decision: Optional[str] = None

    @property
    def upper_boundary(self) -> float:
        """Log-LR boundary for rejecting H0."""
        return self._upper

    @property
    def lower_boundary(self) -> float:
        """Log-LR boundary for accepting H0."""
        return self._lower

    @property
    def decision(self) -> Optional[str]:
        """Current decision: ``'reject_H0'``, ``'accept_H0'``, or ``None``."""
        return self._decision

    @property
    def log_likelihood_ratio(self) -> float:
        return self._log_lr

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def update(self, observation: float) -> Optional[str]:
        """Incorporate a new *observation* and return the decision if reached.

        Returns ``'reject_H0'``, ``'accept_H0'``, or ``None`` (continue).
        """
        if self._decision is not None:
            return self._decision
        self._observations.append(observation)
        mu1 = self.effect_size
        s2 = self.sigma * self.sigma
        # Log-likelihood ratio increment: log f(x|mu1) - log f(x|0)
        increment = (observation * mu1 - 0.5 * mu1 * mu1) / s2
        self._log_lr += increment

        if self._log_lr >= self._upper:
            self._decision = "reject_H0"
        elif self._log_lr <= self._lower:
            self._decision = "accept_H0"
        return self._decision

    def update_batch(self, observations: Sequence[float]) -> Optional[str]:
        """Process multiple observations, stopping early if a decision is reached."""
        for obs in observations:
            result = self.update(obs)
            if result is not None:
                return result
        return None

    def result(self) -> TestResult:
        """Build a :class:`TestResult` summarising the current state."""
        reject = self._decision == "reject_H0"
        # Approximate p-value from the LR
        if self._log_lr >= 0:
            approx_p = math.exp(-self._log_lr)
        else:
            approx_p = 1.0
        approx_p = max(min(approx_p, 1.0), 0.0)

        return TestResult(
            test_statistic=self._log_lr,
            p_value=approx_p,
            reject_null=reject,
            alpha=self.alpha,
            method="SPRT",
            details={
                "decision": self._decision,
                "n_observations": self.n_observations,
                "upper_boundary": self._upper,
                "lower_boundary": self._lower,
                "effect_size": self.effect_size,
            },
        )
