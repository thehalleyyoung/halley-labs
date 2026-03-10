"""
usability_oracle.analysis.statistical — Statistical testing framework.

Provides effect size computation (Cohen's d, Glass's delta, Hedges' g),
power analysis, multiple comparison corrections (Bonferroni, Holm,
Benjamini-Hochberg), non-parametric tests, and bootstrap inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of a statistical test.

    Attributes:
        test_name: Name of the statistical test.
        statistic: Test statistic value.
        p_value: p-value.
        significant: Whether the result is significant at the given alpha.
        effect_size: Standardised effect size.
        confidence_interval: CI for the effect or difference.
        power: Statistical power (if computed).
        alpha: Significance level used.
    """
    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    effect_size: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    power: float = 0.0
    alpha: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4g} ({sig})\n"
            f"  Effect size: {self.effect_size:.4f}\n"
            f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
            f"  Power: {self.power:.4f}"
        )


@dataclass
class MultipleComparisonResult:
    """Result of multiple comparison correction."""
    original_p_values: list[float] = field(default_factory=list)
    adjusted_p_values: list[float] = field(default_factory=list)
    rejected: list[bool] = field(default_factory=list)
    method: str = ""
    alpha: float = 0.05
    n_rejected: int = 0


@dataclass
class PowerAnalysisResult:
    """Result of power analysis."""
    required_n: int = 0
    achieved_power: float = 0.0
    effect_size: float = 0.0
    alpha: float = 0.05
    power_curve: list[tuple[int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d: standardised mean difference using pooled SD.

    d = (M1 - M2) / S_pooled
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean_diff = float(np.mean(g1) - np.mean(g2))
    var1 = float(np.var(g1, ddof=1))
    var2 = float(np.var(g2, ddof=1))
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(max(pooled_var, 1e-15))
    return mean_diff / pooled_sd


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Hedges' g: bias-corrected Cohen's d for small samples.

    g = d * (1 - 3 / (4*(n1+n2) - 9))
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    if df < 4:
        return d
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * correction


def glass_delta(group1: np.ndarray, control: np.ndarray) -> float:
    """Glass's Δ: standardised by the control group's SD.

    Δ = (M1 - M_control) / S_control
    """
    g1 = np.asarray(group1, dtype=float)
    ctrl = np.asarray(control, dtype=float)
    if len(ctrl) < 2:
        return 0.0
    sd_ctrl = float(np.std(ctrl, ddof=1))
    if sd_ctrl < 1e-15:
        return 0.0
    return float(np.mean(g1) - np.mean(ctrl)) / sd_ctrl


def cliff_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cliff's delta: non-parametric effect size.

    δ = (# concordant - # discordant) / (n1 * n2)
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0

    concordant = 0
    discordant = 0
    for x in g1:
        for y in g2:
            if x > y:
                concordant += 1
            elif x < y:
                discordant += 1
    return (concordant - discordant) / (n1 * n2)


def rank_biserial(group1: np.ndarray, group2: np.ndarray) -> float:
    """Rank-biserial correlation from Mann-Whitney U.

    r = 1 - (2U) / (n1 * n2)
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0

    U = 0.0
    for x in g1:
        for y in g2:
            if x > y:
                U += 1.0
            elif x == y:
                U += 0.5
    return 1.0 - (2.0 * U) / (n1 * n2)


# ---------------------------------------------------------------------------
# Non-parametric tests
# ---------------------------------------------------------------------------

def _mann_whitney_u(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided).

    Returns (U_statistic, p_value).
    Uses normal approximation for large samples.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Compute U
    U1 = 0.0
    for x in g1:
        for y in g2:
            if x > y:
                U1 += 1.0
            elif x == y:
                U1 += 0.5

    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Normal approximation
    mu_U = n1 * n2 / 2.0
    sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_U < 1e-15:
        return float(U), 1.0

    z = (U - mu_U) / sigma_U
    p = 2.0 * _normal_cdf(-abs(z))
    return float(U), p


def _wilcoxon_signed_rank(d: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired differences.

    Returns (W_statistic, p_value).
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return 0.0, 1.0

    abs_d = np.abs(d)
    ranks = _rank_array(abs_d)
    W_plus = float(np.sum(ranks[d > 0]))
    W_minus = float(np.sum(ranks[d < 0]))
    W = min(W_plus, W_minus)

    # Normal approximation
    mu_W = n * (n + 1) / 4.0
    sigma_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma_W < 1e-15:
        return W, 1.0

    z = (W - mu_W) / sigma_W
    p = 2.0 * _normal_cdf(-abs(z))
    return W, p


def _kruskal_wallis(*groups: np.ndarray) -> tuple[float, float]:
    """Kruskal-Wallis H test for k independent groups.

    Returns (H_statistic, p_value).
    """
    k = len(groups)
    if k < 2:
        return 0.0, 1.0

    all_data = np.concatenate(groups)
    N = len(all_data)
    if N < 3:
        return 0.0, 1.0

    ranks = _rank_array(all_data)

    # Split ranks back into groups
    idx = 0
    group_ranks = []
    for g in groups:
        n_g = len(g)
        group_ranks.append(ranks[idx:idx + n_g])
        idx += n_g

    # H statistic
    H = 0.0
    for gr in group_ranks:
        n_i = len(gr)
        if n_i > 0:
            mean_rank = np.mean(gr)
            H += n_i * (mean_rank - (N + 1) / 2.0) ** 2

    H = (12.0 / (N * (N + 1))) * H

    # p-value from chi-squared approximation with k-1 df
    df = k - 1
    p = _chi2_sf(H, df)
    return float(H), p


# ---------------------------------------------------------------------------
# Multiple comparison corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Bonferroni correction: multiply each p-value by the number of tests."""
    m = len(p_values)
    adjusted = [min(p * m, 1.0) for p in p_values]
    rejected = [p < alpha for p in adjusted]
    return MultipleComparisonResult(
        original_p_values=list(p_values),
        adjusted_p_values=adjusted,
        rejected=rejected,
        method="bonferroni",
        alpha=alpha,
        n_rejected=sum(rejected),
    )


def holm_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Holm-Bonferroni step-down correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    rejected = [False] * m

    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (m - rank), 1.0)
        adj_p = max(adj_p, prev_adj)  # enforce monotonicity
        adjusted[orig_idx] = adj_p
        rejected[orig_idx] = adj_p < alpha
        prev_adj = adj_p

    return MultipleComparisonResult(
        original_p_values=list(p_values),
        adjusted_p_values=adjusted,
        rejected=rejected,
        method="holm",
        alpha=alpha,
        n_rejected=sum(rejected),
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    rejected = [False] * m

    prev_adj = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank]
        adj_p = min(p * m / (rank + 1), 1.0)
        adj_p = min(adj_p, prev_adj)  # enforce monotonicity
        adjusted[orig_idx] = adj_p
        rejected[orig_idx] = adj_p < alpha
        prev_adj = adj_p

    return MultipleComparisonResult(
        original_p_values=list(p_values),
        adjusted_p_values=adjusted,
        rejected=rejected,
        method="benjamini_hochberg",
        alpha=alpha,
        n_rejected=sum(rejected),
    )


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def _power_t_test(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Compute power of a two-sample t-test.

    Uses the non-central t-distribution approximation.
    Power = P(|T| > t_crit | delta) ≈ Phi(|delta|*sqrt(n/2) - z_{alpha/2})
    """
    if n < 2 or effect_size == 0:
        return alpha

    ncp = abs(effect_size) * math.sqrt(n / 2.0)
    z_crit = _z_score(1.0 - alpha / 2.0) if two_sided else _z_score(1.0 - alpha)
    power = 1.0 - _normal_cdf(z_crit - ncp)
    if two_sided:
        power += _normal_cdf(-z_crit - ncp)
    return min(max(power, 0.0), 1.0)


def compute_required_n(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> PowerAnalysisResult:
    """Compute required sample size per group for a two-sample t-test.

    Uses binary search over sample sizes.
    """
    if abs(effect_size) < 1e-10:
        return PowerAnalysisResult(required_n=999999, achieved_power=alpha, effect_size=effect_size)

    # Binary search
    lo, hi = 2, 1000000
    while lo < hi:
        mid = (lo + hi) // 2
        p = _power_t_test(effect_size, mid, alpha)
        if p >= power:
            hi = mid
        else:
            lo = mid + 1

    achieved = _power_t_test(effect_size, lo, alpha)

    # Power curve
    curve = []
    sizes = [max(2, lo // 10), lo // 5, lo // 2, lo, lo * 2, lo * 5]
    for sz in sorted(set(sizes)):
        curve.append((sz, _power_t_test(effect_size, sz, alpha)))

    return PowerAnalysisResult(
        required_n=lo,
        achieved_power=achieved,
        effect_size=effect_size,
        alpha=alpha,
        power_curve=curve,
    )


# ---------------------------------------------------------------------------
# StatisticalTester
# ---------------------------------------------------------------------------

class StatisticalTester:
    """Comprehensive statistical testing for oracle evaluation.

    Parameters:
        alpha: Default significance level.
        correction: Default multiple comparison correction method.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction: str = "holm",
    ) -> None:
        self._alpha = alpha
        self._correction = correction

    # ------------------------------------------------------------------
    # Two-sample tests
    # ------------------------------------------------------------------

    def two_sample_test(
        self,
        group1: Sequence[float],
        group2: Sequence[float],
        paired: bool = False,
        parametric: bool = True,
    ) -> TestResult:
        """Run a two-sample comparison test.

        Parameters:
            group1, group2: Sample data.
            paired: If True, use paired test.
            parametric: If True, use t-test; otherwise Mann-Whitney/Wilcoxon.
        """
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)

        if paired:
            if len(g1) != len(g2):
                raise ValueError("Paired test requires equal-length groups")
            d = g1 - g2
            if parametric:
                return self._paired_t_test(d)
            else:
                return self._wilcoxon_test(d)
        else:
            if parametric:
                return self._independent_t_test(g1, g2)
            else:
                return self._mann_whitney_test(g1, g2)

    def _independent_t_test(self, g1: np.ndarray, g2: np.ndarray) -> TestResult:
        """Welch's t-test for independent samples."""
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return TestResult(test_name="t-test", alpha=self._alpha)

        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
        se = math.sqrt(v1 / n1 + v2 / n2)
        if se < 1e-15:
            return TestResult(test_name="t-test", alpha=self._alpha)

        t_stat = (m1 - m2) / se

        # Welch-Satterthwaite degrees of freedom
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 1e-15 else n1 + n2 - 2

        p = 2.0 * _t_sf(abs(t_stat), df)
        d = cohens_d(g1, g2)
        ci = (m1 - m2 - _z_score(1.0 - self._alpha / 2) * se,
              m1 - m2 + _z_score(1.0 - self._alpha / 2) * se)
        power = _power_t_test(d, min(n1, n2), self._alpha)

        return TestResult(
            test_name="Welch's t-test",
            statistic=t_stat,
            p_value=p,
            significant=p < self._alpha,
            effect_size=d,
            confidence_interval=ci,
            power=power,
            alpha=self._alpha,
            metadata={"df": df, "n1": n1, "n2": n2},
        )

    def _paired_t_test(self, d: np.ndarray) -> TestResult:
        """Paired t-test on differences."""
        n = len(d)
        if n < 2:
            return TestResult(test_name="paired t-test", alpha=self._alpha)

        mean_d = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / math.sqrt(n)
        if se < 1e-15:
            return TestResult(test_name="paired t-test", alpha=self._alpha)

        t_stat = mean_d / se
        df = n - 1
        p = 2.0 * _t_sf(abs(t_stat), df)

        sd = float(np.std(d, ddof=1))
        d_z = mean_d / sd if sd > 1e-15 else 0.0

        ci = (mean_d - _z_score(1.0 - self._alpha / 2) * se,
              mean_d + _z_score(1.0 - self._alpha / 2) * se)

        return TestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p,
            significant=p < self._alpha,
            effect_size=d_z,
            confidence_interval=ci,
            alpha=self._alpha,
            metadata={"df": df, "n": n},
        )

    def _wilcoxon_test(self, d: np.ndarray) -> TestResult:
        """Wilcoxon signed-rank test."""
        W, p = _wilcoxon_signed_rank(d)
        r = cliff_delta(d[d > 0], -d[d < 0]) if np.any(d > 0) and np.any(d < 0) else 0.0

        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=W,
            p_value=p,
            significant=p < self._alpha,
            effect_size=r,
            alpha=self._alpha,
        )

    def _mann_whitney_test(self, g1: np.ndarray, g2: np.ndarray) -> TestResult:
        """Mann-Whitney U test."""
        U, p = _mann_whitney_u(g1, g2)
        r = rank_biserial(g1, g2)

        return TestResult(
            test_name="Mann-Whitney U",
            statistic=U,
            p_value=p,
            significant=p < self._alpha,
            effect_size=r,
            alpha=self._alpha,
        )

    # ------------------------------------------------------------------
    # Multiple group tests
    # ------------------------------------------------------------------

    def kruskal_wallis(self, *groups: Sequence[float]) -> TestResult:
        """Kruskal-Wallis H test for k independent groups."""
        arrays = [np.asarray(g, dtype=float) for g in groups]
        H, p = _kruskal_wallis(*arrays)
        # Epsilon-squared effect size
        N = sum(len(g) for g in arrays)
        k = len(arrays)
        eps_sq = (H - k + 1) / (N - k) if N > k else 0.0

        return TestResult(
            test_name="Kruskal-Wallis",
            statistic=H,
            p_value=p,
            significant=p < self._alpha,
            effect_size=eps_sq,
            alpha=self._alpha,
            metadata={"k": k, "N": N},
        )

    # ------------------------------------------------------------------
    # Bootstrap test
    # ------------------------------------------------------------------

    def bootstrap_test(
        self,
        group1: Sequence[float],
        group2: Sequence[float],
        statistic_fn: Optional[Any] = None,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> TestResult:
        """Permutation bootstrap test for arbitrary statistics.

        Parameters:
            statistic_fn: Function(g1, g2) -> float.  Defaults to mean difference.
            n_bootstrap: Number of permutations.
        """
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)
        rng = np.random.RandomState(seed)

        if statistic_fn is None:
            def statistic_fn(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.mean(a) - np.mean(b))

        observed = statistic_fn(g1, g2)
        combined = np.concatenate([g1, g2])
        n1 = len(g1)

        count_extreme = 0
        boot_stats = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(combined)
            perm_g1 = perm[:n1]
            perm_g2 = perm[n1:]
            stat = statistic_fn(perm_g1, perm_g2)
            boot_stats.append(stat)
            if abs(stat) >= abs(observed):
                count_extreme += 1

        p = (count_extreme + 1) / (n_bootstrap + 1)
        boot_arr = np.array(boot_stats)
        ci = (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)))

        return TestResult(
            test_name="Bootstrap permutation",
            statistic=observed,
            p_value=p,
            significant=p < self._alpha,
            effect_size=cohens_d(g1, g2),
            confidence_interval=ci,
            alpha=self._alpha,
            metadata={"n_bootstrap": n_bootstrap},
        )

    # ------------------------------------------------------------------
    # Multiple comparison correction
    # ------------------------------------------------------------------

    def correct_multiple(
        self,
        p_values: Sequence[float],
        method: Optional[str] = None,
    ) -> MultipleComparisonResult:
        """Apply multiple comparison correction."""
        m = method or self._correction
        if m == "bonferroni":
            return bonferroni_correction(p_values, self._alpha)
        elif m == "holm":
            return holm_correction(p_values, self._alpha)
        elif m == "benjamini_hochberg" or m == "bh":
            return benjamini_hochberg(p_values, self._alpha)
        else:
            raise ValueError(f"Unknown correction method: {m}")

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def power_analysis(
        self,
        effect_size: float,
        target_power: float = 0.8,
    ) -> PowerAnalysisResult:
        """Compute required sample size for given effect size and power."""
        return compute_required_n(effect_size, target_power, self._alpha)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Rank values (1-based, average tie-breaking)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _z_score(confidence: float) -> float:
    """Approximate z-score for given confidence level."""
    p = (1.0 + confidence) / 2.0 if confidence < 1.0 else confidence
    if p >= 1.0:
        return 5.0
    if p <= 0.0:
        return -5.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)


def _t_sf(t: float, df: float) -> float:
    """Survival function of t-distribution (approximate)."""
    # Use normal approximation for large df
    if df > 30:
        return 1.0 - _normal_cdf(t)
    # Otherwise use regularised incomplete beta function approximation
    x = df / (df + t * t)
    return 0.5 * _regularized_incomplete_beta(x, df / 2.0, 0.5)


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-squared distribution (approximate)."""
    if k <= 0 or x < 0:
        return 1.0
    # Wilson-Hilferty normal approximation
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 1.0 - _normal_cdf(z)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Approximate regularised incomplete beta function I_x(a, b).

    Uses continued fraction expansion (Lentz's method).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction for I_x(a,b)
    # First compute the prefactor
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    prefix = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a

    # Lentz's continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return prefix * f
