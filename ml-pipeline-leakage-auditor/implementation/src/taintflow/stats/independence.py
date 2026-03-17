"""Independence tests for ML pipeline leakage auditing.

Provides non-parametric independence tests implemented entirely with the
Python standard library.  These are used by TaintFlow to determine whether
a feature's train-set values are statistically independent of test-set
information, which is the core null hypothesis when auditing for leakage.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Mathematical helper functions
# ---------------------------------------------------------------------------

_LANCZOS_COEFFICIENTS: list[float] = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
]


def _log_gamma(x: float) -> float:
    """Lanczos approximation of ln(Gamma(x)) for x > 0."""
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


def _gamma(x: float) -> float:
    """Gamma function via Lanczos approximation."""
    return math.exp(_log_gamma(x))


def _lower_incomplete_gamma(a: float, x: float, *, max_iter: int = 200,
                            eps: float = 1e-12) -> float:
    """Lower incomplete gamma function via series expansion.

    Returns gamma(a, x) = integral from 0 to x of t^(a-1) * e^(-t) dt.
    Uses the series  gamma(a,x) = e^{-x} x^a sum_{n=0}^{inf} x^n / (a*(a+1)*...*(a+n)).
    """
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 0.0
    log_prefix = a * math.log(x) - x - _log_gamma(a)
    term = 1.0 / a
    total = term
    for n in range(1, max_iter):
        term *= x / (a + n)
        total += term
        if abs(term) < eps * abs(total):
            break
    return math.exp(log_prefix) * total * _gamma(a)


def _regularized_lower_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) = gamma(a,x) / Gamma(a)."""
    if x < 0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _lower_inc_gamma_series(a, x)
    return 1.0 - _upper_inc_gamma_cf(a, x)


def _lower_inc_gamma_series(a: float, x: float, max_iter: int = 200,
                            eps: float = 1e-12) -> float:
    """P(a,x) via series expansion."""
    log_prefix = a * math.log(x) - x - _log_gamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(log_prefix)


def _upper_inc_gamma_cf(a: float, x: float, max_iter: int = 200,
                        eps: float = 1e-12) -> float:
    """Q(a,x) = 1 - P(a,x) via continued fraction (Lentz's method)."""
    log_prefix = a * math.log(x) - x - _log_gamma(a)
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b if b != 0 else 1e30
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(log_prefix)


def _chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-squared distribution with *k* degrees of freedom."""
    if x <= 0:
        return 0.0
    return _regularized_lower_gamma(k / 2.0, x / 2.0)


def _erf(x: float) -> float:
    """Approximation of the error function using Abramowitz & Stegun 7.1.26."""
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
    """CDF of the normal distribution N(mu, sigma^2)."""
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * math.sqrt(2.0))))


def _normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse CDF (percent-point function) of the normal distribution.

    Uses a rational approximation (Abramowitz & Stegun 26.2.23).
    """
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


# ---------------------------------------------------------------------------
# Result data-class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndependenceTestResult:
    """Result of an independence test.

    Attributes
    ----------
    test_statistic : float
        The value of the test statistic.
    p_value : float
        Estimated *p*-value under H0 (independence).
    reject_null : bool
        ``True`` when we reject H0 at level ``alpha``.
    alpha : float
        Significance level used for the decision.
    method : str
        Human-readable name of the test that produced this result.
    details : dict
        Optional additional diagnostics.
    """

    test_statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    method: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chi-Squared Test
# ---------------------------------------------------------------------------


class ChiSquaredTest:
    """Pearson's chi-squared test of independence for two discrete variables.

    Constructs a contingency table and computes the chi-squared statistic.
    The *p*-value is obtained from the chi-squared CDF approximation.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(self, x: Sequence, y: Sequence) -> IndependenceTestResult:
        """Run the chi-squared independence test on paired samples *x*, *y*."""
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        n = len(x)
        if n == 0:
            raise ValueError("Samples must be non-empty")

        joint: Counter = Counter(zip(x, y))
        x_vals = sorted(set(x))
        y_vals = sorted(set(y))
        r = len(x_vals)
        c = len(y_vals)
        if r < 2 or c < 2:
            return IndependenceTestResult(
                test_statistic=0.0, p_value=1.0, reject_null=False,
                alpha=self.alpha, method="ChiSquared",
                details={"note": "fewer than 2 categories in one variable"},
            )

        x_idx = {v: i for i, v in enumerate(x_vals)}
        y_idx = {v: i for i, v in enumerate(y_vals)}

        observed = [[0] * c for _ in range(r)]
        for (xi, yi), cnt in joint.items():
            observed[x_idx[xi]][y_idx[yi]] = cnt

        row_sums = [sum(row) for row in observed]
        col_sums = [sum(observed[i][j] for i in range(r)) for j in range(c)]

        chi2 = 0.0
        for i in range(r):
            for j in range(c):
                expected = row_sums[i] * col_sums[j] / n
                if expected > 0:
                    chi2 += (observed[i][j] - expected) ** 2 / expected

        dof = (r - 1) * (c - 1)
        p_value = 1.0 - _chi2_cdf(chi2, dof)

        return IndependenceTestResult(
            test_statistic=chi2,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            method="ChiSquared",
            details={"dof": dof, "n": n, "rows": r, "cols": c},
        )


# ---------------------------------------------------------------------------
# Mutual Information Test
# ---------------------------------------------------------------------------


class MutualInformationTest:
    """Test whether the mutual information between X and Y is significantly > 0.

    Uses a permutation test to build the null distribution of MI.
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 500,
                 seed: Optional[int] = None) -> None:
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed

    @staticmethod
    def _mi(x: Sequence, y: Sequence) -> float:
        """Plugin estimator of mutual information (nats)."""
        n = len(x)
        if n == 0:
            return 0.0
        joint: Counter = Counter(zip(x, y))
        x_counts: Counter = Counter(x)
        y_counts: Counter = Counter(y)
        mi = 0.0
        for (xi, yi), nij in joint.items():
            pxy = nij / n
            px = x_counts[xi] / n
            py = y_counts[yi] / n
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py))
        return mi

    def test(self, x: Sequence, y: Sequence) -> IndependenceTestResult:
        """Run permutation-based MI significance test."""
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        n = len(x)
        if n == 0:
            raise ValueError("Samples must be non-empty")

        observed_mi = self._mi(x, y)

        rng = random.Random(self.seed)
        y_list = list(y)
        count_ge = 0
        for _ in range(self.n_permutations):
            rng.shuffle(y_list)
            perm_mi = self._mi(x, y_list)
            if perm_mi >= observed_mi:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)

        return IndependenceTestResult(
            test_statistic=observed_mi,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            method="MutualInformation",
            details={"n_permutations": self.n_permutations, "n": n},
        )


# ---------------------------------------------------------------------------
# Distance Correlation Test
# ---------------------------------------------------------------------------


class DistanceCorrelationTest:
    """Distance correlation test for non-linear dependence.

    Distance correlation equals zero if and only if the variables are
    independent (for finite second moments).  We use a permutation test
    to assess significance.
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 500,
                 seed: Optional[int] = None) -> None:
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed

    @staticmethod
    def _pairwise_distances(vals: Sequence[float]) -> list[list[float]]:
        n = len(vals)
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(vals[i] - vals[j])
                d[i][j] = dist
                d[j][i] = dist
        return d

    @staticmethod
    def _doubly_centred(d: list[list[float]]) -> list[list[float]]:
        n = len(d)
        row_means = [sum(d[i]) / n for i in range(n)]
        col_means = [sum(d[i][j] for i in range(n)) / n for j in range(n)]
        grand = sum(row_means) / n
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a[i][j] = d[i][j] - row_means[i] - col_means[j] + grand
        return a

    @staticmethod
    def _dcov_sq(a: list[list[float]], b: list[list[float]]) -> float:
        n = len(a)
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += a[i][j] * b[i][j]
        return total / (n * n)

    def _dcor(self, x: Sequence[float], y: Sequence[float]) -> float:
        dx = self._pairwise_distances(x)
        dy = self._pairwise_distances(y)
        a = self._doubly_centred(dx)
        b = self._doubly_centred(dy)
        dcov2 = self._dcov_sq(a, b)
        dvar_x = self._dcov_sq(a, a)
        dvar_y = self._dcov_sq(b, b)
        if dvar_x <= 0 or dvar_y <= 0:
            return 0.0
        return math.sqrt(max(dcov2, 0.0)) / math.sqrt(math.sqrt(dvar_x) * math.sqrt(dvar_y))

    def test(self, x: Sequence[float], y: Sequence[float]) -> IndependenceTestResult:
        """Run distance-correlation permutation test."""
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        n = len(x)
        if n < 3:
            raise ValueError("Need at least 3 observations")

        observed = self._dcor(x, y)

        rng = random.Random(self.seed)
        y_list = list(y)
        count_ge = 0
        for _ in range(self.n_permutations):
            rng.shuffle(y_list)
            if self._dcor(x, y_list) >= observed:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)

        return IndependenceTestResult(
            test_statistic=observed,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            method="DistanceCorrelation",
            details={"n_permutations": self.n_permutations, "n": n},
        )


# ---------------------------------------------------------------------------
# HSIC (Hilbert-Schmidt Independence Criterion)
# ---------------------------------------------------------------------------


class HilbertSchmidtIndependenceTest:
    """Kernel-based independence test using the HSIC statistic.

    Uses a Gaussian (RBF) kernel with the median heuristic for bandwidth
    selection and a permutation test for *p*-value estimation.
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 500,
                 seed: Optional[int] = None) -> None:
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed

    @staticmethod
    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _rbf_kernel_matrix(self, vals: Sequence[float]) -> list[list[float]]:
        n = len(vals)
        dists: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(abs(vals[i] - vals[j]))
        sigma = self._median(dists) if dists else 1.0
        if sigma == 0:
            sigma = 1.0
        gamma = 1.0 / (2.0 * sigma * sigma)
        k = [[0.0] * n for _ in range(n)]
        for i in range(n):
            k[i][i] = 1.0
            for j in range(i + 1, n):
                v = math.exp(-gamma * (vals[i] - vals[j]) ** 2)
                k[i][j] = v
                k[j][i] = v
        return k

    @staticmethod
    def _centre_kernel(k: list[list[float]]) -> list[list[float]]:
        n = len(k)
        row_means = [sum(k[i]) / n for i in range(n)]
        grand = sum(row_means) / n
        col_means = [sum(k[i][j] for i in range(n)) / n for j in range(n)]
        h = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                h[i][j] = k[i][j] - row_means[i] - col_means[j] + grand
        return h

    @staticmethod
    def _hsic_statistic(hk: list[list[float]], hl: list[list[float]]) -> float:
        n = len(hk)
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += hk[i][j] * hl[i][j]
        return total / (n * n)

    def test(self, x: Sequence[float], y: Sequence[float]) -> IndependenceTestResult:
        """Run the HSIC permutation test."""
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        n = len(x)
        if n < 3:
            raise ValueError("Need at least 3 observations")

        kx = self._rbf_kernel_matrix(x)
        hk = self._centre_kernel(kx)

        ky = self._rbf_kernel_matrix(y)
        hl = self._centre_kernel(ky)

        observed = self._hsic_statistic(hk, hl)

        rng = random.Random(self.seed)
        indices = list(range(n))
        count_ge = 0
        for _ in range(self.n_permutations):
            perm = indices[:]
            rng.shuffle(perm)
            hl_perm = [[hl[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
            if self._hsic_statistic(hk, hl_perm) >= observed:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)

        return IndependenceTestResult(
            test_statistic=observed,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            method="HSIC",
            details={"n_permutations": self.n_permutations, "n": n},
        )


# ---------------------------------------------------------------------------
# Conditional Independence Test   X _||_ Y | Z
# ---------------------------------------------------------------------------


class ConditionalIndependenceTest:
    """Test X ⊥ Y | Z by stratifying on Z and combining per-stratum tests.

    For discrete Z the test stratifies exactly.  For continuous Z the values
    are discretised into *n_bins* equal-frequency bins before stratification.
    Per-stratum chi-squared statistics are summed (Cochran–Mantel–Haenszel
    style) and the combined statistic is compared against a chi-squared
    distribution whose degrees of freedom equal the sum of per-stratum dof.
    """

    def __init__(self, alpha: float = 0.05, n_bins: int = 5) -> None:
        self.alpha = alpha
        self.n_bins = n_bins

    def _discretise(self, z: Sequence) -> list:
        """Bin continuous values into *n_bins* equal-frequency groups."""
        indexed = sorted(enumerate(z), key=lambda t: t[1])
        n = len(indexed)
        bin_size = max(n // self.n_bins, 1)
        labels: list = [0] * n
        for rank, (orig_idx, _) in enumerate(indexed):
            labels[orig_idx] = min(rank // bin_size, self.n_bins - 1)
        return labels

    def test(self, x: Sequence, y: Sequence, z: Sequence) -> IndependenceTestResult:
        """Test conditional independence X ⊥ Y | Z."""
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z must have equal length")
        n = len(x)
        if n == 0:
            raise ValueError("Samples must be non-empty")

        z_discrete: Sequence
        if all(isinstance(zi, (int, str, bool)) for zi in z):
            z_discrete = z
        else:
            z_discrete = self._discretise(z)

        strata: dict = {}
        for i in range(n):
            strata.setdefault(z_discrete[i], ([], []))
            strata[z_discrete[i]][0].append(x[i])
            strata[z_discrete[i]][1].append(y[i])

        chi2_test = ChiSquaredTest(alpha=self.alpha)
        total_stat = 0.0
        total_dof = 0
        stratum_results: list[dict] = []
        for z_val, (xs, ys) in sorted(strata.items(), key=lambda t: str(t[0])):
            if len(xs) < 4:
                continue
            result = chi2_test.test(xs, ys)
            dof = result.details.get("dof", 0)
            if dof > 0:
                total_stat += result.test_statistic
                total_dof += dof
                stratum_results.append({
                    "z_value": z_val,
                    "chi2": result.test_statistic,
                    "dof": dof,
                    "n": len(xs),
                })

        if total_dof == 0:
            return IndependenceTestResult(
                test_statistic=0.0, p_value=1.0, reject_null=False,
                alpha=self.alpha, method="ConditionalIndependence",
                details={"note": "insufficient variation in strata"},
            )

        p_value = 1.0 - _chi2_cdf(total_stat, total_dof)

        return IndependenceTestResult(
            test_statistic=total_stat,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            method="ConditionalIndependence",
            details={"total_dof": total_dof, "strata": stratum_results},
        )


# ---------------------------------------------------------------------------
# Multiple Testing Correction
# ---------------------------------------------------------------------------


class MultipleTestingCorrection:
    """Adjust *p*-values for multiple comparisons.

    Supports Bonferroni, Holm step-down, and Benjamini–Hochberg (BH) FDR
    correction methods.
    """

    @staticmethod
    def bonferroni(p_values: Sequence[float]) -> list[float]:
        """Bonferroni correction: multiply each p-value by the number of tests."""
        m = len(p_values)
        return [min(p * m, 1.0) for p in p_values]

    @staticmethod
    def holm(p_values: Sequence[float]) -> list[float]:
        """Holm step-down correction."""
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda t: t[1])
        adjusted = [0.0] * m
        cummax = 0.0
        for rank, (orig_idx, p) in enumerate(indexed):
            adj = p * (m - rank)
            adj = min(adj, 1.0)
            cummax = max(cummax, adj)
            adjusted[orig_idx] = cummax
        return adjusted

    @staticmethod
    def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
        """Benjamini–Hochberg FDR correction."""
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda t: t[1], reverse=True)
        adjusted = [0.0] * m
        cummin = 1.0
        for rank_rev, (orig_idx, p) in enumerate(indexed):
            rank = m - rank_rev  # 1-based rank in ascending order
            adj = p * m / rank
            adj = min(adj, 1.0)
            cummin = min(cummin, adj)
            adjusted[orig_idx] = cummin
        return adjusted

    @classmethod
    def correct(cls, p_values: Sequence[float],
                method: str = "bonferroni") -> list[float]:
        """Correct *p*-values using the specified *method*.

        Parameters
        ----------
        p_values : sequence of float
        method : ``"bonferroni"``, ``"holm"``, or ``"bh"``
        """
        dispatch = {
            "bonferroni": cls.bonferroni,
            "holm": cls.holm,
            "bh": cls.benjamini_hochberg,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"Unknown correction method: {method!r}.  "
                             f"Choose from {list(dispatch)}")
        return fn(p_values)


# ---------------------------------------------------------------------------
# Independence Test Suite
# ---------------------------------------------------------------------------


class IndependenceTestSuite:
    """Run multiple independence tests and aggregate results.

    Designed for the common TaintFlow workflow: given a pair of variables
    (e.g. a feature vector and a leakage indicator), run several tests with
    different statistical assumptions, apply multiple-testing correction, and
    produce an overall verdict.
    """

    def __init__(
        self,
        tests: Optional[List] = None,
        correction: str = "holm",
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.correction = correction
        if tests is not None:
            self.tests = tests
        else:
            self.tests = [
                ChiSquaredTest(alpha=alpha),
                MutualInformationTest(alpha=alpha, seed=seed),
            ]

    def run(self, x: Sequence, y: Sequence) -> Tuple[
        List[IndependenceTestResult], List[float], bool
    ]:
        """Execute all tests and return (results, adjusted_p, overall_reject).

        Returns
        -------
        results : list[IndependenceTestResult]
            Individual test results (unadjusted).
        adjusted_p_values : list[float]
            *p*-values after multiple-testing correction.
        overall_reject : bool
            ``True`` if *any* adjusted *p*-value < ``alpha``.
        """
        results: list[IndependenceTestResult] = []
        for t in self.tests:
            results.append(t.test(x, y))

        raw_p = [r.p_value for r in results]
        adj_p = MultipleTestingCorrection.correct(raw_p, self.correction)

        overall_reject = any(p < self.alpha for p in adj_p)
        return results, adj_p, overall_reject
