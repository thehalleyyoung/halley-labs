"""Distribution utilities for ML pipeline leakage auditing.

Provides empirical distribution estimation, entropy and divergence
calculations, histogram and quantile estimators — all implemented with
the Python standard library only.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Moment Estimator
# ---------------------------------------------------------------------------


class MomentEstimator:
    """Compute sample moments (mean, variance, skewness, kurtosis).

    All statistics are computed in a single pass using Welford's online
    algorithm for numerical stability.
    """

    def __init__(self, data: Sequence[float]) -> None:
        if len(data) == 0:
            raise ValueError("data must be non-empty")
        self._n = len(data)
        self._data = list(data)
        self._mean: Optional[float] = None
        self._var: Optional[float] = None
        self._skew: Optional[float] = None
        self._kurt: Optional[float] = None

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        """Sample mean."""
        if self._mean is None:
            self._compute()
        assert self._mean is not None
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected)."""
        if self._var is None:
            self._compute()
        assert self._var is not None
        return self._var

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """Sample skewness (Fisher definition, bias-corrected)."""
        if self._skew is None:
            self._compute()
        assert self._skew is not None
        return self._skew

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis (Fisher definition, bias-corrected)."""
        if self._kurt is None:
            self._compute()
        assert self._kurt is not None
        return self._kurt

    def _compute(self) -> None:
        n = self._n
        total = math.fsum(self._data)
        mean = total / n
        self._mean = mean

        if n < 2:
            self._var = 0.0
            self._skew = 0.0
            self._kurt = 0.0
            return

        m2 = math.fsum((x - mean) ** 2 for x in self._data)
        self._var = m2 / (n - 1)

        sd = math.sqrt(m2 / n)
        if sd == 0:
            self._skew = 0.0
            self._kurt = 0.0
            return

        m3 = math.fsum((x - mean) ** 3 for x in self._data)
        m4 = math.fsum((x - mean) ** 4 for x in self._data)

        # Bias-corrected skewness
        g1 = (m3 / n) / (sd ** 3)
        if n >= 3:
            self._skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
        else:
            self._skew = g1

        # Bias-corrected excess kurtosis
        g2 = (m4 / n) / (sd ** 4) - 3.0
        if n >= 4:
            self._kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
        else:
            self._kurt = g2


# ---------------------------------------------------------------------------
# Quantile Estimator
# ---------------------------------------------------------------------------


class QuantileEstimator:
    """Quantile estimation with linear interpolation.

    Supports the standard linear interpolation method (equivalent to
    ``numpy.percentile`` with ``interpolation='linear'``).
    """

    def __init__(self, data: Sequence[float]) -> None:
        if len(data) == 0:
            raise ValueError("data must be non-empty")
        self._sorted = sorted(data)
        self._n = len(self._sorted)

    def quantile(self, q: float) -> float:
        """Return the *q*-th quantile (0 ≤ q ≤ 1)."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        if self._n == 1:
            return self._sorted[0]
        pos = q * (self._n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, self._n - 1)
        frac = pos - lo
        return self._sorted[lo] * (1 - frac) + self._sorted[hi] * frac

    def median(self) -> float:
        """Median (0.5 quantile)."""
        return self.quantile(0.5)

    def iqr(self) -> float:
        """Inter-quartile range."""
        return self.quantile(0.75) - self.quantile(0.25)

    def percentile(self, p: float) -> float:
        """Return the *p*-th percentile (0 ≤ p ≤ 100)."""
        return self.quantile(p / 100.0)


# ---------------------------------------------------------------------------
# Histogram Estimator
# ---------------------------------------------------------------------------


class HistogramEstimator:
    """Adaptive histogram with automatic bin-width selection.

    Supports Freedman–Diaconis and Sturges rules for choosing the number
    of bins.
    """

    def __init__(self, data: Sequence[float],
                 method: str = "freedman_diaconis") -> None:
        if len(data) == 0:
            raise ValueError("data must be non-empty")
        self._data = sorted(data)
        self._n = len(self._data)
        self._method = method
        self._bins: Optional[int] = None
        self._edges: Optional[list[float]] = None
        self._counts: Optional[list[int]] = None

    def _freedman_diaconis_bins(self) -> int:
        q = QuantileEstimator(self._data)
        iqr = q.iqr()
        if iqr == 0:
            return max(int(math.sqrt(self._n)), 1)
        width = 2.0 * iqr * (self._n ** (-1.0 / 3.0))
        data_range = self._data[-1] - self._data[0]
        if width <= 0 or data_range <= 0:
            return max(int(math.sqrt(self._n)), 1)
        return max(int(math.ceil(data_range / width)), 1)

    def _sturges_bins(self) -> int:
        return max(int(math.ceil(math.log2(self._n) + 1)), 1)

    @property
    def n_bins(self) -> int:
        """Number of bins chosen by the selected rule."""
        if self._bins is None:
            if self._method == "sturges":
                self._bins = self._sturges_bins()
            else:
                self._bins = self._freedman_diaconis_bins()
        return self._bins

    def compute(self) -> Tuple[list[float], list[int]]:
        """Compute histogram edges and counts.

        Returns ``(edges, counts)`` where ``len(edges) == n_bins + 1``
        and ``len(counts) == n_bins``.
        """
        if self._edges is not None and self._counts is not None:
            return self._edges, self._counts

        k = self.n_bins
        lo = self._data[0]
        hi = self._data[-1]
        if lo == hi:
            self._edges = [lo - 0.5, hi + 0.5]
            self._counts = [self._n]
            return self._edges, self._counts

        width = (hi - lo) / k
        edges = [lo + i * width for i in range(k + 1)]
        edges[-1] = hi  # avoid floating-point overshoot

        counts = [0] * k
        for v in self._data:
            idx = int((v - lo) / width)
            idx = min(idx, k - 1)
            counts[idx] += 1

        self._edges = edges
        self._counts = counts
        return edges, counts

    def density(self) -> Tuple[list[float], list[float]]:
        """Return midpoints and density estimates for each bin."""
        edges, counts = self.compute()
        total = sum(counts)
        midpoints: list[float] = []
        densities: list[float] = []
        for i in range(len(counts)):
            w = edges[i + 1] - edges[i]
            midpoints.append((edges[i] + edges[i + 1]) / 2.0)
            densities.append(counts[i] / (total * w) if (total * w) > 0 else 0.0)
        return midpoints, densities


# ---------------------------------------------------------------------------
# Empirical Distribution
# ---------------------------------------------------------------------------


class EmpiricalDistribution:
    """Non-parametric empirical distribution from sample data.

    Provides CDF, quantile, and moment estimation.
    """

    def __init__(self, data: Sequence[float]) -> None:
        if len(data) == 0:
            raise ValueError("data must be non-empty")
        self._sorted = sorted(data)
        self._n = len(self._sorted)
        self._moments = MomentEstimator(data)
        self._quantiles = QuantileEstimator(data)

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._moments.mean

    @property
    def variance(self) -> float:
        return self._moments.variance

    @property
    def std(self) -> float:
        return self._moments.std

    def cdf(self, x: float) -> float:
        """Empirical CDF: proportion of observations ≤ *x*."""
        lo, hi = 0, self._n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / self._n

    def quantile(self, q: float) -> float:
        """Quantile function (inverse CDF)."""
        return self._quantiles.quantile(q)

    def sample(self, n: int, seed: Optional[int] = None) -> list[float]:
        """Draw *n* samples with replacement."""
        rng = random.Random(seed)
        return [rng.choice(self._sorted) for _ in range(n)]


# ---------------------------------------------------------------------------
# Discrete Distribution
# ---------------------------------------------------------------------------


class DiscreteDistribution:
    """Discrete distribution defined by a probability mass function (PMF)."""

    def __init__(self, pmf: Dict[str, float]) -> None:
        if not pmf:
            raise ValueError("PMF must be non-empty")
        total = sum(pmf.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"PMF must sum to 1.0 (got {total})")
        self._pmf = dict(pmf)
        self._support = sorted(pmf.keys())

    @classmethod
    def from_counts(cls, counts: Dict[str, int]) -> "DiscreteDistribution":
        """Build a distribution from raw counts."""
        total = sum(counts.values())
        if total == 0:
            raise ValueError("Counts must be non-empty")
        pmf = {k: v / total for k, v in counts.items()}
        return cls(pmf)

    @classmethod
    def from_samples(cls, data: Sequence) -> "DiscreteDistribution":
        """Build a distribution from raw sample values."""
        c: Counter = Counter(data)
        return cls.from_counts(dict(c))

    @property
    def support(self) -> list[str]:
        return list(self._support)

    def prob(self, x: str) -> float:
        """P(X = x)."""
        return self._pmf.get(x, 0.0)

    def entropy(self) -> float:
        """Shannon entropy in nats."""
        h = 0.0
        for p in self._pmf.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def sample(self, n: int, seed: Optional[int] = None) -> list[str]:
        """Draw *n* samples from the distribution."""
        rng = random.Random(seed)
        vals = list(self._pmf.keys())
        weights = [self._pmf[v] for v in vals]
        cum: list[float] = []
        total = 0.0
        for w in weights:
            total += w
            cum.append(total)
        result: list[str] = []
        for _ in range(n):
            r = rng.random()
            for i, c in enumerate(cum):
                if r <= c:
                    result.append(vals[i])
                    break
            else:
                result.append(vals[-1])
        return result


# ---------------------------------------------------------------------------
# Continuous Distribution Approximation (KDE)
# ---------------------------------------------------------------------------


class ContinuousDistributionApprox:
    """Kernel density estimation with a Gaussian kernel.

    Bandwidth is selected via Silverman's rule of thumb unless explicitly
    specified.
    """

    def __init__(self, data: Sequence[float],
                 bandwidth: Optional[float] = None) -> None:
        if len(data) == 0:
            raise ValueError("data must be non-empty")
        self._data = list(data)
        self._n = len(self._data)
        self._moments = MomentEstimator(data)

        if bandwidth is not None:
            self._bw = bandwidth
        else:
            self._bw = self._silverman_bandwidth()

    def _silverman_bandwidth(self) -> float:
        """Silverman's rule of thumb: h = 0.9 * min(sd, IQR/1.34) * n^{-1/5}."""
        sd = self._moments.std
        q = QuantileEstimator(self._data)
        iqr = q.iqr()
        spread = min(sd, iqr / 1.34) if iqr > 0 else sd
        if spread <= 0:
            spread = 1.0
        return 0.9 * spread * (self._n ** (-0.2))

    @property
    def bandwidth(self) -> float:
        return self._bw

    def pdf(self, x: float) -> float:
        """Estimated probability density at *x*."""
        h = self._bw
        total = 0.0
        for xi in self._data:
            z = (x - xi) / h
            total += math.exp(-0.5 * z * z)
        return total / (self._n * h * math.sqrt(2.0 * math.pi))

    def pdf_grid(self, lo: float, hi: float, n_points: int = 200) -> Tuple[
        list[float], list[float]
    ]:
        """Evaluate the PDF on a regular grid.

        Returns ``(xs, densities)`` lists of length *n_points*.
        """
        step = (hi - lo) / max(n_points - 1, 1)
        xs = [lo + i * step for i in range(n_points)]
        densities = [self.pdf(x) for x in xs]
        return xs, densities

    def cdf(self, x: float) -> float:
        """Estimated CDF at *x* (by numerical integration of the KDE)."""
        h = self._bw
        total = 0.0
        for xi in self._data:
            z = (x - xi) / h
            total += 0.5 * (1.0 + _erf_approx(z / math.sqrt(2.0)))
        return total / self._n


def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun approximation to the error function."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# Entropy estimation
# ---------------------------------------------------------------------------


def entropy_plugin(data: Sequence) -> float:
    """Plugin (maximum likelihood) estimator of Shannon entropy (nats).

    Computes H = -sum p_i log(p_i) from sample frequencies.
    """
    n = len(data)
    if n == 0:
        return 0.0
    counts: Counter = Counter(data)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p)
    return h


def entropy_miller_madow(data: Sequence) -> float:
    """Miller–Madow bias-corrected entropy estimator (nats).

    Adds the correction term (m - 1) / (2n) where m is the number of
    categories with non-zero counts.
    """
    n = len(data)
    if n == 0:
        return 0.0
    counts: Counter = Counter(data)
    m = len(counts)
    h = entropy_plugin(data)
    correction = (m - 1) / (2.0 * n)
    return h + correction


def entropy_jackknife(data: Sequence) -> float:
    """Jackknife bias-corrected entropy estimator (nats).

    Applies first-order jackknife bias correction by computing leave-one-out
    estimates.
    """
    n = len(data)
    if n <= 1:
        return 0.0
    h_full = entropy_plugin(data)
    counts: Counter = Counter(data)

    h_loo_sum = 0.0
    for i in range(n):
        val = data[i]
        c = counts[val]
        # Incremental leave-one-out: adjust the full entropy
        h_loo = 0.0
        for v, cnt in counts.items():
            adj = cnt - (1 if v == val else 0)
            if adj > 0:
                p = adj / (n - 1)
                h_loo -= p * math.log(p)
        h_loo_sum += h_loo

    h_loo_mean = h_loo_sum / n
    # Jackknife estimate: n * h_full - (n-1) * h_loo_mean
    return n * h_full - (n - 1) * h_loo_mean


# ---------------------------------------------------------------------------
# KL and JS Divergence
# ---------------------------------------------------------------------------


class KLDivergence:
    """Kullback–Leibler divergence estimation between two discrete distributions.

    KL(P || Q) = sum_x P(x) log(P(x) / Q(x))
    """

    @staticmethod
    def from_counts(p_counts: Dict[str, int],
                    q_counts: Dict[str, int],
                    smoothing: float = 1e-10) -> float:
        """Compute KL divergence from count dictionaries.

        A small *smoothing* value is added to avoid log(0).
        """
        all_keys = set(p_counts) | set(q_counts)
        p_total = sum(p_counts.values())
        q_total = sum(q_counts.values())
        if p_total == 0 or q_total == 0:
            raise ValueError("Count dictionaries must be non-empty")

        kl = 0.0
        for k in all_keys:
            p = (p_counts.get(k, 0) + smoothing) / (p_total + smoothing * len(all_keys))
            q = (q_counts.get(k, 0) + smoothing) / (q_total + smoothing * len(all_keys))
            if p > 0:
                kl += p * math.log(p / q)
        return kl

    @staticmethod
    def from_samples(p_data: Sequence, q_data: Sequence,
                     smoothing: float = 1e-10) -> float:
        """Compute KL divergence from raw samples."""
        p_counts = dict(Counter(p_data))
        q_counts = dict(Counter(q_data))
        return KLDivergence.from_counts(p_counts, q_counts, smoothing)

    @staticmethod
    def from_distributions(p: DiscreteDistribution,
                           q: DiscreteDistribution,
                           smoothing: float = 1e-10) -> float:
        """KL(P || Q) from two :class:`DiscreteDistribution` objects."""
        all_keys = set(p.support) | set(q.support)
        kl = 0.0
        for k in all_keys:
            pk = p.prob(k) + smoothing
            qk = q.prob(k) + smoothing
            if pk > 0:
                kl += pk * math.log(pk / qk)
        return kl


class JSDivergence:
    """Jensen–Shannon divergence (symmetric, bounded by log(2))."""

    @staticmethod
    def from_counts(p_counts: Dict[str, int],
                    q_counts: Dict[str, int],
                    smoothing: float = 1e-10) -> float:
        """Compute JS divergence from count dictionaries."""
        all_keys = set(p_counts) | set(q_counts)
        p_total = sum(p_counts.values())
        q_total = sum(q_counts.values())
        if p_total == 0 or q_total == 0:
            raise ValueError("Count dictionaries must be non-empty")

        p_probs: dict[str, float] = {}
        q_probs: dict[str, float] = {}
        n_keys = len(all_keys)
        for k in all_keys:
            p_probs[k] = (p_counts.get(k, 0) + smoothing) / (p_total + smoothing * n_keys)
            q_probs[k] = (q_counts.get(k, 0) + smoothing) / (q_total + smoothing * n_keys)

        js = 0.0
        for k in all_keys:
            m = 0.5 * (p_probs[k] + q_probs[k])
            if p_probs[k] > 0 and m > 0:
                js += 0.5 * p_probs[k] * math.log(p_probs[k] / m)
            if q_probs[k] > 0 and m > 0:
                js += 0.5 * q_probs[k] * math.log(q_probs[k] / m)
        return js

    @staticmethod
    def from_samples(p_data: Sequence, q_data: Sequence,
                     smoothing: float = 1e-10) -> float:
        """Compute JS divergence from raw samples."""
        return JSDivergence.from_counts(
            dict(Counter(p_data)), dict(Counter(q_data)), smoothing,
        )

    @staticmethod
    def from_distributions(p: DiscreteDistribution,
                           q: DiscreteDistribution,
                           smoothing: float = 1e-10) -> float:
        """JS divergence from two :class:`DiscreteDistribution` objects."""
        all_keys = set(p.support) | set(q.support)
        js = 0.0
        for k in all_keys:
            pk = p.prob(k) + smoothing
            qk = q.prob(k) + smoothing
            m = 0.5 * (pk + qk)
            if pk > 0 and m > 0:
                js += 0.5 * pk * math.log(pk / m)
            if qk > 0 and m > 0:
                js += 0.5 * qk * math.log(qk / m)
        return js


# ---------------------------------------------------------------------------
# Two-Sample Test (Kolmogorov–Smirnov style)
# ---------------------------------------------------------------------------


class TwoSampleTest:
    """Non-parametric two-sample test comparing two empirical distributions.

    Computes the Kolmogorov–Smirnov statistic and estimates a *p*-value via
    the asymptotic Kolmogorov distribution.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    @staticmethod
    def _ks_statistic(x: Sequence[float], y: Sequence[float]) -> float:
        """Two-sample KS statistic (supremum of |F_x - F_y|)."""
        n1, n2 = len(x), len(y)
        all_vals = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
        d_max = 0.0
        ecdf1 = 0.0
        ecdf2 = 0.0
        for val, group in all_vals:
            if group == 0:
                ecdf1 += 1.0 / n1
            else:
                ecdf2 += 1.0 / n2
            d_max = max(d_max, abs(ecdf1 - ecdf2))
        return d_max

    @staticmethod
    def _ks_p_value(d: float, n1: int, n2: int) -> float:
        """Asymptotic *p*-value for the two-sample KS statistic.

        Uses the limiting distribution:
        P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} exp(-2 k^2 lambda^2)
        where lambda = d * sqrt(n1*n2 / (n1+n2)).
        """
        en = math.sqrt(n1 * n2 / (n1 + n2))
        lam = (en + 0.12 + 0.11 / en) * d
        if lam == 0:
            return 1.0
        p = 0.0
        for k in range(1, 101):
            sign = 1 if k % 2 == 1 else -1
            term = sign * math.exp(-2.0 * k * k * lam * lam)
            p += term
            if abs(term) < 1e-12:
                break
        p *= 2.0
        return max(min(p, 1.0), 0.0)

    def test(self, x: Sequence[float], y: Sequence[float]) -> dict:
        """Run the two-sample KS test.

        Returns a dict with keys: ``statistic``, ``p_value``, ``reject_null``,
        ``alpha``, ``n1``, ``n2``.
        """
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Both samples must be non-empty")
        d = self._ks_statistic(x, y)
        p = self._ks_p_value(d, len(x), len(y))
        return {
            "statistic": d,
            "p_value": p,
            "reject_null": p < self.alpha,
            "alpha": self.alpha,
            "n1": len(x),
            "n2": len(y),
        }
