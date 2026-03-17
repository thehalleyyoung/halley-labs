"""taintflow.empirical.bootstrap – Bootstrap confidence intervals for leakage.

Provides bootstrap and jackknife methods for constructing confidence
intervals on mutual-information–based leakage estimates, as well as a
refinement procedure that combines an abstract taint-flow bound with an
empirical KSG estimate.

Key classes:
* :class:`BootstrapCI` – percentile, BCa, and basic bootstrap CI.
* :class:`BootstrapLeakageEstimator` – bootstrap the full leakage pipeline.
* :class:`LeakageBoundRefinement` – combine abstract and empirical bounds.
* :class:`StratifiedBootstrap` – bootstrap respecting partitions.
* :class:`JackknifeEstimator` – leave-one-out bias and variance estimation.
* :class:`ConvergenceDiagnostic` – assess bootstrap convergence.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from taintflow.empirical.ksg import KSGEstimator, MutualInformationResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap procedure.

    Attributes
    ----------
    point_estimate:
        Original (non-resampled) statistic.
    ci_lower:
        Lower bound of the confidence interval.
    ci_upper:
        Upper bound of the confidence interval.
    confidence_level:
        Nominal coverage probability (e.g. 0.95).
    n_bootstrap:
        Number of bootstrap replicates.
    method:
        CI method used (``"percentile"``, ``"bca"``, or ``"basic"``).
    bootstrap_estimates:
        All bootstrap replicate values (sorted).
    bias:
        Estimated bias (mean of replicates − point estimate).
    std_error:
        Bootstrap standard error.
    """

    point_estimate: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    confidence_level: float = 0.95
    n_bootstrap: int = 0
    method: str = "percentile"
    bootstrap_estimates: Tuple[float, ...] = ()
    bias: float = 0.0
    std_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "method": self.method,
            "bootstrap_estimates": list(self.bootstrap_estimates),
            "bias": self.bias,
            "std_error": self.std_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BootstrapResult:
        return cls(
            point_estimate=float(data["point_estimate"]),
            ci_lower=float(data.get("ci_lower", 0.0)),
            ci_upper=float(data.get("ci_upper", 0.0)),
            confidence_level=float(data.get("confidence_level", 0.95)),
            n_bootstrap=int(data.get("n_bootstrap", 0)),
            method=str(data.get("method", "percentile")),
            bootstrap_estimates=tuple(data.get("bootstrap_estimates", ())),
            bias=float(data.get("bias", 0.0)),
            std_error=float(data.get("std_error", 0.0)),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not 0.0 < self.confidence_level < 1.0:
            errors.append("confidence_level must be in (0, 1)")
        if self.n_bootstrap < 0:
            errors.append("n_bootstrap must be non-negative")
        if self.ci_lower > self.ci_upper and self.n_bootstrap > 0:
            errors.append("ci_lower must not exceed ci_upper")
        if self.method not in ("percentile", "bca", "basic"):
            errors.append("method must be 'percentile', 'bca', or 'basic'")
        return errors


# ---------------------------------------------------------------------------
# Normal CDF / quantile helpers (standard-library only)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (rational approximation).

    Uses the Beasley–Springer–Moro algorithm.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00,  2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


# ---------------------------------------------------------------------------
# BootstrapCI
# ---------------------------------------------------------------------------

class BootstrapCI:
    """Generic bootstrap confidence-interval engine.

    Wraps any scalar-valued statistic function ``stat_fn(data) -> float``
    and produces confidence intervals by resampling *data*.

    Parameters
    ----------
    stat_fn:
        Callable that takes a list of data rows and returns a float.
    n_bootstrap:
        Number of bootstrap replicates.
    confidence_level:
        Desired coverage (default 0.95).
    method:
        ``"percentile"``, ``"bca"``, or ``"basic"``.
    seed:
        Random seed.
    """

    def __init__(
        self,
        stat_fn: Callable[[List[Any]], float],
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        method: str = "percentile",
        seed: Optional[int] = None,
    ) -> None:
        if method not in ("percentile", "bca", "basic"):
            raise ValueError("method must be 'percentile', 'bca', or 'basic'")
        self.stat_fn = stat_fn
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self._rng = random.Random(seed)

    def compute(self, data: List[Any]) -> BootstrapResult:
        """Run the bootstrap and return a :class:`BootstrapResult`.

        Parameters
        ----------
        data:
            List of sample rows.
        """
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        theta_hat = self.stat_fn(data)

        replicates: List[float] = []
        for _ in range(self.n_bootstrap):
            sample = [data[self._rng.randint(0, n - 1)] for _ in range(n)]
            replicates.append(self.stat_fn(sample))

        replicates.sort()
        b_mean = sum(replicates) / len(replicates)
        bias = b_mean - theta_hat
        variance = sum((v - b_mean) ** 2 for v in replicates) / max(len(replicates) - 1, 1)
        se = math.sqrt(variance)

        if self.method == "percentile":
            lo, hi = self._percentile_ci(replicates)
        elif self.method == "basic":
            lo, hi = self._basic_ci(replicates, theta_hat)
        else:
            lo, hi = self._bca_ci(replicates, theta_hat, data)

        return BootstrapResult(
            point_estimate=theta_hat,
            ci_lower=lo,
            ci_upper=hi,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            method=self.method,
            bootstrap_estimates=tuple(replicates),
            bias=bias,
            std_error=se,
        )

    # -- CI methods ---------------------------------------------------------

    def _percentile_ci(
        self, replicates: List[float]
    ) -> Tuple[float, float]:
        """Percentile method: CI = [θ*_{α/2}, θ*_{1−α/2}]."""
        alpha = 1.0 - self.confidence_level
        b = len(replicates)
        lo = replicates[max(0, int(math.floor(alpha / 2 * b)))]
        hi = replicates[min(b - 1, int(math.floor((1 - alpha / 2) * b)))]
        return lo, hi

    def _basic_ci(
        self, replicates: List[float], theta_hat: float
    ) -> Tuple[float, float]:
        """Basic (reverse percentile) method: CI = [2θ̂ − θ*_{1−α/2}, 2θ̂ − θ*_{α/2}]."""
        alpha = 1.0 - self.confidence_level
        b = len(replicates)
        q_lo = replicates[max(0, int(math.floor(alpha / 2 * b)))]
        q_hi = replicates[min(b - 1, int(math.floor((1 - alpha / 2) * b)))]
        return 2 * theta_hat - q_hi, 2 * theta_hat - q_lo

    def _bca_ci(
        self,
        replicates: List[float],
        theta_hat: float,
        data: List[Any],
    ) -> Tuple[float, float]:
        """Bias-Corrected and Accelerated (BCa) confidence interval.

        Adjusts percentile endpoints using bias-correction factor *z0*
        and acceleration factor *a* estimated via the jackknife.
        """
        b = len(replicates)
        # Bias-correction z0
        count_below = sum(1 for v in replicates if v < theta_hat)
        prop_below = count_below / b
        prop_below = max(1e-10, min(prop_below, 1 - 1e-10))
        z0 = _norm_ppf(prop_below)

        # Acceleration via jackknife
        n = len(data)
        jack_vals: List[float] = []
        for i in range(n):
            jack_sample = data[:i] + data[i + 1:]
            jack_vals.append(self.stat_fn(jack_sample))

        jack_mean = sum(jack_vals) / n
        num = sum((jack_mean - v) ** 3 for v in jack_vals)
        den = sum((jack_mean - v) ** 2 for v in jack_vals)
        den = 6.0 * (max(den, 1e-30) ** 1.5)
        acc = num / den

        alpha = 1.0 - self.confidence_level
        z_alpha_lo = _norm_ppf(alpha / 2)
        z_alpha_hi = _norm_ppf(1 - alpha / 2)

        # Adjusted quantile positions
        def _adj(z_alpha: float) -> float:
            numer = z0 + z_alpha
            adj_z = z0 + numer / max(1 - acc * numer, 1e-10)
            return _norm_cdf(adj_z)

        p_lo = _adj(z_alpha_lo)
        p_hi = _adj(z_alpha_hi)

        idx_lo = max(0, min(b - 1, int(math.floor(p_lo * b))))
        idx_hi = max(0, min(b - 1, int(math.floor(p_hi * b))))
        return replicates[idx_lo], replicates[idx_hi]


# ---------------------------------------------------------------------------
# Bootstrap leakage estimator (wraps KSG + bootstrap)
# ---------------------------------------------------------------------------

class BootstrapLeakageEstimator:
    """Bootstrap the entire KSG leakage estimation pipeline.

    Resamples (X, Y) pairs *n_bootstrap* times and feeds each resample
    through :class:`KSGEstimator` to produce a bootstrap distribution of
    MI estimates.

    Parameters
    ----------
    k:
        k-NN parameter.
    n_bootstrap:
        Number of bootstrap replicates.
    confidence_level:
        Desired CI coverage.
    method:
        ``"percentile"``, ``"bca"``, or ``"basic"``.
    seed:
        Random seed.
    """

    def __init__(
        self,
        k: int = 3,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        method: str = "percentile",
        seed: Optional[int] = None,
    ) -> None:
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self._seed = seed

    def estimate(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> BootstrapResult:
        """Run the bootstrapped leakage estimator.

        Parameters
        ----------
        x, y:
            N × d_x, N × d_y data arrays.
        """
        paired = list(zip(x, y))

        def _mi_stat(data: List[Any]) -> float:
            xs = [row[0] for row in data]
            ys = [row[1] for row in data]
            est = KSGEstimator(k=self.k, seed=self._seed)
            return max(est.estimate(xs, ys).estimate, 0.0)

        ci = BootstrapCI(
            stat_fn=_mi_stat,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            method=self.method,
            seed=self._seed,
        )
        return ci.compute(paired)


# ---------------------------------------------------------------------------
# Leakage bound refinement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinedBound:
    """Result of combining an abstract bound with an empirical estimate.

    Attributes
    ----------
    abstract_bound:
        Original taint-flow bound (bits).
    empirical_estimate:
        KSG MI point estimate (bits).
    confidence_margin:
        Half-width of the CI (bits).
    final_bound:
        min(abstract_bound, empirical_estimate + confidence_margin).
    method:
        Description of how the bound was refined.
    """

    abstract_bound: float
    empirical_estimate: float
    confidence_margin: float
    final_bound: float
    method: str = "min(abstract, empirical + margin)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstract_bound": self.abstract_bound,
            "empirical_estimate": self.empirical_estimate,
            "confidence_margin": self.confidence_margin,
            "final_bound": self.final_bound,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RefinedBound:
        return cls(
            abstract_bound=float(data["abstract_bound"]),
            empirical_estimate=float(data["empirical_estimate"]),
            confidence_margin=float(data["confidence_margin"]),
            final_bound=float(data["final_bound"]),
            method=str(data.get("method", "min(abstract, empirical + margin)")),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.abstract_bound < 0:
            errors.append("abstract_bound must be non-negative")
        if self.confidence_margin < 0:
            errors.append("confidence_margin must be non-negative")
        return errors


class LeakageBoundRefinement:
    """Combine an abstract taint-flow bound with an empirical KSG estimate.

    The refined bound is::

        final = min(abstract_bound, ksg_estimate + confidence_margin)

    where *ksg_estimate* is the bootstrap point estimate converted to bits,
    and *confidence_margin* is the upper CI endpoint minus the point
    estimate (also in bits).

    Parameters
    ----------
    k:
        k-NN parameter.
    n_bootstrap:
        Number of bootstrap replicates for CI.
    confidence_level:
        CI coverage.
    method:
        Bootstrap CI method.
    seed:
        Random seed.
    """

    LN2 = math.log(2.0)

    def __init__(
        self,
        k: int = 3,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        method: str = "bca",
        seed: Optional[int] = None,
    ) -> None:
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self._seed = seed

    def refine(
        self,
        x: List[List[float]],
        y: List[List[float]],
        abstract_bound_bits: float,
    ) -> RefinedBound:
        """Compute the refined leakage bound.

        Parameters
        ----------
        x, y:
            Sample data.
        abstract_bound_bits:
            Upper bound from the abstract taint-flow analysis (in bits).

        Returns
        -------
        RefinedBound
        """
        estimator = BootstrapLeakageEstimator(
            k=self.k,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            method=self.method,
            seed=self._seed,
        )
        result = estimator.estimate(x, y)

        # Convert nats → bits
        est_bits = result.point_estimate / self.LN2
        ci_upper_bits = result.ci_upper / self.LN2
        margin_bits = max(ci_upper_bits - est_bits, 0.0)

        empirical_upper = est_bits + margin_bits
        final = min(abstract_bound_bits, empirical_upper)
        final = max(final, 0.0)

        return RefinedBound(
            abstract_bound=abstract_bound_bits,
            empirical_estimate=est_bits,
            confidence_margin=margin_bits,
            final_bound=final,
        )


# ---------------------------------------------------------------------------
# Convergence diagnostic
# ---------------------------------------------------------------------------

class ConvergenceDiagnostic:
    """Assess whether the bootstrap distribution has converged.

    Convergence is checked by comparing the CI width for the first
    *B/2* replicates with the CI width for all *B* replicates; if the
    relative change is below *tolerance* the distribution is deemed
    converged.  Additionally the running mean is inspected.

    Parameters
    ----------
    tolerance:
        Relative CI-width change threshold (default 0.05 = 5 %).
    min_replicates:
        Minimum number of replicates before convergence can be declared.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        min_replicates: int = 200,
    ) -> None:
        self.tolerance = tolerance
        self.min_replicates = min_replicates

    def check(self, result: BootstrapResult) -> Dict[str, Any]:
        """Evaluate convergence of a bootstrap result.

        Returns
        -------
        dict
            Keys: ``"converged"`` (bool), ``"ci_width_change"`` (float),
            ``"running_mean_change"`` (float), ``"message"`` (str).
        """
        reps = list(result.bootstrap_estimates)
        b = len(reps)

        if b < self.min_replicates:
            return {
                "converged": False,
                "ci_width_change": float("inf"),
                "running_mean_change": float("inf"),
                "message": (
                    f"Only {b} replicates; need at least {self.min_replicates}"
                ),
            }

        half = b // 2
        first_half = sorted(reps[:half])
        full = sorted(reps)

        alpha = 1.0 - result.confidence_level
        lo_pct = alpha / 2
        hi_pct = 1 - alpha / 2

        def _ci_width(arr: List[float]) -> float:
            n = len(arr)
            return arr[min(n - 1, int(hi_pct * n))] - arr[max(0, int(lo_pct * n))]

        w_half = _ci_width(first_half)
        w_full = _ci_width(full)
        denom = max(abs(w_full), 1e-12)
        ci_change = abs(w_full - w_half) / denom

        mean_half = sum(reps[:half]) / half
        mean_full = sum(reps) / b
        mean_denom = max(abs(mean_full), 1e-12)
        mean_change = abs(mean_full - mean_half) / mean_denom

        converged = ci_change < self.tolerance and mean_change < self.tolerance

        return {
            "converged": converged,
            "ci_width_change": ci_change,
            "running_mean_change": mean_change,
            "message": "Converged" if converged else "Not yet converged",
        }


# ---------------------------------------------------------------------------
# Stratified bootstrap
# ---------------------------------------------------------------------------

class StratifiedBootstrap:
    """Bootstrap that preserves the train/test partition structure.

    Resampling is performed *within* each stratum (e.g. train rows and
    test rows are resampled independently) so that the partition sizes
    remain fixed across replicates.

    Parameters
    ----------
    stat_fn:
        Callable ``(data, strata) -> float``.
    n_bootstrap:
        Number of replicates.
    confidence_level:
        CI coverage.
    method:
        ``"percentile"``, ``"bca"``, or ``"basic"``.
    seed:
        Random seed.
    """

    def __init__(
        self,
        stat_fn: Callable[[List[Any], List[int]], float],
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        method: str = "percentile",
        seed: Optional[int] = None,
    ) -> None:
        if method not in ("percentile", "bca", "basic"):
            raise ValueError("method must be 'percentile', 'bca', or 'basic'")
        self.stat_fn = stat_fn
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self._rng = random.Random(seed)

    def compute(
        self,
        data: List[Any],
        strata: List[int],
    ) -> BootstrapResult:
        """Run the stratified bootstrap.

        Parameters
        ----------
        data:
            Full dataset (list of rows).
        strata:
            Integer stratum label for each row (e.g. 0 = train, 1 = test).
        """
        n = len(data)
        if n != len(strata):
            raise ValueError("data and strata must have the same length")

        theta_hat = self.stat_fn(data, strata)

        # Build stratum → index mapping
        stratum_map: Dict[int, List[int]] = {}
        for i, s in enumerate(strata):
            stratum_map.setdefault(s, []).append(i)

        replicates: List[float] = []
        for _ in range(self.n_bootstrap):
            boot_data: List[Any] = [None] * n
            boot_strata: List[int] = [0] * n
            pos = 0
            for s_label, indices in stratum_map.items():
                ns = len(indices)
                boot_idx = [indices[self._rng.randint(0, ns - 1)] for _ in range(ns)]
                for bi in boot_idx:
                    boot_data[pos] = data[bi]
                    boot_strata[pos] = s_label
                    pos += 1
            replicates.append(self.stat_fn(boot_data[:pos], boot_strata[:pos]))

        replicates.sort()
        b_mean = sum(replicates) / len(replicates)
        bias = b_mean - theta_hat
        var = sum((v - b_mean) ** 2 for v in replicates) / max(len(replicates) - 1, 1)
        se = math.sqrt(var)

        lo, hi = self._compute_ci(replicates, theta_hat, data, strata)

        return BootstrapResult(
            point_estimate=theta_hat,
            ci_lower=lo,
            ci_upper=hi,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            method=self.method,
            bootstrap_estimates=tuple(replicates),
            bias=bias,
            std_error=se,
        )

    def _compute_ci(
        self,
        replicates: List[float],
        theta_hat: float,
        data: List[Any],
        strata: List[int],
    ) -> Tuple[float, float]:
        alpha = 1.0 - self.confidence_level
        b = len(replicates)

        if self.method == "percentile":
            lo = replicates[max(0, int(math.floor(alpha / 2 * b)))]
            hi = replicates[min(b - 1, int(math.floor((1 - alpha / 2) * b)))]
            return lo, hi

        if self.method == "basic":
            q_lo = replicates[max(0, int(math.floor(alpha / 2 * b)))]
            q_hi = replicates[min(b - 1, int(math.floor((1 - alpha / 2) * b)))]
            return 2 * theta_hat - q_hi, 2 * theta_hat - q_lo

        # BCa
        count_below = sum(1 for v in replicates if v < theta_hat)
        prop_below = max(1e-10, min(count_below / b, 1 - 1e-10))
        z0 = _norm_ppf(prop_below)

        # Jackknife acceleration
        n = len(data)
        jack: List[float] = []
        for i in range(n):
            jd = data[:i] + data[i + 1:]
            js = strata[:i] + strata[i + 1:]
            jack.append(self.stat_fn(jd, js))

        jm = sum(jack) / n
        num = sum((jm - v) ** 3 for v in jack)
        den = 6.0 * max(sum((jm - v) ** 2 for v in jack), 1e-30) ** 1.5
        acc = num / den

        z_lo = _norm_ppf(alpha / 2)
        z_hi = _norm_ppf(1 - alpha / 2)

        def _adj(z_a: float) -> float:
            numer = z0 + z_a
            return _norm_cdf(z0 + numer / max(1 - acc * numer, 1e-10))

        idx_lo = max(0, min(b - 1, int(math.floor(_adj(z_lo) * b))))
        idx_hi = max(0, min(b - 1, int(math.floor(_adj(z_hi) * b))))
        return replicates[idx_lo], replicates[idx_hi]


# ---------------------------------------------------------------------------
# Jackknife estimator
# ---------------------------------------------------------------------------

class JackknifeEstimator:
    """Leave-one-out jackknife for bias and variance estimation.

    Computes the jackknife estimate of a statistic, its bias, and
    standard error.

    Parameters
    ----------
    stat_fn:
        Callable ``(data) -> float``.
    """

    def __init__(self, stat_fn: Callable[[List[Any]], float]) -> None:
        self.stat_fn = stat_fn

    def estimate(self, data: List[Any]) -> Dict[str, Any]:
        """Run the jackknife.

        Returns
        -------
        dict
            Keys: ``"theta_hat"``, ``"jackknife_estimate"``, ``"bias"``,
            ``"std_error"``, ``"pseudovalues"``.
        """
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        theta_hat = self.stat_fn(data)

        leave_one: List[float] = []
        for i in range(n):
            leave_one.append(self.stat_fn(data[:i] + data[i + 1:]))

        theta_jack_mean = sum(leave_one) / n
        bias = (n - 1) * (theta_jack_mean - theta_hat)
        jack_estimate = theta_hat - bias

        # Pseudovalues for variance estimation
        pseudovalues = [n * theta_hat - (n - 1) * t_i for t_i in leave_one]
        pv_mean = sum(pseudovalues) / n
        var = sum((p - pv_mean) ** 2 for p in pseudovalues) / (n * (n - 1))
        se = math.sqrt(var)

        return {
            "theta_hat": theta_hat,
            "jackknife_estimate": jack_estimate,
            "bias": bias,
            "std_error": se,
            "pseudovalues": pseudovalues,
        }
