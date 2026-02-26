"""Statistical analysis for importance-weighted samples.

Provides confidence bounds (Hoeffding, Bernstein, CLT), stratified and
control-variate variance reduction, Bayesian estimation with abstract-
interpretation priors, and convergence diagnostics.
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
from scipy import stats as scipy_stats

from marace.sampling.importance_sampling import ImportanceWeights


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

class SampleStatistics:
    """Compute statistics from importance-weighted samples."""

    @staticmethod
    def weighted_mean(values: np.ndarray, weights: ImportanceWeights) -> float:
        """Self-normalised importance-weighted mean."""
        w = weights.normalised_weights()
        return float(np.sum(w * values))

    @staticmethod
    def weighted_variance(values: np.ndarray, weights: ImportanceWeights) -> float:
        """Importance-weighted variance (Bessel-corrected for ESS)."""
        w = weights.normalised_weights()
        mean = float(np.sum(w * values))
        ess = 1.0 / float(np.sum(w ** 2))
        var = float(np.sum(w * (values - mean) ** 2))
        # Bessel correction for effective sample size
        if ess > 1:
            var *= ess / (ess - 1)
        return var

    @staticmethod
    def weighted_quantile(
        values: np.ndarray,
        weights: ImportanceWeights,
        quantile: float,
    ) -> float:
        """Importance-weighted quantile."""
        w = weights.normalised_weights()
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_w = w[order]
        cumulative = np.cumsum(sorted_w)

        idx = np.searchsorted(cumulative, quantile)
        idx = min(idx, len(sorted_vals) - 1)
        return float(sorted_vals[idx])

    @staticmethod
    def weighted_std(values: np.ndarray, weights: ImportanceWeights) -> float:
        """Importance-weighted standard deviation."""
        return math.sqrt(
            max(SampleStatistics.weighted_variance(values, weights), 0.0)
        )

    @staticmethod
    def weighted_skewness(values: np.ndarray, weights: ImportanceWeights) -> float:
        """Importance-weighted skewness."""
        w = weights.normalised_weights()
        mean = float(np.sum(w * values))
        std = SampleStatistics.weighted_std(values, weights)
        if std < 1e-15:
            return 0.0
        return float(np.sum(w * ((values - mean) / std) ** 3))

    @staticmethod
    def weighted_kurtosis(values: np.ndarray, weights: ImportanceWeights) -> float:
        """Importance-weighted excess kurtosis."""
        w = weights.normalised_weights()
        mean = float(np.sum(w * values))
        std = SampleStatistics.weighted_std(values, weights)
        if std < 1e-15:
            return 0.0
        return float(np.sum(w * ((values - mean) / std) ** 4)) - 3.0


# ---------------------------------------------------------------------------
# Confidence bounds
# ---------------------------------------------------------------------------

class ConfidenceBound:
    """Hoeffding and Bernstein bounds for bounded random variables."""

    @staticmethod
    def hoeffding(
        n_eff: float,
        delta: float = 0.05,
        value_range: float = 1.0,
    ) -> float:
        """Hoeffding bound: with probability >= 1-δ, the estimation error
        is at most the returned value.

        Parameters:
            n_eff: Effective sample size.
            delta: Failure probability.
            value_range: Range of the random variable (max - min).

        Returns:
            Half-width of the Hoeffding confidence interval.
        """
        if n_eff < 1:
            return float("inf")
        return value_range * math.sqrt(math.log(2.0 / delta) / (2.0 * n_eff))

    @staticmethod
    def bernstein(
        n_eff: float,
        sample_variance: float,
        delta: float = 0.05,
        value_range: float = 1.0,
    ) -> float:
        """Bernstein bound (tighter than Hoeffding when variance is small).

        Parameters:
            n_eff: Effective sample size.
            sample_variance: Estimated variance of the quantity.
            delta: Failure probability.
            value_range: Range of the random variable.

        Returns:
            Half-width of the Bernstein confidence interval.
        """
        if n_eff < 1:
            return float("inf")
        log_term = math.log(2.0 / delta)
        return math.sqrt(2.0 * sample_variance * log_term / n_eff) + (
            value_range * log_term / (3.0 * n_eff)
        )

    @staticmethod
    def clt_interval(
        estimate: float,
        std_err: float,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """CLT-based confidence interval.

        Returns:
            ``(lower, upper)`` bounds.
        """
        z = scipy_stats.norm.ppf(0.5 + confidence / 2.0)
        return (estimate - z * std_err, estimate + z * std_err)


# ---------------------------------------------------------------------------
# Stratified estimator
# ---------------------------------------------------------------------------

class StratifiedEstimator:
    """Stratified sampling for variance reduction.

    Partitions the schedule space into strata (e.g. by interaction group
    or by risk level) and samples proportionally within each stratum.
    The overall estimate combines per-stratum estimates weighted by
    stratum probability.
    """

    def __init__(
        self,
        stratum_probs: np.ndarray,
        stratum_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters:
            stratum_probs: Prior probability of each stratum (sums to 1).
            stratum_labels: Optional names for strata.
        """
        self._probs = stratum_probs / stratum_probs.sum()
        self._labels = stratum_labels or [
            f"stratum_{i}" for i in range(len(stratum_probs))
        ]

    @property
    def num_strata(self) -> int:
        return len(self._probs)

    def allocate_samples(self, total_samples: int) -> np.ndarray:
        """Allocate samples to strata proportionally.

        Returns an integer array of per-stratum sample counts.
        """
        raw = self._probs * total_samples
        counts = np.floor(raw).astype(int)
        # Distribute remaining samples to largest fractional parts
        remainder = total_samples - counts.sum()
        fracs = raw - counts
        for _ in range(int(remainder)):
            idx = int(np.argmax(fracs))
            counts[idx] += 1
            fracs[idx] = 0.0
        return counts

    def combine_estimates(
        self,
        stratum_estimates: np.ndarray,
        stratum_variances: np.ndarray,
    ) -> Tuple[float, float]:
        """Combine per-stratum estimates into an overall estimate.

        Parameters:
            stratum_estimates: Point estimate for each stratum.
            stratum_variances: Variance estimate for each stratum.

        Returns:
            ``(overall_estimate, overall_variance)``
        """
        estimate = float(np.sum(self._probs * stratum_estimates))
        variance = float(np.sum(self._probs ** 2 * stratum_variances))
        return estimate, variance

    def optimal_allocation(
        self,
        stratum_variances: np.ndarray,
        total_samples: int,
    ) -> np.ndarray:
        """Neyman optimal allocation: allocate more samples to strata
        with higher variance.

        Returns integer array of per-stratum sample counts.
        """
        stds = np.sqrt(np.maximum(stratum_variances, 0.0))
        weighted = self._probs * stds
        total_weighted = weighted.sum()
        if total_weighted < 1e-15:
            return self.allocate_samples(total_samples)

        proportions = weighted / total_weighted
        raw = proportions * total_samples
        counts = np.maximum(np.floor(raw).astype(int), 1)

        remainder = total_samples - counts.sum()
        fracs = raw - counts
        for _ in range(max(0, int(remainder))):
            idx = int(np.argmax(fracs))
            counts[idx] += 1
            fracs[idx] = 0.0

        return counts


# ---------------------------------------------------------------------------
# Control variate
# ---------------------------------------------------------------------------

class ControlVariate:
    """Control variate methods using abstract interpretation bounds.

    Given a race indicator ``f(σ)`` and a control variate ``g(σ)``
    whose expectation ``E[g]`` is known (e.g. from abstract interpretation),
    the controlled estimate is:

        ``f̂ = (1/N)Σf(σᵢ) - c * [(1/N)Σg(σᵢ) - E[g]]``

    where ``c`` is the optimal control coefficient.
    """

    def __init__(
        self,
        control_fn: Callable[[np.ndarray], np.ndarray],
        control_expectation: float,
    ) -> None:
        """
        Parameters:
            control_fn: Maps an array of sample indices to control
                variate values.
            control_expectation: Known expectation ``E[g]``.
        """
        self._control_fn = control_fn
        self._E_g = control_expectation

    def apply(
        self,
        values: np.ndarray,
        control_values: np.ndarray,
        weights: Optional[ImportanceWeights] = None,
    ) -> Tuple[float, float]:
        """Apply the control variate correction.

        Parameters:
            values: Primary quantity ``f(σᵢ)``.
            control_values: Control variate ``g(σᵢ)``.
            weights: Optional importance weights.

        Returns:
            ``(corrected_estimate, variance_reduction_ratio)``
        """
        if weights is not None:
            w = weights.normalised_weights()
        else:
            n = len(values)
            w = np.ones(n) / n

        f_bar = float(np.sum(w * values))
        g_bar = float(np.sum(w * control_values))

        # Optimal control coefficient
        cov_fg = float(np.sum(w * (values - f_bar) * (control_values - g_bar)))
        var_g = float(np.sum(w * (control_values - g_bar) ** 2))

        if var_g < 1e-15:
            return f_bar, 1.0

        c_star = cov_fg / var_g

        corrected = f_bar - c_star * (g_bar - self._E_g)

        # Variance reduction ratio
        var_f = float(np.sum(w * (values - f_bar) ** 2))
        corr = cov_fg / (math.sqrt(max(var_f, 1e-15)) * math.sqrt(var_g))
        vr_ratio = 1.0 - corr ** 2

        return corrected, max(vr_ratio, 0.0)


# ---------------------------------------------------------------------------
# Bayesian estimator
# ---------------------------------------------------------------------------

class BayesianEstimator:
    """Bayesian estimation of race probability with prior from abstract
    interpretation.

    Models ``P(race) ~ Beta(α, β)`` and updates with observed
    importance-weighted race counts.
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        """
        Parameters:
            prior_alpha: Alpha parameter of the Beta prior.
            prior_beta: Beta parameter of the Beta prior.
        """
        self._alpha = prior_alpha
        self._beta = prior_beta

    @classmethod
    def from_abstract_bound(
        cls,
        abstract_upper_bound: float,
        confidence_equivalent_samples: int = 10,
    ) -> "BayesianEstimator":
        """Construct a prior from an abstract-interpretation upper bound.

        Centres the prior at ``abstract_upper_bound`` with the given
        equivalent sample strength.

        Parameters:
            abstract_upper_bound: Upper bound on P(race) from abstract interp.
            confidence_equivalent_samples: Pseudo-count strength of the prior.
        """
        p = min(max(abstract_upper_bound, 1e-6), 1.0 - 1e-6)
        n = confidence_equivalent_samples
        alpha = p * n
        beta = (1 - p) * n
        return cls(alpha, beta)

    @property
    def prior_mean(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    @property
    def posterior_mean(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    def update(
        self,
        race_indicators: np.ndarray,
        weights: Optional[ImportanceWeights] = None,
    ) -> None:
        """Update the posterior with observed (importance-weighted) data.

        Parameters:
            race_indicators: Binary array (1 = race observed).
            weights: Optional importance weights.
        """
        if weights is not None:
            w = weights.normalised_weights()
            ess = 1.0 / float(np.sum(w ** 2))
            weighted_successes = float(np.sum(w * race_indicators)) * ess
            weighted_failures = ess - weighted_successes
        else:
            weighted_successes = float(np.sum(race_indicators))
            weighted_failures = float(len(race_indicators) - np.sum(race_indicators))

        self._alpha += max(weighted_successes, 0.0)
        self._beta += max(weighted_failures, 0.0)

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Equal-tailed Bayesian credible interval.

        Returns:
            ``(lower, upper)``
        """
        tail = (1.0 - level) / 2.0
        lower = float(scipy_stats.beta.ppf(tail, self._alpha, self._beta))
        upper = float(scipy_stats.beta.ppf(1.0 - tail, self._alpha, self._beta))
        return lower, upper

    def probability_above(self, threshold: float) -> float:
        """Posterior probability that ``P(race) > threshold``."""
        return 1.0 - float(scipy_stats.beta.cdf(threshold, self._alpha, self._beta))


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

class ConvergenceDiagnostics:
    """Assess whether enough samples have been drawn.

    Provides several diagnostics:
    - Running mean stability (coefficient of variation of the running mean).
    - ESS-based criterion (ESS > min_ess).
    - Relative CI width criterion.
    """

    def __init__(
        self,
        min_ess: float = 100.0,
        max_cv: float = 0.1,
        max_relative_ci_width: float = 0.5,
    ) -> None:
        self._min_ess = min_ess
        self._max_cv = max_cv
        self._max_rel_width = max_relative_ci_width

    def is_converged(
        self,
        values: np.ndarray,
        weights: ImportanceWeights,
        confidence_level: float = 0.95,
    ) -> Tuple[bool, Dict[str, float]]:
        """Check convergence and return diagnostics.

        Parameters:
            values: Race indicators or quantity of interest.
            weights: Importance weights.
            confidence_level: CI confidence level.

        Returns:
            ``(converged, diagnostics_dict)``
        """
        w = weights.normalised_weights()
        ess = 1.0 / float(np.sum(w ** 2))
        estimate = float(np.sum(w * values))

        # Running mean stability
        n = len(values)
        if n >= 10:
            running_means = np.cumsum(w * values) / np.cumsum(w)
            recent = running_means[max(0, n - n // 5):]
            cv = float(np.std(recent) / (np.abs(np.mean(recent)) + 1e-15))
        else:
            cv = float("inf")

        # CI width
        var_est = float(np.sum(w ** 2 * (values - estimate) ** 2))
        se = math.sqrt(max(var_est, 0.0))
        z = scipy_stats.norm.ppf(0.5 + confidence_level / 2.0)
        ci_width = 2 * z * se
        rel_width = ci_width / (abs(estimate) + 1e-15)

        diagnostics = {
            "ess": ess,
            "coefficient_of_variation": cv,
            "relative_ci_width": rel_width,
            "estimate": estimate,
            "std_error": se,
        }

        converged = (
            ess >= self._min_ess
            and cv <= self._max_cv
            and rel_width <= self._max_rel_width
        )

        return converged, diagnostics

    @staticmethod
    def running_mean(values: np.ndarray, weights: ImportanceWeights) -> np.ndarray:
        """Compute the running (cumulative) importance-weighted mean."""
        w = weights.normalised_weights()
        return np.cumsum(w * values) / np.cumsum(w)

    @staticmethod
    def batch_means_test(
        values: np.ndarray,
        weights: ImportanceWeights,
        num_batches: int = 10,
    ) -> Tuple[float, float]:
        """Batch means test for autocorrelation.

        Splits samples into batches, computes per-batch means, and
        tests whether batch means are consistent (low variance relative
        to overall variance).

        Returns:
            ``(batch_variance, overall_variance)``
        """
        w = weights.normalised_weights()
        n = len(values)
        batch_size = max(1, n // num_batches)

        batch_means: List[float] = []
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bw = w[i:end]
            bv = values[i:end]
            if bw.sum() > 1e-15:
                batch_means.append(float(np.sum(bw * bv) / bw.sum()))

        overall = float(np.sum(w * values))
        batch_var = float(np.var(batch_means)) if len(batch_means) > 1 else 0.0
        overall_var = float(np.sum(w * (values - overall) ** 2))

        return batch_var, overall_var
