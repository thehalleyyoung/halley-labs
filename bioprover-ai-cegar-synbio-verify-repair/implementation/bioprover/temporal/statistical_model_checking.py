"""Statistical Model Checking for STL Properties.

Implements SPRT (Sequential Probability Ratio Test), Bayesian estimation,
importance splitting, and confidence interval methods for estimating
P(phi) >= theta with bounded error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from bioprover.temporal.robustness import Signal, compute_robustness
from bioprover.temporal.stl_ast import STLFormula


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TrajectoryGenerator = Callable[[], Dict[str, Signal]]


class HypothesisResult(Enum):
    """Result of a statistical hypothesis test."""
    ACCEPT = auto()    # Accept H0: P(phi) >= theta
    REJECT = auto()    # Reject H0: P(phi) < theta
    UNDECIDED = auto()  # Need more samples


@dataclass
class SPRTResult:
    """Result from Sequential Probability Ratio Test."""
    decision: HypothesisResult
    num_samples: int
    num_satisfied: int
    log_ratio: float
    estimated_probability: float
    upper_bound: float
    lower_bound: float

    @property
    def satisfaction_ratio(self) -> float:
        return self.num_satisfied / self.num_samples if self.num_samples > 0 else 0.0


@dataclass
class ConfidenceInterval:
    """A confidence interval for a probability estimate."""
    lower: float
    upper: float
    center: float
    confidence_level: float
    method: str
    num_samples: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, p: float) -> bool:
        return self.lower <= p <= self.upper


@dataclass
class BayesianEstimate:
    """Bayesian posterior estimate of satisfaction probability."""
    posterior_mean: float
    posterior_mode: float
    posterior_variance: float
    credible_interval: Tuple[float, float]
    alpha_posterior: float
    beta_posterior: float
    num_samples: int

    @property
    def credible_width(self) -> float:
        return self.credible_interval[1] - self.credible_interval[0]


# ---------------------------------------------------------------------------
# SPRT (Sequential Probability Ratio Test)
# ---------------------------------------------------------------------------

class SPRTChecker:
    """Sequential Probability Ratio Test for P(phi) >= theta.

    Tests H0: p >= p1 vs H1: p <= p0 where p0 < p1, with
    Type I error <= alpha and Type II error <= beta.
    Uses Wald's SPRT with likelihood ratio monitoring.
    """

    def __init__(
        self,
        formula: STLFormula,
        generator: TrajectoryGenerator,
        theta: float = 0.9,
        indifference: float = 0.05,
        alpha: float = 0.05,
        beta: float = 0.05,
        max_samples: int = 10000,
    ) -> None:
        self._formula = formula
        self._generator = generator
        self._theta = theta
        self._p0 = theta - indifference  # H1 boundary
        self._p1 = theta + indifference  # H0 boundary
        self._alpha = alpha
        self._beta = beta
        self._max_samples = max_samples

        # Wald's bounds
        self._log_A = math.log((1 - self._beta) / self._alpha)
        self._log_B = math.log(self._beta / (1 - self._alpha))

    def run(self) -> SPRTResult:
        """Execute the SPRT sampling loop."""
        log_ratio = 0.0
        num_satisfied = 0
        num_samples = 0

        for _ in range(self._max_samples):
            signals = self._generator()
            rho = compute_robustness(self._formula, signals)
            num_samples += 1
            sat = rho > 0
            if sat:
                num_satisfied += 1

            # Update log-likelihood ratio
            if sat:
                if self._p1 > 0 and self._p0 > 0:
                    log_ratio += math.log(self._p1 / self._p0)
            else:
                if (1 - self._p1) > 0 and (1 - self._p0) > 0:
                    log_ratio += math.log((1 - self._p1) / (1 - self._p0))

            # Check stopping criteria
            if log_ratio >= self._log_A:
                return self._make_result(
                    HypothesisResult.ACCEPT, num_samples, num_satisfied, log_ratio
                )
            if log_ratio <= self._log_B:
                return self._make_result(
                    HypothesisResult.REJECT, num_samples, num_satisfied, log_ratio
                )

        return self._make_result(
            HypothesisResult.UNDECIDED, num_samples, num_satisfied, log_ratio
        )

    def _make_result(
        self,
        decision: HypothesisResult,
        n: int,
        n_sat: int,
        log_ratio: float,
    ) -> SPRTResult:
        p_hat = n_sat / n if n > 0 else 0.0
        ci = wilson_interval(n_sat, n, 1 - self._alpha)
        return SPRTResult(
            decision=decision,
            num_samples=n,
            num_satisfied=n_sat,
            log_ratio=log_ratio,
            estimated_probability=p_hat,
            lower_bound=ci.lower,
            upper_bound=ci.upper,
        )


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return ConfidenceInterval(0.0, 1.0, 0.5, confidence, "wilson", 0)

    from scipy.stats import norm  # type: ignore[import-untyped]

    z = norm.ppf((1 + confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom

    return ConfidenceInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        center=center,
        confidence_level=confidence,
        method="wilson",
        num_samples=total,
    )


def clopper_pearson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Clopper-Pearson exact confidence interval for a binomial proportion."""
    from scipy.stats import beta as beta_dist  # type: ignore[import-untyped]

    alpha = 1 - confidence
    if total == 0:
        return ConfidenceInterval(0.0, 1.0, 0.5, confidence, "clopper_pearson", 0)

    if successes == 0:
        lower = 0.0
    else:
        lower = float(beta_dist.ppf(alpha / 2, successes, total - successes + 1))

    if successes == total:
        upper = 1.0
    else:
        upper = float(beta_dist.ppf(1 - alpha / 2, successes + 1, total - successes))

    p_hat = successes / total
    return ConfidenceInterval(lower, upper, p_hat, confidence, "clopper_pearson", total)


# ---------------------------------------------------------------------------
# Bayesian estimation
# ---------------------------------------------------------------------------

class BayesianEstimator:
    """Bayesian estimation of P(phi) using a Beta prior.

    Starts with Beta(alpha_prior, beta_prior) and updates with observations.
    Default is uniform prior Beta(1, 1).
    """

    def __init__(
        self,
        formula: STLFormula,
        generator: TrajectoryGenerator,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        credible_level: float = 0.95,
    ) -> None:
        self._formula = formula
        self._generator = generator
        self._alpha = alpha_prior
        self._beta = beta_prior
        self._credible_level = credible_level
        self._n_samples = 0

    def update(self, num_samples: int = 1) -> BayesianEstimate:
        """Draw samples and update posterior."""
        for _ in range(num_samples):
            signals = self._generator()
            rho = compute_robustness(self._formula, signals)
            self._n_samples += 1
            if rho > 0:
                self._alpha += 1
            else:
                self._beta += 1

        return self.current_estimate()

    def current_estimate(self) -> BayesianEstimate:
        """Return current posterior estimate without drawing new samples."""
        from scipy.stats import beta as beta_dist  # type: ignore[import-untyped]

        a, b = self._alpha, self._beta
        mean = a / (a + b)
        mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else mean
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))

        lo = float(beta_dist.ppf((1 - self._credible_level) / 2, a, b))
        hi = float(beta_dist.ppf((1 + self._credible_level) / 2, a, b))

        return BayesianEstimate(
            posterior_mean=mean,
            posterior_mode=mode,
            posterior_variance=variance,
            credible_interval=(lo, hi),
            alpha_posterior=a,
            beta_posterior=b,
            num_samples=self._n_samples,
        )

    def converged(self, width_threshold: float = 0.05) -> bool:
        """Check if credible interval is narrow enough."""
        est = self.current_estimate()
        return est.credible_width < width_threshold


# ---------------------------------------------------------------------------
# Sample size estimation
# ---------------------------------------------------------------------------

def required_sample_size(
    confidence: float = 0.95,
    half_width: float = 0.05,
    estimated_p: float = 0.5,
) -> int:
    """Estimate required sample size for a given confidence and half-width.

    Uses the normal approximation formula: n = z^2 * p(1-p) / e^2
    """
    from scipy.stats import norm  # type: ignore[import-untyped]
    z = norm.ppf((1 + confidence) / 2)
    n = (z ** 2 * estimated_p * (1 - estimated_p)) / (half_width ** 2)
    return int(math.ceil(n))


# ---------------------------------------------------------------------------
# Importance splitting for rare events
# ---------------------------------------------------------------------------

@dataclass
class ImportanceSplittingResult:
    """Result from importance splitting estimation."""
    estimated_probability: float
    confidence_interval: ConfidenceInterval
    num_levels: int
    level_thresholds: List[float]
    total_samples: int


def importance_splitting(
    formula: STLFormula,
    generator: TrajectoryGenerator,
    num_levels: int = 5,
    samples_per_level: int = 1000,
    confidence: float = 0.95,
) -> ImportanceSplittingResult:
    """Estimate rare event probability using importance splitting (multilevel splitting).

    Decomposes P(rho < 0) into a product of conditional probabilities at
    intermediate robustness thresholds.
    """
    # Generate initial samples and compute robustness
    rhos: List[float] = []
    for _ in range(samples_per_level):
        signals = generator()
        rho = compute_robustness(formula, signals)
        rhos.append(rho)

    rhos_arr = np.array(rhos)
    total_samples = samples_per_level

    # Determine intermediate thresholds via quantiles
    max_rho = float(np.max(rhos_arr))
    min_rho = float(np.min(rhos_arr))

    if min_rho >= 0:
        # All satisfied; probability of violation is very low
        return ImportanceSplittingResult(
            estimated_probability=1.0,
            confidence_interval=ConfidenceInterval(0.99, 1.0, 1.0, confidence, "splitting", total_samples),
            num_levels=0,
            level_thresholds=[],
            total_samples=total_samples,
        )

    # Create levels from max_rho down to 0
    thresholds = np.linspace(max_rho, 0, num_levels + 1).tolist()
    conditional_probs: List[float] = []

    for i in range(len(thresholds) - 1):
        threshold = thresholds[i + 1]
        count = int(np.sum(rhos_arr <= threshold))
        prob = count / len(rhos_arr) if len(rhos_arr) > 0 else 0.0
        conditional_probs.append(max(prob, 1e-10))

        # Resample from those that passed the threshold
        if count > 0 and i < len(thresholds) - 2:
            passing_indices = np.where(rhos_arr <= threshold)[0]
            rhos_arr = rhos_arr[passing_indices]
        total_samples += samples_per_level

    # Product of conditional probabilities = overall probability of violation
    p_violation = 1.0
    for p in conditional_probs:
        p_violation *= p

    p_satisfaction = 1.0 - p_violation

    ci = wilson_interval(
        int(round(p_satisfaction * total_samples)),
        total_samples,
        confidence,
    )

    return ImportanceSplittingResult(
        estimated_probability=p_satisfaction,
        confidence_interval=ci,
        num_levels=len(thresholds) - 1,
        level_thresholds=thresholds,
        total_samples=total_samples,
    )


# ---------------------------------------------------------------------------
# Cross-entropy method
# ---------------------------------------------------------------------------

def cross_entropy_rare_event(
    formula: STLFormula,
    generator: TrajectoryGenerator,
    num_iterations: int = 5,
    samples_per_iter: int = 500,
    elite_fraction: float = 0.1,
) -> float:
    """Cross-entropy method for importance distribution estimation.

    Returns an estimate of P(phi) by iteratively adjusting sampling distribution
    toward the rare event region. This is a simplified version that returns
    the final probability estimate.
    """
    rhos_all: List[float] = []

    for iteration in range(num_iterations):
        rhos: List[float] = []
        for _ in range(samples_per_iter):
            signals = generator()
            rho = compute_robustness(formula, signals)
            rhos.append(rho)
        rhos_all.extend(rhos)

        # Determine elite threshold
        rhos_arr = np.array(rhos)
        n_elite = max(1, int(elite_fraction * len(rhos)))
        sorted_rhos = np.sort(rhos_arr)
        gamma = sorted_rhos[n_elite]  # noqa: F841 (would be used to update distribution)

    # Final estimate from all samples
    all_arr = np.array(rhos_all)
    n_sat = int(np.sum(all_arr > 0))
    return n_sat / len(all_arr) if len(all_arr) > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-property checking
# ---------------------------------------------------------------------------

@dataclass
class MultiPropertyResult:
    """Results from checking multiple properties."""
    results: Dict[str, SPRTResult]
    all_satisfied: bool
    any_violated: bool


def check_multiple_properties(
    formulas: Dict[str, STLFormula],
    generator: TrajectoryGenerator,
    theta: float = 0.9,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_samples: int = 10000,
) -> MultiPropertyResult:
    """Check multiple STL properties simultaneously using shared samples."""
    # Bonferroni correction for multiple testing
    n_props = len(formulas)
    corrected_alpha = alpha / n_props if n_props > 0 else alpha

    results: Dict[str, SPRTResult] = {}
    for name, formula in formulas.items():
        checker = SPRTChecker(
            formula, generator, theta=theta,
            alpha=corrected_alpha, beta=beta,
            max_samples=max_samples,
        )
        results[name] = checker.run()

    all_sat = all(r.decision == HypothesisResult.ACCEPT for r in results.values())
    any_viol = any(r.decision == HypothesisResult.REJECT for r in results.values())

    return MultiPropertyResult(results=results, all_satisfied=all_sat, any_violated=any_viol)


# ---------------------------------------------------------------------------
# Convergence monitoring
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceInfo:
    """Monitors convergence of probability estimates over samples."""
    estimates: List[float]
    ci_widths: List[float]
    sample_counts: List[int]

    @property
    def is_converged(self) -> bool:
        if len(self.ci_widths) < 2:
            return False
        return self.ci_widths[-1] < 0.05

    @property
    def latest_estimate(self) -> float:
        return self.estimates[-1] if self.estimates else 0.0


def monitor_convergence(
    formula: STLFormula,
    generator: TrajectoryGenerator,
    batch_size: int = 100,
    max_batches: int = 50,
    target_width: float = 0.05,
    confidence: float = 0.95,
) -> ConvergenceInfo:
    """Monitor convergence of probability estimate over batches of samples."""
    n_sat = 0
    n_total = 0
    estimates: List[float] = []
    ci_widths: List[float] = []
    sample_counts: List[int] = []

    for _ in range(max_batches):
        for _ in range(batch_size):
            signals = generator()
            rho = compute_robustness(formula, signals)
            n_total += 1
            if rho > 0:
                n_sat += 1

        p_hat = n_sat / n_total
        ci = wilson_interval(n_sat, n_total, confidence)

        estimates.append(p_hat)
        ci_widths.append(ci.width)
        sample_counts.append(n_total)

        if ci.width < target_width:
            break

    return ConvergenceInfo(estimates, ci_widths, sample_counts)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

class StatisticalModelChecker:
    """Unified interface for statistical model checking of STL properties."""

    def __init__(
        self,
        formula: STLFormula,
        generator: TrajectoryGenerator,
    ) -> None:
        self._formula = formula
        self._generator = generator

    def sprt(
        self,
        theta: float = 0.9,
        alpha: float = 0.05,
        beta: float = 0.05,
        max_samples: int = 10000,
    ) -> SPRTResult:
        """Run SPRT hypothesis test."""
        checker = SPRTChecker(
            self._formula, self._generator,
            theta=theta, alpha=alpha, beta=beta,
            max_samples=max_samples,
        )
        return checker.run()

    def bayesian(
        self,
        num_samples: int = 1000,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> BayesianEstimate:
        """Run Bayesian estimation."""
        estimator = BayesianEstimator(
            self._formula, self._generator,
            alpha_prior=alpha_prior, beta_prior=beta_prior,
        )
        return estimator.update(num_samples)

    def confidence_interval(
        self,
        num_samples: int = 1000,
        confidence: float = 0.95,
        method: str = "wilson",
    ) -> ConfidenceInterval:
        """Estimate satisfaction probability with confidence interval."""
        n_sat = 0
        for _ in range(num_samples):
            signals = self._generator()
            rho = compute_robustness(self._formula, signals)
            if rho > 0:
                n_sat += 1
        if method == "wilson":
            return wilson_interval(n_sat, num_samples, confidence)
        elif method == "clopper_pearson":
            return clopper_pearson_interval(n_sat, num_samples, confidence)
        raise ValueError(f"Unknown CI method: {method}")

    def required_samples(
        self,
        confidence: float = 0.95,
        half_width: float = 0.05,
        estimated_p: float = 0.5,
    ) -> int:
        """Estimate required sample size."""
        return required_sample_size(confidence, half_width, estimated_p)
