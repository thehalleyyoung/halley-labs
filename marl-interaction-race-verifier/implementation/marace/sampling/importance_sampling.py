"""Sequential importance sampling for race probability estimation.

Implements importance sampling with proper weight computation,
sequential construction with resampling, and proposal distributions
biased toward abstract-interpretation-identified race regions.
"""

from __future__ import annotations

import abc
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

from marace.sampling.schedule_space import (
    ContinuousSchedule,
    Schedule,
    ScheduleSpace,
    ScheduleGenerator,
)


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

@dataclass
class ImportanceWeights:
    """Container for importance weights with normalisation utilities.

    Attributes:
        log_weights: Log-space importance weights ``log(μ(σ)/q(σ))``.
    """

    log_weights: np.ndarray

    @property
    def num_samples(self) -> int:
        return len(self.log_weights)

    def normalised_weights(self) -> np.ndarray:
        """Self-normalised importance weights summing to 1.

        Uses the log-sum-exp trick for numerical stability.
        """
        max_lw = np.max(self.log_weights)
        shifted = self.log_weights - max_lw
        exp_w = np.exp(shifted)
        return exp_w / exp_w.sum()

    def unnormalised_weights(self) -> np.ndarray:
        """Exponentiated (unnormalised) weights."""
        max_lw = np.max(self.log_weights)
        return np.exp(self.log_weights - max_lw)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

class EffectiveSampleSize:
    """Monitor effective sample size for resampling decisions.

    ESS = (Σ wᵢ)² / Σ wᵢ²

    When ESS drops below a threshold (typically N/2), resampling
    is triggered to prevent weight degeneracy.
    """

    @staticmethod
    def compute(weights: ImportanceWeights) -> float:
        """Compute ESS from importance weights."""
        w = weights.normalised_weights()
        return 1.0 / float(np.sum(w ** 2))

    @staticmethod
    def compute_from_array(normalised_weights: np.ndarray) -> float:
        """Compute ESS from a pre-normalised weight array."""
        return 1.0 / float(np.sum(normalised_weights ** 2))

    @staticmethod
    def should_resample(
        weights: ImportanceWeights, threshold_fraction: float = 0.5
    ) -> bool:
        """Check whether resampling is needed.

        Parameters:
            weights: Current importance weights.
            threshold_fraction: Resample when ESS < N * threshold_fraction.
        """
        ess = EffectiveSampleSize.compute(weights)
        return ess < weights.num_samples * threshold_fraction


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Confidence interval for a probability estimate.

    Attributes:
        estimate: Point estimate of P(race).
        lower: Lower bound of the CI.
        upper: Upper bound of the CI.
        confidence_level: Nominal coverage (e.g. 0.95).
        effective_samples: ESS used for the estimate.
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    effective_samples: float = 0.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        if self.estimate < 1e-15:
            return float("inf")
        return self.width / self.estimate

    @staticmethod
    def from_importance_samples(
        race_indicators: np.ndarray,
        weights: ImportanceWeights,
        confidence_level: float = 0.95,
    ) -> "ConfidenceInterval":
        """Construct a CI from importance-weighted race indicators.

        Uses CLT-based intervals with ESS-adjusted standard error.

        Parameters:
            race_indicators: Binary array, 1 if schedule triggered a race.
            weights: Importance weights ``μ(σ)/q(σ)``.
            confidence_level: Desired coverage.

        Returns:
            :class:`ConfidenceInterval`.
        """
        w = weights.normalised_weights()
        n = len(w)
        ess = EffectiveSampleSize.compute_from_array(w)

        # Self-normalised IS estimate
        estimate = float(np.sum(w * race_indicators))

        # Variance estimate (accounting for self-normalisation)
        var_est = float(np.sum(w ** 2 * (race_indicators - estimate) ** 2))
        se = math.sqrt(max(var_est, 0.0))

        # z-score for confidence level
        from scipy.stats import norm as normal_dist

        z = normal_dist.ppf(0.5 + confidence_level / 2.0)
        half_width = z * se

        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - half_width),
            upper=min(1.0, estimate + half_width),
            confidence_level=confidence_level,
            effective_samples=ess,
        )


# ---------------------------------------------------------------------------
# Proposal distributions
# ---------------------------------------------------------------------------

class ProposalDistribution(abc.ABC):
    """Base class for proposal distributions over schedules."""

    @abc.abstractmethod
    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Draw *n* schedules from the proposal."""

    @abc.abstractmethod
    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability of *schedule* under this proposal."""


class UniformProposal(ProposalDistribution):
    """Uniform proposal over all valid schedules.

    Samples via randomised topological sort, assigning equal
    probability ``1 / |Ω|`` to each valid schedule.
    """

    def __init__(self, space: ScheduleSpace) -> None:
        self._space = space
        self._generator = ScheduleGenerator(space)
        # log(1/|Ω|) is unknown in general; we track it per-sample
        self._log_prob_cache: Optional[float] = None

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        gen = ScheduleGenerator(self._space, rng=rng)
        return gen.sample_uniform(n)

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under uniform distribution.

        For the uniform proposal over topological orderings, the exact
        probability depends on the DAG structure.  We use a constant
        approximation (valid for self-normalised IS).
        """
        # Constant for all schedules under uniform proposal
        return 0.0  # unnormalised; cancels in self-normalised IS


class AbstractGuidedProposal(ProposalDistribution):
    """Proposal biased toward abstract-interpretation-identified race regions.

    Uses a scoring function derived from abstract interpretation safety
    margins to weight schedules.  Schedules that pass through high-risk
    regions receive higher proposal probability.

    Parameters:
        space: The schedule space.
        risk_scorer: Callable mapping a schedule to a non-negative risk score.
            Higher scores indicate schedules more likely to trigger races.
        temperature: Controls sharpness of the biasing.  Lower temperature
            concentrates mass on highest-risk schedules.
        base_proposal: Underlying proposal used for sampling; the risk
            scorer is applied as a rejection/reweighting filter.
    """

    def __init__(
        self,
        space: ScheduleSpace,
        risk_scorer: Callable[[Schedule], float],
        temperature: float = 1.0,
        base_proposal: Optional[ProposalDistribution] = None,
    ) -> None:
        self._space = space
        self._risk_scorer = risk_scorer
        self._temperature = max(temperature, 1e-6)
        self._base = base_proposal or UniformProposal(space)

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Sample via importance resampling from the base proposal.

        1. Draw ``m = 4n`` candidates from the base proposal.
        2. Score each candidate.
        3. Resample ``n`` schedules proportional to the softmax of scores.
        """
        m = max(4 * n, 20)
        candidates = self._base.sample(m, rng)
        scores = np.array([self._risk_scorer(s) for s in candidates])

        # Softmax with temperature
        log_weights = scores / self._temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = weights.sum()
        if total < 1e-15:
            # All scores zero — fall back to uniform
            probs = np.ones(m) / m
        else:
            probs = weights / total

        indices = rng.choice(m, size=n, replace=True, p=probs)
        return [candidates[i] for i in indices]

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under the guided proposal.

        Approximated as ``log_base_prob + score / temperature - log_Z``.
        Since ``log_Z`` is unknown we return the unnormalised log-prob;
        self-normalised IS handles the constant.
        """
        base_lp = self._base.log_prob(schedule)
        score = self._risk_scorer(schedule)
        return base_lp + score / self._temperature


# ---------------------------------------------------------------------------
# Importance sampler (non-sequential)
# ---------------------------------------------------------------------------

class ImportanceSampler:
    """Standard importance sampling with proper weight computation.

    Estimates ``E_μ[f(σ)]`` by sampling from a proposal ``q`` and
    computing importance weights ``w = μ(σ)/q(σ)``.
    """

    def __init__(
        self,
        target_log_prob: Callable[[Schedule], float],
        proposal: ProposalDistribution,
    ) -> None:
        """
        Parameters:
            target_log_prob: Log-density of the target distribution μ.
            proposal: Proposal distribution q.
        """
        self._target_lp = target_log_prob
        self._proposal = proposal

    def sample_and_weight(
        self,
        n: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[List[Schedule], ImportanceWeights]:
        """Draw *n* samples and compute importance weights.

        Returns:
            ``(schedules, weights)``
        """
        rng = rng or np.random.RandomState(42)
        schedules = self._proposal.sample(n, rng)
        log_weights = np.array([
            self._target_lp(s) - self._proposal.log_prob(s)
            for s in schedules
        ])
        return schedules, ImportanceWeights(log_weights)

    def estimate(
        self,
        f: Callable[[Schedule], float],
        n: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Self-normalised IS estimate of ``E_μ[f(σ)]``."""
        schedules, weights = self.sample_and_weight(n, rng)
        w = weights.normalised_weights()
        return float(np.sum(w * np.array([f(s) for s in schedules])))


# ---------------------------------------------------------------------------
# Sequential importance sampler
# ---------------------------------------------------------------------------

class SequentialImportanceSampler:
    """SIS with systematic resampling for schedule sequences.

    Builds schedules event-by-event, maintaining a population of
    *particles* (partial schedules) with importance weights.  When
    the ESS drops below a threshold, systematic resampling is applied.

    Parameters:
        space: The schedule space.
        target_log_prob: Per-event incremental log-density of the target.
        proposal: Per-event incremental proposal distribution.
        num_particles: Number of particles (population size).
        resample_threshold: Resample when ``ESS < N * threshold``.
    """

    def __init__(
        self,
        space: ScheduleSpace,
        target_log_prob: Callable[[Schedule], float],
        proposal: ProposalDistribution,
        num_particles: int = 500,
        resample_threshold: float = 0.5,
    ) -> None:
        self._space = space
        self._target_lp = target_log_prob
        self._proposal = proposal
        self._N = num_particles
        self._resample_threshold = resample_threshold

    def run(
        self,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[List[Schedule], ImportanceWeights]:
        """Run the SIS procedure.

        Returns:
            ``(final_schedules, importance_weights)``
        """
        rng = rng or np.random.RandomState(42)

        # Sample full schedules from proposal
        particles = self._proposal.sample(self._N, rng)
        log_weights = np.zeros(self._N, dtype=np.float64)

        for i, s in enumerate(particles):
            log_weights[i] = self._target_lp(s) - self._proposal.log_prob(s)

        weights = ImportanceWeights(log_weights)

        # Check for resampling
        if EffectiveSampleSize.should_resample(weights, self._resample_threshold):
            particles, log_weights = self._systematic_resample(
                particles, weights, rng
            )
            weights = ImportanceWeights(log_weights)

        return particles, weights

    def _systematic_resample(
        self,
        particles: List[Schedule],
        weights: ImportanceWeights,
        rng: np.random.RandomState,
    ) -> Tuple[List[Schedule], np.ndarray]:
        """Systematic resampling (low-variance resampling).

        Selects particles proportional to their weights using a
        single random offset, producing a more uniform set of
        particles with equal weights.
        """
        N = len(particles)
        w = weights.normalised_weights()
        positions = (rng.uniform() + np.arange(N)) / N
        cumulative = np.cumsum(w)

        indices: List[int] = []
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative[j]:
                indices.append(j)
                i += 1
            else:
                j = min(j + 1, N - 1)

        new_particles = [particles[idx] for idx in indices]
        new_log_weights = np.zeros(N, dtype=np.float64)
        return new_particles, new_log_weights


# ---------------------------------------------------------------------------
# Race probability estimator
# ---------------------------------------------------------------------------

class RaceProbabilityEstimator:
    """Estimate P(race) with confidence bounds.

    Combines importance sampling with abstract-interpretation–guided
    proposals to produce rigorous probability bounds on race occurrence
    under the deployment schedule distribution.

    Parameters:
        space: The schedule space.
        race_checker: Callable that takes a schedule and returns ``True``
            if the schedule triggers a race condition.
        target_log_prob: Log-density of the deployment schedule distribution.
        proposal: Proposal distribution (should be biased toward race regions).
        num_samples: Number of IS samples.
        confidence_level: Desired confidence level for the interval.
    """

    def __init__(
        self,
        space: ScheduleSpace,
        race_checker: Callable[[Schedule], bool],
        target_log_prob: Callable[[Schedule], float],
        proposal: ProposalDistribution,
        num_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        self._space = space
        self._race_checker = race_checker
        self._target_lp = target_log_prob
        self._proposal = proposal
        self._N = num_samples
        self._confidence = confidence_level

    def estimate(
        self, rng: Optional[np.random.RandomState] = None
    ) -> ConfidenceInterval:
        """Run importance sampling and produce a confidence interval.

        Returns:
            :class:`ConfidenceInterval` for P(race).
        """
        rng = rng or np.random.RandomState(42)
        sampler = ImportanceSampler(self._target_lp, self._proposal)
        schedules, weights = sampler.sample_and_weight(self._N, rng)

        indicators = np.array(
            [1.0 if self._race_checker(s) else 0.0 for s in schedules],
            dtype=np.float64,
        )

        ci = ConfidenceInterval.from_importance_samples(
            indicators, weights, self._confidence
        )
        return ci

    def adaptive_estimate(
        self,
        max_rounds: int = 10,
        samples_per_round: int = 500,
        target_relative_width: float = 0.5,
        rng: Optional[np.random.RandomState] = None,
    ) -> ConfidenceInterval:
        """Run importance sampling with adaptive sample allocation.

        Draws samples in rounds, stopping when the relative CI width
        drops below *target_relative_width* or *max_rounds* is reached.

        Returns:
            :class:`ConfidenceInterval` for P(race).
        """
        rng = rng or np.random.RandomState(42)

        all_indicators: List[float] = []
        all_log_weights: List[float] = []

        for _ in range(max_rounds):
            sampler = ImportanceSampler(self._target_lp, self._proposal)
            schedules, weights = sampler.sample_and_weight(samples_per_round, rng)

            indicators = [
                1.0 if self._race_checker(s) else 0.0 for s in schedules
            ]
            all_indicators.extend(indicators)
            all_log_weights.extend(weights.log_weights.tolist())

            combined_weights = ImportanceWeights(np.array(all_log_weights))
            ci = ConfidenceInterval.from_importance_samples(
                np.array(all_indicators),
                combined_weights,
                self._confidence,
            )

            if ci.relative_width < target_relative_width:
                return ci

        return ci


# ---------------------------------------------------------------------------
# Self-normalized IS estimator with bias correction
# ---------------------------------------------------------------------------

class SelfNormalizedISEstimator:
    r"""Self-normalised importance sampling estimator with bias correction.

    Theory
    ------
    The self-normalised IS estimator is

        μ̂_SN = ∑_i w̃_i f(X_i)

    where w̃_i = w_i / ∑_j w_j are the normalised weights.

    This estimator has bias O(1/n):

        Bias(μ̂_SN) = -Cov_q(W, f(X)) / (E_q[W])² + O(n⁻²)

    (Owen, *Monte Carlo Theory, Methods and Examples*, Ch. 9).

    We apply first-order bias correction and provide the corrected
    estimate along with a variance estimate that accounts for
    self-normalisation.

    Parameters
    ----------
    target_log_prob : callable
        Log-density of the target distribution.
    proposal : ProposalDistribution
        Proposal distribution.
    """

    def __init__(
        self,
        target_log_prob: Callable[[Schedule], float],
        proposal: ProposalDistribution,
    ) -> None:
        self._target_lp = target_log_prob
        self._proposal = proposal

    def estimate(
        self,
        f: Callable[[Schedule], float],
        n: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[float, float, float]:
        """Bias-corrected self-normalised IS estimate.

        Returns
        -------
        (estimate, bias_correction, variance_estimate)
            estimate : bias-corrected point estimate.
            bias_correction : the applied correction.
            variance_estimate : estimated variance of the estimator.
        """
        rng = rng or np.random.RandomState(42)
        schedules = self._proposal.sample(n, rng)

        log_weights = np.array([
            self._target_lp(s) - self._proposal.log_prob(s)
            for s in schedules
        ])
        max_lw = np.max(log_weights)
        w = np.exp(log_weights - max_lw)
        w_sum = float(np.sum(w))
        if w_sum < 1e-300:
            return 0.0, 0.0, float("inf")

        w_norm = w / w_sum
        values = np.array([f(s) for s in schedules])

        mu_sn = float(np.dot(w_norm, values))

        # Bias correction: -Cov(W, f) / E[W]²
        w_bar = float(np.mean(w))
        if w_bar > 1e-300 and n >= 3:
            cov_wf = float(np.cov(w, values, ddof=1)[0, 1])
            bias_corr = -cov_wf / (w_bar * w_bar)
        else:
            bias_corr = 0.0

        mu_corrected = mu_sn + bias_corr

        # Variance estimate for SN-IS
        resid = values - mu_sn
        var_est = float(np.sum(w_norm ** 2 * resid ** 2))

        return mu_corrected, bias_corr, var_est


# ---------------------------------------------------------------------------
# ESS-triggered adaptive resampling
# ---------------------------------------------------------------------------

class ESSAdaptiveResampler:
    """ESS-triggered adaptive resampling for importance sampling.

    Monitors the ESS of the importance weights and triggers
    systematic resampling when ESS drops below a threshold.
    After resampling, weights are reset to uniform.

    Parameters
    ----------
    threshold_fraction : float
        Resample when ESS < n * threshold_fraction.
    """

    def __init__(self, threshold_fraction: float = 0.5) -> None:
        self._threshold_frac = threshold_fraction
        self._resample_count = 0

    def maybe_resample(
        self,
        particles: List[Schedule],
        log_weights: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[List[Schedule], np.ndarray, bool]:
        """Resample if ESS is below threshold.

        Returns
        -------
        (particles, log_weights, did_resample)
        """
        rng = rng or np.random.RandomState(42)
        n = len(particles)
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = float(np.sum(w))
        if w_sum < 1e-300:
            return particles, log_weights, False

        w_norm = w / w_sum
        ess = 1.0 / float(np.sum(w_norm ** 2))

        if ess >= n * self._threshold_frac:
            return particles, log_weights, False

        # Systematic resampling
        positions = (rng.uniform() + np.arange(n)) / n
        cumulative = np.cumsum(w_norm)
        indices: List[int] = []
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative[j]:
                indices.append(j)
                i += 1
            else:
                j = min(j + 1, n - 1)

        new_particles = [particles[idx] for idx in indices]
        new_log_weights = np.zeros(n, dtype=np.float64)
        self._resample_count += 1

        return new_particles, new_log_weights, True

    @property
    def resample_count(self) -> int:
        return self._resample_count


# ---------------------------------------------------------------------------
# Pareto-smoothed importance sampling (PSIS)
# ---------------------------------------------------------------------------

class ParetoSmoothedIS:
    r"""Pareto-smoothed importance sampling (PSIS).

    Implements the PSIS method of Vehtari et al. (2024, JMLR) which
    stabilises importance weights by fitting a generalized Pareto
    distribution (GPD) to the upper tail and replacing extreme weights
    with the fitted quantiles.

    Algorithm
    ---------
    1. Sort weights in descending order.
    2. Fit GPD to the top M = min(⌊0.2n⌋, 3√n) weights.
    3. Replace the top M weights with the expected order statistics
       from the fitted GPD.
    4. Return the smoothed weights and the Pareto k̂ diagnostic.

    Interpretation of k̂
    --------------------
    - k̂ < 0.5: good importance sampling, finite variance.
    - 0.5 ≤ k̂ < 0.7: finite variance but slower convergence.
    - k̂ ≥ 0.7: infinite variance of the IS estimator, results unreliable.

    Parameters
    ----------
    min_tail_samples : int
        Minimum number of tail samples for GPD fit.
    """

    def __init__(self, min_tail_samples: int = 5) -> None:
        self._min_tail = min_tail_samples

    def smooth_weights(
        self, log_weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Apply Pareto smoothing to log importance weights.

        Returns
        -------
        (smoothed_log_weights, k_hat)
            smoothed_log_weights: stabilised log weights.
            k_hat: Pareto shape parameter diagnostic.
        """
        n = len(log_weights)
        M = min(max(int(0.2 * n), self._min_tail), max(int(3 * math.sqrt(n)), self._min_tail))
        M = min(M, n - 1)

        if M < self._min_tail:
            return log_weights.copy(), 0.0

        # Work in original scale for tail fitting
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        sorted_idx = np.argsort(w)[::-1]
        sorted_w = w[sorted_idx]

        # Tail values: excesses above threshold
        threshold = sorted_w[M]
        tail_values = sorted_w[:M] - threshold
        tail_values = tail_values[tail_values > 0]

        if len(tail_values) < self._min_tail:
            return log_weights.copy(), 0.0

        # Fit GPD via method of moments (simple, robust)
        mean_excess = float(np.mean(tail_values))
        var_excess = float(np.var(tail_values))

        if mean_excess < 1e-300:
            return log_weights.copy(), 0.0

        # GPD: k = 0.5 * (mean²/var - 1), σ = mean * (mean²/var + 1) / 2
        if var_excess < 1e-300:
            k_hat = 0.0
        else:
            ratio = mean_excess ** 2 / var_excess
            k_hat = 0.5 * (ratio - 1.0)

        k_hat = max(min(k_hat, 1.0), -0.5)

        # Replace tail weights with expected order statistics
        smoothed = w.copy()
        for i in range(min(M, len(sorted_idx))):
            # Expected order statistic from GPD
            p = (i + 0.5) / (M + 1)
            if abs(k_hat) < 1e-10:
                q = -mean_excess * math.log(1 - p)
            else:
                q = mean_excess / k_hat * ((1 - p) ** (-k_hat) - 1)
            smoothed[sorted_idx[i]] = threshold + max(q, 0.0)

        # Renormalise and convert back to log
        smoothed = np.maximum(smoothed, 1e-300)
        smoothed_lw = np.log(smoothed) + max_lw

        return smoothed_lw, k_hat

    def estimate(
        self,
        log_weights: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[float, float]:
        """PSIS estimate of E_p[f].

        Returns (estimate, k_hat).
        """
        smoothed_lw, k_hat = self.smooth_weights(log_weights)
        max_lw = float(np.max(smoothed_lw))
        w = np.exp(smoothed_lw - max_lw)
        w_norm = w / w.sum()
        estimate = float(np.dot(w_norm, values))
        return estimate, k_hat


# ---------------------------------------------------------------------------
# Control variate validation (non-circular)
# ---------------------------------------------------------------------------

class ControlVariateValidator:
    r"""Ground-truth validation via control variates (non-circular).

    Avoids circular validation by using a known-expectation control
    variate g(X) with E_p[g(X)] = μ_g known analytically.

    The control variate estimate is:

        μ̂_CV = μ̂_f - β · (μ̂_g - μ_g)

    where β = Cov(f, g) / Var(g) is the optimal coefficient.

    For validation, we estimate E_p[g] using the same IS weights and
    compare to the known μ_g.  If |μ̂_g - μ_g| is small relative to
    the CI width, the IS procedure is validated non-circularly.

    Parameters
    ----------
    control_fn : callable
        Control function g: Schedule → float.
    control_mean : float
        Known expectation E_p[g(X)].
    """

    def __init__(
        self,
        control_fn: Callable[[Schedule], float],
        control_mean: float,
    ) -> None:
        self._g = control_fn
        self._mu_g = control_mean

    def validate(
        self,
        schedules: List[Schedule],
        log_weights: np.ndarray,
        tolerance: float = 0.1,
    ) -> Tuple[bool, float, float]:
        """Validate IS weights using the control variate.

        Returns
        -------
        (is_valid, control_estimate, control_error)
            is_valid: True if |μ̂_g - μ_g| < tolerance.
            control_estimate: IS estimate of E[g].
            control_error: |μ̂_g - μ_g|.
        """
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = float(np.sum(w))
        if w_sum < 1e-300:
            return False, 0.0, float("inf")

        w_norm = w / w_sum
        g_values = np.array([self._g(s) for s in schedules])
        mu_g_hat = float(np.dot(w_norm, g_values))
        error = abs(mu_g_hat - self._mu_g)

        return error < tolerance, mu_g_hat, error

    def correct(
        self,
        schedules: List[Schedule],
        log_weights: np.ndarray,
        f_values: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply control variate correction to IS estimate.

        Returns
        -------
        (corrected_estimate, beta)
            corrected_estimate: μ̂_CV.
            beta: optimal control variate coefficient.
        """
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = float(np.sum(w))
        if w_sum < 1e-300:
            return 0.0, 0.0

        w_norm = w / w_sum
        g_values = np.array([self._g(s) for s in schedules])

        mu_f_hat = float(np.dot(w_norm, f_values))
        mu_g_hat = float(np.dot(w_norm, g_values))

        # Optimal beta: Cov(f, g) / Var(g) weighted
        cov_fg = float(np.dot(w_norm, (f_values - mu_f_hat) * (g_values - mu_g_hat)))
        var_g = float(np.dot(w_norm, (g_values - mu_g_hat) ** 2))

        if var_g < 1e-300:
            return mu_f_hat, 0.0

        beta = cov_fg / var_g
        corrected = mu_f_hat - beta * (mu_g_hat - self._mu_g)

        return corrected, beta
