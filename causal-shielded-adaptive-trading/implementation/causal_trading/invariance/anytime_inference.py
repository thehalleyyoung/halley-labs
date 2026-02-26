"""
Anytime-valid inference for sequential causal invariance testing.

Implements confidence sequences, mixture martingales, sub-Gaussian and
sub-exponential e-processes, and sequential tests with optional stopping.
All constructions provide time-uniform validity.

References:
    - Howard et al. (2021). Time-uniform, nonparametric, nonasymptotic
      confidence sequences.
    - Ramdas et al. (2023). Game-theoretic statistics and safe anytime-valid
      inference.
    - de la Peña et al. (2009). Self-normalized processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, special, stats

from .e_values import ConfidenceSequence, WealthProcess


class BoundType(Enum):
    """Type of concentration bound for confidence sequences."""
    HOEFFDING = "hoeffding"
    EMPIRICAL_BERNSTEIN = "empirical_bernstein"
    POLYNOMIAL = "polynomial"
    STITCHED = "stitched"


@dataclass
class SequentialTestResult:
    """Result from a sequential test.

    Attributes:
        rejected: Whether the null was rejected.
        stopped: Whether the test stopped early.
        stopping_time: Time at which the test stopped (None if not stopped).
        e_value: Final e-value.
        p_value: Anytime-valid p-value.
        confidence_sequence: Optional confidence sequence.
        n_observations: Total observations processed.
    """
    rejected: bool
    stopped: bool
    stopping_time: Optional[int]
    e_value: float
    p_value: float
    confidence_sequence: Optional[ConfidenceSequence] = None
    n_observations: int = 0


class MixtureMartingale:
    """Mixture martingale for anytime-valid inference.

    Constructs a test martingale by mixing over a family of likelihood ratios
    parameterized by a mixing distribution. The resulting process is a
    non-negative martingale under the null for any stopping time.

    Args:
        mixing_density: Function theta -> weight for continuous mixing.
        grid_points: Discrete grid for numerical integration.
        n_grid: Number of grid points if grid_points not provided.
        grid_range: Range for the grid (min, max).
    """

    def __init__(
        self,
        mixing_density: Optional[Callable[[float], float]] = None,
        grid_points: Optional[NDArray[np.float64]] = None,
        n_grid: int = 100,
        grid_range: Tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        if grid_points is not None:
            self.grid = np.asarray(grid_points, dtype=np.float64)
        else:
            self.grid = np.linspace(grid_range[0], grid_range[1], n_grid)

        if mixing_density is not None:
            self.weights = np.array([mixing_density(th) for th in self.grid])
        else:
            # Default: standard normal prior
            self.weights = stats.norm.pdf(self.grid)

        self.weights = self.weights / (np.sum(self.weights) + 1e-300)

        # State
        self._log_likelihoods = np.zeros(len(self.grid))
        self._t = 0
        self._history: List[float] = [1.0]
        self._wealth = WealthProcess()

    @property
    def value(self) -> float:
        """Current mixture martingale value."""
        shifted = self._log_likelihoods - np.max(self._log_likelihoods)
        weighted = np.exp(np.clip(shifted, -700, 0)) * self.weights
        return float(np.exp(np.max(self._log_likelihoods)) * np.sum(weighted))

    @property
    def log_value(self) -> float:
        """Current log mixture martingale value."""
        v = self.value
        return float(np.log(max(v, 1e-300)))

    def update(
        self,
        x_t: float,
        null_log_likelihood: float,
        alt_log_likelihood_fn: Callable[[float, float], float],
    ) -> float:
        """Update the mixture martingale with a new observation.

        Args:
            x_t: New observation.
            null_log_likelihood: Log-likelihood under the null.
            alt_log_likelihood_fn: Function (x, theta) -> log p_theta(x).

        Returns:
            Updated martingale value.
        """
        for i, theta in enumerate(self.grid):
            log_lr = alt_log_likelihood_fn(x_t, theta) - null_log_likelihood
            self._log_likelihoods[i] += log_lr

        self._t += 1
        current = self.value
        self._history.append(current)
        self._wealth.update(current / (self._history[-2] if len(self._history) > 2 else 1.0))

        return current

    def update_gaussian(
        self,
        x_t: float,
        null_mean: float = 0.0,
        variance: float = 1.0,
    ) -> float:
        """Specialized update for testing Gaussian mean shift.

        Tests H0: mu = null_mean against H1: mu = theta for each grid point.

        Args:
            x_t: New observation.
            null_mean: Mean under the null.
            variance: Known or estimated variance.

        Returns:
            Updated martingale value.
        """
        sigma = np.sqrt(variance)
        null_ll = stats.norm.logpdf(x_t, loc=null_mean, scale=sigma)

        def alt_ll(x: float, theta: float) -> float:
            return float(stats.norm.logpdf(x, loc=theta, scale=sigma))

        return self.update(x_t, float(null_ll), alt_ll)

    def reject(self, alpha: float = 0.05) -> bool:
        """Test rejection at level alpha."""
        return self.value >= 1.0 / alpha

    def p_value(self) -> float:
        """Anytime-valid p-value."""
        v = self.value
        if v <= 0:
            return 1.0
        return min(1.0, 1.0 / v)

    def get_posterior_weights(self) -> NDArray[np.float64]:
        """Compute posterior weights over the grid (Bayesian interpretation)."""
        log_post = np.log(self.weights + 1e-300) + self._log_likelihoods
        log_post -= special.logsumexp(log_post)
        return np.exp(log_post)

    def posterior_mean(self) -> float:
        """Posterior mean of the parameter."""
        post = self.get_posterior_weights()
        return float(np.dot(post, self.grid))

    def posterior_credible_interval(
        self,
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute posterior credible interval."""
        post = self.get_posterior_weights()
        cum = np.cumsum(post)
        alpha_half = (1.0 - level) / 2.0
        lower_idx = np.searchsorted(cum, alpha_half)
        upper_idx = np.searchsorted(cum, 1.0 - alpha_half)
        lower_idx = np.clip(lower_idx, 0, len(self.grid) - 1)
        upper_idx = np.clip(upper_idx, 0, len(self.grid) - 1)
        return float(self.grid[lower_idx]), float(self.grid[upper_idx])

    def reset(self) -> None:
        self._log_likelihoods = np.zeros(len(self.grid))
        self._t = 0
        self._history = [1.0]
        self._wealth = WealthProcess()


class SubGaussianEProcess:
    """Sub-Gaussian e-process for sequential testing.

    For sub-Gaussian observations (bounded or light-tailed), constructs an
    e-process using the sub-Gaussian MGF bound.

    E_t = exp(lambda * S_t - lambda^2 * t * sigma^2 / 2)

    where S_t = sum(X_s - mu_0) and sigma is the sub-Gaussian parameter.

    Args:
        null_mean: Mean under the null hypothesis.
        sigma: Sub-Gaussian parameter (variance proxy).
        lambda_strategy: Strategy for choosing lambda.
        truncation: Maximum lambda value.
    """

    def __init__(
        self,
        null_mean: float = 0.0,
        sigma: float = 1.0,
        lambda_strategy: str = "predictable_plugin",
        truncation: float = 1.0,
    ) -> None:
        self.null_mean = null_mean
        self.sigma = sigma
        self.lambda_strategy = lambda_strategy
        self.truncation = truncation

        self._sum = 0.0
        self._sum_sq = 0.0
        self._t = 0
        self._log_e = 0.0
        self._history: List[float] = [1.0]
        self._lambda_history: List[float] = []

    @property
    def value(self) -> float:
        return float(np.exp(np.clip(self._log_e, -700, 700)))

    @property
    def log_value(self) -> float:
        return self._log_e

    def _choose_lambda(self) -> float:
        """Choose lambda based on the specified strategy."""
        if self._t == 0:
            return 0.0

        if self.lambda_strategy == "fixed":
            return min(1.0 / self.sigma, self.truncation)
        elif self.lambda_strategy == "predictable_plugin":
            # Use running mean as plug-in estimate of the alternative
            running_mean = self._sum / self._t - self.null_mean
            lam = running_mean / (self.sigma ** 2)
            return float(np.clip(lam, -self.truncation, self.truncation))
        elif self.lambda_strategy == "ons":
            # Online Newton Step (Cutkosky & Orabona, 2018)
            if self._t < 2:
                return 0.0
            var_est = max(
                self._sum_sq / self._t - (self._sum / self._t) ** 2,
                1e-10,
            )
            grad = (self._sum / self._t - self.null_mean) / var_est
            return float(np.clip(grad, -self.truncation, self.truncation))
        else:
            return 0.0

    def update(self, x_t: float) -> float:
        """Update the e-process with a new observation.

        Args:
            x_t: New observation.

        Returns:
            Updated e-process value.
        """
        lam = self._choose_lambda()
        self._lambda_history.append(lam)

        centered = x_t - self.null_mean
        self._sum += x_t
        self._sum_sq += x_t ** 2
        self._t += 1

        # E_t = exp(lambda * (x_t - mu_0) - lambda^2 * sigma^2 / 2)
        log_increment = lam * centered - 0.5 * lam ** 2 * self.sigma ** 2
        self._log_e += log_increment

        self._history.append(self.value)
        return self.value

    def update_batch(self, observations: Sequence[float]) -> float:
        """Process a batch of observations sequentially."""
        for x in observations:
            self.update(x)
        return self.value

    def reject(self, alpha: float = 0.05) -> bool:
        return self.value >= 1.0 / alpha

    def p_value(self) -> float:
        v = self.value
        if v <= 0:
            return 1.0
        return min(1.0, 1.0 / v)

    def get_confidence_sequence(
        self,
        alpha: float = 0.05,
    ) -> ConfidenceSequence:
        """Derive sub-Gaussian confidence sequence.

        CS_t = x_bar_t +/- sigma * sqrt(2 * log(1/alpha) / t)
        with iterated logarithm correction for uniformity.
        """
        if self._t == 0:
            return ConfidenceSequence(
                lower=np.array([]),
                upper=np.array([]),
                times=np.array([], dtype=np.int64),
                alpha=alpha,
            )

        times = np.arange(1, self._t + 1)
        lower = np.zeros(self._t)
        upper = np.zeros(self._t)

        cum_sum = 0.0
        for i in range(self._t):
            cum_sum += self._history[i + 1] if i + 1 < len(self._history) else 0
            n = i + 1
            x_bar = self._sum / self._t  # approximate
            # Stitched boundary (Howard et al. 2021)
            log_term = np.log(max(np.log(max(2.0 * n, np.e)), 1.0) / alpha)
            radius = self.sigma * np.sqrt(2 * log_term / n)
            lower[i] = x_bar - radius
            upper[i] = x_bar + radius

        return ConfidenceSequence(
            lower=lower,
            upper=upper,
            times=times,
            alpha=alpha,
            parameter_name="mean",
        )

    def reset(self) -> None:
        self._sum = 0.0
        self._sum_sq = 0.0
        self._t = 0
        self._log_e = 0.0
        self._history = [1.0]
        self._lambda_history = []


class SubExponentialEProcess:
    """Sub-exponential e-process for heavy-tailed sequential testing.

    For sub-exponential observations, uses truncated exponential bounds
    that are valid in the heavy-tail regime.

    E_t = exp(sum_s psi*(lambda * (X_s - mu_0)))

    where psi* is the Cramér transform for sub-exponential distributions.

    Args:
        null_mean: Mean under the null hypothesis.
        scale: Sub-exponential scale parameter (b).
        variance: Variance parameter (v).
        truncation_quantile: Quantile for observation truncation.
    """

    def __init__(
        self,
        null_mean: float = 0.0,
        scale: float = 1.0,
        variance: float = 1.0,
        truncation_quantile: float = 0.99,
    ) -> None:
        self.null_mean = null_mean
        self.scale = scale
        self.variance = variance
        self.truncation_quantile = truncation_quantile

        # Optimal lambda for sub-exponential: min(sqrt(v)/b, 1/(2b))
        self._lambda_max = min(
            np.sqrt(variance) / (scale + 1e-10),
            1.0 / (2 * scale + 1e-10),
        )

        self._t = 0
        self._log_e = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._observations: List[float] = []
        self._history: List[float] = [1.0]

    @property
    def value(self) -> float:
        return float(np.exp(np.clip(self._log_e, -700, 700)))

    def _cramer_transform(self, lam: float, x: float) -> float:
        """Compute the Cramér transform psi*(lambda * x).

        For sub-exponential with parameters (v, b):
        psi*(u) = u^2 / (2v) if |u| <= v/b
        psi*(u) = |u|/(2b) - v/(2b^2) if |u| > v/b
        """
        u = lam * x
        threshold = self.variance / (self.scale + 1e-10)

        if abs(u) <= threshold:
            return u ** 2 / (2 * self.variance)
        else:
            return abs(u) / (2 * self.scale) - self.variance / (2 * self.scale ** 2)

    def _choose_lambda(self) -> float:
        """Adaptive lambda choice for sub-exponential e-process."""
        if self._t < 2:
            return self._lambda_max * 0.1

        running_mean = self._sum / self._t
        deviation = running_mean - self.null_mean
        var_est = max(
            self._sum_sq / self._t - running_mean ** 2,
            1e-10,
        )

        # Optimal lambda balancing signal and noise
        lam = deviation / (var_est + self.scale * abs(deviation) + 1e-10)
        return float(np.clip(lam, -self._lambda_max, self._lambda_max))

    def update(self, x_t: float) -> float:
        """Update the sub-exponential e-process.

        Args:
            x_t: New observation.

        Returns:
            Updated e-process value.
        """
        # Truncation for robustness
        if len(self._observations) > 10:
            q_high = np.quantile(self._observations, self.truncation_quantile)
            q_low = np.quantile(self._observations, 1.0 - self.truncation_quantile)
            x_trunc = np.clip(x_t, q_low, q_high)
        else:
            x_trunc = x_t

        lam = self._choose_lambda()
        centered = x_trunc - self.null_mean

        # Log e-value increment via Cramér transform
        log_increment = lam * centered - self._cramer_transform(lam, centered)
        self._log_e += log_increment

        self._t += 1
        self._sum += x_t
        self._sum_sq += x_t ** 2
        self._observations.append(x_t)
        self._history.append(self.value)

        return self.value

    def update_batch(self, observations: Sequence[float]) -> float:
        for x in observations:
            self.update(x)
        return self.value

    def reject(self, alpha: float = 0.05) -> bool:
        return self.value >= 1.0 / alpha

    def p_value(self) -> float:
        v = self.value
        if v <= 0:
            return 1.0
        return min(1.0, 1.0 / v)

    def reset(self) -> None:
        self._t = 0
        self._log_e = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._observations = []
        self._history = [1.0]


class SequentialTest:
    """Flexible sequential test with optional stopping.

    Combines e-values from multiple sources with configurable stopping rules,
    spending functions, and multiple testing corrections.

    Args:
        alpha: Significance level.
        spending_fn: Alpha-spending function for group sequential design.
        max_observations: Maximum allowed observations.
        futility_bound: Futility boundary for early acceptance.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        spending_fn: Optional[Callable[[float], float]] = None,
        max_observations: Optional[int] = None,
        futility_bound: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.max_observations = max_observations
        self.futility_bound = futility_bound

        if spending_fn is None:
            # O'Brien-Fleming-like spending
            self.spending_fn = self._obrien_fleming_spending
        else:
            self.spending_fn = spending_fn

        self._wealth = WealthProcess()
        self._t = 0
        self._stopped = False
        self._rejected = False
        self._stopping_time: Optional[int] = None
        self._e_values: List[float] = []
        self._interim_alphas: List[float] = []

    def _obrien_fleming_spending(self, info_fraction: float) -> float:
        """O'Brien-Fleming alpha-spending function.

        alpha(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
        """
        if info_fraction <= 0:
            return 0.0
        z = stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(info_fraction)
        return float(2 * (1 - stats.norm.cdf(z)))

    def _pocock_spending(self, info_fraction: float) -> float:
        """Pocock alpha-spending function: alpha * log(1 + (e-1)*t)."""
        if info_fraction <= 0:
            return 0.0
        return float(self.alpha * np.log(1 + (np.e - 1) * info_fraction))

    def update(self, e_value: float) -> SequentialTestResult:
        """Process a new e-value and check stopping criteria.

        Args:
            e_value: New e-value.

        Returns:
            SequentialTestResult with current test status.
        """
        if self._stopped:
            return SequentialTestResult(
                rejected=self._rejected,
                stopped=True,
                stopping_time=self._stopping_time,
                e_value=self._wealth.current_wealth,
                p_value=min(1.0, 1.0 / max(self._wealth.current_wealth, 1e-300)),
                n_observations=self._t,
            )

        self._t += 1
        self._e_values.append(e_value)
        self._wealth.update(e_value)

        # Compute current information fraction
        if self.max_observations:
            info_frac = self._t / self.max_observations
        else:
            info_frac = min(1.0, self._t / 1000.0)

        # Current spending
        current_alpha = self.spending_fn(info_frac)
        self._interim_alphas.append(current_alpha)

        current_wealth = self._wealth.current_wealth

        # Check rejection: E_t >= 1/alpha_t
        if current_alpha > 0 and current_wealth >= 1.0 / current_alpha:
            self._stopped = True
            self._rejected = True
            self._stopping_time = self._t
        # Futility stopping
        elif (
            self.max_observations
            and self._t >= self.max_observations * 0.5
            and current_wealth < self.futility_bound
        ):
            self._stopped = True
            self._rejected = False
            self._stopping_time = self._t
        # Maximum observations reached
        elif self.max_observations and self._t >= self.max_observations:
            self._stopped = True
            self._rejected = current_wealth >= 1.0 / self.alpha
            self._stopping_time = self._t

        return SequentialTestResult(
            rejected=self._rejected,
            stopped=self._stopped,
            stopping_time=self._stopping_time,
            e_value=current_wealth,
            p_value=min(1.0, 1.0 / max(current_wealth, 1e-300)),
            n_observations=self._t,
        )

    def run(
        self,
        observations: NDArray[np.float64],
        e_value_fn: Callable[[float], float],
    ) -> SequentialTestResult:
        """Run the sequential test on a batch of observations.

        Args:
            observations: Array of observations.
            e_value_fn: Function mapping observation to e-value.

        Returns:
            Final SequentialTestResult.
        """
        result = SequentialTestResult(
            rejected=False,
            stopped=False,
            stopping_time=None,
            e_value=1.0,
            p_value=1.0,
        )
        for x in observations:
            e = e_value_fn(float(x))
            result = self.update(e)
            if result.stopped:
                break
        return result

    def reset(self) -> None:
        self._wealth = WealthProcess()
        self._t = 0
        self._stopped = False
        self._rejected = False
        self._stopping_time = None
        self._e_values = []
        self._interim_alphas = []


class AnytimeInference:
    """Anytime-valid inference engine combining multiple testing tools.

    Provides a unified interface for constructing confidence sequences,
    running sequential tests, and combining e-values from multiple sources.

    Args:
        alpha: Default significance level.
        bound_type: Type of concentration bound for confidence sequences.
        variance_estimator: How to estimate variance ("known", "plugin", "upper").
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bound_type: BoundType = BoundType.STITCHED,
        variance_estimator: str = "plugin",
    ) -> None:
        self.alpha = alpha
        self.bound_type = bound_type
        self.variance_estimator = variance_estimator

        self._observations: List[float] = []
        self._t = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def _estimate_variance(self) -> float:
        """Estimate variance based on configured strategy."""
        if self._t < 2:
            return 1.0
        if self.variance_estimator == "known":
            return 1.0
        elif self.variance_estimator == "plugin":
            mean = self._sum / self._t
            return max(self._sum_sq / self._t - mean ** 2, 1e-10)
        elif self.variance_estimator == "upper":
            mean = self._sum / self._t
            sample_var = max(self._sum_sq / self._t - mean ** 2, 1e-10)
            # Use upper confidence bound on variance
            chi2_lower = stats.chi2.ppf(0.025, df=self._t - 1)
            return sample_var * (self._t - 1) / chi2_lower
        return 1.0

    def add_observation(self, x: float) -> None:
        """Add a new observation to the running statistics."""
        self._observations.append(x)
        self._t += 1
        self._sum += x
        self._sum_sq += x ** 2

    def confidence_sequence(
        self,
        alpha: Optional[float] = None,
        observations: Optional[Sequence[float]] = None,
    ) -> ConfidenceSequence:
        """Construct a time-uniform confidence sequence.

        Implements the stitched method from Howard et al. (2021) which
        provides near-optimal width at all sample sizes.

        Args:
            alpha: Significance level (default: self.alpha).
            observations: Optional observations to process first.

        Returns:
            ConfidenceSequence with time-uniform guarantee.
        """
        if alpha is None:
            alpha = self.alpha

        if observations is not None:
            for x in observations:
                self.add_observation(x)

        if self._t == 0:
            return ConfidenceSequence(
                lower=np.array([]),
                upper=np.array([]),
                times=np.array([], dtype=np.int64),
                alpha=alpha,
            )

        obs = np.array(self._observations)
        times = np.arange(1, self._t + 1)
        lower = np.zeros(self._t)
        upper = np.zeros(self._t)

        if self.bound_type == BoundType.HOEFFDING:
            lower, upper = self._hoeffding_cs(obs, times, alpha)
        elif self.bound_type == BoundType.EMPIRICAL_BERNSTEIN:
            lower, upper = self._empirical_bernstein_cs(obs, times, alpha)
        elif self.bound_type == BoundType.POLYNOMIAL:
            lower, upper = self._polynomial_cs(obs, times, alpha)
        elif self.bound_type == BoundType.STITCHED:
            lower, upper = self._stitched_cs(obs, times, alpha)

        return ConfidenceSequence(
            lower=lower,
            upper=upper,
            times=times,
            alpha=alpha,
            parameter_name="mean",
        )

    def _hoeffding_cs(
        self,
        obs: NDArray[np.float64],
        times: NDArray[np.int64],
        alpha: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Hoeffding-style confidence sequence (sub-Gaussian bound)."""
        cum_sum = np.cumsum(obs)
        running_mean = cum_sum / times
        sigma = np.sqrt(self._estimate_variance())

        # Time-uniform Hoeffding bound with iterated log correction
        log_factor = np.log(
            np.maximum(np.log(np.maximum(2.0 * times, np.e)), 1.0) / alpha
        )
        radius = sigma * np.sqrt(2 * log_factor / times)

        return running_mean - radius, running_mean + radius

    def _empirical_bernstein_cs(
        self,
        obs: NDArray[np.float64],
        times: NDArray[np.int64],
        alpha: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Empirical Bernstein confidence sequence.

        Uses the empirical variance for tighter bounds when the variance
        is much smaller than the range.
        """
        cum_sum = np.cumsum(obs)
        running_mean = cum_sum / times

        # Running empirical variance
        cum_sum_sq = np.cumsum(obs ** 2)
        running_var = np.zeros_like(running_mean)
        for i in range(len(times)):
            n = times[i]
            if n < 2:
                running_var[i] = self._estimate_variance()
            else:
                running_var[i] = max(
                    cum_sum_sq[i] / n - (cum_sum[i] / n) ** 2,
                    1e-10,
                )

        # Empirical Bernstein bound
        log_factor = np.log(
            np.maximum(np.log(np.maximum(2.0 * times, np.e)), 1.0) / alpha
        )

        # Range term (assuming bounded observations)
        obs_range = np.max(np.abs(obs)) + 1e-10
        bernstein_radius = (
            np.sqrt(2 * running_var * log_factor / times)
            + obs_range * log_factor / (3 * times)
        )

        return running_mean - bernstein_radius, running_mean + bernstein_radius

    def _polynomial_cs(
        self,
        obs: NDArray[np.float64],
        times: NDArray[np.int64],
        alpha: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Polynomial confidence sequence using poly-stitching.

        Provides polynomial-time decay of width with good constants.
        """
        cum_sum = np.cumsum(obs)
        running_mean = cum_sum / times

        cum_sum_sq = np.cumsum(obs ** 2)
        running_var = np.maximum(
            cum_sum_sq / times - (cum_sum / times) ** 2,
            1e-10,
        )

        # Polynomial stitching boundary
        s = 1.4  # stitching parameter (near-optimal)
        log_factor = np.log(special.zeta(s, 1) / alpha)  # Riemann zeta
        poly_radius = np.sqrt(
            2 * running_var * (s * np.log(np.maximum(times, 1.0)) + log_factor) / times
        )

        return running_mean - poly_radius, running_mean + poly_radius

    def _stitched_cs(
        self,
        obs: NDArray[np.float64],
        times: NDArray[np.int64],
        alpha: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stitched confidence sequence (Howard et al. 2021).

        Near-optimal at all sample sizes by stitching together confidence
        intervals at geometrically-spaced checkpoints.
        """
        cum_sum = np.cumsum(obs)
        running_mean = cum_sum / times

        cum_sum_sq = np.cumsum(obs ** 2)
        running_var = np.maximum(
            cum_sum_sq / times - (cum_sum / times) ** 2,
            1e-10,
        )

        # Stitching parameter
        eta = 2.0  # geometric spacing factor
        s = 1.4    # polynomial decay exponent

        # Stitched boundary function
        def boundary(n: int, v: float) -> float:
            # Number of stitches up to n
            k = max(int(np.log(max(n, 1)) / np.log(eta)), 1)
            # Boundary value
            log_term = s * np.log(max(k, 1)) + np.log(special.zeta(s, 1) / alpha)
            return np.sqrt(2 * v * log_term / n)

        radii = np.array([
            boundary(int(times[i]), float(running_var[i]))
            for i in range(len(times))
        ])

        return running_mean - radii, running_mean + radii

    def running_intersection_cs(
        self,
        alpha: Optional[float] = None,
        observations: Optional[Sequence[float]] = None,
    ) -> ConfidenceSequence:
        """Confidence sequence with running intersection.

        Takes the intersection of all confidence sets up to each time,
        producing monotonically non-increasing width.
        """
        cs = self.confidence_sequence(alpha, observations)
        return cs.running_intersection()

    def combine_e_values(
        self,
        e_values: List[float],
        method: str = "product",
        weights: Optional[List[float]] = None,
    ) -> float:
        """Combine e-values from multiple sources.

        Args:
            e_values: List of e-values to combine.
            method: Combination method ("product", "average", "harmonic").
            weights: Weights for averaging methods.

        Returns:
            Combined e-value.
        """
        arr = np.array(e_values, dtype=np.float64)
        if len(arr) == 0:
            return 1.0

        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            w = w / w.sum()
        else:
            w = np.ones(len(arr)) / len(arr)

        if method == "product":
            log_sum = np.sum(np.log(np.maximum(arr, 1e-300)))
            return float(np.exp(np.clip(log_sum, -700, 700)))
        elif method == "average":
            # Arithmetic mean (valid e-value by convexity)
            return float(np.dot(w, arr))
        elif method == "harmonic":
            # Harmonic mean
            inv = 1.0 / np.maximum(arr, 1e-300)
            return float(1.0 / np.dot(w, inv))
        else:
            raise ValueError(f"Unknown method: {method}")

    def reset(self) -> None:
        self._observations = []
        self._t = 0
        self._sum = 0.0
        self._sum_sq = 0.0
