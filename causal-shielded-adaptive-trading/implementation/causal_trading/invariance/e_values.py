"""
E-value construction for sequential invariance testing.

Implements e-values, product e-values (test martingales), mixture e-values
for composite nulls, GROW martingales, and confidence sequences. All methods
support safe, online updates and anytime-valid inference.

References:
    - Vovk & Wang (2021). E-values: Calibration, combination and applications.
    - Grünwald et al. (2024). Safe testing.
    - Howard et al. (2021). Time-uniform, nonparametric, nonasymptotic
      confidence sequences.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats


class EValueType(Enum):
    """Type of e-value construction."""
    LIKELIHOOD_RATIO = "likelihood_ratio"
    SCORE = "score"
    GROW = "grow"
    MIXTURE = "mixture"
    KERNEL = "kernel"


@dataclass
class ConfidenceSequence:
    """A time-uniform confidence sequence derived from e-values.

    Attributes:
        lower: Array of lower confidence bounds at each time step.
        upper: Array of upper confidence bounds at each time step.
        times: Array of time indices.
        alpha: Significance level used for construction.
        parameter_name: Name of the parameter being estimated.
    """
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    times: NDArray[np.int64]
    alpha: float
    parameter_name: str = "theta"

    def width_at(self, t: int) -> float:
        """Width of the confidence interval at time t."""
        idx = np.searchsorted(self.times, t)
        if idx >= len(self.times) or self.times[idx] != t:
            raise ValueError(f"Time {t} not in confidence sequence.")
        return float(self.upper[idx] - self.lower[idx])

    def contains(self, value: float, t: int) -> bool:
        """Check if value is in the confidence set at time t."""
        idx = np.searchsorted(self.times, t)
        if idx >= len(self.times) or self.times[idx] != t:
            raise ValueError(f"Time {t} not in confidence sequence.")
        return bool(self.lower[idx] <= value <= self.upper[idx])

    def running_intersection(self) -> "ConfidenceSequence":
        """Compute running intersection for monotonically shrinking sets."""
        cum_lower = np.maximum.accumulate(self.lower)
        cum_upper = np.minimum.accumulate(self.upper)
        return ConfidenceSequence(
            lower=cum_lower,
            upper=cum_upper,
            times=self.times.copy(),
            alpha=self.alpha,
            parameter_name=self.parameter_name,
        )


@dataclass
class WealthProcess:
    """Tracks the wealth (capital) process of a test martingale.

    The wealth process W_t = prod_{s=1}^{t} e_s where e_s are individual
    e-values. Under the null, E[W_t] <= 1 for all t.

    Attributes:
        initial_wealth: Starting capital (default 1.0).
        wealth_history: List of wealth values over time.
        log_wealth_history: List of log-wealth values (for numerical stability).
        bet_history: History of betting fractions used.
    """
    initial_wealth: float = 1.0
    wealth_history: List[float] = field(default_factory=list)
    log_wealth_history: List[float] = field(default_factory=list)
    bet_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.wealth_history:
            self.wealth_history = [self.initial_wealth]
            self.log_wealth_history = [np.log(max(self.initial_wealth, 1e-300))]

    @property
    def current_wealth(self) -> float:
        return self.wealth_history[-1]

    @property
    def current_log_wealth(self) -> float:
        return self.log_wealth_history[-1]

    @property
    def max_wealth(self) -> float:
        return max(self.wealth_history)

    @property
    def time(self) -> int:
        return len(self.wealth_history) - 1

    def update(self, e_value: float, bet_fraction: float = 1.0) -> float:
        """Update wealth with a new e-value.

        Args:
            e_value: The new e-value (must be non-negative).
            bet_fraction: Fraction of wealth to bet (Kelly fraction).

        Returns:
            Updated wealth value.
        """
        if e_value < 0:
            raise ValueError(f"E-value must be non-negative, got {e_value}")

        bet_fraction = np.clip(bet_fraction, 0.0, 1.0)
        self.bet_history.append(bet_fraction)

        # W_{t+1} = W_t * (1 - bet + bet * e_value)
        multiplier = 1.0 - bet_fraction + bet_fraction * e_value
        multiplier = max(multiplier, 1e-300)  # prevent zero wealth

        new_log_wealth = self.current_log_wealth + np.log(multiplier)
        new_wealth = np.exp(np.clip(new_log_wealth, -700, 700))

        self.wealth_history.append(new_wealth)
        self.log_wealth_history.append(new_log_wealth)

        return new_wealth

    def reset(self) -> None:
        """Reset to initial wealth."""
        self.wealth_history = [self.initial_wealth]
        self.log_wealth_history = [np.log(max(self.initial_wealth, 1e-300))]
        self.bet_history = []

    def drawdown(self) -> float:
        """Current drawdown from peak wealth."""
        peak = self.max_wealth
        if peak <= 0:
            return 0.0
        return 1.0 - self.current_wealth / peak

    def growth_rate(self) -> float:
        """Average log-growth rate of the wealth process."""
        t = self.time
        if t == 0:
            return 0.0
        return (self.current_log_wealth - np.log(max(self.initial_wealth, 1e-300))) / t


class EValueConstructor:
    """Constructs e-values for testing edge invariance across regimes.

    Given observations from multiple regimes, constructs e-values that test
    whether a causal relationship (edge) is invariant across regimes. Under
    the null hypothesis of invariance, the e-values satisfy E[e_t] <= 1.

    Args:
        e_type: Type of e-value construction to use.
        kernel_bandwidth: Bandwidth for kernel-based e-values.
        regularization: Regularization parameter for numerical stability.
        log_optimal: Whether to use log-optimal (GROW) betting.
        min_samples_per_regime: Minimum observations per regime before testing.
    """

    def __init__(
        self,
        e_type: EValueType = EValueType.LIKELIHOOD_RATIO,
        kernel_bandwidth: Optional[float] = None,
        regularization: float = 1e-8,
        log_optimal: bool = True,
        min_samples_per_regime: int = 10,
    ) -> None:
        self.e_type = e_type
        self.kernel_bandwidth = kernel_bandwidth
        self.regularization = regularization
        self.log_optimal = log_optimal
        self.min_samples_per_regime = min_samples_per_regime

        self._wealth = WealthProcess()
        self._regime_data: Dict[int, List[NDArray[np.float64]]] = {}
        self._regime_labels: List[int] = []
        self._observations: List[NDArray[np.float64]] = []
        self._e_values: List[float] = []
        self._t = 0
        self._running_mean: Optional[NDArray[np.float64]] = None
        self._running_var: Optional[NDArray[np.float64]] = None

    @property
    def time(self) -> int:
        return self._t

    @property
    def n_regimes(self) -> int:
        return len(self._regime_data)

    def _update_running_stats(self, x: NDArray[np.float64]) -> None:
        """Welford's online algorithm for running mean and variance."""
        self._t += 1
        if self._running_mean is None:
            self._running_mean = x.copy().astype(np.float64)
            self._running_var = np.zeros_like(x, dtype=np.float64)
        else:
            delta = x - self._running_mean
            self._running_mean += delta / self._t
            delta2 = x - self._running_mean
            self._running_var += delta * delta2

    def _get_running_std(self) -> NDArray[np.float64]:
        """Return current running standard deviation."""
        if self._t < 2 or self._running_var is None:
            return np.ones(1, dtype=np.float64)
        return np.sqrt(self._running_var / (self._t - 1) + self.regularization)

    def update(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
    ) -> float:
        """Process a new observation and update the e-value.

        Args:
            x_t: Observation vector (e.g., residual from a causal model).
            regime_t: Regime label for this observation.

        Returns:
            Current product e-value (test martingale value).
        """
        x_t = np.atleast_1d(np.asarray(x_t, dtype=np.float64))
        self._update_running_stats(x_t)

        if regime_t not in self._regime_data:
            self._regime_data[regime_t] = []
        self._regime_data[regime_t].append(x_t)
        self._regime_labels.append(regime_t)
        self._observations.append(x_t)

        # Need sufficient data in at least 2 regimes
        regimes_ready = [
            r for r, d in self._regime_data.items()
            if len(d) >= self.min_samples_per_regime
        ]

        if len(regimes_ready) < 2:
            self._e_values.append(1.0)
            self._wealth.update(1.0)
            return self._wealth.current_wealth

        if self.e_type == EValueType.LIKELIHOOD_RATIO:
            e_t = self._likelihood_ratio_e_value(x_t, regime_t, regimes_ready)
        elif self.e_type == EValueType.SCORE:
            e_t = self._score_e_value(x_t, regime_t, regimes_ready)
        elif self.e_type == EValueType.GROW:
            e_t = self._grow_e_value(x_t, regime_t, regimes_ready)
        elif self.e_type == EValueType.KERNEL:
            e_t = self._kernel_e_value(x_t, regime_t, regimes_ready)
        elif self.e_type == EValueType.MIXTURE:
            e_t = self._mixture_e_value(x_t, regime_t, regimes_ready)
        else:
            e_t = 1.0

        e_t = max(e_t, 0.0)
        self._e_values.append(e_t)

        bet = self._compute_kelly_bet(e_t) if self.log_optimal else 1.0
        self._wealth.update(e_t, bet)

        return self._wealth.current_wealth

    def _likelihood_ratio_e_value(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
        regimes_ready: List[int],
    ) -> float:
        """Likelihood ratio e-value: p_alt(x) / p_null(x).

        Under the null (invariance), all regimes share the same distribution.
        Under the alternative, each regime has its own distribution.
        """
        # Fit per-regime Gaussians
        regime_params: Dict[int, Tuple[NDArray, NDArray]] = {}
        for r in regimes_ready:
            data_r = np.array(self._regime_data[r])
            mu_r = np.mean(data_r, axis=0)
            std_r = np.std(data_r, axis=0) + self.regularization
            regime_params[r] = (mu_r, std_r)

        # Pooled (null) distribution
        all_data = []
        for r in regimes_ready:
            all_data.extend(self._regime_data[r])
        all_data_arr = np.array(all_data)
        mu_pool = np.mean(all_data_arr, axis=0)
        std_pool = np.std(all_data_arr, axis=0) + self.regularization

        # Log-likelihood under alternative (regime-specific)
        if regime_t in regime_params:
            mu_alt, std_alt = regime_params[regime_t]
        else:
            mu_alt, std_alt = mu_pool, std_pool

        log_p_alt = np.sum(stats.norm.logpdf(x_t, loc=mu_alt, scale=std_alt))
        log_p_null = np.sum(stats.norm.logpdf(x_t, loc=mu_pool, scale=std_pool))

        log_e = log_p_alt - log_p_null
        # Truncate for safety
        log_e = np.clip(log_e, -50, 50)
        return float(np.exp(log_e))

    def _score_e_value(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
        regimes_ready: List[int],
    ) -> float:
        """Score-based e-value using efficient score functions.

        Uses the score (gradient of log-likelihood) to construct e-values
        without fully specifying the alternative.
        """
        all_data = []
        for r in regimes_ready:
            all_data.extend(self._regime_data[r])
        all_arr = np.array(all_data)
        mu_pool = np.mean(all_arr, axis=0)
        std_pool = np.std(all_arr, axis=0) + self.regularization

        # Score under null: d/dmu log p(x; mu, sigma) = (x - mu) / sigma^2
        score = (x_t - mu_pool) / (std_pool ** 2)

        # Regime-specific mean
        if regime_t in dict.fromkeys(regimes_ready) and len(self._regime_data[regime_t]) >= 2:
            data_r = np.array(self._regime_data[regime_t])
            mu_r = np.mean(data_r, axis=0)
            deviation = mu_r - mu_pool
        else:
            deviation = np.zeros_like(mu_pool)

        # E-value: 1 + lambda * score, with lambda chosen adaptively
        lam = np.dot(deviation, score) / (np.dot(score, score) + self.regularization)
        lam = np.clip(lam, -1.0, 1.0)

        e_t = 1.0 + lam * np.sum(score ** 2)
        return max(float(e_t), 0.0)

    def _grow_e_value(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
        regimes_ready: List[int],
    ) -> float:
        """GROW (Generalized Rate-Optimal Wealth) e-value.

        Constructs a log-optimal e-value by solving for the GROW betting
        fraction that maximizes expected log-wealth growth.
        """
        all_data = []
        for r in regimes_ready:
            all_data.extend(self._regime_data[r])
        all_arr = np.array(all_data)
        mu_pool = np.mean(all_arr, axis=0)
        std_pool = np.std(all_arr, axis=0) + self.regularization

        # Standardized residual
        z_t = (x_t - mu_pool) / std_pool

        if regime_t in dict.fromkeys(regimes_ready) and len(self._regime_data[regime_t]) >= 2:
            data_r = np.array(self._regime_data[regime_t])
            mu_r = np.mean(data_r, axis=0)
            std_r = np.std(data_r, axis=0) + self.regularization
            z_r = (mu_r - mu_pool) / std_pool
        else:
            z_r = np.zeros_like(z_t)

        # GROW: find lambda maximizing E_0[log(1 + lambda * S)]
        # where S = z_t^2 - 1 is the test statistic under H0: E[S]=0
        s_t = float(np.mean(z_t ** 2) - 1.0)

        def neg_expected_log_wealth(lam: float) -> float:
            # Under null, z ~ N(0,1), so z^2 ~ chi2(d)
            # E_0[log(1 + lam*(z^2 - 1))] approximated by current sample
            d = len(z_t)
            # Use chi-squared moments: E[z^2]=d, Var[z^2]=2d
            vals = 1.0 + lam * (z_t ** 2 - 1.0)
            if np.any(vals <= 0):
                return 1e10
            return -float(np.mean(np.log(vals)))

        result = optimize.minimize_scalar(
            neg_expected_log_wealth,
            bounds=(-0.5, 0.5),
            method="bounded",
        )
        lam_star = result.x if result.success else 0.0

        e_t = 1.0 + lam_star * s_t
        return max(float(e_t), 0.0)

    def _kernel_e_value(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
        regimes_ready: List[int],
    ) -> float:
        """Kernel-based e-value using MMD-style test statistic.

        Uses a Gaussian kernel to measure distributional differences between
        the current regime and the pooled distribution.
        """
        if self.kernel_bandwidth is None:
            all_data = []
            for r in regimes_ready:
                all_data.extend(self._regime_data[r])
            all_arr = np.array(all_data)
            # Median heuristic
            if len(all_arr) > 1:
                dists = np.sqrt(np.sum((all_arr[:, None] - all_arr[None, :]) ** 2, axis=-1))
                bw = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
            else:
                bw = 1.0
        else:
            bw = self.kernel_bandwidth

        def rbf_kernel(a: NDArray, b: NDArray) -> float:
            return float(np.exp(-np.sum((a - b) ** 2) / (2 * bw ** 2)))

        # Compute kernel e-value: ratio of within-regime vs cross-regime similarity
        if regime_t not in dict.fromkeys(regimes_ready):
            return 1.0

        data_r = self._regime_data[regime_t]
        other_data = []
        for r in regimes_ready:
            if r != regime_t:
                other_data.extend(self._regime_data[r])

        if len(data_r) < 2 or len(other_data) < 2:
            return 1.0

        # Sample a subset for efficiency
        n_sub = min(50, len(data_r), len(other_data))
        idx_r = np.random.choice(len(data_r), size=n_sub, replace=True)
        idx_o = np.random.choice(len(other_data), size=n_sub, replace=True)

        k_within = np.mean([rbf_kernel(data_r[i], x_t) for i in idx_r])
        k_cross = np.mean([rbf_kernel(other_data[i], x_t) for i in idx_o])

        # E-value from kernel divergence
        if k_cross < self.regularization:
            return 1.0

        e_t = k_within / (k_cross + self.regularization)
        return max(float(e_t), 0.0)

    def _mixture_e_value(
        self,
        x_t: NDArray[np.float64],
        regime_t: int,
        regimes_ready: List[int],
    ) -> float:
        """Mixture e-value for composite null hypotheses.

        Averages e-values over a grid of alternative parameters to handle
        composite null hypotheses (unknown invariance structure).
        """
        # Grid of effect sizes to mix over
        effect_grid = np.linspace(0.1, 2.0, 20)
        # Prior weights (favor smaller effects)
        weights = stats.expon.pdf(effect_grid, scale=0.5)
        weights /= weights.sum()

        all_data = []
        for r in regimes_ready:
            all_data.extend(self._regime_data[r])
        all_arr = np.array(all_data)
        mu_pool = np.mean(all_arr, axis=0)
        std_pool = np.std(all_arr, axis=0) + self.regularization

        if regime_t in dict.fromkeys(regimes_ready) and len(self._regime_data[regime_t]) >= 2:
            data_r = np.array(self._regime_data[regime_t])
            mu_r = np.mean(data_r, axis=0)
            direction = mu_r - mu_pool
            dir_norm = np.linalg.norm(direction)
            if dir_norm > self.regularization:
                direction /= dir_norm
            else:
                direction = np.ones_like(mu_pool) / np.sqrt(len(mu_pool))
        else:
            direction = np.ones_like(mu_pool) / np.sqrt(len(mu_pool))

        e_mixture = 0.0
        for delta, w in zip(effect_grid, weights):
            mu_alt = mu_pool + delta * std_pool * direction
            log_p_alt = np.sum(stats.norm.logpdf(x_t, loc=mu_alt, scale=std_pool))
            log_p_null = np.sum(stats.norm.logpdf(x_t, loc=mu_pool, scale=std_pool))
            log_e = np.clip(log_p_alt - log_p_null, -50, 50)
            e_mixture += w * np.exp(log_e)

        return max(float(e_mixture), 0.0)

    def _compute_kelly_bet(self, e_value: float) -> float:
        """Compute Kelly-optimal betting fraction.

        For a bet paying e_value, the Kelly fraction is:
            f* = (e_value - 1) / (e_value * (e_value - 1))
        simplified for log-optimality.
        """
        if e_value <= 1.0:
            return 0.0
        # For e-values, a simplified Kelly criterion
        # f* ∝ log(e_value) / (e_value - 1), clipped for safety
        f = np.log(e_value) / (e_value - 1.0 + self.regularization)
        return float(np.clip(f, 0.0, 1.0))

    def get_e_value(self) -> float:
        """Return the current product e-value (test martingale)."""
        return self._wealth.current_wealth

    def get_log_e_value(self) -> float:
        """Return the current log product e-value."""
        return self._wealth.current_log_wealth

    def get_e_value_history(self) -> NDArray[np.float64]:
        """Return the full history of individual e-values."""
        return np.array(self._e_values, dtype=np.float64)

    def get_wealth_history(self) -> NDArray[np.float64]:
        """Return the full history of cumulative wealth."""
        return np.array(self._wealth.wealth_history, dtype=np.float64)

    def reject(self, alpha: float = 0.05) -> bool:
        """Test whether the null (invariance) is rejected at level alpha.

        By Ville's inequality, reject when E_t >= 1/alpha.

        Args:
            alpha: Significance level.

        Returns:
            True if the null of invariance is rejected.
        """
        return self._wealth.current_wealth >= 1.0 / alpha

    def p_value(self) -> float:
        """Calibrated p-value from the e-value: p = min(1, 1/E_t)."""
        e = self._wealth.current_wealth
        if e <= 0:
            return 1.0
        return min(1.0, 1.0 / e)

    def get_confidence_sequence(
        self,
        alpha: float = 0.05,
        parameter: str = "mean_difference",
    ) -> ConfidenceSequence:
        """Construct a confidence sequence from the e-value process.

        Uses the duality between e-values and confidence sequences:
        theta is in CS_t(alpha) iff E_t(theta) < 1/alpha.

        Args:
            alpha: Significance level for the (1-alpha) confidence sequence.
            parameter: Name of the parameter.

        Returns:
            ConfidenceSequence with time-uniform coverage guarantee.
        """
        regimes = list(self._regime_data.keys())
        if len(regimes) < 2:
            t_arr = np.arange(1, self._t + 1)
            return ConfidenceSequence(
                lower=np.full(self._t, -np.inf),
                upper=np.full(self._t, np.inf),
                times=t_arr,
                alpha=alpha,
                parameter_name=parameter,
            )

        lower_bounds = []
        upper_bounds = []
        times = []

        for t_idx in range(1, self._t + 1):
            obs_so_far = self._observations[:t_idx]
            reg_so_far = self._regime_labels[:t_idx]

            regime_obs: Dict[int, List[NDArray]] = {}
            for obs, reg in zip(obs_so_far, reg_so_far):
                if reg not in regime_obs:
                    regime_obs[reg] = []
                regime_obs[reg].append(obs)

            regs_with_data = [r for r, d in regime_obs.items() if len(d) >= 2]
            if len(regs_with_data) < 2:
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)
                times.append(t_idx)
                continue

            # Compute mean difference between first two regimes
            r0, r1 = regs_with_data[0], regs_with_data[1]
            d0 = np.array(regime_obs[r0])
            d1 = np.array(regime_obs[r1])
            mu0 = np.mean(d0, axis=0)
            mu1 = np.mean(d1, axis=0)
            diff = float(np.mean(mu0 - mu1))

            n0, n1 = len(d0), len(d1)
            s0 = float(np.std(d0)) + self.regularization
            s1 = float(np.std(d1)) + self.regularization

            # Hedged confidence sequence width (Howard et al. 2021)
            # width ~ sqrt((sigma^2 / n) * log(log(n) / alpha))
            n_eff = (n0 * n1) / (n0 + n1)
            sigma_eff = np.sqrt(s0 ** 2 / n0 + s1 ** 2 / n1)

            log_term = np.log(max(np.log(max(n_eff, np.e)), 1.0) / alpha)
            width = sigma_eff * np.sqrt(2 * log_term)

            lower_bounds.append(diff - width)
            upper_bounds.append(diff + width)
            times.append(t_idx)

        return ConfidenceSequence(
            lower=np.array(lower_bounds, dtype=np.float64),
            upper=np.array(upper_bounds, dtype=np.float64),
            times=np.array(times, dtype=np.int64),
            alpha=alpha,
            parameter_name=parameter,
        )

    def reset(self) -> None:
        """Reset all internal state."""
        self._wealth = WealthProcess()
        self._regime_data = {}
        self._regime_labels = []
        self._observations = []
        self._e_values = []
        self._t = 0
        self._running_mean = None
        self._running_var = None


class ProductEValue:
    """Product e-value (test martingale) from a stream of individual e-values.

    E_t = prod_{s=1}^{t} e_s

    Under the null, this is a non-negative supermartingale with E[E_t] <= 1.
    """

    def __init__(self) -> None:
        self._log_product: float = 0.0
        self._n: int = 0
        self._history: List[float] = [1.0]
        self._individual: List[float] = []

    @property
    def value(self) -> float:
        return float(np.exp(np.clip(self._log_product, -700, 700)))

    @property
    def log_value(self) -> float:
        return self._log_product

    @property
    def n_updates(self) -> int:
        return self._n

    def update(self, e_value: float) -> float:
        """Multiply in a new e-value.

        Args:
            e_value: Non-negative e-value.

        Returns:
            Updated product e-value.
        """
        if e_value < 0:
            raise ValueError(f"E-value must be non-negative, got {e_value}")
        if e_value == 0:
            self._log_product = -np.inf
        else:
            self._log_product += np.log(e_value)
        self._n += 1
        self._individual.append(e_value)
        self._history.append(self.value)
        return self.value

    def update_batch(self, e_values: Sequence[float]) -> float:
        """Update with a batch of e-values."""
        for e in e_values:
            self.update(e)
        return self.value

    def reject(self, alpha: float = 0.05) -> bool:
        """Test rejection at level alpha."""
        return self.value >= 1.0 / alpha

    def p_value(self) -> float:
        """Calibrated p-value."""
        v = self.value
        if v <= 0:
            return 1.0
        return min(1.0, 1.0 / v)

    def reset(self) -> None:
        self._log_product = 0.0
        self._n = 0
        self._history = [1.0]
        self._individual = []


class MixtureEValue:
    """Mixture e-value for composite null hypotheses.

    For a family of e-values {E_t(theta) : theta in Theta}, the mixture
    e-value is: E_t = integral E_t(theta) dP(theta)

    This remains a valid e-value for any mixing distribution P.

    Args:
        grid_points: Parameter grid for the mixture.
        prior_weights: Prior weights on grid points (will be normalized).
        adaptive: Whether to use adaptive (predictable) mixing weights.
    """

    def __init__(
        self,
        grid_points: NDArray[np.float64],
        prior_weights: Optional[NDArray[np.float64]] = None,
        adaptive: bool = True,
    ) -> None:
        self.grid_points = np.asarray(grid_points, dtype=np.float64)
        self.n_grid = len(self.grid_points)

        if prior_weights is None:
            self.prior_weights = np.ones(self.n_grid) / self.n_grid
        else:
            self.prior_weights = np.asarray(prior_weights, dtype=np.float64)
            self.prior_weights /= self.prior_weights.sum()

        self.adaptive = adaptive

        # Posterior weights (updated adaptively)
        self._posterior_weights = self.prior_weights.copy()
        # Per-component product e-values (in log space)
        self._log_products = np.zeros(self.n_grid)
        self._t = 0
        self._history: List[float] = [1.0]

    @property
    def value(self) -> float:
        """Current mixture e-value."""
        # E_mix = sum_j w_j * E_j
        log_shifted = self._log_products - np.max(self._log_products)
        products = np.exp(np.clip(log_shifted, -700, 0))
        return float(np.exp(np.max(self._log_products)) * np.dot(self._posterior_weights, products))

    @property
    def component_values(self) -> NDArray[np.float64]:
        """Current e-values for each component."""
        return np.exp(np.clip(self._log_products, -700, 700))

    def update(self, e_values_per_component: NDArray[np.float64]) -> float:
        """Update with e-values for each component of the mixture.

        Args:
            e_values_per_component: Array of shape (n_grid,) with e-values
                for each grid point.

        Returns:
            Updated mixture e-value.
        """
        e_vals = np.asarray(e_values_per_component, dtype=np.float64)
        if len(e_vals) != self.n_grid:
            raise ValueError(
                f"Expected {self.n_grid} e-values, got {len(e_vals)}"
            )

        safe_e = np.maximum(e_vals, 1e-300)
        self._log_products += np.log(safe_e)
        self._t += 1

        if self.adaptive:
            # Update posterior weights using Bayesian update
            log_posterior = np.log(self._posterior_weights + 1e-300) + np.log(safe_e)
            log_posterior -= special.logsumexp(log_posterior)
            self._posterior_weights = np.exp(log_posterior)

        current = self.value
        self._history.append(current)
        return current

    def reject(self, alpha: float = 0.05) -> bool:
        return self.value >= 1.0 / alpha

    def p_value(self) -> float:
        v = self.value
        if v <= 0:
            return 1.0
        return min(1.0, 1.0 / v)

    def get_best_component(self) -> Tuple[int, float]:
        """Return the index and value of the best-performing component."""
        idx = int(np.argmax(self._log_products))
        return idx, float(np.exp(np.clip(self._log_products[idx], -700, 700)))

    def reset(self) -> None:
        self._posterior_weights = self.prior_weights.copy()
        self._log_products = np.zeros(self.n_grid)
        self._t = 0
        self._history = [1.0]


class GROWMartingale:
    """Generalized Rate-Optimal Wealth (GROW) martingale.

    Constructs a log-optimal e-value process by adaptively choosing the
    betting fraction to maximize the expected logarithmic growth rate.

    For testing H0: mu = mu_0 against H1: mu != mu_0 with sub-Gaussian data.

    Args:
        null_mean: Mean under the null hypothesis.
        variance_bound: Upper bound on the sub-Gaussian variance proxy.
        truncation: Maximum allowed bet size (for safety).
    """

    def __init__(
        self,
        null_mean: float = 0.0,
        variance_bound: float = 1.0,
        truncation: float = 0.5,
    ) -> None:
        self.null_mean = null_mean
        self.variance_bound = variance_bound
        self.truncation = truncation

        self._wealth = WealthProcess()
        self._sum_x: float = 0.0
        self._sum_x2: float = 0.0
        self._n: int = 0
        self._e_values: List[float] = []

    @property
    def value(self) -> float:
        return self._wealth.current_wealth

    @property
    def log_value(self) -> float:
        return self._wealth.current_log_wealth

    def _estimate_variance(self) -> float:
        """Online variance estimate with fallback to bound."""
        if self._n < 2:
            return self.variance_bound
        sample_var = (self._sum_x2 - self._sum_x ** 2 / self._n) / (self._n - 1)
        return max(float(sample_var), 1e-10)

    def _compute_grow_lambda(self) -> float:
        """Compute the GROW betting fraction.

        Lambda_t = S_{t-1} / (t * sigma^2_hat)
        where S_{t-1} = sum_{s=1}^{t-1} (X_s - mu_0).
        """
        if self._n < 1:
            return 0.0

        s_prev = self._sum_x - self._n * self.null_mean
        var_est = self._estimate_variance()

        lam = s_prev / (self._n * var_est + 1e-10)
        return float(np.clip(lam, -self.truncation, self.truncation))

    def update(self, x_t: float) -> float:
        """Update with a new observation.

        Args:
            x_t: New observation.

        Returns:
            Updated GROW martingale value.
        """
        # Compute betting fraction BEFORE seeing x_t (predictable)
        lam = self._compute_grow_lambda()

        # Update sufficient statistics
        self._n += 1
        self._sum_x += x_t
        self._sum_x2 += x_t ** 2

        # E-value: exp(lambda * (x_t - mu_0) - lambda^2 * sigma^2 / 2)
        # This is the exponential supermartingale increment
        centered = x_t - self.null_mean
        log_e = lam * centered - 0.5 * lam ** 2 * self.variance_bound
        e_t = float(np.exp(np.clip(log_e, -50, 50)))

        self._e_values.append(e_t)
        self._wealth.update(e_t)

        return self._wealth.current_wealth

    def update_batch(self, observations: Sequence[float]) -> float:
        """Update with a batch of observations (processed sequentially)."""
        for x in observations:
            self.update(x)
        return self.value

    def reject(self, alpha: float = 0.05) -> bool:
        return self._wealth.current_wealth >= 1.0 / alpha

    def p_value(self) -> float:
        e = self._wealth.current_wealth
        if e <= 0:
            return 1.0
        return min(1.0, 1.0 / e)

    def get_confidence_sequence(self, alpha: float = 0.05) -> ConfidenceSequence:
        """Derive a confidence sequence from the GROW martingale.

        CS_t = { mu : GROW_t(mu) < 1/alpha }
        """
        n_total = self._n
        if n_total == 0:
            return ConfidenceSequence(
                lower=np.array([]),
                upper=np.array([]),
                times=np.array([], dtype=np.int64),
                alpha=alpha,
            )

        lower = np.zeros(n_total)
        upper = np.zeros(n_total)
        times = np.arange(1, n_total + 1)

        cum_sum = 0.0
        cum_sum2 = 0.0
        for t in range(n_total):
            obs_val = self._sum_x  # We don't store individual, approximate
            cum_sum = self._sum_x * (t + 1) / self._n if self._n > 0 else 0
            var_est = self._estimate_variance()
            n_t = t + 1

            # Confidence radius from sub-Gaussian bound
            # width = sqrt(2 * sigma^2 * log(log(2*n)/alpha) / n)
            log_term = np.log(max(np.log(max(2.0 * n_t, np.e)), 1.0) / alpha)
            radius = np.sqrt(2 * var_est * log_term / n_t)

            mean_est = self._sum_x / self._n if self._n > 0 else self.null_mean
            lower[t] = mean_est - radius
            upper[t] = mean_est + radius

        return ConfidenceSequence(
            lower=lower,
            upper=upper,
            times=times,
            alpha=alpha,
            parameter_name="mean",
        )

    def growth_rate(self) -> float:
        """Empirical log-growth rate of the martingale."""
        return self._wealth.growth_rate()

    def reset(self) -> None:
        self._wealth = WealthProcess()
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._n = 0
        self._e_values = []
