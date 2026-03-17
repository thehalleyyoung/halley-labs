"""Statistical power analysis for collusion detection tests.

Computes required sample sizes, power curves, and effect size estimates
for the composite hypothesis testing framework.
"""

import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Optional, Tuple, Any


class PowerAnalyzer:
    """Statistical power analysis for collusion detection."""

    def __init__(self, alpha: float = 0.05, num_simulations: int = 5000,
                 random_state: Optional[int] = None):
        self.alpha = alpha
        self.num_simulations = num_simulations
        self.rng = np.random.RandomState(random_state)

    def required_sample_size(self, effect_size: float, power: float = 0.8,
                             test_type: str = "one_sample_t",
                             alternative: str = "greater") -> int:
        """Compute minimum sample size for desired power.

        Args:
            effect_size: Cohen's d or equivalent.
            power: Desired power (1 - beta).
            test_type: "one_sample_t", "two_sample_t", "correlation", "proportion".
            alternative: "two-sided", "greater", "less".

        Returns:
            Minimum integer sample size achieving the requested power.
        """
        if effect_size <= 0:
            raise ValueError("effect_size must be positive")
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")

        if test_type == "one_sample_t":
            def power_func(n):
                return self._analytical_power_one_sample_t(
                    int(n), effect_size, self.alpha, alternative
                )
            return self._solve_for_n(power_func, power)

        elif test_type == "two_sample_t":
            def power_func(n):
                return self._analytical_power_two_sample_t(
                    int(n), int(n), effect_size, self.alpha
                )
            return self._solve_for_n(power_func, power)

        elif test_type == "correlation":
            def power_func(n):
                return self.power_for_correlation_test(int(n), effect_size)
            return self._solve_for_n(power_func, power)

        elif test_type == "proportion":
            def power_func(n):
                return self._power_proportion_test(int(n), effect_size, self.alpha)
            return self._solve_for_n(power_func, power)

        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    def power_curve(self, sample_sizes: Optional[List[int]] = None,
                    effect_size: float = 0.5,
                    test_type: str = "one_sample_t") -> Dict[str, Any]:
        """Compute power as function of sample size.

        Returns:
            Dict with 'sample_sizes' and 'power_values' arrays, plus metadata.
        """
        if sample_sizes is None:
            sample_sizes = list(range(10, 510, 10))

        power_values = np.empty(len(sample_sizes))
        for i, n in enumerate(sample_sizes):
            if test_type == "one_sample_t":
                power_values[i] = self._analytical_power_one_sample_t(
                    n, effect_size, self.alpha
                )
            elif test_type == "two_sample_t":
                power_values[i] = self._analytical_power_two_sample_t(
                    n, n, effect_size, self.alpha
                )
            elif test_type == "correlation":
                power_values[i] = self.power_for_correlation_test(n, effect_size)
            else:
                power_values[i] = self._analytical_power_one_sample_t(
                    n, effect_size, self.alpha
                )

        idx_80 = np.searchsorted(power_values, 0.8)
        n_for_80 = sample_sizes[idx_80] if idx_80 < len(sample_sizes) else None

        return {
            "sample_sizes": np.array(sample_sizes),
            "power_values": power_values,
            "effect_size": effect_size,
            "test_type": test_type,
            "alpha": self.alpha,
            "n_for_80_power": n_for_80,
        }

    def power_vs_effect_size(self, sample_size: int,
                             effect_sizes: Optional[np.ndarray] = None,
                             test_type: str = "one_sample_t") -> Dict[str, Any]:
        """Compute power as function of effect size for fixed n.

        Returns:
            Dict with 'effect_sizes', 'power_values' arrays and metadata.
        """
        if effect_sizes is None:
            effect_sizes = np.linspace(0.01, 2.0, 100)

        power_values = np.empty(len(effect_sizes))
        for i, d in enumerate(effect_sizes):
            if test_type == "one_sample_t":
                power_values[i] = self._analytical_power_one_sample_t(
                    sample_size, d, self.alpha
                )
            elif test_type == "two_sample_t":
                power_values[i] = self._analytical_power_two_sample_t(
                    sample_size, sample_size, d, self.alpha
                )
            elif test_type == "correlation":
                power_values[i] = self.power_for_correlation_test(sample_size, d)
            else:
                power_values[i] = self._analytical_power_one_sample_t(
                    sample_size, d, self.alpha
                )

        idx_80 = np.searchsorted(power_values, 0.8)
        mde = effect_sizes[idx_80] if idx_80 < len(effect_sizes) else None

        return {
            "effect_sizes": effect_sizes,
            "power_values": power_values,
            "sample_size": sample_size,
            "test_type": test_type,
            "alpha": self.alpha,
            "minimum_detectable_effect": mde,
        }

    def simulate_power(self, sample_size: int, effect_size: float,
                       test_func: Optional[callable] = None,
                       noise_std: float = 1.0) -> float:
        """Estimate power via Monte Carlo simulation.

        If *test_func* is None a one-sample t-test (H1: mean > 0) is used.
        *test_func* should accept a 1-d array and return a p-value.
        """
        rejections = 0
        for _ in range(self.num_simulations):
            data = self.rng.normal(
                loc=effect_size * noise_std, scale=noise_std, size=sample_size
            )
            if test_func is not None:
                p_value = test_func(data)
            else:
                t_stat, p_two = stats.ttest_1samp(data, 0.0)
                p_value = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

            if p_value < self.alpha:
                rejections += 1

        return rejections / self.num_simulations

    def minimum_detectable_effect(self, sample_size: int, power: float = 0.8,
                                  test_type: str = "one_sample_t") -> float:
        """Find minimum detectable effect size for given n and power.

        Uses Brent's method to invert the power function.
        """
        def objective(d):
            if test_type == "one_sample_t":
                pwr = self._analytical_power_one_sample_t(
                    sample_size, d, self.alpha
                )
            elif test_type == "two_sample_t":
                pwr = self._analytical_power_two_sample_t(
                    sample_size, sample_size, d, self.alpha
                )
            elif test_type == "correlation":
                pwr = self.power_for_correlation_test(sample_size, d)
            else:
                pwr = self._analytical_power_one_sample_t(
                    sample_size, d, self.alpha
                )
            return pwr - power

        try:
            result = optimize.brentq(objective, 1e-6, 10.0)
        except ValueError:
            result = np.nan
        return float(result)

    def collusion_detection_power(self, num_rounds: int, num_players: int,
                                  true_premium: float, noise_std: float,
                                  nash_price: float = 1.0,
                                  monopoly_price: float = 5.5) -> Dict[str, Any]:
        """Compute power specifically for collusion premium detection.

        Simulates price trajectories with a known collusive premium above
        *nash_price*, adds Gaussian noise, and estimates the rejection rate
        of a one-sample t-test against the Nash baseline.
        """
        rejections = 0
        observed_premiums: List[float] = []

        for _ in range(self.num_simulations):
            prices = self.rng.normal(
                loc=nash_price + true_premium, scale=noise_std,
                size=(num_rounds, num_players),
            )
            avg_price = prices.mean(axis=1)
            premium = avg_price - nash_price
            mean_premium = premium.mean()
            observed_premiums.append(mean_premium)

            t_stat, p_two = stats.ttest_1samp(premium, 0.0)
            p_value = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
            if p_value < self.alpha:
                rejections += 1

        estimated_power = rejections / self.num_simulations
        price_range = monopoly_price - nash_price
        normalised_premium = true_premium / price_range if price_range > 0 else np.nan

        return {
            "estimated_power": estimated_power,
            "num_rounds": num_rounds,
            "num_players": num_players,
            "true_premium": true_premium,
            "noise_std": noise_std,
            "nash_price": nash_price,
            "monopoly_price": monopoly_price,
            "normalised_premium": normalised_premium,
            "mean_observed_premium": float(np.mean(observed_premiums)),
            "std_observed_premium": float(np.std(observed_premiums)),
            "alpha": self.alpha,
            "num_simulations": self.num_simulations,
        }

    def power_for_correlation_test(self, sample_size: int,
                                   true_correlation: float) -> float:
        """Power of correlation test using Fisher z-transform.

        Uses the asymptotic normal approximation to the distribution of
        the Fisher-transformed sample correlation coefficient.
        """
        if sample_size < 4:
            return 0.0
        z_r = np.arctanh(true_correlation)
        se = 1.0 / np.sqrt(sample_size - 3)
        z_crit = stats.norm.ppf(1.0 - self.alpha)
        power = 1.0 - stats.norm.cdf(z_crit - z_r / se)
        return float(np.clip(power, 0.0, 1.0))

    def power_for_granger_test(self, sample_size: int, num_lags: int,
                               true_effect: float = 0.1) -> float:
        """Approximate power of Granger causality test.

        Models the Granger test as an incremental F-test with *num_lags*
        additional regressors, using the noncentral F distribution.
        """
        if sample_size <= 2 * num_lags + 1:
            return 0.0

        df1 = num_lags
        df2 = sample_size - 2 * num_lags - 1
        noncentrality = sample_size * true_effect ** 2
        f_crit = stats.f.ppf(1.0 - self.alpha, df1, df2)
        power = 1.0 - stats.ncf.cdf(f_crit, df1, df2, noncentrality)
        return float(np.clip(power, 0.0, 1.0))

    def sample_size_table(self, effect_sizes: List[float],
                          power_levels: List[float],
                          test_type: str = "one_sample_t") -> Dict[str, Any]:
        """Generate sample size table for multiple effect/power combos.

        Returns:
            Dict with 'effect_sizes', 'power_levels', and a 2-d
            'sample_sizes' array of shape (len(effect_sizes), len(power_levels)).
        """
        table = np.empty((len(effect_sizes), len(power_levels)), dtype=int)
        for i, d in enumerate(effect_sizes):
            for j, pwr in enumerate(power_levels):
                table[i, j] = self.required_sample_size(
                    d, power=pwr, test_type=test_type
                )
        return {
            "effect_sizes": effect_sizes,
            "power_levels": power_levels,
            "sample_sizes": table,
            "test_type": test_type,
            "alpha": self.alpha,
        }

    def retrospective_power(self, observed_effect: float, sample_size: int,
                            observed_std: float) -> float:
        """Compute observed (retrospective / post-hoc) power.

        Given the effect and standard deviation that were actually observed,
        returns the probability that the test would have rejected at level
        *alpha*.
        """
        if observed_std <= 0:
            raise ValueError("observed_std must be positive")
        d = observed_effect / observed_std
        return self._analytical_power_one_sample_t(sample_size, d, self.alpha)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analytical_power_one_sample_t(self, n: int, d: float, alpha: float,
                                       alternative: str = "greater") -> float:
        """Analytical power for one-sample t-test using noncentral t.

        Parameters:
            n: sample size
            d: Cohen's d (effect_size / std)
            alpha: significance level
            alternative: "greater", "less", or "two-sided"
        """
        if n < 2:
            return 0.0
        df = n - 1
        noncentrality = d * np.sqrt(n)

        if alternative == "two-sided":
            t_crit_upper = stats.t.ppf(1.0 - alpha / 2.0, df)
            power = (
                1.0 - stats.nct.cdf(t_crit_upper, df, noncentrality)
                + stats.nct.cdf(-t_crit_upper, df, noncentrality)
            )
        elif alternative == "greater":
            t_crit = stats.t.ppf(1.0 - alpha, df)
            power = 1.0 - stats.nct.cdf(t_crit, df, noncentrality)
        elif alternative == "less":
            t_crit = stats.t.ppf(alpha, df)
            power = stats.nct.cdf(t_crit, df, noncentrality)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        return float(np.clip(power, 0.0, 1.0))

    def _analytical_power_two_sample_t(self, n1: int, n2: int, d: float,
                                       alpha: float) -> float:
        """Analytical power for two-sample t-test (two-sided)."""
        if n1 < 2 or n2 < 2:
            return 0.0
        df = n1 + n2 - 2
        noncentrality = d * np.sqrt(n1 * n2 / (n1 + n2))
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        power = (
            1.0 - stats.nct.cdf(t_crit, df, noncentrality)
            + stats.nct.cdf(-t_crit, df, noncentrality)
        )
        return float(np.clip(power, 0.0, 1.0))

    def _power_proportion_test(self, n: int, effect_size: float,
                               alpha: float) -> float:
        """Power for a one-sample proportion z-test.

        *effect_size* is Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p0)).
        """
        if n < 1:
            return 0.0
        z_crit = stats.norm.ppf(1.0 - alpha)
        z_effect = effect_size * np.sqrt(n)
        power = 1.0 - stats.norm.cdf(z_crit - z_effect)
        return float(np.clip(power, 0.0, 1.0))

    def _solve_for_n(self, power_func, target_power: float,
                     n_min: int = 10, n_max: int = 1_000_000) -> int:
        """Binary search for required n.

        Finds the smallest integer *n* in [n_min, n_max] such that
        ``power_func(n) >= target_power``.
        """
        if power_func(n_max) < target_power:
            return n_max

        lo, hi = n_min, n_max
        while lo < hi:
            mid = (lo + hi) // 2
            if power_func(mid) >= target_power:
                hi = mid
            else:
                lo = mid + 1
        return lo
