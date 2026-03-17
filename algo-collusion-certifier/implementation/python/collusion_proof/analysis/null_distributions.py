"""Null distribution simulation for collusion hypothesis testing.

Generates null distributions under competitive equilibrium assumptions
for calibrating test statistics and computing p-values.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Callable


class NullDistributionSimulator:
    """Simulate null distributions under competitive behavior hypotheses."""

    def __init__(self, num_simulations: int = 10000, random_state: Optional[int] = None):
        self.num_simulations = num_simulations
        self.rng = np.random.RandomState(random_state)

    def competitive_trajectory(self, num_rounds: int, num_players: int,
                               nash_price: float, noise_std: float = 0.1,
                               autocorrelation: float = 0.0) -> np.ndarray:
        """Simulate a competitive price trajectory under the null.

        Generates prices centered around Nash equilibrium with noise
        and optional autocorrelation (AR(1) process).

        Args:
            num_rounds: Number of time periods
            num_players: Number of firms
            nash_price: Nash equilibrium price
            noise_std: Standard deviation of noise
            autocorrelation: AR(1) coefficient (0 = i.i.d.)

        Returns:
            prices array of shape (num_rounds, num_players)
        """
        phi = np.clip(autocorrelation, -0.999, 0.999)
        # Innovation variance chosen so marginal variance = noise_std^2
        if abs(phi) < 1e-12:
            innovation_std = noise_std
        else:
            innovation_std = noise_std * np.sqrt(1.0 - phi ** 2)

        prices = np.empty((num_rounds, num_players))

        # Draw initial prices from the stationary distribution
        prices[0, :] = nash_price + self.rng.normal(0.0, noise_std, size=num_players)

        for t in range(1, num_rounds):
            innovation = self.rng.normal(0.0, innovation_std, size=num_players)
            prices[t, :] = nash_price + phi * (prices[t - 1, :] - nash_price) + innovation

        return prices

    def simulate_null_statistic(self, statistic_func: Callable[[np.ndarray], float],
                                num_rounds: int, num_players: int,
                                nash_price: float, noise_std: float = 0.1,
                                autocorrelation: float = 0.0) -> np.ndarray:
        """Simulate null distribution of a test statistic.

        Generates competitive trajectories and computes the statistic on each.

        Returns:
            Array of simulated statistics under the null.
        """
        null_stats = np.empty(self.num_simulations)
        for i in range(self.num_simulations):
            trajectory = self.competitive_trajectory(
                num_rounds, num_players, nash_price, noise_std, autocorrelation
            )
            null_stats[i] = statistic_func(trajectory)
        return null_stats

    def monte_carlo_p_value(self, observed_statistic: float,
                            null_distribution: np.ndarray,
                            alternative: str = "greater") -> float:
        """Compute Monte Carlo p-value from null distribution.

        Uses the (b+1)/(m+1) correction to avoid zero p-values,
        where b is the count of null values at least as extreme
        as observed, and m is the number of simulations.
        """
        m = len(null_distribution)
        if alternative == "greater":
            b = np.sum(null_distribution >= observed_statistic)
        elif alternative == "less":
            b = np.sum(null_distribution <= observed_statistic)
        elif alternative == "two-sided":
            center = np.median(null_distribution)
            obs_dev = abs(observed_statistic - center)
            b = np.sum(np.abs(null_distribution - center) >= obs_dev)
        else:
            raise ValueError(f"Unknown alternative '{alternative}'. "
                             "Use 'greater', 'less', or 'two-sided'.")
        return (b + 1) / (m + 1)

    def calibrated_critical_value(self, null_distribution: np.ndarray,
                                  alpha: float = 0.05,
                                  alternative: str = "greater") -> float:
        """Get calibrated critical value from null distribution."""
        if alternative == "greater":
            return float(np.percentile(null_distribution, 100 * (1 - alpha)))
        elif alternative == "less":
            return float(np.percentile(null_distribution, 100 * alpha))
        elif alternative == "two-sided":
            lower = float(np.percentile(null_distribution, 100 * alpha / 2))
            upper = float(np.percentile(null_distribution, 100 * (1 - alpha / 2)))
            return upper  # return the upper tail; caller can derive both
        else:
            raise ValueError(f"Unknown alternative '{alternative}'.")

    def berry_esseen_correction(self, data: np.ndarray,
                                null_mean: float = 0.0) -> Dict[str, Any]:
        """Apply Berry-Esseen theorem correction for non-normal sampling.

        Provides bounds on the error of normal approximation
        using the third moment.
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-15:
            return {"correction": 0.0, "bound": 0.0, "reliable": False}

        # Third absolute moment
        centered = data - mean
        rho = np.mean(np.abs(centered) ** 3)
        sigma3 = std ** 3

        # Berry-Esseen bound: C * rho / (sigma^3 * sqrt(n))
        C_BE = 0.4748  # Best known constant
        bound = C_BE * rho / (sigma3 * np.sqrt(n) + 1e-15)

        # Corrected z-statistic
        z = np.sqrt(n) * (mean - null_mean) / std

        # Corrected p-value bounds
        p_normal = stats.norm.sf(abs(z))
        p_lower = max(0, p_normal - bound)
        p_upper = min(1, p_normal + bound)

        return {
            "z_statistic": z,
            "p_value_normal": p_normal,
            "p_value_lower": p_lower,
            "p_value_upper": p_upper,
            "berry_esseen_bound": bound,
            "correction": bound,
            "reliable": bound < 0.01,
            "third_moment_ratio": rho / sigma3,
        }

    def simulate_price_level_null(self, num_rounds: int, num_players: int,
                                  nash_price: float, noise_std: float = 0.1) -> np.ndarray:
        """Null distribution for the price level test statistic.

        Under H0, prices are Nash + noise.
        Test statistic: (mean_price - nash_price) / SE(mean_price)
        """
        null_stats = np.empty(self.num_simulations)
        for i in range(self.num_simulations):
            trajectory = self.competitive_trajectory(
                num_rounds, num_players, nash_price, noise_std
            )
            grand_mean = np.mean(trajectory)
            se = np.std(trajectory, ddof=1) / np.sqrt(trajectory.size)
            if se < 1e-15:
                null_stats[i] = 0.0
            else:
                null_stats[i] = (grand_mean - nash_price) / se
        return null_stats

    def simulate_correlation_null(self, num_rounds: int, num_players: int,
                                  noise_std: float = 0.1,
                                  autocorrelation: float = 0.0) -> np.ndarray:
        """Null distribution for cross-firm correlation.

        Under H0, firms price independently.
        Accounts for spurious correlation from common trends.
        """
        null_stats = np.empty(self.num_simulations)
        for i in range(self.num_simulations):
            trajectory = self.competitive_trajectory(
                num_rounds, num_players, nash_price=1.0,
                noise_std=noise_std, autocorrelation=autocorrelation
            )
            # Compute average pairwise Pearson correlation
            if num_players < 2:
                null_stats[i] = 0.0
                continue
            corr_sum = 0.0
            pair_count = 0
            for a in range(num_players):
                for b in range(a + 1, num_players):
                    r, _ = stats.pearsonr(trajectory[:, a], trajectory[:, b])
                    corr_sum += r
                    pair_count += 1
            null_stats[i] = corr_sum / pair_count if pair_count > 0 else 0.0
        return null_stats

    def simulate_punishment_null(self, num_rounds: int, num_players: int,
                                 nash_price: float, noise_std: float = 0.1) -> np.ndarray:
        """Null distribution for punishment test.

        Under H0, there is no systematic price response to deviations.
        Test statistic: regression coefficient of price_change(t) on
        negative_deviation(t-1), averaged across players.
        """
        null_stats = np.empty(self.num_simulations)
        for i in range(self.num_simulations):
            trajectory = self.competitive_trajectory(
                num_rounds, num_players, nash_price, noise_std
            )
            betas = []
            for p in range(num_players):
                prices_p = trajectory[:, p]
                mean_others = np.mean(
                    np.delete(trajectory, p, axis=1), axis=1
                ) if num_players > 1 else prices_p

                # Deviation of player p below the mean of others
                deviation = prices_p - mean_others
                negative_dev = np.where(deviation < 0, deviation, 0.0)

                # Price change of others in the next period
                if num_players > 1:
                    others_mean = np.mean(np.delete(trajectory, p, axis=1), axis=1)
                    price_change = np.diff(others_mean)
                else:
                    price_change = np.diff(prices_p)

                x = negative_dev[:-1]
                y = price_change
                if len(x) < 2 or np.std(x) < 1e-15:
                    betas.append(0.0)
                    continue

                # OLS slope
                x_centered = x - np.mean(x)
                beta = np.dot(x_centered, y - np.mean(y)) / (np.dot(x_centered, x_centered) + 1e-15)
                betas.append(beta)

            null_stats[i] = np.mean(betas) if betas else 0.0
        return null_stats

    def adaptive_calibration(self, observed_data: np.ndarray,
                             statistic_func: Callable,
                             nash_price: float) -> Dict[str, Any]:
        """Adaptively calibrate the null distribution to the data.

        Estimates noise and autocorrelation from data, then simulates
        null matching these characteristics.
        """
        num_rounds, num_players = observed_data.shape

        # Estimate noise_std from residuals around nash_price
        residuals = observed_data - nash_price
        noise_std_hat = float(np.std(residuals, ddof=1))

        # Estimate autocorrelation per player and average
        ac_estimates = []
        for p in range(num_players):
            series = observed_data[:, p] - nash_price
            if len(series) < 3:
                ac_estimates.append(0.0)
                continue
            lag1_cov = np.mean(series[1:] * series[:-1])
            var = np.mean(series ** 2)
            if var < 1e-15:
                ac_estimates.append(0.0)
            else:
                ac_estimates.append(np.clip(lag1_cov / var, -0.999, 0.999))
        autocorrelation_hat = float(np.mean(ac_estimates))

        # Simulate null with estimated parameters
        null_dist = self.simulate_null_statistic(
            statistic_func, num_rounds, num_players,
            nash_price, noise_std_hat, autocorrelation_hat
        )

        observed_stat = statistic_func(observed_data)
        p_value = self.monte_carlo_p_value(observed_stat, null_dist, alternative="greater")

        return {
            "null_distribution": null_dist,
            "observed_statistic": observed_stat,
            "p_value": p_value,
            "estimated_noise_std": noise_std_hat,
            "estimated_autocorrelation": autocorrelation_hat,
            "num_simulations": self.num_simulations,
            "critical_value_05": self.calibrated_critical_value(null_dist, alpha=0.05),
            "critical_value_01": self.calibrated_critical_value(null_dist, alpha=0.01),
        }

    def permutation_null(self, data: np.ndarray,
                         statistic_func: Callable[[np.ndarray], float],
                         n_permutations: Optional[int] = None) -> np.ndarray:
        """Generate null via permutation.

        Permutes time indices to break temporal dependence while
        preserving marginal distributions of each player.
        """
        n_perm = n_permutations if n_permutations is not None else self.num_simulations
        null_stats = np.empty(n_perm)
        num_rounds = data.shape[0]

        for i in range(n_perm):
            permuted = data.copy()
            perm_idx = self.rng.permutation(num_rounds)
            permuted = permuted[perm_idx, :]
            null_stats[i] = statistic_func(permuted)

        return null_stats

    def phase_shuffle_null(self, data: np.ndarray,
                           statistic_func: Callable[[np.ndarray], float]) -> np.ndarray:
        """Phase-shuffle surrogate: preserves power spectrum but breaks cross-correlation.

        Randomizes Fourier phases while keeping magnitudes.
        Each column (player) receives independent random phases.
        """
        num_rounds, num_players = data.shape
        null_stats = np.empty(self.num_simulations)

        for i in range(self.num_simulations):
            surrogate = np.empty_like(data)
            for p in range(num_players):
                series = data[:, p]
                fft_vals = np.fft.rfft(series)
                magnitudes = np.abs(fft_vals)
                n_freq = len(fft_vals)

                # Random phases; DC component (index 0) keeps its phase.
                # If num_rounds is even, the Nyquist component also keeps its phase.
                random_phases = self.rng.uniform(0, 2 * np.pi, size=n_freq)
                random_phases[0] = 0.0
                if num_rounds % 2 == 0 and n_freq > 1:
                    random_phases[-1] = 0.0

                shuffled_fft = magnitudes * np.exp(1j * random_phases)
                surrogate[:, p] = np.fft.irfft(shuffled_fft, n=num_rounds)

            null_stats[i] = statistic_func(surrogate)

        return null_stats

    def block_permutation_null(self, data: np.ndarray,
                               statistic_func: Callable[[np.ndarray], float],
                               block_size: int = 100) -> np.ndarray:
        """Block permutation null that preserves local dependence.

        Splits the time series into non-overlapping blocks of the given
        size, permutes the block order, and recomputes the statistic.
        The final partial block is kept as-is at the end.
        """
        num_rounds = data.shape[0]
        null_stats = np.empty(self.num_simulations)

        effective_block = max(1, min(block_size, num_rounds))
        n_full_blocks = num_rounds // effective_block
        remainder = num_rounds % effective_block

        for i in range(self.num_simulations):
            # Split into blocks
            blocks: List[np.ndarray] = []
            for b in range(n_full_blocks):
                start = b * effective_block
                end = start + effective_block
                blocks.append(data[start:end, :])

            if remainder > 0:
                blocks.append(data[n_full_blocks * effective_block:, :])

            # Permute full blocks; keep partial block at end
            if remainder > 0:
                full_blocks = blocks[:-1]
                partial_block = blocks[-1]
                perm_idx = self.rng.permutation(len(full_blocks))
                permuted_blocks = [full_blocks[j] for j in perm_idx]
                permuted_blocks.append(partial_block)
            else:
                perm_idx = self.rng.permutation(len(blocks))
                permuted_blocks = [blocks[j] for j in perm_idx]

            permuted_data = np.concatenate(permuted_blocks, axis=0)
            null_stats[i] = statistic_func(permuted_data)

        return null_stats

    def null_distribution_summary(self, null_dist: np.ndarray,
                                  observed: float) -> Dict[str, Any]:
        """Summary statistics of null distribution relative to observed."""
        mean_null = float(np.mean(null_dist))
        std_null = float(np.std(null_dist, ddof=1))
        median_null = float(np.median(null_dist))
        min_null = float(np.min(null_dist))
        max_null = float(np.max(null_dist))

        if std_null > 1e-15:
            z_score = (observed - mean_null) / std_null
        else:
            z_score = 0.0

        p_greater = self.monte_carlo_p_value(observed, null_dist, "greater")
        p_less = self.monte_carlo_p_value(observed, null_dist, "less")
        p_two_sided = self.monte_carlo_p_value(observed, null_dist, "two-sided")

        percentile_rank = float(np.mean(null_dist <= observed) * 100)

        # Quantiles of the null distribution
        quantiles = {
            "q01": float(np.percentile(null_dist, 1)),
            "q05": float(np.percentile(null_dist, 5)),
            "q10": float(np.percentile(null_dist, 10)),
            "q25": float(np.percentile(null_dist, 25)),
            "q50": float(np.percentile(null_dist, 50)),
            "q75": float(np.percentile(null_dist, 75)),
            "q90": float(np.percentile(null_dist, 90)),
            "q95": float(np.percentile(null_dist, 95)),
            "q99": float(np.percentile(null_dist, 99)),
        }

        # Normality test on the null distribution
        if len(null_dist) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(
                null_dist if len(null_dist) <= 5000
                else self.rng.choice(null_dist, 5000, replace=False)
            )
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        return {
            "mean": mean_null,
            "std": std_null,
            "median": median_null,
            "min": min_null,
            "max": max_null,
            "z_score": z_score,
            "observed": observed,
            "p_value_greater": p_greater,
            "p_value_less": p_less,
            "p_value_two_sided": p_two_sided,
            "percentile_rank": percentile_rank,
            "quantiles": quantiles,
            "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else None,
            "null_is_normal": bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else None,
            "num_simulations": len(null_dist),
        }
