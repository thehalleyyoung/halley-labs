"""
Power analysis for sequential causal invariance tests.

Provides tools for computing minimum sample sizes, effect sizes, power curves,
and simulation-based power estimation for HSIC-based invariance tests.

References:
    - Gretton et al. (2012). A kernel two-sample test.
    - Pfister et al. (2018). Kernel-based tests for joint independence.
    - Howard et al. (2021). Time-uniform confidence sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, special, stats


@dataclass
class HSICEffectSize:
    """HSIC-based effect size for invariance testing.

    Encapsulates the computation of effect sizes based on the Hilbert-Schmidt
    Independence Criterion for measuring dependence between causal mechanisms
    and regime indicators.

    Attributes:
        hsic_value: Estimated HSIC value.
        normalized_hsic: HSIC normalized by marginal MMDs.
        dimension: Dimensionality of the data.
        kernel_bandwidth: Kernel bandwidth used.
        n_samples: Number of samples used for estimation.
    """
    hsic_value: float
    normalized_hsic: float
    dimension: int
    kernel_bandwidth: float
    n_samples: int

    @property
    def cohen_d_equivalent(self) -> float:
        """Convert HSIC effect size to Cohen's d equivalent.

        Uses the relationship between HSIC and the L2 distance between
        conditional distributions to derive an equivalent Cohen's d.
        """
        return float(np.sqrt(2 * max(self.normalized_hsic, 0.0)))

    @property
    def is_small(self) -> bool:
        return self.cohen_d_equivalent < 0.2

    @property
    def is_medium(self) -> bool:
        return 0.2 <= self.cohen_d_equivalent < 0.8

    @property
    def is_large(self) -> bool:
        return self.cohen_d_equivalent >= 0.8


@dataclass
class PowerCurve:
    """Power curve data for visualization and analysis.

    Attributes:
        sample_sizes: Array of sample sizes.
        powers: Array of power values at each sample size.
        effect_size: Effect size used.
        alpha: Significance level.
        n_regimes: Number of regimes.
        method: Method used for power computation.
    """
    sample_sizes: NDArray[np.int64]
    powers: NDArray[np.float64]
    effect_size: float
    alpha: float
    n_regimes: int
    method: str

    def required_n_for_power(self, target_power: float = 0.8) -> int:
        """Find the minimum sample size for a given target power.

        Uses linear interpolation between computed power values.
        """
        if np.all(self.powers < target_power):
            return int(self.sample_sizes[-1] * 2)
        if np.all(self.powers >= target_power):
            return int(self.sample_sizes[0])

        idx = np.searchsorted(self.powers, target_power)
        if idx == 0:
            return int(self.sample_sizes[0])

        # Linear interpolation
        p_low = self.powers[idx - 1]
        p_high = self.powers[idx]
        n_low = self.sample_sizes[idx - 1]
        n_high = self.sample_sizes[idx]

        frac = (target_power - p_low) / (p_high - p_low + 1e-10)
        n_interp = n_low + frac * (n_high - n_low)
        return int(np.ceil(n_interp))


class SampleSizeCalculator:
    """Calculator for minimum sample sizes in sequential invariance tests.

    Accounts for the sequential nature of the test, the number of regimes,
    the conditioning set dimension, and the desired power/alpha levels.

    Args:
        alpha: Significance level (Type I error rate).
        power: Desired statistical power (1 - Type II error rate).
        n_regimes: Number of regimes to compare.
        sequential_overhead: Overhead factor for sequential testing vs fixed-n.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.8,
        n_regimes: int = 2,
        sequential_overhead: float = 1.3,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"power must be in (0,1), got {power}")

        self.alpha = alpha
        self.power = power
        self.n_regimes = n_regimes
        self.sequential_overhead = sequential_overhead

    def for_mean_shift(
        self,
        effect_size: float,
        variance: float = 1.0,
    ) -> int:
        """Sample size for detecting a mean shift between regimes.

        Uses the formula: n = (z_{1-alpha} + z_{power})^2 * 2 * sigma^2 / delta^2
        with sequential overhead adjustment.

        Args:
            effect_size: Expected mean difference (delta).
            variance: Population variance.

        Returns:
            Minimum sample size per regime.
        """
        if abs(effect_size) < 1e-10:
            return int(1e6)

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_power = stats.norm.ppf(self.power)

        # Two-sample z-test formula
        n_fixed = 2 * variance * (z_alpha + z_power) ** 2 / effect_size ** 2

        # Adjust for multiple regimes (Bonferroni-like)
        n_pairs = self.n_regimes * (self.n_regimes - 1) / 2
        alpha_adj = self.alpha / n_pairs
        z_alpha_adj = stats.norm.ppf(1 - alpha_adj / 2)
        n_adjusted = 2 * variance * (z_alpha_adj + z_power) ** 2 / effect_size ** 2

        # Sequential overhead
        n_seq = n_adjusted * self.sequential_overhead

        return int(np.ceil(n_seq))

    def for_hsic(
        self,
        effect_size: float,
        dimension: int = 1,
        kernel_bandwidth: float = 1.0,
    ) -> int:
        """Sample size for HSIC-based invariance test.

        The HSIC test statistic has a gamma-like distribution under the null,
        and the power depends on the dimension of the conditioning set.

        Args:
            effect_size: HSIC effect size (normalized).
            dimension: Dimension of the conditioning set.
            kernel_bandwidth: Kernel bandwidth.

        Returns:
            Minimum sample size per regime.
        """
        if effect_size < 1e-10:
            return int(1e6)

        # HSIC test has effective df ~ n * d
        # Power approximation: 1 - Phi(z_alpha - sqrt(n) * delta / sigma_hsic)
        # where sigma_hsic ~ 1/sqrt(n * d)

        z_alpha = stats.norm.ppf(1 - self.alpha)
        z_power = stats.norm.ppf(self.power)

        # HSIC variance under null scales as 1/(n^2 * d)
        # Signal scales as effect_size
        sigma_factor = 1.0 / np.sqrt(max(dimension, 1))

        n_fixed = ((z_alpha + z_power) * sigma_factor / effect_size) ** 2
        n_seq = n_fixed * self.sequential_overhead

        # Dimension penalty
        n_dim = n_seq * (1 + 0.1 * dimension)

        return int(np.ceil(n_dim))

    def for_e_value(
        self,
        expected_log_e: float,
        variance_log_e: float = 1.0,
    ) -> int:
        """Sample size for e-value based test.

        The e-value test rejects when E_t >= 1/alpha. Under the alternative,
        log(E_t) grows as t * E[log(e_s)].

        Args:
            expected_log_e: Expected log e-value per observation under alt.
            variance_log_e: Variance of log e-value per observation.

        Returns:
            Minimum sample size.
        """
        if expected_log_e <= 0:
            return int(1e6)

        # E_t = prod e_s, so log(E_t) = sum log(e_s)
        # Need: sum log(e_s) >= log(1/alpha)
        # Under alt: E[log(e_s)] = expected_log_e
        # Var[log(e_s)] = variance_log_e
        # By CLT: P(sum > log(1/alpha)) ~ P(Z > (log(1/alpha) - n*mu) / sqrt(n*var))

        target = np.log(1.0 / self.alpha)
        z_power = stats.norm.ppf(self.power)

        # Solve: (target - n * mu) / sqrt(n * var) = -z_power
        # n * mu - z_power * sqrt(n * var) = target
        # Let u = sqrt(n): mu * u^2 - z_power * sqrt(var) * u - target = 0

        a = expected_log_e
        b = -z_power * np.sqrt(variance_log_e)
        c = -target

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return int(1e6)

        u = (-b + np.sqrt(discriminant)) / (2 * a)
        n = u ** 2

        return int(np.ceil(n * self.sequential_overhead))


class PowerAnalyzer:
    """Comprehensive power analysis for causal invariance testing.

    Provides methods for computing power, generating power curves,
    running simulation-based power estimation, and computing effect sizes.

    Args:
        alpha: Significance level.
        n_regimes: Number of regimes.
        n_simulations: Number of simulations for Monte Carlo estimation.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_regimes: int = 2,
        n_simulations: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.n_regimes = n_regimes
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_state)
        self._calculator = SampleSizeCalculator(
            alpha=alpha, n_regimes=n_regimes
        )

    def compute_hsic_effect_size(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        regimes: NDArray[np.int64],
        kernel_bandwidth: Optional[float] = None,
    ) -> HSICEffectSize:
        """Compute the HSIC-based effect size from data.

        Estimates the Hilbert-Schmidt Independence Criterion between the
        causal mechanism residuals and regime indicators.

        Args:
            X: Input features of shape (n_samples, n_features).
            Y: Target variable of shape (n_samples,).
            regimes: Regime labels of shape (n_samples,).
            kernel_bandwidth: Bandwidth for Gaussian kernel (median heuristic if None).

        Returns:
            HSICEffectSize with estimated effect size.
        """
        n = len(X)
        if n < 4:
            return HSICEffectSize(
                hsic_value=0.0,
                normalized_hsic=0.0,
                dimension=X.shape[1] if X.ndim > 1 else 1,
                kernel_bandwidth=1.0,
                n_samples=n,
            )

        # Kernel bandwidth via median heuristic
        if kernel_bandwidth is None:
            pairwise_sq = np.sum(
                (X[:, None] - X[None, :]) ** 2, axis=-1
            ) if X.ndim > 1 else (X[:, None] - X[None, :]) ** 2
            median_dist = np.median(np.sqrt(pairwise_sq[pairwise_sq > 0]))
            kernel_bandwidth = max(float(median_dist), 1e-10)

        bw = kernel_bandwidth

        # Gram matrices
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        K_X = np.exp(-np.sum((X[:, None] - X[None, :]) ** 2, axis=-1) / (2 * bw ** 2))

        # Regime kernel (delta kernel for categorical)
        K_R = (regimes[:, None] == regimes[None, :]).astype(np.float64)

        # Residual kernel
        Y_arr = Y.ravel()
        K_Y_sq = (Y_arr[:, None] - Y_arr[None, :]) ** 2
        bw_y = max(float(np.median(np.sqrt(K_Y_sq[K_Y_sq > 0]))), 1e-10) if np.any(K_Y_sq > 0) else 1.0
        K_Y = np.exp(-K_Y_sq / (2 * bw_y ** 2))

        # Centering matrices
        H = np.eye(n) - np.ones((n, n)) / n

        # HSIC between residual pattern and regime
        # HSIC(Y, R | X) ≈ tr(K_Y H K_R H) / (n-1)^2
        # Simplified: test if the Y|X relationship changes across regimes
        K_resid = K_Y * K_X  # element-wise product captures residual structure
        K_resid_c = H @ K_resid @ H
        K_R_c = H @ K_R @ H

        hsic = float(np.trace(K_resid_c @ K_R_c)) / ((n - 1) ** 2)
        hsic = max(hsic, 0.0)

        # Normalize by marginal MMDs
        mmd_resid = float(np.trace(K_resid_c @ K_resid_c)) / ((n - 1) ** 2)
        mmd_regime = float(np.trace(K_R_c @ K_R_c)) / ((n - 1) ** 2)
        normalizer = np.sqrt(max(mmd_resid * mmd_regime, 1e-20))

        normalized = hsic / normalizer if normalizer > 1e-10 else 0.0

        dim = X.shape[1]
        return HSICEffectSize(
            hsic_value=hsic,
            normalized_hsic=normalized,
            dimension=dim,
            kernel_bandwidth=bw,
            n_samples=n,
        )

    def power_for_sample_size(
        self,
        n_per_regime: int,
        effect_size: float,
        dimension: int = 1,
        method: str = "asymptotic",
    ) -> float:
        """Compute power for a given sample size and effect size.

        Args:
            n_per_regime: Sample size per regime.
            effect_size: Cohen's d or equivalent effect size.
            dimension: Dimension of the conditioning set.
            method: "asymptotic" or "simulation".

        Returns:
            Estimated power.
        """
        if method == "asymptotic":
            return self._asymptotic_power(n_per_regime, effect_size, dimension)
        elif method == "simulation":
            return self._simulation_power(n_per_regime, effect_size, dimension)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _asymptotic_power(
        self,
        n: int,
        effect_size: float,
        dimension: int,
    ) -> float:
        """Asymptotic power approximation.

        Uses the non-central chi-squared distribution for the asymptotic
        distribution of the test statistic under the alternative.
        """
        if n < 2 or effect_size < 1e-10:
            return self.alpha

        # For e-value based test with GROW martingale:
        # Power ≈ P(chi2_ncp >= threshold)
        # Non-centrality parameter: ncp = n * delta^2 / sigma^2
        ncp = n * effect_size ** 2

        # Degrees of freedom (accounts for conditioning set dimension)
        df = max(dimension, 1)

        # Threshold from e-value: E_t >= 1/alpha
        # Under null, E_t ~ exp(chi2(df)/2 - df/2)
        # Threshold in chi2 scale:
        threshold = 2 * np.log(1.0 / self.alpha) + df

        # Power = P(chi2(df, ncp) >= threshold)
        power = 1.0 - stats.ncx2.cdf(threshold, df=df, nc=ncp)

        # Dimension penalty: power decreases with dimension
        dim_factor = np.exp(-0.05 * max(dimension - 1, 0))
        power *= dim_factor

        # Sequential overhead reduces power slightly
        power *= 0.95  # approximate 5% loss from sequential boundary

        return float(np.clip(power, 0.0, 1.0))

    def _simulation_power(
        self,
        n: int,
        effect_size: float,
        dimension: int,
    ) -> float:
        """Simulation-based power estimation.

        Runs Monte Carlo simulations of the full sequential test procedure
        under the alternative hypothesis.
        """
        rejections = 0

        for sim in range(self.n_simulations):
            # Generate data under alternative
            data_per_regime = []
            for r in range(self.n_regimes):
                mean_shift = effect_size * r / (self.n_regimes - 1) if self.n_regimes > 1 else 0
                X_r = self.rng.randn(n, max(dimension, 1)) + mean_shift
                data_per_regime.append(X_r)

            # Run sequential e-value test
            log_wealth = 0.0
            rejected = False

            all_data = np.vstack(data_per_regime)
            all_regimes = np.concatenate([
                np.full(n, r) for r in range(self.n_regimes)
            ])

            # Shuffle to simulate sequential arrival
            perm = self.rng.permutation(len(all_data))
            all_data = all_data[perm]
            all_regimes = all_regimes[perm]

            # Simple likelihood ratio e-values
            regime_stats: Dict[int, Tuple[NDArray, NDArray, int]] = {}
            global_sum = np.zeros(max(dimension, 1))
            global_sum_sq = np.zeros(max(dimension, 1))
            global_n = 0

            for t in range(len(all_data)):
                x = all_data[t]
                r = int(all_regimes[t])

                global_sum += x
                global_sum_sq += x ** 2
                global_n += 1

                if r not in regime_stats:
                    regime_stats[r] = (np.zeros(max(dimension, 1)), np.zeros(max(dimension, 1)), 0)
                s, sq, cnt = regime_stats[r]
                regime_stats[r] = (s + x, sq + x ** 2, cnt + 1)

                # Need at least 5 per regime
                ready = [
                    rr for rr, (_, _, c) in regime_stats.items()
                    if c >= 5
                ]
                if len(ready) < 2:
                    continue

                # Compute LR e-value
                mu_global = global_sum / global_n
                var_global = np.maximum(global_sum_sq / global_n - mu_global ** 2, 1e-10)
                std_global = np.sqrt(var_global)

                s_r, sq_r, n_r = regime_stats[r]
                mu_r = s_r / n_r
                var_r = np.maximum(sq_r / n_r - mu_r ** 2, 1e-10)
                std_r = np.sqrt(var_r)

                log_alt = np.sum(stats.norm.logpdf(x, loc=mu_r, scale=std_r + 1e-8))
                log_null = np.sum(stats.norm.logpdf(x, loc=mu_global, scale=std_global + 1e-8))
                log_e = np.clip(log_alt - log_null, -20, 20)

                log_wealth += log_e

                if log_wealth >= np.log(1.0 / self.alpha):
                    rejected = True
                    break

            if rejected:
                rejections += 1

        return rejections / self.n_simulations

    def power_curve(
        self,
        effect_size: float,
        sample_sizes: Optional[Sequence[int]] = None,
        dimension: int = 1,
        method: str = "asymptotic",
    ) -> PowerCurve:
        """Generate a power curve over a range of sample sizes.

        Args:
            effect_size: Effect size to test.
            sample_sizes: Array of sample sizes. If None, auto-generated.
            dimension: Conditioning set dimension.
            method: Power computation method.

        Returns:
            PowerCurve object.
        """
        if sample_sizes is None:
            # Auto-generate reasonable range
            n_min = max(10, 2 * dimension)
            n_approx = self._calculator.for_mean_shift(effect_size)
            n_max = min(n_approx * 3, 10000)
            n_max = max(n_max, n_min * 10)
            sample_sizes = np.unique(np.geomspace(
                n_min, n_max, num=30, dtype=int
            ))

        sizes = np.asarray(sample_sizes, dtype=np.int64)
        powers = np.array([
            self.power_for_sample_size(int(n), effect_size, dimension, method)
            for n in sizes
        ])

        return PowerCurve(
            sample_sizes=sizes,
            powers=powers,
            effect_size=effect_size,
            alpha=self.alpha,
            n_regimes=self.n_regimes,
            method=method,
        )

    def power_vs_dimension(
        self,
        effect_size: float,
        n_per_regime: int,
        dimensions: Optional[Sequence[int]] = None,
        method: str = "asymptotic",
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Compute power as a function of conditioning set dimension.

        Args:
            effect_size: Effect size.
            n_per_regime: Sample size per regime.
            dimensions: Array of dimensions to evaluate.
            method: Power computation method.

        Returns:
            Tuple of (dimensions array, powers array).
        """
        if dimensions is None:
            dimensions = np.arange(1, 21)

        dims = np.asarray(dimensions, dtype=np.int64)
        powers = np.array([
            self.power_for_sample_size(n_per_regime, effect_size, int(d), method)
            for d in dims
        ])

        return dims, powers

    def minimum_sample_size(
        self,
        effect_size: float,
        target_power: float = 0.8,
        dimension: int = 1,
        method: str = "asymptotic",
    ) -> int:
        """Find minimum sample size per regime for desired power.

        Uses binary search over sample sizes.

        Args:
            effect_size: Expected effect size.
            target_power: Desired power (default 0.8).
            dimension: Conditioning set dimension.
            method: Power computation method.

        Returns:
            Minimum sample size per regime.
        """
        if effect_size < 1e-10:
            return int(1e6)

        # Binary search
        lo, hi = 2, 100000
        while lo < hi:
            mid = (lo + hi) // 2
            pwr = self.power_for_sample_size(mid, effect_size, dimension, method)
            if pwr >= target_power:
                hi = mid
            else:
                lo = mid + 1

        return lo

    def power_heatmap(
        self,
        effect_sizes: Sequence[float],
        sample_sizes: Sequence[int],
        dimension: int = 1,
        method: str = "asymptotic",
    ) -> NDArray[np.float64]:
        """Compute power grid over effect sizes and sample sizes.

        Args:
            effect_sizes: Array of effect sizes.
            sample_sizes: Array of sample sizes.
            dimension: Conditioning set dimension.
            method: Power computation method.

        Returns:
            2D array of power values (n_effects x n_samples).
        """
        effects = np.asarray(effect_sizes)
        sizes = np.asarray(sample_sizes, dtype=int)

        grid = np.zeros((len(effects), len(sizes)))
        for i, delta in enumerate(effects):
            for j, n in enumerate(sizes):
                grid[i, j] = self.power_for_sample_size(
                    int(n), float(delta), dimension, method
                )

        return grid

    def expected_stopping_time(
        self,
        effect_size: float,
        dimension: int = 1,
        method: str = "asymptotic",
    ) -> float:
        """Estimate expected stopping time of the sequential test.

        Under the alternative with effect size delta, the expected stopping
        time is approximately log(1/alpha) / KL(P_1 || P_0).

        Args:
            effect_size: Effect size under alternative.
            dimension: Conditioning set dimension.
            method: Estimation method.

        Returns:
            Expected number of observations until rejection.
        """
        if effect_size < 1e-10:
            return float("inf")

        if method == "asymptotic":
            # KL divergence for Gaussian mean shift
            kl_div = 0.5 * effect_size ** 2

            # Expected stopping time ≈ log(1/alpha) / KL
            est = np.log(1.0 / self.alpha) / kl_div

            # Dimension factor
            est *= (1 + 0.1 * max(dimension - 1, 0))

            # Regime factor
            est *= self.n_regimes / 2.0

            return float(est)

        elif method == "simulation":
            stopping_times = []
            for sim in range(min(self.n_simulations, 200)):
                n_max = 10000
                log_wealth = 0.0
                stopped = False

                for t in range(1, n_max + 1):
                    regime = self.rng.randint(0, self.n_regimes)
                    mean_shift = effect_size * regime / max(self.n_regimes - 1, 1)
                    x = self.rng.randn(max(dimension, 1)) + mean_shift

                    z = float(np.sum(x ** 2))
                    log_e = 0.5 * (z - dimension) * effect_size ** 2 / (
                        dimension + effect_size ** 2
                    )
                    log_e = np.clip(log_e, -10, 10)
                    log_wealth += log_e

                    if log_wealth >= np.log(1.0 / self.alpha):
                        stopping_times.append(t)
                        stopped = True
                        break

                if not stopped:
                    stopping_times.append(n_max)

            return float(np.mean(stopping_times))

        return float("inf")

    def sensitivity_analysis(
        self,
        n_per_regime: int,
        dimension: int = 1,
        effect_range: Tuple[float, float] = (0.05, 2.0),
        n_points: int = 50,
    ) -> Dict[str, NDArray[np.float64]]:
        """Run sensitivity analysis varying effect size and alpha.

        Args:
            n_per_regime: Sample size per regime.
            dimension: Conditioning set dimension.
            effect_range: Range of effect sizes to evaluate.
            n_points: Number of evaluation points.

        Returns:
            Dictionary with 'effect_sizes', 'powers', 'stopping_times'.
        """
        effects = np.linspace(effect_range[0], effect_range[1], n_points)
        powers = np.zeros(n_points)
        stopping_times = np.zeros(n_points)

        for i, delta in enumerate(effects):
            powers[i] = self._asymptotic_power(n_per_regime, delta, dimension)
            # Approximate stopping time
            kl = 0.5 * delta ** 2
            if kl > 1e-10:
                stopping_times[i] = np.log(1.0 / self.alpha) / kl
            else:
                stopping_times[i] = float("inf")

        return {
            "effect_sizes": effects,
            "powers": powers,
            "stopping_times": stopping_times,
        }

    def compare_methods(
        self,
        n_per_regime: int,
        effect_size: float,
        dimension: int = 1,
    ) -> Dict[str, float]:
        """Compare power of different testing methods.

        Args:
            n_per_regime: Sample size per regime.
            effect_size: Effect size.
            dimension: Conditioning set dimension.

        Returns:
            Dictionary mapping method name to estimated power.
        """
        results = {}

        # Standard fixed-sample test (z-test)
        ncp = n_per_regime * effect_size ** 2
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        results["fixed_z_test"] = float(
            1 - stats.norm.cdf(z_alpha - np.sqrt(ncp))
            + stats.norm.cdf(-z_alpha - np.sqrt(ncp))
        )

        # Sequential e-value test
        results["sequential_e_value"] = self._asymptotic_power(
            n_per_regime, effect_size, dimension
        )

        # HSIC-based test (nonparametric)
        hsic_power = self._asymptotic_power(
            n_per_regime, effect_size * 0.8, dimension  # HSIC slightly less efficient
        )
        results["hsic_test"] = hsic_power

        # Permutation test
        perm_power = self._asymptotic_power(
            n_per_regime, effect_size * 0.9, dimension
        )
        results["permutation_test"] = perm_power

        return results
