"""Bootstrap methods for statistical inference in collusion detection.

Provides nonparametric, block, parametric, BCa, studentized, wild,
and double bootstrap methods with confidence intervals and p-values.
"""

import numpy as np
from scipy import stats
from typing import Callable, Optional, Tuple, List, Dict, Any


class BootstrapEngine:
    """Comprehensive bootstrap inference engine."""

    def __init__(self, n_bootstrap: int = 10000, random_state: Optional[int] = None,
                 confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)
        self.confidence_level = confidence_level

    def nonparametric(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                      confidence: Optional[float] = None) -> Dict[str, Any]:
        """Standard nonparametric bootstrap.

        Args:
            data: Input data array
            statistic: Function that computes a scalar statistic from data
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            Dict with keys: point_estimate, ci_lower, ci_upper, std_error,
                           bootstrap_distribution, confidence_level
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_bootstrap_indices(n)
            boot_stats[i] = statistic(data[idx])

        ci_lower, ci_upper = self.percentile_ci(boot_stats, conf)
        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_stats, ddof=1)),
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "bias": float(np.mean(boot_stats) - observed),
            "n_bootstrap": self.n_bootstrap,
        }

    def block_bootstrap(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                        block_size: Optional[int] = None,
                        confidence: Optional[float] = None) -> Dict[str, Any]:
        """Moving block bootstrap for time-series/dependent data.

        Args:
            data: Time series data
            statistic: Statistic function
            block_size: Block size (auto-computed if None)
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)

        if block_size is None:
            block_size = self.optimal_block_size(data)
        block_size = max(1, min(block_size, n))

        observed = float(statistic(data))

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_block_indices(n, block_size)
            boot_stats[i] = statistic(data[idx])

        ci_lower, ci_upper = self.percentile_ci(boot_stats, conf)
        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_stats, ddof=1)),
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "block_size": block_size,
            "bias": float(np.mean(boot_stats) - observed),
        }

    def parametric(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                   distribution: str = "normal",
                   confidence: Optional[float] = None) -> Dict[str, Any]:
        """Parametric bootstrap assuming a distribution.

        Fits distribution to data, generates samples from fitted distribution.
        Supported distributions: normal, lognormal, exponential, poisson, t.
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data, dtype=float)
        n = len(data)
        observed = float(statistic(data))

        generators = {
            "normal": self._gen_normal,
            "lognormal": self._gen_lognormal,
            "exponential": self._gen_exponential,
            "poisson": self._gen_poisson,
            "t": self._gen_t,
        }
        if distribution not in generators:
            raise ValueError(f"Unsupported distribution '{distribution}'. "
                             f"Choose from {list(generators.keys())}")

        gen_func = generators[distribution]

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            synthetic = gen_func(data, n)
            boot_stats[i] = statistic(synthetic)

        ci_lower, ci_upper = self.percentile_ci(boot_stats, conf)
        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_stats, ddof=1)),
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "distribution": distribution,
            "bias": float(np.mean(boot_stats) - observed),
        }

    def _gen_normal(self, data: np.ndarray, n: int) -> np.ndarray:
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        return self.rng.normal(mu, max(sigma, 1e-15), size=n)

    def _gen_lognormal(self, data: np.ndarray, n: int) -> np.ndarray:
        positive = data[data > 0]
        if len(positive) < 2:
            return np.abs(self.rng.normal(np.mean(data), max(np.std(data, ddof=1), 1e-15), size=n))
        log_data = np.log(positive)
        mu, sigma = np.mean(log_data), np.std(log_data, ddof=1)
        return self.rng.lognormal(mu, max(sigma, 1e-15), size=n)

    def _gen_exponential(self, data: np.ndarray, n: int) -> np.ndarray:
        scale = max(np.mean(data), 1e-15)
        return self.rng.exponential(scale, size=n)

    def _gen_poisson(self, data: np.ndarray, n: int) -> np.ndarray:
        lam = max(np.mean(data), 1e-15)
        return self.rng.poisson(lam, size=n).astype(float)

    def _gen_t(self, data: np.ndarray, n: int) -> np.ndarray:
        df, loc, scale = stats.t.fit(data)
        return stats.t.rvs(df, loc=loc, scale=scale, size=n, random_state=self.rng)

    def bca(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
            confidence: Optional[float] = None) -> Dict[str, Any]:
        """BCa (bias-corrected and accelerated) confidence interval.

        Corrects for bias and skewness in the bootstrap distribution.
        Uses jackknife for acceleration estimate.
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_bootstrap_indices(n)
            boot_stats[i] = statistic(data[idx])

        ci_lower, ci_upper = self.bca_ci(data, statistic, boot_stats, conf)

        jackknife_stats = np.empty(n)
        for i in range(n):
            jackknife_stats[i] = statistic(np.delete(data, i))
        jack_mean = np.mean(jackknife_stats)
        diffs = jack_mean - jackknife_stats
        a = float(np.sum(diffs ** 3) / (6.0 * (np.sum(diffs ** 2)) ** 1.5 + 1e-15))

        prop_below = np.mean(boot_stats < observed)
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        z0 = float(stats.norm.ppf(prop_below))

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_stats, ddof=1)),
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "bias_correction": z0,
            "acceleration": a,
            "bias": float(np.mean(boot_stats) - observed),
        }

    def percentile_ci(self, bootstrap_dist: np.ndarray,
                      confidence: float = 0.95) -> Tuple[float, float]:
        """Simple percentile confidence interval."""
        alpha = 1.0 - confidence
        return (float(np.percentile(bootstrap_dist, 100 * alpha / 2)),
                float(np.percentile(bootstrap_dist, 100 * (1 - alpha / 2))))

    def basic_ci(self, observed: float, bootstrap_dist: np.ndarray,
                 confidence: float = 0.95) -> Tuple[float, float]:
        """Basic (reverse percentile) CI: 2*observed - percentile."""
        alpha = 1.0 - confidence
        upper_pct = np.percentile(bootstrap_dist, 100 * alpha / 2)
        lower_pct = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2))
        return (float(2 * observed - lower_pct), float(2 * observed - upper_pct))

    def bca_ci(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
               bootstrap_dist: np.ndarray,
               confidence: float = 0.95) -> Tuple[float, float]:
        """Full BCa CI computation with jackknife acceleration."""
        observed = statistic(data)
        n = len(data)

        prop_below = np.mean(bootstrap_dist < observed)
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        z0 = stats.norm.ppf(prop_below)

        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)
        jack_mean = np.mean(jackknife_stats)
        diffs = jack_mean - jackknife_stats
        denom = 6.0 * (np.sum(diffs ** 2)) ** 1.5
        a = np.sum(diffs ** 3) / (denom + 1e-15)

        alpha = 1.0 - confidence
        z_lower = stats.norm.ppf(alpha / 2)
        z_upper = stats.norm.ppf(1 - alpha / 2)

        def adjusted_percentile(z):
            numer = z0 + z
            denom_adj = 1.0 - a * numer
            if abs(denom_adj) < 1e-15:
                denom_adj = 1e-15 * np.sign(denom_adj) if denom_adj != 0 else 1e-15
            return stats.norm.cdf(z0 + numer / denom_adj)

        p_lower = adjusted_percentile(z_lower)
        p_upper = adjusted_percentile(z_upper)

        p_lower = np.clip(p_lower, 0.001, 0.999)
        p_upper = np.clip(p_upper, 0.001, 0.999)

        return (float(np.percentile(bootstrap_dist, 100 * p_lower)),
                float(np.percentile(bootstrap_dist, 100 * p_upper)))

    def bootstrap_p_value(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                          null_value: float = 0.0,
                          alternative: str = "two-sided") -> float:
        """Bootstrap p-value for testing H0: statistic(data) = null_value.

        Recenters bootstrap distribution around null_value by shifting
        the data so the statistic equals null_value under H0.
        """
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))

        shift = observed - null_value
        centered_data = data - shift

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_bootstrap_indices(n)
            boot_stats[i] = statistic(centered_data[idx])

        if alternative == "two-sided":
            p_val = float(np.mean(np.abs(boot_stats - null_value) >= abs(observed - null_value)))
        elif alternative == "greater":
            p_val = float(np.mean(boot_stats >= observed))
        elif alternative == "less":
            p_val = float(np.mean(boot_stats <= observed))
        else:
            raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got '{alternative}'")

        return max(p_val, 1.0 / (self.n_bootstrap + 1))

    def studentized_bootstrap(self, data: np.ndarray,
                              statistic: Callable[[np.ndarray], float],
                              se_func: Callable[[np.ndarray], float],
                              confidence: Optional[float] = None) -> Dict[str, Any]:
        """Studentized (bootstrap-t) interval.

        Uses the distribution of (statistic - observed)/SE to construct intervals.
        More accurate coverage than percentile methods for smooth statistics.
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))
        observed_se = float(se_func(data))

        t_stats = np.empty(self.n_bootstrap)
        boot_estimates = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_bootstrap_indices(n)
            boot_sample = data[idx]
            boot_stat = statistic(boot_sample)
            boot_se = se_func(boot_sample)
            boot_estimates[i] = boot_stat
            if boot_se > 1e-15:
                t_stats[i] = (boot_stat - observed) / boot_se
            else:
                t_stats[i] = 0.0

        alpha = 1.0 - conf
        t_lower = np.percentile(t_stats, 100 * alpha / 2)
        t_upper = np.percentile(t_stats, 100 * (1 - alpha / 2))

        ci_lower = observed - t_upper * observed_se
        ci_upper = observed - t_lower * observed_se

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(observed_se),
            "bootstrap_distribution": boot_estimates,
            "t_distribution": t_stats,
            "confidence_level": conf,
            "bias": float(np.mean(boot_estimates) - observed),
        }

    def wild_bootstrap(self, residuals: np.ndarray, X: np.ndarray, y: np.ndarray,
                       confidence: Optional[float] = None) -> Dict[str, Any]:
        """Wild bootstrap for regression with heteroscedastic errors.

        Multiplies residuals by Rademacher random variables (±1 with equal
        probability) to form bootstrap responses, then re-estimates coefficients.
        """
        conf = confidence if confidence is not None else self.confidence_level
        residuals = np.asarray(residuals, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        XtX_inv = np.linalg.pinv(X.T @ X)
        beta_hat = XtX_inv @ X.T @ y
        y_hat = X @ beta_hat
        p = X.shape[1]

        boot_coefs = np.empty((self.n_bootstrap, p))
        for i in range(self.n_bootstrap):
            rademacher = self.rng.choice([-1.0, 1.0], size=n)
            y_star = y_hat + residuals * rademacher
            boot_coefs[i] = XtX_inv @ X.T @ y_star

        ci_lower = np.empty(p)
        ci_upper = np.empty(p)
        alpha = 1.0 - conf
        for j in range(p):
            ci_lower[j] = np.percentile(boot_coefs[:, j], 100 * alpha / 2)
            ci_upper[j] = np.percentile(boot_coefs[:, j], 100 * (1 - alpha / 2))

        return {
            "point_estimate": beta_hat.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "std_error": np.std(boot_coefs, axis=0, ddof=1).tolist(),
            "bootstrap_coefficients": boot_coefs,
            "confidence_level": conf,
            "n_predictors": p,
            "bias": (np.mean(boot_coefs, axis=0) - beta_hat).tolist(),
        }

    def double_bootstrap(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                         n_outer: int = 1000, n_inner: int = 250,
                         confidence: Optional[float] = None) -> Dict[str, Any]:
        """Double (nested) bootstrap for improved coverage accuracy.

        Uses iterated bootstrap to calibrate the confidence level so the
        actual coverage of a percentile interval matches the nominal level.
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))

        outer_stats = np.empty(n_outer)
        for i in range(n_outer):
            idx = self._generate_bootstrap_indices(n)
            outer_stats[i] = statistic(data[idx])

        coverages = np.empty(n_outer)
        for i in range(n_outer):
            idx_outer = self._generate_bootstrap_indices(n)
            outer_sample = data[idx_outer]
            outer_stat = statistic(outer_sample)

            inner_stats = np.empty(n_inner)
            for j in range(n_inner):
                idx_inner = self.rng.randint(0, n, size=n)
                inner_stats[j] = statistic(outer_sample[idx_inner])

            ci_lo, ci_hi = self.percentile_ci(inner_stats, conf)
            coverages[i] = 1.0 if ci_lo <= observed <= ci_hi else 0.0

        actual_coverage = float(np.mean(coverages))

        alpha_nominal = 1.0 - conf
        if actual_coverage > 0.01 and actual_coverage < 0.99:
            calibration_ratio = alpha_nominal / (1.0 - actual_coverage)
            alpha_calibrated = np.clip(alpha_nominal * calibration_ratio, 0.001, 0.5)
        else:
            alpha_calibrated = alpha_nominal

        calibrated_conf = 1.0 - alpha_calibrated
        ci_lower, ci_upper = self.percentile_ci(outer_stats, calibrated_conf)

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(outer_stats, ddof=1)),
            "bootstrap_distribution": outer_stats,
            "confidence_level": conf,
            "calibrated_confidence": float(calibrated_conf),
            "actual_coverage": actual_coverage,
            "n_outer": n_outer,
            "n_inner": n_inner,
            "bias": float(np.mean(outer_stats) - observed),
        }

    def optimal_block_size(self, data: np.ndarray) -> int:
        """Estimate optimal block size for block bootstrap.

        Uses the method of Politis & White (2004) based on
        the flat-top lag-window estimator.
        """
        n = len(data)
        max_lag = min(int(np.sqrt(n)), n // 3)
        x_centered = data - np.mean(data)
        var = np.var(data)
        if var < 1e-15:
            return 1

        acf = np.correlate(x_centered, x_centered, mode='full')[n - 1:n - 1 + max_lag + 1] / (n * var)

        threshold = 1.96 / np.sqrt(n)
        cutoff = max_lag
        for lag in range(1, max_lag):
            if abs(acf[lag]) < threshold:
                cutoff = lag
                break

        acf_sum = 1.0 + 2.0 * np.sum(np.abs(acf[1:cutoff + 1]))
        block_size = max(1, int(np.ceil((acf_sum ** (2.0 / 3.0)) * (n ** (1.0 / 3.0)))))
        return min(block_size, n // 2)

    def bootstrap_correlation(self, x: np.ndarray, y: np.ndarray,
                              method: str = "pearson",
                              confidence: Optional[float] = None) -> Dict[str, Any]:
        """Bootstrap CI for correlation coefficient.

        Supports pearson, spearman, and kendall correlation methods.
        Resamples paired observations to preserve the dependence structure.
        """
        conf = confidence if confidence is not None else self.confidence_level
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n != len(y):
            raise ValueError("x and y must have the same length")

        corr_funcs = {
            "pearson": lambda a, b: float(np.corrcoef(a, b)[0, 1]),
            "spearman": lambda a, b: float(stats.spearmanr(a, b).correlation),
            "kendall": lambda a, b: float(stats.kendalltau(a, b).correlation),
        }
        if method not in corr_funcs:
            raise ValueError(f"method must be one of {list(corr_funcs.keys())}")

        corr_func = corr_funcs[method]
        observed = corr_func(x, y)

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._generate_bootstrap_indices(n)
            boot_stats[i] = corr_func(x[idx], y[idx])

        valid_mask = np.isfinite(boot_stats)
        boot_valid = boot_stats[valid_mask]
        if len(boot_valid) < 10:
            ci_lower = ci_upper = observed
        else:
            ci_lower, ci_upper = self.percentile_ci(boot_valid, conf)

        return {
            "point_estimate": float(observed),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_valid, ddof=1)) if len(boot_valid) > 1 else 0.0,
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "method": method,
            "n_valid_bootstraps": int(np.sum(valid_mask)),
            "bias": float(np.mean(boot_valid) - observed) if len(boot_valid) > 0 else 0.0,
        }

    def bootstrap_ratio(self, numerator: np.ndarray, denominator: np.ndarray,
                        confidence: Optional[float] = None) -> Dict[str, Any]:
        """Bootstrap CI for a ratio of means (e.g., collusion premium).

        Resamples (numerator, denominator) pairs jointly when lengths match,
        or independently otherwise.
        """
        conf = confidence if confidence is not None else self.confidence_level
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)

        denom_mean = np.mean(denominator)
        if abs(denom_mean) < 1e-15:
            raise ValueError("Denominator mean is effectively zero; ratio undefined")
        observed = float(np.mean(numerator) / denom_mean)

        paired = len(numerator) == len(denominator)
        n_num = len(numerator)
        n_den = len(denominator)

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            if paired:
                idx = self._generate_bootstrap_indices(n_num)
                num_sample = numerator[idx]
                den_sample = denominator[idx]
            else:
                idx_num = self.rng.randint(0, n_num, size=n_num)
                idx_den = self.rng.randint(0, n_den, size=n_den)
                num_sample = numerator[idx_num]
                den_sample = denominator[idx_den]
            den_mean = np.mean(den_sample)
            if abs(den_mean) < 1e-15:
                boot_stats[i] = np.nan
            else:
                boot_stats[i] = np.mean(num_sample) / den_mean

        valid = boot_stats[np.isfinite(boot_stats)]
        if len(valid) < 10:
            ci_lower = ci_upper = observed
        else:
            ci_lower, ci_upper = self.percentile_ci(valid, conf)

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "n_valid_bootstraps": int(len(valid)),
            "bias": float(np.mean(valid) - observed) if len(valid) > 0 else 0.0,
        }

    def bootstrap_difference(self, x: np.ndarray, y: np.ndarray,
                             statistic: Callable[[np.ndarray], float] = np.mean,
                             confidence: Optional[float] = None) -> Dict[str, Any]:
        """Bootstrap CI for difference in statistic between two groups.

        Independently resamples from each group and computes
        statistic(x*) - statistic(y*) for each replicate.
        """
        conf = confidence if confidence is not None else self.confidence_level
        x = np.asarray(x)
        y = np.asarray(y)
        n_x = len(x)
        n_y = len(y)
        observed = float(statistic(x) - statistic(y))

        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx_x = self.rng.randint(0, n_x, size=n_x)
            idx_y = self.rng.randint(0, n_y, size=n_y)
            boot_stats[i] = statistic(x[idx_x]) - statistic(y[idx_y])

        ci_lower, ci_upper = self.percentile_ci(boot_stats, conf)

        p_value_two_sided = float(np.mean(np.abs(boot_stats - np.mean(boot_stats))
                                          >= abs(observed - np.mean(boot_stats))))
        p_value_two_sided = max(p_value_two_sided, 1.0 / (self.n_bootstrap + 1))

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(boot_stats, ddof=1)),
            "bootstrap_distribution": boot_stats,
            "confidence_level": conf,
            "p_value": p_value_two_sided,
            "bias": float(np.mean(boot_stats) - observed),
        }

    def permutation_test(self, x: np.ndarray, y: np.ndarray,
                         statistic: Callable[[np.ndarray], float] = np.mean,
                         n_permutations: Optional[int] = None,
                         alternative: str = "two-sided") -> Dict[str, Any]:
        """Permutation test for comparing two groups.

        Randomly permutes group labels to build a null distribution
        of the test statistic difference.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n_x = len(x)
        combined = np.concatenate([x, y])
        n_total = len(combined)
        n_perm = n_permutations if n_permutations is not None else self.n_bootstrap

        observed_diff = float(statistic(x) - statistic(y))

        perm_stats = np.empty(n_perm)
        for i in range(n_perm):
            perm_idx = self.rng.permutation(n_total)
            perm_x = combined[perm_idx[:n_x]]
            perm_y = combined[perm_idx[n_x:]]
            perm_stats[i] = statistic(perm_x) - statistic(perm_y)

        if alternative == "two-sided":
            p_val = float(np.mean(np.abs(perm_stats) >= abs(observed_diff)))
        elif alternative == "greater":
            p_val = float(np.mean(perm_stats >= observed_diff))
        elif alternative == "less":
            p_val = float(np.mean(perm_stats <= observed_diff))
        else:
            raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less'")

        p_val = max(p_val, 1.0 / (n_perm + 1))

        return {
            "observed_difference": observed_diff,
            "p_value": p_val,
            "null_distribution": perm_stats,
            "n_permutations": n_perm,
            "alternative": alternative,
        }

    def subsampling(self, data: np.ndarray, statistic: Callable[[np.ndarray], float],
                    subsample_size: Optional[int] = None,
                    confidence: Optional[float] = None) -> Dict[str, Any]:
        """Subsampling inference (Politis, Romano & Wolf).

        Draws subsamples of size b < n without replacement to
        approximate the sampling distribution.
        """
        conf = confidence if confidence is not None else self.confidence_level
        data = np.asarray(data)
        n = len(data)
        if subsample_size is None:
            subsample_size = max(2, int(n ** 0.67))
        subsample_size = min(subsample_size, n - 1)

        observed = float(statistic(data))

        n_sub = min(self.n_bootstrap, int(np.math.factorial(min(n, 20))
                    / np.math.factorial(min(n - subsample_size, 20))
                    if n <= 20 else self.n_bootstrap))
        n_sub = max(n_sub, 100)

        sub_stats = np.empty(n_sub)
        scaling = np.sqrt(subsample_size / n)
        for i in range(n_sub):
            idx = self.rng.choice(n, size=subsample_size, replace=False)
            sub_stats[i] = statistic(data[idx])

        scaled_diffs = (sub_stats - observed) / scaling

        alpha = 1.0 - conf
        q_lower = np.percentile(scaled_diffs, 100 * alpha / 2)
        q_upper = np.percentile(scaled_diffs, 100 * (1 - alpha / 2))

        ci_lower = observed + q_lower * scaling
        ci_upper = observed + q_upper * scaling

        return {
            "point_estimate": observed,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(sub_stats, ddof=1)),
            "subsample_distribution": sub_stats,
            "confidence_level": conf,
            "subsample_size": subsample_size,
        }

    def jackknife_estimate(self, data: np.ndarray,
                           statistic: Callable[[np.ndarray], float]) -> Dict[str, Any]:
        """Delete-one jackknife for bias and variance estimation.

        Computes the leave-one-out estimates and derives jackknife
        bias correction and standard error.
        """
        data = np.asarray(data)
        n = len(data)
        observed = float(statistic(data))

        jack_stats = np.empty(n)
        for i in range(n):
            jack_stats[i] = statistic(np.delete(data, i))

        jack_mean = float(np.mean(jack_stats))
        bias = float((n - 1) * (jack_mean - observed))
        bias_corrected = float(observed - bias)

        pseudo_values = n * observed - (n - 1) * jack_stats
        se = float(np.std(pseudo_values, ddof=1) / np.sqrt(n))

        return {
            "point_estimate": observed,
            "bias": bias,
            "bias_corrected_estimate": bias_corrected,
            "std_error": se,
            "jackknife_values": jack_stats,
            "pseudo_values": pseudo_values,
        }

    def moving_block_jackknife(self, data: np.ndarray,
                                statistic: Callable[[np.ndarray], float],
                                block_size: Optional[int] = None) -> Dict[str, Any]:
        """Moving block jackknife for dependent data.

        Deletes one block at a time to estimate variance of the statistic.
        """
        data = np.asarray(data)
        n = len(data)
        if block_size is None:
            block_size = self.optimal_block_size(data)
        block_size = max(1, min(block_size, n))

        observed = float(statistic(data))
        n_blocks = n - block_size + 1

        jack_stats = np.empty(n_blocks)
        for i in range(n_blocks):
            jack_sample = np.concatenate([data[:i], data[i + block_size:]])
            if len(jack_sample) == 0:
                jack_stats[i] = observed
            else:
                jack_stats[i] = statistic(jack_sample)

        jack_mean = float(np.mean(jack_stats))
        h = n / block_size
        se_sq = (h - 1.0) / h * np.sum((jack_stats - jack_mean) ** 2)
        se = float(np.sqrt(max(se_sq, 0.0)))

        return {
            "point_estimate": observed,
            "std_error": se,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "jackknife_values": jack_stats,
        }

    def _generate_bootstrap_indices(self, n: int) -> np.ndarray:
        """Generate bootstrap sample indices."""
        return self.rng.randint(0, n, size=n)

    def _generate_block_indices(self, n: int, block_size: int) -> np.ndarray:
        """Generate block bootstrap indices."""
        n_blocks = int(np.ceil(n / block_size))
        starts = self.rng.randint(0, max(1, n - block_size + 1), size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])
        return indices[:n]

    def summary(self, result: Dict[str, Any]) -> str:
        """Format a bootstrap result dict as a human-readable summary string."""
        lines = []
        lines.append(f"Point estimate : {result['point_estimate']:.6f}")
        lines.append(f"Std error      : {result.get('std_error', float('nan')):.6f}")
        conf = result.get('confidence_level', self.confidence_level)
        lines.append(f"{100 * conf:.1f}% CI        : [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
        if 'bias' in result:
            lines.append(f"Bias           : {result['bias']:.6f}")
        if 'p_value' in result:
            lines.append(f"P-value        : {result['p_value']:.6f}")
        if 'bias_correction' in result:
            lines.append(f"BCa z0         : {result['bias_correction']:.6f}")
            lines.append(f"BCa accel      : {result['acceleration']:.6f}")
        if 'block_size' in result:
            lines.append(f"Block size     : {result['block_size']}")
        if 'calibrated_confidence' in result:
            lines.append(f"Calibrated conf: {result['calibrated_confidence']:.4f}")
            lines.append(f"Actual coverage: {result['actual_coverage']:.4f}")
        return "\n".join(lines)
