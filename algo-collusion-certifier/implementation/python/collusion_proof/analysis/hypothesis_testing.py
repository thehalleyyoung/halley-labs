"""General hypothesis testing framework for collusion detection.

Provides t-tests, permutation tests, multiple testing corrections,
test combination methods, and equivalence tests.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Callable


class HypothesisTestFramework:
    """Framework for running and combining hypothesis tests."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._results: List[Dict[str, Any]] = []

    # ── Parametric Tests ─────────────────────────────────────────────────────

    def t_test(self, x: np.ndarray, mu0: float = 0.0,
               alternative: str = "two-sided") -> Dict[str, Any]:
        """One-sample t-test.

        Returns dict with test_statistic, p_value, reject, ci, effect_size, df.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        mean_x = np.mean(x)
        sd_x = np.std(x, ddof=1)
        se = sd_x / np.sqrt(n)
        t_stat = (mean_x - mu0) / (se + 1e-15)
        df = n - 1

        if alternative == "two-sided":
            p_value = 2.0 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = mean_x - t_crit * se
        ci_upper = mean_x + t_crit * se

        effect = (mean_x - mu0) / (sd_x + 1e-15)

        result = {
            "test_name": "one_sample_t_test",
            "test_statistic": t_stat,
            "p_value": p_value,
            "reject": p_value < self.alpha,
            "ci": (ci_lower, ci_upper),
            "effect_size": effect,
            "df": df,
            "alternative": alternative,
            "sample_size": n,
        }
        self._results.append(result)
        return result

    def two_sample_t_test(self, x: np.ndarray, y: np.ndarray,
                          equal_var: bool = False,
                          alternative: str = "two-sided") -> Dict[str, Any]:
        """Two-sample t-test (Welch's by default)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        mx, my = np.mean(x), np.mean(y)
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)

        if equal_var:
            sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
            se = np.sqrt(sp2 * (1.0 / nx + 1.0 / ny))
            df = nx + ny - 2
        else:
            se = np.sqrt(vx / nx + vy / ny)
            numerator = (vx / nx + vy / ny) ** 2
            denominator = ((vx / nx) ** 2 / (nx - 1)
                           + (vy / ny) ** 2 / (ny - 1))
            df = numerator / (denominator + 1e-30)

        t_stat = (mx - my) / (se + 1e-15)

        if alternative == "two-sided":
            p_value = 2.0 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        diff = mx - my
        ci = (diff - t_crit * se, diff + t_crit * se)

        pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        cohens_d = (mx - my) / (pooled_sd + 1e-15)

        result = {
            "test_name": "two_sample_t_test",
            "test_statistic": t_stat,
            "p_value": p_value,
            "reject": p_value < self.alpha,
            "ci": ci,
            "effect_size": cohens_d,
            "df": df,
            "alternative": alternative,
            "equal_var": equal_var,
            "sample_sizes": (nx, ny),
            "means": (mx, my),
        }
        self._results.append(result)
        return result

    def paired_t_test(self, x: np.ndarray, y: np.ndarray,
                      alternative: str = "two-sided") -> Dict[str, Any]:
        """Paired t-test."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        diffs = x - y
        n = len(diffs)
        mean_d = np.mean(diffs)
        sd_d = np.std(diffs, ddof=1)
        se = sd_d / np.sqrt(n)
        t_stat = mean_d / (se + 1e-15)
        df = n - 1

        if alternative == "two-sided":
            p_value = 2.0 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci = (mean_d - t_crit * se, mean_d + t_crit * se)
        effect = mean_d / (sd_d + 1e-15)

        result = {
            "test_name": "paired_t_test",
            "test_statistic": t_stat,
            "p_value": p_value,
            "reject": p_value < self.alpha,
            "ci": ci,
            "effect_size": effect,
            "df": df,
            "alternative": alternative,
            "sample_size": n,
            "mean_difference": mean_d,
        }
        self._results.append(result)
        return result

    # ── Nonparametric Tests ──────────────────────────────────────────────────

    def permutation_test(self, x: np.ndarray, y: np.ndarray,
                         statistic: Optional[Callable] = None,
                         n_permutations: int = 10000,
                         alternative: str = "two-sided") -> Dict[str, Any]:
        """Permutation test for two groups."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        combined = np.concatenate([x, y])
        nx = len(x)
        n_total = len(combined)

        if statistic is None:
            def statistic(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.mean(a) - np.mean(b))

        observed = statistic(x, y)
        rng = np.random.default_rng()
        count = 0

        perm_stats = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(n_total)
            perm_x = combined[perm[:nx]]
            perm_y = combined[perm[nx:]]
            perm_stats[i] = statistic(perm_x, perm_y)

        if alternative == "two-sided":
            count = int(np.sum(np.abs(perm_stats) >= abs(observed)))
        elif alternative == "greater":
            count = int(np.sum(perm_stats >= observed))
        else:
            count = int(np.sum(perm_stats <= observed))

        p_value = (count + 1) / (n_permutations + 1)

        result = {
            "test_name": "permutation_test",
            "test_statistic": observed,
            "p_value": p_value,
            "reject": p_value < self.alpha,
            "alternative": alternative,
            "n_permutations": n_permutations,
            "sample_sizes": (nx, n_total - nx),
            "perm_distribution_quantiles": {
                "2.5%": float(np.percentile(perm_stats, 2.5)),
                "50%": float(np.percentile(perm_stats, 50.0)),
                "97.5%": float(np.percentile(perm_stats, 97.5)),
            },
        }
        self._results.append(result)
        return result

    def one_sample_permutation(self, x: np.ndarray, mu0: float = 0.0,
                               n_permutations: int = 10000,
                               alternative: str = "two-sided") -> Dict[str, Any]:
        """Permutation test for one sample against mu0.

        Randomly flips the sign of each (x_i - mu0) to build a null distribution.
        """
        x = np.asarray(x, dtype=float)
        centered = x - mu0
        n = len(centered)
        observed = float(np.mean(centered))

        rng = np.random.default_rng()
        perm_stats = np.empty(n_permutations)
        for i in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=n)
            perm_stats[i] = np.mean(centered * signs)

        if alternative == "two-sided":
            count = int(np.sum(np.abs(perm_stats) >= abs(observed)))
        elif alternative == "greater":
            count = int(np.sum(perm_stats >= observed))
        else:
            count = int(np.sum(perm_stats <= observed))

        p_value = (count + 1) / (n_permutations + 1)

        result = {
            "test_name": "one_sample_permutation",
            "test_statistic": observed,
            "p_value": p_value,
            "reject": p_value < self.alpha,
            "alternative": alternative,
            "n_permutations": n_permutations,
            "sample_size": n,
            "mu0": mu0,
        }
        self._results.append(result)
        return result

    def sign_test(self, x: np.ndarray, mu0: float = 0.0,
                  alternative: str = "two-sided") -> Dict[str, Any]:
        """Sign test (nonparametric)."""
        x = np.asarray(x, dtype=float)
        diffs = x - mu0
        diffs = diffs[diffs != 0.0]
        n = len(diffs)
        n_positive = int(np.sum(diffs > 0))

        if alternative == "two-sided":
            p_value = 2.0 * min(
                stats.binom.cdf(n_positive, n, 0.5),
                stats.binom.sf(n_positive - 1, n, 0.5),
            )
            p_value = min(p_value, 1.0)
        elif alternative == "greater":
            p_value = stats.binom.sf(n_positive - 1, n, 0.5)
        else:
            p_value = stats.binom.cdf(n_positive, n, 0.5)

        result = {
            "test_name": "sign_test",
            "test_statistic": n_positive,
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "alternative": alternative,
            "n_nonzero": n,
            "n_positive": n_positive,
            "n_negative": n - n_positive,
            "mu0": mu0,
            "sample_size": len(x),
        }
        self._results.append(result)
        return result

    def wilcoxon_test(self, x: np.ndarray, y: Optional[np.ndarray] = None,
                      alternative: str = "two-sided") -> Dict[str, Any]:
        """Wilcoxon signed-rank test.

        If y is provided, tests paired differences x - y against zero.
        Otherwise tests x against zero.
        """
        x = np.asarray(x, dtype=float)
        if y is not None:
            y = np.asarray(y, dtype=float)
            diffs = x - y
        else:
            diffs = x.copy()

        diffs = diffs[diffs != 0.0]
        n = len(diffs)

        abs_diffs = np.abs(diffs)
        ranks = stats.rankdata(abs_diffs)
        signed_ranks = ranks * np.sign(diffs)
        w_plus = float(np.sum(signed_ranks[signed_ranks > 0]))
        w_minus = float(-np.sum(signed_ranks[signed_ranks < 0]))

        if alternative == "two-sided":
            w_stat = min(w_plus, w_minus)
        elif alternative == "greater":
            w_stat = w_plus
        else:
            w_stat = w_minus

        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0

        unique_abs = np.unique(abs_diffs)
        if len(unique_abs) < n:
            for val in unique_abs:
                t_i = float(np.sum(abs_diffs == val))
                if t_i > 1:
                    var_w -= (t_i ** 3 - t_i) / 48.0

        se_w = np.sqrt(var_w + 1e-30)

        if alternative == "two-sided":
            z_stat = (w_stat - mean_w) / se_w
            p_value = 2.0 * stats.norm.sf(abs(z_stat))
        elif alternative == "greater":
            z_stat = (w_plus - mean_w) / se_w
            p_value = stats.norm.sf(z_stat)
        else:
            z_stat = (w_minus - mean_w) / se_w
            p_value = stats.norm.sf(z_stat)

        effect_r = z_stat / np.sqrt(n) if n > 0 else 0.0

        result = {
            "test_name": "wilcoxon_signed_rank",
            "test_statistic": w_stat,
            "z_statistic": z_stat,
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "alternative": alternative,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "effect_size_r": effect_r,
            "n_nonzero": n,
        }
        self._results.append(result)
        return result

    def mann_whitney_test(self, x: np.ndarray, y: np.ndarray,
                          alternative: str = "two-sided") -> Dict[str, Any]:
        """Mann-Whitney U test."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)

        combined = np.concatenate([x, y])
        ranks = stats.rankdata(combined)
        r_x = ranks[:nx]

        u_x = np.sum(r_x) - nx * (nx + 1) / 2.0
        u_y = nx * ny - u_x
        u_stat = min(u_x, u_y)

        mean_u = nx * ny / 2.0
        var_u = nx * ny * (nx + ny + 1) / 12.0

        unique_vals, counts = np.unique(combined, return_counts=True)
        for t_i in counts:
            if t_i > 1:
                var_u -= nx * ny * (t_i ** 3 - t_i) / (12.0 * (nx + ny) * (nx + ny - 1))

        se_u = np.sqrt(var_u + 1e-30)

        if alternative == "two-sided":
            z_stat = (u_stat - mean_u) / se_u
            p_value = 2.0 * stats.norm.sf(abs(z_stat))
        elif alternative == "greater":
            z_stat = (u_x - mean_u) / se_u
            p_value = stats.norm.sf(z_stat)
        else:
            z_stat = (u_x - mean_u) / se_u
            p_value = stats.norm.cdf(z_stat)

        rank_biserial = 1.0 - 2.0 * u_stat / (nx * ny) if nx * ny > 0 else 0.0

        result = {
            "test_name": "mann_whitney_u",
            "test_statistic": u_stat,
            "z_statistic": z_stat,
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "alternative": alternative,
            "u_x": float(u_x),
            "u_y": float(u_y),
            "effect_size_rank_biserial": rank_biserial,
            "sample_sizes": (nx, ny),
        }
        self._results.append(result)
        return result

    # ── Equivalence / Superiority Tests ──────────────────────────────────────

    def equivalence_test(self, x: np.ndarray, mu0: float = 0.0,
                         margin: float = 0.1) -> Dict[str, Any]:
        """TOST (two one-sided tests) equivalence test.

        Tests H0: |mean(x) - mu0| >= margin against
              H1: |mean(x) - mu0| < margin.
        Both one-sided tests must reject for equivalence.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        mean_x = np.mean(x)
        se = np.std(x, ddof=1) / np.sqrt(n)
        df = n - 1

        t_upper = (mean_x - mu0 - margin) / (se + 1e-15)
        p_upper = stats.t.cdf(t_upper, df)

        t_lower = (mean_x - mu0 + margin) / (se + 1e-15)
        p_lower = stats.t.sf(t_lower, df)

        p_value = max(p_upper, p_lower)

        t_crit = stats.t.ppf(1 - self.alpha, df)
        ci_90 = (mean_x - t_crit * se, mean_x + t_crit * se)

        result = {
            "test_name": "tost_equivalence",
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "p_upper": float(p_upper),
            "p_lower": float(p_lower),
            "t_upper": float(t_upper),
            "t_lower": float(t_lower),
            "equivalence_margin": margin,
            "mu0": mu0,
            "mean": float(mean_x),
            "ci_90": ci_90,
            "df": df,
            "sample_size": n,
        }
        self._results.append(result)
        return result

    def superiority_test(self, x: np.ndarray, y: np.ndarray,
                         margin: float = 0.0,
                         alternative: str = "greater") -> Dict[str, Any]:
        """Superiority test: tests whether mean(x) - mean(y) > margin."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        mx, my = np.mean(x), np.mean(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

        se = np.sqrt(vx / nx + vy / ny)
        diff = mx - my
        t_stat = (diff - margin) / (se + 1e-15)

        num = (vx / nx + vy / ny) ** 2
        denom = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
        df = num / (denom + 1e-30)

        if alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        t_crit = stats.t.ppf(1 - self.alpha, df)
        if alternative == "greater":
            ci_lower_bound = diff - t_crit * se
            ci = (ci_lower_bound, np.inf)
        else:
            ci_upper_bound = diff + t_crit * se
            ci = (-np.inf, ci_upper_bound)

        result = {
            "test_name": "superiority_test",
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "alternative": alternative,
            "margin": margin,
            "mean_difference": float(diff),
            "ci": ci,
            "df": float(df),
            "sample_sizes": (nx, ny),
        }
        self._results.append(result)
        return result

    # ── Multiple Testing Correction ──────────────────────────────────────────

    def holm_bonferroni(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Holm-Bonferroni step-down correction."""
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(1.0, sorted_p[i] * (n - i))

        cum_max = 0.0
        for i in range(n):
            idx = sorted_idx[i]
            adjusted[idx] = max(adjusted[idx], cum_max)
            cum_max = adjusted[idx]

        return {
            "method": "holm_bonferroni",
            "adjusted_p_values": adjusted,
            "reject": adjusted < self.alpha,
            "alpha": self.alpha,
            "n_rejections": int(np.sum(adjusted < self.alpha)),
        }

    def benjamini_hochberg(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Benjamini-Hochberg FDR correction."""
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(1.0, sorted_p[i] * n / (i + 1))

        cum_min = 1.0
        for i in range(n - 1, -1, -1):
            idx = sorted_idx[i]
            adjusted[idx] = min(adjusted[idx], cum_min)
            cum_min = adjusted[idx]

        return {
            "method": "benjamini_hochberg",
            "adjusted_p_values": adjusted,
            "reject": adjusted < self.alpha,
            "alpha": self.alpha,
            "n_rejections": int(np.sum(adjusted < self.alpha)),
        }

    def bonferroni(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Simple Bonferroni correction."""
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        adjusted = np.minimum(p_values * n, 1.0)

        return {
            "method": "bonferroni",
            "adjusted_p_values": adjusted,
            "reject": adjusted < self.alpha,
            "alpha": self.alpha,
            "n_rejections": int(np.sum(adjusted < self.alpha)),
        }

    def sidak(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Šidák correction: adjusted_p = 1 - (1 - p)^n."""
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        adjusted = 1.0 - (1.0 - p_values) ** n
        adjusted = np.minimum(adjusted, 1.0)

        return {
            "method": "sidak",
            "adjusted_p_values": adjusted,
            "reject": adjusted < self.alpha,
            "alpha": self.alpha,
            "n_rejections": int(np.sum(adjusted < self.alpha)),
        }

    def hochberg(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Hochberg step-up correction.

        Step up from the largest p-value: adjusted_p[i] = min(p[i] * (n-i), 1).
        Enforces monotonicity downward from the largest rank.
        """
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(1.0, sorted_p[i] * (n - i))

        cum_min = 1.0
        for i in range(n - 1, -1, -1):
            idx = sorted_idx[i]
            adjusted[idx] = min(adjusted[idx], cum_min)
            cum_min = adjusted[idx]

        return {
            "method": "hochberg",
            "adjusted_p_values": adjusted,
            "reject": adjusted < self.alpha,
            "alpha": self.alpha,
            "n_rejections": int(np.sum(adjusted < self.alpha)),
        }

    # ── Test Combination ─────────────────────────────────────────────────────

    def fisher_combination(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Fisher's method for combining p-values.

        Test statistic: -2 * sum(log(p_i)) ~ chi2(2k).
        """
        p_values = np.asarray(p_values, dtype=float)
        k = len(p_values)
        clamped = np.clip(p_values, 1e-300, 1.0)
        chi2_stat = -2.0 * np.sum(np.log(clamped))
        combined_p = float(stats.chi2.sf(chi2_stat, 2 * k))
        return {
            "method": "fisher",
            "test_statistic": float(chi2_stat),
            "combined_p_value": combined_p,
            "reject": combined_p < self.alpha,
            "df": 2 * k,
            "n_tests": k,
        }

    def stouffer_combination(self, p_values: np.ndarray,
                             weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Stouffer's Z-method for combining p-values.

        Converts each p-value to a z-score via the inverse normal CDF, then
        combines: Z = sum(w_i * z_i) / sqrt(sum(w_i^2)).
        """
        p_values = np.asarray(p_values, dtype=float)
        k = len(p_values)
        clamped = np.clip(p_values, 1e-15, 1.0 - 1e-15)
        z_scores = stats.norm.ppf(1.0 - clamped)

        if weights is None:
            weights = np.ones(k)
        else:
            weights = np.asarray(weights, dtype=float)

        z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights ** 2))
        combined_p = float(stats.norm.sf(z_combined))

        return {
            "method": "stouffer",
            "test_statistic": float(z_combined),
            "combined_p_value": combined_p,
            "reject": combined_p < self.alpha,
            "n_tests": k,
            "individual_z_scores": z_scores.tolist(),
        }

    def harmonic_mean_combination(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Harmonic mean p-value (HMP) combination.

        HMP = k / sum(1/p_i). The combined p-value is approximately
        HMP adjusted by the asymptotic Landau distribution factor.
        """
        p_values = np.asarray(p_values, dtype=float)
        k = len(p_values)
        clamped = np.clip(p_values, 1e-300, 1.0)
        hmp = float(k / np.sum(1.0 / clamped))

        combined_p = min(1.0, hmp * np.log(k) + hmp * np.euler_gamma) if k > 1 else hmp
        combined_p = min(max(combined_p, 0.0), 1.0)

        return {
            "method": "harmonic_mean",
            "harmonic_mean_p": hmp,
            "combined_p_value": combined_p,
            "reject": combined_p < self.alpha,
            "n_tests": k,
        }

    def simes_test(self, p_values: np.ndarray) -> Dict[str, Any]:
        """Simes' test for the global null hypothesis.

        Rejects if min_i { k * p_(i) / i } < alpha, where p_(i) are sorted.
        """
        p_values = np.asarray(p_values, dtype=float)
        k = len(p_values)
        sorted_p = np.sort(p_values)
        ranks = np.arange(1, k + 1)
        simes_values = k * sorted_p / ranks
        combined_p = float(np.min(simes_values))

        return {
            "method": "simes",
            "combined_p_value": min(combined_p, 1.0),
            "reject": combined_p < self.alpha,
            "n_tests": k,
            "critical_index": int(np.argmin(simes_values)),
        }

    def cauchy_combination(self, p_values: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Cauchy combination test (Liu & Xie 2020).

        Uses the Cauchy distribution: T = sum(w_i * tan((0.5 - p_i) * pi)).
        The null distribution is approximately Cauchy(0,1).
        """
        p_values = np.asarray(p_values, dtype=float)
        k = len(p_values)
        clamped = np.clip(p_values, 1e-15, 1.0 - 1e-15)

        if weights is None:
            weights = np.ones(k) / k
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / np.sum(weights)

        cauchy_values = np.tan((0.5 - clamped) * np.pi)
        t_stat = float(np.sum(weights * cauchy_values))
        combined_p = float(stats.cauchy.sf(t_stat))
        combined_p = min(max(combined_p, 0.0), 1.0)

        return {
            "method": "cauchy",
            "test_statistic": t_stat,
            "combined_p_value": combined_p,
            "reject": combined_p < self.alpha,
            "n_tests": k,
        }

    # ── HAC Standard Errors ──────────────────────────────────────────────────

    def t_test_hac(self, x: np.ndarray, mu0: float = 0.0,
                   max_lag: Optional[int] = None,
                   alternative: str = "two-sided") -> Dict[str, Any]:
        """T-test with Newey-West HAC standard errors for autocorrelated data.

        Uses the Bartlett kernel with automatic lag selection
        (floor(4*(n/100)^(2/9))) when max_lag is not specified.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        mean_x = float(np.mean(x))
        resid = x - mean_x

        if max_lag is None:
            max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        max_lag = min(max_lag, n - 1)

        gamma_0 = float(np.mean(resid ** 2))
        nw_var = gamma_0
        for lag in range(1, max_lag + 1):
            gamma_lag = float(np.mean(resid[lag:] * resid[:-lag]))
            bartlett_weight = 1.0 - lag / (max_lag + 1.0)
            nw_var += 2.0 * bartlett_weight * gamma_lag

        se = np.sqrt(max(nw_var, 0.0) / n)
        t_stat = (mean_x - mu0) / (se + 1e-15)
        df = n - 1

        if alternative == "two-sided":
            p_value = 2.0 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci = (mean_x - t_crit * se, mean_x + t_crit * se)

        result = {
            "test_name": "t_test_hac",
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "reject": float(p_value) < self.alpha,
            "ci": ci,
            "hac_se": float(se),
            "max_lag": max_lag,
            "nw_variance": float(nw_var),
            "alternative": alternative,
            "df": df,
            "sample_size": n,
            "mean": mean_x,
        }
        self._results.append(result)
        return result

    # ── Utility Methods ──────────────────────────────────────────────────────

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()

    def summary(self) -> str:
        """Human-readable summary of all tests."""
        if not self._results:
            return "No tests have been run."

        lines = [
            f"Hypothesis Test Summary (alpha={self.alpha})",
            "=" * 60,
        ]
        for i, r in enumerate(self._results):
            name = r.get("test_name", "unknown")
            p = r.get("p_value", r.get("combined_p_value", float("nan")))
            reject = r.get("reject", False)
            verdict = "REJECT H0" if reject else "FAIL TO REJECT H0"

            line = f"  [{i + 1}] {name}: p={p:.6f} -> {verdict}"
            stat = r.get("test_statistic", None)
            if stat is not None:
                line += f"  (stat={stat:.4f})"
            effect = r.get("effect_size", None)
            if effect is not None:
                line += f"  (d={effect:.4f})"
            lines.append(line)

        n_rejected = sum(1 for r in self._results if r.get("reject", False))
        lines.append("-" * 60)
        lines.append(
            f"Total: {len(self._results)} tests, "
            f"{n_rejected} rejected, "
            f"{len(self._results) - n_rejected} not rejected"
        )
        return "\n".join(lines)

    def all_p_values(self) -> np.ndarray:
        """Get all p-values from stored results."""
        pvals = []
        for r in self._results:
            if "p_value" in r:
                pvals.append(r["p_value"])
            elif "combined_p_value" in r:
                pvals.append(r["combined_p_value"])
        return np.array(pvals)

    def correct_all(self, method: str = "holm") -> Dict[str, Any]:
        """Apply multiple testing correction to all stored results.

        Supported methods: 'holm', 'bh' (Benjamini-Hochberg),
        'bonferroni', 'sidak', 'hochberg'.
        """
        pvals = self.all_p_values()
        if len(pvals) == 0:
            return {
                "method": method,
                "adjusted_p_values": np.array([]),
                "reject": np.array([], dtype=bool),
                "alpha": self.alpha,
                "n_rejections": 0,
            }

        dispatch: Dict[str, Callable[[np.ndarray], Dict[str, Any]]] = {
            "holm": self.holm_bonferroni,
            "bh": self.benjamini_hochberg,
            "bonferroni": self.bonferroni,
            "sidak": self.sidak,
            "hochberg": self.hochberg,
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown correction method '{method}'. "
                f"Choose from: {list(dispatch.keys())}"
            )

        correction = dispatch[method](pvals)

        adjusted = correction["adjusted_p_values"]
        reject = correction["reject"]
        for i, r in enumerate(self._results):
            if i < len(adjusted):
                r["adjusted_p_value"] = float(adjusted[i])
                r["reject_corrected"] = bool(reject[i])

        return correction
