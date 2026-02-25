"""Hypothesis testing for phase transitions in finite-width neural networks.

Provides parametric and non-parametric tests for detecting phase transitions,
change-point detection along parameter paths, and multiple comparison
corrections for grid-sweep analyses.

Mathematical background
-----------------------
Phase transitions manifest as abrupt changes in order-parameter distributions.
We employ Welch's t-test, Mann–Whitney U, Kolmogorov–Smirnov, and permutation
tests to detect distributional differences, and likelihood-ratio change-point
detection to locate transitions along one-dimensional parameter paths.

Multiple comparisons across a grid of hypothesis tests are controlled via
Bonferroni, Holm, or Benjamini–Hochberg corrections.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class TestResult:
    """Result of a single hypothesis test.

    Parameters
    ----------
    test_name : str
        Identifier of the test (e.g. ``"welch_t"``, ``"mann_whitney"``).
    statistic : float
        Test statistic value.
    p_value : float
        Two-sided p-value.
    reject_null : bool
        Whether the null hypothesis is rejected at the chosen α.
    effect_size : float
        Standardised effect size (e.g. Cohen's *d*).
    confidence_interval : tuple of float
        Confidence interval for the effect or difference.
    details : dict
        Additional information specific to the test.
    """

    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 1.0
    reject_null: bool = False
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    details: Dict = field(default_factory=dict)


# ======================================================================
#  Enumerations
# ======================================================================


class MultipleTestCorrection(enum.Enum):
    """Methods for multiple-comparison correction."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BH = "benjamini_hochberg"
    NONE = "none"


# ======================================================================
#  Main class
# ======================================================================


class PhaseTransitionTester:
    """Hypothesis tests tailored to phase-transition detection.

    Parameters
    ----------
    significance_level : float
        Family-wise or per-test significance level α.
    correction : MultipleTestCorrection
        Correction method applied when many tests are performed.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        correction: MultipleTestCorrection = MultipleTestCorrection.BONFERRONI,
    ) -> None:
        if not 0.0 < significance_level < 1.0:
            raise ValueError(
                f"significance_level must be in (0, 1), got {significance_level}"
            )
        self.alpha = significance_level
        self.correction = correction

    # ------------------------------------------------------------------
    #  Two-sample regime difference
    # ------------------------------------------------------------------

    def test_regime_difference(
        self,
        samples_a: NDArray[np.floating],
        samples_b: NDArray[np.floating],
        test_type: str = "welch",
    ) -> TestResult:
        """Test whether order-parameter distributions differ between regimes.

        Parameters
        ----------
        samples_a, samples_b : ndarray
            Order-parameter samples from two parameter points.
        test_type : str
            ``"welch"`` for Welch's t-test (default) or ``"mann_whitney"``
            for a non-parametric rank-based test.

        Returns
        -------
        TestResult
        """
        a = np.asarray(samples_a, dtype=np.float64).ravel()
        b = np.asarray(samples_b, dtype=np.float64).ravel()

        if test_type == "welch":
            stat, pval = stats.ttest_ind(a, b, equal_var=False)
            es = self._effect_size_cohens_d(a, b)
            # CI for difference in means via Welch–Satterthwaite
            diff = np.mean(a) - np.mean(b)
            se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
            df = _welch_df(a, b)
            t_crit = stats.t.ppf(1.0 - self.alpha / 2.0, df)
            ci = (diff - t_crit * se, diff + t_crit * se)
            name = "welch_t"
        elif test_type == "mann_whitney":
            stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            es = self._effect_size_rank_biserial(a, b)
            ci = (float("nan"), float("nan"))
            name = "mann_whitney_u"
        else:
            raise ValueError(f"Unknown test_type: {test_type!r}")

        return TestResult(
            test_name=name,
            statistic=float(stat),
            p_value=float(pval),
            reject_null=float(pval) < self.alpha,
            effect_size=float(es),
            confidence_interval=ci,
            details={"n_a": len(a), "n_b": len(b), "test_type": test_type},
        )

    # ------------------------------------------------------------------
    #  Boundary existence via change-point detection
    # ------------------------------------------------------------------

    def test_boundary_existence(
        self,
        order_params_path: NDArray[np.floating],
        positions: NDArray[np.floating],
    ) -> TestResult:
        """Test for a phase transition along a parameter path.

        Uses single change-point detection by maximising the Gaussian
        likelihood ratio over all candidate split points.

        Parameters
        ----------
        order_params_path : ndarray, shape (n,)
            Order-parameter values measured along the path.
        positions : ndarray, shape (n,)
            Corresponding parameter-space positions along the path.

        Returns
        -------
        TestResult
        """
        data = np.asarray(order_params_path, dtype=np.float64).ravel()
        pos = np.asarray(positions, dtype=np.float64).ravel()

        if len(data) < 4:
            return TestResult(
                test_name="changepoint_lr",
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                effect_size=0.0,
                confidence_interval=(float("nan"), float("nan")),
                details={"reason": "insufficient data"},
            )

        cp_idx, lr_stat = self._likelihood_ratio_changepoint(data)

        # Approximate p-value via permutation null (fast, 2000 permutations)
        rng = np.random.default_rng(42)
        n_perm = 2000
        null_stats = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(data)
            _, null_stats[i] = self._likelihood_ratio_changepoint(perm)
        pval = float(np.mean(null_stats >= lr_stat))

        # Effect size: standardised mean difference across the change point
        if 0 < cp_idx < len(data):
            es = self._effect_size_cohens_d(data[:cp_idx], data[cp_idx:])
        else:
            es = 0.0

        return TestResult(
            test_name="changepoint_lr",
            statistic=float(lr_stat),
            p_value=pval,
            reject_null=pval < self.alpha,
            effect_size=float(es),
            confidence_interval=(float(pos[cp_idx]), float(pos[cp_idx])),
            details={
                "changepoint_index": int(cp_idx),
                "changepoint_position": float(pos[cp_idx]),
                "n_permutations": n_perm,
            },
        )

    @staticmethod
    def _likelihood_ratio_changepoint(
        data: NDArray[np.floating],
    ) -> Tuple[int, float]:
        """Find the single change point maximising the Gaussian LR statistic.

        Parameters
        ----------
        data : ndarray, shape (n,)
            Observed values.

        Returns
        -------
        best_idx : int
            Index of the optimal split (data[:idx] vs data[idx:]).
        best_lr : float
            Likelihood-ratio statistic at the best split.
        """
        n = len(data)
        total_var = np.var(data, ddof=0)
        if total_var == 0.0:
            return n // 2, 0.0

        log_l_full = -0.5 * n * np.log(2.0 * np.pi * total_var) - 0.5 * n

        best_lr = -np.inf
        best_idx = 1
        for k in range(2, n - 1):
            left = data[:k]
            right = data[k:]
            var_l = np.var(left, ddof=0)
            var_r = np.var(right, ddof=0)
            # Avoid log(0)
            var_l = max(var_l, 1e-300)
            var_r = max(var_r, 1e-300)
            log_l_split = (
                -0.5 * k * np.log(2.0 * np.pi * var_l)
                - 0.5 * k
                + -0.5 * (n - k) * np.log(2.0 * np.pi * var_r)
                - 0.5 * (n - k)
            )
            lr = log_l_split - log_l_full
            if lr > best_lr:
                best_lr = lr
                best_idx = k

        return best_idx, float(max(best_lr, 0.0))

    # ------------------------------------------------------------------
    #  Sequential testing along a path
    # ------------------------------------------------------------------

    def sequential_test_along_path(
        self,
        order_param_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        path_points: NDArray[np.floating],
        early_stop: bool = True,
    ) -> List[TestResult]:
        """Sequentially test for transitions between adjacent path points.

        Parameters
        ----------
        order_param_fn : callable
            Maps a parameter-space point to order-parameter samples.
        path_points : ndarray, shape (n_points, dim)
            Ordered points along a path in parameter space.
        early_stop : bool
            If *True*, stop after the first significant result.

        Returns
        -------
        list of TestResult
            One result per adjacent pair.
        """
        path_points = np.atleast_2d(path_points)
        results: List[TestResult] = []

        prev_samples = np.atleast_1d(order_param_fn(path_points[0]))
        for i in range(1, len(path_points)):
            cur_samples = np.atleast_1d(order_param_fn(path_points[i]))
            res = self.test_regime_difference(prev_samples, cur_samples)
            res.details["path_index"] = i
            results.append(res)
            if early_stop and res.reject_null:
                break
            prev_samples = cur_samples

        # Apply multiple comparison correction to collected p-values
        if len(results) > 1 and self.correction != MultipleTestCorrection.NONE:
            pvals = np.array([r.p_value for r in results])
            adj_pvals = self.correct_multiple_comparisons(pvals, self.correction)
            for r, ap in zip(results, adj_pvals):
                r.p_value = float(ap)
                r.reject_null = float(ap) < self.alpha

        return results

    # ------------------------------------------------------------------
    #  Multiple comparison corrections
    # ------------------------------------------------------------------

    def correct_multiple_comparisons(
        self,
        p_values: NDArray[np.floating],
        method: Optional[MultipleTestCorrection] = None,
    ) -> NDArray[np.floating]:
        """Apply a multiple-comparison correction.

        Parameters
        ----------
        p_values : ndarray
            Raw p-values.
        method : MultipleTestCorrection, optional
            Override the instance default.

        Returns
        -------
        ndarray
            Adjusted p-values.
        """
        p_values = np.asarray(p_values, dtype=np.float64)
        if method is None:
            method = self.correction

        if method == MultipleTestCorrection.BONFERRONI:
            return self._bonferroni(p_values, self.alpha)
        elif method == MultipleTestCorrection.HOLM:
            return self._holm(p_values, self.alpha)
        elif method == MultipleTestCorrection.BH:
            return self._benjamini_hochberg(p_values, self.alpha)
        elif method == MultipleTestCorrection.NONE:
            return p_values
        else:
            raise ValueError(f"Unknown correction method: {method}")

    @staticmethod
    def _bonferroni(
        p_values: NDArray[np.floating], alpha: float
    ) -> NDArray[np.floating]:
        """Bonferroni correction: adjusted p_i = min(m · p_i, 1).

        Parameters
        ----------
        p_values : ndarray
            Raw p-values.
        alpha : float
            Significance level (unused in adjustment, included for API
            consistency).

        Returns
        -------
        ndarray
            Adjusted p-values.
        """
        m = len(p_values)
        return np.minimum(p_values * m, 1.0)

    @staticmethod
    def _holm(
        p_values: NDArray[np.floating], alpha: float
    ) -> NDArray[np.floating]:
        """Holm step-down correction.

        Parameters
        ----------
        p_values : ndarray
            Raw p-values.
        alpha : float
            Significance level (unused in adjustment).

        Returns
        -------
        ndarray
            Adjusted p-values.
        """
        m = len(p_values)
        order = np.argsort(p_values)
        adjusted = np.empty(m, dtype=np.float64)
        cummax = 0.0
        for i, idx in enumerate(order):
            val = p_values[idx] * (m - i)
            cummax = max(cummax, val)
            adjusted[idx] = min(cummax, 1.0)
        return adjusted

    @staticmethod
    def _benjamini_hochberg(
        p_values: NDArray[np.floating], alpha: float
    ) -> NDArray[np.floating]:
        """Benjamini–Hochberg (BH) procedure for FDR control.

        Parameters
        ----------
        p_values : ndarray
            Raw p-values.
        alpha : float
            Significance level (unused in adjustment).

        Returns
        -------
        ndarray
            Adjusted p-values.
        """
        m = len(p_values)
        order = np.argsort(p_values)
        adjusted = np.empty(m, dtype=np.float64)

        # Process in reverse (largest to smallest)
        cummin = 1.0
        for i in range(m - 1, -1, -1):
            idx = order[i]
            rank = i + 1
            val = p_values[idx] * m / rank
            cummin = min(cummin, val)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted

    # ------------------------------------------------------------------
    #  Grid sweep tests
    # ------------------------------------------------------------------

    def grid_sweep_tests(
        self,
        sweep_result: dict,
        test_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], TestResult
        ],
    ) -> dict:
        """Apply a test at each cell of a grid sweep.

        Parameters
        ----------
        sweep_result : dict
            Must contain ``"grid_samples"`` (list of (samples_a, samples_b)
            per cell) and ``"grid_shape"`` (tuple).
        test_fn : callable
            Function ``(samples_a, samples_b) -> TestResult``.

        Returns
        -------
        dict
            Keys: ``"results"`` (list of TestResult), ``"p_values"``
            (ndarray), ``"reject_map"`` (bool ndarray of grid shape),
            ``"corrected_p_values"`` (ndarray).
        """
        grid_samples = sweep_result["grid_samples"]
        grid_shape = sweep_result.get("grid_shape", (len(grid_samples),))

        results: List[TestResult] = []
        for sa, sb in grid_samples:
            results.append(test_fn(np.asarray(sa), np.asarray(sb)))

        raw_pvals = np.array([r.p_value for r in results])
        adj_pvals = self.correct_multiple_comparisons(raw_pvals)

        # Update results with corrected p-values
        for r, ap in zip(results, adj_pvals):
            r.p_value = float(ap)
            r.reject_null = float(ap) < self.alpha

        reject_map = (adj_pvals < self.alpha).reshape(grid_shape)

        return {
            "results": results,
            "p_values": raw_pvals,
            "corrected_p_values": adj_pvals,
            "reject_map": reject_map,
        }

    # ------------------------------------------------------------------
    #  Permutation test
    # ------------------------------------------------------------------

    def permutation_test(
        self,
        samples_a: NDArray[np.floating],
        samples_b: NDArray[np.floating],
        n_permutations: int = 10000,
        seed: Optional[int] = None,
    ) -> TestResult:
        """Non-parametric permutation test for a difference in means.

        Parameters
        ----------
        samples_a, samples_b : ndarray
            Samples from two groups.
        n_permutations : int
            Number of random permutations.
        seed : int, optional
            Random seed.

        Returns
        -------
        TestResult
        """
        a = np.asarray(samples_a, dtype=np.float64).ravel()
        b = np.asarray(samples_b, dtype=np.float64).ravel()
        rng = np.random.default_rng(seed)

        obs_diff = float(np.abs(np.mean(a) - np.mean(b)))
        combined = np.concatenate([a, b])
        n_a = len(a)

        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diff = np.abs(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
            if perm_diff >= obs_diff:
                count += 1

        pval = (count + 1) / (n_permutations + 1)
        es = self._effect_size_cohens_d(a, b)

        return TestResult(
            test_name="permutation",
            statistic=obs_diff,
            p_value=pval,
            reject_null=pval < self.alpha,
            effect_size=float(es),
            confidence_interval=(float("nan"), float("nan")),
            details={"n_permutations": n_permutations},
        )

    # ------------------------------------------------------------------
    #  Bootstrap confidence interval
    # ------------------------------------------------------------------

    def bootstrap_confidence_interval(
        self,
        data: NDArray[np.floating],
        statistic_fn: Callable[[NDArray[np.floating]], float],
        n_bootstrap: int = 10000,
        confidence: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for an arbitrary statistic.

        Parameters
        ----------
        data : ndarray
            Observed data.
        statistic_fn : callable
            Maps a data array to a scalar statistic.
        n_bootstrap : int
            Number of bootstrap resamples.
        confidence : float, optional
            Confidence level; defaults to ``1 - alpha``.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple of float
            ``(lower, point_estimate, upper)``.
        """
        data = np.asarray(data, dtype=np.float64)
        rng = np.random.default_rng(seed)
        if confidence is None:
            confidence = 1.0 - self.alpha

        point = float(statistic_fn(data))
        boot_stats = np.empty(n_bootstrap)
        n = len(data)
        for i in range(n_bootstrap):
            resample = data[rng.integers(0, n, size=n)]
            boot_stats[i] = statistic_fn(resample)

        alpha_half = (1.0 - confidence) / 2.0
        lower = float(np.percentile(boot_stats, 100 * alpha_half))
        upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha_half)))
        return (lower, point, upper)

    # ------------------------------------------------------------------
    #  Kolmogorov–Smirnov test
    # ------------------------------------------------------------------

    def kolmogorov_smirnov_test(
        self,
        samples_a: NDArray[np.floating],
        samples_b: NDArray[np.floating],
    ) -> TestResult:
        """Two-sample Kolmogorov–Smirnov test.

        Parameters
        ----------
        samples_a, samples_b : ndarray
            Samples from two distributions.

        Returns
        -------
        TestResult
        """
        a = np.asarray(samples_a, dtype=np.float64).ravel()
        b = np.asarray(samples_b, dtype=np.float64).ravel()
        stat, pval = stats.ks_2samp(a, b)
        es = self._effect_size_cohens_d(a, b)

        return TestResult(
            test_name="kolmogorov_smirnov",
            statistic=float(stat),
            p_value=float(pval),
            reject_null=float(pval) < self.alpha,
            effect_size=float(es),
            confidence_interval=(float("nan"), float("nan")),
            details={"n_a": len(a), "n_b": len(b)},
        )

    # ------------------------------------------------------------------
    #  Effect-size helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _effect_size_cohens_d(
        a: NDArray[np.floating], b: NDArray[np.floating]
    ) -> float:
        """Cohen's *d* (pooled standard deviation).

        Parameters
        ----------
        a, b : ndarray
            Two samples.

        Returns
        -------
        float
        """
        na, nb = len(a), len(b)
        var_a = np.var(a, ddof=1) if na > 1 else 0.0
        var_b = np.var(b, ddof=1) if nb > 1 else 0.0
        pooled_std = np.sqrt(
            ((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1)
        )
        if pooled_std == 0.0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)

    @staticmethod
    def _effect_size_rank_biserial(
        a: NDArray[np.floating], b: NDArray[np.floating]
    ) -> float:
        """Rank-biserial correlation for the Mann–Whitney U test.

        Parameters
        ----------
        a, b : ndarray
            Two samples.

        Returns
        -------
        float
            Rank-biserial *r* in [-1, 1].
        """
        na, nb = len(a), len(b)
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = 1.0 - (2.0 * u_stat) / (na * nb)
        return float(r)

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(results: List[TestResult]) -> str:
        """Return a human-readable summary of multiple test results.

        Parameters
        ----------
        results : list of TestResult
            Test results to summarise.

        Returns
        -------
        str
            Formatted multi-line summary.
        """
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("Hypothesis Test Summary")
        lines.append("=" * 70)
        for i, r in enumerate(results):
            marker = "REJECT" if r.reject_null else "ACCEPT"
            lines.append(
                f"  [{i + 1}] {r.test_name:25s}  stat={r.statistic:+.4f}  "
                f"p={r.p_value:.4e}  [{marker}]  d={r.effect_size:+.3f}"
            )
        n_reject = sum(1 for r in results if r.reject_null)
        lines.append("-" * 70)
        lines.append(f"  {n_reject}/{len(results)} null hypotheses rejected.")
        lines.append("=" * 70)
        return "\n".join(lines)


# ======================================================================
#  Module-level helpers
# ======================================================================


def _welch_df(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Welch–Satterthwaite degrees of freedom.

    Parameters
    ----------
    a, b : ndarray
        Two samples.

    Returns
    -------
    float
        Approximate degrees of freedom.
    """
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1) / na
    vb = np.var(b, ddof=1) / nb
    num = (va + vb) ** 2
    denom = va ** 2 / max(na - 1, 1) + vb ** 2 / max(nb - 1, 1)
    if denom == 0.0:
        return 1.0
    return float(num / denom)
