"""
usability_oracle.comparison.hypothesis — Statistical hypothesis testing
for usability regression detection.

Implements :class:`RegressionTester`, which tests the null hypothesis:

    H₀: μ_B ≤ μ_A  (no regression — after version is no worse)
    H₁: μ_B > μ_A  (regression — after version is worse)

using Welch's t-test (unequal variance), Mann–Whitney U (non-parametric),
and bootstrap testing.  Multiple-testing corrections (Bonferroni,
Holm–Bonferroni, Benjamini–Hochberg) are provided for multi-task
comparisons.

References
----------
- Welch, B. L. (1947). The generalization of "Student's" problem when
  several different population variances are involved. *Biometrika*, 34.
- Mann, H. B. & Whitney, D. R. (1947). On a test of whether one of two
  random variables is stochastically larger than the other. *Annals of
  Mathematical Statistics*, 18(1).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
  rate. *J. Royal Statistical Society B*, 57(1).
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HypothesisResult
# ---------------------------------------------------------------------------


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test.

    Attributes
    ----------
    test_statistic : float
        The test statistic (t, U, or bootstrap Δ).
    p_value : float
        One-sided p-value for H₁: μ_B > μ_A.
    reject_null : bool
        Whether H₀ is rejected at the specified significance level.
    effect_size : float
        Cohen's *d* effect size.
    ci_lower : float
        Lower bound of the confidence interval for the mean difference.
    ci_upper : float
        Upper bound of the confidence interval for the mean difference.
    method : str
        Name of the statistical test used.
    n_a : int
        Sample size for group A.
    n_b : int
        Sample size for group B.
    """

    test_statistic: float = 0.0
    p_value: float = 1.0
    reject_null: bool = False
    effect_size: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    method: str = "welch_t"
    n_a: int = 0
    n_b: int = 0


# ---------------------------------------------------------------------------
# RegressionTester
# ---------------------------------------------------------------------------


class RegressionTester:
    """Statistical regression tester with multiple test methods.

    Provides parametric (Welch's t), non-parametric (Mann–Whitney U),
    and bootstrap hypothesis tests for cost regression detection.

    Parameters
    ----------
    method : str
        Default test method: ``"welch_t"``, ``"mann_whitney"``, or
        ``"bootstrap"``.
    n_bootstrap : int
        Number of bootstrap resamples (for bootstrap test).
    min_samples : int
        Minimum sample size required for valid testing.
    """

    def __init__(
        self,
        method: str = "welch_t",
        n_bootstrap: int = 10_000,
        min_samples: int = 10,
    ) -> None:
        if method not in {"welch_t", "mann_whitney", "bootstrap"}:
            raise ValueError(f"Unknown method {method!r}")
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.min_samples = min_samples

    def test(
        self,
        cost_samples_a: np.ndarray,
        cost_samples_b: np.ndarray,
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Run the hypothesis test for cost regression.

        H₀: μ_B ≤ μ_A  (no regression)
        H₁: μ_B > μ_A  (regression)

        Parameters
        ----------
        cost_samples_a : np.ndarray
            Cost samples from the *before* version.
        cost_samples_b : np.ndarray
            Cost samples from the *after* version.
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        HypothesisResult
            Full test result with statistic, p-value, decision, and CI.

        Raises
        ------
        ValueError
            If samples are too small or contain non-finite values.
        """
        a = np.asarray(cost_samples_a, dtype=np.float64).ravel()
        b = np.asarray(cost_samples_b, dtype=np.float64).ravel()

        if len(a) < self.min_samples or len(b) < self.min_samples:
            logger.warning(
                "Insufficient samples: n_a=%d, n_b=%d (min=%d)",
                len(a), len(b), self.min_samples,
            )
            return HypothesisResult(
                method=self.method, n_a=len(a), n_b=len(b),
            )

        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("Samples contain non-finite values")

        if self.method == "welch_t":
            stat, pval = self._welch_t_test(a, b)
        elif self.method == "mann_whitney":
            stat, pval = self._mann_whitney_u(a, b)
        else:
            stat, pval = self._bootstrap_test(a, b, self.n_bootstrap)

        effect_d, ci_lo, ci_hi = self._effect_size_ci(a, b, alpha)

        reject = pval < alpha

        return HypothesisResult(
            test_statistic=stat,
            p_value=pval,
            reject_null=reject,
            effect_size=effect_d,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            method=self.method,
            n_a=len(a),
            n_b=len(b),
        )

    # ------------------------------------------------------------------
    # Test implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _welch_t_test(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float]:
        """Welch's t-test for unequal variances (one-sided).

        Tests H₁: μ_B > μ_A.

        The test statistic is:

        .. math::
            t = \\frac{\\bar{X}_B - \\bar{X}_A}
                      {\\sqrt{s_A^2/n_A + s_B^2/n_B}}

        with Welch–Satterthwaite degrees of freedom:

        .. math::
            \\nu = \\frac{(s_A^2/n_A + s_B^2/n_B)^2}
                        {(s_A^2/n_A)^2/(n_A-1) + (s_B^2/n_B)^2/(n_B-1)}

        Parameters
        ----------
        a, b : np.ndarray
            Cost samples from versions A and B.

        Returns
        -------
        tuple[float, float]
            ``(t_statistic, p_value)``.
        """
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-15:
            return (0.0, 1.0)

        t_stat = (mean_b - mean_a) / se

        # Welch–Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        if denom < 1e-15:
            df = n_a + n_b - 2
        else:
            df = num / denom

        # One-sided p-value: P(T > t_stat) under H₀
        p_value = float(sp_stats.t.sf(t_stat, df))

        return (float(t_stat), p_value)

    @staticmethod
    def _mann_whitney_u(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float]:
        """Mann–Whitney U test (one-sided, non-parametric).

        Tests H₁: P(B > A) > 0.5  (stochastic dominance).

        The U statistic counts the number of pairs (aᵢ, bⱼ) where
        bⱼ > aᵢ:

        .. math::
            U = \\sum_{i=1}^{n_A} \\sum_{j=1}^{n_B} \\mathbb{1}[b_j > a_i]

        Parameters
        ----------
        a, b : np.ndarray

        Returns
        -------
        tuple[float, float]
            ``(U_statistic, p_value)``.
        """
        result = sp_stats.mannwhitneyu(
            b, a, alternative="greater", method="auto"
        )
        return (float(result.statistic), float(result.pvalue))

    @staticmethod
    def _bootstrap_test(
        a: np.ndarray,
        b: np.ndarray,
        n_bootstrap: int = 10_000,
    ) -> tuple[float, float]:
        """Permutation/bootstrap test for mean difference (one-sided).

        Under H₀, the labels A/B are exchangeable.  We estimate the
        p-value as the fraction of permuted datasets where the mean
        difference exceeds the observed difference.

        Parameters
        ----------
        a, b : np.ndarray
        n_bootstrap : int
            Number of permutation resamples.

        Returns
        -------
        tuple[float, float]
            ``(observed_delta, p_value)``.
        """
        rng = np.random.default_rng(seed=42)
        observed_delta = float(np.mean(b) - np.mean(a))
        combined = np.concatenate([a, b])
        n_a = len(a)
        n_total = len(combined)

        count_ge = 0
        for _ in range(n_bootstrap):
            perm = rng.permutation(combined)
            perm_a = perm[:n_a]
            perm_b = perm[n_a:]
            perm_delta = np.mean(perm_b) - np.mean(perm_a)
            if perm_delta >= observed_delta:
                count_ge += 1

        p_value = (count_ge + 1) / (n_bootstrap + 1)  # +1 for continuity
        return (observed_delta, p_value)

    # ------------------------------------------------------------------
    # Multiple testing correction
    # ------------------------------------------------------------------

    @staticmethod
    def _multiple_testing_correction(
        p_values: list[float],
        method: str = "holm",
    ) -> list[float]:
        """Adjust p-values for multiple comparisons.

        Parameters
        ----------
        p_values : list[float]
            Raw p-values from individual tests.
        method : str
            Correction method:
            - ``"bonferroni"``: Bonferroni correction (p' = m·p)
            - ``"holm"``: Holm–Bonferroni step-down procedure
            - ``"bh"``: Benjamini–Hochberg FDR control

        Returns
        -------
        list[float]
            Adjusted p-values, clipped to [0, 1].

        References
        ----------
        Bonferroni, C. E. (1936). Teoria statistica delle classi.
        Holm, S. (1979). A simple sequentially rejective multiple test procedure.
            *Scandinavian J. Statistics*, 6(2), 65–70.
        Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
            rate. *J. Royal Statistical Society B*, 57(1), 289–300.
        """
        m = len(p_values)
        if m == 0:
            return []

        if method == "bonferroni":
            return [min(p * m, 1.0) for p in p_values]

        elif method == "holm":
            # Holm step-down: order p-values, multiply by (m - rank + 1)
            indexed = sorted(enumerate(p_values), key=lambda x: x[1])
            adjusted = [0.0] * m
            cummax = 0.0
            for rank, (orig_idx, p) in enumerate(indexed):
                adj_p = p * (m - rank)
                cummax = max(cummax, adj_p)
                adjusted[orig_idx] = min(cummax, 1.0)
            return adjusted

        elif method == "bh":
            # Benjamini–Hochberg step-up FDR control
            indexed = sorted(enumerate(p_values), key=lambda x: x[1])
            adjusted = [0.0] * m
            cummin = 1.0
            for rank in range(m - 1, -1, -1):
                orig_idx, p = indexed[rank]
                adj_p = p * m / (rank + 1)
                cummin = min(cummin, adj_p)
                adjusted[orig_idx] = min(cummin, 1.0)
            return adjusted

        else:
            raise ValueError(
                f"Unknown correction method {method!r}; "
                "use 'bonferroni', 'holm', or 'bh'"
            )

    # ------------------------------------------------------------------
    # Effect size with confidence interval
    # ------------------------------------------------------------------

    @staticmethod
    def _effect_size_ci(
        a: np.ndarray,
        b: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[float, float, float]:
        """Compute Cohen's *d* with confidence interval.

        Cohen's *d* is defined as:

        .. math::
            d = \\frac{\\bar{X}_B - \\bar{X}_A}{s_p}

        where the pooled standard deviation is:

        .. math::
            s_p = \\sqrt{\\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}

        The CI uses the non-central t-distribution approximation from
        Hedges & Olkin (1985):

        .. math::
            SE(d) = \\sqrt{\\frac{n_A + n_B}{n_A n_B} + \\frac{d^2}{2(n_A + n_B)}}

        Parameters
        ----------
        a, b : np.ndarray
            Cost samples.
        alpha : float
            Significance level for the CI.

        Returns
        -------
        tuple[float, float, float]
            ``(cohens_d, ci_lower, ci_upper)``.

        References
        ----------
        Hedges, L. V. & Olkin, I. (1985). *Statistical Methods for
        Meta-Analysis*. Academic Press.
        """
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))

        # Pooled standard deviation
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        sp = math.sqrt(max(pooled_var, 1e-15))

        d = (mean_b - mean_a) / sp

        # Standard error of d (Hedges & Olkin approximation)
        se_d = math.sqrt(
            (n_a + n_b) / (n_a * n_b)
            + (d ** 2) / (2 * (n_a + n_b))
        )

        z = sp_stats.norm.ppf(1 - alpha / 2)
        ci_lower = d - z * se_d
        ci_upper = d + z * se_d

        return (d, ci_lower, ci_upper)

    # ------------------------------------------------------------------
    # Batch testing
    # ------------------------------------------------------------------

    def test_multiple(
        self,
        samples_pairs: list[tuple[np.ndarray, np.ndarray]],
        alpha: float = 0.05,
        correction: str = "holm",
    ) -> list[HypothesisResult]:
        """Run multiple hypothesis tests with correction.

        Parameters
        ----------
        samples_pairs : list[tuple[np.ndarray, np.ndarray]]
            List of ``(cost_a, cost_b)`` sample pairs.
        alpha : float
            Family-wise error rate.
        correction : str
            Multiple testing correction method.

        Returns
        -------
        list[HypothesisResult]
            One result per pair, with corrected p-values.
        """
        raw_results = [self.test(a, b, alpha=alpha) for a, b in samples_pairs]
        raw_pvals = [r.p_value for r in raw_results]
        adj_pvals = self._multiple_testing_correction(raw_pvals, correction)

        corrected: list[HypothesisResult] = []
        for result, adj_p in zip(raw_results, adj_pvals):
            corrected.append(HypothesisResult(
                test_statistic=result.test_statistic,
                p_value=adj_p,
                reject_null=adj_p < alpha,
                effect_size=result.effect_size,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                method=f"{result.method}+{correction}",
                n_a=result.n_a,
                n_b=result.n_b,
            ))

        return corrected
