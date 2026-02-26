"""
Statistical testing infrastructure for strategy comparison.

Provides paired bootstrap confidence intervals, Bayesian credible
intervals, effect sizes, multiple comparison corrections, specification
coverage, and power analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.special import comb as sp_comb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Result of a paired bootstrap comparison."""
    observed_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_resamples: int
    confidence: float
    metric_name: str = ""
    significant: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        sig = "YES" if self.significant else "no"
        return (
            f"{self.metric_name}: diff={self.observed_diff:+.6f}  "
            f"CI=[{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]  "
            f"p={self.p_value:.4f}  sig={sig}"
        )


@dataclass
class BayesianResult:
    """Result of a Bayesian comparison."""
    posterior_mean: float
    hdi_lower: float
    hdi_upper: float
    prob_positive: float
    rope_lower: float
    rope_upper: float
    prob_in_rope: float
    metric_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.metric_name}: mean={self.posterior_mean:+.6f}  "
            f"HDI=[{self.hdi_lower:+.6f}, {self.hdi_upper:+.6f}]  "
            f"P(>0)={self.prob_positive:.4f}  P(ROPE)={self.prob_in_rope:.4f}"
        )


@dataclass
class EffectSizeResult:
    """Effect size measures."""
    cohens_d: float
    cliffs_delta: float
    common_language_effect: float
    metric_name: str = ""


@dataclass
class CorrectedPValues:
    """Result of multiple-comparison correction."""
    original_pvalues: NDArray[np.float64]
    corrected_pvalues: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    method: str
    alpha: float
    n_rejected: int


@dataclass
class PowerAnalysisResult:
    """Power analysis result."""
    required_n: int
    achieved_power: float
    effect_size: float
    alpha: float
    target_power: float


# ---------------------------------------------------------------------------
# Statistical test suite
# ---------------------------------------------------------------------------

class StatisticalTestSuite:
    """Comprehensive statistical testing for strategy evaluation.

    Provides bootstrap CIs, Bayesian credible intervals, effect sizes,
    multiple comparison corrections, and power analysis.

    Parameters
    ----------
    n_resamples : number of bootstrap resamples (default 10_000)
    confidence : confidence level for intervals (default 0.95)
    seed : random seed for reproducibility
    """

    def __init__(
        self,
        n_resamples: int = 10_000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> None:
        self._n_resamples = n_resamples
        self._confidence = confidence
        self._rng = np.random.default_rng(seed)
        self._results: List[BootstrapResult] = []

    # ---- Paired bootstrap CI -----------------------------------------------

    def bootstrap_ci(
        self,
        metric1: NDArray[np.float64],
        metric2: NDArray[np.float64],
        metric_name: str = "",
        statistic: str = "mean",
        confidence: Optional[float] = None,
    ) -> BootstrapResult:
        """Paired bootstrap confidence interval for the difference.

        Tests H0: statistic(metric1) == statistic(metric2).

        Parameters
        ----------
        metric1 : per-observation metric values for strategy A
        metric2 : per-observation metric values for strategy B
        metric_name : label for reporting
        statistic : "mean", "median", or "sharpe"
        confidence : override for suite-level confidence

        Returns
        -------
        BootstrapResult
        """
        m1 = np.asarray(metric1, dtype=np.float64).ravel()
        m2 = np.asarray(metric2, dtype=np.float64).ravel()
        n = min(len(m1), len(m2))
        m1, m2 = m1[:n], m2[:n]

        conf = confidence or self._confidence
        stat_fn = self._get_stat_fn(statistic)

        observed = stat_fn(m1) - stat_fn(m2)

        boot_diffs = np.empty(self._n_resamples, dtype=np.float64)
        for b in range(self._n_resamples):
            idx = self._rng.integers(0, n, size=n)
            boot_diffs[b] = stat_fn(m1[idx]) - stat_fn(m2[idx])

        alpha = (1 - conf) / 2.0
        ci_lo = float(np.percentile(boot_diffs, 100 * alpha))
        ci_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha)))

        # Two-sided p-value: fraction of bootstrap samples on opposite side of 0
        if observed >= 0:
            p_val = float(np.mean(boot_diffs <= 0)) * 2
        else:
            p_val = float(np.mean(boot_diffs >= 0)) * 2
        p_val = min(p_val, 1.0)

        significant = p_val < (1 - conf)

        result = BootstrapResult(
            observed_diff=float(observed),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_val,
            n_resamples=self._n_resamples,
            confidence=conf,
            metric_name=metric_name,
            significant=significant,
        )
        self._results.append(result)
        return result

    def bootstrap_single(
        self,
        metric: NDArray[np.float64],
        metric_name: str = "",
        statistic: str = "mean",
        confidence: Optional[float] = None,
    ) -> BootstrapResult:
        """Bootstrap CI for a single sample statistic.

        Parameters
        ----------
        metric : per-observation metric values
        metric_name : label
        statistic : "mean", "median", or "sharpe"

        Returns
        -------
        BootstrapResult with observed_diff as the point estimate
        """
        m = np.asarray(metric, dtype=np.float64).ravel()
        n = len(m)
        conf = confidence or self._confidence
        stat_fn = self._get_stat_fn(statistic)

        observed = stat_fn(m)

        boot_vals = np.empty(self._n_resamples, dtype=np.float64)
        for b in range(self._n_resamples):
            idx = self._rng.integers(0, n, size=n)
            boot_vals[b] = stat_fn(m[idx])

        alpha = (1 - conf) / 2.0
        ci_lo = float(np.percentile(boot_vals, 100 * alpha))
        ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))

        # p-value for mean > 0
        p_val = float(np.mean(boot_vals <= 0))
        p_val = min(2 * p_val, 1.0)

        return BootstrapResult(
            observed_diff=float(observed),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_val,
            n_resamples=self._n_resamples,
            confidence=conf,
            metric_name=metric_name,
            significant=p_val < (1 - conf),
        )

    # ---- Bayesian credible intervals (conjugate normal) --------------------

    def bayesian_credible_interval(
        self,
        metric1: NDArray[np.float64],
        metric2: NDArray[np.float64],
        rope: Tuple[float, float] = (-0.01, 0.01),
        n_posterior_samples: int = 50_000,
        metric_name: str = "",
    ) -> BayesianResult:
        """Bayesian comparison using a conjugate normal model.

        Uses a non-informative prior (Jeffreys) for the mean difference
        of paired observations.

        Parameters
        ----------
        metric1, metric2 : per-observation metrics
        rope : Region Of Practical Equivalence
        n_posterior_samples : number of draws from posterior
        metric_name : label

        Returns
        -------
        BayesianResult
        """
        m1 = np.asarray(metric1, dtype=np.float64).ravel()
        m2 = np.asarray(metric2, dtype=np.float64).ravel()
        n = min(len(m1), len(m2))
        diff = m1[:n] - m2[:n]

        mean_d = float(np.mean(diff))
        std_d = float(np.std(diff, ddof=1))
        se = std_d / np.sqrt(n)

        # Posterior: t-distribution with df = n-1
        posterior = sp_stats.t(df=n - 1, loc=mean_d, scale=se)
        samples = posterior.rvs(size=n_posterior_samples, random_state=self._rng)

        # HDI (Highest Density Interval) via percentile-based approach
        alpha = 1 - self._confidence
        hdi_lo, hdi_hi = self._compute_hdi(samples, alpha)

        prob_pos = float(np.mean(samples > 0))
        prob_rope = float(np.mean((samples >= rope[0]) & (samples <= rope[1])))

        return BayesianResult(
            posterior_mean=float(np.mean(samples)),
            hdi_lower=hdi_lo,
            hdi_upper=hdi_hi,
            prob_positive=prob_pos,
            rope_lower=rope[0],
            rope_upper=rope[1],
            prob_in_rope=prob_rope,
            metric_name=metric_name,
        )

    @staticmethod
    def _compute_hdi(
        samples: NDArray[np.float64],
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute the Highest Density Interval for an array of samples."""
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        ci_size = int(np.ceil((1 - alpha) * n))
        if ci_size >= n:
            return float(sorted_s[0]), float(sorted_s[-1])

        widths = sorted_s[ci_size:] - sorted_s[: n - ci_size]
        best_idx = int(np.argmin(widths))
        return float(sorted_s[best_idx]), float(sorted_s[best_idx + ci_size])

    # ---- Effect sizes ------------------------------------------------------

    def effect_sizes(
        self,
        metric1: NDArray[np.float64],
        metric2: NDArray[np.float64],
        metric_name: str = "",
    ) -> EffectSizeResult:
        """Compute Cohen's d and Cliff's delta.

        Parameters
        ----------
        metric1, metric2 : sample arrays

        Returns
        -------
        EffectSizeResult
        """
        m1 = np.asarray(metric1, dtype=np.float64).ravel()
        m2 = np.asarray(metric2, dtype=np.float64).ravel()

        cohens_d = self._cohens_d(m1, m2)
        cliffs = self._cliffs_delta(m1, m2)
        cle = self._common_language_effect(m1, m2)

        return EffectSizeResult(
            cohens_d=cohens_d,
            cliffs_delta=cliffs,
            common_language_effect=cle,
            metric_name=metric_name,
        )

    @staticmethod
    def _cohens_d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Cohen's d with pooled standard deviation."""
        na, nb = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))
        if pooled_std < 1e-12:
            return 0.0
        return float((mean_a - mean_b) / pooled_std)

    @staticmethod
    def _cliffs_delta(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Cliff's delta: non-parametric effect size.

        δ = (#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)
        """
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return 0.0
        # Vectorised comparison
        comparisons = np.sign(a[:, None] - b[None, :])
        return float(np.mean(comparisons))

    @staticmethod
    def _common_language_effect(
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> float:
        """Common Language Effect Size (probability of superiority)."""
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return 0.5
        count = 0
        total = na * nb
        for ai in a:
            count += int(np.sum(ai > b))
            count += int(np.sum(ai == b)) / 2
        return float(count / total)

    # ---- Multiple comparison correction ------------------------------------

    @staticmethod
    def correct_pvalues(
        pvalues: NDArray[np.float64],
        method: str = "holm",
        alpha: float = 0.05,
    ) -> CorrectedPValues:
        """Apply multiple comparison correction.

        Parameters
        ----------
        pvalues : array of raw p-values
        method : "holm" (Holm-Bonferroni) or "bh" (Benjamini-Hochberg)
        alpha : family-wise error rate

        Returns
        -------
        CorrectedPValues
        """
        pv = np.asarray(pvalues, dtype=np.float64).ravel()
        m = len(pv)

        if method == "holm":
            corrected = StatisticalTestSuite._holm_bonferroni(pv)
        elif method == "bh":
            corrected = StatisticalTestSuite._benjamini_hochberg(pv)
        else:
            raise ValueError(f"Unknown method: {method}")

        rejected = corrected <= alpha

        return CorrectedPValues(
            original_pvalues=pv,
            corrected_pvalues=corrected,
            rejected=rejected,
            method=method,
            alpha=alpha,
            n_rejected=int(np.sum(rejected)),
        )

    @staticmethod
    def _holm_bonferroni(pvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Holm-Bonferroni step-down correction."""
        m = len(pvalues)
        order = np.argsort(pvalues)
        corrected = np.empty(m, dtype=np.float64)
        cummax = 0.0
        for rank, idx in enumerate(order):
            adjusted = pvalues[idx] * (m - rank)
            cummax = max(cummax, adjusted)
            corrected[idx] = min(cummax, 1.0)
        return corrected

    @staticmethod
    def _benjamini_hochberg(pvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Benjamini-Hochberg step-up correction (FDR control)."""
        m = len(pvalues)
        order = np.argsort(pvalues)
        corrected = np.empty(m, dtype=np.float64)

        # Step up from largest to smallest
        cummin = 1.0
        for rank_idx in range(m - 1, -1, -1):
            idx = order[rank_idx]
            adjusted = pvalues[idx] * m / (rank_idx + 1)
            cummin = min(cummin, adjusted)
            corrected[idx] = min(cummin, 1.0)
        return corrected

    # ---- Specification coverage --------------------------------------------

    @staticmethod
    def specification_coverage(
        visited_states: NDArray,
        safety_relevant_states: NDArray,
    ) -> Dict[str, float]:
        """Fraction of safety-relevant states visited during evaluation.

        Parameters
        ----------
        visited_states : (N, D) array of states visited
        safety_relevant_states : (M, D) array of states that matter for safety

        Returns
        -------
        dict with coverage, n_visited_relevant, n_total_relevant
        """
        visited = np.asarray(visited_states, dtype=np.float64)
        relevant = np.asarray(safety_relevant_states, dtype=np.float64)

        if visited.ndim == 1:
            visited = visited.reshape(-1, 1)
        if relevant.ndim == 1:
            relevant = relevant.reshape(-1, 1)

        M = relevant.shape[0]
        if M == 0:
            return {"coverage": 1.0, "n_visited_relevant": 0, "n_total_relevant": 0}

        # For each safety-relevant state, check if any visited state is "close"
        # using L2 distance with adaptive threshold (median nearest-neighbor distance)
        if visited.shape[0] == 0:
            return {"coverage": 0.0, "n_visited_relevant": 0, "n_total_relevant": M}

        # Compute pairwise distances between relevant and visited
        # Use broadcasting: (M, 1, D) - (1, N, D) → (M, N, D)
        # For memory, batch if needed
        N = visited.shape[0]
        threshold = _adaptive_threshold(visited)

        n_covered = 0
        batch_size = 1000
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            chunk = relevant[start:end]  # (chunk_size, D)
            # (chunk_size, N)
            dists = np.sqrt(
                np.sum((chunk[:, None, :] - visited[None, :, :]) ** 2, axis=2)
            )
            min_dists = np.min(dists, axis=1)  # (chunk_size,)
            n_covered += int(np.sum(min_dists <= threshold))

        return {
            "coverage": n_covered / M,
            "n_visited_relevant": n_covered,
            "n_total_relevant": M,
        }

    # ---- Power analysis ----------------------------------------------------

    def power_analysis(
        self,
        metric1: NDArray[np.float64],
        metric2: NDArray[np.float64],
        alpha: float = 0.05,
        target_power: float = 0.80,
    ) -> PowerAnalysisResult:
        """Estimate required sample size for given power.

        Uses the observed effect size to estimate the sample size needed
        to detect the difference at the specified power level.

        Parameters
        ----------
        metric1, metric2 : pilot sample arrays
        alpha : significance level
        target_power : desired power (1 - β)

        Returns
        -------
        PowerAnalysisResult
        """
        m1 = np.asarray(metric1, dtype=np.float64).ravel()
        m2 = np.asarray(metric2, dtype=np.float64).ravel()
        d = abs(self._cohens_d(m1, m2))
        if d < 1e-6:
            return PowerAnalysisResult(
                required_n=999999,
                achieved_power=0.0,
                effect_size=d,
                alpha=alpha,
                target_power=target_power,
            )

        # Binary search for required n
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(target_power)

        # Closed-form for two-sample t-test
        n_required = int(np.ceil(2 * ((z_alpha + z_beta) / d) ** 2))
        n_required = max(n_required, 2)

        # Verify achieved power at the current sample size
        n_current = min(len(m1), len(m2))
        achieved = self._compute_power(d, n_current, alpha)

        return PowerAnalysisResult(
            required_n=n_required,
            achieved_power=achieved,
            effect_size=d,
            alpha=alpha,
            target_power=target_power,
        )

    @staticmethod
    def _compute_power(
        effect_size: float,
        n: int,
        alpha: float,
    ) -> float:
        """Compute statistical power for a two-sample t-test."""
        if n < 2 or effect_size < 1e-12:
            return 0.0
        se = np.sqrt(2.0 / n)
        ncp = effect_size / se  # non-centrality parameter
        crit = sp_stats.t.ppf(1 - alpha / 2, df=2 * n - 2)
        # Power = P(|T_ncp| > crit)
        power = 1.0 - sp_stats.nct.cdf(crit, df=2 * n - 2, nc=ncp) + \
                sp_stats.nct.cdf(-crit, df=2 * n - 2, nc=ncp)
        return float(power)

    # ---- Reporting ---------------------------------------------------------

    def report(self) -> str:
        """Generate a summary report of all bootstrap tests run so far."""
        if not self._results:
            return "No tests run."
        lines = ["=== Statistical Test Report ==="]
        for r in self._results:
            lines.append(r.summary())

        # Multiple comparison correction on collected p-values
        pvals = np.array([r.p_value for r in self._results], dtype=np.float64)
        if len(pvals) > 1:
            corr = self.correct_pvalues(pvals, method="holm")
            lines.append(f"\nHolm-Bonferroni correction: {corr.n_rejected}/{len(pvals)} rejected")
            for i, r in enumerate(self._results):
                lines.append(
                    f"  {r.metric_name}: raw_p={r.p_value:.4f} → adj_p={corr.corrected_pvalues[i]:.4f}"
                )
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear stored results."""
        self._results = []

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _get_stat_fn(name: str):
        """Return a function that computes the named statistic."""
        if name == "mean":
            return lambda x: float(np.mean(x))
        elif name == "median":
            return lambda x: float(np.median(x))
        elif name == "sharpe":
            def _sharpe(x):
                mu = np.mean(x)
                sigma = np.std(x, ddof=1)
                return float(mu / max(sigma, 1e-12))
            return _sharpe
        else:
            raise ValueError(f"Unknown statistic: {name}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _adaptive_threshold(visited: NDArray[np.float64]) -> float:
    """Compute an adaptive distance threshold based on the data distribution.

    Uses the median of k-nearest-neighbor distances (k=5) as the threshold.
    """
    N = visited.shape[0]
    if N < 2:
        return 1.0
    k = min(5, N - 1)
    # Sample for efficiency
    sample_size = min(N, 500)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    sample = visited[sample_idx]

    dists = np.sqrt(
        np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(dists, np.inf)
    knn_dists = np.sort(dists, axis=1)[:, :k]
    return float(np.median(knn_dists[:, -1])) * 2.0
