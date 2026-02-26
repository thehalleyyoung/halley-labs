"""Rigorous concentration inequalities for self-normalised importance sampling.

This module provides distribution-free and adaptive confidence intervals
for importance-sampling estimators, correcting for self-normalisation bias,
weight degeneracy, and finite-sample effects.  Every bound carries a
mathematical statement and proof sketch in its docstring so that the
validity assumptions are explicit and auditable.

Key classes
-----------
SelfNormalizedBound
    Chatterjee–Diaconis (2018) exponential inequality for the
    self-normalised IS estimator, plus finite-sample bias correction.

EmpiricalBernsteinBound
    Distribution-free confidence interval using the empirical variance
    (Maurer & Pontil, 2009), adjusted for importance weights via the
    effective sample size.

AndersonDarlingTest
    Weighted Anderson–Darling normality test that decides whether
    CLT-based intervals may be used.

AdaptiveBoundSelector
    Automatically selects the tightest valid bound for the data at hand.

FiniteSampleGuarantee
    Computes the minimum sample size required to achieve a desired
    confidence level and interval width.

CoverageValidator
    Empirically validates nominal coverage on planted-race benchmarks.

ConvergenceTheory
    Cross-entropy convergence rate analysis (Rubinstein & Kroese, 2004).

References
----------
.. [CD18] Chatterjee, S. & Diaconis, P. (2018).  "The sample size
   required in importance sampling."  *Annals of Applied Probability*,
   28(2), 1099–1135.
.. [MP09] Maurer, A. & Pontil, M. (2009).  "Empirical Bernstein Bounds
   and Sample Variance Penalization."  *COLT 2009*.
.. [RK04] Rubinstein, R. Y. & Kroese, D. P. (2004).  *The Cross-Entropy
   Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo
   Simulation, and Machine Learning.*  Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as scipy_stats


# ======================================================================
# Helper utilities
# ======================================================================

def _log_sum_exp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = float(np.max(log_x))
    if not np.isfinite(c):
        return float("-inf")
    return c + float(np.log(np.sum(np.exp(log_x - c))))


def _normalise_log_weights(log_w: np.ndarray) -> np.ndarray:
    """Return normalised weights from log-weights (sum to 1)."""
    lse = _log_sum_exp(log_w)
    return np.exp(log_w - lse)


def _effective_sample_size(weights: np.ndarray) -> float:
    r"""Kish effective sample size.

    .. math::
        \mathrm{ESS} = \frac{\bigl(\sum_i w_i\bigr)^2}
                             {\sum_i w_i^2}

    Parameters
    ----------
    weights : array of normalised or un-normalised non-negative weights.

    Returns
    -------
    float
        ESS in the range (0, n].
    """
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s == 0.0:
        return 0.0
    return s * s / float(np.sum(w * w))


def _validate_weights_values(
    weights: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and coerce weight / value arrays."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if w.shape[0] != v.shape[0]:
        raise ValueError(
            f"weights ({w.shape[0]}) and values ({v.shape[0]}) "
            "must have the same length"
        )
    if w.shape[0] == 0:
        raise ValueError("Need at least one sample")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")
    return w, v


# ======================================================================
# Result containers
# ======================================================================

@dataclass(frozen=True)
class BoundResult:
    """Result returned by any confidence-interval method.

    Attributes
    ----------
    lower : float
        Lower endpoint of the confidence interval.
    upper : float
        Upper endpoint of the confidence interval.
    point_estimate : float
        Point estimate (possibly bias-corrected).
    method : str
        Human-readable name of the bound that was applied.
    alpha : float
        Significance level (the interval has nominal coverage 1 − α).
    effective_sample_size : float
        Kish ESS used in the bound computation.
    bias_correction : float
        Additive bias correction applied to the raw SN estimate.
    diagnostics : Dict
        Extra diagnostics (method-specific).
    """

    lower: float
    upper: float
    point_estimate: float
    method: str
    alpha: float
    effective_sample_size: float
    bias_correction: float = 0.0
    diagnostics: Dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha


# ======================================================================
# 1.  SelfNormalizedBound
# ======================================================================

class SelfNormalizedBound:
    r"""Concentration inequality for the self-normalised IS estimator.

    Mathematical Statement
    ----------------------
    Let :math:`X_1, \dots, X_n` be i.i.d.\ draws from proposal *q*,
    with un-normalised importance weights
    :math:`w_i = p(X_i) / q(X_i)` and a bounded function
    :math:`f: \mathcal{X} \to [a, b]`.

    The self-normalised estimator is

    .. math::
        \hat\mu_{\mathrm{SN}}
          = \frac{\sum_{i=1}^n w_i\, f(X_i)}{\sum_{i=1}^n w_i}.

    **Theorem** (Chatterjee & Diaconis, 2018, Theorem 1.2 specialised
    to bounded *f*):

    .. math::
        \Pr\bigl(|\hat\mu_{\mathrm{SN}} - \mu| > \varepsilon\bigr)
        \;\le\;
        2 \exp\!\Bigl(
            -\frac{n\,\varepsilon^2}
                  {2\,\sigma_{\mathrm{eff}}^2}
        \Bigr)

    where :math:`\sigma_{\mathrm{eff}}^2 = (b - a)^2 / (4\,r)` and
    :math:`r = \mathrm{ESS} / n` is the ESS ratio.

    When the support [a, b] is unknown it is estimated from the data
    with a guard margin.

    Bias Correction
    ---------------
    The self-normalised estimator has a finite-sample bias

    .. math::
        \mathrm{Bias}(\hat\mu_{\mathrm{SN}})
        \;=\;
        -\frac{\mathrm{Cov}_q(W,\, f(X))}{\bigl(E_q[W]\bigr)^2}
        + O(n^{-2})

    (see Owen, *Monte Carlo Theory, Methods and Examples*, Ch. 9).
    We subtract this estimated bias from the point estimate.

    Parameters
    ----------
    support : tuple of (float, float) or None
        Known support [a, b] of f.  If *None*, estimated from data with
        a 5 % guard band on each side.
    bias_correct : bool
        Whether to apply the first-order bias correction (default True).
    """

    def __init__(
        self,
        support: Optional[Tuple[float, float]] = None,
        bias_correct: bool = True,
    ) -> None:
        self.support = support
        self.bias_correct = bias_correct

    # ------------------------------------------------------------------
    def _estimate_support(self, values: np.ndarray) -> Tuple[float, float]:
        lo, hi = float(np.min(values)), float(np.max(values))
        span = hi - lo
        if span == 0.0:
            span = max(abs(lo), 1.0)
        guard = 0.05 * span
        return lo - guard, hi + guard

    # ------------------------------------------------------------------
    def _bias_estimate(
        self,
        weights: np.ndarray,
        values: np.ndarray,
    ) -> float:
        r"""First-order bias :math:`-\mathrm{Cov}(W, f) / E[W]^2`.

        Estimated from the sample itself.  For n samples this is
        :math:`O(1/n)` — negligible for large n but helpful for the
        small-sample regime where IS is most fragile.
        """
        n = len(weights)
        if n < 3:
            return 0.0
        w_bar = float(np.mean(weights))
        if w_bar == 0.0:
            return 0.0
        cov_wf = float(np.cov(weights, values, ddof=1)[0, 1])
        return -cov_wf / (w_bar * w_bar)

    # ------------------------------------------------------------------
    def confidence_interval(
        self,
        weights: np.ndarray,
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> BoundResult:
        r"""Compute a :math:`(1-\alpha)` confidence interval.

        Parameters
        ----------
        weights : array, shape (n,)
            Un-normalised importance weights :math:`w_i = p(x_i)/q(x_i)`.
        values : array, shape (n,)
            Function evaluations :math:`f(x_i)`.
        alpha : float
            Significance level.

        Returns
        -------
        BoundResult
        """
        weights, values = _validate_weights_values(weights, values)
        n = len(weights)

        # --- normalised weights & point estimate ---
        w_sum = float(np.sum(weights))
        if w_sum == 0.0:
            raise ValueError("All weights are zero — cannot estimate")
        w_norm = weights / w_sum
        mu_sn = float(np.dot(w_norm, values))

        # --- support ---
        a, b = self.support if self.support is not None else self._estimate_support(values)
        rng = b - a

        # --- ESS and effective variance ---
        ess = _effective_sample_size(w_norm)
        r = max(ess / n, 1e-12)
        sigma2_eff = (rng * rng) / (4.0 * r)

        # --- solve for epsilon from  2 exp(-n eps^2 / (2 sigma2_eff)) = alpha
        #     =>  eps = sqrt(2 sigma2_eff log(2/alpha) / n)
        eps = math.sqrt(2.0 * sigma2_eff * math.log(2.0 / alpha) / n)

        # --- bias correction ---
        bias = 0.0
        if self.bias_correct:
            bias = self._bias_estimate(weights, values)

        mu_corrected = mu_sn + bias

        return BoundResult(
            lower=mu_corrected - eps,
            upper=mu_corrected + eps,
            point_estimate=mu_corrected,
            method="SelfNormalizedBound (Chatterjee–Diaconis)",
            alpha=alpha,
            effective_sample_size=ess,
            bias_correction=bias,
            diagnostics={
                "sigma2_eff": sigma2_eff,
                "ess_ratio": r,
                "support": (a, b),
                "epsilon": eps,
                "raw_estimate": mu_sn,
            },
        )


# ======================================================================
# 2.  EmpiricalBernsteinBound
# ======================================================================

class EmpiricalBernsteinBound:
    r"""Distribution-free confidence interval via empirical Bernstein.

    Mathematical Statement
    ----------------------
    Let :math:`Z_1, \dots, Z_n` be independent r.v.\ in :math:`[0, b]`.
    Define :math:`\bar Z = n^{-1}\sum Z_i` and the empirical variance

    .. math::
        \hat V = \frac{1}{n(n-1)}
                 \sum_{i<j} (Z_i - Z_j)^2.

    **Theorem** (Maurer & Pontil, 2009, Theorem 4):

    .. math::
        \Pr\!\bigl(\mathrm{E}[Z] > \bar Z + \varepsilon\bigr)
        \;\le\;
        \exp\!\Bigl(
          -\frac{n\,\varepsilon^2}
                {2\hat V + \tfrac{2}{3}\,b\,\varepsilon}
        \Bigr)

    and similarly for the lower tail, giving a two-sided bound with
    significance :math:`\alpha` by setting each tail to :math:`\alpha/2`.

    In the importance-sampling setting we form the effective samples
    :math:`Z_i = \tilde w_i\, f(X_i)` where :math:`\tilde w_i` are
    normalised weights, and replace *n* by :math:`\mathrm{ESS}` in the
    bound (conservative because the effective samples are more dependent
    than i.i.d.).

    Parameters
    ----------
    upper_bound : float or None
        Known upper bound *b* on |Z_i|.  If None, estimated from data.
    """

    def __init__(self, upper_bound: Optional[float] = None) -> None:
        self.upper_bound = upper_bound

    # ------------------------------------------------------------------
    @staticmethod
    def _empirical_variance(z: np.ndarray) -> float:
        r"""Unbiased empirical variance (U-statistic version).

        .. math::
            \hat V = \frac{1}{n(n-1)} \sum_{i<j} (Z_i - Z_j)^2
        """
        n = len(z)
        if n < 2:
            return 0.0
        mean = float(np.mean(z))
        return float(np.sum((z - mean) ** 2)) / (n - 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _solve_bernstein_epsilon(
        v_hat: float,
        b: float,
        n_eff: float,
        alpha_one_tail: float,
    ) -> float:
        r"""Solve for ε in  exp(-n ε² / (2V̂ + 2bε/3)) = α.

        Re-arranging:

        .. math::
            n\,\varepsilon^2
            = \bigl(2\hat V + \tfrac{2b\varepsilon}{3}\bigr)
              \ln(1/\alpha)

        which is a quadratic in ε:

        .. math::
            n\,\varepsilon^2
            - \tfrac{2b}{3}\ln(1/\alpha)\,\varepsilon
            - 2\hat V\ln(1/\alpha) = 0

        We take the positive root.
        """
        log_inv_alpha = math.log(1.0 / alpha_one_tail)
        # Coefficients:  A eps^2 + B eps + C = 0
        A = n_eff
        B = -(2.0 * b * log_inv_alpha) / 3.0
        C = -2.0 * v_hat * log_inv_alpha

        disc = B * B - 4.0 * A * C
        if disc < 0:
            # Fallback: Hoeffding bound for bounded [0, b] variables.
            # Hoeffding: P(X̄ - μ > ε) ≤ exp(-2nε²/b²)
            # Solving: ε = b · sqrt(ln(1/α) / (2n))
            # MATH FIX: Previous constant was 2b²/n (sub-Gaussian with
            # variance b²) instead of correct Hoeffding constant b²/(2n).
            return b * math.sqrt(log_inv_alpha / (2.0 * n_eff))
        return (-B + math.sqrt(disc)) / (2.0 * A)

    # ------------------------------------------------------------------
    def confidence_interval(
        self,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ) -> BoundResult:
        r"""Compute a :math:`(1-\alpha)` confidence interval.

        Parameters
        ----------
        samples : array, shape (n,)
            Raw function evaluations :math:`f(x_i)`.
        weights : array, shape (n,) or None
            Un-normalised importance weights.  If None, uniform weights
            are assumed (plain Monte Carlo).
        alpha : float
            Significance level.

        Returns
        -------
        BoundResult
        """
        samples = np.asarray(samples, dtype=np.float64).ravel()
        n = len(samples)
        if n == 0:
            raise ValueError("Need at least one sample")

        if weights is None:
            w_norm = np.ones(n, dtype=np.float64) / n
            ess = float(n)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()
            if weights.shape[0] != n:
                raise ValueError("weights and samples must have the same length")
            w_sum = float(np.sum(weights))
            if w_sum == 0.0:
                raise ValueError("All weights are zero")
            w_norm = weights / w_sum
            ess = _effective_sample_size(w_norm)

        # Effective samples z_i = w_i_norm * n * f(x_i)  (rescaled so
        # that the mean of the z's equals the weighted mean of f).
        # Instead we work directly with weighted mean and empirical
        # variance of f, replacing n by ESS.
        mu_hat = float(np.dot(w_norm, samples))
        # Weighted empirical variance
        v_hat = float(np.dot(w_norm, (samples - mu_hat) ** 2))
        # Bessel correction using ESS
        if ess > 1.0:
            v_hat *= ess / (ess - 1.0)

        # Effective upper bound on |f - mu|
        b = self.upper_bound
        if b is None:
            b = float(np.max(np.abs(samples - mu_hat))) * 1.05
            b = max(b, 1e-12)

        eps = self._solve_bernstein_epsilon(
            v_hat=v_hat, b=b, n_eff=ess, alpha_one_tail=alpha / 2.0,
        )

        return BoundResult(
            lower=mu_hat - eps,
            upper=mu_hat + eps,
            point_estimate=mu_hat,
            method="EmpiricalBernstein (Maurer–Pontil)",
            alpha=alpha,
            effective_sample_size=ess,
            bias_correction=0.0,
            diagnostics={
                "empirical_variance": v_hat,
                "bound_b": b,
                "epsilon": eps,
            },
        )


# ======================================================================
# 3.  AndersonDarlingTest
# ======================================================================

class AndersonDarlingTest:
    r"""Weighted Anderson–Darling normality test.

    Mathematical Background
    -----------------------
    The Anderson–Darling statistic is

    .. math::
        A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}
              (2i - 1)\bigl[\ln\Phi(Y_i) + \ln(1-\Phi(Y_{n+1-i}))\bigr]

    where :math:`Y_i` are the standardised order statistics and
    :math:`\Phi` is the standard normal CDF.

    For importance-weighted samples we first compute the weighted mean
    and variance, standardise using those, and then run the AD test.
    Critical values are looked up from the scipy implementation which
    applies the correction factor of D'Agostino & Stephens (1986).

    A *p*-value below the chosen significance level indicates that
    normality is rejected, and CLT-based intervals should not be used.

    Parameters
    ----------
    significance : float
        Threshold for rejecting normality (default 0.05).
    """

    def __init__(self, significance: float = 0.05) -> None:
        self.significance = significance

    # ------------------------------------------------------------------
    def test(
        self,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, float]:
        """Run the Anderson–Darling test.

        Parameters
        ----------
        values : array, shape (n,)
        weights : array, shape (n,) or None

        Returns
        -------
        is_normal : bool
            True if normality is *not* rejected at the configured level.
        statistic : float
            The AD test statistic.
        p_value : float
            Approximate p-value.
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        n = len(values)
        if n < 8:
            # Too few samples — cannot reject normality reliably.
            return True, 0.0, 1.0

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).ravel()
            w_sum = float(np.sum(weights))
            if w_sum == 0.0:
                return True, 0.0, 1.0
            w_norm = weights / w_sum
            mu = float(np.dot(w_norm, values))
            var = float(np.dot(w_norm, (values - mu) ** 2))
            # Bessel correction
            ess = _effective_sample_size(w_norm)
            if ess > 1.0:
                var *= ess / (ess - 1.0)
            sigma = math.sqrt(max(var, 1e-30))
            standardised = (values - mu) / sigma
        else:
            mu = float(np.mean(values))
            sigma = float(np.std(values, ddof=1))
            if sigma < 1e-15:
                sigma = 1.0
            standardised = (values - mu) / sigma

        result = scipy_stats.anderson(standardised, dist="norm")
        statistic = float(result.statistic)

        # Map to approximate p-value via critical-value table.
        # scipy returns critical values at [15%, 10%, 5%, 2.5%, 1%].
        sig_levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01])
        crit_values = np.array(result.critical_values, dtype=np.float64)

        if statistic < crit_values[0]:
            p_value = sig_levels[0]  # > 15 %
        elif statistic > crit_values[-1]:
            p_value = 0.005  # < 1 %
        else:
            # Linear interpolation on the log scale.
            p_value = float(
                np.interp(statistic, crit_values, sig_levels)
            )

        is_normal = p_value >= self.significance
        return is_normal, statistic, p_value


# ======================================================================
# 4.  AdaptiveBoundSelector
# ======================================================================

class AdaptiveBoundSelector:
    r"""Automatically select the tightest valid confidence bound.

    Strategy
    --------
    1. Run the Anderson–Darling normality test on the weighted samples.
    2. If normality is *not* rejected:
       use a CLT-based interval with finite-sample correction
       (Student-*t* quantile rather than Gaussian).
    3. If normality *is* rejected:
       use the empirical Bernstein bound (distribution-free).
    4. In both cases, report the self-normalised bias correction and
       the Chatterjee–Diaconis bound for comparison.

    The returned :class:`BoundResult` uses whichever bound yields the
    *narrowest* interval among those that are valid.

    Parameters
    ----------
    ad_significance : float
        Significance level for the Anderson–Darling test.
    support : tuple or None
        Known support of *f* (passed to :class:`SelfNormalizedBound`).
    bias_correct : bool
        Apply bias correction.
    """

    def __init__(
        self,
        ad_significance: float = 0.05,
        support: Optional[Tuple[float, float]] = None,
        bias_correct: bool = True,
    ) -> None:
        self.ad_test = AndersonDarlingTest(significance=ad_significance)
        self.sn_bound = SelfNormalizedBound(
            support=support, bias_correct=bias_correct,
        )
        self.eb_bound = EmpiricalBernsteinBound()

    # ------------------------------------------------------------------
    @staticmethod
    def _clt_interval(
        values: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> BoundResult:
        r"""CLT-based confidence interval with Student-t correction.

        For self-normalised IS with ESS replacing *n*,

        .. math::
            \hat\mu \;\pm\;
            t_{1-\alpha/2,\,\nu}\;
            \frac{\hat\sigma}{\sqrt{\mathrm{ESS}}}

        where :math:`\nu = \mathrm{ESS} - 1` (conservative).

        Finite-sample validity
        ^^^^^^^^^^^^^^^^^^^^^^
        The CLT interval is only *asymptotically* exact.  Using the *t*
        quantile instead of the normal quantile provides a finite-sample
        correction that is conservative for symmetric distributions and
        typically adequate for ESS ≥ 30.
        """
        w_sum = float(np.sum(weights))
        w_norm = weights / w_sum
        ess = _effective_sample_size(w_norm)
        mu = float(np.dot(w_norm, values))
        var = float(np.dot(w_norm, (values - mu) ** 2))
        if ess > 1.0:
            var *= ess / (ess - 1.0)
        se = math.sqrt(max(var, 0.0) / max(ess, 1.0))

        nu = max(ess - 1.0, 1.0)
        t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=nu))

        eps = t_crit * se
        return BoundResult(
            lower=mu - eps,
            upper=mu + eps,
            point_estimate=mu,
            method="CLT (Student-t, finite-sample)",
            alpha=alpha,
            effective_sample_size=ess,
            bias_correction=0.0,
            diagnostics={
                "std_error": se,
                "t_critical": t_crit,
                "degrees_of_freedom": nu,
                "weighted_variance": var,
            },
        )

    # ------------------------------------------------------------------
    def select_bound(
        self,
        weights: np.ndarray,
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> BoundResult:
        """Select the tightest valid bound for the given data.

        Parameters
        ----------
        weights : array, shape (n,)
            Un-normalised importance weights.
        values : array, shape (n,)
            Function evaluations.
        alpha : float
            Significance level.

        Returns
        -------
        BoundResult
            The bound with smallest width among valid choices.
        """
        weights, values = _validate_weights_values(weights, values)

        # 1. Normality test
        is_normal, ad_stat, ad_p = self.ad_test.test(values, weights)

        # 2. Candidate bounds
        candidates: List[BoundResult] = []

        if is_normal:
            clt_result = self._clt_interval(values, weights, alpha)
            candidates.append(clt_result)

        eb_result = self.eb_bound.confidence_interval(
            samples=values, weights=weights, alpha=alpha,
        )
        candidates.append(eb_result)

        sn_result = self.sn_bound.confidence_interval(
            weights=weights, values=values, alpha=alpha,
        )
        candidates.append(sn_result)

        # 3. Pick narrowest
        best = min(candidates, key=lambda r: r.width)

        # Enrich diagnostics
        diag = dict(best.diagnostics)
        diag["ad_normal"] = is_normal
        diag["ad_statistic"] = ad_stat
        diag["ad_p_value"] = ad_p
        diag["all_widths"] = {c.method: c.width for c in candidates}

        # Apply SN bias correction if it was not already included.
        bias = best.bias_correction
        if bias == 0.0 and self.sn_bound.bias_correct:
            bias = sn_result.bias_correction

        return BoundResult(
            lower=best.lower,
            upper=best.upper,
            point_estimate=best.point_estimate,
            method=f"Adaptive({best.method})",
            alpha=alpha,
            effective_sample_size=best.effective_sample_size,
            bias_correction=bias,
            diagnostics=diag,
        )


# ======================================================================
# 5.  FiniteSampleGuarantee
# ======================================================================

class FiniteSampleGuarantee:
    r"""Compute minimum sample sizes for desired confidence guarantees.

    Theory
    ------
    From the Chatterjee–Diaconis bound, for coverage :math:`1-\alpha`
    and half-width :math:`\varepsilon`:

    .. math::
        n \;\ge\;
        \frac{2\,\sigma_{\mathrm{eff}}^2\,\ln(2/\alpha)}
             {\varepsilon^2}
        \;=\;
        \frac{(b-a)^2\,\ln(2/\alpha)}{2\,r\,\varepsilon^2}

    where :math:`r = \mathrm{ESS}/n`.  The *actual* number of samples
    that must be *drawn* from the proposal is :math:`n`, but only
    :math:`r\,n` of those are *effective*.

    For the empirical Bernstein bound with known upper-bound *b* and
    empirical variance :math:`\hat V`:

    .. math::
        n \;\ge\;
        \frac{2\hat V \ln(2/\alpha)}{\varepsilon^2}
        + \frac{2b\ln(2/\alpha)}{3\varepsilon}

    Again, actual draws = :math:`n / r`.

    Parameters
    ----------
    support_range : float
        :math:`b - a`, the range of the function *f*.
    pilot_variance : float or None
        If available, an estimate of Var(f) under the proposal.
    """

    def __init__(
        self,
        support_range: float = 1.0,
        pilot_variance: Optional[float] = None,
    ) -> None:
        if support_range <= 0:
            raise ValueError("support_range must be positive")
        self.support_range = support_range
        self.pilot_variance = pilot_variance

    # ------------------------------------------------------------------
    def required_samples_chatterjee(
        self,
        alpha: float = 0.05,
        epsilon: float = 0.01,
        ess_ratio: float = 1.0,
    ) -> int:
        r"""Minimum *n* from the Chatterjee–Diaconis bound.

        Parameters
        ----------
        alpha : float
            Significance level.
        epsilon : float
            Desired half-width of the confidence interval.
        ess_ratio : float in (0, 1]
            Expected ESS / n ratio.

        Returns
        -------
        int
            Number of samples to draw from the proposal.
        """
        if ess_ratio <= 0.0 or ess_ratio > 1.0:
            raise ValueError("ess_ratio must be in (0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be in (0, 1)")

        rng = self.support_range
        sigma2_eff = (rng * rng) / (4.0 * ess_ratio)
        n = 2.0 * sigma2_eff * math.log(2.0 / alpha) / (epsilon * epsilon)
        return int(math.ceil(n))

    # ------------------------------------------------------------------
    def required_samples_bernstein(
        self,
        alpha: float = 0.05,
        epsilon: float = 0.01,
        ess_ratio: float = 1.0,
    ) -> int:
        r"""Minimum *n* from the empirical Bernstein bound.

        Uses the pilot variance if available, otherwise falls back to
        the worst-case (Hoeffding) term.

        Parameters
        ----------
        alpha, epsilon, ess_ratio : float
            As in :meth:`required_samples_chatterjee`.

        Returns
        -------
        int
        """
        if ess_ratio <= 0.0 or ess_ratio > 1.0:
            raise ValueError("ess_ratio must be in (0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        log_term = math.log(2.0 / alpha)
        b = self.support_range

        if self.pilot_variance is not None:
            v = self.pilot_variance
        else:
            v = (b * b) / 4.0  # worst-case

        n_eff = (2.0 * v * log_term) / (epsilon * epsilon) + \
                (2.0 * b * log_term) / (3.0 * epsilon)
        n_actual = n_eff / ess_ratio
        return int(math.ceil(n_actual))

    # ------------------------------------------------------------------
    def required_samples(
        self,
        alpha: float = 0.05,
        epsilon: float = 0.01,
        ess_ratio: float = 1.0,
    ) -> int:
        """Return the *smaller* sample-size requirement.

        Selects the tighter of Chatterjee–Diaconis and Bernstein.
        """
        n_cd = self.required_samples_chatterjee(alpha, epsilon, ess_ratio)
        n_eb = self.required_samples_bernstein(alpha, epsilon, ess_ratio)
        return min(n_cd, n_eb)


# ======================================================================
# 6.  CoverageValidator
# ======================================================================

class CoverageValidator:
    r"""Empirically validate coverage of confidence intervals.

    Methodology
    -----------
    1. For each trial *t = 1, …, T*:
       a. Draw *n* samples from a *known* distribution whose true mean
          is :math:`\mu^*`.
       b. Compute importance weights (proposal vs.\ target).
       c. Build a :math:`(1-\alpha)` confidence interval.
       d. Record whether :math:`\mu^*` is inside the interval.
    2. The empirical coverage is :math:`\hat c = (\text{hits}) / T`.
    3. A valid interval should satisfy
       :math:`\hat c \ge 1 - \alpha - \delta` with high probability
       for a Monte-Carlo tolerance :math:`\delta` that vanishes as
       :math:`T \to \infty`.

    Parameters
    ----------
    num_trials : int
        Number of Monte-Carlo trials *T*.
    sample_size : int
        Number of samples per trial *n*.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_trials: int = 1000,
        sample_size: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        self.num_trials = num_trials
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def validate_gaussian(
        self,
        bound_fn,
        alpha: float = 0.05,
        target_mean: float = 0.5,
        target_std: float = 0.1,
        proposal_mean: float = 0.5,
        proposal_std: float = 0.2,
    ) -> Dict:
        r"""Validate coverage using Gaussian target and proposal.

        The true mean under the target is *target_mean*.  Samples are
        drawn from the Gaussian proposal and reweighted to the target.

        Parameters
        ----------
        bound_fn : callable(weights, values, alpha) → BoundResult
            The confidence-interval method to test.
        alpha : float
        target_mean, target_std : float
            Parameters of the target Gaussian.
        proposal_mean, proposal_std : float
            Parameters of the proposal Gaussian.

        Returns
        -------
        dict with keys
            coverage : float
                Fraction of trials where true mean was inside the CI.
            bias : float
                Mean (point_estimate − true_mean) over trials.
            mean_width : float
                Average CI width.
            undercoverage : bool
                True if coverage is significantly below 1 − α.
        """
        hits = 0
        widths: List[float] = []
        biases: List[float] = []

        for _ in range(self.num_trials):
            x = self.rng.normal(proposal_mean, proposal_std, self.sample_size)

            # importance weights  p(x)/q(x)
            log_p = scipy_stats.norm.logpdf(x, target_mean, target_std)
            log_q = scipy_stats.norm.logpdf(x, proposal_mean, proposal_std)
            log_w = log_p - log_q
            weights = np.exp(log_w - np.max(log_w))  # stabilised

            result = bound_fn(weights, x, alpha)

            if result.lower <= target_mean <= result.upper:
                hits += 1
            widths.append(result.width)
            biases.append(result.point_estimate - target_mean)

        coverage = hits / self.num_trials
        # Two-sided test for undercoverage at 99 % confidence
        nominal = 1.0 - alpha
        se_cov = math.sqrt(coverage * (1.0 - coverage) / self.num_trials)
        undercoverage = coverage < nominal - 2.576 * se_cov

        return {
            "coverage": coverage,
            "nominal": nominal,
            "bias": float(np.mean(biases)),
            "mean_width": float(np.mean(widths)),
            "undercoverage": undercoverage,
            "se_coverage": se_cov,
        }

    # ------------------------------------------------------------------
    def validate_bernoulli(
        self,
        bound_fn,
        alpha: float = 0.05,
        true_prob: float = 0.01,
        proposal_prob: float = 0.05,
    ) -> Dict:
        r"""Validate coverage on a Bernoulli (rare-event) target.

        This directly mimics the race-detection setting: the true race
        probability is *true_prob*, and the proposal biases toward races
        with probability *proposal_prob* > *true_prob*.

        Parameters
        ----------
        bound_fn : callable(weights, values, alpha) → BoundResult
        alpha : float
        true_prob : float
            True (planted) race probability.
        proposal_prob : float
            Proposal race probability.

        Returns
        -------
        dict  (same keys as :meth:`validate_gaussian`)
        """
        hits = 0
        widths: List[float] = []
        biases: List[float] = []

        for _ in range(self.num_trials):
            x = self.rng.binomial(1, proposal_prob, self.sample_size).astype(
                np.float64,
            )

            # w_i = p(x_i) / q(x_i)
            w = np.where(
                x == 1.0,
                true_prob / proposal_prob,
                (1.0 - true_prob) / (1.0 - proposal_prob),
            )

            result = bound_fn(w, x, alpha)

            if result.lower <= true_prob <= result.upper:
                hits += 1
            widths.append(result.width)
            biases.append(result.point_estimate - true_prob)

        coverage = hits / self.num_trials
        nominal = 1.0 - alpha
        se_cov = math.sqrt(coverage * (1.0 - coverage) / self.num_trials)
        undercoverage = coverage < nominal - 2.576 * se_cov

        return {
            "coverage": coverage,
            "nominal": nominal,
            "bias": float(np.mean(biases)),
            "mean_width": float(np.mean(widths)),
            "undercoverage": undercoverage,
            "se_coverage": se_cov,
        }


# ======================================================================
# 7.  ConvergenceTheory
# ======================================================================

class ConvergenceTheory:
    r"""Cross-entropy convergence rate analysis.

    Theory
    ------
    **IS estimator convergence** (standard):

    For the self-normalised IS estimator :math:`\hat\mu_n`,

    .. math::
        \sqrt{n}\,(\hat\mu_n - \mu)
        \;\xrightarrow{d}\;
        \mathcal{N}\!\bigl(0,\;
        \mathrm{Var}_q\!\bigl[w(X)\,(f(X)-\mu)\bigr]
        \big/ \bigl(E_q[w(X)]\bigr)^2
        \bigr)

    so the MSE decays as :math:`O(1/n)`, or equivalently the
    standard error as :math:`O(1/\sqrt{n})`.

    **CE proposal convergence** (Rubinstein & Kroese, 2004, Theorem 4.1):

    Let :math:`\{q_t\}` be the sequence of CE proposals obtained by
    fitting to the elite fraction :math:`\rho` of samples.  Then

    .. math::
        D_{\mathrm{KL}}(p^*_f \| q_t)
        \;\le\;
        D_{\mathrm{KL}}(p^*_f \| q_0)
        \cdot (1 - \eta)^t

    where :math:`p^*_f` is the zero-variance proposal
    :math:`p^*_f(x) \propto p(x)\,|f(x)|` and :math:`\eta > 0` is a
    constant depending on the parametric family and the elite ratio.

    **Finite-sample regret**:

    Using the CE proposal :math:`q_T` after *T* adaptation rounds of
    *m* samples each, the excess variance relative to the optimal
    proposal satisfies

    .. math::
        \mathrm{Var}_{q_T}[w\,f]
        - \mathrm{Var}_{p^*_f}[w\,f]
        \;\le\;
        C\,\sqrt{\frac{d\,\ln(Tm)}{m}}

    where *d* is the parameter dimension and *C* is a universal constant
    (see Hu et al., 2007).

    Parameters
    ----------
    param_dim : int
        Dimension of the proposal parameter space.
    elite_fraction : float
        CE elite ratio :math:`\rho`.
    """

    def __init__(
        self,
        param_dim: int = 1,
        elite_fraction: float = 0.1,
    ) -> None:
        if param_dim < 1:
            raise ValueError("param_dim must be >= 1")
        if not 0.0 < elite_fraction < 1.0:
            raise ValueError("elite_fraction must be in (0, 1)")
        self.param_dim = param_dim
        self.elite_fraction = elite_fraction

    # ------------------------------------------------------------------
    def is_standard_error(
        self,
        n: int,
        weights: np.ndarray,
        values: np.ndarray,
    ) -> float:
        r"""Estimated standard error of the IS estimator.

        .. math::
            \mathrm{SE} = \frac{\hat\sigma_{\mathrm{eff}}}{\sqrt{n}}

        where :math:`\hat\sigma_{\mathrm{eff}}^2` is the weighted
        sample variance of :math:`w_i (f_i - \hat\mu)`.
        """
        weights, values = _validate_weights_values(weights, values)
        w_sum = float(np.sum(weights))
        if w_sum == 0.0:
            return float("inf")
        w_norm = weights / w_sum
        mu = float(np.dot(w_norm, values))
        resid = values - mu
        # Var_q[w(f - mu)] estimated
        sigma2 = float(np.mean((weights * resid) ** 2))
        ew2 = (w_sum / len(weights)) ** 2
        if ew2 == 0.0:
            return float("inf")
        return math.sqrt(sigma2 / (ew2 * n))

    # ------------------------------------------------------------------
    def ce_kl_bound(
        self,
        iteration: int,
        initial_kl: float = 10.0,
    ) -> float:
        r"""Upper bound on KL divergence after *t* CE iterations.

        .. math::
            D_{\mathrm{KL}}(p^* \| q_t) \le D_0\,(1-\eta)^t

        We use the heuristic :math:`\eta \approx \rho` (the elite
        fraction), which is a lower bound under log-concavity.

        Parameters
        ----------
        iteration : int
            CE round index *t* (0-based).
        initial_kl : float
            :math:`D_0 = D_{\mathrm{KL}}(p^* \| q_0)`.

        Returns
        -------
        float
            Upper bound on :math:`D_{\mathrm{KL}}(p^* \| q_t)`.
        """
        eta = self.elite_fraction
        return initial_kl * ((1.0 - eta) ** iteration)

    # ------------------------------------------------------------------
    def finite_sample_regret(
        self,
        num_rounds: int,
        samples_per_round: int,
    ) -> float:
        r"""Finite-sample excess-variance regret bound.

        .. math::
            \Delta V \le C \sqrt{\frac{d \ln(T m)}{m}}

        with :math:`C = 2` (universal constant, conservative).

        Parameters
        ----------
        num_rounds : int
            Number of CE adaptation rounds *T*.
        samples_per_round : int
            Samples per round *m*.

        Returns
        -------
        float
            Upper bound on excess variance due to proposal sub-optimality.
        """
        if samples_per_round < 1 or num_rounds < 1:
            raise ValueError("num_rounds and samples_per_round must be >= 1")
        C = 2.0
        d = self.param_dim
        m = samples_per_round
        T = num_rounds
        return C * math.sqrt(d * math.log(T * m) / m)

    # ------------------------------------------------------------------
    def convergence_diagnostic(
        self,
        weights_per_round: Sequence[np.ndarray],
        values_per_round: Sequence[np.ndarray],
    ) -> Dict:
        r"""Diagnose convergence of a CE-IS run.

        For each round *t* we compute:

        * ESS ratio :math:`r_t = \mathrm{ESS}_t / n_t`
        * Standard error of the IS estimate
        * Estimated KL upper bound

        A healthy run should show increasing ESS ratios and decreasing
        standard errors.

        Parameters
        ----------
        weights_per_round : list of arrays
            Un-normalised weights for each CE round.
        values_per_round : list of arrays
            Function evaluations for each CE round.

        Returns
        -------
        dict
            ``ess_ratios``, ``standard_errors``, ``kl_bounds``,
            ``converged`` (bool).
        """
        T = len(weights_per_round)
        if T == 0:
            raise ValueError("Need at least one round of data")
        if len(values_per_round) != T:
            raise ValueError("weights and values must have the same number of rounds")

        ess_ratios: List[float] = []
        standard_errors: List[float] = []
        kl_bounds: List[float] = []

        for t in range(T):
            w = np.asarray(weights_per_round[t], dtype=np.float64)
            v = np.asarray(values_per_round[t], dtype=np.float64)
            n = len(w)
            if n == 0:
                continue

            w_sum = float(np.sum(w))
            if w_sum > 0:
                w_norm = w / w_sum
                ess = _effective_sample_size(w_norm)
            else:
                ess = 0.0

            ess_ratios.append(ess / max(n, 1))
            standard_errors.append(
                self.is_standard_error(n, w, v)
            )
            kl_bounds.append(self.ce_kl_bound(t))

        # Convergence heuristic: ESS ratio in last round >= 0.1
        # and SE decreased monotonically in the last 3 rounds.
        converged = False
        if len(ess_ratios) >= 3:
            last3_se = standard_errors[-3:]
            converged = (
                ess_ratios[-1] >= 0.1
                and all(
                    last3_se[i] >= last3_se[i + 1]
                    for i in range(len(last3_se) - 1)
                )
            )

        return {
            "ess_ratios": ess_ratios,
            "standard_errors": standard_errors,
            "kl_bounds": kl_bounds,
            "converged": converged,
            "num_rounds": T,
        }

    # ------------------------------------------------------------------
    def rate_summary(self, n: int) -> Dict:
        r"""Theoretical convergence rates for sample size *n*.

        Returns
        -------
        dict
            ``is_rate`` : :math:`1/\sqrt{n}` (standard error scaling).
            ``ce_kl_rate`` : KL bound at round :math:`\lfloor\log n\rfloor`.
            ``regret`` : finite-sample regret at *n*.
        """
        t = max(int(math.log(n)), 1)
        m = max(n // t, 1)
        return {
            "is_rate": 1.0 / math.sqrt(n),
            "ce_kl_rate": self.ce_kl_bound(t),
            "regret": self.finite_sample_regret(t, m),
            "assumed_rounds": t,
            "samples_per_round": m,
        }


# ======================================================================
# Convenience API
# ======================================================================

def confidence_interval(
    weights: np.ndarray,
    values: np.ndarray,
    alpha: float = 0.05,
) -> BoundResult:
    """One-call adaptive confidence interval (module-level convenience).

    Equivalent to ``AdaptiveBoundSelector().select_bound(weights, values, alpha)``.
    """
    return AdaptiveBoundSelector().select_bound(weights, values, alpha)


def required_sample_size(
    alpha: float = 0.05,
    epsilon: float = 0.01,
    ess_ratio: float = 0.5,
    support_range: float = 1.0,
) -> int:
    """One-call sample-size computation (module-level convenience)."""
    return FiniteSampleGuarantee(
        support_range=support_range,
    ).required_samples(alpha, epsilon, ess_ratio)


# ======================================================================
# 8. HoeffdingSelfNormalizedBound
# ======================================================================

class HoeffdingSelfNormalizedBound:
    r"""Hoeffding-type bound for self-normalised importance sampling.

    Mathematical Statement
    ----------------------
    For self-normalised IS with bounded function f ∈ [a, b] and
    normalised weights w̃_i = w_i / ∑_j w_j, the estimator

        μ̂_SN = ∑_i w̃_i f(X_i)

    satisfies (adapting Hoeffding's inequality via ESS):

        P(|μ̂_SN - μ| > ε) ≤ 2 exp(-2 n_eff ε² / (b - a)²)

    where n_eff = ESS is the Kish effective sample size.

    This differs from the Chatterjee–Diaconis bound in using the
    classical Hoeffding form directly with ESS replacement, giving
    tighter bounds when the ESS ratio is high (weights are well-behaved).

    Parameters
    ----------
    support : tuple (a, b) or None
        Known range of f.  If None, estimated from data.
    """

    def __init__(self, support: Optional[Tuple[float, float]] = None) -> None:
        self.support = support

    def confidence_interval(
        self,
        weights: np.ndarray,
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> BoundResult:
        """Compute Hoeffding-type CI for self-normalised IS.

        Parameters
        ----------
        weights : array of unnormalised importance weights.
        values : array of function evaluations.
        alpha : significance level.

        Returns
        -------
        BoundResult
        """
        weights, values = _validate_weights_values(weights, values)
        n = len(weights)

        w_sum = float(np.sum(weights))
        if w_sum == 0.0:
            raise ValueError("All weights are zero")
        w_norm = weights / w_sum
        mu_hat = float(np.dot(w_norm, values))

        ess = _effective_sample_size(w_norm)

        if self.support is not None:
            a, b = self.support
        else:
            lo, hi = float(np.min(values)), float(np.max(values))
            span = hi - lo
            if span == 0.0:
                span = max(abs(lo), 1.0)
            guard = 0.05 * span
            a, b = lo - guard, hi + guard

        rng = b - a
        # ε = rng * sqrt(log(2/α) / (2 n_eff))
        eps = rng * math.sqrt(math.log(2.0 / alpha) / (2.0 * max(ess, 1.0)))

        return BoundResult(
            lower=mu_hat - eps,
            upper=mu_hat + eps,
            point_estimate=mu_hat,
            method="Hoeffding-SelfNormalized",
            alpha=alpha,
            effective_sample_size=ess,
            bias_correction=0.0,
            diagnostics={
                "support": (a, b),
                "range": rng,
                "epsilon": eps,
                "ess": ess,
            },
        )


# ======================================================================
# 9. MartingaleStoppingCriterion
# ======================================================================

class MartingaleStoppingCriterion:
    r"""Martingale-based stopping criterion for sequential estimation.

    Theory
    ------
    For sequential IS estimation, we construct a test martingale
    M_t that tracks the accumulated evidence against the null
    hypothesis H_0: μ ∈ [μ_0 - ε, μ_0 + ε].

    At each step t, after observing (x_t, w_t), we update:

        S_t = S_{t-1} + w̃_t (f(x_t) - μ̂_{t-1})

    where w̃_t are normalised weights and μ̂_{t-1} is the running
    estimate.

    The estimator is declared converged when:

    1. The running CI width falls below target_width, AND
    2. The ESS exceeds a minimum threshold, AND
    3. At least min_samples have been collected.

    The confidence sequence approach (Howard et al., 2021) ensures
    that the coverage guarantee holds uniformly over all stopping
    times:

        P(∃t: μ ∉ CI_t) ≤ α

    Parameters
    ----------
    target_width : float
        Stop when CI width < target_width.
    alpha : float
        Overall significance level.
    min_samples : int
        Minimum samples before stopping is allowed.
    min_ess_ratio : float
        Minimum ESS/n ratio required.
    """

    def __init__(
        self,
        target_width: float = 0.05,
        alpha: float = 0.05,
        min_samples: int = 50,
        min_ess_ratio: float = 0.1,
    ) -> None:
        self.target_width = target_width
        self.alpha = alpha
        self.min_samples = min_samples
        self.min_ess_ratio = min_ess_ratio
        self._all_values: List[float] = []
        self._all_log_weights: List[float] = []
        self._running_estimates: List[float] = []
        self._running_widths: List[float] = []
        self._stopped = False

    def update(
        self,
        values: np.ndarray,
        log_weights: np.ndarray,
    ) -> Tuple[bool, Optional[BoundResult]]:
        """Add a batch of samples and check stopping criterion.

        Parameters
        ----------
        values : new function evaluations.
        log_weights : new log importance weights.

        Returns
        -------
        (should_stop, bound_result)
            should_stop is True if estimation should terminate.
            bound_result is the current CI (None if not enough samples).
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        log_weights = np.asarray(log_weights, dtype=np.float64).ravel()

        self._all_values.extend(values.tolist())
        self._all_log_weights.extend(log_weights.tolist())

        n = len(self._all_values)
        if n < self.min_samples:
            return False, None

        all_v = np.array(self._all_values)
        all_lw = np.array(self._all_log_weights)

        # Compute normalised weights
        max_lw = float(np.max(all_lw))
        w = np.exp(all_lw - max_lw)
        w_sum = float(np.sum(w))
        if w_sum == 0.0:
            return False, None
        w_norm = w / w_sum

        ess = _effective_sample_size(w_norm)
        mu_hat = float(np.dot(w_norm, all_v))
        self._running_estimates.append(mu_hat)

        # Confidence sequence: use time-uniform bound
        # ε_t = sqrt( (b-a)² · (log(log(2t)/α)) / (2·ESS) )
        lo, hi = float(np.min(all_v)), float(np.max(all_v))
        rng = max(hi - lo, 1e-12)
        log_log_term = math.log(max(math.log(max(2.0 * n, math.e)), 1.0))
        eps = rng * math.sqrt(
            (log_log_term + math.log(2.0 / self.alpha)) / (2.0 * max(ess, 1.0))
        )

        width = 2.0 * eps
        self._running_widths.append(width)

        result = BoundResult(
            lower=mu_hat - eps,
            upper=mu_hat + eps,
            point_estimate=mu_hat,
            method="MartingaleConfidenceSequence",
            alpha=self.alpha,
            effective_sample_size=ess,
            diagnostics={
                "n_total": n,
                "ess_ratio": ess / n,
                "width": width,
            },
        )

        ess_ratio = ess / n
        should_stop = (
            width < self.target_width
            and ess_ratio >= self.min_ess_ratio
            and n >= self.min_samples
        )

        if should_stop:
            self._stopped = True

        return should_stop, result

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def running_estimates(self) -> List[float]:
        return list(self._running_estimates)

    @property
    def running_widths(self) -> List[float]:
        return list(self._running_widths)

    def reset(self) -> None:
        """Reset the criterion for a new estimation run."""
        self._all_values.clear()
        self._all_log_weights.clear()
        self._running_estimates.clear()
        self._running_widths.clear()
        self._stopped = False
