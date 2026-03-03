"""Statistical distribution utilities for the CPA engine.

Implements Gaussian conditional distributions, divergence measures
(KL, JSD), partial-correlation and conditional-independence testing,
multiple-testing corrections, bootstrap utilities, effect-size
estimators, and permutation tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ===================================================================
# Constants
# ===================================================================

_EPS = 1e-300  # guard against log(0)
_SQRT2 = math.sqrt(2.0)


# ===================================================================
# GaussianConditional
# ===================================================================


@dataclass
class GaussianConditional:
    """Represent P(X | Pa(X)) as a linear-Gaussian conditional.

    .. math::

        X = \\sum_j \\beta_j \\cdot \\text{Pa}_j + \\varepsilon,
        \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)

    Parameters
    ----------
    variable : str
        Name of the child variable *X*.
    parents : list of str
        Names of the parent variables (ordered).
    coefficients : np.ndarray
        Regression coefficients :math:`\\beta`, shape ``(len(parents),)``.
    intercept : float
        Intercept (bias) term.
    residual_variance : float
        Residual variance :math:`\\sigma^2` (must be > 0).

    Examples
    --------
    >>> gc = GaussianConditional("Y", ["X1", "X2"],
    ...     np.array([0.5, -0.3]), intercept=1.0, residual_variance=0.25)
    >>> gc.mean(np.array([2.0, 1.0]))
    1.7
    """

    variable: str
    parents: List[str]
    coefficients: NDArray[np.floating]
    intercept: float = 0.0
    residual_variance: float = 1.0

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if self.coefficients.ndim != 1:
            raise ValueError(
                f"coefficients must be 1-D, got shape {self.coefficients.shape}"
            )
        if len(self.coefficients) != len(self.parents):
            raise ValueError(
                f"coefficients length {len(self.coefficients)} != "
                f"parents length {len(self.parents)}"
            )
        if self.residual_variance <= 0:
            raise ValueError(
                f"residual_variance must be > 0, got {self.residual_variance}"
            )

    @property
    def num_parents(self) -> int:
        """Number of parent variables."""
        return len(self.parents)

    @property
    def residual_std(self) -> float:
        """Residual standard deviation."""
        return math.sqrt(self.residual_variance)

    def mean(self, parent_values: NDArray[np.floating]) -> float:
        """Conditional mean E[X | Pa = parent_values].

        Parameters
        ----------
        parent_values : np.ndarray
            Values of parent variables, shape ``(num_parents,)``.

        Returns
        -------
        float
        """
        parent_values = np.asarray(parent_values, dtype=np.float64)
        if parent_values.shape != (self.num_parents,):
            raise ValueError(
                f"Expected parent_values shape ({self.num_parents},), "
                f"got {parent_values.shape}"
            )
        return float(self.intercept + np.dot(self.coefficients, parent_values))

    def log_prob(self, x: float, parent_values: NDArray[np.floating]) -> float:
        """Log-probability log P(X=x | Pa=parent_values).

        Parameters
        ----------
        x : float
            Observed value of *X*.
        parent_values : np.ndarray
            Parent values, shape ``(num_parents,)``.

        Returns
        -------
        float
        """
        mu = self.mean(parent_values)
        return float(
            sp_stats.norm.logpdf(x, loc=mu, scale=self.residual_std)
        )

    def sample(
        self,
        parent_values: NDArray[np.floating],
        *,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.floating]:
        """Sample from the conditional distribution.

        Parameters
        ----------
        parent_values : np.ndarray
            Parent values, shape ``(num_parents,)`` or ``(n, num_parents)``.
        n : int
            Number of samples (ignored if parent_values is 2-D).
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Samples, shape ``(n,)``.
        """
        rng = rng or np.random.default_rng()
        pv = np.asarray(parent_values, dtype=np.float64)
        if pv.ndim == 1:
            mu = self.mean(pv)
            return rng.normal(mu, self.residual_std, size=n)
        elif pv.ndim == 2:
            means = self.intercept + pv @ self.coefficients
            return rng.normal(means, self.residual_std)
        raise ValueError(f"parent_values must be 1-D or 2-D, got {pv.ndim}-D")

    def kl_divergence_to(self, other: "GaussianConditional") -> float:
        """KL(self || other) assuming same parent values = 0.

        Computes KL divergence between two univariate Gaussians with
        the same mean but potentially different variances and
        coefficients.  This is a *mechanism-level* divergence that
        measures how different the two conditional distributions are
        for the *same* parent configuration (zero vector).

        Parameters
        ----------
        other : GaussianConditional
            The other conditional distribution.

        Returns
        -------
        float
            KL divergence (>= 0).
        """
        mu_self = self.intercept
        mu_other = other.intercept
        s2_self = self.residual_variance
        s2_other = other.residual_variance
        return kl_gaussian(mu_self, s2_self, mu_other, s2_other)

    def jsd_to(self, other: "GaussianConditional") -> float:
        """Jensen-Shannon divergence to *other* at parent_values=0.

        Parameters
        ----------
        other : GaussianConditional
            The other conditional distribution.

        Returns
        -------
        float
            JSD in nats (>= 0).
        """
        return jsd_gaussian(
            self.intercept, self.residual_variance,
            other.intercept, other.residual_variance,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "variable": self.variable,
            "parents": list(self.parents),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "residual_variance": self.residual_variance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianConditional":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        GaussianConditional
        """
        return cls(
            variable=d["variable"],
            parents=list(d["parents"]),
            coefficients=np.array(d["coefficients"], dtype=np.float64),
            intercept=float(d.get("intercept", 0.0)),
            residual_variance=float(d.get("residual_variance", 1.0)),
        )

    @classmethod
    def fit(
        cls,
        variable: str,
        parents: List[str],
        X_child: NDArray[np.floating],
        X_parents: NDArray[np.floating],
    ) -> "GaussianConditional":
        """Fit a linear-Gaussian conditional from data via OLS.

        Parameters
        ----------
        variable : str
            Child variable name.
        parents : list of str
            Parent variable names.
        X_child : np.ndarray
            Observations of the child, shape ``(n,)``.
        X_parents : np.ndarray
            Observations of parents, shape ``(n, len(parents))``.

        Returns
        -------
        GaussianConditional
            Fitted conditional distribution.
        """
        X_child = np.asarray(X_child, dtype=np.float64).ravel()
        X_parents = np.asarray(X_parents, dtype=np.float64)
        n = X_child.shape[0]
        if X_parents.ndim == 1:
            X_parents = X_parents.reshape(n, 1)
        if X_parents.shape[0] != n:
            raise ValueError("X_child and X_parents must have same n")
        if len(parents) == 0:
            intercept = float(np.mean(X_child))
            resid_var = float(np.var(X_child, ddof=1)) if n > 1 else 1.0
            return cls(variable, [], np.array([]), intercept, max(resid_var, 1e-10))

        # OLS: X_child = X_parents @ beta + intercept + eps
        ones = np.ones((n, 1), dtype=np.float64)
        design = np.hstack([X_parents, ones])
        result, residuals, _, _ = np.linalg.lstsq(design, X_child, rcond=None)
        coefficients = result[:-1]
        intercept = float(result[-1])
        predicted = design @ result
        resid = X_child - predicted
        resid_var = float(np.var(resid, ddof=max(1, len(parents) + 1)))
        return cls(
            variable, list(parents), coefficients, intercept, max(resid_var, 1e-10)
        )

    def __repr__(self) -> str:
        coeff_str = ", ".join(
            f"{p}={c:.4f}" for p, c in zip(self.parents, self.coefficients)
        )
        return (
            f"GaussianConditional({self.variable} | {coeff_str}, "
            f"intercept={self.intercept:.4f}, σ²={self.residual_variance:.4f})"
        )


# ===================================================================
# KL divergence
# ===================================================================


def kl_discrete(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """Kullback-Leibler divergence KL(P || Q) for discrete distributions.

    Parameters
    ----------
    p : np.ndarray
        Probability mass function P, shape ``(k,)``.
    q : np.ndarray
        Probability mass function Q, shape ``(k,)``.

    Returns
    -------
    float
        KL divergence in nats (>= 0).

    Raises
    ------
    ValueError
        If shapes mismatch or inputs are not valid PMFs.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p={p.shape}, q={q.shape}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("PMFs must be non-negative")
    p_sum, q_sum = p.sum(), q.sum()
    if abs(p_sum - 1.0) > 1e-6 or abs(q_sum - 1.0) > 1e-6:
        warnings.warn("PMFs do not sum to 1; normalizing", stacklevel=2)
        p = p / p_sum
        q = q / q_sum

    mask = p > 0
    if np.any(mask & (q <= 0)):
        return float("inf")
    result = np.sum(p[mask] * np.log(p[mask] / (q[mask] + _EPS)))
    return max(0.0, float(result))


def kl_gaussian(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
) -> float:
    """KL(N(mu1,var1) || N(mu2,var2)) for univariate Gaussians.

    Parameters
    ----------
    mu1 : float
        Mean of distribution P.
    var1 : float
        Variance of P (must be > 0).
    mu2 : float
        Mean of distribution Q.
    var2 : float
        Variance of Q (must be > 0).

    Returns
    -------
    float
        KL divergence in nats (>= 0).
    """
    if var1 <= 0 or var2 <= 0:
        raise ValueError(f"Variances must be > 0, got var1={var1}, var2={var2}")
    return 0.5 * (
        math.log(var2 / var1)
        + var1 / var2
        + (mu1 - mu2) ** 2 / var2
        - 1.0
    )


def kl_gaussian_mv(
    mu1: NDArray[np.floating],
    cov1: NDArray[np.floating],
    mu2: NDArray[np.floating],
    cov2: NDArray[np.floating],
) -> float:
    """KL(N(mu1,cov1) || N(mu2,cov2)) for multivariate Gaussians.

    Parameters
    ----------
    mu1, mu2 : np.ndarray
        Mean vectors, shape ``(d,)``.
    cov1, cov2 : np.ndarray
        Covariance matrices, shape ``(d, d)``.

    Returns
    -------
    float
        KL divergence in nats.
    """
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    cov1, cov2 = np.asarray(cov1), np.asarray(cov2)
    d = mu1.shape[0]
    cov2_inv = np.linalg.inv(cov2)
    diff = mu2 - mu1
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    return 0.5 * float(
        np.trace(cov2_inv @ cov1)
        + diff @ cov2_inv @ diff
        - d
        + logdet2 - logdet1
    )


# ===================================================================
# Jensen-Shannon divergence
# ===================================================================


def jsd_discrete(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """Jensen-Shannon divergence for discrete distributions.

    Parameters
    ----------
    p : np.ndarray
        PMF P.
    q : np.ndarray
        PMF Q.

    Returns
    -------
    float
        JSD in nats, in [0, ln(2)].
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p={p.shape}, q={q.shape}")
    p = p / (p.sum() + _EPS)
    q = q / (q.sum() + _EPS)
    m = 0.5 * (p + q)
    return max(0.0, 0.5 * kl_discrete(p, m) + 0.5 * kl_discrete(q, m))


def jsd_gaussian(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
) -> float:
    """Jensen-Shannon divergence for two univariate Gaussians.

    Uses the analytic formula via mixture entropy bounds.

    Parameters
    ----------
    mu1 : float
        Mean of P.
    var1 : float
        Variance of P.
    mu2 : float
        Mean of Q.
    var2 : float
        Variance of Q.

    Returns
    -------
    float
        JSD in nats (>= 0).
    """
    if var1 <= 0 or var2 <= 0:
        raise ValueError(f"Variances must be > 0, got var1={var1}, var2={var2}")
    # Monte-Carlo free: use 0.5*KL(P||M)+0.5*KL(Q||M) where M is the
    # mixture.  For Gaussians the mixture is NOT Gaussian, so we use
    # numerical integration via scipy.
    from scipy.integrate import quad

    std1, std2 = math.sqrt(var1), math.sqrt(var2)

    def p_pdf(x: float) -> float:
        return float(sp_stats.norm.pdf(x, mu1, std1))

    def q_pdf(x: float) -> float:
        return float(sp_stats.norm.pdf(x, mu2, std2))

    def integrand(x: float) -> float:
        px = p_pdf(x)
        qx = q_pdf(x)
        mx = 0.5 * (px + qx)
        val = 0.0
        if px > _EPS:
            val += 0.5 * px * math.log(px / (mx + _EPS))
        if qx > _EPS:
            val += 0.5 * qx * math.log(qx / (mx + _EPS))
        return val

    center = 0.5 * (mu1 + mu2)
    spread = 5.0 * max(std1, std2) + abs(mu1 - mu2)
    result, _ = quad(integrand, center - spread, center + spread, limit=200)
    return max(0.0, result)


def sqrt_jsd_discrete(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """Square root of JSD (a proper metric) for discrete distributions.

    Parameters
    ----------
    p, q : np.ndarray
        PMFs.

    Returns
    -------
    float
        sqrt(JSD) in [0, sqrt(ln(2))].
    """
    return math.sqrt(max(0.0, jsd_discrete(p, q)))


def sqrt_jsd_gaussian(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
) -> float:
    """Square root of JSD for univariate Gaussians.

    Parameters
    ----------
    mu1, var1 : float
        Parameters of P.
    mu2, var2 : float
        Parameters of Q.

    Returns
    -------
    float
    """
    return math.sqrt(max(0.0, jsd_gaussian(mu1, var1, mu2, var2)))


# ===================================================================
# Partial correlation & CI testing
# ===================================================================


def partial_correlation(
    X: NDArray[np.floating],
    i: int,
    j: int,
    S: Sequence[int] = (),
) -> float:
    """Compute the partial correlation between variables i and j given S.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    i : int
        First variable index.
    j : int
        Second variable index.
    S : sequence of int
        Conditioning set variable indices.

    Returns
    -------
    float
        Partial correlation in [-1, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if n < 3:
        raise ValueError(f"Need at least 3 samples, got {n}")
    indices = [i, j] + list(S)
    for idx in indices:
        if idx < 0 or idx >= p:
            raise ValueError(f"Variable index {idx} out of range [0, {p})")
    if len(S) == 0:
        return float(np.corrcoef(X[:, i], X[:, j])[0, 1])

    # Residualise i and j on S using OLS
    S_data = X[:, list(S)]
    ones = np.ones((n, 1))
    design = np.hstack([S_data, ones])

    def _residualise(y: NDArray) -> NDArray:
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ beta

    resid_i = _residualise(X[:, i])
    resid_j = _residualise(X[:, j])

    denom = np.sqrt(np.sum(resid_i**2) * np.sum(resid_j**2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(resid_i * resid_j) / denom)


def partial_correlation_matrix(
    X: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the partial correlation matrix from the precision matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.

    Returns
    -------
    np.ndarray
        Partial correlation matrix, shape ``(p, p)``.
    """
    X = np.asarray(X, dtype=np.float64)
    cov = np.cov(X, rowvar=False)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec))
    d[d < 1e-15] = 1.0
    pcor = -prec / np.outer(d, d)
    np.fill_diagonal(pcor, 1.0)
    return pcor


def fisher_z_test(
    r: float,
    n: int,
    k: int = 0,
) -> Tuple[float, float]:
    """Fisher's z-test for partial correlation.

    Tests H₀: ρ = 0 against H₁: ρ ≠ 0.

    Parameters
    ----------
    r : float
        Sample partial correlation.
    n : int
        Sample size.
    k : int
        Size of conditioning set.

    Returns
    -------
    (z_statistic, p_value) : tuple of float
    """
    if n - k - 3 <= 0:
        return 0.0, 1.0
    r = np.clip(r, -1 + 1e-10, 1 - 1e-10)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - k - 3)
    z_stat = abs(z) / se
    p_val = 2.0 * (1.0 - sp_stats.norm.cdf(z_stat))
    return float(z_stat), float(p_val)


def partial_correlation_test(
    X: NDArray[np.floating],
    i: int,
    j: int,
    S: Sequence[int] = (),
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """Test conditional independence using partial correlation + Fisher's z.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    i, j : int
        Variable indices.
    S : sequence of int
        Conditioning set.
    alpha : float
        Significance level.

    Returns
    -------
    (partial_corr, p_value, is_independent) : tuple
    """
    r = partial_correlation(X, i, j, S)
    n = X.shape[0]
    _, p_val = fisher_z_test(r, n, len(S))
    return r, p_val, p_val > alpha


# ===================================================================
# Multiple testing corrections
# ===================================================================


def bonferroni_correction(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    (adjusted_p, rejected) : tuple
        Adjusted p-values and boolean rejection mask.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted <= alpha
    return adjusted, rejected


def bh_fdr_correction(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    alpha : float
        Target false discovery rate.

    Returns
    -------
    (adjusted_p, rejected) : tuple
        Adjusted p-values and boolean rejection mask.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    adjusted = np.empty(m, dtype=np.float64)

    # Step-up procedure
    cum_min = 1.0
    for rank in range(m, 0, -1):
        idx = rank - 1
        adj = sorted_p[idx] * m / rank
        cum_min = min(cum_min, adj)
        adjusted[sorted_idx[idx]] = min(cum_min, 1.0)

    rejected = adjusted <= alpha
    return adjusted, rejected


# ===================================================================
# Bootstrap utilities
# ===================================================================


def bootstrap_ci(
    data: NDArray[np.floating],
    statistic: Callable[[NDArray[np.floating]], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    method: str = "percentile",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Input data (any shape; resampled along axis 0).
    statistic : callable
        Function mapping data → scalar.
    n_bootstrap : int
        Number of bootstrap replicates.
    confidence : float
        Confidence level (e.g. 0.95).
    method : ``"percentile"`` or ``"bca"``
        CI method.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    (point_estimate, lower, upper) : tuple of float
    """
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    if n == 0:
        raise ValueError("data is empty")

    point = statistic(data)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic(data[idx])

    alpha = 1.0 - confidence

    if method == "percentile":
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    elif method == "bca":
        # Bias-corrected and accelerated
        z0 = sp_stats.norm.ppf(np.mean(boot_stats < point))
        # Jackknife for acceleration
        jack = np.empty(n, dtype=np.float64)
        for i in range(n):
            jack[i] = statistic(np.delete(data, i, axis=0))
        jack_mean = jack.mean()
        num = np.sum((jack_mean - jack) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5
        a = num / den if abs(den) > 1e-15 else 0.0
        z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
        z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

        def _bca_percentile(z_alpha: float) -> float:
            num = z0 + z_alpha
            p = sp_stats.norm.cdf(z0 + num / (1 - a * num))
            return float(np.percentile(boot_stats, 100 * p))

        lower = _bca_percentile(z_alpha_lo)
        upper = _bca_percentile(z_alpha_hi)
    else:
        raise ValueError(f"Unknown method {method!r}")

    return point, lower, upper


def stability_selection(
    X: NDArray[np.floating],
    selection_fn: Callable[[NDArray[np.floating]], NDArray[np.bool_]],
    *,
    n_subsamples: int = 100,
    subsample_fraction: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Stability selection: compute selection probabilities.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    selection_fn : callable
        Function that takes a data matrix and returns a boolean mask
        of selected features, shape ``(p,)``.
    n_subsamples : int
        Number of sub-sampling iterations.
    subsample_fraction : float
        Fraction of samples to use per iteration.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Selection probabilities, shape ``(p,)``, in [0, 1].
    """
    rng = rng or np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    sub_n = max(1, int(n * subsample_fraction))
    counts = np.zeros(p, dtype=np.float64)
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=sub_n, replace=False)
        selected = selection_fn(X[idx])
        counts += selected.astype(np.float64)
    return counts / n_subsamples


# ===================================================================
# Effect-size estimators
# ===================================================================


def cohens_d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    """Cohen's d effect size for two independent samples.

    Parameters
    ----------
    x : np.ndarray
        Sample 1.
    y : np.ndarray
        Sample 2.

    Returns
    -------
    float
        Cohen's d.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 samples in each group")
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def hedges_g(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    """Hedges' g effect size (bias-corrected Cohen's d).

    Parameters
    ----------
    x : np.ndarray
        Sample 1.
    y : np.ndarray
        Sample 2.

    Returns
    -------
    float
        Hedges' g.
    """
    d = cohens_d(x, y)
    n = len(x) + len(y)
    # Correction factor for small samples
    correction = 1.0 - 3.0 / (4.0 * (n - 2) - 1.0)
    return d * correction


# ===================================================================
# Permutation tests
# ===================================================================


def permutation_test(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    statistic: Callable[
        [NDArray[np.floating], NDArray[np.floating]], float
    ],
    *,
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Permutation test for two independent samples.

    Parameters
    ----------
    x : np.ndarray
        Sample 1.
    y : np.ndarray
        Sample 2.
    statistic : callable
        Test statistic function ``(x, y) → float``.
    n_permutations : int
        Number of permutations.
    alternative : ``"two-sided"``, ``"greater"``, or ``"less"``
        Alternative hypothesis.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    (observed_stat, p_value) : tuple of float
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    observed = statistic(x, y)
    combined = np.concatenate([x, y])
    n1 = len(x)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic(combined[:n1], combined[n1:])
        if alternative == "two-sided":
            if abs(perm_stat) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if perm_stat >= observed:
                count += 1
        elif alternative == "less":
            if perm_stat <= observed:
                count += 1
        else:
            raise ValueError(f"Unknown alternative {alternative!r}")
    p_value = (count + 1) / (n_permutations + 1)
    return float(observed), float(p_value)


# ===================================================================
# Utility: empirical CDF & histogram matching
# ===================================================================


def empirical_cdf(
    data: NDArray[np.floating],
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Evaluate the empirical CDF of *data* at points *x*.

    Parameters
    ----------
    data : np.ndarray
        1-D data sample.
    x : np.ndarray
        Evaluation points.

    Returns
    -------
    np.ndarray
        CDF values.
    """
    data = np.sort(np.asarray(data, dtype=np.float64).ravel())
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.searchsorted(data, x, side="right").astype(np.float64) / len(data)


def two_sample_ks_statistic(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test statistic and p-value.

    Parameters
    ----------
    x, y : np.ndarray
        1-D samples.

    Returns
    -------
    (ks_stat, p_value) : tuple of float
    """
    result = sp_stats.ks_2samp(
        np.asarray(x, dtype=np.float64).ravel(),
        np.asarray(y, dtype=np.float64).ravel(),
    )
    return float(result.statistic), float(result.pvalue)
