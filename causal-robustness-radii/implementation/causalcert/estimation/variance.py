"""
Advanced variance estimation methods for causal inference.

Implements influence-function-based sandwich variance, wild bootstrap,
pairs bootstrap with BCa intervals, HAC (Newey-West) estimation,
cluster-robust variance, and the delta method for transformed parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class VarianceResult:
    """Result of a variance estimation procedure.

    Attributes
    ----------
    variance : float
        Estimated variance of the parameter.
    se : float
        Standard error.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    method : str
        Variance estimation method.
    """

    variance: float
    se: float
    ci_lower: float
    ci_upper: float
    method: str = "sandwich"


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bootstrap inference result.

    Attributes
    ----------
    se : float
        Bootstrap standard error.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    ci_method : str
        CI construction method.
    bias : float
        Bootstrap bias estimate.
    n_bootstrap : int
        Number of replicates used.
    boot_distribution : np.ndarray
        Bootstrap distribution of estimates.
    """

    se: float
    ci_lower: float
    ci_upper: float
    ci_method: str = "percentile"
    bias: float = 0.0
    n_bootstrap: int = 0
    boot_distribution: np.ndarray = field(default_factory=lambda: np.array([]))


# ===================================================================
# 1. Influence function sandwich variance
# ===================================================================


def sandwich_variance(
    psi: np.ndarray,
    estimate: float,
    alpha: float = 0.05,
) -> VarianceResult:
    """Influence-function-based sandwich variance estimator.

    Var(θ̂) = (1/n²) Σ ψ²_i, where ψ is the influence function.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values (centred), shape ``(n,)``.
    estimate : float
        Point estimate.
    alpha : float
        Significance level.

    Returns
    -------
    VarianceResult
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)
    var = float(np.mean(psi ** 2) / n)
    se = np.sqrt(var)
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        method="sandwich",
    )


def robust_sandwich(
    psi: np.ndarray,
    estimate: float,
    *,
    alpha: float = 0.05,
    dof_correction: bool = True,
) -> VarianceResult:
    """HC1 (heteroskedasticity-consistent) sandwich variance.

    Uses Σ = (n/(n−1)) * mean(ψ²) for finite-sample correction.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values.
    estimate : float
        Point estimate.
    alpha : float
        Significance level.
    dof_correction : bool
        Apply degrees-of-freedom correction.

    Returns
    -------
    VarianceResult
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)
    correction = n / (n - 1) if dof_correction and n > 1 else 1.0
    var = correction * float(np.mean(psi ** 2) / n)
    se = np.sqrt(var)
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        method="hc1_sandwich",
    )


# ===================================================================
# 2. Wild bootstrap
# ===================================================================


def wild_bootstrap(
    Y: np.ndarray,
    residuals: np.ndarray,
    estimator_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
    distribution: str = "rademacher",
) -> BootstrapResult:
    """Wild bootstrap for heteroskedasticity-robust inference.

    Generates bootstrap samples by perturbing residuals with random
    weights, preserving the heteroskedastic structure.

    Parameters
    ----------
    Y : np.ndarray
        Outcome values, shape ``(n,)``.
    residuals : np.ndarray
        Residuals from the model, shape ``(n,)``.
    estimator_fn : Callable
        Function mapping bootstrapped Y to a scalar estimate.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    distribution : str
        Weight distribution: ``"rademacher"`` (±1) or ``"mammen"``
        (two-point Mammen distribution).

    Returns
    -------
    BootstrapResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(Y)
    rng = np.random.default_rng(seed)

    fitted = Y - residuals
    theta_obs = estimator_fn(Y)
    boot_thetas = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        if distribution == "rademacher":
            weights = rng.choice([-1.0, 1.0], size=n)
        elif distribution == "mammen":
            # Two-point Mammen distribution
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            u = rng.random(n)
            weights = np.where(
                u < p,
                -(np.sqrt(5) - 1) / 2,
                (np.sqrt(5) + 1) / 2,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

        Y_star = fitted + weights * residuals
        boot_thetas[b] = estimator_fn(Y_star)

    se = float(np.std(boot_thetas, ddof=1))
    bias = float(np.mean(boot_thetas) - theta_obs)
    ci_lo = float(np.percentile(boot_thetas, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_thetas, 100 * (1.0 - alpha / 2)))

    return BootstrapResult(
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_method="wild_percentile",
        bias=bias,
        n_bootstrap=n_bootstrap,
        boot_distribution=boot_thetas,
    )


# ===================================================================
# 3. Pairs bootstrap with BCa intervals
# ===================================================================


def pairs_bootstrap(
    data: np.ndarray,
    estimator_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
    bca: bool = True,
) -> BootstrapResult:
    """Pairs bootstrap with optional BCa (bias-corrected accelerated) CIs.

    Parameters
    ----------
    data : np.ndarray
        Data matrix, shape ``(n, p)``. Each row is an observation.
    estimator_fn : Callable
        Function mapping a data matrix to a scalar estimate.
    n_bootstrap : int
        Number of replicates.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    bca : bool
        If ``True``, compute BCa intervals.  Otherwise, percentile.

    Returns
    -------
    BootstrapResult
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    theta_obs = estimator_fn(data)
    boot_thetas = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_thetas[b] = estimator_fn(data[idx])

    se = float(np.std(boot_thetas, ddof=1))
    bias = float(np.mean(boot_thetas) - theta_obs)

    if bca:
        ci_lo, ci_hi = _bca_intervals(
            data, estimator_fn, boot_thetas, theta_obs, alpha,
        )
    else:
        ci_lo = float(np.percentile(boot_thetas, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_thetas, 100 * (1.0 - alpha / 2)))

    return BootstrapResult(
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_method="bca" if bca else "percentile",
        bias=bias,
        n_bootstrap=n_bootstrap,
        boot_distribution=boot_thetas,
    )


def _bca_intervals(
    data: np.ndarray,
    estimator_fn: Callable[[np.ndarray], float],
    boot_thetas: np.ndarray,
    theta_obs: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute BCa confidence interval endpoints.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix.
    estimator_fn : Callable
        Estimator function.
    boot_thetas : np.ndarray
        Bootstrap distribution.
    theta_obs : float
        Observed estimate.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` BCa confidence bounds.
    """
    n = data.shape[0]
    B = len(boot_thetas)

    # Bias correction factor z0
    prop_below = float(np.mean(boot_thetas < theta_obs))
    prop_below = np.clip(prop_below, 1e-6, 1.0 - 1e-6)
    z0 = sp_stats.norm.ppf(prop_below)

    # Acceleration factor a (jackknife)
    jack_thetas = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_data = np.delete(data, i, axis=0)
        jack_thetas[i] = estimator_fn(jack_data)
    jack_mean = float(np.mean(jack_thetas))
    diff = jack_mean - jack_thetas
    a_num = float(np.sum(diff ** 3))
    a_den = float(np.sum(diff ** 2)) ** 1.5
    a = a_num / (6.0 * max(a_den, 1e-12))

    # Adjusted quantiles
    z_alpha_lo = sp_stats.norm.ppf(alpha / 2.0)
    z_alpha_hi = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    denom_lo = 1.0 - a * (z0 + z_alpha_lo)
    denom_hi = 1.0 - a * (z0 + z_alpha_hi)
    if abs(denom_lo) < 1e-12:
        denom_lo = 1e-12
    if abs(denom_hi) < 1e-12:
        denom_hi = 1e-12

    alpha_lo = sp_stats.norm.cdf(z0 + (z0 + z_alpha_lo) / denom_lo)
    alpha_hi = sp_stats.norm.cdf(z0 + (z0 + z_alpha_hi) / denom_hi)

    alpha_lo = np.clip(alpha_lo, 1.0 / B, 1.0 - 1.0 / B)
    alpha_hi = np.clip(alpha_hi, 1.0 / B, 1.0 - 1.0 / B)

    ci_lo = float(np.percentile(boot_thetas, 100 * alpha_lo))
    ci_hi = float(np.percentile(boot_thetas, 100 * alpha_hi))

    return ci_lo, ci_hi


# ===================================================================
# 4. HAC (Newey-West) estimation
# ===================================================================


def hac_variance(
    psi: np.ndarray,
    estimate: float,
    *,
    n_lags: int | None = None,
    kernel: str = "bartlett",
    alpha: float = 0.05,
) -> VarianceResult:
    """Heteroskedasticity and Autocorrelation Consistent (HAC) variance.

    Implements the Newey-West estimator with Bartlett or Parzen kernel.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values (or moment conditions), shape ``(n,)``.
    estimate : float
        Point estimate.
    n_lags : int or None
        Number of lags.  If ``None``, uses floor(n^(1/3)).
    kernel : str
        Kernel type: ``"bartlett"`` or ``"parzen"``.
    alpha : float
        Significance level.

    Returns
    -------
    VarianceResult
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)

    if n_lags is None:
        n_lags = int(np.floor(n ** (1.0 / 3.0)))

    # Autocovariance at lag 0
    gamma_0 = float(np.mean(psi ** 2))

    # Sum weighted autocovariances
    omega = gamma_0
    for j in range(1, n_lags + 1):
        if kernel == "bartlett":
            w = 1.0 - j / (n_lags + 1.0)
        elif kernel == "parzen":
            ratio = j / (n_lags + 1.0)
            if ratio <= 0.5:
                w = 1.0 - 6.0 * ratio ** 2 + 6.0 * ratio ** 3
            else:
                w = 2.0 * (1.0 - ratio) ** 3
        else:
            raise ValueError(f"Unknown kernel: {kernel!r}")

        gamma_j = float(np.mean(psi[j:] * psi[:-j]))
        omega += 2.0 * w * gamma_j

    var = omega / n
    se = np.sqrt(max(var, 0.0))
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        method=f"hac_{kernel}",
    )


# ===================================================================
# 5. Cluster-robust variance
# ===================================================================


def cluster_robust_variance(
    psi: np.ndarray,
    cluster_ids: np.ndarray,
    estimate: float,
    *,
    alpha: float = 0.05,
) -> VarianceResult:
    """Cluster-robust (CR0) variance estimator.

    Computes variance allowing for arbitrary within-cluster correlation.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values, shape ``(n,)``.
    cluster_ids : np.ndarray
        Cluster membership for each observation, shape ``(n,)``.
    estimate : float
        Point estimate.
    alpha : float
        Significance level.

    Returns
    -------
    VarianceResult
    """
    psi = np.asarray(psi, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    n = len(psi)

    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    # Sum ψ within each cluster
    cluster_sums = np.array([
        float(np.sum(psi[cluster_ids == c])) for c in unique_clusters
    ])

    # CR0 variance: (1/n²) * Σ_g S_g²
    var = float(np.sum(cluster_sums ** 2)) / (n ** 2)

    # Small-sample correction: G/(G-1) * (n-1)/n
    if G > 1:
        correction = (G / (G - 1.0)) * ((n - 1.0) / n)
        var *= correction

    se = np.sqrt(max(var, 0.0))
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        method="cluster_robust",
    )


# ===================================================================
# 6. Delta method
# ===================================================================


def delta_method(
    theta: np.ndarray,
    cov: np.ndarray,
    g: Callable[[np.ndarray], float],
    *,
    h: float = 1e-5,
    alpha: float = 0.05,
) -> VarianceResult:
    """Delta method for transformed parameters.

    For θ̂ ~ N(θ, Σ), the transformation g(θ̂) has approximate
    variance g'(θ)ᵀ Σ g'(θ).

    The gradient is estimated numerically via central differences.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector, shape ``(k,)``.
    cov : np.ndarray
        Covariance matrix of θ̂, shape ``(k, k)``.
    g : Callable
        Scalar transformation function.
    h : float
        Step size for numerical gradient.
    alpha : float
        Significance level.

    Returns
    -------
    VarianceResult
    """
    theta = np.asarray(theta, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    k = len(theta)

    g_theta = g(theta)

    # Numerical gradient
    grad = np.empty(k, dtype=np.float64)
    for j in range(k):
        e_j = np.zeros(k)
        e_j[j] = h
        grad[j] = (g(theta + e_j) - g(theta - e_j)) / (2.0 * h)

    var = float(grad @ cov @ grad)
    se = np.sqrt(max(var, 0.0))
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=g_theta - z * se,
        ci_upper=g_theta + z * se,
        method="delta_method",
    )


def delta_method_ratio(
    theta1: float,
    theta2: float,
    se1: float,
    se2: float,
    cov12: float = 0.0,
    *,
    alpha: float = 0.05,
) -> VarianceResult:
    """Delta method for the ratio θ₁/θ₂.

    Parameters
    ----------
    theta1, theta2 : float
        Numerator and denominator estimates.
    se1, se2 : float
        Standard errors.
    cov12 : float
        Covariance between θ₁ and θ₂.
    alpha : float
        Significance level.

    Returns
    -------
    VarianceResult
    """
    if abs(theta2) < 1e-12:
        return VarianceResult(
            variance=float("inf"), se=float("inf"),
            ci_lower=float("-inf"), ci_upper=float("inf"),
            method="delta_ratio",
        )
    ratio = theta1 / theta2
    # Var(θ₁/θ₂) ≈ (1/θ₂²)[Var(θ₁) - 2(θ₁/θ₂)Cov(θ₁,θ₂) + (θ₁/θ₂)²Var(θ₂)]
    var = (1.0 / theta2 ** 2) * (
        se1 ** 2 - 2.0 * ratio * cov12 + ratio ** 2 * se2 ** 2
    )
    se = np.sqrt(max(var, 0.0))
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    return VarianceResult(
        variance=var,
        se=se,
        ci_lower=ratio - z * se,
        ci_upper=ratio + z * se,
        method="delta_ratio",
    )
