"""Statistical testing utilities, bootstrap helpers, and kernel functions.

Provides building blocks used by :mod:`causalcert.ci_testing` and
:mod:`causalcert.estimation`, but also useful standalone.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

FloatArray = NDArray[np.floating]


# ====================================================================
# 1. Hypothesis Testing Utilities
# ====================================================================

def z_test_two_sided(z: float) -> float:
    """Two-sided p-value from a z-statistic."""
    return 2.0 * sp_stats.norm.sf(abs(z))


def t_test_two_sided(t_stat: float, df: int) -> float:
    """Two-sided p-value from a t-statistic."""
    return 2.0 * sp_stats.t.sf(abs(t_stat), df)


def chi2_test(stat: float, df: int) -> float:
    """Right-tail p-value for a chi-squared statistic."""
    return float(sp_stats.chi2.sf(stat, df))


def fisher_exact_combine(pvalues: Sequence[float]) -> float:
    """Fisher's method for combining independent p-values.

    Test statistic: -2 Σ log(p_i) ~ χ²(2k).
    """
    pvals = np.asarray(pvalues, dtype=np.float64)
    pvals = np.clip(pvals, 1e-300, 1.0)
    stat = -2.0 * np.sum(np.log(pvals))
    return float(sp_stats.chi2.sf(stat, 2 * len(pvals)))


def cauchy_combine(pvalues: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Cauchy combination test (Liu & Xie, 2020).

    Transforms p-values via tan(π(0.5 − p)) and takes a weighted average.
    """
    pvals = np.asarray(pvalues, dtype=np.float64)
    pvals = np.clip(pvals, 1e-15, 1 - 1e-15)

    if weights is None:
        w = np.ones(len(pvals)) / len(pvals)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

    T = np.sum(w * np.tan(np.pi * (0.5 - pvals)))
    p_combined = 0.5 - np.arctan(T) / np.pi
    return float(np.clip(p_combined, 0.0, 1.0))


def stouffer_combine(
    pvalues: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Stouffer's weighted Z-method for combining p-values."""
    pvals = np.asarray(pvalues, dtype=np.float64)
    pvals = np.clip(pvals, 1e-300, 1 - 1e-15)
    zscores = sp_stats.norm.ppf(1 - pvals)

    if weights is None:
        w = np.ones(len(pvals))
    else:
        w = np.asarray(weights, dtype=np.float64)

    z_combined = np.sum(w * zscores) / np.sqrt(np.sum(w ** 2))
    return float(sp_stats.norm.sf(z_combined))


# ====================================================================
# 2. Multiple-Comparison Correction
# ====================================================================

def bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni correction: reject if p ≤ α/m."""
    m = len(pvalues)
    threshold = alpha / m
    return [p <= threshold for p in pvalues]


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down procedure."""
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    reject = [False] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini–Hochberg FDR control."""
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    reject = [False] * m
    max_k = 0

    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= rank / m * alpha:
            max_k = rank

    for rank, (orig_idx, _) in enumerate(indexed, 1):
        if rank <= max_k:
            reject[orig_idx] = True

    return reject


def adjusted_pvalues_bh(pvalues: Sequence[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values."""
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * m

    prev = 1.0
    for rank in range(m, 0, -1):
        orig_idx, p = indexed[rank - 1]
        adj = min(prev, p * m / rank)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev = adj

    return adjusted


# ====================================================================
# 3. Bootstrap Utilities
# ====================================================================

def bootstrap_statistic(
    data: FloatArray,
    statistic_fn: Callable[[FloatArray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, FloatArray]:
    """Non-parametric bootstrap for a scalar statistic.

    Returns (point_estimate, std_error, bootstrap_distribution).
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    boot_stats = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic_fn(data[idx])

    point = statistic_fn(data)
    se = float(np.std(boot_stats, ddof=1))
    return point, se, boot_stats


def bootstrap_ci(
    boot_dist: FloatArray,
    alpha: float = 0.05,
    method: str = "percentile",
) -> tuple[float, float]:
    """Confidence interval from a bootstrap distribution.

    Methods: "percentile", "basic", "bca" (bias-corrected accelerated, stub).
    """
    if method == "percentile":
        lo = float(np.percentile(boot_dist, 100 * alpha / 2))
        hi = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))
        return lo, hi
    elif method == "basic":
        theta = float(np.mean(boot_dist))
        lo = 2 * theta - float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))
        hi = 2 * theta - float(np.percentile(boot_dist, 100 * alpha / 2))
        return lo, hi
    else:
        # Fallback to percentile
        return bootstrap_ci(boot_dist, alpha, "percentile")


def paired_bootstrap_test(
    data: FloatArray,
    stat_a: Callable[[FloatArray], float],
    stat_b: Callable[[FloatArray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test for H₀: stat_a(data) = stat_b(data).

    Returns p-value for the two-sided test.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    observed_diff = stat_a(data) - stat_b(data)
    count = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = stat_a(data[idx]) - stat_b(data[idx])
        if abs(diff) >= abs(observed_diff):
            count += 1

    return (count + 1) / (n_bootstrap + 1)


# ====================================================================
# 4. Confidence Interval Computation
# ====================================================================

def wald_ci(estimate: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wald (normal approximation) confidence interval."""
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return estimate - z * se, estimate + z * se


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - margin, centre + margin


def agresti_coull_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Agresti–Coull interval for a binomial proportion."""
    z = sp_stats.norm.ppf(1 - alpha / 2)
    n_tilde = n + z ** 2
    p_tilde = (successes + z ** 2 / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return p_tilde - margin, p_tilde + margin


# ====================================================================
# 5. Kernel Functions
# ====================================================================

def gaussian_kernel(x: FloatArray, y: FloatArray, sigma: float = 1.0) -> float:
    """Gaussian (RBF) kernel: exp(-‖x-y‖² / (2σ²))."""
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma ** 2)))


def gaussian_kernel_matrix(X: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Compute the n×n Gaussian kernel matrix."""
    n = X.shape[0]
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def polynomial_kernel(x: FloatArray, y: FloatArray, degree: int = 3, c: float = 1.0) -> float:
    """Polynomial kernel: (x·y + c)^d."""
    return float((np.dot(x, y) + c) ** degree)


def laplacian_kernel(x: FloatArray, y: FloatArray, sigma: float = 1.0) -> float:
    """Laplacian kernel: exp(-‖x-y‖₁ / σ)."""
    return float(np.exp(-np.sum(np.abs(x - y)) / sigma))


def median_bandwidth(X: FloatArray) -> float:
    """Median heuristic for kernel bandwidth selection.

    Returns the median of pairwise Euclidean distances.
    """
    from scipy.spatial.distance import pdist
    dists = pdist(X, metric="euclidean")
    return float(np.median(dists)) if len(dists) > 0 else 1.0


# ====================================================================
# 6. Distance Metrics
# ====================================================================

def total_variation(p: FloatArray, q: FloatArray) -> float:
    """Total variation distance between two discrete distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def kl_divergence(p: FloatArray, q: FloatArray) -> float:
    """KL divergence D_KL(p ‖ q) for discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-300))))


def hellinger_distance(p: FloatArray, q: FloatArray) -> float:
    """Hellinger distance between two discrete distributions."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def energy_distance(X: FloatArray, Y: FloatArray) -> float:
    """Energy distance between two empirical distributions."""
    from scipy.spatial.distance import cdist
    n, m = X.shape[0], Y.shape[0]
    XY = cdist(X, Y).mean()
    XX = cdist(X, X).mean()
    YY = cdist(Y, Y).mean()
    return float(2 * XY - XX - YY)


def mmd_squared(
    X: FloatArray,
    Y: FloatArray,
    sigma: float = 1.0,
) -> float:
    """Squared maximum mean discrepancy (MMD²) with Gaussian kernel."""
    Kxx = gaussian_kernel_matrix(X, sigma)
    Kyy = gaussian_kernel_matrix(Y, sigma)
    n, m = X.shape[0], Y.shape[0]

    # Cross-kernel
    sq_dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    Kxy = np.exp(-sq_dists / (2 * sigma ** 2))

    return float(Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2 * Kxy.sum() / (n * m))
