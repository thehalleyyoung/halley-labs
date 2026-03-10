"""
usability_oracle.information_theory.estimators — Information measure estimators.

Finite-sample estimators for entropy, mutual information, and related
quantities, with bias correction and confidence intervals.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln

from usability_oracle.information_theory.entropy import (
    _as_prob,
    _safe_log,
    shannon_entropy,
)


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Plug-in (maximum likelihood) estimator
# ═══════════════════════════════════════════════════════════════════════════

def plugin_entropy(
    counts: Union[Sequence[int], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Plug-in (maximum likelihood) entropy estimator.

    Estimates H(X) from observed counts by computing the empirical
    distribution p̂(x) = n_x / N and plugging into Shannon entropy.

    Parameters
    ----------
    counts : array-like of int
        Observed counts for each category.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated entropy.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n == 0:
        return 0.0
    p = c / n
    return shannon_entropy(p, base=base)


def plugin_mutual_information(
    contingency: Union[Sequence[Sequence[int]], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Plug-in mutual information estimator from contingency table.

    Parameters
    ----------
    contingency : 2-D array-like of int
        Contingency table of joint counts.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated mutual information.
    """
    ct = np.asarray(contingency, dtype=np.float64)
    n = ct.sum()
    if n == 0:
        return 0.0
    pxy = ct / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = shannon_entropy(px, base=base)
    hy = shannon_entropy(py, base=base)
    hxy = shannon_entropy(pxy.ravel(), base=base)
    return max(hx + hy - hxy, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Miller-Madow bias correction
# ═══════════════════════════════════════════════════════════════════════════

def miller_madow_entropy(
    counts: Union[Sequence[int], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Miller-Madow bias-corrected entropy estimator.

    Ĥ_MM = Ĥ_plugin + (m - 1) / (2N)

    where m is the number of categories with non-zero counts and N is
    the total sample size.

    Parameters
    ----------
    counts : array-like of int
        Observed counts.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Bias-corrected entropy estimate.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n == 0:
        return 0.0
    h_plugin = plugin_entropy(c, base=base)
    m = np.sum(c > 0)  # number of observed categories
    correction = (m - 1) / (2.0 * n)
    if base != math.e:
        correction /= math.log(base)
    return h_plugin + correction


# ═══════════════════════════════════════════════════════════════════════════
# Jackknife estimator
# ═══════════════════════════════════════════════════════════════════════════

def jackknife_entropy(
    counts: Union[Sequence[int], NDArray],
    *,
    base: float = 2.0,
) -> Tuple[float, float]:
    """Jackknife bias-corrected entropy estimator.

    Uses the delete-1 jackknife to reduce the O(1/N) bias of the
    plug-in estimator.

    Parameters
    ----------
    counts : array-like of int
        Observed counts.
    base : float
        Logarithm base.

    Returns
    -------
    tuple[float, float]
        (jackknife_estimate, standard_error)
    """
    c = np.asarray(counts, dtype=np.float64)
    n = int(c.sum())
    if n <= 1:
        return (0.0, 0.0)

    h_full = plugin_entropy(c, base=base)

    # Compute leave-one-out estimates for each category
    jackknife_values = []
    for i in range(len(c)):
        if c[i] > 0:
            c_loo = c.copy()
            c_loo[i] -= 1
            h_loo = plugin_entropy(c_loo, base=base)
            # Weight by count (each observation in category i gives same LOO)
            for _ in range(int(c[i])):
                jackknife_values.append(h_loo)

    jk = np.array(jackknife_values)
    # Jackknife estimate: n * h_full - (n-1) * mean(h_loo)
    h_jk = n * h_full - (n - 1) * jk.mean()
    # Standard error
    pseudovalues = n * h_full - (n - 1) * jk
    se = float(np.std(pseudovalues, ddof=1) / math.sqrt(n))
    return (max(h_jk, 0.0), se)


# ═══════════════════════════════════════════════════════════════════════════
# Grassberger estimator
# ═══════════════════════════════════════════════════════════════════════════

def grassberger_entropy(
    counts: Union[Sequence[int], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Grassberger entropy estimator.

    Uses the digamma function for improved bias correction:
    Ĥ = log(N) - (1/N) Σ n_i ψ(n_i) - (-1)^{n_i} / (n_i(n_i+1))

    where ψ is the digamma function.

    Reference: Grassberger (2003), "Entropy estimates from insufficient samplings".

    Parameters
    ----------
    counts : array-like of int
        Observed counts.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Grassberger entropy estimate.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n == 0:
        return 0.0

    h = math.log(n)  # in nats
    for ni in c:
        if ni > 0:
            ni = int(ni)
            # Grassberger correction: replace log(n_i) with ψ(n_i) + (-1)^{n_i}/(n_i(n_i+1))
            correction = digamma(ni) + ((-1) ** ni) / (ni * (ni + 1))
            h -= (ni / n) * correction

    if base != math.e:
        h /= math.log(base)
    return max(float(h), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# KSG estimator for continuous MI
# ═══════════════════════════════════════════════════════════════════════════

def ksg_mutual_information(
    x: Union[Sequence[float], NDArray],
    y: Union[Sequence[float], NDArray],
    *,
    k: int = 3,
    base: float = 2.0,
) -> float:
    """KSG (Kraskov-Stögbauer-Grassberger) mutual information estimator.

    Non-parametric MI estimator for continuous variables using
    k-nearest-neighbor distances.

    Reference: Kraskov, Stögbauer, Grassberger (2004), PRE 69, 066138.

    Parameters
    ----------
    x, y : array-like
        Samples from X and Y (same length).
    k : int
        Number of nearest neighbors.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated mutual information.
    """
    from scipy.spatial import KDTree

    x_arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y_arr = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if x_arr.shape[0] == 1:
        x_arr = x_arr.T
    if y_arr.shape[0] == 1:
        y_arr = y_arr.T
    n = x_arr.shape[0]
    if n < k + 1:
        return 0.0

    # Joint space
    xy = np.hstack([x_arr, y_arr])
    tree_xy = KDTree(xy)
    tree_x = KDTree(x_arr)
    tree_y = KDTree(y_arr)

    # For each point, find k-th neighbor distance in joint space
    mi_sum = 0.0
    for i in range(n):
        # k+1 because query includes the point itself
        dists, _ = tree_xy.query(xy[i], k=k + 1)
        eps = dists[-1]  # distance to k-th neighbor
        if eps == 0:
            eps = 1e-10

        # Count neighbors within eps in marginal spaces (Chebyshev norm)
        n_x = tree_x.query_ball_point(x_arr[i], eps - 1e-15)
        n_y = tree_y.query_ball_point(y_arr[i], eps - 1e-15)
        nx = len(n_x) - 1  # exclude the point itself
        ny = len(n_y) - 1

        mi_sum += digamma(max(nx, 1)) + digamma(max(ny, 1))

    mi_nats = digamma(k) - mi_sum / n + digamma(n)
    mi = mi_nats / math.log(base)
    return max(mi, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Nearest-neighbor entropy estimator
# ═══════════════════════════════════════════════════════════════════════════

def nn_entropy(
    samples: Union[Sequence[float], NDArray],
    *,
    k: int = 1,
    base: float = 2.0,
) -> float:
    """k-nearest-neighbor differential entropy estimator.

    Estimates h(X) from samples using:
    ĥ = (d/N) Σ log ρ_k(i) + log(N-1) - ψ(k) + log V_d

    where ρ_k(i) is the distance to the k-th neighbor of sample i,
    d is dimensionality, and V_d is the volume of the d-dimensional unit ball.

    Parameters
    ----------
    samples : array-like
        Samples (1-D or 2-D with rows as samples).
    k : int
        Number of nearest neighbors.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated differential entropy.
    """
    from scipy.spatial import KDTree

    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if x.shape[0] == 1:
        x = x.T
    n, d = x.shape
    if n < k + 1:
        return 0.0

    tree = KDTree(x)
    dists, _ = tree.query(x, k=k + 1)
    rho_k = dists[:, -1]  # k-th neighbor distances

    # Replace zeros with small value
    rho_k = np.maximum(rho_k, 1e-300)

    # Volume of d-dimensional unit ball
    log_vd = (d / 2.0) * math.log(math.pi) - gammaln(d / 2.0 + 1)

    h_nats = (d / n) * np.sum(np.log(rho_k)) + math.log(n - 1) - digamma(k) + log_vd

    if base != math.e:
        return float(h_nats) / math.log(base)
    return float(h_nats)


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_entropy_ci(
    counts: Union[Sequence[int], NDArray],
    *,
    base: float = 2.0,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    estimator: str = "plugin",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for entropy.

    Parameters
    ----------
    counts : array-like of int
        Observed counts.
    base : float
        Logarithm base.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    n_bootstrap : int
        Number of bootstrap resamples.
    estimator : str
        Entropy estimator: "plugin", "miller_madow", or "grassberger".
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng()

    c = np.asarray(counts, dtype=np.float64)
    n = int(c.sum())
    if n == 0:
        return (0.0, 0.0, 0.0)

    est_fn = {
        "plugin": plugin_entropy,
        "miller_madow": miller_madow_entropy,
        "grassberger": grassberger_entropy,
    }.get(estimator, plugin_entropy)

    point = est_fn(c, base=base)

    # Expand counts to samples for bootstrapping
    p_hat = c / n
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Resample from multinomial
        boot_counts = rng.multinomial(n, p_hat)
        bootstrap_estimates.append(est_fn(boot_counts, base=base))

    boot = np.array(bootstrap_estimates)
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2)))

    return (point, ci_lo, ci_hi)


def bootstrap_mi_ci(
    contingency: Union[Sequence[Sequence[int]], NDArray],
    *,
    base: float = 2.0,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mutual information.

    Parameters
    ----------
    contingency : 2-D array-like of int
        Joint count table.
    base : float
        Logarithm base.
    confidence : float
        Confidence level.
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng()

    ct = np.asarray(contingency, dtype=np.float64)
    n = int(ct.sum())
    if n == 0:
        return (0.0, 0.0, 0.0)

    point = plugin_mutual_information(ct, base=base)
    p_flat = ct.ravel() / n
    shape = ct.shape

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        boot_flat = rng.multinomial(n, p_flat)
        boot_ct = boot_flat.reshape(shape)
        bootstrap_estimates.append(plugin_mutual_information(boot_ct, base=base))

    boot = np.array(bootstrap_estimates)
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2)))

    return (point, ci_lo, ci_hi)


# ═══════════════════════════════════════════════════════════════════════════
# Sample size requirements
# ═══════════════════════════════════════════════════════════════════════════

def entropy_sample_size(
    alphabet_size: int,
    precision_bits: float = 0.1,
    confidence: float = 0.95,
) -> int:
    """Minimum sample size for entropy estimation with given precision.

    Uses the Miller-Madow bias formula and normal approximation to
    determine how many samples are needed so that the entropy estimate
    is within ±precision_bits of the true value with the given confidence.

    Parameters
    ----------
    alphabet_size : int
        Number of categories |X|.
    precision_bits : float
        Desired precision (half-width of CI) in bits.
    confidence : float
        Confidence level.

    Returns
    -------
    int
        Minimum sample size.
    """
    from scipy.stats import norm

    if alphabet_size < 2:
        return 1

    z = norm.ppf(0.5 + confidence / 2)
    # Miller-Madow bias is (m-1)/(2N), and variance ≈ (log²(m))/(N)
    # We need: bias + z*se ≤ precision
    # Rough: N ≥ (m-1)/(2*precision) + z²*log²(m)/precision²
    m = alphabet_size
    log_m = math.log2(m)
    bias_term = (m - 1) / (2 * precision_bits)
    var_term = (z * log_m / precision_bits) ** 2
    return max(int(math.ceil(bias_term + var_term)), m)


def mi_sample_size(
    alphabet_size_x: int,
    alphabet_size_y: int,
    precision_bits: float = 0.1,
    confidence: float = 0.95,
) -> int:
    """Minimum sample size for MI estimation with given precision.

    Parameters
    ----------
    alphabet_size_x, alphabet_size_y : int
        Alphabet sizes for X and Y.
    precision_bits : float
        Desired precision in bits.
    confidence : float
        Confidence level.

    Returns
    -------
    int
        Minimum sample size.
    """
    from scipy.stats import norm

    z = norm.ppf(0.5 + confidence / 2)
    mx, my = alphabet_size_x, alphabet_size_y
    # MI estimation bias is O((mx*my - mx - my + 1) / (2N))
    # Joint table has mx*my cells
    m_joint = mx * my
    bias_term = (m_joint - mx - my + 1) / (2 * precision_bits)
    log_m = math.log2(max(m_joint, 2))
    var_term = (z * log_m / precision_bits) ** 2
    return max(int(math.ceil(bias_term + var_term)), m_joint)


__all__ = [
    "bootstrap_entropy_ci",
    "bootstrap_mi_ci",
    "entropy_sample_size",
    "grassberger_entropy",
    "jackknife_entropy",
    "ksg_mutual_information",
    "mi_sample_size",
    "miller_madow_entropy",
    "nn_entropy",
    "plugin_entropy",
    "plugin_mutual_information",
]
