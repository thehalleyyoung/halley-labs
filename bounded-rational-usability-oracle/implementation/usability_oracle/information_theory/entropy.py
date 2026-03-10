"""
usability_oracle.information_theory.entropy — Entropy computations.

Numerically stable implementations of Shannon entropy and its generalizations
used throughout the bounded-rational usability oracle.  All functions operate
on numpy arrays and handle edge cases (zero probabilities, empty distributions)
gracefully.

Convention: all "bits" functions use log base 2; "nats" variants use natural log.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sp_optimize


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_LOG2 = math.log(2.0)
_EPS = np.finfo(np.float64).tiny  # ≈ 5e-324


def _as_prob(p: Union[Sequence[float], NDArray]) -> NDArray:
    """Convert to float64 array, clip tiny negatives to 0."""
    a = np.asarray(p, dtype=np.float64)
    np.clip(a, 0.0, None, out=a)
    return a


def _safe_log2(p: NDArray) -> NDArray:
    """Element-wise log2 that maps 0 → 0 (convention 0·log 0 = 0)."""
    out = np.zeros_like(p)
    mask = p > 0
    out[mask] = np.log2(p[mask])
    return out


def _safe_log(p: NDArray) -> NDArray:
    """Element-wise ln that maps 0 → 0."""
    out = np.zeros_like(p)
    mask = p > 0
    out[mask] = np.log(p[mask])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Shannon entropy
# ═══════════════════════════════════════════════════════════════════════════

def shannon_entropy(p: Union[Sequence[float], NDArray], *, base: float = 2.0) -> float:
    """Shannon entropy H(X) = -Σ p(x) log p(x).

    Parameters
    ----------
    p : array-like
        Probability distribution (must sum to ≈ 1).
    base : float
        Logarithm base (2 for bits, e for nats, 10 for hartleys).

    Returns
    -------
    float
        Entropy in the specified unit.
    """
    p = _as_prob(p)
    if p.size == 0:
        return 0.0
    logp = _safe_log(p)
    h = -float(np.dot(p, logp))
    if base != math.e:
        h /= math.log(base)
    return max(h, 0.0)


def shannon_entropy_bits(p: Union[Sequence[float], NDArray]) -> float:
    """Shannon entropy in bits."""
    return shannon_entropy(p, base=2.0)


def shannon_entropy_nats(p: Union[Sequence[float], NDArray]) -> float:
    """Shannon entropy in nats."""
    return shannon_entropy(p, base=math.e)


# ═══════════════════════════════════════════════════════════════════════════
# Conditional entropy  H(X|Y)
# ═══════════════════════════════════════════════════════════════════════════

def conditional_entropy(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Conditional entropy H(X|Y) from joint distribution p(x, y).

    H(X|Y) = H(X,Y) - H(Y).

    Parameters
    ----------
    joint : 2-D array-like
        Joint distribution p(x, y) with shape (|X|, |Y|).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Conditional entropy H(X|Y).
    """
    pxy = _as_prob(joint)
    py = pxy.sum(axis=0)
    return max(joint_entropy(pxy, base=base) - shannon_entropy(py, base=base), 0.0)


def conditional_entropy_yx(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Conditional entropy H(Y|X) from joint distribution p(x, y).

    H(Y|X) = H(X,Y) - H(X).
    """
    pxy = _as_prob(joint)
    px = pxy.sum(axis=1)
    return max(joint_entropy(pxy, base=base) - shannon_entropy(px, base=base), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Joint entropy  H(X, Y)
# ═══════════════════════════════════════════════════════════════════════════

def joint_entropy(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Joint entropy H(X, Y) = -Σ p(x,y) log p(x,y).

    Parameters
    ----------
    joint : array-like
        Joint distribution (any shape — flattened internally).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Joint entropy.
    """
    pxy = _as_prob(joint).ravel()
    return shannon_entropy(pxy, base=base)


# ═══════════════════════════════════════════════════════════════════════════
# Cross entropy  H(p, q)
# ═══════════════════════════════════════════════════════════════════════════

def cross_entropy(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Cross entropy H(p, q) = -Σ p(x) log q(x).

    Parameters
    ----------
    p, q : array-like
        Probability distributions of the same length.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Cross entropy.  May be +∞ if supp(p) ⊄ supp(q).
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    # Where p > 0 but q == 0 → +∞
    bad = (p_arr > 0) & (q_arr == 0)
    if np.any(bad):
        return float("inf")
    logq = _safe_log(q_arr)
    h = -float(np.dot(p_arr, logq))
    if base != math.e:
        h /= math.log(base)
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Rényi entropy  H_α(X)
# ═══════════════════════════════════════════════════════════════════════════

def renyi_entropy(
    p: Union[Sequence[float], NDArray],
    alpha: float,
    *,
    base: float = 2.0,
) -> float:
    """Rényi entropy of order α.

    H_α(X) = (1/(1-α)) log(Σ p(x)^α)

    Special cases:
      α → 0  : Hartley entropy (log of support size)
      α → 1  : Shannon entropy (limit)
      α → ∞  : min-entropy -log max p(x)

    Parameters
    ----------
    p : array-like
        Probability distribution.
    alpha : float
        Order parameter (α ≥ 0).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Rényi entropy.
    """
    if alpha < 0:
        raise ValueError("Rényi order α must be ≥ 0")
    p_arr = _as_prob(p)
    p_arr = p_arr[p_arr > 0]
    if p_arr.size == 0:
        return 0.0

    if abs(alpha - 1.0) < 1e-12:
        return shannon_entropy(p_arr, base=base)
    if alpha == 0.0:
        h_nats = math.log(p_arr.size)
    elif alpha == float("inf"):
        h_nats = -math.log(float(p_arr.max()))
    else:
        h_nats = (1.0 / (1.0 - alpha)) * math.log(float(np.sum(p_arr ** alpha)))

    if base != math.e:
        return h_nats / math.log(base)
    return h_nats


def min_entropy(p: Union[Sequence[float], NDArray], *, base: float = 2.0) -> float:
    """Min-entropy H_∞(X) = -log max p(x)."""
    return renyi_entropy(p, float("inf"), base=base)


def hartley_entropy(p: Union[Sequence[float], NDArray], *, base: float = 2.0) -> float:
    """Hartley entropy H_0(X) = log |support(X)|."""
    return renyi_entropy(p, 0.0, base=base)


# ═══════════════════════════════════════════════════════════════════════════
# Tsallis entropy
# ═══════════════════════════════════════════════════════════════════════════

def tsallis_entropy(
    p: Union[Sequence[float], NDArray],
    q_param: float,
) -> float:
    """Tsallis entropy of order q.

    S_q(X) = (1/(q-1)) (1 - Σ p(x)^q)

    Reduces to Shannon entropy (in nats) as q → 1.

    Parameters
    ----------
    p : array-like
        Probability distribution.
    q_param : float
        Entropic index (q > 0).

    Returns
    -------
    float
        Tsallis entropy (dimensionless).
    """
    if q_param <= 0:
        raise ValueError("Tsallis index q must be > 0")
    p_arr = _as_prob(p)
    p_arr = p_arr[p_arr > 0]
    if p_arr.size == 0:
        return 0.0

    if abs(q_param - 1.0) < 1e-12:
        return shannon_entropy_nats(p_arr)

    return float((1.0 / (q_param - 1.0)) * (1.0 - np.sum(p_arr ** q_param)))


# ═══════════════════════════════════════════════════════════════════════════
# Differential entropy (continuous)
# ═══════════════════════════════════════════════════════════════════════════

def differential_entropy_gaussian(variance: float) -> float:
    """Differential entropy h(X) for a Gaussian with given variance, in nats.

    h(X) = 0.5 * ln(2πeσ²)

    Parameters
    ----------
    variance : float
        Variance σ² > 0.

    Returns
    -------
    float
        Differential entropy in nats.
    """
    if variance <= 0:
        raise ValueError("Variance must be positive")
    return 0.5 * math.log(2.0 * math.pi * math.e * variance)


def differential_entropy_gaussian_bits(variance: float) -> float:
    """Differential entropy of a Gaussian in bits."""
    return differential_entropy_gaussian(variance) / _LOG2


def differential_entropy_kde(
    samples: Union[Sequence[float], NDArray],
    *,
    n_grid: int = 1024,
    bandwidth: Optional[float] = None,
) -> float:
    """Estimate differential entropy from samples via KDE, in nats.

    Uses Gaussian kernel density estimation on a grid.

    Parameters
    ----------
    samples : array-like
        Samples from the continuous distribution.
    n_grid : int
        Number of grid points for numerical integration.
    bandwidth : float or None
        KDE bandwidth.  If None, uses Silverman's rule.

    Returns
    -------
    float
        Estimated differential entropy in nats.
    """
    from scipy.stats import gaussian_kde

    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    if bandwidth is not None:
        kde = gaussian_kde(x, bw_method=bandwidth)
    else:
        kde = gaussian_kde(x)
    lo, hi = float(x.min()), float(x.max())
    margin = 3.0 * float(kde.factor * x.std())
    grid = np.linspace(lo - margin, hi + margin, n_grid)
    pdf = kde(grid)
    pdf = np.maximum(pdf, _EPS)
    dx = grid[1] - grid[0]
    h = -float(np.trapz(pdf * np.log(pdf), dx=dx))
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Maximum-entropy distributions
# ═══════════════════════════════════════════════════════════════════════════

def max_entropy_discrete(n: int) -> NDArray:
    """Maximum-entropy distribution over n outcomes = uniform.

    Parameters
    ----------
    n : int
        Number of outcomes.

    Returns
    -------
    NDArray
        Uniform distribution of length n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(n, 1.0 / n)


def max_entropy_with_mean(
    n: int,
    target_mean: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> NDArray:
    """Maximum-entropy distribution over {0, 1, ..., n-1} with given mean.

    Uses the exponential family form: p(k) ∝ exp(λ·k) and solves for λ
    via Newton's method.

    Parameters
    ----------
    n : int
        Number of outcomes.
    target_mean : float
        Desired expected value.
    tol : float
        Convergence tolerance on the mean.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    NDArray
        Maximum-entropy distribution with the specified mean constraint.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    values = np.arange(n, dtype=np.float64)
    if not (values[0] <= target_mean <= values[-1]):
        raise ValueError(
            f"target_mean {target_mean} out of range [{values[0]}, {values[-1]}]"
        )
    # If mean is the midpoint → uniform
    if abs(target_mean - values.mean()) < tol:
        return np.full(n, 1.0 / n)

    lam = 0.0
    for _ in range(max_iter):
        log_unnorm = lam * values
        log_unnorm -= log_unnorm.max()  # numerical stability
        unnorm = np.exp(log_unnorm)
        Z = unnorm.sum()
        p = unnorm / Z
        current_mean = float(np.dot(p, values))
        err = current_mean - target_mean
        if abs(err) < tol:
            return p
        # Newton update: dMean/dλ = Var(X)
        var = float(np.dot(p, (values - current_mean) ** 2))
        if var < 1e-30:
            break
        lam -= err / var
    # Return best found
    log_unnorm = lam * values
    log_unnorm -= log_unnorm.max()
    unnorm = np.exp(log_unnorm)
    return unnorm / unnorm.sum()


def max_entropy_with_moments(
    n: int,
    moment_functions: Sequence[Callable[[NDArray], NDArray]],
    moment_targets: Sequence[float],
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> NDArray:
    """Maximum-entropy distribution under multiple moment constraints.

    Solves: max H(p) s.t. E_p[f_k(x)] = μ_k for k = 1..K

    Uses iterative scaling (dual formulation).

    Parameters
    ----------
    n : int
        Number of outcomes.
    moment_functions : sequence of callables
        Each f_k maps array of indices → array of function values.
    moment_targets : sequence of float
        Target moments μ_k.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    NDArray
        Maximum-entropy distribution satisfying the constraints.
    """
    values = np.arange(n, dtype=np.float64)
    K = len(moment_functions)
    if K != len(moment_targets):
        raise ValueError("Number of functions must match number of targets")

    # Evaluate feature functions
    features = np.zeros((K, n))
    for k, f in enumerate(moment_functions):
        features[k] = f(values)

    lambdas = np.zeros(K)
    targets = np.array(moment_targets, dtype=np.float64)

    for _ in range(max_iter):
        log_unnorm = features.T @ lambdas
        log_unnorm -= log_unnorm.max()
        unnorm = np.exp(log_unnorm)
        Z = unnorm.sum()
        p = unnorm / Z
        current_moments = features @ p
        err = current_moments - targets
        if np.max(np.abs(err)) < tol:
            return p
        # Gradient descent on dual with adaptive step
        # Jacobian = -Cov matrix
        cov = features @ np.diag(p) @ features.T - np.outer(current_moments, current_moments)
        try:
            delta = np.linalg.solve(cov, err)
        except np.linalg.LinAlgError:
            delta = err * 0.1
        lambdas -= delta

    log_unnorm = features.T @ lambdas
    log_unnorm -= log_unnorm.max()
    unnorm = np.exp(log_unnorm)
    return unnorm / unnorm.sum()


# ═══════════════════════════════════════════════════════════════════════════
# Entropy rate for Markov chains
# ═══════════════════════════════════════════════════════════════════════════

def entropy_rate_markov(
    transition_matrix: Union[Sequence[Sequence[float]], NDArray],
    *,
    stationary: Optional[Union[Sequence[float], NDArray]] = None,
    base: float = 2.0,
) -> float:
    """Entropy rate of a stationary Markov chain.

    H_rate = Σ_i π(i) H(P(·|i))

    where π is the stationary distribution and P is the transition matrix.

    Parameters
    ----------
    transition_matrix : 2-D array-like
        Row-stochastic transition matrix P(j|i).
    stationary : array-like or None
        Stationary distribution π.  If None, computed from the transition matrix.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Entropy rate.
    """
    P = _as_prob(transition_matrix)
    n = P.shape[0]
    if P.ndim != 2 or P.shape[1] != n:
        raise ValueError("Transition matrix must be square")

    if stationary is None:
        pi = _stationary_distribution(P)
    else:
        pi = _as_prob(stationary)

    rate = 0.0
    for i in range(n):
        rate += pi[i] * shannon_entropy(P[i], base=base)
    return float(rate)


def _stationary_distribution(P: NDArray) -> NDArray:
    """Compute stationary distribution of an irreducible Markov chain.

    Solves π P = π, Σ π_i = 1 via eigenvalue decomposition.
    """
    n = P.shape[0]
    # Find left eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find the eigenvector closest to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    if total < 1e-15:
        return np.full(n, 1.0 / n)
    return pi / total


# ═══════════════════════════════════════════════════════════════════════════
# Batch / vectorized operations
# ═══════════════════════════════════════════════════════════════════════════

def batch_entropy(
    distributions: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> NDArray:
    """Compute Shannon entropy for each row of a 2-D array.

    Parameters
    ----------
    distributions : 2-D array-like
        Each row is a probability distribution.
    base : float
        Logarithm base.

    Returns
    -------
    NDArray
        1-D array of entropies, one per row.
    """
    P = _as_prob(distributions)
    if P.ndim == 1:
        return np.array([shannon_entropy(P, base=base)])
    logP = np.where(P > 0, np.log(P), 0.0)
    h = -np.sum(P * logP, axis=1)
    if base != math.e:
        h /= math.log(base)
    np.maximum(h, 0.0, out=h)
    return h


def batch_cross_entropy(
    p: Union[Sequence[Sequence[float]], NDArray],
    q: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> NDArray:
    """Batch cross entropy: H(p_i, q_i) for each row pair.

    Parameters
    ----------
    p, q : 2-D array-like
        Matched rows of distributions.
    base : float
        Logarithm base.

    Returns
    -------
    NDArray
        1-D array of cross entropies.
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    if p_arr.ndim == 1:
        p_arr = p_arr.reshape(1, -1)
        q_arr = q_arr.reshape(1, -1)

    logq = np.where(q_arr > 0, np.log(q_arr), 0.0)
    # Where p > 0 but q == 0 → inf
    bad = (p_arr > 0) & (q_arr == 0)
    h = -np.sum(p_arr * logq, axis=1)
    # Set inf for bad rows
    bad_rows = np.any(bad, axis=1)
    h[bad_rows] = float("inf")
    if base != math.e:
        h /= math.log(base)
    return h


def batch_renyi_entropy(
    distributions: Union[Sequence[Sequence[float]], NDArray],
    alpha: float,
    *,
    base: float = 2.0,
) -> NDArray:
    """Batch Rényi entropy for each row.

    Parameters
    ----------
    distributions : 2-D array-like
        Each row is a probability distribution.
    alpha : float
        Rényi order.
    base : float
        Logarithm base.

    Returns
    -------
    NDArray
        1-D array of Rényi entropies.
    """
    P = _as_prob(distributions)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if abs(alpha - 1.0) < 1e-12:
        return batch_entropy(P, base=base)

    result = np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        result[i] = renyi_entropy(P[i], alpha, base=base)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Utility: binary entropy
# ═══════════════════════════════════════════════════════════════════════════

def binary_entropy(p: float, *, base: float = 2.0) -> float:
    """Binary entropy function h(p) = -p log p - (1-p) log(1-p).

    Parameters
    ----------
    p : float
        Probability in [0, 1].
    base : float
        Logarithm base.

    Returns
    -------
    float
        Binary entropy.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return shannon_entropy([p, 1.0 - p], base=base)


__all__ = [
    "batch_cross_entropy",
    "batch_entropy",
    "batch_renyi_entropy",
    "binary_entropy",
    "conditional_entropy",
    "conditional_entropy_yx",
    "cross_entropy",
    "differential_entropy_gaussian",
    "differential_entropy_gaussian_bits",
    "differential_entropy_kde",
    "entropy_rate_markov",
    "hartley_entropy",
    "joint_entropy",
    "max_entropy_discrete",
    "max_entropy_with_mean",
    "max_entropy_with_moments",
    "min_entropy",
    "renyi_entropy",
    "shannon_entropy",
    "shannon_entropy_bits",
    "shannon_entropy_nats",
    "tsallis_entropy",
]
