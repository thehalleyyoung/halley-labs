"""
usability_oracle.utils.entropy — Information-theoretic computations.

Implements the Blahut-Arimoto algorithm for channel capacity, rate-
distortion function computation, conditional entropy, information gain,
and effective number (perplexity).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Channel capacity (Blahut-Arimoto)
# ---------------------------------------------------------------------------

def channel_capacity(
    transition_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> float:
    """Compute the channel capacity C = max_p I(X;Y) using the Blahut-Arimoto algorithm.

    Parameters:
        transition_matrix: P(Y|X) matrix of shape ``(|X|, |Y|)`` where rows
            are input symbols and columns are output symbols.  Each row must
            sum to 1.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the capacity estimate.

    Returns:
        Channel capacity in bits.
    """
    W = np.asarray(transition_matrix, dtype=float)
    if W.ndim != 2:
        raise ValueError("transition_matrix must be 2-D")
    n_x, n_y = W.shape
    if n_x == 0 or n_y == 0:
        return 0.0

    # Normalise rows
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    W = W / row_sums

    # Initialise input distribution to uniform
    p = np.full(n_x, 1.0 / n_x)
    capacity = 0.0

    for _ in range(max_iter):
        # Output distribution q(y) = Σ_x p(x) W(y|x)
        q = p @ W  # shape (n_y,)
        q = np.maximum(q, _EPS)

        # Compute c(x) = Σ_y W(y|x) log2(W(y|x) / q(y))
        # Using log-ratio: for each x, sum over y
        log_ratio = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                if W[i, j] > _EPS:
                    log_ratio[i, j] = W[i, j] * math.log2(W[i, j] / q[j])
        c = log_ratio.sum(axis=1)  # shape (n_x,)

        # Update input distribution
        exp_c = np.exp2(c)
        p_new = p * exp_c
        total = p_new.sum()
        if total > _EPS:
            p_new /= total
        else:
            p_new = np.full(n_x, 1.0 / n_x)

        # Capacity bounds
        cap_lower = float(np.sum(p_new * c))
        cap_upper = float(np.log2(np.sum(p * exp_c)))

        new_capacity = 0.5 * (cap_lower + cap_upper)
        if abs(new_capacity - capacity) < tol:
            capacity = new_capacity
            break
        capacity = new_capacity
        p = p_new

    return max(capacity, 0.0)


# ---------------------------------------------------------------------------
# Rate-distortion
# ---------------------------------------------------------------------------

def rate_distortion(
    source_dist: np.ndarray,
    distortion_matrix: np.ndarray,
    target_rate: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Compute the rate-distortion function R(D) via the Blahut-Arimoto algorithm.

    Parameters:
        source_dist: Source distribution p(x) of length ``|X|``.
        distortion_matrix: d(x, x̂) matrix of shape ``(|X|, |X̂|)``.
        target_rate: Target rate in bits.

    Returns:
        (achieved_distortion, conditional_distribution q(x̂|x))
    """
    p = np.asarray(source_dist, dtype=float).ravel()
    D = np.asarray(distortion_matrix, dtype=float)
    n_x = len(p)
    n_xhat = D.shape[1] if D.ndim == 2 else n_x

    # Binary search for the Lagrange multiplier beta
    beta_lo, beta_hi = 0.01, 100.0
    best_dist = float("inf")
    best_q = np.full((n_x, n_xhat), 1.0 / n_xhat)

    for _ in range(50):
        beta = (beta_lo + beta_hi) / 2.0
        q_cond, rate, distortion = _ba_rd_iteration(p, D, beta, max_iter, tol)
        if rate > target_rate:
            beta_hi = beta
        else:
            beta_lo = beta
        if abs(rate - target_rate) < tol:
            return distortion, q_cond
        if abs(distortion - best_dist) > tol:
            best_dist = distortion
            best_q = q_cond

    return best_dist, best_q


def _ba_rd_iteration(
    p: np.ndarray,
    D: np.ndarray,
    beta: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, float]:
    """Single Blahut-Arimoto iteration for rate-distortion at given beta."""
    n_x, n_xhat = D.shape
    # Initialise reproduction distribution to uniform
    m = np.full(n_xhat, 1.0 / n_xhat)

    q_cond = np.zeros((n_x, n_xhat))

    for _ in range(max_iter):
        # q(x̂|x) ∝ m(x̂) exp(-β d(x, x̂))
        for i in range(n_x):
            log_q = np.log(np.maximum(m, _EPS)) - beta * D[i, :]
            log_q -= np.max(log_q)  # numerical stability
            q_row = np.exp(log_q)
            q_row /= q_row.sum()
            q_cond[i, :] = q_row

        # Update m(x̂) = Σ_x p(x) q(x̂|x)
        m_new = p @ q_cond
        m_new = np.maximum(m_new, _EPS)
        m_new /= m_new.sum()

        if np.max(np.abs(m_new - m)) < tol:
            m = m_new
            break
        m = m_new

    # Compute rate and distortion
    rate = 0.0
    distortion = 0.0
    for i in range(n_x):
        for j in range(n_xhat):
            if q_cond[i, j] > _EPS and m[j] > _EPS:
                rate += p[i] * q_cond[i, j] * math.log2(q_cond[i, j] / m[j])
            distortion += p[i] * q_cond[i, j] * D[i, j]

    return q_cond, max(rate, 0.0), distortion


# ---------------------------------------------------------------------------
# Conditional entropy
# ---------------------------------------------------------------------------

def conditional_entropy(joint: np.ndarray) -> float:
    """Conditional entropy H(Y|X) from a joint distribution P(X, Y).

    H(Y|X) = H(X,Y) − H(X) = −Σ p(x,y) log₂ p(y|x)
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D array")
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    p_x = joint.sum(axis=1)
    h = 0.0
    for i in range(joint.shape[0]):
        if p_x[i] <= _EPS:
            continue
        for j in range(joint.shape[1]):
            if joint[i, j] > _EPS:
                p_y_given_x = joint[i, j] / p_x[i]
                h -= joint[i, j] * math.log2(p_y_given_x)
    return h


# ---------------------------------------------------------------------------
# Information gain
# ---------------------------------------------------------------------------

def information_gain(prior: np.ndarray, posterior: np.ndarray) -> float:
    """Information gain: IG = H(prior) − H(posterior).

    Positive values indicate that the posterior is more certain
    (lower entropy) than the prior.
    """
    h_prior = _entropy_internal(prior)
    h_post = _entropy_internal(posterior)
    return h_prior - h_post


def _entropy_internal(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Effective number (perplexity)
# ---------------------------------------------------------------------------

def effective_number(probs: np.ndarray) -> float:
    """Effective number of outcomes: 2^H(P).

    Also known as the perplexity of the distribution.
    """
    h = _entropy_internal(probs)
    return 2.0 ** h


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence D_KL(P || Q) in bits.

    D_KL(P || Q) = Σ p(x) log₂(p(x) / q(x))
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    mask = p > _EPS
    q_safe = np.maximum(q[mask], _EPS)
    return float(np.sum(p[mask] * np.log2(p[mask] / q_safe)))


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence JSD(P || Q) in bits.

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = (P+Q)/2.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ---------------------------------------------------------------------------
# Mutual information from joint
# ---------------------------------------------------------------------------

def mutual_information(joint: np.ndarray) -> float:
    """Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) from joint P(X,Y)."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D array")
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    h_x = _entropy_internal(p_x)
    h_y = _entropy_internal(p_y)
    h_xy = _entropy_internal(joint.ravel())
    return max(h_x + h_y - h_xy, 0.0)


# ---------------------------------------------------------------------------
# Normalised mutual information
# ---------------------------------------------------------------------------

def normalised_mutual_information(joint: np.ndarray) -> float:
    """Normalised MI: NMI = 2 * I(X;Y) / (H(X) + H(Y)).

    Values in [0, 1]; 1 means perfect agreement.
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D array")
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    h_x = _entropy_internal(p_x)
    h_y = _entropy_internal(p_y)
    denom = h_x + h_y
    if denom < _EPS:
        return 0.0
    mi = mutual_information(joint * total)  # undo normalisation for the call
    return 2.0 * mi / denom


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy H(P, Q) = -Σ p(x) log₂ q(x)."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    q_safe = np.maximum(q, _EPS)
    return float(-np.sum(p * np.log2(q_safe)))


# ---------------------------------------------------------------------------
# Rényi entropy
# ---------------------------------------------------------------------------

def renyi_entropy(probs: np.ndarray, alpha: float = 2.0) -> float:
    """Rényi entropy of order α.

    H_α(P) = (1/(1-α)) log₂(Σ p(x)^α)

    Special cases: α → 1 gives Shannon entropy, α = 2 gives collision entropy.
    """
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-10:
        return _entropy_internal(p)
    return float(math.log2(np.sum(p ** alpha)) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# Tsallis entropy
# ---------------------------------------------------------------------------

def tsallis_entropy(probs: np.ndarray, q: float = 2.0) -> float:
    """Tsallis entropy of order q.

    S_q(P) = (1/(q-1)) (1 - Σ p(x)^q)
    """
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    if abs(q - 1.0) < 1e-10:
        return _entropy_internal(p) * math.log(2)  # convert to nats-like
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


# ---------------------------------------------------------------------------
# Entropy rate of a Markov chain
# ---------------------------------------------------------------------------

def markov_entropy_rate(transition_matrix: np.ndarray) -> float:
    """Entropy rate of a stationary Markov chain.

    H_rate = -Σ_i π_i Σ_j P(j|i) log₂ P(j|i)

    where π is the stationary distribution.
    """
    P = np.asarray(transition_matrix, dtype=float)
    n = P.shape[0]
    if n == 0:
        return 0.0

    # Compute stationary distribution via eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvector corresponding to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.abs(eigenvectors[:, idx])
    pi = pi / pi.sum()

    h_rate = 0.0
    for i in range(n):
        for j in range(n):
            if P[i, j] > _EPS:
                h_rate -= pi[i] * P[i, j] * math.log2(P[i, j])
    return max(h_rate, 0.0)


# ---------------------------------------------------------------------------
# Maximum entropy distribution
# ---------------------------------------------------------------------------

def max_entropy_distribution(n: int) -> np.ndarray:
    """Return the maximum entropy (uniform) distribution over n outcomes."""
    if n <= 0:
        return np.array([])
    return np.full(n, 1.0 / n)
