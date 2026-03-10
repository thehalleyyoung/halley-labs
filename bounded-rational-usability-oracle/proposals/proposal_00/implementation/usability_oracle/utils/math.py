"""
usability_oracle.utils.math — Numerically stable mathematical utilities.

All functions operate on numpy arrays and use standard numerical
stability techniques (log-sum-exp trick, epsilon guards, etc.).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linprog
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Basic information measures
# ---------------------------------------------------------------------------

def log2_safe(x: float) -> float:
    """Compute log2(x) safely, returning -inf for x ≤ 0 instead of raising."""
    if x <= 0.0:
        return 0.0
    return math.log2(x)


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits: H(X) = −Σ p(x) log₂ p(x).

    Zero probabilities are handled via the convention 0 log 0 = 0.
    """
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence D_KL(P || Q) in bits.

    Uses an epsilon guard to avoid log(0).  Returns +inf when the support
    of Q does not cover P.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    # Where p > 0 but q ≈ 0, divergence is infinite
    mask = p > 0
    if not np.any(mask):
        return 0.0
    q_safe = np.where(q > _EPS, q, _EPS)
    return float(np.sum(p[mask] * np.log2(p[mask] / q_safe[mask])))


def mutual_information(joint: np.ndarray) -> float:
    """Mutual information I(X;Y) in bits from a joint distribution table.

    ``joint[i, j] = P(X=i, Y=j)``
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
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > _EPS and p_x[i] > _EPS and p_y[j] > _EPS:
                mi += joint[i, j] * math.log2(joint[i, j] / (p_x[i] * p_y[j]))
    return mi


# ---------------------------------------------------------------------------
# Softmax and log-sum-exp
# ---------------------------------------------------------------------------

def log_sum_exp(values: np.ndarray) -> float:
    """Numerically stable log-sum-exp.

    ``log Σ exp(values) = max(values) + log Σ exp(values − max(values))``
    """
    v = np.asarray(values, dtype=float).ravel()
    if len(v) == 0:
        return -math.inf
    c = float(np.max(v))
    if math.isinf(c) and c < 0:
        return -math.inf
    return c + math.log(float(np.sum(np.exp(v - c))))


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional *temperature* scaling.

    ``softmax(v)_i = exp(v_i / T) / Σ exp(v_j / T)``
    """
    v = np.asarray(values, dtype=float).ravel()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = v / temperature
    shifted = scaled - np.max(scaled)
    exp_v = np.exp(shifted)
    return exp_v / exp_v.sum()


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------

def normalize_distribution(values: np.ndarray) -> np.ndarray:
    """Normalise *values* to sum to 1.  Returns uniform if sum is zero."""
    v = np.asarray(values, dtype=float).ravel()
    v = np.maximum(v, 0.0)
    total = v.sum()
    if total <= _EPS:
        n = len(v)
        return np.full(n, 1.0 / n) if n > 0 else v
    return v / total


# ---------------------------------------------------------------------------
# Divergence measures
# ---------------------------------------------------------------------------

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance: TV(P, Q) = ½ Σ |p_i − q_i|."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    return 0.5 * float(np.sum(np.abs(p - q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetrised KL) in bits.

    JSD(P, Q) = ½ D_KL(P || M) + ½ D_KL(Q || M)
    where M = ½(P + Q).
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_distance(
    p: np.ndarray,
    q: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
) -> float:
    """Earth-mover / Wasserstein distance between distributions.

    When *cost_matrix* is ``None`` and scipy is available, uses
    ``scipy.stats.wasserstein_distance`` for 1-D distributions.  Otherwise
    uses a linear-programming formulation with the given cost matrix.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()

    if cost_matrix is None:
        if _HAS_SCIPY:
            return float(_scipy_wasserstein(
                np.arange(len(p)), np.arange(len(q)), u_weights=p, v_weights=q
            ))
        # Fall back to L1 distance of CDFs
        p_norm = p / (p.sum() or 1.0)
        q_norm = q / (q.sum() or 1.0)
        max_len = max(len(p_norm), len(q_norm))
        p_pad = np.pad(p_norm, (0, max_len - len(p_norm)))
        q_pad = np.pad(q_norm, (0, max_len - len(q_norm)))
        return float(np.sum(np.abs(np.cumsum(p_pad) - np.cumsum(q_pad))))

    # LP formulation for general cost matrices
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    m, n = len(p), len(q)
    if cost_matrix.shape != (m, n):
        raise ValueError(f"cost_matrix shape {cost_matrix.shape} != ({m}, {n})")

    if _HAS_SCIPY:
        # Variables: flow[i,j] for all (i,j)
        c = cost_matrix.ravel()
        n_vars = m * n
        # Supply constraints: Σ_j flow[i,j] = p[i]
        A_eq_rows: list[np.ndarray] = []
        b_eq: list[float] = []
        for i in range(m):
            row = np.zeros(n_vars)
            for j in range(n):
                row[i * n + j] = 1.0
            A_eq_rows.append(row)
            b_eq.append(float(p[i]))
        # Demand constraints: Σ_i flow[i,j] = q[j]
        for j in range(n):
            row = np.zeros(n_vars)
            for i in range(m):
                row[i * n + j] = 1.0
            A_eq_rows.append(row)
            b_eq.append(float(q[j]))
        A_eq = np.array(A_eq_rows)
        b_eq_arr = np.array(b_eq)
        bounds = [(0, None)] * n_vars
        result = linprog(c, A_eq=A_eq, b_eq=b_eq_arr, bounds=bounds, method="highs")
        if result.success:
            return float(result.fun)

    # Fallback: sum of absolute CDF differences
    return float(np.sum(np.abs(p - q)))
