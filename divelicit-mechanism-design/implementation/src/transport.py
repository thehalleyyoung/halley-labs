"""Optimal transport, Sinkhorn divergence, and repulsive energy."""

from typing import Optional

import numpy as np


def cost_matrix(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise cost matrix C_ij = c(X_i, Y_j)."""
    if metric == "euclidean":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        C = x_sq + y_sq.T - 2.0 * X @ Y.T
        C = np.maximum(C, 0.0)
        return np.sqrt(C)
    elif metric == "sqeuclidean":
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        C = x_sq + y_sq.T - 2.0 * X @ Y.T
        return np.maximum(C, 0.0)
    elif metric == "cosine":
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return 1.0 - X_norm @ Y_norm.T
    else:
        raise ValueError(f"Unknown metric: {metric}")


def sinkhorn_distance(a: np.ndarray, b: np.ndarray, M: np.ndarray,
                      reg: float = 0.1, n_iter: int = 100,
                      tol: float = 1e-8) -> float:
    """Entropic optimal transport via Sinkhorn-Knopp algorithm.

    Solves: min_{P in U(a,b)} <P, M> + reg * KL(P || ab^T)

    Args:
        a: source marginal (n,)
        b: target marginal (m,)
        M: cost matrix (n, m)
        reg: entropic regularization
        n_iter: max number of Sinkhorn iterations
        tol: convergence tolerance on marginal violation

    Returns:
        Regularized OT cost.
    """
    n, m = M.shape
    K = np.exp(-M / reg)

    u = np.ones(n)
    v = np.ones(m)

    for _ in range(n_iter):
        u_prev = u.copy()
        u = a / np.maximum(K @ v, 1e-30)
        v = b / np.maximum(K.T @ u, 1e-30)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    P = np.diag(u) @ K @ np.diag(v)
    return float(np.sum(P * M))


def transport_plan(a: np.ndarray, b: np.ndarray, M: np.ndarray,
                   reg: float = 0.1, n_iter: int = 100,
                   tol: float = 1e-8) -> np.ndarray:
    """Compute the optimal transport coupling matrix."""
    n, m = M.shape
    K = np.exp(-M / reg)

    u = np.ones(n)
    v = np.ones(m)

    for _ in range(n_iter):
        u_prev = u.copy()
        u = a / np.maximum(K @ v, 1e-30)
        v = b / np.maximum(K.T @ u, 1e-30)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    return np.diag(u) @ K @ np.diag(v)


def sinkhorn_divergence(X: np.ndarray, Y: np.ndarray,
                        reg: float = 0.1, n_iter: int = 50) -> float:
    """Debiased Sinkhorn divergence: S_eps(mu, nu) = OT_eps(mu,nu) - 0.5*OT_eps(mu,mu) - 0.5*OT_eps(nu,nu).

    This is the key diversity objective. Unlike raw entropic OT, Sinkhorn
    divergence is:
    - Non-negative
    - Zero iff mu = nu (metrizes weak convergence)
    - Differentiable w.r.t. sample positions
    - Unbiased (no entropic blur)
    """
    n = X.shape[0]
    m = Y.shape[0]

    a = np.ones(n) / n
    b = np.ones(m) / m

    M_xy = cost_matrix(X, Y, metric="sqeuclidean")
    M_xx = cost_matrix(X, X, metric="sqeuclidean")
    M_yy = cost_matrix(Y, Y, metric="sqeuclidean")

    ot_xy = sinkhorn_distance(a, b, M_xy, reg, n_iter)
    ot_xx = sinkhorn_distance(a, a, M_xx, reg, n_iter)
    ot_yy = sinkhorn_distance(b, b, M_yy, reg, n_iter)

    return max(0.0, ot_xy - 0.5 * ot_xx - 0.5 * ot_yy)


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein distance (closed-form via sorted quantiles)."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


class RepulsiveEnergy:
    """Coulomb-like repulsive energy in embedding space.

    E(y, Y_hist) = -sum_{y_i in Y_hist} log ||y - y_i||

    Points that are close to existing history points receive high (positive)
    energy penalty, encouraging exploration of uncovered regions.
    """

    def __call__(self, y: np.ndarray, history: np.ndarray, eps: float = 1e-6) -> float:
        """Compute repulsive energy of point y given history."""
        if history.ndim < 2 or history.shape[0] == 0:
            return 0.0
        dists = np.linalg.norm(history - y, axis=1)
        dists = np.maximum(dists, eps)
        return float(-np.sum(np.log(dists)))

    def energy_for_index(self, y_idx: int, all_points: np.ndarray, eps: float = 1e-6) -> float:
        """Compute repulsive energy of point at index y_idx against all others."""
        y = all_points[y_idx]
        others = np.delete(all_points, y_idx, axis=0)
        return self(y, others, eps)


def sinkhorn_potentials(X: np.ndarray, Y: np.ndarray,
                        reg: float = 0.1, n_iter: int = 100,
                        tol: float = 1e-8) -> tuple:
    """Compute Sinkhorn dual potentials (f, g) between empirical measures on X and Y.

    The dual potential g(y_j) measures how "underserved" location y_j is by
    the current set X. High g values indicate regions needing more coverage.

    Returns:
        (f, g): dual potential vectors of shape (n,) and (m,).
    """
    n = X.shape[0]
    m = Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    M = cost_matrix(X, Y, metric="sqeuclidean")
    K = np.exp(-M / reg)

    u = np.ones(n)
    v = np.ones(m)

    for _ in range(n_iter):
        u_prev = u.copy()
        u = a / np.maximum(K @ v, 1e-30)
        v = b / np.maximum(K.T @ u, 1e-30)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    # Dual potentials in log-domain
    f = reg * np.log(np.maximum(u, 1e-30))
    g = reg * np.log(np.maximum(v, 1e-30))

    return f, g


def sinkhorn_candidate_scores(candidates: np.ndarray, history: np.ndarray,
                               reference: np.ndarray, reg: float = None,
                               n_iter: int = 50) -> np.ndarray:
    """Score each candidate by marginal Sinkhorn divergence reduction.

    For each candidate, computes how much adding it to history reduces
    the Sinkhorn divergence to the reference distribution. This directly
    measures coverage improvement rather than using a heuristic proxy.

    Uses cosine distance for numerical stability in high dimensions.
    Regularization defaults to 0.1 * median pairwise distance.
    """
    n_cand = candidates.shape[0]
    scores = np.zeros(n_cand)

    if history.shape[0] == 0:
        # No history: score by distance to centroid (favor spread)
        centroid = np.mean(reference, axis=0)
        for i in range(n_cand):
            scores[i] = np.linalg.norm(candidates[i] - centroid)
        return scores

    # Auto-tune regularization based on distance scale
    if reg is None:
        sample = np.vstack([history, reference[:min(50, len(reference))]])
        C_sample = cost_matrix(sample, sample, metric="cosine")
        med = float(np.median(C_sample[C_sample > 1e-10]))
        reg = max(0.05 * med, 0.01)

    # Current Sinkhorn divergence (using cosine cost)
    current_div = _sinkhorn_divergence_cosine(history, reference, reg, n_iter)

    # Score each candidate by marginal divergence reduction
    for i in range(n_cand):
        augmented = np.vstack([history, candidates[i:i+1]])
        new_div = _sinkhorn_divergence_cosine(augmented, reference, reg, n_iter)
        scores[i] = current_div - new_div

    return scores


def _sinkhorn_divergence_cosine(X: np.ndarray, Y: np.ndarray,
                                  reg: float = 0.05, n_iter: int = 50) -> float:
    """Sinkhorn divergence using cosine distance (numerically stable in high-d)."""
    n = X.shape[0]
    m = Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    M_xy = cost_matrix(X, Y, metric="cosine")
    M_xx = cost_matrix(X, X, metric="cosine")
    M_yy = cost_matrix(Y, Y, metric="cosine")

    ot_xy = sinkhorn_distance(a, b, M_xy, reg, n_iter)
    ot_xx = sinkhorn_distance(a, a, M_xx, reg, n_iter)
    ot_yy = sinkhorn_distance(b, b, M_yy, reg, n_iter)

    return max(0.0, ot_xy - 0.5 * ot_xx - 0.5 * ot_yy)


def sinkhorn_candidate_scores_fast(candidates: np.ndarray, history: np.ndarray,
                                    reference: np.ndarray, reg: float = 0.05,
                                    n_iter: int = 50) -> np.ndarray:
    """Fast approximate scoring using dual potentials (for large N).

    Uses dual potential g to approximate marginal divergence reduction
    without recomputing full Sinkhorn for each candidate.
    """
    n_cand = candidates.shape[0]
    scores = np.zeros(n_cand)

    if history.shape[0] == 0:
        return np.ones(n_cand)

    _, g = sinkhorn_potentials(history, reference, reg, n_iter)

    for i in range(n_cand):
        dists = np.sum((reference - candidates[i]) ** 2, axis=1)
        weights = np.exp(-dists / (2 * reg))
        scores[i] = np.dot(weights, g)

    return scores


def sinkhorn_gradient(X: np.ndarray, Y: np.ndarray,
                      reg: float = 0.1, n_iter: int = 50,
                      delta: float = 1e-4) -> np.ndarray:
    """Gradient of Sinkhorn divergence w.r.t. X positions (numerical).

    Enables gradient-based diversity optimization: move candidate positions
    to maximize diversity from the target set Y.
    """
    n, d = X.shape
    grad = np.zeros_like(X)

    base = sinkhorn_divergence(X, Y, reg, n_iter)

    for i in range(n):
        for j in range(d):
            X_pert = X.copy()
            X_pert[i, j] += delta
            perturbed = sinkhorn_divergence(X_pert, Y, reg, n_iter)
            grad[i, j] = (perturbed - base) / delta

    return grad
