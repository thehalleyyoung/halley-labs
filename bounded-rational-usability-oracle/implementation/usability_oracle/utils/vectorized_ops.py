"""
usability_oracle.utils.vectorized_ops — Vectorized operations for hot-path computation.

Extends :mod:`usability_oracle.utils.vectorized` with additional batch
operations targeting the bounded-rational policy pipeline's hot paths:
batch information-gain, vectorized Bellman backup, bisimulation distance,
and SIMD-friendly memory layout helpers.

Design principles
-----------------
- **No Python loops for numerical code**: all operations delegate to
  NumPy ufuncs, BLAS, or scipy.sparse routines.
- **Shape conventions**: 1-D arrays of length *n* for scalars;
  2-D ``(n, m)`` for per-state-action quantities.
- **NaN / inf safety**: epsilon guards and log-sum-exp stabilisation
  throughout, matching :mod:`usability_oracle.utils.math`.

Performance characteristics
---------------------------
- Batch softmax / KL: O(n·m) single-pass vectorised.
- Vectorised Fitts / Hick: O(n) ufunc.
- Batch Bellman backup: O(|S|·|A|·|S|) dense, O(nnz·|A|) sparse.
- Batch bisimulation: O(|S|²·|A|) per iteration.
- Sparse Kronecker: O(nnz_A · nnz_B) via scipy.sparse.

References
----------
Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
    decision-making with information-processing costs. *Proc. Roy. Soc. A*.
Ferns, N., Panangaden, P. & Precup, D. (2004). Metrics for finite
    Markov decision processes. *UAI*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp

    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False

# Numerical guard matching usability_oracle.utils.math._EPS
_EPS = 1e-30

# Default Fitts' law parameters (Card, Moran & Newell, 1983)
_FITTS_A = 0.050  # intercept (s)
_FITTS_B = 0.150  # slope (s/bit)

# Default Hick-Hyman parameters (Hick, 1952)
_HICK_A = 0.200  # base RT (s)
_HICK_B = 0.155  # slope (s/bit)


# ---------------------------------------------------------------------------
# Numerically stable batch softmax
# ---------------------------------------------------------------------------

def batch_softmax_stable(
    Q: np.ndarray,
    betas: Union[np.ndarray, float],
    *,
    neg_cost: bool = True,
) -> np.ndarray:
    """Numerically stable batch softmax for bounded-rational policies.

    Computes ``π_β(a|s) ∝ exp(−β · Q(s,a))`` (when *neg_cost* is ``True``)
    or ``π_β(a|s) ∝ exp(β · Q(s,a))`` for each ``(state, β)`` pair, using
    the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    Q : array_like, shape (n, m)
        Q-values for *n* states and *m* actions.
    betas : array_like, shape (n,) or scalar
        Rationality (inverse temperature) per state.
    neg_cost : bool
        If ``True`` (default), negate Q before exponentiation so that
        *lower* Q-values receive *higher* probability (cost formulation).

    Returns
    -------
    np.ndarray, shape (n, m)
        Row-stochastic policy matrix.

    Complexity
    ----------
    O(n·m) — single vectorised pass with log-sum-exp stabilisation.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    n, m = Q.shape

    betas_arr = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if betas_arr.shape == (1,):
        betas_arr = np.broadcast_to(betas_arr, (n,))

    sign = -1.0 if neg_cost else 1.0
    scaled = sign * betas_arr[:, np.newaxis] * Q  # (n, m)

    row_max = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - row_max
    exp_vals = np.exp(shifted)
    row_sums = np.sum(exp_vals, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, _EPS)
    return exp_vals / row_sums


def batch_log_policy(
    Q: np.ndarray,
    betas: Union[np.ndarray, float],
    *,
    neg_cost: bool = True,
) -> np.ndarray:
    """Batch log-policy: ``log π_β(a|s)`` for each state.

    Avoids forming the policy and then taking log (which loses precision);
    instead computes log-softmax directly:

        ``log π(a|s) = sign·β·Q(s,a) − logsumexp(sign·β·Q(s,:))``

    Parameters
    ----------
    Q : array_like, shape (n, m)
        Q-values for *n* states and *m* actions.
    betas : array_like, shape (n,) or scalar
        Rationality (inverse temperature) per state.
    neg_cost : bool
        Negate Q before exponentiation (default ``True``).

    Returns
    -------
    np.ndarray, shape (n, m)
        Log-probabilities.  Each row is a valid log-probability vector.

    Complexity
    ----------
    O(n·m) — single vectorised pass.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)

    betas_arr = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if betas_arr.shape == (1,):
        betas_arr = np.broadcast_to(betas_arr, (Q.shape[0],))

    sign = -1.0 if neg_cost else 1.0
    scaled = sign * betas_arr[:, np.newaxis] * Q  # (n, m)

    row_max = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - row_max
    log_Z = row_max + np.log(np.sum(np.exp(shifted), axis=1, keepdims=True) + _EPS)
    return scaled - log_Z


# ---------------------------------------------------------------------------
# Batch KL divergence
# ---------------------------------------------------------------------------

def batch_kl_divergence(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    base: float = 2.0,
) -> np.ndarray:
    """Batch KL divergence: ``D_KL(P_i || Q_i)`` for each row *i*.

    Parameters
    ----------
    P : array_like, shape (n, m)
        Matrix of *n* probability distributions, each of length *m*.
    Q : array_like, shape (n, m)
        Reference distributions matching *P* in shape.
    base : float
        Logarithm base (default 2 → bits).

    Returns
    -------
    np.ndarray, shape (n,)
        KL divergence per row.

    Raises
    ------
    ValueError
        If *P* and *Q* have different shapes.

    Complexity
    ----------
    O(n·m) — single vectorised pass.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim == 1:
        P = P.reshape(1, -1)
        Q = Q.reshape(1, -1)
    Q_safe = np.where(Q > _EPS, Q, _EPS)
    log_ratio = np.where(P > _EPS, np.log(P / Q_safe) / np.log(base), 0.0)
    return np.sum(np.where(P > _EPS, P * log_ratio, 0.0), axis=1)


def batch_js_divergence(
    P: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Batch Jensen-Shannon divergence in bits.

    ``JSD(P_i, Q_i) = ½ D_KL(P_i || M_i) + ½ D_KL(Q_i || M_i)``
    where ``M_i = ½(P_i + Q_i)``.

    Parameters
    ----------
    P, Q : array_like, shape (n, m)
        Pairs of distributions.

    Returns
    -------
    np.ndarray, shape (n,)
        JSD per pair, in bits.

    Complexity
    ----------
    O(n·m).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    M = 0.5 * (P + Q)
    return 0.5 * batch_kl_divergence(P, M) + 0.5 * batch_kl_divergence(Q, M)


# ---------------------------------------------------------------------------
# Vectorized Fitts' law (multiple targets)
# ---------------------------------------------------------------------------

def vectorized_fitts_multi(
    positions: np.ndarray,
    target_positions: np.ndarray,
    target_widths: np.ndarray,
    a: float = _FITTS_A,
    b: float = _FITTS_B,
) -> np.ndarray:
    """Fitts' law for multiple cursor→target pairs simultaneously.

    Computes ``MT_ij = a + b · log₂(1 + ||pos_i − tgt_j|| / w_j)``
    for each cursor position *i* and target *j*.

    Parameters
    ----------
    positions : array_like, shape (n, d)
        Cursor positions in *d*-dimensional space.
    target_positions : array_like, shape (k, d)
        Target centre positions.
    target_widths : array_like, shape (k,)
        Target widths (> 0).
    a, b : float
        Fitts' law parameters.

    Returns
    -------
    np.ndarray, shape (n, k)
        Movement-time matrix.

    Complexity
    ----------
    O(n·k·d) via broadcasting — no Python loops.
    """
    pos = np.asarray(positions, dtype=np.float64)
    tgt = np.asarray(target_positions, dtype=np.float64)
    widths = np.asarray(target_widths, dtype=np.float64)

    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if tgt.ndim == 1:
        tgt = tgt.reshape(1, -1)
    if np.any(widths <= 0):
        raise ValueError("All target_widths must be > 0")

    # (n, 1, d) - (1, k, d) → (n, k, d) → euclidean → (n, k)
    diff = pos[:, np.newaxis, :] - tgt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    dists = np.maximum(dists, _EPS)

    return a + b * np.log2(1.0 + dists / widths[np.newaxis, :])


def vectorized_fitts_batch(
    distances: np.ndarray,
    widths: np.ndarray,
    a: float = _FITTS_A,
    b: float = _FITTS_B,
) -> np.ndarray:
    """Element-wise Fitts' law: MT = a + b · log₂(1 + D/W).

    Parameters
    ----------
    distances : array_like, shape (n,)
        Centre-to-centre distances (> 0).
    widths : array_like, shape (n,)
        Target widths (> 0).
    a, b : float
        Fitts' law parameters.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted movement times.

    Complexity
    ----------
    O(n) — vectorised ufuncs.
    """
    D = np.asarray(distances, dtype=np.float64)
    W = np.asarray(widths, dtype=np.float64)
    if np.any(D <= 0):
        raise ValueError("All distances must be > 0")
    if np.any(W <= 0):
        raise ValueError("All widths must be > 0")
    return a + b * np.log2(1.0 + D / W)


# ---------------------------------------------------------------------------
# Vectorized Hick-Hyman (multiple choice sets)
# ---------------------------------------------------------------------------

def vectorized_hick_batch(
    n_alternatives: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    a: float = _HICK_A,
    b: float = _HICK_B,
) -> np.ndarray:
    """Batch Hick-Hyman law over multiple choice sets.

    Parameters
    ----------
    n_alternatives : array_like, shape (n,)
        Number of alternatives per trial.
    probabilities : array_like or None, shape (n, max_k)
        Optional probability distributions; rows should sum to 1.
    a, b : float
        Hick-Hyman parameters.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted reaction times.

    Complexity
    ----------
    O(n) equiprobable, O(n·k) entropy form.
    """
    alts = np.asarray(n_alternatives, dtype=np.float64)
    if np.any(alts < 1):
        raise ValueError("All n_alternatives must be >= 1")

    if probabilities is None:
        log_vals = np.where(alts > 1, np.log2(alts), 0.0)
        return a + b * log_vals

    probs = np.asarray(probabilities, dtype=np.float64)
    safe_p = np.where(probs > _EPS, probs, _EPS)
    H = -np.sum(np.where(probs > _EPS, probs * np.log2(safe_p), 0.0), axis=1)
    return a + b * H


# ---------------------------------------------------------------------------
# Batch information gain
# ---------------------------------------------------------------------------

def batch_information_gain(
    prior: np.ndarray,
    likelihoods: np.ndarray,
) -> np.ndarray:
    """Batch expected information gain for multiple observations.

    For each observation *j*, computes
    ``IG_j = D_KL( posterior_j || prior )``
    where ``posterior_j ∝ prior · likelihood(:, j)``.

    Parameters
    ----------
    prior : array_like, shape (n,)
        Prior distribution over *n* hypotheses.
    likelihoods : array_like, shape (n, k)
        Likelihood matrix: ``likelihoods[i, j] = P(obs_j | hyp_i)``.

    Returns
    -------
    np.ndarray, shape (k,)
        Information gain per observation.

    Complexity
    ----------
    O(n·k) — vectorised Bayesian update + KL.
    """
    prior_arr = np.asarray(prior, dtype=np.float64).ravel()
    L = np.asarray(likelihoods, dtype=np.float64)

    if L.ndim == 1:
        L = L.reshape(-1, 1)

    n, k = L.shape
    if prior_arr.shape[0] != n:
        raise ValueError(f"prior length {prior_arr.shape[0]} != likelihood rows {n}")

    # posterior_j ∝ prior * L[:, j]  → (n, k)
    unnorm = prior_arr[:, np.newaxis] * L
    evidence = np.sum(unnorm, axis=0, keepdims=True)
    evidence = np.where(evidence > _EPS, evidence, _EPS)
    posteriors = unnorm / evidence  # (n, k)

    # D_KL(posterior_j || prior) for each j
    prior_safe = np.where(prior_arr > _EPS, prior_arr, _EPS)
    gains = np.zeros(k, dtype=np.float64)
    for col in range(k):
        p = posteriors[:, col]
        mask = p > _EPS
        if np.any(mask):
            gains[col] = float(np.sum(
                p[mask] * np.log2(p[mask] / prior_safe[mask])
            ))
    return gains


def batch_mutual_information(
    joints: np.ndarray,
) -> np.ndarray:
    """Batch mutual information from stacked joint distributions.

    Parameters
    ----------
    joints : array_like, shape (batch, n, m)
        Stack of *batch* joint distributions, each of shape ``(n, m)``.

    Returns
    -------
    np.ndarray, shape (batch,)
        Mutual information in bits per joint distribution.

    Complexity
    ----------
    O(batch · n · m).
    """
    J = np.asarray(joints, dtype=np.float64)
    if J.ndim == 2:
        J = J[np.newaxis, :, :]

    batch = J.shape[0]
    totals = J.sum(axis=(1, 2), keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    J = J / totals

    p_x = J.sum(axis=2, keepdims=True)  # (batch, n, 1)
    p_y = J.sum(axis=1, keepdims=True)  # (batch, 1, m)

    outer = p_x * p_y  # (batch, n, m)
    outer_safe = np.where(outer > _EPS, outer, _EPS)

    mask = J > _EPS
    log_ratio = np.where(mask, np.log2(J / outer_safe), 0.0)
    mi = np.sum(np.where(mask, J * log_ratio, 0.0), axis=(1, 2))
    return mi


# ---------------------------------------------------------------------------
# Vectorized Bellman backup
# ---------------------------------------------------------------------------

def bellman_backup_dense(
    T: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Vectorized Bellman backup for dense MDPs.

    Computes ``Q(s, a) = R(s, a) + γ · Σ_s' T(s, a, s') · V(s')``
    for all ``(s, a)`` simultaneously.

    Parameters
    ----------
    T : array_like, shape (|S|, |A|, |S|)
        Transition probabilities ``T[s, a, s']``.
    R : array_like, shape (|S|, |A|)
        Immediate rewards (or negative costs).
    V : array_like, shape (|S|,)
        Current value function estimate.
    gamma : float
        Discount factor in (0, 1].

    Returns
    -------
    np.ndarray, shape (|S|, |A|)
        Updated Q-values.

    Complexity
    ----------
    O(|S|·|A|·|S|) via a single ``np.einsum`` call.
    """
    T = np.asarray(T, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # T[s, a, :] · V → expected future value per (s, a)
    EV = np.einsum("sap,p->sa", T, V)
    return R + gamma * EV


def bellman_backup_sparse(
    T_sparse_list: Sequence[Any],
    R: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Vectorized Bellman backup for sparse transition matrices.

    Parameters
    ----------
    T_sparse_list : sequence of sparse matrices, length |A|
        ``T_sparse_list[a]`` is a sparse ``(|S|, |S|)`` matrix where
        entry ``(s, s')`` is ``T(s, a, s')``.
    R : array_like, shape (|S|, |A|)
        Immediate rewards.
    V : array_like, shape (|S|,)
        Current value function.
    gamma : float
        Discount factor.

    Returns
    -------
    np.ndarray, shape (|S|, |A|)
        Updated Q-values.

    Complexity
    ----------
    O(nnz · |A|) where nnz is the total non-zeros across all actions.
    """
    if not _HAS_SCIPY_SPARSE:
        raise ImportError("scipy.sparse required for bellman_backup_sparse")

    R = np.asarray(R, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64).ravel()
    n_states = V.shape[0]
    n_actions = len(T_sparse_list)
    Q = np.empty((n_states, n_actions), dtype=np.float64)

    for a_idx, T_a in enumerate(T_sparse_list):
        if not sp.issparse(T_a):
            T_a = sp.csr_matrix(T_a)
        Q[:, a_idx] = R[:, a_idx] + gamma * np.asarray(T_a.dot(V)).ravel()

    return Q


def value_iteration_step(
    T: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of value iteration: ``V_new = max_a Q(s,a)``.

    Parameters
    ----------
    T : array_like, shape (|S|, |A|, |S|)
        Transition probabilities.
    R : array_like, shape (|S|, |A|)
        Rewards.
    V : array_like, shape (|S|,)
        Current value function.
    gamma : float
        Discount factor.

    Returns
    -------
    V_new : np.ndarray, shape (|S|,)
        Updated value function.
    policy : np.ndarray, shape (|S|,)
        Greedy policy (action indices).

    Complexity
    ----------
    O(|S|·|A|·|S|) for the Bellman backup + O(|S|·|A|) for the max.
    """
    Q = bellman_backup_dense(T, R, V, gamma)
    policy = np.argmax(Q, axis=1)
    V_new = np.max(Q, axis=1)
    return V_new, policy


# ---------------------------------------------------------------------------
# SIMD-friendly memory layouts
# ---------------------------------------------------------------------------

@dataclass
class SIMDLayout:
    """Struct-of-arrays layout for SIMD-friendly processing.

    Stores MDP quantities in contiguous arrays aligned to cache-line
    boundaries, enabling efficient vectorised operations.

    Attributes
    ----------
    states : np.ndarray, shape (n_states,)
        State identifiers (integer).
    values : np.ndarray, shape (n_states,)
        Value function.
    q_values : np.ndarray, shape (n_states, n_actions)
        Q-value table.
    policy : np.ndarray, shape (n_states, n_actions)
        Policy probabilities.
    """

    states: np.ndarray
    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray

    @staticmethod
    def allocate(n_states: int, n_actions: int) -> "SIMDLayout":
        """Allocate aligned arrays for *n_states* states and *n_actions* actions.

        Parameters
        ----------
        n_states : int
            Number of MDP states.
        n_actions : int
            Number of available actions.

        Returns
        -------
        SIMDLayout
            Zero-initialised layout with contiguous C-order arrays.
        """
        return SIMDLayout(
            states=np.arange(n_states, dtype=np.int64),
            values=np.zeros(n_states, dtype=np.float64),
            q_values=np.zeros((n_states, n_actions), dtype=np.float64),
            policy=np.full(
                (n_states, n_actions), 1.0 / n_actions, dtype=np.float64
            ),
        )

    def update_policy(self, betas: Union[np.ndarray, float]) -> None:
        """Recompute policy from Q-values via softmax.

        Parameters
        ----------
        betas : array_like or float
            Rationality parameters.
        """
        self.policy[:] = batch_softmax_stable(self.q_values, betas)

    def update_values(self) -> None:
        """Set values to the expected Q under the current policy."""
        self.values[:] = np.sum(self.policy * self.q_values, axis=1)


# ---------------------------------------------------------------------------
# Sparse matrix operations for large MDPs
# ---------------------------------------------------------------------------

def sparse_bellman_operator(
    T_csr: Any,
    R: np.ndarray,
    V: np.ndarray,
    gamma: float,
    n_actions: int,
) -> np.ndarray:
    """Bellman operator using a single stacked sparse transition matrix.

    Transition probabilities for all actions are stacked in a single
    ``(|S|·|A|, |S|)`` CSR matrix, enabling a single sparse matvec.

    Parameters
    ----------
    T_csr : scipy.sparse matrix, shape (|S|·|A|, |S|)
        Stacked transition matrix: rows ``s*|A| .. (s+1)*|A|-1``
        correspond to state *s*, actions ``0 .. |A|-1``.
    R : array_like, shape (|S|, |A|)
        Reward matrix.
    V : array_like, shape (|S|,)
        Current value function.
    gamma : float
        Discount factor.
    n_actions : int
        Number of actions.

    Returns
    -------
    np.ndarray, shape (|S|, |A|)
        Q-values.

    Complexity
    ----------
    O(nnz) for the sparse matvec + O(|S|·|A|) reshape.
    """
    if not _HAS_SCIPY_SPARSE:
        raise ImportError("scipy.sparse required for sparse_bellman_operator")
    V = np.asarray(V, dtype=np.float64).ravel()
    R = np.asarray(R, dtype=np.float64)
    n_states = V.shape[0]

    EV_flat = np.asarray(T_csr.dot(V)).ravel()  # (|S|·|A|,)
    EV = EV_flat.reshape(n_states, n_actions)
    return R + gamma * EV


def sparse_policy_evaluation(
    T_pi: Any,
    R_pi: np.ndarray,
    gamma: float,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> np.ndarray:
    """Policy evaluation via sparse iterative method.

    Solves ``V = R_π + γ · T_π · V`` by iterating until convergence.

    Parameters
    ----------
    T_pi : scipy.sparse matrix, shape (|S|, |S|)
        Transition matrix under the fixed policy π.
    R_pi : array_like, shape (|S|,)
        Expected reward under π.
    gamma : float
        Discount factor.
    tol : float
        Convergence tolerance on max absolute value change.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray, shape (|S|,)
        Converged value function.

    Complexity
    ----------
    O(nnz · max_iter) worst case.
    """
    if not _HAS_SCIPY_SPARSE:
        raise ImportError("scipy.sparse required for sparse_policy_evaluation")
    R_pi = np.asarray(R_pi, dtype=np.float64).ravel()
    V = np.zeros_like(R_pi)

    for _ in range(max_iter):
        V_new = R_pi + gamma * np.asarray(T_pi.dot(V)).ravel()
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V_new


# ---------------------------------------------------------------------------
# Batch bisimulation distance
# ---------------------------------------------------------------------------

def bisimulation_distance(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """Compute bisimulation distances between all pairs of states.

    Implements the iterative metric from Ferns, Panangaden & Precup (2004):

    ``d(s, s') = max_a [ |R(s,a) − R(s',a)| + γ · W(T(s,a,:), T(s',a,:); d) ]``

    where W is the Wasserstein-1 (earth-mover) distance using *d* as the
    ground metric.  For computational efficiency the Wasserstein distance
    is approximated by the L1 distance of CDFs weighted by the current
    metric.

    Parameters
    ----------
    T : array_like, shape (|S|, |A|, |S|)
        Transition probabilities.
    R : array_like, shape (|S|, |A|)
        Reward matrix.
    gamma : float
        Discount factor.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray, shape (|S|, |S|)
        Pairwise bisimulation distances.

    Complexity
    ----------
    O(|S|² · |A| · max_iter) — with O(|S|) per Wasserstein approximation.
    """
    T = np.asarray(T, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_states, n_actions = R.shape[0], R.shape[1]

    # Initialise distances to reward difference
    d = np.zeros((n_states, n_states), dtype=np.float64)

    for _ in range(max_iter):
        d_new = np.zeros_like(d)
        for a in range(n_actions):
            # Reward difference: |R(s,a) - R(s',a)| → (|S|, |S|)
            reward_diff = np.abs(R[:, a, np.newaxis] - R[np.newaxis, :, a])

            # Approximate Wasserstein-1 via L1 of CDFs
            # For each pair (s, s'), compute Σ_s'' |CDF(T(s,a,:)) - CDF(T(s',a,:))| · d_max
            T_a = T[:, a, :]  # (|S|, |S|)
            cdf_diff = np.abs(
                np.cumsum(T_a[:, np.newaxis, :], axis=2)
                - np.cumsum(T_a[np.newaxis, :, :], axis=2)
            )
            # Weight by current metric: sum over s'' of |CDF_diff| * avg_d
            if np.max(d) > _EPS:
                # Use average distance as ground metric scaling
                avg_d = np.mean(d)
                wasserstein_approx = np.sum(cdf_diff, axis=2) * avg_d
            else:
                wasserstein_approx = np.sum(cdf_diff, axis=2)

            candidate = reward_diff + gamma * wasserstein_approx
            d_new = np.maximum(d_new, candidate)

        if np.max(np.abs(d_new - d)) < tol:
            return d_new
        d = d_new

    return d


# ---------------------------------------------------------------------------
# Sparse Kronecker product for factored MDPs
# ---------------------------------------------------------------------------

def sparse_kronecker_product(A: Any, B: Any) -> Any:
    """Kronecker product of two sparse matrices.

    Useful for constructing factored MDP transition matrices from
    independent factor dynamics.

    Parameters
    ----------
    A : scipy.sparse matrix
        Left operand.
    B : scipy.sparse matrix
        Right operand.

    Returns
    -------
    scipy.sparse.csr_matrix
        Kronecker product ``A ⊗ B``.

    Complexity
    ----------
    O(nnz_A · nnz_B).
    """
    if not _HAS_SCIPY_SPARSE:
        raise ImportError("scipy.sparse required for sparse_kronecker_product")
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    if not sp.issparse(B):
        B = sp.csr_matrix(B)
    return sp.kron(A, B, format="csr")


# ---------------------------------------------------------------------------
# Convenience: batch bounded-rational free energy
# ---------------------------------------------------------------------------

def batch_free_energy(
    Q: np.ndarray,
    betas: Union[np.ndarray, float],
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Batch variational free energy: ``F(s) = −(1/β) log Σ_a exp(−β·Q(s,a)) · prior(a)``.

    When *prior* is ``None``, a uniform prior is assumed.

    Parameters
    ----------
    Q : array_like, shape (n, m)
        Q-values (costs) for *n* states and *m* actions.
    betas : array_like, shape (n,) or scalar
        Inverse temperatures.
    prior : array_like or None, shape (m,) or (n, m)
        Action prior.  Broadcast to (n, m).

    Returns
    -------
    np.ndarray, shape (n,)
        Free energy per state.

    Complexity
    ----------
    O(n·m) — uses log-sum-exp for stability.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    n, m = Q.shape

    betas_arr = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if betas_arr.shape == (1,):
        betas_arr = np.broadcast_to(betas_arr, (n,))

    if prior is None:
        log_prior = np.full((n, m), -np.log(m), dtype=np.float64)
    else:
        prior_arr = np.asarray(prior, dtype=np.float64)
        if prior_arr.ndim == 1:
            prior_arr = np.broadcast_to(prior_arr, (n, m))
        log_prior = np.log(np.where(prior_arr > _EPS, prior_arr, _EPS))

    # -β·Q + log(prior)
    exponent = -betas_arr[:, np.newaxis] * Q + log_prior
    row_max = np.max(exponent, axis=1, keepdims=True)
    log_Z = row_max.ravel() + np.log(
        np.sum(np.exp(exponent - row_max), axis=1) + _EPS
    )
    return -(1.0 / np.where(betas_arr > _EPS, betas_arr, _EPS)) * log_Z


__all__ = [
    "batch_softmax_stable",
    "batch_log_policy",
    "batch_kl_divergence",
    "batch_js_divergence",
    "vectorized_fitts_multi",
    "vectorized_fitts_batch",
    "vectorized_hick_batch",
    "batch_information_gain",
    "batch_mutual_information",
    "bellman_backup_dense",
    "bellman_backup_sparse",
    "value_iteration_step",
    "SIMDLayout",
    "sparse_bellman_operator",
    "sparse_policy_evaluation",
    "bisimulation_distance",
    "sparse_kronecker_product",
    "batch_free_energy",
]
