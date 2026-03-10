"""
usability_oracle.utils.sparse — Sparse data structures for large MDPs.

Provides memory-efficient representations for MDP transition matrices,
cost vectors, policies, and belief states when the state-action space
is large but sparse (many zero-probability transitions).

Key components
--------------
- :class:`SparseMDP` — sparse transition and reward representation.
- :class:`SparseCostVector` — compressed cost vector with default value.
- :class:`SparsePolicy` — compressed policy representation.
- :func:`sparse_csr_matvec` — CSR matrix-vector product.
- :func:`sparse_value_iteration` — value iteration on sparse MDPs.
- :func:`sparse_kronecker` — Kronecker product for factored MDPs.
- :class:`CompressedBeliefState` — memory-efficient belief representation.

Performance characteristics
---------------------------
- ``sparse_csr_matvec``: O(nnz) via scipy.sparse.
- ``sparse_value_iteration``: O(nnz · |A| · max_iter).
- ``sparse_kronecker``: O(nnz_A · nnz_B).

References
----------
Boutilier, C., Dearden, R. & Goldszmidt, M. (2000). Stochastic dynamic
    programming with factored representations. *Artificial Intelligence*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Sparse MDP transition matrix
# ---------------------------------------------------------------------------


@dataclass
class SparseMDP:
    """Sparse MDP representation.

    Stores transitions per action as scipy CSR matrices and rewards as
    a dense ``(|S|, |A|)`` array (rewards are typically dense).

    Attributes
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    transitions : list of scipy.sparse.csr_matrix
        ``transitions[a]`` is a ``(|S|, |S|)`` CSR matrix where entry
        ``(s, s')`` is ``T(s, a, s')``.
    rewards : np.ndarray, shape (|S|, |A|)
        Immediate reward ``R(s, a)``.
    gamma : float
        Discount factor.
    """

    n_states: int
    n_actions: int
    transitions: List[Any] = field(default_factory=list)
    rewards: np.ndarray = field(default_factory=lambda: np.empty(0))
    gamma: float = 0.99

    @staticmethod
    def from_dense(
        T: np.ndarray,
        R: np.ndarray,
        gamma: float = 0.99,
    ) -> "SparseMDP":
        """Construct from dense arrays, sparsifying transitions.

        Parameters
        ----------
        T : array_like, shape (|S|, |A|, |S|)
            Dense transition tensor.
        R : array_like, shape (|S|, |A|)
            Reward matrix.
        gamma : float
            Discount factor.

        Returns
        -------
        SparseMDP
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy required for SparseMDP")
        T = np.asarray(T, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        n_states, n_actions = R.shape
        transitions = [sp.csr_matrix(T[:, a, :]) for a in range(n_actions)]
        return SparseMDP(
            n_states=n_states,
            n_actions=n_actions,
            transitions=transitions,
            rewards=R,
            gamma=gamma,
        )

    @staticmethod
    def from_transitions(
        n_states: int,
        n_actions: int,
        transition_tuples: Sequence[Tuple[int, int, int, float]],
        reward_tuples: Sequence[Tuple[int, int, float]],
        gamma: float = 0.99,
    ) -> "SparseMDP":
        """Construct from sparse tuples.

        Parameters
        ----------
        n_states : int
            Number of states.
        n_actions : int
            Number of actions.
        transition_tuples : sequence of (s, a, s', prob)
            Sparse transition entries.
        reward_tuples : sequence of (s, a, reward)
            Reward entries.
        gamma : float
            Discount factor.

        Returns
        -------
        SparseMDP
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy required for SparseMDP")

        # Build per-action COO data
        rows: Dict[int, List[int]] = {a: [] for a in range(n_actions)}
        cols: Dict[int, List[int]] = {a: [] for a in range(n_actions)}
        vals: Dict[int, List[float]] = {a: [] for a in range(n_actions)}

        for s, a, sp_idx, prob in transition_tuples:
            rows[a].append(s)
            cols[a].append(sp_idx)
            vals[a].append(prob)

        transitions = []
        for a in range(n_actions):
            mat = sp.coo_matrix(
                (vals[a], (rows[a], cols[a])),
                shape=(n_states, n_states),
                dtype=np.float64,
            ).tocsr()
            transitions.append(mat)

        R = np.zeros((n_states, n_actions), dtype=np.float64)
        for s, a, r in reward_tuples:
            R[s, a] = r

        return SparseMDP(
            n_states=n_states,
            n_actions=n_actions,
            transitions=transitions,
            rewards=R,
            gamma=gamma,
        )

    @property
    def total_nnz(self) -> int:
        """Total non-zero entries across all action transition matrices."""
        return sum(T_a.nnz for T_a in self.transitions)

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in the transition tensor."""
        total = self.n_states * self.n_actions * self.n_states
        return 1.0 - (self.total_nnz / total) if total > 0 else 1.0

    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        mem = self.rewards.nbytes
        for T_a in self.transitions:
            mem += T_a.data.nbytes + T_a.indices.nbytes + T_a.indptr.nbytes
        return mem


# ---------------------------------------------------------------------------
# Sparse cost vector
# ---------------------------------------------------------------------------


@dataclass
class SparseCostVector:
    """Compressed cost vector with a default value for unset entries.

    Useful when most states share a common cost.

    Attributes
    ----------
    size : int
        Logical length of the vector.
    default : float
        Default value for entries not explicitly set.
    entries : dict[int, float]
        Explicitly stored entries.
    """

    size: int
    default: float = 0.0
    entries: Dict[int, float] = field(default_factory=dict)

    def __getitem__(self, idx: int) -> float:
        return self.entries.get(idx, self.default)

    def __setitem__(self, idx: int, val: float) -> None:
        if val == self.default:
            self.entries.pop(idx, None)
        else:
            self.entries[idx] = val

    def to_dense(self) -> np.ndarray:
        """Convert to a dense numpy array."""
        arr = np.full(self.size, self.default, dtype=np.float64)
        for idx, val in self.entries.items():
            arr[idx] = val
        return arr

    @staticmethod
    def from_dense(arr: np.ndarray, default: float = 0.0) -> "SparseCostVector":
        """Construct from a dense array, storing only non-default entries."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        entries = {int(i): float(arr[i]) for i in range(len(arr)) if arr[i] != default}
        return SparseCostVector(size=len(arr), default=default, entries=entries)

    @property
    def nnz(self) -> int:
        """Number of explicitly stored entries."""
        return len(self.entries)

    def dot(self, other: np.ndarray) -> float:
        """Dot product with a dense vector."""
        other = np.asarray(other, dtype=np.float64).ravel()
        result = self.default * np.sum(other)
        for idx, val in self.entries.items():
            result += (val - self.default) * other[idx]
        return float(result)


# ---------------------------------------------------------------------------
# Sparse policy representation
# ---------------------------------------------------------------------------


@dataclass
class SparsePolicy:
    """Compressed policy representation.

    Stores the policy as a deterministic action per state (greedy) or as
    a sparse probability table for stochastic policies.

    Attributes
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    deterministic : np.ndarray or None, shape (|S|,)
        Deterministic action per state (if not ``None``).
    stochastic : dict[(int, int), float] or None
        Sparse ``(state, action) → probability`` map for
        non-uniform entries.  Entries absent from this map default
        to ``1/n_actions`` if *deterministic* is ``None``.
    """

    n_states: int
    n_actions: int
    deterministic: Optional[np.ndarray] = None
    stochastic: Optional[Dict[Tuple[int, int], float]] = None

    @staticmethod
    def from_greedy(q_values: np.ndarray) -> "SparsePolicy":
        """Construct a deterministic greedy policy from Q-values.

        Parameters
        ----------
        q_values : array_like, shape (|S|, |A|)
            Q-value table.

        Returns
        -------
        SparsePolicy
        """
        Q = np.asarray(q_values, dtype=np.float64)
        return SparsePolicy(
            n_states=Q.shape[0],
            n_actions=Q.shape[1],
            deterministic=np.argmax(Q, axis=1).astype(np.int64),
        )

    @staticmethod
    def from_softmax(
        q_values: np.ndarray,
        beta: float,
        threshold: float = 1e-4,
    ) -> "SparsePolicy":
        """Construct a sparse stochastic policy via softmax.

        Only stores entries whose probability exceeds *threshold*.

        Parameters
        ----------
        q_values : array_like, shape (|S|, |A|)
            Q-value table (costs — lower is better).
        beta : float
            Inverse temperature.
        threshold : float
            Minimum probability to store.

        Returns
        -------
        SparsePolicy
        """
        Q = np.asarray(q_values, dtype=np.float64)
        n_states, n_actions = Q.shape
        scaled = -beta * Q
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        stoch: Dict[Tuple[int, int], float] = {}
        for s in range(n_states):
            for a in range(n_actions):
                if probs[s, a] > threshold:
                    stoch[(s, a)] = float(probs[s, a])

        return SparsePolicy(
            n_states=n_states,
            n_actions=n_actions,
            stochastic=stoch,
        )

    def action_probs(self, state: int) -> np.ndarray:
        """Return the full probability vector for *state*.

        Returns
        -------
        np.ndarray, shape (n_actions,)
        """
        if self.deterministic is not None:
            probs = np.zeros(self.n_actions, dtype=np.float64)
            probs[int(self.deterministic[state])] = 1.0
            return probs
        if self.stochastic is not None:
            probs = np.zeros(self.n_actions, dtype=np.float64)
            total = 0.0
            for a in range(self.n_actions):
                p = self.stochastic.get((state, a), 0.0)
                probs[a] = p
                total += p
            if total > _EPS:
                probs /= total
            else:
                probs[:] = 1.0 / self.n_actions
            return probs
        return np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)

    def to_dense(self) -> np.ndarray:
        """Convert to a dense ``(|S|, |A|)`` policy matrix."""
        result = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        for s in range(self.n_states):
            result[s, :] = self.action_probs(s)
        return result

    @property
    def nnz(self) -> int:
        """Number of explicitly stored probability entries."""
        if self.deterministic is not None:
            return self.n_states
        if self.stochastic is not None:
            return len(self.stochastic)
        return 0


# ---------------------------------------------------------------------------
# CSR operations
# ---------------------------------------------------------------------------


def sparse_csr_matvec(A: Any, x: np.ndarray) -> np.ndarray:
    """Sparse CSR matrix-vector product.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (m, n)
        Sparse matrix in any format (converted to CSR).
    x : array_like, shape (n,)
        Dense vector.

    Returns
    -------
    np.ndarray, shape (m,)
        Result of A · x.

    Complexity
    ----------
    O(nnz).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy.sparse required for sparse_csr_matvec")
    x = np.asarray(x, dtype=np.float64).ravel()
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    return np.asarray(A.dot(x)).ravel()


def sparse_csr_matmat(A: Any, B: Any) -> Any:
    """Sparse CSR matrix-matrix product.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (m, k)
    B : scipy.sparse matrix, shape (k, n)

    Returns
    -------
    scipy.sparse.csr_matrix, shape (m, n)

    Complexity
    ----------
    O(nnz_A · avg_nnz_per_row_B).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy.sparse required for sparse_csr_matmat")
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    if not sp.issparse(B):
        B = sp.csr_matrix(B)
    return (A @ B).tocsr()


# ---------------------------------------------------------------------------
# Sparse value iteration
# ---------------------------------------------------------------------------


def sparse_value_iteration(
    mdp: SparseMDP,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Value iteration on a sparse MDP.

    Parameters
    ----------
    mdp : SparseMDP
        The MDP to solve.
    tol : float
        Convergence tolerance on max |V_new − V|.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    V : np.ndarray, shape (|S|,)
        Converged value function.
    policy : np.ndarray, shape (|S|,)
        Greedy policy (action indices).

    Complexity
    ----------
    O(nnz · |A| · max_iter).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for sparse_value_iteration")

    V = np.zeros(mdp.n_states, dtype=np.float64)
    Q = np.empty((mdp.n_states, mdp.n_actions), dtype=np.float64)

    for _ in range(max_iter):
        for a in range(mdp.n_actions):
            Q[:, a] = mdp.rewards[:, a] + mdp.gamma * np.asarray(
                mdp.transitions[a].dot(V)
            ).ravel()
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    policy = np.argmax(Q, axis=1).astype(np.int64)
    return V, policy


def sparse_policy_transition(
    mdp: SparseMDP,
    policy: np.ndarray,
) -> Any:
    """Construct the transition matrix under a deterministic policy.

    Parameters
    ----------
    mdp : SparseMDP
        The MDP.
    policy : array_like, shape (|S|,)
        Action per state.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (|S|, |S|)
        ``T_π[s, s'] = T(s, π(s), s')``.
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for sparse_policy_transition")
    policy = np.asarray(policy, dtype=np.int64).ravel()
    n = mdp.n_states
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for s in range(n):
        a = int(policy[s])
        T_a = mdp.transitions[a]
        # Extract row s from CSR
        start, end = T_a.indptr[s], T_a.indptr[s + 1]
        for idx in range(start, end):
            sp_idx = T_a.indices[idx]
            prob = T_a.data[idx]
            rows.append(s)
            cols.append(sp_idx)
            vals.append(prob)

    return sp.coo_matrix(
        (vals, (rows, cols)), shape=(n, n), dtype=np.float64
    ).tocsr()


# ---------------------------------------------------------------------------
# Sparse Kronecker product for factored MDPs
# ---------------------------------------------------------------------------


def sparse_kronecker(A: Any, B: Any) -> Any:
    """Kronecker product of two sparse matrices.

    Constructs ``A ⊗ B`` for factored MDP transition dynamics.

    Parameters
    ----------
    A : scipy.sparse matrix
    B : scipy.sparse matrix

    Returns
    -------
    scipy.sparse.csr_matrix

    Complexity
    ----------
    O(nnz_A · nnz_B).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for sparse_kronecker")
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    if not sp.issparse(B):
        B = sp.csr_matrix(B)
    return sp.kron(A, B, format="csr")


def factored_transition(
    factor_transitions: Sequence[Any],
) -> Any:
    """Build a joint transition matrix from independent factor transitions.

    Parameters
    ----------
    factor_transitions : sequence of sparse matrices
        Transition matrices for each independent factor.

    Returns
    -------
    scipy.sparse.csr_matrix
        Joint transition matrix (Kronecker product of all factors).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for factored_transition")
    if len(factor_transitions) == 0:
        raise ValueError("At least one factor transition required")

    result = factor_transitions[0]
    if not sp.issparse(result):
        result = sp.csr_matrix(result)
    for T_f in factor_transitions[1:]:
        result = sp.kron(result, T_f, format="csr")
    return result


# ---------------------------------------------------------------------------
# Memory-efficient belief state
# ---------------------------------------------------------------------------


@dataclass
class CompressedBeliefState:
    """Memory-efficient belief state representation.

    Only stores entries whose probability exceeds *threshold*.

    Attributes
    ----------
    size : int
        Total number of states in the underlying MDP.
    entries : dict[int, float]
        State → probability for states with non-negligible belief.
    threshold : float
        Minimum probability to retain.
    """

    size: int
    entries: Dict[int, float] = field(default_factory=dict)
    threshold: float = 1e-6

    @staticmethod
    def from_dense(
        belief: np.ndarray,
        threshold: float = 1e-6,
    ) -> "CompressedBeliefState":
        """Construct from a dense belief vector.

        Parameters
        ----------
        belief : array_like, shape (|S|,)
            Dense belief distribution.
        threshold : float
            Minimum probability to store.
        """
        b = np.asarray(belief, dtype=np.float64).ravel()
        entries = {
            int(i): float(b[i]) for i in range(len(b)) if b[i] > threshold
        }
        return CompressedBeliefState(
            size=len(b), entries=entries, threshold=threshold
        )

    def to_dense(self) -> np.ndarray:
        """Convert to a normalised dense vector."""
        arr = np.zeros(self.size, dtype=np.float64)
        for idx, prob in self.entries.items():
            arr[idx] = prob
        total = arr.sum()
        if total > _EPS:
            arr /= total
        return arr

    def update(
        self,
        transition_row: Any,
        observation_likelihood: np.ndarray,
    ) -> "CompressedBeliefState":
        """Bayesian belief update: predict + correct.

        Parameters
        ----------
        transition_row : sparse or dense, shape (|S|, |S|) or (|S|,)
            Transition probabilities from current state distribution.
        observation_likelihood : array_like, shape (|S|,)
            ``P(obs | state)`` for the observed observation.

        Returns
        -------
        CompressedBeliefState
            Updated belief.
        """
        b = self.to_dense()

        # Predict step
        if _HAS_SCIPY and sp.issparse(transition_row):
            b_pred = np.asarray(transition_row.T.dot(b)).ravel()
        else:
            T = np.asarray(transition_row, dtype=np.float64)
            if T.ndim == 2:
                b_pred = T.T @ b
            else:
                b_pred = T * b

        # Correct step
        obs = np.asarray(observation_likelihood, dtype=np.float64).ravel()
        b_new = b_pred * obs
        total = b_new.sum()
        if total > _EPS:
            b_new /= total

        return CompressedBeliefState.from_dense(b_new, self.threshold)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief in bits."""
        probs = np.array(list(self.entries.values()), dtype=np.float64)
        total = probs.sum()
        if total <= _EPS:
            return 0.0
        probs = probs / total
        probs = probs[probs > _EPS]
        return float(-np.sum(probs * np.log2(probs)))

    @property
    def nnz(self) -> int:
        return len(self.entries)

    @property
    def compression_ratio(self) -> float:
        """Ratio of stored to total entries (lower = more compressed)."""
        return self.nnz / self.size if self.size > 0 else 0.0


__all__ = [
    "SparseMDP",
    "SparseCostVector",
    "SparsePolicy",
    "sparse_csr_matvec",
    "sparse_csr_matmat",
    "sparse_value_iteration",
    "sparse_policy_transition",
    "sparse_kronecker",
    "factored_transition",
    "CompressedBeliefState",
]
