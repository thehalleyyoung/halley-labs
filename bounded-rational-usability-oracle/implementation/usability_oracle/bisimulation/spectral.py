"""
usability_oracle.bisimulation.spectral — Spectral bisimulation methods.

Uses eigenvalue decomposition and spectral graph theory for approximate
bisimulation analysis.  States are embedded in the spectral space of the
transition graph Laplacian, and clustering in this space yields partitions
that respect the transition structure.

Key capabilities:
  - Eigenvalue-based state equivalence
  - Spectral clustering for approximate bisimulation
  - Graph Laplacian construction from MDP transitions
  - Fiedler value / vector for graph bi-partitioning
  - Spectral gap analysis for mixing-time estimation
  - Low-rank approximation of the transition matrix
  - Dimensionality reduction for large MDPs

References
----------
- Chung, F. R. K. (1997). *Spectral Graph Theory*.
- Ng, A., Jordan, M. & Weiss, Y. (2001). On spectral clustering:
  analysis and an algorithm. *NIPS*.
- Duan, J. et al. (2019). State aggregation learning from Markov
  transition data. *NeurIPS*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from usability_oracle.bisimulation.models import (
    CognitiveDistanceMatrix,
    Partition,
)
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transition matrix utilities
# ---------------------------------------------------------------------------

def build_transition_matrix(mdp: MDP) -> tuple[np.ndarray, list[str]]:
    """Build a dense state-to-state transition matrix.

    Aggregates over actions using uniform weighting:

        P(i, j) = (1 / |A(sᵢ)|) Σ_a T(sⱼ | sᵢ, a)

    Parameters
    ----------
    mdp : MDP

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (P, state_ids) where P is (n, n) row-stochastic.
    """
    state_ids = sorted(mdp.states.keys())
    n = len(state_ids)
    state_idx = {sid: i for i, sid in enumerate(state_ids)}

    P = np.zeros((n, n), dtype=np.float64)

    for sid in state_ids:
        i = state_idx[sid]
        actions = mdp.get_actions(sid)
        if not actions:
            P[i, i] = 1.0  # absorbing
            continue
        weight = 1.0 / len(actions)
        for aid in actions:
            for target, prob, _ in mdp.get_transitions(sid, aid):
                j = state_idx.get(target)
                if j is not None:
                    P[i, j] += weight * prob

    # Ensure row-stochastic
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums

    return P, state_ids


def build_adjacency_matrix(mdp: MDP) -> tuple[np.ndarray, list[str]]:
    """Build a symmetric adjacency/affinity matrix from the MDP.

    W(i, j) = max_{a} T(sⱼ | sᵢ, a)  symmetrised as (W + Wᵀ) / 2.

    Parameters
    ----------
    mdp : MDP

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (W, state_ids).
    """
    state_ids = sorted(mdp.states.keys())
    n = len(state_ids)
    state_idx = {sid: i for i, sid in enumerate(state_ids)}

    W = np.zeros((n, n), dtype=np.float64)

    for sid in state_ids:
        i = state_idx[sid]
        for aid in mdp.get_actions(sid):
            for target, prob, _ in mdp.get_transitions(sid, aid):
                j = state_idx.get(target)
                if j is not None:
                    W[i, j] = max(W[i, j], prob)

    W = (W + W.T) / 2.0
    return W, state_ids


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

@dataclass
class GraphLaplacian:
    """Laplacian of the MDP transition graph.

    Supports unnormalised, random-walk, and symmetric normalised variants.

    Attributes
    ----------
    laplacian : np.ndarray
        The Laplacian matrix (n, n).
    eigenvalues : np.ndarray
        Sorted eigenvalues.
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns).
    state_ids : list[str]
        State identifiers.
    variant : str
        Laplacian variant used.
    """

    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    state_ids: list[str]
    variant: str = "symmetric"

    @classmethod
    def from_mdp(
        cls,
        mdp: MDP,
        variant: str = "symmetric",
        n_eigenvalues: Optional[int] = None,
    ) -> "GraphLaplacian":
        """Construct the graph Laplacian from an MDP.

        Parameters
        ----------
        mdp : MDP
        variant : str
            ``"unnormalised"``, ``"random_walk"``, or ``"symmetric"``
            (default).
        n_eigenvalues : int or None
            Number of smallest eigenvalues to compute. If None, compute all.

        Returns
        -------
        GraphLaplacian
        """
        W, state_ids = build_adjacency_matrix(mdp)
        n = W.shape[0]

        D = np.diag(W.sum(axis=1))
        D_diag = np.diag(D)

        if variant == "unnormalised":
            L = D - W
        elif variant == "random_walk":
            D_inv = np.diag(1.0 / np.maximum(D_diag, 1e-10))
            L = np.eye(n) - D_inv @ W
        else:  # symmetric
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_diag, 1e-10)))
            L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Ensure symmetry for eigendecomposition
        L = (L + L.T) / 2.0

        if n_eigenvalues is not None and n_eigenvalues < n:
            try:
                eigenvalues, eigenvectors = eigsh(
                    L, k=n_eigenvalues, which="SM",
                )
                order = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
            except Exception:
                eigenvalues, eigenvectors = np.linalg.eigh(L)
                eigenvalues = eigenvalues[:n_eigenvalues]
                eigenvectors = eigenvectors[:, :n_eigenvalues]
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(L)

        return cls(
            laplacian=L,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            state_ids=state_ids,
            variant=variant,
        )

    @property
    def fiedler_value(self) -> float:
        """The algebraic connectivity (second smallest eigenvalue).

        A larger Fiedler value indicates a better-connected graph
        (harder to partition).
        """
        if len(self.eigenvalues) < 2:
            return 0.0
        return float(self.eigenvalues[1])

    @property
    def fiedler_vector(self) -> np.ndarray:
        """The eigenvector corresponding to the Fiedler value.

        Sign structure gives a natural bi-partition of the graph.
        """
        if self.eigenvectors.shape[1] < 2:
            return np.zeros(len(self.state_ids))
        return self.eigenvectors[:, 1]

    @property
    def spectral_gap(self) -> float:
        """Gap between the two smallest non-zero eigenvalues.

        A large spectral gap indicates that the graph has a clear
        cluster structure and that random walks mix quickly within
        clusters but slowly between them.
        """
        nonzero = self.eigenvalues[self.eigenvalues > 1e-10]
        if len(nonzero) < 2:
            return 0.0
        return float(nonzero[1] - nonzero[0])

    def mixing_time_bound(self) -> float:
        """Upper bound on random walk mixing time: O(1 / λ₂).

        Returns
        -------
        float
            Estimated mixing time in steps.
        """
        lam2 = self.fiedler_value
        if lam2 < 1e-10:
            return float("inf")
        return 1.0 / lam2


# ---------------------------------------------------------------------------
# Spectral bisimulation clustering
# ---------------------------------------------------------------------------

@dataclass
class SpectralBisimulation:
    """Spectral clustering for approximate bisimulation.

    Embeds states in the space of the bottom-k Laplacian eigenvectors and
    clusters using k-means to produce an approximate bisimulation partition.

    Parameters
    ----------
    n_clusters : int or None
        Number of clusters. If None, determined from the spectral gap.
    laplacian_variant : str
        Laplacian variant (default ``"symmetric"``).
    max_clusters : int
        Maximum number of clusters when auto-detecting.
    """

    n_clusters: Optional[int] = None
    laplacian_variant: str = "symmetric"
    max_clusters: int = 50

    def compute(self, mdp: MDP) -> Partition:
        """Compute the spectral bisimulation partition.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        Partition
        """
        n = len(mdp.states)
        if n <= 1:
            return Partition.trivial(sorted(mdp.states.keys()))

        k = self.n_clusters
        if k is None:
            k = self._auto_k(mdp)
        k = max(2, min(k, n))

        lap = GraphLaplacian.from_mdp(
            mdp, variant=self.laplacian_variant, n_eigenvalues=k,
        )

        V = lap.eigenvectors[:, :k]

        # Normalise rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        V = V / norms

        labels = _simple_kmeans(V, k)
        partition = _labels_to_partition(labels, lap.state_ids)

        logger.info(
            "Spectral bisimulation: %d states → %d clusters "
            "(Fiedler=%.4f, gap=%.4f)",
            n, partition.n_blocks, lap.fiedler_value, lap.spectral_gap,
        )
        return partition

    def compute_embedding(
        self, mdp: MDP, n_dims: int = 10,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute a low-dimensional spectral embedding of the MDP states.

        Parameters
        ----------
        mdp : MDP
        n_dims : int
            Number of dimensions (eigenvectors) to use.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            (embedding, state_ids) where embedding is (n_states, n_dims).
        """
        n = len(mdp.states)
        n_dims = min(n_dims, n)

        lap = GraphLaplacian.from_mdp(
            mdp, variant=self.laplacian_variant, n_eigenvalues=n_dims,
        )
        return lap.eigenvectors[:, :n_dims], lap.state_ids

    def spectral_distance_matrix(self, mdp: MDP) -> CognitiveDistanceMatrix:
        """Compute pairwise distances in spectral embedding space.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        CognitiveDistanceMatrix
        """
        n = len(mdp.states)
        k = self.n_clusters or min(10, n)
        k = min(k, n)

        embedding, state_ids = self.compute_embedding(mdp, n_dims=k)

        # Pairwise Euclidean distances
        diffs = embedding[:, None, :] - embedding[None, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))

        # Normalise to [0, 1]
        diam = distances.max()
        if diam > 0:
            distances /= diam

        return CognitiveDistanceMatrix(distances=distances, state_ids=state_ids)

    def _auto_k(self, mdp: MDP) -> int:
        """Auto-detect number of clusters from the spectral gap."""
        n = len(mdp.states)
        max_k = min(self.max_clusters, n)

        lap = GraphLaplacian.from_mdp(
            mdp, variant=self.laplacian_variant,
            n_eigenvalues=min(max_k + 1, n),
        )

        eigenvalues = lap.eigenvalues
        if len(eigenvalues) < 3:
            return 2

        # Find the largest gap in eigenvalues
        gaps = np.diff(eigenvalues)
        # Skip the first gap (trivial zero eigenvalue)
        if len(gaps) > 1:
            best_k = int(np.argmax(gaps[1:])) + 2
        else:
            best_k = 2

        return max(2, min(best_k, max_k))


# ---------------------------------------------------------------------------
# Low-rank transition approximation
# ---------------------------------------------------------------------------

@dataclass
class LowRankTransitionApproximation:
    """Low-rank approximation of the MDP transition matrix.

    Uses SVD to find a rank-r approximation of the transition matrix,
    enabling dimensionality reduction for large MDPs.

    Parameters
    ----------
    rank : int
        Target rank for the approximation.
    """

    rank: int = 10

    def compute(self, mdp: MDP) -> tuple[np.ndarray, list[str], float]:
        """Compute a low-rank approximation of the transition matrix.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        tuple[np.ndarray, list[str], float]
            (P_approx, state_ids, reconstruction_error) where P_approx is
            the rank-r approximation and reconstruction_error is the
            Frobenius norm of the residual.
        """
        P, state_ids = build_transition_matrix(mdp)
        n = P.shape[0]
        r = min(self.rank, n)

        U, S, Vt = np.linalg.svd(P, full_matrices=False)

        # Truncate to rank r
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        P_approx = U_r @ np.diag(S_r) @ Vt_r

        # Clamp to valid probabilities and re-normalise
        P_approx = np.maximum(P_approx, 0.0)
        row_sums = P_approx.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_approx /= row_sums

        # Reconstruction error
        error = float(np.linalg.norm(P - P_approx, "fro"))

        logger.info(
            "Low-rank approximation: rank %d/%d, Frobenius error=%.4f, "
            "energy retained=%.2f%%",
            r, n, error,
            100.0 * np.sum(S[:r] ** 2) / max(np.sum(S ** 2), 1e-10),
        )

        return P_approx, state_ids, error

    def singular_value_spectrum(self, mdp: MDP) -> np.ndarray:
        """Return the singular values of the transition matrix.

        Useful for choosing the rank parameter.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        np.ndarray
            Sorted singular values (descending).
        """
        P, _ = build_transition_matrix(mdp)
        return np.linalg.svd(P, compute_uv=False)


# ---------------------------------------------------------------------------
# Fiedler bisection
# ---------------------------------------------------------------------------

def fiedler_partition(mdp: MDP) -> Partition:
    """Partition the MDP state space using the Fiedler vector.

    The sign structure of the Fiedler vector (eigenvector of the second
    smallest eigenvalue of the graph Laplacian) gives a natural
    bi-partition that minimises the graph cut.

    Parameters
    ----------
    mdp : MDP

    Returns
    -------
    Partition
        A two-block partition.
    """
    state_ids = sorted(mdp.states.keys())
    if len(state_ids) <= 1:
        return Partition.trivial(state_ids)

    lap = GraphLaplacian.from_mdp(mdp, variant="symmetric", n_eigenvalues=2)
    fv = lap.fiedler_vector

    block_pos = frozenset(
        state_ids[i] for i in range(len(state_ids)) if fv[i] >= 0
    )
    block_neg = frozenset(
        state_ids[i] for i in range(len(state_ids)) if fv[i] < 0
    )

    blocks = [b for b in [block_pos, block_neg] if b]
    if len(blocks) < 2:
        return Partition.trivial(state_ids)

    return Partition.from_blocks(blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """K-means clustering with k-means++ initialisation."""
    n, d = X.shape
    k = min(k, n)
    rng = np.random.default_rng(seed)

    centroids = np.zeros((k, d), dtype=np.float64)
    centroids[0] = X[rng.integers(n)]

    for c in range(1, k):
        dists = np.min(
            np.sum((X[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-15)
        centroids[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2, axis=2,
        )
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)

    return labels


def _labels_to_partition(
    labels: np.ndarray, state_ids: list[str],
) -> Partition:
    """Convert cluster labels to a Partition."""
    groups: dict[int, set[str]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), set()).add(state_ids[i])
    blocks = [frozenset(g) for g in groups.values() if g]
    return Partition.from_blocks(blocks)
