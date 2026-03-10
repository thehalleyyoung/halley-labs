"""
usability_oracle.bisimulation.clustering — Heuristic feature-based clustering.

Provides a fast fallback for state-space reduction when full bisimulation
refinement is too expensive (e.g., very large MDPs).  Uses feature vectors
extracted from states and standard clustering algorithms from scipy.

Clustering methods:
  - Agglomerative (Ward's linkage) — ``scipy.cluster.hierarchy``
  - Spectral — graph Laplacian of the transition matrix

The number of clusters can be automatically selected via the elbow method
or silhouette analysis.

References
----------
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective
  function. *JASA* 58, 236–244.
- Ng, Jordan & Weiss (2001). On spectral clustering. *NIPS*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from usability_oracle.bisimulation.models import Partition
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureBasedClustering
# ---------------------------------------------------------------------------

@dataclass
class FeatureBasedClustering:
    """Fast state-space clustering using extracted feature vectors.

    This is a heuristic fallback for when full bisimulation refinement
    is computationally infeasible.

    Parameters
    ----------
    method : str
        Clustering method: ``"agglomerative"`` (default) or ``"spectral"``.
    linkage_method : str
        Linkage criterion for agglomerative clustering (default ``"ward"``).
    auto_k : bool
        If True, automatically determine the number of clusters.
    max_k : int
        Maximum number of clusters to consider when ``auto_k`` is True.
    """

    method: str = "agglomerative"
    linkage_method: str = "ward"
    auto_k: bool = True
    max_k: int = 50

    # ── Public API --------------------------------------------------------

    def cluster(self, mdp: MDP, n_clusters: Optional[int] = None) -> Partition:
        """Cluster MDP states into a partition.

        Parameters
        ----------
        mdp : MDP
            The MDP whose states should be clustered.
        n_clusters : int or None
            Number of clusters.  If None and ``auto_k`` is True, the number
            is determined automatically.

        Returns
        -------
        Partition
            A partition of the state space based on feature similarity.
        """
        features, state_ids = self._extract_features(mdp)

        if features.shape[0] <= 1:
            return Partition.trivial(state_ids)

        if n_clusters is None:
            if self.auto_k:
                n_clusters = self._determine_k(features, min(self.max_k, len(state_ids)))
            else:
                n_clusters = min(10, len(state_ids))

        n_clusters = max(1, min(n_clusters, len(state_ids)))

        logger.info(
            "Clustering %d states into %d clusters using %s",
            len(state_ids), n_clusters, self.method,
        )

        if self.method == "spectral":
            partition = self._spectral(features, state_ids, n_clusters, mdp)
        else:
            partition = self._agglomerative(features, state_ids, n_clusters)

        quality = self._validate_clusters(partition, mdp)
        logger.info("Clustering quality score: %.4f", quality)

        return partition

    # ── Feature extraction ------------------------------------------------

    def _extract_features(self, mdp: MDP) -> tuple[np.ndarray, list[str]]:
        """Extract a feature matrix from the MDP states.

        Features include:
          - All numeric state features
          - Number of available actions
          - Is-goal / is-terminal flags
          - Out-degree (number of successor states)
          - Average transition cost per action
          - Average transition entropy per action

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        tuple[np.ndarray, list[str]]
            (n_states × n_features) matrix and ordered state ids.
        """
        state_ids = sorted(mdp.states.keys())
        if not state_ids:
            return np.empty((0, 0)), []

        # Collect all feature keys across states
        all_feature_keys: set[str] = set()
        for sid in state_ids:
            state = mdp.states[sid]
            all_feature_keys.update(state.features.keys())
        feature_keys = sorted(all_feature_keys)

        n_states = len(state_ids)
        # Structural features: n_actions, is_goal, is_terminal, out_degree,
        # avg_cost, transition_entropy
        n_structural = 6
        n_features = len(feature_keys) + n_structural
        features = np.zeros((n_states, n_features), dtype=np.float64)

        for i, sid in enumerate(state_ids):
            state = mdp.states[sid]

            # State features
            for j, key in enumerate(feature_keys):
                features[i, j] = state.features.get(key, 0.0)

            # Structural features
            offset = len(feature_keys)
            actions = mdp.get_actions(sid)
            features[i, offset] = len(actions)
            features[i, offset + 1] = float(state.is_goal)
            features[i, offset + 2] = float(state.is_terminal)
            features[i, offset + 3] = len(mdp.get_successors(sid))

            # Average cost across actions
            total_cost = 0.0
            n_trans = 0
            for aid in actions:
                for _, prob, cost in mdp.get_transitions(sid, aid):
                    total_cost += prob * cost
                    n_trans += 1
            features[i, offset + 4] = total_cost / max(n_trans, 1)

            # Transition entropy
            entropy = 0.0
            for aid in actions:
                trans = mdp.get_transitions(sid, aid)
                for _, prob, _ in trans:
                    if prob > 0:
                        entropy -= prob * np.log(prob + 1e-15)
            features[i, offset + 5] = entropy / max(len(actions), 1)

        # Normalise columns to zero mean and unit variance
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds[stds < 1e-10] = 1.0
        features = (features - means) / stds

        return features, state_ids

    # ── Agglomerative clustering ------------------------------------------

    def _agglomerative(
        self,
        features: np.ndarray,
        state_ids: list[str],
        n_clusters: int,
    ) -> Partition:
        """Agglomerative (hierarchical) clustering with Ward's linkage.

        Parameters
        ----------
        features : np.ndarray
            (n_states, n_features) normalised feature matrix.
        state_ids : list[str]
        n_clusters : int

        Returns
        -------
        Partition
        """
        if features.shape[0] <= n_clusters:
            return Partition.discrete(state_ids)

        # Compute pairwise distances
        dists = pdist(features, metric="euclidean")

        # Hierarchical clustering
        Z = linkage(dists, method=self.linkage_method)

        # Cut the dendrogram
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        return self._labels_to_partition(labels, state_ids)

    # ── Spectral clustering -----------------------------------------------

    def _spectral(
        self,
        features: np.ndarray,
        state_ids: list[str],
        n_clusters: int,
        mdp: MDP,
    ) -> Partition:
        """Spectral clustering on the MDP transition graph.

        Constructs an affinity matrix from the transition probabilities,
        computes the graph Laplacian, and clusters in the space of the
        bottom-k eigenvectors.

        Parameters
        ----------
        features : np.ndarray
        state_ids : list[str]
        n_clusters : int
        mdp : MDP

        Returns
        -------
        Partition
        """
        n = len(state_ids)
        if n <= n_clusters:
            return Partition.discrete(state_ids)

        state_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Build adjacency / affinity matrix from transitions
        W = np.zeros((n, n), dtype=np.float64)
        for sid in state_ids:
            i = state_idx[sid]
            for aid in mdp.get_actions(sid):
                for target, prob, _ in mdp.get_transitions(sid, aid):
                    j = state_idx.get(target)
                    if j is not None:
                        W[i, j] = max(W[i, j], prob)

        # Symmetrise: W = (W + W^T) / 2
        W = (W + W.T) / 2.0

        # Degree matrix
        D = np.diag(np.sum(W, axis=1))

        # Normalised graph Laplacian: L = I - D^{-1/2} W D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
        L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Compute bottom-k eigenvectors (smallest eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        k = min(n_clusters, n)
        V = eigenvectors[:, :k]

        # Normalise rows
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        row_norms[row_norms < 1e-10] = 1.0
        V = V / row_norms

        # K-means on the spectral embedding (manual implementation)
        labels = self._simple_kmeans(V, n_clusters)

        return self._labels_to_partition(labels, state_ids)

    # ── Automatic k selection ---------------------------------------------

    def _determine_k(self, features: np.ndarray, max_k: int) -> int:
        """Determine the optimal number of clusters using the elbow method.

        Computes within-cluster sum of squares (WCSS) for k = 1..max_k and
        selects the k with the largest "elbow" (second derivative).

        Falls back to silhouette analysis if the elbow is ambiguous.

        Parameters
        ----------
        features : np.ndarray
        max_k : int

        Returns
        -------
        int
            Recommended number of clusters.
        """
        n = features.shape[0]
        max_k = min(max_k, n)
        if max_k <= 2:
            return max(1, max_k)

        dists = pdist(features, metric="euclidean")

        wcss_values: list[float] = []
        for k in range(1, max_k + 1):
            Z = linkage(dists, method=self.linkage_method)
            labels = fcluster(Z, t=k, criterion="maxclust")
            wcss = self._compute_wcss(features, labels)
            wcss_values.append(wcss)

        # Elbow detection via second derivative
        if len(wcss_values) < 3:
            return 2

        wcss_arr = np.array(wcss_values)
        second_derivative = np.diff(wcss_arr, n=2)
        if len(second_derivative) == 0:
            return 2

        # The elbow is where the second derivative is largest
        elbow_idx = int(np.argmax(second_derivative)) + 2  # +2 for the offset

        # Silhouette validation
        best_k = max(2, min(elbow_idx, max_k))

        logger.debug("Elbow method suggests k=%d", best_k)
        return best_k

    def _compute_wcss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute within-cluster sum of squares."""
        wcss = 0.0
        for label in np.unique(labels):
            mask = labels == label
            cluster_points = features[mask]
            centroid = np.mean(cluster_points, axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
        return float(wcss)

    # ── Cluster validation ------------------------------------------------

    def _validate_clusters(self, partition: Partition, mdp: MDP) -> float:
        """Compute an abstraction quality metric for the clustering.

        The quality is measured as 1 - (average inter-block transition
        probability), i.e. how well the partition respects the transition
        structure.  A score of 1.0 means all transitions stay within blocks.

        Parameters
        ----------
        partition : Partition
        mdp : MDP

        Returns
        -------
        float
            Quality score ∈ [0, 1], higher is better.
        """
        total_transitions = 0
        within_block = 0

        for sid in mdp.states:
            block_idx = partition.state_to_block.get(sid)
            if block_idx is None:
                continue
            for aid in mdp.get_actions(sid):
                for target, prob, _ in mdp.get_transitions(sid, aid):
                    target_block = partition.state_to_block.get(target)
                    total_transitions += 1
                    if target_block == block_idx:
                        within_block += 1

        if total_transitions == 0:
            return 1.0
        return within_block / total_transitions

    # ── Utility -----------------------------------------------------------

    def _labels_to_partition(
        self,
        labels: np.ndarray,
        state_ids: list[str],
    ) -> Partition:
        """Convert cluster labels to a Partition."""
        groups: dict[int, set[str]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(int(label), set()).add(state_ids[i])
        blocks = [frozenset(g) for g in groups.values() if g]
        return Partition.from_blocks(blocks)

    def _simple_kmeans(
        self,
        X: np.ndarray,
        k: int,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Simple k-means clustering (Lloyd's algorithm).

        Parameters
        ----------
        X : np.ndarray
            (n, d) data matrix.
        k : int
            Number of clusters.
        max_iter : int

        Returns
        -------
        np.ndarray
            Cluster labels (n,).
        """
        n, d = X.shape
        k = min(k, n)

        # Initialise centroids with k-means++
        rng = np.random.default_rng(42)
        centroids = np.zeros((k, d), dtype=np.float64)
        centroids[0] = X[rng.integers(n)]

        for c in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / (np.sum(dists) + 1e-15)
            centroids[c] = X[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # Assignment
            dists = np.sum(
                (X[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(dists, axis=1).astype(np.int32)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(X[mask], axis=0)

        return labels
