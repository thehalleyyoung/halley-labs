"""Structural behavioral descriptors for directed acyclic graphs (DAGs).

This module provides :class:`StructuralDescriptor`, a configurable descriptor
computer that extracts up to ten graph-theoretic features from a DAG's
adjacency matrix.  Every feature is normalized to the ``[0, 1]`` interval so
that the resulting behavioral descriptor is directly usable by a MAP-Elites
archive without additional scaling.

Features
--------
The ten supported features are:

1.  **edge_density** – fraction of possible directed edges that are present.
2.  **max_in_degree** – maximum node in-degree as a ratio of ``n - 1``.
3.  **v_structure_count** – number of collider (v-structure) motifs,
    normalized by the combinatorial upper bound.
4.  **longest_path** – length of the longest directed path (via DP on a
    topological ordering), normalized by ``n - 1``.
5.  **avg_path_length** – mean shortest directed-path length over all
    reachable source–target pairs, normalized by ``n - 1``.
6.  **clustering_coefficient** – average local clustering coefficient
    computed on the moralized (undirected) skeleton.
7.  **betweenness_centrality** – maximum node betweenness centrality
    (Brandes' algorithm on the DAG), normalized to ``[0, 1]``.
8.  **connected_components** – number of weakly connected components
    divided by ``n``.
9.  **dag_depth** – longest directed path from any source (zero in-degree
    node) to any sink (zero out-degree node), normalized by ``n - 1``.
    Equivalent to *longest_path* when the graph has well-defined sources
    and sinks.
10. **parent_set_entropy** – Shannon entropy of the parent-set-size
    distribution, normalized by ``log(n)``.

All heavy computation is implemented with :mod:`numpy` vectorization and
:class:`collections.deque`-based BFS to keep runtime manageable for graphs
of moderate size (up to a few hundred nodes).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.descriptors.descriptor_base import DescriptorComputer
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ALL_FEATURES: List[str] = [
    "edge_density",
    "max_in_degree",
    "v_structure_count",
    "longest_path",
    "avg_path_length",
    "clustering_coefficient",
    "betweenness_centrality",
    "connected_components",
    "dag_depth",
    "parent_set_entropy",
]
"""Canonical ordered list of every structural feature name."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class StructuralDescriptor(DescriptorComputer):
    """Configurable structural-feature descriptor for DAGs.

    By default all ten features listed in :data:`ALL_FEATURES` are computed.
    Pass a subset via the *features* argument to select only the features
    you need—this both reduces the descriptor dimensionality and avoids the
    computational cost of unused features.

    Parameters
    ----------
    features : list[str] | None
        Ordered list of feature names to include in the descriptor.  Each
        name must be one of the entries in :data:`ALL_FEATURES`.  When
        ``None`` (the default), all features are used.

    Raises
    ------
    ValueError
        If *features* contains an unrecognized name.

    Examples
    --------
    >>> desc = StructuralDescriptor(features=["edge_density", "dag_depth"])
    >>> desc.descriptor_dim
    2
    >>> dag = np.array([[0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [0, 0, 0]], dtype=np.int8)
    >>> vec = desc.compute(dag)
    >>> vec.shape
    (2,)
    """

    # Mapping from feature name → static computation method.  Populated once
    # at class-definition time via the ``_register`` helper below.
    _FEATURE_REGISTRY: Dict[str, Callable[..., float]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, features: list[str] | None = None) -> None:
        """Initialise the descriptor with the requested feature set.

        Parameters
        ----------
        features : list[str] | None
            Feature names to include.  ``None`` selects all features.

        Raises
        ------
        ValueError
            If any name in *features* is not a recognised feature.
        """
        if features is None:
            self._features: list[str] = list(ALL_FEATURES)
        else:
            unknown = set(features) - set(ALL_FEATURES)
            if unknown:
                raise ValueError(
                    f"Unknown structural feature(s): {sorted(unknown)}. "
                    f"Valid features: {ALL_FEATURES}"
                )
            self._features = list(features)

    # ------------------------------------------------------------------
    # DescriptorComputer interface
    # ------------------------------------------------------------------

    @property
    def descriptor_dim(self) -> int:  # noqa: D401
        """Dimensionality of the descriptor vector.

        Equal to the number of selected features.
        """
        return len(self._features)

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-dimension ``[0, 1]`` bounds.

        All structural features are normalized to the unit interval, so the
        lower bound is a zero-vector and the upper bound is a one-vector.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(low, high)`` where each array has shape ``(descriptor_dim,)``.
        """
        low = np.zeros(self.descriptor_dim, dtype=np.float64)
        high = np.ones(self.descriptor_dim, dtype=np.float64)
        return low, high

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute the behavioral descriptor for *dag*.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the directed acyclic graph,
            where ``dag[i, j] == 1`` indicates the edge ``i → j``.
        data : DataMatrix | None
            Optional observed data matrix (unused by structural features but
            accepted for API compatibility with the base class).

        Returns
        -------
        BehavioralDescriptor
            1-D ``float64`` array of length :pyattr:`descriptor_dim`, with
            each element in ``[0, 1]``.
        """
        n: int = dag.shape[0]
        values: list[float] = []
        for name in self._features:
            fn = self._FEATURE_REGISTRY[name]
            values.append(fn(dag, n))
        return np.asarray(values, dtype=np.float64)

    # ------------------------------------------------------------------
    # Feature 1: Edge density
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_density(dag: AdjacencyMatrix, n: int) -> float:
        """Fraction of possible directed edges that are present.

        For a DAG on *n* nodes the maximum number of directed edges
        (without creating a cycle) is ``n * (n - 1) / 2``.  The density
        is therefore::

            |E| / (n * (n - 1) / 2)

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2.0
        return float(np.sum(dag)) / max_edges

    # ------------------------------------------------------------------
    # Feature 2: Max in-degree ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _max_in_degree_ratio(dag: AdjacencyMatrix, n: int) -> float:
        """Maximum in-degree divided by ``n - 1``.

        The in-degree of node *j* is the number of parents
        ``sum_i dag[i, j]``.  The theoretical maximum for any single node
        in a DAG is ``n - 1`` (every other node is a parent).

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 1:
            return 0.0
        in_degrees: npt.NDArray[np.int64] = dag.sum(axis=0)
        return float(in_degrees.max()) / (n - 1)

    # ------------------------------------------------------------------
    # Feature 3: V-structure count
    # ------------------------------------------------------------------

    @staticmethod
    def _v_structure_count(dag: AdjacencyMatrix, n: int) -> float:
        """Normalized count of v-structures (colliders) in the DAG.

        A *v-structure* (or *collider*) is a triple ``(i, k, j)`` with
        ``i → k ← j`` where ``i`` and ``j`` are **not** adjacent (no edge
        in either direction between them).

        The result is normalized by the combinatorial upper bound
        ``C(n, 2) * (n - 2)``, which equals the maximum number of ordered
        collider triples that could exist in a graph on *n* nodes.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n < 3:
            return 0.0

        # Precompute the adjacency indicator (either direction) for the
        # non-adjacency check.
        adj_any = (dag.astype(bool) | dag.T.astype(bool))

        count = 0
        for k in range(n):
            # Parents of k: nodes i where dag[i, k] == 1
            parents = np.where(dag[:, k])[0]
            num_parents = len(parents)
            if num_parents < 2:
                continue
            # Check every pair of parents for non-adjacency.
            for idx_a in range(num_parents):
                i = parents[idx_a]
                for idx_b in range(idx_a + 1, num_parents):
                    j = parents[idx_b]
                    if not adj_any[i, j]:
                        count += 1

        # Upper bound: C(n, 2) * (n - 2)
        max_v = (n * (n - 1) // 2) * (n - 2)
        if max_v == 0:
            return 0.0
        return min(float(count) / max_v, 1.0)

    # ------------------------------------------------------------------
    # Feature 4: Longest path (DP on topological order)
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_path(dag: AdjacencyMatrix, n: int) -> float:
        """Length of the longest directed path, normalized by ``n - 1``.

        Uses Kahn's algorithm to produce a topological ordering, then
        computes the longest-path length via dynamic programming::

            dist[j] = max(dist[i] + 1)  for all parents i of j

        The value is divided by ``n - 1`` to yield a normalized result in
        ``[0, 1]``.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 1:
            return 0.0

        # --- Kahn's algorithm for topological sort ---
        in_deg = dag.sum(axis=0).astype(int).copy()
        queue: deque[int] = deque()
        for v in range(n):
            if in_deg[v] == 0:
                queue.append(v)

        topo_order: list[int] = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            children = np.where(dag[v] != 0)[0]
            for c in children:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)

        # If the graph has a cycle (shouldn't for a DAG) fall back to
        # partial ordering.
        if len(topo_order) != n:
            # Include remaining nodes in arbitrary order so DP can proceed.
            remaining = set(range(n)) - set(topo_order)
            topo_order.extend(remaining)

        # --- DP for longest path ---
        dist = np.zeros(n, dtype=int)
        for v in topo_order:
            parents = np.where(dag[:, v] != 0)[0]
            if len(parents) > 0:
                dist[v] = int(dist[parents].max()) + 1

        longest = int(dist.max())
        return float(longest) / (n - 1)

    # ------------------------------------------------------------------
    # Feature 5: Average shortest-path length
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_path_length(dag: AdjacencyMatrix, n: int) -> float:
        """Mean shortest directed-path length over all reachable pairs.

        Performs BFS from every node using :class:`collections.deque`.
        Only pairs ``(s, t)`` where ``t`` is reachable from ``s`` (and
        ``s ≠ t``) contribute to the average.  The result is normalized
        by ``n - 1``.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` when no pair is
            reachable (e.g., graph with zero edges or single node).
        """
        if n <= 1:
            return 0.0

        total_length: int = 0
        pair_count: int = 0

        # Pre-compute children lists for faster BFS.
        children: list[npt.NDArray[np.intp]] = [
            np.where(dag[v] != 0)[0] for v in range(n)
        ]

        for source in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            dist[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                d = dist[node]
                for neighbor in children[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        queue.append(neighbor)
            reachable = dist[dist > 0]
            total_length += int(reachable.sum())
            pair_count += len(reachable)

        if pair_count == 0:
            return 0.0
        return (float(total_length) / pair_count) / (n - 1)

    # ------------------------------------------------------------------
    # Feature 6: Clustering coefficient (moralized / undirected skeleton)
    # ------------------------------------------------------------------

    @staticmethod
    def _clustering_coefficient(dag: AdjacencyMatrix, n: int) -> float:
        """Average local clustering coefficient on the moralized skeleton.

        The *moralized* (undirected) skeleton is obtained by taking
        ``skeleton = dag | dag^T`` and treating the result as undirected.
        For each node *v* with degree ``k >= 2``, the local clustering
        coefficient is::

            C_v = 2 * |{edges among neighbours}| / (k * (k - 1))

        Nodes with ``k < 2`` contribute ``0``.  The returned value is the
        arithmetic mean over all nodes.

        We use numpy sub-matrix indexing to count edges among neighbours,
        which is substantially faster than a pure-Python double loop for
        larger graphs.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n < 3:
            return 0.0

        skeleton = (dag.astype(bool) | dag.T.astype(bool))
        coefficients = np.zeros(n, dtype=np.float64)

        for v in range(n):
            neighbors = np.where(skeleton[v])[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count edges among neighbours via the skeleton sub-matrix.
            sub = skeleton[np.ix_(neighbors, neighbors)]
            # Each undirected edge is counted once (upper triangle).
            links = int(np.triu(sub, k=1).sum())
            coefficients[v] = 2.0 * links / (k * (k - 1))

        return float(coefficients.mean())

    # ------------------------------------------------------------------
    # Feature 7: Betweenness centrality (Brandes' algorithm for DAGs)
    # ------------------------------------------------------------------

    @staticmethod
    def _betweenness_centrality(dag: AdjacencyMatrix, n: int) -> float:
        """Maximum node betweenness centrality, normalized to ``[0, 1]``.

        Implements Brandes' algorithm adapted for unweighted directed
        graphs.  For each source *s* a BFS is performed to compute
        shortest-path counts (``sigma``) and distances.  A back-propagation
        pass then accumulates pair dependencies (``delta``) which are added
        to the global betweenness score of each intermediate node.

        The maximum betweenness across all nodes is returned, normalized by
        ``(n - 1) * (n - 2)`` (the maximum possible betweenness for a
        single node in a directed graph on *n* nodes).

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 2:
            return 0.0

        betweenness = np.zeros(n, dtype=np.float64)

        # Pre-compute children lists.
        children: list[npt.NDArray[np.intp]] = [
            np.where(dag[v] != 0)[0] for v in range(n)
        ]

        for s in range(n):
            # --- BFS phase ---
            stack: list[int] = []
            predecessors: list[list[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=np.float64)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            queue: deque[int] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                d_v = dist[v]
                for w in children[v]:
                    # First visit?
                    if dist[w] == -1:
                        dist[w] = d_v + 1
                        queue.append(w)
                    # Shortest-path via v?
                    if dist[w] == d_v + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            # --- Back-propagation phase ---
            delta = np.zeros(n, dtype=np.float64)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # Normalize by (n-1)*(n-2) — the max betweenness for directed graphs.
        norm = (n - 1) * (n - 2)
        if norm == 0:
            return 0.0
        max_bc = float(betweenness.max()) / norm
        return min(max_bc, 1.0)

    # ------------------------------------------------------------------
    # Feature 8: Weakly connected components
    # ------------------------------------------------------------------

    @staticmethod
    def _connected_component_sizes(dag: AdjacencyMatrix, n: int) -> float:
        """Number of weakly connected components divided by ``n``.

        Two nodes are in the same weakly connected component if they are
        connected by a path when edge directions are ignored.  The
        undirected skeleton ``dag | dag^T`` is used for BFS exploration.

        A graph with a single connected component yields ``1/n``; a graph
        with ``n`` isolated nodes yields ``1.0``.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        skeleton = (dag.astype(bool) | dag.T.astype(bool))
        visited = np.zeros(n, dtype=bool)
        num_components = 0

        for start in range(n):
            if visited[start]:
                continue
            num_components += 1
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                v = queue.popleft()
                neighbors = np.where(skeleton[v] & ~visited)[0]
                for nb in neighbors:
                    visited[nb] = True
                    queue.append(nb)

        return float(num_components) / n

    # ------------------------------------------------------------------
    # Feature 9: DAG depth (source → sink longest path)
    # ------------------------------------------------------------------

    @staticmethod
    def _dag_depth(dag: AdjacencyMatrix, n: int) -> float:
        """Longest directed path from any source to any sink, normalized.

        A *source* is a node with in-degree zero; a *sink* is a node with
        out-degree zero.  The DAG depth is the length of the longest
        directed path between any source–sink pair.

        The computation is identical to :meth:`_longest_path`—the longest
        path in a DAG always starts at some source and ends at some sink—
        but is conceptually distinct: it emphasizes the *depth* of the
        causal hierarchy.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 1:
            return 0.0

        # --- Kahn's algorithm for topological sort ---
        in_deg = dag.sum(axis=0).astype(int).copy()
        queue: deque[int] = deque()
        for v in range(n):
            if in_deg[v] == 0:
                queue.append(v)

        topo_order: list[int] = []
        in_deg_work = in_deg.copy()
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            children = np.where(dag[v] != 0)[0]
            for c in children:
                in_deg_work[c] -= 1
                if in_deg_work[c] == 0:
                    queue.append(c)

        if len(topo_order) != n:
            remaining = set(range(n)) - set(topo_order)
            topo_order.extend(remaining)

        # --- DP for longest path from sources ---
        dist = np.zeros(n, dtype=int)
        for v in topo_order:
            parents = np.where(dag[:, v] != 0)[0]
            if len(parents) > 0:
                dist[v] = int(dist[parents].max()) + 1

        # The depth is the maximum distance attained at any sink.
        out_deg = dag.sum(axis=1).astype(int)
        sinks = np.where(out_deg == 0)[0]
        if len(sinks) == 0:
            # No sinks (shouldn't happen in a DAG); use global max.
            depth = int(dist.max())
        else:
            depth = int(dist[sinks].max())

        return float(depth) / (n - 1)

    # ------------------------------------------------------------------
    # Feature 10: Parent-set entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _parent_set_entropy(dag: AdjacencyMatrix, n: int) -> float:
        """Shannon entropy of the parent-set-size distribution, normalized.

        The parent-set size of node *j* is ``sum_i dag[i, j]``.  We form
        the empirical distribution over these sizes and compute the Shannon
        entropy (base-*e*).  The result is normalized by ``log(n)`` (the
        maximum entropy for a discrete distribution on *n* outcomes).

        A DAG where every node has the same number of parents has zero
        entropy; a DAG with a diverse mix of parent-set sizes yields higher
        entropy.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        if n <= 1:
            return 0.0

        in_degrees = dag.sum(axis=0).astype(int)

        # Build empirical probability distribution over parent-set sizes.
        unique_sizes, counts = np.unique(in_degrees, return_counts=True)
        probs = counts.astype(np.float64) / n

        # Shannon entropy (natural log).
        entropy = -float(np.sum(probs * np.log(probs + 1e-300)))

        log_n = math.log(n)
        if log_n == 0.0:
            return 0.0
        result = entropy / log_n
        # Avoid returning -0.0 when entropy is exactly zero.
        if result == 0.0:
            return 0.0
        return min(result, 1.0)

    # ------------------------------------------------------------------
    # Feature registration
    # ------------------------------------------------------------------


# Populate the feature registry after all static methods are defined.
StructuralDescriptor._FEATURE_REGISTRY = {
    "edge_density": StructuralDescriptor._edge_density,
    "max_in_degree": StructuralDescriptor._max_in_degree_ratio,
    "v_structure_count": StructuralDescriptor._v_structure_count,
    "longest_path": StructuralDescriptor._longest_path,
    "avg_path_length": StructuralDescriptor._avg_path_length,
    "clustering_coefficient": StructuralDescriptor._clustering_coefficient,
    "betweenness_centrality": StructuralDescriptor._betweenness_centrality,
    "connected_components": StructuralDescriptor._connected_component_sizes,
    "dag_depth": StructuralDescriptor._dag_depth,
    "parent_set_entropy": StructuralDescriptor._parent_set_entropy,
}
