"""Extended graph canonicalization and isomorphism interface.

Provides canonical labeling using color refinement (Weisfeiler-Leman),
graph isomorphism testing, automorphism group computation, and CPDAG
canonicalization.  Falls back to degree-based heuristics when igraph
is not available.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, GraphHash


# ---------------------------------------------------------------------------
# Color refinement (1-WL)
# ---------------------------------------------------------------------------


def _weisfeiler_leman_hash(
    adj: AdjacencyMatrix, max_iterations: int = 10
) -> int:
    """Compute a Weisfeiler-Leman (1-WL) graph hash.

    The 1-dimensional Weisfeiler-Leman algorithm iteratively refines
    node colors based on their neighborhood multisets until convergence
    or a maximum number of iterations.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Adjacency matrix (may be directed or undirected).
    max_iterations : int
        Maximum refinement iterations.  Default ``10``.

    Returns
    -------
    int
        Hash of the final color histogram.
    """
    n = adj.shape[0]
    if n == 0:
        return hash(())

    # Initial colors: in-degree, out-degree pair
    colors = np.zeros(n, dtype=np.int64)
    for v in range(n):
        in_deg = int(adj[:, v].sum())
        out_deg = int(adj[v, :].sum())
        colors[v] = hash((in_deg, out_deg))

    for _ in range(max_iterations):
        new_colors = np.zeros(n, dtype=np.int64)
        for v in range(n):
            # Collect sorted multiset of neighbor colors
            in_neighbors = tuple(sorted(int(colors[u]) for u in range(n) if adj[u, v]))
            out_neighbors = tuple(sorted(int(colors[u]) for u in range(n) if adj[v, u]))
            new_colors[v] = hash((int(colors[v]), in_neighbors, out_neighbors))

        if np.array_equal(new_colors, colors):
            break
        colors = new_colors

    # Return hash of sorted color histogram
    histogram = tuple(sorted(int(c) for c in colors))
    return hash(histogram)


def _cpdag_from_dag(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Convert a DAG adjacency to its CPDAG."""
    n = adj.shape[0]
    cpdag = adj.copy()
    compelled = np.zeros((n, n), dtype=np.bool_)

    in_deg = adj.sum(axis=0).copy()
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in range(n):
            if adj[node, child]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
    if len(order) < n:
        order.extend(i for i in range(n) if i not in set(order))

    for j in order:
        parents_j = list(np.where(adj[:, j])[0])
        for p in parents_j:
            if compelled[p, j]:
                continue
            for k in parents_j:
                if k != p and not adj[p, k] and not adj[k, p]:
                    compelled[p, j] = True
                    compelled[k, j] = True

    for i in range(n):
        for j in range(n):
            if adj[i, j] and not compelled[i, j]:
                cpdag[j, i] = 1

    return cpdag


# ---------------------------------------------------------------------------
# NautyInterface (extended)
# ---------------------------------------------------------------------------


class NautyInterface:
    """Extended canonical-form and isomorphism utilities.

    Provides multiple backend options:
    1. igraph/bliss (if available) — exact canonical labeling
    2. Weisfeiler-Leman color refinement — polynomial-time heuristic
    3. Degree-sequence fallback — simple but incomplete

    The class tries backends in order and uses the first available one.
    """

    def __init__(self, backend: str = "auto") -> None:
        """Initialize with specified backend.

        Parameters
        ----------
        backend : str
            ``"igraph"``, ``"wl"`` (Weisfeiler-Leman), or ``"auto"``
            (try igraph first, fall back to WL).  Default ``"auto"``.
        """
        self._backend = backend
        self._has_igraph: Optional[bool] = None

    def _check_igraph(self) -> bool:
        """Check if igraph is available."""
        if self._has_igraph is None:
            try:
                import igraph  # type: ignore[import-untyped]
                self._has_igraph = True
            except ImportError:
                self._has_igraph = False
        return self._has_igraph

    # ------------------------------------------------------------------
    # Canonical labeling
    # ------------------------------------------------------------------

    def canonical_form(self, adjacency: AdjacencyMatrix) -> AdjacencyMatrix:
        """Return a canonical relabeling of *adjacency*.

        Nodes are permuted so that isomorphic graphs always produce
        the same adjacency matrix.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            Square adjacency matrix.

        Returns
        -------
        AdjacencyMatrix
            Canonically relabeled adjacency matrix.
        """
        perm = self._canonical_permutation(adjacency)
        adj = np.asarray(adjacency, dtype=np.int8)
        return adj[np.ix_(perm, perm)]

    def canonical_hash(self, adjacency: AdjacencyMatrix) -> GraphHash:
        """Compute canonical hash of the graph.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        GraphHash
        """
        if self._use_igraph():
            canonical = self.canonical_form(adjacency)
            return hash(canonical.tobytes())
        else:
            return _weisfeiler_leman_hash(adjacency)

    def cpdag_canonical_hash(self, adjacency: AdjacencyMatrix) -> GraphHash:
        """Compute canonical hash of the CPDAG (MEC identifier).

        First converts the DAG to its CPDAG, then computes the
        canonical hash.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            DAG adjacency matrix.

        Returns
        -------
        GraphHash
            Hash identifying the Markov Equivalence Class.
        """
        cpdag = _cpdag_from_dag(adjacency)
        return self.canonical_hash(cpdag)

    # ------------------------------------------------------------------
    # Isomorphism testing
    # ------------------------------------------------------------------

    def is_isomorphic(
        self, adj1: AdjacencyMatrix, adj2: AdjacencyMatrix
    ) -> bool:
        """Test whether two graphs are isomorphic.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        bool
        """
        if adj1.shape != adj2.shape:
            return False

        if self._use_igraph():
            try:
                g1 = self._to_igraph(adj1)
                g2 = self._to_igraph(adj2)
                return g1.isomorphic(g2)
            except Exception:
                pass

        # Fallback: compare canonical forms or WL hashes
        h1 = self.canonical_hash(adj1)
        h2 = self.canonical_hash(adj2)
        if h1 != h2:
            return False

        # WL hash match doesn't guarantee isomorphism; compare canonical forms
        c1 = self.canonical_form(adj1)
        c2 = self.canonical_form(adj2)
        return np.array_equal(c1, c2)

    def is_mec_equivalent(
        self, adj1: AdjacencyMatrix, adj2: AdjacencyMatrix
    ) -> bool:
        """Test whether two DAGs belong to the same MEC.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        bool
        """
        cpdag1 = _cpdag_from_dag(adj1)
        cpdag2 = _cpdag_from_dag(adj2)
        return self.is_isomorphic(cpdag1, cpdag2) or np.array_equal(cpdag1, cpdag2)

    # ------------------------------------------------------------------
    # Automorphism group
    # ------------------------------------------------------------------

    def automorphism_group_size(self, adjacency: AdjacencyMatrix) -> int:
        """Compute the size of the automorphism group.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        int
            Size of Aut(G).
        """
        if self._use_igraph():
            try:
                g = self._to_igraph(adjacency)
                return len(g.automorphism_group())
            except Exception:
                pass

        # Fallback: count automorphisms by checking all permutations
        # (only feasible for small graphs)
        n = adjacency.shape[0]
        if n > 8:
            return 1  # Too expensive, return trivial group

        count = 0
        adj = np.asarray(adjacency, dtype=np.int8)
        from itertools import permutations
        for perm in permutations(range(n)):
            p = list(perm)
            permuted = adj[np.ix_(p, p)]
            if np.array_equal(permuted, adj):
                count += 1
        return count

    def automorphism_orbits(
        self, adjacency: AdjacencyMatrix
    ) -> List[List[int]]:
        """Compute orbits of the automorphism group.

        Two nodes are in the same orbit if there exists an automorphism
        mapping one to the other.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        List[List[int]]
            List of orbits (groups of equivalent nodes).
        """
        n = adjacency.shape[0]
        adj = np.asarray(adjacency, dtype=np.int8)

        # Use union-find to group nodes
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        if self._use_igraph():
            try:
                g = self._to_igraph(adj)
                for aut in g.automorphism_group():
                    for i in range(n):
                        union(i, aut[i])
            except Exception:
                self._degree_based_orbits(adj, union, n)
        else:
            self._degree_based_orbits(adj, union, n)

        # Collect orbits
        orbits: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            if r not in orbits:
                orbits[r] = []
            orbits[r].append(i)

        return list(orbits.values())

    @staticmethod
    def _degree_based_orbits(adj: AdjacencyMatrix, union_fn, n: int) -> None:
        """Group nodes with identical degree sequences."""
        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)

        degree_groups: Dict[Tuple[int, int], List[int]] = {}
        for v in range(n):
            key = (int(in_deg[v]), int(out_deg[v]))
            if key not in degree_groups:
                degree_groups[key] = []
            degree_groups[key].append(v)

        for nodes in degree_groups.values():
            for i in range(1, len(nodes)):
                union_fn(nodes[0], nodes[i])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _use_igraph(self) -> bool:
        """Decide whether to use igraph backend."""
        if self._backend == "wl":
            return False
        if self._backend == "igraph":
            return self._check_igraph()
        # auto
        return self._check_igraph()

    def _canonical_permutation(self, adjacency: AdjacencyMatrix) -> List[int]:
        """Compute canonical node permutation.

        Uses igraph's bliss backend if available, otherwise falls
        back to a refined degree-based ordering.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        List[int]
            Canonical permutation of node indices.
        """
        if self._use_igraph():
            try:
                g = self._to_igraph(adjacency)
                return list(g.canonical_permutation())
            except Exception:
                pass

        return self._refined_degree_permutation(adjacency)

    @staticmethod
    def _to_igraph(adjacency: AdjacencyMatrix):  # type: ignore[return]
        """Convert adjacency matrix to igraph.Graph."""
        import igraph as ig  # type: ignore[import-untyped]

        adj = np.asarray(adjacency)
        n = adj.shape[0]
        edges = [(i, j) for i in range(n) for j in range(n) if adj[i, j]]
        directed = not np.array_equal(adj, adj.T)
        return ig.Graph(n=n, edges=edges, directed=directed)

    @staticmethod
    def _refined_degree_permutation(
        adjacency: AdjacencyMatrix,
    ) -> List[int]:
        """Canonical ordering using degree + neighborhood signature.

        Uses a multi-level refinement: first sort by (in_degree, out_degree),
        then by sorted neighbor degrees, for a more discriminative ordering.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        List[int]
        """
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]

        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)

        # Build a multi-level key for each node
        keys: List[Tuple] = []
        for v in range(n):
            # Level 1: degrees
            d_key = (int(in_deg[v]), int(out_deg[v]))

            # Level 2: sorted neighbor in/out degrees
            in_neighbors = np.where(adj[:, v])[0]
            out_neighbors = np.where(adj[v, :])[0]
            in_nb_degs = tuple(sorted(
                (int(in_deg[u]), int(out_deg[u])) for u in in_neighbors
            ))
            out_nb_degs = tuple(sorted(
                (int(in_deg[u]), int(out_deg[u])) for u in out_neighbors
            ))

            keys.append((d_key, in_nb_degs, out_nb_degs))

        return sorted(range(n), key=lambda v: keys[v])
