"""Graph isomorphism utilities backed by igraph."""
from __future__ import annotations

from typing import List

import numpy as np

from causal_qd.types import AdjacencyMatrix


class NautyInterface:
    """Canonical-form and isomorphism utilities using python-igraph.

    `igraph <https://python.igraph.org/>`_ wraps the *bliss* (or optionally
    *nauty*) graph-isomorphism backend to provide efficient canonical
    labelling and automorphism group computation.

    If *igraph* is not installed the methods fall back to a simple
    degree-sequence heuristic (which is **not** isomorphism-complete).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canonical_form(self, adjacency: AdjacencyMatrix) -> AdjacencyMatrix:
        """Return a canonical relabelling of *adjacency*.

        Nodes are permuted so that isomorphic graphs always produce the
        same adjacency matrix.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            Square (possibly non-symmetric) adjacency matrix.

        Returns
        -------
        AdjacencyMatrix
        """
        perm = self._canonical_permutation(adjacency)
        adj = np.asarray(adjacency, dtype=np.int8)
        return adj[np.ix_(perm, perm)]

    def automorphism_group_size(self, adjacency: AdjacencyMatrix) -> int:
        """Return the size of the automorphism group of the graph.

        Parameters
        ----------
        adjacency : AdjacencyMatrix

        Returns
        -------
        int
        """
        try:
            g = self._to_igraph(adjacency)
            return len(g.automorphism_group())
        except Exception:
            # Fallback: trivial group
            return 1

    def is_isomorphic(
        self, adj1: AdjacencyMatrix, adj2: AdjacencyMatrix
    ) -> bool:
        """Check whether two graphs are isomorphic.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        bool
        """
        if adj1.shape != adj2.shape:
            return False
        try:
            g1 = self._to_igraph(adj1)
            g2 = self._to_igraph(adj2)
            return g1.isomorphic(g2)
        except Exception:
            # Fallback: compare canonical forms
            c1 = self.canonical_form(adj1)
            c2 = self.canonical_form(adj2)
            return np.array_equal(c1, c2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_igraph(adjacency: AdjacencyMatrix):  # type: ignore[return]
        """Convert a numpy adjacency matrix to an igraph.Graph."""
        import igraph as ig  # type: ignore[import-untyped]

        adj = np.asarray(adjacency)
        n = adj.shape[0]
        edges: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    edges.append((i, j))
        directed = not np.array_equal(adj, adj.T)
        return ig.Graph(n=n, edges=edges, directed=directed)

    def _canonical_permutation(self, adjacency: AdjacencyMatrix) -> List[int]:
        """Return the canonical permutation of node indices.

        Tries igraph's bliss backend first; falls back to a simple
        degree-based ordering.
        """
        try:
            g = self._to_igraph(adjacency)
            return list(g.canonical_permutation())
        except Exception:
            return self._degree_permutation(adjacency)

    @staticmethod
    def _degree_permutation(adjacency: AdjacencyMatrix) -> List[int]:
        """Fallback canonical ordering based on degree sequences."""
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]
        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)
        keys = [(int(in_deg[v]), int(out_deg[v])) for v in range(n)]
        return sorted(range(n), key=lambda v: keys[v])
