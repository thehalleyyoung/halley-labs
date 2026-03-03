"""Factory methods for constructing common DAG topologies.

All methods return :class:`~causal_qd.core.dag.DAG` instances.
Supports Erdős–Rényi, scale-free (Barabási–Albert), small-world,
chain, tree, star, and complete DAG generation.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix, TopologicalOrder


class GraphFactory:
    """Collection of static helpers for building DAGs."""

    # ------------------------------------------------------------------
    # Basic topologies
    # ------------------------------------------------------------------

    @staticmethod
    def empty(n_nodes: int) -> DAG:
        """Return a DAG with *n_nodes* and no edges."""
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        return DAG(adj)

    @staticmethod
    def complete(n_nodes: int) -> DAG:
        """Return a complete DAG with topological order ``0, 1, …, n-1``.

        Edge *i → j* exists for every *i < j*.
        """
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                adj[i, j] = 1
        return DAG(adj)

    @staticmethod
    def complete_dag(n_nodes: int) -> DAG:
        """Alias for :meth:`complete`."""
        return GraphFactory.complete(n_nodes)

    @staticmethod
    def chain(n_nodes: int) -> DAG:
        """Return a chain DAG: ``0 → 1 → 2 → … → n-1``."""
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for i in range(n_nodes - 1):
            adj[i, i + 1] = 1
        return DAG(adj)

    @staticmethod
    def chain_graph(n_nodes: int) -> DAG:
        """Alias for :meth:`chain`."""
        return GraphFactory.chain(n_nodes)

    @staticmethod
    def star(n_nodes: int, center: int = 0) -> DAG:
        """Return a star DAG where *center* points to every other node."""
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for i in range(n_nodes):
            if i != center:
                adj[center, i] = 1
        return DAG(adj)

    # ------------------------------------------------------------------
    # Random DAG generators
    # ------------------------------------------------------------------

    @staticmethod
    def random_dag(
        n_nodes: int,
        edge_prob: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        model: str = "er",
    ) -> DAG:
        """Generate a random DAG using the specified model.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        edge_prob : float
            Edge probability (interpretation depends on model).
        rng : Generator, optional
            NumPy random generator.
        model : str
            One of ``"er"`` (Erdős–Rényi), ``"sf"`` (scale-free),
            ``"sw"`` (small-world).  Default ``"er"``.

        Returns
        -------
        DAG
        """
        if rng is None:
            rng = np.random.default_rng()
        if model == "er":
            return GraphFactory._random_er(n_nodes, edge_prob, rng)
        elif model == "sf":
            return GraphFactory._random_scale_free(n_nodes, rng)
        elif model == "sw":
            return GraphFactory._random_small_world(n_nodes, edge_prob, rng)
        else:
            raise ValueError(f"Unknown model: {model!r}")

    @staticmethod
    def _random_er(
        n_nodes: int,
        edge_prob: float,
        rng: np.random.Generator,
    ) -> DAG:
        """Erdős–Rényi random DAG: each forward edge included with *edge_prob*.

        Nodes are assigned a random topological order via permutation.
        """
        perm = rng.permutation(n_nodes).tolist()
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for a in range(n_nodes):
            for b in range(a + 1, n_nodes):
                if rng.random() < edge_prob:
                    adj[perm[a], perm[b]] = 1
        return DAG(adj)

    @staticmethod
    def _random_scale_free(
        n_nodes: int,
        rng: np.random.Generator,
        m: int = 2,
    ) -> DAG:
        """Barabási–Albert scale-free DAG.

        Nodes are added one at a time.  Each new node attaches to *m*
        existing nodes using preferential attachment based on in-degree + 1
        (to avoid zero probability for degree-0 nodes).  Edges are directed
        from earlier to later nodes in insertion order, guaranteeing a DAG.

        Parameters
        ----------
        n_nodes : int
            Total number of nodes.
        rng : Generator
            Random generator.
        m : int
            Number of edges each new node creates (default 2).
        """
        if n_nodes <= m:
            return GraphFactory.complete(n_nodes)

        perm = rng.permutation(n_nodes).tolist()
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)

        # Start with a complete graph on the first m+1 nodes
        for a in range(m + 1):
            for b in range(a + 1, m + 1):
                adj[perm[a], perm[b]] = 1

        # Degree array for preferential attachment
        in_degree = np.zeros(n_nodes, dtype=np.float64)
        for j in range(n_nodes):
            in_degree[j] = adj[:, j].sum()

        for new_idx in range(m + 1, n_nodes):
            new_node = perm[new_idx]
            candidates = [perm[k] for k in range(new_idx)]

            # Preferential attachment: prob ∝ in_degree + 1
            probs = np.array([in_degree[c] + 1.0 for c in candidates])
            probs /= probs.sum()

            chosen = rng.choice(
                len(candidates),
                size=min(m, len(candidates)),
                replace=False,
                p=probs,
            )
            for idx in chosen:
                parent = candidates[idx]
                adj[parent, new_node] = 1
                in_degree[new_node] += 1

        return DAG(adj)

    @staticmethod
    def _random_small_world(
        n_nodes: int,
        rewire_prob: float,
        rng: np.random.Generator,
        k: int = 4,
    ) -> DAG:
        """Watts–Strogatz-style small-world DAG.

        Starts with a ring lattice where each node connects to its *k*
        nearest (forward) neighbours, then rewires each edge with
        probability *rewire_prob* to a random forward node.  All edges
        are directed from lower to higher node indices in a random
        permutation to guarantee acyclicity.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        rewire_prob : float
            Probability of rewiring each edge.
        rng : Generator
            Random generator.
        k : int
            Number of nearest neighbours on each side (default 4).
        """
        perm = rng.permutation(n_nodes).tolist()
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)

        # Create ring lattice edges (directed forward in perm order)
        half_k = max(k // 2, 1)
        for a in range(n_nodes):
            for offset in range(1, half_k + 1):
                b = (a + offset) % n_nodes
                # Ensure a -> b is forward in perm ordering
                if a < b:
                    adj[perm[a], perm[b]] = 1
                else:
                    adj[perm[b], perm[a]] = 1

        # Rewire edges
        edges = list(zip(*np.nonzero(adj)))
        for i_node, j_node in edges:
            if rng.random() < rewire_prob:
                adj[i_node, j_node] = 0
                # Find position in perm
                pos_i = perm.index(i_node)
                # Pick a random forward target
                forward_nodes = [perm[q] for q in range(pos_i + 1, n_nodes)
                                 if not adj[i_node, perm[q]] and perm[q] != i_node]
                if forward_nodes:
                    new_target = forward_nodes[rng.integers(0, len(forward_nodes))]
                    adj[i_node, new_target] = 1
                else:
                    adj[i_node, j_node] = 1  # Revert if no valid target

        return DAG(adj)

    @staticmethod
    def random_sparse(
        n_nodes: int,
        expected_edges: int,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Random DAG with approximately *expected_edges* edges.

        The edge probability is calibrated to the number of possible forward
        edges ``n*(n-1)/2``.
        """
        if rng is None:
            rng = np.random.default_rng()
        max_edges = n_nodes * (n_nodes - 1) // 2
        prob = min(expected_edges / max(max_edges, 1), 1.0)
        return GraphFactory._random_er(n_nodes, edge_prob=prob, rng=rng)

    # ------------------------------------------------------------------
    # Tree generation
    # ------------------------------------------------------------------

    @staticmethod
    def tree(
        n_nodes: int,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Random directed tree rooted at node 0.

        Each node ``k`` (for ``k >= 1``) picks a parent uniformly at random
        from ``{0, …, k-1}``, guaranteeing a DAG by construction.
        """
        if rng is None:
            rng = np.random.default_rng()
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for k in range(1, n_nodes):
            parent = int(rng.integers(0, k))
            adj[parent, k] = 1
        return DAG(adj)

    @staticmethod
    def tree_graph(
        n_nodes: int,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Alias for :meth:`tree`."""
        return GraphFactory.tree(n_nodes, rng)

    # ------------------------------------------------------------------
    # Order-based construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_ordering(
        ordering: TopologicalOrder,
        edge_prob: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Random DAG consistent with a given topological *ordering*.

        For every pair ``(ordering[a], ordering[b])`` with ``a < b``, the
        edge is included with probability *edge_prob*.
        """
        if rng is None:
            rng = np.random.default_rng()
        n = len(ordering)
        adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < edge_prob:
                    adj[ordering[a], ordering[b]] = 1
        return DAG(adj)

    @staticmethod
    def from_topological_order(
        ordering: TopologicalOrder,
        edge_prob: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Alias for :meth:`from_ordering`."""
        return GraphFactory.from_ordering(ordering, edge_prob, rng)

    # ------------------------------------------------------------------
    # Bipartite / layered
    # ------------------------------------------------------------------

    @staticmethod
    def layered(
        layer_sizes: Sequence[int],
        edge_prob: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Random layered DAG.

        Nodes are arranged in layers. Edges only go from earlier layers to
        later layers.  Between each pair of consecutive layers, each edge
        exists independently with probability *edge_prob*.

        Parameters
        ----------
        layer_sizes : Sequence[int]
            Number of nodes in each layer.
        edge_prob : float
            Edge probability between any two nodes in consecutive layers.
        rng : Generator, optional
            Random generator.
        """
        if rng is None:
            rng = np.random.default_rng()
        n = sum(layer_sizes)
        adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)

        offsets: List[int] = []
        s = 0
        for sz in layer_sizes:
            offsets.append(s)
            s += sz

        for layer_idx in range(len(layer_sizes) - 1):
            for di in range(layer_sizes[layer_idx]):
                for dj in range(layer_sizes[layer_idx + 1]):
                    if rng.random() < edge_prob:
                        src = offsets[layer_idx] + di
                        tgt = offsets[layer_idx + 1] + dj
                        adj[src, tgt] = 1

        return DAG(adj)
