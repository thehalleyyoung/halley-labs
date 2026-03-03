"""Directed Acyclic Graph (DAG) representation for causal discovery.

Provides a mutable DAG backed by a numpy adjacency matrix with
topological-order caching, CPDAG conversion, d-separation testing,
moralization, and many graph-theoretic queries.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix, EdgeList, NodeSet, TopologicalOrder


class DAGError(Exception):
    """Raised when a DAG invariant is violated (e.g., cycle detected)."""


class DAG:
    """Mutable directed acyclic graph backed by an adjacency matrix.

    Parameters
    ----------
    adjacency : AdjacencyMatrix
        Square binary matrix where ``adjacency[i, j] = 1`` iff edge *i → j*.

    Raises
    ------
    DAGError
        If *adjacency* contains a cycle or is not square.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, adjacency: AdjacencyMatrix) -> None:
        adjacency = np.asarray(adjacency, dtype=np.int8)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise DAGError("Adjacency matrix must be square.")
        if not DAG.is_acyclic(adjacency):
            raise DAGError("Adjacency matrix contains a cycle.")
        self._adj: AdjacencyMatrix = adjacency.copy()
        self._n: int = adjacency.shape[0]
        self._topo_order: Optional[TopologicalOrder] = None
        self._parent_cache: Dict[int, NodeSet] = {}
        self._children_cache: Dict[int, NodeSet] = {}

    # ------------------------------------------------------------------
    # Class methods / alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def is_acyclic(cls, adjacency: AdjacencyMatrix) -> bool:
        """Return ``True`` if *adjacency* encodes a DAG (no directed cycles).

        Uses Kahn's algorithm (iterative topological sort).
        """
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]
        in_degree = adj.sum(axis=0).copy()
        queue = deque(int(i) for i in range(n) if in_degree[i] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in range(n):
                if adj[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited == n

    @classmethod
    def from_edges(cls, n: int, edges: EdgeList) -> "DAG":
        """Create a DAG from a node count and edge list."""
        adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
        for src, tgt in edges:
            adj[src, tgt] = 1
        return cls(adj)

    @classmethod
    def from_edge_list(cls, n: int, edges: EdgeList) -> "DAG":
        """Alias for :meth:`from_edges`."""
        return cls.from_edges(n, edges)

    @classmethod
    def from_adjacency(cls, adjacency: AdjacencyMatrix) -> "DAG":
        """Create a DAG from a raw adjacency matrix (convenience wrapper)."""
        return cls(adjacency)

    @classmethod
    def from_adjacency_matrix(cls, adjacency: AdjacencyMatrix) -> "DAG":
        """Create a DAG from a raw adjacency matrix."""
        return cls(adjacency)

    @classmethod
    def from_networkx(cls, graph: "nx.DiGraph") -> "DAG":  # type: ignore[name-defined]
        """Create a DAG from a ``networkx.DiGraph``.

        Node labels are mapped to contiguous integers in sorted order.
        """
        import networkx as nx  # noqa: F811

        if not isinstance(graph, nx.DiGraph):
            raise DAGError("Expected a networkx.DiGraph instance.")
        nodes = sorted(graph.nodes())
        idx = {nd: i for i, nd in enumerate(nodes)}
        n = len(nodes)
        adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
        for u, v in graph.edges():
            adj[idx[u], idx[v]] = 1
        return cls(adj)

    @classmethod
    def empty(cls, n: int) -> "DAG":
        """Create an empty DAG with *n* nodes and no edges."""
        return cls(np.zeros((n, n), dtype=np.int8))

    @staticmethod
    def random_dag(
        n_nodes: int,
        edge_prob: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> "DAG":
        """Generate an Erdős–Rényi random DAG.

        Nodes are given a random permutation as their topological order.
        Each forward edge ``(π[i], π[j])`` for ``i < j`` is included
        independently with probability *edge_prob*.
        """
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n_nodes).tolist()
        adj: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for a in range(n_nodes):
            for b in range(a + 1, n_nodes):
                if rng.random() < edge_prob:
                    adj[perm[a], perm[b]] = 1
        return DAG(adj)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._n

    @property
    def n_nodes(self) -> int:
        """Alias for :pyattr:`num_nodes`."""
        return self._n

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return int(self._adj.sum())

    @property
    def n_edges(self) -> int:
        """Alias for :pyattr:`num_edges`."""
        return int(self._adj.sum())

    @property
    def adjacency(self) -> AdjacencyMatrix:
        """Return a *copy* of the adjacency matrix."""
        return self._adj.copy()

    @property
    def adjacency_matrix(self) -> AdjacencyMatrix:
        """Return a *copy* of the adjacency matrix (alias)."""
        return self._adj.copy()

    @property
    def edges(self) -> EdgeList:
        """Return the edge list as ``(source, target)`` pairs."""
        rows, cols = np.nonzero(self._adj)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def edge_list(self) -> EdgeList:
        """Alias for :pyattr:`edges`."""
        return self.edges

    @property
    def node_list(self) -> List[int]:
        """Return the list of node indices."""
        return list(range(self._n))

    @property
    def topological_order(self) -> TopologicalOrder:
        """Return a cached topological ordering of the nodes (Kahn's)."""
        if self._topo_order is None:
            self._topo_order = self._compute_topo_order()
        return list(self._topo_order)

    @property
    def density(self) -> float:
        """Edge density: num_edges / (n*(n-1))."""
        max_edges = self._n * (self._n - 1)
        return self.num_edges / max_edges if max_edges > 0 else 0.0

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    def parents(self, node: int) -> NodeSet:
        """Return the parent set of *node* (cached)."""
        if node not in self._parent_cache:
            self._parent_cache[node] = frozenset(
                int(i) for i in np.nonzero(self._adj[:, node])[0]
            )
        return self._parent_cache[node]

    def children(self, node: int) -> NodeSet:
        """Return the child set of *node* (cached)."""
        if node not in self._children_cache:
            self._children_cache[node] = frozenset(
                int(i) for i in np.nonzero(self._adj[node, :])[0]
            )
        return self._children_cache[node]

    def ancestors(self, node: int) -> NodeSet:
        """Return all ancestors of *node* (excluding *node* itself) via BFS."""
        visited: set[int] = set()
        queue = deque(self.parents(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.parents(cur) - visited)
        return frozenset(visited)

    def descendants(self, node: int) -> NodeSet:
        """Return all descendants of *node* (excluding *node* itself) via BFS."""
        visited: set[int] = set()
        queue = deque(self.children(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.children(cur) - visited)
        return frozenset(visited)

    def neighbors(self, node: int) -> Set[int]:
        """Return undirected neighbours (parents ∪ children)."""
        return set(self.parents(node)) | set(self.children(node))

    def in_degree(self, node: int) -> int:
        """Number of incoming edges (parents) of *node*."""
        return int(self._adj[:, node].sum())

    def out_degree(self, node: int) -> int:
        """Number of outgoing edges (children) of *node*."""
        return int(self._adj[node, :].sum())

    def degree(self, node: int) -> int:
        """Total degree (in + out) of *node*."""
        return self.in_degree(node) + self.out_degree(node)

    # ------------------------------------------------------------------
    # Edge queries / mutations
    # ------------------------------------------------------------------

    def has_edge(self, i: int, j: int) -> bool:
        """Return ``True`` if edge *i → j* exists."""
        return bool(self._adj[i, j])

    def add_edge(self, i: int, j: int) -> None:
        """Add edge *i → j*, raising :class:`DAGError` if it creates a cycle.

        Uses a targeted DFS reachability check: adding i→j creates a cycle
        iff j can already reach i.
        """
        if i == j:
            raise DAGError("Self-loops are not allowed.")
        if self._adj[i, j]:
            return  # Already present
        # Check if j can reach i (would form cycle i→j→...→i)
        if self._can_reach(j, i):
            raise DAGError(f"Adding edge {i} -> {j} would create a cycle.")
        self._adj[i, j] = 1
        self._invalidate_caches()

    def remove_edge(self, i: int, j: int) -> None:
        """Remove edge *i → j* (no-op if absent)."""
        if self._adj[i, j]:
            self._adj[i, j] = 0
            self._invalidate_caches()

    def reverse_edge(self, i: int, j: int) -> None:
        """Reverse edge *i → j* to *j → i*.

        Raises :class:`DAGError` if the edge doesn't exist or reversal
        would create a cycle.  Uses DFS from i in the modified graph
        to check if i can reach j (which would form j→i→...→j).
        """
        if not self._adj[i, j]:
            raise DAGError(f"Edge {i} -> {j} does not exist.")
        # Temporarily remove i→j and check if adding j→i is safe
        self._adj[i, j] = 0
        can_create_cycle = self._can_reach(i, j)
        if can_create_cycle:
            self._adj[i, j] = 1  # Revert
            raise DAGError(f"Reversing edge {i} -> {j} would create a cycle.")
        self._adj[j, i] = 1
        self._invalidate_caches()

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """Return ``True`` if the current graph contains a directed cycle.

        Uses DFS with 3-colour marking.
        """
        return not DAG.is_acyclic(self._adj)

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> TopologicalOrder:
        """Return a topological ordering using Kahn's algorithm."""
        return self.topological_order

    # ------------------------------------------------------------------
    # d-separation
    # ------------------------------------------------------------------

    def d_separated(self, x: Set[int], y: Set[int], z: Set[int]) -> bool:
        """Test whether *x* and *y* are d-separated given *z*.

        Implements the Bayes-Ball algorithm: X ⊥ Y | Z iff no active trail
        exists between any node in X and any node in Y given Z.

        Parameters
        ----------
        x, y : Set[int]
            Disjoint non-empty sets of query nodes.
        z : Set[int]
            Conditioning set (may be empty).

        Returns
        -------
        bool
            ``True`` if X ⊥ Y | Z in this DAG.
        """
        # Use the reachability-based algorithm on the moralized ancestral graph
        # More efficient: Bayes-Ball algorithm
        reachable = self._bayes_ball_reachable(x, z)
        return len(reachable & y) == 0

    def _bayes_ball_reachable(self, source: Set[int], observed: Set[int]) -> Set[int]:
        """Find all nodes reachable from *source* via active trails given *observed*.

        Uses the Bayes-Ball algorithm (Shachter 1998).  A trail is active if:
        - Chain a→b→c or fork a←b→c: b is NOT in *observed*
        - Collider a→b←c: b or a descendant of b IS in *observed*
        """
        z_set = set(observed)

        # Pre-compute which nodes have a descendant in z (or are in z)
        has_desc_in_z: Set[int] = set()
        for node in z_set:
            has_desc_in_z.add(node)
            # Add all ancestors of observed nodes
            for anc in self.ancestors(node):
                has_desc_in_z.add(anc)

        # BFS on (node, direction) pairs
        # direction: "up" means we arrived at node going up (from child to parent)
        #            "down" means we arrived going down (from parent to child)
        visited: Set[Tuple[int, str]] = set()
        queue: deque[Tuple[int, str]] = deque()
        reachable: Set[int] = set()

        # Initialize: schedule visits from source nodes in both directions
        for s in source:
            queue.append((s, "up"))
            queue.append((s, "down"))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            reachable.add(node)

            if direction == "up" and node not in z_set:
                # Came from a child, node not observed:
                # Can continue up to parents (chain/fork)
                for parent in self.parents(node):
                    if (parent, "up") not in visited:
                        queue.append((parent, "up"))
                # Can continue down to children (fork)
                for child in self.children(node):
                    if (child, "down") not in visited:
                        queue.append((child, "down"))

            elif direction == "down":
                # Came from a parent
                if node not in z_set:
                    # Not observed: continue down to children (chain)
                    for child in self.children(node):
                        if (child, "down") not in visited:
                            queue.append((child, "down"))
                if node in has_desc_in_z:
                    # Node or descendant observed: collider is active,
                    # continue up to parents
                    for parent in self.parents(node):
                        if (parent, "up") not in visited:
                            queue.append((parent, "up"))

        return reachable

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def v_structures(self) -> List[Tuple[int, int, int]]:
        """Find all v-structures (colliders) i → j ← k where i and k are
        not adjacent (no edge in either direction).

        Returns
        -------
        List[Tuple[int, int, int]]
            List of ``(i, j, k)`` triples with ``i < k`` for canonical ordering.
        """
        result: List[Tuple[int, int, int]] = []
        for j in range(self._n):
            pa = sorted(self.parents(j))
            for a_idx in range(len(pa)):
                for b_idx in range(a_idx + 1, len(pa)):
                    i, k = pa[a_idx], pa[b_idx]
                    # Check not adjacent
                    if not self._adj[i, k] and not self._adj[k, i]:
                        result.append((i, j, k))
        return result

    def skeleton(self) -> AdjacencyMatrix:
        """Return the undirected skeleton (symmetric binary matrix)."""
        skel = self._adj | self._adj.T
        return skel.astype(np.int8)

    def moralize(self) -> AdjacencyMatrix:
        """Return the moralized graph (marry parents, drop directions)."""
        return self.moral_graph()

    def moral_graph(self) -> AdjacencyMatrix:
        """Return the moralized graph (marry parents, drop directions)."""
        n = self.num_nodes
        moral: AdjacencyMatrix = self.skeleton()
        for node in range(n):
            pa = list(self.parents(node))
            for a in range(len(pa)):
                for b in range(a + 1, len(pa)):
                    moral[pa[a], pa[b]] = 1
                    moral[pa[b], pa[a]] = 1
        return moral

    def subgraph(self, nodes: List[int]) -> "DAG":
        """Return the induced sub-DAG over *nodes*."""
        idx = sorted(nodes)
        sub = self._adj[np.ix_(idx, idx)]
        return DAG(sub)

    def is_connected(self) -> bool:
        """Return ``True`` if the underlying undirected graph is connected."""
        if self._n <= 1:
            return True
        visited = set()
        queue: deque[int] = deque([0])
        visited.add(0)
        undirected = (self._adj | self._adj.T).astype(bool)
        while queue:
            node = queue.popleft()
            for nb in range(self._n):
                if undirected[node, nb] and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == self._n

    def connected_components(self) -> List[List[int]]:
        """Return weakly connected components as lists of node indices."""
        undirected = (self._adj | self._adj.T).astype(bool)
        visited = np.zeros(self._n, dtype=bool)
        components: List[List[int]] = []
        for start in range(self._n):
            if visited[start]:
                continue
            component: List[int] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in range(self._n):
                    if undirected[node, nb] and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            components.append(component)
        return components

    def longest_path(self) -> int:
        """Return the length of the longest directed path (number of edges).

        Uses dynamic programming on the topological order.
        """
        if self._n == 0:
            return 0
        order = self.topological_order
        dist = np.zeros(self._n, dtype=int)
        for u in order:
            for v in range(self._n):
                if self._adj[u, v]:
                    if dist[u] + 1 > dist[v]:
                        dist[v] = dist[u] + 1
        return int(dist.max())

    # ------------------------------------------------------------------
    # CPDAG conversion
    # ------------------------------------------------------------------

    def to_cpdag(self) -> AdjacencyMatrix:
        """Convert to the Completed Partially Directed Acyclic Graph (CPDAG).

        Computable edges (those in every Markov-equivalent DAG) stay directed;
        reversible edges become undirected (represented as two directed entries).
        An edge *i → j* is compelled iff there exists *k → j* with *k* not
        adjacent to *i*, otherwise it is reversible.
        """
        n = self.num_nodes
        cpdag: AdjacencyMatrix = self._adj.copy()
        order = self.topological_order

        compelled = np.zeros((n, n), dtype=np.bool_)

        for j in order:
            parents_j = list(self.parents(j))
            for p in parents_j:
                if compelled[p, j]:
                    continue
                is_compelled = False
                for k in parents_j:
                    if k == p:
                        continue
                    if not self._adj[p, k] and not self._adj[k, p]:
                        is_compelled = True
                        break
                if is_compelled:
                    compelled[p, j] = True
                    for k in parents_j:
                        if k != p and not self._adj[p, k] and not self._adj[k, p]:
                            compelled[k, j] = True

        for i in range(n):
            for j in range(n):
                if self._adj[i, j] and not compelled[i, j]:
                    cpdag[j, i] = 1

        return cpdag

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_networkx(self) -> "nx.DiGraph":  # type: ignore[name-defined]
        """Convert to a ``networkx.DiGraph``."""
        import networkx as nx  # noqa: F811

        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(self.edges)
        return G

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check all DAG invariants. Returns ``True`` if valid.

        Checks:
        1. Adjacency matrix is square and binary
        2. No self-loops
        3. Graph is acyclic
        4. Cached topological order is consistent (if present)
        """
        # Square
        if self._adj.ndim != 2 or self._adj.shape[0] != self._adj.shape[1]:
            return False
        if self._adj.shape[0] != self._n:
            return False
        # Binary
        if not np.all((self._adj == 0) | (self._adj == 1)):
            return False
        # No self-loops
        if np.any(np.diag(self._adj)):
            return False
        # Acyclic
        if not DAG.is_acyclic(self._adj):
            return False
        # Cached topo order consistency
        if self._topo_order is not None:
            if len(self._topo_order) != self._n:
                return False
            if set(self._topo_order) != set(range(self._n)):
                return False
            pos = {node: i for i, node in enumerate(self._topo_order)}
            for i, j in self.edges:
                if pos[i] >= pos[j]:
                    return False
        return True

    # ------------------------------------------------------------------
    # Copy / equality / hashing
    # ------------------------------------------------------------------

    def copy(self) -> "DAG":
        """Return a deep copy."""
        new = object.__new__(DAG)
        new._adj = self._adj.copy()
        new._n = self._n
        new._topo_order = list(self._topo_order) if self._topo_order is not None else None
        new._parent_cache = {}
        new._children_cache = {}
        return new

    def __copy__(self) -> "DAG":
        return self.copy()

    def __deepcopy__(self, memo: dict) -> "DAG":
        new = object.__new__(DAG)
        new._adj = self._adj.copy()
        new._n = self._n
        new._topo_order = list(self._topo_order) if self._topo_order is not None else None
        new._parent_cache = {}
        new._children_cache = {}
        memo[id(self)] = new
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAG):
            return NotImplemented
        return np.array_equal(self._adj, other._adj)

    def __hash__(self) -> int:
        return hash(self._adj.tobytes())

    def __repr__(self) -> str:
        return f"DAG(nodes={self.num_nodes}, edges={self.num_edges})"

    def __len__(self) -> int:
        return self._n

    def __contains__(self, edge: Tuple[int, int]) -> bool:
        i, j = edge
        return bool(self._adj[i, j])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._n))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_topo_order(self) -> TopologicalOrder:
        """Kahn's algorithm."""
        n = self.num_nodes
        in_deg = self._adj.sum(axis=0).copy()
        queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
        order: TopologicalOrder = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in range(n):
                if self._adj[node, child]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        return order

    def _invalidate_caches(self) -> None:
        """Clear all cached data after a structural change."""
        self._topo_order = None
        self._parent_cache.clear()
        self._children_cache.clear()

    def _can_reach(self, source: int, target: int) -> bool:
        """Return True if there is a directed path from *source* to *target*.

        Uses BFS on children.
        """
        if source == target:
            return True
        visited: Set[int] = set()
        queue: deque[int] = deque()
        queue.append(source)
        visited.add(source)
        while queue:
            node = queue.popleft()
            for child in range(self._n):
                if self._adj[node, child]:
                    if child == target:
                        return True
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
        return False
