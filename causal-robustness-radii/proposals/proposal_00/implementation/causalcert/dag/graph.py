"""
CausalDAG — core directed acyclic graph representation.

The :class:`CausalDAG` wraps a NumPy adjacency matrix and exposes efficient
operations for edge queries, neighbourhood look-ups, and structural mutations
(add / delete / reverse edge).  Acyclicity is enforced on construction.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Iterator

import numpy as np

from causalcert.types import AdjacencyMatrix, EdgeTuple, NodeId, NodeSet, StructuralEdit
from causalcert.exceptions import CyclicGraphError, InvalidEdgeError, NodeNotFoundError


class CausalDAG:
    """Mutable directed acyclic graph backed by a dense adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix | int
        Square binary matrix where ``adj[i, j] == 1`` iff *i → j*,
        or an integer *n* to create an empty DAG with *n* nodes.
    node_names : list[str] | None
        Optional human-readable names for nodes.
    validate : bool
        If ``True`` (default), check acyclicity on construction.

    Raises
    ------
    CyclicGraphError
        If the adjacency matrix contains a cycle and *validate* is ``True``.
    """

    def __init__(
        self,
        adj: AdjacencyMatrix | int,
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> None:
        if isinstance(adj, (int, np.integer)):
            adj = np.zeros((int(adj), int(adj)), dtype=np.int8)
        self._adj = np.asarray(adj, dtype=np.int8).copy()
        n = self._adj.shape[0]
        if self._adj.shape != (n, n):
            raise ValueError(f"Adjacency matrix must be square, got {self._adj.shape}")
        self._node_names = node_names or [str(i) for i in range(n)]
        if len(self._node_names) != n:
            raise ValueError(
                f"node_names length {len(self._node_names)} != matrix dim {n}"
            )
        if validate:
            from causalcert.dag.validation import is_dag, find_cycle

            if not is_dag(self._adj):
                cycle = find_cycle(self._adj)
                raise CyclicGraphError(cycle)

    # -- Constructors ---------------------------------------------------------

    @classmethod
    def empty(cls, n: int, node_names: list[str] | None = None) -> CausalDAG:
        """Create an empty DAG with *n* nodes and no edges."""
        return cls(np.zeros((n, n), dtype=np.int8), node_names, validate=False)

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: list[EdgeTuple],
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> CausalDAG:
        """Create a DAG from an edge list."""
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
        return cls(adj, node_names, validate=validate)

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: np.ndarray,
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> CausalDAG:
        """Create a DAG from a numpy adjacency matrix."""
        return cls(matrix, node_names, validate=validate)

    # -- Properties -----------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._adj.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return int(self._adj.sum())

    @property
    def adj(self) -> AdjacencyMatrix:
        """Read-only view of the adjacency matrix."""
        view = self._adj.view()
        view.flags.writeable = False
        return view

    @property
    def node_names(self) -> list[str]:
        """Human-readable node names."""
        return list(self._node_names)

    @property
    def density(self) -> float:
        """Edge density: n_edges / (n_nodes * (n_nodes - 1))."""
        n = self.n_nodes
        if n <= 1:
            return 0.0
        return self.n_edges / (n * (n - 1))

    @property
    def max_in_degree(self) -> int:
        """Maximum in-degree across all nodes."""
        if self.n_nodes == 0:
            return 0
        return int(self._adj.sum(axis=0).max())

    @property
    def max_out_degree(self) -> int:
        """Maximum out-degree across all nodes."""
        if self.n_nodes == 0:
            return 0
        return int(self._adj.sum(axis=1).max())

    @property
    def max_degree(self) -> int:
        """Maximum total degree (in + out) across all nodes."""
        if self.n_nodes == 0:
            return 0
        return int((self._adj.sum(axis=0) + self._adj.sum(axis=1)).max())

    # -- Node helpers ---------------------------------------------------------

    def _check_node(self, v: NodeId) -> None:
        if v < 0 or v >= self.n_nodes:
            raise NodeNotFoundError(v, self.n_nodes)

    def node_name(self, v: NodeId) -> str:
        """Return the human-readable name of node *v*."""
        self._check_node(v)
        return self._node_names[v]

    def node_id(self, name: str) -> NodeId:
        """Return the node id for the given *name*."""
        try:
            return self._node_names.index(name)
        except ValueError:
            raise KeyError(f"No node named {name!r}")

    def nodes(self) -> Iterator[NodeId]:
        """Iterate over all node ids."""
        return iter(range(self.n_nodes))

    # -- Degree queries -------------------------------------------------------

    def in_degree(self, v: NodeId) -> int:
        """Return the in-degree of node *v*."""
        self._check_node(v)
        return int(self._adj[:, v].sum())

    def out_degree(self, v: NodeId) -> int:
        """Return the out-degree of node *v*."""
        self._check_node(v)
        return int(self._adj[v, :].sum())

    def degree(self, v: NodeId) -> int:
        """Return total degree (in + out) of node *v*."""
        return self.in_degree(v) + self.out_degree(v)

    def roots(self) -> NodeSet:
        """Return the set of root nodes (in-degree 0)."""
        in_deg = self._adj.sum(axis=0)
        return frozenset(int(i) for i in np.where(in_deg == 0)[0])

    def leaves(self) -> NodeSet:
        """Return the set of leaf nodes (out-degree 0)."""
        out_deg = self._adj.sum(axis=1)
        return frozenset(int(i) for i in np.where(out_deg == 0)[0])

    # -- Edge queries ---------------------------------------------------------

    def has_edge(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if edge *u → v* exists."""
        return bool(self._adj[u, v])

    def parents(self, v: NodeId) -> NodeSet:
        """Return the parent set of node *v*."""
        self._check_node(v)
        return frozenset(int(i) for i in np.nonzero(self._adj[:, v])[0])

    def children(self, v: NodeId) -> NodeSet:
        """Return the child set of node *v*."""
        self._check_node(v)
        return frozenset(int(i) for i in np.nonzero(self._adj[v, :])[0])

    def neighbors(self, v: NodeId) -> NodeSet:
        """Return parents ∪ children of *v*."""
        return self.parents(v) | self.children(v)

    def ancestors(self, v: NodeId) -> NodeSet:
        """Return all ancestors of *v* (not including *v* itself)."""
        self._check_node(v)
        visited: set[int] = set()
        queue = deque(int(p) for p in np.nonzero(self._adj[:, v])[0])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for p in np.nonzero(self._adj[:, node])[0]:
                    p = int(p)
                    if p not in visited:
                        queue.append(p)
        return frozenset(visited)

    def descendants(self, v: NodeId) -> NodeSet:
        """Return all descendants of *v* (not including *v* itself)."""
        self._check_node(v)
        visited: set[int] = set()
        queue = deque(int(c) for c in np.nonzero(self._adj[v, :])[0])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for c in np.nonzero(self._adj[node, :])[0]:
                    c = int(c)
                    if c not in visited:
                        queue.append(c)
        return frozenset(visited)

    def is_ancestor(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is an ancestor of *v*."""
        return u in self.ancestors(v)

    def is_descendant(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is a descendant of *v*."""
        return u in self.descendants(v)

    def edges(self) -> Iterator[EdgeTuple]:
        """Iterate over all directed edges as ``(source, target)`` tuples."""
        rows, cols = np.nonzero(self._adj)
        for r, c in zip(rows, cols):
            yield (int(r), int(c))

    def edge_list(self) -> list[EdgeTuple]:
        """Return a list of all directed edges."""
        return list(self.edges())

    # -- Subgraph operations --------------------------------------------------

    def subgraph(self, nodes: NodeSet | set[int] | list[int]) -> CausalDAG:
        """Return the induced subgraph on the given *nodes*.

        The returned DAG has nodes re-indexed to 0..len(nodes)-1 in
        sorted order of the original indices.
        """
        node_list = sorted(nodes)
        idx = np.array(node_list, dtype=int)
        sub_adj = self._adj[np.ix_(idx, idx)]
        sub_names = [self._node_names[i] for i in node_list]
        return CausalDAG(sub_adj, sub_names, validate=False)

    def ancestral_subgraph(self, targets: NodeSet | set[int]) -> CausalDAG:
        """Return the subgraph induced by the ancestors of *targets* (inclusive)."""
        anc: set[int] = set(targets)
        queue = deque(targets)
        while queue:
            node = queue.popleft()
            for p in np.nonzero(self._adj[:, node])[0]:
                p = int(p)
                if p not in anc:
                    anc.add(p)
                    queue.append(p)
        return self.subgraph(anc)

    # -- Topological sort -----------------------------------------------------

    def topological_sort(self) -> list[NodeId]:
        """Return a topological ordering of nodes (Kahn's algorithm)."""
        n = self.n_nodes
        in_deg = self._adj.sum(axis=0).astype(int).copy()
        queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in np.nonzero(self._adj[node])[0]:
                child = int(child)
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if len(order) != n:
            raise CyclicGraphError()
        return order

    # -- Path queries ---------------------------------------------------------

    def has_directed_path(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if a directed path from *u* to *v* exists."""
        if u == v:
            return True
        visited: set[int] = set()
        queue = deque([u])
        while queue:
            node = queue.popleft()
            for child in np.nonzero(self._adj[node])[0]:
                child = int(child)
                if child == v:
                    return True
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return False

    # -- Mutations (checked) --------------------------------------------------

    def add_edge(self, u: NodeId, v: NodeId) -> None:
        """Add edge *u → v*, raising if it would create a cycle.

        Raises
        ------
        InvalidEdgeError
            If the edge already exists or would create a cycle.
        """
        self._check_node(u)
        self._check_node(v)
        if u == v:
            raise InvalidEdgeError((u, v), "self-loops not allowed")
        if self.has_edge(u, v):
            raise InvalidEdgeError((u, v), "edge already exists")
        # Fast cycle check: would adding u->v create a cycle?
        # A cycle exists iff v can already reach u via directed paths.
        if self.has_directed_path(v, u):
            raise InvalidEdgeError((u, v), "would create a cycle")
        self._adj[u, v] = 1

    def delete_edge(self, u: NodeId, v: NodeId) -> None:
        """Delete edge *u → v*.

        Raises
        ------
        InvalidEdgeError
            If the edge does not exist.
        """
        self._check_node(u)
        self._check_node(v)
        if not self.has_edge(u, v):
            raise InvalidEdgeError((u, v), "edge does not exist")
        self._adj[u, v] = 0

    def reverse_edge(self, u: NodeId, v: NodeId) -> None:
        """Reverse edge *u → v* to *v → u*.

        Raises
        ------
        InvalidEdgeError
            If the original edge does not exist or the reversal creates a cycle.
        """
        self.delete_edge(u, v)
        try:
            self.add_edge(v, u)
        except InvalidEdgeError:
            self._adj[u, v] = 1  # rollback
            raise InvalidEdgeError((u, v), "reversal would create a cycle")

    def apply_edit(self, edit: StructuralEdit) -> None:
        """Apply a :class:`StructuralEdit` to this DAG in place."""
        from causalcert.types import EditType

        if edit.edit_type == EditType.ADD:
            self.add_edge(edit.source, edit.target)
        elif edit.edit_type == EditType.DELETE:
            self.delete_edge(edit.source, edit.target)
        elif edit.edit_type == EditType.REVERSE:
            self.reverse_edge(edit.source, edit.target)

    # -- Copy / deepcopy ------------------------------------------------------

    def copy(self) -> CausalDAG:
        """Return a deep copy of this DAG."""
        return CausalDAG(self._adj.copy(), list(self._node_names), validate=False)

    def __copy__(self) -> CausalDAG:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> CausalDAG:
        result = CausalDAG(
            copy.deepcopy(self._adj, memo),
            copy.deepcopy(self._node_names, memo),
            validate=False,
        )
        memo[id(self)] = result
        return result

    # -- Dunder ---------------------------------------------------------------

    def __repr__(self) -> str:
        return f"CausalDAG(n_nodes={self.n_nodes}, n_edges={self.n_edges})"

    def __str__(self) -> str:
        lines = [f"CausalDAG with {self.n_nodes} nodes and {self.n_edges} edges:"]
        for u, v in self.edges():
            lines.append(f"  {self._node_names[u]} -> {self._node_names[v]}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalDAG):
            return NotImplemented
        return np.array_equal(self._adj, other._adj)

    def __hash__(self) -> int:
        return hash(self._adj.tobytes())

    def __len__(self) -> int:
        """Return the number of nodes."""
        return self.n_nodes

    def __contains__(self, edge: EdgeTuple) -> bool:
        """Check if edge ``(u, v)`` is in the DAG."""
        u, v = edge
        if u < 0 or u >= self.n_nodes or v < 0 or v >= self.n_nodes:
            return False
        return bool(self._adj[u, v])

    def __iter__(self) -> Iterator[NodeId]:
        """Iterate over node ids."""
        return iter(range(self.n_nodes))

    # -- Serialization helpers ------------------------------------------------

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return a copy of the adjacency matrix."""
        return self._adj.copy()

    def to_edge_list(self) -> list[EdgeTuple]:
        """Return a list of ``(source, target)`` tuples."""
        return list(self.edges())

    # -- Degree statistics ----------------------------------------------------

    def degree_sequence(self) -> np.ndarray:
        """Return array of total degrees for all nodes."""
        return (self._adj.sum(axis=0) + self._adj.sum(axis=1)).astype(int)

    def in_degree_sequence(self) -> np.ndarray:
        """Return array of in-degrees for all nodes."""
        return self._adj.sum(axis=0).astype(int)

    def out_degree_sequence(self) -> np.ndarray:
        """Return array of out-degrees for all nodes."""
        return self._adj.sum(axis=1).astype(int)

    def mean_degree(self) -> float:
        """Return the average total degree."""
        if self.n_nodes == 0:
            return 0.0
        return float(self.degree_sequence().mean())

    # -- Structural queries ---------------------------------------------------

    def is_connected(self) -> bool:
        """Return ``True`` if the underlying undirected graph is connected."""
        if self.n_nodes <= 1:
            return True
        undirected = self._adj | self._adj.T
        visited: set[int] = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            node = queue.popleft()
            for nb in np.nonzero(undirected[node])[0]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == self.n_nodes

    def connected_components(self) -> list[NodeSet]:
        """Return connected components of the underlying undirected graph."""
        undirected = self._adj | self._adj.T
        visited: set[int] = set()
        components: list[NodeSet] = []
        for start in range(self.n_nodes):
            if start in visited:
                continue
            comp: set[int] = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in comp:
                    continue
                comp.add(node)
                visited.add(node)
                for nb in np.nonzero(undirected[node])[0]:
                    nb = int(nb)
                    if nb not in comp:
                        queue.append(nb)
            components.append(frozenset(comp))
        return components

    def is_complete(self) -> bool:
        """Return ``True`` if the DAG is a complete DAG (tournament)."""
        n = self.n_nodes
        if n <= 1:
            return True
        # A complete DAG on n nodes has n*(n-1)/2 edges
        return self.n_edges == n * (n - 1) // 2 and np.all(
            (self._adj + self._adj.T + np.eye(n, dtype=np.int8)) > 0
        )
