"""
Sparse graph operations for large causal DAGs.

Provides :class:`SparseDAG` backed by SciPy CSR sparse matrices for
memory-efficient storage and fast neighbour enumeration on graphs with
many nodes but few edges.  Also provides sparse topological sort, sparse
transitive closure, and conversion utilities.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from causalcert.types import AdjacencyMatrix, EditType, EdgeTuple, NodeId, NodeSet, StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SparseDAG
# ---------------------------------------------------------------------------


class SparseDAG:
    """Directed acyclic graph backed by a SciPy CSR sparse matrix.

    Designed for large graphs (hundreds to thousands of nodes) where the
    adjacency matrix is mostly zeros.  Provides efficient neighbour
    enumeration, topological sort, and transitive closure computation.

    Parameters
    ----------
    adj_sparse : csr_matrix
        Sparse binary adjacency matrix of shape ``(n, n)``
        where entry ``(i, j)`` is 1 iff edge *i → j* exists.
    node_names : list[str] | None
        Optional human-readable names for nodes.
    validate : bool
        If ``True`` (default), check acyclicity.
    """

    def __init__(
        self,
        adj_sparse: csr_matrix,
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> None:
        if not sparse.issparse(adj_sparse):
            raise TypeError(f"Expected sparse matrix, got {type(adj_sparse)}")
        self._csr: csr_matrix = csr_matrix(adj_sparse, dtype=np.int8)
        n = self._csr.shape[0]
        if self._csr.shape != (n, n):
            raise ValueError(
                f"Adjacency matrix must be square, got {self._csr.shape}"
            )
        self._n = n
        self._node_names = node_names or [str(i) for i in range(n)]
        if len(self._node_names) != n:
            raise ValueError(
                f"node_names length {len(self._node_names)} != n={n}"
            )
        # CSC for efficient column (parent) access
        self._csc: csc_matrix = self._csr.tocsc()

        if validate and not self._is_dag():
            raise ValueError("Graph contains a cycle.")

    # -- constructors ---

    @classmethod
    def from_dense(
        cls,
        adj: AdjacencyMatrix,
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> SparseDAG:
        """Create a :class:`SparseDAG` from a dense adjacency matrix.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Dense binary adjacency matrix.
        node_names : list[str] | None
            Optional node names.
        validate : bool
            Check acyclicity.

        Returns
        -------
        SparseDAG
        """
        adj = np.asarray(adj, dtype=np.int8)
        return cls(csr_matrix(adj), node_names, validate=validate)

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: Sequence[EdgeTuple],
        node_names: list[str] | None = None,
        validate: bool = True,
    ) -> SparseDAG:
        """Create a :class:`SparseDAG` from an edge list.

        Parameters
        ----------
        n : int
            Number of nodes.
        edges : Sequence[EdgeTuple]
            List of ``(source, target)`` pairs.
        node_names : list[str] | None
            Optional node names.
        validate : bool
            Check acyclicity.

        Returns
        -------
        SparseDAG
        """
        if not edges:
            return cls(csr_matrix((n, n), dtype=np.int8), node_names, validate=False)
        rows, cols = zip(*edges)
        data = np.ones(len(edges), dtype=np.int8)
        mat = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)
        return cls(mat, node_names, validate=validate)

    @classmethod
    def empty(cls, n: int, node_names: list[str] | None = None) -> SparseDAG:
        """Create an empty DAG with *n* nodes."""
        return cls(csr_matrix((n, n), dtype=np.int8), node_names, validate=False)

    # -- properties ---

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._n

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return int(self._csr.nnz)

    @property
    def density(self) -> float:
        """Edge density: ``n_edges / (n * (n-1))``."""
        max_edges = self._n * (self._n - 1)
        if max_edges == 0:
            return 0.0
        return self.n_edges / max_edges

    @property
    def node_names(self) -> list[str]:
        """Node name list."""
        return list(self._node_names)

    @property
    def csr(self) -> csr_matrix:
        """Return the CSR sparse adjacency matrix (read-only copy)."""
        return self._csr.copy()

    # -- edge queries ---

    def has_edge(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if edge *u → v* exists."""
        return bool(self._csr[u, v])

    def edges(self) -> list[EdgeTuple]:
        """Return the edge list as ``(source, target)`` tuples."""
        coo = self._csr.tocoo()
        return list(zip(coo.row.tolist(), coo.col.tolist()))

    # -- neighbour enumeration ---

    def children(self, v: NodeId) -> list[int]:
        """Return children of *v* (nodes that *v* points to).

        Parameters
        ----------
        v : NodeId
            Query node.

        Returns
        -------
        list[int]
            Sorted list of child indices.
        """
        return self._csr[v].indices.tolist()

    def parents(self, v: NodeId) -> list[int]:
        """Return parents of *v* (nodes that point to *v*).

        Parameters
        ----------
        v : NodeId
            Query node.

        Returns
        -------
        list[int]
            Sorted list of parent indices.
        """
        return self._csc[:, v].indices.tolist()

    def in_degree(self, v: NodeId) -> int:
        """Return the in-degree of *v*."""
        return len(self.parents(v))

    def out_degree(self, v: NodeId) -> int:
        """Return the out-degree of *v*."""
        return len(self.children(v))

    def in_degrees(self) -> NDArray[np.int64]:
        """Return in-degree array for all nodes."""
        return np.asarray(self._csr.sum(axis=0)).ravel().astype(np.int64)

    def out_degrees(self) -> NDArray[np.int64]:
        """Return out-degree array for all nodes."""
        return np.asarray(self._csr.sum(axis=1)).ravel().astype(np.int64)

    def neighbours(self, v: NodeId) -> NodeSet:
        """Return the union of parents and children of *v*."""
        return frozenset(self.parents(v) + self.children(v))

    # -- topological sort ---

    def topological_sort(self) -> list[int]:
        """Return a topological ordering of nodes via Kahn's algorithm.

        Returns
        -------
        list[int]
            Topological order. Empty if the graph has a cycle.
        """
        in_deg = self.in_degrees().copy()
        queue = deque(int(i) for i in range(self._n) if in_deg[i] == 0)
        order: list[int] = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for c in self.children(v):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return order

    # -- acyclicity ---

    def _is_dag(self) -> bool:
        """Check acyclicity via topological sort."""
        return len(self.topological_sort()) == self._n

    # -- ancestor / descendant ---

    def ancestors(self, v: NodeId) -> NodeSet:
        """Return all ancestors of *v* (including *v*) via reverse BFS.

        Parameters
        ----------
        v : NodeId
            Query node.

        Returns
        -------
        NodeSet
        """
        visited: set[int] = {v}
        queue: deque[int] = deque([v])
        while queue:
            node = queue.popleft()
            for p in self.parents(node):
                if p not in visited:
                    visited.add(p)
                    queue.append(p)
        return frozenset(visited)

    def descendants(self, v: NodeId) -> NodeSet:
        """Return all descendants of *v* (including *v*) via forward BFS.

        Parameters
        ----------
        v : NodeId
            Query node.

        Returns
        -------
        NodeSet
        """
        visited: set[int] = {v}
        queue: deque[int] = deque([v])
        while queue:
            node = queue.popleft()
            for c in self.children(node):
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
        return frozenset(visited)

    def ancestors_all(self) -> list[NodeSet]:
        """Compute ancestor sets for every node.

        Uses topological order for efficient propagation.

        Returns
        -------
        list[NodeSet]
            ``result[v]`` is the ancestor set of node *v* (including *v*).
        """
        topo = self.topological_sort()
        anc: list[set[int]] = [set() for _ in range(self._n)]
        for v in topo:
            anc[v].add(v)
        for v in topo:
            for c in self.children(v):
                anc[c].update(anc[v])
        return [frozenset(s) for s in anc]

    def descendants_all(self) -> list[NodeSet]:
        """Compute descendant sets for every node.

        Returns
        -------
        list[NodeSet]
            ``result[v]`` is the descendant set of node *v* (including *v*).
        """
        topo = self.topological_sort()
        desc: list[set[int]] = [set() for _ in range(self._n)]
        for v in topo:
            desc[v].add(v)
        for v in reversed(topo):
            for p in self.parents(v):
                desc[p].update(desc[v])
        return [frozenset(s) for s in desc]

    # -- transitive closure ---

    def transitive_closure(self) -> csr_matrix:
        """Compute the transitive closure as a sparse matrix.

        Entry ``(i, j) == 1`` iff there exists a directed path from *i* to *j*.
        Uses iterative sparse matrix squaring with ``O(log V)`` iterations.

        Returns
        -------
        csr_matrix
            Sparse binary transitive closure matrix.
        """
        n = self._n
        # Start with adjacency + identity (reflexive transitive closure)
        tc = self._csr.astype(np.float64) + sparse.eye(n, dtype=np.float64)

        # Repeated squaring until convergence
        for _ in range(int(np.ceil(np.log2(max(n, 2))))):
            tc_new = tc @ tc
            # Binarise: any positive value → 1
            tc_new.data[:] = 1.0
            tc_new.eliminate_zeros()
            if (tc_new - tc).nnz == 0:
                break
            tc = tc_new

        # Remove self-loops unless already present in original
        tc = tc.astype(np.int8)
        return tc

    def reachability_matrix(self) -> csr_matrix:
        """Return the reachability matrix (transitive closure without self-loops).

        Entry ``(i, j) == 1`` iff there exists a *non-trivial* directed path
        from *i* to *j*.

        Returns
        -------
        csr_matrix
        """
        tc = self.transitive_closure()
        tc.setdiag(0)
        tc.eliminate_zeros()
        return tc

    # -- mutation ---

    def add_edge(self, u: NodeId, v: NodeId, validate: bool = True) -> SparseDAG:
        """Return a new DAG with edge *u → v* added.

        Parameters
        ----------
        u, v : NodeId
            Source and target of the new edge.
        validate : bool
            Check that the result is still acyclic.

        Returns
        -------
        SparseDAG
        """
        lil = self._csr.tolil()
        lil[u, v] = 1
        return SparseDAG(lil.tocsr(), list(self._node_names), validate=validate)

    def remove_edge(self, u: NodeId, v: NodeId) -> SparseDAG:
        """Return a new DAG with edge *u → v* removed.

        Parameters
        ----------
        u, v : NodeId
            Source and target of the edge to remove.

        Returns
        -------
        SparseDAG
        """
        lil = self._csr.tolil()
        lil[u, v] = 0
        return SparseDAG(lil.tocsr(), list(self._node_names), validate=False)

    def reverse_edge(self, u: NodeId, v: NodeId, validate: bool = True) -> SparseDAG:
        """Return a new DAG with edge *u → v* reversed to *v → u*.

        Parameters
        ----------
        u, v : NodeId
            Source and target of the edge to reverse.
        validate : bool
            Check acyclicity of the result.

        Returns
        -------
        SparseDAG
        """
        lil = self._csr.tolil()
        lil[u, v] = 0
        lil[v, u] = 1
        return SparseDAG(lil.tocsr(), list(self._node_names), validate=validate)

    def apply_edit(self, edit: StructuralEdit, validate: bool = True) -> SparseDAG:
        """Apply a :class:`StructuralEdit` and return the new DAG.

        Parameters
        ----------
        edit : StructuralEdit
            The edge edit.
        validate : bool
            Check acyclicity after the edit.

        Returns
        -------
        SparseDAG
        """
        if edit.edit_type == EditType.ADD:
            return self.add_edge(edit.source, edit.target, validate=validate)
        if edit.edit_type == EditType.DELETE:
            return self.remove_edge(edit.source, edit.target)
        if edit.edit_type == EditType.REVERSE:
            return self.reverse_edge(edit.source, edit.target, validate=validate)
        raise ValueError(f"Unknown edit type: {edit.edit_type}")

    # -- conversion ---

    def to_dense(self) -> AdjacencyMatrix:
        """Convert to a dense NumPy adjacency matrix.

        Returns
        -------
        AdjacencyMatrix
        """
        return np.asarray(self._csr.todense(), dtype=np.int8)

    @classmethod
    def from_causal_dag(cls, dag: object) -> SparseDAG:
        """Create a :class:`SparseDAG` from a :class:`CausalDAG` instance.

        Parameters
        ----------
        dag : CausalDAG
            A dense DAG instance (must have ``adj`` property or ``_adj`` attribute).

        Returns
        -------
        SparseDAG
        """
        # Support the CausalDAG interface
        adj: np.ndarray
        if hasattr(dag, "adj"):
            adj = np.asarray(dag.adj, dtype=np.int8)  # type: ignore[union-attr]
        elif hasattr(dag, "_adj"):
            adj = np.asarray(dag._adj, dtype=np.int8)  # type: ignore[union-attr]
        else:
            raise TypeError("Cannot extract adjacency matrix from dag object.")

        names: list[str] | None = None
        if hasattr(dag, "node_names"):
            names = list(dag.node_names)  # type: ignore[union-attr]
        elif hasattr(dag, "_node_names"):
            names = list(dag._node_names)  # type: ignore[union-attr]

        return cls.from_dense(adj, names, validate=False)

    # -- memory info ---

    def memory_bytes(self) -> int:
        """Approximate memory usage of the sparse representation in bytes."""
        return (
            self._csr.data.nbytes
            + self._csr.indices.nbytes
            + self._csr.indptr.nbytes
            + self._csc.data.nbytes
            + self._csc.indices.nbytes
            + self._csc.indptr.nbytes
        )

    def dense_memory_bytes(self) -> int:
        """Memory that an equivalent dense matrix would require."""
        return self._n * self._n  # int8

    def compression_ratio(self) -> float:
        """Ratio of dense memory to sparse memory (higher is better)."""
        dm = self.dense_memory_bytes()
        sm = self.memory_bytes()
        return dm / sm if sm > 0 else float("inf")

    # -- repr ---

    def __repr__(self) -> str:
        return (
            f"SparseDAG(n_nodes={self._n}, n_edges={self.n_edges}, "
            f"density={self.density:.4f}, "
            f"memory={self.memory_bytes()} bytes)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseDAG):
            return NotImplemented
        if self._n != other._n:
            return False
        return (self._csr != other._csr).nnz == 0
