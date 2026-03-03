"""Sparse matrix operations for large DAGs.

Memory-efficient DAG representation using sparse adjacency storage
with standard graph queries (parents, children, topological sort),
structural Hamming distance, and sparse alignment matrices.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


# ---------------------------------------------------------------------------
# SparseDAG
# ---------------------------------------------------------------------------

class SparseDAG:
    """Sparse adjacency representation of a directed acyclic graph.

    Uses ``scipy.sparse.lil_matrix`` for efficient incremental edge
    insertion / removal and ``csr_matrix`` for fast row / column
    queries.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    """

    def __init__(self, n_nodes: int) -> None:
        self._n_nodes = n_nodes
        self._adj = sparse.lil_matrix(
            (n_nodes, n_nodes), dtype=np.int8
        )
        self._edge_count = 0

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_dense(cls, adj_matrix: NDArray) -> "SparseDAG":
        """Construct a :class:`SparseDAG` from a dense adjacency matrix."""
        adj_matrix = np.asarray(adj_matrix)
        n = adj_matrix.shape[0]
        dag = cls(n)
        dag._adj = sparse.lil_matrix(
            (adj_matrix != 0).astype(np.int8)
        )
        dag._edge_count = int(dag._adj.nnz)
        return dag

    @classmethod
    def from_adjacency_dict(
        cls, adj_dict: Dict[int, List[int]], n_nodes: int
    ) -> "SparseDAG":
        """Construct from a dictionary ``{parent: [children, ...]}``."""
        dag = cls(n_nodes)
        for parent, children in adj_dict.items():
            for child in children:
                dag.add_edge(parent, child)
        return dag

    # -- conversion --------------------------------------------------------

    def to_dense(self) -> NDArray:
        """Return the dense ``(n, n)`` adjacency matrix."""
        return np.asarray(self._adj.toarray(), dtype=np.float64)

    def to_csr(self) -> sparse.csr_matrix:
        """Return a CSR copy for efficient arithmetic."""
        return self._adj.tocsr()

    # -- edge operations ---------------------------------------------------

    def add_edge(self, i: int, j: int) -> None:
        """Add directed edge *i* → *j*."""
        self._check_idx(i)
        self._check_idx(j)
        if self._adj[i, j] == 0:
            self._adj[i, j] = 1
            self._edge_count += 1

    def remove_edge(self, i: int, j: int) -> None:
        """Remove directed edge *i* → *j*."""
        self._check_idx(i)
        self._check_idx(j)
        if self._adj[i, j] != 0:
            self._adj[i, j] = 0
            self._edge_count -= 1

    def has_edge(self, i: int, j: int) -> bool:
        """Return ``True`` if edge *i* → *j* exists."""
        self._check_idx(i)
        self._check_idx(j)
        return bool(self._adj[i, j] != 0)

    # -- neighbourhood queries ---------------------------------------------

    def parents(self, node: int) -> FrozenSet[int]:
        """Return the parent set of *node* as a frozenset."""
        self._check_idx(node)
        col = self._adj.tocsc()[:, node].toarray().ravel()
        return frozenset(int(i) for i in np.nonzero(col)[0])

    def children(self, node: int) -> Set[int]:
        """Return the children of *node*."""
        self._check_idx(node)
        row = self._adj[node, :].toarray().ravel()
        return set(int(j) for j in np.nonzero(row)[0])

    def ancestors(self, node: int) -> Set[int]:
        """Return all ancestors of *node* (transitive)."""
        visited: Set[int] = set()
        stack = list(self.parents(node))
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(self.parents(v) - visited)
        return visited

    def descendants(self, node: int) -> Set[int]:
        """Return all descendants of *node* (transitive)."""
        visited: Set[int] = set()
        stack = list(self.children(node))
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(self.children(v) - visited)
        return visited

    def markov_blanket(self, node: int) -> Set[int]:
        """Return the Markov blanket of *node*."""
        pa = self.parents(node)
        ch = self.children(node)
        co_parents: Set[int] = set()
        for c in ch:
            co_parents.update(self.parents(c))
        mb = set(pa) | ch | co_parents
        mb.discard(node)
        return mb

    # -- topological sort (Kahn's algorithm) --------------------------------

    def topological_sort(self) -> List[int]:
        """Return a topological ordering via Kahn's algorithm.

        Raises ``ValueError`` if the graph contains a cycle.
        """
        n = self._n_nodes
        csr = self._adj.tocsr()
        in_deg = np.zeros(n, dtype=np.int64)
        for j in range(n):
            in_deg[j] = len(self.parents(j))

        queue: deque[int] = deque()
        for i in range(n):
            if in_deg[i] == 0:
                queue.append(i)

        order: List[int] = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self.children(v):
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)

        if len(order) != n:
            raise ValueError("Graph contains a cycle")
        return order

    # -- acyclicity check ---------------------------------------------------

    def is_acyclic(self) -> bool:
        """Return ``True`` if the graph contains no directed cycles."""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def _dfs_cycle_check(self, start: int, target: int) -> bool:
        """DFS from *start* checking if *target* is reachable."""
        visited: Set[int] = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v == target:
                return True
            if v in visited:
                continue
            visited.add(v)
            stack.extend(self.children(v))
        return False

    # -- structural comparison ---------------------------------------------

    def structural_hamming_distance(self, other: "SparseDAG") -> int:
        """Structural Hamming distance to *other* (for equal-size DAGs)."""
        n = min(self._n_nodes, other._n_nodes)
        a1 = self._adj.tocsr()[:n, :n]
        a2 = other._adj.tocsr()[:n, :n]
        diff = a1 - a2
        # SHD counts additions, deletions, and reversals
        shd = 0
        diff_dense = diff.toarray()
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = diff_dense[i, j]
                d_ji = diff_dense[j, i]
                if d_ij != 0 or d_ji != 0:
                    if d_ij == 1 and d_ji == -1:
                        shd += 1  # reversal
                    elif d_ij == -1 and d_ji == 1:
                        shd += 1  # reversal
                    else:
                        shd += abs(d_ij) + abs(d_ji)
        return shd

    def edge_set(self) -> Set[Tuple[int, int]]:
        """Return the set of all directed edges ``(i, j)``."""
        coo = self._adj.tocoo()
        return set(zip(coo.row.tolist(), coo.col.tolist()))

    # -- metrics -----------------------------------------------------------

    def n_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self._edge_count

    @property
    def num_edges(self) -> int:
        return self._edge_count

    @property
    def num_nodes(self) -> int:
        return self._n_nodes

    def density(self) -> float:
        """Edge density in [0, 1]."""
        max_edges = self._n_nodes * (self._n_nodes - 1)
        if max_edges == 0:
            return 0.0
        return self._edge_count / max_edges

    def in_degree(self, node: Optional[int] = None) -> int:
        """Return in-degree of *node* (or array for all)."""
        if node is not None:
            return len(self.parents(node))
        return sum(len(self.parents(i)) for i in range(self._n_nodes))

    def out_degree(self, node: Optional[int] = None) -> int:
        """Return out-degree of *node* (or total for all)."""
        if node is not None:
            return len(self.children(node))
        return sum(len(self.children(i)) for i in range(self._n_nodes))

    # -- validation --------------------------------------------------------

    def _check_idx(self, i: int) -> None:
        if i < 0 or i >= self._n_nodes:
            raise IndexError(
                f"Node {i} out of range [0, {self._n_nodes})"
            )

    def __repr__(self) -> str:
        return (
            f"SparseDAG(n_nodes={self._n_nodes}, "
            f"n_edges={self._edge_count})"
        )


# ---------------------------------------------------------------------------
# SparseAlignmentMatrix
# ---------------------------------------------------------------------------

class SparseAlignmentMatrix:
    """Sparse representation of an ``n × m`` alignment matrix.

    Used for variable alignment between two DAGs where most entries
    are zero.

    Parameters
    ----------
    n, m : int
        Dimensions of the alignment matrix.
    """

    def __init__(self, n: int, m: int) -> None:
        self._n = n
        self._m = m
        self._mat = sparse.lil_matrix((n, m), dtype=np.float64)

    def set(self, i: int, j: int, value: float) -> None:
        """Set entry ``(i, j)`` to *value*."""
        self._mat[i, j] = value

    def get(self, i: int, j: int) -> float:
        """Get entry ``(i, j)``."""
        return float(self._mat[i, j])

    def row_max(self, i: int) -> Tuple[int, float]:
        """Index and value of the maximum in row *i*."""
        row = self._mat[i, :].toarray().ravel()
        j = int(np.argmax(row))
        return j, float(row[j])

    def col_max(self, j: int) -> Tuple[int, float]:
        """Index and value of the maximum in column *j*."""
        col = self._mat.tocsc()[:, j].toarray().ravel()
        i = int(np.argmax(col))
        return i, float(col[i])

    def to_dense(self) -> NDArray:
        """Convert to a dense ``(n, m)`` array."""
        return np.asarray(self._mat.toarray(), dtype=np.float64)

    def from_dense(self, arr: NDArray) -> None:
        """Populate from a dense array."""
        self._mat = sparse.lil_matrix(
            np.asarray(arr, dtype=np.float64)
        )

    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return self._mat.nnz

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._n, self._m)

    def __repr__(self) -> str:
        return (
            f"SparseAlignmentMatrix(n={self._n}, m={self._m}, "
            f"nnz={self.nnz})"
        )


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def sparse_dag_operations(dag: SparseDAG) -> Dict:
    """Return summary statistics for *dag* (nodes, edges, density).

    Parameters
    ----------
    dag : SparseDAG

    Returns
    -------
    dict with keys ``n_nodes``, ``n_edges``, ``density``,
    ``is_acyclic``, ``max_in_degree``, ``max_out_degree``.
    """
    n = dag.num_nodes
    in_degrees = [len(dag.parents(i)) for i in range(n)]
    out_degrees = [len(dag.children(i)) for i in range(n)]
    return {
        "n_nodes": n,
        "n_edges": dag.n_edges(),
        "density": dag.density(),
        "is_acyclic": dag.is_acyclic(),
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
    }
