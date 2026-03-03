"""CPDAG construction and operations.

Implements Completed Partially Directed Acyclic Graphs using
Chickering's transformational characterization.  Supports conversion
from/to DAG adjacency matrices, Meek rule application, structural
Hamming distance, and MEC size computation.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _topological_sort(adj: NDArray[np.int_]) -> List[int]:
    """Kahn's algorithm for topological sort.  Raises on cycles."""
    n = adj.shape[0]
    in_deg = np.sum(adj, axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order: List[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for j in range(n):
            if adj[v, j]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    if len(order) != n:
        raise ValueError("Graph contains a cycle; not a DAG")
    return order


def _is_dag(adj: NDArray[np.int_]) -> bool:
    """Check if adjacency matrix represents a DAG."""
    try:
        _topological_sort(adj)
        return True
    except ValueError:
        return False


# -------------------------------------------------------------------
# CPDAG class
# -------------------------------------------------------------------

class CPDAG:
    """Completed Partially Directed Acyclic Graph.

    A CPDAG represents a Markov Equivalence Class (MEC) of DAGs.
    It contains both directed edges (compelled) and undirected edges
    (reversible).

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    directed_edges : set[tuple[int, int]] or None
        Initial directed edges (i -> j).
    undirected_edges : set[tuple[int, int]] or None
        Initial undirected edges as canonical (min, max) pairs.
    """

    def __init__(
        self,
        n_nodes: int,
        directed_edges: set[tuple[int, int]] | None = None,
        undirected_edges: set[tuple[int, int]] | None = None,
    ) -> None:
        if n_nodes < 0:
            raise ValueError("n_nodes must be non-negative")
        self.n_nodes = n_nodes
        self.directed_edges: set[tuple[int, int]] = set(directed_edges or set())
        self.undirected_edges: set[tuple[int, int]] = set()
        # Canonicalise undirected edges to (min, max)
        for e in (undirected_edges or set()):
            self.undirected_edges.add((min(e), max(e)))

    # -----------------------------------------------------------------
    # Construction methods
    # -----------------------------------------------------------------

    @classmethod
    def from_adjacency_matrix(cls, adj: NDArray[np.int_]) -> CPDAG:
        """Construct a CPDAG from an adjacency matrix.

        In the matrix, adj[i,j]=1 and adj[j,i]=0 means i -> j (directed).
        adj[i,j]=1 and adj[j,i]=1 means i - j (undirected).
        """
        adj = np.asarray(adj, dtype=np.int_)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got {adj.shape}")
        n = adj.shape[0]
        directed: set[tuple[int, int]] = set()
        undirected: set[tuple[int, int]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                ij = bool(adj[i, j])
                ji = bool(adj[j, i])
                if ij and ji:
                    undirected.add((i, j))
                else:
                    if ij:
                        directed.add((i, j))
                    if ji:
                        directed.add((j, i))
        return cls(n, directed, undirected)

    @classmethod
    def from_dag(cls, dag_adj: NDArray[np.int_]) -> CPDAG:
        """Convert a DAG adjacency matrix to its CPDAG via Chickering's algorithm.

        Steps:
        1. Identify all v-structures.
        2. Label edges as compelled or reversible.
        3. Apply Meek rules R1-R4 to propagate orientations.
        """
        dag_adj = np.asarray(dag_adj, dtype=np.int_)
        dag_adj = (dag_adj != 0).astype(np.int_)
        if not _is_dag(dag_adj):
            raise ValueError("Input is not a valid DAG")
        n = dag_adj.shape[0]

        # Step 1: Find v-structures: a -> b <- c where a and c are not adjacent
        v_struct_edges: Set[Tuple[int, int]] = set()
        for b in range(n):
            parents_b = [i for i in range(n) if dag_adj[i, b]]
            for idx_a in range(len(parents_b)):
                for idx_c in range(idx_a + 1, len(parents_b)):
                    a = parents_b[idx_a]
                    c = parents_b[idx_c]
                    if not dag_adj[a, c] and not dag_adj[c, a]:
                        v_struct_edges.add((a, b))
                        v_struct_edges.add((c, b))

        # Step 2: Build CPDAG with v-structure edges directed, rest undirected
        directed: set[tuple[int, int]] = set(v_struct_edges)
        undirected: set[tuple[int, int]] = set()
        for i in range(n):
            for j in range(n):
                if dag_adj[i, j]:
                    if (i, j) not in directed:
                        canon = (min(i, j), max(i, j))
                        undirected.add(canon)

        cpdag = cls(n, directed, undirected)

        # Step 3: Apply Meek rules
        cpdag._apply_meek_rules()
        return cpdag

    @classmethod
    def from_pdag(cls, pdag_adj: NDArray[np.int_]) -> CPDAG:
        """Convert a PDAG adjacency matrix to a CPDAG by completing orientation.

        Uses the same from_adjacency_matrix followed by Meek rules.
        """
        cpdag = cls.from_adjacency_matrix(pdag_adj)
        cpdag._apply_meek_rules()
        return cpdag

    def to_adjacency_matrix(self) -> NDArray[np.int_]:
        """Return the adjacency-matrix representation.

        Directed i -> j: adj[i,j] = 1, adj[j,i] = 0.
        Undirected i - j: adj[i,j] = 1, adj[j,i] = 1.
        """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int_)
        for i, j in self.directed_edges:
            adj[i, j] = 1
        for i, j in self.undirected_edges:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    # -----------------------------------------------------------------
    # Edge operations
    # -----------------------------------------------------------------

    def has_directed_edge(self, i: int, j: int) -> bool:
        """Check if directed edge i -> j exists."""
        return (i, j) in self.directed_edges

    def has_undirected_edge(self, i: int, j: int) -> bool:
        """Check if undirected edge i - j exists."""
        return (min(i, j), max(i, j)) in self.undirected_edges

    def add_directed_edge(self, i: int, j: int) -> None:
        """Add directed edge i -> j."""
        self._check_idx(i)
        self._check_idx(j)
        if i == j:
            raise ValueError("Self-loops are not allowed")
        self.directed_edges.add((i, j))

    def remove_directed_edge(self, i: int, j: int) -> None:
        """Remove directed edge i -> j."""
        self.directed_edges.discard((i, j))

    def add_undirected_edge(self, i: int, j: int) -> None:
        """Add undirected edge i - j."""
        self._check_idx(i)
        self._check_idx(j)
        if i == j:
            raise ValueError("Self-loops are not allowed")
        self.undirected_edges.add((min(i, j), max(i, j)))

    def remove_undirected_edge(self, i: int, j: int) -> None:
        """Remove undirected edge i - j."""
        self.undirected_edges.discard((min(i, j), max(i, j)))

    def _check_idx(self, i: int) -> None:
        if i < 0 or i >= self.n_nodes:
            raise ValueError(f"Node index {i} out of range [0, {self.n_nodes})")

    # -----------------------------------------------------------------
    # Adjacency queries
    # -----------------------------------------------------------------

    def neighbors(self, node: int) -> Set[int]:
        """Return nodes adjacent via an undirected edge."""
        self._check_idx(node)
        result: Set[int] = set()
        for i, j in self.undirected_edges:
            if i == node:
                result.add(j)
            elif j == node:
                result.add(i)
        return result

    def adjacent(self, node: int) -> Set[int]:
        """All nodes adjacent to *node* (directed or undirected)."""
        self._check_idx(node)
        result = self.neighbors(node)
        result |= self.children(node)
        result |= self.parents(node)
        return result

    def children(self, node: int) -> Set[int]:
        """Return children (directed successors) of *node*."""
        self._check_idx(node)
        return {j for i, j in self.directed_edges if i == node}

    def parents(self, node: int) -> Set[int]:
        """Return parents (directed predecessors) of *node*."""
        self._check_idx(node)
        return {i for i, j in self.directed_edges if j == node}

    def _is_adjacent(self, i: int, j: int) -> bool:
        """Check if *i* and *j* are connected by any edge."""
        return (self.has_directed_edge(i, j)
                or self.has_directed_edge(j, i)
                or self.has_undirected_edge(i, j))

    # -----------------------------------------------------------------
    # Meek rules
    # -----------------------------------------------------------------

    def _apply_meek_rules(self, max_iter: int = 100) -> int:
        """Apply Meek rules R1-R4 exhaustively until convergence.

        Returns the number of iterations with changes.
        """
        from cpa.mec.orientation import MeekRules
        return MeekRules().apply_all(self, max_iter=max_iter)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def is_valid(self) -> bool:
        """Check whether the CPDAG is consistent.

        A valid CPDAG must:
        1. Have no self-loops.
        2. The directed part must be acyclic.
        3. Be maximally oriented (Meek rules produce no further changes).
        """
        return self.is_valid_cpdag()

    def is_valid_cpdag(self) -> bool:
        """Comprehensive CPDAG validity check."""
        # Check no self-loops
        for i, j in self.directed_edges:
            if i == j:
                return False
        for i, j in self.undirected_edges:
            if i == j:
                return False

        # Check directed part is acyclic
        adj_dir = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int_)
        for i, j in self.directed_edges:
            adj_dir[i, j] = 1
        if not _is_dag(adj_dir):
            return False

        # Check no contradictions: edge can't be both directed and undirected
        for i, j in self.directed_edges:
            if self.has_undirected_edge(i, j):
                return False
            # No opposing directed edge for undirected
        for i, j in self.undirected_edges:
            if self.has_directed_edge(i, j) or self.has_directed_edge(j, i):
                return False

        # Check Meek-closed: applying Meek rules shouldn't change anything
        copy = CPDAG(
            self.n_nodes,
            set(self.directed_edges),
            set(self.undirected_edges),
        )
        from cpa.mec.orientation import MeekRules
        changes = MeekRules().apply_all(copy, max_iter=1)
        if changes > 0:
            return False

        return True

    # -----------------------------------------------------------------
    # Compelled and reversible edges
    # -----------------------------------------------------------------

    def compelled_edges(self) -> Set[Tuple[int, int]]:
        """Return the set of compelled (directed) edges.

        An edge is compelled if it has the same orientation in every
        DAG in the MEC.  In a CPDAG, these are exactly the directed edges.
        """
        return set(self.directed_edges)

    def reversible_edges(self) -> Set[Tuple[int, int]]:
        """Return the set of reversible edges.

        An edge is reversible if there exist DAGs in the MEC with it
        oriented in both directions.  These are the undirected edges,
        returned as canonical (min, max) pairs.
        """
        return set(self.undirected_edges)

    # -----------------------------------------------------------------
    # DAG sampling
    # -----------------------------------------------------------------

    def to_dag(self, seed: int | None = None) -> NDArray[np.int_]:
        """Sample a DAG consistent with this CPDAG.

        Uses a random topological extension of the undirected components.
        """
        rng = np.random.default_rng(seed)
        n = self.n_nodes
        dag = np.zeros((n, n), dtype=np.int_)

        # Copy directed edges
        for i, j in self.directed_edges:
            dag[i, j] = 1

        # Orient each undirected edge randomly but consistently (acyclic)
        undirected_list = list(self.undirected_edges)
        rng.shuffle(undirected_list)

        for i, j in undirected_list:
            # Try i -> j first; if it creates a cycle, try j -> i
            dag[i, j] = 1
            if not _is_dag(dag):
                dag[i, j] = 0
                dag[j, i] = 1
                if not _is_dag(dag):
                    # Neither works — shouldn't happen for valid CPDAG
                    dag[j, i] = 0
                    raise RuntimeError(
                        f"Cannot orient edge ({i}, {j}) without creating a cycle"
                    )

        return dag

    # -----------------------------------------------------------------
    # Structural Hamming Distance
    # -----------------------------------------------------------------

    def structural_hamming_distance(self, other: CPDAG) -> int:
        """Compute Structural Hamming Distance (SHD) to another CPDAG.

        SHD counts edge differences:
        - Missing or extra directed edges.
        - Missing or extra undirected edges.
        - Edges with different types (directed vs undirected).
        """
        if self.n_nodes != other.n_nodes:
            raise ValueError("CPDAGs must have same number of nodes")

        adj1 = self.to_adjacency_matrix()
        adj2 = other.to_adjacency_matrix()

        shd = 0
        n = self.n_nodes
        for i in range(n):
            for j in range(i + 1, n):
                # Compare edge status between i and j
                e1_ij = bool(adj1[i, j])
                e1_ji = bool(adj1[j, i])
                e2_ij = bool(adj2[i, j])
                e2_ji = bool(adj2[j, i])
                if (e1_ij, e1_ji) != (e2_ij, e2_ji):
                    shd += 1
        return shd

    # -----------------------------------------------------------------
    # MEC size
    # -----------------------------------------------------------------

    def num_dags_in_mec(self) -> int:
        """Count the number of DAGs in the MEC.

        Uses recursive decomposition by chain components.  The total
        count is the product of counts over independent chain components.
        """
        from cpa.mec.enumeration import count_dags_in_mec
        return count_dags_in_mec(self)

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CPDAG(n_nodes={self.n_nodes}, "
            f"directed={len(self.directed_edges)}, "
            f"undirected={len(self.undirected_edges)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CPDAG):
            return NotImplemented
        return (
            self.n_nodes == other.n_nodes
            and self.directed_edges == other.directed_edges
            and self.undirected_edges == other.undirected_edges
        )

    def copy(self) -> CPDAG:
        """Return a deep copy."""
        return CPDAG(
            self.n_nodes,
            set(self.directed_edges),
            set(self.undirected_edges),
        )


# -------------------------------------------------------------------
# Module-level functions
# -------------------------------------------------------------------

def dag_to_cpdag(adj_matrix: NDArray[np.int_]) -> CPDAG:
    """Convert a DAG adjacency matrix to its CPDAG.

    This is the primary entry point for constructing a CPDAG from a DAG.
    Uses Chickering's algorithm: identify v-structures, then apply Meek
    rules R1-R4 exhaustively.

    Parameters
    ----------
    adj_matrix : NDArray
        Binary adjacency matrix where adj[i,j] = 1 means i -> j.

    Returns
    -------
    CPDAG
        The completed partially directed acyclic graph.
    """
    return CPDAG.from_dag(adj_matrix)


def cpdag_to_dags(cpdag: CPDAG) -> List[NDArray[np.int_]]:
    """Enumerate all DAGs in the Markov equivalence class of *cpdag*.

    Uses recursive orientation of undirected edges.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the equivalence class.

    Returns
    -------
    list of NDArray
        List of DAG adjacency matrices.
    """
    from cpa.mec.enumeration import enumerate_dags
    return enumerate_dags(cpdag)


def _find_v_structures(adj: NDArray[np.int_]) -> List[Tuple[int, int, int]]:
    """Find all v-structures in a DAG.

    A v-structure is a triple (a, b, c) where a -> b <- c and a, c
    are not adjacent.

    Returns list of (a, b, c) triples.
    """
    adj = np.asarray(adj, dtype=np.int_)
    n = adj.shape[0]
    v_structs = []
    for b in range(n):
        parents_b = [i for i in range(n) if adj[i, b] and not adj[b, i]]
        for idx_a in range(len(parents_b)):
            for idx_c in range(idx_a + 1, len(parents_b)):
                a = parents_b[idx_a]
                c = parents_b[idx_c]
                if not adj[a, c] and not adj[c, a]:
                    v_structs.append((a, b, c))
    return v_structs
