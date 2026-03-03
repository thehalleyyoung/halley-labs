"""Low-level adjacency matrix operations for causal graph analysis.

All functions operate on dense ``(n, n)`` numpy adjacency matrices and are
designed for maximum throughput via vectorized numpy operations and, where
beneficial, Numba JIT compilation.

Key algorithms
--------------
* Cycle detection via matrix power method
* Floyd-Warshall transitive closure
* Topological layer partitioning
* Pre-computed O(1) reachability queries
* Minimum feedback arc set (greedy + LP relaxation)
* Maximum acyclic subgraph
* DAG edit distance
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrapper(fn):  # type: ignore[no-untyped-def]
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrapper

    prange = range  # type: ignore[assignment,misc]

try:
    import scipy.sparse as sp
    from scipy.optimize import linprog

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

__all__ = [
    "cycle_detection_matrix",
    "transitive_closure_warshall",
    "transitive_closure_matmul",
    "topological_layers",
    "reachability_query",
    "ReachabilityIndex",
    "minimum_feedback_arc_set",
    "maximum_acyclic_subgraph",
    "dag_edit_distance",
    "topological_sort_kahn",
    "strongly_connected_components",
]


# ---------------------------------------------------------------------------
# Cycle detection via matrix power
# ---------------------------------------------------------------------------


def cycle_detection_matrix(adj: np.ndarray, early_stop: bool = True) -> bool:
    """Detect cycles using the matrix power method.

    A directed graph has a cycle if and only if ``trace(sum_{k=1}^{n} A^k) > 0``.
    With ``early_stop=True``, returns as soon as a positive trace contribution
    is found.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.
    early_stop : bool
        If True, terminate as soon as a cycle is detected.

    Returns
    -------
    bool
        True if the graph contains at least one directed cycle.

    Notes
    -----
    Complexity: O(n^4) worst case, but typically much faster with early
    termination.  For large sparse graphs, prefer DFS-based detection.
    """
    n = adj.shape[0]
    if n == 0:
        return False

    A = adj.astype(np.float64)
    power = np.eye(n, dtype=np.float64)
    trace_sum = 0.0

    for k in range(1, n + 1):
        power = power @ A
        tr = np.trace(power)
        trace_sum += tr
        if early_stop and tr > 0:
            return True
        # If power is zero matrix, no more paths
        if np.max(np.abs(power)) == 0:
            break

    return trace_sum > 0


@njit(cache=True)  # type: ignore[misc]
def _jit_cycle_detection_trace(adj: np.ndarray) -> bool:
    """Numba-compiled cycle detection via trace of matrix powers."""
    n = adj.shape[0]
    if n == 0:
        return False

    power = np.eye(n)
    A = adj.astype(np.float64)
    for _ in range(n):
        power = power @ A
        tr = 0.0
        for i in range(n):
            tr += power[i, i]
        if tr > 0:
            return True
        # Check zero
        mx = 0.0
        for i in range(n):
            for j in range(n):
                if abs(power[i, j]) > mx:
                    mx = abs(power[i, j])
        if mx == 0.0:
            break
    return False


# ---------------------------------------------------------------------------
# Transitive closure
# ---------------------------------------------------------------------------


def transitive_closure_warshall(adj: np.ndarray) -> np.ndarray:
    """Floyd-Warshall transitive closure with numpy vectorization.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.

    Returns
    -------
    np.ndarray
        Boolean ``(n, n)`` reachability matrix.

    Notes
    -----
    Uses a vectorized inner loop: for each intermediate node ``k``, update
    the entire matrix at once using outer product.  Complexity: O(n^3).
    """
    n = adj.shape[0]
    reach = adj.astype(np.bool_).copy()

    for k in range(n):
        # reach[i, j] |= reach[i, k] & reach[k, j]
        col_k = reach[:, k].reshape(-1, 1)  # (n, 1)
        row_k = reach[k, :].reshape(1, -1)  # (1, n)
        reach |= col_k & row_k

    return reach


def transitive_closure_matmul(adj: np.ndarray) -> np.ndarray:
    """Transitive closure via repeated boolean matrix multiplication.

    Typically faster than Warshall for dense graphs due to BLAS utilization.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.

    Returns
    -------
    np.ndarray
        Boolean ``(n, n)`` reachability matrix.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.bool_)

    A = adj.astype(np.float64)
    R = A.copy()
    power = A.copy()

    for _ in range(n - 1):
        power = power @ A
        new_R = R + power
        if np.array_equal(new_R > 0, R > 0):
            break
        R = new_R

    return R > 0


@njit(cache=True)  # type: ignore[misc]
def _jit_warshall(adj: np.ndarray) -> np.ndarray:
    """Numba-compiled Floyd-Warshall."""
    n = adj.shape[0]
    reach = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                reach[i, j] = True

    for k in range(n):
        for i in range(n):
            if reach[i, k]:
                for j in range(n):
                    if reach[k, j]:
                        reach[i, j] = True
    return reach


# ---------------------------------------------------------------------------
# Topological layers
# ---------------------------------------------------------------------------


def topological_layers(adj: np.ndarray) -> List[List[int]]:
    """Partition nodes into topological layers.

    Layer 0 contains nodes with in-degree 0.  Layer ``k`` contains nodes
    whose parents are all in layers < ``k``.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.

    Returns
    -------
    list of list of int
        Each inner list is a layer of node indices.

    Raises
    ------
    ValueError
        If the graph contains a cycle (not all nodes assigned to layers).
    """
    n = adj.shape[0]
    if n == 0:
        return []

    in_deg = adj.sum(axis=0).astype(np.int64)
    remaining = np.ones(n, dtype=np.bool_)
    layers: List[List[int]] = []
    processed = 0

    while processed < n:
        # Current layer: nodes with in-degree 0 among remaining
        layer_mask = (in_deg == 0) & remaining
        layer = np.nonzero(layer_mask)[0].tolist()
        if not layer:
            raise ValueError(
                "Graph contains a cycle; not all nodes can be layered."
            )
        layers.append(layer)
        processed += len(layer)

        # Remove layer nodes and update in-degrees
        for u in layer:
            remaining[u] = False
            children = np.nonzero(adj[u])[0]
            in_deg[children] -= 1

    return layers


@njit(cache=True)  # type: ignore[misc]
def _jit_topological_layers(adj: np.ndarray) -> np.ndarray:
    """Numba-compiled topological layer assignment.

    Returns an array of length n where ``result[i]`` is the layer index of
    node ``i``.  Returns -1 for all nodes if a cycle is detected.
    """
    n = adj.shape[0]
    in_deg = np.zeros(n, dtype=np.int64)
    for j in range(n):
        for i in range(n):
            if adj[i, j] != 0:
                in_deg[j] += 1

    layer_of = np.full(n, -1, dtype=np.int64)
    remaining = np.ones(n, dtype=np.bool_)
    current_layer = 0
    processed = 0

    while processed < n:
        found_any = False
        for i in range(n):
            if remaining[i] and in_deg[i] == 0:
                layer_of[i] = current_layer
                remaining[i] = False
                processed += 1
                found_any = True
        if not found_any:
            layer_of[:] = -1
            return layer_of

        # Update in-degrees
        for i in range(n):
            if layer_of[i] == current_layer:
                for j in range(n):
                    if adj[i, j] != 0 and remaining[j]:
                        in_deg[j] -= 1
        current_layer += 1

    return layer_of


# ---------------------------------------------------------------------------
# Reachability index for O(1) queries
# ---------------------------------------------------------------------------


class ReachabilityIndex:
    """Pre-computed reachability index for O(1) pairwise reachability queries.

    After construction, ``query(i, j)`` returns whether node ``i`` can reach
    node ``j`` in O(1) time by table lookup.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix of a DAG.

    Examples
    --------
    >>> idx = ReachabilityIndex(adj)
    >>> idx.query(0, 3)
    True
    >>> idx.ancestors(3)
    frozenset({0, 1, 2})
    """

    __slots__ = ("_reach", "_n")

    def __init__(self, adj: np.ndarray) -> None:
        self._n = adj.shape[0]
        self._reach = transitive_closure_warshall(adj)

    def query(self, source: int, target: int) -> bool:
        """O(1) reachability query: can *source* reach *target*?"""
        return bool(self._reach[source, target])

    def ancestors(self, node: int) -> frozenset:
        """All nodes that can reach *node*."""
        return frozenset(np.nonzero(self._reach[:, node])[0].tolist())

    def descendants(self, node: int) -> frozenset:
        """All nodes reachable from *node*."""
        return frozenset(np.nonzero(self._reach[node, :])[0].tolist())

    def common_ancestors(self, u: int, v: int) -> frozenset:
        """Nodes that can reach both *u* and *v*."""
        mask = self._reach[:, u] & self._reach[:, v]
        return frozenset(np.nonzero(mask)[0].tolist())

    def common_descendants(self, u: int, v: int) -> frozenset:
        """Nodes reachable from both *u* and *v*."""
        mask = self._reach[u, :] & self._reach[v, :]
        return frozenset(np.nonzero(mask)[0].tolist())

    @property
    def matrix(self) -> np.ndarray:
        """The full reachability matrix (read-only view)."""
        result = self._reach.copy()
        result.flags.writeable = False
        return result

    def __repr__(self) -> str:
        return f"ReachabilityIndex(n={self._n})"


def reachability_query(
    adj: np.ndarray, source: int, target: int
) -> bool:
    """One-off reachability query using BFS.

    For repeated queries, prefer :class:`ReachabilityIndex`.
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    stack = [source]
    visited[source] = True
    while stack:
        u = stack.pop()
        if u == target:
            return True
        for v in np.nonzero(adj[u])[0]:
            if not visited[v]:
                visited[v] = True
                stack.append(int(v))
    return False


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm, numpy-based)
# ---------------------------------------------------------------------------


def topological_sort_kahn(adj: np.ndarray) -> List[int]:
    """Kahn's algorithm for topological sorting using numpy arrays.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.

    Returns
    -------
    list of int
        Topological ordering of node indices.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(np.int64).copy()

    queue = list(np.nonzero(in_deg == 0)[0])
    order: List[int] = []

    while queue:
        u = queue.pop(0)
        order.append(int(u))
        children = np.nonzero(adj[u])[0]
        for v in children:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(int(v))

    if len(order) != n:
        raise ValueError("Graph contains a cycle.")
    return order


# ---------------------------------------------------------------------------
# Strongly connected components (Tarjan's)
# ---------------------------------------------------------------------------


def strongly_connected_components(adj: np.ndarray) -> List[List[int]]:
    """Find all SCCs using iterative Tarjan's algorithm.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix.

    Returns
    -------
    list of list of int
        Each inner list is a strongly connected component.
    """
    n = adj.shape[0]
    index_counter = [0]
    stack: List[int] = []
    lowlink = np.full(n, -1, dtype=np.int64)
    index = np.full(n, -1, dtype=np.int64)
    on_stack = np.zeros(n, dtype=np.bool_)
    result: List[List[int]] = []

    def strongconnect(v: int) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in np.nonzero(adj[v])[0]:
            w = int(w)
            if index[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: List[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for v in range(n):
        if index[v] == -1:
            strongconnect(v)

    return result


# ---------------------------------------------------------------------------
# Minimum feedback arc set
# ---------------------------------------------------------------------------


def minimum_feedback_arc_set(
    adj: np.ndarray, method: str = "greedy"
) -> List[Tuple[int, int]]:
    """Find an approximate minimum feedback arc set.

    A feedback arc set is a set of edges whose removal makes the graph acyclic.
    Finding the exact minimum is NP-hard; we provide greedy and LP-relaxation
    approximations.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix (may contain cycles).
    method : str
        ``"greedy"`` for greedy heuristic, ``"lp"`` for LP relaxation
        (requires scipy).

    Returns
    -------
    list of (int, int)
        Edges in the feedback arc set.
    """
    if method == "greedy":
        return _fas_greedy(adj)
    elif method == "lp":
        if not HAS_SCIPY:
            raise ImportError("scipy required for LP-based FAS.")
        return _fas_lp_relaxation(adj)
    else:
        raise ValueError(f"Unknown FAS method: {method}")


def _fas_greedy(adj: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy FAS based on topological ordering heuristic.

    Algorithm: repeatedly remove sources (in-degree 0) and sinks (out-degree 0),
    then pick the node with maximum (out-degree - in-degree) from the remaining
    graph, adding its back edges to the FAS.
    """
    n = adj.shape[0]
    A = adj.astype(np.int8).copy()
    remaining = set(range(n))
    s1: List[int] = []  # front of ordering
    s2: List[int] = []  # back of ordering

    while remaining:
        changed = True
        while changed:
            changed = False
            to_remove = []
            for v in list(remaining):
                in_d = int(A[:, v].sum())
                out_d = int(A[v, :].sum())
                if in_d == 0:
                    s1.append(v)
                    to_remove.append(v)
                    changed = True
                elif out_d == 0:
                    s2.append(v)
                    to_remove.append(v)
                    changed = True
            for v in to_remove:
                remaining.discard(v)
                A[v, :] = 0
                A[:, v] = 0

        if remaining:
            # Pick node with max (out_degree - in_degree)
            best_v = -1
            best_delta = -np.inf
            for v in remaining:
                delta = float(A[v, :].sum()) - float(A[:, v].sum())
                if delta > best_delta:
                    best_delta = delta
                    best_v = v
            s1.append(best_v)
            remaining.discard(best_v)
            A[best_v, :] = 0
            A[:, best_v] = 0

    ordering = s1 + list(reversed(s2))
    order_pos = {node: pos for pos, node in enumerate(ordering)}

    # FAS = edges that go backwards in the ordering
    fas: List[Tuple[int, int]] = []
    rows, cols = np.nonzero(adj)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if order_pos.get(i, 0) >= order_pos.get(j, 0):
            fas.append((i, j))

    return fas


def _fas_lp_relaxation(adj: np.ndarray) -> List[Tuple[int, int]]:
    """LP relaxation based FAS approximation.

    Relaxes the integer program for FAS to a linear program, then rounds
    fractional solutions.
    """
    n = adj.shape[0]
    edges = list(zip(*np.nonzero(adj)))
    m = len(edges)
    if m == 0:
        return []

    edge_idx = {e: i for i, e in enumerate(edges)}

    # Decision variables: x_e for each edge (probability of being in FAS)
    # Minimize: sum x_e
    c = np.ones(m)

    # For each cycle, sum of x_e over edges in cycle >= 1
    # Finding all cycles is expensive; use SCC-based constraint generation
    sccs = strongly_connected_components(adj)
    cycle_sccs = [scc for scc in sccs if len(scc) > 1]

    if not cycle_sccs:
        return []

    A_ub_rows = []
    b_ub_vals = []

    for scc in cycle_sccs:
        scc_set = set(scc)
        scc_edges = []
        for u, v in edges:
            if u in scc_set and v in scc_set:
                scc_edges.append(edge_idx[(u, v)])
        if scc_edges:
            row = np.zeros(m)
            for idx in scc_edges:
                row[idx] = -1.0  # negate for <= constraint
            A_ub_rows.append(row)
            b_ub_vals.append(-1.0)

    if not A_ub_rows:
        return []

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    bounds = [(0, 1) for _ in range(m)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if result.success:
            # Round: include edge if x_e > 0.5
            fas = [edges[i] for i in range(m) if result.x[i] > 0.5]
            # If rounding doesn't break all cycles, fall back to greedy
            test_adj = adj.copy()
            for u, v in fas:
                test_adj[u, v] = 0
            if cycle_detection_matrix(test_adj):
                return _fas_greedy(adj)
            return fas
    except Exception:
        pass

    return _fas_greedy(adj)


# ---------------------------------------------------------------------------
# Maximum acyclic subgraph
# ---------------------------------------------------------------------------


def maximum_acyclic_subgraph(adj: np.ndarray) -> np.ndarray:
    """Find the largest acyclic subgraph by removing a minimum FAS.

    Parameters
    ----------
    adj : np.ndarray
        ``(n, n)`` adjacency matrix (may contain cycles).

    Returns
    -------
    np.ndarray
        Adjacency matrix of the maximum acyclic subgraph.
    """
    fas = minimum_feedback_arc_set(adj, method="greedy")
    result = adj.copy()
    for u, v in fas:
        result[u, v] = 0
    return result


# ---------------------------------------------------------------------------
# DAG edit distance
# ---------------------------------------------------------------------------


def dag_edit_distance(
    adj1: np.ndarray,
    adj2: np.ndarray,
    add_cost: float = 1.0,
    remove_cost: float = 1.0,
    reverse_cost: float = 1.0,
) -> float:
    """Minimum weighted edge operations to transform one DAG to another.

    Operations:
        - Add an edge (cost: ``add_cost``)
        - Remove an edge (cost: ``remove_cost``)
        - Reverse an edge (cost: ``reverse_cost``)

    This is a lower bound on the true edit distance (which would also
    consider acyclicity constraints on intermediate states).

    Parameters
    ----------
    adj1, adj2 : np.ndarray
        ``(n, n)`` adjacency matrices of the two DAGs.
    add_cost, remove_cost, reverse_cost : float
        Costs for each operation type.

    Returns
    -------
    float
        The (approximate) minimum total cost.
    """
    a1 = adj1.astype(np.int8)
    a2 = adj2.astype(np.int8)
    n = a1.shape[0]

    total_cost = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            e1_ij = bool(a1[i, j])
            e1_ji = bool(a1[j, i])
            e2_ij = bool(a2[i, j])
            e2_ji = bool(a2[j, i])

            if e1_ij == e2_ij and e1_ji == e2_ji:
                continue  # identical

            if e1_ij and not e1_ji and not e2_ij and e2_ji:
                # i->j needs to become j->i: reverse
                total_cost += min(reverse_cost, remove_cost + add_cost)
            elif not e1_ij and e1_ji and e2_ij and not e2_ji:
                # j->i needs to become i->j: reverse
                total_cost += min(reverse_cost, remove_cost + add_cost)
            else:
                # General case: count individual add/remove
                if e1_ij and not e2_ij:
                    total_cost += remove_cost
                elif not e1_ij and e2_ij:
                    total_cost += add_cost
                if e1_ji and not e2_ji:
                    total_cost += remove_cost
                elif not e1_ji and e2_ji:
                    total_cost += add_cost

    return total_cost


def dag_edit_distance_symmetric(
    adj1: np.ndarray, adj2: np.ndarray
) -> int:
    """Structural Hamming Distance: number of edge disagreements.

    Counts positions ``(i, j)`` where ``adj1[i,j] != adj2[i,j]``.
    This equals the standard SHD metric for DAGs.
    """
    return int(np.sum(adj1.astype(np.int8) != adj2.astype(np.int8)))


# ---------------------------------------------------------------------------
# Utility: batch operations on multiple adjacency matrices
# ---------------------------------------------------------------------------


def batch_topological_sort(
    adjs: np.ndarray,
) -> List[Optional[List[int]]]:
    """Topological sort for a batch of adjacency matrices.

    Parameters
    ----------
    adjs : np.ndarray
        ``(batch, n, n)`` stack of adjacency matrices.

    Returns
    -------
    list of list of int or None
        Topological orders; None for matrices containing cycles.
    """
    batch_size = adjs.shape[0]
    results: List[Optional[List[int]]] = []
    for b in range(batch_size):
        try:
            results.append(topological_sort_kahn(adjs[b]))
        except ValueError:
            results.append(None)
    return results


def batch_cycle_check(adjs: np.ndarray) -> np.ndarray:
    """Check for cycles in a batch of adjacency matrices.

    Parameters
    ----------
    adjs : np.ndarray
        ``(batch, n, n)`` stack of adjacency matrices.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(batch,)``, True where cycles exist.
    """
    batch_size = adjs.shape[0]
    result = np.empty(batch_size, dtype=np.bool_)
    for b in range(batch_size):
        result[b] = cycle_detection_matrix(adjs[b])
    return result


def batch_edge_count(adjs: np.ndarray) -> np.ndarray:
    """Count edges in a batch of adjacency matrices.

    Parameters
    ----------
    adjs : np.ndarray
        ``(batch, n, n)`` stack of adjacency matrices.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(batch,)`` with edge counts.
    """
    return adjs.reshape(adjs.shape[0], -1).sum(axis=1).astype(np.int64)


def batch_density(adjs: np.ndarray) -> np.ndarray:
    """Compute densities for a batch of adjacency matrices.

    Parameters
    ----------
    adjs : np.ndarray
        ``(batch, n, n)`` stack of adjacency matrices.

    Returns
    -------
    np.ndarray
        Float array of shape ``(batch,)`` with densities.
    """
    n = adjs.shape[1]
    max_edges = n * (n - 1)
    if max_edges == 0:
        return np.zeros(adjs.shape[0], dtype=np.float64)
    counts = batch_edge_count(adjs).astype(np.float64)
    return counts / max_edges
