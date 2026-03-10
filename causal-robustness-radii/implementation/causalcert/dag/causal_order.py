"""Causal ordering algorithms for DAGs and CPDAGs.

Provides topological ordering variants, causal-hierarchy computation,
compatible-ordering enumeration, order-based DAG reconstruction,
and partial-order extraction from CPDAGs.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict, deque
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from causalcert.dag.graph import CausalDAG

NodeId = int


# ===================================================================
# 1.  Standard topological orderings
# ===================================================================

def topological_sort_kahn(dag: CausalDAG) -> List[NodeId]:
    """Kahn's algorithm: BFS-based topological sort (lexicographically smallest)."""
    adj = dag.adj
    n = dag.n_nodes
    in_deg = adj.sum(axis=0).astype(int)
    queue: List[NodeId] = sorted(i for i in range(n) if in_deg[i] == 0)
    order: List[NodeId] = []
    while queue:
        queue.sort()
        v = queue.pop(0)
        order.append(v)
        for w in range(n):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    if len(order) != n:
        raise ValueError("Graph contains a cycle")
    return order


def topological_sort_dfs(dag: CausalDAG) -> List[NodeId]:
    """DFS-based topological sort (reverse post-order)."""
    adj = dag.adj
    n = dag.n_nodes
    visited = [False] * n
    stack: List[NodeId] = []

    def _dfs(v: NodeId) -> None:
        visited[v] = True
        for w in range(n):
            if adj[v, w] and not visited[w]:
                _dfs(w)
        stack.append(v)

    for v in range(n):
        if not visited[v]:
            _dfs(v)

    stack.reverse()
    return stack


def topological_sort_random(
    dag: CausalDAG,
    *,
    rng: Optional[random.Random] = None,
) -> List[NodeId]:
    """Random topological sort: break ties randomly among sources."""
    if rng is None:
        rng = random.Random()
    adj = dag.adj
    n = dag.n_nodes
    in_deg = adj.sum(axis=0).astype(int)
    available = [i for i in range(n) if in_deg[i] == 0]
    order: List[NodeId] = []
    while available:
        v = rng.choice(available)
        available.remove(v)
        order.append(v)
        for w in range(n):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    available.append(w)
    return order


def topological_sort_bfs_layers(dag: CausalDAG) -> List[List[NodeId]]:
    """BFS-based layer decomposition: nodes at the same 'depth' are grouped.

    Returns list of layers where each layer contains nodes whose parents
    are all in previous layers.
    """
    adj = dag.adj
    n = dag.n_nodes
    in_deg = adj.sum(axis=0).astype(int)
    current = sorted(i for i in range(n) if in_deg[i] == 0)
    layers: List[List[NodeId]] = []
    while current:
        layers.append(sorted(current))
        nxt: List[NodeId] = []
        for v in current:
            for w in range(n):
                if adj[v, w]:
                    in_deg[w] -= 1
                    if in_deg[w] == 0:
                        nxt.append(w)
        current = nxt
    return layers


# ===================================================================
# 2.  All topological orderings enumeration
# ===================================================================

def all_topological_sorts(
    dag: CausalDAG,
    *,
    max_count: int = 10000,
) -> List[List[NodeId]]:
    """Enumerate all topological orderings (up to *max_count*).

    Uses backtracking.  Exponential in general but bounded by *max_count*.
    """
    adj = dag.adj
    n = dag.n_nodes
    in_deg = adj.sum(axis=0).astype(int).tolist()
    results: List[List[NodeId]] = []
    current: List[NodeId] = []
    used = [False] * n

    def _backtrack() -> None:
        if len(results) >= max_count:
            return
        if len(current) == n:
            results.append(list(current))
            return
        sources = [v for v in range(n) if not used[v] and in_deg[v] == 0]
        for v in sources:
            used[v] = True
            current.append(v)
            for w in range(n):
                if adj[v, w]:
                    in_deg[w] -= 1
            _backtrack()
            current.pop()
            used[v] = False
            for w in range(n):
                if adj[v, w]:
                    in_deg[w] += 1

    _backtrack()
    return results


def count_topological_sorts(dag: CausalDAG, *, max_count: int = 100000) -> int:
    """Count topological orderings (bounded by *max_count*)."""
    return len(all_topological_sorts(dag, max_count=max_count))


# ===================================================================
# 3.  Causal ordering for identification
# ===================================================================

def causal_order(dag: CausalDAG) -> List[NodeId]:
    """A topological order that respects causal priority.

    Equivalent to any valid topological sort; named for clarity in
    identification algorithms.
    """
    return topological_sort_kahn(dag)


def is_valid_causal_order(dag: CausalDAG, order: Sequence[NodeId]) -> bool:
    """Check if *order* is a valid topological ordering of *dag*."""
    if set(order) != set(range(dag.n_nodes)):
        return False
    rank = {node: idx for idx, node in enumerate(order)}
    for u, v in dag.edge_list():
        if rank[u] >= rank[v]:
            return False
    return True


def causal_predecessors(
    dag: CausalDAG,
    order: Sequence[NodeId],
    node: NodeId,
) -> List[NodeId]:
    """All nodes preceding *node* in the causal order *order*."""
    rank = {n: idx for idx, n in enumerate(order)}
    r = rank[node]
    return [n for n in order if rank[n] < r]


def causal_successors(
    dag: CausalDAG,
    order: Sequence[NodeId],
    node: NodeId,
) -> List[NodeId]:
    """All nodes following *node* in the causal order *order*."""
    rank = {n: idx for idx, n in enumerate(order)}
    r = rank[node]
    return [n for n in order if rank[n] > r]


# ===================================================================
# 4.  Compatible causal orderings enumeration
# ===================================================================

def compatible_orderings(
    dag: CausalDAG,
    constraints: List[Tuple[NodeId, NodeId]],
    *,
    max_count: int = 5000,
) -> List[List[NodeId]]:
    """Topological orderings that also respect additional ordering constraints.

    Parameters
    ----------
    constraints : list of (a, b)
        Each constraint requires ``a`` to appear before ``b`` in the ordering.
    """
    adj = dag.adj.copy()
    n = dag.n_nodes

    for a, b in constraints:
        adj[a, b] = 1

    if not _is_acyclic(adj):
        return []

    temp_dag = CausalDAG.from_adjacency_matrix(adj)
    return all_topological_sorts(temp_dag, max_count=max_count)


def _is_acyclic(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for w in range(n):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    return count == n


# ===================================================================
# 5.  Order-based DAG reconstruction
# ===================================================================

def dag_from_order_and_parents(
    n_nodes: int,
    order: Sequence[NodeId],
    parent_sets: Dict[NodeId, Set[NodeId]],
) -> CausalDAG:
    """Construct a DAG from a causal order and parent-set assignments.

    Each node in *order* gets the parents specified in *parent_sets*.
    Validates that all parents precede the node in the ordering.
    """
    rank = {node: idx for idx, node in enumerate(order)}
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for child, parents in parent_sets.items():
        for p in parents:
            if rank[p] >= rank[child]:
                raise ValueError(
                    f"Parent {p} does not precede child {child} in ordering"
                )
            adj[p, child] = 1
    return CausalDAG.from_adjacency_matrix(adj)


def order_to_complete_dag(
    n_nodes: int,
    order: Sequence[NodeId],
) -> CausalDAG:
    """Build the complete DAG consistent with *order* (every earlier node is a parent)."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for idx, v in enumerate(order):
        for prev_idx in range(idx):
            adj[order[prev_idx], v] = 1
    return CausalDAG.from_adjacency_matrix(adj)


def sparse_dag_from_order(
    n_nodes: int,
    order: Sequence[NodeId],
    *,
    max_parents: int = 3,
    rng: Optional[random.Random] = None,
) -> CausalDAG:
    """Build a sparse random DAG consistent with *order*.

    Each node gets up to *max_parents* randomly chosen predecessors.
    """
    if rng is None:
        rng = random.Random()
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for idx, v in enumerate(order):
        predecessors = [order[j] for j in range(idx)]
        if not predecessors:
            continue
        n_parents = min(max_parents, len(predecessors))
        parents = rng.sample(predecessors, rng.randint(1, n_parents))
        for p in parents:
            adj[p, v] = 1
    return CausalDAG.from_adjacency_matrix(adj)


# ===================================================================
# 6.  Causal hierarchy (layers)
# ===================================================================

def causal_layers(dag: CausalDAG) -> List[List[NodeId]]:
    """Synonym for :func:`topological_sort_bfs_layers`.

    Returns layers where layer 0 = roots, layer k depends only on
    layers 0..k-1.
    """
    return topological_sort_bfs_layers(dag)


def layer_of(dag: CausalDAG, node: NodeId) -> int:
    """Return the layer index of *node*."""
    layers = causal_layers(dag)
    for idx, layer in enumerate(layers):
        if node in layer:
            return idx
    raise ValueError(f"Node {node} not found")


def depth(dag: CausalDAG) -> int:
    """Maximum depth (number of layers - 1)."""
    return len(causal_layers(dag)) - 1


def width(dag: CausalDAG) -> int:
    """Maximum width (largest layer size)."""
    layers = causal_layers(dag)
    return max(len(layer) for layer in layers) if layers else 0


def layer_adjacency(dag: CausalDAG) -> Dict[Tuple[int, int], int]:
    """Count of edges between each pair of layers.

    Returns dict mapping (layer_i, layer_j) → edge count.
    """
    layers = causal_layers(dag)
    node_to_layer = {}
    for idx, layer in enumerate(layers):
        for v in layer:
            node_to_layer[v] = idx
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for u, v in dag.edge_list():
        counts[(node_to_layer[u], node_to_layer[v])] += 1
    return dict(counts)


def is_layered_dag(dag: CausalDAG) -> bool:
    """Check if all edges go between consecutive layers."""
    layers = causal_layers(dag)
    node_to_layer = {}
    for idx, layer in enumerate(layers):
        for v in layer:
            node_to_layer[v] = idx
    for u, v in dag.edge_list():
        if node_to_layer[v] - node_to_layer[u] != 1:
            return False
    return True


# ===================================================================
# 7.  Partial ordering from CPDAG
# ===================================================================

def partial_order_from_cpdag(cpdag: np.ndarray) -> List[Tuple[NodeId, NodeId]]:
    """Extract the partial order implied by a CPDAG.

    A directed edge i→j in the CPDAG implies i must precede j.
    Returns list of (i, j) pairs representing the partial order.
    """
    n = cpdag.shape[0]
    relations: List[Tuple[NodeId, NodeId]] = []
    for i in range(n):
        for j in range(n):
            if cpdag[i, j] == 1 and cpdag[j, i] == 0:
                relations.append((i, j))
    return relations


def partial_order_transitive_closure(
    n_nodes: int,
    relations: List[Tuple[NodeId, NodeId]],
) -> np.ndarray:
    """Compute transitive closure of a partial order.

    Returns boolean adjacency matrix where ``out[i,j] = True`` means
    i must precede j.
    """
    tc = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i, j in relations:
        tc[i, j] = True

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if tc[i, k] and tc[k, j]:
                    tc[i, j] = True
    return tc


def linear_extensions_of_partial_order(
    n_nodes: int,
    relations: List[Tuple[NodeId, NodeId]],
    *,
    max_count: int = 5000,
) -> List[List[NodeId]]:
    """Enumerate linear extensions of a partial order.

    A linear extension is a total ordering consistent with the partial order.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i, j in relations:
        adj[i, j] = 1

    if not _is_acyclic(adj):
        return []

    dag = CausalDAG.from_adjacency_matrix(adj)
    return all_topological_sorts(dag, max_count=max_count)


def cpdag_compatible_orderings(
    dag: CausalDAG,
    *,
    max_count: int = 500,
) -> List[List[NodeId]]:
    """Orderings compatible with the CPDAG of *dag*.

    These are all linear extensions of the partial order implied by
    the directed edges in the CPDAG.
    """
    from causalcert.dag.equivalence import to_cpdag

    cp = to_cpdag(dag)
    relations = partial_order_from_cpdag(cp)
    return linear_extensions_of_partial_order(
        dag.n_nodes, relations, max_count=max_count
    )


# ===================================================================
# 8.  Order-based metrics
# ===================================================================

def kendall_tau_distance(order1: Sequence[NodeId], order2: Sequence[NodeId]) -> int:
    """Kendall-tau distance between two orderings (number of pairwise disagreements)."""
    n = len(order1)
    if set(order1) != set(order2):
        raise ValueError("Orderings must contain the same elements")
    rank1 = {v: i for i, v in enumerate(order1)}
    rank2 = {v: i for i, v in enumerate(order2)}
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order1[i], order1[j]
            if (rank1[a] - rank1[b]) * (rank2[a] - rank2[b]) < 0:
                dist += 1
    return dist


def spearman_footrule(order1: Sequence[NodeId], order2: Sequence[NodeId]) -> int:
    """Spearman footrule distance: sum of |rank1(v) - rank2(v)|."""
    rank1 = {v: i for i, v in enumerate(order1)}
    rank2 = {v: i for i, v in enumerate(order2)}
    return sum(abs(rank1[v] - rank2[v]) for v in rank1)


def order_consistency_score(
    dag: CausalDAG,
    order: Sequence[NodeId],
) -> float:
    """Fraction of edges consistent with the ordering (1.0 = valid)."""
    rank = {v: i for i, v in enumerate(order)}
    edges = dag.edge_list()
    if not edges:
        return 1.0
    consistent = sum(1 for u, v in edges if rank[u] < rank[v])
    return consistent / len(edges)
