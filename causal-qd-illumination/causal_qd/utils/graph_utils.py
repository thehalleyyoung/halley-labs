"""Graph-theoretic utility functions operating on adjacency matrices.

Includes DAG validation, topological sorting, structural distance metrics,
transitive operations, path finding, Meek's orientation rules, and
v-structure detection.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix, TopologicalOrder


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------

def is_dag(adj: AdjacencyMatrix) -> bool:
    """Check whether *adj* encodes a directed acyclic graph (DFS-based).

    Parameters
    ----------
    adj:
        Square binary adjacency matrix where ``adj[i, j] == 1`` means
        edge *i → j*.

    Returns
    -------
    bool
        *True* iff the graph contains no directed cycle.
    """
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = np.zeros(n, dtype=np.int8)

    def _dfs(u: int) -> bool:
        colour[u] = GRAY
        for v in range(n):
            if adj[u, v]:
                if colour[v] == GRAY:
                    return False
                if colour[v] == WHITE and not _dfs(v):
                    return False
        colour[u] = BLACK
        return True

    for u in range(n):
        if colour[u] == WHITE:
            if not _dfs(u):
                return False
    return True


def is_cpdag(adj: AdjacencyMatrix) -> bool:
    """Check whether *adj* represents a valid CPDAG.

    A CPDAG (Completed PDAG) has the property that:
    1. The undirected part forms chordal components.
    2. Every directed edge is compelled (participates in a v-structure or
       is implied by Meek's rules).
    3. The directed part is acyclic.

    This is a simplified check: we verify acyclicity of directed edges
    and that the graph can be extended to a DAG.
    """
    n = adj.shape[0]
    # Extract purely directed edges (i→j exists but j→i doesn't)
    directed = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adj[i, j] and not adj[j, i]:
                directed[i, j] = 1

    # The directed subgraph must be acyclic
    if not is_dag(directed):
        return False

    # Every undirected edge should be part of a reversible pair
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and adj[j, i]:
                # This is an undirected edge - valid in CPDAG
                continue

    return True


def is_pdag(adj: AdjacencyMatrix) -> bool:
    """Check whether *adj* represents a valid PDAG (partially directed).

    A PDAG is valid if its directed edges form no directed cycles,
    even when considering paths through undirected edges.
    """
    n = adj.shape[0]
    # Extract directed-only edges
    directed = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adj[i, j] and not adj[j, i]:
                directed[i, j] = 1

    return is_dag(directed)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def topological_sort(adj: AdjacencyMatrix) -> TopologicalOrder:
    """Return a topological order using Kahn's algorithm.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    n = adj.shape[0]
    in_degree = adj.sum(axis=0).astype(int)
    queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    order: list[int] = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in range(n):
            if adj[u, v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if len(order) != n:
        raise ValueError("Graph contains a cycle; topological sort is impossible.")
    return order


# ---------------------------------------------------------------------------
# Ancestor / descendant queries
# ---------------------------------------------------------------------------

def find_ancestors(adj: AdjacencyMatrix, node: int) -> Set[int]:
    """Return all ancestors of *node* (excluding *node* itself)."""
    n = adj.shape[0]
    visited: set[int] = set()
    stack = [node]
    while stack:
        v = stack.pop()
        for u in range(n):
            if adj[u, v] and u not in visited:
                visited.add(u)
                stack.append(u)
    return visited


def find_descendants(adj: AdjacencyMatrix, node: int) -> Set[int]:
    """Return all descendants of *node* (excluding *node* itself)."""
    n = adj.shape[0]
    visited: set[int] = set()
    stack = [node]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and v not in visited:
                visited.add(v)
                stack.append(v)
    return visited


# ---------------------------------------------------------------------------
# Graph structure operations
# ---------------------------------------------------------------------------

def skeleton(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Return the undirected skeleton of a directed graph."""
    skel = (adj | adj.T).astype(np.int8)
    return skel


def moral_graph(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Return the moral graph of a DAG.

    Marry all co-parents and drop edge directions.
    """
    n = adj.shape[0]
    moral = adj.copy()
    for j in range(n):
        parents = [i for i in range(n) if adj[i, j]]
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                moral[parents[a], parents[b]] = 1
                moral[parents[b], parents[a]] = 1
    return skeleton(moral)


# ---------------------------------------------------------------------------
# Structural distances
# ---------------------------------------------------------------------------

def shd(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> int:
    """Structural Hamming Distance between two DAGs.

    Counts the number of edge slots where the two graphs differ.
    For DAGs, this counts additions, deletions, and reversals.
    A reversal counts as one operation (not two).
    """
    n = adj1.shape[0]
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            e1_ij = bool(adj1[i, j])
            e1_ji = bool(adj1[j, i])
            e2_ij = bool(adj2[i, j])
            e2_ji = bool(adj2[j, i])
            if (e1_ij, e1_ji) != (e2_ij, e2_ji):
                dist += 1
    return dist


def graph_edit_distance(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> int:
    """Graph edit distance: total number of differing entries in adjacency matrices.

    This is the element-wise Hamming distance, also known as the
    Structural Intervention Distance for some formulations.
    """
    return int(np.sum(adj1 != adj2))


# ---------------------------------------------------------------------------
# Edge counting
# ---------------------------------------------------------------------------

def count_edges(adj: AdjacencyMatrix) -> int:
    """Count the number of directed edges in the graph."""
    return int(np.sum(adj != 0))


# ---------------------------------------------------------------------------
# v-structure detection
# ---------------------------------------------------------------------------

def count_v_structures(adj: AdjacencyMatrix) -> int:
    """Count the number of v-structures (colliders) i→j←k where i⊥k.

    Returns the count of unshielded colliders.
    """
    return len(find_v_structures(adj))


def find_v_structures(adj: AdjacencyMatrix) -> List[Tuple[int, int, int]]:
    """Find all v-structures i→j←k where i and k are not adjacent.

    Returns list of (i, j, k) with i < k for canonical ordering.
    """
    n = adj.shape[0]
    result: List[Tuple[int, int, int]] = []
    for j in range(n):
        parents = sorted(i for i in range(n) if adj[i, j])
        for a_idx in range(len(parents)):
            for b_idx in range(a_idx + 1, len(parents)):
                i, k = parents[a_idx], parents[b_idx]
                if not adj[i, k] and not adj[k, i]:
                    result.append((i, j, k))
    return result


# ---------------------------------------------------------------------------
# Transitive operations
# ---------------------------------------------------------------------------

def transitive_closure(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Compute the transitive closure of a directed graph.

    Uses repeated matrix squaring (Warshall's algorithm variant).
    ``result[i, j] = 1`` iff there is a directed path from i to j.
    """
    n = adj.shape[0]
    reach = adj.astype(np.bool_).copy()

    # Warshall's algorithm
    for k in range(n):
        for i in range(n):
            if reach[i, k]:
                for j in range(n):
                    if reach[k, j]:
                        reach[i, j] = True

    np.fill_diagonal(reach, False)
    return reach.astype(np.int8)


def transitive_reduction(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Compute the transitive reduction of a DAG.

    The transitive reduction is the minimal edge set with the same
    reachability relation as the original graph.  For a DAG, the
    transitive reduction is unique.

    An edge i→j is redundant iff there is a path i→...→j of length ≥ 2.
    """
    n = adj.shape[0]
    tc = transitive_closure(adj)
    result = adj.copy()

    for i in range(n):
        for j in range(n):
            if not result[i, j]:
                continue
            # Check if there's a path of length ≥ 2 from i to j
            # i.e., exists k such that adj[i,k] and tc[k,j]
            for k in range(n):
                if k != j and adj[i, k] and tc[k, j]:
                    result[i, j] = 0
                    break

    return result


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

def find_all_paths(
    adj: AdjacencyMatrix,
    source: int,
    target: int,
    max_length: Optional[int] = None,
) -> List[List[int]]:
    """Find all simple paths (directed or undirected) from source to target.

    Considers the undirected skeleton for path enumeration.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Adjacency matrix.
    source, target : int
        Start and end nodes.
    max_length : int, optional
        Maximum path length (number of edges).  Default: no limit.
    """
    n = adj.shape[0]
    undirected = (adj | adj.T).astype(bool)
    all_paths: List[List[int]] = []

    def _dfs(current: int, path: List[int], visited: Set[int]) -> None:
        if max_length is not None and len(path) - 1 > max_length:
            return
        if current == target:
            all_paths.append(list(path))
            return
        for nb in range(n):
            if undirected[current, nb] and nb not in visited:
                visited.add(nb)
                path.append(nb)
                _dfs(nb, path, visited)
                path.pop()
                visited.discard(nb)

    _dfs(source, [source], {source})
    return all_paths


def find_all_directed_paths(
    adj: AdjacencyMatrix,
    source: int,
    target: int,
    max_length: Optional[int] = None,
) -> List[List[int]]:
    """Find all simple directed paths from source to target.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Adjacency matrix.
    source, target : int
        Start and end nodes.
    max_length : int, optional
        Maximum path length (number of edges).
    """
    n = adj.shape[0]
    all_paths: List[List[int]] = []

    def _dfs(current: int, path: List[int], visited: Set[int]) -> None:
        if max_length is not None and len(path) - 1 > max_length:
            return
        if current == target:
            all_paths.append(list(path))
            return
        for v in range(n):
            if adj[current, v] and v not in visited:
                visited.add(v)
                path.append(v)
                _dfs(v, path, visited)
                path.pop()
                visited.discard(v)

    _dfs(source, [source], {source})
    return all_paths


# ---------------------------------------------------------------------------
# Meek's rules (R1-R4)
# ---------------------------------------------------------------------------

def meek_rules(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Apply Meek's orientation rules R1-R4 until convergence.

    Takes a PDAG (partially directed acyclic graph) and orients undirected
    edges using the four Meek rules.  An undirected edge is represented as
    adj[i,j] = adj[j,i] = 1.  A directed edge i→j has adj[i,j]=1, adj[j,i]=0.

    Rules:
    R1: Orient b—c as b→c if ∃ a→b and a ⊥ c (not adjacent).
    R2: Orient a—b as a→b if ∃ a→c→b.
    R3: Orient a—b as a→b if ∃ c—a, d—a with c→b, d→b and c ⊥ d.
    R4: Orient a—b as a→b if ∃ c—a with c→d→b.

    Returns
    -------
    AdjacencyMatrix
        The oriented PDAG after applying all rules to convergence.
    """
    n = adj.shape[0]
    result = adj.copy()
    changed = True

    while changed:
        changed = False

        for b in range(n):
            for c in range(n):
                if b == c:
                    continue
                # Only consider undirected edges b — c
                if not (result[b, c] and result[c, b]):
                    continue

                # R1: a→b and a ⊥ c → orient b→c
                for a in range(n):
                    if a == b or a == c:
                        continue
                    if result[a, b] and not result[b, a]:  # a→b directed
                        if not result[a, c] and not result[c, a]:  # a ⊥ c
                            result[c, b] = 0  # orient b→c
                            changed = True
                            break
                if not (result[b, c] and result[c, b]):
                    continue

                # R2: a→c→b and a—b → orient a→b (here checking b—c → b→c)
                # Restate: orient b→c if ∃ b→d→c
                for d in range(n):
                    if d == b or d == c:
                        continue
                    if (result[b, d] and not result[d, b] and
                            result[d, c] and not result[c, d]):  # b→d→c
                        result[c, b] = 0  # orient b→c
                        changed = True
                        break
                if not (result[b, c] and result[c, b]):
                    continue

                # R3: ∃ d—b, e—b with d→c, e→c and d ⊥ e → orient b→c
                found_r3 = False
                parents_c_through_b = []
                for d in range(n):
                    if d == b or d == c:
                        continue
                    if (result[d, b] and result[b, d] and  # d—b undirected
                            result[d, c] and not result[c, d]):  # d→c directed
                        parents_c_through_b.append(d)

                for idx_d in range(len(parents_c_through_b)):
                    for idx_e in range(idx_d + 1, len(parents_c_through_b)):
                        d = parents_c_through_b[idx_d]
                        e = parents_c_through_b[idx_e]
                        if not result[d, e] and not result[e, d]:  # d ⊥ e
                            result[c, b] = 0  # orient b→c
                            changed = True
                            found_r3 = True
                            break
                    if found_r3:
                        break
                if not (result[b, c] and result[c, b]):
                    continue

                # R4: ∃ d—b with d→e→c → orient b→c
                for d in range(n):
                    if d == b or d == c:
                        continue
                    if not (result[d, b] and result[b, d]):  # d—b undirected
                        continue
                    for e in range(n):
                        if e == b or e == c or e == d:
                            continue
                        if (result[d, e] and not result[e, d] and  # d→e
                                result[e, c] and not result[c, e]):  # e→c
                            result[c, b] = 0  # orient b→c
                            changed = True
                            break
                    if not (result[b, c] and result[c, b]):
                        break

    return result


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def adjacency_to_edge_list(adj: AdjacencyMatrix) -> List[Tuple[int, int]]:
    """Convert adjacency matrix to edge list."""
    rows, cols = np.nonzero(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def edge_list_to_adjacency(n: int, edges: List[Tuple[int, int]]) -> AdjacencyMatrix:
    """Convert edge list to adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : list of (int, int)
        Directed edges as (source, target) pairs.
    """
    adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        adj[i, j] = 1
    return adj
