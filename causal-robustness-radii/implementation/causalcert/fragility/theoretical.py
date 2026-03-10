"""
Theoretical (data-free) fragility analysis.

Computes fragility scores using only the DAG structure, without access to
observational data.  Uses graph-theoretic properties such as bridge edges,
articulation points, vertex cuts, and structural centrality to identify
which edges are most critical for the causal conclusion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EdgeTuple,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parents(adj: np.ndarray, v: int) -> list[int]:
    return [int(p) for p in np.nonzero(adj[:, v])[0]]


def _children(adj: np.ndarray, v: int) -> list[int]:
    return [int(c) for c in np.nonzero(adj[v, :])[0]]


def _descendants_of(adj: np.ndarray, v: int) -> set[int]:
    visited: set[int] = set()
    queue = deque(_children(adj, v))
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(c for c in _children(adj, node) if c not in visited)
    return visited


def _ancestors_of(adj: np.ndarray, v: int) -> set[int]:
    visited: set[int] = set()
    queue = deque(_parents(adj, v))
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(p for p in _parents(adj, node) if p not in visited)
    return visited


def _has_directed_path(adj: np.ndarray, source: int, target: int) -> bool:
    if source == target:
        return True
    visited: set[int] = set()
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for c in _children(adj, node):
            if c == target:
                return True
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return False


def _is_dag(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


# ---------------------------------------------------------------------------
# Bridge edge detection
# ---------------------------------------------------------------------------


def skeleton_bridge_edges(adj: AdjacencyMatrix) -> list[EdgeTuple]:
    """Find bridge edges in the DAG skeleton (undirected version).

    A bridge edge is one whose removal disconnects the skeleton graph.
    Uses Tarjan's bridge-finding algorithm on the undirected skeleton.

    Returns
    -------
    list[EdgeTuple]
        Directed edges whose skeleton-equivalent is a bridge.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    skeleton = (adj | adj.T).astype(np.int8)

    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges_undirected: set[tuple[int, int]] = set()
    timer = [0]

    def _dfs(u: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in range(n):
            if not skeleton[u, v]:
                continue
            if disc[v] == -1:
                parent[v] = u
                _dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges_undirected.add((min(u, v), max(u, v)))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            _dfs(i)

    # Map undirected bridges back to directed edges
    result: list[EdgeTuple] = []
    for u in range(n):
        for v in range(n):
            if adj[u, v]:
                key = (min(u, v), max(u, v))
                if key in bridges_undirected:
                    result.append((u, v))
    return result


def is_bridge_edge(
    adj: AdjacencyMatrix, source: NodeId, target: NodeId
) -> bool:
    """Check whether removing edge source→target disconnects the skeleton."""
    bridges = skeleton_bridge_edges(adj)
    return (source, target) in bridges


# ---------------------------------------------------------------------------
# Articulation points
# ---------------------------------------------------------------------------


def articulation_points(adj: AdjacencyMatrix) -> list[NodeId]:
    """Find articulation points in the DAG skeleton.

    An articulation point is a node whose removal disconnects the skeleton.

    Returns
    -------
    list[NodeId]
        Sorted list of articulation point node ids.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    skeleton = (adj | adj.T).astype(np.int8)

    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    ap: set[int] = set()
    timer = [0]

    def _dfs(u: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        child_count = 0
        for v in range(n):
            if not skeleton[u, v]:
                continue
            if disc[v] == -1:
                child_count += 1
                parent[v] = u
                _dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and child_count > 1:
                    ap.add(u)
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            _dfs(i)

    return sorted(ap)


def causal_articulation_points(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[NodeId]:
    """Find articulation points on directed paths from treatment to outcome.

    These are nodes whose removal would sever all directed paths from
    treatment to outcome.
    """
    adj = np.asarray(adj, dtype=np.int8)
    # Collect all nodes on any directed path from treatment to outcome
    path_nodes: set[int] = set()
    stack: list[tuple[int, list[int]]] = [(treatment, [treatment])]
    while stack:
        node, path = stack.pop()
        for ch in _children(adj, node):
            if ch in path:
                continue
            new_path = path + [ch]
            if ch == outcome:
                path_nodes.update(new_path)
            else:
                stack.append((ch, new_path))

    # Remove treatment and outcome themselves
    interior = path_nodes - {treatment, outcome}

    # Check which interior nodes are essential
    essential: list[int] = []
    for v in sorted(interior):
        # Remove node v and check if treatment→outcome path still exists
        mask = np.ones(adj.shape[0], dtype=bool)
        mask[v] = False
        indices = np.where(mask)[0]
        sub_adj = adj[np.ix_(indices, indices)]
        # Map treatment and outcome to new indices
        old_to_new = {old: new for new, old in enumerate(indices)}
        if treatment not in old_to_new or outcome not in old_to_new:
            essential.append(v)
            continue
        new_t = old_to_new[treatment]
        new_o = old_to_new[outcome]
        if not _has_directed_path(sub_adj, new_t, new_o):
            essential.append(v)

    return essential


# ---------------------------------------------------------------------------
# Vertex cuts for treatment-outcome connectivity
# ---------------------------------------------------------------------------


def minimum_vertex_cut(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[NodeSet]:
    """Find all minimum vertex cuts separating treatment from outcome.

    A vertex cut is a set of nodes whose removal disconnects all directed
    paths from treatment to outcome.  Returns all minimum-size cuts.

    Uses brute-force enumeration (suitable for small DAGs).
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    if not _has_directed_path(adj, treatment, outcome):
        return [frozenset()]

    interior = [i for i in range(n) if i != treatment and i != outcome]

    for size in range(len(interior) + 1):
        cuts: list[NodeSet] = []
        for combo in combinations(interior, size):
            cut_set = set(combo)
            # Check if removing cut_set disconnects treatment from outcome
            mask = np.ones(n, dtype=bool)
            for v in cut_set:
                mask[v] = False
            indices = np.where(mask)[0]
            sub_adj = adj[np.ix_(indices, indices)]
            old_to_new = {old: new for new, old in enumerate(indices)}
            if treatment in old_to_new and outcome in old_to_new:
                new_t = old_to_new[treatment]
                new_o = old_to_new[outcome]
                if not _has_directed_path(sub_adj, new_t, new_o):
                    cuts.append(frozenset(combo))
            else:
                cuts.append(frozenset(combo))
        if cuts:
            return cuts

    return []


def minimum_edge_cut(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[list[EdgeTuple]]:
    """Find all minimum edge cuts separating treatment from outcome.

    An edge cut is a set of edges whose removal disconnects all directed
    paths from treatment to outcome.
    """
    adj = np.asarray(adj, dtype=np.int8)

    if not _has_directed_path(adj, treatment, outcome):
        return [[]]

    # Collect all edges on paths from treatment to outcome
    edge_list: list[EdgeTuple] = []
    rows, cols = np.nonzero(adj)
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        if (_has_directed_path(adj, treatment, r) or r == treatment) and \
           (_has_directed_path(adj, c, outcome) or c == outcome):
            edge_list.append((r, c))

    for size in range(1, len(edge_list) + 1):
        cuts: list[list[EdgeTuple]] = []
        for combo in combinations(edge_list, size):
            test_adj = adj.copy()
            for u, v in combo:
                test_adj[u, v] = 0
            if not _has_directed_path(test_adj, treatment, outcome):
                cuts.append(list(combo))
        if cuts:
            return cuts

    return []


# ---------------------------------------------------------------------------
# Structural fragility scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TheoreticalFragilityResult:
    """Result of theoretical fragility analysis for one edge."""
    edge: EdgeTuple
    bridge_score: float
    centrality_score: float
    cut_score: float
    path_fraction: float
    total_score: float


def edge_betweenness_centrality(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> dict[EdgeTuple, float]:
    """Compute edge betweenness centrality w.r.t. treatment-outcome paths.

    For each edge, counts the fraction of treatment→outcome directed paths
    that use it.
    """
    adj = np.asarray(adj, dtype=np.int8)
    from causalcert.dag.paths import all_directed_paths
    paths = all_directed_paths(adj, treatment, outcome)
    if not paths:
        return {}

    edge_count: dict[EdgeTuple, int] = {}
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_count[edge] = edge_count.get(edge, 0) + 1

    n_paths = len(paths)
    return {e: c / n_paths for e, c in edge_count.items()}


def theoretical_fragility_scores(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    bridge_weight: float = 0.3,
    centrality_weight: float = 0.4,
    cut_weight: float = 0.3,
) -> list[TheoreticalFragilityResult]:
    """Compute theoretical fragility scores for all edges.

    Combines three graph-theoretic measures:
    1. Bridge score: 1.0 if the edge is a skeleton bridge, else 0.0
    2. Centrality score: edge betweenness w.r.t. treatment-outcome paths
    3. Cut score: 1/|min_cut| if edge is part of a minimum edge cut

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    bridge_weight, centrality_weight, cut_weight : float
        Weights for the three components.

    Returns
    -------
    list[TheoreticalFragilityResult]
        Sorted by total_score descending.
    """
    adj = np.asarray(adj, dtype=np.int8)

    bridges = set(skeleton_bridge_edges(adj))
    centrality = edge_betweenness_centrality(adj, treatment, outcome)
    min_cuts = minimum_edge_cut(adj, treatment, outcome)
    min_cut_size = len(min_cuts[0]) if min_cuts and min_cuts[0] else 0
    cut_edges: set[EdgeTuple] = set()
    for cut in min_cuts:
        for e in cut:
            cut_edges.add(e)

    results: list[TheoreticalFragilityResult] = []
    rows, cols = np.nonzero(adj)
    for r, c in zip(rows, cols):
        edge = (int(r), int(c))
        b_score = 1.0 if edge in bridges else 0.0
        c_score = centrality.get(edge, 0.0)
        cu_score = (1.0 / min_cut_size) if (min_cut_size > 0 and edge in cut_edges) else 0.0
        total = bridge_weight * b_score + centrality_weight * c_score + cut_weight * cu_score
        path_frac = c_score  # same as betweenness
        results.append(TheoreticalFragilityResult(
            edge=edge,
            bridge_score=b_score,
            centrality_score=c_score,
            cut_score=cu_score,
            path_fraction=path_frac,
            total_score=total,
        ))

    results.sort(key=lambda x: x.total_score, reverse=True)
    return results


def structural_fragility_ranking(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[FragilityScore]:
    """Convert theoretical fragility to standard FragilityScore format.

    This allows theoretical scores to be used interchangeably with
    data-driven fragility scores in the pipeline.
    """
    theo = theoretical_fragility_scores(adj, treatment, outcome)
    result: list[FragilityScore] = []
    for tf in theo:
        channels = {
            FragilityChannel.D_SEPARATION: tf.bridge_score,
            FragilityChannel.IDENTIFICATION: tf.centrality_score,
            FragilityChannel.ESTIMATION: tf.cut_score,
        }
        result.append(FragilityScore(
            edge=tf.edge,
            total_score=tf.total_score,
            channel_scores=channels,
        ))
    return result


# ---------------------------------------------------------------------------
# Causal effect identifiability analysis
# ---------------------------------------------------------------------------


def is_identifiable_via_backdoor(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> bool:
    """Check if the causal effect is identifiable via the backdoor criterion.

    A valid adjustment set exists if there is a set S that:
    1. Does not contain any descendant of treatment
    2. Blocks all backdoor paths from treatment to outcome
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    desc_t = _descendants_of(adj, treatment)
    candidates = [i for i in range(n) if i != treatment and i != outcome
                  and i not in desc_t]

    # Check all subsets
    for size in range(len(candidates) + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            # Check if S blocks all backdoor paths
            # Use d-separation: X ⊥ Y | S in the manipulated graph
            # (graph with outgoing edges from X removed)
            manip_adj = adj.copy()
            manip_adj[treatment, :] = 0  # remove outgoing edges from treatment
            from causalcert.dag.dsep import DSeparationOracle
            oracle = DSeparationOracle(manip_adj)
            if oracle.is_d_separated(treatment, outcome, s):
                return True
    return False


def edge_removal_identifiability_impact(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> dict[EdgeTuple, bool]:
    """Check whether removing each edge changes identifiability status.

    Returns a dict mapping edge → True if removing it changes whether
    the causal effect is identifiable.
    """
    adj = np.asarray(adj, dtype=np.int8)
    base_id = is_identifiable_via_backdoor(adj, treatment, outcome)

    impact: dict[EdgeTuple, bool] = {}
    rows, cols = np.nonzero(adj)
    for r, c in zip(rows, cols):
        edge = (int(r), int(c))
        test_adj = adj.copy()
        test_adj[r, c] = 0
        new_id = is_identifiable_via_backdoor(test_adj, treatment, outcome)
        impact[edge] = (new_id != base_id)

    return impact


# ---------------------------------------------------------------------------
# Structural bounds on fragility
# ---------------------------------------------------------------------------


def structural_robustness_lower_bound(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> int:
    """Compute a lower bound on the robustness radius from graph structure.

    The minimum edge cut size between treatment and outcome gives a lower
    bound on the number of edits needed to sever all causal paths.
    """
    min_cuts = minimum_edge_cut(adj, treatment, outcome)
    if not min_cuts or not min_cuts[0]:
        return 0
    return len(min_cuts[0])


def structural_robustness_upper_bound(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> int:
    """Compute an upper bound on the robustness radius from graph structure.

    The number of edges on the shortest directed path gives an upper bound,
    since removing all edges on any single path is sufficient.
    """
    from causalcert.dag.paths import shortest_directed_path
    sp = shortest_directed_path(adj, treatment, outcome)
    if sp is None:
        return 0
    return len(sp) - 1
