"""
Path algorithms for causal DAGs.

Provides directed path enumeration, backdoor path detection, mediation path
extraction, and path blocking analysis.  All algorithms operate directly on
NumPy adjacency matrices for consistency with the rest of CausalCert.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parents(adj: np.ndarray, v: int) -> list[int]:
    """Return parents of *v*."""
    return [int(p) for p in np.nonzero(adj[:, v])[0]]


def _children(adj: np.ndarray, v: int) -> list[int]:
    """Return children of *v*."""
    return [int(c) for c in np.nonzero(adj[v, :])[0]]


# ---------------------------------------------------------------------------
# Directed path enumeration
# ---------------------------------------------------------------------------


def all_directed_paths(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
    max_length: int | None = None,
) -> list[list[NodeId]]:
    """Enumerate all directed paths from *source* to *target*.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    source, target : NodeId
        Start and end nodes.
    max_length : int | None
        Maximum path length (number of edges). ``None`` means no limit.

    Returns
    -------
    list[list[NodeId]]
        Each path is a list of node ids from *source* to *target*.
    """
    adj = np.asarray(adj, dtype=np.int8)
    if source == target:
        return [[source]]

    n = adj.shape[0]
    if max_length is None:
        max_length = n

    paths: list[list[NodeId]] = []
    stack: list[tuple[int, list[int]]] = [(source, [source])]

    while stack:
        node, path = stack.pop()
        if len(path) - 1 >= max_length:
            continue
        for child in _children(adj, node):
            if child in path:
                continue
            new_path = path + [child]
            if child == target:
                paths.append(new_path)
            else:
                stack.append((child, new_path))

    return paths


def has_directed_path(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
) -> bool:
    """Check whether a directed path from *source* to *target* exists."""
    adj = np.asarray(adj, dtype=np.int8)
    if source == target:
        return True
    visited: set[int] = set()
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for child in _children(adj, node):
            if child == target:
                return True
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return False


def shortest_directed_path(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
) -> list[NodeId] | None:
    """Return a shortest directed path or ``None`` if unreachable."""
    adj = np.asarray(adj, dtype=np.int8)
    if source == target:
        return [source]
    visited: set[int] = {source}
    queue: deque[tuple[int, list[int]]] = deque([(source, [source])])
    while queue:
        node, path = queue.popleft()
        for child in _children(adj, node):
            if child == target:
                return path + [child]
            if child not in visited:
                visited.add(child)
                queue.append((child, path + [child]))
    return None


def longest_directed_path(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
) -> list[NodeId] | None:
    """Return a longest simple directed path (via DFS), or ``None``."""
    all_p = all_directed_paths(adj, source, target)
    if not all_p:
        return None
    return max(all_p, key=len)


def directed_path_lengths(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
) -> list[int]:
    """Return sorted list of all distinct directed-path lengths."""
    paths = all_directed_paths(adj, source, target)
    lengths = sorted({len(p) - 1 for p in paths})
    return lengths


# ---------------------------------------------------------------------------
# Backdoor paths
# ---------------------------------------------------------------------------


def _all_undirected_paths(
    adj: np.ndarray,
    source: int,
    target: int,
    max_length: int | None = None,
) -> list[list[int]]:
    """Enumerate all simple paths in the skeleton (undirected version)."""
    n = adj.shape[0]
    skeleton = adj | adj.T
    if max_length is None:
        max_length = n

    paths: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if len(path) - 1 >= max_length:
            continue
        for nb in np.nonzero(skeleton[node])[0]:
            nb = int(nb)
            if nb in path:
                continue
            new_path = path + [nb]
            if nb == target:
                paths.append(new_path)
            else:
                stack.append((nb, new_path))
    return paths


def backdoor_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    max_length: int | None = None,
) -> list[list[NodeId]]:
    """Enumerate all backdoor paths between *treatment* and *outcome*.

    A backdoor path is any path from treatment to outcome that begins with
    an arrow *into* treatment (i.e. starts with treatment ← ...).  We
    enumerate all simple undirected paths and filter those whose first edge
    is an incoming edge to the treatment node.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    max_length : int | None
        Maximum path length.

    Returns
    -------
    list[list[NodeId]]
        Backdoor paths (node sequences from treatment to outcome).
    """
    adj = np.asarray(adj, dtype=np.int8)
    all_paths = _all_undirected_paths(adj, treatment, outcome, max_length)
    bd_paths: list[list[NodeId]] = []
    for path in all_paths:
        if len(path) < 2:
            continue
        first_nb = path[1]
        # Backdoor: first edge goes INTO treatment, i.e. first_nb -> treatment
        if adj[first_nb, treatment]:
            bd_paths.append(path)
    return bd_paths


def causal_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[list[NodeId]]:
    """Return all directed (causal) paths from treatment to outcome."""
    return all_directed_paths(adj, treatment, outcome)


# ---------------------------------------------------------------------------
# Path blocking
# ---------------------------------------------------------------------------


def _is_collider_on_path(
    adj: np.ndarray, path: list[int], idx: int
) -> bool:
    """Check whether node at position *idx* is a collider on *path*."""
    if idx <= 0 or idx >= len(path) - 1:
        return False
    prev_node = path[idx - 1]
    curr_node = path[idx]
    next_node = path[idx + 1]
    return bool(adj[prev_node, curr_node]) and bool(adj[next_node, curr_node])


def is_path_blocked(
    adj: AdjacencyMatrix,
    path: list[NodeId],
    conditioning: NodeSet,
) -> bool:
    """Check whether a path is blocked by the conditioning set.

    A path is blocked if there exists an intermediate node that is either:
      - A non-collider that is in the conditioning set, OR
      - A collider such that neither it nor any of its descendants are in
        the conditioning set.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    path : list[NodeId]
        A path (sequence of node ids).
    conditioning : NodeSet
        Conditioning set.

    Returns
    -------
    bool
        ``True`` if the path is blocked.
    """
    adj = np.asarray(adj, dtype=np.int8)
    cond = set(conditioning)

    if len(path) <= 2:
        # Direct edge path with no intermediate nodes — never blocked
        # by conditioning on intermediates
        return False

    # Precompute descendants of conditioning set members
    desc_of_cond: set[int] = set()
    queue = deque(cond)
    visited_desc: set[int] = set(cond)
    while queue:
        node = queue.popleft()
        for c in _children(adj, node):
            if c not in visited_desc:
                visited_desc.add(c)
                queue.append(c)
    # Actually we need ancestors of cond for collider logic; correct approach:
    # a collider is active iff the collider or any descendant is in cond
    # So we need: is there any descendant of path[idx] that is in cond?

    for idx in range(1, len(path) - 1):
        node = path[idx]
        is_collider = _is_collider_on_path(adj, path, idx)

        if is_collider:
            # Collider: path blocked unless node or a descendant is conditioned
            # Check if node or any descendant is in conditioning
            if node in cond:
                continue  # active collider
            # BFS descendants of node
            desc_queue = deque([node])
            desc_visited: set[int] = {node}
            activated = False
            while desc_queue and not activated:
                d = desc_queue.popleft()
                for ch in _children(adj, d):
                    if ch in cond:
                        activated = True
                        break
                    if ch not in desc_visited:
                        desc_visited.add(ch)
                        desc_queue.append(ch)
            if not activated:
                return True  # collider not activated → path blocked
        else:
            # Non-collider: path blocked if node is conditioned
            if node in cond:
                return True

    return False


def count_blocked_paths(
    adj: AdjacencyMatrix,
    paths: list[list[NodeId]],
    conditioning: NodeSet,
) -> int:
    """Count how many paths are blocked by the conditioning set."""
    return sum(1 for p in paths if is_path_blocked(adj, p, conditioning))


def all_open_paths(
    adj: AdjacencyMatrix,
    source: NodeId,
    target: NodeId,
    conditioning: NodeSet,
) -> list[list[NodeId]]:
    """Return all paths from source to target that are NOT blocked."""
    adj_np = np.asarray(adj, dtype=np.int8)
    all_p = _all_undirected_paths(adj_np, source, target)
    return [p for p in all_p if not is_path_blocked(adj_np, p, conditioning)]


# ---------------------------------------------------------------------------
# Mediation paths
# ---------------------------------------------------------------------------


def mediation_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    mediator: NodeId,
    outcome: NodeId,
) -> list[list[NodeId]]:
    """Enumerate paths from treatment → mediator → outcome.

    Returns all directed paths from treatment to outcome that pass through
    the mediator.
    """
    adj = np.asarray(adj, dtype=np.int8)
    t_to_m = all_directed_paths(adj, treatment, mediator)
    m_to_y = all_directed_paths(adj, mediator, outcome)

    results: list[list[NodeId]] = []
    for p1 in t_to_m:
        for p2 in m_to_y:
            # p2 starts with mediator, so skip first element to avoid dup
            combined = p1 + p2[1:]
            # Check no repeated nodes (simple path)
            if len(set(combined)) == len(combined):
                results.append(combined)
    return results


def direct_effect_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    mediators: NodeSet,
) -> list[list[NodeId]]:
    """Return directed paths that do NOT pass through any mediator."""
    all_p = all_directed_paths(adj, treatment, outcome)
    return [p for p in all_p if not (set(p[1:-1]) & set(mediators))]


def indirect_effect_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    mediators: NodeSet,
) -> list[list[NodeId]]:
    """Return directed paths that pass through at least one mediator."""
    all_p = all_directed_paths(adj, treatment, outcome)
    return [p for p in all_p if set(p[1:-1]) & set(mediators)]


# ---------------------------------------------------------------------------
# Instrumental variable paths
# ---------------------------------------------------------------------------


def instrument_paths(
    adj: AdjacencyMatrix,
    instrument: NodeId,
    treatment: NodeId,
    outcome: NodeId,
) -> dict[str, list[list[NodeId]]]:
    """Analyze paths relevant to an instrumental variable.

    Returns a dict with:
      - 'instrument_to_treatment': directed paths Z → X
      - 'instrument_to_outcome_direct': directed paths Z → Y not through X
      - 'treatment_to_outcome': directed paths X → Y
    """
    adj = np.asarray(adj, dtype=np.int8)
    z_to_x = all_directed_paths(adj, instrument, treatment)
    z_to_y = all_directed_paths(adj, instrument, outcome)
    x_to_y = all_directed_paths(adj, treatment, outcome)

    z_to_y_direct = [
        p for p in z_to_y if treatment not in p[1:-1]
    ]

    return {
        "instrument_to_treatment": z_to_x,
        "instrument_to_outcome_direct": z_to_y_direct,
        "treatment_to_outcome": x_to_y,
    }


# ---------------------------------------------------------------------------
# Path-based graph metrics
# ---------------------------------------------------------------------------


def all_pairs_shortest_path_lengths(adj: AdjacencyMatrix) -> np.ndarray:
    """Compute shortest directed-path lengths between all pairs.

    Uses BFS from each node. Unreachable pairs get length -1.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    dist = np.full((n, n), -1, dtype=int)
    for s in range(n):
        dist[s, s] = 0
        visited: set[int] = {s}
        queue: deque[tuple[int, int]] = deque([(s, 0)])
        while queue:
            node, d = queue.popleft()
            for ch in _children(adj, node):
                if ch not in visited:
                    visited.add(ch)
                    dist[s, ch] = d + 1
                    queue.append((ch, d + 1))
    return dist


def dag_diameter(adj: AdjacencyMatrix) -> int:
    """Return the diameter of the DAG (longest shortest directed path).

    Returns 0 if the DAG has no edges, or -1 if there are unreachable pairs.
    """
    dists = all_pairs_shortest_path_lengths(adj)
    reachable = dists[dists > 0]
    if len(reachable) == 0:
        return 0
    return int(reachable.max())


def path_count_matrix(adj: AdjacencyMatrix) -> np.ndarray:
    """Compute the number of directed paths between all pairs via DP.

    Uses topological order: count[u][v] = sum_{w: child of u} count[w][v].
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    count = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        count[i, i] = 1

    # Topological sort
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    topo: list[int] = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in _children(adj, v):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    # Process in reverse topological order
    for v in reversed(topo):
        for c in _children(adj, v):
            count[v] += count[c]

    return count


def reachability_matrix(adj: AdjacencyMatrix) -> np.ndarray:
    """Compute the transitive closure (boolean reachability matrix)."""
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    reach = np.eye(n, dtype=bool)
    for i in range(n):
        visited: set[int] = {i}
        queue = deque([i])
        while queue:
            node = queue.popleft()
            for c in _children(adj, node):
                if c not in visited:
                    visited.add(c)
                    reach[i, c] = True
                    queue.append(c)
    return reach
