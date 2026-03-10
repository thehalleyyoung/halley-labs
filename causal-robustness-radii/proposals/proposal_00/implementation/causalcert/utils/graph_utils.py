"""Graph-theoretic utilities for DAGs and undirected graphs.

Operates on NumPy adjacency matrices.  For full causal-DAG semantics
(d-separation, moral graphs, MEC) see :mod:`causalcert.dag`.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

IntMatrix = NDArray[np.integer]
NodeSet = set[int]


# ====================================================================
# 1. Path Finding
# ====================================================================

def bfs_shortest_path(
    adj: IntMatrix,
    source: int,
    target: int,
) -> list[int] | None:
    """Return the shortest directed path from *source* to *target*, or None."""
    n = adj.shape[0]
    visited = [False] * n
    parent: dict[int, int] = {}
    queue: deque[int] = deque([source])
    visited[source] = True

    while queue:
        node = queue.popleft()
        if node == target:
            path = []
            cur = target
            while cur != source:
                path.append(cur)
                cur = parent[cur]
            path.append(source)
            return path[::-1]
        for nbr in range(n):
            if adj[node, nbr] and not visited[nbr]:
                visited[nbr] = True
                parent[nbr] = node
                queue.append(nbr)
    return None


def all_simple_paths(
    adj: IntMatrix,
    source: int,
    target: int,
    max_depth: int | None = None,
) -> list[list[int]]:
    """Enumerate all simple directed paths from *source* to *target*.

    Warning: exponential in the worst case.  Use *max_depth* to limit.
    """
    n = adj.shape[0]
    limit = n if max_depth is None else max_depth
    results: list[list[int]] = []

    def _dfs(node: int, path: list[int], visited: set[int]) -> None:
        if len(path) > limit + 1:
            return
        if node == target:
            results.append(list(path))
            return
        for nbr in range(n):
            if adj[node, nbr] and nbr not in visited:
                visited.add(nbr)
                path.append(nbr)
                _dfs(nbr, path, visited)
                path.pop()
                visited.discard(nbr)

    _dfs(source, [source], {source})
    return results


def all_directed_paths_dag(
    adj: IntMatrix,
    source: int,
    target: int,
) -> list[list[int]]:
    """Enumerate directed paths in a DAG (uses topological order for speed)."""
    order = topological_sort(adj)
    if order is None:
        return all_simple_paths(adj, source, target)

    src_pos = order.index(source)
    tgt_pos = order.index(target)
    if src_pos >= tgt_pos:
        return []

    # Only need nodes between source and target in topo order
    relevant = set(order[src_pos: tgt_pos + 1])
    sub_adj = adj.copy()
    for i in range(adj.shape[0]):
        if i not in relevant:
            sub_adj[i, :] = 0
            sub_adj[:, i] = 0

    return all_simple_paths(sub_adj, source, target)


def shortest_path_length(adj: IntMatrix, source: int, target: int) -> int:
    """Return the length of the shortest directed path, or -1 if unreachable."""
    path = bfs_shortest_path(adj, source, target)
    return len(path) - 1 if path else -1


# ====================================================================
# 2. Reachability
# ====================================================================

def descendants(adj: IntMatrix, node: int) -> NodeSet:
    """Return all descendants of *node* (excluding *node* itself)."""
    n = adj.shape[0]
    visited: NodeSet = set()
    queue: deque[int] = deque()
    for nbr in range(n):
        if adj[node, nbr]:
            queue.append(nbr)
            visited.add(nbr)
    while queue:
        cur = queue.popleft()
        for nbr in range(n):
            if adj[cur, nbr] and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


def ancestors(adj: IntMatrix, node: int) -> NodeSet:
    """Return all ancestors of *node* (excluding *node* itself)."""
    n = adj.shape[0]
    visited: NodeSet = set()
    queue: deque[int] = deque()
    for par in range(n):
        if adj[par, node]:
            queue.append(par)
            visited.add(par)
    while queue:
        cur = queue.popleft()
        for par in range(n):
            if adj[par, cur] and par not in visited:
                visited.add(par)
                queue.append(par)
    return visited


def is_ancestor(adj: IntMatrix, u: int, v: int) -> bool:
    """True if *u* is an ancestor of *v*."""
    return u in ancestors(adj, v)


def reachable_from(adj: IntMatrix, sources: set[int]) -> NodeSet:
    """All nodes reachable from *sources* via directed edges."""
    visited = set(sources)
    queue = deque(sources)
    n = adj.shape[0]
    while queue:
        cur = queue.popleft()
        for nbr in range(n):
            if adj[cur, nbr] and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


# ====================================================================
# 3. Connected Components
# ====================================================================

def undirected_skeleton(adj: IntMatrix) -> IntMatrix:
    """Return the undirected skeleton (symmetric adjacency matrix)."""
    return np.maximum(adj, adj.T).astype(adj.dtype)


def connected_components(adj: IntMatrix) -> list[NodeSet]:
    """Return connected components of the undirected skeleton."""
    skeleton = undirected_skeleton(adj)
    n = skeleton.shape[0]
    visited = [False] * n
    components: list[NodeSet] = []

    for start in range(n):
        if visited[start]:
            continue
        comp: NodeSet = set()
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            comp.add(node)
            for nbr in range(n):
                if skeleton[node, nbr] and not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        components.append(comp)

    return components


def is_connected(adj: IntMatrix) -> bool:
    """True if the undirected skeleton is connected."""
    return len(connected_components(adj)) == 1


def strongly_connected_components(adj: IntMatrix) -> list[NodeSet]:
    """Tarjan's SCC algorithm for the directed graph."""
    n = adj.shape[0]
    index_counter = [0]
    stack: list[int] = []
    lowlink = [0] * n
    index = [0] * n
    on_stack = [False] * n
    index_initialized = [False] * n
    result: list[NodeSet] = []

    def _strongconnect(v: int) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        index_initialized[v] = True
        stack.append(v)
        on_stack[v] = True

        for w in range(n):
            if not adj[v, w]:
                continue
            if not index_initialized[w]:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: NodeSet = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.add(w)
                if w == v:
                    break
            result.append(component)

    for v in range(n):
        if not index_initialized[v]:
            _strongconnect(v)

    return result


# ====================================================================
# 4. Graph Metrics
# ====================================================================

def density(adj: IntMatrix) -> float:
    """Edge density |E| / (|V|·(|V|-1))."""
    n = adj.shape[0]
    max_e = n * (n - 1)
    return int(adj.sum()) / max_e if max_e > 0 else 0.0


def diameter(adj: IntMatrix) -> int:
    """Diameter of the undirected skeleton (longest shortest path), or -1."""
    skeleton = undirected_skeleton(adj)
    n = skeleton.shape[0]
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            cur = queue.popleft()
            for nbr in range(n):
                if skeleton[cur, nbr] and dist[nbr] < 0:
                    dist[nbr] = dist[cur] + 1
                    queue.append(nbr)
        if any(d < 0 for d in dist):
            return -1  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist


def clustering_coefficient(adj: IntMatrix) -> float:
    """Average local clustering coefficient of the undirected skeleton."""
    skeleton = undirected_skeleton(adj)
    n = skeleton.shape[0]
    total = 0.0
    count = 0
    for v in range(n):
        nbrs = [u for u in range(n) if skeleton[v, u]]
        k = len(nbrs)
        if k < 2:
            continue
        links = sum(1 for i, u in enumerate(nbrs) for w in nbrs[i + 1 :] if skeleton[u, w])
        total += 2.0 * links / (k * (k - 1))
        count += 1
    return total / count if count > 0 else 0.0


def average_degree(adj: IntMatrix) -> float:
    """Average total degree."""
    n = adj.shape[0]
    return float(np.sum(adj) + np.sum(adj.T)) / n if n > 0 else 0.0


# ====================================================================
# 5. Cycle Detection
# ====================================================================

def topological_sort(adj: IntMatrix) -> list[int] | None:
    """Kahn's algorithm.  Returns None if the graph has a cycle."""
    n = adj.shape[0]
    in_deg = np.sum(adj, axis=0).astype(int)
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for nbr in range(n):
            if adj[node, nbr]:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0:
                    queue.append(nbr)

    return order if len(order) == n else None


def has_cycle(adj: IntMatrix) -> bool:
    """True if the directed graph contains at least one cycle."""
    return topological_sort(adj) is None


def find_cycle(adj: IntMatrix) -> list[int] | None:
    """Return one directed cycle if it exists, else None."""
    n = adj.shape[0]
    WHITE, GREY, BLACK = 0, 1, 2
    colour = [WHITE] * n
    parent: dict[int, int] = {}

    def _dfs(v: int) -> list[int] | None:
        colour[v] = GREY
        for w in range(n):
            if not adj[v, w]:
                continue
            if colour[w] == GREY:
                cycle = [w, v]
                cur = v
                while cur != w:
                    cur = parent.get(cur, w)
                    cycle.append(cur)
                return cycle[::-1]
            if colour[w] == WHITE:
                parent[w] = v
                result = _dfs(w)
                if result is not None:
                    return result
        colour[v] = BLACK
        return None

    for v in range(n):
        if colour[v] == WHITE:
            cycle = _dfs(v)
            if cycle is not None:
                return cycle
    return None


# ====================================================================
# 6. Maximum Matching (undirected skeleton)
# ====================================================================

def greedy_maximum_matching(adj: IntMatrix) -> list[tuple[int, int]]:
    """Greedy maximal matching on the undirected skeleton.

    Not guaranteed to be maximum, but runs in O(|V| + |E|).
    """
    skeleton = undirected_skeleton(adj)
    n = skeleton.shape[0]
    matched = [False] * n
    matching: list[tuple[int, int]] = []

    for u in range(n):
        if matched[u]:
            continue
        for v in range(u + 1, n):
            if skeleton[u, v] and not matched[v]:
                matching.append((u, v))
                matched[u] = True
                matched[v] = True
                break

    return matching


def is_perfect_matching(adj: IntMatrix, matching: list[tuple[int, int]]) -> bool:
    """True if the matching covers all nodes."""
    n = adj.shape[0]
    covered = set()
    for u, v in matching:
        covered.add(u)
        covered.add(v)
    return len(covered) == n
