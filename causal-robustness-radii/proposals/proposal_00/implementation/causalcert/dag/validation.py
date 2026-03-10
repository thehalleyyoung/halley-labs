"""
DAG validation utilities — acyclicity checks and cycle detection.

Provides O(V+E) topological-sort-based acyclicity verification and
cycle extraction for error reporting.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId


def is_dag(adj: AdjacencyMatrix) -> bool:
    """Check whether an adjacency matrix encodes a DAG (no directed cycles).

    Uses Kahn's algorithm (BFS topological sort) — O(V + E).

    Parameters
    ----------
    adj : AdjacencyMatrix
        Square binary matrix.

    Returns
    -------
    bool
        ``True`` if the graph is acyclic.
    """
    adj = np.asarray(adj)
    n = adj.shape[0]
    in_degree = adj.sum(axis=0).astype(int)
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.nonzero(adj[node])[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))
    return visited == n


def find_cycle(adj: AdjacencyMatrix) -> list[NodeId] | None:
    """Find and return a directed cycle, or ``None`` if acyclic.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Square binary matrix.

    Returns
    -------
    list[NodeId] | None
        A list of node ids forming a cycle, or ``None``.
    """
    adj = np.asarray(adj)
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    parent: dict[int, int] = {}

    def dfs(u: int) -> list[NodeId] | None:
        color[u] = GRAY
        for v in np.nonzero(adj[u])[0]:
            v = int(v)
            if color[v] == GRAY:
                # Reconstruct cycle
                cycle = [v, u]
                node = u
                while node != v:
                    node = parent.get(node, v)
                    cycle.append(node)
                cycle.reverse()
                return cycle
            if color[v] == WHITE:
                parent[v] = u
                result = dfs(v)
                if result is not None:
                    return result
        color[u] = BLACK
        return None

    for start in range(n):
        if color[start] == WHITE:
            result = dfs(start)
            if result is not None:
                return result
    return None


def is_dag_dfs(adj: AdjacencyMatrix) -> bool:
    """Check acyclicity using DFS-based cycle detection.

    Alternative to :func:`is_dag` that uses depth-first search.
    Returns the same result but with different traversal order.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Square binary matrix.

    Returns
    -------
    bool
    """
    return find_cycle(adj) is None


def validate_adjacency_matrix(adj: AdjacencyMatrix) -> list[str]:
    """Run comprehensive validation checks on an adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Matrix to validate.

    Returns
    -------
    list[str]
        List of error/warning messages. Empty if all checks pass.
    """
    issues: list[str] = []
    adj = np.asarray(adj)

    # Check square
    if adj.ndim != 2:
        issues.append(f"Expected 2D matrix, got {adj.ndim}D")
        return issues

    n, m = adj.shape
    if n != m:
        issues.append(f"Matrix must be square, got {n}x{m}")
        return issues

    # Check binary
    unique_vals = np.unique(adj)
    if not np.all(np.isin(unique_vals, [0, 1])):
        issues.append(f"Matrix should be binary, found values: {unique_vals}")

    # Check no self-loops
    if np.any(np.diag(adj)):
        loops = list(np.where(np.diag(adj))[0])
        issues.append(f"Self-loops at nodes: {loops}")

    # Check acyclicity
    if not is_dag(adj):
        cycle = find_cycle(adj)
        issues.append(f"Graph contains a cycle: {cycle}")

    return issues


def check_no_self_loops(adj: AdjacencyMatrix) -> bool:
    """Return ``True`` if the adjacency matrix has no self-loops."""
    adj = np.asarray(adj)
    return not np.any(np.diag(adj))


def check_binary(adj: AdjacencyMatrix) -> bool:
    """Return ``True`` if the adjacency matrix is binary (0s and 1s only)."""
    adj = np.asarray(adj)
    return bool(np.all(np.isin(adj, [0, 1])))


def check_faithfulness_heuristic(adj: AdjacencyMatrix) -> list[str]:
    """Heuristic checks for potential faithfulness violations.

    Faithfulness violations occur when a DAG has specific parameter values
    that create additional CI relations not implied by d-separation.
    This function identifies structural patterns that are PRONE to
    faithfulness violations (not definitive).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[str]
        Warning messages about potential faithfulness issues.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    warnings: list[str] = []

    # Check for nearly-canceling paths: if there are multiple directed paths
    # between two nodes, faithfulness violations are more likely
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Count number of directed paths from i to j
            n_paths = _count_directed_paths(adj, i, j)
            if n_paths > 2:
                warnings.append(
                    f"Multiple directed paths ({n_paths}) from {i} to {j}: "
                    f"prone to faithfulness violation via path cancellation"
                )

    # Check for long chains with bypasses
    for i in range(n):
        children_i = set(np.nonzero(adj[i])[0].astype(int))
        for c in children_i:
            grandchildren = set(np.nonzero(adj[int(c)])[0].astype(int))
            # If i is also a direct parent of a grandchild, we have a triangle
            for gc in grandchildren:
                if adj[i, gc]:
                    warnings.append(
                        f"Triangle {i}->{c}->{gc} with bypass {i}->{gc}: "
                        f"potential faithfulness issue"
                    )

    return warnings


def _count_directed_paths(
    adj: np.ndarray,
    source: int,
    target: int,
    max_count: int = 10,
) -> int:
    """Count directed paths from *source* to *target* (capped at *max_count*)."""
    n = adj.shape[0]
    count = 0

    # DFS with path tracking
    stack: list[tuple[int, set[int]]] = [(source, {source})]
    while stack and count < max_count:
        node, visited = stack.pop()
        for child in np.nonzero(adj[node])[0]:
            child = int(child)
            if child == target:
                count += 1
                if count >= max_count:
                    break
            elif child not in visited:
                stack.append((child, visited | {child}))

    return count


def structural_consistency_check(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> list[str]:
    """Run structural consistency checks on a DAG.

    Checks for common structural issues that may indicate data-entry
    errors or model misspecification.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names for readable messages.

    Returns
    -------
    list[str]
        Warning/error messages.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]
    issues: list[str] = []

    # Check for isolated nodes (no edges at all)
    for v in range(n):
        if adj[v, :].sum() == 0 and adj[:, v].sum() == 0:
            issues.append(f"Node '{names[v]}' ({v}) is isolated (no edges)")

    # Check for high in-degree nodes (potential misspecification)
    in_degrees = adj.sum(axis=0).astype(int)
    threshold = max(5, n // 3)
    for v in range(n):
        if in_degrees[v] > threshold:
            issues.append(
                f"Node '{names[v]}' ({v}) has high in-degree {in_degrees[v]} "
                f"(threshold: {threshold})"
            )

    # Check for symmetric edge pairs (shouldn't exist in a DAG but
    # could indicate confusion with undirected graph)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and adj[j, i]:
                issues.append(
                    f"Bidirectional edge between '{names[i]}' ({i}) and "
                    f"'{names[j]}' ({j}) — not valid in a DAG"
                )

    # Check connectivity
    undirected = adj | adj.T
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
    if len(visited) < n:
        disconnected = [names[i] for i in range(n) if i not in visited]
        issues.append(
            f"Graph is disconnected; unreachable nodes: {disconnected}"
        )

    return issues
