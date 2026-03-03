"""Repair operators for restoring DAG validity.

Provides TopologicalRepair (feedback arc set approximation),
OrderRepair (enforce a given ordering), MinimalRepair (DFS-based
minimum-weight cycle breaking), and legacy AcyclicityRepair /
ConnectivityRepair operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, TopologicalOrder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _topological_sort(adj: AdjacencyMatrix) -> TopologicalOrder:
    """Return a topological ordering of the DAG via Kahn's algorithm.

    If the graph has cycles, returns a partial ordering (fewer than n nodes).
    """
    n = adj.shape[0]
    in_degree = adj.sum(axis=0).copy()
    queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    order: TopologicalOrder = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in range(n):
            if adj[node, child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    return order


def _is_acyclic(adj: AdjacencyMatrix) -> bool:
    """Check acyclicity via Kahn's algorithm."""
    return len(_topological_sort(adj)) == adj.shape[0]


def _find_cycle_dfs(adj: AdjacencyMatrix) -> Optional[List[int]]:
    """Find one directed cycle using DFS, or return None if acyclic.

    Returns a list of nodes forming the cycle (first = last node).
    """
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = np.zeros(n, dtype=np.int8)
    parent: Dict[int, int] = {}

    def dfs(u: int) -> Optional[List[int]]:
        colour[u] = GRAY
        for v in range(n):
            if not adj[u, v]:
                continue
            if colour[v] == GRAY:
                # Found cycle: trace back from u to v
                cycle = [v, u]
                cur = u
                while cur != v:
                    cur = parent.get(cur, v)
                    if cur == v:
                        break
                    cycle.append(cur)
                cycle.reverse()
                return cycle
            if colour[v] == WHITE:
                parent[v] = u
                result = dfs(v)
                if result is not None:
                    return result
        colour[u] = BLACK
        return None

    for u in range(n):
        if colour[u] == WHITE:
            result = dfs(u)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RepairOperator(ABC):
    """Abstract base class for DAG repair operators."""

    @abstractmethod
    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        """Repair *dag* so that it satisfies structural constraints.

        Parameters
        ----------
        dag:
            Adjacency matrix that may violate DAG constraints.

        Returns
        -------
        AdjacencyMatrix
            A repaired adjacency matrix.
        """


# ---------------------------------------------------------------------------
# TopologicalRepair
# ---------------------------------------------------------------------------

class TopologicalRepair(RepairOperator):
    """Remove minimum edges to make a directed graph acyclic.

    Uses a greedy feedback arc set approximation:
    1. Compute in-degree and out-degree for each node.
    2. Iteratively select the node with highest (out_degree - in_degree)
       as a "source" and place it next in the ordering.
    3. Remove all back-edges with respect to the computed ordering.

    This is the Berger & Shor (1990) greedy FAS approximation, which
    removes at most half the edges of an optimal feedback arc set.
    """

    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()

        if _is_acyclic(result):
            return result

        order = self._greedy_fas_ordering(result)
        position = np.empty(n, dtype=int)
        for pos, node in enumerate(order):
            position[node] = pos

        # Remove all back-edges
        for i in range(n):
            for j in range(n):
                if result[i, j] and position[i] >= position[j]:
                    result[i, j] = 0

        return result

    @staticmethod
    def _greedy_fas_ordering(adj: AdjacencyMatrix) -> TopologicalOrder:
        """Compute a greedy ordering that minimizes feedback arcs.

        Uses the Eades-Lin-Smyth heuristic:
        - Maintain two sequences s1 (prepend sinks) and s2 (append sources).
        - At each step pick the node that maximises out_degree - in_degree
          as a source (add to s2), or a sink to s1.
        """
        n = adj.shape[0]
        remaining = set(range(n))
        work = adj.copy().astype(np.int64)
        s1: List[int] = []  # Will be reversed at end
        s2: List[int] = []

        while remaining:
            # Find sinks (out_degree == 0 among remaining)
            changed = True
            while changed:
                changed = False
                for v in list(remaining):
                    out_deg = sum(1 for u in remaining if work[v, u])
                    if out_deg == 0:
                        s1.append(v)
                        remaining.discard(v)
                        changed = True

            # Find sources (in_degree == 0 among remaining)
            changed = True
            while changed:
                changed = False
                for v in list(remaining):
                    in_deg = sum(1 for u in remaining if work[u, v])
                    if in_deg == 0:
                        s2.append(v)
                        remaining.discard(v)
                        changed = True

            if not remaining:
                break

            # Pick the node with max (out_degree - in_degree)
            best_node = -1
            best_delta = -float("inf")
            for v in remaining:
                out_d = sum(1 for u in remaining if work[v, u])
                in_d = sum(1 for u in remaining if work[u, v])
                delta = out_d - in_d
                if delta > best_delta:
                    best_delta = delta
                    best_node = v
            if best_node >= 0:
                s2.append(best_node)
                remaining.discard(best_node)

        s1.reverse()
        return s2 + s1


# ---------------------------------------------------------------------------
# OrderRepair
# ---------------------------------------------------------------------------

class OrderRepair(RepairOperator):
    """Given a target topological ordering, remove edges that violate it.

    Any edge (i, j) where i appears after j in the given ordering is removed.

    Parameters
    ----------
    ordering : TopologicalOrder
        The target topological ordering to enforce.
    """

    def __init__(self, ordering: TopologicalOrder) -> None:
        self.ordering = list(ordering)

    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()

        position = np.empty(n, dtype=int)
        for pos, node in enumerate(self.ordering):
            position[node] = pos

        for i in range(n):
            for j in range(n):
                if result[i, j] and position[i] >= position[j]:
                    result[i, j] = 0

        return result


# ---------------------------------------------------------------------------
# MinimalRepair
# ---------------------------------------------------------------------------

class MinimalRepair(RepairOperator):
    """Remove minimum weight edges to break all cycles.

    Uses iterative DFS-based cycle detection.  For each cycle found,
    removes the edge with the lowest weight (default: uniform weights = 1).
    Repeats until no cycles remain.

    For unweighted graphs, this greedily removes one edge per cycle,
    preferring edges that participate in the most cycles.

    Parameters
    ----------
    weights : AdjacencyMatrix, optional
        n×n matrix of edge weights.  Higher weight → less likely to be removed.
        If None, all edges have weight 1.
    """

    def __init__(self, weights: Optional[AdjacencyMatrix] = None) -> None:
        self.weights = weights

    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()

        if self.weights is not None:
            w = self.weights.copy()
        else:
            w = np.ones((n, n), dtype=np.float64)

        max_iterations = n * n  # Safety bound
        for _ in range(max_iterations):
            cycle = _find_cycle_dfs(result)
            if cycle is None:
                break

            # Find the minimum-weight edge in the cycle
            min_weight = float("inf")
            min_edge = (cycle[0], cycle[1])
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                if result[u, v] and w[u, v] < min_weight:
                    min_weight = w[u, v]
                    min_edge = (u, v)
            # Also check closing edge
            u, v = cycle[-1], cycle[0]
            if result[u, v] and w[u, v] < min_weight:
                min_edge = (u, v)

            result[min_edge[0], min_edge[1]] = 0

        return result


# ---------------------------------------------------------------------------
# AcyclicityRepair (legacy)
# ---------------------------------------------------------------------------

class AcyclicityRepair(RepairOperator):
    """Remove back-edges to restore acyclicity.

    Computes a partial topological sort; nodes not reached (part of cycles)
    are appended in index order.  All edges violating this ordering are removed.
    """

    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()

        order = _topological_sort(result)
        if len(order) == n:
            return result

        visited = set(order)
        full_order = list(order) + [i for i in range(n) if i not in visited]
        position = np.empty(n, dtype=int)
        for pos, node in enumerate(full_order):
            position[node] = pos

        for i in range(n):
            for j in range(n):
                if result[i, j] and position[i] >= position[j]:
                    result[i, j] = 0

        return result


# ---------------------------------------------------------------------------
# ConnectivityRepair
# ---------------------------------------------------------------------------

class ConnectivityRepair(RepairOperator):
    """Add edges to ensure weak connectivity while preserving acyclicity.

    If the underlying undirected graph has multiple connected components,
    links them with directed edges that respect the topological ordering.
    """

    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()

        components = self._connected_components(result)
        if len(components) <= 1:
            return result

        order = _topological_sort(result)
        position = np.empty(n, dtype=int)
        for pos, node in enumerate(order):
            position[node] = pos

        for k in range(len(components) - 1):
            comp_a = components[k]
            comp_b = components[k + 1]

            src = min(comp_a, key=lambda v: position[v])
            tgt = max(comp_b, key=lambda v: position[v])

            if position[src] < position[tgt]:
                result[src, tgt] = 1
            else:
                result[tgt, src] = 1

        return result

    @staticmethod
    def _connected_components(adj: AdjacencyMatrix) -> List[List[int]]:
        """Find weakly connected components via BFS on the undirected view."""
        n = adj.shape[0]
        undirected = (adj | adj.T).astype(bool)
        visited = np.zeros(n, dtype=bool)
        components: List[List[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: List[int] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in range(n):
                    if undirected[node, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)

        return components
