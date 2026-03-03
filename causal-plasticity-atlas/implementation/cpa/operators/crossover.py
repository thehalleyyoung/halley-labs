"""Markov blanket and edge-preserving crossover operators.

Crossover operators that combine two parent DAGs into an offspring
while respecting structural constraints such as acyclicity.

Operators
---------
MarkovBlanketCrossover
    Transplants a Markov blanket from one parent into the other.
EdgePreservingCrossover
    Keeps shared edges, mixes unique edges from both parents.
SubgraphCrossover
    Exchanges connected subgraphs between parents.
UniformCrossover
    Independently mixes each edge position.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_acyclic(adj: NDArray) -> bool:
    """Return True if *adj* represents a DAG (Kahn's algorithm)."""
    n = adj.shape[0]
    in_deg = (adj != 0).sum(axis=0).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return visited == n


def _topological_order(adj: NDArray) -> List[int]:
    """Return a topological ordering of *adj* (Kahn's)."""
    n = adj.shape[0]
    in_deg = (adj != 0).sum(axis=0).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    order: List[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return order


def _reachable(adj: NDArray, source: int) -> Set[int]:
    """Return the set of nodes reachable from *source* (BFS)."""
    n = adj.shape[0]
    visited: Set[int] = set()
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if adj[u, v] != 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def _parents(adj: NDArray, node: int) -> List[int]:
    """Return parent indices of *node* in *adj*."""
    return [int(i) for i in range(adj.shape[0]) if adj[i, node] != 0]


def _children(adj: NDArray, node: int) -> List[int]:
    """Return child indices of *node* in *adj*."""
    return [int(j) for j in range(adj.shape[1]) if adj[node, j] != 0]


def _markov_blanket(adj: NDArray, node: int) -> Set[int]:
    """Return the Markov blanket of *node*: parents + children + co-parents."""
    pa = set(_parents(adj, node))
    ch = set(_children(adj, node))
    co_parents: Set[int] = set()
    for c in ch:
        co_parents.update(_parents(adj, c))
    mb = pa | ch | co_parents
    mb.discard(node)
    return mb


def _remove_back_edges(adj: NDArray, order: List[int]) -> NDArray:
    """Zero out back-edges w.r.t. *order* to guarantee acyclicity."""
    pos = {v: i for i, v in enumerate(order)}
    result = adj.copy()
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if result[i, j] != 0 and pos.get(i, 0) >= pos.get(j, 0):
                result[i, j] = 0.0
    return result


def _repair_dag(adj: NDArray) -> NDArray:
    """Repair an adjacency matrix so it becomes a DAG.

    Uses iterative back-edge removal: compute a pseudo-topological
    ordering from in-degrees, then remove edges that violate it.
    """
    result = adj.copy()
    for _ in range(result.shape[0]):
        if _is_acyclic(result):
            return result
        order = _topological_order(result)
        if len(order) == result.shape[0]:
            return result
        # Nodes not in partial order form cycle(s).  Break the cycle
        # by removing the edge with smallest absolute weight in
        # each strongly-connected component.
        in_order = set(order)
        n = result.shape[0]
        for i in range(n):
            if i in in_order:
                continue
            for j in range(n):
                if j in in_order:
                    continue
                if result[i, j] != 0:
                    result[i, j] = 0.0
                    if _is_acyclic(result):
                        return result
    return result


# ------------------------------------------------------------------
# MarkovBlanketCrossover
# ------------------------------------------------------------------


class MarkovBlanketCrossover:
    """Crossover based on Markov blanket subgraph transplantation.

    Selects a focal node, extracts its Markov blanket from one parent,
    and grafts it into the other parent's structure while maintaining
    acyclicity.

    Parameters
    ----------
    crossover_rate : float
        Probability that crossover is actually applied (otherwise
        the first parent is returned unchanged).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        crossover_rate: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.crossover_rate = crossover_rate
        self._rng = np.random.default_rng(seed)

    # -- public API (backward-compat) --------------------------------

    def crossover(
        self,
        parent1: NDArray,
        parent2: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Return an offspring DAG by blanket transplantation."""
        rng = rng or self._rng
        n = parent1.shape[0]
        if rng.random() > self.crossover_rate or n == 0:
            return parent1.copy()

        focal = self._select_focal_node(parent1, parent2, rng)
        offspring = self._transfer_markov_blanket(parent2, parent1, focal)
        offspring = _repair_dag(offspring)
        return offspring

    def select_subgraph(self, dag: NDArray, node: int) -> Set[int]:
        """Return the Markov blanket of *node* in *dag*."""
        return _markov_blanket(dag, node)

    # -- internals ---------------------------------------------------

    def _select_focal_node(
        self, adj1: NDArray, adj2: NDArray, rng: np.random.Generator
    ) -> int:
        """Select focal node, preferring nodes whose MB differs most."""
        n = adj1.shape[0]
        diffs = np.zeros(n)
        for v in range(n):
            mb1 = _markov_blanket(adj1, v)
            mb2 = _markov_blanket(adj2, v)
            union = mb1 | mb2
            inter = mb1 & mb2
            diffs[v] = len(union) - len(inter) if union else 0
        total = diffs.sum()
        if total == 0:
            return int(rng.integers(0, n))
        probs = diffs / total
        return int(rng.choice(n, p=probs))

    def _transfer_markov_blanket(
        self, source: NDArray, target: NDArray, node: int
    ) -> NDArray:
        """Graft the MB of *node* from *source* into *target*."""
        result = target.copy()
        mb_nodes = _markov_blanket(source, node) | {node}
        # Remove existing edges involving node in target
        for m in mb_nodes:
            result[m, node] = 0.0
            result[node, m] = 0.0
        # Copy edges from source for the MB subgraph
        for m in mb_nodes:
            result[m, node] = source[m, node]
            result[node, m] = source[node, m]
        # Also copy inter-MB edges from source
        mb_list = list(mb_nodes)
        for i_idx in range(len(mb_list)):
            u = mb_list[i_idx]
            for j_idx in range(len(mb_list)):
                v = mb_list[j_idx]
                if u != v:
                    result[u, v] = source[u, v]
        return result


# ------------------------------------------------------------------
# EdgePreservingCrossover
# ------------------------------------------------------------------


class EdgePreservingCrossover:
    """Crossover that preserves edges common to both parents.

    Shared edges are always kept; non-shared edges are sampled from
    either parent with probability *preservation_rate*.

    Parameters
    ----------
    crossover_rate : float
        Probability of applying crossover (vs. returning parent 1).
    preservation_rate : float
        Probability of retaining a non-shared edge from parent 1
        (1 - preservation_rate for parent 2).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        crossover_rate: float = 0.8,
        preservation_rate: float = 0.5,
        seed: Optional[int] = None,
        *,
        preserve_ratio: Optional[float] = None,
    ) -> None:
        self.crossover_rate = crossover_rate
        self.preservation_rate = (
            preserve_ratio if preserve_ratio is not None else preservation_rate
        )
        self._rng = np.random.default_rng(seed)

    def crossover(
        self,
        parent1: NDArray,
        parent2: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Return an offspring DAG preserving shared edges."""
        rng = rng or self._rng
        n = parent1.shape[0]
        if rng.random() > self.crossover_rate or n == 0:
            return parent1.copy()

        shared = self._shared_edges(parent1, parent2)
        unique1 = self._unique_edges(parent1, parent2)
        unique2 = self._unique_edges(parent2, parent1)

        offspring = np.zeros_like(parent1)

        # Always keep shared edges
        for i, j in shared:
            offspring[i, j] = parent1[i, j]

        # Mix unique edges
        for i, j in unique1:
            if rng.random() < self.preservation_rate:
                offspring[i, j] = parent1[i, j]
        for i, j in unique2:
            if rng.random() < (1.0 - self.preservation_rate):
                offspring[i, j] = parent2[i, j]

        offspring = _repair_dag(offspring)
        return offspring

    def identify_common_edges(
        self, dag1: NDArray, dag2: NDArray
    ) -> Set[Tuple[int, int]]:
        """Return the set of edges present in both DAGs."""
        return self._shared_edges(dag1, dag2)

    def _shared_edges(
        self, adj1: NDArray, adj2: NDArray
    ) -> Set[Tuple[int, int]]:
        """Find edges present in both parents."""
        n = adj1.shape[0]
        edges: Set[Tuple[int, int]] = set()
        for i in range(n):
            for j in range(n):
                if adj1[i, j] != 0 and adj2[i, j] != 0:
                    edges.add((i, j))
        return edges

    def _unique_edges(
        self, adj1: NDArray, adj2: NDArray
    ) -> Set[Tuple[int, int]]:
        """Find edges in *adj1* but not in *adj2*."""
        n = adj1.shape[0]
        edges: Set[Tuple[int, int]] = set()
        for i in range(n):
            for j in range(n):
                if adj1[i, j] != 0 and adj2[i, j] == 0:
                    edges.add((i, j))
        return edges


# ------------------------------------------------------------------
# SubgraphCrossover
# ------------------------------------------------------------------


class SubgraphCrossover:
    """Exchange connected subgraphs between two parent DAGs.

    Selects a random connected subgraph from one parent, removes
    internal edges for that subgraph from the other parent, and
    grafts the source subgraph in.

    Parameters
    ----------
    crossover_rate : float
        Probability that crossover is applied.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        crossover_rate: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.crossover_rate = crossover_rate
        self._rng = np.random.default_rng(seed)

    def crossover(
        self,
        parent1_adj: NDArray,
        parent2_adj: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Exchange connected subgraphs between parents."""
        rng = rng or self._rng
        n = parent1_adj.shape[0]
        if rng.random() > self.crossover_rate or n < 2:
            return parent1_adj.copy()

        subgraph = self._select_subgraph(parent2_adj, rng)
        if len(subgraph) == 0:
            return parent1_adj.copy()

        offspring = self._graft_subgraph(parent1_adj, parent2_adj, subgraph)
        offspring = _repair_dag(offspring)
        return offspring

    def _select_subgraph(
        self, adj: NDArray, rng: np.random.Generator
    ) -> Set[int]:
        """Select a random connected subgraph via BFS from a seed node."""
        n = adj.shape[0]
        seed_node = int(rng.integers(0, n))
        # BFS on the skeleton (undirected version)
        skeleton = (adj != 0) | (adj.T != 0)
        max_size = max(2, int(rng.integers(2, max(3, n // 2 + 1))))
        visited: Set[int] = {seed_node}
        frontier = deque([seed_node])
        while frontier and len(visited) < max_size:
            u = frontier.popleft()
            neighbors = [
                int(v)
                for v in range(n)
                if skeleton[u, v] and v not in visited
            ]
            rng.shuffle(np.array(neighbors))  # type: ignore[arg-type]
            for v in neighbors:
                if len(visited) >= max_size:
                    break
                visited.add(v)
                frontier.append(v)
        return visited

    def _graft_subgraph(
        self,
        target: NDArray,
        source: NDArray,
        subgraph_nodes: Set[int],
    ) -> NDArray:
        """Graft subgraph edges from *source* into *target*."""
        result = target.copy()
        sg = list(subgraph_nodes)
        # Replace all internal edges with source's
        for u in sg:
            for v in sg:
                result[u, v] = source[u, v]
        return result


# ------------------------------------------------------------------
# UniformCrossover
# ------------------------------------------------------------------


class UniformCrossover:
    """Independently mix each edge position from two parents.

    Each edge slot (i, j) is taken from parent 1 with probability 0.5,
    otherwise from parent 2.  A DAG repair step removes any resulting
    cycles.

    Parameters
    ----------
    crossover_rate : float
        Probability that crossover is applied.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        crossover_rate: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.crossover_rate = crossover_rate
        self._rng = np.random.default_rng(seed)

    def crossover(
        self,
        parent1_adj: NDArray,
        parent2_adj: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Independently mix each edge from two parents."""
        rng = rng or self._rng
        n = parent1_adj.shape[0]
        if rng.random() > self.crossover_rate or n == 0:
            return parent1_adj.copy()

        mask = rng.random((n, n)) < 0.5
        offspring = np.where(mask, parent1_adj, parent2_adj)
        np.fill_diagonal(offspring, 0.0)
        offspring = self._repair_cycles(offspring)
        return offspring

    def _repair_cycles(self, adj: NDArray) -> NDArray:
        """Remove cycles by greedily removing back-edges.

        Compute a topological-ish ordering from in-degrees and remove
        edges that point backwards.
        """
        result = adj.copy()
        if _is_acyclic(result):
            return result

        n = result.shape[0]
        # Build ordering greedily by in-degree
        in_deg = (result != 0).sum(axis=0).copy()
        order: List[int] = []
        remaining = set(range(n))
        while remaining:
            # Pick node with min in-degree among remaining
            best = min(remaining, key=lambda x: in_deg[x])
            order.append(best)
            remaining.remove(best)
            for v in range(n):
                if result[best, v] != 0 and v in remaining:
                    in_deg[v] -= 1

        result = _remove_back_edges(result, order)
        return result
