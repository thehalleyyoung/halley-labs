"""Crossover operators for DAG recombination.

Implements order-based, uniform, skeleton, and subgraph crossover
strategies for combining parent DAGs while maintaining acyclicity.
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
    """Return a topological ordering of the DAG via Kahn's algorithm."""
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


def _has_cycle(adj: AdjacencyMatrix) -> bool:
    """Check whether the adjacency matrix contains a directed cycle."""
    return len(_topological_sort(adj)) != adj.shape[0]


def _break_cycles_with_ordering(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Remove minimum back-edges using a greedy topological ordering.

    Computes a partial topological sort, extends it with remaining nodes,
    and removes any edge that violates the resulting ordering.
    """
    n = adj.shape[0]
    result = adj.copy()
    order = _topological_sort(result)
    if len(order) == n:
        return result  # Already acyclic

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


def _position_array(order: TopologicalOrder) -> npt.NDArray[np.int64]:
    """Create array mapping node → position in ordering."""
    n = len(order)
    pos = np.empty(n, dtype=np.int64)
    for i, node in enumerate(order):
        pos[node] = i
    return pos


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CrossoverOperator(ABC):
    """Abstract base class for DAG crossover operators."""

    @abstractmethod
    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        """Recombine *parent1* and *parent2* to produce two offspring.

        Parameters
        ----------
        parent1:
            Adjacency matrix of the first parent DAG.
        parent2:
            Adjacency matrix of the second parent DAG.
        rng:
            NumPy random generator for reproducibility.

        Returns
        -------
        Tuple[AdjacencyMatrix, AdjacencyMatrix]
            Two offspring adjacency matrices.
        """


# ---------------------------------------------------------------------------
# OrderCrossover (OX1-style)
# ---------------------------------------------------------------------------

class OrderCrossover(CrossoverOperator):
    """Merge topological orderings of two parents using OX1-style crossover.

    Algorithm:
    1. Extract topological orderings π1, π2 from parents.
    2. Select a random subsequence [i, j) from π1 as the "preserved segment".
    3. Fill remaining positions from π2, preserving relative order.
    4. For each edge in G1 ∪ G2, keep if consistent with merged ordering.
    5. The second child uses the symmetric operation.
    """

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]
        order1 = _topological_sort(parent1)
        order2 = _topological_sort(parent2)

        # OX1: select two cut points
        if n <= 2:
            c1, c2 = 0, n
        else:
            cuts = sorted(rng.choice(n, size=2, replace=False).tolist())
            c1, c2 = int(cuts[0]), int(cuts[1])
            if c1 == c2:
                c2 = min(c2 + 1, n)

        new_order1 = self._ox1_merge(order1, order2, c1, c2)
        new_order2 = self._ox1_merge(order2, order1, c1, c2)

        child1 = self._build_dag(new_order1, parent1, parent2)
        child2 = self._build_dag(new_order2, parent1, parent2)

        return child1, child2

    @staticmethod
    def _ox1_merge(
        primary: TopologicalOrder,
        secondary: TopologicalOrder,
        c1: int,
        c2: int,
    ) -> TopologicalOrder:
        """OX1-style merge: preserve primary[c1:c2], fill rest from secondary.

        The segment primary[c1:c2] is kept in-place.  Remaining positions
        are filled by scanning secondary starting after c2 (wrapping around),
        skipping elements already in the preserved segment.
        """
        n = len(primary)
        preserved = set(primary[c1:c2])
        # Elements from secondary not in the preserved segment, in secondary's order
        fill = [x for x in secondary if x not in preserved]

        result = [0] * n
        # Place preserved segment
        result[c1:c2] = primary[c1:c2]
        # Fill remaining positions
        fill_idx = 0
        for pos in list(range(c2, n)) + list(range(0, c1)):
            result[pos] = fill[fill_idx]
            fill_idx += 1

        return result

    @staticmethod
    def _build_dag(
        order: TopologicalOrder,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
    ) -> AdjacencyMatrix:
        """Build DAG: keep edges from union of parents that go forward in ordering."""
        n = len(order)
        position = _position_array(order)
        dag: AdjacencyMatrix = np.zeros_like(parent1)
        for i in range(n):
            for j in range(n):
                if position[i] < position[j] and (parent1[i, j] or parent2[i, j]):
                    dag[i, j] = 1
        return dag


# ---------------------------------------------------------------------------
# OrderBasedCrossover (single-point, legacy interface)
# ---------------------------------------------------------------------------

class OrderBasedCrossover(CrossoverOperator):
    """Combine topological orderings via single-point crossover.

    Each parent's topological order is computed.  A single-point crossover
    is applied on the orderings.  Edges are then reconstructed so that only
    forward edges from the union of both parents are kept.
    """

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]
        order1 = _topological_sort(parent1)
        order2 = _topological_sort(parent2)

        point = rng.integers(1, n) if n > 1 else 1

        new_order1 = self._merge_order(order1, order2, point)
        new_order2 = self._merge_order(order2, order1, point)

        child1 = self._build_dag(new_order1, parent1, parent2)
        child2 = self._build_dag(new_order2, parent1, parent2)

        return child1, child2

    @staticmethod
    def _merge_order(
        primary: TopologicalOrder,
        secondary: TopologicalOrder,
        point: int,
    ) -> TopologicalOrder:
        """Take first *point* nodes from primary, rest from secondary."""
        head = primary[:point]
        seen = set(head)
        tail = [n for n in secondary if n not in seen]
        return head + tail

    @staticmethod
    def _build_dag(
        order: TopologicalOrder,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
    ) -> AdjacencyMatrix:
        n = len(order)
        position = _position_array(order)
        dag: AdjacencyMatrix = np.zeros_like(parent1)
        for i in range(n):
            for j in range(n):
                if position[i] < position[j] and (parent1[i, j] or parent2[i, j]):
                    dag[i, j] = 1
        return dag


# ---------------------------------------------------------------------------
# UniformCrossover
# ---------------------------------------------------------------------------

class UniformCrossover(CrossoverOperator):
    """For each possible edge, take from parent1 or parent2 with equal
    probability, then repair any cycles.

    For each pair (i, j) with i ≠ j, the offspring inherits the edge
    status (present/absent) from one parent chosen uniformly at random.
    After assembly, cycles are broken using a greedy ordering repair.
    """

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]
        mask = rng.random((n, n)) < 0.5
        np.fill_diagonal(mask, False)

        child1 = np.where(mask, parent1, parent2).astype(np.int8)
        child2 = np.where(mask, parent2, parent1).astype(np.int8)
        np.fill_diagonal(child1, 0)
        np.fill_diagonal(child2, 0)

        child1 = _break_cycles_with_ordering(child1)
        child2 = _break_cycles_with_ordering(child2)

        return child1, child2


# ---------------------------------------------------------------------------
# SkeletonCrossover
# ---------------------------------------------------------------------------

class SkeletonCrossover(CrossoverOperator):
    """Merge skeletons and orient using merged v-structures.

    Algorithm:
    1. Compute undirected skeletons S1, S2 of both parents.
    2. Merge: S_child = S1 ∪ S2 (union of skeletons).
    3. Collect v-structures from both parents.
    4. Apply v-structures to orient edges in the merged skeleton.
    5. Apply Meek's R1 rule to propagate orientations.
    6. Orient remaining edges using a topological order from parent1/parent2.
    7. Break any remaining cycles.
    """

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]

        # Compute skeletons
        skel1 = (parent1 | parent1.T).astype(np.int8)
        skel2 = (parent2 | parent2.T).astype(np.int8)

        # Merged skeleton: union
        merged_skel = (skel1 | skel2).astype(np.int8)

        # Collect v-structures from both parents
        vstruct1 = self._find_v_structures(parent1)
        vstruct2 = self._find_v_structures(parent2)

        # Build child1 using v-structures from parent1, skeleton from union
        child1 = self._orient_skeleton(merged_skel.copy(), vstruct1, parent1, rng)
        # Build child2 using v-structures from parent2, skeleton from union
        child2 = self._orient_skeleton(merged_skel.copy(), vstruct2, parent2, rng)

        return child1, child2

    @staticmethod
    def _find_v_structures(adj: AdjacencyMatrix) -> List[Tuple[int, int, int]]:
        """Find v-structures i→j←k where i and k are not adjacent."""
        n = adj.shape[0]
        result: List[Tuple[int, int, int]] = []
        for j in range(n):
            parents = [i for i in range(n) if adj[i, j]]
            for a in range(len(parents)):
                for b in range(a + 1, len(parents)):
                    i, k = parents[a], parents[b]
                    if not adj[i, k] and not adj[k, i]:
                        result.append((i, j, k))
        return result

    @staticmethod
    def _orient_skeleton(
        skel: AdjacencyMatrix,
        v_structures: List[Tuple[int, int, int]],
        reference_dag: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Orient edges in skeleton using v-structures and reference DAG.

        1. Apply v-structures as directed edges.
        2. For remaining undirected edges, use the orientation from reference_dag.
        3. Orient any still-undirected edges randomly forward in a topological order.
        4. Break cycles.
        """
        n = skel.shape[0]
        dag = np.zeros((n, n), dtype=np.int8)

        # Apply v-structures
        oriented = set()
        for i, j, k in v_structures:
            if skel[i, j] or skel[j, i]:
                dag[i, j] = 1
                oriented.add((i, j))
            if skel[k, j] or skel[j, k]:
                dag[k, j] = 1
                oriented.add((k, j))

        # Orient remaining edges from reference DAG
        for i in range(n):
            for j in range(i + 1, n):
                if not (skel[i, j] or skel[j, i]):
                    continue
                if (i, j) in oriented or (j, i) in oriented:
                    continue
                if reference_dag[i, j]:
                    dag[i, j] = 1
                    oriented.add((i, j))
                elif reference_dag[j, i]:
                    dag[j, i] = 1
                    oriented.add((j, i))

        # Orient any still-undirected edges using random ordering
        unoriented = []
        for i in range(n):
            for j in range(i + 1, n):
                if (skel[i, j] or skel[j, i]) and (i, j) not in oriented and (j, i) not in oriented:
                    unoriented.append((i, j))

        if unoriented:
            perm = rng.permutation(n).tolist()
            pos = np.empty(n, dtype=int)
            for idx, node in enumerate(perm):
                pos[node] = idx
            for i, j in unoriented:
                if pos[i] < pos[j]:
                    dag[i, j] = 1
                else:
                    dag[j, i] = 1

        return _break_cycles_with_ordering(dag)


# ---------------------------------------------------------------------------
# SubgraphCrossover
# ---------------------------------------------------------------------------

class SubgraphCrossover(CrossoverOperator):
    """Swap induced subgraphs between two parents.

    A random subset of nodes is selected.  The induced sub-graph on those
    nodes is swapped between the two parents while edges outside the subset
    are inherited from the original parent.  Edges that would create a cycle
    are dropped.
    """

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]
        mask = rng.random(n) < 0.5
        subset = np.where(mask)[0]

        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in subset:
            for j in subset:
                child1[i, j] = parent2[i, j]
                child2[i, j] = parent1[i, j]

        child1 = _break_cycles_with_ordering(child1)
        child2 = _break_cycles_with_ordering(child2)

        return child1, child2


# ---------------------------------------------------------------------------
# MarkovBlanketCrossover (Algorithm 5)
# ---------------------------------------------------------------------------

class MarkovBlanketCrossover(CrossoverOperator):
    """Markov blanket-aware subgraph transplant crossover (Algorithm 5).

    Selects a pivot variable, extracts its Markov blanket from one parent,
    and transplants it into the other parent while preserving acyclicity.
    This preserves local conditional independence structure during recombination.
    """

    def __init__(self, repair_cycles: bool = True) -> None:
        self._repair_cycles = repair_cycles

    @staticmethod
    def _markov_blanket(adj: AdjacencyMatrix, pivot: int) -> Set[int]:
        """Return the Markov blanket of *pivot*: parents, children, co-parents."""
        n = adj.shape[0]
        parents: Set[int] = {i for i in range(n) if adj[i, pivot]}
        children: Set[int] = {j for j in range(n) if adj[pivot, j]}
        coparents: Set[int] = set()
        for ch in children:
            for k in range(n):
                if adj[k, ch] and k != pivot:
                    coparents.add(k)
        return parents | children | coparents

    @staticmethod
    def _transplant(
        donor: AdjacencyMatrix,
        recipient: AdjacencyMatrix,
        pivot: int,
        mb_nodes: Set[int],
    ) -> AdjacencyMatrix:
        """Create child by transplanting MB edges from *donor* into *recipient*."""
        child = recipient.copy()
        involved = mb_nodes | {pivot}
        for i in involved:
            for j in involved:
                if i != j:
                    child[i, j] = donor[i, j]
        return child

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        n = parent1.shape[0]
        pivot = int(rng.integers(0, n))

        mb1 = self._markov_blanket(parent1, pivot)
        mb2 = self._markov_blanket(parent2, pivot)

        child1 = self._transplant(parent1, parent2, pivot, mb1)
        child2 = self._transplant(parent2, parent1, pivot, mb2)

        if self._repair_cycles:
            child1 = _break_cycles_with_ordering(child1)
            child2 = _break_cycles_with_ordering(child2)

        return child1, child2
