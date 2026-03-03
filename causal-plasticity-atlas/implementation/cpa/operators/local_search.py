"""Local search operators (GES-like forward / backward / turn).

Implements greedy equivalence search style operators that can be
used as local refinement steps within a larger meta-heuristic.

Classes
-------
ForwardOperator / ForwardStep
    Greedy edge-addition with GES clique validity check.
BackwardOperator / BackwardStep
    Greedy edge-removal.
TurnOperator / TurnStep
    Greedy edge-reversal.
GESLocalSearch / GreedyLocalSearch
    Full forward–backward–turn local search.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_acyclic(adj: NDArray) -> bool:
    """Return True if *adj* is a DAG (Kahn's algorithm)."""
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


def _has_path(adj: NDArray, source: int, target: int) -> bool:
    """Return True if there is a directed path from *source* to *target*."""
    n = adj.shape[0]
    visited: Set[int] = set()
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == target:
            return True
        for v in range(n):
            if adj[u, v] != 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return False


def _parents(adj: NDArray, node: int) -> List[int]:
    """Return parent indices of *node*."""
    return [int(i) for i in range(adj.shape[0]) if adj[i, node] != 0]


def _children(adj: NDArray, node: int) -> List[int]:
    """Return child indices of *node*."""
    return [int(j) for j in range(adj.shape[1]) if adj[node, j] != 0]


def _neighbors_undirected(adj: NDArray, node: int) -> Set[int]:
    """Return nodes adjacent to *node* in the skeleton."""
    n = adj.shape[0]
    nbrs: Set[int] = set()
    for k in range(n):
        if adj[node, k] != 0 or adj[k, node] != 0:
            nbrs.add(k)
    return nbrs


def _total_score(
    adj: NDArray,
    score_fn: Callable[[int, Sequence[int]], float],
) -> float:
    """Sum of local scores over all nodes."""
    n = adj.shape[0]
    total = 0.0
    for j in range(n):
        pa = _parents(adj, j)
        total += score_fn(j, pa)
    return total


def _existing_edges(adj: NDArray) -> List[Tuple[int, int]]:
    """Return list of (i, j) with adj[i, j] != 0."""
    rows, cols = np.nonzero(adj)
    return list(zip(rows.tolist(), cols.tolist()))


# ------------------------------------------------------------------
# ForwardOperator / ForwardStep
# ------------------------------------------------------------------


class ForwardOperator:
    """Greedy edge-addition operator (GES forward step).

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    """

    def __init__(
        self, score_fn: Callable[[int, Sequence[int]], float]
    ) -> None:
        self.score_fn = score_fn

    def apply(self, dag: NDArray) -> Tuple[NDArray, float]:
        """Apply the best single edge addition to *dag*.

        Returns
        -------
        new_dag : NDArray
            The DAG after the best addition (unchanged if none found).
        gain : float
            Score improvement (0.0 if no valid addition).
        """
        best_edge, best_gain = self.find_best_addition(dag)
        if best_gain <= 0:
            return dag.copy(), 0.0
        result = dag.copy()
        i, j = best_edge
        result[i, j] = 1.0
        return result, best_gain

    def find_best_addition(
        self, dag: NDArray
    ) -> Tuple[Tuple[int, int], float]:
        """Return the edge and score-gain of the best addition."""
        n = dag.shape[0]
        best_edge = (0, 0)
        best_gain = 0.0

        for j in range(n):
            pa_j = _parents(dag, j)
            current_score = self.score_fn(j, pa_j)
            for i in range(n):
                if i == j or dag[i, j] != 0:
                    continue
                if _has_path(dag, j, i):
                    continue
                if not self._clique_check(dag, i, j):
                    continue
                new_pa = sorted(pa_j + [i])
                new_score = self.score_fn(j, new_pa)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        return best_edge, best_gain

    def _valid_operators(self, dag: NDArray) -> List[Tuple[int, int]]:
        """All valid forward steps (edge additions)."""
        n = dag.shape[0]
        ops: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and dag[i, j] == 0 and not _has_path(dag, j, i):
                    if self._clique_check(dag, i, j):
                        ops.append((i, j))
        return ops

    def _score_gain(self, dag: NDArray, i: int, j: int) -> float:
        """Score improvement from adding i->j."""
        pa_j = _parents(dag, j)
        current = self.score_fn(j, pa_j)
        new_pa = sorted(pa_j + [i])
        return self.score_fn(j, new_pa) - current

    def _clique_check(
        self,
        dag: NDArray,
        i: int,
        j: int,
        T: Optional[Set[int]] = None,
    ) -> bool:
        """GES validity: neighbors of j that are adjacent to i must form a clique.

        In the CPDAG/GES setting the set T of undirected neighbors of j
        that are not adjacent to i must form a clique with the parents
        of j.  For DAG-space search this is relaxed: we check that
        adding i->j does not create a new v-structure with an existing
        undirected neighbor.  In pure DAG search this always returns
        True (the acyclicity check suffices).
        """
        if T is not None:
            nbrs_j = _neighbors_undirected(dag, j)
            t_set = T & nbrs_j
            t_list = list(t_set)
            for a_idx in range(len(t_list)):
                for b_idx in range(a_idx + 1, len(t_list)):
                    a, b = t_list[a_idx], t_list[b_idx]
                    if dag[a, b] == 0 and dag[b, a] == 0:
                        return False
        return True


# Alias
ForwardStep = ForwardOperator


# ------------------------------------------------------------------
# BackwardOperator / BackwardStep
# ------------------------------------------------------------------


class BackwardOperator:
    """Greedy edge-removal operator (GES backward step).

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    """

    def __init__(
        self, score_fn: Callable[[int, Sequence[int]], float]
    ) -> None:
        self.score_fn = score_fn

    def apply(self, dag: NDArray) -> Tuple[NDArray, float]:
        """Apply the best single edge removal to *dag*."""
        best_edge, best_gain = self.find_best_removal(dag)
        if best_gain <= 0:
            return dag.copy(), 0.0
        result = dag.copy()
        i, j = best_edge
        result[i, j] = 0.0
        return result, best_gain

    def find_best_removal(
        self, dag: NDArray
    ) -> Tuple[Tuple[int, int], float]:
        """Return the edge and score-gain of the best removal."""
        edges = _existing_edges(dag)
        best_edge = (0, 0)
        best_gain = 0.0

        for i, j in edges:
            gain = self._score_gain(dag, i, j)
            if gain > best_gain:
                best_gain = gain
                best_edge = (i, j)

        return best_edge, best_gain

    def _score_gain(self, dag: NDArray, i: int, j: int) -> float:
        """Score improvement from removing i->j."""
        pa_j = _parents(dag, j)
        current = self.score_fn(j, pa_j)
        new_pa = [p for p in pa_j if p != i]
        return self.score_fn(j, new_pa) - current


# Alias
BackwardStep = BackwardOperator


# ------------------------------------------------------------------
# TurnOperator / TurnStep
# ------------------------------------------------------------------


class TurnOperator:
    """Greedy edge-reversal operator.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    """

    def __init__(
        self, score_fn: Callable[[int, Sequence[int]], float]
    ) -> None:
        self.score_fn = score_fn

    def apply(self, dag: NDArray) -> Tuple[NDArray, float]:
        """Apply the best single edge reversal to *dag*."""
        best_edge, best_gain = self.find_best_reversal(dag)
        if best_gain <= 0:
            return dag.copy(), 0.0
        result = dag.copy()
        i, j = best_edge
        w = result[i, j]
        result[i, j] = 0.0
        result[j, i] = w
        return result, best_gain

    def find_best_reversal(
        self, dag: NDArray
    ) -> Tuple[Tuple[int, int], float]:
        """Return the edge and score-gain of the best reversal."""
        edges = _existing_edges(dag)
        best_edge = (0, 0)
        best_gain = 0.0

        for i, j in edges:
            gain = self._score_gain(dag, i, j)
            if gain > best_gain:
                best_gain = gain
                best_edge = (i, j)

        return best_edge, best_gain

    def _score_gain(self, dag: NDArray, i: int, j: int) -> float:
        """Score improvement from reversing i->j to j->i."""
        # Check acyclicity of the reversal
        tmp = dag.copy()
        tmp[i, j] = 0.0
        if _has_path(tmp, i, j):
            return -np.inf
        if tmp[j, i] != 0:
            return -np.inf

        pa_j = _parents(dag, j)
        pa_i = _parents(dag, i)
        score_before = self.score_fn(j, pa_j) + self.score_fn(i, pa_i)

        new_pa_j = [p for p in pa_j if p != i]
        new_pa_i = sorted(pa_i + [j])
        score_after = self.score_fn(j, new_pa_j) + self.score_fn(i, new_pa_i)

        return score_after - score_before

    def search(self, dag: NDArray) -> Tuple[NDArray, float]:
        """Alias for apply — find best edge reversal."""
        return self.apply(dag)


# Alias
TurnStep = TurnOperator


# ------------------------------------------------------------------
# GESLocalSearch / GreedyLocalSearch
# ------------------------------------------------------------------


class GESLocalSearch:
    """Full GES-style local search combining forward, backward, and turn phases.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    max_iter : int
        Maximum number of greedy improvement iterations per phase.
    operators : Optional[list]
        Custom operator instances.  If ``None``, creates default
        ForwardOperator, BackwardOperator, TurnOperator.
    """

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
        max_iter: int = 100,
        operators: Optional[list] = None,
        *,
        max_iterations: Optional[int] = None,
    ) -> None:
        self.score_fn = score_fn
        self.max_iter = max_iterations if max_iterations is not None else max_iter
        if operators is not None:
            self.operators = operators
        else:
            self.operators = [
                ForwardOperator(score_fn),
                BackwardOperator(score_fn),
                TurnOperator(score_fn),
            ]

    def search(self, initial_dag: NDArray) -> NDArray:
        """Run forward, backward, and turn phases from *initial_dag*.

        Returns the locally-optimal DAG found.
        """
        dag = initial_dag.copy()
        dag = self._forward_phase(dag)
        dag = self._backward_phase(dag)
        dag = self._turn_phase(dag)
        return dag

    def forward_phase(self, dag: NDArray) -> NDArray:
        """Public alias for the forward phase."""
        return self._forward_phase(dag)

    def backward_phase(self, dag: NDArray) -> NDArray:
        """Public alias for the backward phase."""
        return self._backward_phase(dag)

    def _forward_phase(self, dag: NDArray) -> NDArray:
        """Greedily add edges until no improvement is possible."""
        fwd = None
        for op in self.operators:
            if isinstance(op, ForwardOperator):
                fwd = op
                break
        if fwd is None:
            fwd = ForwardOperator(self.score_fn)

        current = dag.copy()
        for _ in range(self.max_iter):
            new_dag, gain = fwd.apply(current)
            if gain <= 0:
                break
            current = new_dag
        return current

    def _backward_phase(self, dag: NDArray) -> NDArray:
        """Greedily remove edges until no improvement is possible."""
        bwd = None
        for op in self.operators:
            if isinstance(op, BackwardOperator):
                bwd = op
                break
        if bwd is None:
            bwd = BackwardOperator(self.score_fn)

        current = dag.copy()
        for _ in range(self.max_iter):
            new_dag, gain = bwd.apply(current)
            if gain <= 0:
                break
            current = new_dag
        return current

    def _turn_phase(self, dag: NDArray) -> NDArray:
        """Optional turn phase: greedily reverse edges."""
        turn = None
        for op in self.operators:
            if isinstance(op, TurnOperator):
                turn = op
                break
        if turn is None:
            turn = TurnOperator(self.score_fn)

        current = dag.copy()
        for _ in range(self.max_iter):
            new_dag, gain = turn.apply(current)
            if gain <= 0:
                break
            current = new_dag
        return current


# Alias
GreedyLocalSearch = GESLocalSearch
