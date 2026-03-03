"""Mutation operators for DAG manipulation.

Provides acyclicity-preserving mutations using topological ordering
maintenance, with composite and adaptive variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, MutationConfig, MutationRecord, MutationType, TopologicalOrder


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


def _can_reach(adj: AdjacencyMatrix, source: int, target: int) -> bool:
    """Return True if there is a directed path from *source* to *target*."""
    n = adj.shape[0]
    if source == target:
        return True
    visited = set()
    queue: deque[int] = deque([source])
    visited.add(source)
    while queue:
        node = queue.popleft()
        for child in range(n):
            if adj[node, child]:
                if child == target:
                    return True
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
    return False


def _topo_position(adj: AdjacencyMatrix) -> npt.NDArray[np.int64]:
    """Return array mapping node -> position in topological order."""
    order = _topological_sort(adj)
    n = adj.shape[0]
    pos = np.empty(n, dtype=np.int64)
    for i, node in enumerate(order):
        pos[node] = i
    return pos


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MutationOperator(ABC):
    """Abstract base class for DAG mutation operators."""

    @abstractmethod
    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Apply a mutation to *dag* and return the mutated copy.

        Parameters
        ----------
        dag:
            Adjacency matrix of a directed acyclic graph.
        rng:
            NumPy random generator for reproducibility.

        Returns
        -------
        AdjacencyMatrix
            A new adjacency matrix representing the mutated DAG.
        """


# ---------------------------------------------------------------------------
# TopologicalMutation: maintains topological ordering
# ---------------------------------------------------------------------------

class TopologicalMutation(MutationOperator):
    """Mutation that maintains a topological ordering.

    Pre-computes the topological order and only considers operations
    that are consistent with it, guaranteeing acyclicity without
    expensive cycle checks.
    """

    def __init__(self, add_prob: float = 0.4, remove_prob: float = 0.3, reverse_prob: float = 0.3) -> None:
        total = add_prob + remove_prob + reverse_prob
        self.add_prob = add_prob / total
        self.remove_prob = remove_prob / total
        self.reverse_prob = reverse_prob / total

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Apply a random topological mutation (add/remove/reverse)."""
        result = dag.copy()
        roll = rng.random()
        if roll < self.add_prob:
            return self._add_edge(result, rng)
        elif roll < self.add_prob + self.remove_prob:
            return self._remove_edge(result, rng)
        else:
            return self._reverse_edge(result, rng)

    def _add_edge(self, adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Add a random forward edge consistent with topological order."""
        n = adj.shape[0]
        order = _topological_sort(adj)
        pos = np.empty(n, dtype=int)
        for i, node in enumerate(order):
            pos[node] = i

        # Collect all non-existing forward edges
        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and pos[i] < pos[j] and not adj[i, j]:
                    candidates.append((i, j))
        if candidates:
            i, j = candidates[rng.integers(0, len(candidates))]
            adj[i, j] = 1
        return adj

    def _remove_edge(self, adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Remove a random existing edge."""
        edges = list(zip(*np.nonzero(adj)))
        if edges:
            idx = rng.integers(0, len(edges))
            i, j = edges[idx]
            adj[i, j] = 0
        return adj

    def _reverse_edge(self, adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Reverse a random edge if the reversal preserves acyclicity.

        After removing i→j, checks if adding j→i creates a cycle by
        testing if i can reach j in the remaining graph.
        """
        edges = list(zip(*np.nonzero(adj)))
        if not edges:
            return adj
        idx = rng.integers(0, len(edges))
        i, j = edges[idx]
        adj[i, j] = 0
        if not _can_reach(adj, i, j):
            adj[j, i] = 1
        else:
            adj[i, j] = 1  # Revert removal
        return adj


# ---------------------------------------------------------------------------
# EdgeAddMutation
# ---------------------------------------------------------------------------

class EdgeAddMutation(MutationOperator):
    """Add a random edge consistent with the topological ordering.

    Computes a topological ordering and randomly selects a pair (i, j)
    where i appears before j and no edge i→j exists.  This guarantees
    acyclicity without any DFS cycle check.
    """

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()
        order = _topological_sort(result)
        pos = np.empty(n, dtype=int)
        for i, node in enumerate(order):
            pos[node] = i

        candidates = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j and pos[i] < pos[j] and not result[i, j]
        ]
        if candidates:
            i, j = candidates[rng.integers(0, len(candidates))]
            result[i, j] = 1
        return result


# ---------------------------------------------------------------------------
# EdgeRemoveMutation
# ---------------------------------------------------------------------------

class EdgeRemoveMutation(MutationOperator):
    """Remove a random existing edge."""

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        result = dag.copy()
        edges = list(zip(*np.nonzero(result)))
        if edges:
            idx = rng.integers(0, len(edges))
            i, j = edges[idx]
            result[i, j] = 0
        return result


# ---------------------------------------------------------------------------
# EdgeReverseMutation
# ---------------------------------------------------------------------------

class EdgeReverseMutation(MutationOperator):
    """Reverse a random edge (i→j) to (j→i) if acyclicity is preserved.

    After removing i→j, uses DFS from i to check if i can reach j
    in the remaining graph.  If so, the reversal would create a cycle
    j→i→...→j, so the original edge is restored.
    """

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        result = dag.copy()
        edges = list(zip(*np.nonzero(result)))
        if not edges:
            return result

        idx = rng.integers(0, len(edges))
        i, j = edges[idx]

        result[i, j] = 0
        # Check if reversal is safe: j→i creates cycle iff i can reach j
        if _can_reach(result, i, j):
            result[i, j] = 1  # Unsafe, revert
        else:
            result[j, i] = 1  # Safe to reverse
        return result


# ---------------------------------------------------------------------------
# EdgeFlipMutation (legacy)
# ---------------------------------------------------------------------------

class EdgeFlipMutation(MutationOperator):
    """Randomly adds or removes one edge, ensuring acyclicity."""

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Flip a random edge (add if absent, remove if present).

        If adding an edge would introduce a cycle the mutation falls back to
        removing a random existing edge instead.
        """
        n = dag.shape[0]
        result = dag.copy()

        i, j = rng.integers(0, n), rng.integers(0, n)
        while i == j:
            i, j = rng.integers(0, n), rng.integers(0, n)

        if result[i, j]:
            result[i, j] = 0
        else:
            result[i, j] = 1
            if _has_cycle(result):
                result[i, j] = 0
                edges = list(zip(*np.nonzero(result)))
                if edges:
                    ei, ej = edges[rng.integers(0, len(edges))]
                    result[ei, ej] = 0

        return result


# ---------------------------------------------------------------------------
# EdgeReversalMutation (legacy name)
# ---------------------------------------------------------------------------

class EdgeReversalMutation(MutationOperator):
    """Pick a random edge and reverse it if the result is still acyclic."""

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        result = dag.copy()
        edges = list(zip(*np.nonzero(result)))
        if not edges:
            return result

        idx = rng.integers(0, len(edges))
        i, j = edges[idx]

        result[i, j] = 0
        result[j, i] = 1

        if _has_cycle(result):
            result[j, i] = 0
            result[i, j] = 1

        return result


# ---------------------------------------------------------------------------
# AcyclicEdgeAddition (legacy name)
# ---------------------------------------------------------------------------

class AcyclicEdgeAddition(MutationOperator):
    """Add a random edge guaranteed to preserve acyclicity.

    Uses topological ordering to find valid forward edges.
    """

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        n = dag.shape[0]
        result = dag.copy()
        order = _topological_sort(result)
        position = np.empty(n, dtype=int)
        for pos, node in enumerate(order):
            position[node] = pos

        candidates = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j and position[i] < position[j] and not result[i, j]
        ]
        if candidates:
            i, j = candidates[rng.integers(0, len(candidates))]
            result[i, j] = 1
        return result


# ---------------------------------------------------------------------------
# CompositeMutation
# ---------------------------------------------------------------------------

class CompositeMutation(MutationOperator):
    """Apply a sequence of mutations with configurable probabilities.

    Parameters
    ----------
    operators : Sequence[MutationOperator]
        List of mutation operators.
    probabilities : Sequence[float], optional
        Probability of applying each operator.  Normalised internally.
        If not given, uniform probabilities are used.
    n_mutations : int
        Number of mutations to apply per call (default 1).
    """

    def __init__(
        self,
        operators: Sequence[MutationOperator],
        probabilities: Optional[Sequence[float]] = None,
        n_mutations: int = 1,
    ) -> None:
        self.operators = list(operators)
        if probabilities is None:
            self.probabilities = np.ones(len(operators)) / len(operators)
        else:
            p = np.array(probabilities, dtype=np.float64)
            self.probabilities = p / p.sum()
        self.n_mutations = n_mutations

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Apply *n_mutations* randomly chosen mutations sequentially."""
        result = dag.copy()
        for _ in range(self.n_mutations):
            idx = rng.choice(len(self.operators), p=self.probabilities)
            result = self.operators[idx].mutate(result, rng)
        return result


# ---------------------------------------------------------------------------
# AdaptiveMutation
# ---------------------------------------------------------------------------

class AdaptiveMutation(MutationOperator):
    """Adjusts mutation rates based on archive improvement history.

    Maintains a running count of how many times each operator produced
    an improvement (was accepted into the archive).  Probabilities are
    updated using a softmax over these counts with a temperature
    parameter that controls exploration vs. exploitation.

    Parameters
    ----------
    operators : Sequence[MutationOperator]
        Available mutation operators.
    window_size : int
        Number of recent trials to consider (sliding window).
    temperature : float
        Softmax temperature.  Higher → more uniform, lower → greedy.
    min_prob : float
        Minimum probability for any operator to prevent starvation.
    """

    def __init__(
        self,
        operators: Sequence[MutationOperator],
        window_size: int = 100,
        temperature: float = 1.0,
        min_prob: float = 0.05,
    ) -> None:
        self.operators = list(operators)
        self.window_size = window_size
        self.temperature = temperature
        self.min_prob = min_prob
        n_ops = len(operators)
        self._success_counts = np.zeros(n_ops, dtype=np.float64)
        self._trial_counts = np.zeros(n_ops, dtype=np.float64)
        self._history: List[Tuple[int, bool]] = []  # (operator_idx, was_improvement)
        self._last_operator_idx: int = 0

    @property
    def probabilities(self) -> npt.NDArray[np.float64]:
        """Current selection probabilities for each operator."""
        n_ops = len(self.operators)
        if self._trial_counts.sum() == 0:
            return np.ones(n_ops) / n_ops

        # Compute success rates with Laplace smoothing
        rates = (self._success_counts + 1) / (self._trial_counts + 2)
        # Softmax
        logits = rates / max(self.temperature, 1e-10)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Enforce minimum probability
        probs = np.maximum(probs, self.min_prob)
        probs /= probs.sum()
        return probs

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Select an operator based on adaptive probabilities and apply it."""
        idx = rng.choice(len(self.operators), p=self.probabilities)
        self._last_operator_idx = idx
        self._trial_counts[idx] += 1
        return self.operators[idx].mutate(dag, rng)

    def report_result(self, was_improvement: bool) -> None:
        """Report whether the last mutation led to an archive improvement.

        Call this after evaluating the mutated offspring.

        Parameters
        ----------
        was_improvement : bool
            True if the offspring was accepted into the archive.
        """
        idx = self._last_operator_idx
        self._history.append((idx, was_improvement))

        if was_improvement:
            self._success_counts[idx] += 1

        # Maintain sliding window
        while len(self._history) > self.window_size:
            old_idx, old_success = self._history.pop(0)
            self._trial_counts[old_idx] -= 1
            if old_success:
                self._success_counts[old_idx] -= 1
            # Clamp to zero
            self._trial_counts[old_idx] = max(0, self._trial_counts[old_idx])
            self._success_counts[old_idx] = max(0, self._success_counts[old_idx])

    def reset(self) -> None:
        """Reset all adaptation statistics."""
        n_ops = len(self.operators)
        self._success_counts = np.zeros(n_ops, dtype=np.float64)
        self._trial_counts = np.zeros(n_ops, dtype=np.float64)
        self._history.clear()
