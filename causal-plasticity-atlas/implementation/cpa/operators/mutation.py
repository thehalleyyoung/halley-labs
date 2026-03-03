"""Targeted mutation operators for DAG structures.

Provides edge-level mutations (add, remove, reverse), weight
perturbation, adaptive mutation, and score-guided targeted mutation
that biases changes toward the most promising edges.

Operators
---------
DAGMutation
    Low-level single-edge mutation primitives.
EdgeMutation
    Composite add/remove/reverse with tunable rate.
WeightMutation
    Gaussian and Cauchy perturbation of edge weights.
StructuralMutation
    Structured mutation with configurable operation probabilities.
AdaptiveMutation
    Mutation rate adapts based on fitness trajectory.
TargetedMutation
    Score-guided targeted mutation.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

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
    visited = set()
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


def _existing_edges(adj: NDArray) -> List[Tuple[int, int]]:
    """Return list of (i,j) where adj[i,j] != 0."""
    rows, cols = np.nonzero(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def _absent_edges(adj: NDArray) -> List[Tuple[int, int]]:
    """Return list of (i,j) where adj[i,j] == 0 and i != j."""
    n = adj.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 0:
                edges.append((i, j))
    return edges


# ------------------------------------------------------------------
# DAGMutation
# ------------------------------------------------------------------


class DAGMutation:
    """Low-level single-edge mutation primitives.

    Parameters
    ----------
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def add_edge(self, dag: NDArray) -> NDArray:
        """Add a random edge while preserving acyclicity."""
        result = dag.copy()
        n = result.shape[0]
        # Collect valid additions (i->j where no path j->i exists)
        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and result[i, j] == 0:
                    if not _has_path(result, j, i):
                        candidates.append((i, j))
        if not candidates:
            return result
        idx = self._rng.integers(0, len(candidates))
        i, j = candidates[idx]
        result[i, j] = 1.0
        return result

    def remove_edge(self, dag: NDArray) -> NDArray:
        """Remove a random existing edge."""
        result = dag.copy()
        edges = _existing_edges(result)
        if not edges:
            return result
        idx = self._rng.integers(0, len(edges))
        i, j = edges[idx]
        result[i, j] = 0.0
        return result

    def reverse_edge(self, dag: NDArray) -> NDArray:
        """Reverse a random existing edge if the result is a DAG."""
        result = dag.copy()
        edges = _existing_edges(result)
        if not edges:
            return result
        self._rng.shuffle(edges)  # type: ignore[arg-type]
        for i, j in edges:
            result[i, j] = 0.0
            if not _has_path(result, i, j):
                result[j, i] = result[i, j] if result[i, j] != 0 else 1.0
                # Correct: copy the weight value
                result[j, i] = dag[i, j]
                return result
            result[i, j] = dag[i, j]
        return result

    def is_valid_dag(self, adj: NDArray) -> bool:
        """Return ``True`` if *adj* is a valid DAG (acyclic)."""
        return _is_acyclic(adj)


# ------------------------------------------------------------------
# EdgeMutation
# ------------------------------------------------------------------


class EdgeMutation:
    """Composite edge mutation: add, remove, or reverse.

    Parameters
    ----------
    mutation_rate : float
        Probability of applying a mutation.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        adj: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Add, remove, or reverse a random edge."""
        rng = rng or self._rng
        if rng.random() > self.mutation_rate:
            return adj.copy()
        op = rng.integers(0, 3)
        if op == 0:
            return self._add_edge(adj, rng)
        elif op == 1:
            return self._remove_edge(adj, rng)
        else:
            return self._reverse_edge(adj, rng)

    def _add_edge(
        self, adj: NDArray, rng: np.random.Generator
    ) -> NDArray:
        """Add edge maintaining acyclicity."""
        result = adj.copy()
        n = result.shape[0]
        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and result[i, j] == 0:
                    if self._is_acyclic_after_add(result, i, j):
                        candidates.append((i, j))
        if not candidates:
            return result
        idx = int(rng.integers(0, len(candidates)))
        i, j = candidates[idx]
        result[i, j] = 1.0
        return result

    def _remove_edge(
        self, adj: NDArray, rng: np.random.Generator
    ) -> NDArray:
        """Remove random existing edge."""
        result = adj.copy()
        edges = _existing_edges(result)
        if not edges:
            return result
        idx = int(rng.integers(0, len(edges)))
        i, j = edges[idx]
        result[i, j] = 0.0
        return result

    def _reverse_edge(
        self, adj: NDArray, rng: np.random.Generator
    ) -> NDArray:
        """Reverse random edge if acyclic."""
        result = adj.copy()
        edges = _existing_edges(result)
        if not edges:
            return result
        order = list(range(len(edges)))
        rng.shuffle(np.array(order))  # type: ignore[arg-type]
        for idx in order:
            i, j = edges[idx]
            w = result[i, j]
            result[i, j] = 0.0
            if not _has_path(result, i, j):
                result[j, i] = w
                return result
            result[i, j] = w
        return result

    def _is_acyclic_after_add(
        self, adj: NDArray, i: int, j: int
    ) -> bool:
        """Check acyclicity of adding i->j via reachability from j to i."""
        if i == j:
            return False
        return not _has_path(adj, j, i)


# ------------------------------------------------------------------
# WeightMutation
# ------------------------------------------------------------------


class WeightMutation:
    """Perturb edge weights via Gaussian or Cauchy noise.

    Parameters
    ----------
    mutation_rate : float
        Probability that each edge weight is perturbed.
    sigma : float
        Standard deviation for Gaussian perturbation.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        sigma: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        coefficients: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Perturb edge weights with Gaussian noise."""
        rng = rng or self._rng
        result = coefficients.copy()
        mask = (result != 0)
        n_edges = int(mask.sum())
        if n_edges == 0:
            return result
        mutate_mask = mask & (rng.random(result.shape) < self.mutation_rate)
        noise = self._gaussian_perturbation(
            np.zeros(result.shape), self.sigma, rng
        )
        result[mutate_mask] += noise[mutate_mask]
        return result

    def _gaussian_perturbation(
        self,
        coeff: NDArray,
        sigma: float,
        rng: np.random.Generator,
    ) -> NDArray:
        """Add Gaussian noise with standard deviation *sigma*."""
        return rng.normal(0.0, sigma, size=coeff.shape)

    def _cauchy_perturbation(
        self,
        coeff: NDArray,
        scale: float,
        rng: np.random.Generator,
    ) -> NDArray:
        """Heavy-tailed perturbation from Cauchy distribution."""
        return scale * np.tan(np.pi * (rng.random(coeff.shape) - 0.5))


# ------------------------------------------------------------------
# StructuralMutation
# ------------------------------------------------------------------


class StructuralMutation:
    """Structured mutation with configurable operation probabilities.

    Parameters
    ----------
    add_prob : float
        Probability of an add-edge operation.
    remove_prob : float
        Probability of a remove-edge operation.
    reverse_prob : float
        Probability of a reverse-edge operation.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        add_prob: float = 0.4,
        remove_prob: float = 0.3,
        reverse_prob: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        total = add_prob + remove_prob + reverse_prob
        self.add_prob = add_prob / total
        self.remove_prob = remove_prob / total
        self.reverse_prob = reverse_prob / total
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        adj: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Apply a structural mutation with configured probabilities."""
        rng = rng or self._rng
        r = rng.random()
        if r < self.add_prob:
            candidates = self._valid_additions(adj)
            if not candidates:
                return adj.copy()
            idx = int(rng.integers(0, len(candidates)))
            result = adj.copy()
            i, j = candidates[idx]
            result[i, j] = 1.0
            return result
        elif r < self.add_prob + self.remove_prob:
            edges = _existing_edges(adj)
            if not edges:
                return adj.copy()
            idx = int(rng.integers(0, len(edges)))
            result = adj.copy()
            i, j = edges[idx]
            result[i, j] = 0.0
            return result
        else:
            candidates = self._valid_reversals(adj)
            if not candidates:
                return adj.copy()
            idx = int(rng.integers(0, len(candidates)))
            result = adj.copy()
            i, j = candidates[idx]
            w = result[i, j]
            result[i, j] = 0.0
            result[j, i] = w
            return result

    def _valid_additions(
        self, adj: NDArray
    ) -> List[Tuple[int, int]]:
        """All valid edge additions maintaining DAG property."""
        n = adj.shape[0]
        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] == 0:
                    if not _has_path(adj, j, i):
                        candidates.append((i, j))
        return candidates

    def _valid_reversals(
        self, adj: NDArray
    ) -> List[Tuple[int, int]]:
        """All valid edge reversals maintaining DAG property."""
        edges = _existing_edges(adj)
        valid: List[Tuple[int, int]] = []
        for i, j in edges:
            tmp = adj.copy()
            tmp[i, j] = 0.0
            if not _has_path(tmp, i, j):
                valid.append((i, j))
        return valid


# ------------------------------------------------------------------
# AdaptiveMutation
# ------------------------------------------------------------------


class AdaptiveMutation:
    """Mutation with rate that adapts based on fitness trajectory.

    Increases rate when fitness stagnates, decreases when improving.

    Parameters
    ----------
    initial_rate : float
        Starting mutation rate.
    adaptation_rate : float
        Speed of rate adaptation.
    min_rate : float
        Lower bound on mutation rate.
    max_rate : float
        Upper bound on mutation rate.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        initial_rate: float = 0.1,
        adaptation_rate: float = 0.01,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.rate = initial_rate
        self.adaptation_rate = adaptation_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._base = EdgeMutation(mutation_rate=initial_rate, seed=seed)
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        adj: NDArray,
        fitness_history: Optional[Sequence[float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Mutate with adapted rate based on *fitness_history*."""
        rng = rng or self._rng
        if fitness_history is not None and len(fitness_history) >= 2:
            self.rate = self._adapt_rate(fitness_history)
        self._base.mutation_rate = self.rate
        return self._base.mutate(adj, rng)

    def _adapt_rate(self, fitness_history: Sequence[float]) -> float:
        """Adjust mutation rate: up if stagnant, down if improving."""
        recent = fitness_history[-5:] if len(fitness_history) >= 5 else list(fitness_history)
        if len(recent) < 2:
            return self.rate
        improvement = recent[-1] - recent[0]
        std = float(np.std(recent))
        if std < 1e-10 or improvement <= 0:
            # Stagnant or worsening: increase rate
            new_rate = self.rate + self.adaptation_rate
        else:
            # Improving: decrease rate
            new_rate = self.rate - self.adaptation_rate
        return float(np.clip(new_rate, self.min_rate, self.max_rate))


# ------------------------------------------------------------------
# TargetedMutation
# ------------------------------------------------------------------


class TargetedMutation:
    """Score-guided targeted mutation.

    Biases mutations toward edges whose addition, removal, or reversal
    yields the largest score improvement.

    Parameters
    ----------
    mutation_rate : float
        Expected fraction of edges mutated per call.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        dag: NDArray,
        scores: Optional[NDArray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Apply random mutations, optionally guided by *scores*.

        If *scores* is provided it should be a matrix of the same shape
        as *dag* where higher values indicate more promising edge
        changes.  Mutation operations are biased toward high-score
        positions.
        """
        rng = rng or self._rng
        result = dag.copy()
        n = result.shape[0]
        if n < 2:
            return result

        n_mutations = max(1, int(self.mutation_rate * n * n))

        for _ in range(n_mutations):
            op = int(rng.integers(0, 3))
            if scores is not None:
                # Bias selection toward high-score positions
                flat_scores = np.abs(scores).ravel()
                total = flat_scores.sum()
                if total > 0:
                    probs = flat_scores / total
                    flat_idx = rng.choice(n * n, p=probs)
                    i, j = divmod(int(flat_idx), n)
                else:
                    i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            else:
                i, j = int(rng.integers(0, n)), int(rng.integers(0, n))

            if i == j:
                continue

            if op == 0 and result[i, j] == 0:
                if not _has_path(result, j, i):
                    result[i, j] = 1.0
            elif op == 1 and result[i, j] != 0:
                result[i, j] = 0.0
            elif op == 2 and result[i, j] != 0:
                w = result[i, j]
                result[i, j] = 0.0
                if not _has_path(result, i, j) and result[j, i] == 0:
                    result[j, i] = w
                else:
                    result[i, j] = w

        return result

    def score_guided_mutation(
        self,
        dag: NDArray,
        score_fn: Callable[[int, Sequence[int]], float],
    ) -> NDArray:
        """Mutate *dag* by greedily maximising *score_fn*.

        Evaluates all single-edge operations and applies the one with
        the largest score improvement.
        """
        result = dag.copy()
        n = result.shape[0]
        best_gain = 0.0
        best_op: Optional[Tuple[str, int, int]] = None

        for j in range(n):
            parents_j = [int(p) for p in range(n) if result[p, j] != 0]
            current_score = score_fn(j, parents_j)

            # Try adding each absent parent
            for i in range(n):
                if i != j and result[i, j] == 0:
                    if not _has_path(result, j, i):
                        new_parents = sorted(parents_j + [i])
                        gain = score_fn(j, new_parents) - current_score
                        if gain > best_gain:
                            best_gain = gain
                            best_op = ("add", i, j)

            # Try removing each existing parent
            for i in parents_j:
                new_parents = [p for p in parents_j if p != i]
                gain = score_fn(j, new_parents) - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_op = ("remove", i, j)

        # Try reversals
        edges = _existing_edges(result)
        for i, j in edges:
            parents_j = [int(p) for p in range(n) if result[p, j] != 0]
            parents_i = [int(p) for p in range(n) if result[p, i] != 0]
            score_before = score_fn(j, parents_j) + score_fn(i, parents_i)

            tmp = result.copy()
            tmp[i, j] = 0.0
            if not _has_path(tmp, i, j) and tmp[j, i] == 0:
                new_parents_j = [p for p in parents_j if p != i]
                new_parents_i = sorted(parents_i + [j])
                score_after = score_fn(j, new_parents_j) + score_fn(i, new_parents_i)
                gain = score_after - score_before
                if gain > best_gain:
                    best_gain = gain
                    best_op = ("reverse", i, j)

        if best_op is not None:
            op_type, i, j = best_op
            if op_type == "add":
                result[i, j] = 1.0
            elif op_type == "remove":
                result[i, j] = 0.0
            elif op_type == "reverse":
                w = result[i, j]
                result[i, j] = 0.0
                result[j, i] = w

        return result
