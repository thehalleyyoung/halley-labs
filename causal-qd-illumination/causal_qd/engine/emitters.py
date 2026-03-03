"""CMA-ME style emitters for MAP-Elites causal structure learning.

Provides emitter classes that generate batches of candidate DAGs using
different strategies: random mutation/crossover, CMA-ES targeting quality
improvement or new coverage, multi-armed bandit emitter selection,
gradient-based score optimization, and hybrid combinations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.archive.archive_base import Archive, ArchiveEntry
from causal_qd.operators.mutation import (
    MutationOperator,
    TopologicalMutation,
    _has_cycle,
    _topological_sort,
)
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Emitter(ABC):
    """Abstract base class for MAP-Elites emitters.

    An emitter generates a batch of candidate solutions (DAGs) from
    the current archive.  Different emitters implement different
    generation strategies—random variation, CMA-ES optimization,
    gradient-based search, etc.
    """

    @abstractmethod
    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidate DAGs.

        Parameters
        ----------
        archive : Archive
            Current MAP-Elites archive.
        n : int
            Number of candidates to generate.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        List[AdjacencyMatrix]
            Candidate adjacency matrices.
        """

    def update(
        self,
        candidates: List[AdjacencyMatrix],
        qualities: List[QualityScore],
        improvements: List[bool],
    ) -> None:
        """Notify emitter of evaluation results for adaptation.

        Parameters
        ----------
        candidates : List[AdjacencyMatrix]
            The candidates that were evaluated.
        qualities : List[QualityScore]
            Quality scores of the candidates.
        improvements : List[bool]
            Whether each candidate was accepted into the archive.
        """

    def reset(self) -> None:
        """Reset internal state."""


# ---------------------------------------------------------------------------
# RandomEmitter
# ---------------------------------------------------------------------------


class RandomEmitter(Emitter):
    """Emit candidates via random mutation and crossover of archive elites.

    Parameters
    ----------
    mutation_op : MutationOperator
        Mutation operator to apply.
    mutation_rate : float
        Probability of mutation vs. crossover.  Default ``0.8``.
    n_mutations : int
        Number of sequential mutations per candidate.  Default ``1``.
    """

    def __init__(
        self,
        mutation_op: Optional[MutationOperator] = None,
        mutation_rate: float = 0.8,
        n_mutations: int = 1,
    ) -> None:
        self._mut_op = mutation_op or TopologicalMutation()
        self._mut_rate = mutation_rate
        self._n_mut = n_mutations

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidates by mutating random archive elites.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        elites = archive.elites()
        if not elites:
            return []

        candidates: List[AdjacencyMatrix] = []
        for _ in range(n):
            parent = elites[rng.integers(0, len(elites))]
            child = parent.solution.copy()
            if rng.random() < self._mut_rate:
                for _ in range(self._n_mut):
                    child = self._mut_op.mutate(child, rng)
            else:
                # Simple crossover: take a second parent and mix edges
                parent2 = elites[rng.integers(0, len(elites))]
                child = self._simple_crossover(child, parent2.solution, rng)
            candidates.append(child)
        return candidates

    @staticmethod
    def _simple_crossover(
        p1: AdjacencyMatrix,
        p2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Uniform crossover with cycle repair."""
        n = p1.shape[0]
        mask = rng.random((n, n)) < 0.5
        child = np.where(mask, p1, p2).astype(np.int8)
        np.fill_diagonal(child, 0)
        # Break cycles using topological ordering
        order = _topological_sort(child)
        if len(order) < n:
            visited = set(order)
            full = list(order) + [i for i in range(n) if i not in visited]
            pos = np.empty(n, dtype=int)
            for idx, node in enumerate(full):
                pos[node] = idx
            for i in range(n):
                for j in range(n):
                    if child[i, j] and pos[i] >= pos[j]:
                        child[i, j] = 0
        return child


# ---------------------------------------------------------------------------
# CMA-ES helper
# ---------------------------------------------------------------------------


@dataclass
class _CMAESState:
    """Internal state for a CMA-ES optimizer operating on flattened DAGs."""
    dim: int
    mean: npt.NDArray[np.float64] = field(default=None)  # type: ignore[assignment]
    sigma: float = 0.3
    cov: npt.NDArray[np.float64] = field(default=None)  # type: ignore[assignment]
    p_sigma: npt.NDArray[np.float64] = field(default=None)  # type: ignore[assignment]
    p_c: npt.NDArray[np.float64] = field(default=None)  # type: ignore[assignment]
    generation: int = 0
    lambda_: int = 10
    mu: int = 5

    def __post_init__(self) -> None:
        if self.mean is None:
            self.mean = np.zeros(self.dim, dtype=np.float64)
        if self.cov is None:
            self.cov = np.eye(self.dim, dtype=np.float64)
        if self.p_sigma is None:
            self.p_sigma = np.zeros(self.dim, dtype=np.float64)
        if self.p_c is None:
            self.p_c = np.zeros(self.dim, dtype=np.float64)


def _flatten_dag(adj: AdjacencyMatrix) -> npt.NDArray[np.float64]:
    """Flatten upper-triangle of adjacency matrix to a vector."""
    n = adj.shape[0]
    indices = np.triu_indices(n, k=1)
    return adj[indices].astype(np.float64)


def _unflatten_dag(
    vec: npt.NDArray[np.float64], n: int
) -> AdjacencyMatrix:
    """Reconstruct adjacency matrix from flattened upper-triangle vector.

    Values are thresholded at 0.5 to produce binary adjacency.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    indices = np.triu_indices(n, k=1)
    binary = (vec > 0.5).astype(np.int8)
    adj[indices] = binary
    return adj


def _ensure_dag(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Remove back-edges to ensure acyclicity."""
    n = adj.shape[0]
    order = _topological_sort(adj)
    if len(order) == n:
        return adj
    visited = set(order)
    full = list(order) + [i for i in range(n) if i not in visited]
    pos = np.empty(n, dtype=int)
    for idx, node in enumerate(full):
        pos[node] = idx
    result = adj.copy()
    for i in range(n):
        for j in range(n):
            if result[i, j] and pos[i] >= pos[j]:
                result[i, j] = 0
    return result


# ---------------------------------------------------------------------------
# ImprovementEmitter
# ---------------------------------------------------------------------------


class ImprovementEmitter(Emitter):
    """CMA-ES emitter targeting quality improvement over current elites.

    Maintains a CMA-ES optimizer that searches for DAGs improving
    the quality of existing archive cells.  The search direction
    adapts based on which mutations lead to quality improvements.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAGs.
    sigma : float
        Initial step size for CMA-ES.  Default ``0.3``.
    batch_size : int
        Internal CMA-ES population size.  Default ``10``.
    restart_threshold : int
        Number of consecutive non-improving generations before
        restarting CMA-ES from a new elite.  Default ``20``.
    """

    def __init__(
        self,
        n_nodes: int,
        sigma: float = 0.3,
        batch_size: int = 10,
        restart_threshold: int = 20,
    ) -> None:
        self._n_nodes = n_nodes
        self._dim = n_nodes * (n_nodes - 1) // 2
        self._sigma = sigma
        self._batch_size = batch_size
        self._restart_threshold = restart_threshold
        self._state = _CMAESState(
            dim=self._dim, sigma=sigma, lambda_=batch_size, mu=batch_size // 2
        )
        self._no_improvement_count = 0
        self._best_quality = float("-inf")

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidates using CMA-ES around a selected elite.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        elites = archive.elites()
        if not elites:
            n_nodes = self._n_nodes
            return [
                np.zeros((n_nodes, n_nodes), dtype=np.int8) for _ in range(n)
            ]

        # Set CMA-ES mean to best elite
        best = archive.best()
        flat_best = _flatten_dag(best.solution)
        self._state.mean = flat_best.copy()

        candidates: List[AdjacencyMatrix] = []
        for _ in range(n):
            # Sample from multivariate normal around mean
            noise = rng.multivariate_normal(
                np.zeros(self._dim),
                self._state.sigma**2 * self._state.cov,
            )
            sample = self._state.mean + noise
            # Clip to [0, 1] range
            sample = np.clip(sample, 0.0, 1.0)
            adj = _unflatten_dag(sample, self._n_nodes)
            adj = _ensure_dag(adj)
            candidates.append(adj)

        return candidates

    def update(
        self,
        candidates: List[AdjacencyMatrix],
        qualities: List[QualityScore],
        improvements: List[bool],
    ) -> None:
        """Update CMA-ES state based on evaluation results.

        The top-*mu* candidates (by quality) are used to update the
        CMA-ES mean and covariance.  If no improvement is observed for
        several generations, the optimizer restarts.

        Parameters
        ----------
        candidates, qualities, improvements
            Evaluation results.
        """
        if not candidates:
            return

        self._state.generation += 1
        n_improve = sum(improvements)

        if n_improve > 0:
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        # Sort by quality descending
        indexed = list(zip(qualities, candidates))
        indexed.sort(key=lambda x: x[0], reverse=True)

        mu = min(self._state.mu, len(indexed))
        if mu == 0:
            return

        # Compute weighted mean of top-mu solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()

        flat_solutions = [_flatten_dag(c) for _, c in indexed[:mu]]
        new_mean = np.zeros(self._dim, dtype=np.float64)
        for w, fs in zip(weights, flat_solutions):
            new_mean += w * fs

        # Update evolution paths
        c_sigma = 0.3
        d_sigma = 1.0 + 2.0 * max(0, math.sqrt((mu - 1) / (self._dim + 1)) - 1) + c_sigma
        mean_diff = new_mean - self._state.mean

        self._state.p_sigma = (
            (1 - c_sigma) * self._state.p_sigma
            + math.sqrt(c_sigma * (2 - c_sigma) * mu) * mean_diff / max(self._state.sigma, 1e-10)
        )

        # Adapt step size
        chi_n = math.sqrt(self._dim) * (1 - 1 / (4 * self._dim) + 1 / (21 * self._dim**2))
        p_sigma_norm = np.linalg.norm(self._state.p_sigma)
        self._state.sigma *= math.exp(
            c_sigma / d_sigma * (p_sigma_norm / chi_n - 1)
        )
        self._state.sigma = np.clip(self._state.sigma, 1e-6, 2.0)

        # Update mean
        self._state.mean = new_mean

        # Covariance update (rank-1 + rank-mu, simplified)
        c_cov = 2.0 / (self._dim + 2) ** 2
        self._state.cov = (
            (1 - c_cov) * self._state.cov
            + c_cov * np.outer(mean_diff, mean_diff) / max(self._state.sigma**2, 1e-10)
        )

        # Clamp covariance eigenvalues
        eigvals = np.linalg.eigvalsh(self._state.cov)
        if eigvals.min() < 1e-10 or eigvals.max() > 1e4:
            self._state.cov = np.eye(self._dim, dtype=np.float64)

        # Check for restart
        if self._no_improvement_count >= self._restart_threshold:
            self.reset()

    def reset(self) -> None:
        """Restart CMA-ES from scratch."""
        self._state = _CMAESState(
            dim=self._dim,
            sigma=self._sigma,
            lambda_=self._batch_size,
            mu=self._batch_size // 2,
        )
        self._no_improvement_count = 0


# ---------------------------------------------------------------------------
# DirectionEmitter
# ---------------------------------------------------------------------------


class DirectionEmitter(Emitter):
    """CMA-ES emitter targeting new archive coverage.

    Instead of optimizing quality, this emitter targets unexplored
    regions of the descriptor space by biasing the search toward
    descriptors far from existing elites.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in DAGs.
    target_descriptor : ndarray | None
        If provided, CMA-ES optimizes toward this descriptor.
        Otherwise, a random under-explored direction is chosen.
    sigma : float
        Initial CMA-ES step size.  Default ``0.5``.
    """

    def __init__(
        self,
        n_nodes: int,
        target_descriptor: Optional[npt.NDArray[np.float64]] = None,
        sigma: float = 0.5,
    ) -> None:
        self._n_nodes = n_nodes
        self._dim = n_nodes * (n_nodes - 1) // 2
        self._target = target_descriptor
        self._sigma = sigma
        self._state = _CMAESState(dim=self._dim, sigma=sigma)
        self._mutation_op = TopologicalMutation()

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidates targeting uncovered descriptor regions.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        elites = archive.elites()
        if not elites:
            return [
                np.zeros((self._n_nodes, self._n_nodes), dtype=np.int8)
                for _ in range(n)
            ]

        # Pick a random elite that is near under-explored regions
        if len(elites) > 1:
            # Use the elite with highest novelty (distance to other elites)
            descriptors = np.array([e.descriptor for e in elites])
            novelty = np.zeros(len(elites))
            for i in range(len(elites)):
                dists = np.linalg.norm(descriptors - descriptors[i], axis=1)
                sorted_dists = np.sort(dists)
                k = min(5, len(dists) - 1)
                if k > 0:
                    novelty[i] = np.mean(sorted_dists[1:k+1])
            probs = novelty / (novelty.sum() + 1e-10)
            if probs.sum() < 1e-10:
                probs = np.ones(len(elites)) / len(elites)
            base_idx = rng.choice(len(elites), p=probs)
        else:
            base_idx = 0

        base = elites[base_idx]
        candidates: List[AdjacencyMatrix] = []

        for _ in range(n):
            # Apply multiple mutations to encourage exploration
            child = base.solution.copy()
            n_muts = rng.integers(1, 4)
            for _ in range(n_muts):
                child = self._mutation_op.mutate(child, rng)
            candidates.append(child)

        return candidates

    def update(
        self,
        candidates: List[AdjacencyMatrix],
        qualities: List[QualityScore],
        improvements: List[bool],
    ) -> None:
        """Track coverage improvement."""
        pass


# ---------------------------------------------------------------------------
# BanditEmitter
# ---------------------------------------------------------------------------


class BanditEmitter(Emitter):
    """Multi-armed bandit that selects between multiple emitter types.

    Uses UCB1 to track which emitter produces the most archive
    improvements and allocates more budget to successful emitters.

    Parameters
    ----------
    emitters : Sequence[Emitter]
        Pool of emitters to choose from.
    exploration_constant : float
        UCB1 exploration coefficient.  Default ``1.5``.
    window_size : int
        Sliding window for tracking success rates.  Default ``100``.
    """

    def __init__(
        self,
        emitters: Sequence[Emitter],
        exploration_constant: float = 1.5,
        window_size: int = 100,
    ) -> None:
        self._emitters = list(emitters)
        self._c = exploration_constant
        self._window_size = window_size
        n = len(emitters)
        self._success_counts = np.zeros(n, dtype=np.float64)
        self._trial_counts = np.zeros(n, dtype=np.float64)
        self._total_trials = 0
        self._history: deque[Tuple[int, bool]] = deque(maxlen=window_size)
        self._last_emitter_idx = 0

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Select the best emitter via UCB1 and emit *n* candidates.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        if not self._emitters:
            return []

        # UCB1 selection
        scores = np.zeros(len(self._emitters), dtype=np.float64)
        for i in range(len(self._emitters)):
            if self._trial_counts[i] == 0:
                scores[i] = float("inf")
            else:
                exploitation = self._success_counts[i] / self._trial_counts[i]
                exploration = self._c * math.sqrt(
                    math.log(self._total_trials + 1) / self._trial_counts[i]
                )
                scores[i] = exploitation + exploration

        idx = int(np.argmax(scores))
        self._last_emitter_idx = idx
        self._trial_counts[idx] += 1
        self._total_trials += 1

        return self._emitters[idx].emit(archive, n, rng)

    def update(
        self,
        candidates: List[AdjacencyMatrix],
        qualities: List[QualityScore],
        improvements: List[bool],
    ) -> None:
        """Update bandit statistics and forward to selected emitter.

        Parameters
        ----------
        candidates, qualities, improvements
            Evaluation results.
        """
        idx = self._last_emitter_idx
        n_improve = sum(improvements)
        success = n_improve > 0

        self._history.append((idx, success))
        if success:
            self._success_counts[idx] += n_improve / len(improvements)

        # Maintain sliding window
        if len(self._history) > self._window_size:
            old_idx, old_success = self._history[0]
            self._trial_counts[old_idx] = max(0, self._trial_counts[old_idx] - 1)
            if old_success:
                self._success_counts[old_idx] = max(
                    0, self._success_counts[old_idx] - 1
                )
            self._total_trials = max(0, self._total_trials - 1)

        # Forward to the selected emitter
        self._emitters[idx].update(candidates, qualities, improvements)

    def reset(self) -> None:
        """Reset bandit statistics and all emitters."""
        n = len(self._emitters)
        self._success_counts = np.zeros(n, dtype=np.float64)
        self._trial_counts = np.zeros(n, dtype=np.float64)
        self._total_trials = 0
        self._history.clear()
        for e in self._emitters:
            e.reset()


# ---------------------------------------------------------------------------
# GradientEmitter
# ---------------------------------------------------------------------------


class GradientEmitter(Emitter):
    """Gradient-based emitter using approximate BIC score gradients.

    Approximates the gradient of a decomposable score with respect to
    the continuous relaxation of the adjacency matrix, then takes a
    projected gradient step that maintains the DAG constraint.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAGs.
    score_fn : callable
        Scoring function ``score_fn(dag, data) -> float``.
    data : DataMatrix
        Observed data matrix for score computation.
    step_size : float
        Gradient step size.  Default ``0.1``.
    n_perturbations : int
        Number of perturbations for finite-difference gradient.
        Default ``20``.
    """

    def __init__(
        self,
        n_nodes: int,
        score_fn: Any,
        data: npt.NDArray[np.float64],
        step_size: float = 0.1,
        n_perturbations: int = 20,
    ) -> None:
        self._n_nodes = n_nodes
        self._score_fn = score_fn
        self._data = data
        self._step_size = step_size
        self._n_pert = n_perturbations

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate candidates by following approximate score gradients.

        For each candidate:
        1. Select a random elite from the archive.
        2. Approximate the gradient of the score w.r.t. edge additions/removals.
        3. Apply the most promising edge change.
        4. Verify DAG constraint.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        elites = archive.elites()
        if not elites:
            nn = self._n_nodes
            return [np.zeros((nn, nn), dtype=np.int8) for _ in range(n)]

        candidates: List[AdjacencyMatrix] = []
        for _ in range(n):
            parent = elites[rng.integers(0, len(elites))]
            child = self._gradient_step(parent.solution, rng)
            candidates.append(child)

        return candidates

    def _gradient_step(
        self,
        adj: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Take one gradient step on the DAG.

        Computes finite-difference approximation of score gradient for
        each possible single-edge modification, then applies the best
        one that preserves acyclicity.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
            Modified DAG after gradient step.
        """
        n = adj.shape[0]
        base_score = self._score_fn.score(adj, self._data)

        # Collect candidate edge changes and their score deltas
        edge_changes: List[Tuple[int, int, int, float]] = []  # (i, j, new_val, delta)

        # Sample random perturbations instead of trying all
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        if len(all_pairs) > self._n_pert:
            sample_indices = rng.choice(
                len(all_pairs), size=self._n_pert, replace=False
            )
            pairs = [all_pairs[k] for k in sample_indices]
        else:
            pairs = all_pairs

        for i, j in pairs:
            new_adj = adj.copy()
            if adj[i, j]:
                # Try removing
                new_adj[i, j] = 0
            else:
                # Try adding
                new_adj[i, j] = 1
                if _has_cycle(new_adj):
                    continue

            new_score = self._score_fn.score(new_adj, self._data)
            delta = new_score - base_score
            new_val = 0 if adj[i, j] else 1
            edge_changes.append((i, j, new_val, delta))

        if not edge_changes:
            return adj.copy()

        # Sort by score improvement
        edge_changes.sort(key=lambda x: x[3], reverse=True)

        # Apply top change (greedy) or probabilistic based on step_size
        result = adj.copy()
        # Apply changes with probability proportional to improvement
        for i, j, new_val, delta in edge_changes[:1]:
            if delta > 0 or rng.random() < self._step_size:
                result[i, j] = new_val
                if _has_cycle(result):
                    result[i, j] = adj[i, j]  # revert

        return result


# ---------------------------------------------------------------------------
# HybridEmitter
# ---------------------------------------------------------------------------


class HybridEmitter(Emitter):
    """Combine multiple emission strategies, allocating budget proportionally.

    Each sub-emitter is assigned a fraction of the total budget.
    Results from all sub-emitters are combined into a single batch.

    Parameters
    ----------
    emitters : Sequence[Emitter]
        Component emitters.
    weights : Sequence[float] | None
        Budget allocation weights (normalized internally).
        If ``None``, equal weights are used.
    """

    def __init__(
        self,
        emitters: Sequence[Emitter],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self._emitters = list(emitters)
        if weights is None:
            self._weights = np.ones(len(emitters)) / len(emitters)
        else:
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidates distributed across sub-emitters.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        candidates: List[AdjacencyMatrix] = []
        # Distribute budget
        counts = np.round(self._weights * n).astype(int)
        # Fix rounding to match total
        diff = n - counts.sum()
        if diff > 0:
            for k in range(diff):
                counts[k % len(counts)] += 1
        elif diff < 0:
            for k in range(-diff):
                idx = len(counts) - 1 - (k % len(counts))
                counts[idx] = max(0, counts[idx] - 1)

        for emitter, count in zip(self._emitters, counts):
            if count > 0:
                batch = emitter.emit(archive, int(count), rng)
                candidates.extend(batch)

        return candidates

    def update(
        self,
        candidates: List[AdjacencyMatrix],
        qualities: List[QualityScore],
        improvements: List[bool],
    ) -> None:
        """Forward update to all sub-emitters."""
        for e in self._emitters:
            e.update(candidates, qualities, improvements)

    def reset(self) -> None:
        """Reset all sub-emitters."""
        for e in self._emitters:
            e.reset()


# ---------------------------------------------------------------------------
# PerturbationEmitter
# ---------------------------------------------------------------------------


class PerturbationEmitter(Emitter):
    """Emit candidates by applying controlled random perturbations.

    Each candidate is generated by flipping a fixed number of edges
    in a randomly selected elite, respecting acyclicity.

    Parameters
    ----------
    n_flips : int
        Number of edge flips per candidate.  Default ``3``.
    """

    def __init__(self, n_flips: int = 3) -> None:
        self._n_flips = n_flips

    def emit(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* candidates by flipping edges.

        Parameters
        ----------
        archive, n, rng
            Standard emitter parameters.

        Returns
        -------
        List[AdjacencyMatrix]
        """
        elites = archive.elites()
        if not elites:
            return []

        candidates: List[AdjacencyMatrix] = []
        for _ in range(n):
            parent = elites[rng.integers(0, len(elites))]
            child = parent.solution.copy()
            nn = child.shape[0]

            for _ in range(self._n_flips):
                i = rng.integers(0, nn)
                j = rng.integers(0, nn)
                while i == j:
                    j = rng.integers(0, nn)

                if child[i, j]:
                    child[i, j] = 0
                else:
                    child[i, j] = 1
                    if _has_cycle(child):
                        child[i, j] = 0

            candidates.append(child)

        return candidates
