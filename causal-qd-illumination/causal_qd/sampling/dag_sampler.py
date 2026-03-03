"""DAG sampling methods for population initialisation and posterior inference.

Provides uniform sampling via the Erdős-Rényi-DAG model, score-weighted
importance sampling, and bootstrap-based sampling for uncertainty estimation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import (
    AdjacencyMatrix,
    BootstrapSample,
    DataMatrix,
    QualityScore,
    TopologicalOrder,
)
from causal_qd.scores.score_base import DecomposableScore, ScoreFunction
from causal_qd.operators.mutation import _has_cycle, _topological_sort


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class DAGSampler(ABC):
    """Abstract base class for DAG samplers."""

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Draw *n_samples* DAGs.

        Parameters
        ----------
        n_samples : int
            Number of DAGs to generate.
        rng : optional
            NumPy random generator.

        Returns
        -------
        List[AdjacencyMatrix]
        """


# ---------------------------------------------------------------------------
# Uniform DAG sampler
# ---------------------------------------------------------------------------

class UniformDAGSampler(DAGSampler):
    """Sample DAGs (approximately) uniformly at random.

    Uses the Erdős-Rényi-DAG model: draw a random topological ordering, then
    include each forward-consistent edge independently with probability
    *edge_prob*.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAG.
    edge_prob : float
        Independent probability of including each forward edge.
    max_parents : int
        Hard cap on in-degree.  ``-1`` means no limit.
    enforce_connected : bool
        If True, reject DAGs whose underlying skeleton has > 1 connected
        component.
    """

    def __init__(
        self,
        n_nodes: int,
        edge_prob: float = 0.3,
        max_parents: int = -1,
        enforce_connected: bool = False,
    ) -> None:
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must be in [0, 1]")
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.max_parents = max_parents
        self.enforce_connected = enforce_connected

    def _sample_one(self, rng: np.random.Generator) -> AdjacencyMatrix:
        n = self.n_nodes
        order = list(rng.permutation(n))
        adj = np.zeros((n, n), dtype=np.int8)

        for idx, node in enumerate(order):
            predecessors = order[:idx]
            if not predecessors:
                continue
            for pred in predecessors:
                if rng.random() < self.edge_prob:
                    adj[pred, node] = 1

            # Enforce max_parents
            if self.max_parents > 0:
                parents = np.where(adj[:, node])[0]
                if len(parents) > self.max_parents:
                    keep = rng.choice(parents, size=self.max_parents, replace=False)
                    adj[:, node] = 0
                    for p in keep:
                        adj[p, node] = 1

        return adj

    @staticmethod
    def _is_connected(adj: AdjacencyMatrix) -> bool:
        """Check if the skeleton is connected via BFS."""
        n = adj.shape[0]
        if n <= 1:
            return True
        skeleton = adj | adj.T
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop()
            for nb in range(n):
                if skeleton[node, nb] and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == n

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        rng = rng or np.random.default_rng()
        dags: List[AdjacencyMatrix] = []
        max_attempts = n_samples * 100

        attempts = 0
        while len(dags) < n_samples and attempts < max_attempts:
            dag = self._sample_one(rng)
            attempts += 1
            if self.enforce_connected and not self._is_connected(dag):
                continue
            dags.append(dag)

        return dags

    def sample_with_density(
        self,
        n_samples: int,
        target_density: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Sample DAGs targeting a specific edge density.

        Parameters
        ----------
        n_samples : int
            Number of DAGs.
        target_density : float
            Target density = num_edges / (n*(n-1)/2).
        rng : optional
            NumPy random generator.
        """
        rng = rng or np.random.default_rng()
        n = self.n_nodes
        max_edges = n * (n - 1) // 2
        # Adjust edge_prob to target the density
        ep = target_density if max_edges > 0 else 0.0
        saved = self.edge_prob
        self.edge_prob = min(1.0, max(0.0, ep))
        result = self.sample(n_samples, rng)
        self.edge_prob = saved
        return result


# ---------------------------------------------------------------------------
# Score-weighted importance sampler
# ---------------------------------------------------------------------------

@dataclass
class ImportanceSampleResult:
    """Result of importance-weighted DAG sampling."""
    dags: List[AdjacencyMatrix]
    scores: List[float]
    weights: npt.NDArray[np.float64]
    effective_sample_size: float
    best_dag: AdjacencyMatrix
    best_score: float


class ScoreWeightedSampler(DAGSampler):
    """Importance sampling of DAGs weighted by a scoring function.

    Draws DAGs from a proposal distribution (uniform DAG sampler) and
    assigns importance weights proportional to exp(score - max_score) to
    correct for the difference between the proposal and the target.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    score_fn : ScoreFunction
        Scoring function for evaluating DAGs.
    proposal_edge_prob : float
        Edge probability for the uniform proposal.
    max_parents : int
        Maximum in-degree.
    temperature : float
        Softmax temperature for the weights.
    """

    def __init__(
        self,
        n_nodes: int,
        score_fn: ScoreFunction,
        proposal_edge_prob: float = 0.3,
        max_parents: int = -1,
        temperature: float = 1.0,
    ) -> None:
        self.n_nodes = n_nodes
        self.score_fn = score_fn
        self.proposal_edge_prob = proposal_edge_prob
        self.max_parents = max_parents
        self.temperature = max(temperature, 1e-12)
        self._proposal = UniformDAGSampler(
            n_nodes, proposal_edge_prob, max_parents,
        )

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Draw DAGs (unweighted convenience wrapper)."""
        result = self.sample_weighted(n_samples, data=None, rng=rng)
        return result.dags

    def sample_weighted(
        self,
        n_samples: int,
        data: Optional[DataMatrix] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ImportanceSampleResult:
        """Draw *n_samples* DAGs with importance weights.

        Parameters
        ----------
        n_samples : int
            Number of proposals to draw.
        data : DataMatrix or None
            Data matrix for score evaluation.  If ``None``, scores are set to
            zero (uniform weights).
        rng : optional
            NumPy random generator.

        Returns
        -------
        ImportanceSampleResult
        """
        rng = rng or np.random.default_rng()
        dags = self._proposal.sample(n_samples, rng)
        scores: List[float] = []

        for dag in dags:
            if data is not None:
                s = self.score_fn.score(dag, data)
            else:
                s = 0.0
            scores.append(s)

        scores_arr = np.array(scores, dtype=np.float64)
        # Numerically stable softmax for weights
        shifted = (scores_arr - scores_arr.max()) / self.temperature
        exp_w = np.exp(shifted)
        weights = exp_w / exp_w.sum()

        # Effective sample size
        ess = 1.0 / np.sum(weights ** 2) if np.any(weights > 0) else 0.0

        best_idx = int(np.argmax(scores_arr))

        return ImportanceSampleResult(
            dags=dags,
            scores=scores,
            weights=weights,
            effective_sample_size=ess,
            best_dag=dags[best_idx],
            best_score=scores[best_idx],
        )

    def resample(
        self,
        result: ImportanceSampleResult,
        n_resample: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Resample DAGs with replacement according to importance weights.

        This converts the weighted sample into an approximately unweighted
        sample from the target distribution.
        """
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(result.dags), size=n_resample, replace=True, p=result.weights)
        return [result.dags[i] for i in indices]

    def edge_probabilities(
        self,
        result: ImportanceSampleResult,
    ) -> npt.NDArray[np.float64]:
        """Compute weighted edge inclusion probabilities."""
        n = result.dags[0].shape[0]
        probs = np.zeros((n, n), dtype=np.float64)
        for dag, w in zip(result.dags, result.weights):
            probs += w * dag.astype(np.float64)
        return probs


# ---------------------------------------------------------------------------
# Bootstrap sampler
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Result of bootstrap-based DAG sampling."""
    dags: List[AdjacencyMatrix]
    scores: List[float]
    edge_inclusion_probs: npt.NDArray[np.float64]
    confidence_edges: npt.NDArray[np.bool_]
    confidence_level: float


class BootstrapSampler:
    """Bootstrap-based DAG stability assessment.

    For each bootstrap replicate, resamples the data with replacement and
    runs a score-based structure learning procedure (greedy hill-climbing
    from a random initial DAG) to obtain a DAG.  The collection of DAGs
    gives edge stability (inclusion probability) estimates.

    Parameters
    ----------
    score_fn : DecomposableScore
        Decomposable score used for greedy search.
    max_parents : int
        Maximum in-degree.
    greedy_restarts : int
        Number of random restarts per bootstrap replicate.
    greedy_max_iter : int
        Maximum iterations in greedy hill climbing.
    """

    def __init__(
        self,
        score_fn: DecomposableScore,
        max_parents: int = 5,
        greedy_restarts: int = 1,
        greedy_max_iter: int = 200,
    ) -> None:
        self.score_fn = score_fn
        self.max_parents = max_parents
        self.greedy_restarts = greedy_restarts
        self.greedy_max_iter = greedy_max_iter

    def _greedy_search(
        self,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, float]:
        """Simple greedy hill-climbing search over DAGs."""
        p = data.shape[1]
        best_dag = np.zeros((p, p), dtype=np.int8)
        best_score = self.score_fn.score(best_dag, data)

        for _ in range(self.greedy_restarts):
            # Start from a random sparse DAG
            order = list(rng.permutation(p))
            adj = np.zeros((p, p), dtype=np.int8)
            for idx, node in enumerate(order):
                for pred in order[:idx]:
                    if rng.random() < 0.2:
                        adj[pred, node] = 1
                parents = np.where(adj[:, node])[0]
                if self.max_parents > 0 and len(parents) > self.max_parents:
                    keep = rng.choice(parents, size=self.max_parents, replace=False)
                    adj[:, node] = 0
                    for k in keep:
                        adj[k, node] = 1

            current = adj.copy()
            current_score = self.score_fn.score(current, data)

            for _ in range(self.greedy_max_iter):
                improved = False
                # Try single-edge operations
                for i in range(p):
                    for j in range(p):
                        if i == j:
                            continue
                        # Try add / remove / reverse
                        candidates = []

                        if current[i, j] == 0:
                            # Add i -> j
                            trial = current.copy()
                            trial[i, j] = 1
                            if not _has_cycle(trial):
                                parents_j = np.where(trial[:, j])[0]
                                if self.max_parents < 0 or len(parents_j) <= self.max_parents:
                                    candidates.append(trial)

                        if current[i, j] == 1:
                            # Remove i -> j
                            trial = current.copy()
                            trial[i, j] = 0
                            candidates.append(trial)

                            # Reverse i -> j to j -> i
                            trial2 = current.copy()
                            trial2[i, j] = 0
                            trial2[j, i] = 1
                            if not _has_cycle(trial2):
                                parents_i = np.where(trial2[:, i])[0]
                                if self.max_parents < 0 or len(parents_i) <= self.max_parents:
                                    candidates.append(trial2)

                        for cand in candidates:
                            s = self.score_fn.score(cand, data)
                            if s > current_score:
                                current = cand
                                current_score = s
                                improved = True

                if not improved:
                    break

            if current_score > best_score:
                best_dag = current
                best_score = current_score

        return best_dag, best_score

    def run(
        self,
        data: DataMatrix,
        n_bootstrap: int = 100,
        confidence_level: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> BootstrapResult:
        """Run bootstrap structure learning.

        Parameters
        ----------
        data : DataMatrix
            N × p data matrix.
        n_bootstrap : int
            Number of bootstrap replicates.
        confidence_level : float
            Threshold for declaring an edge "stable" (e.g., 0.5 = majority).
        rng : optional
            NumPy random generator.

        Returns
        -------
        BootstrapResult
        """
        rng = rng or np.random.default_rng()
        n_obs, p = data.shape

        dags: List[AdjacencyMatrix] = []
        scores: List[float] = []

        for b in range(n_bootstrap):
            # Resample rows with replacement
            indices = rng.choice(n_obs, size=n_obs, replace=True)
            boot_data = data[indices]

            dag, score = self._greedy_search(boot_data, rng)
            dags.append(dag)
            scores.append(score)

        # Edge inclusion probabilities
        edge_probs = np.zeros((p, p), dtype=np.float64)
        for dag in dags:
            edge_probs += dag.astype(np.float64)
        edge_probs /= max(n_bootstrap, 1)

        confidence_edges = edge_probs >= confidence_level

        return BootstrapResult(
            dags=dags,
            scores=scores,
            edge_inclusion_probs=edge_probs,
            confidence_edges=confidence_edges,
            confidence_level=confidence_level,
        )

    def stability_selection(
        self,
        data: DataMatrix,
        n_bootstrap: int = 100,
        threshold: float = 0.6,
        subsample_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.bool_]:
        """Stability selection (Meinshausen & Bühlmann 2010) for edge detection.

        Uses sub-sampling instead of bootstrap for more conservative
        selection guarantees.

        Parameters
        ----------
        data : DataMatrix
            N × p data matrix.
        n_bootstrap : int
            Number of sub-sample replicates.
        threshold : float
            Selection probability threshold in (0.5, 1].
        subsample_fraction : float
            Fraction of data to use in each sub-sample.
        rng : optional
            NumPy random generator.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean matrix of stably selected edges.
        """
        rng = rng or np.random.default_rng()
        n_obs, p = data.shape
        sub_size = max(1, int(n_obs * subsample_fraction))

        counts = np.zeros((p, p), dtype=np.float64)

        for _ in range(n_bootstrap):
            indices = rng.choice(n_obs, size=sub_size, replace=False)
            sub_data = data[indices]
            dag, _ = self._greedy_search(sub_data, rng)
            counts += dag.astype(np.float64)

        probs = counts / max(n_bootstrap, 1)
        return probs >= threshold
