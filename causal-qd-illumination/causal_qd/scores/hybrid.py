"""Hybrid and composite scoring functions for causal structure learning.

Provides scoring classes that combine multiple base scores, add
structural penalties, compute robust scores under data perturbation,
and incorporate interventional data.
"""

from __future__ import annotations

import math
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.scores.score_base import DecomposableScore, ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_edges(adj: AdjacencyMatrix) -> int:
    """Count directed edges."""
    return int(np.sum(adj))


def _max_in_degree(adj: AdjacencyMatrix) -> int:
    """Maximum in-degree."""
    return int(adj.sum(axis=0).max()) if adj.shape[0] > 0 else 0


def _longest_path_length(adj: AdjacencyMatrix) -> int:
    """Longest directed path via DP on topological order."""
    from collections import deque

    n = adj.shape[0]
    if n <= 1:
        return 0
    in_deg = adj.sum(axis=0).copy()
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    if len(order) < n:
        order.extend(i for i in range(n) if i not in set(order))
    dist = np.zeros(n, dtype=int)
    for v in order:
        parents = np.where(adj[:, v])[0]
        if len(parents) > 0:
            dist[v] = int(dist[parents].max()) + 1
    return int(dist.max())


# ---------------------------------------------------------------------------
# HybridScore
# ---------------------------------------------------------------------------


class HybridScore(ScoreFunction):
    """Combine multiple scoring functions via weighted sum.

    Computes a weighted average of several base scores.  Supports
    both simple weighted sum and Pareto-based multi-objective scoring.

    Parameters
    ----------
    score_fns : Sequence[ScoreFunction]
        Base scoring functions.
    weights : Sequence[float] | None
        Per-score weights.  Normalized internally.
        If ``None``, equal weights are used.
    mode : str
        Combination mode: ``"weighted_sum"`` or ``"pareto_rank"``.
        Default ``"weighted_sum"``.
    """

    def __init__(
        self,
        score_fns: Sequence[ScoreFunction],
        weights: Optional[Sequence[float]] = None,
        mode: str = "weighted_sum",
    ) -> None:
        self._fns = list(score_fns)
        if weights is None:
            self._weights = np.ones(len(score_fns)) / len(score_fns)
        else:
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()
        self._mode = mode

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute the combined score.

        Parameters
        ----------
        dag : AdjacencyMatrix
            DAG adjacency matrix.
        data : DataMatrix
            N × p data matrix.

        Returns
        -------
        QualityScore
            Combined score value.
        """
        scores = np.array(
            [fn.score(dag, data) for fn in self._fns], dtype=np.float64
        )

        if self._mode == "pareto_rank":
            return self._pareto_score(scores)
        else:
            return float(np.dot(self._weights, scores))

    def component_scores(
        self, dag: AdjacencyMatrix, data: DataMatrix
    ) -> Dict[int, float]:
        """Return individual component scores.

        Parameters
        ----------
        dag, data

        Returns
        -------
        Dict[int, float]
            Score per component index.
        """
        return {
            i: fn.score(dag, data)
            for i, fn in enumerate(self._fns)
        }

    def _pareto_score(self, scores: npt.NDArray[np.float64]) -> float:
        """Convert multi-objective scores to a single value using rank.

        Uses the sum of normalized scores as a proxy for Pareto ranking
        on a single solution.  For full Pareto comparison, use
        MultiObjectiveSelection.

        Parameters
        ----------
        scores : ndarray
            Component scores.

        Returns
        -------
        float
        """
        # Normalize each score to [0, 1] range using sigmoid
        normalized = 1.0 / (1.0 + np.exp(-scores))
        return float(np.sum(self._weights * normalized))


# ---------------------------------------------------------------------------
# PenalizedScore
# ---------------------------------------------------------------------------


class PenalizedScore(ScoreFunction):
    """Add structural penalties to a base score.

    Supports L0 (edge count), L1 (adjacency sum), degree penalties,
    and path length penalties.

    Parameters
    ----------
    base_score : ScoreFunction
        Base scoring function.
    sparsity_penalty : float
        Weight for sparsity (edge count) penalty.  Default ``0.0``.
    sparsity_norm : str
        ``"l0"`` (number of edges) or ``"l1"`` (sum of adjacency).
        Default ``"l0"``.
    degree_penalty : float
        Penalty per unit of max in-degree exceeding ``max_degree``.
        Default ``0.0``.
    max_degree : int
        Soft threshold for degree penalty.  Default ``5``.
    path_penalty : float
        Penalty per unit of longest path length.  Default ``0.0``.
    """

    def __init__(
        self,
        base_score: ScoreFunction,
        sparsity_penalty: float = 0.0,
        sparsity_norm: str = "l0",
        degree_penalty: float = 0.0,
        max_degree: int = 5,
        path_penalty: float = 0.0,
    ) -> None:
        self._base = base_score
        self._sp_pen = sparsity_penalty
        self._sp_norm = sparsity_norm
        self._deg_pen = degree_penalty
        self._max_deg = max_degree
        self._path_pen = path_penalty

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute penalized score = base_score - penalties.

        Parameters
        ----------
        dag, data

        Returns
        -------
        QualityScore
        """
        base = self._base.score(dag, data)
        penalty = 0.0

        # Sparsity penalty
        if self._sp_pen > 0:
            if self._sp_norm == "l0":
                penalty += self._sp_pen * _count_edges(dag)
            else:  # l1
                penalty += self._sp_pen * float(np.sum(np.abs(dag)))

        # Degree penalty
        if self._deg_pen > 0:
            max_deg = _max_in_degree(dag)
            excess = max(0, max_deg - self._max_deg)
            penalty += self._deg_pen * excess

        # Path length penalty
        if self._path_pen > 0:
            path_len = _longest_path_length(dag)
            penalty += self._path_pen * path_len

        return base - penalty


# ---------------------------------------------------------------------------
# RobustScore
# ---------------------------------------------------------------------------


class RobustScore(ScoreFunction):
    """Score robustly via bootstrap aggregation.

    Evaluates the score over multiple bootstrap resamples and returns
    a robust statistic (mean, worst-case, or quantile).

    Parameters
    ----------
    base_score : ScoreFunction
        Base scoring function.
    n_bootstrap : int
        Number of bootstrap samples.  Default ``10``.
    aggregation : str
        How to aggregate: ``"mean"``, ``"min"`` (worst-case),
        ``"median"``, or ``"quantile"``.  Default ``"mean"``.
    quantile : float
        Quantile level (used when aggregation is ``"quantile"``).
        Default ``0.1`` (10th percentile).
    seed : int | None
        Random seed for reproducible bootstrap.  Default ``None``.
    """

    def __init__(
        self,
        base_score: ScoreFunction,
        n_bootstrap: int = 10,
        aggregation: str = "mean",
        quantile: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self._base = base_score
        self._n_boot = n_bootstrap
        self._agg = aggregation
        self._quantile = quantile
        self._seed = seed

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute the robust score over bootstrap samples.

        Parameters
        ----------
        dag, data

        Returns
        -------
        QualityScore
        """
        rng = np.random.default_rng(self._seed)
        n_samples = data.shape[0]
        scores: List[float] = []

        for _ in range(self._n_boot):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_data = data[indices]
            s = self._base.score(dag, boot_data)
            scores.append(s)

        scores_arr = np.array(scores)

        if self._agg == "min":
            return float(scores_arr.min())
        elif self._agg == "median":
            return float(np.median(scores_arr))
        elif self._agg == "quantile":
            return float(np.quantile(scores_arr, self._quantile))
        else:  # mean
            return float(scores_arr.mean())

    def bootstrap_scores(
        self, dag: AdjacencyMatrix, data: DataMatrix
    ) -> npt.NDArray[np.float64]:
        """Return all bootstrap scores for analysis.

        Parameters
        ----------
        dag, data

        Returns
        -------
        ndarray, shape (n_bootstrap,)
        """
        rng = np.random.default_rng(self._seed)
        n_samples = data.shape[0]
        scores = np.zeros(self._n_boot, dtype=np.float64)

        for i in range(self._n_boot):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_data = data[indices]
            scores[i] = self._base.score(dag, boot_data)

        return scores


# ---------------------------------------------------------------------------
# InterventionalScore
# ---------------------------------------------------------------------------


class InterventionalScore(ScoreFunction):
    """Score using both observational and interventional data.

    Combines an observational score (e.g., BIC) with an interventional
    likelihood term that measures how well the DAG explains
    interventional data.

    Parameters
    ----------
    observational_score : ScoreFunction
        Score computed on observational data.
    obs_weight : float
        Weight for observational component.  Default ``0.7``.
    int_weight : float
        Weight for interventional component.  Default ``0.3``.
    intervention_targets : List[int] | None
        Which variables have interventional data.
        If ``None``, all variables are assumed.
    """

    def __init__(
        self,
        observational_score: ScoreFunction,
        obs_weight: float = 0.7,
        int_weight: float = 0.3,
        intervention_targets: Optional[List[int]] = None,
    ) -> None:
        self._obs_score = observational_score
        self._obs_w = obs_weight
        self._int_w = int_weight
        self._targets = intervention_targets
        self._int_data: Dict[int, DataMatrix] = {}

    def set_interventional_data(
        self, target: int, data: DataMatrix
    ) -> None:
        """Register interventional data for a given target variable.

        Parameters
        ----------
        target : int
            Index of the intervened variable.
        data : DataMatrix
            Data collected under do(target = value).
        """
        self._int_data[target] = data

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute the combined observational + interventional score.

        Parameters
        ----------
        dag : AdjacencyMatrix
            DAG adjacency matrix.
        data : DataMatrix
            Observational data.

        Returns
        -------
        QualityScore
        """
        obs = self._obs_score.score(dag, data)

        if not self._int_data:
            return obs

        int_score = 0.0
        n_int = 0

        targets = (
            self._targets if self._targets is not None
            else list(self._int_data.keys())
        )

        for target in targets:
            if target not in self._int_data:
                continue

            int_data = self._int_data[target]
            score_val = self._interventional_likelihood(
                dag, target, int_data, data
            )
            int_score += score_val
            n_int += 1

        if n_int > 0:
            int_score /= n_int

        return self._obs_w * obs + self._int_w * int_score

    def _interventional_likelihood(
        self,
        dag: AdjacencyMatrix,
        target: int,
        int_data: DataMatrix,
        obs_data: DataMatrix,
    ) -> float:
        """Compute interventional log-likelihood for a single target.

        Under do(target), the target node is disconnected from its
        parents.  Other nodes follow their usual conditional
        distributions.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current DAG.
        target : int
            Intervened variable.
        int_data : DataMatrix
            Interventional data.
        obs_data : DataMatrix
            Observational data (for estimating parameters).

        Returns
        -------
        float
            Interventional log-likelihood.
        """
        n = dag.shape[0]
        m = int_data.shape[0]
        total_ll = 0.0

        for node in range(n):
            if node == target:
                # Under intervention, target follows the interventional
                # distribution (its parents are cut off).
                y = int_data[:, node]
                var = float(np.var(y)) + 1e-10
                ll = -0.5 * m * math.log(2 * math.pi * var) - 0.5 * m
            else:
                parents = list(np.where(dag[:, node])[0])
                if parents:
                    # Estimate regression from observational data
                    X_obs = np.column_stack(
                        [np.ones(obs_data.shape[0]), obs_data[:, parents]]
                    )
                    y_obs = obs_data[:, node]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(
                            X_obs, y_obs, rcond=None
                        )
                    except np.linalg.LinAlgError:
                        coeffs = np.zeros(len(parents) + 1)
                        coeffs[0] = np.mean(y_obs)

                    # Evaluate on interventional data
                    X_int = np.column_stack(
                        [np.ones(m), int_data[:, parents]]
                    )
                    predictions = X_int @ coeffs
                    residuals = int_data[:, node] - predictions
                    var = float(np.mean(residuals**2)) + 1e-10
                    ll = -0.5 * m * math.log(2 * math.pi * var) - 0.5 * m
                else:
                    y = int_data[:, node]
                    var = float(np.var(y)) + 1e-10
                    ll = -0.5 * m * math.log(2 * math.pi * var) - 0.5 * m

            total_ll += ll

        # BIC-style penalty
        k = _count_edges(dag) + n  # edges + intercepts
        penalty = -0.5 * k * math.log(m) if m > 1 else 0.0

        return total_ll + penalty


# ---------------------------------------------------------------------------
# DecomposableHybridScore
# ---------------------------------------------------------------------------


class DecomposableHybridScore(DecomposableScore):
    """Hybrid score that preserves decomposability.

    Combines multiple decomposable scores by summing their local
    scores with weights, maintaining the decomposability property
    for efficient incremental updates.

    Parameters
    ----------
    score_fns : Sequence[DecomposableScore]
        Decomposable base scores.
    weights : Sequence[float] | None
        Per-score weights.
    """

    def __init__(
        self,
        score_fns: Sequence[DecomposableScore],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self._fns = list(score_fns)
        if weights is None:
            self._weights = np.ones(len(score_fns)) / len(score_fns)
        else:
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()

    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute weighted sum of local scores.

        Parameters
        ----------
        node, parents, data

        Returns
        -------
        float
        """
        total = 0.0
        for fn, w in zip(self._fns, self._weights):
            total += w * fn.local_score(node, parents, data)
        return total
