"""DAG bootstrap for uncertainty quantification.

Repeatedly resamples the data, runs a structure learner on each
resample, and aggregates the resulting DAGs into edge-confidence
estimates.

The non-parametric bootstrap provides a model-free approach to
uncertainty quantification in causal discovery.  By learning a DAG
from each bootstrap replicate and computing edge frequencies, we
obtain an empirical estimate of the posterior probability that each
edge is present.  Edges that appear consistently across replicates
are deemed stable, while those that appear infrequently indicate
structural uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from typing import Any, Dict, List, Optional, Tuple


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Aggregated result of a DAG bootstrap procedure.

    Attributes
    ----------
    edge_probabilities : NDArray
        Matrix of shape ``(p, p)`` with bootstrap edge frequencies.
    dag_samples : List[NDArray]
        Adjacency matrices from individual bootstrap replicates.
    confidence_dag : NDArray
        Binary DAG obtained by thresholding *edge_probabilities*.
    confidence_intervals : Optional[NDArray]
        Array of shape ``(p, p, 2)`` with lower and upper bounds
        of confidence intervals for edge probabilities.
    n_samples : int
        Number of bootstrap replicates used.
    """

    edge_probabilities: NDArray = field(default_factory=lambda: np.empty(0))
    dag_samples: List[NDArray] = field(default_factory=list)
    confidence_dag: NDArray = field(default_factory=lambda: np.empty(0))
    confidence_intervals: Optional[NDArray] = None
    n_samples: int = 0


# -------------------------------------------------------------------
# DAGBootstrap
# -------------------------------------------------------------------

class DAGBootstrap:
    """Non-parametric DAG bootstrap.

    Resamples the data with replacement, runs a structure learner on
    each resample, and aggregates the resulting DAGs into edge
    confidence estimates.

    Parameters
    ----------
    structure_learner : Any
        A structure learning object exposing a ``fit(data)`` method
        that returns an adjacency matrix.
    n_bootstrap : int
        Number of bootstrap replicates.
    confidence_level : float
        Confidence level for interval estimation (default 0.95).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        structure_learner: Any,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.structure_learner = structure_learner
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self._rng = np.random.default_rng(seed)
        self._result: Optional[BootstrapResult] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self, data: NDArray, threshold: float = 0.5) -> BootstrapResult:
        """Execute the bootstrap and return aggregated results.

        Parameters
        ----------
        data : NDArray
            Observation matrix of shape ``(n_samples, n_variables)``.
        threshold : float
            Edge inclusion threshold for the confidence DAG.

        Returns
        -------
        BootstrapResult
            Aggregated bootstrap results.
        """
        data = np.asarray(data, dtype=np.float64)
        n_obs, n_vars = data.shape

        dag_samples: List[NDArray] = []

        for b in range(self.n_bootstrap):
            bootstrap_data = self.resample_data(data)
            dag = self._learn_dag(bootstrap_data)
            if dag is not None:
                dag_samples.append(dag)

        if not dag_samples:
            empty = np.zeros((n_vars, n_vars), dtype=np.float64)
            self._result = BootstrapResult(
                edge_probabilities=empty,
                dag_samples=[],
                confidence_dag=empty,
                n_samples=0,
            )
            return self._result

        edge_probs = self.aggregate_dags(dag_samples)
        confidence_dag = (edge_probs >= threshold).astype(np.float64)
        ci = self._compute_confidence_intervals(dag_samples, n_vars)

        self._result = BootstrapResult(
            edge_probabilities=edge_probs,
            dag_samples=dag_samples,
            confidence_dag=confidence_dag,
            confidence_intervals=ci,
            n_samples=len(dag_samples),
        )
        return self._result

    def resample_data(self, data: NDArray) -> NDArray:
        """Draw a bootstrap resample (with replacement) from *data*.

        Parameters
        ----------
        data : NDArray
            Original data matrix of shape ``(n_samples, n_variables)``.

        Returns
        -------
        NDArray
            Resampled data matrix of the same shape.
        """
        n = data.shape[0]
        indices = self._rng.integers(0, n, size=n)
        return data[indices]

    def aggregate_dags(self, dag_samples: List[NDArray]) -> NDArray:
        """Compute edge-frequency matrix from a list of DAGs.

        Parameters
        ----------
        dag_samples : List[NDArray]
            List of adjacency matrices.

        Returns
        -------
        NDArray
            Matrix of shape ``(p, p)`` with edge inclusion frequencies.
        """
        if not dag_samples:
            return np.empty((0, 0))
        n = dag_samples[0].shape[0]
        freq = np.zeros((n, n), dtype=np.float64)
        for dag in dag_samples:
            freq += (np.asarray(dag) != 0).astype(np.float64)
        freq /= len(dag_samples)
        return freq

    def confidence_edges(self, threshold: float = 0.5) -> NDArray:
        """Return a binary DAG with edges above *threshold*.

        Parameters
        ----------
        threshold : float
            Minimum edge probability to include.

        Returns
        -------
        NDArray
            Binary adjacency matrix.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before confidence_edges()")
        return (self._result.edge_probabilities >= threshold).astype(np.float64)

    def edge_confidence(self, i: int, j: int) -> float:
        """Return the bootstrap probability of edge i -> j.

        Parameters
        ----------
        i, j : int
            Source and target node indices.

        Returns
        -------
        float
            Estimated edge probability.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before edge_confidence()")
        return float(self._result.edge_probabilities[i, j])

    def edge_confidence_matrix(self) -> NDArray:
        """Return the full edge confidence matrix.

        Returns
        -------
        NDArray
            Matrix of edge inclusion probabilities.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before edge_confidence_matrix()")
        return self._result.edge_probabilities.copy()

    def threshold_graph(self, threshold: float = 0.5) -> NDArray:
        """Return graph with edges above *threshold*.

        Parameters
        ----------
        threshold : float
            Minimum edge probability.

        Returns
        -------
        NDArray
            Binary adjacency matrix.
        """
        return self.confidence_edges(threshold)

    def stable_edges(self, threshold: float = 0.9) -> List[Tuple[int, int]]:
        """Return highly stable edges (above *threshold*).

        Parameters
        ----------
        threshold : float
            Stability threshold (default 0.9).

        Returns
        -------
        List[Tuple[int, int]]
            List of ``(i, j)`` pairs for stable edges.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before stable_edges()")
        probs = self._result.edge_probabilities
        edges = []
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                if probs[i, j] >= threshold:
                    edges.append((i, j))
        return edges

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _learn_dag(self, bootstrap_data: NDArray) -> Optional[NDArray]:
        """Learn a DAG from bootstrap data using the structure learner.

        Parameters
        ----------
        bootstrap_data : NDArray
            Resampled data.

        Returns
        -------
        Optional[NDArray]
            Adjacency matrix, or None if learning failed.
        """
        try:
            result = self.structure_learner.fit(bootstrap_data)
            if isinstance(result, np.ndarray):
                return result
            # Try to extract adjacency matrix from result object
            if hasattr(result, "adjacency_matrix"):
                adj = result.adjacency_matrix
                if callable(adj):
                    return adj()
                return np.asarray(adj)
            if hasattr(result, "graph"):
                return np.asarray(result.graph)
            return None
        except Exception:
            return None

    def _compute_confidence_intervals(
        self,
        dag_samples: List[NDArray],
        n_vars: int,
    ) -> NDArray:
        """Compute confidence intervals for edge probabilities.

        Uses the Wilson score interval for binomial proportions,
        which is well-calibrated even for small sample sizes and
        extreme probabilities.

        Parameters
        ----------
        dag_samples : List[NDArray]
            Bootstrap DAG samples.
        n_vars : int
            Number of variables.

        Returns
        -------
        NDArray
            Array of shape ``(n_vars, n_vars, 2)`` with [lower, upper].
        """
        n = len(dag_samples)
        ci = np.zeros((n_vars, n_vars, 2), dtype=np.float64)

        if n == 0:
            return ci

        alpha = 1.0 - self.confidence_level
        z = sp_stats.norm.ppf(1 - alpha / 2)

        for i in range(n_vars):
            for j in range(n_vars):
                k = sum(
                    1 for dag in dag_samples if dag[i, j] != 0
                )
                p_hat = k / n
                # Wilson score interval
                denom = 1 + z ** 2 / n
                center = (p_hat + z ** 2 / (2 * n)) / denom
                margin = z * math.sqrt(
                    (p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n
                ) / denom
                ci[i, j, 0] = max(0.0, center - margin)
                ci[i, j, 1] = min(1.0, center + margin)

        return ci

    # -----------------------------------------------------------------
    # Advanced analysis
    # -----------------------------------------------------------------

    def structural_variability(self) -> float:
        """Compute structural variability across bootstrap DAGs.

        Returns the average pairwise Structural Hamming Distance
        (SHD) between bootstrap DAGs, normalized by the number
        of possible edges.

        Returns
        -------
        float
            Normalized structural variability in [0, 1].
        """
        if self._result is None or len(self._result.dag_samples) < 2:
            return 0.0

        dags = self._result.dag_samples
        n = dags[0].shape[0]
        n_possible = n * (n - 1)
        if n_possible == 0:
            return 0.0

        total_shd = 0.0
        count = 0
        # Sample pairs for efficiency if too many DAGs
        n_dags = len(dags)
        if n_dags > 50:
            pairs = min(500, n_dags * (n_dags - 1) // 2)
            for _ in range(pairs):
                i, j = self._rng.integers(0, n_dags, size=2)
                while i == j:
                    j = int(self._rng.integers(0, n_dags))
                total_shd += self._shd(dags[i], dags[j])
                count += 1
        else:
            for i in range(n_dags):
                for j in range(i + 1, n_dags):
                    total_shd += self._shd(dags[i], dags[j])
                    count += 1

        return (total_shd / count) / n_possible if count > 0 else 0.0

    @staticmethod
    def _shd(dag1: NDArray, dag2: NDArray) -> int:
        """Structural Hamming Distance between two DAGs.

        Parameters
        ----------
        dag1, dag2 : NDArray
            Adjacency matrices.

        Returns
        -------
        int
        """
        b1 = (dag1 != 0).astype(int)
        b2 = (dag2 != 0).astype(int)
        return int(np.sum(np.abs(b1 - b2)))

    def edge_stability_ranking(self) -> List[Tuple[int, int, float]]:
        """Rank edges by their bootstrap stability.

        Returns
        -------
        List[Tuple[int, int, float]]
            List of ``(i, j, probability)`` sorted by decreasing probability.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before edge_stability_ranking()")
        probs = self._result.edge_probabilities
        edges: List[Tuple[int, int, float]] = []
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                if probs[i, j] > 0:
                    edges.append((i, j, float(probs[i, j])))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges

    def consensus_dag(
        self,
        threshold: float = 0.5,
        enforce_acyclicity: bool = True,
    ) -> NDArray:
        """Build a consensus DAG from bootstrap results.

        Includes edges whose bootstrap probability exceeds *threshold*.
        If *enforce_acyclicity* is True, removes the weakest edges
        to break any cycles.

        Parameters
        ----------
        threshold : float
            Edge inclusion threshold.
        enforce_acyclicity : bool
            Whether to enforce the DAG constraint.

        Returns
        -------
        NDArray
            Consensus adjacency matrix.
        """
        if self._result is None:
            raise RuntimeError("Must call run() before consensus_dag()")

        probs = self._result.edge_probabilities
        n = probs.shape[0]
        dag = (probs >= threshold).astype(np.float64)

        if enforce_acyclicity:
            dag = self._break_cycles(dag, probs)

        return dag

    @staticmethod
    def _break_cycles(dag: NDArray, weights: NDArray) -> NDArray:
        """Remove weakest edges to make the graph acyclic.

        Uses iterative cycle detection and removal of the lowest-
        weight edge in each cycle.

        Parameters
        ----------
        dag : NDArray
            Binary adjacency matrix (may contain cycles).
        weights : NDArray
            Edge weight matrix (bootstrap probabilities).

        Returns
        -------
        NDArray
            Acyclic adjacency matrix.
        """
        from collections import deque

        result = dag.copy()
        n = result.shape[0]

        for _ in range(n * n):
            # Check acyclicity via Kahn's algorithm
            in_deg = np.sum(result != 0, axis=0).astype(int)
            queue = deque(i for i in range(n) if in_deg[i] == 0)
            visited = 0
            while queue:
                node = queue.popleft()
                visited += 1
                for child in range(n):
                    if result[node, child] != 0:
                        in_deg[child] -= 1
                        if in_deg[child] == 0:
                            queue.append(child)

            if visited == n:
                break  # It's a DAG

            # Find an edge in the cycle with lowest weight and remove it
            # Nodes with remaining in-degree > 0 are in cycles
            cycle_nodes = [i for i in range(n) if in_deg[i] > 0]
            min_weight = float("inf")
            min_edge = (0, 0)
            for i in cycle_nodes:
                for j in cycle_nodes:
                    if result[i, j] != 0 and weights[i, j] < min_weight:
                        min_weight = weights[i, j]
                        min_edge = (i, j)
            result[min_edge[0], min_edge[1]] = 0.0

        return result
