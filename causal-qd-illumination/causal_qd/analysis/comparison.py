"""Algorithm comparison and benchmarking for causal discovery.

Provides tools for running multiple algorithms on the same data,
computing comparison metrics, performing statistical significance
tests, and managing standard benchmark datasets.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing one algorithm on one dataset."""
    algorithm_name: str
    dataset_name: str
    predicted_dag: AdjacencyMatrix
    true_dag: Optional[AdjacencyMatrix]
    shd: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    elapsed_time: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StatisticalTestResult:
    """Result of a paired statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def structural_hamming_distance(
    pred: AdjacencyMatrix, true: AdjacencyMatrix
) -> int:
    """Compute the Structural Hamming Distance (SHD).

    SHD counts the number of edge additions, deletions, and reversals
    needed to transform *pred* into *true*.

    Parameters
    ----------
    pred, true : AdjacencyMatrix
        Predicted and true DAG adjacency matrices.

    Returns
    -------
    int
        SHD value (lower is better).
    """
    pred = np.asarray(pred, dtype=np.int8)
    true = np.asarray(true, dtype=np.int8)
    n = pred.shape[0]

    additions = 0  # in true but not in pred
    deletions = 0  # in pred but not in true
    reversals = 0  # edge exists in both but reversed

    for i in range(n):
        for j in range(i + 1, n):
            pred_ij = bool(pred[i, j])
            pred_ji = bool(pred[j, i])
            true_ij = bool(true[i, j])
            true_ji = bool(true[j, i])

            if pred_ij and not true_ij and not true_ji:
                deletions += 1
            elif pred_ji and not true_ij and not true_ji:
                deletions += 1
            elif true_ij and not pred_ij and not pred_ji:
                additions += 1
            elif true_ji and not pred_ij and not pred_ji:
                additions += 1
            elif pred_ij and true_ji:
                reversals += 1
            elif pred_ji and true_ij:
                reversals += 1

    return additions + deletions + reversals


def edge_precision_recall_f1(
    pred: AdjacencyMatrix, true: AdjacencyMatrix
) -> Tuple[float, float, float]:
    """Compute edge-level precision, recall, and F1 score.

    Treats each directed edge as a binary prediction.

    Parameters
    ----------
    pred, true : AdjacencyMatrix

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, F1).
    """
    pred = np.asarray(pred, dtype=np.int8)
    true = np.asarray(true, dtype=np.int8)

    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def skeleton_f1(
    pred: AdjacencyMatrix, true: AdjacencyMatrix
) -> float:
    """Compute F1 score on the undirected skeleton.

    Ignores edge directions and evaluates skeleton recovery.

    Parameters
    ----------
    pred, true : AdjacencyMatrix

    Returns
    -------
    float
        Skeleton F1 score.
    """
    pred_skel = (np.asarray(pred) | np.asarray(pred).T).astype(bool)
    true_skel = (np.asarray(true) | np.asarray(true).T).astype(bool)

    # Only upper triangle to avoid double-counting
    n = pred.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    tp = int(np.sum(pred_skel[mask] & true_skel[mask]))
    fp = int(np.sum(pred_skel[mask] & ~true_skel[mask]))
    fn = int(np.sum(~pred_skel[mask] & true_skel[mask]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


# ---------------------------------------------------------------------------
# AlgorithmComparator
# ---------------------------------------------------------------------------


class AlgorithmComparator:
    """Run multiple algorithms on the same data and compare results.

    Parameters
    ----------
    algorithms : Dict[str, Callable]
        Named algorithms.  Each callable takes ``(data, **kwargs)``
        and returns an ``AdjacencyMatrix``.
    """

    def __init__(
        self,
        algorithms: Dict[str, Callable[..., AdjacencyMatrix]],
    ) -> None:
        self._algorithms = dict(algorithms)

    def run_comparison(
        self,
        data: DataMatrix,
        true_dag: Optional[AdjacencyMatrix] = None,
        dataset_name: str = "default",
        n_repetitions: int = 1,
        **kwargs: Any,
    ) -> List[ComparisonResult]:
        """Run all algorithms and collect comparison metrics.

        Parameters
        ----------
        data : DataMatrix
            Observed data.
        true_dag : AdjacencyMatrix | None
            Ground truth DAG (if available).
        dataset_name : str
            Name of the dataset.
        n_repetitions : int
            Number of runs per algorithm.  Default ``1``.
        **kwargs
            Extra keyword arguments passed to each algorithm.

        Returns
        -------
        List[ComparisonResult]
            One result per (algorithm, repetition) pair.
        """
        results: List[ComparisonResult] = []

        for alg_name, alg_fn in self._algorithms.items():
            for rep in range(n_repetitions):
                start = time.perf_counter()
                try:
                    predicted = alg_fn(data, **kwargs)
                except Exception as e:
                    n = data.shape[1]
                    predicted = np.zeros((n, n), dtype=np.int8)

                elapsed = time.perf_counter() - start

                result = ComparisonResult(
                    algorithm_name=alg_name,
                    dataset_name=dataset_name,
                    predicted_dag=predicted,
                    true_dag=true_dag,
                    elapsed_time=elapsed,
                )

                if true_dag is not None:
                    result.shd = structural_hamming_distance(predicted, true_dag)
                    p, r, f = edge_precision_recall_f1(predicted, true_dag)
                    result.precision = p
                    result.recall = r
                    result.f1 = f
                    result.extra_metrics["skeleton_f1"] = skeleton_f1(
                        predicted, true_dag
                    )

                results.append(result)

        return results

    def paired_test(
        self,
        results1: List[ComparisonResult],
        results2: List[ComparisonResult],
        metric: str = "shd",
    ) -> StatisticalTestResult:
        """Perform a paired statistical test between two algorithms.

        Uses paired t-test if n ≥ 30, otherwise Wilcoxon signed-rank.

        Parameters
        ----------
        results1, results2 : List[ComparisonResult]
            Paired results from two algorithms.
        metric : str
            Which metric to compare.

        Returns
        -------
        StatisticalTestResult
        """
        vals1 = self._extract_metric(results1, metric)
        vals2 = self._extract_metric(results2, metric)

        n = min(len(vals1), len(vals2))
        if n < 2:
            return StatisticalTestResult(
                test_name="insufficient_data",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )

        v1 = np.array(vals1[:n])
        v2 = np.array(vals2[:n])
        diffs = v1 - v2

        if n >= 30:
            return self._paired_t_test(diffs)
        else:
            return self._wilcoxon_test(diffs)

    @staticmethod
    def _extract_metric(
        results: List[ComparisonResult], metric: str
    ) -> List[float]:
        """Extract a metric from results."""
        values: List[float] = []
        for r in results:
            if metric == "shd":
                values.append(float(r.shd))
            elif metric == "f1":
                values.append(r.f1)
            elif metric == "precision":
                values.append(r.precision)
            elif metric == "recall":
                values.append(r.recall)
            elif metric == "time":
                values.append(r.elapsed_time)
            elif metric in r.extra_metrics:
                values.append(r.extra_metrics[metric])
        return values

    @staticmethod
    def _paired_t_test(
        diffs: npt.NDArray[np.float64],
    ) -> StatisticalTestResult:
        """Paired t-test on differences."""
        n = len(diffs)
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1))
        se = std_diff / math.sqrt(n) if n > 0 else 1.0

        t_stat = mean_diff / se if se > 1e-15 else 0.0

        # Two-tailed p-value approximation using normal distribution
        # (appropriate for large n)
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

        # Cohen's d effect size
        pooled_std = std_diff if std_diff > 1e-15 else 1.0
        effect_size = mean_diff / pooled_std

        return StatisticalTestResult(
            test_name="paired_t_test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < 0.05,
            effect_size=abs(effect_size),
            details={"mean_diff": mean_diff, "std_diff": std_diff, "n": n},
        )

    @staticmethod
    def _wilcoxon_test(
        diffs: npt.NDArray[np.float64],
    ) -> StatisticalTestResult:
        """Wilcoxon signed-rank test on differences.

        Manual implementation (no scipy dependency).
        """
        # Remove zeros
        nonzero_mask = np.abs(diffs) > 1e-15
        nonzero_diffs = diffs[nonzero_mask]
        n = len(nonzero_diffs)

        if n == 0:
            return StatisticalTestResult(
                test_name="wilcoxon",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )

        # Rank absolute differences
        abs_diffs = np.abs(nonzero_diffs)
        ranks = np.argsort(np.argsort(abs_diffs)) + 1.0

        # Compute W+ (sum of ranks for positive differences)
        w_plus = float(np.sum(ranks[nonzero_diffs > 0]))
        w_minus = float(np.sum(ranks[nonzero_diffs < 0]))
        w_stat = min(w_plus, w_minus)

        # Normal approximation for p-value
        mean_w = n * (n + 1) / 4.0
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

        if std_w > 1e-15:
            z = (w_stat - mean_w) / std_w
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        else:
            p_value = 1.0

        # Effect size: r = Z / sqrt(N)
        effect_size = abs(z) / math.sqrt(n) if std_w > 1e-15 else 0.0

        return StatisticalTestResult(
            test_name="wilcoxon",
            statistic=w_stat,
            p_value=p_value,
            significant=p_value < 0.05,
            effect_size=effect_size,
            details={"w_plus": w_plus, "w_minus": w_minus, "n": n},
        )

    def summary_table(
        self,
        results: List[ComparisonResult],
    ) -> Dict[str, Dict[str, float]]:
        """Generate a summary table of results per algorithm.

        Parameters
        ----------
        results : List[ComparisonResult]

        Returns
        -------
        Dict[str, Dict[str, float]]
            algorithm_name → {metric: mean_value}
        """
        by_alg: Dict[str, List[ComparisonResult]] = {}
        for r in results:
            if r.algorithm_name not in by_alg:
                by_alg[r.algorithm_name] = []
            by_alg[r.algorithm_name].append(r)

        table: Dict[str, Dict[str, float]] = {}
        for alg_name, alg_results in by_alg.items():
            shds = [r.shd for r in alg_results]
            f1s = [r.f1 for r in alg_results]
            times = [r.elapsed_time for r in alg_results]
            table[alg_name] = {
                "mean_shd": float(np.mean(shds)),
                "std_shd": float(np.std(shds)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "mean_time": float(np.mean(times)),
            }

        return table


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Standard causal discovery benchmarks.

    Generates data from well-known causal network structures for
    reproducible benchmarking.
    """

    @staticmethod
    def asia_network() -> Tuple[AdjacencyMatrix, int]:
        """Return the Asia (Lauritzen-Spiegelhalter) network.

        8 nodes: Asia, Smoking, Tuberculosis, LungCancer,
        Bronchitis, Either, XRay, Dyspnea.

        Returns
        -------
        Tuple[AdjacencyMatrix, int]
            (adjacency_matrix, n_nodes)
        """
        n = 8
        adj = np.zeros((n, n), dtype=np.int8)
        # Asia -> Tuberculosis
        adj[0, 2] = 1
        # Smoking -> LungCancer
        adj[1, 3] = 1
        # Smoking -> Bronchitis
        adj[1, 4] = 1
        # Tuberculosis -> Either
        adj[2, 5] = 1
        # LungCancer -> Either
        adj[3, 5] = 1
        # Either -> XRay
        adj[5, 6] = 1
        # Either -> Dyspnea
        adj[5, 7] = 1
        # Bronchitis -> Dyspnea
        adj[4, 7] = 1
        return adj, n

    @staticmethod
    def sachs_network() -> Tuple[AdjacencyMatrix, int]:
        """Return a simplified Sachs protein signaling network.

        11 nodes representing protein kinases and phospholipids.

        Returns
        -------
        Tuple[AdjacencyMatrix, int]
        """
        n = 11
        adj = np.zeros((n, n), dtype=np.int8)
        # Simplified version of the Sachs network
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 5),
            (3, 4), (3, 5), (4, 5), (5, 6),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (2, 10), (1, 4),
        ]
        for i, j in edges:
            adj[i, j] = 1
        return adj, n

    @staticmethod
    def chain_network(n: int = 5) -> Tuple[AdjacencyMatrix, int]:
        """Generate a simple chain DAG: 0 → 1 → 2 → ... → n-1.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        Tuple[AdjacencyMatrix, int]
        """
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n - 1):
            adj[i, i + 1] = 1
        return adj, n

    @staticmethod
    def fork_network(n_children: int = 3) -> Tuple[AdjacencyMatrix, int]:
        """Generate a fork DAG: root → child1, root → child2, etc.

        Parameters
        ----------
        n_children : int
            Number of children.

        Returns
        -------
        Tuple[AdjacencyMatrix, int]
        """
        n = n_children + 1
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(1, n):
            adj[0, i] = 1
        return adj, n

    @staticmethod
    def collider_network(n_parents: int = 3) -> Tuple[AdjacencyMatrix, int]:
        """Generate a collider DAG: parent1, parent2, ... → child.

        Parameters
        ----------
        n_parents : int
            Number of parents.

        Returns
        -------
        Tuple[AdjacencyMatrix, int]
        """
        n = n_parents + 1
        adj = np.zeros((n, n), dtype=np.int8)
        child = n_parents
        for i in range(n_parents):
            adj[i, child] = 1
        return adj, n

    @staticmethod
    def generate_linear_gaussian_data(
        dag: AdjacencyMatrix,
        n_samples: int = 500,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ) -> DataMatrix:
        """Generate data from a linear Gaussian structural equation model.

        For each node j in topological order:
            X_j = Σ_{i ∈ Pa(j)} β_{ij} X_i + ε_j

        where β_{ij} ~ Uniform(0.5, 2.0) with random sign and
        ε_j ~ N(0, noise_std²).

        Parameters
        ----------
        dag : AdjacencyMatrix
            DAG structure.
        n_samples : int
            Number of observations.
        noise_std : float
            Standard deviation of noise terms.
        seed : int | None
            Random seed.

        Returns
        -------
        DataMatrix
            (n_samples, n_nodes) data matrix.
        """
        rng = np.random.default_rng(seed)
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]

        # Generate edge weights
        weights = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    w = rng.uniform(0.5, 2.0)
                    if rng.random() < 0.5:
                        w = -w
                    weights[i, j] = w

        # Topological sort
        from collections import deque
        in_deg = adj.sum(axis=0).copy()
        queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
        order: List[int] = []
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

        # Generate data
        data = np.zeros((n_samples, n), dtype=np.float64)
        for node in order:
            noise = rng.normal(0, noise_std, n_samples)
            parents = np.where(adj[:, node])[0]
            if len(parents) > 0:
                data[:, node] = data[:, parents] @ weights[parents, node] + noise
            else:
                data[:, node] = noise

        return data

    def run_benchmark(
        self,
        algorithms: Dict[str, Callable[..., AdjacencyMatrix]],
        n_samples: int = 500,
        n_repetitions: int = 5,
        seed: int = 42,
    ) -> List[ComparisonResult]:
        """Run all algorithms on all benchmark networks.

        Parameters
        ----------
        algorithms : Dict[str, Callable]
            Named algorithms.
        n_samples : int
            Samples per dataset.
        n_repetitions : int
            Repetitions per (algorithm, dataset) pair.
        seed : int
            Base random seed.

        Returns
        -------
        List[ComparisonResult]
        """
        benchmarks = {
            "asia": self.asia_network,
            "chain_5": lambda: self.chain_network(5),
            "fork_3": lambda: self.fork_network(3),
            "collider_3": lambda: self.collider_network(3),
        }

        comparator = AlgorithmComparator(algorithms)
        all_results: List[ComparisonResult] = []

        for bm_name, bm_fn in benchmarks.items():
            dag, n_nodes = bm_fn()
            for rep in range(n_repetitions):
                data = self.generate_linear_gaussian_data(
                    dag, n_samples=n_samples, seed=seed + rep
                )
                results = comparator.run_comparison(
                    data, true_dag=dag, dataset_name=f"{bm_name}_rep{rep}"
                )
                all_results.extend(results)

        return all_results
