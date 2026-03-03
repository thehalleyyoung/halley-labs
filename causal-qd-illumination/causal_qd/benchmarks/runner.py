"""Benchmark runner for systematic evaluation and comparison.

Runs CausalQD and baseline algorithms on standard and synthetic benchmarks,
aggregates results, and generates comparison reports.

Classes
-------
* :class:`BenchmarkRunner` – run all benchmarks with a given algorithm
* :class:`ComparisonRunner` – compare CausalQD against baselines
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.benchmarks.standard_networks import (
    BenchmarkNetwork,
    BenchmarkResult,
    all_benchmarks,
)
from causal_qd.benchmarks.synthetic import (
    RandomDAGBenchmark,
    ScalabilityBenchmark,
    SparsityBenchmark,
    SyntheticBenchmark,
    SyntheticBenchmarkSpec,
)
from causal_qd.types import AdjacencyMatrix, DataMatrix

__all__ = ["BenchmarkRunner", "ComparisonRunner"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmRunResult:
    """Result of running an algorithm on a single benchmark."""

    benchmark_name: str
    algorithm_name: str
    n_nodes: int
    shd: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_correct_edges: int = 0
    n_extra_edges: int = 0
    n_missing_edges: int = 0
    n_reversed_edges: int = 0
    elapsed_seconds: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from a benchmark suite run."""

    algorithm_name: str
    results: List[AlgorithmRunResult] = field(default_factory=list)

    @property
    def mean_shd(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.shd for r in self.results]))

    @property
    def mean_f1(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.f1 for r in self.results]))

    @property
    def mean_precision(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.precision for r in self.results]))

    @property
    def mean_recall(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.recall for r in self.results]))

    @property
    def total_time(self) -> float:
        return float(sum(r.elapsed_seconds for r in self.results))

    def summary(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "n_benchmarks": len(self.results),
            "mean_shd": self.mean_shd,
            "mean_f1": self.mean_f1,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "total_time": self.total_time,
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Run a causal discovery algorithm on all benchmarks.

    Takes an algorithm function that accepts data and returns an estimated
    DAG, and evaluates it on standard and synthetic benchmarks.

    Parameters
    ----------
    algorithm_fn : callable
        ``(data: DataMatrix, n_nodes: int) -> AdjacencyMatrix``
    algorithm_name : str
        Name for reporting.
    n_samples : int
        Number of samples to generate per benchmark.
    seed : int
        Random seed for data generation.

    Examples
    --------
    >>> def my_algorithm(data, n_nodes):
    ...     return np.zeros((n_nodes, n_nodes), dtype=np.int8)
    >>> runner = BenchmarkRunner(my_algorithm, "empty_graph")
    >>> results = runner.run_standard()
    >>> for r in results.results:
    ...     print(f"{r.benchmark_name}: SHD={r.shd}")
    """

    def __init__(
        self,
        algorithm_fn: Callable[[DataMatrix, int], AdjacencyMatrix],
        algorithm_name: str = "algorithm",
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        self._algorithm_fn = algorithm_fn
        self._algorithm_name = algorithm_name
        self._n_samples = n_samples
        self._seed = seed

    def run_standard(
        self,
        benchmarks: Optional[List[BenchmarkNetwork]] = None,
    ) -> BenchmarkSuiteResult:
        """Run algorithm on standard network benchmarks.

        Parameters
        ----------
        benchmarks : list of BenchmarkNetwork, optional
            Benchmarks to run. Defaults to all standard benchmarks.

        Returns
        -------
        BenchmarkSuiteResult
            Aggregated results.
        """
        if benchmarks is None:
            benchmarks = all_benchmarks()

        suite = BenchmarkSuiteResult(algorithm_name=self._algorithm_name)
        rng = np.random.default_rng(self._seed)

        for bench in benchmarks:
            data = bench.generate_data(self._n_samples, rng=rng)

            t0 = time.perf_counter()
            try:
                estimated = self._algorithm_fn(data, bench.n_nodes)
            except Exception as e:
                estimated = np.zeros(
                    (bench.n_nodes, bench.n_nodes), dtype=np.int8
                )
            elapsed = time.perf_counter() - t0

            eval_result = bench.evaluate(estimated)

            suite.results.append(
                AlgorithmRunResult(
                    benchmark_name=bench.name,
                    algorithm_name=self._algorithm_name,
                    n_nodes=bench.n_nodes,
                    shd=eval_result.shd,
                    precision=eval_result.precision,
                    recall=eval_result.recall,
                    f1=eval_result.f1,
                    n_correct_edges=eval_result.n_correct_edges,
                    n_extra_edges=eval_result.n_extra_edges,
                    n_missing_edges=eval_result.n_missing_edges,
                    n_reversed_edges=eval_result.n_reversed_edges,
                    elapsed_seconds=elapsed,
                )
            )

        return suite

    def run_synthetic(
        self,
        specs: Optional[List[SyntheticBenchmarkSpec]] = None,
        n_nodes: int = 10,
        n_instances: int = 5,
    ) -> BenchmarkSuiteResult:
        """Run algorithm on synthetic benchmarks.

        Parameters
        ----------
        specs : list of SyntheticBenchmarkSpec, optional
            Pre-generated benchmark specs. If None, generates random DAGs.
        n_nodes : int
            Number of nodes (if generating).
        n_instances : int
            Number of instances (if generating).

        Returns
        -------
        BenchmarkSuiteResult
        """
        if specs is None:
            gen = RandomDAGBenchmark(
                n_nodes=n_nodes,
                n_instances=n_instances,
                n_samples=self._n_samples,
                seed=self._seed,
            )
            specs = gen.generate()

        suite = BenchmarkSuiteResult(algorithm_name=self._algorithm_name)

        for spec in specs:
            t0 = time.perf_counter()
            try:
                estimated = self._algorithm_fn(spec.data, spec.n_nodes)
            except Exception:
                estimated = np.zeros(
                    (spec.n_nodes, spec.n_nodes), dtype=np.int8
                )
            elapsed = time.perf_counter() - t0

            shd = SyntheticBenchmark.shd(spec.true_dag, estimated)

            # Compute precision/recall
            true_edges = set(
                zip(*np.nonzero(spec.true_dag))
            )
            est_edges = set(zip(*np.nonzero(estimated)))
            correct = len(true_edges & est_edges)
            precision = correct / len(est_edges) if est_edges else 0.0
            recall = correct / len(true_edges) if true_edges else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            suite.results.append(
                AlgorithmRunResult(
                    benchmark_name=spec.name,
                    algorithm_name=self._algorithm_name,
                    n_nodes=spec.n_nodes,
                    shd=shd,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    n_correct_edges=correct,
                    n_extra_edges=len(est_edges - true_edges),
                    n_missing_edges=len(true_edges - est_edges),
                    elapsed_seconds=elapsed,
                    extra_metrics=spec.params,
                )
            )

        return suite

    def run_scalability(
        self,
        sizes: Optional[List[int]] = None,
    ) -> BenchmarkSuiteResult:
        """Run scalability benchmarks.

        Parameters
        ----------
        sizes : list of int, optional
            Node counts to test.

        Returns
        -------
        BenchmarkSuiteResult
        """
        bench = ScalabilityBenchmark(
            sizes=sizes or [5, 10, 20, 50],
            n_samples=self._n_samples,
            seed=self._seed,
        )
        return self.run_synthetic(bench.generate())

    def run_all(self) -> Dict[str, BenchmarkSuiteResult]:
        """Run all benchmark types.

        Returns
        -------
        dict
            Mapping from benchmark type to results.
        """
        return {
            "standard": self.run_standard(),
            "random": self.run_synthetic(),
            "scalability": self.run_scalability(),
        }

    def report(
        self, results: Optional[Dict[str, BenchmarkSuiteResult]] = None
    ) -> str:
        """Generate a text report of benchmark results.

        Parameters
        ----------
        results : dict, optional
            Results from ``run_all()``. If None, runs all benchmarks.

        Returns
        -------
        str
            Formatted report.
        """
        if results is None:
            results = self.run_all()

        lines = [
            f"=== Benchmark Report: {self._algorithm_name} ===",
            "",
        ]

        for suite_name, suite in results.items():
            lines.append(f"--- {suite_name.upper()} ---")
            lines.append(
                f"{'Benchmark':<30s} {'SHD':>5s} {'Prec':>6s} "
                f"{'Rec':>6s} {'F1':>6s} {'Time':>8s}"
            )
            lines.append("-" * 70)

            for r in suite.results:
                lines.append(
                    f"{r.benchmark_name:<30s} {r.shd:>5d} "
                    f"{r.precision:>6.3f} {r.recall:>6.3f} "
                    f"{r.f1:>6.3f} {r.elapsed_seconds:>8.3f}s"
                )

            lines.append(
                f"\nSummary: mean_SHD={suite.mean_shd:.1f}, "
                f"mean_F1={suite.mean_f1:.3f}, "
                f"total_time={suite.total_time:.3f}s"
            )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComparisonRunner
# ---------------------------------------------------------------------------


class ComparisonRunner:
    """Compare CausalQD against baseline algorithms.

    Runs each registered algorithm on each benchmark and produces
    comparison tables.

    Parameters
    ----------
    n_samples : int
        Data samples per benchmark.
    seed : int
        Random seed.

    Examples
    --------
    >>> comp = ComparisonRunner()
    >>> comp.add_algorithm("empty", lambda d, n: np.zeros((n, n), dtype=np.int8))
    >>> comp.add_algorithm("random", random_dag_fn)
    >>> results = comp.run_standard()
    >>> print(comp.comparison_table(results))
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        self._algorithms: Dict[
            str, Callable[[DataMatrix, int], AdjacencyMatrix]
        ] = {}
        self._n_samples = n_samples
        self._seed = seed

    def add_algorithm(
        self,
        name: str,
        fn: Callable[[DataMatrix, int], AdjacencyMatrix],
    ) -> None:
        """Register an algorithm for comparison.

        Parameters
        ----------
        name : str
            Algorithm name.
        fn : callable
            ``(data, n_nodes) -> adjacency_matrix``
        """
        self._algorithms[name] = fn

    def run_standard(
        self,
        benchmarks: Optional[List[BenchmarkNetwork]] = None,
    ) -> Dict[str, BenchmarkSuiteResult]:
        """Run all algorithms on standard benchmarks.

        Returns
        -------
        dict
            Mapping from algorithm name to results.
        """
        results: Dict[str, BenchmarkSuiteResult] = {}
        for name, fn in self._algorithms.items():
            runner = BenchmarkRunner(
                fn, name, self._n_samples, self._seed
            )
            results[name] = runner.run_standard(benchmarks)
        return results

    def run_synthetic(
        self,
        specs: Optional[List[SyntheticBenchmarkSpec]] = None,
        n_nodes: int = 10,
        n_instances: int = 5,
    ) -> Dict[str, BenchmarkSuiteResult]:
        """Run all algorithms on synthetic benchmarks.

        Uses the same benchmark instances for all algorithms.
        """
        # Generate once
        if specs is None:
            gen = RandomDAGBenchmark(
                n_nodes=n_nodes,
                n_instances=n_instances,
                n_samples=self._n_samples,
                seed=self._seed,
            )
            specs = gen.generate()

        results: Dict[str, BenchmarkSuiteResult] = {}
        for name, fn in self._algorithms.items():
            runner = BenchmarkRunner(
                fn, name, self._n_samples, self._seed
            )
            results[name] = runner.run_synthetic(specs)
        return results

    def comparison_table(
        self,
        results: Dict[str, BenchmarkSuiteResult],
    ) -> str:
        """Generate a comparison table.

        Parameters
        ----------
        results : dict
            Mapping from algorithm name to BenchmarkSuiteResult.

        Returns
        -------
        str
            Formatted comparison table.
        """
        lines = ["=== Algorithm Comparison ===", ""]

        # Header
        algo_names = list(results.keys())
        header = f"{'Benchmark':<25s}"
        for name in algo_names:
            header += f" | {name:>12s}"
        lines.append(header)
        lines.append("-" * len(header))

        # Group by benchmark name
        all_benchmarks_set: Dict[str, Dict[str, AlgorithmRunResult]] = (
            defaultdict(dict)
        )
        for algo_name, suite in results.items():
            for r in suite.results:
                all_benchmarks_set[r.benchmark_name][algo_name] = r

        # Rows (SHD)
        lines.append("SHD (lower is better):")
        for bench_name in sorted(all_benchmarks_set.keys()):
            row = f"  {bench_name:<23s}"
            for algo in algo_names:
                if algo in all_benchmarks_set[bench_name]:
                    shd = all_benchmarks_set[bench_name][algo].shd
                    row += f" | {shd:>12d}"
                else:
                    row += f" | {'N/A':>12s}"
            lines.append(row)

        # Summary
        lines.append("")
        lines.append("Summary:")
        summary_header = f"  {'Metric':<20s}"
        for name in algo_names:
            summary_header += f" | {name:>12s}"
        lines.append(summary_header)
        lines.append("  " + "-" * (len(summary_header) - 2))

        for metric, getter in [
            ("Mean SHD", lambda s: f"{s.mean_shd:.1f}"),
            ("Mean F1", lambda s: f"{s.mean_f1:.3f}"),
            ("Mean Precision", lambda s: f"{s.mean_precision:.3f}"),
            ("Mean Recall", lambda s: f"{s.mean_recall:.3f}"),
            ("Total Time (s)", lambda s: f"{s.total_time:.2f}"),
        ]:
            row = f"  {metric:<20s}"
            for name in algo_names:
                row += f" | {getter(results[name]):>12s}"
            lines.append(row)

        return "\n".join(lines)

    def rank_algorithms(
        self,
        results: Dict[str, BenchmarkSuiteResult],
        metric: str = "mean_shd",
    ) -> List[Tuple[str, float]]:
        """Rank algorithms by a metric.

        Parameters
        ----------
        results : dict
            Results from a comparison run.
        metric : str
            Metric to rank by.

        Returns
        -------
        list of (str, float)
            Sorted list of (algorithm_name, metric_value).
        """
        rankings: List[Tuple[str, float]] = []
        for name, suite in results.items():
            value = getattr(suite, metric, 0.0)
            rankings.append((name, value))

        reverse = metric in ("mean_f1", "mean_precision", "mean_recall")
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        return rankings

    def statistical_test(
        self,
        results: Dict[str, BenchmarkSuiteResult],
        baseline: str,
        alternative: str,
        metric: str = "shd",
    ) -> Dict[str, Any]:
        """Perform a paired statistical test between two algorithms.

        Uses the Wilcoxon signed-rank test on per-benchmark SHD values.

        Parameters
        ----------
        results : dict
            Results from a comparison run.
        baseline : str
            Baseline algorithm name.
        alternative : str
            Alternative algorithm name.
        metric : str
            Metric to compare.

        Returns
        -------
        dict
            Test results including statistic, p-value, and effect size.
        """
        if baseline not in results or alternative not in results:
            return {"error": "Algorithm not found in results."}

        base_results = results[baseline].results
        alt_results = results[alternative].results

        # Match by benchmark name
        base_map = {r.benchmark_name: r for r in base_results}
        alt_map = {r.benchmark_name: r for r in alt_results}

        common = sorted(set(base_map.keys()) & set(alt_map.keys()))
        if len(common) < 3:
            return {
                "error": "Not enough common benchmarks for statistical test."
            }

        base_vals = np.array(
            [getattr(base_map[b], metric, 0) for b in common],
            dtype=np.float64,
        )
        alt_vals = np.array(
            [getattr(alt_map[b], metric, 0) for b in common],
            dtype=np.float64,
        )

        diffs = base_vals - alt_vals
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        # Simple sign test
        n_positive = int(np.sum(diffs > 0))
        n_negative = int(np.sum(diffs < 0))
        n_ties = int(np.sum(diffs == 0))

        return {
            "baseline": baseline,
            "alternative": alternative,
            "metric": metric,
            "n_benchmarks": len(common),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "n_wins_baseline": n_positive,
            "n_wins_alternative": n_negative,
            "n_ties": n_ties,
            "effect_size": mean_diff / (std_diff + 1e-15),
        }
