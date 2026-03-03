"""Profiling and benchmarking utilities for CausalQD.

Provides hierarchical timing, memory profiling, operation counting,
benchmark running, and a MAP-Elites profiling callback.

Key classes
-----------
* :class:`Timer` – context manager with hierarchical timing
* :class:`MemoryProfiler` – track memory allocation
* :class:`OperationCounter` – count operations per component
* :class:`PerformanceReport` – generate profiling reports
* :class:`BenchmarkRunner` – run standardized benchmarks
* :class:`ProfilingCallback` – per-generation profiling for MAP-Elites
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Timer",
    "MemoryProfiler",
    "OperationCounter",
    "PerformanceReport",
    "BenchmarkRunner",
    "ProfilingCallback",
    "BenchmarkResult",
]


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Hierarchical timing context manager.

    Supports nested sections for fine-grained profiling of computation
    phases.

    Examples
    --------
    >>> timer = Timer()
    >>> with timer.section("scoring"):
    ...     with timer.section("local_score"):
    ...         compute_score()
    ...     with timer.section("cache_lookup"):
    ...         check_cache()
    >>> timer.report()
    """

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[Tuple[str, float]] = []
        self._total_start: Optional[float] = None
        self._total_elapsed: float = 0.0

    @contextmanager
    def section(self, name: str):  # type: ignore[no-untyped-def]
        """Time a named section.

        Parameters
        ----------
        name : str
            Section name.  Nested sections are joined with ``/``.

        Yields
        ------
        None
        """
        full_name = "/".join([s[0] for s in self._stack] + [name])
        t0 = time.perf_counter()
        self._stack.append((name, t0))
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._stack.pop()
            self._timings[full_name].append(elapsed)

    def start(self) -> None:
        """Start the total timer."""
        self._total_start = time.perf_counter()

    def stop(self) -> float:
        """Stop the total timer and return elapsed time."""
        if self._total_start is not None:
            self._total_elapsed = time.perf_counter() - self._total_start
            self._total_start = None
        return self._total_elapsed

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    @property
    def sections(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all sections.

        Returns
        -------
        dict
            Mapping from section name to stats dict with keys:
            ``total``, ``count``, ``mean``, ``min``, ``max``, ``std``.
        """
        result: Dict[str, Dict[str, float]] = {}
        for name, times in self._timings.items():
            arr = np.array(times)
            result[name] = {
                "total": float(np.sum(arr)),
                "count": float(len(arr)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            }
        return result

    @property
    def total_elapsed(self) -> float:
        """Total elapsed time."""
        if self._total_start is not None:
            return time.perf_counter() - self._total_start
        return self._total_elapsed

    def report(self) -> str:
        """Generate a human-readable timing report.

        Returns
        -------
        str
            Formatted report.
        """
        lines = ["=== Timing Report ==="]
        lines.append(f"Total elapsed: {self._total_elapsed:.4f}s")
        lines.append("")

        sections = self.sections
        if not sections:
            lines.append("No sections recorded.")
            return "\n".join(lines)

        # Sort by total time descending
        sorted_sections = sorted(
            sections.items(), key=lambda x: x[1]["total"], reverse=True
        )

        max_name_len = max(len(name) for name, _ in sorted_sections)
        fmt = f"  {{:<{max_name_len}}}  {{:>10.4f}}s  {{:>6}}x  {{:>10.4f}}s avg"

        for name, stats in sorted_sections:
            lines.append(
                fmt.format(
                    name,
                    stats["total"],
                    int(stats["count"]),
                    stats["mean"],
                )
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all timings."""
        self._timings.clear()
        self._stack.clear()
        self._total_start = None
        self._total_elapsed = 0.0


# ---------------------------------------------------------------------------
# MemoryProfiler
# ---------------------------------------------------------------------------


class MemoryProfiler:
    """Track memory allocation using ``tracemalloc``.

    Examples
    --------
    >>> mp = MemoryProfiler()
    >>> mp.start()
    >>> # ... allocate memory ...
    >>> snapshot = mp.snapshot()
    >>> mp.stop()
    >>> print(snapshot)
    """

    def __init__(self) -> None:
        self._snapshots: List[Dict[str, Any]] = []
        self._started = False

    def start(self) -> None:
        """Start memory tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._started = True

    def stop(self) -> None:
        """Stop memory tracking."""
        if self._started and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started = False

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot.

        Parameters
        ----------
        label : str
            Label for the snapshot.

        Returns
        -------
        dict
            Snapshot data with keys: ``label``, ``current_bytes``,
            ``peak_bytes``, ``timestamp``.
        """
        if not tracemalloc.is_tracing():
            return {
                "label": label,
                "current_bytes": 0,
                "peak_bytes": 0,
                "timestamp": time.time(),
            }

        current, peak = tracemalloc.get_traced_memory()
        snap = {
            "label": label,
            "current_bytes": current,
            "peak_bytes": peak,
            "timestamp": time.time(),
        }
        self._snapshots.append(snap)
        return snap

    def current_usage(self) -> int:
        """Current memory usage in bytes."""
        if not tracemalloc.is_tracing():
            return 0
        current, _ = tracemalloc.get_traced_memory()
        return current

    def peak_usage(self) -> int:
        """Peak memory usage in bytes."""
        if not tracemalloc.is_tracing():
            return 0
        _, peak = tracemalloc.get_traced_memory()
        return peak

    @property
    def snapshots(self) -> List[Dict[str, Any]]:
        return self._snapshots

    def report(self) -> str:
        """Generate memory usage report."""
        lines = ["=== Memory Report ==="]
        for snap in self._snapshots:
            current_mb = snap["current_bytes"] / (1024 * 1024)
            peak_mb = snap["peak_bytes"] / (1024 * 1024)
            lines.append(
                f"  {snap['label']:>20s}  "
                f"current={current_mb:.2f}MB  peak={peak_mb:.2f}MB"
            )
        return "\n".join(lines)

    def __enter__(self) -> MemoryProfiler:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# OperationCounter
# ---------------------------------------------------------------------------


class OperationCounter:
    """Count operations per component for profiling.

    Tracks call counts, success rates, and categorized operations.

    Examples
    --------
    >>> counter = OperationCounter()
    >>> counter.increment("score_computation")
    >>> counter.increment("score_computation")
    >>> counter.increment("cache_hit")
    >>> counter.get("score_computation")
    2
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)
        self._values: Dict[str, List[float]] = defaultdict(list)

    def increment(self, operation: str, count: int = 1) -> None:
        """Increment count for an operation.

        Parameters
        ----------
        operation : str
            Operation name.
        count : int
            Increment amount.
        """
        self._counts[operation] += count

    def record(self, operation: str, value: float) -> None:
        """Record a numeric value for an operation.

        Parameters
        ----------
        operation : str
            Operation name.
        value : float
            Value to record.
        """
        self._counts[operation] += 1
        self._values[operation].append(value)

    def get(self, operation: str) -> int:
        """Get count for an operation."""
        return self._counts.get(operation, 0)

    def get_values(self, operation: str) -> List[float]:
        """Get recorded values for an operation."""
        return self._values.get(operation, [])

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation with recorded values."""
        values = self._values.get(operation, [])
        if not values:
            return {
                "count": float(self._counts.get(operation, 0)),
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        arr = np.array(values)
        return {
            "count": float(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @property
    def operations(self) -> Dict[str, int]:
        """All operation counts."""
        return dict(self._counts)

    def reset(self) -> None:
        """Reset all counters."""
        self._counts.clear()
        self._values.clear()

    def report(self) -> str:
        """Generate operation count report."""
        lines = ["=== Operation Counts ==="]
        for op, count in sorted(
            self._counts.items(), key=lambda x: x[1], reverse=True
        ):
            line = f"  {op:>30s}: {count:>10d}"
            if op in self._values:
                stats = self.get_stats(op)
                line += f"  (mean={stats['mean']:.4f})"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PerformanceReport
# ---------------------------------------------------------------------------


class PerformanceReport:
    """Aggregate profiling data from Timer, MemoryProfiler, and OperationCounter.

    Parameters
    ----------
    timer : Timer, optional
        Timer instance.
    memory : MemoryProfiler, optional
        Memory profiler instance.
    counter : OperationCounter, optional
        Operation counter instance.
    """

    def __init__(
        self,
        timer: Optional[Timer] = None,
        memory: Optional[MemoryProfiler] = None,
        counter: Optional[OperationCounter] = None,
    ) -> None:
        self.timer = timer or Timer()
        self.memory = memory or MemoryProfiler()
        self.counter = counter or OperationCounter()

    def full_report(self) -> str:
        """Generate complete profiling report."""
        parts = []
        parts.append(self.timer.report())
        parts.append("")
        if self.memory.snapshots:
            parts.append(self.memory.report())
            parts.append("")
        parts.append(self.counter.report())
        return "\n".join(parts)

    def summary(self) -> Dict[str, Any]:
        """Structured summary of profiling data."""
        return {
            "total_time": self.timer.total_elapsed,
            "timing_sections": self.timer.sections,
            "memory_snapshots": self.memory.snapshots,
            "operation_counts": self.counter.operations,
        }


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes
    ----------
    name : str
        Benchmark name.
    n_nodes : int
        Problem size (number of nodes).
    time_seconds : float
        Elapsed wall time.
    memory_bytes : int
        Peak memory usage.
    iterations_per_second : float
        Throughput metric.
    extra : dict
        Additional benchmark-specific metrics.
    """

    name: str
    n_nodes: int
    time_seconds: float
    memory_bytes: int = 0
    iterations_per_second: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Run standardized benchmarks for the CausalQD pipeline.

    Provides a benchmark suite for different problem sizes, scaling analysis,
    and comparison with baseline implementations.

    Examples
    --------
    >>> runner = BenchmarkRunner()
    >>> runner.add_benchmark("dag_creation", create_dag_benchmark)
    >>> results = runner.run_all()
    >>> runner.scaling_report(results)
    """

    def __init__(self) -> None:
        self._benchmarks: Dict[str, Callable[..., BenchmarkResult]] = {}
        self._results: List[BenchmarkResult] = []

    def add_benchmark(
        self,
        name: str,
        fn: Callable[..., BenchmarkResult],
    ) -> None:
        """Register a benchmark function.

        Parameters
        ----------
        name : str
            Benchmark name.
        fn : callable
            Function that takes ``(n_nodes, **kwargs)`` and returns
            a :class:`BenchmarkResult`.
        """
        self._benchmarks[name] = fn

    def run_benchmark(
        self,
        name: str,
        n_nodes: int,
        n_repeats: int = 3,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run a specific benchmark.

        Parameters
        ----------
        name : str
            Benchmark name.
        n_nodes : int
            Problem size.
        n_repeats : int
            Number of repetitions (best time used).
        **kwargs
            Extra arguments passed to the benchmark function.

        Returns
        -------
        BenchmarkResult
        """
        if name not in self._benchmarks:
            raise KeyError(f"Benchmark '{name}' not found.")

        fn = self._benchmarks[name]
        best_time = float("inf")
        best_result = None

        for _ in range(n_repeats):
            gc.collect()
            result = fn(n_nodes, **kwargs)
            if result.time_seconds < best_time:
                best_time = result.time_seconds
                best_result = result

        assert best_result is not None
        self._results.append(best_result)
        return best_result

    def run_all(
        self,
        sizes: Optional[List[int]] = None,
        n_repeats: int = 3,
        **kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run all registered benchmarks at multiple sizes.

        Parameters
        ----------
        sizes : list of int, optional
            Problem sizes. Default: ``[5, 10, 20, 50]``.
        n_repeats : int
            Repetitions per configuration.

        Returns
        -------
        list of BenchmarkResult
        """
        if sizes is None:
            sizes = [5, 10, 20, 50]

        results: List[BenchmarkResult] = []
        for name in self._benchmarks:
            for n in sizes:
                result = self.run_benchmark(name, n, n_repeats, **kwargs)
                results.append(result)

        return results

    def scaling_analysis(
        self,
        results: Optional[List[BenchmarkResult]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze how time and memory scale with problem size.

        Parameters
        ----------
        results : list of BenchmarkResult, optional
            Results to analyze. Uses stored results if None.

        Returns
        -------
        dict
            Mapping from benchmark name to arrays of sizes and times.
        """
        if results is None:
            results = self._results

        by_name: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for r in results:
            by_name[r.name].append(r)

        analysis: Dict[str, Dict[str, np.ndarray]] = {}
        for name, runs in by_name.items():
            runs_sorted = sorted(runs, key=lambda r: r.n_nodes)
            sizes = np.array([r.n_nodes for r in runs_sorted])
            times = np.array([r.time_seconds for r in runs_sorted])
            memory = np.array([r.memory_bytes for r in runs_sorted])
            analysis[name] = {
                "sizes": sizes,
                "times": times,
                "memory": memory,
            }

        return analysis

    def scaling_report(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> str:
        """Generate scaling analysis report."""
        analysis = self.scaling_analysis(results)
        lines = ["=== Scaling Report ==="]

        for name, data in analysis.items():
            lines.append(f"\n  {name}:")
            sizes = data["sizes"]
            times = data["times"]

            for i in range(len(sizes)):
                mem_str = ""
                if data["memory"][i] > 0:
                    mem_mb = data["memory"][i] / (1024 * 1024)
                    mem_str = f"  mem={mem_mb:.2f}MB"
                lines.append(
                    f"    n={sizes[i]:>4d}  "
                    f"time={times[i]:.6f}s{mem_str}"
                )

            # Estimate scaling exponent via log-log regression
            if len(sizes) >= 2:
                log_n = np.log(sizes.astype(np.float64))
                log_t = np.log(np.maximum(times, 1e-12))
                if np.std(log_n) > 0:
                    coeff = np.polyfit(log_n, log_t, 1)
                    lines.append(
                        f"    Estimated complexity: O(n^{coeff[0]:.2f})"
                    )

        return "\n".join(lines)

    def comparison_report(
        self,
        results_a: List[BenchmarkResult],
        results_b: List[BenchmarkResult],
        label_a: str = "A",
        label_b: str = "B",
    ) -> str:
        """Compare two sets of benchmark results.

        Parameters
        ----------
        results_a, results_b : list of BenchmarkResult
            Two result sets to compare.
        label_a, label_b : str
            Labels for the two sets.

        Returns
        -------
        str
            Comparison report.
        """
        lines = [f"=== Comparison: {label_a} vs {label_b} ==="]

        # Group by (name, n_nodes)
        a_map: Dict[Tuple[str, int], float] = {}
        b_map: Dict[Tuple[str, int], float] = {}
        for r in results_a:
            a_map[(r.name, r.n_nodes)] = r.time_seconds
        for r in results_b:
            b_map[(r.name, r.n_nodes)] = r.time_seconds

        all_keys = sorted(set(a_map.keys()) & set(b_map.keys()))

        for name, n in all_keys:
            t_a = a_map[(name, n)]
            t_b = b_map[(name, n)]
            speedup = t_a / t_b if t_b > 0 else float("inf")
            lines.append(
                f"  {name} n={n}: "
                f"{label_a}={t_a:.6f}s  {label_b}={t_b:.6f}s  "
                f"speedup={speedup:.2f}x"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProfilingCallback
# ---------------------------------------------------------------------------


class ProfilingCallback:
    """Plug into MAP-Elites for per-generation profiling.

    Tracks timing, memory, and operation counts for each generation
    of the MAP-Elites loop.

    Usage with CausalMAPElites::

        profiler = ProfilingCallback()
        engine = CausalMAPElites(..., callbacks=[profiler])
        engine.run(data, n_iterations=100)
        print(profiler.report())

    The callback signature matches the MAP-Elites callback protocol:
    ``callback(iteration, archive, stats)``.
    """

    def __init__(self, track_memory: bool = False) -> None:
        self._timer = Timer()
        self._counter = OperationCounter()
        self._memory = MemoryProfiler()
        self._track_memory = track_memory
        self._generation_times: List[float] = []
        self._generation_stats: List[Dict[str, Any]] = []
        self._last_time: float = time.perf_counter()

    def __call__(
        self,
        iteration: int,
        archive: Any,
        stats: Any,
    ) -> None:
        """Record profiling data for a generation.

        Parameters
        ----------
        iteration : int
            Current iteration/generation number.
        archive : object
            Archive instance (must have ``size``, ``qd_score`` attributes
            or similar).
        stats : object
            Iteration statistics.
        """
        now = time.perf_counter()
        gen_time = now - self._last_time
        self._last_time = now

        self._generation_times.append(gen_time)
        self._counter.increment("generations")

        gen_stats: Dict[str, Any] = {
            "iteration": iteration,
            "gen_time": gen_time,
        }

        # Try to extract archive stats
        if hasattr(archive, "size"):
            gen_stats["archive_size"] = archive.size
        if hasattr(archive, "qd_score"):
            gen_stats["qd_score"] = archive.qd_score()  # type: ignore[operator]
        if hasattr(stats, "improvements"):
            gen_stats["improvements"] = stats.improvements

        if self._track_memory:
            snap = self._memory.snapshot(f"gen_{iteration}")
            gen_stats["memory_bytes"] = snap["current_bytes"]

        self._generation_stats.append(gen_stats)

    @property
    def generation_times(self) -> np.ndarray:
        """Array of per-generation wall times."""
        return np.array(self._generation_times)

    @property
    def total_time(self) -> float:
        """Total elapsed time across all recorded generations."""
        return float(np.sum(self._generation_times))

    @property
    def avg_generation_time(self) -> float:
        """Average time per generation."""
        if not self._generation_times:
            return 0.0
        return float(np.mean(self._generation_times))

    @property
    def throughput(self) -> float:
        """Generations per second."""
        if self.total_time <= 0:
            return 0.0
        return len(self._generation_times) / self.total_time

    def report(self) -> str:
        """Generate profiling report."""
        lines = ["=== MAP-Elites Profiling Report ==="]
        lines.append(f"Total generations: {len(self._generation_times)}")
        lines.append(f"Total time: {self.total_time:.4f}s")
        lines.append(f"Avg generation time: {self.avg_generation_time:.4f}s")
        lines.append(f"Throughput: {self.throughput:.1f} gen/s")

        if self._generation_times:
            times = np.array(self._generation_times)
            lines.append(f"Min gen time: {np.min(times):.4f}s")
            lines.append(f"Max gen time: {np.max(times):.4f}s")
            lines.append(f"Std gen time: {np.std(times):.4f}s")

        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Structured profiling summary."""
        return {
            "total_generations": len(self._generation_times),
            "total_time": self.total_time,
            "avg_generation_time": self.avg_generation_time,
            "throughput": self.throughput,
            "generation_stats": self._generation_stats,
        }

    def reset(self) -> None:
        """Reset all profiling data."""
        self._generation_times.clear()
        self._generation_stats.clear()
        self._counter.reset()
        self._last_time = time.perf_counter()
