"""
usability_oracle.benchmarks.suite — Benchmark execution engine.

Runs a collection of :class:`BenchmarkCase` instances through the
usability-oracle pipeline and aggregates the results into a
:class:`BenchmarkReport` with accuracy, precision, recall, F1 and
confusion-matrix statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from usability_oracle.core.enums import RegressionVerdict


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    """A single benchmark test case.

    Attributes:
        name: Human-readable identifier for the case.
        source_a: Serialised accessibility tree (version A).
        source_b: Serialised accessibility tree (version B).
        task_spec: Task specification (path / dict / string).
        expected_verdict: Ground-truth verdict.
        category: Grouping tag (e.g. "form", "navigation").
        metadata: Arbitrary extra information.
    """

    name: str
    source_a: Any
    source_b: Any
    task_spec: Any = None
    expected_verdict: RegressionVerdict = RegressionVerdict.NEUTRAL
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running one benchmark case."""

    case_name: str
    actual_verdict: RegressionVerdict
    expected_verdict: RegressionVerdict
    timing: float = 0.0
    correct: bool = False
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.correct = self.actual_verdict == self.expected_verdict


@dataclass
class BenchmarkReport:
    """Aggregated report across all benchmark cases."""

    results: list[BenchmarkResult] = field(default_factory=list)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    timing_stats: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Benchmark Report ({len(self.results)} cases)",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1 Score:  {self.f1:.4f}",
        ]
        if self.timing_stats:
            lines.append(f"  Mean time: {self.timing_stats.get('mean', 0):.4f} s")
            lines.append(f"  P95 time:  {self.timing_stats.get('p95', 0):.4f} s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Execute a suite of benchmark cases and produce an aggregate report.

    Parameters:
        pipeline_fn: A callable that accepts ``(source_a, source_b, task_spec)``
            and returns a :class:`RegressionVerdict` (or an object with a
            ``.verdict`` attribute).
        positive_class: The verdict value treated as "positive" for
            precision / recall computation.
        verbose: Print progress to stdout.
    """

    def __init__(
        self,
        pipeline_fn: Optional[Callable[..., Any]] = None,
        positive_class: RegressionVerdict = RegressionVerdict.REGRESSION,
        verbose: bool = False,
    ) -> None:
        self._pipeline_fn = pipeline_fn
        self._positive_class = positive_class
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: dict[str, Any] | None = None, cases: list[BenchmarkCase] | None = None) -> BenchmarkReport:
        """Run all *cases* and return a :class:`BenchmarkReport`.

        Parameters:
            config: Optional configuration overrides forwarded to the pipeline.
            cases: The list of benchmark cases.  If ``None``, uses config to
                generate synthetic cases via :class:`DatasetManager`.
        """
        if cases is None:
            cases = []
        results: list[BenchmarkResult] = []
        for idx, case in enumerate(cases):
            if self._verbose:
                print(f"  [{idx + 1}/{len(cases)}] {case.name} …", end=" ", flush=True)
            result = self._run_single(case, config)
            results.append(result)
            if self._verbose:
                tag = "✓" if result.correct else "✗"
                print(f"{tag} ({result.timing:.3f}s)")
        return self._aggregate(results)

    # ------------------------------------------------------------------
    # Single execution
    # ------------------------------------------------------------------

    def _run_single(self, case: BenchmarkCase, config: dict[str, Any] | None = None) -> BenchmarkResult:
        t0 = time.perf_counter()
        try:
            if self._pipeline_fn is None:
                verdict = self._default_verdict(case)
            else:
                raw = self._pipeline_fn(case.source_a, case.source_b, case.task_spec, **(config or {}))
                verdict = raw if isinstance(raw, RegressionVerdict) else getattr(raw, "verdict", RegressionVerdict.INCONCLUSIVE)
            elapsed = time.perf_counter() - t0
            return BenchmarkResult(
                case_name=case.name,
                actual_verdict=verdict,
                expected_verdict=case.expected_verdict,
                timing=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return BenchmarkResult(
                case_name=case.name,
                actual_verdict=RegressionVerdict.INCONCLUSIVE,
                expected_verdict=case.expected_verdict,
                timing=elapsed,
                error=str(exc),
            )

    @staticmethod
    def _default_verdict(case: BenchmarkCase) -> RegressionVerdict:
        """Trivial default: always predict NEUTRAL."""
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results: list[BenchmarkResult]) -> BenchmarkReport:
        if not results:
            return BenchmarkReport()

        n = len(results)
        correct = sum(r.correct for r in results)
        accuracy = correct / n if n else 0.0

        # Precision / recall for positive class
        tp = sum(
            1 for r in results
            if r.actual_verdict == self._positive_class and r.expected_verdict == self._positive_class
        )
        fp = sum(
            1 for r in results
            if r.actual_verdict == self._positive_class and r.expected_verdict != self._positive_class
        )
        fn = sum(
            1 for r in results
            if r.actual_verdict != self._positive_class and r.expected_verdict == self._positive_class
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Timing statistics
        times = np.array([r.timing for r in results])
        timing_stats = {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
        }

        # Confusion matrix
        labels = sorted({r.expected_verdict for r in results} | {r.actual_verdict for r in results}, key=lambda v: v.value)
        label_idx = {v: i for i, v in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for r in results:
            cm[label_idx[r.expected_verdict], label_idx[r.actual_verdict]] += 1

        return BenchmarkReport(
            results=results,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            timing_stats=timing_stats,
            confusion_matrix=cm,
            metadata={"n_cases": n, "labels": [v.value for v in labels]},
        )

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------

    def per_category_report(
        self, results: list[BenchmarkResult],
    ) -> dict[str, dict[str, float]]:
        """Produce per-category metrics from benchmark results."""
        if not results:
            return {}

        cats: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            cat = r.metadata.get("category", "unknown")
            cats.setdefault(cat, []).append(r)

        summaries: dict[str, dict[str, float]] = {}
        for cat, res in cats.items():
            n_total = len(res)
            n_correct = sum(1 for r in res if r.correct)
            times = np.array([r.timing for r in res])
            summaries[cat] = {
                "count": n_total,
                "accuracy": n_correct / n_total if n_total > 0 else 0.0,
                "mean_time": float(np.mean(times)),
                "median_time": float(np.median(times)),
                "p95_time": float(np.percentile(times, 95)),
            }
        return summaries

    # ------------------------------------------------------------------
    # Statistical comparison of two benchmark runs
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        results_a: list[BenchmarkResult],
        results_b: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Compare two benchmark runs via paired metrics and effect sizes."""
        correct_a = np.array([int(r.correct) for r in results_a], dtype=float)
        correct_b = np.array([int(r.correct) for r in results_b], dtype=float)
        times_a = np.array([r.timing for r in results_a], dtype=float)
        times_b = np.array([r.timing for r in results_b], dtype=float)

        acc_a = float(np.mean(correct_a))
        acc_b = float(np.mean(correct_b))

        # McNemar-like: how often do they disagree?
        n = min(len(correct_a), len(correct_b))
        a_right_b_wrong = sum(1 for i in range(n) if correct_a[i] and not correct_b[i])
        a_wrong_b_right = sum(1 for i in range(n) if not correct_a[i] and correct_b[i])

        # Cohen's d for timing
        pooled_std = np.sqrt((np.var(times_a) + np.var(times_b)) / 2.0)
        cohens_d = (float(np.mean(times_b)) - float(np.mean(times_a))) / pooled_std if pooled_std > 1e-12 else 0.0

        return {
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "accuracy_diff": acc_b - acc_a,
            "a_right_b_wrong": a_right_b_wrong,
            "a_wrong_b_right": a_wrong_b_right,
            "timing_mean_a": float(np.mean(times_a)),
            "timing_mean_b": float(np.mean(times_b)),
            "timing_cohens_d": cohens_d,
        }

    # ------------------------------------------------------------------
    # Warmup + repeated measurement
    # ------------------------------------------------------------------

    def run_with_warmup(
        self,
        cases: list[BenchmarkCase],
        n_warmup: int = 3,
        n_repeats: int = 5,
    ) -> dict[str, Any]:
        """Run benchmark cases with warmup iterations.

        Discards first *n_warmup* executions per case, then averages
        *n_repeats* measured executions.
        """
        stable_results: list[BenchmarkResult] = []
        for case in cases:
            timings: list[float] = []
            last_result: BenchmarkResult | None = None
            for rep in range(n_warmup + n_repeats):
                result = self._run_single(case)
                if rep >= n_warmup:
                    timings.append(result.timing)
                    last_result = result
            if last_result is not None:
                last_result.timing = float(np.mean(timings))
                last_result.metadata["timing_std"] = float(np.std(timings))
                last_result.metadata["timing_min"] = float(min(timings))
                last_result.metadata["timing_max"] = float(max(timings))
                stable_results.append(last_result)
        return self._aggregate(stable_results).__dict__

    # ------------------------------------------------------------------
    # Memory profiling
    # ------------------------------------------------------------------

    def profile_memory(self, cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
        """Profile peak memory usage for each benchmark case.

        Uses a simple tracking allocator approach (gc-based).
        """
        import gc

        profiles: list[dict[str, Any]] = []
        for case in cases:
            gc.collect()
            gc.collect()
            before = _get_process_memory()
            result = self._run_single(case)
            after = _get_process_memory()
            delta_kb = max(after - before, 0)

            profiles.append({
                "case_name": case.name,
                "correct": result.correct,
                "memory_before_kb": before,
                "memory_after_kb": after,
                "memory_delta_kb": delta_kb,
                "timing": result.timing,
            })
        return profiles


# ---------------------------------------------------------------------------
# Process memory helper
# ---------------------------------------------------------------------------

def _get_process_memory() -> float:
    """Return current process RSS in kilobytes (best effort)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss)  # macOS: bytes; Linux: KB
    except ImportError:
        return 0.0
