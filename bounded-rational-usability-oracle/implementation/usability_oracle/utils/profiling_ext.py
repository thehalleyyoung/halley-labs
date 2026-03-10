"""
usability_oracle.utils.profiling_ext — Extended profiling utilities.

Extends :mod:`usability_oracle.utils.profiling` with statistical timing,
flame-graph data generation, and performance regression detection.

Key components
--------------
- :class:`TimingContext` — context manager for timing code blocks.
- :class:`MemoryContext` — context manager for memory tracking.
- :class:`CallCounter` — instrumentation for counting function invocations.
- :func:`generate_flamegraph_data` — produce folded-stack format for
  flame-graph visualisation.
- :class:`RegressionDetector` — statistical detection of performance
  regressions between runs.
- :class:`BenchmarkSuite` — run multiple benchmarks and compare results.
- :func:`statistical_timing` — run a function multiple times, reporting
  mean, std, and confidence interval.

Performance characteristics
---------------------------
- ``TimingContext`` overhead: ~100 ns (two ``perf_counter`` calls).
- ``CallCounter`` overhead: ~50 ns per call (atomic increment).
- ``statistical_timing``: O(n_runs · f) where f is the function cost.
"""

from __future__ import annotations

import functools
import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Context manager for timing code blocks
# ---------------------------------------------------------------------------


class TimingContext:
    """Context manager that measures wall-clock time of a code block.

    Usage::

        with TimingContext("solve") as t:
            result = solve(mdp)
        print(f"solve took {t.elapsed:.3f}s")

    Overhead: ~100 ns (two ``time.perf_counter`` calls).

    Parameters
    ----------
    name : str
        Human-readable label.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __repr__(self) -> str:
        if self.elapsed < 1.0:
            return f"TimingContext({self.name!r}, {self.elapsed * 1000:.2f}ms)"
        return f"TimingContext({self.name!r}, {self.elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# Memory usage tracking
# ---------------------------------------------------------------------------


class MemoryContext:
    """Context manager that measures peak memory delta of a code block.

    Uses :mod:`tracemalloc` under the hood.

    Usage::

        with MemoryContext("build_tree") as m:
            tree = build(data)
        print(f"peak delta: {m.peak_bytes} B")

    Parameters
    ----------
    name : str
        Human-readable label.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.peak_bytes: int = 0
        self.current_bytes: int = 0
        self._was_tracing: bool = False

    def __enter__(self) -> "MemoryContext":
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start()
        self._snap_before = tracemalloc.take_snapshot()
        return self

    def __exit__(self, *exc: Any) -> None:
        snap_after = tracemalloc.take_snapshot()
        stats = snap_after.compare_to(self._snap_before, "lineno")
        self.peak_bytes = max(sum(s.size_diff for s in stats), 0)
        self.current_bytes = sum(s.size for s in snap_after.statistics("lineno"))
        if not self._was_tracing:
            tracemalloc.stop()


# ---------------------------------------------------------------------------
# Call count instrumentation
# ---------------------------------------------------------------------------


class CallCounter:
    """Decorator / wrapper that counts function invocations.

    Usage::

        counter = CallCounter()

        @counter.track
        def process(x):
            ...

        process(1); process(2)
        print(counter.counts)  # {'process': 2}

    Overhead: ~50 ns per call (dict lookup + increment).
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._total_time: Dict[str, float] = {}

    def track(self, fn: Callable) -> Callable:
        """Decorator that registers *fn* for call counting."""
        name = fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._counts[name] = self._counts.get(name, 0) + 1
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            self._total_time[name] = (
                self._total_time.get(name, 0.0) + time.perf_counter() - t0
            )
            return result

        return wrapper

    @property
    def counts(self) -> Dict[str, int]:
        """Invocation counts per function."""
        return dict(self._counts)

    @property
    def total_times(self) -> Dict[str, float]:
        """Cumulative wall-clock time per function in seconds."""
        return dict(self._total_time)

    def avg_time(self, name: str) -> float:
        """Average call time for *name* in seconds."""
        c = self._counts.get(name, 0)
        return self._total_time.get(name, 0.0) / c if c > 0 else 0.0

    def reset(self) -> None:
        self._counts.clear()
        self._total_time.clear()

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Call Counts & Timings", "-" * 50]
        for name in sorted(self._counts):
            c = self._counts[name]
            t = self._total_time.get(name, 0.0)
            avg = t / c if c > 0 else 0.0
            lines.append(f"  {name:30s}  calls={c:6d}  total={t:.3f}s  avg={avg * 1000:.3f}ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flame graph data generation
# ---------------------------------------------------------------------------


@dataclass
class StackSample:
    """A single sample of a call stack with timing.

    Attributes
    ----------
    stack : list of str
        Function names from outermost to innermost.
    duration_s : float
        Duration of the sample in seconds.
    """

    stack: List[str] = field(default_factory=list)
    duration_s: float = 0.0


def generate_flamegraph_data(samples: Sequence[StackSample]) -> str:
    """Produce folded-stack format suitable for flame-graph visualisation.

    Each line is ``func1;func2;func3 count`` where count is the number
    of samples (or total microseconds) with that exact stack.

    Parameters
    ----------
    samples : sequence of StackSample
        Collected stack samples.

    Returns
    -------
    str
        Folded-stack text (one line per unique stack).

    References
    ----------
    Gregg, B. (2016). *The Flame Graph*. CACM 59(6).
    """
    aggregated: Dict[str, int] = {}
    for sample in samples:
        key = ";".join(sample.stack) if sample.stack else "(idle)"
        us = max(1, int(sample.duration_s * 1_000_000))
        aggregated[key] = aggregated.get(key, 0) + us

    lines = [f"{stack} {count}" for stack, count in sorted(aggregated.items())]
    return "\n".join(lines)


def collect_stack_samples(
    fn: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    interval_s: float = 0.001,
    max_samples: int = 1000,
) -> List[StackSample]:
    """Collect stack samples by calling *fn* and recording its execution.

    This is a simplified profiler that records entry/exit of *fn*.

    Parameters
    ----------
    fn : callable
        Function to profile.
    args : tuple
        Positional arguments.
    kwargs : dict or None
        Keyword arguments.
    interval_s : float
        Ignored (single-call profiling).
    max_samples : int
        Maximum stack depth.

    Returns
    -------
    list of StackSample
        Collected samples.
    """
    if kwargs is None:
        kwargs = {}

    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    name = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    return [StackSample(stack=[name], duration_s=elapsed)]


# ---------------------------------------------------------------------------
# Performance regression detection
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Statistical summary of repeated timing measurements.

    Attributes
    ----------
    name : str
        Benchmark name.
    mean : float
        Mean time in seconds.
    std : float
        Standard deviation.
    median : float
        Median time.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    n_runs : int
        Number of runs.
    raw : list of float
        Raw timings.
    """

    name: str = ""
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_runs: int = 0
    raw: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TimingResult({self.name!r}, mean={self.mean * 1000:.2f}ms, "
            f"std={self.std * 1000:.2f}ms, n={self.n_runs}, "
            f"CI=[{self.ci_lower * 1000:.2f}, {self.ci_upper * 1000:.2f}]ms)"
        )


def statistical_timing(
    fn: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    n_runs: int = 30,
    warmup: int = 5,
    name: str = "",
    confidence: float = 0.95,
) -> TimingResult:
    """Run *fn* multiple times with statistical summary.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    args : tuple
        Positional arguments.
    kwargs : dict or None
        Keyword arguments.
    n_runs : int
        Number of timed runs (after warmup).
    warmup : int
        Number of untimed warmup calls.
    name : str
        Label for the result.
    confidence : float
        Confidence level for interval (default 0.95).

    Returns
    -------
    TimingResult
        Statistical summary.

    Complexity
    ----------
    O(n_runs · f) where f is the cost of one call.
    """
    if kwargs is None:
        kwargs = {}

    for _ in range(warmup):
        fn(*args, **kwargs)

    timings: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        timings.append(time.perf_counter() - t0)

    arr = np.array(timings, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_runs > 1 else 0.0
    median = float(np.median(arr))

    # Confidence interval via t-distribution approximation
    if n_runs > 1:
        se = std / math.sqrt(n_runs)
        # z for 95% ≈ 1.96; use normal approximation for simplicity
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        ci_lower = mean - z * se
        ci_upper = mean + z * se
    else:
        ci_lower = mean
        ci_upper = mean

    return TimingResult(
        name=name or getattr(fn, "__qualname__", "unknown"),
        mean=mean,
        std=std,
        median=median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_runs=n_runs,
        raw=timings,
    )


class RegressionDetector:
    """Detect performance regressions by comparing timing results.

    Uses a threshold-based approach: if the new mean exceeds the
    baseline mean by more than *threshold_pct* percent (accounting for
    noise), a regression is flagged.

    Parameters
    ----------
    threshold_pct : float
        Percentage above baseline mean to consider a regression (default 10%).
    min_effect_s : float
        Minimum absolute effect in seconds to report (default 0.001).
    """

    def __init__(
        self,
        threshold_pct: float = 10.0,
        min_effect_s: float = 0.001,
    ) -> None:
        self.threshold_pct = threshold_pct
        self.min_effect_s = min_effect_s

    def compare(
        self,
        baseline: TimingResult,
        current: TimingResult,
    ) -> Tuple[bool, str]:
        """Compare *current* against *baseline*.

        Returns
        -------
        is_regression : bool
            ``True`` if a significant regression is detected.
        message : str
            Human-readable comparison message.
        """
        if baseline.mean < 1e-12:
            return False, f"{current.name}: baseline too fast to compare"

        delta = current.mean - baseline.mean
        pct_change = (delta / baseline.mean) * 100.0
        is_regression = (
            pct_change > self.threshold_pct
            and abs(delta) > self.min_effect_s
        )

        if is_regression:
            msg = (
                f"⚠ REGRESSION {current.name}: "
                f"{baseline.mean * 1000:.2f}ms → {current.mean * 1000:.2f}ms "
                f"(+{pct_change:.1f}%, Δ={delta * 1000:.2f}ms)"
            )
        elif pct_change < -self.threshold_pct and abs(delta) > self.min_effect_s:
            msg = (
                f"✓ IMPROVEMENT {current.name}: "
                f"{baseline.mean * 1000:.2f}ms → {current.mean * 1000:.2f}ms "
                f"({pct_change:.1f}%, Δ={delta * 1000:.2f}ms)"
            )
        else:
            msg = (
                f"  {current.name}: "
                f"{baseline.mean * 1000:.2f}ms → {current.mean * 1000:.2f}ms "
                f"({pct_change:+.1f}%)"
            )

        return is_regression, msg


# ---------------------------------------------------------------------------
# Benchmark comparison suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Collection of named benchmarks for comparison.

    Usage::

        suite = BenchmarkSuite()
        suite.add("softmax_100", softmax, args=(Q_100, betas_100))
        suite.add("softmax_1000", softmax, args=(Q_1000, betas_1000))
        results = suite.run()
        print(suite.format_results(results))
    """

    def __init__(self) -> None:
        self._benchmarks: Dict[str, Tuple[Callable, Tuple, Dict[str, Any]]] = {}

    def add(
        self,
        name: str,
        fn: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a benchmark."""
        self._benchmarks[name] = (fn, args, kwargs or {})

    def run(
        self,
        n_runs: int = 30,
        warmup: int = 5,
    ) -> Dict[str, TimingResult]:
        """Run all registered benchmarks.

        Returns
        -------
        dict[str, TimingResult]
            Results keyed by benchmark name.
        """
        results: Dict[str, TimingResult] = {}
        for name, (fn, args, kwargs) in self._benchmarks.items():
            results[name] = statistical_timing(
                fn, args=args, kwargs=kwargs, n_runs=n_runs, warmup=warmup, name=name
            )
        return results

    @staticmethod
    def format_results(results: Dict[str, TimingResult]) -> str:
        """Format results as a human-readable table."""
        lines = [
            "Benchmark Results",
            "=" * 70,
            f"  {'Name':30s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'CI 95%':>18s}",
            "  " + "-" * 66,
        ]
        for name, r in sorted(results.items()):
            lines.append(
                f"  {name:30s} "
                f"{r.mean * 1000:9.2f}ms "
                f"{r.std * 1000:9.2f}ms "
                f"{r.median * 1000:9.2f}ms "
                f"[{r.ci_lower * 1000:.2f}, {r.ci_upper * 1000:.2f}]ms"
            )
        return "\n".join(lines)

    @staticmethod
    def compare_results(
        baseline: Dict[str, TimingResult],
        current: Dict[str, TimingResult],
        threshold_pct: float = 10.0,
    ) -> str:
        """Compare two sets of benchmark results.

        Returns
        -------
        str
            Human-readable comparison report.
        """
        detector = RegressionDetector(threshold_pct=threshold_pct)
        lines = ["Benchmark Comparison", "=" * 70]
        regressions = 0
        for name in sorted(set(baseline) | set(current)):
            if name in baseline and name in current:
                is_reg, msg = detector.compare(baseline[name], current[name])
                if is_reg:
                    regressions += 1
                lines.append(f"  {msg}")
            elif name in current:
                lines.append(f"  (new) {name}: {current[name].mean * 1000:.2f}ms")
            else:
                lines.append(f"  (removed) {name}")
        lines.append("-" * 70)
        if regressions > 0:
            lines.append(f"  ⚠ {regressions} regression(s) detected!")
        else:
            lines.append("  ✓ No regressions detected.")
        return "\n".join(lines)


__all__ = [
    "TimingContext",
    "MemoryContext",
    "CallCounter",
    "StackSample",
    "generate_flamegraph_data",
    "collect_stack_samples",
    "TimingResult",
    "statistical_timing",
    "RegressionDetector",
    "BenchmarkSuite",
]
