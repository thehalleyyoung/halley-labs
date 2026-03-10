"""
usability_oracle.utils.profiling — Profiling utilities for the usability oracle.

Provides decorators, context managers, and reporting tools for measuring
execution time and peak memory usage of each pipeline stage.

Key components
--------------
- :class:`PipelineProfiler` — profile each pipeline stage end-to-end.
- :func:`timed` — decorator measuring wall-clock time (complements
  :mod:`usability_oracle.utils.timing` with memory tracking).
- :func:`memory_tracked` — decorator measuring peak memory delta.
- :class:`StageTimer` — context manager for timing named stages.
- :class:`MemoryTracker` — per-component memory accounting.
- :class:`ProfileReport` — structured profiling results.
- :func:`hotspot_analysis` — identify the most time-consuming stages.
- :func:`memory_report` — memory breakdown by component.
- :func:`format_profile` — human-readable profile output.
- :func:`compare_profiles` — compare two profile runs.

Performance characteristics
---------------------------
Profiling overhead is kept minimal: ``StageTimer`` adds a single
``time.perf_counter`` call pair (~100 ns); ``MemoryTracker`` relies on
``tracemalloc`` snapshots whose overhead depends on the number of live
allocations but is typically < 5 % of total runtime.
"""

from __future__ import annotations

import functools
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# ProfileReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfileReport:
    """Structured profiling results for a pipeline run.

    Attributes
    ----------
    stage_times : dict[str, float]
        Wall-clock time per stage in seconds.
    stage_memory : dict[str, int]
        Peak memory delta per stage in bytes.
    total_time : float
        Sum of all stage times.
    peak_memory : int
        Overall peak memory usage in bytes (from tracemalloc).
    metadata : dict[str, Any]
        Arbitrary metadata (e.g. input size, configuration hash).
    """

    stage_times: Dict[str, float] = field(default_factory=dict)
    stage_memory: Dict[str, int] = field(default_factory=dict)
    total_time: float = 0.0
    peak_memory: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StageTimer context manager
# ---------------------------------------------------------------------------

class StageTimer:
    """Context manager that records wall-clock time for a named stage.

    Usage::

        profiler = PipelineProfiler()
        with StageTimer("parse", profiler):
            parse(data)

    Overhead: two ``time.perf_counter`` calls (~100 ns total).
    """

    def __init__(self, name: str, profiler: Optional["PipelineProfiler"] = None) -> None:
        self.name = name
        self._profiler = profiler
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self._profiler is not None:
            self._profiler.record_time(self.name, self.elapsed)


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Track memory allocation per component using :mod:`tracemalloc`.

    Call :meth:`start` before the workload and :meth:`snapshot` after
    each component to record the memory delta.

    Overhead: depends on the number of live allocations; typically < 5 %
    of total runtime.
    """

    def __init__(self) -> None:
        self._components: Dict[str, int] = {}
        self._started = False
        self._last_snapshot: Optional[tracemalloc.Snapshot] = None

    def start(self) -> None:
        """Begin tracing memory allocations."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._started = True
        self._last_snapshot = tracemalloc.take_snapshot()

    def snapshot(self, component: str) -> int:
        """Record current memory delta for *component*.

        Returns the delta in bytes since the last snapshot.
        """
        if not self._started:
            raise RuntimeError("MemoryTracker.start() must be called first")
        current = tracemalloc.take_snapshot()
        if self._last_snapshot is not None:
            stats = current.compare_to(self._last_snapshot, "lineno")
            delta = sum(s.size_diff for s in stats)
        else:
            delta = 0
        delta = max(delta, 0)
        self._components[component] = delta
        self._last_snapshot = current
        return delta

    def stop(self) -> None:
        """Stop tracing (does not clear recorded data)."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started = False

    @property
    def components(self) -> Dict[str, int]:
        """Memory delta per component in bytes."""
        return dict(self._components)

    @property
    def total(self) -> int:
        """Total tracked memory across all components."""
        return sum(self._components.values())


# ---------------------------------------------------------------------------
# PipelineProfiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Profile each stage of the usability-oracle pipeline.

    Collects wall-clock times and optional memory snapshots, then
    produces a :class:`ProfileReport`.

    Usage::

        profiler = PipelineProfiler()
        profiler.start()
        with StageTimer("parse", profiler):
            tree = parse(html)
        with StageTimer("solve", profiler):
            policy = solve(mdp)
        report = profiler.report()
    """

    def __init__(self) -> None:
        self._stage_times: Dict[str, float] = {}
        self._stage_order: List[str] = []
        self._memory_tracker: Optional[MemoryTracker] = None
        self._started = False
        self._start_time: float = 0.0

    def start(self, track_memory: bool = False) -> None:
        """Begin profiling.

        Parameters
        ----------
        track_memory : bool
            If ``True``, enable :mod:`tracemalloc`-based memory tracking.
        """
        self._started = True
        self._start_time = time.perf_counter()
        if track_memory:
            self._memory_tracker = MemoryTracker()
            self._memory_tracker.start()

    def record_time(self, stage: str, elapsed: float) -> None:
        """Record *elapsed* seconds for *stage*."""
        self._stage_times[stage] = elapsed
        if stage not in self._stage_order:
            self._stage_order.append(stage)
        if self._memory_tracker is not None:
            self._memory_tracker.snapshot(stage)

    def report(self) -> ProfileReport:
        """Produce a :class:`ProfileReport` from collected data."""
        total_time = sum(self._stage_times.values())
        stage_mem: Dict[str, int] = {}
        peak_mem = 0
        if self._memory_tracker is not None:
            stage_mem = self._memory_tracker.components
            peak_mem = self._memory_tracker.total
            self._memory_tracker.stop()
        return ProfileReport(
            stage_times=dict(self._stage_times),
            stage_memory=stage_mem,
            total_time=total_time,
            peak_memory=peak_mem,
        )


# ---------------------------------------------------------------------------
# @timed decorator (profiling-aware)
# ---------------------------------------------------------------------------

def timed(fn: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Decorator that measures wall-clock time of each call.

    The decorated function gains ``last_timing`` (float, seconds) and
    ``all_timings`` (list[float]) attributes.

    Overhead: one ``time.perf_counter`` pair per call (~100 ns).

    Usage::

        @timed
        def solve(mdp): ...

        @timed(name="parsing")
        def parse(html): ...
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            wrapper.last_timing = elapsed  # type: ignore[attr-defined]
            wrapper.all_timings.append(elapsed)  # type: ignore[attr-defined]
            return result

        wrapper.last_timing = 0.0  # type: ignore[attr-defined]
        wrapper.all_timings = []  # type: ignore[attr-defined]
        wrapper._timed_name = label  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# @memory_tracked decorator
# ---------------------------------------------------------------------------

def memory_tracked(fn: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Decorator that measures peak memory delta of each call.

    Uses :mod:`tracemalloc` to measure the difference in traced memory
    before and after the function call.  The decorated function gains a
    ``last_memory_bytes`` attribute.

    Overhead: two ``tracemalloc`` snapshots per call; suitable for
    profiling but not production hot-paths.

    Usage::

        @memory_tracked
        def build_tree(data): ...
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()
            snap_before = tracemalloc.take_snapshot()
            result = func(*args, **kwargs)
            snap_after = tracemalloc.take_snapshot()
            stats = snap_after.compare_to(snap_before, "lineno")
            delta = sum(s.size_diff for s in stats)
            wrapper.last_memory_bytes = max(delta, 0)  # type: ignore[attr-defined]
            if not was_tracing:
                tracemalloc.stop()
            return result

        wrapper.last_memory_bytes = 0  # type: ignore[attr-defined]
        wrapper._memory_tracked_name = label  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def hotspot_analysis(
    report: ProfileReport,
    top_n: int = 5,
) -> List[Tuple[str, float, float]]:
    """Identify the most time-consuming stages.

    Parameters
    ----------
    report : ProfileReport
        A completed profiling report.
    top_n : int
        Number of stages to return.

    Returns
    -------
    list[tuple[str, float, float]]
        ``(stage_name, elapsed_seconds, fraction_of_total)`` sorted by
        elapsed time descending.  *fraction_of_total* is in [0, 1].
    """
    total = report.total_time or 1e-9
    items = sorted(report.stage_times.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, t, t / total) for name, t in items[:top_n]]


def memory_report(tracker: MemoryTracker) -> List[Tuple[str, int, float]]:
    """Memory breakdown by component.

    Parameters
    ----------
    tracker : MemoryTracker
        A tracker that has recorded at least one snapshot.

    Returns
    -------
    list[tuple[str, int, float]]
        ``(component, bytes, fraction_of_total)`` sorted by bytes
        descending.
    """
    total = tracker.total or 1
    items = sorted(tracker.components.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, mem, mem / total) for name, mem in items]


def format_profile(report: ProfileReport) -> str:
    """Format a :class:`ProfileReport` as a human-readable string.

    Returns
    -------
    str
        Multi-line report with stage times, percentages, and optional
        memory breakdown.
    """
    total = report.total_time or 1e-9
    lines = ["Pipeline Profile Report", "=" * 60]

    # Time breakdown
    lines.append("  Stage Timings:")
    lines.append("  " + "-" * 56)
    for stage, elapsed in sorted(report.stage_times.items(), key=lambda kv: -kv[1]):
        pct = (elapsed / total) * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if elapsed < 1.0:
            t_str = f"{elapsed * 1000:8.1f} ms"
        else:
            t_str = f"{elapsed:8.3f} s "
        lines.append(f"    {stage:24s} {t_str}  {pct:5.1f}% {bar}")

    lines.append("  " + "-" * 56)
    if report.total_time < 1.0:
        lines.append(f"    {'Total':24s} {report.total_time * 1000:8.1f} ms")
    else:
        lines.append(f"    {'Total':24s} {report.total_time:8.3f} s")

    # Memory breakdown (if available)
    if report.stage_memory:
        lines.append("")
        lines.append("  Memory Breakdown:")
        lines.append("  " + "-" * 56)
        mem_total = sum(report.stage_memory.values()) or 1
        for stage, mem in sorted(report.stage_memory.items(), key=lambda kv: -kv[1]):
            pct = (mem / mem_total) * 100
            if mem < 1024:
                m_str = f"{mem:8d} B "
            elif mem < 1024 * 1024:
                m_str = f"{mem / 1024:8.1f} KB"
            else:
                m_str = f"{mem / (1024 * 1024):8.1f} MB"
            lines.append(f"    {stage:24s} {m_str}  {pct:5.1f}%")

    return "\n".join(lines)


def compare_profiles(before: ProfileReport, after: ProfileReport) -> str:
    """Compare two profile runs and highlight regressions / improvements.

    Parameters
    ----------
    before : ProfileReport
        Baseline profile.
    after : ProfileReport
        New profile to compare against the baseline.

    Returns
    -------
    str
        Human-readable comparison showing time deltas and speedup
        ratios per stage.
    """
    lines = ["Profile Comparison (before → after)", "=" * 60]

    all_stages = sorted(set(before.stage_times) | set(after.stage_times))
    for stage in all_stages:
        t_before = before.stage_times.get(stage, 0.0)
        t_after = after.stage_times.get(stage, 0.0)
        if t_before > 1e-9:
            ratio = t_after / t_before
            if ratio < 1.0:
                indicator = f"  ↓ {(1 - ratio) * 100:.0f}% faster"
            elif ratio > 1.0:
                indicator = f"  ↑ {(ratio - 1) * 100:.0f}% slower"
            else:
                indicator = "  = unchanged"
        else:
            indicator = "  (new)"

        def _fmt(t: float) -> str:
            return f"{t * 1000:.1f}ms" if t < 1.0 else f"{t:.3f}s"

        lines.append(f"  {stage:24s} {_fmt(t_before):>10s} → {_fmt(t_after):>10s}{indicator}")

    # Total
    lines.append("-" * 60)
    tb = before.total_time
    ta = after.total_time

    def _fmt(t: float) -> str:
        return f"{t * 1000:.1f}ms" if t < 1.0 else f"{t:.3f}s"

    if tb > 1e-9:
        overall = ta / tb
        if overall < 1.0:
            tag = f"  ↓ {(1 - overall) * 100:.0f}% faster"
        elif overall > 1.0:
            tag = f"  ↑ {(overall - 1) * 100:.0f}% slower"
        else:
            tag = "  = unchanged"
    else:
        tag = ""
    lines.append(f"  {'Total':24s} {_fmt(tb):>10s} → {_fmt(ta):>10s}{tag}")

    return "\n".join(lines)
