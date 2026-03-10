"""
usability_oracle.utils.timing — Timing utilities.

Provides the ``@timed`` decorator, a ``Timer`` context manager, and
a ``TimingReport`` data structure for recording per-stage execution times.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# TimingReport
# ---------------------------------------------------------------------------

@dataclass
class TimingReport:
    """Collects elapsed times for named stages."""

    stages: dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def record(self, name: str, elapsed: float) -> None:
        """Record *elapsed* seconds for stage *name*."""
        self.stages[name] = elapsed
        self.total = sum(self.stages.values())

    def __str__(self) -> str:
        return format_timing(self)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that measures wall-clock time.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed)

    Can also record into a :class:`TimingReport`::

        report = TimingReport()
        with Timer("parse", report=report):
            parse()
    """

    def __init__(
        self,
        name: str = "",
        report: Optional[TimingReport] = None,
    ) -> None:
        self.name = name
        self._report = report
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self._report is not None and self.name:
            self._report.record(self.name, self.elapsed)


# ---------------------------------------------------------------------------
# @timed decorator
# ---------------------------------------------------------------------------

def timed(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """Decorator that prints the execution time of a function.

    Usage::

        @timed
        def my_function():
            ...

        @timed(name="custom_label")
        def another():
            ...

    The decorated function gains a ``last_timing`` attribute with the most
    recent elapsed time in seconds.
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            wrapper.last_timing = elapsed  # type: ignore[attr-defined]
            return result

        wrapper.last_timing = 0.0  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        # Called without arguments: @timed
        return decorator(fn)
    # Called with arguments: @timed(name=...)
    return decorator


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_timing(report: TimingReport) -> str:
    """Format a :class:`TimingReport` as a human-readable string."""
    if not report.stages:
        return "No timing data recorded."
    total = report.total or 1e-9
    lines = ["Pipeline Timing Report", "-" * 50]
    for stage, elapsed in report.stages.items():
        pct = (elapsed / total) * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if elapsed < 1.0:
            t_str = f"{elapsed * 1000:8.1f} ms"
        else:
            t_str = f"{elapsed:8.3f} s "
        lines.append(f"  {stage:20s} {t_str}  {pct:5.1f}% {bar}")
    lines.append("-" * 50)
    if report.total < 1.0:
        lines.append(f"  {'Total':20s} {report.total * 1000:8.1f} ms")
    else:
        lines.append(f"  {'Total':20s} {report.total:8.3f} s")
    return "\n".join(lines)
