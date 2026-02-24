"""Structured logging for the finite-width phase diagram system.

Provides component loggers, timing decorators, progress tracking,
memory monitoring, and computation budget tracking.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_LOGGERS: Dict[str, logging.Logger] = {}

_DEFAULT_FORMAT = (
    "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"
)
_SHORT_FORMAT = "%(name)-20s | %(message)s"


def get_logger(
    name: str = "fwpd",
    level: Optional[int] = None,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Return a named logger, creating it on first call.

    Parameters
    ----------
    name : str
        Logger name (e.g. ``"fwpd.calibration"``).
    level : int, optional
        Logging level. Defaults to ``INFO`` or ``FWPD_LOG_LEVEL`` env var.
    fmt : str, optional
        Log format string.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    if level is None:
        env_level = os.environ.get("FWPD_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger


def set_global_level(level: Union[int, str]) -> None:
    """Set the logging level for all ``fwpd.*`` loggers."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    for logger in _LOGGERS.values():
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def timer(fn: Optional[Callable] = None, *, logger_name: str = "fwpd.timer"):
    """Decorator that logs execution time of a function.

    Can be used with or without arguments::

        @timer
        def foo(): ...

        @timer(logger_name="fwpd.kernel")
        def bar(): ...
    """
    def decorator(func: Callable) -> Callable:
        log = get_logger(logger_name)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            log.info("%s completed in %.3f s", func.__qualname__, elapsed)
            return result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


@contextmanager
def timed_block(label: str, logger_name: str = "fwpd.timer"):
    """Context manager that logs elapsed time for a block.

    Usage::

        with timed_block("NTK computation"):
            compute_ntk(...)
    """
    log = get_logger(logger_name)
    log.info("[START] %s", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.info("[DONE]  %s  (%.3f s)", label, elapsed)


class TimingAccumulator:
    """Accumulate timing data across multiple calls."""

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = {}

    def record(self, name: str, elapsed: float) -> None:
        self._timings.setdefault(name, []).append(elapsed)

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        yield
        self.record(name, time.perf_counter() - t0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        import numpy as np

        result: Dict[str, Dict[str, float]] = {}
        for name, times in self._timings.items():
            arr = np.array(times)
            result[name] = {
                "count": len(arr),
                "total": float(arr.sum()),
                "mean": float(arr.mean()),
                "std": float(arr.std()) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        return result

    def report(self, logger_name: str = "fwpd.timer") -> str:
        log = get_logger(logger_name)
        lines = ["Timing Summary", "-" * 60]
        for name, stats in self.summary().items():
            lines.append(
                f"  {name:30s}  calls={stats['count']:4.0f}  "
                f"total={stats['total']:8.3f}s  mean={stats['mean']:.3f}s"
            )
        report = "\n".join(lines)
        log.info("\n%s", report)
        return report


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Track and display progress for iterative computations.

    Parameters
    ----------
    total : int
        Total number of items.
    label : str
        Description of the task.
    log_interval : int
        Log every N items.
    logger_name : str
        Logger to use.
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        log_interval: int = 10,
        logger_name: str = "fwpd.progress",
    ) -> None:
        self.total = max(total, 1)
        self.label = label
        self.log_interval = max(log_interval, 1)
        self._log = get_logger(logger_name)
        self._count = 0
        self._t_start = time.perf_counter()
        self._t_last = self._t_start

    def update(self, n: int = 1, message: str = "") -> None:
        """Record completion of *n* items."""
        self._count = min(self._count + n, self.total)
        now = time.perf_counter()
        if self._count % self.log_interval == 0 or self._count == self.total:
            pct = 100.0 * self._count / self.total
            elapsed = now - self._t_start
            rate = self._count / max(elapsed, 1e-9)
            eta = (self.total - self._count) / max(rate, 1e-9)
            msg = (
                f"{self.label}: {self._count}/{self.total} "
                f"({pct:.0f}%) [{elapsed:.1f}s elapsed, ETA {eta:.1f}s]"
            )
            if message:
                msg += f" — {message}"
            self._log.info(msg)
        self._t_last = now

    @property
    def fraction(self) -> float:
        return self._count / self.total

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t_start

    def done(self) -> None:
        """Mark completion and log final time."""
        self._count = self.total
        self._log.info(
            "%s: complete (%d items in %.1f s)",
            self.label,
            self.total,
            self.elapsed,
        )


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Monitor process memory usage.

    Uses ``resource`` on Unix or ``psutil`` if available.
    """

    def __init__(self, logger_name: str = "fwpd.memory") -> None:
        self._log = get_logger(logger_name)
        self._snapshots: List[Dict[str, Any]] = []

    def snapshot(self, label: str = "") -> Dict[str, float]:
        """Take a memory snapshot and return usage in MB."""
        usage_mb = self._get_rss_mb()
        snap = {"label": label, "rss_mb": usage_mb, "time": time.perf_counter()}
        self._snapshots.append(snap)
        self._log.debug("Memory [%s]: %.1f MB", label or "snapshot", usage_mb)
        return {"rss_mb": usage_mb}

    def log_current(self, label: str = "") -> None:
        """Log current memory usage."""
        mb = self._get_rss_mb()
        self._log.info("Memory [%s]: %.1f MB RSS", label or "current", mb)

    def peak(self) -> float:
        """Return peak RSS in MB across all snapshots."""
        if not self._snapshots:
            return self._get_rss_mb()
        return max(s["rss_mb"] for s in self._snapshots)

    def report(self) -> str:
        """Return a summary of memory snapshots."""
        if not self._snapshots:
            return "No memory snapshots recorded."
        lines = ["Memory Report", "-" * 40]
        for s in self._snapshots:
            lines.append(f"  {s['label']:30s}  {s['rss_mb']:.1f} MB")
        lines.append(f"  {'Peak':30s}  {self.peak():.1f} MB")
        return "\n".join(lines)

    @staticmethod
    def _get_rss_mb() -> float:
        """Get current RSS in MB."""
        try:
            import resource
            # maxrss is in KB on Linux, bytes on macOS
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            rss = rusage.ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 * 1024)
            return rss / 1024
        except ImportError:
            pass
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


# ---------------------------------------------------------------------------
# Computation budget
# ---------------------------------------------------------------------------

@dataclass
class ComputationBudget:
    """Track computation budget (time, memory, FLOPs)."""

    max_time_s: float = 3600.0
    max_memory_mb: float = 8192.0
    warn_threshold: float = 0.8

    _start_time: float = field(default_factory=time.perf_counter, repr=False)
    _log: logging.Logger = field(
        default_factory=lambda: get_logger("fwpd.budget"), repr=False
    )

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.max_time_s - self.elapsed)

    @property
    def time_fraction(self) -> float:
        return self.elapsed / max(self.max_time_s, 1e-9)

    def check(self) -> bool:
        """Check budget. Returns True if within budget, warns if near limit."""
        frac = self.time_fraction
        if frac >= 1.0:
            self._log.warning("Time budget EXCEEDED (%.1f s / %.1f s)", self.elapsed, self.max_time_s)
            return False
        if frac >= self.warn_threshold:
            self._log.warning(
                "Time budget at %.0f%% (%.1f s / %.1f s)",
                frac * 100, self.elapsed, self.max_time_s,
            )
        return True

    def enforce(self) -> None:
        """Raise if budget exceeded."""
        if not self.check():
            raise RuntimeError(
                f"Computation budget exceeded: {self.elapsed:.1f}s / {self.max_time_s:.1f}s"
            )
