"""Structured logging, timing, memory tracking, and progress reporting.

Provides lightweight infrastructure for performance monitoring and
structured log output throughout the CPA engine.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import resource
from contextlib import contextmanager
from typing import Any, Dict, Optional, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def _configure_root() -> None:
    """One-time root logger configuration."""
    global _configured
    if _configured:
        return
    level_str = os.environ.get("CPA_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_str, logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root = logging.getLogger("cpa")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``cpa`` namespace.

    Parameters
    ----------
    name : str
        Dot-separated logger name (e.g. ``"core.scm"``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    _configure_root()
    return logging.getLogger(f"cpa.{name}")


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class TimingContext:
    """Context manager that records wall-clock and CPU time of a block.

    Parameters
    ----------
    label : str
        Human-readable label for the timed section.
    logger : logging.Logger, optional
        If provided, elapsed time is logged at DEBUG level on exit.

    Attributes
    ----------
    elapsed_wall : float
        Wall-clock seconds after exiting the block.
    elapsed_cpu : float
        CPU seconds (user + system) after exiting the block.

    Examples
    --------
    >>> with TimingContext("fit model") as t:
    ...     expensive_computation()
    >>> print(t.elapsed_wall)
    """

    def __init__(self, label: str, logger: Optional[logging.Logger] = None) -> None:
        self.label = label
        self.logger = logger
        self.elapsed_wall: float = 0.0
        self.elapsed_cpu: float = 0.0
        self._wall_start: float = 0.0
        self._cpu_start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_wall = time.perf_counter() - self._wall_start
        self.elapsed_cpu = time.process_time() - self._cpu_start
        if self.logger is not None:
            self.logger.debug(
                "%s completed in %.4fs wall / %.4fs cpu",
                self.label,
                self.elapsed_wall,
                self.elapsed_cpu,
            )

    def __repr__(self) -> str:
        return (
            f"TimingContext(label={self.label!r}, "
            f"wall={self.elapsed_wall:.4f}s, cpu={self.elapsed_cpu:.4f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize timing results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary with label, elapsed_wall, and elapsed_cpu.
        """
        return {
            "label": self.label,
            "elapsed_wall": self.elapsed_wall,
            "elapsed_cpu": self.elapsed_cpu,
        }


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Track peak resident set size around a computation.

    Uses ``resource.getrusage`` on Unix-like systems.  On macOS the value
    is in bytes; on Linux it is in kilobytes.  The tracker normalises to
    megabytes.

    Parameters
    ----------
    label : str
        Description of the tracked section.
    logger : logging.Logger, optional
        If provided, memory delta is logged at DEBUG level on exit.

    Attributes
    ----------
    delta_mb : float
        Change in max-RSS (megabytes) between enter and exit.
    """

    _SCALE: float = 1.0  # bytes on macOS

    def __init__(self, label: str, logger: Optional[logging.Logger] = None) -> None:
        self.label = label
        self.logger = logger
        self.delta_mb: float = 0.0
        self._start_rss: int = 0

    @staticmethod
    def _max_rss_bytes() -> int:
        """Return max RSS in bytes."""
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports bytes, Linux reports KB
        if sys.platform == "darwin":
            return ru.ru_maxrss
        return ru.ru_maxrss * 1024

    def __enter__(self) -> "MemoryTracker":
        self._start_rss = self._max_rss_bytes()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_rss = self._max_rss_bytes()
        self.delta_mb = (end_rss - self._start_rss) / (1024 * 1024)
        if self.logger is not None:
            self.logger.debug(
                "%s memory delta: %.2f MB (peak RSS)",
                self.label,
                self.delta_mb,
            )

    def __repr__(self) -> str:
        return f"MemoryTracker(label={self.label!r}, delta_mb={self.delta_mb:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {"label": self.label, "delta_mb": self.delta_mb}


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Minimal progress reporting for long-running iterations.

    Parameters
    ----------
    total : int
        Total number of items to process.
    label : str
        Description of the task.
    logger : logging.Logger, optional
        Logger instance. If ``None``, prints to stderr.
    report_every : int
        Report progress every *n* items (default 10).

    Examples
    --------
    >>> pr = ProgressReporter(100, "bootstrap")
    >>> for i in range(100):
    ...     do_work(i)
    ...     pr.update()
    >>> pr.finish()
    """

    def __init__(
        self,
        total: int,
        label: str = "progress",
        logger: Optional[logging.Logger] = None,
        report_every: int = 10,
    ) -> None:
        if total <= 0:
            raise ValueError(f"total must be > 0, got {total}")
        self.total = total
        self.label = label
        self.logger = logger
        self.report_every = max(1, report_every)
        self._count = 0
        self._start = time.perf_counter()

    @property
    def fraction(self) -> float:
        """Fraction of work completed in [0, 1]."""
        return self._count / self.total

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since creation."""
        return time.perf_counter() - self._start

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds remaining, or None if no progress yet."""
        if self._count == 0:
            return None
        rate = self._count / self.elapsed
        remaining = self.total - self._count
        return remaining / rate

    def update(self, n: int = 1) -> None:
        """Record *n* completed items and optionally report.

        Parameters
        ----------
        n : int
            Number of items just completed.
        """
        self._count += n
        if self._count % self.report_every == 0 or self._count == self.total:
            self._report()

    def _report(self) -> None:
        pct = 100.0 * self.fraction
        eta = self.eta_seconds
        eta_str = f"{eta:.1f}s" if eta is not None else "?"
        msg = f"[{self.label}] {self._count}/{self.total} ({pct:.1f}%) ETA {eta_str}"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    def finish(self) -> None:
        """Mark the task as complete and report final timing."""
        self._count = self.total
        msg = f"[{self.label}] done — {self.elapsed:.2f}s total"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    def __repr__(self) -> str:
        return (
            f"ProgressReporter(label={self.label!r}, "
            f"{self._count}/{self.total})"
        )
