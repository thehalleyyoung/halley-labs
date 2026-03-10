"""
Parallel execution support for the CausalCert pipeline.

Provides a thin abstraction over ``concurrent.futures`` for parallelising
CI test batches, fragility scoring, and estimation across folds.

Features:
- Thread pool for I/O-bound tasks
- Process pool for CPU-bound tasks (CI tests, fragility scoring)
- Async-friendly execution
- Resource management (memory limits, task prioritisation)
- Chunked map for large workloads
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task priority wrapper
# ---------------------------------------------------------------------------


@dataclass(order=True, slots=True)
class PrioritisedTask:
    """A task with an associated priority (lower = higher priority)."""

    priority: int
    task_id: int = field(compare=False)
    fn: Any = field(compare=False, repr=False)
    args: tuple = field(compare=False, repr=False, default=())
    kwargs: dict = field(compare=False, repr=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Lightweight resource-usage tracker.

    Parameters
    ----------
    max_memory_mb : float
        Maximum RSS memory allowed.  ``0`` disables checking.
    """

    def __init__(self, max_memory_mb: float = 0.0) -> None:
        self.max_memory_mb = max_memory_mb

    def current_rss_mb(self) -> float:
        """Return current RSS in megabytes (best effort)."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)
            return usage.ru_maxrss / 1024
        except Exception:
            return 0.0

    def check(self) -> bool:
        """Return ``True`` if within limits."""
        if self.max_memory_mb <= 0:
            return True
        return self.current_rss_mb() < self.max_memory_mb


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------


class ParallelExecutor:
    """Configurable parallel executor.

    Parameters
    ----------
    n_jobs : int
        Number of workers.  ``-1`` uses all available CPUs.
        ``1`` runs sequentially (no multiprocessing overhead).
    backend : str
        ``"process"`` for CPU-bound work, ``"thread"`` for I/O-bound.
    max_memory_mb : float
        Memory limit for resource monitoring.
    """

    def __init__(
        self,
        n_jobs: int = 1,
        backend: str = "process",
        max_memory_mb: float = 0.0,
    ) -> None:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        self.n_jobs = max(1, n_jobs)
        self.backend = backend
        self._monitor = ResourceMonitor(max_memory_mb)
        self._active_pool: ProcessPoolExecutor | ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Basic map
    # ------------------------------------------------------------------

    def map(
        self,
        fn: Callable[..., R],
        items: Sequence[T],
        **kwargs: Any,
    ) -> list[R]:
        """Apply *fn* to each item in parallel.

        Parameters
        ----------
        fn : Callable
            Function to apply.
        items : Sequence
            Items to process.

        Returns
        -------
        list
            Results in input order.
        """
        if not items:
            return []

        if self.n_jobs == 1:
            return [fn(item, **kwargs) for item in items]

        Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=self.n_jobs) as pool:
            futures: list[Future[R]] = [pool.submit(fn, item, **kwargs) for item in items]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------
    # Chunked map (for large workloads)
    # ------------------------------------------------------------------

    def map_chunked(
        self,
        fn: Callable[..., R],
        items: Sequence[T],
        chunk_size: int = 50,
        **kwargs: Any,
    ) -> list[R]:
        """Apply *fn* to items in chunks to manage memory.

        Parameters
        ----------
        fn : Callable
            Function to apply.
        items : Sequence
            Items to process.
        chunk_size : int
            Number of items per chunk.

        Returns
        -------
        list
            Results in input order.
        """
        results: list[R] = []
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            chunk_results = self.map(fn, chunk, **kwargs)
            results.extend(chunk_results)
            if not self._monitor.check():
                logger.warning(
                    "Memory limit approaching (%.0f MB), processing sequentially",
                    self._monitor.current_rss_mb(),
                )
                for item in items[start + chunk_size :]:
                    results.append(fn(item, **kwargs))
                break
        return results

    # ------------------------------------------------------------------
    # Unordered map (results as they complete)
    # ------------------------------------------------------------------

    def map_unordered(
        self,
        fn: Callable[..., R],
        items: Sequence[T],
        **kwargs: Any,
    ) -> Iterator[R]:
        """Apply *fn* in parallel and yield results as they complete.

        Parameters
        ----------
        fn : Callable
        items : Sequence

        Yields
        ------
        R
        """
        if not items:
            return

        if self.n_jobs == 1:
            for item in items:
                yield fn(item, **kwargs)
            return

        Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=self.n_jobs) as pool:
            future_to_idx = {
                pool.submit(fn, item, **kwargs): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                yield future.result()

    # ------------------------------------------------------------------
    # Prioritised execution
    # ------------------------------------------------------------------

    def map_prioritised(
        self,
        tasks: Sequence[PrioritisedTask],
    ) -> list[Any]:
        """Execute tasks in priority order.

        Lower priority values are executed first.  Results are returned
        in the original task order (by ``task_id``).

        Parameters
        ----------
        tasks : Sequence[PrioritisedTask]

        Returns
        -------
        list[Any]
        """
        if not tasks:
            return []

        sorted_tasks = sorted(tasks)
        id_to_result: dict[int, Any] = {}

        if self.n_jobs == 1:
            for t in sorted_tasks:
                id_to_result[t.task_id] = t.fn(*t.args, **t.kwargs)
        else:
            Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
            with Executor(max_workers=self.n_jobs) as pool:
                future_to_id: dict[Future, int] = {}
                for t in sorted_tasks:
                    fut = pool.submit(t.fn, *t.args, **t.kwargs)
                    future_to_id[fut] = t.task_id
                for fut in as_completed(future_to_id):
                    id_to_result[future_to_id[fut]] = fut.result()

        original_ids = [t.task_id for t in tasks]
        return [id_to_result[tid] for tid in original_ids]

    # ------------------------------------------------------------------
    # Batched star-map (multi-arg)
    # ------------------------------------------------------------------

    def starmap(
        self,
        fn: Callable[..., R],
        args_list: Sequence[tuple],
    ) -> list[R]:
        """Apply *fn* to each ``*args`` tuple in parallel.

        Parameters
        ----------
        fn : Callable
        args_list : Sequence[tuple]

        Returns
        -------
        list
        """
        if not args_list:
            return []

        if self.n_jobs == 1:
            return [fn(*args) for args in args_list]

        Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=self.n_jobs) as pool:
            futures = [pool.submit(fn, *args) for args in args_list]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------
    # Timed execution
    # ------------------------------------------------------------------

    def map_timed(
        self,
        fn: Callable[..., R],
        items: Sequence[T],
        timeout_s: float = 60.0,
        **kwargs: Any,
    ) -> list[R | None]:
        """Apply *fn* with per-item timeout.

        Items that exceed the timeout return ``None``.

        Parameters
        ----------
        fn : Callable
        items : Sequence
        timeout_s : float
            Per-item timeout in seconds.

        Returns
        -------
        list[R | None]
        """
        if not items:
            return []

        if self.n_jobs == 1:
            results: list[R | None] = []
            for item in items:
                t0 = time.monotonic()
                try:
                    results.append(fn(item, **kwargs))
                except Exception:
                    results.append(None)
            return results

        Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=self.n_jobs) as pool:
            futures = [pool.submit(fn, item, **kwargs) for item in items]
            out: list[R | None] = []
            for f in futures:
                try:
                    out.append(f.result(timeout=timeout_s))
                except Exception:
                    out.append(None)
            return out

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down any active pool."""
        if self._active_pool is not None:
            self._active_pool.shutdown(wait=False)
            self._active_pool = None

    def __repr__(self) -> str:
        return f"ParallelExecutor(n_jobs={self.n_jobs}, backend={self.backend!r})"
