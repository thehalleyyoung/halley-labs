"""
usability_oracle.pipeline.parallel — Parallel execution utilities.

Provides :class:`ParallelExecutor` for running independent pipeline
stages or analysis tasks concurrently using :mod:`concurrent.futures`.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Parallel task result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of a single parallel task.

    Attributes
    ----------
    index : int
        Original index in the task list.
    result : Any
        Task return value (None on failure).
    error : str | None
        Error message if the task failed.
    elapsed : float
        Execution time in seconds.
    """

    index: int
    result: Any = None
    error: Optional[str] = None
    elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Execute tasks in parallel using process or thread pools.

    Parameters
    ----------
    max_workers : int
        Maximum number of parallel workers.
    use_threads : bool
        If True, use ThreadPoolExecutor (for I/O-bound work).
        If False, use ProcessPoolExecutor (for CPU-bound work).
    default_timeout : float | None
        Default per-task timeout in seconds.  None = no timeout.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_threads: bool = False,
        default_timeout: float | None = None,
    ) -> None:
        self.max_workers = max(1, max_workers)
        self.use_threads = use_threads
        self.default_timeout = default_timeout

    # ── Pool factory ------------------------------------------------------

    def _make_pool(
        self, max_workers: int | None = None
    ) -> ProcessPoolExecutor | ThreadPoolExecutor:
        """Create the appropriate executor pool."""
        workers = max_workers or self.max_workers
        if self.use_threads:
            return ThreadPoolExecutor(max_workers=workers)
        return ProcessPoolExecutor(max_workers=workers)

    # ── execute_parallel --------------------------------------------------

    def execute_parallel(
        self,
        tasks: list[Callable[[], Any]],
        max_workers: int | None = None,
        timeout: float | None = None,
    ) -> list[Any]:
        """Execute a list of callables in parallel.

        Parameters
        ----------
        tasks : list of callable
            Zero-argument callables to execute.
        max_workers : int, optional
            Override the default max_workers.
        timeout : float, optional
            Per-task timeout in seconds.

        Returns
        -------
        list[Any]
            Results in the same order as *tasks*.
            Failed tasks produce None.
        """
        if not tasks:
            return []

        workers = max_workers or self.max_workers
        effective_timeout = timeout or self.default_timeout
        results: list[Any] = [None] * len(tasks)

        pool = self._make_pool(workers)
        try:
            future_to_idx: dict[Future[Any], int] = {}
            for i, task in enumerate(tasks):
                future = pool.submit(task)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx, timeout=effective_timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=0)
                except FuturesTimeoutError:
                    logger.warning("Task %d timed out", idx)
                except Exception as exc:
                    logger.warning("Task %d failed: %s", idx, exc)

        except FuturesTimeoutError:
            logger.warning(
                "Parallel execution timed out (%.1fs)", effective_timeout
            )
        finally:
            pool.shutdown(wait=False)

        return results

    # ── execute_map -------------------------------------------------------

    def execute_map(
        self,
        fn: Callable[..., Any],
        items: list[Any],
        max_workers: int | None = None,
        timeout: float | None = None,
    ) -> list[Any]:
        """Apply *fn* to each item in parallel.

        Parameters
        ----------
        fn : callable
            Function taking a single argument.
        items : list
            Items to process.
        max_workers : int, optional
        timeout : float, optional

        Returns
        -------
        list
            Results in the same order as *items*.
        """
        if not items:
            return []

        workers = max_workers or self.max_workers
        effective_timeout = timeout or self.default_timeout
        results: list[Any] = [None] * len(items)

        pool = self._make_pool(workers)
        try:
            future_to_idx: dict[Future[Any], int] = {}
            for i, item in enumerate(items):
                future = pool.submit(fn, item)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx, timeout=effective_timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=0)
                except FuturesTimeoutError:
                    logger.warning("Map task %d timed out", idx)
                except Exception as exc:
                    logger.warning("Map task %d failed: %s", idx, exc)

        except FuturesTimeoutError:
            logger.warning(
                "Map execution timed out (%.1fs)", effective_timeout
            )
        finally:
            pool.shutdown(wait=False)

        return results

    # ── execute_with_results ----------------------------------------------

    def execute_with_results(
        self,
        tasks: list[Callable[[], Any]],
        max_workers: int | None = None,
        timeout: float | None = None,
    ) -> list[TaskResult]:
        """Execute tasks and return detailed results including timing.

        Returns
        -------
        list[TaskResult]
            Detailed results in the same order as *tasks*.
        """
        if not tasks:
            return []

        workers = max_workers or self.max_workers
        effective_timeout = timeout or self.default_timeout
        results: list[TaskResult] = [
            TaskResult(index=i) for i in range(len(tasks))
        ]

        pool = self._make_pool(workers)
        try:
            future_to_idx: dict[Future[Any], int] = {}
            submit_times: dict[int, float] = {}

            for i, task in enumerate(tasks):
                submit_times[i] = time.monotonic()
                future = pool.submit(task)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx, timeout=effective_timeout):
                idx = future_to_idx[future]
                elapsed = time.monotonic() - submit_times[idx]

                try:
                    value = future.result(timeout=0)
                    results[idx] = TaskResult(
                        index=idx, result=value, elapsed=elapsed,
                    )
                except FuturesTimeoutError:
                    results[idx] = TaskResult(
                        index=idx, error="timeout", elapsed=elapsed,
                    )
                except Exception as exc:
                    results[idx] = TaskResult(
                        index=idx, error=str(exc), elapsed=elapsed,
                    )

        except FuturesTimeoutError:
            for i in range(len(tasks)):
                if results[i].result is None and results[i].error is None:
                    results[i].error = "global_timeout"
        finally:
            pool.shutdown(wait=False)

        return results

    # ── with_timeout ------------------------------------------------------

    @staticmethod
    def _with_timeout(
        fn: Callable[[], Any],
        timeout: float,
    ) -> Any:
        """Execute a function with a timeout using a thread.

        Parameters
        ----------
        fn : callable
        timeout : float
            Timeout in seconds.

        Returns
        -------
        Any
            Function result.

        Raises
        ------
        TimeoutError
            If the function does not complete within *timeout*.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Function did not complete within {timeout}s"
                ) from None

    # ── batched execution ------------------------------------------------

    def execute_batched(
        self,
        items: list[Any],
        fn: Callable[[Any], Any],
        batch_size: int = 10,
        timeout_per_batch: float | None = None,
    ) -> list["ParallelResult"]:
        """Execute items in batches with optional per-batch timeout.

        Items are split into chunks of *batch_size*, and each chunk is
        processed in parallel.  This reduces peak memory when the item
        list is very large.
        """
        all_results: list["ParallelResult"] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_results = self.execute_map(
                fn, batch, timeout=timeout_per_batch or self._timeout
            )
            all_results.extend(batch_results)
        return all_results

    # ── adaptive pool sizing ---------------------------------------------

    @staticmethod
    def suggest_workers(
        n_items: int,
        item_estimated_seconds: float = 0.1,
        max_workers: int | None = None,
    ) -> int:
        """Heuristically suggest the number of workers.

        Uses a simple model:
          - if items are fast (< 0.01s), use fewer workers (IO overhead dominates)
          - if items are slow, saturate available cores
        """
        import os

        cpu_count = os.cpu_count() or 4
        cap = max_workers or cpu_count * 2

        if item_estimated_seconds < 0.01:
            suggested = min(n_items, max(1, cpu_count // 2))
        elif item_estimated_seconds < 1.0:
            suggested = min(n_items, cpu_count)
        else:
            suggested = min(n_items, cpu_count * 2)

        return max(1, min(suggested, cap))

    # ── progress-reporting map -------------------------------------------

    def execute_map_with_progress(
        self,
        fn: Callable[[Any], Any],
        items: list[Any],
        callback: Callable[[int, int, float], None] | None = None,
        timeout: float | None = None,
    ) -> list["ParallelResult"]:
        """Like execute_map, but invokes *callback(completed, total, elapsed)* after each item."""
        import time as _time

        n = len(items)
        results = [ParallelResult(result=None, error=None) for _ in range(n)]
        t_start = _time.time()
        completed = 0

        pool = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            futures = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
            effective_timeout = timeout or self._timeout
            for future in as_completed(futures, timeout=effective_timeout):
                idx = futures[future]
                try:
                    results[idx] = ParallelResult(result=future.result(timeout=0), error=None)
                except Exception as exc:
                    results[idx] = ParallelResult(result=None, error=str(exc))
                completed += 1
                if callback:
                    callback(completed, n, _time.time() - t_start)
        except FuturesTimeoutError:
            for i in range(n):
                if results[i].result is None and results[i].error is None:
                    results[i].error = "global_timeout"
        finally:
            pool.shutdown(wait=False)

        return results

    # ── gather with ordering guarantee -----------------------------------

    def ordered_map(
        self,
        fn: Callable[[Any], Any],
        items: list[Any],
    ) -> list[Any]:
        """Execute *fn* on each item in parallel, returning results in input order.

        Unlike execute_map, returns raw values (not ParallelResult wrappers).
        Raises the first exception encountered.
        """
        pool = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            futures = [pool.submit(fn, item) for item in items]
            return [f.result(timeout=self._timeout) for f in futures]
        finally:
            pool.shutdown(wait=False)
