"""
usability_oracle.utils.parallel — Parallel computation utilities.

Provides thread-pool and process-pool abstractions for Monte Carlo
sampling, independent MDP solving, and batch processing with progress
tracking and failure recovery.

Key components
--------------
- :class:`ThreadPool` — thread pool for I/O-bound or GIL-releasing work.
- :class:`ProcessPool` — process pool for CPU-bound MDP solving.
- :func:`chunked_parallel_map` — chunked parallel map with progress.
- :class:`ProgressTracker` — track progress of parallel operations.
- :func:`resource_aware_workers` — auto-detect worker count from CPU/memory.
- :func:`async_batch_process` — async batch processing.

Performance characteristics
---------------------------
- Chunk-based scheduling amortises pool overhead: O(N/chunk_size) tasks.
- Resource-aware parallelism prevents memory pressure on large MDPs.
- Worker failure recovery via per-item retry with exponential backoff.

References
----------
Dean, J. & Ghemawat, S. (2004). MapReduce: Simplified data processing
    on large clusters. *OSDI*.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Resource-aware worker count
# ---------------------------------------------------------------------------


def resource_aware_workers(
    max_workers: Optional[int] = None,
    memory_per_worker_mb: float = 512.0,
) -> int:
    """Determine worker count respecting CPU and memory limits.

    Parameters
    ----------
    max_workers : int or None
        Upper bound.  ``None`` uses ``os.cpu_count()``.
    memory_per_worker_mb : float
        Estimated memory usage per worker in MB.

    Returns
    -------
    int
        Number of workers (≥ 1).
    """
    cpu_count = os.cpu_count() or 1
    n = max_workers if max_workers is not None else cpu_count

    # Heuristic: try to stay under 80% of available physical RAM
    try:
        import psutil

        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        mem_limit = max(1, int(avail_mb * 0.8 / max(memory_per_worker_mb, 1.0)))
        n = min(n, mem_limit)
    except ImportError:
        pass

    return max(1, min(n, cpu_count))


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class ProgressTracker:
    """Thread-safe progress tracker for parallel operations.

    Attributes
    ----------
    total : int
        Total number of items.
    completed : int
        Number of completed items.
    failed : int
        Number of failed items.
    start_time : float
        Wall-clock start time.
    """

    total: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self) -> None:
        """Record a successfully completed item."""
        with self._lock:
            self.completed += 1

    def record_failure(self) -> None:
        """Record a failed item."""
        with self._lock:
            self.failed += 1

    @property
    def progress(self) -> float:
        """Fraction completed in [0, 1]."""
        return (self.completed + self.failed) / self.total if self.total > 0 else 0.0

    @property
    def elapsed_s(self) -> float:
        """Elapsed seconds since start."""
        return time.monotonic() - self.start_time

    @property
    def eta_s(self) -> float:
        """Estimated seconds remaining."""
        done = self.completed + self.failed
        if done == 0:
            return float("inf")
        rate = done / self.elapsed_s
        remaining = self.total - done
        return remaining / rate if rate > 0 else float("inf")

    def summary(self) -> str:
        """Human-readable progress summary."""
        return (
            f"Progress: {self.completed}/{self.total} done, "
            f"{self.failed} failed, "
            f"{self.progress:.0%}, "
            f"elapsed={self.elapsed_s:.1f}s, ETA={self.eta_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Thread pool for Monte Carlo sampling
# ---------------------------------------------------------------------------


class ThreadPool(Generic[T, R]):
    """Thread pool wrapper for I/O-bound or GIL-releasing parallel work.

    Designed for Monte Carlo sampling where each sample involves
    numpy operations that release the GIL.

    Parameters
    ----------
    fn : callable
        Worker function ``fn(item) -> result``.
    n_workers : int or None
        Number of threads.
    """

    def __init__(
        self,
        fn: Callable[[T], R],
        n_workers: Optional[int] = None,
    ) -> None:
        self._fn = fn
        self._n_workers = n_workers or min(32, (os.cpu_count() or 1) + 4)

    def map(self, items: Sequence[T]) -> List[R]:
        """Apply *fn* to each item in parallel.

        Returns
        -------
        list of R
            Results in the same order as *items*.
        """
        if len(items) == 0:
            return []
        if len(items) == 1 or self._n_workers <= 1:
            return [self._fn(item) for item in items]

        with ThreadPoolExecutor(max_workers=self._n_workers) as pool:
            futures = {pool.submit(self._fn, item): i for i, item in enumerate(items)}
            results: List[Optional[R]] = [None] * len(items)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            return results  # type: ignore[return-value]

    def map_with_progress(
        self,
        items: Sequence[T],
    ) -> Tuple[List[R], ProgressTracker]:
        """Map with progress tracking.

        Returns
        -------
        results : list of R
        tracker : ProgressTracker
        """
        tracker = ProgressTracker(total=len(items))
        if len(items) == 0:
            return [], tracker

        results: List[Optional[R]] = [None] * len(items)

        def _wrapped(idx_item: Tuple[int, T]) -> Tuple[int, R]:
            idx, item = idx_item
            result = self._fn(item)
            tracker.record_success()
            return idx, result

        indexed = list(enumerate(items))
        with ThreadPoolExecutor(max_workers=self._n_workers) as pool:
            for idx, result in pool.map(_wrapped, indexed):
                results[idx] = result

        return results, tracker  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Process pool for independent MDP solving
# ---------------------------------------------------------------------------


class ProcessPool(Generic[T, R]):
    """Process pool for CPU-bound parallel work.

    Uses the ``spawn`` multiprocessing context for platform safety.

    Parameters
    ----------
    fn : callable
        Worker function (must be picklable).
    n_workers : int or None
        Number of processes.
    """

    def __init__(
        self,
        fn: Callable[[T], R],
        n_workers: Optional[int] = None,
    ) -> None:
        self._fn = fn
        self._n_workers = n_workers or resource_aware_workers()

    def map(self, items: Sequence[T]) -> List[R]:
        """Apply *fn* to each item in separate processes.

        Returns
        -------
        list of R
            Results in the same order as *items*.
        """
        if len(items) == 0:
            return []
        if len(items) == 1 or self._n_workers <= 1:
            return [self._fn(item) for item in items]

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self._n_workers, mp_context=ctx
        ) as pool:
            return list(pool.map(self._fn, items))


# ---------------------------------------------------------------------------
# Chunked parallel map
# ---------------------------------------------------------------------------


def chunked_parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    chunk_size: int = 100,
    n_workers: Optional[int] = None,
    use_threads: bool = True,
) -> List[R]:
    """Map *fn* over *items* in chunks, using thread or process pool.

    Amortises pool scheduling overhead by batching items.

    Parameters
    ----------
    fn : callable
        Worker function.
    items : sequence
        Input items.
    chunk_size : int
        Items per chunk.
    n_workers : int or None
        Number of workers.
    use_threads : bool
        If ``True``, use threads; otherwise processes.

    Returns
    -------
    list of R
        Results preserving input order.

    Complexity
    ----------
    O(N/chunk_size) pool submissions + O(N · f / n_workers) work time.
    """
    if len(items) == 0:
        return []

    chunks: List[List[T]] = []
    for i in range(0, len(items), chunk_size):
        chunks.append(list(items[i : i + chunk_size]))

    def _process_chunk(chunk: List[T]) -> List[R]:
        return [fn(item) for item in chunk]

    n = n_workers or resource_aware_workers()

    if n <= 1 or len(chunks) <= 1:
        results: List[R] = []
        for chunk in chunks:
            results.extend(_process_chunk(chunk))
        return results

    if use_threads:
        PoolClass = ThreadPoolExecutor
        pool_kwargs: Dict[str, Any] = {"max_workers": n}
    else:
        ctx = multiprocessing.get_context("spawn")
        PoolClass = ProcessPoolExecutor  # type: ignore[assignment]
        pool_kwargs = {"max_workers": n, "mp_context": ctx}

    results = []
    with PoolClass(**pool_kwargs) as pool:
        future_to_idx = {
            pool.submit(_process_chunk, chunk): i
            for i, chunk in enumerate(chunks)
        }
        chunk_results: List[Optional[List[R]]] = [None] * len(chunks)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk_results[idx] = future.result()

    for cr in chunk_results:
        if cr is not None:
            results.extend(cr)

    return results


# ---------------------------------------------------------------------------
# Worker failure recovery
# ---------------------------------------------------------------------------


def retry_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_retries: int = 3,
    backoff_base: float = 0.1,
    n_workers: Optional[int] = None,
) -> Tuple[List[Optional[R]], List[int]]:
    """Map with per-item retry and exponential backoff.

    Parameters
    ----------
    fn : callable
        Worker function.
    items : sequence
        Input items.
    max_retries : int
        Maximum attempts per item.
    backoff_base : float
        Base delay in seconds (doubles each retry).
    n_workers : int or None
        Thread pool size.

    Returns
    -------
    results : list of R or None
        ``None`` for permanently failed items.
    failed_indices : list of int
        Indices of items that failed after all retries.
    """
    results: List[Optional[R]] = [None] * len(items)
    failed: List[int] = []

    def _attempt(idx: int) -> Tuple[int, Optional[R], bool]:
        item = items[idx]
        for attempt in range(max_retries):
            try:
                return idx, fn(item), True
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(backoff_base * (2 ** attempt))
        return idx, None, False

    n = n_workers or resource_aware_workers()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_attempt, i) for i in range(len(items))]
        for future in as_completed(futures):
            idx, result, success = future.result()
            results[idx] = result
            if not success:
                failed.append(idx)

    return results, sorted(failed)


# ---------------------------------------------------------------------------
# Async batch processing
# ---------------------------------------------------------------------------


@dataclass
class BatchResult(Generic[R]):
    """Result of an async batch operation.

    Attributes
    ----------
    results : list of R or None
        Per-item results.
    tracker : ProgressTracker
        Progress information.
    failed_indices : list of int
        Indices of failed items.
    elapsed_s : float
        Total elapsed time.
    """

    results: List[Optional[R]] = field(default_factory=list)
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    failed_indices: List[int] = field(default_factory=list)
    elapsed_s: float = 0.0


def async_batch_process(
    fn: Callable[[T], R],
    items: Sequence[T],
    n_workers: Optional[int] = None,
    chunk_size: int = 50,
    max_retries: int = 2,
) -> BatchResult[R]:
    """Process a batch of items with progress, chunking, and retry.

    Combines :func:`chunked_parallel_map` with :func:`retry_map`
    for robust batch processing.

    Parameters
    ----------
    fn : callable
        Worker function.
    items : sequence
        Input items.
    n_workers : int or None
        Number of workers.
    chunk_size : int
        Items per chunk.
    max_retries : int
        Retries per failed item.

    Returns
    -------
    BatchResult
        Aggregated results with tracking info.
    """
    t0 = time.perf_counter()
    tracker = ProgressTracker(total=len(items))

    if len(items) == 0:
        return BatchResult(
            results=[], tracker=tracker, elapsed_s=time.perf_counter() - t0
        )

    n = n_workers or resource_aware_workers()

    # First pass: chunked parallel
    results: List[Optional[R]] = [None] * len(items)
    failed_indices: List[int] = []

    def _safe_fn(idx_item: Tuple[int, T]) -> Tuple[int, Optional[R], bool]:
        idx, item = idx_item
        try:
            result = fn(item)
            tracker.record_success()
            return idx, result, True
        except Exception:
            tracker.record_failure()
            return idx, None, False

    indexed = list(enumerate(items))
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_safe_fn, pair) for pair in indexed]
        for future in as_completed(futures):
            idx, result, success = future.result()
            results[idx] = result
            if not success:
                failed_indices.append(idx)

    # Retry failed items
    if failed_indices and max_retries > 0:
        retry_items = [items[i] for i in failed_indices]
        retry_results, still_failed_rel = retry_map(
            fn, retry_items, max_retries=max_retries, n_workers=n
        )
        still_failed = [failed_indices[i] for i in still_failed_rel]
        for i, idx in enumerate(failed_indices):
            if retry_results[i] is not None:
                results[idx] = retry_results[i]
        failed_indices = still_failed

    return BatchResult(
        results=results,
        tracker=tracker,
        failed_indices=failed_indices,
        elapsed_s=time.perf_counter() - t0,
    )


__all__ = [
    "resource_aware_workers",
    "ProgressTracker",
    "ThreadPool",
    "ProcessPool",
    "chunked_parallel_map",
    "retry_map",
    "BatchResult",
    "async_batch_process",
]
