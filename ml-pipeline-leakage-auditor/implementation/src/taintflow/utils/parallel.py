"""
taintflow.utils.parallel – Parallel execution utilities.

Provides thread/process-pool execution, work-stealing scheduling,
memory-aware parallelism, and utilities for analysing independent DAG
branches concurrently.  Built entirely on :mod:`concurrent.futures`.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import sys
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
    Deque,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "ParallelExecutor",
    "parallel_map",
    "parallel_branch_analysis",
    "WorkStealingQueue",
    "TaskResult",
    "chunked_iter",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


@dataclass
class TaskResult(Generic[R]):
    """Wrapper for the result of a parallel task with timing metadata."""

    value: Optional[R] = None
    error: Optional[Exception] = None
    elapsed_seconds: float = 0.0
    worker_id: Optional[str] = None
    task_id: Optional[str] = None
    succeeded: bool = True

    def unwrap(self) -> R:
        """Return the value or re-raise the stored exception."""
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Chunked iteration
# ---------------------------------------------------------------------------


def chunked_iter(
    iterable: Iterable[T],
    chunk_size: int,
) -> Iterator[List[T]]:
    """Yield successive chunks of *chunk_size* from *iterable*.

    >>> list(chunked_iter([1,2,3,4,5], 2))
    [[1, 2], [3, 4], [5]]
    """
    chunk: List[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _available_memory_mb() -> float:
    """Best-effort estimate of available system memory in MB."""
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        elif sys.platform == "darwin":
            # Use os.sysconf as a rough proxy (total memory)
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 * 1024) * 0.5  # assume ~50% free
    except Exception:
        pass
    return 4096.0  # default fallback: 4 GB


def memory_aware_workers(
    per_worker_mb: float = 512.0,
    max_workers: Optional[int] = None,
) -> int:
    """Choose worker count based on available memory.

    Parameters
    ----------
    per_worker_mb : float
        Estimated memory per worker in MB.
    max_workers : int or None
        Hard ceiling; defaults to CPU count.

    Returns
    -------
    int
        Recommended number of workers (≥ 1).
    """
    avail = _available_memory_mb()
    mem_based = max(1, int(avail / max(per_worker_mb, 1.0)))
    cpu_based = max_workers or os.cpu_count() or 4
    return min(mem_based, cpu_based)


# ---------------------------------------------------------------------------
# WorkStealingQueue
# ---------------------------------------------------------------------------


class WorkStealingQueue(Generic[T]):
    """Simple lock-based work-stealing deque for load balancing.

    Each worker has a local deque.  When a worker's deque is empty it
    *steals* from the back of another worker's deque.
    """

    def __init__(self, n_workers: int) -> None:
        self._n_workers = max(n_workers, 1)
        self._deques: List[Deque[T]] = [collections.deque() for _ in range(self._n_workers)]
        self._locks: List[threading.Lock] = [threading.Lock() for _ in range(self._n_workers)]
        self._total = 0
        self._total_lock = threading.Lock()

    def push(self, worker_id: int, item: T) -> None:
        """Push *item* to the front of worker *worker_id*'s deque."""
        wid = worker_id % self._n_workers
        with self._locks[wid]:
            self._deques[wid].appendleft(item)
        with self._total_lock:
            self._total += 1

    def push_bulk(self, items: Iterable[T]) -> None:
        """Distribute *items* round-robin across workers."""
        for i, item in enumerate(items):
            self.push(i % self._n_workers, item)

    def pop(self, worker_id: int) -> Optional[T]:
        """Pop from the front of *worker_id*'s deque, or steal from others."""
        wid = worker_id % self._n_workers
        # Try own deque first
        with self._locks[wid]:
            if self._deques[wid]:
                item = self._deques[wid].popleft()
                with self._total_lock:
                    self._total -= 1
                return item
        # Steal from others (from the back)
        for offset in range(1, self._n_workers):
            victim = (wid + offset) % self._n_workers
            with self._locks[victim]:
                if self._deques[victim]:
                    item = self._deques[victim].pop()
                    with self._total_lock:
                        self._total -= 1
                    return item
        return None

    def is_empty(self) -> bool:
        with self._total_lock:
            return self._total <= 0

    def size(self) -> int:
        with self._total_lock:
            return self._total

    def sizes(self) -> List[int]:
        """Return per-worker deque sizes (for diagnostics)."""
        result: List[int] = []
        for i in range(self._n_workers):
            with self._locks[i]:
                result.append(len(self._deques[i]))
        return result


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


@dataclass
class ProgressInfo:
    """Snapshot of parallel execution progress."""

    completed: int = 0
    total: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def fraction(self) -> float:
        return self.completed / max(self.total, 1)


ProgressCallback = Callable[[ProgressInfo], None]


def _noop_progress(info: ProgressInfo) -> None:
    pass


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------


class ParallelExecutor:
    """Configurable parallel executor wrapping :mod:`concurrent.futures`.

    Parameters
    ----------
    max_workers : int or None
        Max parallelism.  Defaults to CPU count.
    use_processes : bool
        If True use ``ProcessPoolExecutor``, else ``ThreadPoolExecutor``.
    per_worker_mb : float
        Memory budget per worker for :func:`memory_aware_workers`.
    timeout : float or None
        Per-task timeout in seconds.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        per_worker_mb: float = 512.0,
        timeout: Optional[float] = None,
    ) -> None:
        if max_workers is None:
            max_workers = memory_aware_workers(per_worker_mb)
        self._max_workers = max(max_workers, 1)
        self._use_processes = use_processes
        self._timeout = timeout
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None

    def _make_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        if self._use_processes:
            return ProcessPoolExecutor(max_workers=self._max_workers)
        return ThreadPoolExecutor(max_workers=self._max_workers)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "ParallelExecutor":
        self._executor = self._make_executor()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # -- public API ---------------------------------------------------------

    def map(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        progress: Optional[ProgressCallback] = None,
        chunk_size: int = 1,
    ) -> List[TaskResult[R]]:
        """Apply *fn* to each item in parallel and return :class:`TaskResult` list.

        Results are returned in the same order as *items*.
        """
        if progress is None:
            progress = _noop_progress

        results: List[TaskResult[R]] = [TaskResult() for _ in items]
        start = time.monotonic()

        executor = self._executor or self._make_executor()
        own_executor = self._executor is None

        try:
            futures: Dict[Future, int] = {}
            for idx, item in enumerate(items):
                fut = executor.submit(self._run_task, fn, item, idx)
                futures[fut] = idx

            completed_count = 0
            failed_count = 0
            for fut in as_completed(futures, timeout=self._timeout):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                    if not results[idx].succeeded:
                        failed_count += 1
                except Exception as exc:
                    results[idx] = TaskResult(
                        error=exc,
                        succeeded=False,
                        task_id=str(idx),
                    )
                    failed_count += 1
                completed_count += 1
                progress(
                    ProgressInfo(
                        completed=completed_count,
                        total=len(items),
                        failed=failed_count,
                        elapsed_seconds=time.monotonic() - start,
                    )
                )
        except TimeoutError:
            logger.warning("parallel map timed out after %.1fs", self._timeout)
        finally:
            if own_executor:
                executor.shutdown(wait=False)

        return results

    @staticmethod
    def _run_task(fn: Callable[[T], R], item: T, idx: int) -> TaskResult[R]:
        """Execute a single task and wrap the result."""
        t0 = time.monotonic()
        try:
            value = fn(item)
            return TaskResult(
                value=value,
                elapsed_seconds=time.monotonic() - t0,
                task_id=str(idx),
                succeeded=True,
            )
        except Exception as exc:
            return TaskResult(
                error=exc,
                elapsed_seconds=time.monotonic() - t0,
                task_id=str(idx),
                succeeded=False,
            )

    def submit(self, fn: Callable[..., R], *args: Any) -> Future:
        """Submit a single callable to the pool."""
        if self._executor is None:
            raise RuntimeError("executor not started; use as context manager")
        return self._executor.submit(fn, *args)

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None


# ---------------------------------------------------------------------------
# parallel_map convenience function
# ---------------------------------------------------------------------------


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    progress: Optional[ProgressCallback] = None,
    timeout: Optional[float] = None,
) -> List[TaskResult[R]]:
    """Convenience wrapper for :meth:`ParallelExecutor.map`.

    >>> results = parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2)
    >>> [r.value for r in results]
    [2, 4, 6]
    """
    with ParallelExecutor(
        max_workers=max_workers,
        use_processes=use_processes,
        timeout=timeout,
    ) as executor:
        return executor.map(fn, items, progress=progress)


# ---------------------------------------------------------------------------
# parallel_branch_analysis – analyse independent DAG branches concurrently
# ---------------------------------------------------------------------------


def parallel_branch_analysis(
    branches: Sequence[Sequence[Any]],
    analyse_fn: Callable[[Sequence[Any]], R],
    max_workers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> List[TaskResult[R]]:
    """Analyse independent DAG branches in parallel.

    Parameters
    ----------
    branches : sequence of sequences
        Each inner sequence is a list of node IDs forming an independent
        branch (no edges between branches).
    analyse_fn : callable
        Function that takes a branch (list of nodes) and returns an
        analysis result.
    max_workers : int or None
        Max parallelism.
    progress : callable or None
        Progress callback.

    Returns
    -------
    list of TaskResult
        One result per branch, in order.
    """
    return parallel_map(
        analyse_fn,
        list(branches),
        max_workers=max_workers,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Rate-limited executor
# ---------------------------------------------------------------------------


class RateLimitedExecutor:
    """Executor that limits the rate of task submission.

    Useful when downstream resources (e.g. disk I/O) would be overwhelmed
    by unbounded parallelism.

    Parameters
    ----------
    max_workers : int
        Pool size.
    max_per_second : float
        Maximum number of tasks started per second.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_per_second: float = 100.0,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._interval = 1.0 / max(max_per_second, 0.01)
        self._last_submit = 0.0
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., R], *args: Any) -> Future:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_submit)
            if wait > 0:
                time.sleep(wait)
            self._last_submit = time.monotonic()
        return self._executor.submit(fn, *args)

    def map(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
    ) -> List[R]:
        futures = [self.submit(fn, item) for item in items]
        return [f.result() for f in futures]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "RateLimitedExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=True)


# ---------------------------------------------------------------------------
# BatchProcessor – process items in batches with parallelism
# ---------------------------------------------------------------------------


class BatchProcessor(Generic[T, R]):
    """Process items in fixed-size batches with per-batch parallelism.

    Parameters
    ----------
    fn : callable
        Function to apply to each item.
    batch_size : int
        Number of items per batch.
    max_workers : int
        Parallelism within each batch.
    """

    def __init__(
        self,
        fn: Callable[[T], R],
        batch_size: int = 100,
        max_workers: int = 4,
    ) -> None:
        self._fn = fn
        self._batch_size = max(batch_size, 1)
        self._max_workers = max(max_workers, 1)

    def process(
        self,
        items: Sequence[T],
        progress: Optional[ProgressCallback] = None,
    ) -> List[TaskResult[R]]:
        """Process all *items* in batches.

        Returns a flat list of :class:`TaskResult` in original order.
        """
        if progress is None:
            progress = _noop_progress

        all_results: List[TaskResult[R]] = []
        total = len(items)
        start = time.monotonic()

        for chunk in chunked_iter(items, self._batch_size):
            batch_results = parallel_map(
                self._fn,
                chunk,
                max_workers=self._max_workers,
            )
            all_results.extend(batch_results)
            progress(
                ProgressInfo(
                    completed=len(all_results),
                    total=total,
                    failed=sum(1 for r in all_results if not r.succeeded),
                    elapsed_seconds=time.monotonic() - start,
                )
            )

        return all_results


# ---------------------------------------------------------------------------
# DAG-aware parallel scheduler
# ---------------------------------------------------------------------------


def dag_parallel_execute(
    graph: Mapping[Any, Sequence[Any]],
    fn: Callable[[Any], R],
    max_workers: Optional[int] = None,
) -> Dict[Any, TaskResult[R]]:
    """Execute *fn* on each node of a DAG respecting dependencies.

    A node is only scheduled after all its predecessors have completed.

    Parameters
    ----------
    graph : mapping node → list of successors
        DAG adjacency list.
    fn : callable
        Function to apply to each node.
    max_workers : int or None

    Returns
    -------
    dict
        Node → TaskResult mapping.
    """
    all_nodes: Set[Any] = set(graph)
    for n in graph:
        for s in graph[n]:
            all_nodes.add(s)

    # Build predecessor map and in-degree
    pred: Dict[Any, Set[Any]] = {n: set() for n in all_nodes}
    for n in graph:
        for s in graph[n]:
            pred[s].add(n)

    in_deg = {n: len(pred[n]) for n in all_nodes}
    results: Dict[Any, TaskResult[R]] = {}
    lock = threading.Lock()
    done_event = threading.Event()

    remaining = len(all_nodes)
    if remaining == 0:
        return results

    workers = max_workers or os.cpu_count() or 4
    executor = ThreadPoolExecutor(max_workers=workers)
    pending_futures: Dict[Future, Any] = {}

    def _on_complete(future: Future) -> None:
        nonlocal remaining
        node = pending_futures[future]
        t_result: TaskResult[R]
        try:
            t_result = future.result()
        except Exception as exc:
            t_result = TaskResult(error=exc, succeeded=False, task_id=str(node))

        with lock:
            results[node] = t_result
            remaining -= 1
            # Unblock successors
            for succ in graph.get(node, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    _submit(succ)
            if remaining == 0:
                done_event.set()

    def _submit(node: Any) -> None:
        fut = executor.submit(ParallelExecutor._run_task, fn, node, 0)
        pending_futures[fut] = node
        fut.add_done_callback(_on_complete)

    # Start with sources (in-degree 0)
    with lock:
        for n in all_nodes:
            if in_deg[n] == 0:
                _submit(n)

    done_event.wait()
    executor.shutdown(wait=True)
    return results
