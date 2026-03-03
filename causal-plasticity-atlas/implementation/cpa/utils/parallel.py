"""Parallel computation utilities for the CPA engine.

Provides helper functions for parallelising embarrassingly-parallel
workloads across threads or processes, a thread-safe accumulator for
collecting results, and work-distribution strategies.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


# ===================================================================
# Defaults
# ===================================================================

_DEFAULT_WORKERS: int = min(os.cpu_count() or 1, 8)


def _resolve_workers(n_workers: Optional[int]) -> int:
    """Resolve number of workers (default: min(cpu_count, 8)).

    Parameters
    ----------
    n_workers : int or None
        Requested workers.  ``None`` → default, ``-1`` → cpu_count.

    Returns
    -------
    int
    """
    if n_workers is None:
        return _DEFAULT_WORKERS
    if n_workers == -1:
        return os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or -1, got {n_workers}")
    return n_workers


# ===================================================================
# parallel_map
# ===================================================================


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    n_workers: Optional[int] = None,
    backend: str = "thread",
    chunk_size: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ordered: bool = True,
) -> List[R]:
    """Apply *fn* to each item in *items* using a pool of workers.

    Parameters
    ----------
    fn : callable
        Function applied to each item.
    items : sequence
        Input items.
    n_workers : int, optional
        Number of workers.  ``None`` → default.
    backend : ``"thread"`` or ``"process"``
        Concurrency backend.
    chunk_size : int
        Number of items submitted per future (process backend only).
    progress_callback : callable, optional
        Called with ``(completed, total)`` after each future completes.
    ordered : bool
        If ``True``, results are returned in input order.  If ``False``,
        results are returned in completion order (slightly faster).

    Returns
    -------
    list
        Results in the same order as *items* (when ``ordered=True``).

    Raises
    ------
    ValueError
        On invalid parameters.
    RuntimeError
        If any worker raises an exception, it is re-raised.
    """
    if not items:
        return []

    workers = _resolve_workers(n_workers)
    total = len(items)

    # Fast path: single worker
    if workers == 1:
        results: List[R] = []
        for idx, item in enumerate(items):
            results.append(fn(item))
            if progress_callback is not None:
                progress_callback(idx + 1, total)
        return results

    pool_cls: type
    if backend == "thread":
        pool_cls = ThreadPoolExecutor
    elif backend == "process":
        pool_cls = ProcessPoolExecutor
    else:
        raise ValueError(f"backend must be 'thread' or 'process', got {backend!r}")

    results_map: Dict[int, R] = {}
    completed_count = 0

    with pool_cls(max_workers=workers) as pool:
        futures: Dict[Future, int] = {}
        for idx, item in enumerate(items):
            fut = pool.submit(fn, item)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            results_map[idx] = fut.result()  # propagates exceptions
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, total)

    if ordered:
        return [results_map[i] for i in range(total)]
    return list(results_map.values())


# ===================================================================
# Batch processing
# ===================================================================


def _split_batches(items: Sequence[T], batch_size: int) -> List[List[T]]:
    """Split *items* into batches of at most *batch_size*.

    Parameters
    ----------
    items : sequence
        Items to split.
    batch_size : int
        Max items per batch.

    Returns
    -------
    list of lists
    """
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def batch_process(
    fn: Callable[[List[T]], List[R]],
    items: Sequence[T],
    *,
    batch_size: int = 64,
    n_workers: Optional[int] = None,
    backend: str = "thread",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """Process *items* in batches, optionally in parallel.

    Parameters
    ----------
    fn : callable
        Function that accepts a *list* of items and returns a list of
        results (same length).
    items : sequence
        Input items.
    batch_size : int
        Number of items per batch.
    n_workers : int, optional
        Worker count for parallel batch processing.
    backend : ``"thread"`` or ``"process"``
        Concurrency backend.
    progress_callback : callable, optional
        Called with ``(items_done, total)`` after each batch.

    Returns
    -------
    list
        Concatenated results in input order.
    """
    if not items:
        return []
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batches = _split_batches(items, batch_size)
    total = len(items)
    done = 0

    def _process_batch(batch: List[T]) -> List[R]:
        return fn(batch)

    workers = _resolve_workers(n_workers)
    if workers == 1:
        all_results: List[R] = []
        for batch in batches:
            all_results.extend(_process_batch(batch))
            done_now = len(all_results)
            if progress_callback is not None:
                progress_callback(min(done_now, total), total)
        return all_results

    batch_results = parallel_map(
        _process_batch,
        batches,
        n_workers=workers,
        backend=backend,
    )
    flat: List[R] = []
    for br in batch_results:
        flat.extend(br)
        done += len(br)
        if progress_callback is not None:
            progress_callback(min(done, total), total)
    return flat


# ===================================================================
# Thread-safe accumulator
# ===================================================================


class ThreadSafeAccumulator(Generic[T]):
    """Thread-safe container for collecting results from concurrent workers.

    Parameters
    ----------
    initial : T, optional
        Initial value.  If ``None``, the accumulator starts empty and
        the first ``add`` call sets the initial value.

    Examples
    --------
    >>> acc = ThreadSafeAccumulator(0.0)
    >>> acc.add(1.5)
    >>> acc.add(2.3)
    >>> acc.value  # 3.8
    """

    def __init__(self, initial: Optional[T] = None) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = initial
        self._count: int = 0

    @property
    def value(self) -> Optional[T]:
        """Current accumulated value."""
        with self._lock:
            return self._value

    @property
    def count(self) -> int:
        """Number of ``add`` calls."""
        with self._lock:
            return self._count

    def add(self, item: T) -> None:
        """Add *item* to the accumulator using ``+`` operator.

        Parameters
        ----------
        item : T
            Value to add.  Must support ``__add__`` with the current value.
        """
        with self._lock:
            if self._value is None:
                self._value = item
            else:
                self._value = self._value + item  # type: ignore[operator]
            self._count += 1

    def add_to_list(self, item: Any) -> None:
        """Append *item* to an internal list (auto-created on first call).

        Parameters
        ----------
        item : Any
            Item to append.
        """
        with self._lock:
            if self._value is None:
                self._value = [item]  # type: ignore[assignment]
            else:
                self._value.append(item)  # type: ignore[union-attr]
            self._count += 1

    def reset(self, initial: Optional[T] = None) -> Optional[T]:
        """Reset and return the current value.

        Parameters
        ----------
        initial : T, optional
            New initial value.

        Returns
        -------
        T or None
            Previous value before reset.
        """
        with self._lock:
            prev = self._value
            self._value = initial
            self._count = 0
            return prev

    def __repr__(self) -> str:
        return f"ThreadSafeAccumulator(count={self._count}, value={self._value!r})"


# ===================================================================
# Work distribution utilities
# ===================================================================


def distribute_evenly(total: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Distribute *total* items into *n_chunks* near-equal ranges.

    Parameters
    ----------
    total : int
        Total number of items.
    n_chunks : int
        Number of chunks.

    Returns
    -------
    list of (start, end) tuples
        Each tuple gives a half-open range ``[start, end)``.
    """
    if total <= 0:
        return []
    n_chunks = min(n_chunks, total)
    base, extra = divmod(total, n_chunks)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def pairwise_indices(n: int, *, include_diagonal: bool = False) -> List[Tuple[int, int]]:
    """Generate all upper-triangle index pairs for an n×n matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.
    include_diagonal : bool
        Include ``(i, i)`` pairs.

    Returns
    -------
    list of (int, int)
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        start = i if include_diagonal else i + 1
        for j in range(start, n):
            pairs.append((i, j))
    return pairs
