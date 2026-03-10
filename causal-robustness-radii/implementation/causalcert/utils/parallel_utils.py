"""Parallel-execution utilities: parallel map, chunking, retry logic.

Built on :mod:`concurrent.futures` to avoid heavyweight dependencies.
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# ====================================================================
# 1. Parallel Map with Progress
# ====================================================================

def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    n_workers: int | None = None,
    backend: str = "thread",
    description: str = "",
    show_progress: bool = True,
) -> list[R]:
    """Apply *fn* to each item in parallel and return results in order.

    Parameters
    ----------
    fn : callable taking one argument and returning a result.
    items : sequence of inputs.
    n_workers : number of workers (default: min(len(items), cpu_count)).
    backend : ``"thread"`` or ``"process"``.
    description : label for the progress bar.
    show_progress : whether to print a progress bar.
    """
    if not items:
        return []

    n = len(items)
    n_workers = min(n, n_workers or _default_workers())

    Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    results: list[R | None] = [None] * n
    completed = 0

    with Executor(max_workers=n_workers) as pool:
        future_to_idx: dict[Future[R], int] = {}
        for idx, item in enumerate(items):
            fut = pool.submit(fn, item)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            completed += 1
            if show_progress:
                _print_progress(description, completed, n)

    if show_progress:
        sys.stderr.write("\n")

    return results  # type: ignore[return-value]


def _default_workers() -> int:
    """Sensible default: CPU count, clamped to [1, 8]."""
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    return max(1, min(cpus, 8))


def _print_progress(desc: str, done: int, total: int) -> None:
    frac = done / total if total else 1.0
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(f"\r{desc} |{bar}| {done}/{total}")
    sys.stderr.flush()


# ====================================================================
# 2. Chunk Splitting
# ====================================================================

def split_into_chunks(
    items: Sequence[T],
    n_chunks: int | None = None,
    chunk_size: int | None = None,
) -> list[list[T]]:
    """Split *items* into roughly equal chunks.

    Specify either *n_chunks* or *chunk_size* (not both).
    """
    n = len(items)
    if n == 0:
        return []

    if chunk_size is not None:
        n_chunks = max(1, math.ceil(n / chunk_size))
    elif n_chunks is None:
        n_chunks = _default_workers()

    n_chunks = max(1, min(n_chunks, n))
    base, extra = divmod(n, n_chunks)
    chunks: list[list[T]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(list(items[start: start + size]))
        start += size
    return chunks


def flatten(chunks: Sequence[Sequence[T]]) -> list[T]:
    """Flatten a list of chunks back into a single list."""
    return [item for chunk in chunks for item in chunk]


# ====================================================================
# 3. Memory-Bounded Parallel Execution
# ====================================================================

@dataclass
class MemoryBudget:
    """Simple memory-budget tracker (does NOT measure real RSS)."""
    max_bytes: int
    _used: int = 0

    def acquire(self, n_bytes: int) -> bool:
        """Reserve *n_bytes*.  Returns True on success."""
        if self._used + n_bytes > self.max_bytes:
            return False
        self._used += n_bytes
        return True

    def release(self, n_bytes: int) -> None:
        self._used = max(0, self._used - n_bytes)

    @property
    def available(self) -> int:
        return self.max_bytes - self._used


def memory_bounded_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    item_size_fn: Callable[[T], int],
    max_memory_bytes: int = 1 * 1024 ** 3,  # 1 GB default
    n_workers: int | None = None,
    backend: str = "thread",
) -> list[R]:
    """Like parallel_map but limits concurrent work by memory budget.

    *item_size_fn* should return an estimated byte cost per item.
    Items are submitted to the pool only when there is enough budget.
    """
    if not items:
        return []

    n = len(items)
    budget = MemoryBudget(max_memory_bytes)
    n_workers = min(n, n_workers or _default_workers())

    Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    results: list[R | None] = [None] * n

    with Executor(max_workers=n_workers) as pool:
        pending: dict[Future[R], tuple[int, int]] = {}
        idx = 0

        while idx < n or pending:
            # Submit items within budget
            while idx < n:
                cost = item_size_fn(items[idx])
                if not budget.acquire(cost):
                    break
                fut = pool.submit(fn, items[idx])
                pending[fut] = (idx, cost)
                idx += 1

            # Wait for at least one to finish
            if pending:
                done_futs = []
                for fut in as_completed(pending):
                    done_futs.append(fut)
                    break  # just one

                for fut in done_futs:
                    i, cost = pending.pop(fut)
                    results[i] = fut.result()
                    budget.release(cost)

    return results  # type: ignore[return-value]


# ====================================================================
# 4. Retry Logic
# ====================================================================

@dataclass
class RetryConfig:
    """Configuration for retry-with-backoff."""
    max_retries: int = 3
    initial_delay_s: float = 0.1
    backoff_factor: float = 2.0
    max_delay_s: float = 30.0
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)


def retry(
    fn: Callable[..., R],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> R:
    """Call *fn* with automatic retries on failure.

    Uses exponential backoff between retries.
    """
    cfg = config or RetryConfig()
    delay = cfg.initial_delay_s

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except cfg.retryable_exceptions:
            if attempt >= cfg.max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * cfg.backoff_factor, cfg.max_delay_s)

    # Unreachable, but keeps mypy happy
    raise RuntimeError("retry exhausted")


def retry_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    config: RetryConfig | None = None,
) -> list[R | Exception]:
    """Apply *fn* to each item with retries; collect results or exceptions."""
    cfg = config or RetryConfig()
    results: list[R | Exception] = []

    for item in items:
        try:
            result = retry(fn, item, config=cfg)
            results.append(result)
        except Exception as exc:
            results.append(exc)

    return results


# ====================================================================
# 5. Miscellaneous Helpers
# ====================================================================

def cpu_count() -> int:
    """Return usable CPU count, at least 1."""
    return max(1, os.cpu_count() or 1)


def run_sequential(
    fn: Callable[[T], R],
    items: Sequence[T],
) -> list[R]:
    """Sequential fallback for debugging (same interface as parallel_map)."""
    return [fn(item) for item in items]
