"""Managed process pool wrapper around :class:`concurrent.futures.ProcessPoolExecutor`."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ManagedProcessPool:
    """Thin wrapper providing context-manager semantics for a process pool.

    Parameters
    ----------
    n_workers:
        Maximum number of worker processes.
    """

    def __init__(self, n_workers: int = 1) -> None:
        self._n_workers = max(1, n_workers)
        self._pool: ProcessPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ManagedProcessPool:
        self._pool = ProcessPoolExecutor(max_workers=self._n_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self, fn: Callable[..., R], items: Iterable[T]) -> List[R]:
        """Apply *fn* to every element of *items* in parallel.

        Falls back to sequential execution when the pool is not active.
        """
        items_list = list(items)
        if self._pool is not None and self._n_workers > 1:
            return list(self._pool.map(fn, items_list))
        return [fn(item) for item in items_list]
