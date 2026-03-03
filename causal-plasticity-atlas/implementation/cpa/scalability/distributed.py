"""Distributed computation primitives.

Lightweight abstractions for partitioning work across multiple
workers and collecting results, backed by ``concurrent.futures``
and optionally ``multiprocessing``.  Includes parallel DAG score
evaluation and a thread-safe shared score cache.
"""

from __future__ import annotations

import os
import time
import threading
from collections import OrderedDict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    Future,
    as_completed,
)
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Container for the result of a single distributed task."""

    task_id: str
    result: Any
    elapsed_time: float
    worker_id: str


# ---------------------------------------------------------------------------
# WorkPartitioner
# ---------------------------------------------------------------------------

class WorkPartitioner:
    """Utility for splitting work items across workers.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers to partition across.
    """

    def __init__(self, n_workers: int) -> None:
        self._n_workers = max(1, n_workers)

    def partition_nodes(
        self, nodes: List[int]
    ) -> List[List[int]]:
        """Split *nodes* into roughly equal chunks per worker.

        Uses round-robin assignment so that consecutive nodes are
        distributed across different workers (improves load balance
        when node processing time correlates with index).
        """
        chunks: List[List[int]] = [[] for _ in range(self._n_workers)]
        for idx, node in enumerate(nodes):
            chunks[idx % self._n_workers].append(node)
        return [c for c in chunks if c]

    def partition_parent_sets(
        self, node: int, candidates: Set[int], max_parents: int = 3
    ) -> List[List[Tuple[int, FrozenSet[int]]]]:
        """Partition ``(node, parent_set)`` work items across workers.

        Enumerates all parent sets of *node* up to size *max_parents*
        from *candidates* and distributes them evenly.
        """
        items: List[Tuple[int, FrozenSet[int]]] = []
        cands = sorted(candidates - {node})
        for k in range(max_parents + 1):
            for combo in combinations(cands, k):
                items.append((node, frozenset(combo)))

        return self._chunk_items(items, self._n_workers)

    @staticmethod
    def _chunk_items(
        items: List[Any], n_chunks: int
    ) -> List[List[Any]]:
        """Divide *items* into *n_chunks* balanced lists."""
        n_chunks = max(1, n_chunks)
        size = len(items)
        base, extra = divmod(size, n_chunks)
        chunks: List[List[Any]] = []
        start = 0
        for i in range(n_chunks):
            end = start + base + (1 if i < extra else 0)
            if start < end:
                chunks.append(items[start:end])
            start = end
        return chunks


# ---------------------------------------------------------------------------
# SharedScoreCache – thread-safe
# ---------------------------------------------------------------------------

class SharedScoreCache:
    """Thread-safe score cache for use with parallel workers.

    Parameters
    ----------
    max_size : int
        Maximum entries before LRU eviction.
    """

    def __init__(self, max_size: int = 100_000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _serialize_key(key: Any) -> str:
        """Deterministic string representation of *key*."""
        if isinstance(key, tuple) and len(key) == 2:
            node, parents = key
            return f"{node}|{','.join(map(str, sorted(parents)))}"
        return str(key)

    def get(self, key: Any) -> Optional[float]:
        """Thread-safe lookup."""
        sk = self._serialize_key(key)
        with self._lock:
            if sk in self._cache:
                self._cache.move_to_end(sk)
                self._hits += 1
                return self._cache[sk]
            self._misses += 1
            return None

    def put(self, key: Any, value: float) -> None:
        """Thread-safe insert."""
        sk = self._serialize_key(key)
        with self._lock:
            if sk in self._cache:
                self._cache.move_to_end(sk)
                self._cache[sk] = value
            else:
                self._cache[sk] = value
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def get_or_compute(
        self, key: Any, compute_fn: Callable[[], float]
    ) -> float:
        """Atomic get-or-compute: avoids redundant computation."""
        val = self.get(key)
        if val is not None:
            return val
        result = compute_fn()
        self.put(key, result)
        return result

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"SharedScoreCache(size={len(self._cache)}, "
            f"max={self._max_size}, hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# DAGParallelEvaluator
# ---------------------------------------------------------------------------

class DAGParallelEvaluator:
    """Evaluate local scores in parallel across multiple workers.

    Parameters
    ----------
    score_fn : callable
        ``score_fn(node, parent_set)`` → float.
    n_workers : int
        Number of parallel threads/processes.
    cache : SharedScoreCache or None
        Optional shared cache for de-duplication.
    """

    def __init__(
        self,
        score_fn: Callable[[int, FrozenSet[int]], float],
        n_workers: int = 4,
        cache: Optional[SharedScoreCache] = None,
    ) -> None:
        self._score_fn = score_fn
        self._n_workers = max(1, n_workers)
        self._cache = cache or SharedScoreCache()

    def parallel_local_scores(
        self,
        nodes: List[int],
        parent_sets: Dict[int, List[FrozenSet[int]]],
    ) -> Dict[Tuple[int, FrozenSet[int]], float]:
        """Evaluate local scores for all ``(node, parent_set)`` pairs.

        Parameters
        ----------
        nodes : list of int
            Nodes to score.
        parent_sets : dict
            ``{node: [frozenset, ...]}`` parent-set candidates.

        Returns
        -------
        dict mapping ``(node, parents)`` → score.
        """
        items: List[Tuple[int, FrozenSet[int]]] = []
        for node in nodes:
            for ps in parent_sets.get(node, [frozenset()]):
                items.append((node, ps))

        results: Dict[Tuple[int, FrozenSet[int]], float] = {}
        batches = WorkPartitioner._chunk_items(
            items, self._n_workers
        )

        with ThreadPoolExecutor(
            max_workers=self._n_workers
        ) as executor:
            futures: Dict[Future, List[Tuple[int, FrozenSet[int]]]] = {}
            for batch in batches:
                f = executor.submit(self._batch_score, batch)
                futures[f] = batch

            for future in as_completed(futures):
                batch_results = future.result()
                results.update(batch_results)

        return results

    def parallel_dag_scores(
        self, dags: List[NDArray]
    ) -> List[float]:
        """Score multiple complete DAGs in parallel.

        Each DAG is scored as the sum of local scores for every node.
        """
        with ThreadPoolExecutor(
            max_workers=self._n_workers
        ) as executor:
            futures = {
                executor.submit(self._score_dag, dag): idx
                for idx, dag in enumerate(dags)
            }
            scores = [0.0] * len(dags)
            for future in as_completed(futures):
                idx = futures[future]
                scores[idx] = future.result()
        return scores

    def _batch_score(
        self, batch: List[Tuple[int, FrozenSet[int]]]
    ) -> Dict[Tuple[int, FrozenSet[int]], float]:
        """Score a batch of ``(node, parent_set)`` pairs."""
        results: Dict[Tuple[int, FrozenSet[int]], float] = {}
        for node, parents in batch:
            key = (node, parents)
            score = self._cache.get_or_compute(
                key, lambda n=node, p=parents: self._score_fn(n, p)
            )
            results[key] = score
        return results

    def _score_dag(self, dag: NDArray) -> float:
        """Score a single DAG (sum of local scores)."""
        dag = np.asarray(dag)
        n = dag.shape[0]
        total = 0.0
        for j in range(n):
            parents = frozenset(int(i) for i in np.nonzero(dag[:, j])[0])
            total += self._cache.get_or_compute(
                (j, parents),
                lambda n=j, p=parents: self._score_fn(n, p),
            )
        return total


# ---------------------------------------------------------------------------
# DistributedEngine
# ---------------------------------------------------------------------------

class DistributedEngine:
    """Simple distributed execution engine.

    Parameters
    ----------
    n_workers : int
        Number of worker threads/processes.
    backend : str
        ``"threading"`` or ``"multiprocessing"``.
    """

    def __init__(
        self,
        n_workers: int = 4,
        backend: str = "multiprocessing",
    ) -> None:
        self._n_workers = max(1, n_workers)
        self._backend = backend
        self._pool: Any = None
        self._active = False
        self._task_counter = 0

    def _get_executor(self) -> Any:
        """Create or return the executor."""
        if self._pool is None:
            if self._backend == "threading":
                self._pool = ThreadPoolExecutor(
                    max_workers=self._n_workers
                )
            else:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._n_workers
                )
            self._active = True
        return self._pool

    def map(
        self, fn: Callable, items: List
    ) -> List[TaskResult]:
        """Apply *fn* to each element of *items* in parallel."""
        executor = self._get_executor()
        results: List[TaskResult] = []
        futures: Dict[Future, Tuple[int, Any]] = {}

        for idx, item in enumerate(items):
            f = executor.submit(self._timed_call, fn, item)
            futures[f] = (idx, item)

        ordered: List[Optional[TaskResult]] = [None] * len(items)
        for future in as_completed(futures):
            idx, _ = futures[future]
            elapsed, result = future.result()
            self._task_counter += 1
            ordered[idx] = TaskResult(
                task_id=f"task-{self._task_counter}",
                result=result,
                elapsed_time=elapsed,
                worker_id=f"worker-{idx % self._n_workers}",
            )

        return [r for r in ordered if r is not None]

    def starmap(
        self, fn: Callable, arg_tuples: List[Tuple]
    ) -> List[TaskResult]:
        """Apply *fn* to each tuple of arguments in parallel."""
        executor = self._get_executor()
        results: List[TaskResult] = []
        futures: Dict[Future, int] = {}

        for idx, args in enumerate(arg_tuples):
            f = executor.submit(self._timed_call_star, fn, args)
            futures[f] = idx

        ordered: List[Optional[TaskResult]] = [None] * len(arg_tuples)
        for future in as_completed(futures):
            idx = futures[future]
            elapsed, result = future.result()
            self._task_counter += 1
            ordered[idx] = TaskResult(
                task_id=f"task-{self._task_counter}",
                result=result,
                elapsed_time=elapsed,
                worker_id=f"worker-{idx % self._n_workers}",
            )

        return [r for r in ordered if r is not None]

    def submit(self, fn: Callable, *args: Any) -> TaskResult:
        """Submit a single task for synchronous execution."""
        t0 = time.monotonic()
        result = fn(*args)
        elapsed = time.monotonic() - t0
        self._task_counter += 1
        return TaskResult(
            task_id=f"task-{self._task_counter}",
            result=result,
            elapsed_time=elapsed,
            worker_id="worker-0",
        )

    def shutdown(self) -> None:
        """Shut down all worker processes."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._active = False

    def active_workers(self) -> int:
        """Return the number of currently active workers."""
        if not self._active:
            return 0
        return self._n_workers

    @staticmethod
    def _timed_call(fn: Callable, arg: Any) -> Tuple[float, Any]:
        """Call *fn(arg)* and return ``(elapsed, result)``."""
        t0 = time.monotonic()
        result = fn(arg)
        return time.monotonic() - t0, result

    @staticmethod
    def _timed_call_star(
        fn: Callable, args: Tuple
    ) -> Tuple[float, Any]:
        """Call *fn(*args)* and return ``(elapsed, result)``."""
        t0 = time.monotonic()
        result = fn(*args)
        return time.monotonic() - t0, result

    def __enter__(self) -> "DistributedEngine":
        self._get_executor()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"DistributedEngine(n_workers={self._n_workers}, "
            f"backend={self._backend!r}, "
            f"active={self._active})"
        )
