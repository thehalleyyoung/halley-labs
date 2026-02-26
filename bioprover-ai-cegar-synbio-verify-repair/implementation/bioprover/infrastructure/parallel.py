"""
Parallel execution engine for BioProver.

Provides high-level abstractions over Python's concurrent.futures:
task executors, portfolio runners (first-result-wins), deterministic
single-threaded execution for reproducibility, work-stealing pools
for ensemble simulations, and resource monitoring.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMED_OUT = auto()


@dataclass
class TaskResult(Generic[T]):
    """Result of a parallel task, carrying status, value or exception."""

    task_id: str
    status: TaskStatus
    value: Optional[T] = None
    error: Optional[BaseException] = None
    elapsed_s: float = 0.0
    worker_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def unwrap(self) -> T:
        """Return value or re-raise error."""
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise ValueError(f"Task {self.task_id} produced no value")
        return self.value


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------

@dataclass
class ProgressUpdate:
    """Progress information passed to user callbacks."""
    task_id: str
    completed: int
    total: int
    elapsed_s: float
    message: str = ""


ProgressCallback = Callable[[ProgressUpdate], None]


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Unified executor wrapping process- and thread-based parallelism.

    Provides ``map``, ``submit``, ``gather`` with timeout, cancellation,
    and optional progress callbacks.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        thread_name_prefix: str = "bioprover",
    ) -> None:
        self.max_workers = max_workers
        self.use_processes = use_processes
        self._thread_prefix = thread_name_prefix
        self._pool: Optional[Any] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._pool is not None:
            return
        if self.use_processes:
            self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix=self._thread_prefix,
            )
        logger.info(
            "TaskExecutor started (%s, %d workers)",
            "process" if self.use_processes else "thread",
            self.max_workers,
        )

    def shutdown(self, wait_for: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait_for)
            self._pool = None
            logger.info("TaskExecutor shut down")

    def __enter__(self) -> "TaskExecutor":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # -- submission ---------------------------------------------------------

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        task_id: str = "",
        **kwargs: Any,
    ) -> Future[TaskResult[T]]:
        """Submit a single callable; returns a Future[TaskResult]."""
        self._ensure_started()
        tid = task_id or f"task-{id(fn)}"

        def _wrapper() -> TaskResult[T]:
            t0 = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                return TaskResult(
                    task_id=tid,
                    status=TaskStatus.COMPLETED,
                    value=result,
                    elapsed_s=time.monotonic() - t0,
                )
            except Exception as exc:
                return TaskResult(
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error=exc,
                    elapsed_s=time.monotonic() - t0,
                )

        return self._pool.submit(_wrapper)  # type: ignore[union-attr]

    def map(
        self,
        fn: Callable[..., T],
        items: Sequence[Any],
        *,
        task_prefix: str = "map",
        timeout_s: Optional[float] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[TaskResult[T]]:
        """Apply *fn* to each item in parallel; return ordered results."""
        self._ensure_started()
        futures: List[Future[TaskResult[T]]] = []
        total = len(items)

        for idx, item in enumerate(items):
            tid = f"{task_prefix}-{idx}"
            futures.append(self.submit(fn, item, task_id=tid))

        return self._gather(futures, total, timeout_s, progress)

    def gather(
        self,
        futures: Sequence[Future[TaskResult[T]]],
        *,
        timeout_s: Optional[float] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[TaskResult[T]]:
        """Wait for all futures and return their results."""
        return self._gather(list(futures), len(futures), timeout_s, progress)

    # -- internal -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._pool is None:
            self.start()

    def _gather(
        self,
        futures: List[Future[TaskResult[T]]],
        total: int,
        timeout_s: Optional[float],
        progress: Optional[ProgressCallback],
    ) -> List[TaskResult[T]]:
        t0 = time.monotonic()
        results: List[TaskResult[T]] = []
        completed = 0

        try:
            for fut in as_completed(futures, timeout=timeout_s):
                result = fut.result()
                results.append(result)
                completed += 1
                if progress is not None:
                    progress(ProgressUpdate(
                        task_id=result.task_id,
                        completed=completed,
                        total=total,
                        elapsed_s=time.monotonic() - t0,
                    ))
        except TimeoutError:
            for fut in futures:
                if not fut.done():
                    fut.cancel()
                    results.append(TaskResult(
                        task_id="unknown",
                        status=TaskStatus.TIMED_OUT,
                        elapsed_s=time.monotonic() - t0,
                    ))

        return results


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------

class PortfolioRunner:
    """Run multiple strategies in parallel; return the first successful result.

    Used for portfolio solving: try several solver configurations or
    verification strategies concurrently and use whichever finishes first.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
    ) -> None:
        self.max_workers = max_workers
        self.use_processes = use_processes

    def run(
        self,
        strategies: Sequence[Callable[[], T]],
        *,
        timeout_s: Optional[float] = None,
        accept: Optional[Callable[[T], bool]] = None,
    ) -> Optional[TaskResult[T]]:
        """Execute all *strategies* and return the first accepted result.

        Parameters
        ----------
        strategies:
            Zero-argument callables, each implementing a strategy.
        timeout_s:
            Overall wall-clock budget.
        accept:
            Predicate on a result value; defaults to ``lambda v: v is not None``.

        Returns ``None`` if no strategy succeeds within the budget.
        """
        accept = accept or (lambda v: v is not None)

        if self.use_processes:
            pool = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            pool = ThreadPoolExecutor(max_workers=self.max_workers)

        futures: Dict[Future[TaskResult[T]], int] = {}
        try:
            for idx, strat in enumerate(strategies):
                tid = f"strategy-{idx}"

                def _run(s: Callable[[], T] = strat, t: str = tid) -> TaskResult[T]:
                    t0 = time.monotonic()
                    try:
                        val = s()
                        return TaskResult(
                            task_id=t,
                            status=TaskStatus.COMPLETED,
                            value=val,
                            elapsed_s=time.monotonic() - t0,
                        )
                    except Exception as exc:
                        return TaskResult(
                            task_id=t,
                            status=TaskStatus.FAILED,
                            error=exc,
                            elapsed_s=time.monotonic() - t0,
                        )

                fut: Future[TaskResult[T]] = pool.submit(_run)
                futures[fut] = idx

            deadline = time.monotonic() + (timeout_s or float("inf"))

            while futures:
                remaining = max(0.01, deadline - time.monotonic())
                done, _ = wait(
                    futures.keys(), timeout=remaining, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    result = fut.result()
                    if result.ok and accept(result.value):  # type: ignore[arg-type]
                        # Cancel remaining
                        for other in futures:
                            if other is not fut:
                                other.cancel()
                        logger.info(
                            "Portfolio: strategy %s won in %.2fs",
                            result.task_id,
                            result.elapsed_s,
                        )
                        return result
                    del futures[fut]

                if time.monotonic() >= deadline:
                    break

            return None
        finally:
            pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Deterministic executor
# ---------------------------------------------------------------------------

class DeterministicExecutor:
    """Single-threaded executor for bit-reproducible runs.

    Seeds the RNG and executes all tasks sequentially in submission order.
    Useful for debugging and regression testing.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def map(
        self,
        fn: Callable[..., T],
        items: Sequence[Any],
        *,
        task_prefix: str = "det",
        progress: Optional[ProgressCallback] = None,
    ) -> List[TaskResult[T]]:
        random.seed(self.seed)
        results: List[TaskResult[T]] = []
        total = len(items)
        t0 = time.monotonic()

        for idx, item in enumerate(items):
            tid = f"{task_prefix}-{idx}"
            t_start = time.monotonic()
            try:
                val = fn(item)
                results.append(TaskResult(
                    task_id=tid,
                    status=TaskStatus.COMPLETED,
                    value=val,
                    elapsed_s=time.monotonic() - t_start,
                ))
            except Exception as exc:
                results.append(TaskResult(
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error=exc,
                    elapsed_s=time.monotonic() - t_start,
                ))
            if progress is not None:
                progress(ProgressUpdate(
                    task_id=tid,
                    completed=idx + 1,
                    total=total,
                    elapsed_s=time.monotonic() - t0,
                ))

        return results

    def run_one(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> TaskResult[T]:
        random.seed(self.seed)
        t0 = time.monotonic()
        try:
            val = fn(*args, **kwargs)
            return TaskResult(
                task_id="det-single",
                status=TaskStatus.COMPLETED,
                value=val,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return TaskResult(
                task_id="det-single",
                status=TaskStatus.FAILED,
                error=exc,
                elapsed_s=time.monotonic() - t0,
            )


# ---------------------------------------------------------------------------
# Work-stealing pool
# ---------------------------------------------------------------------------

class WorkStealingPool:
    """Load-balanced parallel pool with per-worker local queues and stealing.

    Designed for ensemble stochastic simulations where individual trajectory
    runtimes vary widely.  Workers that finish early steal work from others.
    """

    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = max(1, num_workers)
        self._queues: List[Deque[Any]] = [deque() for _ in range(self.num_workers)]
        self._results: List[TaskResult[Any]] = []
        self._lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._progress_cb: Optional[ProgressCallback] = None
        self._completed = 0
        self._total = 0

    def map(
        self,
        fn: Callable[[Any], T],
        items: Sequence[Any],
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> List[TaskResult[T]]:
        """Distribute *items* across workers with work-stealing."""
        self._results = []
        self._completed = 0
        self._total = len(items)
        self._progress_cb = progress

        # Round-robin initial distribution
        for idx, item in enumerate(items):
            worker_idx = idx % self.num_workers
            self._queues[worker_idx].append((idx, item))

        threads: List[threading.Thread] = []
        for wid in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(wid, fn),
                name=f"ws-worker-{wid}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Sort by original index
        self._results.sort(key=lambda r: int(r.task_id.split("-")[1]))
        return self._results  # type: ignore[return-value]

    def _worker_loop(self, wid: int, fn: Callable[[Any], Any]) -> None:
        while True:
            item = self._try_dequeue(wid)
            if item is None:
                item = self._try_steal(wid)
            if item is None:
                return  # all queues empty

            idx, payload = item
            tid = f"ws-{idx}"
            t0 = time.monotonic()
            try:
                val = fn(payload)
                result = TaskResult(
                    task_id=tid,
                    status=TaskStatus.COMPLETED,
                    value=val,
                    elapsed_s=time.monotonic() - t0,
                    worker_id=f"worker-{wid}",
                )
            except Exception as exc:
                result = TaskResult(
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error=exc,
                    elapsed_s=time.monotonic() - t0,
                    worker_id=f"worker-{wid}",
                )

            with self._result_lock:
                self._results.append(result)
                self._completed += 1
                if self._progress_cb is not None:
                    self._progress_cb(ProgressUpdate(
                        task_id=tid,
                        completed=self._completed,
                        total=self._total,
                        elapsed_s=result.elapsed_s,
                    ))

    def _try_dequeue(self, wid: int) -> Optional[Any]:
        q = self._queues[wid]
        with self._lock:
            if q:
                return q.popleft()
        return None

    def _try_steal(self, wid: int) -> Optional[Any]:
        """Attempt to steal from the longest queue of another worker."""
        with self._lock:
            best_idx = -1
            best_len = 0
            for i, q in enumerate(self._queues):
                if i != wid and len(q) > best_len:
                    best_idx = i
                    best_len = len(q)
            if best_idx >= 0 and best_len > 0:
                return self._queues[best_idx].pop()  # steal from back
        return None


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage."""
    timestamp: float
    memory_rss_mb: float
    cpu_percent: float
    active_threads: int


class ResourceMonitor:
    """Lightweight resource monitor that samples periodically."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._samples: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="resource-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    @property
    def samples(self) -> List[ResourceSnapshot]:
        return list(self._samples)

    @property
    def peak_memory_mb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.memory_rss_mb for s in self._samples)

    def _sample_loop(self) -> None:
        while self._running:
            snap = self._take_snapshot()
            self._samples.append(snap)
            time.sleep(self.interval_s)

    @staticmethod
    def _take_snapshot() -> ResourceSnapshot:
        try:
            import resource as res_mod

            usage = res_mod.getrusage(res_mod.RUSAGE_SELF)
            mem_mb = usage.ru_maxrss / 1024.0  # macOS: bytes; Linux: KB
            import sys
            if sys.platform == "linux":
                mem_mb = usage.ru_maxrss / 1024.0
            else:
                mem_mb = usage.ru_maxrss / (1024.0 * 1024.0)
        except Exception:
            mem_mb = 0.0

        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = 0.0

        return ResourceSnapshot(
            timestamp=time.time(),
            memory_rss_mb=mem_mb,
            cpu_percent=load * 100.0 / max(os.cpu_count() or 1, 1),
            active_threads=threading.active_count(),
        )
