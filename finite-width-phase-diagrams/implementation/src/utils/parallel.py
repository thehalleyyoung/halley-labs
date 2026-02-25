"""Parallel computation utilities for the finite-width phase diagram system.

Provides parallel grid sweep execution, parallel NTK computation across
widths, parallel ground-truth training, progress aggregation, and result
collection from parallel workers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
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
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from .logging import ProgressTracker, get_logger

_log = get_logger("fwpd.parallel")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    n_workers: int = 1
    backend: str = "multiprocessing"   # "multiprocessing", "threading", "sequential"
    chunk_size: int = 10
    timeout: float = 3600.0
    progress: bool = True
    maxtasksperchild: Optional[int] = None

    @property
    def is_parallel(self) -> bool:
        return self.n_workers > 1 and self.backend != "sequential"


# ---------------------------------------------------------------------------
# Worker result wrapper
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Result from a single parallel worker."""

    index: int
    success: bool
    value: Any = None
    error: Optional[str] = None
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Generic parallel map
# ---------------------------------------------------------------------------

def parallel_map(
    fn: Callable[..., T],
    items: Sequence[Any],
    config: Optional[ParallelConfig] = None,
    label: str = "parallel",
    unpack: bool = False,
) -> List[WorkerResult]:
    """Apply *fn* to each item in parallel, returning ordered results.

    Parameters
    ----------
    fn : callable
        Function to apply. Receives one item (or unpacked args if *unpack*).
    items : sequence
        Inputs to map over.
    config : ParallelConfig, optional
        Parallelism settings.
    label : str
        Label for progress logging.
    unpack : bool
        If True, unpack each item as ``fn(*item)``.

    Returns
    -------
    List of WorkerResult in the same order as *items*.
    """
    if config is None:
        config = ParallelConfig()

    n = len(items)
    if n == 0:
        return []

    if not config.is_parallel:
        return _sequential_map(fn, items, label, config.progress, unpack)

    return _pool_map(fn, items, config, label, unpack)


def _sequential_map(
    fn: Callable,
    items: Sequence[Any],
    label: str,
    progress: bool,
    unpack: bool,
) -> List[WorkerResult]:
    results: List[WorkerResult] = []
    tracker = ProgressTracker(len(items), label=label) if progress else None
    for i, item in enumerate(items):
        t0 = time.perf_counter()
        try:
            val = fn(*item) if unpack else fn(item)
            results.append(WorkerResult(i, True, val, elapsed=time.perf_counter() - t0))
        except Exception as exc:
            results.append(WorkerResult(
                i, False, error=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - t0,
            ))
        if tracker:
            tracker.update()
    if tracker:
        tracker.done()
    return results


def _pool_map(
    fn: Callable,
    items: Sequence[Any],
    config: ParallelConfig,
    label: str,
    unpack: bool,
) -> List[WorkerResult]:
    PoolClass = (
        ProcessPoolExecutor
        if config.backend == "multiprocessing"
        else ThreadPoolExecutor
    )
    n = len(items)
    results: List[Optional[WorkerResult]] = [None] * n
    tracker = ProgressTracker(n, label=label) if config.progress else None

    pool_kwargs: Dict[str, Any] = {"max_workers": config.n_workers}
    if config.backend == "multiprocessing" and config.maxtasksperchild:
        pool_kwargs["mp_context"] = mp.get_context("spawn")

    with PoolClass(**pool_kwargs) as executor:
        futures: Dict[Future, int] = {}
        for i, item in enumerate(items):
            if unpack:
                fut = executor.submit(_worker_wrapper, fn, item, True)
            else:
                fut = executor.submit(_worker_wrapper, fn, item, False)
            futures[fut] = i

        for fut in as_completed(futures, timeout=config.timeout):
            idx = futures[fut]
            try:
                wr = fut.result()
                wr.index = idx
                results[idx] = wr
            except Exception as exc:
                results[idx] = WorkerResult(
                    idx, False, error=f"{type(exc).__name__}: {exc}"
                )
            if tracker:
                tracker.update()

    if tracker:
        tracker.done()
    return [r if r is not None else WorkerResult(i, False, error="missing") for i, r in enumerate(results)]


def _worker_wrapper(fn: Callable, item: Any, unpack: bool) -> WorkerResult:
    t0 = time.perf_counter()
    try:
        val = fn(*item) if unpack else fn(item)
        return WorkerResult(0, True, val, elapsed=time.perf_counter() - t0)
    except Exception as exc:
        return WorkerResult(
            0, False, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            elapsed=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Domain-specific parallel functions
# ---------------------------------------------------------------------------

def parallel_grid_sweep(
    order_param_fn: Callable[[Dict[str, float]], float],
    grid_points: List[Dict[str, float]],
    config: Optional[ParallelConfig] = None,
) -> List[Dict[str, Any]]:
    """Evaluate an order parameter function over a grid of hyperparameters.

    Parameters
    ----------
    order_param_fn : callable
        Function mapping hyperparameter dict → order parameter value.
    grid_points : list of dicts
        Each dict specifies one grid point, e.g. ``{"lr": 0.01, "width": 128}``.
    config : ParallelConfig, optional

    Returns
    -------
    List of dicts with keys: coords, value, success, error.
    """
    if config is None:
        config = ParallelConfig()

    def _eval_point(point: Dict[str, float]) -> Dict[str, Any]:
        val = order_param_fn(point)
        return {"coords": point, "value": float(val)}

    worker_results = parallel_map(
        _eval_point, grid_points, config, label="grid sweep"
    )

    results = []
    for wr in worker_results:
        if wr.success:
            results.append({**wr.value, "success": True, "error": None})
        else:
            pt = grid_points[wr.index] if wr.index < len(grid_points) else {}
            results.append({
                "coords": pt,
                "value": np.nan,
                "success": False,
                "error": wr.error,
            })
    return results


def parallel_ntk_widths(
    ntk_fn: Callable[[int], np.ndarray],
    widths: Sequence[int],
    config: Optional[ParallelConfig] = None,
) -> Dict[int, np.ndarray]:
    """Compute NTKs at multiple widths in parallel.

    Parameters
    ----------
    ntk_fn : callable
        Function mapping width → NTK matrix.
    widths : sequence of int
        Widths to evaluate.
    config : ParallelConfig, optional

    Returns
    -------
    Dict mapping width → NTK array.
    """
    if config is None:
        config = ParallelConfig()

    worker_results = parallel_map(
        ntk_fn, list(widths), config, label="NTK widths"
    )

    result: Dict[int, np.ndarray] = {}
    for wr, w in zip(worker_results, widths):
        if wr.success:
            result[w] = wr.value
        else:
            _log.warning("NTK computation failed for width=%d: %s", w, wr.error)
    return result


def parallel_training(
    train_fn: Callable[[int], Any],
    seeds: Sequence[int],
    config: Optional[ParallelConfig] = None,
) -> List[Any]:
    """Run ground-truth training across multiple seeds in parallel.

    Parameters
    ----------
    train_fn : callable
        Function mapping seed → training result.
    seeds : sequence of int
        Random seeds.
    config : ParallelConfig, optional

    Returns
    -------
    List of training results (None for failed runs).
    """
    if config is None:
        config = ParallelConfig()

    worker_results = parallel_map(
        train_fn, list(seeds), config, label="training"
    )

    results = []
    for wr, seed in zip(worker_results, seeds):
        if wr.success:
            results.append(wr.value)
        else:
            _log.warning("Training failed for seed=%d: %s", seed, wr.error)
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def collect_results(
    worker_results: List[WorkerResult],
) -> Tuple[List[Any], List[str]]:
    """Separate successful values from error messages."""
    values = []
    errors = []
    for wr in worker_results:
        if wr.success:
            values.append(wr.value)
        else:
            errors.append(wr.error or "unknown error")
    return values, errors


def aggregate_timing(
    worker_results: List[WorkerResult],
) -> Dict[str, float]:
    """Compute timing statistics across workers."""
    times = [wr.elapsed for wr in worker_results if wr.elapsed > 0]
    if not times:
        return {"total": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0}
    arr = np.array(times)
    return {
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "std": float(arr.std()) if len(arr) > 1 else 0.0,
    }


def success_rate(worker_results: List[WorkerResult]) -> float:
    """Fraction of successful results."""
    if not worker_results:
        return 0.0
    return sum(1 for wr in worker_results if wr.success) / len(worker_results)
