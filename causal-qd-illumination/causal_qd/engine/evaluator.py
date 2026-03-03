"""Batch evaluation of DAG candidates with caching, timeout, and parallelism.

Provides :class:`BatchEvaluator` which scores and computes behavioural
descriptors for batches of DAG candidates.  Supports serial execution,
``multiprocessing``-based parallelism, timeouts, and result caching.
"""

from __future__ import annotations

import logging
import signal
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    DataMatrix,
    QualityScore,
)

logger = logging.getLogger(__name__)

# Type aliases for the pluggable functions
ScoreFn = Callable[[AdjacencyMatrix, DataMatrix], float]
DescriptorFn = Callable[[AdjacencyMatrix, DataMatrix], BehavioralDescriptor]


def _evaluate_single(
    dag: AdjacencyMatrix,
    data: DataMatrix,
    score_fn: ScoreFn,
    descriptor_fn: DescriptorFn,
) -> Tuple[QualityScore, BehavioralDescriptor]:
    """Evaluate a single DAG (top-level function for pickling).

    Parameters
    ----------
    dag : AdjacencyMatrix
    data : DataMatrix
    score_fn : ScoreFn
    descriptor_fn : DescriptorFn

    Returns
    -------
    Tuple[QualityScore, BehavioralDescriptor]
    """
    quality = score_fn(dag, data)
    descriptor = descriptor_fn(dag, data)
    return quality, descriptor


# ---------------------------------------------------------------------------
# LRU cache for evaluation results
# ---------------------------------------------------------------------------

class _EvalCache:
    """Bounded LRU cache mapping DAG hash → (quality, descriptor).

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[int, Tuple[QualityScore, BehavioralDescriptor]] = (
            OrderedDict()
        )
        self._hits: int = 0
        self._misses: int = 0

    def _key(self, dag: AdjacencyMatrix) -> bytes:
        """Compute a hash key for the adjacency matrix."""
        return hash(dag.tobytes())

    def get(
        self, dag: AdjacencyMatrix
    ) -> Optional[Tuple[QualityScore, BehavioralDescriptor]]:
        """Lookup a cached result.

        Parameters
        ----------
        dag : AdjacencyMatrix

        Returns
        -------
        Tuple[QualityScore, BehavioralDescriptor] or None
        """
        key = self._key(dag)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return result
        self._misses += 1
        return None

    def put(
        self,
        dag: AdjacencyMatrix,
        result: Tuple[QualityScore, BehavioralDescriptor],
    ) -> None:
        """Store a result in the cache.

        Parameters
        ----------
        dag : AdjacencyMatrix
        result : Tuple[QualityScore, BehavioralDescriptor]
        """
        key = self._key(dag)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate in [0, 1]."""
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# BatchEvaluator
# ---------------------------------------------------------------------------

class BatchEvaluator:
    """Evaluate a batch of DAG candidates, optionally in parallel.

    Features
    --------
    - Serial and multiprocessing modes.
    - LRU result cache to avoid redundant evaluations.
    - Per-candidate timeout support.
    - Error handling with configurable fallback values.
    - Evaluation statistics (timing, cache hits, failures).

    Parameters
    ----------
    score_fn : ScoreFn
        Scoring function ``(adjacency_matrix, data) -> float``.
    descriptor_fn : DescriptorFn
        Descriptor function ``(adjacency_matrix, data) -> BehavioralDescriptor``.
    n_workers : int
        Number of parallel workers.  ``1`` (default) disables
        parallelism; ``0`` or ``None`` uses ``os.cpu_count()``.
    cache_size : int
        Maximum number of cached evaluation results (0 = disabled).
    timeout : float or None
        Per-candidate timeout in seconds (``None`` = no timeout).
        Only effective in parallel mode.
    default_quality : float
        Quality returned when evaluation fails.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        descriptor_fn: DescriptorFn,
        n_workers: int = 1,
        cache_size: int = 10_000,
        timeout: Optional[float] = None,
        default_quality: float = float("-inf"),
    ) -> None:
        self._score_fn = score_fn
        self._descriptor_fn = descriptor_fn
        self._n_workers = n_workers
        self._timeout = timeout
        self._default_quality = default_quality

        # Result cache
        self._cache: Optional[_EvalCache] = None
        if cache_size > 0:
            self._cache = _EvalCache(max_size=cache_size)

        # Statistics
        self._total_evaluated: int = 0
        self._total_failed: int = 0
        self._total_cached: int = 0
        self._total_time: float = 0.0

    # -- public API ----------------------------------------------------------

    def evaluate(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Score and compute descriptors for each DAG.

        Parameters
        ----------
        dags : List[AdjacencyMatrix]
            List of adjacency matrices to evaluate.
        data : DataMatrix
            Observed data matrix (N × p).

        Returns
        -------
        List[Tuple[QualityScore, BehavioralDescriptor]]
            Parallel list of ``(quality, descriptor)`` pairs.
        """
        if not dags:
            return []

        start = time.monotonic()

        # Check cache first
        results: List[Optional[Tuple[QualityScore, BehavioralDescriptor]]] = [
            None
        ] * len(dags)
        uncached_indices: List[int] = []

        for i, dag in enumerate(dags):
            if self._cache is not None:
                cached = self._cache.get(dag)
                if cached is not None:
                    results[i] = cached
                    self._total_cached += 1
                    continue
            uncached_indices.append(i)

        # Evaluate uncached
        if uncached_indices:
            uncached_dags = [dags[i] for i in uncached_indices]
            if self._n_workers == 1:
                uncached_results = self._evaluate_sequential(uncached_dags, data)
            else:
                uncached_results = self._evaluate_parallel(uncached_dags, data)

            for j, idx in enumerate(uncached_indices):
                results[idx] = uncached_results[j]
                if self._cache is not None:
                    self._cache.put(dags[idx], uncached_results[j])

        self._total_evaluated += len(dags)
        self._total_time += time.monotonic() - start

        return [r for r in results if r is not None]  # type: ignore[misc]

    def evaluate_single(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
    ) -> Tuple[QualityScore, BehavioralDescriptor]:
        """Evaluate a single DAG.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix

        Returns
        -------
        Tuple[QualityScore, BehavioralDescriptor]
        """
        results = self.evaluate([dag], data)
        return results[0]

    # -- statistics ----------------------------------------------------------

    @property
    def total_evaluated(self) -> int:
        """Total number of DAGs submitted for evaluation."""
        return self._total_evaluated

    @property
    def total_failed(self) -> int:
        """Total number of failed evaluations."""
        return self._total_failed

    @property
    def total_cached(self) -> int:
        """Total number of cache hits."""
        return self._total_cached

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        if self._cache is None:
            return 0.0
        return self._cache.hit_rate

    @property
    def mean_eval_time(self) -> float:
        """Mean time per evaluation call in seconds."""
        if self._total_evaluated == 0:
            return 0.0
        return self._total_time / self._total_evaluated

    def stats(self) -> Dict[str, float]:
        """Return evaluation statistics as a dictionary."""
        return {
            "total_evaluated": float(self._total_evaluated),
            "total_failed": float(self._total_failed),
            "total_cached": float(self._total_cached),
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size": float(self._cache.size if self._cache else 0),
            "mean_eval_time": self.mean_eval_time,
            "total_time": self._total_time,
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        if self._cache is not None:
            self._cache.clear()

    # -- internal ------------------------------------------------------------

    def _default_result(
        self, n_desc_dims: int
    ) -> Tuple[QualityScore, BehavioralDescriptor]:
        """Return a fallback result for failed evaluations."""
        return (self._default_quality, np.zeros(n_desc_dims, dtype=np.float64))

    def _evaluate_sequential(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Evaluate DAGs one at a time in the current process."""
        results: List[Tuple[QualityScore, BehavioralDescriptor]] = []
        for dag in dags:
            try:
                quality = self._score_fn(dag, data)
                descriptor = self._descriptor_fn(dag, data)
                results.append((quality, descriptor))
            except Exception:
                logger.debug("Sequential evaluation failed; using default.")
                self._total_failed += 1
                results.append(self._default_result(data.shape[1]))
        return results

    def _evaluate_parallel(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Evaluate DAGs in parallel using a process pool."""
        n_workers = self._n_workers
        if n_workers is None or n_workers <= 0:
            import os
            n_workers = os.cpu_count() or 1

        results: List[Optional[Tuple[QualityScore, BehavioralDescriptor]]] = [
            None
        ] * len(dags)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _evaluate_single,
                    dag,
                    data,
                    self._score_fn,
                    self._descriptor_fn,
                ): i
                for i, dag in enumerate(dags)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=self._timeout)
                    results[idx] = result
                except TimeoutError:
                    logger.warning(
                        "Evaluation timed out for DAG at index %d", idx
                    )
                    self._total_failed += 1
                    results[idx] = self._default_result(data.shape[1])
                except Exception:
                    logger.exception(
                        "Evaluation failed for DAG at index %d", idx
                    )
                    self._total_failed += 1
                    results[idx] = self._default_result(data.shape[1])

        return [r for r in results if r is not None]  # type: ignore[misc]
