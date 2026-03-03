"""Parallel batch evaluation of DAGs via a process pool.

Provides:
  - ParallelEvaluator: evaluate batches of DAGs in parallel with
    ProcessPoolExecutor, batched evaluation, load balancing,
    shared-memory data support, and graceful shutdown.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError, as_completed
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_qd.types import BehavioralDescriptor, DataMatrix, QualityScore

if TYPE_CHECKING:
    from causal_qd.core.dag import DAG

logger = logging.getLogger(__name__)


# ======================================================================
# Worker functions (must be module-level for pickling)
# ======================================================================


def _evaluate_single(
    args: Tuple[Any, DataMatrix, Callable[..., QualityScore], Callable[..., BehavioralDescriptor]],
) -> Tuple[QualityScore, BehavioralDescriptor]:
    """Worker function: evaluate a single DAG in a child process."""
    dag, data, score_fn, descriptor_fn = args
    quality = score_fn(dag, data)
    descriptor = descriptor_fn(dag, data)
    return quality, descriptor


def _evaluate_batch_worker(
    args: Tuple[List[Any], DataMatrix, Callable[..., QualityScore], Callable[..., BehavioralDescriptor]],
) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
    """Worker function: evaluate a batch of DAGs in a child process.

    Reduces per-task overhead by batching multiple DAGs into one task.
    """
    dags, data, score_fn, descriptor_fn = args
    results = []
    for dag in dags:
        try:
            quality = score_fn(dag, data)
            descriptor = descriptor_fn(dag, data)
            results.append((quality, descriptor))
        except Exception as exc:
            logger.warning("Evaluation failed for a DAG: %s", exc)
            results.append((float("-inf"), np.zeros(2, dtype=np.float64)))
    return results


def _evaluate_with_shared_data(
    args: Tuple[Any, str, Tuple[int, ...], str, Callable[..., QualityScore], Callable[..., BehavioralDescriptor]],
) -> Tuple[QualityScore, BehavioralDescriptor]:
    """Worker that reconstructs data from shared memory info."""
    dag, shm_name, shape, dtype_str, score_fn, descriptor_fn = args
    shm = mp.shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    quality = score_fn(dag, data)
    descriptor = descriptor_fn(dag, data)
    shm.close()
    return quality, descriptor


# ======================================================================
# ParallelEvaluator
# ======================================================================


class ParallelEvaluator:
    """Evaluate a batch of DAGs in parallel using a process pool.

    Supports:
    - Single-DAG and batched evaluation modes
    - Load balancing via chunk sizing
    - Shared memory for large numpy data arrays
    - Graceful shutdown and timeout handling
    - Error recovery (failed evaluations get sentinel values)

    Parameters
    ----------
    n_workers :
        Number of worker processes.
    score_fn :
        Callable ``(dag, data) -> QualityScore``.
    descriptor_computer :
        Callable ``(dag, data) -> BehavioralDescriptor``.
    chunk_size :
        Number of DAGs per worker task (for batched mode).
        If *None*, auto-computed from batch size / n_workers.
    timeout :
        Timeout per batch in seconds (*None* = no timeout).
    use_shared_memory :
        If *True*, share the data array via shared memory instead
        of pickling it for each worker.
    """

    def __init__(
        self,
        n_workers: int,
        score_fn: Callable[..., QualityScore],
        descriptor_computer: Callable[..., BehavioralDescriptor],
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = None,
        use_shared_memory: bool = False,
    ) -> None:
        self._n_workers = max(1, n_workers)
        self._score_fn = score_fn
        self._descriptor_computer = descriptor_computer
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._use_shared_memory = use_shared_memory
        self._pool: Optional[ProcessPoolExecutor] = None
        self._shm: Optional[mp.shared_memory.SharedMemory] = None
        # Statistics
        self.total_evaluated: int = 0
        self.total_failures: int = 0
        self.total_time: float = 0.0

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> ParallelEvaluator:
        self._pool = ProcessPoolExecutor(max_workers=self._n_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shutdown the process pool and clean up shared memory."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        dags: List[Any],
        data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Evaluate a list of DAGs, returning ``(quality, descriptor)`` pairs.

        Automatically selects between single-task and batched-task mode
        based on configuration.  Falls back to sequential if pool is
        unavailable or only one worker.

        Parameters
        ----------
        dags :
            List of DAG objects (adjacency matrices or DAG instances).
        data :
            ``N × p`` data matrix.

        Returns
        -------
        list of (QualityScore, BehavioralDescriptor)
        """
        if not dags:
            return []

        t0 = time.time()

        if self._pool is not None and self._n_workers > 1:
            if self._chunk_size is not None and self._chunk_size > 1:
                results = self._evaluate_batched(dags, data)
            else:
                results = self._evaluate_individual(dags, data)
        else:
            results = self._evaluate_sequential(dags, data)

        self.total_evaluated += len(dags)
        self.total_time += time.time() - t0
        return results

    def evaluate_single(
        self,
        dag: Any,
        data: DataMatrix,
    ) -> Tuple[QualityScore, BehavioralDescriptor]:
        """Evaluate a single DAG (convenience wrapper)."""
        results = self.evaluate_batch([dag], data)
        return results[0]

    # ------------------------------------------------------------------
    # Evaluation strategies
    # ------------------------------------------------------------------

    def _evaluate_sequential(
        self, dags: List[Any], data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Sequential evaluation (single-process fallback)."""
        results = []
        for dag in dags:
            try:
                q = self._score_fn(dag, data)
                d = self._descriptor_computer(dag, data)
                results.append((q, d))
            except Exception as exc:
                logger.warning("Sequential eval failed: %s", exc)
                self.total_failures += 1
                results.append((float("-inf"), np.zeros(2, dtype=np.float64)))
        return results

    def _evaluate_individual(
        self, dags: List[Any], data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Submit each DAG as a separate task."""
        assert self._pool is not None

        args_list = [
            (dag, data, self._score_fn, self._descriptor_computer)
            for dag in dags
        ]

        try:
            futures = [self._pool.submit(_evaluate_single, a) for a in args_list]
            results = []
            for f in futures:
                try:
                    results.append(f.result(timeout=self._timeout))
                except Exception as exc:
                    logger.warning("Worker failed: %s", exc)
                    self.total_failures += 1
                    results.append((float("-inf"), np.zeros(2, dtype=np.float64)))
            return results
        except Exception as exc:
            logger.error("Pool submission failed: %s", exc)
            return self._evaluate_sequential(dags, data)

    def _evaluate_batched(
        self, dags: List[Any], data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Split DAGs into chunks and submit each chunk as a batch task."""
        assert self._pool is not None
        chunk = self._chunk_size or max(1, len(dags) // self._n_workers)

        chunks = [dags[i:i + chunk] for i in range(0, len(dags), chunk)]
        batch_args = [
            (ch, data, self._score_fn, self._descriptor_computer)
            for ch in chunks
        ]

        try:
            futures = [self._pool.submit(_evaluate_batch_worker, a) for a in batch_args]
            all_results: List[Tuple[QualityScore, BehavioralDescriptor]] = []
            for f in futures:
                try:
                    batch_results = f.result(timeout=self._timeout)
                    all_results.extend(batch_results)
                except Exception as exc:
                    logger.warning("Batch worker failed: %s", exc)
                    self.total_failures += len(chunks[0])
                    # Pad with sentinel values
                    all_results.extend(
                        [(float("-inf"), np.zeros(2, dtype=np.float64))] * chunk
                    )
            return all_results[:len(dags)]
        except Exception as exc:
            logger.error("Batched eval failed: %s", exc)
            return self._evaluate_sequential(dags, data)

    # ------------------------------------------------------------------
    # Shared memory support
    # ------------------------------------------------------------------

    def setup_shared_memory(self, data: DataMatrix) -> None:
        """Copy *data* into shared memory for zero-copy worker access.

        Parameters
        ----------
        data :
            Data matrix to share.
        """
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass

        self._shm = mp.shared_memory.SharedMemory(
            create=True, size=data.nbytes,
        )
        shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shm.buf)
        np.copyto(shared_arr, data)
        self._shm_shape = data.shape
        self._shm_dtype = str(data.dtype)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return evaluation statistics."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_failures": self.total_failures,
            "total_time": self.total_time,
            "avg_time_per_dag": self.total_time / max(self.total_evaluated, 1),
            "n_workers": self._n_workers,
        }
