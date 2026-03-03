"""Batch operations on sets of DAGs for high-throughput MAP-Elites.

This module provides vectorized batch processing of multiple DAGs
simultaneously, including scoring, descriptor computation, mutation,
crossover, and archive updates.  It is the primary performance layer
for the MAP-Elites inner loop.

Key classes
-----------
* :class:`BatchDAGOperator` – score / describe / mutate / crossover batches
* :class:`VectorizedArchiveUpdate` – batch archive insertion
* :class:`ParallelBatchProcessor` – distribute batches across processes
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    CellIndex,
    DataMatrix,
    QualityScore,
)

__all__ = [
    "BatchDAGOperator",
    "VectorizedArchiveUpdate",
    "ParallelBatchProcessor",
    "BatchResult",
    "batch_score_dags",
    "batch_compute_descriptors",
]


# ---------------------------------------------------------------------------
# Protocol definitions for pluggable components
# ---------------------------------------------------------------------------


class ScoreFnProtocol(Protocol):
    """Protocol for scoring functions."""

    def __call__(
        self, dag: AdjacencyMatrix, data: DataMatrix
    ) -> QualityScore: ...


class DescriptorFnProtocol(Protocol):
    """Protocol for descriptor computation functions."""

    def __call__(
        self, dag: AdjacencyMatrix, data: Optional[DataMatrix] = None
    ) -> BehavioralDescriptor: ...


class MutationFnProtocol(Protocol):
    """Protocol for mutation operators."""

    def __call__(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix: ...


class CrossoverFnProtocol(Protocol):
    """Protocol for crossover operators."""

    def __call__(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """Container for batch evaluation results.

    Attributes
    ----------
    qualities : np.ndarray
        Array of shape ``(batch_size,)`` with quality scores.
    descriptors : np.ndarray
        Array of shape ``(batch_size, descriptor_dim)`` with descriptors.
    valid_mask : np.ndarray
        Boolean array indicating which DAGs were valid (no errors).
    timings : dict
        Timing breakdown for profiling.
    """

    qualities: np.ndarray
    descriptors: np.ndarray
    valid_mask: np.ndarray
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.qualities)

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def mean_quality(self) -> float:
        if self.n_valid == 0:
            return float("-inf")
        return float(np.mean(self.qualities[self.valid_mask]))

    @property
    def best_quality(self) -> float:
        if self.n_valid == 0:
            return float("-inf")
        return float(np.max(self.qualities[self.valid_mask]))


# ---------------------------------------------------------------------------
# BatchDAGOperator
# ---------------------------------------------------------------------------


class BatchDAGOperator:
    """Batch operations on sets of DAGs: score, describe, mutate, crossover.

    Wraps individual scoring/descriptor/mutation/crossover functions and
    applies them to lists of DAGs, collecting results into vectorized
    numpy arrays for efficient downstream processing.

    Parameters
    ----------
    score_fn : callable
        ``(dag, data) -> float`` scoring function.
    descriptor_fn : callable
        ``(dag, data) -> np.ndarray`` descriptor function.
    mutation_fns : list of callables, optional
        ``(dag, rng) -> dag`` mutation operators.
    crossover_fns : list of callables, optional
        ``(parent1, parent2, rng) -> (child1, child2)`` crossover operators.
    mutation_probs : np.ndarray, optional
        Selection probabilities for mutation operators.
    crossover_probs : np.ndarray, optional
        Selection probabilities for crossover operators.

    Examples
    --------
    >>> op = BatchDAGOperator(bic_score, structural_descriptor)
    >>> result = op.batch_score(dags, data)
    >>> result.qualities.shape
    (100,)
    """

    def __init__(
        self,
        score_fn: Callable[..., QualityScore],
        descriptor_fn: Callable[..., BehavioralDescriptor],
        mutation_fns: Optional[List[Callable]] = None,
        crossover_fns: Optional[List[Callable]] = None,
        mutation_probs: Optional[np.ndarray] = None,
        crossover_probs: Optional[np.ndarray] = None,
    ) -> None:
        self._score_fn = score_fn
        self._descriptor_fn = descriptor_fn
        self._mutation_fns = mutation_fns or []
        self._crossover_fns = crossover_fns or []

        n_mut = len(self._mutation_fns)
        n_cx = len(self._crossover_fns)
        self._mutation_probs = (
            np.asarray(mutation_probs, dtype=np.float64)
            if mutation_probs is not None
            else np.ones(n_mut) / max(n_mut, 1)
        )
        self._crossover_probs = (
            np.asarray(crossover_probs, dtype=np.float64)
            if crossover_probs is not None
            else np.ones(n_cx) / max(n_cx, 1)
        )

        # Statistics
        self._total_scored = 0
        self._total_mutated = 0
        self._total_crossovers = 0
        self._total_score_time = 0.0
        self._total_descriptor_time = 0.0

    # -- Batch scoring ------------------------------------------------------

    def batch_score(
        self, dags: Sequence[AdjacencyMatrix], data: DataMatrix
    ) -> BatchResult:
        """Score multiple DAGs simultaneously.

        Parameters
        ----------
        dags : sequence of np.ndarray
            List of adjacency matrices.
        data : np.ndarray
            ``(N, p)`` data matrix.

        Returns
        -------
        BatchResult
            Contains qualities, descriptors, and validity mask.
        """
        batch_size = len(dags)
        qualities = np.full(batch_size, float("-inf"), dtype=np.float64)
        valid_mask = np.zeros(batch_size, dtype=np.bool_)

        t0 = time.perf_counter()
        for i, dag in enumerate(dags):
            try:
                qualities[i] = self._score_fn(dag, data)
                valid_mask[i] = True
            except Exception:
                pass
        score_time = time.perf_counter() - t0

        self._total_scored += batch_size
        self._total_score_time += score_time

        return BatchResult(
            qualities=qualities,
            descriptors=np.empty((batch_size, 0)),
            valid_mask=valid_mask,
            timings={"score": score_time},
        )

    def batch_descriptor(
        self,
        dags: Sequence[AdjacencyMatrix],
        data: Optional[DataMatrix] = None,
    ) -> BatchResult:
        """Compute descriptors for multiple DAGs.

        Parameters
        ----------
        dags : sequence of np.ndarray
            List of adjacency matrices.
        data : np.ndarray, optional
            ``(N, p)`` data matrix (some descriptors need it).

        Returns
        -------
        BatchResult
            Contains descriptors and validity mask.
        """
        batch_size = len(dags)
        valid_mask = np.zeros(batch_size, dtype=np.bool_)
        descriptors_list: List[np.ndarray] = []

        t0 = time.perf_counter()
        for i, dag in enumerate(dags):
            try:
                desc = self._descriptor_fn(dag, data)
                descriptors_list.append(np.asarray(desc, dtype=np.float64))
                valid_mask[i] = True
            except Exception:
                descriptors_list.append(np.array([], dtype=np.float64))
        desc_time = time.perf_counter() - t0

        self._total_descriptor_time += desc_time

        # Stack into matrix
        if descriptors_list and valid_mask.any():
            first_valid = next(i for i in range(batch_size) if valid_mask[i])
            dim = descriptors_list[first_valid].shape[0]
            descriptors = np.full(
                (batch_size, dim), np.nan, dtype=np.float64
            )
            for i in range(batch_size):
                if valid_mask[i] and descriptors_list[i].shape[0] == dim:
                    descriptors[i] = descriptors_list[i]
        else:
            descriptors = np.empty((batch_size, 0), dtype=np.float64)

        return BatchResult(
            qualities=np.zeros(batch_size, dtype=np.float64),
            descriptors=descriptors,
            valid_mask=valid_mask,
            timings={"descriptor": desc_time},
        )

    def batch_evaluate(
        self, dags: Sequence[AdjacencyMatrix], data: DataMatrix
    ) -> BatchResult:
        """Score and compute descriptors for a batch in one pass.

        Parameters
        ----------
        dags : sequence of np.ndarray
            List of adjacency matrices.
        data : np.ndarray
            ``(N, p)`` data matrix.

        Returns
        -------
        BatchResult
            Contains both qualities and descriptors.
        """
        batch_size = len(dags)
        qualities = np.full(batch_size, float("-inf"), dtype=np.float64)
        valid_mask = np.zeros(batch_size, dtype=np.bool_)
        descriptors_list: List[np.ndarray] = []

        t_score = 0.0
        t_desc = 0.0

        for i, dag in enumerate(dags):
            try:
                t0 = time.perf_counter()
                qualities[i] = self._score_fn(dag, data)
                t_score += time.perf_counter() - t0

                t0 = time.perf_counter()
                desc = self._descriptor_fn(dag, data)
                t_desc += time.perf_counter() - t0

                descriptors_list.append(np.asarray(desc, dtype=np.float64))
                valid_mask[i] = True
            except Exception:
                descriptors_list.append(np.array([], dtype=np.float64))

        self._total_scored += batch_size
        self._total_score_time += t_score
        self._total_descriptor_time += t_desc

        # Stack descriptors
        if descriptors_list and valid_mask.any():
            first_valid = next(i for i in range(batch_size) if valid_mask[i])
            dim = descriptors_list[first_valid].shape[0]
            descriptors = np.full(
                (batch_size, dim), np.nan, dtype=np.float64
            )
            for i in range(batch_size):
                if valid_mask[i] and descriptors_list[i].shape[0] == dim:
                    descriptors[i] = descriptors_list[i]
        else:
            descriptors = np.empty((batch_size, 0), dtype=np.float64)

        return BatchResult(
            qualities=qualities,
            descriptors=descriptors,
            valid_mask=valid_mask,
            timings={"score": t_score, "descriptor": t_desc},
        )

    # -- Batch mutation/crossover -------------------------------------------

    def batch_mutate(
        self,
        dags: Sequence[AdjacencyMatrix],
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Apply mutations to multiple DAGs.

        Each DAG is mutated by a randomly selected mutation operator.

        Parameters
        ----------
        dags : sequence of np.ndarray
            Parent DAGs.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        list of np.ndarray
            Mutated offspring DAGs.
        """
        if not self._mutation_fns:
            return [dag.copy() for dag in dags]

        if rng is None:
            rng = np.random.default_rng()

        results: List[AdjacencyMatrix] = []
        n_ops = len(self._mutation_fns)
        op_indices = rng.choice(n_ops, size=len(dags), p=self._mutation_probs)

        for i, dag in enumerate(dags):
            try:
                mutated = self._mutation_fns[op_indices[i]](dag, rng)
                results.append(mutated)
            except Exception:
                results.append(dag.copy())

        self._total_mutated += len(dags)
        return results

    def batch_crossover(
        self,
        parents1: Sequence[AdjacencyMatrix],
        parents2: Sequence[AdjacencyMatrix],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[AdjacencyMatrix], List[AdjacencyMatrix]]:
        """Perform multiple crossovers.

        Parameters
        ----------
        parents1, parents2 : sequence of np.ndarray
            Paired parent DAGs.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        tuple of (list of np.ndarray, list of np.ndarray)
            Two lists of offspring DAGs.
        """
        if not self._crossover_fns:
            return (
                [p.copy() for p in parents1],
                [p.copy() for p in parents2],
            )

        if rng is None:
            rng = np.random.default_rng()

        children1: List[AdjacencyMatrix] = []
        children2: List[AdjacencyMatrix] = []
        n_ops = len(self._crossover_fns)
        op_indices = rng.choice(
            n_ops, size=len(parents1), p=self._crossover_probs
        )

        for i in range(len(parents1)):
            try:
                c1, c2 = self._crossover_fns[op_indices[i]](
                    parents1[i], parents2[i], rng
                )
                children1.append(c1)
                children2.append(c2)
            except Exception:
                children1.append(parents1[i].copy())
                children2.append(parents2[i].copy())

        self._total_crossovers += len(parents1)
        return children1, children2

    # -- Statistics ---------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        return {
            "total_scored": self._total_scored,
            "total_mutated": self._total_mutated,
            "total_crossovers": self._total_crossovers,
            "total_score_time": self._total_score_time,
            "total_descriptor_time": self._total_descriptor_time,
            "avg_score_time": (
                self._total_score_time / self._total_scored
                if self._total_scored > 0
                else 0.0
            ),
            "avg_descriptor_time": (
                self._total_descriptor_time / self._total_scored
                if self._total_scored > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all counters."""
        self._total_scored = 0
        self._total_mutated = 0
        self._total_crossovers = 0
        self._total_score_time = 0.0
        self._total_descriptor_time = 0.0


# ---------------------------------------------------------------------------
# VectorizedArchiveUpdate
# ---------------------------------------------------------------------------


class VectorizedArchiveUpdate:
    """Batch archive update with vectorized cell computation and quality comparison.

    This class computes archive cell indices and quality comparisons for
    entire batches at once using numpy vectorization, minimizing Python loop
    overhead.

    Parameters
    ----------
    dims : tuple of int
        Grid dimensions for each descriptor axis.
    lower_bounds : np.ndarray
        Lower bounds for each descriptor dimension.
    upper_bounds : np.ndarray
        Upper bounds for each descriptor dimension.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        self._dims = np.asarray(dims, dtype=np.int64)
        self._lower = np.asarray(lower_bounds, dtype=np.float64)
        self._upper = np.asarray(upper_bounds, dtype=np.float64)
        self._d = len(dims)
        self._total_cells = int(np.prod(self._dims))

        # Cell widths for fast index computation
        self._widths = (self._upper - self._lower) / self._dims.astype(
            np.float64
        )

        # Storage: flat arrays indexed by linearized cell index
        self._qualities = np.full(self._total_cells, float("-inf"), dtype=np.float64)
        self._solutions: Dict[int, AdjacencyMatrix] = {}
        self._descriptors_store: Dict[int, np.ndarray] = {}
        self._metadata_store: Dict[int, Dict[str, Any]] = {}

        # Statistics
        self._total_attempts = 0
        self._total_improvements = 0
        self._total_fills = 0

    # -- Vectorized cell computation ----------------------------------------

    def _descriptors_to_cells(
        self, descriptors: np.ndarray
    ) -> np.ndarray:
        """Map batch of descriptors to linearized cell indices.

        Parameters
        ----------
        descriptors : np.ndarray
            ``(batch, d)`` array of descriptor vectors.

        Returns
        -------
        np.ndarray
            ``(batch,)`` array of linearized cell indices.
        """
        # Clip to bounds
        clipped = np.clip(descriptors, self._lower, self._upper - 1e-12)
        # Compute per-dimension cell index
        cell_coords = ((clipped - self._lower) / self._widths).astype(np.int64)
        cell_coords = np.clip(cell_coords, 0, self._dims - 1)

        # Linearize
        multipliers = np.ones(self._d, dtype=np.int64)
        for i in range(self._d - 2, -1, -1):
            multipliers[i] = multipliers[i + 1] * self._dims[i + 1]

        return (cell_coords * multipliers).sum(axis=1)

    def _cell_to_coords(self, flat_index: int) -> Tuple[int, ...]:
        """Convert linearized cell index to coordinate tuple."""
        coords = []
        remaining = flat_index
        for i in range(self._d):
            div = int(np.prod(self._dims[i + 1 :]))
            coords.append(remaining // div)
            remaining %= div
        return tuple(coords)

    # -- Batch add ----------------------------------------------------------

    def batch_add(
        self,
        solutions: Sequence[AdjacencyMatrix],
        qualities: np.ndarray,
        descriptors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Add multiple solutions to the archive at once.

        Steps:
        1. Vectorized cell index computation for all descriptors.
        2. For each cell, compare quality against incumbent.
        3. Update in-place where improvement occurs.

        Parameters
        ----------
        solutions : sequence of np.ndarray
            Adjacency matrices.
        qualities : np.ndarray
            ``(batch,)`` quality scores.
        descriptors : np.ndarray
            ``(batch, d)`` descriptor vectors.
        metadata : list of dict, optional
            Per-solution metadata.

        Returns
        -------
        np.ndarray
            Boolean array indicating which solutions were added.
        """
        batch_size = len(solutions)
        self._total_attempts += batch_size

        cell_indices = self._descriptors_to_cells(descriptors)
        added = np.zeros(batch_size, dtype=np.bool_)

        # Vectorized quality comparison
        incumbent_qualities = self._qualities[cell_indices]
        improvement_mask = qualities > incumbent_qualities

        for i in range(batch_size):
            if improvement_mask[i]:
                ci = int(cell_indices[i])
                was_empty = ci not in self._solutions
                self._qualities[ci] = qualities[i]
                self._solutions[ci] = solutions[i].copy()
                self._descriptors_store[ci] = descriptors[i].copy()
                if metadata is not None:
                    self._metadata_store[ci] = metadata[i]
                added[i] = True
                if was_empty:
                    self._total_fills += 1
                else:
                    self._total_improvements += 1

        return added

    # -- Query methods ------------------------------------------------------

    def get(self, cell_index: int) -> Optional[Tuple[AdjacencyMatrix, float, np.ndarray]]:
        """Get solution at a specific cell."""
        if cell_index not in self._solutions:
            return None
        return (
            self._solutions[cell_index],
            float(self._qualities[cell_index]),
            self._descriptors_store[cell_index],
        )

    def sample(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> List[Tuple[AdjacencyMatrix, float, np.ndarray]]:
        """Sample *n* solutions uniformly from occupied cells."""
        if rng is None:
            rng = np.random.default_rng()
        occupied = list(self._solutions.keys())
        if not occupied:
            return []
        indices = rng.choice(occupied, size=min(n, len(occupied)), replace=True)
        return [
            (
                self._solutions[i],
                float(self._qualities[i]),
                self._descriptors_store[i],
            )
            for i in indices
        ]

    @property
    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        return len(self._solutions) / self._total_cells

    @property
    def qd_score(self) -> float:
        """Sum of qualities over occupied cells."""
        if not self._solutions:
            return 0.0
        occupied_indices = np.array(list(self._solutions.keys()))
        return float(np.sum(self._qualities[occupied_indices]))

    @property
    def num_elites(self) -> int:
        """Number of occupied cells."""
        return len(self._solutions)

    @property
    def best_quality(self) -> float:
        """Best quality score across all occupied cells."""
        if not self._solutions:
            return float("-inf")
        occupied_indices = np.array(list(self._solutions.keys()))
        return float(np.max(self._qualities[occupied_indices]))

    def quality_grid(self) -> np.ndarray:
        """Return quality values as a grid array."""
        return self._qualities.reshape(tuple(self._dims))

    def occupancy_grid(self) -> np.ndarray:
        """Return boolean occupancy grid."""
        occ = np.zeros(self._total_cells, dtype=np.bool_)
        for ci in self._solutions:
            occ[ci] = True
        return occ.reshape(tuple(self._dims))

    def stats(self) -> Dict[str, Any]:
        """Return archive statistics."""
        return {
            "total_cells": self._total_cells,
            "occupied_cells": self.num_elites,
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "best_quality": self.best_quality,
            "total_attempts": self._total_attempts,
            "total_fills": self._total_fills,
            "total_improvements": self._total_improvements,
        }

    def clear(self) -> None:
        """Remove all solutions from the archive."""
        self._qualities[:] = float("-inf")
        self._solutions.clear()
        self._descriptors_store.clear()
        self._metadata_store.clear()
        self._total_attempts = 0
        self._total_improvements = 0
        self._total_fills = 0


# ---------------------------------------------------------------------------
# ParallelBatchProcessor
# ---------------------------------------------------------------------------


def _worker_evaluate(
    args: Tuple[List[np.ndarray], np.ndarray, Any, Any],
) -> List[Tuple[float, np.ndarray]]:
    """Worker function for parallel batch evaluation (module-level for pickle)."""
    dags, data, score_fn, descriptor_fn = args
    results = []
    for dag in dags:
        try:
            q = score_fn(dag, data)
            d = descriptor_fn(dag, data)
            results.append((float(q), np.asarray(d, dtype=np.float64)))
        except Exception:
            results.append((float("-inf"), np.array([], dtype=np.float64)))
    return results


class ParallelBatchProcessor:
    """Distribute batch operations across processes.

    Uses ``ProcessPoolExecutor`` for CPU-bound operations and chunks the
    work evenly across workers.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes. Defaults to CPU count.
    chunk_size : int, optional
        Number of DAGs per worker chunk. If None, divides evenly.
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        self._n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self._chunk_size = chunk_size
        self._executor: Optional[ProcessPoolExecutor] = None

        # Statistics
        self._total_processed = 0
        self._total_time = 0.0

    def __enter__(self) -> ParallelBatchProcessor:
        self._executor = ProcessPoolExecutor(max_workers=self._n_workers)
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Shut down the worker pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _chunk_list(
        self, items: List[Any], chunk_size: int
    ) -> List[List[Any]]:
        """Split a list into chunks."""
        return [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

    def batch_evaluate(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
        score_fn: Callable,
        descriptor_fn: Callable,
    ) -> BatchResult:
        """Evaluate a batch of DAGs in parallel.

        Parameters
        ----------
        dags : list of np.ndarray
            DAGs to evaluate.
        data : np.ndarray
            Data matrix.
        score_fn : callable
            Scoring function.
        descriptor_fn : callable
            Descriptor function.

        Returns
        -------
        BatchResult
            Aggregated results from all workers.
        """
        batch_size = len(dags)
        if batch_size == 0:
            return BatchResult(
                qualities=np.array([], dtype=np.float64),
                descriptors=np.empty((0, 0), dtype=np.float64),
                valid_mask=np.array([], dtype=np.bool_),
            )

        t0 = time.perf_counter()

        chunk_size = self._chunk_size or max(
            1, batch_size // self._n_workers
        )
        chunks = self._chunk_list(list(dags), chunk_size)

        # If no executor or single chunk, run sequentially
        if self._executor is None or len(chunks) <= 1:
            all_results = _worker_evaluate((list(dags), data, score_fn, descriptor_fn))
        else:
            futures = []
            for chunk in chunks:
                future = self._executor.submit(
                    _worker_evaluate, (chunk, data, score_fn, descriptor_fn)
                )
                futures.append(future)

            all_results = []
            for future in futures:
                all_results.extend(future.result())

        elapsed = time.perf_counter() - t0
        self._total_processed += batch_size
        self._total_time += elapsed

        # Aggregate
        qualities = np.array([r[0] for r in all_results], dtype=np.float64)
        valid_mask = np.isfinite(qualities)

        # Stack descriptors
        valid_descs = [r[1] for r in all_results if r[1].shape[0] > 0]
        if valid_descs:
            dim = valid_descs[0].shape[0]
            descriptors = np.full(
                (batch_size, dim), np.nan, dtype=np.float64
            )
            for i, r in enumerate(all_results):
                if r[1].shape[0] == dim:
                    descriptors[i] = r[1]
        else:
            descriptors = np.empty((batch_size, 0), dtype=np.float64)

        return BatchResult(
            qualities=qualities,
            descriptors=descriptors,
            valid_mask=valid_mask,
            timings={"total": elapsed},
        )

    def batch_mutate(
        self,
        dags: List[AdjacencyMatrix],
        mutation_fn: Callable,
        rng: Optional[np.random.Generator] = None,
    ) -> List[AdjacencyMatrix]:
        """Mutate a batch of DAGs (runs sequentially due to RNG state).

        For mutations, sequential execution is preferred to maintain
        reproducibility with a single RNG.
        """
        if rng is None:
            rng = np.random.default_rng()
        results = []
        for dag in dags:
            try:
                results.append(mutation_fn(dag, rng))
            except Exception:
                results.append(dag.copy())
        return results

    def stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            "n_workers": self._n_workers,
            "total_processed": self._total_processed,
            "total_time": self._total_time,
            "avg_throughput": (
                self._total_processed / self._total_time
                if self._total_time > 0
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def batch_score_dags(
    dags: Sequence[AdjacencyMatrix],
    data: DataMatrix,
    score_fn: Callable[..., QualityScore],
) -> np.ndarray:
    """Score a batch of DAGs and return quality array.

    Parameters
    ----------
    dags : sequence of np.ndarray
        DAGs to score.
    data : np.ndarray
        Data matrix.
    score_fn : callable
        ``(dag, data) -> float`` scoring function.

    Returns
    -------
    np.ndarray
        Array of quality scores (``-inf`` for failures).
    """
    qualities = np.full(len(dags), float("-inf"), dtype=np.float64)
    for i, dag in enumerate(dags):
        try:
            qualities[i] = score_fn(dag, data)
        except Exception:
            pass
    return qualities


def batch_compute_descriptors(
    dags: Sequence[AdjacencyMatrix],
    descriptor_fn: Callable[..., BehavioralDescriptor],
    data: Optional[DataMatrix] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute descriptors for a batch of DAGs.

    Parameters
    ----------
    dags : sequence of np.ndarray
        DAGs to describe.
    descriptor_fn : callable
        Descriptor computation function.
    data : np.ndarray, optional
        Data matrix.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(descriptors, valid_mask)`` where descriptors has shape
        ``(batch, dim)`` and valid_mask is boolean.
    """
    batch_size = len(dags)
    results: List[np.ndarray] = []
    valid_mask = np.zeros(batch_size, dtype=np.bool_)

    for i, dag in enumerate(dags):
        try:
            desc = descriptor_fn(dag, data)
            results.append(np.asarray(desc, dtype=np.float64))
            valid_mask[i] = True
        except Exception:
            results.append(np.array([], dtype=np.float64))

    if results and valid_mask.any():
        first_valid = next(i for i in range(batch_size) if valid_mask[i])
        dim = results[first_valid].shape[0]
        descriptors = np.full((batch_size, dim), np.nan, dtype=np.float64)
        for i in range(batch_size):
            if valid_mask[i] and results[i].shape[0] == dim:
                descriptors[i] = results[i]
    else:
        descriptors = np.empty((batch_size, 0), dtype=np.float64)

    return descriptors, valid_mask
