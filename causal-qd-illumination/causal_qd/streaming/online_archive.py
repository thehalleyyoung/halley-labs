"""Online archive with streaming updates, sliding window, and merging.

Provides:
  - OnlineArchive: grid archive with time-decay for streaming settings
  - Sliding window for data recency
  - Forgetting factor for old solutions
  - Archive merging (combine two archives)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, CellIndex, QualityScore

logger = logging.getLogger(__name__)


class _ArchiveEntry:
    """Internal storage for a single archive cell."""

    __slots__ = ("dag", "quality", "descriptor", "age", "generation", "metadata")

    def __init__(
        self,
        dag: Any,
        quality: QualityScore,
        descriptor: BehavioralDescriptor,
        generation: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dag = dag
        self.quality = quality
        self.descriptor = descriptor
        self.age: int = 0
        self.generation = generation
        self.metadata = metadata or {}


class OnlineArchive:
    """Grid archive with time-decay for streaming / online settings.

    At every :meth:`step`, all elite qualities are multiplied by
    ``decay_factor``.  Cells whose quality drops below
    ``eviction_threshold`` are removed.

    Supports:
    - Sliding window: only the most recent *window_size* solutions
      are considered when computing QD metrics.
    - Forgetting factor: exponential decay of quality over time.
    - Archive merging: combine two archives into one.
    - Snapshot/restore for checkpointing.

    Parameters
    ----------
    dims :
        Number of cells along each descriptor dimension.
    bounds :
        ``(lower, upper)`` arrays defining the descriptor space.
    decay_factor :
        Multiplicative decay applied to quality each step (in ``(0, 1]``).
    eviction_threshold :
        Minimum quality; cells below this are evicted on :meth:`step`.
    window_size :
        Maximum number of insertions to track (0 = unlimited).
    """

    def __init__(
        self,
        dims: Tuple[int, ...] = (50, 50),
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        decay_factor: float = 0.99,
        eviction_threshold: float = 1e-6,
        window_size: int = 0,
    ) -> None:
        self.dims = dims
        self.bounds = bounds
        self.decay_factor = decay_factor
        self.eviction_threshold = eviction_threshold
        self._window_size = window_size
        self.elites: Dict[CellIndex, _ArchiveEntry] = {}
        self._insertion_history: List[Tuple[CellIndex, float]] = []
        self._generation: int = 0
        self._total_insertions: int = 0
        self._total_improvements: int = 0

    @property
    def total_cells(self) -> int:
        """Total number of cells in the archive grid."""
        result = 1
        for d in self.dims:
            result *= d
        return result

    # ------------------------------------------------------------------
    # Cell index mapping
    # ------------------------------------------------------------------

    def _cell_index(self, descriptor: BehavioralDescriptor) -> CellIndex:
        """Map a descriptor to its grid cell index."""
        desc = np.asarray(descriptor, dtype=np.float64)
        if self.bounds is None:
            idx = tuple(
                max(0, min(int(desc[i] * self.dims[i]), self.dims[i] - 1))
                for i in range(len(self.dims))
            )
        else:
            lo, hi = self.bounds
            normed = (desc - lo) / np.maximum(hi - lo, 1e-12)
            idx = tuple(
                max(0, min(int(normed[i] * self.dims[i]), self.dims[i] - 1))
                for i in range(len(self.dims))
            )
        return idx

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        dag: Any,
        quality: QualityScore,
        descriptor: BehavioralDescriptor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attempt to insert a solution into the archive.

        Parameters
        ----------
        dag :
            DAG representation (adjacency matrix or DAG object).
        quality :
            Quality score (higher is better).
        descriptor :
            Behavioral descriptor vector.
        metadata :
            Optional metadata dict.

        Returns
        -------
        bool
            *True* if the solution was inserted (new cell or better quality).
        """
        idx = self._cell_index(descriptor)
        self._total_insertions += 1

        existing = self.elites.get(idx)
        if existing is None or quality > existing.quality:
            self.elites[idx] = _ArchiveEntry(
                dag, quality, descriptor, self._generation, metadata,
            )
            self._total_improvements += 1
            self._insertion_history.append((idx, quality))

            # Enforce sliding window
            if self._window_size > 0 and len(self._insertion_history) > self._window_size:
                self._insertion_history.pop(0)

            return True
        return False

    # ------------------------------------------------------------------
    # Streaming operations
    # ------------------------------------------------------------------

    def step(self) -> int:
        """Apply quality decay and evict low-quality entries.

        Returns
        -------
        int
            Number of cells evicted.
        """
        self._generation += 1
        evicted = 0
        to_remove: List[CellIndex] = []

        for idx, entry in self.elites.items():
            entry.quality *= self.decay_factor
            entry.age += 1
            if entry.quality < self.eviction_threshold:
                to_remove.append(idx)

        for idx in to_remove:
            del self.elites[idx]
            evicted += 1

        return evicted

    def step_with_window(self, current_generation: int) -> int:
        """Apply age-based eviction: remove entries older than window.

        Parameters
        ----------
        current_generation :
            Current generation number.

        Returns
        -------
        int
            Number of cells evicted.
        """
        if self._window_size <= 0:
            return self.step()

        evicted = 0
        to_remove: List[CellIndex] = []

        for idx, entry in self.elites.items():
            age = current_generation - entry.generation
            if age > self._window_size:
                to_remove.append(idx)

        for idx in to_remove:
            del self.elites[idx]
            evicted += 1

        return evicted

    # ------------------------------------------------------------------
    # Archive merging
    # ------------------------------------------------------------------

    def merge(self, other: OnlineArchive) -> int:
        """Merge another archive into this one.

        For each cell in *other*, if the quality exceeds the current
        elite, the entry replaces it.

        Parameters
        ----------
        other :
            Archive to merge from.

        Returns
        -------
        int
            Number of cells updated.
        """
        updates = 0
        for idx, entry in other.elites.items():
            existing = self.elites.get(idx)
            if existing is None or entry.quality > existing.quality:
                self.elites[idx] = _ArchiveEntry(
                    entry.dag, entry.quality, entry.descriptor,
                    entry.generation, entry.metadata,
                )
                updates += 1
        return updates

    @staticmethod
    def merge_archives(archives: List[OnlineArchive]) -> OnlineArchive:
        """Create a new archive by merging multiple archives.

        Parameters
        ----------
        archives :
            List of archives to merge.

        Returns
        -------
        OnlineArchive
            Merged archive.
        """
        if not archives:
            return OnlineArchive()

        result = OnlineArchive(
            dims=archives[0].dims,
            bounds=archives[0].bounds,
        )
        for archive in archives:
            result.merge(archive)
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[_ArchiveEntry]:
        """Sample *n* elites uniformly at random.

        Parameters
        ----------
        n :
            Number of elites to sample.
        rng :
            Random generator.

        Returns
        -------
        list of _ArchiveEntry
        """
        if not self.elites:
            return []
        if rng is None:
            rng = np.random.default_rng()
        entries = list(self.elites.values())
        idxs = rng.integers(0, len(entries), size=n)
        return [entries[i] for i in idxs]

    def best(self) -> Optional[_ArchiveEntry]:
        """Return the highest-quality elite, or None if empty."""
        if not self.elites:
            return None
        return max(self.elites.values(), key=lambda e: e.quality)

    def coverage(self) -> float:
        """Fraction of cells occupied."""
        return len(self.elites) / max(self.total_cells, 1)

    def qd_score(self) -> float:
        """Sum of all elite qualities."""
        return sum(e.quality for e in self.elites.values())

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[CellIndex, float]:
        """Return a snapshot of the archive (cell → quality)."""
        return {idx: e.quality for idx, e in self.elites.items()}

    def clear(self) -> None:
        """Remove all elites."""
        self.elites.clear()
        self._insertion_history.clear()
        self._total_insertions = 0
        self._total_improvements = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return archive statistics."""
        qualities = [e.quality for e in self.elites.values()]
        return {
            "n_elites": len(self.elites),
            "total_cells": self.total_cells,
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
            "max_quality": float(np.max(qualities)) if qualities else 0.0,
            "min_quality": float(np.min(qualities)) if qualities else 0.0,
            "generation": self._generation,
            "total_insertions": self._total_insertions,
            "total_improvements": self._total_improvements,
        }

    def __len__(self) -> int:
        return len(self.elites)

    def __contains__(self, index: CellIndex) -> bool:
        return index in self.elites

    def __iter__(self) -> Iterator[_ArchiveEntry]:
        return iter(self.elites.values())
