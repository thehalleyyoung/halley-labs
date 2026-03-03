"""Grid-based MAP-Elites archive with uniform tessellation.

Provides a fixed-resolution grid archive that partitions the behavioural
descriptor space into a regular hyper-grid.  Each cell stores a single
elite (the highest-quality solution mapped to that cell).

Features
--------
- Configurable resolution per descriptor dimension.
- Full insertion history tracking (fills, replacements, improvement deltas).
- Serialisation to / from disk (pickle + JSON metadata).
- Diversity computation via mean pairwise descriptor distance.
- Curiosity-based and quality-proportional sampling.
- Statistics: total insertions, replacement count, per-cell occupancy duration.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.archive.archive_base import Archive, ArchiveEntry
from causal_qd.types import BehavioralDescriptor, CellIndex, QualityScore


# ------------------------------------------------------------------
# History record
# ------------------------------------------------------------------

@dataclass
class CellHistoryRecord:
    """A single event in a cell's history.

    Attributes
    ----------
    timestamp : int
        Monotonic insertion counter when this event occurred.
    quality : float
        Quality of the solution that triggered this event.
    event_type : str
        ``"fill"`` (first occupant) or ``"replace"`` (improved occupant).
    improvement : float
        Quality delta relative to the previous occupant (0 for fills).
    wall_time : float
        Wall-clock time of the event (``time.monotonic()``).
    """

    timestamp: int
    quality: float
    event_type: str
    improvement: float = 0.0
    wall_time: float = 0.0


@dataclass
class CellInfo:
    """Per-cell bookkeeping information.

    Attributes
    ----------
    first_filled : int
        Timestamp when the cell was first occupied.
    last_updated : int
        Timestamp of the most recent replacement.
    replacement_count : int
        Number of times the elite was replaced with a better one.
    history : list of CellHistoryRecord
        Full event log for this cell.
    """

    first_filled: int = 0
    last_updated: int = 0
    replacement_count: int = 0
    history: List[CellHistoryRecord] = field(default_factory=list)


# ------------------------------------------------------------------
# GridArchive
# ------------------------------------------------------------------

class GridArchive(Archive):
    """Fixed-resolution grid archive for MAP-Elites.

    The descriptor space is partitioned into a regular grid with
    ``dims[0] × dims[1] × …`` cells.  Each cell stores at most one
    elite—the solution with the highest quality score that maps to it.

    Parameters
    ----------
    dims : Tuple[int, ...]
        Number of bins along each descriptor dimension.
    lower_bounds : npt.NDArray[np.float64]
        Per-dimension lower bounds of the descriptor space.
    upper_bounds : npt.NDArray[np.float64]
        Per-dimension upper bounds of the descriptor space.
    track_history : bool
        If ``True`` (default), maintain per-cell history records.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        lower_bounds: npt.NDArray[np.float64],
        upper_bounds: npt.NDArray[np.float64],
        track_history: bool = True,
    ) -> None:
        super().__init__(lower_bounds, upper_bounds)
        self._dims = dims
        self._cells: Dict[CellIndex, ArchiveEntry] = {}
        self._total_cells: int = 1
        for d in dims:
            self._total_cells *= d

        # History / stats tracking
        self._track_history = track_history
        self._cell_info: Dict[CellIndex, CellInfo] = {}
        self._total_insertions: int = 0
        self._total_replacements: int = 0
        self._total_fills: int = 0
        self._improvement_history: List[Tuple[int, float]] = []
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dims(self) -> Tuple[int, ...]:
        """Number of bins along each descriptor dimension."""
        return self._dims

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self._total_cells

    @property
    def descriptor_dim(self) -> int:
        """Dimensionality of the descriptor space."""
        return len(self._dims)

    @property
    def total_insertions(self) -> int:
        """Total number of ``add()`` calls that resulted in an insertion."""
        return self._total_fills + self._total_replacements

    @property
    def total_attempts(self) -> int:
        """Total number of ``add()`` calls (including rejections)."""
        return self._total_insertions

    @property
    def total_replacements(self) -> int:
        """Number of times an existing elite was replaced by a better one."""
        return self._total_replacements

    @property
    def total_fills(self) -> int:
        """Number of first-time cell fills."""
        return self._total_fills

    @property
    def improvement_history(self) -> List[Tuple[int, float]]:
        """List of ``(timestamp, quality_delta)`` for every replacement."""
        return list(self._improvement_history)

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _descriptor_to_index(self, descriptor: BehavioralDescriptor) -> CellIndex:
        """Map a descriptor vector to a grid cell index.

        Descriptors are clipped to the archive bounds before binning.

        Parameters
        ----------
        descriptor : BehavioralDescriptor
            Real-valued descriptor vector of length ``descriptor_dim``.

        Returns
        -------
        CellIndex
            Tuple of integer bin indices, one per dimension.
        """
        desc = np.asarray(descriptor, dtype=np.float64)
        clipped = np.clip(desc, self._lower, self._upper)
        spans = np.maximum(self._upper - self._lower, 1e-12)
        normed = (clipped - self._lower) / spans
        indices: List[int] = []
        for i, d in enumerate(self._dims):
            idx = int(normed[i] * d)
            idx = min(idx, d - 1)
            indices.append(idx)
        return tuple(indices)

    def index_to_descriptor_center(self, index: CellIndex) -> BehavioralDescriptor:
        """Return the centre of the cell identified by *index*.

        Parameters
        ----------
        index : CellIndex
            Grid cell index tuple.

        Returns
        -------
        BehavioralDescriptor
            Descriptor at the cell centre.
        """
        centers = np.empty(len(self._dims), dtype=np.float64)
        spans = self._upper - self._lower
        for i, d in enumerate(self._dims):
            centers[i] = self._lower[i] + (index[i] + 0.5) * spans[i] / d
        return centers

    # ------------------------------------------------------------------
    # Archive interface
    # ------------------------------------------------------------------

    def add(self, entry: ArchiveEntry) -> bool:
        """Insert *entry* into the archive if its cell is empty or it improves quality.

        Parameters
        ----------
        entry : ArchiveEntry
            The candidate solution to insert.

        Returns
        -------
        bool
            ``True`` if the entry was added (new cell or improvement).
        """
        idx = self._descriptor_to_index(entry.descriptor)
        self._insertion_count += 1
        self._total_insertions += 1
        entry.timestamp = self._insertion_count
        now = time.monotonic()

        existing = self._cells.get(idx)

        if existing is None:
            # First occupant
            self._cells[idx] = entry
            self._total_fills += 1
            if self._track_history:
                info = CellInfo(
                    first_filled=self._insertion_count,
                    last_updated=self._insertion_count,
                )
                info.history.append(
                    CellHistoryRecord(
                        timestamp=self._insertion_count,
                        quality=entry.quality,
                        event_type="fill",
                        wall_time=now - self._start_time,
                    )
                )
                self._cell_info[idx] = info
            return True

        if entry.quality > existing.quality:
            improvement = entry.quality - existing.quality
            self._cells[idx] = entry
            self._total_replacements += 1
            self._improvement_history.append(
                (self._insertion_count, improvement)
            )
            if self._track_history:
                info = self._cell_info.get(idx) or CellInfo()
                info.last_updated = self._insertion_count
                info.replacement_count += 1
                info.history.append(
                    CellHistoryRecord(
                        timestamp=self._insertion_count,
                        quality=entry.quality,
                        event_type="replace",
                        improvement=improvement,
                        wall_time=now - self._start_time,
                    )
                )
                self._cell_info[idx] = info
            return True

        return False

    def add_batch(
        self,
        entries: Sequence[ArchiveEntry],
    ) -> List[bool]:
        """Insert multiple entries, returning per-entry success flags.

        Parameters
        ----------
        entries : Sequence[ArchiveEntry]
            Entries to insert.

        Returns
        -------
        List[bool]
            Per-entry insertion result.
        """
        return [self.add(e) for e in entries]

    def get(self, index: CellIndex) -> Optional[ArchiveEntry]:
        """Return the elite at *index*, or ``None`` if unoccupied."""
        return self._cells.get(index)

    def get_cell_info(self, index: CellIndex) -> Optional[CellInfo]:
        """Return per-cell bookkeeping for *index*, or ``None``."""
        return self._cell_info.get(index)

    def sample(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Uniformly sample *n* elites with replacement.

        Parameters
        ----------
        n : int
            Number of elites to sample.
        rng : np.random.Generator
            Random generator for reproducibility.

        Returns
        -------
        List[ArchiveEntry]
            Sampled elites (may contain duplicates).
        """
        if not self._cells:
            return []
        entries = list(self._cells.values())
        idxs = rng.integers(0, len(entries), size=n)
        return [entries[i] for i in idxs]

    def sample_curiosity(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Sample elites weighted inversely by replacement count.

        Cells that have been updated fewer times are more likely to be
        selected, encouraging exploration of under-visited regions.

        Parameters
        ----------
        n : int
            Number of elites to sample.
        rng : np.random.Generator
            Random generator.

        Returns
        -------
        List[ArchiveEntry]
            Sampled elites (curiosity-biased).
        """
        if not self._cells:
            return []
        indices = list(self._cells.keys())
        entries = [self._cells[idx] for idx in indices]

        weights = np.array([
            1.0 / (1.0 + self._cell_info.get(idx, CellInfo()).replacement_count)
            for idx in indices
        ], dtype=np.float64)
        probs = weights / weights.sum()
        chosen = rng.choice(len(entries), size=n, replace=True, p=probs)
        return [entries[i] for i in chosen]

    def sample_quality_proportional(
        self, n: int, rng: np.random.Generator
    ) -> List[ArchiveEntry]:
        """Sample elites proportional to their quality scores.

        Parameters
        ----------
        n : int
            Number of elites to sample.
        rng : np.random.Generator

        Returns
        -------
        List[ArchiveEntry]
        """
        if not self._cells:
            return []
        entries = list(self._cells.values())
        qualities = np.array([e.quality for e in entries], dtype=np.float64)
        # Shift so minimum is 0
        shifted = qualities - qualities.min()
        total = shifted.sum()
        if total < 1e-15:
            probs = np.ones(len(entries)) / len(entries)
        else:
            probs = shifted / total
        chosen = rng.choice(len(entries), size=n, replace=True, p=probs)
        return [entries[i] for i in chosen]

    def best(self) -> ArchiveEntry:
        """Return the highest-quality elite.

        Raises
        ------
        ValueError
            If the archive is empty.
        """
        if not self._cells:
            raise ValueError("Archive is empty.")
        return max(self._cells.values(), key=lambda e: e.quality)

    def elites(self) -> List[ArchiveEntry]:
        """Return all elites currently stored in the archive."""
        return list(self._cells.values())

    def coverage(self) -> float:
        """Return the fraction of grid cells that contain an elite."""
        return len(self._cells) / max(self._total_cells, 1)

    def qd_score(self) -> float:
        """Return the QD-score: sum of all elite quality scores."""
        return sum(e.quality for e in self._cells.values())

    def mean_quality(self) -> float:
        """Mean quality across all occupied cells.

        Returns
        -------
        float
            0.0 if the archive is empty.
        """
        if not self._cells:
            return 0.0
        return self.qd_score() / len(self._cells)

    def diversity(self) -> float:
        """Mean pairwise Euclidean distance between elite descriptors.

        Returns 0.0 for fewer than two elites.
        """
        if len(self._cells) < 2:
            return 0.0
        descs = np.array([e.descriptor for e in self._cells.values()])
        n = len(descs)
        # Vectorised pairwise distance (upper triangle)
        diff = descs[:, np.newaxis, :] - descs[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        return float(dists.sum() / (n * (n - 1)))

    def descriptor_variance(self) -> npt.NDArray[np.float64]:
        """Per-dimension variance of elite descriptors.

        Returns
        -------
        npt.NDArray[np.float64]
            Variance vector of length ``descriptor_dim``, or zeros if empty.
        """
        if not self._cells:
            return np.zeros(len(self._dims), dtype=np.float64)
        descs = np.array([e.descriptor for e in self._cells.values()])
        return descs.var(axis=0)

    def occupied_indices(self) -> List[CellIndex]:
        """Return a list of all occupied cell indices."""
        return list(self._cells.keys())

    def empty_neighbor_count(self, index: CellIndex) -> int:
        """Count the number of empty direct neighbours of *index*.

        Two cells are neighbours if they differ by ±1 in exactly one
        dimension (von Neumann neighbourhood).

        Parameters
        ----------
        index : CellIndex
            The cell to query.

        Returns
        -------
        int
            Number of empty neighbours.
        """
        count = 0
        for dim in range(len(self._dims)):
            for delta in (-1, 1):
                neighbor = list(index)
                neighbor[dim] += delta
                if 0 <= neighbor[dim] < self._dims[dim]:
                    nidx = tuple(neighbor)
                    if nidx not in self._cells:
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of occupied cells."""
        return len(self._cells)

    def __iter__(self) -> Iterator[ArchiveEntry]:
        """Iterate over all elites."""
        return iter(self._cells.values())

    def __contains__(self, index: object) -> bool:
        """Return ``True`` if *index* has an elite."""
        return index in self._cells

    def __repr__(self) -> str:
        return (
            f"GridArchive(dims={self._dims}, "
            f"occupied={len(self._cells)}/{self._total_cells}, "
            f"qd_score={self.qd_score():.4f})"
        )

    # ------------------------------------------------------------------
    # Reset / clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all elites from the archive but keep configuration."""
        self._cells.clear()
        self._cell_info.clear()
        self._insertion_count = 0
        self._total_insertions = 0
        self._total_replacements = 0
        self._total_fills = 0
        self._improvement_history.clear()

    def reset(self) -> None:
        """Completely reset the archive including internal timers."""
        self.clear()
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Heatmap data
    # ------------------------------------------------------------------

    def as_quality_grid(self) -> npt.NDArray[np.float64]:
        """Return a dense array of quality values for all cells.

        Unoccupied cells are filled with ``NaN``.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape ``self._dims``.
        """
        grid = np.full(self._dims, np.nan, dtype=np.float64)
        for idx, entry in self._cells.items():
            grid[idx] = entry.quality
        return grid

    def as_occupancy_grid(self) -> npt.NDArray[np.bool_]:
        """Return a boolean grid marking occupied cells.

        Returns
        -------
        npt.NDArray[np.bool_]
        """
        grid = np.zeros(self._dims, dtype=np.bool_)
        for idx in self._cells:
            grid[idx] = True
        return grid

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the archive to disk.

        Creates two files:
        - ``<path>.pkl`` — pickled archive data (cells, history).
        - ``<path>.meta.json`` — human-readable metadata.

        Parameters
        ----------
        path : str
            Base path (without extension).
        """
        pkl_path = Path(path).with_suffix(".pkl")
        meta_path = Path(f"{path}.meta.json")

        state = {
            "cells": self._cells,
            "cell_info": self._cell_info,
            "insertion_count": self._insertion_count,
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
            "total_fills": self._total_fills,
            "improvement_history": self._improvement_history,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta: Dict[str, Any] = {
            "dims": list(self._dims),
            "lower_bounds": self._lower.tolist(),
            "upper_bounds": self._upper.tolist(),
            "total_cells": self._total_cells,
            "occupied_cells": len(self._cells),
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best_quality": self.best().quality if self._cells else None,
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GridArchive":
        """Restore an archive from files created by :meth:`save`.

        Parameters
        ----------
        path : str
            Base path (without extension) used when saving.

        Returns
        -------
        GridArchive
        """
        meta_path = Path(f"{path}.meta.json")
        pkl_path = Path(path).with_suffix(".pkl")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        archive = cls(
            dims=tuple(meta["dims"]),
            lower_bounds=np.array(meta["lower_bounds"], dtype=np.float64),
            upper_bounds=np.array(meta["upper_bounds"], dtype=np.float64),
        )

        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        archive._cells = state["cells"]
        archive._cell_info = state.get("cell_info", {})
        archive._insertion_count = state.get("insertion_count", 0)
        archive._total_insertions = state.get("total_insertions", 0)
        archive._total_replacements = state.get("total_replacements", 0)
        archive._total_fills = state.get("total_fills", 0)
        archive._improvement_history = state.get("improvement_history", [])
        return archive

    # ------------------------------------------------------------------
    # Summary statistics (convenience)
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary of summary statistics.

        Returns
        -------
        Dict[str, Any]
            Keys: ``coverage``, ``qd_score``, ``best_quality``,
            ``mean_quality``, ``num_elites``, ``diversity``,
            ``total_insertions``, ``total_replacements``.
        """
        return {
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best_quality": self.best().quality if self._cells else float("-inf"),
            "mean_quality": self.mean_quality(),
            "num_elites": len(self._cells),
            "diversity": self.diversity(),
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
            "total_fills": self._total_fills,
        }
