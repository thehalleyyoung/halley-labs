"""CVT (Centroidal Voronoi Tessellation) MAP-Elites archive.

Partitions the behavioural descriptor space into Voronoi cells defined
by centroids computed via k-means.  Optionally uses a KD-tree for
efficient nearest-centroid lookup on high-dimensional descriptors.

Features
--------
- K-means centroid initialisation from uniform random samples.
- KD-tree acceleration for nearest-centroid queries.
- Optional adaptive centroid recomputation.
- History tracking, serialisation, and diversity metrics.
- Curiosity-based and quality-proportional sampling.
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
# Per-cell bookkeeping
# ------------------------------------------------------------------

@dataclass
class CVTCellInfo:
    """Bookkeeping for a single Voronoi cell.

    Attributes
    ----------
    first_filled : int
        Timestamp when the cell was first occupied.
    last_updated : int
        Timestamp of the most recent replacement.
    replacement_count : int
        Number of quality improvements in this cell.
    descriptors_seen : int
        Total number of descriptors mapped to this cell.
    """

    first_filled: int = 0
    last_updated: int = 0
    replacement_count: int = 0
    descriptors_seen: int = 0


# ------------------------------------------------------------------
# CVTArchive
# ------------------------------------------------------------------

class CVTArchive(Archive):
    """Archive that partitions descriptor space using Voronoi tessellation.

    The archive maintains a set of centroids computed via k-means over
    uniformly sampled points in the bounded descriptor space.  Each
    incoming descriptor is assigned to the nearest centroid (Voronoi cell),
    and only the best-quality solution per cell is retained.

    Parameters
    ----------
    n_cells : int
        Number of Voronoi cells (centroids).
    descriptor_dim : int
        Dimensionality of the descriptor space.
    lower_bounds : npt.NDArray[np.float64]
        Per-dimension lower bounds.
    upper_bounds : npt.NDArray[np.float64]
        Per-dimension upper bounds.
    rng : np.random.Generator, optional
        Random number generator for k-means initialisation.
    n_samples : int
        Number of random samples used to compute k-means centroids.
    max_iter : int
        Maximum k-means iterations.
    use_kd_tree : bool
        If ``True`` (default), build a ``scipy.spatial.cKDTree`` for
        fast nearest-centroid queries.  Falls back to brute-force if
        scipy is not available.
    track_history : bool
        If ``True`` (default), maintain per-cell bookkeeping.
    """

    def __init__(
        self,
        n_cells: int,
        descriptor_dim: int,
        lower_bounds: npt.NDArray[np.float64],
        upper_bounds: npt.NDArray[np.float64],
        rng: Optional[np.random.Generator] = None,
        n_samples: int = 10_000,
        max_iter: int = 100,
        use_kd_tree: bool = True,
        track_history: bool = True,
    ) -> None:
        super().__init__(lower_bounds, upper_bounds)
        if rng is None:
            rng = np.random.default_rng()
        self._n_cells = n_cells
        self._dim = descriptor_dim
        self._cells: Dict[CellIndex, ArchiveEntry] = {}
        self._track_history = track_history
        self._cell_info: Dict[CellIndex, CVTCellInfo] = {}
        self._use_kd_tree = use_kd_tree

        # Compute centroids via k-means on uniform random samples
        self._centroids = self._compute_centroids(
            n_cells, descriptor_dim, n_samples, max_iter, rng,
        )

        # Build KD-tree for fast lookup
        self._kd_tree: Any = None
        if use_kd_tree:
            self._build_kd_tree()

        # Statistics
        self._total_insertions: int = 0
        self._total_replacements: int = 0
        self._total_fills: int = 0
        self._improvement_history: List[Tuple[int, float]] = []
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # k-means centroid computation
    # ------------------------------------------------------------------

    def _compute_centroids(
        self,
        k: int,
        dim: int,
        n_samples: int,
        max_iter: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Run k-means on uniform samples in the bounded descriptor space.

        Uses k-means++ style initialisation for better convergence.

        Parameters
        ----------
        k : int
            Number of clusters / centroids.
        dim : int
            Descriptor dimensionality.
        n_samples : int
            Number of uniform samples.
        max_iter : int
            Maximum iterations for k-means.
        rng : np.random.Generator

        Returns
        -------
        npt.NDArray[np.float64]
            ``(k, dim)`` array of centroid positions.
        """
        samples = rng.uniform(self._lower, self._upper, size=(n_samples, dim))

        # K-means++ initialisation
        centroids = np.empty((k, dim), dtype=np.float64)
        first_idx = rng.integers(0, n_samples)
        centroids[0] = samples[first_idx]

        for c in range(1, k):
            dists = np.min(
                np.linalg.norm(
                    samples[:, np.newaxis, :] - centroids[np.newaxis, :c, :],
                    axis=2,
                ),
                axis=1,
            )
            dists_sq = dists ** 2
            total = dists_sq.sum()
            if total < 1e-15:
                centroids[c] = samples[rng.integers(0, n_samples)]
            else:
                probs = dists_sq / total
                centroids[c] = samples[rng.choice(n_samples, p=probs)]

        # Lloyd's algorithm iterations
        for _ in range(max_iter):
            dists = np.linalg.norm(
                samples[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2,
            )
            labels = np.argmin(dists, axis=1)

            new_centroids = np.empty_like(centroids)
            for c in range(k):
                members = samples[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(new_centroids, centroids, atol=1e-8):
                break
            centroids = new_centroids

        return centroids

    def _build_kd_tree(self) -> None:
        """Build a KD-tree from the current centroids."""
        try:
            from scipy.spatial import cKDTree
            self._kd_tree = cKDTree(self._centroids)
        except ImportError:
            self._kd_tree = None
            self._use_kd_tree = False

    # ------------------------------------------------------------------
    # Centroid access and recomputation
    # ------------------------------------------------------------------

    @property
    def centroids(self) -> npt.NDArray[np.float64]:
        """Return a copy of the centroid positions.

        Returns
        -------
        npt.NDArray[np.float64]
            ``(n_cells, descriptor_dim)`` array.
        """
        return self._centroids.copy()

    def recompute_centroids(self) -> None:
        """Recompute centroids as the mean descriptor of their occupants.

        Centroids with no occupant are left unchanged.  The KD-tree is
        rebuilt after recomputation.
        """
        new_centroids = self._centroids.copy()
        for idx, entry in self._cells.items():
            cell_id = idx[0]
            new_centroids[cell_id] = entry.descriptor
        self._centroids = new_centroids
        if self._use_kd_tree:
            self._build_kd_tree()

    def recompute_centroids_from_history(
        self,
        all_descriptors: npt.NDArray[np.float64],
    ) -> None:
        """Recompute centroids from a batch of descriptors using k-means assignment.

        Parameters
        ----------
        all_descriptors : npt.NDArray[np.float64]
            ``(m, descriptor_dim)`` array of all descriptors seen so far.
        """
        if len(all_descriptors) == 0:
            return
        labels = self._assign_labels(all_descriptors)
        new_centroids = self._centroids.copy()
        for c in range(self._n_cells):
            members = all_descriptors[labels == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
        self._centroids = new_centroids
        if self._use_kd_tree:
            self._build_kd_tree()

    def _assign_labels(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Assign each point to its nearest centroid.

        Parameters
        ----------
        points : npt.NDArray[np.float64]
            ``(m, dim)`` array of points.

        Returns
        -------
        npt.NDArray[np.int64]
            Centroid index for each point.
        """
        if self._kd_tree is not None:
            _, labels = self._kd_tree.query(points)
            return np.asarray(labels, dtype=np.int64)
        dists = np.linalg.norm(
            points[:, np.newaxis, :] - self._centroids[np.newaxis, :, :],
            axis=2,
        )
        return np.argmin(dists, axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _descriptor_to_index(self, descriptor: BehavioralDescriptor) -> CellIndex:
        """Return the index of the nearest centroid.

        Parameters
        ----------
        descriptor : BehavioralDescriptor

        Returns
        -------
        CellIndex
            Single-element tuple ``(centroid_index,)``.
        """
        desc = np.asarray(descriptor, dtype=np.float64)
        if self._kd_tree is not None:
            _, idx = self._kd_tree.query(desc)
            return (int(idx),)
        dists = np.linalg.norm(self._centroids - desc, axis=1)
        return (int(np.argmin(dists)),)

    def centroid_for_index(self, index: CellIndex) -> npt.NDArray[np.float64]:
        """Return the centroid position for a given cell index.

        Parameters
        ----------
        index : CellIndex

        Returns
        -------
        npt.NDArray[np.float64]
        """
        return self._centroids[index[0]].copy()

    # ------------------------------------------------------------------
    # Archive interface
    # ------------------------------------------------------------------

    def add(self, entry: ArchiveEntry) -> bool:
        """Insert *entry* if its Voronoi cell is empty or quality improves.

        Parameters
        ----------
        entry : ArchiveEntry

        Returns
        -------
        bool
            ``True`` if inserted or replaced.
        """
        idx = self._descriptor_to_index(entry.descriptor)
        self._insertion_count += 1
        self._total_insertions += 1
        entry.timestamp = self._insertion_count
        now = time.monotonic()

        if self._track_history:
            info = self._cell_info.get(idx) or CVTCellInfo()
            info.descriptors_seen += 1
            self._cell_info[idx] = info

        existing = self._cells.get(idx)
        if existing is None:
            self._cells[idx] = entry
            self._total_fills += 1
            if self._track_history:
                info = self._cell_info[idx]
                info.first_filled = self._insertion_count
                info.last_updated = self._insertion_count
            return True

        if entry.quality > existing.quality:
            improvement = entry.quality - existing.quality
            self._cells[idx] = entry
            self._total_replacements += 1
            self._improvement_history.append(
                (self._insertion_count, improvement)
            )
            if self._track_history:
                info = self._cell_info[idx]
                info.last_updated = self._insertion_count
                info.replacement_count += 1
            return True

        return False

    def add_batch(
        self,
        entries: Sequence[ArchiveEntry],
    ) -> List[bool]:
        """Insert multiple entries, returning per-entry success flags."""
        return [self.add(e) for e in entries]

    def get(self, index: CellIndex) -> Optional[ArchiveEntry]:
        """Return the elite at *index*, or ``None``."""
        return self._cells.get(index)

    def sample(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Uniformly sample *n* elites with replacement."""
        if not self._cells:
            return []
        entries = list(self._cells.values())
        idxs = rng.integers(0, len(entries), size=n)
        return [entries[i] for i in idxs]

    def sample_curiosity(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Sample elites weighted inversely by visit count."""
        if not self._cells:
            return []
        indices = list(self._cells.keys())
        entries = [self._cells[idx] for idx in indices]
        weights = np.array([
            1.0 / (1.0 + self._cell_info.get(idx, CVTCellInfo()).descriptors_seen)
            for idx in indices
        ], dtype=np.float64)
        probs = weights / weights.sum()
        chosen = rng.choice(len(entries), size=n, replace=True, p=probs)
        return [entries[i] for i in chosen]

    def best(self) -> ArchiveEntry:
        """Return the highest-quality elite."""
        if not self._cells:
            raise ValueError("Archive is empty.")
        return max(self._cells.values(), key=lambda e: e.quality)

    def elites(self) -> List[ArchiveEntry]:
        """Return all elites."""
        return list(self._cells.values())

    def coverage(self) -> float:
        """Fraction of Voronoi cells occupied."""
        return len(self._cells) / max(self._n_cells, 1)

    def qd_score(self) -> float:
        """Sum of all elite qualities."""
        return sum(e.quality for e in self._cells.values())

    def mean_quality(self) -> float:
        """Mean quality across occupied cells."""
        if not self._cells:
            return 0.0
        return self.qd_score() / len(self._cells)

    def diversity(self) -> float:
        """Mean pairwise Euclidean distance between elite descriptors."""
        if len(self._cells) < 2:
            return 0.0
        descs = np.array([e.descriptor for e in self._cells.values()])
        n = len(descs)
        diff = descs[:, np.newaxis, :] - descs[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        return float(dists.sum() / (n * (n - 1)))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Total number of Voronoi cells."""
        return self._n_cells

    @property
    def descriptor_dim(self) -> int:
        """Dimensionality of the descriptor space."""
        return self._dim

    @property
    def total_insertions(self) -> int:
        return self._total_insertions

    @property
    def total_replacements(self) -> int:
        return self._total_replacements

    @property
    def total_fills(self) -> int:
        return self._total_fills

    @property
    def improvement_history(self) -> List[Tuple[int, float]]:
        return list(self._improvement_history)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[ArchiveEntry]:
        return iter(self._cells.values())

    def __contains__(self, index: object) -> bool:
        return index in self._cells

    def __repr__(self) -> str:
        return (
            f"CVTArchive(n_cells={self._n_cells}, "
            f"occupied={len(self._cells)}/{self._n_cells}, "
            f"qd_score={self.qd_score():.4f})"
        )

    # ------------------------------------------------------------------
    # Reset / clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all elites but keep centroids and configuration."""
        self._cells.clear()
        self._cell_info.clear()
        self._insertion_count = 0
        self._total_insertions = 0
        self._total_replacements = 0
        self._total_fills = 0
        self._improvement_history.clear()

    def reset(self) -> None:
        """Full reset including timers."""
        self.clear()
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the archive to disk.

        Creates ``<path>.pkl`` (data) and ``<path>.meta.json`` (metadata).
        """
        pkl_path = Path(path).with_suffix(".pkl")
        meta_path = Path(f"{path}.meta.json")

        state = {
            "cells": self._cells,
            "cell_info": self._cell_info,
            "centroids": self._centroids,
            "insertion_count": self._insertion_count,
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
            "total_fills": self._total_fills,
            "improvement_history": self._improvement_history,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta: Dict[str, Any] = {
            "n_cells": self._n_cells,
            "descriptor_dim": self._dim,
            "lower_bounds": self._lower.tolist(),
            "upper_bounds": self._upper.tolist(),
            "occupied_cells": len(self._cells),
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best_quality": self.best().quality if self._cells else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CVTArchive":
        """Restore an archive from files created by :meth:`save`."""
        meta_path = Path(f"{path}.meta.json")
        pkl_path = Path(path).with_suffix(".pkl")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        archive = cls(
            n_cells=meta["n_cells"],
            descriptor_dim=meta["descriptor_dim"],
            lower_bounds=np.array(meta["lower_bounds"], dtype=np.float64),
            upper_bounds=np.array(meta["upper_bounds"], dtype=np.float64),
            n_samples=0,  # skip k-means; centroids come from pickle
            max_iter=0,
        )

        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        archive._centroids = state["centroids"]
        archive._cells = state["cells"]
        archive._cell_info = state.get("cell_info", {})
        archive._insertion_count = state.get("insertion_count", 0)
        archive._total_insertions = state.get("total_insertions", 0)
        archive._total_replacements = state.get("total_replacements", 0)
        archive._total_fills = state.get("total_fills", 0)
        archive._improvement_history = state.get("improvement_history", [])
        if archive._use_kd_tree:
            archive._build_kd_tree()
        return archive

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary of summary statistics."""
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
