"""Centroidal Voronoi Tessellation for QD archive cell management.

Implements Lloyd's algorithm (a k-means variant) to partition the
4-D behavior descriptor space into cells.  Each cell can hold at most
one elite genome, forming the MAP-Elites-style archive.

Classes
-------
CVTTessellation
    Fixed-cell CVT with Lloyd's algorithm.
AdaptiveCVT
    Dynamic CVT with cell splitting/merging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("exploration.cvt")


# ---------------------------------------------------------------------------
# CVT Tessellation
# ---------------------------------------------------------------------------


@dataclass
class CVTCellStats:
    """Statistics for a single CVT cell.

    Attributes
    ----------
    centroid : np.ndarray
        Cell centroid in R^4.
    occupancy : int
        Number of times a genome was placed in this cell.
    best_quality : float
        Best quality score seen in this cell.
    visit_count : int
        Total number of visits (evaluations mapped to this cell).
    """

    centroid: np.ndarray
    occupancy: int = 0
    best_quality: float = -np.inf
    visit_count: int = 0


class CVTTessellation:
    """Centroidal Voronoi Tessellation for the QD behavior descriptor space.

    Partitions the 4-D descriptor space R^4 into ``n_cells`` Voronoi
    cells using Lloyd's algorithm with k-means++ initialization.

    Parameters
    ----------
    n_cells : int
        Number of cells (default 1000).
    n_dims : int
        Dimensionality of the behavior descriptor space (default 4).
    n_samples : int
        Number of random samples for CVT initialization (default 10000).
    n_lloyd_iters : int
        Number of Lloyd's algorithm iterations (default 50).
    seed : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> cvt = CVTTessellation(n_cells=500, seed=42)
    >>> cvt.initialize()
    >>> cell_idx = cvt.find_cell(descriptor_array)
    """

    def __init__(
        self,
        n_cells: int = 1000,
        n_dims: int = 4,
        n_samples: int = 10000,
        n_lloyd_iters: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.n_cells = n_cells
        self.n_dims = n_dims
        self.n_samples = n_samples
        self.n_lloyd_iters = n_lloyd_iters
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self.centroids: Optional[np.ndarray] = None
        self._initialized = False

        self._cell_visit_counts: Optional[np.ndarray] = None
        self._cell_best_quality: Optional[np.ndarray] = None
        self._cell_occupancy: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        """Whether the tessellation has been initialized."""
        return self._initialized

    # ----- Initialization -----

    def initialize(self, bounds: Optional[np.ndarray] = None) -> None:
        """Initialize the CVT using k-means++ and Lloyd's algorithm.

        Parameters
        ----------
        bounds : np.ndarray, optional
            Shape (n_dims, 2) array of [low, high] bounds per dimension.
            Defaults to [0, 1] for fractions and [0, log2(3)] for entropy.
        """
        if bounds is None:
            bounds = np.array([
                [0.0, 1.0],    # frac_invariant
                [0.0, 1.0],    # frac_parametric
                [0.0, 1.0],    # frac_structural_emergent
                [0.0, np.log2(3)],  # entropy
            ])

        logger.info(
            "Initializing CVT: %d cells, %d samples, %d Lloyd iters",
            self.n_cells, self.n_samples, self.n_lloyd_iters,
        )

        # Generate random samples in the bounded space
        samples = np.zeros((self.n_samples, self.n_dims))
        for d in range(self.n_dims):
            samples[:, d] = self._rng.uniform(
                bounds[d, 0], bounds[d, 1], size=self.n_samples
            )

        # K-means++ initialization
        centroids = self._kmeans_plus_plus_init(samples)

        # Lloyd's algorithm
        centroids = self._lloyd_iterations(samples, centroids, bounds)

        self.centroids = centroids
        self._cell_visit_counts = np.zeros(self.n_cells, dtype=np.int64)
        self._cell_best_quality = np.full(self.n_cells, -np.inf)
        self._cell_occupancy = np.zeros(self.n_cells, dtype=np.int64)
        self._initialized = True

        logger.info("CVT initialization complete: %d centroids", self.n_cells)

    def _kmeans_plus_plus_init(self, samples: np.ndarray) -> np.ndarray:
        """K-means++ centroid initialization.

        Parameters
        ----------
        samples : np.ndarray
            Shape (n_samples, n_dims) data points.

        Returns
        -------
        np.ndarray
            Shape (n_cells, n_dims) initial centroids.
        """
        n = samples.shape[0]
        centroids = np.zeros((self.n_cells, self.n_dims))

        # First centroid: random sample
        first_idx = int(self._rng.integers(0, n))
        centroids[0] = samples[first_idx]

        # Remaining centroids: D^2 weighted sampling
        for k in range(1, self.n_cells):
            # Compute distances to nearest existing centroid
            dists = np.min(
                np.sum((samples[:, np.newaxis, :] - centroids[np.newaxis, :k, :]) ** 2, axis=2),
                axis=1,
            )
            # Probability proportional to D^2
            probs = dists / (dists.sum() + 1e-15)
            idx = int(self._rng.choice(n, p=probs))
            centroids[k] = samples[idx]

        return centroids

    def _lloyd_iterations(
        self,
        samples: np.ndarray,
        centroids: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Run Lloyd's algorithm to refine centroids.

        Parameters
        ----------
        samples : np.ndarray
            Shape (n_samples, n_dims).
        centroids : np.ndarray
            Shape (n_cells, n_dims) initial centroids.
        bounds : np.ndarray
            Shape (n_dims, 2) bounds for clamping.

        Returns
        -------
        np.ndarray
            Refined centroids.
        """
        for iteration in range(self.n_lloyd_iters):
            # Assignment step: find nearest centroid for each sample
            assignments = self._assign_samples(samples, centroids)

            # Update step: compute new centroids as cluster means
            new_centroids = np.copy(centroids)
            for k in range(self.n_cells):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = samples[mask].mean(axis=0)

            # Clamp to bounds
            for d in range(self.n_dims):
                new_centroids[:, d] = np.clip(
                    new_centroids[:, d], bounds[d, 0], bounds[d, 1]
                )

            # Check convergence
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
            max_shift = shift.max()
            centroids = new_centroids

            if max_shift < 1e-6:
                logger.debug("Lloyd's converged at iteration %d", iteration)
                break

        return centroids

    @staticmethod
    def _assign_samples(
        samples: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Assign samples to nearest centroids (vectorized).

        Parameters
        ----------
        samples : np.ndarray
            Shape (N, D).
        centroids : np.ndarray
            Shape (K, D).

        Returns
        -------
        np.ndarray
            Shape (N,) integer assignment array.
        """
        # Compute squared distances using broadcasting
        # ||s - c||^2 = ||s||^2 - 2*s.c + ||c||^2
        s_sq = np.sum(samples ** 2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        cross = samples @ centroids.T  # (N, K)
        sq_dists = s_sq - 2 * cross + c_sq
        return np.argmin(sq_dists, axis=1)

    # ----- Cell operations -----

    def find_cell(self, descriptor: np.ndarray) -> int:
        """Find the nearest CVT cell for a behavior descriptor.

        Parameters
        ----------
        descriptor : np.ndarray
            Shape (4,) behavior descriptor.

        Returns
        -------
        int
            Index of the nearest cell.

        Raises
        ------
        RuntimeError
            If the tessellation has not been initialized.
        """
        if not self._initialized or self.centroids is None:
            raise RuntimeError("CVT not initialized. Call initialize() first.")
        sq_dists = np.sum((self.centroids - descriptor) ** 2, axis=1)
        return int(np.argmin(sq_dists))

    def find_cells_batch(self, descriptors: np.ndarray) -> np.ndarray:
        """Find nearest CVT cells for a batch of descriptors.

        Parameters
        ----------
        descriptors : np.ndarray
            Shape (N, 4) array of behavior descriptors.

        Returns
        -------
        np.ndarray
            Shape (N,) integer array of cell indices.
        """
        if not self._initialized or self.centroids is None:
            raise RuntimeError("CVT not initialized. Call initialize() first.")
        return self._assign_samples(descriptors, self.centroids)

    def cell_distance(self, cell_a: int, cell_b: int) -> float:
        """Euclidean distance between two cell centroids.

        Parameters
        ----------
        cell_a, cell_b : int
            Cell indices.

        Returns
        -------
        float
        """
        if self.centroids is None:
            raise RuntimeError("CVT not initialized.")
        return float(np.linalg.norm(
            self.centroids[cell_a] - self.centroids[cell_b]
        ))

    def get_centroid(self, cell_idx: int) -> np.ndarray:
        """Return the centroid of a cell.

        Parameters
        ----------
        cell_idx : int
            Cell index.

        Returns
        -------
        np.ndarray
            Shape (4,) centroid.
        """
        if self.centroids is None:
            raise RuntimeError("CVT not initialized.")
        return self.centroids[cell_idx].copy()

    def record_visit(self, cell_idx: int, quality: float) -> None:
        """Record a visit to a cell.

        Parameters
        ----------
        cell_idx : int
            Cell index.
        quality : float
            Quality of the visiting genome.
        """
        if self._cell_visit_counts is None:
            raise RuntimeError("CVT not initialized.")
        self._cell_visit_counts[cell_idx] += 1
        self._cell_occupancy[cell_idx] = 1
        if quality > self._cell_best_quality[cell_idx]:
            self._cell_best_quality[cell_idx] = quality

    def get_visit_count(self, cell_idx: int) -> int:
        """Return the visit count for a cell.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        int
        """
        if self._cell_visit_counts is None:
            return 0
        return int(self._cell_visit_counts[cell_idx])

    # ----- Statistics -----

    def coverage(self) -> float:
        """Fraction of cells that have been occupied.

        Returns
        -------
        float
            Coverage in [0, 1].
        """
        if self._cell_occupancy is None:
            return 0.0
        return float(np.mean(self._cell_occupancy > 0))

    def n_occupied(self) -> int:
        """Number of occupied cells.

        Returns
        -------
        int
        """
        if self._cell_occupancy is None:
            return 0
        return int(np.sum(self._cell_occupancy > 0))

    def visit_count_stats(self) -> Dict[str, float]:
        """Summary statistics of visit counts across cells.

        Returns
        -------
        dict
            Keys: mean, std, min, max, median, total.
        """
        if self._cell_visit_counts is None:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "total": 0}

        counts = self._cell_visit_counts
        return {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "min": float(np.min(counts)),
            "max": float(np.max(counts)),
            "median": float(np.median(counts)),
            "total": float(np.sum(counts)),
        }

    def quality_stats(self) -> Dict[str, float]:
        """Summary statistics of best quality scores across occupied cells.

        Returns
        -------
        dict
        """
        if self._cell_best_quality is None:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        occupied_mask = self._cell_occupancy > 0
        if not np.any(occupied_mask):
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        q = self._cell_best_quality[occupied_mask]
        return {
            "mean": float(np.mean(q)),
            "std": float(np.std(q)),
            "min": float(np.min(q)),
            "max": float(np.max(q)),
        }

    def least_visited_cells(self, n: int = 10) -> np.ndarray:
        """Return indices of the n least-visited cells.

        Parameters
        ----------
        n : int
            Number of cells to return.

        Returns
        -------
        np.ndarray
            Cell indices sorted by ascending visit count.
        """
        if self._cell_visit_counts is None:
            return np.array([], dtype=np.int64)
        order = np.argsort(self._cell_visit_counts)
        return order[:n]

    def most_visited_cells(self, n: int = 10) -> np.ndarray:
        """Return indices of the n most-visited cells.

        Parameters
        ----------
        n : int

        Returns
        -------
        np.ndarray
        """
        if self._cell_visit_counts is None:
            return np.array([], dtype=np.int64)
        order = np.argsort(self._cell_visit_counts)[::-1]
        return order[:n]

    # ----- Visualization data -----

    def get_coverage_heatmap_data(self) -> Dict[str, Any]:
        """Return data for visualizing archive coverage.

        Returns a dictionary with centroid coordinates and occupancy
        info suitable for plotting.

        Returns
        -------
        dict
        """
        if not self._initialized or self.centroids is None:
            return {"centroids": [], "occupied": [], "visit_counts": []}

        return {
            "centroids": self.centroids.tolist(),
            "occupied": (self._cell_occupancy > 0).tolist(),
            "visit_counts": self._cell_visit_counts.tolist(),
            "best_quality": self._cell_best_quality.tolist(),
            "n_cells": self.n_cells,
            "n_occupied": self.n_occupied(),
            "coverage": self.coverage(),
        }

    def get_cell_boundaries_2d(
        self, dim_x: int = 0, dim_y: int = 1, resolution: int = 100
    ) -> Dict[str, np.ndarray]:
        """Compute 2-D Voronoi cell boundaries for visualization.

        Projects cells onto two chosen dimensions.

        Parameters
        ----------
        dim_x, dim_y : int
            Dimensions to project onto.
        resolution : int
            Grid resolution per axis.

        Returns
        -------
        dict
            Keys: grid_x, grid_y, cell_map (resolution x resolution).
        """
        if self.centroids is None:
            raise RuntimeError("CVT not initialized.")

        centroids_2d = self.centroids[:, [dim_x, dim_y]]
        x_range = (centroids_2d[:, 0].min() - 0.05, centroids_2d[:, 0].max() + 0.05)
        y_range = (centroids_2d[:, 1].min() - 0.05, centroids_2d[:, 1].max() + 0.05)

        gx = np.linspace(x_range[0], x_range[1], resolution)
        gy = np.linspace(y_range[0], y_range[1], resolution)
        grid_x, grid_y = np.meshgrid(gx, gy)

        # Flatten and compute nearest centroid in 2D
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        sq_dists = (
            np.sum(points ** 2, axis=1, keepdims=True)
            - 2 * points @ centroids_2d.T
            + np.sum(centroids_2d ** 2, axis=1, keepdims=True).T
        )
        cell_map = np.argmin(sq_dists, axis=1).reshape(resolution, resolution)

        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "cell_map": cell_map,
        }

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tessellation.

        Returns
        -------
        dict
        """
        d: Dict[str, Any] = {
            "n_cells": self.n_cells,
            "n_dims": self.n_dims,
            "n_samples": self.n_samples,
            "n_lloyd_iters": self.n_lloyd_iters,
            "seed": self.seed,
            "initialized": self._initialized,
        }
        if self.centroids is not None:
            d["centroids"] = self.centroids.tolist()
        if self._cell_visit_counts is not None:
            d["visit_counts"] = self._cell_visit_counts.tolist()
        if self._cell_best_quality is not None:
            d["best_quality"] = self._cell_best_quality.tolist()
        if self._cell_occupancy is not None:
            d["occupancy"] = self._cell_occupancy.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVTTessellation":
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        CVTTessellation
        """
        cvt = cls(
            n_cells=d["n_cells"],
            n_dims=d.get("n_dims", 4),
            n_samples=d.get("n_samples", 10000),
            n_lloyd_iters=d.get("n_lloyd_iters", 50),
            seed=d.get("seed"),
        )
        if d.get("initialized", False) and "centroids" in d:
            cvt.centroids = np.array(d["centroids"], dtype=np.float64)
            cvt._cell_visit_counts = np.array(
                d.get("visit_counts", [0] * cvt.n_cells), dtype=np.int64
            )
            cvt._cell_best_quality = np.array(
                d.get("best_quality", [-np.inf] * cvt.n_cells), dtype=np.float64
            )
            cvt._cell_occupancy = np.array(
                d.get("occupancy", [0] * cvt.n_cells), dtype=np.int64
            )
            cvt._initialized = True
        return cvt

    def reset_stats(self) -> None:
        """Reset all visit counts and occupancy stats without changing centroids."""
        if self._initialized:
            self._cell_visit_counts = np.zeros(self.n_cells, dtype=np.int64)
            self._cell_best_quality = np.full(self.n_cells, -np.inf)
            self._cell_occupancy = np.zeros(self.n_cells, dtype=np.int64)

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "uninitialized"
        occ = self.n_occupied() if self._initialized else 0
        return (
            f"CVTTessellation(cells={self.n_cells}, dims={self.n_dims}, "
            f"occupied={occ}, {status})"
        )


# ---------------------------------------------------------------------------
# Adaptive CVT
# ---------------------------------------------------------------------------


class AdaptiveCVT(CVTTessellation):
    """Adaptive CVT with density-aware cell splitting and merging.

    Extends the base CVT by allowing cells to be split in high-density
    regions and merged in low-density regions, improving coverage of
    the behavior space where interesting patterns concentrate.

    Parameters
    ----------
    n_cells : int
        Initial number of cells.
    n_dims : int
        Descriptor dimensionality.
    min_cells : int
        Minimum number of cells (prevents over-merging).
    max_cells : int
        Maximum number of cells (prevents over-splitting).
    split_threshold : int
        Split a cell if its visit count exceeds this threshold.
    merge_threshold : int
        Merge two adjacent cells if both have visit count below this.
    seed : int or None
    """

    def __init__(
        self,
        n_cells: int = 1000,
        n_dims: int = 4,
        min_cells: int = 200,
        max_cells: int = 5000,
        split_threshold: int = 50,
        merge_threshold: int = 2,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_cells=n_cells, n_dims=n_dims, seed=seed, **kwargs
        )
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self._split_history: List[Tuple[int, int, int]] = []
        self._merge_history: List[Tuple[int, int, int]] = []
        self._adaptation_step = 0

    def adapt(self) -> Dict[str, int]:
        """Perform one round of adaptive cell splitting and merging.

        Splits high-traffic cells and merges low-traffic adjacent cells
        to redistribute resolution toward interesting regions of the
        behavior space.

        Returns
        -------
        dict
            Keys: n_splits, n_merges, n_cells_after.
        """
        if not self._initialized or self.centroids is None:
            raise RuntimeError("CVT not initialized.")

        n_splits = 0
        n_merges = 0
        self._adaptation_step += 1

        # Phase 1: Split high-traffic cells
        if self.n_cells < self.max_cells:
            n_splits = self._split_busy_cells()

        # Phase 2: Merge low-traffic adjacent cells
        if self.n_cells > self.min_cells:
            n_merges = self._merge_quiet_cells()

        logger.info(
            "Adaptive CVT step %d: %d splits, %d merges → %d cells",
            self._adaptation_step, n_splits, n_merges, self.n_cells,
        )

        return {
            "n_splits": n_splits,
            "n_merges": n_merges,
            "n_cells_after": self.n_cells,
        }

    def _split_busy_cells(self) -> int:
        """Split cells with visit counts above the threshold.

        Returns
        -------
        int
            Number of cells split.
        """
        if self._cell_visit_counts is None or self.centroids is None:
            return 0

        cells_to_split = np.where(
            self._cell_visit_counts > self.split_threshold
        )[0]

        n_splits = 0
        for cell_idx in cells_to_split:
            if self.n_cells >= self.max_cells:
                break

            centroid = self.centroids[cell_idx]
            # Split by adding a new cell offset in a random direction
            direction = self._rng.standard_normal(self.n_dims)
            direction /= (np.linalg.norm(direction) + 1e-15)

            # Offset proportional to cell "radius" (approximated)
            mean_dist = self._estimate_cell_radius(cell_idx)
            offset = direction * mean_dist * 0.5

            new_centroid = centroid + offset
            new_centroid = np.clip(new_centroid, 0.0, None)

            # Add the new cell
            self.centroids = np.vstack([self.centroids, new_centroid[np.newaxis, :]])
            self._cell_visit_counts = np.append(self._cell_visit_counts, 0)
            self._cell_best_quality = np.append(self._cell_best_quality, -np.inf)
            self._cell_occupancy = np.append(self._cell_occupancy, 0)

            # Move original centroid slightly
            self.centroids[cell_idx] = centroid - offset * 0.3

            new_idx = self.n_cells
            self.n_cells += 1
            self._split_history.append(
                (self._adaptation_step, cell_idx, new_idx)
            )
            n_splits += 1

        return n_splits

    def _merge_quiet_cells(self) -> int:
        """Merge pairs of low-traffic adjacent cells.

        Returns
        -------
        int
            Number of cells merged.
        """
        if self._cell_visit_counts is None or self.centroids is None:
            return 0

        quiet_cells = np.where(
            self._cell_visit_counts < self.merge_threshold
        )[0]

        if len(quiet_cells) < 2:
            return 0

        n_merges = 0
        merged: set = set()

        # Find nearest pairs among quiet cells
        for i in range(len(quiet_cells)):
            if quiet_cells[i] in merged:
                continue
            if self.n_cells <= self.min_cells:
                break

            cell_a = quiet_cells[i]
            best_dist = np.inf
            best_partner = -1

            for j in range(i + 1, len(quiet_cells)):
                if quiet_cells[j] in merged:
                    continue
                cell_b = quiet_cells[j]
                dist = float(np.linalg.norm(
                    self.centroids[cell_a] - self.centroids[cell_b]
                ))
                if dist < best_dist:
                    best_dist = dist
                    best_partner = cell_b

            if best_partner >= 0:
                # Merge: average centroids, remove one
                self.centroids[cell_a] = (
                    self.centroids[cell_a] + self.centroids[best_partner]
                ) / 2.0
                self._cell_visit_counts[cell_a] += self._cell_visit_counts[best_partner]

                merged.add(best_partner)
                self._merge_history.append(
                    (self._adaptation_step, cell_a, best_partner)
                )
                n_merges += 1

        # Actually remove merged cells
        if merged:
            keep_mask = np.ones(self.n_cells, dtype=bool)
            for idx in merged:
                keep_mask[idx] = False
            self.centroids = self.centroids[keep_mask]
            self._cell_visit_counts = self._cell_visit_counts[keep_mask]
            self._cell_best_quality = self._cell_best_quality[keep_mask]
            self._cell_occupancy = self._cell_occupancy[keep_mask]
            self.n_cells = len(self.centroids)

        return n_merges

    def _estimate_cell_radius(self, cell_idx: int) -> float:
        """Estimate the radius of a cell using nearest-neighbor centroids.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        float
            Approximate cell radius.
        """
        if self.centroids is None:
            return 0.1
        dists = np.sqrt(np.sum(
            (self.centroids - self.centroids[cell_idx]) ** 2, axis=1
        ))
        dists[cell_idx] = np.inf  # Exclude self
        k = min(5, len(dists) - 1)
        nearest_dists = np.partition(dists, k)[:k]
        return float(np.mean(nearest_dists)) / 2.0

    def get_adaptation_history(self) -> Dict[str, Any]:
        """Return the history of adaptive operations.

        Returns
        -------
        dict
        """
        return {
            "adaptation_steps": self._adaptation_step,
            "splits": self._split_history,
            "merges": self._merge_history,
            "current_cells": self.n_cells,
        }

    def coverage_guided_refine(
        self,
        target_coverage: float = 0.5,
        max_iters: int = 10,
    ) -> int:
        """Iteratively adapt until target coverage is reached.

        Parameters
        ----------
        target_coverage : float
            Target fraction of occupied cells.
        max_iters : int
            Maximum adaptation iterations.

        Returns
        -------
        int
            Number of adaptation rounds performed.
        """
        for i in range(max_iters):
            current = self.coverage()
            if current >= target_coverage:
                logger.info(
                    "Coverage target %.2f reached (%.2f) after %d rounds",
                    target_coverage, current, i,
                )
                return i

            self.adapt()

        return max_iters

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "uninitialized"
        return (
            f"AdaptiveCVT(cells={self.n_cells}, "
            f"range=[{self.min_cells}, {self.max_cells}], "
            f"splits={len(self._split_history)}, "
            f"merges={len(self._merge_history)}, {status})"
        )
