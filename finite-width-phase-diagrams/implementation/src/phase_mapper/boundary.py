"""Phase boundary extraction from grid sweep results.

Provides:
  - BoundaryPoint: a single point on a detected boundary
  - BoundaryCurve: ordered curve of boundary points with confidence bands
  - BoundaryConfig: thresholds and smoothing parameters
  - BoundaryExtractor: gradient-based boundary detection and comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.distance import directed_hausdorff


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class BoundaryPoint:
    """A single point on a detected phase boundary.

    Parameters
    ----------
    coordinates : np.ndarray
        Parameter-space coordinates.
    gradient_magnitude : float
        Magnitude of the order-parameter gradient at this point.
    confidence : float
        Confidence score for this being a genuine boundary point (0–1).
    neighbors : list of int
        Indices of neighbouring boundary points in the parent curve.
    """

    coordinates: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gradient_magnitude: float = 0.0
    confidence: float = 1.0
    neighbors: List[int] = field(default_factory=list)


@dataclass
class BoundaryCurve:
    """Ordered curve of boundary points with associated metadata.

    Parameters
    ----------
    points : list of BoundaryPoint
        Boundary points in curve order.
    smoothed_coords : np.ndarray or None
        Smoothed coordinate array of shape ``(n, 2)``.
    confidence_band_width : np.ndarray or None
        Width of the confidence band at each point.
    length : float
        Total arclength of the curve.
    closed : bool
        Whether the curve forms a closed loop.
    """

    points: List[BoundaryPoint] = field(default_factory=list)
    smoothed_coords: Optional[np.ndarray] = None
    confidence_band_width: Optional[np.ndarray] = None
    length: float = 0.0
    closed: bool = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def raw_coords(self) -> np.ndarray:
        """Return raw coordinates as a 2-D array.

        Returns
        -------
        np.ndarray
            Shape ``(n_points, dim)``.
        """
        return np.array([p.coordinates for p in self.points])

    def effective_coords(self) -> np.ndarray:
        """Return smoothed coordinates if available, else raw.

        Returns
        -------
        np.ndarray
        """
        if self.smoothed_coords is not None:
            return self.smoothed_coords
        return self.raw_coords()


@dataclass
class BoundaryConfig:
    """Configuration for boundary extraction.

    Parameters
    ----------
    gradient_threshold : float
        Fraction of max gradient used to threshold boundary pixels.
    smoothing_sigma : float
        Standard deviation of Gaussian smoothing for the final curve.
    min_curve_length : int
        Minimum number of points for a valid boundary curve.
    confidence_level : float
        Confidence level for the confidence band (0–1).
    """

    gradient_threshold: float = 0.3
    smoothing_sigma: float = 1.0
    min_curve_length: int = 3
    confidence_level: float = 0.95


# ======================================================================
# Boundary extractor
# ======================================================================


class BoundaryExtractor:
    """Extract phase boundaries from grid sweep results.

    Detects boundaries as ridges of high order-parameter gradient and
    connects them into ordered curves with confidence information.

    Parameters
    ----------
    config : BoundaryConfig
        Extraction parameters.
    """

    def __init__(self, config: Optional[BoundaryConfig] = None) -> None:
        self.config = config or BoundaryConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_grid(
        self,
        sweep_result: Any,
    ) -> List[BoundaryCurve]:
        """Identify boundary curves from a grid sweep result.

        Parameters
        ----------
        sweep_result : SweepResult
            Must expose ``to_array()``, ``parameter_names``,
            ``grid_shape``, and ``grid_points`` attributes (as produced
            by :class:`GridSweeper`).

        Returns
        -------
        List[BoundaryCurve]
            Detected boundary curves sorted by decreasing length.
        """
        values = sweep_result.to_array()
        grad_field = self._compute_gradient_field(values)
        clusters = self._threshold_and_cluster(grad_field)

        # Build coordinate grids for mapping indices -> parameter values
        from .grid_sweep import GridConfig  # local to avoid circular at import time

        param_names = sweep_result.parameter_names
        grids = self._extract_grids(sweep_result)

        curves: List[BoundaryCurve] = []
        for cluster_indices in clusters:
            raw_points = self._indices_to_boundary_points(
                cluster_indices, grids, grad_field, values,
            )
            curve = self._connect_points(raw_points)
            if len(curve.points) < self.config.min_curve_length:
                continue
            curve = self._smooth_curve(curve, self.config.smoothing_sigma)
            curve = self._compute_confidence_band(curve, values, grids)
            curves.append(curve)

        curves.sort(key=lambda c: c.length, reverse=True)
        return curves

    def compare_boundaries(
        self,
        predicted: List[BoundaryCurve],
        ground_truth: List[BoundaryCurve],
    ) -> Dict[str, float]:
        """Compare predicted boundary curves against ground truth.

        Parameters
        ----------
        predicted : list of BoundaryCurve
        ground_truth : list of BoundaryCurve

        Returns
        -------
        dict
            Comparison metrics: ``hausdorff_distance``,
            ``mean_distance``, ``coverage``.
        """
        pred_pts = self._collect_coords(predicted)
        gt_pts = self._collect_coords(ground_truth)

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return {
                "hausdorff_distance": float("inf"),
                "mean_distance": float("inf"),
                "coverage": 0.0,
            }

        hausdorff_fwd = directed_hausdorff(pred_pts, gt_pts)[0]
        hausdorff_rev = directed_hausdorff(gt_pts, pred_pts)[0]
        hausdorff = max(hausdorff_fwd, hausdorff_rev)

        # Mean distance: for each ground-truth point, nearest predicted
        from scipy.spatial import cKDTree

        tree = cKDTree(pred_pts)
        dists, _ = tree.query(gt_pts)
        mean_dist = float(np.mean(dists))

        # Coverage: fraction of GT points within one grid spacing of a
        # predicted point
        if len(pred_pts) > 1:
            nn_dists, _ = cKDTree(pred_pts).query(pred_pts, k=2)
            spacing = float(np.median(nn_dists[:, 1]))
        else:
            spacing = 1.0
        coverage = float(np.mean(dists < spacing))

        return {
            "hausdorff_distance": hausdorff,
            "mean_distance": mean_dist,
            "coverage": coverage,
        }

    # ------------------------------------------------------------------
    # Internal: gradient field
    # ------------------------------------------------------------------

    def _compute_gradient_field(self, values: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude on the N-D grid.

        Parameters
        ----------
        values : np.ndarray
            Order parameter values on a regular grid.

        Returns
        -------
        np.ndarray
            Gradient magnitude array of same shape as *values*.
        """
        grads = np.gradient(values)
        if isinstance(grads, np.ndarray):
            grads = [grads]
        mag = np.zeros_like(values, dtype=np.float64)
        for g in grads:
            mag += g ** 2
        return np.sqrt(mag)

    # ------------------------------------------------------------------
    # Internal: thresholding and clustering
    # ------------------------------------------------------------------

    def _threshold_and_cluster(
        self,
        gradient_field: np.ndarray,
    ) -> List[List[Tuple[int, ...]]]:
        """Threshold the gradient field and cluster connected components.

        Parameters
        ----------
        gradient_field : np.ndarray
            Gradient magnitude array.

        Returns
        -------
        list of list of tuple
            Each inner list is a cluster of N-D index tuples.
        """
        max_grad = np.max(gradient_field)
        if max_grad == 0:
            return []
        thresh = self.config.gradient_threshold * max_grad
        binary = (gradient_field > thresh).astype(np.int32)

        labelled, n_clusters = label(binary)
        clusters: List[List[Tuple[int, ...]]] = []
        for k in range(1, n_clusters + 1):
            indices = list(zip(*np.where(labelled == k)))
            clusters.append(indices)
        return clusters

    # ------------------------------------------------------------------
    # Internal: index -> BoundaryPoint mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grids(sweep_result: Any) -> List[np.ndarray]:
        """Extract 1-D coordinate grids from a sweep result.

        Parameters
        ----------
        sweep_result : SweepResult

        Returns
        -------
        list of np.ndarray
        """
        param_names = sweep_result.parameter_names
        shape = sweep_result.grid_shape
        n_total = 1
        for s in shape:
            n_total *= s

        grids: List[np.ndarray] = []
        for dim, name in enumerate(param_names):
            seen: List[float] = []
            stride = 1
            for d2 in range(dim + 1, len(param_names)):
                stride *= shape[d2]
            for gp in sweep_result.grid_points:
                val = gp.coordinates[name]
                if val not in seen:
                    seen.append(val)
                if len(seen) == shape[dim]:
                    break
            grids.append(np.array(seen))
        return grids

    def _indices_to_boundary_points(
        self,
        indices: List[Tuple[int, ...]],
        grids: List[np.ndarray],
        grad_field: np.ndarray,
        values: np.ndarray,
    ) -> List[BoundaryPoint]:
        """Convert grid indices to BoundaryPoint objects.

        Parameters
        ----------
        indices : list of tuple
        grids : list of np.ndarray
        grad_field : np.ndarray
        values : np.ndarray

        Returns
        -------
        list of BoundaryPoint
        """
        max_grad = np.max(grad_field)
        points: List[BoundaryPoint] = []
        for idx in indices:
            coords = np.array([grids[d][i] for d, i in enumerate(idx)])
            gm = float(grad_field[idx])
            conf = gm / (max_grad + 1e-30)
            points.append(
                BoundaryPoint(
                    coordinates=coords,
                    gradient_magnitude=gm,
                    confidence=conf,
                )
            )
        return points

    # ------------------------------------------------------------------
    # Internal: curve ordering
    # ------------------------------------------------------------------

    def _connect_points(
        self,
        points: List[BoundaryPoint],
    ) -> BoundaryCurve:
        """Order boundary points into a curve via nearest-neighbour chain.

        Parameters
        ----------
        points : list of BoundaryPoint

        Returns
        -------
        BoundaryCurve
        """
        if len(points) <= 1:
            return BoundaryCurve(points=list(points), length=0.0)

        coords = np.array([p.coordinates for p in points])
        n = len(points)
        visited = np.zeros(n, dtype=bool)
        order = [0]
        visited[0] = True

        for _ in range(n - 1):
            last = order[-1]
            dists = np.linalg.norm(coords - coords[last], axis=-1)
            dists[visited] = np.inf
            nearest = int(np.argmin(dists))
            order.append(nearest)
            visited[nearest] = True

        ordered_points = [points[i] for i in order]

        # Set neighbor indices
        for k, pt in enumerate(ordered_points):
            nbrs: List[int] = []
            if k > 0:
                nbrs.append(k - 1)
            if k < len(ordered_points) - 1:
                nbrs.append(k + 1)
            pt.neighbors = nbrs

        # Compute arclength
        total_length = 0.0
        for k in range(1, len(ordered_points)):
            total_length += float(
                np.linalg.norm(
                    ordered_points[k].coordinates
                    - ordered_points[k - 1].coordinates
                )
            )

        # Check if closed
        closure_dist = float(
            np.linalg.norm(
                ordered_points[-1].coordinates - ordered_points[0].coordinates
            )
        )
        avg_spacing = total_length / max(len(ordered_points) - 1, 1)
        closed = closure_dist < 2.0 * avg_spacing if avg_spacing > 0 else False

        return BoundaryCurve(
            points=ordered_points, length=total_length, closed=closed,
        )

    # ------------------------------------------------------------------
    # Internal: smoothing
    # ------------------------------------------------------------------

    def _smooth_curve(
        self,
        curve: BoundaryCurve,
        sigma: float,
    ) -> BoundaryCurve:
        """Apply Gaussian smoothing to a boundary curve.

        Parameters
        ----------
        curve : BoundaryCurve
        sigma : float
            Standard deviation of the Gaussian kernel (in index units).

        Returns
        -------
        BoundaryCurve
            Same curve with ``smoothed_coords`` populated.
        """
        raw = curve.raw_coords()
        if len(raw) < 3 or sigma <= 0:
            curve.smoothed_coords = raw.copy()
            return curve

        smoothed = np.empty_like(raw)
        for dim in range(raw.shape[1]):
            smoothed[:, dim] = gaussian_filter(raw[:, dim], sigma=sigma)

        curve.smoothed_coords = smoothed
        # Recompute length from smoothed coords
        diffs = np.diff(smoothed, axis=0)
        curve.length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return curve

    # ------------------------------------------------------------------
    # Internal: confidence band
    # ------------------------------------------------------------------

    def _compute_confidence_band(
        self,
        curve: BoundaryCurve,
        values: np.ndarray,
        grids: List[np.ndarray],
    ) -> BoundaryCurve:
        """Estimate confidence band widths from local curvature.

        The width at each point is inversely proportional to the local
        gradient magnitude, normalised by the grid spacing.

        Parameters
        ----------
        curve : BoundaryCurve
        values : np.ndarray
            Order parameter values grid.
        grids : list of np.ndarray
            Coordinate grids.

        Returns
        -------
        BoundaryCurve
            Same curve with ``confidence_band_width`` populated.
        """
        n = len(curve.points)
        widths = np.empty(n, dtype=np.float64)

        # Typical grid spacing
        spacings = []
        for g in grids:
            if len(g) > 1:
                spacings.append(float(np.median(np.diff(g))))
        avg_spacing = float(np.mean(spacings)) if spacings else 1.0

        for i, pt in enumerate(curve.points):
            gm = pt.gradient_magnitude
            if gm > 0:
                widths[i] = avg_spacing / gm
            else:
                widths[i] = avg_spacing

        # Scale so that the median width corresponds to the confidence level
        median_w = float(np.median(widths))
        if median_w > 0:
            widths *= avg_spacing / median_w

        curve.confidence_band_width = widths
        return curve

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_coords(curves: List[BoundaryCurve]) -> np.ndarray:
        """Collect all coordinates from a list of curves into one array.

        Parameters
        ----------
        curves : list of BoundaryCurve

        Returns
        -------
        np.ndarray
            Shape ``(total_points, dim)``.
        """
        all_pts: List[np.ndarray] = []
        for c in curves:
            all_pts.append(c.effective_coords())
        if not all_pts:
            return np.empty((0, 2))
        return np.concatenate(all_pts, axis=0)
