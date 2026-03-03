"""
Grid construction strategies for DP-Forge mechanism synthesis.

Each strategy implements the :class:`GridStrategy` protocol, producing a
sorted array of grid points and their associated bin widths for numerical
integration.  The bin-width vector ``w`` satisfies ``sum(w) ≈ y[-1] - y[0]``
and is used by the LP builder to weight the discretised probability table.

Strategies
----------
- :class:`UniformGrid` — evenly spaced points; simplest, O(B/k) error.
- :class:`ChebyshevGrid` — Chebyshev nodes; reduced Runge phenomenon.
- :class:`MassAdaptiveGrid` — concentrate points where mechanism mass is
  high; requires an existing mechanism estimate.
- :class:`CurvatureAdaptiveGrid` — concentrate points where the mechanism
  probability changes rapidly (large second derivative).
- :class:`TailPrunedGrid` — remove far-tail points with negligible mass
  below a threshold, yielding a smaller LP.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_MIN_K: int = 2


# ---------------------------------------------------------------------------
# GridResult — return type for all strategies
# ---------------------------------------------------------------------------


@dataclass
class GridResult:
    """Result of a grid construction strategy.

    Attributes:
        points: Sorted grid point locations, shape ``(k,)``.
        widths: Bin widths for trapezoidal / midpoint integration, shape ``(k,)``.
        metadata: Strategy-specific metadata (e.g., pruning statistics).
    """

    points: npt.NDArray[np.float64]
    widths: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.widths = np.asarray(self.widths, dtype=np.float64)
        if self.points.ndim != 1 or self.widths.ndim != 1:
            raise ValueError("points and widths must be 1-D arrays")
        if len(self.points) != len(self.widths):
            raise ValueError(
                f"points ({len(self.points)}) and widths ({len(self.widths)}) "
                f"must have the same length"
            )
        if len(self.points) < _MIN_K:
            raise ValueError(f"grid must have at least {_MIN_K} points")

    @property
    def k(self) -> int:
        """Number of grid points."""
        return len(self.points)

    @property
    def span(self) -> float:
        """Total span of the grid."""
        return float(self.points[-1] - self.points[0])

    def __repr__(self) -> str:
        return f"GridResult(k={self.k}, span=[{self.points[0]:.4f}, {self.points[-1]:.4f}])"


# ---------------------------------------------------------------------------
# GridStrategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GridStrategy(Protocol):
    """Protocol for grid construction strategies.

    Any class implementing this protocol can be plugged into the adaptive
    grid refiner as the grid builder.
    """

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        **kwargs: Any,
    ) -> GridResult:
        """Build a grid of *k* points covering *f_range*.

        Parameters
        ----------
        f_range : tuple (f_min, f_max)
            The output value range (before padding).
        k : int
            Number of grid points to create.
        **kwargs
            Strategy-specific parameters.

        Returns
        -------
        GridResult
            The constructed grid with points and bin widths.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_midpoint_widths(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute midpoint-rule bin widths for a sorted grid.

    For interior points, the width is the average of the two adjacent gaps.
    For boundary points, the width is the adjacent gap.

    Parameters
    ----------
    points : array of shape (k,)
        Sorted grid points.

    Returns
    -------
    widths : array of shape (k,)
        Bin widths.
    """
    k = len(points)
    if k < 2:
        raise ValueError("Need at least 2 points for midpoint widths")
    gaps = np.diff(points)
    widths = np.empty(k, dtype=np.float64)
    widths[0] = gaps[0]
    widths[-1] = gaps[-1]
    if k > 2:
        widths[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0
    return widths


def _pad_range(
    f_min: float, f_max: float, padding: float
) -> Tuple[float, float]:
    """Pad a range symmetrically.

    If f_min == f_max (degenerate), expand by max(1, |f_min|) on each side.
    """
    if f_min == f_max:
        half = max(1.0, abs(f_min))
        f_min -= half
        f_max += half
    return f_min - padding, f_max + padding


# ---------------------------------------------------------------------------
# UniformGrid
# ---------------------------------------------------------------------------


class UniformGrid:
    """Evenly spaced grid on a padded output range.

    The grid spans ``[f_min - B, f_max + B]`` where ``B`` is a configurable
    padding that ensures adequate support in the mechanism tails.

    Parameters
    ----------
    padding : float
        Absolute padding on each side of the query value range.
        If ``None``, defaults to ``0.5 * (f_max - f_min) / (k - 1)``.
    """

    def __init__(self, padding: Optional[float] = None) -> None:
        self._padding = padding

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        **kwargs: Any,
    ) -> GridResult:
        """Build a uniform grid.

        Parameters
        ----------
        f_range : (f_min, f_max)
            Raw query output range.
        k : int
            Number of grid points (must be >= 2).

        Returns
        -------
        GridResult with evenly spaced points and uniform bin widths.
        """
        if k < _MIN_K:
            raise ConfigurationError(
                f"k must be >= {_MIN_K}, got {k}",
                parameter="k",
                value=k,
                constraint=f">= {_MIN_K}",
            )
        f_min, f_max = f_range
        if self._padding is not None:
            pad = self._padding
        else:
            span = f_max - f_min
            if span == 0:
                span = max(2.0, 2.0 * abs(f_min))
            pad = 0.5 * span / max(k - 1, 1)

        lo, hi = _pad_range(f_min, f_max, pad)
        points = np.linspace(lo, hi, k)
        widths = _compute_midpoint_widths(points)
        return GridResult(points=points, widths=widths, metadata={"strategy": "uniform"})

    def __repr__(self) -> str:
        pad = f"padding={self._padding}" if self._padding is not None else "auto"
        return f"UniformGrid({pad})"


# ---------------------------------------------------------------------------
# ChebyshevGrid
# ---------------------------------------------------------------------------


class ChebyshevGrid:
    """Chebyshev nodes on a padded output range.

    Chebyshev nodes cluster near the boundaries of the interval, reducing
    the Runge phenomenon when the mechanism probability density is
    interpolated as a polynomial.  The nodes are given by::

        y_j = (lo + hi)/2 + (hi - lo)/2 * cos((2j+1)/(2k) * π),  j = 0..k-1

    Parameters
    ----------
    padding : float or None
        Absolute padding on each side.  ``None`` for auto-padding.
    """

    def __init__(self, padding: Optional[float] = None) -> None:
        self._padding = padding

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        **kwargs: Any,
    ) -> GridResult:
        """Build a Chebyshev node grid.

        Parameters
        ----------
        f_range : (f_min, f_max)
            Raw query output range.
        k : int
            Number of Chebyshev nodes (>= 2).

        Returns
        -------
        GridResult with Chebyshev-spaced points and midpoint widths.
        """
        if k < _MIN_K:
            raise ConfigurationError(
                f"k must be >= {_MIN_K}, got {k}",
                parameter="k",
                value=k,
                constraint=f">= {_MIN_K}",
            )
        f_min, f_max = f_range
        if self._padding is not None:
            pad = self._padding
        else:
            span = f_max - f_min
            if span == 0:
                span = max(2.0, 2.0 * abs(f_min))
            pad = 0.5 * span / max(k - 1, 1)

        lo, hi = _pad_range(f_min, f_max, pad)
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0

        # Chebyshev nodes of the first kind
        j = np.arange(k, dtype=np.float64)
        nodes = mid + half * np.cos((2.0 * j + 1.0) / (2.0 * k) * np.pi)
        # Sort ascending
        nodes = np.sort(nodes)
        widths = _compute_midpoint_widths(nodes)
        return GridResult(
            points=nodes,
            widths=widths,
            metadata={"strategy": "chebyshev"},
        )

    def __repr__(self) -> str:
        pad = f"padding={self._padding}" if self._padding is not None else "auto"
        return f"ChebyshevGrid({pad})"


# ---------------------------------------------------------------------------
# MassAdaptiveGrid
# ---------------------------------------------------------------------------


class MassAdaptiveGrid:
    """Concentrate grid points where mechanism mass is high.

    Given an existing mechanism probability table ``p`` on a coarse grid,
    this strategy places more points in output bins where
    ``max_i p[i][j]`` exceeds a threshold.  The density of new points in
    a region is proportional to the mass in that region.

    Parameters
    ----------
    mass_threshold : float
        Bins with ``max_i p[i][j]`` below this threshold receive the
        base density; bins above it receive proportionally more.
    min_density : float
        Minimum fraction of points allocated to low-mass regions,
        preventing total neglect of the tails.
    padding : float or None
        Absolute padding.  ``None`` for auto-padding.
    """

    def __init__(
        self,
        mass_threshold: float = 0.01,
        min_density: float = 0.1,
        padding: Optional[float] = None,
    ) -> None:
        if mass_threshold <= 0:
            raise ValueError(f"mass_threshold must be > 0, got {mass_threshold}")
        if not (0.0 < min_density < 1.0):
            raise ValueError(f"min_density must be in (0, 1), got {min_density}")
        self._mass_threshold = mass_threshold
        self._min_density = min_density
        self._padding = padding

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        *,
        mechanism: Optional[npt.NDArray[np.float64]] = None,
        old_grid: Optional[npt.NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> GridResult:
        """Build a mass-adaptive grid.

        Parameters
        ----------
        f_range : (f_min, f_max)
            Raw query output range.
        k : int
            Target number of grid points (>= 2).
        mechanism : array of shape (n, k_old), optional
            Mechanism probability table from a previous refinement level.
            If ``None``, falls back to uniform spacing.
        old_grid : array of shape (k_old,), optional
            Grid points corresponding to ``mechanism``.  Required if
            ``mechanism`` is provided.

        Returns
        -------
        GridResult with mass-adaptive spacing.
        """
        if k < _MIN_K:
            raise ConfigurationError(
                f"k must be >= {_MIN_K}, got {k}",
                parameter="k",
                value=k,
                constraint=f">= {_MIN_K}",
            )
        if mechanism is None or old_grid is None:
            logger.debug("MassAdaptiveGrid: no mechanism provided, falling back to uniform")
            return UniformGrid(padding=self._padding).build(f_range, k)

        mechanism = np.asarray(mechanism, dtype=np.float64)
        old_grid = np.asarray(old_grid, dtype=np.float64)

        if mechanism.ndim != 2 or mechanism.shape[1] != len(old_grid):
            raise ValueError(
                f"mechanism shape {mechanism.shape} incompatible with "
                f"old_grid length {len(old_grid)}"
            )

        # Compute per-bin mass: max over databases
        bin_mass = np.max(mechanism, axis=0)  # shape (k_old,)

        # Determine padded range
        f_min, f_max = f_range
        if self._padding is not None:
            pad = self._padding
        else:
            span = f_max - f_min
            if span == 0:
                span = max(2.0, 2.0 * abs(f_min))
            pad = 0.5 * span / max(k - 1, 1)
        lo, hi = _pad_range(f_min, f_max, pad)

        # Allocate points proportional to mass density
        # Ensure minimum density for low-mass regions
        weight = np.where(
            bin_mass > self._mass_threshold, bin_mass, self._mass_threshold * 0.1
        )
        weight = weight / weight.sum()

        # Reserve min_density fraction for uniform coverage
        k_uniform = max(_MIN_K, int(round(k * self._min_density)))
        k_adaptive = k - k_uniform

        # Uniform component
        uniform_pts = np.linspace(lo, hi, k_uniform)

        # Adaptive component: sample from CDF of the mass distribution
        if k_adaptive > 0:
            # Build CDF on old grid
            cdf = np.cumsum(weight)
            cdf = cdf / cdf[-1]

            # Quantile positions for adaptive points
            quantiles = np.linspace(0.0, 1.0, k_adaptive + 2)[1:-1]
            adaptive_pts = np.interp(quantiles, cdf, old_grid)
            # Clip to padded range
            adaptive_pts = np.clip(adaptive_pts, lo, hi)

            # Merge and deduplicate
            all_pts = np.concatenate([uniform_pts, adaptive_pts])
        else:
            all_pts = uniform_pts

        # Sort and remove near-duplicates
        all_pts = np.sort(all_pts)
        if len(all_pts) > 1:
            min_gap = (hi - lo) / (10.0 * k)
            unique = [all_pts[0]]
            for pt in all_pts[1:]:
                if pt - unique[-1] > min_gap:
                    unique.append(pt)
            all_pts = np.array(unique, dtype=np.float64)

        # Ensure we have exactly k points by interpolating if needed
        if len(all_pts) < _MIN_K:
            all_pts = np.linspace(lo, hi, _MIN_K)
        if len(all_pts) != k:
            # Resample to get exactly k points via linear interpolation on indices
            old_idx = np.linspace(0, 1, len(all_pts))
            new_idx = np.linspace(0, 1, k)
            all_pts = np.interp(new_idx, old_idx, all_pts)

        widths = _compute_midpoint_widths(all_pts)
        return GridResult(
            points=all_pts,
            widths=widths,
            metadata={
                "strategy": "mass_adaptive",
                "k_uniform": k_uniform,
                "k_adaptive": k_adaptive,
                "n_high_mass_bins": int(np.sum(bin_mass > self._mass_threshold)),
            },
        )

    def __repr__(self) -> str:
        return (
            f"MassAdaptiveGrid(threshold={self._mass_threshold}, "
            f"min_density={self._min_density})"
        )


# ---------------------------------------------------------------------------
# CurvatureAdaptiveGrid
# ---------------------------------------------------------------------------


class CurvatureAdaptiveGrid:
    """Concentrate grid points where mechanism probability changes rapidly.

    Uses a finite-difference estimate of the second derivative of the
    mechanism's max-probability curve to allocate more points in
    high-curvature regions.

    Parameters
    ----------
    curvature_weight : float
        Exponent controlling how strongly curvature attracts grid points.
        Higher values concentrate more aggressively.
    min_density : float
        Minimum fraction of points for uniform background coverage.
    padding : float or None
        Absolute padding.
    """

    def __init__(
        self,
        curvature_weight: float = 0.5,
        min_density: float = 0.15,
        padding: Optional[float] = None,
    ) -> None:
        if curvature_weight <= 0:
            raise ValueError(f"curvature_weight must be > 0, got {curvature_weight}")
        if not (0.0 < min_density < 1.0):
            raise ValueError(f"min_density must be in (0, 1), got {min_density}")
        self._curvature_weight = curvature_weight
        self._min_density = min_density
        self._padding = padding

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        *,
        mechanism: Optional[npt.NDArray[np.float64]] = None,
        old_grid: Optional[npt.NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> GridResult:
        """Build a curvature-adaptive grid.

        Parameters
        ----------
        f_range : (f_min, f_max)
            Raw query output range.
        k : int
            Target number of grid points.
        mechanism : array of shape (n, k_old), optional
            Existing mechanism table.  Falls back to uniform if absent.
        old_grid : array of shape (k_old,), optional
            Grid points for ``mechanism``.

        Returns
        -------
        GridResult with curvature-adaptive spacing.
        """
        if k < _MIN_K:
            raise ConfigurationError(
                f"k must be >= {_MIN_K}, got {k}",
                parameter="k",
                value=k,
                constraint=f">= {_MIN_K}",
            )
        if mechanism is None or old_grid is None:
            logger.debug("CurvatureAdaptiveGrid: no mechanism, falling back to uniform")
            return UniformGrid(padding=self._padding).build(f_range, k)

        mechanism = np.asarray(mechanism, dtype=np.float64)
        old_grid = np.asarray(old_grid, dtype=np.float64)
        k_old = len(old_grid)

        if mechanism.shape[1] != k_old:
            raise ValueError(
                f"mechanism columns ({mechanism.shape[1]}) != old_grid length ({k_old})"
            )

        # Max probability curve
        p_max = np.max(mechanism, axis=0)  # shape (k_old,)

        # Finite-difference second derivative estimate
        if k_old < 3:
            # Not enough points for curvature — fall back
            return UniformGrid(padding=self._padding).build(f_range, k)

        gaps = np.diff(old_grid)
        gaps = np.maximum(gaps, 1e-15)  # avoid division by zero
        dp = np.diff(p_max)
        first_deriv = dp / gaps
        d2p = np.diff(first_deriv) / ((gaps[:-1] + gaps[1:]) / 2.0)
        curvature = np.abs(d2p)

        # Pad curvature to match k_old length (boundary values = neighbour)
        curv_full = np.empty(k_old, dtype=np.float64)
        curv_full[0] = curvature[0] if len(curvature) > 0 else 0.0
        curv_full[-1] = curvature[-1] if len(curvature) > 0 else 0.0
        if k_old > 2:
            curv_full[1:-1] = curvature

        # Build density: uniform baseline + curvature contribution
        curv_normed = curv_full / (curv_full.sum() + 1e-30)
        density = self._min_density / k_old + (1.0 - self._min_density) * curv_normed
        density = density ** self._curvature_weight
        density = density / density.sum()

        # Padded range
        f_min, f_max = f_range
        if self._padding is not None:
            pad = self._padding
        else:
            span = f_max - f_min
            if span == 0:
                span = max(2.0, 2.0 * abs(f_min))
            pad = 0.5 * span / max(k - 1, 1)
        lo, hi = _pad_range(f_min, f_max, pad)

        # Build CDF and invert to place k points
        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]
        quantiles = np.linspace(0.0, 1.0, k)
        points = np.interp(quantiles, cdf, old_grid)
        points = np.clip(points, lo, hi)

        # Ensure monotonicity after clipping
        points = np.sort(points)
        # Remove near-duplicates
        if len(points) > 1:
            min_gap = (hi - lo) / (10.0 * k)
            unique = [points[0]]
            for pt in points[1:]:
                if pt - unique[-1] > min_gap:
                    unique.append(pt)
            points = np.array(unique, dtype=np.float64)

        # Pad to k if deduplication removed too many
        if len(points) < k:
            extra = np.linspace(lo, hi, k - len(points) + 2)[1:-1]
            points = np.sort(np.concatenate([points, extra]))
        if len(points) > k:
            # Subsample uniformly from the candidate set
            idx = np.round(np.linspace(0, len(points) - 1, k)).astype(int)
            points = points[idx]
        if len(points) < _MIN_K:
            points = np.linspace(lo, hi, _MIN_K)

        widths = _compute_midpoint_widths(points)
        return GridResult(
            points=points,
            widths=widths,
            metadata={
                "strategy": "curvature_adaptive",
                "max_curvature": float(curvature.max()) if len(curvature) > 0 else 0.0,
                "mean_curvature": float(curvature.mean()) if len(curvature) > 0 else 0.0,
            },
        )

    def __repr__(self) -> str:
        return (
            f"CurvatureAdaptiveGrid(weight={self._curvature_weight}, "
            f"min_density={self._min_density})"
        )


# ---------------------------------------------------------------------------
# TailPrunedGrid
# ---------------------------------------------------------------------------


class TailPrunedGrid:
    """Remove far-tail grid points with negligible mechanism mass.

    Starting from an existing grid, this strategy prunes bins in both
    tails where the total mechanism mass (summed over all databases) is
    below a threshold.  The result is a smaller grid that excludes
    negligible-mass regions, reducing LP size.

    Parameters
    ----------
    tail_threshold : float
        Cumulative tail mass below which bins are pruned.
    min_points : int
        Minimum number of points to retain (never prune below this).
    """

    def __init__(
        self,
        tail_threshold: float = 1e-6,
        min_points: int = 10,
    ) -> None:
        if tail_threshold <= 0:
            raise ValueError(f"tail_threshold must be > 0, got {tail_threshold}")
        if min_points < _MIN_K:
            raise ValueError(f"min_points must be >= {_MIN_K}, got {min_points}")
        self._tail_threshold = tail_threshold
        self._min_points = min_points

    def build(
        self,
        f_range: Tuple[float, float],
        k: int,
        *,
        mechanism: Optional[npt.NDArray[np.float64]] = None,
        old_grid: Optional[npt.NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> GridResult:
        """Build a tail-pruned grid.

        Parameters
        ----------
        f_range : (f_min, f_max)
            Raw query output range (used as fallback only).
        k : int
            Target number of points (may return fewer after pruning).
        mechanism : array of shape (n, k_old)
            Mechanism table from which to identify negligible tails.
            Falls back to uniform if ``None``.
        old_grid : array of shape (k_old,)
            Grid points corresponding to ``mechanism``.

        Returns
        -------
        GridResult with tail-pruned points.
        """
        if mechanism is None or old_grid is None:
            logger.debug("TailPrunedGrid: no mechanism, falling back to uniform")
            return UniformGrid().build(f_range, k)

        mechanism = np.asarray(mechanism, dtype=np.float64)
        old_grid = np.asarray(old_grid, dtype=np.float64)
        k_old = len(old_grid)

        if mechanism.shape[1] != k_old:
            raise ValueError(
                f"mechanism columns ({mechanism.shape[1]}) != old_grid length ({k_old})"
            )

        # Total mass per bin across all databases
        total_mass = np.sum(mechanism, axis=0)  # shape (k_old,)
        total_mass = total_mass / (total_mass.sum() + 1e-30)

        # Find left trim index: cumulative mass from left
        cum_left = np.cumsum(total_mass)
        left_idx = 0
        for i in range(k_old):
            if cum_left[i] >= self._tail_threshold:
                left_idx = i
                break

        # Find right trim index: cumulative mass from right
        cum_right = np.cumsum(total_mass[::-1])
        right_idx = k_old - 1
        for i in range(k_old):
            if cum_right[i] >= self._tail_threshold:
                right_idx = k_old - 1 - i
                break

        # Ensure minimum points
        if right_idx - left_idx + 1 < self._min_points:
            center = (left_idx + right_idx) // 2
            half = self._min_points // 2
            left_idx = max(0, center - half)
            right_idx = min(k_old - 1, left_idx + self._min_points - 1)
            if right_idx >= k_old:
                right_idx = k_old - 1
                left_idx = max(0, right_idx - self._min_points + 1)

        pruned_grid = old_grid[left_idx:right_idx + 1]
        n_pruned = k_old - len(pruned_grid)

        # If target k differs from pruned length, resample
        if len(pruned_grid) < _MIN_K:
            pruned_grid = old_grid  # keep all if pruning is too aggressive
            n_pruned = 0

        if len(pruned_grid) != k and k >= _MIN_K:
            new_grid = np.linspace(pruned_grid[0], pruned_grid[-1], k)
        else:
            new_grid = pruned_grid

        widths = _compute_midpoint_widths(new_grid)
        return GridResult(
            points=new_grid,
            widths=widths,
            metadata={
                "strategy": "tail_pruned",
                "n_pruned": n_pruned,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "original_k": k_old,
            },
        )

    def __repr__(self) -> str:
        return (
            f"TailPrunedGrid(threshold={self._tail_threshold}, "
            f"min_points={self._min_points})"
        )
