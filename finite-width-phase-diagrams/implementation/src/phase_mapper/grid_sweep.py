"""Coarse grid sweep over hyperparameter space.

Provides:
  - ParameterRange: specification of a single parameter axis
  - GridConfig: collection of parameter ranges for a sweep
  - GridPoint: evaluation result at a single grid coordinate
  - SweepResult: full grid of evaluated points with metadata
  - GridSweeper: parallel grid evaluation with caching and adaptive refinement
"""

from __future__ import annotations

import hashlib
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class ParameterRange:
    """Specification of a single parameter axis for grid sweeping.

    Parameters
    ----------
    min_val : float
        Minimum value of the parameter range.
    max_val : float
        Maximum value of the parameter range.
    num_points : int
        Number of grid points along this axis.
    log_scale : bool
        Whether to space points logarithmically.
    """

    min_val: float
    max_val: float
    num_points: int
    log_scale: bool = False

    def to_grid(self) -> np.ndarray:
        """Generate grid values for this parameter range.

        Returns
        -------
        np.ndarray
            1-D array of grid values, length ``num_points``.
        """
        if self.log_scale:
            if self.min_val <= 0 or self.max_val <= 0:
                raise ValueError(
                    "Log-scale requires strictly positive min_val and max_val, "
                    f"got min_val={self.min_val}, max_val={self.max_val}"
                )
            return np.logspace(
                np.log10(self.min_val),
                np.log10(self.max_val),
                self.num_points,
            )
        return np.linspace(self.min_val, self.max_val, self.num_points)


@dataclass
class GridConfig:
    """Configuration for a hyperparameter grid sweep.

    Parameters
    ----------
    learning_rate : ParameterRange or None
        Range for the learning rate axis.
    width : ParameterRange or None
        Range for the network width axis.
    init_scale : ParameterRange or None
        Range for the initialization scale axis.
    depth : ParameterRange or None
        Range for the network depth axis.
    extra_ranges : dict
        Additional named parameter ranges.
    """

    learning_rate: Optional[ParameterRange] = None
    width: Optional[ParameterRange] = None
    init_scale: Optional[ParameterRange] = None
    depth: Optional[ParameterRange] = None
    extra_ranges: Dict[str, ParameterRange] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def active_ranges(self) -> Dict[str, ParameterRange]:
        """Return dict of non-None parameter ranges.

        Returns
        -------
        Dict[str, ParameterRange]
            Mapping from parameter name to its range specification.
        """
        ranges: Dict[str, ParameterRange] = {}
        for name in ("learning_rate", "width", "init_scale", "depth"):
            val = getattr(self, name)
            if val is not None:
                ranges[name] = val
        ranges.update(self.extra_ranges)
        return ranges

    def total_points(self) -> int:
        """Total number of grid points across all active axes.

        Returns
        -------
        int
        """
        active = self.active_ranges()
        if not active:
            return 0
        total = 1
        for pr in active.values():
            total *= pr.num_points
        return total


@dataclass
class GridPoint:
    """Result at a single grid coordinate.

    Parameters
    ----------
    coordinates : dict
        Mapping from parameter name to value at this grid point.
    order_parameter_value : float
        Evaluated order parameter at this coordinate.
    regime_label : str
        Assigned regime label (e.g., ``"lazy"``, ``"rich"``).
    metadata : dict
        Arbitrary additional data.
    """

    coordinates: Dict[str, float] = field(default_factory=dict)
    order_parameter_value: float = 0.0
    regime_label: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepResult:
    """Full grid of evaluated points with metadata.

    Parameters
    ----------
    grid_points : list of GridPoint
        Flat list of all evaluated grid points.
    parameter_names : list of str
        Ordered names of the swept parameters.
    grid_shape : tuple of int
        Shape of the N-D grid (one entry per parameter axis).
    elapsed_seconds : float
        Wall-clock time for the sweep.
    metadata : dict
        Extra information (e.g., worker count, config hash).
    """

    grid_points: List[GridPoint] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    grid_shape: Tuple[int, ...] = ()
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return order parameter values as an N-D array matching grid_shape.

        Returns
        -------
        np.ndarray
            Array of shape ``grid_shape`` with order parameter values.
        """
        values = np.array(
            [gp.order_parameter_value for gp in self.grid_points],
            dtype=np.float64,
        )
        return values.reshape(self.grid_shape)

    def coordinates_array(self) -> np.ndarray:
        """Return coordinates as a 2-D array (num_points, num_params).

        Returns
        -------
        np.ndarray
            Shape ``(total_points, num_params)``.
        """
        n_params = len(self.parameter_names)
        coords = np.empty((len(self.grid_points), n_params), dtype=np.float64)
        for i, gp in enumerate(self.grid_points):
            for j, name in enumerate(self.parameter_names):
                coords[i, j] = gp.coordinates[name]
        return coords

    def labels(self) -> List[str]:
        """Return regime labels in grid order.

        Returns
        -------
        List[str]
        """
        return [gp.regime_label for gp in self.grid_points]


# ======================================================================
# Grid sweeper
# ======================================================================


def _coords_hash(coords: Dict[str, float]) -> str:
    """Deterministic hash for a coordinate dictionary."""
    key = "|".join(f"{k}={v:.15e}" for k, v in sorted(coords.items()))
    return hashlib.sha256(key.encode()).hexdigest()


class GridSweeper:
    """Evaluate an order parameter over a hyperparameter grid.

    Parameters
    ----------
    config : GridConfig
        Specification of parameter ranges.
    order_param_fn : callable
        ``(coords: dict) -> float``.  Evaluates the order parameter at a
        single hyperparameter coordinate.
    n_workers : int
        Number of parallel workers (1 = sequential).
    progress_callback : callable or None
        ``(completed: int, total: int) -> None``.  Called after each point.
    """

    def __init__(
        self,
        config: GridConfig,
        order_param_fn: Callable[[Dict[str, float]], float],
        n_workers: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.config = config
        self.order_param_fn = order_param_fn
        self.n_workers = max(1, n_workers)
        self.progress_callback = progress_callback

        self._cache: Dict[str, GridPoint] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sweep(self) -> SweepResult:
        """Evaluate the order parameter at every grid point.

        Returns
        -------
        SweepResult
            Grid of evaluated points with timing information.
        """
        active = self.config.active_ranges()
        if not active:
            return SweepResult()

        param_names = list(active.keys())
        grids = [active[name].to_grid() for name in param_names]
        grid_shape = tuple(len(g) for g in grids)
        all_coords = list(itertools.product(*grids))
        total = len(all_coords)

        t0 = time.monotonic()
        grid_points = self._evaluate_all(param_names, all_coords, total)
        elapsed = time.monotonic() - t0

        return SweepResult(
            grid_points=grid_points,
            parameter_names=param_names,
            grid_shape=grid_shape,
            elapsed_seconds=elapsed,
            metadata={"n_workers": self.n_workers},
        )

    def adaptive_refine(
        self,
        result: SweepResult,
        threshold: float = 0.5,
    ) -> SweepResult:
        """Refine the grid near detected phase boundaries.

        Adds extra points where the gradient of the order parameter exceeds
        ``threshold`` times its maximum value.

        Parameters
        ----------
        result : SweepResult
            Previous sweep result.
        threshold : float
            Fraction of max gradient above which to refine (0–1).

        Returns
        -------
        SweepResult
            New sweep with additional points near boundaries.
        """
        values = result.to_array()
        grad_mag = self._gradient_magnitude(values)
        max_grad = np.max(grad_mag)
        if max_grad == 0:
            return result

        active = self.config.active_ranges()
        param_names = result.parameter_names
        grids = [active[name].to_grid() for name in param_names]

        # Identify cells exceeding threshold
        mask = grad_mag > threshold * max_grad
        refine_indices = list(zip(*np.where(mask)))

        new_coords_list: List[Tuple[float, ...]] = []
        for idx in refine_indices:
            midpoints = []
            for dim, (i, grid) in enumerate(zip(idx, grids)):
                lo = grid[max(i - 1, 0)]
                hi = grid[min(i + 1, len(grid) - 1)]
                mid = 0.5 * (lo + hi)
                midpoints.append(mid)
            new_coords_list.append(tuple(midpoints))

        if not new_coords_list:
            return result

        t0 = time.monotonic()
        new_points = self._evaluate_all(
            param_names, new_coords_list, len(new_coords_list)
        )
        elapsed = time.monotonic() - t0

        all_points = list(result.grid_points) + new_points
        return SweepResult(
            grid_points=all_points,
            parameter_names=param_names,
            grid_shape=result.grid_shape,
            elapsed_seconds=result.elapsed_seconds + elapsed,
            metadata={**result.metadata, "refined_points": len(new_points)},
        )

    def interpolate(self, result: SweepResult, coords: Dict[str, float]) -> float:
        """Interpolate the order parameter at an arbitrary point.

        Parameters
        ----------
        result : SweepResult
            Previous sweep result (must have regular grid shape).
        coords : dict
            Coordinates at which to interpolate.

        Returns
        -------
        float
            Interpolated order parameter value.
        """
        active = self.config.active_ranges()
        param_names = result.parameter_names
        grids = [active[name].to_grid() for name in param_names]

        values = result.to_array()
        interp = RegularGridInterpolator(
            tuple(grids), values, method="linear", bounds_error=False,
            fill_value=None,
        )
        point = np.array([coords[name] for name in param_names])
        return float(interp(point.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_point(self, coords: Dict[str, float]) -> GridPoint:
        """Evaluate the order parameter at a single coordinate.

        Parameters
        ----------
        coords : dict
            Parameter name -> value mapping.

        Returns
        -------
        GridPoint
        """
        h = _coords_hash(coords)
        if h in self._cache:
            return self._cache[h]

        value = float(self.order_param_fn(coords))
        gp = GridPoint(
            coordinates=dict(coords),
            order_parameter_value=value,
        )
        self._cache[h] = gp
        return gp

    def _evaluate_all(
        self,
        param_names: List[str],
        all_coords: List[Tuple[float, ...]],
        total: int,
    ) -> List[GridPoint]:
        """Evaluate all coordinate tuples, optionally in parallel.

        Parameters
        ----------
        param_names : list of str
            Ordered parameter names.
        all_coords : list of tuple
            Coordinate tuples to evaluate.
        total : int
            Total number of points (for progress tracking).

        Returns
        -------
        List[GridPoint]
        """
        coord_dicts = [
            {name: val for name, val in zip(param_names, vals)}
            for vals in all_coords
        ]

        if self.n_workers <= 1:
            return self._evaluate_sequential(coord_dicts, total)
        return self._evaluate_parallel(coord_dicts, total)

    def _evaluate_sequential(
        self,
        coord_dicts: List[Dict[str, float]],
        total: int,
    ) -> List[GridPoint]:
        """Evaluate coordinate dicts sequentially.

        Parameters
        ----------
        coord_dicts : list of dict
        total : int

        Returns
        -------
        List[GridPoint]
        """
        results: List[GridPoint] = []
        for i, cd in enumerate(coord_dicts):
            results.append(self._evaluate_point(cd))
            if self.progress_callback is not None:
                self.progress_callback(i + 1, total)
        return results

    def _evaluate_parallel(
        self,
        coord_dicts: List[Dict[str, float]],
        total: int,
    ) -> List[GridPoint]:
        """Evaluate coordinate dicts in parallel using ProcessPoolExecutor.

        Parameters
        ----------
        coord_dicts : list of dict
        total : int

        Returns
        -------
        List[GridPoint]
        """
        # We cannot pickle self, so use the raw function
        fn = self.order_param_fn
        index_map: Dict[str, int] = {}
        results: List[Optional[GridPoint]] = [None] * len(coord_dicts)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for i, cd in enumerate(coord_dicts):
                h = _coords_hash(cd)
                if h in self._cache:
                    results[i] = self._cache[h]
                    completed += 1
                    continue
                fut = executor.submit(_evaluate_point_worker, fn, cd)
                futures[fut] = (i, cd, h)

            for fut in as_completed(futures):
                i, cd, h = futures[fut]
                value = fut.result()
                gp = GridPoint(
                    coordinates=dict(cd),
                    order_parameter_value=value,
                )
                self._cache[h] = gp
                results[i] = gp
                completed += 1
                if self.progress_callback is not None:
                    self.progress_callback(completed, total)

        return [r for r in results if r is not None]  # type: ignore[misc]

    @staticmethod
    def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude on a regular grid.

        Parameters
        ----------
        values : np.ndarray
            N-D array of order parameter values.

        Returns
        -------
        np.ndarray
            Gradient magnitude at each interior grid point.
        """
        grads = np.gradient(values)
        if isinstance(grads, np.ndarray):
            grads = [grads]
        mag = np.zeros_like(values)
        for g in grads:
            mag += g ** 2
        return np.sqrt(mag)


# ======================================================================
# Module-level helper for pickling
# ======================================================================


def _evaluate_point_worker(
    fn: Callable[[Dict[str, float]], float],
    coords: Dict[str, float],
) -> float:
    """Worker function for parallel grid evaluation.

    Parameters
    ----------
    fn : callable
        Order parameter function.
    coords : dict
        Coordinates at which to evaluate.

    Returns
    -------
    float
    """
    return float(fn(coords))
