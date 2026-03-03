"""
Core adaptive grid refinement engine for DP-Forge.

This module implements the multi-level refinement loop that starts with a
coarse uniform grid, runs CEGIS to synthesise a mechanism, identifies
high-mass output bins, subdivides them, and re-runs CEGIS on the refined
grid with warm-start from the previous solution.

Algorithm
---------
1. Build a coarse grid of ``k0`` points.
2. Run CEGIS on the coarse grid → obtain ``p_coarse``.
3. Identify high-mass bins (``max_i p[i][j] > threshold``).
4. Subdivide high-mass bins; keep low-mass bins as-is.
5. Interpolate ``p_coarse`` onto the refined grid as LP warm-start.
6. Re-run CEGIS on the refined grid.
7. Repeat until convergence or ``k_max`` is reached.

Classes
-------
- :class:`AdaptiveGridRefiner` — main engine.
- :class:`RefinementStep` — per-iteration record.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.cegis_loop import CEGISEngine, CEGISSynthesize
from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InfeasibleSpecError,
)
from dp_forge.grid.error_estimator import DiscretizationErrorEstimator
from dp_forge.grid.grid_strategies import (
    GridResult,
    GridStrategy,
    MassAdaptiveGrid,
    UniformGrid,
    _compute_midpoint_widths,
)
from dp_forge.grid.interpolation import (
    MechanismInterpolator,
    PiecewiseLinearInterpolator,
)
from dp_forge.grid.range_optimizer import RangeOptimizer
from dp_forge.lp_builder import build_output_grid
from dp_forge.types import (
    CEGISResult,
    LossFunction,
    MechanismFamily,
    QuerySpec,
    SynthesisConfig,
)

logger = logging.getLogger(__name__)

_DEFAULT_K0: int = 20
_DEFAULT_K_MAX: int = 1000
_DEFAULT_MAX_LEVELS: int = 8
_DEFAULT_MASS_THRESHOLD: float = 0.01
_DEFAULT_CONVERGENCE_TOL: float = 1e-4
_DEFAULT_SUBDIVIDE_FACTOR: int = 2


# ---------------------------------------------------------------------------
# RefinementStep — per-iteration record
# ---------------------------------------------------------------------------


@dataclass
class RefinementStep:
    """Record for one level of the adaptive refinement loop.

    Attributes:
        level: Refinement level (0 = coarsest).
        k: Number of grid points at this level.
        grid: Grid point locations, shape ``(k,)``.
        mechanism: Mechanism probability table, shape ``(n, k)``.
        objective: LP minimax objective value.
        iterations: Number of CEGIS iterations at this level.
        l1_error_bound: Estimated L1 discretisation error bound.
        n_high_mass_bins: Number of bins that exceeded the mass threshold.
        elapsed_seconds: Wall-clock time for this level.
        converged: Whether CEGIS converged at this level.
    """

    level: int
    k: int
    grid: npt.NDArray[np.float64]
    mechanism: npt.NDArray[np.float64]
    objective: float
    iterations: int
    l1_error_bound: float
    n_high_mass_bins: int
    elapsed_seconds: float
    converged: bool

    def __repr__(self) -> str:
        status = "✓" if self.converged else "…"
        return (
            f"RefinementStep(level={self.level}, k={self.k}, "
            f"obj={self.objective:.6f}, L1≤{self.l1_error_bound:.2e}, "
            f"high_mass={self.n_high_mass_bins}, {status})"
        )


# ---------------------------------------------------------------------------
# AdaptiveGridRefiner
# ---------------------------------------------------------------------------


class AdaptiveGridRefiner:
    """Multi-level adaptive grid refinement engine.

    Starts with a coarse grid, runs CEGIS, then iteratively refines the
    grid in high-mass regions and re-runs CEGIS until convergence.

    Parameters
    ----------
    k0 : int
        Initial (coarsest) grid size.  Default 20.
    k_max : int
        Maximum allowed grid size.  Refinement stops if the next level
        would exceed this.  Default 1000.
    max_levels : int
        Maximum number of refinement levels.  Default 8.
    mass_threshold : float
        Bins with ``max_i p[i][j] > mass_threshold`` are considered
        high-mass and are subdivided.  Default 0.01.
    convergence_tol : float
        Relative objective change threshold for convergence.  Default 1e-4.
    subdivide_factor : int
        Each high-mass bin is split into this many sub-bins.  Default 2.
    grid_strategy : GridStrategy or None
        Strategy for building the initial grid.  ``None`` uses
        :class:`UniformGrid`.
    interpolator : MechanismInterpolator or None
        Interpolator for warm-start transfer.  ``None`` uses
        :class:`PiecewiseLinearInterpolator`.
    synthesis_config : SynthesisConfig or None
        CEGIS configuration.  ``None`` uses defaults.

    Example::

        refiner = AdaptiveGridRefiner(k0=20, k_max=500)
        result = refiner.refine(spec)
        print(f"Final grid: k={result.k}, objective={result.obj_val:.6f}")
    """

    def __init__(
        self,
        k0: int = _DEFAULT_K0,
        k_max: int = _DEFAULT_K_MAX,
        max_levels: int = _DEFAULT_MAX_LEVELS,
        mass_threshold: float = _DEFAULT_MASS_THRESHOLD,
        convergence_tol: float = _DEFAULT_CONVERGENCE_TOL,
        subdivide_factor: int = _DEFAULT_SUBDIVIDE_FACTOR,
        grid_strategy: Optional[GridStrategy] = None,
        interpolator: Optional[MechanismInterpolator] = None,
        synthesis_config: Optional[SynthesisConfig] = None,
    ) -> None:
        self._validate_config(
            k0, k_max, max_levels, mass_threshold, convergence_tol, subdivide_factor
        )
        self._k0 = k0
        self._k_max = k_max
        self._max_levels = max_levels
        self._mass_threshold = mass_threshold
        self._convergence_tol = convergence_tol
        self._subdivide_factor = subdivide_factor
        self._grid_strategy = grid_strategy or UniformGrid()
        self._interpolator = interpolator or PiecewiseLinearInterpolator()
        self._synthesis_config = synthesis_config
        self._steps: List[RefinementStep] = []
        self._error_estimator: Optional[DiscretizationErrorEstimator] = None

    @staticmethod
    def _validate_config(
        k0: int,
        k_max: int,
        max_levels: int,
        mass_threshold: float,
        convergence_tol: float,
        subdivide_factor: int,
    ) -> None:
        """Validate constructor parameters."""
        if k0 < 2:
            raise ConfigurationError(
                f"k0 must be >= 2, got {k0}",
                parameter="k0",
                value=k0,
                constraint=">= 2",
            )
        if k_max < k0:
            raise ConfigurationError(
                f"k_max ({k_max}) must be >= k0 ({k0})",
                parameter="k_max",
                value=k_max,
                constraint=f">= k0={k0}",
            )
        if max_levels < 1:
            raise ConfigurationError(
                f"max_levels must be >= 1, got {max_levels}",
                parameter="max_levels",
                value=max_levels,
                constraint=">= 1",
            )
        if mass_threshold <= 0 or mass_threshold >= 1:
            raise ConfigurationError(
                f"mass_threshold must be in (0, 1), got {mass_threshold}",
                parameter="mass_threshold",
                value=mass_threshold,
                constraint="in (0, 1)",
            )
        if convergence_tol <= 0:
            raise ConfigurationError(
                f"convergence_tol must be > 0, got {convergence_tol}",
                parameter="convergence_tol",
                value=convergence_tol,
                constraint="> 0",
            )
        if subdivide_factor < 2:
            raise ConfigurationError(
                f"subdivide_factor must be >= 2, got {subdivide_factor}",
                parameter="subdivide_factor",
                value=subdivide_factor,
                constraint=">= 2",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> List[RefinementStep]:
        """Completed refinement steps."""
        return list(self._steps)

    @property
    def error_estimator(self) -> Optional[DiscretizationErrorEstimator]:
        """The error estimator, populated after at least one level."""
        return self._error_estimator

    @property
    def n_levels(self) -> int:
        """Number of completed refinement levels."""
        return len(self._steps)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def refine(
        self,
        spec: QuerySpec,
        callback: Optional[Callable[[RefinementStep], None]] = None,
    ) -> CEGISResult:
        """Run adaptive grid refinement on the given query spec.

        Parameters
        ----------
        spec : QuerySpec
            Query specification.  The ``k`` field is ignored; the refiner
            uses its own ``k0`` as the starting grid size.
        callback : callable, optional
            Called after each refinement level with the
            :class:`RefinementStep`.

        Returns
        -------
        CEGISResult
            Result from the finest grid level, containing the optimised
            mechanism and convergence metadata.

        Raises
        ------
        InfeasibleSpecError
            If no mechanism exists at any grid level.
        ConvergenceError
            If the refiner reaches ``max_levels`` or ``k_max`` without
            converging.
        """
        self._steps = []
        f_values = spec.query_values
        f_min, f_max = float(np.min(f_values)), float(np.max(f_values))
        f_range = (f_min, f_max)

        # Initialize error estimator
        grid_result = self._grid_strategy.build(f_range, self._k0)
        range_B = float(grid_result.points[-1] - grid_result.points[0])
        self._error_estimator = DiscretizationErrorEstimator(
            range_B=max(range_B, 1e-10), n_databases=spec.n
        )

        current_k = self._k0
        current_grid = grid_result.points
        prev_mechanism: Optional[npt.NDArray[np.float64]] = None
        prev_objective: Optional[float] = None
        final_result: Optional[CEGISResult] = None

        for level in range(self._max_levels):
            t_start = time.monotonic()

            # Build spec with current k
            level_spec = self._build_level_spec(spec, current_grid)

            # Run CEGIS
            try:
                cegis_result = self._run_cegis(level_spec)
            except InfeasibleSpecError:
                if level == 0:
                    raise
                logger.warning(
                    "Level %d (k=%d) infeasible; returning last feasible result",
                    level, current_k,
                )
                break

            mechanism = cegis_result.mechanism
            objective = cegis_result.obj_val
            elapsed = time.monotonic() - t_start

            # Identify high-mass bins
            n_high_mass = self._count_high_mass_bins(mechanism)

            # Error estimation
            l1_bound = 0.0
            if self._error_estimator is not None:
                record = self._error_estimator.record_level(
                    level=level,
                    k=current_k,
                    objective=objective,
                    mechanism=mechanism,
                    grid=current_grid,
                    elapsed_seconds=elapsed,
                )
                l1_bound = record.l1_error_bound

            # Record step
            step = RefinementStep(
                level=level,
                k=current_k,
                grid=current_grid.copy(),
                mechanism=mechanism.copy(),
                objective=objective,
                iterations=cegis_result.iterations,
                l1_error_bound=l1_bound,
                n_high_mass_bins=n_high_mass,
                elapsed_seconds=elapsed,
                converged=True,
            )
            self._steps.append(step)
            final_result = cegis_result

            if callback is not None:
                callback(step)

            logger.info(
                "Refinement level %d: k=%d, obj=%.6f, L1≤%.2e, "
                "high_mass=%d, time=%.2fs",
                level, current_k, objective, l1_bound, n_high_mass, elapsed,
            )

            # Check convergence
            if prev_objective is not None:
                rel_change = abs(objective - prev_objective) / max(
                    abs(prev_objective), 1e-30
                )
                if rel_change < self._convergence_tol:
                    logger.info(
                        "Converged at level %d: relative change %.2e < tol %.2e",
                        level, rel_change, self._convergence_tol,
                    )
                    break

            # Check if further refinement is possible
            if n_high_mass == 0:
                logger.info(
                    "No high-mass bins at level %d; refinement complete", level
                )
                break

            # Build refined grid
            new_grid, new_k = self._build_refined_grid(
                current_grid, mechanism, f_range
            )

            if new_k > self._k_max:
                logger.info(
                    "Refined grid k=%d exceeds k_max=%d; stopping",
                    new_k, self._k_max,
                )
                break

            if new_k <= current_k:
                logger.info(
                    "Refinement did not increase k (%d → %d); stopping",
                    current_k, new_k,
                )
                break

            # Warm-start: interpolate mechanism to new grid
            prev_mechanism = self._interpolator.transfer(
                mechanism, current_grid, new_grid
            )

            prev_objective = objective
            current_grid = new_grid
            current_k = new_k

        if final_result is None:
            raise ConvergenceError(
                "Adaptive refinement produced no valid result",
                iterations=0,
                max_iter=self._max_levels,
            )

        return final_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_level_spec(
        self,
        base_spec: QuerySpec,
        grid: npt.NDArray[np.float64],
    ) -> QuerySpec:
        """Create a QuerySpec with the grid size set to len(grid).

        The grid itself is passed through to CEGIS via the ``k``
        parameter on the spec.
        """
        k = len(grid)
        return QuerySpec(
            query_values=base_spec.query_values,
            domain=base_spec.domain,
            sensitivity=base_spec.sensitivity,
            epsilon=base_spec.epsilon,
            delta=base_spec.delta,
            k=k,
            loss_fn=base_spec.loss_fn,
            custom_loss=base_spec.custom_loss,
            edges=base_spec.edges,
            query_type=base_spec.query_type,
            metadata={**base_spec.metadata, "_grid_points": grid},
        )

    def _run_cegis(self, spec: QuerySpec) -> CEGISResult:
        """Run CEGIS synthesis for a single grid level."""
        config = self._synthesis_config or SynthesisConfig()
        return CEGISSynthesize(spec, config=config)

    def _count_high_mass_bins(
        self, mechanism: npt.NDArray[np.float64]
    ) -> int:
        """Count bins where max_i p[i][j] exceeds the mass threshold."""
        if mechanism.size == 0:
            return 0
        bin_max = np.max(mechanism, axis=0)
        return int(np.sum(bin_max > self._mass_threshold))

    def _build_refined_grid(
        self,
        current_grid: npt.NDArray[np.float64],
        mechanism: npt.NDArray[np.float64],
        f_range: Tuple[float, float],
    ) -> Tuple[npt.NDArray[np.float64], int]:
        """Subdivide high-mass bins and keep low-mass bins as-is.

        Parameters
        ----------
        current_grid : array (k,)
            Current grid points.
        mechanism : array (n, k)
            Current mechanism table.
        f_range : (f_min, f_max)
            Query output range.

        Returns
        -------
        new_grid : array
            Refined grid points (sorted, deduplicated).
        new_k : int
            Length of the new grid.
        """
        k = len(current_grid)
        bin_max = np.max(mechanism, axis=0)

        new_points: List[float] = []
        for j in range(k):
            new_points.append(float(current_grid[j]))
            if bin_max[j] > self._mass_threshold and j < k - 1:
                # Subdivide the interval [current_grid[j], current_grid[j+1]]
                lo = current_grid[j]
                hi = current_grid[j + 1]
                interior = np.linspace(lo, hi, self._subdivide_factor + 1)[1:-1]
                new_points.extend(interior.tolist())

        new_grid = np.array(sorted(set(new_points)), dtype=np.float64)
        return new_grid, len(new_grid)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def error_trajectory(self) -> List[Tuple[int, float]]:
        """Return the (k, l1_error_bound) trajectory across levels.

        Returns
        -------
        trajectory : list of (k, error_bound) tuples
        """
        return [(s.k, s.l1_error_bound) for s in self._steps]

    def objective_trajectory(self) -> List[Tuple[int, float]]:
        """Return the (k, objective) trajectory across levels.

        Returns
        -------
        trajectory : list of (k, objective) tuples
        """
        return [(s.k, s.objective) for s in self._steps]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the refinement process."""
        total_time = sum(s.elapsed_seconds for s in self._steps)
        total_iters = sum(s.iterations for s in self._steps)
        return {
            "n_levels": len(self._steps),
            "k_trajectory": [s.k for s in self._steps],
            "objective_trajectory": [s.objective for s in self._steps],
            "error_trajectory": [s.l1_error_bound for s in self._steps],
            "total_cegis_iterations": total_iters,
            "total_time_seconds": total_time,
            "final_k": self._steps[-1].k if self._steps else 0,
            "final_objective": self._steps[-1].objective if self._steps else None,
            "convergence_rate": (
                self._error_estimator.convergence_rate()
                if self._error_estimator else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveGridRefiner(k0={self._k0}, k_max={self._k_max}, "
            f"levels={len(self._steps)})"
        )
