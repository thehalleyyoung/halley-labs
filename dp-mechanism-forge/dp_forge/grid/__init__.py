"""
Adaptive grid refinement for DP-Forge mechanism synthesis.

This subpackage implements hierarchical grid refinement for discrete DP
mechanism synthesis.  Starting from a coarse uniform grid, the refinement
loop identifies high-mass output bins and subdivides them, re-running
CEGIS on progressively finer grids until the discretisation error is
within a user-specified tolerance.

Modules
-------
- :mod:`adaptive_refine` — Core :class:`AdaptiveGridRefiner` engine.
- :mod:`grid_strategies` — Pluggable grid construction strategies
  (uniform, Chebyshev, mass-adaptive, curvature-adaptive, tail-pruned).
- :mod:`error_estimator` — Discretisation error bounds and convergence
  rate tracking.
- :mod:`interpolation` — Mechanism interpolation between grids
  (piecewise-constant, piecewise-linear, spline).
- :mod:`range_optimizer` — Optimal output range computation for given
  privacy parameters.

Typical usage::

    from dp_forge.grid import AdaptiveGridRefiner, UniformGrid

    refiner = AdaptiveGridRefiner(k0=20, k_max=500)
    result = refiner.refine(spec)
"""

from dp_forge.grid.adaptive_refine import AdaptiveGridRefiner, RefinementStep
from dp_forge.grid.error_estimator import DiscretizationErrorEstimator
from dp_forge.grid.grid_strategies import (
    ChebyshevGrid,
    CurvatureAdaptiveGrid,
    GridStrategy,
    MassAdaptiveGrid,
    TailPrunedGrid,
    UniformGrid,
)
from dp_forge.grid.interpolation import (
    MechanismInterpolator,
    PiecewiseConstantInterpolator,
    PiecewiseLinearInterpolator,
    SplineInterpolator,
)
from dp_forge.grid.range_optimizer import RangeOptimizer

__all__ = [
    "AdaptiveGridRefiner",
    "RefinementStep",
    "DiscretizationErrorEstimator",
    "GridStrategy",
    "UniformGrid",
    "ChebyshevGrid",
    "MassAdaptiveGrid",
    "CurvatureAdaptiveGrid",
    "TailPrunedGrid",
    "MechanismInterpolator",
    "PiecewiseConstantInterpolator",
    "PiecewiseLinearInterpolator",
    "SplineInterpolator",
    "RangeOptimizer",
]
