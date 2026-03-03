"""
Mechanism interpolation between grids for DP-Forge.

When the adaptive refiner moves from a coarse grid ``G1`` to a finer grid
``G2``, the mechanism probability table ``p`` must be transferred.  This
module provides three interpolation strategies with increasing fidelity:

- :class:`PiecewiseConstantInterpolator` — nearest-neighbour; simplest
  but introduces step artefacts.
- :class:`PiecewiseLinearInterpolator` — linear between grid points;
  preserves non-negativity.
- :class:`SplineInterpolator` — monotone cubic spline with positivity
  clamp; smoothest but most expensive.

All interpolators satisfy two critical invariants:

1. **Normalization**: each row of the transferred table sums to 1.
2. **Non-negativity**: all entries are ≥ 0.

The abstract :class:`MechanismInterpolator` protocol defines the
interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from dp_forge.exceptions import ConfigurationError, InvalidMechanismError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MechanismInterpolator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MechanismInterpolator(Protocol):
    """Protocol for mechanism grid-transfer operators.

    Given a mechanism ``p`` on grid ``old_grid``, produce a mechanism
    ``p_new`` on ``new_grid`` that preserves normalization and
    non-negativity.
    """

    def transfer(
        self,
        mechanism: npt.NDArray[np.float64],
        old_grid: npt.NDArray[np.float64],
        new_grid: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Transfer a mechanism from ``old_grid`` to ``new_grid``.

        Parameters
        ----------
        mechanism : array of shape (n, k_old)
            Probability table on the old grid.
        old_grid : array of shape (k_old,)
            Sorted old grid points.
        new_grid : array of shape (k_new,)
            Sorted new grid points.

        Returns
        -------
        p_new : array of shape (n, k_new)
            Transferred mechanism table.  Each row sums to 1 and all
            entries are non-negative.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    mechanism: npt.NDArray[np.float64],
    old_grid: npt.NDArray[np.float64],
    new_grid: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and coerce interpolation inputs."""
    mechanism = np.asarray(mechanism, dtype=np.float64)
    old_grid = np.asarray(old_grid, dtype=np.float64)
    new_grid = np.asarray(new_grid, dtype=np.float64)

    if mechanism.ndim != 2:
        raise InvalidMechanismError(
            f"mechanism must be 2-D, got shape {mechanism.shape}",
            reason="wrong_ndim",
            actual_shape=mechanism.shape,
        )
    if old_grid.ndim != 1 or new_grid.ndim != 1:
        raise ConfigurationError(
            "old_grid and new_grid must be 1-D",
            parameter="grid",
        )
    if mechanism.shape[1] != len(old_grid):
        raise InvalidMechanismError(
            f"mechanism columns ({mechanism.shape[1]}) != old_grid length "
            f"({len(old_grid)})",
            reason="shape_mismatch",
        )
    if len(old_grid) < 2:
        raise ConfigurationError(
            f"old_grid must have >= 2 points, got {len(old_grid)}",
            parameter="old_grid",
        )
    if len(new_grid) < 2:
        raise ConfigurationError(
            f"new_grid must have >= 2 points, got {len(new_grid)}",
            parameter="new_grid",
        )
    return mechanism, old_grid, new_grid


def _normalize_rows(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Clamp to non-negative and renormalize each row to sum to 1."""
    p = np.maximum(p, 0.0)
    row_sums = p.sum(axis=1, keepdims=True)
    # Avoid division by zero for all-zero rows
    row_sums = np.maximum(row_sums, 1e-30)
    return p / row_sums


# ---------------------------------------------------------------------------
# PiecewiseConstantInterpolator
# ---------------------------------------------------------------------------


class PiecewiseConstantInterpolator:
    """Nearest-neighbour interpolation for grid transfer.

    For each new grid point ``y_new[j]``, find the closest old grid
    point ``y_old[j*]`` and assign ``p_new[i][j] = p_old[i][j*]``.
    Rows are then renormalized to sum to 1.

    This is the simplest and fastest interpolator but introduces step
    discontinuities in the transferred mechanism.
    """

    def transfer(
        self,
        mechanism: npt.NDArray[np.float64],
        old_grid: npt.NDArray[np.float64],
        new_grid: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Transfer via nearest-neighbour lookup.

        Parameters
        ----------
        mechanism : array (n, k_old)
            Source mechanism.
        old_grid : array (k_old,)
            Source grid.
        new_grid : array (k_new,)
            Target grid.

        Returns
        -------
        p_new : array (n, k_new)
            Transferred mechanism with normalized rows.
        """
        mechanism, old_grid, new_grid = _validate_inputs(
            mechanism, old_grid, new_grid
        )
        n = mechanism.shape[0]
        k_new = len(new_grid)

        # Find nearest old-grid index for each new-grid point
        # Using searchsorted for efficiency
        indices = np.searchsorted(old_grid, new_grid, side="left")
        indices = np.clip(indices, 0, len(old_grid) - 1)

        # Check if the left or right neighbour is closer
        for j in range(k_new):
            idx = indices[j]
            if idx > 0:
                d_left = abs(new_grid[j] - old_grid[idx - 1])
                d_right = abs(new_grid[j] - old_grid[idx])
                if d_left < d_right:
                    indices[j] = idx - 1

        p_new = mechanism[:, indices]
        return _normalize_rows(p_new)

    def __repr__(self) -> str:
        return "PiecewiseConstantInterpolator()"


# ---------------------------------------------------------------------------
# PiecewiseLinearInterpolator
# ---------------------------------------------------------------------------


class PiecewiseLinearInterpolator:
    """Linear interpolation preserving non-negativity.

    For each new grid point ``y_new[j]``, locate the bracketing interval
    ``[y_old[m], y_old[m+1]]`` and linearly interpolate::

        p_new[i][j] = (1-t) * p_old[i][m] + t * p_old[i][m+1]

    where ``t = (y_new[j] - y_old[m]) / (y_old[m+1] - y_old[m])``.

    Points outside the old grid range are assigned the boundary value.
    Rows are clamped to non-negative and renormalized.
    """

    def transfer(
        self,
        mechanism: npt.NDArray[np.float64],
        old_grid: npt.NDArray[np.float64],
        new_grid: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Transfer via linear interpolation.

        Parameters
        ----------
        mechanism : array (n, k_old)
            Source mechanism.
        old_grid : array (k_old,)
            Source grid.
        new_grid : array (k_new,)
            Target grid.

        Returns
        -------
        p_new : array (n, k_new)
            Transferred mechanism with normalized rows.
        """
        mechanism, old_grid, new_grid = _validate_inputs(
            mechanism, old_grid, new_grid
        )
        n = mechanism.shape[0]
        k_new = len(new_grid)
        p_new = np.empty((n, k_new), dtype=np.float64)

        for i in range(n):
            p_new[i, :] = np.interp(new_grid, old_grid, mechanism[i, :])

        return _normalize_rows(p_new)

    def __repr__(self) -> str:
        return "PiecewiseLinearInterpolator()"


# ---------------------------------------------------------------------------
# SplineInterpolator
# ---------------------------------------------------------------------------


class SplineInterpolator:
    """Monotone cubic spline interpolation with positivity constraints.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) which
    preserves monotonicity of the data — preventing overshoot that could
    yield negative probabilities.

    Any residual negative values are clamped to zero before row
    renormalization.

    Parameters
    ----------
    extrapolate : bool
        Whether to extrapolate beyond the old grid range.
        If ``False`` (default), boundary values are used.
    """

    def __init__(self, extrapolate: bool = False) -> None:
        self._extrapolate = extrapolate

    def transfer(
        self,
        mechanism: npt.NDArray[np.float64],
        old_grid: npt.NDArray[np.float64],
        new_grid: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Transfer via PCHIP interpolation.

        Parameters
        ----------
        mechanism : array (n, k_old)
            Source mechanism.
        old_grid : array (k_old,)
            Source grid (must be strictly increasing).
        new_grid : array (k_new,)
            Target grid.

        Returns
        -------
        p_new : array (n, k_new)
            Transferred mechanism with normalized rows.
        """
        mechanism, old_grid, new_grid = _validate_inputs(
            mechanism, old_grid, new_grid
        )
        n = mechanism.shape[0]
        k_new = len(new_grid)
        p_new = np.empty((n, k_new), dtype=np.float64)

        for i in range(n):
            try:
                interp = PchipInterpolator(
                    old_grid, mechanism[i, :], extrapolate=self._extrapolate
                )
                p_new[i, :] = interp(new_grid)
            except ValueError:
                # PCHIP requires strictly increasing x; fall back to linear
                logger.debug(
                    "SplineInterpolator: PCHIP failed for row %d, "
                    "falling back to linear interpolation",
                    i,
                )
                p_new[i, :] = np.interp(new_grid, old_grid, mechanism[i, :])

        # Handle extrapolation: clamp out-of-range values
        if not self._extrapolate:
            lo_mask = new_grid < old_grid[0]
            hi_mask = new_grid > old_grid[-1]
            if np.any(lo_mask):
                p_new[:, lo_mask] = mechanism[:, 0:1]
            if np.any(hi_mask):
                p_new[:, hi_mask] = mechanism[:, -1:]

        return _normalize_rows(p_new)

    def __repr__(self) -> str:
        return f"SplineInterpolator(extrapolate={self._extrapolate})"
