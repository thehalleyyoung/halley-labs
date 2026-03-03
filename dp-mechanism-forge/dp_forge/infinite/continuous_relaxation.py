"""
Continuous mechanism representation from infinite LP solutions.

Converts the finite-grid LP solution into a continuous density over the
output space.  Provides:

- Piecewise-constant density directly from LP probability weights.
- Kernel density estimate (KDE) with Silverman's rule-of-thumb bandwidth.
- CDF and inverse-CDF for sampling.
- Integration utilities for computing expected loss over continuous output.

Classes
-------
- :class:`ContinuousMechanism` — Continuous density/CDF representation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate, interpolate, stats

from dp_forge.types import LossFunction, QuerySpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BANDWIDTH: float = 1e-10
_INTEGRATION_LIMIT: int = 200
_CDF_GRID_SIZE: int = 2000


# ---------------------------------------------------------------------------
# Continuous mechanism
# ---------------------------------------------------------------------------


@dataclass
class ContinuousMechanism:
    """Continuous mechanism density built from a finite-grid LP solution.

    Given grid points ``y_grid`` and probability weights ``weights`` for a
    single database row (i.e., ``p[i, :]`` from the LP), this class builds
    a continuous density representation.

    Parameters
    ----------
    y_grid : array of shape (k,)
        Sorted output grid points.
    weights : array of shape (k,)
        Probability weights at grid points (must sum to ~1).
    bandwidth : float or None
        KDE bandwidth.  If None, uses Silverman's rule.
    db_index : int
        Database index this mechanism row corresponds to.
    """

    y_grid: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    bandwidth: Optional[float] = None
    db_index: int = 0

    # Cached state (not part of __init__)
    _cdf_grid_y: Optional[npt.NDArray[np.float64]] = field(
        default=None, repr=False, init=False,
    )
    _cdf_grid_vals: Optional[npt.NDArray[np.float64]] = field(
        default=None, repr=False, init=False,
    )
    _cdf_interp: Optional[interpolate.interp1d] = field(
        default=None, repr=False, init=False,
    )

    def __post_init__(self) -> None:
        self.y_grid = np.asarray(self.y_grid, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)

        if self.y_grid.ndim != 1:
            raise ValueError(f"y_grid must be 1-D, got shape {self.y_grid.shape}")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {self.weights.shape}")
        if len(self.y_grid) != len(self.weights):
            raise ValueError(
                f"y_grid ({len(self.y_grid)}) and weights ({len(self.weights)}) "
                f"must have the same length"
            )
        k = len(self.y_grid)
        if k < 2:
            raise ValueError(f"Need at least 2 grid points, got {k}")

        # Normalise weights
        w_sum = float(np.sum(self.weights))
        if w_sum <= 0:
            raise ValueError(f"weights must sum to > 0, got {w_sum}")
        self.weights = self.weights / w_sum

        # Clip negative weights from numerical noise
        self.weights = np.maximum(self.weights, 0.0)
        self.weights /= np.sum(self.weights)

        # Sort by grid position
        order = np.argsort(self.y_grid)
        self.y_grid = self.y_grid[order]
        self.weights = self.weights[order]

        # Compute bandwidth via Silverman's rule if not provided
        if self.bandwidth is None:
            self.bandwidth = self._silverman_bandwidth()
        self.bandwidth = max(self.bandwidth, _MIN_BANDWIDTH)

    def _silverman_bandwidth(self) -> float:
        """Silverman's rule-of-thumb bandwidth for KDE.

        h = 0.9 * min(std, IQR/1.34) * n^{-1/5}

        where n is the effective sample size (inverse of max weight).
        """
        # Weighted statistics
        mean = float(np.dot(self.weights, self.y_grid))
        var = float(np.dot(self.weights, (self.y_grid - mean) ** 2))
        std = math.sqrt(max(var, 1e-30))

        # Weighted IQR approximation
        cum_w = np.cumsum(self.weights)
        q25_idx = int(np.searchsorted(cum_w, 0.25))
        q75_idx = int(np.searchsorted(cum_w, 0.75))
        q25_idx = np.clip(q25_idx, 0, len(self.y_grid) - 1)
        q75_idx = np.clip(q75_idx, 0, len(self.y_grid) - 1)
        iqr = float(self.y_grid[q75_idx] - self.y_grid[q25_idx])

        spread = min(std, iqr / 1.34) if iqr > 0 else std

        # Effective sample size
        n_eff = max(1.0 / float(np.max(self.weights)), 2.0)

        h = 0.9 * spread * n_eff ** (-0.2)
        return max(h, _MIN_BANDWIDTH)

    # -- Density evaluation -------------------------------------------------

    def density(self, y: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate the KDE density at point(s) y.

        Parameters
        ----------
        y : float or array
            Point(s) at which to evaluate the density.

        Returns
        -------
        float or array
            Density value(s).
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        h = self.bandwidth

        # Gaussian KDE: f(y) = Σ_j w_j * K((y - y_j) / h) / h
        # K(u) = (1/√(2π)) exp(-u²/2)
        diffs = y_arr[:, np.newaxis] - self.y_grid[np.newaxis, :]  # (m, k)
        u = diffs / h
        kernel_vals = np.exp(-0.5 * u ** 2) / (h * math.sqrt(2.0 * math.pi))
        result = np.dot(kernel_vals, self.weights)

        if np.isscalar(y):
            return float(result[0])
        return result

    def density_piecewise_constant(
        self, y: Union[float, npt.NDArray[np.float64]],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate the piecewise-constant density at point(s) y.

        The density in the bin ``[y_grid[j], y_grid[j+1])`` is
        ``weights[j] / bin_width``.

        Parameters
        ----------
        y : float or array

        Returns
        -------
        float or array
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        k = len(self.y_grid)
        bin_widths = np.diff(self.y_grid)

        # Find which bin each y falls in
        indices = np.searchsorted(self.y_grid, y_arr, side="right") - 1
        indices = np.clip(indices, 0, k - 2)

        # Density = weight / bin_width
        result = self.weights[indices] / np.maximum(bin_widths[indices], 1e-30)

        if np.isscalar(y):
            return float(result[0])
        return result

    # -- CDF ----------------------------------------------------------------

    def cdf(self, y: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate the CDF at point(s) y using the KDE density.

        Parameters
        ----------
        y : float or array

        Returns
        -------
        float or array
            CDF value(s) in [0, 1].
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        h = self.bandwidth

        # CDF of Gaussian KDE:
        # F(y) = Σ_j w_j Φ((y - y_j) / h)
        u = (y_arr[:, np.newaxis] - self.y_grid[np.newaxis, :]) / h
        phi_vals = stats.norm.cdf(u)
        result = np.dot(phi_vals, self.weights)

        if np.isscalar(y):
            return float(result[0])
        return result

    # -- Expected loss ------------------------------------------------------

    def expected_loss(
        self,
        true_value: float,
        loss_fn: LossFunction = LossFunction.L2,
        custom_loss: Optional[Callable[[float, float], float]] = None,
        integration_points: int = _INTEGRATION_LIMIT,
    ) -> float:
        """Compute expected loss E_y[loss(true_value, y)] under this density.

        Parameters
        ----------
        true_value : float
            True query answer.
        loss_fn : LossFunction
            Loss function type.
        custom_loss : callable, optional
            Custom loss function (required when loss_fn is CUSTOM).
        integration_points : int
            Number of quadrature points for numerical integration.

        Returns
        -------
        float
            Expected loss.
        """
        # For KDE, expected loss = Σ_j w_j * E_{N(y_j, h²)}[loss(true, Y)]
        # For L2 loss: E[loss] = Σ_j w_j * [(true - y_j)² + h²]
        h = self.bandwidth

        if loss_fn == LossFunction.L2:
            sq_diffs = (true_value - self.y_grid) ** 2
            # E[(true - Y)²] where Y ~ N(y_j, h²) = (true - y_j)² + h²
            per_component = sq_diffs + h ** 2
            return float(np.dot(self.weights, per_component))

        if loss_fn == LossFunction.L1 or loss_fn == LossFunction.LINF:
            # E[|true - Y|] where Y ~ N(y_j, h²)
            # = σ·√(2/π)·exp(-μ²/(2σ²)) + μ·(1 - 2Φ(-μ/σ))
            # where μ = true - y_j, σ = h
            mu = true_value - self.y_grid
            sigma = h
            z = mu / sigma
            per_component = sigma * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z ** 2) + mu * (
                1.0 - 2.0 * stats.norm.cdf(-z)
            )
            return float(np.dot(self.weights, np.abs(per_component)))

        # Custom loss: numerical integration
        if custom_loss is None and loss_fn == LossFunction.CUSTOM:
            raise ValueError("custom_loss required for CUSTOM loss function")
        loss_callable = custom_loss if custom_loss is not None else loss_fn.fn
        assert loss_callable is not None

        # Numerical quadrature over the support
        lo = float(self.y_grid[0] - 5.0 * h)
        hi = float(self.y_grid[-1] + 5.0 * h)

        def integrand(y: float) -> float:
            return loss_callable(true_value, y) * float(self.density(y))

        result, _ = integrate.quad(integrand, lo, hi, limit=integration_points)
        return float(result)

    # -- Sampling -----------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw samples from the continuous mechanism.

        Uses the mixture-of-Gaussians interpretation of the KDE:
        1. Pick grid point j with probability weights[j].
        2. Add Gaussian noise N(0, bandwidth²).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : numpy Generator, optional
            Random number generator.

        Returns
        -------
        array of shape (n_samples,)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: pick grid points
        component_indices = rng.choice(
            len(self.y_grid), size=n_samples, p=self.weights,
        )
        centers = self.y_grid[component_indices]

        # Step 2: add Gaussian noise
        noise = rng.normal(0.0, self.bandwidth, size=n_samples)
        return centers + noise

    # -- Factory from LP solution ------------------------------------------

    @classmethod
    def from_lp_solution(
        cls,
        mechanism_table: npt.NDArray[np.float64],
        y_grid: npt.NDArray[np.float64],
        bandwidth: Optional[float] = None,
    ) -> List[ContinuousMechanism]:
        """Create one ContinuousMechanism per database row from an LP solution.

        Parameters
        ----------
        mechanism_table : array of shape (n, k)
            Probability table from the LP solution.
        y_grid : array of shape (k,)
            Output discretisation grid.
        bandwidth : float, optional
            KDE bandwidth override (same for all rows).

        Returns
        -------
        list of ContinuousMechanism
            One per database row.
        """
        mechanism_table = np.asarray(mechanism_table, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        if mechanism_table.ndim != 2:
            raise ValueError(
                f"mechanism_table must be 2-D, got shape {mechanism_table.shape}"
            )
        n, k = mechanism_table.shape
        if k != len(y_grid):
            raise ValueError(
                f"mechanism_table has {k} columns but y_grid has {len(y_grid)} points"
            )

        result = []
        for i in range(n):
            mech = cls(
                y_grid=y_grid.copy(),
                weights=mechanism_table[i].copy(),
                bandwidth=bandwidth,
                db_index=i,
            )
            result.append(mech)
        return result

    # -- Comparison ---------------------------------------------------------

    def total_variation(self, other: ContinuousMechanism, n_points: int = 1000) -> float:
        """Approximate total variation distance to another mechanism.

        TV(P, Q) = (1/2) ∫ |p(y) - q(y)| dy

        Uses numerical quadrature on a common grid.

        Parameters
        ----------
        other : ContinuousMechanism
            The other mechanism to compare against.
        n_points : int
            Number of quadrature points.

        Returns
        -------
        float
            Approximate total variation distance.
        """
        lo = min(float(self.y_grid[0]), float(other.y_grid[0])) - 5.0 * max(self.bandwidth, other.bandwidth)
        hi = max(float(self.y_grid[-1]), float(other.y_grid[-1])) + 5.0 * max(self.bandwidth, other.bandwidth)

        y_eval = np.linspace(lo, hi, n_points)
        p_vals = self.density(y_eval)
        q_vals = other.density(y_eval)

        dy = (hi - lo) / (n_points - 1)
        return 0.5 * float(np.sum(np.abs(p_vals - q_vals)) * dy)

    def __repr__(self) -> str:
        return (
            f"ContinuousMechanism(db={self.db_index}, k={len(self.y_grid)}, "
            f"bw={self.bandwidth:.4e})"
        )
