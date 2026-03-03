"""
Discretization error estimation for DP-Forge adaptive grid refinement.

This module estimates the L1 gap between a k-point discrete mechanism and
the continuous optimal, tracks convergence rates across refinement levels,
and provides theoretical error bounds from Theorem T5.

Key results
-----------
- For a uniform grid of k points over range B, the discretisation error is
  ``O(B / k)`` (Theorem T5, part 1).
- For an adaptive grid concentrating points in high-mass regions, the error
  improves to ``O(B² L / k)`` where L is the Lipschitz constant of the
  mechanism probability function (Theorem T5, part 2).
- The Lipschitz constant is estimated from finite differences of the
  mechanism probability table.

Classes
-------
- :class:`DiscretizationErrorEstimator` — main estimator class.
- :class:`ConvergenceRecord` — per-level convergence metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConvergenceRecord
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceRecord:
    """Metrics for one refinement level.

    Attributes:
        level: Refinement level (0 = coarsest).
        k: Number of grid points at this level.
        objective: LP minimax objective at this level.
        l1_error_bound: Upper bound on L1 discretisation error.
        lipschitz_estimate: Estimated Lipschitz constant of the mechanism.
        elapsed_seconds: Wall-clock time for this level (seconds).
    """

    level: int
    k: int
    objective: float
    l1_error_bound: float
    lipschitz_estimate: float
    elapsed_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ConvergenceRecord(level={self.level}, k={self.k}, "
            f"obj={self.objective:.6f}, L1≤{self.l1_error_bound:.2e})"
        )


# ---------------------------------------------------------------------------
# DiscretizationErrorEstimator
# ---------------------------------------------------------------------------


class DiscretizationErrorEstimator:
    """Estimates discretisation error and tracks convergence.

    The estimator maintains a history of :class:`ConvergenceRecord` entries
    across refinement levels, enabling convergence rate computation and
    extrapolation of the error to finer grids.

    Parameters
    ----------
    range_B : float
        Total output range span (``y_max - y_min``).
    n_databases : int
        Number of database inputs ``n``.

    Example::

        est = DiscretizationErrorEstimator(range_B=10.0, n_databases=5)
        est.record_level(level=0, k=20, objective=0.5,
                         mechanism=p, grid=y)
        print(est.estimate_l1_gap(k=100))
    """

    def __init__(self, range_B: float, n_databases: int) -> None:
        if range_B <= 0:
            raise ConfigurationError(
                f"range_B must be > 0, got {range_B}",
                parameter="range_B",
                value=range_B,
                constraint="> 0",
            )
        if n_databases < 1:
            raise ConfigurationError(
                f"n_databases must be >= 1, got {n_databases}",
                parameter="n_databases",
                value=n_databases,
                constraint=">= 1",
            )
        self._B = range_B
        self._n = n_databases
        self._history: List[ConvergenceRecord] = []
        self._lipschitz_cache: Optional[float] = None

    @property
    def history(self) -> List[ConvergenceRecord]:
        """Convergence history across refinement levels."""
        return list(self._history)

    @property
    def n_levels(self) -> int:
        """Number of refinement levels recorded."""
        return len(self._history)

    # ------------------------------------------------------------------
    # Lipschitz estimation
    # ------------------------------------------------------------------

    def estimate_lipschitz(
        self,
        mechanism: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> float:
        """Estimate the Lipschitz constant of the mechanism probability curve.

        The Lipschitz constant is the maximum absolute finite-difference
        slope of the max-probability curve ``max_i p[i][j]`` over the
        grid.

        Parameters
        ----------
        mechanism : array (n, k)
            Mechanism probability table.
        grid : array (k,)
            Grid point locations.

        Returns
        -------
        L : float
            Estimated Lipschitz constant (non-negative).
        """
        mechanism = np.asarray(mechanism, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)

        if mechanism.ndim != 2 or mechanism.shape[1] != len(grid):
            raise ValueError(
                f"mechanism shape {mechanism.shape} incompatible with "
                f"grid length {len(grid)}"
            )

        k = len(grid)
        if k < 2:
            return 0.0

        # Max probability curve
        p_max = np.max(mechanism, axis=0)

        # Finite-difference slopes
        gaps = np.diff(grid)
        gaps = np.maximum(gaps, 1e-15)
        slopes = np.abs(np.diff(p_max)) / gaps

        L = float(np.max(slopes)) if len(slopes) > 0 else 0.0
        self._lipschitz_cache = L
        return L

    # ------------------------------------------------------------------
    # L1 error bounds
    # ------------------------------------------------------------------

    def estimate_l1_gap(
        self,
        k: int,
        *,
        lipschitz: Optional[float] = None,
        adaptive: bool = False,
    ) -> float:
        """Estimate the L1 discretisation error bound for *k* grid points.

        Applies Theorem T5:

        - **Uniform grid**: ``error ≤ B / k``
        - **Adaptive grid**: ``error ≤ B² · L / k``

        where ``B`` is the range span and ``L`` is the Lipschitz constant.

        Parameters
        ----------
        k : int
            Number of grid points.
        lipschitz : float, optional
            Lipschitz constant.  If ``None``, uses the most recent cached
            estimate (from :meth:`estimate_lipschitz`).
        adaptive : bool
            Whether to use the adaptive (tighter) bound.

        Returns
        -------
        error_bound : float
            Upper bound on the L1 gap.
        """
        if k < 1:
            raise ConfigurationError(
                f"k must be >= 1, got {k}",
                parameter="k",
                value=k,
                constraint=">= 1",
            )

        if not adaptive:
            # Theorem T5 part 1: O(B/k)
            return self._B / k

        # Theorem T5 part 2: O(B²L/k)
        L = lipschitz if lipschitz is not None else self._lipschitz_cache
        if L is None:
            logger.warning(
                "No Lipschitz estimate available; using uniform bound"
            )
            return self._B / k
        return self._B ** 2 * L / k

    # ------------------------------------------------------------------
    # Theoretical bounds for specific mechanisms
    # ------------------------------------------------------------------

    def laplace_error_bound(
        self, epsilon: float, k: int
    ) -> float:
        """Theoretical L1 error bound for Laplace mechanism discretisation.

        The Laplace PDF has Lipschitz constant ``ε/2``, giving a uniform-
        grid discretisation error of ``O(B · ε / (2k))``.

        Parameters
        ----------
        epsilon : float
            Privacy parameter ε.
        k : int
            Number of grid points.

        Returns
        -------
        bound : float
        """
        L_laplace = epsilon / 2.0
        return self._B * L_laplace / k

    # ------------------------------------------------------------------
    # Record and convergence rate
    # ------------------------------------------------------------------

    def record_level(
        self,
        level: int,
        k: int,
        objective: float,
        mechanism: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
        elapsed_seconds: float = 0.0,
    ) -> ConvergenceRecord:
        """Record metrics for a completed refinement level.

        Computes the Lipschitz constant and L1 error bound, then appends
        a :class:`ConvergenceRecord` to the internal history.

        Parameters
        ----------
        level : int
            Refinement level index.
        k : int
            Number of grid points.
        objective : float
            LP minimax objective.
        mechanism : array (n, k)
            Mechanism table.
        grid : array (k,)
            Grid points.
        elapsed_seconds : float
            Wall-clock time for this level.

        Returns
        -------
        ConvergenceRecord
        """
        L = self.estimate_lipschitz(mechanism, grid)
        error = self.estimate_l1_gap(k, lipschitz=L, adaptive=True)
        record = ConvergenceRecord(
            level=level,
            k=k,
            objective=objective,
            l1_error_bound=error,
            lipschitz_estimate=L,
            elapsed_seconds=elapsed_seconds,
        )
        self._history.append(record)
        logger.debug(
            "Recorded level %d: k=%d, obj=%.6f, L1≤%.2e, L=%.4f",
            level, k, objective, error, L,
        )
        return record

    def convergence_rate(self) -> Optional[float]:
        """Estimate the empirical convergence rate from recorded levels.

        Fits a power law ``error ∝ k^(-α)`` to the (k, L1_error_bound)
        pairs and returns α.  Returns ``None`` if fewer than 2 levels
        have been recorded.

        Returns
        -------
        alpha : float or None
            Estimated convergence rate exponent, or ``None``.
        """
        if len(self._history) < 2:
            return None

        # Use log-log regression: log(error) = -α * log(k) + C
        log_k = np.array(
            [math.log(r.k) for r in self._history], dtype=np.float64
        )
        log_err = np.array(
            [math.log(max(r.l1_error_bound, 1e-30)) for r in self._history],
            dtype=np.float64,
        )

        # Simple least squares: y = m*x + b → m = -α
        if np.std(log_k) < 1e-15:
            return None
        m, _ = np.polyfit(log_k, log_err, 1)
        alpha = -m
        return float(alpha)

    def extrapolate_error(self, target_k: int) -> Optional[float]:
        """Extrapolate the error bound to a finer grid using the fitted rate.

        Uses the power-law fit from :meth:`convergence_rate` to predict
        the error at ``target_k`` grid points.

        Parameters
        ----------
        target_k : int
            Target number of grid points.

        Returns
        -------
        predicted_error : float or None
            Predicted L1 error bound, or ``None`` if insufficient data.
        """
        alpha = self.convergence_rate()
        if alpha is None or len(self._history) == 0:
            return None
        last = self._history[-1]
        if last.k <= 0 or target_k <= 0:
            return None
        # error ∝ k^(-α) → error_new = error_old * (k_old / k_new)^α
        ratio = last.k / target_k
        return last.l1_error_bound * (ratio ** alpha)

    def is_converged(
        self,
        tol: float = 1e-4,
        min_levels: int = 2,
    ) -> bool:
        """Check if the refinement has converged within tolerance.

        Convergence is declared if the relative change in objective
        between the last two levels is below ``tol``.

        Parameters
        ----------
        tol : float
            Relative objective change threshold.
        min_levels : int
            Minimum number of levels before convergence can be declared.

        Returns
        -------
        converged : bool
        """
        if len(self._history) < min_levels:
            return False
        last = self._history[-1]
        prev = self._history[-2]
        denom = max(abs(prev.objective), 1e-30)
        rel_change = abs(last.objective - prev.objective) / denom
        return rel_change < tol

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the convergence history."""
        return {
            "n_levels": len(self._history),
            "range_B": self._B,
            "n_databases": self._n,
            "convergence_rate": self.convergence_rate(),
            "levels": [
                {
                    "level": r.level,
                    "k": r.k,
                    "objective": r.objective,
                    "l1_error_bound": r.l1_error_bound,
                    "lipschitz": r.lipschitz_estimate,
                    "time_s": r.elapsed_seconds,
                }
                for r in self._history
            ],
        }

    def __repr__(self) -> str:
        rate = self.convergence_rate()
        rate_str = f", α={rate:.2f}" if rate is not None else ""
        return (
            f"DiscretizationErrorEstimator(B={self._B:.2f}, "
            f"levels={len(self._history)}{rate_str})"
        )
