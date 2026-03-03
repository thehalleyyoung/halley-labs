"""
Convergence tracking for the cutting-plane infinite LP solver.

Monitors upper/lower bounds across iterations, estimates convergence rates,
and provides early termination criteria based on duality gap, improvement
rate, and iteration limits.

Theory
------
For the cutting-plane method on the infinite LP, the convergence rate is
O(B / k_t) where B is a problem-dependent constant (related to the Lipschitz
constant of the reduced cost) and k_t is the grid size at iteration t.  This
module fits the empirical convergence curve to the model gap(t) ~ C / t^α
to estimate the rate exponent α and predict remaining iterations.

Classes
-------
- :class:`ConvergenceMonitor` — Track bounds and decide termination.
- :class:`ConvergenceSnapshot` — Immutable snapshot of convergence state.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convergence snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceSnapshot:
    """Immutable snapshot of convergence state at a single iteration.

    Attributes:
        iteration: Zero-based iteration index.
        upper_bound: LP objective on current finite grid (primal feasible).
        lower_bound: Dual bound (valid lower bound on infinite-LP optimum).
        gap: Absolute duality gap ``upper_bound - lower_bound``.
        relative_gap: Gap normalised by ``max(|upper_bound|, 1)``.
        grid_size: Number of output points in the current grid.
        violation: Magnitude of the most-violated reduced cost.
        elapsed: Wall-clock time since solver start, in seconds.
    """

    iteration: int
    upper_bound: float
    lower_bound: float
    gap: float
    relative_gap: float
    grid_size: int
    violation: float
    elapsed: float


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------


class ConvergenceMonitor:
    """Track upper/lower bounds and decide when to terminate the cutting-plane loop.

    The monitor maintains a history of :class:`ConvergenceSnapshot` objects
    and provides methods for convergence rate estimation, gap certification,
    and early termination.

    Parameters
    ----------
    target_tol : float
        Target absolute duality gap for convergence.
    max_iter : int
        Maximum number of cutting-plane iterations.
    min_improvement : float
        Minimum relative gap improvement per iteration before stalling
        detection triggers.  Default ``1e-10``.
    stall_window : int
        Number of consecutive stalling iterations before early termination.
        Default ``5``.
    time_limit : float or None
        Wall-clock time limit in seconds.  ``None`` for no limit.
    """

    def __init__(
        self,
        target_tol: float,
        max_iter: int = 500,
        min_improvement: float = 1e-10,
        stall_window: int = 5,
        time_limit: Optional[float] = None,
    ) -> None:
        if target_tol <= 0:
            raise ValueError(f"target_tol must be > 0, got {target_tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if stall_window < 1:
            raise ValueError(f"stall_window must be >= 1, got {stall_window}")

        self._target_tol = target_tol
        self._max_iter = max_iter
        self._min_improvement = min_improvement
        self._stall_window = stall_window
        self._time_limit = time_limit

        self._history: List[ConvergenceSnapshot] = []
        self._start_time: float = time.monotonic()
        self._best_upper: float = math.inf
        self._best_lower: float = -math.inf
        self._stall_count: int = 0
        self._termination_reason: Optional[str] = None

    # -- Properties ---------------------------------------------------------

    @property
    def history(self) -> List[ConvergenceSnapshot]:
        """Full convergence history."""
        return list(self._history)

    @property
    def n_iterations(self) -> int:
        """Number of iterations recorded."""
        return len(self._history)

    @property
    def current_gap(self) -> float:
        """Current absolute duality gap."""
        return self._best_upper - self._best_lower

    @property
    def current_relative_gap(self) -> float:
        """Current relative duality gap."""
        denom = max(abs(self._best_upper), 1.0)
        return self.current_gap / denom

    @property
    def best_upper(self) -> float:
        """Best (lowest) upper bound seen."""
        return self._best_upper

    @property
    def best_lower(self) -> float:
        """Best (highest) lower bound seen."""
        return self._best_lower

    @property
    def target_tol(self) -> float:
        """Target tolerance for convergence."""
        return self._target_tol

    @property
    def termination_reason(self) -> Optional[str]:
        """Reason for termination, or None if not yet terminated."""
        return self._termination_reason

    # -- Core API -----------------------------------------------------------

    def update(
        self,
        upper_bound: float,
        lower_bound: float,
        grid_size: int,
        violation: float,
    ) -> ConvergenceSnapshot:
        """Record a new iteration and update bounds.

        Parameters
        ----------
        upper_bound : float
            LP objective on the current finite grid.
        lower_bound : float
            Dual bound from the current LP solution.
        grid_size : int
            Number of grid points after this iteration's enrichment.
        violation : float
            Magnitude of the most-violated reduced cost found by the oracle.

        Returns
        -------
        ConvergenceSnapshot
            The snapshot for this iteration.
        """
        prev_gap = self.current_gap

        self._best_upper = min(self._best_upper, upper_bound)
        self._best_lower = max(self._best_lower, lower_bound)

        gap = self._best_upper - self._best_lower
        # Clamp to zero if numerical noise makes gap slightly negative
        gap = max(gap, 0.0)

        rel_gap = gap / max(abs(self._best_upper), 1.0)
        elapsed = time.monotonic() - self._start_time

        snapshot = ConvergenceSnapshot(
            iteration=len(self._history),
            upper_bound=self._best_upper,
            lower_bound=self._best_lower,
            gap=gap,
            relative_gap=rel_gap,
            grid_size=grid_size,
            violation=violation,
            elapsed=elapsed,
        )
        self._history.append(snapshot)

        # Stall detection
        if prev_gap < math.inf:
            improvement = (prev_gap - gap) / max(prev_gap, 1e-30)
            if improvement < self._min_improvement:
                self._stall_count += 1
            else:
                self._stall_count = 0
        else:
            self._stall_count = 0

        logger.debug(
            "Iteration %d: gap=%.3e, rel_gap=%.3e, grid=%d, violation=%.3e",
            snapshot.iteration,
            gap,
            rel_gap,
            grid_size,
            violation,
        )

        return snapshot

    def should_terminate(self) -> bool:
        """Check whether the cutting-plane loop should stop.

        Returns
        -------
        bool
            ``True`` if any termination criterion is met.

        Side Effects
        ------------
        Sets :attr:`termination_reason` to a human-readable string.
        """
        if not self._history:
            return False

        # 1. Gap below target tolerance
        if self.current_gap <= self._target_tol:
            self._termination_reason = (
                f"gap {self.current_gap:.3e} <= target_tol {self._target_tol:.3e}"
            )
            return True

        # 2. Max iterations reached
        if len(self._history) >= self._max_iter:
            self._termination_reason = (
                f"max_iter {self._max_iter} reached"
            )
            return True

        # 3. Stalling
        if self._stall_count >= self._stall_window:
            self._termination_reason = (
                f"stalled for {self._stall_count} iterations "
                f"(min_improvement={self._min_improvement:.1e})"
            )
            return True

        # 4. Time limit
        if self._time_limit is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._time_limit:
                self._termination_reason = (
                    f"time_limit {self._time_limit:.1f}s exceeded "
                    f"(elapsed={elapsed:.1f}s)"
                )
                return True

        # 5. Violation below tolerance (oracle found nothing significant)
        last = self._history[-1]
        if last.violation <= self._target_tol * 0.01:
            self._termination_reason = (
                f"violation {last.violation:.3e} negligible "
                f"relative to target_tol {self._target_tol:.3e}"
            )
            return True

        return False

    def convergence_rate(self) -> Tuple[float, float]:
        """Estimate convergence rate by fitting gap(t) ~ C / t^α.

        Fits a log-linear model: log(gap) = log(C) - α·log(t) using
        least-squares on the convergence history.

        Returns
        -------
        alpha : float
            Estimated rate exponent.  α ≈ 1 is expected for cutting-plane.
        C : float
            Estimated constant factor.

        Raises
        ------
        ValueError
            If fewer than 3 data points are available.
        """
        gaps = [s.gap for s in self._history if s.gap > 0]
        if len(gaps) < 3:
            raise ValueError(
                f"Need >= 3 data points with positive gap for rate estimation, "
                f"got {len(gaps)}"
            )

        # Use 1-based iteration indices
        t_vals = np.arange(1, len(gaps) + 1, dtype=np.float64)
        log_t = np.log(t_vals)
        log_gap = np.log(np.array(gaps, dtype=np.float64))

        # Least-squares: log_gap = log_C - alpha * log_t
        A = np.column_stack([np.ones_like(log_t), -log_t])
        result, _, _, _ = np.linalg.lstsq(A, log_gap, rcond=None)
        log_C, alpha = result[0], result[1]
        C = math.exp(log_C)

        return float(alpha), float(C)

    def predict_iterations_remaining(self) -> Optional[int]:
        """Predict how many more iterations are needed to reach target_tol.

        Uses the fitted convergence rate model.  Returns ``None`` if the
        rate cannot be estimated or the prediction is unreliable.

        Returns
        -------
        int or None
            Estimated iterations remaining, or None.
        """
        try:
            alpha, C = self.convergence_rate()
        except ValueError:
            return None

        if alpha <= 0 or C <= 0:
            return None

        # gap(t) = C / t^alpha => t = (C / gap_target)^(1/alpha)
        t_target = (C / self._target_tol) ** (1.0 / alpha)
        current_t = len(self._history)
        remaining = int(math.ceil(t_target - current_t))
        return max(remaining, 0)

    def gap_history(self) -> npt.NDArray[np.float64]:
        """Return array of absolute gaps across iterations."""
        return np.array([s.gap for s in self._history], dtype=np.float64)

    def upper_bound_history(self) -> npt.NDArray[np.float64]:
        """Return array of upper bounds across iterations."""
        return np.array([s.upper_bound for s in self._history], dtype=np.float64)

    def lower_bound_history(self) -> npt.NDArray[np.float64]:
        """Return array of lower bounds across iterations."""
        return np.array([s.lower_bound for s in self._history], dtype=np.float64)

    def summary(self) -> str:
        """Return a human-readable convergence summary."""
        if not self._history:
            return "ConvergenceMonitor: no iterations recorded"

        last = self._history[-1]
        lines = [
            f"ConvergenceMonitor: {len(self._history)} iterations",
            f"  Gap: {last.gap:.6e} (target: {self._target_tol:.1e})",
            f"  Bounds: [{self._best_lower:.6f}, {self._best_upper:.6f}]",
            f"  Grid size: {last.grid_size}",
            f"  Elapsed: {last.elapsed:.2f}s",
        ]
        if self._termination_reason:
            lines.append(f"  Terminated: {self._termination_reason}")
        try:
            alpha, C = self.convergence_rate()
            lines.append(f"  Rate: gap ~ {C:.2f} / t^{alpha:.2f}")
        except ValueError:
            pass
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self._history)
        gap = f"{self.current_gap:.3e}" if self._history else "N/A"
        return f"ConvergenceMonitor(iter={n}, gap={gap}, target={self._target_tol:.1e})"
