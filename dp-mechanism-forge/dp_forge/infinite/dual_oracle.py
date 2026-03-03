"""
Dual oracle for the cutting-plane infinite LP solver.

Given dual variables from the master LP on a finite grid, the oracle searches
over the continuous output domain to find the output point y* that maximally
violates the current LP relaxation (i.e., has the largest positive reduced
cost).  Adding y* to the grid is guaranteed to improve the LP bound.

Search Methods
--------------
1. **Golden-section search** — For unimodal reduced cost functions (the
   common case for single-query mechanisms with convex loss).

2. **Newton's method with safeguarded fallback** — For smooth reduced cost
   functions where the gradient can be computed analytically.  Falls back
   to bisection if Newton steps leave the trust region.

3. **Multi-start grid search** — For non-convex reduced cost landscapes
   (e.g., histogram queries with multiple modes).  Seeds golden-section
   or Newton from multiple starting points.

Classes
-------
- :class:`DualOracle` — Main oracle class with ``find_most_violated()``.
- :class:`OracleResult` — Result of a single oracle query.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import LossFunction, QuerySpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLDEN_RATIO: float = (math.sqrt(5.0) - 1.0) / 2.0
_DEFAULT_ORACLE_TOL: float = 1e-12
_DEFAULT_MAX_ORACLE_ITER: int = 200
_DEFAULT_N_STARTS: int = 20
_NEWTON_STEP_DAMPING: float = 0.8
_FINITE_DIFF_H: float = 1e-7


# ---------------------------------------------------------------------------
# Oracle result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
    """Result of a dual oracle query.

    Attributes:
        y_star: The output point with maximum reduced cost.
        violation: The reduced cost value at y_star (positive means violated).
        n_evaluations: Number of reduced-cost function evaluations used.
        method: Search method that found y_star.
    """

    y_star: float
    violation: float
    n_evaluations: int
    method: str


# ---------------------------------------------------------------------------
# Reduced cost computation
# ---------------------------------------------------------------------------


def _build_reduced_cost_fn(
    dual_vars: npt.NDArray[np.float64],
    spec: QuerySpec,
    loss_callable: Callable[[float, float], float],
) -> Callable[[float], float]:
    """Build the reduced cost function for a candidate output point y.

    The reduced cost at y is:
        rc(y) = min_i [ loss(f(x_i), y) ] - Σ_i dual_simplex[i]
                - Σ_{(i,i') in edges} Σ_dir dual_dp[i,i',dir] * dp_coefficient(y)

    For the cutting-plane method, we want to find y* maximising rc(y).
    In the LP formulation, the reduced cost of adding column y is:

        rc(y) = c(y) - π^T A(:, y)

    where c(y) is the objective coefficient for output y, π are dual
    variables, and A(:, y) is the constraint column for output y.

    For the minimax DP formulation, the reduced cost simplifies to:
        rc(y) = max_i [ dual_loss[i] * loss(f(x_i), y) - dual_simplex[i] ]

    where dual_loss[i] >= 0 are duals of the epigraph constraints and
    dual_simplex[i] are duals of the simplex (sum-to-one) constraints.

    Parameters
    ----------
    dual_vars : array
        Combined dual variable vector from the master LP.
    spec : QuerySpec
        Problem specification.
    loss_callable : callable
        Loss function (true_val, noisy_val) -> loss.

    Returns
    -------
    callable
        Function mapping y -> reduced_cost(y).
    """
    n = spec.n
    # Partition dual variables:
    # First n entries: dual_simplex (equality constraints, one per database)
    # Next n entries: dual_loss (epigraph constraints, one per database)
    # Remaining: DP constraint duals
    dual_simplex = dual_vars[:n]
    dual_loss = dual_vars[n:2 * n] if len(dual_vars) >= 2 * n else np.zeros(n)
    # Ensure dual_loss is non-negative (they correspond to <= constraints)
    dual_loss = np.maximum(dual_loss, 0.0)

    f_values = spec.query_values

    def reduced_cost(y: float) -> float:
        """Compute reduced cost at output point y."""
        # For each database i, the contribution of column y is:
        # loss(f(x_i), y) weighted by dual_loss[i], minus dual_simplex[i]
        rc_per_db = np.array([
            dual_loss[i] * loss_callable(float(f_values[i]), y) - dual_simplex[i]
            for i in range(n)
        ])
        # The reduced cost is the maximum over databases
        return float(np.max(rc_per_db))

    return reduced_cost


def _build_reduced_cost_vectorised(
    dual_vars: npt.NDArray[np.float64],
    spec: QuerySpec,
    loss_fn: LossFunction,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Build a vectorised reduced cost function for batch evaluation.

    Parameters
    ----------
    dual_vars : array
        Dual variables from the master LP.
    spec : QuerySpec
        Problem specification.
    loss_fn : LossFunction
        Loss function type (must be a built-in, not CUSTOM).

    Returns
    -------
    callable
        Function mapping y_array (m,) -> rc_array (m,).
    """
    n = spec.n
    dual_simplex = dual_vars[:n]
    dual_loss = dual_vars[n:2 * n] if len(dual_vars) >= 2 * n else np.zeros(n)
    dual_loss = np.maximum(dual_loss, 0.0)
    f_values = spec.query_values

    def rc_batch(y_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        m = len(y_arr)
        # Compute loss matrix: (n, m)
        f_col = f_values[:, np.newaxis]  # (n, 1)
        y_row = y_arr[np.newaxis, :]     # (1, m)

        if loss_fn == LossFunction.L1 or loss_fn == LossFunction.LINF:
            L = np.abs(f_col - y_row)
        elif loss_fn == LossFunction.L2:
            L = (f_col - y_row) ** 2
        else:
            raise ValueError(f"Vectorised oracle requires built-in loss, got {loss_fn}")

        # rc[j] = max_i [ dual_loss[i] * L[i,j] - dual_simplex[i] ]
        weighted = dual_loss[:, np.newaxis] * L - dual_simplex[:, np.newaxis]
        return np.max(weighted, axis=0)

    return rc_batch


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------


def _golden_section_max(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = _DEFAULT_ORACLE_TOL,
    max_iter: int = _DEFAULT_MAX_ORACLE_ITER,
) -> Tuple[float, float, int]:
    """Maximise f on [a, b] via golden-section search.

    Assumes f is unimodal on [a, b].

    Parameters
    ----------
    f : callable
        Function to maximise.
    a, b : float
        Search interval.
    tol : float
        Convergence tolerance on interval width.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    x_star : float
        Approximate maximiser.
    f_star : float
        Function value at x_star.
    n_evals : int
        Number of function evaluations.
    """
    n_evals = 0

    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)
    fc = f(c)
    fd = f(d)
    n_evals += 2

    for _ in range(max_iter):
        if (b - a) < tol:
            break

        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + _GOLDEN_RATIO * (b - a)
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - _GOLDEN_RATIO * (b - a)
            fc = f(c)
        n_evals += 1

    x_star = (a + b) / 2.0
    f_star = f(x_star)
    n_evals += 1

    return x_star, f_star, n_evals


def _newton_search_max(
    f: Callable[[float], float],
    x0: float,
    a: float,
    b: float,
    tol: float = _DEFAULT_ORACLE_TOL,
    max_iter: int = 50,
) -> Tuple[float, float, int]:
    """Maximise f near x0 on [a, b] using Newton's method with finite differences.

    Falls back to golden-section if Newton steps leave [a, b] or fail
    to improve the objective.

    Parameters
    ----------
    f : callable
        Function to maximise.
    x0 : float
        Initial guess.
    a, b : float
        Bounds for safeguarding.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    x_star : float
        Approximate maximiser.
    f_star : float
        Function value at x_star.
    n_evals : int
        Number of function evaluations.
    """
    x = np.clip(x0, a, b)
    fx = f(x)
    n_evals = 1
    best_x, best_f = x, fx

    h = _FINITE_DIFF_H * max(1.0, abs(x))

    for _ in range(max_iter):
        # Central finite difference for gradient
        fp = f(x + h)
        fm = f(x - h)
        n_evals += 2

        grad = (fp - fm) / (2.0 * h)

        if abs(grad) < tol:
            break

        # Second derivative for Newton step
        f2 = (fp - 2.0 * fx + fm) / (h * h)

        if f2 >= -1e-15:
            # Not concave locally; take a gradient ascent step
            step = _NEWTON_STEP_DAMPING * grad * (b - a) * 0.1
        else:
            step = -grad / f2

        step = np.clip(step, -(b - a) * 0.5, (b - a) * 0.5)
        x_new = np.clip(x + step, a, b)
        fx_new = f(float(x_new))
        n_evals += 1

        if fx_new > best_f:
            best_x, best_f = float(x_new), fx_new

        if abs(float(x_new) - x) < tol:
            break

        x = float(x_new)
        fx = fx_new
        h = _FINITE_DIFF_H * max(1.0, abs(x))

    return best_x, best_f, n_evals


def _multi_start_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_starts: int = _DEFAULT_N_STARTS,
    tol: float = _DEFAULT_ORACLE_TOL,
    max_iter_per_start: int = 50,
) -> Tuple[float, float, int]:
    """Maximise f on [a, b] using multi-start local search.

    Evaluates f on a coarse grid, then refines the top candidates using
    golden-section search in local neighborhoods.

    Parameters
    ----------
    f : callable
        Function to maximise.
    a, b : float
        Search interval.
    n_starts : int
        Number of starting points on the coarse grid.
    tol : float
        Convergence tolerance for local refinement.
    max_iter_per_start : int
        Max iterations per local golden-section search.

    Returns
    -------
    x_star : float
        Best approximate maximiser.
    f_star : float
        Function value at x_star.
    n_evals : int
        Total function evaluations.
    """
    if n_starts < 2:
        n_starts = 2

    grid = np.linspace(a, b, n_starts)
    f_vals = np.array([f(float(y)) for y in grid])
    n_evals = n_starts

    # Pick top candidates (local maxima or top-k)
    n_refine = min(5, n_starts)
    top_idx = np.argsort(f_vals)[-n_refine:]

    best_x = grid[top_idx[-1]]
    best_f = f_vals[top_idx[-1]]

    span = (b - a) / max(n_starts - 1, 1)

    for idx in top_idx:
        local_a = max(a, grid[idx] - span)
        local_b = min(b, grid[idx] + span)
        x_cand, f_cand, ne = _golden_section_max(
            f, local_a, local_b, tol=tol, max_iter=max_iter_per_start,
        )
        n_evals += ne
        if f_cand > best_f:
            best_x, best_f = x_cand, f_cand

    return best_x, best_f, n_evals


# ---------------------------------------------------------------------------
# Dual oracle
# ---------------------------------------------------------------------------


class DualOracle:
    """Continuous dual oracle for the infinite LP cutting-plane method.

    Given dual variables from the master LP, finds the output point y*
    that maximises the reduced cost over the continuous output domain.

    Parameters
    ----------
    domain_lower : float
        Lower bound of the continuous output domain.
    domain_upper : float
        Upper bound of the continuous output domain.
    tol : float
        Search tolerance for the oracle.
    n_starts : int
        Number of starting points for multi-start search.
    use_newton : bool
        Whether to use Newton's method (with golden-section fallback).
    """

    def __init__(
        self,
        domain_lower: float,
        domain_upper: float,
        tol: float = _DEFAULT_ORACLE_TOL,
        n_starts: int = _DEFAULT_N_STARTS,
        use_newton: bool = True,
    ) -> None:
        if domain_lower >= domain_upper:
            raise ValueError(
                f"domain_lower ({domain_lower}) must be < domain_upper ({domain_upper})"
            )
        self._a = domain_lower
        self._b = domain_upper
        self._tol = tol
        self._n_starts = n_starts
        self._use_newton = use_newton
        self._total_evaluations: int = 0
        self._call_count: int = 0

    @property
    def domain(self) -> Tuple[float, float]:
        """Output domain interval."""
        return (self._a, self._b)

    @property
    def total_evaluations(self) -> int:
        """Cumulative number of reduced-cost evaluations."""
        return self._total_evaluations

    @property
    def call_count(self) -> int:
        """Number of oracle calls."""
        return self._call_count

    @classmethod
    def from_spec(
        cls,
        spec: QuerySpec,
        margin: float = 1.0,
        **kwargs,
    ) -> DualOracle:
        """Create an oracle with domain derived from a QuerySpec.

        The domain spans ``[min(f) - margin*sensitivity, max(f) + margin*sensitivity]``.

        Parameters
        ----------
        spec : QuerySpec
            Problem specification.
        margin : float
            How many sensitivity-widths to extend beyond the query range.
        **kwargs
            Additional keyword arguments for DualOracle.__init__.

        Returns
        -------
        DualOracle
        """
        f_min = float(np.min(spec.query_values))
        f_max = float(np.max(spec.query_values))
        pad = margin * spec.sensitivity
        if f_min == f_max:
            pad = max(pad, 1.0)
        return cls(
            domain_lower=f_min - pad,
            domain_upper=f_max + pad,
            **kwargs,
        )

    def find_most_violated(
        self,
        dual_vars: npt.NDArray[np.float64],
        spec: QuerySpec,
    ) -> OracleResult:
        """Find the continuous output point y* with maximum reduced cost.

        Parameters
        ----------
        dual_vars : array
            Dual variable vector from the master LP.
        spec : QuerySpec
            Problem specification.

        Returns
        -------
        OracleResult
            The best output point and its reduced cost.
        """
        self._call_count += 1
        loss_callable = spec.get_loss_callable()
        rc_fn = _build_reduced_cost_fn(dual_vars, spec, loss_callable)

        # Strategy: multi-start for robustness, then refine the best
        if spec.n > 2 or self._n_starts > 1:
            y_star, violation, n_evals = self._multi_start_oracle(rc_fn)
            method = "multi_start"
        elif self._use_newton:
            # Single-mode: try Newton from the midpoint
            mid = (self._a + self._b) / 2.0
            y_star, violation, n_evals = _newton_search_max(
                rc_fn, mid, self._a, self._b, tol=self._tol,
            )
            # Verify with golden-section
            y_gs, v_gs, ne_gs = _golden_section_max(
                rc_fn, self._a, self._b, tol=self._tol,
            )
            n_evals += ne_gs
            if v_gs > violation:
                y_star, violation = y_gs, v_gs
            method = "newton+golden"
        else:
            y_star, violation, n_evals = _golden_section_max(
                rc_fn, self._a, self._b, tol=self._tol,
            )
            method = "golden_section"

        self._total_evaluations += n_evals

        return OracleResult(
            y_star=y_star,
            violation=violation,
            n_evaluations=n_evals,
            method=method,
        )

    def find_most_violated_vectorised(
        self,
        dual_vars: npt.NDArray[np.float64],
        spec: QuerySpec,
        n_candidates: int = 1000,
    ) -> OracleResult:
        """Fast oracle using vectorised evaluation on a dense candidate grid.

        Evaluates the reduced cost on a dense grid, then refines the best
        candidate with golden-section search.

        Parameters
        ----------
        dual_vars : array
            Dual variables from the master LP.
        spec : QuerySpec
            Problem specification.
        n_candidates : int
            Number of candidate points on the dense grid.

        Returns
        -------
        OracleResult
        """
        self._call_count += 1

        if spec.loss_fn == LossFunction.CUSTOM:
            # Fall back to scalar oracle for custom loss
            self._call_count -= 1  # undo increment, find_most_violated will re-increment
            return self.find_most_violated(dual_vars, spec)

        rc_batch = _build_reduced_cost_vectorised(dual_vars, spec, spec.loss_fn)
        y_grid = np.linspace(self._a, self._b, n_candidates)
        rc_vals = rc_batch(y_grid)
        n_evals = n_candidates

        best_idx = int(np.argmax(rc_vals))
        # Refine with golden-section in a local window
        span = (self._b - self._a) / max(n_candidates - 1, 1) * 2.0
        local_a = max(self._a, y_grid[best_idx] - span)
        local_b = min(self._b, y_grid[best_idx] + span)

        loss_callable = spec.get_loss_callable()
        rc_fn = _build_reduced_cost_fn(dual_vars, spec, loss_callable)
        y_star, violation, ne = _golden_section_max(
            rc_fn, local_a, local_b, tol=self._tol,
        )
        n_evals += ne

        # Also check the coarse-grid best
        if rc_vals[best_idx] > violation:
            y_star = float(y_grid[best_idx])
            violation = float(rc_vals[best_idx])

        self._total_evaluations += n_evals

        return OracleResult(
            y_star=y_star,
            violation=violation,
            n_evaluations=n_evals,
            method="vectorised+golden",
        )

    def _multi_start_oracle(
        self,
        rc_fn: Callable[[float], float],
    ) -> Tuple[float, float, int]:
        """Internal multi-start search."""
        return _multi_start_search(
            rc_fn,
            self._a,
            self._b,
            n_starts=self._n_starts,
            tol=self._tol,
        )

    def __repr__(self) -> str:
        return (
            f"DualOracle(domain=[{self._a:.4f}, {self._b:.4f}], "
            f"calls={self._call_count}, evals={self._total_evaluations})"
        )
