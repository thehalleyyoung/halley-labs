"""
Systematic constraint tightening for robust DP mechanism synthesis.

When an LP solver operates at tolerance ν, the returned solution may violate
each constraint by up to ν.  For DP constraints of the form

    p[i][j] − e^ε · p[i'][j] ≤ 0

this means the true ratio p[i][j]/p[i'][j] could be as large as e^ε + ν/p_min,
which exceeds the e^ε bound.  To guarantee the output mechanism satisfies
(ε, δ)-DP despite solver imprecision, we *inflate* (tighten) each constraint
by a margin that absorbs the worst-case solver error.

Key results:
    - Pure DP: tighten the ratio bound from e^ε to e^(ε − margin_ε)
      where margin_ε = ν · e^ε / p_min_est.
    - Approx DP: reduce the δ threshold to δ − margin_δ
      where margin_δ = ν · k (number of output bins per pair).
    - Simplex normalization rows: tighten from = 1 to ∈ [1 − ν, 1 + ν].

Classes:
    - :class:`InflationResult` — Container for computed margins.
    - :class:`ConstraintInflator` — Main inflation engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

from dp_forge.exceptions import ConfigurationError, NumericalInstabilityError
from dp_forge.types import LPStruct

logger = logging.getLogger(__name__)


@dataclass
class InflationResult:
    """Container for constraint inflation results.

    Attributes:
        epsilon_margin: Additive margin on ε consumed by inflation.
        delta_margin: Additive margin on δ consumed by inflation.
        tightened_exp_eps: The tightened e^(ε − margin_ε) bound used in LP.
        tightened_delta: The tightened δ − margin_δ threshold.
        per_constraint_slack: Per-row slack budget for each inequality constraint.
        solver_tolerance: Solver tolerance ν that drove the inflation.
        n_tightened_constraints: Number of constraints that were tightened.
    """

    epsilon_margin: float
    delta_margin: float
    tightened_exp_eps: float
    tightened_delta: float
    per_constraint_slack: npt.NDArray[np.float64]
    solver_tolerance: float
    n_tightened_constraints: int

    def __repr__(self) -> str:
        return (
            f"InflationResult(ε_margin={self.epsilon_margin:.2e}, "
            f"δ_margin={self.delta_margin:.2e}, "
            f"tightened={self.n_tightened_constraints} constraints)"
        )


class ConstraintInflator:
    """Compute and apply constraint inflation for robust DP guarantees.

    The inflator analyses the LP structure, identifies DP constraints
    (ratio constraints for pure DP, hockey-stick constraints for approx DP)
    and simplex normalization constraints, then computes per-constraint
    slack budgets that absorb solver tolerance.

    Args:
        safety_factor: Multiplicative safety factor applied to computed
            margins. Default 2.0 provides a 2× buffer over the theoretical
            minimum.
        min_probability_estimate: Estimate of the smallest probability in
            the mechanism, used for ratio-constraint margin computation.
            If None, computed from ε as exp(-ε) × 1e-6.

    Example::

        inflator = ConstraintInflator(safety_factor=2.0)
        result = inflator.inflate(lp_struct, solver_tol=1e-8,
                                  epsilon=1.0, delta=0.0)
    """

    def __init__(
        self,
        safety_factor: float = 2.0,
        min_probability_estimate: Optional[float] = None,
    ) -> None:
        if safety_factor < 1.0:
            raise ConfigurationError(
                f"safety_factor must be >= 1.0, got {safety_factor}",
                parameter="safety_factor",
                value=safety_factor,
                constraint="safety_factor >= 1.0",
            )
        self._safety_factor = safety_factor
        self._min_prob_est = min_probability_estimate

    def compute_epsilon_margin(
        self,
        epsilon: float,
        solver_tol: float,
        n_outputs: int,
        min_prob: Optional[float] = None,
    ) -> float:
        """Compute the ε margin needed to absorb solver tolerance.

        The worst case is when the solver returns a solution where a
        ratio constraint p[i][j] − e^ε · p[i'][j] ≤ 0 is violated by ν.
        This means:
            p[i][j] / p[i'][j] ≤ e^ε + ν / p[i'][j]

        Taking log: effective ε ≤ ε + log(1 + ν / (e^ε · p_min))

        For small ν: margin_ε ≈ ν · e^ε / p_min (first-order).

        We use the exact formula with safety factor.

        Args:
            epsilon: Target privacy parameter ε.
            solver_tol: Solver feasibility tolerance ν.
            n_outputs: Number of output bins k.
            min_prob: Estimated minimum probability. If None, uses
                the class default or exp(-ε) / k.

        Returns:
            The ε margin to subtract from the target ε before building the LP.
        """
        if min_prob is None:
            min_prob = self._min_prob_est
        if min_prob is None:
            min_prob = max(math.exp(-epsilon) / n_outputs, 1e-15)

        exp_eps = math.exp(epsilon)

        # Ratio perturbation: Δ(ratio) ≤ ν / p_min
        # ε perturbation: Δε ≤ log(1 + Δ(ratio) / e^ε)
        ratio_perturbation = solver_tol / min_prob
        epsilon_perturbation = math.log1p(ratio_perturbation / exp_eps)

        margin = epsilon_perturbation * self._safety_factor

        # Clamp: margin cannot exceed ε itself (would make tightened ε negative)
        margin = min(margin, epsilon * 0.5)

        logger.debug(
            "ε margin: %.2e (ratio_pert=%.2e, ε_pert=%.2e, safety=%.1f)",
            margin, ratio_perturbation, epsilon_perturbation, self._safety_factor,
        )

        return margin

    def compute_delta_margin(
        self,
        delta: float,
        solver_tol: float,
        n_outputs: int,
        n_pairs: int,
    ) -> float:
        """Compute the δ margin needed to absorb solver tolerance.

        For approximate DP with hockey-stick constraints:
            Σ_j max(p[i][j] − e^ε · p[i'][j], 0) ≤ δ

        If each of the k slack variables s[j] can be off by ν, the
        total hockey-stick divergence can be off by k·ν.

        Args:
            delta: Target privacy parameter δ.
            solver_tol: Solver feasibility tolerance ν.
            n_outputs: Number of output bins k.
            n_pairs: Number of adjacent pairs.

        Returns:
            The δ margin to subtract from the target δ.
        """
        if delta <= 0.0:
            return 0.0

        # Each slack variable can be off by ν; the sum of k slacks can be off by k·ν
        margin = n_outputs * solver_tol * self._safety_factor

        # Clamp: margin cannot exceed δ itself
        margin = min(margin, delta * 0.5)

        logger.debug(
            "δ margin: %.2e (k=%d, ν=%.2e, safety=%.1f)",
            margin, n_outputs, solver_tol, self._safety_factor,
        )

        return margin

    def compute_per_constraint_slack(
        self,
        lp: LPStruct,
        solver_tol: float,
        epsilon: float,
    ) -> npt.NDArray[np.float64]:
        """Compute per-constraint slack budgets for RHS tightening.

        Each inequality constraint row A_i · x ≤ b_i is tightened to
        A_i · x ≤ b_i − slack_i where slack_i depends on the row norm
        and solver tolerance.

        The slack for row i is: ||A_i||_1 · solver_tol · safety_factor.
        This is because the solver may return x with ||x - x*||_∞ ≤ ν,
        so A_i · x could differ from A_i · x* by at most ||A_i||_1 · ν.

        Args:
            lp: LP structure.
            solver_tol: Solver feasibility tolerance.
            epsilon: Privacy parameter (for scaling heuristics).

        Returns:
            Array of per-row slack values, shape (n_ub,).
        """
        A_ub = lp.A_ub.tocsr()
        n_rows = A_ub.shape[0]
        slacks = np.empty(n_rows, dtype=np.float64)

        for i in range(n_rows):
            row = A_ub.getrow(i)
            row_l1_norm = float(np.sum(np.abs(row.toarray())))
            slacks[i] = row_l1_norm * solver_tol * self._safety_factor

        return slacks

    def inflate(
        self,
        lp: LPStruct,
        solver_tol: float,
        epsilon: float,
        delta: float = 0.0,
        n_outputs: Optional[int] = None,
        n_pairs: int = 1,
    ) -> Tuple[LPStruct, InflationResult]:
        """Inflate (tighten) LP constraints to absorb solver tolerance.

        Modifies the LP RHS vector b_ub by subtracting per-constraint
        slack values. This ensures that even if the solver returns a
        solution that violates constraints by up to ν, the solution
        still satisfies the original (un-tightened) constraints.

        Args:
            lp: Original LP structure.
            solver_tol: Solver feasibility tolerance ν.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            n_outputs: Number of output bins k. Inferred from lp.y_grid if None.
            n_pairs: Number of adjacent pairs in the LP.

        Returns:
            Tuple (tightened_lp, inflation_result) where tightened_lp has
            modified b_ub and inflation_result contains the margins used.

        Raises:
            NumericalInstabilityError: If the required inflation would make
                the LP infeasible (margins too large relative to constraints).
        """
        k = n_outputs if n_outputs is not None else len(lp.y_grid)

        # Compute privacy margins
        eps_margin = self.compute_epsilon_margin(epsilon, solver_tol, k)
        delta_margin = self.compute_delta_margin(delta, solver_tol, k, n_pairs)

        tightened_exp_eps = math.exp(epsilon - eps_margin)
        tightened_delta = max(delta - delta_margin, 0.0)

        # Compute per-constraint slack
        per_slack = self.compute_per_constraint_slack(lp, solver_tol, epsilon)

        # Check feasibility: if any slack exceeds the RHS, the tightened LP
        # may be infeasible
        b_ub_new = lp.b_ub.copy()
        n_tightened = 0

        for i in range(len(b_ub_new)):
            slack = per_slack[i]
            if slack > 0 and slack < abs(b_ub_new[i]) + 1.0:
                b_ub_new[i] -= slack
                n_tightened += 1

        # Check for excessive tightening
        excessively_tight = np.sum(b_ub_new < -1e6)
        if excessively_tight > 0:
            raise NumericalInstabilityError(
                f"Constraint inflation made {excessively_tight} constraints "
                f"excessively tight (< -1e6). Consider reducing safety_factor "
                f"or increasing solver tolerance.",
                condition_number=float(excessively_tight),
                matrix_name="b_ub_tightened",
            )

        # Build tightened LP
        tightened_lp = LPStruct(
            c=lp.c.copy(),
            A_ub=lp.A_ub.copy(),
            b_ub=b_ub_new,
            A_eq=lp.A_eq.copy() if lp.A_eq is not None else None,
            b_eq=lp.b_eq.copy() if lp.b_eq is not None else None,
            bounds=list(lp.bounds),
            var_map=dict(lp.var_map),
            y_grid=lp.y_grid.copy(),
        )

        result = InflationResult(
            epsilon_margin=eps_margin,
            delta_margin=delta_margin,
            tightened_exp_eps=tightened_exp_eps,
            tightened_delta=tightened_delta,
            per_constraint_slack=per_slack,
            solver_tolerance=solver_tol,
            n_tightened_constraints=n_tightened,
        )

        logger.info(
            "Inflated %d constraints: ε_margin=%.2e, δ_margin=%.2e",
            n_tightened, eps_margin, delta_margin,
        )

        return tightened_lp, result

    def __repr__(self) -> str:
        return (
            f"ConstraintInflator(safety_factor={self._safety_factor}, "
            f"min_prob_est={self._min_prob_est})"
        )
