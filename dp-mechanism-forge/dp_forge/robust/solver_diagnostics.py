"""
Post-solve diagnostics for LP solutions in DP mechanism synthesis.

Provides tools for:
    - Constraint satisfaction auditing: check every LP constraint and report
      violations with their magnitudes.
    - Residual analysis: compute primal and dual residuals to assess solution
      quality.
    - Basis condition number computation using the active constraint set.
    - Iterative refinement for ill-conditioned solutions: correct the LP
      solution using residual-based updates.

Key class:
    - :class:`SolverDiagnostics` — Main diagnostics engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

from dp_forge.exceptions import NumericalInstabilityError
from dp_forge.types import LPStruct

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """A single constraint violation.

    Attributes:
        index: Row index in A_ub or A_eq.
        constraint_type: 'inequality' or 'equality'.
        residual: Signed residual (positive = violated for ≤ constraints).
        magnitude: Absolute magnitude of the violation.
    """

    index: int
    constraint_type: str
    residual: float
    magnitude: float

    def __repr__(self) -> str:
        return (
            f"ConstraintViolation(idx={self.index}, "
            f"type={self.constraint_type}, "
            f"residual={self.residual:.2e})"
        )


@dataclass
class AuditReport:
    """Result of constraint satisfaction audit.

    Attributes:
        n_inequality_checked: Number of inequality constraints checked.
        n_equality_checked: Number of equality constraints checked.
        n_inequality_violated: Number of violated inequality constraints.
        n_equality_violated: Number of violated equality constraints.
        max_inequality_violation: Maximum violation among inequality constraints.
        max_equality_violation: Maximum violation among equality constraints.
        violations: List of individual violations (worst first).
        is_feasible: Whether the solution satisfies all constraints within tol.
    """

    n_inequality_checked: int
    n_equality_checked: int
    n_inequality_violated: int
    n_equality_violated: int
    max_inequality_violation: float
    max_equality_violation: float
    violations: List[ConstraintViolation]
    is_feasible: bool

    def __repr__(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        return (
            f"AuditReport({status}, "
            f"ub_viol={self.n_inequality_violated}/{self.n_inequality_checked}, "
            f"eq_viol={self.n_equality_violated}/{self.n_equality_checked}, "
            f"max_ub={self.max_inequality_violation:.2e}, "
            f"max_eq={self.max_equality_violation:.2e})"
        )

    def summary(self) -> str:
        """Human-readable summary of the audit."""
        lines = [
            f"Constraint Audit: {'FEASIBLE' if self.is_feasible else 'INFEASIBLE'}",
            f"  Inequality: {self.n_inequality_violated}/{self.n_inequality_checked} violated "
            f"(max={self.max_inequality_violation:.2e})",
            f"  Equality: {self.n_equality_violated}/{self.n_equality_checked} violated "
            f"(max={self.max_equality_violation:.2e})",
        ]
        if self.violations:
            lines.append(f"  Top violations:")
            for v in self.violations[:5]:
                lines.append(f"    [{v.constraint_type}:{v.index}] residual={v.residual:.4e}")
        return "\n".join(lines)


@dataclass
class ResidualReport:
    """Primal and dual residual analysis.

    Attributes:
        primal_ub_residuals: Per-constraint residuals for A_ub·x - b_ub.
        primal_eq_residuals: Per-constraint residuals for A_eq·x - b_eq.
        primal_infeasibility: Max constraint violation (should be ≤ solver_tol).
        bound_violations: Per-variable bound violations.
        max_bound_violation: Maximum variable bound violation.
    """

    primal_ub_residuals: npt.NDArray[np.float64]
    primal_eq_residuals: npt.NDArray[np.float64]
    primal_infeasibility: float
    bound_violations: npt.NDArray[np.float64]
    max_bound_violation: float

    def __repr__(self) -> str:
        return (
            f"ResidualReport(primal_infeas={self.primal_infeasibility:.2e}, "
            f"max_bound_viol={self.max_bound_violation:.2e})"
        )


@dataclass
class RefinementResult:
    """Result of iterative refinement.

    Attributes:
        x_refined: Refined solution vector.
        iterations: Number of refinement iterations performed.
        initial_residual: Residual norm before refinement.
        final_residual: Residual norm after refinement.
        converged: Whether refinement converged.
    """

    x_refined: npt.NDArray[np.float64]
    iterations: int
    initial_residual: float
    final_residual: float
    converged: bool

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"RefinementResult({status}, iter={self.iterations}, "
            f"residual: {self.initial_residual:.2e} → {self.final_residual:.2e})"
        )


class SolverDiagnostics:
    """Post-solve diagnostics and solution refinement for LP solutions.

    Provides tools to audit constraint satisfaction, analyse residuals,
    compute basis condition numbers, and iteratively refine solutions
    that are near the feasibility boundary.

    Args:
        feasibility_tol: Tolerance for considering a constraint satisfied.
        max_refinement_iter: Maximum iterations for iterative refinement.
        max_condition_number: Threshold for ill-conditioning warnings.

    Example::

        diagnostics = SolverDiagnostics()
        report = diagnostics.audit_constraints(x_solution, lp_struct)
        if not report.is_feasible:
            refined = diagnostics.iterative_refine(x_solution, lp_struct)
    """

    def __init__(
        self,
        feasibility_tol: float = 1e-8,
        max_refinement_iter: int = 20,
        max_condition_number: float = 1e12,
    ) -> None:
        if feasibility_tol <= 0:
            raise ValueError(f"feasibility_tol must be > 0, got {feasibility_tol}")
        if max_refinement_iter < 1:
            raise ValueError(f"max_refinement_iter must be >= 1, got {max_refinement_iter}")
        self._feas_tol = feasibility_tol
        self._max_refine_iter = max_refinement_iter
        self._max_cond = max_condition_number

    def audit_constraints(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
        tol: Optional[float] = None,
    ) -> AuditReport:
        """Check every LP constraint against the solution vector.

        Computes A_ub·x - b_ub for inequality constraints (positive = violated)
        and |A_eq·x - b_eq| for equality constraints.

        Args:
            x: Solution vector, shape (n_vars,).
            lp: LP structure.
            tol: Feasibility tolerance. Uses class default if None.

        Returns:
            AuditReport with violation details.
        """
        tol = tol if tol is not None else self._feas_tol
        x = np.asarray(x, dtype=np.float64)

        violations: List[ConstraintViolation] = []

        # Inequality constraints: A_ub · x ≤ b_ub
        ub_residuals = np.asarray(lp.A_ub @ x).ravel() - lp.b_ub
        n_ub = len(ub_residuals)
        ub_violated = ub_residuals > tol
        n_ub_violated = int(np.sum(ub_violated))
        max_ub_viol = float(np.max(ub_residuals)) if n_ub > 0 else 0.0

        for idx in np.where(ub_violated)[0]:
            violations.append(ConstraintViolation(
                index=int(idx),
                constraint_type="inequality",
                residual=float(ub_residuals[idx]),
                magnitude=float(ub_residuals[idx]),
            ))

        # Equality constraints: A_eq · x = b_eq
        n_eq = 0
        n_eq_violated = 0
        max_eq_viol = 0.0
        if lp.A_eq is not None and lp.b_eq is not None:
            eq_residuals = np.abs(np.asarray(lp.A_eq @ x).ravel() - lp.b_eq)
            n_eq = len(eq_residuals)
            eq_violated = eq_residuals > tol
            n_eq_violated = int(np.sum(eq_violated))
            max_eq_viol = float(np.max(eq_residuals)) if n_eq > 0 else 0.0

            for idx in np.where(eq_violated)[0]:
                violations.append(ConstraintViolation(
                    index=int(idx),
                    constraint_type="equality",
                    residual=float(eq_residuals[idx]),
                    magnitude=float(eq_residuals[idx]),
                ))

        # Sort violations by magnitude (worst first)
        violations.sort(key=lambda v: -v.magnitude)

        is_feasible = n_ub_violated == 0 and n_eq_violated == 0

        return AuditReport(
            n_inequality_checked=n_ub,
            n_equality_checked=n_eq,
            n_inequality_violated=n_ub_violated,
            n_equality_violated=n_eq_violated,
            max_inequality_violation=max(max_ub_viol, 0.0),
            max_equality_violation=max(max_eq_viol, 0.0),
            violations=violations,
            is_feasible=is_feasible,
        )

    def compute_residuals(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
    ) -> ResidualReport:
        """Compute primal residuals for the LP solution.

        Args:
            x: Solution vector, shape (n_vars,).
            lp: LP structure.

        Returns:
            ResidualReport with per-constraint residuals.
        """
        x = np.asarray(x, dtype=np.float64)

        # Inequality residuals: A_ub · x - b_ub (positive = violated)
        ub_residuals = np.asarray(lp.A_ub @ x).ravel() - lp.b_ub

        # Equality residuals: A_eq · x - b_eq
        if lp.A_eq is not None and lp.b_eq is not None:
            eq_residuals = np.asarray(lp.A_eq @ x).ravel() - lp.b_eq
        else:
            eq_residuals = np.array([], dtype=np.float64)

        # Primal infeasibility: max violation
        max_ub = float(np.max(ub_residuals)) if len(ub_residuals) > 0 else 0.0
        max_eq = float(np.max(np.abs(eq_residuals))) if len(eq_residuals) > 0 else 0.0
        primal_infeas = max(max_ub, max_eq, 0.0)

        # Variable bound violations
        bound_viols = np.zeros(len(x), dtype=np.float64)
        for i, (lb, ub) in enumerate(lp.bounds):
            if lb is not None and x[i] < lb:
                bound_viols[i] = lb - x[i]
            if ub is not None and x[i] > ub:
                bound_viols[i] = x[i] - ub
        max_bound = float(np.max(bound_viols)) if len(bound_viols) > 0 else 0.0

        return ResidualReport(
            primal_ub_residuals=ub_residuals,
            primal_eq_residuals=eq_residuals,
            primal_infeasibility=primal_infeas,
            bound_violations=bound_viols,
            max_bound_violation=max_bound,
        )

    def compute_basis_condition(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
        binding_tol: Optional[float] = None,
    ) -> float:
        """Compute the condition number of the active constraint basis.

        Identifies binding inequality constraints (residual within tol)
        and equality constraints, forms the active constraint matrix,
        and computes its condition number.

        Args:
            x: Solution vector.
            lp: LP structure.
            binding_tol: Tolerance for identifying binding constraints.

        Returns:
            Condition number of the active basis matrix.

        Raises:
            NumericalInstabilityError: If condition number exceeds threshold.
        """
        binding_tol = binding_tol if binding_tol is not None else self._feas_tol * 10.0
        x = np.asarray(x, dtype=np.float64)

        # Find binding inequality constraints
        ub_residuals = np.asarray(lp.A_ub @ x).ravel() - lp.b_ub
        binding_mask = np.abs(ub_residuals) < binding_tol
        binding_rows = lp.A_ub.toarray()[binding_mask]

        # Add equality constraints
        if lp.A_eq is not None:
            eq_rows = lp.A_eq.toarray()
            if len(binding_rows) > 0:
                active_matrix = np.vstack([binding_rows, eq_rows])
            else:
                active_matrix = eq_rows
        else:
            active_matrix = binding_rows

        if active_matrix.size == 0:
            return 1.0

        try:
            cond = float(np.linalg.cond(active_matrix))
        except np.linalg.LinAlgError:
            cond = float("inf")

        if cond > self._max_cond:
            raise NumericalInstabilityError(
                f"Active basis condition number {cond:.2e} exceeds threshold "
                f"{self._max_cond:.2e}",
                condition_number=cond,
                max_condition_number=self._max_cond,
                matrix_name="active_basis",
            )

        return cond

    def iterative_refine(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
        target_tol: Optional[float] = None,
    ) -> RefinementResult:
        """Iteratively refine an LP solution to reduce constraint violations.

        Uses a simple projected gradient step: for each violated inequality
        constraint A_i · x > b_i, project x back toward feasibility.

        This does not re-solve the LP; it only adjusts x to satisfy
        constraints more precisely while staying close to the original
        solution.

        The refinement proceeds:
        1. Compute residuals r = A_ub · x - b_ub.
        2. For violated constraints (r_i > 0), compute correction:
           Δx = -A_i^T · r_i / ||A_i||² (project onto constraint hyperplane).
        3. Apply corrections and enforce variable bounds.
        4. Re-normalise probability rows (simplex constraint).
        5. Repeat until residuals are within tolerance or max iterations.

        Args:
            x: Initial solution vector.
            lp: LP structure.
            target_tol: Target feasibility tolerance. Uses class default if None.

        Returns:
            RefinementResult with the refined solution.
        """
        target_tol = target_tol if target_tol is not None else self._feas_tol
        x_ref = np.array(x, dtype=np.float64, copy=True)
        A_ub = lp.A_ub.tocsr()

        initial_residual = self._max_violation(x_ref, lp)

        for iteration in range(self._max_refine_iter):
            residuals = np.asarray(A_ub @ x_ref).ravel() - lp.b_ub
            max_viol = float(np.max(residuals)) if len(residuals) > 0 else 0.0

            if max_viol <= target_tol:
                return RefinementResult(
                    x_refined=x_ref,
                    iterations=iteration,
                    initial_residual=initial_residual,
                    final_residual=max_viol,
                    converged=True,
                )

            # Process violated constraints
            violated_idx = np.where(residuals > target_tol)[0]
            for idx in violated_idx:
                row = A_ub.getrow(idx).toarray().ravel()
                row_norm_sq = float(np.dot(row, row))
                if row_norm_sq < 1e-300:
                    continue
                # Step size: just enough to satisfy this constraint
                step = residuals[idx] / row_norm_sq
                x_ref -= step * row

            # Enforce variable bounds
            for i, (lb, ub) in enumerate(lp.bounds):
                if lb is not None:
                    x_ref[i] = max(x_ref[i], lb)
                if ub is not None:
                    x_ref[i] = min(x_ref[i], ub)

            # Enforce equality constraints (re-normalise simplex rows)
            if lp.A_eq is not None and lp.b_eq is not None:
                A_eq = lp.A_eq.toarray()
                b_eq = lp.b_eq
                eq_residuals = A_eq @ x_ref - b_eq
                for i in range(len(b_eq)):
                    row = A_eq[i]
                    row_norm_sq = float(np.dot(row, row))
                    if row_norm_sq < 1e-300:
                        continue
                    step = eq_residuals[i] / row_norm_sq
                    x_ref -= step * row

        final_residual = self._max_violation(x_ref, lp)
        return RefinementResult(
            x_refined=x_ref,
            iterations=self._max_refine_iter,
            initial_residual=initial_residual,
            final_residual=final_residual,
            converged=final_residual <= target_tol,
        )

    def _max_violation(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
    ) -> float:
        """Compute maximum constraint violation."""
        residuals = np.asarray(lp.A_ub @ x).ravel() - lp.b_ub
        max_ub = float(np.max(residuals)) if len(residuals) > 0 else 0.0
        max_eq = 0.0
        if lp.A_eq is not None and lp.b_eq is not None:
            eq_res = np.abs(np.asarray(lp.A_eq @ x).ravel() - lp.b_eq)
            max_eq = float(np.max(eq_res)) if len(eq_res) > 0 else 0.0
        return max(max_ub, max_eq, 0.0)

    def full_diagnostics(
        self,
        x: npt.NDArray[np.float64],
        lp: LPStruct,
    ) -> Dict[str, object]:
        """Run all diagnostics and return a summary dictionary.

        Args:
            x: Solution vector.
            lp: LP structure.

        Returns:
            Dictionary with keys 'audit', 'residuals', 'condition_number'.
        """
        audit = self.audit_constraints(x, lp)
        residuals = self.compute_residuals(x, lp)
        try:
            cond = self.compute_basis_condition(x, lp)
        except NumericalInstabilityError as e:
            cond = e.condition_number or float("inf")

        return {
            "audit": audit,
            "residuals": residuals,
            "condition_number": cond,
        }

    def __repr__(self) -> str:
        return (
            f"SolverDiagnostics(feas_tol={self._feas_tol:.2e}, "
            f"max_refine={self._max_refine_iter})"
        )
