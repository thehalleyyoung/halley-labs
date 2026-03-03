"""
Dual LP certificate generation and gap convergence tracking for DP-Forge.

Provides tools for extracting dual solutions from LP solvers (HiGHS via
scipy.optimize.linprog), verifying complementary slackness, computing
rigorous duality gap bounds, and constructing human-readable optimality
proofs.

The :class:`GapTracker` monitors convergence across CEGIS iterations,
implementing the analysis from Theorem T14 (linear gap decrease in
CEGIS iterations).

Key Components:
    - ``DualCertificateExtractor``: Extract, verify, and package dual
      certificates from LP solver output.
    - ``GapTracker``: Track duality gap convergence across CEGIS
      iterations with rate estimation and convergence prediction.

Mathematical Background:
    For an LP ``min c^T x s.t. Ax <= b, x >= 0``, the dual is
    ``max b^T y s.t. A^T y <= c, y >= 0``.  Strong duality gives
    ``c^T x* = b^T y*`` at optimality.  The duality gap
    ``c^T x - b^T y >= 0`` certifies sub-optimality.

    Complementary slackness: at optimality, for each constraint i,
    either the constraint is tight (slack = 0) or the dual variable
    is zero: ``y_i * (b_i - a_i^T x) = 0``.

Usage::

    from dp_forge.dual_certificates import DualCertificateExtractor, GapTracker

    extractor = DualCertificateExtractor()
    cert = extractor.extract(lp_result, lp_struct)
    cs_report = extractor.verify_complementary_slackness(cert, lp_struct)
    gap = extractor.duality_gap(cert)

    tracker = GapTracker()
    for iteration in range(max_iter):
        ...
        tracker.record(iteration, gap)
    print(tracker.summary())
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .exceptions import (
    ConfigurationError,
    DPForgeError,
    NumericalInstabilityError,
    SolverError,
)
from .types import LPStruct, OptimalityCertificate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dual certificate dataclass
# ---------------------------------------------------------------------------


@dataclass
class DualCertificate:
    """Complete dual certificate for an LP solution.

    Contains dual variables, slack vectors, gap information, and
    complementary slackness diagnostics.

    Attributes:
        dual_ub: Dual variables for inequality constraints (λ ≥ 0).
        dual_eq: Dual variables for equality constraints (ν, free).
        primal_solution: Primal decision variable vector x*.
        primal_obj: Primal objective value c^T x*.
        dual_obj: Dual objective value b^T y*.
        duality_gap: Absolute duality gap |primal_obj − dual_obj|.
        primal_slack: Per-inequality slack s = b_ub − A_ub x.
        dual_slack: Reduced costs r = c − A_ub^T λ − A_eq^T ν.
        complementary_slackness_violation: Max |λ_i * s_i| over all i.
        n_vars: Number of primal variables.
        n_ub: Number of inequality constraints.
        n_eq: Number of equality constraints.
        solver_name: Name of the solver that produced the solution.
        timestamp: ISO-format timestamp of certificate generation.
    """

    dual_ub: npt.NDArray[np.float64]
    dual_eq: Optional[npt.NDArray[np.float64]]
    primal_solution: npt.NDArray[np.float64]
    primal_obj: float
    dual_obj: float
    duality_gap: float
    primal_slack: npt.NDArray[np.float64]
    dual_slack: npt.NDArray[np.float64]
    complementary_slackness_violation: float = 0.0
    n_vars: int = 0
    n_ub: int = 0
    n_eq: int = 0
    solver_name: str = "unknown"
    timestamp: str = ""

    def __post_init__(self) -> None:
        self.dual_ub = np.asarray(self.dual_ub, dtype=np.float64)
        self.primal_solution = np.asarray(self.primal_solution, dtype=np.float64)
        self.primal_slack = np.asarray(self.primal_slack, dtype=np.float64)
        self.dual_slack = np.asarray(self.dual_slack, dtype=np.float64)
        if self.dual_eq is not None:
            self.dual_eq = np.asarray(self.dual_eq, dtype=np.float64)
        if self.n_vars == 0:
            self.n_vars = len(self.primal_solution)
        if self.n_ub == 0:
            self.n_ub = len(self.dual_ub)
        if self.n_eq == 0 and self.dual_eq is not None:
            self.n_eq = len(self.dual_eq)
        if not self.timestamp:
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def relative_gap(self) -> float:
        """Relative duality gap: gap / max(|primal_obj|, 1)."""
        denom = max(abs(self.primal_obj), 1.0)
        return self.duality_gap / denom

    @property
    def is_optimal(self) -> bool:
        """Whether the gap is within numerical tolerance (1e-6)."""
        return self.relative_gap <= 1e-6

    @property
    def n_active_constraints(self) -> int:
        """Number of inequality constraints with near-zero slack."""
        return int(np.sum(np.abs(self.primal_slack) < 1e-8))

    @property
    def n_active_duals(self) -> int:
        """Number of dual variables that are non-negligible."""
        return int(np.sum(np.abs(self.dual_ub) > 1e-8))

    def __repr__(self) -> str:
        return (
            f"DualCertificate(gap={self.duality_gap:.2e}, "
            f"rel_gap={self.relative_gap:.2e}, "
            f"cs_viol={self.complementary_slackness_violation:.2e}, "
            f"vars={self.n_vars}, ub={self.n_ub})"
        )


@dataclass
class ComplementarySlacknessReport:
    """Detailed report on complementary slackness conditions.

    Attributes:
        max_violation: Maximum |λ_i * s_i| across all inequality constraints.
        mean_violation: Mean |λ_i * s_i|.
        n_violated: Number of constraints violating CS beyond tolerance.
        violated_indices: Indices of violated constraints.
        per_constraint_product: Array of |λ_i * s_i| for each constraint.
        tolerance: Tolerance used for violation check.
        satisfied: Whether all CS conditions are satisfied within tolerance.
    """

    max_violation: float
    mean_violation: float
    n_violated: int
    violated_indices: npt.NDArray[np.intp]
    per_constraint_product: npt.NDArray[np.float64]
    tolerance: float
    satisfied: bool

    def __repr__(self) -> str:
        status = "SATISFIED" if self.satisfied else "VIOLATED"
        return (
            f"ComplementarySlacknessReport({status}, "
            f"max_viol={self.max_violation:.2e}, "
            f"n_violated={self.n_violated}/{len(self.per_constraint_product)})"
        )


@dataclass
class OptimalityProof:
    """Human-readable optimality proof for a mechanism.

    Attributes:
        summary: One-line summary of optimality status.
        primal_feasibility: Description of primal feasibility check.
        dual_feasibility: Description of dual feasibility check.
        strong_duality: Description of strong duality / gap analysis.
        complementary_slackness: Description of CS conditions.
        conclusion: Final conclusion with error bound.
        certificate: The underlying dual certificate.
    """

    summary: str
    primal_feasibility: str
    dual_feasibility: str
    strong_duality: str
    complementary_slackness: str
    conclusion: str
    certificate: DualCertificate

    def format(self) -> str:
        """Format the proof as a multi-line string."""
        lines = [
            "=" * 72,
            "OPTIMALITY PROOF FOR DP MECHANISM",
            "=" * 72,
            "",
            f"Summary: {self.summary}",
            "",
            "1. PRIMAL FEASIBILITY",
            f"   {self.primal_feasibility}",
            "",
            "2. DUAL FEASIBILITY",
            f"   {self.dual_feasibility}",
            "",
            "3. STRONG DUALITY",
            f"   {self.strong_duality}",
            "",
            "4. COMPLEMENTARY SLACKNESS",
            f"   {self.complementary_slackness}",
            "",
            "CONCLUSION",
            f"   {self.conclusion}",
            "",
            f"Generated: {self.certificate.timestamp}",
            f"Solver: {self.certificate.solver_name}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"OptimalityProof({self.summary!r})"


# ---------------------------------------------------------------------------
# DualCertificateExtractor
# ---------------------------------------------------------------------------


class DualCertificateExtractor:
    """Extract and verify dual LP certificates from solver output.

    Extracts dual variables from scipy.optimize.linprog (HiGHS backend),
    computes primal/dual slack, verifies complementary slackness, and
    constructs human-readable optimality proofs.

    Args:
        tol: Tolerance for complementary slackness and feasibility checks.
        max_condition_number: Maximum condition number for numerical
            stability checks on the constraint matrix.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_condition_number: float = 1e12,
    ) -> None:
        if tol <= 0:
            raise ConfigurationError(
                "Tolerance must be positive",
                parameter="tol",
                value=tol,
                constraint="tol > 0",
            )
        self.tol = tol
        self.max_condition_number = max_condition_number

    def extract(
        self,
        lp_result: Any,
        lp_struct: LPStruct,
        *,
        solver_name: str = "HiGHS",
    ) -> DualCertificate:
        """Extract a dual certificate from an LP solver result.

        Parses the solver output to obtain primal and dual variable
        values, computes slack vectors, and packages everything into
        a :class:`DualCertificate`.

        Args:
            lp_result: Result object from scipy.optimize.linprog.
                Expected attributes: ``x``, ``fun``, ``ineqlin``,
                ``eqlin``.
            lp_struct: The LP structure that was solved.
            solver_name: Name of the solver for provenance.

        Returns:
            A fully populated :class:`DualCertificate`.

        Raises:
            SolverError: If primal or dual variables cannot be extracted.
            NumericalInstabilityError: If the constraint matrix condition
                number exceeds the threshold.
        """
        # Extract primal solution
        x = self._extract_primal_solution(lp_result)
        if len(x) != lp_struct.n_vars:
            raise SolverError(
                f"Primal solution length {len(x)} != expected {lp_struct.n_vars}",
                solver_name=solver_name,
            )

        # Extract primal objective
        primal_obj = self._extract_objective(lp_result)

        # Extract dual variables for inequality constraints
        dual_ub = self._extract_inequality_duals(lp_result, lp_struct.n_ub)

        # Extract dual variables for equality constraints
        dual_eq = self._extract_equality_duals(lp_result, lp_struct.n_eq)

        # Compute primal slack: s = b_ub - A_ub @ x
        A_ub_dense = (
            lp_struct.A_ub.toarray()
            if sparse.issparse(lp_struct.A_ub)
            else np.asarray(lp_struct.A_ub)
        )
        primal_slack = lp_struct.b_ub - A_ub_dense @ x

        # Compute dual slack (reduced costs): r = c - A_ub^T λ - A_eq^T ν
        dual_slack = lp_struct.c.copy()
        dual_slack = dual_slack - A_ub_dense.T @ dual_ub
        if lp_struct.A_eq is not None and dual_eq is not None:
            A_eq_dense = (
                lp_struct.A_eq.toarray()
                if sparse.issparse(lp_struct.A_eq)
                else np.asarray(lp_struct.A_eq)
            )
            dual_slack = dual_slack - A_eq_dense.T @ dual_eq

        # Compute dual objective: b_ub^T λ + b_eq^T ν
        dual_obj = float(lp_struct.b_ub @ dual_ub)
        if lp_struct.b_eq is not None and dual_eq is not None:
            dual_obj += float(lp_struct.b_eq @ dual_eq)

        duality_gap = abs(primal_obj - dual_obj)

        # Complementary slackness violation
        cs_products = np.abs(dual_ub * primal_slack)
        cs_violation = float(np.max(cs_products)) if len(cs_products) > 0 else 0.0

        # Condition number check
        self._check_condition_number(A_ub_dense, solver_name)

        cert = DualCertificate(
            dual_ub=dual_ub,
            dual_eq=dual_eq,
            primal_solution=x,
            primal_obj=primal_obj,
            dual_obj=dual_obj,
            duality_gap=duality_gap,
            primal_slack=primal_slack,
            dual_slack=dual_slack,
            complementary_slackness_violation=cs_violation,
            n_vars=lp_struct.n_vars,
            n_ub=lp_struct.n_ub,
            n_eq=lp_struct.n_eq,
            solver_name=solver_name,
        )

        logger.info(
            "Dual certificate extracted: gap=%.2e, rel_gap=%.2e, cs_viol=%.2e",
            cert.duality_gap,
            cert.relative_gap,
            cert.complementary_slackness_violation,
        )

        return cert

    def verify_complementary_slackness(
        self,
        cert: DualCertificate,
        lp_struct: Optional[LPStruct] = None,
        *,
        tolerance: Optional[float] = None,
    ) -> ComplementarySlacknessReport:
        """Verify complementary slackness conditions for a dual certificate.

        For each inequality constraint i, checks that
        ``|λ_i × s_i| ≤ tolerance`` where λ_i is the dual variable
        and s_i is the primal slack.

        Additionally checks dual CS: for each variable j,
        ``|x_j × r_j| ≤ tolerance`` where r_j is the reduced cost.

        Args:
            cert: The dual certificate to verify.
            lp_struct: Optional LP struct for additional context.
            tolerance: Override the default tolerance.

        Returns:
            A :class:`ComplementarySlacknessReport` with detailed diagnostics.
        """
        tol = tolerance if tolerance is not None else self.tol

        # Primal CS: λ_i * s_i = 0
        cs_products = np.abs(cert.dual_ub * cert.primal_slack)
        violated_mask = cs_products > tol
        violated_indices = np.where(violated_mask)[0]

        max_violation = float(np.max(cs_products)) if len(cs_products) > 0 else 0.0
        mean_violation = float(np.mean(cs_products)) if len(cs_products) > 0 else 0.0

        # Dual CS: x_j * r_j = 0
        dual_cs_products = np.abs(cert.primal_solution * cert.dual_slack)
        dual_violated = np.where(dual_cs_products > tol)[0]

        n_violated = len(violated_indices) + len(dual_violated)

        # Combine for report (report primal CS products as primary)
        satisfied = n_violated == 0

        report = ComplementarySlacknessReport(
            max_violation=max(max_violation, float(np.max(dual_cs_products)) if len(dual_cs_products) > 0 else 0.0),
            mean_violation=mean_violation,
            n_violated=n_violated,
            violated_indices=violated_indices,
            per_constraint_product=cs_products,
            tolerance=tol,
            satisfied=satisfied,
        )

        logger.info(
            "CS verification: %s, max_viol=%.2e, n_violated=%d/%d",
            "PASS" if satisfied else "FAIL",
            report.max_violation,
            n_violated,
            len(cs_products) + len(dual_cs_products),
        )

        return report

    def duality_gap(
        self,
        cert: DualCertificate,
        *,
        rigorous: bool = True,
    ) -> Dict[str, float]:
        """Compute duality gap with rigorous bounds.

        Returns absolute gap, relative gap, and (if ``rigorous=True``)
        a certified upper bound on sub-optimality that accounts for
        numerical precision of the solver.

        The rigorous bound adds a term proportional to the solver
        tolerance and the problem scale to the raw gap.

        Args:
            cert: The dual certificate.
            rigorous: Whether to compute the rigorous bound including
                numerical precision adjustment.

        Returns:
            Dict with keys:
            - ``"absolute_gap"``: |primal_obj − dual_obj|
            - ``"relative_gap"``: gap / max(|primal_obj|, 1)
            - ``"certified_bound"``: Rigorous upper bound on sub-optimality
            - ``"cs_violation"``: Complementary slackness violation
        """
        absolute_gap = cert.duality_gap
        relative_gap = cert.relative_gap

        if rigorous:
            # Add numerical precision buffer: solver_tol × problem_scale
            problem_scale = max(
                float(np.max(np.abs(cert.primal_solution))) if len(cert.primal_solution) > 0 else 1.0,
                abs(cert.primal_obj),
                1.0,
            )
            precision_buffer = self.tol * problem_scale * math.sqrt(cert.n_vars)
            certified_bound = absolute_gap + precision_buffer
        else:
            certified_bound = absolute_gap

        return {
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
            "certified_bound": certified_bound,
            "cs_violation": cert.complementary_slackness_violation,
        }

    def construct_proof(
        self,
        cert: DualCertificate,
        lp_struct: LPStruct,
    ) -> OptimalityProof:
        """Construct a human-readable optimality proof.

        Checks primal feasibility, dual feasibility, strong duality,
        and complementary slackness, then assembles a structured proof.

        Args:
            cert: The dual certificate to verify.
            lp_struct: The LP structure for feasibility checks.

        Returns:
            An :class:`OptimalityProof` with all verification details.
        """
        # 1. Primal feasibility: A_ub x <= b_ub
        max_primal_infeas = float(np.max(-cert.primal_slack))
        primal_feasible = max_primal_infeas <= self.tol
        primal_msg = (
            f"All {cert.n_ub} inequality constraints satisfied "
            f"(max infeasibility: {max_primal_infeas:.2e})."
            if primal_feasible
            else f"INFEASIBLE: {int(np.sum(-cert.primal_slack > self.tol))} constraints "
            f"violated (max: {max_primal_infeas:.2e})."
        )

        # Check equality constraints
        if lp_struct.A_eq is not None and lp_struct.b_eq is not None:
            A_eq_dense = (
                lp_struct.A_eq.toarray()
                if sparse.issparse(lp_struct.A_eq)
                else np.asarray(lp_struct.A_eq)
            )
            eq_residual = A_eq_dense @ cert.primal_solution - lp_struct.b_eq
            max_eq_infeas = float(np.max(np.abs(eq_residual)))
            if max_eq_infeas > self.tol:
                primal_msg += (
                    f" Equality violation: {max_eq_infeas:.2e}."
                )
                primal_feasible = False

        # 2. Dual feasibility: λ >= 0, reduced costs
        min_dual = float(np.min(cert.dual_ub)) if len(cert.dual_ub) > 0 else 0.0
        dual_feasible = min_dual >= -self.tol
        # Check reduced cost non-negativity for bounded variables
        n_negative_rc = int(np.sum(cert.dual_slack < -self.tol))
        dual_msg = (
            f"All {cert.n_ub} dual variables non-negative "
            f"(min: {min_dual:.2e}). "
            f"{n_negative_rc} reduced costs negative beyond tolerance."
        )
        if not dual_feasible:
            dual_msg = (
                f"INFEASIBLE: {int(np.sum(cert.dual_ub < -self.tol))} "
                f"dual variables negative (min: {min_dual:.2e})."
            )

        # 3. Strong duality
        gap_info = self.duality_gap(cert, rigorous=True)
        duality_msg = (
            f"Primal obj = {cert.primal_obj:.8f}, "
            f"Dual obj = {cert.dual_obj:.8f}. "
            f"Absolute gap = {gap_info['absolute_gap']:.2e}, "
            f"Relative gap = {gap_info['relative_gap']:.2e}. "
            f"Certified bound on sub-optimality: {gap_info['certified_bound']:.2e}."
        )

        # 4. Complementary slackness
        cs_report = self.verify_complementary_slackness(cert)
        cs_msg = (
            f"{'SATISFIED' if cs_report.satisfied else 'VIOLATED'}: "
            f"max |λ·s| = {cs_report.max_violation:.2e}, "
            f"{cs_report.n_violated} violations beyond tol={cs_report.tolerance:.0e}."
        )

        # Conclusion
        all_pass = primal_feasible and dual_feasible and cert.relative_gap <= self.tol
        if all_pass:
            conclusion = (
                f"The mechanism is OPTIMAL within tolerance {self.tol:.0e}. "
                f"The objective value {cert.primal_obj:.8f} is within "
                f"{gap_info['certified_bound']:.2e} of the true optimum."
            )
            summary = f"Optimal (gap={cert.relative_gap:.2e})"
        elif primal_feasible and dual_feasible:
            conclusion = (
                f"The mechanism is NEAR-OPTIMAL. "
                f"The objective value {cert.primal_obj:.8f} is within "
                f"{gap_info['certified_bound']:.2e} of the true optimum."
            )
            summary = f"Near-optimal (gap={cert.relative_gap:.2e})"
        else:
            issues = []
            if not primal_feasible:
                issues.append("primal infeasible")
            if not dual_feasible:
                issues.append("dual infeasible")
            conclusion = (
                f"Optimality CANNOT be certified: {', '.join(issues)}."
            )
            summary = f"Uncertified ({', '.join(issues)})"

        return OptimalityProof(
            summary=summary,
            primal_feasibility=primal_msg,
            dual_feasibility=dual_msg,
            strong_duality=duality_msg,
            complementary_slackness=cs_msg,
            conclusion=conclusion,
            certificate=cert,
        )

    def to_optimality_certificate(self, cert: DualCertificate) -> OptimalityCertificate:
        """Convert a DualCertificate to the standard OptimalityCertificate type.

        Args:
            cert: The dual certificate to convert.

        Returns:
            An OptimalityCertificate compatible with the rest of the pipeline.
        """
        return OptimalityCertificate(
            dual_vars={
                "dual_ub": cert.dual_ub,
                "dual_eq": cert.dual_eq,
            },
            duality_gap=cert.duality_gap,
            primal_obj=cert.primal_obj,
            dual_obj=cert.dual_obj,
        )

    # --- Internal helpers ---

    def _extract_primal_solution(self, result: Any) -> npt.NDArray[np.float64]:
        """Extract primal variable vector from solver result."""
        if hasattr(result, "x") and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)
        if isinstance(result, dict) and "x" in result:
            return np.asarray(result["x"], dtype=np.float64)
        raise SolverError("Cannot extract primal solution from LP result")

    def _extract_objective(self, result: Any) -> float:
        """Extract primal objective value from solver result."""
        if hasattr(result, "fun"):
            return float(result.fun)
        if hasattr(result, "objective_value"):
            return float(result.objective_value)
        if isinstance(result, dict):
            for key in ("fun", "objective_value"):
                if key in result:
                    return float(result[key])
        raise SolverError("Cannot extract objective value from LP result")

    def _extract_inequality_duals(
        self, result: Any, n_ub: int
    ) -> npt.NDArray[np.float64]:
        """Extract dual variables for inequality constraints."""
        # scipy.optimize.linprog with HiGHS
        if hasattr(result, "ineqlin"):
            marginals = getattr(result.ineqlin, "marginals", None)
            if marginals is not None:
                return np.asarray(marginals, dtype=np.float64)
        if hasattr(result, "dual_ub"):
            return np.asarray(result.dual_ub, dtype=np.float64)
        if isinstance(result, dict):
            for key in ("dual_ub", "lambda", "y"):
                if key in result:
                    return np.asarray(result[key], dtype=np.float64)
        # Fall back to zeros
        logger.warning("No inequality duals found; using zeros (n_ub=%d)", n_ub)
        return np.zeros(n_ub, dtype=np.float64)

    def _extract_equality_duals(
        self, result: Any, n_eq: int
    ) -> Optional[npt.NDArray[np.float64]]:
        """Extract dual variables for equality constraints."""
        if n_eq == 0:
            return None
        if hasattr(result, "eqlin"):
            marginals = getattr(result.eqlin, "marginals", None)
            if marginals is not None:
                return np.asarray(marginals, dtype=np.float64)
        if hasattr(result, "dual_eq"):
            return np.asarray(result.dual_eq, dtype=np.float64)
        if isinstance(result, dict) and "dual_eq" in result:
            return np.asarray(result["dual_eq"], dtype=np.float64)
        logger.warning("No equality duals found; using zeros (n_eq=%d)", n_eq)
        return np.zeros(n_eq, dtype=np.float64)

    def _check_condition_number(
        self, A: npt.NDArray[np.float64], solver_name: str
    ) -> None:
        """Check constraint matrix condition number for numerical stability."""
        if A.shape[0] == 0 or A.shape[1] == 0:
            return
        # Use a fast estimate for large matrices
        if min(A.shape) > 500:
            try:
                s = np.linalg.svd(A, compute_uv=False)
                if s[-1] > 0:
                    cond = s[0] / s[-1]
                else:
                    cond = float("inf")
            except np.linalg.LinAlgError:
                return
        else:
            try:
                cond = float(np.linalg.cond(A))
            except np.linalg.LinAlgError:
                return

        if cond > self.max_condition_number:
            raise NumericalInstabilityError(
                f"Constraint matrix condition number {cond:.2e} exceeds "
                f"threshold {self.max_condition_number:.2e}",
                condition_number=cond,
                max_condition_number=self.max_condition_number,
                matrix_name="A_ub",
            )
        elif cond > self.max_condition_number * 0.1:
            logger.warning(
                "Constraint matrix condition number %.2e approaching "
                "threshold %.2e",
                cond,
                self.max_condition_number,
            )


# ---------------------------------------------------------------------------
# GapTracker — convergence monitoring across CEGIS iterations
# ---------------------------------------------------------------------------


@dataclass
class GapRecord:
    """Record of a duality gap at a particular CEGIS iteration.

    Attributes:
        iteration: CEGIS iteration number.
        absolute_gap: Absolute duality gap.
        relative_gap: Relative duality gap.
        primal_obj: Primal objective at this iteration.
        dual_obj: Dual objective at this iteration.
        n_constraints: Number of LP constraints at this iteration.
    """

    iteration: int
    absolute_gap: float
    relative_gap: float
    primal_obj: float
    dual_obj: float
    n_constraints: int = 0


class GapTracker:
    """Track duality gap convergence across CEGIS iterations.

    Implements the convergence analysis from Theorem T14: in each CEGIS
    iteration the duality gap decreases at least linearly as new
    counterexamples tighten the LP relaxation.

    The tracker records gap values at each iteration, estimates the
    convergence rate, and predicts how many additional iterations are
    needed to reach a target gap.

    Args:
        target_gap: Target relative gap for convergence.
        max_history: Maximum number of records to retain.

    Usage::

        tracker = GapTracker(target_gap=1e-6)
        for it in range(max_iter):
            cert = extractor.extract(result, lp_struct)
            tracker.record(it, cert.duality_gap, cert.relative_gap,
                           cert.primal_obj, cert.dual_obj)
            if tracker.converged:
                break
        print(tracker.summary())
    """

    def __init__(
        self,
        target_gap: float = 1e-6,
        max_history: int = 1000,
    ) -> None:
        if target_gap <= 0:
            raise ConfigurationError(
                "target_gap must be positive",
                parameter="target_gap",
                value=target_gap,
                constraint="target_gap > 0",
            )
        self.target_gap = target_gap
        self.max_history = max_history
        self._records: List[GapRecord] = []

    def record(
        self,
        iteration: int,
        absolute_gap: float,
        relative_gap: float = 0.0,
        primal_obj: float = 0.0,
        dual_obj: float = 0.0,
        n_constraints: int = 0,
    ) -> None:
        """Record a gap measurement at a CEGIS iteration.

        Args:
            iteration: CEGIS iteration number.
            absolute_gap: Absolute duality gap.
            relative_gap: Relative duality gap. If 0, computed from absolute.
            primal_obj: Primal objective value.
            dual_obj: Dual objective value.
            n_constraints: Number of LP constraints.
        """
        if relative_gap == 0.0 and primal_obj != 0.0:
            relative_gap = absolute_gap / max(abs(primal_obj), 1.0)

        rec = GapRecord(
            iteration=iteration,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            primal_obj=primal_obj,
            dual_obj=dual_obj,
            n_constraints=n_constraints,
        )
        self._records.append(rec)

        # Trim to max_history
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]

        logger.debug(
            "Gap tracker [iter %d]: abs=%.2e, rel=%.2e",
            iteration, absolute_gap, relative_gap,
        )

    def record_from_certificate(
        self,
        iteration: int,
        cert: DualCertificate,
        n_constraints: int = 0,
    ) -> None:
        """Record a gap from a DualCertificate.

        Args:
            iteration: CEGIS iteration number.
            cert: The dual certificate for this iteration.
            n_constraints: Number of LP constraints.
        """
        self.record(
            iteration=iteration,
            absolute_gap=cert.duality_gap,
            relative_gap=cert.relative_gap,
            primal_obj=cert.primal_obj,
            dual_obj=cert.dual_obj,
            n_constraints=n_constraints or cert.n_ub,
        )

    @property
    def converged(self) -> bool:
        """Whether the latest relative gap is below the target."""
        if not self._records:
            return False
        return self._records[-1].relative_gap <= self.target_gap

    @property
    def n_records(self) -> int:
        """Number of recorded gap measurements."""
        return len(self._records)

    @property
    def latest_gap(self) -> float:
        """Latest absolute gap, or inf if no records."""
        if not self._records:
            return float("inf")
        return self._records[-1].absolute_gap

    @property
    def latest_relative_gap(self) -> float:
        """Latest relative gap, or inf if no records."""
        if not self._records:
            return float("inf")
        return self._records[-1].relative_gap

    @property
    def gaps(self) -> npt.NDArray[np.float64]:
        """Array of absolute gaps across all recorded iterations."""
        return np.array([r.absolute_gap for r in self._records], dtype=np.float64)

    @property
    def relative_gaps(self) -> npt.NDArray[np.float64]:
        """Array of relative gaps across all recorded iterations."""
        return np.array([r.relative_gap for r in self._records], dtype=np.float64)

    def convergence_rate(self) -> Optional[float]:
        """Estimate the linear convergence rate from gap history.

        Fits a linear model to log(gap) vs iteration to estimate
        the convergence rate ρ such that gap_t ≈ gap_0 · ρ^t.

        Following Theorem T14, in the worst case ρ ≤ 1 - 1/N where
        N is the total number of adjacency pairs.

        Returns:
            Estimated convergence rate ρ ∈ (0, 1), or ``None`` if
            insufficient data or gaps are non-decreasing.
        """
        if len(self._records) < 3:
            return None

        gaps = self.gaps
        # Filter to positive gaps
        positive = gaps > 0
        if np.sum(positive) < 3:
            return None

        log_gaps = np.log(gaps[positive])
        iters = np.array([r.iteration for r in self._records], dtype=np.float64)
        iters = iters[positive]

        # Linear regression on log(gap) vs iteration
        if len(iters) < 2:
            return None

        n = len(iters)
        mean_t = np.mean(iters)
        mean_lg = np.mean(log_gaps)
        cov_t_lg = np.sum((iters - mean_t) * (log_gaps - mean_lg))
        var_t = np.sum((iters - mean_t) ** 2)

        if var_t < 1e-15:
            return None

        slope = cov_t_lg / var_t
        # ρ = exp(slope) — slope should be negative for convergence
        rate = math.exp(slope)

        if rate >= 1.0:
            return None  # Not converging

        return rate

    def predicted_iterations_remaining(self) -> Optional[int]:
        """Predict how many more iterations to reach the target gap.

        Uses the estimated convergence rate to extrapolate.

        Returns:
            Estimated number of remaining iterations, or ``None``
            if the rate cannot be estimated.
        """
        rate = self.convergence_rate()
        if rate is None or rate >= 1.0:
            return None

        current_gap = self.latest_relative_gap
        if current_gap <= self.target_gap:
            return 0
        if current_gap <= 0:
            return None

        # gap_t = current_gap * rate^t <= target_gap
        # t >= log(target_gap / current_gap) / log(rate)
        try:
            t = math.log(self.target_gap / current_gap) / math.log(rate)
            return max(1, int(math.ceil(t)))
        except (ValueError, ZeroDivisionError):
            return None

    def gap_decrease_per_iteration(self) -> Optional[float]:
        """Average absolute gap decrease per iteration.

        Returns:
            Average decrease, or ``None`` if insufficient data.
        """
        if len(self._records) < 2:
            return None

        first = self._records[0]
        last = self._records[-1]
        n_iters = last.iteration - first.iteration
        if n_iters <= 0:
            return None

        return (first.absolute_gap - last.absolute_gap) / n_iters

    def is_stalled(self, window: int = 5, min_improvement: float = 1e-10) -> bool:
        """Check whether convergence has stalled.

        Examines the last ``window`` records and returns True if the
        gap improvement is below ``min_improvement``.

        Args:
            window: Number of recent records to examine.
            min_improvement: Minimum improvement threshold.

        Returns:
            True if convergence appears stalled.
        """
        if len(self._records) < window:
            return False

        recent = self._records[-window:]
        gap_range = max(r.absolute_gap for r in recent) - min(r.absolute_gap for r in recent)
        return gap_range < min_improvement

    def summary(self) -> str:
        """Generate a human-readable convergence summary.

        Returns:
            Multi-line summary string.
        """
        if not self._records:
            return "GapTracker: No records."

        lines = [
            f"GapTracker Summary ({self.n_records} iterations recorded)",
            f"  Target relative gap: {self.target_gap:.2e}",
            f"  Initial gap: {self._records[0].absolute_gap:.2e} "
            f"(rel: {self._records[0].relative_gap:.2e})",
            f"  Current gap: {self.latest_gap:.2e} "
            f"(rel: {self.latest_relative_gap:.2e})",
            f"  Converged: {self.converged}",
        ]

        rate = self.convergence_rate()
        if rate is not None:
            lines.append(f"  Convergence rate ρ: {rate:.4f}")
            remaining = self.predicted_iterations_remaining()
            if remaining is not None:
                lines.append(f"  Predicted remaining: {remaining} iterations")

        avg_decrease = self.gap_decrease_per_iteration()
        if avg_decrease is not None:
            lines.append(f"  Avg gap decrease/iter: {avg_decrease:.2e}")

        if self.is_stalled():
            lines.append("  ⚠ Convergence appears STALLED")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GapTracker(n={self.n_records}, target={self.target_gap:.2e}, "
            f"latest={self.latest_gap:.2e}, converged={self.converged})"
        )
