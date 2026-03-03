"""
LP perturbation theory for bounding privacy loss under numerical errors.

Given an LP solution p* and a perturbation Δp (due to solver tolerance),
this module bounds the resulting change in privacy parameters:
    Δε ≤ ||Δp||₁ · condition_factor

The analysis uses:
    - Condition number estimation of the LP basis matrix.
    - Sensitivity analysis via LP duality: how the optimal objective
      changes when ε is perturbed.
    - Dual norm estimation for the feasibility region boundary.

Key class:
    - :class:`PerturbationAnalyzer` — Main analysis engine.

Theory:
    For a mechanism p* satisfying e^ε-DP, a perturbation Δp with
    ||Δp||_∞ ≤ ν introduces a privacy loss bounded by:

        effective_ε ≤ ε + max_{(i,i'),j} log((p*[i][j] + ν) / (p*[i'][j] - ν))
                    ≤ ε + log(1 + 2ν / (p_min - ν))

    where p_min is the minimum probability in the mechanism.  For small ν,
    this is ≈ ε + 2ν / p_min.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from dp_forge.exceptions import NumericalInstabilityError
from dp_forge.types import LPStruct

logger = logging.getLogger(__name__)


@dataclass
class PerturbationBound:
    """Result of perturbation analysis.

    Attributes:
        epsilon_bound: Upper bound on |Δε| due to perturbation.
        delta_bound: Upper bound on |Δδ| due to perturbation.
        condition_number: Estimated condition number of the LP basis.
        sensitivity_coefficient: dε*/dε coefficient from LP sensitivity.
        p_min: Minimum probability observed in the mechanism.
        solver_tolerance: Solver tolerance used in the analysis.
        is_well_conditioned: Whether the LP is numerically well-conditioned.
    """

    epsilon_bound: float
    delta_bound: float
    condition_number: float
    sensitivity_coefficient: float
    p_min: float
    solver_tolerance: float
    is_well_conditioned: bool

    def __repr__(self) -> str:
        status = "well-cond" if self.is_well_conditioned else "ILL-COND"
        return (
            f"PerturbationBound(Δε≤{self.epsilon_bound:.2e}, "
            f"Δδ≤{self.delta_bound:.2e}, "
            f"κ={self.condition_number:.2e}, {status})"
        )


@dataclass
class SensitivityResult:
    """Result of LP sensitivity analysis.

    Attributes:
        objective_sensitivity: dobj*/dε — rate of change of optimal
            objective with respect to ε.
        binding_constraints: Indices of binding (active) inequality constraints.
        dual_values: Dual variable values for inequality constraints.
        shadow_prices: Shadow prices (absolute dual values) for DP constraints.
    """

    objective_sensitivity: float
    binding_constraints: List[int]
    dual_values: Optional[npt.NDArray[np.float64]]
    shadow_prices: npt.NDArray[np.float64]

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(dobj/dε={self.objective_sensitivity:.4e}, "
            f"n_binding={len(self.binding_constraints)})"
        )


class PerturbationAnalyzer:
    """Analyse LP solution sensitivity to numerical perturbations.

    Given a mechanism probability table and the LP structure, this
    class bounds the privacy loss due to solver imprecision and
    estimates the condition number of the LP basis.

    Args:
        max_condition_number: Threshold above which the LP is considered
            ill-conditioned. Default 1e12.

    Example::

        analyzer = PerturbationAnalyzer()
        bound = analyzer.bound_epsilon_change(
            p_solution, solver_tol=1e-8, epsilon=1.0, edges=[(0,1)])
    """

    def __init__(
        self,
        max_condition_number: float = 1e12,
    ) -> None:
        if max_condition_number <= 0:
            raise ValueError(
                f"max_condition_number must be > 0, got {max_condition_number}"
            )
        self._max_cond = max_condition_number

    def bound_epsilon_change(
        self,
        p_solution: npt.NDArray[np.float64],
        solver_tol: float,
        epsilon: float,
        edges: Sequence[Tuple[int, int]],
        delta: float = 0.0,
    ) -> PerturbationBound:
        """Bound the change in privacy parameters due to solver tolerance.

        For pure DP: computes the worst-case ε increase when every
        probability value p[i][j] could be off by ±solver_tol.

        For approx DP: also bounds the increase in hockey-stick divergence.

        Args:
            p_solution: Mechanism probability table, shape (n, k).
            solver_tol: Solver feasibility tolerance ν.
            epsilon: Target privacy parameter ε.
            edges: Adjacent database pairs.
            delta: Privacy parameter δ.

        Returns:
            PerturbationBound with conservative error bounds.
        """
        p = np.asarray(p_solution, dtype=np.float64)
        n, k = p.shape
        nu = solver_tol

        # Minimum probability in the mechanism
        p_min = float(np.min(p[p > 0])) if np.any(p > 0) else nu

        # Guard against p_min being too close to ν
        if p_min <= nu:
            # In this regime, the perturbation bound diverges.
            # We use a conservative bound based on p_min = ν.
            p_min_effective = nu
            logger.warning(
                "p_min (%.2e) ≤ solver_tol (%.2e): perturbation bound "
                "may be loose",
                p_min, nu,
            )
        else:
            p_min_effective = p_min

        # Pure DP epsilon bound
        # Worst case: numerator increases by ν, denominator decreases by ν
        # ratio ≤ (p[i][j] + ν) / (p[i'][j] - ν)
        # Δε ≤ log((e^ε + ν/p_min) / (1 - ν/p_min)) - ε  (if p_min > ν)
        # Simplified: Δε ≤ log(1 + 2ν / (p_min - ν))
        if p_min_effective > nu:
            eps_bound = math.log1p(2.0 * nu / (p_min_effective - nu))
        else:
            # Degenerate case: use a large but finite bound
            eps_bound = math.log1p(2.0 * nu / max(p_min_effective, 1e-15))

        # Approximate DP delta bound
        # Hockey-stick divergence perturbation:
        # Each term max(p[i][j] - e^ε * p[i'][j], 0) can increase by
        # at most ν + e^ε * ν = ν(1 + e^ε).
        # The sum of k terms can increase by at most k * ν * (1 + e^ε).
        exp_eps = math.exp(epsilon)
        if delta > 0.0:
            delta_bound = k * nu * (1.0 + exp_eps)
        else:
            delta_bound = 0.0

        # Condition number estimation (heuristic based on probability range)
        p_max = float(np.max(p)) if p.size > 0 else 1.0
        cond_est = p_max / max(p_min, 1e-300)

        is_well_cond = cond_est < self._max_cond

        # Sensitivity coefficient: in the LP, ε appears in the constraint
        # coefficients as e^ε.  The sensitivity of the optimal objective
        # to ε is approximately: dobj*/dε ≈ -sum(λ_j · dp_coeff/dε)
        # where λ_j are the dual variables.  Without dual values, we
        # estimate it as the number of binding ratio constraints × e^ε.
        sensitivity_coeff = exp_eps

        bound = PerturbationBound(
            epsilon_bound=eps_bound,
            delta_bound=delta_bound,
            condition_number=cond_est,
            sensitivity_coefficient=sensitivity_coeff,
            p_min=p_min,
            solver_tolerance=nu,
            is_well_conditioned=is_well_cond,
        )

        if not is_well_cond:
            logger.warning(
                "LP is ill-conditioned (κ=%.2e > %.2e): solution may be "
                "numerically unreliable",
                cond_est, self._max_cond,
            )

        return bound

    def estimate_condition(
        self,
        lp: LPStruct,
        basis_indices: Optional[npt.NDArray[np.int64]] = None,
    ) -> float:
        """Estimate the condition number of the LP basis matrix.

        If basis_indices are provided, extracts the corresponding columns
        of A_ub to form the basis matrix and computes its condition number.
        Otherwise, estimates the condition number from the full constraint
        matrix using sparse SVD.

        Args:
            lp: LP structure.
            basis_indices: Column indices of the LP basis variables.

        Returns:
            Estimated condition number κ(B).

        Raises:
            NumericalInstabilityError: If condition number exceeds threshold.
        """
        A = lp.A_ub

        if basis_indices is not None and len(basis_indices) > 0:
            # Extract basis columns
            A_dense = A.toarray()
            n_basis = min(len(basis_indices), A_dense.shape[0])
            basis_cols = basis_indices[:n_basis]
            valid_cols = basis_cols[basis_cols < A_dense.shape[1]]
            if len(valid_cols) == 0:
                return 1.0
            B = A_dense[:, valid_cols]
            # Compute condition number of the (possibly rectangular) basis
            try:
                cond = float(np.linalg.cond(B))
            except np.linalg.LinAlgError:
                cond = float("inf")
        else:
            # Sparse SVD estimate: κ ≈ σ_max / σ_min
            try:
                m, n_cols = A.shape
                k_svd = min(min(m, n_cols) - 1, 6)
                if k_svd < 1:
                    return 1.0
                sv_large = sp_linalg.svds(A.tocsc(), k=1, which="LM", return_singular_vectors=False)
                sv_small = sp_linalg.svds(A.tocsc(), k=1, which="SM", return_singular_vectors=False)
                sigma_max = float(sv_large[0])
                sigma_min = float(sv_small[0])
                if sigma_min < 1e-300:
                    cond = float("inf")
                else:
                    cond = sigma_max / sigma_min
            except Exception:
                logger.debug("Sparse SVD failed for condition estimation; using dense fallback")
                try:
                    A_dense = A.toarray()
                    cond = float(np.linalg.cond(A_dense))
                except Exception:
                    cond = float("inf")

        if cond > self._max_cond:
            raise NumericalInstabilityError(
                f"LP basis condition number {cond:.2e} exceeds threshold "
                f"{self._max_cond:.2e}",
                condition_number=cond,
                max_condition_number=self._max_cond,
                matrix_name="LP_basis",
            )

        return cond

    def sensitivity_analysis(
        self,
        p_solution: npt.NDArray[np.float64],
        lp: LPStruct,
        epsilon: float,
        edges: Sequence[Tuple[int, int]],
        dual_values: Optional[npt.NDArray[np.float64]] = None,
    ) -> SensitivityResult:
        """Perform LP sensitivity analysis on privacy constraints.

        Identifies binding constraints and computes shadow prices to
        understand how the optimal mechanism changes as ε varies.

        Args:
            p_solution: Optimal mechanism table, shape (n, k).
            lp: LP structure used to produce p_solution.
            epsilon: Privacy parameter ε.
            edges: Adjacent database pairs.
            dual_values: Dual variable values from the LP solve, if available.

        Returns:
            SensitivityResult with binding constraints and shadow prices.
        """
        p = np.asarray(p_solution, dtype=np.float64)
        n, k = p.shape
        exp_eps = math.exp(epsilon)
        binding_tol = 1e-7

        # Identify binding ratio constraints
        binding = []
        constraint_idx = 0
        for i, ip in edges:
            for direction in [(i, ip), (ip, i)]:
                row_a, row_b = direction
                for j in range(k):
                    residual = exp_eps * p[row_b, j] - p[row_a, j]
                    if abs(residual) < binding_tol:
                        binding.append(constraint_idx)
                    constraint_idx += 1

        # Shadow prices
        if dual_values is not None and len(dual_values) > 0:
            shadow = np.abs(dual_values)
        else:
            shadow = np.zeros(max(constraint_idx, 1), dtype=np.float64)

        # Objective sensitivity: dobj*/dε ≈ -Σ_{binding} λ_j · d(e^ε)/dε
        # = -Σ_{binding} λ_j · e^ε
        if dual_values is not None and len(binding) > 0:
            valid_binding = [b for b in binding if b < len(dual_values)]
            obj_sens = -exp_eps * float(np.sum(dual_values[valid_binding]))
        else:
            obj_sens = 0.0

        return SensitivityResult(
            objective_sensitivity=obj_sens,
            binding_constraints=binding,
            dual_values=dual_values,
            shadow_prices=shadow,
        )

    def estimate_dual_norm(
        self,
        lp: LPStruct,
        dual_values: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """Estimate the dual norm of the feasibility region boundary.

        The dual norm ||y||_* gives a measure of how "tight" the feasible
        region is at the optimal solution.  A large dual norm means small
        perturbations in the constraints can cause large changes in the
        optimal solution.

        Args:
            lp: LP structure.
            dual_values: Dual variables from the LP solve.

        Returns:
            Estimated dual norm (L1 norm of dual variables, or row-norm
            estimate if duals unavailable).
        """
        if dual_values is not None and len(dual_values) > 0:
            return float(np.sum(np.abs(dual_values)))

        # Fallback: estimate from constraint matrix structure
        A = lp.A_ub.tocsr()
        row_norms = np.array([
            float(np.sum(np.abs(A.getrow(i).toarray())))
            for i in range(A.shape[0])
        ])
        if len(row_norms) == 0:
            return 0.0
        return float(np.max(row_norms))

    def __repr__(self) -> str:
        return f"PerturbationAnalyzer(max_cond={self._max_cond:.2e})"
