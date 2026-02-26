"""
Bound Extractor
================

Extracts worst-case bounds from the LP solution, computes expectation,
probability, and quantile bounds, and performs sensitivity analysis.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoundResult:
    """Holds a computed bound and its certificate."""
    lower: float
    upper: float
    bound_type: str   # "probability", "expectation", "quantile"
    query_description: str = ""
    dual_certificate_lower: Optional[np.ndarray] = None
    dual_certificate_upper: Optional[np.ndarray] = None
    tightness: float = 0.0  # 0 = loose, 1 = point-identified

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        return self.lower - 1e-10 <= value <= self.upper + 1e-10


@dataclass
class SensitivityEntry:
    """Sensitivity of bounds to a constraint perturbation."""
    constraint_name: str
    constraint_index: int
    delta_lower: float
    delta_upper: float
    dual_value_lower: float
    dual_value_upper: float


@dataclass
class SensitivityReport:
    """Full sensitivity analysis report."""
    entries: List[SensitivityEntry]
    perturbation_size: float
    base_lower: float
    base_upper: float

    def most_sensitive_constraints(self, top_k: int = 5) -> List[SensitivityEntry]:
        """Return constraints with largest bound sensitivity."""
        scored = sorted(
            self.entries,
            key=lambda e: abs(e.delta_lower) + abs(e.delta_upper),
            reverse=True,
        )
        return scored[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_size": self.perturbation_size,
            "base_bounds": [self.base_lower, self.base_upper],
            "entries": [
                {
                    "name": e.constraint_name,
                    "index": e.constraint_index,
                    "delta_lower": e.delta_lower,
                    "delta_upper": e.delta_upper,
                    "dual_lower": e.dual_value_lower,
                    "dual_upper": e.dual_value_upper,
                }
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
#  Bound extractor
# ---------------------------------------------------------------------------

class BoundExtractor:
    """
    Extracts various worst-case bounds from the causal polytope LP.

    Given the polytope defined by A x = b, x >= 0, this class optimises
    different linear objectives over it to obtain bounds on
    probabilities, expectations, and quantiles.

    Parameters
    ----------
    dag : DAGSpec
        The (possibly mutilated) DAG.
    A_eq : sparse matrix
        Equality constraints.
    b_eq : ndarray
        RHS.
    """

    def __init__(self, dag, A_eq: sparse.spmatrix, b_eq: np.ndarray):
        self.dag = dag
        self.A_eq = A_eq
        self.b_eq = b_eq
        self._topo = dag.topological_order()
        self._strides = self._compute_strides()
        self._total_vars = self._compute_total_vars()

    # ------------------------------------------------------------------
    #  Probability bounds
    # ------------------------------------------------------------------

    def probability_bounds(
        self,
        target_var: str,
        target_val: int,
        intervention: Optional[Dict[str, int]] = None,
    ) -> BoundResult:
        """
        Compute bounds on P(target_var = target_val | do(intervention)).

        Parameters
        ----------
        target_var : str
        target_val : int
        intervention : dict, optional
            {var: val} pairs for the do-operator.

        Returns
        -------
        BoundResult
        """
        c = self._build_probability_objective(target_var, target_val, intervention)

        lb, dual_lb = self._solve_lp(c, sense="min")
        ub, dual_ub = self._solve_lp(c, sense="max")

        lb = max(0.0, min(1.0, lb))
        ub = max(0.0, min(1.0, ub))
        if lb > ub:
            lb, ub = ub, lb

        tightness = 1.0 - (ub - lb) if ub > lb else 1.0

        desc = f"P({target_var}={target_val}"
        if intervention:
            desc += " | do(" + ", ".join(f"{k}={v}" for k, v in intervention.items()) + ")"
        desc += ")"

        return BoundResult(
            lower=lb,
            upper=ub,
            bound_type="probability",
            query_description=desc,
            dual_certificate_lower=dual_lb,
            dual_certificate_upper=dual_ub,
            tightness=tightness,
        )

    def probability_bounds_threshold(
        self,
        target_var: str,
        threshold: int,
        intervention: Optional[Dict[str, int]] = None,
    ) -> BoundResult:
        """
        Compute bounds on P(target_var > threshold | do(intervention)).
        """
        card = self.dag.card[target_var]
        c = np.zeros(self._total_vars, dtype=np.float64)

        for val in range(threshold + 1, card):
            c_part = self._build_probability_objective(target_var, val, intervention)
            c += c_part

        lb, dual_lb = self._solve_lp(c, sense="min")
        ub, dual_ub = self._solve_lp(c, sense="max")

        lb = max(0.0, min(1.0, lb))
        ub = max(0.0, min(1.0, ub))
        if lb > ub:
            lb, ub = ub, lb

        desc = f"P({target_var}>{threshold}"
        if intervention:
            desc += " | do(" + ", ".join(f"{k}={v}" for k, v in intervention.items()) + ")"
        desc += ")"

        return BoundResult(
            lower=lb,
            upper=ub,
            bound_type="probability",
            query_description=desc,
            dual_certificate_lower=dual_lb,
            dual_certificate_upper=dual_ub,
            tightness=1.0 - (ub - lb),
        )

    # ------------------------------------------------------------------
    #  Expectation bounds
    # ------------------------------------------------------------------

    def expectation_bounds(
        self,
        target_var: str,
        intervention: Optional[Dict[str, int]] = None,
        values: Optional[np.ndarray] = None,
    ) -> BoundResult:
        """
        Compute bounds on E[target_var | do(intervention)].

        Parameters
        ----------
        target_var : str
        intervention : dict, optional
        values : ndarray, optional
            Custom mapping from target_var values to reals.
            Defaults to [0, 1, ..., card-1] / (card-1).

        Returns
        -------
        BoundResult
        """
        card = self.dag.card[target_var]

        if values is None:
            values = np.arange(card, dtype=np.float64)
            if card > 1:
                values = values / (card - 1)

        c = np.zeros(self._total_vars, dtype=np.float64)
        for val_idx in range(card):
            weight = float(values[val_idx])
            c_part = self._build_probability_objective(
                target_var, val_idx, intervention
            )
            c += weight * c_part

        lb, dual_lb = self._solve_lp(c, sense="min")
        ub, dual_ub = self._solve_lp(c, sense="max")

        if lb > ub:
            lb, ub = ub, lb

        desc = f"E[{target_var}"
        if intervention:
            desc += " | do(" + ", ".join(f"{k}={v}" for k, v in intervention.items()) + ")"
        desc += "]"

        max_range = float(values.max() - values.min()) if len(values) > 1 else 1.0
        tightness = 1.0 - (ub - lb) / max(max_range, 1e-10)
        tightness = max(0.0, min(1.0, tightness))

        return BoundResult(
            lower=lb,
            upper=ub,
            bound_type="expectation",
            query_description=desc,
            dual_certificate_lower=dual_lb,
            dual_certificate_upper=dual_ub,
            tightness=tightness,
        )

    def ate_bounds(
        self,
        treatment: str,
        outcome: str,
        values: Optional[np.ndarray] = None,
    ) -> BoundResult:
        """
        Compute bounds on the Average Treatment Effect:
            ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
        for binary treatment.

        Uses Fréchet-type inequalities.
        """
        r1 = self.expectation_bounds(outcome, intervention={treatment: 1}, values=values)
        r0 = self.expectation_bounds(outcome, intervention={treatment: 0}, values=values)

        ate_lb = r1.lower - r0.upper
        ate_ub = r1.upper - r0.lower

        return BoundResult(
            lower=ate_lb,
            upper=ate_ub,
            bound_type="expectation",
            query_description=f"ATE = E[{outcome}|do({treatment}=1)] - E[{outcome}|do({treatment}=0)]",
            tightness=max(0.0, 1.0 - (ate_ub - ate_lb) / 2.0),
        )

    # ------------------------------------------------------------------
    #  Quantile bounds
    # ------------------------------------------------------------------

    def quantile_bounds(
        self,
        target_var: str,
        quantile: float,
        intervention: Optional[Dict[str, int]] = None,
    ) -> BoundResult:
        """
        Compute bounds on the q-th quantile of target_var under intervention.

        The q-quantile is the smallest v such that P(Y <= v | do(X=x)) >= q.

        We binary-search over the threshold t and compute
        bounds on P(Y <= t | do(X=x)).
        """
        card = self.dag.card[target_var]

        lower_q = 0
        upper_q = card - 1

        # For each threshold, check if P(Y<=t) can be >= q
        for t in range(card):
            cdf_bounds = self._cdf_bounds(target_var, t, intervention)
            if cdf_bounds.upper < quantile:
                lower_q = t + 1
            if cdf_bounds.lower >= quantile and t < upper_q:
                upper_q = t

        desc = f"Q_{quantile:.2f}({target_var}"
        if intervention:
            desc += " | do(" + ", ".join(f"{k}={v}" for k, v in intervention.items()) + ")"
        desc += ")"

        return BoundResult(
            lower=float(lower_q),
            upper=float(upper_q),
            bound_type="quantile",
            query_description=desc,
            tightness=1.0 - float(upper_q - lower_q) / max(card - 1, 1),
        )

    def _cdf_bounds(
        self,
        target_var: str,
        threshold: int,
        intervention: Optional[Dict[str, int]] = None,
    ) -> BoundResult:
        """Bounds on P(Y <= threshold | do(X=x))."""
        c = np.zeros(self._total_vars, dtype=np.float64)
        for val in range(threshold + 1):
            c_part = self._build_probability_objective(target_var, val, intervention)
            c += c_part

        lb, _ = self._solve_lp(c, sense="min")
        ub, _ = self._solve_lp(c, sense="max")
        lb = max(0.0, min(1.0, lb))
        ub = max(0.0, min(1.0, ub))

        return BoundResult(
            lower=lb, upper=ub, bound_type="probability",
            query_description=f"P({target_var}<={threshold})",
        )

    # ------------------------------------------------------------------
    #  Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        target_var: str,
        target_val: int,
        intervention: Optional[Dict[str, int]] = None,
        perturbation: float = 0.01,
        constraint_names: Optional[List[str]] = None,
    ) -> SensitivityReport:
        """
        Compute how the bounds change when each constraint RHS is perturbed.

        For each constraint i, we solve with b_i +/- perturbation and
        measure the change in bounds.

        If dual variables are available, the sensitivity equals the dual
        variable (by LP duality).
        """
        c = self._build_probability_objective(target_var, target_val, intervention)

        base_lb, dual_lb = self._solve_lp(c, sense="min")
        base_ub, dual_ub = self._solve_lp(c, sense="max")

        entries: List[SensitivityEntry] = []
        m = len(self.b_eq)
        names = constraint_names or [f"constraint_{i}" for i in range(m)]

        for i in range(m):
            # Dual-based sensitivity (first-order exact)
            d_lower = float(dual_lb[i]) * perturbation if dual_lb is not None else 0.0
            d_upper = float(dual_ub[i]) * perturbation if dual_ub is not None else 0.0

            # Verification by re-solving (for a subset of constraints)
            if i < min(20, m):
                b_pert = self.b_eq.copy()
                b_pert[i] += perturbation

                lb_pert, _ = self._solve_lp_with_b(c, b_pert, sense="min")
                ub_pert, _ = self._solve_lp_with_b(c, b_pert, sense="max")

                d_lower = lb_pert - base_lb
                d_upper = ub_pert - base_ub

            name = names[i] if i < len(names) else f"constraint_{i}"
            entries.append(SensitivityEntry(
                constraint_name=name,
                constraint_index=i,
                delta_lower=d_lower,
                delta_upper=d_upper,
                dual_value_lower=float(dual_lb[i]) if dual_lb is not None else 0.0,
                dual_value_upper=float(dual_ub[i]) if dual_ub is not None else 0.0,
            ))

        return SensitivityReport(
            entries=entries,
            perturbation_size=perturbation,
            base_lower=base_lb,
            base_upper=base_ub,
        )

    def dual_certificate_validity(
        self,
        dual_values: np.ndarray,
        c: np.ndarray,
        is_lower: bool = True,
    ) -> Dict[str, Any]:
        """
        Check the validity of a dual certificate.

        For a lower bound: the dual certificate y satisfies:
            A^T y <= c   and   b^T y = bound

        For an upper bound: A^T y >= c   and   b^T y = bound
        """
        if sparse.issparse(self.A_eq):
            Aty = self.A_eq.T.dot(dual_values)
        else:
            Aty = self.A_eq.T @ dual_values

        bty = float(self.b_eq @ dual_values)

        if is_lower:
            violations = np.sum(Aty > c + 1e-10)
            max_violation = float(np.max(Aty - c))
        else:
            violations = np.sum(Aty < c - 1e-10)
            max_violation = float(np.max(c - Aty))

        return {
            "valid": int(violations) == 0,
            "num_violations": int(violations),
            "max_violation": max_violation,
            "dual_objective": bty,
            "is_lower": is_lower,
        }

    def bound_tightness_analysis(
        self,
        target_var: str,
        intervention: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse how tight the bounds are by computing bounds on each
        probability P(Y=y | do(X=x)) and checking if any are point-identified.
        """
        card = self.dag.card[target_var]
        results: Dict[str, Any] = {"variable": target_var, "values": {}}

        total_gap = 0.0
        for y_val in range(card):
            br = self.probability_bounds(target_var, y_val, intervention)
            results["values"][y_val] = {
                "lower": br.lower,
                "upper": br.upper,
                "gap": br.gap,
                "tightness": br.tightness,
                "point_identified": br.gap < 1e-6,
            }
            total_gap += br.gap

        results["total_gap"] = total_gap
        results["avg_tightness"] = 1.0 - total_gap / max(card, 1)

        # Check monotonicity of bounds
        lowers = [results["values"][y]["lower"] for y in range(card)]
        uppers = [results["values"][y]["upper"] for y in range(card)]
        results["sum_lowers"] = sum(lowers)
        results["sum_uppers"] = sum(uppers)
        # Fréchet bounds: sum of lowers <= 1 <= sum of uppers (ideally)
        results["frechet_consistent"] = (
            sum(lowers) <= 1.0 + 1e-6 and sum(uppers) >= 1.0 - 1e-6
        )

        return results

    # ------------------------------------------------------------------
    #  LP solving
    # ------------------------------------------------------------------

    def _solve_lp(
        self,
        c: np.ndarray,
        sense: str = "min",
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Solve  min/max c^T x  s.t.  A x = b, x >= 0.

        Returns (optimal_value, dual_variables).
        """
        return self._solve_lp_with_b(c, self.b_eq, sense)

    def _solve_lp_with_b(
        self,
        c: np.ndarray,
        b: np.ndarray,
        sense: str = "min",
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Solve LP with a custom RHS vector b."""
        n = self._total_vars
        obj = c.copy() if sense == "min" else -c.copy()

        if sparse.issparse(self.A_eq):
            A_dense = self.A_eq.toarray()
        else:
            A_dense = np.asarray(self.A_eq)

        m = A_dense.shape[0]
        b_use = b[:m].copy()

        bounds = [(0.0, None)] * n

        try:
            result = linprog(
                c=obj,
                A_eq=A_dense,
                b_eq=b_use,
                bounds=bounds,
                method="highs",
                options={"maxiter": 10000, "presolve": True,
                         "dual_feasibility_tolerance": 1e-10,
                         "primal_feasibility_tolerance": 1e-10},
            )
        except Exception as exc:
            logger.warning("LP solve failed: %s", exc)
            if sense == "min":
                return 0.0, None
            else:
                return 1.0, None

        if result.success:
            val = result.fun if sense == "min" else -result.fun

            # Extract duals
            duals = None
            if hasattr(result, "eqlin") and hasattr(result.eqlin, "marginals"):
                duals = np.array(result.eqlin.marginals, dtype=np.float64)
            else:
                duals = np.zeros(m, dtype=np.float64)

            return val, duals
        else:
            logger.debug("LP status %d: %s", result.status, result.message)
            if sense == "min":
                return 0.0, np.zeros(m, dtype=np.float64)
            else:
                return 1.0, np.zeros(m, dtype=np.float64)

    # ------------------------------------------------------------------
    #  Objective construction
    # ------------------------------------------------------------------

    def _build_probability_objective(
        self,
        target_var: str,
        target_val: int,
        intervention: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Build objective vector c such that c^T x = P(target_var = target_val).

        Entries corresponding to joint assignments where target_var == target_val
        (and intervention variables match) are set to 1.
        """
        c = np.zeros(self._total_vars, dtype=np.float64)

        for flat_idx in range(self._total_vars):
            assign = self._flat_to_assignment(flat_idx)

            if assign[target_var] != target_val:
                continue

            if intervention is not None:
                match = True
                for iv_var, iv_val in intervention.items():
                    if iv_var in assign and assign[iv_var] != iv_val:
                        match = False
                        break
                if not match:
                    continue

            c[flat_idx] = 1.0

        return c

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def _flat_to_assignment(self, flat_idx: int) -> Dict[str, int]:
        assignment: Dict[str, int] = {}
        for node in self._topo:
            card = self.dag.card[node]
            stride = self._strides[node]
            assignment[node] = (flat_idx // stride) % card
        return assignment

    def _compute_strides(self) -> Dict[str, int]:
        strides: Dict[str, int] = {}
        s = 1
        for node in reversed(self._topo):
            strides[node] = s
            s *= self.dag.card[node]
        return strides

    def _compute_total_vars(self) -> int:
        total = 1
        for n in self.dag.nodes:
            total *= self.dag.card[n]
        return total
