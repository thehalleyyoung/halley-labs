"""Quantifier handling for SMT-based DP verification.

Implements quantifier elimination, Skolemization, and instantiation
strategies for handling universally and existentially quantified
privacy constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import Formula

from dp_forge.smt.theory_solver import (
    BoundPropagation,
    ConstraintOp,
    FeasibilityChecker,
    LinearConstraint,
    ModelGeneration,
    parse_linear_constraint,
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class QuantifiedFormula:
    """A quantified formula: ∀x. φ(x) or ∃x. φ(x)."""

    quantifier: str  # "forall" or "exists"
    bound_vars: List[str]
    body: List[LinearConstraint]
    free_vars: Set[str] = field(default_factory=set)

    @property
    def is_universal(self) -> bool:
        return self.quantifier == "forall"

    @property
    def is_existential(self) -> bool:
        return self.quantifier == "exists"

    def __repr__(self) -> str:
        q = "∀" if self.is_universal else "∃"
        bv = ", ".join(self.bound_vars)
        return f"{q}({bv}). [{len(self.body)} constraints]"


@dataclass
class EliminationResult:
    """Result of quantifier elimination."""

    constraints: List[LinearConstraint]
    eliminated_vars: List[str]
    remaining_vars: Set[str]
    success: bool = True

    def __repr__(self) -> str:
        return (f"EliminationResult(eliminated={len(self.eliminated_vars)}, "
                f"remaining={len(self.remaining_vars)}, "
                f"constraints={len(self.constraints)})")


# ---------------------------------------------------------------------------
# FourierMotzkin
# ---------------------------------------------------------------------------

class FourierMotzkin:
    """Fourier-Motzkin quantifier elimination for linear real arithmetic.

    Given a set of linear constraints and a variable to eliminate,
    produces an equivalent set of constraints without that variable
    by combining upper-bound and lower-bound constraints.

    Complexity: O(n²) per variable eliminated (constraint blowup).
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance

    def eliminate_variable(
        self,
        constraints: List[LinearConstraint],
        variable: str,
    ) -> List[LinearConstraint]:
        """Eliminate a single variable from a set of constraints.

        Partitions constraints into:
        - upper bounds on variable (coeff > 0 in LEQ / coeff < 0 in GEQ)
        - lower bounds on variable
        - unrelated constraints

        Then combines each upper bound with each lower bound.
        """
        upper: List[LinearConstraint] = []  # variable <= ...
        lower: List[LinearConstraint] = []  # variable >= ...
        unrelated: List[LinearConstraint] = []

        for c in constraints:
            coeff = c.coefficients.get(variable, 0.0)
            if abs(coeff) < self.tolerance:
                unrelated.append(c)
                continue

            # Normalize to isolate variable
            norm = self._normalize_for_var(c, variable)
            if norm is None:
                unrelated.append(c)
                continue

            bound_type, _ = norm
            if bound_type == "upper":
                upper.append(c)
            elif bound_type == "lower":
                lower.append(c)
            else:
                unrelated.append(c)

        # Combine upper and lower bounds
        result = list(unrelated)
        for ub_c in upper:
            for lb_c in lower:
                combined = self._combine(ub_c, lb_c, variable)
                if combined is not None:
                    result.append(combined)

        return result

    def eliminate_all(
        self,
        constraints: List[LinearConstraint],
        variables: List[str],
    ) -> EliminationResult:
        """Eliminate multiple variables in sequence.

        Variables are eliminated in the given order. The order affects
        the size of the resulting constraint set.
        """
        current = list(constraints)
        eliminated: List[str] = []
        remaining_vars: Set[str] = set()
        for c in constraints:
            remaining_vars.update(c.variables)

        for var in variables:
            if var not in remaining_vars:
                eliminated.append(var)
                continue
            current = self.eliminate_variable(current, var)
            remaining_vars.discard(var)
            eliminated.append(var)

            # Simplify trivially true/false constraints
            current = self._simplify(current)

        return EliminationResult(
            constraints=current,
            eliminated_vars=eliminated,
            remaining_vars=remaining_vars,
        )

    def _normalize_for_var(
        self,
        constraint: LinearConstraint,
        variable: str,
    ) -> Optional[Tuple[str, float]]:
        """Determine if constraint gives upper or lower bound on variable.

        Returns:
            ("upper" | "lower", normalized_rhs) or None
        """
        coeff = constraint.coefficients.get(variable, 0.0)
        if abs(coeff) < self.tolerance:
            return None

        if constraint.op == ConstraintOp.LEQ:
            if coeff > 0:
                return "upper", constraint.rhs / coeff
            else:
                return "lower", constraint.rhs / coeff
        elif constraint.op == ConstraintOp.GEQ:
            if coeff > 0:
                return "lower", constraint.rhs / coeff
            else:
                return "upper", constraint.rhs / coeff
        elif constraint.op == ConstraintOp.EQ:
            # Equality gives both bounds
            return "upper", constraint.rhs / coeff
        return None

    def _combine(
        self,
        upper_c: LinearConstraint,
        lower_c: LinearConstraint,
        variable: str,
    ) -> Optional[LinearConstraint]:
        """Combine an upper-bound and lower-bound constraint to eliminate variable.

        If upper gives: coeff_u * x + rest_u ≤ rhs_u
        And lower gives: coeff_l * x + rest_l ≥ rhs_l
        We derive: rest terms combined appropriately.
        """
        coeff_u = upper_c.coefficients.get(variable, 0.0)
        coeff_l = lower_c.coefficients.get(variable, 0.0)

        if abs(coeff_u) < self.tolerance or abs(coeff_l) < self.tolerance:
            return None

        # Normalize both constraints to have positive coefficient for variable
        # Upper: c_u * x <= rhs_u - rest_u  =>  x <= (rhs_u - rest_u) / c_u
        # Lower: c_l * x >= rhs_l - rest_l  =>  x >= (rhs_l - rest_l) / c_l
        # Combined: (rhs_l - rest_l) / c_l <= (rhs_u - rest_u) / c_u

        # Scale to eliminate variable
        scale_u = abs(coeff_l)
        scale_l = abs(coeff_u)

        new_coeffs: Dict[str, float] = {}

        # Add scaled upper constraint terms (excluding variable)
        for v, c in upper_c.coefficients.items():
            if v == variable:
                continue
            new_coeffs[v] = new_coeffs.get(v, 0.0) + scale_u * c

        # Determine sign handling
        if (coeff_u > 0 and coeff_l < 0) or (coeff_u < 0 and coeff_l > 0):
            # Same direction after normalization: add
            for v, c in lower_c.coefficients.items():
                if v == variable:
                    continue
                new_coeffs[v] = new_coeffs.get(v, 0.0) + scale_l * c
            new_rhs = scale_u * upper_c.rhs + scale_l * lower_c.rhs
        else:
            # Subtract
            for v, c in lower_c.coefficients.items():
                if v == variable:
                    continue
                new_coeffs[v] = new_coeffs.get(v, 0.0) - scale_l * c
            new_rhs = scale_u * upper_c.rhs - scale_l * lower_c.rhs

        # Remove near-zero coefficients
        new_coeffs = {v: c for v, c in new_coeffs.items() if abs(c) > self.tolerance}

        return LinearConstraint(
            coefficients=new_coeffs,
            op=ConstraintOp.LEQ,
            rhs=new_rhs,
            label=f"fm_elim_{variable}",
        )

    def _simplify(self, constraints: List[LinearConstraint]) -> List[LinearConstraint]:
        """Remove trivially true constraints and detect trivially false ones."""
        result = []
        for c in constraints:
            if not c.coefficients:
                # Constant constraint
                if c.op == ConstraintOp.LEQ and 0.0 <= c.rhs + self.tolerance:
                    continue  # trivially true: 0 <= rhs
                elif c.op == ConstraintOp.GEQ and 0.0 >= c.rhs - self.tolerance:
                    continue  # trivially true: 0 >= rhs
                elif c.op == ConstraintOp.LEQ and 0.0 > c.rhs + self.tolerance:
                    result.append(c)  # infeasible
                else:
                    result.append(c)
            else:
                result.append(c)
        return result


# ---------------------------------------------------------------------------
# CylindricalAlgebraicDecomposition (simplified for polynomial privacy)
# ---------------------------------------------------------------------------

class CylindricalAlgebraicDecomposition:
    """Cylindrical Algebraic Decomposition for polynomial constraints.

    Simplified implementation for low-degree polynomial constraints
    that arise in privacy analysis (e.g., Rényi divergence bounds).
    For purely linear constraints, delegates to Fourier-Motzkin.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self._fm = FourierMotzkin(tolerance)

    def eliminate(
        self,
        constraints: List[LinearConstraint],
        variables: List[str],
    ) -> EliminationResult:
        """Eliminate variables using CAD (falls back to FM for linear)."""
        # Check if all constraints are linear
        all_linear = all(self._is_linear(c) for c in constraints)
        if all_linear:
            return self._fm.eliminate_all(constraints, variables)

        # For polynomial constraints, use a projection-based approach
        return self._cad_eliminate(constraints, variables)

    def _is_linear(self, constraint: LinearConstraint) -> bool:
        """Check if a constraint is linear (always true for LinearConstraint)."""
        return True  # Our representation is inherently linear

    def _cad_eliminate(
        self,
        constraints: List[LinearConstraint],
        variables: List[str],
    ) -> EliminationResult:
        """CAD-based elimination (projection phase + lifting)."""
        # For our LinearConstraint type, this is equivalent to FM
        return self._fm.eliminate_all(constraints, variables)

    def decompose(
        self,
        constraints: List[LinearConstraint],
        variable: str,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Compute a decomposition of the real line into sign-invariant intervals.

        For linear constraints in `variable`, the critical points are the
        roots of the boundary expressions.

        Returns:
            List of (lower, upper) intervals where the constraint set
            has constant truth value.
        """
        critical_points: List[float] = []

        for c in constraints:
            coeff = c.coefficients.get(variable, 0.0)
            if abs(coeff) < self.tolerance:
                continue
            # root: coeff * var + rest = rhs
            rest = sum(v_c for v, v_c in c.coefficients.items() if v != variable)
            root = (c.rhs - rest) / coeff if abs(coeff) > self.tolerance else 0.0
            critical_points.append(root)

        critical_points.sort()
        critical_points = list(dict.fromkeys(critical_points))  # deduplicate

        intervals: List[Tuple[Optional[float], Optional[float]]] = []
        if not critical_points:
            intervals.append((None, None))
            return intervals

        # Before first critical point
        intervals.append((None, critical_points[0]))
        # Between consecutive critical points
        for i in range(len(critical_points) - 1):
            intervals.append((critical_points[i], critical_points[i + 1]))
        # After last critical point
        intervals.append((critical_points[-1], None))

        return intervals


# ---------------------------------------------------------------------------
# Skolemization
# ---------------------------------------------------------------------------

class Skolemization:
    """Skolemize existential quantifiers in DP verification formulas.

    Replaces existentially quantified variables with Skolem functions
    (or constants when there are no enclosing universal quantifiers).
    """

    def __init__(self) -> None:
        self._skolem_counter = 0

    def skolemize(
        self,
        formula: QuantifiedFormula,
        enclosing_universals: Optional[List[str]] = None,
    ) -> Tuple[List[LinearConstraint], Dict[str, str]]:
        """Skolemize an existentially quantified formula.

        Args:
            formula: The quantified formula to Skolemize.
            enclosing_universals: Variables bound by enclosing ∀.

        Returns:
            (modified_constraints, skolem_map) where skolem_map maps
            bound variables to their Skolem constant/function names.
        """
        if formula.is_universal:
            # Universal quantifiers remain (or are handled by instantiation)
            return list(formula.body), {}

        skolem_map: Dict[str, str] = {}
        for bv in formula.bound_vars:
            self._skolem_counter += 1
            if enclosing_universals:
                # Skolem function: sk_n(u1, u2, ...)
                args = ", ".join(enclosing_universals)
                sk_name = f"__sk_{self._skolem_counter}({args})"
            else:
                # Skolem constant
                sk_name = f"__sk_{self._skolem_counter}"
            skolem_map[bv] = sk_name

        # Substitute in constraints
        result = []
        for c in formula.body:
            new_coeffs: Dict[str, float] = {}
            for var, coeff in c.coefficients.items():
                new_var = skolem_map.get(var, var)
                new_coeffs[new_var] = new_coeffs.get(new_var, 0.0) + coeff
            result.append(LinearConstraint(
                coefficients=new_coeffs,
                op=c.op,
                rhs=c.rhs,
                label=c.label,
            ))

        return result, skolem_map

    def skolemize_nested(
        self,
        formulas: List[QuantifiedFormula],
    ) -> Tuple[List[LinearConstraint], Dict[str, str]]:
        """Skolemize a sequence of nested quantified formulas.

        Processes in order, tracking enclosing universals for
        proper Skolem function construction.
        """
        all_constraints: List[LinearConstraint] = []
        all_skolem: Dict[str, str] = {}
        universals: List[str] = []

        for f in formulas:
            if f.is_universal:
                universals.extend(f.bound_vars)
                all_constraints.extend(f.body)
            else:
                constraints, sk_map = self.skolemize(f, universals if universals else None)
                all_constraints.extend(constraints)
                all_skolem.update(sk_map)

        return all_constraints, all_skolem

    def reset(self) -> None:
        self._skolem_counter = 0


# ---------------------------------------------------------------------------
# InstantiationEngine
# ---------------------------------------------------------------------------

class InstantiationEngine:
    """E-matching based quantifier instantiation.

    For universal quantifiers ∀x. φ(x), generates ground instances
    φ(t) for relevant terms t found in the current formula set.
    """

    def __init__(self, max_instances: int = 1000) -> None:
        self.max_instances = max_instances
        self._instance_count = 0

    def instantiate_universal(
        self,
        formula: QuantifiedFormula,
        ground_terms: Dict[str, List[float]],
    ) -> List[LinearConstraint]:
        """Instantiate a universal quantifier with ground terms.

        Args:
            formula: ∀x. φ(x) to instantiate.
            ground_terms: Map from bound var name to list of ground values.

        Returns:
            List of ground instances.
        """
        if not formula.is_universal:
            return list(formula.body)

        instances: List[LinearConstraint] = []
        combos = self._combinations(formula.bound_vars, ground_terms)

        for combo in combos:
            if self._instance_count >= self.max_instances:
                break
            for c in formula.body:
                inst = self._substitute(c, combo)
                instances.append(inst)
                self._instance_count += 1

        return instances

    def e_match(
        self,
        pattern_constraints: List[LinearConstraint],
        existing_constraints: List[LinearConstraint],
        pattern_vars: Set[str],
    ) -> List[Dict[str, float]]:
        """Find substitutions by E-matching patterns against existing constraints.

        Simple implementation: collect all constant values appearing
        in existing constraints as potential ground terms.
        """
        ground_values: Set[float] = set()
        for c in existing_constraints:
            ground_values.add(c.rhs)
            for coeff in c.coefficients.values():
                if abs(coeff) > 1e-15 and abs(coeff) != 1.0:
                    ground_values.add(coeff)

        # Generate substitutions
        subs: List[Dict[str, float]] = []
        values = sorted(ground_values)
        for val in values[:20]:  # limit substitutions
            for pv in pattern_vars:
                subs.append({pv: val})
        return subs

    def _combinations(
        self,
        bound_vars: List[str],
        ground_terms: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        """Generate combinations of ground term substitutions."""
        if not bound_vars:
            return [{}]

        result: List[Dict[str, float]] = [{}]
        for bv in bound_vars:
            values = ground_terms.get(bv, [0.0])
            new_result: List[Dict[str, float]] = []
            for combo in result:
                for val in values:
                    new_combo = dict(combo)
                    new_combo[bv] = val
                    new_result.append(new_combo)
                    if len(new_result) >= self.max_instances:
                        return new_result
            result = new_result

        return result

    def _substitute(
        self,
        constraint: LinearConstraint,
        substitution: Dict[str, float],
    ) -> LinearConstraint:
        """Substitute ground values for variables in a constraint."""
        new_coeffs: Dict[str, float] = {}
        rhs_adjustment = 0.0

        for var, coeff in constraint.coefficients.items():
            if var in substitution:
                rhs_adjustment += coeff * substitution[var]
            else:
                new_coeffs[var] = new_coeffs.get(var, 0.0) + coeff

        return LinearConstraint(
            coefficients=new_coeffs,
            op=constraint.op,
            rhs=constraint.rhs - rhs_adjustment,
            label=f"{constraint.label}_inst",
        )

    def reset(self) -> None:
        self._instance_count = 0


# ---------------------------------------------------------------------------
# ModelBasedProjection
# ---------------------------------------------------------------------------

class ModelBasedProjection:
    """Model-based quantifier instantiation (MBQI).

    Uses models from the theory solver to guide quantifier instantiation.
    When a candidate model is found, checks if any universal quantifier
    is violated, and if so, generates a blocking instance.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self._checker = FeasibilityChecker(tolerance)
        self._model_gen = ModelGeneration(tolerance)
        self._instantiator = InstantiationEngine()

    def project(
        self,
        constraints: List[LinearConstraint],
        variables_to_project: List[str],
        model: Optional[Dict[str, float]] = None,
    ) -> List[LinearConstraint]:
        """Project out variables using model-based instantiation.

        Given a model, finds the tightest constraints that hold at
        the model point and eliminates the projected variables.
        """
        if model is None:
            # No model available; fall back to Fourier-Motzkin
            fm = FourierMotzkin(self.tolerance)
            return fm.eliminate_all(constraints, variables_to_project).constraints

        # Use model values to instantiate projected variables
        result: List[LinearConstraint] = []
        for c in constraints:
            projected_vars = [v for v in c.variables if v in variables_to_project]
            if not projected_vars:
                result.append(c)
                continue

            # Substitute model values for projected variables
            new_coeffs: Dict[str, float] = {}
            rhs_adjust = 0.0
            for var, coeff in c.coefficients.items():
                if var in variables_to_project:
                    val = model.get(var, 0.0)
                    rhs_adjust += coeff * val
                else:
                    new_coeffs[var] = new_coeffs.get(var, 0.0) + coeff

            if new_coeffs:
                result.append(LinearConstraint(
                    coefficients=new_coeffs,
                    op=c.op,
                    rhs=c.rhs - rhs_adjust,
                    label=f"mbp_{c.label}",
                ))

        return result

    def refine(
        self,
        universal_formula: QuantifiedFormula,
        candidate_model: Dict[str, float],
    ) -> Optional[List[LinearConstraint]]:
        """Check if a universal formula is violated at the candidate model.

        If violated, returns blocking constraints (instances that
        refute the model).
        """
        if not universal_formula.is_universal:
            return None

        # Evaluate the body at the model
        # Find bound variable values that violate the body
        violated_instances: List[LinearConstraint] = []

        # Try the model values as instantiation
        ground_terms: Dict[str, List[float]] = {}
        for bv in universal_formula.bound_vars:
            val = candidate_model.get(bv, 0.0)
            # Try the model value and perturbations
            ground_terms[bv] = [val, val + 0.1, val - 0.1, 0.0, 1.0]

        instances = self._instantiator.instantiate_universal(
            universal_formula, ground_terms
        )

        # Check which instances are violated
        for inst in instances:
            if not inst.is_satisfied(candidate_model, self.tolerance):
                violated_instances.append(inst)

        return violated_instances if violated_instances else None

    def mbqi_loop(
        self,
        universal_formulas: List[QuantifiedFormula],
        base_constraints: List[LinearConstraint],
        max_iterations: int = 100,
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Run the MBQI loop: iterate between model finding and refinement.

        Returns:
            (is_satisfiable, model_or_none)
        """
        current = list(base_constraints)

        for _ in range(max_iterations):
            # Check satisfiability of current constraints
            feasible, model = self._checker.check(current)
            if not feasible:
                return False, None

            if model is None:
                return False, None

            # Check all universal formulas
            all_satisfied = True
            for uf in universal_formulas:
                violations = self.refine(uf, model)
                if violations:
                    all_satisfied = False
                    current.extend(violations)
                    break

            if all_satisfied:
                return True, model

        return False, None  # timeout


# ---------------------------------------------------------------------------
# QuantifierEliminator — main interface
# ---------------------------------------------------------------------------

class QuantifierEliminator:
    """Unified quantifier elimination for DP verification.

    Selects the appropriate elimination strategy based on the
    constraint type and quantifier structure.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self.fourier_motzkin = FourierMotzkin(tolerance)
        self.cad = CylindricalAlgebraicDecomposition(tolerance)
        self.skolemizer = Skolemization()
        self.instantiator = InstantiationEngine()
        self.mbp = ModelBasedProjection(tolerance)

    def eliminate(
        self,
        formula: QuantifiedFormula,
        strategy: str = "auto",
    ) -> EliminationResult:
        """Eliminate quantifiers from a formula.

        Args:
            formula: Quantified formula.
            strategy: "fm" (Fourier-Motzkin), "cad", "skolem", or "auto".

        Returns:
            EliminationResult with quantifier-free constraints.
        """
        if formula.is_existential:
            if strategy in ("auto", "skolem"):
                return self._eliminate_existential(formula)
            elif strategy == "fm":
                return self.fourier_motzkin.eliminate_all(
                    formula.body, formula.bound_vars
                )
            elif strategy == "cad":
                return self.cad.eliminate(formula.body, formula.bound_vars)
        else:
            if strategy in ("auto", "fm"):
                return self._eliminate_universal(formula)
            elif strategy == "cad":
                return self.cad.eliminate(formula.body, formula.bound_vars)

        return self.fourier_motzkin.eliminate_all(formula.body, formula.bound_vars)

    def _eliminate_existential(
        self,
        formula: QuantifiedFormula,
    ) -> EliminationResult:
        """Eliminate existential quantifier via Skolemization + FM."""
        constraints, sk_map = self.skolemizer.skolemize(formula)

        # The Skolem constants are fresh variables — no elimination needed
        # unless we want to project them out afterward
        remaining = set()
        for c in constraints:
            remaining.update(c.variables)

        return EliminationResult(
            constraints=constraints,
            eliminated_vars=list(formula.bound_vars),
            remaining_vars=remaining,
            success=True,
        )

    def _eliminate_universal(
        self,
        formula: QuantifiedFormula,
    ) -> EliminationResult:
        """Eliminate universal quantifier.

        For ∀x. φ(x), we negate to ¬∃x. ¬φ(x), eliminate x, and negate back.
        In practice for DP constraints, we use Fourier-Motzkin directly.
        """
        return self.fourier_motzkin.eliminate_all(formula.body, formula.bound_vars)

    def eliminate_from_formulas(
        self,
        formulas: List[Formula],
        variables: List[str],
    ) -> List[Formula]:
        """Eliminate variables from a list of Formula objects.

        Parses formulas into LinearConstraints, eliminates, and
        converts back to Formula objects.
        """
        constraints: List[LinearConstraint] = []
        unparsed: List[Formula] = []

        for f in formulas:
            lc = parse_linear_constraint(f)
            if lc is not None:
                constraints.append(lc)
            else:
                unparsed.append(f)

        result = self.fourier_motzkin.eliminate_all(constraints, variables)

        output_formulas: List[Formula] = []
        for c in result.constraints:
            vs = frozenset(c.variables)
            output_formulas.append(Formula(
                expr=str(c),
                variables=vs,
                formula_type="linear_arithmetic",
            ))
        output_formulas.extend(unparsed)

        return output_formulas

    def reset(self) -> None:
        """Reset all internal state."""
        self.skolemizer.reset()
        self.instantiator.reset()


__all__ = [
    "QuantifiedFormula",
    "EliminationResult",
    "FourierMotzkin",
    "CylindricalAlgebraicDecomposition",
    "Skolemization",
    "InstantiationEngine",
    "ModelBasedProjection",
    "QuantifierEliminator",
]
