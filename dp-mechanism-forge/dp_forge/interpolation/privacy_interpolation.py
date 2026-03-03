"""
Privacy-specific interpolation for DP mechanism synthesis.

Specialises Craig interpolation to the domain of differential privacy:
privacy constraint violations, counterexample generalisation,
mechanism bisection, privacy separation hyperplanes,
abstraction refinement, and inductive interpolants for loops.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from dp_forge.types import (
    Formula as DPFormula,
    InterpolantType,
    Predicate,
    PrivacyBudget,
)
from dp_forge.interpolation import (
    Interpolant,
    InterpolantConfig,
    InterpolantStrength,
    InterpolationResult,
    ProofSystem,
    SequenceInterpolant,
)
from dp_forge.interpolation.formula import (
    Formula,
    FormulaNode,
    NodeKind,
    QuantifierElimination,
    SatisfiabilityChecker,
    Simplifier,
    SubstitutionEngine,
)
from dp_forge.interpolation.craig import (
    CraigInterpolant,
    SequenceInterpolation,
    StrengthReduction,
)


# ---------------------------------------------------------------------------
# Privacy Interpolant
# ---------------------------------------------------------------------------


class PrivacyInterpolant:
    """Interpolants for privacy constraint violations.

    Given mechanism constraints M and violation constraints V (encoding
    that some (ε,δ)-DP condition fails), computes an interpolant I such
    that M ⊨ I and I ∧ V is UNSAT.  The interpolant characterises the
    privacy-feasible region and can serve as a predicate for CEGAR
    refinement.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)
        self._simplifier = Simplifier()
        self._checker = SatisfiabilityChecker()

    def compute(
        self,
        mechanism_constraints: DPFormula,
        violation_constraints: DPFormula,
        budget: Optional[PrivacyBudget] = None,
    ) -> InterpolationResult:
        """Compute interpolant separating feasible mechanisms from violations.

        If a budget is provided, the violation constraints are augmented
        with the budget bound.
        """
        if budget is not None:
            violation_constraints = self._augment_with_budget(
                violation_constraints, budget,
            )

        return self._craig.compute(mechanism_constraints, violation_constraints)

    def _augment_with_budget(
        self,
        violation: DPFormula,
        budget: PrivacyBudget,
    ) -> DPFormula:
        """Add epsilon/delta budget constraints to violation formula."""
        eps_var = "epsilon"
        delta_var = "delta"
        aug_parts = [f"({violation.expr})"]
        aug_vars: Set[str] = set(violation.variables)

        # epsilon > budget.epsilon  (violation exceeds budget)
        aug_parts.append(f"(-1*{eps_var} <= -{budget.epsilon})")
        aug_vars.add(eps_var)

        if budget.delta > 0:
            aug_parts.append(f"(-1*{delta_var} <= -{budget.delta})")
            aug_vars.add(delta_var)

        return DPFormula(
            expr=" ∧ ".join(aug_parts),
            variables=frozenset(aug_vars),
        )

    def extract_predicates(
        self,
        interpolant: Interpolant,
        *,
        max_predicates: int = 20,
    ) -> List[Predicate]:
        """Extract atomic predicates from a privacy interpolant.

        Decomposes the interpolant into atomic linear constraints
        suitable for predicate abstraction in CEGAR.
        """
        f = Formula.from_dp_formula(interpolant.formula)
        atoms = self._collect_atoms(f.node)

        predicates: List[Predicate] = []
        for i, atom in enumerate(atoms[:max_predicates]):
            atom_f = Formula(atom)
            pred = Predicate(
                name=f"priv_pred_{i}",
                formula=atom_f.to_dp_formula(),
                is_atomic=True,
            )
            predicates.append(pred)

        return predicates

    def _collect_atoms(self, n: FormulaNode) -> List[FormulaNode]:
        if n.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            return [n]
        if n.kind == NodeKind.VAR:
            return [n]
        if n.kind == NodeKind.CONST:
            return []
        result: List[FormulaNode] = []
        for c in n.children:
            result.extend(self._collect_atoms(c))
        return result


# ---------------------------------------------------------------------------
# Counterexample Generalisation
# ---------------------------------------------------------------------------


@dataclass
class Counterexample:
    """A concrete counterexample to privacy."""

    assignment: Dict[str, float]
    privacy_ratio: float = 0.0
    is_spurious: bool = False

    @property
    def variables(self) -> FrozenSet[str]:
        return frozenset(self.assignment.keys())


class CounterexampleGeneralization:
    """Generalize counterexamples via interpolation.

    Given a concrete counterexample (point violating DP), uses
    interpolation to find a formula characterising a region around
    the counterexample that also violates privacy.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)
        self._checker = SatisfiabilityChecker()

    def generalize(
        self,
        counterexample: Counterexample,
        mechanism_constraints: DPFormula,
        violation_constraints: DPFormula,
    ) -> Optional[Interpolant]:
        """Generalize a counterexample to a region of violations.

        Constructs formula A encoding the counterexample neighbourhood
        and formula B encoding the mechanism constraints.  The interpolant
        describes the generalised violation region.
        """
        # Build formula encoding the counterexample point
        ce_parts: List[str] = []
        ce_vars: Set[str] = set()
        for var, val in counterexample.assignment.items():
            # Encode as tight box:  val - eps <= var <= val + eps
            eps = 0.1 * max(abs(val), 1.0)
            ce_parts.append(f"({var} <= {val + eps})")
            ce_parts.append(f"(-1*{var} <= {-(val - eps)})")
            ce_vars.add(var)

        # Conjoin with violation
        ce_parts.append(f"({violation_constraints.expr})")
        ce_vars.update(violation_constraints.variables)

        ce_formula = DPFormula(
            expr=" ∧ ".join(ce_parts),
            variables=frozenset(ce_vars),
        )

        result = self._craig.compute(ce_formula, mechanism_constraints)
        if result.success and result.interpolant is not None:
            return result.interpolant
        return None

    def is_spurious(
        self,
        counterexample: Counterexample,
        mechanism_constraints: DPFormula,
    ) -> bool:
        """Check whether a counterexample is spurious.

        A counterexample is spurious if the concrete assignment does
        not actually satisfy the mechanism constraints.
        """
        f = Formula.from_dp_formula(mechanism_constraints)
        qe = QuantifierElimination()
        constraints = qe._collect_constraints(f.node)

        for coeffs, rhs in constraints:
            val = sum(
                c * counterexample.assignment.get(v, 0.0)
                for v, c in coeffs.items()
            )
            if val > rhs + 1e-8:
                return True
        return False

    def generalize_batch(
        self,
        counterexamples: List[Counterexample],
        mechanism_constraints: DPFormula,
        violation_constraints: DPFormula,
    ) -> List[Interpolant]:
        """Generalize a batch of counterexamples."""
        results: List[Interpolant] = []
        for ce in counterexamples:
            itp = self.generalize(ce, mechanism_constraints, violation_constraints)
            if itp is not None:
                results.append(itp)
        return results


# ---------------------------------------------------------------------------
# Mechanism Bisection
# ---------------------------------------------------------------------------


class MechanismBisection:
    """Bisect mechanism space using interpolants.

    Iteratively narrows the mechanism parameter space by using
    interpolants to separate feasible from infeasible regions.
    """

    def __init__(
        self,
        config: Optional[InterpolantConfig] = None,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> None:
        self.config = config or InterpolantConfig()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._craig = CraigInterpolant(config)
        self._checker = SatisfiabilityChecker()

    def bisect(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        feasibility_formula: DPFormula,
        violation_formula: DPFormula,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Interpolant]]:
        """Bisect parameter space to find feasible region boundary.

        Returns refined parameter bounds and the interpolants used
        at each bisection step.
        """
        bounds = dict(param_bounds)
        interpolants: List[Interpolant] = []

        for iteration in range(self.max_iterations):
            # Check convergence
            max_range = max(hi - lo for lo, hi in bounds.values())
            if max_range < self.tolerance:
                break

            # Pick dimension with largest range
            dim = max(bounds.keys(), key=lambda v: bounds[v][1] - bounds[v][0])
            lo, hi = bounds[dim]
            mid = (lo + hi) / 2.0

            # Test lower half
            lower_constraints = self._make_box_formula(
                {d: (lo_d, hi_d) if d != dim else (lo, mid)
                 for d, (lo_d, hi_d) in bounds.items()},
            )
            lower_formula = DPFormula(
                expr=f"({lower_constraints}) ∧ ({feasibility_formula.expr})",
                variables=frozenset(bounds.keys()) | feasibility_formula.variables,
            )

            result = self._craig.compute(lower_formula, violation_formula)
            if result.success and result.interpolant is not None:
                interpolants.append(result.interpolant)
                bounds[dim] = (mid, hi)
            else:
                bounds[dim] = (lo, mid)

        return bounds, interpolants

    def _make_box_formula(
        self, bounds: Dict[str, Tuple[float, float]],
    ) -> str:
        parts: List[str] = []
        for var, (lo, hi) in bounds.items():
            parts.append(f"({var} <= {hi})")
            parts.append(f"(-1*{var} <= {-lo})")
        return " ∧ ".join(parts)

    def find_boundary(
        self,
        param_name: str,
        bounds: Tuple[float, float],
        mechanism_template: DPFormula,
        privacy_constraint: DPFormula,
    ) -> Tuple[float, Optional[Interpolant]]:
        """Find the boundary value for a single parameter.

        Binary searches for the critical value of ``param_name``
        at which the mechanism transitions from feasible to infeasible.
        """
        lo, hi = bounds
        best_itp: Optional[Interpolant] = None

        for _ in range(self.max_iterations):
            if hi - lo < self.tolerance:
                break

            mid = (lo + hi) / 2.0
            # Test with param = mid
            test_formula = DPFormula(
                expr=f"({mechanism_template.expr}) ∧ ({param_name} <= {mid}) ∧ (-1*{param_name} <= {-mid})",
                variables=frozenset(mechanism_template.variables | {param_name}),
            )

            result = self._craig.compute(test_formula, privacy_constraint)
            if result.success and result.interpolant is not None:
                best_itp = result.interpolant
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0, best_itp


# ---------------------------------------------------------------------------
# Privacy Separation
# ---------------------------------------------------------------------------


class PrivacySeparation:
    """Find separating hyperplane in privacy space.

    Given sets of feasible and infeasible mechanism configurations,
    finds a linear separator (hyperplane) using interpolation.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)

    def separate(
        self,
        feasible_points: np.ndarray,
        infeasible_points: np.ndarray,
        variable_names: List[str],
    ) -> Optional[Interpolant]:
        """Find a separating hyperplane between feasible and infeasible sets.

        Args:
            feasible_points: (n, d) array of feasible configurations.
            infeasible_points: (m, d) array of infeasible configurations.
            variable_names: Names for the d dimensions.
        """
        if feasible_points.shape[1] != len(variable_names):
            return None
        if infeasible_points.shape[1] != len(variable_names):
            return None

        # Build convex hull constraints for feasible and infeasible sets
        feasible_formula = self._points_to_formula(
            feasible_points, variable_names, "feasible",
        )
        infeasible_formula = self._points_to_formula(
            infeasible_points, variable_names, "infeasible",
        )

        result = self._craig.compute(feasible_formula, infeasible_formula)
        if result.success:
            return result.interpolant
        return None

    def _points_to_formula(
        self,
        points: np.ndarray,
        var_names: List[str],
        label: str,
    ) -> DPFormula:
        """Convert a point set to a bounding-box formula."""
        n, d = points.shape
        parts: List[str] = []
        vars_set: Set[str] = set()

        for j in range(d):
            lo = float(np.min(points[:, j]))
            hi = float(np.max(points[:, j]))
            v = var_names[j]
            vars_set.add(v)
            parts.append(f"({v} <= {hi})")
            parts.append(f"(-1*{v} <= {-lo})")

        return DPFormula(
            expr=" ∧ ".join(parts) if parts else "true",
            variables=frozenset(vars_set),
        )

    def find_margin(
        self,
        feasible_formula: DPFormula,
        infeasible_formula: DPFormula,
    ) -> float:
        """Compute the separation margin between two constraint sets.

        Returns the minimum distance between the feasible and infeasible
        regions, or 0 if they overlap.
        """
        fa = Formula.from_dp_formula(feasible_formula)
        fb = Formula.from_dp_formula(infeasible_formula)
        combined = fa & fb
        checker = SatisfiabilityChecker()
        if not checker.is_unsat(combined):
            return 0.0

        # Estimate margin from interpolant coefficients
        result = self._craig.compute(feasible_formula, infeasible_formula)
        if result.success and result.interpolant is not None:
            f = Formula.from_dp_formula(result.interpolant.formula)
            if f.node.kind == NodeKind.LEQ and f.node.coefficients:
                norm = math.sqrt(sum(c ** 2 for c in f.node.coefficients.values()))
                if norm > 1e-15:
                    return abs(f.node.rhs or 0.0) / norm
        return 0.0


# ---------------------------------------------------------------------------
# Abstraction Refinement
# ---------------------------------------------------------------------------


class AbstractionRefinement:
    """Refine abstract domain using interpolants.

    Implements the interpolation-based refinement step in CEGAR:
    given a spurious counterexample path, compute interpolants
    along the path and add them as new predicates.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._seq_interp = SequenceInterpolation(config)
        self._strength_red = StrengthReduction()
        self._simplifier = Simplifier()

    def refine_from_path(
        self,
        path_formulas: List[DPFormula],
        existing_predicates: Optional[List[Predicate]] = None,
    ) -> List[Predicate]:
        """Refine abstraction by computing sequence interpolants along path.

        Returns new predicates extracted from the interpolant sequence.
        """
        seq = self._seq_interp.compute(path_formulas)
        if seq is None:
            return []

        new_predicates: List[Predicate] = []
        existing_names = {p.name for p in (existing_predicates or [])}

        for i, itp in enumerate(seq.interpolants):
            pred = itp.as_predicate(f"refine_pred_{i}")
            if pred.name not in existing_names:
                new_predicates.append(pred)

        return new_predicates

    def refine_from_counterexample(
        self,
        counterexample: Counterexample,
        abstraction_formulas: List[DPFormula],
        violation_formula: DPFormula,
    ) -> List[Predicate]:
        """Refine by interpolating between abstraction and violation.

        Uses the counterexample to guide which formulas to include.
        """
        relevant: List[DPFormula] = []
        for f in abstraction_formulas:
            # Include formula if counterexample touches its variables
            if f.variables & counterexample.variables:
                relevant.append(f)

        if not relevant:
            relevant = abstraction_formulas[:1] if abstraction_formulas else []

        relevant.append(violation_formula)
        return self.refine_from_path(relevant)

    def iterative_refinement(
        self,
        mechanism_constraints: DPFormula,
        violation_constraints: DPFormula,
        *,
        max_iterations: int = 10,
        max_predicates: int = 50,
    ) -> List[Predicate]:
        """Iteratively refine until convergence or limit.

        Alternates between computing interpolants and adding predicates.
        """
        all_predicates: List[Predicate] = []
        craig = CraigInterpolant(self.config)

        for iteration in range(max_iterations):
            result = craig.compute(mechanism_constraints, violation_constraints)
            if not result.success or result.interpolant is None:
                break

            # Extract predicates from interpolant
            f = Formula.from_dp_formula(result.interpolant.formula)
            atoms = self._collect_atoms(f.node)

            new_found = False
            existing_exprs = {p.formula.expr for p in all_predicates}

            for i, atom in enumerate(atoms):
                atom_f = Formula(atom)
                dp_f = atom_f.to_dp_formula()
                if dp_f.expr not in existing_exprs:
                    pred = Predicate(
                        name=f"iter_{iteration}_pred_{i}",
                        formula=dp_f,
                        is_atomic=True,
                    )
                    all_predicates.append(pred)
                    existing_exprs.add(dp_f.expr)
                    new_found = True

            if not new_found or len(all_predicates) >= max_predicates:
                break

        return all_predicates[:max_predicates]

    def _collect_atoms(self, n: FormulaNode) -> List[FormulaNode]:
        if n.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT, NodeKind.VAR):
            return [n]
        if n.kind == NodeKind.CONST:
            return []
        result: List[FormulaNode] = []
        for c in n.children:
            result.extend(self._collect_atoms(c))
        return result


# ---------------------------------------------------------------------------
# Inductive Interpolant
# ---------------------------------------------------------------------------


class InductiveInterpolant:
    """Compute inductive interpolants for loops.

    Given a transition relation T(x, x') and a safety property P(x),
    computes an inductive invariant I such that:
      - Init(x) ⊨ I(x)           (I holds initially)
      - I(x) ∧ T(x,x') ⊨ I(x')   (I is inductive)
      - I(x) ⊨ P(x)              (I implies safety)

    Uses iterative interpolation: compute I₀ between Init and ¬P,
    strengthen via transition relation until fixed point.
    """

    def __init__(
        self,
        config: Optional[InterpolantConfig] = None,
        *,
        max_iterations: int = 100,
    ) -> None:
        self.config = config or InterpolantConfig()
        self.max_iterations = max_iterations
        self._craig = CraigInterpolant(config)
        self._simplifier = Simplifier()
        self._checker = SatisfiabilityChecker()
        self._subst = SubstitutionEngine()

    def compute(
        self,
        init_formula: DPFormula,
        transition_formula: DPFormula,
        safety_formula: DPFormula,
    ) -> Optional[Interpolant]:
        """Compute an inductive invariant via iterative interpolation.

        The algorithm iteratively strengthens the candidate invariant
        by interpolating between (I ∧ T) and ¬P at primed variables.
        """
        # Step 1: initial interpolant between Init and ¬Safety
        neg_safety = DPFormula(
            expr=f"¬({safety_formula.expr})",
            variables=safety_formula.variables,
        )

        result = self._craig.compute(init_formula, neg_safety)
        if not result.success or result.interpolant is None:
            return None

        candidate = result.interpolant
        common = init_formula.variables & safety_formula.variables

        for iteration in range(self.max_iterations):
            # Step 2: Check inductiveness
            # I(x) ∧ T(x,x') ⊨ I(x')
            # Equivalent to: I(x) ∧ T(x,x') ∧ ¬I(x') is UNSAT

            primed = self._prime_formula(candidate.formula)
            neg_primed = DPFormula(
                expr=f"¬({primed.expr})",
                variables=primed.variables,
            )

            lhs = DPFormula(
                expr=f"({candidate.formula.expr}) ∧ ({transition_formula.expr})",
                variables=candidate.formula.variables | transition_formula.variables,
            )

            fa = Formula.from_dp_formula(lhs)
            fb = Formula.from_dp_formula(neg_primed)
            combined = fa & fb

            if self._checker.is_unsat(combined):
                # Candidate is inductive — verify it implies safety
                fi = Formula.from_dp_formula(candidate.formula)
                fs = Formula.from_dp_formula(safety_formula)
                if self._checker.implies(fi, fs):
                    return candidate
                # Need stronger invariant
                result = self._craig.compute(candidate.formula, neg_safety)
                if result.success and result.interpolant is not None:
                    candidate = result.interpolant
                    continue
                return candidate

            # Step 3: Not inductive — strengthen via interpolation
            result = self._craig.compute(
                DPFormula(
                    expr=lhs.expr,
                    variables=lhs.variables,
                ),
                neg_primed,
            )
            if not result.success or result.interpolant is None:
                return candidate

            # Conjoin new interpolant with candidate
            new_formula = DPFormula(
                expr=f"({candidate.formula.expr}) ∧ ({result.interpolant.formula.expr})",
                variables=candidate.formula.variables | result.interpolant.formula.variables,
            )
            new_common = candidate.common_variables | result.interpolant.common_variables

            candidate = Interpolant(
                formula=new_formula,
                interpolant_type=self.config.interpolant_type,
                common_variables=new_common,
                strength=self.config.strength,
            )

        return candidate

    def _prime_formula(self, f: DPFormula) -> DPFormula:
        """Create primed version of formula: x -> x'."""
        import re
        primed_expr = f.expr
        primed_vars: Set[str] = set()
        for v in sorted(f.variables, key=len, reverse=True):
            primed_v = f"{v}'"
            primed_expr = re.sub(r'\b' + re.escape(v) + r'\b', primed_v, primed_expr)
            primed_vars.add(primed_v)
        return DPFormula(
            expr=primed_expr,
            variables=frozenset(primed_vars),
        )

    def check_inductiveness(
        self,
        invariant: Interpolant,
        transition_formula: DPFormula,
    ) -> bool:
        """Check whether an invariant is inductive w.r.t. transition."""
        primed = self._prime_formula(invariant.formula)
        neg_primed = DPFormula(
            expr=f"¬({primed.expr})",
            variables=primed.variables,
        )

        lhs = DPFormula(
            expr=f"({invariant.formula.expr}) ∧ ({transition_formula.expr})",
            variables=invariant.formula.variables | transition_formula.variables,
        )

        fa = Formula.from_dp_formula(lhs)
        fb = Formula.from_dp_formula(neg_primed)
        combined = fa & fb

        return self._checker.is_unsat(combined)

    def compute_k_inductive(
        self,
        init_formula: DPFormula,
        transition_formula: DPFormula,
        safety_formula: DPFormula,
        k: int = 3,
    ) -> Optional[Interpolant]:
        """Compute k-inductive invariant via bounded unrolling.

        Unrolls the transition relation k times and computes an
        interpolant for the resulting path.
        """
        path_formulas: List[DPFormula] = [init_formula]

        current_vars = init_formula.variables
        for step in range(k):
            # Rename variables in transition for step i -> i+1
            suffix = f"_{step}"
            next_suffix = f"_{step + 1}"
            renamed_expr = transition_formula.expr
            renamed_vars: Set[str] = set()

            for v in sorted(transition_formula.variables, key=len, reverse=True):
                if v.endswith("'"):
                    base = v[:-1]
                    renamed_expr = renamed_expr.replace(v, f"{base}{next_suffix}")
                    renamed_vars.add(f"{base}{next_suffix}")
                else:
                    renamed_expr = renamed_expr.replace(v, f"{v}{suffix}")
                    renamed_vars.add(f"{v}{suffix}")

            path_formulas.append(DPFormula(
                expr=renamed_expr,
                variables=frozenset(renamed_vars),
            ))

        # Safety at final step
        final_suffix = f"_{k}"
        safety_expr = safety_formula.expr
        safety_vars: Set[str] = set()
        for v in sorted(safety_formula.variables, key=len, reverse=True):
            safety_expr = safety_expr.replace(v, f"{v}{final_suffix}")
            safety_vars.add(f"{v}{final_suffix}")

        neg_safety = DPFormula(
            expr=f"¬({safety_expr})",
            variables=frozenset(safety_vars),
        )
        path_formulas.append(neg_safety)

        seq = SequenceInterpolation(self.config)
        result = seq.compute(path_formulas)
        if result is not None and result.interpolants:
            return result.interpolants[0]
        return None


__all__ = [
    "Counterexample",
    "PrivacyInterpolant",
    "CounterexampleGeneralization",
    "MechanismBisection",
    "PrivacySeparation",
    "AbstractionRefinement",
    "InductiveInterpolant",
]
