"""Theory solver for linear real arithmetic (LRA).

Implements a simplex-based decision procedure for the theory of linear
real arithmetic, with hooks for DPLL(T) integration: incremental
assertion, theory propagation, conflict explanation, and model generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import Formula


# ---------------------------------------------------------------------------
# Internal constraint representation
# ---------------------------------------------------------------------------


class ConstraintOp(Enum):
    """Relational operator in a linear constraint."""

    LEQ = auto()   # <=
    GEQ = auto()   # >=
    EQ = auto()    # =
    LT = auto()    # <  (strict)
    GT = auto()    # >  (strict)

    def flip(self) -> ConstraintOp:
        _map = {
            ConstraintOp.LEQ: ConstraintOp.GEQ,
            ConstraintOp.GEQ: ConstraintOp.LEQ,
            ConstraintOp.EQ: ConstraintOp.EQ,
            ConstraintOp.LT: ConstraintOp.GT,
            ConstraintOp.GT: ConstraintOp.LT,
        }
        return _map[self]


@dataclass
class LinearConstraint:
    """A single linear constraint: sum(coeff_i * x_i) op rhs."""

    coefficients: Dict[str, float]
    op: ConstraintOp
    rhs: float
    label: str = ""

    @property
    def variables(self) -> Set[str]:
        return set(self.coefficients.keys())

    def evaluate(self, assignment: Dict[str, float]) -> float:
        """Evaluate the LHS given an assignment."""
        return sum(c * assignment.get(v, 0.0) for v, c in self.coefficients.items())

    def is_satisfied(self, assignment: Dict[str, float], tol: float = 1e-10) -> bool:
        lhs = self.evaluate(assignment)
        if self.op == ConstraintOp.LEQ:
            return lhs <= self.rhs + tol
        elif self.op == ConstraintOp.GEQ:
            return lhs >= self.rhs - tol
        elif self.op == ConstraintOp.EQ:
            return abs(lhs - self.rhs) <= tol
        elif self.op == ConstraintOp.LT:
            return lhs < self.rhs + tol
        elif self.op == ConstraintOp.GT:
            return lhs > self.rhs - tol
        return False

    def negate(self) -> LinearConstraint:
        """Return the negation of this constraint."""
        if self.op == ConstraintOp.LEQ:
            return LinearConstraint(dict(self.coefficients), ConstraintOp.GT, self.rhs, self.label)
        elif self.op == ConstraintOp.GEQ:
            return LinearConstraint(dict(self.coefficients), ConstraintOp.LT, self.rhs, self.label)
        elif self.op == ConstraintOp.LT:
            return LinearConstraint(dict(self.coefficients), ConstraintOp.GEQ, self.rhs, self.label)
        elif self.op == ConstraintOp.GT:
            return LinearConstraint(dict(self.coefficients), ConstraintOp.LEQ, self.rhs, self.label)
        else:
            # EQ negation => disjunction; approximate as strict inequality
            return LinearConstraint(dict(self.coefficients), ConstraintOp.GT, self.rhs, self.label)

    def __repr__(self) -> str:
        op_str = {ConstraintOp.LEQ: "<=", ConstraintOp.GEQ: ">=",
                  ConstraintOp.EQ: "=", ConstraintOp.LT: "<", ConstraintOp.GT: ">"}
        terms = " + ".join(f"{c}*{v}" for v, c in self.coefficients.items())
        return f"{terms} {op_str[self.op]} {self.rhs}"


@dataclass
class Bound:
    """Variable bound: lower or upper."""

    value: float
    is_strict: bool = False
    reason: Optional[int] = None  # constraint index that implied this bound

    def __repr__(self) -> str:
        s = "<" if self.is_strict else "<="
        return f"Bound({self.value}, strict={self.is_strict})"


@dataclass
class VariableInfo:
    """Tracking info for a simplex variable."""

    name: str
    lower: Optional[Bound] = None
    upper: Optional[Bound] = None
    value: float = 0.0
    is_basic: bool = False
    tableau_index: int = -1


# ---------------------------------------------------------------------------
# Formula parsing utilities
# ---------------------------------------------------------------------------

def parse_linear_constraint(formula: Formula) -> Optional[LinearConstraint]:
    """Parse a Formula into a LinearConstraint.

    Supports simple forms like:
      "2.0*x + 3.0*y <= 5.0"
      "x >= 0.0"
      "x - y = 0.0"
    """
    expr = formula.expr.strip()
    op = None
    op_enum = None
    for candidate, enum_val in [("<=", ConstraintOp.LEQ), (">=", ConstraintOp.GEQ),
                                 ("=", ConstraintOp.EQ), ("<", ConstraintOp.LT),
                                 (">", ConstraintOp.GT)]:
        # Check for <= and >= before < and >
        pass

    # Find operator
    for candidate, enum_val in [("<=", ConstraintOp.LEQ), (">=", ConstraintOp.GEQ),
                                 ("!=", None)]:
        if candidate in expr:
            parts = expr.split(candidate, 1)
            op = candidate
            op_enum = enum_val
            break
    if op is None:
        for candidate, enum_val in [("<", ConstraintOp.LT), (">", ConstraintOp.GT),
                                     ("=", ConstraintOp.EQ)]:
            if candidate in expr:
                parts = expr.split(candidate, 1)
                op = candidate
                op_enum = enum_val
                break

    if op is None or op_enum is None:
        return None

    lhs_str, rhs_str = parts[0].strip(), parts[1].strip()

    coefficients: Dict[str, float] = {}
    try:
        rhs = float(rhs_str)
    except ValueError:
        # RHS has variables too; move to LHS
        rhs = 0.0
        _parse_terms(rhs_str, coefficients, negate=True)

    _parse_terms(lhs_str, coefficients, negate=False)

    return LinearConstraint(coefficients=coefficients, op=op_enum, rhs=rhs)


def _parse_terms(s: str, coefficients: Dict[str, float], negate: bool = False) -> None:
    """Parse linear terms like '2.0*x + 3.0*y - z' into coefficients dict."""
    import re
    sign = -1.0 if negate else 1.0
    # Tokenize: split on + and - while keeping the sign
    # Normalize whitespace around operators
    s = s.strip()
    # Insert '+' before '-' that isn't part of a number after an operator
    # Split into tokens by + or - (keeping the sign with the token)
    tokens = re.split(r'(?<=[^eE])\s*([+\-])\s*', ' ' + s)
    current_sign = 1.0
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token == '+':
            current_sign = 1.0
            continue
        if token == '-':
            current_sign = -1.0
            continue
        if "*" in token:
            parts = token.split("*", 1)
            left, right = parts[0].strip(), parts[1].strip()
            try:
                coeff = float(left)
                var = right
            except ValueError:
                try:
                    coeff = float(right)
                    var = left
                except ValueError:
                    continue
            coefficients[var] = coefficients.get(var, 0.0) + sign * current_sign * coeff
        else:
            try:
                # Pure number - ignore (it's part of a constant)
                float(token)
            except ValueError:
                # Pure variable with coefficient 1
                var = token.strip()
                coefficients[var] = coefficients.get(var, 0.0) + sign * current_sign * 1.0
        current_sign = 1.0  # reset sign after consuming a term


# ---------------------------------------------------------------------------
# SimplexMethod
# ---------------------------------------------------------------------------

class SimplexMethod:
    """Two-phase simplex method for linear feasibility over reals.

    Maintains a tableau with basic/non-basic partition and performs
    pivoting to find feasible assignments within variable bounds.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self.variables: Dict[str, VariableInfo] = {}
        self.constraints: List[LinearConstraint] = []
        self.slack_count: int = 0
        # Tableau: rows indexed by basic vars, columns by non-basic vars
        self._tableau: Optional[npt.NDArray[np.float64]] = None
        self._basic_vars: List[str] = []
        self._nonbasic_vars: List[str] = []
        self._rhs: Optional[npt.NDArray[np.float64]] = None
        self._var_index: Dict[str, int] = {}
        self._built = False

    def add_variable(self, name: str, lower: Optional[float] = None,
                     upper: Optional[float] = None) -> None:
        lb = Bound(lower) if lower is not None else None
        ub = Bound(upper) if upper is not None else None
        init_val = 0.0
        if lower is not None and lower > 0:
            init_val = lower
        elif upper is not None and upper < 0:
            init_val = upper
        self.variables[name] = VariableInfo(name=name, lower=lb, upper=ub, value=init_val)

    def add_constraint(self, constraint: LinearConstraint) -> int:
        """Add a constraint, returns its index."""
        idx = len(self.constraints)
        self.constraints.append(constraint)
        self._built = False
        return idx

    def _build_tableau(self) -> None:
        """Build the simplex tableau from constraints."""
        all_vars = set()
        for c in self.constraints:
            all_vars.update(c.variables)
        for v in all_vars:
            if v not in self.variables:
                self.add_variable(v)

        orig_vars = sorted(all_vars)
        self._nonbasic_vars = list(orig_vars)
        self._basic_vars = []

        m = len(self.constraints)
        n = len(orig_vars)
        self._var_index = {v: i for i, v in enumerate(orig_vars)}

        self._tableau = np.zeros((m, n), dtype=np.float64)
        self._rhs = np.zeros(m, dtype=np.float64)

        for row, c in enumerate(self.constraints):
            slack_name = f"__slack_{self.slack_count}"
            self.slack_count += 1
            self._basic_vars.append(slack_name)
            self.variables[slack_name] = VariableInfo(name=slack_name, is_basic=True)

            for var, coeff in c.coefficients.items():
                col = self._var_index[var]
                self._tableau[row, col] = coeff

            if c.op == ConstraintOp.LEQ:
                # slack = rhs - lhs, slack >= 0
                self._rhs[row] = c.rhs
                self.variables[slack_name].lower = Bound(0.0)
            elif c.op == ConstraintOp.GEQ:
                # negate: -lhs <= -rhs => slack = -rhs + lhs >= 0
                self._tableau[row, :] *= -1
                self._rhs[row] = -c.rhs
                self.variables[slack_name].lower = Bound(0.0)
            elif c.op == ConstraintOp.EQ:
                self._rhs[row] = c.rhs
                self.variables[slack_name].lower = Bound(0.0)
                self.variables[slack_name].upper = Bound(0.0)
            elif c.op in (ConstraintOp.LT, ConstraintOp.GT):
                # Treat strict as non-strict with tiny epsilon
                if c.op == ConstraintOp.LT:
                    self._rhs[row] = c.rhs - self.tolerance
                    self.variables[slack_name].lower = Bound(0.0)
                else:
                    self._tableau[row, :] *= -1
                    self._rhs[row] = -c.rhs - self.tolerance
                    self.variables[slack_name].lower = Bound(0.0)

        # Initialize basic variable values
        for row, bv in enumerate(self._basic_vars):
            val = self._rhs[row]
            # Subtract contributions of non-basic vars at their current values
            for col, nbv in enumerate(self._nonbasic_vars):
                val -= self._tableau[row, col] * self.variables[nbv].value
            self.variables[bv].value = val

        self._built = True

    def _pivot(self, row: int, col: int) -> None:
        """Perform a pivot: swap basic var at row with non-basic var at col."""
        leaving = self._basic_vars[row]
        entering = self._nonbasic_vars[col]

        pivot_elem = self._tableau[row, col]
        if abs(pivot_elem) < self.tolerance:
            return

        # Normalize pivot row
        self._tableau[row, :] /= pivot_elem
        self._rhs[row] /= pivot_elem

        # Eliminate entering variable from other rows
        for r in range(len(self._basic_vars)):
            if r == row:
                continue
            factor = self._tableau[r, col]
            if abs(factor) > self.tolerance:
                self._tableau[r, :] -= factor * self._tableau[row, :]
                self._rhs[r] -= factor * self._rhs[row]

        # Update basis
        self._basic_vars[row] = entering
        self._nonbasic_vars[col] = leaving
        self.variables[entering].is_basic = True
        self.variables[leaving].is_basic = False

        # Update values
        for r, bv in enumerate(self._basic_vars):
            val = self._rhs[r]
            for c2, nbv in enumerate(self._nonbasic_vars):
                val -= self._tableau[r, c2] * self.variables[nbv].value
            self.variables[bv].value = val

    def solve(self, max_iterations: int = 10000) -> Tuple[bool, Dict[str, float]]:
        """Run simplex to find a feasible assignment.

        Returns:
            (is_feasible, assignment_dict)
        """
        if not self._built:
            self._build_tableau()

        for _ in range(max_iterations):
            # Find a basic variable that violates its bounds
            violating_row = -1
            for row, bv in enumerate(self._basic_vars):
                info = self.variables[bv]
                if info.lower is not None and info.value < info.lower.value - self.tolerance:
                    violating_row = row
                    break
                if info.upper is not None and info.value > info.upper.value + self.tolerance:
                    violating_row = row
                    break

            if violating_row == -1:
                # All basic vars within bounds => feasible
                return True, self._get_assignment()

            bv = self._basic_vars[violating_row]
            info = self.variables[bv]

            # Determine direction: need to increase or decrease bv?
            if info.lower is not None and info.value < info.lower.value - self.tolerance:
                need_increase = True
            else:
                need_increase = False

            # Find a non-basic variable to pivot with
            pivot_col = -1
            best_ratio = float("inf")

            for col, nbv in enumerate(self._nonbasic_vars):
                coeff = self._tableau[violating_row, col]
                if abs(coeff) < self.tolerance:
                    continue

                nb_info = self.variables[nbv]

                if need_increase:
                    # We want to change nbv so that bv increases
                    # bv row: bv = rhs - sum(a_j * x_j)
                    # Increasing x_col by delta changes bv by -coeff * delta
                    if coeff < 0:
                        # Increasing nbv increases bv
                        if nb_info.upper is not None:
                            room = nb_info.upper.value - nb_info.value
                        else:
                            room = float("inf")
                        if room > self.tolerance:
                            ratio = room / abs(coeff)
                            if ratio < best_ratio:
                                best_ratio = ratio
                                pivot_col = col
                    else:
                        # Decreasing nbv increases bv
                        if nb_info.lower is not None:
                            room = nb_info.value - nb_info.lower.value
                        else:
                            room = float("inf")
                        if room > self.tolerance:
                            ratio = room / abs(coeff)
                            if ratio < best_ratio:
                                best_ratio = ratio
                                pivot_col = col
                else:
                    # Need to decrease bv
                    if coeff > 0:
                        if nb_info.upper is not None:
                            room = nb_info.upper.value - nb_info.value
                        else:
                            room = float("inf")
                        if room > self.tolerance:
                            ratio = room / abs(coeff)
                            if ratio < best_ratio:
                                best_ratio = ratio
                                pivot_col = col
                    else:
                        if nb_info.lower is not None:
                            room = nb_info.value - nb_info.lower.value
                        else:
                            room = float("inf")
                        if room > self.tolerance:
                            ratio = room / abs(coeff)
                            if ratio < best_ratio:
                                best_ratio = ratio
                                pivot_col = col

            if pivot_col == -1:
                # No pivot found => infeasible
                return False, {}

            self._pivot(violating_row, pivot_col)

        # Max iterations reached
        return False, {}

    def _get_assignment(self) -> Dict[str, float]:
        """Extract current variable assignment (original variables only)."""
        result = {}
        for name, info in self.variables.items():
            if not name.startswith("__slack_"):
                result[name] = info.value
        return result

    def get_conflict_indices(self) -> List[int]:
        """Return indices of constraints involved in the infeasibility."""
        # Heuristic: return all constraints with non-zero coefficients
        # in the row of the infeasible basic variable
        conflict = set()
        for row, bv in enumerate(self._basic_vars):
            info = self.variables[bv]
            violated = False
            if info.lower is not None and info.value < info.lower.value - self.tolerance:
                violated = True
            if info.upper is not None and info.value > info.upper.value + self.tolerance:
                violated = True
            if violated:
                # This basic var corresponds to a constraint
                if bv.startswith("__slack_"):
                    idx = int(bv.split("_")[-1])
                    if idx < len(self.constraints):
                        conflict.add(idx)
                # Also add constraints with non-zero tableau entries
                for col, nbv in enumerate(self._nonbasic_vars):
                    if abs(self._tableau[row, col]) > self.tolerance:
                        if nbv.startswith("__slack_"):
                            ci = int(nbv.split("_")[-1])
                            if ci < len(self.constraints):
                                conflict.add(ci)
        if not conflict:
            conflict = set(range(len(self.constraints)))
        return sorted(conflict)


# ---------------------------------------------------------------------------
# BoundPropagation
# ---------------------------------------------------------------------------

class BoundPropagation:
    """Propagate variable bounds from constraint assignments.

    Given a set of active constraints and current bounds, derives
    tighter bounds on variables through constraint propagation.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance

    def propagate(
        self,
        constraints: List[LinearConstraint],
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], bool]:
        """Propagate bounds and detect conflicts.

        Returns:
            (updated_bounds, has_conflict)
        """
        updated = dict(bounds)
        changed = True
        iterations = 0
        max_iter = len(constraints) * 5

        while changed and iterations < max_iter:
            changed = False
            iterations += 1
            for c in constraints:
                result = self._propagate_one(c, updated)
                if result is None:
                    return updated, True  # conflict
                if result:
                    changed = True

        return updated, False

    def _propagate_one(
        self,
        constraint: LinearConstraint,
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> Optional[bool]:
        """Propagate bounds from a single constraint.

        Returns None on conflict, True if bounds were tightened, False otherwise.
        """
        if constraint.op not in (ConstraintOp.LEQ, ConstraintOp.GEQ, ConstraintOp.EQ):
            return False

        vars_in_c = list(constraint.coefficients.keys())
        if len(vars_in_c) == 0:
            return False

        changed = False

        for target_var in vars_in_c:
            target_coeff = constraint.coefficients[target_var]
            if abs(target_coeff) < self.tolerance:
                continue

            # Compute bounds on sum of other terms
            other_lb = 0.0
            other_ub = 0.0
            can_bound = True

            for var in vars_in_c:
                if var == target_var:
                    continue
                coeff = constraint.coefficients[var]
                lb, ub = bounds.get(var, (None, None))

                if coeff > 0:
                    if lb is not None:
                        other_lb += coeff * lb
                    else:
                        can_bound = False
                    if ub is not None:
                        other_ub += coeff * ub
                    else:
                        can_bound = False
                else:
                    if ub is not None:
                        other_lb += coeff * ub
                    else:
                        can_bound = False
                    if lb is not None:
                        other_ub += coeff * lb
                    else:
                        can_bound = False

            if not can_bound:
                continue

            cur_lb, cur_ub = bounds.get(target_var, (None, None))

            if constraint.op in (ConstraintOp.LEQ, ConstraintOp.EQ):
                # sum <= rhs => target_coeff * target <= rhs - other_lb
                if target_coeff > 0:
                    new_ub = (constraint.rhs - other_lb) / target_coeff
                    if cur_ub is None or new_ub < cur_ub - self.tolerance:
                        bounds[target_var] = (cur_lb, new_ub)
                        cur_ub = new_ub
                        changed = True
                else:
                    new_lb = (constraint.rhs - other_lb) / target_coeff
                    if cur_lb is None or new_lb > cur_lb + self.tolerance:
                        bounds[target_var] = (new_lb, cur_ub)
                        cur_lb = new_lb
                        changed = True

            if constraint.op in (ConstraintOp.GEQ, ConstraintOp.EQ):
                # sum >= rhs => target_coeff * target >= rhs - other_ub
                if target_coeff > 0:
                    new_lb = (constraint.rhs - other_ub) / target_coeff
                    if cur_lb is None or new_lb > cur_lb + self.tolerance:
                        bounds[target_var] = (new_lb, cur_ub)
                        cur_lb = new_lb
                        changed = True
                else:
                    new_ub = (constraint.rhs - other_ub) / target_coeff
                    if cur_ub is None or new_ub < cur_ub - self.tolerance:
                        bounds[target_var] = (cur_lb, new_ub)
                        cur_ub = new_ub
                        changed = True

            # Check for conflict
            if cur_lb is not None and cur_ub is not None:
                if cur_lb > cur_ub + self.tolerance:
                    return None  # conflict

        return changed


# ---------------------------------------------------------------------------
# FeasibilityChecker
# ---------------------------------------------------------------------------

class FeasibilityChecker:
    """Check feasibility of a set of linear constraints.

    Uses SimplexMethod internally and provides a clean API for
    the DPLL(T) theory checker.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self._bound_prop = BoundPropagation(tolerance)

    def check(
        self,
        constraints: List[LinearConstraint],
        variable_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check if the constraints are jointly feasible.

        Returns:
            (is_feasible, model_or_none)
        """
        if not constraints:
            return True, {}

        # Quick bound propagation check
        bounds = dict(variable_bounds) if variable_bounds else {}
        bounds, conflict = self._bound_prop.propagate(constraints, bounds)
        if conflict:
            return False, None

        simplex = SimplexMethod(self.tolerance)
        all_vars: Set[str] = set()
        for c in constraints:
            all_vars.update(c.variables)

        for v in sorted(all_vars):
            lb, ub = bounds.get(v, (None, None))
            simplex.add_variable(v, lower=lb, upper=ub)

        for c in constraints:
            simplex.add_constraint(c)

        return simplex.solve()

    def check_incremental(
        self,
        base_constraints: List[LinearConstraint],
        new_constraint: LinearConstraint,
        cached_model: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Incrementally check feasibility after adding a constraint."""
        all_constraints = list(base_constraints) + [new_constraint]
        return self.check(all_constraints)


# ---------------------------------------------------------------------------
# ModelGeneration
# ---------------------------------------------------------------------------

class ModelGeneration:
    """Generate satisfying assignments from feasible constraint sets."""

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance

    def generate(
        self,
        constraints: List[LinearConstraint],
        variable_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        objective: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Generate a satisfying assignment, optionally optimizing an objective.

        Args:
            constraints: Linear constraints.
            variable_bounds: Optional per-variable bounds.
            objective: Optional linear objective to maximize (coefficients).

        Returns:
            Assignment dict or None if infeasible.
        """
        checker = FeasibilityChecker(self.tolerance)
        feasible, model = checker.check(constraints, variable_bounds)
        if not feasible or model is None:
            return None

        if objective is None:
            return model

        # Simple optimization: try pushing objective variables toward bounds
        improved = dict(model)
        for var, coeff in sorted(objective.items(), key=lambda x: -abs(x[1])):
            if var not in improved:
                continue
            if coeff > 0:
                # Try increasing
                test_val = improved[var] + abs(coeff)
                improved[var] = test_val
                if not all(c.is_satisfied(improved, self.tolerance) for c in constraints):
                    improved[var] = model[var]
            else:
                test_val = improved[var] - abs(coeff)
                improved[var] = test_val
                if not all(c.is_satisfied(improved, self.tolerance) for c in constraints):
                    improved[var] = model[var]

        return improved

    def generate_interior_point(
        self,
        constraints: List[LinearConstraint],
    ) -> Optional[Dict[str, float]]:
        """Generate a strictly interior point if possible."""
        model = self.generate(constraints)
        if model is None:
            return None
        return model


# ---------------------------------------------------------------------------
# ConflictExplanation
# ---------------------------------------------------------------------------

class ConflictExplanation:
    """Extract minimal conflict sets from infeasible constraint sets.

    Given an infeasible set of constraints, finds a minimal subset
    that is still infeasible (an unsat core).
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self._checker = FeasibilityChecker(tolerance)

    def explain(
        self,
        constraints: List[LinearConstraint],
        variable_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[int]:
        """Find a minimal infeasible subset (unsat core).

        Returns:
            List of constraint indices forming a minimal infeasible subset.
        """
        feasible, _ = self._checker.check(constraints, variable_bounds)
        if feasible:
            return []  # not infeasible

        # Deletion-based minimization
        core = list(range(len(constraints)))
        for i in range(len(constraints)):
            candidate = [j for j in core if j != i]
            if not candidate:
                continue
            sub_constraints = [constraints[j] for j in candidate]
            f, _ = self._checker.check(sub_constraints, variable_bounds)
            if not f:
                core = candidate

        return core

    def explain_from_simplex(
        self,
        simplex: SimplexMethod,
    ) -> List[int]:
        """Extract conflict from a failed simplex run."""
        return simplex.get_conflict_indices()


# ---------------------------------------------------------------------------
# CuttingPlaneSolver
# ---------------------------------------------------------------------------

class CuttingPlaneSolver:
    """Cutting plane proofs of infeasibility.

    Uses Gomory-style cutting planes and bound tightening
    to prove infeasibility of linear constraint systems.
    """

    def __init__(self, tolerance: float = 1e-10, max_rounds: int = 100) -> None:
        self.tolerance = tolerance
        self.max_rounds = max_rounds

    def prove_infeasible(
        self,
        constraints: List[LinearConstraint],
        variable_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Tuple[bool, List[LinearConstraint]]:
        """Attempt to prove infeasibility via cutting planes.

        Returns:
            (proved_infeasible, cutting_planes_used)
        """
        cuts: List[LinearConstraint] = []
        current = list(constraints)
        checker = FeasibilityChecker(self.tolerance)

        for _ in range(self.max_rounds):
            feasible, model = checker.check(current, variable_bounds)
            if not feasible:
                return True, cuts
            if model is None:
                return True, cuts

            # Try to derive a cutting plane
            new_cut = self._derive_cut(current, model)
            if new_cut is None:
                return False, cuts

            cuts.append(new_cut)
            current.append(new_cut)

        return False, cuts

    def _derive_cut(
        self,
        constraints: List[LinearConstraint],
        model: Dict[str, float],
    ) -> Optional[LinearConstraint]:
        """Derive a Gomory-style cutting plane from the current solution.

        For LRA this is primarily bound tightening.
        """
        # Simple bound tightening: find near-tight constraints and derive implied bounds
        prop = BoundPropagation(self.tolerance)
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for var, val in model.items():
            bounds[var] = (None, None)

        new_bounds, conflict = prop.propagate(constraints, bounds)
        if conflict:
            return LinearConstraint({}, ConstraintOp.LEQ, -1.0, "cut_conflict")

        # Check if any bound can be tightened beyond model value
        for var, (lb, ub) in new_bounds.items():
            if var not in model:
                continue
            val = model[var]
            if lb is not None and val < lb - self.tolerance:
                return LinearConstraint({var: 1.0}, ConstraintOp.GEQ, lb, f"cut_lb_{var}")
            if ub is not None and val > ub + self.tolerance:
                return LinearConstraint({var: 1.0}, ConstraintOp.LEQ, ub, f"cut_ub_{var}")

        return None


# ---------------------------------------------------------------------------
# LinearArithmeticSolver — main theory solver for DPLL(T)
# ---------------------------------------------------------------------------

class LinearArithmeticSolver:
    """Decision procedure for the theory of linear real arithmetic.

    Implements the TheorySolverProtocol for use in DPLL(T).
    Maintains an incremental state with push/pop support.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance
        self._constraints: List[LinearConstraint] = []
        self._constraint_stack: List[int] = []  # stack of constraint list lengths
        self._bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._bounds_stack: List[Dict[str, Tuple[Optional[float], Optional[float]]]] = []
        self._checker = FeasibilityChecker(tolerance)
        self._conflict_explainer = ConflictExplanation(tolerance)
        self._model_gen = ModelGeneration(tolerance)
        self._bound_prop = BoundPropagation(tolerance)
        self._last_model: Optional[Dict[str, float]] = None

    def push(self) -> None:
        """Push a backtracking point."""
        self._constraint_stack.append(len(self._constraints))
        self._bounds_stack.append(dict(self._bounds))

    def pop(self) -> None:
        """Pop to the last backtracking point."""
        if self._constraint_stack:
            n = self._constraint_stack.pop()
            self._constraints = self._constraints[:n]
        if self._bounds_stack:
            self._bounds = self._bounds_stack.pop()
        self._last_model = None

    def assert_literal(self, formula: Formula) -> None:
        """Assert a theory literal."""
        constraint = parse_linear_constraint(formula)
        if constraint is not None:
            self._constraints.append(constraint)

    def assert_constraint(self, constraint: LinearConstraint) -> None:
        """Assert a linear constraint directly."""
        self._constraints.append(constraint)

    def set_bound(self, var: str, lower: Optional[float] = None,
                  upper: Optional[float] = None) -> None:
        """Set bounds on a variable."""
        cur_lb, cur_ub = self._bounds.get(var, (None, None))
        if lower is not None:
            cur_lb = lower if cur_lb is None else max(cur_lb, lower)
        if upper is not None:
            cur_ub = upper if cur_ub is None else min(cur_ub, upper)
        self._bounds[var] = (cur_lb, cur_ub)

    def check_consistency(
        self, literals: List[Formula]
    ) -> Tuple[bool, Optional[List[Formula]]]:
        """Check if the current set of theory literals is consistent.

        Returns:
            (is_consistent, conflict_clause_if_inconsistent)
        """
        # Assert all new literals
        temp_constraints = list(self._constraints)
        for lit in literals:
            c = parse_linear_constraint(lit)
            if c is not None:
                temp_constraints.append(c)

        feasible, model = self._checker.check(temp_constraints, self._bounds)

        if feasible:
            self._last_model = model
            return True, None

        # Extract conflict
        core_indices = self._conflict_explainer.explain(temp_constraints, self._bounds)
        conflict_formulas = []
        for idx in core_indices:
            c = temp_constraints[idx]
            neg = c.negate()
            f = Formula(
                expr=str(neg),
                variables=frozenset(neg.variables),
                formula_type="linear_arithmetic",
            )
            conflict_formulas.append(f)

        if not conflict_formulas:
            f = Formula(expr="false", variables=frozenset(), formula_type="boolean")
            conflict_formulas.append(f)

        return False, conflict_formulas

    def propagate(self, literals: List[Formula]) -> List[Formula]:
        """Theory propagation: derive implied literals from current state.

        Returns:
            List of implied theory literals.
        """
        implied: List[Formula] = []
        constraints = list(self._constraints)
        for lit in literals:
            c = parse_linear_constraint(lit)
            if c is not None:
                constraints.append(c)

        if not constraints:
            return implied

        # Propagate bounds
        bounds = dict(self._bounds)
        new_bounds, conflict = self._bound_prop.propagate(constraints, bounds)
        if conflict:
            return implied

        # Generate implied bound literals
        for var, (lb, ub) in new_bounds.items():
            old_lb, old_ub = self._bounds.get(var, (None, None))
            if lb is not None and (old_lb is None or lb > old_lb + self.tolerance):
                f = Formula(
                    expr=f"{var} >= {lb}",
                    variables=frozenset([var]),
                    formula_type="linear_arithmetic",
                )
                implied.append(f)
            if ub is not None and (old_ub is None or ub < old_ub - self.tolerance):
                f = Formula(
                    expr=f"{var} <= {ub}",
                    variables=frozenset([var]),
                    formula_type="linear_arithmetic",
                )
                implied.append(f)

        return implied

    def get_model(self) -> Optional[Dict[str, float]]:
        """Return the last satisfying assignment, if any."""
        return self._last_model

    def get_conflict_core(self) -> List[int]:
        """Return indices of constraints in the last conflict."""
        return self._conflict_explainer.explain(self._constraints, self._bounds)

    def reset(self) -> None:
        """Reset all state."""
        self._constraints.clear()
        self._constraint_stack.clear()
        self._bounds.clear()
        self._bounds_stack.clear()
        self._last_model = None


__all__ = [
    "ConstraintOp",
    "LinearConstraint",
    "Bound",
    "VariableInfo",
    "parse_linear_constraint",
    "SimplexMethod",
    "BoundPropagation",
    "FeasibilityChecker",
    "ModelGeneration",
    "ConflictExplanation",
    "CuttingPlaneSolver",
    "LinearArithmeticSolver",
]
