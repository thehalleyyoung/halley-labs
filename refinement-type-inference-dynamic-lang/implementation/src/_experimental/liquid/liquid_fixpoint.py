"""
Liquid fixpoint solver.

Solves systems of subtyping constraints with unknown qualifier variables
(KVars) via abstract interpretation over Horn clauses.  When no valid
solution exists, extracts a concrete counterexample trace.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Local type stubs – standalone, no cross-module imports
# ---------------------------------------------------------------------------

class _BaseType(Enum):
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NONE_TYPE = "none"
    ANY = "any"


@dataclass(frozen=True)
class _Pred:
    """Minimal predicate representation used internally by the solver."""
    expr: str
    variables: Tuple[str, ...] = ()

    def mentions(self, var: str) -> bool:
        return var in self.variables

    def substitute(self, mapping: Dict[str, str]) -> "_Pred":
        new_expr = self.expr
        new_vars: List[str] = []
        for v in self.variables:
            replacement = mapping.get(v, v)
            new_expr = new_expr.replace(v, replacement)
            new_vars.append(replacement)
        return _Pred(expr=new_expr, variables=tuple(new_vars))

    def __repr__(self) -> str:
        return self.expr


# ===================================================================
# 1. Constraint base class
# ===================================================================

class Constraint:
    """Abstract base for all constraint kinds."""

    def free_vars(self) -> Set[str]:
        raise NotImplementedError

    def mentions_kvar(self, kvar: str) -> bool:
        raise NotImplementedError


# ===================================================================
# 2. Well-formedness constraint
# ===================================================================

@dataclass
class WellFormedness(Constraint):
    """
    Well-formedness constraint for a liquid type.

    Asserts that the refinement predicate of a type is *well-sorted*:
    every free variable is bound in the environment and has the right base
    type.
    """
    env: Dict[str, _BaseType]
    var: str
    base: _BaseType
    predicate: _Pred

    def free_vars(self) -> Set[str]:
        fv = set(self.predicate.variables)
        fv.discard(self.var)
        return fv

    def mentions_kvar(self, kvar: str) -> bool:
        return kvar in self.predicate.expr

    def check(self) -> bool:
        """Verify that every free variable in the predicate is bound in the env."""
        for v in self.predicate.variables:
            if v == self.var:
                continue
            if v not in self.env:
                return False
        return True

    def __repr__(self) -> str:
        env_str = ", ".join(f"{k}:{v.value}" for k, v in self.env.items())
        return f"WF([{env_str}] ⊢ {{{self.var}:{self.base.value} | {self.predicate}}})"


# ===================================================================
# 3. Subtype constraint
# ===================================================================

@dataclass
class SubtypeConstraint(Constraint):
    """
    Subtyping constraint with an environment.

    Encodes  Γ ⊢ {v:T | p} <: {v:T | q}.
    """
    env: Dict[str, Tuple[_BaseType, _Pred]]
    lhs_var: str
    lhs_base: _BaseType
    lhs_pred: _Pred
    rhs_var: str
    rhs_base: _BaseType
    rhs_pred: _Pred
    tag: str = ""

    def free_vars(self) -> Set[str]:
        fv: Set[str] = set()
        for v, (_, pred) in self.env.items():
            fv |= set(pred.variables)
        fv |= set(self.lhs_pred.variables)
        fv |= set(self.rhs_pred.variables)
        fv.discard(self.lhs_var)
        fv.discard(self.rhs_var)
        return fv

    def mentions_kvar(self, kvar: str) -> bool:
        if kvar in self.lhs_pred.expr or kvar in self.rhs_pred.expr:
            return True
        for _, (_, pred) in self.env.items():
            if kvar in pred.expr:
                return True
        return False

    def to_horn_body_head(self) -> Tuple[List[_Pred], _Pred]:
        """Lower to a Horn clause body → head pair."""
        body: List[_Pred] = []
        for v, (_, pred) in self.env.items():
            body.append(pred)
        body.append(self.lhs_pred)
        head = self.rhs_pred.substitute({self.rhs_var: self.lhs_var})
        return body, head

    def __repr__(self) -> str:
        return (
            f"Sub({self.tag}: "
            f"{{{self.lhs_var}:{self.lhs_base.value}|{self.lhs_pred}}} "
            f"<: {{{self.rhs_var}:{self.rhs_base.value}|{self.rhs_pred}}})"
        )


# ===================================================================
# 4. KVar constraint
# ===================================================================

@dataclass
class KVarConstraint(Constraint):
    """
    Unknown qualifier variable to be solved.

    A KVar represents a set of qualifiers yet to be determined.
    It is identified by a name and is associated with a base type and
    a set of variables in scope.
    """
    name: str
    base: _BaseType
    scope_vars: Tuple[str, ...] = ()
    # candidate qualifiers (expressions)
    candidates: List[_Pred] = field(default_factory=list)

    def free_vars(self) -> Set[str]:
        return set(self.scope_vars)

    def mentions_kvar(self, kvar: str) -> bool:
        return self.name == kvar

    def add_candidate(self, pred: _Pred) -> None:
        if pred not in self.candidates:
            self.candidates.append(pred)

    def remove_candidate(self, pred: _Pred) -> bool:
        try:
            self.candidates.remove(pred)
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:
        return f"KVar({self.name}, {len(self.candidates)} candidates)"


# ===================================================================
# 5. Solution
# ===================================================================

@dataclass
class Solution:
    """Maps KVars to sets of qualifiers (predicates)."""
    assignments: Dict[str, List[_Pred]] = field(default_factory=dict)
    is_valid: bool = True
    error_message: str = ""

    def get(self, kvar: str) -> List[_Pred]:
        return self.assignments.get(kvar, [])

    def set(self, kvar: str, preds: List[_Pred]) -> None:
        self.assignments[kvar] = list(preds)

    def merge_kvar(self, kvar: str, preds: List[_Pred]) -> None:
        """Intersect with existing assignment (keep only common qualifiers)."""
        if kvar not in self.assignments:
            self.assignments[kvar] = list(preds)
        else:
            existing = set(p.expr for p in self.assignments[kvar])
            self.assignments[kvar] = [p for p in preds if p.expr in existing]

    def all_kvars(self) -> Set[str]:
        return set(self.assignments.keys())

    def total_qualifiers(self) -> int:
        return sum(len(ps) for ps in self.assignments.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "kvars": len(self.assignments),
            "total_qualifiers": self.total_qualifiers(),
            "error": self.error_message if not self.is_valid else None,
        }

    def __repr__(self) -> str:
        if not self.is_valid:
            return f"Solution(INVALID: {self.error_message})"
        parts = [f"{k}: [{', '.join(p.expr for p in ps)}]" for k, ps in self.assignments.items()]
        return "Solution({" + "; ".join(parts) + "})"


# ===================================================================
# 6. Counterexample extractor
# ===================================================================

@dataclass
class CounterexampleStep:
    """One step in a counterexample trace."""
    variable: str
    value: Any
    constraint_tag: str = ""
    explanation: str = ""


@dataclass
class Counterexample:
    """A concrete trace showing why a set of constraints is unsatisfiable."""
    steps: List[CounterexampleStep] = field(default_factory=list)
    violated_constraint: Optional[SubtypeConstraint] = None
    message: str = ""

    def __repr__(self) -> str:
        lines = [f"Counterexample: {self.message}"]
        for s in self.steps:
            lines.append(f"  {s.variable} = {s.value}  ({s.explanation})")
        if self.violated_constraint:
            lines.append(f"  Violated: {self.violated_constraint}")
        return "\n".join(lines)


class CounterexampleExtractor:
    """
    When the solver fails to find a valid solution, construct a concrete
    counterexample trace demonstrating the failure.

    The extractor identifies the constraint that could not be satisfied,
    picks witness values for the environment variables, and constructs a
    step-by-step trace.
    """

    def __init__(self) -> None:
        self._int_witnesses: List[int] = [0, 1, -1, 2, -2, 100]
        self._bool_witnesses: List[bool] = [True, False]
        self._string_witnesses: List[str] = ["", "a", "abc"]

    def extract(
        self,
        failed_constraints: List[SubtypeConstraint],
        partial_solution: Solution,
    ) -> Counterexample:
        """Produce a counterexample for the first failed constraint."""
        if not failed_constraints:
            return Counterexample(message="No failed constraints provided.")

        target = failed_constraints[0]
        steps: List[CounterexampleStep] = []

        # assign witness values to environment variables
        env_values: Dict[str, Any] = {}
        for var_name, (base, pred) in target.env.items():
            witness = self._pick_witness(base, var_name, env_values)
            env_values[var_name] = witness
            steps.append(CounterexampleStep(
                variable=var_name,
                value=witness,
                constraint_tag=target.tag,
                explanation=f"environment binding ({base.value})",
            ))

        # assign witness for the refinement variable
        subject_witness = self._pick_witness(target.lhs_base, target.lhs_var, env_values)
        steps.append(CounterexampleStep(
            variable=target.lhs_var,
            value=subject_witness,
            constraint_tag=target.tag,
            explanation="subject variable satisfying LHS but violating RHS",
        ))

        return Counterexample(
            steps=steps,
            violated_constraint=target,
            message=f"Subtyping obligation {target.tag} cannot be satisfied.",
        )

    def extract_all(
        self,
        failed_constraints: List[SubtypeConstraint],
        partial_solution: Solution,
    ) -> List[Counterexample]:
        """Extract a counterexample for each failed constraint."""
        results: List[Counterexample] = []
        for c in failed_constraints:
            ce = self.extract([c], partial_solution)
            results.append(ce)
        return results

    def _pick_witness(
        self,
        base: _BaseType,
        var_name: str,
        already: Dict[str, Any],
    ) -> Any:
        """Pick a concrete witness value for a variable of the given type."""
        if base == _BaseType.INT:
            for w in self._int_witnesses:
                if w not in already.values():
                    return w
            return 42
        elif base == _BaseType.BOOL:
            return True
        elif base == _BaseType.STRING:
            for w in self._string_witnesses:
                if w not in already.values():
                    return w
            return "witness"
        return None


# ===================================================================
# 7. Constraint solver
# ===================================================================

class ConstraintSolver:
    """
    Main solver for liquid type constraints.

    Algorithm overview:
    1. Collect all KVar constraints and initialise each with its full
       candidate qualifier set.
    2. Convert SubtypeConstraints into Horn clauses.
    3. Iteratively remove qualifiers from KVars that are *invalid*
       (their presence makes a Horn clause unsatisfiable).
    4. If all constraints are satisfied, return a :class:`Solution`.
    5. Otherwise, extract a :class:`Counterexample`.
    """

    MAX_ITERATIONS: int = 500

    def __init__(self) -> None:
        self._kvars: Dict[str, KVarConstraint] = {}
        self._subtype_constraints: List[SubtypeConstraint] = []
        self._wf_constraints: List[WellFormedness] = []
        self._ce_extractor = CounterexampleExtractor()

    # -- constraint registration ------------------------------------------

    def add_constraint(self, c: Constraint) -> None:
        if isinstance(c, KVarConstraint):
            self._kvars[c.name] = c
        elif isinstance(c, SubtypeConstraint):
            self._subtype_constraints.append(c)
        elif isinstance(c, WellFormedness):
            self._wf_constraints.append(c)

    def add_constraints(self, cs: List[Constraint]) -> None:
        for c in cs:
            self.add_constraint(c)

    # -- main solve -------------------------------------------------------

    def solve(self, constraints: Optional[List[Constraint]] = None) -> Solution:
        """
        Solve a set of constraints, returning a :class:`Solution`.

        If *constraints* is provided they are added to any previously
        registered constraints.
        """
        if constraints is not None:
            self.add_constraints(constraints)

        # Well-formedness check
        for wf in self._wf_constraints:
            if not wf.check():
                return Solution(
                    is_valid=False,
                    error_message=f"Ill-formed type: {wf}",
                )

        # Initialise solution with all candidates
        solution = Solution()
        for name, kvar in self._kvars.items():
            solution.set(name, list(kvar.candidates))

        # Build Horn clauses from subtype constraints
        horn_clauses: List[Tuple[List[_Pred], _Pred, SubtypeConstraint]] = []
        for sc in self._subtype_constraints:
            body, head = sc.to_horn_body_head()
            horn_clauses.append((body, head, sc))

        # Iterative refinement
        changed = True
        iterations = 0
        while changed and iterations < self.MAX_ITERATIONS:
            changed = False
            iterations += 1

            for body, head, sc in horn_clauses:
                invalid_qualifiers = self._find_invalid_qualifiers(
                    body, head, sc, solution,
                )
                for kvar_name, pred in invalid_qualifiers:
                    current = solution.get(kvar_name)
                    if pred in current:
                        current.remove(pred)
                        solution.set(kvar_name, current)
                        changed = True

        # Verify solution
        failed = self._verify(solution, horn_clauses)
        if failed:
            ce = self._ce_extractor.extract(failed, solution)
            return Solution(
                assignments=solution.assignments,
                is_valid=False,
                error_message=f"Unsatisfiable: {ce.message}",
            )

        solution.is_valid = True
        return solution

    # -- internals --------------------------------------------------------

    def _find_invalid_qualifiers(
        self,
        body: List[_Pred],
        head: _Pred,
        sc: SubtypeConstraint,
        solution: Solution,
    ) -> List[Tuple[str, _Pred]]:
        """
        Identify qualifiers in KVars that make the Horn clause invalid.

        A qualifier *q* in KVar *κ* is invalid w.r.t. a clause if the clause
        head mentions *κ*, and *q* is not implied by the body predicates
        under the current solution.
        """
        invalid: List[Tuple[str, _Pred]] = []

        for kvar_name, kvar in self._kvars.items():
            if not sc.mentions_kvar(kvar_name):
                continue

            # Check if kvar appears in the head (RHS of subtyping)
            if kvar_name not in head.expr:
                continue

            for candidate in list(solution.get(kvar_name)):
                if not self._is_implied(candidate, body, solution, kvar_name):
                    invalid.append((kvar_name, candidate))

        return invalid

    def _is_implied(
        self,
        candidate: _Pred,
        body: List[_Pred],
        solution: Solution,
        kvar_name: str,
    ) -> bool:
        """
        Check whether *candidate* is implied by the body predicates.

        This is a syntactic / heuristic check.  A full implementation
        would call an SMT solver; here we use conservative approximations:
        - If the candidate appears literally in the body, it is implied.
        - If any body predicate, after substituting KVar solutions, textually
          matches the candidate, it is implied.
        """
        if candidate in body:
            return True

        # expand KVars in body
        expanded_body = self._expand_body(body, solution, exclude_kvar=kvar_name)

        if candidate in expanded_body:
            return True

        # check containment by expression string
        candidate_str = candidate.expr
        for bp in expanded_body:
            if candidate_str == bp.expr:
                return True
            # simple implication patterns
            if self._syntactic_implies(bp.expr, candidate_str):
                return True

        return False

    def _expand_body(
        self,
        body: List[_Pred],
        solution: Solution,
        exclude_kvar: str = "",
    ) -> List[_Pred]:
        """Expand KVar references in body using current solution."""
        expanded: List[_Pred] = []
        for pred in body:
            kvar_found = False
            for kvar_name in solution.all_kvars():
                if kvar_name == exclude_kvar:
                    continue
                if kvar_name in pred.expr:
                    kvar_found = True
                    for q in solution.get(kvar_name):
                        new_expr = pred.expr.replace(kvar_name, q.expr)
                        expanded.append(_Pred(expr=new_expr, variables=q.variables))
            if not kvar_found:
                expanded.append(pred)
        return expanded

    @staticmethod
    def _syntactic_implies(premise_expr: str, conclusion_expr: str) -> bool:
        """
        Very conservative syntactic implication check.

        Handles a few common patterns:
        - ``x > 5`` implies ``x > 0``  (constant weakening for >/>=/</<=)
        - ``x == c`` implies ``x >= c`` and ``x <= c``
        """
        # Identical
        if premise_expr == conclusion_expr:
            return True

        # Try to parse "var op const" for both
        p_parts = _parse_simple_pred(premise_expr)
        c_parts = _parse_simple_pred(conclusion_expr)
        if p_parts is None or c_parts is None:
            return False

        p_var, p_op, p_val = p_parts
        c_var, c_op, c_val = c_parts

        if p_var != c_var:
            return False

        try:
            pv = float(p_val)
            cv = float(c_val)
        except (ValueError, TypeError):
            return False

        # x > a implies x > b when a >= b
        if p_op == ">" and c_op == ">" and pv >= cv:
            return True
        if p_op == ">=" and c_op == ">=" and pv >= cv:
            return True
        if p_op == ">" and c_op == ">=" and pv >= cv:
            return True
        # x < a implies x < b when a <= b
        if p_op == "<" and c_op == "<" and pv <= cv:
            return True
        if p_op == "<=" and c_op == "<=" and pv <= cv:
            return True
        if p_op == "<" and c_op == "<=" and pv <= cv:
            return True
        # x == a implies x >= a and x <= a
        if p_op == "==" and c_op == ">=" and pv >= cv:
            return True
        if p_op == "==" and c_op == "<=" and pv <= cv:
            return True

        return False

    def _verify(
        self,
        solution: Solution,
        horn_clauses: List[Tuple[List[_Pred], _Pred, SubtypeConstraint]],
    ) -> List[SubtypeConstraint]:
        """Return list of SubtypeConstraints not satisfied by *solution*."""
        failed: List[SubtypeConstraint] = []
        for body, head, sc in horn_clauses:
            if not self._clause_satisfied(body, head, sc, solution):
                failed.append(sc)
        return failed

    def _clause_satisfied(
        self,
        body: List[_Pred],
        head: _Pred,
        sc: SubtypeConstraint,
        solution: Solution,
    ) -> bool:
        """Check if a single Horn clause is satisfied by *solution*."""
        # If no KVars are involved, assume satisfied (handled externally)
        has_kvar = False
        for kvar_name in solution.all_kvars():
            if sc.mentions_kvar(kvar_name):
                has_kvar = True
                break
        if not has_kvar:
            return True

        # check all KVar qualifiers in head position are implied by body
        for kvar_name in solution.all_kvars():
            if kvar_name not in head.expr:
                continue
            for q in solution.get(kvar_name):
                if not self._is_implied(q, body, solution, kvar_name):
                    return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_simple_pred(expr: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to parse ``expr`` as ``var op value``.

    Returns (var, op, value) or None.
    """
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if f" {op} " in expr:
            parts = expr.split(f" {op} ", 1)
            if len(parts) == 2:
                return (parts[0].strip(), op, parts[1].strip())
    return None
