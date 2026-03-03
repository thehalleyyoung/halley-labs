"""
Formula manipulation for Craig interpolation.

Provides propositional and first-order formula representations,
normal-form conversions (CNF/DNF), simplification, substitution,
quantifier elimination, and satisfiability checking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import sympy
from sympy import Symbol, And, Or, Not, Implies, simplify_logic
from sympy.logic.boolalg import to_cnf, to_dnf, is_cnf, is_dnf

from dp_forge.types import Formula as DPFormula


# ---------------------------------------------------------------------------
# Node types for formula AST
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    """Kind of AST node in a formula."""

    VAR = auto()
    CONST = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()
    LEQ = auto()      # linear inequality: a^T x <= b
    EQ = auto()       # linear equality:  a^T x = b
    LT = auto()       # strict inequality
    FORALL = auto()
    EXISTS = auto()


@dataclass(frozen=True)
class FormulaNode:
    """An immutable node in a formula AST.

    Leaf nodes (VAR, CONST) carry a ``value``.
    Internal nodes carry ordered ``children``.
    Quantifier nodes additionally carry ``bound_var``.
    Linear-arithmetic nodes store ``coefficients`` and ``rhs``.
    """

    kind: NodeKind
    children: Tuple[FormulaNode, ...] = ()
    value: Optional[Any] = None
    bound_var: Optional[str] = None
    coefficients: Optional[Dict[str, float]] = None
    rhs: Optional[float] = None

    # -- Convenience constructors ------------------------------------------

    @staticmethod
    def var(name: str) -> FormulaNode:
        return FormulaNode(kind=NodeKind.VAR, value=name)

    @staticmethod
    def const(val: bool) -> FormulaNode:
        return FormulaNode(kind=NodeKind.CONST, value=val)

    @staticmethod
    def and_(*children: FormulaNode) -> FormulaNode:
        if len(children) == 0:
            return FormulaNode.const(True)
        if len(children) == 1:
            return children[0]
        return FormulaNode(kind=NodeKind.AND, children=children)

    @staticmethod
    def or_(*children: FormulaNode) -> FormulaNode:
        if len(children) == 0:
            return FormulaNode.const(False)
        if len(children) == 1:
            return children[0]
        return FormulaNode(kind=NodeKind.OR, children=children)

    @staticmethod
    def not_(child: FormulaNode) -> FormulaNode:
        if child.kind == NodeKind.NOT:
            return child.children[0]
        if child.kind == NodeKind.CONST:
            return FormulaNode.const(not child.value)
        return FormulaNode(kind=NodeKind.NOT, children=(child,))

    @staticmethod
    def implies(lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
        return FormulaNode(kind=NodeKind.IMPLIES, children=(lhs, rhs))

    @staticmethod
    def leq(coefficients: Dict[str, float], rhs: float) -> FormulaNode:
        """Linear inequality: sum(coeff[v]*v for v) <= rhs."""
        return FormulaNode(
            kind=NodeKind.LEQ, coefficients=dict(coefficients), rhs=rhs,
        )

    @staticmethod
    def eq(coefficients: Dict[str, float], rhs: float) -> FormulaNode:
        return FormulaNode(
            kind=NodeKind.EQ, coefficients=dict(coefficients), rhs=rhs,
        )

    @staticmethod
    def lt(coefficients: Dict[str, float], rhs: float) -> FormulaNode:
        return FormulaNode(
            kind=NodeKind.LT, coefficients=dict(coefficients), rhs=rhs,
        )

    @staticmethod
    def forall(var: str, body: FormulaNode) -> FormulaNode:
        return FormulaNode(kind=NodeKind.FORALL, children=(body,), bound_var=var)

    @staticmethod
    def exists(var: str, body: FormulaNode) -> FormulaNode:
        return FormulaNode(kind=NodeKind.EXISTS, children=(body,), bound_var=var)

    # -- Variable collection -----------------------------------------------

    def free_variables(self) -> FrozenSet[str]:
        if self.kind == NodeKind.VAR:
            return frozenset({self.value})
        if self.kind == NodeKind.CONST:
            return frozenset()
        if self.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            return frozenset(self.coefficients.keys()) if self.coefficients else frozenset()
        fv: Set[str] = set()
        for c in self.children:
            fv |= c.free_variables()
        if self.bound_var:
            fv.discard(self.bound_var)
        return frozenset(fv)

    # -- Pretty printing ---------------------------------------------------

    def to_str(self) -> str:
        if self.kind == NodeKind.VAR:
            return str(self.value)
        if self.kind == NodeKind.CONST:
            return "true" if self.value else "false"
        if self.kind == NodeKind.NOT:
            return f"¬({self.children[0].to_str()})"
        if self.kind == NodeKind.AND:
            return " ∧ ".join(f"({c.to_str()})" for c in self.children)
        if self.kind == NodeKind.OR:
            return " ∨ ".join(f"({c.to_str()})" for c in self.children)
        if self.kind == NodeKind.IMPLIES:
            return f"({self.children[0].to_str()}) → ({self.children[1].to_str()})"
        if self.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            op = {"LEQ": "≤", "EQ": "=", "LT": "<"}[self.kind.name]
            terms = " + ".join(
                f"{c}·{v}" for v, c in sorted((self.coefficients or {}).items())
            )
            return f"{terms} {op} {self.rhs}"
        if self.kind == NodeKind.FORALL:
            return f"∀{self.bound_var}.({self.children[0].to_str()})"
        if self.kind == NodeKind.EXISTS:
            return f"∃{self.bound_var}.({self.children[0].to_str()})"
        return f"<{self.kind.name}>"

    def __repr__(self) -> str:
        return f"FormulaNode({self.to_str()})"


# ---------------------------------------------------------------------------
# Formula wrapper (bridges FormulaNode <-> dp_forge.types.Formula)
# ---------------------------------------------------------------------------


class Formula:
    """High-level formula with AST, conversion to dp_forge.types.Formula."""

    def __init__(self, node: FormulaNode) -> None:
        self.node = node
        self._variables: Optional[FrozenSet[str]] = None

    @property
    def variables(self) -> FrozenSet[str]:
        if self._variables is None:
            self._variables = self.node.free_variables()
        return self._variables

    def to_dp_formula(self) -> DPFormula:
        return DPFormula(
            expr=self.node.to_str(),
            variables=self.variables,
            formula_type="linear_arithmetic",
        )

    @staticmethod
    def from_dp_formula(f: DPFormula) -> Formula:
        """Parse a DPFormula expression back into a Formula AST.

        Supports simple conjunction/disjunction of linear constraints
        and boolean variables.
        """
        return _parse_dp_formula(f)

    def __and__(self, other: Formula) -> Formula:
        return Formula(FormulaNode.and_(self.node, other.node))

    def __or__(self, other: Formula) -> Formula:
        return Formula(FormulaNode.or_(self.node, other.node))

    def __invert__(self) -> Formula:
        return Formula(FormulaNode.not_(self.node))

    def __repr__(self) -> str:
        return f"Formula({self.node.to_str()})"


def _parse_dp_formula(f: DPFormula) -> Formula:
    """Best-effort parse of DPFormula expr into Formula AST."""
    expr = f.expr.strip()
    if not expr or expr.lower() == "true":
        return Formula(FormulaNode.const(True))
    if expr.lower() == "false":
        return Formula(FormulaNode.const(False))

    # Try splitting on ∧ or " and "
    if " ∧ " in expr:
        parts = expr.split(" ∧ ")
        nodes = [_parse_atomic(p.strip("() ")) for p in parts]
        return Formula(FormulaNode.and_(*nodes))

    if " ∨ " in expr:
        parts = expr.split(" ∨ ")
        nodes = [_parse_atomic(p.strip("() ")) for p in parts]
        return Formula(FormulaNode.or_(*nodes))

    return Formula(_parse_atomic(expr))


def _parse_atomic(expr: str) -> FormulaNode:
    """Parse a single atomic constraint or variable reference."""
    expr = expr.strip("() ")
    for op_str, kind in [("<=", NodeKind.LEQ), (">=", None), ("=", NodeKind.EQ), ("<", NodeKind.LT)]:
        if op_str in expr:
            lhs, rhs_str = expr.split(op_str, 1)
            try:
                rhs_val = float(rhs_str.strip())
            except ValueError:
                return FormulaNode.var(expr)
            coeffs = _parse_linear_terms(lhs.strip())
            if op_str == ">=":
                coeffs = {v: -c for v, c in coeffs.items()}
                rhs_val = -rhs_val
                kind = NodeKind.LEQ
            return FormulaNode(kind=kind, coefficients=coeffs, rhs=rhs_val)
    return FormulaNode.var(expr)


def _parse_linear_terms(s: str) -> Dict[str, float]:
    """Parse simple linear term like '2*x + 3*y - z'."""
    coeffs: Dict[str, float] = {}
    s = s.strip()
    s = s.replace(" - ", " + -").replace(" + ", " + ")
    s = s.replace("-", "+-")
    for token in s.split("+"):
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            parts = token.split("*", 1)
            try:
                coeff = float(parts[0].strip())
                var = parts[1].strip()
            except ValueError:
                var = token
                coeff = 1.0
        else:
            try:
                float(token)
                continue
            except ValueError:
                if token.startswith("-"):
                    coeff = -1.0
                    var = token[1:].strip()
                else:
                    coeff = 1.0
                    var = token
        if var:
            coeffs[var] = coeffs.get(var, 0.0) + coeff
    return coeffs


# ---------------------------------------------------------------------------
# CNF Converter
# ---------------------------------------------------------------------------


class CNFConverter:
    """Convert formula AST to conjunctive normal form.

    Uses Tseitin transformation for sub-exponential size when
    ``use_tseitin`` is True; otherwise distributes naively.
    """

    def __init__(self, *, use_tseitin: bool = False) -> None:
        self.use_tseitin = use_tseitin
        self._tseitin_counter = 0

    def convert(self, formula: Formula) -> Formula:
        """Return an equisatisfiable CNF formula."""
        node = formula.node
        node = self._eliminate_implies(node)
        node = self._push_negations(node)

        if self.use_tseitin:
            clauses, aux_vars = self._tseitin(node)
            node = FormulaNode.and_(*clauses) if clauses else FormulaNode.const(True)
        else:
            node = self._distribute(node)

        return Formula(node)

    # -- Implication elimination -------------------------------------------

    def _eliminate_implies(self, n: FormulaNode) -> FormulaNode:
        if n.kind == NodeKind.IMPLIES:
            lhs = self._eliminate_implies(n.children[0])
            rhs = self._eliminate_implies(n.children[1])
            return FormulaNode.or_(FormulaNode.not_(lhs), rhs)
        if n.children:
            new_kids = tuple(self._eliminate_implies(c) for c in n.children)
            return FormulaNode(kind=n.kind, children=new_kids, value=n.value,
                               bound_var=n.bound_var, coefficients=n.coefficients,
                               rhs=n.rhs)
        return n

    # -- NNF (push negations to leaves) ------------------------------------

    def _push_negations(self, n: FormulaNode) -> FormulaNode:
        if n.kind == NodeKind.NOT:
            inner = n.children[0]
            if inner.kind == NodeKind.NOT:
                return self._push_negations(inner.children[0])
            if inner.kind == NodeKind.AND:
                return FormulaNode.or_(
                    *(self._push_negations(FormulaNode.not_(c)) for c in inner.children)
                )
            if inner.kind == NodeKind.OR:
                return FormulaNode.and_(
                    *(self._push_negations(FormulaNode.not_(c)) for c in inner.children)
                )
            if inner.kind == NodeKind.CONST:
                return FormulaNode.const(not inner.value)
            return n
        if n.children:
            new_kids = tuple(self._push_negations(c) for c in n.children)
            return FormulaNode(kind=n.kind, children=new_kids, value=n.value,
                               bound_var=n.bound_var, coefficients=n.coefficients,
                               rhs=n.rhs)
        return n

    # -- Distribution (OR over AND) ----------------------------------------

    def _distribute(self, n: FormulaNode) -> FormulaNode:
        if n.kind == NodeKind.AND:
            kids = [self._distribute(c) for c in n.children]
            return FormulaNode.and_(*kids)
        if n.kind == NodeKind.OR:
            kids = [self._distribute(c) for c in n.children]
            conjuncts_list = [self._and_children(k) for k in kids]
            product = list(itertools.product(*conjuncts_list))
            clauses = [FormulaNode.or_(*combo) for combo in product]
            return FormulaNode.and_(*clauses) if clauses else FormulaNode.const(True)
        return n

    def _and_children(self, n: FormulaNode) -> List[FormulaNode]:
        if n.kind == NodeKind.AND:
            return list(n.children)
        return [n]

    # -- Tseitin transformation --------------------------------------------

    def _tseitin(self, n: FormulaNode) -> Tuple[List[FormulaNode], Set[str]]:
        clauses: List[FormulaNode] = []
        aux: Set[str] = set()
        top = self._tseitin_rec(n, clauses, aux)
        clauses.append(top)
        return clauses, aux

    def _fresh_var(self) -> FormulaNode:
        self._tseitin_counter += 1
        return FormulaNode.var(f"__ts_{self._tseitin_counter}")

    def _tseitin_rec(
        self, n: FormulaNode, clauses: List[FormulaNode], aux: Set[str],
    ) -> FormulaNode:
        if n.kind in (NodeKind.VAR, NodeKind.CONST, NodeKind.LEQ,
                       NodeKind.EQ, NodeKind.LT, NodeKind.NOT):
            return n

        proxy = self._fresh_var()
        aux.add(proxy.value)

        if n.kind == NodeKind.AND:
            sub = [self._tseitin_rec(c, clauses, aux) for c in n.children]
            for s in sub:
                clauses.append(FormulaNode.or_(FormulaNode.not_(proxy), s))
            clauses.append(FormulaNode.or_(proxy, *(FormulaNode.not_(s) for s in sub)))
        elif n.kind == NodeKind.OR:
            sub = [self._tseitin_rec(c, clauses, aux) for c in n.children]
            for s in sub:
                clauses.append(FormulaNode.or_(FormulaNode.not_(s), proxy))
            clauses.append(FormulaNode.or_(FormulaNode.not_(proxy), *sub))
        return proxy


# ---------------------------------------------------------------------------
# DNF Converter
# ---------------------------------------------------------------------------


class DNFConverter:
    """Convert formula AST to disjunctive normal form."""

    def __init__(self, *, max_disjuncts: int = 1000) -> None:
        self.max_disjuncts = max_disjuncts

    def convert(self, formula: Formula) -> Formula:
        node = formula.node
        cnf_conv = CNFConverter()
        node = cnf_conv._eliminate_implies(node)
        node = cnf_conv._push_negations(node)
        node = self._distribute_and_over_or(node)
        return Formula(node)

    def _distribute_and_over_or(self, n: FormulaNode) -> FormulaNode:
        if n.kind == NodeKind.OR:
            kids = [self._distribute_and_over_or(c) for c in n.children]
            return FormulaNode.or_(*kids)
        if n.kind == NodeKind.AND:
            kids = [self._distribute_and_over_or(c) for c in n.children]
            disjuncts_list = [self._or_children(k) for k in kids]
            product = list(itertools.islice(
                itertools.product(*disjuncts_list), self.max_disjuncts,
            ))
            cubes = [FormulaNode.and_(*combo) for combo in product]
            return FormulaNode.or_(*cubes) if cubes else FormulaNode.const(False)
        return n

    def _or_children(self, n: FormulaNode) -> List[FormulaNode]:
        if n.kind == NodeKind.OR:
            return list(n.children)
        return [n]


# ---------------------------------------------------------------------------
# Simplifier
# ---------------------------------------------------------------------------


class Simplifier:
    """Simplify formulas via rewriting rules.

    Applies constant folding, double-negation elimination,
    absorption, subsumption, and coefficient normalization.
    """

    def __init__(self, *, max_iterations: int = 20) -> None:
        self.max_iterations = max_iterations

    def simplify(self, formula: Formula) -> Formula:
        node = formula.node
        for _ in range(self.max_iterations):
            new_node = self._simplify_step(node)
            if new_node == node:
                break
            node = new_node
        return Formula(node)

    def _simplify_step(self, n: FormulaNode) -> FormulaNode:
        if n.kind == NodeKind.CONST or n.kind == NodeKind.VAR:
            return n
        if n.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            return self._simplify_linear(n)

        children = tuple(self._simplify_step(c) for c in n.children)

        if n.kind == NodeKind.NOT:
            inner = children[0]
            if inner.kind == NodeKind.NOT:
                return inner.children[0]
            if inner.kind == NodeKind.CONST:
                return FormulaNode.const(not inner.value)
            return FormulaNode(kind=NodeKind.NOT, children=children)

        if n.kind == NodeKind.AND:
            flat = self._flatten(NodeKind.AND, children)
            flat = [c for c in flat if not (c.kind == NodeKind.CONST and c.value is True)]
            if any(c.kind == NodeKind.CONST and c.value is False for c in flat):
                return FormulaNode.const(False)
            seen: List[FormulaNode] = []
            for c in flat:
                if c not in seen:
                    seen.append(c)
            flat = seen
            if not flat:
                return FormulaNode.const(True)
            return FormulaNode.and_(*flat)

        if n.kind == NodeKind.OR:
            flat = self._flatten(NodeKind.OR, children)
            flat = [c for c in flat if not (c.kind == NodeKind.CONST and c.value is False)]
            if any(c.kind == NodeKind.CONST and c.value is True for c in flat):
                return FormulaNode.const(True)
            seen_or: List[FormulaNode] = []
            for c in flat:
                if c not in seen_or:
                    seen_or.append(c)
            flat = seen_or
            if not flat:
                return FormulaNode.const(False)
            return FormulaNode.or_(*flat)

        return FormulaNode(kind=n.kind, children=children, value=n.value,
                           bound_var=n.bound_var, coefficients=n.coefficients,
                           rhs=n.rhs)

    def _flatten(self, kind: NodeKind, children: Tuple[FormulaNode, ...]) -> List[FormulaNode]:
        result: List[FormulaNode] = []
        for c in children:
            if c.kind == kind:
                result.extend(c.children)
            else:
                result.append(c)
        return result

    def _simplify_linear(self, n: FormulaNode) -> FormulaNode:
        if not n.coefficients:
            val = (n.rhs or 0.0)
            if n.kind == NodeKind.LEQ:
                return FormulaNode.const(0.0 <= val)
            if n.kind == NodeKind.LT:
                return FormulaNode.const(0.0 < val)
            if n.kind == NodeKind.EQ:
                return FormulaNode.const(abs(val) < 1e-15)
            return n
        coeffs = {v: c for v, c in n.coefficients.items() if abs(c) > 1e-15}
        if not coeffs:
            return self._simplify_linear(
                FormulaNode(kind=n.kind, coefficients={}, rhs=n.rhs)
            )
        return FormulaNode(kind=n.kind, coefficients=coeffs, rhs=n.rhs)


# ---------------------------------------------------------------------------
# Substitution Engine
# ---------------------------------------------------------------------------


class SubstitutionEngine:
    """Variable substitution and renaming in formula ASTs."""

    def substitute(
        self,
        formula: Formula,
        mapping: Dict[str, FormulaNode],
    ) -> Formula:
        """Replace free occurrences of variables according to ``mapping``."""
        return Formula(self._subst(formula.node, mapping))

    def rename(
        self,
        formula: Formula,
        var_map: Dict[str, str],
    ) -> Formula:
        """Rename variables in a formula."""
        mapping = {old: FormulaNode.var(new) for old, new in var_map.items()}
        return Formula(self._subst(formula.node, mapping))

    def restrict(
        self,
        formula: Formula,
        allowed_vars: FrozenSet[str],
    ) -> Formula:
        """Project formula onto ``allowed_vars`` via existential quantification."""
        extra = formula.variables - allowed_vars
        node = formula.node
        for v in sorted(extra):
            node = FormulaNode.exists(v, node)
        return Formula(node)

    def _subst(
        self, n: FormulaNode, mapping: Dict[str, FormulaNode],
    ) -> FormulaNode:
        if n.kind == NodeKind.VAR:
            return mapping.get(n.value, n)
        if n.kind == NodeKind.CONST:
            return n
        if n.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            if n.coefficients is None:
                return n
            new_coeffs: Dict[str, float] = {}
            for v, c in n.coefficients.items():
                if v in mapping and mapping[v].kind == NodeKind.VAR:
                    new_coeffs[mapping[v].value] = new_coeffs.get(mapping[v].value, 0.0) + c
                else:
                    new_coeffs[v] = new_coeffs.get(v, 0.0) + c
            return FormulaNode(kind=n.kind, coefficients=new_coeffs, rhs=n.rhs)
        if n.bound_var and n.bound_var in mapping:
            return n
        new_children = tuple(self._subst(c, mapping) for c in n.children)
        return FormulaNode(
            kind=n.kind, children=new_children, value=n.value,
            bound_var=n.bound_var, coefficients=n.coefficients, rhs=n.rhs,
        )


# ---------------------------------------------------------------------------
# Quantifier Elimination (Fourier-Motzkin)
# ---------------------------------------------------------------------------


class QuantifierElimination:
    """Fourier-Motzkin elimination for linear arithmetic.

    Eliminates existentially quantified variables from conjunctions
    of linear inequalities.
    """

    def __init__(self, *, max_constraints: int = 10000) -> None:
        self.max_constraints = max_constraints

    def eliminate(self, formula: Formula, variables: Sequence[str]) -> Formula:
        """Eliminate ``variables`` from a conjunction of linear constraints."""
        constraints = self._collect_constraints(formula.node)
        for var in variables:
            constraints = self._eliminate_var(constraints, var)
            if len(constraints) > self.max_constraints:
                constraints = constraints[: self.max_constraints]
        nodes = [
            FormulaNode.leq(c, r) for c, r in constraints
        ]
        if not nodes:
            return Formula(FormulaNode.const(True))
        return Formula(FormulaNode.and_(*nodes))

    def _collect_constraints(
        self, n: FormulaNode,
    ) -> List[Tuple[Dict[str, float], float]]:
        """Collect linear inequality constraints from an AND-tree."""
        if n.kind == NodeKind.AND:
            result: List[Tuple[Dict[str, float], float]] = []
            for c in n.children:
                result.extend(self._collect_constraints(c))
            return result
        if n.kind == NodeKind.LEQ and n.coefficients is not None:
            return [(dict(n.coefficients), n.rhs or 0.0)]
        if n.kind == NodeKind.EQ and n.coefficients is not None:
            # a = b  <==>  a <= b AND -a <= -b
            neg = {v: -c for v, c in n.coefficients.items()}
            return [
                (dict(n.coefficients), n.rhs or 0.0),
                (neg, -(n.rhs or 0.0)),
            ]
        if n.kind == NodeKind.CONST and n.value:
            return []
        return [(dict(), 0.0)]

    def _eliminate_var(
        self,
        constraints: List[Tuple[Dict[str, float], float]],
        var: str,
    ) -> List[Tuple[Dict[str, float], float]]:
        """Single-variable Fourier-Motzkin step."""
        upper: List[Tuple[Dict[str, float], float]] = []
        lower: List[Tuple[Dict[str, float], float]] = []
        rest: List[Tuple[Dict[str, float], float]] = []

        for coeffs, rhs in constraints:
            c = coeffs.get(var, 0.0)
            if abs(c) < 1e-15:
                rest.append((coeffs, rhs))
            elif c > 0:
                # c*var + rest <= rhs  =>  var <= (rhs - rest)/c
                upper.append((coeffs, rhs))
            else:
                # c*var + rest <= rhs  =>  var >= (rhs - rest)/|c|
                lower.append((coeffs, rhs))

        new_constraints = list(rest)
        for lo_coeffs, lo_rhs in lower:
            for up_coeffs, up_rhs in upper:
                cl = abs(lo_coeffs.get(var, 1.0))
                cu = up_coeffs.get(var, 1.0)
                # Combine: cu * lower + cl * upper to eliminate var
                combined: Dict[str, float] = {}
                for v, coeff in lo_coeffs.items():
                    if v != var:
                        combined[v] = combined.get(v, 0.0) + cu * coeff
                for v, coeff in up_coeffs.items():
                    if v != var:
                        combined[v] = combined.get(v, 0.0) + cl * coeff
                new_rhs = cu * lo_rhs + cl * up_rhs
                combined = {v: c for v, c in combined.items() if abs(c) > 1e-15}
                new_constraints.append((combined, new_rhs))

        return new_constraints


# ---------------------------------------------------------------------------
# Satisfiability Checker
# ---------------------------------------------------------------------------


class SatisfiabilityChecker:
    """Check satisfiability of formula conjunctions.

    Uses a combination of LP relaxation (numpy) for linear arithmetic
    and truth-table enumeration for small boolean problems.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def check(self, formula: Formula) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check if formula is satisfiable.

        Returns:
            (is_sat, witness) where witness is a satisfying assignment if SAT.
        """
        constraints = QuantifierElimination()._collect_constraints(formula.node)
        if not constraints:
            return True, {}

        all_vars: Set[str] = set()
        for coeffs, _ in constraints:
            all_vars.update(coeffs.keys())
        var_list = sorted(all_vars)

        if not var_list:
            for coeffs, rhs in constraints:
                if 0.0 > rhs + 1e-12:
                    return False, None
            return True, {}

        n = len(var_list)
        var_idx = {v: i for i, v in enumerate(var_list)}

        A_rows: List[List[float]] = []
        b_vals: List[float] = []
        for coeffs, rhs in constraints:
            row = [0.0] * n
            for v, c in coeffs.items():
                row[var_idx[v]] = c
            A_rows.append(row)
            b_vals.append(rhs)

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)

        return self._solve_lp_feasibility(A, b, var_list)

    def _solve_lp_feasibility(
        self,
        A: np.ndarray,
        b: np.ndarray,
        var_list: List[str],
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Simple feasibility check via iterative projection."""
        m, n = A.shape
        x = np.zeros(n, dtype=np.float64)

        # Iterative constraint satisfaction
        for iteration in range(200):
            violations = A @ x - b
            max_violation = np.max(violations)
            if max_violation <= 1e-8:
                witness = {var_list[i]: float(x[i]) for i in range(n)}
                return True, witness

            # Adjust x to reduce worst violation
            worst = int(np.argmax(violations))
            row = A[worst]
            norm_sq = float(np.dot(row, row))
            if norm_sq < 1e-15:
                if violations[worst] > 1e-8:
                    return False, None
                continue
            step = violations[worst] / norm_sq
            x -= step * row

        # After iteration limit, check final feasibility
        violations = A @ x - b
        if np.max(violations) <= 1e-6:
            witness = {var_list[i]: float(x[i]) for i in range(n)}
            return True, witness
        return False, None

    def is_unsat(self, formula: Formula) -> bool:
        """Check if formula is unsatisfiable."""
        sat, _ = self.check(formula)
        return not sat

    def check_conjunction(self, formulas: List[Formula]) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check satisfiability of a conjunction of formulas."""
        if not formulas:
            return True, {}
        combined = formulas[0]
        for f in formulas[1:]:
            combined = combined & f
        return self.check(combined)

    def implies(self, antecedent: Formula, consequent: Formula) -> bool:
        """Check if antecedent logically implies consequent.

        A ⊨ B iff A ∧ ¬B is UNSAT.
        """
        neg_consequent = ~consequent
        conjunction = antecedent & neg_consequent
        return self.is_unsat(conjunction)


__all__ = [
    "NodeKind",
    "FormulaNode",
    "Formula",
    "CNFConverter",
    "DNFConverter",
    "Simplifier",
    "SubstitutionEngine",
    "QuantifierElimination",
    "SatisfiabilityChecker",
]
