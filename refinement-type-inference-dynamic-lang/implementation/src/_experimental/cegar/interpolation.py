from __future__ import annotations

"""
Unsat core-based predicate extraction for counterexample-guided contract discovery.

When a counterexample is spurious, this module extracts Craig interpolants
that eliminate it and projects them into the predicate language P used by
the abstract domain.  Supports linear integer arithmetic (QF_LIA via Farkas),
type-tag constraints, nullity constraints, structural (hasattr/dict-key)
constraints, and their reduced-product combination.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ===================================================================
# Local expression / formula AST (self-contained, no project imports)
# ===================================================================

class UnaryOp(Enum):
    NOT = "not"
    NEG = "neg"
    ABS = "abs"
    LEN = "len"
    IS_NONE = "is_none"
    IS_TRUTHY = "is_truthy"


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    FLOOR_DIV = "//"
    AND = "and"
    OR = "or"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IMPLIES = "=>"
    IN = "in"
    ISINSTANCE = "isinstance"
    HASATTR = "hasattr"


class Quantifier(Enum):
    FORALL = "forall"
    EXISTS = "exists"


@dataclass(frozen=True)
class Var:
    name: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return mapping.get(self.name, self)

    def to_smt(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntLit:
    value: int

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return str(self.value) if self.value >= 0 else f"(- {-self.value})"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BoolLit:
    value: bool

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return "true" if self.value else "false"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class StrLit:
    value: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return f'"{self.value}"'

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass(frozen=True)
class NoneLit:
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return "none"

    def __str__(self) -> str:
        return "None"


@dataclass(frozen=True)
class UnaryExpr:
    op: UnaryOp
    operand: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return UnaryExpr(self.op, self.operand.substitute(mapping))

    def to_smt(self) -> str:
        op_map = {
            UnaryOp.NOT: "not", UnaryOp.NEG: "-", UnaryOp.ABS: "abs",
            UnaryOp.LEN: "len", UnaryOp.IS_NONE: "is_none",
            UnaryOp.IS_TRUTHY: "is_truthy",
        }
        return f"({op_map[self.op]} {self.operand.to_smt()})"

    def __str__(self) -> str:
        return f"({self.op.value} {self.operand})"


@dataclass(frozen=True)
class BinaryExpr:
    op: BinaryOp
    left: Expr
    right: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return BinaryExpr(
            self.op,
            self.left.substitute(mapping),
            self.right.substitute(mapping),
        )

    def to_smt(self) -> str:
        op_map = {
            BinaryOp.ADD: "+", BinaryOp.SUB: "-", BinaryOp.MUL: "*",
            BinaryOp.DIV: "div", BinaryOp.MOD: "mod", BinaryOp.FLOOR_DIV: "div",
            BinaryOp.AND: "and", BinaryOp.OR: "or",
            BinaryOp.EQ: "=", BinaryOp.NE: "distinct",
            BinaryOp.LT: "<", BinaryOp.LE: "<=",
            BinaryOp.GT: ">", BinaryOp.GE: ">=",
            BinaryOp.IMPLIES: "=>",
            BinaryOp.IN: "contains",
            BinaryOp.ISINSTANCE: "isinstance",
            BinaryOp.HASATTR: "hasattr",
        }
        return f"({op_map[self.op]} {self.left.to_smt()} {self.right.to_smt()})"

    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"


@dataclass(frozen=True)
class QuantifiedExpr:
    quantifier: Quantifier
    var_name: str
    var_sort: str
    body: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var_name}

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        new_mapping = {k: v for k, v in mapping.items() if k != self.var_name}
        return QuantifiedExpr(
            self.quantifier, self.var_name, self.var_sort,
            self.body.substitute(new_mapping),
        )

    def to_smt(self) -> str:
        q = self.quantifier.value
        return f"({q} (({self.var_name} {self.var_sort})) {self.body.to_smt()})"

    def __str__(self) -> str:
        return f"({self.quantifier.value} {self.var_name}:{self.var_sort}. {self.body})"


@dataclass(frozen=True)
class FuncApp:
    func_name: str
    args: Tuple[Expr, ...]

    def free_vars(self) -> FrozenSet[str]:
        r: FrozenSet[str] = frozenset()
        for a in self.args:
            r = r | a.free_vars()
        return r

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return FuncApp(self.func_name, tuple(a.substitute(mapping) for a in self.args))

    def to_smt(self) -> str:
        if not self.args:
            return self.func_name
        return f"({self.func_name} {' '.join(a.to_smt() for a in self.args)})"

    def __str__(self) -> str:
        return f"{self.func_name}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True)
class IteExpr:
    cond: Expr
    then_branch: Expr
    else_branch: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.cond.free_vars() | self.then_branch.free_vars() | self.else_branch.free_vars()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return IteExpr(
            self.cond.substitute(mapping),
            self.then_branch.substitute(mapping),
            self.else_branch.substitute(mapping),
        )

    def to_smt(self) -> str:
        return (
            f"(ite {self.cond.to_smt()} "
            f"{self.then_branch.to_smt()} {self.else_branch.to_smt()})"
        )

    def __str__(self) -> str:
        return f"(if {self.cond} then {self.then_branch} else {self.else_branch})"


Expr = Union[
    Var, IntLit, BoolLit, StrLit, NoneLit,
    UnaryExpr, BinaryExpr, QuantifiedExpr,
    FuncApp, IteExpr,
]

# Minimal statement types used in path encoding
@dataclass(frozen=True)
class AssignStmt:
    target: str
    value: Expr

@dataclass(frozen=True)
class AssumeStmt:
    condition: Expr

@dataclass(frozen=True)
class AssertStmt:
    condition: Expr

@dataclass(frozen=True)
class CallStmt:
    target: Optional[str]
    func_name: str
    args: Tuple[Expr, ...]

@dataclass(frozen=True)
class SkipStmt:
    pass

@dataclass(frozen=True)
class SeqStmt:
    first: Stmt
    second: Stmt

@dataclass(frozen=True)
class RaiseStmt:
    exception_type: str
    message: Optional[Expr] = None

Stmt = Union[AssignStmt, AssumeStmt, AssertStmt, CallStmt, SkipStmt, SeqStmt, RaiseStmt]


# ===================================================================
# Expression helpers
# ===================================================================

TRUE = BoolLit(True)
FALSE = BoolLit(False)


def _var(name: str) -> Var:
    return Var(name)


def _int(v: int) -> IntLit:
    return IntLit(v)


def _and(a: Expr, b: Expr) -> Expr:
    if isinstance(a, BoolLit) and a.value:
        return b
    if isinstance(b, BoolLit) and b.value:
        return a
    if isinstance(a, BoolLit) and not a.value:
        return FALSE
    if isinstance(b, BoolLit) and not b.value:
        return FALSE
    return BinaryExpr(BinaryOp.AND, a, b)


def _or(a: Expr, b: Expr) -> Expr:
    if isinstance(a, BoolLit) and a.value:
        return TRUE
    if isinstance(b, BoolLit) and b.value:
        return TRUE
    if isinstance(a, BoolLit) and not a.value:
        return b
    if isinstance(b, BoolLit) and not b.value:
        return a
    return BinaryExpr(BinaryOp.OR, a, b)


def _not(a: Expr) -> Expr:
    if isinstance(a, BoolLit):
        return BoolLit(not a.value)
    if isinstance(a, UnaryExpr) and a.op == UnaryOp.NOT:
        return a.operand
    return UnaryExpr(UnaryOp.NOT, a)


def _implies(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.IMPLIES, a, b)


def _eq(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.EQ, a, b)


def _ne(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.NE, a, b)


def _lt(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.LT, a, b)


def _le(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.LE, a, b)


def _gt(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.GT, a, b)


def _ge(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.GE, a, b)


def _add(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.ADD, a, b)


def _sub(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.SUB, a, b)


def _mul(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.MUL, a, b)


def _conjunction(exprs: List[Expr]) -> Expr:
    if not exprs:
        return TRUE
    r = exprs[0]
    for e in exprs[1:]:
        r = _and(r, e)
    return r


def _disjunction(exprs: List[Expr]) -> Expr:
    if not exprs:
        return FALSE
    r = exprs[0]
    for e in exprs[1:]:
        r = _or(r, e)
    return r


def _collect_free_vars(expr: Expr) -> FrozenSet[str]:
    return expr.free_vars()


def _expr_size(expr: Expr) -> int:
    """Count AST nodes."""
    if isinstance(expr, (Var, IntLit, BoolLit, StrLit, NoneLit)):
        return 1
    if isinstance(expr, UnaryExpr):
        return 1 + _expr_size(expr.operand)
    if isinstance(expr, BinaryExpr):
        return 1 + _expr_size(expr.left) + _expr_size(expr.right)
    if isinstance(expr, QuantifiedExpr):
        return 1 + _expr_size(expr.body)
    if isinstance(expr, FuncApp):
        return 1 + sum(_expr_size(a) for a in expr.args)
    if isinstance(expr, IteExpr):
        return 1 + _expr_size(expr.cond) + _expr_size(expr.then_branch) + _expr_size(expr.else_branch)
    return 1


def _structural_hash(expr: Expr) -> str:
    """Hash an expression structurally (stable across runs)."""
    h = hashlib.sha256(str(expr).encode()).hexdigest()[:16]
    return h


# ===================================================================
# Theory classification
# ===================================================================

class Theory(Enum):
    LIA = "lia"           # Linear integer arithmetic
    TYPE_TAG = "type_tag"  # isinstance / tag_of constraints
    NULLITY = "nullity"    # is_none constraints
    STRUCTURAL = "structural"  # hasattr / dict key constraints
    BOOLEAN = "boolean"    # Pure boolean
    UNKNOWN = "unknown"


def classify_theory(expr: Expr) -> Theory:
    """Determine which theory an expression belongs to."""
    if isinstance(expr, (IntLit, BoolLit)):
        return Theory.BOOLEAN
    if isinstance(expr, Var):
        return Theory.LIA  # default numeric

    if isinstance(expr, UnaryExpr):
        if expr.op == UnaryOp.IS_NONE:
            return Theory.NULLITY
        if expr.op == UnaryOp.IS_TRUTHY:
            return Theory.TYPE_TAG
        if expr.op in (UnaryOp.NEG, UnaryOp.ABS):
            return Theory.LIA
        if expr.op == UnaryOp.NOT:
            return classify_theory(expr.operand)
        return Theory.UNKNOWN

    if isinstance(expr, BinaryExpr):
        if expr.op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL,
                        BinaryOp.DIV, BinaryOp.MOD, BinaryOp.FLOOR_DIV,
                        BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE):
            return Theory.LIA
        if expr.op == BinaryOp.ISINSTANCE:
            return Theory.TYPE_TAG
        if expr.op == BinaryOp.HASATTR:
            return Theory.STRUCTURAL
        if expr.op in (BinaryOp.AND, BinaryOp.OR, BinaryOp.IMPLIES):
            lt = classify_theory(expr.left)
            rt = classify_theory(expr.right)
            if lt == rt:
                return lt
            return Theory.UNKNOWN
        if expr.op in (BinaryOp.EQ, BinaryOp.NE):
            lt = classify_theory(expr.left)
            rt = classify_theory(expr.right)
            if lt == Theory.NULLITY or rt == Theory.NULLITY:
                return Theory.NULLITY
            if lt == Theory.TYPE_TAG or rt == Theory.TYPE_TAG:
                return Theory.TYPE_TAG
            return Theory.LIA

    if isinstance(expr, FuncApp):
        if expr.func_name in ("is_none",):
            return Theory.NULLITY
        if expr.func_name in ("isinstance", "tag_of", "is_truthy"):
            return Theory.TYPE_TAG
        if expr.func_name in ("hasattr", "dict_has_key"):
            return Theory.STRUCTURAL
        if expr.func_name in ("len", "str_len", "list_len", "dict_len", "list_get"):
            return Theory.LIA

    return Theory.UNKNOWN


# ===================================================================
# Interpolant
# ===================================================================

@dataclass
class Interpolant:
    """Represents a Craig interpolant for the pair (A, B).

    Properties:
      - A |= I  (the interpolant is implied by A)
      - I ∧ B is unsatisfiable
      - free_vars(I) ⊆ free_vars(A) ∩ free_vars(B)
    """
    formula: Expr
    variables: FrozenSet[str]
    source_a: Expr
    source_b: Expr
    theory: Theory = Theory.UNKNOWN
    size: int = 0
    generation_time_ms: float = 0.0
    _properties_checked: bool = False

    def __post_init__(self) -> None:
        if self.size == 0:
            object.__setattr__(self, "size", _expr_size(self.formula))

    @property
    def common_variables(self) -> FrozenSet[str]:
        return self.source_a.free_vars() & self.source_b.free_vars()

    def check_variable_condition(self) -> bool:
        """Check that free vars of interpolant ⊆ common variables."""
        return self.formula.free_vars() <= self.common_variables

    def to_smt_check(self) -> str:
        """Generate SMT-LIB script to check interpolant properties."""
        lines: List[str] = []
        lines.append("; Check interpolant properties")
        lines.append("(set-logic QF_LIA)")
        all_vars = (
            self.source_a.free_vars()
            | self.source_b.free_vars()
            | self.formula.free_vars()
        )
        for v in sorted(all_vars):
            lines.append(f"(declare-const {v} Int)")

        # Check A => I
        lines.append("; Check: A => I")
        lines.append("(push)")
        lines.append(f"(assert {self.source_a.to_smt()})")
        lines.append(f"(assert (not {self.formula.to_smt()}))")
        lines.append("(check-sat)  ; expect unsat")
        lines.append("(pop)")

        # Check I ∧ B is unsat
        lines.append("; Check: I ∧ B is unsat")
        lines.append("(push)")
        lines.append(f"(assert {self.formula.to_smt()})")
        lines.append(f"(assert {self.source_b.to_smt()})")
        lines.append("(check-sat)  ; expect unsat")
        lines.append("(pop)")

        lines.append("(exit)")
        return "\n".join(lines)


# ===================================================================
# Linear constraint representation (for Farkas-based interpolation)
# ===================================================================

@dataclass(frozen=True)
class LinearConstraint:
    """Represents a_1*x_1 + ... + a_n*x_n + c <= 0  (or == 0, or < 0)."""

    class Kind(Enum):
        LE = "<="  # <= 0
        LT = "<"   # < 0
        EQ = "=="  # == 0

    coefficients: Dict[str, int]  # variable name → coefficient
    constant: int
    kind: Kind = Kind.LE

    @property
    def variables(self) -> FrozenSet[str]:
        return frozenset(self.coefficients.keys())

    def evaluate(self, assignment: Dict[str, int]) -> int:
        """Evaluate LHS given an assignment."""
        total = self.constant
        for var, coeff in self.coefficients.items():
            total += coeff * assignment.get(var, 0)
        return total

    def is_satisfied(self, assignment: Dict[str, int]) -> bool:
        val = self.evaluate(assignment)
        if self.kind == self.Kind.LE:
            return val <= 0
        if self.kind == self.Kind.LT:
            return val < 0
        return val == 0

    def negate(self) -> LinearConstraint:
        """Negate this constraint. -(ax+c <= 0) ↔ ax+c > 0 ↔ -ax-c-1 <= 0."""
        if self.kind == self.Kind.LE:
            new_coeffs = {v: -c for v, c in self.coefficients.items()}
            return LinearConstraint(new_coeffs, -self.constant - 1, self.Kind.LE)
        if self.kind == self.Kind.LT:
            new_coeffs = {v: -c for v, c in self.coefficients.items()}
            return LinearConstraint(new_coeffs, -self.constant, self.Kind.LE)
        # EQ: split into two cases (not a single negate)
        return LinearConstraint(dict(self.coefficients), self.constant, self.Kind.LE)

    def scale(self, factor: int) -> LinearConstraint:
        new_coeffs = {v: c * factor for v, c in self.coefficients.items()}
        return LinearConstraint(new_coeffs, self.constant * factor, self.kind)

    @staticmethod
    def add(a: LinearConstraint, b: LinearConstraint) -> LinearConstraint:
        merged: Dict[str, int] = dict(a.coefficients)
        for v, c in b.coefficients.items():
            merged[v] = merged.get(v, 0) + c
        merged = {v: c for v, c in merged.items() if c != 0}
        kind = LinearConstraint.Kind.LE
        if a.kind == LinearConstraint.Kind.LT or b.kind == LinearConstraint.Kind.LT:
            kind = LinearConstraint.Kind.LT
        return LinearConstraint(merged, a.constant + b.constant, kind)

    def to_expr(self) -> Expr:
        """Convert back to an Expr."""
        terms: List[Expr] = []
        for v in sorted(self.coefficients):
            c = self.coefficients[v]
            if c == 0:
                continue
            if c == 1:
                terms.append(_var(v))
            elif c == -1:
                terms.append(UnaryExpr(UnaryOp.NEG, _var(v)))
            else:
                terms.append(_mul(_int(c), _var(v)))
        if not terms:
            lhs: Expr = _int(self.constant)
        else:
            lhs = terms[0]
            for t in terms[1:]:
                lhs = _add(lhs, t)
            if self.constant != 0:
                lhs = _add(lhs, _int(self.constant))

        if self.kind == LinearConstraint.Kind.LE:
            return _le(lhs, _int(0))
        if self.kind == LinearConstraint.Kind.LT:
            return _lt(lhs, _int(0))
        return _eq(lhs, _int(0))

    def __str__(self) -> str:
        parts: List[str] = []
        for v in sorted(self.coefficients):
            c = self.coefficients[v]
            if c == 0:
                continue
            if c == 1:
                parts.append(v)
            elif c == -1:
                parts.append(f"-{v}")
            else:
                parts.append(f"{c}*{v}")
        if not parts:
            lhs_str = str(self.constant)
        else:
            lhs_str = " + ".join(parts)
            if self.constant != 0:
                lhs_str += f" + {self.constant}"
        return f"{lhs_str} {self.kind.value} 0"


def _expr_to_linear_constraints(expr: Expr) -> List[LinearConstraint]:
    """Extract linear constraints from an expression (best-effort)."""
    constraints: List[LinearConstraint] = []

    if isinstance(expr, BinaryExpr):
        if expr.op == BinaryOp.AND:
            constraints.extend(_expr_to_linear_constraints(expr.left))
            constraints.extend(_expr_to_linear_constraints(expr.right))
            return constraints

        # Handle comparison operators
        lc = _try_extract_single_constraint(expr)
        if lc is not None:
            constraints.append(lc)
    elif isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
        inner = expr.operand
        lc = _try_extract_single_constraint(inner)
        if lc is not None:
            constraints.append(lc.negate())

    return constraints


def _try_extract_single_constraint(expr: Expr) -> Optional[LinearConstraint]:
    """Try to extract a single linear constraint from a comparison."""
    if not isinstance(expr, BinaryExpr):
        return None

    if expr.op not in (BinaryOp.LE, BinaryOp.LT, BinaryOp.GE, BinaryOp.GT, BinaryOp.EQ):
        return None

    left_coeffs, left_const = _extract_linear_terms(expr.left)
    right_coeffs, right_const = _extract_linear_terms(expr.right)

    if left_coeffs is None or right_coeffs is None:
        return None

    # Normalize to LHS - RHS  {<=,<,==}  0
    merged: Dict[str, int] = dict(left_coeffs)
    for v, c in right_coeffs.items():
        merged[v] = merged.get(v, 0) - c
    merged = {v: c for v, c in merged.items() if c != 0}
    constant = left_const - right_const

    if expr.op == BinaryOp.LE:
        return LinearConstraint(merged, constant, LinearConstraint.Kind.LE)
    if expr.op == BinaryOp.LT:
        return LinearConstraint(merged, constant, LinearConstraint.Kind.LT)
    if expr.op == BinaryOp.GE:
        neg = {v: -c for v, c in merged.items()}
        return LinearConstraint(neg, -constant, LinearConstraint.Kind.LE)
    if expr.op == BinaryOp.GT:
        neg = {v: -c for v, c in merged.items()}
        return LinearConstraint(neg, -constant, LinearConstraint.Kind.LT)
    if expr.op == BinaryOp.EQ:
        return LinearConstraint(merged, constant, LinearConstraint.Kind.EQ)

    return None


def _extract_linear_terms(
    expr: Expr,
) -> Tuple[Optional[Dict[str, int]], int]:
    """Extract coefficients and constant from a linear expression."""
    if isinstance(expr, IntLit):
        return {}, expr.value
    if isinstance(expr, Var):
        return {expr.name: 1}, 0
    if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NEG:
        inner_coeffs, inner_const = _extract_linear_terms(expr.operand)
        if inner_coeffs is None:
            return None, 0
        return {v: -c for v, c in inner_coeffs.items()}, -inner_const
    if isinstance(expr, BinaryExpr):
        if expr.op == BinaryOp.ADD:
            lc, lk = _extract_linear_terms(expr.left)
            rc, rk = _extract_linear_terms(expr.right)
            if lc is None or rc is None:
                return None, 0
            merged = dict(lc)
            for v, c in rc.items():
                merged[v] = merged.get(v, 0) + c
            return merged, lk + rk
        if expr.op == BinaryOp.SUB:
            lc, lk = _extract_linear_terms(expr.left)
            rc, rk = _extract_linear_terms(expr.right)
            if lc is None or rc is None:
                return None, 0
            merged = dict(lc)
            for v, c in rc.items():
                merged[v] = merged.get(v, 0) - c
            return merged, lk - rk
        if expr.op == BinaryOp.MUL:
            # Only handle constant * var or var * constant
            if isinstance(expr.left, IntLit):
                rc, rk = _extract_linear_terms(expr.right)
                if rc is None:
                    return None, 0
                factor = expr.left.value
                return {v: c * factor for v, c in rc.items()}, rk * factor
            if isinstance(expr.right, IntLit):
                lc, lk = _extract_linear_terms(expr.left)
                if lc is None:
                    return None, 0
                factor = expr.right.value
                return {v: c * factor for v, c in lc.items()}, lk * factor
    if isinstance(expr, FuncApp):
        if expr.func_name in ("len", "str_len", "list_len", "dict_len"):
            # Treat as uninterpreted variable
            synth_name = f"__{expr.func_name}_{expr.args[0]}" if expr.args else expr.func_name
            return {synth_name: 1}, 0
    return None, 0


# ===================================================================
# FarkasLemma
# ===================================================================

class FarkasLemma:
    """
    Implementation of Farkas' lemma for linear arithmetic interpolation.

    Given a system Ax ≤ b that is infeasible, Farkas' lemma states there exist
    non-negative multipliers λ ≥ 0 such that λᵀA = 0 and λᵀb < 0.
    These multipliers form an infeasibility certificate.

    For interpolation, we partition the constraints into A-constraints and
    B-constraints and compute an interpolant from the Farkas coefficients
    restricted to common variables.
    """

    def __init__(self) -> None:
        self._epsilon: float = 1e-9

    def compute_farkas_coefficients(
        self,
        constraints: List[LinearConstraint],
    ) -> Optional[List[float]]:
        """
        Compute Farkas coefficients for an infeasible system.

        Uses a simple iterative approach: try to find non-negative λ
        such that Σ λᵢ · constraint_i yields 0 ≤ -1 (contradiction).

        For small systems we use direct enumeration / pivoting.
        """
        n = len(constraints)
        if n == 0:
            return None

        all_vars = set()
        for c in constraints:
            all_vars |= set(c.coefficients.keys())
        var_list = sorted(all_vars)
        m = len(var_list)
        var_idx = {v: i for i, v in enumerate(var_list)}

        # Build matrix A (n × m) and vector b (n)
        A: List[List[int]] = []
        b: List[int] = []
        for c in constraints:
            row = [0] * m
            for v, coeff in c.coefficients.items():
                row[var_idx[v]] = coeff
            A.append(row)
            b.append(c.constant)

        # Try to find Farkas coefficients by solving the dual
        # λ ≥ 0, λᵀA = 0, λᵀb < 0
        # Heuristic: try combinations of constraints
        lambdas = self._solve_farkas_dual(A, b, n, m)
        return lambdas

    def _solve_farkas_dual(
        self,
        A: List[List[int]],
        b: List[int],
        n: int,
        m: int,
    ) -> Optional[List[float]]:
        """
        Solve the Farkas dual system.

        For small n, try all subsets of size ≤ m+1.
        For larger n, use Gaussian elimination on the dual.
        """
        if n <= 10:
            return self._enumerate_small(A, b, n, m)
        return self._gaussian_dual(A, b, n, m)

    def _enumerate_small(
        self,
        A: List[List[int]],
        b: List[int],
        n: int,
        m: int,
    ) -> Optional[List[float]]:
        """Try subsets of constraints to find a Farkas certificate."""
        # Try pairs first
        for i in range(n):
            for j in range(i + 1, n):
                lambdas = [0.0] * n
                # Try λ_i * A_i + λ_j * A_j = 0
                result = self._solve_two_constraint(A[i], b[i], A[j], b[j], m)
                if result is not None:
                    li, lj = result
                    if li >= -self._epsilon and lj >= -self._epsilon:
                        val = li * b[i] + lj * b[j]
                        if val < -self._epsilon:
                            lambdas[i] = max(0.0, li)
                            lambdas[j] = max(0.0, lj)
                            return lambdas

        # Try triples
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    result = self._solve_three_constraint(
                        A[i], b[i], A[j], b[j], A[k], b[k], m
                    )
                    if result is not None:
                        li, lj, lk = result
                        if (li >= -self._epsilon and lj >= -self._epsilon
                                and lk >= -self._epsilon):
                            val = li * b[i] + lj * b[j] + lk * b[k]
                            if val < -self._epsilon:
                                lambdas = [0.0] * n
                                lambdas[i] = max(0.0, li)
                                lambdas[j] = max(0.0, lj)
                                lambdas[k] = max(0.0, lk)
                                return lambdas

        # Uniform distribution as fallback
        lambdas = [1.0] * n
        col_sums = [0.0] * m
        for i in range(n):
            for j in range(m):
                col_sums[j] += A[i][j]
        all_zero = all(abs(s) < self._epsilon for s in col_sums)
        if all_zero:
            b_sum = sum(b)
            if b_sum < -self._epsilon:
                return lambdas
        return None

    def _solve_two_constraint(
        self,
        a1: List[int], b1: int,
        a2: List[int], b2: int,
        m: int,
    ) -> Optional[Tuple[float, float]]:
        """Solve λ₁*a₁ + λ₂*a₂ = 0 for scalar λ₁, λ₂."""
        # For each variable j: λ₁*a₁[j] + λ₂*a₂[j] = 0
        # If a₁[j] ≠ 0 and a₂[j] ≠ 0: λ₂/λ₁ = -a₁[j]/a₂[j]
        ratio: Optional[float] = None
        for j in range(m):
            if a1[j] == 0 and a2[j] == 0:
                continue
            if a1[j] == 0 or a2[j] == 0:
                return None  # incompatible
            r = -a1[j] / a2[j]
            if ratio is None:
                ratio = r
            elif abs(r - ratio) > self._epsilon:
                return None
        if ratio is None:
            return (1.0, 1.0)
        return (1.0, ratio)

    def _solve_three_constraint(
        self,
        a1: List[int], b1: int,
        a2: List[int], b2: int,
        a3: List[int], b3: int,
        m: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Solve λ₁*a₁ + λ₂*a₂ + λ₃*a₃ = 0 for λ₁=1."""
        # Fix λ₁ = 1, solve for λ₂, λ₃ from first two non-trivial equations
        equations: List[Tuple[int, int, int]] = []
        for j in range(m):
            if a2[j] != 0 or a3[j] != 0:
                equations.append((a1[j], a2[j], a3[j]))
        if len(equations) < 2:
            if len(equations) == 0:
                return (1.0, 0.0, 0.0)
            eq = equations[0]
            if eq[1] != 0:
                l2 = -eq[0] / eq[1]
                return (1.0, l2, 0.0)
            if eq[2] != 0:
                l3 = -eq[0] / eq[2]
                return (1.0, 0.0, l3)
            return None

        # 2x2 system: eq[0][1]*l2 + eq[0][2]*l3 = -eq[0][0]
        #             eq[1][1]*l2 + eq[1][2]*l3 = -eq[1][0]
        e0, e1 = equations[0], equations[1]
        det = e0[1] * e1[2] - e0[2] * e1[1]
        if abs(det) < self._epsilon:
            return None
        l2 = (-e0[0] * e1[2] - e0[2] * (-e1[0])) / det
        l3 = (e0[1] * (-e1[0]) - (-e0[0]) * e1[1]) / det

        # Verify remaining equations
        for eq in equations[2:]:
            val = eq[0] + l2 * eq[1] + l3 * eq[2]
            if abs(val) > self._epsilon:
                return None

        return (1.0, l2, l3)

    def _gaussian_dual(
        self,
        A: List[List[int]],
        b: List[int],
        n: int,
        m: int,
    ) -> Optional[List[float]]:
        """Gaussian elimination on the transposed system for larger instances."""
        # Aᵀλ = 0, where Aᵀ is m × n
        # Augmented matrix [Aᵀ | 0]
        mat: List[List[float]] = []
        for j in range(m):
            row = [float(A[i][j]) for i in range(n)] + [0.0]
            mat.append(row)

        # Forward elimination
        pivot_col = 0
        for row_idx in range(m):
            if pivot_col >= n:
                break
            # Find pivot
            max_row = row_idx
            for r in range(row_idx + 1, m):
                if abs(mat[r][pivot_col]) > abs(mat[max_row][pivot_col]):
                    max_row = r
            if abs(mat[max_row][pivot_col]) < self._epsilon:
                pivot_col += 1
                continue
            mat[row_idx], mat[max_row] = mat[max_row], mat[row_idx]
            pivot_val = mat[row_idx][pivot_col]
            for c in range(n + 1):
                mat[row_idx][c] /= pivot_val
            for r in range(m):
                if r != row_idx and abs(mat[r][pivot_col]) > self._epsilon:
                    factor = mat[r][pivot_col]
                    for c in range(n + 1):
                        mat[r][c] -= factor * mat[row_idx][c]
            pivot_col += 1

        # Extract null space: free variables get λ = 1
        lambdas = [0.0] * n
        pivot_vars: Set[int] = set()
        for row_idx in range(min(m, n)):
            for c in range(n):
                if abs(mat[row_idx][c] - 1.0) < self._epsilon:
                    is_pivot = all(
                        abs(mat[r][c]) < self._epsilon
                        for r in range(m) if r != row_idx
                    )
                    if is_pivot:
                        pivot_vars.add(c)
                        break

        free_vars = [i for i in range(n) if i not in pivot_vars]
        if not free_vars:
            return [1.0] * n

        for fv in free_vars:
            lambdas[fv] = 1.0
        for row_idx in range(min(m, n)):
            for pv in pivot_vars:
                if abs(mat[row_idx][pv] - 1.0) < self._epsilon:
                    val = 0.0
                    for fv in free_vars:
                        val -= mat[row_idx][fv]
                    lambdas[pv] = val
                    break

        # Check non-negativity
        if any(l < -self._epsilon for l in lambdas):
            # Try negating
            neg_lambdas = [-l for l in lambdas]
            if all(l >= -self._epsilon for l in neg_lambdas):
                lambdas = neg_lambdas
            else:
                return None

        # Check λᵀb < 0
        b_sum = sum(lambdas[i] * b[i] for i in range(n))
        if b_sum < -self._epsilon:
            return [max(0.0, l) for l in lambdas]
        return None

    def extract_interpolant(
        self,
        a_constraints: List[LinearConstraint],
        b_constraints: List[LinearConstraint],
        lambdas: List[float],
        common_vars: FrozenSet[str],
    ) -> Expr:
        """
        Given Farkas coefficients for the combined system A ∪ B,
        extract an interpolant over the common variables.

        The interpolant is: Σ_{i ∈ A-indices} λᵢ · constraint_i
        restricted to common variables.
        """
        n_a = len(a_constraints)
        # Sum the A-part weighted constraints
        combined_coeffs: Dict[str, float] = {}
        combined_const: float = 0.0

        for i in range(n_a):
            lam = lambdas[i]
            if lam < self._epsilon:
                continue
            for v, c in a_constraints[i].coefficients.items():
                combined_coeffs[v] = combined_coeffs.get(v, 0.0) + lam * c
            combined_const += lam * a_constraints[i].constant

        # Restrict to common variables
        restricted_coeffs: Dict[str, int] = {}
        for v in common_vars:
            if v in combined_coeffs and abs(combined_coeffs[v]) > self._epsilon:
                restricted_coeffs[v] = int(round(combined_coeffs[v]))

        restricted_const = int(round(combined_const))

        # The interpolant is: restricted sum <= 0
        if restricted_coeffs or restricted_const != 0:
            interp = LinearConstraint(
                restricted_coeffs, restricted_const, LinearConstraint.Kind.LE
            )
            return interp.to_expr()

        # Trivial interpolant
        if combined_const < -self._epsilon:
            return FALSE
        return TRUE

    def generate_certificate(
        self,
        constraints: List[LinearConstraint],
        lambdas: List[float],
    ) -> List[str]:
        """Generate a human-readable infeasibility certificate."""
        lines: List[str] = []
        lines.append("Farkas Infeasibility Certificate:")
        lines.append(f"  Number of constraints: {len(constraints)}")
        lines.append(f"  Farkas multipliers: {[round(l, 4) for l in lambdas]}")
        lines.append("")
        lines.append("  Weighted sum of constraints:")
        for i, (c, l) in enumerate(zip(constraints, lambdas)):
            if l > self._epsilon:
                lines.append(f"    λ_{i} = {l:.4f} · ({c})")
        lines.append("")

        # Compute combined
        combined_const = sum(lambdas[i] * constraints[i].constant for i in range(len(constraints)))
        lines.append(f"  Yields: 0 ≤ {combined_const:.4f}")
        if combined_const < -self._epsilon:
            lines.append("  Contradiction! (0 ≤ negative number)")
        else:
            lines.append("  WARNING: Certificate may be invalid")

        return lines


# ===================================================================
# LinearInterpolator
# ===================================================================

class LinearInterpolator:
    """Unsat core-based predicate extraction for quantifier-free linear integer arithmetic (QF_LIA).

    Uses Farkas' lemma to extract interpolants from proofs of unsatisfiability.
    """

    def __init__(self) -> None:
        self._farkas = FarkasLemma()

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        """Compute Craig interpolant for (A, B) where A ∧ B is unsat."""
        start = time.time()

        a_constraints = _expr_to_linear_constraints(formula_a)
        b_constraints = _expr_to_linear_constraints(formula_b)

        if not a_constraints or not b_constraints:
            return None

        all_constraints = a_constraints + b_constraints
        lambdas = self._farkas.compute_farkas_coefficients(all_constraints)
        if lambdas is None:
            return None

        common_vars = formula_a.free_vars() & formula_b.free_vars()
        interp_formula = self._farkas.extract_interpolant(
            a_constraints, b_constraints, lambdas, common_vars
        )

        # Optimize interpolant
        interp_formula = self._optimize_interpolant(interp_formula, common_vars)

        elapsed = (time.time() - start) * 1000.0
        return Interpolant(
            formula=interp_formula,
            variables=interp_formula.free_vars(),
            source_a=formula_a,
            source_b=formula_b,
            theory=Theory.LIA,
            generation_time_ms=elapsed,
        )

    def _optimize_interpolant(
        self, formula: Expr, common_vars: FrozenSet[str]
    ) -> Expr:
        """Simplify and optimize the interpolant."""
        # Remove variables not in common set
        if not formula.free_vars() <= common_vars:
            # Project out non-common variables (approximate with TRUE)
            non_common = formula.free_vars() - common_vars
            if non_common:
                # Substitute non-common vars with 0 (heuristic)
                mapping = {v: _int(0) for v in non_common}
                formula = formula.substitute(mapping)
        return formula

    def interpolate_with_floor_ceiling(
        self,
        formula_a: Expr,
        formula_b: Expr,
    ) -> Optional[Interpolant]:
        """Handle integer-specific floor/ceiling constraints."""
        # For integer arithmetic, x < y is equivalent to x ≤ y - 1
        a_tightened = self._tighten_strict_inequalities(formula_a)
        b_tightened = self._tighten_strict_inequalities(formula_b)
        return self.interpolate(a_tightened, b_tightened)

    def _tighten_strict_inequalities(self, expr: Expr) -> Expr:
        """Convert strict inequalities to non-strict for integers."""
        if isinstance(expr, BinaryExpr):
            left = self._tighten_strict_inequalities(expr.left)
            right = self._tighten_strict_inequalities(expr.right)
            if expr.op == BinaryOp.LT:
                # x < y ↔ x ≤ y - 1
                return _le(left, _sub(right, _int(1)))
            if expr.op == BinaryOp.GT:
                # x > y ↔ x ≥ y + 1 ↔ y ≤ x - 1
                return _le(right, _sub(left, _int(1)))
            return BinaryExpr(expr.op, left, right)
        if isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, self._tighten_strict_inequalities(expr.operand))
        return expr


# ===================================================================
# TypeTagInterpolator
# ===================================================================

class TypeTagInterpolator:
    """Interpolation for type-tag constraints (isinstance, tag_of)."""

    KNOWN_TAGS = frozenset({
        "IntTag", "BoolTag", "StrTag", "FloatTag",
        "NoneTag", "ListTag", "DictTag", "SetTag", "TupleTag",
    })

    def __init__(self) -> None:
        self._type_hierarchy: Dict[str, Set[str]] = {
            "BoolTag": {"IntTag"},  # bool is subtype of int in Python
        }

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        start = time.time()

        a_tags = self._extract_tag_constraints(formula_a)
        b_tags = self._extract_tag_constraints(formula_b)

        if not a_tags and not b_tags:
            return None

        common_vars = formula_a.free_vars() & formula_b.free_vars()
        interp = self._compute_tag_interpolant(a_tags, b_tags, common_vars)

        if interp is None:
            return None

        elapsed = (time.time() - start) * 1000.0
        return Interpolant(
            formula=interp,
            variables=interp.free_vars(),
            source_a=formula_a,
            source_b=formula_b,
            theory=Theory.TYPE_TAG,
            generation_time_ms=elapsed,
        )

    def _extract_tag_constraints(
        self, expr: Expr
    ) -> Dict[str, Tuple[Set[str], Set[str]]]:
        """
        Extract per-variable positive and negative tag constraints.
        Returns: {var: (must_be_tags, must_not_be_tags)}
        """
        result: Dict[str, Tuple[Set[str], Set[str]]] = {}

        if isinstance(expr, BinaryExpr):
            if expr.op == BinaryOp.AND:
                left = self._extract_tag_constraints(expr.left)
                right = self._extract_tag_constraints(expr.right)
                return self._merge_tag_constraints(left, right)

            if expr.op == BinaryOp.ISINSTANCE:
                if isinstance(expr.left, Var) and isinstance(expr.right, Var):
                    if expr.right.name in self.KNOWN_TAGS:
                        var = expr.left.name
                        result[var] = ({expr.right.name}, set())
                        return result

            if expr.op == BinaryOp.EQ:
                if isinstance(expr.left, FuncApp) and expr.left.func_name == "tag_of":
                    if expr.left.args and isinstance(expr.left.args[0], Var):
                        if isinstance(expr.right, Var) and expr.right.name in self.KNOWN_TAGS:
                            var = expr.left.args[0].name
                            result[var] = ({expr.right.name}, set())
                            return result

        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
            if isinstance(expr.operand, BinaryExpr):
                if expr.operand.op == BinaryOp.ISINSTANCE:
                    if isinstance(expr.operand.left, Var) and isinstance(expr.operand.right, Var):
                        if expr.operand.right.name in self.KNOWN_TAGS:
                            var = expr.operand.left.name
                            result[var] = (set(), {expr.operand.right.name})
                            return result

        if isinstance(expr, FuncApp):
            if expr.func_name == "isinstance" and len(expr.args) == 2:
                if isinstance(expr.args[0], Var) and isinstance(expr.args[1], Var):
                    if expr.args[1].name in self.KNOWN_TAGS:
                        var = expr.args[0].name
                        result[var] = ({expr.args[1].name}, set())
                        return result

        return result

    def _merge_tag_constraints(
        self,
        a: Dict[str, Tuple[Set[str], Set[str]]],
        b: Dict[str, Tuple[Set[str], Set[str]]],
    ) -> Dict[str, Tuple[Set[str], Set[str]]]:
        merged = dict(a)
        for v, (pos_b, neg_b) in b.items():
            if v in merged:
                pos_a, neg_a = merged[v]
                merged[v] = (pos_a | pos_b, neg_a | neg_b)
            else:
                merged[v] = (pos_b, neg_b)
        return merged

    def _compute_tag_interpolant(
        self,
        a_tags: Dict[str, Tuple[Set[str], Set[str]]],
        b_tags: Dict[str, Tuple[Set[str], Set[str]]],
        common_vars: FrozenSet[str],
    ) -> Optional[Expr]:
        """Compute an interpolant from tag constraints."""
        conjuncts: List[Expr] = []

        for var in common_vars:
            if var not in a_tags:
                continue
            a_pos, a_neg = a_tags[var]
            b_pos, b_neg = b_tags.get(var, (set(), set()))

            # If A says x must be T and B says x must not be T → contradiction
            # Interpolant: isinstance(x, T)
            for tag in a_pos:
                if tag in b_neg:
                    conjuncts.append(
                        FuncApp("isinstance", (_var(var), _var(tag)))
                    )

            # If A says x must not be T and B says x must be T → contradiction
            # Interpolant: ¬isinstance(x, T)
            for tag in a_neg:
                if tag in b_pos:
                    conjuncts.append(
                        _not(FuncApp("isinstance", (_var(var), _var(tag))))
                    )

            # If A narrows to a set of tags that is disjoint from B's required tags
            if a_pos and b_pos and not a_pos & b_pos:
                for tag in a_pos:
                    conjuncts.append(
                        FuncApp("isinstance", (_var(var), _var(tag)))
                    )

        if not conjuncts:
            return None

        return _conjunction(conjuncts)


# ===================================================================
# NullityInterpolator
# ===================================================================

class NullityInterpolator:
    """Interpolation for nullity constraints (is_none / is_not_none)."""

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        start = time.time()

        a_null = self._extract_nullity(formula_a)
        b_null = self._extract_nullity(formula_b)

        common_vars = formula_a.free_vars() & formula_b.free_vars()
        interp = self._compute_nullity_interpolant(a_null, b_null, common_vars)

        if interp is None:
            return None

        elapsed = (time.time() - start) * 1000.0
        return Interpolant(
            formula=interp,
            variables=interp.free_vars(),
            source_a=formula_a,
            source_b=formula_b,
            theory=Theory.NULLITY,
            generation_time_ms=elapsed,
        )

    def _extract_nullity(
        self, expr: Expr
    ) -> Dict[str, Optional[bool]]:
        """
        Extract nullity constraints: {var: True=must_be_none, False=must_not_be_none}.
        """
        result: Dict[str, Optional[bool]] = {}

        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            left = self._extract_nullity(expr.left)
            right = self._extract_nullity(expr.right)
            result.update(left)
            result.update(right)
            return result

        if isinstance(expr, FuncApp) and expr.func_name == "is_none":
            if expr.args and isinstance(expr.args[0], Var):
                result[expr.args[0].name] = True

        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.IS_NONE:
            if isinstance(expr.operand, Var):
                result[expr.operand.name] = True

        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
            inner = expr.operand
            if isinstance(inner, FuncApp) and inner.func_name == "is_none":
                if inner.args and isinstance(inner.args[0], Var):
                    result[inner.args[0].name] = False
            if isinstance(inner, UnaryExpr) and inner.op == UnaryOp.IS_NONE:
                if isinstance(inner.operand, Var):
                    result[inner.operand.name] = False

        return result

    def _compute_nullity_interpolant(
        self,
        a_null: Dict[str, Optional[bool]],
        b_null: Dict[str, Optional[bool]],
        common_vars: FrozenSet[str],
    ) -> Optional[Expr]:
        conjuncts: List[Expr] = []

        for var in common_vars:
            a_val = a_null.get(var)
            b_val = b_null.get(var)

            if a_val is None or b_val is None:
                continue

            # Contradiction: A says is_none, B says is_not_none (or vice versa)
            if a_val is True and b_val is False:
                conjuncts.append(FuncApp("is_none", (_var(var),)))
            elif a_val is False and b_val is True:
                conjuncts.append(_not(FuncApp("is_none", (_var(var),))))

        if not conjuncts:
            return None
        return _conjunction(conjuncts)


# ===================================================================
# StructuralInterpolator
# ===================================================================

class StructuralInterpolator:
    """Interpolation for structural constraints (hasattr, dict keys)."""

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        start = time.time()

        a_struct = self._extract_structural(formula_a)
        b_struct = self._extract_structural(formula_b)

        common_vars = formula_a.free_vars() & formula_b.free_vars()
        interp = self._compute_structural_interpolant(a_struct, b_struct, common_vars)

        if interp is None:
            return None

        elapsed = (time.time() - start) * 1000.0
        return Interpolant(
            formula=interp,
            variables=interp.free_vars(),
            source_a=formula_a,
            source_b=formula_b,
            theory=Theory.STRUCTURAL,
            generation_time_ms=elapsed,
        )

    def _extract_structural(
        self, expr: Expr
    ) -> Dict[str, Tuple[Set[str], Set[str]]]:
        """Extract {var: (has_attrs, not_has_attrs)}."""
        result: Dict[str, Tuple[Set[str], Set[str]]] = {}

        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            left = self._extract_structural(expr.left)
            right = self._extract_structural(expr.right)
            for v, (lp, ln) in left.items():
                rp, rn = right.get(v, (set(), set()))
                result[v] = (lp | rp, ln | rn)
            for v, (rp, rn) in right.items():
                if v not in result:
                    result[v] = (rp, rn)
            return result

        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.HASATTR:
            if isinstance(expr.left, Var) and isinstance(expr.right, StrLit):
                result[expr.left.name] = ({expr.right.value}, set())

        if isinstance(expr, FuncApp):
            if expr.func_name == "hasattr" and len(expr.args) == 2:
                if isinstance(expr.args[0], Var) and isinstance(expr.args[1], StrLit):
                    result[expr.args[0].name] = ({expr.args[1].value}, set())
            if expr.func_name == "dict_has_key" and len(expr.args) == 2:
                if isinstance(expr.args[0], Var) and isinstance(expr.args[1], StrLit):
                    result[expr.args[0].name] = ({expr.args[1].value}, set())

        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
            inner = expr.operand
            if isinstance(inner, BinaryExpr) and inner.op == BinaryOp.HASATTR:
                if isinstance(inner.left, Var) and isinstance(inner.right, StrLit):
                    result[inner.left.name] = (set(), {inner.right.value})
            if isinstance(inner, FuncApp):
                if inner.func_name == "hasattr" and len(inner.args) == 2:
                    if isinstance(inner.args[0], Var) and isinstance(inner.args[1], StrLit):
                        result[inner.args[0].name] = (set(), {inner.args[1].value})
                if inner.func_name == "dict_has_key" and len(inner.args) == 2:
                    if isinstance(inner.args[0], Var) and isinstance(inner.args[1], StrLit):
                        result[inner.args[0].name] = (set(), {inner.args[1].value})

        return result

    def _compute_structural_interpolant(
        self,
        a_struct: Dict[str, Tuple[Set[str], Set[str]]],
        b_struct: Dict[str, Tuple[Set[str], Set[str]]],
        common_vars: FrozenSet[str],
    ) -> Optional[Expr]:
        conjuncts: List[Expr] = []

        for var in common_vars:
            a_has, a_not = a_struct.get(var, (set(), set()))
            b_has, b_not = b_struct.get(var, (set(), set()))

            for attr in a_has:
                if attr in b_not:
                    conjuncts.append(
                        FuncApp("hasattr", (_var(var), StrLit(attr)))
                    )
            for attr in a_not:
                if attr in b_has:
                    conjuncts.append(
                        _not(FuncApp("hasattr", (_var(var), StrLit(attr))))
                    )

        if not conjuncts:
            return None
        return _conjunction(conjuncts)


# ===================================================================
# CompoundInterpolator
# ===================================================================

class CompoundInterpolator:
    """Combine theory-specific interpolators for the reduced product."""

    def __init__(self) -> None:
        self._linear = LinearInterpolator()
        self._type_tag = TypeTagInterpolator()
        self._nullity = NullityInterpolator()
        self._structural = StructuralInterpolator()

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        """
        Decompose formulas into theory-specific parts, interpolate each,
        and combine the results.
        """
        start = time.time()

        a_parts = self._decompose(formula_a)
        b_parts = self._decompose(formula_b)

        interpolants: List[Expr] = []

        # LIA interpolation
        if Theory.LIA in a_parts and Theory.LIA in b_parts:
            lia_a = _conjunction(a_parts[Theory.LIA])
            lia_b = _conjunction(b_parts[Theory.LIA])
            result = self._linear.interpolate(lia_a, lia_b)
            if result is not None:
                interpolants.append(result.formula)

        # Type tag interpolation
        if Theory.TYPE_TAG in a_parts and Theory.TYPE_TAG in b_parts:
            tag_a = _conjunction(a_parts[Theory.TYPE_TAG])
            tag_b = _conjunction(b_parts[Theory.TYPE_TAG])
            result = self._type_tag.interpolate(tag_a, tag_b)
            if result is not None:
                interpolants.append(result.formula)

        # Nullity interpolation
        if Theory.NULLITY in a_parts and Theory.NULLITY in b_parts:
            null_a = _conjunction(a_parts[Theory.NULLITY])
            null_b = _conjunction(b_parts[Theory.NULLITY])
            result = self._nullity.interpolate(null_a, null_b)
            if result is not None:
                interpolants.append(result.formula)

        # Structural interpolation
        if Theory.STRUCTURAL in a_parts and Theory.STRUCTURAL in b_parts:
            struct_a = _conjunction(a_parts[Theory.STRUCTURAL])
            struct_b = _conjunction(b_parts[Theory.STRUCTURAL])
            result = self._structural.interpolate(struct_a, struct_b)
            if result is not None:
                interpolants.append(result.formula)

        # Handle inter-theory implications
        inter_theory = self._handle_inter_theory(a_parts, b_parts)
        if inter_theory is not None:
            interpolants.append(inter_theory)

        if not interpolants:
            # Fall back to trying each interpolator on the full formulas
            for interp_fn in [
                self._linear.interpolate,
                self._type_tag.interpolate,
                self._nullity.interpolate,
                self._structural.interpolate,
            ]:
                result = interp_fn(formula_a, formula_b)
                if result is not None:
                    elapsed = (time.time() - start) * 1000.0
                    return Interpolant(
                        formula=result.formula,
                        variables=result.variables,
                        source_a=formula_a,
                        source_b=formula_b,
                        theory=result.theory,
                        generation_time_ms=elapsed,
                    )
            return None

        combined = _conjunction(interpolants)
        elapsed = (time.time() - start) * 1000.0

        return Interpolant(
            formula=combined,
            variables=combined.free_vars(),
            source_a=formula_a,
            source_b=formula_b,
            theory=Theory.UNKNOWN,
            generation_time_ms=elapsed,
        )

    def _decompose(self, expr: Expr) -> Dict[Theory, List[Expr]]:
        """Decompose a conjunction into theory-specific parts."""
        parts: Dict[Theory, List[Expr]] = {}
        conjuncts = self._flatten_conjunction(expr)
        for c in conjuncts:
            theory = classify_theory(c)
            if theory not in parts:
                parts[theory] = []
            parts[theory].append(c)
        return parts

    def _flatten_conjunction(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            return (
                self._flatten_conjunction(expr.left)
                + self._flatten_conjunction(expr.right)
            )
        return [expr]

    def _handle_inter_theory(
        self,
        a_parts: Dict[Theory, List[Expr]],
        b_parts: Dict[Theory, List[Expr]],
    ) -> Optional[Expr]:
        """
        Handle implications between theories.
        E.g., is_none(x) implies tag_of(x) == NoneTag.
        """
        inter_implications: List[Expr] = []

        # If A has nullity constraint and B has tag constraint
        if Theory.NULLITY in a_parts and Theory.TYPE_TAG in b_parts:
            for null_expr in a_parts[Theory.NULLITY]:
                for tag_expr in b_parts[Theory.TYPE_TAG]:
                    impl = self._check_null_tag_interaction(null_expr, tag_expr)
                    if impl is not None:
                        inter_implications.append(impl)

        if not inter_implications:
            return None
        return _conjunction(inter_implications)

    def _check_null_tag_interaction(
        self, null_expr: Expr, tag_expr: Expr
    ) -> Optional[Expr]:
        """Check if is_none(x) interacts with isinstance(x, T)."""
        # If null_expr says is_none(x) and tag_expr says not isinstance(x, NoneTag)
        # Then interpolant can be: is_none(x)
        null_vars: Set[str] = set()
        if isinstance(null_expr, FuncApp) and null_expr.func_name == "is_none":
            if null_expr.args and isinstance(null_expr.args[0], Var):
                null_vars.add(null_expr.args[0].name)

        if not null_vars:
            return None

        # Check if tag_expr mentions any of these variables with NoneTag
        tag_info = TypeTagInterpolator()._extract_tag_constraints(tag_expr)
        for v in null_vars:
            if v in tag_info:
                _, neg_tags = tag_info[v]
                if "NoneTag" in neg_tags:
                    return FuncApp("is_none", (_var(v),))

        return None


# ===================================================================
# CounterexamplePath
# ===================================================================

@dataclass
class PathLocation:
    """A location in the program (abstracted)."""
    label: str
    line: int = 0
    function: str = ""


@dataclass
class PathState:
    """Abstract state at a path location."""
    predicates: Dict[str, bool] = field(default_factory=dict)
    variable_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathStep:
    """A single step in a counterexample path."""
    location: PathLocation
    state: PathState
    statement: Optional[Stmt] = None
    guard: Optional[Expr] = None
    annotation: str = ""


@dataclass
class CounterexamplePath:
    """A counterexample path: sequence of (location, state) pairs."""
    steps: List[PathStep] = field(default_factory=list)
    is_spurious: Optional[bool] = None
    concrete_inputs: Optional[Dict[str, Any]] = None

    @property
    def length(self) -> int:
        return len(self.steps)

    def add_step(self, step: PathStep) -> None:
        self.steps.append(step)

    def locations(self) -> List[PathLocation]:
        return [s.location for s in self.steps]

    def guards(self) -> List[Expr]:
        return [s.guard for s in self.steps if s.guard is not None]

    def statements(self) -> List[Stmt]:
        return [s.statement for s in self.steps if s.statement is not None]

    def partition(self, split_index: int) -> Tuple[CounterexamplePath, CounterexamplePath]:
        """Split path into prefix [0..split_index) and suffix [split_index..)."""
        prefix = CounterexamplePath(steps=list(self.steps[:split_index]))
        suffix = CounterexamplePath(steps=list(self.steps[split_index:]))
        return prefix, suffix

    def partition_all(self) -> List[Tuple[CounterexamplePath, CounterexamplePath]]:
        """Generate all possible partitions."""
        partitions: List[Tuple[CounterexamplePath, CounterexamplePath]] = []
        for i in range(1, len(self.steps)):
            partitions.append(self.partition(i))
        return partitions

    def extract_path_formula(self) -> Expr:
        """Encode entire path as a single formula."""
        conjuncts: List[Expr] = []
        for step in self.steps:
            if step.guard is not None:
                conjuncts.append(step.guard)
            if step.statement is not None:
                stmt_formula = self._stmt_to_formula(step.statement)
                if stmt_formula is not None:
                    conjuncts.append(stmt_formula)
        return _conjunction(conjuncts) if conjuncts else TRUE

    def _stmt_to_formula(self, stmt: Stmt) -> Optional[Expr]:
        if isinstance(stmt, AssumeStmt):
            return stmt.condition
        if isinstance(stmt, AssertStmt):
            return stmt.condition
        if isinstance(stmt, AssignStmt):
            return _eq(_var(stmt.target), stmt.value)
        return None


# ===================================================================
# PathEncoder
# ===================================================================

class PathEncoder:
    """Encode a counterexample path as an SMT formula using SSA."""

    def __init__(self) -> None:
        self._ssa_counters: Dict[str, int] = {}

    def _fresh(self, var: str) -> str:
        count = self._ssa_counters.get(var, 0) + 1
        self._ssa_counters[var] = count
        return f"{var}__ssa_{count}"

    def _current(self, var: str) -> str:
        count = self._ssa_counters.get(var, 0)
        if count == 0:
            return var
        return f"{var}__ssa_{count}"

    def encode(self, path: CounterexamplePath) -> Tuple[Expr, List[Tuple[str, str]]]:
        """
        Encode a path as an SMT formula.
        Returns (formula, list of (original_var, ssa_var) pairs).
        """
        self._ssa_counters.clear()
        constraints: List[Expr] = []
        var_mappings: List[Tuple[str, str]] = []

        for step in path.steps:
            if step.guard is not None:
                renamed_guard = self._rename_expr(step.guard)
                constraints.append(renamed_guard)

            if step.statement is not None:
                self._encode_stmt(step.statement, constraints, var_mappings)

        return _conjunction(constraints), var_mappings

    def _encode_stmt(
        self,
        stmt: Stmt,
        constraints: List[Expr],
        var_mappings: List[Tuple[str, str]],
    ) -> None:
        if isinstance(stmt, AssignStmt):
            rhs = self._rename_expr(stmt.value)
            new_name = self._fresh(stmt.target)
            constraints.append(_eq(_var(new_name), rhs))
            var_mappings.append((stmt.target, new_name))

        elif isinstance(stmt, AssumeStmt):
            constraints.append(self._rename_expr(stmt.condition))

        elif isinstance(stmt, AssertStmt):
            constraints.append(self._rename_expr(stmt.condition))

        elif isinstance(stmt, CallStmt):
            if stmt.target:
                new_name = self._fresh(stmt.target)
                var_mappings.append((stmt.target, new_name))
                # Call result is unconstrained (fresh variable)

        elif isinstance(stmt, SeqStmt):
            self._encode_stmt(stmt.first, constraints, var_mappings)
            self._encode_stmt(stmt.second, constraints, var_mappings)

        elif isinstance(stmt, SkipStmt):
            pass

    def _rename_expr(self, expr: Expr) -> Expr:
        if isinstance(expr, Var):
            return _var(self._current(expr.name))
        if isinstance(expr, (IntLit, BoolLit, StrLit, NoneLit)):
            return expr
        if isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, self._rename_expr(expr.operand))
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                expr.op,
                self._rename_expr(expr.left),
                self._rename_expr(expr.right),
            )
        if isinstance(expr, FuncApp):
            return FuncApp(
                expr.func_name,
                tuple(self._rename_expr(a) for a in expr.args),
            )
        if isinstance(expr, IteExpr):
            return IteExpr(
                self._rename_expr(expr.cond),
                self._rename_expr(expr.then_branch),
                self._rename_expr(expr.else_branch),
            )
        if isinstance(expr, QuantifiedExpr):
            return QuantifiedExpr(
                expr.quantifier, expr.var_name, expr.var_sort,
                self._rename_expr(expr.body),
            )
        return expr

    def encode_with_partitions(
        self, path: CounterexamplePath
    ) -> List[Tuple[Expr, Expr]]:
        """
        Encode path and generate all (A, B) partitions for interpolation.
        A = constraints from prefix, B = constraints from suffix.
        """
        self._ssa_counters.clear()
        step_formulas: List[Expr] = []

        for step in path.steps:
            step_constraints: List[Expr] = []
            if step.guard is not None:
                step_constraints.append(self._rename_expr(step.guard))
            if step.statement is not None:
                stmt_constraints: List[Expr] = []
                self._encode_stmt(step.statement, stmt_constraints, [])
                step_constraints.extend(stmt_constraints)
            if step_constraints:
                step_formulas.append(_conjunction(step_constraints))
            else:
                step_formulas.append(TRUE)

        partitions: List[Tuple[Expr, Expr]] = []
        for i in range(1, len(step_formulas)):
            a = _conjunction(step_formulas[:i])
            b = _conjunction(step_formulas[i:])
            partitions.append((a, b))

        return partitions


# ===================================================================
# SpuriousnessChecker
# ===================================================================

@dataclass
class SpuriousnessResult:
    is_spurious: bool
    model: Optional[Dict[str, Any]] = None   # if not spurious (SAT model)
    unsat_core: Optional[List[Expr]] = None  # if spurious
    infeasible_step: Optional[int] = None     # which step makes it infeasible
    explanation: str = ""


class SpuriousnessChecker:
    """Check whether a counterexample path is spurious."""

    def __init__(self) -> None:
        self._encoder = PathEncoder()

    def check(self, path: CounterexamplePath) -> SpuriousnessResult:
        """
        Check if a counterexample path is spurious.

        Encode the path as an SMT formula:
          - SAT → real counterexample, extract concrete inputs
          - UNSAT → spurious, extract unsat core
        """
        formula, var_mappings = self._encoder.encode(path)

        # Attempt to check satisfiability (lightweight check)
        result = self._check_sat(formula, path)
        return result

    def _check_sat(
        self, formula: Expr, path: CounterexamplePath
    ) -> SpuriousnessResult:
        """
        Lightweight satisfiability check.
        A full implementation would invoke an SMT solver.
        """
        # Check for obvious contradictions
        contradiction = self._find_contradiction(formula)
        if contradiction is not None:
            step_idx, explanation = contradiction
            return SpuriousnessResult(
                is_spurious=True,
                infeasible_step=step_idx,
                explanation=explanation,
            )

        # Check for unsatisfiable linear constraints
        constraints = _expr_to_linear_constraints(formula)
        if constraints:
            farkas = FarkasLemma()
            lambdas = farkas.compute_farkas_coefficients(constraints)
            if lambdas is not None:
                return SpuriousnessResult(
                    is_spurious=True,
                    explanation="Linear constraint system is infeasible (Farkas certificate found)",
                )

        # Check for type contradictions
        tag_interp = TypeTagInterpolator()
        tag_constraints = tag_interp._extract_tag_constraints(formula)
        for var, (pos, neg) in tag_constraints.items():
            if pos & neg:
                return SpuriousnessResult(
                    is_spurious=True,
                    explanation=f"Type contradiction for {var}: must be and must not be {pos & neg}",
                )

        # Check nullity contradictions
        null_interp = NullityInterpolator()
        null_constraints = null_interp._extract_nullity(formula)
        for var, val in null_constraints.items():
            # This alone can't find contradictions without a second source
            pass

        # If no contradiction found, conservatively say unknown
        # (in full implementation, would call SMT solver)
        return SpuriousnessResult(
            is_spurious=False,
            explanation="No obvious contradiction found (would need SMT solver for definitive answer)",
        )

    def _find_contradiction(
        self, formula: Expr
    ) -> Optional[Tuple[int, str]]:
        """Find obvious contradictions in a formula."""
        conjuncts = self._flatten_and(formula)

        # Check for x = a ∧ x = b where a ≠ b
        equalities: Dict[str, List[Expr]] = {}
        for i, c in enumerate(conjuncts):
            if isinstance(c, BinaryExpr) and c.op == BinaryOp.EQ:
                if isinstance(c.left, Var):
                    equalities.setdefault(c.left.name, []).append(c.right)
                if isinstance(c.right, Var):
                    equalities.setdefault(c.right.name, []).append(c.left)

        for var, values in equalities.items():
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if (isinstance(values[i], IntLit) and isinstance(values[j], IntLit)
                            and values[i].value != values[j].value):
                        return (0, f"Contradiction: {var} = {values[i]} and {var} = {values[j]}")

        # Check for p ∧ ¬p
        pos_formulas: Set[str] = set()
        neg_formulas: Set[str] = set()
        for c in conjuncts:
            if isinstance(c, UnaryExpr) and c.op == UnaryOp.NOT:
                neg_formulas.add(str(c.operand))
            else:
                pos_formulas.add(str(c))
        overlap = pos_formulas & neg_formulas
        if overlap:
            f = next(iter(overlap))
            return (0, f"Contradiction: both {f} and ¬({f}) hold")

        return None

    def _flatten_and(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]

    def generate_smt_check(self, path: CounterexamplePath) -> str:
        """Generate SMT-LIB script to check spuriousness."""
        formula, var_mappings = self._encoder.encode(path)
        all_vars = formula.free_vars()

        lines: List[str] = []
        lines.append("; Spuriousness check for counterexample path")
        lines.append("(set-logic QF_LIA)")
        for v in sorted(all_vars):
            lines.append(f"(declare-const {v} Int)")
        lines.append(f"(assert {formula.to_smt()})")
        lines.append("(check-sat)")
        lines.append("(get-model)")
        lines.append("(exit)")
        return "\n".join(lines)


# ===================================================================
# PredicateProjection
# ===================================================================

@dataclass
class PredicateTemplate:
    """Template for predicates in the predicate language P."""
    name: str
    arity: int
    builder: Callable[..., Expr]
    description: str = ""


class PredicateProjection:
    """Project interpolants into the predicate language P."""

    def __init__(self) -> None:
        self._templates: List[PredicateTemplate] = self._default_templates()

    def _default_templates(self) -> List[PredicateTemplate]:
        return [
            PredicateTemplate(
                "non_negative", 1,
                lambda v: _ge(_var(v), _int(0)),
                "x >= 0",
            ),
            PredicateTemplate(
                "positive", 1,
                lambda v: _gt(_var(v), _int(0)),
                "x > 0",
            ),
            PredicateTemplate(
                "zero", 1,
                lambda v: _eq(_var(v), _int(0)),
                "x == 0",
            ),
            PredicateTemplate(
                "non_zero", 1,
                lambda v: _ne(_var(v), _int(0)),
                "x != 0",
            ),
            PredicateTemplate(
                "negative", 1,
                lambda v: _lt(_var(v), _int(0)),
                "x < 0",
            ),
            PredicateTemplate(
                "less_than", 2,
                lambda v1, v2: _lt(_var(v1), _var(v2)),
                "x < y",
            ),
            PredicateTemplate(
                "less_equal", 2,
                lambda v1, v2: _le(_var(v1), _var(v2)),
                "x <= y",
            ),
            PredicateTemplate(
                "equal", 2,
                lambda v1, v2: _eq(_var(v1), _var(v2)),
                "x == y",
            ),
            PredicateTemplate(
                "not_none", 1,
                lambda v: _not(FuncApp("is_none", (_var(v),))),
                "x is not None",
            ),
            PredicateTemplate(
                "is_none", 1,
                lambda v: FuncApp("is_none", (_var(v),)),
                "x is None",
            ),
            PredicateTemplate(
                "is_int", 1,
                lambda v: FuncApp("isinstance", (_var(v), _var("IntTag"))),
                "isinstance(x, int)",
            ),
            PredicateTemplate(
                "is_str", 1,
                lambda v: FuncApp("isinstance", (_var(v), _var("StrTag"))),
                "isinstance(x, str)",
            ),
            PredicateTemplate(
                "is_list", 1,
                lambda v: FuncApp("isinstance", (_var(v), _var("ListTag"))),
                "isinstance(x, list)",
            ),
            PredicateTemplate(
                "is_dict", 1,
                lambda v: FuncApp("isinstance", (_var(v), _var("DictTag"))),
                "isinstance(x, dict)",
            ),
            PredicateTemplate(
                "is_bool", 1,
                lambda v: FuncApp("isinstance", (_var(v), _var("BoolTag"))),
                "isinstance(x, bool)",
            ),
            PredicateTemplate(
                "len_positive", 1,
                lambda v: _gt(FuncApp("list_len", (_var(v),)), _int(0)),
                "len(x) > 0",
            ),
            PredicateTemplate(
                "bounded_above", 2,
                lambda v, c: _le(_var(v), _int(int(c))),
                "x <= c",
            ),
            PredicateTemplate(
                "bounded_below", 2,
                lambda v, c: _ge(_var(v), _int(int(c))),
                "x >= c",
            ),
        ]

    def add_template(self, template: PredicateTemplate) -> None:
        self._templates.append(template)

    def project_to_P(
        self,
        interpolant: Interpolant,
        program_vars: Optional[List[str]] = None,
    ) -> List[Expr]:
        """
        Project an interpolant into the predicate language P.

        Returns a list of predicates from P that:
        1. Are implied by the interpolant (or approximate it from above)
        2. Use only variables from the interpolant
        3. Eliminate the spurious counterexample
        """
        interp_vars = list(interpolant.variables)
        if program_vars:
            interp_vars = [v for v in interp_vars if v in program_vars]

        candidates: List[Expr] = []

        # Try to match interpolant directly against templates
        direct_match = self._match_template(interpolant.formula)
        if direct_match is not None:
            candidates.append(direct_match)

        # Generate candidate predicates from templates
        for template in self._templates:
            if template.arity == 1:
                for v in interp_vars:
                    try:
                        pred = template.builder(v)
                        if self._is_implied_by(interpolant.formula, pred):
                            candidates.append(pred)
                    except (TypeError, ValueError):
                        pass
            elif template.arity == 2:
                for v1 in interp_vars:
                    for v2 in interp_vars:
                        if v1 != v2:
                            try:
                                pred = template.builder(v1, v2)
                                if self._is_implied_by(interpolant.formula, pred):
                                    candidates.append(pred)
                            except (TypeError, ValueError):
                                pass

        # Extract constants from the interpolant for bounded predicates
        constants = self._extract_constants(interpolant.formula)
        for v in interp_vars:
            for c in constants:
                pred_le = _le(_var(v), _int(c))
                if self._is_implied_by(interpolant.formula, pred_le):
                    candidates.append(pred_le)
                pred_ge = _ge(_var(v), _int(c))
                if self._is_implied_by(interpolant.formula, pred_ge):
                    candidates.append(pred_ge)

        # If no candidates found, use the interpolant itself as a predicate
        if not candidates:
            candidates.append(interpolant.formula)

        # Deduplicate
        seen: Set[str] = set()
        unique: List[Expr] = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                unique.append(c)

        return unique

    def _match_template(self, formula: Expr) -> Optional[Expr]:
        """Try to match formula against known templates."""
        for template in self._templates:
            if template.arity == 1:
                vars_in = list(formula.free_vars())
                for v in vars_in:
                    try:
                        expected = template.builder(v)
                        if str(expected) == str(formula):
                            return formula
                    except (TypeError, ValueError):
                        pass
        return None

    def _is_implied_by(self, antecedent: Expr, consequent: Expr) -> bool:
        """
        Check if antecedent ⇒ consequent (lightweight check).
        A full implementation would use an SMT solver.
        """
        if str(antecedent) == str(consequent):
            return True

        # Check structural patterns
        if isinstance(antecedent, BinaryExpr) and isinstance(consequent, BinaryExpr):
            # x <= c1 implies x <= c2 if c1 <= c2
            if (antecedent.op == BinaryOp.LE and consequent.op == BinaryOp.LE
                    and str(antecedent.left) == str(consequent.left)):
                if isinstance(antecedent.right, IntLit) and isinstance(consequent.right, IntLit):
                    return antecedent.right.value <= consequent.right.value
            # x >= c1 implies x >= c2 if c1 >= c2
            if (antecedent.op == BinaryOp.GE and consequent.op == BinaryOp.GE
                    and str(antecedent.left) == str(consequent.left)):
                if isinstance(antecedent.right, IntLit) and isinstance(consequent.right, IntLit):
                    return antecedent.right.value >= consequent.right.value
            # x < c1 implies x < c2 if c1 <= c2
            if (antecedent.op == BinaryOp.LT and consequent.op == BinaryOp.LT
                    and str(antecedent.left) == str(consequent.left)):
                if isinstance(antecedent.right, IntLit) and isinstance(consequent.right, IntLit):
                    return antecedent.right.value <= consequent.right.value
            # x >= 0 is implied by x > 0
            if (antecedent.op == BinaryOp.GT and consequent.op == BinaryOp.GE
                    and str(antecedent.left) == str(consequent.left)):
                if isinstance(antecedent.right, IntLit) and isinstance(consequent.right, IntLit):
                    return antecedent.right.value >= consequent.right.value

        # Check if consequent appears as a conjunct of antecedent
        if isinstance(antecedent, BinaryExpr) and antecedent.op == BinaryOp.AND:
            return (self._is_implied_by(antecedent.left, consequent)
                    or self._is_implied_by(antecedent.right, consequent))

        return False

    def _extract_constants(self, expr: Expr) -> Set[int]:
        """Extract integer constants from an expression."""
        constants: Set[int] = set()
        if isinstance(expr, IntLit):
            constants.add(expr.value)
        elif isinstance(expr, UnaryExpr):
            constants |= self._extract_constants(expr.operand)
        elif isinstance(expr, BinaryExpr):
            constants |= self._extract_constants(expr.left)
            constants |= self._extract_constants(expr.right)
        elif isinstance(expr, FuncApp):
            for a in expr.args:
                constants |= self._extract_constants(a)
        elif isinstance(expr, IteExpr):
            constants |= self._extract_constants(expr.cond)
            constants |= self._extract_constants(expr.then_branch)
            constants |= self._extract_constants(expr.else_branch)
        return constants


# ===================================================================
# PredicateRefinement
# ===================================================================

@dataclass
class PredicateQuality:
    """Quality metrics for a predicate."""
    predicate: Expr
    size: int
    eliminates_ce: bool
    redundant: bool
    subsumed: bool
    score: float = 0.0


class PredicateRefinement:
    """Refine the predicate set from interpolants."""

    def __init__(self) -> None:
        self._projection = PredicateProjection()

    def refine(
        self,
        current_predicates: List[Expr],
        interpolant: Interpolant,
        program_vars: Optional[List[str]] = None,
    ) -> List[Expr]:
        """
        Refine predicate set using an interpolant.
        Returns new predicates to add.
        """
        # Project interpolant to P
        candidates = self._projection.project_to_P(interpolant, program_vars)

        # Filter candidates
        new_predicates: List[Expr] = []
        current_strs = {str(p) for p in current_predicates}

        for candidate in candidates:
            cand_str = str(candidate)

            # Skip if already present
            if cand_str in current_strs:
                continue

            # Check for redundancy
            if self._is_redundant(candidate, current_predicates):
                continue

            # Check for subsumption
            if self._is_subsumed(candidate, current_predicates):
                continue

            new_predicates.append(candidate)

        # Rank by quality
        ranked = self._rank_predicates(new_predicates, current_predicates, interpolant)

        return ranked

    def _is_redundant(self, pred: Expr, existing: List[Expr]) -> bool:
        """Check if pred is logically equivalent to some existing predicate."""
        pred_str = str(pred)
        for e in existing:
            if str(e) == pred_str:
                return True
            # Check negation equivalence
            if isinstance(pred, UnaryExpr) and pred.op == UnaryOp.NOT:
                if str(pred.operand) == str(e):
                    return False  # negation is not redundant
        return False

    def _is_subsumed(self, pred: Expr, existing: List[Expr]) -> bool:
        """Check if pred is implied by some existing predicate."""
        for e in existing:
            if self._implies(e, pred):
                return True
        return False

    def _implies(self, a: Expr, b: Expr) -> bool:
        """Lightweight implication check."""
        if str(a) == str(b):
            return True
        # x >= c1 implies x >= c2 if c1 >= c2
        if isinstance(a, BinaryExpr) and isinstance(b, BinaryExpr):
            if (a.op == BinaryOp.GE and b.op == BinaryOp.GE
                    and str(a.left) == str(b.left)):
                if isinstance(a.right, IntLit) and isinstance(b.right, IntLit):
                    return a.right.value >= b.right.value
            if (a.op == BinaryOp.LE and b.op == BinaryOp.LE
                    and str(a.left) == str(b.left)):
                if isinstance(a.right, IntLit) and isinstance(b.right, IntLit):
                    return a.right.value <= b.right.value
        return False

    def _rank_predicates(
        self,
        candidates: List[Expr],
        existing: List[Expr],
        interpolant: Interpolant,
    ) -> List[Expr]:
        """Rank predicates by quality (smaller, more relevant first)."""
        scored: List[Tuple[float, Expr]] = []
        for c in candidates:
            size = _expr_size(c)
            # Prefer smaller predicates
            score = 1.0 / (1.0 + size)
            # Prefer predicates that share variables with interpolant
            shared_vars = c.free_vars() & interpolant.variables
            score += 0.1 * len(shared_vars)
            # Prefer predicates not involving SSA variables
            ssa_count = sum(1 for v in c.free_vars() if "__ssa_" in v)
            score -= 0.2 * ssa_count
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]


# ===================================================================
# InterpolationCache
# ===================================================================

class InterpolationCache:
    """Cache interpolation results to avoid redundant computation."""

    def __init__(self) -> None:
        self._cache: Dict[str, Interpolant] = {}
        self._hits: int = 0
        self._misses: int = 0

    def _make_key(self, formula_a: Expr, formula_b: Expr) -> str:
        """Create a cache key from two formulas (structural, modulo var renaming)."""
        return _structural_hash(formula_a) + "|" + _structural_hash(formula_b)

    def get(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        key = self._make_key(formula_a, formula_b)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(
        self, formula_a: Expr, formula_b: Expr, interpolant: Interpolant
    ) -> None:
        key = self._make_key(formula_a, formula_b)
        self._cache[key] = interpolant

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def total_queries(self) -> int:
        return self._hits + self._misses


# ===================================================================
# InterpolationStatistics
# ===================================================================

@dataclass
class InterpolationStatistics:
    """Statistics about interpolation operations."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time_ms: float = 0.0
    total_interpolant_size: int = 0
    predicates_discovered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    per_theory_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def average_time_ms(self) -> float:
        return self.total_time_ms / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def average_interpolant_size(self) -> float:
        return (self.total_interpolant_size / self.successful_queries
                if self.successful_queries > 0 else 0.0)

    @property
    def success_rate(self) -> float:
        return (self.successful_queries / self.total_queries
                if self.total_queries > 0 else 0.0)

    @property
    def predicate_discovery_rate(self) -> float:
        return (self.predicates_discovered / self.successful_queries
                if self.successful_queries > 0 else 0.0)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def record_query(
        self,
        success: bool,
        time_ms: float,
        interpolant_size: int = 0,
        predicates: int = 0,
        theory: str = "unknown",
        cached: bool = False,
    ) -> None:
        self.total_queries += 1
        self.total_time_ms += time_ms
        if success:
            self.successful_queries += 1
            self.total_interpolant_size += interpolant_size
            self.predicates_discovered += predicates
        else:
            self.failed_queries += 1
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.per_theory_counts[theory] = self.per_theory_counts.get(theory, 0) + 1

    def summary(self) -> str:
        lines: List[str] = []
        lines.append(f"Total interpolation queries: {self.total_queries}")
        lines.append(f"Successful: {self.successful_queries}")
        lines.append(f"Failed: {self.failed_queries}")
        lines.append(f"Success rate: {self.success_rate:.1%}")
        lines.append(f"Average time: {self.average_time_ms:.1f}ms")
        lines.append(f"Average interpolant size: {self.average_interpolant_size:.1f}")
        lines.append(f"Predicates discovered: {self.predicates_discovered}")
        lines.append(f"Predicate discovery rate: {self.predicate_discovery_rate:.1f}")
        lines.append(f"Cache hit rate: {self.cache_hit_rate:.1%}")
        if self.per_theory_counts:
            lines.append("Per-theory counts:")
            for theory, count in sorted(self.per_theory_counts.items()):
                lines.append(f"  {theory}: {count}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.success_rate,
            "average_time_ms": self.average_time_ms,
            "average_interpolant_size": self.average_interpolant_size,
            "predicates_discovered": self.predicates_discovered,
            "cache_hit_rate": self.cache_hit_rate,
            "per_theory_counts": dict(self.per_theory_counts),
        }


# ===================================================================
# ProofExplainer
# ===================================================================

class ProofExplainer:
    """Generate human-readable explanations for why a CE is spurious."""

    def explain_spurious(
        self,
        path: CounterexamplePath,
        result: SpuriousnessResult,
        interpolant: Optional[Interpolant] = None,
    ) -> str:
        """Explain why a counterexample is spurious."""
        parts: List[str] = []
        parts.append("=" * 60)
        parts.append("COUNTEREXAMPLE ANALYSIS")
        parts.append("=" * 60)
        parts.append("")

        # Path description
        parts.append(f"Path length: {path.length} steps")
        for i, step in enumerate(path.steps):
            loc = step.location
            parts.append(
                f"  Step {i}: {loc.function}:{loc.label} (line {loc.line})"
            )
            if step.guard is not None:
                parts.append(f"    Guard: {step.guard}")
            if step.statement is not None:
                parts.append(f"    Statement: {step.statement}")
        parts.append("")

        # Spuriousness result
        if result.is_spurious:
            parts.append("VERDICT: SPURIOUS (path is infeasible)")
            parts.append("")

            if result.infeasible_step is not None:
                parts.append(
                    f"Infeasible at step: {result.infeasible_step}"
                )
                if result.infeasible_step < len(path.steps):
                    step = path.steps[result.infeasible_step]
                    parts.append(
                        f"  Location: {step.location.function}:{step.location.label}"
                    )
                    if step.guard is not None:
                        parts.append(f"  Guard that makes path infeasible: {step.guard}")

            if result.explanation:
                parts.append(f"Explanation: {result.explanation}")
            parts.append("")

            if interpolant is not None:
                parts.append("INTERPOLANT (predicate to add):")
                parts.append(f"  Formula: {interpolant.formula}")
                parts.append(f"  Variables: {sorted(interpolant.variables)}")
                parts.append(f"  Theory: {interpolant.theory.value}")
                parts.append(f"  Size: {interpolant.size}")
                parts.append("")

                parts.append("EXPLANATION:")
                parts.append(
                    "  Adding this predicate to the abstract domain will"
                )
                parts.append(
                    "  refine the abstraction to eliminate this spurious"
                )
                parts.append(
                    "  counterexample in subsequent contract discovery iterations."
                )
        else:
            parts.append("VERDICT: REAL COUNTEREXAMPLE")
            parts.append("")
            if result.model:
                parts.append("Concrete inputs that trigger this path:")
                for var, val in sorted(result.model.items()):
                    parts.append(f"  {var} = {val}")
            if result.explanation:
                parts.append(f"Note: {result.explanation}")

        parts.append("")
        parts.append("=" * 60)
        return "\n".join(parts)

    def explain_refinement(
        self,
        old_predicates: List[Expr],
        new_predicates: List[Expr],
        interpolant: Interpolant,
    ) -> str:
        """Explain how predicates were refined."""
        parts: List[str] = []
        parts.append("PREDICATE REFINEMENT")
        parts.append("-" * 40)
        parts.append(f"Interpolant: {interpolant.formula}")
        parts.append(f"Theory: {interpolant.theory.value}")
        parts.append("")

        parts.append(f"Previous predicates ({len(old_predicates)}):")
        for p in old_predicates:
            parts.append(f"  • {p}")
        parts.append("")

        added = [p for p in new_predicates if str(p) not in {str(o) for o in old_predicates}]
        if added:
            parts.append(f"New predicates added ({len(added)}):")
            for p in added:
                parts.append(f"  + {p}")
        else:
            parts.append("No new predicates added.")

        parts.append("")
        parts.append(f"Total predicates: {len(old_predicates) + len(added)}")
        return "\n".join(parts)

    def explain_guard_infeasibility(
        self,
        path: CounterexamplePath,
    ) -> str:
        """Explain which guard makes the path infeasible."""
        guards = path.guards()
        if not guards:
            return "No guards on path."

        parts: List[str] = []
        parts.append("Guard analysis:")
        accumulated = TRUE
        for i, guard in enumerate(guards):
            accumulated = _and(accumulated, guard)
            parts.append(f"  After guard {i}: {guard}")
            parts.append(f"    Accumulated: {accumulated}")

            # Check for contradiction
            checker = SpuriousnessChecker()
            flat = checker._flatten_and(accumulated)
            pos = {str(f) for f in flat if not (isinstance(f, UnaryExpr) and f.op == UnaryOp.NOT)}
            neg = {str(f.operand) if isinstance(f, UnaryExpr) and f.op == UnaryOp.NOT else "" for f in flat}
            if pos & neg:
                parts.append(f"    *** INFEASIBLE at guard {i} ***")
                parts.append(f"    Contradicts: {pos & neg}")
                break

        return "\n".join(parts)


# ===================================================================
# InterpolationEngine (main driver)
# ===================================================================

class InterpolationEngine:
    """
    Main interpolation driver that orchestrates theory-specific
    interpolators, caching, and predicate projection.
    """

    def __init__(self) -> None:
        self._compound = CompoundInterpolator()
        self._linear = LinearInterpolator()
        self._type_tag = TypeTagInterpolator()
        self._nullity = NullityInterpolator()
        self._structural = StructuralInterpolator()
        self._cache = InterpolationCache()
        self._stats = InterpolationStatistics()
        self._projection = PredicateProjection()
        self._refinement = PredicateRefinement()
        self._encoder = PathEncoder()
        self._checker = SpuriousnessChecker()

    @property
    def statistics(self) -> InterpolationStatistics:
        return self._stats

    @property
    def cache(self) -> InterpolationCache:
        return self._cache

    # ---- core interpolation -----------------------------------------------

    def interpolate(
        self, formula_a: Expr, formula_b: Expr
    ) -> Optional[Interpolant]:
        """Compute Craig interpolant for A ∧ B is unsat."""
        start = time.time()

        # Check cache
        cached = self._cache.get(formula_a, formula_b)
        if cached is not None:
            elapsed = (time.time() - start) * 1000.0
            self._stats.record_query(
                True, elapsed, cached.size, theory=cached.theory.value, cached=True,
            )
            return cached

        # Try compound interpolation
        result = self._compound.interpolate(formula_a, formula_b)

        elapsed = (time.time() - start) * 1000.0

        if result is not None:
            self._cache.put(formula_a, formula_b, result)
            self._stats.record_query(
                True, elapsed, result.size,
                theory=result.theory.value, cached=False,
            )
        else:
            self._stats.record_query(False, elapsed, cached=False)

        return result

    # ---- sequence interpolation -------------------------------------------

    def sequence_interpolation(
        self, formulas: List[Expr]
    ) -> List[Optional[Interpolant]]:
        """
        Compute sequence interpolants I_1, ..., I_{n-1} for formulas F_1, ..., F_n
        such that:
          F_1 |= I_1
          I_i ∧ F_{i+1} |= I_{i+1}
          I_{n-1} ∧ F_n is unsat
        """
        n = len(formulas)
        if n < 2:
            return []

        interpolants: List[Optional[Interpolant]] = []

        for i in range(1, n):
            prefix = _conjunction(formulas[:i])
            suffix = _conjunction(formulas[i:])

            interp = self.interpolate(prefix, suffix)
            interpolants.append(interp)

        return interpolants

    # ---- tree interpolation -----------------------------------------------

    def tree_interpolation(
        self, formula_tree: Dict[str, Tuple[Expr, List[str]]]
    ) -> Dict[str, Optional[Interpolant]]:
        """
        Compute tree interpolants for a formula tree.

        formula_tree: {node_id: (formula, [child_ids])}
        Returns: {node_id: interpolant_at_node}
        """
        result: Dict[str, Optional[Interpolant]] = {}

        # Find root (node not referenced as child by anyone)
        all_children: Set[str] = set()
        for _, (_, children) in formula_tree.items():
            for c in children:
                all_children.add(c)
        roots = [nid for nid in formula_tree if nid not in all_children]

        if not roots:
            return result

        root = roots[0]
        self._tree_interpolate_rec(root, formula_tree, result)
        return result

    def _tree_interpolate_rec(
        self,
        node_id: str,
        tree: Dict[str, Tuple[Expr, List[str]]],
        result: Dict[str, Optional[Interpolant]],
    ) -> Expr:
        """Recursively compute tree interpolants (post-order)."""
        formula, children = tree[node_id]

        if not children:
            result[node_id] = None
            return formula

        # Recurse on children first
        child_formulas: List[Expr] = []
        for child_id in children:
            child_f = self._tree_interpolate_rec(child_id, tree, result)
            child_formulas.append(child_f)

        # Compute interpolant: A = node formula ∧ child interpolants, B = rest
        a_formula = _conjunction([formula] + child_formulas)
        # B is the conjunction of all other nodes' formulas
        other_formulas: List[Expr] = []
        for nid, (nf, _) in tree.items():
            if nid != node_id and nid not in self._descendants(node_id, tree):
                other_formulas.append(nf)

        if other_formulas:
            b_formula = _conjunction(other_formulas)
            interp = self.interpolate(a_formula, b_formula)
            result[node_id] = interp
            if interp is not None:
                return interp.formula

        result[node_id] = None
        return a_formula

    def _descendants(
        self, node_id: str, tree: Dict[str, Tuple[Expr, List[str]]]
    ) -> Set[str]:
        """Get all descendants of a node."""
        _, children = tree.get(node_id, (TRUE, []))
        result: Set[str] = set(children)
        for c in children:
            result |= self._descendants(c, tree)
        return result

    # ---- Contract discovery integration ------------------------------------------------

    def check_and_refine(
        self,
        path: CounterexamplePath,
        current_predicates: List[Expr],
        program_vars: Optional[List[str]] = None,
    ) -> Tuple[SpuriousnessResult, List[Expr]]:
        """
        Full contract discovery refinement step:
        1. Check if counterexample is spurious
        2. If spurious, extract interpolant
        3. Project to predicate language
        4. Return new predicates
        """
        # Step 1: Check spuriousness
        spur_result = self._checker.check(path)

        if not spur_result.is_spurious:
            return spur_result, []

        # Step 2: Extract interpolant via sequence interpolation
        partitions = self._encoder.encode_with_partitions(path)
        new_predicates: List[Expr] = []

        for a_formula, b_formula in partitions:
            interp = self.interpolate(a_formula, b_formula)
            if interp is not None:
                # Step 3: Project and refine
                refined = self._refinement.refine(
                    current_predicates + new_predicates,
                    interp,
                    program_vars,
                )
                new_predicates.extend(refined)
                self._stats.record_query(
                    True, interp.generation_time_ms,
                    interp.size, len(refined),
                    theory=interp.theory.value,
                )

        return spur_result, new_predicates

    # ---- SMT-LIB generation for external solver ---------------------------

    def generate_interpolation_query(
        self, formula_a: Expr, formula_b: Expr
    ) -> str:
        """Generate SMT-LIB 2.6 interpolation query."""
        all_vars = formula_a.free_vars() | formula_b.free_vars()
        lines: List[str] = []
        lines.append("; Craig Interpolation Query")
        lines.append("(set-logic QF_LIA)")
        lines.append("(set-option :produce-interpolants true)")
        for v in sorted(all_vars):
            lines.append(f"(declare-const {v} Int)")
        lines.append(f"(assert (! {formula_a.to_smt()} :named partA))")
        lines.append(f"(assert (! {formula_b.to_smt()} :named partB))")
        lines.append("(check-sat)")
        lines.append("(get-interpolant partA partB)")
        lines.append("(exit)")
        return "\n".join(lines)


# ===================================================================
# Contract discovery loop integration helpers
# ===================================================================

@dataclass
class CEGARIteration:
    """Record of a single contract discovery iteration."""
    iteration: int
    path: CounterexamplePath
    is_spurious: bool
    interpolant: Optional[Interpolant]
    new_predicates: List[Expr]
    total_predicates: int
    time_ms: float


class CEGARRefinementLoop:
    """Manage the contract discovery refinement loop state."""

    def __init__(self) -> None:
        self._engine = InterpolationEngine()
        self._predicates: List[Expr] = []
        self._iterations: List[CEGARIteration] = []
        self._program_vars: List[str] = []
        self._explainer = ProofExplainer()

    @property
    def predicates(self) -> List[Expr]:
        return list(self._predicates)

    @property
    def iterations(self) -> List[CEGARIteration]:
        return list(self._iterations)

    @property
    def statistics(self) -> InterpolationStatistics:
        return self._engine.statistics

    def set_program_vars(self, vars: List[str]) -> None:
        self._program_vars = list(vars)

    def add_initial_predicates(self, preds: List[Expr]) -> None:
        for p in preds:
            if str(p) not in {str(e) for e in self._predicates}:
                self._predicates.append(p)

    def refine(self, path: CounterexamplePath) -> CEGARIteration:
        """Perform one contract discovery refinement step."""
        start = time.time()
        iteration_num = len(self._iterations) + 1

        spur_result, new_preds = self._engine.check_and_refine(
            path, self._predicates, self._program_vars
        )

        # Update predicate set
        for p in new_preds:
            if str(p) not in {str(e) for e in self._predicates}:
                self._predicates.append(p)

        elapsed = (time.time() - start) * 1000.0

        # Find best interpolant for this iteration
        best_interp: Optional[Interpolant] = None
        if spur_result.is_spurious:
            partitions = self._engine._encoder.encode_with_partitions(path)
            for a, b in partitions:
                interp = self._engine.interpolate(a, b)
                if interp is not None:
                    if best_interp is None or interp.size < best_interp.size:
                        best_interp = interp

        iteration = CEGARIteration(
            iteration=iteration_num,
            path=path,
            is_spurious=spur_result.is_spurious,
            interpolant=best_interp,
            new_predicates=new_preds,
            total_predicates=len(self._predicates),
            time_ms=elapsed,
        )
        self._iterations.append(iteration)
        return iteration

    def explain_last_iteration(self) -> str:
        """Explain the last contract discovery iteration."""
        if not self._iterations:
            return "No iterations yet."

        last = self._iterations[-1]
        spur_result = SpuriousnessResult(
            is_spurious=last.is_spurious,
            explanation="See interpolant" if last.is_spurious else "Real counterexample",
        )
        return self._explainer.explain_spurious(
            last.path, spur_result, last.interpolant
        )

    def summary(self) -> str:
        parts: List[str] = []
        parts.append("Contract Discovery Refinement Summary")
        parts.append("=" * 40)
        parts.append(f"Total iterations: {len(self._iterations)}")
        spurious_count = sum(1 for it in self._iterations if it.is_spurious)
        real_count = len(self._iterations) - spurious_count
        parts.append(f"Spurious CEs: {spurious_count}")
        parts.append(f"Real CEs: {real_count}")
        parts.append(f"Current predicates: {len(self._predicates)}")
        if self._predicates:
            parts.append("Predicates:")
            for p in self._predicates:
                parts.append(f"  • {p}")
        parts.append("")
        parts.append(self._engine.statistics.summary())
        return "\n".join(parts)


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Core types
    "Expr", "Var", "IntLit", "BoolLit", "StrLit", "NoneLit",
    "UnaryExpr", "BinaryExpr", "QuantifiedExpr", "FuncApp", "IteExpr",
    "UnaryOp", "BinaryOp", "Quantifier",
    # Statements
    "Stmt", "AssignStmt", "AssumeStmt", "AssertStmt", "CallStmt",
    "SkipStmt", "SeqStmt", "RaiseStmt",
    # Interpolant
    "Interpolant", "Theory",
    # Interpolators
    "InterpolationEngine", "LinearInterpolator", "TypeTagInterpolator",
    "NullityInterpolator", "StructuralInterpolator", "CompoundInterpolator",
    # Linear constraints
    "LinearConstraint", "FarkasLemma",
    # Counterexample path
    "CounterexamplePath", "PathStep", "PathLocation", "PathState",
    "PathEncoder",
    # Spuriousness
    "SpuriousnessChecker", "SpuriousnessResult",
    # Predicate refinement
    "PredicateProjection", "PredicateTemplate", "PredicateRefinement",
    "PredicateQuality",
    # Cache and statistics
    "InterpolationCache", "InterpolationStatistics",
    # Explanation
    "ProofExplainer",
    # Contract discovery integration
    "CEGARIteration", "CEGARRefinementLoop",
]
