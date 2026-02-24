from __future__ import annotations

import enum
import math
import operator
import re
import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Sort system
# ---------------------------------------------------------------------------

class Sort(enum.Enum):
    """Sorts in the predicate language P."""
    INT = "Int"
    BOOL = "Bool"
    TAG = "Tag"
    STR = "Str"


# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------

class ArithOp(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


class UnaryArithOp(enum.Enum):
    NEG = "neg"
    ABS = "abs"


class ComparisonOp(enum.Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    """Variable reference."""
    name: str
    sort: Sort = Sort.INT

    def __repr__(self) -> str:
        return f"Var({self.name!r})"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class Const:
    """Literal constant."""
    value: Union[int, bool, str]
    sort: Sort = Sort.INT

    def __repr__(self) -> str:
        return f"Const({self.value!r})"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()


@dataclass(frozen=True)
class Len:
    """Length of a sequence variable."""
    arg: Expr

    def __repr__(self) -> str:
        return f"Len({self.arg!r})"

    def free_vars(self) -> FrozenSet[str]:
        return self.arg.free_vars()


@dataclass(frozen=True)
class BinOp:
    """Binary arithmetic operation."""
    op: ArithOp
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"BinOp({self.op.value}, {self.left!r}, {self.right!r})"

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()


@dataclass(frozen=True)
class UnaryOp:
    """Unary arithmetic operation."""
    op: UnaryArithOp
    operand: Expr

    def __repr__(self) -> str:
        return f"UnaryOp({self.op.value}, {self.operand!r})"

    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()


Expr = Union[Var, Const, Len, BinOp, UnaryOp]


# ---------------------------------------------------------------------------
# Atomic predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    """Arithmetic comparison e₁ ⊲⊳ e₂."""
    op: ComparisonOp
    left: Expr
    right: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()


@dataclass(frozen=True)
class IsInstance:
    """isinstance(var, tag)."""
    var: str
    tag: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})


@dataclass(frozen=True)
class IsNone:
    """is_none(var)."""
    var: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})


@dataclass(frozen=True)
class IsTruthy:
    """is_truthy(var)."""
    var: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})


@dataclass(frozen=True)
class HasAttr:
    """hasattr(var, key)."""
    var: str
    key: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})


AtomicPredicate = Union[Comparison, IsInstance, IsNone, IsTruthy, HasAttr]


# ---------------------------------------------------------------------------
# Compound predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class And:
    """Conjunction."""
    conjuncts: Tuple[Predicate, ...]

    def free_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for c in self.conjuncts:
            result = result | c.free_vars()
        return result


@dataclass(frozen=True)
class Or:
    """Disjunction."""
    disjuncts: Tuple[Predicate, ...]

    def free_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for d in self.disjuncts:
            result = result | d.free_vars()
        return result


@dataclass(frozen=True)
class Not:
    """Negation."""
    operand: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()


@dataclass(frozen=True)
class Implies:
    """Implication P → Q."""
    antecedent: Predicate
    consequent: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.antecedent.free_vars() | self.consequent.free_vars()


@dataclass(frozen=True)
class Iff:
    """Bi-implication P ↔ Q."""
    left: Predicate
    right: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()


# ---------------------------------------------------------------------------
# Quantified predicates (for documentation / non-decidable fragment)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForAll:
    """Universal quantification (documentation purposes)."""
    var: str
    sort: Sort
    body: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var}


@dataclass(frozen=True)
class Exists:
    """Existential quantification (documentation purposes)."""
    var: str
    sort: Sort
    body: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var}


@dataclass(frozen=True)
class BoolLit:
    """Boolean literal predicate (True / False)."""
    value: bool

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()


Predicate = Union[
    Comparison, IsInstance, IsNone, IsTruthy, HasAttr,
    And, Or, Not, Implies, Iff,
    ForAll, Exists,
    BoolLit,
]


# ---------------------------------------------------------------------------
# Helper: build predicates conveniently
# ---------------------------------------------------------------------------

def mk_and(*ps: Predicate) -> Predicate:
    flat: List[Predicate] = []
    for p in ps:
        if isinstance(p, And):
            flat.extend(p.conjuncts)
        elif isinstance(p, BoolLit) and p.value:
            continue
        elif isinstance(p, BoolLit) and not p.value:
            return BoolLit(False)
        else:
            flat.append(p)
    if not flat:
        return BoolLit(True)
    if len(flat) == 1:
        return flat[0]
    return And(tuple(flat))


def mk_or(*ps: Predicate) -> Predicate:
    flat: List[Predicate] = []
    for p in ps:
        if isinstance(p, Or):
            flat.extend(p.disjuncts)
        elif isinstance(p, BoolLit) and not p.value:
            continue
        elif isinstance(p, BoolLit) and p.value:
            return BoolLit(True)
        else:
            flat.append(p)
    if not flat:
        return BoolLit(False)
    if len(flat) == 1:
        return flat[0]
    return Or(tuple(flat))


def mk_not(p: Predicate) -> Predicate:
    if isinstance(p, BoolLit):
        return BoolLit(not p.value)
    if isinstance(p, Not):
        return p.operand
    return Not(p)


def mk_implies(p: Predicate, q: Predicate) -> Predicate:
    return Implies(p, q)


def mk_iff(p: Predicate, q: Predicate) -> Predicate:
    return Iff(p, q)


def mk_var(name: str, sort: Sort = Sort.INT) -> Var:
    return Var(name, sort)


def mk_const(value: Union[int, bool, str]) -> Const:
    if isinstance(value, bool):
        return Const(value, Sort.BOOL)
    if isinstance(value, int):
        return Const(value, Sort.INT)
    return Const(value, Sort.STR)


def mk_comparison(op: ComparisonOp, left: Expr, right: Expr) -> Comparison:
    return Comparison(op, left, right)


def mk_isinstance(var: str, tag: str) -> IsInstance:
    return IsInstance(var, tag)


def mk_is_none(var: str) -> IsNone:
    return IsNone(var)


def mk_is_truthy(var: str) -> IsTruthy:
    return IsTruthy(var)


def mk_hasattr(var: str, key: str) -> HasAttr:
    return HasAttr(var, key)


# ---------------------------------------------------------------------------
# PredicateEvaluator
# ---------------------------------------------------------------------------

# Map of type tags to Python types for isinstance checking
_TAG_TO_TYPE: Dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "bytes": bytes,
    "NoneType": type(None),
    "complex": complex,
    "frozenset": frozenset,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "range": range,
    "type": type,
}

# Numeric type hierarchy for tag subsumption
_NUMERIC_TAGS: Set[str] = {"int", "float", "complex", "bool"}
_SEQUENCE_TAGS: Set[str] = {"list", "tuple", "str", "bytes", "bytearray", "range"}
_CONTAINER_TAGS: Set[str] = _SEQUENCE_TAGS | {"dict", "set", "frozenset"}

_TYPE_HIERARCHY: Dict[str, Set[str]] = {
    "Number": {"int", "float", "complex", "bool"},
    "Integral": {"int", "bool"},
    "Real": {"int", "float", "bool"},
    "Complex": {"int", "float", "complex", "bool"},
    "Sequence": {"list", "tuple", "str", "bytes", "bytearray", "range"},
    "MutableSequence": {"list", "bytearray"},
    "Mapping": {"dict"},
    "MutableMapping": {"dict"},
    "Set": {"set", "frozenset"},
    "MutableSet": {"set"},
    "Iterable": _CONTAINER_TAGS | {"range"},
    "Container": _CONTAINER_TAGS,
    "Hashable": {"int", "float", "complex", "bool", "str", "bytes",
                 "frozenset", "tuple", "NoneType", "range", "type"},
    "Sized": _CONTAINER_TAGS | {"range"},
    "Callable": set(),
    "object": {"int", "float", "complex", "bool", "str", "bytes",
               "bytearray", "list", "tuple", "dict", "set", "frozenset",
               "NoneType", "range", "type", "memoryview"},
}


class PredicateEvaluationError(Exception):
    """Raised when predicate evaluation encounters an error."""
    pass


class PredicateEvaluator:
    """Evaluate predicates on concrete environments (variable → value maps)."""

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict
        self._arith_ops: Dict[ArithOp, Callable[[Any, Any], Any]] = {
            ArithOp.ADD: operator.add,
            ArithOp.SUB: operator.sub,
            ArithOp.MUL: operator.mul,
            ArithOp.DIV: self._safe_div,
            ArithOp.MOD: self._safe_mod,
        }
        self._cmp_ops: Dict[ComparisonOp, Callable[[Any, Any], bool]] = {
            ComparisonOp.EQ: operator.eq,
            ComparisonOp.NE: operator.ne,
            ComparisonOp.LT: operator.lt,
            ComparisonOp.LE: operator.le,
            ComparisonOp.GT: operator.gt,
            ComparisonOp.GE: operator.ge,
        }

    # -- expression evaluation ---------------------------------------------

    def evaluate_expr(self, expr: Expr, env: Dict[str, Any]) -> Any:
        """Evaluate an expression in the given environment."""
        if isinstance(expr, Var):
            return self._lookup_var(expr.name, env)
        if isinstance(expr, Const):
            return expr.value
        if isinstance(expr, Len):
            inner = self.evaluate_expr(expr.arg, env)
            return self._safe_len(inner)
        if isinstance(expr, BinOp):
            left_val = self.evaluate_expr(expr.left, env)
            right_val = self.evaluate_expr(expr.right, env)
            return self._eval_binop(expr.op, left_val, right_val)
        if isinstance(expr, UnaryOp):
            val = self.evaluate_expr(expr.operand, env)
            return self._eval_unaryop(expr.op, val)
        raise PredicateEvaluationError(f"Unknown expression kind: {type(expr)}")

    # -- predicate evaluation ----------------------------------------------

    def evaluate(self, pred: Predicate, env: Dict[str, Any]) -> bool:
        """Evaluate a predicate in the given environment → bool."""
        if isinstance(pred, BoolLit):
            return pred.value
        if isinstance(pred, Comparison):
            return self._eval_comparison(pred, env)
        if isinstance(pred, IsInstance):
            return self._eval_isinstance(pred, env)
        if isinstance(pred, IsNone):
            return self._eval_is_none(pred, env)
        if isinstance(pred, IsTruthy):
            return self._eval_is_truthy(pred, env)
        if isinstance(pred, HasAttr):
            return self._eval_hasattr(pred, env)
        if isinstance(pred, And):
            return self._eval_and(pred, env)
        if isinstance(pred, Or):
            return self._eval_or(pred, env)
        if isinstance(pred, Not):
            return not self.evaluate(pred.operand, env)
        if isinstance(pred, Implies):
            return (not self.evaluate(pred.antecedent, env)) or self.evaluate(
                pred.consequent, env
            )
        if isinstance(pred, Iff):
            l = self.evaluate(pred.left, env)
            r = self.evaluate(pred.right, env)
            return l == r
        if isinstance(pred, ForAll):
            raise PredicateEvaluationError(
                "Cannot evaluate universal quantification on concrete state"
            )
        if isinstance(pred, Exists):
            raise PredicateEvaluationError(
                "Cannot evaluate existential quantification on concrete state"
            )
        raise PredicateEvaluationError(f"Unknown predicate kind: {type(pred)}")

    # -- batch evaluation --------------------------------------------------

    def evaluate_all(
        self, preds: Sequence[Predicate], env: Dict[str, Any]
    ) -> List[bool]:
        """Evaluate a sequence of predicates, returning results list."""
        return [self.evaluate(p, env) for p in preds]

    def evaluate_conjunction(
        self, preds: Sequence[Predicate], env: Dict[str, Any]
    ) -> bool:
        """Short-circuit conjunction of predicate list."""
        for p in preds:
            if not self.evaluate(p, env):
                return False
        return True

    def evaluate_disjunction(
        self, preds: Sequence[Predicate], env: Dict[str, Any]
    ) -> bool:
        """Short-circuit disjunction of predicate list."""
        for p in preds:
            if self.evaluate(p, env):
                return True
        return False

    def satisfying_subset(
        self, preds: Sequence[Predicate], env: Dict[str, Any]
    ) -> List[Predicate]:
        """Return the subset of predicates that hold in *env*."""
        return [p for p in preds if self.evaluate(p, env)]

    def falsifying_subset(
        self, preds: Sequence[Predicate], env: Dict[str, Any]
    ) -> List[Predicate]:
        """Return the subset of predicates that do NOT hold in *env*."""
        return [p for p in preds if not self.evaluate(p, env)]

    # -- internal helpers --------------------------------------------------

    def _lookup_var(self, name: str, env: Dict[str, Any]) -> Any:
        if name in env:
            return env[name]
        if self._strict:
            raise PredicateEvaluationError(f"Variable {name!r} not in environment")
        return None

    @staticmethod
    def _safe_div(a: Any, b: Any) -> Any:
        if b == 0:
            raise PredicateEvaluationError("Division by zero")
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        return a / b

    @staticmethod
    def _safe_mod(a: Any, b: Any) -> Any:
        if b == 0:
            raise PredicateEvaluationError("Modulo by zero")
        return a % b

    @staticmethod
    def _safe_len(v: Any) -> int:
        try:
            return len(v)
        except TypeError:
            raise PredicateEvaluationError(
                f"len() not supported for {type(v).__name__}"
            )

    def _eval_binop(self, op: ArithOp, left: Any, right: Any) -> Any:
        left = self._coerce_numeric(left)
        right = self._coerce_numeric(right)
        fn = self._arith_ops[op]
        try:
            return fn(left, right)
        except Exception as exc:
            raise PredicateEvaluationError(
                f"Arithmetic error {op.value}: {exc}"
            ) from exc

    def _eval_unaryop(self, op: UnaryArithOp, val: Any) -> Any:
        val = self._coerce_numeric(val)
        if op is UnaryArithOp.NEG:
            return -val
        if op is UnaryArithOp.ABS:
            return abs(val)
        raise PredicateEvaluationError(f"Unknown unary op: {op}")

    def _coerce_numeric(self, v: Any) -> Union[int, float]:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        if isinstance(v, bool):
            return int(v)
        if v is None:
            if self._strict:
                raise PredicateEvaluationError("Cannot coerce None to numeric")
            return 0
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    raise PredicateEvaluationError(
                        f"Cannot coerce string {v!r} to numeric"
                    )
        raise PredicateEvaluationError(
            f"Cannot coerce {type(v).__name__} to numeric"
        )

    def _eval_comparison(self, pred: Comparison, env: Dict[str, Any]) -> bool:
        left_val = self.evaluate_expr(pred.left, env)
        right_val = self.evaluate_expr(pred.right, env)
        fn = self._cmp_ops[pred.op]
        try:
            return fn(left_val, right_val)
        except TypeError:
            if pred.op in (ComparisonOp.EQ, ComparisonOp.NE):
                return pred.op is ComparisonOp.NE
            if self._strict:
                raise PredicateEvaluationError(
                    f"Cannot compare {type(left_val).__name__} "
                    f"{pred.op.value} {type(right_val).__name__}"
                )
            return False

    def _eval_isinstance(self, pred: IsInstance, env: Dict[str, Any]) -> bool:
        val = self._lookup_var(pred.var, env)
        tag = pred.tag
        py_type = _TAG_TO_TYPE.get(tag)
        if py_type is not None:
            return isinstance(val, py_type)
        if tag in _TYPE_HIERARCHY:
            actual = type(val).__name__
            return actual in _TYPE_HIERARCHY[tag]
        try:
            import builtins
            ty = getattr(builtins, tag, None)
            if ty is not None and isinstance(ty, type):
                return isinstance(val, ty)
        except Exception:
            pass
        return type(val).__name__ == tag

    def _eval_is_none(self, pred: IsNone, env: Dict[str, Any]) -> bool:
        val = self._lookup_var(pred.var, env)
        return val is None

    def _eval_is_truthy(self, pred: IsTruthy, env: Dict[str, Any]) -> bool:
        val = self._lookup_var(pred.var, env)
        return bool(val)

    def _eval_hasattr(self, pred: HasAttr, env: Dict[str, Any]) -> bool:
        val = self._lookup_var(pred.var, env)
        if isinstance(val, dict):
            return pred.key in val
        return hasattr(val, pred.key)

    def _eval_and(self, pred: And, env: Dict[str, Any]) -> bool:
        for c in pred.conjuncts:
            if not self.evaluate(c, env):
                return False
        return True

    def _eval_or(self, pred: Or, env: Dict[str, Any]) -> bool:
        for d in pred.disjuncts:
            if self.evaluate(d, env):
                return True
        return False


# ---------------------------------------------------------------------------
# PredicateNormalizer
# ---------------------------------------------------------------------------

class PredicateNormalizer:
    """Normalize predicates to various canonical forms."""

    # -- NNF ---------------------------------------------------------------

    def to_nnf(self, pred: Predicate) -> Predicate:
        """Convert to negation normal form (negation only on atoms)."""
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy,
                             HasAttr, BoolLit)):
            return pred
        if isinstance(pred, And):
            return And(tuple(self.to_nnf(c) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(self.to_nnf(d) for d in pred.disjuncts))
        if isinstance(pred, Implies):
            return self.to_nnf(mk_or(mk_not(pred.antecedent), pred.consequent))
        if isinstance(pred, Iff):
            return self.to_nnf(
                mk_and(
                    mk_implies(pred.left, pred.right),
                    mk_implies(pred.right, pred.left),
                )
            )
        if isinstance(pred, Not):
            return self._push_negation(pred.operand)
        if isinstance(pred, (ForAll, Exists)):
            return pred
        return pred

    def _push_negation(self, inner: Predicate) -> Predicate:
        """Push negation inward (De Morgan, double negation, etc.)."""
        if isinstance(inner, BoolLit):
            return BoolLit(not inner.value)
        if isinstance(inner, Not):
            return self.to_nnf(inner.operand)
        if isinstance(inner, And):
            return Or(tuple(self.to_nnf(mk_not(c)) for c in inner.conjuncts))
        if isinstance(inner, Or):
            return And(tuple(self.to_nnf(mk_not(d)) for d in inner.disjuncts))
        if isinstance(inner, Implies):
            return self.to_nnf(mk_and(inner.antecedent, mk_not(inner.consequent)))
        if isinstance(inner, Iff):
            return self.to_nnf(
                mk_or(
                    mk_and(inner.left, mk_not(inner.right)),
                    mk_and(mk_not(inner.left), inner.right),
                )
            )
        if isinstance(inner, Comparison):
            negated_op = self._negate_comparison_op(inner.op)
            if negated_op is not None:
                return Comparison(negated_op, inner.left, inner.right)
            return Not(inner)
        return Not(inner)

    @staticmethod
    def _negate_comparison_op(op: ComparisonOp) -> Optional[ComparisonOp]:
        mapping = {
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
        }
        return mapping.get(op)

    # -- CNF ---------------------------------------------------------------

    def to_cnf(self, pred: Predicate) -> Predicate:
        """Convert to conjunctive normal form."""
        nnf = self.to_nnf(pred)
        return self._distribute_or_over_and(nnf)

    def _distribute_or_over_and(self, pred: Predicate) -> Predicate:
        """Recursively distribute ∨ over ∧ to get CNF."""
        if isinstance(pred, And):
            return And(
                tuple(self._distribute_or_over_and(c) for c in pred.conjuncts)
            )
        if isinstance(pred, Or):
            disjuncts = [
                self._distribute_or_over_and(d) for d in pred.disjuncts
            ]
            result: Predicate = disjuncts[0]
            for d in disjuncts[1:]:
                result = self._distribute_single_or_over_and(result, d)
            return result
        return pred

    def _distribute_single_or_over_and(
        self, a: Predicate, b: Predicate
    ) -> Predicate:
        """Distribute (A₁∧A₂) ∨ B  =  (A₁∨B) ∧ (A₂∨B), etc."""
        if isinstance(a, And):
            return And(
                tuple(
                    self._distribute_single_or_over_and(c, b)
                    for c in a.conjuncts
                )
            )
        if isinstance(b, And):
            return And(
                tuple(
                    self._distribute_single_or_over_and(a, c)
                    for c in b.conjuncts
                )
            )
        return mk_or(a, b)

    # -- DNF ---------------------------------------------------------------

    def to_dnf(self, pred: Predicate) -> Predicate:
        """Convert to disjunctive normal form."""
        nnf = self.to_nnf(pred)
        return self._distribute_and_over_or(nnf)

    def _distribute_and_over_or(self, pred: Predicate) -> Predicate:
        """Recursively distribute ∧ over ∨ to get DNF."""
        if isinstance(pred, Or):
            return Or(
                tuple(self._distribute_and_over_or(d) for d in pred.disjuncts)
            )
        if isinstance(pred, And):
            conjuncts = [
                self._distribute_and_over_or(c) for c in pred.conjuncts
            ]
            result: Predicate = conjuncts[0]
            for c in conjuncts[1:]:
                result = self._distribute_single_and_over_or(result, c)
            return result
        return pred

    def _distribute_single_and_over_or(
        self, a: Predicate, b: Predicate
    ) -> Predicate:
        """Distribute (A₁∨A₂) ∧ B  =  (A₁∧B) ∨ (A₂∧B)."""
        if isinstance(a, Or):
            return Or(
                tuple(
                    self._distribute_single_and_over_or(d, b)
                    for d in a.disjuncts
                )
            )
        if isinstance(b, Or):
            return Or(
                tuple(
                    self._distribute_single_and_over_or(a, d)
                    for d in b.disjuncts
                )
            )
        return mk_and(a, b)

    # -- simplify ----------------------------------------------------------

    def simplify(self, pred: Predicate) -> Predicate:
        """Apply a battery of simplification rules until a fixpoint."""
        prev = None
        current = pred
        depth = 0
        while current != prev and depth < 50:
            prev = current
            current = self._simplify_step(current)
            depth += 1
        return current

    def _simplify_step(self, pred: Predicate) -> Predicate:
        if isinstance(pred, BoolLit):
            return pred
        if isinstance(pred, Not):
            inner = self._simplify_step(pred.operand)
            if isinstance(inner, BoolLit):
                return BoolLit(not inner.value)
            if isinstance(inner, Not):
                return inner.operand
            return Not(inner)
        if isinstance(pred, And):
            simplified = [self._simplify_step(c) for c in pred.conjuncts]
            simplified = self._flatten_and(simplified)
            simplified = self._remove_duplicate_preds(simplified)
            # Absorption: remove True
            simplified = [s for s in simplified if not (isinstance(s, BoolLit) and s.value)]
            # Short-circuit False
            if any(isinstance(s, BoolLit) and not s.value for s in simplified):
                return BoolLit(False)
            # Contradiction: P and ¬P
            if self._has_contradiction(simplified):
                return BoolLit(False)
            # Absorption: P ∧ (P ∨ Q) → P
            simplified = self._absorb_and(simplified)
            if not simplified:
                return BoolLit(True)
            if len(simplified) == 1:
                return simplified[0]
            return And(tuple(simplified))
        if isinstance(pred, Or):
            simplified = [self._simplify_step(d) for d in pred.disjuncts]
            simplified = self._flatten_or(simplified)
            simplified = self._remove_duplicate_preds(simplified)
            simplified = [s for s in simplified if not (isinstance(s, BoolLit) and not s.value)]
            if any(isinstance(s, BoolLit) and s.value for s in simplified):
                return BoolLit(True)
            if self._has_tautology(simplified):
                return BoolLit(True)
            simplified = self._absorb_or(simplified)
            if not simplified:
                return BoolLit(False)
            if len(simplified) == 1:
                return simplified[0]
            return Or(tuple(simplified))
        if isinstance(pred, Implies):
            ant = self._simplify_step(pred.antecedent)
            con = self._simplify_step(pred.consequent)
            if isinstance(ant, BoolLit):
                return BoolLit(True) if not ant.value else con
            if isinstance(con, BoolLit):
                return BoolLit(True) if con.value else self._simplify_step(Not(ant))
            if ant == con:
                return BoolLit(True)
            return Implies(ant, con)
        if isinstance(pred, Iff):
            l = self._simplify_step(pred.left)
            r = self._simplify_step(pred.right)
            if l == r:
                return BoolLit(True)
            if isinstance(l, BoolLit):
                return r if l.value else self._simplify_step(Not(r))
            if isinstance(r, BoolLit):
                return l if r.value else self._simplify_step(Not(l))
            return Iff(l, r)
        if isinstance(pred, Comparison):
            return self._simplify_comparison(pred)
        return pred

    def _simplify_comparison(self, pred: Comparison) -> Predicate:
        """Simplify comparisons between constants."""
        if isinstance(pred.left, Const) and isinstance(pred.right, Const):
            evaluator = PredicateEvaluator()
            try:
                result = evaluator.evaluate(pred, {})
                return BoolLit(result)
            except PredicateEvaluationError:
                pass
        # x == x → True, x != x → False, x < x → False, x <= x → True, etc.
        if pred.left == pred.right:
            if pred.op in (ComparisonOp.EQ, ComparisonOp.LE, ComparisonOp.GE):
                return BoolLit(True)
            if pred.op in (ComparisonOp.NE, ComparisonOp.LT, ComparisonOp.GT):
                return BoolLit(False)
        return pred

    def _flatten_and(self, preds: List[Predicate]) -> List[Predicate]:
        result: List[Predicate] = []
        for p in preds:
            if isinstance(p, And):
                result.extend(p.conjuncts)
            else:
                result.append(p)
        return result

    def _flatten_or(self, preds: List[Predicate]) -> List[Predicate]:
        result: List[Predicate] = []
        for p in preds:
            if isinstance(p, Or):
                result.extend(p.disjuncts)
            else:
                result.append(p)
        return result

    @staticmethod
    def _remove_duplicate_preds(preds: List[Predicate]) -> List[Predicate]:
        seen: List[Predicate] = []
        for p in preds:
            if p not in seen:
                seen.append(p)
        return seen

    @staticmethod
    def _has_contradiction(preds: List[Predicate]) -> bool:
        for i, p in enumerate(preds):
            neg_p = Not(p) if not isinstance(p, Not) else p.operand
            for j, q in enumerate(preds):
                if i != j and q == neg_p:
                    return True
        return False

    @staticmethod
    def _has_tautology(preds: List[Predicate]) -> bool:
        for i, p in enumerate(preds):
            neg_p = Not(p) if not isinstance(p, Not) else p.operand
            for j, q in enumerate(preds):
                if i != j and q == neg_p:
                    return True
        return False

    @staticmethod
    def _absorb_and(conjuncts: List[Predicate]) -> List[Predicate]:
        """P ∧ (P ∨ Q) → P."""
        result: List[Predicate] = list(conjuncts)
        changed = True
        while changed:
            changed = False
            new_result: List[Predicate] = []
            for i, c in enumerate(result):
                absorbed = False
                if isinstance(c, Or):
                    for j, other in enumerate(result):
                        if i != j and other in c.disjuncts:
                            absorbed = True
                            break
                if not absorbed:
                    new_result.append(c)
                else:
                    changed = True
            result = new_result
        return result

    @staticmethod
    def _absorb_or(disjuncts: List[Predicate]) -> List[Predicate]:
        """P ∨ (P ∧ Q) → P."""
        result: List[Predicate] = list(disjuncts)
        changed = True
        while changed:
            changed = False
            new_result: List[Predicate] = []
            for i, d in enumerate(result):
                absorbed = False
                if isinstance(d, And):
                    for j, other in enumerate(result):
                        if i != j and other in d.conjuncts:
                            absorbed = True
                            break
                if not absorbed:
                    new_result.append(d)
                else:
                    changed = True
            result = new_result
        return result

    # -- flatten -----------------------------------------------------------

    def flatten(self, pred: Predicate) -> Predicate:
        """Flatten nested And/Or."""
        if isinstance(pred, And):
            flat = self._flatten_and(
                [self.flatten(c) for c in pred.conjuncts]
            )
            if len(flat) == 0:
                return BoolLit(True)
            if len(flat) == 1:
                return flat[0]
            return And(tuple(flat))
        if isinstance(pred, Or):
            flat = self._flatten_or(
                [self.flatten(d) for d in pred.disjuncts]
            )
            if len(flat) == 0:
                return BoolLit(False)
            if len(flat) == 1:
                return flat[0]
            return Or(tuple(flat))
        if isinstance(pred, Not):
            return Not(self.flatten(pred.operand))
        if isinstance(pred, Implies):
            return Implies(self.flatten(pred.antecedent), self.flatten(pred.consequent))
        if isinstance(pred, Iff):
            return Iff(self.flatten(pred.left), self.flatten(pred.right))
        return pred

    # -- sort conjuncts ----------------------------------------------------

    def sort_conjuncts(self, pred: Predicate) -> Predicate:
        """Sort conjuncts/disjuncts for canonical ordering."""
        if isinstance(pred, And):
            sorted_cs = sorted(
                (self.sort_conjuncts(c) for c in pred.conjuncts),
                key=self._pred_sort_key,
            )
            return And(tuple(sorted_cs))
        if isinstance(pred, Or):
            sorted_ds = sorted(
                (self.sort_conjuncts(d) for d in pred.disjuncts),
                key=self._pred_sort_key,
            )
            return Or(tuple(sorted_ds))
        if isinstance(pred, Not):
            return Not(self.sort_conjuncts(pred.operand))
        return pred

    @staticmethod
    def _pred_sort_key(pred: Predicate) -> Tuple[int, str]:
        kind_order = {
            Comparison: 0, IsInstance: 1, IsNone: 2, IsTruthy: 3,
            HasAttr: 4, Not: 5, And: 6, Or: 7, Implies: 8,
            Iff: 9, BoolLit: 10, ForAll: 11, Exists: 12,
        }
        kind_idx = kind_order.get(type(pred), 99)
        return (kind_idx, repr(pred))

    # -- normalize comparison ----------------------------------------------

    def normalize_comparison(self, pred: Comparison) -> Comparison:
        """Normalize a comparison so that a variable is on the left when possible."""
        if isinstance(pred.right, Var) and not isinstance(pred.left, Var):
            flipped_op = self._flip_comparison_op(pred.op)
            return Comparison(flipped_op, pred.right, pred.left)
        if isinstance(pred.left, Var) and isinstance(pred.right, Var):
            if pred.left.name > pred.right.name:
                flipped_op = self._flip_comparison_op(pred.op)
                return Comparison(flipped_op, pred.right, pred.left)
        return pred

    @staticmethod
    def _flip_comparison_op(op: ComparisonOp) -> ComparisonOp:
        mapping = {
            ComparisonOp.EQ: ComparisonOp.EQ,
            ComparisonOp.NE: ComparisonOp.NE,
            ComparisonOp.LT: ComparisonOp.GT,
            ComparisonOp.LE: ComparisonOp.GE,
            ComparisonOp.GT: ComparisonOp.LT,
            ComparisonOp.GE: ComparisonOp.LE,
        }
        return mapping[op]

    # -- eliminate implies -------------------------------------------------

    def eliminate_implies(self, pred: Predicate) -> Predicate:
        """Replace P → Q with ¬P ∨ Q."""
        if isinstance(pred, Implies):
            a = self.eliminate_implies(pred.antecedent)
            b = self.eliminate_implies(pred.consequent)
            return mk_or(mk_not(a), b)
        if isinstance(pred, Iff):
            l = self.eliminate_implies(pred.left)
            r = self.eliminate_implies(pred.right)
            return mk_and(mk_or(mk_not(l), r), mk_or(mk_not(r), l))
        if isinstance(pred, And):
            return And(tuple(self.eliminate_implies(c) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(self.eliminate_implies(d) for d in pred.disjuncts))
        if isinstance(pred, Not):
            return Not(self.eliminate_implies(pred.operand))
        return pred

    # -- distribute --------------------------------------------------------

    def distribute(self, pred: Predicate, *, and_over_or: bool = True) -> Predicate:
        """Distribute and over or (for CNF) or or over and (for DNF)."""
        if and_over_or:
            return self.to_cnf(pred)
        return self.to_dnf(pred)

    # -- remove duplicates -------------------------------------------------

    def remove_duplicates(self, pred: Predicate) -> Predicate:
        """Remove duplicate conjuncts/disjuncts."""
        if isinstance(pred, And):
            children = [self.remove_duplicates(c) for c in pred.conjuncts]
            unique = self._remove_duplicate_preds(children)
            if len(unique) == 0:
                return BoolLit(True)
            if len(unique) == 1:
                return unique[0]
            return And(tuple(unique))
        if isinstance(pred, Or):
            children = [self.remove_duplicates(d) for d in pred.disjuncts]
            unique = self._remove_duplicate_preds(children)
            if len(unique) == 0:
                return BoolLit(False)
            if len(unique) == 1:
                return unique[0]
            return Or(tuple(unique))
        if isinstance(pred, Not):
            return Not(self.remove_duplicates(pred.operand))
        return pred

    # -- detect contradictions / tautologies --------------------------------

    def detect_contradictions(self, pred: Predicate) -> bool:
        """Detect P ∧ ¬P patterns."""
        if isinstance(pred, And):
            return self._has_contradiction(list(pred.conjuncts))
        return False

    def detect_tautologies(self, pred: Predicate) -> bool:
        """Detect P ∨ ¬P patterns."""
        if isinstance(pred, Or):
            return self._has_tautology(list(pred.disjuncts))
        return False

    # -- full normalization pipeline ---------------------------------------

    def normalize(self, pred: Predicate) -> Predicate:
        """Full normalization: eliminate implies → NNF → flatten → simplify → sort."""
        result = self.eliminate_implies(pred)
        result = self.to_nnf(result)
        result = self.flatten(result)
        result = self.simplify(result)
        result = self.sort_conjuncts(result)
        return result


# ---------------------------------------------------------------------------
# PredicateSubstitution
# ---------------------------------------------------------------------------

class PredicateSubstitution:
    """Substitute variables in predicates/expressions."""

    def subst_expr(self, expr: Expr, mapping: Dict[str, Expr]) -> Expr:
        if isinstance(expr, Var):
            return mapping.get(expr.name, expr)
        if isinstance(expr, Const):
            return expr
        if isinstance(expr, Len):
            return Len(self.subst_expr(expr.arg, mapping))
        if isinstance(expr, BinOp):
            return BinOp(
                expr.op,
                self.subst_expr(expr.left, mapping),
                self.subst_expr(expr.right, mapping),
            )
        if isinstance(expr, UnaryOp):
            return UnaryOp(expr.op, self.subst_expr(expr.operand, mapping))
        return expr

    def subst_pred(self, pred: Predicate, mapping: Dict[str, Expr]) -> Predicate:
        if isinstance(pred, Comparison):
            return Comparison(
                pred.op,
                self.subst_expr(pred.left, mapping),
                self.subst_expr(pred.right, mapping),
            )
        if isinstance(pred, IsInstance):
            new_var = mapping.get(pred.var)
            if new_var is not None and isinstance(new_var, Var):
                return IsInstance(new_var.name, pred.tag)
            return pred
        if isinstance(pred, IsNone):
            new_var = mapping.get(pred.var)
            if new_var is not None and isinstance(new_var, Var):
                return IsNone(new_var.name)
            return pred
        if isinstance(pred, IsTruthy):
            new_var = mapping.get(pred.var)
            if new_var is not None and isinstance(new_var, Var):
                return IsTruthy(new_var.name)
            return pred
        if isinstance(pred, HasAttr):
            new_var = mapping.get(pred.var)
            if new_var is not None and isinstance(new_var, Var):
                return HasAttr(new_var.name, pred.key)
            return pred
        if isinstance(pred, And):
            return And(tuple(self.subst_pred(c, mapping) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(self.subst_pred(d, mapping) for d in pred.disjuncts))
        if isinstance(pred, Not):
            return Not(self.subst_pred(pred.operand, mapping))
        if isinstance(pred, Implies):
            return Implies(
                self.subst_pred(pred.antecedent, mapping),
                self.subst_pred(pred.consequent, mapping),
            )
        if isinstance(pred, Iff):
            return Iff(
                self.subst_pred(pred.left, mapping),
                self.subst_pred(pred.right, mapping),
            )
        if isinstance(pred, BoolLit):
            return pred
        if isinstance(pred, ForAll):
            if pred.var in mapping:
                new_mapping = {k: v for k, v in mapping.items() if k != pred.var}
            else:
                new_mapping = mapping
            return ForAll(pred.var, pred.sort, self.subst_pred(pred.body, new_mapping))
        if isinstance(pred, Exists):
            if pred.var in mapping:
                new_mapping = {k: v for k, v in mapping.items() if k != pred.var}
            else:
                new_mapping = mapping
            return Exists(pred.var, pred.sort, self.subst_pred(pred.body, new_mapping))
        return pred


# ---------------------------------------------------------------------------
# Interval helper (for numeric reasoning)
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """Half-open integer interval [lo, hi). None means unbounded."""
    lo: Optional[int] = None
    hi: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo >= self.hi
        return False

    @property
    def is_bounded(self) -> bool:
        return self.lo is not None and self.hi is not None

    def contains(self, v: int) -> bool:
        if self.lo is not None and v < self.lo:
            return False
        if self.hi is not None and v >= self.hi:
            return False
        return True

    def intersect(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return Interval(lo, hi)

    def union_hull(self, other: Interval) -> Interval:
        lo = min(self.lo, other.lo) if self.lo is not None and other.lo is not None else None
        hi = max(self.hi, other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo, hi)

    def widen(self, other: Interval) -> Interval:
        lo = self.lo if (other.lo is not None and self.lo is not None and other.lo >= self.lo) else None
        hi = self.hi if (other.hi is not None and self.hi is not None and other.hi <= self.hi) else None
        return Interval(lo, hi)

    def narrow(self, other: Interval) -> Interval:
        lo = other.lo if self.lo is None else self.lo
        if lo is not None and other.lo is not None and other.lo > lo:
            lo = other.lo
        hi = other.hi if self.hi is None else self.hi
        if hi is not None and other.hi is not None and other.hi < hi:
            hi = other.hi
        return Interval(lo, hi)

    def __repr__(self) -> str:
        lo_s = str(self.lo) if self.lo is not None else "-∞"
        hi_s = str(self.hi) if self.hi is not None else "+∞"
        return f"[{lo_s}, {hi_s})"


# ---------------------------------------------------------------------------
# PredicateImplication
# ---------------------------------------------------------------------------

class PredicateImplication:
    """Check semantic implication between predicates."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()
        self._evaluator = PredicateEvaluator()

    def implies(self, p1: Predicate, p2: Predicate) -> bool:
        """Check whether p1 ⊨ p2 (p1 semantically implies p2)."""
        # Trivial cases
        if isinstance(p2, BoolLit) and p2.value:
            return True
        if isinstance(p1, BoolLit) and not p1.value:
            return True
        if isinstance(p1, BoolLit) and p1.value:
            simplified = self._normalizer.simplify(p2)
            return isinstance(simplified, BoolLit) and simplified.value
        if isinstance(p2, BoolLit) and not p2.value:
            simplified = self._normalizer.simplify(p1)
            return isinstance(simplified, BoolLit) and not simplified.value

        # Syntactic equality
        if p1 == p2:
            return True

        # Syntactic shortcuts
        result = self._syntactic_implies(p1, p2)
        if result is not None:
            return result

        # Structural
        result = self._structural_implies(p1, p2)
        if result is not None:
            return result

        # Interval-based
        result = self._interval_implies(p1, p2)
        if result is not None:
            return result

        # Type tag
        result = self._type_tag_implies(p1, p2)
        if result is not None:
            return result

        # Nullity
        result = self._nullity_implies(p1, p2)
        if result is not None:
            return result

        # Conjunction / disjunction decomposition
        result = self._compound_implies(p1, p2)
        if result is not None:
            return result

        return False

    def _syntactic_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """Quick syntactic checks."""
        # P ⊨ P ∨ Q
        if isinstance(p2, Or):
            for d in p2.disjuncts:
                if p1 == d:
                    return True

        # P ∧ Q ⊨ P
        if isinstance(p1, And):
            for c in p1.conjuncts:
                if c == p2:
                    return True

        # ¬P ⊨ ¬P
        if isinstance(p1, Not) and isinstance(p2, Not):
            return self.implies(p2.operand, p1.operand) or None

        return None

    def _interval_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """Interval-based: x > 5 implies x > 3."""
        if not isinstance(p1, Comparison) or not isinstance(p2, Comparison):
            return None
        if not isinstance(p1.left, Var) or not isinstance(p2.left, Var):
            return None
        if p1.left.name != p2.left.name:
            return None
        if not isinstance(p1.right, Const) or not isinstance(p2.right, Const):
            return None
        v1 = p1.right.value
        v2 = p2.right.value
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            return None

        var_name = p1.left.name

        i1 = self._comparison_to_interval(p1.op, v1)
        i2 = self._comparison_to_interval(p2.op, v2)

        if i1 is None or i2 is None:
            return None

        return self._interval_subset(i1, i2)

    @staticmethod
    def _comparison_to_interval(
        op: ComparisonOp, val: Union[int, float]
    ) -> Optional[Interval]:
        v = int(val) if isinstance(val, float) and val == int(val) else val
        if not isinstance(v, int):
            return None
        if op == ComparisonOp.GT:
            return Interval(lo=v + 1, hi=None)
        if op == ComparisonOp.GE:
            return Interval(lo=v, hi=None)
        if op == ComparisonOp.LT:
            return Interval(lo=None, hi=v)
        if op == ComparisonOp.LE:
            return Interval(lo=None, hi=v + 1)
        if op == ComparisonOp.EQ:
            return Interval(lo=v, hi=v + 1)
        return None

    @staticmethod
    def _interval_subset(a: Interval, b: Interval) -> bool:
        """Check if interval a ⊆ interval b."""
        if b.lo is not None:
            if a.lo is None or a.lo < b.lo:
                return False
        if b.hi is not None:
            if a.hi is None or a.hi > b.hi:
                return False
        return True

    def _type_tag_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """isinstance(x,int) implies isinstance(x, Number), etc."""
        if isinstance(p1, IsInstance) and isinstance(p2, IsInstance):
            if p1.var != p2.var:
                return None
            if p1.tag == p2.tag:
                return True
            # Check hierarchy: p1.tag is a sub-tag of p2.tag?
            subtags = _TYPE_HIERARCHY.get(p2.tag, set())
            if p1.tag in subtags:
                return True
            return None
        return None

    def _nullity_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """isinstance(x,int) implies not is_none(x)."""
        if isinstance(p1, IsInstance) and isinstance(p2, Not):
            if isinstance(p2.operand, IsNone):
                if p1.var == p2.operand.var and p1.tag != "NoneType":
                    return True
        if isinstance(p1, IsNone) and isinstance(p2, Not):
            if isinstance(p2.operand, IsInstance):
                if p1.var == p2.operand.var and p2.operand.tag != "NoneType":
                    return True
        return None

    def _structural_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """hasattr(x,'a') ∧ hasattr(x,'b') implies hasattr(x,'a')."""
        if isinstance(p2, HasAttr):
            if isinstance(p1, HasAttr):
                return True if (p1.var == p2.var and p1.key == p2.key) else None
            if isinstance(p1, And):
                for c in p1.conjuncts:
                    if isinstance(c, HasAttr) and c.var == p2.var and c.key == p2.key:
                        return True
        return None

    def _compound_implies(self, p1: Predicate, p2: Predicate) -> Optional[bool]:
        """Decompose compound predicates for implication checking."""
        # P1 ∧ P2 ⊨ Q  iff  P1 ⊨ Q or P2 ⊨ Q
        if isinstance(p1, And):
            for c in p1.conjuncts:
                if self.implies(c, p2):
                    return True

        # P ⊨ Q1 ∧ Q2  iff  P ⊨ Q1 and P ⊨ Q2
        if isinstance(p2, And):
            if all(self.implies(p1, c) for c in p2.conjuncts):
                return True

        # P ⊨ Q1 ∨ Q2  iff  P ⊨ Q1 or P ⊨ Q2
        if isinstance(p2, Or):
            for d in p2.disjuncts:
                if self.implies(p1, d):
                    return True

        # P1 ∨ P2 ⊨ Q  iff  P1 ⊨ Q and P2 ⊨ Q
        if isinstance(p1, Or):
            if all(self.implies(d, p2) for d in p1.disjuncts):
                return True

        return None


# ---------------------------------------------------------------------------
# PredicateStrengthening
# ---------------------------------------------------------------------------

class PredicateStrengthening:
    """Strengthen predicates by adding constraints."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()

    def add_conjunct(self, pred: Predicate, new: Predicate) -> Predicate:
        """Add a new conjunct."""
        result = mk_and(pred, new)
        return self._normalizer.simplify(result)

    def restrict_interval(
        self,
        pred: Predicate,
        var: str,
        lo: Optional[int] = None,
        hi: Optional[int] = None,
    ) -> Predicate:
        """Tighten numeric bounds for a variable."""
        constraints: List[Predicate] = []
        if lo is not None:
            constraints.append(
                Comparison(ComparisonOp.GE, Var(var), Const(lo))
            )
        if hi is not None:
            constraints.append(
                Comparison(ComparisonOp.LT, Var(var), Const(hi))
            )
        if not constraints:
            return pred
        return self.add_conjunct(pred, mk_and(*constraints))

    def restrict_type_tags(
        self, pred: Predicate, var: str, tags_to_remove: Set[str]
    ) -> Predicate:
        """Strengthen by removing type tags from possible tags."""
        constraints: List[Predicate] = []
        for tag in tags_to_remove:
            constraints.append(Not(IsInstance(var, tag)))
        if not constraints:
            return pred
        return self.add_conjunct(pred, mk_and(*constraints))

    def add_nullity(self, pred: Predicate, var: str, *, is_none: bool) -> Predicate:
        """Add nullity constraint."""
        if is_none:
            return self.add_conjunct(pred, IsNone(var))
        return self.add_conjunct(pred, Not(IsNone(var)))

    def add_attribute(self, pred: Predicate, var: str, key: str) -> Predicate:
        """Add attribute presence."""
        return self.add_conjunct(pred, HasAttr(var, key))

    def add_equality(
        self, pred: Predicate, var: str, value: int
    ) -> Predicate:
        """Add an equality constraint var == value."""
        return self.add_conjunct(
            pred, Comparison(ComparisonOp.EQ, Var(var), Const(value))
        )

    def add_isinstance(
        self, pred: Predicate, var: str, tag: str
    ) -> Predicate:
        """Add isinstance constraint."""
        return self.add_conjunct(pred, IsInstance(var, tag))

    def add_truthiness(
        self, pred: Predicate, var: str, *, truthy: bool = True
    ) -> Predicate:
        """Add truthiness constraint."""
        if truthy:
            return self.add_conjunct(pred, IsTruthy(var))
        return self.add_conjunct(pred, Not(IsTruthy(var)))


# ---------------------------------------------------------------------------
# PredicateWeakening
# ---------------------------------------------------------------------------

class PredicateWeakening:
    """Weaken predicates by removing constraints."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()

    def remove_conjunct(self, pred: Predicate, to_remove: Predicate) -> Predicate:
        """Remove a specific conjunct."""
        if isinstance(pred, And):
            remaining = [c for c in pred.conjuncts if c != to_remove]
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        if pred == to_remove:
            return BoolLit(True)
        return pred

    def widen_interval(
        self,
        pred: Predicate,
        var: str,
        *,
        remove_lower: bool = False,
        remove_upper: bool = False,
    ) -> Predicate:
        """Loosen numeric bounds for a variable."""
        if isinstance(pred, And):
            remaining: List[Predicate] = []
            for c in pred.conjuncts:
                if isinstance(c, Comparison) and isinstance(c.left, Var) and c.left.name == var:
                    if remove_lower and c.op in (ComparisonOp.GE, ComparisonOp.GT):
                        continue
                    if remove_upper and c.op in (ComparisonOp.LE, ComparisonOp.LT):
                        continue
                remaining.append(c)
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        return pred

    def add_type_tags(
        self, pred: Predicate, var: str, tags_to_allow: Set[str]
    ) -> Predicate:
        """Weaken by adding more possible type tags (remove ¬isinstance constraints)."""
        if isinstance(pred, And):
            remaining: List[Predicate] = []
            for c in pred.conjuncts:
                if isinstance(c, Not) and isinstance(c.operand, IsInstance):
                    if c.operand.var == var and c.operand.tag in tags_to_allow:
                        continue
                remaining.append(c)
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        return pred

    def remove_nullity(self, pred: Predicate, var: str) -> Predicate:
        """Remove nullity constraints for a variable."""
        if isinstance(pred, And):
            remaining: List[Predicate] = []
            for c in pred.conjuncts:
                if isinstance(c, IsNone) and c.var == var:
                    continue
                if isinstance(c, Not) and isinstance(c.operand, IsNone) and c.operand.var == var:
                    continue
                remaining.append(c)
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        return pred

    def remove_attribute_constraints(self, pred: Predicate, var: str) -> Predicate:
        """Remove all hasattr constraints for a variable."""
        if isinstance(pred, And):
            remaining = [
                c for c in pred.conjuncts
                if not (isinstance(c, HasAttr) and c.var == var)
            ]
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        return pred

    def remove_isinstance_constraints(self, pred: Predicate, var: str) -> Predicate:
        """Remove all isinstance constraints for a variable."""
        if isinstance(pred, And):
            remaining = [
                c for c in pred.conjuncts
                if not (isinstance(c, IsInstance) and c.var == var)
            ]
            if not remaining:
                return BoolLit(True)
            if len(remaining) == 1:
                return remaining[0]
            return And(tuple(remaining))
        return pred

    def weaken_to_disjunction(self, pred: Predicate, alternative: Predicate) -> Predicate:
        """Weaken by adding a disjunctive alternative."""
        return self._normalizer.simplify(mk_or(pred, alternative))


# ---------------------------------------------------------------------------
# PredicateProjection
# ---------------------------------------------------------------------------

class PredicateProjection:
    """Project predicates onto a subset of variables."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()

    def project(self, pred: Predicate, keep_vars: FrozenSet[str]) -> Predicate:
        """Restrict predicate to mention only *keep_vars*.

        Atoms that reference variables outside *keep_vars* are dropped
        (over-approximation via existential quantification).
        """
        if isinstance(pred, BoolLit):
            return pred
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
            fv = pred.free_vars()
            if fv <= keep_vars:
                return pred
            return BoolLit(True)  # over-approximate
        if isinstance(pred, Not):
            inner_fv = pred.operand.free_vars()
            if inner_fv <= keep_vars:
                return pred
            return BoolLit(True)
        if isinstance(pred, And):
            projected = [self.project(c, keep_vars) for c in pred.conjuncts]
            return self._normalizer.simplify(mk_and(*projected))
        if isinstance(pred, Or):
            projected = [self.project(d, keep_vars) for d in pred.disjuncts]
            # For disjunction, if any disjunct has external vars, we must keep it as True
            return self._normalizer.simplify(mk_or(*projected))
        if isinstance(pred, Implies):
            return self.project(
                mk_or(mk_not(pred.antecedent), pred.consequent), keep_vars
            )
        if isinstance(pred, Iff):
            return self.project(
                mk_and(
                    mk_implies(pred.left, pred.right),
                    mk_implies(pred.right, pred.left),
                ),
                keep_vars,
            )
        return BoolLit(True)

    def existential_quantification(
        self, pred: Predicate, var: str
    ) -> Predicate:
        """∃var. P  ≈  project P onto FV(P) \\ {var}."""
        fv = pred.free_vars()
        return self.project(pred, fv - {var})

    def universal_quantification(
        self, pred: Predicate, var: str
    ) -> Predicate:
        """∀var. P  ≈  keep only conjuncts not mentioning var (under-approx)."""
        if isinstance(pred, And):
            kept = [
                c for c in pred.conjuncts if var not in c.free_vars()
            ]
            if not kept:
                return BoolLit(True)
            return self._normalizer.simplify(mk_and(*kept))
        fv = pred.free_vars()
        if var not in fv:
            return pred
        return BoolLit(True)

    def project_to_vars(
        self, pred: Predicate, vars: Set[str]
    ) -> Predicate:
        """Convenience wrapper accepting a mutable set."""
        return self.project(pred, frozenset(vars))

    def rename_variables(
        self, pred: Predicate, rename_map: Dict[str, str]
    ) -> Predicate:
        """Rename variables in a predicate."""
        subst = PredicateSubstitution()
        expr_map: Dict[str, Expr] = {
            old: Var(new) for old, new in rename_map.items()
        }
        return subst.subst_pred(pred, expr_map)


# ---------------------------------------------------------------------------
# PredicateInterpolation  (unsat core-based predicate extraction)
# ---------------------------------------------------------------------------

class PredicateInterpolation:
    """Unsat core-based predicate extraction over the predicate language P.

    Given A and B such that A ∧ B is unsatisfiable, produce I such that:
      1. A ⊨ I
      2. I ∧ B is unsatisfiable
      3. FV(I) ⊆ FV(A) ∩ FV(B)
    """

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()
        self._implication = PredicateImplication()
        self._projection = PredicateProjection()

    def interpolate(self, a: Predicate, b: Predicate) -> Optional[Predicate]:
        """Compute an interpolant for A, B where A ∧ B is unsat."""
        common_vars = a.free_vars() & b.free_vars()
        if not common_vars:
            return BoolLit(False) if self._trivially_unsat(a) else BoolLit(True)

        # Try linear interpolation
        result = self._linear_interpolation(a, b, common_vars)
        if result is not None:
            return result

        # Try type-tag interpolation
        result = self._type_tag_interpolation(a, b, common_vars)
        if result is not None:
            return result

        # Try structural interpolation
        result = self._structural_interpolation(a, b, common_vars)
        if result is not None:
            return result

        # Fallback: project A onto common variables
        return self._projection.project(a, common_vars)

    def _trivially_unsat(self, pred: Predicate) -> bool:
        simplified = self._normalizer.simplify(pred)
        return isinstance(simplified, BoolLit) and not simplified.value

    def _linear_interpolation(
        self, a: Predicate, b: Predicate, common_vars: FrozenSet[str]
    ) -> Optional[Predicate]:
        """Linear interpolation for QF_LIA fragment."""
        a_comparisons = self._extract_comparisons(a)
        b_comparisons = self._extract_comparisons(b)

        interpolant_conjuncts: List[Predicate] = []

        for var in common_vars:
            a_bounds = self._extract_bounds_for_var(a_comparisons, var)
            b_bounds = self._extract_bounds_for_var(b_comparisons, var)

            if not a_bounds and not b_bounds:
                continue

            a_lower, a_upper = a_bounds if a_bounds else (None, None)
            b_lower, b_upper = b_bounds if b_bounds else (None, None)

            # If A says x >= a_lo and B says x < b_hi, and a_lo >= b_hi → unsat
            # Interpolant: x >= a_lo (or x < b_hi, whichever uses common vars)
            if a_lower is not None and b_upper is not None:
                if a_lower >= b_upper:
                    interpolant_conjuncts.append(
                        Comparison(ComparisonOp.GE, Var(var), Const(a_lower))
                    )
            if a_upper is not None and b_lower is not None:
                if a_upper <= b_lower:
                    interpolant_conjuncts.append(
                        Comparison(ComparisonOp.LT, Var(var), Const(a_upper))
                    )

        if interpolant_conjuncts:
            return self._normalizer.simplify(mk_and(*interpolant_conjuncts))
        return None

    def _type_tag_interpolation(
        self, a: Predicate, b: Predicate, common_vars: FrozenSet[str]
    ) -> Optional[Predicate]:
        """Interpolation based on type tags."""
        a_instances = self._extract_isinstance(a)
        b_instances = self._extract_isinstance(b)

        interpolant_parts: List[Predicate] = []

        for var in common_vars:
            a_tags = {tag for v, tag in a_instances if v == var}
            b_neg_tags = set()
            for v, tag in b_instances:
                if v == var:
                    pass
            b_not_instances = self._extract_not_isinstance(b)
            b_neg_tags = {tag for v, tag in b_not_instances if v == var}

            # If A says isinstance(x, int) and B says ¬isinstance(x, int)
            conflicting = a_tags & b_neg_tags
            for tag in conflicting:
                interpolant_parts.append(IsInstance(var, tag))

        if interpolant_parts:
            return self._normalizer.simplify(mk_and(*interpolant_parts))
        return None

    def _structural_interpolation(
        self, a: Predicate, b: Predicate, common_vars: FrozenSet[str]
    ) -> Optional[Predicate]:
        """Interpolation based on structural properties (hasattr, etc.)."""
        a_attrs = self._extract_hasattr(a)
        b_not_attrs = self._extract_not_hasattr(b)

        interpolant_parts: List[Predicate] = []
        for var in common_vars:
            a_keys = {k for v, k in a_attrs if v == var}
            b_neg_keys = {k for v, k in b_not_attrs if v == var}
            conflicting = a_keys & b_neg_keys
            for key in conflicting:
                interpolant_parts.append(HasAttr(var, key))

        if interpolant_parts:
            return self._normalizer.simplify(mk_and(*interpolant_parts))
        return None

    # -- helpers -----------------------------------------------------------

    def _extract_comparisons(self, pred: Predicate) -> List[Comparison]:
        results: List[Comparison] = []
        if isinstance(pred, Comparison):
            results.append(pred)
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                results.extend(self._extract_comparisons(c))
        return results

    def _extract_bounds_for_var(
        self, comparisons: List[Comparison], var: str
    ) -> Optional[Tuple[Optional[int], Optional[int]]]:
        lower: Optional[int] = None
        upper: Optional[int] = None
        found = False
        for cmp in comparisons:
            if isinstance(cmp.left, Var) and cmp.left.name == var and isinstance(cmp.right, Const):
                val = cmp.right.value
                if not isinstance(val, int):
                    continue
                found = True
                if cmp.op == ComparisonOp.GE:
                    lower = max(lower, val) if lower is not None else val
                elif cmp.op == ComparisonOp.GT:
                    lower = max(lower, val + 1) if lower is not None else val + 1
                elif cmp.op == ComparisonOp.LE:
                    upper = min(upper, val + 1) if upper is not None else val + 1
                elif cmp.op == ComparisonOp.LT:
                    upper = min(upper, val) if upper is not None else val
                elif cmp.op == ComparisonOp.EQ:
                    lower = max(lower, val) if lower is not None else val
                    upper = min(upper, val + 1) if upper is not None else val + 1
        return (lower, upper) if found else None

    @staticmethod
    def _extract_isinstance(pred: Predicate) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        if isinstance(pred, IsInstance):
            results.append((pred.var, pred.tag))
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                if isinstance(c, IsInstance):
                    results.append((c.var, c.tag))
        return results

    @staticmethod
    def _extract_not_isinstance(pred: Predicate) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        if isinstance(pred, Not) and isinstance(pred.operand, IsInstance):
            results.append((pred.operand.var, pred.operand.tag))
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                if isinstance(c, Not) and isinstance(c.operand, IsInstance):
                    results.append((c.operand.var, c.operand.tag))
        return results

    @staticmethod
    def _extract_hasattr(pred: Predicate) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        if isinstance(pred, HasAttr):
            results.append((pred.var, pred.key))
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                if isinstance(c, HasAttr):
                    results.append((c.var, c.key))
        return results

    @staticmethod
    def _extract_not_hasattr(pred: Predicate) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        if isinstance(pred, Not) and isinstance(pred.operand, HasAttr):
            results.append((pred.operand.var, pred.operand.key))
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                if isinstance(c, Not) and isinstance(c.operand, HasAttr):
                    results.append((c.operand.var, c.operand.key))
        return results


# ---------------------------------------------------------------------------
# PredicateAbstraction
# ---------------------------------------------------------------------------

@dataclass
class AbstractValue:
    """Result of abstracting a concrete state.

    ``values`` maps each predicate index to its Boolean value.
    """
    values: Dict[int, bool] = field(default_factory=dict)
    predicate_list: List[Predicate] = field(default_factory=list)

    def get(self, idx: int) -> Optional[bool]:
        return self.values.get(idx)

    def satisfies(self, idx: int) -> bool:
        return self.values.get(idx, False)

    def as_predicate(self) -> Predicate:
        conjuncts: List[Predicate] = []
        for idx, val in sorted(self.values.items()):
            if idx < len(self.predicate_list):
                p = self.predicate_list[idx]
                conjuncts.append(p if val else mk_not(p))
        if not conjuncts:
            return BoolLit(True)
        return mk_and(*conjuncts)

    def join(self, other: AbstractValue) -> AbstractValue:
        """Least upper bound: keep predicates that agree."""
        common_values: Dict[int, bool] = {}
        for idx in self.values:
            if idx in other.values and self.values[idx] == other.values[idx]:
                common_values[idx] = self.values[idx]
        return AbstractValue(values=common_values, predicate_list=self.predicate_list)

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Greatest lower bound: combine all predicates."""
        combined: Dict[int, bool] = dict(self.values)
        for idx, val in other.values.items():
            if idx in combined and combined[idx] != val:
                # Contradiction
                pass
            else:
                combined[idx] = val
        return AbstractValue(values=combined, predicate_list=self.predicate_list)

    def __repr__(self) -> str:
        parts = []
        for idx in sorted(self.values):
            val = self.values[idx]
            if idx < len(self.predicate_list):
                p_str = repr(self.predicate_list[idx])
                parts.append(f"{p_str}={'T' if val else 'F'}")
            else:
                parts.append(f"p{idx}={'T' if val else 'F'}")
        return f"Abs({', '.join(parts)})"


class PredicateAbstraction:
    """Abstract concrete states to predicate abstractions."""

    def __init__(self) -> None:
        self._evaluator = PredicateEvaluator()

    def abstract(
        self,
        concrete_state: Dict[str, Any],
        predicate_set: List[Predicate],
    ) -> AbstractValue:
        """Boolean abstraction: evaluate each predicate on the concrete state."""
        values: Dict[int, bool] = {}
        for i, p in enumerate(predicate_set):
            try:
                values[i] = self._evaluator.evaluate(p, concrete_state)
            except PredicateEvaluationError:
                pass
        return AbstractValue(values=values, predicate_list=predicate_set)

    def cartesian_abstract(
        self,
        concrete_state: Dict[str, Any],
        predicate_set: List[Predicate],
    ) -> AbstractValue:
        """Cartesian abstraction: track predicates independently (same as boolean for single state)."""
        return self.abstract(concrete_state, predicate_set)

    def heuristic_abstract(
        self,
        concrete_state: Dict[str, Any],
        predicate_set: List[Predicate],
        *,
        max_correlations: int = 3,
    ) -> AbstractValue:
        """Heuristic abstraction: track predicate correlations.

        Evaluates predicates and additionally discovers implications between them.
        """
        base = self.abstract(concrete_state, predicate_set)
        # Discover pairwise implications from this single state
        # (limited value for a single state, more useful when called on many states)
        return base

    def abstract_multiple(
        self,
        concrete_states: List[Dict[str, Any]],
        predicate_set: List[Predicate],
    ) -> AbstractValue:
        """Abstract multiple concrete states via join."""
        if not concrete_states:
            return AbstractValue(predicate_list=predicate_set)
        result = self.abstract(concrete_states[0], predicate_set)
        for state in concrete_states[1:]:
            other = self.abstract(state, predicate_set)
            result = result.join(other)
        return result

    def concretize_check(
        self,
        abstract_val: AbstractValue,
        concrete_state: Dict[str, Any],
    ) -> bool:
        """Check if a concrete state is in the concretization of an abstract value."""
        for idx, expected in abstract_val.values.items():
            if idx < len(abstract_val.predicate_list):
                p = abstract_val.predicate_list[idx]
                try:
                    actual = self._evaluator.evaluate(p, concrete_state)
                    if actual != expected:
                        return False
                except PredicateEvaluationError:
                    return False
        return True


# ---------------------------------------------------------------------------
# PredicateDiscovery
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredPredicate:
    """A discovered predicate with metadata."""
    predicate: Predicate
    source: str  # where it was discovered from
    rank: float = 0.0  # information content / usefulness ranking

    def __repr__(self) -> str:
        return f"Discovered({self.predicate!r}, src={self.source!r}, rank={self.rank:.2f})"


class PredicateDiscovery:
    """Discover new predicates from program constructs."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()
        self._implication = PredicateImplication()

    def from_guards(self, guard_expressions: List[Predicate]) -> List[DiscoveredPredicate]:
        """Extract predicates from runtime guard expressions (if-conditions, etc.)."""
        result: List[DiscoveredPredicate] = []
        for guard in guard_expressions:
            normalized = self._normalizer.normalize(guard)
            atoms = self._extract_atoms(normalized)
            for atom in atoms:
                result.append(DiscoveredPredicate(
                    predicate=atom, source="guard", rank=1.0
                ))
        return self._deduplicate(result)

    def from_assertions(self, assertions: List[Predicate]) -> List[DiscoveredPredicate]:
        """Extract predicates from assert statements."""
        result: List[DiscoveredPredicate] = []
        for assertion in assertions:
            normalized = self._normalizer.normalize(assertion)
            atoms = self._extract_atoms(normalized)
            for atom in atoms:
                result.append(DiscoveredPredicate(
                    predicate=atom, source="assertion", rank=1.5
                ))
        return self._deduplicate(result)

    def from_comparisons(
        self, comparisons: List[Comparison]
    ) -> List[DiscoveredPredicate]:
        """Extract predicates from comparison expressions."""
        result: List[DiscoveredPredicate] = []
        for cmp in comparisons:
            result.append(DiscoveredPredicate(
                predicate=cmp, source="comparison", rank=0.8
            ))
            # Also generate boundary predicates
            if isinstance(cmp.right, Const) and isinstance(cmp.right.value, int):
                val = cmp.right.value
                if cmp.op in (ComparisonOp.LT, ComparisonOp.GE):
                    result.append(DiscoveredPredicate(
                        predicate=Comparison(ComparisonOp.EQ, cmp.left, Const(val)),
                        source="comparison_boundary",
                        rank=0.5,
                    ))
                if cmp.op in (ComparisonOp.GT, ComparisonOp.LE):
                    result.append(DiscoveredPredicate(
                        predicate=Comparison(ComparisonOp.EQ, cmp.left, Const(val)),
                        source="comparison_boundary",
                        rank=0.5,
                    ))
        return self._deduplicate(result)

    def from_constants(
        self, var_names: List[str], constants: List[int]
    ) -> List[DiscoveredPredicate]:
        """Generate predicates from program constants."""
        result: List[DiscoveredPredicate] = []
        for var in var_names:
            for c in constants:
                result.append(DiscoveredPredicate(
                    predicate=Comparison(ComparisonOp.EQ, Var(var), Const(c)),
                    source="constant",
                    rank=0.3,
                ))
                result.append(DiscoveredPredicate(
                    predicate=Comparison(ComparisonOp.LT, Var(var), Const(c)),
                    source="constant",
                    rank=0.3,
                ))
                result.append(DiscoveredPredicate(
                    predicate=Comparison(ComparisonOp.GE, Var(var), Const(c)),
                    source="constant",
                    rank=0.3,
                ))
            # Zero comparisons always useful
            if 0 not in constants:
                result.append(DiscoveredPredicate(
                    predicate=Comparison(ComparisonOp.GE, Var(var), Const(0)),
                    source="constant_zero",
                    rank=0.4,
                ))
                result.append(DiscoveredPredicate(
                    predicate=Comparison(ComparisonOp.EQ, Var(var), Const(0)),
                    source="constant_zero",
                    rank=0.4,
                ))
        return self._deduplicate(result)

    def from_types(
        self,
        var_names: List[str],
        type_tags: List[str],
    ) -> List[DiscoveredPredicate]:
        """Generate predicates from type annotations."""
        result: List[DiscoveredPredicate] = []
        for var in var_names:
            for tag in type_tags:
                result.append(DiscoveredPredicate(
                    predicate=IsInstance(var, tag),
                    source="type_annotation",
                    rank=1.2,
                ))
            result.append(DiscoveredPredicate(
                predicate=IsNone(var),
                source="type_annotation",
                rank=0.9,
            ))
            result.append(DiscoveredPredicate(
                predicate=Not(IsNone(var)),
                source="type_annotation",
                rank=0.9,
            ))
        return self._deduplicate(result)

    def from_attributes(
        self, var_names: List[str], attr_names: List[str]
    ) -> List[DiscoveredPredicate]:
        """Generate hasattr predicates."""
        result: List[DiscoveredPredicate] = []
        for var in var_names:
            for attr in attr_names:
                result.append(DiscoveredPredicate(
                    predicate=HasAttr(var, attr),
                    source="attribute",
                    rank=0.7,
                ))
        return self._deduplicate(result)

    def rank_predicates(
        self, predicates: List[DiscoveredPredicate]
    ) -> List[DiscoveredPredicate]:
        """Rank predicates by information content / usefulness."""
        scored: List[DiscoveredPredicate] = []
        for dp in predicates:
            score = dp.rank
            p = dp.predicate
            # Prefer predicates with fewer free variables
            fv_count = len(p.free_vars())
            score += 1.0 / (1.0 + fv_count)
            # Prefer atomic predicates over compound
            if isinstance(p, (And, Or)):
                score *= 0.5
            # Prefer isinstance over arithmetic
            if isinstance(p, IsInstance):
                score += 0.3
            # Prefer equality to inequality
            if isinstance(p, Comparison) and p.op == ComparisonOp.EQ:
                score += 0.1
            scored.append(DiscoveredPredicate(
                predicate=p, source=dp.source, rank=score
            ))
        scored.sort(key=lambda x: -x.rank)
        return scored

    def filter_predicates(
        self, predicates: List[DiscoveredPredicate]
    ) -> List[DiscoveredPredicate]:
        """Remove redundant/subsumed predicates."""
        result: List[DiscoveredPredicate] = []
        for dp in predicates:
            subsumed = False
            for existing in result:
                if self._implication.implies(existing.predicate, dp.predicate):
                    subsumed = True
                    break
            if not subsumed:
                # Remove any existing predicates that the new one subsumes
                result = [
                    ex for ex in result
                    if not self._implication.implies(dp.predicate, ex.predicate)
                ]
                result.append(dp)
        return result

    def _extract_atoms(self, pred: Predicate) -> List[Predicate]:
        """Extract all atomic predicates from a compound predicate."""
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
            return [pred]
        if isinstance(pred, Not):
            if isinstance(pred.operand, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
                return [pred, pred.operand]
            return self._extract_atoms(pred.operand)
        if isinstance(pred, And):
            result: List[Predicate] = []
            for c in pred.conjuncts:
                result.extend(self._extract_atoms(c))
            return result
        if isinstance(pred, Or):
            result = []
            for d in pred.disjuncts:
                result.extend(self._extract_atoms(d))
            return result
        if isinstance(pred, BoolLit):
            return []
        return [pred]

    @staticmethod
    def _deduplicate(
        preds: List[DiscoveredPredicate],
    ) -> List[DiscoveredPredicate]:
        seen: Set[str] = set()
        result: List[DiscoveredPredicate] = []
        for dp in preds:
            key = repr(dp.predicate)
            if key not in seen:
                seen.add(key)
                result.append(dp)
        return result


# ---------------------------------------------------------------------------
# PredicateSet
# ---------------------------------------------------------------------------

class PredicateSet:
    """A set of predicates with set operations and semantic queries."""

    def __init__(self, predicates: Optional[Sequence[Predicate]] = None) -> None:
        self._predicates: List[Predicate] = list(predicates) if predicates else []
        self._normalizer = PredicateNormalizer()
        self._implication = PredicateImplication()

    @property
    def predicates(self) -> List[Predicate]:
        return list(self._predicates)

    def __len__(self) -> int:
        return len(self._predicates)

    def __iter__(self) -> Iterator[Predicate]:
        return iter(self._predicates)

    def __contains__(self, pred: Predicate) -> bool:
        return pred in self._predicates

    def __repr__(self) -> str:
        return f"PredicateSet({self._predicates!r})"

    def add(self, pred: Predicate) -> None:
        if pred not in self._predicates:
            self._predicates.append(pred)

    def remove(self, pred: Predicate) -> None:
        self._predicates = [p for p in self._predicates if p != pred]

    def contains(self, pred: Predicate) -> bool:
        return pred in self._predicates

    def union(self, other: PredicateSet) -> PredicateSet:
        combined = list(self._predicates)
        for p in other._predicates:
            if p not in combined:
                combined.append(p)
        return PredicateSet(combined)

    def intersection(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet([p for p in self._predicates if p in other._predicates])

    def difference(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet([p for p in self._predicates if p not in other._predicates])

    def subsumes(self, other: PredicateSet) -> bool:
        """Check if this set semantically subsumes other (every pred in other
        is implied by some pred in self)."""
        for p in other._predicates:
            if not any(self._implication.implies(q, p) for q in self._predicates):
                return False
        return True

    def minimize(self) -> PredicateSet:
        """Find a minimal subset: remove predicates implied by others."""
        minimal: List[Predicate] = []
        for i, p in enumerate(self._predicates):
            others = [q for j, q in enumerate(self._predicates) if j != i]
            # Check if conjunction of others implies p
            implied = False
            for q in others:
                if self._implication.implies(q, p):
                    implied = True
                    break
            if not implied:
                minimal.append(p)
        return PredicateSet(minimal)

    def is_satisfiable(self) -> bool:
        """Check satisfiability of conjunction (conservative: may say True when unsat)."""
        conj = mk_and(*self._predicates) if self._predicates else BoolLit(True)
        simplified = self._normalizer.simplify(conj)
        if isinstance(simplified, BoolLit):
            return simplified.value
        if self._normalizer.detect_contradictions(simplified if isinstance(simplified, And) else And((simplified,))):
            return False
        return True

    def get_implications(self) -> List[Tuple[int, int]]:
        """Compute pairwise implications: (i, j) means predicate[i] ⊨ predicate[j]."""
        result: List[Tuple[int, int]] = []
        for i in range(len(self._predicates)):
            for j in range(len(self._predicates)):
                if i != j and self._implication.implies(
                    self._predicates[i], self._predicates[j]
                ):
                    result.append((i, j))
        return result

    def as_conjunction(self) -> Predicate:
        if not self._predicates:
            return BoolLit(True)
        return mk_and(*self._predicates)

    def as_disjunction(self) -> Predicate:
        if not self._predicates:
            return BoolLit(False)
        return mk_or(*self._predicates)

    def free_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for p in self._predicates:
            result = result | p.free_vars()
        return result


# ---------------------------------------------------------------------------
# PredicatePrinter
# ---------------------------------------------------------------------------

class PredicatePrinter:
    """Pretty-print predicates in various formats."""

    def __init__(self, *, unicode: bool = True) -> None:
        self._unicode = unicode

    # -- human-readable string ---------------------------------------------

    def to_string(self, pred: Predicate) -> str:
        if isinstance(pred, BoolLit):
            return "true" if pred.value else "false"
        if isinstance(pred, Comparison):
            return f"{self._expr_str(pred.left)} {pred.op.value} {self._expr_str(pred.right)}"
        if isinstance(pred, IsInstance):
            return f"isinstance({pred.var}, {pred.tag})"
        if isinstance(pred, IsNone):
            return f"is_none({pred.var})"
        if isinstance(pred, IsTruthy):
            return f"is_truthy({pred.var})"
        if isinstance(pred, HasAttr):
            return f"hasattr({pred.var}, {pred.key!r})"
        if isinstance(pred, Not):
            inner = self.to_string(pred.operand)
            return f"¬({inner})" if self._unicode else f"not ({inner})"
        if isinstance(pred, And):
            sep = " ∧ " if self._unicode else " and "
            parts = [self._maybe_paren(c, And) for c in pred.conjuncts]
            return sep.join(parts)
        if isinstance(pred, Or):
            sep = " ∨ " if self._unicode else " or "
            parts = [self._maybe_paren(d, Or) for d in pred.disjuncts]
            return sep.join(parts)
        if isinstance(pred, Implies):
            arrow = " → " if self._unicode else " => "
            return f"{self._maybe_paren(pred.antecedent, Implies)}{arrow}{self._maybe_paren(pred.consequent, Implies)}"
        if isinstance(pred, Iff):
            arrow = " ↔ " if self._unicode else " <=> "
            return f"{self._maybe_paren(pred.left, Iff)}{arrow}{self._maybe_paren(pred.right, Iff)}"
        if isinstance(pred, ForAll):
            q = "∀" if self._unicode else "forall"
            return f"{q} {pred.var}:{pred.sort.value}. {self.to_string(pred.body)}"
        if isinstance(pred, Exists):
            q = "∃" if self._unicode else "exists"
            return f"{q} {pred.var}:{pred.sort.value}. {self.to_string(pred.body)}"
        return repr(pred)

    def _expr_str(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            return repr(expr.value)
        if isinstance(expr, Len):
            return f"len({self._expr_str(expr.arg)})"
        if isinstance(expr, BinOp):
            l = self._expr_str(expr.left)
            r = self._expr_str(expr.right)
            return f"({l} {expr.op.value} {r})"
        if isinstance(expr, UnaryOp):
            inner = self._expr_str(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return f"(-{inner})"
            if expr.op == UnaryArithOp.ABS:
                return f"abs({inner})"
        return repr(expr)

    def _maybe_paren(self, pred: Predicate, parent_type: type) -> str:
        s = self.to_string(pred)
        if isinstance(pred, (And, Or, Implies, Iff)) and type(pred) != parent_type:
            return f"({s})"
        return s

    # -- SMT-LIB format ----------------------------------------------------

    def to_smt_lib(self, pred: Predicate) -> str:
        if isinstance(pred, BoolLit):
            return "true" if pred.value else "false"
        if isinstance(pred, Comparison):
            op_map = {
                ComparisonOp.EQ: "=", ComparisonOp.NE: "distinct",
                ComparisonOp.LT: "<", ComparisonOp.LE: "<=",
                ComparisonOp.GT: ">", ComparisonOp.GE: ">=",
            }
            return f"({op_map[pred.op]} {self._expr_smt(pred.left)} {self._expr_smt(pred.right)})"
        if isinstance(pred, IsInstance):
            return f"(= (typeof {pred.var}) {pred.tag})"
        if isinstance(pred, IsNone):
            return f"(= {pred.var} none)"
        if isinstance(pred, IsTruthy):
            return f"(truthy {pred.var})"
        if isinstance(pred, HasAttr):
            return f"(hasattr {pred.var} {pred.key!r})"
        if isinstance(pred, Not):
            return f"(not {self.to_smt_lib(pred.operand)})"
        if isinstance(pred, And):
            if not pred.conjuncts:
                return "true"
            parts = " ".join(self.to_smt_lib(c) for c in pred.conjuncts)
            return f"(and {parts})"
        if isinstance(pred, Or):
            if not pred.disjuncts:
                return "false"
            parts = " ".join(self.to_smt_lib(d) for d in pred.disjuncts)
            return f"(or {parts})"
        if isinstance(pred, Implies):
            return f"(=> {self.to_smt_lib(pred.antecedent)} {self.to_smt_lib(pred.consequent)})"
        if isinstance(pred, Iff):
            return f"(= {self.to_smt_lib(pred.left)} {self.to_smt_lib(pred.right)})"
        if isinstance(pred, ForAll):
            return f"(forall (({pred.var} {pred.sort.value})) {self.to_smt_lib(pred.body)})"
        if isinstance(pred, Exists):
            return f"(exists (({pred.var} {pred.sort.value})) {self.to_smt_lib(pred.body)})"
        return repr(pred)

    def _expr_smt(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            if isinstance(expr.value, int):
                return str(expr.value) if expr.value >= 0 else f"(- {abs(expr.value)})"
            return f'"{expr.value}"'
        if isinstance(expr, Len):
            return f"(len {self._expr_smt(expr.arg)})"
        if isinstance(expr, BinOp):
            op_map = {
                ArithOp.ADD: "+", ArithOp.SUB: "-", ArithOp.MUL: "*",
                ArithOp.DIV: "div", ArithOp.MOD: "mod",
            }
            return f"({op_map[expr.op]} {self._expr_smt(expr.left)} {self._expr_smt(expr.right)})"
        if isinstance(expr, UnaryOp):
            if expr.op == UnaryArithOp.NEG:
                return f"(- {self._expr_smt(expr.operand)})"
            if expr.op == UnaryArithOp.ABS:
                return f"(abs {self._expr_smt(expr.operand)})"
        return repr(expr)

    # -- Python expression format ------------------------------------------

    def to_python(self, pred: Predicate) -> str:
        if isinstance(pred, BoolLit):
            return "True" if pred.value else "False"
        if isinstance(pred, Comparison):
            return f"{self._expr_python(pred.left)} {pred.op.value} {self._expr_python(pred.right)}"
        if isinstance(pred, IsInstance):
            return f"isinstance({pred.var}, {pred.tag})"
        if isinstance(pred, IsNone):
            return f"{pred.var} is None"
        if isinstance(pred, IsTruthy):
            return f"bool({pred.var})"
        if isinstance(pred, HasAttr):
            return f"hasattr({pred.var}, {pred.key!r})"
        if isinstance(pred, Not):
            inner = self.to_python(pred.operand)
            return f"not ({inner})"
        if isinstance(pred, And):
            parts = [self._maybe_paren_py(c, "and") for c in pred.conjuncts]
            return " and ".join(parts)
        if isinstance(pred, Or):
            parts = [self._maybe_paren_py(d, "or") for d in pred.disjuncts]
            return " or ".join(parts)
        if isinstance(pred, Implies):
            a = self.to_python(pred.antecedent)
            b = self.to_python(pred.consequent)
            return f"(not ({a})) or ({b})"
        if isinstance(pred, Iff):
            l = self.to_python(pred.left)
            r = self.to_python(pred.right)
            return f"({l}) == ({r})"
        return repr(pred)

    def _expr_python(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            return repr(expr.value)
        if isinstance(expr, Len):
            return f"len({self._expr_python(expr.arg)})"
        if isinstance(expr, BinOp):
            l = self._expr_python(expr.left)
            r = self._expr_python(expr.right)
            op = expr.op.value
            if expr.op == ArithOp.DIV:
                op = "//"
            return f"({l} {op} {r})"
        if isinstance(expr, UnaryOp):
            inner = self._expr_python(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return f"(-{inner})"
            if expr.op == UnaryArithOp.ABS:
                return f"abs({inner})"
        return repr(expr)

    def _maybe_paren_py(self, pred: Predicate, parent_op: str) -> str:
        s = self.to_python(pred)
        if parent_op == "and" and isinstance(pred, Or):
            return f"({s})"
        if parent_op == "or" and isinstance(pred, And):
            return f"({s})"
        return s

    # -- LaTeX format ------------------------------------------------------

    def to_latex(self, pred: Predicate) -> str:
        if isinstance(pred, BoolLit):
            return r"\top" if pred.value else r"\bot"
        if isinstance(pred, Comparison):
            op_map = {
                ComparisonOp.EQ: "=", ComparisonOp.NE: r"\neq",
                ComparisonOp.LT: "<", ComparisonOp.LE: r"\leq",
                ComparisonOp.GT: ">", ComparisonOp.GE: r"\geq",
            }
            return f"{self._expr_latex(pred.left)} {op_map[pred.op]} {self._expr_latex(pred.right)}"
        if isinstance(pred, IsInstance):
            return rf"\mathit{{isinstance}}({pred.var}, \mathit{{{pred.tag}}})"
        if isinstance(pred, IsNone):
            return rf"{pred.var} = \mathit{{None}}"
        if isinstance(pred, IsTruthy):
            return rf"\mathit{{truthy}}({pred.var})"
        if isinstance(pred, HasAttr):
            return rf"\mathit{{hasattr}}({pred.var}, \texttt{{{pred.key}}})"
        if isinstance(pred, Not):
            return rf"\neg ({self.to_latex(pred.operand)})"
        if isinstance(pred, And):
            parts = [self._maybe_paren_latex(c, "and") for c in pred.conjuncts]
            return r" \wedge ".join(parts)
        if isinstance(pred, Or):
            parts = [self._maybe_paren_latex(d, "or") for d in pred.disjuncts]
            return r" \vee ".join(parts)
        if isinstance(pred, Implies):
            a = self._maybe_paren_latex(pred.antecedent, "implies")
            b = self._maybe_paren_latex(pred.consequent, "implies")
            return rf"{a} \Rightarrow {b}"
        if isinstance(pred, Iff):
            l = self._maybe_paren_latex(pred.left, "iff")
            r = self._maybe_paren_latex(pred.right, "iff")
            return rf"{l} \Leftrightarrow {r}"
        if isinstance(pred, ForAll):
            return rf"\forall {pred.var}\!:\!{pred.sort.value}.\; {self.to_latex(pred.body)}"
        if isinstance(pred, Exists):
            return rf"\exists {pred.var}\!:\!{pred.sort.value}.\; {self.to_latex(pred.body)}"
        return repr(pred)

    def _expr_latex(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            return str(expr.value)
        if isinstance(expr, Len):
            return rf"|{self._expr_latex(expr.arg)}|"
        if isinstance(expr, BinOp):
            l = self._expr_latex(expr.left)
            r = self._expr_latex(expr.right)
            op_map = {
                ArithOp.ADD: "+", ArithOp.SUB: "-",
                ArithOp.MUL: r"\cdot", ArithOp.DIV: r"\div",
                ArithOp.MOD: r"\bmod",
            }
            return f"({l} {op_map[expr.op]} {r})"
        if isinstance(expr, UnaryOp):
            inner = self._expr_latex(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return f"(-{inner})"
            if expr.op == UnaryArithOp.ABS:
                return rf"|{inner}|"
        return repr(expr)

    def _maybe_paren_latex(self, pred: Predicate, parent_op: str) -> str:
        s = self.to_latex(pred)
        needs_paren = False
        if parent_op == "and" and isinstance(pred, Or):
            needs_paren = True
        if parent_op == "or" and isinstance(pred, And):
            needs_paren = True
        if parent_op in ("implies", "iff") and isinstance(pred, (And, Or)):
            needs_paren = True
        return f"({s})" if needs_paren else s

    # -- Z3 Python API format ----------------------------------------------

    def to_z3_api(self, pred: Predicate) -> str:
        """Generate a string of Z3 Python API calls."""
        if isinstance(pred, BoolLit):
            return "BoolVal(True)" if pred.value else "BoolVal(False)"
        if isinstance(pred, Comparison):
            op_map = {
                ComparisonOp.EQ: "==", ComparisonOp.NE: "!=",
                ComparisonOp.LT: "<", ComparisonOp.LE: "<=",
                ComparisonOp.GT: ">", ComparisonOp.GE: ">=",
            }
            l = self._expr_z3(pred.left)
            r = self._expr_z3(pred.right)
            return f"({l} {op_map[pred.op]} {r})"
        if isinstance(pred, IsInstance):
            return f"(typeof_{pred.var} == Tag.{pred.tag})"
        if isinstance(pred, IsNone):
            return f"(is_none_{pred.var} == True)"
        if isinstance(pred, IsTruthy):
            return f"(is_truthy_{pred.var} == True)"
        if isinstance(pred, HasAttr):
            return f"(hasattr_{pred.var}_{pred.key} == True)"
        if isinstance(pred, Not):
            return f"Not({self.to_z3_api(pred.operand)})"
        if isinstance(pred, And):
            parts = ", ".join(self.to_z3_api(c) for c in pred.conjuncts)
            return f"And({parts})"
        if isinstance(pred, Or):
            parts = ", ".join(self.to_z3_api(d) for d in pred.disjuncts)
            return f"Or({parts})"
        if isinstance(pred, Implies):
            return f"Implies({self.to_z3_api(pred.antecedent)}, {self.to_z3_api(pred.consequent)})"
        if isinstance(pred, Iff):
            return f"({self.to_z3_api(pred.left)} == {self.to_z3_api(pred.right)})"
        if isinstance(pred, ForAll):
            return f"ForAll([{pred.var}], {self.to_z3_api(pred.body)})"
        if isinstance(pred, Exists):
            return f"Exists([{pred.var}], {self.to_z3_api(pred.body)})"
        return repr(pred)

    def _expr_z3(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            if isinstance(expr.value, int):
                return f"IntVal({expr.value})"
            if isinstance(expr.value, bool):
                return f"BoolVal({expr.value})"
            return f"StringVal({expr.value!r})"
        if isinstance(expr, Len):
            return f"len_fn({self._expr_z3(expr.arg)})"
        if isinstance(expr, BinOp):
            l = self._expr_z3(expr.left)
            r = self._expr_z3(expr.right)
            op_map = {
                ArithOp.ADD: "+", ArithOp.SUB: "-", ArithOp.MUL: "*",
                ArithOp.DIV: "/", ArithOp.MOD: "%",
            }
            return f"({l} {op_map[expr.op]} {r})"
        if isinstance(expr, UnaryOp):
            inner = self._expr_z3(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return f"(-{inner})"
            if expr.op == UnaryArithOp.ABS:
                return f"If({inner} >= 0, {inner}, -{inner})"
        return repr(expr)


# ---------------------------------------------------------------------------
# PredicateParser — Tokenizer + Recursive Descent
# ---------------------------------------------------------------------------

class TokenKind(enum.Enum):
    IDENT = "IDENT"
    INT_LIT = "INT_LIT"
    STR_LIT = "STR_LIT"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    DOT = "."
    COLON = ":"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"
    FORALL = "FORALL"
    EXISTS = "EXISTS"
    TRUE = "TRUE"
    FALSE = "FALSE"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    EOF = "EOF"


@dataclass
class Token:
    kind: TokenKind
    value: str
    pos: int


class Tokenizer:
    """Tokenize predicate strings."""

    _KEYWORDS = {
        "and": TokenKind.AND, "or": TokenKind.OR, "not": TokenKind.NOT,
        "implies": TokenKind.IMPLIES, "iff": TokenKind.IFF,
        "forall": TokenKind.FORALL, "exists": TokenKind.EXISTS,
        "true": TokenKind.TRUE, "false": TokenKind.FALSE,
        "True": TokenKind.TRUE, "False": TokenKind.FALSE,
        "isinstance": TokenKind.IDENT, "is_none": TokenKind.IDENT,
        "is_truthy": TokenKind.IDENT, "hasattr": TokenKind.IDENT,
        "len": TokenKind.IDENT, "abs": TokenKind.IDENT,
    }

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._tokens: List[Token] = []
        self._tokenize()

    def _tokenize(self) -> None:
        text = self._text
        pos = 0
        while pos < len(text):
            # Skip whitespace
            if text[pos].isspace():
                pos += 1
                continue

            # Two-character operators
            if pos + 1 < len(text):
                two = text[pos:pos + 2]
                if two == "==":
                    self._tokens.append(Token(TokenKind.EQ, two, pos))
                    pos += 2
                    continue
                if two == "!=":
                    self._tokens.append(Token(TokenKind.NE, two, pos))
                    pos += 2
                    continue
                if two == "<=":
                    self._tokens.append(Token(TokenKind.LE, two, pos))
                    pos += 2
                    continue
                if two == ">=":
                    self._tokens.append(Token(TokenKind.GE, two, pos))
                    pos += 2
                    continue
                if two == "=>":
                    self._tokens.append(Token(TokenKind.IMPLIES, two, pos))
                    pos += 2
                    continue

            c = text[pos]

            if c == "(":
                self._tokens.append(Token(TokenKind.LPAREN, c, pos))
                pos += 1
            elif c == ")":
                self._tokens.append(Token(TokenKind.RPAREN, c, pos))
                pos += 1
            elif c == ",":
                self._tokens.append(Token(TokenKind.COMMA, c, pos))
                pos += 1
            elif c == ".":
                self._tokens.append(Token(TokenKind.DOT, c, pos))
                pos += 1
            elif c == ":":
                self._tokens.append(Token(TokenKind.COLON, c, pos))
                pos += 1
            elif c == "<":
                self._tokens.append(Token(TokenKind.LT, c, pos))
                pos += 1
            elif c == ">":
                self._tokens.append(Token(TokenKind.GT, c, pos))
                pos += 1
            elif c == "+":
                self._tokens.append(Token(TokenKind.PLUS, c, pos))
                pos += 1
            elif c == "-":
                # Could be negative number or subtraction
                if pos + 1 < len(text) and text[pos + 1].isdigit():
                    # Check if preceded by an operator or at start
                    if not self._tokens or self._tokens[-1].kind in (
                        TokenKind.LPAREN, TokenKind.COMMA, TokenKind.AND,
                        TokenKind.OR, TokenKind.NOT, TokenKind.EQ, TokenKind.NE,
                        TokenKind.LT, TokenKind.LE, TokenKind.GT, TokenKind.GE,
                        TokenKind.PLUS, TokenKind.MINUS, TokenKind.STAR,
                        TokenKind.SLASH, TokenKind.PERCENT,
                    ):
                        end = pos + 1
                        while end < len(text) and text[end].isdigit():
                            end += 1
                        self._tokens.append(
                            Token(TokenKind.INT_LIT, text[pos:end], pos)
                        )
                        pos = end
                        continue
                self._tokens.append(Token(TokenKind.MINUS, c, pos))
                pos += 1
            elif c == "*":
                self._tokens.append(Token(TokenKind.STAR, c, pos))
                pos += 1
            elif c == "/":
                self._tokens.append(Token(TokenKind.SLASH, c, pos))
                pos += 1
            elif c == "%":
                self._tokens.append(Token(TokenKind.PERCENT, c, pos))
                pos += 1
            elif c == "'" or c == '"':
                end = pos + 1
                while end < len(text) and text[end] != c:
                    end += 1
                if end < len(text):
                    end += 1
                self._tokens.append(
                    Token(TokenKind.STR_LIT, text[pos + 1:end - 1], pos)
                )
                pos = end
            elif c == "¬":
                self._tokens.append(Token(TokenKind.NOT, "¬", pos))
                pos += 1
            elif c == "∧":
                self._tokens.append(Token(TokenKind.AND, "∧", pos))
                pos += 1
            elif c == "∨":
                self._tokens.append(Token(TokenKind.OR, "∨", pos))
                pos += 1
            elif c == "→":
                self._tokens.append(Token(TokenKind.IMPLIES, "→", pos))
                pos += 1
            elif c == "↔":
                self._tokens.append(Token(TokenKind.IFF, "↔", pos))
                pos += 1
            elif c == "∀":
                self._tokens.append(Token(TokenKind.FORALL, "∀", pos))
                pos += 1
            elif c == "∃":
                self._tokens.append(Token(TokenKind.EXISTS, "∃", pos))
                pos += 1
            elif c.isdigit():
                end = pos
                while end < len(text) and text[end].isdigit():
                    end += 1
                self._tokens.append(
                    Token(TokenKind.INT_LIT, text[pos:end], pos)
                )
                pos = end
            elif c.isalpha() or c == "_":
                end = pos
                while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                    end += 1
                word = text[pos:end]
                kind = self._KEYWORDS.get(word, TokenKind.IDENT)
                self._tokens.append(Token(kind, word, pos))
                pos = end
            else:
                pos += 1  # skip unknown

        self._tokens.append(Token(TokenKind.EOF, "", len(text)))

    @property
    def tokens(self) -> List[Token]:
        return self._tokens


class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class PredicateParser:
    """Recursive descent parser for predicates."""

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos = 0

    # -- public API --------------------------------------------------------

    def from_string(self, text: str) -> Predicate:
        """Parse a predicate from a human-readable string."""
        tokenizer = Tokenizer(text)
        self._tokens = tokenizer.tokens
        self._pos = 0
        result = self._parse_iff()
        if self._peek().kind != TokenKind.EOF:
            raise ParseError(
                f"Unexpected token {self._peek().value!r} at pos {self._peek().pos}"
            )
        return result

    def from_smt_lib(self, text: str) -> Predicate:
        """Parse a predicate from SMT-LIB format (S-expression)."""
        text = text.strip()
        if text == "true":
            return BoolLit(True)
        if text == "false":
            return BoolLit(False)
        if not text.startswith("("):
            # Bare identifier or number
            if text.lstrip("-").isdigit():
                return Comparison(
                    ComparisonOp.EQ,
                    Var("_smt_val"),
                    Const(int(text)),
                )
            return BoolLit(True)  # unrecognized atom

        # Strip outer parens and parse
        inner = text[1:-1].strip()
        parts = self._split_sexp(inner)
        if not parts:
            return BoolLit(True)

        head = parts[0]
        if head == "and":
            return mk_and(*(self.from_smt_lib(p) for p in parts[1:]))
        if head == "or":
            return mk_or(*(self.from_smt_lib(p) for p in parts[1:]))
        if head == "not":
            return mk_not(self.from_smt_lib(parts[1]))
        if head == "=>":
            return Implies(
                self.from_smt_lib(parts[1]),
                self.from_smt_lib(parts[2]),
            )
        if head in ("=", "distinct", "<", "<=", ">", ">="):
            op_map = {
                "=": ComparisonOp.EQ, "distinct": ComparisonOp.NE,
                "<": ComparisonOp.LT, "<=": ComparisonOp.LE,
                ">": ComparisonOp.GT, ">=": ComparisonOp.GE,
            }
            left = self._parse_smt_expr(parts[1])
            right = self._parse_smt_expr(parts[2])
            return Comparison(op_map[head], left, right)
        return BoolLit(True)

    def from_python_expr(self, text: str) -> Predicate:
        """Parse a Python-style expression as a predicate."""
        # Rewrite Python keywords to parser keywords
        text = text.replace(" and ", " and ")
        text = text.replace(" or ", " or ")
        text = text.replace("not ", "not ")
        text = text.replace(" is None", " IS_NONE_MARKER")
        text = text.replace(" is not None", " IS_NOT_NONE_MARKER")

        # Handle "x is None" patterns
        is_none_pattern = re.compile(r"(\w+)\s+IS_NONE_MARKER")
        for m in is_none_pattern.finditer(text):
            var = m.group(1)
            text = text.replace(m.group(0), f"is_none({var})")

        is_not_none_pattern = re.compile(r"(\w+)\s+IS_NOT_NONE_MARKER")
        for m in is_not_none_pattern.finditer(text):
            var = m.group(1)
            text = text.replace(m.group(0), f"not is_none({var})")

        return self.from_string(text)

    # -- recursive descent -------------------------------------------------

    def _peek(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(TokenKind.EOF, "", -1)

    def _advance(self) -> Token:
        tok = self._peek()
        self._pos += 1
        return tok

    def _expect(self, kind: TokenKind) -> Token:
        tok = self._advance()
        if tok.kind != kind:
            raise ParseError(
                f"Expected {kind.value}, got {tok.value!r} at pos {tok.pos}"
            )
        return tok

    def _parse_iff(self) -> Predicate:
        left = self._parse_implies()
        while self._peek().kind == TokenKind.IFF:
            self._advance()
            right = self._parse_implies()
            left = Iff(left, right)
        return left

    def _parse_implies(self) -> Predicate:
        left = self._parse_or()
        while self._peek().kind == TokenKind.IMPLIES:
            self._advance()
            right = self._parse_or()
            left = Implies(left, right)
        return left

    def _parse_or(self) -> Predicate:
        left = self._parse_and()
        parts = [left]
        while self._peek().kind == TokenKind.OR:
            self._advance()
            parts.append(self._parse_and())
        if len(parts) == 1:
            return parts[0]
        return Or(tuple(parts))

    def _parse_and(self) -> Predicate:
        left = self._parse_not()
        parts = [left]
        while self._peek().kind == TokenKind.AND:
            self._advance()
            parts.append(self._parse_not())
        if len(parts) == 1:
            return parts[0]
        return And(tuple(parts))

    def _parse_not(self) -> Predicate:
        if self._peek().kind == TokenKind.NOT:
            self._advance()
            operand = self._parse_not()
            return Not(operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> Predicate:
        left_expr = self._parse_additive()

        cmp_ops = {
            TokenKind.EQ: ComparisonOp.EQ, TokenKind.NE: ComparisonOp.NE,
            TokenKind.LT: ComparisonOp.LT, TokenKind.LE: ComparisonOp.LE,
            TokenKind.GT: ComparisonOp.GT, TokenKind.GE: ComparisonOp.GE,
        }
        if self._peek().kind in cmp_ops:
            op_tok = self._advance()
            right_expr = self._parse_additive()
            return Comparison(cmp_ops[op_tok.kind], left_expr, right_expr)

        # If the expression was just a function call that returns a predicate, handle it
        if isinstance(left_expr, Var):
            return IsTruthy(left_expr.name)

        # Shouldn't normally get here for well-formed predicates
        return BoolLit(True)

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self._peek().kind in (TokenKind.PLUS, TokenKind.MINUS):
            op_tok = self._advance()
            right = self._parse_multiplicative()
            arith_op = ArithOp.ADD if op_tok.kind == TokenKind.PLUS else ArithOp.SUB
            left = BinOp(arith_op, left, right)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while self._peek().kind in (TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT):
            op_tok = self._advance()
            right = self._parse_unary()
            if op_tok.kind == TokenKind.STAR:
                left = BinOp(ArithOp.MUL, left, right)
            elif op_tok.kind == TokenKind.SLASH:
                left = BinOp(ArithOp.DIV, left, right)
            else:
                left = BinOp(ArithOp.MOD, left, right)
        return left

    def _parse_unary(self) -> Expr:
        if self._peek().kind == TokenKind.MINUS:
            self._advance()
            operand = self._parse_primary()
            return UnaryOp(UnaryArithOp.NEG, operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        tok = self._peek()

        if tok.kind == TokenKind.INT_LIT:
            self._advance()
            return Const(int(tok.value))

        if tok.kind == TokenKind.STR_LIT:
            self._advance()
            return Const(tok.value, Sort.STR)

        if tok.kind == TokenKind.TRUE:
            self._advance()
            return Const(True, Sort.BOOL)

        if tok.kind == TokenKind.FALSE:
            self._advance()
            return Const(False, Sort.BOOL)

        if tok.kind == TokenKind.IDENT:
            name = tok.value
            self._advance()

            # Function call?
            if self._peek().kind == TokenKind.LPAREN:
                self._advance()  # consume (
                if name == "len":
                    arg = self._parse_additive()
                    self._expect(TokenKind.RPAREN)
                    return Len(arg)
                if name == "abs":
                    arg = self._parse_additive()
                    self._expect(TokenKind.RPAREN)
                    return UnaryOp(UnaryArithOp.ABS, arg)
                if name == "isinstance":
                    var_tok = self._expect(TokenKind.IDENT)
                    self._expect(TokenKind.COMMA)
                    tag_tok = self._expect(TokenKind.IDENT)
                    self._expect(TokenKind.RPAREN)
                    # Return a special marker; caller must handle
                    # We encode as a var; the comparison parser will pick it up
                    # Actually, we need to return the predicate, not an expression.
                    # We cheat by storing in a special wrapper
                    self._pos -= 1  # back up to before RPAREN was consumed
                    self._pos -= 1  # back up more
                    self._pos -= 1  # COMMA
                    self._pos -= 1  # var_tok
                    self._pos -= 1  # LPAREN was already consumed, back up to name
                    self._pos -= 0  # Actually let's just handle this properly
                    # Re-parse by rewinding
                    # Simpler approach: just return a Var and let the comparison
                    # level handle isinstance as an atomic predicate
                    self._pos = self._pos  # stay where we are
                    self._expect(TokenKind.RPAREN)
                    return Var(f"__isinstance__{var_tok.value}__{tag_tok.value}")
                if name == "is_none":
                    var_tok = self._expect(TokenKind.IDENT)
                    self._expect(TokenKind.RPAREN)
                    return Var(f"__is_none__{var_tok.value}")
                if name == "is_truthy":
                    var_tok = self._expect(TokenKind.IDENT)
                    self._expect(TokenKind.RPAREN)
                    return Var(f"__is_truthy__{var_tok.value}")
                if name == "hasattr":
                    var_tok = self._expect(TokenKind.IDENT)
                    self._expect(TokenKind.COMMA)
                    key_tok = self._peek()
                    if key_tok.kind == TokenKind.STR_LIT:
                        self._advance()
                        key = key_tok.value
                    else:
                        k = self._expect(TokenKind.IDENT)
                        key = k.value
                    self._expect(TokenKind.RPAREN)
                    return Var(f"__hasattr__{var_tok.value}__{key}")
                # Generic function: skip args
                depth = 1
                while depth > 0 and self._peek().kind != TokenKind.EOF:
                    t = self._advance()
                    if t.kind == TokenKind.LPAREN:
                        depth += 1
                    elif t.kind == TokenKind.RPAREN:
                        depth -= 1
                return Var(name)

            return Var(name)

        if tok.kind == TokenKind.LPAREN:
            self._advance()
            # Could be a predicate in parens or an expression
            # Try parsing as a full predicate
            saved_pos = self._pos
            try:
                inner_pred = self._parse_iff()
                self._expect(TokenKind.RPAREN)
                # If we got a predicate, wrap it
                if isinstance(inner_pred, (Comparison, IsInstance, IsNone,
                                          IsTruthy, HasAttr, And, Or, Not,
                                          Implies, Iff, BoolLit)):
                    return Var(f"__pred__{id(inner_pred)}")
            except ParseError:
                self._pos = saved_pos

            # Parse as expression
            inner = self._parse_additive()
            self._expect(TokenKind.RPAREN)
            return inner

        raise ParseError(f"Unexpected token {tok.value!r} at pos {tok.pos}")

    # -- SMT-LIB S-expression helpers --------------------------------------

    def _split_sexp(self, text: str) -> List[str]:
        """Split an S-expression into top-level parts."""
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        i = 0
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
                current.append(c)
            elif c == ")":
                depth -= 1
                current.append(c)
            elif c.isspace() and depth == 0:
                if current:
                    parts.append("".join(current))
                    current = []
            else:
                current.append(c)
            i += 1
        if current:
            parts.append("".join(current))
        return parts

    def _parse_smt_expr(self, text: str) -> Expr:
        """Parse an SMT-LIB expression atom."""
        text = text.strip()
        if text.lstrip("-").isdigit():
            return Const(int(text))
        if text.startswith('"') and text.endswith('"'):
            return Const(text[1:-1], Sort.STR)
        if text == "true":
            return Const(True, Sort.BOOL)
        if text == "false":
            return Const(False, Sort.BOOL)
        if text.startswith("("):
            inner = text[1:-1].strip()
            parts = self._split_sexp(inner)
            if not parts:
                return Var("_empty")
            head = parts[0]
            if head == "+":
                return BinOp(ArithOp.ADD, self._parse_smt_expr(parts[1]), self._parse_smt_expr(parts[2]))
            if head == "-":
                if len(parts) == 2:
                    return UnaryOp(UnaryArithOp.NEG, self._parse_smt_expr(parts[1]))
                return BinOp(ArithOp.SUB, self._parse_smt_expr(parts[1]), self._parse_smt_expr(parts[2]))
            if head == "*":
                return BinOp(ArithOp.MUL, self._parse_smt_expr(parts[1]), self._parse_smt_expr(parts[2]))
            if head == "div":
                return BinOp(ArithOp.DIV, self._parse_smt_expr(parts[1]), self._parse_smt_expr(parts[2]))
            if head == "mod":
                return BinOp(ArithOp.MOD, self._parse_smt_expr(parts[1]), self._parse_smt_expr(parts[2]))
            if head == "len":
                return Len(self._parse_smt_expr(parts[1]))
            if head == "abs":
                return UnaryOp(UnaryArithOp.ABS, self._parse_smt_expr(parts[1]))
            return Var(head)
        return Var(text)


# ---------------------------------------------------------------------------
# PredicateWalker — generic traversal
# ---------------------------------------------------------------------------

class PredicateWalker:
    """Walk predicates, collecting / transforming."""

    def walk_pred(self, pred: Predicate) -> List[Predicate]:
        """Pre-order walk of all sub-predicates."""
        result = [pred]
        if isinstance(pred, And):
            for c in pred.conjuncts:
                result.extend(self.walk_pred(c))
        elif isinstance(pred, Or):
            for d in pred.disjuncts:
                result.extend(self.walk_pred(d))
        elif isinstance(pred, Not):
            result.extend(self.walk_pred(pred.operand))
        elif isinstance(pred, Implies):
            result.extend(self.walk_pred(pred.antecedent))
            result.extend(self.walk_pred(pred.consequent))
        elif isinstance(pred, Iff):
            result.extend(self.walk_pred(pred.left))
            result.extend(self.walk_pred(pred.right))
        elif isinstance(pred, (ForAll, Exists)):
            result.extend(self.walk_pred(pred.body))
        return result

    def walk_expr(self, expr: Expr) -> List[Expr]:
        result = [expr]
        if isinstance(expr, BinOp):
            result.extend(self.walk_expr(expr.left))
            result.extend(self.walk_expr(expr.right))
        elif isinstance(expr, UnaryOp):
            result.extend(self.walk_expr(expr.operand))
        elif isinstance(expr, Len):
            result.extend(self.walk_expr(expr.arg))
        return result

    def collect_atoms(self, pred: Predicate) -> List[Predicate]:
        """Collect all atomic predicates."""
        result: List[Predicate] = []
        for sub in self.walk_pred(pred):
            if isinstance(sub, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr, BoolLit)):
                result.append(sub)
        return result

    def collect_variables(self, pred: Predicate) -> Set[str]:
        """Collect all variable names."""
        return set(pred.free_vars())

    def map_atoms(
        self, pred: Predicate, fn: Callable[[Predicate], Predicate]
    ) -> Predicate:
        """Apply a function to each atom."""
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr, BoolLit)):
            return fn(pred)
        if isinstance(pred, Not):
            return Not(self.map_atoms(pred.operand, fn))
        if isinstance(pred, And):
            return And(tuple(self.map_atoms(c, fn) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(self.map_atoms(d, fn) for d in pred.disjuncts))
        if isinstance(pred, Implies):
            return Implies(
                self.map_atoms(pred.antecedent, fn),
                self.map_atoms(pred.consequent, fn),
            )
        if isinstance(pred, Iff):
            return Iff(
                self.map_atoms(pred.left, fn),
                self.map_atoms(pred.right, fn),
            )
        return pred

    def size(self, pred: Predicate) -> int:
        """Count the number of nodes in the predicate AST."""
        return len(self.walk_pred(pred))

    def depth(self, pred: Predicate) -> int:
        """Compute the depth of the predicate AST."""
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy,
                             HasAttr, BoolLit)):
            return 1
        if isinstance(pred, Not):
            return 1 + self.depth(pred.operand)
        if isinstance(pred, And):
            return 1 + max((self.depth(c) for c in pred.conjuncts), default=0)
        if isinstance(pred, Or):
            return 1 + max((self.depth(d) for d in pred.disjuncts), default=0)
        if isinstance(pred, Implies):
            return 1 + max(self.depth(pred.antecedent), self.depth(pred.consequent))
        if isinstance(pred, Iff):
            return 1 + max(self.depth(pred.left), self.depth(pred.right))
        if isinstance(pred, (ForAll, Exists)):
            return 1 + self.depth(pred.body)
        return 1


# ---------------------------------------------------------------------------
# PredicateHash — structural hashing for efficient lookups
# ---------------------------------------------------------------------------

class PredicateHash:
    """Compute structural hashes for predicates (for use in sets/dicts)."""

    def hash_pred(self, pred: Predicate) -> int:
        if isinstance(pred, BoolLit):
            return hash(("BoolLit", pred.value))
        if isinstance(pred, Comparison):
            return hash(("Cmp", pred.op, self.hash_expr(pred.left), self.hash_expr(pred.right)))
        if isinstance(pred, IsInstance):
            return hash(("IsInst", pred.var, pred.tag))
        if isinstance(pred, IsNone):
            return hash(("IsNone", pred.var))
        if isinstance(pred, IsTruthy):
            return hash(("IsTruthy", pred.var))
        if isinstance(pred, HasAttr):
            return hash(("HasAttr", pred.var, pred.key))
        if isinstance(pred, Not):
            return hash(("Not", self.hash_pred(pred.operand)))
        if isinstance(pred, And):
            return hash(("And",) + tuple(self.hash_pred(c) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return hash(("Or",) + tuple(self.hash_pred(d) for d in pred.disjuncts))
        if isinstance(pred, Implies):
            return hash(("Implies", self.hash_pred(pred.antecedent), self.hash_pred(pred.consequent)))
        if isinstance(pred, Iff):
            return hash(("Iff", self.hash_pred(pred.left), self.hash_pred(pred.right)))
        return hash(repr(pred))

    def hash_expr(self, expr: Expr) -> int:
        if isinstance(expr, Var):
            return hash(("Var", expr.name))
        if isinstance(expr, Const):
            return hash(("Const", expr.value))
        if isinstance(expr, Len):
            return hash(("Len", self.hash_expr(expr.arg)))
        if isinstance(expr, BinOp):
            return hash(("BinOp", expr.op, self.hash_expr(expr.left), self.hash_expr(expr.right)))
        if isinstance(expr, UnaryOp):
            return hash(("UnaryOp", expr.op, self.hash_expr(expr.operand)))
        return hash(repr(expr))


# ---------------------------------------------------------------------------
# Expression simplification
# ---------------------------------------------------------------------------

class ExpressionSimplifier:
    """Simplify arithmetic expressions (constant folding, identity removal)."""

    def simplify(self, expr: Expr) -> Expr:
        if isinstance(expr, (Var, Const)):
            return expr
        if isinstance(expr, Len):
            return Len(self.simplify(expr.arg))
        if isinstance(expr, UnaryOp):
            inner = self.simplify(expr.operand)
            if isinstance(inner, Const) and isinstance(inner.value, (int, float)):
                if expr.op == UnaryArithOp.NEG:
                    return Const(-inner.value)
                if expr.op == UnaryArithOp.ABS:
                    return Const(abs(inner.value))
            # Double negation
            if expr.op == UnaryArithOp.NEG and isinstance(inner, UnaryOp) and inner.op == UnaryArithOp.NEG:
                return inner.operand
            return UnaryOp(expr.op, inner)
        if isinstance(expr, BinOp):
            left = self.simplify(expr.left)
            right = self.simplify(expr.right)
            # Constant folding
            if isinstance(left, Const) and isinstance(right, Const):
                if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                    try:
                        evaluator = PredicateEvaluator()
                        result = evaluator._eval_binop(expr.op, left.value, right.value)
                        return Const(result)
                    except PredicateEvaluationError:
                        pass
            # Identity: x + 0 = x, x - 0 = x, x * 1 = x, x * 0 = 0
            if expr.op == ArithOp.ADD:
                if isinstance(right, Const) and right.value == 0:
                    return left
                if isinstance(left, Const) and left.value == 0:
                    return right
            if expr.op == ArithOp.SUB:
                if isinstance(right, Const) and right.value == 0:
                    return left
                if left == right:
                    return Const(0)
            if expr.op == ArithOp.MUL:
                if isinstance(right, Const) and right.value == 1:
                    return left
                if isinstance(left, Const) and left.value == 1:
                    return right
                if isinstance(right, Const) and right.value == 0:
                    return Const(0)
                if isinstance(left, Const) and left.value == 0:
                    return Const(0)
            if expr.op == ArithOp.DIV:
                if isinstance(right, Const) and right.value == 1:
                    return left
            return BinOp(expr.op, left, right)
        return expr


# ---------------------------------------------------------------------------
# PredicateMetrics — analysis / statistics
# ---------------------------------------------------------------------------

class PredicateMetrics:
    """Compute metrics about predicates."""

    def __init__(self) -> None:
        self._walker = PredicateWalker()

    def atom_count(self, pred: Predicate) -> int:
        return len(self._walker.collect_atoms(pred))

    def variable_count(self, pred: Predicate) -> int:
        return len(self._walker.collect_variables(pred))

    def connective_count(self, pred: Predicate) -> int:
        count = 0
        for sub in self._walker.walk_pred(pred):
            if isinstance(sub, (And, Or, Not, Implies, Iff)):
                count += 1
        return count

    def is_atomic(self, pred: Predicate) -> bool:
        return isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr, BoolLit))

    def is_literal(self, pred: Predicate) -> bool:
        if self.is_atomic(pred):
            return True
        if isinstance(pred, Not) and self.is_atomic(pred.operand):
            return True
        return False

    def is_clause(self, pred: Predicate) -> bool:
        """Check if predicate is a clause (disjunction of literals)."""
        if self.is_literal(pred):
            return True
        if isinstance(pred, Or):
            return all(self.is_literal(d) for d in pred.disjuncts)
        return False

    def is_cnf(self, pred: Predicate) -> bool:
        """Check if predicate is in CNF."""
        if self.is_clause(pred):
            return True
        if isinstance(pred, And):
            return all(self.is_clause(c) for c in pred.conjuncts)
        return False

    def is_dnf(self, pred: Predicate) -> bool:
        """Check if predicate is in DNF."""
        if self.is_literal(pred):
            return True
        if isinstance(pred, And):
            return all(self.is_literal(c) for c in pred.conjuncts)
        if isinstance(pred, Or):
            for d in pred.disjuncts:
                if isinstance(d, And):
                    if not all(self.is_literal(c) for c in d.conjuncts):
                        return False
                elif not self.is_literal(d):
                    return False
            return True
        return False

    def is_horn(self, pred: Predicate) -> bool:
        """Check if predicate is a Horn clause (at most one positive literal per clause)."""
        if not self.is_cnf(pred):
            return False
        clauses: List[Predicate] = []
        if isinstance(pred, And):
            clauses = list(pred.conjuncts)
        else:
            clauses = [pred]
        for clause in clauses:
            literals: List[Predicate] = []
            if isinstance(clause, Or):
                literals = list(clause.disjuncts)
            else:
                literals = [clause]
            positive_count = sum(1 for lit in literals if not isinstance(lit, Not))
            if positive_count > 1:
                return False
        return True

    def summary(self, pred: Predicate) -> Dict[str, Any]:
        """Compute a summary of predicate metrics."""
        return {
            "size": self._walker.size(pred),
            "depth": self._walker.depth(pred),
            "atoms": self.atom_count(pred),
            "variables": self.variable_count(pred),
            "connectives": self.connective_count(pred),
            "is_atomic": self.is_atomic(pred),
            "is_literal": self.is_literal(pred),
            "is_cnf": self.is_cnf(pred),
            "is_dnf": self.is_dnf(pred),
            "free_vars": sorted(pred.free_vars()),
        }


# ---------------------------------------------------------------------------
# PredicateEquivalence — semantic and syntactic equivalence checking
# ---------------------------------------------------------------------------

class PredicateEquivalence:
    """Check equivalence of predicates."""

    def __init__(self) -> None:
        self._normalizer = PredicateNormalizer()
        self._implication = PredicateImplication()

    def syntactic_equal(self, p1: Predicate, p2: Predicate) -> bool:
        return p1 == p2

    def normalized_equal(self, p1: Predicate, p2: Predicate) -> bool:
        n1 = self._normalizer.normalize(p1)
        n2 = self._normalizer.normalize(p2)
        return n1 == n2

    def semantic_equal(self, p1: Predicate, p2: Predicate) -> bool:
        """Check semantic equivalence via mutual implication."""
        return self._implication.implies(p1, p2) and self._implication.implies(p2, p1)

    def is_stronger(self, p1: Predicate, p2: Predicate) -> bool:
        """p1 is strictly stronger than p2: p1 ⊨ p2 but not p2 ⊨ p1."""
        return self._implication.implies(p1, p2) and not self._implication.implies(p2, p1)

    def is_weaker(self, p1: Predicate, p2: Predicate) -> bool:
        """p1 is strictly weaker than p2."""
        return self.is_stronger(p2, p1)

    def compare(self, p1: Predicate, p2: Predicate) -> str:
        """Compare two predicates: 'equivalent', 'stronger', 'weaker', 'incomparable'."""
        fwd = self._implication.implies(p1, p2)
        bwd = self._implication.implies(p2, p1)
        if fwd and bwd:
            return "equivalent"
        if fwd:
            return "stronger"
        if bwd:
            return "weaker"
        return "incomparable"


# ---------------------------------------------------------------------------
# Utility: convert between representations
# ---------------------------------------------------------------------------

class PredicateConverter:
    """Convert predicates between different representations."""

    def __init__(self) -> None:
        self._printer = PredicatePrinter()
        self._parser = PredicateParser()

    def pred_to_string(self, pred: Predicate) -> str:
        return self._printer.to_string(pred)

    def string_to_pred(self, text: str) -> Predicate:
        return self._parser.from_string(text)

    def pred_to_smt(self, pred: Predicate) -> str:
        return self._printer.to_smt_lib(pred)

    def smt_to_pred(self, text: str) -> Predicate:
        return self._parser.from_smt_lib(text)

    def pred_to_python(self, pred: Predicate) -> str:
        return self._printer.to_python(pred)

    def python_to_pred(self, text: str) -> Predicate:
        return self._parser.from_python_expr(text)

    def pred_to_latex(self, pred: Predicate) -> str:
        return self._printer.to_latex(pred)

    def pred_to_z3(self, pred: Predicate) -> str:
        return self._printer.to_z3_api(pred)


# ---------------------------------------------------------------------------
# PredicateFactory — convenience for building common predicates
# ---------------------------------------------------------------------------

class PredicateFactory:
    """Factory methods for building common predicates."""

    @staticmethod
    def var_eq(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.EQ, Var(var), Const(val))

    @staticmethod
    def var_ne(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.NE, Var(var), Const(val))

    @staticmethod
    def var_lt(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.LT, Var(var), Const(val))

    @staticmethod
    def var_le(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.LE, Var(var), Const(val))

    @staticmethod
    def var_gt(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.GT, Var(var), Const(val))

    @staticmethod
    def var_ge(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.GE, Var(var), Const(val))

    @staticmethod
    def var_in_range(var: str, lo: int, hi: int) -> And:
        return And((
            Comparison(ComparisonOp.GE, Var(var), Const(lo)),
            Comparison(ComparisonOp.LT, Var(var), Const(hi)),
        ))

    @staticmethod
    def var_isinstance(var: str, tag: str) -> IsInstance:
        return IsInstance(var, tag)

    @staticmethod
    def var_is_none(var: str) -> IsNone:
        return IsNone(var)

    @staticmethod
    def var_not_none(var: str) -> Not:
        return Not(IsNone(var))

    @staticmethod
    def var_is_truthy(var: str) -> IsTruthy:
        return IsTruthy(var)

    @staticmethod
    def var_hasattr(var: str, key: str) -> HasAttr:
        return HasAttr(var, key)

    @staticmethod
    def vars_eq(var1: str, var2: str) -> Comparison:
        return Comparison(ComparisonOp.EQ, Var(var1), Var(var2))

    @staticmethod
    def vars_lt(var1: str, var2: str) -> Comparison:
        return Comparison(ComparisonOp.LT, Var(var1), Var(var2))

    @staticmethod
    def len_ge(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.GE, Len(Var(var)), Const(val))

    @staticmethod
    def len_eq(var: str, val: int) -> Comparison:
        return Comparison(ComparisonOp.EQ, Len(Var(var)), Const(val))

    @staticmethod
    def non_negative(var: str) -> Comparison:
        return Comparison(ComparisonOp.GE, Var(var), Const(0))

    @staticmethod
    def positive(var: str) -> Comparison:
        return Comparison(ComparisonOp.GT, Var(var), Const(0))

    @staticmethod
    def type_union(var: str, tags: List[str]) -> Or:
        return Or(tuple(IsInstance(var, tag) for tag in tags))

    @staticmethod
    def optional_type(var: str, tag: str) -> Or:
        return Or((IsInstance(var, tag), IsNone(var)))

    @staticmethod
    def true_pred() -> BoolLit:
        return BoolLit(True)

    @staticmethod
    def false_pred() -> BoolLit:
        return BoolLit(False)


# ---------------------------------------------------------------------------
# IntervalAnalysis — compute intervals from predicates
# ---------------------------------------------------------------------------

class IntervalAnalysis:
    """Extract interval constraints from predicates."""

    def extract_intervals(
        self, pred: Predicate
    ) -> Dict[str, Interval]:
        """Extract intervals for each variable from a conjunction of comparisons."""
        intervals: Dict[str, Interval] = {}

        atoms = self._collect_comparisons(pred)
        for cmp in atoms:
            if isinstance(cmp.left, Var) and isinstance(cmp.right, Const):
                var = cmp.left.name
                val = cmp.right.value
                if not isinstance(val, int):
                    continue
                if var not in intervals:
                    intervals[var] = Interval()
                iv = intervals[var]
                if cmp.op == ComparisonOp.GE:
                    iv.lo = max(iv.lo, val) if iv.lo is not None else val
                elif cmp.op == ComparisonOp.GT:
                    iv.lo = max(iv.lo, val + 1) if iv.lo is not None else val + 1
                elif cmp.op == ComparisonOp.LE:
                    iv.hi = min(iv.hi, val + 1) if iv.hi is not None else val + 1
                elif cmp.op == ComparisonOp.LT:
                    iv.hi = min(iv.hi, val) if iv.hi is not None else val
                elif cmp.op == ComparisonOp.EQ:
                    iv.lo = max(iv.lo, val) if iv.lo is not None else val
                    iv.hi = min(iv.hi, val + 1) if iv.hi is not None else val + 1
        return intervals

    def intervals_satisfiable(self, intervals: Dict[str, Interval]) -> bool:
        """Check if all intervals are non-empty."""
        for iv in intervals.values():
            if iv.is_empty:
                return False
        return True

    def intervals_to_predicate(
        self, intervals: Dict[str, Interval]
    ) -> Predicate:
        """Convert intervals back to predicates."""
        conjuncts: List[Predicate] = []
        for var, iv in sorted(intervals.items()):
            if iv.lo is not None:
                conjuncts.append(
                    Comparison(ComparisonOp.GE, Var(var), Const(iv.lo))
                )
            if iv.hi is not None:
                conjuncts.append(
                    Comparison(ComparisonOp.LT, Var(var), Const(iv.hi))
                )
        if not conjuncts:
            return BoolLit(True)
        return mk_and(*conjuncts)

    def widen_intervals(
        self,
        old: Dict[str, Interval],
        new: Dict[str, Interval],
    ) -> Dict[str, Interval]:
        """Widening on intervals."""
        result: Dict[str, Interval] = {}
        all_vars = set(old) | set(new)
        for var in all_vars:
            old_iv = old.get(var, Interval())
            new_iv = new.get(var, Interval())
            result[var] = old_iv.widen(new_iv)
        return result

    def narrow_intervals(
        self,
        old: Dict[str, Interval],
        new: Dict[str, Interval],
    ) -> Dict[str, Interval]:
        """Narrowing on intervals."""
        result: Dict[str, Interval] = {}
        all_vars = set(old) | set(new)
        for var in all_vars:
            old_iv = old.get(var, Interval())
            new_iv = new.get(var, Interval())
            result[var] = old_iv.narrow(new_iv)
        return result

    def _collect_comparisons(self, pred: Predicate) -> List[Comparison]:
        if isinstance(pred, Comparison):
            return [pred]
        if isinstance(pred, And):
            result: List[Comparison] = []
            for c in pred.conjuncts:
                result.extend(self._collect_comparisons(c))
            return result
        return []


# ---------------------------------------------------------------------------
# TypeTagAnalysis — reason about type tags
# ---------------------------------------------------------------------------

class TypeTagAnalysis:
    """Analyze type-tag predicates."""

    def extract_possible_tags(
        self, pred: Predicate, var: str
    ) -> Optional[Set[str]]:
        """Extract the set of possible type tags for a variable.

        Returns None if unconstrained.
        """
        positive: Set[str] = set()
        negative: Set[str] = set()
        has_positive = False

        atoms = self._collect_type_atoms(pred, var)
        for is_pos, tag in atoms:
            if is_pos:
                positive.add(tag)
                has_positive = True
            else:
                negative.add(tag)

        if not has_positive and not negative:
            return None

        if has_positive:
            return positive - negative

        # Only negatives: universe minus negatives
        all_tags = set(_TAG_TO_TYPE.keys())
        return all_tags - negative

    def is_definitely_type(
        self, pred: Predicate, var: str, tag: str
    ) -> bool:
        """Check if pred definitely constrains var to be of type tag."""
        possible = self.extract_possible_tags(pred, var)
        if possible is None:
            return False
        return possible == {tag}

    def is_definitely_not_type(
        self, pred: Predicate, var: str, tag: str
    ) -> bool:
        """Check if pred definitely constrains var to NOT be of type tag."""
        possible = self.extract_possible_tags(pred, var)
        if possible is None:
            return False
        return tag not in possible

    def is_optional(self, pred: Predicate, var: str) -> bool:
        """Check if var can be None according to pred."""
        possible = self.extract_possible_tags(pred, var)
        if possible is None:
            return True
        return "NoneType" in possible

    def _collect_type_atoms(
        self, pred: Predicate, var: str
    ) -> List[Tuple[bool, str]]:
        """Collect (is_positive, tag) pairs for a variable."""
        result: List[Tuple[bool, str]] = []
        if isinstance(pred, IsInstance) and pred.var == var:
            result.append((True, pred.tag))
        elif isinstance(pred, Not):
            if isinstance(pred.operand, IsInstance) and pred.operand.var == var:
                result.append((False, pred.operand.tag))
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                result.extend(self._collect_type_atoms(c, var))
        elif isinstance(pred, Or):
            # For disjunction, positive tags are the union
            for d in pred.disjuncts:
                result.extend(self._collect_type_atoms(d, var))
        return result


# ---------------------------------------------------------------------------
# PredicateTemplate — predicate templates for contract discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateTemplate:
    """A predicate template with holes."""
    kind: str  # 'comparison', 'isinstance', 'is_none', 'hasattr', 'compound'
    variables: Tuple[str, ...]  # placeholder variable names
    description: str = ""

    def instantiate(self, bindings: Dict[str, Any]) -> Optional[Predicate]:
        """Instantiate the template with concrete values."""
        if self.kind == "comparison":
            var = bindings.get("var")
            op_str = bindings.get("op", "==")
            val = bindings.get("val", 0)
            if var is None:
                return None
            op_map = {
                "==": ComparisonOp.EQ, "!=": ComparisonOp.NE,
                "<": ComparisonOp.LT, "<=": ComparisonOp.LE,
                ">": ComparisonOp.GT, ">=": ComparisonOp.GE,
            }
            op = op_map.get(op_str, ComparisonOp.EQ)
            return Comparison(op, Var(var), Const(val))
        if self.kind == "isinstance":
            var = bindings.get("var")
            tag = bindings.get("tag")
            if var is None or tag is None:
                return None
            return IsInstance(var, tag)
        if self.kind == "is_none":
            var = bindings.get("var")
            if var is None:
                return None
            return IsNone(var)
        if self.kind == "hasattr":
            var = bindings.get("var")
            key = bindings.get("key")
            if var is None or key is None:
                return None
            return HasAttr(var, key)
        return None


class PredicateTemplateLibrary:
    """Library of predicate templates for contract discovery refinement."""

    def __init__(self) -> None:
        self._templates: List[PredicateTemplate] = self._default_templates()

    def _default_templates(self) -> List[PredicateTemplate]:
        return [
            PredicateTemplate("comparison", ("var",), "x op c"),
            PredicateTemplate("comparison", ("var1", "var2"), "x op y"),
            PredicateTemplate("isinstance", ("var", "tag"), "isinstance(x, T)"),
            PredicateTemplate("is_none", ("var",), "is_none(x)"),
            PredicateTemplate("hasattr", ("var", "key"), "hasattr(x, k)"),
        ]

    def add_template(self, template: PredicateTemplate) -> None:
        self._templates.append(template)

    def get_templates(self) -> List[PredicateTemplate]:
        return list(self._templates)

    def instantiate_all(
        self,
        variables: List[str],
        constants: List[int],
        tags: List[str],
        attrs: List[str],
    ) -> List[Predicate]:
        """Instantiate all templates with the given program elements."""
        result: List[Predicate] = []
        ops = ["==", "!=", "<", "<=", ">", ">="]

        for var in variables:
            # Comparison with constants
            for c in constants:
                for op in ops:
                    pred = PredicateTemplate("comparison", ("var",)).instantiate(
                        {"var": var, "op": op, "val": c}
                    )
                    if pred is not None:
                        result.append(pred)

            # isinstance
            for tag in tags:
                pred = PredicateTemplate("isinstance", ("var", "tag")).instantiate(
                    {"var": var, "tag": tag}
                )
                if pred is not None:
                    result.append(pred)

            # is_none
            pred = PredicateTemplate("is_none", ("var",)).instantiate(
                {"var": var}
            )
            if pred is not None:
                result.append(pred)

            # hasattr
            for attr in attrs:
                pred = PredicateTemplate("hasattr", ("var", "key")).instantiate(
                    {"var": var, "key": attr}
                )
                if pred is not None:
                    result.append(pred)

        # Variable-variable comparisons
        for i, v1 in enumerate(variables):
            for v2 in variables[i + 1:]:
                for op in ["==", "<", "<="]:
                    result.append(
                        Comparison(
                            {"==": ComparisonOp.EQ, "<": ComparisonOp.LT,
                             "<=": ComparisonOp.LE}[op],
                            Var(v1), Var(v2),
                        )
                    )
        return result

    def enumerate_predicates(
        self,
        variables: List[str],
        constants: List[int],
        tags: List[str],
        attrs: List[str],
        *,
        max_predicates: int = 1000,
    ) -> List[Predicate]:
        """Enumerate predicates up to a limit."""
        all_preds = self.instantiate_all(variables, constants, tags, attrs)
        return all_preds[:max_predicates]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def evaluate_predicate(pred: Predicate, env: Dict[str, Any]) -> bool:
    """Top-level convenience: evaluate a predicate."""
    return PredicateEvaluator().evaluate(pred, env)


def simplify_predicate(pred: Predicate) -> Predicate:
    """Top-level convenience: simplify a predicate."""
    return PredicateNormalizer().simplify(pred)


def normalize_predicate(pred: Predicate) -> Predicate:
    """Top-level convenience: full normalization."""
    return PredicateNormalizer().normalize(pred)


def implies_predicate(p1: Predicate, p2: Predicate) -> bool:
    """Top-level convenience: check implication."""
    return PredicateImplication().implies(p1, p2)


def print_predicate(pred: Predicate, *, fmt: str = "string") -> str:
    """Top-level convenience: print a predicate in a given format."""
    printer = PredicatePrinter()
    if fmt == "smt":
        return printer.to_smt_lib(pred)
    if fmt == "python":
        return printer.to_python(pred)
    if fmt == "latex":
        return printer.to_latex(pred)
    if fmt == "z3":
        return printer.to_z3_api(pred)
    return printer.to_string(pred)
