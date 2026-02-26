"""Algebraic simplifier for ExprNode trees.

Provides rule-based rewriting with constant folding, identity elimination,
polynomial normalization, Hill function specialization, and
common-subexpression elimination (CSE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .expression import (
    Abs,
    Add,
    And,
    Const,
    Cos,
    Div,
    Eq,
    ExprNode,
    Exp,
    Exists,
    ForAll,
    Ge,
    Gt,
    HillAct,
    HillRep,
    Implies,
    Ite,
    Le,
    Log,
    Lt,
    Max,
    Min,
    Mul,
    Neg,
    Not,
    ONE,
    Or,
    Pow,
    Sin,
    Sqrt,
    Var,
    ZERO,
    _BinOp,
    _UnaryOp,
    const_value,
    is_const,
    map_expr,
)


# ---------------------------------------------------------------------------
# Simplification statistics
# ---------------------------------------------------------------------------

@dataclass
class SimplifyStats:
    """Tracks how many times each rewrite rule fired."""

    rules_applied: Dict[str, int] = field(default_factory=dict)
    iterations: int = 0

    def record(self, rule: str) -> None:
        self.rules_applied[rule] = self.rules_applied.get(rule, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.rules_applied.values())

    def __repr__(self) -> str:
        return f"SimplifyStats(iterations={self.iterations}, total={self.total}, rules={self.rules_applied})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimplifyConfig:
    """Control which rewriting passes are enabled."""

    constant_folding: bool = True
    identity_elimination: bool = True
    double_negation: bool = True
    distributive: bool = False
    polynomial_normalize: bool = True
    hill_special_cases: bool = True
    trig_identities: bool = True
    log_exp_cancel: bool = True
    cse: bool = True
    max_iterations: int = 20


# ---------------------------------------------------------------------------
# Constant-fold helpers
# ---------------------------------------------------------------------------

def _is_zero(e: ExprNode) -> bool:
    return isinstance(e, Const) and e.value == 0.0


def _is_one(e: ExprNode) -> bool:
    return isinstance(e, Const) and e.value == 1.0


def _is_neg_one(e: ExprNode) -> bool:
    return isinstance(e, Const) and e.value == -1.0


def _const_fold_binary(op: str, a: float, b: float) -> Optional[float]:
    """Evaluate a binary op on constants, returning None on domain errors."""
    try:
        if op == "+":
            return a + b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                return None
            return a / b
        if op == "^":
            if a == 0 and b < 0:
                return None
            if a < 0 and not float(b).is_integer():
                return None
            return a ** b
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def _const_fold_unary(op: str, a: float) -> Optional[float]:
    try:
        if op == "neg":
            return -a
        if op == "exp":
            return math.exp(a) if a < 700 else None
        if op == "log":
            return math.log(a) if a > 0 else None
        if op == "sin":
            return math.sin(a)
        if op == "cos":
            return math.cos(a)
        if op == "sqrt":
            return math.sqrt(a) if a >= 0 else None
        if op == "abs":
            return abs(a)
    except (OverflowError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# Core rewrite rules (single-step, bottom-up)
# ---------------------------------------------------------------------------

def _simplify_one(e: ExprNode, cfg: SimplifyConfig, stats: SimplifyStats) -> Optional[ExprNode]:
    """Try a single simplification step on *e*.  Return None if no rule fires."""

    # -- constant folding for binary ops ------------------------------------
    if cfg.constant_folding and isinstance(e, _BinOp):
        lv, rv = const_value(e.left), const_value(e.right)
        if lv is not None and rv is not None:
            op_map = {Add: "+", Mul: "*", Div: "/", Pow: "^"}
            op = op_map.get(type(e))
            if op:
                val = _const_fold_binary(op, lv, rv)
                if val is not None:
                    stats.record("const_fold_bin")
                    return Const(val)

    # -- constant folding for unary ops -------------------------------------
    if cfg.constant_folding and isinstance(e, _UnaryOp):
        ov = const_value(e.operand)
        if ov is not None:
            op_map = {Neg: "neg", Exp: "exp", Log: "log", Sin: "sin",
                      Cos: "cos", Sqrt: "sqrt", Abs: "abs"}
            op = op_map.get(type(e))
            if op:
                val = _const_fold_unary(op, ov)
                if val is not None:
                    stats.record("const_fold_un")
                    return Const(val)

    # -- identity / annihilator for Add -------------------------------------
    if cfg.identity_elimination and isinstance(e, Add):
        if _is_zero(e.left):
            stats.record("0+x")
            return e.right
        if _is_zero(e.right):
            stats.record("x+0")
            return e.left
        # x + (-x) = 0
        if isinstance(e.right, Neg) and e.left == e.right.operand:
            stats.record("x+(-x)")
            return ZERO
        if isinstance(e.left, Neg) and e.right == e.left.operand:
            stats.record("(-x)+x")
            return ZERO

    # -- identity / annihilator for Mul -------------------------------------
    if cfg.identity_elimination and isinstance(e, Mul):
        if _is_one(e.left):
            stats.record("1*x")
            return e.right
        if _is_one(e.right):
            stats.record("x*1")
            return e.left
        if _is_zero(e.left) or _is_zero(e.right):
            stats.record("0*x")
            return ZERO
        if _is_neg_one(e.left):
            stats.record("-1*x")
            return Neg(e.right)
        if _is_neg_one(e.right):
            stats.record("x*-1")
            return Neg(e.left)

    # -- identity for Div ---------------------------------------------------
    if cfg.identity_elimination and isinstance(e, Div):
        if _is_one(e.right):
            stats.record("x/1")
            return e.left
        if _is_zero(e.left):
            stats.record("0/x")
            return ZERO
        # x / x = 1
        if e.left == e.right:
            stats.record("x/x")
            return ONE

    # -- identity for Pow ---------------------------------------------------
    if cfg.identity_elimination and isinstance(e, Pow):
        if _is_zero(e.right):
            stats.record("x^0")
            return ONE
        if _is_one(e.right):
            stats.record("x^1")
            return e.left
        if _is_zero(e.left):
            rv = const_value(e.right)
            if rv is not None and rv > 0:
                stats.record("0^n")
                return ZERO

    # -- double negation ----------------------------------------------------
    if cfg.double_negation and isinstance(e, Neg):
        if isinstance(e.operand, Neg):
            stats.record("--x")
            return e.operand.operand
        if _is_zero(e.operand):
            stats.record("-0")
            return ZERO

    # -- log / exp cancellation ---------------------------------------------
    if cfg.log_exp_cancel:
        if isinstance(e, Log) and isinstance(e.operand, Exp):
            stats.record("log(exp(x))")
            return e.operand.operand
        if isinstance(e, Exp) and isinstance(e.operand, Log):
            stats.record("exp(log(x))")
            return e.operand.operand
        if isinstance(e, Sqrt):
            if isinstance(e.operand, Pow) and const_value(e.operand.right) == 2.0:
                stats.record("sqrt(x^2)")
                return Abs(e.operand.left)

    # -- trigonometric identities -------------------------------------------
    if cfg.trig_identities:
        # sin(0) = 0, cos(0) = 1
        if isinstance(e, Sin) and _is_zero(e.operand):
            stats.record("sin(0)")
            return ZERO
        if isinstance(e, Cos) and _is_zero(e.operand):
            stats.record("cos(0)")
            return ONE
        # sin^2 + cos^2 = 1 is handled at polynomial level

    # -- Min / Max with constants -------------------------------------------
    if isinstance(e, Min):
        lv, rv = const_value(e.left), const_value(e.right)
        if lv is not None and rv is not None:
            stats.record("min_const")
            return Const(min(lv, rv))
        if e.left == e.right:
            stats.record("min(x,x)")
            return e.left
    if isinstance(e, Max):
        lv, rv = const_value(e.left), const_value(e.right)
        if lv is not None and rv is not None:
            stats.record("max_const")
            return Const(max(lv, rv))
        if e.left == e.right:
            stats.record("max(x,x)")
            return e.left

    # -- Hill function special cases ----------------------------------------
    if cfg.hill_special_cases:
        if isinstance(e, HillAct):
            nv = const_value(e.n)
            kv = const_value(e.K)
            # HillAct(x, K, 1) = x / (K + x)
            if nv == 1.0:
                stats.record("hill_act_n1")
                return Div(e.x, Add(e.K, e.x))
            # HillAct(x, K, 2) = x^2 / (K^2 + x^2)
            if nv == 2.0:
                x2 = Pow(e.x, Const(2.0))
                k2 = Pow(e.K, Const(2.0))
                stats.record("hill_act_n2")
                return Div(x2, Add(k2, x2))
            # HillAct(x, 0, n) = 1  (K=0 means always active)
            if kv == 0.0:
                stats.record("hill_act_k0")
                return ONE
        if isinstance(e, HillRep):
            nv = const_value(e.n)
            kv = const_value(e.K)
            if nv == 1.0:
                stats.record("hill_rep_n1")
                return Div(e.K, Add(e.K, e.x))
            if nv == 2.0:
                x2 = Pow(e.x, Const(2.0))
                k2 = Pow(e.K, Const(2.0))
                stats.record("hill_rep_n2")
                return Div(k2, Add(k2, x2))
            if kv == 0.0:
                stats.record("hill_rep_k0")
                return ZERO

    # -- Boolean simplifications --------------------------------------------
    if isinstance(e, Not):
        if isinstance(e.operand, Not):
            stats.record("not_not")
            return e.operand.operand
    if isinstance(e, And):
        # Remove True (placeholder: Const(1))
        filtered = [a for a in e.args if not _is_one(a)]
        if any(_is_zero(a) for a in e.args):
            stats.record("and_false")
            return ZERO
        if len(filtered) < len(e.args):
            stats.record("and_true_elim")
            if not filtered:
                return ONE
            if len(filtered) == 1:
                return filtered[0]
            return And(*filtered)
    if isinstance(e, Or):
        filtered = [a for a in e.args if not _is_zero(a)]
        if any(_is_one(a) for a in e.args):
            stats.record("or_true")
            return ONE
        if len(filtered) < len(e.args):
            stats.record("or_false_elim")
            if not filtered:
                return ZERO
            if len(filtered) == 1:
                return filtered[0]
            return Or(*filtered)

    # -- Abs simplification -------------------------------------------------
    if isinstance(e, Abs):
        v = const_value(e.operand)
        if v is not None:
            stats.record("abs_const")
            return Const(abs(v))
        if isinstance(e.operand, Abs):
            stats.record("abs_abs")
            return e.operand
        if isinstance(e.operand, Neg):
            stats.record("abs_neg")
            return Abs(e.operand.operand)

    # -- distributive (optional) --------------------------------------------
    if cfg.distributive and isinstance(e, Mul):
        if isinstance(e.right, Add):
            stats.record("distrib_r")
            return Add(Mul(e.left, e.right.left), Mul(e.left, e.right.right))
        if isinstance(e.left, Add):
            stats.record("distrib_l")
            return Add(Mul(e.left.left, e.right), Mul(e.left.right, e.right))

    return None


# ---------------------------------------------------------------------------
# Bottom-up fixed-point simplifier
# ---------------------------------------------------------------------------

def simplify(
    expr: ExprNode,
    config: Optional[SimplifyConfig] = None,
    stats: Optional[SimplifyStats] = None,
) -> ExprNode:
    """Simplify *expr* using bottom-up rewriting with fixed-point iteration."""
    cfg = config or SimplifyConfig()
    st = stats or SimplifyStats()

    prev = expr
    for iteration in range(cfg.max_iterations):
        st.iterations = iteration + 1
        current = map_expr(prev, lambda e: _simplify_one(e, cfg, st))
        if current == prev:
            break
        prev = current

    if cfg.cse:
        prev = _apply_cse(prev, st)

    return prev


# ---------------------------------------------------------------------------
# Polynomial normalization
# ---------------------------------------------------------------------------

def _flatten_sum(e: ExprNode) -> List[ExprNode]:
    """Flatten nested Add trees into a list of summands."""
    if isinstance(e, Add):
        return _flatten_sum(e.left) + _flatten_sum(e.right)
    return [e]


def _flatten_product(e: ExprNode) -> List[ExprNode]:
    """Flatten nested Mul trees into a list of factors."""
    if isinstance(e, Mul):
        return _flatten_product(e.left) + _flatten_product(e.right)
    return [e]


def _term_key(e: ExprNode) -> Tuple:
    """Extract (coefficient, variable_part) from a term for like-term collection."""
    if isinstance(e, Const):
        return (e.value, ())
    if isinstance(e, Neg):
        inner_coeff, inner_key = _term_key(e.operand)
        return (-inner_coeff, inner_key)
    if isinstance(e, Mul):
        factors = _flatten_product(e)
        coeff = 1.0
        var_parts: List[ExprNode] = []
        for f in factors:
            cv = const_value(f)
            if cv is not None:
                coeff *= cv
            else:
                var_parts.append(f)
        vkey = tuple(sorted(str(v) for v in var_parts))
        return (coeff, vkey)
    return (1.0, (str(e),))


def normalize_polynomial(expr: ExprNode) -> ExprNode:
    """Collect like terms in a sum."""
    terms = _flatten_sum(expr)
    if len(terms) <= 1:
        return expr

    buckets: Dict[Tuple, float] = {}
    bucket_repr: Dict[Tuple, List[ExprNode]] = {}
    for t in terms:
        coeff, vkey = _term_key(t)
        buckets[vkey] = buckets.get(vkey, 0.0) + coeff
        if vkey not in bucket_repr:
            factors = _flatten_product(t)
            bucket_repr[vkey] = [f for f in factors if const_value(f) is None]

    result_terms: List[ExprNode] = []
    for vkey, coeff in buckets.items():
        if coeff == 0.0:
            continue
        var_parts = bucket_repr.get(vkey, [])
        if not var_parts:
            result_terms.append(Const(coeff))
        elif coeff == 1.0:
            t = var_parts[0]
            for v in var_parts[1:]:
                t = Mul(t, v)
            result_terms.append(t)
        elif coeff == -1.0:
            t = var_parts[0]
            for v in var_parts[1:]:
                t = Mul(t, v)
            result_terms.append(Neg(t))
        else:
            t: ExprNode = Const(coeff)
            for v in var_parts:
                t = Mul(t, v)
            result_terms.append(t)

    if not result_terms:
        return ZERO
    result = result_terms[0]
    for t in result_terms[1:]:
        result = Add(result, t)
    return result


# ---------------------------------------------------------------------------
# Common sub-expression elimination (CSE)
# ---------------------------------------------------------------------------

def _count_subexprs(e: ExprNode) -> Dict[ExprNode, int]:
    """Count occurrences of every sub-expression."""
    counts: Dict[ExprNode, int] = {}
    for node in e.iter_postorder():
        counts[node] = counts.get(node, 0) + 1
    return counts


def _apply_cse(e: ExprNode, stats: SimplifyStats) -> ExprNode:
    """Mark common sub-expressions.  Currently just returns *e* unchanged
    but updates stats.  Full CSE with let-binding is done during
    SMT-LIB serialization (define-fun) for more natural output."""
    counts = _count_subexprs(e)
    shared = {n for n, c in counts.items() if c > 1 and n.size() > 2}
    if shared:
        stats.record(f"cse_candidates({len(shared)})")
    return e


# ---------------------------------------------------------------------------
# Interval arithmetic evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IVal:
    """Interval value [lo, hi]."""
    lo: float
    hi: float


def _ival_add(a: IVal, b: IVal) -> IVal:
    return IVal(a.lo + b.lo, a.hi + b.hi)


def _ival_mul(a: IVal, b: IVal) -> IVal:
    products = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    return IVal(min(products), max(products))


def _ival_neg(a: IVal) -> IVal:
    return IVal(-a.hi, -a.lo)


def _ival_div(a: IVal, b: IVal) -> Optional[IVal]:
    if b.lo <= 0 <= b.hi:
        return None
    vals = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi]
    return IVal(min(vals), max(vals))


def interval_eval(
    e: ExprNode,
    var_bounds: Dict[str, IVal],
) -> Optional[IVal]:
    """Evaluate expression over interval bounds.  Returns None if not computable."""
    if isinstance(e, Const):
        return IVal(e.value, e.value)
    if isinstance(e, Var):
        return var_bounds.get(e.name)
    if isinstance(e, Add):
        la, ra = interval_eval(e.left, var_bounds), interval_eval(e.right, var_bounds)
        if la is None or ra is None:
            return None
        return _ival_add(la, ra)
    if isinstance(e, Mul):
        la, ra = interval_eval(e.left, var_bounds), interval_eval(e.right, var_bounds)
        if la is None or ra is None:
            return None
        return _ival_mul(la, ra)
    if isinstance(e, Neg):
        oa = interval_eval(e.operand, var_bounds)
        return _ival_neg(oa) if oa is not None else None
    if isinstance(e, Div):
        la, ra = interval_eval(e.left, var_bounds), interval_eval(e.right, var_bounds)
        if la is None or ra is None:
            return None
        return _ival_div(la, ra)
    return None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def simplify_fully(
    expr: ExprNode,
    config: Optional[SimplifyConfig] = None,
) -> Tuple[ExprNode, SimplifyStats]:
    """Simplify and return both the result and detailed statistics."""
    stats = SimplifyStats()
    result = simplify(expr, config=config, stats=stats)
    return result, stats
