"""
Refined numeric types beyond simple intervals.

Models arithmetic refinement propagation for patterns like:
    {x: int | x >= 0} + {y: int | y >= 0}  →  {r: int | r >= 0}
    {x: int | x > 0}  * {y: int | y > 0}   →  {r: int | r > 0}
    abs({x: int | -5 <= x <= 3})            →  {r: int | 0 <= r <= 5}

Also provides common refined type constructors (nat, pos_int, probability, etc.)
and safety checks for division-by-zero and overflow.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.refinement_lattice import (
    BaseTypeKind,
    BaseTypeR,
    Pred,
    PredOp,
    RefType,
    FLOAT_TYPE,
    INT_TYPE,
    ANY_TYPE,
    NEVER_TYPE,
)

# ---------------------------------------------------------------------------
# Supporting enums and dataclasses
# ---------------------------------------------------------------------------

class SignInfo(Enum):
    """Sign abstraction for a numeric value."""
    POSITIVE = auto()
    NEGATIVE = auto()
    NON_NEGATIVE = auto()
    NON_POSITIVE = auto()
    ZERO = auto()
    UNKNOWN = auto()


class ParityInfo(Enum):
    """Parity abstraction for an integer value."""
    EVEN = auto()
    ODD = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class NumericBounds:
    """Concrete interval bounds extracted from a refinement predicate."""
    lo: Optional[int] = None
    hi: Optional[int] = None
    lo_inclusive: bool = True
    hi_inclusive: bool = True
    is_integer: bool = True

    @property
    def is_bounded(self) -> bool:
        return self.lo is not None and self.hi is not None

    def sign(self) -> SignInfo:
        if self.lo is not None and self.hi is not None and self.lo == 0 and self.hi == 0:
            return SignInfo.ZERO
        if self.lo is not None:
            if self.lo > 0 or (self.lo == 0 and not self.lo_inclusive):
                return SignInfo.POSITIVE
            if self.lo >= 0:
                return SignInfo.NON_NEGATIVE
        if self.hi is not None:
            if self.hi < 0 or (self.hi == 0 and not self.hi_inclusive):
                return SignInfo.NEGATIVE
            if self.hi <= 0:
                return SignInfo.NON_POSITIVE
        return SignInfo.UNKNOWN

    def width(self) -> Optional[int]:
        if self.lo is not None and self.hi is not None:
            return self.hi - self.lo
        return None


@dataclass(frozen=True)
class DivisionSafety:
    """Result of a division safety check."""
    is_safe: bool
    reason: Optional[str] = None
    suggested_guard: Optional[str] = None


# ---------------------------------------------------------------------------
# Common refined type constructors
# ---------------------------------------------------------------------------

_NU = "ν"


def nat_type() -> RefType:
    """{x: int | x >= 0}"""
    return RefType(_NU, INT_TYPE, Pred.var_ge(_NU, 0))


def pos_int_type() -> RefType:
    """{x: int | x > 0}"""
    return RefType(_NU, INT_TYPE, Pred.var_gt(_NU, 0))


def probability_type() -> RefType:
    """{x: float | 0.0 <= x <= 1.0}"""
    return RefType(_NU, FLOAT_TYPE, Pred.in_range(_NU, 0, 1))


def non_nan_float() -> RefType:
    """{x: float | not isnan(x)} — modelled as x == x."""
    return RefType(_NU, FLOAT_TYPE, Pred.var_eq(_NU, _NU))


def even_int() -> RefType:
    """{x: int | x % 2 == 0}"""
    return RefType(_NU, INT_TYPE, Pred.divisible(_NU, 2))


def bounded_int(lo: int, hi: int) -> RefType:
    """{x: int | lo <= x <= hi}"""
    return RefType(_NU, INT_TYPE, Pred.in_range(_NU, lo, hi))


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class NumericRefinementAnalyzer:
    """Propagates numeric refinements through arithmetic operations.

    Given refined operand types, computes the refined result type by
    combining interval bounds, sign information, and parity.
    """

    def __init__(self) -> None:
        self._binder = _NU

    # ---- public API -------------------------------------------------------

    def analyze_numeric_op(
        self, op: str, left: RefType, right: RefType,
    ) -> RefType:
        """Infer a refined type for ``left <op> right``.

        Supports +, -, *, //, **, %, &, |, ^, <<, >>.
        """
        result_base = self._result_type(op, left.base, right.base)
        lb = self._extract_bounds(left.pred, left.binder)
        rb = self._extract_bounds(right.pred, right.binder)

        pred = self._add_bounds(left.pred, right.pred, op)

        # sign propagation
        ls = self._sign_of(left.pred, left.binder)
        rs = self._sign_of(right.pred, right.binder)
        sign = self._propagate_sign(op, ls, rs)
        pred = self._apply_sign(pred, sign)

        # parity propagation (integers only)
        if result_base.kind == BaseTypeKind.INT:
            lp = self._parity_of(left.pred, left.binder)
            rp = self._parity_of(right.pred, right.binder)
            par = self._propagate_parity(op, lp, rp)
            if par == ParityInfo.EVEN:
                pred = pred.and_(Pred.divisible(self._binder, 2))

        return RefType(self._binder, result_base, pred)

    def analyze_division(
        self, dividend: RefType, divisor: RefType,
    ) -> Tuple[RefType, Optional[str]]:
        """Analyze ``dividend / divisor``, returning (result, warning)."""
        safe, reason = self.check_division_safety(divisor)
        warning = None if safe else reason

        result_base = FLOAT_TYPE  # true division always yields float
        pred = Pred.true_()

        lb = self._extract_bounds(dividend.pred, dividend.binder)
        rb = self._extract_bounds(divisor.pred, divisor.binder)
        if lb and rb and rb[0] is not None and rb[0] > 0:
            # dividend / positive-divisor preserves sign of dividend
            ls = self._sign_of(dividend.pred, dividend.binder)
            pred = self._apply_sign(pred, ls)

        return RefType(self._binder, result_base, pred), warning

    def analyze_modulo(self, val: RefType, mod: RefType) -> RefType:
        """Analyze ``val % mod``.

        If mod is provably positive with known bound n, result is in [0, n).
        """
        result_base = self._result_type("%", val.base, mod.base)
        rb = self._extract_bounds(mod.pred, mod.binder)
        if rb and rb[0] is not None and rb[0] > 0:
            hi = rb[1] if rb[1] is not None else rb[0]
            pred = Pred.var_ge(self._binder, 0).and_(
                Pred.var_lt(self._binder, hi),
            )
            return RefType(self._binder, result_base, pred)

        # At least we know result base type
        return RefType(self._binder, result_base, Pred.true_())

    def analyze_power(self, base: RefType, exp: RefType) -> RefType:
        """Analyze ``base ** exp``."""
        result_base = self._result_type("**", base.base, exp.base)
        bs = self._sign_of(base.pred, base.binder)
        eb = self._extract_bounds(exp.pred, exp.binder)

        pred = Pred.true_()
        # non-negative base => non-negative result
        if bs in (SignInfo.POSITIVE, SignInfo.NON_NEGATIVE):
            pred = Pred.var_ge(self._binder, 0)
        # even exponent => non-negative result
        if eb and eb[0] is not None and eb[0] == eb[1] and eb[0] % 2 == 0:
            pred = Pred.var_ge(self._binder, 0)
        # x**0 == 1
        if eb and eb[0] == 0 and eb[1] == 0:
            pred = Pred.var_eq(self._binder, 1)
        return RefType(self._binder, result_base, pred)

    def analyze_unary(self, op: str, operand: RefType) -> RefType:
        """Analyze unary ``op operand`` (-x, +x, ~x, abs)."""
        if op == "abs":
            return self._refine_abs(operand)

        result_base = operand.base
        ob = self._extract_bounds(operand.pred, operand.binder)

        if op == "+":
            return RefType(self._binder, result_base, operand.pred.substitute(operand.binder, self._binder))

        if op == "-":
            if ob and ob[0] is not None and ob[1] is not None:
                new_lo, new_hi = -ob[1], -ob[0]
                pred = Pred.var_ge(self._binder, new_lo).and_(
                    Pred.var_le(self._binder, new_hi),
                )
                return RefType(self._binder, result_base, pred)
            # flip sign
            s = self._sign_of(operand.pred, operand.binder)
            flipped = {
                SignInfo.POSITIVE: SignInfo.NEGATIVE,
                SignInfo.NEGATIVE: SignInfo.POSITIVE,
                SignInfo.NON_NEGATIVE: SignInfo.NON_POSITIVE,
                SignInfo.NON_POSITIVE: SignInfo.NON_NEGATIVE,
                SignInfo.ZERO: SignInfo.ZERO,
            }.get(s, SignInfo.UNKNOWN)
            return RefType(self._binder, result_base, self._apply_sign(Pred.true_(), flipped))

        if op == "~":
            # ~x == -(x+1)
            if ob and ob[0] is not None and ob[1] is not None:
                new_lo, new_hi = -(ob[1] + 1), -(ob[0] + 1)
                pred = Pred.var_ge(self._binder, new_lo).and_(
                    Pred.var_le(self._binder, new_hi),
                )
                return RefType(self._binder, INT_TYPE, pred)
            return RefType(self._binder, INT_TYPE, Pred.true_())

        return RefType(self._binder, result_base, Pred.true_())

    def analyze_builtin_numeric(
        self, func: str, args: List[RefType],
    ) -> RefType:
        """Analyze built-in numeric functions (abs, min, max, sum, len, round)."""
        if func == "abs" and len(args) == 1:
            return self._refine_abs(args[0])

        if func in ("min", "max") and args:
            return self._refine_min_max(func, args)

        if func == "sum":
            return RefType(self._binder, INT_TYPE, Pred.true_())

        if func == "len":
            return RefType(self._binder, INT_TYPE, Pred.var_ge(self._binder, 0))

        if func == "round" and len(args) >= 1:
            return RefType(self._binder, INT_TYPE, Pred.true_())

        return RefType(self._binder, ANY_TYPE, Pred.true_())

    def analyze_math_function(
        self, func: str, args: List[RefType],
    ) -> RefType:
        """Analyze functions from the ``math`` module."""
        if func == "sqrt":
            return RefType(self._binder, FLOAT_TYPE, Pred.var_ge(self._binder, 0))

        if func == "log":
            # log returns float, may be negative
            return RefType(self._binder, FLOAT_TYPE, Pred.true_())

        if func in ("sin", "cos"):
            return RefType(
                self._binder, FLOAT_TYPE,
                Pred.in_range(self._binder, -1, 1),
            )

        if func == "exp":
            return RefType(self._binder, FLOAT_TYPE, Pred.var_gt(self._binder, 0))

        if func in ("fabs", "hypot"):
            return RefType(self._binder, FLOAT_TYPE, Pred.var_ge(self._binder, 0))

        if func == "floor" or func == "ceil" or func == "trunc":
            return RefType(self._binder, INT_TYPE, Pred.true_())

        if func == "factorial":
            return RefType(self._binder, INT_TYPE, Pred.var_ge(self._binder, 1))

        if func == "gcd":
            return RefType(self._binder, INT_TYPE, Pred.var_ge(self._binder, 0))

        if func in ("pi", "e", "tau", "inf"):
            return RefType(self._binder, FLOAT_TYPE, Pred.var_gt(self._binder, 0))

        return RefType(self._binder, FLOAT_TYPE, Pred.true_())

    def analyze_comparison_chain(
        self, comparisons: List[Tuple[str, RefType, RefType]],
    ) -> Pred:
        """Combine a chained comparison like ``0 < x < 10`` into a predicate.

        Each element is ``(op_str, left_type, right_type)``.
        """
        result = Pred.true_()
        for op_str, left, right in comparisons:
            # Extract the variable being constrained and the constant bound.
            lvar = left.binder
            rvar = right.binder
            lb = self._extract_bounds(left.pred, lvar)
            rb = self._extract_bounds(right.pred, rvar)

            # If right is a concrete constant, constrain left variable.
            if rb and rb[0] is not None and rb[0] == rb[1]:
                val = rb[0]
                p = Pred.var_cmp(lvar, op_str, val)
                result = result.and_(p)
            elif lb and lb[0] is not None and lb[0] == lb[1]:
                inv = {"<": ">", "<=": ">=", ">": "<", ">=": "<=",
                       "==": "==", "!=": "!="}.get(op_str, op_str)
                val = lb[0]
                p = Pred.var_cmp(rvar, inv, val)
                result = result.and_(p)
            else:
                # Cannot determine concrete constraint
                pass
        return result

    def check_division_safety(
        self, divisor: RefType,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether *divisor* is provably non-zero."""
        bounds = self._extract_bounds(divisor.pred, divisor.binder)
        if bounds:
            lo, hi = bounds
            if lo is not None and lo > 0:
                return True, None
            if hi is not None and hi < 0:
                return True, None
            if lo is not None and hi is not None and lo == 0 and hi == 0:
                return False, "divisor is provably zero"

        # Check for explicit neq-zero predicate
        if self._has_neq_zero(divisor.pred, divisor.binder):
            return True, None

        return False, "cannot prove divisor ≠ 0; consider adding a guard"

    def check_overflow_risk(
        self, op: str, left: RefType, right: RefType,
    ) -> Optional[str]:
        """Warn when an operation may produce very large values."""
        lb = self._extract_bounds(left.pred, left.binder)
        rb = self._extract_bounds(right.pred, right.binder)

        if op == "**" and rb:
            _, rhi = rb
            if rhi is not None and rhi > 64:
                return f"exponent may be as large as {rhi}; risk of overflow"

        if op == "*" and lb and rb:
            lhi = lb[1]
            rhi = rb[1]
            if lhi is not None and rhi is not None:
                product = abs(lhi) * abs(rhi)
                if product > 2**63:
                    return f"product bound ~{product} exceeds 64-bit range"

        if op == "<<" and rb:
            _, rhi = rb
            if rhi is not None and rhi > 63:
                return f"left-shift by up to {rhi} bits; risk of overflow"

        return None

    # ---- private helpers ---------------------------------------------------

    def _add_bounds(
        self, left_pred: Pred, right_pred: Pred, op: str,
    ) -> Pred:
        """Combine interval bounds through an arithmetic operation."""
        lb = self._extract_bounds(left_pred, _NU)
        rb = self._extract_bounds(right_pred, _NU)
        if not lb or not rb:
            return Pred.true_()
        l_lo, l_hi = lb
        r_lo, r_hi = rb

        if op == "+":
            new_lo = (l_lo + r_lo) if l_lo is not None and r_lo is not None else None
            new_hi = (l_hi + r_hi) if l_hi is not None and r_hi is not None else None
        elif op == "-":
            new_lo = (l_lo - r_hi) if l_lo is not None and r_hi is not None else None
            new_hi = (l_hi - r_lo) if l_hi is not None and r_lo is not None else None
        elif op == "*":
            if all(v is not None for v in (l_lo, l_hi, r_lo, r_hi)):
                products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
                new_lo, new_hi = min(products), max(products)
            else:
                new_lo, new_hi = None, None
        elif op == "//":
            if r_lo is not None and r_lo > 0 and l_lo is not None:
                new_lo = l_lo // r_hi if r_hi else None
                new_hi = l_hi // r_lo if l_hi is not None else None
            else:
                new_lo, new_hi = None, None
        else:
            return Pred.true_()

        pred = Pred.true_()
        if new_lo is not None:
            pred = pred.and_(Pred.var_ge(self._binder, new_lo))
        if new_hi is not None:
            pred = pred.and_(Pred.var_le(self._binder, new_hi))
        return pred

    def _extract_bounds(
        self, pred: Pred, var: str,
    ) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Extract (lo, hi) bounds for *var* from a predicate, or None."""
        lo: Optional[int] = None
        hi: Optional[int] = None
        found = False

        for p in self._flatten_and(pred):
            if p.args and len(p.args) >= 2 and p.args[0] == var:
                val = p.args[1]
                if not isinstance(val, (int, float)):
                    continue
                val = int(val)
                if p.op == PredOp.GE:
                    lo = val if lo is None else max(lo, val)
                    found = True
                elif p.op == PredOp.GT:
                    lo = val + 1 if lo is None else max(lo, val + 1)
                    found = True
                elif p.op == PredOp.LE:
                    hi = val if hi is None else min(hi, val)
                    found = True
                elif p.op == PredOp.LT:
                    hi = val - 1 if hi is None else min(hi, val - 1)
                    found = True
                elif p.op == PredOp.EQ:
                    lo = val
                    hi = val
                    found = True
            if p.op == PredOp.IN_RANGE and p.args and p.args[0] == var:
                lo = int(p.args[1])
                hi = int(p.args[2])
                found = True
        return (lo, hi) if found else None

    def _flatten_and(self, pred: Pred) -> List[Pred]:
        """Flatten nested AND predicates into a list of conjuncts."""
        if pred.op == PredOp.AND:
            result: List[Pred] = []
            for c in pred.children:
                result.extend(self._flatten_and(c))
            return result
        if pred.op == PredOp.TRUE:
            return []
        return [pred]

    def _sign_of(self, pred: Pred, var: str) -> SignInfo:
        """Infer sign from a predicate."""
        bounds = self._extract_bounds(pred, var)
        if bounds:
            return NumericBounds(lo=bounds[0], hi=bounds[1]).sign()
        return SignInfo.UNKNOWN

    def _parity_of(self, pred: Pred, var: str) -> ParityInfo:
        """Check if predicate constrains parity via DIVISIBLE."""
        for p in self._flatten_and(pred):
            if p.op == PredOp.DIVISIBLE and p.args and p.args[0] == var:
                if int(p.args[1]) % 2 == 0:
                    return ParityInfo.EVEN
        # exact constant
        bounds = self._extract_bounds(pred, var)
        if bounds and bounds[0] is not None and bounds[0] == bounds[1]:
            return ParityInfo.EVEN if bounds[0] % 2 == 0 else ParityInfo.ODD
        return ParityInfo.UNKNOWN

    def _propagate_sign(
        self, op: str, left_sign: SignInfo, right_sign: SignInfo,
    ) -> SignInfo:
        """Determine result sign from operand signs."""
        if op in ("+", "-"):
            if left_sign == right_sign == SignInfo.POSITIVE:
                return SignInfo.POSITIVE if op == "+" else SignInfo.UNKNOWN
            if left_sign == right_sign == SignInfo.NON_NEGATIVE:
                return SignInfo.NON_NEGATIVE if op == "+" else SignInfo.UNKNOWN
            if left_sign == right_sign == SignInfo.NEGATIVE:
                return SignInfo.NEGATIVE if op == "+" else SignInfo.UNKNOWN
            return SignInfo.UNKNOWN

        if op == "*":
            _table: Dict[Tuple[SignInfo, SignInfo], SignInfo] = {
                (SignInfo.POSITIVE, SignInfo.POSITIVE): SignInfo.POSITIVE,
                (SignInfo.POSITIVE, SignInfo.NEGATIVE): SignInfo.NEGATIVE,
                (SignInfo.NEGATIVE, SignInfo.POSITIVE): SignInfo.NEGATIVE,
                (SignInfo.NEGATIVE, SignInfo.NEGATIVE): SignInfo.POSITIVE,
                (SignInfo.NON_NEGATIVE, SignInfo.NON_NEGATIVE): SignInfo.NON_NEGATIVE,
                (SignInfo.NON_NEGATIVE, SignInfo.NON_POSITIVE): SignInfo.NON_POSITIVE,
                (SignInfo.NON_POSITIVE, SignInfo.NON_NEGATIVE): SignInfo.NON_POSITIVE,
                (SignInfo.NON_POSITIVE, SignInfo.NON_POSITIVE): SignInfo.NON_NEGATIVE,
            }
            if (left_sign, right_sign) in _table:
                return _table[(left_sign, right_sign)]
            if left_sign == SignInfo.ZERO or right_sign == SignInfo.ZERO:
                return SignInfo.ZERO
            return SignInfo.UNKNOWN

        if op in ("//", "/"):
            # same rules as multiplication for sign
            return self._propagate_sign("*", left_sign, right_sign)

        return SignInfo.UNKNOWN

    def _propagate_parity(
        self, op: str, left_even: Optional[bool], right_even: Optional[bool],
    ) -> Optional[bool]:
        """Propagate parity (True=even, False=odd, None=unknown)."""
        if left_even is None or right_even is None:
            # multiplication: anything * even = even
            if op == "*":
                if left_even is True or right_even is True:
                    return True
            return None

        if op == "+":
            return left_even == right_even  # even+even=even, odd+odd=even
        if op == "-":
            return left_even == right_even
        if op == "*":
            return left_even or right_even  # at least one even => even
        return None

    def _result_type(
        self, op: str, left_base: BaseTypeR, right_base: BaseTypeR,
    ) -> BaseTypeR:
        """Determine result base type for an arithmetic operation."""
        if op == "/":
            return FLOAT_TYPE
        if op == "**":
            if right_base.kind == BaseTypeKind.FLOAT:
                return FLOAT_TYPE
            if left_base.kind == BaseTypeKind.FLOAT:
                return FLOAT_TYPE
            return INT_TYPE
        if left_base.kind == BaseTypeKind.FLOAT or right_base.kind == BaseTypeKind.FLOAT:
            return FLOAT_TYPE
        if op in ("&", "|", "^", "<<", ">>", "//", "%"):
            return INT_TYPE
        return INT_TYPE

    def _apply_sign(self, pred: Pred, sign: SignInfo) -> Pred:
        """Add a sign constraint to a predicate."""
        if sign == SignInfo.POSITIVE:
            return pred.and_(Pred.var_gt(self._binder, 0))
        if sign == SignInfo.NEGATIVE:
            return pred.and_(Pred.var_lt(self._binder, 0))
        if sign == SignInfo.NON_NEGATIVE:
            return pred.and_(Pred.var_ge(self._binder, 0))
        if sign == SignInfo.NON_POSITIVE:
            return pred.and_(Pred.var_le(self._binder, 0))
        if sign == SignInfo.ZERO:
            return pred.and_(Pred.var_eq(self._binder, 0))
        return pred

    def _has_neq_zero(self, pred: Pred, var: str) -> bool:
        """Check whether predicate contains ``var ≠ 0``."""
        for p in self._flatten_and(pred):
            if p.op == PredOp.NEQ and p.args and p.args[0] == var and p.args[1] == 0:
                return True
            if p.op == PredOp.GT and p.args and p.args[0] == var and p.args[1] == 0:
                return True
            if p.op == PredOp.LT and p.args and p.args[0] == var and p.args[1] == 0:
                return True
        return False

    def _refine_abs(self, operand: RefType) -> RefType:
        """abs({x: int | lo <= x <= hi}) → {r: int | 0 <= r <= max(|lo|, |hi|)}."""
        result_base = operand.base
        bounds = self._extract_bounds(operand.pred, operand.binder)
        if bounds:
            lo, hi = bounds
            candidates: List[int] = []
            if lo is not None:
                candidates.append(abs(lo))
            if hi is not None:
                candidates.append(abs(hi))

            new_lo = 0
            # if both bounds same sign and both present, tighten lower
            if lo is not None and hi is not None:
                if lo >= 0:
                    new_lo = lo
                elif hi <= 0:
                    new_lo = abs(hi)
                # else interval straddles zero, new_lo = 0

            new_hi = max(candidates) if candidates else None
            pred = Pred.var_ge(self._binder, new_lo)
            if new_hi is not None:
                pred = pred.and_(Pred.var_le(self._binder, new_hi))
            return RefType(self._binder, result_base, pred)

        # Fallback: result is non-negative
        return RefType(self._binder, result_base, Pred.var_ge(self._binder, 0))

    def _refine_min_max(self, func: str, args: List[RefType]) -> RefType:
        """Refine min/max over a list of refined types."""
        if not args:
            return RefType(self._binder, INT_TYPE, Pred.true_())

        base = args[0].base
        for a in args[1:]:
            if a.base.kind == BaseTypeKind.FLOAT:
                base = FLOAT_TYPE

        all_bounds = [self._extract_bounds(a.pred, a.binder) for a in args]

        if func == "min":
            # min result <= each argument's upper bound; >= smallest lower bound
            his = [b[1] for b in all_bounds if b and b[1] is not None]
            los = [b[0] for b in all_bounds if b and b[0] is not None]
            pred = Pred.true_()
            if his:
                pred = pred.and_(Pred.var_le(self._binder, min(his)))
            if los:
                pred = pred.and_(Pred.var_ge(self._binder, min(los)))
            return RefType(self._binder, base, pred)
        else:  # max
            his = [b[1] for b in all_bounds if b and b[1] is not None]
            los = [b[0] for b in all_bounds if b and b[0] is not None]
            pred = Pred.true_()
            if his:
                pred = pred.and_(Pred.var_le(self._binder, max(his)))
            if los:
                pred = pred.and_(Pred.var_ge(self._binder, max(los)))
            return RefType(self._binder, base, pred)
