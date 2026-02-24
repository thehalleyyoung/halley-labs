from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union
)
from enum import Enum, auto
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TypeTag(Enum):
    INT = auto()
    FLOAT = auto()
    COMPLEX = auto()
    BOOL = auto()
    DECIMAL = auto()
    FRACTION = auto()
    STR = auto()
    NONE = auto()
    ANY = auto()
    LIST = auto()
    TUPLE = auto()
    DICT = auto()
    SET = auto()
    BYTES = auto()
    CALLABLE = auto()
    NEVER = auto()


class NullityTag(Enum):
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


class RoundingMode(Enum):
    ROUND_HALF_UP = auto()
    ROUND_DOWN = auto()
    ROUND_CEILING = auto()
    ROUND_FLOOR = auto()
    ROUND_HALF_EVEN = auto()
    ROUND_05UP = auto()
    ROUND_HALF_DOWN = auto()


# ---------------------------------------------------------------------------
# Core data-classes
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """Numeric interval [lo, hi].  None means unbounded (−∞ / +∞)."""
    lo: Optional[float] = None
    hi: Optional[float] = None

    # -- predicates -----------------------------------------------------------

    def is_bottom(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False

    def is_top(self) -> bool:
        return self.lo is None and self.hi is None

    def contains(self, value: float) -> bool:
        if self.is_bottom():
            return False
        if self.lo is not None and value < self.lo:
            return False
        if self.hi is not None and value > self.hi:
            return False
        return True

    def contains_zero(self) -> bool:
        return self.contains(0.0)

    def is_positive(self) -> bool:
        return self.lo is not None and self.lo > 0

    def is_negative(self) -> bool:
        return self.hi is not None and self.hi < 0

    def is_non_negative(self) -> bool:
        return self.lo is not None and self.lo >= 0

    def is_non_positive(self) -> bool:
        return self.hi is not None and self.hi <= 0

    def is_singleton(self) -> bool:
        return (self.lo is not None and self.hi is not None
                and self.lo == self.hi)

    def width(self) -> Optional[float]:
        if self.lo is None or self.hi is None:
            return None
        return self.hi - self.lo

    # -- lattice operations ---------------------------------------------------

    def join(self, other: Interval) -> Interval:
        """Least upper bound (union hull)."""
        if self.is_bottom():
            return other
        if other.is_bottom():
            return Interval(self.lo, self.hi)
        lo = _min_opt(self.lo, other.lo)
        hi = _max_opt(self.hi, other.hi)
        return Interval(lo, hi)

    def meet(self, other: Interval) -> Interval:
        """Greatest lower bound (intersection)."""
        lo = _max_bound(self.lo, other.lo)
        hi = _min_bound(self.hi, other.hi)
        return Interval(lo, hi)

    def widen(self, other: Interval) -> Interval:
        """Standard widening: jump to ±∞ where *other* exceeds *self*."""
        if self.is_bottom():
            return other
        if other.is_bottom():
            return Interval(self.lo, self.hi)
        lo = self.lo if (other.lo is not None and self.lo is not None
                         and other.lo >= self.lo) else None
        hi = self.hi if (other.hi is not None and self.hi is not None
                         and other.hi <= self.hi) else None
        return Interval(lo, hi)

    def narrow(self, other: Interval) -> Interval:
        """Standard narrowing."""
        lo = other.lo if self.lo is None else self.lo
        hi = other.hi if self.hi is None else self.hi
        return Interval(lo, hi)

    # -- arithmetic helpers ---------------------------------------------------

    def __add__(self, other: Interval) -> Interval:
        lo = _add_opt(self.lo, other.lo)
        hi = _add_opt(self.hi, other.hi)
        return Interval(lo, hi)

    def __sub__(self, other: Interval) -> Interval:
        lo = _sub_opt(self.lo, other.hi)
        hi = _sub_opt(self.hi, other.lo)
        return Interval(lo, hi)

    def __neg__(self) -> Interval:
        new_lo = -self.hi if self.hi is not None else None
        new_hi = -self.lo if self.lo is not None else None
        return Interval(new_lo, new_hi)

    def __repr__(self) -> str:
        lo_s = "-∞" if self.lo is None else str(self.lo)
        hi_s = "+∞" if self.hi is None else str(self.hi)
        return f"[{lo_s}, {hi_s}]"


@dataclass
class Predicate:
    kind: str  # e.g. ">=", "!=", "is_int", "in_range", "divisible_by"
    args: List[Any] = field(default_factory=list)

    def evaluate(self, value: float) -> bool:
        if self.kind == ">=":
            return value >= self.args[0]
        if self.kind == "<=":
            return value <= self.args[0]
        if self.kind == ">":
            return value > self.args[0]
        if self.kind == "<":
            return value < self.args[0]
        if self.kind == "==":
            return value == self.args[0]
        if self.kind == "!=":
            return value != self.args[0]
        if self.kind == "in_range":
            return self.args[0] <= value <= self.args[1]
        if self.kind == "divisible_by":
            return value % self.args[0] == 0
        if self.kind == "is_int":
            return float(value).is_integer()
        if self.kind == "true":
            return True
        if self.kind == "false":
            return False
        return True

    def negate(self) -> Predicate:
        neg_map: Dict[str, str] = {
            ">=": "<", "<=": ">", ">": "<=", "<": ">=",
            "==": "!=", "!=": "==", "true": "false", "false": "true",
        }
        if self.kind in neg_map:
            return Predicate(neg_map[self.kind], list(self.args))
        return Predicate("not_" + self.kind, list(self.args))

    def __repr__(self) -> str:
        if self.args:
            return f"Pred({self.kind} {self.args})"
        return f"Pred({self.kind})"


@dataclass
class RefinementType:
    base_type: str
    predicate: Optional[Predicate] = None
    interval: Optional[Interval] = None
    nullity: NullityTag = NullityTag.DEFINITELY_NOT_NULL
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_subtype_of(self, other: RefinementType) -> bool:
        if other.base_type == "any":
            return True
        if self.base_type != other.base_type:
            if self.base_type == "bool" and other.base_type == "int":
                pass  # bool <: int
            elif self.base_type == "int" and other.base_type == "float":
                pass  # int <: float in Python numeric tower
            else:
                return False
        if self.interval and other.interval:
            if not _interval_subset(self.interval, other.interval):
                return False
        if self.predicate and other.predicate:
            pass  # conservative: assume OK
        return True

    def join(self, other: RefinementType) -> RefinementType:
        bt = self.base_type if self.base_type == other.base_type else "any"
        iv = None
        if self.interval and other.interval:
            iv = self.interval.join(other.interval)
        elif self.interval:
            iv = self.interval
        elif other.interval:
            iv = other.interval
        n = _join_nullity(self.nullity, other.nullity)
        return RefinementType(bt, None, iv, n)

    def meet(self, other: RefinementType) -> RefinementType:
        if self.base_type != other.base_type and self.base_type != "any" and other.base_type != "any":
            return RefinementType("never")
        bt = self.base_type if self.base_type != "any" else other.base_type
        iv = None
        if self.interval and other.interval:
            iv = self.interval.meet(other.interval)
        elif self.interval:
            iv = Interval(self.interval.lo, self.interval.hi)
        elif other.interval:
            iv = Interval(other.interval.lo, other.interval.hi)
        n = _meet_nullity(self.nullity, other.nullity)
        return RefinementType(bt, None, iv, n)

    def __repr__(self) -> str:
        parts = [self.base_type]
        if self.interval:
            parts.append(str(self.interval))
        if self.predicate:
            parts.append(str(self.predicate))
        if self.nullity != NullityTag.DEFINITELY_NOT_NULL:
            parts.append(self.nullity.name)
        return "{" + " & ".join(parts) + "}"


@dataclass
class NumericValue:
    interval: Interval = field(default_factory=Interval)
    type_tag: TypeTag = TypeTag.ANY
    special_values: Set[str] = field(default_factory=set)
    is_exact: bool = False

    def may_be_nan(self) -> bool:
        return "nan" in self.special_values

    def may_be_inf(self) -> bool:
        return "inf" in self.special_values or "-inf" in self.special_values


@dataclass
class FunctionSignature:
    name: str
    param_types: List[RefinementType] = field(default_factory=list)
    return_type: Optional[RefinementType] = None
    preconditions: List[Predicate] = field(default_factory=list)
    postconditions: List[Predicate] = field(default_factory=list)
    may_raise: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    is_pure: bool = True


@dataclass
class TransferResult:
    result_type: RefinementType = field(
        default_factory=lambda: RefinementType("any"))
    exceptions: List[str] = field(default_factory=list)
    narrowings: Dict[str, RefinementType] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern data-classes
# ---------------------------------------------------------------------------

@dataclass
class CounterPattern:
    variable: str
    init_value: float
    step: float
    direction: str  # "up" | "down"
    bound: Optional[float] = None

@dataclass
class AccumulatorPattern:
    variable: str
    init_value: float
    operation: str  # "add" | "mul"
    source: str
    invariant: Optional[Interval] = None

@dataclass
class MinMaxPattern:
    variable: str
    kind: str  # "min" | "max"
    source: str
    invariant: Optional[Interval] = None

@dataclass
class AveragePattern:
    sum_var: str
    count_var: str
    result_var: Optional[str] = None

@dataclass
class BoundsCheckPattern:
    variable: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    checked_before_use: bool = False

@dataclass
class DivisionGuardPattern:
    divisor_var: str
    guard_kind: str  # "!= 0" | "> 0" etc.
    division_op: str


# ---------------------------------------------------------------------------
# Helper functions (interval option arithmetic)
# ---------------------------------------------------------------------------

def _add_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a + b

def _sub_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b

def _mul_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        if a == 0 or b == 0:
            return 0.0
        return None
    return a * b

def _min_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return min(a, b)

def _max_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return max(a, b)

def _max_bound(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """For meet lower bound: take the tighter (larger) lower bound."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)

def _min_bound(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """For meet upper bound: take the tighter (smaller) upper bound."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def _interval_subset(a: Interval, b: Interval) -> bool:
    lo_ok = (b.lo is None) or (a.lo is not None and a.lo >= b.lo)
    hi_ok = (b.hi is None) or (a.hi is not None and a.hi <= b.hi)
    return lo_ok and hi_ok

def _join_nullity(a: NullityTag, b: NullityTag) -> NullityTag:
    if a == b:
        return a
    return NullityTag.MAYBE_NULL

def _meet_nullity(a: NullityTag, b: NullityTag) -> NullityTag:
    if a == NullityTag.DEFINITELY_NULL or b == NullityTag.DEFINITELY_NULL:
        return NullityTag.DEFINITELY_NULL
    if a == NullityTag.DEFINITELY_NOT_NULL and b == NullityTag.DEFINITELY_NOT_NULL:
        return NullityTag.DEFINITELY_NOT_NULL
    return NullityTag.MAYBE_NULL


def _safe_lo(iv: Optional[Interval]) -> Optional[float]:
    return iv.lo if iv else None

def _safe_hi(iv: Optional[Interval]) -> Optional[float]:
    return iv.hi if iv else None

def _ensure_interval(rt: RefinementType) -> Interval:
    return rt.interval if rt.interval else Interval()


def _make_int(iv: Optional[Interval] = None,
              pred: Optional[Predicate] = None) -> RefinementType:
    return RefinementType("int", pred, iv, NullityTag.DEFINITELY_NOT_NULL)


def _make_float(iv: Optional[Interval] = None,
                specials: Optional[Set[str]] = None) -> RefinementType:
    extra: Dict[str, Any] = {}
    if specials:
        extra["special_values"] = specials
    return RefinementType("float", None, iv, NullityTag.DEFINITELY_NOT_NULL, extra)


def _make_bool(value: Optional[bool] = None) -> RefinementType:
    iv = None
    if value is True:
        iv = Interval(1, 1)
    elif value is False:
        iv = Interval(0, 0)
    else:
        iv = Interval(0, 1)
    return RefinementType("bool", None, iv, NullityTag.DEFINITELY_NOT_NULL)


def _make_complex() -> RefinementType:
    return RefinementType("complex", None, None, NullityTag.DEFINITELY_NOT_NULL)


def _make_str() -> RefinementType:
    return RefinementType("str", None, None, NullityTag.DEFINITELY_NOT_NULL)


def _make_none() -> RefinementType:
    return RefinementType("NoneType", None, None, NullityTag.DEFINITELY_NULL)


def _make_list(elem: Optional[RefinementType] = None) -> RefinementType:
    extra: Dict[str, Any] = {}
    if elem:
        extra["element_type"] = elem
    return RefinementType("list", None, None, NullityTag.DEFINITELY_NOT_NULL, extra)


def _make_tuple(*elems: RefinementType) -> RefinementType:
    return RefinementType("tuple", None, None, NullityTag.DEFINITELY_NOT_NULL,
                          {"element_types": list(elems)})


# ---------------------------------------------------------------------------
# NumericIntervalArithmetic (standalone interval library)
# ---------------------------------------------------------------------------

class NumericIntervalArithmetic:
    """Full interval arithmetic library."""

    @staticmethod
    def add(a: Interval, b: Interval) -> Interval:
        return a + b

    @staticmethod
    def sub(a: Interval, b: Interval) -> Interval:
        return a - b

    @staticmethod
    def mul(a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval(1, -1)  # bottom
        candidates = [
            _mul_opt(a.lo, b.lo),
            _mul_opt(a.lo, b.hi),
            _mul_opt(a.hi, b.lo),
            _mul_opt(a.hi, b.hi),
        ]
        finites = [c for c in candidates if c is not None]
        nones = [c for c in candidates if c is None]
        if not finites and nones:
            return Interval(None, None)
        if nones:
            # Some products are infinite
            lo: Optional[float] = None
            hi: Optional[float] = None
            if finites:
                lo = min(finites)
                hi = max(finites)
            # If any product is None (unbounded), extend appropriately
            return Interval(lo if not any(
                NumericIntervalArithmetic._product_sign_neg(a, b, i)
                for i in range(4) if candidates[i] is None
            ) else None,
                hi if not any(
                NumericIntervalArithmetic._product_sign_pos(a, b, i)
                for i in range(4) if candidates[i] is None
            ) else None)
        return Interval(min(finites), max(finites))

    @staticmethod
    def _product_sign_neg(a: Interval, b: Interval, idx: int) -> bool:
        """Check if unbounded product at index could be -∞."""
        pairs = [(a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)]
        x, y = pairs[idx]
        # x * y where one is None
        if x is None and y is not None:
            return y > 0  # -∞ * positive = -∞
        if y is None and x is not None:
            return x > 0
        if x is None and y is None:
            return True  # could go either way
        return False

    @staticmethod
    def _product_sign_pos(a: Interval, b: Interval, idx: int) -> bool:
        pairs = [(a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)]
        x, y = pairs[idx]
        if x is None and y is not None:
            return y < 0  # -∞ * negative = +∞
        if y is None and x is not None:
            return x < 0
        if x is None and y is None:
            return True
        return False

    @staticmethod
    def div(a: Interval, b: Interval) -> Interval:
        """Division with zero handling."""
        if a.is_bottom() or b.is_bottom():
            return Interval(1, -1)
        if b.contains_zero():
            if b.is_singleton() and b.lo == 0:
                return Interval(1, -1)  # bottom — division by exactly zero
            neg_part, pos_part = NumericIntervalArithmetic.split_at_zero(b)
            results: List[Interval] = []
            if not neg_part.is_bottom():
                results.append(NumericIntervalArithmetic._div_no_zero(a, neg_part))
            if not pos_part.is_bottom():
                results.append(NumericIntervalArithmetic._div_no_zero(a, pos_part))
            if not results:
                return Interval(None, None)
            res = results[0]
            for r in results[1:]:
                res = res.join(r)
            return res
        return NumericIntervalArithmetic._div_no_zero(a, b)

    @staticmethod
    def _div_no_zero(a: Interval, b: Interval) -> Interval:
        """Division where b does not contain zero."""
        if b.lo is not None and b.lo > 0:
            inv_b = Interval(
                1.0 / b.hi if b.hi is not None else 0.0,
                1.0 / b.lo
            )
        elif b.hi is not None and b.hi < 0:
            inv_b = Interval(
                1.0 / b.hi,
                1.0 / b.lo if b.lo is not None else 0.0
            )
        else:
            return Interval(None, None)
        return NumericIntervalArithmetic.mul(a, inv_b)

    @staticmethod
    def mod(a: Interval, b: Interval) -> Interval:
        if b.is_bottom() or a.is_bottom():
            return Interval(1, -1)
        if b.contains_zero():
            return Interval(None, None)
        if b.is_positive():
            bmax = b.hi if b.hi is not None else None
            if bmax is not None:
                return Interval(0, bmax - 1)
            return Interval(0, None)
        if b.is_negative():
            bmin = b.lo if b.lo is not None else None
            if bmin is not None:
                return Interval(bmin + 1, 0)
            return Interval(None, 0)
        return Interval(None, None)

    @staticmethod
    def pow_interval(base: Interval, exp: Interval) -> Interval:
        if base.is_bottom() or exp.is_bottom():
            return Interval(1, -1)
        if exp.is_singleton() and exp.lo is not None:
            e = exp.lo
            if e == 0:
                return Interval(1, 1)
            if e == 1:
                return Interval(base.lo, base.hi)
            if e == 2:
                return NumericIntervalArithmetic._square(base)
            if e == int(e) and e > 0:
                return NumericIntervalArithmetic._int_pow(base, int(e))
        return Interval(None, None)

    @staticmethod
    def _square(a: Interval) -> Interval:
        if a.is_non_negative():
            lo = a.lo ** 2 if a.lo is not None else 0
            hi = a.hi ** 2 if a.hi is not None else None
            return Interval(lo, hi)
        if a.is_non_positive():
            lo = a.hi ** 2 if a.hi is not None else 0
            hi = a.lo ** 2 if a.lo is not None else None
            return Interval(lo, hi)
        # straddles zero
        candidates = []
        if a.lo is not None:
            candidates.append(a.lo ** 2)
        if a.hi is not None:
            candidates.append(a.hi ** 2)
        hi = max(candidates) if candidates else None
        return Interval(0, hi)

    @staticmethod
    def _int_pow(base: Interval, n: int) -> Interval:
        if n == 0:
            return Interval(1, 1)
        if n == 1:
            return Interval(base.lo, base.hi)
        if n % 2 == 0:
            half = NumericIntervalArithmetic._int_pow(base, n // 2)
            return NumericIntervalArithmetic._square(half)
        sub = NumericIntervalArithmetic._int_pow(base, n - 1)
        return NumericIntervalArithmetic.mul(base, sub)

    @staticmethod
    def floor_div(a: Interval, b: Interval) -> Interval:
        if b.contains_zero():
            return Interval(None, None)
        raw = NumericIntervalArithmetic.div(a, b)
        lo = math.floor(raw.lo) if raw.lo is not None else None
        hi = math.floor(raw.hi) if raw.hi is not None else None
        return Interval(lo, hi)

    @staticmethod
    def negate(a: Interval) -> Interval:
        return -a

    @staticmethod
    def abs_interval(a: Interval) -> Interval:
        if a.is_non_negative():
            return Interval(a.lo, a.hi)
        if a.is_non_positive():
            new_lo = -a.hi if a.hi is not None else None
            new_hi = -a.lo if a.lo is not None else None
            return Interval(new_lo, new_hi)
        # straddles zero
        neg_abs = -a.lo if a.lo is not None else None
        pos_abs = a.hi
        hi = _max_opt(neg_abs, pos_abs)
        return Interval(0, hi)

    @staticmethod
    def sqrt_interval(a: Interval) -> Interval:
        lo = a.lo if a.lo is not None else 0
        lo = max(lo, 0)
        lo_r = math.sqrt(lo)
        if a.hi is not None:
            if a.hi < 0:
                return Interval(1, -1)  # bottom
            hi_r: Optional[float] = math.sqrt(a.hi)
        else:
            hi_r = None
        return Interval(lo_r, hi_r)

    @staticmethod
    def union(a: Interval, b: Interval) -> Interval:
        return a.join(b)

    @staticmethod
    def intersect(a: Interval, b: Interval) -> Interval:
        return a.meet(b)

    @staticmethod
    def widen(a: Interval, b: Interval) -> Interval:
        return a.widen(b)

    @staticmethod
    def narrow(a: Interval, b: Interval) -> Interval:
        return a.narrow(b)

    @staticmethod
    def split_at_zero(a: Interval) -> Tuple[Interval, Interval]:
        neg = Interval(a.lo, min(a.hi, -1e-300) if a.hi is not None else -1e-300)
        pos = Interval(max(a.lo, 1e-300) if a.lo is not None else 1e-300, a.hi)
        return neg, pos

    @staticmethod
    def is_positive(a: Interval) -> bool:
        return a.is_positive()

    @staticmethod
    def is_negative(a: Interval) -> bool:
        return a.is_negative()

    @staticmethod
    def is_non_negative(a: Interval) -> bool:
        return a.is_non_negative()

    @staticmethod
    def is_non_positive(a: Interval) -> bool:
        return a.is_non_positive()

    @staticmethod
    def contains_zero(a: Interval) -> bool:
        return a.contains_zero()


_IA = NumericIntervalArithmetic


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class NumericModelBase(ABC):
    """Abstract base for numeric stdlib models."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def supported_operations(self) -> List[str]: ...

    @abstractmethod
    def get_signature(self, op_name: str) -> FunctionSignature: ...

    @abstractmethod
    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult: ...

    def check_precondition(self, op_name: str,
                           *operand_types: RefinementType) -> List[str]:
        sig = self.get_signature(op_name)
        violations: List[str] = []
        for pre in sig.preconditions:
            for ot in operand_types:
                iv = _ensure_interval(ot)
                if pre.kind == "!=" and len(pre.args) >= 1:
                    val = pre.args[0]
                    if iv.is_singleton() and iv.lo == val:
                        violations.append(
                            f"Precondition {pre} violated: operand is exactly {val}")
                elif pre.kind == ">=" and len(pre.args) >= 1:
                    val = pre.args[0]
                    if iv.hi is not None and iv.hi < val:
                        violations.append(
                            f"Precondition {pre} violated: operand upper bound {iv.hi} < {val}")
                elif pre.kind == ">" and len(pre.args) >= 1:
                    val = pre.args[0]
                    if iv.hi is not None and iv.hi <= val:
                        violations.append(
                            f"Precondition {pre} violated: operand upper bound {iv.hi} <= {val}")
                elif pre.kind == "<=" and len(pre.args) >= 1:
                    val = pre.args[0]
                    if iv.lo is not None and iv.lo > val:
                        violations.append(
                            f"Precondition {pre} violated: operand lower bound {iv.lo} > {val}")
                elif pre.kind == "in_range" and len(pre.args) >= 2:
                    lo_r, hi_r = pre.args[0], pre.args[1]
                    if iv.hi is not None and iv.hi < lo_r:
                        violations.append(f"Precondition {pre} violated: out of range")
                    if iv.lo is not None and iv.lo > hi_r:
                        violations.append(f"Precondition {pre} violated: out of range")
        return violations

    def _dispatch(self, op_name: str,
                  *operand_types: RefinementType) -> TransferResult:
        method = getattr(self, "_transfer_" + op_name, None)
        if method is not None:
            return method(*operand_types)
        return TransferResult(RefinementType("any"))


# ---------------------------------------------------------------------------
# IntegerModels
# ---------------------------------------------------------------------------

class IntegerModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "int"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__add__", "__sub__", "__mul__", "__floordiv__", "__truediv__",
            "__mod__", "__pow__", "__lt__", "__le__", "__gt__", "__ge__",
            "__eq__", "__ne__", "__and__", "__or__", "__xor__", "__lshift__",
            "__rshift__", "__invert__", "__int__", "__float__", "__str__",
            "__bool__", "__abs__", "__neg__", "__pos__", "bit_length",
            "bit_count", "to_bytes", "from_bytes", "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        sigs: Dict[str, FunctionSignature] = {
            "__add__": FunctionSignature(
                "int.__add__",
                [_make_int(), _make_int()],
                _make_int(),
                is_pure=True,
            ),
            "__sub__": FunctionSignature(
                "int.__sub__",
                [_make_int(), _make_int()],
                _make_int(),
                is_pure=True,
            ),
            "__mul__": FunctionSignature(
                "int.__mul__",
                [_make_int(), _make_int()],
                _make_int(),
                is_pure=True,
            ),
            "__floordiv__": FunctionSignature(
                "int.__floordiv__",
                [_make_int(), _make_int()],
                _make_int(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            ),
            "__truediv__": FunctionSignature(
                "int.__truediv__",
                [_make_int(), _make_int()],
                _make_float(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            ),
            "__mod__": FunctionSignature(
                "int.__mod__",
                [_make_int(), _make_int()],
                _make_int(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            ),
            "__pow__": FunctionSignature(
                "int.__pow__",
                [_make_int(), _make_int()],
                _make_int(),
                is_pure=True,
            ),
            "__lt__": FunctionSignature("int.__lt__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__le__": FunctionSignature("int.__le__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__gt__": FunctionSignature("int.__gt__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__ge__": FunctionSignature("int.__ge__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__eq__": FunctionSignature("int.__eq__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__ne__": FunctionSignature("int.__ne__",
                                        [_make_int(), _make_int()], _make_bool(), is_pure=True),
            "__and__": FunctionSignature("int.__and__",
                                         [_make_int(), _make_int()], _make_int(), is_pure=True),
            "__or__": FunctionSignature("int.__or__",
                                        [_make_int(), _make_int()], _make_int(), is_pure=True),
            "__xor__": FunctionSignature("int.__xor__",
                                         [_make_int(), _make_int()], _make_int(), is_pure=True),
            "__lshift__": FunctionSignature("int.__lshift__",
                                            [_make_int(), _make_int()], _make_int(),
                                            preconditions=[Predicate(">=", [0])],
                                            may_raise=["ValueError"], is_pure=True),
            "__rshift__": FunctionSignature("int.__rshift__",
                                            [_make_int(), _make_int()], _make_int(),
                                            preconditions=[Predicate(">=", [0])],
                                            may_raise=["ValueError"], is_pure=True),
            "__invert__": FunctionSignature("int.__invert__",
                                            [_make_int()], _make_int(), is_pure=True),
            "__int__": FunctionSignature("int.__int__",
                                         [_make_int()], _make_int(), is_pure=True),
            "__float__": FunctionSignature("int.__float__",
                                           [_make_int()], _make_float(),
                                           may_raise=["OverflowError"], is_pure=True),
            "__str__": FunctionSignature("int.__str__",
                                         [_make_int()], _make_str(), is_pure=True),
            "__bool__": FunctionSignature("int.__bool__",
                                          [_make_int()], _make_bool(), is_pure=True),
            "__abs__": FunctionSignature("int.__abs__",
                                         [_make_int()], _make_int(Interval(0, None)), is_pure=True),
            "__neg__": FunctionSignature("int.__neg__",
                                         [_make_int()], _make_int(), is_pure=True),
            "__pos__": FunctionSignature("int.__pos__",
                                         [_make_int()], _make_int(), is_pure=True),
            "bit_length": FunctionSignature("int.bit_length",
                                             [_make_int()], _make_int(Interval(0, None)), is_pure=True),
            "bit_count": FunctionSignature("int.bit_count",
                                            [_make_int()], _make_int(Interval(0, None)), is_pure=True),
            "to_bytes": FunctionSignature("int.to_bytes",
                                           [_make_int()],
                                           RefinementType("bytes"),
                                           may_raise=["OverflowError"],
                                           is_pure=True),
            "from_bytes": FunctionSignature("int.from_bytes",
                                             [RefinementType("bytes")],
                                             _make_int(), is_pure=True),
            "constructor": FunctionSignature("int",
                                              [RefinementType("any")],
                                              _make_int(),
                                              may_raise=["ValueError", "TypeError"],
                                              is_pure=True),
        }
        return sigs.get(op_name, FunctionSignature(f"int.{op_name}"))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    # -- arithmetic transfers ------------------------------------------------

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.add(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.sub(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.mul(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer___floordiv__(self, a: RefinementType,
                                b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        narrowings: Dict[str, RefinementType] = {}
        if bi.contains_zero():
            exceptions.append("ZeroDivisionError")
            neg, pos = _IA.split_at_zero(bi)
            narrowings["__arg1__"] = _make_int(neg.join(pos))
        ai = _ensure_interval(a)
        ri = _IA.floor_div(ai, bi)
        return TransferResult(_make_int(ri), exceptions, narrowings)

    def _transfer___truediv__(self, a: RefinementType,
                               b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ZeroDivisionError")
        ai = _ensure_interval(a)
        ri = _IA.div(ai, bi)
        return TransferResult(_make_float(ri), exceptions)

    def _transfer___mod__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ZeroDivisionError")
        ai = _ensure_interval(a)
        ri = _IA.mod(ai, bi)
        return TransferResult(_make_int(ri), exceptions)

    def _transfer___pow__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        # negative exponent → float
        if bi.hi is not None and bi.hi < 0:
            ri = _IA.pow_interval(ai, bi)
            return TransferResult(_make_float(ri))
        if bi.lo is not None and bi.lo >= 0:
            ri = _IA.pow_interval(ai, bi)
            return TransferResult(_make_int(ri))
        # mixed
        ri = _IA.pow_interval(ai, bi)
        return TransferResult(RefinementType("any", None, ri))

    # -- comparison transfers ------------------------------------------------

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        narrowings: Dict[str, RefinementType] = {}
        if ai.hi is not None and bi.lo is not None and ai.hi < bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo >= bi.hi:
            return TransferResult(_make_bool(False))
        # narrow: if True then a < b
        true_a = Interval(ai.lo, _min_bound(ai.hi,
                          _sub_opt(bi.hi, 1) if bi.hi is not None else None))
        true_b = Interval(_max_bound(bi.lo,
                          _add_opt(ai.lo, 1) if ai.lo is not None else None), bi.hi)
        narrowings["__arg0__:true"] = _make_int(true_a)
        narrowings["__arg1__:true"] = _make_int(true_b)
        narrowings["__arg0__:false"] = _make_int(
            Interval(_max_bound(ai.lo, bi.lo), ai.hi))
        narrowings["__arg1__:false"] = _make_int(
            Interval(bi.lo, _min_bound(bi.hi, ai.hi)))
        return TransferResult(_make_bool(), [], narrowings)

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        narrowings: Dict[str, RefinementType] = {}
        if ai.hi is not None and bi.lo is not None and ai.hi <= bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo > bi.hi:
            return TransferResult(_make_bool(False))
        narrowings["__arg0__:true"] = _make_int(
            Interval(ai.lo, _min_bound(ai.hi, bi.hi)))
        narrowings["__arg1__:true"] = _make_int(
            Interval(_max_bound(bi.lo, ai.lo), bi.hi))
        narrowings["__arg0__:false"] = _make_int(
            Interval(_max_bound(ai.lo,
                     _add_opt(bi.hi, 1) if bi.hi is not None else None), ai.hi))
        narrowings["__arg1__:false"] = _make_int(
            Interval(bi.lo, _min_bound(bi.hi,
                     _sub_opt(ai.lo, 1) if ai.lo is not None else None)))
        return TransferResult(_make_bool(), [], narrowings)

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___lt__(b, a)

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___le__(b, a)

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        narrowings: Dict[str, RefinementType] = {}
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(False))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(True))
        narrowings["__arg0__:true"] = _make_int(overlap)
        narrowings["__arg1__:true"] = _make_int(overlap)
        return TransferResult(_make_bool(), [], narrowings)

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        narrowings: Dict[str, RefinementType] = {}
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(True))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool(), [], narrowings)

    # -- bitwise transfers ---------------------------------------------------

    def _transfer___and__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_non_negative() and bi.is_non_negative():
            hi_a = ai.hi if ai.hi is not None else None
            hi_b = bi.hi if bi.hi is not None else None
            hi = _min_bound(hi_a, hi_b)
            return TransferResult(_make_int(Interval(0, hi)))
        return TransferResult(_make_int())

    def _transfer___or__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_non_negative() and bi.is_non_negative():
            hi_a = ai.hi if ai.hi is not None else None
            hi_b = bi.hi if bi.hi is not None else None
            # OR can be at most 2*max - 1 roughly, use max*2 as approximation
            if hi_a is not None and hi_b is not None:
                combined = max(hi_a, hi_b)
                # next power of 2
                if combined > 0:
                    bits = int(combined).bit_length()
                    upper = (1 << bits) - 1
                else:
                    upper = 0
                return TransferResult(_make_int(Interval(max(ai.lo or 0, bi.lo or 0), upper)))
        return TransferResult(_make_int())

    def _transfer___xor__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_non_negative() and bi.is_non_negative():
            hi_a = ai.hi if ai.hi is not None else None
            hi_b = bi.hi if bi.hi is not None else None
            if hi_a is not None and hi_b is not None:
                combined = max(hi_a, hi_b)
                if combined > 0:
                    bits = int(combined).bit_length()
                    upper = (1 << bits) - 1
                else:
                    upper = 0
                return TransferResult(_make_int(Interval(0, upper)))
        return TransferResult(_make_int())

    def _transfer___lshift__(self, a: RefinementType,
                              b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        exceptions: List[str] = []
        if bi.lo is not None and bi.lo < 0:
            exceptions.append("ValueError")
        if ai.is_singleton() and bi.is_singleton() and bi.lo is not None and bi.lo >= 0:
            val = int(ai.lo) << int(bi.lo)  # type: ignore[arg-type]
            return TransferResult(_make_int(Interval(val, val)), exceptions)
        return TransferResult(_make_int(), exceptions)

    def _transfer___rshift__(self, a: RefinementType,
                              b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        exceptions: List[str] = []
        if bi.lo is not None and bi.lo < 0:
            exceptions.append("ValueError")
        if ai.is_non_negative():
            lo = 0
            hi = ai.hi if (bi.lo is None or bi.lo == 0) and ai.hi is not None else ai.hi
            if ai.hi is not None and bi.lo is not None and bi.lo > 0:
                hi = ai.hi / (2 ** bi.lo)
            return TransferResult(_make_int(Interval(lo, hi)), exceptions)
        return TransferResult(_make_int(), exceptions)

    def _transfer___invert__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        # ~x = -x - 1
        lo = (-ai.hi - 1) if ai.hi is not None else None
        hi = (-ai.lo - 1) if ai.lo is not None else None
        return TransferResult(_make_int(Interval(lo, hi)))

    # -- conversion transfers ------------------------------------------------

    def _transfer___int__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int(_ensure_interval(a)))

    def _transfer___float__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        return TransferResult(_make_float(ai))

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo == 0:
            return TransferResult(_make_bool(False))
        if ai.lo is not None and ai.lo > 0:
            return TransferResult(_make_bool(True))
        if ai.hi is not None and ai.hi < 0:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer___abs__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.abs_interval(ai)
        return TransferResult(_make_int(ri))

    def _transfer___neg__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.negate(ai)
        return TransferResult(_make_int(ri))

    def _transfer___pos__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int(_ensure_interval(a)))

    def _transfer_bit_length(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo is not None:
            bl = int(abs(ai.lo)).bit_length()
            return TransferResult(_make_int(Interval(bl, bl)))
        return TransferResult(_make_int(Interval(0, None)))

    def _transfer_bit_count(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo is not None:
            bc = bin(int(abs(ai.lo))).count("1")
            return TransferResult(_make_int(Interval(bc, bc)))
        return TransferResult(_make_int(Interval(0, None)))

    def _transfer_to_bytes(self, a: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("bytes"), ["OverflowError"])

    def _transfer_from_bytes(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int())

    def _transfer_constructor(self, a: RefinementType) -> TransferResult:
        if a.base_type == "str":
            return TransferResult(_make_int(), ["ValueError"])
        if a.base_type == "float":
            ai = _ensure_interval(a)
            if ai.lo is not None:
                lo: Optional[float] = math.floor(ai.lo) if ai.lo >= 0 else math.ceil(ai.lo)
            else:
                lo = None
            if ai.hi is not None:
                hi: Optional[float] = math.floor(ai.hi) if ai.hi >= 0 else math.ceil(ai.hi)
            else:
                hi = None
            return TransferResult(_make_int(Interval(lo, hi)))
        if a.base_type == "bool":
            return TransferResult(_make_int(Interval(0, 1)))
        if a.base_type == "int":
            return TransferResult(_make_int(_ensure_interval(a)))
        return TransferResult(_make_int(), ["TypeError", "ValueError"])


# ---------------------------------------------------------------------------
# FloatModels
# ---------------------------------------------------------------------------

class FloatModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "float"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "__lt__", "__le__", "__gt__", "__ge__",
            "__eq__", "__ne__", "__abs__", "__neg__", "__pos__",
            "__int__", "__float__", "__str__", "__bool__",
            "is_integer", "is_finite", "hex", "fromhex", "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        div_sig = FunctionSignature(
            f"float.{op_name}",
            [_make_float(), _make_float()],
            _make_float(),
            preconditions=[Predicate("!=", [0])],
            may_raise=["ZeroDivisionError"],
            is_pure=True,
        )
        sigs: Dict[str, FunctionSignature] = {
            "__truediv__": div_sig,
            "__floordiv__": FunctionSignature(
                "float.__floordiv__",
                [_make_float(), _make_float()],
                _make_float(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            ),
            "__mod__": FunctionSignature(
                "float.__mod__",
                [_make_float(), _make_float()],
                _make_float(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            ),
            "__int__": FunctionSignature(
                "float.__int__",
                [_make_float()],
                _make_int(),
                may_raise=["OverflowError", "ValueError"],
                is_pure=True,
            ),
            "constructor": FunctionSignature(
                "float",
                [RefinementType("any")],
                _make_float(),
                may_raise=["ValueError"],
                is_pure=True,
            ),
        }
        if op_name in sigs:
            return sigs[op_name]
        return FunctionSignature(f"float.{op_name}", is_pure=True)

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _specials(self, rt: RefinementType) -> Set[str]:
        return set(rt.extra.get("special_values", set()))

    def _merge_specials(self, a: RefinementType, b: RefinementType,
                        op: str) -> Set[str]:
        sa, sb = self._specials(a), self._specials(b)
        result: Set[str] = set()
        if "nan" in sa or "nan" in sb:
            result.add("nan")
        if op in ("__add__", "__sub__", "__mul__", "__truediv__", "__pow__"):
            if "inf" in sa or "inf" in sb or "-inf" in sa or "-inf" in sb:
                result.add("inf")
                result.add("-inf")
        return result

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.add(ai, bi)
        sp = self._merge_specials(a, b, "__add__")
        if "inf" in self._specials(a) and "-inf" in self._specials(b):
            sp.add("nan")
        return TransferResult(_make_float(ri, sp))

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.sub(ai, bi)
        sp = self._merge_specials(a, b, "__sub__")
        return TransferResult(_make_float(ri, sp))

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.mul(ai, bi)
        sp = self._merge_specials(a, b, "__mul__")
        # 0 * inf = nan
        if (ai.contains_zero() and ("inf" in self._specials(b) or "-inf" in self._specials(b))):
            sp.add("nan")
        if (bi.contains_zero() and ("inf" in self._specials(a) or "-inf" in self._specials(a))):
            sp.add("nan")
        return TransferResult(_make_float(ri, sp))

    def _transfer___truediv__(self, a: RefinementType,
                               b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        sp = self._merge_specials(a, b, "__truediv__")
        if bi.contains_zero() and "nan" not in sp:
            if bi.is_singleton() and bi.lo == 0:
                exceptions.append("ZeroDivisionError")
        ai = _ensure_interval(a)
        ri = _IA.div(ai, bi)
        if ai.contains_zero() and bi.contains_zero():
            sp.add("nan")
        return TransferResult(_make_float(ri, sp), exceptions)

    def _transfer___floordiv__(self, a: RefinementType,
                                b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ZeroDivisionError")
        ai = _ensure_interval(a)
        ri = _IA.floor_div(ai, bi)
        sp = self._merge_specials(a, b, "__floordiv__")
        return TransferResult(_make_float(ri, sp), exceptions)

    def _transfer___mod__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ZeroDivisionError")
        ai = _ensure_interval(a)
        ri = _IA.mod(ai, bi)
        return TransferResult(_make_float(ri), exceptions)

    def _transfer___pow__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        sp = self._merge_specials(a, b, "__pow__")
        # 0**negative → ZeroDivisionError
        exceptions: List[str] = []
        if ai.contains_zero() and bi.lo is not None and bi.lo < 0:
            exceptions.append("ZeroDivisionError")
        # negative base with non-integer exponent → ValueError
        if ai.lo is not None and ai.lo < 0:
            exceptions.append("ValueError")
        ri = _IA.pow_interval(ai, bi)
        return TransferResult(_make_float(ri, sp), exceptions)

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        sa, sb = self._specials(a), self._specials(b)
        if "nan" in sa or "nan" in sb:
            return TransferResult(_make_bool(False))
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi < bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo >= bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        sa, sb = self._specials(a), self._specials(b)
        if "nan" in sa or "nan" in sb:
            return TransferResult(_make_bool(False))
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi <= bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo > bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___lt__(b, a)

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___le__(b, a)

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        sa, sb = self._specials(a), self._specials(b)
        # nan != nan
        if "nan" in sa or "nan" in sb:
            return TransferResult(_make_bool(False))
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(False))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        sa, sb = self._specials(a), self._specials(b)
        if "nan" in sa or "nan" in sb:
            return TransferResult(_make_bool(True))
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(True))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___abs__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.abs_interval(ai)
        return TransferResult(_make_float(ri))

    def _transfer___neg__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.negate(ai)
        sp = self._specials(a)
        new_sp: Set[str] = set()
        if "inf" in sp:
            new_sp.add("-inf")
        if "-inf" in sp:
            new_sp.add("inf")
        if "nan" in sp:
            new_sp.add("nan")
        return TransferResult(_make_float(ri, new_sp))

    def _transfer___pos__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(_ensure_interval(a), self._specials(a)))

    def _transfer___int__(self, a: RefinementType) -> TransferResult:
        sp = self._specials(a)
        exceptions: List[str] = []
        if "inf" in sp or "-inf" in sp:
            exceptions.append("OverflowError")
        if "nan" in sp:
            exceptions.append("ValueError")
        ai = _ensure_interval(a)
        lo: Optional[float] = None
        hi: Optional[float] = None
        if ai.lo is not None:
            lo = math.ceil(ai.lo) if ai.lo < 0 else math.floor(ai.lo)
        if ai.hi is not None:
            hi = math.floor(ai.hi) if ai.hi >= 0 else math.ceil(ai.hi)
        return TransferResult(_make_int(Interval(lo, hi)), exceptions)

    def _transfer___float__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(_ensure_interval(a), self._specials(a)))

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        sp = self._specials(a)
        if "nan" in sp:
            return TransferResult(_make_bool(True))
        if ai.is_singleton() and ai.lo == 0.0:
            return TransferResult(_make_bool(False))
        if (ai.lo is not None and ai.lo > 0) or (ai.hi is not None and ai.hi < 0):
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer_is_integer(self, a: RefinementType) -> TransferResult:
        sp = self._specials(a)
        if "nan" in sp or "inf" in sp or "-inf" in sp:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer_is_finite(self, a: RefinementType) -> TransferResult:
        sp = self._specials(a)
        if not sp:
            return TransferResult(_make_bool(True))
        if "inf" in sp or "-inf" in sp or "nan" in sp:
            return TransferResult(_make_bool())
        return TransferResult(_make_bool(True))

    def _transfer_hex(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer_fromhex(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(), ["ValueError"])

    def _transfer_constructor(self, a: RefinementType) -> TransferResult:
        if a.base_type == "str":
            return TransferResult(_make_float(), ["ValueError"])
        if a.base_type == "int":
            return TransferResult(_make_float(_ensure_interval(a)))
        if a.base_type == "bool":
            return TransferResult(_make_float(Interval(0.0, 1.0)))
        return TransferResult(_make_float(), ["TypeError", "ValueError"])


# ---------------------------------------------------------------------------
# ComplexModels
# ---------------------------------------------------------------------------

class ComplexModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "complex"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__add__", "__sub__", "__mul__", "__truediv__", "__pow__",
            "__eq__", "__ne__", "__abs__", "__neg__", "__pos__",
            "__bool__", "__str__", "real", "imag", "conjugate",
            "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        if op_name == "__truediv__":
            return FunctionSignature(
                "complex.__truediv__",
                [_make_complex(), _make_complex()],
                _make_complex(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            )
        if op_name in ("__lt__", "__le__", "__gt__", "__ge__"):
            return FunctionSignature(
                f"complex.{op_name}",
                [_make_complex(), _make_complex()],
                _make_bool(),
                may_raise=["TypeError"],
                is_pure=True,
            )
        return FunctionSignature(f"complex.{op_name}", is_pure=True)

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___truediv__(self, a: RefinementType,
                               b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex(), ["ZeroDivisionError"])

    def _transfer___pow__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex(), ["ZeroDivisionError"])

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer___abs__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer___neg__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___pos__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer_real(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_imag(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_conjugate(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(), ["TypeError"])

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(), ["TypeError"])

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(), ["TypeError"])

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(), ["TypeError"])

    def _transfer_constructor(self, *args: RefinementType) -> TransferResult:
        return TransferResult(_make_complex(), ["ValueError", "TypeError"])


# ---------------------------------------------------------------------------
# BoolModels
# ---------------------------------------------------------------------------

class BoolModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "bool"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__and__", "__or__", "__xor__", "__not__",
            "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
            "__int__", "__float__", "__str__", "__bool__",
            "__add__", "__sub__", "__mul__",
            "truthiness", "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        return FunctionSignature(f"bool.{op_name}", is_pure=True)

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer___and__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        # Python 'and' returns first falsy or last value
        # bitwise & on bools
        if ai.is_singleton() and bi.is_singleton():
            va = int(ai.lo or 0)
            vb = int(bi.lo or 0)
            r = va & vb
            return TransferResult(_make_bool(bool(r)))
        return TransferResult(_make_bool())

    def _transfer___or__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            va = int(ai.lo or 0)
            vb = int(bi.lo or 0)
            r = va | vb
            return TransferResult(_make_bool(bool(r)))
        return TransferResult(_make_bool())

    def _transfer___xor__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            va = int(ai.lo or 0)
            vb = int(bi.lo or 0)
            r = va ^ vb
            return TransferResult(_make_bool(bool(r)))
        return TransferResult(_make_bool())

    def _transfer___not__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton():
            v = bool(ai.lo)
            return TransferResult(_make_bool(not v))
        return TransferResult(_make_bool())

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            return TransferResult(_make_bool(ai.lo == bi.lo))
        return TransferResult(_make_bool())

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            return TransferResult(_make_bool(ai.lo != bi.lo))
        return TransferResult(_make_bool())

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        # bool subtype of int: False < True
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            return TransferResult(_make_bool(bool((ai.lo or 0) < (bi.lo or 0))))
        return TransferResult(_make_bool())

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.is_singleton() and bi.is_singleton():
            return TransferResult(_make_bool(bool((ai.lo or 0) <= (bi.lo or 0))))
        return TransferResult(_make_bool())

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___lt__(b, a)

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___le__(b, a)

    def _transfer___int__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int(_ensure_interval(a)))

    def _transfer___float__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        return TransferResult(_make_float(Interval(
            float(ai.lo) if ai.lo is not None else 0.0,
            float(ai.hi) if ai.hi is not None else 1.0)))

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        # bool + bool → int (True+True=2)
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.add(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.sub(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.mul(ai, bi)
        return TransferResult(_make_int(ri))

    def _transfer_truthiness(self, a: RefinementType) -> TransferResult:
        bt = a.base_type
        ai = _ensure_interval(a)
        if bt in ("int", "float", "bool"):
            if ai.is_singleton() and ai.lo == 0:
                return TransferResult(_make_bool(False))
            if ai.lo is not None and ai.lo > 0:
                return TransferResult(_make_bool(True))
            if ai.hi is not None and ai.hi < 0:
                return TransferResult(_make_bool(True))
        if bt == "NoneType":
            return TransferResult(_make_bool(False))
        if bt == "str":
            # empty string → False, else True
            return TransferResult(_make_bool())
        return TransferResult(_make_bool())

    def _transfer_constructor(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())


# ---------------------------------------------------------------------------
# DecimalModels
# ---------------------------------------------------------------------------

class DecimalModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "Decimal"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "__abs__", "__neg__", "__pos__",
            "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
            "__int__", "__float__", "__str__", "__bool__",
            "quantize", "normalize", "to_integral_value",
            "sqrt", "ln", "log10", "exp",
            "compare", "copy_abs", "copy_negate", "copy_sign",
            "is_finite", "is_infinite", "is_nan", "is_zero",
            "is_signed", "is_normal", "is_subnormal",
            "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        if op_name in ("__truediv__", "__floordiv__", "__mod__"):
            return FunctionSignature(
                f"Decimal.{op_name}",
                [RefinementType("Decimal"), RefinementType("Decimal")],
                RefinementType("Decimal"),
                preconditions=[Predicate("!=", [0])],
                may_raise=["InvalidOperation", "DivisionByZero"],
                is_pure=True,
            )
        if op_name == "sqrt":
            return FunctionSignature(
                "Decimal.sqrt",
                [RefinementType("Decimal")],
                RefinementType("Decimal"),
                preconditions=[Predicate(">=", [0])],
                may_raise=["InvalidOperation"],
                is_pure=True,
            )
        if op_name == "ln":
            return FunctionSignature(
                "Decimal.ln",
                [RefinementType("Decimal")],
                RefinementType("Decimal"),
                preconditions=[Predicate(">", [0])],
                may_raise=["InvalidOperation"],
                is_pure=True,
            )
        return FunctionSignature(f"Decimal.{op_name}", is_pure=True)

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _dec_type(self, iv: Optional[Interval] = None) -> RefinementType:
        return RefinementType("Decimal", None, iv, NullityTag.DEFINITELY_NOT_NULL)

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        return TransferResult(self._dec_type(_IA.add(ai, bi)))

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        return TransferResult(self._dec_type(_IA.sub(ai, bi)))

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        return TransferResult(self._dec_type(_IA.mul(ai, bi)))

    def _transfer___truediv__(self, a: RefinementType,
                               b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("DivisionByZero")
        ai = _ensure_interval(a)
        ri = _IA.div(ai, bi)
        return TransferResult(self._dec_type(ri), exceptions)

    def _transfer___floordiv__(self, a: RefinementType,
                                b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("DivisionByZero")
        ai = _ensure_interval(a)
        ri = _IA.floor_div(ai, bi)
        return TransferResult(self._dec_type(ri), exceptions)

    def _transfer___mod__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("DivisionByZero")
        ai = _ensure_interval(a)
        ri = _IA.mod(ai, bi)
        return TransferResult(self._dec_type(ri), exceptions)

    def _transfer___pow__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        ri = _IA.pow_interval(ai, bi)
        return TransferResult(self._dec_type(ri))

    def _transfer___abs__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.abs_interval(ai)
        return TransferResult(self._dec_type(ri))

    def _transfer___neg__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        return TransferResult(self._dec_type(_IA.negate(ai)))

    def _transfer___pos__(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._dec_type(_ensure_interval(a)))

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi < bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo >= bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi <= bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo > bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___lt__(b, a)

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___le__(b, a)

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(False))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        overlap = ai.meet(bi)
        if overlap.is_bottom():
            return TransferResult(_make_bool(True))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___int__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo = math.floor(ai.lo) if ai.lo is not None else None
        hi = math.ceil(ai.hi) if ai.hi is not None else None
        return TransferResult(_make_int(Interval(lo, hi)))

    def _transfer___float__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(_ensure_interval(a)))

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo == 0:
            return TransferResult(_make_bool(False))
        if ai.lo is not None and ai.lo > 0:
            return TransferResult(_make_bool(True))
        if ai.hi is not None and ai.hi < 0:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer_quantize(self, a: RefinementType,
                            b: RefinementType) -> TransferResult:
        return TransferResult(self._dec_type(), ["InvalidOperation"])

    def _transfer_normalize(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._dec_type(_ensure_interval(a)))

    def _transfer_to_integral_value(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo = math.floor(ai.lo) if ai.lo is not None else None
        hi = math.ceil(ai.hi) if ai.hi is not None else None
        return TransferResult(self._dec_type(Interval(lo, hi)))

    def _transfer_sqrt(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is not None and ai.lo < 0:
            exceptions.append("InvalidOperation")
        ri = _IA.sqrt_interval(ai)
        return TransferResult(self._dec_type(ri), exceptions)

    def _transfer_ln(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo <= 0:
            exceptions.append("InvalidOperation")
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None and ai.lo > 0:
            lo_r = math.log(ai.lo)
        if ai.hi is not None and ai.hi > 0:
            hi_r = math.log(ai.hi)
        return TransferResult(self._dec_type(Interval(lo_r, hi_r)), exceptions)

    def _transfer_log10(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo <= 0:
            exceptions.append("InvalidOperation")
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None and ai.lo > 0:
            lo_r = math.log10(ai.lo)
        if ai.hi is not None and ai.hi > 0:
            hi_r = math.log10(ai.hi)
        return TransferResult(self._dec_type(Interval(lo_r, hi_r)), exceptions)

    def _transfer_exp(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo_r = math.exp(ai.lo) if ai.lo is not None else 0.0
        hi_r = math.exp(ai.hi) if ai.hi is not None else None
        return TransferResult(self._dec_type(Interval(lo_r, hi_r)))

    def _transfer_compare(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(self._dec_type(Interval(-1, 1)))

    def _transfer_copy_abs(self, a: RefinementType) -> TransferResult:
        return self._transfer___abs__(a)

    def _transfer_copy_negate(self, a: RefinementType) -> TransferResult:
        return self._transfer___neg__(a)

    def _transfer_copy_sign(self, a: RefinementType,
                             b: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.abs_interval(ai)
        bi = _ensure_interval(b)
        if bi.is_negative():
            ri = _IA.negate(ri)
        elif not bi.is_non_negative():
            neg = _IA.negate(ri)
            ri = ri.join(neg)
        return TransferResult(self._dec_type(ri))

    def _transfer_is_finite(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(True))

    def _transfer_is_infinite(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(False))

    def _transfer_is_nan(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool(False))

    def _transfer_is_zero(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo == 0:
            return TransferResult(_make_bool(True))
        if not ai.contains_zero():
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer_is_signed(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_negative():
            return TransferResult(_make_bool(True))
        if ai.is_non_negative():
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer_is_normal(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_is_subnormal(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_constructor(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._dec_type(), ["InvalidOperation"])


# ---------------------------------------------------------------------------
# FractionModels
# ---------------------------------------------------------------------------

class FractionModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "Fraction"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "__add__", "__sub__", "__mul__", "__truediv__",
            "__floordiv__", "__mod__", "__pow__",
            "__abs__", "__neg__", "__pos__",
            "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
            "__int__", "__float__", "__str__", "__bool__",
            "numerator", "denominator", "limit_denominator",
            "constructor",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        frac = RefinementType("Fraction")
        if op_name in ("__truediv__", "__floordiv__", "__mod__"):
            return FunctionSignature(
                f"Fraction.{op_name}", [frac, frac], frac,
                preconditions=[Predicate("!=", [0])],
                may_raise=["ZeroDivisionError"],
                is_pure=True,
            )
        return FunctionSignature(f"Fraction.{op_name}", is_pure=True)

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _frac(self, iv: Optional[Interval] = None) -> RefinementType:
        return RefinementType("Fraction", None, iv, NullityTag.DEFINITELY_NOT_NULL)

    def _transfer___add__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(self._frac(
            _IA.add(_ensure_interval(a), _ensure_interval(b))))

    def _transfer___sub__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(self._frac(
            _IA.sub(_ensure_interval(a), _ensure_interval(b))))

    def _transfer___mul__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(self._frac(
            _IA.mul(_ensure_interval(a), _ensure_interval(b))))

    def _transfer___truediv__(self, a: RefinementType,
                               b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exc: List[str] = []
        if bi.contains_zero():
            exc.append("ZeroDivisionError")
        return TransferResult(self._frac(
            _IA.div(_ensure_interval(a), bi)), exc)

    def _transfer___floordiv__(self, a: RefinementType,
                                b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exc: List[str] = []
        if bi.contains_zero():
            exc.append("ZeroDivisionError")
        return TransferResult(_make_int(
            _IA.floor_div(_ensure_interval(a), bi)), exc)

    def _transfer___mod__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exc: List[str] = []
        if bi.contains_zero():
            exc.append("ZeroDivisionError")
        return TransferResult(self._frac(
            _IA.mod(_ensure_interval(a), bi)), exc)

    def _transfer___pow__(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        return TransferResult(self._frac(
            _IA.pow_interval(ai, bi)))

    def _transfer___abs__(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._frac(
            _IA.abs_interval(_ensure_interval(a))))

    def _transfer___neg__(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._frac(
            _IA.negate(_ensure_interval(a))))

    def _transfer___pos__(self, a: RefinementType) -> TransferResult:
        return TransferResult(self._frac(_ensure_interval(a)))

    def _transfer___lt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi < bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo >= bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___le__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.hi is not None and bi.lo is not None and ai.hi <= bi.lo:
            return TransferResult(_make_bool(True))
        if ai.lo is not None and bi.hi is not None and ai.lo > bi.hi:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___gt__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___lt__(b, a)

    def _transfer___ge__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return self._transfer___le__(b, a)

    def _transfer___eq__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.meet(bi).is_bottom():
            return TransferResult(_make_bool(False))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer___ne__(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if ai.meet(bi).is_bottom():
            return TransferResult(_make_bool(True))
        if ai.is_singleton() and bi.is_singleton() and ai.lo == bi.lo:
            return TransferResult(_make_bool(False))
        return TransferResult(_make_bool())

    def _transfer___int__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo = math.floor(ai.lo) if ai.lo is not None else None
        hi = math.ceil(ai.hi) if ai.hi is not None else None
        return TransferResult(_make_int(Interval(lo, hi)))

    def _transfer___float__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(_ensure_interval(a)))

    def _transfer___str__(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_str())

    def _transfer___bool__(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo == 0:
            return TransferResult(_make_bool(False))
        if not ai.contains_zero():
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer_numerator(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int())

    def _transfer_denominator(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_int(Interval(1, None)))

    def _transfer_limit_denominator(self, a: RefinementType,
                                     *args: RefinementType) -> TransferResult:
        return TransferResult(self._frac(_ensure_interval(a)))

    def _transfer_constructor(self, *args: RefinementType) -> TransferResult:
        return TransferResult(self._frac(), ["ValueError", "TypeError", "ZeroDivisionError"])


# ---------------------------------------------------------------------------
# MathModuleModels
# ---------------------------------------------------------------------------

class MathModuleModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "math"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "sqrt", "ceil", "floor", "trunc",
            "log", "log2", "log10", "exp",
            "sin", "cos", "tan",
            "asin", "acos", "atan", "atan2",
            "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh",
            "degrees", "radians",
            "hypot", "pow", "fabs", "fmod",
            "factorial", "gcd", "lcm",
            "comb", "perm", "isqrt",
            "prod", "fsum",
            "copysign", "frexp", "ldexp", "modf",
            "remainder", "nextafter", "ulp",
            "isfinite", "isinf", "isnan", "isclose",
            "pi", "e", "inf", "nan", "tau",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        sigs: Dict[str, FunctionSignature] = {
            "sqrt": FunctionSignature(
                "math.sqrt", [_make_float()], _make_float(Interval(0, None)),
                preconditions=[Predicate(">=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "ceil": FunctionSignature(
                "math.ceil", [_make_float()], _make_int(), is_pure=True,
            ),
            "floor": FunctionSignature(
                "math.floor", [_make_float()], _make_int(), is_pure=True,
            ),
            "trunc": FunctionSignature(
                "math.trunc", [_make_float()], _make_int(), is_pure=True,
            ),
            "log": FunctionSignature(
                "math.log", [_make_float()], _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "log2": FunctionSignature(
                "math.log2", [_make_float()], _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "log10": FunctionSignature(
                "math.log10", [_make_float()], _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "exp": FunctionSignature(
                "math.exp", [_make_float()],
                _make_float(Interval(0, None)),
                may_raise=["OverflowError"],
                is_pure=True,
            ),
            "sin": FunctionSignature(
                "math.sin", [_make_float()],
                _make_float(Interval(-1.0, 1.0)),
                is_pure=True,
            ),
            "cos": FunctionSignature(
                "math.cos", [_make_float()],
                _make_float(Interval(-1.0, 1.0)),
                is_pure=True,
            ),
            "tan": FunctionSignature(
                "math.tan", [_make_float()], _make_float(),
                is_pure=True,
            ),
            "asin": FunctionSignature(
                "math.asin", [_make_float()],
                _make_float(Interval(-math.pi / 2, math.pi / 2)),
                preconditions=[Predicate("in_range", [-1, 1])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "acos": FunctionSignature(
                "math.acos", [_make_float()],
                _make_float(Interval(0, math.pi)),
                preconditions=[Predicate("in_range", [-1, 1])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "atan": FunctionSignature(
                "math.atan", [_make_float()],
                _make_float(Interval(-math.pi / 2, math.pi / 2)),
                is_pure=True,
            ),
            "atan2": FunctionSignature(
                "math.atan2", [_make_float(), _make_float()],
                _make_float(Interval(-math.pi, math.pi)),
                is_pure=True,
            ),
            "sinh": FunctionSignature(
                "math.sinh", [_make_float()], _make_float(),
                may_raise=["OverflowError"],
                is_pure=True,
            ),
            "cosh": FunctionSignature(
                "math.cosh", [_make_float()],
                _make_float(Interval(1.0, None)),
                may_raise=["OverflowError"],
                is_pure=True,
            ),
            "tanh": FunctionSignature(
                "math.tanh", [_make_float()],
                _make_float(Interval(-1.0, 1.0)),
                is_pure=True,
            ),
            "asinh": FunctionSignature(
                "math.asinh", [_make_float()], _make_float(),
                is_pure=True,
            ),
            "acosh": FunctionSignature(
                "math.acosh", [_make_float()], _make_float(Interval(0, None)),
                preconditions=[Predicate(">=", [1])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "atanh": FunctionSignature(
                "math.atanh", [_make_float()], _make_float(),
                preconditions=[Predicate("in_range", [-1, 1])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "degrees": FunctionSignature(
                "math.degrees", [_make_float()], _make_float(), is_pure=True,
            ),
            "radians": FunctionSignature(
                "math.radians", [_make_float()], _make_float(), is_pure=True,
            ),
            "hypot": FunctionSignature(
                "math.hypot", [_make_float(), _make_float()],
                _make_float(Interval(0, None)),
                is_pure=True,
            ),
            "pow": FunctionSignature(
                "math.pow", [_make_float(), _make_float()], _make_float(),
                may_raise=["ValueError", "OverflowError"],
                is_pure=True,
            ),
            "fabs": FunctionSignature(
                "math.fabs", [_make_float()],
                _make_float(Interval(0, None)),
                is_pure=True,
            ),
            "fmod": FunctionSignature(
                "math.fmod", [_make_float(), _make_float()], _make_float(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "factorial": FunctionSignature(
                "math.factorial", [_make_int()],
                _make_int(Interval(1, None)),
                preconditions=[Predicate(">=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "gcd": FunctionSignature(
                "math.gcd", [_make_int(), _make_int()],
                _make_int(Interval(0, None)),
                is_pure=True,
            ),
            "lcm": FunctionSignature(
                "math.lcm", [_make_int(), _make_int()],
                _make_int(Interval(0, None)),
                is_pure=True,
            ),
            "comb": FunctionSignature(
                "math.comb", [_make_int(), _make_int()],
                _make_int(Interval(0, None)),
                preconditions=[Predicate(">=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "perm": FunctionSignature(
                "math.perm", [_make_int(), _make_int()],
                _make_int(Interval(0, None)),
                preconditions=[Predicate(">=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "isqrt": FunctionSignature(
                "math.isqrt", [_make_int()],
                _make_int(Interval(0, None)),
                preconditions=[Predicate(">=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "prod": FunctionSignature(
                "math.prod", [_make_list()], RefinementType("any"),
                is_pure=True,
            ),
            "fsum": FunctionSignature(
                "math.fsum", [_make_list()], _make_float(),
                is_pure=True,
            ),
            "copysign": FunctionSignature(
                "math.copysign", [_make_float(), _make_float()], _make_float(),
                is_pure=True,
            ),
            "frexp": FunctionSignature(
                "math.frexp", [_make_float()],
                _make_tuple(_make_float(), _make_int()),
                is_pure=True,
            ),
            "ldexp": FunctionSignature(
                "math.ldexp", [_make_float(), _make_int()], _make_float(),
                may_raise=["OverflowError"],
                is_pure=True,
            ),
            "modf": FunctionSignature(
                "math.modf", [_make_float()],
                _make_tuple(_make_float(), _make_float()),
                is_pure=True,
            ),
            "remainder": FunctionSignature(
                "math.remainder", [_make_float(), _make_float()], _make_float(),
                preconditions=[Predicate("!=", [0])],
                may_raise=["ValueError"],
                is_pure=True,
            ),
            "nextafter": FunctionSignature(
                "math.nextafter", [_make_float(), _make_float()], _make_float(),
                is_pure=True,
            ),
            "ulp": FunctionSignature(
                "math.ulp", [_make_float()],
                _make_float(Interval(0, None)),
                is_pure=True,
            ),
            "isfinite": FunctionSignature(
                "math.isfinite", [_make_float()], _make_bool(), is_pure=True,
            ),
            "isinf": FunctionSignature(
                "math.isinf", [_make_float()], _make_bool(), is_pure=True,
            ),
            "isnan": FunctionSignature(
                "math.isnan", [_make_float()], _make_bool(), is_pure=True,
            ),
            "isclose": FunctionSignature(
                "math.isclose", [_make_float(), _make_float()], _make_bool(),
                is_pure=True,
            ),
        }
        return sigs.get(op_name, FunctionSignature(f"math.{op_name}", is_pure=True))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    # -- individual transfer functions -----------------------------------------

    def _transfer_sqrt(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo < 0:
            exceptions.append("ValueError")
        ri = _IA.sqrt_interval(ai)
        return TransferResult(_make_float(ri), exceptions)

    def _transfer_ceil(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo = math.ceil(ai.lo) if ai.lo is not None else None
        hi = math.ceil(ai.hi) if ai.hi is not None else None
        return TransferResult(_make_int(Interval(lo, hi)))

    def _transfer_floor(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo = math.floor(ai.lo) if ai.lo is not None else None
        hi = math.floor(ai.hi) if ai.hi is not None else None
        return TransferResult(_make_int(Interval(lo, hi)))

    def _transfer_trunc(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo: Optional[float] = None
        hi: Optional[float] = None
        if ai.lo is not None:
            lo = math.trunc(ai.lo)
        if ai.hi is not None:
            hi = math.trunc(ai.hi)
        return TransferResult(_make_int(Interval(lo, hi)))

    def _transfer_log(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo <= 0:
            exceptions.append("ValueError")
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None and ai.lo > 0:
            lo_r = math.log(ai.lo)
        if ai.hi is not None and ai.hi > 0:
            hi_r = math.log(ai.hi)
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_log2(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo <= 0:
            exceptions.append("ValueError")
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None and ai.lo > 0:
            lo_r = math.log2(ai.lo)
        if ai.hi is not None and ai.hi > 0:
            hi_r = math.log2(ai.hi)
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_log10(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo <= 0:
            exceptions.append("ValueError")
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None and ai.lo > 0:
            lo_r = math.log10(ai.lo)
        if ai.hi is not None and ai.hi > 0:
            hi_r = math.log10(ai.hi)
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_exp(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        lo_r = math.exp(ai.lo) if ai.lo is not None else 0.0
        hi_r: Optional[float] = None
        if ai.hi is not None:
            try:
                hi_r = math.exp(ai.hi)
            except OverflowError:
                hi_r = None
                exceptions.append("OverflowError")
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_sin(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        # If the interval width >= 2π, result is [-1, 1]
        w = ai.width()
        if w is None or w >= 2 * math.pi:
            return TransferResult(_make_float(Interval(-1.0, 1.0)))
        # Exact for singletons
        if ai.is_singleton() and ai.lo is not None:
            v = math.sin(ai.lo)
            return TransferResult(_make_float(Interval(v, v)))
        return TransferResult(_make_float(Interval(-1.0, 1.0)))

    def _transfer_cos(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        w = ai.width()
        if w is None or w >= 2 * math.pi:
            return TransferResult(_make_float(Interval(-1.0, 1.0)))
        if ai.is_singleton() and ai.lo is not None:
            v = math.cos(ai.lo)
            return TransferResult(_make_float(Interval(v, v)))
        return TransferResult(_make_float(Interval(-1.0, 1.0)))

    def _transfer_tan(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo is not None:
            v = math.tan(ai.lo)
            return TransferResult(_make_float(Interval(v, v)))
        return TransferResult(_make_float())

    def _transfer_asin(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if (ai.lo is not None and ai.lo < -1) or (ai.hi is not None and ai.hi > 1):
            exceptions.append("ValueError")
        clamped = ai.meet(Interval(-1.0, 1.0))
        if clamped.is_bottom():
            return TransferResult(_make_float(), ["ValueError"])
        lo_r = math.asin(clamped.lo) if clamped.lo is not None else -math.pi / 2
        hi_r = math.asin(clamped.hi) if clamped.hi is not None else math.pi / 2
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_acos(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if (ai.lo is not None and ai.lo < -1) or (ai.hi is not None and ai.hi > 1):
            exceptions.append("ValueError")
        clamped = ai.meet(Interval(-1.0, 1.0))
        if clamped.is_bottom():
            return TransferResult(_make_float(), ["ValueError"])
        # acos is decreasing
        lo_r = math.acos(clamped.hi) if clamped.hi is not None else 0.0
        hi_r = math.acos(clamped.lo) if clamped.lo is not None else math.pi
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_atan(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo_r = math.atan(ai.lo) if ai.lo is not None else -math.pi / 2
        hi_r = math.atan(ai.hi) if ai.hi is not None else math.pi / 2
        return TransferResult(_make_float(Interval(lo_r, hi_r)))

    def _transfer_atan2(self, a: RefinementType,
                         b: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(-math.pi, math.pi)))

    def _transfer_sinh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        lo_r: Optional[float] = None
        hi_r: Optional[float] = None
        if ai.lo is not None:
            try:
                lo_r = math.sinh(ai.lo)
            except OverflowError:
                lo_r = None
                exceptions.append("OverflowError")
        if ai.hi is not None:
            try:
                hi_r = math.sinh(ai.hi)
            except OverflowError:
                hi_r = None
                exceptions.append("OverflowError")
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_cosh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        # cosh is even, minimum at 0 is 1
        abs_iv = _IA.abs_interval(ai)
        lo_r = 1.0
        hi_r: Optional[float] = None
        if abs_iv.hi is not None:
            try:
                hi_r = math.cosh(abs_iv.hi)
            except OverflowError:
                hi_r = None
                exceptions.append("OverflowError")
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_tanh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo_r = math.tanh(ai.lo) if ai.lo is not None else -1.0
        hi_r = math.tanh(ai.hi) if ai.hi is not None else 1.0
        return TransferResult(_make_float(Interval(lo_r, hi_r)))

    def _transfer_asinh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        lo_r = math.asinh(ai.lo) if ai.lo is not None else None
        hi_r = math.asinh(ai.hi) if ai.hi is not None else None
        return TransferResult(_make_float(Interval(lo_r, hi_r)))

    def _transfer_acosh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo < 1:
            exceptions.append("ValueError")
        clamped = ai.meet(Interval(1.0, None))
        if clamped.is_bottom():
            return TransferResult(_make_float(), ["ValueError"])
        lo_r = math.acosh(max(clamped.lo, 1.0)) if clamped.lo is not None else 0.0
        hi_r = math.acosh(clamped.hi) if clamped.hi is not None else None
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_atanh(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if (ai.lo is not None and ai.lo <= -1) or (ai.hi is not None and ai.hi >= 1):
            exceptions.append("ValueError")
        clamped = ai.meet(Interval(-0.9999999999, 0.9999999999))
        if clamped.is_bottom():
            return TransferResult(_make_float(), ["ValueError"])
        lo_r = math.atanh(clamped.lo) if clamped.lo is not None else None
        hi_r = math.atanh(clamped.hi) if clamped.hi is not None else None
        return TransferResult(_make_float(Interval(lo_r, hi_r)), exceptions)

    def _transfer_degrees(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        factor = 180.0 / math.pi
        lo = ai.lo * factor if ai.lo is not None else None
        hi = ai.hi * factor if ai.hi is not None else None
        return TransferResult(_make_float(Interval(lo, hi)))

    def _transfer_radians(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        factor = math.pi / 180.0
        lo = ai.lo * factor if ai.lo is not None else None
        hi = ai.hi * factor if ai.hi is not None else None
        return TransferResult(_make_float(Interval(lo, hi)))

    def _transfer_hypot(self, *args: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer_pow(self, a: RefinementType,
                       b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        exceptions: List[str] = []
        if ai.lo is not None and ai.lo < 0:
            exceptions.append("ValueError")
        if ai.contains_zero() and bi.lo is not None and bi.lo < 0:
            exceptions.append("ValueError")
        ri = _IA.pow_interval(ai, bi)
        return TransferResult(_make_float(ri), exceptions)

    def _transfer_fabs(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        ri = _IA.abs_interval(ai)
        return TransferResult(_make_float(ri))

    def _transfer_fmod(self, a: RefinementType,
                        b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ValueError")
        ai = _ensure_interval(a)
        ri = _IA.mod(ai, bi)
        return TransferResult(_make_float(ri), exceptions)

    def _transfer_factorial(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo < 0:
            exceptions.append("ValueError")
        lo_r = 1
        hi_r: Optional[float] = None
        if ai.hi is not None and ai.hi >= 0:
            try:
                hi_r = float(math.factorial(int(ai.hi)))
            except (ValueError, OverflowError):
                hi_r = None
        return TransferResult(_make_int(Interval(lo_r, hi_r)), exceptions)

    def _transfer_gcd(self, a: RefinementType,
                       b: RefinementType) -> TransferResult:
        return TransferResult(_make_int(Interval(0, None)))

    def _transfer_lcm(self, a: RefinementType,
                       b: RefinementType) -> TransferResult:
        return TransferResult(_make_int(Interval(0, None)))

    def _transfer_comb(self, a: RefinementType,
                        b: RefinementType) -> TransferResult:
        exceptions: List[str] = []
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if bi.lo is None or bi.lo < 0:
            exceptions.append("ValueError")
        if ai.lo is None or ai.lo < 0:
            exceptions.append("ValueError")
        return TransferResult(_make_int(Interval(0, None)), exceptions)

    def _transfer_perm(self, a: RefinementType,
                        b: RefinementType) -> TransferResult:
        exceptions: List[str] = []
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        if bi.lo is None or bi.lo < 0:
            exceptions.append("ValueError")
        if ai.lo is None or ai.lo < 0:
            exceptions.append("ValueError")
        return TransferResult(_make_int(Interval(0, None)), exceptions)

    def _transfer_isqrt(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        exceptions: List[str] = []
        if ai.lo is None or ai.lo < 0:
            exceptions.append("ValueError")
        ri = _IA.sqrt_interval(ai)
        lo = math.floor(ri.lo) if ri.lo is not None else 0
        hi = math.floor(ri.hi) if ri.hi is not None else None
        return TransferResult(_make_int(Interval(lo, hi)), exceptions)

    def _transfer_prod(self, a: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("any"))

    def _transfer_fsum(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_copysign(self, a: RefinementType,
                            b: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        bi = _ensure_interval(b)
        mag = _IA.abs_interval(ai)
        if bi.is_negative():
            ri = _IA.negate(mag)
        elif bi.is_non_negative():
            ri = mag
        else:
            ri = mag.join(_IA.negate(mag))
        return TransferResult(_make_float(ri))

    def _transfer_frexp(self, a: RefinementType) -> TransferResult:
        mantissa = _make_float(Interval(-1.0, 1.0))
        exponent = _make_int()
        return TransferResult(_make_tuple(mantissa, exponent))

    def _transfer_ldexp(self, a: RefinementType,
                         b: RefinementType) -> TransferResult:
        return TransferResult(_make_float(), ["OverflowError"])

    def _transfer_modf(self, a: RefinementType) -> TransferResult:
        frac = _make_float(Interval(-1.0, 1.0))
        integer = _make_float()
        return TransferResult(_make_tuple(frac, integer))

    def _transfer_remainder(self, a: RefinementType,
                             b: RefinementType) -> TransferResult:
        bi = _ensure_interval(b)
        exceptions: List[str] = []
        if bi.contains_zero():
            exceptions.append("ValueError")
        return TransferResult(_make_float(), exceptions)

    def _transfer_nextafter(self, a: RefinementType,
                             b: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_ulp(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer_isfinite(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isinf(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isnan(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isclose(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    # Constants
    def _transfer_pi(self) -> TransferResult:
        return TransferResult(_make_float(Interval(math.pi, math.pi)))

    def _transfer_e(self) -> TransferResult:
        return TransferResult(_make_float(Interval(math.e, math.e)))

    def _transfer_tau(self) -> TransferResult:
        return TransferResult(_make_float(Interval(math.tau, math.tau)))

    def _transfer_inf(self) -> TransferResult:
        return TransferResult(_make_float(None, {"inf"}))

    def _transfer_nan(self) -> TransferResult:
        return TransferResult(_make_float(None, {"nan"}))


# ---------------------------------------------------------------------------
# CmathModuleModels
# ---------------------------------------------------------------------------

class CmathModuleModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "cmath"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "sqrt", "log", "log10", "exp",
            "sin", "cos", "tan",
            "asin", "acos", "atan",
            "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh",
            "phase", "polar", "rect",
            "isfinite", "isinf", "isnan", "isclose",
            "pi", "e", "inf", "nan", "infj", "nanj",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        complex_in_out = FunctionSignature(
            f"cmath.{op_name}",
            [_make_complex()],
            _make_complex(),
            is_pure=True,
        )
        sigs: Dict[str, FunctionSignature] = {
            "sqrt": complex_in_out,
            "log": FunctionSignature(
                "cmath.log", [_make_complex()], _make_complex(),
                may_raise=["ValueError"], is_pure=True),
            "log10": FunctionSignature(
                "cmath.log10", [_make_complex()], _make_complex(),
                may_raise=["ValueError"], is_pure=True),
            "exp": complex_in_out,
            "sin": complex_in_out,
            "cos": complex_in_out,
            "tan": complex_in_out,
            "asin": complex_in_out,
            "acos": complex_in_out,
            "atan": complex_in_out,
            "sinh": complex_in_out,
            "cosh": complex_in_out,
            "tanh": complex_in_out,
            "asinh": complex_in_out,
            "acosh": complex_in_out,
            "atanh": complex_in_out,
            "phase": FunctionSignature(
                "cmath.phase", [_make_complex()],
                _make_float(Interval(-math.pi, math.pi)), is_pure=True),
            "polar": FunctionSignature(
                "cmath.polar", [_make_complex()],
                _make_tuple(_make_float(Interval(0, None)),
                            _make_float(Interval(-math.pi, math.pi))),
                is_pure=True),
            "rect": FunctionSignature(
                "cmath.rect", [_make_float(), _make_float()],
                _make_complex(), is_pure=True),
            "isfinite": FunctionSignature(
                "cmath.isfinite", [_make_complex()], _make_bool(), is_pure=True),
            "isinf": FunctionSignature(
                "cmath.isinf", [_make_complex()], _make_bool(), is_pure=True),
            "isnan": FunctionSignature(
                "cmath.isnan", [_make_complex()], _make_bool(), is_pure=True),
            "isclose": FunctionSignature(
                "cmath.isclose", [_make_complex(), _make_complex()],
                _make_bool(), is_pure=True),
        }
        return sigs.get(op_name, FunctionSignature(f"cmath.{op_name}", is_pure=True))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer_sqrt(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_log(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex(), ["ValueError"])

    def _transfer_log10(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex(), ["ValueError"])

    def _transfer_exp(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_sin(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_cos(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_tan(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_asin(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_acos(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_atan(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_sinh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_cosh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_tanh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_asinh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_acosh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_atanh(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_phase(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(-math.pi, math.pi)))

    def _transfer_polar(self, a: RefinementType) -> TransferResult:
        r = _make_float(Interval(0, None))
        theta = _make_float(Interval(-math.pi, math.pi))
        return TransferResult(_make_tuple(r, theta))

    def _transfer_rect(self, a: RefinementType,
                        b: RefinementType) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_isfinite(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isinf(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isnan(self, a: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_isclose(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_pi(self) -> TransferResult:
        return TransferResult(_make_float(Interval(math.pi, math.pi)))

    def _transfer_e(self) -> TransferResult:
        return TransferResult(_make_float(Interval(math.e, math.e)))

    def _transfer_inf(self) -> TransferResult:
        return TransferResult(_make_float(None, {"inf"}))

    def _transfer_nan(self) -> TransferResult:
        return TransferResult(_make_float(None, {"nan"}))

    def _transfer_infj(self) -> TransferResult:
        return TransferResult(_make_complex())

    def _transfer_nanj(self) -> TransferResult:
        return TransferResult(_make_complex())


# ---------------------------------------------------------------------------
# StatisticsModuleModels
# ---------------------------------------------------------------------------

class StatisticsModuleModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "statistics"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "mean", "median", "mode",
            "stdev", "variance", "pstdev", "pvariance",
            "geometric_mean", "harmonic_mean",
            "median_low", "median_high", "median_grouped",
            "quantiles", "multimode",
            "NormalDist",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        data_in = [_make_list()]
        sigs: Dict[str, FunctionSignature] = {
            "mean": FunctionSignature(
                "statistics.mean", data_in, _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "median": FunctionSignature(
                "statistics.median", data_in, _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "mode": FunctionSignature(
                "statistics.mode", data_in, RefinementType("any"),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "stdev": FunctionSignature(
                "statistics.stdev", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">=", [2])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "variance": FunctionSignature(
                "statistics.variance", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">=", [2])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "pstdev": FunctionSignature(
                "statistics.pstdev", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "pvariance": FunctionSignature(
                "statistics.pvariance", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "geometric_mean": FunctionSignature(
                "statistics.geometric_mean", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "harmonic_mean": FunctionSignature(
                "statistics.harmonic_mean", data_in,
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "median_low": FunctionSignature(
                "statistics.median_low", data_in, RefinementType("any"),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "median_high": FunctionSignature(
                "statistics.median_high", data_in, RefinementType("any"),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "median_grouped": FunctionSignature(
                "statistics.median_grouped", data_in, _make_float(),
                preconditions=[Predicate(">", [0])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "quantiles": FunctionSignature(
                "statistics.quantiles", data_in, _make_list(_make_float()),
                preconditions=[Predicate(">=", [2])],
                may_raise=["StatisticsError"],
                is_pure=True,
            ),
            "multimode": FunctionSignature(
                "statistics.multimode", data_in, _make_list(),
                is_pure=True,
            ),
        }
        return sigs.get(op_name, FunctionSignature(f"statistics.{op_name}", is_pure=True))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer_mean(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(), ["StatisticsError"])

    def _transfer_median(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(), ["StatisticsError"])

    def _transfer_mode(self, data: RefinementType) -> TransferResult:
        elem = data.extra.get("element_type", RefinementType("any"))
        return TransferResult(elem, ["StatisticsError"])

    def _transfer_stdev(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_variance(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_pstdev(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_pvariance(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_geometric_mean(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_harmonic_mean(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)), ["StatisticsError"])

    def _transfer_median_low(self, data: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("any"), ["StatisticsError"])

    def _transfer_median_high(self, data: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("any"), ["StatisticsError"])

    def _transfer_median_grouped(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_float(), ["StatisticsError"])

    def _transfer_quantiles(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_list(_make_float()), ["StatisticsError"])

    def _transfer_multimode(self, data: RefinementType) -> TransferResult:
        return TransferResult(_make_list())

    def _transfer_NormalDist(self, *args: RefinementType) -> TransferResult:
        return TransferResult(
            RefinementType("NormalDist", None, None,
                           NullityTag.DEFINITELY_NOT_NULL,
                           {"mu": _make_float(), "sigma": _make_float(Interval(0, None))}))


# ---------------------------------------------------------------------------
# NormalDistModels (sub-model for statistics.NormalDist)
# ---------------------------------------------------------------------------

class NormalDistModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "NormalDist"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "mean", "stdev", "variance", "median",
            "pdf", "cdf", "inv_cdf",
            "overlap", "quantiles", "zscore",
            "samples",
            "__add__", "__sub__", "__mul__", "__truediv__",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        nd = RefinementType("NormalDist")
        sigs: Dict[str, FunctionSignature] = {
            "pdf": FunctionSignature("NormalDist.pdf", [nd, _make_float()],
                                      _make_float(Interval(0, None)), is_pure=True),
            "cdf": FunctionSignature("NormalDist.cdf", [nd, _make_float()],
                                      _make_float(Interval(0, 1)), is_pure=True),
            "inv_cdf": FunctionSignature("NormalDist.inv_cdf", [nd, _make_float()],
                                          _make_float(),
                                          preconditions=[Predicate("in_range", [0, 1])],
                                          may_raise=["StatisticsError"],
                                          is_pure=True),
        }
        return sigs.get(op_name, FunctionSignature(f"NormalDist.{op_name}", is_pure=True))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer_mean(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_stdev(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer_variance(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer_median(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_pdf(self, nd: RefinementType,
                       x: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)))

    def _transfer_cdf(self, nd: RefinementType,
                       x: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, 1)))

    def _transfer_inv_cdf(self, nd: RefinementType,
                           p: RefinementType) -> TransferResult:
        pi = _ensure_interval(p)
        exceptions: List[str] = []
        if pi.lo is not None and pi.lo <= 0:
            exceptions.append("StatisticsError")
        if pi.hi is not None and pi.hi >= 1:
            exceptions.append("StatisticsError")
        return TransferResult(_make_float(), exceptions)

    def _transfer_overlap(self, nd1: RefinementType,
                           nd2: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, 1)))

    def _transfer_quantiles(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_list(_make_float()))

    def _transfer_zscore(self, nd: RefinementType,
                          x: RefinementType) -> TransferResult:
        return TransferResult(_make_float())

    def _transfer_samples(self, nd: RefinementType) -> TransferResult:
        return TransferResult(_make_list(_make_float()),
                              side_effects=["rng_state_change"])

    def _transfer___add__(self, nd: RefinementType,
                          other: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("NormalDist"))

    def _transfer___sub__(self, nd: RefinementType,
                          other: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("NormalDist"))

    def _transfer___mul__(self, nd: RefinementType,
                          other: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("NormalDist"))

    def _transfer___truediv__(self, nd: RefinementType,
                               other: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("NormalDist"), ["ZeroDivisionError"])


# ---------------------------------------------------------------------------
# RandomModuleModels
# ---------------------------------------------------------------------------

class RandomModuleModels(NumericModelBase):

    @property
    def model_name(self) -> str:
        return "random"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "random", "randint", "randrange",
            "choice", "choices", "sample", "shuffle",
            "uniform", "gauss", "normalvariate",
            "lognormvariate", "expovariate",
            "vonmisesvariate", "gammavariate",
            "betavariate", "paretovariate", "weibullvariate",
            "triangular",
            "seed", "getstate", "setstate",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        sigs: Dict[str, FunctionSignature] = {
            "random": FunctionSignature(
                "random.random", [], _make_float(Interval(0.0, 1.0)),
                is_pure=False, side_effects=["rng_state_change"]),
            "randint": FunctionSignature(
                "random.randint", [_make_int(), _make_int()], _make_int(),
                is_pure=False, side_effects=["rng_state_change"]),
            "randrange": FunctionSignature(
                "random.randrange", [_make_int()], _make_int(),
                may_raise=["ValueError"],
                is_pure=False, side_effects=["rng_state_change"]),
            "choice": FunctionSignature(
                "random.choice", [_make_list()], RefinementType("any"),
                preconditions=[Predicate(">", [0])],
                may_raise=["IndexError"],
                is_pure=False, side_effects=["rng_state_change"]),
            "choices": FunctionSignature(
                "random.choices", [_make_list()], _make_list(),
                is_pure=False, side_effects=["rng_state_change"]),
            "sample": FunctionSignature(
                "random.sample", [_make_list(), _make_int()], _make_list(),
                may_raise=["ValueError"],
                is_pure=False, side_effects=["rng_state_change"]),
            "shuffle": FunctionSignature(
                "random.shuffle", [_make_list()], _make_none(),
                is_pure=False, side_effects=["mutates_arg0", "rng_state_change"]),
            "uniform": FunctionSignature(
                "random.uniform", [_make_float(), _make_float()], _make_float(),
                is_pure=False, side_effects=["rng_state_change"]),
            "gauss": FunctionSignature(
                "random.gauss", [_make_float(), _make_float()], _make_float(),
                is_pure=False, side_effects=["rng_state_change"]),
            "normalvariate": FunctionSignature(
                "random.normalvariate", [_make_float(), _make_float()], _make_float(),
                is_pure=False, side_effects=["rng_state_change"]),
            "lognormvariate": FunctionSignature(
                "random.lognormvariate", [_make_float(), _make_float()],
                _make_float(Interval(0, None)),
                is_pure=False, side_effects=["rng_state_change"]),
            "expovariate": FunctionSignature(
                "random.expovariate", [_make_float()],
                _make_float(Interval(0, None)),
                is_pure=False, side_effects=["rng_state_change"]),
            "vonmisesvariate": FunctionSignature(
                "random.vonmisesvariate", [_make_float(), _make_float()],
                _make_float(Interval(0, 2 * math.pi)),
                is_pure=False, side_effects=["rng_state_change"]),
            "gammavariate": FunctionSignature(
                "random.gammavariate", [_make_float(), _make_float()],
                _make_float(Interval(0, None)),
                preconditions=[Predicate(">", [0])],
                is_pure=False, side_effects=["rng_state_change"]),
            "betavariate": FunctionSignature(
                "random.betavariate", [_make_float(), _make_float()],
                _make_float(Interval(0, 1)),
                preconditions=[Predicate(">", [0])],
                is_pure=False, side_effects=["rng_state_change"]),
            "paretovariate": FunctionSignature(
                "random.paretovariate", [_make_float()],
                _make_float(Interval(1, None)),
                is_pure=False, side_effects=["rng_state_change"]),
            "weibullvariate": FunctionSignature(
                "random.weibullvariate", [_make_float(), _make_float()],
                _make_float(Interval(0, None)),
                is_pure=False, side_effects=["rng_state_change"]),
            "triangular": FunctionSignature(
                "random.triangular", [_make_float(), _make_float()],
                _make_float(),
                is_pure=False, side_effects=["rng_state_change"]),
            "seed": FunctionSignature(
                "random.seed", [RefinementType("any")], _make_none(),
                is_pure=False, side_effects=["rng_state_change"]),
            "getstate": FunctionSignature(
                "random.getstate", [], RefinementType("tuple"),
                is_pure=True),
            "setstate": FunctionSignature(
                "random.setstate", [RefinementType("tuple")], _make_none(),
                is_pure=False, side_effects=["rng_state_change"]),
        }
        return sigs.get(op_name, FunctionSignature(f"random.{op_name}"))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        return self._dispatch(op_name, *operand_types)

    def _transfer_random(self) -> TransferResult:
        return TransferResult(
            _make_float(Interval(0.0, 1.0)),
            side_effects=["rng_state_change"])

    def _transfer_randint(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        lo = ai.lo
        hi = bi.hi
        return TransferResult(
            _make_int(Interval(lo, hi)),
            side_effects=["rng_state_change"])

    def _transfer_randrange(self, *args: RefinementType) -> TransferResult:
        if len(args) >= 2:
            ai = _ensure_interval(args[0])
            bi = _ensure_interval(args[1])
            lo = ai.lo
            hi = _sub_opt(bi.hi, 1) if bi.hi is not None else None
            return TransferResult(
                _make_int(Interval(lo, hi)),
                ["ValueError"],
                side_effects=["rng_state_change"])
        if len(args) == 1:
            ai = _ensure_interval(args[0])
            return TransferResult(
                _make_int(Interval(0, _sub_opt(ai.hi, 1) if ai.hi is not None else None)),
                ["ValueError"],
                side_effects=["rng_state_change"])
        return TransferResult(_make_int(), ["ValueError"],
                              side_effects=["rng_state_change"])

    def _transfer_choice(self, seq: RefinementType) -> TransferResult:
        elem = seq.extra.get("element_type", RefinementType("any"))
        return TransferResult(elem, ["IndexError"],
                              side_effects=["rng_state_change"])

    def _transfer_choices(self, population: RefinementType,
                           *args: RefinementType) -> TransferResult:
        elem = population.extra.get("element_type", RefinementType("any"))
        return TransferResult(_make_list(elem),
                              side_effects=["rng_state_change"])

    def _transfer_sample(self, population: RefinementType,
                          k: RefinementType) -> TransferResult:
        elem = population.extra.get("element_type", RefinementType("any"))
        return TransferResult(_make_list(elem), ["ValueError"],
                              side_effects=["rng_state_change"])

    def _transfer_shuffle(self, x: RefinementType) -> TransferResult:
        return TransferResult(_make_none(),
                              side_effects=["mutates_arg0", "rng_state_change"])

    def _transfer_uniform(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        ai, bi = _ensure_interval(a), _ensure_interval(b)
        lo = _min_opt(ai.lo, bi.lo)
        hi = _max_opt(ai.hi, bi.hi)
        return TransferResult(_make_float(Interval(lo, hi)),
                              side_effects=["rng_state_change"])

    def _transfer_gauss(self, mu: RefinementType,
                         sigma: RefinementType) -> TransferResult:
        return TransferResult(_make_float(),
                              side_effects=["rng_state_change"])

    def _transfer_normalvariate(self, mu: RefinementType,
                                 sigma: RefinementType) -> TransferResult:
        return TransferResult(_make_float(),
                              side_effects=["rng_state_change"])

    def _transfer_lognormvariate(self, mu: RefinementType,
                                  sigma: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)),
                              side_effects=["rng_state_change"])

    def _transfer_expovariate(self, lambd: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)),
                              side_effects=["rng_state_change"])

    def _transfer_vonmisesvariate(self, mu: RefinementType,
                                   kappa: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, 2 * math.pi)),
                              side_effects=["rng_state_change"])

    def _transfer_gammavariate(self, alpha: RefinementType,
                                beta: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)),
                              side_effects=["rng_state_change"])

    def _transfer_betavariate(self, alpha: RefinementType,
                               beta: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, 1)),
                              side_effects=["rng_state_change"])

    def _transfer_paretovariate(self, alpha: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(1, None)),
                              side_effects=["rng_state_change"])

    def _transfer_weibullvariate(self, alpha: RefinementType,
                                  beta: RefinementType) -> TransferResult:
        return TransferResult(_make_float(Interval(0, None)),
                              side_effects=["rng_state_change"])

    def _transfer_triangular(self, *args: RefinementType) -> TransferResult:
        if len(args) >= 2:
            ai, bi = _ensure_interval(args[0]), _ensure_interval(args[1])
            lo = _min_opt(ai.lo, bi.lo)
            hi = _max_opt(ai.hi, bi.hi)
            return TransferResult(_make_float(Interval(lo, hi)),
                                  side_effects=["rng_state_change"])
        return TransferResult(_make_float(Interval(0, 1)),
                              side_effects=["rng_state_change"])

    def _transfer_seed(self, *args: RefinementType) -> TransferResult:
        return TransferResult(_make_none(),
                              side_effects=["rng_state_change"])

    def _transfer_getstate(self) -> TransferResult:
        return TransferResult(RefinementType("tuple"))

    def _transfer_setstate(self, state: RefinementType) -> TransferResult:
        return TransferResult(_make_none(),
                              side_effects=["rng_state_change"])


# ---------------------------------------------------------------------------
# OperatorModuleModels
# ---------------------------------------------------------------------------

class OperatorModuleModels(NumericModelBase):

    def __init__(self) -> None:
        self._int_model = IntegerModels()
        self._float_model = FloatModels()

    @property
    def model_name(self) -> str:
        return "operator"

    @property
    def supported_operations(self) -> List[str]:
        return [
            "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "eq", "ne", "lt", "le", "gt", "ge",
            "and_", "or_", "xor", "invert", "lshift", "rshift",
            "neg", "pos", "abs",
            "not_", "truth", "is_", "is_not",
            "index", "concat", "contains", "countOf", "indexOf",
            "itemgetter", "attrgetter", "methodcaller",
            "iadd", "isub", "imul", "itruediv", "ifloordiv", "imod", "ipow",
            "iand", "ior", "ixor", "ilshift", "irshift",
        ]

    def get_signature(self, op_name: str) -> FunctionSignature:
        dunder = "__" + op_name.rstrip("_") + "__"
        arith_ops = {"add", "sub", "mul", "truediv", "floordiv", "mod", "pow"}
        cmp_ops = {"eq", "ne", "lt", "le", "gt", "ge"}
        bitwise_ops = {"and_", "or_", "xor", "lshift", "rshift"}
        stripped = op_name.rstrip("_")

        if stripped in arith_ops or stripped in cmp_ops or stripped in bitwise_ops:
            return self._int_model.get_signature(dunder)
        if op_name == "invert":
            return self._int_model.get_signature("__invert__")
        if op_name in ("neg", "pos", "abs"):
            return self._int_model.get_signature(f"__{op_name}__")

        sigs: Dict[str, FunctionSignature] = {
            "not_": FunctionSignature("operator.not_", [RefinementType("any")],
                                       _make_bool(), is_pure=True),
            "truth": FunctionSignature("operator.truth", [RefinementType("any")],
                                        _make_bool(), is_pure=True),
            "is_": FunctionSignature("operator.is_",
                                      [RefinementType("any"), RefinementType("any")],
                                      _make_bool(), is_pure=True),
            "is_not": FunctionSignature("operator.is_not",
                                         [RefinementType("any"), RefinementType("any")],
                                         _make_bool(), is_pure=True),
            "index": FunctionSignature("operator.index",
                                        [RefinementType("any")], _make_int(),
                                        may_raise=["TypeError"], is_pure=True),
            "concat": FunctionSignature("operator.concat",
                                         [RefinementType("any"), RefinementType("any")],
                                         RefinementType("any"),
                                         may_raise=["TypeError"], is_pure=True),
            "contains": FunctionSignature("operator.contains",
                                           [RefinementType("any"), RefinementType("any")],
                                           _make_bool(), is_pure=True),
            "countOf": FunctionSignature("operator.countOf",
                                          [RefinementType("any"), RefinementType("any")],
                                          _make_int(Interval(0, None)), is_pure=True),
            "indexOf": FunctionSignature("operator.indexOf",
                                          [RefinementType("any"), RefinementType("any")],
                                          _make_int(Interval(0, None)),
                                          may_raise=["ValueError"], is_pure=True),
            "itemgetter": FunctionSignature("operator.itemgetter",
                                             [RefinementType("any")],
                                             RefinementType("callable"), is_pure=True),
            "attrgetter": FunctionSignature("operator.attrgetter",
                                             [_make_str()],
                                             RefinementType("callable"), is_pure=True),
            "methodcaller": FunctionSignature("operator.methodcaller",
                                               [_make_str()],
                                               RefinementType("callable"), is_pure=True),
        }
        # in-place operators
        if op_name.startswith("i") and op_name[1:] in arith_ops:
            base = op_name[1:]
            return FunctionSignature(
                f"operator.{op_name}",
                [RefinementType("any"), RefinementType("any")],
                RefinementType("any"),
                is_pure=False,
                side_effects=["mutates_arg0"],
            )
        return sigs.get(op_name, FunctionSignature(f"operator.{op_name}", is_pure=True))

    def apply_transfer(self, op_name: str,
                       *operand_types: RefinementType) -> TransferResult:
        arith_map = {
            "add": "__add__", "sub": "__sub__", "mul": "__mul__",
            "truediv": "__truediv__", "floordiv": "__floordiv__",
            "mod": "__mod__", "pow": "__pow__",
        }
        cmp_map = {
            "eq": "__eq__", "ne": "__ne__", "lt": "__lt__",
            "le": "__le__", "gt": "__gt__", "ge": "__ge__",
        }
        bit_map = {
            "and_": "__and__", "or_": "__or__", "xor": "__xor__",
            "lshift": "__lshift__", "rshift": "__rshift__",
        }
        unary_map = {
            "neg": "__neg__", "pos": "__pos__", "abs": "__abs__",
            "invert": "__invert__",
        }

        stripped = op_name.rstrip("_")
        if stripped in arith_map:
            dunder = arith_map[stripped]
            if operand_types and operand_types[0].base_type == "float":
                return self._float_model.apply_transfer(dunder, *operand_types)
            return self._int_model.apply_transfer(dunder, *operand_types)
        if stripped in cmp_map:
            dunder = cmp_map[stripped]
            return self._int_model.apply_transfer(dunder, *operand_types)
        if op_name in bit_map:
            return self._int_model.apply_transfer(bit_map[op_name], *operand_types)
        if op_name in unary_map:
            return self._int_model.apply_transfer(unary_map[op_name], *operand_types)

        return self._dispatch(op_name, *operand_types)

    def _transfer_not_(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton():
            return TransferResult(_make_bool(not bool(ai.lo)))
        return TransferResult(_make_bool())

    def _transfer_truth(self, a: RefinementType) -> TransferResult:
        ai = _ensure_interval(a)
        if ai.is_singleton() and ai.lo == 0:
            return TransferResult(_make_bool(False))
        if ai.lo is not None and ai.lo > 0:
            return TransferResult(_make_bool(True))
        return TransferResult(_make_bool())

    def _transfer_is_(self, a: RefinementType,
                       b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_is_not(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_index(self, a: RefinementType) -> TransferResult:
        if a.base_type not in ("int", "bool"):
            return TransferResult(_make_int(), ["TypeError"])
        return TransferResult(_make_int(_ensure_interval(a)))

    def _transfer_concat(self, a: RefinementType,
                          b: RefinementType) -> TransferResult:
        if a.base_type == "str" and b.base_type == "str":
            return TransferResult(_make_str())
        if a.base_type == "list":
            return TransferResult(_make_list())
        return TransferResult(RefinementType("any"), ["TypeError"])

    def _transfer_contains(self, a: RefinementType,
                            b: RefinementType) -> TransferResult:
        return TransferResult(_make_bool())

    def _transfer_countOf(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(_make_int(Interval(0, None)))

    def _transfer_indexOf(self, a: RefinementType,
                           b: RefinementType) -> TransferResult:
        return TransferResult(_make_int(Interval(0, None)), ["ValueError"])

    def _transfer_itemgetter(self, *args: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("callable"))

    def _transfer_attrgetter(self, *args: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("callable"))

    def _transfer_methodcaller(self, *args: RefinementType) -> TransferResult:
        return TransferResult(RefinementType("callable"))


# ---------------------------------------------------------------------------
# NumericRefinementTransfer
# ---------------------------------------------------------------------------

class NumericRefinementTransfer:
    """Orchestrates forward and backward transfer for numeric operations."""

    def __init__(self) -> None:
        self._models: Dict[str, NumericModelBase] = {
            "int": IntegerModels(),
            "float": FloatModels(),
            "complex": ComplexModels(),
            "bool": BoolModels(),
            "Decimal": DecimalModels(),
            "Fraction": FractionModels(),
            "math": MathModuleModels(),
            "cmath": CmathModuleModels(),
            "statistics": StatisticsModuleModels(),
            "random": RandomModuleModels(),
            "operator": OperatorModuleModels(),
            "NormalDist": NormalDistModels(),
        }

    def get_model(self, name: str) -> Optional[NumericModelBase]:
        return self._models.get(name)

    def forward_transfer(self, op: str,
                         *operands: RefinementType) -> TransferResult:
        """Dispatch to the appropriate model based on operand types."""
        if not operands:
            # Module-level function: determine model from op prefix
            if "." in op:
                mod, func = op.rsplit(".", 1)
                model = self._models.get(mod)
                if model:
                    return model.apply_transfer(func)
            return TransferResult(RefinementType("any"))

        primary = operands[0]
        bt = primary.base_type

        # Direct match
        model = self._models.get(bt)
        if model:
            return model.apply_transfer(op, *operands)

        # Try module prefix
        if "." in op:
            mod, func = op.rsplit(".", 1)
            model = self._models.get(mod)
            if model:
                return model.apply_transfer(func, *operands)

        # Numeric promotion: bool → int → float
        if bt == "bool":
            return self._models["bool"].apply_transfer(op, *operands)
        if bt in ("int", "float"):
            return self._models[bt].apply_transfer(op, *operands)

        return TransferResult(RefinementType("any"))

    def backward_transfer(self, op: str, result: RefinementType,
                          *operands: RefinementType) -> Dict[str, RefinementType]:
        """Given a desired result type, narrow operands backward."""
        narrowings: Dict[str, RefinementType] = {}

        if op == "__add__" and len(operands) == 2:
            # result = a + b  →  a = result - b,  b = result - a
            ri = _ensure_interval(result)
            a_iv, b_iv = _ensure_interval(operands[0]), _ensure_interval(operands[1])
            new_a = _IA.sub(ri, b_iv).meet(a_iv)
            new_b = _IA.sub(ri, a_iv).meet(b_iv)
            narrowings["__arg0__"] = RefinementType(operands[0].base_type, None, new_a)
            narrowings["__arg1__"] = RefinementType(operands[1].base_type, None, new_b)

        elif op == "__sub__" and len(operands) == 2:
            ri = _ensure_interval(result)
            a_iv, b_iv = _ensure_interval(operands[0]), _ensure_interval(operands[1])
            new_a = _IA.add(ri, b_iv).meet(a_iv)
            new_b = _IA.sub(a_iv, ri).meet(b_iv)
            narrowings["__arg0__"] = RefinementType(operands[0].base_type, None, new_a)
            narrowings["__arg1__"] = RefinementType(operands[1].base_type, None, new_b)

        elif op == "__mul__" and len(operands) == 2:
            ri = _ensure_interval(result)
            a_iv, b_iv = _ensure_interval(operands[0]), _ensure_interval(operands[1])
            if not b_iv.contains_zero():
                new_a = _IA.div(ri, b_iv).meet(a_iv)
                narrowings["__arg0__"] = RefinementType(operands[0].base_type, None, new_a)
            if not a_iv.contains_zero():
                new_b = _IA.div(ri, a_iv).meet(b_iv)
                narrowings["__arg1__"] = RefinementType(operands[1].base_type, None, new_b)

        elif op == "__truediv__" and len(operands) == 2:
            ri = _ensure_interval(result)
            a_iv, b_iv = _ensure_interval(operands[0]), _ensure_interval(operands[1])
            if not b_iv.contains_zero():
                new_a = _IA.mul(ri, b_iv).meet(a_iv)
                narrowings["__arg0__"] = RefinementType(operands[0].base_type, None, new_a)

        elif op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
            fwd = self.forward_transfer(op, *operands)
            narrowings.update(fwd.narrowings)

        elif op == "sqrt" and len(operands) == 1:
            ri = _ensure_interval(result)
            # x = result² clamped to [0, ∞)
            if ri.lo is not None and ri.hi is not None:
                new_x = Interval(ri.lo ** 2, ri.hi ** 2)
                orig = _ensure_interval(operands[0])
                narrowings["__arg0__"] = RefinementType(
                    operands[0].base_type, None, new_x.meet(orig))

        return narrowings

    def meet_transfer(self, constraint: Predicate,
                      value: RefinementType) -> RefinementType:
        """Narrow *value* by applying *constraint*."""
        iv = _ensure_interval(value)

        if constraint.kind == ">=" and constraint.args:
            bound = float(constraint.args[0])
            new_iv = iv.meet(Interval(bound, None))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        if constraint.kind == ">" and constraint.args:
            bound = float(constraint.args[0])
            eps = 1 if value.base_type == "int" else 1e-300
            new_iv = iv.meet(Interval(bound + eps, None))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        if constraint.kind == "<=" and constraint.args:
            bound = float(constraint.args[0])
            new_iv = iv.meet(Interval(None, bound))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        if constraint.kind == "<" and constraint.args:
            bound = float(constraint.args[0])
            eps = 1 if value.base_type == "int" else 1e-300
            new_iv = iv.meet(Interval(None, bound - eps))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        if constraint.kind == "==" and constraint.args:
            bound = float(constraint.args[0])
            new_iv = iv.meet(Interval(bound, bound))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        if constraint.kind == "!=" and constraint.args:
            # Cannot easily refine interval for !=, keep predicate
            return RefinementType(value.base_type, constraint, iv, value.nullity)

        if constraint.kind == "in_range" and len(constraint.args) >= 2:
            lo_c, hi_c = float(constraint.args[0]), float(constraint.args[1])
            new_iv = iv.meet(Interval(lo_c, hi_c))
            return RefinementType(value.base_type, constraint, new_iv, value.nullity)

        return RefinementType(value.base_type, constraint, iv, value.nullity)


# ---------------------------------------------------------------------------
# NumericPatternDetector
# ---------------------------------------------------------------------------

@dataclass
class _Assignment:
    variable: str
    op: str
    operands: List[str]
    value: Optional[float] = None

@dataclass
class _Guard:
    variable: str
    op: str  # "<", "<=", ">", ">=", "==", "!="
    bound: Optional[float] = None
    other_var: Optional[str] = None


class NumericPatternDetector:
    """Detects common numeric patterns in assignment / guard sequences."""

    def detect_counter(self, assignments: List[Dict[str, Any]]
                       ) -> Optional[CounterPattern]:
        """Detect i = i + step or i = i - step patterns."""
        for a in assignments:
            var = a.get("variable", "")
            op = a.get("op", "")
            operands = a.get("operands", [])
            if op in ("__add__", "+") and len(operands) == 2:
                if operands[0] == var and _is_const(operands[1]):
                    step = float(operands[1])
                    init = float(a.get("init", 0))
                    direction = "up" if step > 0 else "down"
                    bound_val = a.get("bound", None)
                    bound = float(bound_val) if bound_val is not None else None
                    return CounterPattern(var, init, step, direction, bound)
                if operands[1] == var and _is_const(operands[0]):
                    step = float(operands[0])
                    init = float(a.get("init", 0))
                    direction = "up" if step > 0 else "down"
                    return CounterPattern(var, init, step, direction)
            if op in ("__sub__", "-") and len(operands) == 2:
                if operands[0] == var and _is_const(operands[1]):
                    step = -float(operands[1])
                    init = float(a.get("init", 0))
                    direction = "up" if step > 0 else "down"
                    return CounterPattern(var, init, step, direction)
        return None

    def detect_accumulator(self, assignments: List[Dict[str, Any]]
                           ) -> Optional[AccumulatorPattern]:
        """Detect s = s + x or p = p * x patterns."""
        for a in assignments:
            var = a.get("variable", "")
            op = a.get("op", "")
            operands = a.get("operands", [])
            if op in ("__add__", "+") and len(operands) == 2:
                if operands[0] == var and operands[1] != var:
                    init = float(a.get("init", 0))
                    return AccumulatorPattern(var, init, "add", operands[1])
                if operands[1] == var and operands[0] != var:
                    init = float(a.get("init", 0))
                    return AccumulatorPattern(var, init, "add", operands[0])
            if op in ("__mul__", "*") and len(operands) == 2:
                if operands[0] == var and operands[1] != var:
                    init = float(a.get("init", 1))
                    return AccumulatorPattern(var, init, "mul", operands[1])
                if operands[1] == var and operands[0] != var:
                    init = float(a.get("init", 1))
                    return AccumulatorPattern(var, init, "mul", operands[0])
        return None

    def detect_min_max_tracking(self, assignments: List[Dict[str, Any]]
                                ) -> Optional[MinMaxPattern]:
        """Detect m = min(m, x) or m = max(m, x)."""
        for a in assignments:
            var = a.get("variable", "")
            op = a.get("op", "")
            operands = a.get("operands", [])
            if op == "min" and len(operands) == 2:
                if operands[0] == var:
                    return MinMaxPattern(var, "min", operands[1])
                if operands[1] == var:
                    return MinMaxPattern(var, "min", operands[0])
            if op == "max" and len(operands) == 2:
                if operands[0] == var:
                    return MinMaxPattern(var, "max", operands[1])
                if operands[1] == var:
                    return MinMaxPattern(var, "max", operands[0])
            # conditional pattern: if x < m: m = x
            if op == "conditional_assign":
                cond = a.get("condition", {})
                cond_op = cond.get("op", "")
                cond_operands = cond.get("operands", [])
                if cond_op in ("<", "__lt__") and len(cond_operands) == 2:
                    if cond_operands[0] != var and cond_operands[1] == var:
                        src = cond_operands[0]
                        return MinMaxPattern(var, "min", src)
                if cond_op in (">", "__gt__") and len(cond_operands) == 2:
                    if cond_operands[0] != var and cond_operands[1] == var:
                        src = cond_operands[0]
                        return MinMaxPattern(var, "max", src)
        return None

    def detect_average(self, assignments: List[Dict[str, Any]]
                       ) -> Optional[AveragePattern]:
        """Detect sum/count pattern for computing averages."""
        sum_var: Optional[str] = None
        count_var: Optional[str] = None
        result_var: Optional[str] = None
        for a in assignments:
            var = a.get("variable", "")
            op = a.get("op", "")
            operands = a.get("operands", [])
            if op in ("__add__", "+") and len(operands) == 2:
                if operands[0] == var and not _is_const(operands[1]):
                    sum_var = var
                elif operands[1] == var and not _is_const(operands[0]):
                    sum_var = var
                elif operands[0] == var and _is_const(operands[1]):
                    try:
                        if float(operands[1]) == 1:
                            count_var = var
                    except (ValueError, TypeError):
                        pass
            if op in ("__truediv__", "/") and len(operands) == 2:
                if sum_var and count_var:
                    if operands[0] == sum_var and operands[1] == count_var:
                        result_var = var
                    elif operands[0] == sum_var:
                        result_var = var
        if sum_var and count_var:
            return AveragePattern(sum_var, count_var, result_var)
        return None

    def detect_bounds_check(self, guards: List[Dict[str, Any]]
                            ) -> Optional[BoundsCheckPattern]:
        """Detect lower <= x < upper guard patterns."""
        vars_to_bounds: Dict[str, Dict[str, Optional[float]]] = {}
        for g in guards:
            var = g.get("variable", "")
            op = g.get("op", "")
            bound = g.get("bound")
            if var not in vars_to_bounds:
                vars_to_bounds[var] = {"lower": None, "upper": None}
            if op in (">=", ">"):
                bval = float(bound) if bound is not None else None
                if op == ">" and bval is not None:
                    bval += 1
                vars_to_bounds[var]["lower"] = bval
            elif op in ("<=", "<"):
                bval = float(bound) if bound is not None else None
                if op == "<" and bval is not None:
                    bval -= 1
                vars_to_bounds[var]["upper"] = bval
        for var, bnds in vars_to_bounds.items():
            if bnds["lower"] is not None or bnds["upper"] is not None:
                return BoundsCheckPattern(
                    var, bnds["lower"], bnds["upper"], checked_before_use=True)
        return None

    def detect_division_guard(self, guards: List[Dict[str, Any]],
                              divisions: List[Dict[str, Any]]
                              ) -> Optional[DivisionGuardPattern]:
        """Detect if x != 0: ... x / y patterns."""
        guarded_vars: Dict[str, str] = {}
        for g in guards:
            var = g.get("variable", "")
            op = g.get("op", "")
            bound = g.get("bound")
            if op == "!=" and bound is not None and float(bound) == 0:
                guarded_vars[var] = "!= 0"
            elif op == ">" and bound is not None and float(bound) == 0:
                guarded_vars[var] = "> 0"
            elif op == "<" and bound is not None and float(bound) == 0:
                guarded_vars[var] = "< 0"

        for d in divisions:
            divisor = d.get("divisor", "")
            div_op = d.get("op", "/")
            if divisor in guarded_vars:
                return DivisionGuardPattern(divisor, guarded_vars[divisor], div_op)
        return None


def _is_const(s: Any) -> bool:
    """Check if a string represents a numeric constant."""
    if isinstance(s, (int, float)):
        return True
    if isinstance(s, str):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    return False


# ---------------------------------------------------------------------------
# ModelRegistry — central lookup for all models
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Registry that maps module / type names to their NumericModelBase."""

    def __init__(self) -> None:
        self._models: Dict[str, NumericModelBase] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register("int", IntegerModels())
        self.register("float", FloatModels())
        self.register("complex", ComplexModels())
        self.register("bool", BoolModels())
        self.register("Decimal", DecimalModels())
        self.register("Fraction", FractionModels())
        self.register("math", MathModuleModels())
        self.register("cmath", CmathModuleModels())
        self.register("statistics", StatisticsModuleModels())
        self.register("random", RandomModuleModels())
        self.register("operator", OperatorModuleModels())
        self.register("NormalDist", NormalDistModels())

    def register(self, name: str, model: NumericModelBase) -> None:
        self._models[name] = model

    def lookup(self, name: str) -> Optional[NumericModelBase]:
        return self._models.get(name)

    def resolve_operation(self, qualified: str
                          ) -> Tuple[Optional[NumericModelBase], str]:
        """Resolve 'math.sqrt' → (MathModuleModels, 'sqrt')."""
        if "." in qualified:
            mod, func = qualified.rsplit(".", 1)
            model = self.lookup(mod)
            return model, func
        # Unqualified: try all models
        for model in self._models.values():
            if qualified in model.supported_operations:
                return model, qualified
        return None, qualified

    def all_signatures(self) -> Dict[str, FunctionSignature]:
        result: Dict[str, FunctionSignature] = {}
        for name, model in self._models.items():
            for op in model.supported_operations:
                key = f"{name}.{op}"
                result[key] = model.get_signature(op)
        return result

    def check_all_preconditions(self, qualified: str,
                                *operand_types: RefinementType
                                ) -> List[str]:
        model, func = self.resolve_operation(qualified)
        if model is None:
            return [f"Unknown operation: {qualified}"]
        return model.check_precondition(func, *operand_types)


# ---------------------------------------------------------------------------
# NumericTypeWidener — widening strategies
# ---------------------------------------------------------------------------

class NumericTypeWidener:
    """Widening strategies for numeric refinement types at loop heads."""

    def __init__(self, max_iterations: int = 10) -> None:
        self.max_iterations = max_iterations

    def widen_type(self, old: RefinementType,
                   new: RefinementType) -> RefinementType:
        if old.base_type != new.base_type:
            return RefinementType("any")
        iv_old = _ensure_interval(old)
        iv_new = _ensure_interval(new)
        iv_wide = iv_old.widen(iv_new)
        n = _join_nullity(old.nullity, new.nullity)
        return RefinementType(old.base_type, None, iv_wide, n)

    def narrow_type(self, wide: RefinementType,
                    precise: RefinementType) -> RefinementType:
        if wide.base_type != precise.base_type and wide.base_type != "any":
            return wide
        iv_wide = _ensure_interval(wide)
        iv_prec = _ensure_interval(precise)
        iv_nar = iv_wide.narrow(iv_prec)
        return RefinementType(
            precise.base_type if wide.base_type == "any" else wide.base_type,
            precise.predicate, iv_nar, precise.nullity)

    def should_widen(self, iteration: int) -> bool:
        return iteration >= self.max_iterations

    def widen_sequence(self, types: List[RefinementType]) -> RefinementType:
        if not types:
            return RefinementType("never")
        result = types[0]
        for i, t in enumerate(types[1:], 1):
            if self.should_widen(i):
                result = self.widen_type(result, t)
            else:
                result = result.join(t)
        return result


# ---------------------------------------------------------------------------
# NumericExceptionAnalyzer
# ---------------------------------------------------------------------------

class NumericExceptionAnalyzer:
    """Determines which numeric exceptions are possible / impossible."""

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    def analyze_operation(self, op: str,
                          *operand_types: RefinementType
                          ) -> Dict[str, str]:
        """Return {exception_name: 'possible' | 'impossible' | 'certain'}."""
        model, func = self._registry.resolve_operation(op)
        if model is None:
            return {}
        sig = model.get_signature(func)
        result: Dict[str, str] = {}
        for exc in sig.may_raise:
            result[exc] = "possible"

        violations = model.check_precondition(func, *operand_types)
        if violations:
            for exc in sig.may_raise:
                result[exc] = "certain"
        else:
            # Check if preconditions are definitely satisfied
            all_sat = True
            for pre in sig.preconditions:
                for ot in operand_types:
                    iv = _ensure_interval(ot)
                    if pre.kind == "!=" and pre.args:
                        if iv.contains(float(pre.args[0])):
                            all_sat = False
                    elif pre.kind == ">=" and pre.args:
                        if iv.lo is None or iv.lo < float(pre.args[0]):
                            all_sat = False
                    elif pre.kind == ">" and pre.args:
                        if iv.lo is None or iv.lo <= float(pre.args[0]):
                            all_sat = False
                    elif pre.kind == "in_range" and len(pre.args) >= 2:
                        lo_p, hi_p = float(pre.args[0]), float(pre.args[1])
                        if iv.lo is None or iv.lo < lo_p or iv.hi is None or iv.hi > hi_p:
                            all_sat = False
            if all_sat:
                for exc in sig.may_raise:
                    result[exc] = "impossible"

        return result

    def safe_division_check(self, divisor: RefinementType) -> str:
        iv = _ensure_interval(divisor)
        if iv.is_singleton() and iv.lo == 0:
            return "certain"
        if not iv.contains_zero():
            return "impossible"
        return "possible"

    def safe_sqrt_check(self, operand: RefinementType) -> str:
        iv = _ensure_interval(operand)
        if iv.is_non_negative():
            return "impossible"
        if iv.hi is not None and iv.hi < 0:
            return "certain"
        return "possible"

    def safe_log_check(self, operand: RefinementType) -> str:
        iv = _ensure_interval(operand)
        if iv.is_positive():
            return "impossible"
        if iv.hi is not None and iv.hi <= 0:
            return "certain"
        return "possible"


# ---------------------------------------------------------------------------
# NumericPromotionRules
# ---------------------------------------------------------------------------

class NumericPromotionRules:
    """Models Python's numeric type promotion (tower)."""

    _TOWER: Dict[str, int] = {
        "bool": 0, "int": 1, "float": 2, "complex": 3,
        "Decimal": 2, "Fraction": 2,
    }

    def promote(self, a: RefinementType,
                b: RefinementType) -> Tuple[RefinementType, RefinementType]:
        ra = self._TOWER.get(a.base_type, -1)
        rb = self._TOWER.get(b.base_type, -1)
        if ra == rb:
            return a, b
        if ra < rb:
            return self._coerce(a, b.base_type), b
        return a, self._coerce(b, a.base_type)

    def _coerce(self, value: RefinementType, target: str) -> RefinementType:
        iv = _ensure_interval(value)
        return RefinementType(target, value.predicate, iv, value.nullity)

    def common_type(self, *types: RefinementType) -> str:
        if not types:
            return "any"
        best = types[0].base_type
        best_rank = self._TOWER.get(best, -1)
        for t in types[1:]:
            r = self._TOWER.get(t.base_type, -1)
            if r > best_rank:
                best = t.base_type
                best_rank = r
        return best

    def is_numeric(self, rt: RefinementType) -> bool:
        return rt.base_type in self._TOWER

    def is_integral(self, rt: RefinementType) -> bool:
        return rt.base_type in ("bool", "int")

    def is_real(self, rt: RefinementType) -> bool:
        return rt.base_type in ("bool", "int", "float", "Decimal", "Fraction")

    def can_compare(self, a: RefinementType, b: RefinementType) -> bool:
        if a.base_type == "complex" or b.base_type == "complex":
            return False
        return self.is_numeric(a) and self.is_numeric(b)

    def mixed_arithmetic_type(self, a: RefinementType,
                              b: RefinementType) -> str:
        # Decimal + float → TypeError in Python
        if {a.base_type, b.base_type} == {"Decimal", "float"}:
            return "error"
        return self.common_type(a, b)


# ---------------------------------------------------------------------------
# Convenience: get_all_models
# ---------------------------------------------------------------------------

def get_all_models() -> Dict[str, NumericModelBase]:
    """Return a dictionary of all numeric models."""
    return {
        "int": IntegerModels(),
        "float": FloatModels(),
        "complex": ComplexModels(),
        "bool": BoolModels(),
        "Decimal": DecimalModels(),
        "Fraction": FractionModels(),
        "math": MathModuleModels(),
        "cmath": CmathModuleModels(),
        "statistics": StatisticsModuleModels(),
        "random": RandomModuleModels(),
        "operator": OperatorModuleModels(),
        "NormalDist": NormalDistModels(),
    }


def create_transfer_engine() -> NumericRefinementTransfer:
    """Create a fully configured transfer engine."""
    return NumericRefinementTransfer()


def create_registry() -> ModelRegistry:
    """Create a fully configured model registry."""
    return ModelRegistry()
