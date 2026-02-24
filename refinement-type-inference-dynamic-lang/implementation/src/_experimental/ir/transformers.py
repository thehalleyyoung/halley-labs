from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union
)
from enum import Enum, auto
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TypeTag(Enum):
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()
    NONE = auto()
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    CALLABLE = auto()
    OBJECT = auto()
    ANY = auto()
    UNKNOWN = auto()
    BYTES = auto()
    COMPLEX = auto()


class NullityTag(Enum):
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    lo: Optional[float] = None  # None means -inf
    hi: Optional[float] = None  # None means +inf

    @staticmethod
    def bottom() -> Interval:
        return Interval(lo=1.0, hi=0.0)

    @staticmethod
    def top() -> Interval:
        return Interval(lo=None, hi=None)

    @staticmethod
    def const(v: float) -> Interval:
        return Interval(lo=v, hi=v)

    @staticmethod
    def zero() -> Interval:
        return Interval.const(0.0)

    def is_bottom(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False

    def is_top(self) -> bool:
        return self.lo is None and self.hi is None

    def is_const(self) -> bool:
        return (
            self.lo is not None
            and self.hi is not None
            and self.lo == self.hi
            and not self.is_bottom()
        )

    def contains(self, v: float) -> bool:
        if self.is_bottom():
            return False
        lo_ok = self.lo is None or self.lo <= v
        hi_ok = self.hi is None or v <= self.hi
        return lo_ok and hi_ok

    def contains_zero(self) -> bool:
        return self.contains(0.0)

    def overlaps(self, other: Interval) -> bool:
        if self.is_bottom() or other.is_bottom():
            return False
        lo = _max_opt(self.lo, other.lo)
        hi = _min_opt(self.hi, other.hi)
        if lo is not None and hi is not None:
            return lo <= hi
        return True

    def join(self, other: Interval) -> Interval:
        if self.is_bottom():
            return Interval(lo=other.lo, hi=other.hi)
        if other.is_bottom():
            return Interval(lo=self.lo, hi=self.hi)
        return Interval(
            lo=_min_opt(self.lo, other.lo),
            hi=_max_opt(self.hi, other.hi),
        )

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        lo = _max_opt(self.lo, other.lo)
        hi = _min_opt(self.hi, other.hi)
        if lo is not None and hi is not None and lo > hi:
            return Interval.bottom()
        return Interval(lo=lo, hi=hi)

    def widen(self, other: Interval) -> Interval:
        if self.is_bottom():
            return Interval(lo=other.lo, hi=other.hi)
        if other.is_bottom():
            return Interval(lo=self.lo, hi=self.hi)
        new_lo = self.lo
        new_hi = self.hi
        if other.lo is not None and self.lo is not None:
            if other.lo < self.lo:
                new_lo = None
        elif other.lo is None:
            new_lo = None
        if other.hi is not None and self.hi is not None:
            if other.hi > self.hi:
                new_hi = None
        elif other.hi is None:
            new_hi = None
        return Interval(lo=new_lo, hi=new_hi)

    def narrow(self, other: Interval) -> Interval:
        if self.is_bottom():
            return Interval.bottom()
        if other.is_bottom():
            return Interval.bottom()
        new_lo = self.lo
        new_hi = self.hi
        if self.lo is None and other.lo is not None:
            new_lo = other.lo
        if self.hi is None and other.hi is not None:
            new_hi = other.hi
        if new_lo is not None and new_hi is not None and new_lo > new_hi:
            return Interval.bottom()
        return Interval(lo=new_lo, hi=new_hi)

    def negate(self) -> Interval:
        if self.is_bottom():
            return Interval.bottom()
        new_lo = -self.hi if self.hi is not None else None
        new_hi = -self.lo if self.lo is not None else None
        return Interval(lo=new_lo, hi=new_hi)

    def abs_interval(self) -> Interval:
        if self.is_bottom():
            return Interval.bottom()
        if self.lo is not None and self.lo >= 0:
            return Interval(lo=self.lo, hi=self.hi)
        if self.hi is not None and self.hi <= 0:
            return self.negate()
        pos_hi = self.hi
        neg_hi = -self.lo if self.lo is not None else None
        return Interval(lo=0.0, hi=_max_opt(pos_hi, neg_hi))

    def __repr__(self) -> str:
        lo_s = "-∞" if self.lo is None else str(self.lo)
        hi_s = "+∞" if self.hi is None else str(self.hi)
        return f"[{lo_s}, {hi_s}]"


def _min_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return a
    if b is None:
        return b
    return min(a, b)


def _max_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return a
    if b is None:
        return b
    return max(a, b)


# ---------------------------------------------------------------------------
# Interval arithmetic helpers
# ---------------------------------------------------------------------------

def _interval_add(a: Interval, b: Interval) -> Interval:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom()
    lo = (a.lo + b.lo) if (a.lo is not None and b.lo is not None) else None
    hi = (a.hi + b.hi) if (a.hi is not None and b.hi is not None) else None
    return Interval(lo=lo, hi=hi)


def _interval_sub(a: Interval, b: Interval) -> Interval:
    return _interval_add(a, b.negate())


def _interval_mul(a: Interval, b: Interval) -> Interval:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom()
    a_lo = a.lo if a.lo is not None else -math.inf
    a_hi = a.hi if a.hi is not None else math.inf
    b_lo = b.lo if b.lo is not None else -math.inf
    b_hi = b.hi if b.hi is not None else math.inf
    products = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
    products_clean = [p for p in products if not math.isnan(p)]
    if not products_clean:
        return Interval.top()
    lo = min(products_clean)
    hi = max(products_clean)
    res_lo: Optional[float] = lo if not math.isinf(lo) else None
    res_hi: Optional[float] = hi if not math.isinf(hi) else None
    if res_lo is not None and math.isinf(lo) and lo < 0:
        res_lo = None
    if res_hi is not None and math.isinf(hi) and hi > 0:
        res_hi = None
    return Interval(lo=res_lo, hi=res_hi)


def _interval_div(a: Interval, b: Interval) -> Tuple[Interval, bool]:
    """Returns (result_interval, may_raise_ZeroDivisionError)."""
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom(), False
    may_raise = b.contains_zero()
    if b.lo is not None and b.hi is not None and b.lo == 0.0 and b.hi == 0.0:
        return Interval.bottom(), True
    parts: List[Interval] = []
    if b.lo is not None and b.lo > 0:
        parts.append(_div_positive(a, b))
    elif b.hi is not None and b.hi < 0:
        parts.append(_div_negative(a, b))
    else:
        if b.hi is not None and b.hi > 0:
            parts.append(_div_positive(a, Interval(lo=max(b.lo if b.lo is not None else 1.0, 1e-300), hi=b.hi)))
        if b.lo is not None and b.lo < 0:
            parts.append(_div_negative(a, Interval(lo=b.lo, hi=min(b.hi if b.hi is not None else -1e-300, -1e-300))))
        if not parts:
            return Interval.top(), may_raise
    result = parts[0]
    for p in parts[1:]:
        result = result.join(p)
    return result, may_raise


def _div_positive(a: Interval, b: Interval) -> Interval:
    a_lo = a.lo if a.lo is not None else -math.inf
    a_hi = a.hi if a.hi is not None else math.inf
    b_lo = b.lo if b.lo is not None else 1e-300
    b_hi = b.hi if b.hi is not None else math.inf
    if b_lo <= 0:
        b_lo = 1e-300
    vals = [a_lo / b_lo, a_lo / b_hi, a_hi / b_lo, a_hi / b_hi]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return Interval.top()
    lo = min(vals)
    hi = max(vals)
    return Interval(
        lo=lo if not math.isinf(lo) else None,
        hi=hi if not math.isinf(hi) else None,
    )


def _div_negative(a: Interval, b: Interval) -> Interval:
    a_lo = a.lo if a.lo is not None else -math.inf
    a_hi = a.hi if a.hi is not None else math.inf
    b_lo = b.lo if b.lo is not None else -math.inf
    b_hi = b.hi if b.hi is not None else -1e-300
    if b_hi >= 0:
        b_hi = -1e-300
    vals = [a_lo / b_lo, a_lo / b_hi, a_hi / b_lo, a_hi / b_hi]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return Interval.top()
    lo = min(vals)
    hi = max(vals)
    return Interval(
        lo=lo if not math.isinf(lo) else None,
        hi=hi if not math.isinf(hi) else None,
    )


def _interval_mod(a: Interval, b: Interval) -> Tuple[Interval, bool]:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom(), False
    may_raise = b.contains_zero()
    if b.lo is not None and b.hi is not None and b.lo == 0.0 and b.hi == 0.0:
        return Interval.bottom(), True
    if b.lo is not None and b.lo > 0:
        return Interval(lo=0.0, hi=b.hi if b.hi is not None else None), may_raise
    if b.hi is not None and b.hi < 0:
        return Interval(lo=b.lo if b.lo is not None else None, hi=0.0), may_raise
    return Interval.top(), may_raise


def _interval_floordiv(a: Interval, b: Interval) -> Tuple[Interval, bool]:
    result, may_raise = _interval_div(a, b)
    if result.is_bottom():
        return result, may_raise
    lo = math.floor(result.lo) if result.lo is not None else None
    hi = math.floor(result.hi) if result.hi is not None else None
    return Interval(lo=lo, hi=hi), may_raise


def _interval_pow(a: Interval, b: Interval) -> Interval:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom()
    if b.is_const() and b.lo == 0.0:
        return Interval.const(1.0)
    if a.is_const() and a.lo == 0.0:
        if b.lo is not None and b.lo > 0:
            return Interval.const(0.0)
        return Interval.top()
    if a.is_const() and a.lo == 1.0:
        return Interval.const(1.0)
    if b.is_const() and b.lo is not None:
        exp = b.lo
        if exp == int(exp) and abs(exp) <= 10:
            e = int(exp)
            if e >= 0:
                a_lo = a.lo if a.lo is not None else -math.inf
                a_hi = a.hi if a.hi is not None else math.inf
                vals = []
                try:
                    vals.append(a_lo ** e)
                except (OverflowError, ValueError):
                    return Interval.top()
                try:
                    vals.append(a_hi ** e)
                except (OverflowError, ValueError):
                    return Interval.top()
                if e % 2 == 0 and a_lo < 0 < a_hi:
                    vals.append(0.0)
                vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
                if not vals:
                    return Interval.top()
                return Interval(lo=min(vals), hi=max(vals))
    return Interval.top()


def _interval_min(a: Interval, b: Interval) -> Interval:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom()
    return Interval(
        lo=_min_opt(a.lo, b.lo),
        hi=_min_opt(a.hi, b.hi),
    )


def _interval_max(a: Interval, b: Interval) -> Interval:
    if a.is_bottom() or b.is_bottom():
        return Interval.bottom()
    return Interval(
        lo=_max_opt(a.lo, b.lo),
        hi=_max_opt(a.hi, b.hi),
    )


# ---------------------------------------------------------------------------
# Expression types
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    pass


@dataclass
class VarExpr(Expr):
    name: str = ""


@dataclass
class ConstExpr(Expr):
    value: Any = None
    type_tag: TypeTag = TypeTag.ANY


@dataclass
class BinOpExpr(Expr):
    op: str = ""
    left: Expr = field(default_factory=Expr)
    right: Expr = field(default_factory=Expr)


@dataclass
class UnaryOpExpr(Expr):
    op: str = ""
    operand: Expr = field(default_factory=Expr)


@dataclass
class CallExpr(Expr):
    func: Expr = field(default_factory=Expr)
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class AttrExpr(Expr):
    obj: Expr = field(default_factory=Expr)
    attr: str = ""


@dataclass
class SubscriptExpr(Expr):
    obj: Expr = field(default_factory=Expr)
    index: Expr = field(default_factory=Expr)


@dataclass
class CompareExpr(Expr):
    left: Expr = field(default_factory=Expr)
    ops: List[str] = field(default_factory=list)
    comparators: List[Expr] = field(default_factory=list)


@dataclass
class BoolOpExpr(Expr):
    op: str = ""
    values: List[Expr] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IR Node types
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    node_id: int = 0
    label: str = ""
    line: int = 0


@dataclass
class AssignNode(IRNode):
    target: str = ""
    value: Expr = field(default_factory=Expr)
    annotation: Optional[str] = None


@dataclass
class GuardNode(IRNode):
    condition: Expr = field(default_factory=Expr)
    true_branch: Optional[IRNode] = None
    false_branch: Optional[IRNode] = None
    is_negated: bool = False


@dataclass
class PhiNode(IRNode):
    target: str = ""
    sources: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class CallNode(IRNode):
    target: Optional[str] = None
    func: Expr = field(default_factory=Expr)
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class ReturnNode(IRNode):
    value: Optional[Expr] = None


@dataclass
class RaiseNode(IRNode):
    exception: Optional[Expr] = None
    cause: Optional[Expr] = None


@dataclass
class ExceptClause:
    exception_type: Optional[str] = None
    name: Optional[str] = None
    body: List[IRNode] = field(default_factory=list)


@dataclass
class TryExceptNode(IRNode):
    body: List[IRNode] = field(default_factory=list)
    handlers: List[ExceptClause] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)
    finalbody: List[IRNode] = field(default_factory=list)


@dataclass
class WhileNode(IRNode):
    condition: Expr = field(default_factory=Expr)
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)


@dataclass
class ForNode(IRNode):
    target: str = ""
    iter_expr: Expr = field(default_factory=Expr)
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)


@dataclass
class ComprehensionNode(IRNode):
    kind: str = "list"  # list, dict, set, generator
    element: Expr = field(default_factory=Expr)
    key_expr: Optional[Expr] = None  # for dict comp
    generators: List[Tuple[str, Expr, List[Expr]]] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class ContainerOpNode(IRNode):
    container: Expr = field(default_factory=Expr)
    op: str = ""
    args: List[Expr] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class StringOpNode(IRNode):
    string_expr: Expr = field(default_factory=Expr)
    op: str = ""
    args: List[Expr] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class ImportNode(IRNode):
    module: str = ""
    names: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    is_from: bool = False
    is_star: bool = False


@dataclass
class BlockNode(IRNode):
    statements: List[IRNode] = field(default_factory=list)


@dataclass
class FunctionNode(IRNode):
    name: str = ""
    params: List[str] = field(default_factory=list)
    defaults: List[Optional[Expr]] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    is_method: bool = False


@dataclass
class AttributeAccessNode(IRNode):
    obj: Expr = field(default_factory=Expr)
    attr: str = ""
    target: Optional[str] = None


@dataclass
class AttributeSetNode(IRNode):
    obj: Expr = field(default_factory=Expr)
    attr: str = ""
    value: Expr = field(default_factory=Expr)


@dataclass
class BreakNode(IRNode):
    pass


@dataclass
class ContinueNode(IRNode):
    pass


@dataclass
class UnaryOpNode(IRNode):
    op: str = ""
    operand: Expr = field(default_factory=Expr)
    target: Optional[str] = None


@dataclass
class BinaryOpNode(IRNode):
    op: str = ""
    left: Expr = field(default_factory=Expr)
    right: Expr = field(default_factory=Expr)
    target: Optional[str] = None


@dataclass
class CompareNode(IRNode):
    left: Expr = field(default_factory=Expr)
    ops: List[str] = field(default_factory=list)
    comparators: List[Expr] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class BoolOpNode(IRNode):
    op: str = ""
    values: List[Expr] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class SubscriptNode(IRNode):
    obj: Expr = field(default_factory=Expr)
    index: Expr = field(default_factory=Expr)
    target: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract Value / State
# ---------------------------------------------------------------------------

@dataclass
class AbstractValue:
    interval: Interval = field(default_factory=Interval.top)
    type_tags: Set[TypeTag] = field(default_factory=lambda: {TypeTag.ANY})
    nullity: NullityTag = NullityTag.MAYBE_NULL
    string_values: Optional[FrozenSet[str]] = None  # None means unknown
    container_length: Interval = field(default_factory=Interval.top)
    attributes: Dict[str, AbstractValue] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def top() -> AbstractValue:
        return AbstractValue(
            interval=Interval.top(),
            type_tags={TypeTag.ANY},
            nullity=NullityTag.MAYBE_NULL,
            string_values=None,
            container_length=Interval.top(),
        )

    @staticmethod
    def bottom() -> AbstractValue:
        return AbstractValue(
            interval=Interval.bottom(),
            type_tags=set(),
            nullity=NullityTag.DEFINITELY_NOT_NULL,
            string_values=frozenset(),
            container_length=Interval.bottom(),
        )

    @staticmethod
    def from_type(tag: TypeTag) -> AbstractValue:
        nullity = NullityTag.DEFINITELY_NULL if tag == TypeTag.NONE else NullityTag.DEFINITELY_NOT_NULL
        return AbstractValue(
            interval=Interval.top(),
            type_tags={tag},
            nullity=nullity,
        )

    @staticmethod
    def from_const(value: Any) -> AbstractValue:
        if value is None:
            return AbstractValue(
                interval=Interval.bottom(),
                type_tags={TypeTag.NONE},
                nullity=NullityTag.DEFINITELY_NULL,
            )
        if isinstance(value, bool):
            v = 1.0 if value else 0.0
            return AbstractValue(
                interval=Interval.const(v),
                type_tags={TypeTag.BOOL},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        if isinstance(value, int):
            return AbstractValue(
                interval=Interval.const(float(value)),
                type_tags={TypeTag.INT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        if isinstance(value, float):
            return AbstractValue(
                interval=Interval.const(value),
                type_tags={TypeTag.FLOAT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        if isinstance(value, str):
            return AbstractValue(
                interval=Interval.const(float(len(value))),
                type_tags={TypeTag.STR},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                string_values=frozenset({value}),
            )
        if isinstance(value, list):
            return AbstractValue(
                interval=Interval.bottom(),
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval.const(float(len(value))),
            )
        if isinstance(value, dict):
            return AbstractValue(
                interval=Interval.bottom(),
                type_tags={TypeTag.DICT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval.const(float(len(value))),
            )
        if isinstance(value, set):
            return AbstractValue(
                interval=Interval.bottom(),
                type_tags={TypeTag.SET},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval.const(float(len(value))),
            )
        if isinstance(value, tuple):
            return AbstractValue(
                interval=Interval.bottom(),
                type_tags={TypeTag.TUPLE},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval.const(float(len(value))),
            )
        return AbstractValue.top()

    def is_bottom(self) -> bool:
        return len(self.type_tags) == 0

    def is_numeric(self) -> bool:
        return bool(self.type_tags & {TypeTag.INT, TypeTag.FLOAT, TypeTag.BOOL, TypeTag.COMPLEX})

    def is_string(self) -> bool:
        return self.type_tags == {TypeTag.STR}

    def is_container(self) -> bool:
        return bool(self.type_tags & {TypeTag.LIST, TypeTag.DICT, TypeTag.SET, TypeTag.TUPLE})

    def is_none(self) -> bool:
        return self.type_tags == {TypeTag.NONE}

    def join(self, other: AbstractValue) -> AbstractValue:
        if self.is_bottom():
            return copy.deepcopy(other)
        if other.is_bottom():
            return copy.deepcopy(self)
        new_interval = self.interval.join(other.interval)
        new_tags = self.type_tags | other.type_tags
        new_nullity = _join_nullity(self.nullity, other.nullity)
        new_strings = _join_string_sets(self.string_values, other.string_values)
        new_cont_len = self.container_length.join(other.container_length)
        new_attrs: Dict[str, AbstractValue] = {}
        all_keys = set(self.attributes.keys()) | set(other.attributes.keys())
        for k in all_keys:
            a = self.attributes.get(k)
            b = other.attributes.get(k)
            if a is not None and b is not None:
                new_attrs[k] = a.join(b)
            elif a is not None:
                new_attrs[k] = copy.deepcopy(a)
            else:
                assert b is not None
                new_attrs[k] = copy.deepcopy(b)
        new_extra: Dict[str, Any] = {}
        for k in set(self.extra.keys()) | set(other.extra.keys()):
            if k in self.extra and k in other.extra:
                new_extra[k] = self.extra[k]
            elif k in self.extra:
                new_extra[k] = self.extra[k]
            else:
                new_extra[k] = other.extra[k]
        return AbstractValue(
            interval=new_interval,
            type_tags=new_tags,
            nullity=new_nullity,
            string_values=new_strings,
            container_length=new_cont_len,
            attributes=new_attrs,
            extra=new_extra,
        )

    def meet(self, other: AbstractValue) -> AbstractValue:
        if self.is_bottom() or other.is_bottom():
            return AbstractValue.bottom()
        new_tags = self.type_tags & other.type_tags
        if TypeTag.ANY in self.type_tags:
            new_tags = other.type_tags.copy()
        elif TypeTag.ANY in other.type_tags:
            new_tags = self.type_tags.copy()
        if not new_tags:
            return AbstractValue.bottom()
        new_interval = self.interval.meet(other.interval)
        new_nullity = _meet_nullity(self.nullity, other.nullity)
        new_strings = _meet_string_sets(self.string_values, other.string_values)
        new_cont_len = self.container_length.meet(other.container_length)
        new_attrs: Dict[str, AbstractValue] = {}
        for k in set(self.attributes.keys()) & set(other.attributes.keys()):
            new_attrs[k] = self.attributes[k].meet(other.attributes[k])
        return AbstractValue(
            interval=new_interval,
            type_tags=new_tags,
            nullity=new_nullity,
            string_values=new_strings,
            container_length=new_cont_len,
            attributes=new_attrs,
        )

    def widen(self, other: AbstractValue) -> AbstractValue:
        if self.is_bottom():
            return copy.deepcopy(other)
        if other.is_bottom():
            return copy.deepcopy(self)
        new_interval = self.interval.widen(other.interval)
        new_tags = self.type_tags | other.type_tags
        new_nullity = _join_nullity(self.nullity, other.nullity)
        new_strings = _widen_string_sets(self.string_values, other.string_values)
        new_cont_len = self.container_length.widen(other.container_length)
        new_attrs: Dict[str, AbstractValue] = {}
        all_keys = set(self.attributes.keys()) | set(other.attributes.keys())
        for k in all_keys:
            a = self.attributes.get(k)
            b = other.attributes.get(k)
            if a is not None and b is not None:
                new_attrs[k] = a.widen(b)
            elif a is not None:
                new_attrs[k] = copy.deepcopy(a)
            else:
                assert b is not None
                new_attrs[k] = copy.deepcopy(b)
        return AbstractValue(
            interval=new_interval,
            type_tags=new_tags,
            nullity=new_nullity,
            string_values=new_strings,
            container_length=new_cont_len,
            attributes=new_attrs,
        )

    def narrow(self, other: AbstractValue) -> AbstractValue:
        if self.is_bottom():
            return AbstractValue.bottom()
        if other.is_bottom():
            return AbstractValue.bottom()
        new_interval = self.interval.narrow(other.interval)
        new_tags = self.type_tags if self.type_tags != {TypeTag.ANY} else other.type_tags
        new_nullity = self.nullity
        if self.nullity == NullityTag.MAYBE_NULL and other.nullity != NullityTag.MAYBE_NULL:
            new_nullity = other.nullity
        new_strings = other.string_values if self.string_values is None and other.string_values is not None else self.string_values
        new_cont_len = self.container_length.narrow(other.container_length)
        return AbstractValue(
            interval=new_interval,
            type_tags=new_tags,
            nullity=new_nullity,
            string_values=new_strings,
            container_length=new_cont_len,
            attributes=copy.deepcopy(self.attributes),
        )

    def equals(self, other: AbstractValue) -> bool:
        if self.is_bottom() and other.is_bottom():
            return True
        if self.is_bottom() or other.is_bottom():
            return False
        if self.type_tags != other.type_tags:
            return False
        if self.nullity != other.nullity:
            return False
        if self.interval.lo != other.interval.lo or self.interval.hi != other.interval.hi:
            return False
        if self.string_values != other.string_values:
            return False
        if self.container_length.lo != other.container_length.lo or self.container_length.hi != other.container_length.hi:
            return False
        if set(self.attributes.keys()) != set(other.attributes.keys()):
            return False
        for k in self.attributes:
            if not self.attributes[k].equals(other.attributes[k]):
                return False
        return True


# Nullity helpers

def _join_nullity(a: NullityTag, b: NullityTag) -> NullityTag:
    if a == b:
        return a
    return NullityTag.MAYBE_NULL


def _meet_nullity(a: NullityTag, b: NullityTag) -> NullityTag:
    if a == b:
        return a
    if a == NullityTag.MAYBE_NULL:
        return b
    if b == NullityTag.MAYBE_NULL:
        return a
    return NullityTag.DEFINITELY_NOT_NULL


# String set helpers

_MAX_STRING_VALUES = 16


def _join_string_sets(a: Optional[FrozenSet[str]], b: Optional[FrozenSet[str]]) -> Optional[FrozenSet[str]]:
    if a is None or b is None:
        return None
    combined = a | b
    if len(combined) > _MAX_STRING_VALUES:
        return None
    return combined


def _meet_string_sets(a: Optional[FrozenSet[str]], b: Optional[FrozenSet[str]]) -> Optional[FrozenSet[str]]:
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _widen_string_sets(a: Optional[FrozenSet[str]], b: Optional[FrozenSet[str]]) -> Optional[FrozenSet[str]]:
    if a is None or b is None:
        return None
    combined = a | b
    if len(combined) > _MAX_STRING_VALUES:
        return None
    return combined


# ---------------------------------------------------------------------------
# AbstractState
# ---------------------------------------------------------------------------

@dataclass
class ExceptionState:
    active: bool = False
    exception_types: Set[str] = field(default_factory=set)
    exception_value: Optional[AbstractValue] = None
    may_propagate: bool = False

    def join(self, other: ExceptionState) -> ExceptionState:
        if not self.active:
            return copy.deepcopy(other)
        if not other.active:
            return copy.deepcopy(self)
        new_val = self.exception_value
        if new_val is not None and other.exception_value is not None:
            new_val = new_val.join(other.exception_value)
        elif other.exception_value is not None:
            new_val = copy.deepcopy(other.exception_value)
        return ExceptionState(
            active=True,
            exception_types=self.exception_types | other.exception_types,
            exception_value=new_val,
            may_propagate=self.may_propagate or other.may_propagate,
        )

    def equals(self, other: ExceptionState) -> bool:
        if self.active != other.active:
            return False
        if self.exception_types != other.exception_types:
            return False
        if self.may_propagate != other.may_propagate:
            return False
        if self.exception_value is None and other.exception_value is None:
            return True
        if self.exception_value is None or other.exception_value is None:
            return False
        return self.exception_value.equals(other.exception_value)


@dataclass
class AbstractState:
    variables: Dict[str, AbstractValue] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    exception_state: ExceptionState = field(default_factory=ExceptionState)
    return_values: List[AbstractValue] = field(default_factory=list)
    is_bottom_state: bool = False
    loop_depth: int = 0
    break_states: List[AbstractState] = field(default_factory=list)
    continue_states: List[AbstractState] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)

    @staticmethod
    def bottom() -> AbstractState:
        return AbstractState(is_bottom_state=True)

    @staticmethod
    def top() -> AbstractState:
        return AbstractState()

    def is_bottom(self) -> bool:
        return self.is_bottom_state

    def deep_copy(self) -> AbstractState:
        return copy.deepcopy(self)

    def get_value(self, name: str) -> AbstractValue:
        return self.variables.get(name, AbstractValue.top())

    def set_value(self, name: str, val: AbstractValue) -> AbstractState:
        new_state = self.deep_copy()
        new_state.variables[name] = val
        return new_state

    def join_with(self, other: AbstractState) -> AbstractState:
        if self.is_bottom():
            return other.deep_copy()
        if other.is_bottom():
            return self.deep_copy()
        new_vars: Dict[str, AbstractValue] = {}
        all_keys = set(self.variables.keys()) | set(other.variables.keys())
        for k in all_keys:
            a = self.variables.get(k, AbstractValue.bottom())
            b = other.variables.get(k, AbstractValue.bottom())
            new_vars[k] = a.join(b)
        new_exc = self.exception_state.join(other.exception_state)
        new_rets = list(self.return_values)
        for rv in other.return_values:
            new_rets.append(rv)
        new_call_stack = list(self.call_stack) if self.call_stack else list(other.call_stack)
        return AbstractState(
            variables=new_vars,
            call_stack=new_call_stack,
            exception_state=new_exc,
            return_values=new_rets,
            is_bottom_state=False,
            loop_depth=max(self.loop_depth, other.loop_depth),
        )

    def widen_with(self, other: AbstractState) -> AbstractState:
        if self.is_bottom():
            return other.deep_copy()
        if other.is_bottom():
            return self.deep_copy()
        new_vars: Dict[str, AbstractValue] = {}
        all_keys = set(self.variables.keys()) | set(other.variables.keys())
        for k in all_keys:
            a = self.variables.get(k, AbstractValue.bottom())
            b = other.variables.get(k, AbstractValue.bottom())
            new_vars[k] = a.widen(b)
        new_exc = self.exception_state.join(other.exception_state)
        return AbstractState(
            variables=new_vars,
            call_stack=list(self.call_stack),
            exception_state=new_exc,
            return_values=list(self.return_values) + list(other.return_values),
            is_bottom_state=False,
            loop_depth=max(self.loop_depth, other.loop_depth),
        )

    def narrow_with(self, other: AbstractState) -> AbstractState:
        if self.is_bottom():
            return AbstractState.bottom()
        if other.is_bottom():
            return AbstractState.bottom()
        new_vars: Dict[str, AbstractValue] = {}
        for k in self.variables:
            a = self.variables[k]
            b = other.variables.get(k, AbstractValue.top())
            new_vars[k] = a.narrow(b)
        return AbstractState(
            variables=new_vars,
            call_stack=list(self.call_stack),
            exception_state=copy.deepcopy(self.exception_state),
            return_values=list(self.return_values),
            is_bottom_state=False,
            loop_depth=self.loop_depth,
        )

    def meet_with(self, other: AbstractState) -> AbstractState:
        if self.is_bottom() or other.is_bottom():
            return AbstractState.bottom()
        new_vars: Dict[str, AbstractValue] = {}
        all_keys = set(self.variables.keys()) | set(other.variables.keys())
        for k in all_keys:
            a = self.variables.get(k, AbstractValue.top())
            b = other.variables.get(k, AbstractValue.top())
            met = a.meet(b)
            if met.is_bottom():
                return AbstractState.bottom()
            new_vars[k] = met
        return AbstractState(
            variables=new_vars,
            call_stack=list(self.call_stack),
            exception_state=copy.deepcopy(self.exception_state),
            return_values=list(self.return_values),
            is_bottom_state=False,
            loop_depth=self.loop_depth,
        )

    def equals(self, other: AbstractState) -> bool:
        if self.is_bottom() and other.is_bottom():
            return True
        if self.is_bottom() or other.is_bottom():
            return False
        if set(self.variables.keys()) != set(other.variables.keys()):
            return False
        for k in self.variables:
            if not self.variables[k].equals(other.variables[k]):
                return False
        if not self.exception_state.equals(other.exception_state):
            return False
        return True


# ---------------------------------------------------------------------------
# Expression evaluator helper
# ---------------------------------------------------------------------------

def _eval_expr(expr: Expr, state: AbstractState) -> AbstractValue:
    if isinstance(expr, VarExpr):
        return state.get_value(expr.name)
    if isinstance(expr, ConstExpr):
        return AbstractValue.from_const(expr.value)
    if isinstance(expr, BinOpExpr):
        left_val = _eval_expr(expr.left, state)
        right_val = _eval_expr(expr.right, state)
        return _eval_binop(expr.op, left_val, right_val)
    if isinstance(expr, UnaryOpExpr):
        operand_val = _eval_expr(expr.operand, state)
        return _eval_unaryop(expr.op, operand_val)
    if isinstance(expr, CallExpr):
        return AbstractValue.top()
    if isinstance(expr, AttrExpr):
        obj_val = _eval_expr(expr.obj, state)
        if expr.attr in obj_val.attributes:
            return obj_val.attributes[expr.attr]
        return AbstractValue.top()
    if isinstance(expr, SubscriptExpr):
        obj_val = _eval_expr(expr.obj, state)
        if TypeTag.LIST in obj_val.type_tags or TypeTag.TUPLE in obj_val.type_tags:
            return AbstractValue.top()
        if TypeTag.DICT in obj_val.type_tags:
            return AbstractValue.top()
        if TypeTag.STR in obj_val.type_tags:
            return AbstractValue.from_type(TypeTag.STR)
        return AbstractValue.top()
    if isinstance(expr, CompareExpr):
        return AbstractValue.from_type(TypeTag.BOOL)
    if isinstance(expr, BoolOpExpr):
        if not expr.values:
            return AbstractValue.from_type(TypeTag.BOOL)
        result = _eval_expr(expr.values[0], state)
        for v in expr.values[1:]:
            other = _eval_expr(v, state)
            result = result.join(other)
        return result
    return AbstractValue.top()


def _eval_binop(op: str, left: AbstractValue, right: AbstractValue) -> AbstractValue:
    if left.is_bottom() or right.is_bottom():
        return AbstractValue.bottom()
    if op == "+":
        if left.is_string() and right.is_string():
            new_strings: Optional[FrozenSet[str]] = None
            if left.string_values is not None and right.string_values is not None:
                combined = set()
                for s1 in left.string_values:
                    for s2 in right.string_values:
                        combined.add(s1 + s2)
                if len(combined) <= _MAX_STRING_VALUES:
                    new_strings = frozenset(combined)
            return AbstractValue(
                interval=Interval.top(),
                type_tags={TypeTag.STR},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                string_values=new_strings,
            )
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_add(left.interval, right.interval)
            result_tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                result_tags.add(TypeTag.FLOAT)
            elif TypeTag.INT in left.type_tags or TypeTag.INT in right.type_tags:
                result_tags.add(TypeTag.INT)
            else:
                result_tags.add(TypeTag.INT)
            return AbstractValue(
                interval=result_iv,
                type_tags=result_tags,
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        if left.is_container() and right.is_container():
            shared = left.type_tags & right.type_tags & {TypeTag.LIST, TypeTag.TUPLE}
            if shared:
                new_len = _interval_add(left.container_length, right.container_length)
                return AbstractValue(
                    type_tags=shared,
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    container_length=new_len,
                )
        return AbstractValue.top()
    if op == "-":
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_sub(left.interval, right.interval)
            result_tags = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                result_tags.add(TypeTag.FLOAT)
            else:
                result_tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=result_tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()
    if op == "*":
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_mul(left.interval, right.interval)
            result_tags = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                result_tags.add(TypeTag.FLOAT)
            else:
                result_tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=result_tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if left.is_string() and right.is_numeric():
            return AbstractValue.from_type(TypeTag.STR)
        if left.is_numeric() and right.is_string():
            return AbstractValue.from_type(TypeTag.STR)
        if left.is_container() and right.is_numeric():
            return AbstractValue(
                type_tags=left.type_tags.copy(),
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=_interval_mul(left.container_length, right.interval),
            )
        return AbstractValue.top()
    if op == "/":
        if left.is_numeric() and right.is_numeric():
            result_iv, _ = _interval_div(left.interval, right.interval)
            return AbstractValue(interval=result_iv, type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()
    if op == "//":
        if left.is_numeric() and right.is_numeric():
            result_iv, _ = _interval_floordiv(left.interval, right.interval)
            return AbstractValue(interval=result_iv, type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()
    if op == "%":
        if left.is_numeric() and right.is_numeric():
            result_iv, _ = _interval_mod(left.interval, right.interval)
            result_tags = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                result_tags.add(TypeTag.FLOAT)
            else:
                result_tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=result_tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if left.is_string():
            return AbstractValue.from_type(TypeTag.STR)
        return AbstractValue.top()
    if op == "**":
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_pow(left.interval, right.interval)
            result_tags = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                result_tags.add(TypeTag.FLOAT)
            else:
                result_tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=result_tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()
    if op == "&":
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    if op == "|":
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    if op == "^":
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    if op == "<<" or op == ">>":
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    return AbstractValue.top()


def _eval_unaryop(op: str, operand: AbstractValue) -> AbstractValue:
    if operand.is_bottom():
        return AbstractValue.bottom()
    if op == "-" or op == "USub":
        if operand.is_numeric():
            return AbstractValue(
                interval=operand.interval.negate(),
                type_tags=operand.type_tags.copy(),
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        return AbstractValue.top()
    if op == "+" or op == "UAdd":
        if operand.is_numeric():
            return AbstractValue(
                interval=Interval(lo=operand.interval.lo, hi=operand.interval.hi),
                type_tags=operand.type_tags.copy(),
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        return AbstractValue.top()
    if op == "~" or op == "Invert":
        if TypeTag.INT in operand.type_tags or TypeTag.BOOL in operand.type_tags:
            if operand.interval.is_const() and operand.interval.lo is not None:
                v = int(operand.interval.lo)
                return AbstractValue(
                    interval=Interval.const(float(~v)),
                    type_tags={TypeTag.INT},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
            new_lo: Optional[float] = None
            new_hi: Optional[float] = None
            if operand.interval.hi is not None:
                new_lo = float(~int(operand.interval.hi))
            if operand.interval.lo is not None:
                new_hi = float(~int(operand.interval.lo))
            return AbstractValue(
                interval=Interval(lo=new_lo, hi=new_hi),
                type_tags={TypeTag.INT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        return AbstractValue.top()
    if op == "not" or op == "Not":
        return AbstractValue(
            interval=Interval(lo=0.0, hi=1.0),
            type_tags={TypeTag.BOOL},
            nullity=NullityTag.DEFINITELY_NOT_NULL,
        )
    return AbstractValue.top()


def _eval_comparison(op: str, left: AbstractValue, right: AbstractValue) -> AbstractValue:
    if left.is_bottom() or right.is_bottom():
        return AbstractValue.bottom()
    result = AbstractValue.from_type(TypeTag.BOOL)
    if left.interval.is_const() and right.interval.is_const():
        lv = left.interval.lo
        rv = right.interval.lo
        if lv is not None and rv is not None:
            if op in ("<", "Lt"):
                val = lv < rv
            elif op in ("<=", "LtE"):
                val = lv <= rv
            elif op in (">", "Gt"):
                val = lv > rv
            elif op in (">=", "GtE"):
                val = lv >= rv
            elif op in ("==", "Eq"):
                val = lv == rv
            elif op in ("!=", "NotEq"):
                val = lv != rv
            else:
                return result
            return AbstractValue.from_const(val)
    if op in ("<", "Lt"):
        if left.interval.hi is not None and right.interval.lo is not None:
            if left.interval.hi < right.interval.lo:
                return AbstractValue.from_const(True)
        if left.interval.lo is not None and right.interval.hi is not None:
            if left.interval.lo >= right.interval.hi:
                return AbstractValue.from_const(False)
    if op in ("<=", "LtE"):
        if left.interval.hi is not None and right.interval.lo is not None:
            if left.interval.hi <= right.interval.lo:
                return AbstractValue.from_const(True)
        if left.interval.lo is not None and right.interval.hi is not None:
            if left.interval.lo > right.interval.hi:
                return AbstractValue.from_const(False)
    if op in (">", "Gt"):
        if left.interval.lo is not None and right.interval.hi is not None:
            if left.interval.lo > right.interval.hi:
                return AbstractValue.from_const(True)
        if left.interval.hi is not None and right.interval.lo is not None:
            if left.interval.hi <= right.interval.lo:
                return AbstractValue.from_const(False)
    if op in (">=", "GtE"):
        if left.interval.lo is not None and right.interval.hi is not None:
            if left.interval.lo >= right.interval.hi:
                return AbstractValue.from_const(True)
        if left.interval.hi is not None and right.interval.lo is not None:
            if left.interval.hi < right.interval.lo:
                return AbstractValue.from_const(False)
    return result


# ---------------------------------------------------------------------------
# Abstract Transformer (ABC base)
# ---------------------------------------------------------------------------

class AbstractTransformer(ABC):
    @abstractmethod
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        ...

    def transform_block(self, block: BlockNode, state: AbstractState) -> AbstractState:
        current = state
        for stmt in block.statements:
            if current.is_bottom():
                break
            current = self.transform(stmt, current)
        return current

    def transform_function(self, func: FunctionNode, entry_state: AbstractState) -> Dict[str, AbstractState]:
        body_state = entry_state.deep_copy()
        body_state.call_stack = list(entry_state.call_stack) + [func.name]
        for i, param in enumerate(func.params):
            if param not in body_state.variables:
                body_state.variables[param] = AbstractValue.top()
        for i, param in enumerate(func.params):
            default_idx = i - (len(func.params) - len(func.defaults))
            if 0 <= default_idx < len(func.defaults):
                default_expr = func.defaults[default_idx]
                if default_expr is not None:
                    default_val = _eval_expr(default_expr, body_state)
                    existing = body_state.get_value(param)
                    body_state.variables[param] = existing.join(default_val)
        block = BlockNode(statements=func.body)
        result_state = self.transform_block(block, body_state)
        results: Dict[str, AbstractState] = {
            "normal": result_state,
        }
        if result_state.return_values:
            ret_state = result_state.deep_copy()
            results["return"] = ret_state
        if result_state.exception_state.active:
            exc_state = result_state.deep_copy()
            results["exception"] = exc_state
        return results


# ---------------------------------------------------------------------------
# AssignTransformer
# ---------------------------------------------------------------------------

class AssignTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, AssignNode):
            return self._transform_assign(node, state)
        return state

    def _transform_assign(self, node: AssignNode, state: AbstractState) -> AbstractState:
        value = self._evaluate_expr(node.value, state)
        new_state = state.deep_copy()
        new_state.variables[node.target] = value
        return new_state

    def _evaluate_expr(self, expr: Expr, state: AbstractState) -> AbstractValue:
        if isinstance(expr, ConstExpr):
            return AbstractValue.from_const(expr.value)
        if isinstance(expr, VarExpr):
            return state.get_value(expr.name)
        if isinstance(expr, BinOpExpr):
            left_val = self._evaluate_expr(expr.left, state)
            right_val = self._evaluate_expr(expr.right, state)
            return _eval_binop(expr.op, left_val, right_val)
        if isinstance(expr, UnaryOpExpr):
            operand_val = self._evaluate_expr(expr.operand, state)
            return _eval_unaryop(expr.op, operand_val)
        if isinstance(expr, CallExpr):
            return self._evaluate_call(expr, state)
        if isinstance(expr, AttrExpr):
            obj_val = self._evaluate_expr(expr.obj, state)
            return self._evaluate_attr_access(obj_val, expr.attr, state)
        if isinstance(expr, SubscriptExpr):
            obj_val = self._evaluate_expr(expr.obj, state)
            idx_val = self._evaluate_expr(expr.index, state)
            return self._evaluate_subscript(obj_val, idx_val)
        if isinstance(expr, CompareExpr):
            return self._evaluate_compare(expr, state)
        if isinstance(expr, BoolOpExpr):
            return self._evaluate_boolop(expr, state)
        return AbstractValue.top()

    def _evaluate_call(self, expr: CallExpr, state: AbstractState) -> AbstractValue:
        if isinstance(expr.func, VarExpr):
            fname = expr.func.name
            args_vals = [self._evaluate_expr(a, state) for a in expr.args]
            return _apply_builtin_call(fname, args_vals, state)
        if isinstance(expr.func, AttrExpr):
            obj_val = self._evaluate_expr(expr.func.obj, state)
            args_vals = [self._evaluate_expr(a, state) for a in expr.args]
            return _apply_method_call(obj_val, expr.func.attr, args_vals, state)
        return AbstractValue.top()

    def _evaluate_attr_access(self, obj_val: AbstractValue, attr: str, state: AbstractState) -> AbstractValue:
        if obj_val.nullity == NullityTag.DEFINITELY_NULL:
            return AbstractValue.bottom()
        if attr in obj_val.attributes:
            return obj_val.attributes[attr]
        return AbstractValue.top()

    def _evaluate_subscript(self, obj_val: AbstractValue, idx_val: AbstractValue) -> AbstractValue:
        if obj_val.is_bottom():
            return AbstractValue.bottom()
        if TypeTag.STR in obj_val.type_tags:
            return AbstractValue.from_type(TypeTag.STR)
        if TypeTag.LIST in obj_val.type_tags or TypeTag.TUPLE in obj_val.type_tags:
            return AbstractValue.top()
        if TypeTag.DICT in obj_val.type_tags:
            return AbstractValue.top()
        if TypeTag.BYTES in obj_val.type_tags:
            return AbstractValue.from_type(TypeTag.INT)
        return AbstractValue.top()

    def _evaluate_compare(self, expr: CompareExpr, state: AbstractState) -> AbstractValue:
        left_val = self._evaluate_expr(expr.left, state)
        result = AbstractValue.from_type(TypeTag.BOOL)
        current_left = left_val
        for op, comp_expr in zip(expr.ops, expr.comparators):
            right_val = self._evaluate_expr(comp_expr, state)
            partial = _eval_comparison(op, current_left, right_val)
            result = result.meet(partial)
            current_left = right_val
        return result

    def _evaluate_boolop(self, expr: BoolOpExpr, state: AbstractState) -> AbstractValue:
        if not expr.values:
            return AbstractValue.from_type(TypeTag.BOOL)
        first_val = self._evaluate_expr(expr.values[0], state)
        if expr.op == "and" or expr.op == "And":
            current = first_val
            for v in expr.values[1:]:
                if current.is_bottom():
                    return AbstractValue.bottom()
                if (current.interval.is_const() and current.interval.lo == 0.0
                        and TypeTag.BOOL in current.type_tags):
                    return current
                next_val = self._evaluate_expr(v, state)
                current = next_val
            return current
        if expr.op == "or" or expr.op == "Or":
            current = first_val
            for v in expr.values[1:]:
                if current.is_bottom():
                    next_val = self._evaluate_expr(v, state)
                    current = next_val
                    continue
                if _is_definitely_truthy(current):
                    return current
                next_val = self._evaluate_expr(v, state)
                current = current.join(next_val)
            return current
        return AbstractValue.top()


def _is_definitely_truthy(val: AbstractValue) -> bool:
    if val.is_bottom():
        return False
    if val.nullity == NullityTag.DEFINITELY_NULL:
        return False
    if TypeTag.NONE in val.type_tags and len(val.type_tags) == 1:
        return False
    if val.interval.is_const() and val.interval.lo == 0.0 and TypeTag.BOOL in val.type_tags:
        return False
    if val.interval.lo is not None and val.interval.lo > 0:
        return True
    if val.interval.hi is not None and val.interval.hi < 0:
        return True
    if val.is_string() and val.string_values is not None:
        return all(len(s) > 0 for s in val.string_values)
    return False


def _is_definitely_falsy(val: AbstractValue) -> bool:
    if val.is_bottom():
        return True
    if val.nullity == NullityTag.DEFINITELY_NULL:
        return True
    if val.interval.is_const() and val.interval.lo == 0.0 and (
        TypeTag.BOOL in val.type_tags or TypeTag.INT in val.type_tags
    ):
        return True
    if val.is_string() and val.string_values is not None and val.string_values == frozenset({""}) :
        return True
    return False


# ---------------------------------------------------------------------------
# Builtin call handler
# ---------------------------------------------------------------------------

def _apply_builtin_call(fname: str, args: List[AbstractValue], state: AbstractState) -> AbstractValue:
    if fname == "len":
        if args:
            arg = args[0]
            if arg.is_container():
                return AbstractValue(
                    interval=arg.container_length,
                    type_tags={TypeTag.INT},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
            if arg.is_string():
                if arg.string_values is not None:
                    lengths = {float(len(s)) for s in arg.string_values}
                    lo = min(lengths)
                    hi = max(lengths)
                    return AbstractValue(
                        interval=Interval(lo=lo, hi=hi),
                        type_tags={TypeTag.INT},
                        nullity=NullityTag.DEFINITELY_NOT_NULL,
                    )
                return AbstractValue(
                    interval=Interval(lo=0.0, hi=None),
                    type_tags={TypeTag.INT},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "isinstance":
        return AbstractValue.from_type(TypeTag.BOOL)

    if fname == "type":
        return AbstractValue.from_type(TypeTag.CALLABLE)

    if fname == "id":
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "hash":
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "repr":
        return AbstractValue.from_type(TypeTag.STR)

    if fname == "str":
        if args:
            arg = args[0]
            if arg.is_string():
                return copy.deepcopy(arg)
            if arg.is_numeric() and arg.interval.is_const() and arg.interval.lo is not None:
                s = str(int(arg.interval.lo)) if TypeTag.INT in arg.type_tags else str(arg.interval.lo)
                return AbstractValue.from_const(s)
        return AbstractValue.from_type(TypeTag.STR)

    if fname == "int":
        if args:
            arg = args[0]
            if TypeTag.INT in arg.type_tags:
                return AbstractValue(interval=Interval(lo=arg.interval.lo, hi=arg.interval.hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
            if TypeTag.FLOAT in arg.type_tags:
                lo = math.floor(arg.interval.lo) if arg.interval.lo is not None else None
                hi = math.floor(arg.interval.hi) if arg.interval.hi is not None else None
                return AbstractValue(interval=Interval(lo=lo, hi=hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
            if TypeTag.BOOL in arg.type_tags:
                return AbstractValue(interval=Interval(lo=0.0, hi=1.0), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "float":
        if args:
            arg = args[0]
            if arg.is_numeric():
                return AbstractValue(interval=Interval(lo=arg.interval.lo, hi=arg.interval.hi), type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue(type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "bool":
        if args:
            arg = args[0]
            if _is_definitely_truthy(arg):
                return AbstractValue.from_const(True)
            if _is_definitely_falsy(arg):
                return AbstractValue.from_const(False)
        return AbstractValue.from_type(TypeTag.BOOL)

    if fname == "list":
        if args:
            arg = args[0]
            new_len = arg.container_length if arg.is_container() else Interval.top()
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))

    if fname == "dict":
        if args:
            arg = args[0]
            new_len = arg.container_length if arg.is_container() else Interval.top()
            return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))

    if fname == "set":
        if args:
            arg = args[0]
            new_len = Interval(lo=0.0, hi=arg.container_length.hi) if arg.is_container() else Interval.top()
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))

    if fname == "tuple":
        if args:
            arg = args[0]
            new_len = arg.container_length if arg.is_container() else Interval.top()
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))

    if fname == "range":
        if len(args) == 1:
            arg = args[0]
            if arg.interval.lo is not None and arg.interval.hi is not None:
                count_lo = max(0.0, arg.interval.lo)
                count_hi = max(0.0, arg.interval.hi)
                return AbstractValue(
                    type_tags={TypeTag.LIST},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    container_length=Interval(lo=count_lo, hi=count_hi),
                )
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval(lo=0.0, hi=None))

    if fname == "enumerate":
        if args:
            arg = args[0]
            return AbstractValue(
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=arg.container_length if arg.is_container() else Interval.top(),
            )
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "zip":
        if args:
            lengths = []
            for a in args:
                if a.is_container():
                    lengths.append(a.container_length)
            if lengths:
                result_len = lengths[0]
                for l in lengths[1:]:
                    result_len = result_len.meet(l)
                return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=result_len)
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "map":
        if len(args) >= 2:
            iter_arg = args[1]
            return AbstractValue(
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=iter_arg.container_length if iter_arg.is_container() else Interval.top(),
            )
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "filter":
        if len(args) >= 2:
            iter_arg = args[1]
            new_len = Interval(lo=0.0, hi=iter_arg.container_length.hi) if iter_arg.is_container() else Interval(lo=0.0, hi=None)
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "sorted":
        if args:
            arg = args[0]
            return AbstractValue(
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=arg.container_length if arg.is_container() else Interval.top(),
            )
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "reversed":
        if args:
            arg = args[0]
            return AbstractValue(
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=arg.container_length if arg.is_container() else Interval.top(),
            )
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "any" or fname == "all":
        return AbstractValue.from_type(TypeTag.BOOL)

    if fname == "sum":
        return AbstractValue(type_tags={TypeTag.INT, TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "min" or fname == "max":
        if len(args) == 2:
            a, b = args[0], args[1]
            if a.is_numeric() and b.is_numeric():
                if fname == "min":
                    result_iv = _interval_min(a.interval, b.interval)
                else:
                    result_iv = _interval_max(a.interval, b.interval)
                tags = a.type_tags | b.type_tags
                return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if args:
            return AbstractValue.top()
        return AbstractValue.top()

    if fname == "abs":
        if args:
            arg = args[0]
            if arg.is_numeric():
                return AbstractValue(
                    interval=arg.interval.abs_interval(),
                    type_tags=arg.type_tags.copy(),
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
        return AbstractValue.top()

    if fname == "round":
        if args:
            arg = args[0]
            if arg.is_numeric():
                lo = round(arg.interval.lo) if arg.interval.lo is not None else None
                hi = round(arg.interval.hi) if arg.interval.hi is not None else None
                return AbstractValue(interval=Interval(lo=lo, hi=hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "pow":
        if len(args) >= 2:
            return _eval_binop("**", args[0], args[1])
        return AbstractValue.top()

    if fname == "print":
        return AbstractValue.from_const(None)

    if fname == "input":
        return AbstractValue.from_type(TypeTag.STR)

    if fname == "open":
        return AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "getattr":
        if len(args) >= 2:
            obj_val = args[0]
            if len(args) >= 3:
                default_val = args[2]
                return AbstractValue.top().join(default_val)
            return AbstractValue.top()
        return AbstractValue.top()

    if fname == "setattr":
        return AbstractValue.from_const(None)

    if fname == "hasattr":
        return AbstractValue.from_type(TypeTag.BOOL)

    if fname == "delattr":
        return AbstractValue.from_const(None)

    if fname == "callable":
        return AbstractValue.from_type(TypeTag.BOOL)

    if fname == "iter":
        return AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if fname == "next":
        return AbstractValue.top()

    return AbstractValue.top()


def _apply_method_call(obj_val: AbstractValue, method: str, args: List[AbstractValue], state: AbstractState) -> AbstractValue:
    if TypeTag.STR in obj_val.type_tags:
        return _apply_string_method(obj_val, method, args)
    if TypeTag.LIST in obj_val.type_tags:
        return _apply_list_method(obj_val, method, args)
    if TypeTag.DICT in obj_val.type_tags:
        return _apply_dict_method(obj_val, method, args)
    if TypeTag.SET in obj_val.type_tags:
        return _apply_set_method(obj_val, method, args)
    if TypeTag.TUPLE in obj_val.type_tags:
        return _apply_tuple_method(obj_val, method, args)
    return AbstractValue.top()


def _apply_string_method(obj_val: AbstractValue, method: str, args: List[AbstractValue]) -> AbstractValue:
    str_result = AbstractValue.from_type(TypeTag.STR)
    bool_result = AbstractValue.from_type(TypeTag.BOOL)
    int_result = AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method in ("strip", "lstrip", "rstrip", "lower", "upper", "title",
                   "capitalize", "swapcase", "replace", "center", "ljust",
                   "rjust", "zfill", "expandtabs"):
        if obj_val.string_values is not None and method in ("lower", "upper", "title", "capitalize", "swapcase", "strip", "lstrip", "rstrip"):
            new_vals = set()
            for s in obj_val.string_values:
                if method == "lower":
                    new_vals.add(s.lower())
                elif method == "upper":
                    new_vals.add(s.upper())
                elif method == "title":
                    new_vals.add(s.title())
                elif method == "capitalize":
                    new_vals.add(s.capitalize())
                elif method == "swapcase":
                    new_vals.add(s.swapcase())
                elif method == "strip":
                    new_vals.add(s.strip())
                elif method == "lstrip":
                    new_vals.add(s.lstrip())
                elif method == "rstrip":
                    new_vals.add(s.rstrip())
            if len(new_vals) <= _MAX_STRING_VALUES:
                return AbstractValue(
                    type_tags={TypeTag.STR},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    string_values=frozenset(new_vals),
                )
        return str_result

    if method in ("find", "rfind", "index", "rindex", "count"):
        if method in ("find", "rfind"):
            return AbstractValue(interval=Interval(lo=-1.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if method == "count":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method in ("startswith", "endswith", "isalpha", "isdigit", "isalnum",
                   "isspace", "isupper", "islower", "istitle"):
        if obj_val.string_values is not None:
            results = set()
            for s in obj_val.string_values:
                try:
                    if method == "isalpha":
                        results.add(s.isalpha())
                    elif method == "isdigit":
                        results.add(s.isdigit())
                    elif method == "isalnum":
                        results.add(s.isalnum())
                    elif method == "isspace":
                        results.add(s.isspace())
                    elif method == "isupper":
                        results.add(s.isupper())
                    elif method == "islower":
                        results.add(s.islower())
                    elif method == "istitle":
                        results.add(s.istitle())
                    else:
                        results.add(True)
                        results.add(False)
                except Exception:
                    results.add(True)
                    results.add(False)
            if len(results) == 1:
                return AbstractValue.from_const(results.pop())
        return bool_result

    if method in ("split", "rsplit"):
        return AbstractValue(
            type_tags={TypeTag.LIST},
            nullity=NullityTag.DEFINITELY_NOT_NULL,
            container_length=Interval(lo=1.0, hi=None),
        )

    if method == "join":
        return str_result

    if method == "format":
        return str_result

    if method == "encode":
        return AbstractValue.from_type(TypeTag.BYTES)

    if method == "decode":
        return str_result

    if method in ("partition", "rpartition"):
        return AbstractValue(
            type_tags={TypeTag.TUPLE},
            nullity=NullityTag.DEFINITELY_NOT_NULL,
            container_length=Interval.const(3.0),
        )

    if method == "maketrans":
        return AbstractValue.from_type(TypeTag.DICT)

    if method == "translate":
        return str_result

    return AbstractValue.top()


def _apply_list_method(obj_val: AbstractValue, method: str, args: List[AbstractValue]) -> AbstractValue:
    if method == "append":
        new_len = _interval_add(obj_val.container_length, Interval.const(1.0))
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)

    if method == "extend":
        if args and args[0].is_container():
            new_len = _interval_add(obj_val.container_length, args[0].container_length)
        else:
            new_len = Interval(lo=obj_val.container_length.lo, hi=None)
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)

    if method == "insert":
        new_len = _interval_add(obj_val.container_length, Interval.const(1.0))
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)

    if method == "pop":
        lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
        hi = (obj_val.container_length.hi - 1.0) if obj_val.container_length.hi is not None else None
        if hi is not None:
            hi = max(0.0, hi)
        return AbstractValue.top()

    if method == "remove":
        lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
        hi = (obj_val.container_length.hi - 1.0) if obj_val.container_length.hi is not None else None
        if hi is not None:
            hi = max(0.0, hi)
        return AbstractValue.from_const(None)

    if method in ("sort", "reverse"):
        return AbstractValue.from_const(None)

    if method == "count":
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method == "index":
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method == "clear":
        return AbstractValue.from_const(None)

    if method == "copy":
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    return AbstractValue.top()


def _apply_dict_method(obj_val: AbstractValue, method: str, args: List[AbstractValue]) -> AbstractValue:
    if method == "get":
        if len(args) >= 2:
            return AbstractValue.top().join(args[1])
        return AbstractValue(
            type_tags={TypeTag.ANY},
            nullity=NullityTag.MAYBE_NULL,
        )

    if method == "keys":
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    if method == "values":
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    if method == "items":
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    if method == "update":
        if args and args[0].is_container():
            new_len = Interval(
                lo=_max_opt(obj_val.container_length.lo, args[0].container_length.lo),
                hi=None if obj_val.container_length.hi is None or args[0].container_length.hi is None else obj_val.container_length.hi + args[0].container_length.hi,
            )
            return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue.from_const(None)

    if method == "pop":
        if len(args) >= 2:
            return AbstractValue.top().join(args[1])
        return AbstractValue.top()

    if method == "popitem":
        return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(2.0))

    if method == "setdefault":
        if len(args) >= 2:
            return args[1]
        return AbstractValue(nullity=NullityTag.MAYBE_NULL, type_tags={TypeTag.NONE, TypeTag.ANY})

    if method == "clear":
        return AbstractValue.from_const(None)

    if method == "copy":
        return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    return AbstractValue.top()


def _apply_set_method(obj_val: AbstractValue, method: str, args: List[AbstractValue]) -> AbstractValue:
    if method == "add":
        new_len = Interval(
            lo=obj_val.container_length.lo,
            hi=(obj_val.container_length.hi + 1.0) if obj_val.container_length.hi is not None else None,
        )
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)

    if method in ("remove", "discard"):
        lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
        hi = obj_val.container_length.hi
        return AbstractValue.from_const(None)

    if method == "pop":
        return AbstractValue.top()

    if method == "clear":
        return AbstractValue.from_const(None)

    if method == "union":
        if args and args[0].is_container():
            new_len = Interval(
                lo=_max_opt(obj_val.container_length.lo, args[0].container_length.lo),
                hi=None if obj_val.container_length.hi is None or args[0].container_length.hi is None else obj_val.container_length.hi + args[0].container_length.hi,
            )
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method == "intersection":
        if args and args[0].is_container():
            new_len = Interval(
                lo=0.0,
                hi=_min_opt(obj_val.container_length.hi, args[0].container_length.hi),
            )
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval(lo=0.0, hi=obj_val.container_length.hi))

    if method == "difference":
        new_len = Interval(lo=0.0, hi=obj_val.container_length.hi)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)

    if method == "symmetric_difference":
        if args and args[0].is_container():
            new_len = Interval(
                lo=0.0,
                hi=None if obj_val.container_length.hi is None or args[0].container_length.hi is None else obj_val.container_length.hi + args[0].container_length.hi,
            )
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    if method == "copy":
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=obj_val.container_length)

    return AbstractValue.top()


def _apply_tuple_method(obj_val: AbstractValue, method: str, args: List[AbstractValue]) -> AbstractValue:
    if method == "count":
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    if method == "index":
        return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
    return AbstractValue.top()


# ---------------------------------------------------------------------------
# GuardTransformer
# ---------------------------------------------------------------------------

class GuardTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, GuardNode):
            true_state, false_state = self._split_guard(node.condition, state)
            if node.is_negated:
                true_state, false_state = false_state, true_state
            return true_state
        return state

    def transform_guard_both(self, node: GuardNode, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if state.is_bottom():
            return state, state
        true_state, false_state = self._split_guard(node.condition, state)
        if node.is_negated:
            true_state, false_state = false_state, true_state
        return true_state, false_state

    def _split_guard(self, cond: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if isinstance(cond, CallExpr):
            return self._split_call_guard(cond, state)
        if isinstance(cond, CompareExpr):
            return self._split_compare_guard(cond, state)
        if isinstance(cond, BoolOpExpr):
            return self._split_boolop_guard(cond, state)
        if isinstance(cond, UnaryOpExpr):
            return self._split_unary_guard(cond, state)
        if isinstance(cond, VarExpr):
            return self._split_truthiness_guard(cond.name, state)
        if isinstance(cond, AttrExpr):
            return self._split_attr_guard(cond, state)
        return state.deep_copy(), state.deep_copy()

    def _split_call_guard(self, call: CallExpr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if isinstance(call.func, VarExpr):
            fname = call.func.name
            if fname == "isinstance" and len(call.args) == 2:
                return self._split_isinstance_guard(call.args[0], call.args[1], state)
            if fname == "hasattr" and len(call.args) == 2:
                return self._split_hasattr_guard(call.args[0], call.args[1], state)
            if fname == "callable" and len(call.args) == 1:
                return self._split_callable_guard(call.args[0], state)
        return state.deep_copy(), state.deep_copy()

    def _split_isinstance_guard(self, obj_expr: Expr, type_expr: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if not isinstance(obj_expr, VarExpr):
            return state.deep_copy(), state.deep_copy()
        var_name = obj_expr.name
        current_val = state.get_value(var_name)
        type_tags_map = {
            "int": TypeTag.INT,
            "float": TypeTag.FLOAT,
            "str": TypeTag.STR,
            "bool": TypeTag.BOOL,
            "list": TypeTag.LIST,
            "dict": TypeTag.DICT,
            "set": TypeTag.SET,
            "tuple": TypeTag.TUPLE,
            "bytes": TypeTag.BYTES,
            "complex": TypeTag.COMPLEX,
        }
        target_tags: Set[TypeTag] = set()
        if isinstance(type_expr, VarExpr):
            tag = type_tags_map.get(type_expr.name)
            if tag is not None:
                target_tags.add(tag)
        elif isinstance(type_expr, ConstExpr) and isinstance(type_expr.value, str):
            tag = type_tags_map.get(type_expr.value)
            if tag is not None:
                target_tags.add(tag)

        if not target_tags:
            return state.deep_copy(), state.deep_copy()

        true_state = state.deep_copy()
        false_state = state.deep_copy()
        true_val = copy.deepcopy(current_val)
        if TypeTag.ANY not in true_val.type_tags:
            true_val.type_tags = true_val.type_tags & target_tags
        else:
            true_val.type_tags = target_tags.copy()
        if not true_val.type_tags:
            true_state = AbstractState.bottom()
        else:
            true_val.nullity = NullityTag.DEFINITELY_NOT_NULL
            true_state.variables[var_name] = true_val
        false_val = copy.deepcopy(current_val)
        if TypeTag.ANY not in false_val.type_tags:
            false_val.type_tags = false_val.type_tags - target_tags
        if not false_val.type_tags:
            false_state = AbstractState.bottom()
        else:
            false_state.variables[var_name] = false_val
        return true_state, false_state

    def _split_hasattr_guard(self, obj_expr: Expr, attr_expr: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if not isinstance(obj_expr, VarExpr):
            return state.deep_copy(), state.deep_copy()
        var_name = obj_expr.name
        attr_name = ""
        if isinstance(attr_expr, ConstExpr) and isinstance(attr_expr.value, str):
            attr_name = attr_expr.value
        if not attr_name:
            return state.deep_copy(), state.deep_copy()
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        true_val = copy.deepcopy(state.get_value(var_name))
        true_val.nullity = NullityTag.DEFINITELY_NOT_NULL
        if attr_name not in true_val.attributes:
            true_val.attributes[attr_name] = AbstractValue.top()
        true_state.variables[var_name] = true_val
        return true_state, false_state

    def _split_callable_guard(self, obj_expr: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if not isinstance(obj_expr, VarExpr):
            return state.deep_copy(), state.deep_copy()
        var_name = obj_expr.name
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        true_val = copy.deepcopy(state.get_value(var_name))
        if TypeTag.ANY in true_val.type_tags:
            true_val.type_tags = {TypeTag.CALLABLE}
        else:
            true_val.type_tags = true_val.type_tags & {TypeTag.CALLABLE}
            if not true_val.type_tags:
                true_val.type_tags = {TypeTag.CALLABLE}
        true_val.nullity = NullityTag.DEFINITELY_NOT_NULL
        true_state.variables[var_name] = true_val
        false_val = copy.deepcopy(state.get_value(var_name))
        false_val.type_tags = false_val.type_tags - {TypeTag.CALLABLE}
        if not false_val.type_tags:
            false_val.type_tags = {TypeTag.ANY}
        false_state.variables[var_name] = false_val
        return true_state, false_state

    def _split_compare_guard(self, comp: CompareExpr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if len(comp.ops) == 1 and len(comp.comparators) == 1:
            op = comp.ops[0]
            left = comp.left
            right = comp.comparators[0]
            if op in ("is", "Is"):
                return self._split_is_guard(left, right, state)
            if op in ("is not", "IsNot"):
                t, f = self._split_is_guard(left, right, state)
                return f, t
            if op in ("==", "Eq"):
                return self._split_eq_guard(left, right, state)
            if op in ("!=", "NotEq"):
                t, f = self._split_eq_guard(left, right, state)
                return f, t
            if op in ("<", "Lt"):
                return self._split_lt_guard(left, right, state)
            if op in ("<=", "LtE"):
                return self._split_lte_guard(left, right, state)
            if op in (">", "Gt"):
                return self._split_gt_guard(left, right, state)
            if op in (">=", "GtE"):
                return self._split_gte_guard(left, right, state)
            if op in ("in", "In"):
                return self._split_in_guard(left, right, state)
            if op in ("not in", "NotIn"):
                t, f = self._split_in_guard(left, right, state)
                return f, t
        return state.deep_copy(), state.deep_copy()

    def _split_is_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if isinstance(right, ConstExpr) and right.value is None and isinstance(left, VarExpr):
            return self._split_none_check(left.name, state)
        if isinstance(left, ConstExpr) and left.value is None and isinstance(right, VarExpr):
            return self._split_none_check(right.name, state)
        return state.deep_copy(), state.deep_copy()

    def _split_none_check(self, var_name: str, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        true_val = AbstractValue(
            interval=Interval.bottom(),
            type_tags={TypeTag.NONE},
            nullity=NullityTag.DEFINITELY_NULL,
        )
        true_state.variables[var_name] = true_val
        false_val = copy.deepcopy(state.get_value(var_name))
        false_val.nullity = NullityTag.DEFINITELY_NOT_NULL
        false_val.type_tags = false_val.type_tags - {TypeTag.NONE}
        if not false_val.type_tags:
            false_val.type_tags = {TypeTag.ANY}
        false_state.variables[var_name] = false_val
        return true_state, false_state

    def _split_eq_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        if isinstance(left, VarExpr) and isinstance(right, ConstExpr):
            var_name = left.name
            const_val = AbstractValue.from_const(right.value)
            current_val = state.get_value(var_name)
            met = current_val.meet(const_val)
            if met.is_bottom():
                true_state = AbstractState.bottom()
            else:
                true_state.variables[var_name] = met
        elif isinstance(right, VarExpr) and isinstance(left, ConstExpr):
            var_name = right.name
            const_val = AbstractValue.from_const(left.value)
            current_val = state.get_value(var_name)
            met = current_val.meet(const_val)
            if met.is_bottom():
                true_state = AbstractState.bottom()
            else:
                true_state.variables[var_name] = met
        elif isinstance(left, VarExpr) and isinstance(right, VarExpr):
            lv = state.get_value(left.name)
            rv = state.get_value(right.name)
            met = lv.meet(rv)
            if met.is_bottom():
                true_state = AbstractState.bottom()
            else:
                true_state.variables[left.name] = met
                true_state.variables[right.name] = met
        return true_state, false_state

    def _split_lt_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        if isinstance(left, VarExpr):
            var_name = left.name
            current_val = state.get_value(var_name)
            right_val = _eval_expr(right, state)
            if right_val.interval.hi is not None:
                upper = right_val.interval.hi
                if right_val.interval.is_const():
                    upper = right_val.interval.lo
                if upper is not None:
                    constraint = Interval(lo=None, hi=upper - 1e-15)
                    true_val = copy.deepcopy(current_val)
                    true_val.interval = true_val.interval.meet(constraint)
                    true_state.variables[var_name] = true_val
            if right_val.interval.lo is not None:
                lo_val = right_val.interval.lo
                constraint_f = Interval(lo=lo_val, hi=None)
                false_val = copy.deepcopy(current_val)
                false_val.interval = false_val.interval.meet(constraint_f)
                false_state.variables[var_name] = false_val
        if isinstance(right, VarExpr):
            var_name_r = right.name
            current_val_r = state.get_value(var_name_r)
            left_val = _eval_expr(left, state)
            if left_val.interval.lo is not None:
                lo = left_val.interval.lo
                constraint = Interval(lo=lo + 1e-15, hi=None)
                true_val_r = copy.deepcopy(current_val_r)
                true_val_r.interval = true_val_r.interval.meet(constraint)
                true_state.variables[var_name_r] = true_val_r
        return true_state, false_state

    def _split_lte_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        if isinstance(left, VarExpr):
            var_name = left.name
            current_val = state.get_value(var_name)
            right_val = _eval_expr(right, state)
            if right_val.interval.hi is not None:
                upper = right_val.interval.hi
                if right_val.interval.is_const():
                    upper = right_val.interval.lo
                if upper is not None:
                    constraint = Interval(lo=None, hi=upper)
                    true_val = copy.deepcopy(current_val)
                    true_val.interval = true_val.interval.meet(constraint)
                    true_state.variables[var_name] = true_val
            if right_val.interval.lo is not None:
                lo = right_val.interval.lo
                if right_val.interval.is_const():
                    lo = right_val.interval.lo
                if lo is not None:
                    constraint_f = Interval(lo=lo + 1e-15, hi=None)
                    false_val = copy.deepcopy(current_val)
                    false_val.interval = false_val.interval.meet(constraint_f)
                    false_state.variables[var_name] = false_val
        if isinstance(right, VarExpr):
            var_name_r = right.name
            current_val_r = state.get_value(var_name_r)
            left_val = _eval_expr(left, state)
            if left_val.interval.lo is not None:
                constraint_r = Interval(lo=left_val.interval.lo, hi=None)
                true_val_r = copy.deepcopy(current_val_r)
                true_val_r.interval = true_val_r.interval.meet(constraint_r)
                true_state.variables[var_name_r] = true_val_r
        return true_state, false_state

    def _split_gt_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        f, t = self._split_lte_guard(left, right, state)
        return t, f

    def _split_gte_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        f, t = self._split_lt_guard(left, right, state)
        return t, f

    def _split_in_guard(self, left: Expr, right: Expr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        return state.deep_copy(), state.deep_copy()

    def _split_boolop_guard(self, expr: BoolOpExpr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if expr.op in ("and", "And"):
            current_true = state.deep_copy()
            accumulated_false_states: List[AbstractState] = []
            for val_expr in expr.values:
                t, f = self._split_guard(val_expr, current_true)
                accumulated_false_states.append(f)
                current_true = t
                if current_true.is_bottom():
                    break
            combined_false = AbstractState.bottom()
            for fs in accumulated_false_states:
                combined_false = combined_false.join_with(fs)
            return current_true, combined_false
        if expr.op in ("or", "Or"):
            current_false = state.deep_copy()
            accumulated_true_states: List[AbstractState] = []
            for val_expr in expr.values:
                t, f = self._split_guard(val_expr, current_false)
                accumulated_true_states.append(t)
                current_false = f
                if current_false.is_bottom():
                    break
            combined_true = AbstractState.bottom()
            for ts in accumulated_true_states:
                combined_true = combined_true.join_with(ts)
            return combined_true, current_false
        return state.deep_copy(), state.deep_copy()

    def _split_unary_guard(self, expr: UnaryOpExpr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if expr.op in ("not", "Not"):
            t, f = self._split_guard(expr.operand, state)
            return f, t
        return state.deep_copy(), state.deep_copy()

    def _split_truthiness_guard(self, var_name: str, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        true_state = state.deep_copy()
        false_state = state.deep_copy()
        current_val = state.get_value(var_name)
        true_val = copy.deepcopy(current_val)
        false_val = copy.deepcopy(current_val)
        true_val.nullity = NullityTag.DEFINITELY_NOT_NULL
        true_val.type_tags = true_val.type_tags - {TypeTag.NONE}
        if not true_val.type_tags:
            true_state = AbstractState.bottom()
        else:
            if true_val.interval.contains(0.0) and (TypeTag.INT in true_val.type_tags or TypeTag.FLOAT in true_val.type_tags or TypeTag.BOOL in true_val.type_tags):
                if true_val.interval.lo is not None and true_val.interval.lo == 0.0:
                    true_val.interval = Interval(lo=1e-15, hi=true_val.interval.hi)
                elif true_val.interval.hi is not None and true_val.interval.hi == 0.0:
                    true_val.interval = Interval(lo=true_val.interval.lo, hi=-1e-15)
            true_state.variables[var_name] = true_val
        if TypeTag.NONE in current_val.type_tags or current_val.nullity != NullityTag.DEFINITELY_NOT_NULL:
            pass
        if current_val.is_numeric():
            if not false_val.interval.contains(0.0):
                false_val_alt = copy.deepcopy(current_val)
                false_val_alt.type_tags = {TypeTag.NONE}
                false_val_alt.nullity = NullityTag.DEFINITELY_NULL
                false_state.variables[var_name] = false_val_alt
            else:
                false_val.interval = false_val.interval.meet(Interval.const(0.0))
                false_state.variables[var_name] = false_val
        else:
            false_state.variables[var_name] = false_val
        return true_state, false_state

    def _split_attr_guard(self, expr: AttrExpr, state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        if isinstance(expr.obj, VarExpr):
            obj_val = state.get_value(expr.obj.name)
            if expr.attr in obj_val.attributes:
                attr_val = obj_val.attributes[expr.attr]
                true_state = state.deep_copy()
                false_state = state.deep_copy()
                true_obj = copy.deepcopy(obj_val)
                true_obj.nullity = NullityTag.DEFINITELY_NOT_NULL
                true_state.variables[expr.obj.name] = true_obj
                return true_state, false_state
        return state.deep_copy(), state.deep_copy()


# ---------------------------------------------------------------------------
# PhiTransformer
# ---------------------------------------------------------------------------

class PhiTransformer(AbstractTransformer):
    def __init__(self, is_loop_header: bool = False) -> None:
        self._is_loop_header = is_loop_header

    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, PhiNode):
            return self._transform_phi(node, state)
        return state

    def _transform_phi(self, node: PhiNode, state: AbstractState) -> AbstractState:
        vals: List[AbstractValue] = []
        for src_var, _block_id in node.sources:
            vals.append(state.get_value(src_var))
        if not vals:
            return state
        merged = vals[0]
        for v in vals[1:]:
            merged = merged.join(v)
        new_state = state.deep_copy()
        new_state.variables[node.target] = merged
        return new_state

    def join_states(self, states: List[AbstractState]) -> AbstractState:
        if not states:
            return AbstractState.bottom()
        result = states[0]
        for s in states[1:]:
            result = result.join_with(s)
        return result

    def join_states_with_widening(self, old_state: AbstractState, new_state: AbstractState) -> AbstractState:
        if self._is_loop_header:
            return old_state.widen_with(new_state)
        return old_state.join_with(new_state)

    def selective_join(self, old_state: AbstractState, new_state: AbstractState, changed_vars: Set[str]) -> AbstractState:
        if old_state.is_bottom():
            return new_state.deep_copy()
        if new_state.is_bottom():
            return old_state.deep_copy()
        result = old_state.deep_copy()
        for var_name in changed_vars:
            old_val = old_state.get_value(var_name)
            new_val = new_state.get_value(var_name)
            result.variables[var_name] = old_val.join(new_val)
        return result

    def detect_changed_variables(self, old_state: AbstractState, new_state: AbstractState) -> Set[str]:
        changed: Set[str] = set()
        all_vars = set(old_state.variables.keys()) | set(new_state.variables.keys())
        for v in all_vars:
            old_val = old_state.variables.get(v)
            new_val = new_state.variables.get(v)
            if old_val is None or new_val is None:
                changed.add(v)
            elif not old_val.equals(new_val):
                changed.add(v)
        return changed


# ---------------------------------------------------------------------------
# CallTransformer
# ---------------------------------------------------------------------------

class CallTransformer(AbstractTransformer):
    def __init__(self) -> None:
        self._function_summaries: Dict[str, Callable[[List[AbstractValue], AbstractState], AbstractValue]] = {}
        self._side_effects: Dict[str, Callable[[List[AbstractValue], AbstractState], AbstractState]] = {}

    def register_summary(self, func_name: str, summary: Callable[[List[AbstractValue], AbstractState], AbstractValue]) -> None:
        self._function_summaries[func_name] = summary

    def register_side_effect(self, func_name: str, effect: Callable[[List[AbstractValue], AbstractState], AbstractState]) -> None:
        self._side_effects[func_name] = effect

    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, CallNode):
            return self._transform_call(node, state)
        return state

    def _transform_call(self, node: CallNode, state: AbstractState) -> AbstractState:
        args_vals = [_eval_expr(a, state) for a in node.args]
        kwargs_vals = {k: _eval_expr(v, state) for k, v in node.kwargs.items()}
        result_val = self._dispatch_call(node.func, args_vals, kwargs_vals, state)
        new_state = self._apply_side_effects(node.func, args_vals, state)
        if node.target is not None:
            new_state.variables[node.target] = result_val
        new_state = self._handle_call_exceptions(node.func, args_vals, new_state)
        return new_state

    def _dispatch_call(self, func_expr: Expr, args: List[AbstractValue], kwargs: Dict[str, AbstractValue], state: AbstractState) -> AbstractValue:
        if isinstance(func_expr, VarExpr):
            fname = func_expr.name
            if fname in self._function_summaries:
                return self._function_summaries[fname](args, state)
            return _apply_builtin_call(fname, args, state)
        if isinstance(func_expr, AttrExpr):
            obj_val = _eval_expr(func_expr.obj, state)
            return _apply_method_call(obj_val, func_expr.attr, args, state)
        func_val = _eval_expr(func_expr, state)
        if TypeTag.CALLABLE in func_val.type_tags:
            return self._handle_higher_order_call(func_val, args, state)
        return AbstractValue.top()

    def _handle_higher_order_call(self, func_val: AbstractValue, args: List[AbstractValue], state: AbstractState) -> AbstractValue:
        return AbstractValue.top()

    def _apply_side_effects(self, func_expr: Expr, args: List[AbstractValue], state: AbstractState) -> AbstractState:
        new_state = state.deep_copy()
        if isinstance(func_expr, VarExpr):
            fname = func_expr.name
            if fname in self._side_effects:
                return self._side_effects[fname](args, new_state)
            if fname == "print":
                return new_state
            if fname == "setattr" and len(args) >= 3:
                return new_state
        if isinstance(func_expr, AttrExpr) and isinstance(func_expr.obj, VarExpr):
            obj_name = func_expr.obj.name
            obj_val = new_state.get_value(obj_name)
            method = func_expr.attr
            if TypeTag.LIST in obj_val.type_tags:
                if method == "append":
                    new_len = _interval_add(obj_val.container_length, Interval.const(1.0))
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = new_len
                    new_state.variables[obj_name] = obj_val
                elif method == "extend" and args:
                    ext_len = args[0].container_length if args[0].is_container() else Interval(lo=0.0, hi=None)
                    new_len = _interval_add(obj_val.container_length, ext_len)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = new_len
                    new_state.variables[obj_name] = obj_val
                elif method == "insert":
                    new_len = _interval_add(obj_val.container_length, Interval.const(1.0))
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = new_len
                    new_state.variables[obj_name] = obj_val
                elif method == "pop":
                    lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
                    hi = (obj_val.container_length.hi - 1.0) if obj_val.container_length.hi is not None else None
                    if hi is not None:
                        hi = max(0.0, hi)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval(lo=lo, hi=hi)
                    new_state.variables[obj_name] = obj_val
                elif method == "remove":
                    lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
                    hi = (obj_val.container_length.hi - 1.0) if obj_val.container_length.hi is not None else None
                    if hi is not None:
                        hi = max(0.0, hi)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval(lo=lo, hi=hi)
                    new_state.variables[obj_name] = obj_val
                elif method == "clear":
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval.const(0.0)
                    new_state.variables[obj_name] = obj_val
                elif method in ("sort", "reverse"):
                    pass
            elif TypeTag.DICT in obj_val.type_tags:
                if method == "clear":
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval.const(0.0)
                    new_state.variables[obj_name] = obj_val
                elif method == "pop" or method == "popitem":
                    lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
                    hi = (obj_val.container_length.hi - 1.0) if obj_val.container_length.hi is not None else None
                    if hi is not None:
                        hi = max(0.0, hi)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval(lo=lo, hi=hi)
                    new_state.variables[obj_name] = obj_val
                elif method == "update" and args:
                    if args[0].is_container():
                        new_len = Interval(
                            lo=_max_opt(obj_val.container_length.lo, args[0].container_length.lo),
                            hi=None if obj_val.container_length.hi is None or args[0].container_length.hi is None else obj_val.container_length.hi + args[0].container_length.hi,
                        )
                    else:
                        new_len = Interval(lo=obj_val.container_length.lo, hi=None)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = new_len
                    new_state.variables[obj_name] = obj_val
            elif TypeTag.SET in obj_val.type_tags:
                if method == "add":
                    new_len = Interval(lo=obj_val.container_length.lo, hi=(obj_val.container_length.hi + 1.0) if obj_val.container_length.hi is not None else None)
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = new_len
                    new_state.variables[obj_name] = obj_val
                elif method in ("remove", "discard", "pop"):
                    lo = max(0.0, (obj_val.container_length.lo or 0.0) - 1.0)
                    hi = obj_val.container_length.hi
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval(lo=lo, hi=hi)
                    new_state.variables[obj_name] = obj_val
                elif method == "clear":
                    obj_val = copy.deepcopy(obj_val)
                    obj_val.container_length = Interval.const(0.0)
                    new_state.variables[obj_name] = obj_val
        return new_state

    def _handle_call_exceptions(self, func_expr: Expr, args: List[AbstractValue], state: AbstractState) -> AbstractState:
        if isinstance(func_expr, VarExpr):
            fname = func_expr.name
            if fname in ("int", "float"):
                if args and TypeTag.STR in args[0].type_tags:
                    new_state = state.deep_copy()
                    new_state.exception_state = ExceptionState(
                        active=True,
                        exception_types={"ValueError"},
                        may_propagate=True,
                    )
                    return new_state
            if fname == "open":
                new_state = state.deep_copy()
                exc = ExceptionState(
                    active=True,
                    exception_types={"FileNotFoundError", "PermissionError", "IOError"},
                    may_propagate=True,
                )
                new_state.exception_state = new_state.exception_state.join(exc)
                return new_state
            if fname == "next":
                new_state = state.deep_copy()
                exc = ExceptionState(
                    active=True,
                    exception_types={"StopIteration"},
                    may_propagate=True,
                )
                new_state.exception_state = new_state.exception_state.join(exc)
                return new_state
        return state


# ---------------------------------------------------------------------------
# ReturnTransformer
# ---------------------------------------------------------------------------

class ReturnTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, ReturnNode):
            return self._transform_return(node, state)
        return state

    def _transform_return(self, node: ReturnNode, state: AbstractState) -> AbstractState:
        new_state = state.deep_copy()
        if node.value is not None:
            ret_val = _eval_expr(node.value, state)
        else:
            ret_val = AbstractValue.from_const(None)
        new_state.return_values.append(ret_val)
        return new_state

    def collect_return_type(self, state: AbstractState) -> AbstractValue:
        if not state.return_values:
            return AbstractValue.from_const(None)
        result = state.return_values[0]
        for rv in state.return_values[1:]:
            result = result.join(rv)
        return result

    def merge_returns(self, states: List[AbstractState]) -> AbstractValue:
        all_returns: List[AbstractValue] = []
        for s in states:
            all_returns.extend(s.return_values)
        if not all_returns:
            return AbstractValue.from_const(None)
        result = all_returns[0]
        for rv in all_returns[1:]:
            result = result.join(rv)
        return result

    def split_normal_exception(self, state: AbstractState) -> Tuple[AbstractValue, Optional[ExceptionState]]:
        ret_type = self.collect_return_type(state)
        exc = state.exception_state if state.exception_state.active else None
        return ret_type, exc


# ---------------------------------------------------------------------------
# ExceptionTransformer
# ---------------------------------------------------------------------------

class ExceptionTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, RaiseNode):
            return self._transform_raise(node, state)
        if isinstance(node, TryExceptNode):
            return self._transform_try_except(node, state)
        return state

    def _transform_raise(self, node: RaiseNode, state: AbstractState) -> AbstractState:
        new_state = state.deep_copy()
        exc_types: Set[str] = set()
        exc_val: Optional[AbstractValue] = None
        if node.exception is not None:
            exc_abs = _eval_expr(node.exception, state)
            if isinstance(node.exception, CallExpr) and isinstance(node.exception.func, VarExpr):
                exc_types.add(node.exception.func.name)
            elif isinstance(node.exception, VarExpr):
                exc_types.add(node.exception.name)
            else:
                exc_types.add("Exception")
            exc_val = exc_abs
        else:
            exc_types.add("Exception")
        new_exc = ExceptionState(
            active=True,
            exception_types=exc_types,
            exception_value=exc_val,
            may_propagate=True,
        )
        if node.cause is not None:
            cause_val = _eval_expr(node.cause, state)
            if new_exc.exception_value is not None:
                new_exc.exception_value.extra["__cause__"] = cause_val
        new_state.exception_state = new_state.exception_state.join(new_exc)
        return new_state

    def _transform_try_except(self, node: TryExceptNode, state: AbstractState) -> AbstractState:
        body_block = BlockNode(statements=node.body)
        body_transformer = _make_composite_transformer()
        body_state = body_transformer.transform_block(body_block, state)
        normal_state = body_state.deep_copy()
        normal_state.exception_state = ExceptionState()
        exc_state = body_state.deep_copy()
        handler_states: List[AbstractState] = []
        for handler in node.handlers:
            handler_state = self._apply_handler(handler, exc_state)
            handler_states.append(handler_state)
        merged = normal_state
        for hs in handler_states:
            merged = merged.join_with(hs)
        if node.orelse:
            else_block = BlockNode(statements=node.orelse)
            else_state = body_transformer.transform_block(else_block, normal_state)
            merged = merged.join_with(else_state)
        if node.finalbody:
            finally_block = BlockNode(statements=node.finalbody)
            merged = body_transformer.transform_block(finally_block, merged)
        return merged

    def _apply_handler(self, handler: ExceptClause, exc_state: AbstractState) -> AbstractState:
        handler_entry = exc_state.deep_copy()
        if handler.exception_type is not None:
            matched_types = {handler.exception_type}
            caught = exc_state.exception_state.exception_types & matched_types
            if not caught and exc_state.exception_state.exception_types:
                base_exceptions = {
                    "Exception": {"ValueError", "TypeError", "KeyError", "IndexError",
                                  "AttributeError", "RuntimeError", "IOError",
                                  "FileNotFoundError", "PermissionError", "StopIteration",
                                  "ZeroDivisionError", "OverflowError", "OSError",
                                  "NotImplementedError", "ImportError", "NameError"},
                    "BaseException": {"Exception", "KeyboardInterrupt", "SystemExit",
                                      "GeneratorExit"},
                }
                children = base_exceptions.get(handler.exception_type, set())
                caught = exc_state.exception_state.exception_types & (matched_types | children)
            if caught:
                handler_entry.exception_state = ExceptionState(
                    active=True,
                    exception_types=caught,
                    exception_value=exc_state.exception_state.exception_value,
                    may_propagate=False,
                )
            else:
                return AbstractState.bottom()
        else:
            handler_entry.exception_state = ExceptionState(
                active=True,
                exception_types=exc_state.exception_state.exception_types.copy(),
                exception_value=exc_state.exception_state.exception_value,
                may_propagate=False,
            )
        if handler.name is not None:
            exc_val = handler_entry.exception_state.exception_value
            if exc_val is not None:
                handler_entry.variables[handler.name] = exc_val
            else:
                handler_entry.variables[handler.name] = AbstractValue(
                    type_tags={TypeTag.OBJECT},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
        body_block = BlockNode(statements=handler.body)
        body_transformer = _make_composite_transformer()
        handler_result = body_transformer.transform_block(body_block, handler_entry)
        handler_result.exception_state = ExceptionState()
        return handler_result

    def apply_finally(self, state: AbstractState, finally_body: List[IRNode]) -> AbstractState:
        block = BlockNode(statements=finally_body)
        transformer = _make_composite_transformer()
        return transformer.transform_block(block, state)

    def chain_exception(self, state: AbstractState, cause: AbstractValue) -> AbstractState:
        new_state = state.deep_copy()
        if new_state.exception_state.active and new_state.exception_state.exception_value is not None:
            new_state.exception_state.exception_value.extra["__cause__"] = cause
            new_state.exception_state.exception_value.extra["__context__"] = cause
        return new_state


# ---------------------------------------------------------------------------
# LoopTransformer
# ---------------------------------------------------------------------------

_LOOP_UNROLL_THRESHOLD = 10


class LoopTransformer(AbstractTransformer):
    def __init__(self, max_iterations: int = 100, widening_delay: int = 3) -> None:
        self._max_iterations = max_iterations
        self._widening_delay = widening_delay

    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, WhileNode):
            return self._transform_while(node, state)
        if isinstance(node, ForNode):
            return self._transform_for(node, state)
        return state

    def _transform_while(self, node: WhileNode, state: AbstractState) -> AbstractState:
        guard_transformer = GuardTransformer()
        body_transformer = _make_composite_transformer()
        can_unroll = self._can_unroll_while(node, state)
        if can_unroll:
            return self._unroll_while(node, state, guard_transformer, body_transformer)
        current = state.deep_copy()
        current.loop_depth += 1
        iteration = 0
        while iteration < self._max_iterations:
            guard_node = GuardNode(condition=node.condition)
            true_state, false_state = guard_transformer.transform_guard_both(guard_node, current)
            if true_state.is_bottom():
                if node.orelse:
                    else_block = BlockNode(statements=node.orelse)
                    false_state = body_transformer.transform_block(else_block, false_state)
                result = false_state
                result.loop_depth = max(0, result.loop_depth - 1)
                result = self._merge_break_states(result, current)
                return result
            body_block = BlockNode(statements=node.body)
            body_state = body_transformer.transform_block(body_block, true_state)
            body_state = self._handle_continue(body_state, true_state)
            if iteration < self._widening_delay:
                next_state = current.join_with(body_state)
            else:
                next_state = current.widen_with(body_state)
            if current.equals(next_state):
                guard_node_final = GuardNode(condition=node.condition)
                _, exit_state = guard_transformer.transform_guard_both(guard_node_final, next_state)
                if node.orelse:
                    else_block = BlockNode(statements=node.orelse)
                    exit_state = body_transformer.transform_block(else_block, exit_state)
                exit_state.loop_depth = max(0, exit_state.loop_depth - 1)
                exit_state = self._merge_break_states(exit_state, current)
                return exit_state
            current = next_state
            iteration += 1
        guard_node_final = GuardNode(condition=node.condition)
        _, exit_state = guard_transformer.transform_guard_both(guard_node_final, current)
        exit_state.loop_depth = max(0, exit_state.loop_depth - 1)
        exit_state = self._merge_break_states(exit_state, current)
        return exit_state

    def _can_unroll_while(self, node: WhileNode, state: AbstractState) -> bool:
        if isinstance(node.condition, CompareExpr) and len(node.condition.ops) == 1:
            op = node.condition.ops[0]
            left = node.condition.left
            right = node.condition.comparators[0]
            if isinstance(left, VarExpr) and isinstance(right, ConstExpr):
                val = state.get_value(left.name)
                if val.interval.is_const() and val.interval.lo is not None:
                    if isinstance(right.value, (int, float)):
                        start = val.interval.lo
                        end = float(right.value)
                        if op in ("<", "Lt"):
                            count = max(0, end - start)
                        elif op in ("<=", "LtE"):
                            count = max(0, end - start + 1)
                        else:
                            return False
                        if 0 < count <= _LOOP_UNROLL_THRESHOLD:
                            return True
        return False

    def _unroll_while(self, node: WhileNode, state: AbstractState, guard_transformer: GuardTransformer, body_transformer: AbstractTransformer) -> AbstractState:
        current = state.deep_copy()
        current.loop_depth += 1
        for _ in range(_LOOP_UNROLL_THRESHOLD):
            guard_node = GuardNode(condition=node.condition)
            true_state, false_state = guard_transformer.transform_guard_both(guard_node, current)
            if true_state.is_bottom():
                result = false_state
                result.loop_depth = max(0, result.loop_depth - 1)
                return result
            body_block = BlockNode(statements=node.body)
            body_state = body_transformer.transform_block(body_block, true_state)
            body_state = self._handle_continue(body_state, true_state)
            has_break = False
            for stmt in node.body:
                if isinstance(stmt, BreakNode):
                    has_break = True
                    break
            if has_break:
                result = body_state
                result.loop_depth = max(0, result.loop_depth - 1)
                return result
            current = body_state
        guard_node_f = GuardNode(condition=node.condition)
        _, exit_state = guard_transformer.transform_guard_both(guard_node_f, current)
        exit_state.loop_depth = max(0, exit_state.loop_depth - 1)
        return exit_state

    def _transform_for(self, node: ForNode, state: AbstractState) -> AbstractState:
        body_transformer = _make_composite_transformer()
        iter_val = _eval_expr(node.iter_expr, state)
        elem_type = self._infer_iter_element_type(iter_val, node.iter_expr, state)
        can_unroll = self._can_unroll_for(iter_val, node.iter_expr, state)
        if can_unroll:
            return self._unroll_for(node, state, iter_val, elem_type, body_transformer)
        current = state.deep_copy()
        current.loop_depth += 1
        current.variables[node.target] = elem_type
        iteration = 0
        while iteration < self._max_iterations:
            body_block = BlockNode(statements=node.body)
            body_state = body_transformer.transform_block(body_block, current)
            body_state = self._handle_continue(body_state, current)
            new_elem = elem_type.join(body_state.get_value(node.target))
            body_state.variables[node.target] = new_elem
            if iteration < self._widening_delay:
                next_state = current.join_with(body_state)
            else:
                next_state = current.widen_with(body_state)
            if current.equals(next_state):
                exit_state = next_state.deep_copy()
                if node.orelse:
                    else_block = BlockNode(statements=node.orelse)
                    exit_state = body_transformer.transform_block(else_block, exit_state)
                exit_state.loop_depth = max(0, exit_state.loop_depth - 1)
                exit_state = self._merge_break_states(exit_state, current)
                return exit_state
            current = next_state
            iteration += 1
        result = current.deep_copy()
        result.loop_depth = max(0, result.loop_depth - 1)
        result = self._merge_break_states(result, current)
        return result

    def _can_unroll_for(self, iter_val: AbstractValue, iter_expr: Expr, state: AbstractState) -> bool:
        if isinstance(iter_expr, CallExpr) and isinstance(iter_expr.func, VarExpr) and iter_expr.func.name == "range":
            if len(iter_expr.args) == 1:
                end_val = _eval_expr(iter_expr.args[0], state)
                if end_val.interval.is_const() and end_val.interval.lo is not None:
                    count = end_val.interval.lo
                    if 0 < count <= _LOOP_UNROLL_THRESHOLD:
                        return True
            elif len(iter_expr.args) == 2:
                start_val = _eval_expr(iter_expr.args[0], state)
                end_val = _eval_expr(iter_expr.args[1], state)
                if (start_val.interval.is_const() and end_val.interval.is_const()
                        and start_val.interval.lo is not None and end_val.interval.lo is not None):
                    count = end_val.interval.lo - start_val.interval.lo
                    if 0 < count <= _LOOP_UNROLL_THRESHOLD:
                        return True
        if iter_val.is_container() and iter_val.container_length.is_const():
            if iter_val.container_length.lo is not None and 0 < iter_val.container_length.lo <= _LOOP_UNROLL_THRESHOLD:
                return True
        return False

    def _unroll_for(self, node: ForNode, state: AbstractState, iter_val: AbstractValue, elem_type: AbstractValue, body_transformer: AbstractTransformer) -> AbstractState:
        current = state.deep_copy()
        current.loop_depth += 1
        count = 0
        if isinstance(node.iter_expr, CallExpr) and isinstance(node.iter_expr.func, VarExpr) and node.iter_expr.func.name == "range":
            if len(node.iter_expr.args) == 1:
                end_val = _eval_expr(node.iter_expr.args[0], state)
                if end_val.interval.is_const() and end_val.interval.lo is not None:
                    count = int(end_val.interval.lo)
                    for i in range(count):
                        current.variables[node.target] = AbstractValue.from_const(i)
                        body_block = BlockNode(statements=node.body)
                        current = body_transformer.transform_block(body_block, current)
                        current = self._handle_continue(current, current)
                    current.loop_depth = max(0, current.loop_depth - 1)
                    return current
            elif len(node.iter_expr.args) == 2:
                start_val = _eval_expr(node.iter_expr.args[0], state)
                end_val = _eval_expr(node.iter_expr.args[1], state)
                if (start_val.interval.is_const() and end_val.interval.is_const()
                        and start_val.interval.lo is not None and end_val.interval.lo is not None):
                    start = int(start_val.interval.lo)
                    end = int(end_val.interval.lo)
                    for i in range(start, end):
                        current.variables[node.target] = AbstractValue.from_const(i)
                        body_block = BlockNode(statements=node.body)
                        current = body_transformer.transform_block(body_block, current)
                        current = self._handle_continue(current, current)
                    current.loop_depth = max(0, current.loop_depth - 1)
                    return current

        if iter_val.container_length.is_const() and iter_val.container_length.lo is not None:
            count = int(iter_val.container_length.lo)
        else:
            count = _LOOP_UNROLL_THRESHOLD
        for _ in range(count):
            current.variables[node.target] = elem_type
            body_block = BlockNode(statements=node.body)
            current = body_transformer.transform_block(body_block, current)
            current = self._handle_continue(current, current)
        current.loop_depth = max(0, current.loop_depth - 1)
        return current

    def _infer_iter_element_type(self, iter_val: AbstractValue, iter_expr: Expr, state: AbstractState) -> AbstractValue:
        if isinstance(iter_expr, CallExpr) and isinstance(iter_expr.func, VarExpr):
            fname = iter_expr.func.name
            if fname == "range":
                if len(iter_expr.args) == 1:
                    end_val = _eval_expr(iter_expr.args[0], state)
                    return AbstractValue(
                        interval=Interval(lo=0.0, hi=end_val.interval.hi if end_val.interval.hi is not None else None),
                        type_tags={TypeTag.INT},
                        nullity=NullityTag.DEFINITELY_NOT_NULL,
                    )
                elif len(iter_expr.args) >= 2:
                    start_val = _eval_expr(iter_expr.args[0], state)
                    end_val = _eval_expr(iter_expr.args[1], state)
                    return AbstractValue(
                        interval=Interval(lo=start_val.interval.lo, hi=end_val.interval.hi),
                        type_tags={TypeTag.INT},
                        nullity=NullityTag.DEFINITELY_NOT_NULL,
                    )
                return AbstractValue.from_type(TypeTag.INT)
            if fname == "enumerate":
                return AbstractValue(
                    type_tags={TypeTag.TUPLE},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    container_length=Interval.const(2.0),
                )
            if fname == "zip":
                return AbstractValue(
                    type_tags={TypeTag.TUPLE},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
        if TypeTag.STR in iter_val.type_tags:
            return AbstractValue.from_type(TypeTag.STR)
        if TypeTag.DICT in iter_val.type_tags:
            return AbstractValue.top()
        if TypeTag.BYTES in iter_val.type_tags:
            return AbstractValue.from_type(TypeTag.INT)
        return AbstractValue.top()

    def _handle_continue(self, body_state: AbstractState, loop_state: AbstractState) -> AbstractState:
        if body_state.continue_states:
            result = body_state
            for cs in body_state.continue_states:
                result = result.join_with(cs)
            result.continue_states = []
            return result
        return body_state

    def _merge_break_states(self, exit_state: AbstractState, loop_state: AbstractState) -> AbstractState:
        result = exit_state
        if loop_state.break_states:
            for bs in loop_state.break_states:
                result = result.join_with(bs)
        if exit_state.break_states:
            for bs in exit_state.break_states:
                result = result.join_with(bs)
            result.break_states = []
        return result

    def loop_invariant_maintenance(self, state: AbstractState, body_vars: Set[str]) -> AbstractState:
        result = state.deep_copy()
        for var in body_vars:
            if var in result.variables:
                val = result.variables[var]
                result.variables[var] = val
        return result


# ---------------------------------------------------------------------------
# ComprehensionTransformer
# ---------------------------------------------------------------------------

class ComprehensionTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, ComprehensionNode):
            return self._transform_comprehension(node, state)
        return state

    def _transform_comprehension(self, node: ComprehensionNode, state: AbstractState) -> AbstractState:
        new_state = state.deep_copy()
        comp_state = state.deep_copy()
        for target, iter_expr, conditions in node.generators:
            iter_val = _eval_expr(iter_expr, comp_state)
            elem_type = self._infer_generator_element(iter_val, iter_expr, comp_state)
            comp_state.variables[target] = elem_type
            guard_transformer = GuardTransformer()
            for cond in conditions:
                guard_node = GuardNode(condition=cond)
                comp_state = guard_transformer.transform(guard_node, comp_state)
        elem_val = _eval_expr(node.element, comp_state)
        result_length = self._estimate_comprehension_length(node, state)
        if node.kind == "list":
            result_val = AbstractValue(
                type_tags={TypeTag.LIST},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=result_length,
            )
        elif node.kind == "set":
            result_val = AbstractValue(
                type_tags={TypeTag.SET},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval(lo=0.0, hi=result_length.hi),
            )
        elif node.kind == "dict":
            key_val = _eval_expr(node.key_expr, comp_state) if node.key_expr is not None else AbstractValue.top()
            result_val = AbstractValue(
                type_tags={TypeTag.DICT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
                container_length=Interval(lo=0.0, hi=result_length.hi),
            )
        elif node.kind == "generator":
            result_val = AbstractValue(
                type_tags={TypeTag.OBJECT},
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        else:
            result_val = AbstractValue.top()
        if node.target is not None:
            new_state.variables[node.target] = result_val
        return new_state

    def _infer_generator_element(self, iter_val: AbstractValue, iter_expr: Expr, state: AbstractState) -> AbstractValue:
        if isinstance(iter_expr, CallExpr) and isinstance(iter_expr.func, VarExpr):
            fname = iter_expr.func.name
            if fname == "range":
                return AbstractValue.from_type(TypeTag.INT)
            if fname == "enumerate":
                return AbstractValue(
                    type_tags={TypeTag.TUPLE},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    container_length=Interval.const(2.0),
                )
            if fname == "zip":
                return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if TypeTag.STR in iter_val.type_tags:
            return AbstractValue.from_type(TypeTag.STR)
        if TypeTag.DICT in iter_val.type_tags:
            return AbstractValue.top()
        return AbstractValue.top()

    def _estimate_comprehension_length(self, node: ComprehensionNode, state: AbstractState) -> Interval:
        if not node.generators:
            return Interval.const(0.0)
        first_target, first_iter, first_conds = node.generators[0]
        iter_val = _eval_expr(first_iter, state)
        if iter_val.is_container():
            base_len = iter_val.container_length
        else:
            base_len = Interval(lo=0.0, hi=None)
        if first_conds:
            base_len = Interval(lo=0.0, hi=base_len.hi)
        for target, iter_expr, conditions in node.generators[1:]:
            inner_val = _eval_expr(iter_expr, state)
            if inner_val.is_container():
                inner_len = inner_val.container_length
            else:
                inner_len = Interval(lo=0.0, hi=None)
            base_len = _interval_mul(base_len, inner_len)
            if conditions:
                base_len = Interval(lo=0.0, hi=base_len.hi)
        return base_len


# ---------------------------------------------------------------------------
# ContainerTransformer
# ---------------------------------------------------------------------------

class ContainerTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, ContainerOpNode):
            return self._transform_container_op(node, state)
        if isinstance(node, SubscriptNode):
            return self._transform_subscript(node, state)
        return state

    def _transform_container_op(self, node: ContainerOpNode, state: AbstractState) -> AbstractState:
        container_val = _eval_expr(node.container, state)
        args_vals = [_eval_expr(a, state) for a in node.args]
        new_state = state.deep_copy()
        if TypeTag.LIST in container_val.type_tags:
            result_val, updated_container = self._apply_list_op(container_val, node.op, args_vals)
        elif TypeTag.DICT in container_val.type_tags:
            result_val, updated_container = self._apply_dict_op(container_val, node.op, args_vals)
        elif TypeTag.SET in container_val.type_tags:
            result_val, updated_container = self._apply_set_op(container_val, node.op, args_vals)
        elif TypeTag.TUPLE in container_val.type_tags:
            result_val, updated_container = self._apply_tuple_op(container_val, node.op, args_vals)
        else:
            result_val = AbstractValue.top()
            updated_container = container_val
        if isinstance(node.container, VarExpr):
            new_state.variables[node.container.name] = updated_container
        if node.target is not None:
            new_state.variables[node.target] = result_val
        return new_state

    def _apply_list_op(self, container: AbstractValue, op: str, args: List[AbstractValue]) -> Tuple[AbstractValue, AbstractValue]:
        updated = copy.deepcopy(container)
        if op == "append":
            updated.container_length = _interval_add(updated.container_length, Interval.const(1.0))
            return AbstractValue.from_const(None), updated
        if op == "extend":
            if args and args[0].is_container():
                updated.container_length = _interval_add(updated.container_length, args[0].container_length)
            else:
                updated.container_length = Interval(lo=updated.container_length.lo, hi=None)
            return AbstractValue.from_const(None), updated
        if op == "insert":
            updated.container_length = _interval_add(updated.container_length, Interval.const(1.0))
            return AbstractValue.from_const(None), updated
        if op == "pop":
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            hi = (updated.container_length.hi - 1.0) if updated.container_length.hi is not None else None
            if hi is not None:
                hi = max(0.0, hi)
            updated.container_length = Interval(lo=lo, hi=hi)
            return AbstractValue.top(), updated
        if op == "remove":
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            hi = (updated.container_length.hi - 1.0) if updated.container_length.hi is not None else None
            if hi is not None:
                hi = max(0.0, hi)
            updated.container_length = Interval(lo=lo, hi=hi)
            return AbstractValue.from_const(None), updated
        if op == "clear":
            updated.container_length = Interval.const(0.0)
            return AbstractValue.from_const(None), updated
        if op == "copy":
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        if op == "sort" or op == "reverse":
            return AbstractValue.from_const(None), updated
        if op == "count":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), updated
        if op == "index":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), updated
        if op == "getitem":
            return AbstractValue.top(), updated
        if op == "setitem":
            return AbstractValue.from_const(None), updated
        if op == "slice":
            new_len = Interval(lo=0.0, hi=updated.container_length.hi)
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), updated
        return AbstractValue.top(), updated

    def _apply_dict_op(self, container: AbstractValue, op: str, args: List[AbstractValue]) -> Tuple[AbstractValue, AbstractValue]:
        updated = copy.deepcopy(container)
        if op == "setitem":
            new_len = Interval(
                lo=updated.container_length.lo,
                hi=(updated.container_length.hi + 1.0) if updated.container_length.hi is not None else None,
            )
            updated.container_length = new_len
            return AbstractValue.from_const(None), updated
        if op == "getitem":
            return AbstractValue.top(), updated
        if op == "get":
            if len(args) >= 2:
                return AbstractValue.top().join(args[1]), updated
            return AbstractValue(type_tags={TypeTag.ANY}, nullity=NullityTag.MAYBE_NULL), updated
        if op == "update":
            if args and args[0].is_container():
                new_len = Interval(
                    lo=_max_opt(updated.container_length.lo, args[0].container_length.lo),
                    hi=None if updated.container_length.hi is None or args[0].container_length.hi is None else updated.container_length.hi + args[0].container_length.hi,
                )
                updated.container_length = new_len
            return AbstractValue.from_const(None), updated
        if op == "keys":
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        if op == "values":
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        if op == "items":
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        if op == "pop":
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            hi = updated.container_length.hi
            updated.container_length = Interval(lo=lo, hi=hi)
            if len(args) >= 2:
                return AbstractValue.top().join(args[1]), updated
            return AbstractValue.top(), updated
        if op == "popitem":
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            hi = (updated.container_length.hi - 1.0) if updated.container_length.hi is not None else None
            if hi is not None:
                hi = max(0.0, hi)
            updated.container_length = Interval(lo=lo, hi=hi)
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(2.0)), updated
        if op == "setdefault":
            new_len = Interval(
                lo=updated.container_length.lo,
                hi=(updated.container_length.hi + 1.0) if updated.container_length.hi is not None else None,
            )
            updated.container_length = new_len
            if len(args) >= 2:
                return args[1], updated
            return AbstractValue(nullity=NullityTag.MAYBE_NULL, type_tags={TypeTag.NONE}), updated
        if op == "clear":
            updated.container_length = Interval.const(0.0)
            return AbstractValue.from_const(None), updated
        if op == "copy":
            return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        return AbstractValue.top(), updated

    def _apply_set_op(self, container: AbstractValue, op: str, args: List[AbstractValue]) -> Tuple[AbstractValue, AbstractValue]:
        updated = copy.deepcopy(container)
        if op == "add":
            new_len = Interval(
                lo=updated.container_length.lo,
                hi=(updated.container_length.hi + 1.0) if updated.container_length.hi is not None else None,
            )
            updated.container_length = new_len
            return AbstractValue.from_const(None), updated
        if op in ("remove", "discard"):
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            updated.container_length = Interval(lo=lo, hi=updated.container_length.hi)
            return AbstractValue.from_const(None), updated
        if op == "pop":
            lo = max(0.0, (updated.container_length.lo or 0.0) - 1.0)
            hi = (updated.container_length.hi - 1.0) if updated.container_length.hi is not None else None
            if hi is not None:
                hi = max(0.0, hi)
            updated.container_length = Interval(lo=lo, hi=hi)
            return AbstractValue.top(), updated
        if op == "clear":
            updated.container_length = Interval.const(0.0)
            return AbstractValue.from_const(None), updated
        if op == "union":
            if args and args[0].is_container():
                new_len = Interval(
                    lo=_max_opt(updated.container_length.lo, args[0].container_length.lo),
                    hi=None if updated.container_length.hi is None or args[0].container_length.hi is None else updated.container_length.hi + args[0].container_length.hi,
                )
                return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), updated
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL), updated
        if op == "intersection":
            if args and args[0].is_container():
                new_len = Interval(lo=0.0, hi=_min_opt(updated.container_length.hi, args[0].container_length.hi))
                return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), updated
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval(lo=0.0, hi=updated.container_length.hi)), updated
        if op == "difference":
            new_len = Interval(lo=0.0, hi=updated.container_length.hi)
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), updated
        if op == "symmetric_difference":
            if args and args[0].is_container():
                new_len = Interval(
                    lo=0.0,
                    hi=None if updated.container_length.hi is None or args[0].container_length.hi is None else updated.container_length.hi + args[0].container_length.hi,
                )
                return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), updated
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL), updated
        if op == "copy":
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=updated.container_length), updated
        return AbstractValue.top(), updated

    def _apply_tuple_op(self, container: AbstractValue, op: str, args: List[AbstractValue]) -> Tuple[AbstractValue, AbstractValue]:
        if op == "getitem":
            return AbstractValue.top(), container
        if op == "count":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), container
        if op == "index":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), container
        if op == "unpack":
            return AbstractValue.top(), container
        if op == "slice":
            new_len = Interval(lo=0.0, hi=container.container_length.hi)
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len), container
        return AbstractValue.top(), container

    def _transform_subscript(self, node: SubscriptNode, state: AbstractState) -> AbstractState:
        obj_val = _eval_expr(node.obj, state)
        idx_val = _eval_expr(node.index, state)
        new_state = state.deep_copy()
        if TypeTag.LIST in obj_val.type_tags or TypeTag.TUPLE in obj_val.type_tags:
            if idx_val.interval.lo is not None and obj_val.container_length.hi is not None:
                if idx_val.interval.lo >= obj_val.container_length.hi:
                    new_state.exception_state = ExceptionState(
                        active=True,
                        exception_types={"IndexError"},
                        may_propagate=True,
                    )
        result_val = AbstractValue.top()
        if TypeTag.STR in obj_val.type_tags:
            result_val = AbstractValue.from_type(TypeTag.STR)
        elif TypeTag.BYTES in obj_val.type_tags:
            result_val = AbstractValue.from_type(TypeTag.INT)
        if node.target is not None:
            new_state.variables[node.target] = result_val
        return new_state


# ---------------------------------------------------------------------------
# StringTransformer
# ---------------------------------------------------------------------------

class StringTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, StringOpNode):
            return self._transform_string_op(node, state)
        return state

    def _transform_string_op(self, node: StringOpNode, state: AbstractState) -> AbstractState:
        string_val = _eval_expr(node.string_expr, state)
        args_vals = [_eval_expr(a, state) for a in node.args]
        new_state = state.deep_copy()
        result_val = self._dispatch_string_op(string_val, node.op, args_vals)
        if node.target is not None:
            new_state.variables[node.target] = result_val
        return new_state

    def _dispatch_string_op(self, string_val: AbstractValue, op: str, args: List[AbstractValue]) -> AbstractValue:
        if op == "concat":
            return self._string_concat(string_val, args)
        if op == "format":
            return self._string_format(string_val, args)
        if op == "fstring":
            return AbstractValue.from_type(TypeTag.STR)
        if op in ("strip", "lstrip", "rstrip"):
            return self._string_strip(string_val, op, args)
        if op in ("lower", "upper", "title", "capitalize", "swapcase"):
            return self._string_case_transform(string_val, op)
        if op == "replace":
            return self._string_replace(string_val, args)
        if op in ("find", "rfind"):
            return AbstractValue(interval=Interval(lo=-1.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if op in ("index", "rindex"):
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if op == "count":
            return AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
        if op in ("startswith", "endswith"):
            return self._string_predicate(string_val, op, args)
        if op in ("isalpha", "isdigit", "isalnum", "isspace", "isupper", "islower", "istitle"):
            return self._string_is_predicate(string_val, op)
        if op in ("split", "rsplit"):
            return self._string_split(string_val, op, args)
        if op == "join":
            return self._string_join(string_val, args)
        if op == "encode":
            return AbstractValue.from_type(TypeTag.BYTES)
        if op == "decode":
            return AbstractValue.from_type(TypeTag.STR)
        if op in ("center", "ljust", "rjust"):
            return AbstractValue.from_type(TypeTag.STR)
        if op == "zfill":
            return AbstractValue.from_type(TypeTag.STR)
        if op == "expandtabs":
            return AbstractValue.from_type(TypeTag.STR)
        if op in ("partition", "rpartition"):
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(3.0))
        if op == "maketrans":
            return AbstractValue.from_type(TypeTag.DICT)
        if op == "translate":
            return AbstractValue.from_type(TypeTag.STR)
        return AbstractValue.top()

    def _string_concat(self, string_val: AbstractValue, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return string_val
        other = args[0]
        if string_val.string_values is not None and other.string_values is not None:
            combined: Set[str] = set()
            for s1 in string_val.string_values:
                for s2 in other.string_values:
                    combined.add(s1 + s2)
            if len(combined) <= _MAX_STRING_VALUES:
                return AbstractValue(
                    type_tags={TypeTag.STR},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    string_values=frozenset(combined),
                )
        return AbstractValue.from_type(TypeTag.STR)

    def _string_format(self, string_val: AbstractValue, args: List[AbstractValue]) -> AbstractValue:
        return AbstractValue.from_type(TypeTag.STR)

    def _string_strip(self, string_val: AbstractValue, op: str, args: List[AbstractValue]) -> AbstractValue:
        if string_val.string_values is not None:
            new_vals: Set[str] = set()
            for s in string_val.string_values:
                if op == "strip":
                    new_vals.add(s.strip())
                elif op == "lstrip":
                    new_vals.add(s.lstrip())
                elif op == "rstrip":
                    new_vals.add(s.rstrip())
            if len(new_vals) <= _MAX_STRING_VALUES:
                return AbstractValue(
                    type_tags={TypeTag.STR},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    string_values=frozenset(new_vals),
                )
        return AbstractValue.from_type(TypeTag.STR)

    def _string_case_transform(self, string_val: AbstractValue, op: str) -> AbstractValue:
        if string_val.string_values is not None:
            new_vals: Set[str] = set()
            for s in string_val.string_values:
                if op == "lower":
                    new_vals.add(s.lower())
                elif op == "upper":
                    new_vals.add(s.upper())
                elif op == "title":
                    new_vals.add(s.title())
                elif op == "capitalize":
                    new_vals.add(s.capitalize())
                elif op == "swapcase":
                    new_vals.add(s.swapcase())
            if len(new_vals) <= _MAX_STRING_VALUES:
                return AbstractValue(
                    type_tags={TypeTag.STR},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    string_values=frozenset(new_vals),
                )
        return AbstractValue.from_type(TypeTag.STR)

    def _string_replace(self, string_val: AbstractValue, args: List[AbstractValue]) -> AbstractValue:
        if len(args) >= 2 and string_val.string_values is not None:
            old_strs = args[0].string_values
            new_strs = args[1].string_values
            if old_strs is not None and new_strs is not None:
                result_vals: Set[str] = set()
                for s in string_val.string_values:
                    for old_s in old_strs:
                        for new_s in new_strs:
                            result_vals.add(s.replace(old_s, new_s))
                if len(result_vals) <= _MAX_STRING_VALUES:
                    return AbstractValue(
                        type_tags={TypeTag.STR},
                        nullity=NullityTag.DEFINITELY_NOT_NULL,
                        string_values=frozenset(result_vals),
                    )
        return AbstractValue.from_type(TypeTag.STR)

    def _string_predicate(self, string_val: AbstractValue, op: str, args: List[AbstractValue]) -> AbstractValue:
        if string_val.string_values is not None and args and args[0].string_values is not None:
            results: Set[bool] = set()
            for s in string_val.string_values:
                for prefix in args[0].string_values:
                    if op == "startswith":
                        results.add(s.startswith(prefix))
                    elif op == "endswith":
                        results.add(s.endswith(prefix))
            if len(results) == 1:
                return AbstractValue.from_const(results.pop())
        return AbstractValue.from_type(TypeTag.BOOL)

    def _string_is_predicate(self, string_val: AbstractValue, op: str) -> AbstractValue:
        if string_val.string_values is not None:
            results: Set[bool] = set()
            for s in string_val.string_values:
                try:
                    if op == "isalpha":
                        results.add(s.isalpha())
                    elif op == "isdigit":
                        results.add(s.isdigit())
                    elif op == "isalnum":
                        results.add(s.isalnum())
                    elif op == "isspace":
                        results.add(s.isspace())
                    elif op == "isupper":
                        results.add(s.isupper())
                    elif op == "islower":
                        results.add(s.islower())
                    elif op == "istitle":
                        results.add(s.istitle())
                except ValueError:
                    return AbstractValue.from_type(TypeTag.BOOL)
            if len(results) == 1:
                return AbstractValue.from_const(results.pop())
        return AbstractValue.from_type(TypeTag.BOOL)

    def _string_split(self, string_val: AbstractValue, op: str, args: List[AbstractValue]) -> AbstractValue:
        if string_val.string_values is not None:
            min_len = float("inf")
            max_len = 0.0
            for s in string_val.string_values:
                if op == "split":
                    parts = s.split() if not args or (args[0].string_values is None) else None
                    if parts is not None:
                        min_len = min(min_len, len(parts))
                        max_len = max(max_len, len(parts))
                    elif args and args[0].string_values is not None:
                        for sep in args[0].string_values:
                            parts = s.split(sep)
                            min_len = min(min_len, len(parts))
                            max_len = max(max_len, len(parts))
                else:
                    parts = s.rsplit() if not args or (args[0].string_values is None) else None
                    if parts is not None:
                        min_len = min(min_len, len(parts))
                        max_len = max(max_len, len(parts))
                    elif args and args[0].string_values is not None:
                        for sep in args[0].string_values:
                            parts = s.rsplit(sep)
                            min_len = min(min_len, len(parts))
                            max_len = max(max_len, len(parts))
            if min_len != float("inf"):
                return AbstractValue(
                    type_tags={TypeTag.LIST},
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                    container_length=Interval(lo=min_len, hi=max_len),
                )
        return AbstractValue(
            type_tags={TypeTag.LIST},
            nullity=NullityTag.DEFINITELY_NOT_NULL,
            container_length=Interval(lo=1.0, hi=None),
        )

    def _string_join(self, string_val: AbstractValue, args: List[AbstractValue]) -> AbstractValue:
        return AbstractValue.from_type(TypeTag.STR)


# ---------------------------------------------------------------------------
# NumericTransformer
# ---------------------------------------------------------------------------

class NumericTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, BinaryOpNode):
            return self._transform_binary_op(node, state)
        if isinstance(node, UnaryOpNode):
            return self._transform_unary_op(node, state)
        if isinstance(node, CompareNode):
            return self._transform_compare(node, state)
        return state

    def _transform_binary_op(self, node: BinaryOpNode, state: AbstractState) -> AbstractState:
        left_val = _eval_expr(node.left, state)
        right_val = _eval_expr(node.right, state)
        new_state = state.deep_copy()
        if node.op == "+":
            result = self._add(left_val, right_val)
        elif node.op == "-":
            result = self._subtract(left_val, right_val)
        elif node.op == "*":
            result = self._multiply(left_val, right_val)
        elif node.op == "/":
            result, may_raise = self._divide(left_val, right_val)
            if may_raise:
                new_state.exception_state = new_state.exception_state.join(
                    ExceptionState(active=True, exception_types={"ZeroDivisionError"}, may_propagate=True)
                )
        elif node.op == "//":
            result, may_raise = self._floor_divide(left_val, right_val)
            if may_raise:
                new_state.exception_state = new_state.exception_state.join(
                    ExceptionState(active=True, exception_types={"ZeroDivisionError"}, may_propagate=True)
                )
        elif node.op == "%":
            result, may_raise = self._modulo(left_val, right_val)
            if may_raise:
                new_state.exception_state = new_state.exception_state.join(
                    ExceptionState(active=True, exception_types={"ZeroDivisionError"}, may_propagate=True)
                )
        elif node.op == "**":
            result = self._power(left_val, right_val)
        else:
            result = _eval_binop(node.op, left_val, right_val)
        if node.target is not None:
            new_state.variables[node.target] = result
        return new_state

    def _transform_unary_op(self, node: UnaryOpNode, state: AbstractState) -> AbstractState:
        operand_val = _eval_expr(node.operand, state)
        new_state = state.deep_copy()
        result = _eval_unaryop(node.op, operand_val)
        if node.target is not None:
            new_state.variables[node.target] = result
        return new_state

    def _transform_compare(self, node: CompareNode, state: AbstractState) -> AbstractState:
        left_val = _eval_expr(node.left, state)
        new_state = state.deep_copy()
        current = left_val
        result = AbstractValue.from_type(TypeTag.BOOL)
        for op, comp_expr in zip(node.ops, node.comparators):
            right_val = _eval_expr(comp_expr, state)
            cmp_result = _eval_comparison(op, current, right_val)
            result = result.meet(cmp_result)
            current = right_val
        if node.target is not None:
            new_state.variables[node.target] = result
        return new_state

    def _add(self, left: AbstractValue, right: AbstractValue) -> AbstractValue:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom()
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_add(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            elif TypeTag.COMPLEX in left.type_tags or TypeTag.COMPLEX in right.type_tags:
                tags.add(TypeTag.COMPLEX)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return _eval_binop("+", left, right)

    def _subtract(self, left: AbstractValue, right: AbstractValue) -> AbstractValue:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom()
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_sub(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            elif TypeTag.COMPLEX in left.type_tags or TypeTag.COMPLEX in right.type_tags:
                tags.add(TypeTag.COMPLEX)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()

    def _multiply(self, left: AbstractValue, right: AbstractValue) -> AbstractValue:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom()
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_mul(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            elif TypeTag.COMPLEX in left.type_tags or TypeTag.COMPLEX in right.type_tags:
                tags.add(TypeTag.COMPLEX)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return _eval_binop("*", left, right)

    def _divide(self, left: AbstractValue, right: AbstractValue) -> Tuple[AbstractValue, bool]:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom(), False
        if left.is_numeric() and right.is_numeric():
            result_iv, may_raise = _interval_div(left.interval, right.interval)
            return AbstractValue(interval=result_iv, type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL), may_raise
        return AbstractValue.top(), False

    def _floor_divide(self, left: AbstractValue, right: AbstractValue) -> Tuple[AbstractValue, bool]:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom(), False
        if left.is_numeric() and right.is_numeric():
            result_iv, may_raise = _interval_floordiv(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL), may_raise
        return AbstractValue.top(), False

    def _modulo(self, left: AbstractValue, right: AbstractValue) -> Tuple[AbstractValue, bool]:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom(), False
        if left.is_numeric() and right.is_numeric():
            result_iv, may_raise = _interval_mod(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL), may_raise
        return AbstractValue.top(), False

    def _power(self, left: AbstractValue, right: AbstractValue) -> AbstractValue:
        if left.is_bottom() or right.is_bottom():
            return AbstractValue.bottom()
        if left.is_numeric() and right.is_numeric():
            result_iv = _interval_pow(left.interval, right.interval)
            tags: Set[TypeTag] = set()
            if TypeTag.FLOAT in left.type_tags or TypeTag.FLOAT in right.type_tags:
                tags.add(TypeTag.FLOAT)
            else:
                tags.add(TypeTag.INT)
            return AbstractValue(interval=result_iv, type_tags=tags, nullity=NullityTag.DEFINITELY_NOT_NULL)
        return AbstractValue.top()

    def apply_abs(self, val: AbstractValue) -> AbstractValue:
        if val.is_bottom():
            return AbstractValue.bottom()
        if val.is_numeric():
            return AbstractValue(
                interval=val.interval.abs_interval(),
                type_tags=val.type_tags.copy(),
                nullity=NullityTag.DEFINITELY_NOT_NULL,
            )
        return AbstractValue.top()

    def apply_min(self, vals: List[AbstractValue]) -> AbstractValue:
        if not vals:
            return AbstractValue.top()
        result = vals[0]
        for v in vals[1:]:
            if result.is_numeric() and v.is_numeric():
                result = AbstractValue(
                    interval=_interval_min(result.interval, v.interval),
                    type_tags=result.type_tags | v.type_tags,
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
            else:
                result = result.join(v)
        return result

    def apply_max(self, vals: List[AbstractValue]) -> AbstractValue:
        if not vals:
            return AbstractValue.top()
        result = vals[0]
        for v in vals[1:]:
            if result.is_numeric() and v.is_numeric():
                result = AbstractValue(
                    interval=_interval_max(result.interval, v.interval),
                    type_tags=result.type_tags | v.type_tags,
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                )
            else:
                result = result.join(v)
        return result


# ---------------------------------------------------------------------------
# TypeCastTransformer
# ---------------------------------------------------------------------------

class TypeCastTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, CallNode) and isinstance(node.func, VarExpr):
            fname = node.func.name
            if fname in ("int", "float", "str", "bool", "list", "tuple", "set", "dict"):
                return self._transform_cast(node, fname, state)
        return state

    def _transform_cast(self, node: CallNode, cast_func: str, state: AbstractState) -> AbstractState:
        args_vals = [_eval_expr(a, state) for a in node.args]
        new_state = state.deep_copy()
        if cast_func == "int":
            result, exc = self._cast_to_int(args_vals)
            if exc:
                new_state.exception_state = new_state.exception_state.join(exc)
        elif cast_func == "float":
            result, exc = self._cast_to_float(args_vals)
            if exc:
                new_state.exception_state = new_state.exception_state.join(exc)
        elif cast_func == "str":
            result = self._cast_to_str(args_vals)
        elif cast_func == "bool":
            result = self._cast_to_bool(args_vals)
        elif cast_func == "list":
            result = self._cast_to_list(args_vals)
        elif cast_func == "tuple":
            result = self._cast_to_tuple(args_vals)
        elif cast_func == "set":
            result = self._cast_to_set(args_vals)
        elif cast_func == "dict":
            result = self._cast_to_dict(args_vals)
        else:
            result = AbstractValue.top()
        if node.target is not None:
            new_state.variables[node.target] = result
        return new_state

    def _cast_to_int(self, args: List[AbstractValue]) -> Tuple[AbstractValue, Optional[ExceptionState]]:
        if not args:
            return AbstractValue.from_const(0), None
        arg = args[0]
        exc: Optional[ExceptionState] = None
        if TypeTag.INT in arg.type_tags:
            return AbstractValue(interval=Interval(lo=arg.interval.lo, hi=arg.interval.hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), None
        if TypeTag.FLOAT in arg.type_tags:
            lo = math.floor(arg.interval.lo) if arg.interval.lo is not None else None
            hi = math.floor(arg.interval.hi) if arg.interval.hi is not None else None
            return AbstractValue(interval=Interval(lo=lo, hi=hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), None
        if TypeTag.BOOL in arg.type_tags:
            return AbstractValue(interval=Interval(lo=0.0, hi=1.0), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), None
        if TypeTag.STR in arg.type_tags:
            exc = ExceptionState(active=True, exception_types={"ValueError"}, may_propagate=True)
            if arg.string_values is not None:
                int_vals: Set[float] = set()
                may_raise = False
                for s in arg.string_values:
                    try:
                        v = int(s)
                        int_vals.add(float(v))
                    except ValueError:
                        may_raise = True
                if int_vals:
                    lo = min(int_vals)
                    hi = max(int_vals)
                    result = AbstractValue(interval=Interval(lo=lo, hi=hi), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
                    if may_raise:
                        return result, exc
                    return result, None
            return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), exc
        if TypeTag.NONE in arg.type_tags:
            return AbstractValue.bottom(), ExceptionState(active=True, exception_types={"TypeError"}, may_propagate=True)
        return AbstractValue(type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL), exc

    def _cast_to_float(self, args: List[AbstractValue]) -> Tuple[AbstractValue, Optional[ExceptionState]]:
        if not args:
            return AbstractValue.from_const(0.0), None
        arg = args[0]
        if arg.is_numeric():
            return AbstractValue(interval=Interval(lo=arg.interval.lo, hi=arg.interval.hi), type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL), None
        if TypeTag.STR in arg.type_tags:
            exc = ExceptionState(active=True, exception_types={"ValueError"}, may_propagate=True)
            return AbstractValue(type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL), exc
        if TypeTag.NONE in arg.type_tags:
            return AbstractValue.bottom(), ExceptionState(active=True, exception_types={"TypeError"}, may_propagate=True)
        return AbstractValue(type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL), None

    def _cast_to_str(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue.from_const("")
        arg = args[0]
        if arg.is_string():
            return copy.deepcopy(arg)
        if arg.is_numeric() and arg.interval.is_const() and arg.interval.lo is not None:
            if TypeTag.INT in arg.type_tags:
                s = str(int(arg.interval.lo))
            else:
                s = str(arg.interval.lo)
            return AbstractValue.from_const(s)
        if arg.nullity == NullityTag.DEFINITELY_NULL:
            return AbstractValue.from_const("None")
        if TypeTag.BOOL in arg.type_tags and arg.interval.is_const():
            if arg.interval.lo == 1.0:
                return AbstractValue.from_const("True")
            elif arg.interval.lo == 0.0:
                return AbstractValue.from_const("False")
        return AbstractValue.from_type(TypeTag.STR)

    def _cast_to_bool(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue.from_const(False)
        arg = args[0]
        if _is_definitely_truthy(arg):
            return AbstractValue.from_const(True)
        if _is_definitely_falsy(arg):
            return AbstractValue.from_const(False)
        return AbstractValue.from_type(TypeTag.BOOL)

    def _cast_to_list(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))
        arg = args[0]
        if arg.is_container():
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=arg.container_length)
        if TypeTag.STR in arg.type_tags and arg.string_values is not None:
            lengths = {float(len(s)) for s in arg.string_values}
            lo = min(lengths)
            hi = max(lengths)
            return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval(lo=lo, hi=hi))
        return AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    def _cast_to_tuple(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))
        arg = args[0]
        if arg.is_container():
            return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=arg.container_length)
        return AbstractValue(type_tags={TypeTag.TUPLE}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    def _cast_to_set(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))
        arg = args[0]
        if arg.is_container():
            new_len = Interval(lo=0.0, hi=arg.container_length.hi)
            return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=new_len)
        return AbstractValue(type_tags={TypeTag.SET}, nullity=NullityTag.DEFINITELY_NOT_NULL)

    def _cast_to_dict(self, args: List[AbstractValue]) -> AbstractValue:
        if not args:
            return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=Interval.const(0.0))
        arg = args[0]
        if arg.is_container():
            return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL, container_length=arg.container_length)
        return AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL)


# ---------------------------------------------------------------------------
# AttributeTransformer
# ---------------------------------------------------------------------------

class AttributeTransformer(AbstractTransformer):
    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, AttributeAccessNode):
            return self._transform_get_attr(node, state)
        if isinstance(node, AttributeSetNode):
            return self._transform_set_attr(node, state)
        return state

    def _transform_get_attr(self, node: AttributeAccessNode, state: AbstractState) -> AbstractState:
        obj_val = _eval_expr(node.obj, state)
        new_state = state.deep_copy()
        if obj_val.nullity == NullityTag.DEFINITELY_NULL:
            new_state.exception_state = new_state.exception_state.join(
                ExceptionState(active=True, exception_types={"AttributeError"}, may_propagate=True)
            )
            if node.target is not None:
                new_state.variables[node.target] = AbstractValue.bottom()
            return new_state
        if node.attr in obj_val.attributes:
            attr_val = obj_val.attributes[node.attr]
        else:
            attr_val = self._infer_attribute_type(obj_val, node.attr)
        if node.target is not None:
            new_state.variables[node.target] = attr_val
        return new_state

    def _transform_set_attr(self, node: AttributeSetNode, state: AbstractState) -> AbstractState:
        obj_val = _eval_expr(node.obj, state)
        val = _eval_expr(node.value, state)
        new_state = state.deep_copy()
        if obj_val.nullity == NullityTag.DEFINITELY_NULL:
            new_state.exception_state = new_state.exception_state.join(
                ExceptionState(active=True, exception_types={"AttributeError"}, may_propagate=True)
            )
            return new_state
        if isinstance(node.obj, VarExpr):
            updated_obj = copy.deepcopy(obj_val)
            updated_obj.attributes[node.attr] = val
            new_state.variables[node.obj.name] = updated_obj
        return new_state

    def transform_del_attr(self, obj_expr: Expr, attr: str, state: AbstractState) -> AbstractState:
        obj_val = _eval_expr(obj_expr, state)
        new_state = state.deep_copy()
        if obj_val.nullity == NullityTag.DEFINITELY_NULL:
            new_state.exception_state = new_state.exception_state.join(
                ExceptionState(active=True, exception_types={"AttributeError"}, may_propagate=True)
            )
            return new_state
        if isinstance(obj_expr, VarExpr):
            updated_obj = copy.deepcopy(obj_val)
            updated_obj.attributes.pop(attr, None)
            new_state.variables[obj_expr.name] = updated_obj
        return new_state

    def _infer_attribute_type(self, obj_val: AbstractValue, attr: str) -> AbstractValue:
        type_attrs: Dict[TypeTag, Dict[str, AbstractValue]] = {
            TypeTag.STR: {
                "__len__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__contains__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__getitem__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__iter__": AbstractValue.from_type(TypeTag.CALLABLE),
            },
            TypeTag.LIST: {
                "__len__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__contains__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__getitem__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__setitem__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__iter__": AbstractValue.from_type(TypeTag.CALLABLE),
                "append": AbstractValue.from_type(TypeTag.CALLABLE),
                "extend": AbstractValue.from_type(TypeTag.CALLABLE),
                "insert": AbstractValue.from_type(TypeTag.CALLABLE),
                "pop": AbstractValue.from_type(TypeTag.CALLABLE),
                "remove": AbstractValue.from_type(TypeTag.CALLABLE),
                "sort": AbstractValue.from_type(TypeTag.CALLABLE),
                "reverse": AbstractValue.from_type(TypeTag.CALLABLE),
                "clear": AbstractValue.from_type(TypeTag.CALLABLE),
                "copy": AbstractValue.from_type(TypeTag.CALLABLE),
                "count": AbstractValue.from_type(TypeTag.CALLABLE),
                "index": AbstractValue.from_type(TypeTag.CALLABLE),
            },
            TypeTag.DICT: {
                "__len__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__contains__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__getitem__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__setitem__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__iter__": AbstractValue.from_type(TypeTag.CALLABLE),
                "keys": AbstractValue.from_type(TypeTag.CALLABLE),
                "values": AbstractValue.from_type(TypeTag.CALLABLE),
                "items": AbstractValue.from_type(TypeTag.CALLABLE),
                "get": AbstractValue.from_type(TypeTag.CALLABLE),
                "update": AbstractValue.from_type(TypeTag.CALLABLE),
                "pop": AbstractValue.from_type(TypeTag.CALLABLE),
                "popitem": AbstractValue.from_type(TypeTag.CALLABLE),
                "setdefault": AbstractValue.from_type(TypeTag.CALLABLE),
                "clear": AbstractValue.from_type(TypeTag.CALLABLE),
                "copy": AbstractValue.from_type(TypeTag.CALLABLE),
            },
            TypeTag.SET: {
                "__len__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__contains__": AbstractValue.from_type(TypeTag.CALLABLE),
                "__iter__": AbstractValue.from_type(TypeTag.CALLABLE),
                "add": AbstractValue.from_type(TypeTag.CALLABLE),
                "remove": AbstractValue.from_type(TypeTag.CALLABLE),
                "discard": AbstractValue.from_type(TypeTag.CALLABLE),
                "pop": AbstractValue.from_type(TypeTag.CALLABLE),
                "clear": AbstractValue.from_type(TypeTag.CALLABLE),
                "union": AbstractValue.from_type(TypeTag.CALLABLE),
                "intersection": AbstractValue.from_type(TypeTag.CALLABLE),
                "difference": AbstractValue.from_type(TypeTag.CALLABLE),
                "symmetric_difference": AbstractValue.from_type(TypeTag.CALLABLE),
                "copy": AbstractValue.from_type(TypeTag.CALLABLE),
            },
        }
        for tag in obj_val.type_tags:
            if tag in type_attrs and attr in type_attrs[tag]:
                return type_attrs[tag][attr]
        return AbstractValue.top()

    def transform_property_access(self, obj_val: AbstractValue, attr: str, state: AbstractState) -> AbstractValue:
        if attr in obj_val.attributes:
            attr_val = obj_val.attributes[attr]
            if TypeTag.CALLABLE in attr_val.type_tags and "getter" in attr_val.extra:
                return AbstractValue.top()
            return attr_val
        return self._infer_attribute_type(obj_val, attr)


# ---------------------------------------------------------------------------
# ImportTransformer
# ---------------------------------------------------------------------------

class ImportTransformer(AbstractTransformer):
    def __init__(self) -> None:
        self._module_summaries: Dict[str, Dict[str, AbstractValue]] = {}
        self._init_stdlib_summaries()

    def _init_stdlib_summaries(self) -> None:
        self._module_summaries["math"] = {
            "pi": AbstractValue.from_const(math.pi),
            "e": AbstractValue.from_const(math.e),
            "inf": AbstractValue(interval=Interval(lo=None, hi=None), type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "nan": AbstractValue(type_tags={TypeTag.FLOAT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "sqrt": AbstractValue.from_type(TypeTag.CALLABLE),
            "sin": AbstractValue.from_type(TypeTag.CALLABLE),
            "cos": AbstractValue.from_type(TypeTag.CALLABLE),
            "tan": AbstractValue.from_type(TypeTag.CALLABLE),
            "log": AbstractValue.from_type(TypeTag.CALLABLE),
            "exp": AbstractValue.from_type(TypeTag.CALLABLE),
            "floor": AbstractValue.from_type(TypeTag.CALLABLE),
            "ceil": AbstractValue.from_type(TypeTag.CALLABLE),
            "abs": AbstractValue.from_type(TypeTag.CALLABLE),
            "pow": AbstractValue.from_type(TypeTag.CALLABLE),
            "fabs": AbstractValue.from_type(TypeTag.CALLABLE),
            "factorial": AbstractValue.from_type(TypeTag.CALLABLE),
            "gcd": AbstractValue.from_type(TypeTag.CALLABLE),
            "isnan": AbstractValue.from_type(TypeTag.CALLABLE),
            "isinf": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["os"] = {
            "path": AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "getcwd": AbstractValue.from_type(TypeTag.CALLABLE),
            "listdir": AbstractValue.from_type(TypeTag.CALLABLE),
            "makedirs": AbstractValue.from_type(TypeTag.CALLABLE),
            "remove": AbstractValue.from_type(TypeTag.CALLABLE),
            "rename": AbstractValue.from_type(TypeTag.CALLABLE),
            "environ": AbstractValue(type_tags={TypeTag.DICT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "sep": AbstractValue.from_type(TypeTag.STR),
            "linesep": AbstractValue.from_type(TypeTag.STR),
        }
        self._module_summaries["sys"] = {
            "argv": AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "path": AbstractValue(type_tags={TypeTag.LIST}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "stdin": AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "stdout": AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "stderr": AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
            "exit": AbstractValue.from_type(TypeTag.CALLABLE),
            "version": AbstractValue.from_type(TypeTag.STR),
            "platform": AbstractValue.from_type(TypeTag.STR),
            "maxsize": AbstractValue(interval=Interval(lo=0.0, hi=None), type_tags={TypeTag.INT}, nullity=NullityTag.DEFINITELY_NOT_NULL),
        }
        self._module_summaries["json"] = {
            "dumps": AbstractValue.from_type(TypeTag.CALLABLE),
            "loads": AbstractValue.from_type(TypeTag.CALLABLE),
            "dump": AbstractValue.from_type(TypeTag.CALLABLE),
            "load": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["re"] = {
            "compile": AbstractValue.from_type(TypeTag.CALLABLE),
            "match": AbstractValue.from_type(TypeTag.CALLABLE),
            "search": AbstractValue.from_type(TypeTag.CALLABLE),
            "findall": AbstractValue.from_type(TypeTag.CALLABLE),
            "sub": AbstractValue.from_type(TypeTag.CALLABLE),
            "split": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["collections"] = {
            "defaultdict": AbstractValue.from_type(TypeTag.CALLABLE),
            "OrderedDict": AbstractValue.from_type(TypeTag.CALLABLE),
            "Counter": AbstractValue.from_type(TypeTag.CALLABLE),
            "deque": AbstractValue.from_type(TypeTag.CALLABLE),
            "namedtuple": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["typing"] = {
            "List": AbstractValue.from_type(TypeTag.CALLABLE),
            "Dict": AbstractValue.from_type(TypeTag.CALLABLE),
            "Set": AbstractValue.from_type(TypeTag.CALLABLE),
            "Tuple": AbstractValue.from_type(TypeTag.CALLABLE),
            "Optional": AbstractValue.from_type(TypeTag.CALLABLE),
            "Union": AbstractValue.from_type(TypeTag.CALLABLE),
            "Any": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["copy"] = {
            "copy": AbstractValue.from_type(TypeTag.CALLABLE),
            "deepcopy": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["itertools"] = {
            "chain": AbstractValue.from_type(TypeTag.CALLABLE),
            "product": AbstractValue.from_type(TypeTag.CALLABLE),
            "combinations": AbstractValue.from_type(TypeTag.CALLABLE),
            "permutations": AbstractValue.from_type(TypeTag.CALLABLE),
            "count": AbstractValue.from_type(TypeTag.CALLABLE),
            "cycle": AbstractValue.from_type(TypeTag.CALLABLE),
            "repeat": AbstractValue.from_type(TypeTag.CALLABLE),
            "islice": AbstractValue.from_type(TypeTag.CALLABLE),
            "starmap": AbstractValue.from_type(TypeTag.CALLABLE),
            "groupby": AbstractValue.from_type(TypeTag.CALLABLE),
        }
        self._module_summaries["functools"] = {
            "reduce": AbstractValue.from_type(TypeTag.CALLABLE),
            "partial": AbstractValue.from_type(TypeTag.CALLABLE),
            "wraps": AbstractValue.from_type(TypeTag.CALLABLE),
            "lru_cache": AbstractValue.from_type(TypeTag.CALLABLE),
            "cache": AbstractValue.from_type(TypeTag.CALLABLE),
        }

    def register_module(self, module: str, exports: Dict[str, AbstractValue]) -> None:
        self._module_summaries[module] = exports

    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, ImportNode):
            return self._transform_import(node, state)
        return state

    def _transform_import(self, node: ImportNode, state: AbstractState) -> AbstractState:
        new_state = state.deep_copy()
        if not node.is_from:
            for name, alias in node.names:
                bind_name = alias if alias is not None else name
                module_val = self._get_module_value(name)
                new_state.variables[bind_name] = module_val
        else:
            if node.is_star:
                module_exports = self._module_summaries.get(node.module, {})
                for export_name, export_val in module_exports.items():
                    if not export_name.startswith("_"):
                        new_state.variables[export_name] = copy.deepcopy(export_val)
            else:
                module_exports = self._module_summaries.get(node.module, {})
                for name, alias in node.names:
                    bind_name = alias if alias is not None else name
                    if name in module_exports:
                        new_state.variables[bind_name] = copy.deepcopy(module_exports[name])
                    else:
                        new_state.variables[bind_name] = AbstractValue.top()
        return new_state

    def _get_module_value(self, module_name: str) -> AbstractValue:
        if module_name in self._module_summaries:
            mod_val = AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL)
            for attr_name, attr_val in self._module_summaries[module_name].items():
                mod_val.attributes[attr_name] = copy.deepcopy(attr_val)
            return mod_val
        return AbstractValue(type_tags={TypeTag.OBJECT}, nullity=NullityTag.DEFINITELY_NOT_NULL)


# ---------------------------------------------------------------------------
# TransformerRegistry
# ---------------------------------------------------------------------------

class TransformerRegistry:
    def __init__(self) -> None:
        self._registry: Dict[type, AbstractTransformer] = {}
        self._fallback: Optional[AbstractTransformer] = None

    def register(self, node_type: type, transformer: AbstractTransformer) -> None:
        self._registry[node_type] = transformer

    def set_fallback(self, transformer: AbstractTransformer) -> None:
        self._fallback = transformer

    def get_transformer(self, node: IRNode) -> Optional[AbstractTransformer]:
        node_type = type(node)
        if node_type in self._registry:
            return self._registry[node_type]
        for registered_type, transformer in self._registry.items():
            if isinstance(node, registered_type):
                return transformer
        return self._fallback

    def dispatch(self, node: IRNode, state: AbstractState) -> AbstractState:
        transformer = self.get_transformer(node)
        if transformer is not None:
            return transformer.transform(node, state)
        return state

    def list_registered_types(self) -> List[type]:
        return list(self._registry.keys())

    def has_transformer(self, node_type: type) -> bool:
        return node_type in self._registry


# ---------------------------------------------------------------------------
# Composite transformer (for use by other transformers that need to
# transform blocks containing heterogeneous node types)
# ---------------------------------------------------------------------------

class _CompositeTransformer(AbstractTransformer):
    def __init__(self) -> None:
        self._registry = TransformerRegistry()
        self._assign = AssignTransformer()
        self._guard = GuardTransformer()
        self._phi = PhiTransformer()
        self._call = CallTransformer()
        self._ret = ReturnTransformer()
        self._exc = ExceptionTransformer()
        self._loop = LoopTransformer()
        self._comp = ComprehensionTransformer()
        self._container = ContainerTransformer()
        self._string = StringTransformer()
        self._numeric = NumericTransformer()
        self._typecast = TypeCastTransformer()
        self._attr = AttributeTransformer()
        self._import = ImportTransformer()

        self._registry.register(AssignNode, self._assign)
        self._registry.register(GuardNode, self._guard)
        self._registry.register(PhiNode, self._phi)
        self._registry.register(CallNode, self._call)
        self._registry.register(ReturnNode, self._ret)
        self._registry.register(RaiseNode, self._exc)
        self._registry.register(TryExceptNode, self._exc)
        self._registry.register(WhileNode, self._loop)
        self._registry.register(ForNode, self._loop)
        self._registry.register(ComprehensionNode, self._comp)
        self._registry.register(ContainerOpNode, self._container)
        self._registry.register(SubscriptNode, self._container)
        self._registry.register(StringOpNode, self._string)
        self._registry.register(BinaryOpNode, self._numeric)
        self._registry.register(UnaryOpNode, self._numeric)
        self._registry.register(CompareNode, self._numeric)
        self._registry.register(AttributeAccessNode, self._attr)
        self._registry.register(AttributeSetNode, self._attr)
        self._registry.register(ImportNode, self._import)
        self._registry.register(BoolOpNode, self._assign)

    def transform(self, node: IRNode, state: AbstractState) -> AbstractState:
        if state.is_bottom():
            return state
        if isinstance(node, BlockNode):
            return self.transform_block(node, state)
        if isinstance(node, FunctionNode):
            results = self.transform_function(node, state)
            return results.get("normal", state)
        if isinstance(node, BreakNode):
            new_state = state.deep_copy()
            new_state.break_states.append(state.deep_copy())
            return AbstractState.bottom()
        if isinstance(node, ContinueNode):
            new_state = state.deep_copy()
            new_state.continue_states.append(state.deep_copy())
            return AbstractState.bottom()
        return self._registry.dispatch(node, state)


def _make_composite_transformer() -> _CompositeTransformer:
    return _CompositeTransformer()


# ---------------------------------------------------------------------------
# AbstractInterpreterEngine
# ---------------------------------------------------------------------------

class AbstractInterpreterEngine:
    def __init__(
        self,
        max_iterations: int = 100,
        widening_delay: int = 3,
        enable_narrowing: bool = True,
    ) -> None:
        self._max_iterations = max_iterations
        self._widening_delay = widening_delay
        self._enable_narrowing = enable_narrowing
        self._transformer = _make_composite_transformer()
        self._state_at_point: Dict[int, AbstractState] = {}
        self._back_edges: Set[Tuple[int, int]] = set()
        self._loop_headers: Set[int] = set()

    @property
    def state_at_point(self) -> Dict[int, AbstractState]:
        return self._state_at_point

    def analyze_function(self, func: FunctionNode, entry_state: Optional[AbstractState] = None) -> Dict[str, AbstractState]:
        if entry_state is None:
            entry_state = AbstractState()
            for param in func.params:
                entry_state.variables[param] = AbstractValue.top()
        results = self._transformer.transform_function(func, entry_state)
        return results

    def analyze_block(self, block: BlockNode, entry_state: Optional[AbstractState] = None) -> AbstractState:
        if entry_state is None:
            entry_state = AbstractState()
        self._detect_back_edges(block)
        result = self._forward_analysis(block, entry_state)
        if self._enable_narrowing and self._loop_headers:
            result = self._narrowing_phase(block, result)
        return result

    def _forward_analysis(self, block: BlockNode, entry_state: AbstractState) -> AbstractState:
        worklist: List[Tuple[int, IRNode]] = []
        for i, stmt in enumerate(block.statements):
            worklist.append((i, stmt))
        states: Dict[int, AbstractState] = {}
        states[-1] = entry_state
        iteration_counts: Dict[int, int] = {}
        processed: Set[int] = set()
        while worklist:
            idx, node = worklist.pop(0)
            if idx in processed and idx not in self._loop_headers:
                continue
            count = iteration_counts.get(idx, 0)
            if count >= self._max_iterations:
                continue
            iteration_counts[idx] = count + 1
            pred_state = self._get_predecessor_state(idx, states, block)
            if pred_state.is_bottom():
                states[idx] = AbstractState.bottom()
                self._state_at_point[node.node_id] = states[idx]
                processed.add(idx)
                continue
            old_state = states.get(idx)
            if idx in self._loop_headers and old_state is not None:
                if count < self._widening_delay:
                    merged_state = old_state.join_with(pred_state)
                else:
                    merged_state = old_state.widen_with(pred_state)
                if old_state.equals(merged_state):
                    processed.add(idx)
                    continue
                pred_state = merged_state
            new_state = self._transformer.transform(node, pred_state)
            changed = old_state is None or not old_state.equals(new_state)
            states[idx] = new_state
            self._state_at_point[node.node_id] = new_state
            processed.add(idx)
            if changed:
                successors = self._get_successors(idx, block)
                for succ_idx in successors:
                    if succ_idx < len(block.statements):
                        worklist.append((succ_idx, block.statements[succ_idx]))
        final_idx = len(block.statements) - 1
        if final_idx in states:
            return states[final_idx]
        return entry_state

    def _narrowing_phase(self, block: BlockNode, ascending_result: AbstractState) -> AbstractState:
        states: Dict[int, AbstractState] = {}
        for i, stmt in enumerate(block.statements):
            if stmt.node_id in self._state_at_point:
                states[i] = self._state_at_point[stmt.node_id]
        changed = True
        iterations = 0
        while changed and iterations < self._max_iterations:
            changed = False
            iterations += 1
            for i, stmt in enumerate(block.statements):
                if i not in self._loop_headers:
                    continue
                old_state = states.get(i, AbstractState.bottom())
                pred_state = self._get_predecessor_state(i, states, block)
                new_state = self._transformer.transform(stmt, pred_state)
                narrowed = old_state.narrow_with(new_state)
                if not old_state.equals(narrowed):
                    states[i] = narrowed
                    self._state_at_point[stmt.node_id] = narrowed
                    changed = True
                    for j in range(i + 1, len(block.statements)):
                        succ_pred = self._get_predecessor_state(j, states, block)
                        succ_new = self._transformer.transform(block.statements[j], succ_pred)
                        if j in states:
                            succ_narrowed = states[j].narrow_with(succ_new)
                            states[j] = succ_narrowed
                        else:
                            states[j] = succ_new
                        self._state_at_point[block.statements[j].node_id] = states[j]
        final_idx = len(block.statements) - 1
        if final_idx in states:
            return states[final_idx]
        return ascending_result

    def _get_predecessor_state(self, idx: int, states: Dict[int, AbstractState], block: BlockNode) -> AbstractState:
        if idx == 0:
            return states.get(-1, AbstractState.bottom())
        pred_idx = idx - 1
        pred_state = states.get(pred_idx, AbstractState.bottom())
        for src, tgt in self._back_edges:
            if tgt == idx and src in states:
                pred_state = pred_state.join_with(states[src])
        return pred_state

    def _get_successors(self, idx: int, block: BlockNode) -> List[int]:
        successors = []
        if idx + 1 < len(block.statements):
            successors.append(idx + 1)
        for src, tgt in self._back_edges:
            if src == idx:
                successors.append(tgt)
        return successors

    def _detect_back_edges(self, block: BlockNode) -> None:
        self._back_edges.clear()
        self._loop_headers.clear()
        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, (WhileNode, ForNode)):
                self._loop_headers.add(i)
                last_body_idx = i
                if isinstance(stmt, WhileNode) and stmt.body:
                    last_body_idx = i
                elif isinstance(stmt, ForNode) and stmt.body:
                    last_body_idx = i
                self._back_edges.add((last_body_idx, i))

    def get_state_at(self, node_id: int) -> Optional[AbstractState]:
        return self._state_at_point.get(node_id)

    def is_fixed_point_reached(self, old_state: AbstractState, new_state: AbstractState) -> bool:
        return old_state.equals(new_state)

    def reset(self) -> None:
        self._state_at_point.clear()
        self._back_edges.clear()
        self._loop_headers.clear()

    def analyze_with_initial_state(self, nodes: List[IRNode], initial_state: AbstractState) -> AbstractState:
        block = BlockNode(statements=nodes)
        return self.analyze_block(block, initial_state)

    def get_all_states(self) -> Dict[int, AbstractState]:
        return dict(self._state_at_point)

    def get_return_type(self, func: FunctionNode, entry_state: Optional[AbstractState] = None) -> AbstractValue:
        results = self.analyze_function(func, entry_state)
        normal = results.get("normal", AbstractState())
        ret_transformer = ReturnTransformer()
        return ret_transformer.collect_return_type(normal)

    def analyze_with_cegar_refinement(
        self,
        func: FunctionNode,
        entry_state: AbstractState,
        property_check: Callable[[AbstractState], bool],
        max_refinements: int = 10,
    ) -> Tuple[bool, AbstractState]:
        """
        Counterexample-guided analysis: analyze, check property, refine if needed.
        Returns (property_holds, final_state).
        """
        current_state = entry_state
        for refinement in range(max_refinements):
            results = self.analyze_function(func, current_state)
            final_state = results.get("normal", AbstractState())
            if property_check(final_state):
                return True, final_state
            refined = self._refine_state(current_state, final_state, func)
            if refined.equals(current_state):
                return False, final_state
            current_state = refined
        results = self.analyze_function(func, current_state)
        final_state = results.get("normal", AbstractState())
        return property_check(final_state), final_state

    def _refine_state(self, entry: AbstractState, result: AbstractState, func: FunctionNode) -> AbstractState:
        refined = entry.deep_copy()
        for param in func.params:
            if param in result.variables:
                result_val = result.variables[param]
                entry_val = entry.get_value(param)
                narrowed = entry_val.narrow(result_val)
                if not narrowed.is_bottom():
                    refined.variables[param] = narrowed
        return refined

    def analyze_interprocedural(
        self,
        functions: Dict[str, FunctionNode],
        call_graph: Dict[str, List[str]],
        entry_states: Dict[str, AbstractState],
    ) -> Dict[str, Dict[str, AbstractState]]:
        """Analyze multiple functions with interprocedural summaries."""
        summaries: Dict[str, Dict[str, AbstractState]] = {}
        call_transformer = self._transformer._call
        topo_order = self._topological_sort(functions, call_graph)
        for fname in topo_order:
            func = functions[fname]
            entry = entry_states.get(fname, AbstractState())
            for dep_name in call_graph.get(fname, []):
                if dep_name in summaries:
                    dep_summary = summaries[dep_name]
                    normal = dep_summary.get("normal", AbstractState())
                    ret_type = ReturnTransformer().collect_return_type(normal)
                    def make_summary(ret: AbstractValue) -> Callable[[List[AbstractValue], AbstractState], AbstractValue]:
                        def summary_fn(args: List[AbstractValue], state: AbstractState) -> AbstractValue:
                            return ret
                        return summary_fn
                    call_transformer.register_summary(dep_name, make_summary(ret_type))
            results = self.analyze_function(func, entry)
            summaries[fname] = results
        return summaries

    def _topological_sort(self, functions: Dict[str, FunctionNode], call_graph: Dict[str, List[str]]) -> List[str]:
        in_degree: Dict[str, int] = {f: 0 for f in functions}
        for caller, callees in call_graph.items():
            for callee in callees:
                if callee in in_degree:
                    in_degree[callee] = in_degree.get(callee, 0) + 1
        queue = [f for f, d in in_degree.items() if d == 0]
        result: List[str] = []
        visited: Set[str] = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            for callee in call_graph.get(node, []):
                if callee in in_degree:
                    in_degree[callee] -= 1
                    if in_degree[callee] <= 0 and callee not in visited:
                        queue.append(callee)
        for f in functions:
            if f not in visited:
                result.append(f)
        return result
