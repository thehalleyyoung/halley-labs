"""
Interval domain [lo, hi] with extended integers (±∞).

Provides precise numeric tracking via interval arithmetic, comparison-based
refinement, widening/narrowing, and union-of-intervals for additional precision.
"""

from __future__ import annotations

import math
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    AbstractDomain,
    AbstractState,
    AbstractTransformer,
    AbstractValue,
    IRNode,
    WideningStrategy,
)


# ===================================================================
# Bound – extended integer (finite value or ±∞)
# ===================================================================


class BoundKind(Enum):
    NEG_INF = auto()
    FINITE = auto()
    POS_INF = auto()


@dataclass(frozen=True, order=False)
class Bound:
    """Extended integer: finite value or ±∞."""

    kind: BoundKind
    value: int = 0

    # -- constructors --------------------------------------------------------

    @classmethod
    def finite(cls, n: int) -> "Bound":
        return cls(kind=BoundKind.FINITE, value=n)

    @classmethod
    def pos_inf(cls) -> "Bound":
        return cls(kind=BoundKind.POS_INF)

    @classmethod
    def neg_inf(cls) -> "Bound":
        return cls(kind=BoundKind.NEG_INF)

    # -- predicates ----------------------------------------------------------

    @property
    def is_finite(self) -> bool:
        return self.kind == BoundKind.FINITE

    @property
    def is_pos_inf(self) -> bool:
        return self.kind == BoundKind.POS_INF

    @property
    def is_neg_inf(self) -> bool:
        return self.kind == BoundKind.NEG_INF

    @property
    def is_inf(self) -> bool:
        return self.kind != BoundKind.FINITE

    # -- comparison ----------------------------------------------------------

    def _ordering_key(self) -> Tuple[int, int]:
        if self.kind == BoundKind.NEG_INF:
            return (0, 0)
        if self.kind == BoundKind.POS_INF:
            return (2, 0)
        return (1, self.value)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self._ordering_key() < other._ordering_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self._ordering_key() <= other._ordering_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self._ordering_key() > other._ordering_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self._ordering_key() >= other._ordering_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self.kind == other.kind and (
            not self.is_finite or self.value == other.value
        )

    def __hash__(self) -> int:
        if self.is_finite:
            return hash(("finite", self.value))
        return hash(self.kind)

    # -- arithmetic ----------------------------------------------------------

    def __neg__(self) -> "Bound":
        if self.is_pos_inf:
            return Bound.neg_inf()
        if self.is_neg_inf:
            return Bound.pos_inf()
        return Bound.finite(-self.value)

    def __add__(self, other: "Bound") -> "Bound":
        if self.is_pos_inf:
            if other.is_neg_inf:
                raise ValueError("-∞ + +∞ is undefined")
            return Bound.pos_inf()
        if self.is_neg_inf:
            if other.is_pos_inf:
                raise ValueError("-∞ + +∞ is undefined")
            return Bound.neg_inf()
        if other.is_pos_inf:
            return Bound.pos_inf()
        if other.is_neg_inf:
            return Bound.neg_inf()
        return Bound.finite(self.value + other.value)

    def __sub__(self, other: "Bound") -> "Bound":
        return self + (-other)

    def __mul__(self, other: "Bound") -> "Bound":
        if self.is_finite and self.value == 0:
            return Bound.finite(0)
        if other.is_finite and other.value == 0:
            return Bound.finite(0)
        if self.is_finite and other.is_finite:
            return Bound.finite(self.value * other.value)
        # Infinity cases
        self_sign = self._sign()
        other_sign = other._sign()
        if self_sign is None or other_sign is None:
            return Bound.finite(0)
        if self_sign * other_sign > 0:
            return Bound.pos_inf()
        return Bound.neg_inf()

    def _sign(self) -> Optional[int]:
        if self.is_pos_inf:
            return 1
        if self.is_neg_inf:
            return -1
        if self.is_finite:
            if self.value > 0:
                return 1
            if self.value < 0:
                return -1
            return None
        return None

    def __repr__(self) -> str:
        if self.is_pos_inf:
            return "+∞"
        if self.is_neg_inf:
            return "-∞"
        return str(self.value)


def _min_bound(a: Bound, b: Bound) -> Bound:
    return a if a <= b else b


def _max_bound(a: Bound, b: Bound) -> Bound:
    return a if a >= b else b


# ===================================================================
# Interval – [lo, hi]
# ===================================================================


@dataclass(frozen=True)
class Interval:
    """Closed interval [lo, hi] with extended integers."""

    lo: Bound
    hi: Bound

    # -- constructors --------------------------------------------------------

    @classmethod
    def top(cls) -> "Interval":
        return cls(lo=Bound.neg_inf(), hi=Bound.pos_inf())

    @classmethod
    def bottom(cls) -> "Interval":
        return cls(lo=Bound.finite(1), hi=Bound.finite(0))

    @classmethod
    def singleton(cls, n: int) -> "Interval":
        b = Bound.finite(n)
        return cls(lo=b, hi=b)

    @classmethod
    def from_bounds(cls, lo: int, hi: int) -> "Interval":
        return cls(lo=Bound.finite(lo), hi=Bound.finite(hi))

    @classmethod
    def ge(cls, lo: int) -> "Interval":
        return cls(lo=Bound.finite(lo), hi=Bound.pos_inf())

    @classmethod
    def le(cls, hi: int) -> "Interval":
        return cls(lo=Bound.neg_inf(), hi=Bound.finite(hi))

    @classmethod
    def non_negative(cls) -> "Interval":
        return cls(lo=Bound.finite(0), hi=Bound.pos_inf())

    @classmethod
    def positive(cls) -> "Interval":
        return cls(lo=Bound.finite(1), hi=Bound.pos_inf())

    # -- predicates ----------------------------------------------------------

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo.is_neg_inf and self.hi.is_pos_inf

    @property
    def is_singleton(self) -> bool:
        return self.lo == self.hi and self.lo.is_finite

    @property
    def is_non_negative(self) -> bool:
        return not self.is_bottom and self.lo >= Bound.finite(0)

    @property
    def is_positive(self) -> bool:
        return not self.is_bottom and self.lo >= Bound.finite(1)

    @property
    def is_non_positive(self) -> bool:
        return not self.is_bottom and self.hi <= Bound.finite(0)

    @property
    def is_negative(self) -> bool:
        return not self.is_bottom and self.hi <= Bound.finite(-1)

    def contains(self, n: int) -> bool:
        if self.is_bottom:
            return False
        b = Bound.finite(n)
        return self.lo <= b and b <= self.hi

    def contains_zero(self) -> bool:
        return self.contains(0)

    def singleton_value(self) -> Optional[int]:
        if self.is_singleton:
            return self.lo.value
        return None

    # -- lattice operations --------------------------------------------------

    def join(self, other: "Interval") -> "Interval":
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return Interval(lo=_min_bound(self.lo, other.lo), hi=_max_bound(self.hi, other.hi))

    def meet(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        lo = _max_bound(self.lo, other.lo)
        hi = _min_bound(self.hi, other.hi)
        if lo > hi:
            return Interval.bottom()
        return Interval(lo=lo, hi=hi)

    def leq(self, other: "Interval") -> bool:
        if self.is_bottom:
            return True
        if other.is_bottom:
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(lo=self.lo + other.lo, hi=self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(lo=self.lo - other.hi, hi=self.hi - other.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        lo = candidates[0]
        hi = candidates[0]
        for c in candidates[1:]:
            lo = _min_bound(lo, c)
            hi = _max_bound(hi, c)
        return Interval(lo=lo, hi=hi)

    def __neg__(self) -> "Interval":
        if self.is_bottom:
            return Interval.bottom()
        return Interval(lo=-self.hi, hi=-self.lo)

    def abs(self) -> "Interval":
        if self.is_bottom:
            return Interval.bottom()
        if self.is_non_negative:
            return self
        if self.is_non_positive:
            return -self
        neg_part = -Interval(lo=self.lo, hi=Bound.finite(0))
        pos_part = Interval(lo=Bound.finite(0), hi=self.hi)
        return neg_part.join(pos_part)

    def floordiv(self, other: "Interval") -> "Interval":
        """Integer division (//), excluding division by zero."""
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if other.contains_zero():
            if other.is_singleton:
                return Interval.bottom()
            parts: List[Interval] = []
            neg = other.meet(Interval(lo=Bound.neg_inf(), hi=Bound.finite(-1)))
            pos = other.meet(Interval(lo=Bound.finite(1), hi=Bound.pos_inf()))
            if not neg.is_bottom:
                parts.append(self.floordiv(neg))
            if not pos.is_bottom:
                parts.append(self.floordiv(pos))
            if not parts:
                return Interval.bottom()
            result = parts[0]
            for p in parts[1:]:
                result = result.join(p)
            return result

        if self.lo.is_finite and self.hi.is_finite and other.lo.is_finite and other.hi.is_finite:
            candidates = [
                self.lo.value // other.lo.value,
                self.lo.value // other.hi.value,
                self.hi.value // other.lo.value,
                self.hi.value // other.hi.value,
            ]
            return Interval.from_bounds(min(candidates), max(candidates))

        return Interval.top()

    def mod(self, other: "Interval") -> "Interval":
        """Modulo (%), excluding mod by zero."""
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if other.is_singleton and other.lo.is_finite and other.lo.value != 0:
            m = abs(other.lo.value)
            if self.is_non_negative:
                return Interval.from_bounds(0, m - 1)
            return Interval.from_bounds(-(m - 1), m - 1)
        if other.is_non_negative and not other.contains_zero():
            if other.hi.is_finite:
                return Interval.from_bounds(0, other.hi.value - 1)
            return Interval.non_negative()
        return Interval.top()

    def power(self, other: "Interval") -> "Interval":
        """Exponentiation (**)."""
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if other.is_singleton and other.lo.is_finite:
            exp = other.lo.value
            if exp == 0:
                return Interval.singleton(1)
            if exp == 1:
                return self
            if exp == 2:
                return self * self
            if self.is_singleton and self.lo.is_finite:
                try:
                    return Interval.singleton(self.lo.value ** exp)
                except (OverflowError, ValueError):
                    return Interval.top()
        return Interval.top()

    def min(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(
            lo=_min_bound(self.lo, other.lo),
            hi=_min_bound(self.hi, other.hi),
        )

    def max(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(
            lo=_max_bound(self.lo, other.lo),
            hi=_max_bound(self.hi, other.hi),
        )

    # -- bitwise (non-negative operands) -------------------------------------

    def bitwise_and(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not (self.is_non_negative and other.is_non_negative):
            return Interval.top()
        hi = _min_bound(self.hi, other.hi)
        return Interval(lo=Bound.finite(0), hi=hi)

    def bitwise_or(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not (self.is_non_negative and other.is_non_negative):
            return Interval.top()
        if self.hi.is_finite and other.hi.is_finite:
            max_val = self.hi.value | other.hi.value
            next_pow2 = 1
            while next_pow2 <= max_val:
                next_pow2 <<= 1
            return Interval.from_bounds(
                max(self.lo.value, other.lo.value) if self.lo.is_finite and other.lo.is_finite else 0,
                next_pow2 - 1,
            )
        return Interval.non_negative()

    def bitwise_xor(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not (self.is_non_negative and other.is_non_negative):
            return Interval.top()
        if self.hi.is_finite and other.hi.is_finite:
            max_val = max(self.hi.value, other.hi.value)
            next_pow2 = 1
            while next_pow2 <= max_val:
                next_pow2 <<= 1
            return Interval.from_bounds(0, next_pow2 - 1)
        return Interval.non_negative()

    def left_shift(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not other.is_non_negative:
            return Interval.top()
        if self.lo.is_finite and self.hi.is_finite and other.lo.is_finite and other.hi.is_finite:
            if other.hi.value > 63:
                return Interval.top()
            candidates = [
                self.lo.value << other.lo.value,
                self.lo.value << other.hi.value,
                self.hi.value << other.lo.value,
                self.hi.value << other.hi.value,
            ]
            return Interval.from_bounds(min(candidates), max(candidates))
        return Interval.top()

    def right_shift(self, other: "Interval") -> "Interval":
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not other.is_non_negative:
            return Interval.top()
        if self.lo.is_finite and self.hi.is_finite and other.lo.is_finite and other.hi.is_finite:
            candidates = [
                self.lo.value >> other.lo.value,
                self.lo.value >> other.hi.value,
                self.hi.value >> other.lo.value,
                self.hi.value >> other.hi.value,
            ]
            return Interval.from_bounds(min(candidates), max(candidates))
        if self.is_non_negative:
            return Interval(lo=Bound.finite(0), hi=self.hi)
        return Interval.top()

    # -- comparison predicates -----------------------------------------------

    def definitely_less_than(self, other: "Interval") -> Optional[bool]:
        if self.is_bottom or other.is_bottom:
            return None
        if self.hi < other.lo:
            return True
        if self.lo >= other.hi:
            return False
        return None

    def definitely_less_equal(self, other: "Interval") -> Optional[bool]:
        if self.is_bottom or other.is_bottom:
            return None
        if self.hi <= other.lo:
            return True
        if self.lo > other.hi:
            return False
        return None

    def definitely_equal(self, other: "Interval") -> Optional[bool]:
        if self.is_bottom or other.is_bottom:
            return None
        m = self.meet(other)
        if m.is_bottom:
            return False
        if self.is_singleton and other.is_singleton and self.lo == other.lo:
            return True
        return None

    def definitely_not_equal(self, other: "Interval") -> Optional[bool]:
        eq = self.definitely_equal(other)
        if eq is None:
            return None
        return not eq

    # -- special constructors for dynamic-lang features ----------------------

    @classmethod
    def for_len(cls) -> "Interval":
        """len() result is always >= 0."""
        return cls.non_negative()

    @classmethod
    def for_range(cls, n: "Interval") -> "Interval":
        """range(n) produces indices [0, n-1]."""
        if n.is_bottom:
            return cls.bottom()
        if n.hi.is_finite:
            hi = Bound.finite(max(0, n.hi.value - 1))
        else:
            hi = Bound.pos_inf()
        return cls(lo=Bound.finite(0), hi=hi)

    @classmethod
    def for_string_len(cls) -> "Interval":
        """String length: [0, +∞)."""
        return cls.non_negative()

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_bottom:
            return "⊥"
        return f"[{self.lo}, {self.hi}]"


# ===================================================================
# IntervalValue – abstract value wrapping an Interval
# ===================================================================


class IntervalValue(AbstractValue):
    """Abstract value wrapping an Interval."""

    def __init__(self, interval: Interval):
        self.interval = interval

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalValue):
            return NotImplemented
        return self.interval == other.interval

    def __hash__(self) -> int:
        return hash(self.interval)

    def __repr__(self) -> str:
        return f"IntervalValue({self.interval})"

    def is_bottom(self) -> bool:
        return self.interval.is_bottom

    def is_top(self) -> bool:
        return self.interval.is_top


# ===================================================================
# IntervalDomain – AbstractDomain implementation
# ===================================================================


class IntervalDomain(AbstractDomain[IntervalValue]):
    """Abstract domain of integer intervals with widening/narrowing."""

    def top(self) -> IntervalValue:
        return IntervalValue(Interval.top())

    def bottom(self) -> IntervalValue:
        return IntervalValue(Interval.bottom())

    def join(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(a.interval.join(b.interval))

    def meet(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(a.interval.meet(b.interval))

    def leq(self, a: IntervalValue, b: IntervalValue) -> bool:
        return a.interval.leq(b.interval)

    def widen(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(self._widen_interval(a.interval, b.interval))

    def narrow(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(self._narrow_interval(a.interval, b.interval))

    def abstract(self, concrete: Any) -> IntervalValue:
        if isinstance(concrete, int):
            return IntervalValue(Interval.singleton(concrete))
        if isinstance(concrete, (set, frozenset)):
            ints = [x for x in concrete if isinstance(x, int)]
            if not ints:
                return self.bottom()
            return IntervalValue(Interval.from_bounds(min(ints), max(ints)))
        return self.top()

    def concretize(self, abstract_val: IntervalValue) -> Any:
        iv = abstract_val.interval
        if iv.is_bottom:
            return set()
        if iv.lo.is_finite and iv.hi.is_finite:
            size = iv.hi.value - iv.lo.value + 1
            if size <= 1000:
                return set(range(iv.lo.value, iv.hi.value + 1))
        return f"[{iv.lo}, {iv.hi}]"

    # -- widening / narrowing internals --------------------------------------

    @staticmethod
    def _widen_interval(a: Interval, b: Interval) -> Interval:
        if a.is_bottom:
            return b
        if b.is_bottom:
            return a
        lo = a.lo if b.lo >= a.lo else Bound.neg_inf()
        hi = a.hi if b.hi <= a.hi else Bound.pos_inf()
        return Interval(lo=lo, hi=hi)

    @staticmethod
    def _narrow_interval(a: Interval, b: Interval) -> Interval:
        if a.is_bottom:
            return a
        if b.is_bottom:
            return b
        lo = b.lo if a.lo.is_neg_inf else a.lo
        hi = b.hi if a.hi.is_pos_inf else a.hi
        return Interval(lo=lo, hi=hi)


# ===================================================================
# IntervalWidening – standard + threshold widening
# ===================================================================


class IntervalWidening(WideningStrategy[IntervalValue]):
    """Standard interval widening with optional thresholds."""

    def __init__(
        self,
        delay: int = 0,
        thresholds: Optional[List[int]] = None,
    ):
        self.delay = delay
        self.thresholds = sorted(thresholds) if thresholds else []

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self,
        domain: AbstractDomain[IntervalValue],
        old: IntervalValue,
        new: IntervalValue,
        iteration: int,
    ) -> IntervalValue:
        if iteration < self.delay:
            return domain.join(old, new)
        if self.thresholds:
            return IntervalValue(
                self._widen_with_thresholds(old.interval, new.interval)
            )
        return IntervalValue(IntervalDomain._widen_interval(old.interval, new.interval))

    def _widen_with_thresholds(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom:
            return b
        if b.is_bottom:
            return a
        # Lower bound
        if b.lo < a.lo:
            lo = Bound.neg_inf()
            for t in self.thresholds:
                bt = Bound.finite(t)
                if bt <= b.lo:
                    lo = bt
                    break
        else:
            lo = a.lo
        # Upper bound
        if b.hi > a.hi:
            hi = Bound.pos_inf()
            for t in reversed(self.thresholds):
                bt = Bound.finite(t)
                if bt >= b.hi:
                    hi = bt
                    break
        else:
            hi = a.hi
        return Interval(lo=lo, hi=hi)


# ===================================================================
# IntervalNarrowing – standard narrowing
# ===================================================================


class IntervalNarrowing:
    """Standard interval narrowing."""

    @staticmethod
    def narrow(a: Interval, b: Interval) -> Interval:
        if a.is_bottom:
            return a
        if b.is_bottom:
            return b
        lo = b.lo if a.lo.is_neg_inf else a.lo
        hi = b.hi if a.hi.is_pos_inf else a.hi
        if lo > hi:
            return Interval.bottom()
        return Interval(lo=lo, hi=hi)


# ===================================================================
# IntervalMeet – intersection of intervals
# ===================================================================


class IntervalMeet:
    """Intersection of intervals."""

    @staticmethod
    def meet(a: Interval, b: Interval) -> Interval:
        return a.meet(b)

    @staticmethod
    def meet_values(a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(a.interval.meet(b.interval))


# ===================================================================
# IntervalRefiner – refine intervals from guards
# ===================================================================


class IntervalRefiner:
    """Refine intervals from comparison guards.

    E.g. ``x < 5`` on the true branch refines x to [-∞, 4].
    """

    @staticmethod
    def refine_lt(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        """Refine for lhs < rhs."""
        if on_true:
            new_lhs = lhs.meet(Interval(lo=Bound.neg_inf(), hi=rhs.hi - Bound.finite(1)))
            new_rhs = rhs.meet(Interval(lo=lhs.lo + Bound.finite(1), hi=Bound.pos_inf()))
            return new_lhs, new_rhs
        return IntervalRefiner.refine_ge(lhs, rhs, on_true=True)

    @staticmethod
    def refine_le(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        if on_true:
            new_lhs = lhs.meet(Interval(lo=Bound.neg_inf(), hi=rhs.hi))
            new_rhs = rhs.meet(Interval(lo=lhs.lo, hi=Bound.pos_inf()))
            return new_lhs, new_rhs
        return IntervalRefiner.refine_gt(lhs, rhs, on_true=True)

    @staticmethod
    def refine_gt(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        if on_true:
            r, l = IntervalRefiner.refine_lt(rhs, lhs, on_true=True)
            return l, r
        return IntervalRefiner.refine_le(lhs, rhs, on_true=True)

    @staticmethod
    def refine_ge(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        if on_true:
            r, l = IntervalRefiner.refine_le(rhs, lhs, on_true=True)
            return l, r
        return IntervalRefiner.refine_lt(lhs, rhs, on_true=True)

    @staticmethod
    def refine_eq(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        if on_true:
            m = lhs.meet(rhs)
            return m, m
        # on false: can only refine if one is a singleton
        if lhs.is_singleton and lhs.lo.is_finite:
            v = lhs.lo.value
            if rhs.lo.is_finite and rhs.lo.value == v and rhs.hi.is_finite and rhs.hi.value == v:
                return Interval.bottom(), Interval.bottom()
        return lhs, rhs

    @staticmethod
    def refine_ne(
        lhs: Interval, rhs: Interval, *, on_true: bool = True
    ) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_eq(lhs, rhs, on_true=not on_true)


# ===================================================================
# BackwardRefiner – backward refinement from comparison results
# ===================================================================


class BackwardRefiner:
    """Backward refinement from known comparison results.

    Given that a comparison ``lhs op rhs`` is known to be True or False,
    refine the intervals of both operands.
    """

    _REFINERS = {
        "<": IntervalRefiner.refine_lt,
        "<=": IntervalRefiner.refine_le,
        ">": IntervalRefiner.refine_gt,
        ">=": IntervalRefiner.refine_ge,
        "==": IntervalRefiner.refine_eq,
        "!=": IntervalRefiner.refine_ne,
    }

    @classmethod
    def refine(
        cls,
        op: str,
        lhs: Interval,
        rhs: Interval,
        result: bool,
    ) -> Tuple[Interval, Interval]:
        refiner = cls._REFINERS.get(op)
        if refiner is None:
            return lhs, rhs
        return refiner(lhs, rhs, on_true=result)


# ===================================================================
# ConstantThresholdCollector
# ===================================================================


class ConstantThresholdCollector:
    """Collect integer constants from program IR for threshold widening."""

    def __init__(self) -> None:
        self.thresholds: Set[int] = set()

    def collect_from_node(self, node: IRNode) -> None:
        self._collect_from_value(node.expr)
        self._collect_from_value(node.condition)
        for arg in node.args:
            self._collect_from_value(arg)

    def _collect_from_value(self, val: Any) -> None:
        if isinstance(val, int):
            self.thresholds.add(val)
            self.thresholds.add(val - 1)
            self.thresholds.add(val + 1)
        elif isinstance(val, (list, tuple)):
            for v in val:
                self._collect_from_value(v)

    def collect_from_nodes(self, nodes: Iterable[IRNode]) -> None:
        for node in nodes:
            self.collect_from_node(node)

    def get_sorted_thresholds(self) -> List[int]:
        return sorted(self.thresholds)


# ===================================================================
# IntervalUnion – union of disjoint intervals
# ===================================================================


class IntervalUnion:
    """Union of disjoint intervals for increased precision.

    Maintains a sorted list of non-overlapping, non-adjacent intervals.
    """

    def __init__(self, intervals: Optional[List[Interval]] = None):
        if intervals is None:
            self._intervals: List[Interval] = []
        else:
            self._intervals = self._normalize(intervals)

    @classmethod
    def from_interval(cls, iv: Interval) -> "IntervalUnion":
        if iv.is_bottom:
            return cls()
        return cls([iv])

    @classmethod
    def top(cls) -> "IntervalUnion":
        return cls([Interval.top()])

    @classmethod
    def bottom(cls) -> "IntervalUnion":
        return cls()

    @property
    def is_bottom(self) -> bool:
        return len(self._intervals) == 0

    @property
    def is_top(self) -> bool:
        return len(self._intervals) == 1 and self._intervals[0].is_top

    @property
    def intervals(self) -> List[Interval]:
        return list(self._intervals)

    def to_interval(self) -> Interval:
        """Over-approximate as a single interval."""
        if not self._intervals:
            return Interval.bottom()
        lo = self._intervals[0].lo
        hi = self._intervals[-1].hi
        return Interval(lo=lo, hi=hi)

    @staticmethod
    def _normalize(intervals: List[Interval]) -> List[Interval]:
        non_bottom = [iv for iv in intervals if not iv.is_bottom]
        if not non_bottom:
            return []
        sorted_ivs = sorted(non_bottom, key=lambda iv: iv.lo._ordering_key())
        merged: List[Interval] = [sorted_ivs[0]]
        for iv in sorted_ivs[1:]:
            last = merged[-1]
            if iv.lo <= last.hi or (
                last.hi.is_finite
                and iv.lo.is_finite
                and iv.lo.value <= last.hi.value + 1
            ):
                merged[-1] = Interval(lo=last.lo, hi=_max_bound(last.hi, iv.hi))
            else:
                merged.append(iv)
        return merged

    def join(self, other: "IntervalUnion") -> "IntervalUnion":
        return IntervalUnion(self._intervals + other._intervals)

    def meet(self, other: "IntervalUnion") -> "IntervalUnion":
        result: List[Interval] = []
        for a in self._intervals:
            for b in other._intervals:
                m = a.meet(b)
                if not m.is_bottom:
                    result.append(m)
        return IntervalUnion(result)

    def contains(self, n: int) -> bool:
        return any(iv.contains(n) for iv in self._intervals)

    def __repr__(self) -> str:
        if self.is_bottom:
            return "IntervalUnion(⊥)"
        parts = " ∪ ".join(repr(iv) for iv in self._intervals)
        return f"IntervalUnion({parts})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalUnion):
            return NotImplemented
        return self._intervals == other._intervals

    def __hash__(self) -> int:
        return hash(tuple(self._intervals))


# ===================================================================
# WrappedInterval – modular arithmetic intervals
# ===================================================================


class WrappedInterval:
    """Interval under modular arithmetic for fixed-width integers.

    Values wrap around at 2^width. Represents a set of values in [0, 2^w - 1].
    """

    def __init__(self, lo: int, hi: int, width: int = 64):
        self.width = width
        self.modulus = 1 << width
        self.lo = lo % self.modulus
        self.hi = hi % self.modulus
        self._is_top = False

    @classmethod
    def top(cls, width: int = 64) -> "WrappedInterval":
        w = cls(0, (1 << width) - 1, width)
        w._is_top = True
        return w

    @classmethod
    def bottom(cls, width: int = 64) -> "WrappedInterval":
        return cls(1, 0, width)

    @classmethod
    def singleton(cls, n: int, width: int = 64) -> "WrappedInterval":
        m = (1 << width)
        return cls(n % m, n % m, width)

    @property
    def is_bottom(self) -> bool:
        return self.lo == 1 and self.hi == 0 and not self._is_top

    @property
    def is_top(self) -> bool:
        return self._is_top or (self.lo == 0 and self.hi == self.modulus - 1)

    @property
    def wraps_around(self) -> bool:
        return self.lo > self.hi and not self.is_bottom

    def contains(self, n: int) -> bool:
        if self.is_bottom:
            return False
        if self.is_top:
            return True
        n = n % self.modulus
        if self.wraps_around:
            return n >= self.lo or n <= self.hi
        return self.lo <= n <= self.hi

    def size(self) -> int:
        if self.is_bottom:
            return 0
        if self.is_top:
            return self.modulus
        if self.wraps_around:
            return self.modulus - self.lo + self.hi + 1
        return self.hi - self.lo + 1

    def to_interval(self) -> Interval:
        if self.is_bottom:
            return Interval.bottom()
        if self.is_top or self.wraps_around:
            return Interval.from_bounds(0, self.modulus - 1)
        return Interval.from_bounds(self.lo, self.hi)

    def join(self, other: "WrappedInterval") -> "WrappedInterval":
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self.is_top or other.is_top:
            return WrappedInterval.top(self.width)
        lo = min(self.lo, other.lo)
        hi = max(self.hi, other.hi)
        result = WrappedInterval(lo, hi, self.width)
        if result.size() > self.modulus // 2:
            return WrappedInterval.top(self.width)
        return result

    def meet(self, other: "WrappedInterval") -> "WrappedInterval":
        if self.is_bottom or other.is_bottom:
            return WrappedInterval.bottom(self.width)
        if self.is_top:
            return other
        if other.is_top:
            return self
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return WrappedInterval.bottom(self.width)
        return WrappedInterval(lo, hi, self.width)

    def __repr__(self) -> str:
        if self.is_bottom:
            return f"WrappedInterval(⊥, w={self.width})"
        if self.is_top:
            return f"WrappedInterval(⊤, w={self.width})"
        return f"WrappedInterval([{self.lo}, {self.hi}], w={self.width})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WrappedInterval):
            return NotImplemented
        if self.is_bottom and other.is_bottom:
            return True
        if self.is_top and other.is_top:
            return True
        return self.lo == other.lo and self.hi == other.hi and self.width == other.width

    def __hash__(self) -> int:
        if self.is_bottom:
            return hash(("wrapped_bottom", self.width))
        if self.is_top:
            return hash(("wrapped_top", self.width))
        return hash((self.lo, self.hi, self.width))


# ===================================================================
# Interval abstract transformers
# ===================================================================


class IntervalArithTransformer:
    """Abstract transformer for arithmetic operations on intervals."""

    @staticmethod
    def add(a: Interval, b: Interval) -> Interval:
        return a + b

    @staticmethod
    def sub(a: Interval, b: Interval) -> Interval:
        return a - b

    @staticmethod
    def mul(a: Interval, b: Interval) -> Interval:
        return a * b

    @staticmethod
    def floordiv(a: Interval, b: Interval) -> Interval:
        return a.floordiv(b)

    @staticmethod
    def mod(a: Interval, b: Interval) -> Interval:
        return a.mod(b)

    @staticmethod
    def power(a: Interval, b: Interval) -> Interval:
        return a.power(b)

    @staticmethod
    def neg(a: Interval) -> Interval:
        return -a

    @staticmethod
    def abs_(a: Interval) -> Interval:
        return a.abs()

    @staticmethod
    def min_(a: Interval, b: Interval) -> Interval:
        return a.min(b)

    @staticmethod
    def max_(a: Interval, b: Interval) -> Interval:
        return a.max(b)


class IntervalBitwiseTransformer:
    """Abstract transformer for bitwise operations."""

    @staticmethod
    def and_(a: Interval, b: Interval) -> Interval:
        return a.bitwise_and(b)

    @staticmethod
    def or_(a: Interval, b: Interval) -> Interval:
        return a.bitwise_or(b)

    @staticmethod
    def xor(a: Interval, b: Interval) -> Interval:
        return a.bitwise_xor(b)

    @staticmethod
    def lshift(a: Interval, b: Interval) -> Interval:
        return a.left_shift(b)

    @staticmethod
    def rshift(a: Interval, b: Interval) -> Interval:
        return a.right_shift(b)


class IntervalComparisonTransformer:
    """Refine intervals from comparisons, returning refined (lhs, rhs)."""

    @staticmethod
    def lt(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_lt(lhs, rhs, on_true=branch)

    @staticmethod
    def le(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_le(lhs, rhs, on_true=branch)

    @staticmethod
    def gt(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_gt(lhs, rhs, on_true=branch)

    @staticmethod
    def ge(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_ge(lhs, rhs, on_true=branch)

    @staticmethod
    def eq(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_eq(lhs, rhs, on_true=branch)

    @staticmethod
    def ne(lhs: Interval, rhs: Interval, branch: bool) -> Tuple[Interval, Interval]:
        return IntervalRefiner.refine_ne(lhs, rhs, on_true=branch)


class IntervalLenTransformer:
    """Interval transformers for length-related operations."""

    @staticmethod
    def len_result() -> Interval:
        return Interval.for_len()

    @staticmethod
    def range_bounds(n: Interval) -> Interval:
        return Interval.for_range(n)

    @staticmethod
    def string_len() -> Interval:
        return Interval.for_string_len()

    @staticmethod
    def container_len(known_size: Optional[int] = None) -> Interval:
        if known_size is not None:
            return Interval.singleton(known_size)
        return Interval.non_negative()


# ===================================================================
# Full IntervalTransformer (AbstractTransformer implementation)
# ===================================================================


class IntervalTransformer(AbstractTransformer[IntervalValue]):
    """Full abstract transformer for the interval domain."""

    def __init__(self, domain: IntervalDomain):
        self.domain = domain
        self.arith = IntervalArithTransformer()
        self.bitwise = IntervalBitwiseTransformer()
        self.comparison = IntervalComparisonTransformer()
        self.len_xf = IntervalLenTransformer()

    def assign(
        self, state: AbstractState[IntervalValue], var: str, expr: Any
    ) -> AbstractState[IntervalValue]:
        val = self._eval_expr(state, expr)
        return state.set(var, val)

    def guard(
        self, state: AbstractState[IntervalValue], condition: Any, branch: bool
    ) -> AbstractState[IntervalValue]:
        if not isinstance(condition, (list, tuple)) or len(condition) != 3:
            return state
        op, lhs_var, rhs_expr = condition
        lhs_val = state.get(lhs_var)
        if lhs_val is None:
            return state
        rhs_val = self._eval_expr(state, rhs_expr)
        refined_lhs, refined_rhs = self._refine_comparison(
            op, lhs_val.interval, rhs_val.interval, branch
        )
        new_state = state.set(lhs_var, IntervalValue(refined_lhs))
        if isinstance(rhs_expr, str):
            new_state = new_state.set(rhs_expr, IntervalValue(refined_rhs))
        return new_state

    def call(
        self,
        state: AbstractState[IntervalValue],
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[IntervalValue]:
        result = self._eval_call(state, func, args)
        if result_var is not None:
            return state.set(result_var, result)
        return state

    def _eval_expr(self, state: AbstractState[IntervalValue], expr: Any) -> IntervalValue:
        if isinstance(expr, int):
            return IntervalValue(Interval.singleton(expr))
        if isinstance(expr, str):
            val = state.get(expr)
            return val if val is not None else self.domain.top()
        if isinstance(expr, (list, tuple)):
            if len(expr) == 3:
                op, left, right = expr
                lv = self._eval_expr(state, left)
                rv = self._eval_expr(state, right)
                return self._eval_binop(op, lv, rv)
            if len(expr) == 2:
                op, operand = expr
                v = self._eval_expr(state, operand)
                return self._eval_unaryop(op, v)
        return self.domain.top()

    def _eval_binop(
        self, op: str, a: IntervalValue, b: IntervalValue
    ) -> IntervalValue:
        ops = {
            "+": self.arith.add,
            "-": self.arith.sub,
            "*": self.arith.mul,
            "//": self.arith.floordiv,
            "%": self.arith.mod,
            "**": self.arith.power,
            "&": self.bitwise.and_,
            "|": self.bitwise.or_,
            "^": self.bitwise.xor,
            "<<": self.bitwise.lshift,
            ">>": self.bitwise.rshift,
            "min": self.arith.min_,
            "max": self.arith.max_,
        }
        fn = ops.get(op)
        if fn is not None:
            return IntervalValue(fn(a.interval, b.interval))
        return self.domain.top()

    def _eval_unaryop(self, op: str, a: IntervalValue) -> IntervalValue:
        if op == "-":
            return IntervalValue(self.arith.neg(a.interval))
        if op == "abs":
            return IntervalValue(self.arith.abs_(a.interval))
        return self.domain.top()

    def _eval_call(
        self,
        state: AbstractState[IntervalValue],
        func: str,
        args: List[Any],
    ) -> IntervalValue:
        if func == "len":
            if args:
                arg_val = self._eval_expr(state, args[0])
                if arg_val.interval.is_non_negative:
                    return arg_val
            return IntervalValue(Interval.for_len())
        if func == "range" and args:
            n = self._eval_expr(state, args[0])
            return IntervalValue(Interval.for_range(n.interval))
        if func == "abs" and args:
            v = self._eval_expr(state, args[0])
            return IntervalValue(v.interval.abs())
        if func == "min" and len(args) >= 2:
            a = self._eval_expr(state, args[0])
            b = self._eval_expr(state, args[1])
            return IntervalValue(a.interval.min(b.interval))
        if func == "max" and len(args) >= 2:
            a = self._eval_expr(state, args[0])
            b = self._eval_expr(state, args[1])
            return IntervalValue(a.interval.max(b.interval))
        return self.domain.top()

    def _refine_comparison(
        self, op: str, lhs: Interval, rhs: Interval, branch: bool
    ) -> Tuple[Interval, Interval]:
        return BackwardRefiner.refine(op, lhs, rhs, branch)
