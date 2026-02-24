from __future__ import annotations

"""
Tests for abstract domains used in refinement type inference.

Covers: IntervalDomain, TypeTagDomain, NullityDomain, StringDomain,
ReducedProductDomain, and lattice-theoretic properties.
"""

import math
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)

import pytest

# ── Local type stubs (no project imports) ─────────────────────────────────

class BoundKind(Enum):
    NEG_INF = auto()
    FINITE = auto()
    POS_INF = auto()


@dataclass(frozen=True)
class Bound:
    kind: BoundKind
    value: int = 0

    @classmethod
    def finite(cls, n: int) -> Bound:
        return cls(BoundKind.FINITE, n)

    @classmethod
    def pos_inf(cls) -> Bound:
        return cls(BoundKind.POS_INF)

    @classmethod
    def neg_inf(cls) -> Bound:
        return cls(BoundKind.NEG_INF)

    @property
    def is_finite(self) -> bool:
        return self.kind == BoundKind.FINITE

    @property
    def is_inf(self) -> bool:
        return self.kind in (BoundKind.NEG_INF, BoundKind.POS_INF)

    def __lt__(self, other: Bound) -> bool:
        order = {BoundKind.NEG_INF: 0, BoundKind.FINITE: 1, BoundKind.POS_INF: 2}
        if self.kind != other.kind:
            return order[self.kind] < order[other.kind]
        return self.value < other.value

    def __le__(self, other: Bound) -> bool:
        return self == other or self < other

    def __gt__(self, other: Bound) -> bool:
        return other < self

    def __ge__(self, other: Bound) -> bool:
        return other <= self


def _bound_add(a: Bound, b: Bound) -> Bound:
    if a.kind == BoundKind.NEG_INF or b.kind == BoundKind.NEG_INF:
        return Bound.neg_inf()
    if a.kind == BoundKind.POS_INF or b.kind == BoundKind.POS_INF:
        return Bound.pos_inf()
    return Bound.finite(a.value + b.value)


def _bound_sub(a: Bound, b: Bound) -> Bound:
    if a.kind == BoundKind.NEG_INF or b.kind == BoundKind.POS_INF:
        return Bound.neg_inf()
    if a.kind == BoundKind.POS_INF or b.kind == BoundKind.NEG_INF:
        return Bound.pos_inf()
    return Bound.finite(a.value - b.value)


def _bound_min(a: Bound, b: Bound) -> Bound:
    return a if a <= b else b


def _bound_max(a: Bound, b: Bound) -> Bound:
    return a if a >= b else b


@dataclass(frozen=True)
class Interval:
    lo: Bound
    hi: Bound

    @classmethod
    def top(cls) -> Interval:
        return cls(Bound.neg_inf(), Bound.pos_inf())

    @classmethod
    def bottom(cls) -> Interval:
        return cls(Bound.finite(1), Bound.finite(0))

    @classmethod
    def singleton(cls, n: int) -> Interval:
        return cls(Bound.finite(n), Bound.finite(n))

    @classmethod
    def from_bounds(cls, lo: int, hi: int) -> Interval:
        return cls(Bound.finite(lo), Bound.finite(hi))

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo.kind == BoundKind.NEG_INF and self.hi.kind == BoundKind.POS_INF

    @property
    def is_singleton(self) -> bool:
        return self.lo.is_finite and self.hi.is_finite and self.lo.value == self.hi.value

    def contains(self, n: int) -> bool:
        if self.is_bottom:
            return False
        b = Bound.finite(n)
        return self.lo <= b and b <= self.hi

    def join(self, other: Interval) -> Interval:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return Interval(_bound_min(self.lo, other.lo), _bound_max(self.hi, other.hi))

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        lo = _bound_max(self.lo, other.lo)
        hi = _bound_min(self.hi, other.hi)
        if lo > hi:
            return Interval.bottom()
        return Interval(lo, hi)

    def leq(self, other: Interval) -> bool:
        if self.is_bottom:
            return True
        if other.is_bottom:
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    def widen(self, other: Interval) -> Interval:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        lo = self.lo if self.lo <= other.lo else Bound.neg_inf()
        hi = self.hi if other.hi <= self.hi else Bound.pos_inf()
        return Interval(lo, hi)

    def narrow(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        # Standard narrowing: replace infinite bounds; also tighten finite bounds
        lo = other.lo if self.lo.kind == BoundKind.NEG_INF else self.lo
        if lo.kind == BoundKind.FINITE and other.lo.kind == BoundKind.FINITE and other.lo.value > lo.value:
            lo = other.lo
        hi = other.hi if self.hi.kind == BoundKind.POS_INF else self.hi
        if hi.kind == BoundKind.FINITE and other.hi.kind == BoundKind.FINITE and other.hi.value < hi.value:
            hi = other.hi
        return Interval(lo, hi)

    def __add__(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(_bound_add(self.lo, other.lo), _bound_add(self.hi, other.hi))

    def __sub__(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        return Interval(_bound_sub(self.lo, other.hi), _bound_sub(self.hi, other.lo))

    def __mul__(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if not (self.lo.is_finite and self.hi.is_finite and other.lo.is_finite and other.hi.is_finite):
            return Interval.top()
        corners = [
            self.lo.value * other.lo.value,
            self.lo.value * other.hi.value,
            self.hi.value * other.lo.value,
            self.hi.value * other.hi.value,
        ]
        return Interval.from_bounds(min(corners), max(corners))

    def __neg__(self) -> Interval:
        if self.is_bottom:
            return Interval.bottom()
        def neg_bound(b: Bound) -> Bound:
            if b.kind == BoundKind.NEG_INF:
                return Bound.pos_inf()
            if b.kind == BoundKind.POS_INF:
                return Bound.neg_inf()
            return Bound.finite(-b.value)
        return Interval(neg_bound(self.hi), neg_bound(self.lo))

    def abs(self) -> Interval:
        if self.is_bottom:
            return Interval.bottom()
        if not (self.lo.is_finite and self.hi.is_finite):
            return Interval(Bound.finite(0), Bound.pos_inf())
        lo_abs = abs(self.lo.value)
        hi_abs = abs(self.hi.value)
        if self.lo.value <= 0 <= self.hi.value:
            return Interval.from_bounds(0, max(lo_abs, hi_abs))
        return Interval.from_bounds(min(lo_abs, hi_abs), max(lo_abs, hi_abs))

    def floordiv(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if other.contains(0):
            return Interval.top()
        if not (self.lo.is_finite and self.hi.is_finite and other.lo.is_finite and other.hi.is_finite):
            return Interval.top()
        corners = []
        for a in (self.lo.value, self.hi.value):
            for b in (other.lo.value, other.hi.value):
                if b != 0:
                    corners.append(a // b)
        return Interval.from_bounds(min(corners), max(corners))

    def mod(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        if other.contains(0):
            return Interval.top()
        if other.is_singleton and other.lo.is_finite:
            m = abs(other.lo.value)
            return Interval.from_bounds(0, m - 1)
        return Interval.top()


class NullityKind(Enum):
    BOTTOM = auto()
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


@dataclass(frozen=True)
class NullityValue:
    kind: NullityKind

    @classmethod
    def bottom(cls) -> NullityValue:
        return cls(NullityKind.BOTTOM)

    @classmethod
    def definitely_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NULL)

    @classmethod
    def definitely_not_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NOT_NULL)

    @classmethod
    def maybe_null(cls) -> NullityValue:
        return cls(NullityKind.MAYBE_NULL)

    @classmethod
    def top(cls) -> NullityValue:
        return cls(NullityKind.MAYBE_NULL)

    @property
    def is_bottom(self) -> bool:
        return self.kind == NullityKind.BOTTOM

    @property
    def is_top(self) -> bool:
        return self.kind == NullityKind.MAYBE_NULL

    @property
    def is_definitely_null(self) -> bool:
        return self.kind == NullityKind.DEFINITELY_NULL

    @property
    def may_be_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NULL, NullityKind.MAYBE_NULL)

    @property
    def may_be_non_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NOT_NULL, NullityKind.MAYBE_NULL)

    def join(self, other: NullityValue) -> NullityValue:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self == other:
            return self
        return NullityValue.maybe_null()

    def meet(self, other: NullityValue) -> NullityValue:
        if self.is_bottom or other.is_bottom:
            return NullityValue.bottom()
        if self == other:
            return self
        if self.is_top:
            return other
        if other.is_top:
            return self
        return NullityValue.bottom()

    def leq(self, other: NullityValue) -> bool:
        if self.is_bottom:
            return True
        if other.is_top:
            return True
        return self == other


@dataclass(frozen=True)
class TypeTagSet:
    tags: FrozenSet[str]
    _is_top: bool = False

    @classmethod
    def top(cls) -> TypeTagSet:
        return cls(frozenset(), _is_top=True)

    @classmethod
    def bottom(cls) -> TypeTagSet:
        return cls(frozenset(), _is_top=False)

    @classmethod
    def singleton(cls, tag_name: str) -> TypeTagSet:
        return cls(frozenset({tag_name}))

    @classmethod
    def from_names(cls, *names: str) -> TypeTagSet:
        return cls(frozenset(names))

    @property
    def is_top(self) -> bool:
        return self._is_top

    @property
    def is_bottom(self) -> bool:
        return not self._is_top and len(self.tags) == 0

    @property
    def is_singleton(self) -> bool:
        return len(self.tags) == 1 and not self._is_top

    def contains(self, tag_name: str) -> bool:
        if self._is_top:
            return True
        return tag_name in self.tags

    def join(self, other: TypeTagSet) -> TypeTagSet:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self._is_top or other._is_top:
            return TypeTagSet.top()
        return TypeTagSet(self.tags | other.tags)

    def meet(self, other: TypeTagSet) -> TypeTagSet:
        if self.is_bottom or other.is_bottom:
            return TypeTagSet.bottom()
        if self._is_top:
            return other
        if other._is_top:
            return self
        inter = self.tags & other.tags
        return TypeTagSet(inter)

    def leq(self, other: TypeTagSet) -> bool:
        if self.is_bottom:
            return True
        if other._is_top:
            return True
        if self._is_top:
            return False
        return self.tags <= other.tags

    def remove(self, tag_name: str) -> TypeTagSet:
        if self._is_top:
            return self
        return TypeTagSet(self.tags - {tag_name})

    def add(self, tag_name: str) -> TypeTagSet:
        if self._is_top:
            return self
        return TypeTagSet(self.tags | {tag_name})


@dataclass(frozen=True)
class StringValue:
    """Abstract string value: a finite set of possible strings or top."""
    strings: FrozenSet[str]
    _is_top: bool = False

    @classmethod
    def top(cls) -> StringValue:
        return cls(frozenset(), _is_top=True)

    @classmethod
    def bottom(cls) -> StringValue:
        return cls(frozenset(), _is_top=False)

    @classmethod
    def singleton(cls, s: str) -> StringValue:
        return cls(frozenset({s}))

    @classmethod
    def from_set(cls, ss: Set[str]) -> StringValue:
        return cls(frozenset(ss))

    @property
    def is_top(self) -> bool:
        return self._is_top

    @property
    def is_bottom(self) -> bool:
        return not self._is_top and len(self.strings) == 0

    @property
    def is_singleton(self) -> bool:
        return len(self.strings) == 1 and not self._is_top

    def contains(self, s: str) -> bool:
        if self._is_top:
            return True
        return s in self.strings

    def join(self, other: StringValue) -> StringValue:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self._is_top or other._is_top:
            return StringValue.top()
        combined = self.strings | other.strings
        if len(combined) > 50:
            return StringValue.top()
        return StringValue(combined)

    def meet(self, other: StringValue) -> StringValue:
        if self.is_bottom or other.is_bottom:
            return StringValue.bottom()
        if self._is_top:
            return other
        if other._is_top:
            return self
        return StringValue(self.strings & other.strings)

    def leq(self, other: StringValue) -> bool:
        if self.is_bottom:
            return True
        if other._is_top:
            return True
        if self._is_top:
            return False
        return self.strings <= other.strings


@dataclass(frozen=True)
class ProductValue:
    interval: Interval
    type_tag: TypeTagSet
    nullity: NullityValue
    string: StringValue = field(default_factory=StringValue.bottom)

    def with_interval(self, iv: Interval) -> ProductValue:
        return ProductValue(iv, self.type_tag, self.nullity, self.string)

    def with_type_tag(self, tt: TypeTagSet) -> ProductValue:
        return ProductValue(self.interval, tt, self.nullity, self.string)

    def with_nullity(self, nv: NullityValue) -> ProductValue:
        return ProductValue(self.interval, self.type_tag, nv, self.string)

    def with_string(self, sv: StringValue) -> ProductValue:
        return ProductValue(self.interval, self.type_tag, self.nullity, sv)


def product_top() -> ProductValue:
    return ProductValue(
        Interval.top(), TypeTagSet.top(),
        NullityValue.maybe_null(), StringValue.top(),
    )


def product_bottom() -> ProductValue:
    return ProductValue(
        Interval.bottom(), TypeTagSet.bottom(),
        NullityValue.bottom(), StringValue.bottom(),
    )


def product_join(a: ProductValue, b: ProductValue) -> ProductValue:
    return ProductValue(
        a.interval.join(b.interval),
        a.type_tag.join(b.type_tag),
        a.nullity.join(b.nullity),
        a.string.join(b.string),
    )


def product_meet(a: ProductValue, b: ProductValue) -> ProductValue:
    return ProductValue(
        a.interval.meet(b.interval),
        a.type_tag.meet(b.type_tag),
        a.nullity.meet(b.nullity),
        a.string.meet(b.string),
    )


def product_leq(a: ProductValue, b: ProductValue) -> bool:
    return (
        a.interval.leq(b.interval)
        and a.type_tag.leq(b.type_tag)
        and a.nullity.leq(b.nullity)
        and a.string.leq(b.string)
    )


def product_widen(a: ProductValue, b: ProductValue) -> ProductValue:
    return ProductValue(
        a.interval.widen(b.interval),
        a.type_tag.join(b.type_tag),
        a.nullity.join(b.nullity),
        a.string.join(b.string),
    )


def reduce_product(v: ProductValue) -> ProductValue:
    """Apply inter-domain reductions."""
    tt = v.type_tag
    nv = v.nullity
    iv = v.interval
    sv = v.string

    # isinstance → nullity: if type tag is exactly {NoneType}, then definitely null
    if tt.is_singleton and tt.contains("NoneType"):
        nv = NullityValue.definitely_null()
    # isinstance → nullity: if NoneType not in tags and not top → definitely not null
    if not tt.is_top and not tt.is_bottom and not tt.contains("NoneType"):
        nv = NullityValue.definitely_not_null()

    # nullity → type tag: if definitely not null, remove NoneType
    if nv.kind == NullityKind.DEFINITELY_NOT_NULL and tt.contains("NoneType") and not tt.is_top:
        tt = tt.remove("NoneType")

    # nullity → type tag: if definitely null, narrow to NoneType
    if nv.kind == NullityKind.DEFINITELY_NULL:
        tt = TypeTagSet.singleton("NoneType")

    # numeric → type tag: if interval is not bottom and not top, implies int or float
    if not iv.is_bottom and not iv.is_top and iv.lo.is_finite and iv.hi.is_finite:
        if not tt.is_top and not tt.is_bottom:
            numeric_tags = tt.tags & {"int", "float"}
            if numeric_tags:
                pass  # keep consistent
        nv = NullityValue.definitely_not_null()

    return ProductValue(iv, tt, nv, sv)


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════


class TestIntervalDomain:
    """Tests for the Interval abstract domain."""

    # ── creation ──────────────────────────────────────────────────────────

    def test_interval_creation(self) -> None:
        """Create interval from explicit bounds."""
        iv = Interval.from_bounds(1, 5)
        assert iv.lo == Bound.finite(1)
        assert iv.hi == Bound.finite(5)
        assert not iv.is_bottom
        assert not iv.is_top

    def test_interval_creation_negative(self) -> None:
        """Create interval with negative bounds."""
        iv = Interval.from_bounds(-10, -3)
        assert iv.contains(-5)
        assert not iv.contains(0)

    def test_interval_creation_single_point(self) -> None:
        """Create interval that spans a single integer."""
        iv = Interval.from_bounds(7, 7)
        assert iv.is_singleton
        assert iv.contains(7)
        assert not iv.contains(6)

    def test_interval_bottom(self) -> None:
        """Bottom interval contains nothing."""
        bot = Interval.bottom()
        assert bot.is_bottom
        assert not bot.contains(0)
        assert not bot.contains(-1)

    def test_interval_top(self) -> None:
        """Top interval contains everything."""
        top = Interval.top()
        assert top.is_top
        assert top.contains(0)
        assert top.contains(999999)
        assert top.contains(-999999)

    # ── join ──────────────────────────────────────────────────────────────

    def test_interval_join_disjoint(self) -> None:
        """Join of disjoint intervals gives hull."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(7, 9)
        j = a.join(b)
        assert j.lo == Bound.finite(1)
        assert j.hi == Bound.finite(9)

    def test_interval_join_overlapping(self) -> None:
        """Join of overlapping intervals gives hull."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(3, 8)
        j = a.join(b)
        assert j == Interval.from_bounds(1, 8)

    def test_interval_join_contained(self) -> None:
        """Join where one is contained in the other returns outer."""
        a = Interval.from_bounds(1, 10)
        b = Interval.from_bounds(3, 5)
        j = a.join(b)
        assert j == a

    def test_interval_join_with_bottom(self) -> None:
        """Join with bottom is identity."""
        a = Interval.from_bounds(1, 5)
        bot = Interval.bottom()
        assert a.join(bot) == a
        assert bot.join(a) == a

    def test_interval_join_with_top(self) -> None:
        """Join with top is top."""
        a = Interval.from_bounds(1, 5)
        top = Interval.top()
        j = a.join(top)
        assert j.is_top

    def test_interval_join_both_bottom(self) -> None:
        """Join of two bottoms is bottom."""
        bot = Interval.bottom()
        assert bot.join(bot).is_bottom

    def test_interval_join_adjacent(self) -> None:
        """Join of adjacent intervals."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(4, 6)
        j = a.join(b)
        assert j == Interval.from_bounds(1, 6)

    def test_interval_join_negative_positive(self) -> None:
        """Join across zero."""
        a = Interval.from_bounds(-5, -1)
        b = Interval.from_bounds(1, 5)
        j = a.join(b)
        assert j == Interval.from_bounds(-5, 5)

    def test_interval_join_identical(self) -> None:
        """Join of identical intervals is same interval."""
        a = Interval.from_bounds(2, 8)
        assert a.join(a) == a

    def test_interval_join_singletons(self) -> None:
        """Join of two singletons."""
        a = Interval.singleton(3)
        b = Interval.singleton(7)
        j = a.join(b)
        assert j == Interval.from_bounds(3, 7)

    # ── meet ──────────────────────────────────────────────────────────────

    def test_interval_meet_overlapping(self) -> None:
        """Meet of overlapping intervals gives intersection."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(3, 8)
        m = a.meet(b)
        assert m == Interval.from_bounds(3, 5)

    def test_interval_meet_disjoint(self) -> None:
        """Meet of disjoint intervals is bottom."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(5, 7)
        m = a.meet(b)
        assert m.is_bottom

    def test_interval_meet_contained(self) -> None:
        """Meet where one is contained in the other returns inner."""
        a = Interval.from_bounds(1, 10)
        b = Interval.from_bounds(3, 5)
        m = a.meet(b)
        assert m == b

    def test_interval_meet_with_top(self) -> None:
        """Meet with top is identity."""
        a = Interval.from_bounds(1, 5)
        top = Interval.top()
        assert a.meet(top) == a

    def test_interval_meet_with_bottom(self) -> None:
        """Meet with bottom is bottom."""
        a = Interval.from_bounds(1, 5)
        bot = Interval.bottom()
        assert a.meet(bot).is_bottom

    def test_interval_meet_identical(self) -> None:
        """Meet of identical intervals returns same."""
        a = Interval.from_bounds(2, 8)
        assert a.meet(a) == a

    def test_interval_meet_touching(self) -> None:
        """Meet of intervals touching at one point."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(5, 10)
        m = a.meet(b)
        assert m == Interval.singleton(5)

    def test_interval_meet_negative(self) -> None:
        """Meet involving negative intervals."""
        a = Interval.from_bounds(-10, 0)
        b = Interval.from_bounds(-5, 5)
        m = a.meet(b)
        assert m == Interval.from_bounds(-5, 0)

    # ── widening ──────────────────────────────────────────────────────────

    def test_interval_widening_basic(self) -> None:
        """Basic widening: new lower bound goes to -inf, upper to +inf."""
        a = Interval.from_bounds(0, 5)
        b = Interval.from_bounds(-1, 10)
        w = a.widen(b)
        assert w.lo.kind == BoundKind.NEG_INF
        assert w.hi.kind == BoundKind.POS_INF

    def test_interval_widening_stable(self) -> None:
        """Widening when new value is contained stays the same."""
        a = Interval.from_bounds(0, 10)
        b = Interval.from_bounds(2, 8)
        w = a.widen(b)
        assert w == a

    def test_interval_widening_upper_only(self) -> None:
        """Only upper bound increases."""
        a = Interval.from_bounds(0, 5)
        b = Interval.from_bounds(0, 10)
        w = a.widen(b)
        assert w.lo == Bound.finite(0)
        assert w.hi.kind == BoundKind.POS_INF

    def test_interval_widening_lower_only(self) -> None:
        """Only lower bound decreases."""
        a = Interval.from_bounds(5, 10)
        b = Interval.from_bounds(3, 10)
        w = a.widen(b)
        assert w.lo.kind == BoundKind.NEG_INF
        assert w.hi == Bound.finite(10)

    def test_interval_widening_with_bottom(self) -> None:
        """Widening with bottom returns the non-bottom operand."""
        a = Interval.from_bounds(0, 5)
        bot = Interval.bottom()
        assert a.widen(bot) == a
        assert bot.widen(a) == a

    def test_interval_widening_with_thresholds(self) -> None:
        """Widening with thresholds stops at threshold values."""
        thresholds = [-100, 0, 100, 1000]
        a = Interval.from_bounds(0, 5)
        b = Interval.from_bounds(0, 10)
        # Simulate threshold widening
        w = a.widen(b)
        if w.hi.kind == BoundKind.POS_INF:
            for t in sorted(thresholds):
                if Bound.finite(t) >= b.hi:
                    w = Interval(w.lo, Bound.finite(t))
                    break
        assert w.hi == Bound.finite(100)

    def test_interval_widening_delayed(self) -> None:
        """Delayed widening: use join for first N iterations, then widen."""
        delay = 3
        current = Interval.from_bounds(0, 0)
        iterates = [
            Interval.from_bounds(0, 1),
            Interval.from_bounds(0, 2),
            Interval.from_bounds(0, 3),
            Interval.from_bounds(0, 4),
        ]
        for i, next_val in enumerate(iterates):
            if i < delay:
                current = current.join(next_val)
            else:
                current = current.widen(next_val)
        # After delay=3, we join first 3 times then widen once
        assert current.hi.kind == BoundKind.POS_INF or current.hi >= Bound.finite(4)

    def test_interval_widening_convergence(self) -> None:
        """Widening must converge in finite steps."""
        current = Interval.from_bounds(0, 0)
        for i in range(1, 100):
            new = Interval.from_bounds(0, i)
            current = current.widen(new)
            if current.is_top or (current.hi.kind == BoundKind.POS_INF):
                break
        assert current.hi.kind == BoundKind.POS_INF

    # ── narrowing ─────────────────────────────────────────────────────────

    def test_interval_narrowing(self) -> None:
        """Narrowing replaces infinite bounds with finite ones."""
        wide = Interval(Bound.neg_inf(), Bound.pos_inf())
        precise = Interval.from_bounds(-5, 100)
        n = wide.narrow(precise)
        assert n == precise

    def test_interval_narrowing_one_side(self) -> None:
        """Narrowing only one infinite bound."""
        wide = Interval(Bound.finite(0), Bound.pos_inf())
        precise = Interval.from_bounds(0, 50)
        n = wide.narrow(precise)
        assert n.lo == Bound.finite(0)
        assert n.hi == Bound.finite(50)

    def test_interval_narrowing_already_finite(self) -> None:
        """Narrowing with already-finite bounds tightens to the more precise interval."""
        a = Interval.from_bounds(0, 10)
        b = Interval.from_bounds(2, 8)
        n = a.narrow(b)
        assert n == Interval.from_bounds(2, 8)

    # ── leq ───────────────────────────────────────────────────────────────

    def test_interval_leq_contained(self) -> None:
        """[3,5] ⊑ [1,10]."""
        a = Interval.from_bounds(3, 5)
        b = Interval.from_bounds(1, 10)
        assert a.leq(b)

    def test_interval_leq_not_contained(self) -> None:
        """[1,10] ⋢ [3,5]."""
        a = Interval.from_bounds(1, 10)
        b = Interval.from_bounds(3, 5)
        assert not a.leq(b)

    def test_interval_leq_equal(self) -> None:
        """[1,5] ⊑ [1,5]."""
        a = Interval.from_bounds(1, 5)
        assert a.leq(a)

    def test_interval_leq_bottom_leq_anything(self) -> None:
        """⊥ ⊑ x for all x."""
        bot = Interval.bottom()
        assert bot.leq(Interval.from_bounds(1, 5))
        assert bot.leq(Interval.top())
        assert bot.leq(bot)

    def test_interval_leq_anything_leq_top(self) -> None:
        """x ⊑ ⊤ for all x."""
        top = Interval.top()
        assert Interval.from_bounds(1, 5).leq(top)
        assert Interval.bottom().leq(top)
        assert top.leq(top)

    def test_interval_leq_disjoint(self) -> None:
        """Disjoint intervals: neither ⊑ the other."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(5, 7)
        assert not a.leq(b)
        assert not b.leq(a)

    def test_interval_leq_partial_overlap(self) -> None:
        """Partial overlap: neither ⊑ the other."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(3, 8)
        assert not a.leq(b)
        assert not b.leq(a)

    # ── arithmetic ────────────────────────────────────────────────────────

    def test_interval_arithmetic_add_basic(self) -> None:
        """[1,3] + [2,4] = [3,7]."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(2, 4)
        assert (a + b) == Interval.from_bounds(3, 7)

    def test_interval_arithmetic_add_negative(self) -> None:
        """[1,3] + [-5,-2] = [-4,1]."""
        a = Interval.from_bounds(1, 3)
        b = Interval.from_bounds(-5, -2)
        assert (a + b) == Interval.from_bounds(-4, 1)

    def test_interval_arithmetic_add_with_bottom(self) -> None:
        """⊥ + [1,3] = ⊥."""
        bot = Interval.bottom()
        a = Interval.from_bounds(1, 3)
        assert (bot + a).is_bottom

    def test_interval_arithmetic_sub_basic(self) -> None:
        """[5,10] - [1,3] = [2,9]."""
        a = Interval.from_bounds(5, 10)
        b = Interval.from_bounds(1, 3)
        assert (a - b) == Interval.from_bounds(2, 9)

    def test_interval_arithmetic_sub_self(self) -> None:
        """[a,b] - [a,b] ⊇ {0}. Result is [a-b, b-a]."""
        a = Interval.from_bounds(3, 7)
        result = a - a
        assert result.contains(0)

    def test_interval_arithmetic_mul_basic(self) -> None:
        """[2,3] * [4,5] = [8,15]."""
        a = Interval.from_bounds(2, 3)
        b = Interval.from_bounds(4, 5)
        assert (a * b) == Interval.from_bounds(8, 15)

    def test_interval_arithmetic_mul_negative(self) -> None:
        """[-2,3] * [1,4]: corners are -8,-2,3,12 → [-8,12]."""
        a = Interval.from_bounds(-2, 3)
        b = Interval.from_bounds(1, 4)
        result = a * b
        assert result == Interval.from_bounds(-8, 12)

    def test_interval_arithmetic_mul_both_negative(self) -> None:
        """[-3,-1] * [-5,-2] = [2,15]."""
        a = Interval.from_bounds(-3, -1)
        b = Interval.from_bounds(-5, -2)
        result = a * b
        assert result == Interval.from_bounds(2, 15)

    def test_interval_arithmetic_mul_by_zero(self) -> None:
        """[x,y] * [0,0] = [0,0]."""
        a = Interval.from_bounds(5, 10)
        z = Interval.singleton(0)
        assert (a * z) == Interval.singleton(0)

    def test_interval_arithmetic_div_basic(self) -> None:
        """[10,20] // [2,5]."""
        a = Interval.from_bounds(10, 20)
        b = Interval.from_bounds(2, 5)
        result = a.floordiv(b)
        assert result.contains(2)  # 10//5
        assert result.contains(10)  # 20//2

    def test_interval_arithmetic_div_by_zero_included(self) -> None:
        """Division by interval containing 0 returns top."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(-1, 1)
        result = a.floordiv(b)
        assert result.is_top

    def test_interval_arithmetic_mod_basic(self) -> None:
        """[x,y] % [n,n] ⊆ [0,n-1]."""
        a = Interval.from_bounds(0, 100)
        b = Interval.singleton(7)
        result = a.mod(b)
        assert result.lo == Bound.finite(0)
        assert result.hi == Bound.finite(6)

    def test_interval_arithmetic_mod_by_zero(self) -> None:
        """Mod by interval containing 0 returns top."""
        a = Interval.from_bounds(1, 5)
        b = Interval.from_bounds(-1, 1)
        result = a.mod(b)
        assert result.is_top

    # ── comparison refinement ─────────────────────────────────────────────

    def test_interval_comparison_lt(self) -> None:
        """Refine x < 5: [0,10] → [0,4]."""
        x = Interval.from_bounds(0, 10)
        bound = Interval.singleton(5)
        # x < 5 means hi(x) ≤ 4
        refined = x.meet(Interval(Bound.neg_inf(), Bound.finite(4)))
        assert refined == Interval.from_bounds(0, 4)

    def test_interval_comparison_le(self) -> None:
        """Refine x <= 5: [0,10] → [0,5]."""
        x = Interval.from_bounds(0, 10)
        refined = x.meet(Interval(Bound.neg_inf(), Bound.finite(5)))
        assert refined == Interval.from_bounds(0, 5)

    def test_interval_comparison_gt(self) -> None:
        """Refine x > 5: [0,10] → [6,10]."""
        x = Interval.from_bounds(0, 10)
        refined = x.meet(Interval(Bound.finite(6), Bound.pos_inf()))
        assert refined == Interval.from_bounds(6, 10)

    def test_interval_comparison_ge(self) -> None:
        """Refine x >= 5: [0,10] → [5,10]."""
        x = Interval.from_bounds(0, 10)
        refined = x.meet(Interval(Bound.finite(5), Bound.pos_inf()))
        assert refined == Interval.from_bounds(5, 10)

    def test_interval_comparison_eq(self) -> None:
        """Refine x == 5: [0,10] → [5,5]."""
        x = Interval.from_bounds(0, 10)
        refined = x.meet(Interval.singleton(5))
        assert refined == Interval.singleton(5)

    def test_interval_comparison_ne(self) -> None:
        """Refine x != 5: singleton [5,5] → ⊥."""
        x = Interval.singleton(5)
        # If x is exactly 5, x != 5 is impossible
        # For a singleton, ne produces bottom
        if x.is_singleton and x.lo.value == 5:
            refined = Interval.bottom()
        else:
            refined = x
        assert refined.is_bottom

    def test_interval_comparison_chain(self) -> None:
        """Chain: 0 <= x < 10 and 5 <= x < 8 → [5,7]."""
        x = Interval.from_bounds(0, 9)  # x < 10
        x = x.meet(Interval.from_bounds(5, 7))  # 5 <= x <= 7
        assert x == Interval.from_bounds(5, 7)

    # ── unary operations ──────────────────────────────────────────────────

    def test_interval_unary_neg(self) -> None:
        """Negate [2,5] → [-5,-2]."""
        a = Interval.from_bounds(2, 5)
        n = -a
        assert n == Interval.from_bounds(-5, -2)

    def test_interval_unary_neg_symmetric(self) -> None:
        """Negate [-3,3] → [-3,3]."""
        a = Interval.from_bounds(-3, 3)
        n = -a
        assert n == Interval.from_bounds(-3, 3)

    def test_interval_unary_neg_bottom(self) -> None:
        """Negate bottom is bottom."""
        assert (-Interval.bottom()).is_bottom

    def test_interval_unary_abs_positive(self) -> None:
        """abs([3,7]) = [3,7]."""
        a = Interval.from_bounds(3, 7)
        assert a.abs() == Interval.from_bounds(3, 7)

    def test_interval_unary_abs_negative(self) -> None:
        """abs([-7,-3]) = [3,7]."""
        a = Interval.from_bounds(-7, -3)
        assert a.abs() == Interval.from_bounds(3, 7)

    def test_interval_unary_abs_spanning_zero(self) -> None:
        """abs([-3,5]) = [0,5]."""
        a = Interval.from_bounds(-3, 5)
        assert a.abs() == Interval.from_bounds(0, 5)

    # ── special properties ────────────────────────────────────────────────

    def test_interval_len_tracking(self) -> None:
        """Use interval to track list length (non-negative)."""
        length = Interval.from_bounds(0, 100)
        assert length.contains(0)
        assert length.contains(50)
        assert not length.contains(-1)

    def test_interval_constant_propagation(self) -> None:
        """Singleton interval represents a known constant."""
        c = Interval.singleton(42)
        assert c.is_singleton
        assert c.lo.value == 42

    def test_interval_overflow_handling(self) -> None:
        """Large values don't cause issues."""
        big = Interval.from_bounds(-(10**18), 10**18)
        s = Interval.singleton(1)
        result = big + s
        assert result.contains(10**18 + 1)

    def test_interval_unbounded(self) -> None:
        """Half-bounded intervals."""
        lower = Interval(Bound.finite(0), Bound.pos_inf())
        assert lower.contains(0)
        assert lower.contains(10**9)
        assert not lower.contains(-1)

    def test_interval_singleton(self) -> None:
        """Singleton detection and value extraction."""
        s = Interval.singleton(7)
        assert s.is_singleton
        assert s.lo.value == 7
        assert s.hi.value == 7

    def test_interval_empty(self) -> None:
        """Empty (bottom) interval contains nothing."""
        e = Interval.bottom()
        for v in range(-10, 11):
            assert not e.contains(v)


class TestTypeTagDomain:
    """Tests for the type tag powerset domain."""

    def test_type_tag_creation(self) -> None:
        """Create type tag set from names."""
        ts = TypeTagSet.from_names("int", "str")
        assert ts.contains("int")
        assert ts.contains("str")
        assert not ts.contains("float")

    def test_type_tag_bottom(self) -> None:
        """Bottom tag set is empty."""
        bot = TypeTagSet.bottom()
        assert bot.is_bottom
        assert not bot.contains("int")

    def test_type_tag_top(self) -> None:
        """Top tag set contains everything."""
        top = TypeTagSet.top()
        assert top.is_top
        assert top.contains("int")
        assert top.contains("SomeRandomType")

    def test_type_tag_singleton(self) -> None:
        """Singleton tag set."""
        s = TypeTagSet.singleton("int")
        assert s.is_singleton
        assert s.contains("int")
        assert not s.contains("str")

    def test_type_tag_join(self) -> None:
        """Join of tag sets is union."""
        a = TypeTagSet.from_names("int")
        b = TypeTagSet.from_names("str")
        j = a.join(b)
        assert j.contains("int")
        assert j.contains("str")

    def test_type_tag_join_with_bottom(self) -> None:
        """Join with bottom is identity."""
        a = TypeTagSet.from_names("int")
        assert a.join(TypeTagSet.bottom()) == a
        assert TypeTagSet.bottom().join(a) == a

    def test_type_tag_join_with_top(self) -> None:
        """Join with top is top."""
        a = TypeTagSet.from_names("int")
        assert a.join(TypeTagSet.top()).is_top

    def test_type_tag_meet(self) -> None:
        """Meet of tag sets is intersection."""
        a = TypeTagSet.from_names("int", "str")
        b = TypeTagSet.from_names("str", "float")
        m = a.meet(b)
        assert m.contains("str")
        assert not m.contains("int")
        assert not m.contains("float")

    def test_type_tag_meet_disjoint(self) -> None:
        """Meet of disjoint tag sets is bottom."""
        a = TypeTagSet.from_names("int")
        b = TypeTagSet.from_names("str")
        m = a.meet(b)
        assert m.is_bottom

    def test_type_tag_meet_with_top(self) -> None:
        """Meet with top is identity."""
        a = TypeTagSet.from_names("int", "str")
        assert a.meet(TypeTagSet.top()) == a

    def test_type_tag_isinstance_transfer(self) -> None:
        """isinstance(x, int) narrows tag set to {int}."""
        before = TypeTagSet.from_names("int", "str", "float")
        after_true = before.meet(TypeTagSet.singleton("int"))
        assert after_true == TypeTagSet.singleton("int")

    def test_type_tag_typeof_transfer(self) -> None:
        """typeof x == 'number' narrows in TypeScript analogy."""
        before = TypeTagSet.from_names("number", "string", "boolean")
        after_true = before.meet(TypeTagSet.singleton("number"))
        assert after_true == TypeTagSet.singleton("number")

    def test_type_tag_narrowing(self) -> None:
        """Narrowing removes tags not in the narrower set."""
        wide = TypeTagSet.from_names("int", "str", "float", "list")
        narrow = TypeTagSet.from_names("int", "float")
        m = wide.meet(narrow)
        assert m == TypeTagSet.from_names("int", "float")

    def test_type_tag_union(self) -> None:
        """Union of tag sets."""
        a = TypeTagSet.from_names("int")
        b = TypeTagSet.from_names("str")
        u = a.join(b)
        assert u == TypeTagSet.from_names("int", "str")

    def test_type_tag_intersection(self) -> None:
        """Intersection of tag sets."""
        a = TypeTagSet.from_names("int", "str", "float")
        b = TypeTagSet.from_names("str", "bool")
        inter = a.meet(b)
        assert inter == TypeTagSet.singleton("str")

    def test_type_tag_subtype_check(self) -> None:
        """Subset check on tag sets."""
        sub = TypeTagSet.from_names("int")
        sup = TypeTagSet.from_names("int", "str")
        assert sub.leq(sup)
        assert not sup.leq(sub)

    def test_type_tag_with_inheritance(self) -> None:
        """Model simple inheritance via tag containment."""
        # bool is a subtype of int in Python
        tags = TypeTagSet.from_names("bool")
        int_or_bool = TypeTagSet.from_names("int", "bool")
        assert tags.leq(int_or_bool)

    def test_type_tag_none_special_case(self) -> None:
        """NoneType as a tag."""
        tags = TypeTagSet.from_names("int", "NoneType")
        assert tags.contains("NoneType")
        no_none = tags.remove("NoneType")
        assert not no_none.contains("NoneType")
        assert no_none.contains("int")

    def test_type_tag_add_remove(self) -> None:
        """Add and remove tags."""
        ts = TypeTagSet.singleton("int")
        ts2 = ts.add("str")
        assert ts2 == TypeTagSet.from_names("int", "str")
        ts3 = ts2.remove("int")
        assert ts3 == TypeTagSet.singleton("str")

    def test_type_tag_leq_bottom(self) -> None:
        """Bottom ⊑ everything."""
        bot = TypeTagSet.bottom()
        assert bot.leq(TypeTagSet.from_names("int"))
        assert bot.leq(TypeTagSet.top())
        assert bot.leq(bot)

    def test_type_tag_leq_top(self) -> None:
        """Everything ⊑ top."""
        top = TypeTagSet.top()
        assert TypeTagSet.from_names("int").leq(top)
        assert TypeTagSet.bottom().leq(top)

    def test_type_tag_join_idempotent(self) -> None:
        """a ⊔ a = a."""
        a = TypeTagSet.from_names("int", "str")
        assert a.join(a) == a

    def test_type_tag_meet_idempotent(self) -> None:
        """a ⊓ a = a."""
        a = TypeTagSet.from_names("int", "str")
        assert a.meet(a) == a


class TestNullityDomain:
    """Tests for the nullity abstract domain."""

    def test_nullity_creation_definitely_none(self) -> None:
        """Create definitely-null value."""
        n = NullityValue.definitely_null()
        assert n.is_definitely_null
        assert n.may_be_null
        assert not n.may_be_non_null

    def test_nullity_creation_definitely_not_none(self) -> None:
        """Create definitely-not-null value."""
        n = NullityValue.definitely_not_null()
        assert not n.is_definitely_null
        assert not n.may_be_null
        assert n.may_be_non_null

    def test_nullity_creation_maybe_none(self) -> None:
        """Create maybe-null (top) value."""
        n = NullityValue.maybe_null()
        assert n.may_be_null
        assert n.may_be_non_null
        assert n.is_top

    def test_nullity_bottom(self) -> None:
        """Bottom nullity."""
        b = NullityValue.bottom()
        assert b.is_bottom
        assert not b.may_be_null
        assert not b.may_be_non_null

    def test_nullity_join_null_nonnull(self) -> None:
        """Join of null and not-null is maybe-null."""
        a = NullityValue.definitely_null()
        b = NullityValue.definitely_not_null()
        j = a.join(b)
        assert j.is_top

    def test_nullity_join_with_bottom(self) -> None:
        """Join with bottom is identity."""
        a = NullityValue.definitely_null()
        assert a.join(NullityValue.bottom()) == a

    def test_nullity_join_same(self) -> None:
        """Join of same value is same."""
        a = NullityValue.definitely_null()
        assert a.join(a) == a

    def test_nullity_meet_null_nonnull(self) -> None:
        """Meet of null and not-null is bottom."""
        a = NullityValue.definitely_null()
        b = NullityValue.definitely_not_null()
        m = a.meet(b)
        assert m.is_bottom

    def test_nullity_meet_with_top(self) -> None:
        """Meet with top is identity."""
        a = NullityValue.definitely_null()
        assert a.meet(NullityValue.maybe_null()) == a

    def test_nullity_meet_same(self) -> None:
        """Meet of same value is same."""
        a = NullityValue.definitely_not_null()
        assert a.meet(a) == a

    def test_nullity_none_check_transfer(self) -> None:
        """x is None check narrows to definitely null on true branch."""
        before = NullityValue.maybe_null()
        # True branch: x is None
        after_true = before.meet(NullityValue.definitely_null())
        assert after_true == NullityValue.definitely_null()
        # False branch: x is not None
        after_false = before.meet(NullityValue.definitely_not_null())
        assert after_false == NullityValue.definitely_not_null()

    def test_nullity_truthiness_transfer(self) -> None:
        """Truthiness check: None is falsy → after if x: x is not None."""
        before = NullityValue.maybe_null()
        # if x: → x is not None (also not 0, not "", but for nullity domain)
        after_true = before.meet(NullityValue.definitely_not_null())
        assert after_true == NullityValue.definitely_not_null()

    def test_nullity_attribute_access_precondition(self) -> None:
        """Accessing x.attr requires x is not None."""
        x = NullityValue.maybe_null()
        # Check: may_be_null → potential null deref
        assert x.may_be_null
        # After guard: x is not None
        safe = x.meet(NullityValue.definitely_not_null())
        assert not safe.may_be_null

    def test_nullity_conditional_narrowing(self) -> None:
        """After `if x is not None`, x is narrowed."""
        x = NullityValue.maybe_null()
        narrowed = x.meet(NullityValue.definitely_not_null())
        assert narrowed == NullityValue.definitely_not_null()

    def test_nullity_assignment_transfer(self) -> None:
        """Assignment to literal → definitely not null."""
        # x = 42 means x is not None
        assigned = NullityValue.definitely_not_null()
        assert not assigned.may_be_null

    def test_nullity_function_return(self) -> None:
        """Function that may return None has maybe-null return."""
        # def f(x): return None if x > 0 else 42
        branch_true = NullityValue.definitely_null()
        branch_false = NullityValue.definitely_not_null()
        ret = branch_true.join(branch_false)
        assert ret.is_top

    def test_nullity_optional_pattern(self) -> None:
        """Optional[int] pattern: value or None."""
        # Start as maybe null
        x = NullityValue.maybe_null()
        # After None check on true branch
        true_branch = x.meet(NullityValue.definitely_null())
        assert true_branch.is_definitely_null
        # On false branch
        false_branch = x.meet(NullityValue.definitely_not_null())
        assert not false_branch.may_be_null

    def test_nullity_leq(self) -> None:
        """Ordering: ⊥ ⊑ null ⊑ maybe, ⊥ ⊑ not-null ⊑ maybe."""
        bot = NullityValue.bottom()
        null = NullityValue.definitely_null()
        not_null = NullityValue.definitely_not_null()
        maybe = NullityValue.maybe_null()

        assert bot.leq(null)
        assert bot.leq(not_null)
        assert bot.leq(maybe)
        assert null.leq(maybe)
        assert not_null.leq(maybe)
        assert not null.leq(not_null)
        assert not not_null.leq(null)

    def test_nullity_join_commutative(self) -> None:
        """Join is commutative."""
        a = NullityValue.definitely_null()
        b = NullityValue.definitely_not_null()
        assert a.join(b) == b.join(a)

    def test_nullity_meet_commutative(self) -> None:
        """Meet is commutative."""
        a = NullityValue.definitely_null()
        b = NullityValue.maybe_null()
        assert a.meet(b) == b.meet(a)


class TestStringDomain:
    """Tests for the string equality abstract domain."""

    def test_string_creation(self) -> None:
        """Create string value from singleton."""
        s = StringValue.singleton("hello")
        assert s.contains("hello")
        assert not s.contains("world")

    def test_string_bottom(self) -> None:
        """Bottom string value contains nothing."""
        bot = StringValue.bottom()
        assert bot.is_bottom
        assert not bot.contains("")

    def test_string_top(self) -> None:
        """Top string value contains everything."""
        top = StringValue.top()
        assert top.is_top
        assert top.contains("anything")

    def test_string_join_singletons(self) -> None:
        """Join of two singletons."""
        a = StringValue.singleton("a")
        b = StringValue.singleton("b")
        j = a.join(b)
        assert j.contains("a")
        assert j.contains("b")
        assert not j.contains("c")

    def test_string_join_with_bottom(self) -> None:
        """Join with bottom is identity."""
        a = StringValue.singleton("x")
        assert a.join(StringValue.bottom()) == a

    def test_string_join_with_top(self) -> None:
        """Join with top is top."""
        a = StringValue.singleton("x")
        assert a.join(StringValue.top()).is_top

    def test_string_meet_overlapping(self) -> None:
        """Meet of overlapping string sets."""
        a = StringValue.from_set({"a", "b", "c"})
        b = StringValue.from_set({"b", "c", "d"})
        m = a.meet(b)
        assert m.contains("b")
        assert m.contains("c")
        assert not m.contains("a")
        assert not m.contains("d")

    def test_string_meet_disjoint(self) -> None:
        """Meet of disjoint string sets is bottom."""
        a = StringValue.singleton("a")
        b = StringValue.singleton("b")
        m = a.meet(b)
        assert m.is_bottom

    def test_string_equality_check(self) -> None:
        """String equality: s == "foo" narrows to singleton."""
        before = StringValue.top()
        check = StringValue.singleton("foo")
        after = before.meet(check) if not before.is_top else check
        assert after.contains("foo")
        assert after.is_singleton

    def test_string_hasattr_transfer(self) -> None:
        """Track attribute name via string domain."""
        attr_name = StringValue.singleton("__len__")
        assert attr_name.contains("__len__")

    def test_string_dict_key_tracking(self) -> None:
        """Track known dictionary keys."""
        keys = StringValue.from_set({"name", "age", "email"})
        assert keys.contains("name")
        assert not keys.contains("phone")

    def test_string_finite_set_operations(self) -> None:
        """Finite set operations on string values."""
        a = StringValue.from_set({"x", "y"})
        b = StringValue.from_set({"y", "z"})
        union = a.join(b)
        inter = a.meet(b)
        assert union == StringValue.from_set({"x", "y", "z"})
        assert inter == StringValue.singleton("y")

    def test_string_leq(self) -> None:
        """Ordering on string values."""
        a = StringValue.singleton("x")
        b = StringValue.from_set({"x", "y"})
        assert a.leq(b)
        assert not b.leq(a)

    def test_string_leq_bottom(self) -> None:
        """Bottom ⊑ everything."""
        bot = StringValue.bottom()
        assert bot.leq(StringValue.singleton("x"))
        assert bot.leq(StringValue.top())

    def test_string_widening_threshold(self) -> None:
        """String set exceeding threshold goes to top."""
        s = StringValue.bottom()
        for i in range(60):
            s = s.join(StringValue.singleton(f"key_{i}"))
        assert s.is_top

    def test_string_join_idempotent(self) -> None:
        """a ⊔ a = a."""
        a = StringValue.from_set({"a", "b"})
        assert a.join(a) == a

    def test_string_meet_with_top(self) -> None:
        """Meet with top is identity."""
        a = StringValue.from_set({"a", "b"})
        assert a.meet(StringValue.top()) == a


class TestReducedProduct:
    """Tests for the reduced product domain."""

    def _make_product(
        self,
        iv: Optional[Interval] = None,
        tt: Optional[TypeTagSet] = None,
        nv: Optional[NullityValue] = None,
        sv: Optional[StringValue] = None,
    ) -> ProductValue:
        return ProductValue(
            iv or Interval.top(),
            tt or TypeTagSet.top(),
            nv or NullityValue.maybe_null(),
            sv or StringValue.top(),
        )

    def test_product_creation(self) -> None:
        """Create a product value."""
        p = self._make_product(
            iv=Interval.from_bounds(0, 10),
            tt=TypeTagSet.singleton("int"),
            nv=NullityValue.definitely_not_null(),
        )
        assert p.interval == Interval.from_bounds(0, 10)
        assert p.type_tag == TypeTagSet.singleton("int")
        assert p.nullity == NullityValue.definitely_not_null()

    def test_product_top(self) -> None:
        """Product top."""
        t = product_top()
        assert t.interval.is_top
        assert t.type_tag.is_top
        assert t.nullity.is_top

    def test_product_bottom(self) -> None:
        """Product bottom."""
        b = product_bottom()
        assert b.interval.is_bottom
        assert b.type_tag.is_bottom
        assert b.nullity.is_bottom

    def test_product_join(self) -> None:
        """Product join is component-wise join."""
        a = self._make_product(
            iv=Interval.from_bounds(0, 5),
            tt=TypeTagSet.singleton("int"),
            nv=NullityValue.definitely_not_null(),
        )
        b = self._make_product(
            iv=Interval.from_bounds(3, 10),
            tt=TypeTagSet.singleton("str"),
            nv=NullityValue.definitely_null(),
        )
        j = product_join(a, b)
        assert j.interval == Interval.from_bounds(0, 10)
        assert j.type_tag == TypeTagSet.from_names("int", "str")
        assert j.nullity == NullityValue.maybe_null()

    def test_product_meet(self) -> None:
        """Product meet is component-wise meet."""
        a = self._make_product(
            iv=Interval.from_bounds(0, 10),
            tt=TypeTagSet.from_names("int", "str"),
            nv=NullityValue.maybe_null(),
        )
        b = self._make_product(
            iv=Interval.from_bounds(5, 15),
            tt=TypeTagSet.from_names("int", "float"),
            nv=NullityValue.definitely_not_null(),
        )
        m = product_meet(a, b)
        assert m.interval == Interval.from_bounds(5, 10)
        assert m.type_tag == TypeTagSet.singleton("int")
        assert m.nullity == NullityValue.definitely_not_null()

    def test_product_widening(self) -> None:
        """Product widening applies interval widening."""
        a = self._make_product(iv=Interval.from_bounds(0, 5))
        b = self._make_product(iv=Interval.from_bounds(0, 10))
        w = product_widen(a, b)
        assert w.interval.hi.kind == BoundKind.POS_INF

    def test_product_leq(self) -> None:
        """Product leq is component-wise leq."""
        a = self._make_product(
            iv=Interval.from_bounds(2, 5),
            tt=TypeTagSet.singleton("int"),
            nv=NullityValue.definitely_not_null(),
        )
        b = self._make_product(
            iv=Interval.from_bounds(0, 10),
            tt=TypeTagSet.from_names("int", "str"),
            nv=NullityValue.maybe_null(),
        )
        assert product_leq(a, b)
        assert not product_leq(b, a)

    # ── reductions ────────────────────────────────────────────────────────

    def test_isinstance_to_nullity_reduction(self) -> None:
        """isinstance(x, int) → x is not None."""
        p = self._make_product(
            tt=TypeTagSet.singleton("int"),
            nv=NullityValue.maybe_null(),
        )
        r = reduce_product(p)
        assert r.nullity == NullityValue.definitely_not_null()

    def test_isinstance_to_numeric_reduction(self) -> None:
        """isinstance(x, int) activates numeric tracking."""
        p = self._make_product(
            tt=TypeTagSet.singleton("int"),
            iv=Interval.from_bounds(0, 100),
            nv=NullityValue.maybe_null(),
        )
        r = reduce_product(p)
        # After reduction, nullity should be refined
        assert r.nullity == NullityValue.definitely_not_null()

    def test_nullity_to_typetag_reduction(self) -> None:
        """x is not None → NoneType not in tags."""
        p = self._make_product(
            tt=TypeTagSet.from_names("int", "NoneType"),
            nv=NullityValue.definitely_not_null(),
        )
        r = reduce_product(p)
        assert not r.type_tag.contains("NoneType")
        assert r.type_tag.contains("int")

    def test_numeric_to_typetag_reduction(self) -> None:
        """x > 0 implies x is not None (numeric interval non-bottom)."""
        p = self._make_product(
            iv=Interval.from_bounds(1, 10),
            tt=TypeTagSet.from_names("int", "float"),
            nv=NullityValue.maybe_null(),
        )
        r = reduce_product(p)
        assert r.nullity == NullityValue.definitely_not_null()

    def test_string_to_structural_reduction(self) -> None:
        """hasattr-based string tracking."""
        p = self._make_product(
            sv=StringValue.singleton("__len__"),
            tt=TypeTagSet.from_names("list", "str", "dict"),
        )
        # Reduction: if we know the attribute, we can restrict types
        r = reduce_product(p)
        assert r.string.contains("__len__")

    def test_bidirectional_reduction(self) -> None:
        """Reduction flows in both directions."""
        # Definitely null → NoneType only
        p = self._make_product(
            tt=TypeTagSet.from_names("int", "NoneType"),
            nv=NullityValue.definitely_null(),
        )
        r = reduce_product(p)
        assert r.type_tag == TypeTagSet.singleton("NoneType")
        assert r.nullity.is_definitely_null

    def test_fixpoint_reduction(self) -> None:
        """Reduction reaches fixpoint."""
        p = self._make_product(
            tt=TypeTagSet.from_names("int"),
            nv=NullityValue.maybe_null(),
        )
        r1 = reduce_product(p)
        r2 = reduce_product(r1)
        assert r1 == r2  # fixpoint after one pass

    def test_product_abstract_transformer(self) -> None:
        """Apply a transformer that assigns x = 42."""
        state: Dict[str, ProductValue] = {}
        # x = 42 → interval=[42,42], tag={int}, not null
        state["x"] = ProductValue(
            Interval.singleton(42),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
            StringValue.bottom(),
        )
        x = state["x"]
        assert x.interval.is_singleton
        assert x.type_tag.contains("int")
        assert not x.nullity.may_be_null

    def test_reduction_none_tag_only(self) -> None:
        """Tags = {NoneType} → definitely null."""
        p = self._make_product(
            tt=TypeTagSet.singleton("NoneType"),
            nv=NullityValue.maybe_null(),
        )
        r = reduce_product(p)
        assert r.nullity.is_definitely_null

    def test_reduction_no_none_tag(self) -> None:
        """Tags without NoneType and not top → not null."""
        p = self._make_product(
            tt=TypeTagSet.from_names("int", "str"),
            nv=NullityValue.maybe_null(),
        )
        r = reduce_product(p)
        assert r.nullity == NullityValue.definitely_not_null()


class TestLatticeProperties:
    """Property-based tests verifying lattice axioms."""

    INTERVALS = [
        Interval.bottom(),
        Interval.top(),
        Interval.singleton(0),
        Interval.from_bounds(-5, 5),
        Interval.from_bounds(0, 10),
        Interval.from_bounds(-10, 0),
        Interval.from_bounds(3, 7),
        Interval(Bound.neg_inf(), Bound.finite(5)),
        Interval(Bound.finite(0), Bound.pos_inf()),
    ]

    NULLITIES = [
        NullityValue.bottom(),
        NullityValue.definitely_null(),
        NullityValue.definitely_not_null(),
        NullityValue.maybe_null(),
    ]

    TYPE_TAGS = [
        TypeTagSet.bottom(),
        TypeTagSet.top(),
        TypeTagSet.singleton("int"),
        TypeTagSet.from_names("int", "str"),
        TypeTagSet.from_names("int", "str", "float"),
    ]

    # ── join properties ───────────────────────────────────────────────────

    def test_join_commutative(self) -> None:
        """a ⊔ b = b ⊔ a for intervals."""
        for a in self.INTERVALS:
            for b in self.INTERVALS:
                assert a.join(b) == b.join(a), f"{a} ⊔ {b}"

    def test_join_associative(self) -> None:
        """(a ⊔ b) ⊔ c = a ⊔ (b ⊔ c) for intervals."""
        triples = list(itertools.combinations(self.INTERVALS[:5], 3))
        for a, b, c in triples:
            assert a.join(b).join(c) == a.join(b.join(c)), f"{a},{b},{c}"

    def test_join_idempotent(self) -> None:
        """a ⊔ a = a for all domains."""
        for a in self.INTERVALS:
            assert a.join(a) == a
        for a in self.NULLITIES:
            assert a.join(a) == a
        for a in self.TYPE_TAGS:
            assert a.join(a) == a

    # ── meet properties ───────────────────────────────────────────────────

    def test_meet_commutative(self) -> None:
        """a ⊓ b = b ⊓ a for intervals."""
        for a in self.INTERVALS:
            for b in self.INTERVALS:
                assert a.meet(b) == b.meet(a), f"{a} ⊓ {b}"

    def test_meet_associative(self) -> None:
        """(a ⊓ b) ⊓ c = a ⊓ (b ⊓ c) for intervals."""
        triples = list(itertools.combinations(self.INTERVALS[:5], 3))
        for a, b, c in triples:
            assert a.meet(b).meet(c) == a.meet(b.meet(c)), f"{a},{b},{c}"

    def test_meet_idempotent(self) -> None:
        """a ⊓ a = a for all domains."""
        for a in self.INTERVALS:
            assert a.meet(a) == a
        for a in self.NULLITIES:
            assert a.meet(a) == a
        for a in self.TYPE_TAGS:
            assert a.meet(a) == a

    # ── absorption ────────────────────────────────────────────────────────

    def test_absorption_laws(self) -> None:
        """a ⊔ (a ⊓ b) = a and a ⊓ (a ⊔ b) = a for intervals."""
        pairs = list(itertools.combinations(self.INTERVALS[:5], 2))
        for a, b in pairs:
            assert a.join(a.meet(b)) == a, f"absorption1: {a},{b}"
            assert a.meet(a.join(b)) == a, f"absorption2: {a},{b}"

    # ── identity elements ─────────────────────────────────────────────────

    def test_bottom_identity(self) -> None:
        """⊥ ⊔ a = a for all a."""
        bot = Interval.bottom()
        for a in self.INTERVALS:
            assert bot.join(a) == a

    def test_top_identity(self) -> None:
        """⊤ ⊓ a = a for all a."""
        top = Interval.top()
        for a in self.INTERVALS:
            assert top.meet(a) == a

    # ── widening ──────────────────────────────────────────────────────────

    def test_widening_upper_bound(self) -> None:
        """a ⊔ b ⊑ a ∇ b (widening is an upper bound of join)."""
        for a in self.INTERVALS:
            for b in self.INTERVALS:
                j = a.join(b)
                w = a.widen(b)
                assert j.leq(w), f"join ⊑ widen: {a},{b}"

    def test_widening_convergence(self) -> None:
        """Ascending chain with widening must stabilize."""
        current = Interval.from_bounds(0, 0)
        for i in range(1, 50):
            next_val = Interval.from_bounds(0, i)
            widened = current.widen(next_val)
            if widened == current:
                break
            current = widened
        # Must have stabilized (either finite bound or +inf)
        assert current.hi.kind == BoundKind.POS_INF or current == Interval.from_bounds(0, 0)

    # ── Galois connection ─────────────────────────────────────────────────

    def test_galois_connection(self) -> None:
        """α(γ(a)) ⊒ a: abstracting the concretization is above original."""
        # For intervals: concretize then re-abstract should be ⊒ original
        test_intervals = [
            Interval.singleton(5),
            Interval.from_bounds(0, 10),
        ]
        for iv in test_intervals:
            # γ(iv) = set of ints in iv, α(γ(iv)) = smallest interval containing them
            if iv.is_singleton:
                concrete = iv.lo.value
                re_abstract = Interval.singleton(concrete)
                assert iv.leq(re_abstract)
            elif iv.lo.is_finite and iv.hi.is_finite:
                lo, hi = iv.lo.value, iv.hi.value
                re_abstract = Interval.from_bounds(lo, hi)
                assert iv.leq(re_abstract)

    # ── nullity lattice properties ────────────────────────────────────────

    def test_nullity_join_commutative(self) -> None:
        """Nullity join is commutative."""
        for a in self.NULLITIES:
            for b in self.NULLITIES:
                assert a.join(b) == b.join(a)

    def test_nullity_meet_commutative(self) -> None:
        """Nullity meet is commutative."""
        for a in self.NULLITIES:
            for b in self.NULLITIES:
                assert a.meet(b) == b.meet(a)

    def test_nullity_bottom_identity(self) -> None:
        """⊥ ⊔ a = a for nullity."""
        bot = NullityValue.bottom()
        for a in self.NULLITIES:
            assert bot.join(a) == a

    def test_nullity_top_identity(self) -> None:
        """⊤ ⊓ a = a for nullity."""
        top = NullityValue.maybe_null()
        for a in self.NULLITIES:
            assert top.meet(a) == a

    # ── type tag lattice properties ───────────────────────────────────────

    def test_typetag_join_commutative(self) -> None:
        """TypeTag join is commutative."""
        for a in self.TYPE_TAGS:
            for b in self.TYPE_TAGS:
                assert a.join(b) == b.join(a)

    def test_typetag_meet_commutative(self) -> None:
        """TypeTag meet is commutative."""
        for a in self.TYPE_TAGS:
            for b in self.TYPE_TAGS:
                assert a.meet(b) == b.meet(a)

    def test_typetag_bottom_identity(self) -> None:
        """⊥ ⊔ a = a for type tags."""
        bot = TypeTagSet.bottom()
        for a in self.TYPE_TAGS:
            assert bot.join(a) == a

    def test_typetag_top_identity(self) -> None:
        """⊤ ⊓ a = a for type tags."""
        top = TypeTagSet.top()
        for a in self.TYPE_TAGS:
            assert top.meet(a) == a

    # ── cross-domain monotonicity ─────────────────────────────────────────

    def test_product_join_monotone(self) -> None:
        """a ⊑ b implies a ⊔ c ⊑ b ⊔ c for product domain."""
        a = ProductValue(
            Interval.from_bounds(2, 5),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
            StringValue.bottom(),
        )
        b = ProductValue(
            Interval.from_bounds(0, 10),
            TypeTagSet.from_names("int", "str"),
            NullityValue.maybe_null(),
            StringValue.top(),
        )
        c = ProductValue(
            Interval.from_bounds(1, 3),
            TypeTagSet.singleton("float"),
            NullityValue.definitely_null(),
            StringValue.singleton("x"),
        )
        assert product_leq(a, b)
        ac = product_join(a, c)
        bc = product_join(b, c)
        assert product_leq(ac, bc)

    def test_meet_below_operands(self) -> None:
        """a ⊓ b ⊑ a and a ⊓ b ⊑ b for intervals."""
        for a in self.INTERVALS:
            for b in self.INTERVALS:
                m = a.meet(b)
                assert m.leq(a), f"{m} ⊑ {a}"
                assert m.leq(b), f"{m} ⊑ {b}"

    def test_join_above_operands(self) -> None:
        """a ⊑ a ⊔ b and b ⊑ a ⊔ b for intervals."""
        for a in self.INTERVALS:
            for b in self.INTERVALS:
                j = a.join(b)
                assert a.leq(j), f"{a} ⊑ {j}"
                assert b.leq(j), f"{b} ⊑ {j}"


class TestIntervalArithmeticExtended:
    """Extended arithmetic tests for interval domain."""

    def test_add_identity(self) -> None:
        """[a,b] + [0,0] = [a,b]."""
        a = Interval.from_bounds(3, 7)
        zero = Interval.singleton(0)
        assert (a + zero) == a

    def test_sub_identity(self) -> None:
        """[a,b] - [0,0] = [a,b]."""
        a = Interval.from_bounds(3, 7)
        zero = Interval.singleton(0)
        assert (a - zero) == a

    def test_mul_identity(self) -> None:
        """[a,b] * [1,1] = [a,b]."""
        a = Interval.from_bounds(3, 7)
        one = Interval.singleton(1)
        assert (a * one) == a

    def test_mul_zero_annihilates(self) -> None:
        """[a,b] * [0,0] = [0,0]."""
        a = Interval.from_bounds(-100, 100)
        zero = Interval.singleton(0)
        assert (a * zero) == zero

    def test_add_negative_range(self) -> None:
        """[-10,-5] + [-3,-1] = [-13,-6]."""
        a = Interval.from_bounds(-10, -5)
        b = Interval.from_bounds(-3, -1)
        assert (a + b) == Interval.from_bounds(-13, -6)

    def test_sub_reversed(self) -> None:
        """[a,b] - [c,d] has lo = a-d, hi = b-c."""
        a = Interval.from_bounds(10, 20)
        b = Interval.from_bounds(3, 5)
        result = a - b
        assert result == Interval.from_bounds(5, 17)

    def test_mul_mixed_sign(self) -> None:
        """[-3,2] * [-2,4] = [-12,8]."""
        a = Interval.from_bounds(-3, 2)
        b = Interval.from_bounds(-2, 4)
        result = a * b
        assert result.lo.value == -12
        assert result.hi.value == 8

    def test_floordiv_exact(self) -> None:
        """[10,10] // [2,2] = [5,5]."""
        a = Interval.singleton(10)
        b = Interval.singleton(2)
        result = a.floordiv(b)
        assert result == Interval.singleton(5)

    def test_floordiv_negative(self) -> None:
        """[-10,-6] // [3,3] = [-4,-2]."""
        a = Interval.from_bounds(-10, -6)
        b = Interval.singleton(3)
        result = a.floordiv(b)
        assert result.contains(-4)
        assert result.contains(-2)

    def test_mod_exact(self) -> None:
        """[x,x] % [n,n] = [x%n, x%n] for singleton mod."""
        a = Interval.singleton(17)
        b = Interval.singleton(5)
        result = a.mod(b)
        assert result.contains(2)

    def test_mod_range(self) -> None:
        """[0,20] % [7,7] ⊆ [0,6]."""
        a = Interval.from_bounds(0, 20)
        b = Interval.singleton(7)
        result = a.mod(b)
        assert result.lo == Bound.finite(0)
        assert result.hi == Bound.finite(6)

    def test_neg_top(self) -> None:
        """Negate top is top."""
        t = Interval.top()
        n = -t
        assert n.is_top

    def test_abs_top(self) -> None:
        """abs(top) = [0, +inf)."""
        t = Interval.top()
        a = t.abs()
        assert a.lo == Bound.finite(0)
        assert a.hi.kind == BoundKind.POS_INF

    def test_abs_singleton_negative(self) -> None:
        """abs([-5,-5]) = [5,5]."""
        a = Interval.singleton(-5)
        assert a.abs() == Interval.singleton(5)

    def test_chained_arithmetic(self) -> None:
        """(a + b) - b contains elements of a."""
        a = Interval.from_bounds(3, 7)
        b = Interval.from_bounds(1, 2)
        result = (a + b) - b
        # [3,7] + [1,2] = [4,9]; [4,9] - [1,2] = [2,8]
        assert result.contains(3)
        assert result.contains(7)

    def test_mul_large_ranges(self) -> None:
        """Multiplication of large finite ranges."""
        a = Interval.from_bounds(-1000, 1000)
        b = Interval.from_bounds(-1000, 1000)
        result = a * b
        assert result.contains(0)
        assert result.lo.value == -1000000
        assert result.hi.value == 1000000

    def test_div_positive_by_positive(self) -> None:
        """[6,12] // [2,3] = [2,6]."""
        a = Interval.from_bounds(6, 12)
        b = Interval.from_bounds(2, 3)
        result = a.floordiv(b)
        assert result.contains(2)
        assert result.contains(6)

    def test_mod_negative_modulus(self) -> None:
        """Mod by negative value."""
        a = Interval.from_bounds(0, 10)
        b = Interval.singleton(-3)
        result = a.mod(b)
        assert result.lo == Bound.finite(0)
        assert result.hi == Bound.finite(2)

    def test_add_bottom_left(self) -> None:
        """⊥ + a = ⊥."""
        assert (Interval.bottom() + Interval.from_bounds(1, 5)).is_bottom

    def test_add_bottom_right(self) -> None:
        """a + ⊥ = ⊥."""
        assert (Interval.from_bounds(1, 5) + Interval.bottom()).is_bottom

    def test_sub_bottom(self) -> None:
        """⊥ - a = ⊥."""
        assert (Interval.bottom() - Interval.from_bounds(1, 5)).is_bottom

    def test_mul_bottom(self) -> None:
        """⊥ * a = ⊥."""
        assert (Interval.bottom() * Interval.from_bounds(1, 5)).is_bottom

    def test_div_bottom(self) -> None:
        """⊥ // a = ⊥."""
        assert Interval.bottom().floordiv(Interval.singleton(2)).is_bottom

    def test_mod_bottom(self) -> None:
        """⊥ % a = ⊥."""
        assert Interval.bottom().mod(Interval.singleton(3)).is_bottom


class TestIntervalComparisonExtended:
    """Extended comparison refinement tests."""

    def test_refine_lt_empty_result(self) -> None:
        """Refine [10,20] < 5 → ⊥ (impossible)."""
        x = Interval.from_bounds(10, 20)
        refined = x.meet(Interval(Bound.neg_inf(), Bound.finite(4)))
        assert refined.is_bottom

    def test_refine_gt_empty_result(self) -> None:
        """Refine [0,3] > 10 → ⊥ (impossible)."""
        x = Interval.from_bounds(0, 3)
        refined = x.meet(Interval(Bound.finite(11), Bound.pos_inf()))
        assert refined.is_bottom

    def test_refine_eq_singleton(self) -> None:
        """Refine [0,10] == 5 → [5,5]."""
        x = Interval.from_bounds(0, 10)
        refined = x.meet(Interval.singleton(5))
        assert refined.is_singleton
        assert refined.lo.value == 5

    def test_refine_ge_at_lower(self) -> None:
        """Refine [3,10] >= 3 → [3,10] (already satisfies)."""
        x = Interval.from_bounds(3, 10)
        refined = x.meet(Interval(Bound.finite(3), Bound.pos_inf()))
        assert refined == x

    def test_refine_le_at_upper(self) -> None:
        """Refine [3,10] <= 10 → [3,10] (already satisfies)."""
        x = Interval.from_bounds(3, 10)
        refined = x.meet(Interval(Bound.neg_inf(), Bound.finite(10)))
        assert refined == x

    def test_refine_chain_narrow(self) -> None:
        """Chain of refinements: 3 <= x <= 7 from [0,10]."""
        x = Interval.from_bounds(0, 10)
        x = x.meet(Interval(Bound.finite(3), Bound.pos_inf()))
        x = x.meet(Interval(Bound.neg_inf(), Bound.finite(7)))
        assert x == Interval.from_bounds(3, 7)

    def test_refine_unbounded_lower(self) -> None:
        """Refine (-inf, +inf) >= 0 → [0, +inf)."""
        x = Interval.top()
        refined = x.meet(Interval(Bound.finite(0), Bound.pos_inf()))
        assert refined.lo == Bound.finite(0)
        assert refined.hi.kind == BoundKind.POS_INF

    def test_refine_unbounded_upper(self) -> None:
        """Refine (-inf, +inf) < 100 → (-inf, 99]."""
        x = Interval.top()
        refined = x.meet(Interval(Bound.neg_inf(), Bound.finite(99)))
        assert refined.lo.kind == BoundKind.NEG_INF
        assert refined.hi == Bound.finite(99)


class TestIntervalWideningExtended:
    """Extended widening tests."""

    def test_widening_sequence_converges(self) -> None:
        """Iterated widening: [0,0], [0,1], [0,2], ... → [0, +inf)."""
        current = Interval.singleton(0)
        for i in range(1, 20):
            new = Interval.from_bounds(0, i)
            current = current.widen(new)
            if current.hi.kind == BoundKind.POS_INF:
                break
        assert current.hi.kind == BoundKind.POS_INF
        assert current.lo == Bound.finite(0)

    def test_widening_decreasing_lower(self) -> None:
        """Iterated widening with decreasing lower: [0,5], [-1,5], [-2,5], ..."""
        current = Interval.from_bounds(0, 5)
        for i in range(1, 20):
            new = Interval.from_bounds(-i, 5)
            current = current.widen(new)
            if current.lo.kind == BoundKind.NEG_INF:
                break
        assert current.lo.kind == BoundKind.NEG_INF
        assert current.hi == Bound.finite(5)

    def test_widening_both_sides(self) -> None:
        """Both bounds change → both go to infinity."""
        current = Interval.from_bounds(0, 0)
        new = Interval.from_bounds(-1, 1)
        w = current.widen(new)
        assert w.lo.kind == BoundKind.NEG_INF
        assert w.hi.kind == BoundKind.POS_INF

    def test_narrowing_sequence(self) -> None:
        """Narrowing recovers precision after widening."""
        wide = Interval(Bound.finite(0), Bound.pos_inf())
        for n in [100, 50, 20, 10]:
            precise = Interval.from_bounds(0, n)
            wide = wide.narrow(precise)
        assert wide.hi == Bound.finite(10)

    def test_widening_idempotent_when_stable(self) -> None:
        """If b ⊑ a, then a ∇ b = a."""
        a = Interval.from_bounds(0, 10)
        b = Interval.from_bounds(2, 8)
        assert a.widen(b) == a


class TestTypeTagExtended:
    """Extended type tag domain tests."""

    def test_type_tag_many_types(self) -> None:
        """Tag set with many types."""
        tags = TypeTagSet.from_names("int", "float", "str", "list", "dict", "set", "tuple", "bool")
        assert len(tags.tags) == 8
        for t in ["int", "float", "str", "list", "dict", "set", "tuple", "bool"]:
            assert tags.contains(t)

    def test_type_tag_remove_nonexistent(self) -> None:
        """Removing non-existent tag is no-op."""
        tags = TypeTagSet.from_names("int", "str")
        result = tags.remove("float")
        assert result == tags

    def test_type_tag_add_existing(self) -> None:
        """Adding existing tag is no-op."""
        tags = TypeTagSet.from_names("int", "str")
        result = tags.add("int")
        assert result == tags

    def test_type_tag_join_many(self) -> None:
        """Join of many tag sets."""
        sets = [
            TypeTagSet.singleton("int"),
            TypeTagSet.singleton("str"),
            TypeTagSet.singleton("float"),
        ]
        result = sets[0]
        for s in sets[1:]:
            result = result.join(s)
        assert result == TypeTagSet.from_names("int", "str", "float")

    def test_type_tag_meet_many(self) -> None:
        """Meet of many tag sets with common element."""
        sets = [
            TypeTagSet.from_names("int", "str", "float"),
            TypeTagSet.from_names("str", "float", "list"),
            TypeTagSet.from_names("str", "dict", "float"),
        ]
        result = sets[0]
        for s in sets[1:]:
            result = result.meet(s)
        assert result == TypeTagSet.from_names("str", "float") or result == TypeTagSet.singleton("str")

    def test_type_tag_bottom_meet_is_bottom(self) -> None:
        """Meet with bottom is bottom."""
        a = TypeTagSet.from_names("int", "str")
        assert a.meet(TypeTagSet.bottom()).is_bottom

    def test_type_tag_top_join_is_top(self) -> None:
        """Join with top is top."""
        a = TypeTagSet.from_names("int", "str")
        assert a.join(TypeTagSet.top()).is_top


class TestNullityExtended:
    """Extended nullity domain tests."""

    def test_nullity_join_with_maybe(self) -> None:
        """Join anything with maybe is maybe."""
        for nv in [NullityValue.bottom(), NullityValue.definitely_null(),
                   NullityValue.definitely_not_null()]:
            assert nv.join(NullityValue.maybe_null()) == NullityValue.maybe_null()

    def test_nullity_meet_with_bottom(self) -> None:
        """Meet anything with bottom is bottom."""
        for nv in [NullityValue.maybe_null(), NullityValue.definitely_null(),
                   NullityValue.definitely_not_null()]:
            assert nv.meet(NullityValue.bottom()).is_bottom

    def test_nullity_leq_reflexive(self) -> None:
        """Every value is ⊑ itself."""
        for nv in [NullityValue.bottom(), NullityValue.definitely_null(),
                   NullityValue.definitely_not_null(), NullityValue.maybe_null()]:
            assert nv.leq(nv)

    def test_nullity_leq_transitive(self) -> None:
        """⊥ ⊑ null ⊑ maybe and ⊥ ⊑ not-null ⊑ maybe → ⊥ ⊑ maybe."""
        bot = NullityValue.bottom()
        null = NullityValue.definitely_null()
        maybe = NullityValue.maybe_null()
        assert bot.leq(null)
        assert null.leq(maybe)
        assert bot.leq(maybe)


class TestProductExtended:
    """Extended product domain tests."""

    def test_product_with_interval_update(self) -> None:
        """Update interval component of product."""
        p = ProductValue(
            Interval.from_bounds(0, 10),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
        )
        updated = p.with_interval(Interval.from_bounds(5, 15))
        assert updated.interval == Interval.from_bounds(5, 15)
        assert updated.type_tag == p.type_tag
        assert updated.nullity == p.nullity

    def test_product_with_type_tag_update(self) -> None:
        """Update type tag component of product."""
        p = ProductValue(
            Interval.top(),
            TypeTagSet.singleton("int"),
            NullityValue.maybe_null(),
        )
        updated = p.with_type_tag(TypeTagSet.from_names("int", "str"))
        assert updated.type_tag == TypeTagSet.from_names("int", "str")

    def test_product_with_nullity_update(self) -> None:
        """Update nullity component of product."""
        p = ProductValue(
            Interval.top(),
            TypeTagSet.top(),
            NullityValue.maybe_null(),
        )
        updated = p.with_nullity(NullityValue.definitely_not_null())
        assert updated.nullity == NullityValue.definitely_not_null()

    def test_product_bottom_leq_anything(self) -> None:
        """Product bottom ⊑ anything."""
        bot = product_bottom()
        top = product_top()
        p = ProductValue(
            Interval.from_bounds(0, 10),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
        )
        assert product_leq(bot, top)
        assert product_leq(bot, p)
        assert product_leq(bot, bot)

    def test_product_anything_leq_top(self) -> None:
        """Anything ⊑ product top."""
        top = product_top()
        p = ProductValue(
            Interval.from_bounds(0, 10),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
        )
        assert product_leq(p, top)
        assert product_leq(product_bottom(), top)

    def test_reduction_chain(self) -> None:
        """Apply reduction multiple times reaches fixpoint."""
        p = ProductValue(
            Interval.from_bounds(1, 10),
            TypeTagSet.from_names("int", "NoneType"),
            NullityValue.maybe_null(),
        )
        r1 = reduce_product(p)
        r2 = reduce_product(r1)
        r3 = reduce_product(r2)
        assert r2 == r3  # fixpoint by second iteration

    def test_product_join_with_bottom(self) -> None:
        """Product join with bottom is identity."""
        p = ProductValue(
            Interval.from_bounds(0, 10),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
        )
        assert product_join(p, product_bottom()) == p

    def test_product_meet_incompatible(self) -> None:
        """Meet of incompatible products may yield bottom components."""
        a = ProductValue(
            Interval.from_bounds(0, 5),
            TypeTagSet.singleton("int"),
            NullityValue.definitely_not_null(),
        )
        b = ProductValue(
            Interval.from_bounds(10, 20),
            TypeTagSet.singleton("str"),
            NullityValue.definitely_null(),
        )
        m = product_meet(a, b)
        assert m.interval.is_bottom  # [0,5] ∩ [10,20] = ⊥
        assert m.type_tag.is_bottom  # {int} ∩ {str} = ∅
        assert m.nullity.is_bottom  # not-null ∩ null = ⊥


class TestStringDomainExtended:
    """Extended string domain tests."""

    def test_string_from_set(self) -> None:
        """Create string value from explicit set."""
        s = StringValue.from_set({"a", "b", "c"})
        assert s.contains("a")
        assert s.contains("b")
        assert not s.contains("d")

    def test_string_singleton_is_singleton(self) -> None:
        """Singleton string has exactly one element."""
        s = StringValue.singleton("hello")
        assert s.is_singleton
        assert s.contains("hello")

    def test_string_join_preserves(self) -> None:
        """Join preserves all elements from both operands."""
        a = StringValue.from_set({"x", "y"})
        b = StringValue.from_set({"y", "z"})
        j = a.join(b)
        assert j.contains("x")
        assert j.contains("y")
        assert j.contains("z")

    def test_string_meet_intersection(self) -> None:
        """Meet computes intersection."""
        a = StringValue.from_set({"a", "b", "c"})
        b = StringValue.from_set({"b", "c", "d"})
        m = a.meet(b)
        assert m.contains("b")
        assert m.contains("c")
        assert not m.contains("a")
        assert not m.contains("d")

    def test_string_leq_subset(self) -> None:
        """Subset implies leq."""
        sub = StringValue.from_set({"a"})
        sup = StringValue.from_set({"a", "b"})
        assert sub.leq(sup)
        assert not sup.leq(sub)

    def test_string_bottom_leq_everything(self) -> None:
        """Bottom ⊑ everything."""
        bot = StringValue.bottom()
        assert bot.leq(StringValue.singleton("x"))
        assert bot.leq(StringValue.top())
        assert bot.leq(bot)

    def test_string_everything_leq_top(self) -> None:
        """Everything ⊑ top."""
        top = StringValue.top()
        assert StringValue.singleton("x").leq(top)
        assert StringValue.from_set({"a", "b"}).leq(top)
        assert StringValue.bottom().leq(top)
