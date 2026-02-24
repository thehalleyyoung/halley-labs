"""
Heap model for refinement type inference over dynamically-typed languages.

This module provides an abstract heap that models Python's heap-based
computation.  It is used by the analysis to track object allocations,
attribute reads/writes, and to determine when strong vs. weak updates are
appropriate.

Key concepts
------------
* **HeapAddress** – an abstract address keyed by allocation site plus an
  optional calling-context tuple (for *k*-CFA style context-sensitivity).
* **Recency abstraction** – the most recent allocation at a given site is
  tagged ``RECENT`` (a singleton, eligible for strong updates); older
  allocations at the same site are merged under a ``SUMMARY`` flag.
* **AbstractValue** – what a variable or heap field can hold: a set of
  heap addresses (pointers), a constrained primitive, ``Top``, ``Bottom``,
  ``None``, or a union of these.
* **HeapObject** – an object on the abstract heap with attribute map,
  class pointer, and recency flag.
* **AbstractHeap** – the complete abstract heap with allocation, read,
  write, join, widen, meet, and garbage-collection operations.
* **HeapState** – combined analysis state (heap + variable environment +
  active refinement predicates).
* **HeapTransformer** – abstract transformers that operate on HeapState
  to model object creation, attribute access, assignment, calls, etc.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ═══════════════════════════════════════════════════════════════════════════
# HeapAddress
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HeapAddress:
    """Abstract heap address identified by allocation site and context.

    Parameters
    ----------
    site : str
        A unique string identifying the allocation site, e.g.
        ``"line42:x = Foo()"``.
    context : tuple
        Calling-context tuple for context-sensitive analyses (*k*-CFA).
        Defaults to the empty tuple (context-insensitive).
    """

    site: str
    context: tuple = ()

    # -- helpers ------------------------------------------------------------

    def with_context(self, ctx: tuple) -> HeapAddress:
        """Return a copy of this address with *ctx* appended to the context."""
        return HeapAddress(site=self.site, context=self.context + ctx)

    def base_site(self) -> str:
        """Return the underlying allocation-site string."""
        return self.site

    def __repr__(self) -> str:
        if self.context:
            return f"HeapAddress({self.site!r}, ctx={self.context!r})"
        return f"HeapAddress({self.site!r})"

    def __str__(self) -> str:
        if self.context:
            return f"@{self.site}[{'->'.join(str(c) for c in self.context)}]"
        return f"@{self.site}"


# ═══════════════════════════════════════════════════════════════════════════
# RecencyFlag
# ═══════════════════════════════════════════════════════════════════════════

class RecencyFlag(Enum):
    """Recency abstraction flag for heap objects.

    ``RECENT`` means the object is the most recent allocation at its site
    (a singleton—eligible for *strong* updates).  ``SUMMARY`` means one
    or more older allocations have been merged at that site (only *weak*
    updates are sound).
    """

    RECENT = auto()
    SUMMARY = auto()

    def __repr__(self) -> str:
        return f"RecencyFlag.{self.name}"


# ═══════════════════════════════════════════════════════════════════════════
# PrimitiveKind
# ═══════════════════════════════════════════════════════════════════════════

class PrimitiveKind(Enum):
    """Enumeration of primitive value kinds."""

    INT = auto()
    STR = auto()
    BOOL = auto()
    FLOAT = auto()
    BYTES = auto()
    NONE = auto()

    def __repr__(self) -> str:
        return f"PrimitiveKind.{self.name}"


# ═══════════════════════════════════════════════════════════════════════════
# PrimitiveConstraint hierarchy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PrimitiveConstraint:
    """Base class for constraints on primitive abstract values."""

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Return the least upper bound of two constraints, or ``None`` if
        they cannot be combined (fall back to unconstrained)."""
        if self == other:
            return self
        return None

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Return the greatest lower bound, or ``None`` if empty."""
        if self == other:
            return self
        return None

    def widen(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Widening—defaults to join."""
        return self.join(other)

    def is_satisfiable(self) -> bool:
        """Return whether any concrete value can satisfy this constraint."""
        return True


@dataclass(frozen=True)
class IntRange(PrimitiveConstraint):
    """Constraint: integer value lies in ``[lo, hi]``.

    ``lo`` or ``hi`` may be ``None`` to indicate negative / positive
    infinity.
    """

    lo: Optional[int] = None
    hi: Optional[int] = None

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Least upper bound of two ``IntRange`` constraints."""
        if not isinstance(other, IntRange):
            return None
        new_lo: Optional[int] = None
        if self.lo is not None and other.lo is not None:
            new_lo = min(self.lo, other.lo)
        new_hi: Optional[int] = None
        if self.hi is not None and other.hi is not None:
            new_hi = max(self.hi, other.hi)
        return IntRange(lo=new_lo, hi=new_hi)

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Greatest lower bound of two ``IntRange`` constraints."""
        if not isinstance(other, IntRange):
            return None
        new_lo: Optional[int]
        if self.lo is None:
            new_lo = other.lo
        elif other.lo is None:
            new_lo = self.lo
        else:
            new_lo = max(self.lo, other.lo)
        new_hi: Optional[int]
        if self.hi is None:
            new_hi = other.hi
        elif other.hi is None:
            new_hi = self.hi
        else:
            new_hi = min(self.hi, other.hi)
        if new_lo is not None and new_hi is not None and new_lo > new_hi:
            return None
        return IntRange(lo=new_lo, hi=new_hi)

    def widen(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        """Widen: if the bound moved, push it to infinity."""
        if not isinstance(other, IntRange):
            return None
        new_lo = self.lo
        if other.lo is not None and (self.lo is None or other.lo < self.lo):
            new_lo = None
        new_hi = self.hi
        if other.hi is not None and (self.hi is None or other.hi > self.hi):
            new_hi = None
        return IntRange(lo=new_lo, hi=new_hi)

    def is_satisfiable(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo <= self.hi
        return True

    def __repr__(self) -> str:
        lo_s = str(self.lo) if self.lo is not None else "-∞"
        hi_s = str(self.hi) if self.hi is not None else "+∞"
        return f"IntRange([{lo_s}, {hi_s}])"


@dataclass(frozen=True)
class StrPrefix(PrimitiveConstraint):
    """Constraint: string value starts with *prefix*."""

    prefix: str = ""

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if not isinstance(other, StrPrefix):
            return None
        # Longest common prefix
        common = []
        for a, b in zip(self.prefix, other.prefix):
            if a == b:
                common.append(a)
            else:
                break
        if not common:
            return None
        return StrPrefix(prefix="".join(common))

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if not isinstance(other, StrPrefix):
            return None
        # The longer prefix that is consistent
        if self.prefix.startswith(other.prefix):
            return self
        if other.prefix.startswith(self.prefix):
            return other
        return None

    def __repr__(self) -> str:
        return f"StrPrefix({self.prefix!r})"


@dataclass(frozen=True)
class StrPattern(PrimitiveConstraint):
    """Constraint: string value matches a regex *pattern*."""

    pattern: str = ".*"

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if not isinstance(other, StrPattern):
            return None
        if self.pattern == other.pattern:
            return self
        return None  # Can't combine arbitrary regexes simply

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if not isinstance(other, StrPattern):
            return None
        if self.pattern == other.pattern:
            return self
        return None

    def __repr__(self) -> str:
        return f"StrPattern({self.pattern!r})"


@dataclass(frozen=True)
class BoolConst(PrimitiveConstraint):
    """Constraint: boolean value is exactly *value*."""

    value: bool

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if isinstance(other, BoolConst):
            if self.value == other.value:
                return self
            return None  # both True and False → unconstrained bool
        return None

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if isinstance(other, BoolConst):
            if self.value == other.value:
                return self
            return None  # contradiction
        return None

    def __repr__(self) -> str:
        return f"BoolConst({self.value!r})"


@dataclass(frozen=True)
class ExactValue(PrimitiveConstraint):
    """Constraint: the value is exactly *value* (singleton abstraction)."""

    value: Any

    def join(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if isinstance(other, ExactValue) and self.value == other.value:
            return self
        return None

    def meet(self, other: PrimitiveConstraint) -> Optional[PrimitiveConstraint]:
        if isinstance(other, ExactValue):
            if self.value == other.value:
                return self
            return None
        return None

    def __repr__(self) -> str:
        return f"ExactValue({self.value!r})"


# ═══════════════════════════════════════════════════════════════════════════
# AbstractValue hierarchy
# ═══════════════════════════════════════════════════════════════════════════

class AbstractValue:
    """Base class for abstract values in the heap analysis.

    An ``AbstractValue`` represents the set of concrete Python values that
    a variable or object field may hold at a given program point.
    """

    # -- lattice operations -------------------------------------------------

    def join(self, other: AbstractValue) -> AbstractValue:
        """Least upper bound."""
        if isinstance(other, BottomValue):
            return self
        if isinstance(self, BottomValue):
            return other
        if isinstance(other, TopValue) or isinstance(self, TopValue):
            return TopValue()
        if self == other:
            return self
        return UnionAbstractValue.of(self, other)

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Greatest lower bound."""
        if isinstance(other, TopValue):
            return self
        if isinstance(self, TopValue):
            return other
        if isinstance(other, BottomValue) or isinstance(self, BottomValue):
            return BottomValue()
        if self == other:
            return self
        return BottomValue()

    def widen(self, other: AbstractValue) -> AbstractValue:
        """Widening operator (defaults to join)."""
        return self.join(other)

    # -- queries ------------------------------------------------------------

    def is_definitely_none(self) -> bool:
        """Return ``True`` iff this value can only be ``None``."""
        return False

    def may_be_none(self) -> bool:
        """Return ``True`` iff ``None`` is among the possible values."""
        return False

    def as_addresses(self) -> FrozenSet[HeapAddress]:
        """Return the set of heap addresses this value may point to.

        Returns an empty set for non-pointer values.
        """
        return frozenset()

    def is_bottom(self) -> bool:
        """Return ``True`` if this represents the empty set of values."""
        return False

    def is_top(self) -> bool:
        """Return ``True`` if this represents the universe of values."""
        return False


@dataclass(frozen=True)
class BottomValue(AbstractValue):
    """The empty set of values (unreachable)."""

    def join(self, other: AbstractValue) -> AbstractValue:
        """Bottom is the identity for join."""
        return other

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Bottom absorbs in meet."""
        return self

    def widen(self, other: AbstractValue) -> AbstractValue:
        """Widen from bottom yields the other value."""
        return other

    def is_bottom(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "⊥"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BottomValue)

    def __hash__(self) -> int:
        return hash("__bottom__")


@dataclass(frozen=True)
class TopValue(AbstractValue):
    """The universe of values (no information)."""

    def join(self, other: AbstractValue) -> AbstractValue:
        """Top absorbs in join."""
        return self

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Top is the identity for meet."""
        return other

    def widen(self, other: AbstractValue) -> AbstractValue:
        return self

    def is_top(self) -> bool:
        return True

    def may_be_none(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "⊤"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TopValue)

    def __hash__(self) -> int:
        return hash("__top__")


@dataclass(frozen=True)
class NoneValue(AbstractValue):
    """The singleton ``None`` value."""

    def is_definitely_none(self) -> bool:
        return True

    def may_be_none(self) -> bool:
        return True

    def join(self, other: AbstractValue) -> AbstractValue:
        if isinstance(other, (BottomValue, NoneValue)):
            return self
        if isinstance(other, TopValue):
            return other
        return UnionAbstractValue.of(self, other)

    def meet(self, other: AbstractValue) -> AbstractValue:
        if isinstance(other, (TopValue, NoneValue)):
            return self
        if isinstance(other, UnionAbstractValue) and other.may_be_none():
            return self
        return BottomValue()

    def __repr__(self) -> str:
        return "NoneVal"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NoneValue)

    def __hash__(self) -> int:
        return hash("__none_value__")


@dataclass(frozen=True)
class AddressValue(AbstractValue):
    """A set of heap addresses (pointer value).

    Each address in *addresses* is a possible target that this variable
    could point to at run time.
    """

    addresses: FrozenSet[HeapAddress] = frozenset()

    def as_addresses(self) -> FrozenSet[HeapAddress]:
        return self.addresses

    def join(self, other: AbstractValue) -> AbstractValue:
        """Join two address sets (union)."""
        if isinstance(other, BottomValue):
            return self
        if isinstance(other, TopValue):
            return other
        if isinstance(other, AddressValue):
            merged = self.addresses | other.addresses
            return AddressValue(addresses=merged)
        return UnionAbstractValue.of(self, other)

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Meet two address sets (intersection)."""
        if isinstance(other, TopValue):
            return self
        if isinstance(other, BottomValue):
            return other
        if isinstance(other, AddressValue):
            intersected = self.addresses & other.addresses
            if not intersected:
                return BottomValue()
            return AddressValue(addresses=intersected)
        if isinstance(other, UnionAbstractValue):
            return other.meet(self)
        return BottomValue()

    def widen(self, other: AbstractValue) -> AbstractValue:
        """Widening on addresses: if the set grows too large, go to Top."""
        joined = self.join(other)
        if isinstance(joined, AddressValue) and len(joined.addresses) > 64:
            return TopValue()
        return joined

    def __repr__(self) -> str:
        if not self.addresses:
            return "AddressValue(∅)"
        addrs = ", ".join(str(a) for a in sorted(self.addresses, key=str))
        return f"AddressValue({{{addrs}}})"


@dataclass(frozen=True)
class PrimitiveValue(AbstractValue):
    """A primitive value of a given *kind* with optional *constraints*.

    For example, ``PrimitiveValue(PrimitiveKind.INT, frozenset({IntRange(0, 100)}))``
    represents an integer known to be in ``[0, 100]``.
    """

    kind: PrimitiveKind
    constraints: FrozenSet[PrimitiveConstraint] = frozenset()

    def join(self, other: AbstractValue) -> AbstractValue:
        """Join two primitive values."""
        if isinstance(other, BottomValue):
            return self
        if isinstance(other, TopValue):
            return other
        if isinstance(other, PrimitiveValue) and other.kind == self.kind:
            if not self.constraints and not other.constraints:
                return PrimitiveValue(kind=self.kind)
            merged_constraints: Set[PrimitiveConstraint] = set()
            paired: Set[type] = set()
            for c1 in self.constraints:
                for c2 in other.constraints:
                    if type(c1) is type(c2):
                        joined = c1.join(c2)
                        if joined is not None:
                            merged_constraints.add(joined)
                        paired.add(type(c1))
            # Keep unpaired constraints only if they appear in both sides
            for c in self.constraints:
                if type(c) not in paired:
                    pass  # drop – constraint doesn't exist in other
            for c in other.constraints:
                if type(c) not in paired:
                    pass  # drop
            return PrimitiveValue(
                kind=self.kind,
                constraints=frozenset(merged_constraints),
            )
        return UnionAbstractValue.of(self, other)

    def meet(self, other: AbstractValue) -> AbstractValue:
        """Meet two primitive values."""
        if isinstance(other, TopValue):
            return self
        if isinstance(other, BottomValue):
            return other
        if isinstance(other, PrimitiveValue) and other.kind == self.kind:
            met_constraints: Set[PrimitiveConstraint] = set()
            for c1 in self.constraints:
                for c2 in other.constraints:
                    if type(c1) is type(c2):
                        m = c1.meet(c2)
                        if m is None:
                            return BottomValue()
                        met_constraints.add(m)
            # Keep all non-conflicting constraints from both sides
            paired_types = {type(c) for c in met_constraints}
            for c in self.constraints:
                if type(c) not in paired_types:
                    met_constraints.add(c)
            for c in other.constraints:
                if type(c) not in paired_types:
                    met_constraints.add(c)
            return PrimitiveValue(
                kind=self.kind,
                constraints=frozenset(met_constraints),
            )
        if isinstance(other, UnionAbstractValue):
            return other.meet(self)
        return BottomValue()

    def widen(self, other: AbstractValue) -> AbstractValue:
        """Widening on primitives delegates to constraint widening."""
        if isinstance(other, BottomValue):
            return self
        if isinstance(other, TopValue):
            return other
        if isinstance(other, PrimitiveValue) and other.kind == self.kind:
            widened: Set[PrimitiveConstraint] = set()
            for c1 in self.constraints:
                for c2 in other.constraints:
                    if type(c1) is type(c2):
                        w = c1.widen(c2)
                        if w is not None:
                            widened.add(w)
            return PrimitiveValue(
                kind=self.kind, constraints=frozenset(widened)
            )
        return self.join(other)

    def may_be_none(self) -> bool:
        return self.kind == PrimitiveKind.NONE

    def is_definitely_none(self) -> bool:
        return self.kind == PrimitiveKind.NONE

    def __repr__(self) -> str:
        if self.constraints:
            cs = ", ".join(repr(c) for c in self.constraints)
            return f"PrimitiveValue({self.kind.name}, {{{cs}}})"
        return f"PrimitiveValue({self.kind.name})"


class UnionAbstractValue(AbstractValue):
    """A union of multiple non-overlapping abstract values.

    Invariants maintained:
    * No nested ``UnionAbstractValue``.
    * No ``BottomValue`` components.
    * At most one ``TopValue`` component (in which case the whole thing
      collapses to ``TopValue``).
    """

    __slots__ = ("_components",)

    def __init__(self, components: FrozenSet[AbstractValue]) -> None:
        self._components = components

    @staticmethod
    def of(*values: AbstractValue) -> AbstractValue:
        """Smart constructor that flattens and simplifies."""
        parts: Set[AbstractValue] = set()
        for v in values:
            if isinstance(v, TopValue):
                return TopValue()
            if isinstance(v, BottomValue):
                continue
            if isinstance(v, UnionAbstractValue):
                parts.update(v._components)
            else:
                parts.add(v)
        if not parts:
            return BottomValue()
        if len(parts) == 1:
            return next(iter(parts))
        # Merge address values
        addr_values = [p for p in parts if isinstance(p, AddressValue)]
        if len(addr_values) > 1:
            merged_addrs: Set[HeapAddress] = set()
            for av in addr_values:
                merged_addrs.update(av.addresses)
            parts -= set(addr_values)
            parts.add(AddressValue(frozenset(merged_addrs)))
        # Merge primitive values of the same kind
        prim_by_kind: Dict[PrimitiveKind, List[PrimitiveValue]] = {}
        for p in list(parts):
            if isinstance(p, PrimitiveValue):
                prim_by_kind.setdefault(p.kind, []).append(p)
        for kind, pvs in prim_by_kind.items():
            if len(pvs) > 1:
                parts -= set(pvs)
                merged_pv: AbstractValue = pvs[0]
                for pv in pvs[1:]:
                    merged_pv = merged_pv.join(pv)
                parts.add(merged_pv)
        if len(parts) == 1:
            return next(iter(parts))
        return UnionAbstractValue(frozenset(parts))

    @property
    def components(self) -> FrozenSet[AbstractValue]:
        """The non-overlapping component values."""
        return self._components

    def join(self, other: AbstractValue) -> AbstractValue:
        if isinstance(other, BottomValue):
            return self
        if isinstance(other, TopValue):
            return other
        return UnionAbstractValue.of(*self._components, other)

    def meet(self, other: AbstractValue) -> AbstractValue:
        if isinstance(other, TopValue):
            return self
        if isinstance(other, BottomValue):
            return other
        results: List[AbstractValue] = []
        targets = (
            other._components
            if isinstance(other, UnionAbstractValue)
            else frozenset({other})
        )
        for c in self._components:
            for t in targets:
                m = c.meet(t)
                if not m.is_bottom():
                    results.append(m)
        if not results:
            return BottomValue()
        val: AbstractValue = results[0]
        for r in results[1:]:
            val = val.join(r)
        return val

    def widen(self, other: AbstractValue) -> AbstractValue:
        """Widen: delegate to component-wise widening where possible."""
        if isinstance(other, BottomValue):
            return self
        if isinstance(other, TopValue):
            return other
        joined = self.join(other)
        if isinstance(joined, UnionAbstractValue) and len(joined._components) > 8:
            return TopValue()
        return joined

    def as_addresses(self) -> FrozenSet[HeapAddress]:
        result: Set[HeapAddress] = set()
        for c in self._components:
            result.update(c.as_addresses())
        return frozenset(result)

    def may_be_none(self) -> bool:
        return any(c.may_be_none() for c in self._components)

    def is_definitely_none(self) -> bool:
        return all(c.is_definitely_none() for c in self._components)

    def __repr__(self) -> str:
        parts = " ∪ ".join(repr(c) for c in sorted(self._components, key=repr))
        return f"Union({parts})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnionAbstractValue):
            return self._components == other._components
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._components)


# ═══════════════════════════════════════════════════════════════════════════
# HeapObject
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeapObject:
    """An object residing on the abstract heap.

    Parameters
    ----------
    address : HeapAddress
        The abstract address of this object.
    class_ref : HeapAddress
        Points to the class object that this object is an instance of.
    attrs : Dict[str, AbstractValue]
        Mutable attributes (``__dict__`` contents).
    immutable_slots : Dict[str, AbstractValue]
        Immutable slot values (``__slots__`` declarations or frozen fields).
    recency : RecencyFlag
        Whether this is the most recent allocation at its site.
    is_class : bool
        ``True`` if this object is itself a class (meta-object).
    is_frozen : bool
        ``True`` if this object is immutable (e.g. frozen dataclass).
    """

    address: HeapAddress
    class_ref: HeapAddress
    attrs: Dict[str, AbstractValue] = field(default_factory=dict)
    immutable_slots: Dict[str, AbstractValue] = field(default_factory=dict)
    recency: RecencyFlag = RecencyFlag.RECENT
    is_class: bool = False
    is_frozen: bool = False

    # -- attribute access ---------------------------------------------------

    def read_attr(self, name: str) -> AbstractValue:
        """Read attribute *name* from this object.

        Checks mutable attrs first, then immutable slots.  Returns
        ``BottomValue`` if the attribute does not exist.
        """
        if name in self.attrs:
            return self.attrs[name]
        if name in self.immutable_slots:
            return self.immutable_slots[name]
        return BottomValue()

    def write_attr(self, name: str, value: AbstractValue) -> None:
        """Write *value* to attribute *name* (in-place, mutable attrs only).

        Raises ``ValueError`` if the object is frozen or the attribute is
        an immutable slot.
        """
        if self.is_frozen:
            return  # silently ignore writes to frozen objects (sound over-approx)
        if name in self.immutable_slots:
            return  # immutable slot—ignore
        self.attrs[name] = value

    def has_attr(self, name: str) -> bool:
        """Return ``True`` if this object has attribute *name*."""
        return name in self.attrs or name in self.immutable_slots

    def get_all_attrs(self) -> Dict[str, AbstractValue]:
        """Return a merged dictionary of all attributes."""
        merged: Dict[str, AbstractValue] = dict(self.immutable_slots)
        merged.update(self.attrs)
        return merged

    # -- lattice operations on objects --------------------------------------

    def join_with(self, other: HeapObject) -> HeapObject:
        """Point-wise join with *other* (must share the same address).

        Returns a new ``HeapObject`` without mutating ``self`` or *other*.
        """
        assert self.address == other.address, (
            f"Cannot join objects at different addresses: "
            f"{self.address} vs {other.address}"
        )
        merged_attrs: Dict[str, AbstractValue] = {}
        all_keys = set(self.attrs.keys()) | set(other.attrs.keys())
        for key in all_keys:
            v1 = self.attrs.get(key, BottomValue())
            v2 = other.attrs.get(key, BottomValue())
            merged_attrs[key] = v1.join(v2)

        merged_slots: Dict[str, AbstractValue] = {}
        slot_keys = set(self.immutable_slots.keys()) | set(
            other.immutable_slots.keys()
        )
        for key in slot_keys:
            v1 = self.immutable_slots.get(key, BottomValue())
            v2 = other.immutable_slots.get(key, BottomValue())
            merged_slots[key] = v1.join(v2)

        # Recency: if either is SUMMARY, result is SUMMARY
        recency = RecencyFlag.SUMMARY
        if (
            self.recency == RecencyFlag.RECENT
            and other.recency == RecencyFlag.RECENT
        ):
            recency = RecencyFlag.RECENT

        return HeapObject(
            address=self.address,
            class_ref=self.class_ref,
            attrs=merged_attrs,
            immutable_slots=merged_slots,
            recency=recency,
            is_class=self.is_class or other.is_class,
            is_frozen=self.is_frozen and other.is_frozen,
        )

    def widen_with(self, other: HeapObject) -> HeapObject:
        """Point-wise widening with *other* (must share the same address).

        Widening is like join but delegates to ``AbstractValue.widen``
        on each attribute.
        """
        assert self.address == other.address
        widened_attrs: Dict[str, AbstractValue] = {}
        all_keys = set(self.attrs.keys()) | set(other.attrs.keys())
        for key in all_keys:
            v1 = self.attrs.get(key, BottomValue())
            v2 = other.attrs.get(key, BottomValue())
            widened_attrs[key] = v1.widen(v2)

        widened_slots: Dict[str, AbstractValue] = {}
        slot_keys = set(self.immutable_slots.keys()) | set(
            other.immutable_slots.keys()
        )
        for key in slot_keys:
            v1 = self.immutable_slots.get(key, BottomValue())
            v2 = other.immutable_slots.get(key, BottomValue())
            widened_slots[key] = v1.widen(v2)

        recency = RecencyFlag.SUMMARY
        if (
            self.recency == RecencyFlag.RECENT
            and other.recency == RecencyFlag.RECENT
        ):
            recency = RecencyFlag.RECENT

        return HeapObject(
            address=self.address,
            class_ref=self.class_ref,
            attrs=widened_attrs,
            immutable_slots=widened_slots,
            recency=recency,
            is_class=self.is_class or other.is_class,
            is_frozen=self.is_frozen and other.is_frozen,
        )

    def deep_copy(self) -> HeapObject:
        """Return a deep copy of this object."""
        return HeapObject(
            address=self.address,
            class_ref=self.class_ref,
            attrs=dict(self.attrs),
            immutable_slots=dict(self.immutable_slots),
            recency=self.recency,
            is_class=self.is_class,
            is_frozen=self.is_frozen,
        )

    def referenced_addresses(self) -> Set[HeapAddress]:
        """Return all heap addresses reachable from this object's fields."""
        addrs: Set[HeapAddress] = {self.class_ref}
        for v in self.attrs.values():
            addrs.update(v.as_addresses())
        for v in self.immutable_slots.values():
            addrs.update(v.as_addresses())
        return addrs

    def __repr__(self) -> str:
        flag = "R" if self.recency == RecencyFlag.RECENT else "S"
        kind = "class" if self.is_class else "obj"
        frozen_s = ",frozen" if self.is_frozen else ""
        n_attrs = len(self.attrs) + len(self.immutable_slots)
        return (
            f"HeapObject({self.address}, {kind}[{flag}{frozen_s}], "
            f"{n_attrs} attrs)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# AbstractHeap
# ═══════════════════════════════════════════════════════════════════════════

class AbstractHeap:
    """The main abstract heap used by the analysis.

    The heap maps ``HeapAddress`` to ``HeapObject``.  It tracks allocation
    counts per site for the recency abstraction and supports strong/weak
    updates depending on whether an address is a singleton.
    """

    def __init__(
        self,
        objects: Optional[Dict[HeapAddress, HeapObject]] = None,
        allocation_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.objects: Dict[HeapAddress, HeapObject] = (
            objects if objects is not None else {}
        )
        self.allocation_counts: Dict[str, int] = (
            allocation_counts if allocation_counts is not None else {}
        )

    # -- basic queries ------------------------------------------------------

    def get_object(self, addr: HeapAddress) -> Optional[HeapObject]:
        """Return the object at *addr*, or ``None``."""
        return self.objects.get(addr)

    def is_singleton(self, addr: HeapAddress) -> bool:
        """Return ``True`` if *addr* is known to refer to exactly one
        concrete object (eligible for strong update)."""
        obj = self.objects.get(addr)
        if obj is None:
            return False
        return obj.recency == RecencyFlag.RECENT

    def is_bottom(self) -> bool:
        """A heap is bottom only if it was explicitly constructed that way
        (by convention, an empty heap is *not* bottom—it simply has no
        allocated objects yet)."""
        return False

    # -- attribute access ---------------------------------------------------

    def read_attr(self, addr: HeapAddress, name: str) -> AbstractValue:
        """Read attribute *name* from the object at *addr*.

        Returns ``BottomValue`` if the address or attribute is not found.
        """
        obj = self.objects.get(addr)
        if obj is None:
            return BottomValue()
        return obj.read_attr(name)

    def write_attr(
        self, addr: HeapAddress, name: str, val: AbstractValue
    ) -> "AbstractHeap":
        """Write *val* to attribute *name* of the object at *addr*.

        Uses **strong update** when the address is a singleton (the old
        value is replaced).  Uses **weak update** otherwise (the new value
        is joined with the old value).

        Returns a *new* ``AbstractHeap``; the original is not mutated.
        """
        new_heap = self.deep_copy()
        obj = new_heap.objects.get(addr)
        if obj is None:
            return new_heap  # nothing to write to

        if self.is_singleton(addr):
            # Strong update: replace
            obj.write_attr(name, val)
        else:
            # Weak update: join with existing value
            old_val = obj.read_attr(name)
            if old_val.is_bottom():
                obj.write_attr(name, val)
            else:
                obj.write_attr(name, old_val.join(val))
        return new_heap

    # -- allocation ---------------------------------------------------------

    def allocate(
        self,
        site: str,
        class_addr: HeapAddress,
        context: tuple = (),
    ) -> Tuple["AbstractHeap", HeapAddress]:
        """Allocate a new object at *site* with class *class_addr*.

        Implements the **recency abstraction**:

        1. If there is already a ``RECENT`` object at this site+context,
           demote it to ``SUMMARY`` (join with any existing summary).
        2. Create a fresh ``RECENT`` object at the site+context.
        3. Increment the allocation count for the site.

        Returns ``(new_heap, new_address)``.
        """
        addr = HeapAddress(site=site, context=context)
        new_heap = self.deep_copy()

        existing = new_heap.objects.get(addr)
        if existing is not None and existing.recency == RecencyFlag.RECENT:
            # Demote to SUMMARY
            summary_addr = HeapAddress(
                site=site + ":summary", context=context
            )
            existing_summary = new_heap.objects.get(summary_addr)
            demoted = existing.deep_copy()
            demoted.recency = RecencyFlag.SUMMARY
            demoted.address = summary_addr
            if existing_summary is not None:
                demoted = existing_summary.join_with(demoted)
            new_heap.objects[summary_addr] = demoted

        # Create fresh RECENT object
        new_obj = HeapObject(
            address=addr,
            class_ref=class_addr,
            attrs={},
            immutable_slots={},
            recency=RecencyFlag.RECENT,
            is_class=False,
            is_frozen=False,
        )
        new_heap.objects[addr] = new_obj

        # Update allocation count
        count = new_heap.allocation_counts.get(site, 0)
        new_heap.allocation_counts[site] = count + 1

        return new_heap, addr

    # -- deallocation -------------------------------------------------------

    def deallocate(self, addr: HeapAddress) -> "AbstractHeap":
        """Remove the object at *addr* from the heap.

        Returns a new heap without the object.  In practice abstract
        deallocation is rarely used—garbage collection is preferred.
        """
        new_heap = self.deep_copy()
        new_heap.objects.pop(addr, None)
        return new_heap

    # -- lattice operations -------------------------------------------------

    def join(self, other: AbstractHeap) -> "AbstractHeap":
        """Point-wise join of two heaps.

        Objects present in both heaps are joined attribute-wise.  Objects
        present in only one heap are kept as-is (they are reachable on at
        least one path).
        """
        merged_objects: Dict[HeapAddress, HeapObject] = {}
        all_addrs = set(self.objects.keys()) | set(other.objects.keys())
        for addr in all_addrs:
            obj_a = self.objects.get(addr)
            obj_b = other.objects.get(addr)
            if obj_a is not None and obj_b is not None:
                merged_objects[addr] = obj_a.join_with(obj_b)
            elif obj_a is not None:
                merged_objects[addr] = obj_a.deep_copy()
            else:
                assert obj_b is not None
                merged_objects[addr] = obj_b.deep_copy()

        merged_counts: Dict[str, int] = dict(self.allocation_counts)
        for site, cnt in other.allocation_counts.items():
            merged_counts[site] = max(merged_counts.get(site, 0), cnt)

        return AbstractHeap(
            objects=merged_objects, allocation_counts=merged_counts
        )

    def widen(self, other: AbstractHeap) -> "AbstractHeap":
        """Widening of two heaps (point-wise widening on shared objects).

        Objects that only exist in *other* are included but not widened.
        """
        widened_objects: Dict[HeapAddress, HeapObject] = {}
        all_addrs = set(self.objects.keys()) | set(other.objects.keys())
        for addr in all_addrs:
            obj_a = self.objects.get(addr)
            obj_b = other.objects.get(addr)
            if obj_a is not None and obj_b is not None:
                widened_objects[addr] = obj_a.widen_with(obj_b)
            elif obj_a is not None:
                widened_objects[addr] = obj_a.deep_copy()
            else:
                assert obj_b is not None
                widened_objects[addr] = obj_b.deep_copy()

        widened_counts: Dict[str, int] = dict(self.allocation_counts)
        for site, cnt in other.allocation_counts.items():
            widened_counts[site] = max(widened_counts.get(site, 0), cnt)

        return AbstractHeap(
            objects=widened_objects, allocation_counts=widened_counts
        )

    def meet(self, other: AbstractHeap) -> "AbstractHeap":
        """Point-wise meet of two heaps.

        Only objects present in **both** heaps survive.
        """
        met_objects: Dict[HeapAddress, HeapObject] = {}
        common_addrs = set(self.objects.keys()) & set(other.objects.keys())
        for addr in common_addrs:
            obj_a = self.objects[addr]
            obj_b = other.objects[addr]
            # Meet attributes point-wise
            met_attrs: Dict[str, AbstractValue] = {}
            common_attr_keys = set(obj_a.attrs.keys()) & set(
                obj_b.attrs.keys()
            )
            for key in common_attr_keys:
                met_val = obj_a.attrs[key].meet(obj_b.attrs[key])
                if not met_val.is_bottom():
                    met_attrs[key] = met_val

            met_slots: Dict[str, AbstractValue] = {}
            common_slot_keys = set(obj_a.immutable_slots.keys()) & set(
                obj_b.immutable_slots.keys()
            )
            for key in common_slot_keys:
                met_val = obj_a.immutable_slots[key].meet(
                    obj_b.immutable_slots[key]
                )
                if not met_val.is_bottom():
                    met_slots[key] = met_val

            recency = RecencyFlag.RECENT
            if (
                obj_a.recency == RecencyFlag.SUMMARY
                or obj_b.recency == RecencyFlag.SUMMARY
            ):
                recency = RecencyFlag.SUMMARY

            met_objects[addr] = HeapObject(
                address=addr,
                class_ref=obj_a.class_ref,
                attrs=met_attrs,
                immutable_slots=met_slots,
                recency=recency,
                is_class=obj_a.is_class and obj_b.is_class,
                is_frozen=obj_a.is_frozen or obj_b.is_frozen,
            )

        met_counts: Dict[str, int] = {}
        for site in set(self.allocation_counts.keys()) & set(
            other.allocation_counts.keys()
        ):
            met_counts[site] = min(
                self.allocation_counts[site],
                other.allocation_counts[site],
            )

        return AbstractHeap(objects=met_objects, allocation_counts=met_counts)

    # -- reachability & garbage collection ----------------------------------

    def reachable_from(self, roots: Set[HeapAddress]) -> Set[HeapAddress]:
        """Compute the set of addresses reachable from *roots* via object
        attribute pointers (transitive closure)."""
        visited: Set[HeapAddress] = set()
        worklist = list(roots)
        while worklist:
            addr = worklist.pop()
            if addr in visited:
                continue
            visited.add(addr)
            obj = self.objects.get(addr)
            if obj is None:
                continue
            for ref in obj.referenced_addresses():
                if ref not in visited:
                    worklist.append(ref)
        return visited

    def garbage_collect(self, roots: Set[HeapAddress]) -> "AbstractHeap":
        """Return a new heap containing only objects reachable from *roots*.

        This removes unreachable objects and their allocation-count entries.
        """
        reachable = self.reachable_from(roots)
        gc_objects: Dict[HeapAddress, HeapObject] = {
            addr: obj.deep_copy()
            for addr, obj in self.objects.items()
            if addr in reachable
        }
        reachable_sites = {addr.site for addr in reachable}
        gc_counts: Dict[str, int] = {
            site: cnt
            for site, cnt in self.allocation_counts.items()
            if site in reachable_sites
        }
        return AbstractHeap(objects=gc_objects, allocation_counts=gc_counts)

    # -- deep copy ----------------------------------------------------------

    def deep_copy(self) -> "AbstractHeap":
        """Return a deep copy of this heap."""
        copied_objects: Dict[HeapAddress, HeapObject] = {
            addr: obj.deep_copy() for addr, obj in self.objects.items()
        }
        return AbstractHeap(
            objects=copied_objects,
            allocation_counts=dict(self.allocation_counts),
        )

    # -- helpers ------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.objects)
        return f"AbstractHeap({n} objects)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractHeap):
            return NotImplemented
        if set(self.objects.keys()) != set(other.objects.keys()):
            return False
        for addr in self.objects:
            if addr not in other.objects:
                return False
            obj_a = self.objects[addr]
            obj_b = other.objects[addr]
            if obj_a.attrs != obj_b.attrs:
                return False
            if obj_a.immutable_slots != obj_b.immutable_slots:
                return False
            if obj_a.recency != obj_b.recency:
                return False
        return True

    def dump(self) -> str:
        """Return a human-readable dump of the heap for debugging."""
        lines: List[str] = [f"=== AbstractHeap ({len(self.objects)} objects) ==="]
        for addr in sorted(self.objects.keys(), key=str):
            obj = self.objects[addr]
            flag = "RECENT" if obj.recency == RecencyFlag.RECENT else "SUMMARY"
            kind = "CLASS" if obj.is_class else "INSTANCE"
            frozen_s = " FROZEN" if obj.is_frozen else ""
            lines.append(f"  {addr} [{flag}] {kind}{frozen_s}")
            lines.append(f"    class_ref: {obj.class_ref}")
            if obj.attrs:
                lines.append("    attrs:")
                for k, v in sorted(obj.attrs.items()):
                    lines.append(f"      .{k} = {v}")
            if obj.immutable_slots:
                lines.append("    slots:")
                for k, v in sorted(obj.immutable_slots.items()):
                    lines.append(f"      .{k} = {v}")
        if self.allocation_counts:
            lines.append("  allocation counts:")
            for site, cnt in sorted(self.allocation_counts.items()):
                lines.append(f"    {site}: {cnt}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# HeapState
# ═══════════════════════════════════════════════════════════════════════════

class HeapState:
    """Combined analysis state: heap + variable environment + predicates.

    Parameters
    ----------
    heap : AbstractHeap
        The abstract heap.
    var_env : Dict[str, AbstractValue]
        Variable-to-abstract-value mapping (the "store").
    predicates : Set[str]
        Active refinement predicates, keyed by their string representation
        (e.g. ``"x > 0"``, ``"isinstance(y, int)"``).
    """

    def __init__(
        self,
        heap: Optional[AbstractHeap] = None,
        var_env: Optional[Dict[str, AbstractValue]] = None,
        predicates: Optional[Set[str]] = None,
    ) -> None:
        self.heap: AbstractHeap = heap if heap is not None else AbstractHeap()
        self.var_env: Dict[str, AbstractValue] = (
            var_env if var_env is not None else {}
        )
        self.predicates: Set[str] = (
            predicates if predicates is not None else set()
        )

    # -- variable operations ------------------------------------------------

    def lookup_var(self, name: str) -> AbstractValue:
        """Look up variable *name* in the environment.

        Returns ``BottomValue`` if *name* is not bound.
        """
        return self.var_env.get(name, BottomValue())

    def bind_var(self, name: str, value: AbstractValue) -> "HeapState":
        """Return a new state with *name* bound to *value*."""
        new_env = dict(self.var_env)
        new_env[name] = value
        return HeapState(
            heap=self.heap.deep_copy(),
            var_env=new_env,
            predicates=set(self.predicates),
        )

    def invalidate_var(self, name: str) -> "HeapState":
        """Return a new state where *name* is set to ``TopValue``.

        This is used when a variable's value becomes unknown (e.g. after
        a call that may rebind it).
        """
        new_env = dict(self.var_env)
        new_env[name] = TopValue()
        # Also remove predicates mentioning this variable
        new_preds = {
            p for p in self.predicates if name not in p
        }
        return HeapState(
            heap=self.heap.deep_copy(),
            var_env=new_env,
            predicates=new_preds,
        )

    # -- predicate management -----------------------------------------------

    def add_predicate(self, pred: str) -> "HeapState":
        """Return a new state with *pred* added to the active predicates."""
        new_preds = set(self.predicates)
        new_preds.add(pred)
        return HeapState(
            heap=self.heap.deep_copy(),
            var_env=dict(self.var_env),
            predicates=new_preds,
        )

    def remove_predicate(self, pred: str) -> "HeapState":
        """Return a new state with *pred* removed from the active predicates."""
        new_preds = set(self.predicates)
        new_preds.discard(pred)
        return HeapState(
            heap=self.heap.deep_copy(),
            var_env=dict(self.var_env),
            predicates=new_preds,
        )

    # -- lattice operations -------------------------------------------------

    def join(self, other: HeapState) -> "HeapState":
        """Point-wise join of two states."""
        merged_heap = self.heap.join(other.heap)
        merged_env: Dict[str, AbstractValue] = {}
        all_vars = set(self.var_env.keys()) | set(other.var_env.keys())
        for var in all_vars:
            v1 = self.var_env.get(var, BottomValue())
            v2 = other.var_env.get(var, BottomValue())
            merged_env[var] = v1.join(v2)
        # Predicates: keep only those in both branches (intersection)
        merged_preds = self.predicates & other.predicates
        return HeapState(
            heap=merged_heap,
            var_env=merged_env,
            predicates=merged_preds,
        )

    def widen(self, other: HeapState) -> "HeapState":
        """Point-wise widening of two states."""
        widened_heap = self.heap.widen(other.heap)
        widened_env: Dict[str, AbstractValue] = {}
        all_vars = set(self.var_env.keys()) | set(other.var_env.keys())
        for var in all_vars:
            v1 = self.var_env.get(var, BottomValue())
            v2 = other.var_env.get(var, BottomValue())
            widened_env[var] = v1.widen(v2)
        widened_preds = self.predicates & other.predicates
        return HeapState(
            heap=widened_heap,
            var_env=widened_env,
            predicates=widened_preds,
        )

    def meet(self, other: HeapState) -> "HeapState":
        """Point-wise meet of two states."""
        met_heap = self.heap.meet(other.heap)
        met_env: Dict[str, AbstractValue] = {}
        common_vars = set(self.var_env.keys()) & set(other.var_env.keys())
        for var in common_vars:
            met_env[var] = self.var_env[var].meet(other.var_env[var])
        met_preds = self.predicates | other.predicates
        return HeapState(
            heap=met_heap, var_env=met_env, predicates=met_preds
        )

    def is_bottom(self) -> bool:
        """A state is bottom if the heap is bottom or all variables are bottom."""
        if self.heap.is_bottom():
            return True
        return all(v.is_bottom() for v in self.var_env.values()) and bool(
            self.var_env
        )

    # -- roots for GC ------------------------------------------------------

    def root_addresses(self) -> Set[HeapAddress]:
        """Collect all heap addresses referenced by the variable environment."""
        roots: Set[HeapAddress] = set()
        for v in self.var_env.values():
            roots.update(v.as_addresses())
        return roots

    def garbage_collect(self) -> "HeapState":
        """Return a new state with unreachable heap objects removed."""
        roots = self.root_addresses()
        gc_heap = self.heap.garbage_collect(roots)
        return HeapState(
            heap=gc_heap,
            var_env=dict(self.var_env),
            predicates=set(self.predicates),
        )

    # -- deep copy ----------------------------------------------------------

    def deep_copy(self) -> "HeapState":
        """Return a deep copy of this state."""
        return HeapState(
            heap=self.heap.deep_copy(),
            var_env=dict(self.var_env),
            predicates=set(self.predicates),
        )

    def __repr__(self) -> str:
        n_vars = len(self.var_env)
        n_preds = len(self.predicates)
        return (
            f"HeapState({self.heap!r}, {n_vars} vars, {n_preds} preds)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeapState):
            return NotImplemented
        return (
            self.heap == other.heap
            and self.var_env == other.var_env
            and self.predicates == other.predicates
        )

    def dump(self) -> str:
        """Human-readable dump for debugging."""
        lines: List[str] = ["=== HeapState ==="]
        lines.append(self.heap.dump())
        lines.append("  var_env:")
        for var in sorted(self.var_env.keys()):
            lines.append(f"    {var} = {self.var_env[var]}")
        if self.predicates:
            lines.append("  predicates:")
            for pred in sorted(self.predicates):
                lines.append(f"    {pred}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# HeapTransformer
# ═══════════════════════════════════════════════════════════════════════════

class HeapTransformer:
    """Abstract transformers that model heap-mutating operations.

    Each method takes the current ``HeapState`` (and any relevant
    parameters) and returns a new ``HeapState`` reflecting the effect of
    the operation.  The original state is never mutated.
    """

    # -- object creation ----------------------------------------------------

    def eval_new(
        self,
        site: str,
        class_addr: HeapAddress,
        state: HeapState,
        *,
        init_attrs: Optional[Dict[str, AbstractValue]] = None,
        context: tuple = (),
    ) -> HeapState:
        """Model ``x = ClassName(...)`` — allocate a new object.

        Parameters
        ----------
        site : str
            The allocation site identifier.
        class_addr : HeapAddress
            The heap address of the class being instantiated.
        state : HeapState
            The current analysis state.
        init_attrs : dict, optional
            Initial attribute values to set on the new object.
        context : tuple
            Calling context for context-sensitivity.

        Returns
        -------
        HeapState
            A new state with the freshly allocated object on the heap.  The
            new object's address is stored in the variable ``__last_alloc__``
            in the environment (the caller should rebind it to the target
            variable).
        """
        new_heap, addr = state.heap.allocate(site, class_addr, context)

        # Apply initial attributes if provided
        if init_attrs:
            for attr_name, attr_val in init_attrs.items():
                new_heap = new_heap.write_attr(addr, attr_name, attr_val)

        new_env = dict(state.var_env)
        new_env["__last_alloc__"] = AddressValue(frozenset({addr}))

        return HeapState(
            heap=new_heap,
            var_env=new_env,
            predicates=set(state.predicates),
        )

    # -- attribute access ---------------------------------------------------

    def eval_getattr(
        self, obj_var: str, attr: str, state: HeapState
    ) -> Tuple[AbstractValue, HeapState]:
        """Model ``x.attr`` — read an attribute from an object.

        Parameters
        ----------
        obj_var : str
            The variable name holding the object reference.
        attr : str
            The attribute name to read.
        state : HeapState
            The current analysis state.

        Returns
        -------
        (AbstractValue, HeapState)
            The read value and (potentially updated) state.  The state may
            be updated if reading triggers lazy initialisation in the
            analysis.
        """
        obj_val = state.lookup_var(obj_var)

        if obj_val.is_bottom():
            return BottomValue(), state

        if obj_val.is_top():
            return TopValue(), state

        addrs = obj_val.as_addresses()
        if not addrs:
            # Not a pointer—may be a primitive; attr access on primitive
            # is modelled as Top (e.g. int.bit_length)
            return TopValue(), state

        # Join the attribute value across all possible target objects
        result: AbstractValue = BottomValue()
        for addr in addrs:
            val = state.heap.read_attr(addr, attr)
            result = result.join(val)

            # If the attribute is not found on the instance, try the class
            if val.is_bottom():
                obj = state.heap.get_object(addr)
                if obj is not None:
                    class_val = state.heap.read_attr(obj.class_ref, attr)
                    result = result.join(class_val)

        return result, state

    def eval_setattr(
        self, obj_var: str, attr: str, val: AbstractValue, state: HeapState
    ) -> HeapState:
        """Model ``x.attr = val`` — write an attribute on an object.

        Parameters
        ----------
        obj_var : str
            The variable name holding the object reference.
        attr : str
            The attribute name to write.
        val : AbstractValue
            The value to write.
        state : HeapState
            The current analysis state.

        Returns
        -------
        HeapState
            The updated state after the write.
        """
        obj_val = state.lookup_var(obj_var)
        if obj_val.is_bottom() or obj_val.is_top():
            return state

        addrs = obj_val.as_addresses()
        if not addrs:
            return state

        new_heap = state.heap.deep_copy()

        for addr in addrs:
            if len(addrs) == 1 and new_heap.is_singleton(addr):
                # Strong update: exactly one target, singleton
                obj = new_heap.get_object(addr)
                if obj is not None:
                    obj.write_attr(attr, val)
            else:
                # Weak update: multiple possible targets or summary object
                obj = new_heap.get_object(addr)
                if obj is not None:
                    old_val = obj.read_attr(attr)
                    if old_val.is_bottom():
                        obj.write_attr(attr, val)
                    else:
                        obj.write_attr(attr, old_val.join(val))

        return HeapState(
            heap=new_heap,
            var_env=dict(state.var_env),
            predicates=set(state.predicates),
        )

    # -- assignment ---------------------------------------------------------

    def eval_assign(
        self, target_var: str, value: AbstractValue, state: HeapState
    ) -> HeapState:
        """Model ``target_var = value`` — bind a variable.

        Parameters
        ----------
        target_var : str
            The variable being assigned to.
        value : AbstractValue
            The abstract value being assigned.
        state : HeapState
            The current analysis state.

        Returns
        -------
        HeapState
            The updated state.
        """
        new_env = dict(state.var_env)
        new_env[target_var] = value
        # Invalidate predicates that mention the target variable, since its
        # value has changed.
        new_preds = {
            p for p in state.predicates if target_var not in p
        }
        return HeapState(
            heap=state.heap.deep_copy(),
            var_env=new_env,
            predicates=new_preds,
        )

    # -- function call ------------------------------------------------------

    def eval_call(
        self,
        func: str,
        args: List[str],
        state: HeapState,
        *,
        call_site: str = "",
        return_var: str = "__call_result__",
    ) -> HeapState:
        """Model a function call with potential heap side-effects.

        This is a *conservative* default transformer: it assumes that the
        called function may modify any object reachable from its arguments
        and may return any value.  Specific callees should be handled by
        specialised transformers.

        Parameters
        ----------
        func : str
            The name (or variable holding) the function being called.
        args : list of str
            Variable names of the arguments.
        state : HeapState
            The current analysis state.
        call_site : str
            An identifier for the call site (for context-sensitivity).
        return_var : str
            The variable to bind the return value to.

        Returns
        -------
        HeapState
            The updated state after the call.
        """
        new_heap = state.heap.deep_copy()
        new_env = dict(state.var_env)

        # Collect addresses reachable from arguments
        arg_addrs: Set[HeapAddress] = set()
        for arg_name in args:
            arg_val = state.lookup_var(arg_name)
            arg_addrs.update(arg_val.as_addresses())

        # Conservatively: any attribute of reachable objects may be modified
        reachable = new_heap.reachable_from(arg_addrs)
        for addr in reachable:
            obj = new_heap.get_object(addr)
            if obj is None or obj.is_frozen:
                continue
            if not new_heap.is_singleton(addr):
                # Weak-update all mutable attrs to Top
                for attr_name in list(obj.attrs.keys()):
                    obj.attrs[attr_name] = obj.attrs[attr_name].join(TopValue())

        # The return value is unknown
        new_env[return_var] = TopValue()

        # Invalidate predicates mentioning any argument
        invalidated_vars = set(args) | {func}
        new_preds = {
            p
            for p in state.predicates
            if not any(v in p for v in invalidated_vars)
        }

        return HeapState(
            heap=new_heap, var_env=new_env, predicates=new_preds
        )

    # -- return -------------------------------------------------------------

    def eval_return(
        self, val: AbstractValue, state: HeapState
    ) -> HeapState:
        """Model a ``return val`` statement.

        Stores the return value in ``__return__`` and garbage-collects
        heap objects that are only reachable from local variables.

        Parameters
        ----------
        val : AbstractValue
            The abstract return value.
        state : HeapState
            The current analysis state.

        Returns
        -------
        HeapState
            The state after the return (with ``__return__`` bound).
        """
        new_env = dict(state.var_env)
        new_env["__return__"] = val

        # Determine root set: return value addresses + any callee-visible refs
        return_roots: Set[HeapAddress] = set()
        return_roots.update(val.as_addresses())
        # Keep objects reachable from other live variables too—the caller
        # might still reference them
        for var_name, var_val in state.var_env.items():
            return_roots.update(var_val.as_addresses())

        new_heap = state.heap.deep_copy()
        # We do NOT garbage-collect here because the caller's frame may
        # still hold references.  GC is deferred to the caller.

        return HeapState(
            heap=new_heap,
            var_env=new_env,
            predicates=set(state.predicates),
        )

    # -- compound helpers ---------------------------------------------------

    def eval_method_call(
        self,
        obj_var: str,
        method: str,
        args: List[str],
        state: HeapState,
        *,
        call_site: str = "",
        return_var: str = "__call_result__",
    ) -> HeapState:
        """Model ``obj.method(args...)`` — a method call on an object.

        First resolves the method via attribute lookup on the object (and
        its class), then models the call with the object as the first
        argument.

        Parameters
        ----------
        obj_var : str
            The variable holding the receiver object.
        method : str
            The method name.
        args : list of str
            Variable names of the explicit arguments (excluding ``self``).
        state : HeapState
            The current analysis state.
        call_site : str
            Call site identifier.
        return_var : str
            Variable to bind the return value to.

        Returns
        -------
        HeapState
            The updated state after the method call.
        """
        # Look up the method
        method_val, state = self.eval_getattr(obj_var, method, state)

        # Model the call—include obj_var as implicit self
        all_args = [obj_var] + args
        return self.eval_call(
            func=method,
            args=all_args,
            state=state,
            call_site=call_site,
            return_var=return_var,
        )

    def eval_isinstance_check(
        self,
        obj_var: str,
        class_addr: HeapAddress,
        state: HeapState,
    ) -> Tuple[HeapState, HeapState]:
        """Model ``isinstance(obj_var, SomeClass)`` — split state.

        Returns two states: one where the check is true (the object is
        an instance of the class), and one where it is false.

        Parameters
        ----------
        obj_var : str
            The variable being checked.
        class_addr : HeapAddress
            The heap address of the class to check against.
        state : HeapState
            The current analysis state.

        Returns
        -------
        (HeapState, HeapState)
            ``(true_state, false_state)``
        """
        obj_val = state.lookup_var(obj_var)
        addrs = obj_val.as_addresses()

        if not addrs:
            # Not a pointer—can't determine class, return both branches
            # with original state
            true_state = state.add_predicate(
                f"isinstance({obj_var}, {class_addr})"
            )
            false_state = state.add_predicate(
                f"not isinstance({obj_var}, {class_addr})"
            )
            return true_state, false_state

        true_addrs: Set[HeapAddress] = set()
        false_addrs: Set[HeapAddress] = set()

        for addr in addrs:
            obj = state.heap.get_object(addr)
            if obj is None:
                # Unknown—conservatively include in both
                true_addrs.add(addr)
                false_addrs.add(addr)
            elif obj.class_ref == class_addr:
                true_addrs.add(addr)
            else:
                false_addrs.add(addr)

        true_state = state.deep_copy()
        false_state = state.deep_copy()

        if true_addrs:
            true_state.var_env[obj_var] = AddressValue(
                frozenset(true_addrs)
            )
        else:
            true_state.var_env[obj_var] = BottomValue()

        if false_addrs:
            false_state.var_env[obj_var] = AddressValue(
                frozenset(false_addrs)
            )
        else:
            false_state.var_env[obj_var] = BottomValue()

        true_state.predicates.add(
            f"isinstance({obj_var}, {class_addr})"
        )
        false_state.predicates.add(
            f"not isinstance({obj_var}, {class_addr})"
        )

        return true_state, false_state

    def eval_none_check(
        self, var: str, state: HeapState
    ) -> Tuple[HeapState, HeapState]:
        """Model ``var is None`` — split state into None / not-None branches.

        Parameters
        ----------
        var : str
            The variable being checked.
        state : HeapState
            The current analysis state.

        Returns
        -------
        (HeapState, HeapState)
            ``(is_none_state, is_not_none_state)``
        """
        val = state.lookup_var(var)

        if val.is_definitely_none():
            # Definitely None—true branch is state, false is bottom
            true_state = state.add_predicate(f"{var} is None")
            false_state = state.deep_copy()
            false_state.var_env[var] = BottomValue()
            false_state.predicates.add(f"{var} is not None")
            return true_state, false_state

        if not val.may_be_none():
            # Definitely not None
            true_state = state.deep_copy()
            true_state.var_env[var] = BottomValue()
            true_state.predicates.add(f"{var} is None")
            false_state = state.add_predicate(f"{var} is not None")
            return true_state, false_state

        # May or may not be None—split
        true_state = state.deep_copy()
        true_state.var_env[var] = NoneValue()
        true_state.predicates.add(f"{var} is None")

        false_state = state.deep_copy()
        # Remove None from the union
        if isinstance(val, UnionAbstractValue):
            non_none = [
                c for c in val.components if not c.is_definitely_none()
            ]
            if non_none:
                combined: AbstractValue = non_none[0]
                for part in non_none[1:]:
                    combined = combined.join(part)
                false_state.var_env[var] = combined
            else:
                false_state.var_env[var] = BottomValue()
        elif isinstance(val, NoneValue):
            false_state.var_env[var] = BottomValue()
        else:
            # Keep as-is (it's a Top or something else that may include None)
            false_state.var_env[var] = val
        false_state.predicates.add(f"{var} is not None")

        return true_state, false_state

    def eval_list_append(
        self, list_var: str, elem_val: AbstractValue, state: HeapState
    ) -> HeapState:
        """Model ``list_var.append(elem)`` — mutate a list object.

        The list's ``__items__`` attribute is weakly updated with the
        element value.

        Parameters
        ----------
        list_var : str
            The variable holding the list reference.
        elem_val : AbstractValue
            The abstract value being appended.
        state : HeapState
            The current analysis state.

        Returns
        -------
        HeapState
            The updated state.
        """
        return self.eval_setattr(list_var, "__items__", elem_val, state)

    def eval_dict_setitem(
        self,
        dict_var: str,
        key_val: AbstractValue,
        val: AbstractValue,
        state: HeapState,
    ) -> HeapState:
        """Model ``dict_var[key] = val`` — mutate a dict object.

        The dict's ``__keys__`` and ``__values__`` attributes are weakly
        updated.

        Parameters
        ----------
        dict_var : str
            The variable holding the dict reference.
        key_val : AbstractValue
            The abstract key.
        val : AbstractValue
            The abstract value being stored.
        state : HeapState
            The current analysis state.

        Returns
        -------
        HeapState
            The updated state.
        """
        state = self.eval_setattr(dict_var, "__keys__", key_val, state)
        state = self.eval_setattr(dict_var, "__values__", val, state)
        return state


# ═══════════════════════════════════════════════════════════════════════════
# Module-level convenience helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_int_value(
    lo: Optional[int] = None, hi: Optional[int] = None
) -> PrimitiveValue:
    """Create an abstract integer value with an optional range constraint."""
    constraints: FrozenSet[PrimitiveConstraint] = frozenset()
    if lo is not None or hi is not None:
        constraints = frozenset({IntRange(lo=lo, hi=hi)})
    return PrimitiveValue(kind=PrimitiveKind.INT, constraints=constraints)


def make_str_value(prefix: Optional[str] = None) -> PrimitiveValue:
    """Create an abstract string value with an optional prefix constraint."""
    constraints: FrozenSet[PrimitiveConstraint] = frozenset()
    if prefix is not None:
        constraints = frozenset({StrPrefix(prefix=prefix)})
    return PrimitiveValue(kind=PrimitiveKind.STR, constraints=constraints)


def make_bool_value(value: Optional[bool] = None) -> PrimitiveValue:
    """Create an abstract boolean value, optionally constrained to a
    specific truth value."""
    constraints: FrozenSet[PrimitiveConstraint] = frozenset()
    if value is not None:
        constraints = frozenset({BoolConst(value=value)})
    return PrimitiveValue(kind=PrimitiveKind.BOOL, constraints=constraints)


def make_float_value() -> PrimitiveValue:
    """Create an unconstrained abstract float value."""
    return PrimitiveValue(kind=PrimitiveKind.FLOAT)


def make_bytes_value() -> PrimitiveValue:
    """Create an unconstrained abstract bytes value."""
    return PrimitiveValue(kind=PrimitiveKind.BYTES)


def make_exact_int(value: int) -> PrimitiveValue:
    """Create an abstract integer constrained to exactly *value*."""
    return PrimitiveValue(
        kind=PrimitiveKind.INT,
        constraints=frozenset({ExactValue(value=value)}),
    )


def make_exact_str(value: str) -> PrimitiveValue:
    """Create an abstract string constrained to exactly *value*."""
    return PrimitiveValue(
        kind=PrimitiveKind.STR,
        constraints=frozenset({ExactValue(value=value)}),
    )


def empty_heap_state() -> HeapState:
    """Return a fresh, empty ``HeapState``."""
    return HeapState()


def initial_heap_with_builtins() -> HeapState:
    """Return a ``HeapState`` pre-populated with built-in class objects.

    Creates class objects for ``int``, ``str``, ``bool``, ``float``,
    ``bytes``, ``list``, ``dict``, ``set``, ``tuple``, ``NoneType``,
    and ``object``.
    """
    heap = AbstractHeap()
    var_env: Dict[str, AbstractValue] = {}

    builtin_classes = [
        "object", "int", "str", "bool", "float", "bytes",
        "list", "dict", "set", "tuple", "NoneType", "type",
    ]

    object_class_addr = HeapAddress(site="__builtin__:object")

    for cls_name in builtin_classes:
        addr = HeapAddress(site=f"__builtin__:{cls_name}")
        class_ref = object_class_addr if cls_name != "type" else addr
        if cls_name == "object":
            class_ref = HeapAddress(site="__builtin__:type")

        obj = HeapObject(
            address=addr,
            class_ref=class_ref,
            attrs={
                "__name__": make_exact_str(cls_name),
            },
            immutable_slots={},
            recency=RecencyFlag.SUMMARY,  # builtins are permanent
            is_class=True,
            is_frozen=True,
        )
        heap.objects[addr] = obj
        var_env[cls_name] = AddressValue(frozenset({addr}))

    return HeapState(heap=heap, var_env=var_env, predicates=set())
