"""
Refinement types for Python container objects.

Tracks precise type-level information about lists, dicts, sets, tuples,
and generators as they flow through operations. Each refinement captures
structural invariants (length bounds, per-element types, sortedness) and
updates them soundly after mutations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap
from src.refinement.python_refinements import (
    HeapPredicate, HeapPredKind, PyRefinementType, PyType,
    ListPyType, DictPyType, SetPyType, TuplePyType, IntPyType, StrPyType,
    AnyType, NeverType, PyUnionType,
)


# ---------------------------------------------------------------------------
# Length bounds
# ---------------------------------------------------------------------------

_COMPARISON_OPS = {">=", "<=", ">", "<", "==", "!="}


@dataclass(frozen=True)
class LengthBound:
    """A single relational bound on a container's length."""

    op: str
    value: int

    def __post_init__(self) -> None:
        if self.op not in _COMPARISON_OPS:
            raise ValueError(f"Unsupported comparison operator: {self.op}")

    # -- query ---------------------------------------------------------------

    def satisfies(self, n: int) -> bool:
        """Return whether concrete length *n* satisfies this bound."""
        if self.op == ">=":
            return n >= self.value
        if self.op == "<=":
            return n <= self.value
        if self.op == ">":
            return n > self.value
        if self.op == "<":
            return n < self.value
        if self.op == "==":
            return n == self.value
        # !=
        return n != self.value

    def implies(self, other: LengthBound) -> bool:
        """Return *True* when ``self`` logically implies ``other``."""
        if self.op == "==" and other.op == "==":
            return self.value == other.value
        if self.op == "==" and other.op == ">=":
            return self.value >= other.value
        if self.op == "==" and other.op == "<=":
            return self.value <= other.value
        if self.op == "==" and other.op == ">":
            return self.value > other.value
        if self.op == "==" and other.op == "<":
            return self.value < other.value
        if self.op == "==" and other.op == "!=":
            return self.value != other.value
        if self.op == ">=" and other.op == ">=":
            return self.value >= other.value
        if self.op == ">" and other.op == ">=":
            return self.value >= other.value
        if self.op == ">" and other.op == ">":
            return self.value >= other.value
        if self.op == "<=" and other.op == "<=":
            return self.value <= other.value
        if self.op == "<" and other.op == "<=":
            return self.value <= other.value
        if self.op == "<" and other.op == "<":
            return self.value <= other.value
        return False

    # -- lattice operations --------------------------------------------------

    def join(self, other: LengthBound) -> Optional[LengthBound]:
        """Over-approximate union (weaker bound)."""
        if self.op == other.op:
            if self.op in (">=", ">"):
                return LengthBound(self.op, min(self.value, other.value))
            if self.op in ("<=", "<"):
                return LengthBound(self.op, max(self.value, other.value))
            if self.op == "==":
                if self.value == other.value:
                    return self
                return None
            if self.op == "!=":
                if self.value == other.value:
                    return self
                return None
        if {self.op, other.op} == {">=", ">"}:
            v = min(self.value, other.value)
            return LengthBound(">=", v)
        if {self.op, other.op} == {"<=", "<"}:
            v = max(self.value, other.value)
            return LengthBound("<=", v)
        return None

    def meet(self, other: LengthBound) -> Optional[LengthBound]:
        """Under-approximate intersection (stronger bound)."""
        if self.op == other.op:
            if self.op in (">=", ">"):
                return LengthBound(self.op, max(self.value, other.value))
            if self.op in ("<=", "<"):
                return LengthBound(self.op, min(self.value, other.value))
            if self.op == "==":
                if self.value == other.value:
                    return self
                return None
            if self.op == "!=":
                return None
        if {self.op, other.op} == {">=", ">"}:
            v = max(self.value, other.value)
            return LengthBound(">", v)
        if {self.op, other.op} == {"<=", "<"}:
            v = min(self.value, other.value)
            return LengthBound("<", v)
        return None

    def negate(self) -> LengthBound:
        """Return the logical negation of this bound."""
        neg_map = {
            ">=": "<",
            "<=": ">",
            ">": "<=",
            "<": ">=",
            "==": "!=",
            "!=": "==",
        }
        return LengthBound(neg_map[self.op], self.value)

    def pretty(self) -> str:
        return f"len {self.op} {self.value}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join_bounds(a: List[LengthBound], b: List[LengthBound]) -> List[LengthBound]:
    """Join two lists of bounds (weaker)."""
    result: List[LengthBound] = []
    for ba in a:
        for bb in b:
            j = ba.join(bb)
            if j is not None:
                result.append(j)
    return result


def _meet_bounds(a: List[LengthBound], b: List[LengthBound]) -> List[LengthBound]:
    """Meet two lists of bounds (stronger)."""
    combined = list(a) + list(b)
    return combined


def _join_types(a: PyType, b: PyType) -> PyType:
    """Least upper bound of two PyTypes."""
    if isinstance(a, NeverType):
        return b
    if isinstance(b, NeverType):
        return a
    if isinstance(a, AnyType) or isinstance(b, AnyType):
        return AnyType()
    if type(a) == type(b):
        return a
    return PyUnionType(frozenset({a, b}))


def _meet_types(a: PyType, b: PyType) -> PyType:
    """Greatest lower bound of two PyTypes."""
    if isinstance(a, AnyType):
        return b
    if isinstance(b, AnyType):
        return a
    if isinstance(a, NeverType) or isinstance(b, NeverType):
        return NeverType()
    if type(a) == type(b):
        return a
    return NeverType()


def _join_optional_pred(
    a: Optional[HeapPredicate], b: Optional[HeapPredicate]
) -> Optional[HeapPredicate]:
    """Join two optional predicates; *None* means no information."""
    if a is None or b is None:
        return None
    if a == b:
        return a
    return None


def _bounds_imply_nonempty(bounds: List[LengthBound]) -> bool:
    for b in bounds:
        if b.op == ">=" and b.value >= 1:
            return True
        if b.op == ">" and b.value >= 0:
            return True
        if b.op == "==" and b.value >= 1:
            return True
    return False


def _increment_bounds(bounds: List[LengthBound], delta: int) -> List[LengthBound]:
    """Shift numeric value in every bound by *delta*."""
    result: List[LengthBound] = []
    for b in bounds:
        new_val = b.value + delta
        if new_val >= 0 or b.op in ("!=",):
            result.append(LengthBound(b.op, max(new_val, 0)))
    return result


def _shift_element_predicates(
    preds: Dict[int, HeapPredicate], start: int, delta: int
) -> Dict[int, HeapPredicate]:
    """Shift index-keyed predicates by *delta* starting at index *start*."""
    out: Dict[int, HeapPredicate] = {}
    for idx, pred in preds.items():
        if idx >= start:
            new_idx = idx + delta
            if new_idx >= 0:
                out[new_idx] = pred
        else:
            out[idx] = pred
    return out


# ---------------------------------------------------------------------------
# ListRefinement
# ---------------------------------------------------------------------------

@dataclass
class ListRefinement:
    """Refinement information tracked for a list object."""

    element_type: PyType = field(default_factory=AnyType)
    length_bounds: List[LengthBound] = field(default_factory=list)
    element_predicates: Dict[int, HeapPredicate] = field(default_factory=dict)
    universal_predicate: Optional[HeapPredicate] = None
    structural_props: FrozenSet[str] = field(default_factory=frozenset)

    # -- mutation modelling --------------------------------------------------

    def after_append(self, elem_type: PyType) -> ListRefinement:
        new_elem = _join_types(self.element_type, elem_type)
        new_bounds = _increment_bounds(self.length_bounds, 1)
        if not new_bounds:
            new_bounds = [LengthBound(">=", 1)]
        else:
            new_bounds.append(LengthBound(">=", 1))
        props = self.structural_props - {"sorted", "reversed_sorted"}
        return ListRefinement(
            element_type=new_elem,
            length_bounds=new_bounds,
            element_predicates=dict(self.element_predicates),
            universal_predicate=self.universal_predicate,
            structural_props=props,
        )

    def after_pop(self, index: Optional[int] = None) -> ListRefinement:
        new_bounds = _increment_bounds(self.length_bounds, -1)
        preds = dict(self.element_predicates)
        if index is not None and index in preds:
            del preds[index]
            preds = _shift_element_predicates(preds, index, -1)
        elif index is None:
            # pop from end: remove highest known index
            if preds:
                max_idx = max(preds)
                del preds[max_idx]
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=new_bounds,
            element_predicates=preds,
            universal_predicate=self.universal_predicate,
            structural_props=self.structural_props,
        )

    def after_insert(self, index: int, elem_type: PyType) -> ListRefinement:
        new_elem = _join_types(self.element_type, elem_type)
        new_bounds = _increment_bounds(self.length_bounds, 1)
        if not new_bounds:
            new_bounds = [LengthBound(">=", 1)]
        preds = _shift_element_predicates(self.element_predicates, index, 1)
        props = self.structural_props - {"sorted", "reversed_sorted", "unique"}
        return ListRefinement(
            element_type=new_elem,
            length_bounds=new_bounds,
            element_predicates=preds,
            universal_predicate=self.universal_predicate,
            structural_props=props,
        )

    def after_extend(self, other: ListRefinement) -> ListRefinement:
        new_elem = _join_types(self.element_type, other.element_type)
        new_bounds: List[LengthBound] = []
        for bs in self.length_bounds:
            for bo in other.length_bounds:
                if bs.op == ">=" and bo.op == ">=":
                    new_bounds.append(LengthBound(">=", bs.value + bo.value))
                elif bs.op == "==" and bo.op == "==":
                    new_bounds.append(LengthBound("==", bs.value + bo.value))
        if not new_bounds and _bounds_imply_nonempty(self.length_bounds):
            new_bounds = [LengthBound(">=", 1)]
        props = self.structural_props - {"sorted", "reversed_sorted", "unique"}
        univ = _join_optional_pred(self.universal_predicate, other.universal_predicate)
        return ListRefinement(
            element_type=new_elem,
            length_bounds=new_bounds,
            element_predicates={},
            universal_predicate=univ,
            structural_props=props,
        )

    def after_sort(self) -> ListRefinement:
        props = (self.structural_props | {"sorted"}) - {"reversed_sorted"}
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=list(self.length_bounds),
            element_predicates={},
            universal_predicate=self.universal_predicate,
            structural_props=props,
        )

    def after_reverse(self) -> ListRefinement:
        new_props: Set[str] = set(self.structural_props)
        if "sorted" in new_props:
            new_props.discard("sorted")
            new_props.add("reversed_sorted")
        elif "reversed_sorted" in new_props:
            new_props.discard("reversed_sorted")
            new_props.add("sorted")
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=list(self.length_bounds),
            element_predicates={},
            universal_predicate=self.universal_predicate,
            structural_props=frozenset(new_props),
        )

    def after_clear(self) -> ListRefinement:
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=[LengthBound("==", 0)],
            element_predicates={},
            universal_predicate=None,
            structural_props=frozenset(),
        )

    def after_setitem(self, index: int, elem_type: PyType) -> ListRefinement:
        new_elem = _join_types(self.element_type, elem_type)
        preds = dict(self.element_predicates)
        preds.pop(index, None)
        props = self.structural_props - {"sorted", "reversed_sorted", "unique"}
        return ListRefinement(
            element_type=new_elem,
            length_bounds=list(self.length_bounds),
            element_predicates=preds,
            universal_predicate=self.universal_predicate,
            structural_props=props,
        )

    def after_delitem(self, index: int) -> ListRefinement:
        new_bounds = _increment_bounds(self.length_bounds, -1)
        preds = dict(self.element_predicates)
        preds.pop(index, None)
        preds = _shift_element_predicates(preds, index, -1)
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=new_bounds,
            element_predicates=preds,
            universal_predicate=self.universal_predicate,
            structural_props=self.structural_props,
        )

    def after_slice(
        self,
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
    ) -> ListRefinement:
        s = start if start is not None else 0
        bounds: List[LengthBound] = []
        preds: Dict[int, HeapPredicate] = {}
        if stop is not None and step is None or step == 1:
            length = max(stop - s, 0)
            bounds.append(LengthBound("<=", length))
            for idx, pred in self.element_predicates.items():
                if s <= idx < stop:
                    preds[idx - s] = pred
        props = self.structural_props - {"non_empty"}
        return ListRefinement(
            element_type=self.element_type,
            length_bounds=bounds,
            element_predicates=preds,
            universal_predicate=self.universal_predicate,
            structural_props=props,
        )

    # -- queries -------------------------------------------------------------

    def element_at(self, index: int) -> PyRefinementType:
        pred = self.element_predicates.get(index)
        preds: List[HeapPredicate] = []
        if pred is not None:
            preds.append(pred)
        if self.universal_predicate is not None:
            preds.append(self.universal_predicate)
        return PyRefinementType(base=self.element_type, predicates=tuple(preds))

    def is_non_empty(self) -> bool:
        if "non_empty" in self.structural_props:
            return True
        return _bounds_imply_nonempty(self.length_bounds)

    # -- lattice -------------------------------------------------------------

    def join(self, other: ListRefinement) -> ListRefinement:
        elem = _join_types(self.element_type, other.element_type)
        bounds = _join_bounds(self.length_bounds, other.length_bounds)
        shared_preds: Dict[int, HeapPredicate] = {}
        for idx in set(self.element_predicates) & set(other.element_predicates):
            jp = _join_optional_pred(
                self.element_predicates[idx], other.element_predicates[idx]
            )
            if jp is not None:
                shared_preds[idx] = jp
        univ = _join_optional_pred(self.universal_predicate, other.universal_predicate)
        props = self.structural_props & other.structural_props
        return ListRefinement(
            element_type=elem,
            length_bounds=bounds,
            element_predicates=shared_preds,
            universal_predicate=univ,
            structural_props=props,
        )

    def meet(self, other: ListRefinement) -> ListRefinement:
        elem = _meet_types(self.element_type, other.element_type)
        bounds = _meet_bounds(self.length_bounds, other.length_bounds)
        preds = dict(self.element_predicates)
        preds.update(other.element_predicates)
        univ = self.universal_predicate or other.universal_predicate
        props = self.structural_props | other.structural_props
        return ListRefinement(
            element_type=elem,
            length_bounds=bounds,
            element_predicates=preds,
            universal_predicate=univ,
            structural_props=props,
        )

    def widen(self, other: ListRefinement) -> ListRefinement:
        elem = _join_types(self.element_type, other.element_type)
        # Drop bounds that grew between iterations to ensure convergence
        kept: List[LengthBound] = []
        for b in self.length_bounds:
            for ob in other.length_bounds:
                if b.op == ob.op:
                    if b.op in (">=", ">") and ob.value >= b.value:
                        kept.append(b)
                    elif b.op in ("<=", "<") and ob.value <= b.value:
                        kept.append(b)
                    elif b.op == "==" and b.value == ob.value:
                        kept.append(b)
        props = self.structural_props & other.structural_props
        return ListRefinement(
            element_type=elem,
            length_bounds=kept,
            element_predicates={},
            universal_predicate=_join_optional_pred(
                self.universal_predicate, other.universal_predicate
            ),
            structural_props=props,
        )

    # -- serialisation -------------------------------------------------------

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        result: List[HeapPredicate] = []
        for b in self.length_bounds:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.LENGTH,
                    subject=var,
                    detail=b.pretty(),
                )
            )
        for idx, pred in self.element_predicates.items():
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.ELEMENT,
                    subject=f"{var}[{idx}]",
                    detail=str(pred),
                )
            )
        if self.universal_predicate is not None:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.FORALL,
                    subject=var,
                    detail=str(self.universal_predicate),
                )
            )
        for prop in sorted(self.structural_props):
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.STRUCTURAL,
                    subject=var,
                    detail=prop,
                )
            )
        return result

    @staticmethod
    def from_predicates(preds: List[HeapPredicate]) -> ListRefinement:
        bounds: List[LengthBound] = []
        elem_preds: Dict[int, HeapPredicate] = {}
        univ: Optional[HeapPredicate] = None
        props: Set[str] = set()
        for p in preds:
            if p.kind == HeapPredKind.LENGTH:
                parts = p.detail.split()
                if len(parts) == 3:
                    _, op, val_s = parts
                    try:
                        bounds.append(LengthBound(op, int(val_s)))
                    except (ValueError, KeyError):
                        pass
            elif p.kind == HeapPredKind.ELEMENT:
                # subject like "x[3]"
                try:
                    idx_str = p.subject.split("[")[1].rstrip("]")
                    idx = int(idx_str)
                    elem_preds[idx] = p
                except (IndexError, ValueError):
                    pass
            elif p.kind == HeapPredKind.FORALL:
                univ = p
            elif p.kind == HeapPredKind.STRUCTURAL:
                props.add(p.detail)
        return ListRefinement(
            element_type=AnyType(),
            length_bounds=bounds,
            element_predicates=elem_preds,
            universal_predicate=univ,
            structural_props=frozenset(props),
        )

    def pretty(self) -> str:
        parts: List[str] = [f"List[{self.element_type}]"]
        for b in self.length_bounds:
            parts.append(b.pretty())
        if self.structural_props:
            parts.append("{" + ", ".join(sorted(self.structural_props)) + "}")
        if self.universal_predicate:
            parts.append(f"∀elem: {self.universal_predicate}")
        for idx in sorted(self.element_predicates):
            parts.append(f"[{idx}]: {self.element_predicates[idx]}")
        return " & ".join(parts)


# ---------------------------------------------------------------------------
# DictRefinement
# ---------------------------------------------------------------------------

@dataclass
class DictRefinement:
    """Refinement information tracked for a dict object."""

    key_type: PyType = field(default_factory=AnyType)
    value_type: PyType = field(default_factory=AnyType)
    required_keys: Dict[str, PyType] = field(default_factory=dict)
    optional_keys: Dict[str, PyType] = field(default_factory=dict)
    length_bounds: List[LengthBound] = field(default_factory=list)
    key_refinements: Dict[str, HeapPredicate] = field(default_factory=dict)

    # -- mutation modelling --------------------------------------------------

    def after_setitem(self, key: str, val_type: PyType) -> DictRefinement:
        req = dict(self.required_keys)
        opt = dict(self.optional_keys)
        opt.pop(key, None)
        req[key] = val_type
        new_val = _join_types(self.value_type, val_type)
        bounds = list(self.length_bounds)
        if key not in self.required_keys:
            bounds = _increment_bounds(bounds, 1)
            if not bounds:
                bounds = [LengthBound(">=", 1)]
        refs = dict(self.key_refinements)
        refs.pop(key, None)
        return DictRefinement(
            key_type=_join_types(self.key_type, StrPyType()),
            value_type=new_val,
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=refs,
        )

    def after_delitem(self, key: str) -> DictRefinement:
        req = dict(self.required_keys)
        opt = dict(self.optional_keys)
        req.pop(key, None)
        opt.pop(key, None)
        refs = dict(self.key_refinements)
        refs.pop(key, None)
        bounds = _increment_bounds(self.length_bounds, -1)
        return DictRefinement(
            key_type=self.key_type,
            value_type=self.value_type,
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=refs,
        )

    def after_update(self, other: DictRefinement) -> DictRefinement:
        req = dict(self.required_keys)
        req.update(other.required_keys)
        opt = dict(self.optional_keys)
        for k, v in other.optional_keys.items():
            if k not in req:
                opt[k] = v
        for k in other.required_keys:
            opt.pop(k, None)
        new_key = _join_types(self.key_type, other.key_type)
        new_val = _join_types(self.value_type, other.value_type)
        bounds: List[LengthBound] = []
        for bs in self.length_bounds:
            for bo in other.length_bounds:
                if bs.op == ">=" and bo.op == ">=":
                    bounds.append(LengthBound(">=", max(bs.value, bo.value)))
        refs = dict(self.key_refinements)
        refs.update(other.key_refinements)
        return DictRefinement(
            key_type=new_key,
            value_type=new_val,
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=refs,
        )

    def after_pop(self, key: str) -> Tuple[DictRefinement, PyType]:
        val_ty = self.required_keys.get(key) or self.optional_keys.get(key) or self.value_type
        new_dict = self.after_delitem(key)
        return new_dict, val_ty

    def after_clear(self) -> DictRefinement:
        return DictRefinement(
            key_type=self.key_type,
            value_type=self.value_type,
            required_keys={},
            optional_keys={},
            length_bounds=[LengthBound("==", 0)],
            key_refinements={},
        )

    def after_setdefault(self, key: str, default_type: PyType) -> DictRefinement:
        if key in self.required_keys:
            return DictRefinement(
                key_type=self.key_type,
                value_type=self.value_type,
                required_keys=dict(self.required_keys),
                optional_keys=dict(self.optional_keys),
                length_bounds=list(self.length_bounds),
                key_refinements=dict(self.key_refinements),
            )
        req = dict(self.required_keys)
        opt = dict(self.optional_keys)
        existing_ty = opt.pop(key, None)
        if existing_ty is not None:
            req[key] = _join_types(existing_ty, default_type)
        else:
            req[key] = default_type
        bounds = list(self.length_bounds)
        if key not in self.required_keys and key not in self.optional_keys:
            bounds = _increment_bounds(bounds, 1)
            if not bounds:
                bounds = [LengthBound(">=", 1)]
        return DictRefinement(
            key_type=_join_types(self.key_type, StrPyType()),
            value_type=_join_types(self.value_type, default_type),
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=dict(self.key_refinements),
        )

    # -- queries -------------------------------------------------------------

    def has_key(self, key: str) -> Optional[bool]:
        if key in self.required_keys:
            return True
        if key in self.optional_keys:
            return None
        # Could still be an unknown dynamic key
        return None

    def value_for_key(self, key: str) -> PyRefinementType:
        if key in self.required_keys:
            ty = self.required_keys[key]
        elif key in self.optional_keys:
            ty = self.optional_keys[key]
        else:
            ty = self.value_type
        preds: List[HeapPredicate] = []
        if key in self.key_refinements:
            preds.append(self.key_refinements[key])
        return PyRefinementType(base=ty, predicates=tuple(preds))

    # -- lattice -------------------------------------------------------------

    def join(self, other: DictRefinement) -> DictRefinement:
        key_ty = _join_types(self.key_type, other.key_type)
        val_ty = _join_types(self.value_type, other.value_type)
        # Required only if required in both
        req: Dict[str, PyType] = {}
        for k in set(self.required_keys) & set(other.required_keys):
            req[k] = _join_types(self.required_keys[k], other.required_keys[k])
        # Keys required in one but not the other become optional
        opt: Dict[str, PyType] = dict(self.optional_keys)
        opt.update(other.optional_keys)
        for k in set(self.required_keys) ^ set(other.required_keys):
            ty_a = self.required_keys.get(k) or self.optional_keys.get(k)
            ty_b = other.required_keys.get(k) or other.optional_keys.get(k)
            if ty_a and ty_b:
                opt[k] = _join_types(ty_a, ty_b)
            elif ty_a:
                opt[k] = ty_a
            elif ty_b:
                opt[k] = ty_b
        for k in req:
            opt.pop(k, None)
        bounds = _join_bounds(self.length_bounds, other.length_bounds)
        refs: Dict[str, HeapPredicate] = {}
        for k in set(self.key_refinements) & set(other.key_refinements):
            jp = _join_optional_pred(self.key_refinements[k], other.key_refinements[k])
            if jp is not None:
                refs[k] = jp
        return DictRefinement(
            key_type=key_ty,
            value_type=val_ty,
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=refs,
        )

    def meet(self, other: DictRefinement) -> DictRefinement:
        key_ty = _meet_types(self.key_type, other.key_type)
        val_ty = _meet_types(self.value_type, other.value_type)
        req = dict(self.required_keys)
        for k, v in other.required_keys.items():
            if k in req:
                req[k] = _meet_types(req[k], v)
            else:
                req[k] = v
        opt: Dict[str, PyType] = {}
        for k in set(self.optional_keys) & set(other.optional_keys):
            opt[k] = _meet_types(self.optional_keys[k], other.optional_keys[k])
        for k in req:
            opt.pop(k, None)
        bounds = _meet_bounds(self.length_bounds, other.length_bounds)
        refs = dict(self.key_refinements)
        refs.update(other.key_refinements)
        return DictRefinement(
            key_type=key_ty,
            value_type=val_ty,
            required_keys=req,
            optional_keys=opt,
            length_bounds=bounds,
            key_refinements=refs,
        )

    def widen(self, other: DictRefinement) -> DictRefinement:
        key_ty = _join_types(self.key_type, other.key_type)
        val_ty = _join_types(self.value_type, other.value_type)
        req: Dict[str, PyType] = {}
        for k in set(self.required_keys) & set(other.required_keys):
            req[k] = _join_types(self.required_keys[k], other.required_keys[k])
        return DictRefinement(
            key_type=key_ty,
            value_type=val_ty,
            required_keys=req,
            optional_keys={},
            length_bounds=[],
            key_refinements={},
        )

    def is_typed_dict_compatible(self) -> bool:
        """Return *True* when all keys are string-typed and fully known."""
        if not isinstance(self.key_type, StrPyType):
            return False
        if not self.required_keys and not self.optional_keys:
            return False
        for b in self.length_bounds:
            if b.op == "==" and b.value == len(self.required_keys):
                return True
        return bool(self.required_keys) and not self.optional_keys

    # -- serialisation -------------------------------------------------------

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        result: List[HeapPredicate] = []
        for b in self.length_bounds:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.LENGTH,
                    subject=var,
                    detail=b.pretty(),
                )
            )
        for k, ty in sorted(self.required_keys.items()):
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.KEY_PRESENT,
                    subject=var,
                    detail=f"{k}: {ty} (required)",
                )
            )
        for k, ty in sorted(self.optional_keys.items()):
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.KEY_PRESENT,
                    subject=var,
                    detail=f"{k}: {ty} (optional)",
                )
            )
        for k, pred in sorted(self.key_refinements.items()):
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.ELEMENT,
                    subject=f"{var}[{k!r}]",
                    detail=str(pred),
                )
            )
        return result

    def pretty(self) -> str:
        parts: List[str] = [f"Dict[{self.key_type}, {self.value_type}]"]
        for b in self.length_bounds:
            parts.append(b.pretty())
        if self.required_keys:
            keys_s = ", ".join(f"{k}: {v}" for k, v in sorted(self.required_keys.items()))
            parts.append(f"required{{{keys_s}}}")
        if self.optional_keys:
            keys_s = ", ".join(f"{k}: {v}" for k, v in sorted(self.optional_keys.items()))
            parts.append(f"optional{{{keys_s}}}")
        return " & ".join(parts)


# ---------------------------------------------------------------------------
# SetRefinement
# ---------------------------------------------------------------------------

@dataclass
class SetRefinement:
    """Refinement information tracked for a set object."""

    element_type: PyType = field(default_factory=AnyType)
    length_bounds: List[LengthBound] = field(default_factory=list)
    known_elements: Optional[FrozenSet[Any]] = None
    subset_of: Optional[FrozenSet[Any]] = None
    superset_of: Optional[FrozenSet[Any]] = None

    # -- mutation modelling --------------------------------------------------

    def after_add(self, elem_type: PyType) -> SetRefinement:
        new_elem = _join_types(self.element_type, elem_type)
        bounds = list(self.length_bounds)
        if not _bounds_imply_nonempty(bounds):
            bounds.append(LengthBound(">=", 1))
        return SetRefinement(
            element_type=new_elem,
            length_bounds=bounds,
            known_elements=None,
            subset_of=self.subset_of,
            superset_of=self.superset_of,
        )

    def after_discard(self, elem: Any) -> SetRefinement:
        known = None
        if self.known_elements is not None and elem in self.known_elements:
            known = self.known_elements - {elem}
        elif self.known_elements is not None:
            known = self.known_elements
        sup = self.superset_of
        if sup is not None:
            sup = sup - {elem}
        return SetRefinement(
            element_type=self.element_type,
            length_bounds=list(self.length_bounds),
            known_elements=known,
            subset_of=self.subset_of,
            superset_of=sup,
        )

    def after_remove(self, elem: Any) -> SetRefinement:
        known = None
        if self.known_elements is not None:
            known = self.known_elements - {elem}
        sup = self.superset_of
        if sup is not None:
            sup = sup - {elem}
        return SetRefinement(
            element_type=self.element_type,
            length_bounds=list(self.length_bounds),
            known_elements=known,
            subset_of=self.subset_of,
            superset_of=sup,
        )

    def after_union(self, other: SetRefinement) -> SetRefinement:
        new_elem = _join_types(self.element_type, other.element_type)
        known = None
        if self.known_elements is not None and other.known_elements is not None:
            known = self.known_elements | other.known_elements
        sub = None
        if self.subset_of is not None and other.subset_of is not None:
            sub = self.subset_of | other.subset_of
        sup = None
        if self.superset_of is not None:
            sup = self.superset_of
        if other.superset_of is not None:
            sup = (sup or frozenset()) | other.superset_of
        bounds: List[LengthBound] = []
        for bs in self.length_bounds:
            if bs.op == ">=" or bs.op == ">":
                bounds.append(bs)
        return SetRefinement(
            element_type=new_elem,
            length_bounds=bounds,
            known_elements=known,
            subset_of=sub,
            superset_of=sup,
        )

    def after_intersection(self, other: SetRefinement) -> SetRefinement:
        new_elem = _meet_types(self.element_type, other.element_type)
        if isinstance(new_elem, NeverType):
            new_elem = _join_types(self.element_type, other.element_type)
        known = None
        if self.known_elements is not None and other.known_elements is not None:
            known = self.known_elements & other.known_elements
        sub = self.subset_of
        if other.subset_of is not None:
            if sub is not None:
                sub = sub & other.subset_of
            else:
                sub = other.subset_of
        bounds: List[LengthBound] = []
        for bs in self.length_bounds:
            if bs.op in ("<=", "<"):
                bounds.append(bs)
        for bo in other.length_bounds:
            if bo.op in ("<=", "<"):
                bounds.append(bo)
        return SetRefinement(
            element_type=new_elem,
            length_bounds=bounds,
            known_elements=known,
            subset_of=sub,
            superset_of=None,
        )

    def after_difference(self, other: SetRefinement) -> SetRefinement:
        known = None
        if self.known_elements is not None and other.known_elements is not None:
            known = self.known_elements - other.known_elements
        elif self.known_elements is not None:
            known = None
        bounds: List[LengthBound] = []
        for bs in self.length_bounds:
            if bs.op in ("<=", "<"):
                bounds.append(bs)
        return SetRefinement(
            element_type=self.element_type,
            length_bounds=bounds,
            known_elements=known,
            subset_of=self.subset_of,
            superset_of=None,
        )

    def after_clear(self) -> SetRefinement:
        return SetRefinement(
            element_type=self.element_type,
            length_bounds=[LengthBound("==", 0)],
            known_elements=frozenset(),
            subset_of=None,
            superset_of=None,
        )

    # -- queries -------------------------------------------------------------

    def contains(self, elem: Any) -> Optional[bool]:
        if self.known_elements is not None:
            return elem in self.known_elements
        if self.superset_of is not None and elem in self.superset_of:
            return True
        if self.subset_of is not None and elem not in self.subset_of:
            return False
        return None

    # -- lattice -------------------------------------------------------------

    def join(self, other: SetRefinement) -> SetRefinement:
        elem = _join_types(self.element_type, other.element_type)
        bounds = _join_bounds(self.length_bounds, other.length_bounds)
        known = None
        if self.known_elements is not None and other.known_elements is not None:
            if self.known_elements == other.known_elements:
                known = self.known_elements
        sub = None
        if self.subset_of is not None and other.subset_of is not None:
            sub = self.subset_of | other.subset_of
        sup = None
        if self.superset_of is not None and other.superset_of is not None:
            sup = self.superset_of & other.superset_of
            if not sup:
                sup = None
        return SetRefinement(
            element_type=elem,
            length_bounds=bounds,
            known_elements=known,
            subset_of=sub,
            superset_of=sup,
        )

    def meet(self, other: SetRefinement) -> SetRefinement:
        elem = _meet_types(self.element_type, other.element_type)
        if isinstance(elem, NeverType):
            elem = _join_types(self.element_type, other.element_type)
        bounds = _meet_bounds(self.length_bounds, other.length_bounds)
        known = None
        if self.known_elements is not None and other.known_elements is not None:
            known = self.known_elements & other.known_elements
        elif self.known_elements is not None:
            known = self.known_elements
        elif other.known_elements is not None:
            known = other.known_elements
        sub = None
        if self.subset_of is not None and other.subset_of is not None:
            sub = self.subset_of & other.subset_of
        elif self.subset_of is not None:
            sub = self.subset_of
        elif other.subset_of is not None:
            sub = other.subset_of
        sup = None
        if self.superset_of is not None and other.superset_of is not None:
            sup = self.superset_of | other.superset_of
        elif self.superset_of is not None:
            sup = self.superset_of
        elif other.superset_of is not None:
            sup = other.superset_of
        return SetRefinement(
            element_type=elem,
            length_bounds=bounds,
            known_elements=known,
            subset_of=sub,
            superset_of=sup,
        )

    # -- serialisation -------------------------------------------------------

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        result: List[HeapPredicate] = []
        for b in self.length_bounds:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.LENGTH,
                    subject=var,
                    detail=b.pretty(),
                )
            )
        if self.known_elements is not None:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.STRUCTURAL,
                    subject=var,
                    detail=f"known_elements={self.known_elements}",
                )
            )
        if self.subset_of is not None:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.STRUCTURAL,
                    subject=var,
                    detail=f"subset_of={self.subset_of}",
                )
            )
        if self.superset_of is not None:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.STRUCTURAL,
                    subject=var,
                    detail=f"superset_of={self.superset_of}",
                )
            )
        return result

    def pretty(self) -> str:
        parts: List[str] = [f"Set[{self.element_type}]"]
        for b in self.length_bounds:
            parts.append(b.pretty())
        if self.known_elements is not None:
            parts.append(f"exactly={self.known_elements}")
        if self.subset_of is not None:
            parts.append(f"⊆{self.subset_of}")
        if self.superset_of is not None:
            parts.append(f"⊇{self.superset_of}")
        return " & ".join(parts)


# ---------------------------------------------------------------------------
# TupleRefinement
# ---------------------------------------------------------------------------

@dataclass
class TupleRefinement:
    """Refinement information tracked for a tuple object (immutable)."""

    element_types: Tuple[PyType, ...] = ()
    element_predicates: Dict[int, HeapPredicate] = field(default_factory=dict)
    is_named: bool = False
    field_names: Optional[Tuple[str, ...]] = None

    # -- queries -------------------------------------------------------------

    def element_at(self, index: int) -> PyRefinementType:
        if 0 <= index < len(self.element_types):
            ty = self.element_types[index]
        elif index < 0 and abs(index) <= len(self.element_types):
            ty = self.element_types[index]
        else:
            ty = AnyType()
        preds: List[HeapPredicate] = []
        normalised = index if index >= 0 else len(self.element_types) + index
        if normalised in self.element_predicates:
            preds.append(self.element_predicates[normalised])
        return PyRefinementType(base=ty, predicates=tuple(preds))

    def slice(self, start: Optional[int], stop: Optional[int]) -> TupleRefinement:
        s = start if start is not None else 0
        e = stop if stop is not None else len(self.element_types)
        s = max(s, 0)
        e = min(e, len(self.element_types))
        new_types = self.element_types[s:e]
        new_preds: Dict[int, HeapPredicate] = {}
        for idx, pred in self.element_predicates.items():
            if s <= idx < e:
                new_preds[idx - s] = pred
        return TupleRefinement(
            element_types=new_types,
            element_predicates=new_preds,
            is_named=False,
            field_names=None,
        )

    def concat(self, other: TupleRefinement) -> TupleRefinement:
        new_types = self.element_types + other.element_types
        new_preds = dict(self.element_predicates)
        offset = len(self.element_types)
        for idx, pred in other.element_predicates.items():
            new_preds[idx + offset] = pred
        return TupleRefinement(
            element_types=new_types,
            element_predicates=new_preds,
            is_named=False,
            field_names=None,
        )

    # -- lattice -------------------------------------------------------------

    def join(self, other: TupleRefinement) -> TupleRefinement:
        if len(self.element_types) != len(other.element_types):
            all_types = set(self.element_types) | set(other.element_types)
            unified = AnyType()
            for t in all_types:
                unified = _join_types(unified, t)
            max_len = max(len(self.element_types), len(other.element_types))
            new_types = tuple(unified for _ in range(max_len))
            return TupleRefinement(element_types=new_types)
        new_types = tuple(
            _join_types(a, b) for a, b in zip(self.element_types, other.element_types)
        )
        shared_preds: Dict[int, HeapPredicate] = {}
        for idx in set(self.element_predicates) & set(other.element_predicates):
            jp = _join_optional_pred(
                self.element_predicates[idx], other.element_predicates[idx]
            )
            if jp is not None:
                shared_preds[idx] = jp
        named = self.is_named and other.is_named
        names = self.field_names if named and self.field_names == other.field_names else None
        return TupleRefinement(
            element_types=new_types,
            element_predicates=shared_preds,
            is_named=named,
            field_names=names,
        )

    def meet(self, other: TupleRefinement) -> TupleRefinement:
        min_len = min(len(self.element_types), len(other.element_types))
        new_types = tuple(
            _meet_types(self.element_types[i], other.element_types[i])
            for i in range(min_len)
        )
        preds = dict(self.element_predicates)
        preds.update(other.element_predicates)
        named = self.is_named or other.is_named
        names = self.field_names or other.field_names
        return TupleRefinement(
            element_types=new_types,
            element_predicates=preds,
            is_named=named,
            field_names=names,
        )

    # -- serialisation -------------------------------------------------------

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        result: List[HeapPredicate] = []
        result.append(
            HeapPredicate(
                kind=HeapPredKind.LENGTH,
                subject=var,
                detail=f"len == {len(self.element_types)}",
            )
        )
        for idx, pred in sorted(self.element_predicates.items()):
            label = f"{var}[{idx}]"
            if self.field_names and 0 <= idx < len(self.field_names):
                label = f"{var}.{self.field_names[idx]}"
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.ELEMENT,
                    subject=label,
                    detail=str(pred),
                )
            )
        return result

    def pretty(self) -> str:
        inner = ", ".join(str(t) for t in self.element_types)
        base = f"Tuple[{inner}]"
        if self.is_named and self.field_names:
            fields = ", ".join(
                f"{n}: {t}" for n, t in zip(self.field_names, self.element_types)
            )
            base = f"NamedTuple({fields})"
        extras: List[str] = []
        for idx in sorted(self.element_predicates):
            extras.append(f"[{idx}]: {self.element_predicates[idx]}")
        if extras:
            return base + " & " + " & ".join(extras)
        return base


# ---------------------------------------------------------------------------
# GeneratorRefinement
# ---------------------------------------------------------------------------

@dataclass
class GeneratorRefinement:
    """Refinement information tracked for a generator / iterator."""

    yield_type: PyType = field(default_factory=AnyType)
    send_type: PyType = field(default_factory=AnyType)
    return_type: PyType = field(default_factory=AnyType)
    exhausted: Optional[bool] = None
    yields_at_least: Optional[int] = None

    # -- operations ----------------------------------------------------------

    def after_next(self) -> Tuple[GeneratorRefinement, PyRefinementType]:
        remaining = None
        if self.yields_at_least is not None and self.yields_at_least > 0:
            remaining = self.yields_at_least - 1
        new_exhausted = self.exhausted
        if remaining is not None and remaining == 0:
            new_exhausted = None  # might be exhausted now
        yielded = PyRefinementType(base=self.yield_type, predicates=())
        return (
            GeneratorRefinement(
                yield_type=self.yield_type,
                send_type=self.send_type,
                return_type=self.return_type,
                exhausted=new_exhausted,
                yields_at_least=remaining,
            ),
            yielded,
        )

    def after_send(self, val_type: PyType) -> Tuple[GeneratorRefinement, PyRefinementType]:
        new_send = _join_types(self.send_type, val_type)
        remaining = None
        if self.yields_at_least is not None and self.yields_at_least > 0:
            remaining = self.yields_at_least - 1
        yielded = PyRefinementType(base=self.yield_type, predicates=())
        return (
            GeneratorRefinement(
                yield_type=self.yield_type,
                send_type=new_send,
                return_type=self.return_type,
                exhausted=self.exhausted,
                yields_at_least=remaining,
            ),
            yielded,
        )

    def mark_exhausted(self) -> GeneratorRefinement:
        return GeneratorRefinement(
            yield_type=self.yield_type,
            send_type=self.send_type,
            return_type=self.return_type,
            exhausted=True,
            yields_at_least=0,
        )

    # -- lattice -------------------------------------------------------------

    def join(self, other: GeneratorRefinement) -> GeneratorRefinement:
        y = _join_types(self.yield_type, other.yield_type)
        s = _join_types(self.send_type, other.send_type)
        r = _join_types(self.return_type, other.return_type)
        ex: Optional[bool] = None
        if self.exhausted is not None and other.exhausted is not None:
            ex = self.exhausted and other.exhausted
        yld: Optional[int] = None
        if self.yields_at_least is not None and other.yields_at_least is not None:
            yld = min(self.yields_at_least, other.yields_at_least)
        return GeneratorRefinement(
            yield_type=y, send_type=s, return_type=r,
            exhausted=ex, yields_at_least=yld,
        )

    # -- serialisation -------------------------------------------------------

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        result: List[HeapPredicate] = []
        if self.exhausted is True:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.STRUCTURAL,
                    subject=var,
                    detail="exhausted",
                )
            )
        if self.yields_at_least is not None:
            result.append(
                HeapPredicate(
                    kind=HeapPredKind.LENGTH,
                    subject=var,
                    detail=f"yields_at_least >= {self.yields_at_least}",
                )
            )
        return result

    def pretty(self) -> str:
        parts = [f"Generator[{self.yield_type}, {self.send_type}, {self.return_type}]"]
        if self.exhausted is True:
            parts.append("exhausted")
        elif self.exhausted is False:
            parts.append("not_exhausted")
        if self.yields_at_least is not None:
            parts.append(f"yields≥{self.yields_at_least}")
        return " & ".join(parts)


# ---------------------------------------------------------------------------
# ContainerRefinementTracker
# ---------------------------------------------------------------------------

ContainerRef = Union[
    ListRefinement, DictRefinement, SetRefinement, TupleRefinement, GeneratorRefinement
]


@dataclass
class ContainerRefinementTracker:
    """Track container refinements for all live variables."""

    refinements: Dict[str, ContainerRef] = field(default_factory=dict)

    # -- list ops ------------------------------------------------------------

    def track_list_op(
        self, var: str, method_name: str, args: List[Any]
    ) -> ContainerRefinementTracker:
        ref = self.refinements.get(var)
        if not isinstance(ref, ListRefinement):
            return self

        dispatch = {
            "append": lambda: ref.after_append(args[0] if args else AnyType()),
            "pop": lambda: ref.after_pop(args[0] if args else None),
            "insert": lambda: ref.after_insert(
                args[0] if args else 0, args[1] if len(args) > 1 else AnyType()
            ),
            "extend": lambda: ref.after_extend(
                args[0] if isinstance(args[0], ListRefinement) else ListRefinement()
            ) if args else ref,
            "sort": lambda: ref.after_sort(),
            "reverse": lambda: ref.after_reverse(),
            "clear": lambda: ref.after_clear(),
        }
        handler = dispatch.get(method_name)
        if handler is None:
            return self.invalidate(var)
        new_ref = handler()
        new_refs = dict(self.refinements)
        new_refs[var] = new_ref
        return ContainerRefinementTracker(refinements=new_refs)

    # -- dict ops ------------------------------------------------------------

    def track_dict_op(
        self, var: str, method_name: str, args: List[Any]
    ) -> ContainerRefinementTracker:
        ref = self.refinements.get(var)
        if not isinstance(ref, DictRefinement):
            return self

        new_ref: ContainerRef
        if method_name == "setitem" and len(args) >= 2:
            new_ref = ref.after_setitem(str(args[0]), args[1] if isinstance(args[1], PyType) else AnyType())
        elif method_name == "delitem" and args:
            new_ref = ref.after_delitem(str(args[0]))
        elif method_name == "update" and args and isinstance(args[0], DictRefinement):
            new_ref = ref.after_update(args[0])
        elif method_name == "pop" and args:
            new_ref, _ = ref.after_pop(str(args[0]))
        elif method_name == "clear":
            new_ref = ref.after_clear()
        elif method_name == "setdefault" and args:
            default = args[1] if len(args) > 1 and isinstance(args[1], PyType) else AnyType()
            new_ref = ref.after_setdefault(str(args[0]), default)
        else:
            return self.invalidate(var)

        new_refs = dict(self.refinements)
        new_refs[var] = new_ref
        return ContainerRefinementTracker(refinements=new_refs)

    # -- set ops -------------------------------------------------------------

    def track_set_op(
        self, var: str, method_name: str, args: List[Any]
    ) -> ContainerRefinementTracker:
        ref = self.refinements.get(var)
        if not isinstance(ref, SetRefinement):
            return self

        new_ref: ContainerRef
        if method_name == "add" and args:
            new_ref = ref.after_add(args[0] if isinstance(args[0], PyType) else AnyType())
        elif method_name == "discard" and args:
            new_ref = ref.after_discard(args[0])
        elif method_name == "remove" and args:
            new_ref = ref.after_remove(args[0])
        elif method_name == "clear":
            new_ref = ref.after_clear()
        elif method_name == "union" and args and isinstance(args[0], SetRefinement):
            new_ref = ref.after_union(args[0])
        elif method_name == "intersection" and args and isinstance(args[0], SetRefinement):
            new_ref = ref.after_intersection(args[0])
        elif method_name == "difference" and args and isinstance(args[0], SetRefinement):
            new_ref = ref.after_difference(args[0])
        else:
            return self.invalidate(var)

        new_refs = dict(self.refinements)
        new_refs[var] = new_ref
        return ContainerRefinementTracker(refinements=new_refs)

    # -- generic queries -----------------------------------------------------

    def get_refinement(self, var: str) -> Optional[ContainerRef]:
        return self.refinements.get(var)

    def invalidate(self, var: str) -> ContainerRefinementTracker:
        new_refs = dict(self.refinements)
        new_refs.pop(var, None)
        return ContainerRefinementTracker(refinements=new_refs)

    def join(self, other: ContainerRefinementTracker) -> ContainerRefinementTracker:
        merged: Dict[str, ContainerRef] = {}
        all_vars = set(self.refinements) | set(other.refinements)
        for v in all_vars:
            a = self.refinements.get(v)
            b = other.refinements.get(v)
            if a is None or b is None:
                continue
            if type(a) != type(b):
                continue
            if isinstance(a, ListRefinement) and isinstance(b, ListRefinement):
                merged[v] = a.join(b)
            elif isinstance(a, DictRefinement) and isinstance(b, DictRefinement):
                merged[v] = a.join(b)
            elif isinstance(a, SetRefinement) and isinstance(b, SetRefinement):
                merged[v] = a.join(b)
            elif isinstance(a, TupleRefinement) and isinstance(b, TupleRefinement):
                merged[v] = a.join(b)
            elif isinstance(a, GeneratorRefinement) and isinstance(b, GeneratorRefinement):
                merged[v] = a.join(b)
        return ContainerRefinementTracker(refinements=merged)

    def predicates_for(self, var: str) -> List[HeapPredicate]:
        ref = self.refinements.get(var)
        if ref is None:
            return []
        return ref.to_predicates(var)


# ---------------------------------------------------------------------------
# IterationResult
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Result of analysing a for-loop over a container."""

    element_type: PyType
    mutates_container: bool
    iteration_bound: Optional[LengthBound]
    accumulated_refinements: List[HeapPredicate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IterationAnalyzer
# ---------------------------------------------------------------------------

@dataclass
class IterationAnalyzer:
    """Analyse for-loop iteration patterns over refined containers."""

    def analyze_for_loop(
        self,
        iterable_ref: ContainerRef,
        body_mutations: List[Tuple[str, str, List[Any]]],
    ) -> IterationResult:
        """Analyse a ``for x in container`` loop.

        *body_mutations* is a list of ``(var, method, args)`` triples that
        describe the mutations performed inside the loop body.
        """
        elem_ty = self.element_type_from_container(iterable_ref)
        bound = self.iteration_count_bound(iterable_ref)
        container_var: Optional[str] = None
        for var, _, _ in body_mutations:
            container_var = var
            break
        mutates = False
        if container_var is not None:
            body_vars = {var for var, _, _ in body_mutations}
            mutates = self.detect_mutation_during_iteration(container_var, body_vars)
        accum: List[HeapPredicate] = []
        if isinstance(iterable_ref, ListRefinement) and iterable_ref.universal_predicate:
            accum.append(iterable_ref.universal_predicate)
        if isinstance(iterable_ref, ListRefinement):
            for pred in iterable_ref.element_predicates.values():
                accum.append(pred)
        return IterationResult(
            element_type=elem_ty,
            mutates_container=mutates,
            iteration_bound=bound,
            accumulated_refinements=accum,
        )

    def detect_mutation_during_iteration(
        self, container_var: str, body_vars: Set[str]
    ) -> bool:
        """Return *True* if the loop body mutates the container being iterated."""
        return container_var in body_vars

    def element_type_from_container(self, container_ref: ContainerRef) -> PyType:
        """Extract the element type from a container refinement."""
        if isinstance(container_ref, ListRefinement):
            return container_ref.element_type
        if isinstance(container_ref, SetRefinement):
            return container_ref.element_type
        if isinstance(container_ref, DictRefinement):
            return container_ref.key_type
        if isinstance(container_ref, TupleRefinement):
            if container_ref.element_types:
                result: PyType = NeverType()
                for t in container_ref.element_types:
                    result = _join_types(result, t)
                return result
            return AnyType()
        if isinstance(container_ref, GeneratorRefinement):
            return container_ref.yield_type
        return AnyType()

    def iteration_count_bound(
        self, container_ref: ContainerRef
    ) -> Optional[LengthBound]:
        """Return an upper-bound on the number of iterations, if known."""
        bounds: List[LengthBound] = []
        if isinstance(container_ref, (ListRefinement, DictRefinement, SetRefinement)):
            bounds = container_ref.length_bounds
        elif isinstance(container_ref, TupleRefinement):
            return LengthBound("==", len(container_ref.element_types))
        elif isinstance(container_ref, GeneratorRefinement):
            if container_ref.yields_at_least is not None:
                return LengthBound(">=", container_ref.yields_at_least)
            return None
        # Find tightest upper bound
        best: Optional[LengthBound] = None
        for b in bounds:
            if b.op == "==":
                return b
            if b.op in ("<=", "<"):
                if best is None:
                    best = b
                elif b.value < best.value:
                    best = b
        return best
