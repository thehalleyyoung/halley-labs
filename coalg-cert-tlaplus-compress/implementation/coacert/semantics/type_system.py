"""
Runtime type checking for TLA-lite values.

TLA+ is untyped in theory, but for model checking and error reporting it
is useful to track and verify the *sorts* (runtime types) of values.

The ``TypeChecker`` can:
* Check whether a value conforms to an expected type.
* Infer the type of a value.
* Infer domains for quantified variables.
* Report type errors with context.
"""

from __future__ import annotations

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

from .values import (
    TLAValue,
    TLAValueError,
    IntValue,
    BoolValue,
    StringValue,
    SetValue,
    FunctionValue,
    TupleValue,
    RecordValue,
    SequenceValue,
    ModelValue,
)


# ===================================================================
# Type representation
# ===================================================================

class TypeKind(Enum):
    ANY = auto()
    BOOL = auto()
    INT = auto()
    STRING = auto()
    MODEL = auto()
    SET = auto()
    FUNCTION = auto()
    TUPLE = auto()
    RECORD = auto()
    SEQUENCE = auto()
    UNION = auto()
    NONE = auto()   # empty / bottom


@dataclass(frozen=True)
class TLAType:
    """A runtime type descriptor for TLA+ values."""
    kind: TypeKind
    element_type: Optional["TLAType"] = None       # for SET, SEQUENCE
    domain_type: Optional["TLAType"] = None         # for FUNCTION
    range_type: Optional["TLAType"] = None          # for FUNCTION
    field_types: Optional[Tuple[Tuple[str, "TLAType"], ...]] = None  # for RECORD
    tuple_types: Optional[Tuple["TLAType", ...]] = None  # for TUPLE
    union_members: Optional[Tuple["TLAType", ...]] = None  # for UNION
    model_sort: Optional[str] = None                 # for MODEL

    def __repr__(self) -> str:
        return self.pretty()

    def pretty(self) -> str:
        k = self.kind
        if k is TypeKind.ANY:
            return "Any"
        if k is TypeKind.NONE:
            return "None"
        if k is TypeKind.BOOL:
            return "BOOLEAN"
        if k is TypeKind.INT:
            return "Int"
        if k is TypeKind.STRING:
            return "STRING"
        if k is TypeKind.MODEL:
            return f"Model({self.model_sort})" if self.model_sort else "Model"
        if k is TypeKind.SET:
            inner = self.element_type.pretty() if self.element_type else "?"
            return f"Set({inner})"
        if k is TypeKind.FUNCTION:
            d = self.domain_type.pretty() if self.domain_type else "?"
            r = self.range_type.pretty() if self.range_type else "?"
            return f"[{d} -> {r}]"
        if k is TypeKind.TUPLE:
            if self.tuple_types:
                parts = ", ".join(t.pretty() for t in self.tuple_types)
                return f"<<{parts}>>"
            return "<<>>"
        if k is TypeKind.RECORD:
            if self.field_types:
                parts = ", ".join(f"{n}: {t.pretty()}" for n, t in self.field_types)
                return f"[{parts}]"
            return "[]"
        if k is TypeKind.SEQUENCE:
            inner = self.element_type.pretty() if self.element_type else "?"
            return f"Seq({inner})"
        if k is TypeKind.UNION:
            if self.union_members:
                parts = " | ".join(t.pretty() for t in self.union_members)
                return f"({parts})"
            return "()"
        return f"Type({k.name})"


# --- Constructors for common types ----------------------------------------

ANY_TYPE = TLAType(TypeKind.ANY)
BOOL_TYPE = TLAType(TypeKind.BOOL)
INT_TYPE = TLAType(TypeKind.INT)
STRING_TYPE = TLAType(TypeKind.STRING)
NONE_TYPE = TLAType(TypeKind.NONE)

def set_type(elem: TLAType = ANY_TYPE) -> TLAType:
    return TLAType(TypeKind.SET, element_type=elem)

def function_type(dom: TLAType = ANY_TYPE, rng: TLAType = ANY_TYPE) -> TLAType:
    return TLAType(TypeKind.FUNCTION, domain_type=dom, range_type=rng)

def tuple_type(*elems: TLAType) -> TLAType:
    return TLAType(TypeKind.TUPLE, tuple_types=elems)

def record_type(**fields: TLAType) -> TLAType:
    return TLAType(TypeKind.RECORD, field_types=tuple(sorted(fields.items())))

def sequence_type(elem: TLAType = ANY_TYPE) -> TLAType:
    return TLAType(TypeKind.SEQUENCE, element_type=elem)

def union_type(*members: TLAType) -> TLAType:
    flat: List[TLAType] = []
    for m in members:
        if m.kind is TypeKind.UNION and m.union_members:
            flat.extend(m.union_members)
        else:
            flat.append(m)
    unique = list(dict.fromkeys(flat))
    if len(unique) == 1:
        return unique[0]
    return TLAType(TypeKind.UNION, union_members=tuple(unique))

def model_type(sort: str) -> TLAType:
    return TLAType(TypeKind.MODEL, model_sort=sort)


# ===================================================================
# Type error
# ===================================================================

class TypeError(TLAValueError):
    """Raised when a runtime type check fails."""

    def __init__(self, msg: str, value: Optional[TLAValue] = None,
                 expected: Optional[TLAType] = None,
                 actual: Optional[TLAType] = None) -> None:
        parts = [msg]
        if value is not None:
            parts.append(f"  Value: {value.pretty()}")
        if expected is not None:
            parts.append(f"  Expected: {expected.pretty()}")
        if actual is not None:
            parts.append(f"  Actual: {actual.pretty()}")
        super().__init__("\n".join(parts))
        self.value = value
        self.expected = expected
        self.actual = actual


# ===================================================================
# TypeChecker
# ===================================================================

class TypeChecker:
    """Runtime type checker for TLA+ values."""

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict
        self._errors: List[TypeError] = []

    @property
    def errors(self) -> List[TypeError]:
        return list(self._errors)

    def clear_errors(self) -> None:
        self._errors.clear()

    # --- inference --------------------------------------------------------

    def infer_value_type(self, value: TLAValue) -> TLAType:
        """Infer the runtime type of a value."""
        if isinstance(value, BoolValue):
            return BOOL_TYPE
        if isinstance(value, IntValue):
            return INT_TYPE
        if isinstance(value, StringValue):
            return STRING_TYPE
        if isinstance(value, ModelValue):
            return model_type(value.sort_name)

        if isinstance(value, SetValue):
            if value.cardinality() == 0:
                return set_type(NONE_TYPE)
            elem_types = [self.infer_value_type(e) for e in value]
            return set_type(self._join_types(elem_types))

        if isinstance(value, FunctionValue):
            if len(value) == 0:
                return function_type(NONE_TYPE, NONE_TYPE)
            dom_types = [self.infer_value_type(k) for k in value.domain()]
            rng_types = [self.infer_value_type(v) for v in value.range()]
            return function_type(
                self._join_types(dom_types),
                self._join_types(rng_types),
            )

        if isinstance(value, TupleValue):
            return tuple_type(*(self.infer_value_type(e) for e in value.elements))

        if isinstance(value, RecordValue):
            ftypes = {
                name: self.infer_value_type(val)
                for name, val in value.fields.items()
            }
            return record_type(**ftypes)

        if isinstance(value, SequenceValue):
            if value.length() == 0:
                return sequence_type(NONE_TYPE)
            elem_types = [self.infer_value_type(e) for e in value]
            return sequence_type(self._join_types(elem_types))

        return ANY_TYPE

    # --- checking ---------------------------------------------------------

    def check_value_type(self, value: TLAValue, expected: TLAType) -> bool:
        """Check if *value* conforms to *expected* type. Returns True if ok."""
        actual = self.infer_value_type(value)
        result = self._is_subtype(actual, expected)
        if not result and self._strict:
            err = TypeError(
                "Type check failed",
                value=value, expected=expected, actual=actual,
            )
            self._errors.append(err)
        return result

    def check_int(self, value: TLAValue) -> bool:
        return isinstance(value, IntValue)

    def check_bool(self, value: TLAValue) -> bool:
        return isinstance(value, BoolValue)

    def check_string(self, value: TLAValue) -> bool:
        return isinstance(value, StringValue)

    def check_set(self, value: TLAValue) -> bool:
        return isinstance(value, SetValue)

    def check_function(self, value: TLAValue) -> bool:
        return isinstance(value, FunctionValue)

    def check_sequence(self, value: TLAValue) -> bool:
        return isinstance(value, SequenceValue)

    def check_record(self, value: TLAValue) -> bool:
        return isinstance(value, RecordValue)

    def check_tuple(self, value: TLAValue) -> bool:
        return isinstance(value, TupleValue)

    # --- domain inference for quantified variables -----------------------

    def infer_domain_type(self, domain_value: TLAValue) -> TLAType:
        """Given a set used as a quantification domain, infer element type."""
        if isinstance(domain_value, SetValue):
            if domain_value.cardinality() == 0:
                return NONE_TYPE
            elem_types = [self.infer_value_type(e) for e in domain_value]
            return self._join_types(elem_types)
        return ANY_TYPE

    def check_domain_membership(self, value: TLAValue, domain: SetValue) -> bool:
        """Check if *value* is a member of *domain* set."""
        return domain.contains(value)

    # --- subtyping / type compatibility -----------------------------------

    def _is_subtype(self, actual: TLAType, expected: TLAType) -> bool:
        """Check if *actual* type is compatible with *expected*."""
        if expected.kind is TypeKind.ANY:
            return True
        if actual.kind is TypeKind.NONE:
            return True
        if actual.kind is expected.kind:
            return self._check_same_kind(actual, expected)
        if expected.kind is TypeKind.UNION and expected.union_members:
            return any(self._is_subtype(actual, m) for m in expected.union_members)
        if actual.kind is TypeKind.UNION and actual.union_members:
            return all(self._is_subtype(m, expected) for m in actual.union_members)
        # Int is a subtype of Any (already handled above)
        return False

    def _check_same_kind(self, actual: TLAType, expected: TLAType) -> bool:
        """Deep structural check for same-kind types."""
        k = actual.kind

        if k in (TypeKind.BOOL, TypeKind.INT, TypeKind.STRING):
            return True

        if k is TypeKind.MODEL:
            if expected.model_sort is None:
                return True
            return actual.model_sort == expected.model_sort

        if k is TypeKind.SET:
            if actual.element_type and expected.element_type:
                return self._is_subtype(actual.element_type, expected.element_type)
            return True

        if k is TypeKind.FUNCTION:
            dom_ok = True
            rng_ok = True
            if actual.domain_type and expected.domain_type:
                dom_ok = self._is_subtype(actual.domain_type, expected.domain_type)
            if actual.range_type and expected.range_type:
                rng_ok = self._is_subtype(actual.range_type, expected.range_type)
            return dom_ok and rng_ok

        if k is TypeKind.TUPLE:
            if actual.tuple_types and expected.tuple_types:
                if len(actual.tuple_types) != len(expected.tuple_types):
                    return False
                return all(
                    self._is_subtype(a, e)
                    for a, e in zip(actual.tuple_types, expected.tuple_types)
                )
            return True

        if k is TypeKind.RECORD:
            if actual.field_types and expected.field_types:
                exp_fields = dict(expected.field_types)
                act_fields = dict(actual.field_types)
                for fname, ftype in exp_fields.items():
                    if fname not in act_fields:
                        return False
                    if not self._is_subtype(act_fields[fname], ftype):
                        return False
                return True
            return True

        if k is TypeKind.SEQUENCE:
            if actual.element_type and expected.element_type:
                return self._is_subtype(actual.element_type, expected.element_type)
            return True

        return True

    # --- type join (least upper bound) ------------------------------------

    def _join_types(self, types: List[TLAType]) -> TLAType:
        """Compute the least upper bound of a list of types."""
        if not types:
            return NONE_TYPE

        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]

        # Filter out NONE
        non_none = [t for t in unique if t.kind is not TypeKind.NONE]
        if not non_none:
            return NONE_TYPE
        if len(non_none) == 1:
            return non_none[0]

        # If all are the same kind, try to join structurally
        kinds = set(t.kind for t in non_none)
        if len(kinds) == 1:
            k = non_none[0].kind
            if k in (TypeKind.BOOL, TypeKind.INT, TypeKind.STRING):
                return non_none[0]
            if k is TypeKind.SET:
                elem_types = [t.element_type for t in non_none if t.element_type]
                if elem_types:
                    return set_type(self._join_types(elem_types))
                return set_type()
            if k is TypeKind.SEQUENCE:
                elem_types = [t.element_type for t in non_none if t.element_type]
                if elem_types:
                    return sequence_type(self._join_types(elem_types))
                return sequence_type()

        return union_type(*non_none)

    # --- coercion ---------------------------------------------------------

    def coerce(self, value: TLAValue, target: TLAType) -> Optional[TLAValue]:
        """Attempt to coerce *value* to *target* type. Returns None if impossible."""
        if self.check_value_type(value, target):
            return value

        # Tuple <-> Sequence coercion
        if target.kind is TypeKind.SEQUENCE and isinstance(value, TupleValue):
            return SequenceValue(value.elements)
        if target.kind is TypeKind.TUPLE and isinstance(value, SequenceValue):
            return TupleValue(value.elements)

        # Record -> Function coercion
        if target.kind is TypeKind.FUNCTION and isinstance(value, RecordValue):
            return value.to_function()

        # Sequence -> Function coercion
        if target.kind is TypeKind.FUNCTION and isinstance(value, SequenceValue):
            return value.to_function()

        return None
