"""
TLA-lite runtime value representation.

All TLA+ values are immutable and support comparison, hashing, and
serialization.  The type hierarchy mirrors the mathematical objects that
TLA+ expressions can denote:

    TLAValue
    ├── IntValue
    ├── BoolValue
    ├── StringValue
    ├── SetValue        (frozenset-based)
    ├── FunctionValue   (dict-based finite function)
    ├── TupleValue      (ordered)
    ├── RecordValue     (named fields)
    ├── SequenceValue   (1-indexed, with standard ops)
    └── ModelValue      (opaque CONSTANT values)
"""

from __future__ import annotations

import json
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Type-ordering tag used for cross-type comparison
# ---------------------------------------------------------------------------
_TYPE_ORDER = {
    "BoolValue": 0,
    "IntValue": 1,
    "StringValue": 2,
    "ModelValue": 3,
    "TupleValue": 4,
    "SequenceValue": 5,
    "RecordValue": 6,
    "FunctionValue": 7,
    "SetValue": 8,
}


class TLAValueError(Exception):
    """Raised when an operation on TLA+ values is invalid."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class TLAValue(ABC):
    """Abstract base class for every TLA+ runtime value."""

    @abstractmethod
    def sort_key(self) -> Tuple:
        """Return a tuple usable for deterministic ordering."""

    @abstractmethod
    def to_json(self) -> Any:
        """Serialize to a JSON-compatible Python object."""

    @classmethod
    def from_json(cls, obj: Any) -> "TLAValue":
        """Deserialize from a JSON-compatible Python object."""
        if not isinstance(obj, dict) or "_type" not in obj:
            raise TLAValueError(f"Cannot deserialize: {obj!r}")
        tag = obj["_type"]
        dispatch = {
            "Int": IntValue.from_json,
            "Bool": BoolValue.from_json,
            "String": StringValue.from_json,
            "Model": ModelValue.from_json,
            "Set": SetValue.from_json,
            "Function": FunctionValue.from_json,
            "Tuple": TupleValue.from_json,
            "Record": RecordValue.from_json,
            "Sequence": SequenceValue.from_json,
        }
        if tag not in dispatch:
            raise TLAValueError(f"Unknown value type tag: {tag!r}")
        return dispatch[tag](obj)

    @abstractmethod
    def pretty(self, indent: int = 0) -> str:
        """Human-readable string representation."""

    def __repr__(self) -> str:
        return self.pretty()

    def __lt__(self, other: "TLAValue") -> bool:
        if type(self) is type(other):
            return self.sort_key() < other.sort_key()
        return _TYPE_ORDER.get(type(self).__name__, 99) < _TYPE_ORDER.get(
            type(other).__name__, 99
        )

    def __le__(self, other: "TLAValue") -> bool:
        return self == other or self < other

    def __gt__(self, other: "TLAValue") -> bool:
        return not self <= other

    def __ge__(self, other: "TLAValue") -> bool:
        return not self < other

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# IntValue
# ---------------------------------------------------------------------------
class IntValue(TLAValue):
    __slots__ = ("val",)

    def __init__(self, val: int) -> None:
        self.val = val

    def sort_key(self) -> Tuple:
        return (1, self.val)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntValue) and self.val == other.val

    def __hash__(self) -> int:
        return hash(("Int", self.val))

    def to_json(self) -> dict:
        return {"_type": "Int", "val": self.val}

    @classmethod
    def from_json(cls, obj: dict) -> "IntValue":
        return cls(obj["val"])

    def pretty(self, indent: int = 0) -> str:
        return str(self.val)


# ---------------------------------------------------------------------------
# BoolValue
# ---------------------------------------------------------------------------
class BoolValue(TLAValue):
    __slots__ = ("val",)

    def __init__(self, val: bool) -> None:
        self.val = val

    def sort_key(self) -> Tuple:
        return (0, int(self.val))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoolValue) and self.val == other.val

    def __hash__(self) -> int:
        return hash(("Bool", self.val))

    def to_json(self) -> dict:
        return {"_type": "Bool", "val": self.val}

    @classmethod
    def from_json(cls, obj: dict) -> "BoolValue":
        return cls(obj["val"])

    def pretty(self, indent: int = 0) -> str:
        return "TRUE" if self.val else "FALSE"


# ---------------------------------------------------------------------------
# StringValue
# ---------------------------------------------------------------------------
class StringValue(TLAValue):
    __slots__ = ("val",)

    def __init__(self, val: str) -> None:
        self.val = val

    def sort_key(self) -> Tuple:
        return (2, self.val)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StringValue) and self.val == other.val

    def __hash__(self) -> int:
        return hash(("String", self.val))

    def to_json(self) -> dict:
        return {"_type": "String", "val": self.val}

    @classmethod
    def from_json(cls, obj: dict) -> "StringValue":
        return cls(obj["val"])

    def pretty(self, indent: int = 0) -> str:
        return f'"{self.val}"'


# ---------------------------------------------------------------------------
# ModelValue – opaque constants defined via CONSTANT declarations
# ---------------------------------------------------------------------------
class ModelValue(TLAValue):
    __slots__ = ("name", "sort_name")

    def __init__(self, name: str, sort_name: Optional[str] = None) -> None:
        self.name = name
        self.sort_name = sort_name or name

    def sort_key(self) -> Tuple:
        return (3, self.sort_name, self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModelValue) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Model", self.name))

    def to_json(self) -> dict:
        d: dict = {"_type": "Model", "name": self.name}
        if self.sort_name != self.name:
            d["sort"] = self.sort_name
        return d

    @classmethod
    def from_json(cls, obj: dict) -> "ModelValue":
        return cls(obj["name"], obj.get("sort"))

    def pretty(self, indent: int = 0) -> str:
        return self.name


# ---------------------------------------------------------------------------
# SetValue – immutable, frozenset-backed
# ---------------------------------------------------------------------------
class SetValue(TLAValue):
    __slots__ = ("_fs",)

    def __init__(self, elements: Iterable[TLAValue] = ()) -> None:
        self._fs: FrozenSet[TLAValue] = frozenset(elements)

    @property
    def elements(self) -> FrozenSet[TLAValue]:
        return self._fs

    # --- set operations ---------------------------------------------------

    def union(self, other: "SetValue") -> "SetValue":
        return SetValue(self._fs | other._fs)

    def intersect(self, other: "SetValue") -> "SetValue":
        return SetValue(self._fs & other._fs)

    def difference(self, other: "SetValue") -> "SetValue":
        return SetValue(self._fs - other._fs)

    def is_subset(self, other: "SetValue") -> bool:
        return self._fs <= other._fs

    def contains(self, elem: TLAValue) -> bool:
        return elem in self._fs

    def cardinality(self) -> int:
        return len(self._fs)

    def powerset(self) -> "SetValue":
        elems = sorted(self._fs)
        subsets: List[TLAValue] = []
        for r in range(len(elems) + 1):
            for combo in itertools.combinations(elems, r):
                subsets.append(SetValue(combo))
        return SetValue(subsets)

    def big_union(self) -> "SetValue":
        """UNION S – flatten a set of sets."""
        result: Set[TLAValue] = set()
        for elem in self._fs:
            if not isinstance(elem, SetValue):
                raise TLAValueError(f"UNION requires set of sets, got {type(elem).__name__}")
            result.update(elem._fs)
        return SetValue(result)

    def cross(self, other: "SetValue") -> "SetValue":
        """Cartesian product S \\times T, returning set of TupleValue."""
        pairs: List[TLAValue] = []
        for a in sorted(self._fs):
            for b in sorted(other._fs):
                pairs.append(TupleValue((a, b)))
        return SetValue(pairs)

    def __iter__(self) -> Iterator[TLAValue]:
        return iter(sorted(self._fs))

    def __len__(self) -> int:
        return len(self._fs)

    # --- comparison / hash ------------------------------------------------

    def sort_key(self) -> Tuple:
        return (8, tuple(sorted(e.sort_key() for e in self._fs)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SetValue) and self._fs == other._fs

    def __hash__(self) -> int:
        return hash(("Set", self._fs))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        return {"_type": "Set", "elements": [e.to_json() for e in sorted(self._fs)]}

    @classmethod
    def from_json(cls, obj: dict) -> "SetValue":
        return cls(TLAValue.from_json(e) for e in obj["elements"])

    def pretty(self, indent: int = 0) -> str:
        if not self._fs:
            return "{}"
        inner = ", ".join(e.pretty(indent) for e in sorted(self._fs))
        return "{" + inner + "}"


# ---------------------------------------------------------------------------
# FunctionValue – finite function  [domain -> range]
# ---------------------------------------------------------------------------
class FunctionValue(TLAValue):
    """A finite function mapping TLAValue keys to TLAValue results.

    Internally backed by a frozenset of (key, value) pairs so that the
    function itself is hashable and comparable.
    """

    __slots__ = ("_mapping",)

    def __init__(self, mapping: Dict[TLAValue, TLAValue] | None = None,
                 pairs: Iterable[Tuple[TLAValue, TLAValue]] | None = None) -> None:
        if pairs is not None:
            self._mapping: Dict[TLAValue, TLAValue] = dict(pairs)
        elif mapping is not None:
            self._mapping = dict(mapping)
        else:
            self._mapping = {}

    # --- operations -------------------------------------------------------

    def apply(self, arg: TLAValue) -> TLAValue:
        if arg not in self._mapping:
            raise TLAValueError(
                f"Function application error: {arg.pretty()} not in DOMAIN"
            )
        return self._mapping[arg]

    def domain(self) -> SetValue:
        return SetValue(self._mapping.keys())

    def range(self) -> SetValue:
        return SetValue(self._mapping.values())

    def except_update(self, key: TLAValue, val: TLAValue) -> "FunctionValue":
        """[f EXCEPT ![k] = v]"""
        new_map = dict(self._mapping)
        new_map[key] = val
        return FunctionValue(new_map)

    def except_multi(self, updates: Iterable[Tuple[TLAValue, TLAValue]]) -> "FunctionValue":
        """[f EXCEPT ![k1] = v1, ![k2] = v2, ...]"""
        new_map = dict(self._mapping)
        for k, v in updates:
            new_map[k] = v
        return FunctionValue(new_map)

    @property
    def mapping(self) -> Dict[TLAValue, TLAValue]:
        return dict(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    # --- comparison / hash ------------------------------------------------

    def sort_key(self) -> Tuple:
        items = tuple(sorted((k.sort_key(), v.sort_key()) for k, v in self._mapping.items()))
        return (7, items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FunctionValue) and self._mapping == other._mapping

    def __hash__(self) -> int:
        return hash(("Func", frozenset(
            (k, v) for k, v in self._mapping.items()
        )))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        pairs = [
            {"key": k.to_json(), "val": v.to_json()}
            for k, v in sorted(self._mapping.items(), key=lambda kv: kv[0].sort_key())
        ]
        return {"_type": "Function", "pairs": pairs}

    @classmethod
    def from_json(cls, obj: dict) -> "FunctionValue":
        mapping: Dict[TLAValue, TLAValue] = {}
        for p in obj["pairs"]:
            mapping[TLAValue.from_json(p["key"])] = TLAValue.from_json(p["val"])
        return cls(mapping)

    def pretty(self, indent: int = 0) -> str:
        if not self._mapping:
            return "<<>>"
        parts: List[str] = []
        for k in sorted(self._mapping):
            parts.append(f"{k.pretty(indent)} :> {self._mapping[k].pretty(indent)}")
        return "(" + " @@ ".join(parts) + ")"


# ---------------------------------------------------------------------------
# TupleValue – fixed-length ordered tuple
# ---------------------------------------------------------------------------
class TupleValue(TLAValue):
    __slots__ = ("_elems",)

    def __init__(self, elements: Iterable[TLAValue] = ()) -> None:
        self._elems: Tuple[TLAValue, ...] = tuple(elements)

    @property
    def elements(self) -> Tuple[TLAValue, ...]:
        return self._elems

    def index(self, i: int) -> TLAValue:
        """1-based indexing as in TLA+."""
        if i < 1 or i > len(self._elems):
            raise TLAValueError(f"Tuple index {i} out of range [1..{len(self._elems)}]")
        return self._elems[i - 1]

    def __len__(self) -> int:
        return len(self._elems)

    # --- comparison / hash ------------------------------------------------

    def sort_key(self) -> Tuple:
        return (4, tuple(e.sort_key() for e in self._elems))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleValue) and self._elems == other._elems

    def __hash__(self) -> int:
        return hash(("Tuple", self._elems))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        return {"_type": "Tuple", "elements": [e.to_json() for e in self._elems]}

    @classmethod
    def from_json(cls, obj: dict) -> "TupleValue":
        return cls(TLAValue.from_json(e) for e in obj["elements"])

    def pretty(self, indent: int = 0) -> str:
        inner = ", ".join(e.pretty(indent) for e in self._elems)
        return f"<<{inner}>>"


# ---------------------------------------------------------------------------
# RecordValue – named fields (syntactic sugar over functions from strings)
# ---------------------------------------------------------------------------
class RecordValue(TLAValue):
    __slots__ = ("_fields",)

    def __init__(self, fields: Dict[str, TLAValue] | None = None) -> None:
        self._fields: Dict[str, TLAValue] = dict(fields) if fields else {}

    @property
    def fields(self) -> Dict[str, TLAValue]:
        return dict(self._fields)

    def field_names(self) -> FrozenSet[str]:
        return frozenset(self._fields.keys())

    def access(self, name: str) -> TLAValue:
        if name not in self._fields:
            raise TLAValueError(f"Record has no field '{name}'. Fields: {sorted(self._fields)}")
        return self._fields[name]

    def except_update(self, name: str, val: TLAValue) -> "RecordValue":
        """[r EXCEPT !.field = val]"""
        new_fields = dict(self._fields)
        if name not in new_fields:
            raise TLAValueError(f"EXCEPT on non-existent field '{name}'")
        new_fields[name] = val
        return RecordValue(new_fields)

    def except_multi(self, updates: Dict[str, TLAValue]) -> "RecordValue":
        new_fields = dict(self._fields)
        for name, val in updates.items():
            if name not in new_fields:
                raise TLAValueError(f"EXCEPT on non-existent field '{name}'")
            new_fields[name] = val
        return RecordValue(new_fields)

    def to_function(self) -> FunctionValue:
        """Convert to a FunctionValue keyed by StringValue."""
        mapping = {StringValue(k): v for k, v in self._fields.items()}
        return FunctionValue(mapping)

    def __len__(self) -> int:
        return len(self._fields)

    # --- comparison / hash ------------------------------------------------

    def sort_key(self) -> Tuple:
        items = tuple(sorted((k, v.sort_key()) for k, v in self._fields.items()))
        return (6, items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecordValue) and self._fields == other._fields

    def __hash__(self) -> int:
        return hash(("Record", frozenset(
            (k, v) for k, v in self._fields.items()
        )))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        fields = {k: v.to_json() for k, v in sorted(self._fields.items())}
        return {"_type": "Record", "fields": fields}

    @classmethod
    def from_json(cls, obj: dict) -> "RecordValue":
        return cls({k: TLAValue.from_json(v) for k, v in obj["fields"].items()})

    def pretty(self, indent: int = 0) -> str:
        if not self._fields:
            return "[  ]"
        parts = [f"{k} |-> {v.pretty(indent)}" for k, v in sorted(self._fields.items())]
        return "[" + ", ".join(parts) + "]"


# ---------------------------------------------------------------------------
# SequenceValue – 1-indexed sequence with standard TLA+ operations
# ---------------------------------------------------------------------------
class SequenceValue(TLAValue):
    """A TLA+ sequence: a function from 1..n to values.

    Stored internally as a Python tuple for immutability and hashing.
    """

    __slots__ = ("_elems",)

    def __init__(self, elements: Iterable[TLAValue] = ()) -> None:
        self._elems: Tuple[TLAValue, ...] = tuple(elements)

    @property
    def elements(self) -> Tuple[TLAValue, ...]:
        return self._elems

    # --- TLA+ Sequence operations -----------------------------------------

    def length(self) -> int:
        return len(self._elems)

    def head(self) -> TLAValue:
        if not self._elems:
            raise TLAValueError("Head of empty sequence")
        return self._elems[0]

    def tail(self) -> "SequenceValue":
        if not self._elems:
            raise TLAValueError("Tail of empty sequence")
        return SequenceValue(self._elems[1:])

    def append(self, val: TLAValue) -> "SequenceValue":
        return SequenceValue(self._elems + (val,))

    def concat(self, other: "SequenceValue") -> "SequenceValue":
        return SequenceValue(self._elems + other._elems)

    def sub_seq(self, m: int, n: int) -> "SequenceValue":
        """SubSeq(s, m, n) – 1-indexed, inclusive on both ends."""
        if m < 1:
            m = 1
        if n > len(self._elems):
            n = len(self._elems)
        if m > n:
            return SequenceValue()
        return SequenceValue(self._elems[m - 1 : n])

    def select_seq(self, pred: Callable[[TLAValue], bool]) -> "SequenceValue":
        return SequenceValue(e for e in self._elems if pred(e))

    def index(self, i: int) -> TLAValue:
        """1-based indexing."""
        if i < 1 or i > len(self._elems):
            raise TLAValueError(
                f"Sequence index {i} out of range [1..{len(self._elems)}]"
            )
        return self._elems[i - 1]

    def to_function(self) -> FunctionValue:
        """View as a function 1..Len(s) -> values."""
        mapping = {IntValue(i + 1): v for i, v in enumerate(self._elems)}
        return FunctionValue(mapping)

    def __len__(self) -> int:
        return len(self._elems)

    def __iter__(self) -> Iterator[TLAValue]:
        return iter(self._elems)

    # --- comparison / hash ------------------------------------------------

    def sort_key(self) -> Tuple:
        return (5, tuple(e.sort_key() for e in self._elems))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SequenceValue):
            return self._elems == other._elems
        return False

    def __hash__(self) -> int:
        return hash(("Seq", self._elems))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        return {"_type": "Sequence", "elements": [e.to_json() for e in self._elems]}

    @classmethod
    def from_json(cls, obj: dict) -> "SequenceValue":
        return cls(TLAValue.from_json(e) for e in obj["elements"])

    def pretty(self, indent: int = 0) -> str:
        inner = ", ".join(e.pretty(indent) for e in self._elems)
        return f"<<{inner}>>"


# ---------------------------------------------------------------------------
# Utility: value_from_python – convenience constructor
# ---------------------------------------------------------------------------
def value_from_python(obj: Any) -> TLAValue:
    """Convert a plain Python object to a TLAValue (best-effort)."""
    if isinstance(obj, TLAValue):
        return obj
    if isinstance(obj, bool):
        return BoolValue(obj)
    if isinstance(obj, int):
        return IntValue(obj)
    if isinstance(obj, str):
        return StringValue(obj)
    if isinstance(obj, (set, frozenset)):
        return SetValue(value_from_python(e) for e in obj)
    if isinstance(obj, dict):
        return RecordValue({k: value_from_python(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return SequenceValue(value_from_python(e) for e in obj)
    raise TLAValueError(f"Cannot convert {type(obj).__name__} to TLAValue")


def values_to_json_string(val: TLAValue) -> str:
    """Serialize a TLAValue to a JSON string."""
    return json.dumps(val.to_json(), sort_keys=True)


def value_from_json_string(s: str) -> TLAValue:
    """Deserialize a TLAValue from a JSON string."""
    return TLAValue.from_json(json.loads(s))
