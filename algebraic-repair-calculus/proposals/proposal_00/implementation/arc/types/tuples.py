"""
Typed tuples and multiset (bag) data structures for the ARC algebra.

``TypedTuple`` enforces column types at construction time and supports
null handling.  ``MultiSet`` is a full bag implementation with algebraic
operations (union, intersection, difference, symmetric difference) and
multiplicity tracking, suitable for representing intermediate relations
in the delta-algebra engine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from typing import Any, Callable, Iterable, Iterator, Sequence

import attr
from attr import validators as v

from arc.types.base import (
    Column,
    ParameterisedType,
    SQLType,
    Schema,
    TypeCompatibility,
    WideningResult,
)
from arc.types.errors import (
    ARCError,
    ColumnNotFoundError,
    ErrorCode,
    SchemaError,
    TypeCastError,
    TypeMismatchError,
)


# =====================================================================
# Null sentinel
# =====================================================================

class _Null:
    """Singleton sentinel representing SQL NULL.

    Using a dedicated sentinel avoids ambiguity with Python ``None``
    (which we reserve for "value not provided").
    """

    _instance: _Null | None = None

    def __new__(cls) -> _Null:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NULL"

    def __str__(self) -> str:
        return "NULL"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Null)

    def __hash__(self) -> int:
        return hash("__arc_null__")

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return isinstance(other, _Null)


NULL = _Null()


def is_null(value: Any) -> bool:
    """Check whether *value* is the ARC NULL sentinel."""
    return isinstance(value, _Null)


# =====================================================================
# Type coercion helpers
# =====================================================================

def _coerce_value(value: Any, sql_type: ParameterisedType, nullable: bool) -> Any:
    """Attempt to coerce *value* into the target SQL type.

    Returns the (possibly coerced) value.  Raises :class:`TypeCastError`
    on failure.
    """
    if is_null(value) or value is None:
        if not nullable:
            raise TypeCastError(value, f"{sql_type} (NOT NULL)")
        return NULL

    base = sql_type.base

    # Integer family
    if base in (SQLType.SMALLINT, SQLType.INT, SQLType.BIGINT,
                SQLType.SERIAL, SQLType.BIGSERIAL):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise TypeCastError(value, str(sql_type))
        raise TypeCastError(value, str(sql_type))

    # Float family
    if base in (SQLType.REAL, SQLType.FLOAT, SQLType.DOUBLE):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeCastError(value, str(sql_type))
        raise TypeCastError(value, str(sql_type))

    # Decimal / Numeric
    if base in (SQLType.NUMERIC, SQLType.DECIMAL):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeCastError(value, str(sql_type))
        raise TypeCastError(value, str(sql_type))

    # String family
    if base in (SQLType.CHAR, SQLType.VARCHAR, SQLType.TEXT):
        s = str(value)
        if base == SQLType.CHAR and sql_type.params.length is not None:
            s = s[:sql_type.params.length].ljust(sql_type.params.length)
        elif base == SQLType.VARCHAR and sql_type.params.length is not None:
            if len(s) > sql_type.params.length:
                raise TypeCastError(value, f"VARCHAR({sql_type.params.length})")
        return s

    # Boolean
    if base == SQLType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            low = value.lower()
            if low in ("true", "t", "1", "yes", "y"):
                return True
            if low in ("false", "f", "0", "no", "n"):
                return False
            raise TypeCastError(value, "BOOLEAN")
        raise TypeCastError(value, "BOOLEAN")

    # For complex types (JSON, ARRAY, UUID, DATE, etc.) accept as-is
    return value


# =====================================================================
# TypedTuple
# =====================================================================

@attr.s(slots=True, hash=False, repr=False, eq=False)
class TypedTuple:
    """A single row with enforced column types and null handling.

    Construction validates each value against the schema.  Values are
    stored in column-order and are accessible by name or position.
    """

    _schema: Schema = attr.ib(validator=v.instance_of(Schema))
    _values: tuple[Any, ...] = attr.ib(converter=tuple)
    _hash_cache: int | None = attr.ib(default=None, init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        ncols = len(self._schema.columns)
        if len(self._values) != ncols:
            raise SchemaError(
                f"Expected {ncols} values, got {len(self._values)}",
                code=ErrorCode.SCHEMA_INVALID,
            )
        # Coerce and validate
        coerced: list[Any] = []
        for col, val in zip(self._schema.columns, self._values):
            coerced.append(_coerce_value(val, col.sql_type, col.nullable))
        object.__setattr__(self, "_values", tuple(coerced))

    # -- Access ---

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def values(self) -> tuple[Any, ...]:
        return self._values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        if isinstance(key, str):
            for i, col in enumerate(self._schema.columns):
                if col.name == key:
                    return self._values[i]
            raise ColumnNotFoundError(key, available=self._schema.column_list)
        raise TypeError(f"Key must be int or str, not {type(key).__name__}")

    def get(self, name: str, default: Any = NULL) -> Any:
        for i, col in enumerate(self._schema.columns):
            if col.name == name:
                return self._values[i]
        return default

    # -- Transformation ---

    def project(self, names: Sequence[str]) -> TypedTuple:
        new_schema = self._schema.project(names)
        new_values = tuple(self[name] for name in names)
        return TypedTuple(schema=new_schema, values=new_values)

    def extend(self, column: Column, value: Any) -> TypedTuple:
        new_schema = self._schema.add_column(column)
        coerced = _coerce_value(value, column.sql_type, column.nullable)
        return TypedTuple(schema=new_schema, values=self._values + (coerced,))

    def replace(self, name: str, value: Any) -> TypedTuple:
        col = self._schema[name]
        coerced = _coerce_value(value, col.sql_type, col.nullable)
        idx = next(i for i, c in enumerate(self._schema.columns) if c.name == name)
        new_values = self._values[:idx] + (coerced,) + self._values[idx + 1:]
        return TypedTuple(schema=self._schema, values=new_values)

    def rename(self, old_name: str, new_name: str) -> TypedTuple:
        new_schema = self._schema.rename_column(old_name, new_name)
        return TypedTuple(schema=new_schema, values=self._values)

    def drop(self, name: str) -> TypedTuple:
        idx = next(i for i, c in enumerate(self._schema.columns) if c.name == name)
        new_schema = self._schema.drop_column(name)
        new_values = self._values[:idx] + self._values[idx + 1:]
        return TypedTuple(schema=new_schema, values=new_values)

    # -- Null handling ---

    def null_columns(self) -> list[str]:
        return [
            col.name
            for col, val in zip(self._schema.columns, self._values)
            if is_null(val)
        ]

    def has_nulls(self) -> bool:
        return any(is_null(v) for v in self._values)

    def fill_nulls(self, defaults: dict[str, Any]) -> TypedTuple:
        new_values = list(self._values)
        for i, col in enumerate(self._schema.columns):
            if is_null(new_values[i]) and col.name in defaults:
                new_values[i] = _coerce_value(
                    defaults[col.name], col.sql_type, col.nullable,
                )
        return TypedTuple(schema=self._schema, values=tuple(new_values))

    # -- Comparison ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedTuple):
            return NotImplemented
        if self._schema.column_names != other._schema.column_names:
            return False
        return self._values == other._values

    def __hash__(self) -> int:
        if self._hash_cache is None:
            h = hash(self._values)
            object.__setattr__(self, "_hash_cache", h)
        return self._hash_cache  # type: ignore[return-value]

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{col.name}={val!r}"
            for col, val in zip(self._schema.columns, self._values)
        )
        return f"TypedTuple({pairs})"

    def __str__(self) -> str:
        pairs = ", ".join(
            f"{col.name}={val}" for col, val in zip(self._schema.columns, self._values)
        )
        return f"({pairs})"

    # -- Dict conversion ---

    def to_dict(self) -> dict[str, Any]:
        return {
            col.name: (None if is_null(val) else val)
            for col, val in zip(self._schema.columns, self._values)
        }

    @classmethod
    def from_dict(cls, schema: Schema, d: dict[str, Any]) -> TypedTuple:
        values = []
        for col in schema.columns:
            val = d.get(col.name, None)
            values.append(val)
        return cls(schema=schema, values=tuple(values))

    def to_list(self) -> list[Any]:
        return [None if is_null(v) else v for v in self._values]

    @classmethod
    def from_list(cls, schema: Schema, values: Sequence[Any]) -> TypedTuple:
        return cls(schema=schema, values=tuple(values))

    # -- JSON serialisation ---

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, schema: Schema, s: str) -> TypedTuple:
        d = json.loads(s)
        return cls.from_dict(schema, d)

    # -- Content hash ---

    def content_hash(self) -> str:
        h = hashlib.sha256(self.to_json().encode()).hexdigest()
        return h[:16]


# =====================================================================
# MultiSet (Bag)
# =====================================================================

class MultiSet:
    """A multiset (bag) of :class:`TypedTuple` values with full algebraic
    operations and multiplicity tracking.

    Internally backed by a :class:`Counter` mapping each tuple to its
    multiplicity for O(1) membership and efficient set operations.
    """

    __slots__ = ("_schema", "_counter", "_hash_cache")

    def __init__(
        self,
        schema: Schema,
        elements: Iterable[TypedTuple] | None = None,
        counter: Counter[TypedTuple] | None = None,
    ) -> None:
        self._schema = schema
        if counter is not None:
            self._counter = Counter(counter)
        elif elements is not None:
            self._counter: Counter[TypedTuple] = Counter()
            for elem in elements:
                if elem.schema.column_names != schema.column_names:
                    raise SchemaError(
                        "Tuple schema mismatch in MultiSet construction",
                        code=ErrorCode.SCHEMA_INVALID,
                    )
                self._counter[elem] += 1
        else:
            self._counter = Counter()
        self._hash_cache: int | None = None

    @property
    def schema(self) -> Schema:
        return self._schema

    # -- Size ---

    def __len__(self) -> int:
        return sum(self._counter.values())

    def cardinality(self) -> int:
        return len(self)

    def distinct_count(self) -> int:
        return len(self._counter)

    def is_empty(self) -> bool:
        return len(self._counter) == 0

    # -- Multiplicity ---

    def multiplicity(self, element: TypedTuple) -> int:
        return self._counter.get(element, 0)

    def contains(self, element: TypedTuple) -> bool:
        return element in self._counter

    def __contains__(self, element: TypedTuple) -> bool:
        return self.contains(element)

    def most_common(self, n: int | None = None) -> list[tuple[TypedTuple, int]]:
        return self._counter.most_common(n)

    # -- Iteration ---

    def __iter__(self) -> Iterator[TypedTuple]:
        for elem, count in self._counter.items():
            for _ in range(count):
                yield elem

    def distinct(self) -> Iterator[TypedTuple]:
        yield from self._counter.keys()

    def items(self) -> Iterator[tuple[TypedTuple, int]]:
        yield from self._counter.items()

    # -- Mutation (returns new MultiSet) ---

    def add(self, element: TypedTuple, count: int = 1) -> MultiSet:
        new_counter = Counter(self._counter)
        new_counter[element] += count
        return MultiSet(self._schema, counter=new_counter)

    def remove(self, element: TypedTuple, count: int = 1) -> MultiSet:
        new_counter = Counter(self._counter)
        if element in new_counter:
            new_counter[element] -= count
            if new_counter[element] <= 0:
                del new_counter[element]
        return MultiSet(self._schema, counter=new_counter)

    def remove_all(self, element: TypedTuple) -> MultiSet:
        new_counter = Counter(self._counter)
        if element in new_counter:
            del new_counter[element]
        return MultiSet(self._schema, counter=new_counter)

    def set_multiplicity(self, element: TypedTuple, count: int) -> MultiSet:
        new_counter = Counter(self._counter)
        if count <= 0:
            new_counter.pop(element, None)
        else:
            new_counter[element] = count
        return MultiSet(self._schema, counter=new_counter)

    # -- Algebraic operations (bag semantics) ---

    def union(self, other: MultiSet) -> MultiSet:
        """Bag union: multiplicity = max(m_self, m_other)."""
        self._check_compatible(other)
        result: Counter[TypedTuple] = Counter()
        all_keys = set(self._counter) | set(other._counter)
        for key in all_keys:
            result[key] = max(self._counter.get(key, 0), other._counter.get(key, 0))
        return MultiSet(self._schema, counter=result)

    def union_all(self, other: MultiSet) -> MultiSet:
        """Bag union all: multiplicity = m_self + m_other."""
        self._check_compatible(other)
        result = Counter(self._counter)
        result.update(other._counter)
        return MultiSet(self._schema, counter=result)

    def intersection(self, other: MultiSet) -> MultiSet:
        """Bag intersection: multiplicity = min(m_self, m_other)."""
        self._check_compatible(other)
        result: Counter[TypedTuple] = Counter()
        for key in self._counter:
            if key in other._counter:
                result[key] = min(self._counter[key], other._counter[key])
        return MultiSet(self._schema, counter=result)

    def difference(self, other: MultiSet) -> MultiSet:
        """Bag difference: multiplicity = max(0, m_self - m_other)."""
        self._check_compatible(other)
        result: Counter[TypedTuple] = Counter()
        for key, count in self._counter.items():
            remaining = count - other._counter.get(key, 0)
            if remaining > 0:
                result[key] = remaining
        return MultiSet(self._schema, counter=result)

    def symmetric_difference(self, other: MultiSet) -> MultiSet:
        """Symmetric bag difference: |m_self - m_other|."""
        self._check_compatible(other)
        result: Counter[TypedTuple] = Counter()
        all_keys = set(self._counter) | set(other._counter)
        for key in all_keys:
            diff = abs(self._counter.get(key, 0) - other._counter.get(key, 0))
            if diff > 0:
                result[key] = diff
        return MultiSet(self._schema, counter=result)

    # Operator overloads
    def __or__(self, other: MultiSet) -> MultiSet:
        return self.union(other)

    def __and__(self, other: MultiSet) -> MultiSet:
        return self.intersection(other)

    def __sub__(self, other: MultiSet) -> MultiSet:
        return self.difference(other)

    def __add__(self, other: MultiSet) -> MultiSet:
        return self.union_all(other)

    def __xor__(self, other: MultiSet) -> MultiSet:
        return self.symmetric_difference(other)

    # -- Relational operations ---

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> MultiSet:
        result: Counter[TypedTuple] = Counter()
        for elem, count in self._counter.items():
            if predicate(elem):
                result[elem] = count
        return MultiSet(self._schema, counter=result)

    def project(self, columns: Sequence[str]) -> MultiSet:
        new_schema = self._schema.project(columns)
        result: Counter[TypedTuple] = Counter()
        for elem, count in self._counter.items():
            projected = elem.project(columns)
            result[projected] += count
        return MultiSet(new_schema, counter=result)

    def map(self, func: Callable[[TypedTuple], TypedTuple], new_schema: Schema | None = None) -> MultiSet:
        target_schema = new_schema or self._schema
        result: Counter[TypedTuple] = Counter()
        for elem, count in self._counter.items():
            result[func(elem)] += count
        return MultiSet(target_schema, counter=result)

    def flat_map(
        self,
        func: Callable[[TypedTuple], Iterable[TypedTuple]],
        new_schema: Schema | None = None,
    ) -> MultiSet:
        target_schema = new_schema or self._schema
        result: Counter[TypedTuple] = Counter()
        for elem, count in self._counter.items():
            for new_elem in func(elem):
                result[new_elem] += count
        return MultiSet(target_schema, counter=result)

    def group_by(
        self,
        key_columns: Sequence[str],
    ) -> dict[TypedTuple, MultiSet]:
        """Group by key columns, returning a dict of key → sub-multiset."""
        key_schema = self._schema.project(key_columns)
        groups: dict[TypedTuple, Counter[TypedTuple]] = {}
        for elem, count in self._counter.items():
            key = elem.project(key_columns)
            if key not in groups:
                groups[key] = Counter()
            groups[key][elem] += count
        return {
            key: MultiSet(self._schema, counter=ctr)
            for key, ctr in groups.items()
        }

    def distinct_set(self) -> MultiSet:
        """Convert to a set (all multiplicities become 1)."""
        result: Counter[TypedTuple] = Counter()
        for key in self._counter:
            result[key] = 1
        return MultiSet(self._schema, counter=result)

    def cross_join(self, other: MultiSet, result_schema: Schema) -> MultiSet:
        """Cross (Cartesian) product of two multisets."""
        result: Counter[TypedTuple] = Counter()
        for left, l_count in self._counter.items():
            for right, r_count in other._counter.items():
                combined_values = left.values + right.values
                combined = TypedTuple(schema=result_schema, values=combined_values)
                result[combined] += l_count * r_count
        return MultiSet(result_schema, counter=result)

    # -- Aggregation ---

    def count(self) -> int:
        return len(self)

    def sum_column(self, column: str) -> float:
        total = 0.0
        for elem, cnt in self._counter.items():
            val = elem[column]
            if not is_null(val):
                total += float(val) * cnt
        return total

    def avg_column(self, column: str) -> float | None:
        total = 0.0
        count = 0
        for elem, cnt in self._counter.items():
            val = elem[column]
            if not is_null(val):
                total += float(val) * cnt
                count += cnt
        return total / count if count > 0 else None

    def min_column(self, column: str) -> Any:
        vals = [
            elem[column]
            for elem in self._counter
            if not is_null(elem[column])
        ]
        return min(vals) if vals else NULL

    def max_column(self, column: str) -> Any:
        vals = [
            elem[column]
            for elem in self._counter
            if not is_null(elem[column])
        ]
        return max(vals) if vals else NULL

    # -- Comparison ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiSet):
            return NotImplemented
        return self._counter == other._counter

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(frozenset(self._counter.items()))
        return self._hash_cache

    def is_subset_of(self, other: MultiSet) -> bool:
        for key, count in self._counter.items():
            if other._counter.get(key, 0) < count:
                return False
        return True

    def is_superset_of(self, other: MultiSet) -> bool:
        return other.is_subset_of(self)

    # -- Representation ---

    def __repr__(self) -> str:
        n = len(self)
        d = self.distinct_count()
        return f"MultiSet(rows={n}, distinct={d}, schema={self._schema.column_list})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for elem, count in self._counter.most_common(10):
            lines.append(f"  {elem} × {count}")
        if self.distinct_count() > 10:
            lines.append(f"  ... and {self.distinct_count() - 10} more distinct tuples")
        return "\n".join(lines)

    # -- Serialisation ---

    def to_dicts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for elem, count in self._counter.items():
            d = elem.to_dict()
            d["__multiplicity__"] = count
            result.append(d)
        return result

    @classmethod
    def from_dicts(cls, schema: Schema, dicts: list[dict[str, Any]]) -> MultiSet:
        counter: Counter[TypedTuple] = Counter()
        for d in dicts:
            mult = d.pop("__multiplicity__", 1)
            t = TypedTuple.from_dict(schema, d)
            counter[t] += mult
        return cls(schema, counter=counter)

    def to_json(self) -> str:
        data = {
            "schema": self._schema.to_dict(),
            "elements": self.to_dicts(),
        }
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, s: str) -> MultiSet:
        data = json.loads(s)
        schema = Schema.from_dict(data["schema"])
        return cls.from_dicts(schema, data["elements"])

    # -- Content hash ---

    def content_hash(self) -> str:
        sorted_items = sorted(
            (elem.content_hash(), count)
            for elem, count in self._counter.items()
        )
        h = hashlib.sha256(json.dumps(sorted_items).encode()).hexdigest()
        return h[:16]

    # -- Utilities ---

    def sample(self, n: int) -> list[TypedTuple]:
        """Return up to *n* elements (without replacement, from distinct set)."""
        elements = list(self._counter.keys())
        return elements[:n]

    def to_list(self) -> list[TypedTuple]:
        return list(self)

    def to_distinct_list(self) -> list[TypedTuple]:
        return list(self._counter.keys())

    @classmethod
    def empty(cls, schema: Schema) -> MultiSet:
        return cls(schema)

    @classmethod
    def singleton(cls, element: TypedTuple) -> MultiSet:
        return cls(element.schema, elements=[element])

    @classmethod
    def from_tuples(cls, schema: Schema, tuples: Sequence[TypedTuple]) -> MultiSet:
        return cls(schema, elements=tuples)

    @classmethod
    def from_rows(cls, schema: Schema, rows: Sequence[Sequence[Any]]) -> MultiSet:
        elements = [TypedTuple(schema=schema, values=tuple(row)) for row in rows]
        return cls(schema, elements=elements)

    # -- Internal ---

    def _check_compatible(self, other: MultiSet) -> None:
        if self._schema.column_names != other._schema.column_names:
            raise SchemaError(
                f"Incompatible schemas for MultiSet operation: "
                f"{self._schema.column_list} vs {other._schema.column_list}",
                code=ErrorCode.SCHEMA_INVALID,
            )


# =====================================================================
# Delta representation for multisets
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class MultiSetDelta:
    """Represents the difference between two multisets.

    Stored as inserted and deleted tuples with their multiplicities,
    suitable for incremental computation in the delta algebra.
    """

    schema: Schema = attr.ib(validator=v.instance_of(Schema))
    inserts: MultiSet = attr.ib()
    deletes: MultiSet = attr.ib()

    @classmethod
    def compute(cls, before: MultiSet, after: MultiSet) -> MultiSetDelta:
        """Compute the delta that transforms *before* into *after*."""
        inserts = after.difference(before)
        deletes = before.difference(after)
        return cls(schema=before.schema, inserts=inserts, deletes=deletes)

    @classmethod
    def empty(cls, schema: Schema) -> MultiSetDelta:
        return cls(
            schema=schema,
            inserts=MultiSet.empty(schema),
            deletes=MultiSet.empty(schema),
        )

    @property
    def is_empty(self) -> bool:
        return self.inserts.is_empty() and self.deletes.is_empty()

    @property
    def net_change(self) -> int:
        return self.inserts.cardinality() - self.deletes.cardinality()

    def apply(self, base: MultiSet) -> MultiSet:
        """Apply this delta to a base multiset: (base - deletes) + inserts."""
        return base.difference(self.deletes).union_all(self.inserts)

    def compose(self, other: MultiSetDelta) -> MultiSetDelta:
        """Compose two deltas: self then other.

        The result is a single delta equivalent to applying self then other.
        """
        # Inserts that survive: inserts(self) not deleted by other,
        # plus new inserts from other
        surviving_self_inserts = self.inserts.difference(other.deletes)
        combined_inserts = surviving_self_inserts.union_all(other.inserts)
        # Deletes: deletes from self (not re-inserted by other),
        # plus deletes from other (that aren't from self's inserts)
        surviving_self_deletes = self.deletes.difference(other.inserts)
        new_deletes = other.deletes.difference(self.inserts)
        combined_deletes = surviving_self_deletes.union_all(new_deletes)
        return MultiSetDelta(
            schema=self.schema,
            inserts=combined_inserts,
            deletes=combined_deletes,
        )

    def invert(self) -> MultiSetDelta:
        """Return the inverse delta (swap inserts and deletes)."""
        return MultiSetDelta(
            schema=self.schema,
            inserts=self.deletes,
            deletes=self.inserts,
        )

    def __repr__(self) -> str:
        return (
            f"MultiSetDelta(inserts={self.inserts.cardinality()}, "
            f"deletes={self.deletes.cardinality()}, "
            f"net={self.net_change})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "inserts": self.inserts.to_dicts(),
            "deletes": self.deletes.to_dicts(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiSetDelta:
        schema = Schema.from_dict(d["schema"])
        inserts = MultiSet.from_dicts(schema, d.get("inserts", []))
        deletes = MultiSet.from_dicts(schema, d.get("deletes", []))
        return cls(schema=schema, inserts=inserts, deletes=deletes)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, s: str) -> MultiSetDelta:
        return cls.from_dict(json.loads(s))
