"""
Data Delta Group (Δ_D)
======================

Implements the data delta group from the three-sorted delta algebra.
Data deltas represent changes to relation instances with multiset semantics.

Algebraic properties:
- Group: (Δ_D, ∘, ⁻¹, 𝟎) with composition, inverse, and zero element
- Multiset semantics: tuples carry multiplicities
- Normalization: INSERT+DELETE of same tuple cancels, UPDATEs merge

Operations: INSERT, DELETE, UPDATE (with multiset multiplicities)
"""

from __future__ import annotations

import copy
import hashlib
import itertools
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions
# ---------------------------------------------------------------------------

class SQLType(Enum):
    """Supported SQL column types (mirror of schema_delta.SQLType)."""
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    SMALLINT = "SMALLINT"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    DECIMAL = "DECIMAL"
    NUMERIC = "NUMERIC"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    CHAR = "CHAR"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMPTZ = "TIMESTAMPTZ"
    TIME = "TIME"
    INTERVAL = "INTERVAL"
    JSON = "JSON"
    JSONB = "JSONB"
    UUID = "UUID"
    BYTEA = "BYTEA"
    ARRAY = "ARRAY"
    NULL = "NULL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Typed Tuple (Multiset Element)
# ---------------------------------------------------------------------------

class TypedTuple:
    """
    A tuple of typed values representing a row in a relation.

    Values are stored as a mapping from column name to value.
    Supports hashing and equality for multiset membership.
    """

    __slots__ = ("_values", "_hash_cache")

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values: Dict[str, Any] = dict(values)
        self._hash_cache: Optional[int] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> TypedTuple:
        return TypedTuple(d)

    @staticmethod
    def from_row(columns: Sequence[str], values: Sequence[Any]) -> TypedTuple:
        if len(columns) != len(values):
            raise ValueError(
                f"Column count ({len(columns)}) != value count ({len(values)})"
            )
        return TypedTuple(dict(zip(columns, values)))

    @property
    def values(self) -> Dict[str, Any]:
        return dict(self._values)

    @property
    def columns(self) -> List[str]:
        return list(self._values.keys())

    def get(self, column: str, default: Any = None) -> Any:
        return self._values.get(column, default)

    def __getitem__(self, column: str) -> Any:
        return self._values[column]

    def __contains__(self, column: str) -> bool:
        return column in self._values

    def project(self, columns: Set[str]) -> TypedTuple:
        """Project to a subset of columns."""
        return TypedTuple({k: v for k, v in self._values.items() if k in columns})

    def extend(self, column: str, value: Any) -> TypedTuple:
        """Add a new column with a value."""
        new_vals = dict(self._values)
        new_vals[column] = value
        return TypedTuple(new_vals)

    def drop(self, column: str) -> TypedTuple:
        """Remove a column."""
        new_vals = {k: v for k, v in self._values.items() if k != column}
        return TypedTuple(new_vals)

    def rename(self, old_name: str, new_name: str) -> TypedTuple:
        """Rename a column."""
        new_vals = {}
        for k, v in self._values.items():
            new_vals[new_name if k == old_name else k] = v
        return TypedTuple(new_vals)

    def update_value(self, column: str, value: Any) -> TypedTuple:
        """Return a new tuple with the specified column updated."""
        new_vals = dict(self._values)
        new_vals[column] = value
        return TypedTuple(new_vals)

    def coerce_column(self, column: str, coerce_fn: Callable[[Any], Any]) -> TypedTuple:
        """Apply a coercion function to a column value."""
        if column not in self._values:
            return self
        new_vals = dict(self._values)
        new_vals[column] = coerce_fn(self._values[column])
        return TypedTuple(new_vals)

    def merge(self, other: TypedTuple) -> TypedTuple:
        """Merge with another tuple (other's values take precedence)."""
        new_vals = dict(self._values)
        new_vals.update(other._values)
        return TypedTuple(new_vals)

    def _sortable_key(self) -> tuple:
        items = sorted(self._values.items())
        return tuple((k, str(v)) for k, v in items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedTuple):
            return NotImplemented
        return self._values == other._values

    def __hash__(self) -> int:
        if self._hash_cache is None:
            items = sorted(self._values.items())
            hashable = []
            for k, v in items:
                try:
                    hash(v)
                    hashable.append((k, v))
                except TypeError:
                    hashable.append((k, str(v)))
            self._hash_cache = hash(tuple(hashable))
        return self._hash_cache

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Tuple({items})"

    def __len__(self) -> int:
        return len(self._values)


# ---------------------------------------------------------------------------
# MultiSet
# ---------------------------------------------------------------------------

class MultiSet:
    """
    A multiset (bag) of TypedTuples with integer multiplicities.

    Supports standard multiset operations: union, intersection, difference,
    and the algebraic operations needed for the data delta group.
    """

    __slots__ = ("_elements",)

    def __init__(self, elements: Optional[Mapping[TypedTuple, int]] = None) -> None:
        self._elements: Counter[TypedTuple] = Counter()
        if elements:
            for t, count in elements.items():
                if count > 0:
                    self._elements[t] = count

    @staticmethod
    def empty() -> MultiSet:
        return MultiSet()

    @staticmethod
    def from_tuples(tuples: Iterable[TypedTuple]) -> MultiSet:
        ms = MultiSet()
        for t in tuples:
            ms._elements[t] += 1
        return ms

    @staticmethod
    def from_dicts(dicts: Iterable[Dict[str, Any]]) -> MultiSet:
        return MultiSet.from_tuples(TypedTuple(d) for d in dicts)

    @staticmethod
    def from_rows(
        columns: Sequence[str], rows: Iterable[Sequence[Any]]
    ) -> MultiSet:
        return MultiSet.from_tuples(
            TypedTuple.from_row(columns, row) for row in rows
        )

    def add(self, t: TypedTuple, count: int = 1) -> None:
        self._elements[t] += count
        if self._elements[t] <= 0:
            del self._elements[t]

    def remove(self, t: TypedTuple, count: int = 1) -> None:
        if t in self._elements:
            self._elements[t] -= count
            if self._elements[t] <= 0:
                del self._elements[t]

    def contains(self, t: TypedTuple) -> bool:
        return t in self._elements and self._elements[t] > 0

    def multiplicity(self, t: TypedTuple) -> int:
        return self._elements.get(t, 0)

    def union(self, other: MultiSet) -> MultiSet:
        """Multiset union (max of multiplicities)."""
        result = MultiSet(dict(self._elements))
        for t, count in other._elements.items():
            result._elements[t] = max(result._elements.get(t, 0), count)
        return result

    def intersection(self, other: MultiSet) -> MultiSet:
        """Multiset intersection (min of multiplicities)."""
        result = MultiSet()
        for t in self._elements:
            if t in other._elements:
                min_count = min(self._elements[t], other._elements[t])
                if min_count > 0:
                    result._elements[t] = min_count
        return result

    def difference(self, other: MultiSet) -> MultiSet:
        """Multiset difference (subtract multiplicities, floor at 0)."""
        result = MultiSet()
        for t, count in self._elements.items():
            new_count = count - other._elements.get(t, 0)
            if new_count > 0:
                result._elements[t] = new_count
        return result

    def sum(self, other: MultiSet) -> MultiSet:
        """Multiset sum (add multiplicities)."""
        result = MultiSet(dict(self._elements))
        for t, count in other._elements.items():
            result._elements[t] = result._elements.get(t, 0) + count
        return result

    def project(self, columns: Set[str]) -> MultiSet:
        """Project all tuples to a column subset."""
        result = MultiSet()
        for t, count in self._elements.items():
            projected = t.project(columns)
            result._elements[projected] = result._elements.get(projected, 0) + count
        return result

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> MultiSet:
        """Filter tuples by a predicate."""
        result = MultiSet()
        for t, count in self._elements.items():
            if predicate(t):
                result._elements[t] = count
        return result

    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> MultiSet:
        """Apply a function to each tuple."""
        result = MultiSet()
        for t, count in self._elements.items():
            new_t = fn(t)
            result._elements[new_t] = result._elements.get(new_t, 0) + count
        return result

    def distinct(self) -> MultiSet:
        """Convert to set semantics (all multiplicities = 1)."""
        return MultiSet({t: 1 for t in self._elements})

    @property
    def elements(self) -> Dict[TypedTuple, int]:
        return dict(self._elements)

    def tuples(self) -> List[TypedTuple]:
        """Return all tuples with repetition according to multiplicity."""
        result = []
        for t, count in self._elements.items():
            result.extend([t] * count)
        return result

    def unique_tuples(self) -> Set[TypedTuple]:
        """Return the set of unique tuples."""
        return set(self._elements.keys())

    def cardinality(self) -> int:
        """Total count including multiplicities."""
        return sum(self._elements.values())

    def distinct_count(self) -> int:
        """Count of distinct tuples."""
        return len(self._elements)

    def is_empty(self) -> bool:
        return len(self._elements) == 0

    def columns(self) -> Set[str]:
        """Return the union of all column names across tuples."""
        cols: Set[str] = set()
        for t in self._elements:
            cols.update(t.columns)
        return cols

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        result = []
        for t, count in self._elements.items():
            for _ in range(count):
                result.append(t.values)
        return result

    def __len__(self) -> int:
        return self.cardinality()

    def __iter__(self) -> Iterator[TypedTuple]:
        for t, count in self._elements.items():
            for _ in range(count):
                yield t

    def __contains__(self, item: TypedTuple) -> bool:
        return self.contains(item)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiSet):
            return NotImplemented
        return self._elements == other._elements

    def __hash__(self) -> int:
        return hash(frozenset(self._elements.items()))

    def __repr__(self) -> str:
        if self.is_empty():
            return "MultiSet(∅)"
        items = ", ".join(
            f"{t!r}×{c}" if c > 1 else repr(t)
            for t, c in sorted(self._elements.items(), key=lambda x: x[0]._sortable_key())
        )
        return f"MultiSet({{{items}}})"

    def __bool__(self) -> bool:
        return not self.is_empty()

    def copy(self) -> MultiSet:
        return MultiSet(dict(self._elements))


# ---------------------------------------------------------------------------
# Data Operations
# ---------------------------------------------------------------------------

class DataOperation(ABC):
    """Base class for all data operations in Δ_D."""

    @abstractmethod
    def inverse(self) -> DataOperation:
        """Return the inverse operation."""

    @abstractmethod
    def apply(self, relation: MultiSet) -> MultiSet:
        """Apply this operation to a multiset relation."""

    @abstractmethod
    def affected_rows_count(self) -> int:
        """Return the number of affected rows."""

    @abstractmethod
    def is_zero(self) -> bool:
        """Return True if this is a zero/identity operation."""

    @abstractmethod
    def project(self, columns: Set[str]) -> DataOperation:
        """Project the operation to a subset of columns."""

    @abstractmethod
    def filter(self, predicate: Callable[[TypedTuple], bool]) -> DataOperation:
        """Filter tuples in this operation by a predicate."""

    @abstractmethod
    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> DataOperation:
        """Apply a function to all tuples in this operation."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @abstractmethod
    def _key(self) -> tuple:
        """Return a hashable key."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataOperation):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))


@dataclass(frozen=True)
class InsertOp(DataOperation):
    """Insert tuples into a relation."""
    tuples: MultiSet

    def __init__(self, tuples: MultiSet) -> None:
        object.__setattr__(self, "tuples", tuples)

    def inverse(self) -> DataOperation:
        return DeleteOp(self.tuples)

    def apply(self, relation: MultiSet) -> MultiSet:
        return relation.sum(self.tuples)

    def affected_rows_count(self) -> int:
        return self.tuples.cardinality()

    def is_zero(self) -> bool:
        return self.tuples.is_empty()

    def project(self, columns: Set[str]) -> DataOperation:
        return InsertOp(self.tuples.project(columns))

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> DataOperation:
        return InsertOp(self.tuples.filter(predicate))

    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> DataOperation:
        return InsertOp(self.tuples.map_tuples(fn))

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "INSERT", "tuples": self.tuples.to_dicts()}

    def _key(self) -> tuple:
        return (self.tuples,)

    def __repr__(self) -> str:
        return f"INSERT({self.tuples.cardinality()} rows)"


@dataclass(frozen=True)
class DeleteOp(DataOperation):
    """Delete tuples from a relation."""
    tuples: MultiSet

    def __init__(self, tuples: MultiSet) -> None:
        object.__setattr__(self, "tuples", tuples)

    def inverse(self) -> DataOperation:
        return InsertOp(self.tuples)

    def apply(self, relation: MultiSet) -> MultiSet:
        return relation.difference(self.tuples)

    def affected_rows_count(self) -> int:
        return self.tuples.cardinality()

    def is_zero(self) -> bool:
        return self.tuples.is_empty()

    def project(self, columns: Set[str]) -> DataOperation:
        return DeleteOp(self.tuples.project(columns))

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> DataOperation:
        return DeleteOp(self.tuples.filter(predicate))

    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> DataOperation:
        return DeleteOp(self.tuples.map_tuples(fn))

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "DELETE", "tuples": self.tuples.to_dicts()}

    def _key(self) -> tuple:
        return (self.tuples,)

    def __repr__(self) -> str:
        return f"DELETE({self.tuples.cardinality()} rows)"


@dataclass(frozen=True)
class UpdateOp(DataOperation):
    """Update tuples: old_tuples → new_tuples."""
    old_tuples: MultiSet
    new_tuples: MultiSet

    def __init__(self, old_tuples: MultiSet, new_tuples: MultiSet) -> None:
        object.__setattr__(self, "old_tuples", old_tuples)
        object.__setattr__(self, "new_tuples", new_tuples)

    def inverse(self) -> DataOperation:
        return UpdateOp(self.new_tuples, self.old_tuples)

    def apply(self, relation: MultiSet) -> MultiSet:
        result = relation.difference(self.old_tuples)
        return result.sum(self.new_tuples)

    def affected_rows_count(self) -> int:
        return max(self.old_tuples.cardinality(), self.new_tuples.cardinality())

    def is_zero(self) -> bool:
        return self.old_tuples == self.new_tuples

    def project(self, columns: Set[str]) -> DataOperation:
        return UpdateOp(
            self.old_tuples.project(columns),
            self.new_tuples.project(columns),
        )

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> DataOperation:
        filtered_old = self.old_tuples.filter(predicate)
        filtered_new = self.new_tuples.filter(predicate)
        return UpdateOp(filtered_old, filtered_new)

    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> DataOperation:
        return UpdateOp(
            self.old_tuples.map_tuples(fn),
            self.new_tuples.map_tuples(fn),
        )

    def to_delete_insert(self) -> Tuple[DeleteOp, InsertOp]:
        """Decompose UPDATE into DELETE + INSERT."""
        return DeleteOp(self.old_tuples), InsertOp(self.new_tuples)

    def changed_columns(self) -> Set[str]:
        """Return columns that differ between old and new tuples."""
        changed: Set[str] = set()
        old_list = sorted(
            self.old_tuples.unique_tuples(),
            key=lambda t: t._sortable_key(),
        )
        new_list = sorted(
            self.new_tuples.unique_tuples(),
            key=lambda t: t._sortable_key(),
        )
        for old_t, new_t in zip(old_list, new_list):
            for col in set(old_t.columns) | set(new_t.columns):
                if old_t.get(col) != new_t.get(col):
                    changed.add(col)
        return changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "UPDATE",
            "old_tuples": self.old_tuples.to_dicts(),
            "new_tuples": self.new_tuples.to_dicts(),
        }

    def _key(self) -> tuple:
        return (self.old_tuples, self.new_tuples)

    def __repr__(self) -> str:
        return f"UPDATE({self.old_tuples.cardinality()} -> {self.new_tuples.cardinality()} rows)"


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------

def _cancel_insert_delete(ops: List[DataOperation]) -> List[DataOperation]:
    """Cancel INSERT+DELETE of the same tuples."""
    pending_inserts: Counter[TypedTuple] = Counter()
    pending_deletes: Counter[TypedTuple] = Counter()

    for op in ops:
        if isinstance(op, InsertOp):
            for t, count in op.tuples.elements.items():
                pending_inserts[t] += count
        elif isinstance(op, DeleteOp):
            for t, count in op.tuples.elements.items():
                pending_deletes[t] += count
        elif isinstance(op, UpdateOp):
            for t, count in op.old_tuples.elements.items():
                pending_deletes[t] += count
            for t, count in op.new_tuples.elements.items():
                pending_inserts[t] += count

    net_inserts: Counter[TypedTuple] = Counter()
    net_deletes: Counter[TypedTuple] = Counter()
    all_tuples = set(pending_inserts.keys()) | set(pending_deletes.keys())

    for t in all_tuples:
        ins = pending_inserts.get(t, 0)
        dels = pending_deletes.get(t, 0)
        net = ins - dels
        if net > 0:
            net_inserts[t] = net
        elif net < 0:
            net_deletes[t] = -net

    result: List[DataOperation] = []
    if net_deletes:
        result.append(DeleteOp(MultiSet(dict(net_deletes))))
    if net_inserts:
        result.append(InsertOp(MultiSet(dict(net_inserts))))
    return result


def _merge_inserts(ops: List[DataOperation]) -> List[DataOperation]:
    """Merge consecutive InsertOps into one."""
    result: List[DataOperation] = []
    pending_insert: Optional[MultiSet] = None
    for op in ops:
        if isinstance(op, InsertOp):
            if pending_insert is None:
                pending_insert = op.tuples.copy()
            else:
                pending_insert = pending_insert.sum(op.tuples)
        else:
            if pending_insert is not None:
                result.append(InsertOp(pending_insert))
                pending_insert = None
            result.append(op)
    if pending_insert is not None:
        result.append(InsertOp(pending_insert))
    return result


def _merge_deletes(ops: List[DataOperation]) -> List[DataOperation]:
    """Merge consecutive DeleteOps into one."""
    result: List[DataOperation] = []
    pending_delete: Optional[MultiSet] = None
    for op in ops:
        if isinstance(op, DeleteOp):
            if pending_delete is None:
                pending_delete = op.tuples.copy()
            else:
                pending_delete = pending_delete.sum(op.tuples)
        else:
            if pending_delete is not None:
                result.append(DeleteOp(pending_delete))
                pending_delete = None
            result.append(op)
    if pending_delete is not None:
        result.append(DeleteOp(pending_delete))
    return result


def _remove_zero_ops(ops: List[DataOperation]) -> List[DataOperation]:
    """Remove identity/zero operations."""
    return [op for op in ops if not op.is_zero()]


def _convert_matching_update_to_noop(ops: List[DataOperation]) -> List[DataOperation]:
    """Remove UpdateOps where old == new."""
    return [
        op for op in ops
        if not (isinstance(op, UpdateOp) and op.old_tuples == op.new_tuples)
    ]


# ---------------------------------------------------------------------------
# Data Delta Group
# ---------------------------------------------------------------------------

class DataDelta:
    """
    Represents an element of the data delta group Δ_D.

    The group operation is composition (∘), the identity is the zero delta,
    and every delta has a unique inverse.

    Operations are applied in order: ops[0] first.
    """

    __slots__ = ("_operations", "_hash_cache")

    def __init__(self, operations: Optional[List[DataOperation]] = None) -> None:
        self._operations: List[DataOperation] = list(operations) if operations else []
        self._hash_cache: Optional[int] = None

    @property
    def operations(self) -> List[DataOperation]:
        return list(self._operations)

    @staticmethod
    def zero() -> DataDelta:
        """Return the zero (identity) element of the group."""
        return DataDelta([])

    @staticmethod
    def from_operation(op: DataOperation) -> DataDelta:
        return DataDelta([op])

    @staticmethod
    def from_operations(ops: Sequence[DataOperation]) -> DataDelta:
        return DataDelta(list(ops))

    @staticmethod
    def insert(tuples: MultiSet) -> DataDelta:
        """Create a delta that inserts the given tuples."""
        return DataDelta([InsertOp(tuples)])

    @staticmethod
    def delete(tuples: MultiSet) -> DataDelta:
        """Create a delta that deletes the given tuples."""
        return DataDelta([DeleteOp(tuples)])

    @staticmethod
    def update(old_tuples: MultiSet, new_tuples: MultiSet) -> DataDelta:
        """Create a delta that updates tuples."""
        return DataDelta([UpdateOp(old_tuples, new_tuples)])

    @staticmethod
    def from_diff(old_relation: MultiSet, new_relation: MultiSet) -> DataDelta:
        """Compute the DataDelta that transforms old_relation into new_relation."""
        deleted = old_relation.difference(new_relation)
        inserted = new_relation.difference(old_relation)
        ops: List[DataOperation] = []
        if not deleted.is_empty():
            ops.append(DeleteOp(deleted))
        if not inserted.is_empty():
            ops.append(InsertOp(inserted))
        return DataDelta(ops)

    def compose(self, other: DataDelta) -> DataDelta:
        """
        Group operation: self ∘ other.

        Applies self first, then other. Normalization is deferred to
        normalize() for efficiency.
        """
        combined = list(self._operations) + list(other._operations)
        return DataDelta(combined)

    def inverse(self) -> DataDelta:
        """
        Group inverse: returns δ⁻¹ such that δ ∘ δ⁻¹ = 𝟎.

        INSERT → DELETE, DELETE → INSERT, UPDATE(old,new) → UPDATE(new,old).
        Operations are reversed in order.
        """
        inv_ops = [op.inverse() for op in reversed(self._operations)]
        return DataDelta(inv_ops)

    def normalize(self) -> DataDelta:
        """
        Normalize to canonical form:
        1. Remove zero operations
        2. Convert matching updates to no-ops
        3. Cancel INSERT+DELETE of same tuples
        4. Merge consecutive inserts and deletes
        """
        ops = list(self._operations)
        ops = _remove_zero_ops(ops)
        ops = _convert_matching_update_to_noop(ops)
        ops = _cancel_insert_delete(ops)
        ops = _merge_inserts(ops)
        ops = _merge_deletes(ops)
        ops = _remove_zero_ops(ops)
        return DataDelta(ops)

    def is_zero(self) -> bool:
        """Check if this is the zero (identity) element."""
        n = self.normalize()
        return len(n._operations) == 0

    def apply_to_data(self, relation: MultiSet) -> MultiSet:
        """Apply this delta to a multiset relation."""
        result = relation.copy()
        for op in self._operations:
            result = op.apply(result)
        return result

    def affected_rows_count(self) -> int:
        """Total number of affected rows across all operations."""
        return sum(op.affected_rows_count() for op in self._operations)

    def compress(self) -> DataDelta:
        """
        Delta compression: produces a minimal equivalent delta.

        Collects all net insertions and deletions, then checks if
        pairs of (delete, insert) can be represented as updates.
        """
        n = self.normalize()
        if len(n._operations) <= 1:
            return n

        all_deletes: Counter[TypedTuple] = Counter()
        all_inserts: Counter[TypedTuple] = Counter()

        for op in n._operations:
            if isinstance(op, DeleteOp):
                for t, c in op.tuples.elements.items():
                    all_deletes[t] += c
            elif isinstance(op, InsertOp):
                for t, c in op.tuples.elements.items():
                    all_inserts[t] += c
            elif isinstance(op, UpdateOp):
                for t, c in op.old_tuples.elements.items():
                    all_deletes[t] += c
                for t, c in op.new_tuples.elements.items():
                    all_inserts[t] += c

        del_tuples = set(all_deletes.keys())
        ins_tuples = set(all_inserts.keys())

        update_old = MultiSet()
        update_new = MultiSet()
        pure_deletes = MultiSet()
        pure_inserts = MultiSet()

        matched_del: Counter[TypedTuple] = Counter()
        matched_ins: Counter[TypedTuple] = Counter()

        del_by_cols: Dict[FrozenSet[str], List[TypedTuple]] = defaultdict(list)
        ins_by_cols: Dict[FrozenSet[str], List[TypedTuple]] = defaultdict(list)

        for t in del_tuples:
            key = frozenset(t.columns)
            del_by_cols[key].append(t)
        for t in ins_tuples:
            key = frozenset(t.columns)
            ins_by_cols[key].append(t)

        for col_key in del_by_cols:
            if col_key not in ins_by_cols:
                continue
            del_list = del_by_cols[col_key]
            ins_list = ins_by_cols[col_key]
            used_ins: Set[int] = set()
            for dt in del_list:
                d_count = all_deletes[dt] - matched_del[dt]
                if d_count <= 0:
                    continue
                for idx, it in enumerate(ins_list):
                    if idx in used_ins:
                        continue
                    i_count = all_inserts[it] - matched_ins[it]
                    if i_count <= 0:
                        continue
                    pair_count = min(d_count, i_count)
                    update_old.add(dt, pair_count)
                    update_new.add(it, pair_count)
                    matched_del[dt] += pair_count
                    matched_ins[it] += pair_count
                    d_count -= pair_count
                    if d_count <= 0:
                        break

        for t, c in all_deletes.items():
            remaining = c - matched_del.get(t, 0)
            if remaining > 0:
                pure_deletes.add(t, remaining)

        for t, c in all_inserts.items():
            remaining = c - matched_ins.get(t, 0)
            if remaining > 0:
                pure_inserts.add(t, remaining)

        result_ops: List[DataOperation] = []
        if not pure_deletes.is_empty():
            result_ops.append(DeleteOp(pure_deletes))
        if not update_old.is_empty():
            result_ops.append(UpdateOp(update_old, update_new))
        if not pure_inserts.is_empty():
            result_ops.append(InsertOp(pure_inserts))

        return DataDelta(result_ops)

    def split_by_columns(self, columns: Set[str]) -> Dict[FrozenSet[str], DataDelta]:
        """
        Split delta into groups based on which columns are affected.

        Returns a mapping from affected column sets to sub-deltas.
        """
        result: Dict[FrozenSet[str], List[DataOperation]] = defaultdict(list)

        for op in self._operations:
            if isinstance(op, InsertOp):
                for t in op.tuples.unique_tuples():
                    affected = frozenset(c for c in t.columns if c in columns)
                    if not affected:
                        affected = frozenset(["__all__"])
                    result[affected].append(
                        InsertOp(MultiSet({t: op.tuples.multiplicity(t)}))
                    )
            elif isinstance(op, DeleteOp):
                for t in op.tuples.unique_tuples():
                    affected = frozenset(c for c in t.columns if c in columns)
                    if not affected:
                        affected = frozenset(["__all__"])
                    result[affected].append(
                        DeleteOp(MultiSet({t: op.tuples.multiplicity(t)}))
                    )
            elif isinstance(op, UpdateOp):
                changed = op.changed_columns() & columns
                key = frozenset(changed) if changed else frozenset(["__all__"])
                result[key].append(op)

        return {k: DataDelta(v) for k, v in result.items()}

    def project(self, columns: Set[str]) -> DataDelta:
        """Project all operations to a subset of columns."""
        projected_ops = [op.project(columns) for op in self._operations]
        return DataDelta(projected_ops)

    def filter(self, predicate: Callable[[TypedTuple], bool]) -> DataDelta:
        """Filter all operations by a predicate."""
        filtered_ops = [op.filter(predicate) for op in self._operations]
        return DataDelta(filtered_ops)

    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> DataDelta:
        """Apply a transformation function to all tuples in all operations."""
        mapped_ops = [op.map_tuples(fn) for op in self._operations]
        return DataDelta(mapped_ops)

    def operation_count(self) -> int:
        """Return the number of operations."""
        return len(self._operations)

    def insert_count(self) -> int:
        """Return total number of inserted tuples."""
        count = 0
        for op in self._operations:
            if isinstance(op, InsertOp):
                count += op.tuples.cardinality()
            elif isinstance(op, UpdateOp):
                count += op.new_tuples.cardinality()
        return count

    def delete_count(self) -> int:
        """Return total number of deleted tuples."""
        count = 0
        for op in self._operations:
            if isinstance(op, DeleteOp):
                count += op.tuples.cardinality()
            elif isinstance(op, UpdateOp):
                count += op.old_tuples.cardinality()
        return count

    def net_row_change(self) -> int:
        """Return the net change in row count."""
        return self.insert_count() - self.delete_count()

    def get_all_inserts(self) -> MultiSet:
        """Collect all inserted tuples across all operations."""
        result = MultiSet.empty()
        for op in self._operations:
            if isinstance(op, InsertOp):
                result = result.sum(op.tuples)
            elif isinstance(op, UpdateOp):
                result = result.sum(op.new_tuples)
        return result

    def get_all_deletes(self) -> MultiSet:
        """Collect all deleted tuples across all operations."""
        result = MultiSet.empty()
        for op in self._operations:
            if isinstance(op, DeleteOp):
                result = result.sum(op.tuples)
            elif isinstance(op, UpdateOp):
                result = result.sum(op.old_tuples)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"operations": [op.to_dict() for op in self._operations]}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DataDelta:
        ops: List[DataOperation] = []
        for op_dict in data.get("operations", []):
            op_type = op_dict["op"]
            if op_type == "INSERT":
                tuples = MultiSet.from_dicts(op_dict["tuples"])
                ops.append(InsertOp(tuples))
            elif op_type == "DELETE":
                tuples = MultiSet.from_dicts(op_dict["tuples"])
                ops.append(DeleteOp(tuples))
            elif op_type == "UPDATE":
                old_tuples = MultiSet.from_dicts(op_dict["old_tuples"])
                new_tuples = MultiSet.from_dicts(op_dict["new_tuples"])
                ops.append(UpdateOp(old_tuples, new_tuples))
        return DataDelta(ops)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataDelta):
            return NotImplemented
        n1 = self.normalize()
        n2 = other.normalize()
        if len(n1._operations) != len(n2._operations):
            return False
        return all(a == b for a, b in zip(n1._operations, n2._operations))

    def __hash__(self) -> int:
        if self._hash_cache is None:
            n = self.normalize()
            self._hash_cache = hash(tuple(hash(op) for op in n._operations))
        return self._hash_cache

    def __repr__(self) -> str:
        if not self._operations:
            return "DataDelta(zero)"
        ops_str = ", ".join(repr(op) for op in self._operations)
        return f"DataDelta([{ops_str}])"

    def __len__(self) -> int:
        return len(self._operations)

    def __bool__(self) -> bool:
        return not self.is_zero()

    def __iter__(self):
        return iter(self._operations)


# ---------------------------------------------------------------------------
# Convenience: diff two relations
# ---------------------------------------------------------------------------

def diff_relations(old_relation: MultiSet, new_relation: MultiSet) -> DataDelta:
    """Compute the minimal DataDelta that transforms old_relation into new_relation."""
    return DataDelta.from_diff(old_relation, new_relation)
