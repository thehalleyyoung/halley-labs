"""
Index Delta — extension to the three-sorted delta algebra.

Real-world migrations frequently add, drop, or alter indexes. This module
extends the schema delta monoid with index-specific operations that compose
correctly and carry realistic cost metadata.

Index deltas compose as a monoid alongside :class:`SchemaDelta`:
- CreateIndex ∘ DropIndex (same name) = identity
- DropIndex ∘ CreateIndex (same name) = rebuild
- CreateIndex operations are idempotent (IF NOT EXISTS semantics)

Cost implications:
- CREATE INDEX: O(n log n) — full table scan + sort
- DROP INDEX: O(1) — metadata-only
- ALTER INDEX (reindex): O(n log n) — equivalent to drop + create
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class IndexType(Enum):
    """Types of database indexes."""
    BTREE = "BTREE"
    HASH = "HASH"
    GIN = "GIN"
    GIST = "GIST"
    BRIN = "BRIN"
    PARTIAL = "PARTIAL"
    UNIQUE = "UNIQUE"
    COMPOSITE = "COMPOSITE"
    EXPRESSION = "EXPRESSION"


class IndexOpType(Enum):
    """Types of index operations."""
    CREATE_INDEX = auto()
    DROP_INDEX = auto()
    REINDEX = auto()
    ALTER_INDEX_RENAME = auto()


@dataclass(frozen=True)
class IndexSpec:
    """Specification of a database index."""
    name: str
    table_name: str
    columns: Tuple[str, ...] = ()
    index_type: IndexType = IndexType.BTREE
    unique: bool = False
    partial_predicate: Optional[str] = None
    expression: Optional[str] = None

    def __repr__(self) -> str:
        cols = ", ".join(self.columns) if self.columns else self.expression or "?"
        u = "UNIQUE " if self.unique else ""
        return f"{u}{self.index_type.value} INDEX {self.name} ON {self.table_name}({cols})"


@dataclass(frozen=True)
class IndexOperation:
    """A single index operation."""
    op_type: IndexOpType
    index_spec: IndexSpec
    old_name: Optional[str] = None

    def inverse(self) -> IndexOperation:
        """Return the inverse operation."""
        if self.op_type == IndexOpType.CREATE_INDEX:
            return IndexOperation(
                op_type=IndexOpType.DROP_INDEX,
                index_spec=self.index_spec,
            )
        elif self.op_type == IndexOpType.DROP_INDEX:
            return IndexOperation(
                op_type=IndexOpType.CREATE_INDEX,
                index_spec=self.index_spec,
            )
        elif self.op_type == IndexOpType.ALTER_INDEX_RENAME:
            return IndexOperation(
                op_type=IndexOpType.ALTER_INDEX_RENAME,
                index_spec=IndexSpec(
                    name=self.old_name or self.index_spec.name,
                    table_name=self.index_spec.table_name,
                    columns=self.index_spec.columns,
                    index_type=self.index_spec.index_type,
                ),
                old_name=self.index_spec.name,
            )
        else:
            return IndexOperation(
                op_type=IndexOpType.DROP_INDEX,
                index_spec=self.index_spec,
            )

    @property
    def affected_columns(self) -> Set[str]:
        return set(self.index_spec.columns)

    @property
    def is_identity(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "op_type": self.op_type.name,
            "index_name": self.index_spec.name,
            "table": self.index_spec.table_name,
            "columns": list(self.index_spec.columns),
            "index_type": self.index_spec.index_type.value,
            "unique": self.index_spec.unique,
        }
        if self.index_spec.partial_predicate:
            d["partial_predicate"] = self.index_spec.partial_predicate
        if self.old_name:
            d["old_name"] = self.old_name
        return d

    def __repr__(self) -> str:
        return f"{self.op_type.name}({self.index_spec})"


class IndexDelta:
    """
    Index delta monoid — extends the schema delta algebra with
    index-specific operations.

    Composition rules:
    - CREATE then DROP same index → identity (annihilation)
    - DROP then CREATE same index → REINDEX
    - Multiple CREATEs of same index → single CREATE (idempotent)
    - REINDEX composes with itself → single REINDEX
    """

    __slots__ = ("_operations",)

    def __init__(self, operations: Optional[List[IndexOperation]] = None) -> None:
        self._operations: List[IndexOperation] = list(operations) if operations else []

    @property
    def operations(self) -> List[IndexOperation]:
        return list(self._operations)

    @staticmethod
    def identity() -> IndexDelta:
        return IndexDelta([])

    @staticmethod
    def from_operation(op: IndexOperation) -> IndexDelta:
        return IndexDelta([op])

    def compose(self, other: IndexDelta) -> IndexDelta:
        """Monoid composition with index-specific simplification."""
        combined = list(self._operations) + list(other._operations)
        combined = _simplify_index_ops(combined)
        return IndexDelta(combined)

    def inverse(self) -> IndexDelta:
        inv_ops = [op.inverse() for op in reversed(self._operations)]
        return IndexDelta(inv_ops)

    def normalize(self) -> IndexDelta:
        return IndexDelta(_simplify_index_ops(list(self._operations)))

    @property
    def is_identity(self) -> bool:
        normalized = self.normalize()
        return len(normalized._operations) == 0

    @property
    def operation_count(self) -> int:
        return len(self._operations)

    @property
    def affected_tables(self) -> Set[str]:
        return {op.index_spec.table_name for op in self._operations}

    @property
    def affected_columns(self) -> Set[str]:
        result: Set[str] = set()
        for op in self._operations:
            result |= op.affected_columns
        return result

    @property
    def creates(self) -> List[IndexOperation]:
        return [op for op in self._operations if op.op_type == IndexOpType.CREATE_INDEX]

    @property
    def drops(self) -> List[IndexOperation]:
        return [op for op in self._operations if op.op_type == IndexOpType.DROP_INDEX]

    def estimate_cost(self, table_row_counts: Dict[str, int] | None = None) -> float:
        """Estimate the cost of applying this index delta.

        CREATE INDEX: O(n log n) for the table size
        DROP INDEX: O(1) — metadata only
        REINDEX: O(n log n) — full rebuild
        """
        import math
        row_counts = table_row_counts or {}
        total = 0.0
        for op in self._operations:
            n = row_counts.get(op.index_spec.table_name, 10000)
            if op.op_type == IndexOpType.CREATE_INDEX:
                total += n * math.log2(max(n, 2)) * 1e-6
            elif op.op_type == IndexOpType.DROP_INDEX:
                total += 1e-4
            elif op.op_type == IndexOpType.REINDEX:
                total += n * math.log2(max(n, 2)) * 1e-6
            elif op.op_type == IndexOpType.ALTER_INDEX_RENAME:
                total += 1e-4
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": [op.to_dict() for op in self._operations],
            "operation_count": len(self._operations),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexDelta):
            return NotImplemented
        return self._operations == other._operations

    def __repr__(self) -> str:
        return f"IndexDelta({len(self._operations)} ops)"


def _simplify_index_ops(ops: List[IndexOperation]) -> List[IndexOperation]:
    """Simplify a sequence of index operations using algebraic laws.

    - CREATE + DROP same index → removed (annihilation)
    - DROP + CREATE same index → REINDEX
    - Duplicate CREATEs → single CREATE
    """
    result: List[IndexOperation] = []
    skip: Set[int] = set()

    for i, op in enumerate(ops):
        if i in skip:
            continue

        if op.op_type == IndexOpType.CREATE_INDEX:
            # Look ahead for DROP of same index
            annihilated = False
            for j in range(i + 1, len(ops)):
                if j in skip:
                    continue
                if (ops[j].op_type == IndexOpType.DROP_INDEX
                        and ops[j].index_spec.name == op.index_spec.name):
                    skip.add(j)
                    annihilated = True
                    break
            if not annihilated:
                # Check for duplicate CREATE
                is_dup = False
                for prev in result:
                    if (prev.op_type == IndexOpType.CREATE_INDEX
                            and prev.index_spec.name == op.index_spec.name):
                        is_dup = True
                        break
                if not is_dup:
                    result.append(op)

        elif op.op_type == IndexOpType.DROP_INDEX:
            # Look ahead for CREATE of same index → REINDEX
            recreated = False
            for j in range(i + 1, len(ops)):
                if j in skip:
                    continue
                if (ops[j].op_type == IndexOpType.CREATE_INDEX
                        and ops[j].index_spec.name == op.index_spec.name):
                    skip.add(j)
                    result.append(IndexOperation(
                        op_type=IndexOpType.REINDEX,
                        index_spec=ops[j].index_spec,
                    ))
                    recreated = True
                    break
            if not recreated:
                result.append(op)

        else:
            result.append(op)

    return result


def create_index_delta(
    name: str,
    table: str,
    columns: Tuple[str, ...] = (),
    index_type: IndexType = IndexType.BTREE,
    unique: bool = False,
) -> IndexDelta:
    """Helper to create a single CREATE INDEX delta."""
    spec = IndexSpec(
        name=name,
        table_name=table,
        columns=columns,
        index_type=index_type,
        unique=unique,
    )
    return IndexDelta.from_operation(IndexOperation(
        op_type=IndexOpType.CREATE_INDEX,
        index_spec=spec,
    ))


def drop_index_delta(
    name: str,
    table: str = "",
    columns: Tuple[str, ...] = (),
) -> IndexDelta:
    """Helper to create a single DROP INDEX delta."""
    spec = IndexSpec(name=name, table_name=table, columns=columns)
    return IndexDelta.from_operation(IndexOperation(
        op_type=IndexOpType.DROP_INDEX,
        index_spec=spec,
    ))
