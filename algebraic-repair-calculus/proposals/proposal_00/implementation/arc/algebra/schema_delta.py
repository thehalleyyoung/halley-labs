"""
Schema Delta Monoid (Δ_S)
=========================

Implements the schema delta monoid from the three-sorted delta algebra.
Schema deltas represent changes to relation schemas (columns, types, constraints).

Algebraic properties:
- Monoid: (Δ_S, ∘, id) where ∘ is composition and id is the identity delta
- Inverse: Each schema operation has a well-defined inverse
- Normalization: Canonical form removes no-ops and merges compatible operations

Operations: ADD_COLUMN, DROP_COLUMN, RENAME_COLUMN, CHANGE_TYPE,
            ADD_CONSTRAINT, DROP_CONSTRAINT
"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions (will eventually live in arc.types)
# ---------------------------------------------------------------------------

class SQLType(Enum):
    """Supported SQL column types."""
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


class ConstraintType(Enum):
    """Types of schema-level constraints."""
    NOT_NULL = "NOT_NULL"
    UNIQUE = "UNIQUE"
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    CHECK = "CHECK"
    EXCLUSION = "EXCLUSION"
    DEFAULT = "DEFAULT"


class ConflictType(Enum):
    """Types of conflicts between schema operations."""
    COLUMN_NOT_FOUND = auto()
    COLUMN_ALREADY_EXISTS = auto()
    TYPE_INCOMPATIBLE = auto()
    CONSTRAINT_VIOLATION = auto()
    RENAME_CONFLICT = auto()
    DROP_DEPENDENCY = auto()
    POSITION_CONFLICT = auto()
    CIRCULAR_RENAME = auto()


@dataclass(frozen=True)
class ColumnDef:
    """Definition of a table column."""
    name: str
    sql_type: SQLType
    nullable: bool = True
    default_expr: Optional[str] = None
    position: int = -1

    def with_name(self, new_name: str) -> ColumnDef:
        return ColumnDef(
            name=new_name,
            sql_type=self.sql_type,
            nullable=self.nullable,
            default_expr=self.default_expr,
            position=self.position,
        )

    def with_type(self, new_type: SQLType) -> ColumnDef:
        return ColumnDef(
            name=self.name,
            sql_type=new_type,
            nullable=self.nullable,
            default_expr=self.default_expr,
            position=self.position,
        )

    def with_position(self, pos: int) -> ColumnDef:
        return ColumnDef(
            name=self.name,
            sql_type=self.sql_type,
            nullable=self.nullable,
            default_expr=self.default_expr,
            position=pos,
        )


@dataclass(frozen=True)
class ConstraintDef:
    """Definition of a schema constraint."""
    constraint_id: str
    constraint_type: ConstraintType
    columns: Tuple[str, ...]
    predicate: Optional[str] = None
    reference_table: Optional[str] = None
    reference_columns: Optional[Tuple[str, ...]] = None

    def with_columns(self, new_columns: Tuple[str, ...]) -> ConstraintDef:
        return ConstraintDef(
            constraint_id=self.constraint_id,
            constraint_type=self.constraint_type,
            columns=new_columns,
            predicate=self.predicate,
            reference_table=self.reference_table,
            reference_columns=self.reference_columns,
        )

    def references_column(self, col: str) -> bool:
        if col in self.columns:
            return True
        if self.reference_columns and col in self.reference_columns:
            return True
        if self.predicate and col in self.predicate:
            return True
        return False

    def rename_column(self, old_name: str, new_name: str) -> ConstraintDef:
        new_cols = tuple(new_name if c == old_name else c for c in self.columns)
        new_ref_cols = None
        if self.reference_columns:
            new_ref_cols = tuple(
                new_name if c == old_name else c for c in self.reference_columns
            )
        new_predicate = self.predicate
        if new_predicate and old_name in new_predicate:
            new_predicate = new_predicate.replace(old_name, new_name)
        return ConstraintDef(
            constraint_id=self.constraint_id,
            constraint_type=self.constraint_type,
            columns=new_cols,
            predicate=new_predicate,
            reference_table=self.reference_table,
            reference_columns=new_ref_cols,
        )


@dataclass
class Schema:
    """Represents a relation schema with columns and constraints."""
    name: str
    columns: OrderedDict[str, ColumnDef] = field(default_factory=OrderedDict)
    constraints: Dict[str, ConstraintDef] = field(default_factory=dict)

    def copy(self) -> Schema:
        return Schema(
            name=self.name,
            columns=OrderedDict(self.columns),
            constraints=dict(self.constraints),
        )

    def has_column(self, name: str) -> bool:
        return name in self.columns

    def get_column(self, name: str) -> Optional[ColumnDef]:
        return self.columns.get(name)

    def column_names(self) -> List[str]:
        return list(self.columns.keys())

    def add_column(self, col: ColumnDef) -> None:
        self.columns[col.name] = col

    def drop_column(self, name: str) -> Optional[ColumnDef]:
        return self.columns.pop(name, None)

    def rename_column(self, old_name: str, new_name: str) -> None:
        if old_name not in self.columns:
            return
        col = self.columns[old_name]
        new_col = col.with_name(new_name)
        new_columns = OrderedDict()
        for k, v in self.columns.items():
            if k == old_name:
                new_columns[new_name] = new_col
            else:
                new_columns[k] = v
        self.columns = new_columns
        for cid, cdef in list(self.constraints.items()):
            if cdef.references_column(old_name):
                self.constraints[cid] = cdef.rename_column(old_name, new_name)

    def add_constraint(self, cdef: ConstraintDef) -> None:
        self.constraints[cdef.constraint_id] = cdef

    def drop_constraint(self, constraint_id: str) -> Optional[ConstraintDef]:
        return self.constraints.pop(constraint_id, None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return (
            self.name == other.name
            and self.columns == other.columns
            and self.constraints == other.constraints
        )

    def __repr__(self) -> str:
        cols = ", ".join(
            f"{c.name}:{c.sql_type.value}" for c in self.columns.values()
        )
        return f"Schema({self.name}, [{cols}])"


@dataclass(frozen=True)
class Conflict:
    """Describes a conflict between two schema operations."""
    conflict_type: ConflictType
    description: str
    op1_index: int
    op2_index: int
    column: Optional[str] = None
    constraint_id: Optional[str] = None
    severity: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Conflict({self.conflict_type.name}: {self.description}, "
            f"ops=[{self.op1_index},{self.op2_index}])"
        )


# ---------------------------------------------------------------------------
# Schema Operations
# ---------------------------------------------------------------------------

class SchemaOperation(ABC):
    """Base class for all schema operations in Δ_S."""

    @abstractmethod
    def inverse(self) -> SchemaOperation:
        """Return the inverse operation."""

    @abstractmethod
    def apply(self, schema: Schema) -> Schema:
        """Apply this operation to a schema, returning a new schema."""

    @abstractmethod
    def affected_columns(self) -> Set[str]:
        """Return the set of column names affected by this operation."""

    @abstractmethod
    def is_identity(self) -> bool:
        """Return True if this operation is a no-op."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @abstractmethod
    def _key(self) -> tuple:
        """Return a hashable key for equality and hashing."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaOperation):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))


@dataclass(frozen=True)
class AddColumn(SchemaOperation):
    """Add a new column to a schema."""
    name: str
    sql_type: SQLType
    position: int = -1
    default_expr: Optional[str] = None
    nullable: bool = True

    def inverse(self) -> SchemaOperation:
        return DropColumn(name=self.name)

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        pos = self.position if self.position >= 0 else len(s.columns)
        col = ColumnDef(
            name=self.name,
            sql_type=self.sql_type,
            nullable=self.nullable,
            default_expr=self.default_expr,
            position=pos,
        )
        if self.position >= 0 and self.position < len(s.columns):
            new_cols = OrderedDict()
            items = list(s.columns.items())
            for i, (k, v) in enumerate(items):
                if i == self.position:
                    new_cols[self.name] = col
                new_cols[k] = v.with_position(
                    i + 1 if i >= self.position else i
                )
            if self.position >= len(items):
                new_cols[self.name] = col
            s.columns = new_cols
        else:
            s.add_column(col.with_position(pos))
        return s

    def affected_columns(self) -> Set[str]:
        return {self.name}

    def is_identity(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "ADD_COLUMN",
            "name": self.name,
            "sql_type": self.sql_type.value,
            "position": self.position,
            "default_expr": self.default_expr,
            "nullable": self.nullable,
        }

    def _key(self) -> tuple:
        return (self.name, self.sql_type, self.position, self.default_expr, self.nullable)

    def __repr__(self) -> str:
        parts = [f"ADD {self.name} {self.sql_type.value}"]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default_expr:
            parts.append(f"DEFAULT {self.default_expr}")
        return " ".join(parts)


@dataclass(frozen=True)
class DropColumn(SchemaOperation):
    """Drop a column from a schema."""
    name: str
    _preserved_type: Optional[SQLType] = None
    _preserved_nullable: Optional[bool] = None
    _preserved_default: Optional[str] = None
    _preserved_position: Optional[int] = None

    def inverse(self) -> SchemaOperation:
        return AddColumn(
            name=self.name,
            sql_type=self._preserved_type or SQLType.UNKNOWN,
            position=self._preserved_position or -1,
            default_expr=self._preserved_default,
            nullable=self._preserved_nullable if self._preserved_nullable is not None else True,
        )

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        existing = s.get_column(self.name)
        if existing is not None:
            s.drop_column(self.name)
            to_remove = [
                cid for cid, cdef in s.constraints.items()
                if cdef.references_column(self.name)
            ]
            for cid in to_remove:
                s.drop_constraint(cid)
        return s

    def with_preserved_info(
        self,
        sql_type: SQLType,
        nullable: bool = True,
        default_expr: Optional[str] = None,
        position: int = -1,
    ) -> DropColumn:
        return DropColumn(
            name=self.name,
            _preserved_type=sql_type,
            _preserved_nullable=nullable,
            _preserved_default=default_expr,
            _preserved_position=position,
        )

    def affected_columns(self) -> Set[str]:
        return {self.name}

    def is_identity(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "DROP_COLUMN", "name": self.name}

    def _key(self) -> tuple:
        return (self.name,)

    def __repr__(self) -> str:
        return f"DROP {self.name}"


@dataclass(frozen=True)
class RenameColumn(SchemaOperation):
    """Rename a column."""
    old_name: str
    new_name: str

    def inverse(self) -> SchemaOperation:
        return RenameColumn(old_name=self.new_name, new_name=self.old_name)

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        s.rename_column(self.old_name, self.new_name)
        return s

    def affected_columns(self) -> Set[str]:
        return {self.old_name, self.new_name}

    def is_identity(self) -> bool:
        return self.old_name == self.new_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "RENAME_COLUMN",
            "old_name": self.old_name,
            "new_name": self.new_name,
        }

    def _key(self) -> tuple:
        return (self.old_name, self.new_name)

    def __repr__(self) -> str:
        return f"RENAME {self.old_name} -> {self.new_name}"


@dataclass(frozen=True)
class ChangeType(SchemaOperation):
    """Change the type of a column, with optional coercion expression."""
    column_name: str
    old_type: SQLType
    new_type: SQLType
    coercion_expr: Optional[str] = None

    def inverse(self) -> SchemaOperation:
        inv_coercion = None
        if self.coercion_expr:
            inv_coercion = f"CAST({{col}} AS {self.old_type.value})"
        return ChangeType(
            column_name=self.column_name,
            old_type=self.new_type,
            new_type=self.old_type,
            coercion_expr=inv_coercion,
        )

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        col = s.get_column(self.column_name)
        if col is not None:
            new_col = col.with_type(self.new_type)
            s.columns[self.column_name] = new_col
        return s

    def affected_columns(self) -> Set[str]:
        return {self.column_name}

    def is_identity(self) -> bool:
        return self.old_type == self.new_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "CHANGE_TYPE",
            "column_name": self.column_name,
            "old_type": self.old_type.value,
            "new_type": self.new_type.value,
            "coercion_expr": self.coercion_expr,
        }

    def _key(self) -> tuple:
        return (self.column_name, self.old_type, self.new_type, self.coercion_expr)

    def __repr__(self) -> str:
        return f"CHANGE_TYPE {self.column_name}: {self.old_type.value} -> {self.new_type.value}"


@dataclass(frozen=True)
class AddConstraint(SchemaOperation):
    """Add a constraint to a schema."""
    constraint_id: str
    constraint_type: ConstraintType
    predicate: Optional[str] = None
    columns: Tuple[str, ...] = ()

    def inverse(self) -> SchemaOperation:
        return DropConstraint(constraint_id=self.constraint_id)

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        cdef = ConstraintDef(
            constraint_id=self.constraint_id,
            constraint_type=self.constraint_type,
            columns=self.columns,
            predicate=self.predicate,
        )
        s.add_constraint(cdef)
        if self.constraint_type == ConstraintType.NOT_NULL:
            for col_name in self.columns:
                col = s.get_column(col_name)
                if col is not None:
                    s.columns[col_name] = ColumnDef(
                        name=col.name,
                        sql_type=col.sql_type,
                        nullable=False,
                        default_expr=col.default_expr,
                        position=col.position,
                    )
        return s

    def affected_columns(self) -> Set[str]:
        return set(self.columns)

    def is_identity(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "ADD_CONSTRAINT",
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "predicate": self.predicate,
            "columns": list(self.columns),
        }

    def _key(self) -> tuple:
        return (self.constraint_id, self.constraint_type, self.predicate, self.columns)

    def __repr__(self) -> str:
        return (
            f"ADD_CONSTRAINT {self.constraint_id} "
            f"{self.constraint_type.value}({', '.join(self.columns)})"
        )


@dataclass(frozen=True)
class DropConstraint(SchemaOperation):
    """Drop a constraint from a schema."""
    constraint_id: str
    _preserved_type: Optional[ConstraintType] = None
    _preserved_predicate: Optional[str] = None
    _preserved_columns: Optional[Tuple[str, ...]] = None

    def inverse(self) -> SchemaOperation:
        return AddConstraint(
            constraint_id=self.constraint_id,
            constraint_type=self._preserved_type or ConstraintType.CHECK,
            predicate=self._preserved_predicate,
            columns=self._preserved_columns or (),
        )

    def apply(self, schema: Schema) -> Schema:
        s = schema.copy()
        s.drop_constraint(self.constraint_id)
        return s

    def with_preserved_info(
        self,
        constraint_type: ConstraintType,
        predicate: Optional[str] = None,
        columns: Optional[Tuple[str, ...]] = None,
    ) -> DropConstraint:
        return DropConstraint(
            constraint_id=self.constraint_id,
            _preserved_type=constraint_type,
            _preserved_predicate=predicate,
            _preserved_columns=columns,
        )

    def affected_columns(self) -> Set[str]:
        if self._preserved_columns:
            return set(self._preserved_columns)
        return set()

    def is_identity(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "DROP_CONSTRAINT",
            "constraint_id": self.constraint_id,
        }

    def _key(self) -> tuple:
        return (self.constraint_id,)

    def __repr__(self) -> str:
        return f"DROP_CONSTRAINT {self.constraint_id}"


# ---------------------------------------------------------------------------
# Conflict Resolution Helpers
# ---------------------------------------------------------------------------

def _resolve_add_drop_same_column(
    ops: List[SchemaOperation],
) -> List[SchemaOperation]:
    """ADD then DROP same column = no-op."""
    result: List[SchemaOperation] = []
    skip_indices: Set[int] = set()
    for i, op in enumerate(ops):
        if i in skip_indices:
            continue
        if isinstance(op, AddColumn):
            dropped = False
            for j in range(i + 1, len(ops)):
                if j in skip_indices:
                    continue
                if isinstance(ops[j], DropColumn) and ops[j].name == op.name:
                    skip_indices.add(j)
                    dropped = True
                    break
            if not dropped:
                result.append(op)
        else:
            result.append(op)
    return result


def _resolve_drop_add_same_column(
    ops: List[SchemaOperation],
) -> List[SchemaOperation]:
    """DROP then ADD same column = type replacement (ChangeType)."""
    result: List[SchemaOperation] = []
    skip_indices: Set[int] = set()
    for i, op in enumerate(ops):
        if i in skip_indices:
            continue
        if isinstance(op, DropColumn):
            replaced = False
            for j in range(i + 1, len(ops)):
                if j in skip_indices:
                    continue
                if isinstance(ops[j], AddColumn) and ops[j].name == op.name:
                    old_type = op._preserved_type or SQLType.UNKNOWN
                    new_type = ops[j].sql_type
                    if old_type != new_type:
                        result.append(
                            ChangeType(
                                column_name=op.name,
                                old_type=old_type,
                                new_type=new_type,
                                coercion_expr=f"CAST({{col}} AS {new_type.value})",
                            )
                        )
                    skip_indices.add(j)
                    replaced = True
                    break
            if not replaced:
                result.append(op)
        else:
            result.append(op)
    return result


def _chain_renames(ops: List[SchemaOperation]) -> List[SchemaOperation]:
    """RENAME a→b then RENAME b→c = RENAME a→c."""
    rename_chains: Dict[str, str] = {}
    origins: Dict[str, str] = {}
    result: List[SchemaOperation] = []
    rename_indices: Set[int] = set()
    for i, op in enumerate(ops):
        if isinstance(op, RenameColumn):
            rename_indices.add(i)
            if op.old_name in rename_chains.values():
                for orig, curr in list(rename_chains.items()):
                    if curr == op.old_name:
                        rename_chains[orig] = op.new_name
                        break
            else:
                rename_chains[op.old_name] = op.new_name
    for i, op in enumerate(ops):
        if i in rename_indices:
            continue
        result.append(op)
    for orig, final in rename_chains.items():
        if orig != final:
            result.append(RenameColumn(old_name=orig, new_name=final))
    return result


def _merge_change_types(ops: List[SchemaOperation]) -> List[SchemaOperation]:
    """CHANGE_TYPE twice on same column = single CHANGE_TYPE with composed coercion."""
    type_changes: Dict[str, List[Tuple[int, ChangeType]]] = {}
    for i, op in enumerate(ops):
        if isinstance(op, ChangeType):
            type_changes.setdefault(op.column_name, []).append((i, op))
    skip_indices: Set[int] = set()
    merged: Dict[str, ChangeType] = {}
    for col, changes in type_changes.items():
        if len(changes) > 1:
            first_idx, first_ct = changes[0]
            for idx, ct in changes[1:]:
                skip_indices.add(idx)
            last_ct = changes[-1][1]
            if first_ct.coercion_expr and last_ct.coercion_expr:
                composed = last_ct.coercion_expr.replace(
                    "{col}", first_ct.coercion_expr
                )
            elif last_ct.coercion_expr:
                composed = last_ct.coercion_expr
            else:
                composed = first_ct.coercion_expr
            merged[col] = ChangeType(
                column_name=col,
                old_type=first_ct.old_type,
                new_type=last_ct.new_type,
                coercion_expr=composed,
            )
            skip_indices.add(first_idx)
    result: List[SchemaOperation] = []
    merged_added: Set[str] = set()
    for i, op in enumerate(ops):
        if i in skip_indices:
            if isinstance(op, ChangeType) and op.column_name in merged and op.column_name not in merged_added:
                result.append(merged[op.column_name])
                merged_added.add(op.column_name)
            continue
        result.append(op)
    return result


def _remove_identity_ops(ops: List[SchemaOperation]) -> List[SchemaOperation]:
    """Remove operations that are identity/no-ops."""
    return [op for op in ops if not op.is_identity()]


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------

def _detect_conflicts(
    ops1: List[SchemaOperation],
    ops2: List[SchemaOperation],
) -> List[Conflict]:
    """Detect conflicts between two operation sequences."""
    conflicts: List[Conflict] = []

    for i, op1 in enumerate(ops1):
        for j, op2 in enumerate(ops2):
            cols1 = op1.affected_columns()
            cols2 = op2.affected_columns()
            overlap = cols1 & cols2

            if not overlap:
                continue

            for col in overlap:
                if isinstance(op1, AddColumn) and isinstance(op2, AddColumn):
                    if op1.name == op2.name:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.COLUMN_ALREADY_EXISTS,
                                description=f"Both add column '{col}'",
                                op1_index=i,
                                op2_index=j,
                                column=col,
                            )
                        )

                if isinstance(op1, DropColumn) and isinstance(op2, DropColumn):
                    if op1.name == op2.name:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.COLUMN_NOT_FOUND,
                                description=f"Both drop column '{col}'",
                                op1_index=i,
                                op2_index=j,
                                column=col,
                            )
                        )

                if isinstance(op1, DropColumn) and isinstance(op2, RenameColumn):
                    if op1.name == op2.old_name:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.DROP_DEPENDENCY,
                                description=f"Drop of '{col}' conflicts with rename",
                                op1_index=i,
                                op2_index=j,
                                column=col,
                            )
                        )

                if isinstance(op1, RenameColumn) and isinstance(op2, DropColumn):
                    if op1.old_name == op2.name:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.DROP_DEPENDENCY,
                                description=f"Rename of '{col}' conflicts with drop",
                                op1_index=i,
                                op2_index=j,
                                column=col,
                            )
                        )

                if isinstance(op1, RenameColumn) and isinstance(op2, RenameColumn):
                    if op1.old_name == op2.old_name and op1.new_name != op2.new_name:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.RENAME_CONFLICT,
                                description=(
                                    f"Conflicting renames of '{op1.old_name}': "
                                    f"'{op1.new_name}' vs '{op2.new_name}'"
                                ),
                                op1_index=i,
                                op2_index=j,
                                column=op1.old_name,
                            )
                        )

                if isinstance(op1, ChangeType) and isinstance(op2, ChangeType):
                    if op1.column_name == op2.column_name:
                        if op1.new_type != op2.new_type:
                            conflicts.append(
                                Conflict(
                                    conflict_type=ConflictType.TYPE_INCOMPATIBLE,
                                    description=(
                                        f"Conflicting type changes on '{col}': "
                                        f"{op1.new_type.value} vs {op2.new_type.value}"
                                    ),
                                    op1_index=i,
                                    op2_index=j,
                                    column=col,
                                )
                            )

                if isinstance(op1, AddConstraint) and isinstance(op2, AddConstraint):
                    if op1.constraint_id == op2.constraint_id:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.CONSTRAINT_VIOLATION,
                                description=f"Both add constraint '{op1.constraint_id}'",
                                op1_index=i,
                                op2_index=j,
                                constraint_id=op1.constraint_id,
                            )
                        )

    return conflicts


# ---------------------------------------------------------------------------
# Schema Delta Monoid
# ---------------------------------------------------------------------------

class SchemaDelta:
    """
    Represents an element of the schema delta monoid Δ_S.

    The monoid operation is composition (∘), the identity element is an
    empty operation list, and every delta has a well-defined inverse.

    Operations are stored in application order: ops[0] is applied first.
    """

    __slots__ = ("_operations", "_hash_cache")

    def __init__(self, operations: Optional[List[SchemaOperation]] = None) -> None:
        self._operations: List[SchemaOperation] = list(operations) if operations else []
        self._hash_cache: Optional[int] = None

    @property
    def operations(self) -> List[SchemaOperation]:
        return list(self._operations)

    @staticmethod
    def identity() -> SchemaDelta:
        """Return the identity element of the monoid."""
        return SchemaDelta([])

    @staticmethod
    def from_operation(op: SchemaOperation) -> SchemaDelta:
        """Create a schema delta from a single operation."""
        return SchemaDelta([op])

    @staticmethod
    def from_operations(ops: Sequence[SchemaOperation]) -> SchemaDelta:
        """Create a schema delta from a sequence of operations."""
        return SchemaDelta(list(ops))

    def compose(self, other: SchemaDelta) -> SchemaDelta:
        """
        Monoid operation: self ∘ other.

        Applies self first, then other, with conflict resolution:
        - ADD then DROP same column => no-op
        - DROP then ADD same column => type replacement
        - Chained renames are composed
        - Multiple type changes on the same column are merged
        """
        combined = list(self._operations) + list(other._operations)

        combined = _resolve_add_drop_same_column(combined)
        combined = _resolve_drop_add_same_column(combined)
        combined = _chain_renames(combined)
        combined = _merge_change_types(combined)
        combined = _remove_identity_ops(combined)

        return SchemaDelta(combined)

    def inverse(self) -> SchemaDelta:
        """
        Compute the inverse delta.

        For each operation, its inverse is:
        - AddColumn → DropColumn
        - DropColumn → AddColumn (using preserved info)
        - RenameColumn(a,b) → RenameColumn(b,a)
        - ChangeType(T1,T2) → ChangeType(T2,T1)
        - AddConstraint → DropConstraint
        - DropConstraint → AddConstraint (using preserved info)
        """
        inv_ops = [op.inverse() for op in reversed(self._operations)]
        return SchemaDelta(inv_ops)

    def normalize(self) -> SchemaDelta:
        """
        Normalize to canonical form:
        1. Remove identity operations
        2. Chain renames
        3. Merge type changes
        4. Cancel ADD+DROP of same column
        5. Order: drops, renames, type changes, constraint drops, adds, constraint adds
        """
        ops = list(self._operations)
        ops = _remove_identity_ops(ops)
        ops = _resolve_add_drop_same_column(ops)
        ops = _resolve_drop_add_same_column(ops)
        ops = _chain_renames(ops)
        ops = _merge_change_types(ops)
        ops = _remove_identity_ops(ops)

        drops: List[SchemaOperation] = []
        renames: List[SchemaOperation] = []
        type_changes: List[SchemaOperation] = []
        constraint_drops: List[SchemaOperation] = []
        adds: List[SchemaOperation] = []
        constraint_adds: List[SchemaOperation] = []

        for op in ops:
            if isinstance(op, DropColumn):
                drops.append(op)
            elif isinstance(op, RenameColumn):
                renames.append(op)
            elif isinstance(op, ChangeType):
                type_changes.append(op)
            elif isinstance(op, DropConstraint):
                constraint_drops.append(op)
            elif isinstance(op, AddColumn):
                adds.append(op)
            elif isinstance(op, AddConstraint):
                constraint_adds.append(op)
            else:
                adds.append(op)

        ordered = drops + renames + type_changes + constraint_drops + adds + constraint_adds
        return SchemaDelta(ordered)

    def is_identity(self) -> bool:
        """Check if this is the identity element (no operations)."""
        normalized = self.normalize()
        return len(normalized._operations) == 0

    def apply_to_schema(self, schema: Schema) -> Schema:
        """Apply this delta to a schema, producing a new schema."""
        result = schema.copy()
        for op in self._operations:
            result = op.apply(result)
        return result

    def conflicts_with(self, other: SchemaDelta) -> List[Conflict]:
        """Detect conflicts between this delta and another."""
        return _detect_conflicts(self._operations, other._operations)

    def affected_columns(self) -> Set[str]:
        """Return all column names affected by any operation."""
        result: Set[str] = set()
        for op in self._operations:
            result |= op.affected_columns()
        return result

    def affected_constraints(self) -> Set[str]:
        """Return all constraint IDs affected by any operation."""
        result: Set[str] = set()
        for op in self._operations:
            if isinstance(op, (AddConstraint, DropConstraint)):
                result.add(op.constraint_id)
        return result

    def operation_count(self) -> int:
        """Return the number of operations."""
        return len(self._operations)

    def filter_by_column(self, column: str) -> SchemaDelta:
        """Return a sub-delta containing only operations affecting the given column."""
        filtered = [
            op for op in self._operations
            if column in op.affected_columns()
        ]
        return SchemaDelta(filtered)

    def split_by_column(self) -> Dict[str, SchemaDelta]:
        """Split into per-column deltas."""
        result: Dict[str, List[SchemaOperation]] = {}
        for op in self._operations:
            for col in op.affected_columns():
                result.setdefault(col, []).append(op)
        return {col: SchemaDelta(ops) for col, ops in result.items()}

    def contains_operation_type(self, op_type: type) -> bool:
        """Check if any operation is of the given type."""
        return any(isinstance(op, op_type) for op in self._operations)

    def get_operations_of_type(self, op_type: type) -> List[SchemaOperation]:
        """Return all operations of the given type."""
        return [op for op in self._operations if isinstance(op, op_type)]

    def apply_rename_mapping(self) -> Dict[str, str]:
        """Extract the cumulative rename mapping from this delta."""
        mapping: Dict[str, str] = {}
        for op in self._operations:
            if isinstance(op, RenameColumn):
                found_origin = None
                for orig, curr in list(mapping.items()):
                    if curr == op.old_name:
                        found_origin = orig
                        break
                if found_origin:
                    mapping[found_origin] = op.new_name
                else:
                    mapping[op.old_name] = op.new_name
        return mapping

    def type_change_mapping(self) -> Dict[str, Tuple[SQLType, SQLType]]:
        """Extract the cumulative type change mapping."""
        result: Dict[str, Tuple[SQLType, SQLType]] = {}
        for op in self._operations:
            if isinstance(op, ChangeType):
                if op.column_name in result:
                    old_old, _ = result[op.column_name]
                    result[op.column_name] = (old_old, op.new_type)
                else:
                    result[op.column_name] = (op.old_type, op.new_type)
        return result

    def added_columns(self) -> List[AddColumn]:
        """Return list of added columns."""
        return [op for op in self._operations if isinstance(op, AddColumn)]

    def dropped_columns(self) -> List[DropColumn]:
        """Return list of dropped columns."""
        return [op for op in self._operations if isinstance(op, DropColumn)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operations": [op.to_dict() for op in self._operations],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SchemaDelta:
        """Deserialize from dictionary."""
        op_map = {
            "ADD_COLUMN": lambda d: AddColumn(
                name=d["name"],
                sql_type=SQLType(d["sql_type"]),
                position=d.get("position", -1),
                default_expr=d.get("default_expr"),
                nullable=d.get("nullable", True),
            ),
            "DROP_COLUMN": lambda d: DropColumn(name=d["name"]),
            "RENAME_COLUMN": lambda d: RenameColumn(
                old_name=d["old_name"], new_name=d["new_name"]
            ),
            "CHANGE_TYPE": lambda d: ChangeType(
                column_name=d["column_name"],
                old_type=SQLType(d["old_type"]),
                new_type=SQLType(d["new_type"]),
                coercion_expr=d.get("coercion_expr"),
            ),
            "ADD_CONSTRAINT": lambda d: AddConstraint(
                constraint_id=d["constraint_id"],
                constraint_type=ConstraintType(d["constraint_type"]),
                predicate=d.get("predicate"),
                columns=tuple(d.get("columns", [])),
            ),
            "DROP_CONSTRAINT": lambda d: DropConstraint(
                constraint_id=d["constraint_id"]
            ),
        }
        ops = []
        for op_dict in data.get("operations", []):
            factory = op_map.get(op_dict["op"])
            if factory:
                ops.append(factory(op_dict))
        return SchemaDelta(ops)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaDelta):
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
            return "SchemaDelta(identity)"
        ops_str = ", ".join(repr(op) for op in self._operations)
        return f"SchemaDelta([{ops_str}])"

    def __len__(self) -> int:
        return len(self._operations)

    def __bool__(self) -> bool:
        return not self.is_identity()

    def __iter__(self):
        return iter(self._operations)


# ---------------------------------------------------------------------------
# Schema Diff Utility
# ---------------------------------------------------------------------------

def diff_schemas(old_schema: Schema, new_schema: Schema) -> SchemaDelta:
    """
    Compute the SchemaDelta that transforms old_schema into new_schema.

    Detects: dropped columns, added columns, type changes, renamed columns
    (heuristic: same type + position suggests rename), constraint changes.
    """
    ops: List[SchemaOperation] = []

    old_cols = set(old_schema.column_names())
    new_cols = set(new_schema.column_names())

    dropped = old_cols - new_cols
    added = new_cols - old_cols
    common = old_cols & new_cols

    rename_pairs: List[Tuple[str, str]] = []
    unmatched_dropped: Set[str] = set()
    unmatched_added: Set[str] = set()

    if dropped and added:
        dropped_list = sorted(dropped)
        added_list = sorted(added)
        used_added: Set[str] = set()
        for d_col in dropped_list:
            d_def = old_schema.get_column(d_col)
            if d_def is None:
                unmatched_dropped.add(d_col)
                continue
            best_match = None
            for a_col in added_list:
                if a_col in used_added:
                    continue
                a_def = new_schema.get_column(a_col)
                if a_def is None:
                    continue
                if d_def.sql_type == a_def.sql_type:
                    best_match = a_col
                    break
            if best_match:
                rename_pairs.append((d_col, best_match))
                used_added.add(best_match)
            else:
                unmatched_dropped.add(d_col)
        unmatched_added = added - used_added
    else:
        unmatched_dropped = dropped
        unmatched_added = added

    for col in sorted(unmatched_dropped):
        col_def = old_schema.get_column(col)
        if col_def:
            ops.append(
                DropColumn(
                    name=col,
                    _preserved_type=col_def.sql_type,
                    _preserved_nullable=col_def.nullable,
                    _preserved_default=col_def.default_expr,
                    _preserved_position=col_def.position,
                )
            )
        else:
            ops.append(DropColumn(name=col))

    for old_name, new_name in rename_pairs:
        ops.append(RenameColumn(old_name=old_name, new_name=new_name))

    for col in common:
        old_def = old_schema.get_column(col)
        new_def = new_schema.get_column(col)
        if old_def and new_def and old_def.sql_type != new_def.sql_type:
            ops.append(
                ChangeType(
                    column_name=col,
                    old_type=old_def.sql_type,
                    new_type=new_def.sql_type,
                    coercion_expr=f"CAST({{col}} AS {new_def.sql_type.value})",
                )
            )

    for col in sorted(unmatched_added):
        col_def = new_schema.get_column(col)
        if col_def:
            ops.append(
                AddColumn(
                    name=col,
                    sql_type=col_def.sql_type,
                    position=col_def.position,
                    default_expr=col_def.default_expr,
                    nullable=col_def.nullable,
                )
            )

    old_constraints = set(old_schema.constraints.keys())
    new_constraints = set(new_schema.constraints.keys())

    for cid in sorted(old_constraints - new_constraints):
        cdef = old_schema.constraints[cid]
        ops.append(
            DropConstraint(
                constraint_id=cid,
                _preserved_type=cdef.constraint_type,
                _preserved_predicate=cdef.predicate,
                _preserved_columns=cdef.columns,
            )
        )

    for cid in sorted(new_constraints - old_constraints):
        cdef = new_schema.constraints[cid]
        ops.append(
            AddConstraint(
                constraint_id=cid,
                constraint_type=cdef.constraint_type,
                predicate=cdef.predicate,
                columns=cdef.columns,
            )
        )

    return SchemaDelta(ops)


# ---------------------------------------------------------------------------
# Type widening / compatibility checks
# ---------------------------------------------------------------------------

_TYPE_WIDENING: Dict[SQLType, Set[SQLType]] = {
    SQLType.SMALLINT: {SQLType.INTEGER, SQLType.BIGINT, SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC},
    SQLType.INTEGER: {SQLType.BIGINT, SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC},
    SQLType.BIGINT: {SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC},
    SQLType.FLOAT: {SQLType.DOUBLE},
    SQLType.CHAR: {SQLType.VARCHAR, SQLType.TEXT},
    SQLType.VARCHAR: {SQLType.TEXT},
    SQLType.DATE: {SQLType.TIMESTAMP, SQLType.TIMESTAMPTZ},
    SQLType.TIMESTAMP: {SQLType.TIMESTAMPTZ},
    SQLType.JSON: {SQLType.JSONB},
}


def can_widen_type(from_type: SQLType, to_type: SQLType) -> bool:
    """Check if from_type can be safely widened to to_type."""
    if from_type == to_type:
        return True
    return to_type in _TYPE_WIDENING.get(from_type, set())


def widest_type(t1: SQLType, t2: SQLType) -> SQLType:
    """Return the widest type that encompasses both t1 and t2."""
    if t1 == t2:
        return t1
    if can_widen_type(t1, t2):
        return t2
    if can_widen_type(t2, t1):
        return t1
    if t1 in (SQLType.INTEGER, SQLType.BIGINT, SQLType.FLOAT, SQLType.DOUBLE):
        if t2 in (SQLType.INTEGER, SQLType.BIGINT, SQLType.FLOAT, SQLType.DOUBLE):
            return SQLType.DOUBLE
    return SQLType.TEXT


def coercion_expression(from_type: SQLType, to_type: SQLType) -> Optional[str]:
    """Return a coercion expression for type conversion, or None if direct."""
    if from_type == to_type:
        return None
    if can_widen_type(from_type, to_type):
        return None
    return f"CAST({{col}} AS {to_type.value})"
