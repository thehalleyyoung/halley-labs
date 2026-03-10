"""
Schema Catalog — Track and Manage Schemas Across Pipeline Stages
=================================================================

Provides a central registry for table schemas, schema versioning,
schema diffing, compatibility checking, schema inference from SQL,
and edge-level schema validation for pipeline graphs.

The catalog maintains a complete history of schema changes for each
registered table, enabling time-travel queries and delta computation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from arc.algebra.schema_delta import (
    AddColumn,
    AddConstraint,
    ChangeType,
    ColumnDef,
    ConstraintType,
    DropColumn,
    DropConstraint,
    RenameColumn,
    Schema,
    SchemaDelta,
    SchemaOperation,
    SQLType,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Schema Version
# =====================================================================


@dataclass
class SchemaVersion:
    """A versioned snapshot of a table schema.

    Attributes
    ----------
    schema : Schema
        The schema at this version.
    delta_from_previous : SchemaDelta | None
        Delta that produced this version from the previous one.
    timestamp : datetime
        When this version was created.
    source : str
        What triggered the schema change (e.g., "migration", "inference").
    version_number : int
        Sequential version number.
    checksum : str
        Hash of the schema for quick comparison.
    """
    schema: Schema = field(default_factory=Schema)
    delta_from_previous: Optional[SchemaDelta] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    version_number: int = 0
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a deterministic hash of the schema."""
        parts: List[str] = []
        for col in self.schema.columns:
            parts.append(f"{col.name}:{col.sql_type.value}:{col.nullable}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        n_cols = len(self.schema.columns)
        return (
            f"SchemaVersion(v{self.version_number}, "
            f"{n_cols} columns, source={self.source!r})"
        )


# =====================================================================
# Schema Violation
# =====================================================================


class ViolationSeverity(Enum):
    """Severity of a schema violation."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SchemaViolation:
    """A schema compatibility violation on a pipeline edge.

    Attributes
    ----------
    source_node : str
        Source node of the edge.
    target_node : str
        Target node of the edge.
    violation_type : str
        Type of violation (e.g., "missing_column", "type_mismatch").
    details : str
        Human-readable description.
    severity : ViolationSeverity
        How serious the violation is.
    affected_columns : list[str]
        Columns involved in the violation.
    """
    source_node: str = ""
    target_node: str = ""
    violation_type: str = ""
    details: str = ""
    severity: ViolationSeverity = ViolationSeverity.ERROR
    affected_columns: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SchemaViolation({self.source_node} -> {self.target_node}: "
            f"{self.violation_type} [{self.severity.value}])"
        )


# =====================================================================
# Schema Diff
# =====================================================================


@dataclass
class SchemaDiff:
    """Detailed diff between two schemas.

    Attributes
    ----------
    added_columns : list[ColumnDef]
        Columns present in schema2 but not schema1.
    dropped_columns : list[ColumnDef]
        Columns present in schema1 but not schema2.
    renamed_columns : list[tuple[str, str]]
        Detected column renames (old_name, new_name).
    type_changes : list[tuple[str, SQLType, SQLType]]
        Columns with changed types (name, old_type, new_type).
    nullable_changes : list[tuple[str, bool, bool]]
        Columns with changed nullability (name, old, new).
    constraint_changes : list[str]
        Description of constraint changes.
    is_compatible : bool
        True if schema2 is a compatible evolution of schema1.
    is_identical : bool
        True if the schemas are exactly the same.
    """
    added_columns: List[ColumnDef] = field(default_factory=list)
    dropped_columns: List[ColumnDef] = field(default_factory=list)
    renamed_columns: List[Tuple[str, str]] = field(default_factory=list)
    type_changes: List[Tuple[str, Any, Any]] = field(default_factory=list)
    nullable_changes: List[Tuple[str, bool, bool]] = field(default_factory=list)
    constraint_changes: List[str] = field(default_factory=list)
    is_compatible: bool = True
    is_identical: bool = False

    def to_delta(self) -> SchemaDelta:
        """Convert this diff to a SchemaDelta."""
        ops: List[SchemaOperation] = []

        for col in self.added_columns:
            ops.append(AddColumn(column_def=col))

        for col in self.dropped_columns:
            ops.append(DropColumn(column_name=col.name))

        for old_name, new_name in self.renamed_columns:
            ops.append(RenameColumn(old_name=old_name, new_name=new_name))

        for col_name, _, new_type in self.type_changes:
            ops.append(ChangeType(column_name=col_name, new_type=new_type))

        return SchemaDelta(operations=ops)

    @property
    def has_changes(self) -> bool:
        return not self.is_identical

    def summary(self) -> str:
        if self.is_identical:
            return "Schemas are identical"

        parts = []
        if self.added_columns:
            parts.append(f"+{len(self.added_columns)} columns")
        if self.dropped_columns:
            parts.append(f"-{len(self.dropped_columns)} columns")
        if self.renamed_columns:
            parts.append(f"~{len(self.renamed_columns)} renamed")
        if self.type_changes:
            parts.append(f"T{len(self.type_changes)} type changes")
        if self.nullable_changes:
            parts.append(f"N{len(self.nullable_changes)} nullable changes")

        compat = "compatible" if self.is_compatible else "BREAKING"
        return f"SchemaDiff({', '.join(parts)}) [{compat}]"


# =====================================================================
# Schema Catalog
# =====================================================================


class SchemaCatalog:
    """Track and manage schemas across pipeline stages.

    Maintains a versioned registry of table schemas, supports schema
    diffing, compatibility checking, inference from SQL, and edge-level
    validation for pipeline graphs.

    Parameters
    ----------
    enable_history : bool
        Whether to maintain full schema history.
    max_history_per_table : int
        Maximum number of versions to keep per table.
    """

    def __init__(
        self,
        enable_history: bool = True,
        max_history_per_table: int = 100,
    ) -> None:
        self._schemas: Dict[str, Schema] = {}
        self._history: Dict[str, List[SchemaVersion]] = defaultdict(list)
        self._enable_history = enable_history
        self._max_history = max_history_per_table
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # ── Registration and Retrieval ────────────────────────────────

    def register_schema(
        self,
        table_name: str,
        schema: Schema,
        source: str = "manual",
    ) -> None:
        """Register or update a table schema.

        Parameters
        ----------
        table_name : str
            Table name to register.
        schema : Schema
            The schema to register.
        source : str
            What triggered the registration.
        """
        old_schema = self._schemas.get(table_name)
        self._schemas[table_name] = schema

        if self._enable_history:
            delta = None
            if old_schema is not None:
                diff = self.diff_schemas(old_schema, schema)
                if diff.has_changes:
                    delta = diff.to_delta()

            version = SchemaVersion(
                schema=schema,
                delta_from_previous=delta,
                source=source,
                version_number=len(self._history[table_name]),
            )
            self._history[table_name].append(version)

            if len(self._history[table_name]) > self._max_history:
                self._history[table_name] = (
                    self._history[table_name][-self._max_history:]
                )

        logger.debug("Registered schema for %s (%d columns)", table_name, len(schema.columns))

    def get_schema(self, table_name: str) -> Optional[Schema]:
        """Get the current schema for a table.

        Parameters
        ----------
        table_name : str
            Table name.

        Returns
        -------
        Schema | None
        """
        return self._schemas.get(table_name)

    def has_schema(self, table_name: str) -> bool:
        """Check if a schema is registered for a table."""
        return table_name in self._schemas

    def list_tables(self) -> List[str]:
        """List all registered table names."""
        return sorted(self._schemas.keys())

    def remove_schema(self, table_name: str) -> Optional[Schema]:
        """Remove and return a registered schema."""
        return self._schemas.pop(table_name, None)

    def apply_delta(
        self,
        table_name: str,
        delta: SchemaDelta,
        source: str = "delta_application",
    ) -> Schema:
        """Apply a schema delta to a registered table.

        Parameters
        ----------
        table_name : str
            Table name.
        delta : SchemaDelta
            Schema changes to apply.
        source : str
            What triggered the change.

        Returns
        -------
        Schema
            The updated schema.

        Raises
        ------
        KeyError
            If the table is not registered.
        """
        current = self._schemas.get(table_name)
        if current is None:
            raise KeyError(f"Table {table_name!r} not registered")

        new_schema = self._apply_delta_to_schema(current, delta)
        self.register_schema(table_name, new_schema, source=source)
        return new_schema

    def get_schema_history(
        self,
        table_name: str,
    ) -> List[SchemaVersion]:
        """Get the full schema history for a table.

        Parameters
        ----------
        table_name : str
            Table name.

        Returns
        -------
        list[SchemaVersion]
        """
        return list(self._history.get(table_name, []))

    def get_schema_at_version(
        self,
        table_name: str,
        version: int,
    ) -> Optional[Schema]:
        """Get the schema at a specific version number.

        Parameters
        ----------
        table_name : str
            Table name.
        version : int
            Version number.

        Returns
        -------
        Schema | None
        """
        history = self._history.get(table_name, [])
        for sv in history:
            if sv.version_number == version:
                return sv.schema
        return None

    # ── Schema Diffing ────────────────────────────────────────────

    def diff_schemas(
        self,
        schema1: Schema,
        schema2: Schema,
    ) -> SchemaDiff:
        """Compute the diff between two schemas.

        Detects added columns, dropped columns, renames, type changes,
        and nullability changes.

        Parameters
        ----------
        schema1 : Schema
            The "old" schema.
        schema2 : Schema
            The "new" schema.

        Returns
        -------
        SchemaDiff
        """
        cols1 = {c.name: c for c in schema1.columns}
        cols2 = {c.name: c for c in schema2.columns}

        names1 = set(cols1.keys())
        names2 = set(cols2.keys())

        added_names = names2 - names1
        dropped_names = names1 - names2
        common_names = names1 & names2

        added = [cols2[n] for n in sorted(added_names)]
        dropped = [cols1[n] for n in sorted(dropped_names)]

        renamed: List[Tuple[str, str]] = []
        if added_names and dropped_names:
            renamed = self._detect_renames(
                [cols1[n] for n in dropped_names],
                [cols2[n] for n in added_names],
            )
            renamed_old = {r[0] for r in renamed}
            renamed_new = {r[1] for r in renamed}
            added = [c for c in added if c.name not in renamed_new]
            dropped = [c for c in dropped if c.name not in renamed_old]

        type_changes: List[Tuple[str, Any, Any]] = []
        nullable_changes: List[Tuple[str, bool, bool]] = []

        for name in sorted(common_names):
            c1 = cols1[name]
            c2 = cols2[name]

            if c1.sql_type != c2.sql_type:
                type_changes.append((name, c1.sql_type, c2.sql_type))

            if c1.nullable != c2.nullable:
                nullable_changes.append((name, c1.nullable, c2.nullable))

        is_identical = (
            not added and not dropped and not renamed
            and not type_changes and not nullable_changes
        )

        is_compatible = True
        if dropped:
            is_compatible = False
        for _, old_nullable, new_nullable in nullable_changes:
            if old_nullable and not new_nullable:
                is_compatible = False
                break

        return SchemaDiff(
            added_columns=added,
            dropped_columns=dropped,
            renamed_columns=renamed,
            type_changes=type_changes,
            nullable_changes=nullable_changes,
            is_compatible=is_compatible,
            is_identical=is_identical,
        )

    def compatible(
        self,
        schema1: Schema,
        schema2: Schema,
    ) -> bool:
        """Check if schema2 is compatible with schema1.

        Compatible means schema2 can be used where schema1 was expected:
        - All columns in schema1 exist in schema2 (possibly with wider types)
        - No non-nullable column became nullable

        Parameters
        ----------
        schema1 : Schema
            Expected schema.
        schema2 : Schema
            Actual schema.

        Returns
        -------
        bool
        """
        diff = self.diff_schemas(schema1, schema2)
        return diff.is_compatible

    # ── Schema Inference ──────────────────────────────────────────

    def infer_schema_from_sql(
        self,
        query: str,
        input_schemas: Optional[Dict[str, Schema]] = None,
    ) -> Schema:
        """Infer the output schema of a SQL query.

        Uses input schemas and SQL parsing to determine the output columns
        and their types.

        Parameters
        ----------
        query : str
            The SQL query.
        input_schemas : dict[str, Schema], optional
            Schemas of input tables referenced in the query.

        Returns
        -------
        Schema
        """
        inputs = input_schemas or {}
        inputs.update(self._schemas)

        output_columns = self._parse_select_columns(query, inputs)

        return Schema(columns=output_columns)

    def merge_schemas(
        self,
        schemas: List[Schema],
    ) -> Schema:
        """Merge multiple schemas into a single unified schema.

        For columns with the same name, uses the widest type. Nullable
        if any input has nullable.

        Parameters
        ----------
        schemas : list[Schema]
            Schemas to merge.

        Returns
        -------
        Schema
        """
        if not schemas:
            return Schema(columns=[])

        if len(schemas) == 1:
            return schemas[0]

        all_columns: Dict[str, ColumnDef] = OrderedDict()

        for schema in schemas:
            for col in schema.columns:
                if col.name not in all_columns:
                    all_columns[col.name] = col
                else:
                    existing = all_columns[col.name]
                    merged = self._merge_column_defs(existing, col)
                    all_columns[col.name] = merged

        return Schema(columns=list(all_columns.values()))

    # ── Pipeline Validation ───────────────────────────────────────

    def validate_edge_schemas(
        self,
        graph: Any,
    ) -> List[SchemaViolation]:
        """Validate schema compatibility on all edges of a pipeline graph.

        Checks that for each edge (A → B), the output schema of A
        is compatible with the input schema of B.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph to validate.

        Returns
        -------
        list[SchemaViolation]
        """
        violations: List[SchemaViolation] = []

        for (source_id, target_id), edge in graph.edges.items():
            source_node = graph.get_node(source_id)
            target_node = graph.get_node(target_id)

            source_output = source_node.output_schema
            target_input = target_node.input_schema

            if not source_output.columns or not target_input.columns:
                continue

            edge_violations = self._check_edge_compatibility(
                source_id, target_id, source_output, target_input, edge
            )
            violations.extend(edge_violations)

        return violations

    def register_graph_schemas(
        self,
        graph: Any,
        source: str = "graph_import",
    ) -> int:
        """Register all schemas from a pipeline graph.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph.
        source : str
            Registration source.

        Returns
        -------
        int
            Number of schemas registered.
        """
        count = 0
        for nid in graph.node_ids:
            node = graph.get_node(nid)
            if node.output_schema.columns:
                self.register_schema(
                    f"{nid}_output", node.output_schema, source=source
                )
                count += 1
            if node.input_schema.columns:
                self.register_schema(
                    f"{nid}_input", node.input_schema, source=source
                )
                count += 1
        return count

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the catalog to a dictionary."""
        schemas = {}
        for name, schema in self._schemas.items():
            schemas[name] = self._schema_to_dict(schema)

        history = {}
        for name, versions in self._history.items():
            history[name] = [
                {
                    "version": sv.version_number,
                    "source": sv.source,
                    "timestamp": sv.timestamp.isoformat(),
                    "checksum": sv.checksum,
                    "columns": len(sv.schema.columns),
                }
                for sv in versions
            ]

        return {
            "schemas": schemas,
            "history": history,
            "table_count": len(self._schemas),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SchemaCatalog:
        """Deserialize a catalog from a dictionary."""
        catalog = cls()
        for name, schema_data in data.get("schemas", {}).items():
            schema = cls._schema_from_dict(schema_data)
            catalog.register_schema(name, schema, source="deserialization")
        return catalog

    # ── Internal Helpers ──────────────────────────────────────────

    def _apply_delta_to_schema(
        self,
        schema: Schema,
        delta: SchemaDelta,
    ) -> Schema:
        """Apply a schema delta to produce a new schema."""
        columns = list(schema.columns)
        col_map = {c.name: i for i, c in enumerate(columns)}

        for op in delta.operations:
            if isinstance(op, AddColumn):
                columns.append(op.column_def)
                col_map[op.column_def.name] = len(columns) - 1
            elif isinstance(op, DropColumn):
                idx = col_map.get(op.column_name)
                if idx is not None:
                    columns = [c for c in columns if c.name != op.column_name]
                    col_map = {c.name: i for i, c in enumerate(columns)}
            elif isinstance(op, RenameColumn):
                idx = col_map.get(op.old_name)
                if idx is not None:
                    old_col = columns[idx]
                    new_col = ColumnDef(
                        name=op.new_name,
                        sql_type=old_col.sql_type,
                        nullable=old_col.nullable,
                        default_expr=old_col.default_expr,
                    )
                    columns[idx] = new_col
                    del col_map[op.old_name]
                    col_map[op.new_name] = idx
            elif isinstance(op, ChangeType):
                idx = col_map.get(op.column_name)
                if idx is not None:
                    old_col = columns[idx]
                    new_col = ColumnDef(
                        name=old_col.name,
                        sql_type=op.new_type,
                        nullable=old_col.nullable,
                        default_expr=old_col.default_expr,
                    )
                    columns[idx] = new_col

        return Schema(columns=columns)

    def _detect_renames(
        self,
        dropped: List[ColumnDef],
        added: List[ColumnDef],
    ) -> List[Tuple[str, str]]:
        """Detect column renames by matching types."""
        renames: List[Tuple[str, str]] = []
        used_added: Set[str] = set()

        for d_col in dropped:
            best_match: Optional[ColumnDef] = None
            for a_col in added:
                if a_col.name in used_added:
                    continue
                if (a_col.sql_type == d_col.sql_type
                        and a_col.nullable == d_col.nullable):
                    best_match = a_col
                    break

            if best_match is not None:
                renames.append((d_col.name, best_match.name))
                used_added.add(best_match.name)

        return renames

    def _merge_column_defs(
        self,
        col1: ColumnDef,
        col2: ColumnDef,
    ) -> ColumnDef:
        """Merge two column definitions with the same name."""
        wider_type = self._wider_type(col1.sql_type, col2.sql_type)
        nullable = col1.nullable or col2.nullable
        default = col1.default_expr or col2.default_expr

        return ColumnDef(
            name=col1.name,
            sql_type=wider_type,
            nullable=nullable,
            default_expr=default,
        )

    @staticmethod
    def _wider_type(t1: SQLType, t2: SQLType) -> SQLType:
        """Return the wider of two SQL types."""
        if t1 == t2:
            return t1

        int_order = [SQLType.SMALLINT, SQLType.INTEGER, SQLType.BIGINT]
        float_order = [SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL]
        string_order = [SQLType.CHAR, SQLType.VARCHAR, SQLType.TEXT]

        for hierarchy in [int_order, float_order, string_order]:
            if t1 in hierarchy and t2 in hierarchy:
                idx1 = hierarchy.index(t1)
                idx2 = hierarchy.index(t2)
                return hierarchy[max(idx1, idx2)]

        return SQLType.TEXT

    def _parse_select_columns(
        self,
        query: str,
        input_schemas: Dict[str, Schema],
    ) -> List[ColumnDef]:
        """Parse SELECT columns from a SQL query."""
        columns: List[ColumnDef] = []

        select_match = re.search(
            r"\bSELECT\b\s+(.+?)\s+\bFROM\b",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if not select_match:
            return columns

        select_list = select_match.group(1).strip()

        if select_list.strip() == "*":
            for schema in input_schemas.values():
                columns.extend(schema.columns)
            return columns

        for item in self._split_select_items(select_list):
            item = item.strip()
            if not item:
                continue

            alias_match = re.search(r"\bAS\b\s+(\w+)\s*$", item, re.IGNORECASE)
            if alias_match:
                name = alias_match.group(1)
            else:
                parts = item.split(".")
                name = parts[-1].strip()
                name = re.sub(r"[^a-zA-Z0-9_]", "", name)

            if not name:
                name = f"col_{len(columns)}"

            col_type = self._infer_column_type(item, input_schemas)

            columns.append(ColumnDef(
                name=name,
                sql_type=col_type,
                nullable=True,
            ))

        return columns

    @staticmethod
    def _split_select_items(select_list: str) -> List[str]:
        """Split a SELECT list by commas, respecting parentheses."""
        items: List[str] = []
        depth = 0
        current: List[str] = []

        for char in select_list:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                items.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            items.append("".join(current))

        return items

    def _infer_column_type(
        self,
        expression: str,
        input_schemas: Dict[str, Schema],
    ) -> SQLType:
        """Infer the SQL type of a column expression."""
        expr_upper = expression.strip().upper()

        if re.match(r"^COUNT\s*\(", expr_upper):
            return SQLType.BIGINT
        if re.match(r"^(SUM|AVG)\s*\(", expr_upper):
            return SQLType.DOUBLE
        if re.match(r"^(MIN|MAX)\s*\(", expr_upper):
            return SQLType.VARCHAR
        if re.match(r"^CAST\s*\(.+\s+AS\s+(\w+)\)", expr_upper):
            type_match = re.search(r"AS\s+(\w+)", expr_upper)
            if type_match:
                try:
                    return SQLType(type_match.group(1))
                except ValueError:
                    pass

        col_name = expression.strip().split(".")[-1].strip()
        for schema in input_schemas.values():
            for col in schema.columns:
                if col.name.lower() == col_name.lower():
                    return col.sql_type

        return SQLType.VARCHAR

    def _check_edge_compatibility(
        self,
        source_id: str,
        target_id: str,
        source_output: Schema,
        target_input: Schema,
        edge: Any,
    ) -> List[SchemaViolation]:
        """Check schema compatibility for a single edge."""
        violations: List[SchemaViolation] = []

        source_cols = {c.name for c in source_output.columns}
        target_cols = {c.name for c in target_input.columns}

        col_mapping = getattr(edge, "column_mapping", {})

        for target_col in target_cols:
            source_col = col_mapping.get(target_col, target_col)
            if source_col not in source_cols:
                violations.append(SchemaViolation(
                    source_node=source_id,
                    target_node=target_id,
                    violation_type="missing_column",
                    details=(
                        f"Column '{source_col}' required by {target_id} "
                        f"not in output of {source_id}"
                    ),
                    severity=ViolationSeverity.ERROR,
                    affected_columns=[source_col],
                ))

        source_col_map = {c.name: c for c in source_output.columns}
        target_col_map = {c.name: c for c in target_input.columns}

        for name in source_cols & target_cols:
            s_col = source_col_map[name]
            t_col = target_col_map[name]

            if s_col.sql_type != t_col.sql_type:
                violations.append(SchemaViolation(
                    source_node=source_id,
                    target_node=target_id,
                    violation_type="type_mismatch",
                    details=(
                        f"Column '{name}': {source_id} outputs {s_col.sql_type.value} "
                        f"but {target_id} expects {t_col.sql_type.value}"
                    ),
                    severity=ViolationSeverity.WARNING,
                    affected_columns=[name],
                ))

            if s_col.nullable and not t_col.nullable:
                violations.append(SchemaViolation(
                    source_node=source_id,
                    target_node=target_id,
                    violation_type="nullable_mismatch",
                    details=(
                        f"Column '{name}': {source_id} may be NULL "
                        f"but {target_id} requires NOT NULL"
                    ),
                    severity=ViolationSeverity.WARNING,
                    affected_columns=[name],
                ))

        return violations

    @staticmethod
    def _schema_to_dict(schema: Schema) -> Dict[str, Any]:
        """Serialize a Schema to dict."""
        return {
            "columns": [
                {
                    "name": c.name,
                    "type": c.sql_type.value,
                    "nullable": c.nullable,
                    "default": c.default_expr,
                }
                for c in schema.columns
            ]
        }

    @staticmethod
    def _schema_from_dict(data: Dict[str, Any]) -> Schema:
        """Deserialize a Schema from dict."""
        columns = []
        for col_data in data.get("columns", []):
            try:
                sql_type = SQLType(col_data["type"])
            except (ValueError, KeyError):
                sql_type = SQLType.VARCHAR

            columns.append(ColumnDef(
                name=col_data["name"],
                sql_type=sql_type,
                nullable=col_data.get("nullable", True),
                default_expr=col_data.get("default"),
            ))
        return Schema(columns=columns)


# =====================================================================
# Convenience Functions
# =====================================================================


def diff_schemas(schema1: Schema, schema2: Schema) -> SchemaDiff:
    """Convenience: diff two schemas."""
    catalog = SchemaCatalog(enable_history=False)
    return catalog.diff_schemas(schema1, schema2)


def schemas_compatible(schema1: Schema, schema2: Schema) -> bool:
    """Convenience: check schema compatibility."""
    catalog = SchemaCatalog(enable_history=False)
    return catalog.compatible(schema1, schema2)


def merge_schemas(schemas: List[Schema]) -> Schema:
    """Convenience: merge multiple schemas."""
    catalog = SchemaCatalog(enable_history=False)
    return catalog.merge_schemas(schemas)
