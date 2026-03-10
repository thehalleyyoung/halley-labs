"""
DuckDB-based repair plan execution engine.

:class:`ExecutionEngine` connects to a DuckDB instance (in-memory or
file-backed), executes individual repair actions or complete plans,
applies schema and data deltas, and provides table introspection.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from arc.types.base import (
    ActionResult,
    ActionType,
    CompoundPerturbation,
    DataDelta,
    ExecutionError as ExecutionErrorInfo,
    ExecutionResult,
    PipelineGraph,
    PipelineNode,
    RepairAction,
    RepairPlan,
    RowChangeType,
    Schema,
    SchemaDelta,
    SchemaOpType,
    SQLOperator,
    TableStats,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """DuckDB-based repair plan execution engine.

    Parameters
    ----------
    database_path:
        Path to a DuckDB database file.  *None* creates an in-memory
        database.
    read_only:
        Open the database in read-only mode.
    """

    def __init__(
        self,
        database_path: str | None = None,
        read_only: bool = False,
    ) -> None:
        self._db_path = database_path or ":memory:"
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(
            self._db_path, read_only=read_only,
        )
        self._registered_sources: dict[str, Any] = {}
        logger.info("ExecutionEngine connected to %s", self._db_path)

    # ── Connection management ──────────────────────────────────────────

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the underlying DuckDB connection."""
        return self._conn

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
        logger.info("ExecutionEngine closed.")

    def __enter__(self) -> "ExecutionEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ── Plan execution ─────────────────────────────────────────────────

    def execute_plan(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
        validate: bool = True,
    ) -> ExecutionResult:
        """Execute a complete repair plan.

        Parameters
        ----------
        plan:
            The repair plan to execute.
        graph:
            The pipeline graph (for node lookups).
        validate:
            If *True*, run validation after execution.

        Returns
        -------
        ExecutionResult
        """
        start = time.monotonic()
        action_results: list[ActionResult] = []
        errors: list[ExecutionErrorInfo] = []
        total_rows = 0
        all_success = True

        for nid in plan.execution_order:
            action = plan.get_action(nid)
            if action is None or action.is_noop:
                continue

            result = self.execute_action(action, graph)
            action_results.append(result)
            total_rows += result.rows_affected

            if not result.success:
                all_success = False
                if result.error is not None:
                    errors.append(result.error)
                if not (result.error and result.error.recoverable):
                    logger.error(
                        "Fatal error executing action %s at node %s: %s",
                        action.action_id, action.node_id,
                        result.error.message if result.error else "unknown",
                    )
                    break

        elapsed = time.monotonic() - start

        validation: ValidationResult | None = None
        if validate and all_success:
            from arc.execution.validation import RepairValidator
            validator = RepairValidator(self)
            validation = validator.validate_plan(plan, graph)

        return ExecutionResult(
            success=all_success,
            actions_executed=tuple(action_results),
            total_time_seconds=elapsed,
            rows_processed=total_rows,
            errors=tuple(errors),
            validation_result=validation,
            metadata={"database": self._db_path},
        )

    def execute_action(
        self,
        action: RepairAction,
        graph: PipelineGraph | None = None,
    ) -> ActionResult:
        """Execute a single repair action.

        Parameters
        ----------
        action:
            The repair action to execute.
        graph:
            Optional pipeline graph for context.

        Returns
        -------
        ActionResult
        """
        start = time.monotonic()

        try:
            if action.action_type == ActionType.RECOMPUTE:
                rows = self._execute_recompute(action, graph)
            elif action.action_type == ActionType.INCREMENTAL_UPDATE:
                rows = self._execute_incremental(action, graph)
            elif action.action_type == ActionType.SCHEMA_MIGRATE:
                rows = self._execute_schema_migrate(action, graph)
            elif action.action_type == ActionType.CHECKPOINT:
                rows = 0
            elif action.action_type == ActionType.VALIDATE:
                rows = 0
            else:
                rows = 0

            elapsed = time.monotonic() - start
            return ActionResult(
                action_id=action.action_id,
                node_id=action.node_id,
                success=True,
                elapsed_seconds=elapsed,
                rows_affected=rows,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            err = ExecutionErrorInfo(
                node_id=action.node_id,
                action_id=action.action_id,
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=isinstance(exc, (duckdb.CatalogException, duckdb.BinderException)),
            )
            logger.error("Action %s failed: %s", action.action_id, exc)
            return ActionResult(
                action_id=action.action_id,
                node_id=action.node_id,
                success=False,
                elapsed_seconds=elapsed,
                error=err,
            )

    # ── SQL execution ──────────────────────────────────────────────────

    def execute_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> duckdb.DuckDBPyRelation | None:
        """Execute a SQL statement and return the result relation.

        Parameters
        ----------
        sql:
            The SQL to execute.
        params:
            Optional named parameters for parameterised queries.

        Returns
        -------
        duckdb.DuckDBPyRelation or None
            The result, or *None* for DDL/DML without a result set.
        """
        logger.debug("SQL: %s", sql[:200])
        try:
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)
        except Exception:
            logger.exception("SQL execution failed: %s", sql[:200])
            raise

    # ── Table operations ───────────────────────────────────────────────

    def create_table(self, name: str, schema: Schema) -> None:
        """Create a table from a :class:`Schema` definition.

        Uses DuckDB-compatible DDL.
        """
        col_defs: list[str] = []
        for col in schema.columns:
            base = col.sql_type.base_type if hasattr(col.sql_type, "base_type") else None
            type_str = base.value if base is not None else "VARCHAR"
            nullable_str = "" if col.nullable else " NOT NULL"
            col_defs.append(f'"{col.name}" {type_str}{nullable_str}')

        pk_cols = schema.primary_key
        pk_clause = ""
        if pk_cols:
            pk_clause = f', PRIMARY KEY ({", ".join(pk_cols)})'

        ddl = f'CREATE TABLE IF NOT EXISTS "{name}" ({", ".join(col_defs)}{pk_clause})'
        self.execute_sql(ddl)
        logger.info("Created table %s", name)

    def drop_table(self, name: str, if_exists: bool = True) -> None:
        """Drop a table."""
        clause = "IF EXISTS " if if_exists else ""
        self.execute_sql(f'DROP TABLE {clause}"{name}"')

    def table_exists(self, name: str) -> bool:
        """Check whether a table exists."""
        result = self.execute_sql(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            {"1": name},
        )
        if result is not None:
            row = result.fetchone()
            return row is not None and row[0] > 0
        return False

    def apply_schema_delta(self, table: str, delta: SchemaDelta) -> None:
        """Apply a schema delta to an existing table.

        Translates each :class:`SchemaOperation` to an ``ALTER TABLE``
        statement.  Supports both algebra-style class hierarchy (AddColumn,
        DropColumn, ...) and enum-style (op_type == SchemaOpType.*).
        """
        from arc.algebra.schema_delta import (
            AddColumn, DropColumn, RenameColumn, ChangeType,
            AddConstraint, DropConstraint,
        )

        for op in delta.operations:
            # Class-hierarchy dispatch (algebra module)
            if isinstance(op, AddColumn):
                type_str = op.sql_type.value if hasattr(op.sql_type, 'value') else str(op.sql_type)
                sql = f'ALTER TABLE "{table}" ADD COLUMN "{op.name}" {type_str}'
                self.execute_sql(sql)

            elif isinstance(op, DropColumn):
                sql = f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS "{op.name}"'
                self.execute_sql(sql)

            elif isinstance(op, RenameColumn):
                sql = f'ALTER TABLE "{table}" RENAME COLUMN "{op.old_name}" TO "{op.new_name}"'
                self.execute_sql(sql)

            elif isinstance(op, ChangeType):
                new_type = op.new_type.value if hasattr(op.new_type, 'value') else str(op.new_type)
                sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" SET DATA TYPE {new_type}'
                self.execute_sql(sql)

            # Enum-based dispatch (execution module internal types)
            elif hasattr(op, 'op_type'):
                self._apply_enum_schema_op(table, op)

            else:
                logger.warning("Unknown schema operation type: %s", type(op).__name__)

        logger.info("Applied %d schema operations to %s", len(delta.operations), table)

    def _apply_enum_schema_op(self, table: str, op: Any) -> None:
        """Apply schema operation using enum-based dispatch."""
        if op.op_type == SchemaOpType.ADD_COLUMN:
            type_str = op.dtype.value if op.dtype is not None else "VARCHAR"
            sql = f'ALTER TABLE "{table}" ADD COLUMN "{op.column_name}" {type_str}'
            self.execute_sql(sql)
        elif op.op_type == SchemaOpType.DROP_COLUMN:
            sql = f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS "{op.column_name}"'
            self.execute_sql(sql)
        elif op.op_type == SchemaOpType.RENAME_COLUMN:
            sql = f'ALTER TABLE "{table}" RENAME COLUMN "{op.column_name}" TO "{op.new_column_name}"'
            self.execute_sql(sql)
        elif op.op_type == SchemaOpType.RETYPE_COLUMN:
            new_type = op.new_dtype.value if op.new_dtype is not None else "VARCHAR"
            sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" SET DATA TYPE {new_type}'
            self.execute_sql(sql)
        elif op.op_type == SchemaOpType.SET_NULLABLE:
            if op.nullable:
                sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" DROP NOT NULL'
            else:
                sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" SET NOT NULL'
            self.execute_sql(sql)
        elif op.op_type == SchemaOpType.SET_DEFAULT:
            if op.default_val is not None:
                sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" SET DEFAULT {op.default_val!r}'
            else:
                sql = f'ALTER TABLE "{table}" ALTER COLUMN "{op.column_name}" DROP DEFAULT'
            self.execute_sql(sql)

    def apply_data_delta(self, table: str, delta: DataDelta) -> int:
        """Apply a data delta to an existing table.

        Returns the number of rows affected.  Supports both algebra-style
        DataDelta (InsertOp/DeleteOp/UpdateOp) and internal RowChange format.
        """
        from arc.algebra.data_delta import InsertOp, DeleteOp, UpdateOp

        rows = 0

        # Algebra-style DataDelta (has .operations)
        if hasattr(delta, 'operations') and not hasattr(delta, 'changes'):
            for op in delta.operations:
                if isinstance(op, InsertOp):
                    for tup in op.tuples:
                        fields = tup.to_dict() if hasattr(tup, 'to_dict') else tup.values if hasattr(tup, 'values') and isinstance(tup.values, dict) else dict(tup.fields) if hasattr(tup, 'fields') else {}
                        if fields:
                            cols = list(fields.keys())
                            vals = list(fields.values())
                            placeholders = ", ".join("?" for _ in vals)
                            col_list = ", ".join(f'"{c}"' for c in cols)
                            sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
                            self._conn.execute(sql, vals)
                            rows += 1
                elif isinstance(op, DeleteOp):
                    for tup in op.tuples:
                        fields = tup.to_dict() if hasattr(tup, 'to_dict') else tup.values if hasattr(tup, 'values') and isinstance(tup.values, dict) else dict(tup.fields) if hasattr(tup, 'fields') else {}
                        if fields:
                            conditions = " AND ".join(f'"{k}" = ?' for k in fields.keys())
                            sql = f'DELETE FROM "{table}" WHERE {conditions}'
                            self._conn.execute(sql, list(fields.values()))
                            rows += 1
                elif isinstance(op, UpdateOp):
                    for old_tup, new_tup in zip(op.old_tuples, op.new_tuples):
                        old_f = old_tup.to_dict() if hasattr(old_tup, 'to_dict') else old_tup.values if hasattr(old_tup, 'values') and isinstance(old_tup.values, dict) else dict(old_tup.fields) if hasattr(old_tup, 'fields') else {}
                        new_f = new_tup.to_dict() if hasattr(new_tup, 'to_dict') else new_tup.values if hasattr(new_tup, 'values') and isinstance(new_tup.values, dict) else dict(new_tup.fields) if hasattr(new_tup, 'fields') else {}
                        if old_f and new_f:
                            set_clause = ", ".join(f'"{k}" = ?' for k in new_f.keys())
                            where_clause = " AND ".join(f'"{k}" = ?' for k in old_f.keys())
                            sql = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
                            params = list(new_f.values()) + list(old_f.values())
                            self._conn.execute(sql, params)
                            rows += 1
            logger.info("Applied %d algebra data operations to %s (%d rows)", len(delta.operations), table, rows)
            return rows

        # Enum-based RowChange format
        for change in delta.changes:
            if change.change_type == RowChangeType.INSERT:
                cols = list(change.new_values.keys())
                vals = list(change.new_values.values())
                placeholders = ", ".join("?" for _ in vals)
                col_list = ", ".join(f'"{c}"' for c in cols)
                sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
                self._conn.execute(sql, vals)
                rows += 1

            elif change.change_type == RowChangeType.DELETE:
                if change.old_values:
                    conditions = " AND ".join(
                        f'"{k}" = ?' for k in change.old_values.keys()
                    )
                    sql = f'DELETE FROM "{table}" WHERE {conditions}'
                    self._conn.execute(sql, list(change.old_values.values()))
                    rows += 1

            elif change.change_type == RowChangeType.UPDATE:
                if change.new_values and change.old_values:
                    set_clause = ", ".join(
                        f'"{k}" = ?' for k in change.new_values.keys()
                    )
                    where_clause = " AND ".join(
                        f'"{k}" = ?' for k in change.old_values.keys()
                    )
                    sql = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
                    params = list(change.new_values.values()) + list(change.old_values.values())
                    self._conn.execute(sql, params)
                    rows += 1

        logger.info("Applied %d data changes to %s (%d rows)", len(delta.changes), table, rows)
        return rows

    def get_table_schema(self, table: str) -> list[dict[str, Any]]:
        """Return column info for a table as a list of dicts."""
        result = self.execute_sql(
            f"SELECT column_name, data_type, is_nullable "
            f"FROM information_schema.columns "
            f"WHERE table_name = ? "
            f"ORDER BY ordinal_position",
            {"1": table},
        )
        cols: list[dict[str, Any]] = []
        if result is not None:
            for row in result.fetchall():
                cols.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                })
        return cols

    def get_table_stats(self, table: str) -> TableStats:
        """Compute basic statistics for a table."""
        count_result = self.execute_sql(f'SELECT COUNT(*) FROM "{table}"')
        row_count = 0
        if count_result is not None:
            row = count_result.fetchone()
            if row is not None:
                row_count = row[0]

        cols = self.get_table_schema(table)
        null_counts: dict[str, int] = {}
        distinct_counts: dict[str, int] = {}
        min_values: dict[str, Any] = {}
        max_values: dict[str, Any] = {}

        for col_info in cols:
            cname = col_info["name"]
            try:
                stats = self.execute_sql(
                    f'SELECT COUNT(*) - COUNT("{cname}"), '
                    f'COUNT(DISTINCT "{cname}"), '
                    f'MIN("{cname}"), MAX("{cname}") '
                    f'FROM "{table}"'
                )
                if stats is not None:
                    r = stats.fetchone()
                    if r is not None:
                        null_counts[cname] = r[0] if r[0] is not None else 0
                        distinct_counts[cname] = r[1] if r[1] is not None else 0
                        min_values[cname] = r[2]
                        max_values[cname] = r[3]
            except Exception:
                pass

        return TableStats(
            table_name=table,
            row_count=row_count,
            column_count=len(cols),
            null_counts=null_counts,
            distinct_counts=distinct_counts,
            min_values=min_values,
            max_values=max_values,
        )

    def register_source(self, name: str, data: Any) -> None:
        """Register an external data source (e.g. pandas DataFrame).

        The data can then be queried as a table with the given *name*.
        """
        self._conn.register(name, data)
        self._registered_sources[name] = data
        logger.info("Registered external source '%s'", name)

    def unregister_source(self, name: str) -> None:
        """Unregister a previously registered source."""
        try:
            self._conn.unregister(name)
        except Exception:
            pass
        self._registered_sources.pop(name, None)

    # ── Private execution methods ──────────────────────────────────────

    def _execute_recompute(
        self,
        action: RepairAction,
        graph: PipelineGraph | None,
    ) -> int:
        """Execute a RECOMPUTE action.

        If the action has SQL text, execute it directly.
        If the node has SQL text, execute that.
        Otherwise, create a simple SELECT * into a temp table.
        """
        node = graph.nodes.get(action.node_id) if graph else None
        sql = action.sql_text or (node.sql_text if node else "")

        if sql:
            result = self.execute_sql(sql)
            if result is not None:
                try:
                    return len(result.fetchall())
                except Exception:
                    return 0
            return 0

        # Fallback: if node has a table name, do a refresh via CTAS
        if node and node.table_name:
            parents = graph.parents(action.node_id) if graph else []
            if parents:
                parent_node = graph.nodes.get(parents[0]) if graph else None
                if parent_node and parent_node.table_name:
                    temp_name = f"_arc_tmp_{node.table_name}"
                    self.execute_sql(
                        f'CREATE OR REPLACE TABLE "{temp_name}" AS '
                        f'SELECT * FROM "{parent_node.table_name}"'
                    )
                    self.execute_sql(f'DROP TABLE IF EXISTS "{node.table_name}"')
                    self.execute_sql(
                        f'ALTER TABLE "{temp_name}" RENAME TO "{node.table_name}"'
                    )
                    count = self.execute_sql(f'SELECT COUNT(*) FROM "{node.table_name}"')
                    if count is not None:
                        r = count.fetchone()
                        return r[0] if r else 0

        logger.debug("RECOMPUTE %s: no SQL to execute", action.node_id)
        return 0

    def _execute_incremental(
        self,
        action: RepairAction,
        graph: PipelineGraph | None,
    ) -> int:
        """Execute an INCREMENTAL_UPDATE action by applying the data delta."""
        if action.delta_to_apply is None:
            return 0

        node = graph.nodes.get(action.node_id) if graph else None
        table_name = node.table_name if node else action.node_id

        rows = 0
        if action.delta_to_apply.has_schema_change:
            self.apply_schema_delta(table_name, action.delta_to_apply.schema_delta)

        if action.delta_to_apply.has_data_change:
            rows = self.apply_data_delta(table_name, action.delta_to_apply.data_delta)

        return rows

    def _execute_schema_migrate(
        self,
        action: RepairAction,
        graph: PipelineGraph | None,
    ) -> int:
        """Execute a SCHEMA_MIGRATE action."""
        if action.delta_to_apply is None or not action.delta_to_apply.has_schema_change:
            return 0

        node = graph.nodes.get(action.node_id) if graph else None
        table_name = node.table_name if node else action.node_id
        self.apply_schema_delta(table_name, action.delta_to_apply.schema_delta)
        return 0

    def __repr__(self) -> str:
        return f"ExecutionEngine(db={self._db_path!r})"
