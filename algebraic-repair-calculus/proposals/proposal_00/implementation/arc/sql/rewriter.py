"""
SQL Query Rewriting Engine
===========================

Rewrites SQL queries to incorporate schema deltas, add quality filters,
optimize for incremental execution, generate diff queries, and more.
Uses sqlglot for AST-level manipulation when available, with fallback
to regex-based rewriting.

Key operations:
  - Apply schema deltas to queries (add/drop/rename columns, change types)
  - Add quality constraint filters
  - Generate incremental diff queries
  - Generate merge/upsert queries
  - Generate validation queries
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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

try:
    import sqlglot
    import sqlglot.expressions as exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =====================================================================
# Rewrite Result
# =====================================================================


@dataclass
class RewriteResult:
    """Result of a SQL rewrite operation.

    Attributes
    ----------
    original_sql : str
        The original SQL query.
    rewritten_sql : str
        The rewritten SQL query.
    changes_made : list[str]
        Description of changes applied.
    warnings : list[str]
        Warnings encountered during rewriting.
    used_ast : bool
        True if AST-level rewriting was used (vs. regex fallback).
    """
    original_sql: str = ""
    rewritten_sql: str = ""
    changes_made: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    used_ast: bool = False

    @property
    def was_modified(self) -> bool:
        return self.original_sql != self.rewritten_sql


class RewriteDialect(Enum):
    """SQL dialect for rewriting."""
    POSTGRES = "postgres"
    DUCKDB = "duckdb"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    GENERIC = "generic"


# =====================================================================
# Quality Constraint
# =====================================================================


@dataclass
class QualityConstraint:
    """A quality constraint to apply as a SQL filter.

    Attributes
    ----------
    column : str
        Column to constrain.
    constraint_type : str
        Type of constraint (not_null, range, regex, etc.).
    parameters : dict
        Constraint-specific parameters.
    """
    column: str = ""
    constraint_type: str = "not_null"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_sql_predicate(self) -> str:
        """Convert constraint to SQL WHERE predicate."""
        if self.constraint_type == "not_null":
            return f"{self.column} IS NOT NULL"
        if self.constraint_type == "range":
            min_val = self.parameters.get("min")
            max_val = self.parameters.get("max")
            parts = []
            if min_val is not None:
                parts.append(f"{self.column} >= {min_val}")
            if max_val is not None:
                parts.append(f"{self.column} <= {max_val}")
            return " AND ".join(parts) if parts else "TRUE"
        if self.constraint_type == "regex":
            pattern = self.parameters.get("pattern", ".*")
            return f"{self.column} ~ '{pattern}'"
        if self.constraint_type == "enum":
            values = self.parameters.get("values", [])
            vals_str = ", ".join(f"'{v}'" for v in values)
            return f"{self.column} IN ({vals_str})"
        if self.constraint_type == "unique":
            return f"{self.column} IS NOT NULL"
        if self.constraint_type == "custom_sql":
            return self.parameters.get("expression", "TRUE")
        return "TRUE"


# =====================================================================
# Schema Delta Types (local, lightweight)
# =====================================================================


@dataclass
class SchemaDeltaSpec:
    """Lightweight representation of a schema delta for rewriting.

    Attributes
    ----------
    added_columns : list[tuple[str, str, str | None]]
        List of (name, type, default_expr) for added columns.
    dropped_columns : list[str]
        Columns to remove.
    renamed_columns : list[tuple[str, str]]
        List of (old_name, new_name) renames.
    type_changes : list[tuple[str, str]]
        List of (column, new_type) type changes.
    """
    added_columns: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    renamed_columns: List[Tuple[str, str]] = field(default_factory=list)
    type_changes: List[Tuple[str, str]] = field(default_factory=list)


# =====================================================================
# SQL Rewriter
# =====================================================================


class SQLRewriter:
    """Rewrite SQL queries to incorporate schema deltas and quality filters.

    Supports both AST-level rewriting (via sqlglot) and regex-based
    fallback for environments where sqlglot is unavailable.

    Parameters
    ----------
    dialect : RewriteDialect
        SQL dialect for parsing and generation.
    prefer_ast : bool
        Whether to prefer AST-level rewriting when available.
    """

    def __init__(
        self,
        dialect: RewriteDialect = RewriteDialect.DUCKDB,
        prefer_ast: bool = True,
    ) -> None:
        self._dialect = dialect
        self._prefer_ast = prefer_ast and HAS_SQLGLOT
        self._dialect_name = self._dialect.value if self._dialect != RewriteDialect.GENERIC else None

    # ── Schema Delta Application ──────────────────────────────────

    def apply_schema_delta_to_query(
        self,
        query: str,
        delta: SchemaDeltaSpec,
    ) -> RewriteResult:
        """Apply a schema delta to a SQL query.

        Handles column additions, removals, renames, and type changes.

        Parameters
        ----------
        query : str
            The original SQL query.
        delta : SchemaDeltaSpec
            The schema changes to apply.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query, rewritten_sql=query)
        current = query

        for old_name, new_name in delta.renamed_columns:
            r = self.rename_column_in_query(current, old_name, new_name)
            current = r.rewritten_sql
            result.changes_made.extend(r.changes_made)
            result.warnings.extend(r.warnings)

        for col in delta.dropped_columns:
            r = self.remove_column_references(current, col)
            current = r.rewritten_sql
            result.changes_made.extend(r.changes_made)
            result.warnings.extend(r.warnings)

        for col_name, col_type, default_expr in delta.added_columns:
            default = default_expr or self._default_for_type(col_type)
            r = self.add_column_defaults(current, col_name, default)
            current = r.rewritten_sql
            result.changes_made.extend(r.changes_made)
            result.warnings.extend(r.warnings)

        for col_name, new_type in delta.type_changes:
            r = self.change_type_casts(current, col_name, new_type)
            current = r.rewritten_sql
            result.changes_made.extend(r.changes_made)
            result.warnings.extend(r.warnings)

        result.rewritten_sql = current
        result.used_ast = self._prefer_ast
        return result

    def add_column_defaults(
        self,
        query: str,
        column: str,
        default_expr: str,
    ) -> RewriteResult:
        """Add a default expression for a new column in the query output.

        Modifies SELECT clauses to include the new column with its default.

        Parameters
        ----------
        query : str
            The SQL query.
        column : str
            New column name.
        default_expr : str
            Default expression (e.g., "0", "NULL", "'unknown'").

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)

        if self._prefer_ast:
            try:
                rewritten = self._ast_add_column_default(query, column, default_expr)
                result.rewritten_sql = rewritten
                result.changes_made.append(
                    f"Added column {column} with default {default_expr}"
                )
                result.used_ast = True
                return result
            except Exception as exc:
                result.warnings.append(f"AST rewrite failed: {exc}")

        rewritten = self._regex_add_column_default(query, column, default_expr)
        result.rewritten_sql = rewritten
        result.changes_made.append(
            f"Added column {column} with default {default_expr}"
        )
        return result

    def remove_column_references(
        self,
        query: str,
        column: str,
    ) -> RewriteResult:
        """Remove all references to a dropped column from the query.

        Parameters
        ----------
        query : str
            The SQL query.
        column : str
            Column to remove.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)

        if self._prefer_ast:
            try:
                rewritten = self._ast_remove_column(query, column)
                result.rewritten_sql = rewritten
                result.changes_made.append(f"Removed column {column}")
                result.used_ast = True
                return result
            except Exception as exc:
                result.warnings.append(f"AST rewrite failed: {exc}")

        rewritten = self._regex_remove_column(query, column)
        result.rewritten_sql = rewritten
        result.changes_made.append(f"Removed column {column}")
        return result

    def rename_column_in_query(
        self,
        query: str,
        old_name: str,
        new_name: str,
    ) -> RewriteResult:
        """Rename all occurrences of a column in a query.

        Parameters
        ----------
        query : str
            The SQL query.
        old_name : str
            Current column name.
        new_name : str
            New column name.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)

        if self._prefer_ast:
            try:
                rewritten = self._ast_rename_column(query, old_name, new_name)
                result.rewritten_sql = rewritten
                result.changes_made.append(f"Renamed {old_name} -> {new_name}")
                result.used_ast = True
                return result
            except Exception as exc:
                result.warnings.append(f"AST rewrite failed: {exc}")

        rewritten = self._regex_rename_column(query, old_name, new_name)
        result.rewritten_sql = rewritten
        result.changes_made.append(f"Renamed {old_name} -> {new_name}")
        return result

    def change_type_casts(
        self,
        query: str,
        column: str,
        new_type: str,
    ) -> RewriteResult:
        """Wrap column references with CAST to new type.

        Parameters
        ----------
        query : str
            The SQL query.
        column : str
            Column whose type is changing.
        new_type : str
            New SQL type.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)

        if self._prefer_ast:
            try:
                rewritten = self._ast_add_cast(query, column, new_type)
                result.rewritten_sql = rewritten
                result.changes_made.append(f"Added CAST for {column} to {new_type}")
                result.used_ast = True
                return result
            except Exception as exc:
                result.warnings.append(f"AST rewrite failed: {exc}")

        rewritten = self._regex_add_cast(query, column, new_type)
        result.rewritten_sql = rewritten
        result.changes_made.append(f"Added CAST for {column} to {new_type}")
        return result

    def add_quality_filter(
        self,
        query: str,
        constraint: QualityConstraint,
    ) -> RewriteResult:
        """Add a quality constraint as a WHERE filter.

        Parameters
        ----------
        query : str
            The SQL query.
        constraint : QualityConstraint
            The quality constraint.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)
        predicate = constraint.to_sql_predicate()

        if self._prefer_ast:
            try:
                rewritten = self._ast_add_where(query, predicate)
                result.rewritten_sql = rewritten
                result.changes_made.append(f"Added quality filter: {predicate}")
                result.used_ast = True
                return result
            except Exception as exc:
                result.warnings.append(f"AST rewrite failed: {exc}")

        rewritten = self._regex_add_where(query, predicate)
        result.rewritten_sql = rewritten
        result.changes_made.append(f"Added quality filter: {predicate}")
        return result

    def optimize_incremental_query(
        self,
        original: str,
        delta_query: str,
    ) -> RewriteResult:
        """Optimize a query for incremental execution.

        Creates a UNION of the original query restricted to unchanged rows
        and the delta query for changed rows.

        Parameters
        ----------
        original : str
            The original full query.
        delta_query : str
            Query that selects only changed rows.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=original)

        incremental = (
            f"-- Incremental execution\n"
            f"WITH _delta AS (\n"
            f"  {delta_query}\n"
            f"),\n"
            f"_original AS (\n"
            f"  {original}\n"
            f")\n"
            f"SELECT * FROM _original\n"
            f"UNION ALL\n"
            f"SELECT * FROM _delta"
        )

        result.rewritten_sql = incremental
        result.changes_made.append("Converted to incremental execution with UNION ALL")
        return result

    def generate_diff_query(
        self,
        table: str,
        old_snapshot: str,
        new_snapshot: str,
    ) -> RewriteResult:
        """Generate a query that computes the diff between two snapshots.

        Returns rows that are in one snapshot but not the other, tagged
        with their change type (INSERT, DELETE, UPDATE).

        Parameters
        ----------
        table : str
            Table name for context.
        old_snapshot : str
            Table/view name or subquery for old data.
        new_snapshot : str
            Table/view name or subquery for new data.

        Returns
        -------
        RewriteResult
        """
        diff_sql = (
            f"-- Diff between {old_snapshot} and {new_snapshot}\n"
            f"WITH _old AS (\n"
            f"  SELECT *, 'old' AS _snapshot_source FROM {old_snapshot}\n"
            f"),\n"
            f"_new AS (\n"
            f"  SELECT *, 'new' AS _snapshot_source FROM {new_snapshot}\n"
            f"),\n"
            f"_inserts AS (\n"
            f"  SELECT n.*, 'INSERT' AS _change_type\n"
            f"  FROM _new n\n"
            f"  LEFT JOIN _old o ON n.* IS NOT DISTINCT FROM o.*\n"
            f"  WHERE o.* IS NULL\n"
            f"),\n"
            f"_deletes AS (\n"
            f"  SELECT o.*, 'DELETE' AS _change_type\n"
            f"  FROM _old o\n"
            f"  LEFT JOIN _new n ON o.* IS NOT DISTINCT FROM n.*\n"
            f"  WHERE n.* IS NULL\n"
            f")\n"
            f"SELECT * FROM _inserts\n"
            f"UNION ALL\n"
            f"SELECT * FROM _deletes"
        )

        return RewriteResult(
            original_sql="",
            rewritten_sql=diff_sql,
            changes_made=[f"Generated diff query for {table}"],
        )

    def generate_merge_query(
        self,
        target: str,
        source: str,
        key_columns: List[str],
    ) -> RewriteResult:
        """Generate a MERGE (upsert) query.

        Parameters
        ----------
        target : str
            Target table name.
        source : str
            Source table/view name.
        key_columns : list[str]
            Columns to use for matching.

        Returns
        -------
        RewriteResult
        """
        join_cond = " AND ".join(
            f"target.{k} = source.{k}" for k in key_columns
        )

        merge_sql = (
            f"-- Merge {source} into {target}\n"
            f"MERGE INTO {target} AS target\n"
            f"USING {source} AS source\n"
            f"ON {join_cond}\n"
            f"WHEN MATCHED THEN\n"
            f"  UPDATE SET *\n"
            f"WHEN NOT MATCHED THEN\n"
            f"  INSERT *"
        )

        return RewriteResult(
            original_sql="",
            rewritten_sql=merge_sql,
            changes_made=[f"Generated MERGE query for {target} from {source}"],
        )

    def generate_validation_query(
        self,
        table: str,
        constraints: List[QualityConstraint],
    ) -> RewriteResult:
        """Generate a validation query that checks constraints.

        Returns rows that violate any of the specified constraints.

        Parameters
        ----------
        table : str
            Table to validate.
        constraints : list[QualityConstraint]
            Constraints to check.

        Returns
        -------
        RewriteResult
        """
        if not constraints:
            return RewriteResult(
                original_sql="",
                rewritten_sql=f"SELECT * FROM {table} WHERE FALSE",
                changes_made=["Generated empty validation query"],
            )

        violation_parts = []
        for i, c in enumerate(constraints):
            predicate = c.to_sql_predicate()
            negated = f"NOT ({predicate})"
            violation_parts.append(
                f"  SELECT *, '{c.constraint_type}' AS _violation_type, "
                f"'{c.column}' AS _violation_column, "
                f"{i} AS _constraint_id\n"
                f"  FROM {table}\n"
                f"  WHERE {negated}"
            )

        validation_sql = (
            f"-- Validation query for {table}\n"
            + "\nUNION ALL\n".join(violation_parts)
        )

        return RewriteResult(
            original_sql="",
            rewritten_sql=validation_sql,
            changes_made=[
                f"Generated validation query with {len(constraints)} constraints"
            ],
        )

    def generate_schema_migration_sql(
        self,
        table: str,
        delta: SchemaDeltaSpec,
    ) -> RewriteResult:
        """Generate DDL statements for schema migration.

        Parameters
        ----------
        table : str
            Table to migrate.
        delta : SchemaDeltaSpec
            Schema changes.

        Returns
        -------
        RewriteResult
        """
        statements: List[str] = []

        for col_name, col_type, default_expr in delta.added_columns:
            stmt = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
            if default_expr:
                stmt += f" DEFAULT {default_expr}"
            statements.append(stmt + ";")

        for col in delta.dropped_columns:
            statements.append(f"ALTER TABLE {table} DROP COLUMN {col};")

        for old_name, new_name in delta.renamed_columns:
            statements.append(
                f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name};"
            )

        for col_name, new_type in delta.type_changes:
            statements.append(
                f"ALTER TABLE {table} ALTER COLUMN {col_name} TYPE {new_type};"
            )

        migration_sql = "\n".join(statements)

        return RewriteResult(
            original_sql="",
            rewritten_sql=migration_sql,
            changes_made=[f"Generated {len(statements)} DDL statements for {table}"],
        )

    def generate_incremental_refresh_query(
        self,
        view_query: str,
        delta_table: str,
        key_columns: List[str],
    ) -> RewriteResult:
        """Generate query for incremental refresh of a materialized view.

        Parameters
        ----------
        view_query : str
            The original view definition query.
        delta_table : str
            Table containing changed rows.
        key_columns : list[str]
            Key columns for matching.

        Returns
        -------
        RewriteResult
        """
        key_join = " AND ".join(
            f"v.{k} = d.{k}" for k in key_columns
        )

        refresh_sql = (
            f"-- Incremental refresh\n"
            f"DELETE FROM _materialized_view v\n"
            f"USING {delta_table} d\n"
            f"WHERE {key_join};\n\n"
            f"INSERT INTO _materialized_view\n"
            f"SELECT * FROM ({view_query}) sub\n"
            f"WHERE EXISTS (\n"
            f"  SELECT 1 FROM {delta_table} d\n"
            f"  WHERE {' AND '.join(f'sub.{k} = d.{k}' for k in key_columns)}\n"
            f");"
        )

        return RewriteResult(
            original_sql=view_query,
            rewritten_sql=refresh_sql,
            changes_made=["Generated incremental refresh query"],
        )

    def wrap_with_error_handling(
        self,
        query: str,
        fallback_value: str = "NULL",
    ) -> RewriteResult:
        """Wrap column expressions with TRY_CAST for error handling.

        Parameters
        ----------
        query : str
            The SQL query.
        fallback_value : str
            Value to use on cast failure.

        Returns
        -------
        RewriteResult
        """
        result = RewriteResult(original_sql=query)

        cast_pattern = re.compile(
            r"CAST\((.+?)\s+AS\s+(\w+)\)", re.IGNORECASE
        )

        def replace_cast(match: re.Match) -> str:
            expr = match.group(1)
            type_name = match.group(2)
            return f"TRY_CAST({expr} AS {type_name})"

        rewritten = cast_pattern.sub(replace_cast, query)

        if rewritten != query:
            result.rewritten_sql = rewritten
            result.changes_made.append("Replaced CAST with TRY_CAST for error handling")
        else:
            result.rewritten_sql = query

        return result

    # ── AST-based Rewriting (sqlglot) ─────────────────────────────

    def _ast_add_column_default(
        self,
        query: str,
        column: str,
        default_expr: str,
    ) -> str:
        """Add column default using sqlglot AST."""
        tree = sqlglot.parse_one(query, dialect=self._dialect_name)

        select = tree.find(exp.Select)
        if select is None:
            return query

        alias_expr = sqlglot.parse_one(
            f"{default_expr} AS {column}",
            dialect=self._dialect_name,
        )
        select.args["expressions"].append(alias_expr)

        return tree.sql(dialect=self._dialect_name)

    def _ast_remove_column(self, query: str, column: str) -> str:
        """Remove column references using sqlglot AST."""
        tree = sqlglot.parse_one(query, dialect=self._dialect_name)

        for col_ref in tree.find_all(exp.Column):
            if col_ref.name == column:
                col_ref.pop()

        select = tree.find(exp.Select)
        if select:
            new_exprs = []
            for e in select.args.get("expressions", []):
                col_names = [c.name for c in e.find_all(exp.Column)]
                alias_name = getattr(e, "alias", "")
                if column not in col_names and alias_name != column:
                    new_exprs.append(e)
            if new_exprs:
                select.args["expressions"] = new_exprs

        return tree.sql(dialect=self._dialect_name)

    def _ast_rename_column(
        self,
        query: str,
        old_name: str,
        new_name: str,
    ) -> str:
        """Rename column using sqlglot AST."""
        tree = sqlglot.parse_one(query, dialect=self._dialect_name)

        for col_ref in tree.find_all(exp.Column):
            if col_ref.name == old_name:
                col_ref.set("this", exp.to_identifier(new_name))

        return tree.sql(dialect=self._dialect_name)

    def _ast_add_cast(self, query: str, column: str, new_type: str) -> str:
        """Add CAST using sqlglot AST."""
        tree = sqlglot.parse_one(query, dialect=self._dialect_name)

        for col_ref in tree.find_all(exp.Column):
            if col_ref.name == column:
                parent = col_ref.parent
                if not isinstance(parent, exp.Cast):
                    cast_expr = exp.Cast(
                        this=col_ref.copy(),
                        to=exp.DataType.build(new_type),
                    )
                    col_ref.replace(cast_expr)

        return tree.sql(dialect=self._dialect_name)

    def _ast_add_where(self, query: str, predicate: str) -> str:
        """Add WHERE clause using sqlglot AST."""
        tree = sqlglot.parse_one(query, dialect=self._dialect_name)

        pred_expr = sqlglot.parse_one(predicate, dialect=self._dialect_name)

        existing_where = tree.find(exp.Where)
        if existing_where:
            combined = exp.And(
                this=existing_where.this,
                expression=pred_expr,
            )
            existing_where.set("this", combined)
        else:
            where = exp.Where(this=pred_expr)
            select = tree.find(exp.Select)
            if select:
                tree.set("where", where)

        return tree.sql(dialect=self._dialect_name)

    # ── Regex-based Fallback Rewriting ────────────────────────────

    def _regex_add_column_default(
        self,
        query: str,
        column: str,
        default_expr: str,
    ) -> str:
        """Add column default using regex."""
        select_match = re.search(
            r"(\bSELECT\b\s+)(.*?)(\s*\bFROM\b)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if select_match:
            before = select_match.group(1)
            cols = select_match.group(2)
            after = select_match.group(3)
            new_cols = f"{cols}, {default_expr} AS {column}"
            return query[:select_match.start()] + before + new_cols + after + query[select_match.end():]

        return query

    def _regex_remove_column(self, query: str, column: str) -> str:
        """Remove column references using regex."""
        pattern = re.compile(
            r",?\s*\b" + re.escape(column) + r"\b\s*(?:,)?",
            re.IGNORECASE,
        )
        rewritten = pattern.sub("", query)

        rewritten = re.sub(r",\s*,", ",", rewritten)
        rewritten = re.sub(r",\s*\bFROM\b", " FROM", rewritten, flags=re.IGNORECASE)

        return rewritten.strip()

    def _regex_rename_column(
        self,
        query: str,
        old_name: str,
        new_name: str,
    ) -> str:
        """Rename column using regex."""
        pattern = re.compile(
            r"\b" + re.escape(old_name) + r"\b",
            re.IGNORECASE,
        )
        return pattern.sub(new_name, query)

    def _regex_add_cast(self, query: str, column: str, new_type: str) -> str:
        """Add CAST using regex."""
        pattern = re.compile(
            r"(?<!\bCAST\(\s*)\b" + re.escape(column) + r"\b",
            re.IGNORECASE,
        )
        return pattern.sub(f"CAST({column} AS {new_type})", query, count=0)

    def _regex_add_where(self, query: str, predicate: str) -> str:
        """Add WHERE clause using regex."""
        where_match = re.search(r"\bWHERE\b", query, re.IGNORECASE)

        if where_match:
            insert_pos = where_match.end()
            return (
                query[:insert_pos]
                + f" ({predicate}) AND"
                + query[insert_pos:]
            )

        from_match = re.search(
            r"(\bFROM\b\s+\S+(?:\s+\w+)?)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if from_match:
            insert_pos = from_match.end()
            return (
                query[:insert_pos]
                + f"\nWHERE {predicate}"
                + query[insert_pos:]
            )

        return query + f"\nWHERE {predicate}"

    # ── Helpers ───────────────────────────────────────────────────

    def _default_for_type(self, sql_type: str) -> str:
        """Get a default expression for a SQL type."""
        t = sql_type.upper().strip()

        int_types = {"INT", "INTEGER", "BIGINT", "SMALLINT", "SERIAL", "BIGSERIAL"}
        float_types = {"FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC"}
        string_types = {"VARCHAR", "TEXT", "CHAR", "CHARACTER VARYING"}
        bool_types = {"BOOLEAN", "BOOL"}

        if t in int_types or t.startswith(tuple(int_types)):
            return "0"
        if t in float_types or t.startswith(tuple(float_types)):
            return "0.0"
        if t in string_types or t.startswith(tuple(string_types)):
            return "''"
        if t in bool_types:
            return "FALSE"
        if t in {"DATE"}:
            return "'1970-01-01'"
        if "TIMESTAMP" in t:
            return "'1970-01-01 00:00:00'"
        if t in {"JSON", "JSONB"}:
            return "'null'"
        if t in {"UUID"}:
            return "'00000000-0000-0000-0000-000000000000'"

        return "NULL"


# =====================================================================
# Convenience Functions
# =====================================================================


def rewrite_for_schema_delta(
    query: str,
    delta: SchemaDeltaSpec,
    dialect: RewriteDialect = RewriteDialect.DUCKDB,
) -> str:
    """Convenience: apply schema delta and return rewritten SQL."""
    rewriter = SQLRewriter(dialect=dialect)
    result = rewriter.apply_schema_delta_to_query(query, delta)
    return result.rewritten_sql


def add_quality_filters(
    query: str,
    constraints: List[QualityConstraint],
    dialect: RewriteDialect = RewriteDialect.DUCKDB,
) -> str:
    """Convenience: add multiple quality filters to a query."""
    rewriter = SQLRewriter(dialect=dialect)
    current = query
    for constraint in constraints:
        result = rewriter.add_quality_filter(current, constraint)
        current = result.rewritten_sql
    return current


def generate_diff(
    table: str,
    old_snapshot: str,
    new_snapshot: str,
) -> str:
    """Convenience: generate a diff query."""
    rewriter = SQLRewriter()
    result = rewriter.generate_diff_query(table, old_snapshot, new_snapshot)
    return result.rewritten_sql


def generate_merge(
    target: str,
    source: str,
    key_columns: List[str],
) -> str:
    """Convenience: generate a merge query."""
    rewriter = SQLRewriter()
    result = rewriter.generate_merge_query(target, source, key_columns)
    return result.rewritten_sql
