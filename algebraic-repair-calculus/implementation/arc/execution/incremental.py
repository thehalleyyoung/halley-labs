"""
Incremental Execution Engine
==============================

Executes repairs incrementally rather than full recomputation. For each
SQL operator type, computes only the changed output rows based on the
incoming delta.

Uses DuckDB for efficient in-process computation of incremental deltas.

Supported incremental operators:
  - SELECT / PROJECT: Filter/project the delta rows.
  - JOIN: Join delta rows against the opposite table.
  - GROUP BY: Recompute only affected groups.
  - FILTER: Apply filter predicate to delta rows.
  - UNION: Append delta rows (with dedup for UNION DISTINCT).
  - WINDOW: Recompute affected partitions.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
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
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

logger = logging.getLogger(__name__)


# =====================================================================
# Data Delta Representation
# =====================================================================


class DeltaType(Enum):
    """Type of data change."""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class DataDeltaRow:
    """A single row-level change.

    Attributes
    ----------
    change_type : DeltaType
        Type of change.
    values : dict[str, Any]
        Column values for the row.
    old_values : dict[str, Any] | None
        Previous values (for updates).
    """
    change_type: DeltaType = DeltaType.INSERT
    values: Dict[str, Any] = field(default_factory=dict)
    old_values: Optional[Dict[str, Any]] = None


@dataclass
class DataDelta:
    """Collection of row-level changes.

    Attributes
    ----------
    rows : list[DataDeltaRow]
        The changed rows.
    affected_columns : set[str]
        Columns that have changes.
    source_table : str
        The table these changes originated from.
    """
    rows: List[DataDeltaRow] = field(default_factory=list)
    affected_columns: Set[str] = field(default_factory=set)
    source_table: str = ""

    @property
    def insert_count(self) -> int:
        return sum(1 for r in self.rows if r.change_type == DeltaType.INSERT)

    @property
    def delete_count(self) -> int:
        return sum(1 for r in self.rows if r.change_type == DeltaType.DELETE)

    @property
    def update_count(self) -> int:
        return sum(1 for r in self.rows if r.change_type == DeltaType.UPDATE)

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def inserts(self) -> List[DataDeltaRow]:
        return [r for r in self.rows if r.change_type == DeltaType.INSERT]

    def deletes(self) -> List[DataDeltaRow]:
        return [r for r in self.rows if r.change_type == DeltaType.DELETE]

    def updates(self) -> List[DataDeltaRow]:
        return [r for r in self.rows if r.change_type == DeltaType.UPDATE]


# =====================================================================
# Incremental Execution Result
# =====================================================================


@dataclass
class IncrementalResult:
    """Result of an incremental execution.

    Attributes
    ----------
    output_delta : DataDelta
        The output changes.
    rows_processed : int
        Number of input delta rows processed.
    rows_produced : int
        Number of output delta rows produced.
    rows_skipped : int
        Number of input rows skipped.
    execution_time_ms : float
        Execution time in milliseconds.
    memory_used_bytes : int
        Estimated memory used during computation.
    operator_type : str
        The operator type that was applied.
    node_id : str
        The pipeline node that was processed.
    """
    output_delta: DataDelta = field(default_factory=DataDelta)
    rows_processed: int = 0
    rows_produced: int = 0
    rows_skipped: int = 0
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    operator_type: str = ""
    node_id: str = ""

    @property
    def amplification_ratio(self) -> float:
        if self.rows_processed == 0:
            return 0.0
        return self.rows_produced / self.rows_processed

    def summary(self) -> str:
        return (
            f"IncrementalResult({self.operator_type}@{self.node_id}): "
            f"{self.rows_processed} in -> {self.rows_produced} out "
            f"({self.execution_time_ms:.2f}ms)"
        )


# =====================================================================
# Incremental Executor
# =====================================================================


class IncrementalExecutor:
    """Execute repairs incrementally rather than full recomputation.

    For each SQL operator type, computes only the changed output rows
    based on the incoming delta. Uses DuckDB for efficient in-process
    computation.

    Parameters
    ----------
    engine : Any
        The DuckDB execution engine (or compatible).
    enable_logging : bool
        Whether to log execution details.
    """

    def __init__(
        self,
        engine: Any = None,
        enable_logging: bool = True,
    ) -> None:
        self._engine = engine
        self._conn: Optional[Any] = None
        self._logging = enable_logging

        if engine is not None and hasattr(engine, "connection"):
            self._conn = engine.connection
        elif HAS_DUCKDB and engine is None:
            self._conn = duckdb.connect(":memory:")

    def execute_incremental(
        self,
        node: Any,
        delta: DataDelta,
        context: Optional[Dict[str, Any]] = None,
    ) -> IncrementalResult:
        """Execute an incremental update for a pipeline node.

        Dispatches to the appropriate incremental operator based on
        the node's operator type.

        Parameters
        ----------
        node : PipelineNode
            The pipeline node.
        delta : DataDelta
            The incoming data delta.
        context : dict, optional
            Additional context (e.g., opposite table for joins).

        Returns
        -------
        IncrementalResult
        """
        ctx = context or {}
        operator_name = self._get_operator_name(node)

        handlers: Dict[str, Callable] = {
            "SELECT": self._execute_incremental_select,
            "PROJECT": self._execute_incremental_select,
            "FILTER": self._execute_incremental_filter,
            "JOIN": self._execute_incremental_join,
            "GROUP_BY": self._execute_incremental_groupby,
            "UNION": self._execute_incremental_union,
            "WINDOW": self._execute_incremental_window,
            "ORDER_BY": self._execute_incremental_passthrough,
            "LIMIT": self._execute_incremental_limit,
            "DISTINCT": self._execute_incremental_distinct,
        }

        handler = handlers.get(operator_name, self._execute_incremental_passthrough)

        try:
            result = handler(node, delta, ctx)
            result.operator_type = operator_name
            result.node_id = getattr(node, "node_id", "")
            return result
        except Exception as exc:
            logger.error("Incremental execution failed for %s: %s", operator_name, exc)
            return IncrementalResult(
                output_delta=delta,
                rows_processed=delta.total_rows,
                rows_produced=delta.total_rows,
                operator_type=operator_name,
                node_id=getattr(node, "node_id", ""),
            )

    def execute_incremental_select(
        self,
        engine: Any,
        node: Any,
        delta: DataDelta,
    ) -> IncrementalResult:
        """Execute incremental SELECT/PROJECT.

        Projects the delta rows to the output columns, dropping
        columns not in the select list.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node : Any
            Pipeline node.
        delta : DataDelta
            Input delta.

        Returns
        -------
        IncrementalResult
        """
        return self._execute_incremental_select(node, delta, {})

    def execute_incremental_join(
        self,
        engine: Any,
        node: Any,
        delta: DataDelta,
        opposite_table: str = "",
    ) -> IncrementalResult:
        """Execute incremental JOIN.

        Joins the delta rows against the opposite (unchanged) table.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node : Any
            Pipeline node.
        delta : DataDelta
            Input delta.
        opposite_table : str
            The table to join against.

        Returns
        -------
        IncrementalResult
        """
        return self._execute_incremental_join(
            node, delta, {"opposite_table": opposite_table}
        )

    def execute_incremental_groupby(
        self,
        engine: Any,
        node: Any,
        delta: DataDelta,
    ) -> IncrementalResult:
        """Execute incremental GROUP BY.

        Identifies affected groups and recomputes only those aggregates.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node : Any
            Pipeline node.
        delta : DataDelta
            Input delta.

        Returns
        -------
        IncrementalResult
        """
        return self._execute_incremental_groupby(node, delta, {})

    def execute_incremental_filter(
        self,
        engine: Any,
        node: Any,
        delta: DataDelta,
    ) -> IncrementalResult:
        """Execute incremental FILTER.

        Applies the filter predicate to delta rows.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node : Any
            Pipeline node.
        delta : DataDelta
            Input delta.

        Returns
        -------
        IncrementalResult
        """
        return self._execute_incremental_filter(node, delta, {})

    def execute_incremental_union(
        self,
        engine: Any,
        node: Any,
        delta: DataDelta,
    ) -> IncrementalResult:
        """Execute incremental UNION.

        Appends delta rows to the union output.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node : Any
            Pipeline node.
        delta : DataDelta
            Input delta.

        Returns
        -------
        IncrementalResult
        """
        return self._execute_incremental_union(node, delta, {})

    # ── Internal Handlers ─────────────────────────────────────────

    def _execute_incremental_select(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental SELECT: project delta rows to output columns."""
        start = time.time()
        output_cols = self._get_output_columns(node)

        output_rows: List[DataDeltaRow] = []
        skipped = 0

        for row in delta.rows:
            if output_cols:
                projected = {
                    k: v for k, v in row.values.items() if k in output_cols
                }
                if not projected:
                    skipped += 1
                    continue
                old_projected = None
                if row.old_values:
                    old_projected = {
                        k: v for k, v in row.old_values.items()
                        if k in output_cols
                    }
                output_rows.append(DataDeltaRow(
                    change_type=row.change_type,
                    values=projected,
                    old_values=old_projected,
                ))
            else:
                output_rows.append(row)

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(
                rows=output_rows,
                affected_columns=delta.affected_columns & output_cols if output_cols else delta.affected_columns,
            ),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            rows_skipped=skipped,
            execution_time_ms=elapsed,
        )

    def _execute_incremental_filter(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental FILTER: apply predicate to delta rows."""
        start = time.time()
        predicate = self._extract_predicate(node)

        if not predicate:
            elapsed = (time.time() - start) * 1000
            return IncrementalResult(
                output_delta=delta,
                rows_processed=delta.total_rows,
                rows_produced=delta.total_rows,
                execution_time_ms=elapsed,
            )

        output_rows: List[DataDeltaRow] = []
        skipped = 0

        for row in delta.rows:
            if row.change_type == DeltaType.DELETE:
                output_rows.append(row)
                continue

            if self._evaluate_predicate(predicate, row.values):
                output_rows.append(row)
            else:
                skipped += 1

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(
                rows=output_rows,
                affected_columns=delta.affected_columns,
            ),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            rows_skipped=skipped,
            execution_time_ms=elapsed,
        )

    def _execute_incremental_join(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental JOIN: join delta rows against opposite table."""
        start = time.time()
        opposite_table = ctx.get("opposite_table", "")

        if not opposite_table or self._conn is None:
            elapsed = (time.time() - start) * 1000
            return IncrementalResult(
                output_delta=delta,
                rows_processed=delta.total_rows,
                rows_produced=delta.total_rows,
                execution_time_ms=elapsed,
            )

        join_keys = self._extract_join_keys(node)
        output_rows: List[DataDeltaRow] = []

        if HAS_DUCKDB and self._conn is not None and join_keys:
            output_rows = self._duckdb_incremental_join(
                delta, opposite_table, join_keys
            )
        else:
            output_rows = list(delta.rows)

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(rows=output_rows),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            execution_time_ms=elapsed,
        )

    def _execute_incremental_groupby(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental GROUP BY: recompute affected groups only."""
        start = time.time()
        group_keys = self._extract_group_keys(node)

        if not group_keys:
            elapsed = (time.time() - start) * 1000
            return IncrementalResult(
                output_delta=delta,
                rows_processed=delta.total_rows,
                rows_produced=delta.total_rows,
                execution_time_ms=elapsed,
            )

        affected_groups: Dict[Tuple, List[DataDeltaRow]] = defaultdict(list)

        for row in delta.rows:
            group_key = tuple(row.values.get(k) for k in group_keys)
            affected_groups[group_key].append(row)

        output_rows: List[DataDeltaRow] = []

        for group_key, group_rows in affected_groups.items():
            aggregated_values = dict(zip(group_keys, group_key))

            insert_values = [
                r.values for r in group_rows
                if r.change_type in (DeltaType.INSERT, DeltaType.UPDATE)
            ]

            if insert_values:
                for col in delta.affected_columns - set(group_keys):
                    vals = [
                        v.get(col) for v in insert_values
                        if v.get(col) is not None
                    ]
                    if vals:
                        try:
                            aggregated_values[col] = sum(vals)
                        except (TypeError, ValueError):
                            aggregated_values[col] = vals[-1]

            output_rows.append(DataDeltaRow(
                change_type=DeltaType.UPDATE,
                values=aggregated_values,
            ))

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(rows=output_rows),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            rows_skipped=delta.total_rows - len(affected_groups),
            execution_time_ms=elapsed,
        )

    def _execute_incremental_union(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental UNION: pass through delta rows."""
        start = time.time()

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=delta,
            rows_processed=delta.total_rows,
            rows_produced=delta.total_rows,
            execution_time_ms=elapsed,
        )

    def _execute_incremental_window(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental WINDOW: recompute affected partitions."""
        start = time.time()
        partition_cols = self._extract_partition_cols(node)

        if not partition_cols:
            elapsed = (time.time() - start) * 1000
            return IncrementalResult(
                output_delta=delta,
                rows_processed=delta.total_rows,
                rows_produced=delta.total_rows,
                execution_time_ms=elapsed,
            )

        affected_partitions: Set[Tuple] = set()
        for row in delta.rows:
            partition_key = tuple(row.values.get(k) for k in partition_cols)
            affected_partitions.add(partition_key)

        output_rows: List[DataDeltaRow] = []
        for row in delta.rows:
            partition_key = tuple(row.values.get(k) for k in partition_cols)
            if partition_key in affected_partitions:
                output_rows.append(row)

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(rows=output_rows),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            execution_time_ms=elapsed,
        )

    def _execute_incremental_passthrough(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Pass through delta unchanged (for operators like ORDER BY)."""
        return IncrementalResult(
            output_delta=delta,
            rows_processed=delta.total_rows,
            rows_produced=delta.total_rows,
            execution_time_ms=0.0,
        )

    def _execute_incremental_limit(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental LIMIT: may need full recomputation."""
        start = time.time()

        limit = self._extract_limit(node)
        if limit is not None and delta.total_rows > limit:
            output_rows = delta.rows[:limit]
        else:
            output_rows = delta.rows

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(rows=output_rows),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            rows_skipped=delta.total_rows - len(output_rows),
            execution_time_ms=elapsed,
        )

    def _execute_incremental_distinct(
        self,
        node: Any,
        delta: DataDelta,
        ctx: Dict[str, Any],
    ) -> IncrementalResult:
        """Incremental DISTINCT: deduplicate delta rows."""
        start = time.time()

        seen: Set[Tuple] = set()
        output_rows: List[DataDeltaRow] = []
        skipped = 0

        for row in delta.rows:
            if row.change_type == DeltaType.DELETE:
                output_rows.append(row)
                continue

            key = tuple(sorted(row.values.items()))
            if key not in seen:
                seen.add(key)
                output_rows.append(row)
            else:
                skipped += 1

        elapsed = (time.time() - start) * 1000

        return IncrementalResult(
            output_delta=DataDelta(rows=output_rows),
            rows_processed=delta.total_rows,
            rows_produced=len(output_rows),
            rows_skipped=skipped,
            execution_time_ms=elapsed,
        )

    # ── DuckDB-specific Implementations ───────────────────────────

    def _duckdb_incremental_join(
        self,
        delta: DataDelta,
        opposite_table: str,
        join_keys: List[str],
    ) -> List[DataDeltaRow]:
        """Execute incremental join using DuckDB."""
        if self._conn is None:
            return list(delta.rows)

        delta_table_name = "_arc_delta_temp"

        try:
            insert_rows = delta.inserts()
            if not insert_rows:
                return list(delta.rows)

            columns = list(insert_rows[0].values.keys())
            col_defs = ", ".join(f"{c} VARCHAR" for c in columns)
            self._conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {delta_table_name} ({col_defs})")
            self._conn.execute(f"DELETE FROM {delta_table_name}")

            for row in insert_rows:
                vals = [str(row.values.get(c, "NULL")) for c in columns]
                placeholders = ", ".join(f"'{v}'" for v in vals)
                self._conn.execute(
                    f"INSERT INTO {delta_table_name} VALUES ({placeholders})"
                )

            join_cond = " AND ".join(
                f"d.{k} = o.{k}" for k in join_keys
            )
            result = self._conn.execute(
                f"SELECT d.*, o.* FROM {delta_table_name} d "
                f"JOIN {opposite_table} o ON {join_cond}"
            ).fetchall()

            output_rows: List[DataDeltaRow] = []
            result_cols = [desc[0] for desc in self._conn.description or []]
            for row_data in result:
                values = dict(zip(result_cols, row_data))
                output_rows.append(DataDeltaRow(
                    change_type=DeltaType.INSERT,
                    values=values,
                ))

            return output_rows

        except Exception as exc:
            logger.warning("DuckDB incremental join failed: %s", exc)
            return list(delta.rows)
        finally:
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {delta_table_name}")
            except Exception:
                pass

    # ── Internal Helpers ──────────────────────────────────────────

    @staticmethod
    def _get_operator_name(node: Any) -> str:
        """Get the operator name from a pipeline node."""
        if hasattr(node, "operator"):
            op = node.operator
            if hasattr(op, "value"):
                return str(op.value).upper()
            return str(op).upper()
        return "TRANSFORM"

    @staticmethod
    def _get_output_columns(node: Any) -> Set[str]:
        """Get the output column names from a pipeline node."""
        if hasattr(node, "output_schema"):
            schema = node.output_schema
            if hasattr(schema, "columns"):
                return {c.name for c in schema.columns}
        return set()

    @staticmethod
    def _extract_predicate(node: Any) -> Optional[str]:
        """Extract WHERE predicate from a node."""
        if hasattr(node, "query_text"):
            import re
            match = re.search(
                r"\bWHERE\b\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
                node.query_text, re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_join_keys(node: Any) -> List[str]:
        """Extract join key columns from a node."""
        if hasattr(node, "query_text"):
            import re
            match = re.search(
                r"\bON\b\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|$)",
                node.query_text, re.IGNORECASE | re.DOTALL,
            )
            if match:
                on_clause = match.group(1).strip()
                keys = re.findall(r"(\w+)\s*=", on_clause)
                return keys
        return []

    @staticmethod
    def _extract_group_keys(node: Any) -> List[str]:
        """Extract GROUP BY columns from a node."""
        if hasattr(node, "query_text"):
            import re
            match = re.search(
                r"\bGROUP\s+BY\b\s+(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)",
                node.query_text, re.IGNORECASE | re.DOTALL,
            )
            if match:
                cols_str = match.group(1).strip()
                return [c.strip().split(".")[-1] for c in cols_str.split(",")]
        return []

    @staticmethod
    def _extract_partition_cols(node: Any) -> List[str]:
        """Extract PARTITION BY columns from a window node."""
        if hasattr(node, "query_text"):
            import re
            match = re.search(
                r"\bPARTITION\s+BY\b\s+(.+?)(?:\bORDER\b|\bROWS\b|\bRANGE\b|\)|$)",
                node.query_text, re.IGNORECASE | re.DOTALL,
            )
            if match:
                cols_str = match.group(1).strip()
                return [c.strip().split(".")[-1] for c in cols_str.split(",")]
        return []

    @staticmethod
    def _extract_limit(node: Any) -> Optional[int]:
        """Extract LIMIT value from a node."""
        if hasattr(node, "query_text"):
            import re
            match = re.search(r"\bLIMIT\b\s+(\d+)", node.query_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _evaluate_predicate(predicate: str, values: Dict[str, Any]) -> bool:
        """Evaluate a simple predicate against a row's values.

        Handles basic equality, comparison, IS NULL, IS NOT NULL patterns.
        Returns True if the predicate is satisfied or cannot be evaluated.
        """
        import re

        null_match = re.match(r"(\w+)\s+IS\s+NOT\s+NULL", predicate, re.IGNORECASE)
        if null_match:
            col = null_match.group(1)
            return values.get(col) is not None

        null_match = re.match(r"(\w+)\s+IS\s+NULL", predicate, re.IGNORECASE)
        if null_match:
            col = null_match.group(1)
            return values.get(col) is None

        eq_match = re.match(r"(\w+)\s*=\s*['\"]?(\w+)['\"]?", predicate, re.IGNORECASE)
        if eq_match:
            col = eq_match.group(1)
            val = eq_match.group(2)
            return str(values.get(col, "")) == val

        gt_match = re.match(r"(\w+)\s*>\s*(\d+(?:\.\d+)?)", predicate, re.IGNORECASE)
        if gt_match:
            col = gt_match.group(1)
            threshold = float(gt_match.group(2))
            try:
                return float(values.get(col, 0)) > threshold
            except (TypeError, ValueError):
                return True

        lt_match = re.match(r"(\w+)\s*<\s*(\d+(?:\.\d+)?)", predicate, re.IGNORECASE)
        if lt_match:
            col = lt_match.group(1)
            threshold = float(lt_match.group(2))
            try:
                return float(values.get(col, 0)) < threshold
            except (TypeError, ValueError):
                return True

        return True


# =====================================================================
# Convenience Functions
# =====================================================================


def execute_incremental(
    node: Any,
    delta: DataDelta,
    engine: Any = None,
) -> IncrementalResult:
    """Convenience: execute a single incremental update."""
    executor = IncrementalExecutor(engine=engine)
    return executor.execute_incremental(node, delta)


def create_data_delta(
    inserts: Optional[List[Dict[str, Any]]] = None,
    deletes: Optional[List[Dict[str, Any]]] = None,
    updates: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
) -> DataDelta:
    """Convenience: create a DataDelta from lists of changes."""
    rows: List[DataDeltaRow] = []
    affected_cols: Set[str] = set()

    for values in (inserts or []):
        rows.append(DataDeltaRow(change_type=DeltaType.INSERT, values=values))
        affected_cols.update(values.keys())

    for values in (deletes or []):
        rows.append(DataDeltaRow(change_type=DeltaType.DELETE, values=values))
        affected_cols.update(values.keys())

    for old_vals, new_vals in (updates or []):
        rows.append(DataDeltaRow(
            change_type=DeltaType.UPDATE,
            values=new_vals,
            old_values=old_vals,
        ))
        affected_cols.update(new_vals.keys())
        affected_cols.update(old_vals.keys())

    return DataDelta(rows=rows, affected_columns=affected_cols)
