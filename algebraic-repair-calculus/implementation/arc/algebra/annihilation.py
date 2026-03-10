"""
Delta Annihilation Detection (Algorithm A2: ANNIHILATE)
========================================================

Detects when deltas are fully or partially annihilated by SQL operators.
An annihilation occurs when an operator makes a perturbation irrelevant:
  - A SELECT drops the column that was added (schema annihilation)
  - A FILTER removes all affected rows (data annihilation)
  - A GROUP BY absorbs per-row changes into aggregates

Annihilation detection is critical for pruning unnecessary repairs and
reducing the size of repair plans.

Types of annihilation:
  - **Full annihilation**: the delta has no effect on the operator's output.
  - **Schema annihilation**: schema-level changes are absorbed.
  - **Data annihilation**: data-level changes are filtered out.
  - **Quality annihilation**: quality violations are resolved by the operator.
  - **Partial annihilation**: some sub-deltas survive while others are absorbed.
"""

from __future__ import annotations

import logging
import re
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
from arc.algebra.data_delta import (
    DataDelta,
    DataOperation,
    DeleteOp,
    InsertOp,
    MultiSet,
    TypedTuple,
    UpdateOp,
)
from arc.algebra.quality_delta import (
    ConstraintAdded,
    ConstraintRemoved,
    DistributionShift,
    QualityDelta,
    QualityImprovement,
    QualityOperation,
    QualityViolation,
    SeverityLevel,
    ViolationType,
)
from arc.algebra.composition import CompoundPerturbation

logger = logging.getLogger(__name__)


# =====================================================================
# Annihilation Type Enumeration
# =====================================================================


class AnnihilationType(Enum):
    """Classification of annihilation events."""
    NONE = "none"
    FULL = "full"
    SCHEMA = "schema"
    DATA = "data"
    QUALITY = "quality"
    PARTIAL_SCHEMA = "partial_schema"
    PARTIAL_DATA = "partial_data"
    PARTIAL_QUALITY = "partial_quality"


class AnnihilationReason(Enum):
    """Detailed reason for annihilation."""
    NO_ANNIHILATION = "no_annihilation"
    COLUMN_NOT_IN_SELECT = "column_not_in_select"
    COLUMN_NOT_IN_OUTPUT = "column_not_in_output"
    FILTER_CONTRADICTS_DELTA = "filter_contradicts_delta"
    FILTER_REMOVES_ALL_ROWS = "filter_removes_all_rows"
    JOIN_EMPTY_INPUT = "join_empty_input"
    JOIN_KEY_MISMATCH = "join_key_mismatch"
    GROUPBY_NON_KEY_COLUMN = "groupby_non_key_column"
    GROUPBY_ABSORBED_BY_AGG = "groupby_absorbed_by_aggregate"
    UNION_SCHEMA_INCOMPATIBLE = "union_schema_incompatible"
    WINDOW_NON_PARTITION_COLUMN = "window_non_partition_column"
    CTE_UNUSED = "cte_unused"
    SETOP_EXCEPT_REMOVES = "setop_except_removes"
    DISTINCT_ABSORBS_DUPLICATE = "distinct_absorbs_duplicate"
    LIMIT_TRUNCATES = "limit_truncates"
    TYPE_CAST_ABSORBS = "type_cast_absorbs"
    CONSTRAINT_ALREADY_SATISFIED = "constraint_already_satisfied"
    DEFAULT_VALUE_OVERRIDES = "default_value_overrides"


# =====================================================================
# Annihilation Result
# =====================================================================


@dataclass
class AnnihilationResult:
    """Result of an annihilation check for a single operator and delta.

    Attributes
    ----------
    fully_annihilated : bool
        True if the entire delta is annihilated (no effect on output).
    schema_annihilated : bool
        True if all schema changes are annihilated.
    data_annihilated : bool
        True if all data changes are annihilated.
    quality_annihilated : bool
        True if all quality changes are annihilated.
    surviving_delta : CompoundPerturbation | None
        The part of the delta that survives, if partial annihilation.
    annihilated_delta : CompoundPerturbation | None
        The part of the delta that was annihilated.
    annihilation_type : AnnihilationType
        Classification of the annihilation.
    annihilation_reason : str
        Human-readable explanation of why annihilation occurred.
    annihilation_reasons : list[AnnihilationReason]
        Detailed reasons for the annihilation.
    strength : float
        Annihilation strength from 0.0 (no annihilation) to 1.0 (full).
    """
    fully_annihilated: bool = False
    schema_annihilated: bool = False
    data_annihilated: bool = False
    quality_annihilated: bool = False
    surviving_delta: Optional[CompoundPerturbation] = None
    annihilated_delta: Optional[CompoundPerturbation] = None
    annihilation_type: AnnihilationType = AnnihilationType.NONE
    annihilation_reason: str = ""
    annihilation_reasons: List[AnnihilationReason] = field(default_factory=list)
    strength: float = 0.0

    @property
    def is_partial(self) -> bool:
        """True if annihilation is partial (some deltas survive)."""
        return (
            not self.fully_annihilated
            and (self.schema_annihilated or self.data_annihilated
                 or self.quality_annihilated)
        )

    @property
    def has_surviving_delta(self) -> bool:
        """True if there is a surviving delta component."""
        return self.surviving_delta is not None and not self._is_identity(
            self.surviving_delta
        )

    @staticmethod
    def _is_identity(delta: CompoundPerturbation) -> bool:
        s_empty = (
            delta.schema_delta is None or len(delta.schema_delta.operations) == 0
        )
        d_empty = (
            delta.data_delta is None or len(delta.data_delta.operations) == 0
        )
        q_empty = (
            delta.quality_delta is None or len(delta.quality_delta.operations) == 0
        )
        return s_empty and d_empty and q_empty

    @staticmethod
    def no_annihilation(delta: CompoundPerturbation) -> AnnihilationResult:
        """Create a result indicating no annihilation occurred."""
        return AnnihilationResult(
            surviving_delta=delta,
            annihilation_type=AnnihilationType.NONE,
            annihilation_reason="Delta passes through operator unchanged",
            annihilation_reasons=[AnnihilationReason.NO_ANNIHILATION],
            strength=0.0,
        )

    @staticmethod
    def full(reason: str, reasons: Optional[List[AnnihilationReason]] = None) -> AnnihilationResult:
        """Create a result indicating full annihilation."""
        return AnnihilationResult(
            fully_annihilated=True,
            schema_annihilated=True,
            data_annihilated=True,
            quality_annihilated=True,
            surviving_delta=None,
            annihilation_type=AnnihilationType.FULL,
            annihilation_reason=reason,
            annihilation_reasons=reasons or [],
            strength=1.0,
        )

    def summary(self) -> str:
        """Human-readable summary of the annihilation result."""
        if self.fully_annihilated:
            return f"FULLY ANNIHILATED: {self.annihilation_reason}"
        if self.is_partial:
            parts = []
            if self.schema_annihilated:
                parts.append("schema")
            if self.data_annihilated:
                parts.append("data")
            if self.quality_annihilated:
                parts.append("quality")
            return (
                f"PARTIAL ({', '.join(parts)}): {self.annihilation_reason} "
                f"[strength={self.strength:.2f}]"
            )
        return "NO ANNIHILATION"


# =====================================================================
# Operator Configuration Extraction
# =====================================================================


def _extract_select_columns(config: Any) -> Set[str]:
    """Extract selected column names from operator configuration."""
    columns: Set[str] = set()

    if isinstance(config, dict):
        if "output_schema" in config:
            schema = config["output_schema"]
            if hasattr(schema, "columns"):
                columns = {c.name for c in schema.columns}
            elif isinstance(schema, dict) and "columns" in schema:
                columns = {
                    c.get("name", "") for c in schema["columns"]
                    if isinstance(c, dict)
                }
        if "select_columns" in config:
            columns = set(config["select_columns"])

    elif hasattr(config, "output_schema"):
        schema = config.output_schema
        if hasattr(schema, "columns"):
            columns = {c.name for c in schema.columns}

    return columns


def _extract_filter_predicate(config: Any) -> Optional[str]:
    """Extract filter predicate from operator configuration."""
    if isinstance(config, dict):
        pred = config.get("predicate") or config.get("where_clause")
        if pred:
            return str(pred)
        query = config.get("query_text", "")
        if query:
            return _extract_where_from_sql(query)
    elif hasattr(config, "query_text"):
        return _extract_where_from_sql(config.query_text)
    return None


def _extract_where_from_sql(sql: str) -> Optional[str]:
    """Extract WHERE clause from a SQL query string."""
    if not sql:
        return None
    match = re.search(r"\bWHERE\b\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_groupby_keys(config: Any) -> Set[str]:
    """Extract GROUP BY key columns from operator configuration."""
    keys: Set[str] = set()

    if isinstance(config, dict):
        if "group_by_columns" in config:
            keys = set(config["group_by_columns"])
        elif "group_keys" in config:
            keys = set(config["group_keys"])
        elif "query_text" in config:
            keys = _extract_groupby_from_sql(config["query_text"])
    elif hasattr(config, "group_by_columns"):
        keys = set(config.group_by_columns)

    return keys


def _extract_groupby_from_sql(sql: str) -> Set[str]:
    """Extract GROUP BY columns from a SQL query string."""
    if not sql:
        return set()
    match = re.search(r"\bGROUP\s+BY\b\s+(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        cols_str = match.group(1).strip()
        cols = [c.strip().split(".")[-1] for c in cols_str.split(",")]
        return {c for c in cols if c}
    return set()


def _extract_join_keys(config: Any) -> Tuple[Set[str], Set[str]]:
    """Extract join key columns from both sides."""
    left_keys: Set[str] = set()
    right_keys: Set[str] = set()

    if isinstance(config, dict):
        if "join_keys_left" in config:
            left_keys = set(config["join_keys_left"])
        if "join_keys_right" in config:
            right_keys = set(config["join_keys_right"])
        if "on_columns" in config:
            for col in config["on_columns"]:
                left_keys.add(col)
                right_keys.add(col)
    return left_keys, right_keys


def _extract_window_partition_cols(config: Any) -> Set[str]:
    """Extract PARTITION BY columns from window function config."""
    cols: Set[str] = set()
    if isinstance(config, dict):
        if "partition_by" in config:
            cols = set(config["partition_by"])
        elif "query_text" in config:
            cols = _extract_partition_from_sql(config["query_text"])
    return cols


def _extract_partition_from_sql(sql: str) -> Set[str]:
    """Extract PARTITION BY columns from SQL."""
    if not sql:
        return set()
    match = re.search(r"\bPARTITION\s+BY\b\s+(.+?)(?:\bORDER\b|\bROWS\b|\bRANGE\b|\)|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        cols_str = match.group(1).strip()
        cols = [c.strip().split(".")[-1] for c in cols_str.split(",")]
        return {c for c in cols if c}
    return set()


def _extract_cte_names_used(config: Any) -> Set[str]:
    """Extract the names of CTEs that are actually referenced."""
    used: Set[str] = set()
    if isinstance(config, dict):
        query = config.get("query_text", "")
        if query:
            used = _find_cte_references(query)
    elif hasattr(config, "query_text"):
        used = _find_cte_references(config.query_text)
    return used


def _find_cte_references(sql: str) -> Set[str]:
    """Find CTE names referenced in the main query body."""
    cte_match = re.search(r"\bWITH\b\s+(.+?)\bSELECT\b", sql, re.IGNORECASE | re.DOTALL)
    if not cte_match:
        return set()

    main_body_start = cte_match.end() - len("SELECT")
    main_body = sql[main_body_start:]

    cte_defs = cte_match.group(1)
    cte_names: Set[str] = set()
    for match in re.finditer(r"(\w+)\s+AS\s*\(", cte_defs, re.IGNORECASE):
        cte_names.add(match.group(1).lower())

    used: Set[str] = set()
    main_lower = main_body.lower()
    for name in cte_names:
        if re.search(r"\b" + re.escape(name) + r"\b", main_lower):
            used.add(name)

    return used


# =====================================================================
# Schema Annihilation Helpers
# =====================================================================


def _get_affected_columns_from_schema_delta(
    schema_delta: SchemaDelta,
) -> Set[str]:
    """Extract the set of column names affected by a schema delta."""
    columns: Set[str] = set()
    for op in schema_delta.operations:
        if isinstance(op, AddColumn):
            columns.add(op.column_def.name)
        elif isinstance(op, DropColumn):
            columns.add(op.column_name)
        elif isinstance(op, RenameColumn):
            columns.add(op.old_name)
            columns.add(op.new_name)
        elif isinstance(op, ChangeType):
            columns.add(op.column_name)
        elif isinstance(op, (AddConstraint, DropConstraint)):
            if hasattr(op, "column_name") and op.column_name:
                columns.add(op.column_name)
            if hasattr(op, "columns"):
                columns.update(op.columns)
    return columns


def _get_affected_columns_from_data_delta(
    data_delta: DataDelta,
) -> Set[str]:
    """Extract columns affected by data operations."""
    columns: Set[str] = set()
    for op in data_delta.operations:
        if isinstance(op, InsertOp):
            if hasattr(op, "tuple") and op.tuple is not None:
                columns.update(op.tuple.columns())
            elif hasattr(op, "values") and isinstance(op.values, dict):
                columns.update(op.values.keys())
        elif isinstance(op, DeleteOp):
            if hasattr(op, "tuple") and op.tuple is not None:
                columns.update(op.tuple.columns())
        elif isinstance(op, UpdateOp):
            if hasattr(op, "changes") and isinstance(op.changes, dict):
                columns.update(op.changes.keys())
            if hasattr(op, "old_values") and isinstance(op.old_values, dict):
                columns.update(op.old_values.keys())
    return columns


# =====================================================================
# Annihilation Detector
# =====================================================================


class AnnihilationDetector:
    """Detect when deltas are annihilated by SQL operators.

    Implements Algorithm A2 (ANNIHILATE) from the Algebraic Repair Calculus.

    For each operator type, checks whether the incoming delta has any effect
    on the operator's output. If the delta is fully or partially annihilated,
    returns the surviving and annihilated components.

    Parameters
    ----------
    strict_mode : bool
        If True, use conservative checks (may miss some annihilations).
        If False, use aggressive checks (may incorrectly classify some).
    track_reasons : bool
        If True, collect detailed reasons for each annihilation decision.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        track_reasons: bool = True,
    ) -> None:
        self._strict = strict_mode
        self._track_reasons = track_reasons

        self._operator_handlers: Dict[str, Callable] = {
            "SELECT": self._check_select,
            "PROJECT": self._check_select,
            "JOIN": self._check_join,
            "GROUP_BY": self._check_groupby,
            "FILTER": self._check_filter,
            "UNION": self._check_union,
            "WINDOW": self._check_window,
            "CTE": self._check_cte,
            "SET_OP": self._check_setop,
            "INTERSECT": self._check_setop,
            "EXCEPT": self._check_setop,
            "DISTINCT": self._check_distinct,
            "LIMIT": self._check_limit,
            "ORDER_BY": self._check_orderby,
            "CAST": self._check_cast,
        }

    # ── Public API ────────────────────────────────────────────────

    def check_annihilation(
        self,
        operator: Any,
        delta: CompoundPerturbation,
        config: Any = None,
    ) -> AnnihilationResult:
        """Check if a delta is annihilated by an operator.

        Parameters
        ----------
        operator : Any
            The SQL operator (enum value, string, or node object).
        delta : CompoundPerturbation
            The incoming compound delta.
        config : Any
            Operator-specific configuration.

        Returns
        -------
        AnnihilationResult
            Full annihilation analysis.
        """
        if self._is_identity(delta):
            return AnnihilationResult.full(
                "Identity delta is trivially annihilated",
                [AnnihilationReason.NO_ANNIHILATION],
            )

        op_name = self._normalize_operator(operator)
        handler = self._operator_handlers.get(op_name)

        if handler is None:
            return AnnihilationResult.no_annihilation(delta)

        try:
            return handler(delta, config or {})
        except Exception as exc:
            logger.warning(
                "Annihilation check failed for %s: %s", op_name, exc
            )
            return AnnihilationResult.no_annihilation(delta)

    def partial_annihilation(
        self,
        operator: Any,
        delta: CompoundPerturbation,
        config: Any = None,
    ) -> Tuple[CompoundPerturbation, CompoundPerturbation]:
        """Split a delta into annihilated and surviving components.

        Parameters
        ----------
        operator : Any
            The SQL operator.
        delta : CompoundPerturbation
            The incoming compound delta.
        config : Any
            Operator-specific configuration.

        Returns
        -------
        tuple[CompoundPerturbation, CompoundPerturbation]
            (surviving_delta, annihilated_delta) pair.
        """
        result = self.check_annihilation(operator, delta, config)

        if result.fully_annihilated:
            identity = CompoundPerturbation(
                schema_delta=SchemaDelta(operations=[]),
                data_delta=DataDelta(operations=[]),
                quality_delta=QualityDelta(operations=[]),
            )
            return identity, delta

        if result.surviving_delta is not None and result.annihilated_delta is not None:
            return result.surviving_delta, result.annihilated_delta

        if result.surviving_delta is not None:
            annihilated = self._compute_annihilated_portion(delta, result.surviving_delta)
            return result.surviving_delta, annihilated

        return delta, CompoundPerturbation(
            schema_delta=SchemaDelta(operations=[]),
            data_delta=DataDelta(operations=[]),
            quality_delta=QualityDelta(operations=[]),
        )

    def annihilation_strength(
        self,
        operator: Any,
        delta: CompoundPerturbation,
        config: Any = None,
    ) -> float:
        """Quantify the degree of annihilation from 0.0 to 1.0.

        Parameters
        ----------
        operator : Any
            The SQL operator.
        delta : CompoundPerturbation
            The incoming delta.
        config : Any
            Operator-specific configuration.

        Returns
        -------
        float
            0.0 = no annihilation, 1.0 = full annihilation.
        """
        result = self.check_annihilation(operator, delta, config)
        return result.strength

    def check_chain_annihilation(
        self,
        operators: List[Tuple[Any, Any]],
        delta: CompoundPerturbation,
    ) -> AnnihilationResult:
        """Check annihilation across a chain of operators.

        Propagates the surviving delta through each operator in sequence,
        checking for annihilation at each step.

        Parameters
        ----------
        operators : list[tuple[Any, Any]]
            List of (operator, config) pairs in pipeline order.
        delta : CompoundPerturbation
            The initial delta.

        Returns
        -------
        AnnihilationResult
            Combined annihilation result.
        """
        current_delta = delta
        all_reasons: List[AnnihilationReason] = []
        total_strength = 0.0
        original_size = self._delta_size(delta)

        for operator, config in operators:
            result = self.check_annihilation(operator, current_delta, config)

            if result.fully_annihilated:
                return AnnihilationResult(
                    fully_annihilated=True,
                    schema_annihilated=True,
                    data_annihilated=True,
                    quality_annihilated=True,
                    annihilation_type=AnnihilationType.FULL,
                    annihilation_reason=(
                        f"Delta fully annihilated at operator {self._normalize_operator(operator)}"
                    ),
                    annihilation_reasons=all_reasons + result.annihilation_reasons,
                    strength=1.0,
                )

            all_reasons.extend(result.annihilation_reasons)

            if result.surviving_delta is not None:
                current_delta = result.surviving_delta

        surviving_size = self._delta_size(current_delta)
        if original_size > 0:
            total_strength = 1.0 - (surviving_size / original_size)
        else:
            total_strength = 0.0

        return AnnihilationResult(
            fully_annihilated=False,
            schema_annihilated=self._schema_empty(current_delta),
            data_annihilated=self._data_empty(current_delta),
            quality_annihilated=self._quality_empty(current_delta),
            surviving_delta=current_delta,
            annihilation_type=AnnihilationType.PARTIAL_SCHEMA if total_strength > 0 else AnnihilationType.NONE,
            annihilation_reason=f"Chain annihilation strength: {total_strength:.2f}",
            annihilation_reasons=all_reasons,
            strength=total_strength,
        )

    def find_annihilation_points(
        self,
        graph: Any,
        source_node: str,
        delta: CompoundPerturbation,
    ) -> Dict[str, AnnihilationResult]:
        """Find all nodes in a graph where annihilation occurs.

        Traverses the graph from the source node and checks annihilation
        at each downstream node.

        Parameters
        ----------
        graph : Any
            The pipeline graph.
        source_node : str
            The source of the perturbation.
        delta : CompoundPerturbation
            The initial delta.

        Returns
        -------
        dict[str, AnnihilationResult]
            Mapping from node_id to annihilation result for nodes where
            some annihilation was detected.
        """
        results: Dict[str, AnnihilationResult] = {}
        topo_order = graph.topological_sort()

        current_deltas: Dict[str, CompoundPerturbation] = {source_node: delta}
        source_found = False

        for nid in topo_order:
            if nid == source_node:
                source_found = True
                continue

            if not source_found:
                continue

            predecessors = graph.predecessors(nid)
            incoming = [
                current_deltas[p]
                for p in predecessors
                if p in current_deltas and not self._is_identity(current_deltas[p])
            ]

            if not incoming:
                continue

            if len(incoming) == 1:
                current_delta = incoming[0]
            else:
                current_delta = self._merge_deltas(incoming)

            node = graph.get_node(nid)
            operator = getattr(node, "operator", "TRANSFORM")
            config = self._node_to_config(node)

            result = self.check_annihilation(operator, current_delta, config)
            if result.strength > 0:
                results[nid] = result

            if result.fully_annihilated:
                current_deltas[nid] = self._make_identity()
            elif result.surviving_delta is not None:
                current_deltas[nid] = result.surviving_delta
            else:
                current_deltas[nid] = current_delta

        return results

    # ── Operator-Specific Checks ──────────────────────────────────

    def _check_select(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a SELECT/PROJECT operator.

        Schema annihilation occurs when a column affected by the delta
        is not in the select list.
        """
        select_cols = _extract_select_columns(config)
        if not select_cols:
            return AnnihilationResult.no_annihilation(delta)

        schema_affected = set()
        if delta.schema_delta is not None:
            schema_affected = _get_affected_columns_from_schema_delta(
                delta.schema_delta
            )

        data_affected = set()
        if delta.data_delta is not None:
            data_affected = _get_affected_columns_from_data_delta(delta.data_delta)

        all_affected = schema_affected | data_affected

        if not all_affected:
            return AnnihilationResult.no_annihilation(delta)

        surviving_cols = all_affected & select_cols
        annihilated_cols = all_affected - select_cols

        if not surviving_cols and all_affected:
            return AnnihilationResult(
                fully_annihilated=True,
                schema_annihilated=True,
                data_annihilated=True,
                quality_annihilated=True,
                annihilation_type=AnnihilationType.SCHEMA,
                annihilation_reason=(
                    f"All affected columns {sorted(annihilated_cols)} "
                    f"not in SELECT list"
                ),
                annihilation_reasons=[AnnihilationReason.COLUMN_NOT_IN_SELECT],
                strength=1.0,
            )

        if annihilated_cols:
            surviving, annihilated = self._split_by_columns(
                delta, surviving_cols, annihilated_cols
            )
            strength = len(annihilated_cols) / len(all_affected)

            return AnnihilationResult(
                fully_annihilated=False,
                schema_annihilated=not bool(schema_affected & surviving_cols),
                data_annihilated=not bool(data_affected & surviving_cols),
                surviving_delta=surviving,
                annihilated_delta=annihilated,
                annihilation_type=AnnihilationType.PARTIAL_SCHEMA,
                annihilation_reason=(
                    f"Columns {sorted(annihilated_cols)} not in SELECT list; "
                    f"columns {sorted(surviving_cols)} survive"
                ),
                annihilation_reasons=[AnnihilationReason.COLUMN_NOT_IN_SELECT],
                strength=strength,
            )

        return AnnihilationResult.no_annihilation(delta)

    def _check_join(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a JOIN operator.

        Full annihilation can occur when:
        - The join has an empty input on either side
        - The delta affects join keys that result in no matches
        """
        left_keys, right_keys = _extract_join_keys(config)

        if isinstance(config, dict) and config.get("empty_input"):
            return AnnihilationResult.full(
                "JOIN has empty input, all changes annihilated",
                [AnnihilationReason.JOIN_EMPTY_INPUT],
            )

        if not left_keys and not right_keys:
            return AnnihilationResult.no_annihilation(delta)

        schema_affected = set()
        if delta.schema_delta is not None:
            schema_affected = _get_affected_columns_from_schema_delta(
                delta.schema_delta
            )

        all_join_keys = left_keys | right_keys
        key_overlap = schema_affected & all_join_keys

        if key_overlap and self._strict:
            return AnnihilationResult(
                fully_annihilated=False,
                schema_annihilated=False,
                data_annihilated=True,
                annihilation_type=AnnihilationType.DATA,
                annihilation_reason=(
                    f"Schema changes to join keys {sorted(key_overlap)} "
                    f"may annihilate data through key mismatch"
                ),
                annihilation_reasons=[AnnihilationReason.JOIN_KEY_MISMATCH],
                surviving_delta=delta,
                strength=0.3,
            )

        return AnnihilationResult.no_annihilation(delta)

    def _check_groupby(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a GROUP BY operator.

        Schema annihilation occurs when changes affect columns that are
        neither group keys nor aggregate inputs — they simply disappear
        from the output.

        Data annihilation occurs when per-row changes are absorbed by
        aggregate functions.
        """
        group_keys = _extract_groupby_keys(config)
        if not group_keys:
            return AnnihilationResult.no_annihilation(delta)

        schema_affected = set()
        if delta.schema_delta is not None:
            schema_affected = _get_affected_columns_from_schema_delta(
                delta.schema_delta
            )

        non_key_schema = schema_affected - group_keys
        key_schema = schema_affected & group_keys

        data_affected = set()
        if delta.data_delta is not None:
            data_affected = _get_affected_columns_from_data_delta(delta.data_delta)

        non_key_data = data_affected - group_keys

        reasons: List[AnnihilationReason] = []
        strength_components: List[float] = []

        schema_annihilated = False
        if non_key_schema and not key_schema:
            schema_annihilated = True
            reasons.append(AnnihilationReason.GROUPBY_NON_KEY_COLUMN)
            strength_components.append(1.0)
        elif non_key_schema:
            reasons.append(AnnihilationReason.GROUPBY_NON_KEY_COLUMN)
            frac = len(non_key_schema) / len(schema_affected) if schema_affected else 0
            strength_components.append(frac)

        data_annihilated = False
        if non_key_data:
            data_annihilated = not self._strict
            reasons.append(AnnihilationReason.GROUPBY_ABSORBED_BY_AGG)
            frac = len(non_key_data) / len(data_affected) if data_affected else 0
            strength_components.append(frac * 0.7)

        if schema_annihilated and data_annihilated:
            quality_annihilated = True
        else:
            quality_annihilated = False

        fully = schema_annihilated and data_annihilated and quality_annihilated

        total_affected = len(schema_affected) + len(data_affected)
        total_annihilated = len(non_key_schema) + (len(non_key_data) if data_annihilated else 0)
        strength = total_annihilated / total_affected if total_affected > 0 else 0.0

        surviving = None
        if not fully:
            surviving = self._filter_schema_ops_by_columns(delta, group_keys)

        ann_type = AnnihilationType.FULL if fully else (
            AnnihilationType.PARTIAL_SCHEMA if schema_annihilated else
            AnnihilationType.PARTIAL_DATA if data_annihilated else
            AnnihilationType.NONE
        )

        return AnnihilationResult(
            fully_annihilated=fully,
            schema_annihilated=schema_annihilated,
            data_annihilated=data_annihilated,
            quality_annihilated=quality_annihilated,
            surviving_delta=surviving,
            annihilation_type=ann_type,
            annihilation_reason=(
                f"GROUP BY keys={sorted(group_keys)}, "
                f"non-key schema={sorted(non_key_schema)}, "
                f"non-key data={sorted(non_key_data)}"
            ),
            annihilation_reasons=reasons,
            strength=strength,
        )

    def _check_filter(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a FILTER (WHERE) operator.

        Data annihilation occurs when the filter predicate removes all
        rows affected by the data delta.
        """
        predicate = _extract_filter_predicate(config)
        if not predicate:
            return AnnihilationResult.no_annihilation(delta)

        if delta.data_delta is None or len(delta.data_delta.operations) == 0:
            return AnnihilationResult.no_annihilation(delta)

        contradiction = self._check_predicate_contradiction(
            predicate, delta.data_delta
        )

        if contradiction:
            return AnnihilationResult(
                fully_annihilated=False,
                schema_annihilated=False,
                data_annihilated=True,
                quality_annihilated=False,
                surviving_delta=CompoundPerturbation(
                    schema_delta=delta.schema_delta,
                    data_delta=DataDelta(operations=[]),
                    quality_delta=delta.quality_delta,
                ),
                annihilation_type=AnnihilationType.DATA,
                annihilation_reason=(
                    f"Filter predicate '{predicate}' contradicts data delta"
                ),
                annihilation_reasons=[
                    AnnihilationReason.FILTER_CONTRADICTS_DELTA
                ],
                strength=self._compute_filter_strength(delta),
            )

        return AnnihilationResult.no_annihilation(delta)

    def _check_union(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a UNION operator.

        Schema annihilation occurs when the schema delta is incompatible
        with one of the UNION branches.
        """
        if delta.schema_delta is None or len(delta.schema_delta.operations) == 0:
            return AnnihilationResult.no_annihilation(delta)

        if isinstance(config, dict) and "branch_schemas" in config:
            branch_schemas = config["branch_schemas"]
            affected_cols = _get_affected_columns_from_schema_delta(
                delta.schema_delta
            )

            all_branches_have = True
            for branch in branch_schemas:
                branch_cols = set()
                if hasattr(branch, "columns"):
                    branch_cols = {c.name for c in branch.columns}
                elif isinstance(branch, dict) and "columns" in branch:
                    branch_cols = {
                        c.get("name", "") for c in branch["columns"]
                    }

                if not affected_cols.issubset(branch_cols):
                    all_branches_have = False
                    break

            if not all_branches_have:
                return AnnihilationResult(
                    fully_annihilated=False,
                    schema_annihilated=True,
                    data_annihilated=False,
                    quality_annihilated=False,
                    surviving_delta=CompoundPerturbation(
                        schema_delta=SchemaDelta(operations=[]),
                        data_delta=delta.data_delta,
                        quality_delta=delta.quality_delta,
                    ),
                    annihilation_type=AnnihilationType.SCHEMA,
                    annihilation_reason="Schema delta incompatible with UNION branches",
                    annihilation_reasons=[
                        AnnihilationReason.UNION_SCHEMA_INCOMPATIBLE
                    ],
                    strength=self._compute_schema_strength(delta),
                )

        return AnnihilationResult.no_annihilation(delta)

    def _check_window(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a WINDOW function.

        Partial annihilation occurs when changes affect non-partition columns,
        as window functions recompute based on partitions.
        """
        partition_cols = _extract_window_partition_cols(config)
        if not partition_cols:
            return AnnihilationResult.no_annihilation(delta)

        data_affected = set()
        if delta.data_delta is not None:
            data_affected = _get_affected_columns_from_data_delta(delta.data_delta)

        non_partition = data_affected - partition_cols

        if non_partition and not (data_affected & partition_cols):
            return AnnihilationResult(
                fully_annihilated=False,
                schema_annihilated=False,
                data_annihilated=True,
                quality_annihilated=False,
                surviving_delta=CompoundPerturbation(
                    schema_delta=delta.schema_delta,
                    data_delta=DataDelta(operations=[]),
                    quality_delta=delta.quality_delta,
                ),
                annihilation_type=AnnihilationType.PARTIAL_DATA,
                annihilation_reason=(
                    f"Window partition columns {sorted(partition_cols)} "
                    f"unaffected; non-partition changes {sorted(non_partition)} absorbed"
                ),
                annihilation_reasons=[
                    AnnihilationReason.WINDOW_NON_PARTITION_COLUMN
                ],
                strength=0.5,
            )

        return AnnihilationResult.no_annihilation(delta)

    def _check_cte(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by a CTE (Common Table Expression).

        Full annihilation occurs when the CTE affected by the delta
        is not referenced in the main query.
        """
        used_ctes = _extract_cte_names_used(config)
        if not used_ctes:
            return AnnihilationResult.no_annihilation(delta)

        if isinstance(config, dict):
            affected_cte = config.get("affected_cte")
            if affected_cte and affected_cte.lower() not in {
                c.lower() for c in used_ctes
            }:
                return AnnihilationResult.full(
                    f"CTE '{affected_cte}' is not referenced in main query",
                    [AnnihilationReason.CTE_UNUSED],
                )

        return AnnihilationResult.no_annihilation(delta)

    def _check_setop(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by set operations (INTERSECT, EXCEPT).

        For EXCEPT, data insertions may be annihilated if they match
        the second operand.
        """
        op_type = ""
        if isinstance(config, dict):
            op_type = config.get("set_op_type", "").upper()
        elif hasattr(config, "operator"):
            op_type = str(config.operator).upper()

        if op_type == "EXCEPT":
            if (delta.data_delta is not None
                    and len(delta.data_delta.operations) > 0):

                insert_count = sum(
                    1 for op in delta.data_delta.operations
                    if isinstance(op, InsertOp)
                )

                if insert_count > 0:
                    return AnnihilationResult(
                        fully_annihilated=False,
                        data_annihilated=False,
                        schema_annihilated=False,
                        surviving_delta=delta,
                        annihilation_type=AnnihilationType.PARTIAL_DATA,
                        annihilation_reason=(
                            f"EXCEPT may remove {insert_count} inserted rows"
                        ),
                        annihilation_reasons=[
                            AnnihilationReason.SETOP_EXCEPT_REMOVES
                        ],
                        strength=0.3,
                    )

        return AnnihilationResult.no_annihilation(delta)

    def _check_distinct(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by DISTINCT.

        Insertions of duplicate rows are annihilated by DISTINCT.
        """
        if delta.data_delta is None or len(delta.data_delta.operations) == 0:
            return AnnihilationResult.no_annihilation(delta)

        return AnnihilationResult(
            fully_annihilated=False,
            data_annihilated=False,
            surviving_delta=delta,
            annihilation_type=AnnihilationType.PARTIAL_DATA,
            annihilation_reason="DISTINCT may absorb duplicate insertions",
            annihilation_reasons=[AnnihilationReason.DISTINCT_ABSORBS_DUPLICATE],
            strength=0.1,
        )

    def _check_limit(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by LIMIT.

        Rows beyond the LIMIT are truncated, so insertions that fall
        outside the limit are annihilated.
        """
        limit_value = None
        if isinstance(config, dict):
            limit_value = config.get("limit")
        if limit_value is None:
            return AnnihilationResult.no_annihilation(delta)

        if delta.data_delta is not None and len(delta.data_delta.operations) > 0:
            return AnnihilationResult(
                fully_annihilated=False,
                data_annihilated=False,
                surviving_delta=delta,
                annihilation_type=AnnihilationType.PARTIAL_DATA,
                annihilation_reason=f"LIMIT {limit_value} may truncate delta rows",
                annihilation_reasons=[AnnihilationReason.LIMIT_TRUNCATES],
                strength=0.2,
            )

        return AnnihilationResult.no_annihilation(delta)

    def _check_orderby(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """ORDER BY never annihilates deltas — it only reorders."""
        return AnnihilationResult.no_annihilation(delta)

    def _check_cast(
        self,
        delta: CompoundPerturbation,
        config: Any,
    ) -> AnnihilationResult:
        """Check annihilation by type casting.

        Type casts may absorb type-change deltas if the cast already
        converts to the target type.
        """
        if delta.schema_delta is None:
            return AnnihilationResult.no_annihilation(delta)

        cast_target_type = None
        cast_column = None
        if isinstance(config, dict):
            cast_target_type = config.get("target_type")
            cast_column = config.get("column")

        if cast_target_type and cast_column:
            for op in delta.schema_delta.operations:
                if (isinstance(op, ChangeType)
                        and op.column_name == cast_column
                        and str(op.new_type) == str(cast_target_type)):
                    return AnnihilationResult(
                        fully_annihilated=False,
                        schema_annihilated=True,
                        data_annihilated=False,
                        quality_annihilated=False,
                        surviving_delta=CompoundPerturbation(
                            schema_delta=SchemaDelta(operations=[]),
                            data_delta=delta.data_delta,
                            quality_delta=delta.quality_delta,
                        ),
                        annihilation_type=AnnihilationType.SCHEMA,
                        annihilation_reason=(
                            f"CAST already converts {cast_column} to {cast_target_type}"
                        ),
                        annihilation_reasons=[
                            AnnihilationReason.TYPE_CAST_ABSORBS
                        ],
                        strength=0.5,
                    )

        return AnnihilationResult.no_annihilation(delta)

    # ── Internal Helpers ──────────────────────────────────────────

    @staticmethod
    def _normalize_operator(operator: Any) -> str:
        """Normalize an operator to a string name."""
        if isinstance(operator, str):
            return operator.upper()
        if hasattr(operator, "value"):
            return str(operator.value).upper()
        if hasattr(operator, "name"):
            return str(operator.name).upper()
        return str(operator).upper()

    @staticmethod
    def _is_identity(delta: CompoundPerturbation) -> bool:
        s = delta.schema_delta is None or len(delta.schema_delta.operations) == 0
        d = delta.data_delta is None or len(delta.data_delta.operations) == 0
        q = delta.quality_delta is None or len(delta.quality_delta.operations) == 0
        return s and d and q

    @staticmethod
    def _delta_size(delta: CompoundPerturbation) -> int:
        size = 0
        if delta.schema_delta is not None:
            size += len(delta.schema_delta.operations)
        if delta.data_delta is not None:
            size += len(delta.data_delta.operations)
        if delta.quality_delta is not None:
            size += len(delta.quality_delta.operations)
        return size

    @staticmethod
    def _schema_empty(delta: CompoundPerturbation) -> bool:
        return delta.schema_delta is None or len(delta.schema_delta.operations) == 0

    @staticmethod
    def _data_empty(delta: CompoundPerturbation) -> bool:
        return delta.data_delta is None or len(delta.data_delta.operations) == 0

    @staticmethod
    def _quality_empty(delta: CompoundPerturbation) -> bool:
        return delta.quality_delta is None or len(delta.quality_delta.operations) == 0

    @staticmethod
    def _make_identity() -> CompoundPerturbation:
        return CompoundPerturbation(
            schema_delta=SchemaDelta(operations=[]),
            data_delta=DataDelta(operations=[]),
            quality_delta=QualityDelta(operations=[]),
        )

    def _check_predicate_contradiction(
        self,
        predicate: str,
        data_delta: DataDelta,
    ) -> bool:
        """Check if a filter predicate contradicts data operations.

        Uses basic pattern matching to detect obvious contradictions
        like "column = X" when the delta inserts "column = Y".
        """
        pred_lower = predicate.lower().strip()

        eq_match = re.match(r"(\w+)\s*=\s*['\"]?(\w+)['\"]?", pred_lower)
        if not eq_match:
            return False

        pred_col = eq_match.group(1)
        pred_val = eq_match.group(2)

        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                if hasattr(op, "tuple") and op.tuple is not None:
                    val = op.tuple.get(pred_col)
                    if val is not None and str(val).lower() != pred_val:
                        return True
            elif isinstance(op, UpdateOp):
                if hasattr(op, "changes") and isinstance(op.changes, dict):
                    new_val = op.changes.get(pred_col)
                    if new_val is not None and str(new_val).lower() != pred_val:
                        return True

        return False

    def _compute_filter_strength(self, delta: CompoundPerturbation) -> float:
        """Compute annihilation strength for filter operations."""
        total = self._delta_size(delta)
        data_size = 0
        if delta.data_delta is not None:
            data_size = len(delta.data_delta.operations)
        return data_size / total if total > 0 else 0.0

    def _compute_schema_strength(self, delta: CompoundPerturbation) -> float:
        """Compute annihilation strength for schema operations."""
        total = self._delta_size(delta)
        schema_size = 0
        if delta.schema_delta is not None:
            schema_size = len(delta.schema_delta.operations)
        return schema_size / total if total > 0 else 0.0

    def _split_by_columns(
        self,
        delta: CompoundPerturbation,
        surviving_cols: Set[str],
        annihilated_cols: Set[str],
    ) -> Tuple[CompoundPerturbation, CompoundPerturbation]:
        """Split a delta into surviving and annihilated parts by column set."""
        surviving_schema_ops = []
        annihilated_schema_ops = []

        if delta.schema_delta is not None:
            for op in delta.schema_delta.operations:
                op_cols = set()
                if isinstance(op, AddColumn):
                    op_cols.add(op.column_def.name)
                elif isinstance(op, DropColumn):
                    op_cols.add(op.column_name)
                elif isinstance(op, RenameColumn):
                    op_cols.add(op.old_name)
                    op_cols.add(op.new_name)
                elif isinstance(op, ChangeType):
                    op_cols.add(op.column_name)

                if op_cols & surviving_cols:
                    surviving_schema_ops.append(op)
                else:
                    annihilated_schema_ops.append(op)

        surviving = CompoundPerturbation(
            schema_delta=SchemaDelta(operations=surviving_schema_ops),
            data_delta=delta.data_delta,
            quality_delta=delta.quality_delta,
        )
        annihilated = CompoundPerturbation(
            schema_delta=SchemaDelta(operations=annihilated_schema_ops),
            data_delta=DataDelta(operations=[]),
            quality_delta=QualityDelta(operations=[]),
        )
        return surviving, annihilated

    def _filter_schema_ops_by_columns(
        self,
        delta: CompoundPerturbation,
        keep_columns: Set[str],
    ) -> CompoundPerturbation:
        """Filter schema operations to only those affecting keep_columns."""
        if delta.schema_delta is None:
            return delta

        kept_ops = []
        for op in delta.schema_delta.operations:
            op_cols = set()
            if isinstance(op, AddColumn):
                op_cols.add(op.column_def.name)
            elif isinstance(op, DropColumn):
                op_cols.add(op.column_name)
            elif isinstance(op, RenameColumn):
                op_cols.add(op.old_name)
            elif isinstance(op, ChangeType):
                op_cols.add(op.column_name)

            if op_cols & keep_columns:
                kept_ops.append(op)

        return CompoundPerturbation(
            schema_delta=SchemaDelta(operations=kept_ops),
            data_delta=delta.data_delta,
            quality_delta=delta.quality_delta,
        )

    def _compute_annihilated_portion(
        self,
        original: CompoundPerturbation,
        surviving: CompoundPerturbation,
    ) -> CompoundPerturbation:
        """Compute the annihilated portion as original minus surviving."""
        surviving_schema = set()
        if surviving.schema_delta is not None:
            surviving_schema = {repr(op) for op in surviving.schema_delta.operations}

        surviving_data = set()
        if surviving.data_delta is not None:
            surviving_data = {repr(op) for op in surviving.data_delta.operations}

        surviving_quality = set()
        if surviving.quality_delta is not None:
            surviving_quality = {repr(op) for op in surviving.quality_delta.operations}

        ann_schema = []
        if original.schema_delta is not None:
            for op in original.schema_delta.operations:
                if repr(op) not in surviving_schema:
                    ann_schema.append(op)

        ann_data = []
        if original.data_delta is not None:
            for op in original.data_delta.operations:
                if repr(op) not in surviving_data:
                    ann_data.append(op)

        ann_quality = []
        if original.quality_delta is not None:
            for op in original.quality_delta.operations:
                if repr(op) not in surviving_quality:
                    ann_quality.append(op)

        return CompoundPerturbation(
            schema_delta=SchemaDelta(operations=ann_schema),
            data_delta=DataDelta(operations=ann_data),
            quality_delta=QualityDelta(operations=ann_quality),
        )

    def _merge_deltas(
        self,
        deltas: List[CompoundPerturbation],
    ) -> CompoundPerturbation:
        """Merge multiple deltas by concatenating operations."""
        schema_ops: List[SchemaOperation] = []
        data_ops: List[DataOperation] = []
        quality_ops: List[QualityOperation] = []

        for d in deltas:
            if d.schema_delta is not None:
                schema_ops.extend(d.schema_delta.operations)
            if d.data_delta is not None:
                data_ops.extend(d.data_delta.operations)
            if d.quality_delta is not None:
                quality_ops.extend(d.quality_delta.operations)

        return CompoundPerturbation(
            schema_delta=SchemaDelta(operations=schema_ops),
            data_delta=DataDelta(operations=data_ops),
            quality_delta=QualityDelta(operations=quality_ops),
        )

    @staticmethod
    def _node_to_config(node: Any) -> Dict[str, Any]:
        """Convert a pipeline node to an operator config dict."""
        config: Dict[str, Any] = {}
        if hasattr(node, "query_text"):
            config["query_text"] = node.query_text
        if hasattr(node, "input_schema"):
            config["input_schema"] = node.input_schema
        if hasattr(node, "output_schema"):
            config["output_schema"] = node.output_schema
        if hasattr(node, "operator"):
            config["operator"] = node.operator
        return config


# =====================================================================
# Convenience Functions
# =====================================================================


def check_annihilation(
    operator: Any,
    delta: CompoundPerturbation,
    config: Any = None,
) -> AnnihilationResult:
    """Check if a delta is annihilated by an operator.

    Convenience wrapper around AnnihilationDetector.check_annihilation.
    """
    detector = AnnihilationDetector()
    return detector.check_annihilation(operator, delta, config)


def annihilation_strength(
    operator: Any,
    delta: CompoundPerturbation,
    config: Any = None,
) -> float:
    """Compute the annihilation strength for an operator and delta.

    Returns a float from 0.0 (no annihilation) to 1.0 (full).
    """
    detector = AnnihilationDetector()
    return detector.annihilation_strength(operator, delta, config)


def find_first_annihilation(
    graph: Any,
    source_node: str,
    delta: CompoundPerturbation,
) -> Optional[Tuple[str, AnnihilationResult]]:
    """Find the first node where full annihilation occurs.

    Traverses the graph from source and returns the first node where
    the delta is fully annihilated, or None if no annihilation occurs.
    """
    detector = AnnihilationDetector()
    points = detector.find_annihilation_points(graph, source_node, delta)

    for nid, result in points.items():
        if result.fully_annihilated:
            return nid, result

    return None


def compute_annihilation_profile(
    graph: Any,
    source_node: str,
    delta: CompoundPerturbation,
) -> Dict[str, float]:
    """Compute annihilation strength at every downstream node.

    Returns a dict mapping node_id to annihilation strength.
    """
    detector = AnnihilationDetector()
    points = detector.find_annihilation_points(graph, source_node, delta)

    profile: Dict[str, float] = {}
    for nid, result in points.items():
        profile[nid] = result.strength

    return profile
