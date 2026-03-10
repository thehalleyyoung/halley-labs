"""
Push Operators
==============

Implements push operators push_f^X for each SQL operator f and delta sort X.

For each of the 8 SQL operator types, we define how each of the 3 delta sorts
(Schema, Data, Quality) propagates through the operator. This gives
8 × 3 = 24 push implementations.

SQL operators: SELECT, JOIN, GROUP_BY, FILTER, UNION, WINDOW, CTE, SET_OP
Delta sorts:   Schema (Δ_S), Data (Δ_D), Quality (Δ_Q)

Key algebraic property: push preserves composition.
  push_f(δ₁ ∘ δ₂) = push_f(δ₁) ∘ push_f(δ₂)  (when f is linear)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
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

from arc.algebra.schema_delta import (
    AddColumn,
    AddConstraint,
    ChangeType,
    ConstraintType,
    DropColumn,
    DropConstraint,
    RenameColumn,
    SQLType,
    SchemaDelta,
    SchemaOperation,
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
    DistributionSummary,
    QualityDelta,
    QualityImprovement,
    QualityOperation,
    QualityViolation,
    SeverityLevel,
    ViolationType,
    ConstraintType as QConstraintType,
)


# ---------------------------------------------------------------------------
# Operator Context: describes the SQL operator configuration
# ---------------------------------------------------------------------------

class JoinType(Enum):
    """SQL join types."""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    SEMI = "SEMI"
    ANTI = "ANTI"
    NATURAL = "NATURAL"


class SetOpType(Enum):
    """SQL set operation types."""
    INTERSECT = "INTERSECT"
    INTERSECT_ALL = "INTERSECT_ALL"
    EXCEPT = "EXCEPT"
    EXCEPT_ALL = "EXCEPT_ALL"


class AggregateFunction(Enum):
    """SQL aggregate functions."""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    ARRAY_AGG = "ARRAY_AGG"
    STRING_AGG = "STRING_AGG"
    BOOL_AND = "BOOL_AND"
    BOOL_OR = "BOOL_OR"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"
    PERCENTILE = "PERCENTILE"
    MEDIAN = "MEDIAN"


class WindowFrameType(Enum):
    """Window frame types."""
    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


@dataclass
class ColumnRef:
    """Reference to a column, possibly qualified by table."""
    name: str
    table: Optional[str] = None
    alias: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name

    @property
    def output_name(self) -> str:
        return self.alias or self.name


@dataclass
class AggregateSpec:
    """Specification of an aggregate function."""
    function: AggregateFunction
    input_columns: List[ColumnRef]
    output_alias: str
    distinct: bool = False
    filter_predicate: Optional[Callable[[TypedTuple], bool]] = None


@dataclass
class WindowSpec:
    """Specification of a window function."""
    function: AggregateFunction
    input_columns: List[ColumnRef]
    output_alias: str
    partition_by: List[ColumnRef] = field(default_factory=list)
    order_by: List[ColumnRef] = field(default_factory=list)
    frame_type: WindowFrameType = WindowFrameType.RANGE
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


@dataclass
class JoinCondition:
    """Join condition specification."""
    left_column: ColumnRef
    right_column: ColumnRef
    operator: str = "="


@dataclass
class OperatorContext:
    """
    Context describing a SQL operator's configuration.
    Provides all the information needed for push operator implementations.
    """
    operator_type: str

    # SELECT
    select_columns: List[ColumnRef] = field(default_factory=list)
    select_expressions: Dict[str, str] = field(default_factory=dict)

    # JOIN
    join_type: JoinType = JoinType.INNER
    join_conditions: List[JoinCondition] = field(default_factory=list)
    left_columns: List[str] = field(default_factory=list)
    right_columns: List[str] = field(default_factory=list)
    left_data: Optional[MultiSet] = None
    right_data: Optional[MultiSet] = None

    # GROUP BY
    group_by_columns: List[ColumnRef] = field(default_factory=list)
    aggregates: List[AggregateSpec] = field(default_factory=list)
    having_predicate: Optional[Callable[[TypedTuple], bool]] = None

    # FILTER
    filter_predicate: Optional[Callable[[TypedTuple], bool]] = None
    filter_columns: List[str] = field(default_factory=list)

    # UNION / SET OP
    union_all: bool = True
    set_op_type: Optional[SetOpType] = None
    other_branch_data: Optional[MultiSet] = None

    # WINDOW
    window_specs: List[WindowSpec] = field(default_factory=list)

    # CTE
    cte_name: Optional[str] = None
    cte_query_columns: List[str] = field(default_factory=list)
    is_recursive: bool = False
    cte_base_data: Optional[MultiSet] = None

    # Current data state (for incremental computation)
    current_data: Optional[MultiSet] = None

    def select_column_names(self) -> Set[str]:
        return {c.output_name for c in self.select_columns}

    def group_key_names(self) -> Set[str]:
        return {c.output_name for c in self.group_by_columns}


# ---------------------------------------------------------------------------
# Base Push Operator
# ---------------------------------------------------------------------------

class PushOperator(ABC):
    """
    Base class for push operators.

    Each push operator defines how three delta sorts propagate through
    a specific SQL operator type.
    """

    @abstractmethod
    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        """Push a schema delta through this operator."""

    @abstractmethod
    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        """Push a data delta through this operator."""

    @abstractmethod
    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        """Push a quality delta through this operator."""

    def push_all(
        self,
        ctx: OperatorContext,
        schema_delta: SchemaDelta,
        data_delta: DataDelta,
        quality_delta: QualityDelta,
    ) -> Tuple[SchemaDelta, DataDelta, QualityDelta]:
        """Push all three delta sorts through this operator."""
        return (
            self.push_schema(ctx, schema_delta),
            self.push_data(ctx, data_delta),
            self.push_quality(ctx, quality_delta),
        )


# ---------------------------------------------------------------------------
# SELECT Push
# ---------------------------------------------------------------------------

class SelectPush(PushOperator):
    """
    Push through SELECT operator.

    - Schema: Only propagate operations for selected columns
    - Data:   Project delta tuples to selected columns
    - Quality: Propagate constraints on selected columns
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        selected = ctx.select_column_names()
        expr_outputs = set(ctx.select_expressions.keys())
        all_output = selected | expr_outputs

        new_ops: List[SchemaOperation] = []
        for op in schema_delta.operations:
            if isinstance(op, AddColumn):
                if op.name in all_output:
                    new_ops.append(op)
            elif isinstance(op, DropColumn):
                if op.name in all_output:
                    new_ops.append(op)
            elif isinstance(op, RenameColumn):
                if op.old_name in all_output:
                    new_ops.append(op)
                    all_output.discard(op.old_name)
                    all_output.add(op.new_name)
            elif isinstance(op, ChangeType):
                if op.column_name in all_output:
                    new_ops.append(op)
            elif isinstance(op, AddConstraint):
                if any(c in all_output for c in op.columns):
                    new_ops.append(op)
            elif isinstance(op, DropConstraint):
                new_ops.append(op)

        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        selected = ctx.select_column_names()
        if not selected:
            return data_delta

        alias_map: Dict[str, str] = {}
        for col_ref in ctx.select_columns:
            if col_ref.alias and col_ref.alias != col_ref.name:
                alias_map[col_ref.name] = col_ref.alias

        def project_and_alias(t: TypedTuple) -> TypedTuple:
            vals: Dict[str, Any] = {}
            for col_ref in ctx.select_columns:
                src = col_ref.name
                dst = col_ref.output_name
                if src in t:
                    vals[dst] = t[src]
                elif dst in t:
                    vals[dst] = t[dst]
            for expr_name, expr_str in ctx.select_expressions.items():
                if expr_name in t:
                    vals[expr_name] = t[expr_name]
                else:
                    vals[expr_name] = _evaluate_simple_expression(expr_str, t)
            return TypedTuple(vals)

        return data_delta.map_tuples(project_and_alias)

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        selected = ctx.select_column_names()
        new_ops: List[QualityOperation] = []
        for op in quality_delta.operations:
            affected = op.affected_columns()
            if not affected or affected & selected:
                new_ops.append(op)
        return QualityDelta(new_ops)


# ---------------------------------------------------------------------------
# JOIN Push
# ---------------------------------------------------------------------------

class JoinPush(PushOperator):
    """
    Push through JOIN operator (INNER, LEFT, RIGHT, FULL, CROSS, SEMI, ANTI).

    - Schema: Handle both left/right schema evolution
    - Data:   Incremental join: join delta with opposite side
    - Quality: Propagate constraints from both sides
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        left_cols = set(ctx.left_columns)
        right_cols = set(ctx.right_columns)
        new_ops: List[SchemaOperation] = []

        for op in schema_delta.operations:
            affected = op.affected_columns()
            is_left = bool(affected & left_cols)
            is_right = bool(affected & right_cols)

            if isinstance(op, AddColumn):
                new_ops.append(op)
            elif isinstance(op, DropColumn):
                if op.name in left_cols or op.name in right_cols:
                    join_col_used = False
                    for jc in ctx.join_conditions:
                        if op.name in (jc.left_column.name, jc.right_column.name):
                            join_col_used = True
                            break
                    if not join_col_used:
                        new_ops.append(op)
                else:
                    new_ops.append(op)
            elif isinstance(op, RenameColumn):
                new_ops.append(op)
                for jc in ctx.join_conditions:
                    if jc.left_column.name == op.old_name:
                        jc.left_column = ColumnRef(name=op.new_name, table=jc.left_column.table)
                    if jc.right_column.name == op.old_name:
                        jc.right_column = ColumnRef(name=op.new_name, table=jc.right_column.table)
            elif isinstance(op, ChangeType):
                new_ops.append(op)
            elif isinstance(op, (AddConstraint, DropConstraint)):
                new_ops.append(op)

        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        result_ops: List[DataOperation] = []

        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                joined = self._incremental_join_insert(ctx, op.tuples)
                if not joined.is_empty():
                    result_ops.append(InsertOp(joined))
            elif isinstance(op, DeleteOp):
                unjoined = self._incremental_join_delete(ctx, op.tuples)
                if not unjoined.is_empty():
                    result_ops.append(DeleteOp(unjoined))
            elif isinstance(op, UpdateOp):
                old_joined = self._incremental_join_delete(ctx, op.old_tuples)
                new_joined = self._incremental_join_insert(ctx, op.new_tuples)
                if not old_joined.is_empty() or not new_joined.is_empty():
                    result_ops.append(UpdateOp(old_joined, new_joined))

        return DataDelta(result_ops)

    def _incremental_join_insert(
        self, ctx: OperatorContext, delta_tuples: MultiSet
    ) -> MultiSet:
        """Join inserted tuples with the opposite side."""
        opposite = self._get_opposite_data(ctx)
        if opposite is None:
            return delta_tuples

        result = MultiSet()
        for dt in delta_tuples.unique_tuples():
            dt_mult = delta_tuples.multiplicity(dt)
            for ot in opposite.unique_tuples():
                ot_mult = opposite.multiplicity(ot)
                if self._join_match(ctx, dt, ot):
                    merged = dt.merge(ot)
                    result.add(merged, dt_mult * ot_mult)

        if ctx.join_type == JoinType.LEFT:
            for dt in delta_tuples.unique_tuples():
                matched = False
                if opposite:
                    for ot in opposite.unique_tuples():
                        if self._join_match(ctx, dt, ot):
                            matched = True
                            break
                if not matched:
                    null_right = {c: None for c in ctx.right_columns if c not in dt.columns}
                    result.add(dt.merge(TypedTuple(null_right)), delta_tuples.multiplicity(dt))

        elif ctx.join_type == JoinType.RIGHT:
            pass

        elif ctx.join_type == JoinType.FULL:
            for dt in delta_tuples.unique_tuples():
                matched = False
                if opposite:
                    for ot in opposite.unique_tuples():
                        if self._join_match(ctx, dt, ot):
                            matched = True
                            break
                if not matched:
                    null_other = {}
                    for c in (ctx.right_columns if self._is_left_delta(ctx, dt) else ctx.left_columns):
                        if c not in dt.columns:
                            null_other[c] = None
                    result.add(dt.merge(TypedTuple(null_other)), delta_tuples.multiplicity(dt))

        elif ctx.join_type == JoinType.CROSS:
            if opposite is None:
                return delta_tuples
            result = MultiSet()
            for dt in delta_tuples.unique_tuples():
                for ot in opposite.unique_tuples():
                    merged = dt.merge(ot)
                    result.add(merged, delta_tuples.multiplicity(dt) * opposite.multiplicity(ot))

        elif ctx.join_type == JoinType.SEMI:
            result = MultiSet()
            if opposite:
                for dt in delta_tuples.unique_tuples():
                    for ot in opposite.unique_tuples():
                        if self._join_match(ctx, dt, ot):
                            result.add(dt, delta_tuples.multiplicity(dt))
                            break

        elif ctx.join_type == JoinType.ANTI:
            result = MultiSet()
            if opposite:
                for dt in delta_tuples.unique_tuples():
                    matched = False
                    for ot in opposite.unique_tuples():
                        if self._join_match(ctx, dt, ot):
                            matched = True
                            break
                    if not matched:
                        result.add(dt, delta_tuples.multiplicity(dt))
            else:
                result = delta_tuples.copy()

        return result

    def _incremental_join_delete(
        self, ctx: OperatorContext, delta_tuples: MultiSet
    ) -> MultiSet:
        """Compute the output deletions when tuples are deleted from one side."""
        opposite = self._get_opposite_data(ctx)
        if opposite is None:
            return delta_tuples

        result = MultiSet()
        for dt in delta_tuples.unique_tuples():
            dt_mult = delta_tuples.multiplicity(dt)
            for ot in opposite.unique_tuples():
                ot_mult = opposite.multiplicity(ot)
                if self._join_match(ctx, dt, ot):
                    merged = dt.merge(ot)
                    result.add(merged, dt_mult * ot_mult)

        if ctx.join_type in (JoinType.LEFT, JoinType.FULL):
            for dt in delta_tuples.unique_tuples():
                null_right = {c: None for c in ctx.right_columns if c not in dt.columns}
                result.add(dt.merge(TypedTuple(null_right)), delta_tuples.multiplicity(dt))

        return result

    def _get_opposite_data(self, ctx: OperatorContext) -> Optional[MultiSet]:
        if ctx.right_data is not None:
            return ctx.right_data
        if ctx.left_data is not None:
            return ctx.left_data
        return ctx.current_data

    def _is_left_delta(self, ctx: OperatorContext, t: TypedTuple) -> bool:
        left_set = set(ctx.left_columns)
        return bool(set(t.columns) & left_set)

    def _join_match(
        self, ctx: OperatorContext, t1: TypedTuple, t2: TypedTuple
    ) -> bool:
        """Check if two tuples satisfy the join conditions."""
        if ctx.join_type == JoinType.CROSS:
            return True

        for jc in ctx.join_conditions:
            left_val = t1.get(jc.left_column.name)
            if left_val is None:
                left_val = t2.get(jc.left_column.name)
            right_val = t2.get(jc.right_column.name)
            if right_val is None:
                right_val = t1.get(jc.right_column.name)

            if left_val is None or right_val is None:
                return False

            if jc.operator == "=":
                if left_val != right_val:
                    return False
            elif jc.operator == "!=":
                if left_val == right_val:
                    return False
            elif jc.operator == "<":
                if not (left_val < right_val):
                    return False
            elif jc.operator == ">":
                if not (left_val > right_val):
                    return False
            elif jc.operator == "<=":
                if not (left_val <= right_val):
                    return False
            elif jc.operator == ">=":
                if not (left_val >= right_val):
                    return False

        return True

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        all_cols = set(ctx.left_columns) | set(ctx.right_columns)
        new_ops: List[QualityOperation] = []
        for op in quality_delta.operations:
            affected = op.affected_columns()
            if not affected or affected & all_cols:
                new_ops.append(op)
        return QualityDelta(new_ops)


# ---------------------------------------------------------------------------
# GROUP BY Push
# ---------------------------------------------------------------------------

class GroupByPush(PushOperator):
    """
    Push through GROUP BY operator.

    - Schema: Group key columns always propagate; aggregate columns may change
    - Data:   Recompute affected groups (incremental aggregation)
    - Quality: Group-level quality constraint propagation
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        group_keys = ctx.group_key_names()
        agg_outputs = {a.output_alias for a in ctx.aggregates}
        all_output = group_keys | agg_outputs

        new_ops: List[SchemaOperation] = []
        for op in schema_delta.operations:
            if isinstance(op, AddColumn):
                if op.name in group_keys:
                    new_ops.append(op)
                elif op.name in agg_outputs:
                    new_ops.append(op)
            elif isinstance(op, DropColumn):
                if op.name in all_output:
                    new_ops.append(op)
            elif isinstance(op, RenameColumn):
                if op.old_name in all_output:
                    new_ops.append(op)
            elif isinstance(op, ChangeType):
                if op.column_name in all_output:
                    new_ops.append(op)
            elif isinstance(op, (AddConstraint, DropConstraint)):
                new_ops.append(op)

        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        if not ctx.group_by_columns and not ctx.aggregates:
            return data_delta

        current = ctx.current_data or MultiSet.empty()

        old_groups = self._compute_groups(ctx, current)

        new_data = data_delta.apply_to_data(current)
        new_groups = self._compute_groups(ctx, new_data)

        all_group_keys = set(old_groups.keys()) | set(new_groups.keys())

        old_agg = MultiSet()
        new_agg = MultiSet()

        for gk in all_group_keys:
            old_members = old_groups.get(gk, MultiSet.empty())
            new_members = new_groups.get(gk, MultiSet.empty())

            if old_members == new_members:
                continue

            if not old_members.is_empty():
                old_result = self._aggregate_group(ctx, gk, old_members)
                old_agg.add(old_result)

            if not new_members.is_empty():
                new_result = self._aggregate_group(ctx, gk, new_members)
                new_agg.add(new_result)

        result_ops: List[DataOperation] = []
        if not old_agg.is_empty() or not new_agg.is_empty():
            result_ops.append(UpdateOp(old_agg, new_agg))

        return DataDelta(result_ops)

    def _compute_groups(
        self, ctx: OperatorContext, data: MultiSet
    ) -> Dict[tuple, MultiSet]:
        """Partition data into groups by key columns."""
        groups: Dict[tuple, MultiSet] = {}
        key_cols = [c.output_name for c in ctx.group_by_columns]

        for t in data.unique_tuples():
            key_vals = tuple(t.get(k) for k in key_cols)
            if key_vals not in groups:
                groups[key_vals] = MultiSet()
            groups[key_vals].add(t, data.multiplicity(t))

        return groups

    def _aggregate_group(
        self, ctx: OperatorContext, group_key: tuple, members: MultiSet
    ) -> TypedTuple:
        """Compute aggregate values for a single group."""
        key_cols = [c.output_name for c in ctx.group_by_columns]
        result: Dict[str, Any] = {}

        for col_name, val in zip(key_cols, group_key):
            result[col_name] = val

        for agg_spec in ctx.aggregates:
            result[agg_spec.output_alias] = self._compute_aggregate(
                agg_spec, members
            )

        return TypedTuple(result)

    def _compute_aggregate(
        self, agg: AggregateSpec, members: MultiSet
    ) -> Any:
        """Compute a single aggregate value."""
        col_name = agg.input_columns[0].name if agg.input_columns else None

        values: List[Any] = []
        for t in members:
            if agg.filter_predicate and not agg.filter_predicate(t):
                continue
            if col_name:
                val = t.get(col_name)
                if val is not None:
                    values.append(val)
            else:
                values.append(1)

        if agg.function == AggregateFunction.COUNT:
            if col_name:
                return len(values)
            return members.cardinality()

        if agg.function == AggregateFunction.COUNT_DISTINCT:
            return len(set(values))

        if agg.function == AggregateFunction.SUM:
            if not values:
                return None
            try:
                return sum(values)
            except TypeError:
                return None

        if agg.function == AggregateFunction.AVG:
            if not values:
                return None
            try:
                return sum(values) / len(values)
            except (TypeError, ZeroDivisionError):
                return None

        if agg.function == AggregateFunction.MIN:
            if not values:
                return None
            try:
                return min(values)
            except TypeError:
                return None

        if agg.function == AggregateFunction.MAX:
            if not values:
                return None
            try:
                return max(values)
            except TypeError:
                return None

        if agg.function == AggregateFunction.ARRAY_AGG:
            return values

        if agg.function == AggregateFunction.STRING_AGG:
            return ",".join(str(v) for v in values)

        if agg.function == AggregateFunction.BOOL_AND:
            return all(values) if values else None

        if agg.function == AggregateFunction.BOOL_OR:
            return any(values) if values else None

        if agg.function == AggregateFunction.STDDEV:
            if len(values) < 2:
                return None
            try:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                return variance ** 0.5
            except (TypeError, ValueError):
                return None

        if agg.function == AggregateFunction.VARIANCE:
            if len(values) < 2:
                return None
            try:
                mean = sum(values) / len(values)
                return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            except (TypeError, ValueError):
                return None

        if agg.function == AggregateFunction.MEDIAN:
            if not values:
                return None
            try:
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                if n % 2 == 1:
                    return sorted_vals[n // 2]
                return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            except (TypeError, ValueError):
                return None

        return None

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        group_keys = ctx.group_key_names()
        agg_cols = {a.output_alias for a in ctx.aggregates}
        all_cols = group_keys | agg_cols

        new_ops: List[QualityOperation] = []
        for op in quality_delta.operations:
            affected = op.affected_columns()
            if not affected:
                new_ops.append(op)
                continue
            input_cols: Set[str] = set()
            for a in ctx.aggregates:
                for ic in a.input_columns:
                    input_cols.add(ic.name)
            if affected & (all_cols | input_cols | group_keys):
                new_ops.append(op)

        return QualityDelta(new_ops)


# ---------------------------------------------------------------------------
# FILTER Push
# ---------------------------------------------------------------------------

class FilterPush(PushOperator):
    """
    Push through FILTER (WHERE) operator.

    - Schema: All schema changes propagate through filters
    - Data:   Apply filter predicate to delta tuples
    - Quality: Filter may absorb quality violations
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        return schema_delta

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        pred = ctx.filter_predicate
        if pred is None:
            return data_delta

        new_ops: List[DataOperation] = []
        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                filtered = op.tuples.filter(pred)
                if not filtered.is_empty():
                    new_ops.append(InsertOp(filtered))
            elif isinstance(op, DeleteOp):
                filtered = op.tuples.filter(pred)
                if not filtered.is_empty():
                    new_ops.append(DeleteOp(filtered))
            elif isinstance(op, UpdateOp):
                old_pass = op.old_tuples.filter(pred)
                new_pass = op.new_tuples.filter(pred)
                old_fail_new_pass = MultiSet()
                old_pass_new_fail = MultiSet()

                old_list = sorted(
                    op.old_tuples.unique_tuples(), key=lambda t: t._sortable_key()
                )
                new_list = sorted(
                    op.new_tuples.unique_tuples(), key=lambda t: t._sortable_key()
                )

                for old_t, new_t in zip(old_list, new_list):
                    old_passes = pred(old_t)
                    new_passes = pred(new_t)
                    mult = min(
                        op.old_tuples.multiplicity(old_t),
                        op.new_tuples.multiplicity(new_t),
                    )
                    if not old_passes and new_passes:
                        old_fail_new_pass.add(new_t, mult)
                    elif old_passes and not new_passes:
                        old_pass_new_fail.add(old_t, mult)

                if not old_pass.is_empty() or not new_pass.is_empty():
                    new_ops.append(UpdateOp(old_pass, new_pass))
                if not old_fail_new_pass.is_empty():
                    new_ops.append(InsertOp(old_fail_new_pass))
                if not old_pass_new_fail.is_empty():
                    new_ops.append(DeleteOp(old_pass_new_fail))

        return DataDelta(new_ops)

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        filter_cols = set(ctx.filter_columns)
        if not filter_cols:
            return quality_delta

        new_ops: List[QualityOperation] = []
        for op in quality_delta.operations:
            affected = op.affected_columns()
            if isinstance(op, QualityViolation):
                if not affected or affected & filter_cols:
                    new_ops.append(op)
                elif affected - filter_cols:
                    new_ops.append(op)
            else:
                new_ops.append(op)

        return QualityDelta(new_ops)


# ---------------------------------------------------------------------------
# UNION Push
# ---------------------------------------------------------------------------

class UnionPush(PushOperator):
    """
    Push through UNION [ALL] operator.

    - Schema: Schema must be compatible; handle type widening
    - Data:   Delta propagates to union output
    - Quality: Union of quality violations
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        new_ops: List[SchemaOperation] = []
        for op in schema_delta.operations:
            if isinstance(op, ChangeType):
                from arc.algebra.schema_delta import widest_type
                new_ops.append(op)
            elif isinstance(op, AddColumn):
                new_ops.append(op)
            elif isinstance(op, DropColumn):
                new_ops.append(op)
            elif isinstance(op, RenameColumn):
                new_ops.append(op)
            else:
                new_ops.append(op)
        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        if ctx.union_all:
            return data_delta

        result_ops: List[DataOperation] = []
        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                other = ctx.other_branch_data or MultiSet.empty()
                current = ctx.current_data or MultiSet.empty()
                existing = current.union(other)
                new_tuples = MultiSet()
                for t in op.tuples.unique_tuples():
                    if not existing.contains(t):
                        new_tuples.add(t, 1)
                if not new_tuples.is_empty():
                    result_ops.append(InsertOp(new_tuples))
            elif isinstance(op, DeleteOp):
                other = ctx.other_branch_data or MultiSet.empty()
                actually_deleted = MultiSet()
                for t in op.tuples.unique_tuples():
                    if not other.contains(t):
                        actually_deleted.add(t, 1)
                if not actually_deleted.is_empty():
                    result_ops.append(DeleteOp(actually_deleted))
            elif isinstance(op, UpdateOp):
                result_ops.append(op)

        return DataDelta(result_ops)

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        return quality_delta


# ---------------------------------------------------------------------------
# WINDOW Push
# ---------------------------------------------------------------------------

class WindowPush(PushOperator):
    """
    Push through WINDOW function operator.

    - Schema: Window adds output columns
    - Data:   Recompute window functions for affected partitions
    - Quality: Partition-level quality propagation
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        window_output_cols = {ws.output_alias for ws in ctx.window_specs}
        new_ops: List[SchemaOperation] = list(schema_delta.operations)

        for ws in ctx.window_specs:
            if ws.output_alias not in schema_delta.affected_columns():
                pass

        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        if not ctx.window_specs:
            return data_delta

        current = ctx.current_data or MultiSet.empty()
        new_data = data_delta.apply_to_data(current)

        affected_partitions = self._find_affected_partitions(ctx, data_delta)

        old_windowed = self._compute_window_results(ctx, current, affected_partitions)
        new_windowed = self._compute_window_results(ctx, new_data, affected_partitions)

        if old_windowed == new_windowed:
            return data_delta

        result_ops: List[DataOperation] = []
        deleted = old_windowed.difference(new_windowed)
        inserted = new_windowed.difference(old_windowed)
        if not deleted.is_empty():
            result_ops.append(DeleteOp(deleted))
        if not inserted.is_empty():
            result_ops.append(InsertOp(inserted))

        return DataDelta(result_ops)

    def _find_affected_partitions(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> Set[tuple]:
        """Find which partition keys are affected by the data delta."""
        affected: Set[tuple] = set()
        for ws in ctx.window_specs:
            part_cols = [c.output_name for c in ws.partition_by]
            for op in data_delta.operations:
                if isinstance(op, InsertOp):
                    for t in op.tuples.unique_tuples():
                        key = tuple(t.get(c) for c in part_cols)
                        affected.add(key)
                elif isinstance(op, DeleteOp):
                    for t in op.tuples.unique_tuples():
                        key = tuple(t.get(c) for c in part_cols)
                        affected.add(key)
                elif isinstance(op, UpdateOp):
                    for t in op.old_tuples.unique_tuples():
                        key = tuple(t.get(c) for c in part_cols)
                        affected.add(key)
                    for t in op.new_tuples.unique_tuples():
                        key = tuple(t.get(c) for c in part_cols)
                        affected.add(key)
        return affected

    def _compute_window_results(
        self, ctx: OperatorContext, data: MultiSet,
        affected_partitions: Set[tuple],
    ) -> MultiSet:
        """Compute window function results for affected partitions."""
        result = MultiSet()

        for ws in ctx.window_specs:
            part_cols = [c.output_name for c in ws.partition_by]
            order_cols = [c.output_name for c in ws.order_by]

            partitions: Dict[tuple, List[TypedTuple]] = defaultdict(list)
            for t in data.unique_tuples():
                key = tuple(t.get(c) for c in part_cols)
                if key in affected_partitions:
                    for _ in range(data.multiplicity(t)):
                        partitions[key].append(t)

            for part_key, members in partitions.items():
                if order_cols:
                    try:
                        members.sort(key=lambda t: tuple(t.get(c, "") for c in order_cols))
                    except TypeError:
                        pass

                window_values = self._compute_window_for_partition(ws, members)
                for t, wval in zip(members, window_values):
                    new_t = t.extend(ws.output_alias, wval)
                    result.add(new_t)

        if result.is_empty():
            return data

        return result

    def _compute_window_for_partition(
        self, ws: WindowSpec, members: List[TypedTuple]
    ) -> List[Any]:
        """Compute window function values for a single partition."""
        col_name = ws.input_columns[0].name if ws.input_columns else None
        n = len(members)
        values = []
        for t in members:
            if col_name:
                values.append(t.get(col_name))
            else:
                values.append(1)

        results: List[Any] = []

        if ws.function == AggregateFunction.COUNT:
            for i in range(n):
                results.append(i + 1)

        elif ws.function == AggregateFunction.SUM:
            running = 0
            for v in values:
                if v is not None:
                    try:
                        running += v
                    except TypeError:
                        pass
                results.append(running)

        elif ws.function == AggregateFunction.AVG:
            running_sum = 0
            running_count = 0
            for v in values:
                if v is not None:
                    try:
                        running_sum += v
                        running_count += 1
                    except TypeError:
                        pass
                results.append(
                    running_sum / running_count if running_count > 0 else None
                )

        elif ws.function == AggregateFunction.MIN:
            running_min = None
            for v in values:
                if v is not None:
                    if running_min is None:
                        running_min = v
                    else:
                        try:
                            running_min = min(running_min, v)
                        except TypeError:
                            pass
                results.append(running_min)

        elif ws.function == AggregateFunction.MAX:
            running_max = None
            for v in values:
                if v is not None:
                    if running_max is None:
                        running_max = v
                    else:
                        try:
                            running_max = max(running_max, v)
                        except TypeError:
                            pass
                results.append(running_max)

        else:
            for i in range(n):
                results.append(i + 1)

        return results

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        partition_cols: Set[str] = set()
        for ws in ctx.window_specs:
            for c in ws.partition_by:
                partition_cols.add(c.output_name)
            for c in ws.input_columns:
                partition_cols.add(c.name)

        new_ops: List[QualityOperation] = []
        for op in quality_delta.operations:
            affected = op.affected_columns()
            if not affected or affected & partition_cols:
                new_ops.append(op)
            else:
                new_ops.append(op)

        return QualityDelta(new_ops)


# ---------------------------------------------------------------------------
# CTE Push
# ---------------------------------------------------------------------------

class CTEPush(PushOperator):
    """
    Push through Common Table Expression (CTE) operator.

    - Schema: CTE output schema evolution
    - Data:   Incremental CTE evaluation
    - Quality: CTE-level quality propagation
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        cte_cols = set(ctx.cte_query_columns)
        new_ops: List[SchemaOperation] = []
        for op in schema_delta.operations:
            affected = op.affected_columns()
            if not affected or affected & cte_cols:
                new_ops.append(op)
        return SchemaDelta(new_ops)

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        if ctx.is_recursive:
            return self._push_recursive_cte(ctx, data_delta)
        return data_delta

    def _push_recursive_cte(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        """
        Handle recursive CTE delta propagation.

        For recursive CTEs, we iterate the recursive step with the delta
        until a fixed point is reached.
        """
        current = ctx.cte_base_data or MultiSet.empty()
        new_data = data_delta.apply_to_data(current)

        max_iterations = 100
        changed = True
        iteration = 0
        accumulated = new_data.copy()

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            new_tuples = self._apply_recursive_step(ctx, accumulated)
            if not new_tuples.is_empty():
                added = new_tuples.difference(accumulated)
                if not added.is_empty():
                    accumulated = accumulated.sum(added)
                    changed = True

        return DataDelta.from_diff(current, accumulated)

    def _apply_recursive_step(
        self, ctx: OperatorContext, data: MultiSet
    ) -> MultiSet:
        """Apply one step of recursive CTE evaluation."""
        return data

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        if ctx.is_recursive:
            extra_ops: List[QualityOperation] = []
            if quality_delta.has_violations():
                for v in quality_delta.get_violations():
                    extra_ops.append(
                        QualityViolation(
                            constraint_id=f"recursive_{v.constraint_id}",
                            severity=SeverityLevel(
                                min(v.severity.value + 1, SeverityLevel.FATAL.value)
                            ),
                            affected_tuples=v.affected_tuples,
                            violation_type=v.violation_type,
                            columns=v.columns,
                            message=f"Recursive CTE amplifies: {v.message or ''}",
                        )
                    )
            if extra_ops:
                return QualityDelta(list(quality_delta.operations) + extra_ops)
        return quality_delta


# ---------------------------------------------------------------------------
# SET OP Push (INTERSECT, EXCEPT)
# ---------------------------------------------------------------------------

class SetOpPush(PushOperator):
    """
    Push through set operations (INTERSECT, EXCEPT).

    - Schema: Schema evolution for set operations
    - Data:   Set operation semantics on deltas
    - Quality: Quality propagation for set operations
    """

    def push_schema(
        self, ctx: OperatorContext, schema_delta: SchemaDelta
    ) -> SchemaDelta:
        return schema_delta

    def push_data(
        self, ctx: OperatorContext, data_delta: DataDelta
    ) -> DataDelta:
        set_op = ctx.set_op_type
        if set_op is None:
            return data_delta

        other = ctx.other_branch_data or MultiSet.empty()
        current = ctx.current_data or MultiSet.empty()

        if set_op == SetOpType.INTERSECT:
            return self._push_intersect(data_delta, current, other, distinct=True)
        elif set_op == SetOpType.INTERSECT_ALL:
            return self._push_intersect(data_delta, current, other, distinct=False)
        elif set_op == SetOpType.EXCEPT:
            return self._push_except(data_delta, current, other, distinct=True)
        elif set_op == SetOpType.EXCEPT_ALL:
            return self._push_except(data_delta, current, other, distinct=False)

        return data_delta

    def _push_intersect(
        self,
        data_delta: DataDelta,
        current: MultiSet,
        other: MultiSet,
        distinct: bool,
    ) -> DataDelta:
        """
        INTERSECT: Output includes only tuples in both sides.
        Delta inserts only count if the tuple is also in the other branch.
        """
        result_ops: List[DataOperation] = []

        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                kept = MultiSet()
                for t in op.tuples.unique_tuples():
                    if other.contains(t):
                        count = 1 if distinct else min(
                            op.tuples.multiplicity(t),
                            other.multiplicity(t),
                        )
                        kept.add(t, count)
                if not kept.is_empty():
                    result_ops.append(InsertOp(kept))

            elif isinstance(op, DeleteOp):
                removed = MultiSet()
                for t in op.tuples.unique_tuples():
                    if other.contains(t):
                        count = 1 if distinct else min(
                            op.tuples.multiplicity(t),
                            other.multiplicity(t),
                        )
                        removed.add(t, count)
                if not removed.is_empty():
                    result_ops.append(DeleteOp(removed))

            elif isinstance(op, UpdateOp):
                result_ops.append(op)

        return DataDelta(result_ops)

    def _push_except(
        self,
        data_delta: DataDelta,
        current: MultiSet,
        other: MultiSet,
        distinct: bool,
    ) -> DataDelta:
        """
        EXCEPT: Output includes tuples in left but not in right.
        """
        result_ops: List[DataOperation] = []

        for op in data_delta.operations:
            if isinstance(op, InsertOp):
                kept = MultiSet()
                for t in op.tuples.unique_tuples():
                    if not other.contains(t):
                        count = 1 if distinct else op.tuples.multiplicity(t)
                        kept.add(t, count)
                if not kept.is_empty():
                    result_ops.append(InsertOp(kept))

            elif isinstance(op, DeleteOp):
                removed = MultiSet()
                for t in op.tuples.unique_tuples():
                    if not other.contains(t):
                        count = 1 if distinct else op.tuples.multiplicity(t)
                        removed.add(t, count)
                if not removed.is_empty():
                    result_ops.append(DeleteOp(removed))

            elif isinstance(op, UpdateOp):
                result_ops.append(op)

        return DataDelta(result_ops)

    def push_quality(
        self, ctx: OperatorContext, quality_delta: QualityDelta
    ) -> QualityDelta:
        return quality_delta


# ---------------------------------------------------------------------------
# Expression Evaluation Helper
# ---------------------------------------------------------------------------

def _evaluate_simple_expression(expr: str, t: TypedTuple) -> Any:
    """
    Evaluate a simple SQL expression against a tuple.
    Handles column references, basic arithmetic, and CASE WHEN.
    """
    expr = expr.strip()

    if expr in t:
        return t[expr]

    if expr.isdigit():
        return int(expr)
    try:
        return float(expr)
    except ValueError:
        pass

    if expr.startswith("'") and expr.endswith("'"):
        return expr[1:-1]

    if expr.upper() == "NULL":
        return None

    for op_str in [" + ", " - ", " * ", " / ", " || "]:
        if op_str in expr:
            parts = expr.split(op_str, 1)
            left = _evaluate_simple_expression(parts[0], t)
            right = _evaluate_simple_expression(parts[1], t)
            try:
                if op_str == " + ":
                    return left + right
                elif op_str == " - ":
                    return left - right
                elif op_str == " * ":
                    return left * right
                elif op_str == " / ":
                    return left / right if right != 0 else None
                elif op_str == " || ":
                    return str(left) + str(right)
            except (TypeError, ValueError):
                return None

    return None


# ---------------------------------------------------------------------------
# Push Operator Registry
# ---------------------------------------------------------------------------

_PUSH_REGISTRY: Dict[str, PushOperator] = {
    "SELECT": SelectPush(),
    "JOIN": JoinPush(),
    "GROUP_BY": GroupByPush(),
    "FILTER": FilterPush(),
    "UNION": UnionPush(),
    "WINDOW": WindowPush(),
    "CTE": CTEPush(),
    "SET_OP": SetOpPush(),
}


def get_push_operator(operator_type: str) -> Optional[PushOperator]:
    """Get the push operator implementation for a given SQL operator type."""
    return _PUSH_REGISTRY.get(operator_type.upper())


def push_schema_delta(
    operator_type: str,
    ctx: OperatorContext,
    schema_delta: SchemaDelta,
) -> SchemaDelta:
    """Push a schema delta through a SQL operator."""
    push = get_push_operator(operator_type)
    if push is None:
        return schema_delta
    return push.push_schema(ctx, schema_delta)


def push_data_delta(
    operator_type: str,
    ctx: OperatorContext,
    data_delta: DataDelta,
) -> DataDelta:
    """Push a data delta through a SQL operator."""
    push = get_push_operator(operator_type)
    if push is None:
        return data_delta
    return push.push_data(ctx, data_delta)


def push_quality_delta(
    operator_type: str,
    ctx: OperatorContext,
    quality_delta: QualityDelta,
) -> QualityDelta:
    """Push a quality delta through a SQL operator."""
    push = get_push_operator(operator_type)
    if push is None:
        return quality_delta
    return push.push_quality(ctx, quality_delta)


def push_all_deltas(
    operator_type: str,
    ctx: OperatorContext,
    schema_delta: SchemaDelta,
    data_delta: DataDelta,
    quality_delta: QualityDelta,
) -> Tuple[SchemaDelta, DataDelta, QualityDelta]:
    """Push all three delta sorts through a SQL operator."""
    push = get_push_operator(operator_type)
    if push is None:
        return schema_delta, data_delta, quality_delta
    return push.push_all(ctx, schema_delta, data_delta, quality_delta)


def register_push_operator(operator_type: str, push: PushOperator) -> None:
    """Register a custom push operator."""
    _PUSH_REGISTRY[operator_type.upper()] = push


def supported_operators() -> List[str]:
    """Return list of supported SQL operator types."""
    return sorted(_PUSH_REGISTRY.keys())
