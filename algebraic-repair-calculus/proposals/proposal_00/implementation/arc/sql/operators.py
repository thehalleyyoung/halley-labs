"""
SQL Operator Models
====================

Detailed operator models for push operator implementation.
Each operator model describes the semantics, schema requirements,
and algebraic properties of a SQL operator.

Operators: SELECT, JOIN, GROUP_BY, FILTER, UNION, WINDOW, CTE, SET_OP

Each operator has:
- input_schema_requirements
- output_schema_derivation
- determinism_check
- commutativity/associativity properties
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
# Enums
# ---------------------------------------------------------------------------

class SQLOperatorType(Enum):
    """Types of SQL operators in the query plan."""
    SELECT = "SELECT"
    JOIN = "JOIN"
    GROUP_BY = "GROUP_BY"
    FILTER = "FILTER"
    UNION = "UNION"
    WINDOW = "WINDOW"
    CTE = "CTE"
    SET_OP = "SET_OP"
    SUBQUERY = "SUBQUERY"
    ORDER_BY = "ORDER_BY"
    LIMIT = "LIMIT"
    DISTINCT = "DISTINCT"
    VALUES = "VALUES"
    LATERAL = "LATERAL"


class JoinKind(Enum):
    """Join types."""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    SEMI = "SEMI"
    ANTI = "ANTI"
    NATURAL = "NATURAL"
    LATERAL = "LATERAL"


class SetOperationType(Enum):
    """Set operation types."""
    UNION = "UNION"
    UNION_ALL = "UNION_ALL"
    INTERSECT = "INTERSECT"
    INTERSECT_ALL = "INTERSECT_ALL"
    EXCEPT = "EXCEPT"
    EXCEPT_ALL = "EXCEPT_ALL"


class AggregateFunctionType(Enum):
    """Aggregate function types."""
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
    PERCENTILE_CONT = "PERCENTILE_CONT"
    PERCENTILE_DISC = "PERCENTILE_DISC"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    FIRST_VALUE = "FIRST_VALUE"
    LAST_VALUE = "LAST_VALUE"
    NTH_VALUE = "NTH_VALUE"
    LAG = "LAG"
    LEAD = "LEAD"
    ROW_NUMBER = "ROW_NUMBER"
    RANK = "RANK"
    DENSE_RANK = "DENSE_RANK"
    NTILE = "NTILE"
    CUME_DIST = "CUME_DIST"
    PERCENT_RANK = "PERCENT_RANK"


class WindowFrameKind(Enum):
    """Window frame kinds."""
    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


class TransformationType(Enum):
    """Types of column transformations in lineage."""
    DIRECT = "DIRECT"
    COMPUTED = "COMPUTED"
    AGGREGATED = "AGGREGATED"
    WINDOWED = "WINDOWED"
    CONSTANT = "CONSTANT"
    SUBQUERY = "SUBQUERY"


# ---------------------------------------------------------------------------
# Column / Expression References
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnReference:
    """Reference to a column, possibly qualified."""
    name: str
    table: Optional[str] = None
    schema: Optional[str] = None
    alias: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        parts = []
        if self.schema:
            parts.append(self.schema)
        if self.table:
            parts.append(self.table)
        parts.append(self.name)
        return ".".join(parts)

    @property
    def output_name(self) -> str:
        return self.alias or self.name

    def matches(self, other: ColumnReference) -> bool:
        if self.name != other.name:
            return False
        if self.table and other.table and self.table != other.table:
            return False
        return True


@dataclass(frozen=True)
class ExpressionRef:
    """A SQL expression with its type."""
    sql: str
    output_alias: Optional[str] = None
    source_columns: Tuple[str, ...] = ()
    transformation: TransformationType = TransformationType.COMPUTED
    is_deterministic: bool = True

    @property
    def output_name(self) -> str:
        return self.output_alias or self.sql

    def references_column(self, col: str) -> bool:
        return col in self.source_columns


@dataclass(frozen=True)
class TableReference:
    """Reference to a source table."""
    name: str
    schema: Optional[str] = None
    alias: Optional[str] = None
    is_cte: bool = False
    is_subquery: bool = False
    subquery_sql: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.alias or self.name


@dataclass(frozen=True)
class JoinConditionSpec:
    """Specification of a join condition."""
    left: ColumnReference
    right: ColumnReference
    operator: str = "="
    is_natural: bool = False
    using_columns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OrderBySpec:
    """ORDER BY specification."""
    column: ColumnReference
    ascending: bool = True
    nulls_first: Optional[bool] = None


@dataclass(frozen=True)
class AggregateExprSpec:
    """Aggregate expression specification."""
    function: AggregateFunctionType
    input_columns: Tuple[ColumnReference, ...]
    output_alias: str
    distinct: bool = False
    filter_sql: Optional[str] = None
    order_by: Tuple[OrderBySpec, ...] = ()


@dataclass(frozen=True)
class WindowFrameSpec:
    """Window frame specification."""
    kind: WindowFrameKind = WindowFrameKind.RANGE
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    start_unbounded: bool = True
    end_current_row: bool = True


@dataclass(frozen=True)
class WindowFunctionSpec:
    """Window function specification."""
    function: AggregateFunctionType
    input_columns: Tuple[ColumnReference, ...]
    output_alias: str
    partition_by: Tuple[ColumnReference, ...] = ()
    order_by: Tuple[OrderBySpec, ...] = ()
    frame: Optional[WindowFrameSpec] = None


@dataclass(frozen=True)
class CTESpec:
    """CTE specification."""
    name: str
    columns: Tuple[str, ...] = ()
    query_sql: str = ""
    is_recursive: bool = False
    is_materialized: Optional[bool] = None


# ---------------------------------------------------------------------------
# Algebraic Properties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlgebraicProperties:
    """
    Algebraic properties of a SQL operator.
    Used by the push operator framework to determine valid transformations.
    """
    is_commutative: bool = False
    is_associative: bool = False
    is_idempotent: bool = False
    is_monotone: bool = True
    is_linear: bool = False
    preserves_multiplicities: bool = True
    preserves_order: bool = False
    is_deterministic: bool = True
    distributes_over_union: bool = False
    has_absorbing_element: bool = False


# ---------------------------------------------------------------------------
# Schema Derivation
# ---------------------------------------------------------------------------

@dataclass
class SchemaRequirement:
    """Describes what columns/types an operator requires from its input."""
    required_columns: Set[str] = field(default_factory=set)
    optional_columns: Set[str] = field(default_factory=set)
    required_types: Dict[str, str] = field(default_factory=dict)
    min_columns: int = 0
    max_columns: Optional[int] = None

    def is_satisfied_by(self, available_columns: Set[str]) -> bool:
        return self.required_columns.issubset(available_columns)

    def missing_columns(self, available_columns: Set[str]) -> Set[str]:
        return self.required_columns - available_columns


@dataclass
class SchemaDerivation:
    """Describes how an operator derives its output schema from inputs."""
    output_columns: List[ColumnReference] = field(default_factory=list)
    output_expressions: List[ExpressionRef] = field(default_factory=list)
    dropped_columns: Set[str] = field(default_factory=set)
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    added_columns: List[ColumnReference] = field(default_factory=list)
    preserves_all_columns: bool = False

    def output_column_names(self) -> List[str]:
        names = []
        for col in self.output_columns:
            names.append(col.output_name)
        for expr in self.output_expressions:
            names.append(expr.output_name)
        for col in self.added_columns:
            names.append(col.output_name)
        return names


# ---------------------------------------------------------------------------
# SQL Operator Base
# ---------------------------------------------------------------------------

class SQLOperator(ABC):
    """Base class for SQL operator models."""

    @property
    @abstractmethod
    def operator_type(self) -> SQLOperatorType:
        """Return the operator type."""

    @abstractmethod
    def input_schema_requirements(self) -> List[SchemaRequirement]:
        """Return the schema requirements for each input."""

    @abstractmethod
    def output_schema_derivation(self) -> SchemaDerivation:
        """Return how the output schema is derived."""

    @abstractmethod
    def algebraic_properties(self) -> AlgebraicProperties:
        """Return the algebraic properties of this operator."""

    @abstractmethod
    def determinism_check(self) -> bool:
        """Check if this operator is deterministic."""

    @abstractmethod
    def referenced_columns(self) -> Set[str]:
        """Return all columns referenced by this operator."""

    def __repr__(self) -> str:
        return f"{self.operator_type.value}Operator"


# ---------------------------------------------------------------------------
# Concrete Operator Models
# ---------------------------------------------------------------------------

@dataclass
class SelectOperator(SQLOperator):
    """SELECT projection operator."""
    columns: List[ColumnReference] = field(default_factory=list)
    expressions: List[ExpressionRef] = field(default_factory=list)
    distinct: bool = False
    star: bool = False

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.SELECT

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        required = set()
        for col in self.columns:
            required.add(col.name)
        for expr in self.expressions:
            required.update(expr.source_columns)
        return [SchemaRequirement(required_columns=required)]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(
            output_columns=self.columns,
            output_expressions=self.expressions,
            preserves_all_columns=self.star,
        )

    def algebraic_properties(self) -> AlgebraicProperties:
        return AlgebraicProperties(
            is_idempotent=True,
            is_monotone=True,
            is_linear=True,
            preserves_multiplicities=not self.distinct,
            is_deterministic=all(e.is_deterministic for e in self.expressions),
            distributes_over_union=True,
        )

    def determinism_check(self) -> bool:
        return all(e.is_deterministic for e in self.expressions)

    def referenced_columns(self) -> Set[str]:
        cols: Set[str] = set()
        for c in self.columns:
            cols.add(c.name)
        for e in self.expressions:
            cols.update(e.source_columns)
        return cols

    def output_column_names(self) -> List[str]:
        names = [c.output_name for c in self.columns]
        names.extend(e.output_name for e in self.expressions)
        return names

    def __repr__(self) -> str:
        cols = ", ".join(c.output_name for c in self.columns)
        return f"SELECT({cols})"


@dataclass
class JoinOperator(SQLOperator):
    """JOIN operator model."""
    join_kind: JoinKind = JoinKind.INNER
    conditions: List[JoinConditionSpec] = field(default_factory=list)
    using_columns: List[str] = field(default_factory=list)
    left_table: Optional[TableReference] = None
    right_table: Optional[TableReference] = None

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.JOIN

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        left_req = set()
        right_req = set()
        for cond in self.conditions:
            left_req.add(cond.left.name)
            right_req.add(cond.right.name)
        for col in self.using_columns:
            left_req.add(col)
            right_req.add(col)
        return [
            SchemaRequirement(required_columns=left_req),
            SchemaRequirement(required_columns=right_req),
        ]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(preserves_all_columns=True)

    def algebraic_properties(self) -> AlgebraicProperties:
        is_comm = self.join_kind == JoinKind.INNER
        is_assoc = self.join_kind in (JoinKind.INNER, JoinKind.CROSS)
        return AlgebraicProperties(
            is_commutative=is_comm,
            is_associative=is_assoc,
            is_monotone=self.join_kind != JoinKind.ANTI,
            preserves_multiplicities=True,
            is_deterministic=True,
            distributes_over_union=(self.join_kind == JoinKind.INNER),
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        cols: Set[str] = set()
        for cond in self.conditions:
            cols.add(cond.left.name)
            cols.add(cond.right.name)
        cols.update(self.using_columns)
        return cols

    def is_equi_join(self) -> bool:
        return all(c.operator == "=" for c in self.conditions)

    def join_column_pairs(self) -> List[Tuple[str, str]]:
        return [(c.left.name, c.right.name) for c in self.conditions]

    def __repr__(self) -> str:
        kind = self.join_kind.value
        conds = " AND ".join(
            f"{c.left.name}={c.right.name}" for c in self.conditions
        )
        return f"JOIN({kind}, ON {conds})"


@dataclass
class GroupByOperator(SQLOperator):
    """GROUP BY operator model."""
    group_keys: List[ColumnReference] = field(default_factory=list)
    aggregates: List[AggregateExprSpec] = field(default_factory=list)
    having_sql: Optional[str] = None

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.GROUP_BY

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        required = set()
        for k in self.group_keys:
            required.add(k.name)
        for a in self.aggregates:
            for col in a.input_columns:
                required.add(col.name)
        return [SchemaRequirement(required_columns=required)]

    def output_schema_derivation(self) -> SchemaDerivation:
        output_cols = list(self.group_keys)
        output_exprs = [
            ExpressionRef(
                sql=f"{a.function.value}({', '.join(c.name for c in a.input_columns)})",
                output_alias=a.output_alias,
                source_columns=tuple(c.name for c in a.input_columns),
                transformation=TransformationType.AGGREGATED,
            )
            for a in self.aggregates
        ]
        return SchemaDerivation(
            output_columns=output_cols,
            output_expressions=output_exprs,
        )

    def algebraic_properties(self) -> AlgebraicProperties:
        return AlgebraicProperties(
            is_idempotent=True,
            is_monotone=False,
            is_linear=False,
            preserves_multiplicities=False,
            is_deterministic=True,
            distributes_over_union=False,
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        cols: Set[str] = set()
        for k in self.group_keys:
            cols.add(k.name)
        for a in self.aggregates:
            for c in a.input_columns:
                cols.add(c.name)
        return cols

    def is_decomposable(self, agg: AggregateExprSpec) -> bool:
        """Check if an aggregate function is algebraically decomposable."""
        decomposable = {
            AggregateFunctionType.COUNT,
            AggregateFunctionType.SUM,
            AggregateFunctionType.MIN,
            AggregateFunctionType.MAX,
            AggregateFunctionType.BOOL_AND,
            AggregateFunctionType.BOOL_OR,
        }
        return agg.function in decomposable

    def can_incremental_maintain(self) -> bool:
        """Check if this GROUP BY can be incrementally maintained."""
        return all(self.is_decomposable(a) for a in self.aggregates)

    def __repr__(self) -> str:
        keys = ", ".join(k.name for k in self.group_keys)
        aggs = ", ".join(a.output_alias for a in self.aggregates)
        return f"GROUP_BY({keys}, aggs=[{aggs}])"


@dataclass
class FilterOperator(SQLOperator):
    """WHERE/HAVING filter operator."""
    predicate_sql: str = ""
    predicate_columns: Set[str] = field(default_factory=set)
    is_having: bool = False

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.FILTER

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        return [SchemaRequirement(required_columns=self.predicate_columns)]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(preserves_all_columns=True)

    def algebraic_properties(self) -> AlgebraicProperties:
        return AlgebraicProperties(
            is_idempotent=True,
            is_monotone=True,
            is_linear=True,
            preserves_multiplicities=True,
            is_deterministic=True,
            distributes_over_union=True,
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        return set(self.predicate_columns)

    def __repr__(self) -> str:
        return f"FILTER({self.predicate_sql})"


@dataclass
class UnionOperator(SQLOperator):
    """UNION operator model."""
    is_all: bool = True
    branch_count: int = 2

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.UNION

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        return [SchemaRequirement() for _ in range(self.branch_count)]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(preserves_all_columns=True)

    def algebraic_properties(self) -> AlgebraicProperties:
        return AlgebraicProperties(
            is_commutative=True,
            is_associative=True,
            is_idempotent=not self.is_all,
            is_monotone=True,
            is_linear=self.is_all,
            preserves_multiplicities=self.is_all,
            is_deterministic=True,
            distributes_over_union=True,
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        kind = "ALL" if self.is_all else ""
        return f"UNION {kind}({self.branch_count} branches)"


@dataclass
class WindowOperator(SQLOperator):
    """Window function operator model."""
    functions: List[WindowFunctionSpec] = field(default_factory=list)

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.WINDOW

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        required = set()
        for f in self.functions:
            for c in f.input_columns:
                required.add(c.name)
            for c in f.partition_by:
                required.add(c.name)
            for ob in f.order_by:
                required.add(ob.column.name)
        return [SchemaRequirement(required_columns=required)]

    def output_schema_derivation(self) -> SchemaDerivation:
        added = [
            ColumnReference(name=f.output_alias, alias=f.output_alias)
            for f in self.functions
        ]
        return SchemaDerivation(
            preserves_all_columns=True,
            added_columns=added,
        )

    def algebraic_properties(self) -> AlgebraicProperties:
        return AlgebraicProperties(
            is_monotone=False,
            is_linear=False,
            preserves_multiplicities=True,
            preserves_order=True,
            is_deterministic=True,
            distributes_over_union=False,
        )

    def determinism_check(self) -> bool:
        for f in self.functions:
            if f.function in (
                AggregateFunctionType.ROW_NUMBER,
                AggregateFunctionType.RANK,
                AggregateFunctionType.DENSE_RANK,
            ):
                if not f.order_by:
                    return False
        return True

    def referenced_columns(self) -> Set[str]:
        cols: Set[str] = set()
        for f in self.functions:
            for c in f.input_columns:
                cols.add(c.name)
            for c in f.partition_by:
                cols.add(c.name)
            for ob in f.order_by:
                cols.add(ob.column.name)
        return cols

    def output_column_names(self) -> List[str]:
        return [f.output_alias for f in self.functions]

    def __repr__(self) -> str:
        fns = ", ".join(f.output_alias for f in self.functions)
        return f"WINDOW([{fns}])"


@dataclass
class CTEOperator(SQLOperator):
    """CTE operator model."""
    ctes: List[CTESpec] = field(default_factory=list)

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.CTE

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        return [SchemaRequirement()]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(preserves_all_columns=True)

    def algebraic_properties(self) -> AlgebraicProperties:
        has_recursive = any(c.is_recursive for c in self.ctes)
        return AlgebraicProperties(
            is_monotone=not has_recursive,
            is_linear=not has_recursive,
            is_deterministic=True,
            distributes_over_union=False,
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        cols: Set[str] = set()
        for c in self.ctes:
            cols.update(c.columns)
        return cols

    def has_recursive_ctes(self) -> bool:
        return any(c.is_recursive for c in self.ctes)

    def cte_names(self) -> List[str]:
        return [c.name for c in self.ctes]

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self.ctes)
        rec = " (recursive)" if self.has_recursive_ctes() else ""
        return f"CTE([{names}]{rec})"


@dataclass
class SetOpOperator(SQLOperator):
    """Set operation operator (INTERSECT, EXCEPT)."""
    operation: SetOperationType = SetOperationType.INTERSECT
    branch_count: int = 2

    @property
    def operator_type(self) -> SQLOperatorType:
        return SQLOperatorType.SET_OP

    def input_schema_requirements(self) -> List[SchemaRequirement]:
        return [SchemaRequirement() for _ in range(self.branch_count)]

    def output_schema_derivation(self) -> SchemaDerivation:
        return SchemaDerivation(preserves_all_columns=True)

    def algebraic_properties(self) -> AlgebraicProperties:
        is_comm = self.operation in (
            SetOperationType.INTERSECT,
            SetOperationType.INTERSECT_ALL,
        )
        is_all = self.operation in (
            SetOperationType.INTERSECT_ALL,
            SetOperationType.EXCEPT_ALL,
        )
        return AlgebraicProperties(
            is_commutative=is_comm,
            is_associative=is_comm,
            is_idempotent=not is_all,
            is_monotone=False,
            is_linear=False,
            preserves_multiplicities=is_all,
            is_deterministic=True,
        )

    def determinism_check(self) -> bool:
        return True

    def referenced_columns(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        return f"SET_OP({self.operation.value}, {self.branch_count} branches)"


# ---------------------------------------------------------------------------
# Operator Factory
# ---------------------------------------------------------------------------

def create_operator(
    op_type: SQLOperatorType, **kwargs: Any
) -> SQLOperator:
    """Factory function to create SQL operator instances."""
    factories = {
        SQLOperatorType.SELECT: lambda: SelectOperator(**kwargs),
        SQLOperatorType.JOIN: lambda: JoinOperator(**kwargs),
        SQLOperatorType.GROUP_BY: lambda: GroupByOperator(**kwargs),
        SQLOperatorType.FILTER: lambda: FilterOperator(**kwargs),
        SQLOperatorType.UNION: lambda: UnionOperator(**kwargs),
        SQLOperatorType.WINDOW: lambda: WindowOperator(**kwargs),
        SQLOperatorType.CTE: lambda: CTEOperator(**kwargs),
        SQLOperatorType.SET_OP: lambda: SetOpOperator(**kwargs),
    }
    factory = factories.get(op_type)
    if factory is None:
        raise ValueError(f"Unknown operator type: {op_type}")
    return factory()


# ---------------------------------------------------------------------------
# Operator Compatibility
# ---------------------------------------------------------------------------

def check_schema_compatibility(
    operator: SQLOperator,
    input_columns: List[Set[str]],
) -> List[str]:
    """
    Check if the input schemas satisfy an operator's requirements.
    Returns a list of error messages (empty if compatible).
    """
    errors: List[str] = []
    requirements = operator.input_schema_requirements()

    if len(input_columns) < len(requirements):
        errors.append(
            f"Expected {len(requirements)} inputs, got {len(input_columns)}"
        )
        return errors

    for i, (req, available) in enumerate(zip(requirements, input_columns)):
        missing = req.missing_columns(available)
        if missing:
            errors.append(
                f"Input {i}: missing columns: {missing}"
            )
        if req.min_columns > 0 and len(available) < req.min_columns:
            errors.append(
                f"Input {i}: need at least {req.min_columns} columns, "
                f"got {len(available)}"
            )
        if req.max_columns is not None and len(available) > req.max_columns:
            errors.append(
                f"Input {i}: max {req.max_columns} columns allowed, "
                f"got {len(available)}"
            )

    return errors


def is_operator_deterministic(operator: SQLOperator) -> bool:
    """Check if an operator is deterministic."""
    return operator.determinism_check()


def operator_preserves_multiplicities(operator: SQLOperator) -> bool:
    """Check if an operator preserves tuple multiplicities."""
    return operator.algebraic_properties().preserves_multiplicities
