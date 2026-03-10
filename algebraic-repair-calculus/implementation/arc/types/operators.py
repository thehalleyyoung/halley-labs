"""
SQL operator types, join kinds, aggregate functions, and operator metadata.

Provides the algebraic description of every SQL and ETL operator that the
pipeline graph can represent, together with properties needed by the
delta-algebra engine (commutativity, associativity, determinism) and
operator signatures that relate input schemas to output schemas.
"""

from __future__ import annotations

import enum
from typing import Any, Sequence

import attr
from attr import validators as v

from arc.types.base import Schema, ParameterisedType, SQLType


# =====================================================================
# Core SQL operator kinds
# =====================================================================

class SQLOperator(enum.Enum):
    """High-level SQL / relational operator classification."""

    # Relational core
    SELECT = "SELECT"
    PROJECT = "PROJECT"
    FILTER = "FILTER"
    JOIN = "JOIN"
    GROUP_BY = "GROUP_BY"
    ORDER_BY = "ORDER_BY"
    LIMIT = "LIMIT"

    # Set operations
    UNION = "UNION"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"
    SET_OP = "SET_OP"

    # Advanced SQL
    WINDOW = "WINDOW"
    CTE = "CTE"
    SUBQUERY = "SUBQUERY"
    LATERAL = "LATERAL"
    PIVOT = "PIVOT"
    UNPIVOT = "UNPIVOT"

    # DML / mutation
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    MERGE = "MERGE"
    UPSERT = "UPSERT"

    # DDL (schema evolution)
    CREATE_TABLE = "CREATE_TABLE"
    ALTER_TABLE = "ALTER_TABLE"
    DROP_TABLE = "DROP_TABLE"
    CREATE_VIEW = "CREATE_VIEW"
    CREATE_INDEX = "CREATE_INDEX"

    # ETL-specific operators
    SOURCE = "SOURCE"
    SINK = "SINK"
    TRANSFORM = "TRANSFORM"
    RENAME = "RENAME"
    CAST = "CAST"
    FILL = "FILL"
    DEDUP = "DEDUP"
    EXPLODE = "EXPLODE"

    # Python ETL operators
    PANDAS_OP = "PANDAS_OP"
    PYSPARK_OP = "PYSPARK_OP"
    DBT_MODEL = "DBT_MODEL"

    # External / opaque
    EXTERNAL_CALL = "EXTERNAL_CALL"
    UDF = "UDF"
    OPAQUE = "OPAQUE"


# =====================================================================
# Join types
# =====================================================================

class JoinType(enum.Enum):
    """All supported join variants."""

    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    SEMI = "SEMI"
    ANTI = "ANTI"
    NATURAL = "NATURAL"
    LATERAL = "LATERAL"
    SELF = "SELF"

    @property
    def preserves_left(self) -> bool:
        """True if all left-side rows appear in the output."""
        return self in (
            JoinType.LEFT, JoinType.FULL, JoinType.CROSS,
            JoinType.NATURAL, JoinType.LATERAL,
        )

    @property
    def preserves_right(self) -> bool:
        """True if all right-side rows appear in the output."""
        return self in (JoinType.RIGHT, JoinType.FULL, JoinType.CROSS)

    @property
    def may_duplicate(self) -> bool:
        """True if the join can produce more output rows than either input."""
        return self in (
            JoinType.INNER, JoinType.LEFT, JoinType.RIGHT,
            JoinType.FULL, JoinType.CROSS, JoinType.NATURAL,
            JoinType.LATERAL,
        )

    @property
    def is_filtering(self) -> bool:
        """True if the join may reduce rows from one side."""
        return self in (JoinType.INNER, JoinType.SEMI, JoinType.ANTI)


# =====================================================================
# Aggregate functions
# =====================================================================

class AggregateFunction(enum.Enum):
    """Standard SQL aggregate functions."""

    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    STDDEV = "STDDEV"
    STDDEV_POP = "STDDEV_POP"
    STDDEV_SAMP = "STDDEV_SAMP"
    VARIANCE = "VARIANCE"
    VAR_POP = "VAR_POP"
    VAR_SAMP = "VAR_SAMP"
    PERCENTILE = "PERCENTILE"
    PERCENTILE_CONT = "PERCENTILE_CONT"
    PERCENTILE_DISC = "PERCENTILE_DISC"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    ARRAY_AGG = "ARRAY_AGG"
    STRING_AGG = "STRING_AGG"
    LISTAGG = "LISTAGG"
    JSON_AGG = "JSON_AGG"
    JSONB_AGG = "JSONB_AGG"
    BOOL_AND = "BOOL_AND"
    BOOL_OR = "BOOL_OR"
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    FIRST = "FIRST"
    LAST = "LAST"
    NTH_VALUE = "NTH_VALUE"
    CORR = "CORR"
    COVAR_POP = "COVAR_POP"
    COVAR_SAMP = "COVAR_SAMP"
    REGR_SLOPE = "REGR_SLOPE"
    REGR_INTERCEPT = "REGR_INTERCEPT"
    APPROX_COUNT_DISTINCT = "APPROX_COUNT_DISTINCT"
    APPROX_PERCENTILE = "APPROX_PERCENTILE"

    @property
    def is_deterministic(self) -> bool:
        """True if the function always returns the same result for the same input multiset."""
        return self not in (
            AggregateFunction.APPROX_COUNT_DISTINCT,
            AggregateFunction.APPROX_PERCENTILE,
        )

    @property
    def is_commutative(self) -> bool:
        """True if order of input does not matter (always true for true aggregates)."""
        return self not in (
            AggregateFunction.FIRST,
            AggregateFunction.LAST,
            AggregateFunction.NTH_VALUE,
            AggregateFunction.STRING_AGG,
            AggregateFunction.LISTAGG,
            AggregateFunction.ARRAY_AGG,
        )

    @property
    def is_decomposable(self) -> bool:
        """True if the aggregate can be computed in a map-reduce fashion."""
        return self in (
            AggregateFunction.COUNT,
            AggregateFunction.SUM,
            AggregateFunction.MIN,
            AggregateFunction.MAX,
            AggregateFunction.BOOL_AND,
            AggregateFunction.BOOL_OR,
            AggregateFunction.BIT_AND,
            AggregateFunction.BIT_OR,
        )

    @property
    def output_type_for(self) -> SQLType | None:
        """Return a fixed output type, or None if it depends on input."""
        type_map: dict[AggregateFunction, SQLType] = {
            AggregateFunction.COUNT: SQLType.BIGINT,
            AggregateFunction.COUNT_DISTINCT: SQLType.BIGINT,
            AggregateFunction.BOOL_AND: SQLType.BOOLEAN,
            AggregateFunction.BOOL_OR: SQLType.BOOLEAN,
            AggregateFunction.APPROX_COUNT_DISTINCT: SQLType.BIGINT,
        }
        return type_map.get(self)

    @property
    def is_floating_point_sensitive(self) -> bool:
        """True if floating-point ordering can affect the result."""
        return self in (
            AggregateFunction.AVG,
            AggregateFunction.SUM,
            AggregateFunction.STDDEV,
            AggregateFunction.STDDEV_POP,
            AggregateFunction.STDDEV_SAMP,
            AggregateFunction.VARIANCE,
            AggregateFunction.VAR_POP,
            AggregateFunction.VAR_SAMP,
            AggregateFunction.CORR,
            AggregateFunction.COVAR_POP,
            AggregateFunction.COVAR_SAMP,
            AggregateFunction.REGR_SLOPE,
            AggregateFunction.REGR_INTERCEPT,
        )


# =====================================================================
# Window frame types
# =====================================================================

class WindowFrameType(enum.Enum):
    """SQL window frame specifications."""

    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


class WindowBound(enum.Enum):
    """Boundary specifications for window frames."""

    UNBOUNDED_PRECEDING = "UNBOUNDED PRECEDING"
    CURRENT_ROW = "CURRENT ROW"
    UNBOUNDED_FOLLOWING = "UNBOUNDED FOLLOWING"
    PRECEDING = "PRECEDING"  # N PRECEDING
    FOLLOWING = "FOLLOWING"  # N FOLLOWING


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class WindowFrame:
    """Full window frame specification."""

    frame_type: WindowFrameType = attr.ib(
        default=WindowFrameType.ROWS,
        validator=v.instance_of(WindowFrameType),
    )
    start_bound: WindowBound = attr.ib(
        default=WindowBound.UNBOUNDED_PRECEDING,
        validator=v.instance_of(WindowBound),
    )
    start_offset: int | None = attr.ib(default=None, validator=v.optional(v.instance_of(int)))
    end_bound: WindowBound = attr.ib(
        default=WindowBound.CURRENT_ROW,
        validator=v.instance_of(WindowBound),
    )
    end_offset: int | None = attr.ib(default=None, validator=v.optional(v.instance_of(int)))
    exclude: str = attr.ib(default="", validator=v.instance_of(str))

    def __str__(self) -> str:
        start = self.start_bound.value
        if self.start_offset is not None:
            start = f"{self.start_offset} {start}"
        end = self.end_bound.value
        if self.end_offset is not None:
            end = f"{self.end_offset} {end}"
        result = f"{self.frame_type.value} BETWEEN {start} AND {end}"
        if self.exclude:
            result += f" EXCLUDE {self.exclude}"
        return result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "frame_type": self.frame_type.value,
            "start_bound": self.start_bound.value,
            "end_bound": self.end_bound.value,
        }
        if self.start_offset is not None:
            d["start_offset"] = self.start_offset
        if self.end_offset is not None:
            d["end_offset"] = self.end_offset
        if self.exclude:
            d["exclude"] = self.exclude
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowFrame:
        return cls(
            frame_type=WindowFrameType(d.get("frame_type", "ROWS")),
            start_bound=WindowBound(d.get("start_bound", "UNBOUNDED PRECEDING")),
            start_offset=d.get("start_offset"),
            end_bound=WindowBound(d.get("end_bound", "CURRENT ROW")),
            end_offset=d.get("end_offset"),
            exclude=d.get("exclude", ""),
        )

    @classmethod
    def default_rows(cls) -> WindowFrame:
        return cls()

    @classmethod
    def entire_partition(cls) -> WindowFrame:
        return cls(
            start_bound=WindowBound.UNBOUNDED_PRECEDING,
            end_bound=WindowBound.UNBOUNDED_FOLLOWING,
        )


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class WindowSpec:
    """Complete window specification: partition, order, frame."""

    partition_by: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    order_by: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    frame: WindowFrame = attr.ib(factory=WindowFrame)
    window_name: str = attr.ib(default="", validator=v.instance_of(str))

    def __str__(self) -> str:
        parts: list[str] = []
        if self.partition_by:
            parts.append(f"PARTITION BY {', '.join(self.partition_by)}")
        if self.order_by:
            parts.append(f"ORDER BY {', '.join(self.order_by)}")
        parts.append(str(self.frame))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "partition_by": list(self.partition_by),
            "order_by": list(self.order_by),
            "frame": self.frame.to_dict(),
        }
        if self.window_name:
            d["window_name"] = self.window_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowSpec:
        return cls(
            partition_by=tuple(d.get("partition_by", [])),
            order_by=tuple(d.get("order_by", [])),
            frame=WindowFrame.from_dict(d.get("frame", {})),
            window_name=d.get("window_name", ""),
        )


# =====================================================================
# Set operation types
# =====================================================================

class SetOperationType(enum.Enum):
    """SQL set operation variants."""

    UNION = "UNION"
    UNION_ALL = "UNION_ALL"
    INTERSECT = "INTERSECT"
    INTERSECT_ALL = "INTERSECT_ALL"
    EXCEPT = "EXCEPT"
    EXCEPT_ALL = "EXCEPT_ALL"

    @property
    def is_distinct(self) -> bool:
        return self in (
            SetOperationType.UNION,
            SetOperationType.INTERSECT,
            SetOperationType.EXCEPT,
        )

    @property
    def is_commutative(self) -> bool:
        return self in (
            SetOperationType.UNION,
            SetOperationType.UNION_ALL,
            SetOperationType.INTERSECT,
            SetOperationType.INTERSECT_ALL,
        )

    @property
    def is_associative(self) -> bool:
        return self in (
            SetOperationType.UNION,
            SetOperationType.UNION_ALL,
            SetOperationType.INTERSECT,
            SetOperationType.INTERSECT_ALL,
        )


# =====================================================================
# Operator properties
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class OperatorProperties:
    """Algebraic properties of an operator relevant to delta propagation.

    These properties determine whether an operator belongs to Fragment F
    (the deterministic, order-independent fragment) and how deltas
    propagate through it.
    """

    deterministic: bool = attr.ib(default=True, validator=v.instance_of(bool))
    commutative: bool = attr.ib(default=False, validator=v.instance_of(bool))
    associative: bool = attr.ib(default=False, validator=v.instance_of(bool))
    idempotent: bool = attr.ib(default=False, validator=v.instance_of(bool))
    order_independent: bool = attr.ib(default=True, validator=v.instance_of(bool))
    monotone: bool = attr.ib(default=False, validator=v.instance_of(bool))
    preserves_keys: bool = attr.ib(default=False, validator=v.instance_of(bool))
    may_change_cardinality: bool = attr.ib(default=True, validator=v.instance_of(bool))
    has_side_effects: bool = attr.ib(default=False, validator=v.instance_of(bool))
    requires_full_input: bool = attr.ib(default=False, validator=v.instance_of(bool))

    @property
    def in_fragment_f(self) -> bool:
        """True if the operator is in Fragment F (deterministic + order-independent + no side effects)."""
        return (
            self.deterministic
            and self.order_independent
            and not self.has_side_effects
        )

    def to_dict(self) -> dict[str, Any]:
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperatorProperties:
        return cls(**{k: v for k, v in d.items() if k in attr.fields_dict(cls)})


# ── Default property sets for common operators ──

FILTER_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True, monotone=True,
    preserves_keys=True, may_change_cardinality=True,
)

PROJECT_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True, monotone=True,
    preserves_keys=False, may_change_cardinality=False,
)

JOIN_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True, commutative=False,
    may_change_cardinality=True,
)

INNER_JOIN_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True, commutative=True,
    associative=True, may_change_cardinality=True,
)

GROUP_BY_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True,
    may_change_cardinality=True, requires_full_input=True,
)

UNION_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True, commutative=True,
    associative=True, may_change_cardinality=True,
)

WINDOW_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=False,
    may_change_cardinality=False,
)

ORDER_BY_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=False,
    may_change_cardinality=False,
)

EXTERNAL_PROPERTIES = OperatorProperties(
    deterministic=False, order_independent=False,
    has_side_effects=True,
)

UDF_PROPERTIES = OperatorProperties(
    deterministic=False, order_independent=False,
)

SOURCE_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True,
    may_change_cardinality=False,
)

SINK_PROPERTIES = OperatorProperties(
    deterministic=True, order_independent=True,
    has_side_effects=True, may_change_cardinality=False,
)

# Mapping from SQLOperator to default properties
DEFAULT_OPERATOR_PROPERTIES: dict[SQLOperator, OperatorProperties] = {
    SQLOperator.SELECT: PROJECT_PROPERTIES,
    SQLOperator.PROJECT: PROJECT_PROPERTIES,
    SQLOperator.FILTER: FILTER_PROPERTIES,
    SQLOperator.JOIN: JOIN_PROPERTIES,
    SQLOperator.GROUP_BY: GROUP_BY_PROPERTIES,
    SQLOperator.ORDER_BY: ORDER_BY_PROPERTIES,
    SQLOperator.LIMIT: OperatorProperties(deterministic=False, order_independent=False, may_change_cardinality=True),
    SQLOperator.UNION: UNION_PROPERTIES,
    SQLOperator.INTERSECT: UNION_PROPERTIES,
    SQLOperator.EXCEPT: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=True),
    SQLOperator.SET_OP: UNION_PROPERTIES,
    SQLOperator.WINDOW: WINDOW_PROPERTIES,
    SQLOperator.CTE: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.SUBQUERY: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.LATERAL: OperatorProperties(deterministic=True, order_independent=False),
    SQLOperator.PIVOT: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=True),
    SQLOperator.UNPIVOT: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=True),
    SQLOperator.INSERT: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.UPDATE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.DELETE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.MERGE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.UPSERT: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.CREATE_TABLE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.ALTER_TABLE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.DROP_TABLE: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.CREATE_VIEW: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.CREATE_INDEX: OperatorProperties(deterministic=True, has_side_effects=True),
    SQLOperator.SOURCE: SOURCE_PROPERTIES,
    SQLOperator.SINK: SINK_PROPERTIES,
    SQLOperator.TRANSFORM: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.RENAME: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=False),
    SQLOperator.CAST: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=False),
    SQLOperator.FILL: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=False),
    SQLOperator.DEDUP: OperatorProperties(deterministic=True, order_independent=True, idempotent=True, may_change_cardinality=True),
    SQLOperator.EXPLODE: OperatorProperties(deterministic=True, order_independent=True, may_change_cardinality=True),
    SQLOperator.PANDAS_OP: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.PYSPARK_OP: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.DBT_MODEL: OperatorProperties(deterministic=True, order_independent=True),
    SQLOperator.EXTERNAL_CALL: EXTERNAL_PROPERTIES,
    SQLOperator.UDF: UDF_PROPERTIES,
    SQLOperator.OPAQUE: OperatorProperties(deterministic=False, order_independent=False),
}


def get_default_properties(op: SQLOperator) -> OperatorProperties:
    """Return the default algebraic properties for a SQL operator."""
    return DEFAULT_OPERATOR_PROPERTIES.get(op, OperatorProperties())


# =====================================================================
# Operator signature
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class OperatorSignature:
    """Type signature of an operator: inputs → output.

    Used for static type-checking of pipeline edges.
    """

    operator: SQLOperator = attr.ib(validator=v.instance_of(SQLOperator))
    input_schemas: tuple[Schema, ...] = attr.ib(converter=tuple)
    output_schema: Schema = attr.ib(validator=v.instance_of(Schema))
    properties: OperatorProperties = attr.ib(factory=OperatorProperties)
    parameters: dict[str, Any] = attr.ib(factory=dict)

    @property
    def arity(self) -> int:
        return len(self.input_schemas)

    @property
    def is_unary(self) -> bool:
        return self.arity == 1

    @property
    def is_binary(self) -> bool:
        return self.arity == 2

    @property
    def in_fragment_f(self) -> bool:
        return self.properties.in_fragment_f

    def __str__(self) -> str:
        ins = ", ".join(
            s.table_name or f"input_{i}"
            for i, s in enumerate(self.input_schemas)
        )
        out = self.output_schema.table_name or "output"
        frag = " [F]" if self.in_fragment_f else " [~F]"
        return f"{self.operator.value}({ins}) -> {out}{frag}"

    def validate_inputs(self, inputs: Sequence[Schema]) -> list[str]:
        """Check that *inputs* match the expected input schemas.

        Returns a list of mismatch descriptions (empty = valid).
        """
        errors: list[str] = []
        if len(inputs) != self.arity:
            errors.append(
                f"Expected {self.arity} inputs, got {len(inputs)}"
            )
            return errors
        for i, (expected, actual) in enumerate(zip(self.input_schemas, inputs)):
            mismatched = expected.compatible_with(actual)
            if mismatched:
                errors.append(f"Input {i}: mismatched columns {mismatched}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator.value,
            "input_schemas": [s.to_dict() for s in self.input_schemas],
            "output_schema": self.output_schema.to_dict(),
            "properties": self.properties.to_dict(),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperatorSignature:
        return cls(
            operator=SQLOperator(d["operator"]),
            input_schemas=tuple(Schema.from_dict(s) for s in d.get("input_schemas", [])),
            output_schema=Schema.from_dict(d["output_schema"]),
            properties=OperatorProperties.from_dict(d.get("properties", {})),
            parameters=d.get("parameters", {}),
        )


# =====================================================================
# Join specification
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class JoinCondition:
    """A single equi-join predicate: left_col = right_col."""

    left_column: str = attr.ib(validator=v.instance_of(str))
    right_column: str = attr.ib(validator=v.instance_of(str))
    operator: str = attr.ib(default="=", validator=v.instance_of(str))

    def __str__(self) -> str:
        return f"{self.left_column} {self.operator} {self.right_column}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_column": self.left_column,
            "right_column": self.right_column,
            "operator": self.operator,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JoinCondition:
        return cls(
            left_column=d["left_column"],
            right_column=d["right_column"],
            operator=d.get("operator", "="),
        )


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class JoinSpec:
    """Full join specification."""

    join_type: JoinType = attr.ib(validator=v.instance_of(JoinType))
    conditions: tuple[JoinCondition, ...] = attr.ib(factory=tuple, converter=tuple)
    using_columns: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    is_natural: bool = attr.ib(default=False, validator=v.instance_of(bool))

    def __str__(self) -> str:
        jtype = self.join_type.value
        if self.is_natural:
            jtype = f"NATURAL {jtype}"
        if self.using_columns:
            cols = ", ".join(self.using_columns)
            return f"{jtype} JOIN USING ({cols})"
        if self.conditions:
            conds = " AND ".join(str(c) for c in self.conditions)
            return f"{jtype} JOIN ON {conds}"
        return f"{jtype} JOIN"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "join_type": self.join_type.value,
        }
        if self.conditions:
            d["conditions"] = [c.to_dict() for c in self.conditions]
        if self.using_columns:
            d["using_columns"] = list(self.using_columns)
        if self.is_natural:
            d["is_natural"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JoinSpec:
        return cls(
            join_type=JoinType(d["join_type"]),
            conditions=tuple(JoinCondition.from_dict(c) for c in d.get("conditions", [])),
            using_columns=tuple(d.get("using_columns", [])),
            is_natural=d.get("is_natural", False),
        )


# =====================================================================
# Group-by specification
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class AggregateSpec:
    """A single aggregate expression in a GROUP BY."""

    function: AggregateFunction = attr.ib(validator=v.instance_of(AggregateFunction))
    input_columns: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    output_alias: str = attr.ib(default="", validator=v.instance_of(str))
    distinct: bool = attr.ib(default=False, validator=v.instance_of(bool))
    filter_expr: str = attr.ib(default="", validator=v.instance_of(str))
    order_by: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)

    def __str__(self) -> str:
        dist = "DISTINCT " if self.distinct else ""
        cols = ", ".join(self.input_columns) if self.input_columns else "*"
        result = f"{self.function.value}({dist}{cols})"
        if self.filter_expr:
            result += f" FILTER (WHERE {self.filter_expr})"
        if self.output_alias:
            result += f" AS {self.output_alias}"
        return result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "function": self.function.value,
            "input_columns": list(self.input_columns),
        }
        if self.output_alias:
            d["output_alias"] = self.output_alias
        if self.distinct:
            d["distinct"] = True
        if self.filter_expr:
            d["filter_expr"] = self.filter_expr
        if self.order_by:
            d["order_by"] = list(self.order_by)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AggregateSpec:
        return cls(
            function=AggregateFunction(d["function"]),
            input_columns=tuple(d.get("input_columns", [])),
            output_alias=d.get("output_alias", ""),
            distinct=d.get("distinct", False),
            filter_expr=d.get("filter_expr", ""),
            order_by=tuple(d.get("order_by", [])),
        )


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class GroupBySpec:
    """Full GROUP BY specification."""

    group_columns: tuple[str, ...] = attr.ib(converter=tuple)
    aggregates: tuple[AggregateSpec, ...] = attr.ib(factory=tuple, converter=tuple)
    having_expr: str = attr.ib(default="", validator=v.instance_of(str))

    def __str__(self) -> str:
        parts: list[str] = [f"GROUP BY {', '.join(self.group_columns)}"]
        for agg in self.aggregates:
            parts.append(f"  {agg}")
        if self.having_expr:
            parts.append(f"HAVING {self.having_expr}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "group_columns": list(self.group_columns),
            "aggregates": [a.to_dict() for a in self.aggregates],
        }
        if self.having_expr:
            d["having_expr"] = self.having_expr
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupBySpec:
        return cls(
            group_columns=d.get("group_columns", []),
            aggregates=tuple(AggregateSpec.from_dict(a) for a in d.get("aggregates", [])),
            having_expr=d.get("having_expr", ""),
        )


# =====================================================================
# Window function specification
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class WindowFunctionSpec:
    """A window function call with its OVER clause."""

    function: AggregateFunction = attr.ib(validator=v.instance_of(AggregateFunction))
    input_columns: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    window: WindowSpec = attr.ib(factory=WindowSpec)
    output_alias: str = attr.ib(default="", validator=v.instance_of(str))

    def __str__(self) -> str:
        cols = ", ".join(self.input_columns) if self.input_columns else "*"
        result = f"{self.function.value}({cols}) OVER ({self.window})"
        if self.output_alias:
            result += f" AS {self.output_alias}"
        return result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "function": self.function.value,
            "input_columns": list(self.input_columns),
            "window": self.window.to_dict(),
        }
        if self.output_alias:
            d["output_alias"] = self.output_alias
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowFunctionSpec:
        return cls(
            function=AggregateFunction(d["function"]),
            input_columns=tuple(d.get("input_columns", [])),
            window=WindowSpec.from_dict(d.get("window", {})),
            output_alias=d.get("output_alias", ""),
        )


# =====================================================================
# Column lineage tracking
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class ColumnLineage:
    """Tracks which input columns contribute to an output column."""

    output_column: str = attr.ib(validator=v.instance_of(str))
    source_columns: tuple[str, ...] = attr.ib(converter=tuple)
    transform_type: str = attr.ib(default="direct", validator=v.instance_of(str))
    expression: str = attr.ib(default="", validator=v.instance_of(str))
    source_node: str = attr.ib(default="", validator=v.instance_of(str))

    @property
    def is_direct_mapping(self) -> bool:
        return self.transform_type == "direct" and len(self.source_columns) == 1

    def __str__(self) -> str:
        srcs = ", ".join(self.source_columns)
        src_node = f"{self.source_node}." if self.source_node else ""
        return f"{self.output_column} <- {src_node}{srcs} [{self.transform_type}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_column": self.output_column,
            "source_columns": list(self.source_columns),
            "transform_type": self.transform_type,
            "expression": self.expression,
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnLineage:
        return cls(
            output_column=d["output_column"],
            source_columns=d.get("source_columns", []),
            transform_type=d.get("transform_type", "direct"),
            expression=d.get("expression", ""),
            source_node=d.get("source_node", ""),
        )


@attr.s(frozen=True, slots=True, repr=True)
class OperatorLineage:
    """Full column-level lineage for an operator."""

    operator: SQLOperator = attr.ib(validator=v.instance_of(SQLOperator))
    column_lineages: tuple[ColumnLineage, ...] = attr.ib(converter=tuple)
    node_id: str = attr.ib(default="", validator=v.instance_of(str))

    def output_columns(self) -> frozenset[str]:
        return frozenset(cl.output_column for cl in self.column_lineages)

    def source_columns_for(self, output_col: str) -> tuple[str, ...]:
        for cl in self.column_lineages:
            if cl.output_column == output_col:
                return cl.source_columns
        return ()

    def all_source_columns(self) -> frozenset[str]:
        result: set[str] = set()
        for cl in self.column_lineages:
            result.update(cl.source_columns)
        return frozenset(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator.value,
            "column_lineages": [cl.to_dict() for cl in self.column_lineages],
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperatorLineage:
        return cls(
            operator=SQLOperator(d["operator"]),
            column_lineages=tuple(
                ColumnLineage.from_dict(cl) for cl in d.get("column_lineages", [])
            ),
            node_id=d.get("node_id", ""),
        )
