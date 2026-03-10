"""
Foundation types for the Algebraic Repair Calculus.

Defines the SQL type system, type compatibility/widening rules, column and
schema descriptors, quality constraints, and availability contracts.  All
types use ``attrs`` with frozen=True for immutability and carry full
validators so that invalid objects cannot be constructed.
"""

from __future__ import annotations

import enum
import math
import re
from datetime import timedelta
from typing import Any, Sequence

import attr
from attr import validators as v

from arc.types.errors import (
    SchemaError,
    TypeCompatibilityError,
    TypeMismatchError,
    TypeParameterError,
    ColumnNotFoundError,
    DuplicateColumnError,
    ErrorCode,
)

# =====================================================================
# SQL type enumeration
# =====================================================================


class SQLType(enum.Enum):
    """Enumeration of all supported SQL column types.

    Parameterised variants (VARCHAR(n), DECIMAL(p,s), …) are represented
    by pairing a base ``SQLType`` with a :class:`TypeParameters` instance.
    """

    # Integer family
    SMALLINT = "SMALLINT"
    INT = "INT"
    BIGINT = "BIGINT"
    SERIAL = "SERIAL"
    BIGSERIAL = "BIGSERIAL"

    # Floating-point family
    REAL = "REAL"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"

    # Fixed-point family
    NUMERIC = "NUMERIC"
    DECIMAL = "DECIMAL"

    # String family
    CHAR = "CHAR"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"

    # Binary
    BYTEA = "BYTEA"
    BLOB = "BLOB"

    # Boolean
    BOOLEAN = "BOOLEAN"

    # Date/time family
    DATE = "DATE"
    TIME = "TIME"
    TIMETZ = "TIMETZ"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMPTZ = "TIMESTAMPTZ"
    INTERVAL = "INTERVAL"

    # Semi-structured
    JSON = "JSON"
    JSONB = "JSONB"

    # Collection / complex
    ARRAY = "ARRAY"
    UUID = "UUID"

    @classmethod
    def from_string(cls, s: str) -> "SQLType":
        """Parse a type name string (case-insensitive).

        Handles common aliases such as ``INTEGER``, ``DOUBLE PRECISION``,
        ``BOOL``, ``TIMESTAMP WITH TIME ZONE``, etc.
        """
        norm = s.strip().upper()
        # strip parameterisation for base-type lookup
        norm = re.sub(r"\(.*\)", "", norm).strip()
        aliases: dict[str, SQLType] = {
            "INTEGER": cls.INT,
            "INT4": cls.INT,
            "INT8": cls.BIGINT,
            "INT2": cls.SMALLINT,
            "TINYINT": cls.SMALLINT,
            "BOOL": cls.BOOLEAN,
            "FLOAT4": cls.REAL,
            "FLOAT8": cls.DOUBLE,
            "DOUBLE PRECISION": cls.DOUBLE,
            "CHARACTER VARYING": cls.VARCHAR,
            "CHARACTER": cls.CHAR,
            "TIMESTAMP WITH TIME ZONE": cls.TIMESTAMPTZ,
            "TIMESTAMP WITHOUT TIME ZONE": cls.TIMESTAMP,
            "TIME WITH TIME ZONE": cls.TIMETZ,
            "TIME WITHOUT TIME ZONE": cls.TIME,
            "SERIAL4": cls.SERIAL,
            "SERIAL8": cls.BIGSERIAL,
            "BINARY LARGE OBJECT": cls.BLOB,
            "BYTEA": cls.BYTEA,
            "VARBINARY": cls.BYTEA,
        }
        if norm in aliases:
            return aliases[norm]
        try:
            return cls(norm)
        except ValueError:
            pass
        raise ValueError(f"Unknown SQL type: {s!r}")

    # -- Type family helpers --

    @property
    def is_integer(self) -> bool:
        return self in _INTEGER_TYPES

    @property
    def is_floating(self) -> bool:
        return self in _FLOATING_TYPES

    @property
    def is_numeric(self) -> bool:
        return self in _NUMERIC_TYPES

    @property
    def is_string(self) -> bool:
        return self in _STRING_TYPES

    @property
    def is_temporal(self) -> bool:
        return self in _TEMPORAL_TYPES

    @property
    def is_binary(self) -> bool:
        return self in (SQLType.BYTEA, SQLType.BLOB)

    @property
    def is_json(self) -> bool:
        return self in (SQLType.JSON, SQLType.JSONB)

    @property
    def is_parameterised(self) -> bool:
        """True if this base type may carry parameters (length, precision)."""
        return self in _PARAMETERISED_TYPES


_INTEGER_TYPES = frozenset({
    SQLType.SMALLINT, SQLType.INT, SQLType.BIGINT,
    SQLType.SERIAL, SQLType.BIGSERIAL,
})

_FLOATING_TYPES = frozenset({
    SQLType.REAL, SQLType.FLOAT, SQLType.DOUBLE,
})

_FIXED_POINT_TYPES = frozenset({
    SQLType.NUMERIC, SQLType.DECIMAL,
})

_NUMERIC_TYPES = _INTEGER_TYPES | _FLOATING_TYPES | _FIXED_POINT_TYPES

_STRING_TYPES = frozenset({
    SQLType.CHAR, SQLType.VARCHAR, SQLType.TEXT,
})

_TEMPORAL_TYPES = frozenset({
    SQLType.DATE, SQLType.TIME, SQLType.TIMETZ,
    SQLType.TIMESTAMP, SQLType.TIMESTAMPTZ, SQLType.INTERVAL,
})

_PARAMETERISED_TYPES = frozenset({
    SQLType.CHAR, SQLType.VARCHAR,
    SQLType.NUMERIC, SQLType.DECIMAL,
    SQLType.ARRAY,
})


# =====================================================================
# Type parameters
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class TypeParameters:
    """Optional parameters that qualify a :class:`SQLType`.

    Examples
    --------
    ``VARCHAR(255)`` → ``TypeParameters(length=255)``
    ``DECIMAL(18, 4)`` → ``TypeParameters(precision=18, scale=4)``
    ``INT[]`` → ``TypeParameters(element_type=SQLType.INT)``
    """

    length: int | None = attr.ib(default=None, validator=v.optional(v.instance_of(int)))
    precision: int | None = attr.ib(default=None, validator=v.optional(v.instance_of(int)))
    scale: int | None = attr.ib(default=None, validator=v.optional(v.instance_of(int)))
    element_type: SQLType | None = attr.ib(default=None, validator=v.optional(v.instance_of(SQLType)))
    element_params: TypeParameters | None = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        if self.length is not None and self.length < 0:
            raise TypeParameterError("VARCHAR/CHAR", str(self.length), "length must be >= 0")
        if self.precision is not None and self.precision < 1:
            raise TypeParameterError("NUMERIC/DECIMAL", str(self.precision), "precision must be >= 1")
        if self.scale is not None and self.precision is not None:
            if self.scale > self.precision:
                raise TypeParameterError(
                    "NUMERIC/DECIMAL",
                    f"({self.precision},{self.scale})",
                    "scale must be <= precision",
                )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.length is not None:
            d["length"] = self.length
        if self.precision is not None:
            d["precision"] = self.precision
        if self.scale is not None:
            d["scale"] = self.scale
        if self.element_type is not None:
            d["element_type"] = self.element_type.value
        if self.element_params is not None:
            d["element_params"] = self.element_params.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TypeParameters:
        et = None
        if "element_type" in d:
            et = SQLType(d["element_type"])
        ep = None
        if "element_params" in d:
            ep = cls.from_dict(d["element_params"])
        return cls(
            length=d.get("length"),
            precision=d.get("precision"),
            scale=d.get("scale"),
            element_type=et,
            element_params=ep,
        )


# =====================================================================
# Parameterised SQL type
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class ParameterisedType:
    """A SQL type together with optional parameters.

    This is the canonical representation used in :class:`Column` and
    throughout the algebra engine.
    """

    base: SQLType = attr.ib(validator=v.instance_of(SQLType))
    params: TypeParameters = attr.ib(factory=TypeParameters, validator=v.instance_of(TypeParameters))

    @classmethod
    def simple(cls, base: SQLType) -> ParameterisedType:
        return cls(base=base)

    @classmethod
    def varchar(cls, length: int = 255) -> ParameterisedType:
        return cls(base=SQLType.VARCHAR, params=TypeParameters(length=length))

    @classmethod
    def char(cls, length: int = 1) -> ParameterisedType:
        return cls(base=SQLType.CHAR, params=TypeParameters(length=length))

    @classmethod
    def decimal(cls, precision: int = 18, scale: int = 4) -> ParameterisedType:
        return cls(base=SQLType.DECIMAL, params=TypeParameters(precision=precision, scale=scale))

    @classmethod
    def numeric(cls, precision: int = 18, scale: int = 0) -> ParameterisedType:
        return cls(base=SQLType.NUMERIC, params=TypeParameters(precision=precision, scale=scale))

    @classmethod
    def array_of(cls, element: SQLType, element_params: TypeParameters | None = None) -> ParameterisedType:
        return cls(
            base=SQLType.ARRAY,
            params=TypeParameters(element_type=element, element_params=element_params),
        )

    @classmethod
    def from_string(cls, s: str) -> ParameterisedType:
        """Parse ``'VARCHAR(100)'``, ``'DECIMAL(10,2)'``, ``'INT[]'``, etc."""
        s = s.strip()
        # Array shorthand: INT[], VARCHAR(100)[]
        if s.endswith("[]"):
            inner = cls.from_string(s[:-2])
            return cls.array_of(inner.base, inner.params if inner.params != TypeParameters() else None)
        m = re.match(r"^(\w[\w\s]*)(?:\(([^)]+)\))?$", s, re.IGNORECASE)
        if not m:
            raise ValueError(f"Cannot parse SQL type: {s!r}")
        base_str, param_str = m.group(1).strip(), m.group(2)
        base = SQLType.from_string(base_str)
        if param_str is None:
            return cls.simple(base)
        parts = [p.strip() for p in param_str.split(",")]
        if base in (SQLType.VARCHAR, SQLType.CHAR):
            return cls(base=base, params=TypeParameters(length=int(parts[0])))
        if base in (SQLType.NUMERIC, SQLType.DECIMAL):
            prec = int(parts[0])
            scl = int(parts[1]) if len(parts) > 1 else 0
            return cls(base=base, params=TypeParameters(precision=prec, scale=scl))
        return cls.simple(base)

    def __str__(self) -> str:
        base_name = self.base.value
        if self.base == SQLType.ARRAY and self.params.element_type is not None:
            inner = self.params.element_type.value
            if self.params.element_params:
                inner += self._format_inner_params(self.params.element_params)
            return f"{inner}[]"
        suffix = self._format_inner_params(self.params)
        return f"{base_name}{suffix}"

    @staticmethod
    def _format_inner_params(p: TypeParameters) -> str:
        if p.length is not None:
            return f"({p.length})"
        if p.precision is not None:
            if p.scale and p.scale > 0:
                return f"({p.precision},{p.scale})"
            return f"({p.precision})"
        return ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"base": self.base.value}
        pd = self.params.to_dict()
        if pd:
            d["params"] = pd
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParameterisedType:
        base = SQLType(d["base"])
        params = TypeParameters.from_dict(d["params"]) if "params" in d else TypeParameters()
        return cls(base=base, params=params)


# =====================================================================
# Type compatibility / widening matrix
# =====================================================================

class WideningResult(enum.Enum):
    """Outcome of attempting to widen type A to type B."""
    IDENTICAL = "identical"
    SAFE_WIDENING = "safe_widening"
    LOSSY_NARROWING = "lossy_narrowing"
    INCOMPATIBLE = "incompatible"


# Widening precedence: types listed later are "wider"
_INTEGER_WIDENING_ORDER: list[SQLType] = [
    SQLType.SMALLINT, SQLType.INT, SQLType.BIGINT,
]

_FLOAT_WIDENING_ORDER: list[SQLType] = [
    SQLType.REAL, SQLType.FLOAT, SQLType.DOUBLE,
]


class TypeCompatibility:
    """Static methods for computing type compatibility, widening, and
    common supertypes.

    The full compatibility matrix covers integer widening, float widening,
    integer→float promotion, string widening, temporal promotions, and
    parameterised type interactions (e.g., VARCHAR(50) → VARCHAR(100)).
    """

    # ── Core comparison ──

    @staticmethod
    def compare(a: ParameterisedType, b: ParameterisedType) -> WideningResult:
        """Determine the relationship when widening *a* to *b*."""
        if a == b:
            return WideningResult.IDENTICAL

        # Same base type — compare parameters
        if a.base == b.base:
            return TypeCompatibility._compare_params(a, b)

        # Integer widening
        if a.base in _INTEGER_TYPES and b.base in _INTEGER_TYPES:
            return TypeCompatibility._compare_ordered(
                a.base, b.base, _INTEGER_WIDENING_ORDER,
            )

        # Float widening
        if a.base in _FLOATING_TYPES and b.base in _FLOATING_TYPES:
            return TypeCompatibility._compare_ordered(
                a.base, b.base, _FLOAT_WIDENING_ORDER,
            )

        # Integer -> Float promotion (safe but potentially lossy for large ints)
        if a.base in _INTEGER_TYPES and b.base in _FLOATING_TYPES:
            return WideningResult.SAFE_WIDENING

        # Float -> Integer narrowing
        if a.base in _FLOATING_TYPES and b.base in _INTEGER_TYPES:
            return WideningResult.LOSSY_NARROWING

        # Integer / Float -> Decimal is safe
        if a.base in (_INTEGER_TYPES | _FLOATING_TYPES) and b.base in _FIXED_POINT_TYPES:
            return WideningResult.SAFE_WIDENING

        # Decimal -> Float is lossy
        if a.base in _FIXED_POINT_TYPES and b.base in _FLOATING_TYPES:
            return WideningResult.LOSSY_NARROWING

        # Decimal -> Integer is lossy
        if a.base in _FIXED_POINT_TYPES and b.base in _INTEGER_TYPES:
            return WideningResult.LOSSY_NARROWING

        # String widening: CHAR -> VARCHAR -> TEXT
        if a.base in _STRING_TYPES and b.base in _STRING_TYPES:
            return TypeCompatibility._compare_string(a, b)

        # Temporal promotions
        if a.base in _TEMPORAL_TYPES and b.base in _TEMPORAL_TYPES:
            return TypeCompatibility._compare_temporal(a.base, b.base)

        # JSON -> JSONB is safe, JSONB -> JSON is lossy
        if a.base == SQLType.JSON and b.base == SQLType.JSONB:
            return WideningResult.SAFE_WIDENING
        if a.base == SQLType.JSONB and b.base == SQLType.JSON:
            return WideningResult.LOSSY_NARROWING

        # SERIAL -> INT, BIGSERIAL -> BIGINT
        if a.base == SQLType.SERIAL and b.base == SQLType.INT:
            return WideningResult.SAFE_WIDENING
        if a.base == SQLType.BIGSERIAL and b.base == SQLType.BIGINT:
            return WideningResult.SAFE_WIDENING

        # BYTEA <-> BLOB equivalence
        if {a.base, b.base} == {SQLType.BYTEA, SQLType.BLOB}:
            return WideningResult.SAFE_WIDENING

        return WideningResult.INCOMPATIBLE

    # ── Least common supertype ──

    @staticmethod
    def common_supertype(
        a: ParameterisedType,
        b: ParameterisedType,
    ) -> ParameterisedType | None:
        """Return the narrowest type that both *a* and *b* can widen to.

        Returns ``None`` if the types are incompatible.
        """
        if a == b:
            return a

        # Same base — take wider params
        if a.base == b.base:
            return TypeCompatibility._merge_params(a, b)

        # Integer family
        if a.base in _INTEGER_TYPES and b.base in _INTEGER_TYPES:
            wider = TypeCompatibility._wider_in_order(
                a.base, b.base, _INTEGER_WIDENING_ORDER,
            )
            return ParameterisedType.simple(wider) if wider else None

        # Float family
        if a.base in _FLOATING_TYPES and b.base in _FLOATING_TYPES:
            wider = TypeCompatibility._wider_in_order(
                a.base, b.base, _FLOAT_WIDENING_ORDER,
            )
            return ParameterisedType.simple(wider) if wider else None

        # Integer + Float -> Double
        if (a.base in _INTEGER_TYPES and b.base in _FLOATING_TYPES) or \
           (a.base in _FLOATING_TYPES and b.base in _INTEGER_TYPES):
            return ParameterisedType.simple(SQLType.DOUBLE)

        # Anything numeric + DECIMAL -> DECIMAL
        if a.base in _NUMERIC_TYPES and b.base in _FIXED_POINT_TYPES:
            return ParameterisedType.simple(b.base)
        if a.base in _FIXED_POINT_TYPES and b.base in _NUMERIC_TYPES:
            return ParameterisedType.simple(a.base)

        # String family -> TEXT as top
        if a.base in _STRING_TYPES and b.base in _STRING_TYPES:
            return ParameterisedType.simple(SQLType.TEXT)

        # TIMESTAMP <-> TIMESTAMPTZ -> TIMESTAMPTZ
        if {a.base, b.base} == {SQLType.TIMESTAMP, SQLType.TIMESTAMPTZ}:
            return ParameterisedType.simple(SQLType.TIMESTAMPTZ)

        # DATE + TIMESTAMP -> TIMESTAMP
        if {a.base, b.base} == {SQLType.DATE, SQLType.TIMESTAMP}:
            return ParameterisedType.simple(SQLType.TIMESTAMP)
        if {a.base, b.base} == {SQLType.DATE, SQLType.TIMESTAMPTZ}:
            return ParameterisedType.simple(SQLType.TIMESTAMPTZ)

        # TIME <-> TIMETZ -> TIMETZ
        if {a.base, b.base} == {SQLType.TIME, SQLType.TIMETZ}:
            return ParameterisedType.simple(SQLType.TIMETZ)

        # JSON / JSONB -> JSONB
        if {a.base, b.base} == {SQLType.JSON, SQLType.JSONB}:
            return ParameterisedType.simple(SQLType.JSONB)

        # BYTEA / BLOB -> BYTEA
        if {a.base, b.base} == {SQLType.BYTEA, SQLType.BLOB}:
            return ParameterisedType.simple(SQLType.BYTEA)

        return None

    @staticmethod
    def can_widen(source: ParameterisedType, target: ParameterisedType) -> bool:
        """True if *source* can be safely widened to *target*."""
        result = TypeCompatibility.compare(source, target)
        return result in (WideningResult.IDENTICAL, WideningResult.SAFE_WIDENING)

    @staticmethod
    def assert_compatible(
        source: ParameterisedType,
        target: ParameterisedType,
        column_name: str = "<unknown>",
    ) -> None:
        """Raise :class:`TypeCompatibilityError` if widening is impossible."""
        result = TypeCompatibility.compare(source, target)
        if result == WideningResult.INCOMPATIBLE:
            raise TypeCompatibilityError(str(source), str(target))

    # ── Internal helpers ──

    @staticmethod
    def _compare_ordered(
        a: SQLType,
        b: SQLType,
        order: list[SQLType],
    ) -> WideningResult:
        try:
            ia, ib = order.index(a), order.index(b)
        except ValueError:
            return WideningResult.INCOMPATIBLE
        if ia < ib:
            return WideningResult.SAFE_WIDENING
        if ia > ib:
            return WideningResult.LOSSY_NARROWING
        return WideningResult.IDENTICAL

    @staticmethod
    def _wider_in_order(
        a: SQLType,
        b: SQLType,
        order: list[SQLType],
    ) -> SQLType | None:
        try:
            ia, ib = order.index(a), order.index(b)
        except ValueError:
            return None
        return order[max(ia, ib)]

    @staticmethod
    def _compare_params(
        a: ParameterisedType,
        b: ParameterisedType,
    ) -> WideningResult:
        """Compare same-base types by their parameters."""
        if a.base in (SQLType.VARCHAR, SQLType.CHAR):
            la = a.params.length or 0
            lb = b.params.length or 0
            if la == lb:
                return WideningResult.IDENTICAL
            return WideningResult.SAFE_WIDENING if la < lb else WideningResult.LOSSY_NARROWING
        if a.base in (SQLType.NUMERIC, SQLType.DECIMAL):
            pa, sa = a.params.precision or 18, a.params.scale or 0
            pb, sb = b.params.precision or 18, b.params.scale or 0
            if pa == pb and sa == sb:
                return WideningResult.IDENTICAL
            if pa <= pb and sa <= sb:
                return WideningResult.SAFE_WIDENING
            if pa >= pb and sa >= sb:
                return WideningResult.LOSSY_NARROWING
            return WideningResult.LOSSY_NARROWING
        # ARRAY: compare element types
        if a.base == SQLType.ARRAY:
            if a.params.element_type is None or b.params.element_type is None:
                return WideningResult.SAFE_WIDENING
            ea = ParameterisedType(base=a.params.element_type, params=a.params.element_params or TypeParameters())
            eb = ParameterisedType(base=b.params.element_type, params=b.params.element_params or TypeParameters())
            return TypeCompatibility.compare(ea, eb)
        # Same base, same (empty) params -> identical
        return WideningResult.IDENTICAL

    @staticmethod
    def _compare_string(
        a: ParameterisedType,
        b: ParameterisedType,
    ) -> WideningResult:
        _STR_ORDER: list[SQLType] = [SQLType.CHAR, SQLType.VARCHAR, SQLType.TEXT]
        try:
            ia, ib = _STR_ORDER.index(a.base), _STR_ORDER.index(b.base)
        except ValueError:
            return WideningResult.INCOMPATIBLE
        if ia < ib:
            return WideningResult.SAFE_WIDENING
        if ia > ib:
            return WideningResult.LOSSY_NARROWING
        return TypeCompatibility._compare_params(a, b)

    @staticmethod
    def _compare_temporal(a: SQLType, b: SQLType) -> WideningResult:
        _TEMPORAL_ORDER = {
            SQLType.DATE: 0,
            SQLType.TIME: 1,
            SQLType.TIMETZ: 2,
            SQLType.TIMESTAMP: 3,
            SQLType.TIMESTAMPTZ: 4,
            SQLType.INTERVAL: 5,
        }
        ia = _TEMPORAL_ORDER.get(a)
        ib = _TEMPORAL_ORDER.get(b)
        if ia is None or ib is None:
            return WideningResult.INCOMPATIBLE
        # DATE -> TIMESTAMP is safe; INTERVAL is its own kind
        if a == SQLType.INTERVAL or b == SQLType.INTERVAL:
            return WideningResult.INCOMPATIBLE
        if ia < ib:
            return WideningResult.SAFE_WIDENING
        if ia > ib:
            return WideningResult.LOSSY_NARROWING
        return WideningResult.IDENTICAL

    @staticmethod
    def _merge_params(
        a: ParameterisedType,
        b: ParameterisedType,
    ) -> ParameterisedType:
        """Merge parameters by taking the wider of each dimension."""
        if a.base in (SQLType.VARCHAR, SQLType.CHAR):
            la = a.params.length or 0
            lb = b.params.length or 0
            return ParameterisedType(base=a.base, params=TypeParameters(length=max(la, lb)))
        if a.base in (SQLType.NUMERIC, SQLType.DECIMAL):
            pa, sa = a.params.precision or 18, a.params.scale or 0
            pb, sb = b.params.precision or 18, b.params.scale or 0
            return ParameterisedType(
                base=a.base,
                params=TypeParameters(precision=max(pa, pb), scale=max(sa, sb)),
            )
        return a  # no meaningful merge for other types


# =====================================================================
# Constraint types
# =====================================================================

class ConstraintType(enum.Enum):
    """Types of column/table constraints."""
    NOT_NULL = "NOT_NULL"
    UNIQUE = "UNIQUE"
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    CHECK = "CHECK"
    DEFAULT = "DEFAULT"
    RANGE = "RANGE"
    PATTERN = "PATTERN"
    ENUM_VALUES = "ENUM_VALUES"


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class ColumnConstraint:
    """A constraint on a single column."""
    constraint_type: ConstraintType = attr.ib(validator=v.instance_of(ConstraintType))
    expression: str = attr.ib(default="", validator=v.instance_of(str))
    parameters: dict[str, Any] = attr.ib(factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.constraint_type.value}
        if self.expression:
            d["expression"] = self.expression
        if self.parameters:
            d["parameters"] = self.parameters
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnConstraint:
        return cls(
            constraint_type=ConstraintType(d["type"]),
            expression=d.get("expression", ""),
            parameters=d.get("parameters", {}),
        )

    @classmethod
    def not_null(cls) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.NOT_NULL)

    @classmethod
    def unique(cls) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.UNIQUE)

    @classmethod
    def check(cls, expr: str) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.CHECK, expression=expr)

    @classmethod
    def default(cls, expr: str) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.DEFAULT, expression=expr)

    @classmethod
    def range_constraint(cls, min_val: float | None = None, max_val: float | None = None) -> ColumnConstraint:
        params: dict[str, Any] = {}
        if min_val is not None:
            params["min"] = min_val
        if max_val is not None:
            params["max"] = max_val
        return cls(constraint_type=ConstraintType.RANGE, parameters=params)

    @classmethod
    def pattern(cls, regex: str) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.PATTERN, expression=regex)

    @classmethod
    def enum_values(cls, values: list[str]) -> ColumnConstraint:
        return cls(constraint_type=ConstraintType.ENUM_VALUES, parameters={"values": values})


# =====================================================================
# Column
# =====================================================================

def _validate_column_name(instance: Any, attribute: Any, value: str) -> None:
    if not value:
        raise SchemaError("Column name must not be empty", code=ErrorCode.SCHEMA_INVALID)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
        # allow quoted identifiers
        if not (value.startswith('"') and value.endswith('"')):
            raise SchemaError(
                f"Invalid column name: {value!r}",
                code=ErrorCode.SCHEMA_INVALID,
                context={"column": value},
            )


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class Column:
    """Descriptor for a single table/view column.

    Parameters
    ----------
    name:
        Column identifier.
    sql_type:
        Parameterised SQL type.
    nullable:
        Whether the column admits NULL.
    default_expr:
        Default value expression (SQL literal or function call).
    position:
        0-based ordinal within the schema.
    constraints:
        Additional column-level constraints.
    description:
        Human-readable description (documentation).
    """

    name: str = attr.ib(validator=[v.instance_of(str), _validate_column_name])
    sql_type: ParameterisedType = attr.ib(validator=v.instance_of(ParameterisedType))
    nullable: bool = attr.ib(default=True, validator=v.instance_of(bool))
    default_expr: str | None = attr.ib(default=None, validator=v.optional(v.instance_of(str)))
    position: int = attr.ib(default=0, validator=v.instance_of(int))
    constraints: tuple[ColumnConstraint, ...] = attr.ib(factory=tuple, hash=False)
    description: str = attr.ib(default="", validator=v.instance_of(str))

    def __str__(self) -> str:
        parts = [self.name, str(self.sql_type)]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default_expr is not None:
            parts.append(f"DEFAULT {self.default_expr}")
        return " ".join(parts)

    def with_type(self, new_type: ParameterisedType) -> Column:
        return attr.evolve(self, sql_type=new_type)

    def with_nullable(self, nullable: bool) -> Column:
        return attr.evolve(self, nullable=nullable)

    def with_position(self, pos: int) -> Column:
        return attr.evolve(self, position=pos)

    def with_name(self, name: str) -> Column:
        return attr.evolve(self, name=name)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "sql_type": self.sql_type.to_dict(),
            "nullable": self.nullable,
            "position": self.position,
        }
        if self.default_expr is not None:
            d["default_expr"] = self.default_expr
        if self.constraints:
            d["constraints"] = [c.to_dict() for c in self.constraints]
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Column:
        constraints = tuple(
            ColumnConstraint.from_dict(c)
            for c in d.get("constraints", [])
        )
        return cls(
            name=d["name"],
            sql_type=ParameterisedType.from_dict(d["sql_type"]),
            nullable=d.get("nullable", True),
            default_expr=d.get("default_expr"),
            position=d.get("position", 0),
            constraints=constraints,
            description=d.get("description", ""),
        )

    @classmethod
    def quick(
        cls,
        name: str,
        base_type: SQLType,
        nullable: bool = True,
        position: int = 0,
    ) -> Column:
        """Shorthand factory for simple (unparameterised) columns."""
        return cls(
            name=name,
            sql_type=ParameterisedType.simple(base_type),
            nullable=nullable,
            position=position,
        )


# =====================================================================
# Foreign key reference
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class ForeignKey:
    """A foreign key reference."""

    columns: tuple[str, ...] = attr.ib(converter=tuple)
    ref_table: str = attr.ib(validator=v.instance_of(str))
    ref_columns: tuple[str, ...] = attr.ib(converter=tuple)
    on_delete: str = attr.ib(default="NO ACTION", validator=v.instance_of(str))
    on_update: str = attr.ib(default="NO ACTION", validator=v.instance_of(str))
    constraint_name: str | None = attr.ib(default=None)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "columns": list(self.columns),
            "ref_table": self.ref_table,
            "ref_columns": list(self.ref_columns),
            "on_delete": self.on_delete,
            "on_update": self.on_update,
        }
        if self.constraint_name:
            d["constraint_name"] = self.constraint_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForeignKey:
        return cls(
            columns=d["columns"],
            ref_table=d["ref_table"],
            ref_columns=d["ref_columns"],
            on_delete=d.get("on_delete", "NO ACTION"),
            on_update=d.get("on_update", "NO ACTION"),
            constraint_name=d.get("constraint_name"),
        )


# =====================================================================
# Check constraint (table-level)
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class CheckConstraint:
    """A table-level CHECK constraint."""

    expression: str = attr.ib(validator=v.instance_of(str))
    constraint_name: str | None = attr.ib(default=None)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"expression": self.expression}
        if self.constraint_name:
            d["constraint_name"] = self.constraint_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckConstraint:
        return cls(
            expression=d["expression"],
            constraint_name=d.get("constraint_name"),
        )


# =====================================================================
# Schema
# =====================================================================

def _validate_columns(instance: Any, attribute: Any, value: tuple[Column, ...]) -> None:
    if not value:
        return
    names: set[str] = set()
    for col in value:
        if col.name in names:
            raise DuplicateColumnError(col.name)
        names.add(col.name)


@attr.s(frozen=True, slots=True, repr=True, hash=True)
class Schema:
    """Immutable relational schema with full constraint support.

    A schema is the type-level description of a table, view, or
    intermediate relation in the pipeline DAG.
    """

    columns: tuple[Column, ...] = attr.ib(
        converter=tuple,
        validator=[v.deep_iterable(v.instance_of(Column)), _validate_columns],
    )
    primary_key: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    unique_constraints: tuple[tuple[str, ...], ...] = attr.ib(factory=tuple, converter=tuple)
    foreign_keys: tuple[ForeignKey, ...] = attr.ib(factory=tuple, converter=tuple)
    check_constraints: tuple[CheckConstraint, ...] = attr.ib(factory=tuple, converter=tuple)
    schema_name: str = attr.ib(default="", validator=v.instance_of(str))
    table_name: str = attr.ib(default="", validator=v.instance_of(str))

    def __attrs_post_init__(self) -> None:
        col_names = self.column_names
        # Validate primary key references
        for pk_col in self.primary_key:
            if pk_col not in col_names:
                raise ColumnNotFoundError(pk_col, available=list(col_names))
        # Validate unique constraints
        for uc in self.unique_constraints:
            for uc_col in uc:
                if uc_col not in col_names:
                    raise ColumnNotFoundError(uc_col, available=list(col_names))
        # Validate foreign key source columns
        for fk in self.foreign_keys:
            for fk_col in fk.columns:
                if fk_col not in col_names:
                    raise ColumnNotFoundError(fk_col, available=list(col_names))

    # -- Column access ---

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(c.name for c in self.columns)

    @property
    def column_list(self) -> list[str]:
        return [c.name for c in self.columns]

    def __len__(self) -> int:
        return len(self.columns)

    def __contains__(self, name: str) -> bool:
        return name in self.column_names

    def __getitem__(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise ColumnNotFoundError(name, available=self.column_list)

    def get(self, name: str, default: Column | None = None) -> Column | None:
        for c in self.columns:
            if c.name == name:
                return c
        return default

    def column_by_position(self, pos: int) -> Column:
        for c in self.columns:
            if c.position == pos:
                return c
        raise SchemaError(f"No column at position {pos}")

    # -- Transformation methods (return new Schema) ---

    def add_column(self, col: Column) -> Schema:
        if col.name in self.column_names:
            raise DuplicateColumnError(col.name)
        new_col = col.with_position(len(self.columns))
        return attr.evolve(self, columns=self.columns + (new_col,))

    def drop_column(self, name: str) -> Schema:
        if name not in self.column_names:
            raise ColumnNotFoundError(name, available=self.column_list)
        new_cols = tuple(c for c in self.columns if c.name != name)
        # reindex positions
        new_cols = tuple(c.with_position(i) for i, c in enumerate(new_cols))
        # strip from primary key
        new_pk = tuple(k for k in self.primary_key if k != name)
        # strip from unique constraints
        new_ucs = tuple(
            tuple(c for c in uc if c != name)
            for uc in self.unique_constraints
        )
        new_ucs = tuple(uc for uc in new_ucs if uc)
        return attr.evolve(
            self,
            columns=new_cols,
            primary_key=new_pk,
            unique_constraints=new_ucs,
        )

    def rename_column(self, old_name: str, new_name: str) -> Schema:
        if old_name not in self.column_names:
            raise ColumnNotFoundError(old_name, available=self.column_list)
        if new_name in self.column_names:
            raise DuplicateColumnError(new_name)
        new_cols = tuple(
            c.with_name(new_name) if c.name == old_name else c
            for c in self.columns
        )
        new_pk = tuple(new_name if k == old_name else k for k in self.primary_key)
        new_ucs = tuple(
            tuple(new_name if c == old_name else c for c in uc)
            for uc in self.unique_constraints
        )
        return attr.evolve(
            self,
            columns=new_cols,
            primary_key=new_pk,
            unique_constraints=new_ucs,
        )

    def widen_column(self, name: str, new_type: ParameterisedType) -> Schema:
        col = self[name]
        TypeCompatibility.assert_compatible(col.sql_type, new_type, name)
        new_cols = tuple(
            c.with_type(new_type) if c.name == name else c
            for c in self.columns
        )
        return attr.evolve(self, columns=new_cols)

    def set_nullable(self, name: str, nullable: bool) -> Schema:
        _ = self[name]  # validate column exists
        new_cols = tuple(
            c.with_nullable(nullable) if c.name == name else c
            for c in self.columns
        )
        return attr.evolve(self, columns=new_cols)

    def project(self, names: Sequence[str]) -> Schema:
        """Return a schema containing only the specified columns (in order)."""
        result_cols: list[Column] = []
        for i, name in enumerate(names):
            result_cols.append(self[name].with_position(i))
        return Schema(columns=tuple(result_cols))

    def merge(self, other: Schema, prefix: str = "") -> Schema:
        """Merge another schema into this one, optionally prefixing new columns."""
        cols = list(self.columns)
        for c in other.columns:
            name = f"{prefix}{c.name}" if prefix else c.name
            if name in self.column_names:
                continue  # skip duplicates
            cols.append(c.with_name(name).with_position(len(cols)))
        return attr.evolve(self, columns=tuple(cols))

    def is_subschema_of(self, other: Schema) -> bool:
        """True if every column in *self* exists in *other* with a compatible type."""
        for col in self.columns:
            other_col = other.get(col.name)
            if other_col is None:
                return False
            if not TypeCompatibility.can_widen(col.sql_type, other_col.sql_type):
                return False
        return True

    def compatible_with(self, other: Schema) -> list[str]:
        """Return list of mismatched column names."""
        mismatched: list[str] = []
        for col in self.columns:
            other_col = other.get(col.name)
            if other_col is None:
                mismatched.append(col.name)
            elif TypeCompatibility.compare(col.sql_type, other_col.sql_type) == WideningResult.INCOMPATIBLE:
                mismatched.append(col.name)
        for col in other.columns:
            if col.name not in self.column_names:
                mismatched.append(col.name)
        return mismatched

    # -- Serialisation ---

    def __str__(self) -> str:
        lines = []
        header = self.table_name or "Schema"
        lines.append(f"{header}(")
        for c in self.columns:
            lines.append(f"  {c}")
        if self.primary_key:
            lines.append(f"  PRIMARY KEY ({', '.join(self.primary_key)})")
        lines.append(")")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "columns": [c.to_dict() for c in self.columns],
        }
        if self.primary_key:
            d["primary_key"] = list(self.primary_key)
        if self.unique_constraints:
            d["unique_constraints"] = [list(uc) for uc in self.unique_constraints]
        if self.foreign_keys:
            d["foreign_keys"] = [fk.to_dict() for fk in self.foreign_keys]
        if self.check_constraints:
            d["check_constraints"] = [cc.to_dict() for cc in self.check_constraints]
        if self.schema_name:
            d["schema_name"] = self.schema_name
        if self.table_name:
            d["table_name"] = self.table_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Schema:
        columns = tuple(Column.from_dict(c) for c in d.get("columns", []))
        pk = tuple(d.get("primary_key", []))
        ucs = tuple(tuple(uc) for uc in d.get("unique_constraints", []))
        fks = tuple(ForeignKey.from_dict(fk) for fk in d.get("foreign_keys", []))
        ccs = tuple(CheckConstraint.from_dict(cc) for cc in d.get("check_constraints", []))
        return cls(
            columns=columns,
            primary_key=pk,
            unique_constraints=ucs,
            foreign_keys=fks,
            check_constraints=ccs,
            schema_name=d.get("schema_name", ""),
            table_name=d.get("table_name", ""),
        )

    @classmethod
    def from_columns(cls, *cols: tuple[str, SQLType]) -> Schema:
        """Convenience: ``Schema.from_columns(("id", SQLType.INT), ("name", SQLType.TEXT))``."""
        return cls(
            columns=tuple(
                Column.quick(name, st, position=i)
                for i, (name, st) in enumerate(cols)
            )
        )

    @classmethod
    def empty(cls) -> Schema:
        return cls(columns=())


# =====================================================================
# Quality constraint
# =====================================================================

class Severity(enum.Enum):
    """Quality violation severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@attr.s(frozen=True, slots=True, hash=True, repr=True)
class QualityConstraint:
    """A quality invariant attached to a pipeline node or edge.

    Constraints are part of the Δ_Q sort in the three-sorted algebra.
    """

    constraint_id: str = attr.ib(validator=v.instance_of(str))
    predicate: str = attr.ib(validator=v.instance_of(str))
    severity: Severity = attr.ib(default=Severity.ERROR, validator=v.instance_of(Severity))
    severity_threshold: float = attr.ib(default=0.0, validator=v.instance_of(float))
    affected_columns: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    metric_name: str = attr.ib(default="", validator=v.instance_of(str))
    description: str = attr.ib(default="", validator=v.instance_of(str))
    enabled: bool = attr.ib(default=True, validator=v.instance_of(bool))

    def __str__(self) -> str:
        cols = f" on ({', '.join(self.affected_columns)})" if self.affected_columns else ""
        return f"QC[{self.constraint_id}]: {self.predicate}{cols} [{self.severity.value}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "predicate": self.predicate,
            "severity": self.severity.value,
            "severity_threshold": self.severity_threshold,
            "affected_columns": list(self.affected_columns),
            "metric_name": self.metric_name,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QualityConstraint:
        return cls(
            constraint_id=d["constraint_id"],
            predicate=d["predicate"],
            severity=Severity(d.get("severity", "error")),
            severity_threshold=d.get("severity_threshold", 0.0),
            affected_columns=tuple(d.get("affected_columns", [])),
            metric_name=d.get("metric_name", ""),
            description=d.get("description", ""),
            enabled=d.get("enabled", True),
        )

    @classmethod
    def not_null(cls, constraint_id: str, *columns: str) -> QualityConstraint:
        return cls(
            constraint_id=constraint_id,
            predicate="NOT NULL",
            affected_columns=columns,
            metric_name="null_fraction",
            severity=Severity.ERROR,
        )

    @classmethod
    def range_check(
        cls,
        constraint_id: str,
        column: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> QualityConstraint:
        parts: list[str] = []
        if min_val is not None:
            parts.append(f"{column} >= {min_val}")
        if max_val is not None:
            parts.append(f"{column} <= {max_val}")
        return cls(
            constraint_id=constraint_id,
            predicate=" AND ".join(parts) if parts else "TRUE",
            affected_columns=(column,),
            metric_name="range_violation_fraction",
            severity=Severity.ERROR,
        )

    @classmethod
    def uniqueness(cls, constraint_id: str, *columns: str) -> QualityConstraint:
        return cls(
            constraint_id=constraint_id,
            predicate=f"UNIQUE({', '.join(columns)})",
            affected_columns=columns,
            metric_name="duplicate_fraction",
            severity=Severity.ERROR,
        )

    @classmethod
    def freshness(cls, constraint_id: str, column: str, max_staleness_hours: float) -> QualityConstraint:
        return cls(
            constraint_id=constraint_id,
            predicate=f"MAX_AGE({column}) <= {max_staleness_hours}h",
            affected_columns=(column,),
            metric_name="max_staleness_hours",
            severity_threshold=float(max_staleness_hours),
            severity=Severity.WARNING,
        )

    @classmethod
    def row_count(cls, constraint_id: str, min_rows: int = 0, max_rows: int | None = None) -> QualityConstraint:
        parts: list[str] = []
        if min_rows > 0:
            parts.append(f"COUNT(*) >= {min_rows}")
        if max_rows is not None:
            parts.append(f"COUNT(*) <= {max_rows}")
        return cls(
            constraint_id=constraint_id,
            predicate=" AND ".join(parts) if parts else "TRUE",
            metric_name="row_count",
            severity=Severity.WARNING,
        )

    @classmethod
    def distribution(
        cls,
        constraint_id: str,
        column: str,
        test: str = "ks",
        threshold: float = 0.05,
    ) -> QualityConstraint:
        return cls(
            constraint_id=constraint_id,
            predicate=f"DISTRIBUTION_TEST({column}, '{test}') > {threshold}",
            affected_columns=(column,),
            metric_name=f"{test}_p_value",
            severity_threshold=threshold,
            severity=Severity.WARNING,
        )


# =====================================================================
# Availability contract
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class AvailabilityContract:
    """SLA and availability requirements for a pipeline node.

    Part of the system's availability-aware repair planning.
    """

    sla_percentage: float = attr.ib(default=99.0, validator=v.instance_of(float))
    max_downtime: timedelta = attr.ib(factory=lambda: timedelta(hours=1))
    staleness_tolerance: timedelta = attr.ib(factory=lambda: timedelta(hours=24))
    priority: int = attr.ib(default=0, validator=v.instance_of(int))
    description: str = attr.ib(default="", validator=v.instance_of(str))

    def __attrs_post_init__(self) -> None:
        if not (0.0 <= self.sla_percentage <= 100.0):
            raise SchemaError(
                f"SLA percentage must be in [0, 100], got {self.sla_percentage}",
                code=ErrorCode.SCHEMA_INVALID,
            )

    def __str__(self) -> str:
        return (
            f"Availability(sla={self.sla_percentage:.2f}%, "
            f"max_down={self.max_downtime}, "
            f"staleness={self.staleness_tolerance})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sla_percentage": self.sla_percentage,
            "max_downtime_seconds": self.max_downtime.total_seconds(),
            "staleness_tolerance_seconds": self.staleness_tolerance.total_seconds(),
            "priority": self.priority,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AvailabilityContract:
        return cls(
            sla_percentage=d.get("sla_percentage", 99.0),
            max_downtime=timedelta(seconds=d.get("max_downtime_seconds", 3600)),
            staleness_tolerance=timedelta(seconds=d.get("staleness_tolerance_seconds", 86400)),
            priority=d.get("priority", 0),
            description=d.get("description", ""),
        )

    @classmethod
    def critical(cls) -> AvailabilityContract:
        return cls(
            sla_percentage=99.99,
            max_downtime=timedelta(minutes=5),
            staleness_tolerance=timedelta(minutes=15),
            priority=100,
            description="Critical SLA",
        )

    @classmethod
    def standard(cls) -> AvailabilityContract:
        return cls(
            sla_percentage=99.0,
            max_downtime=timedelta(hours=1),
            staleness_tolerance=timedelta(hours=24),
            priority=50,
            description="Standard SLA",
        )

    @classmethod
    def best_effort(cls) -> AvailabilityContract:
        return cls(
            sla_percentage=95.0,
            max_downtime=timedelta(hours=8),
            staleness_tolerance=timedelta(days=7),
            priority=10,
            description="Best effort",
        )

    def meets_sla(self, actual_percentage: float) -> bool:
        return actual_percentage >= self.sla_percentage

    def within_staleness(self, actual_staleness: timedelta) -> bool:
        return actual_staleness <= self.staleness_tolerance

    def within_downtime(self, actual_downtime: timedelta) -> bool:
        return actual_downtime <= self.max_downtime


# =====================================================================
# Cost estimate
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class CostEstimate:
    """Estimated cost to execute (or re-execute) a pipeline node.

    Used by the repair planner to compute cost-optimal plans.
    """

    compute_seconds: float = attr.ib(default=0.0, validator=v.instance_of(float))
    memory_bytes: int = attr.ib(default=0, validator=v.instance_of(int))
    io_bytes: int = attr.ib(default=0, validator=v.instance_of(int))
    row_estimate: int = attr.ib(default=0, validator=v.instance_of(int))
    monetary_cost: float = attr.ib(default=0.0, validator=v.instance_of(float))
    confidence: float = attr.ib(default=0.5, validator=v.instance_of(float))

    @property
    def total_weighted_cost(self) -> float:
        """Scalar cost combining compute, memory, and I/O."""
        # normalise to seconds-equivalent
        compute = self.compute_seconds
        memory = self.memory_bytes / (1024 ** 3)  # GB-seconds approximation
        io = self.io_bytes / (100 * 1024 ** 2)  # 100 MB/s normalisation
        return compute + 0.1 * memory + io + self.monetary_cost

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            compute_seconds=self.compute_seconds + other.compute_seconds,
            memory_bytes=self.memory_bytes + other.memory_bytes,
            io_bytes=self.io_bytes + other.io_bytes,
            row_estimate=self.row_estimate + other.row_estimate,
            monetary_cost=self.monetary_cost + other.monetary_cost,
            confidence=min(self.confidence, other.confidence),
        )

    def scale(self, factor: float) -> CostEstimate:
        return CostEstimate(
            compute_seconds=self.compute_seconds * factor,
            memory_bytes=int(self.memory_bytes * factor),
            io_bytes=int(self.io_bytes * factor),
            row_estimate=int(self.row_estimate * factor),
            monetary_cost=self.monetary_cost * factor,
            confidence=self.confidence,
        )

    def __str__(self) -> str:
        return (
            f"Cost(compute={self.compute_seconds:.2f}s, "
            f"mem={self.memory_bytes / (1024**2):.1f}MB, "
            f"io={self.io_bytes / (1024**2):.1f}MB, "
            f"rows={self.row_estimate}, "
            f"${self.monetary_cost:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_seconds": self.compute_seconds,
            "memory_bytes": self.memory_bytes,
            "io_bytes": self.io_bytes,
            "row_estimate": self.row_estimate,
            "monetary_cost": self.monetary_cost,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CostEstimate:
        return cls(
            compute_seconds=d.get("compute_seconds", 0.0),
            memory_bytes=d.get("memory_bytes", 0),
            io_bytes=d.get("io_bytes", 0),
            row_estimate=d.get("row_estimate", 0),
            monetary_cost=d.get("monetary_cost", 0.0),
            confidence=d.get("confidence", 0.5),
        )

    @classmethod
    def zero(cls) -> CostEstimate:
        return cls()

    @classmethod
    def unknown(cls) -> CostEstimate:
        return cls(confidence=0.0)


# =====================================================================
# Node / edge metadata
# =====================================================================

@attr.s(frozen=True, slots=True, hash=True, repr=True)
class NodeMetadata:
    """Metadata attached to a pipeline node."""

    owner: str = attr.ib(default="", validator=v.instance_of(str))
    tags: tuple[str, ...] = attr.ib(factory=tuple, converter=tuple)
    source_file: str = attr.ib(default="", validator=v.instance_of(str))
    source_line: int = attr.ib(default=0, validator=v.instance_of(int))
    dialect: str = attr.ib(default="", validator=v.instance_of(str))
    created_at: str = attr.ib(default="", validator=v.instance_of(str))
    updated_at: str = attr.ib(default="", validator=v.instance_of(str))
    custom: dict[str, Any] = attr.ib(factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.owner:
            d["owner"] = self.owner
        if self.tags:
            d["tags"] = list(self.tags)
        if self.source_file:
            d["source_file"] = self.source_file
        if self.source_line:
            d["source_line"] = self.source_line
        if self.dialect:
            d["dialect"] = self.dialect
        if self.created_at:
            d["created_at"] = self.created_at
        if self.updated_at:
            d["updated_at"] = self.updated_at
        if self.custom:
            d["custom"] = self.custom
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeMetadata:
        return cls(
            owner=d.get("owner", ""),
            tags=tuple(d.get("tags", [])),
            source_file=d.get("source_file", ""),
            source_line=d.get("source_line", 0),
            dialect=d.get("dialect", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            custom=d.get("custom", {}),
        )


class EdgeType(enum.Enum):
    """Classification of edges in the pipeline DAG."""
    DATA_FLOW = "data_flow"
    SCHEMA_DEPENDENCY = "schema_dependency"
    QUALITY_DEPENDENCY = "quality_dependency"
    CONTROL_FLOW = "control_flow"
    TEMPORAL = "temporal"


# =====================================================================
# SQL operators (local definitions for the planner / execution layer)
# =====================================================================

class SQLOperator(enum.Enum):
    """High-level SQL operator taxonomy used in the pipeline DAG."""
    SELECT = "SELECT"
    FILTER = "FILTER"
    JOIN = "JOIN"
    GROUP_BY = "GROUP_BY"
    ORDER_BY = "ORDER_BY"
    UNION = "UNION"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"
    WINDOW = "WINDOW"
    DISTINCT = "DISTINCT"
    LIMIT = "LIMIT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE_TABLE = "CREATE_TABLE"
    ALTER_TABLE = "ALTER_TABLE"
    DROP_TABLE = "DROP_TABLE"
    CUSTOM = "CUSTOM"

    def is_read_only(self) -> bool:
        return self in {
            SQLOperator.SELECT, SQLOperator.FILTER, SQLOperator.JOIN,
            SQLOperator.GROUP_BY, SQLOperator.ORDER_BY, SQLOperator.UNION,
            SQLOperator.INTERSECT, SQLOperator.EXCEPT, SQLOperator.WINDOW,
            SQLOperator.DISTINCT, SQLOperator.LIMIT,
        }

    def is_ddl(self) -> bool:
        return self in {
            SQLOperator.CREATE_TABLE, SQLOperator.ALTER_TABLE,
            SQLOperator.DROP_TABLE,
        }

    def is_dml(self) -> bool:
        return self in {SQLOperator.INSERT, SQLOperator.UPDATE, SQLOperator.DELETE}


class JoinType(enum.Enum):
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    SEMI = "SEMI"
    ANTI = "ANTI"


class AggregateFunction(enum.Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"
    MEDIAN = "MEDIAN"
    PERCENTILE = "PERCENTILE"
    ARRAY_AGG = "ARRAY_AGG"
    STRING_AGG = "STRING_AGG"
    FIRST = "FIRST"
    LAST = "LAST"
    ANY_VALUE = "ANY_VALUE"


class WindowFrameType(enum.Enum):
    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


# =====================================================================
# Operator configurations
# =====================================================================

@attr.s(frozen=True, slots=True, auto_attribs=True)
class OperatorConfig:
    """Base operator configuration."""
    operator: SQLOperator = SQLOperator.CUSTOM
    sql_text: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SelectConfig(OperatorConfig):
    """Configuration for SELECT projections."""
    columns: tuple[str, ...] = attr.Factory(tuple)
    expressions: dict[str, str] = attr.Factory(dict)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.SELECT)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class FilterConfig(OperatorConfig):
    """Configuration for FILTER (WHERE) operations."""
    predicate: str = ""
    columns_referenced: tuple[str, ...] = attr.Factory(tuple)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.FILTER)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class JoinConfig(OperatorConfig):
    """Configuration for JOIN operations."""
    join_type: JoinType = JoinType.INNER
    left_keys: tuple[str, ...] = attr.Factory(tuple)
    right_keys: tuple[str, ...] = attr.Factory(tuple)
    condition: str = ""
    use_hash: bool = True

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.JOIN)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class GroupByConfig(OperatorConfig):
    """Configuration for GROUP BY aggregation."""
    group_columns: tuple[str, ...] = attr.Factory(tuple)
    aggregates: dict[str, AggregateFunction] = attr.Factory(dict)
    having_predicate: str = ""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.GROUP_BY)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class UnionConfig(OperatorConfig):
    """Configuration for UNION operations."""
    union_all: bool = True

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.UNION)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class WindowConfig(OperatorConfig):
    """Configuration for WINDOW functions."""
    partition_columns: tuple[str, ...] = attr.Factory(tuple)
    order_columns: tuple[str, ...] = attr.Factory(tuple)
    frame_type: WindowFrameType = WindowFrameType.ROWS
    frame_start: int | None = None
    frame_end: int | None = None
    function_name: str = ""
    output_column: str = ""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operator", SQLOperator.WINDOW)


# =====================================================================
# Pipeline DAG
# =====================================================================

@attr.s(frozen=True, slots=True, auto_attribs=True)
class PipelineNode:
    """A single node in the pipeline DAG."""
    node_id: str
    operator: SQLOperator = SQLOperator.CUSTOM
    operator_config: OperatorConfig | None = None
    input_schema: Schema | None = None
    output_schema: Schema | None = None
    sql_text: str = ""
    table_name: str = ""
    estimated_row_count: int = 0
    metadata: dict[str, Any] = attr.Factory(dict)
    is_source: bool = False
    is_sink: bool = False


@attr.s(frozen=True, slots=True, auto_attribs=True)
class PipelineEdge:
    """A directed edge in the pipeline DAG."""
    source: str
    target: str
    columns_referenced: tuple[str, ...] = attr.Factory(tuple)
    edge_type: str = "data"
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(slots=True, auto_attribs=True)
class PipelineGraph:
    """A directed (acyclic or general) graph of :class:`PipelineNode`.

    Provides topological ordering, reachability, and structural queries
    needed by the planner and execution engine.
    """
    nodes: dict[str, PipelineNode] = attr.Factory(dict)
    edges: list[PipelineEdge] = attr.Factory(list)
    metadata: dict[str, Any] = attr.Factory(dict)

    def add_node(self, node: PipelineNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: PipelineEdge) -> None:
        self.edges.append(edge)

    @property
    def adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adj.setdefault(e.source, []).append(e.target)
        return adj

    @property
    def reverse_adjacency(self) -> dict[str, list[str]]:
        rev: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            rev.setdefault(e.target, []).append(e.source)
        return rev

    def parents(self, node_id: str) -> list[str]:
        return [e.source for e in self.edges if e.target == node_id]

    def children(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def sources(self) -> list[str]:
        targets = {e.target for e in self.edges}
        return [nid for nid in self.nodes if nid not in targets]

    def sinks(self) -> list[str]:
        srcs = {e.source for e in self.edges}
        return [nid for nid in self.nodes if nid not in srcs]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm.  Raises *ValueError* on cycle."""
        in_deg: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj = self.adjacency
        for e in self.edges:
            in_deg[e.target] = in_deg.get(e.target, 0) + 1
        queue = sorted(nid for nid, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in adj.get(n, []):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
                    queue.sort()
        if len(order) != len(self.nodes):
            raise ValueError("Pipeline graph contains a cycle")
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except ValueError:
            return False

    def reachable_from(self, node_id: str) -> set[str]:
        visited: set[str] = set()
        stack = [node_id]
        adj = self.adjacency
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            stack.extend(adj.get(n, []))
        return visited

    def ancestors_of(self, node_id: str) -> set[str]:
        visited: set[str] = set()
        stack = [node_id]
        rev = self.reverse_adjacency
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            stack.extend(rev.get(n, []))
        return visited

    def subgraph(self, node_ids: set[str]) -> "PipelineGraph":
        nodes = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        edges = [e for e in self.edges if e.source in node_ids and e.target in node_ids]
        return PipelineGraph(nodes=nodes, edges=edges, metadata=self.metadata)

    def get_edge(self, source: str, target: str) -> PipelineEdge | None:
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


# =====================================================================
# Three-sorted delta algebra types
# =====================================================================

class SchemaOpType(enum.Enum):
    """Types of atomic schema operations."""
    ADD_COLUMN = "ADD_COLUMN"
    DROP_COLUMN = "DROP_COLUMN"
    RENAME_COLUMN = "RENAME_COLUMN"
    RETYPE_COLUMN = "RETYPE_COLUMN"
    ADD_CONSTRAINT = "ADD_CONSTRAINT"
    DROP_CONSTRAINT = "DROP_CONSTRAINT"
    SET_NULLABLE = "SET_NULLABLE"
    SET_DEFAULT = "SET_DEFAULT"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SchemaOperation:
    """A single atomic schema change."""
    op_type: SchemaOpType
    column_name: str = ""
    new_column_name: str = ""
    dtype: SQLType | None = None
    new_dtype: SQLType | None = None
    nullable: bool | None = None
    default_val: Any = None
    constraint: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)

    def columns_affected(self) -> set[str]:
        cols: set[str] = set()
        if self.column_name:
            cols.add(self.column_name)
        if self.new_column_name:
            cols.add(self.new_column_name)
        return cols


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SchemaDelta:
    """Schema delta Δ_S: an ordered sequence of schema operations.

    Composition: (δ₁ ∘ δ₂) applies δ₁ first, then δ₂.
    Inversion: δ⁻¹ undoes the delta (when possible).
    """
    operations: tuple[SchemaOperation, ...] = attr.Factory(tuple)
    source_node: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def is_identity(self) -> bool:
        return len(self.operations) == 0

    @property
    def columns_affected(self) -> set[str]:
        result: set[str] = set()
        for op in self.operations:
            result |= op.columns_affected()
        return result

    def compose(self, other: "SchemaDelta") -> "SchemaDelta":
        return SchemaDelta(
            operations=self.operations + other.operations,
            source_node=self.source_node or other.source_node,
        )

    def invert(self) -> "SchemaDelta":
        inv_ops: list[SchemaOperation] = []
        for op in reversed(self.operations):
            if op.op_type == SchemaOpType.ADD_COLUMN:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.DROP_COLUMN,
                    column_name=op.column_name,
                ))
            elif op.op_type == SchemaOpType.DROP_COLUMN:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.ADD_COLUMN,
                    column_name=op.column_name,
                    dtype=op.dtype,
                    nullable=op.nullable,
                    default_val=op.default_val,
                ))
            elif op.op_type == SchemaOpType.RENAME_COLUMN:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.RENAME_COLUMN,
                    column_name=op.new_column_name,
                    new_column_name=op.column_name,
                ))
            elif op.op_type == SchemaOpType.RETYPE_COLUMN:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.RETYPE_COLUMN,
                    column_name=op.column_name,
                    dtype=op.new_dtype,
                    new_dtype=op.dtype,
                ))
            elif op.op_type == SchemaOpType.SET_NULLABLE:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.SET_NULLABLE,
                    column_name=op.column_name,
                    nullable=not op.nullable if op.nullable is not None else None,
                ))
            elif op.op_type == SchemaOpType.ADD_CONSTRAINT:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.DROP_CONSTRAINT,
                    column_name=op.column_name,
                    constraint=op.constraint,
                ))
            elif op.op_type == SchemaOpType.DROP_CONSTRAINT:
                inv_ops.append(SchemaOperation(
                    op_type=SchemaOpType.ADD_CONSTRAINT,
                    column_name=op.column_name,
                    constraint=op.constraint,
                ))
            else:
                inv_ops.append(op)
        return SchemaDelta(operations=tuple(inv_ops), source_node=self.source_node)

    def apply_to(self, schema: Schema) -> Schema:
        """Apply this delta to a schema (best-effort)."""
        result = schema
        for op in self.operations:
            if op.op_type == SchemaOpType.ADD_COLUMN:
                col = Column.quick(
                    op.column_name,
                    op.dtype if op.dtype is not None else SQLType.TEXT,
                    nullable=op.nullable if op.nullable is not None else True,
                    position=len(result.columns),
                )
                try:
                    result = result.add_column(col)
                except Exception:
                    pass
            elif op.op_type == SchemaOpType.DROP_COLUMN:
                try:
                    result = result.drop_column(op.column_name)
                except Exception:
                    pass
            elif op.op_type == SchemaOpType.RENAME_COLUMN:
                try:
                    result = result.rename_column(op.column_name, op.new_column_name)
                except Exception:
                    pass
        return result


# -- Data deltas (Δ_D) ---

class RowChangeType(enum.Enum):
    INSERT = "INSERT"
    DELETE = "DELETE"
    UPDATE = "UPDATE"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RowChange:
    """A single row-level data change."""
    change_type: RowChangeType
    row_key: tuple[Any, ...] = attr.Factory(tuple)
    old_values: dict[str, Any] = attr.Factory(dict)
    new_values: dict[str, Any] = attr.Factory(dict)
    columns_affected: tuple[str, ...] = attr.Factory(tuple)

    def invert(self) -> "RowChange":
        if self.change_type == RowChangeType.INSERT:
            return RowChange(
                change_type=RowChangeType.DELETE,
                row_key=self.row_key,
                old_values=self.new_values,
            )
        elif self.change_type == RowChangeType.DELETE:
            return RowChange(
                change_type=RowChangeType.INSERT,
                row_key=self.row_key,
                new_values=self.old_values,
            )
        else:
            return RowChange(
                change_type=RowChangeType.UPDATE,
                row_key=self.row_key,
                old_values=self.new_values,
                new_values=self.old_values,
                columns_affected=self.columns_affected,
            )


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DataDelta:
    """Data delta Δ_D: a batch of row-level changes."""
    changes: tuple[RowChange, ...] = attr.Factory(tuple)
    source_node: str = ""
    affected_columns: frozenset[str] = attr.Factory(frozenset)
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def is_identity(self) -> bool:
        return len(self.changes) == 0

    @property
    def insert_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == RowChangeType.INSERT)

    @property
    def delete_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == RowChangeType.DELETE)

    @property
    def update_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == RowChangeType.UPDATE)

    @property
    def total_changes(self) -> int:
        return len(self.changes)

    @property
    def columns_affected(self) -> set[str]:
        result: set[str] = set(self.affected_columns)
        for ch in self.changes:
            result.update(ch.columns_affected)
            result.update(ch.old_values.keys())
            result.update(ch.new_values.keys())
        return result

    def compose(self, other: "DataDelta") -> "DataDelta":
        """Compose two data deltas, merging changes by row key."""
        key_to_change: dict[tuple[Any, ...], RowChange] = {}
        for ch in self.changes:
            key_to_change[ch.row_key] = ch
        for ch in other.changes:
            existing = key_to_change.get(ch.row_key)
            if existing is None:
                key_to_change[ch.row_key] = ch
            elif existing.change_type == RowChangeType.INSERT and ch.change_type == RowChangeType.DELETE:
                del key_to_change[ch.row_key]
            elif existing.change_type == RowChangeType.INSERT and ch.change_type == RowChangeType.UPDATE:
                merged = dict(existing.new_values)
                merged.update(ch.new_values)
                key_to_change[ch.row_key] = RowChange(
                    change_type=RowChangeType.INSERT,
                    row_key=ch.row_key,
                    new_values=merged,
                )
            elif existing.change_type == RowChangeType.UPDATE and ch.change_type == RowChangeType.UPDATE:
                merged_old = dict(existing.old_values)
                merged_new = dict(existing.new_values)
                merged_new.update(ch.new_values)
                for k in ch.old_values:
                    if k not in merged_old:
                        merged_old[k] = ch.old_values[k]
                key_to_change[ch.row_key] = RowChange(
                    change_type=RowChangeType.UPDATE,
                    row_key=ch.row_key,
                    old_values=merged_old,
                    new_values=merged_new,
                    columns_affected=tuple(
                        set(existing.columns_affected) | set(ch.columns_affected)
                    ),
                )
            elif existing.change_type == RowChangeType.UPDATE and ch.change_type == RowChangeType.DELETE:
                key_to_change[ch.row_key] = RowChange(
                    change_type=RowChangeType.DELETE,
                    row_key=ch.row_key,
                    old_values=existing.old_values,
                )
            elif existing.change_type == RowChangeType.DELETE and ch.change_type == RowChangeType.INSERT:
                if existing.old_values == ch.new_values:
                    del key_to_change[ch.row_key]
                else:
                    key_to_change[ch.row_key] = RowChange(
                        change_type=RowChangeType.UPDATE,
                        row_key=ch.row_key,
                        old_values=existing.old_values,
                        new_values=ch.new_values,
                    )
            else:
                key_to_change[ch.row_key] = ch
        all_cols = self.affected_columns | other.affected_columns
        return DataDelta(
            changes=tuple(key_to_change.values()),
            source_node=self.source_node or other.source_node,
            affected_columns=frozenset(all_cols),
        )

    def invert(self) -> "DataDelta":
        return DataDelta(
            changes=tuple(ch.invert() for ch in reversed(self.changes)),
            source_node=self.source_node,
            affected_columns=self.affected_columns,
        )


# -- Quality deltas (Δ_Q) ---

@attr.s(frozen=True, slots=True, auto_attribs=True)
class QualityMetricChange:
    """A change in a single quality metric."""
    metric_name: str
    old_value: float
    new_value: float
    column: str = ""
    threshold: float | None = None
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def delta_value(self) -> float:
        return self.new_value - self.old_value

    @property
    def exceeds_threshold(self) -> bool:
        if self.threshold is None:
            return False
        return abs(self.delta_value) > self.threshold

    def invert(self) -> "QualityMetricChange":
        return QualityMetricChange(
            metric_name=self.metric_name,
            old_value=self.new_value,
            new_value=self.old_value,
            column=self.column,
            threshold=self.threshold,
            metadata=self.metadata,
        )


@attr.s(frozen=True, slots=True, auto_attribs=True)
class QualityDelta:
    """Quality delta Δ_Q: changes in quality metrics and constraints."""
    metric_changes: tuple[QualityMetricChange, ...] = attr.Factory(tuple)
    constraint_violations: tuple[str, ...] = attr.Factory(tuple)
    source_node: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def is_identity(self) -> bool:
        return len(self.metric_changes) == 0 and len(self.constraint_violations) == 0

    @property
    def has_violations(self) -> bool:
        return len(self.constraint_violations) > 0

    def compose(self, other: "QualityDelta") -> "QualityDelta":
        metric_map: dict[tuple[str, str], QualityMetricChange] = {}
        for m in self.metric_changes:
            metric_map[(m.metric_name, m.column)] = m
        for m in other.metric_changes:
            key = (m.metric_name, m.column)
            existing = metric_map.get(key)
            if existing is not None:
                metric_map[key] = QualityMetricChange(
                    metric_name=m.metric_name,
                    old_value=existing.old_value,
                    new_value=m.new_value,
                    column=m.column,
                    threshold=m.threshold or existing.threshold,
                )
            else:
                metric_map[key] = m
        violations = set(self.constraint_violations) | set(other.constraint_violations)
        return QualityDelta(
            metric_changes=tuple(metric_map.values()),
            constraint_violations=tuple(sorted(violations)),
            source_node=self.source_node or other.source_node,
        )

    def invert(self) -> "QualityDelta":
        return QualityDelta(
            metric_changes=tuple(m.invert() for m in self.metric_changes),
            constraint_violations=(),
            source_node=self.source_node,
        )


# -- Compound perturbation ---

@attr.s(frozen=True, slots=True, auto_attribs=True)
class CompoundPerturbation:
    """A compound perturbation bundling schema, data, and quality deltas."""
    schema_delta: SchemaDelta = attr.Factory(SchemaDelta)
    data_delta: DataDelta = attr.Factory(DataDelta)
    quality_delta: QualityDelta = attr.Factory(QualityDelta)
    source_node: str = ""

    @property
    def is_identity(self) -> bool:
        return (
            self.schema_delta.is_identity
            and self.data_delta.is_identity
            and self.quality_delta.is_identity
        )

    @property
    def columns_affected(self) -> set[str]:
        return self.schema_delta.columns_affected | self.data_delta.columns_affected

    @property
    def has_schema_change(self) -> bool:
        return not self.schema_delta.is_identity

    @property
    def has_data_change(self) -> bool:
        return not self.data_delta.is_identity

    @property
    def has_quality_change(self) -> bool:
        return not self.quality_delta.is_identity

    def compose(self, other: "CompoundPerturbation") -> "CompoundPerturbation":
        return CompoundPerturbation(
            schema_delta=self.schema_delta.compose(other.schema_delta),
            data_delta=self.data_delta.compose(other.data_delta),
            quality_delta=self.quality_delta.compose(other.quality_delta),
            source_node=self.source_node or other.source_node,
        )

    def invert(self) -> "CompoundPerturbation":
        return CompoundPerturbation(
            schema_delta=self.schema_delta.invert(),
            data_delta=self.data_delta.invert(),
            quality_delta=self.quality_delta.invert(),
            source_node=self.source_node,
        )


# =====================================================================
# Repair planning types
# =====================================================================

class ActionType(enum.Enum):
    """The kind of repair action to perform at a node."""
    RECOMPUTE = "RECOMPUTE"
    INCREMENTAL_UPDATE = "INCREMENTAL_UPDATE"
    SCHEMA_MIGRATE = "SCHEMA_MIGRATE"
    SKIP = "SKIP"
    CHECKPOINT = "CHECKPOINT"
    ROLLBACK = "ROLLBACK"
    VALIDATE = "VALIDATE"
    NO_OP = "NO_OP"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RepairAction:
    """A single repair action in a repair plan."""
    node_id: str
    action_type: ActionType
    estimated_cost: float = 0.0
    dependencies: tuple[str, ...] = attr.Factory(tuple)
    delta_to_apply: CompoundPerturbation | None = None
    sql_text: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)
    action_id: str = attr.Factory(lambda: __import__("uuid").uuid4().hex[:12])

    @property
    def is_noop(self) -> bool:
        return self.action_type in {ActionType.SKIP, ActionType.NO_OP}


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CostBreakdown:
    """Itemized cost report for a repair plan."""
    compute_cost: float = 0.0
    io_cost: float = 0.0
    materialization_cost: float = 0.0
    network_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_node: dict[str, float] = attr.Factory(dict)
    savings_vs_full_recompute: float = 0.0


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RepairPlan:
    """A complete repair plan: a set of actions with execution order."""
    actions: tuple[RepairAction, ...] = attr.Factory(tuple)
    execution_order: tuple[str, ...] = attr.Factory(tuple)
    total_cost: float = 0.0
    full_recompute_cost: float = 0.0
    savings_ratio: float = 0.0
    affected_nodes: frozenset[str] = attr.Factory(frozenset)
    annihilated_nodes: frozenset[str] = attr.Factory(frozenset)
    plan_metadata: dict[str, Any] = attr.Factory(dict)
    cost_breakdown: CostBreakdown | None = None

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def non_trivial_actions(self) -> list[RepairAction]:
        return [a for a in self.actions if not a.is_noop]

    def get_action(self, node_id: str) -> RepairAction | None:
        for a in self.actions:
            if a.node_id == node_id:
                return a
        return None

    def get_actions_for_type(self, action_type: ActionType) -> list[RepairAction]:
        return [a for a in self.actions if a.action_type == action_type]


# =====================================================================
# Execution types
# =====================================================================

@attr.s(frozen=True, slots=True, auto_attribs=True)
class TableStats:
    """Runtime statistics for a table."""
    table_name: str
    row_count: int = 0
    column_count: int = 0
    size_bytes: int = 0
    null_counts: dict[str, int] = attr.Factory(dict)
    distinct_counts: dict[str, int] = attr.Factory(dict)
    min_values: dict[str, Any] = attr.Factory(dict)
    max_values: dict[str, Any] = attr.Factory(dict)

    @property
    def avg_row_bytes(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.size_bytes / self.row_count


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ExecutionError:
    """A single error that occurred during execution."""
    node_id: str
    action_id: str = ""
    error_type: str = ""
    message: str = ""
    recoverable: bool = False
    context: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ActionResult:
    """Result of executing a single repair action."""
    action_id: str
    node_id: str
    success: bool
    elapsed_seconds: float = 0.0
    rows_affected: int = 0
    error: ExecutionError | None = None
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SchemaViolation:
    """A schema consistency violation."""
    node_id: str
    violation_type: str
    message: str
    column: str = ""
    expected: str = ""
    actual: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class QualityViolation:
    """A quality constraint violation."""
    node_id: str
    constraint_name: str
    message: str
    metric_value: float = 0.0
    threshold: float = 0.0
    column: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class NodeValidation:
    """Validation result for a single node."""
    node_id: str
    is_valid: bool
    exact_match: bool = False
    error_bound: float | None = None
    actual_error: float | None = None
    message: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ValidationResult:
    """Result of validating repair correctness."""
    is_valid: bool
    exact_match: bool = False
    error_bound: float | None = None
    actual_error: float | None = None
    schema_violations: tuple[SchemaViolation, ...] = attr.Factory(tuple)
    quality_violations: tuple[QualityViolation, ...] = attr.Factory(tuple)
    per_node_results: dict[str, NodeValidation] = attr.Factory(dict)
    message: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ExecutionResult:
    """Aggregate result of executing a repair plan."""
    success: bool
    actions_executed: tuple[ActionResult, ...] = attr.Factory(tuple)
    total_time_seconds: float = 0.0
    rows_processed: int = 0
    errors: tuple[ExecutionError, ...] = attr.Factory(tuple)
    validation_result: ValidationResult | None = None
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CheckpointInfo:
    """Metadata about a stored checkpoint."""
    checkpoint_id: str
    created_at: float = 0.0
    tables_saved: tuple[str, ...] = attr.Factory(tuple)
    size_bytes: int = 0
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ResourceSpec:
    """Resource constraints for scheduling."""
    max_parallelism: int = 4
    max_memory_bytes: int = 0
    max_cpu_cores: int = 0
    timeout_seconds: float = 0.0


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ExecutionSchedule:
    """A schedule of repair actions organized into parallel waves."""
    waves: tuple[tuple[RepairAction, ...], ...] = attr.Factory(tuple)
    estimated_wall_time: float = 0.0
    estimated_total_time: float = 0.0
    critical_path_length: float = 0.0
    parallelism_factor: float = 1.0

    @property
    def wave_count(self) -> int:
        return len(self.waves)

    @property
    def total_actions(self) -> int:
        return sum(len(w) for w in self.waves)


# =====================================================================
# Quality / profiling types
# =====================================================================

@attr.s(frozen=True, slots=True, auto_attribs=True)
class Violation:
    """A single constraint violation."""
    constraint_name: str
    constraint_type: str = ""
    column: str = ""
    message: str = ""
    row_count: int = 0
    sample_values: tuple[Any, ...] = attr.Factory(tuple)
    severity: str = "error"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CheckResult:
    """Result of a single quality check."""
    check_name: str
    passed: bool
    metric_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    details: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintResult:
    """Result of evaluating a single constraint."""
    constraint_name: str
    passed: bool
    violations: tuple[Violation, ...] = attr.Factory(tuple)
    metric_value: float = 0.0
    message: str = ""
    execution_time_seconds: float = 0.0


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintSuggestion:
    """A suggested constraint based on data analysis."""
    constraint_id: str
    predicate: str
    confidence: float = 0.0
    reason: str = ""
    sample_support: int = 0


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ColumnProfile:
    """Statistical profile of a single column."""
    column_name: str
    dtype: str = ""
    count: int = 0
    null_count: int = 0
    unique_count: int = 0
    mean: float | None = None
    std: float | None = None
    min_val: Any = None
    max_val: Any = None
    percentiles: dict[int, float] = attr.Factory(dict)
    histogram_counts: tuple[float, ...] = attr.Factory(tuple)
    histogram_edges: tuple[float, ...] = attr.Factory(tuple)
    most_common: tuple[tuple[Any, int], ...] = attr.Factory(tuple)

    @property
    def null_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.null_count / self.count

    @property
    def uniqueness(self) -> float:
        non_null = self.count - self.null_count
        if non_null == 0:
            return 0.0
        return self.unique_count / non_null

    @property
    def completeness(self) -> float:
        return 1.0 - self.null_rate


@attr.s(frozen=True, slots=True, auto_attribs=True)
class TableProfile:
    """Comprehensive profile of a table."""
    table_name: str
    row_count: int = 0
    column_count: int = 0
    column_profiles: dict[str, ColumnProfile] = attr.Factory(dict)
    size_bytes: int = 0
    profiled_at: float = 0.0
    sample_size: int | None = None
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def completeness_score(self) -> float:
        if not self.column_profiles:
            return 1.0
        scores = [p.completeness for p in self.column_profiles.values()]
        return sum(scores) / len(scores)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class Anomaly:
    """An anomaly detected when comparing profiles."""
    column_name: str
    anomaly_type: str
    message: str
    severity: str = "warning"
    old_value: Any = None
    new_value: Any = None
    score: float = 0.0


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ShiftResult:
    """Result of a distribution shift test on a column."""
    column_name: str
    test_name: str
    statistic: float
    p_value: float
    shifted: bool
    threshold: float = 0.1
    message: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class KSResult:
    """Kolmogorov-Smirnov test result."""
    statistic: float
    p_value: float
    sample_size_1: int = 0
    sample_size_2: int = 0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ChiSquaredResult:
    """Chi-squared test result."""
    statistic: float
    p_value: float
    degrees_of_freedom: int = 0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ProfileDiff:
    """Differences between two column profiles."""
    column_name: str
    count_diff: int = 0
    null_count_diff: int = 0
    unique_count_diff: int = 0
    mean_diff: float | None = None
    std_diff: float | None = None
    min_diff: float | None = None
    max_diff: float | None = None
    distribution_shift: ShiftResult | None = None
    anomalies: tuple[Anomaly, ...] = attr.Factory(tuple)

    @property
    def has_significant_changes(self) -> bool:
        if self.distribution_shift is not None and self.distribution_shift.shifted:
            return True
        return len(self.anomalies) > 0


# =====================================================================
# Python ETL analysis types
# =====================================================================

class ETLFramework(enum.Enum):
    PANDAS = "PANDAS"
    PYSPARK = "PYSPARK"
    DBT = "DBT"
    SQLALCHEMY = "SQLALCHEMY"
    RAW_SQL = "RAW_SQL"
    UNKNOWN = "UNKNOWN"


class TransformationType(enum.Enum):
    SOURCE = "SOURCE"
    SINK = "SINK"
    SELECT = "SELECT"
    FILTER = "FILTER"
    JOIN = "JOIN"
    GROUP_BY = "GROUP_BY"
    SORT = "SORT"
    UNION = "UNION"
    WINDOW = "WINDOW"
    RENAME = "RENAME"
    DROP = "DROP"
    ASSIGN = "ASSIGN"
    PIVOT = "PIVOT"
    UNPIVOT = "UNPIVOT"
    FILLNA = "FILLNA"
    DROPNA = "DROPNA"
    CAST = "CAST"
    CUSTOM = "CUSTOM"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class Transformation:
    """A single data transformation operation."""
    transform_type: TransformationType
    input_vars: tuple[str, ...] = attr.Factory(tuple)
    output_var: str = ""
    columns_read: tuple[str, ...] = attr.Factory(tuple)
    columns_written: tuple[str, ...] = attr.Factory(tuple)
    source_line: int = 0
    source_text: str = ""
    parameters: dict[str, Any] = attr.Factory(dict)
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DataflowNode:
    """A node in the extracted dataflow graph."""
    node_id: str
    variable_name: str = ""
    transform: Transformation | None = None
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DataflowEdge:
    """An edge in the extracted dataflow graph."""
    source: str
    target: str
    columns: tuple[str, ...] = attr.Factory(tuple)
    edge_type: str = "data"


@attr.s(slots=True, auto_attribs=True)
class DataflowGraph:
    """A dataflow graph extracted from Python ETL code."""
    nodes: dict[str, DataflowNode] = attr.Factory(dict)
    edges: list[DataflowEdge] = attr.Factory(list)
    framework: ETLFramework = ETLFramework.UNKNOWN
    source_file: str = ""
    metadata: dict[str, Any] = attr.Factory(dict)

    def add_node(self, node: DataflowNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DataflowEdge) -> None:
        self.edges.append(edge)

    def sources(self) -> list[str]:
        targets = {e.target for e in self.edges}
        return [nid for nid in self.nodes if nid not in targets]

    def sinks(self) -> list[str]:
        srcs = {e.source for e in self.edges}
        return [nid for nid in self.nodes if nid not in srcs]

    def parents(self, node_id: str) -> list[str]:
        return [e.source for e in self.edges if e.target == node_id]

    def children(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def to_pipeline_graph(self) -> PipelineGraph:
        """Convert this ETL dataflow to a PipelineGraph."""
        _op_map = {
            TransformationType.SOURCE: SQLOperator.SELECT,
            TransformationType.SINK: SQLOperator.INSERT,
            TransformationType.SELECT: SQLOperator.SELECT,
            TransformationType.FILTER: SQLOperator.FILTER,
            TransformationType.JOIN: SQLOperator.JOIN,
            TransformationType.GROUP_BY: SQLOperator.GROUP_BY,
            TransformationType.SORT: SQLOperator.ORDER_BY,
            TransformationType.UNION: SQLOperator.UNION,
            TransformationType.WINDOW: SQLOperator.WINDOW,
            TransformationType.RENAME: SQLOperator.ALTER_TABLE,
            TransformationType.DROP: SQLOperator.ALTER_TABLE,
            TransformationType.ASSIGN: SQLOperator.SELECT,
            TransformationType.PIVOT: SQLOperator.CUSTOM,
            TransformationType.UNPIVOT: SQLOperator.CUSTOM,
            TransformationType.FILLNA: SQLOperator.UPDATE,
            TransformationType.DROPNA: SQLOperator.FILTER,
            TransformationType.CAST: SQLOperator.ALTER_TABLE,
            TransformationType.CUSTOM: SQLOperator.CUSTOM,
        }
        pg = PipelineGraph(metadata={
            "source": self.source_file,
            "framework": self.framework.value,
        })
        for nid, dn in self.nodes.items():
            op = SQLOperator.CUSTOM
            is_src = False
            is_snk = False
            if dn.transform is not None:
                op = _op_map.get(dn.transform.transform_type, SQLOperator.CUSTOM)
                is_src = dn.transform.transform_type == TransformationType.SOURCE
                is_snk = dn.transform.transform_type == TransformationType.SINK
            pg.add_node(PipelineNode(
                node_id=nid,
                operator=op,
                table_name=dn.variable_name,
                is_source=is_src,
                is_sink=is_snk,
                metadata=dn.metadata,
            ))
        for de in self.edges:
            pg.add_edge(PipelineEdge(
                source=de.source,
                target=de.target,
                columns_referenced=de.columns,
                edge_type=de.edge_type,
            ))
        return pg


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LineageStep:
    """One step in a dataframe lineage chain."""
    variable_name: str
    operation: str
    source_line: int = 0
    columns_in: tuple[str, ...] = attr.Factory(tuple)
    columns_out: tuple[str, ...] = attr.Factory(tuple)
    metadata: dict[str, Any] = attr.Factory(dict)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LineageChain:
    """Complete lineage chain for a single dataframe variable."""
    variable_name: str
    steps: tuple[LineageStep, ...] = attr.Factory(tuple)

    @property
    def all_columns_read(self) -> set[str]:
        result: set[str] = set()
        for s in self.steps:
            result.update(s.columns_in)
        return result

    @property
    def all_columns_written(self) -> set[str]:
        result: set[str] = set()
        for s in self.steps:
            result.update(s.columns_out)
        return result


@attr.s(frozen=True, slots=True, auto_attribs=True)
class PandasLineage:
    """Complete lineage extracted from pandas code."""
    chains: dict[str, LineageChain] = attr.Factory(dict)
    sources: tuple[str, ...] = attr.Factory(tuple)
    sinks: tuple[str, ...] = attr.Factory(tuple)
    transformations: tuple[Transformation, ...] = attr.Factory(tuple)
    dataflow_graph: DataflowGraph | None = None


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SparkLineage:
    """Complete lineage extracted from PySpark code."""
    chains: dict[str, LineageChain] = attr.Factory(dict)
    sources: tuple[str, ...] = attr.Factory(tuple)
    sinks: tuple[str, ...] = attr.Factory(tuple)
    transformations: tuple[Transformation, ...] = attr.Factory(tuple)
    dataflow_graph: DataflowGraph | None = None
    sql_queries: tuple[str, ...] = attr.Factory(tuple)


class QualityPatternType(enum.Enum):
    NULL_CHECK = "NULL_CHECK"
    TYPE_CAST = "TYPE_CAST"
    DEDUPLICATION = "DEDUPLICATION"
    RANGE_VALIDATION = "RANGE_VALIDATION"
    REGEX_VALIDATION = "REGEX_VALIDATION"
    ASSERTION = "ASSERTION"
    LOGGING = "LOGGING"
    ERROR_HANDLING = "ERROR_HANDLING"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"


@attr.s(frozen=True, slots=True, auto_attribs=True)
class QualityPattern:
    """A quality-related pattern detected in ETL code."""
    pattern_type: QualityPatternType
    source_line: int = 0
    source_text: str = ""
    columns: tuple[str, ...] = attr.Factory(tuple)
    description: str = ""


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ETLAnalysisResult:
    """Complete result of analyzing a Python ETL file."""
    source_file: str = ""
    framework: ETLFramework = ETLFramework.UNKNOWN
    transformations: tuple[Transformation, ...] = attr.Factory(tuple)
    dataflow_graph: DataflowGraph | None = None
    lineage: PandasLineage | None = None
    quality_patterns: tuple[QualityPattern, ...] = attr.Factory(tuple)
    errors: tuple[str, ...] = attr.Factory(tuple)
    metadata: dict[str, Any] = attr.Factory(dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def transformation_count(self) -> int:
        return len(self.transformations)
