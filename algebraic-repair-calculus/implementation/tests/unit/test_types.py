"""Comprehensive tests for arc.types.base – the foundation type system.

Covers SQLType, TypeParameters, ParameterisedType, TypeCompatibility,
ColumnConstraint, Column, Schema, QualityConstraint, AvailabilityContract,
CostEstimate, RepairAction, RepairPlan, and related enums/errors.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

try:
    from arc.types.base import (
        SQLType,
        TypeParameters,
        ParameterisedType,
        WideningResult,
        TypeCompatibility,
        ConstraintType,
        ColumnConstraint,
        Column,
        Schema,
        ForeignKey,
        CheckConstraint,
        Severity,
        QualityConstraint,
        AvailabilityContract,
        CostEstimate,
        ActionType,
        RepairAction,
        CostBreakdown,
        RepairPlan,
    )
    from arc.types.errors import (
        SchemaError,
        ColumnNotFoundError,
        DuplicateColumnError,
        TypeParameterError,
        TypeCompatibilityError,
    )

    HAS_TYPES = True
except ImportError:
    HAS_TYPES = False

pytestmark = pytest.mark.skipif(not HAS_TYPES, reason="arc.types not available")


# =====================================================================
# 1. SQLType enum completeness and from_string
# =====================================================================


class TestSQLTypeEnum:
    """Verify the enum members exist and are complete."""

    EXPECTED_MEMBERS = [
        "SMALLINT", "INT", "BIGINT", "SERIAL", "BIGSERIAL",
        "REAL", "FLOAT", "DOUBLE",
        "NUMERIC", "DECIMAL",
        "CHAR", "VARCHAR", "TEXT",
        "BYTEA", "BLOB",
        "BOOLEAN",
        "DATE", "TIME", "TIMETZ", "TIMESTAMP", "TIMESTAMPTZ", "INTERVAL",
        "JSON", "JSONB",
        "ARRAY", "UUID",
    ]

    def test_all_expected_members_exist(self):
        for name in self.EXPECTED_MEMBERS:
            assert hasattr(SQLType, name), f"Missing SQLType.{name}"

    def test_member_count(self):
        assert len(SQLType) == len(self.EXPECTED_MEMBERS)

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_member_value_matches_name(self, name):
        assert SQLType[name].value == name

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_from_string_canonical(self, name):
        assert SQLType.from_string(name) == SQLType[name]

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_from_string_lowercase(self, name):
        assert SQLType.from_string(name.lower()) == SQLType[name]

    @pytest.mark.parametrize("alias,expected", [
        ("INTEGER", "INT"),
        ("INT4", "INT"),
        ("INT8", "BIGINT"),
        ("INT2", "SMALLINT"),
        ("TINYINT", "SMALLINT"),
        ("BOOL", "BOOLEAN"),
        ("FLOAT4", "REAL"),
        ("FLOAT8", "DOUBLE"),
        ("DOUBLE PRECISION", "DOUBLE"),
        ("CHARACTER VARYING", "VARCHAR"),
        ("CHARACTER", "CHAR"),
        ("TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"),
        ("TIMESTAMP WITHOUT TIME ZONE", "TIMESTAMP"),
        ("TIME WITH TIME ZONE", "TIMETZ"),
        ("TIME WITHOUT TIME ZONE", "TIME"),
        ("SERIAL4", "SERIAL"),
        ("SERIAL8", "BIGSERIAL"),
        ("BINARY LARGE OBJECT", "BLOB"),
        ("VARBINARY", "BYTEA"),
    ])
    def test_from_string_aliases(self, alias, expected):
        assert SQLType.from_string(alias) == SQLType[expected]

    @pytest.mark.parametrize("alias", [
        "integer", "bool", "double precision", "character varying",
        "timestamp with time zone",
    ])
    def test_from_string_aliases_case_insensitive(self, alias):
        # Should not raise
        result = SQLType.from_string(alias)
        assert isinstance(result, SQLType)

    def test_from_string_with_whitespace(self):
        assert SQLType.from_string("  INT  ") == SQLType.INT
        assert SQLType.from_string("\tVARCHAR\n") == SQLType.VARCHAR

    def test_from_string_strips_parameterisation(self):
        assert SQLType.from_string("VARCHAR(255)") == SQLType.VARCHAR
        assert SQLType.from_string("DECIMAL(10,2)") == SQLType.DECIMAL

    def test_from_string_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown SQL type"):
            SQLType.from_string("FANTASY_TYPE")

    def test_from_string_empty_raises(self):
        with pytest.raises(ValueError):
            SQLType.from_string("")


class TestSQLTypeFamilyProperties:
    """Verify is_integer, is_floating, is_numeric, etc."""

    @pytest.mark.parametrize("member", ["SMALLINT", "INT", "BIGINT", "SERIAL", "BIGSERIAL"])
    def test_is_integer(self, member):
        assert SQLType[member].is_integer is True

    @pytest.mark.parametrize("member", ["REAL", "FLOAT", "DOUBLE"])
    def test_is_floating(self, member):
        assert SQLType[member].is_floating is True

    @pytest.mark.parametrize("member", [
        "SMALLINT", "INT", "BIGINT", "SERIAL", "BIGSERIAL",
        "REAL", "FLOAT", "DOUBLE",
        "NUMERIC", "DECIMAL",
    ])
    def test_is_numeric(self, member):
        assert SQLType[member].is_numeric is True

    @pytest.mark.parametrize("member", ["TEXT", "BOOLEAN", "DATE", "UUID"])
    def test_non_numeric(self, member):
        assert SQLType[member].is_numeric is False

    @pytest.mark.parametrize("member", ["CHAR", "VARCHAR", "TEXT"])
    def test_is_string(self, member):
        assert SQLType[member].is_string is True

    @pytest.mark.parametrize("member", [
        "DATE", "TIME", "TIMETZ", "TIMESTAMP", "TIMESTAMPTZ", "INTERVAL",
    ])
    def test_is_temporal(self, member):
        assert SQLType[member].is_temporal is True

    @pytest.mark.parametrize("member", ["BYTEA", "BLOB"])
    def test_is_binary(self, member):
        assert SQLType[member].is_binary is True

    @pytest.mark.parametrize("member", ["JSON", "JSONB"])
    def test_is_json(self, member):
        assert SQLType[member].is_json is True

    def test_non_json(self):
        assert SQLType.TEXT.is_json is False

    @pytest.mark.parametrize("member", ["CHAR", "VARCHAR", "NUMERIC", "DECIMAL", "ARRAY"])
    def test_is_parameterised(self, member):
        assert SQLType[member].is_parameterised is True

    @pytest.mark.parametrize("member", ["INT", "TEXT", "BOOLEAN", "UUID"])
    def test_not_parameterised(self, member):
        assert SQLType[member].is_parameterised is False


# =====================================================================
# 2. TypeParameters validation
# =====================================================================


class TestTypeParameters:
    """Validate TypeParameters construction and constraints."""

    def test_default_all_none(self):
        tp = TypeParameters()
        assert tp.length is None
        assert tp.precision is None
        assert tp.scale is None
        assert tp.element_type is None
        assert tp.element_params is None

    def test_length_valid(self):
        tp = TypeParameters(length=255)
        assert tp.length == 255

    def test_length_zero(self):
        tp = TypeParameters(length=0)
        assert tp.length == 0

    def test_length_negative_raises(self):
        with pytest.raises(TypeParameterError):
            TypeParameters(length=-1)

    def test_length_large(self):
        tp = TypeParameters(length=1_000_000)
        assert tp.length == 1_000_000

    def test_precision_valid(self):
        tp = TypeParameters(precision=18)
        assert tp.precision == 18

    def test_precision_one(self):
        tp = TypeParameters(precision=1)
        assert tp.precision == 1

    def test_precision_zero_raises(self):
        with pytest.raises(TypeParameterError):
            TypeParameters(precision=0)

    def test_precision_negative_raises(self):
        with pytest.raises(TypeParameterError):
            TypeParameters(precision=-5)

    def test_scale_within_precision(self):
        tp = TypeParameters(precision=10, scale=4)
        assert tp.scale == 4

    def test_scale_equals_precision(self):
        tp = TypeParameters(precision=10, scale=10)
        assert tp.scale == 10

    def test_scale_exceeds_precision_raises(self):
        with pytest.raises(TypeParameterError):
            TypeParameters(precision=10, scale=11)

    def test_scale_zero(self):
        tp = TypeParameters(precision=18, scale=0)
        assert tp.scale == 0

    def test_element_type(self):
        tp = TypeParameters(element_type=SQLType.INT)
        assert tp.element_type == SQLType.INT

    def test_element_params(self):
        inner = TypeParameters(length=100)
        tp = TypeParameters(element_type=SQLType.VARCHAR, element_params=inner)
        assert tp.element_params.length == 100

    def test_frozen(self):
        tp = TypeParameters(length=10)
        with pytest.raises(AttributeError):
            tp.length = 20  # type: ignore[misc]

    def test_equality(self):
        a = TypeParameters(length=100)
        b = TypeParameters(length=100)
        assert a == b

    def test_inequality(self):
        a = TypeParameters(length=100)
        b = TypeParameters(length=200)
        assert a != b

    def test_hash_consistency(self):
        a = TypeParameters(precision=18, scale=4)
        b = TypeParameters(precision=18, scale=4)
        assert hash(a) == hash(b)


class TestTypeParametersSerialization:
    """to_dict / from_dict round-trip."""

    def test_empty_to_dict(self):
        tp = TypeParameters()
        assert tp.to_dict() == {}

    def test_length_round_trip(self):
        tp = TypeParameters(length=255)
        d = tp.to_dict()
        assert d == {"length": 255}
        assert TypeParameters.from_dict(d) == tp

    def test_precision_scale_round_trip(self):
        tp = TypeParameters(precision=18, scale=4)
        d = tp.to_dict()
        assert d == {"precision": 18, "scale": 4}
        assert TypeParameters.from_dict(d) == tp

    def test_element_type_round_trip(self):
        tp = TypeParameters(element_type=SQLType.INT)
        d = tp.to_dict()
        assert d["element_type"] == "INT"
        assert TypeParameters.from_dict(d) == tp

    def test_nested_element_params_round_trip(self):
        inner = TypeParameters(length=50)
        tp = TypeParameters(element_type=SQLType.VARCHAR, element_params=inner)
        d = tp.to_dict()
        assert "element_params" in d
        restored = TypeParameters.from_dict(d)
        assert restored == tp
        assert restored.element_params.length == 50


# =====================================================================
# 3. ParameterisedType creation, factories, from_string, str()
# =====================================================================


class TestParameterisedType:
    """ParameterisedType construction and factory methods."""

    def test_simple(self):
        pt = ParameterisedType.simple(SQLType.INT)
        assert pt.base == SQLType.INT
        assert pt.params == TypeParameters()

    def test_varchar_default(self):
        pt = ParameterisedType.varchar()
        assert pt.base == SQLType.VARCHAR
        assert pt.params.length == 255

    def test_varchar_custom(self):
        pt = ParameterisedType.varchar(100)
        assert pt.params.length == 100

    def test_char_default(self):
        pt = ParameterisedType.char()
        assert pt.base == SQLType.CHAR
        assert pt.params.length == 1

    def test_char_custom(self):
        pt = ParameterisedType.char(10)
        assert pt.params.length == 10

    def test_decimal_default(self):
        pt = ParameterisedType.decimal()
        assert pt.base == SQLType.DECIMAL
        assert pt.params.precision == 18
        assert pt.params.scale == 4

    def test_decimal_custom(self):
        pt = ParameterisedType.decimal(10, 2)
        assert pt.params.precision == 10
        assert pt.params.scale == 2

    def test_numeric_default(self):
        pt = ParameterisedType.numeric()
        assert pt.base == SQLType.NUMERIC
        assert pt.params.precision == 18
        assert pt.params.scale == 0

    def test_numeric_custom(self):
        pt = ParameterisedType.numeric(5, 3)
        assert pt.params.precision == 5
        assert pt.params.scale == 3

    def test_array_of(self):
        pt = ParameterisedType.array_of(SQLType.INT)
        assert pt.base == SQLType.ARRAY
        assert pt.params.element_type == SQLType.INT
        assert pt.params.element_params is None

    def test_array_of_with_params(self):
        inner = TypeParameters(length=50)
        pt = ParameterisedType.array_of(SQLType.VARCHAR, inner)
        assert pt.params.element_type == SQLType.VARCHAR
        assert pt.params.element_params.length == 50

    def test_frozen(self):
        pt = ParameterisedType.simple(SQLType.INT)
        with pytest.raises(AttributeError):
            pt.base = SQLType.BIGINT  # type: ignore[misc]

    def test_equality(self):
        a = ParameterisedType.varchar(100)
        b = ParameterisedType.varchar(100)
        assert a == b

    def test_inequality(self):
        a = ParameterisedType.varchar(100)
        b = ParameterisedType.varchar(200)
        assert a != b


class TestParameterisedTypeStr:
    """String formatting via __str__."""

    def test_simple_int(self):
        assert str(ParameterisedType.simple(SQLType.INT)) == "INT"

    def test_varchar(self):
        assert str(ParameterisedType.varchar(100)) == "VARCHAR(100)"

    def test_char(self):
        assert str(ParameterisedType.char(10)) == "CHAR(10)"

    def test_decimal_with_scale(self):
        assert str(ParameterisedType.decimal(18, 4)) == "DECIMAL(18,4)"

    def test_decimal_no_scale(self):
        assert str(ParameterisedType.decimal(10, 0)) == "DECIMAL(10)"

    def test_numeric_with_scale(self):
        assert str(ParameterisedType.numeric(12, 3)) == "NUMERIC(12,3)"

    def test_int_array(self):
        assert str(ParameterisedType.array_of(SQLType.INT)) == "INT[]"

    def test_varchar_array(self):
        inner = TypeParameters(length=50)
        pt = ParameterisedType.array_of(SQLType.VARCHAR, inner)
        assert str(pt) == "VARCHAR(50)[]"

    def test_text(self):
        assert str(ParameterisedType.simple(SQLType.TEXT)) == "TEXT"

    def test_boolean(self):
        assert str(ParameterisedType.simple(SQLType.BOOLEAN)) == "BOOLEAN"


class TestParameterisedTypeFromString:
    """Parsing type strings."""

    @pytest.mark.parametrize("s,base", [
        ("INT", SQLType.INT),
        ("TEXT", SQLType.TEXT),
        ("BOOLEAN", SQLType.BOOLEAN),
        ("UUID", SQLType.UUID),
        ("TIMESTAMP", SQLType.TIMESTAMP),
    ])
    def test_simple_types(self, s, base):
        pt = ParameterisedType.from_string(s)
        assert pt.base == base
        assert pt.params == TypeParameters()

    def test_varchar_with_length(self):
        pt = ParameterisedType.from_string("VARCHAR(100)")
        assert pt.base == SQLType.VARCHAR
        assert pt.params.length == 100

    def test_char_with_length(self):
        pt = ParameterisedType.from_string("CHAR(5)")
        assert pt.base == SQLType.CHAR
        assert pt.params.length == 5

    def test_decimal_with_precision_and_scale(self):
        pt = ParameterisedType.from_string("DECIMAL(10,2)")
        assert pt.base == SQLType.DECIMAL
        assert pt.params.precision == 10
        assert pt.params.scale == 2

    def test_decimal_precision_only(self):
        pt = ParameterisedType.from_string("DECIMAL(10)")
        assert pt.base == SQLType.DECIMAL
        assert pt.params.precision == 10
        assert pt.params.scale == 0

    def test_numeric_with_precision_and_scale(self):
        pt = ParameterisedType.from_string("NUMERIC(18,4)")
        assert pt.base == SQLType.NUMERIC
        assert pt.params.precision == 18
        assert pt.params.scale == 4

    def test_int_array(self):
        pt = ParameterisedType.from_string("INT[]")
        assert pt.base == SQLType.ARRAY
        assert pt.params.element_type == SQLType.INT

    def test_varchar_array(self):
        pt = ParameterisedType.from_string("VARCHAR(100)[]")
        assert pt.base == SQLType.ARRAY
        assert pt.params.element_type == SQLType.VARCHAR
        assert pt.params.element_params is not None
        assert pt.params.element_params.length == 100

    def test_case_insensitive(self):
        pt = ParameterisedType.from_string("varchar(50)")
        assert pt.base == SQLType.VARCHAR
        assert pt.params.length == 50

    def test_with_whitespace(self):
        pt = ParameterisedType.from_string("  INT  ")
        assert pt.base == SQLType.INT

    def test_alias_integer(self):
        pt = ParameterisedType.from_string("INTEGER")
        assert pt.base == SQLType.INT

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ParameterisedType.from_string("NOT_A_TYPE")

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            ParameterisedType.from_string("@@@")


class TestParameterisedTypeSerialization:
    """to_dict / from_dict round-trip."""

    def test_simple_round_trip(self):
        pt = ParameterisedType.simple(SQLType.INT)
        d = pt.to_dict()
        assert d == {"base": "INT"}
        assert ParameterisedType.from_dict(d) == pt

    def test_varchar_round_trip(self):
        pt = ParameterisedType.varchar(100)
        d = pt.to_dict()
        assert d["base"] == "VARCHAR"
        assert d["params"]["length"] == 100
        assert ParameterisedType.from_dict(d) == pt

    def test_decimal_round_trip(self):
        pt = ParameterisedType.decimal(10, 2)
        assert ParameterisedType.from_dict(pt.to_dict()) == pt

    def test_array_round_trip(self):
        pt = ParameterisedType.array_of(SQLType.INT)
        assert ParameterisedType.from_dict(pt.to_dict()) == pt


# =====================================================================
# 4. WideningResult and TypeCompatibility
# =====================================================================


class TestWideningResult:
    """WideningResult enum members."""

    def test_members_exist(self):
        assert WideningResult.IDENTICAL.value == "identical"
        assert WideningResult.SAFE_WIDENING.value == "safe_widening"
        assert WideningResult.LOSSY_NARROWING.value == "lossy_narrowing"
        assert WideningResult.INCOMPATIBLE.value == "incompatible"

    def test_member_count(self):
        assert len(WideningResult) == 4


class TestTypeCompatibilityCompare:
    """TypeCompatibility.compare() widening rules."""

    def test_identical(self):
        t = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.compare(t, t) == WideningResult.IDENTICAL

    # Integer widening
    @pytest.mark.parametrize("src,tgt,expected", [
        (SQLType.SMALLINT, SQLType.INT, WideningResult.SAFE_WIDENING),
        (SQLType.SMALLINT, SQLType.BIGINT, WideningResult.SAFE_WIDENING),
        (SQLType.INT, SQLType.BIGINT, WideningResult.SAFE_WIDENING),
        (SQLType.BIGINT, SQLType.INT, WideningResult.LOSSY_NARROWING),
        (SQLType.INT, SQLType.SMALLINT, WideningResult.LOSSY_NARROWING),
    ])
    def test_integer_widening(self, src, tgt, expected):
        a = ParameterisedType.simple(src)
        b = ParameterisedType.simple(tgt)
        assert TypeCompatibility.compare(a, b) == expected

    # Float widening
    @pytest.mark.parametrize("src,tgt,expected", [
        (SQLType.REAL, SQLType.FLOAT, WideningResult.SAFE_WIDENING),
        (SQLType.REAL, SQLType.DOUBLE, WideningResult.SAFE_WIDENING),
        (SQLType.FLOAT, SQLType.DOUBLE, WideningResult.SAFE_WIDENING),
        (SQLType.DOUBLE, SQLType.REAL, WideningResult.LOSSY_NARROWING),
    ])
    def test_float_widening(self, src, tgt, expected):
        a = ParameterisedType.simple(src)
        b = ParameterisedType.simple(tgt)
        assert TypeCompatibility.compare(a, b) == expected

    # Cross-family: int -> float
    def test_int_to_float_safe(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.DOUBLE)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_float_to_int_lossy(self):
        a = ParameterisedType.simple(SQLType.DOUBLE)
        b = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    # Int/Float -> Decimal
    def test_int_to_decimal_safe(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.decimal(18, 4)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_float_to_decimal_safe(self):
        a = ParameterisedType.simple(SQLType.DOUBLE)
        b = ParameterisedType.decimal(18, 4)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    # Decimal -> Float/Int
    def test_decimal_to_float_lossy(self):
        a = ParameterisedType.decimal(18, 4)
        b = ParameterisedType.simple(SQLType.DOUBLE)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    def test_decimal_to_int_lossy(self):
        a = ParameterisedType.decimal(18, 4)
        b = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    # String widening
    @pytest.mark.parametrize("src,tgt,expected", [
        (SQLType.CHAR, SQLType.VARCHAR, WideningResult.SAFE_WIDENING),
        (SQLType.CHAR, SQLType.TEXT, WideningResult.SAFE_WIDENING),
        (SQLType.VARCHAR, SQLType.TEXT, WideningResult.SAFE_WIDENING),
        (SQLType.TEXT, SQLType.VARCHAR, WideningResult.LOSSY_NARROWING),
        (SQLType.TEXT, SQLType.CHAR, WideningResult.LOSSY_NARROWING),
        (SQLType.VARCHAR, SQLType.CHAR, WideningResult.LOSSY_NARROWING),
    ])
    def test_string_widening(self, src, tgt, expected):
        a = ParameterisedType.simple(src)
        b = ParameterisedType.simple(tgt)
        assert TypeCompatibility.compare(a, b) == expected

    # VARCHAR parameter widening
    def test_varchar_length_widening(self):
        a = ParameterisedType.varchar(50)
        b = ParameterisedType.varchar(100)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_varchar_length_narrowing(self):
        a = ParameterisedType.varchar(100)
        b = ParameterisedType.varchar(50)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    def test_varchar_length_identical(self):
        a = ParameterisedType.varchar(100)
        b = ParameterisedType.varchar(100)
        assert TypeCompatibility.compare(a, b) == WideningResult.IDENTICAL

    # Decimal parameter widening
    def test_decimal_precision_widening(self):
        a = ParameterisedType.decimal(10, 2)
        b = ParameterisedType.decimal(18, 4)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_decimal_precision_narrowing(self):
        a = ParameterisedType.decimal(18, 4)
        b = ParameterisedType.decimal(10, 2)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    # Temporal
    def test_date_to_timestamp(self):
        a = ParameterisedType.simple(SQLType.DATE)
        b = ParameterisedType.simple(SQLType.TIMESTAMP)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_timestamp_to_timestamptz(self):
        a = ParameterisedType.simple(SQLType.TIMESTAMP)
        b = ParameterisedType.simple(SQLType.TIMESTAMPTZ)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_time_to_timetz(self):
        a = ParameterisedType.simple(SQLType.TIME)
        b = ParameterisedType.simple(SQLType.TIMETZ)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_interval_incompatible_with_date(self):
        a = ParameterisedType.simple(SQLType.INTERVAL)
        b = ParameterisedType.simple(SQLType.DATE)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    # JSON
    def test_json_to_jsonb(self):
        a = ParameterisedType.simple(SQLType.JSON)
        b = ParameterisedType.simple(SQLType.JSONB)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    def test_jsonb_to_json(self):
        a = ParameterisedType.simple(SQLType.JSONB)
        b = ParameterisedType.simple(SQLType.JSON)
        assert TypeCompatibility.compare(a, b) == WideningResult.LOSSY_NARROWING

    # Special: SERIAL/BIGSERIAL are in _INTEGER_TYPES so the integer
    # widening path handles them; they are not in the ordered widening
    # list, so the result is INCOMPATIBLE (the later explicit checks
    # at lines 449-452 are unreachable).
    def test_serial_to_int_incompatible(self):
        a = ParameterisedType.simple(SQLType.SERIAL)
        b = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    def test_bigserial_to_bigint_incompatible(self):
        a = ParameterisedType.simple(SQLType.BIGSERIAL)
        b = ParameterisedType.simple(SQLType.BIGINT)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    def test_bytea_blob_safe(self):
        a = ParameterisedType.simple(SQLType.BYTEA)
        b = ParameterisedType.simple(SQLType.BLOB)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING

    # Incompatible
    def test_int_to_text_incompatible(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.TEXT)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    def test_boolean_to_int_incompatible(self):
        a = ParameterisedType.simple(SQLType.BOOLEAN)
        b = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    def test_uuid_to_text_incompatible(self):
        a = ParameterisedType.simple(SQLType.UUID)
        b = ParameterisedType.simple(SQLType.TEXT)
        assert TypeCompatibility.compare(a, b) == WideningResult.INCOMPATIBLE

    # Array element type comparison
    def test_array_element_widening(self):
        a = ParameterisedType.array_of(SQLType.INT)
        b = ParameterisedType.array_of(SQLType.BIGINT)
        assert TypeCompatibility.compare(a, b) == WideningResult.SAFE_WIDENING


class TestTypeCompatibilityCanWiden:
    """can_widen convenience method."""

    def test_identical_is_widenable(self):
        t = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.can_widen(t, t) is True

    def test_safe_widening_is_widenable(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.BIGINT)
        assert TypeCompatibility.can_widen(a, b) is True

    def test_lossy_not_widenable(self):
        a = ParameterisedType.simple(SQLType.BIGINT)
        b = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.can_widen(a, b) is False

    def test_incompatible_not_widenable(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.TEXT)
        assert TypeCompatibility.can_widen(a, b) is False


class TestTypeCompatibilityCommonSupertype:
    """common_supertype method."""

    def test_identical(self):
        t = ParameterisedType.simple(SQLType.INT)
        assert TypeCompatibility.common_supertype(t, t) == t

    def test_int_bigint(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.BIGINT)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.BIGINT

    def test_real_double(self):
        a = ParameterisedType.simple(SQLType.REAL)
        b = ParameterisedType.simple(SQLType.DOUBLE)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.DOUBLE

    def test_int_and_float_gives_double(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.FLOAT)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.DOUBLE

    def test_string_family_gives_text(self):
        a = ParameterisedType.simple(SQLType.CHAR)
        b = ParameterisedType.simple(SQLType.VARCHAR)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.TEXT

    def test_timestamp_timestamptz(self):
        a = ParameterisedType.simple(SQLType.TIMESTAMP)
        b = ParameterisedType.simple(SQLType.TIMESTAMPTZ)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.TIMESTAMPTZ

    def test_date_timestamp(self):
        a = ParameterisedType.simple(SQLType.DATE)
        b = ParameterisedType.simple(SQLType.TIMESTAMP)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.TIMESTAMP

    def test_time_timetz(self):
        a = ParameterisedType.simple(SQLType.TIME)
        b = ParameterisedType.simple(SQLType.TIMETZ)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.TIMETZ

    def test_json_jsonb(self):
        a = ParameterisedType.simple(SQLType.JSON)
        b = ParameterisedType.simple(SQLType.JSONB)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.JSONB

    def test_bytea_blob(self):
        a = ParameterisedType.simple(SQLType.BYTEA)
        b = ParameterisedType.simple(SQLType.BLOB)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.BYTEA

    def test_incompatible_returns_none(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.TEXT)
        assert TypeCompatibility.common_supertype(a, b) is None

    def test_varchar_merge_params(self):
        a = ParameterisedType.varchar(50)
        b = ParameterisedType.varchar(100)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.params.length == 100

    def test_decimal_merge_params(self):
        a = ParameterisedType.decimal(10, 2)
        b = ParameterisedType.decimal(18, 4)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.params.precision == 18
        assert result.params.scale == 4

    def test_numeric_and_decimal(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.decimal(18, 4)
        result = TypeCompatibility.common_supertype(a, b)
        assert result is not None
        assert result.base == SQLType.DECIMAL


class TestTypeCompatibilityAssertCompatible:
    """assert_compatible raises on incompatible."""

    def test_compatible_no_raise(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.BIGINT)
        TypeCompatibility.assert_compatible(a, b, "col")  # should not raise

    def test_identical_no_raise(self):
        t = ParameterisedType.simple(SQLType.INT)
        TypeCompatibility.assert_compatible(t, t)

    def test_lossy_no_raise(self):
        a = ParameterisedType.simple(SQLType.BIGINT)
        b = ParameterisedType.simple(SQLType.INT)
        # lossy but not incompatible — should not raise
        TypeCompatibility.assert_compatible(a, b, "col")

    def test_incompatible_raises(self):
        a = ParameterisedType.simple(SQLType.INT)
        b = ParameterisedType.simple(SQLType.TEXT)
        with pytest.raises(TypeCompatibilityError):
            TypeCompatibility.assert_compatible(a, b, "col")


# =====================================================================
# 5. ColumnConstraint
# =====================================================================


class TestConstraintType:
    """ConstraintType enum members."""

    EXPECTED = [
        "NOT_NULL", "UNIQUE", "PRIMARY_KEY", "FOREIGN_KEY",
        "CHECK", "DEFAULT", "RANGE", "PATTERN", "ENUM_VALUES",
    ]

    def test_all_members(self):
        for name in self.EXPECTED:
            assert hasattr(ConstraintType, name)

    def test_count(self):
        assert len(ConstraintType) == len(self.EXPECTED)


class TestColumnConstraint:
    """ColumnConstraint construction and factories."""

    def test_not_null(self):
        c = ColumnConstraint.not_null()
        assert c.constraint_type == ConstraintType.NOT_NULL
        assert c.expression == ""
        assert c.parameters == {}

    def test_unique(self):
        c = ColumnConstraint.unique()
        assert c.constraint_type == ConstraintType.UNIQUE

    def test_check(self):
        c = ColumnConstraint.check("age > 0")
        assert c.constraint_type == ConstraintType.CHECK
        assert c.expression == "age > 0"

    def test_default(self):
        c = ColumnConstraint.default("NOW()")
        assert c.constraint_type == ConstraintType.DEFAULT
        assert c.expression == "NOW()"

    def test_range_constraint_both(self):
        c = ColumnConstraint.range_constraint(0.0, 100.0)
        assert c.constraint_type == ConstraintType.RANGE
        assert c.parameters == {"min": 0.0, "max": 100.0}

    def test_range_constraint_min_only(self):
        c = ColumnConstraint.range_constraint(min_val=0.0)
        assert c.parameters == {"min": 0.0}
        assert "max" not in c.parameters

    def test_range_constraint_max_only(self):
        c = ColumnConstraint.range_constraint(max_val=999.0)
        assert c.parameters == {"max": 999.0}

    def test_range_constraint_neither(self):
        c = ColumnConstraint.range_constraint()
        assert c.parameters == {}

    def test_pattern(self):
        c = ColumnConstraint.pattern(r"^\d{3}-\d{4}$")
        assert c.constraint_type == ConstraintType.PATTERN
        assert c.expression == r"^\d{3}-\d{4}$"

    def test_enum_values(self):
        c = ColumnConstraint.enum_values(["a", "b", "c"])
        assert c.constraint_type == ConstraintType.ENUM_VALUES
        assert c.parameters == {"values": ["a", "b", "c"]}

    def test_frozen(self):
        c = ColumnConstraint.not_null()
        with pytest.raises(AttributeError):
            c.expression = "x"  # type: ignore[misc]


class TestColumnConstraintSerialization:
    """to_dict / from_dict round-trip."""

    def test_not_null_round_trip(self):
        c = ColumnConstraint.not_null()
        d = c.to_dict()
        assert d["type"] == "NOT_NULL"
        restored = ColumnConstraint.from_dict(d)
        assert restored.constraint_type == ConstraintType.NOT_NULL

    def test_check_round_trip(self):
        c = ColumnConstraint.check("x > 0")
        d = c.to_dict()
        assert d["expression"] == "x > 0"
        restored = ColumnConstraint.from_dict(d)
        assert restored == c

    def test_range_round_trip(self):
        c = ColumnConstraint.range_constraint(1.0, 10.0)
        d = c.to_dict()
        assert d["parameters"]["min"] == 1.0
        restored = ColumnConstraint.from_dict(d)
        assert restored.parameters == c.parameters

    def test_enum_values_round_trip(self):
        c = ColumnConstraint.enum_values(["x", "y"])
        d = c.to_dict()
        restored = ColumnConstraint.from_dict(d)
        assert restored.parameters["values"] == ["x", "y"]


# =====================================================================
# 6. Column
# =====================================================================


class TestColumnCreation:
    """Column construction and validation."""

    def test_basic(self):
        c = Column(name="id", sql_type=ParameterisedType.simple(SQLType.INT))
        assert c.name == "id"
        assert c.sql_type.base == SQLType.INT
        assert c.nullable is True
        assert c.default_expr is None
        assert c.position == 0
        assert c.constraints == ()
        assert c.description == ""

    def test_all_fields(self):
        c = Column(
            name="amount",
            sql_type=ParameterisedType.decimal(10, 2),
            nullable=False,
            default_expr="0.00",
            position=3,
            constraints=(ColumnConstraint.not_null(),),
            description="Transaction amount",
        )
        assert c.name == "amount"
        assert c.nullable is False
        assert c.default_expr == "0.00"
        assert c.position == 3
        assert len(c.constraints) == 1
        assert c.description == "Transaction amount"

    def test_empty_name_raises(self):
        with pytest.raises(SchemaError):
            Column(name="", sql_type=ParameterisedType.simple(SQLType.INT))

    @pytest.mark.parametrize("bad_name", [
        "123abc",
        "has space",
        "has-dash",
        "special@char",
        "semi;colon",
        "dot.name",
    ])
    def test_invalid_name_raises(self, bad_name):
        with pytest.raises(SchemaError):
            Column(name=bad_name, sql_type=ParameterisedType.simple(SQLType.INT))

    @pytest.mark.parametrize("good_name", [
        "id", "_private", "col_1", "CamelCase", "ALL_UPPER", "_", "a",
    ])
    def test_valid_names(self, good_name):
        c = Column(name=good_name, sql_type=ParameterisedType.simple(SQLType.INT))
        assert c.name == good_name

    def test_quoted_name_allowed(self):
        c = Column(name='"has space"', sql_type=ParameterisedType.simple(SQLType.INT))
        assert c.name == '"has space"'

    def test_frozen(self):
        c = Column(name="x", sql_type=ParameterisedType.simple(SQLType.INT))
        with pytest.raises(AttributeError):
            c.name = "y"  # type: ignore[misc]


class TestColumnQuick:
    """Column.quick() factory."""

    def test_defaults(self):
        c = Column.quick("age", SQLType.INT)
        assert c.name == "age"
        assert c.sql_type == ParameterisedType.simple(SQLType.INT)
        assert c.nullable is True
        assert c.position == 0

    def test_custom(self):
        c = Column.quick("id", SQLType.BIGINT, nullable=False, position=5)
        assert c.nullable is False
        assert c.position == 5


class TestColumnWithMethods:
    """with_type, with_nullable, with_position, with_name."""

    def test_with_type(self):
        c = Column.quick("x", SQLType.INT)
        c2 = c.with_type(ParameterisedType.simple(SQLType.BIGINT))
        assert c2.sql_type.base == SQLType.BIGINT
        assert c.sql_type.base == SQLType.INT  # original unchanged

    def test_with_nullable(self):
        c = Column.quick("x", SQLType.INT, nullable=True)
        c2 = c.with_nullable(False)
        assert c2.nullable is False
        assert c.nullable is True

    def test_with_position(self):
        c = Column.quick("x", SQLType.INT, position=0)
        c2 = c.with_position(5)
        assert c2.position == 5
        assert c.position == 0

    def test_with_name(self):
        c = Column.quick("old", SQLType.INT)
        c2 = c.with_name("new")
        assert c2.name == "new"
        assert c.name == "old"


class TestColumnStr:
    """__str__ formatting."""

    def test_basic(self):
        c = Column.quick("id", SQLType.INT)
        assert "id" in str(c)
        assert "INT" in str(c)

    def test_not_null(self):
        c = Column(name="id", sql_type=ParameterisedType.simple(SQLType.INT), nullable=False)
        assert "NOT NULL" in str(c)

    def test_default_expr(self):
        c = Column(
            name="created",
            sql_type=ParameterisedType.simple(SQLType.TIMESTAMP),
            default_expr="NOW()",
        )
        assert "DEFAULT NOW()" in str(c)


class TestColumnSerialization:
    """to_dict / from_dict round-trip."""

    def test_basic_round_trip(self):
        c = Column.quick("id", SQLType.INT, position=0)
        d = c.to_dict()
        assert d["name"] == "id"
        assert d["nullable"] is True
        restored = Column.from_dict(d)
        assert restored.name == c.name
        assert restored.sql_type == c.sql_type
        assert restored.nullable == c.nullable

    def test_full_round_trip(self):
        c = Column(
            name="price",
            sql_type=ParameterisedType.decimal(10, 2),
            nullable=False,
            default_expr="0.00",
            position=1,
            constraints=(ColumnConstraint.not_null(), ColumnConstraint.range_constraint(0.0)),
            description="Unit price",
        )
        d = c.to_dict()
        restored = Column.from_dict(d)
        assert restored.name == "price"
        assert restored.nullable is False
        assert restored.default_expr == "0.00"
        assert restored.position == 1
        assert len(restored.constraints) == 2
        assert restored.description == "Unit price"

    def test_dict_has_no_default_expr_when_none(self):
        c = Column.quick("x", SQLType.INT)
        d = c.to_dict()
        assert "default_expr" not in d


# =====================================================================
# 7. Schema operations
# =====================================================================


def _make_schema():
    """Build a small test schema."""
    return Schema(
        columns=(
            Column.quick("id", SQLType.INT, nullable=False, position=0),
            Column.quick("name", SQLType.TEXT, position=1),
            Column.quick("age", SQLType.INT, position=2),
        ),
        primary_key=("id",),
        table_name="users",
    )


class TestSchemaBasic:
    """Schema creation and column access."""

    def test_column_names(self):
        s = _make_schema()
        assert s.column_names == frozenset({"id", "name", "age"})

    def test_column_list_ordered(self):
        s = _make_schema()
        assert s.column_list == ["id", "name", "age"]

    def test_len(self):
        s = _make_schema()
        assert len(s) == 3

    def test_contains(self):
        s = _make_schema()
        assert "id" in s
        assert "missing" not in s

    def test_getitem(self):
        s = _make_schema()
        col = s["id"]
        assert col.name == "id"

    def test_getitem_missing_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            _ = s["nonexistent"]

    def test_get_existing(self):
        s = _make_schema()
        col = s.get("id")
        assert col is not None and col.name == "id"

    def test_get_missing_returns_default(self):
        s = _make_schema()
        assert s.get("missing") is None
        sentinel = Column.quick("x", SQLType.INT)
        assert s.get("missing", sentinel) is sentinel

    def test_empty_schema(self):
        s = Schema.empty()
        assert len(s) == 0
        assert s.column_list == []

    def test_from_columns(self):
        s = Schema.from_columns(("id", SQLType.INT), ("name", SQLType.TEXT))
        assert len(s) == 2
        assert s["id"].position == 0
        assert s["name"].position == 1

    def test_schema_str(self):
        s = _make_schema()
        out = str(s)
        assert "users" in out
        assert "id" in out
        assert "PRIMARY KEY" in out


class TestSchemaAddColumn:
    """add_column operation."""

    def test_add_column(self):
        s = _make_schema()
        new_col = Column.quick("email", SQLType.TEXT)
        s2 = s.add_column(new_col)
        assert len(s2) == 4
        assert "email" in s2
        assert s2["email"].position == 3  # appended with reindexed position

    def test_add_duplicate_raises(self):
        s = _make_schema()
        dup = Column.quick("id", SQLType.INT)
        with pytest.raises(DuplicateColumnError):
            s.add_column(dup)

    def test_original_unchanged(self):
        s = _make_schema()
        new_col = Column.quick("email", SQLType.TEXT)
        s.add_column(new_col)
        assert len(s) == 3  # original untouched


class TestSchemaDropColumn:
    """drop_column operation."""

    def test_drop_column(self):
        s = _make_schema()
        s2 = s.drop_column("age")
        assert len(s2) == 2
        assert "age" not in s2

    def test_drop_reindexes_positions(self):
        s = _make_schema()
        s2 = s.drop_column("name")
        # remaining: id (pos 0), age (pos 1)
        assert s2["id"].position == 0
        assert s2["age"].position == 1

    def test_drop_missing_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            s.drop_column("nonexistent")

    def test_drop_pk_column_strips_from_pk(self):
        s = _make_schema()
        s2 = s.drop_column("id")
        assert "id" not in s2.primary_key

    def test_drop_unique_constraint_column(self):
        s = Schema(
            columns=(
                Column.quick("a", SQLType.INT, position=0),
                Column.quick("b", SQLType.INT, position=1),
            ),
            unique_constraints=(("a", "b"),),
        )
        s2 = s.drop_column("a")
        # unique constraint should only contain "b" now
        assert all("a" not in uc for uc in s2.unique_constraints)


class TestSchemaRenameColumn:
    """rename_column operation."""

    def test_rename(self):
        s = _make_schema()
        s2 = s.rename_column("name", "full_name")
        assert "full_name" in s2
        assert "name" not in s2

    def test_rename_updates_pk(self):
        s = _make_schema()
        s2 = s.rename_column("id", "user_id")
        assert "user_id" in s2.primary_key
        assert "id" not in s2.primary_key

    def test_rename_updates_unique_constraints(self):
        s = Schema(
            columns=(
                Column.quick("a", SQLType.INT, position=0),
                Column.quick("b", SQLType.INT, position=1),
            ),
            unique_constraints=(("a",),),
        )
        s2 = s.rename_column("a", "alpha")
        assert ("alpha",) in s2.unique_constraints

    def test_rename_missing_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            s.rename_column("nope", "new")

    def test_rename_to_existing_raises(self):
        s = _make_schema()
        with pytest.raises(DuplicateColumnError):
            s.rename_column("name", "age")


class TestSchemaWidenColumn:
    """widen_column operation."""

    def test_widen_int_to_bigint(self):
        s = _make_schema()
        s2 = s.widen_column("id", ParameterisedType.simple(SQLType.BIGINT))
        assert s2["id"].sql_type.base == SQLType.BIGINT

    def test_widen_incompatible_raises(self):
        s = _make_schema()
        with pytest.raises(TypeCompatibilityError):
            s.widen_column("id", ParameterisedType.simple(SQLType.TEXT))

    def test_widen_missing_column_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            s.widen_column("nope", ParameterisedType.simple(SQLType.BIGINT))


class TestSchemaSetNullable:
    """set_nullable operation."""

    def test_set_nullable_true(self):
        s = _make_schema()
        s2 = s.set_nullable("id", True)
        assert s2["id"].nullable is True

    def test_set_nullable_false(self):
        s = _make_schema()
        s2 = s.set_nullable("name", False)
        assert s2["name"].nullable is False

    def test_set_nullable_missing_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            s.set_nullable("nope", True)


class TestSchemaProject:
    """project operation."""

    def test_project(self):
        s = _make_schema()
        s2 = s.project(["name", "id"])
        assert s2.column_list == ["name", "id"]
        assert s2["name"].position == 0
        assert s2["id"].position == 1

    def test_project_missing_raises(self):
        s = _make_schema()
        with pytest.raises(ColumnNotFoundError):
            s.project(["id", "nonexistent"])


class TestSchemaMerge:
    """merge operation."""

    def test_merge_adds_new_columns(self):
        s1 = Schema.from_columns(("a", SQLType.INT), ("b", SQLType.TEXT))
        s2 = Schema.from_columns(("c", SQLType.BIGINT), ("d", SQLType.BOOLEAN))
        merged = s1.merge(s2)
        assert len(merged) == 4

    def test_merge_skips_duplicates(self):
        s1 = Schema.from_columns(("a", SQLType.INT), ("b", SQLType.TEXT))
        s2 = Schema.from_columns(("b", SQLType.TEXT), ("c", SQLType.INT))
        merged = s1.merge(s2)
        assert len(merged) == 3  # a, b, c

    def test_merge_with_prefix(self):
        s1 = Schema.from_columns(("a", SQLType.INT),)
        s2 = Schema.from_columns(("a", SQLType.INT),)
        merged = s1.merge(s2, prefix="right_")
        assert "right_a" in merged


class TestSchemaSubschemaAndCompatible:
    """is_subschema_of and compatible_with."""

    def test_is_subschema(self):
        small = Schema.from_columns(("id", SQLType.INT),)
        big = Schema.from_columns(("id", SQLType.INT), ("name", SQLType.TEXT))
        assert small.is_subschema_of(big)

    def test_not_subschema_missing_col(self):
        s1 = Schema.from_columns(("x", SQLType.INT),)
        s2 = Schema.from_columns(("y", SQLType.INT),)
        assert s1.is_subschema_of(s2) is False

    def test_compatible_with_no_mismatch(self):
        s = _make_schema()
        assert s.compatible_with(s) == []

    def test_compatible_with_extra_columns(self):
        s1 = Schema.from_columns(("a", SQLType.INT),)
        s2 = Schema.from_columns(("a", SQLType.INT), ("b", SQLType.TEXT))
        mismatches = s1.compatible_with(s2)
        assert "b" in mismatches


# =====================================================================
# 8. Schema validation
# =====================================================================


class TestSchemaValidation:
    """Schema construction validation: PK/FK refs, duplicate columns."""

    def test_duplicate_column_raises(self):
        with pytest.raises(DuplicateColumnError):
            Schema(
                columns=(
                    Column.quick("x", SQLType.INT, position=0),
                    Column.quick("x", SQLType.INT, position=1),
                ),
            )

    def test_pk_references_nonexistent_raises(self):
        with pytest.raises(ColumnNotFoundError):
            Schema(
                columns=(Column.quick("a", SQLType.INT, position=0),),
                primary_key=("b",),
            )

    def test_pk_valid(self):
        s = Schema(
            columns=(Column.quick("a", SQLType.INT, position=0),),
            primary_key=("a",),
        )
        assert s.primary_key == ("a",)

    def test_unique_constraint_references_nonexistent_raises(self):
        with pytest.raises(ColumnNotFoundError):
            Schema(
                columns=(Column.quick("a", SQLType.INT, position=0),),
                unique_constraints=(("nonexistent",),),
            )

    def test_fk_source_references_nonexistent_raises(self):
        fk = ForeignKey(columns=("missing",), ref_table="other", ref_columns=("id",))
        with pytest.raises(ColumnNotFoundError):
            Schema(
                columns=(Column.quick("a", SQLType.INT, position=0),),
                foreign_keys=(fk,),
            )

    def test_fk_valid(self):
        fk = ForeignKey(columns=("a",), ref_table="other", ref_columns=("id",))
        s = Schema(
            columns=(Column.quick("a", SQLType.INT, position=0),),
            foreign_keys=(fk,),
        )
        assert len(s.foreign_keys) == 1


class TestSchemaSerialization:
    """Schema to_dict / from_dict round-trip."""

    def test_basic_round_trip(self):
        s = _make_schema()
        d = s.to_dict()
        restored = Schema.from_dict(d)
        assert restored.column_names == s.column_names
        assert restored.primary_key == s.primary_key
        assert restored.table_name == s.table_name

    def test_full_round_trip(self):
        fk = ForeignKey(columns=("dept_id",), ref_table="departments", ref_columns=("id",))
        cc = CheckConstraint(expression="age > 0", constraint_name="chk_age")
        s = Schema(
            columns=(
                Column.quick("id", SQLType.INT, nullable=False, position=0),
                Column.quick("dept_id", SQLType.INT, position=1),
                Column.quick("age", SQLType.INT, position=2),
            ),
            primary_key=("id",),
            unique_constraints=(("dept_id",),),
            foreign_keys=(fk,),
            check_constraints=(cc,),
            schema_name="public",
            table_name="employees",
        )
        d = s.to_dict()
        restored = Schema.from_dict(d)
        assert restored.schema_name == "public"
        assert restored.table_name == "employees"
        assert len(restored.foreign_keys) == 1
        assert len(restored.check_constraints) == 1

    def test_empty_schema_round_trip(self):
        s = Schema.empty()
        assert Schema.from_dict(s.to_dict()).column_list == []


# =====================================================================
# 9. QualityConstraint
# =====================================================================


class TestSeverity:
    """Severity enum."""

    def test_members(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"

    def test_count(self):
        assert len(Severity) == 4


class TestQualityConstraint:
    """QualityConstraint construction and factories."""

    def test_basic(self):
        qc = QualityConstraint(
            constraint_id="test_1",
            predicate="x > 0",
        )
        assert qc.constraint_id == "test_1"
        assert qc.predicate == "x > 0"
        assert qc.severity == Severity.ERROR
        assert qc.enabled is True

    def test_not_null_factory(self):
        qc = QualityConstraint.not_null("nn1", "col_a", "col_b")
        assert qc.constraint_id == "nn1"
        assert qc.predicate == "NOT NULL"
        assert qc.affected_columns == ("col_a", "col_b")
        assert qc.metric_name == "null_fraction"
        assert qc.severity == Severity.ERROR

    def test_range_check_both(self):
        qc = QualityConstraint.range_check("rc1", "price", 0.0, 9999.0)
        assert "price >= 0.0" in qc.predicate
        assert "price <= 9999.0" in qc.predicate
        assert qc.affected_columns == ("price",)

    def test_range_check_min_only(self):
        qc = QualityConstraint.range_check("rc2", "age", min_val=0.0)
        assert "age >= 0.0" in qc.predicate
        assert "<=" not in qc.predicate

    def test_range_check_max_only(self):
        qc = QualityConstraint.range_check("rc3", "score", max_val=100.0)
        assert "score <= 100.0" in qc.predicate

    def test_range_check_neither(self):
        qc = QualityConstraint.range_check("rc4", "x")
        assert qc.predicate == "TRUE"

    def test_uniqueness(self):
        qc = QualityConstraint.uniqueness("uq1", "email")
        assert "UNIQUE" in qc.predicate
        assert qc.affected_columns == ("email",)
        assert qc.metric_name == "duplicate_fraction"

    def test_uniqueness_multi_column(self):
        qc = QualityConstraint.uniqueness("uq2", "first", "last")
        assert "first" in qc.predicate
        assert "last" in qc.predicate
        assert qc.affected_columns == ("first", "last")

    def test_freshness(self):
        qc = QualityConstraint.freshness("fr1", "updated_at", 24.0)
        assert "MAX_AGE" in qc.predicate
        assert qc.severity_threshold == 24.0
        assert qc.severity == Severity.WARNING
        assert qc.affected_columns == ("updated_at",)

    def test_row_count_min(self):
        qc = QualityConstraint.row_count("cnt1", min_rows=100)
        assert "COUNT(*) >= 100" in qc.predicate

    def test_row_count_max(self):
        qc = QualityConstraint.row_count("cnt2", max_rows=10000)
        assert "COUNT(*) <= 10000" in qc.predicate

    def test_row_count_both(self):
        qc = QualityConstraint.row_count("cnt3", min_rows=1, max_rows=1000)
        assert ">=" in qc.predicate
        assert "<=" in qc.predicate

    def test_row_count_neither(self):
        qc = QualityConstraint.row_count("cnt4")
        assert qc.predicate == "TRUE"

    def test_distribution(self):
        qc = QualityConstraint.distribution("dist1", "income", "ks", 0.05)
        assert "DISTRIBUTION_TEST" in qc.predicate
        assert qc.metric_name == "ks_p_value"
        assert qc.severity_threshold == 0.05
        assert qc.severity == Severity.WARNING

    def test_str(self):
        qc = QualityConstraint.not_null("nn1", "col_a")
        s = str(qc)
        assert "nn1" in s
        assert "col_a" in s


class TestQualityConstraintSerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip(self):
        qc = QualityConstraint.not_null("nn1", "col_a")
        d = qc.to_dict()
        assert d["constraint_id"] == "nn1"
        assert d["severity"] == "error"
        restored = QualityConstraint.from_dict(d)
        assert restored.constraint_id == qc.constraint_id
        assert restored.predicate == qc.predicate
        assert restored.severity == qc.severity
        assert restored.affected_columns == qc.affected_columns

    def test_full_round_trip(self):
        qc = QualityConstraint(
            constraint_id="full_test",
            predicate="x > 0",
            severity=Severity.CRITICAL,
            severity_threshold=0.01,
            affected_columns=("x", "y"),
            metric_name="test_metric",
            description="A test",
            enabled=False,
        )
        d = qc.to_dict()
        restored = QualityConstraint.from_dict(d)
        assert restored.constraint_id == "full_test"
        assert restored.severity == Severity.CRITICAL
        assert restored.severity_threshold == 0.01
        assert restored.affected_columns == ("x", "y")
        assert restored.metric_name == "test_metric"
        assert restored.description == "A test"
        assert restored.enabled is False


# =====================================================================
# 10. AvailabilityContract
# =====================================================================


class TestAvailabilityContract:
    """AvailabilityContract construction and validation."""

    def test_defaults(self):
        ac = AvailabilityContract()
        assert ac.sla_percentage == 99.0
        assert ac.max_downtime == timedelta(hours=1)
        assert ac.staleness_tolerance == timedelta(hours=24)
        assert ac.priority == 0

    def test_custom(self):
        ac = AvailabilityContract(
            sla_percentage=99.9,
            max_downtime=timedelta(minutes=30),
            staleness_tolerance=timedelta(hours=6),
            priority=10,
            description="High priority",
        )
        assert ac.sla_percentage == 99.9
        assert ac.description == "High priority"

    def test_sla_zero(self):
        ac = AvailabilityContract(sla_percentage=0.0)
        assert ac.sla_percentage == 0.0

    def test_sla_hundred(self):
        ac = AvailabilityContract(sla_percentage=100.0)
        assert ac.sla_percentage == 100.0

    def test_sla_negative_raises(self):
        with pytest.raises(SchemaError):
            AvailabilityContract(sla_percentage=-1.0)

    def test_sla_over_hundred_raises(self):
        with pytest.raises(SchemaError):
            AvailabilityContract(sla_percentage=100.1)

    def test_critical_factory(self):
        ac = AvailabilityContract.critical()
        assert ac.sla_percentage == 99.99
        assert ac.max_downtime == timedelta(minutes=5)
        assert ac.priority == 100

    def test_standard_factory(self):
        ac = AvailabilityContract.standard()
        assert ac.sla_percentage == 99.0
        assert ac.priority == 50

    def test_best_effort_factory(self):
        ac = AvailabilityContract.best_effort()
        assert ac.sla_percentage == 95.0
        assert ac.priority == 10

    def test_meets_sla_true(self):
        ac = AvailabilityContract(sla_percentage=99.0)
        assert ac.meets_sla(99.5) is True

    def test_meets_sla_false(self):
        ac = AvailabilityContract(sla_percentage=99.0)
        assert ac.meets_sla(98.0) is False

    def test_meets_sla_exact(self):
        ac = AvailabilityContract(sla_percentage=99.0)
        assert ac.meets_sla(99.0) is True

    def test_within_staleness_true(self):
        ac = AvailabilityContract(staleness_tolerance=timedelta(hours=24))
        assert ac.within_staleness(timedelta(hours=12)) is True

    def test_within_staleness_false(self):
        ac = AvailabilityContract(staleness_tolerance=timedelta(hours=24))
        assert ac.within_staleness(timedelta(hours=25)) is False

    def test_within_downtime_true(self):
        ac = AvailabilityContract(max_downtime=timedelta(hours=1))
        assert ac.within_downtime(timedelta(minutes=30)) is True

    def test_within_downtime_false(self):
        ac = AvailabilityContract(max_downtime=timedelta(hours=1))
        assert ac.within_downtime(timedelta(hours=2)) is False

    def test_str(self):
        ac = AvailabilityContract.standard()
        s = str(ac)
        assert "99.00%" in s


class TestAvailabilityContractSerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip(self):
        ac = AvailabilityContract(
            sla_percentage=99.5,
            max_downtime=timedelta(minutes=30),
            staleness_tolerance=timedelta(hours=12),
            priority=5,
            description="test",
        )
        d = ac.to_dict()
        assert d["sla_percentage"] == 99.5
        assert d["max_downtime_seconds"] == 1800.0
        restored = AvailabilityContract.from_dict(d)
        assert restored.sla_percentage == ac.sla_percentage
        assert restored.max_downtime == ac.max_downtime
        assert restored.staleness_tolerance == ac.staleness_tolerance
        assert restored.priority == ac.priority
        assert restored.description == ac.description

    def test_critical_round_trip(self):
        ac = AvailabilityContract.critical()
        restored = AvailabilityContract.from_dict(ac.to_dict())
        assert restored.sla_percentage == 99.99
        assert restored.priority == 100


# =====================================================================
# 11. CostEstimate
# =====================================================================


class TestCostEstimate:
    """CostEstimate construction, arithmetic, serialization."""

    def test_zero(self):
        c = CostEstimate.zero()
        assert c.compute_seconds == 0.0
        assert c.memory_bytes == 0
        assert c.io_bytes == 0
        assert c.row_estimate == 0
        assert c.monetary_cost == 0.0
        assert c.confidence == 0.5

    def test_unknown(self):
        c = CostEstimate.unknown()
        assert c.confidence == 0.0

    def test_custom(self):
        c = CostEstimate(
            compute_seconds=10.0,
            memory_bytes=1024,
            io_bytes=2048,
            row_estimate=1000,
            monetary_cost=0.05,
            confidence=0.9,
        )
        assert c.compute_seconds == 10.0
        assert c.memory_bytes == 1024
        assert c.row_estimate == 1000

    def test_total_weighted_cost(self):
        c = CostEstimate(compute_seconds=10.0, memory_bytes=0, io_bytes=0, monetary_cost=0.0)
        assert c.total_weighted_cost >= 10.0

    def test_total_weighted_cost_includes_monetary(self):
        c = CostEstimate(compute_seconds=0.0, monetary_cost=5.0)
        assert c.total_weighted_cost >= 5.0

    def test_addition(self):
        a = CostEstimate(compute_seconds=10.0, memory_bytes=100, io_bytes=200, row_estimate=50, monetary_cost=1.0, confidence=0.8)
        b = CostEstimate(compute_seconds=5.0, memory_bytes=50, io_bytes=100, row_estimate=25, monetary_cost=0.5, confidence=0.6)
        result = a + b
        assert result.compute_seconds == 15.0
        assert result.memory_bytes == 150
        assert result.io_bytes == 300
        assert result.row_estimate == 75
        assert result.monetary_cost == 1.5
        assert result.confidence == 0.6  # min

    def test_scale(self):
        c = CostEstimate(compute_seconds=10.0, memory_bytes=100, io_bytes=200, row_estimate=50, monetary_cost=1.0, confidence=0.8)
        scaled = c.scale(2.0)
        assert scaled.compute_seconds == 20.0
        assert scaled.memory_bytes == 200
        assert scaled.io_bytes == 400
        assert scaled.row_estimate == 100
        assert scaled.monetary_cost == 2.0
        assert scaled.confidence == 0.8  # unchanged

    def test_scale_zero(self):
        c = CostEstimate(compute_seconds=10.0, memory_bytes=100)
        scaled = c.scale(0.0)
        assert scaled.compute_seconds == 0.0
        assert scaled.memory_bytes == 0

    def test_str(self):
        c = CostEstimate(compute_seconds=1.5, memory_bytes=1048576, io_bytes=2097152, row_estimate=100, monetary_cost=0.01)
        s = str(c)
        assert "1.50s" in s
        assert "rows=100" in s


class TestCostEstimateSerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip(self):
        c = CostEstimate(
            compute_seconds=10.0,
            memory_bytes=1024,
            io_bytes=2048,
            row_estimate=500,
            monetary_cost=0.1,
            confidence=0.75,
        )
        d = c.to_dict()
        assert d["compute_seconds"] == 10.0
        restored = CostEstimate.from_dict(d)
        assert restored == c

    def test_zero_round_trip(self):
        c = CostEstimate.zero()
        assert CostEstimate.from_dict(c.to_dict()) == c

    def test_from_dict_defaults(self):
        c = CostEstimate.from_dict({})
        assert c.compute_seconds == 0.0
        assert c.confidence == 0.5


# =====================================================================
# 12. ActionType, RepairAction, CostBreakdown, RepairPlan
# =====================================================================


class TestActionType:
    """ActionType enum."""

    EXPECTED = [
        "RECOMPUTE", "INCREMENTAL_UPDATE", "SCHEMA_MIGRATE",
        "SKIP", "CHECKPOINT", "ROLLBACK", "VALIDATE", "NO_OP",
    ]

    def test_all_members(self):
        for name in self.EXPECTED:
            assert hasattr(ActionType, name)

    def test_count(self):
        assert len(ActionType) == len(self.EXPECTED)

    @pytest.mark.parametrize("name", EXPECTED)
    def test_value_matches_name(self, name):
        assert ActionType[name].value == name


class TestRepairAction:
    """RepairAction construction and properties."""

    def test_basic(self):
        ra = RepairAction(node_id="node_1", action_type=ActionType.RECOMPUTE)
        assert ra.node_id == "node_1"
        assert ra.action_type == ActionType.RECOMPUTE
        assert ra.estimated_cost == 0.0
        assert ra.dependencies == ()
        assert ra.sql_text == ""
        assert ra.is_noop is False

    def test_noop_skip(self):
        ra = RepairAction(node_id="n", action_type=ActionType.SKIP)
        assert ra.is_noop is True

    def test_noop_no_op(self):
        ra = RepairAction(node_id="n", action_type=ActionType.NO_OP)
        assert ra.is_noop is True

    @pytest.mark.parametrize("at", [
        ActionType.RECOMPUTE, ActionType.INCREMENTAL_UPDATE,
        ActionType.SCHEMA_MIGRATE, ActionType.CHECKPOINT,
        ActionType.ROLLBACK, ActionType.VALIDATE,
    ])
    def test_non_noop(self, at):
        ra = RepairAction(node_id="n", action_type=at)
        assert ra.is_noop is False

    def test_with_dependencies(self):
        ra = RepairAction(
            node_id="node_2",
            action_type=ActionType.RECOMPUTE,
            dependencies=("node_1",),
        )
        assert ra.dependencies == ("node_1",)

    def test_with_sql(self):
        ra = RepairAction(
            node_id="n",
            action_type=ActionType.SCHEMA_MIGRATE,
            sql_text="ALTER TABLE ...",
        )
        assert ra.sql_text == "ALTER TABLE ..."

    def test_action_id_generated(self):
        ra1 = RepairAction(node_id="n", action_type=ActionType.RECOMPUTE)
        ra2 = RepairAction(node_id="n", action_type=ActionType.RECOMPUTE)
        assert ra1.action_id != ra2.action_id  # unique IDs


class TestCostBreakdown:
    """CostBreakdown construction."""

    def test_defaults(self):
        cb = CostBreakdown()
        assert cb.compute_cost == 0.0
        assert cb.io_cost == 0.0
        assert cb.total_cost == 0.0
        assert cb.cost_per_node == {}
        assert cb.savings_vs_full_recompute == 0.0

    def test_custom(self):
        cb = CostBreakdown(
            compute_cost=10.0,
            io_cost=5.0,
            materialization_cost=2.0,
            network_cost=1.0,
            total_cost=18.0,
            cost_per_node={"a": 10.0, "b": 8.0},
            savings_vs_full_recompute=0.4,
        )
        assert cb.total_cost == 18.0
        assert cb.cost_per_node["a"] == 10.0
        assert cb.savings_vs_full_recompute == 0.4


class TestRepairPlan:
    """RepairPlan construction and methods."""

    def _make_plan(self):
        a1 = RepairAction(node_id="a", action_type=ActionType.RECOMPUTE, estimated_cost=10.0)
        a2 = RepairAction(node_id="b", action_type=ActionType.SKIP)
        a3 = RepairAction(node_id="c", action_type=ActionType.INCREMENTAL_UPDATE, estimated_cost=5.0)
        return RepairPlan(
            actions=(a1, a2, a3),
            execution_order=("a", "c", "b"),
            total_cost=15.0,
            full_recompute_cost=30.0,
            savings_ratio=0.5,
            affected_nodes=frozenset({"a", "b", "c"}),
        )

    def test_action_count(self):
        plan = self._make_plan()
        assert plan.action_count == 3

    def test_non_trivial_actions(self):
        plan = self._make_plan()
        non_trivial = plan.non_trivial_actions
        assert len(non_trivial) == 2
        assert all(not a.is_noop for a in non_trivial)

    def test_get_action_found(self):
        plan = self._make_plan()
        action = plan.get_action("a")
        assert action is not None
        assert action.node_id == "a"

    def test_get_action_not_found(self):
        plan = self._make_plan()
        assert plan.get_action("nonexistent") is None

    def test_get_actions_for_type(self):
        plan = self._make_plan()
        recomputes = plan.get_actions_for_type(ActionType.RECOMPUTE)
        assert len(recomputes) == 1
        assert recomputes[0].node_id == "a"

    def test_get_actions_for_type_empty(self):
        plan = self._make_plan()
        rollbacks = plan.get_actions_for_type(ActionType.ROLLBACK)
        assert rollbacks == []

    def test_empty_plan(self):
        plan = RepairPlan()
        assert plan.action_count == 0
        assert plan.non_trivial_actions == []
        assert plan.total_cost == 0.0
        assert plan.get_action("x") is None

    def test_annihilated_nodes(self):
        plan = RepairPlan(annihilated_nodes=frozenset({"x", "y"}))
        assert "x" in plan.annihilated_nodes
        assert "y" in plan.annihilated_nodes

    def test_cost_breakdown(self):
        cb = CostBreakdown(total_cost=50.0)
        plan = RepairPlan(cost_breakdown=cb)
        assert plan.cost_breakdown is not None
        assert plan.cost_breakdown.total_cost == 50.0


# =====================================================================
# ForeignKey and CheckConstraint
# =====================================================================


class TestForeignKey:
    """ForeignKey construction and serialization."""

    def test_basic(self):
        fk = ForeignKey(columns=("dept_id",), ref_table="departments", ref_columns=("id",))
        assert fk.columns == ("dept_id",)
        assert fk.ref_table == "departments"
        assert fk.ref_columns == ("id",)
        assert fk.on_delete == "NO ACTION"
        assert fk.on_update == "NO ACTION"
        assert fk.constraint_name is None

    def test_custom(self):
        fk = ForeignKey(
            columns=("a", "b"),
            ref_table="t",
            ref_columns=("x", "y"),
            on_delete="CASCADE",
            on_update="SET NULL",
            constraint_name="fk_test",
        )
        assert fk.on_delete == "CASCADE"
        assert fk.constraint_name == "fk_test"

    def test_round_trip(self):
        fk = ForeignKey(
            columns=("a",),
            ref_table="t",
            ref_columns=("x",),
            on_delete="CASCADE",
            constraint_name="fk_1",
        )
        d = fk.to_dict()
        restored = ForeignKey.from_dict(d)
        assert restored.columns == fk.columns
        assert restored.ref_table == fk.ref_table
        assert restored.on_delete == "CASCADE"
        assert restored.constraint_name == "fk_1"


class TestCheckConstraint:
    """CheckConstraint construction and serialization."""

    def test_basic(self):
        cc = CheckConstraint(expression="age > 0")
        assert cc.expression == "age > 0"
        assert cc.constraint_name is None

    def test_with_name(self):
        cc = CheckConstraint(expression="x > 0", constraint_name="chk_x")
        assert cc.constraint_name == "chk_x"

    def test_round_trip(self):
        cc = CheckConstraint(expression="y > 0", constraint_name="chk_y")
        d = cc.to_dict()
        restored = CheckConstraint.from_dict(d)
        assert restored.expression == "y > 0"
        assert restored.constraint_name == "chk_y"

    def test_round_trip_no_name(self):
        cc = CheckConstraint(expression="z != 0")
        d = cc.to_dict()
        restored = CheckConstraint.from_dict(d)
        assert restored.expression == "z != 0"
        assert restored.constraint_name is None
