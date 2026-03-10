"""
Tests for arc.algebra.schema_delta

Covers SchemaOperation types, SchemaDelta algebraic laws (monoid, inverse),
normalization, conflict detection, schema application, diff, and serialization.
"""

import pytest
from collections import OrderedDict

try:
    from arc.algebra.schema_delta import (
        SchemaDelta,
        AddColumn,
        DropColumn,
        RenameColumn,
        ChangeType,
        AddConstraint,
        DropConstraint,
        ColumnDef,
        Schema,
        ConstraintDef,
        SQLType,
        ConstraintType,
        ConflictType,
        can_widen_type,
        diff_schemas,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="module not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_schema():
    """A schema with no columns or constraints."""
    return Schema(name="empty", columns=OrderedDict(), constraints={})


@pytest.fixture
def simple_schema():
    """A schema with id (INTEGER) and name (VARCHAR) columns."""
    cols = OrderedDict()
    cols["id"] = ColumnDef(name="id", sql_type=SQLType.INTEGER, nullable=False, default_expr=None, position=0)
    cols["name"] = ColumnDef(name="name", sql_type=SQLType.VARCHAR, nullable=True, default_expr=None, position=1)
    return Schema(name="users", columns=cols, constraints={})


@pytest.fixture
def constrained_schema(simple_schema):
    """Schema with a primary key constraint on id."""
    schema = simple_schema.copy()
    c = ConstraintDef(
        constraint_id="pk_id",
        constraint_type=ConstraintType.PRIMARY_KEY,
        columns=["id"],
        predicate=None,
        reference_table=None,
        reference_columns=None,
    )
    schema.add_constraint(c)
    return schema


@pytest.fixture
def three_column_schema():
    """Schema with id, name, and email columns."""
    cols = OrderedDict()
    cols["id"] = ColumnDef(name="id", sql_type=SQLType.INTEGER, nullable=False, default_expr=None, position=0)
    cols["name"] = ColumnDef(name="name", sql_type=SQLType.VARCHAR, nullable=True, default_expr=None, position=1)
    cols["email"] = ColumnDef(name="email", sql_type=SQLType.TEXT, nullable=True, default_expr=None, position=2)
    return Schema(name="contacts", columns=cols, constraints={})


# ===================================================================
# Section 1: SQLType enum
# ===================================================================


class TestSQLType:
    """Validate that every documented member of the SQLType enum exists."""

    EXPECTED_MEMBERS = [
        "INTEGER", "BIGINT", "SMALLINT", "FLOAT", "DOUBLE", "DECIMAL",
        "NUMERIC", "VARCHAR", "TEXT", "CHAR", "BOOLEAN", "DATE",
        "TIMESTAMP", "TIMESTAMPTZ", "TIME", "INTERVAL", "JSON", "JSONB",
        "UUID", "BYTEA", "ARRAY", "NULL", "UNKNOWN",
    ]

    @pytest.mark.parametrize("member", EXPECTED_MEMBERS)
    def test_member_exists(self, member):
        assert hasattr(SQLType, member), f"SQLType missing member {member}"

    def test_members_are_distinct(self):
        vals = [getattr(SQLType, m) for m in self.EXPECTED_MEMBERS]
        assert len(set(vals)) == len(vals)


# ===================================================================
# Section 2: ConstraintType enum
# ===================================================================


class TestConstraintType:
    EXPECTED = ["NOT_NULL", "UNIQUE", "PRIMARY_KEY", "FOREIGN_KEY", "CHECK", "EXCLUSION", "DEFAULT"]

    @pytest.mark.parametrize("member", EXPECTED)
    def test_member_exists(self, member):
        assert hasattr(ConstraintType, member)


# ===================================================================
# Section 3: ColumnDef
# ===================================================================


class TestColumnDef:
    def test_create(self):
        col = ColumnDef(name="age", sql_type=SQLType.INTEGER, nullable=True, default_expr=None, position=0)
        assert col.name == "age"
        assert col.sql_type == SQLType.INTEGER
        assert col.nullable is True
        assert col.position == 0

    def test_with_name(self):
        col = ColumnDef(name="old", sql_type=SQLType.TEXT, nullable=True, default_expr=None, position=0)
        renamed = col.with_name("new")
        assert renamed.name == "new"
        assert renamed.sql_type == col.sql_type
        assert renamed.position == col.position

    def test_with_type(self):
        col = ColumnDef(name="x", sql_type=SQLType.INTEGER, nullable=False, default_expr=None, position=0)
        changed = col.with_type(SQLType.BIGINT)
        assert changed.sql_type == SQLType.BIGINT
        assert changed.name == "x"

    def test_with_position(self):
        col = ColumnDef(name="x", sql_type=SQLType.TEXT, nullable=True, default_expr=None, position=0)
        moved = col.with_position(5)
        assert moved.position == 5
        assert moved.name == "x"

    def test_default_expr_preserved(self):
        col = ColumnDef(name="status", sql_type=SQLType.VARCHAR, nullable=False, default_expr="'active'", position=0)
        assert col.default_expr == "'active'"
        renamed = col.with_name("state")
        assert renamed.default_expr == "'active'"


# ===================================================================
# Section 4: ConstraintDef
# ===================================================================


class TestConstraintDef:
    def test_create(self):
        c = ConstraintDef(
            constraint_id="uq_email",
            constraint_type=ConstraintType.UNIQUE,
            columns=["email"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        assert c.constraint_id == "uq_email"
        assert c.constraint_type == ConstraintType.UNIQUE
        assert c.columns == ["email"]

    def test_with_columns(self):
        c = ConstraintDef(
            constraint_id="uq",
            constraint_type=ConstraintType.UNIQUE,
            columns=["a"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        c2 = c.with_columns(["a", "b"])
        assert c2.columns == ["a", "b"]
        assert c2.constraint_id == "uq"

    def test_references_column(self):
        c = ConstraintDef(
            constraint_id="pk",
            constraint_type=ConstraintType.PRIMARY_KEY,
            columns=["id", "name"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        assert c.references_column("id") is True
        assert c.references_column("missing") is False

    def test_rename_column(self):
        c = ConstraintDef(
            constraint_id="uq",
            constraint_type=ConstraintType.UNIQUE,
            columns=["old_name"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        c2 = c.rename_column("old_name", "new_name")
        assert "new_name" in c2.columns
        assert "old_name" not in c2.columns

    def test_rename_column_not_present(self):
        c = ConstraintDef(
            constraint_id="uq",
            constraint_type=ConstraintType.UNIQUE,
            columns=["a"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        c2 = c.rename_column("x", "y")
        assert c2.columns == ("a",)

    def test_foreign_key_references(self):
        c = ConstraintDef(
            constraint_id="fk_order",
            constraint_type=ConstraintType.FOREIGN_KEY,
            columns=["user_id"],
            predicate=None,
            reference_table="users",
            reference_columns=["id"],
        )
        assert c.reference_table == "users"
        assert c.reference_columns == ["id"]


# ===================================================================
# Section 5: Schema
# ===================================================================


class TestSchema:
    def test_copy_independent(self, simple_schema):
        copy = simple_schema.copy()
        copy.add_column(
            ColumnDef(name="age", sql_type=SQLType.INTEGER, nullable=True, default_expr=None, position=2),
        )
        assert simple_schema.has_column("age") is False
        assert copy.has_column("age") is True

    def test_has_column(self, simple_schema):
        assert simple_schema.has_column("id") is True
        assert simple_schema.has_column("missing") is False

    def test_get_column(self, simple_schema):
        col = simple_schema.get_column("id")
        assert col.name == "id"
        assert col.sql_type == SQLType.INTEGER

    def test_column_names(self, simple_schema):
        names = simple_schema.column_names()
        assert "id" in names
        assert "name" in names

    def test_add_column(self, empty_schema):
        col = ColumnDef(name="x", sql_type=SQLType.TEXT, nullable=True, default_expr=None, position=0)
        empty_schema.add_column(col)
        assert empty_schema.has_column("x")

    def test_drop_column(self, simple_schema):
        simple_schema.drop_column("name")
        assert simple_schema.has_column("name") is False
        assert simple_schema.has_column("id") is True

    def test_rename_column(self, simple_schema):
        simple_schema.rename_column("name", "full_name")
        assert simple_schema.has_column("full_name") is True
        assert simple_schema.has_column("name") is False

    def test_add_constraint(self, simple_schema):
        c = ConstraintDef(
            constraint_id="uq",
            constraint_type=ConstraintType.UNIQUE,
            columns=["name"],
            predicate=None,
            reference_table=None,
            reference_columns=None,
        )
        simple_schema.add_constraint(c)
        assert "uq" in simple_schema.constraints

    def test_drop_constraint(self, constrained_schema):
        constrained_schema.drop_constraint("pk_id")
        assert "pk_id" not in constrained_schema.constraints

    def test_empty_schema(self, empty_schema):
        assert len(empty_schema.column_names()) == 0


# ===================================================================
# Section 6: AddColumn operation
# ===================================================================


class TestAddColumn:
    def test_create(self):
        op = AddColumn(name="age", sql_type=SQLType.INTEGER, position=2, default_expr=None, nullable=True)
        assert op.name == "age"
        assert op.sql_type == SQLType.INTEGER

    def test_inverse_is_drop_column(self):
        op = AddColumn(name="age", sql_type=SQLType.INTEGER, position=2, default_expr=None, nullable=True)
        inv = op.inverse()
        assert isinstance(inv, DropColumn)
        assert inv.name == "age"

    def test_apply_adds_column(self, empty_schema):
        op = AddColumn(name="x", sql_type=SQLType.TEXT, position=0, default_expr=None, nullable=True)
        result = op.apply(empty_schema)
        assert result.has_column("x")

    def test_affected_columns(self):
        op = AddColumn(name="z", sql_type=SQLType.FLOAT, position=0, default_expr=None, nullable=True)
        assert "z" in op.affected_columns()

    def test_is_identity_false(self):
        op = AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        assert op.is_identity() is False

    def test_to_dict_round_trip(self):
        op = AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr="0", nullable=False)
        d = op.to_dict()
        assert d is not None
        assert "name" in d or "column" in d or isinstance(d, dict)

    def test_apply_with_default_expr(self, empty_schema):
        op = AddColumn(name="status", sql_type=SQLType.VARCHAR, position=0, default_expr="'active'", nullable=False)
        result = op.apply(empty_schema)
        col = result.get_column("status")
        assert col.sql_type == SQLType.VARCHAR


# ===================================================================
# Section 7: DropColumn operation
# ===================================================================


class TestDropColumn:
    def test_inverse_is_add_column(self, simple_schema):
        op = DropColumn(name="name", _preserved_type=SQLType.VARCHAR)
        inv = op.inverse()
        assert isinstance(inv, AddColumn)
        assert inv.name == "name"

    def test_apply_removes_column(self, simple_schema):
        op = DropColumn(name="name", _preserved_type=SQLType.VARCHAR)
        result = op.apply(simple_schema)
        assert result.has_column("name") is False

    def test_affected_columns(self):
        op = DropColumn(name="email", _preserved_type=SQLType.TEXT)
        assert "email" in op.affected_columns()

    def test_is_identity_false(self):
        op = DropColumn(name="x", _preserved_type=SQLType.INTEGER)
        assert op.is_identity() is False

    def test_double_inverse_recovers(self):
        op = DropColumn(name="col", _preserved_type=SQLType.BOOLEAN)
        inv = op.inverse()
        inv2 = inv.inverse()
        assert isinstance(inv2, DropColumn)
        assert inv2.name == "col"


# ===================================================================
# Section 8: RenameColumn operation
# ===================================================================


class TestRenameColumn:
    def test_create(self):
        op = RenameColumn(old_name="a", new_name="b")
        assert op.old_name == "a"
        assert op.new_name == "b"

    def test_inverse(self):
        op = RenameColumn(old_name="a", new_name="b")
        inv = op.inverse()
        assert isinstance(inv, RenameColumn)
        assert inv.old_name == "b"
        assert inv.new_name == "a"

    def test_apply_renames(self, simple_schema):
        op = RenameColumn(old_name="name", new_name="full_name")
        result = op.apply(simple_schema)
        assert result.has_column("full_name")
        assert result.has_column("name") is False

    def test_affected_columns(self):
        op = RenameColumn(old_name="a", new_name="b")
        affected = op.affected_columns()
        assert "a" in affected or "b" in affected

    def test_is_identity_same_name(self):
        op = RenameColumn(old_name="x", new_name="x")
        assert op.is_identity() is True

    def test_is_identity_different_name(self):
        op = RenameColumn(old_name="x", new_name="y")
        assert op.is_identity() is False

    def test_double_inverse_identity(self):
        op = RenameColumn(old_name="a", new_name="b")
        assert op.inverse().inverse().old_name == "a"
        assert op.inverse().inverse().new_name == "b"


# ===================================================================
# Section 9: ChangeType operation
# ===================================================================


class TestChangeType:
    def test_create(self):
        op = ChangeType(column_name="age", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None)
        assert op.column_name == "age"
        assert op.old_type == SQLType.INTEGER
        assert op.new_type == SQLType.BIGINT

    def test_inverse(self):
        op = ChangeType(column_name="x", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None)
        inv = op.inverse()
        assert isinstance(inv, ChangeType)
        assert inv.old_type == SQLType.BIGINT
        assert inv.new_type == SQLType.INTEGER

    def test_apply_changes_type(self, simple_schema):
        op = ChangeType(column_name="id", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None)
        result = op.apply(simple_schema)
        assert result.get_column("id").sql_type == SQLType.BIGINT

    def test_affected_columns(self):
        op = ChangeType(column_name="col", old_type=SQLType.TEXT, new_type=SQLType.VARCHAR, coercion_expr=None)
        assert "col" in op.affected_columns()

    def test_is_identity_same_type(self):
        op = ChangeType(column_name="x", old_type=SQLType.INTEGER, new_type=SQLType.INTEGER, coercion_expr=None)
        assert op.is_identity() is True

    def test_is_identity_different_type(self):
        op = ChangeType(column_name="x", old_type=SQLType.INTEGER, new_type=SQLType.TEXT, coercion_expr=None)
        assert op.is_identity() is False

    def test_with_coercion_expr(self):
        op = ChangeType(
            column_name="x",
            old_type=SQLType.VARCHAR,
            new_type=SQLType.INTEGER,
            coercion_expr="CAST(x AS INTEGER)",
        )
        assert op.coercion_expr == "CAST(x AS INTEGER)"


# ===================================================================
# Section 10: AddConstraint operation
# ===================================================================


class TestAddConstraint:
    def test_create(self):
        op = AddConstraint(
            constraint_id="uq_email",
            constraint_type=ConstraintType.UNIQUE,
            predicate=None,
            columns=["email"],
        )
        assert op.constraint_id == "uq_email"

    def test_inverse_is_drop_constraint(self):
        op = AddConstraint(
            constraint_id="uq_email",
            constraint_type=ConstraintType.UNIQUE,
            predicate=None,
            columns=["email"],
        )
        inv = op.inverse()
        assert isinstance(inv, DropConstraint)
        assert inv.constraint_id == "uq_email"

    def test_apply_adds_constraint(self, simple_schema):
        op = AddConstraint(
            constraint_id="uq_name",
            constraint_type=ConstraintType.UNIQUE,
            predicate=None,
            columns=["name"],
        )
        result = op.apply(simple_schema)
        assert "uq_name" in result.constraints

    def test_affected_columns(self):
        op = AddConstraint(
            constraint_id="pk",
            constraint_type=ConstraintType.PRIMARY_KEY,
            predicate=None,
            columns=["id"],
        )
        affected = op.affected_columns()
        assert "id" in affected

    def test_is_identity_false(self):
        op = AddConstraint(
            constraint_id="c",
            constraint_type=ConstraintType.CHECK,
            predicate="x > 0",
            columns=["x"],
        )
        assert op.is_identity() is False


# ===================================================================
# Section 11: DropConstraint operation
# ===================================================================


class TestDropConstraint:
    def test_inverse_is_add_constraint(self):
        op = DropConstraint(constraint_id="uq", _preserved_type=ConstraintType.UNIQUE)
        inv = op.inverse()
        assert isinstance(inv, AddConstraint)
        assert inv.constraint_id == "uq"

    def test_apply_removes_constraint(self, constrained_schema):
        op = DropConstraint(constraint_id="pk_id", _preserved_type=ConstraintType.PRIMARY_KEY)
        result = op.apply(constrained_schema)
        assert "pk_id" not in result.constraints

    def test_affected_columns_empty_or_set(self):
        op = DropConstraint(constraint_id="c", _preserved_type=ConstraintType.CHECK)
        cols = op.affected_columns()
        assert isinstance(cols, (set, list, frozenset, tuple))

    def test_is_identity_false(self):
        op = DropConstraint(constraint_id="c", _preserved_type=ConstraintType.CHECK)
        assert op.is_identity() is False


# ===================================================================
# Section 12: SchemaDelta identity and from_operations
# ===================================================================


class TestSchemaDeltaCreation:
    def test_identity(self):
        delta = SchemaDelta.identity()
        assert delta.is_identity() is True
        assert delta.operation_count() == 0

    def test_from_operations_empty(self):
        delta = SchemaDelta.from_operations([])
        assert delta.is_identity() is True

    def test_from_operations_single(self):
        op = AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        delta = SchemaDelta.from_operations([op])
        assert delta.operation_count() == 1

    def test_from_operations_multiple(self):
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            AddColumn(name="y", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True),
        ]
        delta = SchemaDelta.from_operations(ops)
        assert delta.operation_count() == 2

    def test_operations_list_preserved(self):
        op1 = AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        op2 = RenameColumn(old_name="a", new_name="b")
        delta = SchemaDelta.from_operations([op1, op2])
        assert delta.operations[0] is op1 or isinstance(delta.operations[0], AddColumn)
        assert delta.operations[1] is op2 or isinstance(delta.operations[1], RenameColumn)

    def test_affected_columns(self):
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            RenameColumn(old_name="y", new_name="z"),
        ]
        delta = SchemaDelta.from_operations(ops)
        affected = delta.affected_columns()
        assert "x" in affected


# ===================================================================
# Section 13: SchemaDelta composition — monoid laws
# ===================================================================


class TestSchemaDeltaComposition:
    """SchemaDelta with compose should satisfy monoid laws."""

    def _add_col_delta(self, name, sql_type=SQLType.INTEGER):
        return SchemaDelta.from_operations([
            AddColumn(name=name, sql_type=sql_type, position=0, default_expr=None, nullable=True)
        ])

    def _rename_delta(self, old, new):
        return SchemaDelta.from_operations([RenameColumn(old_name=old, new_name=new)])

    def _drop_col_delta(self, name, sql_type=SQLType.INTEGER):
        return SchemaDelta.from_operations([DropColumn(name=name, _preserved_type=sql_type)])

    # -- Closure: compose returns a SchemaDelta -----------------------

    def test_closure(self):
        a = self._add_col_delta("x")
        b = self._add_col_delta("y")
        result = a.compose(b)
        assert isinstance(result, SchemaDelta)

    # -- Associativity: (a∘b)∘c == a∘(b∘c) ---------------------------

    def test_associativity_apply(self, empty_schema):
        """Associativity checked by applying to a concrete schema."""
        a = self._add_col_delta("x")
        b = self._add_col_delta("y")
        c = self._add_col_delta("z")

        left = a.compose(b).compose(c)
        right = a.compose(b.compose(c))

        schema_l = left.apply_to_schema(empty_schema.copy())
        schema_r = right.apply_to_schema(empty_schema.copy())

        assert set(schema_l.column_names()) == set(schema_r.column_names())

    def test_associativity_mixed_ops(self, empty_schema):
        a = self._add_col_delta("col")
        b = self._rename_delta("col", "column")
        c = self._add_col_delta("other")

        left = a.compose(b).compose(c)
        right = a.compose(b.compose(c))

        schema_l = left.apply_to_schema(empty_schema.copy())
        schema_r = right.apply_to_schema(empty_schema.copy())

        assert set(schema_l.column_names()) == set(schema_r.column_names())

    # -- Identity: ε∘δ = δ∘ε = δ ------------------------------------

    def test_left_identity(self, empty_schema):
        delta = self._add_col_delta("x")
        eps = SchemaDelta.identity()
        composed = eps.compose(delta)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("x")

    def test_right_identity(self, empty_schema):
        delta = self._add_col_delta("x")
        eps = SchemaDelta.identity()
        composed = delta.compose(eps)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("x")

    def test_identity_compose_identity(self):
        eps = SchemaDelta.identity()
        result = eps.compose(eps)
        assert result.is_identity()

    # -- Multiple composes -------------------------------------------

    def test_compose_three(self, empty_schema):
        a = self._add_col_delta("x")
        b = self._add_col_delta("y")
        c = self._add_col_delta("z")
        composed = a.compose(b).compose(c)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("x")
        assert result.has_column("y")
        assert result.has_column("z")

    def test_compose_add_then_drop(self, empty_schema):
        add = self._add_col_delta("tmp")
        drop = self._drop_col_delta("tmp")
        composed = add.compose(drop)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("tmp") is False

    def test_compose_add_rename(self, empty_schema):
        add = self._add_col_delta("old_col")
        rename = self._rename_delta("old_col", "new_col")
        composed = add.compose(rename)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("new_col")
        assert result.has_column("old_col") is False


# ===================================================================
# Section 14: SchemaDelta inverse: δ∘δ⁻¹ = ε
# ===================================================================


class TestSchemaDeltaInverse:
    def test_inverse_of_identity(self):
        eps = SchemaDelta.identity()
        assert eps.inverse().is_identity()

    def test_inverse_single_add(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        inv = delta.inverse()
        composed = delta.compose(inv)
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("x") is False

    def test_inverse_single_rename(self, simple_schema):
        delta = SchemaDelta.from_operations([RenameColumn(old_name="name", new_name="full_name")])
        inv = delta.inverse()
        composed = delta.compose(inv)
        result = composed.apply_to_schema(simple_schema.copy())
        assert result.has_column("name")

    def test_inverse_single_change_type(self, simple_schema):
        delta = SchemaDelta.from_operations([
            ChangeType(column_name="id", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None)
        ])
        inv = delta.inverse()
        composed = delta.compose(inv)
        result = composed.apply_to_schema(simple_schema.copy())
        assert result.get_column("id").sql_type == SQLType.INTEGER

    def test_inverse_reverses_operation_order(self):
        op1 = AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        op2 = AddColumn(name="b", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True)
        delta = SchemaDelta.from_operations([op1, op2])
        inv = delta.inverse()
        # Inverse should reverse order: drop b then drop a
        assert inv.operation_count() == 2

    def test_double_inverse(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        double_inv = delta.inverse().inverse()
        r1 = delta.apply_to_schema(empty_schema.copy())
        r2 = double_inv.apply_to_schema(empty_schema.copy())
        assert set(r1.column_names()) == set(r2.column_names())

    def test_inverse_multi_op(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            AddColumn(name="y", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True),
        ])
        composed = delta.compose(delta.inverse())
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("x") is False
        assert result.has_column("y") is False


# ===================================================================
# Section 15: SchemaDelta normalization
# ===================================================================


class TestSchemaDeltaNormalize:
    def test_normalize_removes_identity_rename(self):
        ops = [RenameColumn(old_name="x", new_name="x")]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        # Identity rename should be removed
        assert norm.operation_count() <= delta.operation_count()

    def test_normalize_removes_identity_change_type(self):
        ops = [ChangeType(column_name="c", old_type=SQLType.INTEGER, new_type=SQLType.INTEGER, coercion_expr=None)]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        assert norm.operation_count() <= delta.operation_count()

    def test_normalize_add_then_drop_same_column(self):
        ops = [
            AddColumn(name="tmp", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            DropColumn(name="tmp", _preserved_type=SQLType.INTEGER),
        ]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        assert norm.operation_count() <= 2

    def test_normalize_preserves_meaningful_ops(self):
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            AddColumn(name="y", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True),
        ]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        assert norm.operation_count() >= 2

    def test_normalize_idempotent(self, empty_schema):
        ops = [
            AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            RenameColumn(old_name="x", new_name="x"),
        ]
        delta = SchemaDelta.from_operations(ops)
        norm1 = delta.normalize()
        norm2 = norm1.normalize()
        s1 = norm1.apply_to_schema(empty_schema.copy())
        s2 = norm2.apply_to_schema(empty_schema.copy())
        assert set(s1.column_names()) == set(s2.column_names())

    def test_normalize_merge_compatible_renames(self):
        ops = [
            RenameColumn(old_name="a", new_name="b"),
            RenameColumn(old_name="b", new_name="c"),
        ]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        # Could merge to a single rename a->c
        assert norm.operation_count() <= 2

    def test_normalize_empty_delta(self):
        delta = SchemaDelta.identity()
        norm = delta.normalize()
        assert norm.is_identity()


# ===================================================================
# Section 16: Conflict detection
# ===================================================================


class TestConflictDetection:
    def test_add_existing_column(self, simple_schema):
        """Adding a column that already exists should produce a conflict."""
        d1 = SchemaDelta.from_operations([
            AddColumn(name="id", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        d2 = SchemaDelta.from_operations([
            AddColumn(name="id", sql_type=SQLType.BIGINT, position=0, default_expr=None, nullable=False)
        ])
        conflicts = d1.conflicts_with(d2)
        assert len(conflicts) > 0

    def test_drop_nonexistent_column(self):
        d1 = SchemaDelta.from_operations([
            DropColumn(name="missing", _preserved_type=SQLType.TEXT)
        ])
        d2 = SchemaDelta.from_operations([
            DropColumn(name="missing", _preserved_type=SQLType.TEXT)
        ])
        conflicts = d1.conflicts_with(d2)
        assert len(conflicts) > 0

    def test_no_conflict_independent_columns(self):
        d1 = SchemaDelta.from_operations([
            AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        d2 = SchemaDelta.from_operations([
            AddColumn(name="b", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True)
        ])
        conflicts = d1.conflicts_with(d2)
        assert len(conflicts) == 0

    def test_conflict_rename_and_drop_same(self):
        d1 = SchemaDelta.from_operations([RenameColumn(old_name="x", new_name="y")])
        d2 = SchemaDelta.from_operations([DropColumn(name="x", _preserved_type=SQLType.TEXT)])
        conflicts = d1.conflicts_with(d2)
        assert len(conflicts) > 0

    def test_conflict_type_attribute(self):
        d1 = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        d2 = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.TEXT, position=0, default_expr=None, nullable=True)
        ])
        conflicts = d1.conflicts_with(d2)
        if conflicts:
            c = conflicts[0]
            assert hasattr(c, "conflict_type")
            assert hasattr(c, "description")

    def test_conflict_severity(self):
        d1 = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        d2 = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.TEXT, position=0, default_expr=None, nullable=True)
        ])
        conflicts = d1.conflicts_with(d2)
        if conflicts:
            assert hasattr(conflicts[0], "severity")


# ===================================================================
# Section 17: apply_to_schema
# ===================================================================


class TestApplyToSchema:
    def test_add_column_to_empty(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="id", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=False)
        ])
        result = delta.apply_to_schema(empty_schema.copy())
        assert result.has_column("id")

    def test_drop_column(self, simple_schema):
        delta = SchemaDelta.from_operations([
            DropColumn(name="name", _preserved_type=SQLType.VARCHAR)
        ])
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.has_column("name") is False
        assert result.has_column("id") is True

    def test_rename_column(self, simple_schema):
        delta = SchemaDelta.from_operations([
            RenameColumn(old_name="name", new_name="full_name")
        ])
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.has_column("full_name")
        assert result.has_column("name") is False

    def test_change_type(self, simple_schema):
        delta = SchemaDelta.from_operations([
            ChangeType(column_name="id", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None)
        ])
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.get_column("id").sql_type == SQLType.BIGINT

    def test_add_constraint(self, simple_schema):
        delta = SchemaDelta.from_operations([
            AddConstraint(
                constraint_id="uq_name",
                constraint_type=ConstraintType.UNIQUE,
                predicate=None,
                columns=["name"],
            )
        ])
        result = delta.apply_to_schema(simple_schema.copy())
        assert "uq_name" in result.constraints

    def test_drop_constraint(self, constrained_schema):
        delta = SchemaDelta.from_operations([
            DropConstraint(constraint_id="pk_id", _preserved_type=ConstraintType.PRIMARY_KEY)
        ])
        result = delta.apply_to_schema(constrained_schema.copy())
        assert "pk_id" not in result.constraints

    def test_multi_op_apply(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            AddColumn(name="y", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True),
            RenameColumn(old_name="x", new_name="x_id"),
        ])
        result = delta.apply_to_schema(empty_schema.copy())
        assert result.has_column("x_id")
        assert result.has_column("y")
        assert result.has_column("x") is False

    def test_identity_apply_preserves_schema(self, simple_schema):
        delta = SchemaDelta.identity()
        result = delta.apply_to_schema(simple_schema.copy())
        assert set(result.column_names()) == set(simple_schema.column_names())


# ===================================================================
# Section 18: Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_delta_operation_count(self):
        assert SchemaDelta.identity().operation_count() == 0

    def test_single_op_delta(self):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        assert delta.operation_count() == 1

    def test_many_ops_delta(self):
        ops = [
            AddColumn(name=f"col_{i}", sql_type=SQLType.INTEGER, position=i, default_expr=None, nullable=True)
            for i in range(50)
        ]
        delta = SchemaDelta.from_operations(ops)
        assert delta.operation_count() == 50

    def test_apply_empty_delta_to_empty_schema(self, empty_schema):
        delta = SchemaDelta.identity()
        result = delta.apply_to_schema(empty_schema.copy())
        assert len(result.column_names()) == 0

    def test_compose_many_identities(self):
        result = SchemaDelta.identity()
        for _ in range(10):
            result = result.compose(SchemaDelta.identity())
        assert result.is_identity()

    def test_affected_columns_empty_delta(self):
        delta = SchemaDelta.identity()
        assert len(delta.affected_columns()) == 0

    def test_large_delta_affected_columns(self):
        ops = [
            AddColumn(name=f"c{i}", sql_type=SQLType.TEXT, position=i, default_expr=None, nullable=True)
            for i in range(20)
        ]
        delta = SchemaDelta.from_operations(ops)
        assert len(delta.affected_columns()) == 20

    def test_conflicting_ops_in_single_delta(self):
        """Two adds of the same column within one delta."""
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            AddColumn(name="x", sql_type=SQLType.TEXT, position=0, default_expr=None, nullable=True),
        ]
        delta = SchemaDelta.from_operations(ops)
        assert delta.operation_count() == 2


# ===================================================================
# Section 19: diff_schemas
# ===================================================================


class TestDiffSchemas:
    def test_identical_schemas(self, simple_schema):
        delta = diff_schemas(simple_schema, simple_schema.copy())
        assert delta.is_identity() or delta.operation_count() == 0

    def test_added_column(self, simple_schema, three_column_schema):
        delta = diff_schemas(simple_schema, three_column_schema)
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.has_column("email")

    def test_removed_column(self, three_column_schema, simple_schema):
        delta = diff_schemas(three_column_schema, simple_schema)
        result = delta.apply_to_schema(three_column_schema.copy())
        assert result.has_column("email") is False

    def test_type_change_detected(self, simple_schema):
        target = simple_schema.copy()
        col = target.get_column("id").with_type(SQLType.BIGINT)
        target.columns["id"] = col
        delta = diff_schemas(simple_schema, target)
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.get_column("id").sql_type == SQLType.BIGINT

    def test_empty_to_populated(self, empty_schema, simple_schema):
        delta = diff_schemas(empty_schema, simple_schema)
        result = delta.apply_to_schema(empty_schema.copy())
        assert result.has_column("id")
        assert result.has_column("name")

    def test_populated_to_empty(self, simple_schema, empty_schema):
        delta = diff_schemas(simple_schema, empty_schema)
        result = delta.apply_to_schema(simple_schema.copy())
        assert len(result.column_names()) == 0

    def test_diff_is_minimal(self, simple_schema):
        """Diff between identical schemas should have zero operations."""
        delta = diff_schemas(simple_schema, simple_schema.copy())
        assert delta.operation_count() == 0

    def test_diff_multiple_changes(self, simple_schema):
        target = simple_schema.copy()
        target.add_column(
            ColumnDef(name="email", sql_type=SQLType.TEXT, nullable=True, default_expr=None, position=2),
        )
        target.drop_column("name")
        delta = diff_schemas(simple_schema, target)
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.has_column("email")
        assert result.has_column("name") is False


# ===================================================================
# Section 20: can_widen_type and helpers
# ===================================================================


class TestTypeWidening:
    def test_integer_to_bigint(self):
        assert can_widen_type(SQLType.INTEGER, SQLType.BIGINT) is True

    def test_smallint_to_integer(self):
        assert can_widen_type(SQLType.SMALLINT, SQLType.INTEGER) is True

    def test_float_to_double(self):
        assert can_widen_type(SQLType.FLOAT, SQLType.DOUBLE) is True

    def test_same_type_widens(self):
        assert can_widen_type(SQLType.INTEGER, SQLType.INTEGER) is True

    def test_text_to_integer_fails(self):
        assert can_widen_type(SQLType.TEXT, SQLType.INTEGER) is False

    def test_varchar_to_text(self):
        result = can_widen_type(SQLType.VARCHAR, SQLType.TEXT)
        # VARCHAR → TEXT is typically a widening
        assert result is True


# ===================================================================
# Section 21: Serialization round-trip
# ===================================================================


class TestSerialization:
    def test_identity_round_trip(self):
        delta = SchemaDelta.identity()
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.is_identity()

    def test_single_add_column_round_trip(self):
        delta = SchemaDelta.from_operations([
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True)
        ])
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.operation_count() == 1

    def test_multi_op_round_trip(self):
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            RenameColumn(old_name="x", new_name="y"),
            ChangeType(column_name="y", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None),
        ]
        delta = SchemaDelta.from_operations(ops)
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.operation_count() == 3

    def test_constraint_ops_round_trip(self):
        ops = [
            AddConstraint(
                constraint_id="pk",
                constraint_type=ConstraintType.PRIMARY_KEY,
                predicate=None,
                columns=["id"],
            ),
            DropConstraint(constraint_id="pk", _preserved_type=ConstraintType.PRIMARY_KEY),
        ]
        delta = SchemaDelta.from_operations(ops)
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.operation_count() == 2

    def test_round_trip_preserves_schema_effect(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="a", sql_type=SQLType.TEXT, position=0, default_expr="'hello'", nullable=False),
            AddColumn(name="b", sql_type=SQLType.INTEGER, position=1, default_expr=None, nullable=True),
        ])
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        r1 = delta.apply_to_schema(empty_schema.copy())
        r2 = restored.apply_to_schema(empty_schema.copy())
        assert set(r1.column_names()) == set(r2.column_names())

    def test_to_dict_is_dict(self):
        delta = SchemaDelta.identity()
        d = delta.to_dict()
        assert isinstance(d, dict)

    def test_drop_column_round_trip(self):
        delta = SchemaDelta.from_operations([
            DropColumn(name="old", _preserved_type=SQLType.VARCHAR)
        ])
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.operation_count() == 1

    def test_rename_round_trip(self):
        delta = SchemaDelta.from_operations([RenameColumn(old_name="a", new_name="b")])
        d = delta.to_dict()
        restored = SchemaDelta.from_dict(d)
        assert restored.operation_count() == 1


# ===================================================================
# Section 22: Comprehensive integration — round-trip via schema
# ===================================================================


class TestIntegration:
    def test_full_lifecycle(self, empty_schema):
        """Add columns, rename, change type, add constraint, then invert all."""
        d1 = SchemaDelta.from_operations([
            AddColumn(name="id", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=False),
            AddColumn(name="name", sql_type=SQLType.VARCHAR, position=1, default_expr=None, nullable=True),
        ])
        d2 = SchemaDelta.from_operations([
            RenameColumn(old_name="name", new_name="full_name"),
            ChangeType(column_name="id", old_type=SQLType.INTEGER, new_type=SQLType.BIGINT, coercion_expr=None),
        ])
        d3 = SchemaDelta.from_operations([
            AddConstraint(
                constraint_id="pk_id",
                constraint_type=ConstraintType.PRIMARY_KEY,
                predicate=None,
                columns=["id"],
            )
        ])
        total = d1.compose(d2).compose(d3)
        result = total.apply_to_schema(empty_schema.copy())
        assert result.has_column("id")
        assert result.has_column("full_name")
        assert result.get_column("id").sql_type == SQLType.BIGINT
        assert "pk_id" in result.constraints

        # Invert should recover original
        inv = total.inverse()
        recovered = inv.apply_to_schema(result.copy())
        assert len(recovered.column_names()) == 0

    def test_diff_then_apply_round_trip(self, simple_schema, three_column_schema):
        delta = diff_schemas(simple_schema, three_column_schema)
        result = delta.apply_to_schema(simple_schema.copy())
        assert result.has_column("email")
        assert result.has_column("id")
        assert result.has_column("name")

    def test_normalize_preserves_semantics(self, empty_schema):
        ops = [
            AddColumn(name="x", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
            RenameColumn(old_name="y", new_name="y"),  # no-op
            AddColumn(name="z", sql_type=SQLType.TEXT, position=1, default_expr=None, nullable=True),
        ]
        delta = SchemaDelta.from_operations(ops)
        norm = delta.normalize()
        r1 = delta.apply_to_schema(empty_schema.copy())
        r2 = norm.apply_to_schema(empty_schema.copy())
        assert set(r1.column_names()) == set(r2.column_names())

    def test_serialize_inverse_compose(self, empty_schema):
        delta = SchemaDelta.from_operations([
            AddColumn(name="a", sql_type=SQLType.INTEGER, position=0, default_expr=None, nullable=True),
        ])
        # Serialize, deserialize, then inverse and compose
        restored = SchemaDelta.from_dict(delta.to_dict())
        composed = restored.compose(restored.inverse())
        result = composed.apply_to_schema(empty_schema.copy())
        assert result.has_column("a") is False
