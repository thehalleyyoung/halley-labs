"""
Tests for arc.algebra.interaction
==================================

Covers: PhiHomomorphism (φ: Δ_S → End(Δ_D)),
        PsiHomomorphism (ψ: Δ_S → End(Δ_Q)),
        ConstraintViolationDetector,
        cross-sort interactions, and edge cases.
"""

import pytest

try:
    from arc.algebra.interaction import (
        PhiHomomorphism,
        PsiHomomorphism,
        ConstraintViolationDetector,
    )
    from arc.algebra.schema_delta import (
        SchemaDelta,
        AddColumn,
        DropColumn,
        RenameColumn,
        ChangeType,
        ColumnDef,
        Schema as AlgSchema,
        SQLType as AlgSQLType,
        AddConstraint,
        DropConstraint,
        ConstraintType as SchemaConstraintType,
    )
    from arc.algebra.data_delta import (
        DataDelta,
        TypedTuple,
        MultiSet,
        InsertOp,
        DeleteOp,
    )
    from arc.algebra.quality_delta import (
        QualityDelta,
        QualityViolation,
        ViolationType,
        SeverityLevel,
        QualityState,
        ConstraintStatus,
        ConstraintType as QCT,
        ConstraintAdded,
        ConstraintRemoved,
        QualityImprovement,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="module not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema(*cols):
    """Build a Schema with the given (name, SQLType) pairs."""
    s = AlgSchema()
    for i, (name, stype) in enumerate(cols):
        s.add_column(ColumnDef(name=name, sql_type=stype, position=i))
    return s


def _tuple(**kw):
    """Shorthand for TypedTuple.from_dict."""
    return TypedTuple.from_dict(kw)


def _mset(*tuples):
    """Shorthand for MultiSet.from_tuples."""
    return MultiSet.from_tuples(tuples)


def _insert_delta(*tuples):
    """DataDelta with a single InsertOp."""
    return DataDelta.from_operation(InsertOp(_mset(*tuples)))


def _delete_delta(*tuples):
    """DataDelta with a single DeleteOp."""
    return DataDelta.from_operation(DeleteOp(_mset(*tuples)))


# ============================================================================
# 1. PhiHomomorphism — AddColumn
# ============================================================================

class TestPhiAddColumn:

    def test_add_column_extends_insert_tuples(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="age", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        inserts = result.get_all_inserts()
        for t in inserts.unique_tuples():
            assert "age" in t
            assert t["age"] == 0  # default for INTEGER

    def test_add_column_with_default_expr(self):
        t1 = _tuple(id=1)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="active", sql_type=AlgSQLType.BOOLEAN, default_expr="true")
        )
        result = PhiHomomorphism.apply(sd, dd)
        inserts = result.get_all_inserts()
        for t in inserts.unique_tuples():
            assert "active" in t
            assert t["active"] is True

    def test_add_column_varchar_default(self):
        t1 = _tuple(id=1)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="tag", sql_type=AlgSQLType.VARCHAR)
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["tag"] == ""

    def test_add_column_already_present(self):
        t1 = _tuple(id=1, age=25)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="age", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["age"] == 25

    def test_add_column_to_empty_delta(self):
        dd = DataDelta.zero()
        sd = SchemaDelta.from_operation(
            AddColumn(name="x", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        assert result.is_zero()


# ============================================================================
# 2. PhiHomomorphism — DropColumn
# ============================================================================

class TestPhiDropColumn:

    def test_drop_column_removes_field(self):
        t1 = _tuple(id=1, name="Alice", age=30)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(DropColumn(name="age"))
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert "age" not in t
            assert "id" in t
            assert "name" in t

    def test_drop_nonexistent_column(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(DropColumn(name="missing"))
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["id"] == 1
            assert t["name"] == "Alice"

    def test_drop_column_delete_op(self):
        t1 = _tuple(id=1, name="Bob", age=40)
        dd = _delete_delta(t1)
        sd = SchemaDelta.from_operation(DropColumn(name="age"))
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_deletes().unique_tuples():
            assert "age" not in t


# ============================================================================
# 3. PhiHomomorphism — RenameColumn
# ============================================================================

class TestPhiRenameColumn:

    def test_rename_column(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="name", new_name="full_name")
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert "full_name" in t
            assert "name" not in t
            assert t["full_name"] == "Alice"

    def test_rename_nonexistent_column(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="missing", new_name="also_missing")
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["id"] == 1
            assert t["name"] == "Alice"

    def test_rename_preserves_value(self):
        t1 = _tuple(x=42)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(RenameColumn(old_name="x", new_name="y"))
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["y"] == 42


# ============================================================================
# 4. PhiHomomorphism — ChangeType
# ============================================================================

class TestPhiChangeType:

    def test_int_to_float_coercion(self):
        t1 = _tuple(id=1, score=95)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="score",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.FLOAT,
            )
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert isinstance(t["score"], (int, float))

    def test_change_type_on_missing_column(self):
        t1 = _tuple(id=1)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="missing",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.VARCHAR,
            )
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert t["id"] == 1

    def test_int_to_varchar(self):
        t1 = _tuple(id=1, code=42)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="code",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.VARCHAR,
            )
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert "code" in t


# ============================================================================
# 5. PhiHomomorphism — Homomorphism preservation
# ============================================================================

class TestPhiHomomorphismProperty:
    """φ(δ₁ ∘ δ₂) = φ(δ₁) ∘ φ(δ₂)"""

    def test_two_independent_adds(self):
        """Adding two independent columns should satisfy the homomorphism."""
        s1 = SchemaDelta.from_operation(
            AddColumn(name="a", sql_type=AlgSQLType.INTEGER)
        )
        s2 = SchemaDelta.from_operation(
            AddColumn(name="b", sql_type=AlgSQLType.VARCHAR)
        )
        dd = _insert_delta(_tuple(id=1))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)

    def test_independent_rename_and_add(self):
        """Rename on one column and add of another column are independent."""
        s1 = SchemaDelta.from_operation(
            RenameColumn(old_name="x", new_name="y")
        )
        s2 = SchemaDelta.from_operation(
            AddColumn(name="z", sql_type=AlgSQLType.INTEGER)
        )
        dd = _insert_delta(_tuple(id=1, x=10))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)

    def test_independent_drops(self):
        """Dropping two independent columns."""
        s1 = SchemaDelta.from_operation(DropColumn(name="a"))
        s2 = SchemaDelta.from_operation(DropColumn(name="b"))
        dd = _insert_delta(_tuple(id=1, a=10, b=20))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)

    def test_independent_rename_and_drop(self):
        """Rename one column, drop a different one."""
        s1 = SchemaDelta.from_operation(
            RenameColumn(old_name="x", new_name="y")
        )
        s2 = SchemaDelta.from_operation(DropColumn(name="z"))
        dd = _insert_delta(_tuple(id=1, x=10, z=20))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)

    def test_independent_change_types(self):
        """Type changes on different columns."""
        s1 = SchemaDelta.from_operation(
            ChangeType(column_name="a", old_type=AlgSQLType.INTEGER,
                       new_type=AlgSQLType.BIGINT)
        )
        s2 = SchemaDelta.from_operation(
            ChangeType(column_name="b", old_type=AlgSQLType.INTEGER,
                       new_type=AlgSQLType.BIGINT)
        )
        dd = _insert_delta(_tuple(id=1, a=10, b=20))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)


# ============================================================================
# 6. PhiHomomorphism — Identity preservation
# ============================================================================

class TestPhiIdentity:
    """φ(ε) applied to data_delta = data_delta unchanged."""

    def test_identity_preserves_data(self):
        dd = _insert_delta(_tuple(id=1, name="Alice"))
        sd = SchemaDelta.identity()
        result = PhiHomomorphism.apply(sd, dd)
        assert result == dd

    def test_identity_preserves_empty(self):
        dd = DataDelta.zero()
        sd = SchemaDelta.identity()
        result = PhiHomomorphism.apply(sd, dd)
        assert result == dd

    def test_identity_preserves_delete(self):
        dd = _delete_delta(_tuple(id=1, name="Bob"))
        sd = SchemaDelta.identity()
        result = PhiHomomorphism.apply(sd, dd)
        assert result == dd


# ============================================================================
# 7. PsiHomomorphism — per operation type
# ============================================================================

class TestPsiAddColumn:

    def test_add_non_nullable_creates_constraint(self):
        sd = SchemaDelta.from_operation(
            AddColumn(name="email", sql_type=AlgSQLType.VARCHAR, nullable=False)
        )
        qd = QualityDelta.bottom()
        result = PsiHomomorphism.apply(sd, qd)
        additions = result.get_constraint_additions()
        assert len(additions) == 1
        assert additions[0].constraint_id == "nn_email"
        assert additions[0].constraint_type == QCT.NOT_NULL

    def test_add_nullable_no_new_constraint(self):
        sd = SchemaDelta.from_operation(
            AddColumn(name="notes", sql_type=AlgSQLType.TEXT, nullable=True)
        )
        qd = QualityDelta.bottom()
        result = PsiHomomorphism.apply(sd, qd)
        assert result.operation_count() == 0

    def test_add_column_preserves_existing_ops(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        sd = SchemaDelta.from_operation(
            AddColumn(name="y", sql_type=AlgSQLType.INTEGER, nullable=False)
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        assert result.operation_count() == 2


class TestPsiDropColumn:

    def test_drop_column_removes_affected_ops(self):
        v = QualityViolation(
            constraint_id="c_email", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",),
        )
        sd = SchemaDelta.from_operation(DropColumn(name="email"))
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        removals = result.get_constraint_removals()
        assert len(removals) >= 1
        assert any(r.constraint_id == "c_email" for r in removals)
        assert result.violation_count() == 0

    def test_drop_column_preserves_unrelated_ops(self):
        v = QualityViolation(
            constraint_id="c_name", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("name",),
        )
        sd = SchemaDelta.from_operation(DropColumn(name="email"))
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        assert result.violation_count() == 1


class TestPsiRenameColumn:

    def test_rename_updates_columns(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("old_col",),
        )
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="old_col", new_name="new_col")
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        violations = result.get_violations()
        assert len(violations) == 1
        assert "new_col" in violations[0].columns
        assert "old_col" not in violations[0].columns


class TestPsiChangeType:

    def test_type_change_cross_family(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("score",),
        )
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="score",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.VARCHAR,
            )
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        type_violations = [
            op for op in result.get_violations()
            if op.violation_type == ViolationType.TYPE_MISMATCH
        ]
        assert len(type_violations) >= 1

    def test_type_change_same_family_no_violation(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("score",),
        )
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="score",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.BIGINT,
            )
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        type_violations = [
            op for op in result.get_violations()
            if op.violation_type == ViolationType.TYPE_MISMATCH
        ]
        assert len(type_violations) == 0


class TestPsiAddConstraint:

    def test_add_constraint_propagates(self):
        sd = SchemaDelta.from_operation(
            AddConstraint(
                constraint_id="uq_email",
                constraint_type=SchemaConstraintType.UNIQUE,
                columns=("email",),
            )
        )
        qd = QualityDelta.bottom()
        result = PsiHomomorphism.apply(sd, qd)
        additions = result.get_constraint_additions()
        assert len(additions) == 1
        assert additions[0].constraint_id == "uq_email"
        assert additions[0].constraint_type == QCT.UNIQUE

    def test_add_constraint_preserves_existing(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        sd = SchemaDelta.from_operation(
            AddConstraint(
                constraint_id="nn_y",
                constraint_type=SchemaConstraintType.NOT_NULL,
                columns=("y",),
            )
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        assert result.violation_count() == 1
        assert len(result.get_constraint_additions()) == 1


class TestPsiDropConstraint:

    def test_drop_constraint_generates_removal(self):
        ca = ConstraintAdded(
            constraint_id="uq_email", constraint_type=QCT.UNIQUE,
            columns=("email",),
        )
        sd = SchemaDelta.from_operation(
            DropConstraint(constraint_id="uq_email")
        )
        qd = QualityDelta.from_operation(ca)
        result = PsiHomomorphism.apply(sd, qd)
        removals = result.get_constraint_removals()
        assert len(removals) >= 1
        assert any(r.constraint_id == "uq_email" for r in removals)


# ============================================================================
# 8. PsiHomomorphism — Homomorphism preservation
# ============================================================================

class TestPsiHomomorphismProperty:
    """ψ(δ₁ ∘ δ₂) = ψ(δ₁) ∘ ψ(δ₂)"""

    def test_two_independent_adds(self):
        """Adding two independent non-nullable columns."""
        s1 = SchemaDelta.from_operation(
            AddColumn(name="col_a", sql_type=AlgSQLType.INTEGER, nullable=False)
        )
        s2 = SchemaDelta.from_operation(
            AddColumn(name="col_b", sql_type=AlgSQLType.VARCHAR, nullable=False)
        )
        qd = QualityDelta.bottom()
        assert PsiHomomorphism.verify_homomorphism(s1, s2, qd)

    def test_independent_rename_and_add(self):
        """Rename one column, add another (non-nullable)."""
        s1 = SchemaDelta.from_operation(
            RenameColumn(old_name="x", new_name="y")
        )
        s2 = SchemaDelta.from_operation(
            AddColumn(name="z", sql_type=AlgSQLType.INTEGER, nullable=False)
        )
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        qd = QualityDelta.from_operation(v)
        assert PsiHomomorphism.verify_homomorphism(s1, s2, qd)

    def test_independent_drops(self):
        """Drop two independent columns."""
        s1 = SchemaDelta.from_operation(DropColumn(name="a"))
        s2 = SchemaDelta.from_operation(DropColumn(name="b"))
        v = QualityViolation(
            constraint_id="c_other", severity=SeverityLevel.WARNING,
            affected_tuples=3, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("c",),
        )
        qd = QualityDelta.from_operation(v)
        assert PsiHomomorphism.verify_homomorphism(s1, s2, qd)

    def test_identity_preservation(self):
        sd = SchemaDelta.identity()
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        qd = QualityDelta.from_operation(v)
        result = PsiHomomorphism.apply(sd, qd)
        assert result == qd


# ============================================================================
# 9. ConstraintViolationDetector
# ============================================================================

class TestConstraintViolationDetector:

    def test_detect_null_violation(self):
        t1 = _tuple(id=1, email=None)
        dd = _insert_delta(t1)
        constraints = {
            "nn_email": {
                "constraint_type": "NOT_NULL",
                "columns": ["email"],
            }
        }
        result = ConstraintViolationDetector.detect_violations(dd, constraints)
        assert result.violation_count() >= 1
        v = result.get_violations()[0]
        assert v.violation_type == ViolationType.NULL_IN_NON_NULL

    def test_no_violation_when_value_present(self):
        t1 = _tuple(id=1, email="alice@test.com")
        dd = _insert_delta(t1)
        constraints = {
            "nn_email": {
                "constraint_type": "NOT_NULL",
                "columns": ["email"],
            }
        }
        result = ConstraintViolationDetector.detect_violations(dd, constraints)
        assert result.violation_count() == 0

    def test_detect_check_violation(self):
        t1 = _tuple(id=1, age=-5)
        dd = _insert_delta(t1)
        constraints = {
            "chk_age": {
                "constraint_type": "CHECK",
                "columns": ["age"],
                "predicate": "age >= 0",
            }
        }
        result = ConstraintViolationDetector.detect_violations(dd, constraints)
        violations = result.get_violations()
        check_violations = [
            v for v in violations
            if v.violation_type == ViolationType.CHECK_VIOLATION
        ]
        assert len(check_violations) >= 1

    def test_validate_tuple_null(self):
        t = _tuple(id=1, name=None)
        constraints = {
            "nn_name": {
                "constraint_type": "NOT_NULL",
                "columns": ["name"],
            }
        }
        violations = ConstraintViolationDetector.validate_tuple(t, constraints)
        assert len(violations) >= 1
        assert violations[0].violation_type == ViolationType.NULL_IN_NON_NULL

    def test_validate_tuple_passes(self):
        t = _tuple(id=1, name="Alice")
        constraints = {
            "nn_name": {
                "constraint_type": "NOT_NULL",
                "columns": ["name"],
            }
        }
        violations = ConstraintViolationDetector.validate_tuple(t, constraints)
        assert len(violations) == 0

    def test_detect_uniqueness_violation(self):
        t1 = _tuple(id=1, email="dup@test.com")
        t2 = _tuple(id=2, email="dup@test.com")
        existing = _mset(t1)
        dd = _insert_delta(t2)
        constraints = {
            "uq_email": {
                "constraint_type": "UNIQUE",
                "columns": ["email"],
            }
        }
        result = ConstraintViolationDetector.detect_violations(
            dd, constraints, existing_data=existing
        )
        uq_violations = [
            v for v in result.get_violations()
            if v.violation_type == ViolationType.UNIQUENESS_VIOLATION
        ]
        assert len(uq_violations) >= 1

    def test_empty_delta_no_violations(self):
        dd = DataDelta.zero()
        constraints = {
            "nn_email": {
                "constraint_type": "NOT_NULL",
                "columns": ["email"],
            }
        }
        result = ConstraintViolationDetector.detect_violations(dd, constraints)
        assert result.violation_count() == 0

    def test_multiple_constraints(self):
        t1 = _tuple(id=1, email=None, age=-1)
        dd = _insert_delta(t1)
        constraints = {
            "nn_email": {
                "constraint_type": "NOT_NULL",
                "columns": ["email"],
            },
            "chk_age": {
                "constraint_type": "CHECK",
                "columns": ["age"],
                "predicate": "age >= 0",
            },
        }
        result = ConstraintViolationDetector.detect_violations(dd, constraints)
        assert result.violation_count() >= 2


# ============================================================================
# 10. Cross-sort interactions
# ============================================================================

class TestCrossSortInteractions:
    """Schema change affects both data and quality simultaneously."""

    def test_add_column_affects_both(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("name",),
        )
        qd = QualityDelta.from_operation(v)
        sd = SchemaDelta.from_operation(
            AddColumn(name="email", sql_type=AlgSQLType.VARCHAR, nullable=False)
        )

        new_dd = PhiHomomorphism.apply(sd, dd)
        new_qd = PsiHomomorphism.apply(sd, qd)

        for t in new_dd.get_all_inserts().unique_tuples():
            assert "email" in t
        additions = new_qd.get_constraint_additions()
        assert any(ca.constraint_id == "nn_email" for ca in additions)

    def test_drop_column_affects_both(self):
        t1 = _tuple(id=1, name="Alice", email="a@b.com")
        dd = _insert_delta(t1)
        v = QualityViolation(
            constraint_id="c_email", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",),
        )
        qd = QualityDelta.from_operation(v)
        sd = SchemaDelta.from_operation(DropColumn(name="email"))

        new_dd = PhiHomomorphism.apply(sd, dd)
        new_qd = PsiHomomorphism.apply(sd, qd)

        for t in new_dd.get_all_inserts().unique_tuples():
            assert "email" not in t
        assert new_qd.violation_count() == 0

    def test_rename_column_affects_both(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        v = QualityViolation(
            constraint_id="c_name", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("name",),
        )
        qd = QualityDelta.from_operation(v)
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="name", new_name="full_name")
        )

        new_dd = PhiHomomorphism.apply(sd, dd)
        new_qd = PsiHomomorphism.apply(sd, qd)

        for t in new_dd.get_all_inserts().unique_tuples():
            assert "full_name" in t
            assert "name" not in t
        violations = new_qd.get_violations()
        assert all("full_name" in v.columns for v in violations)

    def test_change_type_affects_both(self):
        t1 = _tuple(id=1, score=95)
        dd = _insert_delta(t1)
        v = QualityViolation(
            constraint_id="chk_score", severity=SeverityLevel.WARNING,
            affected_tuples=3, violation_type=ViolationType.RANGE_VIOLATION,
            columns=("score",),
        )
        qd = QualityDelta.from_operation(v)
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="score",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.VARCHAR,
            )
        )

        new_dd = PhiHomomorphism.apply(sd, dd)
        new_qd = PsiHomomorphism.apply(sd, qd)

        assert new_dd.get_all_inserts().cardinality() > 0
        type_violations = [
            op for op in new_qd.get_violations()
            if op.violation_type == ViolationType.TYPE_MISMATCH
        ]
        assert len(type_violations) >= 1

    def test_combined_schema_changes(self):
        t1 = _tuple(id=1, old_name="Alice")
        dd = _insert_delta(t1)
        qd = QualityDelta.bottom()

        sd = SchemaDelta.from_operations([
            AddColumn(name="email", sql_type=AlgSQLType.VARCHAR, nullable=False),
            RenameColumn(old_name="old_name", new_name="name"),
        ])

        new_dd = PhiHomomorphism.apply(sd, dd)
        new_qd = PsiHomomorphism.apply(sd, qd)

        for t in new_dd.get_all_inserts().unique_tuples():
            assert "email" in t
            assert "name" in t
            assert "old_name" not in t

        additions = new_qd.get_constraint_additions()
        assert any(ca.constraint_id == "nn_email" for ca in additions)

    def test_apply_schema_interaction_function(self):
        from arc.algebra.interaction import apply_schema_interaction

        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        qd = QualityDelta.bottom()
        sd = SchemaDelta.from_operation(
            AddColumn(name="age", sql_type=AlgSQLType.INTEGER, nullable=False)
        )

        new_dd, new_qd = apply_schema_interaction(sd, dd, qd)
        for t in new_dd.get_all_inserts().unique_tuples():
            assert "age" in t
        assert new_qd.operation_count() >= 1


# ============================================================================
# 11. Edge cases
# ============================================================================

class TestEdgeCases:

    def test_null_handling_in_tuple(self):
        t1 = _tuple(id=None, name=None)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="extra", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert "extra" in t

    def test_empty_schema_delta_phi(self):
        dd = _insert_delta(_tuple(id=1))
        sd = SchemaDelta.identity()
        result = PhiHomomorphism.apply(sd, dd)
        assert result == dd

    def test_empty_schema_delta_psi(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        qd = QualityDelta.from_operation(v)
        sd = SchemaDelta.identity()
        result = PsiHomomorphism.apply(sd, qd)
        assert result == qd

    def test_empty_data_delta_phi(self):
        dd = DataDelta.zero()
        sd = SchemaDelta.from_operation(
            AddColumn(name="col", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        assert result.is_zero()

    def test_empty_quality_delta_psi(self):
        qd = QualityDelta.bottom()
        sd = SchemaDelta.from_operation(
            AddColumn(name="col", sql_type=AlgSQLType.INTEGER, nullable=True)
        )
        result = PsiHomomorphism.apply(sd, qd)
        assert result.operation_count() == 0

    def test_empty_quality_delta_psi_non_null(self):
        qd = QualityDelta.bottom()
        sd = SchemaDelta.from_operation(
            AddColumn(name="col", sql_type=AlgSQLType.INTEGER, nullable=False)
        )
        result = PsiHomomorphism.apply(sd, qd)
        assert result.operation_count() == 1

    def test_type_coercion_with_none_value(self):
        t1 = _tuple(id=1, val=None)
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            ChangeType(
                column_name="val",
                old_type=AlgSQLType.INTEGER,
                new_type=AlgSQLType.VARCHAR,
            )
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_inserts().unique_tuples():
            assert "val" in t

    def test_apply_to_operation_phi(self):
        t1 = _tuple(id=1, name="Alice")
        insert = InsertOp(_mset(t1))
        op = AddColumn(name="age", sql_type=AlgSQLType.INTEGER)
        result = PhiHomomorphism.apply_to_operation(op, insert)
        assert isinstance(result, InsertOp) or hasattr(result, 'apply')

    def test_apply_to_operation_psi(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("old_col",),
        )
        op = RenameColumn(old_name="old_col", new_name="new_col")
        result = PsiHomomorphism.apply_to_operation(op, v)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_multiple_tuples_in_delta(self):
        t1 = _tuple(id=1, name="Alice")
        t2 = _tuple(id=2, name="Bob")
        t3 = _tuple(id=3, name="Charlie")
        dd = _insert_delta(t1, t2, t3)
        sd = SchemaDelta.from_operation(
            AddColumn(name="age", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        inserts = result.get_all_inserts()
        assert inserts.cardinality() == 3
        for t in inserts.unique_tuples():
            assert "age" in t

    def test_delete_then_add_column(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _delete_delta(t1)
        sd = SchemaDelta.from_operation(
            AddColumn(name="age", sql_type=AlgSQLType.INTEGER)
        )
        result = PhiHomomorphism.apply(sd, dd)
        for t in result.get_all_deletes().unique_tuples():
            assert "age" in t

    def test_phi_drop_constraint_is_noop(self):
        t1 = _tuple(id=1, name="Alice")
        dd = _insert_delta(t1)
        sd = SchemaDelta.from_operation(
            DropConstraint(constraint_id="some_constraint")
        )
        result = PhiHomomorphism.apply(sd, dd)
        assert result == dd

    def test_psi_with_distribution_shift(self):
        from arc.algebra.quality_delta import DistributionShift, DistributionSummary
        old_d = DistributionSummary(mean=10.0, stddev=2.0)
        new_d = DistributionSummary(mean=50.0, stddev=10.0)
        ds = DistributionShift(
            column="val", old_dist=old_d, new_dist=new_d,
            psi_score=0.2, ks_statistic=0.1,
        )
        qd = QualityDelta.from_operation(ds)
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="val", new_name="value")
        )
        result = PsiHomomorphism.apply(sd, qd)
        shifts = result.get_distribution_shifts()
        assert len(shifts) == 1
        assert shifts[0].column == "value"

    def test_constraint_type_mapping(self):
        for sct, expected_qct in [
            (SchemaConstraintType.NOT_NULL, QCT.NOT_NULL),
            (SchemaConstraintType.UNIQUE, QCT.UNIQUE),
            (SchemaConstraintType.PRIMARY_KEY, QCT.PRIMARY_KEY),
            (SchemaConstraintType.FOREIGN_KEY, QCT.FOREIGN_KEY),
            (SchemaConstraintType.CHECK, QCT.CHECK),
        ]:
            sd = SchemaDelta.from_operation(
                AddConstraint(
                    constraint_id=f"test_{sct.value}",
                    constraint_type=sct,
                    columns=("col",),
                )
            )
            result = PsiHomomorphism.apply(sd, QualityDelta.bottom())
            additions = result.get_constraint_additions()
            assert len(additions) == 1
            assert additions[0].constraint_type == expected_qct

    def test_verify_homomorphism_with_empty(self):
        s1 = SchemaDelta.identity()
        s2 = SchemaDelta.identity()
        dd = _insert_delta(_tuple(id=1))
        assert PhiHomomorphism.verify_homomorphism(s1, s2, dd)

    def test_verify_psi_homomorphism_with_empty(self):
        s1 = SchemaDelta.identity()
        s2 = SchemaDelta.identity()
        qd = QualityDelta.bottom()
        assert PsiHomomorphism.verify_homomorphism(s1, s2, qd)
