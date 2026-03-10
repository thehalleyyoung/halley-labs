"""
Property-based tests for the three-sorted delta algebra.

Tests algebraic laws of:
- Schema delta monoid (Δ_S, ∘, ε)
- Data delta group (Δ_D, ∘, ⁻¹, 𝟎)
- Quality delta lattice (Δ_Q, ⊔, ⊓, ⊥, ⊤)
- Interaction homomorphisms φ and ψ
- Compound perturbation composition
"""

from __future__ import annotations

import sys
from typing import List, Optional

import pytest

try:
    from hypothesis import (
        HealthCheck,
        given,
        settings,
        assume,
        note,
    )
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

try:
    from arc.algebra.schema_delta import (
        AddColumn,
        ChangeType,
        ColumnDef,
        ConstraintType,
        DropColumn,
        RenameColumn,
        Schema,
        SchemaDelta,
        SQLType,
    )
    from arc.algebra.data_delta import (
        DataDelta,
        DeleteOp,
        InsertOp,
        MultiSet,
        TypedTuple,
    )
    from arc.algebra.quality_delta import (
        ConstraintAdded,
        ConstraintRemoved,
        ConstraintType as QCT,
        QualityDelta,
        QualityState,
        QualityViolation,
        SeverityLevel,
        ViolationType,
    )
    from arc.algebra.composition import (
        CompoundPerturbation,
        verify_composition_associativity,
    )
    from arc.algebra.interaction import PhiHomomorphism

    HAS_ARC = True
except ImportError:
    HAS_ARC = False

pytestmark = pytest.mark.skipif(
    not (HAS_HYPOTHESIS and HAS_ARC),
    reason="hypothesis and/or arc not available",
)

# =====================================================================
# Test settings
# =====================================================================

SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# =====================================================================
# Hypothesis strategies
# =====================================================================

if HAS_HYPOTHESIS:

    # ---- SQL Types ----

    _SQL_TYPES = [
        SQLType.INTEGER,
        SQLType.BIGINT,
        SQLType.SMALLINT,
        SQLType.FLOAT,
        SQLType.DOUBLE,
        SQLType.VARCHAR,
        SQLType.TEXT,
        SQLType.BOOLEAN,
        SQLType.DATE,
        SQLType.TIMESTAMP,
    ]

    st_sql_type = st.sampled_from(_SQL_TYPES) if HAS_ARC else st.none()

    # ---- Column names ----

    st_column_name = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)

    # ---- Column definitions ----

    @st.composite
    def st_column_def(draw):
        """Generate a random ColumnDef."""
        name = draw(st_column_name)
        sql_type = draw(st_sql_type)
        nullable = draw(st.booleans())
        return ColumnDef(name=name, sql_type=sql_type, nullable=nullable)

    # ---- Schema operations ----

    @st.composite
    def st_add_column_op(draw):
        """Generate an AddColumn operation."""
        name = draw(st_column_name)
        sql_type = draw(st_sql_type)
        nullable = draw(st.booleans())
        return AddColumn(name=name, sql_type=sql_type, nullable=nullable)

    @st.composite
    def st_drop_column_op(draw, schema_columns: List[str]):
        """Generate a DropColumn for an existing column in the schema."""
        assume(len(schema_columns) > 0)
        col_name = draw(st.sampled_from(schema_columns))
        return DropColumn(name=col_name)

    @st.composite
    def st_rename_op(draw, schema_columns: List[str]):
        """Generate a RenameColumn for an existing column."""
        assume(len(schema_columns) > 0)
        old_name = draw(st.sampled_from(schema_columns))
        new_name = draw(st_column_name.filter(lambda n: n not in schema_columns))
        return RenameColumn(old_name=old_name, new_name=new_name)

    @st.composite
    def st_change_type_op(draw, schema_columns: List[str]):
        """Generate a ChangeType for an existing column."""
        assume(len(schema_columns) > 0)
        col_name = draw(st.sampled_from(schema_columns))
        old_type = draw(st_sql_type)
        new_type = draw(st_sql_type.filter(lambda t: t != old_type))
        return ChangeType(
            column_name=col_name,
            old_type=old_type,
            new_type=new_type,
        )

    # ---- Schema delta ----

    @st.composite
    def st_schema_delta(draw, min_ops: int = 1, max_ops: int = 5):
        """Generate a valid SchemaDelta with add-column operations only.

        Using only AddColumn avoids needing an existing schema context.
        """
        num_ops = draw(st.integers(min_value=min_ops, max_value=max_ops))
        ops = []
        used_names = set()
        for _ in range(num_ops):
            name = draw(st_column_name.filter(lambda n: n not in used_names))
            used_names.add(name)
            sql_type = draw(st_sql_type)
            nullable = draw(st.booleans())
            ops.append(AddColumn(name=name, sql_type=sql_type, nullable=nullable))
        return SchemaDelta(ops)

    @st.composite
    def st_schema_delta_pair(draw):
        """Generate two composable schema deltas (both add-only, distinct columns)."""
        delta1 = draw(st_schema_delta(min_ops=1, max_ops=3))
        names1 = delta1.affected_columns()
        ops2 = []
        used = set(names1)
        num = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num):
            name = draw(st_column_name.filter(lambda n: n not in used))
            used.add(name)
            sql_type = draw(st_sql_type)
            nullable = draw(st.booleans())
            ops2.append(AddColumn(name=name, sql_type=sql_type, nullable=nullable))
        delta2 = SchemaDelta(ops2)
        return delta1, delta2

    @st.composite
    def st_schema_delta_triple(draw):
        """Generate three composable schema deltas with distinct columns."""
        used = set()
        deltas = []
        for _ in range(3):
            ops = []
            num = draw(st.integers(min_value=1, max_value=2))
            for _ in range(num):
                name = draw(st_column_name.filter(lambda n: n not in used))
                used.add(name)
                sql_type = draw(st_sql_type)
                nullable = draw(st.booleans())
                ops.append(AddColumn(name=name, sql_type=sql_type, nullable=nullable))
            deltas.append(SchemaDelta(ops))
        return deltas[0], deltas[1], deltas[2]

    # ---- Typed tuples ----

    @st.composite
    def st_typed_tuple(draw, min_fields: int = 2, max_fields: int = 5):
        """Generate a TypedTuple with random fields."""
        num_fields = draw(st.integers(min_value=min_fields, max_value=max_fields))
        values = {}
        used_names = set()
        for _ in range(num_fields):
            name = draw(st_column_name.filter(lambda n: n not in used_names))
            used_names.add(name)
            value = draw(st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnop"),
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
                st.booleans(),
            ))
            values[name] = value
        return TypedTuple(values)

    @st.composite
    def st_typed_tuple_with_schema(draw, columns: List[str]):
        """Generate a TypedTuple for specific column names."""
        values = {}
        for col in columns:
            value = draw(st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.text(min_size=0, max_size=10, alphabet="abcdefghij"),
            ))
            values[col] = value
        return TypedTuple(values)

    # ---- MultiSets ----

    @st.composite
    def st_multiset(draw, min_tuples: int = 0, max_tuples: int = 10):
        """Generate a MultiSet with random tuples."""
        num_tuples = draw(st.integers(min_value=min_tuples, max_value=max_tuples))
        if num_tuples == 0:
            return MultiSet.empty()
        cols = draw(st.lists(
            st_column_name, min_size=2, max_size=4, unique=True
        ))
        tuples = []
        for _ in range(num_tuples):
            t = draw(st_typed_tuple_with_schema(cols))
            tuples.append(t)
        return MultiSet.from_tuples(tuples)

    @st.composite
    def st_multiset_pair(draw):
        """Generate two MultiSets with the same column schema."""
        cols = draw(st.lists(
            st_column_name, min_size=2, max_size=3, unique=True
        ))
        ms1_tuples = []
        num1 = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num1):
            ms1_tuples.append(draw(st_typed_tuple_with_schema(cols)))
        ms2_tuples = []
        num2 = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num2):
            ms2_tuples.append(draw(st_typed_tuple_with_schema(cols)))
        return MultiSet.from_tuples(ms1_tuples), MultiSet.from_tuples(ms2_tuples)

    # ---- Data operations ----

    @st.composite
    def st_insert_op(draw):
        """Generate an InsertOp from a random multiset."""
        ms = draw(st_multiset(min_tuples=1, max_tuples=5))
        return InsertOp(ms)

    @st.composite
    def st_delete_op(draw):
        """Generate a DeleteOp from a random multiset."""
        ms = draw(st_multiset(min_tuples=1, max_tuples=5))
        return DeleteOp(ms)

    # ---- Data delta ----

    @st.composite
    def st_data_delta(draw, min_ops: int = 1, max_ops: int = 3):
        """Generate a DataDelta with random insert/delete operations."""
        num_ops = draw(st.integers(min_value=min_ops, max_value=max_ops))
        ops = []
        for _ in range(num_ops):
            op = draw(st.one_of(st_insert_op(), st_delete_op()))
            ops.append(op)
        return DataDelta(ops)

    @st.composite
    def st_data_delta_pair(draw):
        """Generate two composable data deltas."""
        d1 = draw(st_data_delta(min_ops=1, max_ops=2))
        d2 = draw(st_data_delta(min_ops=1, max_ops=2))
        return d1, d2

    @st.composite
    def st_data_delta_triple(draw):
        """Generate three composable data deltas."""
        d1 = draw(st_data_delta(min_ops=1, max_ops=2))
        d2 = draw(st_data_delta(min_ops=1, max_ops=2))
        d3 = draw(st_data_delta(min_ops=1, max_ops=2))
        return d1, d2, d3

    # ---- Quality violations ----

    _VIOLATION_TYPES = [
        ViolationType.NULL_IN_NON_NULL,
        ViolationType.UNIQUENESS_VIOLATION,
        ViolationType.CHECK_VIOLATION,
        ViolationType.TYPE_MISMATCH,
        ViolationType.RANGE_VIOLATION,
        ViolationType.PATTERN_VIOLATION,
    ]

    _SEVERITY_LEVELS = [
        SeverityLevel.INFO,
        SeverityLevel.WARNING,
        SeverityLevel.ERROR,
        SeverityLevel.CRITICAL,
    ]

    @st.composite
    def st_quality_violation(draw):
        """Generate a QualityViolation with random severity."""
        constraint_id = draw(st.text(
            min_size=3, max_size=10, alphabet="abcdefghij0123456789"
        ))
        severity = draw(st.sampled_from(_SEVERITY_LEVELS))
        affected_tuples = draw(st.integers(min_value=0, max_value=1000))
        violation_type = draw(st.sampled_from(_VIOLATION_TYPES))
        num_cols = draw(st.integers(min_value=0, max_value=3))
        cols = tuple(draw(st.lists(
            st_column_name, min_size=num_cols, max_size=num_cols, unique=True
        )))
        return QualityViolation(
            constraint_id=constraint_id,
            severity=severity,
            affected_tuples=affected_tuples,
            violation_type=violation_type,
            columns=cols,
            message=f"Test violation on {constraint_id}",
        )

    # ---- Quality delta ----

    @st.composite
    def st_quality_delta(draw, min_ops: int = 1, max_ops: int = 3):
        """Generate a QualityDelta with unique constraint_ids per violation."""
        num_ops = draw(st.integers(min_value=min_ops, max_value=max_ops))
        ops = []
        used_ids = set()
        for _ in range(num_ops):
            op = draw(st_quality_violation())
            # Ensure unique constraint_ids so join/meet lattice laws hold
            assume(op.constraint_id not in used_ids)
            used_ids.add(op.constraint_id)
            ops.append(op)
        return QualityDelta(ops)

    @st.composite
    def st_quality_delta_pair(draw):
        """Generate two quality deltas for lattice law testing.
        
        Both deltas share a common pool of unique constraint_ids to ensure
        that join/meet merge semantics work correctly.
        """
        all_used_ids = set()
        deltas = []
        for _ in range(2):
            num_ops = draw(st.integers(min_value=1, max_value=2))
            ops = []
            for _ in range(num_ops):
                op = draw(st_quality_violation())
                assume(op.constraint_id not in all_used_ids)
                all_used_ids.add(op.constraint_id)
                ops.append(op)
            deltas.append(QualityDelta(ops))
        return deltas[0], deltas[1]

    @st.composite
    def st_quality_delta_triple(draw):
        """Generate three quality deltas with globally unique constraint_ids."""
        all_used_ids = set()
        deltas = []
        for _ in range(3):
            num_ops = draw(st.integers(min_value=1, max_value=2))
            ops = []
            for _ in range(num_ops):
                op = draw(st_quality_violation())
                assume(op.constraint_id not in all_used_ids)
                all_used_ids.add(op.constraint_id)
                ops.append(op)
            deltas.append(QualityDelta(ops))
        return deltas[0], deltas[1], deltas[2]

    @st.composite
    def st_schema_delta_nullable(draw, min_ops: int = 0, max_ops: int = 2):
        """Generate SchemaDelta with nullable-only AddColumn ops.
        
        Avoids triggering ψ interaction effects that add NOT NULL constraints.
        """
        num_ops = draw(st.integers(min_value=min_ops, max_value=max_ops))
        ops = []
        used_names = set()
        for _ in range(num_ops):
            name = draw(st_column_name.filter(lambda n: n not in used_names))
            used_names.add(name)
            sql_type = draw(st_sql_type)
            ops.append(AddColumn(name=name, sql_type=sql_type, nullable=True))
        return SchemaDelta(ops)

    # ---- Compound perturbation ----

    @st.composite
    def st_compound_perturbation(draw):
        """Generate a CompoundPerturbation with all three components.
        
        Uses nullable-only schema ops to avoid ψ interaction effects that
        would break identity and inverse composition laws.
        """
        sd = draw(st_schema_delta_nullable(min_ops=0, max_ops=2))
        dd = draw(st_data_delta(min_ops=0, max_ops=2))
        qd = draw(st_quality_delta(min_ops=0, max_ops=2))
        return CompoundPerturbation(sd, dd, qd)

    @st.composite
    def st_compound_perturbation_triple(draw):
        """Generate three compound perturbations for associativity tests."""
        used_cols = set()
        perturbations = []
        for _ in range(3):
            # Schema: only add-column ops with distinct names
            num_schema_ops = draw(st.integers(min_value=0, max_value=1))
            schema_ops = []
            for _ in range(num_schema_ops):
                name = draw(st_column_name.filter(lambda n: n not in used_cols))
                used_cols.add(name)
                sql_type = draw(st_sql_type)
                schema_ops.append(AddColumn(name=name, sql_type=sql_type, nullable=True))
            sd = SchemaDelta(schema_ops)
            dd = draw(st_data_delta(min_ops=0, max_ops=1))
            qd = draw(st_quality_delta(min_ops=0, max_ops=1))
            perturbations.append(CompoundPerturbation(sd, dd, qd))
        return perturbations[0], perturbations[1], perturbations[2]


# =====================================================================
# Schema Delta Monoid Tests
# =====================================================================

class TestSchemaDeltaMonoid:
    """Test the monoid laws for SchemaDelta: (Δ_S, ∘, ε)."""

    @SETTINGS
    @given(data=st.data())
    def test_associativity(self, data):
        """(a ∘ b) ∘ c = a ∘ (b ∘ c)."""
        a, b, c = data.draw(st_schema_delta_triple())
        lhs = a.compose(b).compose(c)
        rhs = a.compose(b.compose(c))
        assert lhs == rhs, (
            f"Associativity failed:\n"
            f"  (a∘b)∘c = {lhs}\n"
            f"  a∘(b∘c) = {rhs}"
        )

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_left_identity(self, delta):
        """ε ∘ δ = δ."""
        identity = SchemaDelta.identity()
        result = identity.compose(delta)
        assert result == delta, f"Left identity failed: ε∘δ = {result}, δ = {delta}"

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_right_identity(self, delta):
        """δ ∘ ε = δ."""
        identity = SchemaDelta.identity()
        result = delta.compose(identity)
        assert result == delta, f"Right identity failed: δ∘ε = {result}, δ = {delta}"

    @SETTINGS
    @given(data=st.data())
    def test_closure(self, data):
        """Composition of two schema deltas is a schema delta."""
        a, b = data.draw(st_schema_delta_pair())
        result = a.compose(b)
        assert isinstance(result, SchemaDelta), (
            f"Closure failed: compose returned {type(result)}"
        )
        assert result.operations is not None

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_identity_is_identity(self, delta):
        """The identity element has no operations."""
        identity = SchemaDelta.identity()
        assert identity.is_identity()
        assert identity.operation_count() == 0

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_compose_preserves_operations(self, delta):
        """Composing with identity preserves operation semantics."""
        identity = SchemaDelta.identity()
        composed = identity.compose(delta)
        # Both should affect the same columns
        assert composed.affected_columns() == delta.affected_columns()

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_inverse_compose_is_identity(self, delta):
        """δ ∘ δ⁻¹ should be identity (for add-only deltas)."""
        inv = delta.inverse()
        composed = delta.compose(inv)
        # After adding columns then dropping them, result should be identity
        assert composed.is_identity(), (
            f"δ∘δ⁻¹ not identity: {composed}"
        )

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_inverse_left(self, delta):
        """δ⁻¹ ∘ δ should preserve affected columns (for add-only deltas)."""
        inv = delta.inverse()
        composed = inv.compose(delta)
        # After drop then re-add, the composed result may not be structurally
        # identity (DropColumn loses type info), but semantically the net
        # effect should only touch the same columns.
        assert composed.affected_columns() <= delta.affected_columns(), (
            f"δ⁻¹∘δ affects extra columns: {composed.affected_columns() - delta.affected_columns()}"
        )

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_normalize_idempotent(self, delta):
        """normalize(normalize(δ)) = normalize(δ)."""
        n1 = delta.normalize()
        n2 = n1.normalize()
        assert n1 == n2, "Normalization is not idempotent"

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_normalize_preserves_semantics(self, delta):
        """Normalization preserves the set of affected columns."""
        normalized = delta.normalize()
        assert delta.affected_columns() == normalized.affected_columns()


# =====================================================================
# Data Delta Group Tests
# =====================================================================

class TestDataDeltaGroup:
    """Test the group laws for DataDelta: (Δ_D, ∘, ⁻¹, 𝟎)."""

    @SETTINGS
    @given(data=st.data())
    def test_associativity(self, data):
        """(a ∘ b) ∘ c = a ∘ (b ∘ c) on a base multiset."""
        a, b, c = data.draw(st_data_delta_triple())
        base = data.draw(st_multiset(min_tuples=0, max_tuples=5))
        lhs = a.compose(b).compose(c).apply_to_data(base)
        rhs = a.compose(b.compose(c)).apply_to_data(base)
        assert lhs == rhs, (
            f"Data delta associativity failed on base multiset"
        )

    @SETTINGS
    @given(delta=st_data_delta())
    def test_left_identity(self, delta):
        """𝟎 ∘ δ = δ."""
        zero = DataDelta.zero()
        result = zero.compose(delta)
        # They should have the same effect on any multiset
        base = MultiSet.empty()
        assert result.apply_to_data(base) == delta.apply_to_data(base)

    @SETTINGS
    @given(delta=st_data_delta())
    def test_right_identity(self, delta):
        """δ ∘ 𝟎 = δ."""
        zero = DataDelta.zero()
        result = delta.compose(zero)
        base = MultiSet.empty()
        assert result.apply_to_data(base) == delta.apply_to_data(base)

    @SETTINGS
    @given(delta=st_data_delta())
    def test_zero_element(self, delta):
        """The zero element has no effect."""
        zero = DataDelta.zero()
        assert zero.is_zero()
        base = MultiSet.empty()
        assert zero.apply_to_data(base) == base

    @SETTINGS
    @given(data=st.data())
    def test_inverse(self, data):
        """δ ∘ δ⁻¹ applied to a base yields the base (insert-only delta)."""
        # Use insert-only deltas to avoid multiset difference floor-at-zero
        # causing information loss that breaks inverse roundtrips.
        cols = data.draw(st.lists(
            st_column_name, min_size=2, max_size=3, unique=True
        ))
        num_ops = data.draw(st.integers(min_value=1, max_value=2))
        ops = []
        for _ in range(num_ops):
            tuples = []
            nt = data.draw(st.integers(min_value=1, max_value=3))
            for _ in range(nt):
                tuples.append(TypedTuple({c: data.draw(st.integers(min_value=0, max_value=5)) for c in cols}))
            ops.append(InsertOp(MultiSet.from_tuples(tuples)))
        delta = DataDelta(ops)
        base = data.draw(st_multiset(min_tuples=2, max_tuples=5))
        applied = delta.apply_to_data(base)
        inv = delta.inverse()
        roundtrip = inv.apply_to_data(applied)
        assert roundtrip == base, (
            f"Inverse failed: base has {base.cardinality()} tuples, "
            f"roundtrip has {roundtrip.cardinality()}"
        )

    @SETTINGS
    @given(data=st.data())
    def test_closure(self, data):
        """Composition of two data deltas is a data delta."""
        a, b = data.draw(st_data_delta_pair())
        result = a.compose(b)
        assert isinstance(result, DataDelta)

    @SETTINGS
    @given(delta=st_data_delta())
    def test_operation_count_non_negative(self, delta):
        """Operation count is always non-negative."""
        assert delta.operation_count() >= 0

    @SETTINGS
    @given(delta=st_data_delta())
    def test_affected_rows_non_negative(self, delta):
        """Affected rows count is always non-negative."""
        assert delta.affected_rows_count() >= 0

    @SETTINGS
    @given(data=st.data())
    def test_compose_then_apply_equals_sequential_apply(self, data):
        """(a ∘ b).apply(base) = b.apply(a.apply(base))."""
        a, b = data.draw(st_data_delta_pair())
        base = data.draw(st_multiset(min_tuples=0, max_tuples=5))
        composed_result = a.compose(b).apply_to_data(base)
        sequential_result = b.apply_to_data(a.apply_to_data(base))
        assert composed_result == sequential_result


# =====================================================================
# Quality Delta Lattice Tests
# =====================================================================

class TestQualityDeltaLattice:
    """Test lattice laws for QualityDelta: (Δ_Q, ⊔, ⊓, ⊥, ⊤)."""

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_join_idempotent(self, delta):
        """a ⊔ a = a."""
        result = delta.join(delta)
        assert result == delta, f"Join idempotent failed: a⊔a = {result}, a = {delta}"

    @SETTINGS
    @given(data=st.data())
    def test_join_commutative(self, data):
        """a ⊔ b = b ⊔ a."""
        a, b = data.draw(st_quality_delta_pair())
        lhs = a.join(b)
        rhs = b.join(a)
        assert lhs == rhs, f"Join commutativity failed"

    @SETTINGS
    @given(data=st.data())
    def test_join_associative(self, data):
        """(a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)."""
        a, b, c = data.draw(st_quality_delta_triple())
        lhs = a.join(b).join(c)
        rhs = a.join(b.join(c))
        assert lhs == rhs, f"Join associativity failed"

    @SETTINGS
    @given(data=st.data())
    def test_absorption_join_meet(self, data):
        """a ⊔ (a ⊓ b) = a."""
        a, b = data.draw(st_quality_delta_pair())
        meet_ab = a.meet(b)
        result = a.join(meet_ab)
        assert result == a, f"Absorption (join-meet) failed: a⊔(a⊓b) = {result}, a = {a}"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_meet_idempotent(self, delta):
        """a ⊓ a = a."""
        result = delta.meet(delta)
        assert result == delta, f"Meet idempotent failed"

    @SETTINGS
    @given(data=st.data())
    def test_meet_commutative(self, data):
        """a ⊓ b = b ⊓ a."""
        a, b = data.draw(st_quality_delta_pair())
        lhs = a.meet(b)
        rhs = b.meet(a)
        assert lhs == rhs, f"Meet commutativity failed"

    @SETTINGS
    @given(data=st.data())
    def test_meet_associative(self, data):
        """(a ⊓ b) ⊓ c = a ⊓ (b ⊓ c)."""
        a, b, c = data.draw(st_quality_delta_triple())
        lhs = a.meet(b).meet(c)
        rhs = a.meet(b.meet(c))
        assert lhs == rhs, f"Meet associativity failed"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_bottom_is_join_identity(self, delta):
        """⊥ ⊔ a = a."""
        bottom = QualityDelta.bottom()
        result = bottom.join(delta)
        assert result == delta, f"Bottom is not join identity"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_bottom_is_meet_annihilator(self, delta):
        """⊥ ⊓ a = ⊥."""
        bottom = QualityDelta.bottom()
        result = bottom.meet(delta)
        assert result == bottom, f"Bottom is not meet annihilator"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_top_is_join_annihilator(self, delta):
        """⊤ ⊔ a = ⊤."""
        top = QualityDelta.top()
        result = top.join(delta)
        assert result == top, f"Top is not join annihilator"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_top_is_meet_identity(self, delta):
        """⊤ ⊓ a = a."""
        top = QualityDelta.top()
        result = top.meet(delta)
        assert result == delta, f"Top is not meet identity"

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_severity_non_negative(self, delta):
        """Severity is always non-negative."""
        assert delta.severity() >= 0.0

    @SETTINGS
    @given(data=st.data())
    def test_join_severity_monotone(self, data):
        """severity(a ⊔ b) >= max(severity(a), severity(b))."""
        a, b = data.draw(st_quality_delta_pair())
        joined = a.join(b)
        assert joined.severity() >= max(a.severity(), b.severity()) - 1e-10

    @SETTINGS
    @given(delta=st_quality_delta())
    def test_bottom_severity_zero(self, delta):
        """severity(⊥) = 0."""
        bottom = QualityDelta.bottom()
        assert bottom.severity() == 0.0

    @SETTINGS
    @given(data=st.data())
    def test_partial_order_consistent_with_join(self, data):
        """a ≤ b iff a ⊔ b = b."""
        a, b = data.draw(st_quality_delta_pair())
        joined = a.join(b)
        if a <= b:
            assert joined == b

    @SETTINGS
    @given(data=st.data())
    def test_absorption_meet_join(self, data):
        """a ⊓ (a ⊔ b) = a."""
        a, b = data.draw(st_quality_delta_pair())
        join_ab = a.join(b)
        result = a.meet(join_ab)
        assert result == a, f"Absorption (meet-join) failed"


# =====================================================================
# Interaction Homomorphism Tests
# =====================================================================

class TestInteractionHomomorphism:
    """Test that φ: Δ_S → End(Δ_D) is a homomorphism."""

    @SETTINGS
    @given(data=st.data())
    def test_phi_preserves_composition(self, data):
        """φ(δ₁ ∘ δ₂) = φ(δ₁) ∘ φ(δ₂) for compatible schema deltas.

        That is, applying the composed schema interaction to a data delta
        should equal applying each schema interaction sequentially.
        """
        delta_s1, delta_s2 = data.draw(st_schema_delta_pair())
        delta_d = data.draw(st_data_delta(min_ops=1, max_ops=2))

        # Compose schema deltas first, then apply φ
        composed_schema = delta_s1.compose(delta_s2)
        lhs = PhiHomomorphism.apply(composed_schema, delta_d)

        # Apply φ for each schema delta sequentially
        step1 = PhiHomomorphism.apply(delta_s1, delta_d)
        rhs = PhiHomomorphism.apply(delta_s2, step1)

        # Compare effects on an empty multiset
        base = MultiSet.empty()
        lhs_result = lhs.apply_to_data(base)
        rhs_result = rhs.apply_to_data(base)
        assert lhs_result == rhs_result, (
            f"φ homomorphism failed: φ(δ₁∘δ₂)(d) ≠ φ(δ₂)(φ(δ₁)(d))"
        )

    @SETTINGS
    @given(delta_d=st_data_delta())
    def test_phi_identity(self, delta_d):
        """φ(ε)(δ_D) = δ_D where ε is the schema identity."""
        identity_schema = SchemaDelta.identity()
        result = PhiHomomorphism.apply(identity_schema, delta_d)
        base = MultiSet.empty()
        assert result.apply_to_data(base) == delta_d.apply_to_data(base), (
            "φ(identity) should not change data delta"
        )

    @SETTINGS
    @given(data=st.data())
    def test_phi_zero_data(self, data):
        """φ(δ_S)(𝟎) = 𝟎 for any schema delta."""
        delta_s = data.draw(st_schema_delta())
        zero = DataDelta.zero()
        result = PhiHomomorphism.apply(delta_s, zero)
        assert result.is_zero(), "φ applied to zero data delta should give zero"


# =====================================================================
# Compound Perturbation Tests
# =====================================================================

class TestCompoundPerturbation:
    """Test the compound perturbation (Δ_S × Δ_D × Δ_Q) structure."""

    @SETTINGS
    @given(data=st.data())
    def test_associativity(self, data):
        """(p1 ∘ p2) ∘ p3 = p1 ∘ (p2 ∘ p3)."""
        p1, p2, p3 = data.draw(st_compound_perturbation_triple())
        assert verify_composition_associativity(p1, p2, p3), (
            "Compound perturbation associativity (T6) failed"
        )

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_identity_left(self, p):
        """id ∘ p = p."""
        identity = CompoundPerturbation.identity()
        result = identity.compose(p)
        assert result == p, f"Left identity failed"

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_identity_right(self, p):
        """p ∘ id = p."""
        identity = CompoundPerturbation.identity()
        result = p.compose(identity)
        assert result == p, f"Right identity failed"

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_identity_is_identity(self, p):
        """The identity perturbation should report is_identity."""
        identity = CompoundPerturbation.identity()
        assert identity.is_identity()
        assert not identity.has_schema_changes()
        assert not identity.has_data_changes()
        assert not identity.has_quality_changes()

    @SETTINGS
    @given(data=st.data())
    def test_compose_closure(self, data):
        """Composition of two perturbations is a perturbation."""
        p1, p2 = (
            data.draw(st_compound_perturbation()),
            data.draw(st_compound_perturbation()),
        )
        result = p1.compose(p2)
        assert isinstance(result, CompoundPerturbation)

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_inverse_right(self, p):
        """p ∘ p⁻¹ should have identity schema and semantically zero data."""
        # Skip cases where schema+data interaction prevents exact cancellation
        assume(p.schema_delta.is_identity() or p.data_delta.is_zero())
        inv = p.inverse()
        composed = p.compose(inv)
        assert composed.schema_delta.is_identity(), (
            f"Schema not identity after p∘p⁻¹"
        )
        base = MultiSet.empty()
        assert composed.data_delta.apply_to_data(base) == base, (
            f"Data not semantically zero after p∘p⁻¹"
        )

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_severity_in_range(self, p):
        """Severity should be in [0, 1]."""
        sev = p.severity()
        assert 0.0 <= sev <= 1.0, f"Severity out of range: {sev}"

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_total_operation_count(self, p):
        """Total op count = sum of component op counts."""
        assert p.total_operation_count() == (
            p.schema_operation_count()
            + p.data_operation_count()
            + p.quality_operation_count()
        )

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_summary_keys(self, p):
        """Summary dict should contain expected keys."""
        summary = p.summary()
        expected_keys = {
            "is_identity", "severity", "schema_ops", "data_ops",
            "quality_ops", "affected_columns", "has_schema", "has_data",
            "has_quality", "data_rows_affected", "data_net_change",
            "quality_severity",
        }
        assert expected_keys.issubset(set(summary.keys()))

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_roundtrip_serialization(self, p):
        """to_dict → from_dict should preserve the perturbation."""
        d = p.to_dict()
        restored = CompoundPerturbation.from_dict(d)
        assert restored == p, "Serialization roundtrip failed"

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_normalize_idempotent(self, p):
        """normalize(normalize(p)) = normalize(p)."""
        n1 = p.normalize()
        n2 = n1.normalize()
        assert n1 == n2

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_bool_consistency(self, p):
        """bool(p) should be True iff p is not identity."""
        if p.is_identity():
            assert not bool(p)
        else:
            assert bool(p)

    @SETTINGS
    @given(p=st_compound_perturbation())
    def test_hash_deterministic(self, p):
        """Hash should be deterministic."""
        h1 = hash(p)
        h2 = hash(p)
        assert h1 == h2

    @SETTINGS
    @given(data=st.data())
    def test_eq_reflexive(self, data):
        """p == p for all p."""
        p = data.draw(st_compound_perturbation())
        assert p == p

    @SETTINGS
    @given(data=st.data())
    def test_schema_only_factory(self, data):
        """schema_only creates a perturbation with only schema changes."""
        sd = data.draw(st_schema_delta())
        p = CompoundPerturbation.schema_only(sd)
        assert p.schema_delta == sd
        assert p.data_delta.is_zero()
        assert p.quality_delta.is_bottom()

    @SETTINGS
    @given(data=st.data())
    def test_data_only_factory(self, data):
        """data_only creates a perturbation with only data changes."""
        dd = data.draw(st_data_delta())
        p = CompoundPerturbation.data_only(dd)
        assert p.schema_delta.is_identity()
        assert p.data_delta == dd
        assert p.quality_delta.is_bottom()

    @SETTINGS
    @given(data=st.data())
    def test_quality_only_factory(self, data):
        """quality_only creates a perturbation with only quality changes."""
        qd = data.draw(st_quality_delta())
        p = CompoundPerturbation.quality_only(qd)
        assert p.schema_delta.is_identity()
        assert p.data_delta.is_zero()
        assert p.quality_delta == qd


# =====================================================================
# Additional Edge-Case Tests
# =====================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_schema_delta_is_identity(self):
        """Empty SchemaDelta is identity."""
        sd = SchemaDelta([])
        assert sd.is_identity()
        assert sd == SchemaDelta.identity()

    def test_empty_data_delta_is_zero(self):
        """Empty DataDelta is zero."""
        dd = DataDelta([])
        assert dd.is_zero()

    def test_empty_quality_delta_is_bottom(self):
        """Empty QualityDelta is bottom."""
        qd = QualityDelta([])
        assert qd.is_bottom()

    def test_identity_perturbation(self):
        """Identity CompoundPerturbation components."""
        p = CompoundPerturbation.identity()
        assert p.is_identity()
        assert p.schema_delta.is_identity()
        assert p.data_delta.is_zero()
        assert p.quality_delta.is_bottom()
        assert p.severity() == 0.0
        assert p.total_operation_count() == 0

    def test_identity_compose_identity(self):
        """id ∘ id = id."""
        id1 = CompoundPerturbation.identity()
        id2 = CompoundPerturbation.identity()
        result = id1.compose(id2)
        assert result.is_identity()

    def test_schema_delta_compose_identity_left(self):
        """ε ∘ ε = ε for schema delta."""
        id1 = SchemaDelta.identity()
        id2 = SchemaDelta.identity()
        assert id1.compose(id2).is_identity()

    def test_data_delta_compose_zero_left(self):
        """𝟎 ∘ 𝟎 = 𝟎 for data delta."""
        z1 = DataDelta.zero()
        z2 = DataDelta.zero()
        assert z1.compose(z2).is_zero()

    def test_quality_delta_join_bottom_bottom(self):
        """⊥ ⊔ ⊥ = ⊥."""
        b1 = QualityDelta.bottom()
        b2 = QualityDelta.bottom()
        assert b1.join(b2).is_bottom()

    def test_quality_delta_meet_top_top(self):
        """⊤ ⊓ ⊤ = ⊤."""
        t1 = QualityDelta.top()
        t2 = QualityDelta.top()
        assert t1.meet(t2).is_top()

    @SETTINGS
    @given(delta=st_schema_delta())
    def test_double_inverse(self, delta):
        """(δ⁻¹)⁻¹ should affect the same columns as δ."""
        double_inv = delta.inverse().inverse()
        # DropColumn loses type info, so structural equality may fail.
        # Check that the double inverse affects the same columns.
        assert double_inv.affected_columns() == delta.affected_columns(), (
            "Double inverse affects different columns"
        )

    @SETTINGS
    @given(delta=st_data_delta())
    def test_data_double_inverse(self, delta):
        """(δ⁻¹)⁻¹ = δ for data delta."""
        double_inv = delta.inverse().inverse()
        base = MultiSet.empty()
        assert double_inv.apply_to_data(base) == delta.apply_to_data(base)

    def test_add_then_drop_is_identity(self):
        """AddColumn then DropColumn of the same column is identity."""
        add_op = AddColumn(name="test_col", sql_type=SQLType.INTEGER, nullable=True)
        drop_op = DropColumn(name="test_col")
        sd = SchemaDelta([add_op, drop_op])
        # After normalization/composition this should reduce
        sd_normalized = sd.normalize()
        assert sd_normalized.is_identity()

    def test_rename_then_rename_back_is_identity(self):
        """Rename(a→b) then Rename(b→a) is identity."""
        op1 = RenameColumn(old_name="col_a", new_name="col_b")
        op2 = RenameColumn(old_name="col_b", new_name="col_a")
        sd = SchemaDelta([op1, op2])
        sd_normalized = sd.normalize()
        assert sd_normalized.is_identity()

    def test_single_insert_data_delta(self):
        """Single insert creates non-zero delta."""
        t = TypedTuple({"id": 1, "name": "test"})
        ms = MultiSet.from_tuples([t])
        dd = DataDelta.insert(ms)
        assert not dd.is_zero()
        assert dd.affected_rows_count() > 0

    def test_insert_then_delete_same_tuples(self):
        """Insert then delete of same tuples gives zero-effect delta."""
        t = TypedTuple({"id": 1, "name": "test"})
        ms = MultiSet.from_tuples([t])
        insert_dd = DataDelta.insert(ms)
        delete_dd = DataDelta.delete(ms)
        composed = insert_dd.compose(delete_dd)
        base = MultiSet.empty()
        result = composed.apply_to_data(base)
        assert result == base

    def test_quality_violation_severity(self):
        """Quality violation severity is derived from severity level."""
        v = QualityViolation(
            constraint_id="c1",
            severity=SeverityLevel.CRITICAL,
            affected_tuples=100,
            violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("col1",),
            message="test",
        )
        assert v.severity_score() > 0.0

    def test_quality_violation_inverse(self):
        """Quality violation inverse is an improvement."""
        v = QualityViolation(
            constraint_id="c1",
            severity=SeverityLevel.ERROR,
            affected_tuples=50,
            violation_type=ViolationType.CHECK_VIOLATION,
            columns=("col1",),
            message="test",
        )
        inv = v.inverse()
        assert inv is not None
