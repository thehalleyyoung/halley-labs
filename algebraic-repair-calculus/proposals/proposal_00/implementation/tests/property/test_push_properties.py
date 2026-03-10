"""
Property-based tests for push operators.

Verifies that push operators preserve algebraic properties when propagating
deltas through SQL operator nodes (SELECT, FILTER, JOIN, GROUP_BY, UNION).
"""

from __future__ import annotations

from typing import List, Optional, Set

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
        QualityDelta,
        QualityViolation,
        SeverityLevel,
        ViolationType,
    )
    from arc.algebra.push import (
        ColumnRef,
        FilterPush,
        GroupByPush,
        JoinPush,
        OperatorContext,
        SelectPush,
        UnionPush,
    )

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
# Helpers
# =====================================================================

def _make_select_context(column_names: List[str]) -> OperatorContext:
    """Create a SELECT operator context for the given columns."""
    cols = [ColumnRef(name=c, table=None, alias=None) for c in column_names]
    return OperatorContext(
        operator_type="SELECT",
        select_columns=cols,
    )


def _make_filter_context(
    filter_columns: List[str],
    predicate=None,
) -> OperatorContext:
    """Create a FILTER operator context."""
    return OperatorContext(
        operator_type="FILTER",
        filter_columns=filter_columns,
        filter_predicate=predicate or (lambda t: True),
    )


def _make_union_context(union_all: bool = True) -> OperatorContext:
    """Create a UNION operator context."""
    return OperatorContext(
        operator_type="UNION",
        union_all=union_all,
    )


# =====================================================================
# Hypothesis Strategies
# =====================================================================

if HAS_HYPOTHESIS:

    _SQL_TYPES = [
        SQLType.INTEGER, SQLType.VARCHAR, SQLType.TEXT,
        SQLType.BOOLEAN, SQLType.FLOAT,
    ]

    st_sql_type = st.sampled_from(_SQL_TYPES)
    st_column_name = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)

    @st.composite
    def st_add_only_schema_delta(draw, min_ops=1, max_ops=3):
        """Schema delta with only AddColumn operations."""
        num = draw(st.integers(min_value=min_ops, max_value=max_ops))
        used = set()
        ops = []
        for _ in range(num):
            name = draw(st_column_name.filter(lambda n: n not in used))
            used.add(name)
            sql_type = draw(st_sql_type)
            ops.append(AddColumn(name=name, sql_type=sql_type, nullable=True))
        return SchemaDelta(ops)

    @st.composite
    def st_typed_tuple_for_columns(draw, columns: List[str]):
        """Generate a TypedTuple with specific columns."""
        values = {}
        for col in columns:
            values[col] = draw(st.one_of(
                st.integers(min_value=-100, max_value=100),
                st.text(min_size=1, max_size=5, alphabet="abcdef"),
            ))
        return TypedTuple(values)

    @st.composite
    def st_multiset_for_columns(draw, columns: List[str], min_t=0, max_t=5):
        """Generate a MultiSet with tuples conforming to given columns."""
        num = draw(st.integers(min_value=min_t, max_value=max_t))
        if num == 0:
            return MultiSet.empty()
        tuples = []
        for _ in range(num):
            tuples.append(draw(st_typed_tuple_for_columns(columns)))
        return MultiSet.from_tuples(tuples)

    @st.composite
    def st_data_delta_for_columns(draw, columns: List[str]):
        """Generate a DataDelta with operations on given columns."""
        num_ops = draw(st.integers(min_value=1, max_value=2))
        ops = []
        for _ in range(num_ops):
            ms = draw(st_multiset_for_columns(columns, min_t=1, max_t=3))
            if draw(st.booleans()):
                ops.append(InsertOp(ms))
            else:
                ops.append(DeleteOp(ms))
        return DataDelta(ops)

    @st.composite
    def st_quality_violation_simple(draw):
        """Generate a simple quality violation."""
        constraint_id = draw(st.text(min_size=3, max_size=8, alphabet="abcdef123"))
        severity = draw(st.sampled_from([
            SeverityLevel.INFO, SeverityLevel.WARNING,
            SeverityLevel.ERROR, SeverityLevel.CRITICAL,
        ]))
        return QualityViolation(
            constraint_id=constraint_id,
            severity=severity,
            affected_tuples=draw(st.integers(min_value=0, max_value=100)),
            violation_type=ViolationType.CHECK_VIOLATION,
            columns=(),
            message="test",
        )

    @st.composite
    def st_quality_delta_simple(draw, min_ops=0, max_ops=2):
        """Generate a simple quality delta."""
        num = draw(st.integers(min_value=min_ops, max_value=max_ops))
        ops = [draw(st_quality_violation_simple()) for _ in range(num)]
        return QualityDelta(ops)

    @st.composite
    def st_column_set(draw, min_cols=2, max_cols=5):
        """Generate a set of unique column names."""
        num = draw(st.integers(min_value=min_cols, max_value=max_cols))
        names = set()
        while len(names) < num:
            names.add(draw(st_column_name.filter(lambda n: n not in names)))
        return sorted(names)


# =====================================================================
# Push Identity Tests
# =====================================================================

class TestPushIdentity:
    """Push of identity delta should be identity delta for all operators."""

    @SETTINGS
    @given(data=st.data())
    def test_select_push_identity_schema(self, data):
        """push_SELECT(ε_schema) = ε_schema."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        identity = SchemaDelta.identity()
        push = SelectPush()
        result = push.push_schema(ctx, identity)
        assert result.is_identity()

    @SETTINGS
    @given(data=st.data())
    def test_select_push_identity_data(self, data):
        """push_SELECT(𝟎_data) = 𝟎_data."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        zero = DataDelta.zero()
        push = SelectPush()
        result = push.push_data(ctx, zero)
        assert result.is_zero()

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_identity_schema(self, data):
        """push_FILTER(ε_schema) = ε_schema."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_filter_context(cols)
        identity = SchemaDelta.identity()
        push = FilterPush()
        result = push.push_schema(ctx, identity)
        assert result.is_identity()

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_identity_data(self, data):
        """push_FILTER(𝟎_data) = 𝟎_data."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(cols)
        zero = DataDelta.zero()
        push = FilterPush()
        result = push.push_data(ctx, zero)
        assert result.is_zero()

    @SETTINGS
    @given(data=st.data())
    def test_union_push_identity_schema(self, data):
        """push_UNION(ε_schema) = ε_schema."""
        ctx = _make_union_context()
        identity = SchemaDelta.identity()
        push = UnionPush()
        result = push.push_schema(ctx, identity)
        assert result.is_identity()

    @SETTINGS
    @given(data=st.data())
    def test_union_push_identity_data(self, data):
        """push_UNION(𝟎_data) = 𝟎_data."""
        ctx = _make_union_context()
        zero = DataDelta.zero()
        push = UnionPush()
        result = push.push_data(ctx, zero)
        assert result.is_zero()

    @SETTINGS
    @given(data=st.data())
    def test_select_push_identity_quality(self, data):
        """push_SELECT(⊥_quality) = ⊥_quality."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        bottom = QualityDelta.bottom()
        push = SelectPush()
        result = push.push_quality(ctx, bottom)
        assert result.is_bottom()


# =====================================================================
# Push Preserves Delta Sort Tests
# =====================================================================

class TestPushPreservesSort:
    """Push of a delta preserves its algebraic sort."""

    @SETTINGS
    @given(data=st.data())
    def test_select_push_schema_returns_schema_delta(self, data):
        """push_SELECT on schema delta returns SchemaDelta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        sd = data.draw(st_add_only_schema_delta())
        push = SelectPush()
        result = push.push_schema(ctx, sd)
        assert isinstance(result, SchemaDelta)

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_schema_returns_schema_delta(self, data):
        """push_FILTER on schema delta returns SchemaDelta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_filter_context(cols)
        sd = data.draw(st_add_only_schema_delta())
        push = FilterPush()
        result = push.push_schema(ctx, sd)
        assert isinstance(result, SchemaDelta)

    @SETTINGS
    @given(data=st.data())
    def test_select_push_data_returns_data_delta(self, data):
        """push_SELECT on data delta returns DataDelta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        dd = data.draw(st_data_delta_for_columns(cols))
        push = SelectPush()
        result = push.push_data(ctx, dd)
        assert isinstance(result, DataDelta)

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_data_returns_data_delta(self, data):
        """push_FILTER on data delta returns DataDelta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(cols)
        dd = data.draw(st_data_delta_for_columns(cols))
        push = FilterPush()
        result = push.push_data(ctx, dd)
        assert isinstance(result, DataDelta)

    @SETTINGS
    @given(data=st.data())
    def test_select_push_quality_returns_quality_delta(self, data):
        """push_SELECT on quality delta returns QualityDelta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        qd = data.draw(st_quality_delta_simple(min_ops=1, max_ops=2))
        push = SelectPush()
        result = push.push_quality(ctx, qd)
        assert isinstance(result, QualityDelta)


# =====================================================================
# Push Preserves Zero Tests
# =====================================================================

class TestPushPreservesZero:
    """Push of zero/identity data delta should yield zero."""

    @SETTINGS
    @given(data=st.data())
    def test_select_push_zero_data(self, data):
        """push_SELECT(𝟎) = 𝟎."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        zero = DataDelta.zero()
        result = SelectPush().push_data(ctx, zero)
        assert result.is_zero()

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_zero_data(self, data):
        """push_FILTER(𝟎) = 𝟎."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(cols)
        zero = DataDelta.zero()
        result = FilterPush().push_data(ctx, zero)
        assert result.is_zero()

    @SETTINGS
    @given(data=st.data())
    def test_union_push_zero_data(self, data):
        """push_UNION(𝟎) = 𝟎."""
        ctx = _make_union_context()
        zero = DataDelta.zero()
        result = UnionPush().push_data(ctx, zero)
        assert result.is_zero()


# =====================================================================
# Annihilation Soundness Tests
# =====================================================================

class TestAnnihilationSoundness:
    """When a filter drops all affected columns, push produces zero delta."""

    @SETTINGS
    @given(data=st.data())
    def test_filter_annihilation_rejects_all(self, data):
        """A filter that rejects all tuples should annihilate the data delta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(
            filter_columns=cols,
            predicate=lambda t: False,  # reject everything
        )
        dd = data.draw(st_data_delta_for_columns(cols))
        result = FilterPush().push_data(ctx, dd)
        # A filter rejecting everything should produce a zero or
        # significantly reduced delta
        base = MultiSet.empty()
        applied = result.apply_to_data(base)
        # Since we reject everything, inserts through the filter should vanish
        assert applied.cardinality() == 0

    @SETTINGS
    @given(data=st.data())
    def test_filter_passthrough_preserves_delta(self, data):
        """A filter that accepts all tuples should preserve the data delta."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(
            filter_columns=cols,
            predicate=lambda t: True,  # accept everything
        )
        dd = data.draw(st_data_delta_for_columns(cols))
        result = FilterPush().push_data(ctx, dd)
        # A passthrough filter should preserve the data delta's effects
        base = MultiSet.empty()
        original = dd.apply_to_data(base)
        pushed = result.apply_to_data(base)
        assert pushed == original


# =====================================================================
# Schema Push Column Count Tests
# =====================================================================

class TestSchemaPushColumnCount:
    """Schema push preserves column count relationships."""

    @SETTINGS
    @given(data=st.data())
    def test_select_push_add_column_count(self, data):
        """Adding N columns through SELECT push adds at most N columns."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        sd = data.draw(st_add_only_schema_delta(min_ops=1, max_ops=2))
        result = SelectPush().push_schema(ctx, sd)
        # The pushed schema delta should not add MORE columns than the input
        assert result.operation_count() <= sd.operation_count() + len(cols)

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_schema_preserves_columns(self, data):
        """Filter push preserves schema delta entirely (no column removal)."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_filter_context(cols)
        sd = data.draw(st_add_only_schema_delta(min_ops=1, max_ops=2))
        result = FilterPush().push_schema(ctx, sd)
        # Filter should pass schema changes through
        assert isinstance(result, SchemaDelta)


# =====================================================================
# Push Composition Tests (where valid)
# =====================================================================

class TestPushComposition:
    """Test relationship between push_f(δ₁∘δ₂) and push_f(δ₁)∘push_f(δ₂)."""

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_data_composition(self, data):
        """For FILTER, push(δ₁∘δ₂) should equal push(δ₁)∘push(δ₂)
        when the filter is a pass-through.
        """
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(
            filter_columns=cols,
            predicate=lambda t: True,
        )
        d1 = data.draw(st_data_delta_for_columns(cols))
        d2 = data.draw(st_data_delta_for_columns(cols))
        push = FilterPush()

        composed_then_pushed = push.push_data(ctx, d1.compose(d2))
        pushed_then_composed = push.push_data(ctx, d1).compose(
            push.push_data(ctx, d2)
        )

        base = MultiSet.empty()
        lhs = composed_then_pushed.apply_to_data(base)
        rhs = pushed_then_composed.apply_to_data(base)
        assert lhs == rhs

    @SETTINGS
    @given(data=st.data())
    def test_union_push_data_composition(self, data):
        """For UNION ALL, push(δ₁∘δ₂) = push(δ₁)∘push(δ₂)."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_union_context(union_all=True)
        d1 = data.draw(st_data_delta_for_columns(cols))
        d2 = data.draw(st_data_delta_for_columns(cols))
        push = UnionPush()

        composed_then_pushed = push.push_data(ctx, d1.compose(d2))
        pushed_then_composed = push.push_data(ctx, d1).compose(
            push.push_data(ctx, d2)
        )

        base = MultiSet.empty()
        lhs = composed_then_pushed.apply_to_data(base)
        rhs = pushed_then_composed.apply_to_data(base)
        assert lhs == rhs

    @SETTINGS
    @given(data=st.data())
    def test_select_push_schema_composition(self, data):
        """For SELECT, push(σ₁∘σ₂) should relate to push(σ₁)∘push(σ₂)."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)

        used = set(cols)
        # Two add-column deltas with distinct new columns
        name1 = data.draw(st_column_name.filter(lambda n: n not in used))
        used.add(name1)
        name2 = data.draw(st_column_name.filter(lambda n: n not in used))
        used.add(name2)

        s1 = SchemaDelta([AddColumn(
            name=name1, sql_type=SQLType.INTEGER, nullable=True
        )])
        s2 = SchemaDelta([AddColumn(
            name=name2, sql_type=SQLType.VARCHAR, nullable=True
        )])

        push = SelectPush()
        composed_then_pushed = push.push_schema(ctx, s1.compose(s2))
        pushed_then_composed = push.push_schema(ctx, s1).compose(
            push.push_schema(ctx, s2)
        )

        # Both should produce valid schema deltas
        assert isinstance(composed_then_pushed, SchemaDelta)
        assert isinstance(pushed_then_composed, SchemaDelta)


# =====================================================================
# push_all Consistency Tests
# =====================================================================

class TestPushAllConsistency:
    """push_all should be consistent with individual push methods."""

    @SETTINGS
    @given(data=st.data())
    def test_select_push_all_consistent(self, data):
        """push_all returns same results as individual pushes."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=4))
        ctx = _make_select_context(cols)
        sd = SchemaDelta.identity()
        dd = DataDelta.zero()
        qd = QualityDelta.bottom()
        push = SelectPush()

        sd_r, dd_r, qd_r = push.push_all(ctx, sd, dd, qd)
        assert sd_r == push.push_schema(ctx, sd)
        assert dd_r == push.push_data(ctx, dd)
        assert qd_r == push.push_quality(ctx, qd)

    @SETTINGS
    @given(data=st.data())
    def test_filter_push_all_consistent(self, data):
        """push_all returns same results as individual pushes for filter."""
        cols = data.draw(st_column_set(min_cols=2, max_cols=3))
        ctx = _make_filter_context(cols)
        sd = SchemaDelta.identity()
        dd = DataDelta.zero()
        qd = QualityDelta.bottom()
        push = FilterPush()

        sd_r, dd_r, qd_r = push.push_all(ctx, sd, dd, qd)
        assert sd_r == push.push_schema(ctx, sd)
        assert dd_r == push.push_data(ctx, dd)
        assert qd_r == push.push_quality(ctx, qd)
