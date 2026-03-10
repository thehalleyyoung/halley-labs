"""
Tests for arc.algebra.composition — Compound Perturbation
==========================================================

Covers CompoundPerturbation (identity, factory methods, composition, inverse,
apply, serialization), compose_chain, compose_parallel, diff_states,
and the algebraic verification helpers.
"""

from __future__ import annotations

import unittest
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    from arc.algebra.composition import (
        CompoundPerturbation,
        PipelineState,
        compose_chain,
        compose_parallel,
        diff_states,
        verify_composition_associativity,
        verify_identity,
        verify_inverse,
    )

    _COMP_AVAILABLE = True
except ImportError:
    _COMP_AVAILABLE = False

try:
    from arc.algebra.schema_delta import (
        AddColumn,
        ColumnDef,
        DropColumn,
        RenameColumn,
        Schema,
        SchemaDelta,
        SQLType,
    )

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from arc.algebra.data_delta import (
        DataDelta,
        InsertOp,
        DeleteOp,
        MultiSet,
        TypedTuple,
    )

    _DATA_AVAILABLE = True
except ImportError:
    _DATA_AVAILABLE = False

try:
    from arc.algebra.quality_delta import (
        ConstraintStatus,
        ConstraintType,
        QualityDelta,
        QualityState,
        QualityViolation,
        SeverityLevel,
        ViolationType,
    )

    _QUALITY_AVAILABLE = True
except ImportError:
    _QUALITY_AVAILABLE = False

_ALL_AVAILABLE = (
    _COMP_AVAILABLE and _SCHEMA_AVAILABLE and _DATA_AVAILABLE and _QUALITY_AVAILABLE
)

SKIP_REASON = "Required arc.algebra modules not importable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema(*col_defs: tuple) -> "Schema":
    """Build a Schema from (name, SQLType) pairs."""
    cols = OrderedDict()
    for i, (name, sql_type) in enumerate(col_defs):
        cols[name] = ColumnDef(name=name, sql_type=sql_type, position=i)
    return Schema(name="test", columns=cols)


def _make_multiset(dicts: list) -> "MultiSet":
    return MultiSet.from_dicts(dicts)


def _tt(d: dict) -> "TypedTuple":
    return TypedTuple.from_dict(d)


def _empty_quality_state() -> "QualityState":
    return QualityState()


def _base_state() -> "PipelineState":
    """A small reference PipelineState."""
    schema = _make_schema(
        ("id", SQLType.INTEGER),
        ("name", SQLType.VARCHAR),
        ("age", SQLType.INTEGER),
    )
    data = _make_multiset([
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ])
    quality = _empty_quality_state()
    return PipelineState(schema=schema, data=data, quality=quality)


def _add_col_perturbation() -> "CompoundPerturbation":
    sd = SchemaDelta.from_operation(
        AddColumn(name="email", sql_type=SQLType.VARCHAR)
    )
    return CompoundPerturbation.schema_only(sd)


def _insert_perturbation() -> "CompoundPerturbation":
    dd = DataDelta.insert(_make_multiset([{"id": 3, "name": "Carol", "age": 35}]))
    return CompoundPerturbation.data_only(dd)


def _violation_perturbation() -> "CompoundPerturbation":
    qd = QualityDelta.violation(
        constraint_id="nn_age",
        severity=SeverityLevel.WARNING,
        affected_tuples=5,
        violation_type=ViolationType.NULL_IN_NON_NULL,
        columns=("age",),
    )
    return CompoundPerturbation.quality_only(qd)


def _mixed_perturbation() -> "CompoundPerturbation":
    sd = SchemaDelta.from_operation(
        AddColumn(name="email", sql_type=SQLType.VARCHAR)
    )
    dd = DataDelta.insert(_make_multiset([{"id": 3, "name": "Carol", "age": 35}]))
    qd = QualityDelta.violation(
        constraint_id="nn_email",
        severity=SeverityLevel.WARNING,
        affected_tuples=1,
        violation_type=ViolationType.NULL_IN_NON_NULL,
        columns=("email",),
    )
    return CompoundPerturbation(sd, dd, qd)


# ====================================================================
# 1. CompoundPerturbation.identity()
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestCompoundPerturbationIdentity(unittest.TestCase):
    """CompoundPerturbation.identity() — all three deltas are identities."""

    def test_identity_is_identity(self):
        p = CompoundPerturbation.identity()
        self.assertTrue(p.is_identity())

    def test_identity_schema_delta(self):
        p = CompoundPerturbation.identity()
        self.assertTrue(p.schema_delta.is_identity())

    def test_identity_data_delta(self):
        p = CompoundPerturbation.identity()
        self.assertTrue(p.data_delta.is_zero())

    def test_identity_quality_delta(self):
        p = CompoundPerturbation.identity()
        self.assertTrue(p.quality_delta.is_bottom())

    def test_identity_bool_is_false(self):
        p = CompoundPerturbation.identity()
        self.assertFalse(bool(p))

    def test_identity_operation_counts(self):
        p = CompoundPerturbation.identity()
        self.assertEqual(p.total_operation_count(), 0)
        self.assertEqual(p.schema_operation_count(), 0)
        self.assertEqual(p.data_operation_count(), 0)
        self.assertEqual(p.quality_operation_count(), 0)

    def test_identity_has_no_changes(self):
        p = CompoundPerturbation.identity()
        self.assertFalse(p.has_schema_changes())
        self.assertFalse(p.has_data_changes())
        self.assertFalse(p.has_quality_changes())

    def test_identity_severity_is_zero(self):
        p = CompoundPerturbation.identity()
        self.assertAlmostEqual(p.severity(), 0.0)

    def test_identity_affected_columns_empty(self):
        p = CompoundPerturbation.identity()
        self.assertEqual(p.affected_columns(), set())

    def test_identity_summary(self):
        p = CompoundPerturbation.identity()
        s = p.summary()
        self.assertTrue(s["is_identity"])
        self.assertAlmostEqual(s["severity"], 0.0)


# ====================================================================
# 2. CompoundPerturbation.schema_only()
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestSchemaOnly(unittest.TestCase):
    """CompoundPerturbation.schema_only() — only schema delta non-trivial."""

    def test_schema_only_has_schema(self):
        p = _add_col_perturbation()
        self.assertTrue(p.has_schema_changes())
        self.assertFalse(p.has_data_changes())
        self.assertFalse(p.has_quality_changes())

    def test_schema_only_not_identity(self):
        p = _add_col_perturbation()
        self.assertFalse(p.is_identity())
        self.assertTrue(bool(p))

    def test_schema_only_affected_columns(self):
        p = _add_col_perturbation()
        self.assertIn("email", p.affected_columns())

    def test_schema_only_operation_count(self):
        p = _add_col_perturbation()
        self.assertEqual(p.schema_operation_count(), 1)
        self.assertEqual(p.data_operation_count(), 0)
        self.assertEqual(p.quality_operation_count(), 0)
        self.assertEqual(p.total_operation_count(), 1)


# ====================================================================
# 3. CompoundPerturbation.data_only()
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestDataOnly(unittest.TestCase):
    """CompoundPerturbation.data_only() — only data delta non-trivial."""

    def test_data_only_has_data(self):
        p = _insert_perturbation()
        self.assertFalse(p.has_schema_changes())
        self.assertTrue(p.has_data_changes())
        self.assertFalse(p.has_quality_changes())

    def test_data_only_not_identity(self):
        p = _insert_perturbation()
        self.assertFalse(p.is_identity())

    def test_data_only_operation_count(self):
        p = _insert_perturbation()
        self.assertEqual(p.schema_operation_count(), 0)
        self.assertGreater(p.data_operation_count(), 0)


# ====================================================================
# 4. CompoundPerturbation.quality_only()
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestQualityOnly(unittest.TestCase):
    """CompoundPerturbation.quality_only() — only quality delta non-trivial."""

    def test_quality_only_has_quality(self):
        p = _violation_perturbation()
        self.assertFalse(p.has_schema_changes())
        self.assertFalse(p.has_data_changes())
        self.assertTrue(p.has_quality_changes())

    def test_quality_only_not_identity(self):
        p = _violation_perturbation()
        self.assertFalse(p.is_identity())

    def test_quality_only_operation_count(self):
        p = _violation_perturbation()
        self.assertEqual(p.quality_operation_count(), 1)


# ====================================================================
# 5. Composition: (p1 ∘ p2)
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestComposition(unittest.TestCase):
    """Composition composes all three sorts correctly."""

    def test_compose_two_schema_perturbations(self):
        sd1 = SchemaDelta.from_operation(AddColumn(name="a", sql_type=SQLType.INTEGER))
        sd2 = SchemaDelta.from_operation(AddColumn(name="b", sql_type=SQLType.VARCHAR))
        p1 = CompoundPerturbation.schema_only(sd1)
        p2 = CompoundPerturbation.schema_only(sd2)
        result = p1.compose(p2)
        self.assertTrue(result.has_schema_changes())
        self.assertGreaterEqual(result.schema_operation_count(), 2)

    def test_compose_schema_and_data(self):
        p1 = _add_col_perturbation()
        p2 = _insert_perturbation()
        result = p1.compose(p2)
        self.assertTrue(result.has_schema_changes())
        # data may be transformed by schema interaction (φ)
        self.assertIsInstance(result, CompoundPerturbation)

    def test_compose_all_three(self):
        p1 = _add_col_perturbation()
        p2 = _mixed_perturbation()
        result = p1.compose(p2)
        self.assertTrue(result.has_schema_changes())
        self.assertIsInstance(result, CompoundPerturbation)

    def test_compose_returns_compound(self):
        p1 = _insert_perturbation()
        p2 = _violation_perturbation()
        result = p1.compose(p2)
        self.assertIsInstance(result, CompoundPerturbation)

    def test_compose_identity_left(self):
        p = _mixed_perturbation()
        result = CompoundPerturbation.identity().compose(p)
        self.assertEqual(result, p)

    def test_compose_identity_right(self):
        p = _mixed_perturbation()
        result = p.compose(CompoundPerturbation.identity())
        self.assertEqual(result, p)


# ====================================================================
# 6. Associativity (T6)
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestAssociativity(unittest.TestCase):
    """verify_composition_associativity returns True."""

    def test_associativity_schema_perturbations(self):
        p1 = CompoundPerturbation.schema_only(
            SchemaDelta.from_operation(AddColumn(name="a", sql_type=SQLType.INTEGER))
        )
        p2 = CompoundPerturbation.schema_only(
            SchemaDelta.from_operation(AddColumn(name="b", sql_type=SQLType.VARCHAR))
        )
        p3 = CompoundPerturbation.schema_only(
            SchemaDelta.from_operation(AddColumn(name="c", sql_type=SQLType.FLOAT))
        )
        self.assertTrue(verify_composition_associativity(p1, p2, p3))

    def test_associativity_data_perturbations(self):
        p1 = CompoundPerturbation.data_only(
            DataDelta.insert(_make_multiset([{"id": 1}]))
        )
        p2 = CompoundPerturbation.data_only(
            DataDelta.insert(_make_multiset([{"id": 2}]))
        )
        p3 = CompoundPerturbation.data_only(
            DataDelta.insert(_make_multiset([{"id": 3}]))
        )
        self.assertTrue(verify_composition_associativity(p1, p2, p3))

    def test_associativity_identities(self):
        e = CompoundPerturbation.identity()
        self.assertTrue(verify_composition_associativity(e, e, e))

    def test_associativity_mixed(self):
        p1 = _add_col_perturbation()
        p2 = _insert_perturbation()
        p3 = _violation_perturbation()
        self.assertTrue(verify_composition_associativity(p1, p2, p3))


# ====================================================================
# 7. Identity element
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestIdentityElement(unittest.TestCase):
    """Compose with identity returns same perturbation."""

    def test_verify_identity_schema(self):
        p = _add_col_perturbation()
        self.assertTrue(verify_identity(p))

    def test_verify_identity_data(self):
        p = _insert_perturbation()
        self.assertTrue(verify_identity(p))

    def test_verify_identity_quality(self):
        p = _violation_perturbation()
        self.assertTrue(verify_identity(p))

    def test_verify_identity_mixed(self):
        p = _mixed_perturbation()
        self.assertTrue(verify_identity(p))

    def test_verify_identity_identity(self):
        self.assertTrue(verify_identity(CompoundPerturbation.identity()))


# ====================================================================
# 8. Inverse: p ∘ p⁻¹ ≈ identity
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestInverse(unittest.TestCase):
    """Inverse: p ∘ p⁻¹ ≈ identity (exact for schema/data, approximate for quality)."""

    def test_inverse_schema_only(self):
        p = _add_col_perturbation()
        state = _base_state()
        self.assertTrue(verify_inverse(p, state))

    def test_inverse_data_only(self):
        p = _insert_perturbation()
        state = _base_state()
        self.assertTrue(verify_inverse(p, state))

    def test_inverse_identity(self):
        p = CompoundPerturbation.identity()
        state = _base_state()
        self.assertTrue(verify_inverse(p, state))

    def test_inverse_produces_compound(self):
        p = _mixed_perturbation()
        inv = p.inverse()
        self.assertIsInstance(inv, CompoundPerturbation)

    def test_inverse_of_identity_is_identity(self):
        p = CompoundPerturbation.identity()
        inv = p.inverse()
        self.assertTrue(inv.is_identity())


# ====================================================================
# 9. apply()
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestApply(unittest.TestCase):
    """apply() applies all three deltas to PipelineState."""

    def test_apply_schema_only(self):
        state = _base_state()
        p = _add_col_perturbation()
        result = p.apply(state)
        self.assertIn("email", result.column_names())
        # Original is unchanged
        self.assertNotIn("email", state.column_names())

    def test_apply_data_only(self):
        state = _base_state()
        p = _insert_perturbation()
        result = p.apply(state)
        self.assertEqual(result.row_count(), state.row_count() + 1)

    def test_apply_quality_only(self):
        state = _base_state()
        p = _violation_perturbation()
        result = p.apply(state)
        self.assertTrue(result.has_violations())
        self.assertFalse(state.has_violations())

    def test_apply_identity(self):
        state = _base_state()
        p = CompoundPerturbation.identity()
        result = p.apply(state)
        self.assertEqual(result.schema, state.schema)
        self.assertEqual(result.data, state.data)

    def test_apply_preserves_metadata(self):
        state = _base_state()
        state.metadata["key"] = "value"
        p = CompoundPerturbation.identity()
        result = p.apply(state)
        self.assertEqual(result.metadata["key"], "value")

    def test_apply_mixed(self):
        state = _base_state()
        p = _mixed_perturbation()
        result = p.apply(state)
        self.assertIn("email", result.column_names())
        self.assertGreater(result.row_count(), state.row_count())
        self.assertTrue(result.has_violations())


# ====================================================================
# 10. compose_chain
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestComposeChain(unittest.TestCase):
    """compose_chain: sequential composition of multiple perturbations."""

    def test_empty_chain(self):
        result = compose_chain([])
        self.assertTrue(result.is_identity())

    def test_single_element_chain(self):
        p = _add_col_perturbation()
        result = compose_chain([p])
        self.assertEqual(result, p)

    def test_two_element_chain(self):
        p1 = _add_col_perturbation()
        p2 = _insert_perturbation()
        result = compose_chain([p1, p2])
        self.assertIsInstance(result, CompoundPerturbation)

    def test_chain_with_identities(self):
        e = CompoundPerturbation.identity()
        p = _add_col_perturbation()
        result = compose_chain([e, p, e])
        self.assertEqual(result, p)

    def test_long_chain(self):
        perturbations = [
            CompoundPerturbation.schema_only(
                SchemaDelta.from_operation(
                    AddColumn(name=f"col_{i}", sql_type=SQLType.INTEGER)
                )
            )
            for i in range(5)
        ]
        result = compose_chain(perturbations)
        self.assertGreaterEqual(result.schema_operation_count(), 5)


# ====================================================================
# 11. compose_parallel
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestComposeParallel(unittest.TestCase):
    """compose_parallel: parallel composition of non-conflicting perturbations."""

    def test_empty_parallel(self):
        result = compose_parallel([])
        self.assertTrue(result.is_identity())

    def test_single_element_parallel(self):
        p = _add_col_perturbation()
        result = compose_parallel([p])
        self.assertEqual(result, p)

    def test_two_independent_schema(self):
        p1 = CompoundPerturbation.schema_only(
            SchemaDelta.from_operation(AddColumn(name="a", sql_type=SQLType.INTEGER))
        )
        p2 = CompoundPerturbation.schema_only(
            SchemaDelta.from_operation(AddColumn(name="b", sql_type=SQLType.VARCHAR))
        )
        result = compose_parallel([p1, p2])
        self.assertGreaterEqual(result.schema_operation_count(), 2)

    def test_parallel_data_perturbations(self):
        p1 = CompoundPerturbation.data_only(
            DataDelta.insert(_make_multiset([{"id": 10}]))
        )
        p2 = CompoundPerturbation.data_only(
            DataDelta.insert(_make_multiset([{"id": 20}]))
        )
        result = compose_parallel([p1, p2])
        self.assertTrue(result.has_data_changes())


# ====================================================================
# 12. diff_states
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestDiffStates(unittest.TestCase):
    """diff_states: computes delta between two states."""

    def test_diff_identical_states(self):
        s1 = _base_state()
        s2 = _base_state()
        delta = diff_states(s1, s2)
        self.assertIsInstance(delta, CompoundPerturbation)
        self.assertTrue(delta.schema_delta.is_identity())
        self.assertTrue(delta.data_delta.is_zero())

    def test_diff_with_added_column(self):
        s1 = _base_state()
        s2 = _base_state()
        s2.schema.add_column(
            ColumnDef(name="email", sql_type=SQLType.VARCHAR, position=3)
        )
        delta = diff_states(s1, s2)
        self.assertTrue(delta.has_schema_changes())

    def test_diff_with_added_row(self):
        s1 = _base_state()
        s2 = _base_state()
        s2.data.add(_tt({"id": 3, "name": "Carol", "age": 35}))
        delta = diff_states(s1, s2)
        self.assertTrue(delta.has_data_changes())

    def test_diff_with_removed_row(self):
        s1 = _base_state()
        s2 = _base_state()
        s2.data.remove(_tt({"id": 1, "name": "Alice", "age": 30}))
        delta = diff_states(s1, s2)
        self.assertTrue(delta.has_data_changes())


# ====================================================================
# 13. Interaction effects during composition
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestInteractionEffects(unittest.TestCase):
    """Schema changes affect data delta via φ during composition."""

    def test_rename_then_data(self):
        """Composing a rename perturbation with a data perturbation
        should apply the φ interaction to the data delta."""
        sd = SchemaDelta.from_operation(
            RenameColumn(old_name="name", new_name="full_name")
        )
        p_schema = CompoundPerturbation.schema_only(sd)
        p_data = _insert_perturbation()
        result = p_schema.compose(p_data)
        self.assertIsInstance(result, CompoundPerturbation)
        self.assertTrue(result.has_schema_changes())

    def test_add_column_then_data(self):
        p_schema = _add_col_perturbation()
        p_data = _insert_perturbation()
        result = p_schema.compose(p_data)
        self.assertIsInstance(result, CompoundPerturbation)

    def test_drop_column_then_data(self):
        sd = SchemaDelta.from_operation(DropColumn(name="age"))
        p_schema = CompoundPerturbation.schema_only(sd)
        p_data = _insert_perturbation()
        result = p_schema.compose(p_data)
        self.assertIsInstance(result, CompoundPerturbation)


# ====================================================================
# 14. Serialization round-trip
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestSerialization(unittest.TestCase):
    """Serialization round-trip via to_dict / from_dict."""

    def test_identity_round_trip(self):
        p = CompoundPerturbation.identity()
        d = p.to_dict()
        p2 = CompoundPerturbation.from_dict(d)
        self.assertTrue(p2.is_identity())

    def test_schema_only_round_trip(self):
        p = _add_col_perturbation()
        d = p.to_dict()
        p2 = CompoundPerturbation.from_dict(d)
        self.assertTrue(p2.has_schema_changes())
        self.assertFalse(p2.has_data_changes())

    def test_data_only_round_trip(self):
        p = _insert_perturbation()
        d = p.to_dict()
        p2 = CompoundPerturbation.from_dict(d)
        self.assertTrue(p2.has_data_changes())

    def test_quality_only_round_trip(self):
        p = _violation_perturbation()
        d = p.to_dict()
        p2 = CompoundPerturbation.from_dict(d)
        self.assertTrue(p2.has_quality_changes())

    def test_mixed_round_trip(self):
        p = _mixed_perturbation()
        d = p.to_dict()
        p2 = CompoundPerturbation.from_dict(d)
        self.assertTrue(p2.has_schema_changes())
        self.assertTrue(p2.has_data_changes())
        self.assertTrue(p2.has_quality_changes())

    def test_to_dict_keys(self):
        p = _mixed_perturbation()
        d = p.to_dict()
        self.assertIn("schema_delta", d)
        self.assertIn("data_delta", d)
        self.assertIn("quality_delta", d)

    def test_from_dict_empty(self):
        p = CompoundPerturbation.from_dict({})
        self.assertTrue(p.is_identity())


# ====================================================================
# 15. Edge cases
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty perturbation, single-sort, equality, hashing."""

    def test_empty_perturbation_is_identity(self):
        p = CompoundPerturbation()
        self.assertTrue(p.is_identity())

    def test_single_sort_schema(self):
        sd = SchemaDelta.from_operation(AddColumn(name="x", sql_type=SQLType.INTEGER))
        p = CompoundPerturbation(schema_delta=sd)
        self.assertTrue(p.has_schema_changes())
        self.assertFalse(p.has_data_changes())
        self.assertFalse(p.has_quality_changes())

    def test_single_sort_data(self):
        dd = DataDelta.insert(_make_multiset([{"id": 1}]))
        p = CompoundPerturbation(data_delta=dd)
        self.assertTrue(p.has_data_changes())
        self.assertFalse(p.has_schema_changes())

    def test_single_sort_quality(self):
        qd = QualityDelta.violation(
            constraint_id="c1",
            severity=SeverityLevel.ERROR,
            affected_tuples=1,
            violation_type=ViolationType.UNIQUENESS_VIOLATION,
            columns=("id",),
        )
        p = CompoundPerturbation(quality_delta=qd)
        self.assertTrue(p.has_quality_changes())
        self.assertFalse(p.has_schema_changes())

    def test_equality(self):
        p1 = CompoundPerturbation.identity()
        p2 = CompoundPerturbation.identity()
        self.assertEqual(p1, p2)

    def test_inequality(self):
        p1 = CompoundPerturbation.identity()
        p2 = _add_col_perturbation()
        self.assertNotEqual(p1, p2)

    def test_hash_consistency(self):
        p1 = CompoundPerturbation.identity()
        p2 = CompoundPerturbation.identity()
        self.assertEqual(hash(p1), hash(p2))

    def test_repr_identity(self):
        p = CompoundPerturbation.identity()
        self.assertIn("identity", repr(p))

    def test_repr_non_identity(self):
        p = _mixed_perturbation()
        r = repr(p)
        self.assertIn("CompoundPerturbation", r)

    def test_normalize(self):
        p = _mixed_perturbation()
        n = p.normalize()
        self.assertIsInstance(n, CompoundPerturbation)

    def test_summary_keys(self):
        p = _mixed_perturbation()
        s = p.summary()
        expected_keys = {
            "is_identity", "severity", "schema_ops", "data_ops",
            "quality_ops", "affected_columns", "has_schema",
            "has_data", "has_quality", "data_rows_affected",
            "data_net_change", "quality_severity",
        }
        self.assertTrue(expected_keys.issubset(set(s.keys())))

    def test_severity_bounded(self):
        p = _mixed_perturbation()
        self.assertGreaterEqual(p.severity(), 0.0)
        self.assertLessEqual(p.severity(), 1.0)


# ====================================================================
# 16. PipelineState
# ====================================================================


@unittest.skipUnless(_ALL_AVAILABLE, SKIP_REASON)
class TestPipelineState(unittest.TestCase):
    """PipelineState methods."""

    def test_copy_is_deep(self):
        s = _base_state()
        c = s.copy()
        c.schema.add_column(ColumnDef(name="extra", sql_type=SQLType.TEXT, position=3))
        self.assertNotIn("extra", s.column_names())

    def test_row_count(self):
        s = _base_state()
        self.assertEqual(s.row_count(), 2)

    def test_column_names(self):
        s = _base_state()
        self.assertEqual(s.column_names(), ["id", "name", "age"])

    def test_quality_score_default(self):
        s = _base_state()
        self.assertAlmostEqual(s.quality_score(), 1.0)

    def test_has_violations_default(self):
        s = _base_state()
        self.assertFalse(s.has_violations())

    def test_repr(self):
        s = _base_state()
        r = repr(s)
        self.assertIn("PipelineState", r)


if __name__ == "__main__":
    unittest.main()
