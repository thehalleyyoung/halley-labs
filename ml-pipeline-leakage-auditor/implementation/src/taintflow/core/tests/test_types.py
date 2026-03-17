"""
Comprehensive tests for taintflow.core.types – enumerations and dataclasses.

Tests cover construction, serialisation round-tripping, validation, comparison
operators, edge-case handling, and property accessors for every public type.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

from taintflow.core.types import (
    AnalysisConfig,
    ChannelParams,
    ColumnSchema,
    EdgeKind,
    FeatureLeakage,
    LeakageReport,
    NodeKind,
    OpType,
    Origin,
    PipelineMetadata,
    ProvenanceInfo,
    Severity,
    ShapeMetadata,
    StageLeakage,
    TaintLabel,
    AnalysisPhase,
)


# ===================================================================
#  Origin enum
# ===================================================================


class TestOrigin(unittest.TestCase):
    """Tests for the Origin enumeration."""

    def test_all_members(self):
        """Origin should have TRAIN, TEST, EXTERNAL."""
        self.assertEqual(set(Origin), {Origin.TRAIN, Origin.TEST, Origin.EXTERNAL})

    def test_values(self):
        """Each member should have the expected string value."""
        self.assertEqual(Origin.TRAIN.value, "train")
        self.assertEqual(Origin.TEST.value, "test")
        self.assertEqual(Origin.EXTERNAL.value, "external")

    def test_from_str(self):
        """from_str should parse lowercase strings."""
        self.assertIs(Origin.from_str("train"), Origin.TRAIN)
        self.assertIs(Origin.from_str("test"), Origin.TEST)
        self.assertIs(Origin.from_str("external"), Origin.EXTERNAL)

    def test_from_str_whitespace(self):
        """from_str should handle leading/trailing whitespace."""
        self.assertIs(Origin.from_str("  train  "), Origin.TRAIN)
        self.assertIs(Origin.from_str("  TEST  "), Origin.TEST)

    def test_from_str_case_insensitive(self):
        """from_str should normalise to lowercase."""
        self.assertIs(Origin.from_str("TRAIN"), Origin.TRAIN)
        self.assertIs(Origin.from_str("Test"), Origin.TEST)

    def test_from_str_invalid_raises(self):
        """from_str should raise ValueError for unknown strings."""
        with self.assertRaises(ValueError):
            Origin.from_str("invalid")
        with self.assertRaises(ValueError):
            Origin.from_str("")

    def test_all_class_method(self):
        """Origin.all() should return a frozenset of all members."""
        result = Origin.all()
        self.assertIsInstance(result, frozenset)
        self.assertEqual(len(result), 3)

    def test_repr(self):
        """repr should include the class name."""
        self.assertIn("TRAIN", repr(Origin.TRAIN))


# ===================================================================
#  Severity enum
# ===================================================================


class TestSeverity(unittest.TestCase):
    """Tests for the Severity enumeration and its comparison operators."""

    def test_all_members(self):
        """Severity should have NEGLIGIBLE, WARNING, CRITICAL."""
        self.assertEqual(
            set(Severity),
            {Severity.NEGLIGIBLE, Severity.WARNING, Severity.CRITICAL},
        )

    def test_ordering_lt(self):
        """NEGLIGIBLE < WARNING < CRITICAL."""
        self.assertTrue(Severity.NEGLIGIBLE < Severity.WARNING)
        self.assertTrue(Severity.WARNING < Severity.CRITICAL)
        self.assertTrue(Severity.NEGLIGIBLE < Severity.CRITICAL)

    def test_ordering_le(self):
        """<= should hold for equal and lesser values."""
        self.assertTrue(Severity.NEGLIGIBLE <= Severity.NEGLIGIBLE)
        self.assertTrue(Severity.NEGLIGIBLE <= Severity.WARNING)
        self.assertTrue(Severity.WARNING <= Severity.CRITICAL)

    def test_ordering_gt(self):
        """CRITICAL > WARNING > NEGLIGIBLE."""
        self.assertTrue(Severity.CRITICAL > Severity.WARNING)
        self.assertTrue(Severity.WARNING > Severity.NEGLIGIBLE)

    def test_ordering_ge(self):
        """>=  should hold for equal and greater values."""
        self.assertTrue(Severity.CRITICAL >= Severity.CRITICAL)
        self.assertTrue(Severity.CRITICAL >= Severity.NEGLIGIBLE)

    def test_ordering_not_lt_when_equal(self):
        """Self is not < self."""
        self.assertFalse(Severity.WARNING < Severity.WARNING)

    def test_ordering_not_gt_when_equal(self):
        """Self is not > self."""
        self.assertFalse(Severity.WARNING > Severity.WARNING)

    def test_lt_returns_not_implemented_for_other_types(self):
        """Comparisons with non-Severity should return NotImplemented."""
        self.assertIs(Severity.WARNING.__lt__(42), NotImplemented)
        self.assertIs(Severity.WARNING.__gt__("x"), NotImplemented)

    def test_from_bits_negligible(self):
        """Bits below warn threshold should be NEGLIGIBLE."""
        self.assertEqual(Severity.from_bits(0.0), Severity.NEGLIGIBLE)
        self.assertEqual(Severity.from_bits(0.5), Severity.NEGLIGIBLE)
        self.assertEqual(Severity.from_bits(0.999), Severity.NEGLIGIBLE)

    def test_from_bits_warning(self):
        """Bits at or above warn but below crit should be WARNING."""
        self.assertEqual(Severity.from_bits(1.0), Severity.WARNING)
        self.assertEqual(Severity.from_bits(5.0), Severity.WARNING)
        self.assertEqual(Severity.from_bits(7.99), Severity.WARNING)

    def test_from_bits_critical(self):
        """Bits at or above crit threshold should be CRITICAL."""
        self.assertEqual(Severity.from_bits(8.0), Severity.CRITICAL)
        self.assertEqual(Severity.from_bits(64.0), Severity.CRITICAL)

    def test_from_bits_custom_thresholds(self):
        """Custom thresholds should be honoured."""
        self.assertEqual(Severity.from_bits(0.5, warn=0.3, crit=0.6), Severity.WARNING)
        self.assertEqual(Severity.from_bits(0.7, warn=0.3, crit=0.6), Severity.CRITICAL)


# ===================================================================
#  AnalysisPhase, NodeKind, EdgeKind enums
# ===================================================================


class TestAnalysisPhase(unittest.TestCase):
    """Tests for the AnalysisPhase enum."""

    def test_member_count(self):
        """AnalysisPhase should have 8 phases."""
        self.assertEqual(len(AnalysisPhase), 8)

    def test_repr(self):
        """repr should include the class name."""
        self.assertIn("INSTRUMENTATION", repr(AnalysisPhase.INSTRUMENTATION))


class TestNodeKind(unittest.TestCase):
    """Tests for the NodeKind enum."""

    def test_member_count(self):
        """NodeKind should have 11 members."""
        self.assertEqual(len(NodeKind), 11)

    def test_values_are_strings(self):
        """Every member value should be a string."""
        for member in NodeKind:
            self.assertIsInstance(member.value, str)


class TestEdgeKind(unittest.TestCase):
    """Tests for the EdgeKind enum."""

    def test_member_count(self):
        """EdgeKind should have 7 members."""
        self.assertEqual(len(EdgeKind), 7)

    def test_repr(self):
        """repr should include the member name."""
        self.assertIn("DATA_FLOW", repr(EdgeKind.DATA_FLOW))


# ===================================================================
#  OpType enum
# ===================================================================


class TestOpType(unittest.TestCase):
    """Tests for the OpType enum and its property accessors."""

    def test_has_many_members(self):
        """OpType should have 80+ members."""
        self.assertGreater(len(OpType), 80)

    def test_is_sklearn_positive(self):
        """Known sklearn ops should be flagged."""
        self.assertTrue(OpType.FIT.is_sklearn)
        self.assertTrue(OpType.STANDARD_SCALER.is_sklearn)
        self.assertTrue(OpType.TRAIN_TEST_SPLIT.is_sklearn)
        self.assertTrue(OpType.KNN_IMPUTER.is_sklearn)

    def test_is_sklearn_negative(self):
        """Non-sklearn ops should not be flagged."""
        self.assertFalse(OpType.GETITEM.is_sklearn)
        self.assertFalse(OpType.NP_MEAN.is_sklearn)

    def test_is_numpy(self):
        """Numpy ops should start with np_."""
        self.assertTrue(OpType.NP_MEAN.is_numpy)
        self.assertTrue(OpType.NP_DOT.is_numpy)
        self.assertFalse(OpType.MERGE.is_numpy)

    def test_is_aggregation(self):
        """Aggregation ops should be identified."""
        self.assertTrue(OpType.AGG.is_aggregation)
        self.assertTrue(OpType.GROUPBY.is_aggregation)
        self.assertTrue(OpType.NP_MEAN.is_aggregation)
        self.assertFalse(OpType.GETITEM.is_aggregation)

    def test_may_leak(self):
        """Leak-prone ops should be identified."""
        self.assertTrue(OpType.MERGE.may_leak)
        self.assertTrue(OpType.FIT.may_leak)
        self.assertTrue(OpType.STANDARD_SCALER.may_leak)
        self.assertFalse(OpType.GETITEM.may_leak)
        self.assertFalse(OpType.COPY.may_leak)

    def test_is_pandas(self):
        """Pandas ops are those that are not sklearn, not numpy, not misc."""
        self.assertTrue(OpType.GETITEM.is_pandas)
        self.assertTrue(OpType.MERGE.is_pandas)
        self.assertFalse(OpType.NP_MEAN.is_pandas)
        self.assertFalse(OpType.FIT.is_pandas)
        self.assertFalse(OpType.COPY.is_pandas)

    def test_repr(self):
        """repr should contain the member name."""
        self.assertIn("READ_CSV", repr(OpType.READ_CSV))


# ===================================================================
#  ColumnSchema
# ===================================================================


class TestColumnSchema(unittest.TestCase):
    """Tests for the ColumnSchema frozen dataclass."""

    def test_basic_construction(self):
        """Simple construction with defaults."""
        cs = ColumnSchema(name="age", dtype="int64")
        self.assertEqual(cs.name, "age")
        self.assertEqual(cs.dtype, "int64")
        self.assertTrue(cs.nullable)
        self.assertFalse(cs.is_target)
        self.assertFalse(cs.is_index)
        self.assertIsNone(cs.cardinality)

    def test_full_construction(self):
        """All fields explicitly set."""
        cs = ColumnSchema(
            name="target",
            dtype="float64",
            nullable=False,
            is_target=True,
            is_index=False,
            cardinality=2,
        )
        self.assertTrue(cs.is_target)
        self.assertEqual(cs.cardinality, 2)

    def test_frozen(self):
        """Should not allow mutation."""
        cs = ColumnSchema(name="x")
        with self.assertRaises(AttributeError):
            cs.name = "y"

    def test_to_dict_minimal(self):
        """to_dict should omit default-false optional fields."""
        d = ColumnSchema(name="x").to_dict()
        self.assertEqual(d["name"], "x")
        self.assertNotIn("is_target", d)
        self.assertNotIn("cardinality", d)

    def test_to_dict_full(self):
        """to_dict should include set flags and cardinality."""
        d = ColumnSchema(name="y", is_target=True, cardinality=10).to_dict()
        self.assertTrue(d["is_target"])
        self.assertEqual(d["cardinality"], 10)

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) should produce an equal object."""
        original = ColumnSchema(name="feat", dtype="float32", cardinality=5, is_target=True)
        restored = ColumnSchema.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_from_dict_defaults(self):
        """from_dict with minimal data should use defaults."""
        cs = ColumnSchema.from_dict({"name": "col"})
        self.assertEqual(cs.dtype, "object")
        self.assertTrue(cs.nullable)

    def test_validate_empty_name(self):
        """Validation should flag empty names."""
        cs = ColumnSchema(name="")
        errors = cs.validate()
        self.assertTrue(any("non-empty" in e for e in errors))

    def test_validate_negative_cardinality(self):
        """Validation should flag negative cardinality."""
        cs = ColumnSchema(name="x", cardinality=-1)
        errors = cs.validate()
        self.assertTrue(any("cardinality" in e.lower() for e in errors))

    def test_validate_ok(self):
        """A valid schema should produce no errors."""
        self.assertEqual(ColumnSchema(name="x").validate(), [])

    def test_entropy_bound_cardinality(self):
        """entropy_bound should return log2(cardinality) when set."""
        cs = ColumnSchema(name="x", cardinality=8)
        self.assertAlmostEqual(cs.entropy_bound(), 3.0)

    def test_entropy_bound_bool(self):
        """entropy_bound for bool dtype should return 1.0."""
        self.assertAlmostEqual(ColumnSchema(name="x", dtype="bool").entropy_bound(), 1.0)

    def test_entropy_bound_int32(self):
        """entropy_bound for int32 should return 32.0."""
        self.assertAlmostEqual(ColumnSchema(name="x", dtype="int32").entropy_bound(), 32.0)

    def test_entropy_bound_float64(self):
        """entropy_bound for float64 should return 64.0."""
        self.assertAlmostEqual(ColumnSchema(name="x", dtype="float64").entropy_bound(), 64.0)

    def test_entropy_bound_object_fallback(self):
        """entropy_bound for object dtype should default to 64.0."""
        self.assertAlmostEqual(ColumnSchema(name="x", dtype="object").entropy_bound(), 64.0)

    def test_repr(self):
        """repr should show key fields."""
        cs = ColumnSchema(name="x", dtype="int64", is_target=True)
        r = repr(cs)
        self.assertIn("x", r)
        self.assertIn("TARGET", r)


# ===================================================================
#  ShapeMetadata
# ===================================================================


class TestShapeMetadata(unittest.TestCase):
    """Tests for the ShapeMetadata frozen dataclass."""

    def test_basic_construction(self):
        """Construct with minimal fields."""
        s = ShapeMetadata(n_rows=100, n_cols=10)
        self.assertEqual(s.n_rows, 100)
        self.assertEqual(s.n_cols, 10)

    def test_auto_train_rows(self):
        """n_train_rows should be computed automatically."""
        s = ShapeMetadata(n_rows=100, n_cols=5, n_test_rows=20)
        self.assertEqual(s.n_train_rows, 80)

    def test_explicit_train_rows(self):
        """Explicit n_train_rows should be preserved when non-zero."""
        s = ShapeMetadata(n_rows=100, n_cols=5, n_test_rows=20, n_train_rows=60)
        self.assertEqual(s.n_train_rows, 60)

    def test_test_fraction(self):
        """test_fraction property should compute correctly."""
        s = ShapeMetadata(n_rows=100, n_cols=5, n_test_rows=25)
        self.assertAlmostEqual(s.test_fraction, 0.25)

    def test_test_fraction_zero_rows(self):
        """test_fraction with n_rows=0 should return 0.0."""
        s = ShapeMetadata(n_rows=0, n_cols=0)
        self.assertAlmostEqual(s.test_fraction, 0.0)

    def test_train_fraction(self):
        """train_fraction property should compute correctly."""
        s = ShapeMetadata(n_rows=200, n_cols=5, n_test_rows=50)
        self.assertAlmostEqual(s.train_fraction, 0.75)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip should preserve data."""
        original = ShapeMetadata(n_rows=500, n_cols=20, n_test_rows=100, n_external_rows=50)
        restored = ShapeMetadata.from_dict(original.to_dict())
        self.assertEqual(original.n_rows, restored.n_rows)
        self.assertEqual(original.n_cols, restored.n_cols)
        self.assertEqual(original.n_test_rows, restored.n_test_rows)
        self.assertEqual(original.n_external_rows, restored.n_external_rows)

    def test_validate_negative_rows(self):
        """Validation should flag negative n_rows."""
        s = ShapeMetadata(n_rows=-1, n_cols=5)
        self.assertTrue(len(s.validate()) > 0)

    def test_validate_test_exceeds_total(self):
        """Validation should flag n_test_rows + n_external_rows > n_rows."""
        s = ShapeMetadata(n_rows=10, n_cols=5, n_test_rows=8, n_external_rows=5)
        errors = s.validate()
        self.assertTrue(any("n_test_rows" in e for e in errors))

    def test_validate_ok(self):
        """A valid shape should produce no errors."""
        s = ShapeMetadata(n_rows=100, n_cols=10, n_test_rows=20)
        self.assertEqual(s.validate(), [])


# ===================================================================
#  ProvenanceInfo
# ===================================================================


class TestProvenanceInfo(unittest.TestCase):
    """Tests for the ProvenanceInfo frozen dataclass."""

    def test_basic_construction(self):
        """Construct with test_fraction only."""
        p = ProvenanceInfo(test_fraction=0.2)
        self.assertAlmostEqual(p.test_fraction, 0.2)
        self.assertEqual(p.origin_set, frozenset({Origin.TRAIN}))

    def test_rho_alias(self):
        """rho property should alias test_fraction."""
        p = ProvenanceInfo(test_fraction=0.3)
        self.assertAlmostEqual(p.rho, 0.3)

    def test_is_pure_train(self):
        """is_pure_train when origin_set is {TRAIN}."""
        p = ProvenanceInfo(test_fraction=0.0, origin_set=frozenset({Origin.TRAIN}))
        self.assertTrue(p.is_pure_train)
        self.assertFalse(p.is_pure_test)
        self.assertFalse(p.is_mixed)

    def test_is_pure_test(self):
        """is_pure_test when origin_set is {TEST}."""
        p = ProvenanceInfo(test_fraction=1.0, origin_set=frozenset({Origin.TEST}))
        self.assertTrue(p.is_pure_test)
        self.assertFalse(p.is_pure_train)

    def test_is_mixed(self):
        """is_mixed when both TRAIN and TEST are present."""
        p = ProvenanceInfo(
            test_fraction=0.5,
            origin_set=frozenset({Origin.TRAIN, Origin.TEST}),
        )
        self.assertTrue(p.is_mixed)

    def test_merge(self):
        """merge should union origins and average test_fraction."""
        a = ProvenanceInfo(test_fraction=0.2, origin_set=frozenset({Origin.TRAIN}), source_id="a")
        b = ProvenanceInfo(test_fraction=0.8, origin_set=frozenset({Origin.TEST}), source_id="b")
        merged = a.merge(b)
        self.assertAlmostEqual(merged.test_fraction, 0.5)
        self.assertIn(Origin.TRAIN, merged.origin_set)
        self.assertIn(Origin.TEST, merged.origin_set)
        self.assertIn("a", merged.source_id)
        self.assertIn("b", merged.source_id)

    def test_validate_out_of_range(self):
        """Validation should flag rho outside [0,1]."""
        p = ProvenanceInfo(test_fraction=1.5)
        self.assertTrue(len(p.validate()) > 0)

    def test_validate_empty_origins(self):
        """Validation should flag empty origin_set."""
        p = ProvenanceInfo(test_fraction=0.5, origin_set=frozenset())
        self.assertTrue(any("non-empty" in e for e in p.validate()))

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip should preserve data."""
        original = ProvenanceInfo(
            test_fraction=0.3,
            origin_set=frozenset({Origin.TRAIN, Origin.EXTERNAL}),
            source_id="src1",
            description="desc",
        )
        restored = ProvenanceInfo.from_dict(original.to_dict())
        self.assertAlmostEqual(original.test_fraction, restored.test_fraction)
        self.assertEqual(original.origin_set, restored.origin_set)
        self.assertEqual(original.source_id, restored.source_id)


# ===================================================================
#  TaintLabel
# ===================================================================


class TestTaintLabel(unittest.TestCase):
    """Tests for the TaintLabel frozen dataclass."""

    def test_default_is_clean(self):
        """Default TaintLabel should be clean."""
        t = TaintLabel()
        self.assertTrue(t.is_clean)
        self.assertFalse(t.is_test_tainted)

    def test_test_tainted(self):
        """Should detect test-tainted labels."""
        t = TaintLabel(origins=frozenset({Origin.TEST}), bit_bound=5.0)
        self.assertTrue(t.is_test_tainted)
        self.assertFalse(t.is_clean)

    def test_clean_when_zero_bits(self):
        """Even with TEST origin, zero bits means clean."""
        t = TaintLabel(origins=frozenset({Origin.TEST}), bit_bound=0.0)
        self.assertTrue(t.is_clean)

    def test_severity_property(self):
        """severity should delegate to Severity.from_bits."""
        self.assertEqual(TaintLabel(bit_bound=0.5).severity, Severity.NEGLIGIBLE)
        self.assertEqual(TaintLabel(bit_bound=3.0).severity, Severity.WARNING)
        self.assertEqual(TaintLabel(bit_bound=10.0).severity, Severity.CRITICAL)

    def test_join(self):
        """Join should union origins and take max bit_bound."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        b = TaintLabel(origins=frozenset({Origin.TEST}), bit_bound=5.0)
        joined = a.join(b)
        self.assertEqual(joined.origins, frozenset({Origin.TRAIN, Origin.TEST}))
        self.assertAlmostEqual(joined.bit_bound, 5.0)

    def test_meet(self):
        """Meet should intersect origins and take min bit_bound."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        b = TaintLabel(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        met = a.meet(b)
        self.assertEqual(met.origins, frozenset({Origin.TEST}))
        self.assertAlmostEqual(met.bit_bound, 3.0)

    def test_eq_close_floats(self):
        """Equality should use isclose for bit_bound."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0 + 1e-14)
        self.assertEqual(a, b)

    def test_neq_different_origins(self):
        """Different origins should produce inequality."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintLabel(origins=frozenset({Origin.TEST}), bit_bound=1.0)
        self.assertNotEqual(a, b)

    def test_hash_consistency(self):
        """Equal objects should have equal hashes."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        self.assertEqual(hash(a), hash(b))

    def test_lt_by_origins(self):
        """Subset origins should be less-than."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintLabel(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=1.0)
        self.assertTrue(a < b)

    def test_le(self):
        """<= should hold for subset origins and <= bit_bound."""
        a = TaintLabel(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintLabel(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=2.0)
        self.assertTrue(a <= b)
        self.assertTrue(a <= a)

    def test_validate_negative_bits(self):
        """Validation should flag negative bit_bound."""
        t = TaintLabel(bit_bound=-1.0)
        self.assertTrue(len(t.validate()) > 0)

    def test_validate_nan_bits(self):
        """Validation should flag NaN bit_bound."""
        t = TaintLabel(bit_bound=float("nan"))
        self.assertTrue(len(t.validate()) > 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        original = TaintLabel(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=4.5)
        restored = TaintLabel.from_dict(original.to_dict())
        self.assertEqual(original.origins, restored.origins)
        self.assertAlmostEqual(original.bit_bound, restored.bit_bound)


# ===================================================================
#  FeatureLeakage
# ===================================================================


class TestFeatureLeakage(unittest.TestCase):
    """Tests for the FeatureLeakage dataclass."""

    def _make(self, **kwargs):
        defaults = dict(
            column_name="col_a",
            bit_bound=5.0,
            severity=Severity.WARNING,
            origins=frozenset({Origin.TEST}),
        )
        defaults.update(kwargs)
        return FeatureLeakage(**defaults)

    def test_basic_construction(self):
        """Construct a FeatureLeakage with defaults."""
        fl = self._make()
        self.assertEqual(fl.column_name, "col_a")
        self.assertAlmostEqual(fl.bit_bound, 5.0)

    def test_is_critical(self):
        """is_critical should check severity."""
        self.assertTrue(self._make(severity=Severity.CRITICAL).is_critical)
        self.assertFalse(self._make(severity=Severity.WARNING).is_critical)

    def test_is_clean(self):
        """is_clean should check for NEGLIGIBLE severity."""
        self.assertTrue(self._make(severity=Severity.NEGLIGIBLE).is_clean)
        self.assertFalse(self._make(severity=Severity.WARNING).is_clean)

    def test_validate_empty_column(self):
        """Validation should flag empty column_name."""
        fl = self._make(column_name="")
        self.assertTrue(len(fl.validate()) > 0)

    def test_validate_negative_bits(self):
        """Validation should flag negative bit_bound."""
        fl = self._make(bit_bound=-1.0)
        self.assertTrue(any("bit_bound" in e for e in fl.validate()))

    def test_validate_confidence_out_of_range(self):
        """Validation should flag confidence outside [0,1]."""
        fl = self._make(confidence=1.5)
        self.assertTrue(any("confidence" in e for e in fl.validate()))

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        fl = self._make(
            contributing_stages=["scaler", "imputer"],
            remediation="split before fit",
            explanation="test data leaked",
            confidence=0.9,
        )
        restored = FeatureLeakage.from_dict(fl.to_dict())
        self.assertEqual(fl.column_name, restored.column_name)
        self.assertAlmostEqual(fl.bit_bound, restored.bit_bound)
        self.assertEqual(fl.severity, restored.severity)
        self.assertEqual(fl.origins, restored.origins)
        self.assertEqual(fl.contributing_stages, restored.contributing_stages)

    def test_eq(self):
        """Equality should compare key fields."""
        a = self._make()
        b = self._make()
        self.assertEqual(a, b)

    def test_neq_different_column(self):
        """Different column name should not be equal."""
        a = self._make(column_name="a")
        b = self._make(column_name="b")
        self.assertNotEqual(a, b)


# ===================================================================
#  StageLeakage
# ===================================================================


class TestStageLeakage(unittest.TestCase):
    """Tests for the StageLeakage dataclass."""

    def _make_feature(self, col="x", bits=5.0, sev=Severity.WARNING):
        return FeatureLeakage(
            column_name=col, bit_bound=bits, severity=sev,
            origins=frozenset({Origin.TEST}),
        )

    def _make_stage(self, features=None, **kwargs):
        defaults = dict(
            stage_id="s1",
            stage_name="StandardScaler",
            op_type=OpType.STANDARD_SCALER,
            node_kind=NodeKind.TRANSFORM,
            max_bit_bound=5.0,
            mean_bit_bound=3.0,
        )
        defaults.update(kwargs)
        if features is not None:
            defaults["feature_leakages"] = features
        return StageLeakage(**defaults)

    def test_basic_construction(self):
        """Construct with no features."""
        sl = self._make_stage()
        self.assertEqual(sl.stage_id, "s1")
        self.assertEqual(sl.n_leaking_features, 0)

    def test_auto_severity_from_features(self):
        """Severity should be inferred from feature leakages."""
        features = [
            self._make_feature(col="a", bits=2.0, sev=Severity.WARNING),
            self._make_feature(col="b", bits=10.0, sev=Severity.CRITICAL),
        ]
        sl = self._make_stage(features=features)
        self.assertEqual(sl.severity, Severity.CRITICAL)

    def test_n_leaking_features(self):
        """n_leaking_features counts non-clean features."""
        features = [
            self._make_feature(col="a", bits=0.0, sev=Severity.NEGLIGIBLE),
            self._make_feature(col="b", bits=5.0, sev=Severity.WARNING),
        ]
        sl = self._make_stage(features=features)
        self.assertEqual(sl.n_leaking_features, 1)

    def test_total_bit_bound(self):
        """total_bit_bound sums feature bit_bounds."""
        features = [
            self._make_feature(col="a", bits=2.0),
            self._make_feature(col="b", bits=3.0),
        ]
        sl = self._make_stage(features=features)
        self.assertAlmostEqual(sl.total_bit_bound, 5.0)

    def test_validate_empty_stage_id(self):
        """Validation should flag empty stage_id."""
        sl = self._make_stage(stage_id="")
        self.assertTrue(len(sl.validate()) > 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        features = [self._make_feature(col="c", bits=4.0)]
        sl = self._make_stage(features=features)
        restored = StageLeakage.from_dict(sl.to_dict())
        self.assertEqual(sl.stage_id, restored.stage_id)
        self.assertEqual(sl.op_type, restored.op_type)
        self.assertEqual(len(restored.feature_leakages), 1)


# ===================================================================
#  LeakageReport
# ===================================================================


class TestLeakageReport(unittest.TestCase):
    """Tests for the LeakageReport dataclass."""

    def _make_report(self, stages=None):
        return LeakageReport(
            pipeline_name="test_pipeline",
            stage_leakages=stages or [],
        )

    def _make_stage(self, severity=Severity.WARNING, max_bits=5.0):
        fl = FeatureLeakage(
            column_name="f", bit_bound=max_bits,
            severity=severity, origins=frozenset({Origin.TEST}),
        )
        return StageLeakage(
            stage_id="s1", stage_name="scaler",
            op_type=OpType.STANDARD_SCALER,
            node_kind=NodeKind.TRANSFORM,
            max_bit_bound=max_bits, mean_bit_bound=max_bits,
            feature_leakages=[fl],
        )

    def test_empty_report_is_clean(self):
        """An empty report should be clean."""
        r = self._make_report()
        self.assertTrue(r.is_clean)
        self.assertEqual(r.n_stages, 0)

    def test_report_with_stages(self):
        """Report should compute aggregate statistics."""
        s = self._make_stage(severity=Severity.CRITICAL, max_bits=10.0)
        r = self._make_report(stages=[s])
        self.assertEqual(r.n_stages, 1)
        self.assertEqual(r.overall_severity, Severity.CRITICAL)
        self.assertAlmostEqual(r.total_bit_bound, 10.0)

    def test_summary_line(self):
        """summary_line should be a readable string."""
        r = self._make_report()
        self.assertIn("test_pipeline", r.summary_line)

    def test_stages_by_severity(self):
        """Stages should be sorted with CRITICAL first."""
        s1 = self._make_stage(severity=Severity.WARNING, max_bits=2.0)
        s1.stage_id = "low"
        s2 = self._make_stage(severity=Severity.CRITICAL, max_bits=10.0)
        s2.stage_id = "high"
        r = self._make_report(stages=[s1, s2])
        sorted_stages = r.stages_by_severity()
        self.assertEqual(sorted_stages[0].severity, Severity.CRITICAL)

    def test_validate_empty_name(self):
        """Validation should flag empty pipeline_name."""
        r = LeakageReport(pipeline_name="")
        self.assertTrue(len(r.validate()) > 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        r = self._make_report(stages=[self._make_stage()])
        restored = LeakageReport.from_dict(r.to_dict())
        self.assertEqual(r.pipeline_name, restored.pipeline_name)
        self.assertEqual(len(restored.stage_leakages), 1)

    def test_fingerprint_deterministic(self):
        """Fingerprint should be deterministic."""
        r = self._make_report()
        self.assertEqual(r.fingerprint(), r.fingerprint())

    def test_fingerprint_different_for_different_reports(self):
        """Different reports should (almost certainly) have different fingerprints."""
        r1 = LeakageReport(pipeline_name="a")
        r2 = LeakageReport(pipeline_name="b")
        self.assertNotEqual(r1.fingerprint(), r2.fingerprint())


# ===================================================================
#  PipelineMetadata
# ===================================================================


class TestPipelineMetadata(unittest.TestCase):
    """Tests for the PipelineMetadata frozen dataclass."""

    def test_basic_construction(self):
        """Construct with name only."""
        pm = PipelineMetadata(name="pipe1")
        self.assertEqual(pm.name, "pipe1")
        self.assertEqual(pm.n_stages, 0)

    def test_full_construction(self):
        """All fields set."""
        pm = PipelineMetadata(
            name="pipe2", source_file="main.py", n_stages=5, n_edges=8,
            libraries=("sklearn", "pandas"), python_version="3.11",
            framework="sklearn", description="test pipeline",
        )
        self.assertEqual(pm.libraries, ("sklearn", "pandas"))
        self.assertEqual(pm.framework, "sklearn")

    def test_validate_empty_name(self):
        """Validation should flag empty name."""
        pm = PipelineMetadata(name="")
        self.assertTrue(len(pm.validate()) > 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        pm = PipelineMetadata(
            name="p", n_stages=3, libraries=("numpy",), framework="tf",
        )
        restored = PipelineMetadata.from_dict(pm.to_dict())
        self.assertEqual(pm.name, restored.name)
        self.assertEqual(pm.n_stages, restored.n_stages)
        self.assertEqual(pm.libraries, restored.libraries)

    def test_to_dict_omits_empty_optionals(self):
        """to_dict should omit empty optional fields."""
        pm = PipelineMetadata(name="p")
        d = pm.to_dict()
        self.assertNotIn("source_file", d)
        self.assertNotIn("libraries", d)


# ===================================================================
#  AnalysisConfig
# ===================================================================


class TestAnalysisConfig(unittest.TestCase):
    """Tests for the AnalysisConfig dataclass."""

    def test_default_values(self):
        """Default config should have standard values."""
        ac = AnalysisConfig()
        self.assertAlmostEqual(ac.b_max, 64.0)
        self.assertAlmostEqual(ac.alpha, 0.05)
        self.assertEqual(ac.max_iterations, 1000)
        self.assertTrue(ac.use_widening)
        self.assertTrue(ac.use_narrowing)

    def test_custom_values(self):
        """Custom values should be preserved."""
        ac = AnalysisConfig(b_max=32.0, alpha=0.01, max_iterations=500)
        self.assertAlmostEqual(ac.b_max, 32.0)
        self.assertAlmostEqual(ac.alpha, 0.01)
        self.assertEqual(ac.max_iterations, 500)

    def test_validate_invalid_b_max(self):
        """Validation should flag b_max <= 0."""
        ac = AnalysisConfig(b_max=0.0)
        self.assertTrue(len(ac.validate()) > 0)

    def test_validate_invalid_alpha(self):
        """Validation should flag alpha outside (0,1)."""
        ac = AnalysisConfig(alpha=0.0)
        self.assertTrue(len(ac.validate()) > 0)
        ac2 = AnalysisConfig(alpha=1.0)
        self.assertTrue(len(ac2.validate()) > 0)

    def test_validate_invalid_iterations(self):
        """Validation should flag max_iterations < 1."""
        ac = AnalysisConfig(max_iterations=0)
        self.assertTrue(len(ac.validate()) > 0)

    def test_validate_ok(self):
        """Default config should pass validation."""
        self.assertEqual(AnalysisConfig().validate(), [])

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        ac = AnalysisConfig(b_max=48.0, alpha=0.1, parallel=True, n_workers=4)
        restored = AnalysisConfig.from_dict(ac.to_dict())
        self.assertAlmostEqual(ac.b_max, restored.b_max)
        self.assertAlmostEqual(ac.alpha, restored.alpha)
        self.assertEqual(ac.parallel, restored.parallel)
        self.assertEqual(ac.n_workers, restored.n_workers)


# ===================================================================
#  ChannelParams
# ===================================================================


class TestChannelParams(unittest.TestCase):
    """Tests for the ChannelParams frozen dataclass."""

    def test_default_values(self):
        """Default channel should be deterministic with zero capacity."""
        cp = ChannelParams()
        self.assertAlmostEqual(cp.capacity_bits, 0.0)
        self.assertTrue(cp.is_deterministic)
        self.assertTrue(cp.is_noiseless)

    def test_total_capacity(self):
        """total_capacity = capacity_bits * n_uses."""
        cp = ChannelParams(capacity_bits=3.0, n_uses=4)
        self.assertAlmostEqual(cp.total_capacity, 12.0)

    def test_attenuated_capacity(self):
        """attenuated_capacity scales total_capacity."""
        cp = ChannelParams(capacity_bits=10.0, n_uses=1)
        self.assertAlmostEqual(cp.attenuated_capacity(0.5), 5.0)
        self.assertAlmostEqual(cp.attenuated_capacity(0.0), 0.0)
        self.assertAlmostEqual(cp.attenuated_capacity(1.5), 10.0)

    def test_gaussian_capacity(self):
        """gaussian_capacity should compute 0.5 * log2(1 + SNR)."""
        cp = ChannelParams(capacity_bits=100.0, noise_variance=1.0)
        result = cp.gaussian_capacity(signal_power=1.0)
        self.assertAlmostEqual(result, 0.5 * math.log2(2.0))

    def test_gaussian_capacity_noiseless(self):
        """gaussian_capacity with zero noise should return capacity_bits."""
        cp = ChannelParams(capacity_bits=10.0, noise_variance=0.0)
        self.assertAlmostEqual(cp.gaussian_capacity(1.0), 10.0)

    def test_is_noiseless(self):
        """is_noiseless should check noise_variance."""
        self.assertTrue(ChannelParams(noise_variance=0.0).is_noiseless)
        self.assertFalse(ChannelParams(noise_variance=0.1).is_noiseless)

    def test_validate_negative_capacity(self):
        """Validation should flag negative capacity_bits."""
        cp = ChannelParams(capacity_bits=-1.0)
        self.assertTrue(len(cp.validate()) > 0)

    def test_validate_negative_n_uses(self):
        """Validation should flag n_uses < 1."""
        cp = ChannelParams(n_uses=0)
        self.assertTrue(len(cp.validate()) > 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        cp = ChannelParams(
            capacity_bits=5.0, noise_variance=0.1, n_uses=3,
            channel_type="gaussian", input_alphabet_size=256,
            description="test",
        )
        restored = ChannelParams.from_dict(cp.to_dict())
        self.assertEqual(cp, restored)

    def test_eq(self):
        """Equality should use isclose for floats."""
        a = ChannelParams(capacity_bits=1.0, noise_variance=0.5, n_uses=2, channel_type="g")
        b = ChannelParams(capacity_bits=1.0, noise_variance=0.5, n_uses=2, channel_type="g")
        self.assertEqual(a, b)

    def test_neq(self):
        """Different params should not be equal."""
        a = ChannelParams(capacity_bits=1.0)
        b = ChannelParams(capacity_bits=2.0)
        self.assertNotEqual(a, b)

    def test_hash_consistency(self):
        """Equal objects should have equal hashes."""
        a = ChannelParams(capacity_bits=1.0, n_uses=2)
        b = ChannelParams(capacity_bits=1.0, n_uses=2)
        self.assertEqual(hash(a), hash(b))

    def test_to_dict_omits_defaults(self):
        """to_dict should omit zero noise_variance and default n_uses."""
        cp = ChannelParams(capacity_bits=1.0)
        d = cp.to_dict()
        self.assertNotIn("noise_variance", d)
        self.assertNotIn("n_uses", d)


if __name__ == "__main__":
    unittest.main()
