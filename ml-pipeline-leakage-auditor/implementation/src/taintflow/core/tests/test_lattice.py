"""
Comprehensive tests for taintflow.core.lattice – Partition-Taint Lattice.

Tests cover TaintElement creation and lattice operations (join, meet, leq,
bottom, top), PartitionTaintLattice factory methods, ColumnTaintMap pointwise
operations, DataFrameAbstractState merge and projection, and the
WidenOperator / NarrowOperator behaviour.
"""

from __future__ import annotations

import math
import unittest

from taintflow.core.types import Origin, Severity, ProvenanceInfo, ShapeMetadata
from taintflow.core.lattice import (
    ColumnTaintMap,
    DataFrameAbstractState,
    NarrowOperator,
    PartitionTaintLattice,
    TaintElement,
    WidenOperator,
)


# ===================================================================
#  TaintElement
# ===================================================================


class TestTaintElementCreation(unittest.TestCase):
    """Tests for TaintElement construction and validation."""

    def test_default_is_bottom(self):
        """Default TaintElement should be bottom (∅, 0)."""
        e = TaintElement()
        self.assertEqual(e.origins, frozenset())
        self.assertAlmostEqual(e.bit_bound, 0.0)
        self.assertTrue(e.is_bottom)

    def test_with_origins_and_bits(self):
        """Construct with specific origins and bits."""
        e = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=5.0)
        self.assertIn(Origin.TEST, e.origins)
        self.assertAlmostEqual(e.bit_bound, 5.0)

    def test_negative_bits_clamped_to_zero(self):
        """Negative bit_bound should be clamped to 0."""
        e = TaintElement(bit_bound=-10.0)
        self.assertAlmostEqual(e.bit_bound, 0.0)

    def test_bits_exceeding_bmax_clamped(self):
        """bit_bound > B_MAX should be clamped to B_MAX."""
        e = TaintElement(bit_bound=100.0, B_MAX=64.0)
        self.assertAlmostEqual(e.bit_bound, 64.0)

    def test_nan_bits_clamped_to_bmax(self):
        """NaN bit_bound should be set to B_MAX."""
        e = TaintElement(bit_bound=float("nan"), B_MAX=32.0)
        self.assertAlmostEqual(e.bit_bound, 32.0)

    def test_custom_bmax(self):
        """Custom B_MAX should be honoured."""
        e = TaintElement(bit_bound=50.0, B_MAX=128.0)
        self.assertAlmostEqual(e.bit_bound, 50.0)
        self.assertAlmostEqual(e.B_MAX, 128.0)

    def test_is_top(self):
        """Top element has all origins and bit_bound == B_MAX."""
        all_origins = frozenset(Origin)
        e = TaintElement(origins=all_origins, bit_bound=64.0, B_MAX=64.0)
        self.assertTrue(e.is_top)

    def test_not_top_when_missing_origin(self):
        """Missing an origin means not top."""
        e = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=64.0)
        self.assertFalse(e.is_top)

    def test_is_test_tainted(self):
        """is_test_tainted requires TEST in origins and positive bits."""
        self.assertTrue(
            TaintElement(origins=frozenset({Origin.TEST}), bit_bound=1.0).is_test_tainted
        )
        self.assertFalse(
            TaintElement(origins=frozenset({Origin.TEST}), bit_bound=0.0).is_test_tainted
        )
        self.assertFalse(
            TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0).is_test_tainted
        )

    def test_severity_property(self):
        """severity delegates to Severity.from_bits."""
        self.assertEqual(
            TaintElement(bit_bound=0.5).severity, Severity.NEGLIGIBLE
        )
        self.assertEqual(
            TaintElement(bit_bound=3.0).severity, Severity.WARNING
        )
        self.assertEqual(
            TaintElement(bit_bound=10.0).severity, Severity.CRITICAL
        )

    def test_validate_ok(self):
        """Valid element should produce no errors."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        self.assertEqual(e.validate(), [])


class TestTaintElementLatticeOps(unittest.TestCase):
    """Tests for TaintElement lattice operations (join, meet, leq)."""

    def test_join_union_origins(self):
        """Join should union origins."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        b = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        j = a.join(b)
        self.assertEqual(j.origins, frozenset({Origin.TRAIN, Origin.TEST}))

    def test_join_max_bits(self):
        """Join should take max bit_bound."""
        a = TaintElement(bit_bound=2.0)
        b = TaintElement(bit_bound=5.0)
        self.assertAlmostEqual(a.join(b).bit_bound, 5.0)

    def test_join_with_bottom(self):
        """Joining with bottom should yield self."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        bot = TaintElement()
        self.assertEqual(e.join(bot), e)

    def test_join_idempotent(self):
        """Join of self with self should equal self."""
        e = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=4.0)
        self.assertEqual(e.join(e), e)

    def test_join_commutative(self):
        """a ⊔ b == b ⊔ a."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        b = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=5.0)
        self.assertEqual(a.join(b), b.join(a))

    def test_join_associative(self):
        """(a ⊔ b) ⊔ c == a ⊔ (b ⊔ c)."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        c = TaintElement(origins=frozenset({Origin.EXTERNAL}), bit_bound=2.0)
        self.assertEqual(a.join(b).join(c), a.join(b.join(c)))

    def test_meet_intersect_origins(self):
        """Meet should intersect origins."""
        a = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        b = TaintElement(origins=frozenset({Origin.TEST, Origin.EXTERNAL}), bit_bound=3.0)
        m = a.meet(b)
        self.assertEqual(m.origins, frozenset({Origin.TEST}))

    def test_meet_min_bits(self):
        """Meet should take min bit_bound."""
        a = TaintElement(bit_bound=5.0)
        b = TaintElement(bit_bound=3.0)
        self.assertAlmostEqual(a.meet(b).bit_bound, 3.0)

    def test_meet_with_top(self):
        """Meeting with top should yield self."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        top = TaintElement(origins=frozenset(Origin), bit_bound=64.0)
        result = e.meet(top)
        self.assertEqual(result.origins, e.origins)
        self.assertAlmostEqual(result.bit_bound, e.bit_bound)

    def test_meet_commutative(self):
        """a ⊓ b == b ⊓ a."""
        a = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        b = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        self.assertEqual(a.meet(b), b.meet(a))

    def test_leq_bottom_leq_everything(self):
        """Bottom ⊑ everything."""
        bot = TaintElement()
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        self.assertTrue(bot.leq(e))

    def test_leq_self(self):
        """Self ⊑ self."""
        e = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        self.assertTrue(e.leq(e))

    def test_leq_subset_and_lower_bits(self):
        """⊑ requires subset origins AND lower bits."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        self.assertTrue(a.leq(b))

    def test_not_leq_bigger_origins_smaller_bits(self):
        """Not ⊑ when origins is bigger but bits is smaller."""
        a = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=1.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        self.assertFalse(a.leq(b))

    def test_not_leq_bigger_bits(self):
        """Not ⊑ when bit_bound is larger."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=10.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        self.assertFalse(a.leq(b))

    def test_partial_order_alias(self):
        """partial_order is an alias for leq."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        self.assertEqual(a.leq(b), a.partial_order(b))


class TestTaintElementArithmetic(unittest.TestCase):
    """Tests for TaintElement arithmetic helpers."""

    def test_add_bits(self):
        """add_bits should increase bit_bound."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        result = e.add_bits(3.0)
        self.assertAlmostEqual(result.bit_bound, 5.0)
        self.assertEqual(result.origins, e.origins)

    def test_add_bits_clamped(self):
        """add_bits should clamp to B_MAX."""
        e = TaintElement(bit_bound=60.0, B_MAX=64.0)
        result = e.add_bits(10.0)
        self.assertAlmostEqual(result.bit_bound, 64.0)

    def test_scale_bits(self):
        """scale_bits should multiply bit_bound."""
        e = TaintElement(bit_bound=4.0)
        result = e.scale_bits(0.5)
        self.assertAlmostEqual(result.bit_bound, 2.0)

    def test_scale_bits_negative_factor(self):
        """Negative factor should clamp to 0."""
        e = TaintElement(bit_bound=4.0)
        result = e.scale_bits(-1.0)
        self.assertAlmostEqual(result.bit_bound, 0.0)

    def test_with_origins(self):
        """with_origins should add origins."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=2.0)
        result = e.with_origins(frozenset({Origin.TEST}))
        self.assertEqual(result.origins, frozenset({Origin.TRAIN, Origin.TEST}))
        self.assertAlmostEqual(result.bit_bound, 2.0)

    def test_restrict_origins(self):
        """restrict_origins should keep only specified origins."""
        e = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        result = e.restrict_origins(frozenset({Origin.TRAIN}))
        self.assertEqual(result.origins, frozenset({Origin.TRAIN}))

    def test_restrict_origins_to_empty(self):
        """restrict_origins to empty set should zero bits."""
        e = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=5.0)
        result = e.restrict_origins(frozenset())
        self.assertEqual(result.origins, frozenset())
        self.assertAlmostEqual(result.bit_bound, 0.0)


class TestTaintElementSerialization(unittest.TestCase):
    """Tests for TaintElement to_dict / from_dict / comparison / hashing."""

    def test_to_dict_structure(self):
        """to_dict should include origins, bit_bound, b_max."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0, B_MAX=64.0)
        d = e.to_dict()
        self.assertIn("origins", d)
        self.assertIn("bit_bound", d)
        self.assertIn("b_max", d)
        self.assertIsInstance(d["origins"], list)

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) should produce an equal element."""
        e = TaintElement(
            origins=frozenset({Origin.TRAIN, Origin.EXTERNAL}),
            bit_bound=7.5, B_MAX=128.0,
        )
        restored = TaintElement.from_dict(e.to_dict())
        self.assertEqual(e, restored)

    def test_eq_close_bits(self):
        """Equality uses isclose tolerance."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0 + 1e-14)
        self.assertEqual(a, b)

    def test_hash_equal_objects(self):
        """Equal objects should have equal hashes."""
        a = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        b = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        self.assertEqual(hash(a), hash(b))

    def test_lt(self):
        """< should match leq-and-not-equal."""
        bot = TaintElement()
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        self.assertTrue(bot < e)
        self.assertFalse(e < bot)
        self.assertFalse(e < e)

    def test_le(self):
        """<= should match leq."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        self.assertTrue(e <= e)

    def test_gt(self):
        """> should be the reverse of <."""
        bot = TaintElement()
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        self.assertTrue(e > bot)
        self.assertFalse(bot > e)

    def test_ge(self):
        """>= should be the reverse of <=."""
        e = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=1.0)
        self.assertTrue(e >= e)

    def test_repr_contains_bits(self):
        """repr should show bit_bound."""
        e = TaintElement(bit_bound=3.5)
        self.assertIn("3.5", repr(e))


# ===================================================================
#  PartitionTaintLattice
# ===================================================================


class TestPartitionTaintLattice(unittest.TestCase):
    """Tests for the PartitionTaintLattice factory class."""

    def setUp(self):
        self.L = PartitionTaintLattice(b_max=64.0)

    def test_bottom(self):
        """bottom() should be (∅, 0)."""
        bot = self.L.bottom()
        self.assertTrue(bot.is_bottom)
        self.assertEqual(bot.origins, frozenset())
        self.assertAlmostEqual(bot.bit_bound, 0.0)

    def test_top(self):
        """top() should be (all origins, B_MAX)."""
        top = self.L.top()
        self.assertTrue(top.is_top)
        self.assertEqual(top.origins, frozenset(Origin))
        self.assertAlmostEqual(top.bit_bound, 64.0)

    def test_bottom_leq_top(self):
        """⊥ ⊑ ⊤."""
        self.assertTrue(self.L.leq(self.L.bottom(), self.L.top()))

    def test_not_top_leq_bottom(self):
        """⊤ ⋢ ⊥."""
        self.assertFalse(self.L.leq(self.L.top(), self.L.bottom()))

    def test_join_via_lattice(self):
        """Lattice join should delegate to element join."""
        a = self.L.train_only(2.0)
        b = self.L.test_only(3.0)
        j = self.L.join(a, b)
        self.assertEqual(j.origins, frozenset({Origin.TRAIN, Origin.TEST}))
        self.assertAlmostEqual(j.bit_bound, 3.0)

    def test_meet_via_lattice(self):
        """Lattice meet should delegate to element meet."""
        a = self.L.mixed(5.0)
        b = self.L.test_only(3.0)
        m = self.L.meet(a, b)
        self.assertEqual(m.origins, frozenset({Origin.TEST}))
        self.assertAlmostEqual(m.bit_bound, 3.0)

    def test_height(self):
        """Height should be 69 for default configuration (3 origins, B_MAX=64)."""
        self.assertEqual(self.L.height(), 69)

    def test_height_custom_bmax(self):
        """Height should adapt to custom B_MAX."""
        L2 = PartitionTaintLattice(b_max=32.0)
        self.assertEqual(L2.height(), 3 + 33 + 1)

    def test_invalid_bmax(self):
        """b_max <= 0 should raise ValueError."""
        with self.assertRaises(ValueError):
            PartitionTaintLattice(b_max=0.0)
        with self.assertRaises(ValueError):
            PartitionTaintLattice(b_max=-1.0)

    def test_is_fixpoint_same(self):
        """Identical elements should be a fixpoint."""
        e = self.L.train_only(5.0)
        self.assertTrue(self.L.is_fixpoint(e, e))

    def test_is_fixpoint_close_bits(self):
        """Very close bit_bounds (within epsilon) should be a fixpoint."""
        a = self.L.train_only(5.0)
        b = self.L.element(frozenset({Origin.TRAIN}), 5.0 + 1e-12)
        self.assertTrue(self.L.is_fixpoint(a, b))

    def test_is_not_fixpoint_different_origins(self):
        """Different origins means not a fixpoint."""
        a = self.L.train_only(5.0)
        b = self.L.test_only(5.0)
        self.assertFalse(self.L.is_fixpoint(a, b))

    def test_join_all(self):
        """join_all should be the LUB of a collection."""
        elements = [
            self.L.train_only(1.0),
            self.L.test_only(3.0),
            self.L.element(frozenset({Origin.EXTERNAL}), 2.0),
        ]
        result = self.L.join_all(elements)
        self.assertEqual(result.origins, frozenset(Origin))
        self.assertAlmostEqual(result.bit_bound, 3.0)

    def test_join_all_empty(self):
        """join_all of empty collection should be bottom."""
        result = self.L.join_all([])
        self.assertTrue(result.is_bottom)

    def test_meet_all(self):
        """meet_all should be the GLB of a collection."""
        elements = [
            self.L.top(),
            self.L.element(frozenset({Origin.TRAIN, Origin.TEST}), 10.0),
        ]
        result = self.L.meet_all(elements)
        self.assertEqual(result.origins, frozenset({Origin.TRAIN, Origin.TEST}))
        self.assertAlmostEqual(result.bit_bound, 10.0)

    def test_meet_all_empty(self):
        """meet_all of empty collection should be top."""
        result = self.L.meet_all([])
        self.assertTrue(result.is_top)

    def test_from_origin(self):
        """from_origin creates a single-origin element."""
        e = self.L.from_origin(Origin.TEST, bit_bound=2.0)
        self.assertEqual(e.origins, frozenset({Origin.TEST}))
        self.assertAlmostEqual(e.bit_bound, 2.0)

    def test_train_only(self):
        """train_only convenience method."""
        e = self.L.train_only(4.0)
        self.assertEqual(e.origins, frozenset({Origin.TRAIN}))

    def test_test_only(self):
        """test_only convenience method."""
        e = self.L.test_only(4.0)
        self.assertEqual(e.origins, frozenset({Origin.TEST}))

    def test_mixed(self):
        """mixed convenience method."""
        e = self.L.mixed(6.0)
        self.assertEqual(e.origins, frozenset({Origin.TRAIN, Origin.TEST}))

    def test_repr(self):
        """repr should mention B_max."""
        self.assertIn("64", repr(self.L))


# ===================================================================
#  ColumnTaintMap
# ===================================================================


class TestColumnTaintMap(unittest.TestCase):
    """Tests for ColumnTaintMap: per-column lattice state."""

    def setUp(self):
        self.L = PartitionTaintLattice()

    def test_empty_map(self):
        """Empty map should have length 0."""
        m = ColumnTaintMap()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.columns(), [])

    def test_set_and_get(self):
        """Set and get operations."""
        m = ColumnTaintMap()
        e = self.L.train_only(3.0)
        m["col_a"] = e
        self.assertEqual(m["col_a"], e)

    def test_contains(self):
        """__contains__ checks column existence."""
        m = ColumnTaintMap({"x": self.L.bottom()})
        self.assertIn("x", m)
        self.assertNotIn("y", m)

    def test_get_default(self):
        """get() should return default for missing columns."""
        m = ColumnTaintMap()
        result = m.get("missing")
        self.assertTrue(result.is_bottom)

    def test_delete(self):
        """__delitem__ should remove a column."""
        m = ColumnTaintMap({"x": self.L.bottom()})
        del m["x"]
        self.assertNotIn("x", m)

    def test_iter(self):
        """__iter__ should yield column names."""
        m = ColumnTaintMap({"a": self.L.bottom(), "b": self.L.bottom()})
        self.assertEqual(set(m), {"a", "b"})

    def test_join_maps_union_columns(self):
        """join_maps should include columns from both maps."""
        a = ColumnTaintMap({"x": self.L.train_only(2.0)})
        b = ColumnTaintMap({"y": self.L.test_only(3.0)})
        j = a.join_maps(b)
        self.assertIn("x", j)
        self.assertIn("y", j)

    def test_join_maps_common_column(self):
        """join_maps should join values for common columns."""
        e1 = self.L.train_only(2.0)
        e2 = self.L.test_only(5.0)
        a = ColumnTaintMap({"x": e1})
        b = ColumnTaintMap({"x": e2})
        j = a.join_maps(b)
        self.assertEqual(j["x"].origins, frozenset({Origin.TRAIN, Origin.TEST}))
        self.assertAlmostEqual(j["x"].bit_bound, 5.0)

    def test_meet_maps_common_columns_only(self):
        """meet_maps should only keep common columns."""
        a = ColumnTaintMap({"x": self.L.train_only(5.0), "y": self.L.bottom()})
        b = ColumnTaintMap({"x": self.L.train_only(3.0), "z": self.L.bottom()})
        m = a.meet_maps(b)
        self.assertIn("x", m)
        self.assertNotIn("y", m)
        self.assertNotIn("z", m)

    def test_meet_maps_values(self):
        """meet_maps should meet values for common columns."""
        e1 = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        e2 = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        a = ColumnTaintMap({"x": e1})
        b = ColumnTaintMap({"x": e2})
        m = a.meet_maps(b)
        self.assertEqual(m["x"].origins, frozenset({Origin.TRAIN}))
        self.assertAlmostEqual(m["x"].bit_bound, 3.0)

    def test_leq_maps(self):
        """leq_maps should check pointwise ordering."""
        a = ColumnTaintMap({"x": self.L.train_only(2.0)})
        b = ColumnTaintMap({"x": self.L.mixed(5.0)})
        self.assertTrue(a.leq_maps(b))
        self.assertFalse(b.leq_maps(a))

    def test_leq_maps_extra_column_in_other(self):
        """Extra columns in other shouldn't affect leq."""
        a = ColumnTaintMap({"x": self.L.train_only(1.0)})
        b = ColumnTaintMap({
            "x": self.L.train_only(5.0),
            "y": self.L.test_only(3.0),
        })
        self.assertTrue(a.leq_maps(b))

    def test_project(self):
        """project should keep only specified columns."""
        m = ColumnTaintMap({
            "a": self.L.train_only(1.0),
            "b": self.L.test_only(2.0),
            "c": self.L.bottom(),
        })
        p = m.project(["a", "c"])
        self.assertEqual(set(p), {"a", "c"})
        self.assertNotIn("b", p)

    def test_extend(self):
        """extend should add new columns."""
        m = ColumnTaintMap({"a": self.L.bottom()})
        ext = m.extend({"b": self.L.test_only(3.0)})
        self.assertIn("a", ext)
        self.assertIn("b", ext)

    def test_rename(self):
        """rename should rename columns."""
        m = ColumnTaintMap({"old": self.L.train_only(2.0)})
        renamed = m.rename({"old": "new"})
        self.assertIn("new", renamed)
        self.assertNotIn("old", renamed)

    def test_drop(self):
        """drop should remove specified columns."""
        m = ColumnTaintMap({"a": self.L.bottom(), "b": self.L.bottom()})
        d = m.drop(["a"])
        self.assertNotIn("a", d)
        self.assertIn("b", d)

    def test_max_element(self):
        """max_element should return the element with highest bit_bound."""
        m = ColumnTaintMap({
            "a": self.L.train_only(1.0),
            "b": self.L.test_only(5.0),
            "c": self.L.bottom(),
        })
        self.assertAlmostEqual(m.max_element().bit_bound, 5.0)

    def test_min_element(self):
        """min_element should return element with lowest bit_bound."""
        m = ColumnTaintMap({
            "a": self.L.train_only(1.0),
            "b": self.L.test_only(5.0),
        })
        self.assertAlmostEqual(m.min_element().bit_bound, 1.0)

    def test_total_bits(self):
        """total_bits should sum all bit_bounds."""
        m = ColumnTaintMap({
            "a": self.L.train_only(2.0),
            "b": self.L.test_only(3.0),
        })
        self.assertAlmostEqual(m.total_bits(), 5.0)

    def test_mean_bits(self):
        """mean_bits should average bit_bounds."""
        m = ColumnTaintMap({
            "a": self.L.train_only(2.0),
            "b": self.L.test_only(4.0),
        })
        self.assertAlmostEqual(m.mean_bits(), 3.0)

    def test_mean_bits_empty(self):
        """mean_bits on empty map should be 0."""
        self.assertAlmostEqual(ColumnTaintMap().mean_bits(), 0.0)

    def test_tainted_columns(self):
        """tainted_columns should list columns above threshold."""
        m = ColumnTaintMap({
            "a": self.L.train_only(0.5),
            "b": self.L.test_only(3.0),
        })
        self.assertEqual(m.tainted_columns(threshold=1.0), ["b"])

    def test_test_tainted_columns(self):
        """test_tainted_columns should list columns with test taint."""
        m = ColumnTaintMap({
            "a": self.L.train_only(5.0),
            "b": self.L.test_only(3.0),
        })
        self.assertEqual(m.test_tainted_columns(), ["b"])

    def test_all_origins(self):
        """all_origins should union all origins in the map."""
        m = ColumnTaintMap({
            "a": self.L.train_only(1.0),
            "b": self.L.test_only(1.0),
        })
        self.assertEqual(m.all_origins(), frozenset({Origin.TRAIN, Origin.TEST}))

    def test_copy(self):
        """copy should produce an independent copy."""
        m = ColumnTaintMap({"a": self.L.train_only(1.0)})
        c = m.copy()
        c["b"] = self.L.bottom()
        self.assertNotIn("b", m)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        m = ColumnTaintMap({
            "x": self.L.train_only(3.0),
            "y": self.L.test_only(7.0),
        })
        restored = ColumnTaintMap.from_dict(m.to_dict())
        self.assertEqual(m, restored)

    def test_eq(self):
        """Equal maps should be equal."""
        a = ColumnTaintMap({"x": self.L.train_only(3.0)})
        b = ColumnTaintMap({"x": self.L.train_only(3.0)})
        self.assertEqual(a, b)

    def test_neq_different_columns(self):
        """Maps with different columns should not be equal."""
        a = ColumnTaintMap({"x": self.L.bottom()})
        b = ColumnTaintMap({"y": self.L.bottom()})
        self.assertNotEqual(a, b)

    def test_apply_uniform(self):
        """apply_uniform should apply function to every element."""
        m = ColumnTaintMap({
            "a": self.L.train_only(2.0),
            "b": self.L.train_only(4.0),
        })
        scaled = m.apply_uniform(lambda e: e.scale_bits(0.5))
        self.assertAlmostEqual(scaled["a"].bit_bound, 1.0)
        self.assertAlmostEqual(scaled["b"].bit_bound, 2.0)


# ===================================================================
#  DataFrameAbstractState
# ===================================================================


class TestDataFrameAbstractState(unittest.TestCase):
    """Tests for DataFrameAbstractState."""

    def test_bottom(self):
        """bottom() should be empty."""
        s = DataFrameAbstractState.bottom()
        self.assertEqual(len(s.column_map), 0)
        self.assertTrue(s.is_clean)

    def test_from_train(self):
        """from_train should create train-only state."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        self.assertEqual(len(s.column_map), 2)
        self.assertTrue(s.is_clean)
        self.assertEqual(s.shape.n_rows, 100)
        self.assertEqual(s.shape.n_test_rows, 0)
        self.assertTrue(s.row_provenance.is_pure_train)

    def test_from_test(self):
        """from_test should create test-only state."""
        s = DataFrameAbstractState.from_test(["x"], n_rows=50)
        self.assertTrue(s.row_provenance.is_pure_test)
        self.assertEqual(s.shape.n_test_rows, 50)

    def test_from_mixed(self):
        """from_mixed should create mixed state."""
        s = DataFrameAbstractState.from_mixed(
            ["a", "b"], n_rows=100, n_test_rows=30, bit_bound=2.0,
        )
        self.assertTrue(s.row_provenance.is_mixed)
        self.assertAlmostEqual(s.row_provenance.test_fraction, 0.3)

    def test_join(self):
        """Join should merge column maps and provenance."""
        s1 = DataFrameAbstractState.from_train(["a"], n_rows=100)
        s2 = DataFrameAbstractState.from_test(["b"], n_rows=50)
        j = s1.join(s2)
        self.assertIn("a", j.column_map)
        self.assertIn("b", j.column_map)

    def test_meet(self):
        """Meet should intersect column maps."""
        s1 = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        s2 = DataFrameAbstractState.from_train(["b", "c"], n_rows=100)
        m = s1.meet(s2)
        self.assertIn("b", m.column_map)
        self.assertNotIn("a", m.column_map)
        self.assertNotIn("c", m.column_map)

    def test_leq(self):
        """leq should delegate to column_map.leq_maps."""
        s1 = DataFrameAbstractState.from_train(["a"], n_rows=100)
        s2 = DataFrameAbstractState.from_mixed(["a"], n_rows=100, n_test_rows=30)
        self.assertTrue(s1.leq(s2))

    def test_project(self):
        """project should keep only specified columns."""
        s = DataFrameAbstractState.from_train(["a", "b", "c"], n_rows=100)
        p = s.project(["a", "c"])
        self.assertEqual(set(p.column_map), {"a", "c"})
        self.assertEqual(p.shape.n_cols, 2)

    def test_extend(self):
        """extend should add new columns."""
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        L = PartitionTaintLattice()
        ext = s.extend({"b": L.test_only(3.0)})
        self.assertIn("b", ext.column_map)

    def test_drop_columns(self):
        """drop_columns should remove specified columns."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        d = s.drop_columns(["a"])
        self.assertNotIn("a", d.column_map)
        self.assertIn("b", d.column_map)

    def test_rename_columns(self):
        """rename_columns should rename columns."""
        s = DataFrameAbstractState.from_train(["old"], n_rows=100)
        r = s.rename_columns({"old": "new"})
        self.assertIn("new", r.column_map)
        self.assertNotIn("old", r.column_map)

    def test_add_leakage(self):
        """add_leakage should increase bits and add TEST origin."""
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        leaked = s.add_leakage("a", 5.0)
        self.assertTrue(leaked.column_map["a"].is_test_tainted)
        self.assertGreater(leaked.column_map["a"].bit_bound, 0.0)

    def test_taint_all_from_test(self):
        """taint_all_from_test should add test taint to all columns."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        tainted = s.taint_all_from_test(3.0)
        for col in tainted.column_map:
            self.assertTrue(tainted.column_map[col].is_test_tainted)

    def test_max_bit_bound(self):
        """max_bit_bound should return the highest bit_bound."""
        s = DataFrameAbstractState.from_mixed(
            ["a", "b"], n_rows=100, n_test_rows=20, bit_bound=5.0,
        )
        self.assertAlmostEqual(s.max_bit_bound, 5.0)

    def test_max_bit_bound_empty(self):
        """max_bit_bound on empty state should be 0."""
        s = DataFrameAbstractState.bottom()
        self.assertAlmostEqual(s.max_bit_bound, 0.0)

    def test_is_clean(self):
        """is_clean should be True when no test taint."""
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        self.assertTrue(s.is_clean)

    def test_test_tainted_columns(self):
        """test_tainted_columns should list test-tainted columns."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        s = s.add_leakage("a", 3.0)
        self.assertEqual(s.test_tainted_columns, ["a"])

    def test_to_dict_from_dict_roundtrip(self):
        """Serialisation roundtrip."""
        s = DataFrameAbstractState.from_mixed(
            ["x", "y"], n_rows=200, n_test_rows=50, bit_bound=4.0,
        )
        s.label = "test_state"
        s.stage_id = "s1"
        restored = DataFrameAbstractState.from_dict(s.to_dict())
        self.assertEqual(len(restored.column_map), 2)
        self.assertEqual(restored.label, "test_state")
        self.assertEqual(restored.stage_id, "s1")

    def test_validate_shape_mismatch(self):
        """validate should flag n_cols != len(column_map)."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        # Manually force a shape mismatch
        s.shape = ShapeMetadata(n_rows=100, n_cols=5)
        errors = s.validate()
        self.assertTrue(any("n_cols" in e for e in errors))

    def test_propagate_taint(self):
        """propagate_taint should join source taints into target."""
        s = DataFrameAbstractState.from_train(["a", "b"], n_rows=100)
        s = s.add_leakage("a", 3.0)
        s = s.add_leakage("b", 5.0)
        result = s.propagate_taint(["a", "b"], "c")
        self.assertIn("c", result.column_map)
        self.assertAlmostEqual(result.column_map["c"].bit_bound, 5.0)


# ===================================================================
#  WidenOperator
# ===================================================================


class TestWidenOperator(unittest.TestCase):
    """Tests for the WidenOperator (∇)."""

    def test_widen_element_no_increase(self):
        """When new.bit_bound <= old.bit_bound, bits should stay at old."""
        W = WidenOperator(b_max=64.0)
        old = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        new = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        widened = W.widen_element(old, new)
        self.assertAlmostEqual(widened.bit_bound, 5.0)

    def test_widen_element_increase_jumps_to_bmax(self):
        """When new.bit_bound > old.bit_bound, bits should jump to B_MAX."""
        W = WidenOperator(b_max=64.0)
        old = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        new = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=6.0)
        widened = W.widen_element(old, new)
        self.assertAlmostEqual(widened.bit_bound, 64.0)

    def test_widen_element_unions_origins(self):
        """Widening should union origins."""
        W = WidenOperator()
        old = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        new = TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)
        widened = W.widen_element(old, new)
        self.assertEqual(widened.origins, frozenset({Origin.TRAIN, Origin.TEST}))

    def test_widen_with_delay(self):
        """Within delay iterations, widen should just join."""
        W = WidenOperator(b_max=64.0, delay=3)
        s1 = DataFrameAbstractState.from_train(["a"], n_rows=100)
        s2 = DataFrameAbstractState.from_mixed(
            ["a"], n_rows=100, n_test_rows=20, bit_bound=1.0,
        )
        # First 3 iterations should use join (no widening)
        for i in range(3):
            result = W.widen(s1, s2, node_id="node1")
            # After join, bits should not jump to B_MAX
            self.assertLess(result.column_map["a"].bit_bound, 64.0)

        # 4th iteration should apply widening
        result = W.widen(s1, s2, node_id="node1")
        # Now widening should kick in
        # bit_bound of s2.a (1.0) > bit_bound of s1.a (0.0) => jump to B_MAX
        self.assertAlmostEqual(result.column_map["a"].bit_bound, 64.0)

    def test_widen_map(self):
        """widen_map should apply pointwise widening."""
        W = WidenOperator(b_max=64.0)
        old = ColumnTaintMap({"a": TaintElement(bit_bound=5.0)})
        new = ColumnTaintMap({"a": TaintElement(bit_bound=6.0)})
        widened = W.widen_map(old, new)
        self.assertAlmostEqual(widened["a"].bit_bound, 64.0)

    def test_reset(self):
        """reset should clear iteration counts."""
        W = WidenOperator(delay=2)
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        W.widen(s, s, node_id="n1")
        W.widen(s, s, node_id="n1")
        W.reset("n1")
        # After reset, iteration count should restart
        self.assertNotIn("n1", W._iteration_counts)

    def test_reset_all(self):
        """reset() with no args should clear all counts."""
        W = WidenOperator(delay=2)
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        W.widen(s, s, node_id="n1")
        W.widen(s, s, node_id="n2")
        W.reset()
        self.assertEqual(len(W._iteration_counts), 0)

    def test_repr(self):
        """repr should mention B_max and delay."""
        W = WidenOperator(b_max=32.0, delay=5)
        self.assertIn("32", repr(W))
        self.assertIn("5", repr(W))


# ===================================================================
#  NarrowOperator
# ===================================================================


class TestNarrowOperator(unittest.TestCase):
    """Tests for the NarrowOperator (△)."""

    def test_narrow_element_decrease(self):
        """When new.bit_bound < old.bit_bound, take new."""
        N = NarrowOperator()
        old = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=64.0)
        new = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        narrowed = N.narrow_element(old, new)
        self.assertAlmostEqual(narrowed.bit_bound, 5.0)

    def test_narrow_element_no_decrease(self):
        """When new.bit_bound >= old.bit_bound, keep old."""
        N = NarrowOperator()
        old = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=5.0)
        new = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=10.0)
        narrowed = N.narrow_element(old, new)
        self.assertAlmostEqual(narrowed.bit_bound, 5.0)

    def test_narrow_element_uses_new_origins(self):
        """Narrowing should use new origins."""
        N = NarrowOperator()
        old = TaintElement(origins=frozenset({Origin.TRAIN, Origin.TEST}), bit_bound=5.0)
        new = TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=3.0)
        narrowed = N.narrow_element(old, new)
        self.assertEqual(narrowed.origins, frozenset({Origin.TRAIN}))

    def test_narrow_map(self):
        """narrow_map should apply pointwise narrowing."""
        N = NarrowOperator()
        old = ColumnTaintMap({"a": TaintElement(bit_bound=64.0)})
        new = ColumnTaintMap({"a": TaintElement(bit_bound=5.0)})
        narrowed = N.narrow_map(old, new)
        self.assertAlmostEqual(narrowed["a"].bit_bound, 5.0)

    def test_narrow_state(self):
        """narrow_state should improve precision of a widened state."""
        N = NarrowOperator()
        old = DataFrameAbstractState.from_mixed(
            ["a"], n_rows=100, n_test_rows=20, bit_bound=64.0,
        )
        new = DataFrameAbstractState.from_mixed(
            ["a"], n_rows=100, n_test_rows=20, bit_bound=5.0,
        )
        narrowed = N.narrow_state(old, new)
        self.assertAlmostEqual(narrowed.column_map["a"].bit_bound, 5.0)

    def test_is_stable(self):
        """is_stable should detect convergence."""
        N = NarrowOperator(epsilon=1e-10)
        s = DataFrameAbstractState.from_train(["a"], n_rows=100)
        self.assertTrue(N.is_stable(s, s))

    def test_is_not_stable_different_bits(self):
        """is_stable should detect non-convergence."""
        N = NarrowOperator(epsilon=1e-10)
        s1 = DataFrameAbstractState.from_mixed(
            ["a"], n_rows=100, n_test_rows=20, bit_bound=5.0,
        )
        s2 = DataFrameAbstractState.from_mixed(
            ["a"], n_rows=100, n_test_rows=20, bit_bound=3.0,
        )
        self.assertFalse(N.is_stable(s1, s2))

    def test_is_not_stable_different_columns(self):
        """is_stable should return False for different column sets."""
        N = NarrowOperator()
        s1 = DataFrameAbstractState.from_train(["a"], n_rows=100)
        s2 = DataFrameAbstractState.from_train(["b"], n_rows=100)
        self.assertFalse(N.is_stable(s1, s2))

    def test_max_iterations_property(self):
        """max_iterations should be accessible."""
        N = NarrowOperator(max_iterations=10)
        self.assertEqual(N.max_iterations, 10)

    def test_epsilon_property(self):
        """epsilon should be accessible."""
        N = NarrowOperator(epsilon=1e-8)
        self.assertAlmostEqual(N.epsilon, 1e-8)

    def test_repr(self):
        """repr should mention max_iter and epsilon."""
        N = NarrowOperator(max_iterations=7, epsilon=1e-6)
        self.assertIn("7", repr(N))


if __name__ == "__main__":
    unittest.main()
