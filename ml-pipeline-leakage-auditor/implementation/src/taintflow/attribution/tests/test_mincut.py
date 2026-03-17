"""
Tests for taintflow.attribution.mincut – min-cut leakage attribution.
"""

from __future__ import annotations

import unittest

from taintflow.attribution.mincut import (
    BottleneckRanking,
    BottleneckStage,
    MinCutDecomposition,
    MinCutResult,
    SensitivityEntry,
)


class TestMinCutResult(unittest.TestCase):
    """Tests for MinCutResult dataclass."""

    def test_trivial_empty(self) -> None:
        r = MinCutResult(
            cut_edges=[], total_capacity=0.0,
            per_feature_attribution={}, source_side=set(), sink_side=set(),
        )
        self.assertTrue(r.is_trivial)
        self.assertEqual(r.n_cut_edges, 0)

    def test_nontrivial(self) -> None:
        r = MinCutResult(
            cut_edges=[("a", "b", 3.5)],
            total_capacity=3.5,
            per_feature_attribution={"feat_0": 3.5},
            source_side={"a"},
            sink_side={"b"},
        )
        self.assertFalse(r.is_trivial)
        self.assertEqual(r.n_cut_edges, 1)
        self.assertAlmostEqual(r.total_capacity, 3.5)

    def test_to_dict_roundtrip(self) -> None:
        r = MinCutResult(
            cut_edges=[("x", "y", 1.0)],
            total_capacity=1.0,
            per_feature_attribution={"col": 1.0},
            source_side={"x"},
            sink_side={"y"},
        )
        d = r.to_dict()
        self.assertIn("total_capacity", d)
        self.assertIn("cut_edges", d)

    def test_validate(self) -> None:
        r = MinCutResult(
            cut_edges=[], total_capacity=0.0,
            per_feature_attribution={}, source_side=set(), sink_side=set(),
        )
        errors = r.validate()
        self.assertIsInstance(errors, list)


class TestMinCutDecomposition(unittest.TestCase):
    """Tests for MinCutDecomposition."""

    def test_empty_decomposition(self) -> None:
        d = MinCutDecomposition(total_leakage_bits=0.0, stage_contributions=[])
        self.assertEqual(d.n_stages, 0)
        self.assertAlmostEqual(d.total_leakage_bits, 0.0)

    def test_with_stages(self) -> None:
        stages = [
            BottleneckStage(stage_id="s1", leakage_bits=2.0, fraction_of_total=0.4),
            BottleneckStage(stage_id="s2", leakage_bits=3.0, fraction_of_total=0.6),
        ]
        d = MinCutDecomposition(total_leakage_bits=5.0, stage_contributions=stages)
        self.assertEqual(d.n_stages, 2)
        top = d.top_k(1)
        self.assertEqual(len(top), 1)


class TestBottleneckStage(unittest.TestCase):
    """Tests for BottleneckStage."""

    def test_creation(self) -> None:
        bs = BottleneckStage(stage_id="scaler_fit", leakage_bits=1.5, fraction_of_total=0.75)
        self.assertEqual(bs.stage_id, "scaler_fit")
        self.assertAlmostEqual(bs.leakage_bits, 1.5)

    def test_validate(self) -> None:
        bs = BottleneckStage(stage_id="s", leakage_bits=0.5, fraction_of_total=0.5)
        errors = bs.validate()
        self.assertIsInstance(errors, list)


class TestSensitivityEntry(unittest.TestCase):
    """Tests for SensitivityEntry."""

    def test_creation(self) -> None:
        se = SensitivityEntry(stage_id="s1", original_leakage=5.0, leakage_without_stage=2.0)
        self.assertAlmostEqual(se.delta, 3.0)

    def test_relative_delta(self) -> None:
        se = SensitivityEntry(stage_id="s1", original_leakage=10.0, leakage_without_stage=5.0)
        self.assertAlmostEqual(se.relative_delta, 0.5)


if __name__ == "__main__":
    unittest.main()
