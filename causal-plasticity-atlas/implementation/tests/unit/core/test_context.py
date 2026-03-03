"""Unit tests for cpa.core.context (ContextSpace & ContextPartition)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cpa.core.context import ContextPartition, ContextSpace
from cpa.core.types import Context


# ── helpers ──────────────────────────────────────────────────────────

def _ctx(cid: str, ov: float | None = None, **meta: object) -> Context:
    return Context(id=cid, metadata=dict(meta), ordering_value=ov)


def _ordered_space() -> ContextSpace:
    return ContextSpace(
        [_ctx("a", 0.0, x=1.0), _ctx("b", 1.0, x=2.0), _ctx("c", 2.0, x=4.0)],
        ordered=True,
    )


def _unordered_space() -> ContextSpace:
    return ContextSpace([_ctx("x"), _ctx("y"), _ctx("z")])


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – construction & basic properties
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceConstruction:
    def test_empty(self):
        cs = ContextSpace()
        assert len(cs) == 0
        assert cs.size == 0
        assert cs.ids == []
        assert not cs.ordered

    def test_unordered(self):
        cs = _unordered_space()
        assert cs.size == 3
        assert set(cs.ids) == {"x", "y", "z"}
        assert not cs.ordered

    def test_ordered(self):
        cs = _ordered_space()
        assert cs.ordered
        assert cs.ids == ["a", "b", "c"]

    def test_duplicate_id_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            ContextSpace([_ctx("a"), _ctx("a")])

    def test_ordered_missing_ordering_value_raises(self):
        with pytest.raises(ValueError, match="ordering_value"):
            ContextSpace([_ctx("a")], ordered=True)


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – container protocol
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceContainer:
    def test_contains(self):
        cs = _unordered_space()
        assert "x" in cs
        assert "missing" not in cs

    def test_getitem_int(self):
        cs = _ordered_space()
        assert cs[0].id == "a"
        assert cs[2].id == "c"

    def test_getitem_str(self):
        cs = _ordered_space()
        assert cs["b"].ordering_value == 1.0

    def test_getitem_missing_str_raises(self):
        with pytest.raises(KeyError):
            _ordered_space()["nope"]

    def test_getitem_bad_type_raises(self):
        with pytest.raises(TypeError):
            _ordered_space()[3.14]  # type: ignore[index]

    def test_iter(self):
        cs = _ordered_space()
        ids = [c.id for c in cs]
        assert ids == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – add / remove / get
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceAddRemove:
    def test_add_and_get(self):
        cs = ContextSpace()
        cs.add(_ctx("q"))
        assert cs.get("q") is not None
        assert cs.get("missing") is None

    def test_remove(self):
        cs = _unordered_space()
        removed = cs.remove("y")
        assert removed.id == "y"
        assert "y" not in cs
        assert cs.size == 2

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError):
            _unordered_space().remove("nope")

    def test_add_sorts_ordered_space(self):
        cs = ContextSpace(ordered=True)
        cs.add(_ctx("b", 2.0))
        cs.add(_ctx("a", 0.5))
        cs.add(_ctx("c", 1.0))
        assert cs.ids == ["a", "c", "b"]


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – ordering operations
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceOrdering:
    def test_ordering_values(self):
        cs = _ordered_space()
        assert cs.ordering_values() == [0.0, 1.0, 2.0]

    def test_validate_ordering_strictly_increasing(self):
        cs = _ordered_space()
        assert cs.validate_ordering() is True

    def test_validate_ordering_unordered_returns_true(self):
        assert _unordered_space().validate_ordering() is True

    def test_interpolate_within_range(self):
        cs = _ordered_space()
        left, right = cs.interpolate_ordering(0.5)
        assert left.id == "a"
        assert right.id == "b"

    def test_interpolate_clamp_low(self):
        cs = _ordered_space()
        left, right = cs.interpolate_ordering(-10.0)
        assert left.id == "a"
        assert right.id == "b"

    def test_interpolate_clamp_high(self):
        cs = _ordered_space()
        left, right = cs.interpolate_ordering(100.0)
        assert left.id == "b"
        assert right.id == "c"

    def test_interpolate_unordered_raises(self):
        with pytest.raises(ValueError, match="unordered"):
            _unordered_space().interpolate_ordering(0.5)

    def test_interpolate_too_few_contexts_raises(self):
        cs = ContextSpace([_ctx("a", 1.0)], ordered=True)
        with pytest.raises(ValueError, match="at least 2"):
            cs.interpolate_ordering(0.5)

    def test_sliding_window(self):
        cs = _ordered_space()
        windows = cs.sliding_window(2)
        assert len(windows) == 2
        assert [c.id for c in windows[0]] == ["a", "b"]
        assert [c.id for c in windows[1]] == ["b", "c"]

    def test_sliding_window_larger_than_space(self):
        cs = _ordered_space()
        windows = cs.sliding_window(10)
        assert len(windows) == 1
        assert len(windows[0]) == 3

    def test_sliding_window_invalid_size_raises(self):
        with pytest.raises(ValueError):
            _ordered_space().sliding_window(0)


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – distance
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceDistance:
    def test_ordering_distance(self):
        cs = _ordered_space()
        assert cs.ordering_distance("a", "c") == pytest.approx(2.0)

    def test_ordering_distance_missing_value_raises(self):
        cs = ContextSpace([_ctx("a", 1.0), _ctx("b")])
        with pytest.raises(ValueError, match="ordering values"):
            cs.ordering_distance("a", "b")

    def test_metadata_distance_euclidean(self):
        cs = ContextSpace([
            _ctx("a", x=3.0, y=0.0),
            _ctx("b", x=0.0, y=4.0),
        ])
        d = cs.metadata_distance("a", "b", keys=["x", "y"])
        assert d == pytest.approx(5.0)

    def test_metadata_distance_manhattan(self):
        cs = ContextSpace([
            _ctx("a", x=3.0, y=0.0),
            _ctx("b", x=0.0, y=4.0),
        ])
        d = cs.metadata_distance("a", "b", keys=["x", "y"], metric="manhattan")
        assert d == pytest.approx(7.0)

    def test_metadata_distance_auto_keys(self):
        cs = ContextSpace([_ctx("a", x=1.0), _ctx("b", x=4.0)])
        d = cs.metadata_distance("a", "b")
        assert d == pytest.approx(3.0)

    def test_metadata_distance_no_shared_numeric_keys(self):
        cs = ContextSpace([
            _ctx("a", **{"name": "foo"}),
            _ctx("b", **{"name": "bar"}),
        ])
        assert cs.metadata_distance("a", "b") == 0.0

    def test_metadata_distance_unknown_metric_raises(self):
        cs = ContextSpace([_ctx("a", x=1.0), _ctx("b", x=2.0)])
        with pytest.raises(ValueError, match="Unknown metric"):
            cs.metadata_distance("a", "b", metric="cosine")

    def test_pairwise_distance_matrix_ordering(self):
        cs = _ordered_space()
        D = cs.pairwise_distance_matrix(metric="ordering")
        assert D.shape == (3, 3)
        assert D[0, 1] == pytest.approx(1.0)
        assert D[0, 2] == pytest.approx(2.0)
        np.testing.assert_array_equal(D, D.T)

    def test_pairwise_distance_matrix_metadata(self):
        cs = _ordered_space()
        D = cs.pairwise_distance_matrix(metric="metadata", keys=["x"])
        assert D[0, 1] == pytest.approx(1.0)
        assert D[0, 2] == pytest.approx(3.0)

    def test_pairwise_distance_matrix_bad_metric_raises(self):
        with pytest.raises(ValueError):
            _ordered_space().pairwise_distance_matrix(metric="bad")


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – subset operations
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceSubset:
    def test_subset(self):
        cs = _ordered_space()
        sub = cs.subset(["a", "c"])
        assert sub.ids == ["a", "c"]
        assert sub.ordered

    def test_subset_ignores_missing(self):
        cs = _ordered_space()
        sub = cs.subset(["a", "missing"])
        assert sub.ids == ["a"]

    def test_filter(self):
        cs = _ordered_space()
        filt = cs.filter(lambda c: c.ordering_value is not None and c.ordering_value > 0.5)
        assert set(filt.ids) == {"b", "c"}

    def test_union(self):
        cs1 = ContextSpace([_ctx("a"), _ctx("b")])
        cs2 = ContextSpace([_ctx("b"), _ctx("c")])
        u = cs1.union(cs2)
        assert set(u.ids) == {"a", "b", "c"}

    def test_intersection(self):
        cs1 = ContextSpace([_ctx("a"), _ctx("b")])
        cs2 = ContextSpace([_ctx("b"), _ctx("c")])
        inter = cs1.intersection(cs2)
        assert set(inter.ids) == {"b"}

    def test_difference(self):
        cs1 = ContextSpace([_ctx("a"), _ctx("b")])
        cs2 = ContextSpace([_ctx("b"), _ctx("c")])
        diff = cs1.difference(cs2)
        assert set(diff.ids) == {"a"}


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – pairs
# ═══════════════════════════════════════════════════════════════════


class TestContextSpacePairs:
    def test_pairs(self):
        cs = _ordered_space()
        p = cs.pairs()
        assert len(p) == 3  # C(3,2)

    def test_consecutive_pairs(self):
        cs = _ordered_space()
        cp = cs.consecutive_pairs()
        assert len(cp) == 2
        assert cp[0][0].id == "a" and cp[0][1].id == "b"
        assert cp[1][0].id == "b" and cp[1][1].id == "c"


# ═══════════════════════════════════════════════════════════════════
# ContextSpace – serialization round-trip
# ═══════════════════════════════════════════════════════════════════


class TestContextSpaceSerialization:
    def test_round_trip_ordered(self):
        cs = _ordered_space()
        d = cs.to_dict()
        cs2 = ContextSpace.from_dict(d)
        assert cs2.ordered is True
        assert cs2.ids == cs.ids
        assert cs2.ordering_values() == cs.ordering_values()

    def test_round_trip_unordered(self):
        cs = _unordered_space()
        cs2 = ContextSpace.from_dict(cs.to_dict())
        assert cs2.ordered is False
        assert set(cs2.ids) == set(cs.ids)


# ═══════════════════════════════════════════════════════════════════
# ContextPartition – construction & validation
# ═══════════════════════════════════════════════════════════════════


class TestContextPartitionConstruction:
    def test_basic(self):
        cs = _ordered_space()
        cp = ContextPartition({"lo": ["a"], "hi": ["b", "c"]}, cs)
        assert cp.num_groups == 2
        assert sorted(cp.group_names) == ["hi", "lo"]

    def test_overlapping_raises(self):
        cs = _ordered_space()
        with pytest.raises(ValueError, match="overlaps"):
            ContextPartition({"g1": ["a", "b"], "g2": ["b", "c"]}, cs)

    def test_unknown_id_raises(self):
        cs = _ordered_space()
        with pytest.raises(ValueError, match="not in context space"):
            ContextPartition({"g": ["missing"]}, cs)


# ═══════════════════════════════════════════════════════════════════
# ContextPartition – queries
# ═══════════════════════════════════════════════════════════════════


class TestContextPartitionQueries:
    @pytest.fixture()
    def partition(self):
        cs = _ordered_space()
        return ContextPartition({"lo": ["a"], "hi": ["b", "c"]}, cs)

    def test_group(self, partition):
        assert partition.group("lo") == ["a"]
        assert set(partition.group("hi")) == {"b", "c"}

    def test_group_missing_raises(self, partition):
        with pytest.raises(KeyError):
            partition.group("nope")

    def test_group_space(self, partition):
        sub = partition.group_space("hi")
        assert isinstance(sub, ContextSpace)
        assert set(sub.ids) == {"b", "c"}

    def test_group_sizes(self, partition):
        sizes = partition.group_sizes()
        assert sizes["lo"] == 1
        assert sizes["hi"] == 2

    def test_context_group(self, partition):
        assert partition.context_group("a") == "lo"
        assert partition.context_group("c") == "hi"

    def test_context_group_missing_raises(self, partition):
        with pytest.raises(ValueError, match="not in any"):
            partition.context_group("missing")


# ═══════════════════════════════════════════════════════════════════
# ContextPartition – factory methods
# ═══════════════════════════════════════════════════════════════════


class TestContextPartitionFactory:
    def test_from_metadata_key(self):
        cs = ContextSpace([
            _ctx("a", group="ctrl"),
            _ctx("b", group="ctrl"),
            _ctx("c", group="treat"),
        ])
        cp = ContextPartition.from_metadata_key(cs, "group")
        assert cp.num_groups == 2
        assert set(cp.group("ctrl")) == {"a", "b"}
        assert cp.group("treat") == ["c"]

    def test_from_ordering_splits(self):
        cs = _ordered_space()  # ordering: 0.0, 1.0, 2.0
        cp = ContextPartition.from_ordering_splits(cs, [1.5])
        assert cp.num_groups == 2
        # segment_0: [−∞, 1.5) → a(0.0), b(1.0)
        # segment_1: [1.5, +∞) → c(2.0)
        assert set(cp.group("segment_0")) == {"a", "b"}
        assert cp.group("segment_1") == ["c"]

    def test_from_predicate(self):
        cs = _ordered_space()
        cp = ContextPartition.from_predicate(
            cs, lambda c: "early" if (c.ordering_value or 0) < 1.5 else "late"
        )
        assert set(cp.group("early")) == {"a", "b"}
        assert cp.group("late") == ["c"]


# ═══════════════════════════════════════════════════════════════════
# ContextPartition – pairwise operations
# ═══════════════════════════════════════════════════════════════════


class TestContextPartitionPairs:
    def test_inter_group_pairs(self):
        cs = _ordered_space()
        cp = ContextPartition({"a": ["a"], "b": ["b"], "c": ["c"]}, cs)
        pairs = cp.inter_group_pairs()
        assert len(pairs) == 3  # C(3,2)

    def test_cross_group_context_pairs(self):
        cs = _ordered_space()
        cp = ContextPartition({"lo": ["a"], "hi": ["b", "c"]}, cs)
        cross = cp.cross_group_context_pairs("lo", "hi")
        assert set(cross) == {("a", "b"), ("a", "c")}


# ═══════════════════════════════════════════════════════════════════
# ContextPartition – serialization round-trip
# ═══════════════════════════════════════════════════════════════════


class TestContextPartitionSerialization:
    def test_round_trip(self):
        cs = _ordered_space()
        cp = ContextPartition({"lo": ["a"], "hi": ["b", "c"]}, cs)
        d = cp.to_dict()
        cp2 = ContextPartition.from_dict(d, cs)
        assert cp2.group_names == cp.group_names
        assert cp2.group("lo") == cp.group("lo")
        assert set(cp2.group("hi")) == set(cp.group("hi"))
