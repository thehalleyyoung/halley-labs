"""Tests for usability_oracle.accessibility.spatial — spatial layout analysis.

Covers SpatialAnalyzer.compute_layout(), detect_groups(), visual density,
spacing regularity, nearest neighbours, Fitts' law helpers, reading order,
overlap detection, and alignment scoring.  Also covers LayoutInfo and
NodeGroup dataclasses.
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.spatial import (
    LayoutInfo,
    NodeGroup,
    SpatialAnalyzer,
)
from tests.fixtures.sample_trees import (
    make_simple_form_tree,
    make_navigation_tree,
    make_dashboard_tree,
)


def _make_state(**kw) -> AccessibilityState:
    return AccessibilityState(**kw)


def _make_node(
    id: str,
    role: str = "button",
    name: str = "",
    x: float = 0,
    y: float = 0,
    w: float = 50,
    h: float = 30,
    parent_id: str | None = None,
) -> AccessibilityNode:
    """Helper to build a positioned node."""
    return AccessibilityNode(
        id=id, role=role, name=name,
        bounding_box=BoundingBox(x, y, w, h),
        state=_make_state(), children=[],
        parent_id=parent_id, depth=1, index_in_parent=0,
    )


def _make_positioned_tree(
    nodes: list[AccessibilityNode],
) -> AccessibilityTree:
    """Wrap a list of nodes under a root and return a tree."""
    root = AccessibilityNode(
        id="root", role="document", name="Page",
        bounding_box=BoundingBox(0, 0, 1920, 1080),
        state=_make_state(), children=nodes,
        parent_id=None, depth=0, index_in_parent=0,
    )
    for i, n in enumerate(nodes):
        n.parent_id = "root"
        n.index_in_parent = i
    return AccessibilityTree(root=root)


@pytest.fixture
def analyzer() -> SpatialAnalyzer:
    """Return a default SpatialAnalyzer."""
    return SpatialAnalyzer()


# ── LayoutInfo dataclass ──────────────────────────────────────────────────────


class TestLayoutInfo:
    """Tests for the LayoutInfo dataclass."""

    def test_default_values(self):
        """LayoutInfo() should have sensible defaults."""
        info = LayoutInfo()
        assert info.total_nodes == 0
        assert info.visible_nodes == 0
        assert info.viewport is None
        assert info.visual_density == 0.0
        assert info.spacing_regularity == 0.0
        assert info.mean_element_area == 0.0
        assert info.reading_order == []
        assert info.groups == []

    def test_to_dict(self):
        """to_dict should serialise all fields."""
        info = LayoutInfo(total_nodes=10, visible_nodes=8, visual_density=0.5)
        d = info.to_dict()
        assert d["total_nodes"] == 10
        assert d["visible_nodes"] == 8
        assert d["visual_density"] == 0.5


# ── NodeGroup dataclass ──────────────────────────────────────────────────────


class TestNodeGroup:
    """Tests for the NodeGroup dataclass."""

    def test_to_dict(self):
        """to_dict should serialise all fields including nested BoundingBox."""
        g = NodeGroup(
            group_id=1, node_ids=["a", "b"],
            bounding_box=BoundingBox(0, 0, 100, 100),
            centroid_x=50, centroid_y=50, density=0.4,
        )
        d = g.to_dict()
        assert d["group_id"] == 1
        assert d["node_ids"] == ["a", "b"]
        assert "bounding_box" in d
        assert d["density"] == 0.4


# ── compute_layout ────────────────────────────────────────────────────────────


class TestComputeLayout:
    """Tests for SpatialAnalyzer.compute_layout()."""

    def test_returns_layout_info(self, analyzer):
        """compute_layout should return a LayoutInfo instance."""
        tree = make_simple_form_tree()
        info = analyzer.compute_layout(tree)
        assert isinstance(info, LayoutInfo)

    def test_total_nodes_matches_tree_size(self, analyzer):
        """total_nodes should equal the tree size."""
        tree = make_simple_form_tree()
        info = analyzer.compute_layout(tree)
        assert info.total_nodes == tree.size()

    def test_visible_nodes_positive(self, analyzer):
        """visible_nodes should be > 0 for a tree with visible nodes."""
        tree = make_simple_form_tree()
        info = analyzer.compute_layout(tree)
        assert info.visible_nodes > 0

    def test_viewport_computed(self, analyzer):
        """The viewport should be computed from node extents."""
        tree = make_simple_form_tree()
        info = analyzer.compute_layout(tree)
        assert info.viewport is not None
        assert info.viewport.width > 0

    def test_reading_order_non_empty(self, analyzer):
        """reading_order should contain node IDs."""
        tree = make_simple_form_tree()
        info = analyzer.compute_layout(tree)
        assert len(info.reading_order) > 0

    def test_dashboard_layout(self, analyzer):
        """compute_layout on the dashboard should work with many nodes."""
        tree = make_dashboard_tree()
        info = analyzer.compute_layout(tree)
        assert info.total_nodes == tree.size()
        assert info.visual_density >= 0.0

    def test_navigation_layout(self, analyzer):
        """compute_layout on navigation tree should have visible nodes."""
        tree = make_navigation_tree()
        info = analyzer.compute_layout(tree)
        assert info.visible_nodes > 0


# ── detect_groups ─────────────────────────────────────────────────────────────


class TestDetectGroups:
    """Tests for SpatialAnalyzer.detect_groups()."""

    def test_close_nodes_grouped(self, analyzer):
        """Nodes near each other should be grouped together."""
        nodes = [
            _make_node("a", x=0, y=0),
            _make_node("b", x=10, y=0),
            _make_node("c", x=500, y=500),
        ]
        groups = analyzer.detect_groups(nodes, distance_threshold=100)
        assert len(groups) >= 2  # at least two clusters

    def test_single_node_no_groups(self, analyzer):
        """A single node should produce no groups."""
        nodes = [_make_node("a")]
        groups = analyzer.detect_groups(nodes)
        assert groups == []

    def test_groups_have_node_ids(self, analyzer):
        """Each group should list the node IDs it contains."""
        nodes = [_make_node("a", x=0, y=0), _make_node("b", x=5, y=0)]
        groups = analyzer.detect_groups(nodes, distance_threshold=200)
        all_ids = []
        for g in groups:
            all_ids.extend(g.node_ids)
        assert "a" in all_ids
        assert "b" in all_ids

    def test_groups_have_bounding_box(self, analyzer):
        """Each group should have a bounding box."""
        nodes = [_make_node("a", x=0, y=0), _make_node("b", x=10, y=0)]
        groups = analyzer.detect_groups(nodes, distance_threshold=200)
        for g in groups:
            assert g.bounding_box is not None


# ── compute_visual_density ────────────────────────────────────────────────────


class TestVisualDensity:
    """Tests for SpatialAnalyzer.compute_visual_density()."""

    def test_non_negative(self, analyzer):
        """Visual density should be non-negative."""
        region = BoundingBox(0, 0, 100, 100)
        nodes = [_make_node("a", x=10, y=10, w=20, h=20)]
        density = analyzer.compute_visual_density(region, nodes)
        assert density >= 0.0

    def test_full_coverage_density_one(self, analyzer):
        """A single node covering the entire region should have density 1.0."""
        region = BoundingBox(0, 0, 100, 100)
        nodes = [_make_node("a", x=0, y=0, w=100, h=100)]
        density = analyzer.compute_visual_density(region, nodes)
        assert abs(density - 1.0) < 1e-6

    def test_empty_nodes_zero_density(self, analyzer):
        """No nodes should produce zero density."""
        region = BoundingBox(0, 0, 100, 100)
        density = analyzer.compute_visual_density(region, [])
        assert density == 0.0

    def test_zero_area_region(self, analyzer):
        """A zero-area region should return 0.0 density."""
        region = BoundingBox(0, 0, 0, 100)
        nodes = [_make_node("a", x=0, y=0)]
        density = analyzer.compute_visual_density(region, nodes)
        assert density == 0.0


# ── compute_spacing_regularity ────────────────────────────────────────────────


class TestSpacingRegularity:
    """Tests for SpatialAnalyzer.compute_spacing_regularity()."""

    def test_regular_spacing_high(self, analyzer):
        """Evenly spaced nodes should produce regularity close to 1.0."""
        nodes = [_make_node(f"n{i}", x=i * 100, y=0) for i in range(5)]
        reg = analyzer.compute_spacing_regularity(nodes)
        assert 0.0 <= reg <= 1.0
        assert reg > 0.7  # should be high for regular spacing

    def test_single_node_returns_one(self, analyzer):
        """A single node should have perfect regularity (1.0)."""
        nodes = [_make_node("a")]
        reg = analyzer.compute_spacing_regularity(nodes)
        assert reg == 1.0

    def test_in_range(self, analyzer):
        """Spacing regularity should always be in [0, 1]."""
        nodes = [_make_node(f"n{i}", x=i * 50 + (i % 2) * 200, y=i * 30) for i in range(8)]
        reg = analyzer.compute_spacing_regularity(nodes)
        assert 0.0 <= reg <= 1.0


# ── nearest_neighbors ─────────────────────────────────────────────────────────


class TestNearestNeighbors:
    """Tests for SpatialAnalyzer.nearest_neighbors()."""

    def test_returns_sorted_by_distance(self, analyzer):
        """Results should be sorted by ascending distance."""
        target = _make_node("t", x=0, y=0)
        candidates = [
            _make_node("far", x=500, y=500),
            _make_node("close", x=10, y=0),
            _make_node("mid", x=100, y=0),
        ]
        result = analyzer.nearest_neighbors(target, candidates, k=3)
        distances = [d for _, d in result]
        assert distances == sorted(distances)

    def test_respects_k(self, analyzer):
        """Should return at most k neighbors."""
        target = _make_node("t", x=0, y=0)
        candidates = [_make_node(f"c{i}", x=i * 10, y=0) for i in range(1, 20)]
        result = analyzer.nearest_neighbors(target, candidates, k=3)
        assert len(result) == 3

    def test_excludes_self(self, analyzer):
        """The target node should not appear in its own neighbors."""
        target = _make_node("t", x=0, y=0)
        candidates = [target, _make_node("other", x=10, y=0)]
        result = analyzer.nearest_neighbors(target, candidates, k=5)
        for node, _ in result:
            assert node.id != "t"

    def test_no_bbox_returns_empty(self, analyzer):
        """A target without a bounding box should return empty list."""
        target = AccessibilityNode(
            id="nobbox", role="button", name="X",
            state=_make_state(), children=[],
        )
        candidates = [_make_node("c", x=10, y=0)]
        result = analyzer.nearest_neighbors(target, candidates)
        assert result == []


# ── Fitts' law helpers ────────────────────────────────────────────────────────


class TestFittsLaw:
    """Tests for Fitts' law static methods and fitts_id()."""

    def test_fitts_distance(self):
        """fitts_distance should return center-to-center Euclidean distance."""
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(30, 40, 10, 10)
        d = SpatialAnalyzer.fitts_distance(a, b)
        assert abs(d - 50.0) < 1e-6

    def test_fitts_width_zero_angle(self):
        """At approach angle 0, fitts_width should equal target width."""
        target = BoundingBox(0, 0, 100, 50)
        w = SpatialAnalyzer.fitts_width(target, 0.0)
        assert abs(w - 100.0) < 1e-6

    def test_fitts_width_pi_half(self):
        """At approach angle π/2, fitts_width should equal target height."""
        target = BoundingBox(0, 0, 100, 50)
        w = SpatialAnalyzer.fitts_width(target, math.pi / 2)
        assert abs(w - 50.0) < 1e-3

    def test_fitts_index_of_difficulty_same_position(self):
        """Zero distance should give ID = log2(0/W + 1) = 0."""
        iod = SpatialAnalyzer.fitts_index_of_difficulty(0.0, 50.0)
        assert abs(iod - 0.0) < 1e-6

    def test_fitts_index_of_difficulty_positive(self):
        """Non-zero distance should give positive ID."""
        iod = SpatialAnalyzer.fitts_index_of_difficulty(100.0, 50.0)
        assert iod > 0.0

    def test_fitts_index_of_difficulty_zero_width(self):
        """Zero target width should return infinity."""
        iod = SpatialAnalyzer.fitts_index_of_difficulty(100.0, 0.0)
        assert iod == float("inf")

    def test_fitts_id_combined(self, analyzer):
        """fitts_id should combine distance and effective width into a single ID."""
        src = BoundingBox(0, 0, 20, 20)
        tgt = BoundingBox(200, 0, 40, 40)
        fid = analyzer.fitts_id(src, tgt)
        assert fid > 0.0

    def test_fitts_id_larger_target_easier(self, analyzer):
        """A larger target at same distance should have lower Fitts' ID."""
        src = BoundingBox(0, 0, 10, 10)
        small_target = BoundingBox(200, 0, 10, 10)
        large_target = BoundingBox(200, 0, 100, 100)
        id_small = analyzer.fitts_id(src, small_target)
        id_large = analyzer.fitts_id(src, large_target)
        assert id_large < id_small


# ── compute_reading_order ─────────────────────────────────────────────────────


class TestReadingOrder:
    """Tests for SpatialAnalyzer.compute_reading_order()."""

    def test_ltr_reading_order(self, analyzer):
        """LTR order: top nodes first, then left-to-right within same line."""
        nodes = [
            _make_node("top_right", x=200, y=0),
            _make_node("top_left", x=0, y=0),
            _make_node("bottom", x=0, y=100),
        ]
        order = analyzer.compute_reading_order(nodes)
        ids = [n.id for n in order]
        assert ids.index("top_left") < ids.index("bottom")
        assert ids.index("top_left") < ids.index("top_right")

    def test_rtl_reading_order(self, analyzer):
        """RTL order: rightmost should come first within same line."""
        nodes = [
            _make_node("left", x=0, y=0),
            _make_node("right", x=200, y=0),
        ]
        order = analyzer.compute_reading_order(nodes, rtl=True)
        ids = [n.id for n in order]
        assert ids.index("right") < ids.index("left")

    def test_empty_list(self, analyzer):
        """Empty input should return empty output."""
        order = analyzer.compute_reading_order([])
        assert order == []

    def test_preserves_all_nodes(self, analyzer):
        """All input nodes should appear in the output."""
        nodes = [_make_node(f"n{i}", x=i * 50, y=0) for i in range(5)]
        order = analyzer.compute_reading_order(nodes)
        assert len(order) == 5


# ── find_overlapping_pairs ────────────────────────────────────────────────────


class TestOverlappingPairs:
    """Tests for SpatialAnalyzer.find_overlapping_pairs()."""

    def test_overlapping_detected(self, analyzer):
        """Two overlapping nodes should produce a pair with positive area."""
        nodes = [
            _make_node("a", x=0, y=0, w=100, h=100),
            _make_node("b", x=50, y=50, w=100, h=100),
        ]
        pairs = analyzer.find_overlapping_pairs(nodes)
        assert len(pairs) == 1
        assert pairs[0][2] > 0  # overlap area

    def test_no_overlap(self, analyzer):
        """Non-overlapping nodes should produce no pairs."""
        nodes = [
            _make_node("a", x=0, y=0, w=50, h=50),
            _make_node("b", x=200, y=200, w=50, h=50),
        ]
        pairs = analyzer.find_overlapping_pairs(nodes)
        assert len(pairs) == 0

    def test_multiple_overlaps(self, analyzer):
        """Three mutually overlapping nodes should produce 3 pairs."""
        nodes = [
            _make_node("a", x=0, y=0, w=100, h=100),
            _make_node("b", x=50, y=0, w=100, h=100),
            _make_node("c", x=25, y=50, w=100, h=100),
        ]
        pairs = analyzer.find_overlapping_pairs(nodes)
        assert len(pairs) == 3


# ── compute_alignment_score ───────────────────────────────────────────────────


class TestAlignmentScore:
    """Tests for SpatialAnalyzer.compute_alignment_score()."""

    def test_perfectly_aligned(self, analyzer):
        """Nodes sharing the same x coordinate should score high."""
        nodes = [_make_node(f"n{i}", x=0, y=i * 50) for i in range(5)]
        score = analyzer.compute_alignment_score(nodes)
        assert 0.0 <= score <= 1.0
        assert score >= 0.5  # should be well-aligned

    def test_single_node_perfect(self, analyzer):
        """A single node should get perfect alignment score."""
        nodes = [_make_node("a")]
        score = analyzer.compute_alignment_score(nodes)
        assert score == 1.0

    def test_scattered_nodes_lower(self, analyzer):
        """Randomly positioned nodes should score lower than aligned nodes."""
        aligned = [_make_node(f"a{i}", x=0, y=i * 50) for i in range(5)]
        scattered = [_make_node(f"s{i}", x=i * 73, y=i * 41) for i in range(5)]
        score_aligned = analyzer.compute_alignment_score(aligned)
        score_scattered = analyzer.compute_alignment_score(scattered)
        assert score_aligned >= score_scattered

    def test_in_range(self, analyzer):
        """Alignment score should always be in [0, 1]."""
        nodes = [_make_node(f"n{i}", x=i * 37, y=i * 53) for i in range(10)]
        score = analyzer.compute_alignment_score(nodes)
        assert 0.0 <= score <= 1.0
