"""Unit tests for :class:`usability_oracle.alignment.fuzzy_match.FuzzyMatcher`.

Validates fuzzy bipartite matching: role similarity, name similarity (via
Levenshtein distance), position similarity (Gaussian kernel on bounding-box
centres), structure similarity (Jaccard on child-role multisets), Hungarian
optimal assignment, ``already_matched`` exclusion, and threshold filtering.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentPass,
    BoundingBox,
    NodeMapping,
)
from usability_oracle.alignment.fuzzy_match import (
    FuzzyMatcher,
    _levenshtein,
    _normalised_levenshtein,
)


# ============================================================================
# Helpers
# ============================================================================


def _node(
    nid: str,
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str = "",
    bbox: BoundingBox | None = None,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    child_roles: list[str] | None = None,
) -> AccessibilityNode:
    """Create a minimal :class:`AccessibilityNode` for fuzzy-match tests."""
    props: dict = {}
    if child_roles is not None:
        props["child_roles"] = child_roles
    return AccessibilityNode(
        node_id=nid,
        role=role,
        name=name,
        bounding_box=bbox,
        parent_id=parent_id,
        children_ids=children_ids or [],
        properties=props,
    )


# ============================================================================
# Tests — Levenshtein helpers
# ============================================================================


class TestLevenshteinHelpers:
    """Validate the pure-Python Levenshtein distance functions."""

    def test_identical_strings(self) -> None:
        """Identical strings have distance 0."""
        assert _levenshtein("hello", "hello") == 0

    def test_empty_vs_nonempty(self) -> None:
        """Distance from empty string equals length of the other."""
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("xyz", "") == 3

    def test_both_empty(self) -> None:
        """Two empty strings → distance 0."""
        assert _levenshtein("", "") == 0

    def test_known_distance(self) -> None:
        """Kitten → sitting has known Levenshtein distance 3."""
        assert _levenshtein("kitten", "sitting") == 3

    def test_normalised_identical(self) -> None:
        """Normalised similarity of identical strings is 1.0."""
        assert _normalised_levenshtein("test", "test") == 1.0

    def test_normalised_completely_different(self) -> None:
        """Totally disjoint strings of equal length → similarity 0.0."""
        assert _normalised_levenshtein("abc", "xyz") == pytest.approx(0.0)

    def test_normalised_range(self) -> None:
        """Similarity is always in [0, 1]."""
        sim = _normalised_levenshtein("Submit", "Cancel")
        assert 0.0 <= sim <= 1.0


# ============================================================================
# Tests — role similarity
# ============================================================================


class TestRoleSimilarity:
    """FuzzyMatcher._role_similarity delegates to role_taxonomy_distance."""

    def test_same_role_similarity_one(self) -> None:
        """Identical roles produce similarity 1.0."""
        assert FuzzyMatcher._role_similarity(AccessibilityRole.BUTTON, AccessibilityRole.BUTTON) == 1.0

    def test_same_group_similarity(self) -> None:
        """Roles in the same taxonomy group yield similarity 0.7."""
        sim = FuzzyMatcher._role_similarity(AccessibilityRole.BUTTON, AccessibilityRole.LINK)
        assert sim == pytest.approx(0.7)

    def test_different_group_similarity(self) -> None:
        """Roles in different groups yield similarity 0.3."""
        sim = FuzzyMatcher._role_similarity(AccessibilityRole.BUTTON, AccessibilityRole.HEADING)
        assert sim == pytest.approx(0.3)

    def test_generic_role_similarity_zero(self) -> None:
        """GENERIC / NONE / UNKNOWN roles yield similarity 0.0."""
        sim = FuzzyMatcher._role_similarity(AccessibilityRole.BUTTON, AccessibilityRole.GENERIC)
        assert sim == pytest.approx(0.0)


# ============================================================================
# Tests — name similarity
# ============================================================================


class TestNameSimilarity:
    """FuzzyMatcher._name_similarity is case-insensitive Levenshtein."""

    def test_identical_names(self) -> None:
        """Same names → 1.0."""
        assert FuzzyMatcher._name_similarity("Submit", "Submit") == 1.0

    def test_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        assert FuzzyMatcher._name_similarity("Submit", "submit") == 1.0

    def test_empty_both(self) -> None:
        """Two empty names → 1.0."""
        assert FuzzyMatcher._name_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        """One empty name → 0.0."""
        assert FuzzyMatcher._name_similarity("Hello", "") == 0.0

    def test_partial_match(self) -> None:
        """Similar names produce an intermediate similarity."""
        sim = FuzzyMatcher._name_similarity("Submit Order", "Submit Form")
        assert 0.0 < sim < 1.0


# ============================================================================
# Tests — position similarity
# ============================================================================


class TestPositionSimilarity:
    """Gaussian-kernel similarity on bounding-box centres."""

    def test_identical_bbox(self) -> None:
        """Same bounding box → similarity 1.0."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        matcher = FuzzyMatcher()
        assert matcher._position_similarity(bbox, bbox) == pytest.approx(1.0)

    def test_none_bbox_returns_half(self) -> None:
        """Missing bounding box → neutral 0.5."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        matcher = FuzzyMatcher()
        assert matcher._position_similarity(None, bbox) == 0.5
        assert matcher._position_similarity(bbox, None) == 0.5
        assert matcher._position_similarity(None, None) == 0.5

    def test_distant_boxes_low_similarity(self) -> None:
        """Very distant bounding boxes produce similarity near 0."""
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=9000, y=9000, width=10, height=10)
        matcher = FuzzyMatcher()
        sim = matcher._position_similarity(a, b)
        assert sim < 0.01

    def test_sigma_controls_falloff(self) -> None:
        """A larger position_sigma makes distant boxes more similar."""
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=100, y=0, width=10, height=10)
        narrow = FuzzyMatcher(AlignmentConfig(position_sigma=10.0))
        wide = FuzzyMatcher(AlignmentConfig(position_sigma=500.0))
        assert narrow._position_similarity(a, b) < wide._position_similarity(a, b)


# ============================================================================
# Tests — structure similarity
# ============================================================================


class TestStructureSimilarity:
    """Jaccard index on child-role multisets."""

    def test_both_leaves(self) -> None:
        """Two leaves (no children) → similarity 1.0."""
        sim = FuzzyMatcher._structure_similarity_from_multisets(Counter(), Counter())
        assert sim == 1.0

    def test_one_leaf_one_parent(self) -> None:
        """One leaf, one parent → 0.0."""
        a = Counter({"button": 2})
        b: Counter[str] = Counter()
        assert FuzzyMatcher._structure_similarity_from_multisets(a, b) == 0.0

    def test_identical_children(self) -> None:
        """Identical child-role multisets → 1.0."""
        c = Counter({"button": 2, "link": 1})
        assert FuzzyMatcher._structure_similarity_from_multisets(c, c) == 1.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping multisets → intermediate value."""
        a = Counter({"button": 2, "link": 1})
        b = Counter({"button": 1, "link": 1, "textbox": 1})
        sim = FuzzyMatcher._structure_similarity_from_multisets(a, b)
        assert 0.0 < sim < 1.0


# ============================================================================
# Tests — Hungarian assignment
# ============================================================================


class TestHungarianAssignment:
    """FuzzyMatcher._solve_assignment wraps scipy's linear_sum_assignment."""

    def test_optimal_pairing(self) -> None:
        """The solver picks the assignment that maximises total similarity."""
        import numpy as np

        sim = np.array([
            [0.9, 0.1],
            [0.1, 0.8],
        ])
        pairs = FuzzyMatcher._solve_assignment(sim)
        # Optimal: (0,0) + (1,1) = 1.7  vs  (0,1) + (1,0) = 0.2
        assert (0, 0) in pairs
        assert (1, 1) in pairs

    def test_rectangular_matrix(self) -> None:
        """Assignment works with non-square similarity matrices."""
        import numpy as np

        sim = np.array([
            [0.9, 0.2, 0.1],
            [0.3, 0.8, 0.1],
        ])
        pairs = FuzzyMatcher._solve_assignment(sim)
        assert len(pairs) == 2  # min(rows, cols)


# ============================================================================
# Tests — match() integration
# ============================================================================


class TestFuzzyMatchIntegration:
    """End-to-end tests of FuzzyMatcher.match()."""

    def test_similar_nodes_matched(self) -> None:
        """Two nodes with high similarity (same role, similar name) are paired."""
        src = [_node("s1", AccessibilityRole.BUTTON, "Submit Order")]
        tgt = [_node("t1", AccessibilityRole.BUTTON, "Submit Form")]
        mappings = FuzzyMatcher().match(src, tgt)
        assert len(mappings) == 1
        assert mappings[0].source_id == "s1"
        assert mappings[0].target_id == "t1"
        assert mappings[0].pass_matched == AlignmentPass.FUZZY

    def test_confidence_reflects_similarity(self) -> None:
        """Confidence equals the computed weighted similarity score."""
        src = [_node("s1", AccessibilityRole.BUTTON, "Go")]
        tgt = [_node("t1", AccessibilityRole.BUTTON, "Go")]
        mappings = FuzzyMatcher().match(src, tgt)
        assert len(mappings) == 1
        assert mappings[0].confidence >= 0.8

    def test_already_matched_source_excluded(self) -> None:
        """Nodes in already_matched_source are excluded from matching."""
        src = [
            _node("s1", AccessibilityRole.BUTTON, "OK"),
            _node("s2", AccessibilityRole.BUTTON, "OK"),
        ]
        tgt = [_node("t1", AccessibilityRole.BUTTON, "OK")]
        mappings = FuzzyMatcher().match(src, tgt, already_matched_source={"s1"})
        assert all(m.source_id != "s1" for m in mappings)

    def test_already_matched_target_excluded(self) -> None:
        """Nodes in already_matched_target are excluded from matching."""
        src = [_node("s1", AccessibilityRole.BUTTON, "OK")]
        tgt = [
            _node("t1", AccessibilityRole.BUTTON, "OK"),
            _node("t2", AccessibilityRole.BUTTON, "OK"),
        ]
        mappings = FuzzyMatcher().match(src, tgt, already_matched_target={"t1"})
        assert all(m.target_id != "t1" for m in mappings)

    def test_threshold_filtering(self) -> None:
        """Low-similarity pairs below fuzzy_threshold are dropped."""
        cfg = AlignmentConfig(fuzzy_threshold=0.99)
        src = [_node("s1", AccessibilityRole.BUTTON, "Submit Order")]
        tgt = [_node("t1", AccessibilityRole.HEADING, "About Page")]
        mappings = FuzzyMatcher(cfg).match(src, tgt)
        assert mappings == []

    def test_low_threshold_admits_weak_matches(self) -> None:
        """A very low threshold lets even weak matches through."""
        cfg = AlignmentConfig(fuzzy_threshold=0.01)
        src = [_node("s1", AccessibilityRole.BUTTON, "Submit")]
        tgt = [_node("t1", AccessibilityRole.HEADING, "About")]
        mappings = FuzzyMatcher(cfg).match(src, tgt)
        assert len(mappings) == 1

    def test_empty_source_returns_empty(self) -> None:
        """No source nodes → no mappings."""
        tgt = [_node("t1", AccessibilityRole.BUTTON, "OK")]
        assert FuzzyMatcher().match([], tgt) == []

    def test_empty_target_returns_empty(self) -> None:
        """No target nodes → no mappings."""
        src = [_node("s1", AccessibilityRole.BUTTON, "OK")]
        assert FuzzyMatcher().match(src, []) == []

    def test_default_config_used(self) -> None:
        """Passing config=None uses the default AlignmentConfig."""
        matcher = FuzzyMatcher(config=None)
        assert matcher.config.fuzzy_threshold == pytest.approx(0.40)


# ============================================================================
# Tests — populate_child_roles helper
# ============================================================================


class TestPopulateChildRoles:
    """FuzzyMatcher.populate_child_roles populates the child_roles property."""

    def test_child_roles_populated(self) -> None:
        """After calling populate_child_roles, each node has a child_roles list."""
        root = AccessibilityNode(
            node_id="root",
            role=AccessibilityRole.NAVIGATION,
            name="Nav",
            children_ids=["a", "b"],
        )
        a = AccessibilityNode(node_id="a", role=AccessibilityRole.LINK, name="Link A", parent_id="root")
        b = AccessibilityNode(node_id="b", role=AccessibilityRole.BUTTON, name="Btn B", parent_id="root")
        tree = AccessibilityTree(nodes={"root": root, "a": a, "b": b}, root_ids=["root"])
        FuzzyMatcher.populate_child_roles(tree)
        assert "child_roles" in root.properties
        assert set(root.properties["child_roles"]) == {"link", "button"}

    def test_leaf_child_roles_empty(self) -> None:
        """Leaf nodes get an empty child_roles list."""
        leaf = AccessibilityNode(node_id="leaf", role=AccessibilityRole.BUTTON, name="OK")
        tree = AccessibilityTree(nodes={"leaf": leaf}, root_ids=["leaf"])
        FuzzyMatcher.populate_child_roles(tree)
        assert leaf.properties["child_roles"] == []


# ============================================================================
# Tests — compute_subtree_fingerprint
# ============================================================================


class TestSubtreeFingerprint:
    """FuzzyMatcher.compute_subtree_fingerprint returns a pre-order role string."""

    def test_single_node_fingerprint(self) -> None:
        """A single button's fingerprint is just 'button'."""
        node = AccessibilityNode(node_id="btn", role=AccessibilityRole.BUTTON, name="OK")
        tree = AccessibilityTree(nodes={"btn": node}, root_ids=["btn"])
        fp = FuzzyMatcher.compute_subtree_fingerprint(tree, "btn")
        assert fp == "button"

    def test_tree_fingerprint_preorder(self) -> None:
        """Fingerprint concatenates roles in pre-order with '/' separator."""
        root = AccessibilityNode(
            node_id="r", role=AccessibilityRole.LIST, name="", children_ids=["c1", "c2"]
        )
        c1 = AccessibilityNode(node_id="c1", role=AccessibilityRole.LIST_ITEM, name="", parent_id="r")
        c2 = AccessibilityNode(node_id="c2", role=AccessibilityRole.LIST_ITEM, name="", parent_id="r")
        tree = AccessibilityTree(nodes={"r": root, "c1": c1, "c2": c2}, root_ids=["r"])
        fp = FuzzyMatcher.compute_subtree_fingerprint(tree, "r")
        assert fp == "list/listitem/listitem"
