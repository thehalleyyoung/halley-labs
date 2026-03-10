"""Tests for usability_oracle.accessibility.normalizer — tree normalisation.

Covers the AccessibilityNormalizer.normalize() pipeline including role
normalisation (platform-specific mappings), name normalisation (whitespace
cleaning), bounding box scaling, decorative node removal, and wrapper
collapsing.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.normalizer import (
    AccessibilityNormalizer,
    ROLE_MAPPINGS,
)
from tests.fixtures.sample_trees import (
    make_simple_form_tree,
    make_navigation_tree,
    make_dashboard_tree,
    make_modal_dialog_tree,
)


def _make_state(**overrides) -> AccessibilityState:
    """Helper to build AccessibilityState with overrides."""
    return AccessibilityState(**overrides)


def _build_tree_with_platform_roles() -> AccessibilityTree:
    """Build a small tree with macOS platform-specific roles."""
    child = AccessibilityNode(
        id="btn1", role="AXButton", name="Save",
        bounding_box=BoundingBox(10, 10, 80, 30),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="AXWebArea", name="Page",
        bounding_box=BoundingBox(0, 0, 800, 600),
        state=_make_state(), children=[child], parent_id=None,
        depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "macos"})


def _build_tree_with_decorative_nodes() -> AccessibilityTree:
    """Build a tree containing decorative (presentation/none/generic) leaf nodes."""
    decorative = AccessibilityNode(
        id="deco1", role="presentation", name="",
        bounding_box=BoundingBox(0, 0, 10, 10),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=0,
    )
    generic_empty = AccessibilityNode(
        id="gen1", role="generic", name="",
        bounding_box=BoundingBox(20, 20, 10, 10),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=1,
    )
    named_generic = AccessibilityNode(
        id="gen2", role="generic", name="Important Container",
        bounding_box=BoundingBox(40, 40, 10, 10),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=2,
    )
    button = AccessibilityNode(
        id="btn1", role="button", name="OK",
        bounding_box=BoundingBox(60, 60, 80, 30),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=3,
    )
    root = AccessibilityNode(
        id="root", role="document", name="Page",
        bounding_box=BoundingBox(0, 0, 400, 300),
        state=_make_state(), children=[decorative, generic_empty, named_generic, button],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def _build_tree_with_wrappers() -> AccessibilityTree:
    """Build a tree with single-child generic wrapper nodes."""
    button = AccessibilityNode(
        id="btn1", role="button", name="Submit",
        bounding_box=BoundingBox(10, 10, 80, 30),
        state=_make_state(), children=[], parent_id="wrapper1",
        depth=3, index_in_parent=0,
    )
    wrapper = AccessibilityNode(
        id="wrapper1", role="generic", name="",
        bounding_box=BoundingBox(5, 5, 90, 40),
        state=_make_state(), children=[button], parent_id="root",
        depth=2, index_in_parent=0,
    )
    other = AccessibilityNode(
        id="text1", role="heading", name="Title",
        bounding_box=BoundingBox(10, 60, 200, 30),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=1,
    )
    root = AccessibilityNode(
        id="root", role="document", name="Page",
        bounding_box=BoundingBox(0, 0, 400, 300),
        state=_make_state(), children=[wrapper, other],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def _build_tree_with_messy_names() -> AccessibilityTree:
    """Build a tree with inconsistent whitespace in names."""
    child = AccessibilityNode(
        id="btn1", role="button", name="  Click   Here  ",
        description="  Do   the   thing  ",
        bounding_box=BoundingBox(10, 10, 80, 30),
        state=_make_state(), children=[], parent_id="root",
        depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="document", name="  My   Page  ",
        bounding_box=BoundingBox(0, 0, 800, 600),
        state=_make_state(), children=[child],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


@pytest.fixture
def normalizer() -> AccessibilityNormalizer:
    """Return a default normalizer."""
    return AccessibilityNormalizer()


# ── Normalise preserves structure ─────────────────────────────────────────────


class TestNormalizeStructure:
    """Tests that normalize() preserves overall tree structure."""

    def test_normalize_returns_new_tree(self, normalizer):
        """normalize() should return a new tree, not mutate the original."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        assert result is not tree

    def test_normalize_preserves_size_for_clean_tree(self):
        """For a tree with no decorative nodes, size should stay the same."""
        normalizer = AccessibilityNormalizer(
            remove_decorative=False, collapse_wrappers=False,
        )
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        assert result.size() == tree.size()

    def test_normalize_preserves_root_role(self, normalizer):
        """The root role should remain 'document' for sample trees."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        assert result.root.role == "document"

    def test_normalized_metadata_flag(self, normalizer):
        """The normalised tree should have metadata['normalized'] = True."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        assert result.metadata.get("normalized") is True

    def test_normalize_navigation_tree(self, normalizer):
        """Normalising the navigation tree should not crash."""
        tree = make_navigation_tree()
        result = normalizer.normalize(tree)
        assert result.root is not None

    def test_normalize_dashboard_tree(self, normalizer):
        """Normalising the dashboard tree should produce a valid tree."""
        tree = make_dashboard_tree()
        result = normalizer.normalize(tree)
        assert result.validate() == []

    def test_normalize_modal_dialog_tree(self, normalizer):
        """Normalising the modal dialog tree should produce a valid tree."""
        tree = make_modal_dialog_tree()
        result = normalizer.normalize(tree)
        assert result.validate() == []


# ── Role normalisation ────────────────────────────────────────────────────────


class TestRoleNormalization:
    """Tests for platform-specific role mapping to canonical ARIA roles."""

    def test_macos_button_mapped(self, normalizer):
        """AXButton should be mapped to 'button'."""
        tree = _build_tree_with_platform_roles()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn1")
        assert btn is not None
        assert btn.role == "button"

    def test_macos_web_area_mapped(self, normalizer):
        """AXWebArea should be mapped to 'document'."""
        tree = _build_tree_with_platform_roles()
        result = normalizer.normalize(tree)
        assert result.root.role == "document"

    def test_original_role_preserved_in_properties(self, normalizer):
        """After mapping, the original role should be stored in properties."""
        tree = _build_tree_with_platform_roles()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn1")
        assert btn is not None
        assert btn.properties.get("original_role") == "AXButton"

    def test_standard_roles_unchanged(self, normalizer):
        """Standard ARIA roles should not be changed by normalisation."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn_submit")
        assert btn is not None
        assert btn.role == "button"
        assert "original_role" not in btn.properties

    def test_role_mappings_cover_platforms(self):
        """ROLE_MAPPINGS should include macOS, Windows UIA, and Linux ATK entries."""
        assert "AXButton" in ROLE_MAPPINGS
        assert "UIA.Button" in ROLE_MAPPINGS
        assert "ATK_ROLE_PUSH_BUTTON" in ROLE_MAPPINGS


# ── Name normalisation ────────────────────────────────────────────────────────


class TestNameNormalization:
    """Tests for whitespace and unicode cleaning of names."""

    def test_whitespace_collapsed(self, normalizer):
        """Extra whitespace in names should be collapsed to single spaces."""
        tree = _build_tree_with_messy_names()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn1")
        assert btn is not None
        assert btn.name == "Click Here"

    def test_leading_trailing_stripped(self, normalizer):
        """Leading and trailing whitespace should be stripped."""
        tree = _build_tree_with_messy_names()
        result = normalizer.normalize(tree)
        assert result.root.name == "My Page"

    def test_description_cleaned(self, normalizer):
        """Description should also have whitespace normalised."""
        tree = _build_tree_with_messy_names()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn1")
        assert btn is not None
        assert btn.description == "Do the thing"

    def test_empty_name_stays_empty(self, normalizer):
        """An empty name should remain empty after cleaning."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        # root description is empty in the fixture
        root = result.root
        assert root.description == "" or root.description == root.description.strip()


# ── Bounding box normalisation ────────────────────────────────────────────────


class TestBoundingBoxNormalization:
    """Tests for scaling bounding boxes to the target viewport."""

    def test_bboxes_scaled_to_viewport(self):
        """After normalisation, bounding boxes should fit within target viewport."""
        target = BoundingBox(0, 0, 1920, 1080)
        normalizer = AccessibilityNormalizer(target_viewport=target)
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        for node in result.node_index.values():
            if node.bounding_box is not None:
                bb = node.bounding_box
                assert bb.x >= target.x - 1  # tolerance for float
                assert bb.y >= target.y - 1

    def test_bbox_normalization_disabled(self):
        """With normalize_coordinates=False, bboxes should remain unchanged."""
        normalizer = AccessibilityNormalizer(
            normalize_coordinates=False,
            remove_decorative=False,
            collapse_wrappers=False,
        )
        tree = make_simple_form_tree()
        original_root_bb = deepcopy(tree.root.bounding_box)
        result = normalizer.normalize(tree)
        assert result.root.bounding_box.x == original_root_bb.x
        assert result.root.bounding_box.width == original_root_bb.width


# ── Remove decorative ─────────────────────────────────────────────────────────


class TestRemoveDecorative:
    """Tests for removing presentation/none/generic decorative leaf nodes."""

    def test_decorative_leaves_removed(self, normalizer):
        """Decorative leaf nodes (no name, decorative role) should be removed."""
        tree = _build_tree_with_decorative_nodes()
        result = normalizer.normalize(tree)
        assert result.get_node("deco1") is None
        assert result.get_node("gen1") is None

    def test_named_generic_preserved(self, normalizer):
        """Generic nodes with a name should be preserved."""
        tree = _build_tree_with_decorative_nodes()
        result = normalizer.normalize(tree)
        gen2 = result.get_node("gen2")
        assert gen2 is not None

    def test_non_decorative_preserved(self, normalizer):
        """Non-decorative nodes like buttons should always be preserved."""
        tree = _build_tree_with_decorative_nodes()
        result = normalizer.normalize(tree)
        btn = result.get_node("btn1")
        assert btn is not None

    def test_remove_decorative_disabled(self):
        """With remove_decorative=False, all nodes should be kept."""
        normalizer = AccessibilityNormalizer(
            remove_decorative=False, collapse_wrappers=False,
        )
        tree = _build_tree_with_decorative_nodes()
        result = normalizer.normalize(tree)
        assert result.get_node("deco1") is not None


# ── Collapse wrappers ─────────────────────────────────────────────────────────


class TestCollapseWrappers:
    """Tests for collapsing single-child generic wrapper nodes."""

    def test_single_child_wrapper_collapsed(self, normalizer):
        """A generic wrapper with one child should be collapsed."""
        tree = _build_tree_with_wrappers()
        result = normalizer.normalize(tree)
        # The wrapper should be gone, button should be direct child of root
        assert result.get_node("wrapper1") is None
        btn = result.get_node("btn1")
        assert btn is not None

    def test_other_children_preserved(self, normalizer):
        """Non-wrapper children should remain intact."""
        tree = _build_tree_with_wrappers()
        result = normalizer.normalize(tree)
        heading = result.get_node("text1")
        assert heading is not None
        assert heading.role == "heading"

    def test_collapse_disabled(self):
        """With collapse_wrappers=False, wrappers should remain."""
        normalizer = AccessibilityNormalizer(
            collapse_wrappers=False, remove_decorative=False,
        )
        tree = _build_tree_with_wrappers()
        result = normalizer.normalize(tree)
        assert result.get_node("wrapper1") is not None

    def test_collapse_preserves_tree_validity(self, normalizer):
        """After collapsing, the tree should still validate."""
        tree = _build_tree_with_wrappers()
        result = normalizer.normalize(tree)
        assert result.validate() == []


# ── Semantic level assignment ─────────────────────────────────────────────────


class TestSemanticLevels:
    """Tests for semantic depth annotation after normalisation."""

    def test_semantic_level_assigned(self, normalizer):
        """Nodes should have 'semantic_level' in their properties."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        root = result.root
        assert "semantic_level" in root.properties

    def test_landmark_increases_depth(self, normalizer):
        """Landmark nodes like 'form' should increase semantic depth for children."""
        tree = make_simple_form_tree()
        result = normalizer.normalize(tree)
        root_level = result.root.properties["semantic_level"]
        form = result.get_node("form1")
        assert form is not None
        form_level = form.properties["semantic_level"]
        # Form is a landmark, so its children should get a higher level
        btn = result.get_node("btn_submit")
        assert btn is not None
        btn_level = btn.properties["semantic_level"]
        assert btn_level >= form_level
