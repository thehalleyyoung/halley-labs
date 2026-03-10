"""Tests for usability_oracle.accessibility.models — core data model classes.

Covers BoundingBox geometry, AccessibilityState serialisation,
AccessibilityNode queries/traversal, and AccessibilityTree algorithms.
"""

from __future__ import annotations

import json

import pytest

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from tests.fixtures.sample_trees import (
    make_simple_form_tree,
    make_navigation_tree,
    make_dashboard_tree,
    make_modal_dialog_tree,
)


# ── BoundingBox ──────────────────────────────────────────────────────────────


class TestBoundingBox:
    """Tests for the BoundingBox dataclass."""

    def test_properties(self):
        """right, bottom, center_x, center_y, area should derive from x/y/w/h."""
        bb = BoundingBox(10, 20, 100, 50)
        assert bb.right == 110
        assert bb.bottom == 70
        assert bb.center_x == 60.0
        assert bb.center_y == 45.0
        assert bb.area == 5000.0

    def test_zero_area(self):
        """A zero-dimension box should have zero area."""
        bb = BoundingBox(5, 5, 0, 10)
        assert bb.area == 0.0

    def test_contains_point_inside(self):
        """A point inside the box should be contained."""
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.contains_point(50, 50)

    def test_contains_point_edge_and_outside(self):
        """Points on the boundary should be contained; outside should not."""
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.contains_point(0, 0)
        assert bb.contains_point(100, 100)
        assert not bb.contains_point(101, 50)

    def test_contains_box(self):
        """A larger box should contain a smaller box inside it."""
        outer = BoundingBox(0, 0, 200, 200)
        inner = BoundingBox(10, 10, 50, 50)
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_overlaps_and_no_overlap(self):
        """Overlapping boxes should be detected; non-overlapping should not."""
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 100, 100)
        assert a.overlaps(b) and b.overlaps(a)
        c = BoundingBox(0, 0, 50, 50)
        d = BoundingBox(100, 100, 50, 50)
        assert not c.overlaps(d)

    def test_intersection(self):
        """intersection() should return the overlapping region."""
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 100, 100)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.x == 50
        assert inter.y == 50
        assert inter.width == 50
        assert inter.height == 50

    def test_intersection_none(self):
        """Non-overlapping boxes should yield None intersection."""
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(100, 100, 10, 10)
        assert a.intersection(b) is None

    def test_union(self):
        """union() should encompass both boxes."""
        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(100, 100, 50, 50)
        u = a.union(b)
        assert u.x == 0
        assert u.y == 0
        assert u.right == 150
        assert u.bottom == 150

    def test_distance_to(self):
        """Center-to-center distance should be Euclidean."""
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(30, 40, 10, 10)
        dist = a.distance_to(b)
        # centers: (5,5) and (35,45) => sqrt(30^2 + 40^2) = 50
        assert abs(dist - 50.0) < 1e-6

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict should round-trip faithfully."""
        bb = BoundingBox(1.5, 2.5, 100.0, 200.0)
        d = bb.to_dict()
        restored = BoundingBox.from_dict(d)
        assert restored.x == bb.x
        assert restored.width == bb.width


# ── AccessibilityState ───────────────────────────────────────────────────────


class TestAccessibilityState:
    """Tests for the AccessibilityState dataclass."""

    def test_default_values(self):
        """Default state should be all-off / None for optional fields."""
        s = AccessibilityState()
        assert s.focused is False
        assert s.selected is False
        assert s.expanded is False
        assert s.checked is None
        assert s.disabled is False
        assert s.hidden is False
        assert s.required is False
        assert s.readonly is False
        assert s.pressed is None
        assert s.value is None

    def test_disabled_state(self):
        """Creating a disabled state should work via kwargs."""
        s = AccessibilityState(disabled=True)
        assert s.disabled is True
        assert s.focused is False

    def test_hidden_state(self):
        """Creating a hidden state should set hidden=True."""
        s = AccessibilityState(hidden=True)
        assert s.hidden is True

    def test_focused_state(self):
        """Focused state should be captured correctly."""
        s = AccessibilityState(focused=True)
        assert s.focused is True

    def test_to_dict_includes_optional_when_set(self):
        """to_dict should include checked/pressed/value only when not None."""
        s = AccessibilityState(checked=True, pressed=False, value="42")
        d = s.to_dict()
        assert d["checked"] is True
        assert d["pressed"] is False
        assert d["value"] == "42"

    def test_to_dict_excludes_optional_when_none(self):
        """to_dict should omit checked/pressed/value when None."""
        s = AccessibilityState()
        d = s.to_dict()
        assert "checked" not in d
        assert "pressed" not in d
        assert "value" not in d

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(state)) should reproduce the original."""
        s = AccessibilityState(focused=True, required=True, value="hello")
        restored = AccessibilityState.from_dict(s.to_dict())
        assert restored.focused is True
        assert restored.required is True
        assert restored.value == "hello"

    def test_from_dict_empty(self):
        """from_dict({}) should produce default state."""
        s = AccessibilityState.from_dict({})
        assert s.disabled is False
        assert s.checked is None


# ── AccessibilityNode ────────────────────────────────────────────────────────


class TestAccessibilityNode:
    """Tests for AccessibilityNode queries and traversals."""

    def test_is_interactive_button(self):
        """A button node should be interactive."""
        tree = make_simple_form_tree()
        btn = tree.get_node("btn_submit")
        assert btn is not None
        assert btn.is_interactive() is True

    def test_is_visible_default(self):
        """Nodes without hidden state should be visible."""
        tree = make_simple_form_tree()
        assert tree.root.is_visible() is True

    def test_is_visible_hidden(self):
        """A node with hidden=True should not be visible."""
        node = AccessibilityNode(
            id="h", role="text", name="Secret",
            state=AccessibilityState(hidden=True),
        )
        assert node.is_visible() is False

    def test_is_focusable_interactive(self):
        """Interactive nodes that are not disabled/hidden should be focusable."""
        tree = make_simple_form_tree()
        btn = tree.get_node("btn_submit")
        assert btn is not None
        assert btn.is_focusable() is True

    def test_is_focusable_disabled(self):
        """Disabled interactive nodes should not be focusable."""
        node = AccessibilityNode(
            id="d", role="button", name="Disabled",
            state=AccessibilityState(disabled=True),
        )
        assert node.is_focusable() is False

    def test_is_focusable_tabindex(self):
        """Non-interactive nodes with tabindex >= 0 should be focusable."""
        node = AccessibilityNode(
            id="t", role="group", name="Group",
            properties={"tabindex": "0"},
        )
        assert node.is_focusable() is True

    def test_semantic_hash_deterministic(self):
        """Same tree structure should produce the same hash."""
        tree = make_simple_form_tree()
        h1 = tree.root.semantic_hash()
        h2 = tree.root.semantic_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_semantic_hash_differs_for_different_trees(self):
        """Different trees should (very likely) produce different hashes."""
        t1 = make_simple_form_tree()
        t2 = make_navigation_tree()
        assert t1.root.semantic_hash() != t2.root.semantic_hash()

    def test_subtree_size(self):
        """subtree_size should count the node plus all descendants."""
        tree = make_simple_form_tree()
        # root -> form -> [user, pw, submit] = 5 total
        assert tree.root.subtree_size() == 5

    def test_get_descendants(self):
        """get_descendants returns all descendants in BFS order."""
        tree = make_simple_form_tree()
        form = tree.get_node("form1")
        desc = form.get_descendants()
        desc_ids = [d.id for d in desc]
        assert "input_user" in desc_ids
        assert "input_pw" in desc_ids
        assert "btn_submit" in desc_ids
        assert "form1" not in desc_ids

    def test_find_by_role(self):
        """find_by_role should locate all matching descendants and self."""
        tree = make_simple_form_tree()
        textboxes = tree.root.find_by_role("textbox")
        assert len(textboxes) == 2
        names = {n.name for n in textboxes}
        assert "Username" in names
        assert "Password" in names

    def test_find_by_name_case_insensitive(self):
        """find_by_name should default to case-insensitive search."""
        tree = make_simple_form_tree()
        results = tree.root.find_by_name("submit")
        assert len(results) == 1
        assert results[0].role == "button"

    def test_find_by_name_case_sensitive(self):
        """find_by_name with case_sensitive=True should require exact case."""
        tree = make_simple_form_tree()
        results = tree.root.find_by_name("submit", case_sensitive=True)
        assert len(results) == 0  # "Submit" != "submit"

    def test_iter_preorder(self):
        """Pre-order should visit root first, then children left-to-right."""
        tree = make_simple_form_tree()
        ids = [n.id for n in tree.root.iter_preorder()]
        assert ids[0] == "root"
        assert ids[1] == "form1"
        assert "input_user" in ids
        # root before form before children
        assert ids.index("root") < ids.index("form1")

    def test_iter_postorder(self):
        """Post-order should visit leaves before parents."""
        tree = make_simple_form_tree()
        ids = [n.id for n in tree.root.iter_postorder()]
        assert ids[-1] == "root"
        assert ids[-2] == "form1"
        assert ids.index("input_user") < ids.index("form1")

    def test_node_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict should preserve node structure."""
        tree = make_simple_form_tree()
        d = tree.root.to_dict()
        restored = AccessibilityNode.from_dict(d)
        assert restored.id == "root"
        assert restored.role == "document"
        assert len(restored.children) == 1
        assert restored.children[0].id == "form1"

    def test_node_repr(self):
        """__repr__ should include id, role, name, and child count."""
        tree = make_simple_form_tree()
        r = repr(tree.root)
        assert "root" in r
        assert "document" in r


# ── AccessibilityTree ────────────────────────────────────────────────────────


class TestAccessibilityTree:
    """Tests for AccessibilityTree index, queries, algorithms, and serialisation."""

    def test_build_index(self):
        """build_index should populate node_index with all nodes."""
        tree = make_simple_form_tree()
        assert "root" in tree.node_index
        assert "form1" in tree.node_index
        assert "btn_submit" in tree.node_index

    def test_get_node_existing(self):
        """get_node should return the correct node."""
        tree = make_simple_form_tree()
        node = tree.get_node("input_user")
        assert node is not None
        assert node.name == "Username"

    def test_get_node_missing(self):
        """get_node with unknown id should return None."""
        tree = make_simple_form_tree()
        assert tree.get_node("nonexistent") is None

    def test_get_interactive_nodes(self):
        """get_interactive_nodes should return buttons, textboxes, etc."""
        tree = make_simple_form_tree()
        interactive = tree.get_interactive_nodes()
        roles = {n.role for n in interactive}
        assert "button" in roles
        assert "textbox" in roles
        assert "document" not in roles

    def test_get_visible_nodes(self):
        """get_visible_nodes should return all non-hidden nodes."""
        tree = make_simple_form_tree()
        visible = tree.get_visible_nodes()
        assert len(visible) == tree.size()  # none hidden

    def test_get_focusable_nodes(self):
        """get_focusable_nodes should return interactive, non-disabled nodes."""
        tree = make_simple_form_tree()
        focusable = tree.get_focusable_nodes()
        assert len(focusable) >= 3  # user, pw, submit

    def test_get_nodes_by_role(self):
        """get_nodes_by_role should filter by exact role."""
        tree = make_navigation_tree()
        links = tree.get_nodes_by_role("link")
        assert len(links) == 5

    def test_get_leaves(self):
        """get_leaves should return nodes without children."""
        tree = make_simple_form_tree()
        leaves = tree.get_leaves()
        leaf_ids = {n.id for n in leaves}
        assert "input_user" in leaf_ids
        assert "form1" not in leaf_ids

    def test_depth(self):
        """depth should return maximum depth across all nodes."""
        tree = make_simple_form_tree()
        assert tree.depth() == 2

    def test_size(self):
        """size should return total number of nodes."""
        tree = make_simple_form_tree()
        assert tree.size() == 5

    def test_lca_siblings(self):
        """LCA of two siblings should be their parent."""
        tree = make_simple_form_tree()
        result = tree.lca("input_user", "btn_submit")
        assert result is not None
        assert result.id == "form1"

    def test_lca_parent_child(self):
        """LCA of a node and its ancestor should be the ancestor."""
        tree = make_simple_form_tree()
        result = tree.lca("input_user", "form1")
        assert result is not None
        assert result.id == "form1"

    def test_lca_unknown_node(self):
        """LCA with an unknown node id should return None."""
        tree = make_simple_form_tree()
        assert tree.lca("input_user", "bogus") is None

    def test_subtree(self):
        """subtree should return a new tree rooted at the given node."""
        tree = make_simple_form_tree()
        sub = tree.subtree("form1")
        assert sub is not None
        assert sub.root.id == "form1"
        assert sub.root.parent_id is None
        assert sub.size() == 4  # form + user + pw + submit

    def test_subtree_unknown(self):
        """subtree with unknown node id should return None."""
        tree = make_simple_form_tree()
        assert tree.subtree("bogus") is None

    def test_path_between_siblings(self):
        """path_between two siblings should go through their parent."""
        tree = make_simple_form_tree()
        path = tree.path_between("input_user", "btn_submit")
        assert path is not None
        assert path[0] == "input_user"
        assert path[-1] == "btn_submit"
        assert "form1" in path

    def test_path_between_unknown(self):
        """path_between with unknown node should return None."""
        tree = make_simple_form_tree()
        assert tree.path_between("input_user", "bogus") is None

    def test_iter_bfs(self):
        """BFS should yield root first, then level by level."""
        tree = make_simple_form_tree()
        ids = [n.id for n in tree.iter_bfs()]
        assert ids[0] == "root"
        assert ids[1] == "form1"

    def test_iter_dfs(self):
        """DFS should yield root first, then depth-first."""
        tree = make_simple_form_tree()
        ids = [n.id for n in tree.iter_dfs()]
        assert ids[0] == "root"
        assert ids.index("root") < ids.index("form1")

    def test_validate_valid_tree(self):
        """A well-formed tree should produce no validation errors."""
        tree = make_simple_form_tree()
        errors = tree.validate()
        assert errors == []

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict should produce equivalent tree."""
        tree = make_simple_form_tree()
        d = tree.to_dict()
        restored = AccessibilityTree.from_dict(d)
        assert restored.size() == tree.size()
        assert restored.root.id == "root"

    def test_to_json_from_json_roundtrip(self):
        """to_json/from_json should round-trip through JSON string."""
        tree = make_navigation_tree()
        j = tree.to_json()
        restored = AccessibilityTree.from_json(j)
        assert restored.size() == tree.size()
        assert json.loads(j)["root"]["role"] == "document"

    def test_dashboard_tree_size(self):
        """The dashboard fixture should have the expected number of nodes."""
        tree = make_dashboard_tree()
        # root + main_region + table + 12 cells + chart = 16
        assert tree.size() == 16

    def test_modal_dialog_tree_structure(self):
        """The modal dialog fixture should have dialog with buttons."""
        tree = make_modal_dialog_tree()
        dialog = tree.get_node("dialog1")
        assert dialog is not None
        assert dialog.role == "dialog"
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) == 2

    def test_repr(self):
        """__repr__ should include size and depth info."""
        tree = make_simple_form_tree()
        r = repr(tree)
        assert "size=" in r
        assert "depth=" in r
