"""Unit tests for usability_oracle.repair.mutations.MutationOperator.

Tests the mutation operator that applies structural changes to an
AccessibilityTree—resize, reposition, relabel, remove_node, add_shortcut,
simplify_menu, and add_landmark operations.
"""

from __future__ import annotations

import copy
from typing import Optional

import pytest

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.repair.models import MutationType, UIMutation
from usability_oracle.repair.mutations import MutationOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_state(**overrides) -> AccessibilityState:
    """Build a default AccessibilityState with optional overrides."""
    defaults = dict(
        focused=False, selected=False, expanded=False, checked=None,
        disabled=False, hidden=False, required=False, readonly=False,
        pressed=None, value=None,
    )
    defaults.update(overrides)
    return AccessibilityState(**defaults)


def _node(
    id: str,
    role: str = "button",
    name: str = "",
    bbox: Optional[BoundingBox] = None,
    children: list | None = None,
) -> AccessibilityNode:
    """Create a minimal AccessibilityNode."""
    return AccessibilityNode(
        id=id,
        role=role,
        name=name or id,
        description="",
        bounding_box=bbox or BoundingBox(x=0, y=0, width=40, height=40),
        properties={},
        state=_default_state(),
        children=children or [],
        parent_id=None,
        depth=0,
        index_in_parent=0,
    )


def _make_tree() -> AccessibilityTree:
    """Build a small tree: root → [btn_a, btn_b, menu(item1, item2, item3)]."""
    item1 = _node("item1", role="menuitem", name="Open")
    item2 = _node("item2", role="menuitem", name="Save")
    item3 = _node("item3", role="menuitem", name="Close")
    menu = _node("menu", role="menu", name="File", children=[item1, item2, item3])
    btn_a = _node("btn_a", role="button", name="OK",
                   bbox=BoundingBox(x=10, y=10, width=30, height=30))
    btn_b = _node("btn_b", role="button", name="Cancel",
                   bbox=BoundingBox(x=60, y=10, width=30, height=30))
    root = _node("root", role="group", name="Root", children=[btn_a, btn_b, menu])
    tree = AccessibilityTree(root=root)
    tree.build_index()
    return tree


def _make_wide_menu_tree(n_items: int = 12) -> AccessibilityTree:
    """Build a tree with a menu containing many items for simplify_menu tests."""
    items = [_node(f"item_{i}", role="menuitem", name=f"Item {i}") for i in range(n_items)]
    menu = _node("wide_menu", role="menu", name="BigMenu", children=items)
    root = _node("root", role="group", name="Root", children=[menu])
    tree = AccessibilityTree(root=root)
    tree.build_index()
    return tree


# ---------------------------------------------------------------------------
# apply() basics
# ---------------------------------------------------------------------------

class TestMutationOperatorApply:
    """Tests for MutationOperator.apply returning a modified tree."""

    def test_apply_returns_accessibility_tree(self):
        """apply() returns an AccessibilityTree."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="btn_a",
            parameters={"new_name": "Accept"},
        )
        result = op.apply(tree, mutation)
        assert isinstance(result, AccessibilityTree)

    def test_apply_does_not_mutate_original(self):
        """apply() should not modify the original tree in-place."""
        op = MutationOperator()
        tree = _make_tree()
        original_name = tree.get_node("btn_a").name
        mutation = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="btn_a",
            parameters={"new_name": "Changed"},
        )
        op.apply(tree, mutation)
        assert tree.get_node("btn_a").name == original_name


# ---------------------------------------------------------------------------
# apply_all()
# ---------------------------------------------------------------------------

class TestApplyAll:
    """Tests for MutationOperator.apply_all chaining mutations."""

    def test_apply_all_returns_tree(self):
        """apply_all returns an AccessibilityTree after chaining mutations."""
        op = MutationOperator()
        tree = _make_tree()
        mutations = [
            UIMutation(
                mutation_type=MutationType.RELABEL.value,
                target_node_id="btn_a",
                parameters={"new_name": "Yes"},
            ),
            UIMutation(
                mutation_type=MutationType.RELABEL.value,
                target_node_id="btn_b",
                parameters={"new_name": "No"},
            ),
        ]
        result = op.apply_all(tree, mutations)
        assert isinstance(result, AccessibilityTree)

    def test_apply_all_applies_both(self):
        """Both mutations are applied when chained."""
        op = MutationOperator()
        tree = _make_tree()
        mutations = [
            UIMutation(
                mutation_type=MutationType.RELABEL.value,
                target_node_id="btn_a",
                parameters={"new_name": "Yes"},
            ),
            UIMutation(
                mutation_type=MutationType.RELABEL.value,
                target_node_id="btn_b",
                parameters={"new_name": "No"},
            ),
        ]
        result = op.apply_all(tree, mutations)
        assert result.get_node("btn_a").name == "Yes"
        assert result.get_node("btn_b").name == "No"

    def test_apply_all_empty_list(self):
        """apply_all with empty list returns equivalent tree."""
        op = MutationOperator()
        tree = _make_tree()
        result = op.apply_all(tree, [])
        assert result.get_node("btn_a").name == tree.get_node("btn_a").name


# ---------------------------------------------------------------------------
# Resize mutation
# ---------------------------------------------------------------------------

class TestResizeMutation:
    """Tests that RESIZE mutation changes the bounding box dimensions."""

    def test_resize_width(self):
        """RESIZE with width parameter changes the node's bounding_box width."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.RESIZE.value,
            target_node_id="btn_a",
            parameters={"width": 60.0},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a").bounding_box.width == pytest.approx(60.0)

    def test_resize_height(self):
        """RESIZE with height parameter changes the node's bounding_box height."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.RESIZE.value,
            target_node_id="btn_a",
            parameters={"height": 50.0},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a").bounding_box.height == pytest.approx(50.0)

    def test_resize_preserves_position(self):
        """RESIZE should not change the x, y position of the node."""
        op = MutationOperator()
        tree = _make_tree()
        original = tree.get_node("btn_a").bounding_box
        mutation = UIMutation(
            mutation_type=MutationType.RESIZE.value,
            target_node_id="btn_a",
            parameters={"width": 100.0, "height": 100.0},
        )
        result = op.apply(tree, mutation)
        new_bb = result.get_node("btn_a").bounding_box
        assert new_bb.x == pytest.approx(original.x)
        assert new_bb.y == pytest.approx(original.y)


# ---------------------------------------------------------------------------
# Reposition mutation
# ---------------------------------------------------------------------------

class TestRepositionMutation:
    """Tests that REPOSITION mutation changes x, y coordinates."""

    def test_reposition_x(self):
        """REPOSITION changes the node's bounding_box x coordinate."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.REPOSITION.value,
            target_node_id="btn_a",
            parameters={"x": 200.0},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a").bounding_box.x == pytest.approx(200.0)

    def test_reposition_y(self):
        """REPOSITION changes the node's bounding_box y coordinate."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.REPOSITION.value,
            target_node_id="btn_a",
            parameters={"y": 300.0},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a").bounding_box.y == pytest.approx(300.0)

    def test_reposition_preserves_size(self):
        """REPOSITION should not change width/height."""
        op = MutationOperator()
        tree = _make_tree()
        original = tree.get_node("btn_a").bounding_box
        mutation = UIMutation(
            mutation_type=MutationType.REPOSITION.value,
            target_node_id="btn_a",
            parameters={"x": 500.0, "y": 500.0},
        )
        result = op.apply(tree, mutation)
        new_bb = result.get_node("btn_a").bounding_box
        assert new_bb.width == pytest.approx(original.width)
        assert new_bb.height == pytest.approx(original.height)


# ---------------------------------------------------------------------------
# Relabel mutation
# ---------------------------------------------------------------------------

class TestRelabelMutation:
    """Tests that RELABEL mutation changes the node's name."""

    def test_relabel_changes_name(self):
        """RELABEL sets a new name on the target node."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="btn_a",
            parameters={"new_name": "Confirm"},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a").name == "Confirm"

    def test_relabel_other_nodes_unchanged(self):
        """RELABEL should not affect other nodes."""
        op = MutationOperator()
        tree = _make_tree()
        orig_b = tree.get_node("btn_b").name
        mutation = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="btn_a",
            parameters={"new_name": "Confirm"},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_b").name == orig_b


# ---------------------------------------------------------------------------
# Remove node mutation
# ---------------------------------------------------------------------------

class TestRemoveNodeMutation:
    """Tests that REMOVE mutation removes a node from the tree."""

    def test_remove_node(self):
        """After REMOVE, get_node returns None for the removed node."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.REMOVE.value,
            target_node_id="btn_b",
            parameters={},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_b") is None

    def test_remove_preserves_siblings(self):
        """Removing one child should not affect its siblings."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.REMOVE.value,
            target_node_id="btn_b",
            parameters={},
        )
        result = op.apply(tree, mutation)
        assert result.get_node("btn_a") is not None


# ---------------------------------------------------------------------------
# Add shortcut mutation
# ---------------------------------------------------------------------------

class TestAddShortcutMutation:
    """Tests that ADD_SHORTCUT adds a keyboard shortcut property."""

    def test_add_shortcut(self):
        """ADD_SHORTCUT adds a shortcut property to the node."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.ADD_SHORTCUT.value,
            target_node_id="btn_a",
            parameters={"shortcut_key": "Ctrl+S"},
        )
        result = op.apply(tree, mutation)
        node = result.get_node("btn_a")
        # The shortcut is stored in properties
        assert "keyboard_shortcut" in node.properties or "accesskey" in node.properties


# ---------------------------------------------------------------------------
# Simplify menu mutation
# ---------------------------------------------------------------------------

class TestSimplifyMenuMutation:
    """Tests that SIMPLIFY_MENU limits the number of children."""

    def test_simplify_menu_reduces_children(self):
        """After SIMPLIFY_MENU, the menu node has at most max_items children."""
        op = MutationOperator()
        tree = _make_wide_menu_tree(n_items=12)
        mutation = UIMutation(
            mutation_type=MutationType.SIMPLIFY_MENU.value,
            target_node_id="wide_menu",
            parameters={"max_items": 5},
        )
        result = op.apply(tree, mutation)
        menu_node = result.get_node("wide_menu")
        assert len(menu_node.children) <= 5

    def test_simplify_menu_no_op_when_under_limit(self):
        """SIMPLIFY_MENU is a no-op when children count <= max_items."""
        op = MutationOperator()
        tree = _make_tree()
        # "menu" has 3 children, max_items=7 ⇒ no-op
        mutation = UIMutation(
            mutation_type=MutationType.SIMPLIFY_MENU.value,
            target_node_id="menu",
            parameters={"max_items": 7},
        )
        result = op.apply(tree, mutation)
        assert len(result.get_node("menu").children) == 3


# ---------------------------------------------------------------------------
# Add landmark mutation
# ---------------------------------------------------------------------------

class TestAddLandmarkMutation:
    """Tests that ADD_LANDMARK wraps nodes in a landmark region."""

    def test_add_landmark_returns_tree(self):
        """ADD_LANDMARK returns a valid AccessibilityTree."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.ADD_LANDMARK.value,
            target_node_id="btn_a",
            parameters={"landmark_role": "region"},
        )
        result = op.apply(tree, mutation)
        assert isinstance(result, AccessibilityTree)

    def test_add_landmark_node_still_exists(self):
        """The target node is still reachable after ADD_LANDMARK."""
        op = MutationOperator()
        tree = _make_tree()
        mutation = UIMutation(
            mutation_type=MutationType.ADD_LANDMARK.value,
            target_node_id="btn_a",
            parameters={"landmark_role": "navigation"},
        )
        result = op.apply(tree, mutation)
        # The node should still exist somewhere in the tree
        found = False
        for n in result.root.iter_preorder():
            if n.id == "btn_a":
                found = True
                break
        assert found
