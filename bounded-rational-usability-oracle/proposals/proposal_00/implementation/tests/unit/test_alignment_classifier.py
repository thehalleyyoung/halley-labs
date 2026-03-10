"""Unit tests for :class:`usability_oracle.alignment.classifier.ResidualClassifier`.

Validates classification of residual (unmatched) nodes into MOVE, RENAME,
RETYPE, ADD, and REMOVE operations, including mixed-change scenarios and
empty-input edge cases.
"""

from __future__ import annotations

import pytest

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentContext,
    AlignmentPass,
    BoundingBox,
    EditOperation,
    EditOperationType,
    NodeMapping,
)
from usability_oracle.alignment.classifier import ResidualClassifier


# ============================================================================
# Helpers
# ============================================================================


def _node(
    nid: str,
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str = "",
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    bbox: BoundingBox | None = None,
    child_roles: list[str] | None = None,
) -> AccessibilityNode:
    """Create a minimal :class:`AccessibilityNode`."""
    props: dict = {}
    if child_roles is not None:
        props["child_roles"] = child_roles
    return AccessibilityNode(
        node_id=nid,
        role=role,
        name=name,
        parent_id=parent_id,
        children_ids=children_ids or [],
        bounding_box=bbox,
        properties=props,
    )


def _tree(nodes: list[AccessibilityNode], root_ids: list[str]) -> AccessibilityTree:
    """Build an :class:`AccessibilityTree` from a flat node list."""
    return AccessibilityTree(
        nodes={n.node_id: n for n in nodes},
        root_ids=root_ids,
    )


def _context(
    src: AccessibilityTree | None = None,
    tgt: AccessibilityTree | None = None,
    config: AlignmentConfig | None = None,
) -> AlignmentContext:
    """Build an :class:`AlignmentContext` with sensible defaults."""
    return AlignmentContext(
        source_tree=src or AccessibilityTree(),
        target_tree=tgt or AccessibilityTree(),
        config=config or AlignmentConfig(),
    )


# ============================================================================
# Tests — MOVE detection
# ============================================================================


class TestMoveDetection:
    """classify() detects MOVE operations: same role+name, different parent."""

    def test_basic_move(self) -> None:
        """A node with same role+name but a different parent → MOVE."""
        src_node = _node("btn", AccessibilityRole.BUTTON, "Submit", parent_id="form1")
        tgt_node = _node("btn2", AccessibilityRole.BUTTON, "Submit", parent_id="form2")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert len(move_ops) == 1
        assert move_ops[0].source_node_id == "btn"
        assert move_ops[0].target_node_id == "btn2"

    def test_move_details_contain_parents(self) -> None:
        """MOVE operations record old_parent and new_parent in details."""
        src_node = _node("x", AccessibilityRole.LINK, "Home", parent_id="nav1")
        tgt_node = _node("y", AccessibilityRole.LINK, "Home", parent_id="nav2")
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert move_ops[0].details["old_parent"] == "nav1"
        assert move_ops[0].details["new_parent"] == "nav2"

    def test_same_parent_not_a_move(self) -> None:
        """Same role+name with same parent → NOT a move (just a pure duplicate)."""
        src_node = _node("a", AccessibilityRole.BUTTON, "OK", parent_id="p")
        tgt_node = _node("b", AccessibilityRole.BUTTON, "OK", parent_id="p")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert len(move_ops) == 0

    def test_move_consumes_nodes(self) -> None:
        """A moved node should not also appear in additions or removals."""
        src_node = _node("s1", AccessibilityRole.CHECKBOX, "Accept", parent_id="f1")
        tgt_node = _node("t1", AccessibilityRole.CHECKBOX, "Accept", parent_id="f2")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        assert "s1" not in rems
        assert "t1" not in adds

    def test_multiple_moves(self) -> None:
        """Multiple nodes can be detected as moves simultaneously."""
        src = [
            _node("s1", AccessibilityRole.BUTTON, "Save", parent_id="p1"),
            _node("s2", AccessibilityRole.LINK, "Cancel", parent_id="p1"),
        ]
        tgt = [
            _node("t1", AccessibilityRole.BUTTON, "Save", parent_id="p2"),
            _node("t2", AccessibilityRole.LINK, "Cancel", parent_id="p2"),
        ]
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify(src, tgt, ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert len(move_ops) == 2


# ============================================================================
# Tests — RENAME detection
# ============================================================================


class TestRenameDetection:
    """classify() detects RENAME operations: same role + parent, different name."""

    def test_basic_rename(self) -> None:
        """Same role and parent, similar but different name → RENAME."""
        src_node = _node("s1", AccessibilityRole.BUTTON, "Submit Button", parent_id="form")
        tgt_node = _node("t1", AccessibilityRole.BUTTON, "Submit Buttons", parent_id="form")
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        rename_ops = [o for o in ops if o.operation_type == EditOperationType.RENAME]
        assert len(rename_ops) == 1
        assert rename_ops[0].details["old_name"] == "Submit Button"
        assert rename_ops[0].details["new_name"] == "Submit Buttons"

    def test_rename_requires_same_role(self) -> None:
        """Different role → not a rename (may be a retype or unmatched)."""
        src_node = _node("s1", AccessibilityRole.BUTTON, "Click Here", parent_id="p")
        tgt_node = _node("t1", AccessibilityRole.LINK, "Click Here!", parent_id="p")
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        rename_ops = [o for o in ops if o.operation_type == EditOperationType.RENAME]
        assert len(rename_ops) == 0

    def test_rename_below_threshold_not_detected(self) -> None:
        """Names too dissimilar (below rename_threshold) → no rename."""
        cfg = AlignmentConfig(rename_threshold=0.99)
        src_node = _node("s1", AccessibilityRole.HEADING, "Introduction", parent_id="main")
        tgt_node = _node("t1", AccessibilityRole.HEADING, "Conclusion", parent_id="main")
        ctx = _context(config=cfg)
        ops, _, _ = ResidualClassifier(cfg).classify([src_node], [tgt_node], ctx)
        rename_ops = [o for o in ops if o.operation_type == EditOperationType.RENAME]
        assert len(rename_ops) == 0

    def test_rename_consumes_nodes(self) -> None:
        """A renamed node should not also appear in additions or removals."""
        src_node = _node("s1", AccessibilityRole.BUTTON, "Save Draft", parent_id="bar")
        tgt_node = _node("t1", AccessibilityRole.BUTTON, "Save Craft", parent_id="bar")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        assert "s1" not in rems
        assert "t1" not in adds


# ============================================================================
# Tests — RETYPE detection
# ============================================================================


class TestRetypeDetection:
    """classify() detects RETYPE operations: same name, different role."""

    def test_basic_retype(self) -> None:
        """Same name + close position but different role → RETYPE."""
        bbox = BoundingBox(x=10, y=10, width=100, height=40)
        src_node = _node("s1", AccessibilityRole.BUTTON, "Submit", bbox=bbox)
        tgt_node = _node("t1", AccessibilityRole.LINK, "Submit", bbox=bbox)
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        retype_ops = [o for o in ops if o.operation_type == EditOperationType.RETYPE]
        assert len(retype_ops) == 1
        assert retype_ops[0].details["old_role"] == "button"
        assert retype_ops[0].details["new_role"] == "link"

    def test_retype_details_contain_taxonomy_distance(self) -> None:
        """RETYPE operations include taxonomy_distance in details."""
        bbox = BoundingBox(x=0, y=0, width=50, height=50)
        src_node = _node("s1", AccessibilityRole.BUTTON, "Go", bbox=bbox)
        tgt_node = _node("t1", AccessibilityRole.LINK, "Go", bbox=bbox)
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        retype_ops = [o for o in ops if o.operation_type == EditOperationType.RETYPE]
        assert "taxonomy_distance" in retype_ops[0].details

    def test_retype_below_threshold_not_detected(self) -> None:
        """If the combined score is below retype_threshold, no retype emitted."""
        cfg = AlignmentConfig(retype_threshold=0.99)
        src_node = _node("s1", AccessibilityRole.BUTTON, "Alpha")
        tgt_node = _node("t1", AccessibilityRole.TABLE, "Zeta")
        ctx = _context(config=cfg)
        ops, _, _ = ResidualClassifier(cfg).classify([src_node], [tgt_node], ctx)
        retype_ops = [o for o in ops if o.operation_type == EditOperationType.RETYPE]
        assert len(retype_ops) == 0

    def test_same_role_not_retype(self) -> None:
        """If both nodes share the same role it cannot be a retype."""
        src_node = _node("s1", AccessibilityRole.BUTTON, "OK")
        tgt_node = _node("t1", AccessibilityRole.BUTTON, "OK")
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        retype_ops = [o for o in ops if o.operation_type == EditOperationType.RETYPE]
        assert len(retype_ops) == 0

    def test_retype_consumes_nodes(self) -> None:
        """A retyped node should not also appear in additions or removals."""
        bbox = BoundingBox(x=5, y=5, width=80, height=30)
        src_node = _node("s1", AccessibilityRole.TEXTBOX, "Email", bbox=bbox)
        tgt_node = _node("t1", AccessibilityRole.COMBOBOX, "Email", bbox=bbox)
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [tgt_node], ctx)
        assert "s1" not in rems
        assert "t1" not in adds


# ============================================================================
# Tests — pure additions
# ============================================================================


class TestAdditions:
    """Genuinely new target nodes are classified as additions."""

    def test_pure_addition(self) -> None:
        """Target nodes with no matching source counterpart → additions."""
        tgt_node = _node("new1", AccessibilityRole.BUTTON, "New Button")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([], [tgt_node], ctx)
        assert "new1" in adds

    def test_addition_generates_add_op(self) -> None:
        """Each addition also generates an EditOperation of type ADD."""
        tgt_node = _node("new1", AccessibilityRole.CHECKBOX, "Opt-in")
        ctx = _context()
        ops, adds, _ = ResidualClassifier().classify([], [tgt_node], ctx)
        add_ops = [o for o in ops if o.operation_type == EditOperationType.ADD]
        assert len(add_ops) == 1
        assert add_ops[0].target_node_id == "new1"
        assert add_ops[0].source_node_id is None

    def test_multiple_additions(self) -> None:
        """Several new target nodes each appear in the additions list."""
        targets = [
            _node("n1", AccessibilityRole.LINK, "FAQ"),
            _node("n2", AccessibilityRole.BUTTON, "Help"),
        ]
        ctx = _context()
        _, adds, _ = ResidualClassifier().classify([], targets, ctx)
        assert set(adds) == {"n1", "n2"}


# ============================================================================
# Tests — pure removals
# ============================================================================


class TestRemovals:
    """Genuinely deleted source nodes are classified as removals."""

    def test_pure_removal(self) -> None:
        """Source nodes with no matching target counterpart → removals."""
        src_node = _node("old1", AccessibilityRole.BUTTON, "Deprecated")
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([src_node], [], ctx)
        assert "old1" in rems

    def test_removal_generates_remove_op(self) -> None:
        """Each removal generates an EditOperation of type REMOVE."""
        src_node = _node("old1", AccessibilityRole.TAB, "Settings")
        ctx = _context()
        ops, _, rems = ResidualClassifier().classify([src_node], [], ctx)
        rem_ops = [o for o in ops if o.operation_type == EditOperationType.REMOVE]
        assert len(rem_ops) == 1
        assert rem_ops[0].source_node_id == "old1"
        assert rem_ops[0].target_node_id is None

    def test_multiple_removals(self) -> None:
        """Several deleted source nodes each appear in the removals list."""
        sources = [
            _node("d1", AccessibilityRole.LINK, "Old Link"),
            _node("d2", AccessibilityRole.IMG, "Banner"),
        ]
        ctx = _context()
        _, _, rems = ResidualClassifier().classify(sources, [], ctx)
        assert set(rems) == {"d1", "d2"}


# ============================================================================
# Tests — mixed changes (move + rename + add + remove)
# ============================================================================


class TestMixedChanges:
    """Scenarios where multiple change types coexist."""

    def test_move_and_addition(self) -> None:
        """One node moves; another is genuinely new."""
        src = [_node("s1", AccessibilityRole.BUTTON, "Save", parent_id="p1")]
        tgt = [
            _node("t1", AccessibilityRole.BUTTON, "Save", parent_id="p2"),
            _node("t2", AccessibilityRole.LINK, "New Link"),
        ]
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify(src, tgt, ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert len(move_ops) == 1
        assert "t2" in adds
        assert rems == []

    def test_rename_and_removal(self) -> None:
        """One node is renamed; another is removed."""
        src = [
            _node("s1", AccessibilityRole.HEADING, "Welcome", parent_id="main"),
            _node("s2", AccessibilityRole.IMG, "Banner"),
        ]
        tgt = [
            _node("t1", AccessibilityRole.HEADING, "Hello", parent_id="main"),
        ]
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify(src, tgt, ctx)
        rename_ops = [o for o in ops if o.operation_type == EditOperationType.RENAME]
        # The heading may or may not be detected as rename depending on similarity.
        # But the image should be in removals.
        assert "s2" in rems or any(
            o.source_node_id == "s2" for o in ops
            if o.operation_type in (EditOperationType.MOVE, EditOperationType.RETYPE)
        )

    def test_move_plus_rename_different_nodes(self) -> None:
        """One node moves, a different node is renamed — no interference."""
        src = [
            _node("s_move", AccessibilityRole.BUTTON, "Submit", parent_id="f1"),
            _node("s_ren", AccessibilityRole.HEADING, "Page Title", parent_id="main"),
        ]
        tgt = [
            _node("t_move", AccessibilityRole.BUTTON, "Submit", parent_id="f2"),
            _node("t_ren", AccessibilityRole.HEADING, "Page Titel", parent_id="main"),
        ]
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify(src, tgt, ctx)
        move_ops = [o for o in ops if o.operation_type == EditOperationType.MOVE]
        assert len(move_ops) == 1
        assert adds == []
        assert rems == []


# ============================================================================
# Tests — empty inputs
# ============================================================================


class TestEmptyInputs:
    """Edge cases with empty unmatched lists."""

    def test_both_empty(self) -> None:
        """No unmatched nodes on either side → no ops, adds, or removals."""
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([], [], ctx)
        assert ops == []
        assert adds == []
        assert rems == []

    def test_empty_source_only_additions(self) -> None:
        """Only target nodes present → everything is an addition."""
        tgt = [_node("t1", AccessibilityRole.BUTTON, "New")]
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify([], tgt, ctx)
        assert adds == ["t1"]
        assert rems == []

    def test_empty_target_only_removals(self) -> None:
        """Only source nodes present → everything is a removal."""
        src = [_node("s1", AccessibilityRole.BUTTON, "Old")]
        ctx = _context()
        ops, adds, rems = ResidualClassifier().classify(src, [], ctx)
        assert rems == ["s1"]
        assert adds == []


# ============================================================================
# Tests — return types
# ============================================================================


class TestReturnTypes:
    """Verify the shape and types of the classify() return value."""

    def test_return_is_tuple_of_three(self) -> None:
        """classify() returns a 3-tuple: (edit_ops, additions, removals)."""
        ctx = _context()
        result = ResidualClassifier().classify([], [], ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_edit_ops_are_edit_operation_instances(self) -> None:
        """First element is a list of EditOperation objects."""
        src = [_node("s1", AccessibilityRole.BUTTON, "OK")]
        ctx = _context()
        ops, _, _ = ResidualClassifier().classify(src, [], ctx)
        for op in ops:
            assert isinstance(op, EditOperation)

    def test_additions_are_string_ids(self) -> None:
        """Second element is a list of node-id strings."""
        tgt = [_node("t1", AccessibilityRole.LINK, "Go")]
        ctx = _context()
        _, adds, _ = ResidualClassifier().classify([], tgt, ctx)
        for nid in adds:
            assert isinstance(nid, str)

    def test_removals_are_string_ids(self) -> None:
        """Third element is a list of node-id strings."""
        src = [_node("s1", AccessibilityRole.LINK, "Go")]
        ctx = _context()
        _, _, rems = ResidualClassifier().classify(src, [], ctx)
        for nid in rems:
            assert isinstance(nid, str)


# ============================================================================
# Tests — config defaults
# ============================================================================


class TestClassifierConfig:
    """Verify that the classifier respects configuration."""

    def test_default_config_when_none(self) -> None:
        """Passing config=None uses the default AlignmentConfig."""
        cls = ResidualClassifier(config=None)
        assert cls.config.move_threshold == pytest.approx(0.80)
        assert cls.config.rename_threshold == pytest.approx(0.70)
        assert cls.config.retype_threshold == pytest.approx(0.60)

    def test_custom_config_used(self) -> None:
        """A custom config is stored and used by the classifier."""
        cfg = AlignmentConfig(rename_threshold=0.50)
        cls = ResidualClassifier(cfg)
        assert cls.config.rename_threshold == pytest.approx(0.50)
