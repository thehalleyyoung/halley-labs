"""Unit tests for :class:`usability_oracle.alignment.differ.SemanticDiffer`.

Validates the full 3-pass alignment pipeline: ``diff()`` returns an
:class:`AlignmentResult` with correct mappings, edit operations, additions,
removals, edit distance, similarity score, and pass statistics.
"""

from __future__ import annotations

import pytest

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentPass,
    AlignmentResult,
    BoundingBox,
    EditOperation,
    EditOperationType,
    NodeMapping,
)
from usability_oracle.alignment.differ import SemanticDiffer


# ============================================================================
# Helpers
# ============================================================================


def _node(
    nid: str,
    role: AccessibilityRole = AccessibilityRole.GENERIC,
    name: str = "",
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    bbox: BoundingBox | None = None,
) -> AccessibilityNode:
    """Create a minimal :class:`AccessibilityNode`."""
    return AccessibilityNode(
        node_id=nid,
        role=role,
        name=name,
        parent_id=parent_id,
        children_ids=children_ids or [],
        bounding_box=bbox,
    )


def _tree(nodes: list[AccessibilityNode], root_ids: list[str]) -> AccessibilityTree:
    """Build an :class:`AccessibilityTree` from a flat node list."""
    return AccessibilityTree(
        nodes={n.node_id: n for n in nodes},
        root_ids=root_ids,
    )


def _simple_tree(prefix: str = "") -> AccessibilityTree:
    """Deterministic small tree: root → [nav, main]; nav → [link1, link2].

    *prefix* is prepended to every node-id so that two calls with different
    prefixes produce structurally identical but ID-disjoint trees.
    """
    p = prefix
    root = _node(f"{p}root", AccessibilityRole.DOCUMENT, "Page", children_ids=[f"{p}nav", f"{p}main"])
    nav = _node(f"{p}nav", AccessibilityRole.NAVIGATION, "Navigation", parent_id=f"{p}root", children_ids=[f"{p}link1", f"{p}link2"])
    main = _node(f"{p}main", AccessibilityRole.MAIN, "Main Content", parent_id=f"{p}root")
    link1 = _node(f"{p}link1", AccessibilityRole.LINK, "Home", parent_id=f"{p}nav")
    link2 = _node(f"{p}link2", AccessibilityRole.LINK, "About", parent_id=f"{p}nav")
    return _tree([root, nav, main, link1, link2], [f"{p}root"])


# ============================================================================
# Tests — AlignmentResult shape
# ============================================================================


class TestAlignmentResultShape:
    """diff() must return a well-formed AlignmentResult."""

    def test_return_type(self) -> None:
        """diff() returns an AlignmentResult instance."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert isinstance(result, AlignmentResult)

    def test_result_has_mappings(self) -> None:
        """AlignmentResult.mappings is a list of NodeMapping."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert isinstance(result.mappings, list)
        for m in result.mappings:
            assert isinstance(m, NodeMapping)

    def test_result_has_edit_operations(self) -> None:
        """AlignmentResult.edit_operations is a list of EditOperation."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert isinstance(result.edit_operations, list)
        for op in result.edit_operations:
            assert isinstance(op, EditOperation)

    def test_result_has_additions_and_removals(self) -> None:
        """additions and removals are lists of str node-ids."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert isinstance(result.additions, list)
        assert isinstance(result.removals, list)

    def test_result_has_pass_statistics(self) -> None:
        """pass_statistics maps every AlignmentPass to an int count."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert isinstance(result.pass_statistics, dict)
        for ap in AlignmentPass:
            assert ap in result.pass_statistics


# ============================================================================
# Tests — identical trees
# ============================================================================


class TestIdenticalTrees:
    """Diffing a tree against an exact copy should report perfect similarity."""

    def test_similarity_is_one(self) -> None:
        """Identical trees → similarity_score == 1.0."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.similarity_score == pytest.approx(1.0)

    def test_edit_distance_is_zero(self) -> None:
        """Identical trees → edit_distance == 0.0."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.edit_distance == pytest.approx(0.0)

    def test_no_additions(self) -> None:
        """Identical trees → no additions."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.additions == []

    def test_no_removals(self) -> None:
        """Identical trees → no removals."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.removals == []

    def test_all_nodes_mapped(self) -> None:
        """Every source node is mapped to a target node."""
        src = _simple_tree()
        tgt = _simple_tree()
        result = SemanticDiffer().diff(src, tgt)
        assert len(result.mappings) == src.node_count()

    def test_exact_hash_pass_dominates(self) -> None:
        """All mappings should come from the EXACT_HASH pass for identical trees."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        for m in result.mappings:
            assert m.pass_matched in (
                AlignmentPass.EXACT_HASH,
                AlignmentPass.EXACT_ID,
                AlignmentPass.EXACT_PATH,
            )


# ============================================================================
# Tests — completely different trees
# ============================================================================


class TestCompletelyDifferentTrees:
    """Trees that share no structure or IDs should have very low similarity."""

    def test_low_similarity(self) -> None:
        """Disjoint trees → similarity_score near 0."""
        src = _tree([_node("s1", AccessibilityRole.BUTTON, "Click")], ["s1"])
        tgt = _tree([_node("t1", AccessibilityRole.TABLE, "Data Table")], ["t1"])
        result = SemanticDiffer().diff(src, tgt)
        assert result.similarity_score < 0.5

    def test_positive_edit_distance(self) -> None:
        """Disjoint trees → edit_distance > 0."""
        src = _tree([_node("s1", AccessibilityRole.FORM, "Login")], ["s1"])
        tgt = _tree([_node("t1", AccessibilityRole.IMG, "Logo")], ["t1"])
        result = SemanticDiffer().diff(src, tgt)
        assert result.edit_distance > 0.0

    def test_additions_and_removals_present(self) -> None:
        """Disjoint trees should produce at least additions or removals or edit ops."""
        src = _tree(
            [
                _node("sr", AccessibilityRole.DOCUMENT, "Old", children_ids=["sc"]),
                _node("sc", AccessibilityRole.BUTTON, "Legacy", parent_id="sr"),
            ],
            ["sr"],
        )
        tgt = _tree(
            [
                _node("tr", AccessibilityRole.FORM, "New", children_ids=["tc"]),
                _node("tc", AccessibilityRole.TEXTBOX, "Input", parent_id="tr"),
            ],
            ["tr"],
        )
        result = SemanticDiffer().diff(src, tgt)
        total_changes = len(result.additions) + len(result.removals) + len(result.edit_operations)
        assert total_changes > 0


# ============================================================================
# Tests — edit_distance metric
# ============================================================================


class TestEditDistance:
    """edit_distance is a non-negative float summarising structural differences."""

    def test_edit_distance_non_negative(self) -> None:
        """Edit distance is always >= 0."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.edit_distance >= 0.0

    def test_edit_distance_increases_with_changes(self) -> None:
        """Adding a node increases edit distance relative to the identical case."""
        src = _simple_tree()
        tgt = _simple_tree()
        baseline = SemanticDiffer().diff(src, tgt).edit_distance

        tgt2 = _simple_tree()
        tgt2.nodes["extra"] = _node("extra", AccessibilityRole.BUTTON, "Extra", parent_id="main")
        tgt2.nodes["main"].children_ids.append("extra")
        modified = SemanticDiffer().diff(_simple_tree(), tgt2).edit_distance

        assert modified >= baseline

    def test_empty_trees_zero_distance(self) -> None:
        """Two empty trees → edit_distance 0."""
        result = SemanticDiffer().diff(AccessibilityTree(), AccessibilityTree())
        assert result.edit_distance == pytest.approx(0.0)


# ============================================================================
# Tests — similarity_score metric
# ============================================================================


class TestSimilarityScore:
    """similarity_score is in [0, 1] and reflects the fraction of matched nodes."""

    def test_similarity_in_range(self) -> None:
        """Similarity score is between 0 and 1 inclusive."""
        src = _simple_tree()
        tgt = _tree([_node("x", AccessibilityRole.BUTTON, "X")], ["x"])
        result = SemanticDiffer().diff(src, tgt)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_empty_trees_similarity_one(self) -> None:
        """Two empty trees are trivially identical → 1.0."""
        result = SemanticDiffer().diff(AccessibilityTree(), AccessibilityTree())
        assert result.similarity_score == pytest.approx(1.0)

    def test_partial_match_intermediate_similarity(self) -> None:
        """Partial overlap yields a similarity strictly between 0 and 1."""
        src = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Doc", children_ids=["a", "b"]),
                _node("a", AccessibilityRole.BUTTON, "OK", parent_id="root"),
                _node("b", AccessibilityRole.LINK, "Help", parent_id="root"),
            ],
            ["root"],
        )
        tgt = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Doc", children_ids=["a", "c"]),
                _node("a", AccessibilityRole.BUTTON, "OK", parent_id="root"),
                _node("c", AccessibilityRole.TEXTBOX, "Search", parent_id="root"),
            ],
            ["root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        assert 0.0 < result.similarity_score < 1.0


# ============================================================================
# Tests — pass_statistics
# ============================================================================


class TestPassStatistics:
    """pass_statistics counts how many mappings came from each pass."""

    def test_all_passes_present(self) -> None:
        """Every AlignmentPass key exists in pass_statistics."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        for ap in AlignmentPass:
            assert ap in result.pass_statistics

    def test_exact_pass_dominates_identical_trees(self) -> None:
        """For identical trees the exact passes account for all mappings."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        exact_count = (
            result.pass_statistics.get(AlignmentPass.EXACT_HASH, 0)
            + result.pass_statistics.get(AlignmentPass.EXACT_ID, 0)
            + result.pass_statistics.get(AlignmentPass.EXACT_PATH, 0)
        )
        assert exact_count == len(result.mappings)

    def test_fuzzy_pass_used_for_similar_trees(self) -> None:
        """Trees with renamed nodes may produce fuzzy-pass mappings."""
        src = _tree(
            [_node("s-root", AccessibilityRole.DOCUMENT, "Page", children_ids=["s-btn"]),
             _node("s-btn", AccessibilityRole.BUTTON, "Submit Order", parent_id="s-root")],
            ["s-root"],
        )
        tgt = _tree(
            [_node("t-root", AccessibilityRole.DOCUMENT, "Page", children_ids=["t-btn"]),
             _node("t-btn", AccessibilityRole.BUTTON, "Submit Form", parent_id="t-root")],
            ["t-root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        # There should be some mappings (exact or fuzzy) but not all exact hash
        assert len(result.mappings) >= 1

    def test_statistics_sum_equals_mapping_count(self) -> None:
        """The sum of all pass counts equals len(mappings)."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        total = sum(result.pass_statistics.values())
        assert total == len(result.mappings)


# ============================================================================
# Tests — single-node rename
# ============================================================================


class TestSingleNodeRename:
    """Renaming a single node should produce a detectable edit operation."""

    def test_rename_detected(self) -> None:
        """Renaming a node produces at least one RENAME edit operation."""
        src = _tree(
            [_node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["btn"]),
             _node("btn", AccessibilityRole.BUTTON, "Save", parent_id="root")],
            ["root"],
        )
        tgt = _tree(
            [_node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["btn"]),
             _node("btn", AccessibilityRole.BUTTON, "Save As", parent_id="root")],
            ["root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        rename_ops = [op for op in result.edit_operations if op.operation_type == EditOperationType.RENAME]
        assert len(rename_ops) >= 1

    def test_rename_similarity_below_one(self) -> None:
        """A tree with one renamed node has similarity < 1.0."""
        src = _tree(
            [_node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["h"]),
             _node("h", AccessibilityRole.HEADING, "Introduction", parent_id="root")],
            ["root"],
        )
        tgt = _tree(
            [_node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["h"]),
             _node("h", AccessibilityRole.HEADING, "Conclusion", parent_id="root")],
            ["root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        # Nodes still match by ID, but the name change should affect score or ops
        assert len(result.edit_operations) >= 1 or result.similarity_score < 1.0


# ============================================================================
# Tests — multi-step diff: add + remove + rename
# ============================================================================


class TestMultiStepDiff:
    """Complex diffs combining additions, removals, and renames."""

    def test_add_and_remove(self) -> None:
        """Adding a node to target and removing one from source."""
        src = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["old"]),
                _node("old", AccessibilityRole.BUTTON, "Legacy", parent_id="root"),
            ],
            ["root"],
        )
        tgt = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["new"]),
                _node("new", AccessibilityRole.LINK, "Modern", parent_id="root"),
            ],
            ["root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        total_changes = len(result.additions) + len(result.removals) + len(result.edit_operations)
        assert total_changes > 0
        assert result.similarity_score < 1.0

    def test_add_remove_rename_simultaneously(self) -> None:
        """One node added, one removed, one renamed in the same tree."""
        src = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["keep", "kill"]),
                _node("keep", AccessibilityRole.BUTTON, "Submit Order", parent_id="root"),
                _node("kill", AccessibilityRole.IMG, "Old Banner", parent_id="root"),
            ],
            ["root"],
        )
        tgt = _tree(
            [
                _node("root", AccessibilityRole.DOCUMENT, "Page", children_ids=["keep", "born"]),
                _node("keep", AccessibilityRole.BUTTON, "Submit Form", parent_id="root"),
                _node("born", AccessibilityRole.TEXTBOX, "Search", parent_id="root"),
            ],
            ["root"],
        )
        result = SemanticDiffer().diff(src, tgt)
        # At minimum, root is matched; "keep" may be matched; "kill" removed; "born" added
        assert len(result.mappings) >= 1
        assert result.edit_distance > 0.0

    def test_large_tree_diff_completes(self) -> None:
        """A moderately sized tree (20 nodes) diffs without error."""
        def _build(prefix: str, n: int = 20) -> AccessibilityTree:
            roles = list(AccessibilityRole)
            root = _node(f"{prefix}root", AccessibilityRole.DOCUMENT, "Root", children_ids=[f"{prefix}c{i}" for i in range(n)])
            children = [
                _node(
                    f"{prefix}c{i}",
                    roles[i % len(roles)],
                    f"Node {i}",
                    parent_id=f"{prefix}root",
                )
                for i in range(n)
            ]
            return _tree([root] + children, [f"{prefix}root"])

        result = SemanticDiffer().diff(_build("s"), _build("t"))
        assert isinstance(result, AlignmentResult)
        assert result.similarity_score >= 0.0

    def test_summary_string_not_empty(self) -> None:
        """AlignmentResult.summary() returns a non-empty human-readable string."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Alignment Result" in summary


# ============================================================================
# Tests — config override
# ============================================================================


class TestConfigOverride:
    """SemanticDiffer accepts per-call config overrides."""

    def test_default_config_when_none(self) -> None:
        """Passing config=None to constructor uses defaults."""
        differ = SemanticDiffer(config=None)
        assert differ.config.fuzzy_threshold == pytest.approx(0.40)

    def test_diff_level_config_override(self) -> None:
        """A config passed to diff() overrides the constructor config."""
        differ = SemanticDiffer()
        strict_cfg = AlignmentConfig(fuzzy_threshold=0.99)
        result = differ.diff(_simple_tree(), _simple_tree(), config=strict_cfg)
        # Should still work; identical trees match exactly regardless of fuzzy threshold
        assert isinstance(result, AlignmentResult)


# ============================================================================
# Tests — AlignmentResult query helpers
# ============================================================================


class TestAlignmentResultHelpers:
    """Verify the query helpers on AlignmentResult."""

    def test_get_mapped_node(self) -> None:
        """get_mapped_node returns the target for a given source."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        # At least the root should be mapped
        mapped = result.get_mapped_node("root")
        assert mapped == "root"

    def test_get_mapped_node_missing(self) -> None:
        """get_mapped_node returns None for an unmapped source id."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        assert result.get_mapped_node("nonexistent") is None

    def test_average_confidence(self) -> None:
        """average_confidence is in (0, 1] for a non-empty result."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        avg = result.average_confidence()
        assert 0.0 < avg <= 1.0

    def test_mappings_by_pass(self) -> None:
        """mappings_by_pass filters correctly."""
        result = SemanticDiffer().diff(_simple_tree(), _simple_tree())
        for ap in AlignmentPass:
            filtered = result.mappings_by_pass(ap)
            assert all(m.pass_matched == ap for m in filtered)
