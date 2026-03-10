"""Unit tests for usability_oracle.smt_repair.mutations — UI mutation operators.

Tests ReorderChildren, MergeGroups, SplitGroup, validate_mutation,
apply_mutation, and compose_mutations.
"""

from __future__ import annotations

import copy

import pytest

from usability_oracle.smt_repair.mutations import (
    AddLandmark,
    AddShortcut,
    AdjustSpacing,
    MergeGroups,
    MutationOperator,
    PromoteElement,
    ReorderChildren,
    SplitGroup,
    apply_mutation,
    compose_mutations,
    validate_mutation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, role: str = "button", children: list | None = None,
          parent_id: str | None = None) -> dict:
    return {
        "id": nid, "role": role, "name": nid,
        "bounding_box": {"x": 0, "y": 0, "width": 80, "height": 30},
        "state": {"hidden": False}, "properties": {},
        "parent_id": parent_id, "depth": 0, "index_in_parent": 0,
        "children": children or [],
    }


def _simple_tree() -> dict:
    """Root with three children."""
    c0 = _node("c0", parent_id="root")
    c1 = _node("c1", parent_id="root")
    c2 = _node("c2", parent_id="root")
    for i, c in enumerate([c0, c1, c2]):
        c["index_in_parent"] = i
    root = _node("root", "list", children=[c0, c1, c2])
    return root


def _grouped_tree() -> dict:
    """Root with two group children, each containing items."""
    g1_items = [_node("i1", parent_id="g1"), _node("i2", parent_id="g1")]
    g2_items = [_node("i3", parent_id="g2"), _node("i4", parent_id="g2")]
    for i, c in enumerate(g1_items):
        c["index_in_parent"] = i
    for i, c in enumerate(g2_items):
        c["index_in_parent"] = i
    g1 = _node("g1", "group", children=g1_items, parent_id="root")
    g2 = _node("g2", "group", children=g2_items, parent_id="root")
    g1["index_in_parent"] = 0
    g2["index_in_parent"] = 1
    root = _node("root", "list", children=[g1, g2])
    return root


def _count_nodes(tree: dict) -> int:
    count = 1
    for child in tree.get("children", []):
        count += _count_nodes(child)
    return count


def _all_ids(tree: dict) -> set:
    ids = {tree["id"]}
    for child in tree.get("children", []):
        ids |= _all_ids(child)
    return ids


# ===================================================================
# ReorderChildren
# ===================================================================


class TestReorderChildren:

    def test_produces_valid_permutation(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(2, 0, 1))
        result = mutation.apply(tree)
        new_ids = [c["id"] for c in result["children"]]
        assert new_ids == ["c2", "c0", "c1"]

    def test_preserves_all_children(self):
        tree = _simple_tree()
        original_ids = {c["id"] for c in tree["children"]}
        mutation = ReorderChildren(parent_id="root", permutation=(1, 2, 0))
        result = mutation.apply(tree)
        result_ids = {c["id"] for c in result["children"]}
        assert result_ids == original_ids

    def test_identity_permutation_no_change(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(0, 1, 2))
        result = mutation.apply(tree)
        assert [c["id"] for c in result["children"]] == ["c0", "c1", "c2"]

    def test_validate_rejects_invalid_permutation(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(0, 0, 1))
        assert not mutation.validate(tree)

    def test_validate_rejects_wrong_length(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(0, 1))
        assert not mutation.validate(tree)

    def test_validate_rejects_nonexistent_parent(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="nope", permutation=(0, 1, 2))
        assert not mutation.validate(tree)

    def test_does_not_mutate_original(self):
        tree = _simple_tree()
        original = copy.deepcopy(tree)
        ReorderChildren(parent_id="root", permutation=(2, 0, 1)).apply(tree)
        assert tree == original


# ===================================================================
# MergeGroups
# ===================================================================


class TestMergeGroups:

    def test_reduces_node_count(self):
        tree = _grouped_tree()
        before = _count_nodes(tree)
        mutation = MergeGroups(group_a_id="g1", group_b_id="g2")
        result = mutation.apply(tree)
        after = _count_nodes(result)
        assert after < before

    def test_children_moved_to_target_group(self):
        tree = _grouped_tree()
        mutation = MergeGroups(group_a_id="g1", group_b_id="g2")
        result = mutation.apply(tree)
        # g1 should now have 4 children
        g1 = next(c for c in result["children"] if c["id"] == "g1")
        assert len(g1["children"]) == 4

    def test_validate_rejects_missing_group(self):
        tree = _grouped_tree()
        mutation = MergeGroups(group_a_id="g1", group_b_id="nonexistent")
        assert not mutation.validate(tree)

    def test_preserves_all_leaf_ids(self):
        tree = _grouped_tree()
        original_ids = _all_ids(tree)
        mutation = MergeGroups(group_a_id="g1", group_b_id="g2")
        result = mutation.apply(tree)
        result_ids = _all_ids(result)
        # g2 removed but its children should be in g1
        assert {"i1", "i2", "i3", "i4"}.issubset(result_ids)


# ===================================================================
# SplitGroup
# ===================================================================


class TestSplitGroup:

    def test_increases_group_count(self):
        tree = _grouped_tree()
        # g1 has 2 children; split at index 1
        mutation = SplitGroup(group_id="g1", split_point=1)
        result = mutation.apply(tree)
        # root should now have 3 children (g1, g1_split, g2)
        assert len(result["children"]) == 3

    def test_validate_rejects_split_at_zero(self):
        tree = _grouped_tree()
        mutation = SplitGroup(group_id="g1", split_point=0)
        assert not mutation.validate(tree)

    def test_validate_rejects_split_beyond_length(self):
        tree = _grouped_tree()
        mutation = SplitGroup(group_id="g1", split_point=10)
        assert not mutation.validate(tree)

    def test_preserves_total_children(self):
        tree = _grouped_tree()
        g1_count = len(tree["children"][0]["children"])
        mutation = SplitGroup(group_id="g1", split_point=1)
        result = mutation.apply(tree)
        g1_new = next(c for c in result["children"] if c["id"] == "g1")
        g1_split = next(c for c in result["children"] if c["id"] == "g1_split")
        assert len(g1_new["children"]) + len(g1_split["children"]) == g1_count

    def test_validate_rejects_nonexistent_group(self):
        tree = _grouped_tree()
        mutation = SplitGroup(group_id="nope", split_point=1)
        assert not mutation.validate(tree)


# ===================================================================
# validate_mutation
# ===================================================================


class TestValidateMutation:

    def test_valid_mutation_returns_true(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(2, 1, 0))
        assert validate_mutation(tree, mutation) is True

    def test_invalid_mutation_returns_false(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(0, 0, 0))
        assert validate_mutation(tree, mutation) is False

    def test_rejects_mutation_that_breaks_well_formedness(self):
        # A mutation on a tree without ids should fail well-formedness check
        broken_tree = {"role": "generic", "children": []}
        mutation = MutationOperator()
        assert validate_mutation(broken_tree, mutation) is False


# ===================================================================
# apply_mutation
# ===================================================================


class TestApplyMutation:

    def test_preserves_tree_structure(self):
        tree = _simple_tree()
        mutation = ReorderChildren(parent_id="root", permutation=(1, 0, 2))
        result = apply_mutation(tree, mutation)
        assert "id" in result
        assert "children" in result
        assert len(result["children"]) == 3

    def test_does_not_modify_original(self):
        tree = _simple_tree()
        original = copy.deepcopy(tree)
        apply_mutation(tree, ReorderChildren(parent_id="root", permutation=(2, 0, 1)))
        assert tree == original


# ===================================================================
# compose_mutations
# ===================================================================


class TestComposeMutations:

    def test_applies_in_order(self):
        tree = _simple_tree()
        m1 = ReorderChildren(parent_id="root", permutation=(2, 0, 1))
        m2 = ReorderChildren(parent_id="root", permutation=(2, 0, 1))
        composed = compose_mutations([m1, m2])
        result = composed.apply(tree)
        # Two rotations: [c0,c1,c2] → [c2,c0,c1] → [c1,c2,c0]
        ids = [c["id"] for c in result["children"]]
        assert ids == ["c1", "c2", "c0"]

    def test_empty_composition_is_identity(self):
        tree = _simple_tree()
        composed = compose_mutations([])
        result = composed.apply(tree)
        # Should be equivalent to original
        assert [c["id"] for c in result["children"]] == ["c0", "c1", "c2"]

    def test_composed_validate_checks_each_step(self):
        tree = _simple_tree()
        valid = ReorderChildren(parent_id="root", permutation=(2, 0, 1))
        invalid = ReorderChildren(parent_id="nonexistent", permutation=(0, 1, 2))
        composed = compose_mutations([valid, invalid])
        assert not composed.validate(tree)

    def test_single_mutation_composition(self):
        tree = _simple_tree()
        m = ReorderChildren(parent_id="root", permutation=(1, 2, 0))
        composed = compose_mutations([m])
        result = composed.apply(tree)
        direct = m.apply(tree)
        assert [c["id"] for c in result["children"]] == [c["id"] for c in direct["children"]]


# ===================================================================
# Mutation candidates
# ===================================================================


class TestMutationCandidate:

    def test_reorder_to_candidate(self):
        m = ReorderChildren(parent_id="p1", permutation=(1, 0))
        c = m.to_mutation_candidate()
        assert c.node_id == "p1"

    def test_merge_to_candidate(self):
        m = MergeGroups(group_a_id="g1", group_b_id="g2")
        c = m.to_mutation_candidate()
        assert c.node_id == "g1"

    def test_split_to_candidate(self):
        m = SplitGroup(group_id="g1", split_point=2)
        c = m.to_mutation_candidate()
        assert c.node_id == "g1"
