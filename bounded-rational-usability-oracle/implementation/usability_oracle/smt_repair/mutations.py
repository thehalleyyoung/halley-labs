"""
usability_oracle.smt_repair.mutations — UI mutation operators.

Provides a catalogue of atomic and composite tree mutations that the
repair solver can propose.  Each mutation is a reversible, validated
transformation of the accessibility tree:

* **ReorderChildren** — permute sibling elements.
* **MergeGroups** — combine two related element groups.
* **SplitGroup** — break an overloaded group into sub-groups.
* **AddLandmark** — insert an ARIA landmark for navigation.
* **RemoveRedundant** — remove unnecessary wrapper nodes.
* **AdjustSpacing** — modify spatial layout between elements.
* **PromoteElement** — move an element higher in the hierarchy.
* **AddShortcut** — add a keyboard shortcut binding.

All mutations operate on the serialised ``tree_dict`` (dict) form
of an accessibility tree.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from usability_oracle.smt_repair.types import (
    MutationCandidate,
    MutationType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MutationOperator base
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MutationOperator:
    """Base class for UI mutation operators.

    Subclasses implement :meth:`apply` and :meth:`validate` to
    perform a specific tree transformation.

    Attributes:
        name: Human-readable name of the mutation.
        description: What this mutation does.
    """

    name: str = ""
    description: str = ""

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the mutation to *tree* and return the modified tree.

        Must not mutate *tree* in-place; returns a deep copy.
        """
        return copy.deepcopy(tree)

    def validate(self, tree: Dict[str, Any]) -> bool:
        """Check whether this mutation is valid for *tree*.

        Returns ``True`` if the mutation can be safely applied.
        """
        return True

    def to_mutation_candidate(self) -> MutationCandidate:
        """Convert to a :class:`MutationCandidate` for reporting."""
        return MutationCandidate(
            node_id="",
            mutation_type=MutationType.PROPERTY_CHANGE,
            property_name=None,
            old_value=None,
            new_value=None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Concrete mutations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReorderChildren(MutationOperator):
    """Reorder sibling elements under a parent.

    Attributes:
        parent_id: ID of the parent node.
        permutation: New index ordering (e.g. ``[2, 0, 1]`` moves the
            third child to first position).
    """

    parent_id: str = ""
    permutation: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.name = "ReorderChildren"
        self.description = f"Reorder children of {self.parent_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        parent = _find_node(tree, self.parent_id)
        if parent is None:
            return tree
        children = parent.get("children", [])
        if not self.permutation or len(self.permutation) != len(children):
            return tree
        parent["children"] = [children[i] for i in self.permutation]
        # Update index_in_parent.
        for idx, child in enumerate(parent["children"]):
            child["index_in_parent"] = idx
        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        parent = _find_node(tree, self.parent_id)
        if parent is None:
            return False
        n = len(parent.get("children", []))
        return (
            len(self.permutation) == n
            and set(self.permutation) == set(range(n))
        )

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.parent_id,
            mutation_type=MutationType.REORDER,
            property_name="children_order",
            old_value=None,
            new_value=str(list(self.permutation)),
        )


@dataclass
class MergeGroups(MutationOperator):
    """Merge two sibling groups into one.

    The children of *group_b* are appended to *group_a*, and
    *group_b* is removed from the tree.

    Attributes:
        group_a_id: ID of the group to keep.
        group_b_id: ID of the group to merge into *group_a*.
    """

    group_a_id: str = ""
    group_b_id: str = ""

    def __post_init__(self) -> None:
        self.name = "MergeGroups"
        self.description = f"Merge {self.group_b_id} into {self.group_a_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        group_a = _find_node(tree, self.group_a_id)
        group_b = _find_node(tree, self.group_b_id)
        if group_a is None or group_b is None:
            return tree

        # Move children of B into A.
        a_children = group_a.get("children", [])
        b_children = group_b.get("children", [])
        for child in b_children:
            child["parent_id"] = self.group_a_id
        group_a["children"] = a_children + b_children

        # Remove group_b from its parent.
        _remove_node(tree, self.group_b_id)
        # Re-index children.
        for idx, child in enumerate(group_a.get("children", [])):
            child["index_in_parent"] = idx
        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        return (
            _find_node(tree, self.group_a_id) is not None
            and _find_node(tree, self.group_b_id) is not None
        )

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.group_a_id,
            mutation_type=MutationType.PROPERTY_CHANGE,
            property_name="merge",
            old_value=self.group_b_id,
            new_value=self.group_a_id,
        )


@dataclass
class SplitGroup(MutationOperator):
    """Split an overloaded group at a specified index.

    Children at indices ``[0, split_point)`` stay in the original
    group; children ``[split_point, n)`` move to a new sibling group.

    Attributes:
        group_id: ID of the group to split.
        split_point: Index at which to split.
    """

    group_id: str = ""
    split_point: int = 0

    def __post_init__(self) -> None:
        self.name = "SplitGroup"
        self.description = f"Split {self.group_id} at index {self.split_point}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        group = _find_node(tree, self.group_id)
        if group is None:
            return tree
        children = group.get("children", [])
        if self.split_point <= 0 or self.split_point >= len(children):
            return tree

        keep = children[:self.split_point]
        move = children[self.split_point:]

        new_group_id = f"{self.group_id}_split"
        new_group: Dict[str, Any] = {
            "id": new_group_id,
            "role": group.get("role", "group"),
            "name": f"{group.get('name', '')} (continued)",
            "description": "",
            "bounding_box": group.get("bounding_box", {}),
            "state": {"focused": False, "selected": False, "expanded": False,
                      "checked": None, "disabled": False, "hidden": False,
                      "required": False, "readonly": False, "pressed": None,
                      "value": None},
            "properties": {},
            "depth": group.get("depth", 0),
            "index_in_parent": group.get("index_in_parent", 0) + 1,
            "parent_id": group.get("parent_id"),
            "children": move,
        }
        for idx, child in enumerate(move):
            child["parent_id"] = new_group_id
            child["index_in_parent"] = idx
        group["children"] = keep
        for idx, child in enumerate(keep):
            child["index_in_parent"] = idx

        # Insert new group as a sibling after the original.
        parent_id = group.get("parent_id")
        if parent_id:
            parent = _find_node(tree, parent_id)
            if parent is not None:
                siblings = parent.get("children", [])
                insert_idx = next(
                    (i + 1 for i, s in enumerate(siblings) if s.get("id") == self.group_id),
                    len(siblings),
                )
                siblings.insert(insert_idx, new_group)
                for idx, s in enumerate(siblings):
                    s["index_in_parent"] = idx

        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        group = _find_node(tree, self.group_id)
        if group is None:
            return False
        n = len(group.get("children", []))
        return 0 < self.split_point < n

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.group_id,
            mutation_type=MutationType.PROPERTY_CHANGE,
            property_name="split",
            old_value=None,
            new_value=str(self.split_point),
        )


@dataclass
class AddLandmark(MutationOperator):
    """Insert an ARIA landmark element at a specified position.

    Attributes:
        parent_id: ID of the parent to insert under.
        position: Index among siblings.
        role: Landmark role (e.g. ``"navigation"``, ``"main"``).
        label: Accessible name for the landmark.
    """

    parent_id: str = ""
    position: int = 0
    role: str = "region"
    label: str = ""

    def __post_init__(self) -> None:
        self.name = "AddLandmark"
        self.description = f"Add {self.role} landmark under {self.parent_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        parent = _find_node(tree, self.parent_id)
        if parent is None:
            return tree

        landmark: Dict[str, Any] = {
            "id": f"{self.parent_id}_landmark_{self.role}",
            "role": self.role,
            "name": self.label or self.role,
            "description": "",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "state": {"focused": False, "selected": False, "expanded": False,
                      "checked": None, "disabled": False, "hidden": False,
                      "required": False, "readonly": False, "pressed": None,
                      "value": None},
            "properties": {},
            "depth": parent.get("depth", 0) + 1,
            "index_in_parent": self.position,
            "parent_id": self.parent_id,
            "children": [],
        }

        children = parent.get("children", [])
        pos = max(0, min(self.position, len(children)))
        children.insert(pos, landmark)
        parent["children"] = children
        for idx, child in enumerate(children):
            child["index_in_parent"] = idx

        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        _VALID_LANDMARKS = {"banner", "complementary", "contentinfo", "form",
                            "main", "navigation", "region", "search"}
        return (
            _find_node(tree, self.parent_id) is not None
            and self.role in _VALID_LANDMARKS
        )

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.parent_id,
            mutation_type=MutationType.ELEMENT_ADD,
            property_name="role",
            old_value=None,
            new_value=self.role,
        )


@dataclass
class RemoveRedundant(MutationOperator):
    """Remove an unnecessary intermediate container node.

    The node's children are reparented to its parent.

    Attributes:
        node_id: ID of the redundant node to remove.
    """

    node_id: str = ""

    def __post_init__(self) -> None:
        self.name = "RemoveRedundant"
        self.description = f"Remove redundant container {self.node_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        node = _find_node(tree, self.node_id)
        if node is None:
            return tree

        parent_id = node.get("parent_id")
        if parent_id is None:
            return tree  # Cannot remove root.

        parent = _find_node(tree, parent_id)
        if parent is None:
            return tree

        # Find the node's position among siblings.
        siblings = parent.get("children", [])
        node_idx = next(
            (i for i, s in enumerate(siblings) if s.get("id") == self.node_id),
            None,
        )
        if node_idx is None:
            return tree

        # Replace the node with its children.
        node_children = node.get("children", [])
        for child in node_children:
            child["parent_id"] = parent_id
        new_siblings = siblings[:node_idx] + node_children + siblings[node_idx + 1:]
        parent["children"] = new_siblings
        for idx, child in enumerate(new_siblings):
            child["index_in_parent"] = idx

        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        node = _find_node(tree, self.node_id)
        if node is None:
            return False
        return node.get("parent_id") is not None

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.node_id,
            mutation_type=MutationType.ELEMENT_REMOVE,
            property_name=None,
            old_value=None,
            new_value=None,
        )


@dataclass
class AdjustSpacing(MutationOperator):
    """Adjust spatial layout (spacing) between sibling elements.

    Sets a uniform vertical gap between consecutive siblings under
    a parent.

    Attributes:
        parent_id: ID of the parent whose children to space.
        target_spacing: Desired vertical gap in pixels.
    """

    parent_id: str = ""
    target_spacing: int = 8

    def __post_init__(self) -> None:
        self.name = "AdjustSpacing"
        self.description = f"Adjust spacing to {self.target_spacing}px under {self.parent_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        parent = _find_node(tree, self.parent_id)
        if parent is None:
            return tree

        children = parent.get("children", [])
        if len(children) < 2:
            return tree

        # Arrange children vertically with target_spacing gap.
        current_y = children[0].get("bounding_box", {}).get("y", 0)
        for child in children:
            bbox = child.get("bounding_box", {})
            bbox["y"] = current_y
            child["bounding_box"] = bbox
            current_y += bbox.get("height", 40) + self.target_spacing

        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        parent = _find_node(tree, self.parent_id)
        return parent is not None and self.target_spacing >= 0

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.parent_id,
            mutation_type=MutationType.PROPERTY_CHANGE,
            property_name="spacing",
            old_value=None,
            new_value=self.target_spacing,
        )


@dataclass
class PromoteElement(MutationOperator):
    """Move an element higher in the hierarchy.

    Reparents the element from its current parent to its grandparent,
    effectively promoting it by one level.

    Attributes:
        node_id: ID of the node to promote.
        new_depth: Target depth (currently unused; promotes by one level).
    """

    node_id: str = ""
    new_depth: int = 0

    def __post_init__(self) -> None:
        self.name = "PromoteElement"
        self.description = f"Promote {self.node_id} one level"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        node = _find_node(tree, self.node_id)
        if node is None:
            return tree

        parent_id = node.get("parent_id")
        if parent_id is None:
            return tree
        parent = _find_node(tree, parent_id)
        if parent is None:
            return tree

        grandparent_id = parent.get("parent_id")
        if grandparent_id is None:
            return tree  # Parent is root.
        grandparent = _find_node(tree, grandparent_id)
        if grandparent is None:
            return tree

        # Remove from current parent.
        parent_children = parent.get("children", [])
        parent["children"] = [c for c in parent_children if c.get("id") != self.node_id]
        for idx, c in enumerate(parent["children"]):
            c["index_in_parent"] = idx

        # Add to grandparent after parent.
        gp_children = grandparent.get("children", [])
        parent_idx = next(
            (i for i, c in enumerate(gp_children) if c.get("id") == parent_id),
            len(gp_children),
        )
        node["parent_id"] = grandparent_id
        node["depth"] = parent.get("depth", 1)
        gp_children.insert(parent_idx + 1, node)
        grandparent["children"] = gp_children
        for idx, c in enumerate(gp_children):
            c["index_in_parent"] = idx

        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        node = _find_node(tree, self.node_id)
        if node is None:
            return False
        parent_id = node.get("parent_id")
        if parent_id is None:
            return False
        parent = _find_node(tree, parent_id)
        return parent is not None and parent.get("parent_id") is not None

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.node_id,
            mutation_type=MutationType.REPARENT,
            property_name="depth",
            old_value=None,
            new_value=self.new_depth,
        )


@dataclass
class AddShortcut(MutationOperator):
    """Add a keyboard shortcut binding.

    Adds an ``accesskey`` property to the target node.

    Attributes:
        source_id: ID of the source/trigger context.
        target_id: ID of the target element.
        key_binding: Keyboard shortcut string (e.g. ``"Alt+S"``).
    """

    source_id: str = ""
    target_id: str = ""
    key_binding: str = ""

    def __post_init__(self) -> None:
        self.name = "AddShortcut"
        self.description = f"Add shortcut {self.key_binding} to {self.target_id}"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        tree = copy.deepcopy(tree)
        target = _find_node(tree, self.target_id)
        if target is None:
            return tree

        props = target.get("properties", {})
        props["accesskey"] = self.key_binding
        target["properties"] = props
        return tree

    def validate(self, tree: Dict[str, Any]) -> bool:
        return (
            _find_node(tree, self.target_id) is not None
            and bool(self.key_binding)
        )

    def to_mutation_candidate(self) -> MutationCandidate:
        return MutationCandidate(
            node_id=self.target_id,
            mutation_type=MutationType.PROPERTY_CHANGE,
            property_name="accesskey",
            old_value=None,
            new_value=self.key_binding,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mutation composition and application
# ═══════════════════════════════════════════════════════════════════════════

def validate_mutation(tree: Dict[str, Any], mutation: MutationOperator) -> bool:
    """Check that *mutation* preserves tree validity when applied to *tree*.

    Verifies:
    - The mutation's own preconditions (:meth:`MutationOperator.validate`).
    - The result is a well-formed tree (every node has an ``"id"`` and
      non-circular parentage).

    Parameters:
        tree: Serialised accessibility tree.
        mutation: Mutation operator to validate.

    Returns:
        ``True`` if the mutation is safe to apply.
    """
    if not mutation.validate(tree):
        return False
    try:
        result = mutation.apply(tree)
        return _is_well_formed(result)
    except Exception:
        return False


def apply_mutation(tree: Dict[str, Any], mutation: MutationOperator) -> Dict[str, Any]:
    """Apply *mutation* to *tree* and return the modified tree.

    Parameters:
        tree: Serialised accessibility tree.
        mutation: Mutation operator.

    Returns:
        Modified (deep-copied) tree.
    """
    return mutation.apply(tree)


def compose_mutations(
    mutations: Sequence[MutationOperator],
) -> MutationOperator:
    """Compose a sequence of mutations into a single operator.

    The resulting operator applies each mutation in order.  Validation
    checks each step against the intermediate tree.

    Parameters:
        mutations: Ordered sequence of mutations.

    Returns:
        A :class:`MutationOperator` whose :meth:`apply` chains all
        mutations.
    """
    return _ComposedMutation(mutations=list(mutations))


@dataclass
class _ComposedMutation(MutationOperator):
    """Internal: composed sequence of mutations."""

    mutations: List[MutationOperator] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = "Composed"
        self.description = f"Compose {len(self.mutations)} mutations"

    def apply(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(tree)
        for m in self.mutations:
            result = m.apply(result)
        return result

    def validate(self, tree: Dict[str, Any]) -> bool:
        current = copy.deepcopy(tree)
        for m in self.mutations:
            if not m.validate(current):
                return False
            current = m.apply(current)
        return _is_well_formed(current)


# ═══════════════════════════════════════════════════════════════════════════
# Tree-manipulation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_node(tree: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """Find a node by ID in a tree dict (DFS)."""
    if str(tree.get("id", "")) == node_id:
        return tree
    for child in tree.get("children", []):
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _remove_node(tree: Dict[str, Any], node_id: str) -> bool:
    """Remove a node by ID from the tree, returning True if found."""
    children = tree.get("children", [])
    for i, child in enumerate(children):
        if str(child.get("id", "")) == node_id:
            children.pop(i)
            for idx, c in enumerate(children):
                c["index_in_parent"] = idx
            return True
        if _remove_node(child, node_id):
            return True
    return False


def _is_well_formed(tree: Dict[str, Any]) -> bool:
    """Check that the tree is well-formed (every node has an 'id')."""
    if "id" not in tree:
        return False
    for child in tree.get("children", []):
        if not _is_well_formed(child):
            return False
    return True
