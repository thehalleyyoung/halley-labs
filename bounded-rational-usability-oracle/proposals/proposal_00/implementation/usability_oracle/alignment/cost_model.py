"""
usability_oracle.alignment.cost_model — Edit-distance cost model.

Assigns numerical costs to each :class:`EditOperation` based on:

* **Structural distance** — how far apart the old and new positions are in the
  tree.
* **Content distance** — how different the name / role / properties are.
* **Cognitive impact** — how much the change affects a bounded-rational user's
  ability to complete tasks (interactive widgets weigh more than passive
  structure).

Two aggregate metrics are exposed:

* :meth:`structural_edit_distance` — sum of raw operation costs.
* :meth:`cognitive_edit_distance` — costs re-weighted by cognitive salience.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentResult,
    BoundingBox,
    EditOperation,
    EditOperationType,
    NodeMapping,
    role_taxonomy_distance,
    _ROLE_GROUPS,
)
from usability_oracle.alignment.fuzzy_match import _normalised_levenshtein


# ---------------------------------------------------------------------------
# Cognitive salience helpers
# ---------------------------------------------------------------------------

_INTERACTIVE_ROLES: frozenset[AccessibilityRole] = frozenset({
    AccessibilityRole.BUTTON,
    AccessibilityRole.CHECKBOX,
    AccessibilityRole.COMBOBOX,
    AccessibilityRole.LINK,
    AccessibilityRole.MENU_ITEM,
    AccessibilityRole.RADIO,
    AccessibilityRole.SLIDER,
    AccessibilityRole.SPINBUTTON,
    AccessibilityRole.TAB,
    AccessibilityRole.TEXTBOX,
    AccessibilityRole.OPTION,
})

_LANDMARK_ROLES: frozenset[AccessibilityRole] = frozenset({
    AccessibilityRole.BANNER,
    AccessibilityRole.COMPLEMENTARY,
    AccessibilityRole.CONTENT_INFO,
    AccessibilityRole.FORM,
    AccessibilityRole.MAIN,
    AccessibilityRole.NAVIGATION,
    AccessibilityRole.REGION,
    AccessibilityRole.SEARCH,
})


def _cognitive_multiplier(role: AccessibilityRole, config: AlignmentConfig) -> float:
    """Return a cognitive-weight multiplier for a given role."""
    if role in _INTERACTIVE_ROLES:
        return config.cognitive_weight_interactive
    if role in _LANDMARK_ROLES:
        return config.cognitive_weight_landmark
    return config.cognitive_weight_structure


# ---------------------------------------------------------------------------
# AlignmentCostModel
# ---------------------------------------------------------------------------

class AlignmentCostModel:
    """Computes per-operation costs and aggregate edit distances.

    Parameters
    ----------
    config : AlignmentConfig
        Pipeline-wide configuration.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    # ------------------------------------------------------------------
    # Per-operation cost
    # ------------------------------------------------------------------

    def edit_cost(self, operation: EditOperation,
                  source_tree: Optional[AccessibilityTree] = None,
                  target_tree: Optional[AccessibilityTree] = None) -> float:
        """Dispatch to the specialised cost function for *operation*.

        If *source_tree* / *target_tree* are provided they are used for
        contextual lookups (parent distance, subtree complexity, etc.).
        """
        op = operation.operation_type

        if op == EditOperationType.RENAME:
            old_name = operation.details.get("old_name", "")
            new_name = operation.details.get("new_name", "")
            return self._rename_cost(old_name, new_name)

        if op == EditOperationType.RETYPE:
            old_role_str = operation.details.get("old_role", "unknown")
            new_role_str = operation.details.get("new_role", "unknown")
            return self._retype_cost(old_role_str, new_role_str)

        if op == EditOperationType.MOVE:
            old_parent = operation.details.get("old_parent")
            new_parent = operation.details.get("new_parent")
            tree = source_tree or target_tree
            return self._move_cost(old_parent, new_parent, tree)

        if op == EditOperationType.RESIZE:
            old_bbox = operation.details.get("old_bbox")
            new_bbox = operation.details.get("new_bbox")
            return self._resize_cost(old_bbox, new_bbox)

        if op == EditOperationType.ADD:
            if target_tree and operation.target_node_id:
                node = target_tree.nodes.get(operation.target_node_id)
                if node:
                    return self._add_cost(node)
            return self.config.add_cost_per_node

        if op == EditOperationType.REMOVE:
            if source_tree and operation.source_node_id:
                node = source_tree.nodes.get(operation.source_node_id)
                if node:
                    return self._remove_cost(node)
            return self.config.remove_cost_per_node

        if op == EditOperationType.REORDER:
            return self.config.move_base_cost * 0.5

        if op == EditOperationType.MODIFY_PROPERTY:
            return self.config.rename_base_cost * 0.3

        if op == EditOperationType.RESTRUCTURE:
            return self.config.move_base_cost * 1.5

        return 1.0

    # ------------------------------------------------------------------
    # Specialised cost functions
    # ------------------------------------------------------------------

    def _rename_cost(self, old_name: str, new_name: str) -> float:
        """Cost of renaming: base cost scaled by (1 − name similarity).

        A tiny name tweak (e.g. "Submit" → "Submit Order") is cheap;
        a complete rewrite is expensive.
        """
        sim = _normalised_levenshtein(old_name.lower(), new_name.lower())
        return self.config.rename_base_cost * (1.0 - sim)

    def _retype_cost(self, old_role: str, new_role: str) -> float:
        """Cost of changing a node's role, scaled by taxonomy distance.

        Changing from ``button`` to ``link`` (both interactive widgets) is
        cheaper than changing from ``button`` to ``heading`` (different
        category).
        """
        try:
            role_a = AccessibilityRole(old_role)
            role_b = AccessibilityRole(new_role)
        except ValueError:
            return self.config.retype_base_cost
        dist = role_taxonomy_distance(role_a, role_b)
        return self.config.retype_base_cost * dist

    def _move_cost(
        self,
        old_parent_id: Optional[str],
        new_parent_id: Optional[str],
        tree: Optional[AccessibilityTree],
    ) -> float:
        """Cost of moving a node to a different parent.

        Uses tree-distance between old and new parents when available;
        otherwise falls back to the base cost.
        """
        if old_parent_id == new_parent_id:
            return 0.0
        if tree is None or old_parent_id is None or new_parent_id is None:
            return self.config.move_base_cost

        # Compute tree distance as difference in depths + LCA heuristic
        old_node = tree.nodes.get(old_parent_id)
        new_node = tree.nodes.get(new_parent_id)
        if old_node is None or new_node is None:
            return self.config.move_base_cost

        old_ancestors = self._ancestor_set(old_parent_id, tree)
        new_ancestors = self._ancestor_set(new_parent_id, tree)

        # Symmetric difference in ancestor sets → rough tree distance
        sym_diff = len(old_ancestors.symmetric_difference(new_ancestors))
        normalised = sym_diff / max(len(old_ancestors) + len(new_ancestors), 1)
        return self.config.move_base_cost * (0.3 + 0.7 * normalised)

    @staticmethod
    def _ancestor_set(node_id: str, tree: AccessibilityTree) -> set[str]:
        """Collect all ancestors of *node_id* up to the root."""
        ancestors: set[str] = set()
        current = node_id
        while current:
            ancestors.add(current)
            nd = tree.nodes.get(current)
            if nd is None or nd.parent_id is None:
                break
            current = nd.parent_id
        return ancestors

    def _resize_cost(
        self,
        old_bbox: Optional[dict | BoundingBox],
        new_bbox: Optional[dict | BoundingBox],
    ) -> float:
        """Cost of resizing based on relative area change.

        .. math::

            cost = base \\cdot \\left|1 - \\frac{A_{\\text{new}}}{A_{\\text{old}}}\\right|

        Clamped to ``[0, 2 × base]``.
        """
        if old_bbox is None or new_bbox is None:
            return self.config.resize_base_cost

        def _area(b: dict | BoundingBox) -> float:
            if isinstance(b, BoundingBox):
                return b.area
            return max(0.0, float(b.get("width", 0))) * max(0.0, float(b.get("height", 0)))

        old_area = _area(old_bbox)
        new_area = _area(new_bbox)
        if old_area <= 0:
            return self.config.resize_base_cost

        ratio = abs(1.0 - new_area / old_area)
        return min(self.config.resize_base_cost * ratio, 2.0 * self.config.resize_base_cost)

    def _add_cost(self, node: AccessibilityNode) -> float:
        """Cost of adding a node — proportional to its subtree complexity.

        Interactive widgets cost more to add than structural containers.
        """
        base = self.config.add_cost_per_node
        complexity = math.log2(1 + node.subtree_size)
        multiplier = _cognitive_multiplier(node.role, self.config)
        return base * complexity * (0.5 + 0.5 * multiplier / self.config.cognitive_weight_interactive)

    def _remove_cost(self, node: AccessibilityNode) -> float:
        """Cost of removing a node — similar logic to :meth:`_add_cost`."""
        base = self.config.remove_cost_per_node
        complexity = math.log2(1 + node.subtree_size)
        multiplier = _cognitive_multiplier(node.role, self.config)
        return base * complexity * (0.5 + 0.5 * multiplier / self.config.cognitive_weight_interactive)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def structural_edit_distance(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
        alignment: AlignmentResult,
    ) -> float:
        """Sum of raw edit-operation costs (unweighted).

        Every edit operation in *alignment* is costed and summed.  Matched
        nodes that changed properties also contribute a small cost.
        """
        total = 0.0

        for op in alignment.edit_operations:
            total += self.edit_cost(op, source_tree, target_tree)

        # Matched-but-modified pairs
        for mapping in alignment.mappings:
            s_node = source_tree.nodes.get(mapping.source_id)
            t_node = target_tree.nodes.get(mapping.target_id)
            if s_node is None or t_node is None:
                continue
            total += self._matched_pair_cost(s_node, t_node)

        return total

    def cognitive_edit_distance(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
        alignment: AlignmentResult,
    ) -> float:
        """Cognitively-weighted edit distance.

        Each operation's raw cost is multiplied by the cognitive salience of
        the affected node's role, giving higher weight to changes on
        interactive widgets and landmarks.
        """
        total = 0.0

        for op in alignment.edit_operations:
            raw = self.edit_cost(op, source_tree, target_tree)
            role = self._operation_role(op, source_tree, target_tree)
            total += raw * _cognitive_multiplier(role, self.config)

        for mapping in alignment.mappings:
            s_node = source_tree.nodes.get(mapping.source_id)
            t_node = target_tree.nodes.get(mapping.target_id)
            if s_node is None or t_node is None:
                continue
            pair_cost = self._matched_pair_cost(s_node, t_node)
            role = s_node.role
            total += pair_cost * _cognitive_multiplier(role, self.config)

        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _matched_pair_cost(
        self,
        source: AccessibilityNode,
        target: AccessibilityNode,
    ) -> float:
        """Small residual cost for matched nodes that are not perfectly equal."""
        cost = 0.0

        # Name change within a matched pair
        if source.name != target.name:
            cost += self._rename_cost(source.name, target.name) * 0.5

        # Bounding-box change within a matched pair
        if source.bounding_box and target.bounding_box:
            iou = source.bounding_box.iou(target.bounding_box)
            if iou < 0.95:
                cost += self.config.resize_base_cost * (1.0 - iou) * 0.3

        # Property changes
        src_props = {k: v for k, v in source.properties.items()
                     if k not in ("subtree_size", "child_roles")}
        tgt_props = {k: v for k, v in target.properties.items()
                     if k not in ("subtree_size", "child_roles")}
        if src_props != tgt_props:
            changed_keys = set(src_props.keys()).symmetric_difference(tgt_props.keys())
            changed_keys |= {
                k for k in set(src_props.keys()) & set(tgt_props.keys())
                if src_props[k] != tgt_props[k]
            }
            cost += 0.05 * len(changed_keys)

        return cost

    @staticmethod
    def _operation_role(
        op: EditOperation,
        source_tree: Optional[AccessibilityTree],
        target_tree: Optional[AccessibilityTree],
    ) -> AccessibilityRole:
        """Best-effort role extraction for an edit operation."""
        if op.source_node_id and source_tree:
            node = source_tree.nodes.get(op.source_node_id)
            if node:
                return node.role
        if op.target_node_id and target_tree:
            node = target_tree.nodes.get(op.target_node_id)
            if node:
                return node.role
        # Fallback: try details dict
        role_str = op.details.get("old_role") or op.details.get("new_role")
        if role_str:
            try:
                return AccessibilityRole(role_str)
            except ValueError:
                pass
        return AccessibilityRole.GENERIC
