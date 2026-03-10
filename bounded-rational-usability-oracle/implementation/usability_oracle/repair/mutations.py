"""
usability_oracle.repair.mutations — Mutation operators for accessibility trees.

Provides :class:`MutationOperator` which applies :class:`UIMutation`
instances to :class:`AccessibilityTree` objects.  All operations return
a **new** tree (immutable semantics via deep copy), leaving the original
unchanged.

Supported mutations:
  - resize — change bounding-box dimensions
  - reposition — move node to new (x, y)
  - regroup — collect nodes under a new parent
  - relabel — rename a node
  - remove — delete a node and reparent children
  - add_shortcut — attach keyboard shortcut metadata
  - simplify_menu — prune a menu to *max_items*
  - add_landmark — wrap region nodes in a landmark role
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.repair.models import MutationType, UIMutation

logger = logging.getLogger(__name__)


class MutationOperator:
    """Apply UI mutations to accessibility trees.

    Every public method returns a **new** tree (deep-copy semantics).
    The original tree is never modified.

    Usage::

        op = MutationOperator()
        new_tree = op.apply(tree, mutation)
    """

    # ── Dispatch ----------------------------------------------------------

    _DISPATCH: dict[str, str] = {
        MutationType.RESIZE: "_resize",
        MutationType.REPOSITION: "_reposition",
        MutationType.REGROUP: "_regroup",
        MutationType.RELABEL: "_relabel",
        MutationType.REMOVE: "_remove_node",
        MutationType.ADD_SHORTCUT: "_add_shortcut",
        MutationType.SIMPLIFY_MENU: "_simplify_menu",
        MutationType.ADD_LANDMARK: "_add_landmark",
    }

    def apply(
        self, tree: AccessibilityTree, mutation: UIMutation
    ) -> AccessibilityTree:
        """Apply a single mutation and return the new tree.

        Parameters
        ----------
        tree : AccessibilityTree
            The original (unmodified) tree.
        mutation : UIMutation
            Mutation to apply.

        Returns
        -------
        AccessibilityTree
            Deep copy with the mutation applied.

        Raises
        ------
        ValueError
            If the mutation type is unknown or the target node is missing.
        """
        method_name = self._DISPATCH.get(mutation.mutation_type)
        if method_name is None:
            raise ValueError(f"Unknown mutation type: {mutation.mutation_type!r}")

        new_tree = copy.deepcopy(tree)
        new_tree.build_index()

        method = getattr(self, method_name)
        result = method(new_tree, mutation.target_node_id, **mutation.parameters)
        result.build_index()
        return result

    def apply_all(
        self, tree: AccessibilityTree, mutations: list[UIMutation]
    ) -> AccessibilityTree:
        """Apply a sequence of mutations in order."""
        current = tree
        for m in mutations:
            current = self.apply(current, m)
        return current

    # ── Resize ------------------------------------------------------------

    def _resize(
        self,
        tree: AccessibilityTree,
        node_id: str,
        width: float | None = None,
        height: float | None = None,
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Change the bounding box dimensions of a node.

        If the node has no bounding box, one is created at (0, 0).
        Only the specified dimensions are changed; the other is preserved.
        """
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        if node.bounding_box is None:
            node.bounding_box = BoundingBox(x=0, y=0, width=0, height=0)

        if width is not None:
            node.bounding_box.width = max(1.0, float(width))
        if height is not None:
            node.bounding_box.height = max(1.0, float(height))

        logger.debug(
            "Resized %s to %.0f × %.0f",
            node_id,
            node.bounding_box.width,
            node.bounding_box.height,
        )
        return tree

    # ── Reposition --------------------------------------------------------

    def _reposition(
        self,
        tree: AccessibilityTree,
        node_id: str,
        x: float | None = None,
        y: float | None = None,
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Move a node to new screen coordinates.

        Preserves the bounding-box dimensions; only changes the origin.
        """
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        if node.bounding_box is None:
            node.bounding_box = BoundingBox(x=0, y=0, width=44, height=44)

        if x is not None:
            node.bounding_box.x = max(0.0, float(x))
        if y is not None:
            node.bounding_box.y = max(0.0, float(y))

        logger.debug("Repositioned %s to (%.0f, %.0f)", node_id, node.bounding_box.x, node.bounding_box.y)
        return tree

    # ── Regroup -----------------------------------------------------------

    def _regroup(
        self,
        tree: AccessibilityTree,
        node_id: str,
        node_ids: list[str] | None = None,
        new_parent_role: str = "group",
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Collect specified nodes under a new group parent.

        Creates a new node with the given role and moves the target nodes
        to be children of this new group.  The group is inserted as a child
        of the lowest common ancestor of the target nodes.

        Parameters
        ----------
        tree : AccessibilityTree
        node_id : str
            Primary target node (also included in the group).
        node_ids : list[str], optional
            Additional nodes to include.  If None, only *node_id*.
        new_parent_role : str
            Accessibility role for the new group node.
        """
        targets = set(node_ids or [])
        targets.add(node_id)

        target_nodes = []
        for tid in targets:
            n = tree.get_node(tid)
            if n is not None:
                target_nodes.append(n)

        if len(target_nodes) < 1:
            raise ValueError("No valid target nodes for regroup")

        # Find common parent
        parent_ids = set()
        for tn in target_nodes:
            if tn.parent_id:
                parent_ids.add(tn.parent_id)

        # Use the first common parent or the root
        common_parent_id = None
        if len(parent_ids) == 1:
            common_parent_id = parent_ids.pop()
        else:
            common_parent_id = tree.root.id

        parent_node = tree.get_node(common_parent_id)
        if parent_node is None:
            parent_node = tree.root

        # Create new group node
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        group_node = AccessibilityNode(
            id=group_id,
            role=new_parent_role,
            name=f"Group: {new_parent_role}",
            state=AccessibilityState(),
            children=[],
            parent_id=parent_node.id,
        )

        # Move target nodes from parent to group
        target_id_set = {tn.id for tn in target_nodes}
        remaining_children = []
        for child in parent_node.children:
            if child.id in target_id_set:
                child.parent_id = group_id
                group_node.children.append(child)
            else:
                remaining_children.append(child)

        remaining_children.append(group_node)
        parent_node.children = remaining_children

        # Compute group bounding box as union
        bboxes = [n.bounding_box for n in target_nodes if n.bounding_box]
        if bboxes:
            min_x = min(b.x for b in bboxes)
            min_y = min(b.y for b in bboxes)
            max_r = max(b.x + b.width for b in bboxes)
            max_b = max(b.y + b.height for b in bboxes)
            group_node.bounding_box = BoundingBox(
                x=min_x, y=min_y, width=max_r - min_x, height=max_b - min_y
            )

        logger.debug(
            "Regrouped %d nodes under %s (%s)",
            len(target_nodes), group_id, new_parent_role,
        )
        return tree

    # ── Relabel -----------------------------------------------------------

    def _relabel(
        self,
        tree: AccessibilityTree,
        node_id: str,
        new_name: str = "",
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Change the accessible name of a node."""
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        old_name = node.name
        node.name = new_name
        logger.debug("Relabelled %s: %r → %r", node_id, old_name, new_name)
        return tree

    # ── Remove ------------------------------------------------------------

    def _remove_node(
        self,
        tree: AccessibilityTree,
        node_id: str,
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Remove a node, reparenting its children to its parent.

        The root node cannot be removed.
        """
        if node_id == tree.root.id:
            raise ValueError("Cannot remove root node")

        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        parent = tree.get_node(node.parent_id) if node.parent_id else None
        if parent is None:
            raise ValueError(f"Node {node_id!r} has no parent")

        # Find position in parent's children
        insert_idx = 0
        new_children = []
        for i, child in enumerate(parent.children):
            if child.id == node_id:
                insert_idx = i
                # Reparent grandchildren
                for grandchild in node.children:
                    grandchild.parent_id = parent.id
                    new_children.append(grandchild)
            else:
                new_children.append(child)

        parent.children = new_children

        logger.debug(
            "Removed %s, reparented %d children to %s",
            node_id, len(node.children), parent.id,
        )
        return tree

    # ── Add Shortcut ------------------------------------------------------

    def _add_shortcut(
        self,
        tree: AccessibilityTree,
        node_id: str,
        shortcut_key: str = "",
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Attach a keyboard shortcut to a node.

        Stores the shortcut in the node's ``properties`` dict under
        ``"keyboard_shortcut"``.  Also adds an ``"accesskey"`` property
        for compatibility with ARIA ``aria-keyshortcuts``.
        """
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        if not shortcut_key:
            raise ValueError("shortcut_key must be non-empty")

        node.properties["keyboard_shortcut"] = shortcut_key
        node.properties["accesskey"] = shortcut_key

        # Update description to mention the shortcut
        if shortcut_key not in node.description:
            if node.description:
                node.description += f" (shortcut: {shortcut_key})"
            else:
                node.description = f"Shortcut: {shortcut_key}"

        logger.debug("Added shortcut %r to %s", shortcut_key, node_id)
        return tree

    # ── Simplify Menu -----------------------------------------------------

    def _simplify_menu(
        self,
        tree: AccessibilityTree,
        node_id: str,
        max_items: int = 7,
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Reduce a menu's children to at most *max_items*.

        Items beyond the limit are grouped into a "More…" submenu.
        The most "important" items (those with interactive roles) are
        kept at the top level; the rest are moved to the overflow.
        """
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        if len(node.children) <= max_items:
            return tree  # already within limit

        max_items = max(1, max_items)

        # Prioritise interactive children
        interactive = [c for c in node.children if c.is_interactive()]
        non_interactive = [c for c in node.children if not c.is_interactive()]

        # Keep top interactive items + fill with non-interactive
        keep = interactive[:max_items - 1]
        overflow = interactive[max_items - 1:] + non_interactive
        if len(keep) < max_items - 1:
            extra = non_interactive[: max_items - 1 - len(keep)]
            keep.extend(extra)
            overflow = [c for c in overflow if c not in keep]

        # Create "More…" submenu
        more_id = f"more_{uuid.uuid4().hex[:8]}"
        more_node = AccessibilityNode(
            id=more_id,
            role="menu",
            name="More…",
            state=AccessibilityState(),
            children=overflow,
            parent_id=node.id,
        )
        for child in overflow:
            child.parent_id = more_id

        keep.append(more_node)
        node.children = keep

        logger.debug(
            "Simplified menu %s: %d visible + %d in overflow",
            node_id, len(keep) - 1, len(overflow),
        )
        return tree

    # ── Add Landmark ------------------------------------------------------

    def _add_landmark(
        self,
        tree: AccessibilityTree,
        node_id: str,
        region_ids: list[str] | None = None,
        landmark_role: str = "region",
        **kwargs: Any,
    ) -> AccessibilityTree:
        """Wrap nodes in a landmark region.

        Creates a new node with the specified landmark *role* (e.g.
        ``"navigation"``, ``"region"``, ``"main"``) and makes the target
        nodes its children.  The landmark is inserted in place of the
        first target node within its parent.

        Parameters
        ----------
        tree : AccessibilityTree
        node_id : str
            Primary target node.
        region_ids : list[str], optional
            Additional nodes to wrap.
        landmark_role : str
            ARIA landmark role for the wrapper node.
        """
        targets = set(region_ids or [])
        targets.add(node_id)

        target_nodes = []
        for tid in targets:
            n = tree.get_node(tid)
            if n is not None:
                target_nodes.append(n)

        if not target_nodes:
            raise ValueError("No valid target nodes for landmark")

        # Determine insertion parent
        primary = tree.get_node(node_id)
        if primary is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        parent = tree.get_node(primary.parent_id) if primary.parent_id else tree.root
        if parent is None:
            parent = tree.root

        # Create landmark wrapper
        landmark_id = f"landmark_{uuid.uuid4().hex[:8]}"
        landmark_node = AccessibilityNode(
            id=landmark_id,
            role=landmark_role,
            name=f"{landmark_role.title()} Region",
            state=AccessibilityState(),
            children=[],
            parent_id=parent.id,
        )

        # Move target nodes into landmark
        target_id_set = {tn.id for tn in target_nodes}
        new_parent_children = []
        landmark_inserted = False

        for child in parent.children:
            if child.id in target_id_set:
                child.parent_id = landmark_id
                landmark_node.children.append(child)
                if not landmark_inserted:
                    new_parent_children.append(landmark_node)
                    landmark_inserted = True
            else:
                new_parent_children.append(child)

        if not landmark_inserted:
            new_parent_children.append(landmark_node)

        parent.children = new_parent_children

        # Compute landmark bounding box
        bboxes = [n.bounding_box for n in target_nodes if n.bounding_box]
        if bboxes:
            min_x = min(b.x for b in bboxes)
            min_y = min(b.y for b in bboxes)
            max_r = max(b.x + b.width for b in bboxes)
            max_b = max(b.y + b.height for b in bboxes)
            landmark_node.bounding_box = BoundingBox(
                x=min_x, y=min_y, width=max_r - min_x, height=max_b - min_y
            )

        logger.debug(
            "Added %s landmark wrapping %d nodes",
            landmark_role, len(target_nodes),
        )
        return tree
