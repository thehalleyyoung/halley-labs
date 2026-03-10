"""
usability_oracle.android_a11y.normalizer — Normalize Android view hierarchies.

Cleans up raw Android view hierarchy dumps by removing system UI,
flattening wrapper layers, merging compound views, computing visibility,
and assigning navigation order.

Reference: Android Accessibility Best Practices,
           Android View Hierarchy documentation.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Sequence

from usability_oracle.android_a11y.types import (
    AndroidClassName,
    AndroidNode,
    BoundsInfo,
    ContentDescription,
    ViewHierarchy,
)
from usability_oracle.android_a11y.protocols import HierarchyNormalizer as HierarchyNormalizerProtocol


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# System UI classes to remove
_SYSTEM_UI_CLASSES: FrozenSet[str] = frozenset({
    "com.android.systemui.statusbar.StatusBarWindowView",
    "com.android.systemui.statusbar.phone.StatusBarWindowView",
    "com.android.systemui.navigationbar.NavigationBarView",
    "com.android.systemui.statusbar.phone.NavigationBarView",
})

# Packages considered system UI
_SYSTEM_UI_PACKAGES: FrozenSet[str] = frozenset({
    "com.android.systemui",
})

# DecorView wrapper classes
_DECOR_VIEW_CLASSES: FrozenSet[str] = frozenset({
    "com.android.internal.policy.DecorView",
    "com.android.internal.policy.PhoneWindow$DecorView",
    "android.view.ViewRootImpl",
})

# Layout-only classes that can be merged if they have a single child
_LAYOUT_CLASSES: FrozenSet[str] = frozenset({
    AndroidClassName.FRAME_LAYOUT.value,
    AndroidClassName.LINEAR_LAYOUT.value,
    AndroidClassName.RELATIVE_LAYOUT.value,
    AndroidClassName.VIEW_GROUP.value,
    "android.widget.FrameLayout",
    "android.widget.LinearLayout",
    "android.widget.RelativeLayout",
    "android.view.ViewGroup",
    "androidx.appcompat.widget.ContentFrameLayout",
    "androidx.appcompat.widget.FitWindowsLinearLayout",
    "androidx.appcompat.widget.ActionBarOverlayLayout",
    "androidx.constraintlayout.widget.ConstraintLayout",
    "androidx.cardview.widget.CardView",
    "com.google.android.material.card.MaterialCardView",
})

# Android class → ARIA-like role mapping
_CLASS_TO_ROLE: Dict[str, str] = {
    AndroidClassName.VIEW.value: "generic",
    AndroidClassName.TEXT_VIEW.value: "text",
    AndroidClassName.IMAGE_VIEW.value: "img",
    AndroidClassName.BUTTON.value: "button",
    AndroidClassName.IMAGE_BUTTON.value: "button",
    AndroidClassName.EDIT_TEXT.value: "textbox",
    AndroidClassName.CHECK_BOX.value: "checkbox",
    AndroidClassName.RADIO_BUTTON.value: "radio",
    AndroidClassName.TOGGLE_BUTTON.value: "switch",
    AndroidClassName.SWITCH.value: "switch",
    AndroidClassName.SEEK_BAR.value: "slider",
    AndroidClassName.SPINNER.value: "combobox",
    AndroidClassName.PROGRESS_BAR.value: "progressbar",
    AndroidClassName.SCROLL_VIEW.value: "region",
    AndroidClassName.RECYCLER_VIEW.value: "list",
    AndroidClassName.WEB_VIEW.value: "document",
    AndroidClassName.VIEW_GROUP.value: "group",
    AndroidClassName.LINEAR_LAYOUT.value: "group",
    AndroidClassName.RELATIVE_LAYOUT.value: "group",
    AndroidClassName.FRAME_LAYOUT.value: "group",
    AndroidClassName.TAB_WIDGET.value: "tablist",
}


# ═══════════════════════════════════════════════════════════════════════════
# AndroidHierarchyNormalizer
# ═══════════════════════════════════════════════════════════════════════════

class AndroidHierarchyNormalizer:
    """Normalize an Android view hierarchy.

    Implements the :class:`HierarchyNormalizer` protocol. Applies a
    pipeline of transformations to clean up raw hierarchy dumps.

    Usage::

        normalizer = AndroidHierarchyNormalizer()
        clean = normalizer.normalize(raw_hierarchy)
    """

    def normalize(self, hierarchy: ViewHierarchy) -> ViewHierarchy:
        """Run the full normalization pipeline.

        Steps:
        1. Remove system UI nodes (status bar, nav bar).
        2. Flatten DecorView wrapper layers.
        3. Merge compound views with single semantic children.
        4. Compute actual visibility.
        5. Assign navigation order.

        Parameters:
            hierarchy: Raw :class:`ViewHierarchy`.

        Returns:
            Normalized :class:`ViewHierarchy`.
        """
        # Work on mutable copies
        nodes = dict(hierarchy.nodes)
        root_id = hierarchy.root_id

        # Step 1: Remove system UI
        root_id, nodes = self._remove_system_ui(root_id, nodes)

        # Step 2: Flatten DecorView
        root_id, nodes = self._flatten_decorview(root_id, nodes)

        # Step 3: Merge compound views
        root_id, nodes = self._merge_compound_views(root_id, nodes)

        # Step 4: Compute visibility
        screen_bounds = BoundsInfo(
            screen_left=0, screen_top=0,
            screen_right=hierarchy.screen_width,
            screen_bottom=hierarchy.screen_height,
            parent_left=0, parent_top=0,
            parent_right=hierarchy.screen_width,
            parent_bottom=hierarchy.screen_height,
        )
        nodes = self._compute_visibility(nodes, screen_bounds)

        # Step 5: Assign navigation order
        nav_order = self._assign_navigation_order(root_id, nodes)

        return ViewHierarchy(
            root_id=root_id,
            nodes=nodes,
            package_name=hierarchy.package_name,
            window_title=hierarchy.window_title,
            screen_width=hierarchy.screen_width,
            screen_height=hierarchy.screen_height,
            api_level=hierarchy.api_level,
        )

    def map_class_to_role(self, class_name: str) -> str:
        """Map an Android widget class name to a WAI-ARIA role.

        Parameters:
            class_name: Fully-qualified Android class name.

        Returns:
            ARIA role name string.
        """
        if class_name in _CLASS_TO_ROLE:
            return _CLASS_TO_ROLE[class_name]

        # Try matching by simple class name
        simple = class_name.rsplit(".", 1)[-1] if "." in class_name else class_name
        for full_name, role in _CLASS_TO_ROLE.items():
            if full_name.endswith("." + simple):
                return role

        # Default heuristic based on name patterns
        lower = class_name.lower()
        if "button" in lower:
            return "button"
        if "text" in lower and "edit" in lower:
            return "textbox"
        if "image" in lower:
            return "img"
        if "checkbox" in lower or "check" in lower:
            return "checkbox"
        if "radio" in lower:
            return "radio"
        if "switch" in lower or "toggle" in lower:
            return "switch"
        if "seekbar" in lower or "slider" in lower:
            return "slider"
        if "spinner" in lower:
            return "combobox"
        if "recycler" in lower or "listview" in lower:
            return "list"
        if "scroll" in lower:
            return "region"
        if "tab" in lower:
            return "tablist"
        if "web" in lower:
            return "document"

        return "generic"

    def filter_important_nodes(
        self, hierarchy: ViewHierarchy,
    ) -> Sequence[AndroidNode]:
        """Return only nodes important for accessibility.

        Filters out invisible nodes, non-important nodes, and
        purely decorative containers.

        Parameters:
            hierarchy: Full :class:`ViewHierarchy`.

        Returns:
            Filtered sequence of :class:`AndroidNode`.
        """
        return [
            node for node in hierarchy.nodes.values()
            if node.is_important_for_accessibility
            and node.is_visible_to_user
            and self._is_semantically_meaningful(node)
        ]

    # ── Pipeline steps ────────────────────────────────────────────────────

    def flatten_decorview(self, tree: ViewHierarchy) -> ViewHierarchy:
        """Remove DecorView wrapper layers from the hierarchy.

        DecorView is the Android internal window decoration container.
        It adds no semantic value and should be stripped.

        Parameters:
            tree: Raw :class:`ViewHierarchy`.

        Returns:
            Hierarchy with DecorView layers removed.
        """
        root_id, nodes = self._flatten_decorview(tree.root_id, dict(tree.nodes))
        return ViewHierarchy(
            root_id=root_id, nodes=nodes,
            package_name=tree.package_name,
            window_title=tree.window_title,
            screen_width=tree.screen_width,
            screen_height=tree.screen_height,
            api_level=tree.api_level,
        )

    def merge_compound_views(self, tree: ViewHierarchy) -> ViewHierarchy:
        """Merge compound views with a single child.

        Layout containers (e.g. CardView, FrameLayout) that wrap a
        single semantic child are replaced by the child, inheriting
        the parent's bounds if larger.

        Parameters:
            tree: Input :class:`ViewHierarchy`.

        Returns:
            Hierarchy with compound views merged.
        """
        root_id, nodes = self._merge_compound_views(tree.root_id, dict(tree.nodes))
        return ViewHierarchy(
            root_id=root_id, nodes=nodes,
            package_name=tree.package_name,
            window_title=tree.window_title,
            screen_width=tree.screen_width,
            screen_height=tree.screen_height,
            api_level=tree.api_level,
        )

    def extract_scrollable_regions(
        self, tree: ViewHierarchy,
    ) -> list[AndroidNode]:
        """Identify all scrollable containers in the hierarchy.

        Parameters:
            tree: Input :class:`ViewHierarchy`.

        Returns:
            List of :class:`AndroidNode` that are scrollable.
        """
        return [n for n in tree.nodes.values() if n.is_scrollable]

    def compute_visibility(
        self, tree: ViewHierarchy,
        screen_bounds: Optional[BoundsInfo] = None,
    ) -> ViewHierarchy:
        """Determine actually visible nodes based on screen bounds.

        Parameters:
            tree: Input :class:`ViewHierarchy`.
            screen_bounds: Screen boundary; defaults to full screen.

        Returns:
            Hierarchy with visibility flags updated.
        """
        if screen_bounds is None:
            screen_bounds = BoundsInfo(
                screen_left=0, screen_top=0,
                screen_right=tree.screen_width,
                screen_bottom=tree.screen_height,
                parent_left=0, parent_top=0,
                parent_right=tree.screen_width,
                parent_bottom=tree.screen_height,
            )
        nodes = self._compute_visibility(dict(tree.nodes), screen_bounds)
        return ViewHierarchy(
            root_id=tree.root_id, nodes=nodes,
            package_name=tree.package_name,
            window_title=tree.window_title,
            screen_width=tree.screen_width,
            screen_height=tree.screen_height,
            api_level=tree.api_level,
        )

    def assign_navigation_order(
        self, tree: ViewHierarchy,
    ) -> list[str]:
        """Compute accessibility focus traversal order.

        Returns nodes in the order they would be traversed by
        TalkBack or Switch Access (top-to-bottom, left-to-right).

        Parameters:
            tree: Input :class:`ViewHierarchy`.

        Returns:
            Ordered list of node ids in focus traversal order.
        """
        return self._assign_navigation_order(tree.root_id, tree.nodes)

    def remove_system_ui(self, tree: ViewHierarchy) -> ViewHierarchy:
        """Strip status bar and navigation bar nodes.

        Parameters:
            tree: Input :class:`ViewHierarchy`.

        Returns:
            Hierarchy with system UI nodes removed.
        """
        root_id, nodes = self._remove_system_ui(tree.root_id, dict(tree.nodes))
        return ViewHierarchy(
            root_id=root_id, nodes=nodes,
            package_name=tree.package_name,
            window_title=tree.window_title,
            screen_width=tree.screen_width,
            screen_height=tree.screen_height,
            api_level=tree.api_level,
        )

    # ── Internal implementation ───────────────────────────────────────────

    def _remove_system_ui(
        self, root_id: str, nodes: Dict[str, AndroidNode],
    ) -> tuple[str, Dict[str, AndroidNode]]:
        """Remove system UI nodes and their subtrees."""
        to_remove: set[str] = set()

        for nid, node in nodes.items():
            if (node.class_name in _SYSTEM_UI_CLASSES
                    or node.package_name in _SYSTEM_UI_PACKAGES):
                to_remove.update(self._subtree_ids(nid, nodes))

        if root_id in to_remove:
            return root_id, nodes  # Can't remove root

        # Remove nodes and update child_ids
        new_nodes: Dict[str, AndroidNode] = {}
        for nid, node in nodes.items():
            if nid in to_remove:
                continue
            new_children = tuple(
                cid for cid in node.child_ids if cid not in to_remove
            )
            if new_children != node.child_ids:
                node = AndroidNode(
                    node_id=node.node_id,
                    class_name=node.class_name,
                    package_name=node.package_name,
                    bounds=node.bounds,
                    description=node.description,
                    actions=node.actions,
                    is_clickable=node.is_clickable,
                    is_focusable=node.is_focusable,
                    is_focused=node.is_focused,
                    is_scrollable=node.is_scrollable,
                    is_checkable=node.is_checkable,
                    is_checked=node.is_checked,
                    is_enabled=node.is_enabled,
                    is_visible_to_user=node.is_visible_to_user,
                    is_important_for_accessibility=node.is_important_for_accessibility,
                    child_ids=new_children,
                    depth=node.depth,
                )
            new_nodes[nid] = node

        return root_id, new_nodes

    def _flatten_decorview(
        self, root_id: str, nodes: Dict[str, AndroidNode],
    ) -> tuple[str, Dict[str, AndroidNode]]:
        """Skip DecorView wrapper layers at the root."""
        current_id = root_id
        visited: set[str] = set()

        while current_id in nodes and current_id not in visited:
            visited.add(current_id)
            node = nodes[current_id]
            if node.class_name in _DECOR_VIEW_CLASSES and len(node.child_ids) == 1:
                # Skip this layer
                child_id = node.child_ids[0]
                del nodes[current_id]
                current_id = child_id
            else:
                break

        return current_id, nodes

    def _merge_compound_views(
        self, root_id: str, nodes: Dict[str, AndroidNode],
    ) -> tuple[str, Dict[str, AndroidNode]]:
        """Merge layout containers that have a single semantic child."""
        changed = True
        while changed:
            changed = False
            for nid in list(nodes.keys()):
                if nid not in nodes:
                    continue
                node = nodes[nid]
                if (node.class_name in _LAYOUT_CLASSES
                        and len(node.child_ids) == 1
                        and not self._is_semantically_meaningful(node)):
                    child_id = node.child_ids[0]
                    if child_id not in nodes:
                        continue

                    # Reparent: update all parents that point to this node
                    for other_id, other in nodes.items():
                        if nid in other.child_ids:
                            new_children = tuple(
                                child_id if cid == nid else cid
                                for cid in other.child_ids
                            )
                            nodes[other_id] = AndroidNode(
                                node_id=other.node_id,
                                class_name=other.class_name,
                                package_name=other.package_name,
                                bounds=other.bounds,
                                description=other.description,
                                actions=other.actions,
                                is_clickable=other.is_clickable,
                                is_focusable=other.is_focusable,
                                is_focused=other.is_focused,
                                is_scrollable=other.is_scrollable,
                                is_checkable=other.is_checkable,
                                is_checked=other.is_checked,
                                is_enabled=other.is_enabled,
                                is_visible_to_user=other.is_visible_to_user,
                                is_important_for_accessibility=other.is_important_for_accessibility,
                                child_ids=new_children,
                                depth=other.depth,
                            )

                    if nid == root_id:
                        root_id = child_id

                    del nodes[nid]
                    changed = True

        return root_id, nodes

    def _compute_visibility(
        self, nodes: Dict[str, AndroidNode], screen_bounds: BoundsInfo,
    ) -> Dict[str, AndroidNode]:
        """Update visibility flags based on screen bounds overlap."""
        new_nodes: Dict[str, AndroidNode] = {}

        for nid, node in nodes.items():
            b = node.bounds
            on_screen = (
                b.screen_right > screen_bounds.screen_left
                and b.screen_left < screen_bounds.screen_right
                and b.screen_bottom > screen_bounds.screen_top
                and b.screen_top < screen_bounds.screen_bottom
                and b.screen_width > 0
                and b.screen_height > 0
            )
            visible = node.is_visible_to_user and on_screen

            if visible != node.is_visible_to_user:
                node = AndroidNode(
                    node_id=node.node_id,
                    class_name=node.class_name,
                    package_name=node.package_name,
                    bounds=node.bounds,
                    description=node.description,
                    actions=node.actions,
                    is_clickable=node.is_clickable,
                    is_focusable=node.is_focusable,
                    is_focused=node.is_focused,
                    is_scrollable=node.is_scrollable,
                    is_checkable=node.is_checkable,
                    is_checked=node.is_checked,
                    is_enabled=node.is_enabled,
                    is_visible_to_user=visible,
                    is_important_for_accessibility=node.is_important_for_accessibility,
                    child_ids=node.child_ids,
                    depth=node.depth,
                )
            new_nodes[nid] = node

        return new_nodes

    def _assign_navigation_order(
        self, root_id: str, nodes: Dict[str, AndroidNode],
    ) -> list[str]:
        """Compute focus traversal order (top-to-bottom, left-to-right).

        Android accessibility focus follows a depth-first traversal
        filtered to focusable/important nodes, ordered by screen position.
        """
        order: list[tuple[int, int, str]] = []

        for nid, node in nodes.items():
            if (node.is_visible_to_user
                    and node.is_important_for_accessibility
                    and (node.is_focusable or node.is_clickable)):
                order.append((
                    node.bounds.screen_top,
                    node.bounds.screen_left,
                    node.node_id,
                ))

        order.sort()
        return [nid for _, _, nid in order]

    def _subtree_ids(
        self, root_id: str, nodes: Dict[str, AndroidNode],
    ) -> set[str]:
        """Collect all node ids in the subtree rooted at root_id."""
        ids: set[str] = set()
        stack = [root_id]
        while stack:
            nid = stack.pop()
            if nid in ids or nid not in nodes:
                continue
            ids.add(nid)
            stack.extend(nodes[nid].child_ids)
        return ids

    def _is_semantically_meaningful(self, node: AndroidNode) -> bool:
        """Check if a node carries semantic information beyond being a container."""
        if node.is_clickable or node.is_checkable or node.is_scrollable:
            return True
        if node.description.accessible_name:
            return True
        if node.class_name not in _LAYOUT_CLASSES:
            return True
        return False
