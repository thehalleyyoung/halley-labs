"""
usability_oracle.android_a11y.converter — Convert Android hierarchy to internal model.

Maps Android view hierarchy nodes to the platform-agnostic
:class:`AccessibilityTree` model used by the usability oracle pipeline.

Reference: Android AccessibilityNodeInfo API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.android_a11y.types import (
    AccessibilityAction,
    AccessibilityActionId,
    AndroidClassName,
    AndroidNode,
    ViewHierarchy,
)
from usability_oracle.core.enums import AccessibilityRole
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


# ═══════════════════════════════════════════════════════════════════════════
# Android class → internal role mapping
# ═══════════════════════════════════════════════════════════════════════════

_CLASS_ROLE_MAP: Dict[str, str] = {
    AndroidClassName.BUTTON.value: AccessibilityRole.BUTTON.value,
    AndroidClassName.IMAGE_BUTTON.value: AccessibilityRole.BUTTON.value,
    AndroidClassName.TEXT_VIEW.value: AccessibilityRole.GENERIC.value,
    AndroidClassName.IMAGE_VIEW.value: AccessibilityRole.IMAGE.value,
    AndroidClassName.EDIT_TEXT.value: AccessibilityRole.TEXTFIELD.value,
    AndroidClassName.CHECK_BOX.value: AccessibilityRole.CHECKBOX.value,
    AndroidClassName.RADIO_BUTTON.value: AccessibilityRole.RADIO.value,
    AndroidClassName.TOGGLE_BUTTON.value: AccessibilityRole.CHECKBOX.value,
    AndroidClassName.SWITCH.value: AccessibilityRole.CHECKBOX.value,
    AndroidClassName.SEEK_BAR.value: AccessibilityRole.SLIDER.value,
    AndroidClassName.SPINNER.value: AccessibilityRole.COMBOBOX.value,
    AndroidClassName.PROGRESS_BAR.value: AccessibilityRole.GENERIC.value,
    AndroidClassName.SCROLL_VIEW.value: AccessibilityRole.REGION.value,
    AndroidClassName.RECYCLER_VIEW.value: AccessibilityRole.LIST.value,
    AndroidClassName.WEB_VIEW.value: AccessibilityRole.DOCUMENT.value,
    AndroidClassName.VIEW_GROUP.value: AccessibilityRole.GROUP.value,
    AndroidClassName.LINEAR_LAYOUT.value: AccessibilityRole.GROUP.value,
    AndroidClassName.RELATIVE_LAYOUT.value: AccessibilityRole.GROUP.value,
    AndroidClassName.FRAME_LAYOUT.value: AccessibilityRole.GROUP.value,
    AndroidClassName.TAB_WIDGET.value: AccessibilityRole.TAB.value,
    AndroidClassName.VIEW.value: AccessibilityRole.GENERIC.value,
}


# ═══════════════════════════════════════════════════════════════════════════
# AndroidToAccessibilityConverter
# ═══════════════════════════════════════════════════════════════════════════

class AndroidToAccessibilityConverter:
    """Convert an Android :class:`ViewHierarchy` to :class:`AccessibilityTree`.

    Maps Android widget classes to accessibility roles, translates
    Android node states, and builds the platform-agnostic tree.

    Usage::

        converter = AndroidToAccessibilityConverter()
        a11y_tree = converter.convert(view_hierarchy)
    """

    def convert(self, hierarchy: ViewHierarchy) -> AccessibilityTree:
        """Convert a full :class:`ViewHierarchy` to :class:`AccessibilityTree`.

        Parameters:
            hierarchy: Parsed Android view hierarchy.

        Returns:
            Platform-agnostic :class:`AccessibilityTree`.
        """
        root_node = self._convert_node(hierarchy.root, hierarchy)

        tree = AccessibilityTree(
            root=root_node,
            metadata={
                "source_format": "android",
                "package_name": hierarchy.package_name,
                "window_title": hierarchy.window_title,
                "screen_width": hierarchy.screen_width,
                "screen_height": hierarchy.screen_height,
                "api_level": hierarchy.api_level,
                "node_count": hierarchy.node_count,
            },
        )
        return tree

    def map_class_to_role(self, class_name: str) -> str:
        """Map an Android widget class name to an internal AccessibilityRole.

        Parameters:
            class_name: Fully-qualified Android class name
                (e.g. ``"android.widget.Button"``).

        Returns:
            Internal role string (e.g. ``"button"``).
        """
        if class_name in _CLASS_ROLE_MAP:
            return _CLASS_ROLE_MAP[class_name]

        # Heuristic matching by class name suffix
        lower = class_name.lower()
        if "button" in lower:
            return AccessibilityRole.BUTTON.value
        if "edittext" in lower or ("text" in lower and "edit" in lower):
            return AccessibilityRole.TEXTFIELD.value
        if "imageview" in lower or "image" in lower:
            return AccessibilityRole.IMAGE.value
        if "checkbox" in lower:
            return AccessibilityRole.CHECKBOX.value
        if "radiobutton" in lower or "radio" in lower:
            return AccessibilityRole.RADIO.value
        if "switch" in lower or "toggle" in lower:
            return AccessibilityRole.CHECKBOX.value
        if "seekbar" in lower or "slider" in lower:
            return AccessibilityRole.SLIDER.value
        if "spinner" in lower:
            return AccessibilityRole.COMBOBOX.value
        if "recyclerview" in lower or "listview" in lower:
            return AccessibilityRole.LIST.value
        if "scrollview" in lower:
            return AccessibilityRole.REGION.value
        if "tabwidget" in lower or "tablayout" in lower:
            return AccessibilityRole.TAB.value
        if "webview" in lower:
            return AccessibilityRole.DOCUMENT.value
        if "textview" in lower:
            return AccessibilityRole.GENERIC.value

        return AccessibilityRole.GENERIC.value

    def extract_interaction_model(self, node: AndroidNode) -> list[str]:
        """Determine available interactions for an Android node.

        Based on the node's boolean properties (clickable, long-clickable,
        scrollable, editable) and its action list.

        Parameters:
            node: An :class:`AndroidNode`.

        Returns:
            List of interaction type strings.
        """
        interactions: list[str] = []

        if node.is_clickable:
            interactions.append("click")

        # Long-click — check actions list
        if any(a.action_id == AccessibilityActionId.LONG_CLICK.value
               for a in node.actions):
            interactions.append("long_click")

        if node.is_scrollable:
            interactions.append("scroll")

        if node.is_checkable:
            interactions.append("toggle")

        # Editable — EditText class or has SET_TEXT action
        if (node.class_name == AndroidClassName.EDIT_TEXT.value
                or any(a.action_id == AccessibilityActionId.SET_TEXT.value
                       for a in node.actions)):
            interactions.append("type")

        if node.is_focusable:
            interactions.append("focus")

        return interactions

    def compute_content_label(self, node: AndroidNode) -> str:
        """Resolve the accessible name for an Android node.

        Follows Android precedence: contentDescription → text → hintText.

        Parameters:
            node: An :class:`AndroidNode`.

        Returns:
            Resolved accessible name string, may be empty.
        """
        return node.description.accessible_name or ""

    # ── Internal helpers ──────────────────────────────────────────────────

    def _convert_node(
        self, node: AndroidNode, hierarchy: ViewHierarchy, depth: int = 0,
    ) -> AccessibilityNode:
        """Recursively convert an AndroidNode to an AccessibilityNode."""
        role = self.map_class_to_role(node.class_name)
        name = self.compute_content_label(node)
        state = self._convert_state(node)
        bbox = self._convert_bounds(node)
        interactions = self.extract_interaction_model(node)

        props: dict[str, Any] = {
            "android_class": node.class_name,
            "package": node.package_name,
        }
        if interactions:
            props["interactions"] = interactions
        if node.is_scrollable:
            props["scrollable"] = True

        # Convert children recursively
        children: list[AccessibilityNode] = []
        for child_id in node.child_ids:
            child_node = hierarchy.get_node(child_id)
            if child_node is not None:
                children.append(
                    self._convert_node(child_node, hierarchy, depth=depth + 1)
                )

        return AccessibilityNode(
            id=node.node_id,
            role=role,
            name=name,
            description=node.description.tooltip_text or "",
            bounding_box=bbox,
            properties=props,
            state=state,
            children=children,
            depth=depth,
        )

    def _convert_state(self, node: AndroidNode) -> AccessibilityState:
        """Convert Android node state to AccessibilityState."""
        return AccessibilityState(
            focused=node.is_focused,
            selected=False,
            expanded=False,
            checked=node.is_checked if node.is_checkable else None,
            disabled=not node.is_enabled,
            hidden=not node.is_visible_to_user,
            required=False,
            readonly=not any(
                a.action_id == AccessibilityActionId.SET_TEXT.value
                for a in node.actions
            ) if node.class_name == AndroidClassName.EDIT_TEXT.value else False,
        )

    def _convert_bounds(self, node: AndroidNode) -> BoundingBox:
        """Convert Android BoundsInfo to BoundingBox."""
        b = node.bounds
        return BoundingBox(
            x=float(b.screen_left),
            y=float(b.screen_top),
            width=float(max(0, b.screen_right - b.screen_left)),
            height=float(max(0, b.screen_bottom - b.screen_top)),
        )
