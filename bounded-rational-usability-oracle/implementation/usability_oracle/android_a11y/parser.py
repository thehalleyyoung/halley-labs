"""
usability_oracle.android_a11y.parser — Android accessibility XML parser.

Parses ``uiautomator dump`` XML and AccessibilityNodeInfo JSON formats
into :class:`ViewHierarchy` structures.

Reference: Android AccessibilityNodeInfo API,
           uiautomator dump XML schema.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lxml import etree

from usability_oracle.android_a11y.types import (
    AccessibilityAction,
    AccessibilityActionId,
    AndroidClassName,
    AndroidNode,
    BoundsInfo,
    ContentDescription,
    ViewHierarchy,
)
from usability_oracle.android_a11y.protocols import AndroidParser as AndroidParserProtocol
from usability_oracle.core.errors import ParseError


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

# Standard action names to AccessibilityActionId mapping
_ACTION_MAP: Dict[str, str] = {
    "click": AccessibilityActionId.CLICK.value,
    "long_click": AccessibilityActionId.LONG_CLICK.value,
    "scroll_forward": AccessibilityActionId.SCROLL_FORWARD.value,
    "scroll_backward": AccessibilityActionId.SCROLL_BACKWARD.value,
    "focus": AccessibilityActionId.FOCUS.value,
    "clear_focus": AccessibilityActionId.CLEAR_FOCUS.value,
    "select": AccessibilityActionId.SELECT.value,
    "set_text": AccessibilityActionId.SET_TEXT.value,
}


# ═══════════════════════════════════════════════════════════════════════════
# AndroidAccessibilityParser
# ═══════════════════════════════════════════════════════════════════════════

class AndroidAccessibilityParser:
    """Parse Android accessibility dumps into :class:`ViewHierarchy`.

    Supports both the XML format from ``uiautomator dump`` and JSON
    formats used by accessibility services and the Rico dataset.

    Implements the :class:`AndroidParser` protocol.

    Usage::

        parser = AndroidAccessibilityParser()
        hierarchy = parser.parse_xml(xml_string)
    """

    def __init__(
        self,
        *,
        default_package: str = "unknown",
        default_screen_width: int = 1080,
        default_screen_height: int = 1920,
    ) -> None:
        self._default_package = default_package
        self._default_screen_width = default_screen_width
        self._default_screen_height = default_screen_height
        self._counter = 0
        # Temporary storage for children during parsing (frozen dataclass workaround)
        self._children_map: Dict[str, list[AndroidNode]] = {}

    # ── Public API — XML parsing ──────────────────────────────────────────

    def parse_xml(self, xml_content: str) -> ViewHierarchy:
        """Parse a ``uiautomator dump`` XML string.

        The standard uiautomator XML format has a ``<hierarchy>`` root
        element containing ``<node>`` elements with attributes like
        ``class``, ``text``, ``content-desc``, ``bounds``, etc.

        Parameters:
            xml_content: Raw XML from ``uiautomator dump``.

        Returns:
            Parsed :class:`ViewHierarchy`.

        Raises:
            ParseError: On malformed or empty XML.
        """
        if not xml_content or not xml_content.strip():
            raise ParseError("Empty XML content")

        self._counter = 0
        self._children_map = {}

        try:
            root = etree.fromstring(xml_content.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"Malformed XML: {exc}") from exc

        # uiautomator format: root is <hierarchy>
        if root.tag == "hierarchy":
            return self._parse_uiautomator_hierarchy(root)

        # Alternative: root is directly a <node>
        if root.tag == "node":
            nodes: Dict[str, AndroidNode] = {}
            root_node = self._extract_node_xml(root, depth=0)
            self._collect_nodes(root_node, nodes)
            return ViewHierarchy(
                root_id=root_node.node_id,
                nodes=nodes,
                package_name=root_node.package_name,
                screen_width=self._default_screen_width,
                screen_height=self._default_screen_height,
            )

        raise ParseError(f"Unexpected root element: <{root.tag}>")

    def parse_view_hierarchy(self, xml_content: str) -> ViewHierarchy:
        """Parse Android View Hierarchy format.

        This is an alias for :meth:`parse_xml` that also supports the
        extended View Hierarchy dump format with additional metadata.

        Parameters:
            xml_content: Raw XML string.

        Returns:
            Parsed :class:`ViewHierarchy`.
        """
        return self.parse_xml(xml_content)

    # ── Public API — JSON parsing ─────────────────────────────────────────

    def parse_json(self, json_content: str) -> ViewHierarchy:
        """Parse a JSON-format view hierarchy (e.g. Rico dataset).

        Parameters:
            json_content: Raw JSON string.

        Returns:
            Parsed :class:`ViewHierarchy`.

        Raises:
            ParseError: On malformed or empty JSON.
        """
        if not json_content or not json_content.strip():
            raise ParseError("Empty JSON content")

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Malformed JSON: {exc}") from exc

        return self.parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> ViewHierarchy:
        """Parse an already-deserialised dictionary.

        Supports Rico/JSON format with ``activity``, ``activity_name``,
        and a nested ``root`` or ``children`` structure.

        Parameters:
            data: Dictionary representation of the hierarchy.

        Returns:
            Parsed :class:`ViewHierarchy`.
        """
        self._counter = 0
        self._children_map = {}

        # Rico format: top-level has "activity_name" and "root_element"
        if "activity_name" in data or "activity" in data:
            return self._parse_rico_format(data)

        # ViewHierarchy.from_dict compatible format
        if "root_id" in data and "nodes" in data:
            return ViewHierarchy.from_dict(data)

        # Simple nested dict: treat as a single node tree
        root_node = self._extract_node_dict(data, depth=0)
        nodes: Dict[str, AndroidNode] = {}
        self._collect_nodes(root_node, nodes)
        return ViewHierarchy(
            root_id=root_node.node_id,
            nodes=nodes,
            package_name=root_node.package_name,
            screen_width=self._default_screen_width,
            screen_height=self._default_screen_height,
        )

    # ── Node extraction ───────────────────────────────────────────────────

    def extract_node(self, xml_element: Any) -> AndroidNode:
        """Extract a single node from a uiautomator XML ``<node>`` element.

        Parameters:
            xml_element: An lxml element with uiautomator node attributes.

        Returns:
            Parsed :class:`AndroidNode`.
        """
        return self._extract_node_xml(xml_element, depth=0)

    def extract_bounds(self, bounds_str: str) -> BoundsInfo:
        """Parse Android bounds string in ``[x1,y1][x2,y2]`` format.

        Parameters:
            bounds_str: Bounds string, e.g. ``"[0,96][1080,192]"``.

        Returns:
            :class:`BoundsInfo` with screen and parent bounds.

        Raises:
            ParseError: If the bounds string is malformed.
        """
        match = _BOUNDS_RE.match(bounds_str.strip())
        if not match:
            raise ParseError(f"Malformed bounds string: {bounds_str!r}")

        x1, y1, x2, y2 = (int(g) for g in match.groups())
        return BoundsInfo(
            screen_left=x1, screen_top=y1,
            screen_right=x2, screen_bottom=y2,
            parent_left=x1, parent_top=y1,
            parent_right=x2, parent_bottom=y2,
        )

    def extract_accessibility_actions(
        self, element: Any,
    ) -> list[AccessibilityAction]:
        """Extract available accessibility actions from an element.

        Infers actions from boolean attributes (clickable, scrollable, etc.)
        in the uiautomator XML format.

        Parameters:
            element: An lxml element or a dict with node attributes.

        Returns:
            List of :class:`AccessibilityAction`.
        """
        actions: list[AccessibilityAction] = []

        if isinstance(element, dict):
            get = element.get
        else:
            get = element.get

        if _to_bool(get("clickable", "false")):
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.CLICK.value,
            ))

        if _to_bool(get("long-clickable", "false")):
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.LONG_CLICK.value,
            ))

        if _to_bool(get("scrollable", "false")):
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.SCROLL_FORWARD.value,
            ))
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.SCROLL_BACKWARD.value,
            ))

        if _to_bool(get("focusable", "false")):
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.FOCUS.value,
            ))

        if _to_bool(get("checkable", "false")):
            actions.append(AccessibilityAction(
                action_id=AccessibilityActionId.SELECT.value,
            ))

        return actions

    def extract_content_description(self, element: Any) -> ContentDescription:
        """Extract content description, text, and hint from an element.

        Parameters:
            element: An lxml element or dict with node attributes.

        Returns:
            :class:`ContentDescription` with all available text fields.
        """
        if isinstance(element, dict):
            get = element.get
        else:
            get = element.get

        text = get("text", "") or None
        content_desc = get("content-desc", "") or None
        hint = get("hint", "") or None
        tooltip = get("tooltip", "") or None
        resource_id = get("resource-id", "") or None

        if text == "":
            text = None
        if content_desc == "":
            content_desc = None
        if hint == "":
            hint = None

        return ContentDescription(
            text=text,
            content_description=content_desc,
            hint_text=hint,
            tooltip_text=tooltip,
            labeled_by=resource_id if resource_id and "label" in (resource_id or "") else None,
        )

    # ── Internal XML parsing ──────────────────────────────────────────────

    def _parse_uiautomator_hierarchy(self, hierarchy_el: Any) -> ViewHierarchy:
        """Parse a uiautomator <hierarchy> root element."""
        rotation = hierarchy_el.get("rotation", "0")
        node_elements = list(hierarchy_el)

        if not node_elements:
            raise ParseError("Empty hierarchy: no child nodes")

        # Parse all top-level nodes (usually one DecorView)
        all_nodes: Dict[str, AndroidNode] = {}
        root_ids: list[str] = []

        for child_el in node_elements:
            if child_el.tag == "node":
                node = self._extract_node_xml(child_el, depth=0)
                self._collect_nodes(node, all_nodes)
                root_ids.append(node.node_id)

        if not root_ids:
            raise ParseError("No valid nodes found in hierarchy")

        root_id = root_ids[0]
        package = all_nodes[root_id].package_name

        return ViewHierarchy(
            root_id=root_id,
            nodes=all_nodes,
            package_name=package,
            window_title="",
            screen_width=self._default_screen_width,
            screen_height=self._default_screen_height,
        )

    def _extract_node_xml(self, element: Any, depth: int) -> AndroidNode:
        """Recursively extract an AndroidNode from a <node> element."""
        self._counter += 1
        node_id = element.get("resource-id", "") or f"node-{self._counter}"
        # Ensure uniqueness
        if not element.get("resource-id"):
            node_id = f"node-{self._counter}"

        class_name = element.get("class", AndroidClassName.VIEW.value)
        package = element.get("package", self._default_package)

        # Bounds
        bounds_str = element.get("bounds", "[0,0][0,0]")
        try:
            bounds = self.extract_bounds(bounds_str)
        except ParseError:
            bounds = BoundsInfo(0, 0, 0, 0, 0, 0, 0, 0)

        # Content description
        description = self.extract_content_description(element)

        # Actions
        actions = tuple(self.extract_accessibility_actions(element))

        # Boolean properties
        is_clickable = _to_bool(element.get("clickable", "false"))
        is_focusable = _to_bool(element.get("focusable", "false"))
        is_focused = _to_bool(element.get("focused", "false"))
        is_scrollable = _to_bool(element.get("scrollable", "false"))
        is_checkable = _to_bool(element.get("checkable", "false"))
        is_checked = _to_bool(element.get("checked", "false"))
        is_enabled = _to_bool(element.get("enabled", "true"))
        is_visible = _to_bool(element.get("displayed", "true"))
        is_important = _to_bool(element.get(
            "important-for-accessibility", "true",
        ))

        # Recurse into children
        child_ids: list[str] = []
        child_nodes: list[AndroidNode] = []
        for child_el in element:
            if child_el.tag == "node":
                child = self._extract_node_xml(child_el, depth=depth + 1)
                child_nodes.append(child)
                child_ids.append(child.node_id)

        node = AndroidNode(
            node_id=node_id,
            class_name=class_name,
            package_name=package,
            bounds=bounds,
            description=description,
            actions=actions,
            is_clickable=is_clickable,
            is_focusable=is_focusable,
            is_focused=is_focused,
            is_scrollable=is_scrollable,
            is_checkable=is_checkable,
            is_checked=is_checked,
            is_enabled=is_enabled,
            is_visible_to_user=is_visible,
            is_important_for_accessibility=is_important,
            child_ids=tuple(child_ids),
            depth=depth,
        )
        self._children_map[node_id] = child_nodes
        return node

    def _collect_nodes(
        self, node: AndroidNode, nodes: Dict[str, AndroidNode],
    ) -> None:
        """Recursively collect all nodes into a flat dict."""
        children = self._children_map.get(node.node_id, [])
        nodes[node.node_id] = node
        for child in children:
            self._collect_nodes(child, nodes)

    # ── Internal JSON/dict parsing ────────────────────────────────────────

    def _parse_rico_format(self, data: Dict[str, Any]) -> ViewHierarchy:
        """Parse Rico dataset JSON format."""
        activity = data.get("activity_name", data.get("activity", ""))
        root_data = data.get("root_element", data.get("activity", data))

        if "children" in root_data or "class" in root_data:
            root_node = self._extract_node_dict(root_data, depth=0)
        else:
            raise ParseError("Cannot find root element in Rico data")

        nodes: Dict[str, AndroidNode] = {}
        self._collect_nodes(root_node, nodes)

        return ViewHierarchy(
            root_id=root_node.node_id,
            nodes=nodes,
            package_name=str(activity),
            window_title=str(activity),
            screen_width=self._default_screen_width,
            screen_height=self._default_screen_height,
        )

    def _extract_node_dict(self, data: Dict[str, Any], depth: int) -> AndroidNode:
        """Extract an AndroidNode from a dictionary."""
        self._counter += 1
        node_id = str(data.get("resource-id", data.get("id", f"node-{self._counter}")))
        if not node_id or node_id == "":
            node_id = f"node-{self._counter}"

        class_name = str(data.get("class", data.get("className", AndroidClassName.VIEW.value)))
        package = str(data.get("package", data.get("packageName", self._default_package)))

        # Bounds — support multiple formats
        bounds = self._extract_bounds_from_dict(data)

        # Content
        description = ContentDescription(
            text=data.get("text") or None,
            content_description=data.get("content-desc", data.get("contentDescription")) or None,
            hint_text=data.get("hint", data.get("hintText")) or None,
        )

        # Actions
        actions_data = data.get("actions", [])
        actions: list[AccessibilityAction] = []
        for a in actions_data:
            if isinstance(a, str):
                actions.append(AccessibilityAction(action_id=a))
            elif isinstance(a, dict):
                actions.append(AccessibilityAction.from_dict(a))

        is_clickable = bool(data.get("clickable", False))
        is_focusable = bool(data.get("focusable", False))
        is_scrollable = bool(data.get("scrollable", False))
        is_checkable = bool(data.get("checkable", False))
        is_checked = bool(data.get("checked", False))
        is_enabled = bool(data.get("enabled", True))
        is_visible = bool(data.get("visible-to-user", data.get("visibleToUser", True)))

        # Children
        child_ids: list[str] = []
        child_nodes: list[AndroidNode] = []
        for child_data in data.get("children", []):
            if isinstance(child_data, dict):
                child = self._extract_node_dict(child_data, depth=depth + 1)
                child_nodes.append(child)
                child_ids.append(child.node_id)

        node = AndroidNode(
            node_id=node_id,
            class_name=class_name,
            package_name=package,
            bounds=bounds,
            description=description,
            actions=tuple(actions),
            is_clickable=is_clickable,
            is_focusable=is_focusable,
            is_scrollable=is_scrollable,
            is_checkable=is_checkable,
            is_checked=is_checked,
            is_enabled=is_enabled,
            is_visible_to_user=is_visible,
            child_ids=tuple(child_ids),
            depth=depth,
        )
        self._children_map[node_id] = child_nodes
        return node

    def _extract_bounds_from_dict(self, data: Dict[str, Any]) -> BoundsInfo:
        """Extract bounds from various dict formats."""
        # Format 1: "bounds" string
        bounds_str = data.get("bounds", "")
        if bounds_str and _BOUNDS_RE.match(bounds_str):
            return self.extract_bounds(bounds_str)

        # Format 2: "boundsInScreen" dict
        screen = data.get("boundsInScreen", data.get("bounds_in_screen", {}))
        if isinstance(screen, dict) and "left" in screen:
            parent = data.get("boundsInParent", data.get("bounds_in_parent", screen))
            return BoundsInfo(
                screen_left=int(screen.get("left", 0)),
                screen_top=int(screen.get("top", 0)),
                screen_right=int(screen.get("right", 0)),
                screen_bottom=int(screen.get("bottom", 0)),
                parent_left=int(parent.get("left", 0)),
                parent_top=int(parent.get("top", 0)),
                parent_right=int(parent.get("right", 0)),
                parent_bottom=int(parent.get("bottom", 0)),
            )

        # Format 3: flat coordinates
        if "x" in data and "y" in data:
            x = int(data["x"])
            y = int(data["y"])
            w = int(data.get("width", 0))
            h = int(data.get("height", 0))
            return BoundsInfo(
                screen_left=x, screen_top=y,
                screen_right=x + w, screen_bottom=y + h,
                parent_left=x, parent_top=y,
                parent_right=x + w, parent_bottom=y + h,
            )

        return BoundsInfo(0, 0, 0, 0, 0, 0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def _to_bool(value: str) -> bool:
    """Convert a uiautomator boolean string to Python bool."""
    return str(value).lower() in ("true", "1", "yes")
