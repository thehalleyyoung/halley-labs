"""
usability_oracle.formats.windows — Windows UI Automation format.

Parses Windows UI Automation (UIA) accessibility tree data and converts
to the oracle's internal representation. Supports UIA property dumps
from tools like Inspect.exe, Accessibility Insights, and UI Spy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


_UIA_CONTROL_TYPE_MAP = {
    "Button": "button",
    "Calendar": "generic",
    "CheckBox": "checkbox",
    "ComboBox": "combobox",
    "Custom": "generic",
    "DataGrid": "table",
    "DataItem": "row",
    "Document": "document",
    "Edit": "textfield",
    "Group": "group",
    "Header": "heading",
    "HeaderItem": "heading",
    "Hyperlink": "link",
    "Image": "image",
    "List": "list",
    "ListItem": "listitem",
    "Menu": "menu",
    "MenuBar": "menu",
    "MenuItem": "menuitem",
    "Pane": "region",
    "ProgressBar": "generic",
    "RadioButton": "radio",
    "ScrollBar": "generic",
    "Separator": "separator",
    "Slider": "slider",
    "Spinner": "combobox",
    "SplitButton": "button",
    "StatusBar": "generic",
    "Tab": "tab",
    "TabItem": "tab",
    "Table": "table",
    "Text": "generic",
    "Thumb": "generic",
    "TitleBar": "banner",
    "ToolBar": "toolbar",
    "ToolTip": "generic",
    "Tree": "tree",
    "TreeItem": "treeitem",
    "Window": "region",
}

_UIA_LANDMARK_MAP = {
    "Custom": "region",
    "Form": "form",
    "Main": "main",
    "Navigation": "navigation",
    "Search": "search",
}


class WindowsUIAParser:
    """Parse Windows UI Automation accessibility tree data.

    Supports UIA element hierarchy JSON dumps.
    """

    def __init__(self) -> None:
        self._control_map = dict(_UIA_CONTROL_TYPE_MAP)
        self._landmark_map = dict(_UIA_LANDMARK_MAP)
        self._counter = 0

    # ------------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------------

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse Windows UIA JSON data."""
        self._counter = 0
        if isinstance(data, str):
            data = json.loads(data)

        root_data = data.get("root", data.get("element", data))
        root = self._convert_node(root_data, depth=0)
        idx: dict[str, AccessibilityNode] = {}
        self._index_tree(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    def _convert_node(self, raw: dict, depth: int) -> AccessibilityNode:
        self._counter += 1
        nid = raw.get("AutomationId", raw.get("RuntimeId", f"uia-{self._counter}"))

        control_type = raw.get("ControlType", raw.get("controlType", "Custom"))
        # Remove "ControlType." prefix if present
        control_type = control_type.replace("ControlType.", "")
        role = self._map_control_type(control_type, raw)

        # Name from multiple sources
        name = raw.get("Name", "")
        if not name:
            name = raw.get("HelpText", "")
        if not name:
            name = raw.get("LocalizedControlType", "")

        bounds = self._parse_bounding_rect(raw)
        state = self._parse_state(raw)

        properties: dict[str, str] = {}
        if raw.get("AutomationId"):
            properties["automation-id"] = str(raw["AutomationId"])
        if raw.get("ClassName"):
            properties["class-name"] = raw["ClassName"]
        if raw.get("FrameworkId"):
            properties["framework"] = raw["FrameworkId"]
        if raw.get("ProcessId"):
            properties["process-id"] = str(raw["ProcessId"])
        if raw.get("AcceleratorKey"):
            properties["accelerator-key"] = raw["AcceleratorKey"]
        if raw.get("AccessKey"):
            properties["access-key"] = raw["AccessKey"]
        if raw.get("ItemStatus"):
            properties["item-status"] = raw["ItemStatus"]

        node = AccessibilityNode(
            id=str(nid),
            role=role,
            name=name,
            bounding_box=bounds,
            properties=properties,
            state=state,
            children=[],
            depth=depth,
        )

        for child_data in raw.get("children", raw.get("Children", [])):
            if isinstance(child_data, dict):
                child = self._convert_node(child_data, depth + 1)
                child.parent_id = node.id
                node.children.append(child)

        return node

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _map_control_type(self, control_type: str, raw: dict) -> str:
        # Check landmark type first
        landmark = raw.get("LandmarkType", raw.get("landmarkType", ""))
        if landmark and landmark in self._landmark_map:
            return self._landmark_map[landmark]

        return self._control_map.get(control_type, "generic")

    def _parse_bounding_rect(self, raw: dict) -> BoundingBox:
        rect = raw.get("BoundingRectangle", raw.get("boundingRectangle", None))
        if isinstance(rect, dict):
            return BoundingBox(
                x=rect.get("left", rect.get("x", 0)),
                y=rect.get("top", rect.get("y", 0)),
                width=rect.get("width", rect.get("right", 0) - rect.get("left", 0)),
                height=rect.get("height", rect.get("bottom", 0) - rect.get("top", 0)),
            )
        if isinstance(rect, str):
            parts = re.findall(r'[\d.]+', rect)
            if len(parts) == 4:
                left, top, width, height = (float(p) for p in parts)
                return BoundingBox(x=left, y=top, width=width, height=height)
        if isinstance(rect, list) and len(rect) == 4:
            return BoundingBox(x=rect[0], y=rect[1], width=rect[2], height=rect[3])
        return BoundingBox(x=0, y=0, width=0, height=0)

    def _parse_state(self, raw: dict) -> AccessibilityState:
        is_enabled = raw.get("IsEnabled", raw.get("isEnabled", True))
        is_selected = raw.get("SelectionItemPattern.IsSelected",
                              raw.get("isSelected", False))

        has_focus = raw.get("HasKeyboardFocus",
                            raw.get("hasKeyboardFocus", False))

        checked = None
        toggle_state = raw.get("TogglePattern.ToggleState",
                               raw.get("toggleState", None))
        if toggle_state is not None:
            if toggle_state in (1, "On", True, "Checked"):
                checked = True
            elif toggle_state in (0, "Off", False, "Unchecked"):
                checked = False

        is_offscreen = raw.get("IsOffscreen", raw.get("isOffscreen", False))
        expanded = raw.get("ExpandCollapsePattern.ExpandCollapseState", None)
        if expanded is not None:
            expanded = expanded in ("Expanded", "PartiallyExpanded", 1)

        return AccessibilityState(
            disabled=not is_enabled,
            selected=bool(is_selected),
            focused=bool(has_focus),
            checked=checked,
            hidden=bool(is_offscreen),
            expanded=expanded if isinstance(expanded, bool) else None,
        )

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="region", name="",
            bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )

    @staticmethod
    def _index_tree(node: AccessibilityNode, idx: dict[str, AccessibilityNode]) -> None:
        idx[node.id] = node
        for child in node.children:
            WindowsUIAParser._index_tree(child, idx)

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect_patterns(self, tree: AccessibilityTree) -> dict[str, list[str]]:
        """Detect UIA patterns supported by elements in the tree."""
        patterns: dict[str, list[str]] = {}

        def _walk(node: AccessibilityNode) -> None:
            node_patterns = []
            props = node.properties
            role = node.role

            if role in ("button", "link", "menuitem"):
                node_patterns.append("InvokePattern")
            if role in ("textfield",):
                node_patterns.append("ValuePattern")
                node_patterns.append("TextPattern")
            if role in ("checkbox", "radio"):
                node_patterns.append("TogglePattern")
            if role in ("slider",):
                node_patterns.append("RangeValuePattern")
            if role in ("combobox",):
                node_patterns.append("SelectionPattern")
                node_patterns.append("ExpandCollapsePattern")
            if role in ("tab",):
                node_patterns.append("SelectionItemPattern")
            if role in ("table",):
                node_patterns.append("TablePattern")
                node_patterns.append("GridPattern")
            if role in ("tree", "treeitem"):
                node_patterns.append("ExpandCollapsePattern")
            if role in ("region",) and node.children:
                node_patterns.append("ScrollPattern")

            if node_patterns:
                patterns[node.id] = node_patterns

            for child in node.children:
                _walk(child)

        _walk(tree.root)
        return patterns
