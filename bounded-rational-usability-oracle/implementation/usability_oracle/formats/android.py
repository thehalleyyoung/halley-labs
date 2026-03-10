"""
usability_oracle.formats.android — Android AccessibilityNodeInfo format.

Parses Android accessibility tree dumps (from uiautomator or
AccessibilityService) and converts them to the oracle's internal format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


_ANDROID_CLASS_MAP = {
    "android.widget.Button": "button",
    "android.widget.ImageButton": "button",
    "android.widget.ToggleButton": "button",
    "android.widget.EditText": "textfield",
    "android.widget.AutoCompleteTextView": "combobox",
    "android.widget.CheckBox": "checkbox",
    "android.widget.RadioButton": "radio",
    "android.widget.Switch": "checkbox",
    "android.widget.SeekBar": "slider",
    "android.widget.Spinner": "combobox",
    "android.widget.ImageView": "image",
    "android.widget.TextView": "generic",
    "android.widget.ListView": "list",
    "android.widget.GridView": "list",
    "android.widget.RecyclerView": "list",
    "android.widget.ScrollView": "region",
    "android.widget.HorizontalScrollView": "region",
    "android.widget.TabHost": "tab",
    "android.widget.TabWidget": "tab",
    "android.widget.ProgressBar": "generic",
    "android.widget.LinearLayout": "group",
    "android.widget.RelativeLayout": "group",
    "android.widget.FrameLayout": "group",
    "android.widget.TableLayout": "table",
    "android.widget.TableRow": "row",
    "android.view.View": "generic",
    "android.view.ViewGroup": "group",
    "android.webkit.WebView": "document",
    "android.app.ActionBar": "toolbar",
    "android.widget.Toolbar": "toolbar",
    "android.support.v7.widget.Toolbar": "toolbar",
    "androidx.appcompat.widget.Toolbar": "toolbar",
    "android.support.design.widget.NavigationView": "navigation",
    "com.google.android.material.navigation.NavigationView": "navigation",
    "android.support.design.widget.TabLayout": "tab",
    "com.google.android.material.tabs.TabLayout": "tab",
    "android.widget.SearchView": "search",
    "androidx.appcompat.widget.SearchView": "search",
}


class AndroidParser:
    """Parse Android accessibility tree dumps.

    Supports:
    - uiautomator XML dump format
    - AccessibilityNodeInfo JSON format
    - ViewHierarchy JSON format
    """

    def __init__(self) -> None:
        self._class_map = dict(_ANDROID_CLASS_MAP)
        self._counter = 0

    # ------------------------------------------------------------------
    # Parse JSON format
    # ------------------------------------------------------------------

    def parse_json(self, data: str | dict) -> AccessibilityTree:
        """Parse Android accessibility JSON data."""
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        root_data = data.get("root", data) if isinstance(data, dict) else data
        root = self._convert_json_node(root_data, depth=0)
        idx: dict[str, AccessibilityNode] = {}
        self._index_tree(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    def _convert_json_node(self, raw: dict, depth: int) -> AccessibilityNode:
        self._counter += 1
        nid = raw.get("resourceId", raw.get("id", f"android-{self._counter}"))

        class_name = raw.get("className", raw.get("class", "android.view.View"))
        role = self._map_class(class_name)

        # Content description or text
        name = raw.get("contentDescription", "")
        if not name:
            name = raw.get("text", "")
        if not name:
            name = raw.get("hintText", "")

        bounds = self._parse_bounds(raw)
        state = self._parse_state(raw)

        properties: dict[str, str] = {}
        if raw.get("resourceId"):
            properties["resource-id"] = raw["resourceId"]
        if raw.get("packageName"):
            properties["package"] = raw["packageName"]
        if class_name:
            properties["class"] = class_name

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

        for child_data in raw.get("children", []):
            if isinstance(child_data, dict):
                child = self._convert_json_node(child_data, depth + 1)
                child.parent_id = node.id
                node.children.append(child)

        return node

    # ------------------------------------------------------------------
    # Parse XML format (uiautomator dump)
    # ------------------------------------------------------------------

    def parse_xml(self, xml_string: str) -> AccessibilityTree:
        """Parse uiautomator XML dump format."""
        self._counter = 0

        # Simple regex-based XML parsing (avoids xml.etree dependency issues)
        nodes = self._parse_xml_nodes(xml_string)
        if not nodes:
            root = self._make_empty_root()
            return AccessibilityTree(root=root, node_index={root.id: root})

        root = nodes[0]
        idx: dict[str, AccessibilityNode] = {}
        self._index_tree(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    def _parse_xml_nodes(self, xml: str) -> list[AccessibilityNode]:
        """Parse node elements from uiautomator XML."""
        pattern = re.compile(
            r'<node\s+([^>]+?)(?:/>|>(.*?)</node>)',
            re.DOTALL,
        )

        nodes: list[AccessibilityNode] = []
        stack: list[AccessibilityNode] = []

        for match in pattern.finditer(xml):
            attrs_str = match.group(1)
            attrs = self._parse_xml_attrs(attrs_str)

            self._counter += 1
            nid = attrs.get("resource-id", f"xml-{self._counter}")
            class_name = attrs.get("class", "android.view.View")
            role = self._map_class(class_name)

            name = attrs.get("content-desc", "")
            if not name:
                name = attrs.get("text", "")

            bounds = self._parse_bounds_string(attrs.get("bounds", ""))
            state = AccessibilityState(
                focused=attrs.get("focused") == "true",
                checked=attrs.get("checked") == "true" if attrs.get("checkable") == "true" else None,
                selected=attrs.get("selected") == "true",
                disabled=attrs.get("enabled") != "true",
            )

            node = AccessibilityNode(
                id=str(nid),
                role=role,
                name=name,
                bounding_box=bounds,
                properties={"class": class_name},
                state=state,
                children=[],
                depth=len(stack),
            )

            if stack:
                stack[-1].children.append(node)
                node.parent_id = stack[-1].id
            else:
                nodes.append(node)

            # Determine if self-closing or has children
            if match.group(2) is not None:
                stack.append(node)

        return nodes

    def _parse_xml_attrs(self, attrs_str: str) -> dict[str, str]:
        """Parse XML attributes."""
        result: dict[str, str] = {}
        for match in re.finditer(r'(\w[\w-]*)="([^"]*)"', attrs_str):
            result[match.group(1)] = match.group(2)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_class(self, class_name: str) -> str:
        if class_name in self._class_map:
            return self._class_map[class_name]
        # Try matching suffix
        short = class_name.split(".")[-1] if "." in class_name else class_name
        for full_class, role in self._class_map.items():
            if full_class.endswith(short):
                return role
        return "generic"

    def _parse_bounds(self, raw: dict) -> BoundingBox:
        bounds = raw.get("bounds", raw.get("boundsInScreen", None))
        if isinstance(bounds, dict):
            return BoundingBox(
                x=bounds.get("left", 0),
                y=bounds.get("top", 0),
                width=bounds.get("right", 0) - bounds.get("left", 0),
                height=bounds.get("bottom", 0) - bounds.get("top", 0),
            )
        if isinstance(bounds, str):
            return self._parse_bounds_string(bounds)
        return BoundingBox(x=0, y=0, width=0, height=0)

    @staticmethod
    def _parse_bounds_string(s: str) -> BoundingBox:
        """Parse '[left,top][right,bottom]' format."""
        match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', s)
        if match:
            left, top, right, bottom = (int(x) for x in match.groups())
            return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)
        return BoundingBox(x=0, y=0, width=0, height=0)

    def _parse_state(self, raw: dict) -> AccessibilityState:
        return AccessibilityState(
            focused=raw.get("focused", False) or raw.get("isFocused", False),
            checked=raw.get("checked", None) if raw.get("checkable", False) else None,
            selected=raw.get("selected", False),
            disabled=not raw.get("enabled", True),
            hidden=not raw.get("visibleToUser", True),
        )

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="group", name="",
            bounding_box=BoundingBox(x=0, y=0, width=1080, height=1920),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )

    @staticmethod
    def _index_tree(node: AccessibilityNode, idx: dict[str, AccessibilityNode]) -> None:
        idx[node.id] = node
        for child in node.children:
            AndroidParser._index_tree(child, idx)
