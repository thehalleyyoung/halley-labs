"""
usability_oracle.formats.ios — iOS UIAccessibility format.

Parses iOS accessibility tree dumps from UIAccessibility framework
and XCTest accessibility inspector and converts to internal format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


_IOS_TRAIT_MAP = {
    "UIAccessibilityTraitButton": "button",
    "UIAccessibilityTraitLink": "link",
    "UIAccessibilityTraitImage": "image",
    "UIAccessibilityTraitSearchField": "search",
    "UIAccessibilityTraitHeader": "heading",
    "UIAccessibilityTraitStaticText": "generic",
    "UIAccessibilityTraitTabBar": "tab",
    "UIAccessibilityTraitAdjustable": "slider",
    "UIAccessibilityTraitKeyboardKey": "button",
    "UIAccessibilityTraitSelected": "generic",
    "UIAccessibilityTraitSummaryElement": "generic",
    "UIAccessibilityTraitUpdatesFrequently": "generic",
    "UIAccessibilityTraitNotEnabled": "generic",
}

_IOS_ELEMENT_TYPE_MAP = {
    "Button": "button",
    "Link": "link",
    "TextField": "textfield",
    "SecureTextField": "textfield",
    "TextView": "textfield",
    "StaticText": "generic",
    "Image": "image",
    "Switch": "checkbox",
    "Slider": "slider",
    "Picker": "combobox",
    "DatePicker": "combobox",
    "PageIndicator": "generic",
    "ProgressIndicator": "generic",
    "ActivityIndicator": "generic",
    "SegmentedControl": "tab",
    "Tab": "tab",
    "TabBar": "tab",
    "NavigationBar": "navigation",
    "Toolbar": "toolbar",
    "SearchField": "search",
    "Table": "table",
    "TableRow": "row",
    "Cell": "cell",
    "CollectionView": "list",
    "ScrollView": "region",
    "WebView": "document",
    "Alert": "alert",
    "Sheet": "dialog",
    "Dialog": "dialog",
    "Menu": "menu",
    "MenuItem": "menuitem",
    "Map": "image",
    "Window": "region",
    "Group": "group",
    "Other": "generic",
}


class IOSParser:
    """Parse iOS UIAccessibility tree dumps.

    Supports:
    - XCTest accessibility snapshot JSON
    - Appium inspector JSON format
    - Raw UIAccessibility element hierarchy
    """

    def __init__(self) -> None:
        self._trait_map = dict(_IOS_TRAIT_MAP)
        self._type_map = dict(_IOS_ELEMENT_TYPE_MAP)
        self._counter = 0

    # ------------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------------

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse iOS accessibility JSON data."""
        self._counter = 0
        if isinstance(data, str):
            data = json.loads(data)

        root_data = data.get("hierarchy", data.get("root", data))
        root = self._convert_node(root_data, depth=0)
        idx: dict[str, AccessibilityNode] = {}
        self._index_tree(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    def _convert_node(self, raw: dict, depth: int) -> AccessibilityNode:
        self._counter += 1
        nid = raw.get("identifier", raw.get("id", f"ios-{self._counter}"))

        # Determine role from element type or traits
        element_type = raw.get("elementType", raw.get("type", "Other"))
        role = self._type_map.get(element_type, "generic")

        traits = raw.get("traits", raw.get("accessibilityTraits", []))
        if isinstance(traits, list):
            for trait in traits:
                mapped = self._trait_map.get(trait)
                if mapped and mapped != "generic":
                    role = mapped
                    break

        # Name
        name = raw.get("label", "")
        if not name:
            name = raw.get("accessibilityLabel", "")
        if not name:
            name = raw.get("title", "")
        if not name:
            name = raw.get("value", "")

        bounds = self._parse_frame(raw)
        state = self._parse_state(raw, traits)

        properties: dict[str, str] = {}
        hint = raw.get("hint", raw.get("accessibilityHint", ""))
        if hint:
            properties["hint"] = hint
        value = raw.get("value", raw.get("accessibilityValue", ""))
        if value:
            properties["value"] = str(value)
        if raw.get("identifier"):
            properties["identifier"] = raw["identifier"]

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

        children_data = raw.get("children", raw.get("elements", []))
        for child_data in children_data:
            if isinstance(child_data, dict):
                child = self._convert_node(child_data, depth + 1)
                child.parent_id = node.id
                node.children.append(child)

        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_frame(self, raw: dict) -> BoundingBox:
        frame = raw.get("frame", raw.get("rect", raw.get("accessibilityFrame", None)))
        if isinstance(frame, dict):
            x = frame.get("x", frame.get("X", frame.get("origin", {}).get("x", 0)))
            y = frame.get("y", frame.get("Y", frame.get("origin", {}).get("y", 0)))
            w = frame.get("width", frame.get("Width", frame.get("size", {}).get("width", 0)))
            h = frame.get("height", frame.get("Height", frame.get("size", {}).get("height", 0)))
            return BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
        if isinstance(frame, str):
            # Parse "{{x, y}, {w, h}}" format
            import re
            match = re.match(r'\{\{([\d.]+),\s*([\d.]+)\},\s*\{([\d.]+),\s*([\d.]+)\}\}', frame)
            if match:
                return BoundingBox(
                    x=float(match.group(1)), y=float(match.group(2)),
                    width=float(match.group(3)), height=float(match.group(4)),
                )
        return BoundingBox(x=0, y=0, width=0, height=0)

    def _parse_state(self, raw: dict, traits: Any) -> AccessibilityState:
        is_enabled = raw.get("isEnabled", raw.get("enabled", True))
        is_selected = raw.get("isSelected", raw.get("selected", False))
        is_focused = raw.get("hasFocus", raw.get("focused", False))

        checked = None
        if raw.get("elementType") in ("Switch", "CheckBox"):
            value = raw.get("value", "")
            if isinstance(value, bool):
                checked = value
            elif isinstance(value, str):
                checked = value in ("1", "true", "on")

        trait_set = set(traits) if isinstance(traits, list) else set()
        hidden = "UIAccessibilityTraitNotEnabled" in trait_set

        return AccessibilityState(
            disabled=not is_enabled,
            selected=is_selected,
            focused=is_focused,
            checked=checked,
            hidden=hidden,
        )

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="region", name="",
            bounding_box=BoundingBox(x=0, y=0, width=375, height=812),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )

    @staticmethod
    def _index_tree(node: AccessibilityNode, idx: dict[str, AccessibilityNode]) -> None:
        idx[node.id] = node
        for child in node.children:
            IOSParser._index_tree(child, idx)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_ios_format(self, tree: AccessibilityTree) -> dict:
        """Convert internal tree to iOS-style JSON."""
        return self._node_to_ios(tree.root)

    def _node_to_ios(self, node: AccessibilityNode) -> dict:
        result: dict[str, Any] = {
            "identifier": node.id,
            "label": node.name,
            "elementType": self._reverse_type(node.role),
        }
        if node.bounding_box:
            result["frame"] = {
                "x": node.bounding_box.x,
                "y": node.bounding_box.y,
                "width": node.bounding_box.width,
                "height": node.bounding_box.height,
            }
        if node.children:
            result["children"] = [self._node_to_ios(c) for c in node.children]
        return result

    def _reverse_type(self, role: str) -> str:
        for ios_type, mapped_role in self._type_map.items():
            if mapped_role == role:
                return ios_type
        return "Other"
