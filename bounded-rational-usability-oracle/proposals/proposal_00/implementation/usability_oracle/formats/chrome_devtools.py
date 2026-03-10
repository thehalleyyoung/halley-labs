"""
usability_oracle.formats.chrome_devtools — Chrome DevTools accessibility tree format.

Parses Chrome DevTools Protocol (CDP) accessibility tree snapshots
and converts them to the oracle's internal representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


_CDP_ROLE_MAP = {
    "button": "button",
    "link": "link",
    "textbox": "textfield",
    "TextField": "textfield",
    "CheckBox": "checkbox",
    "RadioButton": "radio",
    "ComboBox": "combobox",
    "slider": "slider",
    "tab": "tab",
    "TabItem": "tab",
    "menuitem": "menuitem",
    "MenuItem": "menuitem",
    "heading": "heading",
    "image": "image",
    "Img": "image",
    "navigation": "navigation",
    "banner": "banner",
    "main": "main",
    "contentinfo": "contentinfo",
    "complementary": "complementary",
    "form": "form",
    "search": "search",
    "region": "region",
    "dialog": "dialog",
    "alert": "alert",
    "table": "table",
    "row": "row",
    "cell": "cell",
    "list": "list",
    "listitem": "listitem",
    "tree": "tree",
    "treeitem": "treeitem",
    "group": "group",
    "toolbar": "toolbar",
    "separator": "separator",
    "menu": "menu",
    "RootWebArea": "document",
    "WebArea": "document",
    "GenericContainer": "generic",
    "generic": "generic",
    "StaticText": "generic",
    "InlineTextBox": "generic",
    "Iframe": "region",
    "IframePresentational": "region",
}


class ChromeDevToolsParser:
    """Parse Chrome DevTools Protocol accessibility tree snapshots.

    Supports CDP Accessibility.getFullAXTree output format.
    """

    def __init__(self) -> None:
        self._role_map = dict(_CDP_ROLE_MAP)
        self._counter = 0

    # ------------------------------------------------------------------
    # Parse CDP AXTree
    # ------------------------------------------------------------------

    def parse(self, data: str | dict | list) -> AccessibilityTree:
        """Parse a CDP accessibility tree snapshot.

        Parameters:
            data: JSON string, dict, or list of AXNode objects from CDP.
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            nodes = data.get("nodes", [data])
        elif isinstance(data, list):
            nodes = data
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")

        if not nodes:
            root = self._make_empty_root()
            return AccessibilityTree(root=root, node_index={root.id: root})

        # Build node index
        node_map: dict[str, dict] = {}
        for raw_node in nodes:
            nid = self._get_node_id(raw_node)
            node_map[nid] = raw_node

        # Find root (first node or node with no parent)
        root_id = self._get_node_id(nodes[0])
        for raw_node in nodes:
            if not raw_node.get("parentId"):
                root_id = self._get_node_id(raw_node)
                break

        # Convert nodes
        converted: dict[str, AccessibilityNode] = {}
        root = self._convert_node(node_map.get(root_id, nodes[0]), node_map, converted, depth=0)

        return AccessibilityTree(root=root, node_index=converted)

    def _convert_node(
        self,
        raw: dict,
        node_map: dict[str, dict],
        converted: dict[str, AccessibilityNode],
        depth: int,
    ) -> AccessibilityNode:
        nid = self._get_node_id(raw)

        role = self._extract_role(raw)
        name = self._extract_name(raw)
        bbox = self._extract_bbox(raw)
        state = self._extract_state(raw)
        properties = self._extract_properties(raw)

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            bounding_box=bbox,
            properties=properties,
            state=state,
            children=[],
            depth=depth,
        )
        converted[nid] = node

        # Process children
        child_ids = raw.get("childIds", raw.get("children", []))
        for child_ref in child_ids:
            child_id = child_ref if isinstance(child_ref, str) else str(child_ref)
            child_raw = node_map.get(child_id)
            if child_raw:
                child_node = self._convert_node(child_raw, node_map, converted, depth + 1)
                child_node.parent_id = nid
                node.children.append(child_node)

        return node

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _get_node_id(self, raw: dict) -> str:
        nid = raw.get("nodeId", raw.get("id", ""))
        if not nid:
            self._counter += 1
            nid = f"cdp-{self._counter}"
        return str(nid)

    def _extract_role(self, raw: dict) -> str:
        role_obj = raw.get("role", {})
        if isinstance(role_obj, dict):
            role_str = role_obj.get("value", "generic")
        else:
            role_str = str(role_obj)
        return self._role_map.get(role_str, role_str.lower())

    def _extract_name(self, raw: dict) -> str:
        name_obj = raw.get("name", {})
        if isinstance(name_obj, dict):
            return name_obj.get("value", "")
        return str(name_obj) if name_obj else ""

    def _extract_bbox(self, raw: dict) -> Optional[BoundingBox]:
        bbox = raw.get("backendDOMNodeId")  # CDP doesn't directly provide bbox in AXTree
        # Try to get from associated DOM node model
        bound = raw.get("boundingBox", raw.get("bounds", None))
        if isinstance(bound, dict):
            return BoundingBox(
                x=bound.get("x", 0),
                y=bound.get("y", 0),
                width=bound.get("width", 0),
                height=bound.get("height", 0),
            )
        if isinstance(bound, list) and len(bound) == 4:
            return BoundingBox(x=bound[0], y=bound[1], width=bound[2], height=bound[3])
        return BoundingBox(x=0, y=0, width=0, height=0)

    def _extract_state(self, raw: dict) -> AccessibilityState:
        props = raw.get("properties", [])
        state = AccessibilityState()

        if isinstance(props, list):
            for p in props:
                name = p.get("name", "")
                value = p.get("value", {})
                val = value.get("value", value) if isinstance(value, dict) else value

                if name == "disabled":
                    state.disabled = bool(val)
                elif name == "hidden":
                    state.hidden = bool(val)
                elif name == "focused":
                    state.focused = bool(val)
                elif name == "checked":
                    state.checked = val if isinstance(val, bool) else str(val).lower() == "true"
                elif name == "selected":
                    state.selected = bool(val)
                elif name == "expanded":
                    state.expanded = bool(val)

        return state

    def _extract_properties(self, raw: dict) -> dict[str, str]:
        result: dict[str, str] = {}
        props = raw.get("properties", [])
        if isinstance(props, list):
            for p in props:
                name = p.get("name", "")
                value = p.get("value", {})
                val = value.get("value", value) if isinstance(value, dict) else value
                if name and name not in ("disabled", "hidden", "focused", "checked", "selected", "expanded"):
                    result[name] = str(val)
        return result

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_cdp_format(self, tree: AccessibilityTree) -> list[dict]:
        """Convert an internal tree back to CDP AXNode format."""
        nodes = []
        self._tree_to_cdp(tree.root, nodes)
        return nodes

    def _tree_to_cdp(self, node: AccessibilityNode, out: list[dict]) -> None:
        cdp_node: dict[str, Any] = {
            "nodeId": node.id,
            "role": {"value": node.role},
            "name": {"value": node.name},
            "childIds": [c.id for c in node.children],
        }
        if node.bounding_box:
            cdp_node["boundingBox"] = {
                "x": node.bounding_box.x,
                "y": node.bounding_box.y,
                "width": node.bounding_box.width,
                "height": node.bounding_box.height,
            }
        props = []
        if node.state:
            if node.state.disabled:
                props.append({"name": "disabled", "value": {"value": True}})
            if node.state.focused:
                props.append({"name": "focused", "value": {"value": True}})
            if node.state.checked is not None:
                props.append({"name": "checked", "value": {"value": node.state.checked}})
        cdp_node["properties"] = props
        out.append(cdp_node)
        for child in node.children:
            self._tree_to_cdp(child, out)
