"""
usability_oracle.formats.converters — Cross-format conversion utilities.

Provides unified conversion between all supported accessibility tree
formats (ARIA/web, axe-core, Chrome DevTools, Android, iOS, Windows UIA).
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


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class FormatDetector:
    """Auto-detect the format of accessibility tree data."""

    @staticmethod
    def detect(data: str | dict | list) -> str:
        """Detect the format of accessibility data.

        Returns one of: 'aria', 'axe_core', 'chrome_devtools', 'android',
        'ios', 'windows_uia', 'generic_json', 'unknown'.
        """
        if isinstance(data, str):
            stripped = data.strip()
            # XML format (likely Android uiautomator)
            if stripped.startswith("<?xml") or stripped.startswith("<hierarchy"):
                return "android_xml"
            try:
                data = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return "unknown"

        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                sample = data[0]
                if "nodeId" in sample and "role" in sample:
                    return "chrome_devtools"
            return "generic_json"

        if not isinstance(data, dict):
            return "unknown"

        # axe-core: has violations/passes/incomplete/inapplicable
        if "violations" in data and "passes" in data:
            return "axe_core"

        # Chrome DevTools: has 'nodes' with 'nodeId'
        if "nodes" in data:
            nodes = data["nodes"]
            if isinstance(nodes, list) and nodes:
                if "nodeId" in nodes[0] or "backendDOMNodeId" in nodes[0]:
                    return "chrome_devtools"

        # Android: has className with android namespace
        if any(k in data for k in ("className", "resourceId", "packageName")):
            class_name = data.get("className", "")
            if "android." in class_name or "androidx." in class_name:
                return "android"

        # iOS: has elementType or accessibilityTraits
        if any(k in data for k in ("elementType", "accessibilityTraits", "accessibilityLabel")):
            return "ios"

        # Windows UIA: has ControlType or AutomationId
        if any(k in data for k in ("ControlType", "AutomationId", "BoundingRectangle")):
            return "windows_uia"

        # Generic ARIA/JSON: has role field
        if "role" in data:
            return "aria"

        return "generic_json"


# ---------------------------------------------------------------------------
# Unified normalisation
# ---------------------------------------------------------------------------

@dataclass
class NormalisedNode:
    """Platform-agnostic normalised accessibility node."""
    id: str
    role: str
    name: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    disabled: bool = False
    focused: bool = False
    checked: Optional[bool] = None
    selected: bool = False
    hidden: bool = False
    expanded: Optional[bool] = None
    properties: dict[str, str] = field(default_factory=dict)
    children: list["NormalisedNode"] = field(default_factory=list)
    source_format: str = ""


def _normalise_node(node: AccessibilityNode, source: str = "") -> NormalisedNode:
    """Convert an internal AccessibilityNode to NormalisedNode."""
    bbox = node.bounding_box or BoundingBox(x=0, y=0, width=0, height=0)
    state = node.state or AccessibilityState()
    return NormalisedNode(
        id=node.id,
        role=node.role,
        name=node.name,
        x=bbox.x, y=bbox.y,
        width=bbox.width, height=bbox.height,
        disabled=state.disabled,
        focused=state.focused,
        checked=state.checked,
        selected=state.selected,
        hidden=state.hidden,
        expanded=state.expanded,
        properties=dict(node.properties),
        children=[_normalise_node(c, source) for c in node.children],
        source_format=source,
    )


# ---------------------------------------------------------------------------
# FormatConverter
# ---------------------------------------------------------------------------

class FormatConverter:
    """Convert between accessibility tree formats.

    Usage::

        converter = FormatConverter()
        tree = converter.from_any(data)  # Auto-detect and parse
        android_json = converter.to_format(tree, 'android')
    """

    def __init__(self) -> None:
        self._detector = FormatDetector()

    # ------------------------------------------------------------------
    # Auto-detect and parse
    # ------------------------------------------------------------------

    def from_any(self, data: str | dict | list) -> AccessibilityTree:
        """Auto-detect format and parse accessibility data."""
        fmt = self._detector.detect(data)
        return self.from_format(data, fmt)

    def from_format(self, data: str | dict | list, fmt: str) -> AccessibilityTree:
        """Parse data in the specified format."""
        if fmt == "chrome_devtools":
            from usability_oracle.formats.chrome_devtools import ChromeDevToolsParser
            return ChromeDevToolsParser().parse(data)
        elif fmt == "android" or fmt == "android_json":
            from usability_oracle.formats.android import AndroidParser
            if isinstance(data, str) and data.strip().startswith("<"):
                return AndroidParser().parse_xml(data)
            return AndroidParser().parse_json(data)
        elif fmt == "android_xml":
            from usability_oracle.formats.android import AndroidParser
            return AndroidParser().parse_xml(data if isinstance(data, str) else json.dumps(data))
        elif fmt == "ios":
            from usability_oracle.formats.ios import IOSParser
            return IOSParser().parse(data)
        elif fmt == "windows_uia":
            from usability_oracle.formats.windows import WindowsUIAParser
            return WindowsUIAParser().parse(data)
        elif fmt in ("aria", "generic_json"):
            return self._parse_generic(data)
        elif fmt == "axe_core":
            # axe-core doesn't have a tree; return empty with metadata
            root = AccessibilityNode(
                id="axe-root", role="document", name="axe-core result",
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                properties={}, state=AccessibilityState(), children=[], depth=0,
            )
            return AccessibilityTree(root=root, node_index={"axe-root": root})
        else:
            return self._parse_generic(data)

    def _parse_generic(self, data: Any) -> AccessibilityTree:
        """Parse generic JSON accessibility tree."""
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list):
            data = data[0] if data else {}

        root = self._generic_node(data, depth=0)
        idx: dict[str, AccessibilityNode] = {}
        self._index(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    def _generic_node(self, raw: dict, depth: int, counter: list[int] | None = None) -> AccessibilityNode:
        if counter is None:
            counter = [0]
        counter[0] += 1
        nid = str(raw.get("id", f"node-{counter[0]}"))
        role = str(raw.get("role", "generic"))
        name = str(raw.get("name", raw.get("label", raw.get("text", ""))))

        bbox = BoundingBox(x=0, y=0, width=0, height=0)
        bbox_data = raw.get("bounding_box", raw.get("bbox", raw.get("bounds", None)))
        if isinstance(bbox_data, dict):
            bbox = BoundingBox(
                x=bbox_data.get("x", 0), y=bbox_data.get("y", 0),
                width=bbox_data.get("width", bbox_data.get("w", 0)),
                height=bbox_data.get("height", bbox_data.get("h", 0)),
            )

        node = AccessibilityNode(
            id=nid, role=role, name=name,
            bounding_box=bbox,
            properties=raw.get("properties", {}),
            state=AccessibilityState(),
            children=[],
            depth=depth,
        )

        for child in raw.get("children", []):
            if isinstance(child, dict):
                cn = self._generic_node(child, depth + 1, counter)
                cn.parent_id = nid
                node.children.append(cn)

        return node

    @staticmethod
    def _index(node: AccessibilityNode, idx: dict[str, AccessibilityNode]) -> None:
        idx[node.id] = node
        for child in node.children:
            FormatConverter._index(child, idx)

    # ------------------------------------------------------------------
    # Export to formats
    # ------------------------------------------------------------------

    def to_format(self, tree: AccessibilityTree, fmt: str) -> Any:
        """Export tree to the specified format."""
        if fmt == "chrome_devtools":
            from usability_oracle.formats.chrome_devtools import ChromeDevToolsParser
            return ChromeDevToolsParser().to_cdp_format(tree)
        elif fmt == "ios":
            from usability_oracle.formats.ios import IOSParser
            return IOSParser().to_ios_format(tree)
        elif fmt == "json":
            return self._to_generic_json(tree.root)
        else:
            return self._to_generic_json(tree.root)

    def _to_generic_json(self, node: AccessibilityNode) -> dict:
        d: dict[str, Any] = {
            "id": node.id,
            "role": node.role,
            "name": node.name,
        }
        if node.bounding_box:
            d["bounding_box"] = {
                "x": node.bounding_box.x,
                "y": node.bounding_box.y,
                "width": node.bounding_box.width,
                "height": node.bounding_box.height,
            }
        if node.properties:
            d["properties"] = node.properties
        if node.children:
            d["children"] = [self._to_generic_json(c) for c in node.children]
        return d

    # ------------------------------------------------------------------
    # Normalised comparison
    # ------------------------------------------------------------------

    def normalise(self, tree: AccessibilityTree, source_format: str = "") -> NormalisedNode:
        """Convert a tree to the normalised representation for cross-platform comparison."""
        return _normalise_node(tree.root, source_format)

    def detect_format(self, data: Any) -> str:
        """Detect the format of accessibility data."""
        return self._detector.detect(data)
