"""Parse JSON accessibility tree formats into AccessibilityTree structures.

Supports:
- Chrome DevTools Protocol accessibility tree snapshots
- axe-core JSON output format
- Generic nested-node JSON format
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


# ── Format detection ──────────────────────────────────────────────────────────

class _Format:
    CHROME_DEVTOOLS = "chrome_devtools"
    AXE_CORE = "axe_core"
    GENERIC = "generic"


def _detect_format(data: Any) -> str:
    """Heuristically detect the JSON format of accessibility data."""
    if isinstance(data, dict):
        # Chrome DevTools: has 'nodes' list with 'nodeId'
        if "nodes" in data and isinstance(data["nodes"], list):
            if data["nodes"] and "nodeId" in data["nodes"][0]:
                return _Format.CHROME_DEVTOOLS

        # axe-core: has 'violations' or 'passes' or 'incomplete' keys
        if any(k in data for k in ("violations", "passes", "incomplete", "inapplicable")):
            return _Format.AXE_CORE

        # Generic tree with 'role' / 'children'
        if "role" in data or "children" in data:
            return _Format.GENERIC

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            if "nodeId" in data[0]:
                return _Format.CHROME_DEVTOOLS
            if "role" in data[0]:
                return _Format.GENERIC

    return _Format.GENERIC


# ── Main parser ───────────────────────────────────────────────────────────────

class JSONAccessibilityParser:
    """Parse JSON accessibility data into :class:`AccessibilityTree`."""

    def __init__(self, *, id_prefix: str = "j") -> None:
        self._id_prefix = id_prefix
        self._counter = 0

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, json_str: str) -> AccessibilityTree:
        """Parse a JSON string into an AccessibilityTree."""
        data = json.loads(json_str)
        return self._parse_data(data)

    def parse_file(self, path: Path) -> AccessibilityTree:
        """Load and parse a JSON file."""
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return self._parse_data(data)

    def parse_dict(self, data: Any) -> AccessibilityTree:
        """Parse an already-loaded dict / list."""
        return self._parse_data(data)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _parse_data(self, data: Any) -> AccessibilityTree:
        self._counter = 0
        fmt = _detect_format(data)

        if fmt == _Format.CHROME_DEVTOOLS:
            root = self._parse_chrome_devtools(data)
        elif fmt == _Format.AXE_CORE:
            root = self._parse_axe_core(data)
        else:
            root = self._parse_generic(data)

        tree = AccessibilityTree(
            root=root,
            metadata={"source": "json", "format": fmt},
        )
        tree.build_index()
        return tree

    # ── Chrome DevTools format ────────────────────────────────────────────

    def _parse_chrome_devtools(self, data: Any) -> AccessibilityNode:
        """Parse Chrome DevTools Protocol accessibility snapshot.

        The snapshot is a flat list of nodes; each has ``nodeId``,
        ``parentId``, ``role``, ``name``, ``properties``, ``childIds``.
        We reconstruct the tree from the flat list.
        """
        raw_nodes: list[dict[str, Any]]
        if isinstance(data, dict) and "nodes" in data:
            raw_nodes = data["nodes"]
        elif isinstance(data, list):
            raw_nodes = data
        else:
            raw_nodes = [data]

        # Index raw nodes
        by_id: dict[int, dict[str, Any]] = {}
        for rn in raw_nodes:
            nid = rn.get("nodeId", rn.get("backendDOMNodeId", 0))
            by_id[nid] = rn

        # Determine root (parentId absent or 0)
        root_candidates = [
            rn for rn in raw_nodes
            if rn.get("parentId") is None or rn.get("parentId") == 0
        ]
        if not root_candidates:
            root_candidates = raw_nodes[:1]

        def build(rn: dict[str, Any], parent_id: Optional[str], depth: int) -> AccessibilityNode:
            nid = rn.get("nodeId", 0)
            str_id = f"{self._id_prefix}-cdp-{nid}"

            role_obj = rn.get("role", {})
            role = role_obj.get("value", "generic") if isinstance(role_obj, dict) else str(role_obj)

            name_obj = rn.get("name", {})
            name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)

            desc_obj = rn.get("description", {})
            desc = desc_obj.get("value", "") if isinstance(desc_obj, dict) else str(desc_obj)

            state = self._chrome_state(rn.get("properties", []))
            bbox = self._chrome_bbox(rn)
            props = self._chrome_properties(rn)

            child_ids = rn.get("childIds", [])
            children: list[AccessibilityNode] = []
            for idx, cid in enumerate(child_ids):
                child_rn = by_id.get(cid)
                if child_rn is not None:
                    child_node = build(child_rn, parent_id=str_id, depth=depth + 1)
                    child_node.index_in_parent = idx
                    children.append(child_node)

            return AccessibilityNode(
                id=str_id,
                role=role,
                name=name,
                description=desc,
                bounding_box=bbox,
                properties=props,
                state=state,
                children=children,
                parent_id=parent_id,
                depth=depth,
            )

        root_rn = root_candidates[0]
        return build(root_rn, parent_id=None, depth=0)

    @staticmethod
    def _chrome_state(props: list[dict[str, Any]]) -> AccessibilityState:
        """Extract AccessibilityState from Chrome properties list."""
        state_map: dict[str, Any] = {}
        for p in props:
            pname = p.get("name", "")
            pval = p.get("value", {})
            if isinstance(pval, dict):
                state_map[pname] = pval.get("value", pval)
            else:
                state_map[pname] = pval

        return AccessibilityState(
            focused=bool(state_map.get("focused", False)),
            selected=bool(state_map.get("selected", False)),
            expanded=bool(state_map.get("expanded", False)),
            checked=state_map.get("checked") if "checked" in state_map else None,
            disabled=bool(state_map.get("disabled", False)),
            hidden=bool(state_map.get("hidden", False)),
            required=bool(state_map.get("required", False)),
            readonly=bool(state_map.get("readonly", False)),
            pressed=state_map.get("pressed") if "pressed" in state_map else None,
            value=str(state_map["value"]) if "value" in state_map else None,
        )

    @staticmethod
    def _chrome_bbox(rn: dict[str, Any]) -> Optional[BoundingBox]:
        bb = rn.get("boundingBox") or rn.get("backendDOMNodeBoundingBox")
        if isinstance(bb, dict):
            try:
                return BoundingBox(
                    x=float(bb.get("x", 0)),
                    y=float(bb.get("y", 0)),
                    width=float(bb.get("width", 0)),
                    height=float(bb.get("height", 0)),
                )
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _chrome_properties(rn: dict[str, Any]) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for p in rn.get("properties", []):
            pname = p.get("name", "")
            pval = p.get("value", {})
            if isinstance(pval, dict):
                props[pname] = pval.get("value", pval)
            else:
                props[pname] = pval
        return props

    # ── axe-core format ───────────────────────────────────────────────────

    def _parse_axe_core(self, data: dict[str, Any]) -> AccessibilityNode:
        """Parse axe-core output into a synthetic AccessibilityTree.

        axe-core output is a flat audit result, not a tree.  We synthesise
        a document root and add each element referenced in violations/passes
        as a child node.
        """
        root_children: list[AccessibilityNode] = []
        idx = 0

        for section_key in ("violations", "passes", "incomplete", "inapplicable"):
            section = data.get(section_key, [])
            for rule in section:
                rule_id = rule.get("id", "unknown")
                for node_data in rule.get("nodes", []):
                    child = self._axe_node(node_data, rule_id, section_key, idx)
                    root_children.append(child)
                    idx += 1

        root = AccessibilityNode(
            id=f"{self._id_prefix}-axe-root",
            role="document",
            name="axe-core audit",
            children=root_children,
            depth=0,
        )
        return root

    def _axe_node(
        self,
        node_data: dict[str, Any],
        rule_id: str,
        section: str,
        idx: int,
    ) -> AccessibilityNode:
        target = node_data.get("target", [])
        html_snippet = node_data.get("html", "")
        node_id = f"{self._id_prefix}-axe-{idx}"

        return AccessibilityNode(
            id=node_id,
            role="generic",
            name=html_snippet[:120] if html_snippet else str(target),
            description=node_data.get("failureSummary", ""),
            properties={
                "axe_rule": rule_id,
                "axe_section": section,
                "axe_target": target,
                "axe_impact": node_data.get("impact", ""),
            },
            parent_id=f"{self._id_prefix}-axe-root",
            depth=1,
            index_in_parent=idx,
        )

    # ── Generic tree format ───────────────────────────────────────────────

    def _parse_generic(self, data: Any) -> AccessibilityNode:
        """Parse a generic JSON tree with role/name/children keys."""
        if isinstance(data, list):
            # Wrap list in a synthetic root
            children = [
                self._parse_node(item, parent_id=f"{self._id_prefix}-root", depth=1)
                for item in data
                if isinstance(item, dict)
            ]
            for i, c in enumerate(children):
                c.index_in_parent = i
            return AccessibilityNode(
                id=f"{self._id_prefix}-root",
                role="document",
                name="root",
                children=children,
                depth=0,
            )
        if isinstance(data, dict):
            return self._parse_node(data, parent_id=None, depth=0)
        raise ValueError(f"Unexpected JSON root type: {type(data)}")

    def _parse_node(
        self,
        data: dict[str, Any],
        parent_id: Optional[str],
        depth: int,
    ) -> AccessibilityNode:
        """Recursively parse a single JSON node."""
        self._counter += 1
        node_id = str(data.get("id", f"{self._id_prefix}-{self._counter}"))
        role = str(data.get("role", "generic"))
        name = str(data.get("name", data.get("label", "")))
        description = str(data.get("description", ""))

        # Bounding box
        bbox = None
        bb_data = data.get("boundingBox") or data.get("bounding_box") or data.get("bounds")
        if isinstance(bb_data, dict):
            try:
                bbox = BoundingBox(
                    x=float(bb_data.get("x", bb_data.get("left", 0))),
                    y=float(bb_data.get("y", bb_data.get("top", 0))),
                    width=float(bb_data.get("width", bb_data.get("w", 0))),
                    height=float(bb_data.get("height", bb_data.get("h", 0))),
                )
            except (ValueError, TypeError):
                pass
        elif isinstance(bb_data, (list, tuple)) and len(bb_data) == 4:
            try:
                bbox = BoundingBox(*[float(v) for v in bb_data])
            except (ValueError, TypeError):
                pass

        # State
        state_data = data.get("state", {})
        state = AccessibilityState.from_dict(state_data) if state_data else AccessibilityState()

        # Properties (everything that's not a standard key)
        known_keys = {
            "id", "role", "name", "label", "description", "boundingBox",
            "bounding_box", "bounds", "state", "children", "parent_id",
        }
        properties = {k: v for k, v in data.items() if k not in known_keys}

        # Children
        children_data = data.get("children", [])
        children: list[AccessibilityNode] = []
        for i, child_data in enumerate(children_data):
            if isinstance(child_data, dict):
                child = self._parse_node(child_data, parent_id=node_id, depth=depth + 1)
                child.index_in_parent = i
                children.append(child)

        return AccessibilityNode(
            id=node_id,
            role=role,
            name=name,
            description=description,
            bounding_box=bbox,
            properties=properties,
            state=state,
            children=children,
            parent_id=parent_id,
            depth=depth,
        )
