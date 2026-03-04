"""
usability_oracle.formats.playwright — Playwright accessibility snapshot format.

Parses the JSON output of Playwright's ``page.accessibility.snapshot()`` and
converts it to the oracle's internal :class:`AccessibilityTree`.

Usage::

    snapshot = await page.accessibility.snapshot()
    parser = PlaywrightParser()
    tree = parser.parse(snapshot)
"""

from __future__ import annotations

import json
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)

# Playwright roles → normalised oracle roles
_PLAYWRIGHT_ROLE_MAP = {
    "WebArea": "document",
    "RootWebArea": "document",
    "button": "button",
    "link": "link",
    "textbox": "textfield",
    "checkbox": "checkbox",
    "radio": "radio",
    "combobox": "combobox",
    "slider": "slider",
    "tab": "tab",
    "menuitem": "menuitem",
    "heading": "heading",
    "img": "image",
    "image": "image",
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
    "text": "generic",
    "paragraph": "paragraph",
    "generic": "generic",
    "none": "generic",
    "presentation": "generic",
    "option": "option",
    "spinbutton": "spinbutton",
    "switch": "switch",
    "progressbar": "progressbar",
    "status": "status",
    "log": "log",
    "marquee": "marquee",
    "timer": "timer",
    "tooltip": "tooltip",
    "gridcell": "gridcell",
    "columnheader": "columnheader",
    "rowheader": "rowheader",
    "definition": "definition",
    "term": "term",
    "article": "article",
    "figure": "figure",
    "math": "math",
    "note": "note",
    "directory": "directory",
    "feed": "feed",
    "grid": "grid",
    "tabpanel": "tabpanel",
    "tablist": "tablist",
    "menubar": "menubar",
    "menuitemcheckbox": "menuitemcheckbox",
    "menuitemradio": "menuitemradio",
}


class PlaywrightParser:
    """Parse Playwright ``page.accessibility.snapshot()`` output.

    Run ``await page.accessibility.snapshot()`` in Playwright to obtain the
    input data.  The snapshot returns a nested JSON tree where each node has
    ``role``, ``name``, and optionally ``value``, ``description``, ``children``,
    ``checked``, ``pressed``, ``level``, ``expanded``, ``disabled``,
    ``selected``, ``focused``, ``readonly``, ``required``, etc.
    """

    def __init__(self) -> None:
        self._role_map = dict(_PLAYWRIGHT_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse a Playwright accessibility snapshot.

        Parameters
        ----------
        data : str or dict
            JSON string or dict from ``page.accessibility.snapshot()``.

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        converted: dict[str, AccessibilityNode] = {}
        root = self._convert_node(data, converted, depth=0)

        tree = AccessibilityTree(root=root, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # Node conversion
    # ------------------------------------------------------------------

    def _convert_node(
        self,
        raw: dict,
        converted: dict[str, AccessibilityNode],
        depth: int,
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"pw-{self._counter}"

        role = self._role_map.get(raw.get("role", ""), raw.get("role", "generic").lower())
        name = raw.get("name", "")
        description = raw.get("description", "")

        state = AccessibilityState(
            focused=bool(raw.get("focused", False)),
            selected=bool(raw.get("selected", False)),
            expanded=raw.get("expanded", False) if raw.get("expanded") is not None else False,
            checked=raw.get("checked") if raw.get("checked") is not None else None,
            disabled=bool(raw.get("disabled", False)),
            hidden=False,
            required=bool(raw.get("required", False)),
            readonly=bool(raw.get("readonly", False)),
            pressed=raw.get("pressed") if raw.get("pressed") is not None else None,
            value=str(raw["value"]) if raw.get("value") is not None else None,
        )

        properties: dict[str, Any] = {}
        if raw.get("level") is not None:
            properties["level"] = raw["level"]
        if raw.get("keyshortcuts"):
            properties["keyshortcuts"] = raw["keyshortcuts"]
        if raw.get("roledescription"):
            properties["roledescription"] = raw["roledescription"]
        if raw.get("valuetext"):
            properties["valuetext"] = raw["valuetext"]
        if raw.get("autocomplete"):
            properties["autocomplete"] = raw["autocomplete"]
        if raw.get("multiselectable") is not None:
            properties["multiselectable"] = raw["multiselectable"]
        if raw.get("orientation"):
            properties["orientation"] = raw["orientation"]
        if raw.get("multiline") is not None:
            properties["multiline"] = raw["multiline"]

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            description=description,
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties=properties,
            state=state,
            children=[],
            depth=depth,
        )
        converted[nid] = node

        for child_raw in raw.get("children", []):
            if isinstance(child_raw, dict):
                child = self._convert_node(child_raw, converted, depth + 1)
                child.parent_id = nid
                node.children.append(child)

        return node

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
