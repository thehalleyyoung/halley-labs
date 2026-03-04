"""
usability_oracle.formats.testing_library_queries — Testing Library queries format.

Parses the structured JSON from Testing Library's ``screen.getByRole``,
``screen.getAllByRole``, and similar query results when serialized, and
converts them to the oracle's :class:`AccessibilityTree`.

Usage::

    // Serialise Testing Library query results:
    const elements = screen.getAllByRole('button');
    const serialized = elements.map(el => ({
        role: el.getAttribute('role') || el.tagName.toLowerCase(),
        name: el.getAttribute('aria-label') || el.textContent,
        hidden: el.getAttribute('aria-hidden') === 'true',
        selected: el.getAttribute('aria-selected') === 'true',
        checked: el.getAttribute('aria-checked') === 'true',
        pressed: el.getAttribute('aria-pressed'),
        expanded: el.getAttribute('aria-expanded'),
        level: el.getAttribute('aria-level'),
        disabled: el.hasAttribute('disabled'),
    }));

    parser = TestingLibraryQueriesParser()
    tree = parser.parse(serialized)
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

_TLQ_ROLE_MAP = {
    "button": "button",
    "link": "link",
    "textbox": "textfield",
    "checkbox": "checkbox",
    "radio": "radio",
    "combobox": "combobox",
    "slider": "slider",
    "tab": "tab",
    "menuitem": "menuitem",
    "listitem": "listitem",
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
    "tree": "tree",
    "treeitem": "treeitem",
    "group": "group",
    "toolbar": "toolbar",
    "separator": "separator",
    "menu": "menu",
    "option": "option",
    "spinbutton": "spinbutton",
    "switch": "switch",
    "progressbar": "progressbar",
    "status": "status",
    "log": "log",
    "tooltip": "tooltip",
    "gridcell": "gridcell",
    "columnheader": "columnheader",
    "rowheader": "rowheader",
    "article": "article",
    "figure": "figure",
    "math": "math",
    "note": "note",
    "tabpanel": "tabpanel",
    "tablist": "tablist",
    "menubar": "menubar",
    "menuitemcheckbox": "menuitemcheckbox",
    "menuitemradio": "menuitemradio",
    "generic": "generic",
    "presentation": "generic",
    "none": "generic",
    "document": "document",
    "alertdialog": "alertdialog",
    "definition": "definition",
    "term": "term",
    "feed": "feed",
    "grid": "grid",
    "listbox": "listbox",
    "scrollbar": "scrollbar",
}


class TestingLibraryQueriesParser:
    """Parse Testing Library structured query results.

    Accepts a JSON array of elements returned by ``screen.getAllByRole()``
    (or similar) that have been serialized with role, name, and state
    properties.
    """

    def __init__(self) -> None:
        self._role_map = dict(_TLQ_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict | list) -> AccessibilityTree:
        """Parse Testing Library query results.

        Parameters
        ----------
        data : str, dict, or list
            JSON string, a list of element dicts, or a single element dict.

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            # Single element — wrap in list
            elements = [data]
        elif isinstance(data, list):
            elements = data
        else:
            raise ValueError(f"Expected list or dict, got {type(data)}")

        converted: dict[str, AccessibilityNode] = {}
        children: list[AccessibilityNode] = []

        for elem in elements:
            if not isinstance(elem, dict):
                continue
            child = self._element_to_node(elem, converted)
            children.append(child)

        root = AccessibilityNode(
            id="tlq-root",
            role="document",
            name="Testing Library query results",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={"source": "testing-library"},
            state=AccessibilityState(),
            children=children,
            depth=0,
        )
        converted["tlq-root"] = root

        for child in children:
            child.parent_id = "tlq-root"

        tree = AccessibilityTree(root=root, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # Node builder
    # ------------------------------------------------------------------

    def _element_to_node(
        self,
        elem: dict,
        converted: dict[str, AccessibilityNode],
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"tlq-{self._counter}"

        raw_role = elem.get("role", "generic")
        role = self._role_map.get(raw_role, raw_role.lower())
        name = elem.get("name", "")

        state = AccessibilityState(
            focused=bool(elem.get("focused", False)),
            selected=bool(elem.get("selected", False)),
            expanded=elem.get("expanded") if elem.get("expanded") is not None else False,
            checked=self._to_optional_bool(elem.get("checked")),
            disabled=bool(elem.get("disabled", False)),
            hidden=bool(elem.get("hidden", False)),
            required=bool(elem.get("required", False)),
            readonly=bool(elem.get("readonly", False)),
            pressed=self._to_optional_bool(elem.get("pressed")),
            value=str(elem["value"]) if elem.get("value") is not None else None,
        )

        properties: dict[str, Any] = {}
        if elem.get("level") is not None:
            properties["level"] = elem["level"]
        if elem.get("description"):
            properties["description"] = elem["description"]

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            description=elem.get("description", ""),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties=properties,
            state=state,
            children=[],
            depth=1,
        )
        converted[nid] = node
        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_optional_bool(val: Any) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        s = str(val).lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return None

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
