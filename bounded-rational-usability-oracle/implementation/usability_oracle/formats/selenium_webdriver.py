"""
usability_oracle.formats.selenium_webdriver — Selenium WebDriver DOM extraction format.

Parses the JSON output of a Selenium WebDriver ``execute_script`` call that
extracts DOM accessibility information, converting it to the oracle's
:class:`AccessibilityTree`.

Usage::

    # In your Selenium test, run a script that walks the DOM:
    data = driver.execute_script('''
        function walk(el) {
            var rect = el.getBoundingClientRect();
            return {
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role') || el.tagName.toLowerCase(),
                "aria-label": el.getAttribute('aria-label') || '',
                attributes: Object.fromEntries(
                    Array.from(el.attributes).map(a => [a.name, a.value])
                ),
                rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
                children: Array.from(el.children).map(walk)
            };
        }
        return walk(document.body);
    ''')
    parser = SeleniumParser()
    tree = parser.parse(data)
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

# HTML tag / ARIA role → normalised oracle role
_SELENIUM_ROLE_MAP = {
    "a": "link",
    "button": "button",
    "input": "textfield",
    "textarea": "textfield",
    "select": "combobox",
    "option": "option",
    "img": "image",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "nav": "navigation",
    "header": "banner",
    "main": "main",
    "footer": "contentinfo",
    "aside": "complementary",
    "form": "form",
    "table": "table",
    "thead": "rowgroup",
    "tbody": "rowgroup",
    "tr": "row",
    "td": "cell",
    "th": "columnheader",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "dialog": "dialog",
    "details": "group",
    "summary": "button",
    "section": "region",
    "article": "article",
    "figure": "figure",
    "figcaption": "generic",
    "label": "generic",
    "fieldset": "group",
    "legend": "generic",
    "div": "generic",
    "span": "generic",
    "p": "paragraph",
    # Explicit ARIA roles pass through
    "link": "link",
    "checkbox": "checkbox",
    "radio": "radio",
    "slider": "slider",
    "combobox": "combobox",
    "textfield": "textfield",
    "textbox": "textfield",
    "tab": "tab",
    "tablist": "tablist",
    "tabpanel": "tabpanel",
    "menuitem": "menuitem",
    "menu": "menu",
    "menubar": "menubar",
    "toolbar": "toolbar",
    "tree": "tree",
    "treeitem": "treeitem",
    "listbox": "listbox",
    "grid": "grid",
    "gridcell": "gridcell",
    "alert": "alert",
    "alertdialog": "alertdialog",
    "status": "status",
    "progressbar": "progressbar",
    "switch": "switch",
    "separator": "separator",
    "search": "search",
    "region": "region",
    "navigation": "navigation",
    "banner": "banner",
    "contentinfo": "contentinfo",
    "complementary": "complementary",
    "heading": "heading",
    "image": "image",
    "group": "group",
    "document": "document",
    "application": "application",
    "log": "log",
    "tooltip": "tooltip",
    "spinbutton": "spinbutton",
    "scrollbar": "scrollbar",
    "note": "note",
    "math": "math",
    "definition": "definition",
    "term": "term",
    "feed": "feed",
}


class SeleniumParser:
    """Parse Selenium WebDriver DOM extraction JSON.

    Expects the output of a ``driver.execute_script()`` call that traverses the
    DOM and returns a nested dict with ``tag``, ``role``, ``aria-label``,
    ``attributes``, ``rect``, and ``children`` for each element.
    """

    def __init__(self) -> None:
        self._role_map = dict(_SELENIUM_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse a Selenium DOM extraction result.

        Parameters
        ----------
        data : str or dict
            JSON string or dict produced by ``driver.execute_script()``.

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
        attrs = raw.get("attributes", {}) or {}
        tag = raw.get("tag", "div").lower()
        explicit_role = raw.get("role", "") or attrs.get("role", "")

        role_key = explicit_role if explicit_role else tag
        role = self._role_map.get(role_key, role_key.lower())

        nid = attrs.get("id", "") or f"sel-{self._counter}"
        # Ensure uniqueness
        if nid in converted:
            nid = f"{nid}-{self._counter}"

        name = (
            raw.get("aria-label", "")
            or attrs.get("aria-label", "")
            or attrs.get("aria-labelledby", "")
            or attrs.get("title", "")
            or attrs.get("alt", "")
            or ""
        )

        description = attrs.get("aria-describedby", "") or attrs.get("aria-description", "") or ""

        bbox = self._extract_bbox(raw)
        state = self._extract_state(raw, attrs, tag)

        properties: dict[str, Any] = {}
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            properties["level"] = int(tag[1])
        if attrs.get("aria-level"):
            try:
                properties["level"] = int(attrs["aria-level"])
            except (ValueError, TypeError):
                pass
        if attrs.get("placeholder"):
            properties["placeholder"] = attrs["placeholder"]
        if attrs.get("type"):
            properties["input_type"] = attrs["type"]

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            description=description,
            bounding_box=bbox,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_bbox(self, raw: dict) -> BoundingBox:
        rect = raw.get("rect", {})
        if isinstance(rect, dict):
            return BoundingBox(
                x=float(rect.get("x", 0)),
                y=float(rect.get("y", 0)),
                width=float(rect.get("width", 0)),
                height=float(rect.get("height", 0)),
            )
        return BoundingBox(x=0, y=0, width=0, height=0)

    def _extract_state(self, raw: dict, attrs: dict, tag: str) -> AccessibilityState:
        return AccessibilityState(
            focused=False,
            selected=self._aria_bool(attrs.get("aria-selected")),
            expanded=self._aria_bool(attrs.get("aria-expanded")),
            checked=self._aria_tristate(attrs.get("aria-checked"))
                    or self._aria_tristate(attrs.get("checked")),
            disabled=self._aria_bool(attrs.get("aria-disabled"))
                     or self._aria_bool(attrs.get("disabled")),
            hidden=self._aria_bool(attrs.get("aria-hidden"))
                   or self._aria_bool(attrs.get("hidden")),
            required=self._aria_bool(attrs.get("aria-required"))
                     or self._aria_bool(attrs.get("required")),
            readonly=self._aria_bool(attrs.get("aria-readonly"))
                     or self._aria_bool(attrs.get("readonly")),
            pressed=self._aria_tristate(attrs.get("aria-pressed")),
            value=attrs.get("aria-valuenow") or attrs.get("value"),
        )

    @staticmethod
    def _aria_bool(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "")

    @staticmethod
    def _aria_tristate(val: Any) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        s = str(val).lower()
        if s == "true":
            return True
        if s == "false":
            return False
        if s == "mixed":
            return None
        return None

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
