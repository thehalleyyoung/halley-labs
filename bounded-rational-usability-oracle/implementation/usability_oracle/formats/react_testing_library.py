"""
usability_oracle.formats.react_testing_library — React Testing Library format.

Parses React Testing Library's ``prettyDOM()`` HTML output or
``@testing-library/dom`` ``logRoles()`` text output and converts either
to the oracle's :class:`AccessibilityTree`.

Usage::

    # prettyDOM — pass the HTML string:
    from testing_library import render, prettyDOM
    container = render(<App />).container
    html = prettyDOM(container)
    parser = ReactTestingLibraryParser()
    tree = parser.parse(html)

    # logRoles — capture the text output:
    #   heading:
    #
    #   Name "Welcome":
    #   <h1>Welcome</h1>
    #
    #   button:
    #
    #   Name "Submit":
    #   <button>Submit</button>
    tree = parser.parse(log_roles_text)
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

_RTL_ROLE_MAP = {
    "heading": "heading",
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
    "generic": "generic",
    "presentation": "generic",
    "none": "generic",
    "document": "document",
    # HTML tags (for prettyDOM parsing)
    "a": "link",
    "input": "textfield",
    "textarea": "textfield",
    "select": "combobox",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "nav": "navigation",
    "header": "banner",
    "footer": "contentinfo",
    "aside": "complementary",
    "section": "region",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "div": "generic",
    "span": "generic",
    "p": "paragraph",
    "td": "cell",
    "th": "columnheader",
    "tr": "row",
}

# Pattern for logRoles output blocks
_LOG_ROLES_BLOCK = re.compile(
    r"^(\w[\w-]*):\s*\n\s*\n"
    r"Name\s+\"([^\"]*)\"\s*:\s*\n"
    r"(.*?)(?=\n\w[\w-]*:\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Pattern to detect logRoles format
_LOG_ROLES_DETECT = re.compile(
    r"^\w[\w-]*:\s*\n\s*\nName\s+\"",
    re.MULTILINE,
)

# Pattern to extract tag + attributes from an HTML snippet
_HTML_TAG_RE = re.compile(r"<(\w+)([^>]*)>")
_ATTR_RE = re.compile(r'([\w-]+)(?:=["\']([^"\']*)["\'])?')


class ReactTestingLibraryParser:
    """Parse React Testing Library output (prettyDOM or logRoles).

    Automatically detects whether the input is ``logRoles()`` text output
    or ``prettyDOM()`` HTML and parses accordingly.
    """

    def __init__(self) -> None:
        self._role_map = dict(_RTL_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse React Testing Library output.

        Parameters
        ----------
        data : str or dict
            - ``logRoles()`` text (role blocks separated by blank lines)
            - ``prettyDOM()`` HTML string
            - dict with ``"html"`` or ``"logRoles"`` key

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, dict):
            if "logRoles" in data:
                return self._parse_log_roles(data["logRoles"])
            if "html" in data:
                return self._parse_pretty_dom(data["html"])
            raise ValueError("Dict must contain 'html' or 'logRoles' key")

        text = data.strip()

        # Detect which format we have
        if _LOG_ROLES_DETECT.search(text):
            return self._parse_log_roles(text)
        return self._parse_pretty_dom(text)

    # ------------------------------------------------------------------
    # logRoles parser
    # ------------------------------------------------------------------

    def _parse_log_roles(self, text: str) -> AccessibilityTree:
        converted: dict[str, AccessibilityNode] = {}
        children: list[AccessibilityNode] = []

        for match in _LOG_ROLES_BLOCK.finditer(text):
            role_str = match.group(1).strip()
            name = match.group(2).strip()
            html_snippet = match.group(3).strip()

            self._counter += 1
            nid = f"rtl-{self._counter}"
            role = self._role_map.get(role_str, role_str.lower())

            properties: dict[str, Any] = {}
            if html_snippet:
                properties["html"] = html_snippet
                tag_match = _HTML_TAG_RE.search(html_snippet)
                if tag_match:
                    tag = tag_match.group(1).lower()
                    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        properties["level"] = int(tag[1])

            node = AccessibilityNode(
                id=nid,
                role=role,
                name=name,
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                properties=properties,
                state=AccessibilityState(),
                children=[],
                depth=1,
            )
            converted[nid] = node
            children.append(node)

        root = AccessibilityNode(
            id="rtl-root",
            role="document",
            name="logRoles output",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={},
            state=AccessibilityState(),
            children=children,
            depth=0,
        )
        converted["rtl-root"] = root

        for child in children:
            child.parent_id = "rtl-root"

        tree = AccessibilityTree(root=root, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # prettyDOM parser (simplified HTML)
    # ------------------------------------------------------------------

    def _parse_pretty_dom(self, html: str) -> AccessibilityTree:
        converted: dict[str, AccessibilityNode] = {}
        root = self._parse_html_recursive(html, converted, depth=0)

        tree = AccessibilityTree(root=root, node_index=converted)
        tree.build_index()
        return tree

    def _parse_html_recursive(
        self,
        html: str,
        converted: dict[str, AccessibilityNode],
        depth: int,
    ) -> AccessibilityNode:
        """Lightweight recursive HTML tag parser for prettyDOM output."""
        html = html.strip()

        tag_match = _HTML_TAG_RE.search(html)
        if not tag_match:
            # Plain text node
            self._counter += 1
            nid = f"rtl-{self._counter}"
            text_content = re.sub(r"<[^>]+>", "", html).strip()
            node = AccessibilityNode(
                id=nid,
                role="generic",
                name=text_content[:200],
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                properties={},
                state=AccessibilityState(),
                children=[],
                depth=depth,
            )
            converted[nid] = node
            return node

        tag = tag_match.group(1).lower()
        attr_str = tag_match.group(2)
        attrs = dict(_ATTR_RE.findall(attr_str))

        self._counter += 1
        nid = attrs.get("id", f"rtl-{self._counter}")
        if nid in converted:
            nid = f"{nid}-{self._counter}"

        explicit_role = attrs.get("role", "")
        role_key = explicit_role if explicit_role else tag
        role = self._role_map.get(role_key, role_key.lower())

        name = (
            attrs.get("aria-label", "")
            or attrs.get("alt", "")
            or attrs.get("title", "")
            or ""
        )

        properties: dict[str, Any] = {}
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            properties["level"] = int(tag[1])

        state = AccessibilityState(
            disabled="disabled" in attrs,
            hidden="hidden" in attrs or attrs.get("aria-hidden") == "true",
            required="required" in attrs,
            readonly="readonly" in attrs,
        )

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties=properties,
            state=state,
            children=[],
            depth=depth,
        )
        converted[nid] = node

        # Extract child tags (simplified: find top-level child tags in the body)
        open_tag = f"<{tag_match.group(1)}"
        close_tag = f"</{tag_match.group(1)}>"
        body_start = html.find(">", tag_match.start()) + 1
        body_end = html.rfind(close_tag)
        if body_start > 0 and body_end > body_start:
            inner = html[body_start:body_end].strip()
            if inner:
                child_tags = self._split_top_level_tags(inner)
                for child_html in child_tags:
                    child = self._parse_html_recursive(child_html, converted, depth + 1)
                    child.parent_id = nid
                    node.children.append(child)

        return node

    @staticmethod
    def _split_top_level_tags(html: str) -> list[str]:
        """Split HTML into top-level element strings."""
        results: list[str] = []
        i = 0
        length = len(html)
        while i < length:
            if html[i] == "<" and i + 1 < length and html[i + 1] != "/":
                tag_match = re.match(r"<(\w+)", html[i:])
                if not tag_match:
                    i += 1
                    continue
                tag_name = tag_match.group(1)
                # Find matching close tag (handle nesting)
                depth = 1
                j = html.find(">", i) + 1
                while j < length and depth > 0:
                    next_open = html.find(f"<{tag_name}", j)
                    next_close = html.find(f"</{tag_name}>", j)
                    if next_close == -1:
                        break
                    if next_open != -1 and next_open < next_close:
                        depth += 1
                        j = next_open + len(tag_name) + 1
                    else:
                        depth -= 1
                        if depth == 0:
                            end = next_close + len(f"</{tag_name}>")
                            results.append(html[i:end])
                            i = end
                            break
                        j = next_close + len(f"</{tag_name}>")
                else:
                    # Self-closing or unmatched — take to next <
                    next_tag = html.find("<", i + 1)
                    if next_tag == -1:
                        results.append(html[i:])
                        break
                    results.append(html[i:next_tag])
                    i = next_tag
                    continue
            else:
                i += 1
        return results

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
