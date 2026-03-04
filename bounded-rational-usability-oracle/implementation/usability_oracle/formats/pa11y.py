"""
usability_oracle.formats.pa11y — pa11y JSON output format.

Parses the JSON array output from the ``pa11y`` accessibility testing tool
and converts it to the oracle's :class:`AccessibilityTree`.

Usage::

    # Run pa11y with JSON reporter:
    #   pa11y --reporter json https://example.com > results.json
    #
    # Output format (array of issues):
    # [
    #   {
    #     "code": "WCAG2AA.Principle1.Guideline1_1.1_1_1.H37",
    #     "type": "error",
    #     "typeCode": 1,
    #     "message": "Img element missing an alt attribute.",
    #     "context": "<img src=\\"logo.png\\">",
    #     "selector": "img.logo",
    #     "runner": "htmlcs"
    #   }
    # ]

    parser = Pa11yParser()
    tree = parser.parse(pa11y_json)
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

_PA11Y_ROLE_MAP = {
    "button": "button",
    "a": "link",
    "link": "link",
    "input": "textfield",
    "textbox": "textfield",
    "textarea": "textfield",
    "select": "combobox",
    "img": "image",
    "image": "image",
    "checkbox": "checkbox",
    "radio": "radio",
    "heading": "heading",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "nav": "navigation",
    "navigation": "navigation",
    "header": "banner",
    "banner": "banner",
    "main": "main",
    "footer": "contentinfo",
    "contentinfo": "contentinfo",
    "aside": "complementary",
    "complementary": "complementary",
    "form": "form",
    "search": "search",
    "region": "region",
    "dialog": "dialog",
    "alert": "alert",
    "table": "table",
    "list": "list",
    "listitem": "listitem",
    "tab": "tab",
    "menuitem": "menuitem",
    "menu": "menu",
    "toolbar": "toolbar",
    "separator": "separator",
    "tree": "tree",
    "treeitem": "treeitem",
    "group": "group",
    "slider": "slider",
    "combobox": "combobox",
    "row": "row",
    "cell": "cell",
    "option": "option",
    "div": "generic",
    "span": "generic",
    "p": "paragraph",
    "section": "region",
    "article": "article",
    "figure": "figure",
    "label": "generic",
    "fieldset": "group",
    "li": "listitem",
    "ul": "list",
    "ol": "list",
    "td": "cell",
    "th": "columnheader",
    "tr": "row",
}

_PA11Y_TYPE_SEVERITY = {
    "error": "critical",
    "warning": "moderate",
    "notice": "minor",
}

_CONTEXT_TAG_RE = re.compile(r"<(\w+)")


class Pa11yParser:
    """Parse pa11y JSON output.

    Run ``pa11y --reporter json <url>`` to obtain the input data.  The
    output is a JSON array of issue objects, each containing ``code``,
    ``type``, ``typeCode``, ``message``, ``context``, ``selector``, and
    ``runner``.
    """

    def __init__(self) -> None:
        self._role_map = dict(_PA11Y_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict | list) -> AccessibilityTree:
        """Parse pa11y JSON output.

        Parameters
        ----------
        data : str, dict, or list
            JSON string or parsed list of pa11y issue objects.  Also accepts
            a dict with an ``"issues"`` or ``"results"`` key containing the
            array.

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            issues = data.get("issues", data.get("results", []))
        elif isinstance(data, list):
            issues = data
        else:
            raise ValueError(f"Expected list or dict, got {type(data)}")

        converted: dict[str, AccessibilityNode] = {}
        children: list[AccessibilityNode] = []

        for issue in issues:
            if not isinstance(issue, dict):
                continue
            child = self._issue_to_node(issue, converted)
            children.append(child)

        root = AccessibilityNode(
            id="pa11y-root",
            role="document",
            name="pa11y audit results",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={"source": "pa11y"},
            state=AccessibilityState(),
            children=children,
            depth=0,
        )
        converted["pa11y-root"] = root

        for child in children:
            child.parent_id = "pa11y-root"

        tree = AccessibilityTree(root=root, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # Node builder
    # ------------------------------------------------------------------

    def _issue_to_node(
        self,
        issue: dict,
        converted: dict[str, AccessibilityNode],
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"pa11y-{self._counter}"

        selector = issue.get("selector", "")
        context = issue.get("context", "")
        role = self._role_from_context(context, selector)

        # Extract a human-readable name from the selector
        name = selector or context[:80] or "unknown"

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            description=issue.get("message", ""),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={
                "code": issue.get("code", ""),
                "type": issue.get("type", ""),
                "typeCode": issue.get("typeCode", 0),
                "context": context,
                "runner": issue.get("runner", ""),
                "severity": _PA11Y_TYPE_SEVERITY.get(
                    issue.get("type", ""), "minor"
                ),
            },
            state=AccessibilityState(),
            children=[],
            depth=1,
        )
        converted[nid] = node
        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _role_from_context(self, context: str, selector: str) -> str:
        """Extract role from HTML context snippet or CSS selector."""
        # Try HTML context first
        if context:
            tag_match = _CONTEXT_TAG_RE.search(context)
            if tag_match:
                tag = tag_match.group(1).lower()
                if tag in self._role_map:
                    return self._role_map[tag]

        # Fall back to selector
        if selector:
            tag = selector.split(".")[0].split("#")[0].split("[")[0].strip().lower()
            if tag in self._role_map:
                return self._role_map[tag]

        return "generic"

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
