"""
usability_oracle.formats.storybook — Storybook accessibility addon format.

Parses JSON output from ``@storybook/addon-a11y`` which wraps axe-core
results with Storybook component metadata, and converts it to the oracle's
:class:`AccessibilityTree`.

Usage::

    // In Storybook, the a11y addon automatically runs axe-core per story.
    // Export the results JSON:
    {
        "storyId": "button--primary",
        "kind": "Button",
        "name": "Primary",
        "axeResults": { "violations": [...], "passes": [...] }
    }

    parser = StorybookParser()
    tree = parser.parse(storybook_json)
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

_STORYBOOK_ROLE_MAP = {
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
    "spinbutton": "spinbutton",
    "switch": "switch",
    "progressbar": "progressbar",
    "status": "status",
    "tooltip": "tooltip",
}


class StorybookParser:
    """Parse Storybook ``@storybook/addon-a11y`` JSON output.

    Expects JSON with ``storyId``, ``kind``, ``name``, and ``axeResults``
    containing axe-core ``violations`` and ``passes`` arrays.  Each entry is
    converted to an :class:`AccessibilityNode` within a tree rooted at the
    story component.
    """

    def __init__(self) -> None:
        self._role_map = dict(_STORYBOOK_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse Storybook a11y addon JSON.

        Parameters
        ----------
        data : str or dict
            JSON string or dict with ``storyId``, ``kind``, ``name``,
            ``axeResults``.

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        story_id = data.get("storyId", "unknown")
        kind = data.get("kind", "")
        name = data.get("name", "")
        axe_results = data.get("axeResults", {})

        violations = axe_results.get("violations", [])
        passes = axe_results.get("passes", [])

        metadata: dict[str, Any] = {
            "storyId": story_id,
            "kind": kind,
            "name": name,
            "source": "storybook-addon-a11y",
        }

        children: list[AccessibilityNode] = []
        converted: dict[str, AccessibilityNode] = {}

        for violation in violations:
            for node_info in violation.get("nodes", []):
                child = self._axe_node_to_a11y(violation, node_info, converted, "violation")
                children.append(child)

        for pass_rule in passes:
            for node_info in pass_rule.get("nodes", []):
                child = self._axe_node_to_a11y(pass_rule, node_info, converted, "pass")
                children.append(child)

        root = AccessibilityNode(
            id=f"sb-{story_id}",
            role="region",
            name=f"{kind} / {name}" if kind else name,
            description=f"Storybook story: {story_id}",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties=metadata,
            state=AccessibilityState(),
            children=children,
            depth=0,
        )
        converted[f"sb-{story_id}"] = root

        for child in children:
            child.parent_id = f"sb-{story_id}"

        tree = AccessibilityTree(root=root, metadata=metadata, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # Node builder
    # ------------------------------------------------------------------

    def _axe_node_to_a11y(
        self,
        rule: dict,
        node_info: dict,
        converted: dict[str, AccessibilityNode],
        result_type: str,
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"sb-n-{self._counter}"

        target = node_info.get("target", ["unknown"])
        selector = target[0] if target else "unknown"
        role = self._role_from_selector(selector)

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=selector,
            description=rule.get("help", ""),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={
                "axe_rule": rule.get("id", ""),
                "impact": rule.get("impact", ""),
                "tags": rule.get("tags", []),
                "html": node_info.get("html", ""),
                "result_type": result_type,
                "failure_summary": node_info.get("failureSummary", ""),
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

    def _role_from_selector(self, selector: str) -> str:
        tag = selector.split(".")[0].split("#")[0].split("[")[0].strip().lower()
        return self._role_map.get(tag, "generic")

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
