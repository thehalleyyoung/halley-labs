"""
usability_oracle.formats.cypress — Cypress accessibility audit format.

Parses Cypress accessibility audit results produced by cypress-axe or
cypress-audit, which wraps axe-core results in a Cypress test structure,
and converts them to the oracle's :class:`AccessibilityTree`.

Usage::

    // In cypress test:
    cy.injectAxe()
    cy.checkA11y(null, null, (violations) => {
        cy.task('log', JSON.stringify({
            testTitle: Cypress.currentTest.title,
            url: cy.url(),
            results: { violations, passes: [] }
        }))
    })

    parser = CypressParser()
    tree = parser.parse(cypress_json)
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

# axe-core HTML target selectors → oracle roles (best-effort mapping)
_CYPRESS_ROLE_MAP = {
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
}


class CypressParser:
    """Parse Cypress accessibility audit results (cypress-axe / cypress-audit).

    Expects JSON with ``testTitle``, ``url``, and ``results`` containing
    axe-core ``violations`` and ``passes`` arrays.  Each violation contains
    ``nodes`` with CSS selectors that are converted to tree nodes.
    """

    def __init__(self) -> None:
        self._role_map = dict(_CYPRESS_ROLE_MAP)
        self._counter = 0

    def parse(self, data: str | dict) -> AccessibilityTree:
        """Parse Cypress accessibility audit JSON.

        Parameters
        ----------
        data : str or dict
            JSON string or dict with ``testTitle``, ``url``, ``results``.

        Returns
        -------
        AccessibilityTree
        """
        self._counter = 0

        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        results = data.get("results", data)
        violations = results.get("violations", [])
        passes = results.get("passes", [])

        metadata: dict[str, Any] = {
            "testTitle": data.get("testTitle", ""),
            "url": data.get("url", ""),
            "source": "cypress-axe",
        }

        children: list[AccessibilityNode] = []
        converted: dict[str, AccessibilityNode] = {}

        for violation in violations:
            for node_info in violation.get("nodes", []):
                child = self._violation_node_to_a11y(violation, node_info, converted)
                children.append(child)

        for pass_rule in passes:
            for node_info in pass_rule.get("nodes", []):
                child = self._pass_node_to_a11y(pass_rule, node_info, converted)
                children.append(child)

        root = AccessibilityNode(
            id="cypress-root",
            role="document",
            name=data.get("testTitle", "Cypress Audit"),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties=metadata,
            state=AccessibilityState(),
            children=children,
            depth=0,
        )
        converted["cypress-root"] = root

        for child in children:
            child.parent_id = "cypress-root"

        tree = AccessibilityTree(root=root, metadata=metadata, node_index=converted)
        tree.build_index()
        return tree

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def _violation_node_to_a11y(
        self,
        violation: dict,
        node_info: dict,
        converted: dict[str, AccessibilityNode],
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"cy-v-{self._counter}"

        target = node_info.get("target", ["unknown"])
        selector = target[0] if target else "unknown"
        role = self._role_from_selector(selector)

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=selector,
            description=violation.get("help", ""),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={
                "axe_rule": violation.get("id", ""),
                "impact": violation.get("impact", ""),
                "failure_summary": node_info.get("failureSummary", ""),
                "tags": violation.get("tags", []),
                "html": node_info.get("html", ""),
                "result_type": "violation",
            },
            state=AccessibilityState(),
            children=[],
            depth=1,
        )
        converted[nid] = node
        return node

    def _pass_node_to_a11y(
        self,
        pass_rule: dict,
        node_info: dict,
        converted: dict[str, AccessibilityNode],
    ) -> AccessibilityNode:
        self._counter += 1
        nid = f"cy-p-{self._counter}"

        target = node_info.get("target", ["unknown"])
        selector = target[0] if target else "unknown"
        role = self._role_from_selector(selector)

        node = AccessibilityNode(
            id=nid,
            role=role,
            name=selector,
            description=pass_rule.get("help", ""),
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={
                "axe_rule": pass_rule.get("id", ""),
                "tags": pass_rule.get("tags", []),
                "html": node_info.get("html", ""),
                "result_type": "pass",
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
        """Best-effort role extraction from a CSS selector."""
        tag = selector.split(".")[0].split("#")[0].split("[")[0].strip().lower()
        return self._role_map.get(tag, "generic")

    def _make_empty_root(self) -> AccessibilityNode:
        return AccessibilityNode(
            id="root", role="document", name="",
            bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
            properties={}, state=AccessibilityState(), children=[], depth=0,
        )
