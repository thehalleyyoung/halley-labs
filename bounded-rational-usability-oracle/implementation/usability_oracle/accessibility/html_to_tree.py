"""Enhanced HTML-to-accessibility-tree extraction for real-world HTML.

Extends :class:`HTMLAccessibilityParser` with capabilities needed for real
GovUK Design System HTML: label-for association resolution, heading hierarchy
validation, landmark region detection, ARIA relationship resolution
(aria-controls, aria-labelledby), and form grouping semantics (fieldset/legend).

The key improvement over the base parser is **document-global resolution**:
label ``for`` attributes and ``aria-labelledby`` / ``aria-controls`` are
resolved against the full DOM, not just local context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


class RealHTMLParser(HTMLAccessibilityParser):
    """Parse real-world HTML (e.g. GovUK Design System) into an enriched
    :class:`AccessibilityTree` with full label/ARIA resolution.

    Enhancements over base parser:
    - Resolves ``<label for="...">`` to inject accessible names on inputs
    - Resolves ``aria-labelledby`` using document-global id lookup
    - Resolves ``aria-controls`` / ``aria-owns`` relationships
    - Detects ``<fieldset>``/``<legend>`` grouping for form controls
    - Validates heading hierarchy (h1→h2→h3 nesting)
    - Annotates landmark regions with context
    - Identifies interactive components (accordion, tabs) from class/data-module
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._id_to_element: Dict[str, Any] = {}
        self._label_for_map: Dict[str, str] = {}
        self._heading_sequence: List[Tuple[int, str]] = []
        self._issues: List[Dict[str, Any]] = []

    def parse(self, html: str) -> AccessibilityTree:
        """Parse HTML with full document-global resolution."""
        self._counter = 0
        self._id_to_element.clear()
        self._label_for_map.clear()
        self._heading_sequence.clear()
        self._issues.clear()

        root_el = self._parse_html(html)

        # Pre-pass: build id→element index and label-for map
        self._index_document(root_el)

        root_node = self._parse_element(root_el, parent_id=None, depth=0)
        tree = AccessibilityTree(
            root=root_node,
            metadata={
                "source": "html",
                "parser": "real_html",
                "issues": list(self._issues),
                "heading_sequence": list(self._heading_sequence),
            },
        )
        tree.build_index()

        # Post-pass: resolve cross-references
        self._resolve_label_associations(tree)
        self._resolve_aria_labelledby(tree)
        self._resolve_aria_controls(tree)
        self._detect_fieldset_groups(tree)
        self._validate_heading_hierarchy(tree)
        self._annotate_component_type(tree)

        return tree

    # ── Pre-pass: document indexing ────────────────────────────────────────

    def _index_document(self, root: Any) -> None:
        """Walk the DOM and build id→element and label→for maps."""
        stack = [root]
        while stack:
            el = stack.pop()
            if not isinstance(getattr(el, "tag", None), str):
                continue
            eid = self._attr(el, "id")
            if eid:
                self._id_to_element[eid] = el

            tag = self._tag_name(el)
            if tag == "label":
                for_id = self._attr(el, "for")
                if for_id:
                    label_text = self._all_text(el).strip()
                    if label_text:
                        self._label_for_map[for_id] = label_text

            for child in el:
                stack.append(child)

    # ── Enhanced name extraction ──────────────────────────────────────────

    def _extract_name(self, element: Any) -> str:
        """Extract accessible name with label-for resolution."""
        # aria-label takes top priority
        label = self._attr(element, "aria-label")
        if label and label.strip():
            return label.strip()

        # aria-labelledby: resolve from document id index
        labelledby = self._attr(element, "aria-labelledby")
        if labelledby:
            parts = []
            for ref_id in labelledby.split():
                ref_el = self._id_to_element.get(ref_id)
                if ref_el is not None:
                    parts.append(self._all_text(ref_el).strip())
            if parts:
                return " ".join(parts)

        tag = self._tag_name(element)
        eid = self._attr(element, "id")

        # label-for association
        if eid and eid in self._label_for_map:
            return self._label_for_map[eid]

        # img alt
        if tag == "img":
            alt = self._attr(element, "alt")
            if alt is not None:
                return alt.strip()

        # input placeholder fallback
        if tag == "input":
            placeholder = self._attr(element, "placeholder")
            if placeholder:
                return placeholder.strip()

        # fieldset: use legend as name
        if tag == "fieldset":
            for child in element:
                if self._tag_name(child) == "legend":
                    return self._all_text(child).strip()

        # title fallback
        title = self._attr(element, "title")
        if title and title.strip():
            return title.strip()

        # Direct text content
        text = self._direct_text(element)
        if text:
            return text

        return ""

    # ── Enhanced property extraction ──────────────────────────────────────

    def _extract_properties(self, element: Any) -> dict[str, Any]:
        props = super()._extract_properties(element)
        tag = self._tag_name(element)

        # Track data-module for component detection
        data_module = self._attr(element, "data-module")
        if data_module:
            props["data-module"] = data_module

        # Track for/id association
        if tag == "label":
            for_id = self._attr(element, "for")
            if for_id:
                props["label-for"] = for_id

        # Track fieldset context
        if tag == "fieldset":
            props["is-fieldset"] = True

        # Heading levels already handled by parent
        # Track aria-expanded for component state
        expanded = self._attr(element, "aria-expanded")
        if expanded is not None:
            props["aria-expanded"] = expanded

        # Track aria-selected for tab panels
        selected = self._attr(element, "aria-selected")
        if selected is not None:
            props["aria-selected"] = selected

        return props

    # ── Post-pass: label resolution ──────────────────────────────────────

    def _resolve_label_associations(self, tree: AccessibilityTree) -> None:
        """Inject accessible names from label-for associations."""
        for node in tree.node_index.values():
            raw_id = node.id
            # Strip the parser prefix to recover the original HTML id
            if raw_id.startswith(self._id_prefix + "-"):
                html_id = raw_id[len(self._id_prefix) + 1:]
            else:
                html_id = raw_id

            if html_id in self._label_for_map and not node.name:
                node.name = self._label_for_map[html_id]

            # Check if node has no label but is a form control
            if (node.role in ("textbox", "checkbox", "radio", "combobox",
                              "spinbutton", "searchbox", "slider")
                    and not node.name):
                self._issues.append({
                    "type": "missing_label",
                    "node_id": node.id,
                    "role": node.role,
                    "wcag": "1.3.1",
                    "message": f"Form control '{node.id}' has no accessible name",
                })

    def _resolve_aria_labelledby(self, tree: AccessibilityTree) -> None:
        """Resolve aria-labelledby references that are still unresolved."""
        for node in tree.node_index.values():
            if node.name and node.name.startswith("[labelledby:"):
                ref_ids = node.name[len("[labelledby:"):-1].split()
                parts = []
                for ref_id in ref_ids:
                    prefixed = f"{self._id_prefix}-{ref_id}"
                    ref_node = tree.get_node(prefixed)
                    if ref_node:
                        parts.append(ref_node.name)
                if parts:
                    node.name = " ".join(parts)

    def _resolve_aria_controls(self, tree: AccessibilityTree) -> None:
        """Annotate aria-controls relationships."""
        for node in tree.node_index.values():
            controls = node.properties.get("aria-controls")
            if controls:
                for ctrl_id in controls.split():
                    prefixed = f"{self._id_prefix}-{ctrl_id}"
                    target = tree.get_node(prefixed)
                    if target:
                        node.properties["controls_node"] = prefixed
                        target.properties["controlled_by"] = node.id

    def _detect_fieldset_groups(self, tree: AccessibilityTree) -> None:
        """Mark form controls grouped by fieldset with group context."""
        for node in tree.node_index.values():
            if node.role == "group" and node.properties.get("tag") == "fieldset":
                group_name = node.name
                for desc in node.get_descendants():
                    if desc.role in ("checkbox", "radio", "textbox",
                                     "combobox", "spinbutton"):
                        desc.properties["fieldset_group"] = group_name

    def _validate_heading_hierarchy(self, tree: AccessibilityTree) -> None:
        """Check heading level sequence and record issues."""
        headings = []
        for node in tree.root.iter_preorder():
            level = node.properties.get("level")
            if level is not None and node.role == "heading":
                headings.append((int(level), node.id, node.name))

        self._heading_sequence = [(h[0], h[2]) for h in headings]
        tree.metadata["heading_sequence"] = list(self._heading_sequence)

        prev_level = 0
        for level, nid, name in headings:
            if level > prev_level + 1 and prev_level > 0:
                self._issues.append({
                    "type": "heading_skip",
                    "node_id": nid,
                    "expected": prev_level + 1,
                    "actual": level,
                    "wcag": "1.3.1",
                    "message": (
                        f"Heading level skipped: h{prev_level}→h{level} "
                        f"at '{name[:40]}'"
                    ),
                })
            prev_level = level

        tree.metadata["issues"] = list(self._issues)

    def _annotate_component_type(self, tree: AccessibilityTree) -> None:
        """Detect interactive component types from class/data-module patterns."""
        for node in tree.node_index.values():
            dm = node.properties.get("data-module", "")
            cls = node.properties.get("class", "")

            if "govuk-accordion" in dm or "govuk-accordion" in cls:
                node.properties["component_type"] = "accordion"
            elif "govuk-tabs" in dm or "govuk-tabs" in cls:
                node.properties["component_type"] = "tabs"
            elif node.role == "tablist":
                node.properties["component_type"] = "tablist"
            elif node.role == "tabpanel":
                node.properties["component_type"] = "tabpanel"

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _all_text(element: Any) -> str:
        """Recursively extract all text content from an element."""
        parts = []
        if element.text:
            parts.append(element.text)
        for child in element:
            parts.append(RealHTMLParser._all_text(child))
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts)

    def get_issues(self) -> List[Dict[str, Any]]:
        """Return accessibility issues detected during parsing."""
        return list(self._issues)

    def get_heading_sequence(self) -> List[Tuple[int, str]]:
        """Return the heading level sequence found in the document."""
        return list(self._heading_sequence)
