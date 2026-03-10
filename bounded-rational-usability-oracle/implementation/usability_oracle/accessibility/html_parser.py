"""Parse HTML documents into AccessibilityTree structures.

Uses lxml (with html5lib fallback) to parse HTML and maps elements to
ARIA roles, extracts accessible names, bounding boxes from data attributes,
and constructs a normalised AccessibilityTree.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Optional

try:
    from lxml import etree
    from lxml.html import fromstring as html_fromstring, tostring as html_tostring

    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

try:
    import html5lib

    _HAS_HTML5LIB = True
except ImportError:
    _HAS_HTML5LIB = False

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.roles import RoleTaxonomy

# ── Constants ─────────────────────────────────────────────────────────────────

_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "head", "meta", "link",
    "base", "title",
})

_IMPLICIT_ROLE_MAP: dict[str, str] = {
    "a": "link",
    "article": "article",
    "aside": "complementary",
    "body": "document",
    "button": "button",
    "datalist": "listbox",
    "dd": "definition",
    "details": "group",
    "dialog": "dialog",
    "dt": "term",
    "fieldset": "group",
    "figure": "figure",
    "footer": "contentinfo",
    "form": "form",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "header": "banner",
    "hr": "separator",
    "html": "document",
    "img": "img",
    "li": "listitem",
    "main": "main",
    "math": "math",
    "menu": "list",
    "nav": "navigation",
    "ol": "list",
    "optgroup": "group",
    "option": "option",
    "output": "status",
    "p": "paragraph",
    "progress": "progressbar",
    "section": "region",
    "select": "combobox",
    "summary": "button",
    "table": "table",
    "tbody": "rowgroup",
    "td": "cell",
    "textarea": "textbox",
    "tfoot": "rowgroup",
    "th": "columnheader",
    "thead": "rowgroup",
    "tr": "row",
    "ul": "list",
}

_INPUT_TYPE_ROLE_MAP: dict[str, str] = {
    "button": "button",
    "checkbox": "checkbox",
    "color": "textbox",
    "date": "textbox",
    "datetime-local": "textbox",
    "email": "textbox",
    "file": "button",
    "hidden": "none",
    "image": "button",
    "month": "textbox",
    "number": "spinbutton",
    "password": "textbox",
    "radio": "radio",
    "range": "slider",
    "reset": "button",
    "search": "searchbox",
    "submit": "button",
    "tel": "textbox",
    "text": "textbox",
    "time": "textbox",
    "url": "textbox",
    "week": "textbox",
}

_taxonomy = RoleTaxonomy()


# ── Parser class ──────────────────────────────────────────────────────────────

class HTMLAccessibilityParser:
    """Parse an HTML document into an :class:`AccessibilityTree`.

    Supports lxml for fast parsing with html5lib as a fallback for
    broken markup.
    """

    def __init__(
        self,
        *,
        use_html5lib: bool = False,
        include_text_nodes: bool = True,
        id_prefix: str = "n",
    ) -> None:
        self._use_html5lib = use_html5lib
        self._include_text_nodes = include_text_nodes
        self._id_prefix = id_prefix
        self._counter = 0

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, html: str) -> AccessibilityTree:
        """Parse an HTML string and return an AccessibilityTree."""
        self._counter = 0
        root_el = self._parse_html(html)
        root_node = self._parse_element(root_el, parent_id=None, depth=0)
        tree = AccessibilityTree(
            root=root_node,
            metadata={"source": "html", "parser": self._parser_name()},
        )
        tree.build_index()
        return tree

    # ── HTML parsing back-end ─────────────────────────────────────────────

    def _parse_html(self, html: str) -> Any:
        if self._use_html5lib and _HAS_HTML5LIB:
            doc = html5lib.parse(html, treebuilder="lxml", namespaceHTMLElements=False)
            return doc
        if _HAS_LXML:
            try:
                return html_fromstring(html)
            except Exception:
                if _HAS_HTML5LIB:
                    doc = html5lib.parse(
                        html, treebuilder="lxml", namespaceHTMLElements=False,
                    )
                    return doc
                raise
        raise ImportError("Either lxml or html5lib is required for HTML parsing")

    def _parser_name(self) -> str:
        if self._use_html5lib and _HAS_HTML5LIB:
            return "html5lib"
        return "lxml"

    # ── Recursive element parsing ─────────────────────────────────────────

    def _parse_element(
        self,
        element: Any,
        parent_id: Optional[str],
        depth: int,
    ) -> AccessibilityNode:
        tag = self._tag_name(element)
        node_id = self._generate_id(element)
        role = self._infer_role(element)
        name = self._extract_name(element)
        description = self._extract_description(element)
        bbox = self._extract_bounding_box(element)
        state = self._extract_state(element)
        properties = self._extract_properties(element)

        children: list[AccessibilityNode] = []
        child_index = 0

        # Process text before first child
        if self._include_text_nodes and element.text and element.text.strip():
            text_node = self._make_text_node(
                element.text.strip(), node_id, depth + 1, child_index,
            )
            children.append(text_node)
            child_index += 1

        for child_el in element:
            if self._should_skip(child_el):
                continue
            child_node = self._parse_element(child_el, parent_id=node_id, depth=depth + 1)
            child_node.index_in_parent = child_index
            children.append(child_node)
            child_index += 1

            # Tail text (text between this child and the next)
            if self._include_text_nodes and child_el.tail and child_el.tail.strip():
                tail_node = self._make_text_node(
                    child_el.tail.strip(), node_id, depth + 1, child_index,
                )
                children.append(tail_node)
                child_index += 1

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
            index_in_parent=0,
        )

    # ── Role inference ────────────────────────────────────────────────────

    def _infer_role(self, element: Any) -> str:
        """Map an HTML element to its implicit ARIA role."""
        # Explicit role attribute takes precedence
        explicit = self._attr(element, "role")
        if explicit:
            return explicit.strip().split()[0].lower()

        tag = self._tag_name(element)

        if tag == "input":
            input_type = (self._attr(element, "type") or "text").lower()
            return _INPUT_TYPE_ROLE_MAP.get(input_type, "textbox")

        if tag == "a":
            return "link" if self._attr(element, "href") is not None else "generic"

        if tag == "img":
            alt = self._attr(element, "alt")
            if alt is not None and alt.strip() == "":
                return "presentation"
            return "img"

        if tag == "th":
            scope = (self._attr(element, "scope") or "").lower()
            return "rowheader" if scope == "row" else "columnheader"

        return _IMPLICIT_ROLE_MAP.get(tag, "generic")

    # ── Name extraction ───────────────────────────────────────────────────

    def _extract_name(self, element: Any) -> str:
        """Extract the accessible name using the ARIA name computation rules."""
        # aria-label
        label = self._attr(element, "aria-label")
        if label and label.strip():
            return label.strip()

        # aria-labelledby (simplified — would need document context)
        labelledby = self._attr(element, "aria-labelledby")
        if labelledby:
            # Cannot resolve id references without document root; store raw
            return f"[labelledby:{labelledby}]"

        tag = self._tag_name(element)

        # img alt
        if tag == "img":
            alt = self._attr(element, "alt")
            if alt is not None:
                return alt.strip()

        # input with associated label via title or placeholder
        if tag == "input":
            placeholder = self._attr(element, "placeholder")
            if placeholder:
                return placeholder.strip()

        # Title attribute as fallback
        title = self._attr(element, "title")
        if title and title.strip():
            return title.strip()

        # Visible text content (direct text only, not deep)
        text = self._direct_text(element)
        if text:
            return text

        return ""

    def _extract_description(self, element: Any) -> str:
        desc = self._attr(element, "aria-describedby")
        if desc:
            return f"[describedby:{desc}]"
        title = self._attr(element, "title")
        if title:
            return title.strip()
        return ""

    # ── Bounding box extraction ───────────────────────────────────────────

    def _extract_bounding_box(self, element: Any) -> Optional[BoundingBox]:
        """Extract bounding box from data-* attributes or inline style."""
        # data-bbox="x,y,w,h"
        bbox_str = self._attr(element, "data-bbox")
        if bbox_str:
            try:
                parts = [float(x) for x in bbox_str.split(",")]
                if len(parts) == 4:
                    return BoundingBox(*parts)
            except (ValueError, TypeError):
                pass

        # Individual data attributes
        dx = self._attr(element, "data-x")
        dy = self._attr(element, "data-y")
        dw = self._attr(element, "data-width")
        dh = self._attr(element, "data-height")
        if all(v is not None for v in (dx, dy, dw, dh)):
            try:
                return BoundingBox(float(dx), float(dy), float(dw), float(dh))  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass

        # Inline style (crude extraction)
        style = self._attr(element, "style")
        if style:
            return self._bbox_from_style(style)

        return None

    @staticmethod
    def _bbox_from_style(style: str) -> Optional[BoundingBox]:
        """Parse pixel values from inline CSS style string."""
        props: dict[str, float] = {}
        for prop_name in ("left", "top", "width", "height"):
            m = re.search(rf"{prop_name}\s*:\s*([\d.]+)\s*px", style)
            if m:
                props[prop_name] = float(m.group(1))
        if {"left", "top", "width", "height"} <= props.keys():
            return BoundingBox(props["left"], props["top"], props["width"], props["height"])
        return None

    # ── State extraction ──────────────────────────────────────────────────

    def _extract_state(self, element: Any) -> AccessibilityState:
        tag = self._tag_name(element)
        return AccessibilityState(
            focused=False,
            selected=self._has_attr(element, "selected")
            or self._attr(element, "aria-selected") == "true",
            expanded=self._attr(element, "aria-expanded") == "true",
            checked=self._parse_tri_state(self._attr(element, "aria-checked"))
            or (
                True
                if self._has_attr(element, "checked")
                and tag == "input"
                else None
            ),
            disabled=self._has_attr(element, "disabled")
            or self._attr(element, "aria-disabled") == "true",
            hidden=self._has_attr(element, "hidden")
            or self._attr(element, "aria-hidden") == "true"
            or self._is_style_hidden(element),
            required=self._has_attr(element, "required")
            or self._attr(element, "aria-required") == "true",
            readonly=self._has_attr(element, "readonly")
            or self._attr(element, "aria-readonly") == "true",
            pressed=self._parse_tri_state(self._attr(element, "aria-pressed")),
            value=self._attr(element, "value")
            or self._attr(element, "aria-valuenow"),
        )

    @staticmethod
    def _parse_tri_state(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        return None  # "mixed" etc.

    def _is_style_hidden(self, element: Any) -> bool:
        style = self._attr(element, "style") or ""
        if "display:none" in style.replace(" ", "").lower():
            return True
        if "visibility:hidden" in style.replace(" ", "").lower():
            return True
        return False

    # ── Property extraction ───────────────────────────────────────────────

    def _extract_properties(self, element: Any) -> dict[str, Any]:
        props: dict[str, Any] = {}
        tag = self._tag_name(element)
        props["tag"] = tag

        for attr_name in ("tabindex", "aria-level", "aria-valuemin", "aria-valuemax",
                          "aria-valuenow", "aria-colcount", "aria-rowcount",
                          "aria-posinset", "aria-setsize", "aria-live",
                          "aria-atomic", "aria-relevant", "aria-controls",
                          "aria-owns", "aria-flowto", "class", "id", "href", "type"):
            val = self._attr(element, attr_name)
            if val is not None:
                props[attr_name] = val

        # heading level
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            props["level"] = int(tag[1])

        return props

    # ── Skip logic ────────────────────────────────────────────────────────

    def _should_skip(self, element: Any) -> bool:
        """True if this element should be excluded from the tree."""
        tag = self._tag_name(element)
        if tag in _SKIP_TAGS:
            return True
        if self._attr(element, "aria-hidden") == "true":
            return True
        # Comments and processing instructions
        if not isinstance(getattr(element, "tag", None), str):
            return True
        return False

    # ── ID generation ─────────────────────────────────────────────────────

    def _generate_id(self, element: Any) -> str:
        explicit_id = self._attr(element, "id")
        if explicit_id:
            return f"{self._id_prefix}-{explicit_id}"
        self._counter += 1
        return f"{self._id_prefix}-{self._counter}"

    # ── Text helpers ──────────────────────────────────────────────────────

    def _make_text_node(
        self, text: str, parent_id: str, depth: int, index: int,
    ) -> AccessibilityNode:
        self._counter += 1
        return AccessibilityNode(
            id=f"{self._id_prefix}-text-{self._counter}",
            role="text",
            name=text,
            parent_id=parent_id,
            depth=depth,
            index_in_parent=index,
        )

    # ── lxml helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _tag_name(element: Any) -> str:
        tag = getattr(element, "tag", "")
        if not isinstance(tag, str):
            return ""
        # Strip namespace if present
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        return tag.lower()

    @staticmethod
    def _attr(element: Any, name: str) -> Optional[str]:
        try:
            return element.get(name)
        except Exception:
            return None

    @staticmethod
    def _has_attr(element: Any, name: str) -> bool:
        try:
            return name in element.attrib
        except Exception:
            return False

    @staticmethod
    def _direct_text(element: Any) -> str:
        """Get direct text content (not tail, not from descendants)."""
        try:
            text = element.text or ""
            return text.strip()
        except Exception:
            return ""
