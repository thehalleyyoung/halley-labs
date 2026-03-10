"""
usability_oracle.aria.parser — ARIA HTML parser.

Parses HTML documents to extract WAI-ARIA roles, states, properties,
and landmark regions, constructing an intermediate AriaTree representation
per the WAI-ARIA 1.2 specification (W3C Recommendation, 6 June 2023).

Reference: https://www.w3.org/TR/wai-aria-1.2/
           https://www.w3.org/TR/html-aam-1.0/
           https://www.w3.org/TR/accname-1.2/
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

try:
    from lxml import etree
    from lxml.html import fromstring as html_fromstring

    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

try:
    import html5lib

    _HAS_HTML5LIB = True
except ImportError:
    _HAS_HTML5LIB = False

from usability_oracle.aria.types import (
    AriaProperty,
    AriaRole,
    AriaState,
    ConformanceLevel,
    ConformanceResult,
    LandmarkRegion,
    PropertyType,
    RoleCategory,
)
from usability_oracle.aria.taxonomy import ROLE_TAXONOMY, get_role
from usability_oracle.core.errors import ParseError


# ═══════════════════════════════════════════════════════════════════════════
# Constants — HTML5 implicit ARIA role mapping (HTML-AAM §3.4)
# ═══════════════════════════════════════════════════════════════════════════

_SKIP_TAGS: FrozenSet[str] = frozenset({
    "script", "style", "noscript", "template", "head", "meta", "link",
    "base", "title",
})

_IMPLICIT_ROLE_MAP: Dict[str, str] = {
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

_INPUT_TYPE_ROLE_MAP: Dict[str, str] = {
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

_LANDMARK_ROLES: FrozenSet[str] = frozenset({
    "banner", "complementary", "contentinfo", "form",
    "main", "navigation", "region", "search",
})

_INTERACTIVE_ROLES: FrozenSet[str] = frozenset({
    "button", "checkbox", "combobox", "gridcell", "link", "listbox",
    "menu", "menubar", "menuitem", "menuitemcheckbox", "menuitemradio",
    "option", "radio", "scrollbar", "searchbox", "slider", "spinbutton",
    "switch", "tab", "textbox", "treeitem",
})


# ═══════════════════════════════════════════════════════════════════════════
# AriaNodeInfo — intermediate per-node representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AriaNodeInfo:
    """Parsed ARIA information for a single DOM element."""

    node_id: str
    tag: str
    role: str
    accessible_name: str = ""
    accessible_description: str = ""
    properties: Dict[str, str] = field(default_factory=dict)
    states: Dict[str, str] = field(default_factory=dict)
    children: list[AriaNodeInfo] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    is_focusable: bool = False
    tabindex: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tag": self.tag,
            "role": self.role,
            "accessible_name": self.accessible_name,
            "accessible_description": self.accessible_description,
            "properties": self.properties,
            "states": self.states,
            "children": [c.to_dict() for c in self.children],
            "depth": self.depth,
            "is_focusable": self.is_focusable,
            "tabindex": self.tabindex,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AriaTree — full parsed document
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AriaTree:
    """Complete ARIA-annotated representation of an HTML document.

    Attributes:
        root: Root node of the ARIA tree.
        node_index: id → AriaNodeInfo mapping for O(1) lookup.
        landmarks: Detected landmark regions.
        document_title: The ``<title>`` of the document, if present.
    """

    root: AriaNodeInfo
    node_index: Dict[str, AriaNodeInfo] = field(default_factory=dict)
    landmarks: List[LandmarkRegion] = field(default_factory=list)
    document_title: str = ""

    def __post_init__(self) -> None:
        if not self.node_index:
            self._build_index()

    def _build_index(self) -> None:
        """Walk the tree and build the id → node index."""
        self.node_index.clear()
        stack: list[AriaNodeInfo] = [self.root]
        while stack:
            node = stack.pop()
            self.node_index[node.node_id] = node
            stack.extend(reversed(node.children))

    def get_node(self, node_id: str) -> Optional[AriaNodeInfo]:
        return self.node_index.get(node_id)

    @property
    def size(self) -> int:
        return len(self.node_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "landmarks": [lm.to_dict() for lm in self.landmarks],
            "document_title": self.document_title,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AriaHTMLParser
# ═══════════════════════════════════════════════════════════════════════════

class AriaHTMLParser:
    """Parse HTML content and extract ARIA roles, states, and properties.

    Implements the HTML-AAM implicit role mapping and WAI-ARIA 1.2
    explicit role extraction.  Uses lxml for parsing with optional
    html5lib fallback for malformed markup.

    Usage::

        parser = AriaHTMLParser()
        tree = parser.parse_html('<nav><a href="/">Home</a></nav>')
        assert tree.root.role == "document"
    """

    def __init__(self, *, use_html5lib: bool = False, id_prefix: str = "aria") -> None:
        self._use_html5lib = use_html5lib
        self._id_prefix = id_prefix
        self._counter = 0
        # Populated during parse for aria-labelledby resolution
        self._id_map: Dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def parse_html(self, html_content: str) -> AriaTree:
        """Parse HTML and extract ARIA roles, states, and properties.

        Parameters:
            html_content: Raw HTML string.

        Returns:
            Fully populated :class:`AriaTree`.

        Raises:
            ParseError: On empty or unparseable content.

        Reference: WAI-ARIA 1.2 §6, HTML-AAM §3.4.
        """
        if not html_content or not html_content.strip():
            raise ParseError("Empty HTML content")

        self._counter = 0
        self._id_map = {}

        doc = self._parse_document(html_content)
        # First pass: build id map for labelledby resolution
        self._build_id_map(doc)
        # Second pass: build AriaTree
        root = self._process_element(doc, depth=0)
        if root is None:
            raise ParseError("Could not extract any accessible nodes")

        title = self._extract_title(doc)
        tree = AriaTree(root=root, document_title=title)
        tree.landmarks = self.extract_landmark_regions(tree)
        return tree

    # ── Element processing ────────────────────────────────────────────────

    def _parse_document(self, html_content: str) -> Any:
        """Parse HTML string into an lxml element tree."""
        if self._use_html5lib and _HAS_HTML5LIB:
            doc = html5lib.parse(
                html_content,
                treebuilder="lxml",
                namespaceHTMLElements=False,
            )
            return doc
        if _HAS_LXML:
            return html_fromstring(html_content)
        raise ParseError("Neither lxml nor html5lib is available")

    def _build_id_map(self, element: Any) -> None:
        """Build a mapping of HTML id attributes to elements."""
        eid = element.get("id") if hasattr(element, "get") else None
        if eid:
            self._id_map[eid] = element
        for child in element:
            if isinstance(child.tag, str):
                self._build_id_map(child)

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self._id_prefix}-{self._counter}"

    def _process_element(
        self,
        element: Any,
        depth: int = 0,
        parent_id: Optional[str] = None,
    ) -> Optional[AriaNodeInfo]:
        """Recursively process a DOM element into an AriaNodeInfo."""
        tag = element.tag if isinstance(element.tag, str) else ""
        if not tag or tag in _SKIP_TAGS:
            return None

        local_tag = tag.split("}")[-1] if "}" in tag else tag
        node_id = self._next_id()

        # Determine role
        role = self.extract_explicit_roles(element)
        if role is None:
            role = self.extract_implicit_roles(element)

        # Extract properties and states
        properties = self.extract_aria_properties(element)
        states: Dict[str, str] = {}
        for key in list(properties.keys()):
            if key in ("checked", "disabled", "expanded", "hidden",
                       "pressed", "selected", "invalid", "grabbed"):
                states[key] = properties.pop(key)

        # Accessible name
        acc_name = self.build_accessible_name(element)
        acc_desc = properties.pop("describedby_text", "")

        # Focusability / tabindex
        tabindex = self._parse_tabindex(element)
        is_focusable = self._compute_focusable(element, role, tabindex)

        node = AriaNodeInfo(
            node_id=node_id,
            tag=local_tag,
            role=role,
            accessible_name=acc_name,
            accessible_description=acc_desc,
            properties=properties,
            states=states,
            depth=depth,
            parent_id=parent_id,
            is_focusable=is_focusable,
            tabindex=tabindex,
        )

        # Process children
        for child_elem in element:
            if isinstance(child_elem.tag, str):
                child_node = self._process_element(
                    child_elem, depth=depth + 1, parent_id=node_id,
                )
                if child_node is not None:
                    node.children.append(child_node)

        return node

    # ── Role extraction ───────────────────────────────────────────────────

    def extract_implicit_roles(self, element: Any) -> str:
        """Map HTML5 elements to implicit ARIA roles per HTML-AAM §3.4.

        Parameters:
            element: An lxml HTML element.

        Returns:
            Implicit ARIA role string, or ``"generic"`` if no mapping.

        Reference: https://www.w3.org/TR/html-aam-1.0/#html-element-role-mappings
        """
        tag = element.tag if isinstance(element.tag, str) else ""
        local_tag = tag.split("}")[-1] if "}" in tag else tag
        local_tag = local_tag.lower()

        if local_tag == "input":
            input_type = (element.get("type") or "text").lower()
            return _INPUT_TYPE_ROLE_MAP.get(input_type, "textbox")

        if local_tag == "a" and element.get("href") is None:
            return "generic"

        if local_tag == "img":
            alt = element.get("alt")
            if alt == "":
                return "presentation"
            return "img"

        if local_tag in ("header", "footer"):
            # Scoped: inside article/aside/main/nav/section → generic
            # At top level → banner/contentinfo
            # Simplified: return the landmark role
            return _IMPLICIT_ROLE_MAP.get(local_tag, "generic")

        if local_tag == "section":
            # section with accessible name → region; otherwise generic
            if element.get("aria-label") or element.get("aria-labelledby"):
                return "region"
            return "generic"

        return _IMPLICIT_ROLE_MAP.get(local_tag, "generic")

    def extract_explicit_roles(self, element: Any) -> Optional[str]:
        """Extract explicit ARIA role from the ``role`` attribute.

        Returns the first valid, non-abstract role from the space-separated
        token list, per WAI-ARIA 1.2 §7.1: *"The first token in the
        sequence that matches the name of a non-abstract WAI-ARIA role"*.

        Parameters:
            element: An lxml HTML element.

        Returns:
            The resolved role string, or ``None`` if no explicit role.

        Reference: WAI-ARIA 1.2 §7.1 — Role attribute processing.
        """
        role_attr = element.get("role")
        if not role_attr:
            return None

        for token in role_attr.strip().lower().split():
            role_def = get_role(token)
            if role_def is not None and not role_def.is_abstract:
                return token

        return None

    def extract_aria_properties(self, element: Any) -> Dict[str, str]:
        """Extract all ``aria-*`` attributes from an element.

        Collects all attributes starting with ``aria-`` and returns them
        as a dict keyed by the property name (without the ``aria-`` prefix).

        Parameters:
            element: An lxml HTML element.

        Returns:
            Dictionary mapping property names to their string values.

        Reference: WAI-ARIA 1.2 §6.6 — State and property attributes.
        """
        props: Dict[str, str] = {}
        for attr, value in element.attrib.items():
            if attr.startswith("aria-"):
                prop_name = attr[5:]  # strip "aria-"
                props[prop_name] = value
        return props

    def extract_landmark_regions(self, tree: AriaTree) -> List[LandmarkRegion]:
        """Identify all landmark regions in the parsed tree.

        Scans the tree for nodes whose roles are ARIA landmark roles and
        constructs :class:`LandmarkRegion` descriptors.

        Parameters:
            tree: A fully parsed :class:`AriaTree`.

        Returns:
            List of :class:`LandmarkRegion` instances.

        Reference: WAI-ARIA 1.2 §5.3.4 — Landmark roles.
        """
        landmarks: List[LandmarkRegion] = []

        for node in tree.node_index.values():
            if node.role in _LANDMARK_ROLES:
                child_lm_ids = tuple(
                    c.node_id
                    for c in node.children
                    if c.role in _LANDMARK_ROLES
                )
                contains_interactive = any(
                    self._subtree_has_interactive(c) for c in node.children
                )
                landmarks.append(LandmarkRegion(
                    role=node.role,
                    label=node.accessible_name,
                    node_id=node.node_id,
                    child_landmark_ids=child_lm_ids,
                    contains_interactive=contains_interactive,
                ))

        return landmarks

    def resolve_label(self, element: Any, document: Any) -> str:
        """Resolve the accessible label following the ARIA label chain.

        Precedence (WAI-ARIA 1.2 §4.3, Accessible Name §4.3):
        1. ``aria-labelledby`` → concatenate text of referenced elements
        2. ``aria-label``
        3. ``<label>`` association (for form controls)
        4. ``title`` attribute
        5. ``placeholder`` attribute

        Parameters:
            element: The target lxml element.
            document: The root document element (for id lookups).

        Returns:
            Resolved label string, may be empty.

        Reference: https://www.w3.org/TR/accname-1.2/#step2
        """
        # 1. aria-labelledby
        labelledby = element.get("aria-labelledby")
        if labelledby:
            parts = []
            for ref_id in labelledby.split():
                ref_el = self._id_map.get(ref_id)
                if ref_el is not None:
                    parts.append(self._get_text_content(ref_el))
            if parts:
                return " ".join(parts)

        # 2. aria-label
        label = element.get("aria-label")
        if label and label.strip():
            return label.strip()

        # 3. title
        title = element.get("title")
        if title and title.strip():
            return title.strip()

        # 4. placeholder
        placeholder = element.get("placeholder")
        if placeholder and placeholder.strip():
            return placeholder.strip()

        return ""

    def build_accessible_name(self, element: Any) -> str:
        """Compute the accessible name per the W3C Accessible Name algorithm.

        Implements a simplified version of the Accessible Name and
        Description Computation 1.2 specification.

        Parameters:
            element: An lxml HTML element.

        Returns:
            The computed accessible name string.

        Reference: https://www.w3.org/TR/accname-1.2/#mapping_additional_nd_te
        """
        # Step 1: aria-labelledby (unless we're already recursing from labelledby)
        labelledby = element.get("aria-labelledby")
        if labelledby:
            parts = []
            for ref_id in labelledby.split():
                ref_el = self._id_map.get(ref_id)
                if ref_el is not None:
                    parts.append(self._get_text_content(ref_el))
            result = " ".join(parts)
            if result.strip():
                return result.strip()

        # Step 2: aria-label
        label = element.get("aria-label")
        if label and label.strip():
            return label.strip()

        # Step 3: Native host language — <label>, alt, value, etc.
        tag = element.tag if isinstance(element.tag, str) else ""
        local_tag = tag.split("}")[-1].lower() if tag else ""

        if local_tag == "img":
            alt = element.get("alt")
            if alt is not None:
                return alt.strip()

        if local_tag == "input":
            input_type = (element.get("type") or "text").lower()
            if input_type == "image":
                alt = element.get("alt")
                if alt:
                    return alt.strip()
            value = element.get("value")
            if value and input_type in ("submit", "reset", "button"):
                return value.strip()

        if local_tag == "textarea":
            text = self._get_text_content(element)
            if text.strip():
                return text.strip()

        # Step 4: For name-from-contents roles, use text content
        explicit_role = self.extract_explicit_roles(element)
        role_name = explicit_role or self.extract_implicit_roles(element)
        role_def = get_role(role_name)
        if role_def is not None and role_def.name_from == "contents":
            text = self._get_text_content(element)
            if text.strip():
                return text.strip()

        # Step 5: title attribute
        title = element.get("title")
        if title and title.strip():
            return title.strip()

        # Step 6: placeholder
        placeholder = element.get("placeholder")
        if placeholder and placeholder.strip():
            return placeholder.strip()

        # Fallback: direct text content for leaf elements
        if len(element) == 0:
            text = self._get_text_content(element)
            if text.strip():
                return text.strip()

        return ""

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_text_content(self, element: Any) -> str:
        """Extract all text content from an element, recursively."""
        parts: list[str] = []
        if element.text:
            parts.append(element.text)
        for child in element:
            parts.append(self._get_text_content(child))
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts)

    def _extract_title(self, doc: Any) -> str:
        """Extract the <title> text from the document."""
        title_el = doc.find(".//title")
        if title_el is not None and title_el.text:
            return title_el.text.strip()
        return ""

    def _parse_tabindex(self, element: Any) -> Optional[int]:
        """Parse the tabindex attribute, returning None if absent."""
        raw = element.get("tabindex")
        if raw is None:
            return None
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None

    def _compute_focusable(
        self, element: Any, role: str, tabindex: Optional[int],
    ) -> bool:
        """Determine whether an element is keyboard-focusable.

        Natively focusable elements (links with href, buttons, inputs,
        textareas, selects) plus any element with tabindex >= 0.

        Reference: HTML 5.2 §6.4.1 — Sequential focus navigation.
        """
        tag = element.tag if isinstance(element.tag, str) else ""
        local_tag = tag.split("}")[-1].lower() if tag else ""

        if tabindex is not None:
            return tabindex >= 0

        natively_focusable = {"a", "button", "input", "select", "textarea"}
        if local_tag in natively_focusable:
            if local_tag == "a" and element.get("href") is None:
                return False
            if local_tag == "input" and (element.get("type") or "").lower() == "hidden":
                return False
            return element.get("disabled") is None

        return False

    def _subtree_has_interactive(self, node: AriaNodeInfo) -> bool:
        """Check if a subtree contains any interactive-role nodes."""
        if node.role in _INTERACTIVE_ROLES:
            return True
        return any(self._subtree_has_interactive(c) for c in node.children)
