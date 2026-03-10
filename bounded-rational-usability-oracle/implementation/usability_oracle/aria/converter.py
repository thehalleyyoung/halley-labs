"""
usability_oracle.aria.converter — Convert ARIA tree to internal AccessibilityTree.

Maps ARIA roles, properties, and states from the parsed :class:`AriaTree`
to the platform-agnostic :class:`AccessibilityTree` model used by the
rest of the usability oracle pipeline.

Reference: WAI-ARIA 1.2 §5, HTML-AAM §3.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.aria.parser import AriaNodeInfo, AriaTree
from usability_oracle.aria.taxonomy import get_role
from usability_oracle.core.enums import AccessibilityRole
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


# ═══════════════════════════════════════════════════════════════════════════
# ARIA role → internal AccessibilityRole mapping
# ═══════════════════════════════════════════════════════════════════════════

_ROLE_MAP: Dict[str, str] = {
    # Widget roles
    "button": AccessibilityRole.BUTTON.value,
    "checkbox": AccessibilityRole.CHECKBOX.value,
    "combobox": AccessibilityRole.COMBOBOX.value,
    "grid": AccessibilityRole.TABLE.value,
    "gridcell": AccessibilityRole.CELL.value,
    "link": AccessibilityRole.LINK.value,
    "listbox": AccessibilityRole.LIST.value,
    "menu": AccessibilityRole.MENU.value,
    "menubar": AccessibilityRole.MENU.value,
    "menuitem": AccessibilityRole.MENUITEM.value,
    "menuitemcheckbox": AccessibilityRole.MENUITEM.value,
    "menuitemradio": AccessibilityRole.MENUITEM.value,
    "option": AccessibilityRole.LISTITEM.value,
    "progressbar": AccessibilityRole.GENERIC.value,
    "radio": AccessibilityRole.RADIO.value,
    "radiogroup": AccessibilityRole.GROUP.value,
    "scrollbar": AccessibilityRole.SEPARATOR.value,
    "searchbox": AccessibilityRole.TEXTFIELD.value,
    "slider": AccessibilityRole.SLIDER.value,
    "spinbutton": AccessibilityRole.TEXTFIELD.value,
    "switch": AccessibilityRole.CHECKBOX.value,
    "tab": AccessibilityRole.TAB.value,
    "tablist": AccessibilityRole.GROUP.value,
    "tabpanel": AccessibilityRole.REGION.value,
    "textbox": AccessibilityRole.TEXTFIELD.value,
    "tree": AccessibilityRole.TREE.value,
    "treegrid": AccessibilityRole.TREE.value,
    "treeitem": AccessibilityRole.TREEITEM.value,
    "separator": AccessibilityRole.SEPARATOR.value,
    # Document structure roles
    "article": AccessibilityRole.DOCUMENT.value,
    "blockquote": AccessibilityRole.GROUP.value,
    "caption": AccessibilityRole.GENERIC.value,
    "cell": AccessibilityRole.CELL.value,
    "code": AccessibilityRole.GENERIC.value,
    "columnheader": AccessibilityRole.CELL.value,
    "definition": AccessibilityRole.GENERIC.value,
    "deletion": AccessibilityRole.GENERIC.value,
    "directory": AccessibilityRole.LIST.value,
    "document": AccessibilityRole.DOCUMENT.value,
    "emphasis": AccessibilityRole.GENERIC.value,
    "feed": AccessibilityRole.LIST.value,
    "figure": AccessibilityRole.GROUP.value,
    "generic": AccessibilityRole.GENERIC.value,
    "group": AccessibilityRole.GROUP.value,
    "heading": AccessibilityRole.HEADING.value,
    "img": AccessibilityRole.IMAGE.value,
    "insertion": AccessibilityRole.GENERIC.value,
    "list": AccessibilityRole.LIST.value,
    "listitem": AccessibilityRole.LISTITEM.value,
    "math": AccessibilityRole.GENERIC.value,
    "meter": AccessibilityRole.GENERIC.value,
    "none": AccessibilityRole.GENERIC.value,
    "note": AccessibilityRole.GROUP.value,
    "paragraph": AccessibilityRole.GENERIC.value,
    "presentation": AccessibilityRole.GENERIC.value,
    "row": AccessibilityRole.ROW.value,
    "rowgroup": AccessibilityRole.GROUP.value,
    "rowheader": AccessibilityRole.CELL.value,
    "strong": AccessibilityRole.GENERIC.value,
    "subscript": AccessibilityRole.GENERIC.value,
    "superscript": AccessibilityRole.GENERIC.value,
    "table": AccessibilityRole.TABLE.value,
    "term": AccessibilityRole.GENERIC.value,
    "time": AccessibilityRole.GENERIC.value,
    "toolbar": AccessibilityRole.TOOLBAR.value,
    "tooltip": AccessibilityRole.GENERIC.value,
    # Landmark roles
    "banner": AccessibilityRole.BANNER.value,
    "complementary": AccessibilityRole.COMPLEMENTARY.value,
    "contentinfo": AccessibilityRole.CONTENTINFO.value,
    "form": AccessibilityRole.FORM.value,
    "main": AccessibilityRole.MAIN.value,
    "navigation": AccessibilityRole.NAVIGATION.value,
    "region": AccessibilityRole.REGION.value,
    "search": AccessibilityRole.SEARCH.value,
    # Live-region roles
    "alert": AccessibilityRole.ALERT.value,
    "log": AccessibilityRole.GROUP.value,
    "marquee": AccessibilityRole.GROUP.value,
    "status": AccessibilityRole.ALERT.value,
    "timer": AccessibilityRole.ALERT.value,
    # Window roles
    "alertdialog": AccessibilityRole.DIALOG.value,
    "dialog": AccessibilityRole.DIALOG.value,
}

# Available interaction types by role
_INTERACTION_MAP: Dict[str, list[str]] = {
    "button": ["click"],
    "link": ["click"],
    "checkbox": ["click"],
    "radio": ["click"],
    "switch": ["click"],
    "textbox": ["type", "focus"],
    "searchbox": ["type", "focus"],
    "combobox": ["click", "type", "select"],
    "slider": ["drag", "focus"],
    "spinbutton": ["type", "focus"],
    "menuitem": ["click"],
    "menuitemcheckbox": ["click"],
    "menuitemradio": ["click"],
    "option": ["click"],
    "tab": ["click"],
    "treeitem": ["click"],
    "listbox": ["select"],
    "grid": ["select"],
    "scrollbar": ["drag"],
}


# ═══════════════════════════════════════════════════════════════════════════
# AriaToAccessibilityConverter
# ═══════════════════════════════════════════════════════════════════════════

class AriaToAccessibilityConverter:
    """Convert an :class:`AriaTree` to the internal :class:`AccessibilityTree`.

    Maps ARIA roles to :class:`AccessibilityRole` values, translates
    ARIA states to :class:`AccessibilityState`, and computes tab order.

    Usage::

        converter = AriaToAccessibilityConverter()
        a11y_tree = converter.convert(aria_tree)
    """

    def convert(self, aria_tree: AriaTree) -> AccessibilityTree:
        """Convert a full :class:`AriaTree` to :class:`AccessibilityTree`.

        Parameters:
            aria_tree: Parsed ARIA tree from :class:`AriaHTMLParser`.

        Returns:
            Platform-agnostic :class:`AccessibilityTree`.
        """
        root_node = self._convert_node(aria_tree.root)
        tree = AccessibilityTree(
            root=root_node,
            metadata={
                "source_format": "aria",
                "document_title": aria_tree.document_title,
                "landmark_count": len(aria_tree.landmarks),
            },
        )

        # Compute and store tab order
        tab_order = self.compute_tab_order(aria_tree)
        tree.metadata["tab_order"] = tab_order

        return tree

    def map_role(self, aria_role: str) -> str:
        """Map an ARIA role name to the internal AccessibilityRole value.

        Parameters:
            aria_role: ARIA role string (e.g. ``"button"``).

        Returns:
            Internal role string (e.g. ``"button"``).
        """
        return _ROLE_MAP.get(aria_role, AccessibilityRole.GENERIC.value)

    def extract_spatial_layout(self, element: AriaNodeInfo) -> Optional[BoundingBox]:
        """Extract bounding box from stored style/CSS properties.

        Looks for ``data-bbox``, ``data-x``/``data-y``/``data-width``/``data-height``,
        or inline style dimensions in the ARIA node properties.

        Parameters:
            element: An :class:`AriaNodeInfo` node.

        Returns:
            :class:`BoundingBox` if spatial info is available, else ``None``.
        """
        props = element.properties

        # Check for data-bbox format: "x,y,width,height"
        bbox_str = props.get("bbox") or props.get("data-bbox")
        if bbox_str:
            try:
                parts = [float(p) for p in bbox_str.split(",")]
                if len(parts) == 4:
                    return BoundingBox(
                        x=parts[0], y=parts[1],
                        width=parts[2], height=parts[3],
                    )
            except (ValueError, IndexError):
                pass

        # Check for individual coordinate properties
        try:
            x = float(props.get("x", props.get("data-x", "")))
            y = float(props.get("y", props.get("data-y", "")))
            w = float(props.get("width", props.get("data-width", "")))
            h = float(props.get("height", props.get("data-height", "")))
            return BoundingBox(x=x, y=y, width=w, height=h)
        except (ValueError, TypeError):
            pass

        return None

    def build_interaction_model(self, element: AriaNodeInfo) -> list[str]:
        """Determine available interactions for an ARIA element.

        Based on the element's role, returns a list of interaction types
        (e.g. ``["click"]``, ``["type", "focus"]``).

        Parameters:
            element: An :class:`AriaNodeInfo` node.

        Returns:
            List of interaction type strings.
        """
        interactions = list(_INTERACTION_MAP.get(element.role, []))

        # All focusable elements support "focus"
        if element.is_focusable and "focus" not in interactions:
            interactions.append("focus")

        return interactions

    def compute_tab_order(self, tree: AriaTree) -> list[str]:
        """Determine keyboard navigation order per tabindex handling.

        Tab order follows DOM order for elements with tabindex=0 or
        natively focusable elements, preceded by elements with positive
        tabindex values in ascending order.

        Parameters:
            tree: The full :class:`AriaTree`.

        Returns:
            Ordered list of node ids in tab sequence.

        Reference: HTML 5.2 §6.4.1 — Sequential focus navigation order.
        """
        positive_tabindex: list[tuple[int, str]] = []
        zero_or_native: list[str] = []

        # Walk in document order (pre-order)
        stack: list[AriaNodeInfo] = [tree.root]
        order: list[AriaNodeInfo] = []
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(reversed(node.children))

        for node in order:
            if not node.is_focusable:
                continue
            if node.tabindex is not None and node.tabindex > 0:
                positive_tabindex.append((node.tabindex, node.node_id))
            else:
                zero_or_native.append(node.node_id)

        # Sort positive tabindex by value (stable sort preserves DOM order)
        positive_tabindex.sort(key=lambda t: t[0])

        return [nid for _, nid in positive_tabindex] + zero_or_native

    # ── Internal helpers ──────────────────────────────────────────────────

    def _convert_node(self, node: AriaNodeInfo, depth: int = 0) -> AccessibilityNode:
        """Recursively convert an AriaNodeInfo to AccessibilityNode."""
        role = self.map_role(node.role)
        state = self._convert_state(node)
        bbox = self.extract_spatial_layout(node)
        interactions = self.build_interaction_model(node)

        props: dict[str, Any] = dict(node.properties)
        props["aria_role"] = node.role
        props["tag"] = node.tag
        if interactions:
            props["interactions"] = interactions
        if node.tabindex is not None:
            props["tabindex"] = node.tabindex

        children = [
            self._convert_node(child, depth=depth + 1)
            for child in node.children
        ]

        return AccessibilityNode(
            id=node.node_id,
            role=role,
            name=node.accessible_name,
            description=node.accessible_description,
            bounding_box=bbox,
            properties=props,
            state=state,
            children=children,
            parent_id=node.parent_id,
            depth=depth,
        )

    def _convert_state(self, node: AriaNodeInfo) -> AccessibilityState:
        """Convert ARIA states dict to AccessibilityState."""
        states = node.states

        def _bool(key: str) -> bool:
            val = states.get(key, "").lower()
            return val == "true"

        def _tri(key: str) -> Optional[bool]:
            val = states.get(key, "").lower()
            if val == "true":
                return True
            if val == "false":
                return False
            if val == "mixed":
                return None
            return None

        return AccessibilityState(
            focused=False,
            selected=_bool("selected"),
            expanded=_bool("expanded"),
            checked=_tri("checked"),
            disabled=_bool("disabled"),
            hidden=_bool("hidden"),
            required=_bool("required"),
            readonly=_bool("readonly"),
            pressed=_tri("pressed"),
            value=states.get("valuenow") or node.properties.get("valuenow"),
        )
