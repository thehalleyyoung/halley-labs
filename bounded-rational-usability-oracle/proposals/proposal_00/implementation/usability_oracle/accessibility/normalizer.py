"""Cross-platform accessibility tree normalisation.

Normalises trees from different sources (HTML, Chrome DevTools, axe-core,
platform APIs) into a uniform representation with canonical ARIA roles,
cleaned names, uniform bounding-box coordinates, and pruned decorative nodes.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.roles import RoleTaxonomy


# ── Platform-specific role mapping ────────────────────────────────────────────

ROLE_MAPPINGS: dict[str, str] = {
    # macOS / NSAccessibility
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXComboBox": "combobox",
    "AXGroup": "group",
    "AXHeading": "heading",
    "AXImage": "img",
    "AXIncrementor": "spinbutton",
    "AXLayoutArea": "generic",
    "AXLayoutItem": "generic",
    "AXLink": "link",
    "AXList": "list",
    "AXMenu": "menu",
    "AXMenuBar": "menubar",
    "AXMenuItem": "menuitem",
    "AXMenuButton": "button",
    "AXOutline": "tree",
    "AXOutlineRow": "treeitem",
    "AXPopUpButton": "combobox",
    "AXRadioButton": "radio",
    "AXRadioGroup": "radiogroup",
    "AXRow": "row",
    "AXScrollArea": "region",
    "AXScrollBar": "scrollbar",
    "AXSlider": "slider",
    "AXSplitGroup": "group",
    "AXSplitter": "separator",
    "AXStaticText": "text",
    "AXTab": "tab",
    "AXTabGroup": "tablist",
    "AXTable": "table",
    "AXTextArea": "textbox",
    "AXTextField": "textbox",
    "AXToolbar": "toolbar",
    "AXValueIndicator": "meter",
    "AXWebArea": "document",
    "AXWindow": "window",
    # Windows / UIA (UI Automation)
    "UIA.Button": "button",
    "UIA.Calendar": "grid",
    "UIA.CheckBox": "checkbox",
    "UIA.ComboBox": "combobox",
    "UIA.DataGrid": "grid",
    "UIA.DataItem": "row",
    "UIA.Document": "document",
    "UIA.Edit": "textbox",
    "UIA.Group": "group",
    "UIA.Header": "banner",
    "UIA.HeaderItem": "columnheader",
    "UIA.Hyperlink": "link",
    "UIA.Image": "img",
    "UIA.List": "list",
    "UIA.ListItem": "listitem",
    "UIA.Menu": "menu",
    "UIA.MenuBar": "menubar",
    "UIA.MenuItem": "menuitem",
    "UIA.Pane": "region",
    "UIA.ProgressBar": "progressbar",
    "UIA.RadioButton": "radio",
    "UIA.ScrollBar": "scrollbar",
    "UIA.Separator": "separator",
    "UIA.Slider": "slider",
    "UIA.Spinner": "spinbutton",
    "UIA.SplitButton": "button",
    "UIA.StatusBar": "status",
    "UIA.Tab": "tablist",
    "UIA.TabItem": "tab",
    "UIA.Table": "table",
    "UIA.Text": "text",
    "UIA.Thumb": "slider",
    "UIA.TitleBar": "banner",
    "UIA.ToolBar": "toolbar",
    "UIA.ToolTip": "tooltip",
    "UIA.Tree": "tree",
    "UIA.TreeItem": "treeitem",
    "UIA.Window": "window",
    # Linux / ATK / AT-SPI
    "ATK_ROLE_PUSH_BUTTON": "button",
    "ATK_ROLE_CHECK_BOX": "checkbox",
    "ATK_ROLE_COMBO_BOX": "combobox",
    "ATK_ROLE_DIALOG": "dialog",
    "ATK_ROLE_ENTRY": "textbox",
    "ATK_ROLE_FRAME": "window",
    "ATK_ROLE_HEADING": "heading",
    "ATK_ROLE_IMAGE": "img",
    "ATK_ROLE_LABEL": "text",
    "ATK_ROLE_LINK": "link",
    "ATK_ROLE_LIST": "list",
    "ATK_ROLE_LIST_ITEM": "listitem",
    "ATK_ROLE_MENU": "menu",
    "ATK_ROLE_MENU_BAR": "menubar",
    "ATK_ROLE_MENU_ITEM": "menuitem",
    "ATK_ROLE_PAGE_TAB": "tab",
    "ATK_ROLE_PAGE_TAB_LIST": "tablist",
    "ATK_ROLE_PANEL": "group",
    "ATK_ROLE_PROGRESS_BAR": "progressbar",
    "ATK_ROLE_RADIO_BUTTON": "radio",
    "ATK_ROLE_SCROLL_BAR": "scrollbar",
    "ATK_ROLE_SEPARATOR": "separator",
    "ATK_ROLE_SLIDER": "slider",
    "ATK_ROLE_SPIN_BUTTON": "spinbutton",
    "ATK_ROLE_TABLE": "table",
    "ATK_ROLE_TABLE_CELL": "cell",
    "ATK_ROLE_TABLE_COLUMN_HEADER": "columnheader",
    "ATK_ROLE_TABLE_ROW_HEADER": "rowheader",
    "ATK_ROLE_TEXT": "text",
    "ATK_ROLE_TOGGLE_BUTTON": "button",
    "ATK_ROLE_TOOL_BAR": "toolbar",
    "ATK_ROLE_TOOL_TIP": "tooltip",
    "ATK_ROLE_TREE": "tree",
    "ATK_ROLE_TREE_TABLE": "treegrid",
    "ATK_ROLE_WINDOW": "window",
    # Common aliases
    "textfield": "textbox",
    "edittext": "textbox",
    "static_text": "text",
    "push_button": "button",
    "check_box": "checkbox",
    "combo_box": "combobox",
    "radio_button": "radio",
    "scroll_bar": "scrollbar",
    "spin_button": "spinbutton",
    "tab_list": "tablist",
    "tree_item": "treeitem",
    "list_item": "listitem",
    "menu_item": "menuitem",
    "progress_bar": "progressbar",
    "tool_bar": "toolbar",
    "tool_tip": "tooltip",
    "status_bar": "status",
    "web_area": "document",
}

_DECORATIVE_ROLES = frozenset({"presentation", "none", "generic"})

_taxonomy = RoleTaxonomy()

_WHITESPACE_RE = re.compile(r"\s+")


# ── Normaliser class ──────────────────────────────────────────────────────────

class AccessibilityNormalizer:
    """Normalise an :class:`AccessibilityTree` into a canonical form."""

    def __init__(
        self,
        *,
        remove_decorative: bool = True,
        collapse_wrappers: bool = True,
        normalize_coordinates: bool = True,
        target_viewport: Optional[BoundingBox] = None,
    ) -> None:
        self.remove_decorative = remove_decorative
        self.collapse_wrappers = collapse_wrappers
        self.normalize_coordinates = normalize_coordinates
        self.target_viewport = target_viewport or BoundingBox(0, 0, 1920, 1080)

    # ── Public API ────────────────────────────────────────────────────────

    def normalize(self, tree: AccessibilityTree) -> AccessibilityTree:
        """Return a normalised deep copy of the tree."""
        new_tree = AccessibilityTree(
            root=deepcopy(tree.root),
            metadata=dict(tree.metadata),
        )
        new_tree.build_index()

        self._normalize_roles(new_tree)
        self._normalize_names(new_tree)
        if self.normalize_coordinates:
            self._normalize_bounding_boxes(new_tree)
        if self.remove_decorative:
            self._remove_decorative(new_tree)
        if self.collapse_wrappers:
            self._collapse_wrappers(new_tree)
        self._assign_semantic_levels(new_tree)

        new_tree.build_index()
        new_tree.metadata["normalized"] = True
        return new_tree

    # ── Role normalization ────────────────────────────────────────────────

    def _normalize_roles(self, tree: AccessibilityTree) -> None:
        """Map platform-specific roles to canonical ARIA roles."""
        for node in tree.node_index.values():
            canonical = ROLE_MAPPINGS.get(node.role)
            if canonical is not None:
                node.properties["original_role"] = node.role
                node.role = canonical

    # ── Name normalization ────────────────────────────────────────────────

    def _normalize_names(self, tree: AccessibilityTree) -> None:
        """Strip extra whitespace, normalise unicode, clean names."""
        for node in tree.node_index.values():
            node.name = self._clean_text(node.name)
            node.description = self._clean_text(node.description)

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        # Unicode NFKC normalisation (combines compatibility chars)
        text = unicodedata.normalize("NFKC", text)
        # Collapse whitespace
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    # ── Bounding box normalization ────────────────────────────────────────

    def _normalize_bounding_boxes(self, tree: AccessibilityTree) -> None:
        """Scale bounding boxes to a uniform coordinate system."""
        boxes = [
            n.bounding_box for n in tree.node_index.values()
            if n.bounding_box is not None
        ]
        if not boxes:
            return

        # Determine source extent
        min_x = min(b.x for b in boxes)
        min_y = min(b.y for b in boxes)
        max_r = max(b.right for b in boxes)
        max_b = max(b.bottom for b in boxes)

        src_w = max_r - min_x
        src_h = max_b - min_y
        if src_w <= 0 or src_h <= 0:
            return

        scale_x = self.target_viewport.width / src_w
        scale_y = self.target_viewport.height / src_h

        for node in tree.node_index.values():
            if node.bounding_box is not None:
                bb = node.bounding_box
                node.bounding_box = BoundingBox(
                    x=(bb.x - min_x) * scale_x + self.target_viewport.x,
                    y=(bb.y - min_y) * scale_y + self.target_viewport.y,
                    width=bb.width * scale_x,
                    height=bb.height * scale_y,
                )

    # ── Remove decorative nodes ───────────────────────────────────────────

    def _remove_decorative(self, tree: AccessibilityTree) -> None:
        """Remove nodes that are purely decorative (presentation/none/generic with no name)."""
        self._prune_subtree(tree.root)

    def _prune_subtree(self, node: AccessibilityNode) -> None:
        """Recursively remove decorative leaf nodes."""
        # Process children first (bottom-up)
        for child in list(node.children):
            self._prune_subtree(child)

        # Filter out decorative leaves
        new_children: list[AccessibilityNode] = []
        for child in node.children:
            if self._is_decorative_leaf(child):
                continue
            new_children.append(child)
        node.children = new_children

    @staticmethod
    def _is_decorative_leaf(node: AccessibilityNode) -> bool:
        """A node is decorative if it has a decorative role, no name, and no children."""
        if node.children:
            return False
        if node.role not in _DECORATIVE_ROLES:
            return False
        if node.name.strip():
            return False
        return True

    # ── Collapse single-child wrappers ────────────────────────────────────

    def _collapse_wrappers(self, tree: AccessibilityTree) -> None:
        """Collapse nodes that have exactly one child and a generic role.

        The child inherits the wrapper's parent relationship.
        """
        self._collapse_node(tree.root)

    def _collapse_node(self, node: AccessibilityNode) -> None:
        """Bottom-up collapse of wrapper nodes."""
        for child in list(node.children):
            self._collapse_node(child)

        new_children: list[AccessibilityNode] = []
        for child in node.children:
            if self._is_collapsible_wrapper(child):
                # Replace wrapper with its single child
                grandchild = child.children[0]
                grandchild.parent_id = node.id
                grandchild.index_in_parent = child.index_in_parent
                # Inherit name if grandchild has none
                if not grandchild.name and child.name:
                    grandchild.name = child.name
                new_children.append(grandchild)
            else:
                new_children.append(child)
        node.children = new_children

    @staticmethod
    def _is_collapsible_wrapper(node: AccessibilityNode) -> bool:
        """True if node is a wrapper that can be collapsed."""
        if len(node.children) != 1:
            return False
        if node.role not in ("generic", "group", "none", "presentation"):
            return False
        # Don't collapse if the wrapper has meaningful state
        if node.state.focused or node.state.selected or node.state.expanded:
            return False
        return True

    # ── Semantic level assignment ─────────────────────────────────────────

    def _assign_semantic_levels(self, tree: AccessibilityTree) -> None:
        """Compute semantic depth: increments only for semantically meaningful roles."""
        self._walk_semantic_depth(tree.root, semantic_depth=0)

    def _walk_semantic_depth(self, node: AccessibilityNode, semantic_depth: int) -> None:
        """Recursively assign semantic depth based on role significance."""
        node.properties["semantic_level"] = semantic_depth

        child_depth = semantic_depth
        if self._is_semantic_boundary(node):
            child_depth = semantic_depth + 1

        for child in node.children:
            self._walk_semantic_depth(child, child_depth)

    @staticmethod
    def _is_semantic_boundary(node: AccessibilityNode) -> bool:
        """True if this node represents a semantic level boundary."""
        if _taxonomy.is_landmark(node.role):
            return True
        if node.role in ("dialog", "alertdialog", "document", "application"):
            return True
        if node.role in ("list", "tree", "grid", "table", "tablist", "menu", "menubar"):
            return True
        if node.role == "heading":
            return True
        return False
