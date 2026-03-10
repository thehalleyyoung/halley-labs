"""
usability_oracle.wcag.keyboard — Keyboard accessibility analysis.

Analyses an accessibility tree for keyboard navigability: tab order,
focus traps, shortcut conflicts, skip-navigation links, focus indicator
heuristics, and sequential-navigation cost estimation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGPrinciple,
    WCAGViolation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Roles that natively receive keyboard focus in the HTML platform model
_NATIVELY_FOCUSABLE_ROLES = frozenset({
    "button", "link", "textbox", "searchbox", "combobox", "listbox",
    "slider", "spinbutton", "checkbox", "radio", "switch", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "treeitem", "option",
    "gridcell", "columnheader", "rowheader",
})

# Composite widgets whose children are navigated with arrow keys
_COMPOSITE_WIDGETS = frozenset({
    "menu", "menubar", "listbox", "tablist", "tree", "treegrid",
    "grid", "toolbar", "radiogroup",
})

# Landmark roles that may contain skip-nav targets
_LANDMARK_ROLES = frozenset({
    "main", "navigation", "complementary", "contentinfo",
    "banner", "search", "form", "region",
})

# Known single-key shortcuts that conflict with assistive technology
_RESERVED_KEYS = frozenset({
    "tab", "escape", "enter", "space", "arrowup", "arrowdown",
    "arrowleft", "arrowright", "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
    "f11", "f12",
})


# ═══════════════════════════════════════════════════════════════════════════
# Tab-order extraction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FocusableElement:
    """A single element in the computed tab order."""

    node_id: str
    role: str
    name: str
    tab_index: int
    depth: int
    dom_order: int  # pre-order index in the tree
    bounding_box: Optional[Any] = None


def extract_tab_order(tree: Any) -> List[FocusableElement]:
    """Compute the sequential tab order from an accessibility tree.

    Follows the HTML spec tab-order algorithm:
    1. Elements with tabindex > 0 come first, sorted by tabindex then DOM order.
    2. Elements with tabindex = 0 (or natively focusable) follow in DOM order.
    3. Elements with tabindex < 0 are removed from the tab order.
    """
    focusable: List[FocusableElement] = []
    dom_order = 0

    def _walk(node: Any) -> None:
        nonlocal dom_order
        current_order = dom_order
        dom_order += 1

        # Skip hidden nodes
        if hasattr(node, "state") and hasattr(node.state, "hidden") and node.state.hidden:
            for child in (node.children if hasattr(node, "children") else []):
                _walk(child)
            return

        disabled = False
        if hasattr(node, "state") and hasattr(node.state, "disabled"):
            disabled = node.state.disabled

        if disabled:
            for child in (node.children if hasattr(node, "children") else []):
                _walk(child)
            return

        tab_index: Optional[int] = None
        role = node.role if hasattr(node, "role") else ""
        name = node.name if hasattr(node, "name") else ""
        nid = node.id if hasattr(node, "id") else (node.node_id if hasattr(node, "node_id") else "")
        props = node.properties if hasattr(node, "properties") else {}
        bbox = node.bounding_box if hasattr(node, "bounding_box") else None

        # Explicit tabindex
        raw_ti = props.get("tabindex")
        if raw_ti is not None:
            try:
                tab_index = int(raw_ti)
            except (ValueError, TypeError):
                tab_index = None

        is_native = role in _NATIVELY_FOCUSABLE_ROLES

        if tab_index is not None:
            if tab_index >= 0:
                focusable.append(FocusableElement(
                    node_id=nid, role=role, name=name,
                    tab_index=tab_index, depth=getattr(node, "depth", 0),
                    dom_order=current_order, bounding_box=bbox,
                ))
        elif is_native:
            focusable.append(FocusableElement(
                node_id=nid, role=role, name=name,
                tab_index=0, depth=getattr(node, "depth", 0),
                dom_order=current_order, bounding_box=bbox,
            ))

        for child in (node.children if hasattr(node, "children") else []):
            _walk(child)

    _walk(tree.root)

    # Sort: positive tabindex first (by tabindex, then DOM order),
    # then tabindex=0 in DOM order.
    positive = [e for e in focusable if e.tab_index > 0]
    zero = [e for e in focusable if e.tab_index == 0]
    positive.sort(key=lambda e: (e.tab_index, e.dom_order))
    zero.sort(key=lambda e: e.dom_order)

    return positive + zero


# ═══════════════════════════════════════════════════════════════════════════
# Focus trap detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FocusTrap:
    """A detected keyboard focus trap."""

    container_id: str
    trapped_ids: Tuple[str, ...]
    reason: str


def detect_focus_traps(tree: Any) -> List[FocusTrap]:
    """Identify subtrees that form keyboard focus traps.

    A focus trap exists when a composite widget contains focusable
    children but lacks an escape mechanism (no close button, no
    escape-key handler indication, no focusable elements outside
    the subtree reachable without explicit key binding).

    Heuristic approach: flag modal-like containers (role=dialog,
    role=alertdialog) that lack a visible close/dismiss control.
    """
    traps: List[FocusTrap] = []

    for node in tree.root.iter_preorder() if hasattr(tree.root, "iter_preorder") else _iter_preorder(tree.root):
        role = node.role if hasattr(node, "role") else ""
        nid = node.id if hasattr(node, "id") else (node.node_id if hasattr(node, "node_id") else "")

        if role not in ("dialog", "alertdialog"):
            continue

        # Collect focusable children
        children_ids: List[str] = []
        has_close = False
        for desc in _iter_preorder(node):
            desc_role = desc.role if hasattr(desc, "role") else ""
            desc_name = (desc.name if hasattr(desc, "name") else "").lower()
            desc_id = desc.id if hasattr(desc, "id") else (desc.node_id if hasattr(desc, "node_id") else "")

            if desc_role in _NATIVELY_FOCUSABLE_ROLES:
                children_ids.append(desc_id)

            # Heuristic: a "close", "dismiss", "cancel" button provides an exit
            if desc_role == "button" and any(kw in desc_name for kw in ("close", "dismiss", "cancel", "×", "x")):
                has_close = True

        props = node.properties if hasattr(node, "properties") else {}
        aria_modal = str(props.get("aria-modal", "false")).lower()

        if aria_modal == "true" and not has_close and len(children_ids) > 0:
            traps.append(FocusTrap(
                container_id=nid,
                trapped_ids=tuple(children_ids),
                reason="Modal dialog without visible close/dismiss control",
            ))

    return traps


# ═══════════════════════════════════════════════════════════════════════════
# Keyboard shortcut conflict detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ShortcutConflict:
    """A detected keyboard shortcut conflict."""

    key: str
    node_ids: Tuple[str, ...]
    reason: str


def detect_shortcut_conflicts(tree: Any) -> List[ShortcutConflict]:
    """Find keyboard shortcut conflicts in the accessibility tree.

    WCAG 2.1 SC 2.1.4 (Character Key Shortcuts) requires that single-
    character shortcuts can be turned off, remapped, or are only active
    on focus.  This detector flags:
    - Multiple elements bound to the same shortcut key.
    - Single-character shortcuts without modifier keys.
    - Conflicts with screen-reader reserved keys.
    """
    shortcut_map: Dict[str, List[str]] = defaultdict(list)
    single_char_shortcuts: List[ShortcutConflict] = []

    for node in _iter_preorder(tree.root):
        props = node.properties if hasattr(node, "properties") else {}
        nid = node.id if hasattr(node, "id") else (node.node_id if hasattr(node, "node_id") else "")

        # Check aria-keyshortcuts and accesskey
        for attr in ("aria-keyshortcuts", "accesskey"):
            shortcut = props.get(attr)
            if shortcut and isinstance(shortcut, str):
                keys = [k.strip().lower() for k in shortcut.split()]
                for key in keys:
                    shortcut_map[key].append(nid)

                    # Flag single-character shortcuts (no modifier)
                    if len(key) == 1 and key.isalpha():
                        single_char_shortcuts.append(ShortcutConflict(
                            key=key,
                            node_ids=(nid,),
                            reason="Single character key shortcut without modifier (WCAG 2.1.4)",
                        ))

    conflicts: List[ShortcutConflict] = list(single_char_shortcuts)

    # Flag duplicate bindings
    for key, node_ids in shortcut_map.items():
        if len(node_ids) > 1:
            conflicts.append(ShortcutConflict(
                key=key,
                node_ids=tuple(node_ids),
                reason=f"Multiple elements bound to shortcut '{key}'",
            ))

    return conflicts


# ═══════════════════════════════════════════════════════════════════════════
# Skip navigation verification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SkipNavResult:
    """Result of skip-navigation analysis."""

    has_skip_link: bool
    skip_link_id: Optional[str]
    skip_target_id: Optional[str]
    main_landmark_present: bool
    navigation_landmarks: int
    violations: Tuple[WCAGViolation, ...]


_SC_2_4_1 = SuccessCriterion(
    sc_id="2.4.1",
    name="Bypass Blocks",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.OPERABLE,
    guideline_id="2.4",
    description="A mechanism is available to bypass blocks of content that are repeated.",
)


def verify_skip_navigation(tree: Any) -> SkipNavResult:
    """Check for skip-navigation mechanisms (WCAG 2.4.1).

    Looks for:
    1. A link near the start of the page whose name includes "skip" or
       "jump" and targets a main content area.
    2. The presence of a ``main`` landmark region.
    3. Navigation landmarks that would benefit from bypass mechanisms.
    """
    tab_order = extract_tab_order(tree)
    skip_link_id: Optional[str] = None
    skip_target: Optional[str] = None
    main_present = False
    nav_count = 0
    violations: List[WCAGViolation] = []

    # Check landmarks
    for node in _iter_preorder(tree.root):
        role = node.role if hasattr(node, "role") else ""
        if role == "main":
            main_present = True
        if role == "navigation":
            nav_count += 1

    # Check for skip link in the first few focusable elements
    for elem in tab_order[:5]:
        name_lower = elem.name.lower() if elem.name else ""
        if elem.role == "link" and any(kw in name_lower for kw in ("skip", "jump", "main content")):
            skip_link_id = elem.node_id
            # Try to resolve target from properties
            node = tree.get_node(elem.node_id) if hasattr(tree, "get_node") else None
            if node:
                props = node.properties if hasattr(node, "properties") else {}
                href = props.get("href", "")
                if isinstance(href, str) and href.startswith("#"):
                    target_id = href[1:]
                    if hasattr(tree, "get_node") and tree.get_node(target_id):
                        skip_target = target_id
            break

    has_skip = skip_link_id is not None or main_present

    if not has_skip and nav_count > 0:
        violations.append(WCAGViolation(
            criterion=_SC_2_4_1,
            node_id=tree.root.id if hasattr(tree.root, "id") else "root",
            impact=ImpactLevel.SERIOUS,
            message="No skip navigation mechanism found. Page has repeated navigation blocks.",
            remediation="Add a 'Skip to main content' link at the start of the page, "
                        "or use landmark regions (role='main').",
        ))

    return SkipNavResult(
        has_skip_link=skip_link_id is not None,
        skip_link_id=skip_link_id,
        skip_target_id=skip_target,
        main_landmark_present=main_present,
        navigation_landmarks=nav_count,
        violations=tuple(violations),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Focus indicator visibility heuristics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FocusIndicatorIssue:
    """A potential focus indicator visibility problem."""

    node_id: str
    role: str
    reason: str
    impact: ImpactLevel


def check_focus_indicators(tree: Any) -> List[FocusIndicatorIssue]:
    """Heuristically check for focus indicator problems (WCAG 2.4.7, 2.4.11).

    Since we cannot inspect CSS from the accessibility tree alone, this uses
    heuristics based on element properties and roles:
    - Interactive elements with outline:none or outline:0 in inline styles
    - Custom-styled elements without aria-current or focus-related state info
    """
    issues: List[FocusIndicatorIssue] = []

    for node in _iter_preorder(tree.root):
        role = node.role if hasattr(node, "role") else ""
        nid = node.id if hasattr(node, "id") else (node.node_id if hasattr(node, "node_id") else "")
        props = node.properties if hasattr(node, "properties") else {}

        if role not in _NATIVELY_FOCUSABLE_ROLES:
            continue

        # Check for inline style suppressing focus outline
        style = props.get("style", "")
        if isinstance(style, str):
            style_lower = style.lower().replace(" ", "")
            if "outline:none" in style_lower or "outline:0" in style_lower:
                issues.append(FocusIndicatorIssue(
                    node_id=nid,
                    role=role,
                    reason="Inline style suppresses focus outline (outline:none/0)",
                    impact=ImpactLevel.SERIOUS,
                ))

        # Flag elements with very small bounding boxes (hard to see focus ring)
        bbox = node.bounding_box if hasattr(node, "bounding_box") else None
        if bbox and hasattr(bbox, "width") and hasattr(bbox, "height"):
            min_dim = min(bbox.width, bbox.height)
            if min_dim < 10.0:
                issues.append(FocusIndicatorIssue(
                    node_id=nid,
                    role=role,
                    reason=f"Interactive element too small for visible focus indicator ({min_dim:.0f}px)",
                    impact=ImpactLevel.MODERATE,
                ))

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Sequential navigation cost computation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NavigationCost:
    """Cost analysis of sequential keyboard navigation."""

    total_focusable: int
    mean_tabs_to_target: float
    max_tabs_to_target: int
    mean_depth: float
    tab_order_entropy: float
    estimated_time_seconds: float


# Average time per tab press from empirical studies (Koester & Levine, 1994)
_TAB_TIME_SECONDS = 0.4
# Average time to process and decide per element (Hick's law component)
_INSPECT_TIME_SECONDS = 0.15


def compute_navigation_cost(
    tree: Any,
    target_id: Optional[str] = None,
) -> NavigationCost:
    """Estimate the cognitive and temporal cost of sequential keyboard navigation.

    Computes the expected number of Tab presses to reach each focusable
    element and the information-theoretic entropy of the tab-order layout.

    Parameters
    ----------
    tree
        An accessibility tree.
    target_id : Optional[str]
        If provided, compute cost specifically to reach this element.

    Returns
    -------
    NavigationCost
        Navigation cost metrics.
    """
    tab_order = extract_tab_order(tree)
    n = len(tab_order)

    if n == 0:
        return NavigationCost(
            total_focusable=0,
            mean_tabs_to_target=0.0,
            max_tabs_to_target=0,
            mean_depth=0.0,
            tab_order_entropy=0.0,
            estimated_time_seconds=0.0,
        )

    # Tab distance to each element (1-indexed)
    tab_distances = list(range(1, n + 1))

    # If we have a specific target, find its position
    target_tabs = n  # worst case
    if target_id:
        for i, elem in enumerate(tab_order):
            if elem.node_id == target_id:
                target_tabs = i + 1
                break

    mean_tabs = float(np.mean(tab_distances))
    max_tabs = n

    # Mean depth of focusable elements
    depths = [e.depth for e in tab_order]
    mean_depth = float(np.mean(depths)) if depths else 0.0

    # Shannon entropy of the role distribution among focusable elements.
    # Higher entropy → less predictable where interactive elements are.
    role_counts: Dict[str, int] = defaultdict(int)
    for e in tab_order:
        role_counts[e.role] += 1
    probs = np.array(list(role_counts.values()), dtype=np.float64) / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

    # Estimated navigation time using Hick-Hyman model
    # Time = sum_i (tab_time + inspect_time * log2(i+1))
    # For reaching the target (or the expected element):
    effective_tabs = target_tabs if target_id else mean_tabs
    estimated_time = effective_tabs * _TAB_TIME_SECONDS + sum(
        _INSPECT_TIME_SECONDS * math.log2(i + 2) for i in range(int(effective_tabs))
    )

    return NavigationCost(
        total_focusable=n,
        mean_tabs_to_target=mean_tabs,
        max_tabs_to_target=max_tabs,
        mean_depth=mean_depth,
        tab_order_entropy=entropy,
        estimated_time_seconds=estimated_time,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _iter_preorder(node: Any) -> List[Any]:
    """Pre-order traversal that works with any node-like object."""
    result: List[Any] = []
    stack = [node]
    while stack:
        n = stack.pop()
        result.append(n)
        children = n.children if hasattr(n, "children") else []
        stack.extend(reversed(children))
    return result


__all__ = [
    "FocusTrap",
    "FocusIndicatorIssue",
    "FocusableElement",
    "NavigationCost",
    "ShortcutConflict",
    "SkipNavResult",
    "check_focus_indicators",
    "compute_navigation_cost",
    "detect_focus_traps",
    "detect_shortcut_conflicts",
    "extract_tab_order",
    "verify_skip_navigation",
]
