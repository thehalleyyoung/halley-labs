"""
usability_oracle.wcag.evaluator — WCAG 2.2 conformance evaluator.

Evaluates an accessibility tree against WCAG success criteria using the
specialised checkers in :mod:`~usability_oracle.wcag.contrast`,
:mod:`~usability_oracle.wcag.keyboard`, and
:mod:`~usability_oracle.wcag.semantic`.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)
from usability_oracle.wcag.parser import WCAGXMLParser
from usability_oracle.wcag.contrast import (
    Color,
    check_contrast,
    contrast_ratio,
    relative_luminance,
)
from usability_oracle.wcag.keyboard import (
    detect_focus_traps,
    detect_shortcut_conflicts,
    extract_tab_order,
    verify_skip_navigation,
)
from usability_oracle.wcag.semantic import (
    analyse_semantics,
    validate_aria_roles,
    validate_form_structure,
    validate_heading_hierarchy,
    validate_landmarks,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Well-known criterion objects (used by individual checkers)
# ═══════════════════════════════════════════════════════════════════════════

def _sc(sc_id: str, name: str, level: ConformanceLevel,
        principle: WCAGPrinciple, gl_id: str, desc: str = "") -> SuccessCriterion:
    return SuccessCriterion(
        sc_id=sc_id, name=name, level=level, principle=principle,
        guideline_id=gl_id, description=desc,
    )


_SC_1_1_1 = _sc("1.1.1", "Non-text Content", ConformanceLevel.A,
                 WCAGPrinciple.PERCEIVABLE, "1.1",
                 "All non-text content has a text alternative.")
_SC_1_3_1 = _sc("1.3.1", "Info and Relationships", ConformanceLevel.A,
                 WCAGPrinciple.PERCEIVABLE, "1.3",
                 "Information and relationships can be programmatically determined.")
_SC_1_4_3 = _sc("1.4.3", "Contrast (Minimum)", ConformanceLevel.AA,
                 WCAGPrinciple.PERCEIVABLE, "1.4",
                 "Text has a contrast ratio of at least 4.5:1.")
_SC_2_1_1 = _sc("2.1.1", "Keyboard", ConformanceLevel.A,
                 WCAGPrinciple.OPERABLE, "2.1",
                 "All functionality is operable through a keyboard.")
_SC_2_4_1 = _sc("2.4.1", "Bypass Blocks", ConformanceLevel.A,
                 WCAGPrinciple.OPERABLE, "2.4",
                 "A mechanism is available to bypass repeated content blocks.")
_SC_2_4_2 = _sc("2.4.2", "Page Titled", ConformanceLevel.A,
                 WCAGPrinciple.OPERABLE, "2.4",
                 "Web pages have titles that describe topic or purpose.")
_SC_2_4_6 = _sc("2.4.6", "Headings and Labels", ConformanceLevel.AA,
                 WCAGPrinciple.OPERABLE, "2.4",
                 "Headings and labels describe topic or purpose.")
_SC_3_1_1 = _sc("3.1.1", "Language of Page", ConformanceLevel.A,
                 WCAGPrinciple.UNDERSTANDABLE, "3.1",
                 "The default human language of each page can be programmatically determined.")
_SC_3_3_2 = _sc("3.3.2", "Labels or Instructions", ConformanceLevel.A,
                 WCAGPrinciple.UNDERSTANDABLE, "3.3",
                 "Labels or instructions are provided when content requires user input.")
_SC_4_1_1 = _sc("4.1.1", "Parsing", ConformanceLevel.A,
                 WCAGPrinciple.ROBUST, "4.1",
                 "Markup elements have complete start and end tags and are properly nested.")
_SC_4_1_2 = _sc("4.1.2", "Name, Role, Value", ConformanceLevel.A,
                 WCAGPrinciple.ROBUST, "4.1",
                 "Name and role can be programmatically determined for all UI components.")

# Map of sc_id to the canonical SuccessCriterion object used by checkers
_SC_LOOKUP: Dict[str, SuccessCriterion] = {
    sc.sc_id: sc for sc in [
        _SC_1_1_1, _SC_1_3_1, _SC_1_4_3, _SC_2_1_1, _SC_2_4_1,
        _SC_2_4_2, _SC_2_4_6, _SC_3_1_1, _SC_3_3_2, _SC_4_1_1, _SC_4_1_2,
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Individual criterion checkers
# ═══════════════════════════════════════════════════════════════════════════

# Type alias for a checker function
_CheckerFn = Callable[[Any], List[WCAGViolation]]


def _check_1_1_1_non_text_content(tree: Any) -> List[WCAGViolation]:
    """SC 1.1.1 — Non-text Content: check that images have alt text."""
    violations: List[WCAGViolation] = []

    for node in _iter_preorder(tree.root):
        role = _role(node)
        nid = _nid(node)
        name = _name(node)
        props = _props(node)

        # Image roles
        if role in ("img", "image"):
            if not name.strip():
                # Check aria-label, aria-labelledby
                aria_label = props.get("aria-label", "")
                aria_labelledby = props.get("aria-labelledby", "")
                alt = props.get("alt", "")
                if not any(str(v).strip() for v in [aria_label, aria_labelledby, alt]):
                    violations.append(WCAGViolation(
                        criterion=_SC_1_1_1,
                        node_id=nid,
                        impact=ImpactLevel.CRITICAL,
                        message="Image has no text alternative (alt text, aria-label, or aria-labelledby).",
                        remediation="Add descriptive alt text, or use aria-label for the image.",
                    ))

        # SVG without accessible name
        if role == "graphics-document" and not name.strip():
            violations.append(WCAGViolation(
                criterion=_SC_1_1_1,
                node_id=nid,
                impact=ImpactLevel.SERIOUS,
                message="SVG graphic has no accessible name.",
                remediation="Add a <title> element inside the SVG or use aria-label.",
            ))

        # Icon fonts / decorative elements with content but no text alternative
        if role in ("presentation", "none"):
            # Decorative — acceptable, skip
            continue

    return violations


def _check_1_3_1_info_relationships(tree: Any) -> List[WCAGViolation]:
    """SC 1.3.1 — Info and Relationships: validate semantic structure."""
    result = analyse_semantics(tree)
    # Collect heading, landmark, list, table violations
    violations: List[WCAGViolation] = []
    violations.extend(result.heading_violations)
    violations.extend(result.landmark_violations)
    violations.extend(result.list_violations)
    violations.extend(result.table_violations)
    return violations


def _check_1_4_3_contrast(tree: Any) -> List[WCAGViolation]:
    """SC 1.4.3 — Contrast (Minimum): check colour contrast ratios.

    Parses foreground/background colour from node properties.
    """
    violations: List[WCAGViolation] = []

    for node in _iter_preorder(tree.root):
        role = _role(node)
        props = _props(node)
        nid = _nid(node)
        name = _name(node)

        # Only check text-bearing nodes
        if not name.strip() and role not in ("text", "statictext", "label"):
            continue

        fg_str = props.get("color", props.get("foreground-color", ""))
        bg_str = props.get("background-color", props.get("bgcolor", ""))

        if not fg_str or not bg_str:
            continue

        try:
            fg = _parse_color(str(fg_str))
            bg = _parse_color(str(bg_str))
        except (ValueError, TypeError):
            continue

        font_size = 16.0
        try:
            font_size = float(props.get("font-size", 16.0))
        except (ValueError, TypeError):
            pass

        is_bold = str(props.get("font-weight", "normal")).lower() in ("bold", "700", "800", "900")

        result = check_contrast(fg, bg, font_size, is_bold)

        # Determine applicable threshold
        from usability_oracle.wcag.contrast import is_large_text
        large = is_large_text(font_size, is_bold)
        threshold = 3.0 if large else 4.5

        if result.ratio < threshold:
            violations.append(WCAGViolation(
                criterion=_SC_1_4_3,
                node_id=nid,
                impact=ImpactLevel.SERIOUS,
                message=f"Insufficient contrast ratio {result.ratio:.2f}:1 "
                        f"(need {threshold:.1f}:1 for {'large' if large else 'normal'} text).",
                evidence={
                    "contrast_ratio": round(result.ratio, 3),
                    "required_ratio": threshold,
                    "foreground": fg.to_hex(),
                    "background": bg.to_hex(),
                    "font_size_px": font_size,
                    "is_bold": is_bold,
                    "is_large_text": large,
                },
                remediation=f"Increase contrast to at least {threshold:.1f}:1. "
                            f"Current: {result.ratio:.2f}:1.",
            ))

    return violations


def _check_2_1_1_keyboard(tree: Any) -> List[WCAGViolation]:
    """SC 2.1.1 — Keyboard: check keyboard accessibility."""
    violations: List[WCAGViolation] = []

    # Check that all interactive elements are keyboard-accessible
    for node in _iter_preorder(tree.root):
        role = _role(node)
        props = _props(node)
        nid = _nid(node)

        if not _is_interactive(role):
            continue

        # Check if element is in the tab order
        tabindex = props.get("tabindex")
        if tabindex is not None:
            try:
                if int(tabindex) < 0:
                    # Removed from tab order
                    violations.append(WCAGViolation(
                        criterion=_SC_2_1_1,
                        node_id=nid,
                        impact=ImpactLevel.CRITICAL,
                        message=f"Interactive element (role='{role}') removed from keyboard navigation "
                                f"with tabindex={tabindex}.",
                        evidence={"tabindex": int(tabindex), "role": role},
                        remediation="Remove the negative tabindex or provide an alternative keyboard mechanism.",
                    ))
            except (ValueError, TypeError):
                pass

        # Check for click handlers without keyboard equivalents
        has_click = bool(props.get("onclick") or props.get("data-onclick"))
        has_key = bool(props.get("onkeydown") or props.get("onkeyup") or props.get("onkeypress"))
        if has_click and not has_key and role not in ("button", "link", "checkbox", "radio"):
            violations.append(WCAGViolation(
                criterion=_SC_2_1_1,
                node_id=nid,
                impact=ImpactLevel.SERIOUS,
                message=f"Element (role='{role}') has click handler but no keyboard event handler.",
                remediation="Add keyboard event handling (onkeydown/onkeyup) or use a natively "
                            "focusable element (button, link).",
            ))

    # Focus traps
    traps = detect_focus_traps(tree)
    for trap in traps:
        violations.append(WCAGViolation(
            criterion=_SC_2_1_1,
            node_id=trap.container_id,
            impact=ImpactLevel.CRITICAL,
            message=f"Keyboard focus trap detected: {trap.reason}",
            evidence={"trapped_element_count": len(trap.trapped_ids)},
            remediation="Ensure users can escape the component with Escape key or Tab.",
        ))

    return violations


def _check_2_4_1_bypass_blocks(tree: Any) -> List[WCAGViolation]:
    """SC 2.4.1 — Bypass Blocks: verify skip navigation."""
    result = verify_skip_navigation(tree)
    return list(result.violations)


def _check_2_4_2_page_titled(tree: Any) -> List[WCAGViolation]:
    """SC 2.4.2 — Page Titled: check for page title."""
    violations: List[WCAGViolation] = []
    root = tree.root
    root_name = _name(root)
    root_props = _props(root)

    # Check root node for title
    title = root_props.get("title", root_props.get("document-title", ""))
    if not title and not root_name.strip():
        # Search for a document/title node
        found_title = False
        for node in _iter_preorder(root):
            role = _role(node)
            if role in ("document", "application"):
                node_name = _name(node)
                if node_name.strip():
                    found_title = True
                    break
                node_title = _props(node).get("title", "")
                if str(node_title).strip():
                    found_title = True
                    break

        if not found_title:
            violations.append(WCAGViolation(
                criterion=_SC_2_4_2,
                node_id=_nid(root),
                impact=ImpactLevel.SERIOUS,
                message="Page has no title. Users and screen readers cannot identify the page purpose.",
                remediation="Add a descriptive <title> element to the HTML document.",
            ))

    return violations


def _check_2_4_6_headings_labels(tree: Any) -> List[WCAGViolation]:
    """SC 2.4.6 — Headings and Labels."""
    return validate_heading_hierarchy(tree)


def _check_3_1_1_language(tree: Any) -> List[WCAGViolation]:
    """SC 3.1.1 — Language of Page: check for lang attribute."""
    violations: List[WCAGViolation] = []
    root = tree.root
    props = _props(root)

    lang = props.get("lang", props.get("xml:lang", props.get("language", "")))

    # Also check document-level node
    if not lang:
        for node in _iter_preorder(root):
            role = _role(node)
            if role in ("document", "application", "webpage"):
                node_props = _props(node)
                lang = node_props.get("lang", node_props.get("xml:lang", ""))
                if lang:
                    break

    if not str(lang).strip():
        violations.append(WCAGViolation(
            criterion=_SC_3_1_1,
            node_id=_nid(root),
            impact=ImpactLevel.SERIOUS,
            message="Page language not specified. Screen readers cannot determine pronunciation.",
            remediation="Add a lang attribute to the root <html> element (e.g. lang='en').",
        ))
    elif not _is_valid_lang_code(str(lang)):
        violations.append(WCAGViolation(
            criterion=_SC_3_1_1,
            node_id=_nid(root),
            impact=ImpactLevel.MODERATE,
            message=f"Invalid language code: '{lang}'.",
            evidence={"lang": str(lang)},
            remediation="Use a valid BCP 47 language tag (e.g. 'en', 'en-US', 'fr').",
        ))

    return violations


def _check_3_3_2_labels(tree: Any) -> List[WCAGViolation]:
    """SC 3.3.2 — Labels or Instructions: check form input labels."""
    return validate_form_structure(tree)


def _check_4_1_1_parsing(tree: Any) -> List[WCAGViolation]:
    """SC 4.1.1 — Parsing: check for duplicate IDs and structure issues.

    Note: In WCAG 2.2, this criterion is always satisfied for HTML/XML
    parsed by a conforming user agent.  We check for duplicate IDs in the
    accessibility tree, which indicates likely markup problems.
    """
    violations: List[WCAGViolation] = []
    seen_ids: Dict[str, str] = {}  # id → first node_id

    for node in _iter_preorder(tree.root):
        nid = _nid(node)
        props = _props(node)

        # Check for duplicate element IDs (different from tree node IDs)
        element_id = props.get("id", props.get("element-id", ""))
        if element_id and isinstance(element_id, str) and element_id.strip():
            eid = element_id.strip()
            if eid in seen_ids:
                violations.append(WCAGViolation(
                    criterion=_SC_4_1_1,
                    node_id=nid,
                    impact=ImpactLevel.SERIOUS,
                    message=f"Duplicate element ID '{eid}' (first seen on node {seen_ids[eid]}).",
                    evidence={"duplicate_id": eid, "first_node": seen_ids[eid]},
                    remediation="Ensure all element IDs are unique within the document.",
                ))
            else:
                seen_ids[eid] = nid

    return violations


def _check_4_1_2_name_role_value(tree: Any) -> List[WCAGViolation]:
    """SC 4.1.2 — Name, Role, Value."""
    return validate_aria_roles(tree)


# ═══════════════════════════════════════════════════════════════════════════
# Checker registry
# ═══════════════════════════════════════════════════════════════════════════

_CHECKER_REGISTRY: Dict[str, _CheckerFn] = {
    "1.1.1": _check_1_1_1_non_text_content,
    "1.3.1": _check_1_3_1_info_relationships,
    "1.4.3": _check_1_4_3_contrast,
    "2.1.1": _check_2_1_1_keyboard,
    "2.4.1": _check_2_4_1_bypass_blocks,
    "2.4.2": _check_2_4_2_page_titled,
    "2.4.6": _check_2_4_6_headings_labels,
    "3.1.1": _check_3_1_1_language,
    "3.3.2": _check_3_3_2_labels,
    "4.1.1": _check_4_1_1_parsing,
    "4.1.2": _check_4_1_2_name_role_value,
}


# ═══════════════════════════════════════════════════════════════════════════
# WCAGConformanceEvaluator
# ═══════════════════════════════════════════════════════════════════════════

class WCAGConformanceEvaluator:
    """Evaluate an accessibility tree against WCAG 2.2 success criteria.

    Implements the :class:`~usability_oracle.wcag.protocols.WCAGEvaluator`
    protocol and the core-level
    :class:`~usability_oracle.core.protocols.WCAGEvaluator` protocol.

    Parameters
    ----------
    parser : WCAGXMLParser, optional
        Parser providing the WCAG criteria catalogue. If *None*, uses
        the built-in catalogue.
    custom_checkers : dict, optional
        Additional or override checker functions keyed by SC id.
    """

    def __init__(
        self,
        parser: Optional[WCAGXMLParser] = None,
        custom_checkers: Optional[Dict[str, _CheckerFn]] = None,
    ) -> None:
        self._parser = parser or WCAGXMLParser()
        self._criteria = {sc.sc_id: sc for sc in self._parser.load_criteria()}
        self._checkers: Dict[str, _CheckerFn] = dict(_CHECKER_REGISTRY)
        if custom_checkers:
            self._checkers.update(custom_checkers)

    # -- WCAGEvaluator protocol ---------------------------------------------

    def evaluate(
        self,
        tree: Any,
        level: ConformanceLevel | str,
        *,
        criteria_ids: Optional[Sequence[str]] = None,
    ) -> WCAGResult:
        """Run conformance evaluation against the given tree.

        Parameters
        ----------
        tree
            An accessibility tree (or duck-typed equivalent).
        level : ConformanceLevel or str
            Target conformance level.
        criteria_ids : Sequence[str], optional
            If given, only evaluate these criteria.

        Returns
        -------
        WCAGResult
        """
        if isinstance(level, str):
            level = ConformanceLevel(level)

        # Determine which criteria to evaluate
        if criteria_ids:
            target_criteria = [
                self._criteria[cid] for cid in criteria_ids
                if cid in self._criteria
            ]
        else:
            target_criteria = [
                sc for sc in self._criteria.values()
                if sc.level <= level
            ]

        all_violations: List[WCAGViolation] = []
        tested = 0
        passed = 0

        for sc in target_criteria:
            checker = self._checkers.get(sc.sc_id)
            if checker is None:
                # No automated checker for this criterion
                continue

            tested += 1
            try:
                violations = checker(tree)
            except Exception as exc:
                logger.warning("Checker for SC %s raised %s: %s", sc.sc_id, type(exc).__name__, exc)
                violations = []

            if violations:
                all_violations.extend(violations)
            else:
                passed += 1

        # Build page URL from tree metadata
        page_url = ""
        if hasattr(tree, "metadata") and isinstance(tree.metadata, dict):
            page_url = tree.metadata.get("url", tree.metadata.get("page_url", ""))

        return WCAGResult(
            violations=tuple(all_violations),
            target_level=level,
            criteria_tested=tested,
            criteria_passed=passed,
            page_url=page_url,
            metadata={
                "evaluator": "WCAGConformanceEvaluator",
                "version": "2.2",
                "checkers_available": sorted(self._checkers.keys()),
            },
        )

    def check_criterion(
        self,
        tree: Any,
        criterion: SuccessCriterion | str,
    ) -> Sequence[WCAGViolation]:
        """Check a single criterion.

        Parameters
        ----------
        tree
            Accessibility tree.
        criterion : SuccessCriterion or str
            Criterion object or dotted id.

        Returns
        -------
        Sequence[WCAGViolation]
        """
        if isinstance(criterion, str):
            sc_id = criterion
        else:
            sc_id = criterion.sc_id

        checker = self._checkers.get(sc_id)
        if checker is None:
            return []

        try:
            return checker(tree)
        except Exception as exc:
            logger.warning("Checker for SC %s raised %s: %s", sc_id, type(exc).__name__, exc)
            return []

    # -- aggregation helpers ------------------------------------------------

    def violations_by_level(
        self, result: WCAGResult
    ) -> Dict[ConformanceLevel, List[WCAGViolation]]:
        """Group violations by conformance level."""
        groups: Dict[ConformanceLevel, List[WCAGViolation]] = {
            lvl: [] for lvl in ConformanceLevel
        }
        for v in result.violations:
            groups[v.conformance_level].append(v)
        return groups

    def violations_by_principle(
        self, result: WCAGResult
    ) -> Dict[WCAGPrinciple, List[WCAGViolation]]:
        """Group violations by WCAG principle."""
        groups: Dict[WCAGPrinciple, List[WCAGViolation]] = {
            p: [] for p in WCAGPrinciple
        }
        for v in result.violations:
            groups[v.criterion.principle].append(v)
        return groups

    def violations_by_criterion(
        self, result: WCAGResult
    ) -> Dict[str, List[WCAGViolation]]:
        """Group violations by success criterion id."""
        groups: Dict[str, List[WCAGViolation]] = defaultdict(list)
        for v in result.violations:
            groups[v.sc_id].append(v)
        return dict(groups)

    @property
    def supported_criteria(self) -> List[str]:
        """Return the list of SC ids that have automated checkers."""
        return sorted(self._checkers.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _iter_preorder(node: Any) -> List[Any]:
    result: List[Any] = []
    stack = [node]
    while stack:
        n = stack.pop()
        result.append(n)
        children = n.children if hasattr(n, "children") else []
        stack.extend(reversed(children))
    return result


def _nid(node: Any) -> str:
    if hasattr(node, "id"):
        return node.id
    if hasattr(node, "node_id"):
        return node.node_id
    return ""


def _role(node: Any) -> str:
    return (node.role if hasattr(node, "role") else "").lower()


def _name(node: Any) -> str:
    return node.name if hasattr(node, "name") else ""


def _props(node: Any) -> Dict[str, Any]:
    return node.properties if hasattr(node, "properties") else {}


_INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "searchbox", "combobox", "listbox",
    "slider", "spinbutton", "checkbox", "radio", "switch", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "treeitem", "option",
    "gridcell", "columnheader", "rowheader",
})


def _is_interactive(role: str) -> bool:
    return role.lower() in _INTERACTIVE_ROLES


def _parse_color(s: str) -> Color:
    """Parse a colour string into a Color object.

    Supports hex (#RRGGBB, #RGB), rgb(r,g,b), and named colours (limited).
    """
    s = s.strip()
    if s.startswith("#"):
        return Color.from_hex(s)

    # rgb(r, g, b) or rgba(r, g, b, a)
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([0-9.]+)\s*)?\)", s)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        a = float(m.group(4)) if m.group(4) else 1.0
        return Color(r, g, b, a)

    # Short hex (#RGB)
    if len(s) == 4 and s.startswith("#"):
        return Color.from_hex(f"#{s[1]*2}{s[2]*2}{s[3]*2}")

    raise ValueError(f"Cannot parse colour: {s!r}")


# BCP 47 language code pattern
_LANG_PATTERN = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$")


def _is_valid_lang_code(lang: str) -> bool:
    """Check if a string looks like a valid BCP 47 language tag."""
    return bool(_LANG_PATTERN.match(lang.strip()))


__all__ = [
    "WCAGConformanceEvaluator",
]
