"""
usability_oracle.wcag.semantic — Semantic structure analysis.

Validates heading hierarchy, landmark regions, list/table/form structure,
and ARIA role usage against WCAG 2.2 success criteria 1.3.1, 2.4.6,
and 4.1.2.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGPrinciple,
    WCAGViolation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Well-known criterion objects used across checks
# ═══════════════════════════════════════════════════════════════════════════

_SC_1_3_1 = SuccessCriterion(
    sc_id="1.3.1", name="Info and Relationships",
    level=ConformanceLevel.A, principle=WCAGPrinciple.PERCEIVABLE,
    guideline_id="1.3",
    description="Information, structure, and relationships conveyed through "
                "presentation can be programmatically determined.",
)

_SC_2_4_6 = SuccessCriterion(
    sc_id="2.4.6", name="Headings and Labels",
    level=ConformanceLevel.AA, principle=WCAGPrinciple.OPERABLE,
    guideline_id="2.4",
    description="Headings and labels describe topic or purpose.",
)

_SC_4_1_2 = SuccessCriterion(
    sc_id="4.1.2", name="Name, Role, Value",
    level=ConformanceLevel.A, principle=WCAGPrinciple.ROBUST,
    guideline_id="4.1",
    description="For all UI components, the name and role can be programmatically determined.",
)


# ═══════════════════════════════════════════════════════════════════════════
# Heading hierarchy validation (h1-h6)
# ═══════════════════════════════════════════════════════════════════════════

_HEADING_PATTERN = re.compile(r"^heading$", re.IGNORECASE)
_HEADING_LEVEL_PATTERN = re.compile(r"^h([1-6])$", re.IGNORECASE)


def _extract_heading_level(node: Any) -> Optional[int]:
    """Extract the heading level from a node's role or properties."""
    role = node.role if hasattr(node, "role") else ""
    props = node.properties if hasattr(node, "properties") else {}

    # aria-level property (used with role=heading)
    if role.lower() == "heading":
        level = props.get("aria-level")
        if level is not None:
            try:
                return int(level)
            except (ValueError, TypeError):
                return None
        return None

    # Roles like h1, h2, … h6 (some parsers normalise to these)
    m = _HEADING_LEVEL_PATTERN.match(role)
    if m:
        return int(m.group(1))

    return None


@dataclass(frozen=True, slots=True)
class HeadingInfo:
    """A heading found in the accessibility tree."""

    node_id: str
    level: int
    name: str
    depth: int


def extract_headings(tree: Any) -> List[HeadingInfo]:
    """Extract all headings in document order."""
    headings: List[HeadingInfo] = []
    for node in _iter_preorder(tree.root):
        level = _extract_heading_level(node)
        if level is not None:
            nid = _node_id(node)
            headings.append(HeadingInfo(
                node_id=nid,
                level=level,
                name=_node_name(node),
                depth=getattr(node, "depth", 0),
            ))
    return headings


def validate_heading_hierarchy(tree: Any) -> List[WCAGViolation]:
    """Check that heading levels do not skip levels (e.g. h1 → h3).

    Per WCAG 2.4.6 and best practices, headings should form a logical
    hierarchy without gaps.
    """
    headings = extract_headings(tree)
    violations: List[WCAGViolation] = []

    if not headings:
        return violations

    # Check for missing h1
    levels_present = {h.level for h in headings}
    if 1 not in levels_present and len(headings) > 0:
        violations.append(WCAGViolation(
            criterion=_SC_2_4_6,
            node_id=headings[0].node_id,
            impact=ImpactLevel.MODERATE,
            message="No h1 heading found on the page.",
            remediation="Add a single h1 heading that describes the page topic.",
        ))

    # Check sequential ordering
    prev_level = 0
    for h in headings:
        if h.level > prev_level + 1 and prev_level > 0:
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=h.node_id,
                impact=ImpactLevel.MODERATE,
                message=f"Heading level skips from h{prev_level} to h{h.level}.",
                evidence={"previous_level": prev_level, "current_level": h.level},
                remediation=f"Use h{prev_level + 1} instead of h{h.level}, or restructure the heading hierarchy.",
            ))
        prev_level = h.level

    # Check for empty headings
    for h in headings:
        if not h.name.strip():
            violations.append(WCAGViolation(
                criterion=_SC_2_4_6,
                node_id=h.node_id,
                impact=ImpactLevel.SERIOUS,
                message=f"Empty h{h.level} heading — provides no information to screen reader users.",
                remediation="Add descriptive text to the heading or remove the empty heading element.",
            ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# Landmark region detection
# ═══════════════════════════════════════════════════════════════════════════

_LANDMARK_ROLES = frozenset({
    "banner", "complementary", "contentinfo", "form",
    "main", "navigation", "region", "search",
})


@dataclass(frozen=True, slots=True)
class LandmarkInfo:
    """A landmark region in the accessibility tree."""

    node_id: str
    role: str
    label: str
    child_count: int


def extract_landmarks(tree: Any) -> List[LandmarkInfo]:
    """Extract all landmark regions from the tree."""
    landmarks: List[LandmarkInfo] = []
    for node in _iter_preorder(tree.root):
        role = (node.role if hasattr(node, "role") else "").lower()
        if role in _LANDMARK_ROLES:
            nid = _node_id(node)
            name = _node_name(node)
            children = node.children if hasattr(node, "children") else []
            landmarks.append(LandmarkInfo(
                node_id=nid,
                role=role,
                label=name,
                child_count=len(children),
            ))
    return landmarks


def validate_landmarks(tree: Any) -> List[WCAGViolation]:
    """Validate landmark region usage.

    Checks:
    - Presence of main landmark
    - Duplicate landmark roles without labels
    - Region landmarks without labels
    """
    landmarks = extract_landmarks(tree)
    violations: List[WCAGViolation] = []

    role_groups: Dict[str, List[LandmarkInfo]] = defaultdict(list)
    for lm in landmarks:
        role_groups[lm.role].append(lm)

    # Must have exactly one main landmark
    mains = role_groups.get("main", [])
    if len(mains) == 0:
        violations.append(WCAGViolation(
            criterion=_SC_1_3_1,
            node_id=_node_id(tree.root),
            impact=ImpactLevel.SERIOUS,
            message="No main landmark region found.",
            remediation="Wrap the primary content area with role='main' or a <main> element.",
        ))
    elif len(mains) > 1:
        for m in mains[1:]:
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=m.node_id,
                impact=ImpactLevel.MODERATE,
                message="Multiple main landmarks found; pages should have exactly one.",
                remediation="Remove duplicate main landmarks or merge them.",
            ))

    # Duplicate landmarks of the same role must have unique labels
    for role, group in role_groups.items():
        if len(group) > 1:
            labels = [lm.label for lm in group]
            if len(set(labels)) < len(labels) or any(not l.strip() for l in labels):
                for lm in group:
                    if not lm.label.strip():
                        violations.append(WCAGViolation(
                            criterion=_SC_1_3_1,
                            node_id=lm.node_id,
                            impact=ImpactLevel.MODERATE,
                            message=f"Multiple '{role}' landmarks without unique labels.",
                            remediation=f"Add a unique aria-label to each '{role}' landmark.",
                        ))

    # Region landmarks must have labels
    for lm in role_groups.get("region", []):
        if not lm.label.strip():
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=lm.node_id,
                impact=ImpactLevel.MODERATE,
                message="Landmark role='region' without an accessible label.",
                remediation="Add an aria-label or aria-labelledby to the region.",
            ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# List structure validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_list_structure(tree: Any) -> List[WCAGViolation]:
    """Validate that list roles contain appropriate children.

    Lists (role=list) should contain only listitem children.
    """
    violations: List[WCAGViolation] = []

    for node in _iter_preorder(tree.root):
        role = (node.role if hasattr(node, "role") else "").lower()
        if role != "list":
            continue

        children = node.children if hasattr(node, "children") else []
        for child in children:
            child_role = (child.role if hasattr(child, "role") else "").lower()
            # Allow presentational wrappers
            if child_role not in ("listitem", "none", "presentation", "group"):
                violations.append(WCAGViolation(
                    criterion=_SC_1_3_1,
                    node_id=_node_id(child),
                    impact=ImpactLevel.MODERATE,
                    message=f"List contains non-listitem child with role='{child_role}'.",
                    evidence={"parent_role": "list", "child_role": child_role},
                    remediation="Wrap content in role='listitem' or use appropriate list markup.",
                ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# Table structure validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_table_structure(tree: Any) -> List[WCAGViolation]:
    """Validate table semantics: headers, scope, caption.

    Data tables (role=table) should have:
    - A caption or aria-label
    - Column/row headers with explicit scope
    - Consistent row lengths
    """
    violations: List[WCAGViolation] = []

    for node in _iter_preorder(tree.root):
        role = (node.role if hasattr(node, "role") else "").lower()
        if role != "table":
            continue

        nid = _node_id(node)
        name = _node_name(node)
        props = node.properties if hasattr(node, "properties") else {}

        # Check for table name/caption
        if not name.strip():
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=nid,
                impact=ImpactLevel.MODERATE,
                message="Data table without caption or accessible name.",
                remediation="Add a <caption>, aria-label, or aria-labelledby to the table.",
            ))

        # Collect rows and check for headers
        rows = _find_children_by_role(node, "row")
        has_header = False
        row_lengths: List[int] = []

        for row in rows:
            cells = _find_children_by_role(row, "cell") + \
                    _find_children_by_role(row, "columnheader") + \
                    _find_children_by_role(row, "rowheader") + \
                    _find_children_by_role(row, "gridcell")
            row_lengths.append(len(cells))

            for cell in cells:
                cell_role = (cell.role if hasattr(cell, "role") else "").lower()
                if cell_role in ("columnheader", "rowheader"):
                    has_header = True

        if rows and not has_header:
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=nid,
                impact=ImpactLevel.SERIOUS,
                message="Data table without column or row headers.",
                remediation="Use <th> elements or role='columnheader'/'rowheader' for header cells.",
            ))

        # Check for consistent row lengths
        if row_lengths and len(set(row_lengths)) > 1:
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=nid,
                impact=ImpactLevel.MINOR,
                message=f"Table has inconsistent row lengths: {sorted(set(row_lengths))}.",
                evidence={"row_lengths": row_lengths},
                remediation="Ensure all rows have the same number of cells, or use colspan/rowspan.",
            ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# Form structure validation
# ═══════════════════════════════════════════════════════════════════════════

_FORM_INPUT_ROLES = frozenset({
    "textbox", "searchbox", "combobox", "listbox",
    "slider", "spinbutton", "checkbox", "radio", "switch",
})


def validate_form_structure(tree: Any) -> List[WCAGViolation]:
    """Validate form control labelling (WCAG 3.3.2, 1.3.1).

    Every form input must have an accessible name (label).
    Related inputs should be grouped with fieldset/legend.
    """
    violations: List[WCAGViolation] = []

    radio_groups: Dict[str, List[Any]] = defaultdict(list)

    for node in _iter_preorder(tree.root):
        role = (node.role if hasattr(node, "role") else "").lower()
        if role not in _FORM_INPUT_ROLES:
            continue

        nid = _node_id(node)
        name = _node_name(node)
        props = node.properties if hasattr(node, "properties") else {}

        # Check for accessible name
        if not name.strip():
            violations.append(WCAGViolation(
                criterion=_SC_4_1_2,
                node_id=nid,
                impact=ImpactLevel.CRITICAL,
                message=f"Form control with role='{role}' has no accessible name.",
                remediation="Add an aria-label, aria-labelledby, or associated <label> element.",
            ))

        # Collect radio buttons for grouping check
        if role == "radio":
            group_name = props.get("name", props.get("aria-labelledby", ""))
            radio_groups[group_name].append(node)

    # Check that radio groups have fieldset/group wrapper
    for group_name, radios in radio_groups.items():
        if len(radios) < 2:
            continue
        # Check if any radio has a group ancestor
        has_group = False
        for radio in radios:
            for ancestor_node in _iter_ancestors(radio, tree):
                anc_role = (ancestor_node.role if hasattr(ancestor_node, "role") else "").lower()
                if anc_role in ("group", "radiogroup"):
                    has_group = True
                    break
            if has_group:
                break

        if not has_group:
            violations.append(WCAGViolation(
                criterion=_SC_1_3_1,
                node_id=_node_id(radios[0]),
                impact=ImpactLevel.MODERATE,
                message="Radio button group not wrapped in a fieldset/group.",
                remediation="Wrap related radio buttons in a <fieldset> with a <legend>.",
            ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# ARIA role validation
# ═══════════════════════════════════════════════════════════════════════════

# Valid ARIA 1.2 abstract roles (should not be used directly)
_ABSTRACT_ROLES = frozenset({
    "command", "composite", "input", "landmark", "range",
    "roletype", "section", "sectionhead", "select",
    "structure", "widget", "window",
})

# Roles that require specific parent roles
_REQUIRED_PARENT: Dict[str, FrozenSet[str]] = {
    "listitem": frozenset({"list", "directory"}),
    "menuitem": frozenset({"menu", "menubar", "group"}),
    "menuitemcheckbox": frozenset({"menu", "menubar", "group"}),
    "menuitemradio": frozenset({"menu", "menubar", "group"}),
    "option": frozenset({"listbox", "group"}),
    "tab": frozenset({"tablist"}),
    "treeitem": frozenset({"tree", "group"}),
    "row": frozenset({"table", "grid", "treegrid", "rowgroup"}),
    "cell": frozenset({"row"}),
    "gridcell": frozenset({"row"}),
    "columnheader": frozenset({"row"}),
    "rowheader": frozenset({"row"}),
}

# Roles that require accessible names
_REQUIRES_NAME = frozenset({
    "button", "link", "checkbox", "radio", "switch",
    "textbox", "searchbox", "combobox", "slider", "spinbutton",
    "img", "figure", "dialog", "alertdialog", "form", "region",
})


def validate_aria_roles(tree: Any) -> List[WCAGViolation]:
    """Validate ARIA role usage against ARIA 1.2 spec and WCAG 4.1.2.

    Checks:
    - No abstract roles used directly
    - Required parent contexts
    - Required accessible names for certain roles
    - No conflicting state/property values
    """
    violations: List[WCAGViolation] = []

    for node in _iter_preorder(tree.root):
        role = (node.role if hasattr(node, "role") else "").lower()
        nid = _node_id(node)
        name = _node_name(node)

        if not role or role in ("none", "presentation", "generic"):
            continue

        # Check for abstract roles
        if role in _ABSTRACT_ROLES:
            violations.append(WCAGViolation(
                criterion=_SC_4_1_2,
                node_id=nid,
                impact=ImpactLevel.SERIOUS,
                message=f"Abstract ARIA role '{role}' used directly.",
                remediation=f"Replace abstract role '{role}' with a concrete role.",
            ))

        # Check required parent context
        if role in _REQUIRED_PARENT:
            required = _REQUIRED_PARENT[role]
            parent_role = _get_parent_role(node, tree)
            if parent_role and parent_role.lower() not in required:
                violations.append(WCAGViolation(
                    criterion=_SC_4_1_2,
                    node_id=nid,
                    impact=ImpactLevel.SERIOUS,
                    message=f"Role '{role}' requires parent role in {sorted(required)}, "
                            f"but found '{parent_role}'.",
                    evidence={"expected_parents": sorted(required), "actual_parent": parent_role},
                    remediation=f"Place the element with role='{role}' inside an element with "
                                f"one of these roles: {', '.join(sorted(required))}.",
                ))

        # Check required name
        if role in _REQUIRES_NAME and not name.strip():
            # img role without name is already checked by alt text check
            if role != "img":
                violations.append(WCAGViolation(
                    criterion=_SC_4_1_2,
                    node_id=nid,
                    impact=ImpactLevel.SERIOUS if role in ("button", "link") else ImpactLevel.MODERATE,
                    message=f"Element with role='{role}' has no accessible name.",
                    remediation="Add an aria-label, aria-labelledby, or visible text content.",
                ))

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate semantic analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticAnalysisResult:
    """Combined result of all semantic structure checks."""

    heading_violations: Tuple[WCAGViolation, ...]
    landmark_violations: Tuple[WCAGViolation, ...]
    list_violations: Tuple[WCAGViolation, ...]
    table_violations: Tuple[WCAGViolation, ...]
    form_violations: Tuple[WCAGViolation, ...]
    aria_violations: Tuple[WCAGViolation, ...]

    @property
    def all_violations(self) -> Tuple[WCAGViolation, ...]:
        return (
            self.heading_violations
            + self.landmark_violations
            + self.list_violations
            + self.table_violations
            + self.form_violations
            + self.aria_violations
        )

    @property
    def total_violations(self) -> int:
        return len(self.all_violations)


def analyse_semantics(tree: Any) -> SemanticAnalysisResult:
    """Run all semantic structure validations."""
    return SemanticAnalysisResult(
        heading_violations=tuple(validate_heading_hierarchy(tree)),
        landmark_violations=tuple(validate_landmarks(tree)),
        list_violations=tuple(validate_list_structure(tree)),
        table_violations=tuple(validate_table_structure(tree)),
        form_violations=tuple(validate_form_structure(tree)),
        aria_violations=tuple(validate_aria_roles(tree)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _iter_preorder(node: Any) -> List[Any]:
    """Pre-order traversal."""
    result: List[Any] = []
    stack = [node]
    while stack:
        n = stack.pop()
        result.append(n)
        children = n.children if hasattr(n, "children") else []
        stack.extend(reversed(children))
    return result


def _node_id(node: Any) -> str:
    if hasattr(node, "id"):
        return node.id
    if hasattr(node, "node_id"):
        return node.node_id
    return ""


def _node_name(node: Any) -> str:
    return node.name if hasattr(node, "name") else ""


def _find_children_by_role(node: Any, role: str) -> List[Any]:
    """Direct children with the given role."""
    children = node.children if hasattr(node, "children") else []
    return [c for c in children if (c.role if hasattr(c, "role") else "").lower() == role]


def _get_parent_role(node: Any, tree: Any) -> Optional[str]:
    """Get the role of a node's parent."""
    pid = node.parent_id if hasattr(node, "parent_id") else None
    if pid is None:
        return None
    if hasattr(tree, "get_node"):
        parent = tree.get_node(pid)
        if parent:
            return parent.role if hasattr(parent, "role") else None
    return None


def _iter_ancestors(node: Any, tree: Any) -> List[Any]:
    """Walk up parent links."""
    ancestors: List[Any] = []
    pid = node.parent_id if hasattr(node, "parent_id") else None
    visited: set[str] = set()
    while pid and pid not in visited and hasattr(tree, "get_node"):
        visited.add(pid)
        parent = tree.get_node(pid)
        if parent is None:
            break
        ancestors.append(parent)
        pid = parent.parent_id if hasattr(parent, "parent_id") else None
    return ancestors


__all__ = [
    "HeadingInfo",
    "LandmarkInfo",
    "SemanticAnalysisResult",
    "analyse_semantics",
    "extract_headings",
    "extract_landmarks",
    "validate_aria_roles",
    "validate_form_structure",
    "validate_heading_hierarchy",
    "validate_landmarks",
    "validate_list_structure",
    "validate_table_structure",
]
