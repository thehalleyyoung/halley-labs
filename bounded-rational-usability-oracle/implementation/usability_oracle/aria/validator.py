"""
usability_oracle.aria.validator — ARIA document validation.

Validates HTML documents for common accessibility issues including
heading hierarchy, form labels, image alt text, link text quality,
and structural color contrast.

Reference: WCAG 2.1, WAI-ARIA 1.2, HTML-AAM 1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.aria.parser import AriaHTMLParser, AriaNodeInfo, AriaTree


# ═══════════════════════════════════════════════════════════════════════════
# Validation result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationIssue:
    """A single validation finding in ARIA document validation."""

    severity: str  # "error", "warning", "info"
    rule: str
    message: str
    node_id: Optional[str] = None
    spec_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "node_id": self.node_id,
            "spec_ref": self.spec_ref,
        }


@dataclass
class ValidationReport:
    """Aggregate result of ARIA document validation.

    Attributes:
        valid: ``True`` if no errors were found.
        issues: All validation findings.
        summary: Counts by severity and rule.
    """

    valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.rule] = counts.get(issue.rule, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Non-descriptive link texts (WCAG 2.4.4)
# ═══════════════════════════════════════════════════════════════════════════

_VAGUE_LINK_TEXTS = frozenset({
    "click here", "here", "more", "read more", "learn more",
    "link", "this", "go", "details", "info",
})


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def validate_aria_document(html: str) -> ValidationReport:
    """Validate an HTML document for common accessibility issues.

    Runs all structural checks and returns a comprehensive report.

    Parameters:
        html: Raw HTML string.

    Returns:
        :class:`ValidationReport` with all findings.
    """
    parser = AriaHTMLParser()
    tree = parser.parse_html(html)

    report = ValidationReport()

    report.issues.extend(check_heading_hierarchy(tree))
    report.issues.extend(check_form_labels(tree))
    report.issues.extend(check_image_alt(tree))
    report.issues.extend(check_link_text(tree))
    report.issues.extend(check_color_contrast_structure(tree))

    report.valid = all(i.severity != "error" for i in report.issues)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════════════════

def check_heading_hierarchy(tree: AriaTree) -> List[ValidationIssue]:
    """Check that heading levels (h1-h6) appear in proper order.

    WCAG 2.1 §1.3.1 — Info and Relationships:
    Headings should not skip levels (e.g. h1 → h3 without h2).
    The document should have at least one heading.

    Parameters:
        tree: Parsed :class:`AriaTree`.

    Returns:
        List of :class:`ValidationIssue` for heading problems.
    """
    issues: list[ValidationIssue] = []
    headings: list[tuple[int, AriaNodeInfo]] = []

    for node in tree.node_index.values():
        if node.role == "heading":
            level = _extract_heading_level(node)
            if level is not None:
                headings.append((level, node))

    if not headings:
        issues.append(ValidationIssue(
            severity="warning",
            rule="heading-present",
            message="Document has no headings",
            spec_ref="WCAG 2.1 §1.3.1",
        ))
        return issues

    # Sort by document order (node_id proxy)
    headings.sort(key=lambda h: h[1].node_id)

    # Check for skipped levels
    prev_level = 0
    for level, node in headings:
        if level > prev_level + 1 and prev_level > 0:
            issues.append(ValidationIssue(
                severity="warning",
                rule="heading-order",
                message=(
                    f"Heading level skipped: h{prev_level} → h{level}. "
                    f"Expected h{prev_level + 1}"
                ),
                node_id=node.node_id,
                spec_ref="WCAG 2.1 §1.3.1",
            ))
        prev_level = level

    # Check: first heading should be h1
    first_level = headings[0][0]
    if first_level != 1:
        issues.append(ValidationIssue(
            severity="warning",
            rule="heading-first-h1",
            message=f"First heading is h{first_level}, expected h1",
            node_id=headings[0][1].node_id,
            spec_ref="WCAG 2.1 §1.3.1",
        ))

    return issues


def check_form_labels(tree: AriaTree) -> List[ValidationIssue]:
    """Check that all form inputs have accessible labels.

    WCAG 2.1 §1.3.1, §3.3.2 — Form controls must have labels that
    describe their purpose.

    Parameters:
        tree: Parsed :class:`AriaTree`.

    Returns:
        List of :class:`ValidationIssue` for unlabelled inputs.
    """
    issues: list[ValidationIssue] = []
    input_roles = {"textbox", "searchbox", "combobox", "spinbutton",
                   "checkbox", "radio", "slider", "switch", "listbox"}

    for node in tree.node_index.values():
        if node.role in input_roles:
            if not node.accessible_name or not node.accessible_name.strip():
                issues.append(ValidationIssue(
                    severity="error",
                    rule="form-label",
                    message=(
                        f"Form control (role='{node.role}', tag='{node.tag}') "
                        f"has no accessible label"
                    ),
                    node_id=node.node_id,
                    spec_ref="WCAG 2.1 §1.3.1, §3.3.2",
                ))

    return issues


def check_image_alt(tree: AriaTree) -> List[ValidationIssue]:
    """Check that all images have alt text or role='presentation'.

    WCAG 2.1 §1.1.1 — Non-text Content:
    All ``<img>`` elements must have ``alt`` text, ``aria-label``,
    ``aria-labelledby``, or ``role="presentation"``/``role="none"``.

    Parameters:
        tree: Parsed :class:`AriaTree`.

    Returns:
        List of :class:`ValidationIssue` for images without alt text.
    """
    issues: list[ValidationIssue] = []

    for node in tree.node_index.values():
        if node.tag == "img" and node.role == "img":
            if not node.accessible_name or not node.accessible_name.strip():
                issues.append(ValidationIssue(
                    severity="error",
                    rule="image-alt",
                    message="Image has no alt text or accessible name",
                    node_id=node.node_id,
                    spec_ref="WCAG 2.1 §1.1.1",
                ))

    return issues


def check_link_text(tree: AriaTree) -> List[ValidationIssue]:
    """Check that links have descriptive text.

    WCAG 2.1 §2.4.4 — Link Purpose:
    Link text should describe the link destination. Generic text like
    'click here' or 'read more' is discouraged.

    Parameters:
        tree: Parsed :class:`AriaTree`.

    Returns:
        List of :class:`ValidationIssue` for non-descriptive link text.
    """
    issues: list[ValidationIssue] = []

    for node in tree.node_index.values():
        if node.role != "link":
            continue

        name = (node.accessible_name or "").strip()
        if not name:
            issues.append(ValidationIssue(
                severity="error",
                rule="link-text",
                message="Link has no accessible name",
                node_id=node.node_id,
                spec_ref="WCAG 2.1 §2.4.4",
            ))
            continue

        if name.lower() in _VAGUE_LINK_TEXTS:
            issues.append(ValidationIssue(
                severity="warning",
                rule="link-text-descriptive",
                message=(
                    f"Link text '{name}' is not descriptive. "
                    f"Use text that describes the link destination"
                ),
                node_id=node.node_id,
                spec_ref="WCAG 2.1 §2.4.4",
            ))

    return issues


def check_color_contrast_structure(tree: AriaTree) -> List[ValidationIssue]:
    """Perform structural color-contrast checks (not pixel-level).

    Checks for structural patterns that commonly lead to contrast
    failures, such as text elements without any indication of contrast
    in their properties, or known problematic patterns.

    This is a best-effort structural check since we lack actual
    rendered pixel data.

    Parameters:
        tree: Parsed :class:`AriaTree`.

    Returns:
        List of :class:`ValidationIssue` for structural contrast concerns.
    """
    issues: list[ValidationIssue] = []

    text_roles = {"heading", "paragraph", "listitem", "cell",
                  "link", "button", "tab", "menuitem"}

    for node in tree.node_index.values():
        if node.role not in text_roles:
            continue
        if not node.accessible_name:
            continue

        # Check for known low-contrast CSS patterns in properties
        fg = node.properties.get("color", "")
        bg = node.properties.get("background-color", "")

        if fg and bg:
            # Structural heuristic: warn if foreground and background
            # appear to be very similar (both start with same hex prefix)
            fg_norm = fg.strip().lower()
            bg_norm = bg.strip().lower()
            if fg_norm == bg_norm:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="contrast-structural",
                    message=(
                        f"Text element appears to have identical foreground "
                        f"and background colors: {fg_norm}"
                    ),
                    node_id=node.node_id,
                    spec_ref="WCAG 2.1 §1.4.3",
                ))

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_heading_level(node: AriaNodeInfo) -> Optional[int]:
    """Extract the heading level from a heading node.

    Checks the ``level`` property first (from aria-level), then
    infers from the HTML tag name (h1-h6).
    """
    # aria-level property
    level_prop = node.properties.get("level")
    if level_prop:
        try:
            return int(level_prop)
        except (ValueError, TypeError):
            pass

    # Infer from tag name
    tag = node.tag.lower()
    match = re.match(r"^h([1-6])$", tag)
    if match:
        return int(match.group(1))

    return None
