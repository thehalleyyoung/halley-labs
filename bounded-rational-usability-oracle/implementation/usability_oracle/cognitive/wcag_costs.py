"""WCAG 2.1 guideline mapping for cognitive cost penalties.

Extends the base WCAG-to-cognitive-cost mapping with:

1. **WCAG success criteria detection** from accessibility tree structure —
   identifies violations of 1.3.1 (Info and Relationships), 2.4.6 (Headings
   and Labels), and 4.1.2 (Name, Role, Value) directly from parsed trees.

2. **Cost penalties** for structural issues: missing labels, broken heading
   hierarchy, and missing ARIA states, calibrated to real WCAG compliance
   patterns rather than synthetic bottleneck categories.

The detector operates on an :class:`AccessibilityTree` (not raw HTML),
so it works with both the base parser and the enhanced
:class:`RealHTMLParser`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from usability_oracle.accessibility.models import AccessibilityNode, AccessibilityTree
from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)
from usability_oracle.wcag.mapping import (
    CognitiveCostDelta,
    compute_cost_delta,
    compute_violation_cost,
)

# ═══════════════════════════════════════════════════════════════════════════
# Pre-defined WCAG 2.1 success criteria used by this module
# ═══════════════════════════════════════════════════════════════════════════

SC_1_3_1 = SuccessCriterion(
    sc_id="1.3.1",
    name="Info and Relationships",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.PERCEIVABLE,
    guideline_id="1.3",
    description=(
        "Information, structure, and relationships conveyed through "
        "presentation can be programmatically determined."
    ),
)

SC_2_4_6 = SuccessCriterion(
    sc_id="2.4.6",
    name="Headings and Labels",
    level=ConformanceLevel.AA,
    principle=WCAGPrinciple.OPERABLE,
    guideline_id="2.4",
    description="Headings and labels describe topic or purpose.",
)

SC_4_1_2 = SuccessCriterion(
    sc_id="4.1.2",
    name="Name, Role, Value",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.ROBUST,
    guideline_id="4.1",
    description=(
        "For all user interface components, the name and role can be "
        "programmatically determined; states, properties, and values "
        "can be programmatically set."
    ),
)

SC_2_4_1 = SuccessCriterion(
    sc_id="2.4.1",
    name="Bypass Blocks",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.OPERABLE,
    guideline_id="2.4",
    description="A mechanism is available to bypass blocks of content.",
)

SC_1_1_1 = SuccessCriterion(
    sc_id="1.1.1",
    name="Non-text Content",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.PERCEIVABLE,
    guideline_id="1.1",
    description="All non-text content has a text alternative.",
)

SC_2_1_1 = SuccessCriterion(
    sc_id="2.1.1",
    name="Keyboard",
    level=ConformanceLevel.A,
    principle=WCAGPrinciple.OPERABLE,
    guideline_id="2.1",
    description="All functionality is operable through a keyboard interface.",
)

# Cognitive cost penalties (bits) for structural WCAG issues.
# These complement the base _CRITERION_OVERRIDES in wcag/mapping.py.
STRUCTURAL_COST_PENALTIES: Dict[str, float] = {
    "missing_label": 3.5,           # No accessible name → user must guess
    "missing_fieldset_group": 1.5,  # Ungrouped radios/checkboxes → context loss
    "heading_skip": 2.0,            # Skipped heading level → navigation confusion
    "missing_aria_expanded": 2.5,   # Interactive control without state → uncertainty
    "missing_aria_selected": 2.0,   # Tab without selected state
    "missing_role": 3.0,            # Interactive element without ARIA role
    "empty_heading": 1.5,           # Heading with no text content
    "duplicate_id": 1.0,            # Duplicate IDs → broken references
}


# ═══════════════════════════════════════════════════════════════════════════
# Structural WCAG violation detector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WCAGStructuralDetector:
    """Detect WCAG violations from accessibility tree structure.

    Checks for:
    - **1.3.1** Missing label associations, broken heading hierarchy,
      ungrouped form controls.
    - **2.4.6** Empty or non-descriptive headings and labels.
    - **4.1.2** Interactive elements missing name, role, or ARIA states.
    """

    check_labels: bool = True
    check_headings: bool = True
    check_aria_states: bool = True
    check_form_groups: bool = True

    def detect(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        """Run all structural checks on the tree."""
        violations: List[WCAGViolation] = []

        if self.check_labels:
            violations.extend(self._check_missing_labels(tree))
        if self.check_headings:
            violations.extend(self._check_heading_hierarchy(tree))
            violations.extend(self._check_empty_headings(tree))
        if self.check_aria_states:
            violations.extend(self._check_missing_aria_states(tree))
            violations.extend(self._check_missing_roles(tree))
        if self.check_form_groups:
            violations.extend(self._check_form_grouping(tree))

        return violations

    def detect_to_result(
        self, tree: AccessibilityTree, page_url: str = ""
    ) -> WCAGResult:
        """Run checks and return a :class:`WCAGResult`."""
        violations = self.detect(tree)
        # Count criteria tested (the ones we check)
        criteria_ids = {"1.3.1", "2.4.6", "4.1.2"}
        violated_ids = {v.sc_id for v in violations}
        passed = len(criteria_ids - violated_ids)
        return WCAGResult(
            violations=tuple(violations),
            target_level=ConformanceLevel.AA,
            criteria_tested=len(criteria_ids),
            criteria_passed=passed,
            page_url=page_url,
        )

    # ── 1.3.1 / 4.1.2: Missing labels ────────────────────────────────────

    def _check_missing_labels(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        form_roles = {"textbox", "checkbox", "radio", "combobox",
                      "spinbutton", "searchbox", "slider", "listbox"}
        for node in tree.node_index.values():
            if node.role in form_roles and not node.name:
                violations.append(WCAGViolation(
                    criterion=SC_4_1_2,
                    node_id=node.id,
                    impact=ImpactLevel.SERIOUS,
                    message=f"Form control ({node.role}) has no accessible name",
                    remediation="Add a <label> element or aria-label attribute",
                ))
        return violations

    # ── 1.3.1: Heading hierarchy ──────────────────────────────────────────

    def _check_heading_hierarchy(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        headings = []
        for node in tree.root.iter_preorder():
            level = node.properties.get("level")
            if level is not None and node.role == "heading":
                headings.append((int(level), node))

        prev_level = 0
        for level, node in headings:
            if level > prev_level + 1 and prev_level > 0:
                violations.append(WCAGViolation(
                    criterion=SC_1_3_1,
                    node_id=node.id,
                    impact=ImpactLevel.MODERATE,
                    message=(
                        f"Heading level skipped: h{prev_level} → h{level} "
                        f"('{node.name[:40]}')"
                    ),
                    remediation=f"Use h{prev_level + 1} instead of h{level}",
                    evidence={"expected": prev_level + 1, "actual": level},
                ))
            prev_level = level
        return violations

    # ── 2.4.6: Empty headings ────────────────────────────────────────────

    def _check_empty_headings(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        for node in tree.node_index.values():
            if node.role == "heading" and not node.name.strip():
                violations.append(WCAGViolation(
                    criterion=SC_2_4_6,
                    node_id=node.id,
                    impact=ImpactLevel.MODERATE,
                    message="Heading element has no text content",
                    remediation="Add descriptive text to the heading",
                ))
        return violations

    # ── 4.1.2: Missing ARIA states ───────────────────────────────────────

    def _check_missing_aria_states(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        for node in tree.node_index.values():
            # Accordion buttons should have aria-expanded
            if (node.properties.get("component_type") == "accordion"
                    or "accordion" in node.properties.get("class", "")):
                for desc in node.get_descendants():
                    if (desc.role == "button"
                            and desc.properties.get("aria-expanded") is None
                            and "section-button" in desc.properties.get("class", "")):
                        violations.append(WCAGViolation(
                            criterion=SC_4_1_2,
                            node_id=desc.id,
                            impact=ImpactLevel.SERIOUS,
                            message="Accordion button missing aria-expanded state",
                            remediation="Add aria-expanded='false' or 'true'",
                        ))

            # Tabs should have aria-selected
            if node.role == "tab" and node.properties.get("aria-selected") is None:
                violations.append(WCAGViolation(
                    criterion=SC_4_1_2,
                    node_id=node.id,
                    impact=ImpactLevel.SERIOUS,
                    message="Tab element missing aria-selected state",
                    remediation="Add aria-selected='true' or 'false'",
                ))

        return violations

    # ── 4.1.2: Missing roles ─────────────────────────────────────────────

    def _check_missing_roles(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        for node in tree.node_index.values():
            tag = node.properties.get("tag", "")
            # div/span used as interactive but no role
            if tag in ("div", "span"):
                tabindex = node.properties.get("tabindex")
                has_onclick = "onclick" in node.properties.get("class", "")
                if tabindex is not None and node.role == "generic":
                    violations.append(WCAGViolation(
                        criterion=SC_4_1_2,
                        node_id=node.id,
                        impact=ImpactLevel.SERIOUS,
                        message=f"Interactive {tag} with tabindex but no ARIA role",
                        remediation="Add an appropriate role attribute",
                    ))
        return violations

    # ── 1.3.1: Form grouping ─────────────────────────────────────────────

    def _check_form_grouping(self, tree: AccessibilityTree) -> List[WCAGViolation]:
        violations = []
        # Find radio/checkbox groups not inside a fieldset
        group_roles = {"radio", "checkbox"}
        grouped_nodes: set[str] = set()

        for node in tree.node_index.values():
            if (node.role == "group"
                    and node.properties.get("tag") == "fieldset"):
                for desc in node.get_descendants():
                    if desc.role in group_roles:
                        grouped_nodes.add(desc.id)

        # Check for ungrouped radios/checkboxes
        name_buckets: Dict[str, List[AccessibilityNode]] = defaultdict(list)
        for node in tree.node_index.values():
            if node.role in group_roles:
                name_key = node.properties.get("name", node.name)
                name_buckets[name_key].append(node)

        for name, nodes in name_buckets.items():
            if len(nodes) > 1:
                ungrouped = [n for n in nodes if n.id not in grouped_nodes]
                if ungrouped:
                    violations.append(WCAGViolation(
                        criterion=SC_1_3_1,
                        node_id=ungrouped[0].id,
                        impact=ImpactLevel.MODERATE,
                        message=(
                            f"{len(ungrouped)} {ungrouped[0].role} controls "
                            f"with name '{name[:30]}' not grouped in fieldset"
                        ),
                        remediation="Wrap related controls in a <fieldset> with <legend>",
                    ))

        return violations


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive cost computation with WCAG penalties
# ═══════════════════════════════════════════════════════════════════════════

def compute_wcag_cognitive_cost(
    tree: AccessibilityTree,
    page_url: str = "",
) -> Dict[str, Any]:
    """Compute cognitive cost with WCAG guideline penalties.

    Returns a dict with:
    - ``violations``: list of detected violations
    - ``wcag_result``: :class:`WCAGResult` object
    - ``cost_delta``: :class:`CognitiveCostDelta` from base mapping
    - ``structural_penalty``: additional penalty from structural issues
    - ``total_cognitive_bits``: total cognitive cost in bits
    - ``wcag_criteria_map``: mapping of violated criteria to details
    """
    detector = WCAGStructuralDetector()
    violations = detector.detect(tree)
    wcag_result = detector.detect_to_result(tree, page_url)

    # Base WCAG cost delta from the mapping module
    cost_delta = compute_cost_delta(wcag_result)

    # Additional structural penalty from tree-level issues
    structural_penalty = 0.0
    issues = tree.metadata.get("issues", [])
    for issue in issues:
        issue_type = issue.get("type", "")
        structural_penalty += STRUCTURAL_COST_PENALTIES.get(issue_type, 0.0)

    # Also add penalties for violations not covered by base model
    for v in violations:
        if v.sc_id == "1.3.1":
            structural_penalty += STRUCTURAL_COST_PENALTIES.get(
                "heading_skip", 2.0
            ) * 0.5  # Scale down to avoid double-counting
        elif v.sc_id == "4.1.2":
            structural_penalty += STRUCTURAL_COST_PENALTIES.get(
                "missing_aria_expanded", 2.5
            ) * 0.3

    total_bits = cost_delta.mu_delta + structural_penalty

    # Build criteria mapping
    criteria_map: Dict[str, Dict[str, Any]] = {}
    for v in violations:
        key = v.sc_id
        if key not in criteria_map:
            criteria_map[key] = {
                "name": v.criterion.name,
                "level": v.criterion.level.value,
                "principle": v.criterion.principle.value,
                "violations": [],
                "cost_bits": 0.0,
            }
        criteria_map[key]["violations"].append({
            "node_id": v.node_id,
            "impact": v.impact.value,
            "message": v.message,
        })
        criteria_map[key]["cost_bits"] += compute_violation_cost(v)

    return {
        "violations": [v.to_dict() for v in violations],
        "wcag_result": wcag_result,
        "cost_delta": cost_delta,
        "structural_penalty": structural_penalty,
        "total_cognitive_bits": total_bits,
        "wcag_criteria_map": criteria_map,
        "violation_count": len(violations),
    }
