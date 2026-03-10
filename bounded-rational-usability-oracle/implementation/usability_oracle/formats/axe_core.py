"""
usability_oracle.formats.axe_core — Axe-core JSON format compatibility.

Parses axe-core accessibility audit results and converts them to the
oracle's internal representation for further analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


@dataclass
class AxeViolation:
    """A single axe-core violation."""
    id: str
    impact: str
    description: str
    help: str
    help_url: str
    tags: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AxeResult:
    """Parsed axe-core result."""
    violations: list[AxeViolation] = field(default_factory=list)
    passes: list[dict[str, Any]] = field(default_factory=list)
    incomplete: list[dict[str, Any]] = field(default_factory=list)
    inapplicable: list[dict[str, Any]] = field(default_factory=list)
    url: str = ""
    timestamp: str = ""
    test_engine: str = ""

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.impact == "critical")

    @property
    def serious_count(self) -> int:
        return sum(1 for v in self.violations if v.impact == "serious")


# ---------------------------------------------------------------------------
# Impact severity mapping
# ---------------------------------------------------------------------------

_IMPACT_SEVERITY = {
    "critical": 1.0,
    "serious": 0.75,
    "moderate": 0.5,
    "minor": 0.25,
}

_AXE_ROLE_MAP = {
    "button": "button",
    "link": "link",
    "textbox": "textfield",
    "checkbox": "checkbox",
    "radio": "radio",
    "combobox": "combobox",
    "slider": "slider",
    "tab": "tab",
    "menuitem": "menuitem",
    "listitem": "listitem",
    "heading": "heading",
    "img": "image",
    "navigation": "navigation",
    "banner": "banner",
    "main": "main",
    "contentinfo": "contentinfo",
    "complementary": "complementary",
    "form": "form",
    "search": "search",
    "region": "region",
    "dialog": "dialog",
    "alert": "alert",
    "table": "table",
    "row": "row",
    "cell": "cell",
    "list": "list",
    "tree": "tree",
    "treeitem": "treeitem",
    "group": "group",
    "toolbar": "toolbar",
    "separator": "separator",
    "menu": "menu",
}


# ---------------------------------------------------------------------------
# AxeCoreParser
# ---------------------------------------------------------------------------

class AxeCoreParser:
    """Parse axe-core accessibility audit results.

    Supports axe-core JSON output format versions 3.x and 4.x.
    """

    def __init__(self) -> None:
        self._role_map = dict(_AXE_ROLE_MAP)

    # ------------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------------

    def parse(self, data: str | dict) -> AxeResult:
        """Parse axe-core JSON data.

        Parameters:
            data: JSON string or dict of axe-core results.
        """
        if isinstance(data, str):
            data = json.loads(data)

        result = AxeResult(
            url=data.get("url", ""),
            timestamp=data.get("timestamp", ""),
            test_engine=data.get("testEngine", {}).get("name", "axe-core"),
        )

        for v in data.get("violations", []):
            result.violations.append(self._parse_violation(v))

        result.passes = data.get("passes", [])
        result.incomplete = data.get("incomplete", [])
        result.inapplicable = data.get("inapplicable", [])

        return result

    def _parse_violation(self, v: dict) -> AxeViolation:
        return AxeViolation(
            id=v.get("id", ""),
            impact=v.get("impact", "minor"),
            description=v.get("description", ""),
            help=v.get("help", ""),
            help_url=v.get("helpUrl", ""),
            tags=v.get("tags", []),
            nodes=v.get("nodes", []),
        )

    # ------------------------------------------------------------------
    # Convert to oracle format
    # ------------------------------------------------------------------

    def to_bottleneck_annotations(
        self,
        axe_result: AxeResult,
    ) -> list[dict[str, Any]]:
        """Convert axe-core violations to bottleneck annotations.

        Maps axe-core rule categories to oracle bottleneck types.
        """
        annotations = []

        for violation in axe_result.violations:
            bottleneck_type = self._map_violation_to_bottleneck(violation)
            severity = _IMPACT_SEVERITY.get(violation.impact, 0.25)

            for node_info in violation.nodes:
                target = node_info.get("target", ["unknown"])
                selector = target[0] if target else "unknown"

                annotations.append({
                    "type": bottleneck_type,
                    "severity": severity,
                    "location": selector,
                    "description": violation.help,
                    "axe_rule": violation.id,
                    "axe_impact": violation.impact,
                    "tags": violation.tags,
                    "failure_summary": node_info.get("failureSummary", ""),
                })

        return annotations

    def _map_violation_to_bottleneck(self, violation: AxeViolation) -> str:
        """Map axe-core violation category to bottleneck type."""
        tags = set(violation.tags)

        if tags & {"wcag2a", "wcag2aa"} and "color-contrast" in violation.id:
            return "perceptual_overload"
        if "keyboard" in violation.id or "focus" in violation.id:
            return "motor_difficulty"
        if "label" in violation.id or "name" in violation.id:
            return "perceptual_overload"
        if "heading" in violation.id or "landmark" in violation.id:
            return "memory_decay"
        if "aria" in violation.id:
            return "cross_channel_interference"

        return "perceptual_overload"

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self, axe_result: AxeResult) -> dict[str, Any]:
        """Compute summary statistics from axe-core results."""
        by_impact: dict[str, int] = {}
        by_tag: dict[str, int] = {}

        for v in axe_result.violations:
            by_impact[v.impact] = by_impact.get(v.impact, 0) + 1
            for tag in v.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        total_nodes = sum(len(v.nodes) for v in axe_result.violations)

        return {
            "n_violations": axe_result.n_violations,
            "n_affected_nodes": total_nodes,
            "n_passes": len(axe_result.passes),
            "n_incomplete": len(axe_result.incomplete),
            "by_impact": by_impact,
            "by_tag": by_tag,
            "critical_count": axe_result.critical_count,
            "serious_count": axe_result.serious_count,
        }

    # ------------------------------------------------------------------
    # Severity score
    # ------------------------------------------------------------------

    def severity_score(self, axe_result: AxeResult) -> float:
        """Compute an overall severity score (0-1) from violations."""
        if not axe_result.violations:
            return 0.0

        total_weight = 0.0
        for v in axe_result.violations:
            weight = _IMPACT_SEVERITY.get(v.impact, 0.25)
            total_weight += weight * len(v.nodes)

        # Normalise by a reference count
        max_expected = 50.0
        return min(total_weight / max_expected, 1.0)
