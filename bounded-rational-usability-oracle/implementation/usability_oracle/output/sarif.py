"""
usability_oracle.output.sarif — SARIF 2.1.0 formatter.

Produces Static Analysis Results Interchange Format (SARIF) v2.1.0 output
that can be consumed by GitHub Code Scanning, VS Code SARIF Viewer, and
other SARIF-compatible tools.

Specification: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

from usability_oracle.core.enums import (
    BottleneckType,
    Severity,
)
from usability_oracle.output.models import (
    BottleneckDescription,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
_SARIF_VERSION = "2.1.0"
_TOOL_NAME = "usability-oracle"
_TOOL_FULL_NAME = "Bounded-Rational Usability Oracle"
_TOOL_VERSION = "1.0.0"
_TOOL_SEMANTIC_VERSION = "1.0.0"
_TOOL_INFORMATION_URI = "https://github.com/usability-oracle/usability-oracle"

# Map BottleneckType to a short, stable rule-id.
_RULE_IDS: dict[BottleneckType, str] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: "UO001",
    BottleneckType.CHOICE_PARALYSIS: "UO002",
    BottleneckType.MOTOR_DIFFICULTY: "UO003",
    BottleneckType.MEMORY_DECAY: "UO004",
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: "UO005",
}

_RULE_DESCRIPTIONS: dict[BottleneckType, dict[str, str]] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: {
        "name": "PerceptualOverload",
        "short": "Perceptual overload detected",
        "full": (
            "The UI presents more perceptual information than a bounded-rational "
            "user can process efficiently, leading to increased visual search time "
            "and potential errors according to multiple-resource theory."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO001",
    },
    BottleneckType.CHOICE_PARALYSIS: {
        "name": "ChoiceParalysis",
        "short": "Choice paralysis detected",
        "full": (
            "The number of actionable choices exceeds Hick-Hyman optimal bounds, "
            "resulting in logarithmically increasing decision time that degrades "
            "the user experience."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO002",
    },
    BottleneckType.MOTOR_DIFFICULTY: {
        "name": "MotorDifficulty",
        "short": "Motor difficulty detected",
        "full": (
            "Target elements violate Fitts' Law ergonomic thresholds — targets "
            "are too small, too distant, or require excessive precision for "
            "comfortable interaction."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO003",
    },
    BottleneckType.MEMORY_DECAY: {
        "name": "MemoryDecay",
        "short": "Memory decay risk detected",
        "full": (
            "The interaction sequence exceeds working-memory capacity limits "
            "(Miller's 7±2 chunks), risking information loss between steps."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO004",
    },
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: {
        "name": "CrossChannelInterference",
        "short": "Cross-channel interference detected",
        "full": (
            "Multiple perceptual or motor channels compete for the same "
            "cognitive resources, violating multiple-resource theory constraints."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO005",
    },
}


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

def _severity_to_sarif_level(severity: Severity) -> str:
    """Map internal severity to SARIF ``level`` enum value.

    SARIF levels: "error", "warning", "note", "none".
    """
    mapping = {
        Severity.CRITICAL: "error",
        Severity.HIGH: "error",
        Severity.MEDIUM: "warning",
        Severity.LOW: "note",
        Severity.INFO: "note",
    }
    return mapping.get(severity, "warning")


def _severity_to_sarif_rank(severity: Severity) -> float:
    """Numeric rank (0–100) for SARIF result ranking."""
    mapping = {
        Severity.CRITICAL: 95.0,
        Severity.HIGH: 80.0,
        Severity.MEDIUM: 50.0,
        Severity.LOW: 25.0,
        Severity.INFO: 10.0,
    }
    return mapping.get(severity, 50.0)


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, 6)


# ---------------------------------------------------------------------------
# SARIFFormatter
# ---------------------------------------------------------------------------

class SARIFFormatter:
    """Formats a :class:`PipelineResult` as SARIF 2.1.0 JSON.

    Usage::

        fmt = SARIFFormatter()
        sarif_str = fmt.format(result)
    """

    def __init__(self, indent: int = 2, pretty: bool = True) -> None:
        self._indent = indent if pretty else None
        self._pretty = pretty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, result: PipelineResult) -> str:
        """Return a SARIF 2.1.0 JSON string for *result*."""
        sarif: dict[str, Any] = {
            "$schema": _SARIF_SCHEMA,
            "version": _SARIF_VERSION,
            "runs": [self._create_run(result)],
        }
        return json.dumps(sarif, indent=self._indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _create_run(self, result: PipelineResult) -> dict[str, Any]:
        run: dict[str, Any] = {
            "tool": self._create_tool_component(),
            "results": self._create_results(result.bottlenecks),
            "invocations": [self._create_invocation(result)],
        }

        # Attach timing as custom properties
        if result.timing:
            run["properties"] = {
                "timing": {
                    t.stage.value: _safe_float(t.elapsed_seconds) for t in result.timing
                },
                "total_seconds": _safe_float(result.total_time),
            }

        # Add verdict as custom property
        run.setdefault("properties", {})["verdict"] = result.verdict.value

        if result.comparison:
            run["properties"]["comparison"] = {
                "percentage_change": _safe_float(result.comparison.percentage_change),
                "channel_deltas": {
                    k: _safe_float(v) for k, v in result.comparison.channel_deltas.items()
                },
            }

        return run

    # ------------------------------------------------------------------
    # Tool component
    # ------------------------------------------------------------------

    def _create_tool_component(self) -> dict[str, Any]:
        return {
            "driver": {
                "name": _TOOL_NAME,
                "fullName": _TOOL_FULL_NAME,
                "version": _TOOL_VERSION,
                "semanticVersion": _TOOL_SEMANTIC_VERSION,
                "informationUri": _TOOL_INFORMATION_URI,
                "rules": self._create_rules(),
            }
        }

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _create_rules(self) -> list[dict[str, Any]]:
        """Generate SARIF ``reportingDescriptor`` entries for each rule."""
        rules: list[dict[str, Any]] = []
        for bt, rule_id in _RULE_IDS.items():
            desc = _RULE_DESCRIPTIONS[bt]
            rules.append({
                "id": rule_id,
                "name": desc["name"],
                "shortDescription": {"text": desc["short"]},
                "fullDescription": {"text": desc["full"]},
                "helpUri": desc["help_uri"],
                "defaultConfiguration": {
                    "level": "warning",
                },
                "properties": {
                    "bottleneck_type": bt.value,
                    "tags": ["usability", "bounded-rationality", bt.value],
                },
            })
        return rules

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _create_results(self, bottlenecks: list[BottleneckDescription]) -> list[dict[str, Any]]:
        return [self._bottleneck_to_result(b, idx) for idx, b in enumerate(bottlenecks)]

    def _bottleneck_to_result(self, bottleneck: BottleneckDescription, index: int) -> dict[str, Any]:
        """Map one bottleneck description to a SARIF result object."""
        rule_id = _RULE_IDS.get(bottleneck.bottleneck_type, "UO000")
        level = _severity_to_sarif_level(bottleneck.severity)
        rank = _severity_to_sarif_rank(bottleneck.severity)

        result: dict[str, Any] = {
            "ruleId": rule_id,
            "ruleIndex": self._rule_index(bottleneck.bottleneck_type),
            "level": level,
            "message": {
                "text": bottleneck.description,
            },
            "rank": rank,
        }

        # Location
        locations = self._build_locations(bottleneck)
        if locations:
            result["locations"] = locations

        # Related elements
        if bottleneck.affected_elements:
            result["relatedLocations"] = [
                {
                    "id": i,
                    "message": {"text": f"Affected element: {eid}"},
                    "physicalLocation": {
                        "artifactLocation": {"uri": bottleneck.location or "ui-tree"},
                        "region": {"startLine": 1},
                    },
                }
                for i, eid in enumerate(bottleneck.affected_elements)
            ]

        # Fixes / recommendations
        if bottleneck.recommendation:
            result["fixes"] = [
                {
                    "description": {"text": bottleneck.recommendation},
                    "artifactChanges": [],
                }
            ]

        # Custom properties
        result["properties"] = {
            "cost_impact": _safe_float(bottleneck.cost_impact),
            "bottleneck_type": bottleneck.bottleneck_type.value,
            "severity": bottleneck.severity.value,
        }

        return result

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_locations(bottleneck: BottleneckDescription) -> list[dict[str, Any]]:
        locations: list[dict[str, Any]] = []
        uri = bottleneck.location or "accessibility-tree"
        loc: dict[str, Any] = {
            "physicalLocation": {
                "artifactLocation": {"uri": uri},
                "region": {"startLine": 1, "startColumn": 1},
            },
        }
        if bottleneck.affected_elements:
            loc["logicalLocations"] = [
                {"name": eid, "kind": "ui-element"}
                for eid in bottleneck.affected_elements[:5]
            ]
        locations.append(loc)
        return locations

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_invocation(result: PipelineResult) -> dict[str, Any]:
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).isoformat()
        has_errors = any(
            b.severity in (Severity.CRITICAL, Severity.HIGH)
            for b in result.bottlenecks
        )
        return {
            "executionSuccessful": True,
            "startTimeUtc": ts,
            "properties": {
                "verdict": result.verdict.value,
                "total_time_seconds": _safe_float(result.total_time),
                "bottleneck_count": len(result.bottlenecks),
                "has_critical_issues": has_errors,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_index(bt: BottleneckType) -> int:
        """Return the index of the rule within the rules array."""
        ordered = list(_RULE_IDS.keys())
        try:
            return ordered.index(bt)
        except ValueError:
            return -1
