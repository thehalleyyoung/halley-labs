"""
taintflow.report.sarif – SARIF v2.1.0 report generation.

Produces `Static Analysis Results Interchange Format`_ (SARIF) output
compatible with GitHub Code Scanning, VS Code SARIF Viewer, and other
standard consumers.  Each leakage finding maps to a SARIF *result*
object with location, severity, and remediation metadata.

.. _Static Analysis Results Interchange Format:
   https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    IO,
    List,
    Optional,
    TextIO,
    Union,
)

from taintflow.core.types import (
    Severity,
    FeatureLeakage,
    StageLeakage,
    LeakageReport,
)
from taintflow.core.config import TaintFlowConfig, SeverityThresholds


# ===================================================================
#  Constants
# ===================================================================

_TOOL_VERSION = "0.1.0"
_SARIF_VERSION = "2.1.0"
_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json"
_TOOL_NAME = "TaintFlow"
_TOOL_URI = "https://github.com/taintflow/taintflow"

# Map Severity enum → SARIF level string
_SEVERITY_TO_SARIF_LEVEL: Dict[str, str] = {
    "negligible": "note",
    "warning": "warning",
    "critical": "error",
}

# Leakage pattern rule definitions
_LEAKAGE_RULES: Dict[str, Dict[str, str]] = {
    "TF001": {
        "id": "TF001",
        "name": "TrainTestLeakage",
        "short": "Train/test information leakage detected",
        "full": (
            "Data from the test partition has leaked into the training path "
            "through this pipeline stage.  The measured upper bound on leaked "
            "information is given in bits."
        ),
        "help_uri": "https://taintflow.dev/rules/TF001",
    },
    "TF002": {
        "id": "TF002",
        "name": "FeatureLeakage",
        "short": "Feature-level leakage detected",
        "full": (
            "A specific feature column carries information from the test set "
            "into training-time computations.  Review the contributing stages "
            "and consider isolating the feature engineering step."
        ),
        "help_uri": "https://taintflow.dev/rules/TF002",
    },
    "TF003": {
        "id": "TF003",
        "name": "StageLeakage",
        "short": "Pipeline stage introduces leakage",
        "full": (
            "This pipeline stage introduces or amplifies train/test leakage.  "
            "Stages that fit on the full dataset before splitting are a common "
            "source.  Move the stage inside a cross-validation fold."
        ),
        "help_uri": "https://taintflow.dev/rules/TF003",
    },
    "TF004": {
        "id": "TF004",
        "name": "HighCapacityChannel",
        "short": "High-capacity information channel detected",
        "full": (
            "The information channel capacity between partitions exceeds the "
            "critical threshold, indicating substantial leakage.  This may "
            "invalidate model evaluation metrics."
        ),
        "help_uri": "https://taintflow.dev/rules/TF004",
    },
}


# ===================================================================
#  SARIFToolComponent
# ===================================================================


@dataclass
class SARIFToolComponent:
    """Metadata about the TaintFlow tool for SARIF ``tool.driver``.

    Attributes
    ----------
    name : str
        Tool display name.
    version : str
        Semantic version of the tool.
    information_uri : str
        URL to the tool's documentation / homepage.
    """

    name: str = _TOOL_NAME
    version: str = _TOOL_VERSION
    information_uri: str = _TOOL_URI

    def to_sarif(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Serialise as a SARIF ``toolComponent`` object.

        Parameters
        ----------
        rules : list of dict
            Pre-built SARIF rule objects to embed.
        """
        return {
            "name": self.name,
            "version": self.version,
            "semanticVersion": self.version,
            "informationUri": self.information_uri,
            "rules": rules,
        }


# ===================================================================
#  SARIFRule
# ===================================================================


@dataclass
class SARIFRule:
    """A SARIF ``reportingDescriptor`` rule definition.

    Each rule corresponds to a leakage pattern recognised by TaintFlow.

    Attributes
    ----------
    rule_id : str
        Stable rule identifier (e.g. ``TF001``).
    name : str
        Human-readable rule name.
    short_description : str
        One-line summary.
    full_description : str
        Detailed explanation of the rule.
    help_uri : str
        URL to online documentation for this rule.
    default_level : str
        SARIF default level (``error``, ``warning``, ``note``).
    """

    rule_id: str = ""
    name: str = ""
    short_description: str = ""
    full_description: str = ""
    help_uri: str = ""
    default_level: str = "warning"

    @classmethod
    def from_template(cls, template_id: str) -> "SARIFRule":
        """Create a rule from one of the built-in templates."""
        tmpl = _LEAKAGE_RULES.get(template_id, {})
        return cls(
            rule_id=tmpl.get("id", template_id),
            name=tmpl.get("name", "UnknownRule"),
            short_description=tmpl.get("short", ""),
            full_description=tmpl.get("full", ""),
            help_uri=tmpl.get("help_uri", ""),
            default_level="warning",
        )

    def to_sarif(self) -> Dict[str, Any]:
        """Serialise as a SARIF ``reportingDescriptor``."""
        rule: Dict[str, Any] = {
            "id": self.rule_id,
            "name": self.name,
            "shortDescription": {"text": self.short_description},
            "fullDescription": {"text": self.full_description},
            "defaultConfiguration": {"level": self.default_level},
        }
        if self.help_uri:
            rule["helpUri"] = self.help_uri
        return rule


# ===================================================================
#  Internal helpers
# ===================================================================

def _sarif_level(severity: Severity) -> str:
    """Map a TaintFlow severity to a SARIF level."""
    return _SEVERITY_TO_SARIF_LEVEL.get(severity.value, "warning")


def _choose_rule_id(fl: FeatureLeakage, sl: StageLeakage) -> str:
    """Select the most appropriate rule for a finding."""
    if fl.severity == Severity.CRITICAL:
        return "TF004"
    if fl.contributing_stages and len(fl.contributing_stages) > 1:
        return "TF003"
    return "TF002"


def _make_physical_location(
    stage: StageLeakage,
    feature: Optional[FeatureLeakage] = None,
) -> Dict[str, Any]:
    """Build a SARIF ``physicalLocation`` from a stage.

    Since TaintFlow operates on runtime traces rather than source files,
    the location uses the stage ID as an ``artifactLocation`` URI with
    a synthetic region.  Tools consuming this can use the URI to link
    back to the pipeline definition file.
    """
    uri = f"pipeline://{stage.stage_id}"
    location: Dict[str, Any] = {
        "artifactLocation": {
            "uri": uri,
            "uriBaseId": "PIPELINE_ROOT",
        },
    }
    # Use a synthetic region – line=1 is a placeholder
    location["region"] = {
        "startLine": 1,
        "startColumn": 1,
        "message": {
            "text": (
                f"Stage '{stage.stage_name}' ({stage.op_type.value})"
                + (f" – column '{feature.column_name}'" if feature else "")
            ),
        },
    }
    return location


def _make_logical_location(
    stage: StageLeakage,
    feature: Optional[FeatureLeakage] = None,
) -> Dict[str, Any]:
    """Build a SARIF ``logicalLocation`` for a finding."""
    parts = [stage.stage_name]
    if feature:
        parts.append(feature.column_name)
    return {
        "name": " / ".join(parts),
        "fullyQualifiedName": f"{stage.stage_id}{'/' + feature.column_name if feature else ''}",
        "kind": "module",
    }


def _make_related_locations(
    feature: FeatureLeakage,
    all_stages: List[StageLeakage],
) -> List[Dict[str, Any]]:
    """Build related-location references for upstream contributing stages."""
    related: List[Dict[str, Any]] = []
    for idx, stage_id in enumerate(feature.contributing_stages):
        matching = [s for s in all_stages if s.stage_id == stage_id]
        if not matching:
            continue
        sl = matching[0]
        related.append({
            "id": idx,
            "message": {
                "text": (
                    f"Upstream stage '{sl.stage_name}' contributes "
                    f"{sl.max_bit_bound:.2f} bits"
                ),
            },
            "physicalLocation": _make_physical_location(sl),
        })
    return related


def _make_fix(
    feature: FeatureLeakage,
    stage: StageLeakage,
) -> Optional[Dict[str, Any]]:
    """Build a SARIF ``fix`` object from a remediation suggestion."""
    if not feature.remediation:
        return None
    return {
        "description": {
            "text": feature.remediation,
        },
        "artifactChanges": [
            {
                "artifactLocation": {
                    "uri": f"pipeline://{stage.stage_id}",
                    "uriBaseId": "PIPELINE_ROOT",
                },
                "replacements": [],
            }
        ],
    }


def _make_result(
    feature: FeatureLeakage,
    stage: StageLeakage,
    all_stages: List[StageLeakage],
    rule_index_map: Dict[str, int],
) -> Dict[str, Any]:
    """Build a single SARIF ``result`` object."""
    rule_id = _choose_rule_id(feature, stage)
    level = _sarif_level(feature.severity)

    message_text = (
        f"Feature '{feature.column_name}' leaks {feature.bit_bound:.4f} bits "
        f"({feature.severity.value}) in stage '{stage.stage_name}' "
        f"({stage.op_type.value})."
    )
    if feature.explanation:
        message_text += f" {feature.explanation}"

    result: Dict[str, Any] = {
        "ruleId": rule_id,
        "ruleIndex": rule_index_map.get(rule_id, 0),
        "level": level,
        "message": {"text": message_text},
        "locations": [
            {
                "physicalLocation": _make_physical_location(stage, feature),
                "logicalLocations": [_make_logical_location(stage, feature)],
            }
        ],
        "properties": {
            "bit_bound": feature.bit_bound,
            "severity": feature.severity.value,
            "confidence": feature.confidence,
            "stage_id": stage.stage_id,
            "column": feature.column_name,
        },
    }

    # Related locations (upstream stages)
    related = _make_related_locations(feature, all_stages)
    if related:
        result["relatedLocations"] = related

    # Fix suggestion
    fix = _make_fix(feature, stage)
    if fix is not None:
        result["fixes"] = [fix]

    return result


# ===================================================================
#  SARIFReportGenerator
# ===================================================================


@dataclass
class SARIFReportGenerator:
    """Generate SARIF v2.1.0 compliant reports for IDE/CI integration.

    Parameters
    ----------
    config : TaintFlowConfig, optional
        Audit configuration.
    tool_name : str
        Override the tool display name in the SARIF output.
    tool_version : str
        Override the tool version string.
    include_fixes : bool
        Include ``fix`` objects with remediation suggestions.
    include_related : bool
        Include ``relatedLocations`` for upstream stages.
    pretty : bool
        Pretty-print the JSON output.
    indent : int
        JSON indentation width when *pretty* is ``True``.
    """

    config: Optional[TaintFlowConfig] = None
    tool_name: str = _TOOL_NAME
    tool_version: str = _TOOL_VERSION
    include_fixes: bool = True
    include_related: bool = True
    pretty: bool = True
    indent: int = 2

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def generate(self, report: LeakageReport) -> str:
        """Return the SARIF report as a JSON string."""
        doc = self._build_sarif(report)
        indent = self.indent if self.pretty else None
        return json.dumps(
            doc,
            indent=indent,
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        ) + "\n"

    def generate_to_file(self, report: LeakageReport, path: str) -> None:
        """Write the SARIF report to *path*."""
        content = self.generate(report)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def generate_to_stream(self, report: LeakageReport, stream: IO[str]) -> None:
        """Write the SARIF report to an open text stream."""
        stream.write(self.generate(report))

    def generate_dict(self, report: LeakageReport) -> Dict[str, Any]:
        """Return the SARIF report as a Python dictionary."""
        return self._build_sarif(report)

    # -----------------------------------------------------------------
    #  SARIF document builder
    # -----------------------------------------------------------------

    def _build_sarif(self, report: LeakageReport) -> Dict[str, Any]:
        """Assemble the complete SARIF log."""
        rules, rule_index_map = self._build_rules(report)
        tool_component = SARIFToolComponent(
            name=self.tool_name,
            version=self.tool_version,
        )

        results = self._build_results(report, rule_index_map)

        invocation = self._build_invocation(report)

        run: Dict[str, Any] = {
            "tool": {"driver": tool_component.to_sarif(rules)},
            "invocations": [invocation],
            "results": results,
        }

        # Artifacts (pipeline stages as logical artifacts)
        artifacts = self._build_artifacts(report)
        if artifacts:
            run["artifacts"] = artifacts

        # Properties bag with summary info
        run["properties"] = {
            "pipeline_name": report.pipeline_name,
            "total_bit_bound": report.total_bit_bound,
            "overall_severity": report.overall_severity.value,
            "n_stages": report.n_stages,
            "n_features": report.n_features,
            "n_leaking_features": report.n_leaking_features,
            "analysis_duration_ms": report.analysis_duration_ms,
        }

        return {
            "$schema": _SARIF_SCHEMA,
            "version": _SARIF_VERSION,
            "runs": [run],
        }

    def _build_rules(
        self,
        report: LeakageReport,
    ) -> tuple:
        """Build the SARIF rule list and an id→index map.

        Returns
        -------
        tuple of (list[dict], dict[str, int])
            The serialised rules and an ``{rule_id: index}`` mapping.
        """
        used_rule_ids: set = set()
        for sl in report.stage_leakages:
            for fl in sl.feature_leakages:
                used_rule_ids.add(_choose_rule_id(fl, sl))

        # Always include the base rule
        used_rule_ids.add("TF001")

        ordered = sorted(used_rule_ids)
        rules: List[Dict[str, Any]] = []
        index_map: Dict[str, int] = {}
        for idx, rid in enumerate(ordered):
            rule = SARIFRule.from_template(rid)
            rules.append(rule.to_sarif())
            index_map[rid] = idx

        return rules, index_map

    def _build_results(
        self,
        report: LeakageReport,
        rule_index_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Build the SARIF results array."""
        results: List[Dict[str, Any]] = []

        for sl in report.stage_leakages:
            for fl in sl.feature_leakages:
                result = _make_result(
                    fl,
                    sl,
                    report.stage_leakages,
                    rule_index_map,
                )
                if not self.include_fixes:
                    result.pop("fixes", None)
                if not self.include_related:
                    result.pop("relatedLocations", None)
                results.append(result)

        # Sort for deterministic output
        results.sort(
            key=lambda r: (
                -r.get("properties", {}).get("bit_bound", 0),
                r.get("properties", {}).get("column", ""),
            )
        )
        return results

    def _build_invocation(self, report: LeakageReport) -> Dict[str, Any]:
        """Build a SARIF ``invocation`` object."""
        has_errors = report.overall_severity == Severity.CRITICAL
        return {
            "executionSuccessful": True,
            "startTimeUtc": report.timestamp,
            "endTimeUtc": datetime.now(timezone.utc).isoformat(),
            "toolExecutionNotifications": [],
            "properties": {
                "analysis_duration_ms": report.analysis_duration_ms,
                "is_clean": report.is_clean,
            },
        }

    def _build_artifacts(self, report: LeakageReport) -> List[Dict[str, Any]]:
        """Build SARIF artifact entries for each pipeline stage."""
        artifacts: List[Dict[str, Any]] = []
        for sl in report.stage_leakages:
            artifacts.append({
                "location": {
                    "uri": f"pipeline://{sl.stage_id}",
                    "uriBaseId": "PIPELINE_ROOT",
                },
                "description": {
                    "text": (
                        f"Pipeline stage '{sl.stage_name}' "
                        f"({sl.op_type.value}, {sl.node_kind.value})"
                    ),
                },
                "properties": {
                    "stage_id": sl.stage_id,
                    "severity": sl.severity.value,
                    "max_bit_bound": sl.max_bit_bound,
                },
            })
        return artifacts


# ===================================================================
#  Convenience function
# ===================================================================


def generate_sarif_report(
    report: LeakageReport,
    config: Optional[TaintFlowConfig] = None,
    path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """One-shot helper: build a SARIF report and optionally write to *path*.

    Parameters
    ----------
    report : LeakageReport
        The audit result to serialise.
    config : TaintFlowConfig, optional
        Audit configuration.
    path : str, optional
        If given, write the SARIF JSON to this file path.
    **kwargs
        Forwarded to :class:`SARIFReportGenerator`.

    Returns
    -------
    str
        The SARIF JSON string.
    """
    gen = SARIFReportGenerator(config=config, **kwargs)
    content = gen.generate(report)
    if path is not None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    return content
