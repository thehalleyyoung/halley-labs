"""
usability_oracle.sarif.validator — SARIF 2.1.0 schema validator.

Validates SARIF documents against the 2.1.0 schema with required-field
checking, cross-reference validation, URI and MIME-type format checks,
and semantic validation beyond the JSON schema (e.g. rule-index validity,
artifact-location consistency, fix-region consistency).

Reports all errors with JSON-path locations for easy triage.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Validation result types
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ValidationSeverity(Enum):
    """Severity of a validation finding."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A single validation finding with JSON-path location."""
    json_path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    rule: str = ""

    def __str__(self) -> str:
        sev = self.severity.value.upper()
        rule_s = f" [{self.rule}]" if self.rule else ""
        return f"{sev}{rule_s} {self.json_path}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    """Aggregate validation report."""
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(
            e.severity == ValidationSeverity.ERROR for e in self.errors
        )

    @property
    def error_count(self) -> int:
        return sum(
            1 for e in self.errors if e.severity == ValidationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1 for e in self.errors if e.severity == ValidationSeverity.WARNING
        )

    def add(
        self,
        path: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        rule: str = "",
    ) -> None:
        self.errors.append(
            ValidationError(
                json_path=path,
                message=message,
                severity=severity,
                rule=rule,
            )
        )

    def summary(self) -> str:
        return (
            f"Validation: {self.error_count} error(s), "
            f"{self.warning_count} warning(s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# URI / MIME helpers
# ═══════════════════════════════════════════════════════════════════════════

_URI_PATTERN = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9+\-.]*://|^[a-zA-Z]:[\\/]|^/|^[^:/?#]+(?:/|$)"
)
_MIME_PATTERN = re.compile(
    r"^(application|audio|font|image|message|model|multipart|text|video)"
    r"/[a-zA-Z0-9!#$&\-^_.+]+$"
)
_VALID_LEVELS = {"none", "note", "warning", "error"}
_VALID_KINDS = {
    "pass", "open", "informational", "notApplicable", "review", "fail",
}
_VALID_BASELINE_STATES = {"new", "unchanged", "updated", "absent"}
_VALID_SARIF_VERSIONS = {"2.1.0", "2.0.0"}


def _is_valid_uri(uri: str) -> bool:
    """Check if *uri* looks like a plausible URI or relative path."""
    if not uri:
        return False
    return bool(_URI_PATTERN.match(uri))


def _is_valid_mime(mime: str) -> bool:
    if not mime:
        return True  # Empty is allowed (optional field).
    return bool(_MIME_PATTERN.match(mime))


# ═══════════════════════════════════════════════════════════════════════════
# Validator
# ═══════════════════════════════════════════════════════════════════════════

class SarifValidator:
    """Validate a SARIF document (as a Python dict) against the 2.1.0 spec.

    Usage::

        validator = SarifValidator()
        report = validator.validate(sarif_dict)
        if not report.is_valid:
            for err in report.errors:
                print(err)
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    def validate(self, data: Any) -> ValidationReport:
        """Validate a complete SARIF document."""
        report = ValidationReport()
        if not isinstance(data, dict):
            report.add("$", "SARIF document must be a JSON object")
            return report

        self._check_top_level(data, report)
        return report

    def validate_run(
        self, run: Any, path: str, report: ValidationReport
    ) -> None:
        """Validate a single run object."""
        self._check_run(run, path, report)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def _check_top_level(
        self, data: Dict[str, Any], report: ValidationReport
    ) -> None:
        # version
        version = data.get("version")
        if version is None:
            report.add("$.version", "Required field 'version' is missing")
        elif version not in _VALID_SARIF_VERSIONS:
            report.add(
                "$.version",
                f"Unsupported version '{version}' (expected 2.1.0)",
            )

        # $schema (recommended)
        schema = data.get("$schema")
        if schema is not None and not isinstance(schema, str):
            report.add("$.$schema", "'$schema' must be a string")
        elif schema and not _is_valid_uri(schema):
            report.add(
                "$.$schema",
                f"Invalid URI for $schema: '{schema}'",
                severity=ValidationSeverity.WARNING,
            )

        # runs
        runs = data.get("runs")
        if runs is None:
            report.add("$.runs", "Required field 'runs' is missing")
        elif not isinstance(runs, list):
            report.add("$.runs", "'runs' must be an array")
        else:
            if not runs:
                report.add(
                    "$.runs",
                    "'runs' should contain at least one run",
                    severity=ValidationSeverity.WARNING,
                )
            for i, run_d in enumerate(runs):
                self._check_run(run_d, f"$.runs[{i}]", report)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _check_run(
        self, run: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(run, dict):
            report.add(path, "Run must be a JSON object")
            return

        # tool (required)
        tool = run.get("tool")
        if tool is None:
            report.add(f"{path}.tool", "Required field 'tool' is missing")
        elif not isinstance(tool, dict):
            report.add(f"{path}.tool", "'tool' must be an object")
        else:
            self._check_tool(tool, f"{path}.tool", report)

        # Collect rules count for cross-ref checking.
        rules = (
            run.get("tool", {}).get("driver", {}).get("rules", [])
            if isinstance(run.get("tool"), dict)
            else []
        )
        num_rules = len(rules) if isinstance(rules, list) else 0
        rule_ids = set()
        if isinstance(rules, list):
            for r in rules:
                if isinstance(r, dict) and "id" in r:
                    rule_ids.add(r["id"])

        # Collect artifact count.
        artifacts = run.get("artifacts", [])
        num_artifacts = len(artifacts) if isinstance(artifacts, list) else 0

        # results
        results = run.get("results")
        if results is not None:
            if not isinstance(results, list):
                report.add(f"{path}.results", "'results' must be an array")
            else:
                for j, res in enumerate(results):
                    self._check_result(
                        res,
                        f"{path}.results[{j}]",
                        report,
                        num_rules=num_rules,
                        rule_ids=rule_ids,
                        num_artifacts=num_artifacts,
                    )

        # artifacts
        if artifacts and isinstance(artifacts, list):
            for j, art in enumerate(artifacts):
                self._check_artifact(art, f"{path}.artifacts[{j}]", report)

        # invocations
        invocations = run.get("invocations")
        if invocations is not None:
            if not isinstance(invocations, list):
                report.add(
                    f"{path}.invocations", "'invocations' must be an array"
                )
            else:
                for j, inv in enumerate(invocations):
                    self._check_invocation(
                        inv, f"{path}.invocations[{j}]", report
                    )

        # taxonomies
        taxonomies = run.get("taxonomies")
        if taxonomies is not None:
            if not isinstance(taxonomies, list):
                report.add(
                    f"{path}.taxonomies", "'taxonomies' must be an array"
                )

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    def _check_tool(
        self, tool: Dict[str, Any], path: str, report: ValidationReport
    ) -> None:
        driver = tool.get("driver")
        if driver is None:
            report.add(
                f"{path}.driver", "Required field 'driver' is missing"
            )
        elif not isinstance(driver, dict):
            report.add(f"{path}.driver", "'driver' must be an object")
        else:
            self._check_tool_component(driver, f"{path}.driver", report)

        extensions = tool.get("extensions")
        if extensions is not None:
            if not isinstance(extensions, list):
                report.add(
                    f"{path}.extensions", "'extensions' must be an array"
                )
            else:
                for j, ext in enumerate(extensions):
                    self._check_tool_component(
                        ext, f"{path}.extensions[{j}]", report
                    )

    def _check_tool_component(
        self, comp: Dict[str, Any], path: str, report: ValidationReport
    ) -> None:
        if "name" not in comp:
            report.add(f"{path}.name", "Required field 'name' is missing")

        rules = comp.get("rules")
        if rules is not None:
            if not isinstance(rules, list):
                report.add(f"{path}.rules", "'rules' must be an array")
            else:
                seen_ids: set[str] = set()
                for j, rule in enumerate(rules):
                    self._check_reporting_descriptor(
                        rule, f"{path}.rules[{j}]", report, seen_ids
                    )

    # ------------------------------------------------------------------
    # ReportingDescriptor
    # ------------------------------------------------------------------

    def _check_reporting_descriptor(
        self,
        rd: Any,
        path: str,
        report: ValidationReport,
        seen_ids: set[str],
    ) -> None:
        if not isinstance(rd, dict):
            report.add(path, "Reporting descriptor must be an object")
            return
        rid = rd.get("id")
        if rid is None:
            report.add(f"{path}.id", "Required field 'id' is missing")
        else:
            if rid in seen_ids:
                report.add(
                    f"{path}.id",
                    f"Duplicate rule id '{rid}'",
                    severity=ValidationSeverity.WARNING,
                    rule="duplicate-rule-id",
                )
            seen_ids.add(str(rid))

        dc = rd.get("defaultConfiguration")
        if dc is not None and isinstance(dc, dict):
            level = dc.get("level")
            if level is not None and level not in _VALID_LEVELS:
                report.add(
                    f"{path}.defaultConfiguration.level",
                    f"Invalid level '{level}'",
                )

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _check_result(
        self,
        res: Any,
        path: str,
        report: ValidationReport,
        *,
        num_rules: int,
        rule_ids: set[str],
        num_artifacts: int,
    ) -> None:
        if not isinstance(res, dict):
            report.add(path, "Result must be an object")
            return

        # message (required)
        if "message" not in res:
            report.add(f"{path}.message", "Required field 'message' is missing")
        else:
            msg = res["message"]
            if isinstance(msg, dict):
                if not msg.get("text") and not msg.get("id"):
                    report.add(
                        f"{path}.message",
                        "Message must have 'text' or 'id'",
                    )

        # level
        level = res.get("level")
        if level is not None and level not in _VALID_LEVELS:
            report.add(f"{path}.level", f"Invalid level '{level}'")

        # kind
        kind = res.get("kind")
        if kind is not None and kind not in _VALID_KINDS:
            report.add(f"{path}.kind", f"Invalid kind '{kind}'")

        # baselineState
        bs = res.get("baselineState")
        if bs is not None and bs not in _VALID_BASELINE_STATES:
            report.add(
                f"{path}.baselineState",
                f"Invalid baselineState '{bs}'",
            )

        # ruleIndex cross-ref
        ri = res.get("ruleIndex")
        if ri is not None:
            if not isinstance(ri, int) or ri < 0:
                report.add(
                    f"{path}.ruleIndex",
                    f"ruleIndex must be a non-negative integer",
                )
            elif ri >= num_rules:
                report.add(
                    f"{path}.ruleIndex",
                    f"ruleIndex {ri} out of range (only {num_rules} rules)",
                    rule="invalid-rule-index",
                )

        # ruleId cross-ref
        rule_id = res.get("ruleId")
        if rule_id is not None and rule_ids and rule_id not in rule_ids:
            report.add(
                f"{path}.ruleId",
                f"ruleId '{rule_id}' not found in rules array",
                severity=ValidationSeverity.WARNING,
                rule="unresolved-rule-id",
            )

        # locations
        locations = res.get("locations")
        if locations is not None:
            if not isinstance(locations, list):
                report.add(f"{path}.locations", "'locations' must be an array")
            else:
                for j, loc in enumerate(locations):
                    self._check_location(
                        loc,
                        f"{path}.locations[{j}]",
                        report,
                        num_artifacts=num_artifacts,
                    )

        # fixes
        fixes = res.get("fixes")
        if fixes is not None and isinstance(fixes, list):
            for j, fix in enumerate(fixes):
                self._check_fix(fix, f"{path}.fixes[{j}]", report)

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    def _check_location(
        self,
        loc: Any,
        path: str,
        report: ValidationReport,
        *,
        num_artifacts: int = 0,
    ) -> None:
        if not isinstance(loc, dict):
            report.add(path, "Location must be an object")
            return

        phys = loc.get("physicalLocation")
        if phys is not None and isinstance(phys, dict):
            al = phys.get("artifactLocation")
            if al is not None and isinstance(al, dict):
                idx = al.get("index")
                if idx is not None and isinstance(idx, int):
                    if idx >= num_artifacts:
                        report.add(
                            f"{path}.physicalLocation.artifactLocation.index",
                            f"Artifact index {idx} out of range "
                            f"(only {num_artifacts} artifacts)",
                            rule="invalid-artifact-index",
                        )
            region = phys.get("region")
            if region is not None and isinstance(region, dict):
                self._check_region(
                    region,
                    f"{path}.physicalLocation.region",
                    report,
                )

    def _check_region(
        self, region: Dict[str, Any], path: str, report: ValidationReport
    ) -> None:
        sl = region.get("startLine")
        if sl is not None:
            if not isinstance(sl, int) or sl < 1:
                report.add(f"{path}.startLine", "startLine must be >= 1")
            el = region.get("endLine")
            if el is not None and isinstance(el, int):
                if isinstance(sl, int) and el < sl:
                    report.add(
                        f"{path}.endLine",
                        "endLine must be >= startLine",
                    )
        co = region.get("charOffset")
        if co is not None and (not isinstance(co, int) or co < 0):
            report.add(f"{path}.charOffset", "charOffset must be >= 0")
        bo = region.get("byteOffset")
        if bo is not None and (not isinstance(bo, int) or bo < 0):
            report.add(f"{path}.byteOffset", "byteOffset must be >= 0")

    # ------------------------------------------------------------------
    # Artifact
    # ------------------------------------------------------------------

    def _check_artifact(
        self, art: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(art, dict):
            report.add(path, "Artifact must be an object")
            return
        mime = art.get("mimeType")
        if mime is not None and not _is_valid_mime(str(mime)):
            report.add(
                f"{path}.mimeType",
                f"Invalid MIME type '{mime}'",
                severity=ValidationSeverity.WARNING,
                rule="invalid-mime",
            )
        length = art.get("length")
        if length is not None and isinstance(length, int) and length < -1:
            report.add(f"{path}.length", "length must be >= -1")

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def _check_invocation(
        self, inv: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(inv, dict):
            report.add(path, "Invocation must be an object")
            return
        if "executionSuccessful" not in inv:
            report.add(
                f"{path}.executionSuccessful",
                "Required field 'executionSuccessful' is missing",
            )

    # ------------------------------------------------------------------
    # Fix
    # ------------------------------------------------------------------

    def _check_fix(
        self, fix: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(fix, dict):
            report.add(path, "Fix must be an object")
            return
        changes = fix.get("artifactChanges")
        if changes is not None and isinstance(changes, list):
            for j, change in enumerate(changes):
                self._check_artifact_change(
                    change, f"{path}.artifactChanges[{j}]", report
                )

    def _check_artifact_change(
        self, change: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(change, dict):
            report.add(path, "ArtifactChange must be an object")
            return
        if "artifactLocation" not in change:
            report.add(
                f"{path}.artifactLocation",
                "Required field 'artifactLocation' is missing",
            )
        replacements = change.get("replacements")
        if replacements is None:
            report.add(
                f"{path}.replacements",
                "Required field 'replacements' is missing",
            )
        elif not isinstance(replacements, list):
            report.add(
                f"{path}.replacements", "'replacements' must be an array"
            )
        elif not replacements:
            report.add(
                f"{path}.replacements",
                "'replacements' must not be empty",
                severity=ValidationSeverity.WARNING,
            )
        else:
            for j, rep in enumerate(replacements):
                self._check_replacement(
                    rep, f"{path}.replacements[{j}]", report
                )

    def _check_replacement(
        self, rep: Any, path: str, report: ValidationReport
    ) -> None:
        if not isinstance(rep, dict):
            report.add(path, "Replacement must be an object")
            return
        if "deletedRegion" not in rep:
            report.add(
                f"{path}.deletedRegion",
                "Required field 'deletedRegion' is missing",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

def validate_sarif(
    data: Dict[str, Any],
    *,
    strict: bool = False,
) -> ValidationReport:
    """Validate a SARIF document dict and return a :class:`ValidationReport`."""
    return SarifValidator(strict=strict).validate(data)
