"""
usability_oracle.sarif.types — Data types for SARIF 2.1.0 output format.

Provides immutable value types conforming to the Static Analysis Results
Interchange Format (SARIF) Version 2.1.0 (OASIS Standard, 27 March 2020).

Only the subset of SARIF relevant to usability-regression results is
modelled; optional sections (graphs, code flows, thread flows) are
omitted for clarity.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class SarifLevel(Enum):
    """SARIF result level (§3.27.10)."""

    NONE = "none"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@unique
class SarifKind(Enum):
    """SARIF result kind (§3.27.9)."""

    PASS = "pass"
    FAIL = "fail"
    OPEN = "open"
    INFORMATIONAL = "informational"
    NOT_APPLICABLE = "notApplicable"
    REVIEW = "review"


@unique
class InvocationStatus(Enum):
    """Whether the tool invocation succeeded."""

    SUCCESS = "success"
    FAILURE = "failure"


# ═══════════════════════════════════════════════════════════════════════════
# SarifArtifact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifArtifact:
    """A SARIF artifact — a file or resource analysed by the tool (§3.24).

    Attributes:
        uri: URI of the artifact (relative or absolute).
        description: Optional human-readable description.
        mime_type: MIME type of the artifact (e.g. ``"text/html"``).
        length: File length in bytes (``-1`` if unknown).
        roles: Artifact roles (``"analysisTarget"``, ``"resultFile"``, …).
    """

    uri: str
    description: str = ""
    mime_type: Optional[str] = None
    length: int = -1
    roles: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "location": {"uri": self.uri},
        }
        if self.description:
            d["description"] = {"text": self.description}
        if self.mime_type is not None:
            d["mimeType"] = self.mime_type
        if self.length >= 0:
            d["length"] = self.length
        if self.roles:
            d["roles"] = list(self.roles)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifArtifact:
        loc = d.get("location", {})
        desc = d.get("description", {})
        return cls(
            uri=str(loc.get("uri", "")),
            description=str(desc.get("text", "")),
            mime_type=d.get("mimeType"),
            length=int(d.get("length", -1)),
            roles=tuple(d.get("roles", [])),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifLocation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifLocation:
    """A SARIF location pinpointing where a result was detected (§3.28).

    For usability results, the "location" is typically a UI element
    identified by its accessibility-tree node id and (optionally) its
    bounding box in screen coordinates.

    Attributes:
        uri: URI of the artefact containing the location.
        node_id: Accessibility-tree node identifier.
        start_line: 1-based start line in the source (if applicable).
        start_column: 1-based start column.
        end_line: 1-based end line.
        end_column: 1-based end column.
        description: Human-readable description of the location.
        properties: Additional properties (e.g. bounding box, role).
    """

    uri: str = ""
    node_id: str = ""
    start_line: Optional[int] = None
    start_column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        physical: Dict[str, Any] = {}
        if self.uri:
            physical["artifactLocation"] = {"uri": self.uri}
        if self.start_line is not None:
            region: Dict[str, Any] = {"startLine": self.start_line}
            if self.start_column is not None:
                region["startColumn"] = self.start_column
            if self.end_line is not None:
                region["endLine"] = self.end_line
            if self.end_column is not None:
                region["endColumn"] = self.end_column
            physical["region"] = region

        d: Dict[str, Any] = {}
        if physical:
            d["physicalLocation"] = physical
        if self.node_id:
            d["logicalLocations"] = [{"name": self.node_id, "kind": "element"}]
        if self.description:
            d["message"] = {"text": self.description}
        if self.properties:
            d["properties"] = dict(self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifLocation:
        phys = d.get("physicalLocation", {})
        artifact_loc = phys.get("artifactLocation", {})
        region = phys.get("region", {})
        logical = d.get("logicalLocations", [{}])
        msg = d.get("message", {})
        return cls(
            uri=str(artifact_loc.get("uri", "")),
            node_id=str(logical[0].get("name", "")) if logical else "",
            start_line=region.get("startLine"),
            start_column=region.get("startColumn"),
            end_line=region.get("endLine"),
            end_column=region.get("endColumn"),
            description=str(msg.get("text", "")),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifRule
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifRule:
    """A SARIF reporting descriptor / rule (§3.49).

    Each rule corresponds to a specific usability check (e.g.
    "target-size-minimum", "hick-hyman-overload", "wcag-2.5.5").

    Attributes:
        rule_id: Stable identifier for the rule (e.g. ``"UO001"``).
        name: Short human-readable name.
        short_description: One-line description.
        full_description: Detailed explanation with remediation guidance.
        help_uri: URL to documentation.
        default_level: Default severity level.
        properties: Arbitrary additional properties (tags, categories).
    """

    rule_id: str
    name: str
    short_description: str = ""
    full_description: str = ""
    help_uri: str = ""
    default_level: SarifLevel = SarifLevel.WARNING
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.rule_id,
            "name": self.name,
        }
        if self.short_description:
            d["shortDescription"] = {"text": self.short_description}
        if self.full_description:
            d["fullDescription"] = {"text": self.full_description}
        if self.help_uri:
            d["helpUri"] = self.help_uri
        d["defaultConfiguration"] = {"level": self.default_level.value}
        if self.properties:
            d["properties"] = dict(self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifRule:
        short = d.get("shortDescription", {})
        full = d.get("fullDescription", {})
        default_cfg = d.get("defaultConfiguration", {})
        return cls(
            rule_id=str(d["id"]),
            name=str(d.get("name", "")),
            short_description=str(short.get("text", "")),
            full_description=str(full.get("text", "")),
            help_uri=str(d.get("helpUri", "")),
            default_level=SarifLevel(default_cfg.get("level", "warning")),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifResult:
    """A single SARIF result — one usability finding (§3.27).

    Attributes:
        rule_id: Identifier of the rule that produced this result.
        rule_index: Index of the rule in the run's ``tool.driver.rules``.
        level: Severity level.
        kind: Result kind (pass, fail, informational, …).
        message: Human-readable result message.
        locations: Where the issue was detected.
        fingerprints: Stable identifiers for result matching across runs.
        partial_fingerprints: Contextual identifiers.
        properties: Additional result properties (effect size, p-value,
            cost delta, …).
    """

    rule_id: str
    rule_index: int
    level: SarifLevel
    kind: SarifKind
    message: str
    locations: Tuple[SarifLocation, ...] = ()
    fingerprints: Dict[str, str] = field(default_factory=dict)
    partial_fingerprints: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ruleId": self.rule_id,
            "ruleIndex": self.rule_index,
            "level": self.level.value,
            "kind": self.kind.value,
            "message": {"text": self.message},
        }
        if self.locations:
            d["locations"] = [loc.to_dict() for loc in self.locations]
        if self.fingerprints:
            d["fingerprints"] = dict(self.fingerprints)
        if self.partial_fingerprints:
            d["partialFingerprints"] = dict(self.partial_fingerprints)
        if self.properties:
            d["properties"] = dict(self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifResult:
        msg = d.get("message", {})
        return cls(
            rule_id=str(d["ruleId"]),
            rule_index=int(d.get("ruleIndex", -1)),
            level=SarifLevel(d.get("level", "warning")),
            kind=SarifKind(d.get("kind", "fail")),
            message=str(msg.get("text", "")),
            locations=tuple(
                SarifLocation.from_dict(loc) for loc in d.get("locations", [])
            ),
            fingerprints=d.get("fingerprints", {}),
            partial_fingerprints=d.get("partialFingerprints", {}),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifInvocation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifInvocation:
    """A SARIF invocation record (§3.20).

    Records how the tool was invoked and whether it succeeded.

    Attributes:
        command_line: Full command line used to invoke the tool.
        execution_successful: Whether the invocation completed without
            internal errors.
        start_time_utc: ISO 8601 start time.
        end_time_utc: ISO 8601 end time.
        working_directory: Working directory at invocation.
        environment_variables: Relevant environment variables.
        exit_code: Process exit code.
        tool_execution_notifications: Informational or error messages
            from the tool itself.
    """

    command_line: str = ""
    execution_successful: bool = True
    start_time_utc: str = ""
    end_time_utc: str = ""
    working_directory: str = ""
    environment_variables: Dict[str, str] = field(default_factory=dict)
    exit_code: int = 0
    tool_execution_notifications: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "executionSuccessful": self.execution_successful,
        }
        if self.command_line:
            d["commandLine"] = self.command_line
        if self.start_time_utc:
            d["startTimeUtc"] = self.start_time_utc
        if self.end_time_utc:
            d["endTimeUtc"] = self.end_time_utc
        if self.working_directory:
            d["workingDirectory"] = {"uri": self.working_directory}
        if self.environment_variables:
            d["environmentVariables"] = dict(self.environment_variables)
        if self.exit_code != 0:
            d["exitCode"] = self.exit_code
        if self.tool_execution_notifications:
            d["toolExecutionNotifications"] = [
                {"message": {"text": n}} for n in self.tool_execution_notifications
            ]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifInvocation:
        wd = d.get("workingDirectory", {})
        notifs = d.get("toolExecutionNotifications", [])
        return cls(
            command_line=str(d.get("commandLine", "")),
            execution_successful=bool(d.get("executionSuccessful", True)),
            start_time_utc=str(d.get("startTimeUtc", "")),
            end_time_utc=str(d.get("endTimeUtc", "")),
            working_directory=str(wd.get("uri", "")),
            environment_variables=d.get("environmentVariables", {}),
            exit_code=int(d.get("exitCode", 0)),
            tool_execution_notifications=tuple(
                n.get("message", {}).get("text", "") for n in notifs
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifRun
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifRun:
    """A single SARIF run — one execution of the usability oracle (§3.14).

    Attributes:
        tool_name: Name of the tool (``"usability-oracle"``).
        tool_version: Semantic version string.
        tool_information_uri: URL to tool documentation.
        rules: Reporting descriptors (one per usability check).
        results: Findings produced by this run.
        artifacts: Files / resources that were analysed.
        invocations: How the tool was invoked.
        properties: Additional run-level properties
            (configuration, timings, summary statistics).
    """

    tool_name: str = "usability-oracle"
    tool_version: str = "0.1.0"
    tool_information_uri: str = ""
    rules: Tuple[SarifRule, ...] = ()
    results: Tuple[SarifResult, ...] = ()
    artifacts: Tuple[SarifArtifact, ...] = ()
    invocations: Tuple[SarifInvocation, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.results)

    @property
    def num_errors(self) -> int:
        return sum(1 for r in self.results if r.level == SarifLevel.ERROR)

    @property
    def num_warnings(self) -> int:
        return sum(1 for r in self.results if r.level == SarifLevel.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        tool: Dict[str, Any] = {
            "driver": {
                "name": self.tool_name,
                "version": self.tool_version,
                "rules": [r.to_dict() for r in self.rules],
            },
        }
        if self.tool_information_uri:
            tool["driver"]["informationUri"] = self.tool_information_uri

        d: Dict[str, Any] = {"tool": tool}
        if self.results:
            d["results"] = [r.to_dict() for r in self.results]
        if self.artifacts:
            d["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.invocations:
            d["invocations"] = [inv.to_dict() for inv in self.invocations]
        if self.properties:
            d["properties"] = dict(self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifRun:
        tool = d.get("tool", {})
        driver = tool.get("driver", {})
        return cls(
            tool_name=str(driver.get("name", "usability-oracle")),
            tool_version=str(driver.get("version", "0.1.0")),
            tool_information_uri=str(driver.get("informationUri", "")),
            rules=tuple(SarifRule.from_dict(r) for r in driver.get("rules", [])),
            results=tuple(
                SarifResult.from_dict(r) for r in d.get("results", [])
            ),
            artifacts=tuple(
                SarifArtifact.from_dict(a) for a in d.get("artifacts", [])
            ),
            invocations=tuple(
                SarifInvocation.from_dict(inv) for inv in d.get("invocations", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifLog
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifLog:
    """Top-level SARIF log object (§3.13).

    This is the root of a SARIF 2.1.0 document.

    Attributes:
        version: SARIF format version (always ``"2.1.0"``).
        schema_uri: URI of the SARIF JSON schema.
        runs: One or more analysis runs.
    """

    version: str = "2.1.0"
    schema_uri: str = (
        "https://docs.oasis-open.org/sarif/sarif/v2.1.0/"
        "cos02/schemas/sarif-schema-2.1.0.json"
    )
    runs: Tuple[SarifRun, ...] = ()

    @property
    def total_results(self) -> int:
        """Total number of results across all runs."""
        return sum(run.num_results for run in self.runs)

    @property
    def total_errors(self) -> int:
        """Total number of error-level results."""
        return sum(run.num_errors for run in self.runs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "$schema": self.schema_uri,
            "version": self.version,
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifLog:
        return cls(
            version=str(d.get("version", "2.1.0")),
            schema_uri=str(d.get("$schema", "")),
            runs=tuple(SarifRun.from_dict(r) for r in d.get("runs", [])),
        )
