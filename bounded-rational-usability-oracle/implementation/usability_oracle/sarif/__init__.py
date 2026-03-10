"""
usability_oracle.sarif — Full SARIF 2.1.0 support.

Provides a complete bidirectional SARIF 2.1.0 implementation:

- **types** — Original lightweight types (SarifLog, SarifRun, …).
- **schema** — Complete SARIF 2.1.0 object model as frozen dataclasses.
- **reader** — Parse SARIF JSON files with validation and streaming.
- **writer** — Generate SARIF 2.1.0 compliant JSON with builder pattern.
- **validator** — Schema and semantic validation with JSON-path errors.
- **converter** — Bidirectional conversion between internal models and SARIF.
- **taxonomy** — Usability-specific SARIF taxonomies (cognitive, WCAG).
- **diff** — Result differencing, baseline comparison, suppression management.

::

    from usability_oracle.sarif import SarifLog, SarifRun, SarifResult
    from usability_oracle.sarif.schema import Result, Run, SarifLog as FullSarifLog
    from usability_oracle.sarif.reader import read_sarif
    from usability_oracle.sarif.writer import SarifBuilder, write_sarif
"""

from __future__ import annotations

# --- Original lightweight types (backward-compatible) --------------------
from usability_oracle.sarif.types import (
    InvocationStatus,
    SarifArtifact,
    SarifInvocation,
    SarifKind,
    SarifLevel,
    SarifLocation,
    SarifLog,
    SarifResult,
    SarifRule,
    SarifRun,
)

# --- Full schema types ---------------------------------------------------
from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Artifact,
    ArtifactChange,
    ArtifactContent,
    ArtifactLocation,
    BaselineState,
    CodeFlow,
    Edge,
    ExceptionData,
    Fix,
    Graph,
    Invocation,
    Kind,
    Level,
    Location,
    LogicalLocation,
    Message,
    MultiformatMessageString,
    Node,
    Notification,
    NotificationLevel,
    PhysicalLocation,
    Region,
    Replacement,
    ReportingConfiguration,
    ReportingDescriptor,
    Result,
    ResultProvenance,
    RoleEnum,
    Run,
    Stack,
    StackFrame,
    Suppression,
    SuppressionKind,
    SuppressionStatus,
    Taxonomy,
    ThreadFlow,
    ThreadFlowImportance,
    ThreadFlowLocation,
    Tool,
    ToolComponent,
    ToolComponentReference,
)

# --- Reader --------------------------------------------------------------
from usability_oracle.sarif.reader import (
    ReaderOptions,
    SarifParseError,
    SarifReader,
    SarifVersionError,
    read_sarif,
    resolve_artifact_uri,
)

# --- Writer --------------------------------------------------------------
from usability_oracle.sarif.writer import (
    SarifBuilder,
    SarifWriter,
    WriterOptions,
    sarif_to_string,
    write_sarif,
)

# --- Validator -----------------------------------------------------------
from usability_oracle.sarif.validator import (
    SarifValidator,
    ValidationError,
    ValidationReport,
    ValidationSeverity,
    validate_sarif,
)

# --- Converter -----------------------------------------------------------
from usability_oracle.sarif.converter import (
    bottleneck_report_to_sarif,
    bottleneck_result_to_sarif,
    bottleneck_rule_id,
    bottleneck_to_rule,
    import_results,
    merge_runs,
    merge_sarif_logs,
    repair_hint_to_fix,
    sarif_result_to_dict,
    severity_to_level,
    wcag_result_to_sarif,
    wcag_rule_id,
    wcag_violation_to_rule,
    wcag_violation_to_sarif,
)

# --- Taxonomy ------------------------------------------------------------
from usability_oracle.sarif.taxonomy import (
    TaxonomyRegistry,
    UsabilityIssueId,
    bottleneck_taxonomy_reference,
    cognitive_bottleneck_taxonomy,
    cognitive_cost_properties,
    default_registry,
    lookup_usability_issue,
    sarif_level_to_severity,
    severity_to_sarif_level,
    wcag_taxonomy,
    wcag_taxonomy_reference,
)

# --- Diff ----------------------------------------------------------------
from usability_oracle.sarif.diff import (
    DiffCategory,
    DiffEntry,
    DiffReport,
    DiffSummary,
    SarifDiffer,
    annotate_baseline,
    compute_fingerprint,
    compute_partial_fingerprint,
    diff_sarif,
    suppress_results,
)

__all__ = [
    # Original types (backward-compatible)
    "InvocationStatus",
    "SarifArtifact",
    "SarifInvocation",
    "SarifKind",
    "SarifLevel",
    "SarifLocation",
    "SarifLog",
    "SarifResult",
    "SarifRule",
    "SarifRun",
    # Schema constants
    "SARIF_SCHEMA_URI",
    "SARIF_VERSION",
    # Full schema types
    "Artifact",
    "ArtifactChange",
    "ArtifactContent",
    "ArtifactLocation",
    "BaselineState",
    "CodeFlow",
    "Edge",
    "ExceptionData",
    "Fix",
    "Graph",
    "Invocation",
    "Kind",
    "Level",
    "Location",
    "LogicalLocation",
    "Message",
    "MultiformatMessageString",
    "Node",
    "Notification",
    "NotificationLevel",
    "PhysicalLocation",
    "Region",
    "Replacement",
    "ReportingConfiguration",
    "ReportingDescriptor",
    "Result",
    "ResultProvenance",
    "RoleEnum",
    "Run",
    "Stack",
    "StackFrame",
    "Suppression",
    "SuppressionKind",
    "SuppressionStatus",
    "Taxonomy",
    "ThreadFlow",
    "ThreadFlowImportance",
    "ThreadFlowLocation",
    "Tool",
    "ToolComponent",
    "ToolComponentReference",
    # Reader
    "ReaderOptions",
    "SarifParseError",
    "SarifReader",
    "SarifVersionError",
    "read_sarif",
    "resolve_artifact_uri",
    # Writer
    "SarifBuilder",
    "SarifWriter",
    "WriterOptions",
    "sarif_to_string",
    "write_sarif",
    # Validator
    "SarifValidator",
    "ValidationError",
    "ValidationReport",
    "ValidationSeverity",
    "validate_sarif",
    # Converter
    "bottleneck_report_to_sarif",
    "bottleneck_result_to_sarif",
    "bottleneck_rule_id",
    "bottleneck_to_rule",
    "import_results",
    "merge_runs",
    "merge_sarif_logs",
    "repair_hint_to_fix",
    "sarif_result_to_dict",
    "severity_to_level",
    "wcag_result_to_sarif",
    "wcag_rule_id",
    "wcag_violation_to_rule",
    "wcag_violation_to_sarif",
    # Taxonomy
    "TaxonomyRegistry",
    "UsabilityIssueId",
    "bottleneck_taxonomy_reference",
    "cognitive_bottleneck_taxonomy",
    "cognitive_cost_properties",
    "default_registry",
    "lookup_usability_issue",
    "sarif_level_to_severity",
    "severity_to_sarif_level",
    "wcag_taxonomy",
    "wcag_taxonomy_reference",
    # Diff
    "DiffCategory",
    "DiffEntry",
    "DiffReport",
    "DiffSummary",
    "SarifDiffer",
    "annotate_baseline",
    "compute_fingerprint",
    "compute_partial_fingerprint",
    "diff_sarif",
    "suppress_results",
]
