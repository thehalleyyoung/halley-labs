"""
usability_oracle.sarif.converter — Convert between internal models and SARIF.

Bidirectional conversion:
  - Bottleneck results → SARIF results
  - WCAG violations → SARIF results
  - Repair suggestions → SARIF fix objects
  - SARIF import / merge / round-trip

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.sarif.schema import (
    SARIF_VERSION,
    Artifact,
    ArtifactChange,
    ArtifactContent,
    ArtifactLocation,
    BaselineState,
    Fix,
    Invocation,
    Kind,
    Level,
    Location,
    LogicalLocation,
    Message,
    MultiformatMessageString,
    PhysicalLocation,
    Region,
    Replacement,
    ReportingConfiguration,
    ReportingDescriptor,
    Result,
    ResultProvenance,
    Run,
    SarifLog,
    Suppression,
    Tool,
    ToolComponent,
    ToolComponentReference,
)

# Lazy imports to avoid circular dependencies.
_BottleneckResult = None  # type: ignore[assignment]
_BottleneckReport = None  # type: ignore[assignment]
_WCAGViolation = None  # type: ignore[assignment]
_WCAGResult = None  # type: ignore[assignment]


def _ensure_imports() -> None:
    """Lazy import of bottleneck / WCAG types to break circular deps."""
    global _BottleneckResult, _BottleneckReport, _WCAGViolation, _WCAGResult
    if _BottleneckResult is None:
        from usability_oracle.bottleneck.models import (
            BottleneckReport as BR,
            BottleneckResult as BRes,
        )
        _BottleneckResult = BRes
        _BottleneckReport = BR
    if _WCAGViolation is None:
        from usability_oracle.wcag.types import (
            WCAGResult as WR,
            WCAGViolation as WV,
        )
        _WCAGViolation = WV
        _WCAGResult = WR


# ═══════════════════════════════════════════════════════════════════════════
# Severity / level mapping
# ═══════════════════════════════════════════════════════════════════════════

_SEVERITY_TO_LEVEL: Dict[str, Level] = {
    "critical": Level.ERROR,
    "high": Level.ERROR,
    "medium": Level.WARNING,
    "low": Level.NOTE,
    "info": Level.NONE,
}

_LEVEL_TO_SEVERITY: Dict[Level, str] = {
    Level.ERROR: "high",
    Level.WARNING: "medium",
    Level.NOTE: "low",
    Level.NONE: "info",
}

_IMPACT_TO_LEVEL: Dict[str, Level] = {
    "critical": Level.ERROR,
    "serious": Level.ERROR,
    "moderate": Level.WARNING,
    "minor": Level.NOTE,
}


def severity_to_level(severity: Severity) -> Level:
    """Map internal :class:`Severity` to SARIF :class:`Level`."""
    return _SEVERITY_TO_LEVEL.get(severity.value, Level.WARNING)


def level_to_severity(level: Level) -> str:
    """Map SARIF :class:`Level` to an internal severity string."""
    return _LEVEL_TO_SEVERITY.get(level, "medium")


# ═══════════════════════════════════════════════════════════════════════════
# Rule-ID generation
# ═══════════════════════════════════════════════════════════════════════════

_BOTTLENECK_RULE_PREFIX = "UO"
_WCAG_RULE_PREFIX = "WCAG"

_BOTTLENECK_RULE_MAP: Dict[BottleneckType, str] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: "UO001",
    BottleneckType.CHOICE_PARALYSIS: "UO002",
    BottleneckType.MOTOR_DIFFICULTY: "UO003",
    BottleneckType.MEMORY_DECAY: "UO004",
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: "UO005",
}


def bottleneck_rule_id(bt: BottleneckType) -> str:
    """Stable SARIF rule ID for a bottleneck type."""
    return _BOTTLENECK_RULE_MAP.get(bt, f"UO{hash(bt.value) % 900 + 100:03d}")


def wcag_rule_id(sc_id: str) -> str:
    """Stable SARIF rule ID for a WCAG success criterion."""
    return f"WCAG-{sc_id}"


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fingerprint(rule_id: str, *parts: str) -> str:
    """Compute a stable fingerprint hash from rule ID and additional parts."""
    data = "/".join([rule_id, *parts])
    return hashlib.sha256(data.encode()).hexdigest()[:32]


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck → SARIF
# ═══════════════════════════════════════════════════════════════════════════

def bottleneck_to_rule(bt: BottleneckType) -> ReportingDescriptor:
    """Convert a :class:`BottleneckType` to a SARIF reporting descriptor."""
    return ReportingDescriptor(
        id=bottleneck_rule_id(bt),
        name=bt.value.replace("_", "-"),
        short_description=MultiformatMessageString(
            text=bt.suggested_action,
        ),
        full_description=MultiformatMessageString(
            text=(
                f"Cognitive bottleneck: {bt.value}. "
                f"Dominant law: {bt.cognitive_law.human_readable}. "
                f"Affected channel: {bt.affected_channel}."
            ),
        ),
        default_configuration=ReportingConfiguration(
            level=_SEVERITY_TO_LEVEL.get(
                {0.9: "critical", 0.8: "high", 0.7: "medium", 0.6: "low", 0.5: "low"}.get(
                    bt.severity_weight, "medium"
                ),
                Level.WARNING,
            ),
        ),
        properties={
            "tags": ["usability", "cognitive-bottleneck", bt.affected_channel],
            "cognitiveLaw": bt.cognitive_law.value,
        },
    )


def bottleneck_result_to_sarif(result: Any) -> Result:
    """Convert a :class:`BottleneckResult` to a SARIF :class:`Result`.

    Accepts a ``BottleneckResult`` from ``usability_oracle.bottleneck.models``.
    """
    _ensure_imports()
    rule_id = bottleneck_rule_id(result.bottleneck_type)
    level = severity_to_level(result.severity)

    # Build locations from affected_states.
    locations: List[Location] = []
    for state in result.affected_states:
        locations.append(
            Location(
                logical_locations=(
                    LogicalLocation(name=state, kind="uiState"),
                ),
            )
        )

    # Related locations for affected actions.
    related: List[Location] = []
    for i, action in enumerate(result.affected_actions):
        related.append(
            Location(
                id=i,
                logical_locations=(
                    LogicalLocation(name=action, kind="uiAction"),
                ),
            )
        )

    # Evidence → properties.
    props: Dict[str, Any] = dict(result.evidence)
    props["channel"] = result.channel
    props["confidence"] = result.confidence
    props["cognitiveLaw"] = result.cognitive_law.value
    if result.repair_hints:
        props["repairHints"] = result.repair_hints

    # Fingerprint.
    states_key = ",".join(sorted(result.affected_states))
    fp = _fingerprint(rule_id, states_key)

    return Result(
        rule_id=rule_id,
        level=level,
        kind=Kind.FAIL,
        message=Message(text=result.description),
        locations=tuple(locations),
        related_locations=tuple(related),
        fingerprints={"usabilityOracle/v1": fp},
        properties=props,
    )


def bottleneck_report_to_sarif(
    report: Any,
    *,
    tool_version: str = "0.1.0",
) -> SarifLog:
    """Convert a full :class:`BottleneckReport` to a SARIF log.

    Accepts a ``BottleneckReport`` from ``usability_oracle.bottleneck.models``.
    """
    _ensure_imports()
    # Collect unique rules.
    rule_types = {b.bottleneck_type for b in report.bottlenecks}
    rules = tuple(bottleneck_to_rule(bt) for bt in sorted(rule_types, key=lambda x: x.value))
    rule_index = {bottleneck_rule_id(bt): i for i, bt in enumerate(sorted(rule_types, key=lambda x: x.value))}

    results: List[Result] = []
    for b in report.bottlenecks:
        r = bottleneck_result_to_sarif(b)
        idx = rule_index.get(r.rule_id, -1)
        # Patch rule_index into result.
        results.append(
            Result(
                rule_id=r.rule_id,
                rule_index=idx,
                level=r.level,
                kind=r.kind,
                message=r.message,
                locations=r.locations,
                related_locations=r.related_locations,
                fingerprints=r.fingerprints,
                properties=r.properties,
            )
        )

    driver = ToolComponent(
        name="usability-oracle",
        version=tool_version,
        rules=rules,
    )
    run = Run(
        tool=Tool(driver=driver),
        results=tuple(results),
        properties={
            "summary": report.summary,
            "totalCostImpact": report.total_cost_impact,
        },
    )
    return SarifLog(runs=(run,))


# ═══════════════════════════════════════════════════════════════════════════
# WCAG → SARIF
# ═══════════════════════════════════════════════════════════════════════════

def wcag_violation_to_rule(violation: Any) -> ReportingDescriptor:
    """Convert a WCAG violation's criterion to a SARIF rule."""
    _ensure_imports()
    sc = violation.criterion
    return ReportingDescriptor(
        id=wcag_rule_id(sc.sc_id),
        name=sc.name.lower().replace(" ", "-"),
        short_description=MultiformatMessageString(text=sc.description),
        help_uri=sc.url,
        default_configuration=ReportingConfiguration(
            level=_IMPACT_TO_LEVEL.get(violation.impact.value, Level.WARNING),
        ),
        properties={
            "tags": ["accessibility", "wcag", sc.principle.value.lower()],
            "conformanceLevel": sc.level.value,
            "guidelineId": sc.guideline_id,
        },
    )


def wcag_violation_to_sarif(violation: Any) -> Result:
    """Convert a :class:`WCAGViolation` to a SARIF :class:`Result`."""
    _ensure_imports()
    rule_id = wcag_rule_id(violation.criterion.sc_id)
    level = _IMPACT_TO_LEVEL.get(violation.impact.value, Level.WARNING)

    # Build location.
    locations: List[Location] = []
    logical_locs: List[LogicalLocation] = []
    if violation.node_id:
        logical_locs.append(
            LogicalLocation(name=violation.node_id, kind="element")
        )

    phys = None
    if violation.bounding_box is not None:
        bb = violation.bounding_box
        phys = PhysicalLocation(
            region=Region(
                properties={
                    "boundingBox": {
                        "x": bb.x,
                        "y": bb.y,
                        "width": bb.width,
                        "height": bb.height,
                    }
                }
            )
        )

    if logical_locs or phys:
        locations.append(
            Location(
                physical_location=phys,
                logical_locations=tuple(logical_locs),
            )
        )

    props: Dict[str, Any] = dict(violation.evidence)
    if violation.remediation:
        props["remediation"] = violation.remediation

    fp = _fingerprint(rule_id, violation.node_id)

    return Result(
        rule_id=rule_id,
        level=level,
        kind=Kind.FAIL,
        message=Message(text=violation.message),
        locations=tuple(locations),
        fingerprints={"usabilityOracle/v1": fp},
        properties=props,
    )


def wcag_result_to_sarif(
    wcag_result: Any,
    *,
    tool_version: str = "0.1.0",
) -> SarifLog:
    """Convert a full :class:`WCAGResult` to a SARIF log."""
    _ensure_imports()
    # Collect unique rules.
    seen_sc: dict[str, ReportingDescriptor] = {}
    for v in wcag_result.violations:
        sc_id = v.criterion.sc_id
        if sc_id not in seen_sc:
            seen_sc[sc_id] = wcag_violation_to_rule(v)
    rules = tuple(seen_sc.values())
    rule_id_to_idx = {r.id: i for i, r in enumerate(rules)}

    results: List[Result] = []
    for v in wcag_result.violations:
        r = wcag_violation_to_sarif(v)
        idx = rule_id_to_idx.get(r.rule_id, -1)
        results.append(
            Result(
                rule_id=r.rule_id,
                rule_index=idx,
                level=r.level,
                kind=r.kind,
                message=r.message,
                locations=r.locations,
                fingerprints=r.fingerprints,
                properties=r.properties,
            )
        )

    driver = ToolComponent(
        name="usability-oracle",
        version=tool_version,
        rules=rules,
    )
    run = Run(
        tool=Tool(driver=driver),
        results=tuple(results),
        properties={
            "pageUrl": wcag_result.page_url,
            "targetLevel": wcag_result.target_level.value,
            "criteriaTested": wcag_result.criteria_tested,
            "criteriaPassed": wcag_result.criteria_passed,
            "isConformant": wcag_result.is_conformant,
        },
    )
    return SarifLog(runs=(run,))


# ═══════════════════════════════════════════════════════════════════════════
# Repair suggestions → SARIF Fix
# ═══════════════════════════════════════════════════════════════════════════

def repair_hint_to_fix(
    hint: str,
    uri: str = "",
    region: Optional[Region] = None,
    replacement_text: str = "",
) -> Fix:
    """Convert a repair hint string into a SARIF :class:`Fix`.

    If *replacement_text* is provided, a concrete replacement is generated.
    Otherwise only the description is set.
    """
    changes: List[ArtifactChange] = []
    if replacement_text and (uri or region is not None):
        changes.append(
            ArtifactChange(
                artifact_location=ArtifactLocation(uri=uri),
                replacements=(
                    Replacement(
                        deleted_region=region or Region(),
                        inserted_content=ArtifactContent(text=replacement_text),
                    ),
                ),
            )
        )
    return Fix(
        description=Message(text=hint),
        artifact_changes=tuple(changes),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SARIF → internal format (round-trip)
# ═══════════════════════════════════════════════════════════════════════════

def sarif_result_to_dict(result: Result) -> Dict[str, Any]:
    """Convert a SARIF :class:`Result` back to a flat dict for internal use."""
    d: Dict[str, Any] = {
        "rule_id": result.rule_id,
        "level": result.level.value,
        "kind": result.kind.value,
        "message": result.message.text,
    }
    if result.locations:
        loc = result.locations[0]
        if loc.logical_locations:
            d["node_id"] = loc.logical_locations[0].name
        if loc.physical_location and loc.physical_location.artifact_location:
            d["uri"] = loc.physical_location.artifact_location.uri
        if loc.physical_location and loc.physical_location.region:
            r = loc.physical_location.region
            if r.start_line is not None:
                d["start_line"] = r.start_line
            if r.end_line is not None:
                d["end_line"] = r.end_line
    d["properties"] = dict(result.properties)
    d["fingerprints"] = dict(result.fingerprints)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# SARIF merge
# ═══════════════════════════════════════════════════════════════════════════

def merge_sarif_logs(*logs: SarifLog) -> SarifLog:
    """Merge multiple SARIF logs into one log with combined runs.

    Each input log's runs are collected into the output's ``runs`` array.
    """
    if not logs:
        return SarifLog()
    all_runs: List[Run] = []
    for log in logs:
        all_runs.extend(log.runs)
    return SarifLog(
        version=SARIF_VERSION,
        runs=tuple(all_runs),
    )


def merge_runs(*runs: Run) -> Run:
    """Merge multiple runs into a single run.

    Rules are deduplicated by id.  Results are concatenated.
    """
    if not runs:
        return Run()
    # Merge rules.
    seen_rules: Dict[str, ReportingDescriptor] = {}
    for run in runs:
        for rule in run.tool.driver.rules:
            if rule.id not in seen_rules:
                seen_rules[rule.id] = rule
    merged_rules = tuple(seen_rules.values())
    rule_idx = {r.id: i for i, r in enumerate(merged_rules)}

    # Reindex results.
    all_results: List[Result] = []
    for run in runs:
        for result in run.results:
            new_idx = rule_idx.get(result.rule_id, -1)
            all_results.append(
                Result(
                    rule_id=result.rule_id,
                    rule_index=new_idx,
                    rule=result.rule,
                    kind=result.kind,
                    level=result.level,
                    message=result.message,
                    locations=result.locations,
                    fingerprints=result.fingerprints,
                    partial_fingerprints=result.partial_fingerprints,
                    related_locations=result.related_locations,
                    fixes=result.fixes,
                    properties=result.properties,
                )
            )

    # Merge artifacts.
    seen_uris: Dict[str, Artifact] = {}
    for run in runs:
        for art in run.artifacts:
            if art.location and art.location.uri:
                seen_uris.setdefault(art.location.uri, art)

    driver = ToolComponent(
        name=runs[0].tool.driver.name,
        version=runs[0].tool.driver.version,
        rules=merged_rules,
    )
    return Run(
        tool=Tool(driver=driver),
        results=tuple(all_results),
        artifacts=tuple(seen_uris.values()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Import from external SARIF
# ═══════════════════════════════════════════════════════════════════════════

def import_results(log: SarifLog) -> List[Dict[str, Any]]:
    """Import results from a SARIF log into flat dicts for internal use."""
    out: List[Dict[str, Any]] = []
    for run in log.runs:
        for result in run.results:
            out.append(sarif_result_to_dict(result))
    return out
