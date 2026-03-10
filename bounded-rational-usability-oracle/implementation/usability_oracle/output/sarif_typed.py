"""SARIF 2.1.0 output format using the ``usability_oracle.sarif.types`` data model.

Provides :class:`SarifFormatter` which converts pipeline results
(regression reports, bottleneck lists) into fully-typed SARIF 2.1.0
:class:`~usability_oracle.sarif.types.SarifLog` objects and JSON
strings.

Unlike the simpler :class:`~usability_oracle.output.sarif.SARIFFormatter`
which builds raw dicts, this module uses the frozen dataclass types
from :mod:`usability_oracle.sarif.types` for type safety and
validation.

Specification
-------------
OASIS Standard: SARIF Version 2.1.0 (27 March 2020).
https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.core.enums import BottleneckType, Severity
from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
)
from usability_oracle.sarif.types import (
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SARIF_SCHEMA = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/"
    "cos02/schemas/sarif-schema-2.1.0.json"
)
_SARIF_VERSION = "2.1.0"
_TOOL_NAME = "usability-oracle"
_TOOL_VERSION = "1.0.0"
_TOOL_INFO_URI = "https://github.com/usability-oracle/usability-oracle"

# Bottleneck type → rule-id mapping
_RULE_MAP: Dict[BottleneckType, str] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: "UO001",
    BottleneckType.CHOICE_PARALYSIS: "UO002",
    BottleneckType.MOTOR_DIFFICULTY: "UO003",
    BottleneckType.MEMORY_DECAY: "UO004",
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: "UO005",
}

_RULE_METADATA: Dict[BottleneckType, Dict[str, str]] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: {
        "name": "PerceptualOverload",
        "short": "Perceptual overload detected",
        "full": (
            "The UI presents more perceptual information than a bounded-rational "
            "user can process efficiently, increasing visual search time and "
            "error rates per multiple-resource theory."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO001",
    },
    BottleneckType.CHOICE_PARALYSIS: {
        "name": "ChoiceParalysis",
        "short": "Choice paralysis detected",
        "full": (
            "The number of actionable choices exceeds Hick-Hyman optimal bounds, "
            "causing logarithmically increasing decision time."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO002",
    },
    BottleneckType.MOTOR_DIFFICULTY: {
        "name": "MotorDifficulty",
        "short": "Motor difficulty detected",
        "full": (
            "Target elements violate Fitts' Law ergonomic thresholds — targets "
            "are too small or too distant for comfortable interaction."
        ),
        "help_uri": "https://usability-oracle.dev/rules/UO003",
    },
    BottleneckType.MEMORY_DECAY: {
        "name": "MemoryDecay",
        "short": "Memory decay risk detected",
        "full": (
            "The interaction sequence exceeds working-memory capacity (7±2 chunks), "
            "risking information loss between steps."
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

# Severity → SARIF level
_SEVERITY_TO_LEVEL: Dict[Severity, SarifLevel] = {
    Severity.CRITICAL: SarifLevel.ERROR,
    Severity.HIGH: SarifLevel.ERROR,
    Severity.MEDIUM: SarifLevel.WARNING,
    Severity.LOW: SarifLevel.NOTE,
    Severity.INFO: SarifLevel.NOTE,
}

# Severity → SARIF kind
_SEVERITY_TO_KIND: Dict[Severity, SarifKind] = {
    Severity.CRITICAL: SarifKind.FAIL,
    Severity.HIGH: SarifKind.FAIL,
    Severity.MEDIUM: SarifKind.FAIL,
    Severity.LOW: SarifKind.REVIEW,
    Severity.INFO: SarifKind.INFORMATIONAL,
}


def _safe_float(value: float) -> float:
    """Coerce NaN/Inf to 0.0 for safe serialisation."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, 6)


# ═══════════════════════════════════════════════════════════════════════════
# SarifFormatter
# ═══════════════════════════════════════════════════════════════════════════

class SarifFormatter:
    """Convert pipeline results to typed SARIF 2.1.0 objects.

    Usage::

        fmt = SarifFormatter()
        sarif_log = fmt.format_regression_report(pipeline_result)
        json_str = fmt.to_json(sarif_log)
    """

    def __init__(
        self,
        tool_name: str = _TOOL_NAME,
        tool_version: str = _TOOL_VERSION,
        tool_info_uri: str = _TOOL_INFO_URI,
    ) -> None:
        self._tool_name = tool_name
        self._tool_version = tool_version
        self._tool_info_uri = tool_info_uri

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_regression_report(
        self, result: PipelineResult
    ) -> SarifLog:
        """Convert a full regression result to a :class:`SarifLog`.

        Parameters
        ----------
        result : PipelineResult
            Pipeline output.

        Returns
        -------
        SarifLog
        """
        run = self.create_sarif_run(result)
        return SarifLog(
            version=_SARIF_VERSION,
            schema_uri=_SARIF_SCHEMA,
            runs=(run,),
        )

    def create_sarif_run(
        self, result: PipelineResult
    ) -> SarifRun:
        """Create a SARIF run from pipeline results.

        Parameters
        ----------
        result : PipelineResult

        Returns
        -------
        SarifRun
        """
        rules = self._build_rules(result.bottlenecks)
        rule_index = {r.rule_id: i for i, r in enumerate(rules)}
        results = tuple(
            self.create_sarif_result(b, rule_index)
            for b in result.bottlenecks
        )
        invocation = self._create_invocation(result)
        artifacts = self._build_artifacts(result)

        properties: Dict[str, Any] = {
            "verdict": result.verdict.value,
            "total_time_seconds": _safe_float(result.total_time),
        }
        if result.comparison:
            properties["percentage_change"] = _safe_float(
                result.comparison.percentage_change
            )

        return SarifRun(
            tool_name=self._tool_name,
            tool_version=self._tool_version,
            tool_information_uri=self._tool_info_uri,
            rules=tuple(rules),
            results=results,
            artifacts=artifacts,
            invocations=(invocation,),
            properties=properties,
        )

    def create_sarif_result(
        self,
        bottleneck: BottleneckDescription,
        rule_index: Dict[str, int],
    ) -> SarifResult:
        """Convert a single bottleneck to a SARIF result.

        Parameters
        ----------
        bottleneck : BottleneckDescription
        rule_index : Dict[str, int]
            Mapping from rule_id to index in the rules array.

        Returns
        -------
        SarifResult
        """
        rule_id = _RULE_MAP.get(bottleneck.bottleneck_type, "UO000")
        level = _SEVERITY_TO_LEVEL.get(bottleneck.severity, SarifLevel.WARNING)
        kind = _SEVERITY_TO_KIND.get(bottleneck.severity, SarifKind.FAIL)

        locations = self._build_locations(bottleneck)
        fingerprint = self._compute_fingerprint(bottleneck)

        return SarifResult(
            rule_id=rule_id,
            rule_index=rule_index.get(rule_id, -1),
            level=level,
            kind=kind,
            message=bottleneck.description,
            locations=tuple(locations),
            fingerprints={"usabilityOracle/v1": fingerprint},
            properties={
                "cost_impact": _safe_float(bottleneck.cost_impact),
                "bottleneck_type": bottleneck.bottleneck_type.value,
                "severity": bottleneck.severity.value,
                "affected_elements": bottleneck.affected_elements[:10],
            },
        )

    def create_sarif_rule(
        self, bottleneck_type: BottleneckType
    ) -> SarifRule:
        """Create a rule descriptor for a bottleneck type.

        Parameters
        ----------
        bottleneck_type : BottleneckType

        Returns
        -------
        SarifRule
        """
        rule_id = _RULE_MAP.get(bottleneck_type, "UO000")
        meta = _RULE_METADATA.get(bottleneck_type, {})

        default_level = SarifLevel.WARNING
        if bottleneck_type in (
            BottleneckType.PERCEPTUAL_OVERLOAD,
            BottleneckType.MOTOR_DIFFICULTY,
        ):
            default_level = SarifLevel.ERROR

        return SarifRule(
            rule_id=rule_id,
            name=meta.get("name", bottleneck_type.value),
            short_description=meta.get("short", ""),
            full_description=meta.get("full", ""),
            help_uri=meta.get("help_uri", ""),
            default_level=default_level,
            properties={
                "tags": ["usability", "bounded-rationality", bottleneck_type.value],
            },
        )

    @staticmethod
    def create_physical_location(
        uri: str = "accessibility-tree",
        start_line: Optional[int] = None,
        start_column: Optional[int] = None,
    ) -> SarifLocation:
        """Create a physical location (source file position).

        Parameters
        ----------
        uri : str
            URI of the source artefact.
        start_line : int, optional
            1-based line number.
        start_column : int, optional
            1-based column number.

        Returns
        -------
        SarifLocation
        """
        return SarifLocation(
            uri=uri,
            start_line=start_line,
            start_column=start_column,
        )

    @staticmethod
    def create_logical_location(
        node_id: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> SarifLocation:
        """Create a logical location (accessibility-tree hierarchy).

        Parameters
        ----------
        node_id : str
            Accessibility-tree node identifier.
        description : str
            Human-readable description.
        properties : dict, optional
            Extra properties (role, bounding box, etc.).

        Returns
        -------
        SarifLocation
        """
        return SarifLocation(
            node_id=node_id,
            description=description,
            properties=properties or {},
        )

    @staticmethod
    def to_json(sarif_log: SarifLog, indent: int = 2) -> str:
        """Serialise a :class:`SarifLog` to JSON.

        Parameters
        ----------
        sarif_log : SarifLog
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            SARIF JSON string.
        """
        return json.dumps(
            sarif_log.to_dict(), indent=indent, ensure_ascii=False
        )

    @staticmethod
    def validate_sarif(json_str: str) -> bool:
        """Validate a SARIF JSON string against basic structural requirements.

        Checks for required top-level fields: ``$schema``, ``version``,
        ``runs``.  Does *not* perform full JSON-Schema validation (that
        would require an external schema file).

        Parameters
        ----------
        json_str : str
            SARIF JSON string.

        Returns
        -------
        bool
            True if the document passes basic structural validation.
        """
        try:
            doc = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return False

        if not isinstance(doc, dict):
            return False

        # Required fields
        if "$schema" not in doc:
            return False
        if doc.get("version") != "2.1.0":
            return False
        runs = doc.get("runs")
        if not isinstance(runs, list):
            return False

        for run in runs:
            if not isinstance(run, dict):
                return False
            tool = run.get("tool")
            if not isinstance(tool, dict):
                return False
            driver = tool.get("driver")
            if not isinstance(driver, dict):
                return False
            if "name" not in driver:
                return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rules(
        self, bottlenecks: Sequence[BottleneckDescription]
    ) -> list[SarifRule]:
        """Build deduplicated rule descriptors from bottlenecks."""
        seen: set[BottleneckType] = set()
        rules: list[SarifRule] = []
        for b in bottlenecks:
            if b.bottleneck_type not in seen:
                seen.add(b.bottleneck_type)
                rules.append(self.create_sarif_rule(b.bottleneck_type))
        return rules

    def _build_locations(
        self, bottleneck: BottleneckDescription
    ) -> list[SarifLocation]:
        """Build SARIF locations for a bottleneck."""
        locations: list[SarifLocation] = []

        # Physical location
        uri = bottleneck.location or "accessibility-tree"
        phys = SarifLocation(
            uri=uri,
            start_line=1,
            start_column=1,
            description=bottleneck.description[:200],
        )
        locations.append(phys)

        # Logical locations for affected elements
        for eid in bottleneck.affected_elements[:5]:
            loc = SarifLocation(
                node_id=eid,
                description=f"Affected element: {eid}",
            )
            locations.append(loc)

        return locations

    @staticmethod
    def _build_artifacts(
        result: PipelineResult,
    ) -> tuple[SarifArtifact, ...]:
        """Build artifact list from pipeline metadata."""
        artifacts: list[SarifArtifact] = []

        source = result.metadata.get("source_uri")
        if source:
            artifacts.append(
                SarifArtifact(
                    uri=str(source),
                    description="Accessibility tree source",
                    roles=("analysisTarget",),
                )
            )

        return tuple(artifacts)

    @staticmethod
    def _create_invocation(
        result: PipelineResult,
    ) -> SarifInvocation:
        """Create an invocation record from pipeline timing."""
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc)
        return SarifInvocation(
            execution_successful=True,
            start_time_utc=ts.isoformat(),
        )

    @staticmethod
    def _compute_fingerprint(bottleneck: BottleneckDescription) -> str:
        """Compute a stable fingerprint for result deduplication."""
        key = (
            f"{bottleneck.bottleneck_type.value}|"
            f"{bottleneck.severity.value}|"
            f"{'|'.join(sorted(bottleneck.affected_elements[:5]))}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]
