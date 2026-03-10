"""
usability_oracle.output.models — Data models for pipeline output.

Defines the core result types produced by the usability oracle pipeline,
including verdict information, cost breakdowns, bottleneck descriptions,
timing data, and annotated UI elements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
    Severity,
    PipelineStage,
)
from usability_oracle.core.types import CostTuple


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    SARIF = "sarif"
    HTML = "html"
    CONSOLE = "console"


@dataclass
class AnnotatedElement:
    """An element in the accessibility tree annotated with analysis results.

    Carries the original element id together with a human-readable annotation,
    a severity level, and an optional source-code / tree location.
    """

    element_id: str
    annotation: str
    severity: Severity = Severity.INFO
    location: Optional[str] = None
    bottleneck_type: Optional[BottleneckType] = None
    cost_contribution: float = 0.0
    recommendation: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "element_id": self.element_id,
            "annotation": self.annotation,
            "severity": self.severity.value if isinstance(self.severity, Enum) else str(self.severity),
            "location": self.location,
            "bottleneck_type": (
                self.bottleneck_type.value
                if self.bottleneck_type is not None
                else None
            ),
            "cost_contribution": self.cost_contribution,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotatedElement":
        sev = data.get("severity", "info")
        if isinstance(sev, str):
            sev = Severity(sev.lower()) if sev.lower() in {s.value for s in Severity} else Severity.INFO
        bt = data.get("bottleneck_type")
        if isinstance(bt, str):
            bt = BottleneckType(bt) if bt in {b.value for b in BottleneckType} else None
        return cls(
            element_id=data["element_id"],
            annotation=data.get("annotation", ""),
            severity=sev,
            location=data.get("location"),
            bottleneck_type=bt,
            cost_contribution=data.get("cost_contribution", 0.0),
            recommendation=data.get("recommendation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OutputSection:
    """A labelled section of the output report.

    Each section can carry a severity colour hint and can be marked as
    collapsible (useful for HTML / console output).
    """

    title: str
    content: str
    severity: Severity = Severity.INFO
    collapsible: bool = False
    subsections: list["OutputSection"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "severity": self.severity.value if isinstance(self.severity, Enum) else str(self.severity),
            "collapsible": self.collapsible,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputSection":
        sev = data.get("severity", "info")
        if isinstance(sev, str):
            sev = Severity(sev.lower()) if sev.lower() in {s.value for s in Severity} else Severity.INFO
        return cls(
            title=data["title"],
            content=data.get("content", ""),
            severity=sev,
            collapsible=data.get("collapsible", False),
            subsections=[cls.from_dict(s) for s in data.get("subsections", [])],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Bottleneck description
# ---------------------------------------------------------------------------

@dataclass
class BottleneckDescription:
    """Full description of a detected usability bottleneck."""

    bottleneck_type: BottleneckType
    severity: Severity
    description: str
    affected_elements: list[str] = field(default_factory=list)
    cost_impact: float = 0.0
    location: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_elements": self.affected_elements,
            "cost_impact": self.cost_impact,
            "location": self.location,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BottleneckDescription":
        return cls(
            bottleneck_type=BottleneckType(data["bottleneck_type"]),
            severity=Severity(data["severity"]),
            description=data.get("description", ""),
            affected_elements=data.get("affected_elements", []),
            cost_impact=data.get("cost_impact", 0.0),
            location=data.get("location"),
            recommendation=data.get("recommendation"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------------------------

@dataclass
class CostComparison:
    """Side-by-side comparison of costs between two UI versions."""

    cost_a: CostTuple
    cost_b: CostTuple
    delta: CostTuple
    percentage_change: float = 0.0
    channel_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_a": self.cost_a.to_dict(),
            "cost_b": self.cost_b.to_dict(),
            "delta": self.delta.to_dict(),
            "percentage_change": self.percentage_change,
            "channel_deltas": self.channel_deltas,
        }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class StageTimingInfo:
    """Timing information for a single pipeline stage."""

    stage: PipelineStage
    elapsed_seconds: float
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ---------------------------------------------------------------------------
# Composite pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """The top-level result produced by the usability-oracle pipeline.

    Aggregates the regression verdict, cost comparison, bottleneck
    descriptions, annotated elements, per-stage timing, and free-form
    metadata.
    """

    verdict: RegressionVerdict
    comparison: Optional[CostComparison] = None
    bottlenecks: list[BottleneckDescription] = field(default_factory=list)
    annotated_elements: list[AnnotatedElement] = field(default_factory=list)
    timing: list[StageTimingInfo] = field(default_factory=list)
    sections: list[OutputSection] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Convenience ----------------------------------------------------------

    @property
    def total_time(self) -> float:
        """Total pipeline execution time in seconds."""
        return sum(t.elapsed_seconds for t in self.timing)

    @property
    def critical_bottlenecks(self) -> list[BottleneckDescription]:
        return [b for b in self.bottlenecks if b.severity == Severity.CRITICAL]

    @property
    def high_severity_bottlenecks(self) -> list[BottleneckDescription]:
        return [
            b for b in self.bottlenecks
            if b.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "annotated_elements": [a.to_dict() for a in self.annotated_elements],
            "timing": [t.to_dict() for t in self.timing],
            "sections": [s.to_dict() for s in self.sections],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "total_time": self.total_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineResult":
        verdict = RegressionVerdict(data["verdict"])
        comparison = None
        if data.get("comparison"):
            cd = data["comparison"]
            comparison = CostComparison(
                cost_a=CostTuple.from_dict(cd["cost_a"]),
                cost_b=CostTuple.from_dict(cd["cost_b"]),
                delta=CostTuple.from_dict(cd["delta"]),
                percentage_change=cd.get("percentage_change", 0.0),
                channel_deltas=cd.get("channel_deltas", {}),
            )
        bottlenecks = [BottleneckDescription.from_dict(b) for b in data.get("bottlenecks", [])]
        elements = [AnnotatedElement.from_dict(a) for a in data.get("annotated_elements", [])]
        timing: list[StageTimingInfo] = []
        for t in data.get("timing", []):
            timing.append(
                StageTimingInfo(
                    stage=PipelineStage(t["stage"]),
                    elapsed_seconds=t["elapsed_seconds"],
                    start_time=t.get("start_time", 0.0),
                    end_time=t.get("end_time", 0.0),
                )
            )
        sections = [OutputSection.from_dict(s) for s in data.get("sections", [])]
        return cls(
            verdict=verdict,
            comparison=comparison,
            bottlenecks=bottlenecks,
            annotated_elements=elements,
            timing=timing,
            sections=sections,
            recommendations=data.get("recommendations", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# Output result (wrapper around PipelineResult with format info)
# ---------------------------------------------------------------------------

@dataclass
class OutputResult:
    """Wrapper that pairs a pipeline result with its serialised output.

    Useful when the formatter has already rendered the result and the caller
    needs both the structured data and the formatted string.
    """

    pipeline_result: PipelineResult
    formatted_output: str = ""
    output_format: OutputFormat = OutputFormat.JSON
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_result": self.pipeline_result.to_dict(),
            "formatted_output": self.formatted_output,
            "output_format": self.output_format.value,
            "metadata": self.metadata,
        }
