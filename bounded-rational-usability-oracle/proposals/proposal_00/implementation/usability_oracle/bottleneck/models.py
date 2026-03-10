"""
usability_oracle.bottleneck.models — Data structures for bottleneck analysis.

Provides:
  - :class:`BottleneckResult` — a single detected usability bottleneck.
  - :class:`BottleneckSignature` — information-theoretic signature for a state.
  - :class:`BottleneckReport` — aggregate report of all detected bottlenecks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity


# ---------------------------------------------------------------------------
# BottleneckSignature
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BottleneckSignature:
    """Information-theoretic signature characterising a UI state.

    The signature captures the cognitive demands imposed by a state:

    - ``entropy``: H(π(·|s)) — decision entropy under bounded-rational policy.
    - ``mutual_info``: I(A; S' | S=s) — mutual information between the chosen
      action and the resulting next state.
    - ``channel_capacity``: C — effective channel capacity (bits/s) for the
      dominant cognitive resource.
    - ``utilization``: ρ = info_rate / C — fraction of channel capacity used.

    A state becomes a bottleneck when utilization ρ approaches or exceeds 1.0.

    Attributes
    ----------
    entropy : float
        Policy entropy H(π(·|s)) in nats.
    mutual_info : float
        Mutual information I(A; S' | S=s) in nats.
    channel_capacity : float
        Effective channel capacity in nats/s.
    utilization : float
        Channel utilization ρ ∈ [0, ∞); values > 1 indicate overload.
    """

    entropy: float = 0.0
    mutual_info: float = 0.0
    channel_capacity: float = float("inf")
    utilization: float = 0.0

    @property
    def entropy_bits(self) -> float:
        """Policy entropy in bits."""
        return self.entropy / math.log(2) if self.entropy > 0 else 0.0

    @property
    def mutual_info_bits(self) -> float:
        """Mutual information in bits."""
        return self.mutual_info / math.log(2) if self.mutual_info > 0 else 0.0

    @property
    def is_overloaded(self) -> bool:
        """True if utilization exceeds 1.0 (channel at capacity)."""
        return self.utilization > 1.0

    def __repr__(self) -> str:
        return (
            f"Signature(H={self.entropy:.3f}, I={self.mutual_info:.3f}, "
            f"C={self.channel_capacity:.1f}, ρ={self.utilization:.3f})"
        )


# ---------------------------------------------------------------------------
# BottleneckResult
# ---------------------------------------------------------------------------

@dataclass
class BottleneckResult:
    """A single detected cognitive bottleneck.

    Attributes
    ----------
    bottleneck_type : BottleneckType
        One of the five bottleneck categories.
    severity : Severity
        Severity classification (CRITICAL, HIGH, MEDIUM, LOW, INFO).
    confidence : float
        Detector confidence ∈ [0, 1].
    affected_states : list[str]
        State ids affected by this bottleneck.
    affected_actions : list[str]
        Action ids implicated.
    cognitive_law : CognitiveLaw
        The underlying cognitive law (Fitts', Hick-Hyman, etc.).
    channel : str
        The cognitive/motor channel that is overloaded.
    evidence : dict[str, float]
        Information-theoretic evidence supporting the classification.
        Keys may include ``"entropy"``, ``"mutual_info"``, ``"utilization"``,
        ``"fitts_id"``, ``"hick_bits"``, ``"wm_load"``, etc.
    description : str
        Human-readable description of the bottleneck.
    recommendation : str
        Suggested action to mitigate the bottleneck.
    repair_hints : list[str]
        Concrete repair strategies.
    metadata : dict[str, Any]
        Additional detector-specific data.
    """

    bottleneck_type: BottleneckType
    severity: Severity
    confidence: float = 0.0
    affected_states: list[str] = field(default_factory=list)
    affected_actions: list[str] = field(default_factory=list)
    cognitive_law: CognitiveLaw = CognitiveLaw.HICK_HYMAN
    channel: str = "cognitive"
    evidence: dict[str, float] = field(default_factory=dict)
    description: str = ""
    recommendation: str = ""
    repair_hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ------------------------------------------------

    @property
    def severity_score(self) -> float:
        """Numeric severity: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1, INFO=0."""
        mapping = {
            Severity.CRITICAL: 4.0,
            Severity.HIGH: 3.0,
            Severity.MEDIUM: 2.0,
            Severity.LOW: 1.0,
            Severity.INFO: 0.0,
        }
        return mapping.get(self.severity, 0.0)

    @property
    def impact_score(self) -> float:
        """Combined impact score = severity_score × confidence."""
        return self.severity_score * self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "affected_states": self.affected_states,
            "affected_actions": self.affected_actions,
            "cognitive_law": self.cognitive_law.value,
            "channel": self.channel,
            "evidence": self.evidence,
            "description": self.description,
            "recommendation": self.recommendation,
            "repair_hints": self.repair_hints,
        }

    def __repr__(self) -> str:
        return (
            f"BottleneckResult(type={self.bottleneck_type.value}, "
            f"severity={self.severity.value}, confidence={self.confidence:.2f}, "
            f"states={len(self.affected_states)})"
        )


# ---------------------------------------------------------------------------
# BottleneckReport
# ---------------------------------------------------------------------------

@dataclass
class BottleneckReport:
    """Aggregate report of all detected bottlenecks.

    Attributes
    ----------
    bottlenecks : list[BottleneckResult]
        All detected bottlenecks, sorted by severity.
    summary : str
        Human-readable summary of the report.
    total_cost_impact : float
        Estimated total cognitive cost impact of all bottlenecks.
    metadata : dict[str, Any]
        Additional analysis data.
    """

    bottlenecks: list[BottleneckResult] = field(default_factory=list)
    summary: str = ""
    total_cost_impact: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ------------------------------------------------

    @property
    def n_bottlenecks(self) -> int:
        return len(self.bottlenecks)

    @property
    def critical_count(self) -> int:
        return sum(1 for b in self.bottlenecks if b.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for b in self.bottlenecks if b.severity == Severity.HIGH)

    def by_type(self, btype: BottleneckType) -> list[BottleneckResult]:
        """Return bottlenecks of a specific type."""
        return [b for b in self.bottlenecks if b.bottleneck_type == btype]

    def by_severity(self, severity: Severity) -> list[BottleneckResult]:
        """Return bottlenecks at a specific severity level."""
        return [b for b in self.bottlenecks if b.severity == severity]

    def affected_states(self) -> set[str]:
        """Return all unique affected state ids."""
        states: set[str] = set()
        for b in self.bottlenecks:
            states.update(b.affected_states)
        return states

    def type_distribution(self) -> dict[str, int]:
        """Return count of bottlenecks per type."""
        dist: dict[str, int] = {}
        for b in self.bottlenecks:
            key = b.bottleneck_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def generate_summary(self) -> str:
        """Generate a human-readable summary."""
        if not self.bottlenecks:
            return "No cognitive bottlenecks detected."

        lines = [
            f"Detected {self.n_bottlenecks} cognitive bottleneck(s):",
        ]
        dist = self.type_distribution()
        for btype, count in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"  • {btype}: {count}")

        lines.append(f"Critical: {self.critical_count}, High: {self.high_count}")
        lines.append(f"Total cost impact: {self.total_cost_impact:.3f}")
        self.summary = "\n".join(lines)
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "n_bottlenecks": self.n_bottlenecks,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "summary": self.summary,
            "total_cost_impact": self.total_cost_impact,
            "type_distribution": self.type_distribution(),
        }

    def __repr__(self) -> str:
        return (
            f"BottleneckReport(n={self.n_bottlenecks}, "
            f"critical={self.critical_count}, "
            f"cost_impact={self.total_cost_impact:.3f})"
        )
