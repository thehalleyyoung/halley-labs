"""
usability_oracle.wcag.types — WCAG 2.2 conformance checking value types.

Immutable dataclasses modelling the four-principle structure of WCAG 2.2:
Perceivable → Operable → Understandable → Robust, with three conformance
levels (A / AA / AAA) and detailed violation records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.types import BoundingBox


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ConformanceLevel(Enum):
    """WCAG 2.2 conformance levels."""

    A = "A"
    """Level A — minimum accessibility."""

    AA = "AA"
    """Level AA — addresses major barriers (legal baseline in many jurisdictions)."""

    AAA = "AAA"
    """Level AAA — highest level of accessibility."""

    @property
    def numeric(self) -> int:
        """Numeric ordering (A=1, AA=2, AAA=3)."""
        return {"A": 1, "AA": 2, "AAA": 3}[self.value]

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, ConformanceLevel):
            return self.numeric < other.numeric
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, ConformanceLevel):
            return self.numeric <= other.numeric
        return NotImplemented

    def __str__(self) -> str:
        return self.value


@unique
class WCAGPrinciple(Enum):
    """The four WCAG 2.2 principles (POUR)."""

    PERCEIVABLE = "perceivable"
    OPERABLE = "operable"
    UNDERSTANDABLE = "understandable"
    ROBUST = "robust"

    @property
    def number(self) -> int:
        """WCAG principle number (1-4)."""
        return {
            "perceivable": 1,
            "operable": 2,
            "understandable": 3,
            "robust": 4,
        }[self.value]

    def __str__(self) -> str:
        return self.value


@unique
class ImpactLevel(Enum):
    """Severity of a WCAG violation's impact on users."""

    MINOR = "minor"
    MODERATE = "moderate"
    SERIOUS = "serious"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        return {"minor": 1, "moderate": 2, "serious": 3, "critical": 4}[self.value]


# ═══════════════════════════════════════════════════════════════════════════
# SuccessCriterion — a single WCAG 2.2 success criterion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """One WCAG 2.2 success criterion (e.g. 1.4.3 Contrast (Minimum)).

    Attributes
    ----------
    sc_id : str
        Dotted criterion id, e.g. ``"1.4.3"``.
    name : str
        Short human-readable name.
    level : ConformanceLevel
        Conformance level (A / AA / AAA).
    principle : WCAGPrinciple
        Which POUR principle this criterion belongs to.
    guideline_id : str
        Parent guideline id, e.g. ``"1.4"``.
    description : str
        One-sentence description of the requirement.
    url : str
        Canonical WCAG 2.2 specification URL.
    """

    sc_id: str
    name: str
    level: ConformanceLevel
    principle: WCAGPrinciple
    guideline_id: str
    description: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sc_id": self.sc_id,
            "name": self.name,
            "level": self.level.value,
            "principle": self.principle.value,
            "guideline_id": self.guideline_id,
            "description": self.description,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SuccessCriterion:
        return cls(
            sc_id=d["sc_id"],
            name=d["name"],
            level=ConformanceLevel(d["level"]),
            principle=WCAGPrinciple(d["principle"]),
            guideline_id=d["guideline_id"],
            description=d.get("description", ""),
            url=d.get("url", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
# WCAGGuideline — groups of success criteria
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WCAGGuideline:
    """A WCAG 2.2 guideline (e.g. 1.4 Distinguishable).

    Attributes
    ----------
    guideline_id : str
        Dotted id, e.g. ``"1.4"``.
    name : str
        Human-readable guideline name.
    principle : WCAGPrinciple
        Parent principle.
    criteria : tuple[SuccessCriterion, ...]
        Success criteria under this guideline.
    """

    guideline_id: str
    name: str
    principle: WCAGPrinciple
    criteria: Tuple[SuccessCriterion, ...] = ()

    @property
    def criterion_count(self) -> int:
        return len(self.criteria)

    def criteria_at_level(self, level: ConformanceLevel) -> Tuple[SuccessCriterion, ...]:
        """Return criteria at or below the given conformance level."""
        return tuple(sc for sc in self.criteria if sc.level <= level)


# ═══════════════════════════════════════════════════════════════════════════
# WCAGViolation — a detected conformance failure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WCAGViolation:
    """A single WCAG conformance violation.

    Captures the criterion violated, the element that triggered the failure,
    impact severity, and an optional suggested remediation.

    Attributes
    ----------
    criterion : SuccessCriterion
        The violated success criterion.
    node_id : str
        Accessibility-tree node id of the offending element.
    impact : ImpactLevel
        Severity of the violation.
    message : str
        Human-readable description of the violation.
    bounding_box : Optional[BoundingBox]
        Screen location of the violating element, if available.
    evidence : Mapping[str, Any]
        Machine-readable evidence (e.g. contrast ratio, target size).
    remediation : str
        Suggested fix.
    """

    criterion: SuccessCriterion
    node_id: str
    impact: ImpactLevel
    message: str
    bounding_box: Optional[BoundingBox] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    remediation: str = ""

    @property
    def sc_id(self) -> str:
        """Shortcut to the criterion id."""
        return self.criterion.sc_id

    @property
    def conformance_level(self) -> ConformanceLevel:
        return self.criterion.level

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "criterion": self.criterion.to_dict(),
            "node_id": self.node_id,
            "impact": self.impact.value,
            "message": self.message,
            "remediation": self.remediation,
            "evidence": dict(self.evidence),
        }
        if self.bounding_box is not None:
            d["bounding_box"] = self.bounding_box.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════════════════
# WCAGResult — aggregate conformance evaluation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WCAGResult:
    """Aggregate result of a WCAG 2.2 conformance evaluation.

    Attributes
    ----------
    violations : tuple[WCAGViolation, ...]
        All detected violations.
    target_level : ConformanceLevel
        The conformance level that was tested against.
    criteria_tested : int
        Number of success criteria evaluated.
    criteria_passed : int
        Number of success criteria that passed.
    page_url : str
        URL or identifier of the evaluated page / component.
    metadata : Mapping[str, Any]
        Additional evaluation metadata (tool version, timestamp, etc.).
    """

    violations: Tuple[WCAGViolation, ...] = ()
    target_level: ConformanceLevel = ConformanceLevel.AA
    criteria_tested: int = 0
    criteria_passed: int = 0
    page_url: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def criteria_failed(self) -> int:
        return self.criteria_tested - self.criteria_passed

    @property
    def conformance_ratio(self) -> float:
        """Fraction of tested criteria that passed (0.0–1.0)."""
        if self.criteria_tested == 0:
            return 1.0
        return self.criteria_passed / self.criteria_tested

    @property
    def is_conformant(self) -> bool:
        """True if every tested criterion at the target level passed."""
        return all(
            v.criterion.level.numeric > self.target_level.numeric
            for v in self.violations
        )

    @property
    def violations_by_impact(self) -> Mapping[ImpactLevel, Tuple[WCAGViolation, ...]]:
        """Group violations by impact severity."""
        result: Dict[ImpactLevel, list[WCAGViolation]] = {lvl: [] for lvl in ImpactLevel}
        for v in self.violations:
            result[v.impact].append(v)
        return {k: tuple(vs) for k, vs in result.items()}

    @property
    def violation_count(self) -> int:
        return len(self.violations)


__all__ = [
    "ConformanceLevel",
    "ImpactLevel",
    "SuccessCriterion",
    "WCAGGuideline",
    "WCAGPrinciple",
    "WCAGResult",
    "WCAGViolation",
]
