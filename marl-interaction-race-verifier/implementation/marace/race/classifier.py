"""
Race classification, severity scoring, and remediation.

Classifies detected races by type and severity, matches known patterns,
identifies root causes, and suggests coordination protocols to prevent
future races.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from marace.race.definition import (
    InteractionRace,
    RaceClassification,
    RaceCondition,
    HBInconsistency,
)


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SeverityReport:
    """Severity assessment for a single race."""
    race_id: str
    level: SeverityLevel
    score: float  # [0, 1]
    probability_component: float
    impact_component: float
    recoverability_component: float
    explanation: str = ""


class SeverityScorer:
    """Score race severity based on probability, impact, and recoverability.

    The overall score is a weighted combination::

        score = w_p · probability + w_i · impact + w_r · (1 − recoverability)

    All components are normalised to [0, 1].

    Args:
        weight_probability: Weight for the probability component.
        weight_impact: Weight for the impact component.
        weight_recoverability: Weight for the recoverability component.
    """

    def __init__(
        self,
        weight_probability: float = 0.4,
        weight_impact: float = 0.4,
        weight_recoverability: float = 0.2,
    ) -> None:
        total = weight_probability + weight_impact + weight_recoverability
        self._wp = weight_probability / total
        self._wi = weight_impact / total
        self._wr = weight_recoverability / total

    def score(self, race: InteractionRace) -> SeverityReport:
        """Score a single race."""
        prob = min(1.0, race.probability)
        impact = self._estimate_impact(race)
        recov = self._estimate_recoverability(race)

        raw_score = self._wp * prob + self._wi * impact + self._wr * (1.0 - recov)
        score = max(0.0, min(1.0, raw_score))
        level = self._level_from_score(score)

        return SeverityReport(
            race_id=race.race_id,
            level=level,
            score=score,
            probability_component=prob,
            impact_component=impact,
            recoverability_component=recov,
            explanation=self._explain(race, level, score),
        )

    def _estimate_impact(self, race: InteractionRace) -> float:
        if race.condition is None:
            return 0.5
        rob = race.condition.robustness
        # Larger negative robustness ⇒ higher impact
        return min(1.0, max(0.0, -rob / 10.0))

    def _estimate_recoverability(self, race: InteractionRace) -> float:
        # Collisions are generally not recoverable
        if race.classification == RaceClassification.COLLISION:
            return 0.1
        if race.classification == RaceClassification.DEADLOCK:
            return 0.3
        if race.classification == RaceClassification.STARVATION:
            return 0.6
        if race.classification == RaceClassification.PRIORITY_INVERSION:
            return 0.5
        return 0.5

    @staticmethod
    def _level_from_score(score: float) -> SeverityLevel:
        if score >= 0.8:
            return SeverityLevel.CRITICAL
        if score >= 0.6:
            return SeverityLevel.HIGH
        if score >= 0.4:
            return SeverityLevel.MEDIUM
        if score >= 0.2:
            return SeverityLevel.LOW
        return SeverityLevel.INFO

    @staticmethod
    def _explain(
        race: InteractionRace, level: SeverityLevel, score: float
    ) -> str:
        return (
            f"Race {race.race_id} classified as {level.value} "
            f"(score={score:.3f}). "
            f"Type: {race.classification.value}, "
            f"agents: {', '.join(race.agents)}."
        )


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

@dataclass
class RacePattern:
    """A known race pattern."""
    name: str
    classification: RaceClassification
    description: str
    match_fn: Callable[[InteractionRace], bool]
    suggested_fix: str = ""


class PatternMatcher:
    """Match races to known patterns.

    Maintains a registry of known race patterns (e.g. intersection
    collision, corridor deadlock) and matches detected races against them.
    """

    def __init__(self) -> None:
        self._patterns: List[RacePattern] = []
        self._register_defaults()

    def register(self, pattern: RacePattern) -> None:
        self._patterns.append(pattern)

    def match(self, race: InteractionRace) -> List[RacePattern]:
        """Return all patterns that match the given race."""
        return [p for p in self._patterns if p.match_fn(race)]

    def best_match(self, race: InteractionRace) -> Optional[RacePattern]:
        matches = self.match(race)
        if not matches:
            return None
        # Prefer exact classification match
        for m in matches:
            if m.classification == race.classification:
                return m
        return matches[0]

    def _register_defaults(self) -> None:
        self._patterns.extend([
            RacePattern(
                name="intersection_collision",
                classification=RaceClassification.COLLISION,
                description=(
                    "Two vehicles enter an unsignalised intersection "
                    "simultaneously from perpendicular approaches."
                ),
                match_fn=lambda r: (
                    r.classification == RaceClassification.COLLISION
                    and len(r.agents) == 2
                ),
                suggested_fix=(
                    "Implement priority ordering or virtual traffic signal "
                    "coordination protocol."
                ),
            ),
            RacePattern(
                name="corridor_deadlock",
                classification=RaceClassification.DEADLOCK,
                description=(
                    "Two robots enter a narrow corridor from opposite ends "
                    "and neither can pass."
                ),
                match_fn=lambda r: (
                    r.classification == RaceClassification.DEADLOCK
                    and len(r.agents) >= 2
                ),
                suggested_fix=(
                    "Implement corridor reservation protocol or "
                    "direction priority."
                ),
            ),
            RacePattern(
                name="merge_collision",
                classification=RaceClassification.COLLISION,
                description=(
                    "A merging vehicle and a highway vehicle collide at "
                    "the merge point due to simultaneous arrival."
                ),
                match_fn=lambda r: (
                    r.classification == RaceClassification.COLLISION
                    and r.schedule_window < 1.0
                ),
                suggested_fix=(
                    "Add yield protocol for merging vehicle or implement "
                    "cooperative gap creation."
                ),
            ),
            RacePattern(
                name="resource_starvation",
                classification=RaceClassification.STARVATION,
                description=(
                    "One agent is consistently denied access to a shared "
                    "resource due to unfair scheduling."
                ),
                match_fn=lambda r: r.classification == RaceClassification.STARVATION,
                suggested_fix="Implement fair scheduling or priority aging.",
            ),
            RacePattern(
                name="priority_inversion",
                classification=RaceClassification.PRIORITY_INVERSION,
                description=(
                    "A high-priority agent is blocked by a low-priority "
                    "agent holding a shared resource."
                ),
                match_fn=lambda r: r.classification == RaceClassification.PRIORITY_INVERSION,
                suggested_fix="Implement priority inheritance protocol.",
            ),
        ])


# ---------------------------------------------------------------------------
# Race classifier
# ---------------------------------------------------------------------------

class RaceClassifier:
    """Classify races by type and severity.

    Combines pattern matching with severity scoring to produce a
    comprehensive classification report.
    """

    def __init__(
        self,
        scorer: Optional[SeverityScorer] = None,
        matcher: Optional[PatternMatcher] = None,
    ) -> None:
        self._scorer = scorer or SeverityScorer()
        self._matcher = matcher or PatternMatcher()

    def classify(self, race: InteractionRace) -> Dict[str, Any]:
        """Classify a single race.

        Returns:
            Dictionary with classification, severity, and pattern match info.
        """
        severity = self._scorer.score(race)
        patterns = self._matcher.match(race)
        best = self._matcher.best_match(race)

        return {
            "race_id": race.race_id,
            "classification": race.classification.value,
            "severity": {
                "level": severity.level.value,
                "score": severity.score,
                "probability": severity.probability_component,
                "impact": severity.impact_component,
                "recoverability": severity.recoverability_component,
            },
            "patterns_matched": [p.name for p in patterns],
            "best_pattern": best.name if best else None,
            "suggested_fix": best.suggested_fix if best else "",
            "explanation": severity.explanation,
        }

    def classify_batch(
        self, races: List[InteractionRace]
    ) -> List[Dict[str, Any]]:
        return [self.classify(r) for r in races]

    def summary(self, races: List[InteractionRace]) -> Dict[str, Any]:
        """Summary classification of a batch of races."""
        results = self.classify_batch(races)
        by_level: Dict[str, int] = {}
        by_class: Dict[str, int] = {}
        for r in results:
            lev = r["severity"]["level"]
            by_level[lev] = by_level.get(lev, 0) + 1
            cls = r["classification"]
            by_class[cls] = by_class.get(cls, 0) + 1
        return {
            "total": len(races),
            "by_severity_level": by_level,
            "by_classification": by_class,
            "critical_count": by_level.get("critical", 0),
            "high_count": by_level.get("high", 0),
        }


# ---------------------------------------------------------------------------
# Root cause analyser
# ---------------------------------------------------------------------------

class RootCauseCategory(Enum):
    MISSING_COORDINATION = "missing_coordination"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    TIMING_MISMATCH = "timing_mismatch"
    UNFAIR_SCHEDULING = "unfair_scheduling"
    RESOURCE_CONTENTION = "resource_contention"
    UNKNOWN = "unknown"


@dataclass
class RootCauseReport:
    """Root cause analysis for a race."""
    race_id: str
    category: RootCauseCategory
    description: str
    contributing_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0


class RootCauseAnalyzer:
    """Identify root cause of a race.

    Analyses the HB inconsistency, race condition, and schedule window
    to determine why the race exists.
    """

    def analyse(self, race: InteractionRace) -> RootCauseReport:
        """Analyse the root cause of a single race."""
        factors: List[str] = []
        category = RootCauseCategory.UNKNOWN
        confidence = 0.5

        # Check HB inconsistency
        if race.hb_inconsistency:
            hb = race.hb_inconsistency
            if hb.coordination_gap:
                factors.append(f"Coordination gap: {hb.coordination_gap}")
                category = RootCauseCategory.MISSING_COORDINATION
                confidence = 0.8
            if hb.time_window < 0.1:
                factors.append(f"Tight timing window: {hb.time_window:.4f}s")
                if category == RootCauseCategory.UNKNOWN:
                    category = RootCauseCategory.TIMING_MISMATCH
                    confidence = 0.7

        # Check condition
        if race.condition:
            if abs(race.condition.robustness) < 1.0:
                factors.append(
                    f"Small safety margin: robustness={race.condition.robustness:.4f}"
                )
                if category == RootCauseCategory.UNKNOWN:
                    category = RootCauseCategory.INSUFFICIENT_MARGIN
                    confidence = 0.6

        # Check classification-specific patterns
        if race.classification == RaceClassification.STARVATION:
            category = RootCauseCategory.UNFAIR_SCHEDULING
            factors.append("Unfair scheduling detected")
            confidence = 0.7
        elif race.classification == RaceClassification.PRIORITY_INVERSION:
            category = RootCauseCategory.RESOURCE_CONTENTION
            factors.append("Priority inversion due to resource contention")
            confidence = 0.7

        desc = self._describe(category, factors)
        return RootCauseReport(
            race_id=race.race_id,
            category=category,
            description=desc,
            contributing_factors=factors,
            confidence=confidence,
        )

    @staticmethod
    def _describe(category: RootCauseCategory, factors: List[str]) -> str:
        base = {
            RootCauseCategory.MISSING_COORDINATION: (
                "The race is caused by missing coordination between agents. "
                "No happens-before ordering enforces safe sequencing."
            ),
            RootCauseCategory.INSUFFICIENT_MARGIN: (
                "The safety margin between agents is too small to tolerate "
                "schedule variations."
            ),
            RootCauseCategory.TIMING_MISMATCH: (
                "Timing differences between agents create a window in which "
                "event reordering leads to a safety violation."
            ),
            RootCauseCategory.UNFAIR_SCHEDULING: (
                "One or more agents are starved of actions due to unfair "
                "scheduling."
            ),
            RootCauseCategory.RESOURCE_CONTENTION: (
                "Agents contend for a shared resource without proper "
                "priority management."
            ),
            RootCauseCategory.UNKNOWN: "Root cause could not be determined.",
        }
        desc = base.get(category, "")
        if factors:
            desc += " Contributing factors: " + "; ".join(factors) + "."
        return desc


# ---------------------------------------------------------------------------
# Remediation suggester
# ---------------------------------------------------------------------------

@dataclass
class Remediation:
    """A suggested remediation for a race."""
    name: str
    description: str
    applicability: float  # [0, 1]
    effort: str  # "low", "medium", "high"


class RemediationSuggester:
    """Suggest coordination protocols to prevent races.

    Based on the root cause and classification, suggests concrete
    remediation strategies.
    """

    _REMEDIATIONS: Dict[RootCauseCategory, List[Remediation]] = {
        RootCauseCategory.MISSING_COORDINATION: [
            Remediation(
                "explicit_ordering",
                "Introduce explicit ordering protocol (e.g. token passing, "
                "virtual traffic signal) to enforce happens-before between "
                "conflicting events.",
                applicability=0.9,
                effort="medium",
            ),
            Remediation(
                "consensus_protocol",
                "Use a consensus protocol to agree on action ordering "
                "before execution.",
                applicability=0.7,
                effort="high",
            ),
        ],
        RootCauseCategory.INSUFFICIENT_MARGIN: [
            Remediation(
                "increase_margin",
                "Increase safety margins (e.g. larger minimum following "
                "distance, wider corridor clearance).",
                applicability=0.8,
                effort="low",
            ),
            Remediation(
                "robust_control",
                "Use robust control that guarantees safety under worst-case "
                "timing.",
                applicability=0.7,
                effort="high",
            ),
        ],
        RootCauseCategory.TIMING_MISMATCH: [
            Remediation(
                "synchronisation_barrier",
                "Add synchronisation barriers to align agent timing at "
                "critical points.",
                applicability=0.8,
                effort="medium",
            ),
            Remediation(
                "timing_guard",
                "Add timing guards that delay action execution until safe "
                "windows are confirmed.",
                applicability=0.7,
                effort="medium",
            ),
        ],
        RootCauseCategory.UNFAIR_SCHEDULING: [
            Remediation(
                "fair_scheduler",
                "Implement a fair scheduling algorithm (e.g. round-robin, "
                "weighted fair queuing).",
                applicability=0.9,
                effort="low",
            ),
        ],
        RootCauseCategory.RESOURCE_CONTENTION: [
            Remediation(
                "priority_inheritance",
                "Implement priority inheritance to prevent priority "
                "inversion.",
                applicability=0.8,
                effort="medium",
            ),
            Remediation(
                "resource_partitioning",
                "Partition shared resources to eliminate contention.",
                applicability=0.6,
                effort="high",
            ),
        ],
    }

    def __init__(self, root_cause_analyzer: Optional[RootCauseAnalyzer] = None) -> None:
        self._rca = root_cause_analyzer or RootCauseAnalyzer()

    def suggest(self, race: InteractionRace) -> List[Remediation]:
        """Suggest remediations for a single race."""
        report = self._rca.analyse(race)
        remediations = list(
            self._REMEDIATIONS.get(report.category, [])
        )
        # Sort by applicability (descending)
        remediations.sort(key=lambda r: r.applicability, reverse=True)
        return remediations

    def suggest_batch(
        self, races: List[InteractionRace]
    ) -> Dict[str, List[Remediation]]:
        return {r.race_id: self.suggest(r) for r in races}

    def top_suggestion(self, race: InteractionRace) -> Optional[Remediation]:
        suggestions = self.suggest(race)
        return suggestions[0] if suggestions else None
