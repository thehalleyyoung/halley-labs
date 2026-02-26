"""
Interaction race definition.

Defines the core data structures for representing interaction races
in multi-agent systems: the race itself, the safety-violating condition,
happens-before inconsistencies, witnesses, absence certificates, and
a classification taxonomy.
"""

from __future__ import annotations

import copy
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Race classification taxonomy
# ---------------------------------------------------------------------------

class RaceClassification(enum.Enum):
    """Taxonomy of interaction race types."""
    COLLISION = "collision"
    DEADLOCK = "deadlock"
    STARVATION = "starvation"
    PRIORITY_INVERSION = "priority_inversion"
    CUSTOM = "custom"

    @staticmethod
    def from_string(s: str) -> "RaceClassification":
        try:
            return RaceClassification(s.lower())
        except ValueError:
            return RaceClassification.CUSTOM


# ---------------------------------------------------------------------------
# Schedule event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleEvent:
    """An atomic event in a multi-agent schedule.

    Attributes:
        agent_id: Agent that produced this event.
        event_type: Type label (e.g. ``"action"``, ``"observation"``).
        timestamp: Logical or wall-clock time.
        data: Arbitrary payload.
    """
    agent_id: str
    event_type: str = "action"
    timestamp: float = 0.0
    data: Any = None

    def __repr__(self) -> str:
        return (
            f"ScheduleEvent({self.agent_id}, {self.event_type}, "
            f"t={self.timestamp:.4f})"
        )


# ---------------------------------------------------------------------------
# Happens-before inconsistency
# ---------------------------------------------------------------------------

@dataclass
class HBInconsistency:
    """Characterise the missing happens-before (HB) ordering that enables a race.

    In a race-free schedule, every pair of conflicting events is ordered by
    a happens-before relation.  An ``HBInconsistency`` records a specific
    pair of events for which no such ordering exists.

    Attributes:
        event_a: First event.
        event_b: Second event.
        expected_order: The ordering that *should* exist to prevent the race
            (``"a_before_b"`` or ``"b_before_a"``).
        coordination_gap: Qualitative description of the missing mechanism.
        required_mechanism: Suggested coordination to establish the ordering.
    """
    event_a: ScheduleEvent
    event_b: ScheduleEvent
    expected_order: str = "a_before_b"
    coordination_gap: str = ""
    required_mechanism: str = ""

    @property
    def agents(self) -> Tuple[str, str]:
        return (self.event_a.agent_id, self.event_b.agent_id)

    @property
    def time_window(self) -> float:
        """Time window within which the ordering is ambiguous."""
        return abs(self.event_a.timestamp - self.event_b.timestamp)

    def __repr__(self) -> str:
        return (
            f"HBInconsistency({self.event_a.agent_id}@{self.event_a.timestamp:.4f} "
            f"<-> {self.event_b.agent_id}@{self.event_b.timestamp:.4f}, "
            f"gap={self.coordination_gap!r})"
        )


# ---------------------------------------------------------------------------
# Race condition
# ---------------------------------------------------------------------------

@dataclass
class RaceCondition:
    """The safety-violating predicate that distinguishes racing schedules.

    A race condition describes *what* goes wrong: the joint predicate that
    is satisfied under some schedule permutations but not others.

    Attributes:
        predicate_name: Human-readable name of the violated predicate.
        violated_agents: Agents whose joint behaviour triggers the violation.
        violated_state: Example joint state at which the violation occurs.
        robustness: Quantitative distance to the satisfaction boundary
            (negative ⇒ violated).
        description: Free-text explanation.
    """
    predicate_name: str
    violated_agents: List[str] = field(default_factory=list)
    violated_state: Optional[Dict[str, np.ndarray]] = None
    robustness: float = 0.0
    description: str = ""

    def severity_estimate(self) -> float:
        """Heuristic severity in [0, 1].  Larger magnitude of negative
        robustness ⇒ higher severity."""
        if self.robustness >= 0:
            return 0.0
        return min(1.0, abs(self.robustness) / 10.0)

    def __repr__(self) -> str:
        return (
            f"RaceCondition({self.predicate_name!r}, "
            f"agents={self.violated_agents}, rob={self.robustness:.4f})"
        )


# ---------------------------------------------------------------------------
# Race witness
# ---------------------------------------------------------------------------

@dataclass
class RaceWitness:
    """A concrete schedule demonstrating an interaction race.

    The witness consists of two schedules (``schedule_safe`` and
    ``schedule_unsafe``) over the same set of events that differ only
    in event ordering.  ``schedule_unsafe`` leads to a safety violation.

    Attributes:
        schedule_safe: Event ordering that satisfies the safety predicate.
        schedule_unsafe: Event ordering that violates the safety predicate.
        divergence_point: Index at which the two schedules first diverge.
        unsafe_state: Joint state at the point of violation.
        safe_state: Joint state at the corresponding point in the safe schedule.
    """
    schedule_safe: List[ScheduleEvent] = field(default_factory=list)
    schedule_unsafe: List[ScheduleEvent] = field(default_factory=list)
    divergence_point: int = 0
    unsafe_state: Optional[Dict[str, np.ndarray]] = None
    safe_state: Optional[Dict[str, np.ndarray]] = None

    @property
    def num_events(self) -> int:
        return len(self.schedule_unsafe)

    @property
    def reordered_agents(self) -> Set[str]:
        """Agents whose events are reordered between the two schedules."""
        agents: Set[str] = set()
        for i in range(min(len(self.schedule_safe), len(self.schedule_unsafe))):
            if self.schedule_safe[i].agent_id != self.schedule_unsafe[i].agent_id:
                agents.add(self.schedule_safe[i].agent_id)
                agents.add(self.schedule_unsafe[i].agent_id)
        return agents

    def __repr__(self) -> str:
        return (
            f"RaceWitness(events={self.num_events}, "
            f"divergence={self.divergence_point})"
        )


# ---------------------------------------------------------------------------
# Race absence certificate
# ---------------------------------------------------------------------------

@dataclass
class RaceAbsence:
    """Proof certificate that no race exists in a region of state space.

    Attributes:
        region_center: Centre of the certified region.
        region_radius: Radius of the certified region (ε).
        predicate_name: Safety predicate that was verified.
        method: Verification method (e.g. ``"zonotope_abstract_interp"``).
        lipschitz_constant: Lipschitz constant used in the proof.
        margin: Safety margin within the region.
        verified: Whether the certificate was verified.
    """
    region_center: Optional[np.ndarray] = None
    region_radius: float = 0.0
    predicate_name: str = ""
    method: str = ""
    lipschitz_constant: float = 1.0
    margin: float = 0.0
    verified: bool = False

    @property
    def certified_volume(self) -> float:
        """Approximate volume of the certified region (hypersphere)."""
        if self.region_center is None:
            return 0.0
        n = len(self.region_center)
        from math import pi, gamma
        return (pi ** (n / 2) / gamma(n / 2 + 1)) * self.region_radius ** n

    def __repr__(self) -> str:
        return (
            f"RaceAbsence(pred={self.predicate_name!r}, "
            f"radius={self.region_radius:.6f}, verified={self.verified})"
        )


# ---------------------------------------------------------------------------
# Interaction race
# ---------------------------------------------------------------------------

@dataclass
class InteractionRace:
    """An interaction race between two or more agents.

    An interaction race is a situation where two (or more) events from
    different agents can be reordered under a feasible schedule change,
    and the reordering causes a safety predicate to transition from
    satisfied to violated.

    Attributes:
        race_id: Unique identifier.
        events: Pair of events that constitute the race.
        agents: Agents involved.
        condition: The safety-violating predicate.
        hb_inconsistency: The missing HB ordering.
        witness: Concrete witness schedules.
        absence: Absence certificate (if no race exists).
        classification: Race type.
        probability: Probability estimate of the race occurring.
        schedule_window: Time window within which the race can occur.
        metadata: Additional metadata.
    """
    race_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    events: Tuple[ScheduleEvent, ...] = field(default_factory=tuple)
    agents: List[str] = field(default_factory=list)
    condition: Optional[RaceCondition] = None
    hb_inconsistency: Optional[HBInconsistency] = None
    witness: Optional[RaceWitness] = None
    absence: Optional[RaceAbsence] = None
    classification: RaceClassification = RaceClassification.CUSTOM
    probability: float = 0.0
    schedule_window: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        """Whether the race has been confirmed by a witness."""
        return self.witness is not None

    @property
    def is_certified_absent(self) -> bool:
        """Whether a race-absence certificate exists and is verified."""
        return self.absence is not None and self.absence.verified

    @property
    def severity(self) -> float:
        """Heuristic severity score in [0, 1]."""
        if self.condition is None:
            return 0.0
        base = self.condition.severity_estimate()
        # Scale by probability
        return min(1.0, base * (1.0 + self.probability))

    def involves_agent(self, agent_id: str) -> bool:
        return agent_id in self.agents

    def summary(self) -> str:
        """One-line summary of the race."""
        agents_str = ", ".join(self.agents)
        cond_str = self.condition.predicate_name if self.condition else "unknown"
        return (
            f"Race {self.race_id}: [{agents_str}] "
            f"{self.classification.value} — {cond_str} "
            f"(p={self.probability:.4f}, sev={self.severity:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "race_id": self.race_id,
            "agents": self.agents,
            "classification": self.classification.value,
            "probability": self.probability,
            "schedule_window": self.schedule_window,
            "severity": self.severity,
            "confirmed": self.is_confirmed,
        }
        if self.condition:
            result["condition"] = {
                "predicate": self.condition.predicate_name,
                "robustness": self.condition.robustness,
                "description": self.condition.description,
            }
        if self.hb_inconsistency:
            result["hb_inconsistency"] = {
                "agents": list(self.hb_inconsistency.agents),
                "time_window": self.hb_inconsistency.time_window,
                "gap": self.hb_inconsistency.coordination_gap,
            }
        result["metadata"] = self.metadata
        return result

    def __repr__(self) -> str:
        return self.summary()
