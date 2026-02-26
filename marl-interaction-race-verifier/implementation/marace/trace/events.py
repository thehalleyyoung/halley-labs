"""Event types for multi-agent system execution traces.

Defines the core event taxonomy used throughout MARACE:
  ACTION        — an agent selects and executes an action
  OBSERVATION   — an agent receives an observation
  COMMUNICATION — directed message between agents
  ENVIRONMENT   — environment dynamics step / state change
  SYNC          — synchronization barrier across agents

Every event carries a vector-clock stamp so that happens-before
(HB) relations can be recovered without a global clock.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vector clock type alias
# ---------------------------------------------------------------------------
VectorClock = Dict[str, int]
"""Mapping from agent_id → logical timestamp component."""


def vc_zero(agents: Sequence[str]) -> VectorClock:
    """Return a fresh zero vector clock for the given agent set."""
    return {a: 0 for a in agents}


def vc_increment(vc: VectorClock, agent_id: str) -> VectorClock:
    """Return a new vector clock with *agent_id*'s component incremented."""
    out = dict(vc)
    out[agent_id] = out.get(agent_id, 0) + 1
    return out


def vc_merge(a: VectorClock, b: VectorClock) -> VectorClock:
    """Component-wise max of two vector clocks."""
    keys = set(a) | set(b)
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in keys}


def vc_leq(a: VectorClock, b: VectorClock) -> bool:
    """True iff *a* ≤ *b* component-wise (a happens-before or equals b)."""
    for k in set(a) | set(b):
        if a.get(k, 0) > b.get(k, 0):
            return False
    return True


def vc_strictly_less(a: VectorClock, b: VectorClock) -> bool:
    """True iff *a* < *b* (a happens strictly before b)."""
    return vc_leq(a, b) and a != b


def vc_concurrent(a: VectorClock, b: VectorClock) -> bool:
    """True iff neither *a* ≤ *b* nor *b* ≤ *a*."""
    return not vc_leq(a, b) and not vc_leq(b, a)


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------
class EventType(Enum):
    """Discriminator for the five event categories."""
    ACTION = auto()
    OBSERVATION = auto()
    COMMUNICATION = auto()
    ENVIRONMENT = auto()
    SYNC = auto()


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------
@dataclass(frozen=False)
class Event:
    """Base event recorded in an execution trace.

    Parameters
    ----------
    event_id : str
        Globally unique identifier (UUID4 by default).
    agent_id : str
        Identifier of the agent that *owns* this event.  For environment
        events the convention is ``"__env__"``.
    timestamp : float
        Wall-clock time (seconds since trace start).  Used only for
        human-readable ordering; the vector clock is authoritative.
    event_type : EventType
        Discriminator tag.
    data : dict
        Arbitrary JSON-serialisable payload.
    vector_clock : VectorClock
        HB stamp at the point this event was recorded.
    causal_predecessors : frozenset[str]
        Set of event_ids that *directly* precede this event in the
        causal partial order (may be empty for the first events).
    """

    agent_id: str
    timestamp: float
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    vector_clock: VectorClock = field(default_factory=dict)
    causal_predecessors: FrozenSet[str] = field(default_factory=frozenset)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # ---- helpers ----------------------------------------------------------
    def happens_before(self, other: "Event") -> bool:
        """True iff ``self`` happens-before ``other`` per vector clocks."""
        return vc_strictly_less(self.vector_clock, other.vector_clock)

    def is_concurrent_with(self, other: "Event") -> bool:
        return vc_concurrent(self.vector_clock, other.vector_clock)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (numpy arrays → lists)."""
        d: Dict[str, Any] = {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.name,
            "data": _serialise_data(self.data),
            "vector_clock": dict(self.vector_clock),
            "causal_predecessors": sorted(self.causal_predecessors),
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Deserialise from a plain dict produced by ``to_dict``."""
        return cls(
            event_id=d["event_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            event_type=EventType[d["event_type"]],
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
        )

    def __hash__(self) -> int:
        return hash(self.event_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self.event_id == other.event_id


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=False)
class ActionEvent(Event):
    """An agent chooses and executes an action.

    ``action_vector`` is the raw action (continuous or one-hot encoded).
    ``state_before`` / ``state_after`` capture the local state delta.
    """

    action_vector: Optional[np.ndarray] = field(default=None, repr=False)
    state_before: Optional[np.ndarray] = field(default=None, repr=False)
    state_after: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.event_type is not EventType.ACTION:
            object.__setattr__(self, "event_type", EventType.ACTION)

    def action_magnitude(self) -> float:
        """L2 norm of the action vector."""
        if self.action_vector is None:
            return 0.0
        return float(np.linalg.norm(self.action_vector))

    def state_delta_norm(self) -> float:
        """L2 norm of state change caused by this action."""
        if self.state_before is None or self.state_after is None:
            return 0.0
        return float(np.linalg.norm(self.state_after - self.state_before))

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["action_vector"] = (
            self.action_vector.tolist() if self.action_vector is not None else None
        )
        d["state_before"] = (
            self.state_before.tolist() if self.state_before is not None else None
        )
        d["state_after"] = (
            self.state_after.tolist() if self.state_after is not None else None
        )
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionEvent":
        av = np.asarray(d["action_vector"]) if d.get("action_vector") is not None else None
        sb = np.asarray(d["state_before"]) if d.get("state_before") is not None else None
        sa = np.asarray(d["state_after"]) if d.get("state_after") is not None else None
        return cls(
            event_id=d["event_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            event_type=EventType.ACTION,
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
            action_vector=av,
            state_before=sb,
            state_after=sa,
        )


@dataclass(frozen=False)
class ObservationEvent(Event):
    """An agent receives an observation from the environment.

    ``source_agents`` lists which other agents' actions contributed
    to the information encoded in this observation (causal attribution).
    """

    observation_vector: Optional[np.ndarray] = field(default=None, repr=False)
    source_agents: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.event_type is not EventType.OBSERVATION:
            object.__setattr__(self, "event_type", EventType.OBSERVATION)

    def observation_norm(self) -> float:
        if self.observation_vector is None:
            return 0.0
        return float(np.linalg.norm(self.observation_vector))

    def is_multi_source(self) -> bool:
        """True if observation was shaped by more than one agent."""
        return len(self.source_agents) > 1

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["observation_vector"] = (
            self.observation_vector.tolist()
            if self.observation_vector is not None
            else None
        )
        d["source_agents"] = list(self.source_agents)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObservationEvent":
        ov = (
            np.asarray(d["observation_vector"])
            if d.get("observation_vector") is not None
            else None
        )
        return cls(
            event_id=d["event_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            event_type=EventType.OBSERVATION,
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
            observation_vector=ov,
            source_agents=d.get("source_agents", []),
        )


@dataclass(frozen=False)
class CommunicationEvent(Event):
    """A directed message between two agents.

    ``sender`` and ``receiver`` are agent ids.  ``channel_id`` allows
    multiplexing several logical channels (e.g. broadcast vs. private).
    """

    sender: str = ""
    receiver: str = ""
    message_payload: Dict[str, Any] = field(default_factory=dict)
    channel_id: str = "default"

    def __post_init__(self) -> None:
        if self.event_type is not EventType.COMMUNICATION:
            object.__setattr__(self, "event_type", EventType.COMMUNICATION)
        if not self.sender:
            object.__setattr__(self, "sender", self.agent_id)

    def message_size(self) -> int:
        """Rough byte size of the serialised payload."""
        import json
        return len(json.dumps(self.message_payload, default=str).encode())

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["sender"] = self.sender
        d["receiver"] = self.receiver
        d["message_payload"] = _serialise_data(self.message_payload)
        d["channel_id"] = self.channel_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunicationEvent":
        return cls(
            event_id=d["event_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            event_type=EventType.COMMUNICATION,
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
            sender=d.get("sender", ""),
            receiver=d.get("receiver", ""),
            message_payload=d.get("message_payload", {}),
            channel_id=d.get("channel_id", "default"),
        )


@dataclass(frozen=False)
class EnvironmentEvent(Event):
    """A state change caused by environment dynamics (not any single agent).

    ``env_state_delta`` is a numpy array encoding the change.
    ``affected_agents`` lists agents whose future observations are impacted.
    """

    env_state_delta: Optional[np.ndarray] = field(default=None, repr=False)
    affected_agents: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.event_type is not EventType.ENVIRONMENT:
            object.__setattr__(self, "event_type", EventType.ENVIRONMENT)
        if not self.agent_id:
            object.__setattr__(self, "agent_id", "__env__")

    def delta_magnitude(self) -> float:
        if self.env_state_delta is None:
            return 0.0
        return float(np.linalg.norm(self.env_state_delta))

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["env_state_delta"] = (
            self.env_state_delta.tolist()
            if self.env_state_delta is not None
            else None
        )
        d["affected_agents"] = list(self.affected_agents)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentEvent":
        delta = (
            np.asarray(d["env_state_delta"])
            if d.get("env_state_delta") is not None
            else None
        )
        return cls(
            event_id=d["event_id"],
            agent_id=d.get("agent_id", "__env__"),
            timestamp=d["timestamp"],
            event_type=EventType.ENVIRONMENT,
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
            env_state_delta=delta,
            affected_agents=d.get("affected_agents", []),
        )


@dataclass(frozen=False)
class SyncEvent(Event):
    """Synchronization barrier.

    All agents listed in ``barrier_agents`` must have reached this barrier
    before any of them may proceed.  ``barrier_id`` groups matching barriers.
    """

    barrier_id: str = ""
    barrier_agents: List[str] = field(default_factory=list)
    arrived_clocks: Dict[str, VectorClock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type is not EventType.SYNC:
            object.__setattr__(self, "event_type", EventType.SYNC)

    def all_arrived(self) -> bool:
        return set(self.barrier_agents) == set(self.arrived_clocks.keys())

    def merged_clock(self) -> VectorClock:
        """Merge all arrived clocks into a single post-barrier clock."""
        result: VectorClock = {}
        for vc in self.arrived_clocks.values():
            result = vc_merge(result, vc)
        return result

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["barrier_id"] = self.barrier_id
        d["barrier_agents"] = list(self.barrier_agents)
        d["arrived_clocks"] = {
            k: dict(v) for k, v in self.arrived_clocks.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyncEvent":
        return cls(
            event_id=d["event_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            event_type=EventType.SYNC,
            data=d.get("data", {}),
            vector_clock=d.get("vector_clock", {}),
            causal_predecessors=frozenset(d.get("causal_predecessors", [])),
            barrier_id=d.get("barrier_id", ""),
            barrier_agents=d.get("barrier_agents", []),
            arrived_clocks=d.get("arrived_clocks", {}),
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
_SUBCLASS_MAP: Dict[EventType, type] = {
    EventType.ACTION: ActionEvent,
    EventType.OBSERVATION: ObservationEvent,
    EventType.COMMUNICATION: CommunicationEvent,
    EventType.ENVIRONMENT: EnvironmentEvent,
    EventType.SYNC: SyncEvent,
}


def event_from_dict(d: Dict[str, Any]) -> Event:
    """Polymorphic deserialisation: pick the right subclass."""
    et = EventType[d["event_type"]]
    subcls = _SUBCLASS_MAP.get(et)
    if subcls is not None:
        return subcls.from_dict(d)
    return Event.from_dict(d)


def make_action_event(
    agent_id: str,
    timestamp: float,
    action_vector: np.ndarray,
    vector_clock: VectorClock,
    *,
    state_before: Optional[np.ndarray] = None,
    state_after: Optional[np.ndarray] = None,
    causal_predecessors: FrozenSet[str] = frozenset(),
    data: Optional[Dict[str, Any]] = None,
) -> ActionEvent:
    return ActionEvent(
        agent_id=agent_id,
        timestamp=timestamp,
        event_type=EventType.ACTION,
        data=data or {},
        vector_clock=vector_clock,
        causal_predecessors=causal_predecessors,
        action_vector=np.asarray(action_vector, dtype=np.float64),
        state_before=(
            np.asarray(state_before, dtype=np.float64) if state_before is not None else None
        ),
        state_after=(
            np.asarray(state_after, dtype=np.float64) if state_after is not None else None
        ),
    )


def make_observation_event(
    agent_id: str,
    timestamp: float,
    observation_vector: np.ndarray,
    vector_clock: VectorClock,
    *,
    source_agents: Optional[List[str]] = None,
    causal_predecessors: FrozenSet[str] = frozenset(),
    data: Optional[Dict[str, Any]] = None,
) -> ObservationEvent:
    return ObservationEvent(
        agent_id=agent_id,
        timestamp=timestamp,
        event_type=EventType.OBSERVATION,
        data=data or {},
        vector_clock=vector_clock,
        causal_predecessors=causal_predecessors,
        observation_vector=np.asarray(observation_vector, dtype=np.float64),
        source_agents=source_agents or [],
    )


def make_communication_event(
    sender: str,
    receiver: str,
    timestamp: float,
    message_payload: Dict[str, Any],
    vector_clock: VectorClock,
    *,
    channel_id: str = "default",
    causal_predecessors: FrozenSet[str] = frozenset(),
    data: Optional[Dict[str, Any]] = None,
) -> CommunicationEvent:
    return CommunicationEvent(
        agent_id=sender,
        timestamp=timestamp,
        event_type=EventType.COMMUNICATION,
        data=data or {},
        vector_clock=vector_clock,
        causal_predecessors=causal_predecessors,
        sender=sender,
        receiver=receiver,
        message_payload=message_payload,
        channel_id=channel_id,
    )


def make_environment_event(
    timestamp: float,
    env_state_delta: np.ndarray,
    affected_agents: List[str],
    vector_clock: VectorClock,
    *,
    causal_predecessors: FrozenSet[str] = frozenset(),
    data: Optional[Dict[str, Any]] = None,
) -> EnvironmentEvent:
    return EnvironmentEvent(
        agent_id="__env__",
        timestamp=timestamp,
        event_type=EventType.ENVIRONMENT,
        data=data or {},
        vector_clock=vector_clock,
        causal_predecessors=causal_predecessors,
        env_state_delta=np.asarray(env_state_delta, dtype=np.float64),
        affected_agents=affected_agents,
    )


def make_sync_event(
    agent_id: str,
    timestamp: float,
    barrier_id: str,
    barrier_agents: List[str],
    vector_clock: VectorClock,
    *,
    arrived_clocks: Optional[Dict[str, VectorClock]] = None,
    causal_predecessors: FrozenSet[str] = frozenset(),
    data: Optional[Dict[str, Any]] = None,
) -> SyncEvent:
    return SyncEvent(
        agent_id=agent_id,
        timestamp=timestamp,
        event_type=EventType.SYNC,
        data=data or {},
        vector_clock=vector_clock,
        causal_predecessors=causal_predecessors,
        barrier_id=barrier_id,
        barrier_agents=barrier_agents,
        arrived_clocks=arrived_clocks or {},
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class EventValidationError(Exception):
    """Raised when an event fails structural validation."""


def validate_event(event: Event, known_agents: Optional[Set[str]] = None) -> List[str]:
    """Return a list of validation warnings (empty = OK).

    Raises ``EventValidationError`` for hard errors.
    """
    warnings: List[str] = []

    if not event.event_id:
        raise EventValidationError("event_id must be non-empty")
    if not event.agent_id:
        raise EventValidationError("agent_id must be non-empty")
    if event.timestamp < 0:
        raise EventValidationError(
            f"Negative timestamp {event.timestamp} on event {event.event_id}"
        )

    if not event.vector_clock:
        warnings.append(f"Event {event.event_id} has an empty vector clock")
    elif event.agent_id in event.vector_clock:
        if event.vector_clock[event.agent_id] < 1 and event.agent_id != "__env__":
            warnings.append(
                f"Event {event.event_id}: owning agent's VC component is 0"
            )

    if known_agents is not None:
        if event.agent_id not in known_agents and event.agent_id != "__env__":
            warnings.append(
                f"Event {event.event_id}: agent_id '{event.agent_id}' not in known set"
            )

    if isinstance(event, ActionEvent):
        if event.action_vector is not None and not np.isfinite(event.action_vector).all():
            warnings.append(f"ActionEvent {event.event_id}: action_vector has non-finite values")
        if event.state_before is not None and event.state_after is not None:
            if event.state_before.shape != event.state_after.shape:
                raise EventValidationError(
                    f"ActionEvent {event.event_id}: state_before and state_after shapes differ"
                )

    if isinstance(event, CommunicationEvent):
        if event.sender == event.receiver:
            warnings.append(f"CommunicationEvent {event.event_id}: sender == receiver")

    if isinstance(event, ObservationEvent):
        if event.observation_vector is not None and not np.isfinite(event.observation_vector).all():
            warnings.append(
                f"ObservationEvent {event.event_id}: observation_vector has non-finite values"
            )

    return warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _serialise_data(data: Any) -> Any:
    """Recursively convert numpy types to JSON-friendly primitives."""
    if isinstance(data, dict):
        return {k: _serialise_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_serialise_data(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    return data
