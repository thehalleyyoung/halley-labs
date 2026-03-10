"""
usability_oracle.simulation.event_queue — Priority event queue for discrete-event simulation.

Provides a min-heap based event queue with support for conditional events,
recurring events, event filtering, and dependency-based event ordering.

The queue is the core scheduling mechanism for the discrete-event cognitive
simulation engine, handling temporal ordering of perceptual, cognitive, and
motor events across multiple simulated processors.

References:
    Banks, J. et al. (2010). *Discrete-Event System Simulation* (5th ed.).
        Prentice Hall.
    Law, A. M. (2015). *Simulation Modeling and Analysis* (5th ed.).
        McGraw-Hill.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Event type taxonomy
# ═══════════════════════════════════════════════════════════════════════════

@unique
class EventType(Enum):
    """Taxonomy of simulation events following MHP/EPIC classification.

    References:
        Card, S. K., Moran, T. P., & Newell, A. (1983).
            *The Psychology of Human-Computer Interaction*. Erlbaum.
        Kieras, D. E., & Meyer, D. E. (1997). An overview of the EPIC
            architecture. *Cognitive Science*, 21(2), 135-183.
    """

    # Perceptual events
    VISUAL_ONSET = auto()
    VISUAL_ENCODING_COMPLETE = auto()
    VISUAL_ATTENTION_SHIFT = auto()
    AUDITORY_ONSET = auto()
    AUDITORY_ENCODING_COMPLETE = auto()

    # Cognitive events
    PRODUCTION_MATCH = auto()
    PRODUCTION_FIRE = auto()
    GOAL_PUSH = auto()
    GOAL_POP = auto()
    RETRIEVAL_REQUEST = auto()
    RETRIEVAL_COMPLETE = auto()
    RETRIEVAL_FAILURE = auto()

    # Motor events
    MOTOR_PREPARATION_START = auto()
    MOTOR_PREPARATION_COMPLETE = auto()
    MOTOR_EXECUTION_START = auto()
    MOTOR_EXECUTION_COMPLETE = auto()
    KEYSTROKE = auto()
    MOUSE_MOVE = auto()
    MOUSE_CLICK = auto()

    # Working memory events
    WM_STORE = auto()
    WM_DECAY_TICK = auto()
    WM_REHEARSAL = auto()
    WM_CHUNK_FORGOTTEN = auto()

    # Task-level events
    TASK_START = auto()
    TASK_COMPLETE = auto()
    SUBTASK_START = auto()
    SUBTASK_COMPLETE = auto()

    # System events
    SYSTEM_RESPONSE = auto()
    SYSTEM_FEEDBACK = auto()
    DISPLAY_UPDATE = auto()

    # Simulation control
    SIMULATION_START = auto()
    SIMULATION_END = auto()
    CHECKPOINT = auto()
    TIMEOUT = auto()

    @property
    def is_perceptual(self) -> bool:
        return self in {
            EventType.VISUAL_ONSET, EventType.VISUAL_ENCODING_COMPLETE,
            EventType.VISUAL_ATTENTION_SHIFT, EventType.AUDITORY_ONSET,
            EventType.AUDITORY_ENCODING_COMPLETE,
        }

    @property
    def is_cognitive(self) -> bool:
        return self in {
            EventType.PRODUCTION_MATCH, EventType.PRODUCTION_FIRE,
            EventType.GOAL_PUSH, EventType.GOAL_POP,
            EventType.RETRIEVAL_REQUEST, EventType.RETRIEVAL_COMPLETE,
            EventType.RETRIEVAL_FAILURE,
        }

    @property
    def is_motor(self) -> bool:
        return self in {
            EventType.MOTOR_PREPARATION_START, EventType.MOTOR_PREPARATION_COMPLETE,
            EventType.MOTOR_EXECUTION_START, EventType.MOTOR_EXECUTION_COMPLETE,
            EventType.KEYSTROKE, EventType.MOUSE_MOVE, EventType.MOUSE_CLICK,
        }

    @property
    def is_memory(self) -> bool:
        return self in {
            EventType.WM_STORE, EventType.WM_DECAY_TICK,
            EventType.WM_REHEARSAL, EventType.WM_CHUNK_FORGOTTEN,
        }

    @property
    def is_task(self) -> bool:
        return self in {
            EventType.TASK_START, EventType.TASK_COMPLETE,
            EventType.SUBTASK_START, EventType.SUBTASK_COMPLETE,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Event priority levels
# ═══════════════════════════════════════════════════════════════════════════

@unique
class EventPriority(Enum):
    """Priority levels for tie-breaking when events share a timestamp.

    Lower numeric value = higher priority.  This follows the EPIC architecture's
    rule that cognitive events take priority over motor when simultaneous.
    """
    CRITICAL = 0
    COGNITIVE = 1
    PERCEPTUAL = 2
    MOTOR = 3
    MEMORY = 4
    SYSTEM = 5
    TASK = 6
    LOW = 7

    @classmethod
    def from_event_type(cls, event_type: EventType) -> EventPriority:
        """Infer priority from event type."""
        if event_type.is_cognitive:
            return cls.COGNITIVE
        if event_type.is_perceptual:
            return cls.PERCEPTUAL
        if event_type.is_motor:
            return cls.MOTOR
        if event_type.is_memory:
            return cls.MEMORY
        if event_type.is_task:
            return cls.TASK
        return cls.SYSTEM


# ═══════════════════════════════════════════════════════════════════════════
# SimulationEvent
# ═══════════════════════════════════════════════════════════════════════════

_event_id_counter = itertools.count()


@dataclass(order=False)
class SimulationEvent:
    """A single event in the discrete-event simulation.

    Events are ordered first by timestamp, then by priority (lower = higher
    priority), then by insertion order (FIFO tie-breaking).

    Attributes:
        event_id: Unique event identifier (auto-assigned).
        timestamp: Simulated time at which this event fires (seconds).
        event_type: Classification of this event.
        source_processor: Name of the processor that generated this event.
        target_processor: Name of the processor that should handle this event.
        payload: Arbitrary data carried by the event.
        priority: Tie-breaking priority.
        cancelled: Whether this event has been cancelled.
        sequence_number: Insertion order for deterministic FIFO tie-breaking.
        dependencies: Set of event IDs that must complete before this fires.
        metadata: Additional metadata for tracing and debugging.
    """

    event_id: int = field(default_factory=lambda: next(_event_id_counter))
    timestamp: float = 0.0
    event_type: EventType = EventType.SIMULATION_START
    source_processor: str = ""
    target_processor: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.SYSTEM
    cancelled: bool = False
    sequence_number: int = 0
    dependencies: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority == EventPriority.SYSTEM and self.event_type != EventType.SIMULATION_START:
            self.priority = EventPriority.from_event_type(self.event_type)

    @property
    def sort_key(self) -> Tuple[float, int, int]:
        """Key for min-heap ordering: (timestamp, priority, sequence)."""
        return (self.timestamp, self.priority.value, self.sequence_number)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.sort_key <= other.sort_key

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.sort_key > other.sort_key

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.sort_key >= other.sort_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.event_id == other.event_id

    def __hash__(self) -> int:
        return hash(self.event_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for trace export."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.name,
            "source_processor": self.source_processor,
            "target_processor": self.target_processor,
            "payload": self.payload,
            "priority": self.priority.name,
            "cancelled": self.cancelled,
            "dependencies": list(self.dependencies),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ConditionalEvent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionalEvent:
    """An event that fires only when a boolean condition becomes true.

    The condition is re-evaluated after each simulation step.  If the
    condition holds, the inner event is inserted into the queue with a
    timestamp equal to the current simulation clock + ``delay``.

    Attributes:
        condition: Callable that receives the current simulation state and
            returns True when the event should fire.
        event_template: The event to insert when the condition is satisfied.
        delay: Additional delay (seconds) after the condition is met.
        max_evaluations: Safety limit on condition evaluations.
        evaluation_count: Number of times the condition has been checked.
        fired: Whether this conditional event has already fired.
    """

    condition: Callable[..., bool] = field(default=lambda: False)
    event_template: SimulationEvent = field(default_factory=SimulationEvent)
    delay: float = 0.0
    max_evaluations: int = 10_000
    evaluation_count: int = 0
    fired: bool = False

    def evaluate(self, sim_state: Any, current_time: float) -> Optional[SimulationEvent]:
        """Check condition and return event if satisfied, else None."""
        if self.fired:
            return None
        self.evaluation_count += 1
        if self.evaluation_count > self.max_evaluations:
            return None
        try:
            if self.condition(sim_state):
                self.fired = True
                event = SimulationEvent(
                    timestamp=current_time + self.delay,
                    event_type=self.event_template.event_type,
                    source_processor=self.event_template.source_processor,
                    target_processor=self.event_template.target_processor,
                    payload=dict(self.event_template.payload),
                    priority=self.event_template.priority,
                    metadata=dict(self.event_template.metadata),
                )
                return event
        except Exception:
            pass
        return None

    @property
    def is_expired(self) -> bool:
        return self.fired or self.evaluation_count >= self.max_evaluations


# ═══════════════════════════════════════════════════════════════════════════
# RecurringEvent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RecurringEvent:
    """A periodic event that fires at regular intervals.

    Used for processes like working memory rehearsal (roughly every 1-3 s,
    per Baddeley 2000), visual sampling, or periodic motor adjustments.

    Attributes:
        event_template: Base event to clone for each firing.
        interval: Period between firings (seconds).
        jitter: Random jitter magnitude (seconds); added uniformly in [-jitter, +jitter].
        start_time: First firing time.
        stop_time: Stop generating after this time (inf = never stop).
        max_firings: Maximum number of firings.
        firing_count: Number of times fired so far.
        active: Whether the recurring event is still active.
    """

    event_template: SimulationEvent = field(default_factory=SimulationEvent)
    interval: float = 1.0
    jitter: float = 0.0
    start_time: float = 0.0
    stop_time: float = float("inf")
    max_firings: int = 1_000_000
    firing_count: int = 0
    active: bool = True

    @property
    def next_firing_time(self) -> float:
        """Compute the next deterministic firing time (before jitter)."""
        return self.start_time + self.firing_count * self.interval

    def generate_next(self, rng_jitter: float = 0.0) -> Optional[SimulationEvent]:
        """Generate the next event instance, or None if exhausted."""
        if not self.active:
            return None
        if self.firing_count >= self.max_firings:
            self.active = False
            return None

        base_time = self.next_firing_time
        actual_time = base_time + rng_jitter
        if actual_time > self.stop_time:
            self.active = False
            return None

        self.firing_count += 1
        return SimulationEvent(
            timestamp=max(actual_time, 0.0),
            event_type=self.event_template.event_type,
            source_processor=self.event_template.source_processor,
            target_processor=self.event_template.target_processor,
            payload={**self.event_template.payload, "_firing": self.firing_count},
            priority=self.event_template.priority,
            metadata={**self.event_template.metadata, "_recurring": True},
        )

    def cancel(self) -> None:
        self.active = False


# ═══════════════════════════════════════════════════════════════════════════
# EventFilter
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EventFilter:
    """Filter events by type, source, target, or time window.

    Multiple criteria are combined with AND semantics.  An empty filter
    matches everything.

    Attributes:
        event_types: If non-empty, only match events with these types.
        source_processors: If non-empty, only match events from these sources.
        target_processors: If non-empty, only match events targeting these.
        time_min: Minimum timestamp (inclusive).
        time_max: Maximum timestamp (inclusive).
        exclude_cancelled: Whether to exclude cancelled events.
        payload_predicate: Optional callable(payload) -> bool for custom filtering.
    """

    event_types: FrozenSet[EventType] = frozenset()
    source_processors: FrozenSet[str] = frozenset()
    target_processors: FrozenSet[str] = frozenset()
    time_min: float = 0.0
    time_max: float = float("inf")
    exclude_cancelled: bool = True
    payload_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None

    def matches(self, event: SimulationEvent) -> bool:
        """Return True if the event passes all filter criteria."""
        if self.exclude_cancelled and event.cancelled:
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.source_processors and event.source_processor not in self.source_processors:
            return False
        if self.target_processors and event.target_processor not in self.target_processors:
            return False
        if event.timestamp < self.time_min or event.timestamp > self.time_max:
            return False
        if self.payload_predicate is not None:
            try:
                if not self.payload_predicate(event.payload):
                    return False
            except Exception:
                return False
        return True

    def filter_list(self, events: List[SimulationEvent]) -> List[SimulationEvent]:
        """Apply filter to a list of events."""
        return [e for e in events if self.matches(e)]

    @classmethod
    def by_type(cls, *event_types: EventType) -> EventFilter:
        """Create a filter matching specific event types."""
        return cls(event_types=frozenset(event_types))

    @classmethod
    def by_source(cls, *sources: str) -> EventFilter:
        """Create a filter matching specific source processors."""
        return cls(source_processors=frozenset(sources))

    @classmethod
    def by_time_window(cls, t_min: float, t_max: float) -> EventFilter:
        """Create a filter matching a time window."""
        return cls(time_min=t_min, time_max=t_max)

    @classmethod
    def cognitive_only(cls) -> EventFilter:
        """Filter for cognitive events only."""
        return cls(event_types=frozenset(
            et for et in EventType if et.is_cognitive
        ))

    @classmethod
    def motor_only(cls) -> EventFilter:
        """Filter for motor events only."""
        return cls(event_types=frozenset(
            et for et in EventType if et.is_motor
        ))


# ═══════════════════════════════════════════════════════════════════════════
# EventQueue — min-heap based priority queue
# ═══════════════════════════════════════════════════════════════════════════

class EventQueue:
    """Min-heap priority queue for discrete-event simulation.

    Events are ordered by (timestamp, priority, sequence_number) to ensure
    deterministic ordering.  Lazy deletion is used for cancelled events.

    This is the central scheduling data structure for the simulation engine,
    analogous to the Future Event List (FEL) in Banks et al. (2010).
    """

    def __init__(self) -> None:
        self._heap: List[SimulationEvent] = []
        self._sequence: int = 0
        self._event_index: Dict[int, SimulationEvent] = {}
        self._completed_events: Set[int] = set()
        self._size: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def insert(self, event: SimulationEvent) -> int:
        """Insert an event into the queue.

        Returns the event_id for later reference (cancellation, dependencies).
        """
        event.sequence_number = self._sequence
        self._sequence += 1
        self._event_index[event.event_id] = event
        heapq.heappush(self._heap, event)
        self._size += 1
        return event.event_id

    def pop(self) -> Optional[SimulationEvent]:
        """Remove and return the next (earliest) non-cancelled event.

        Uses lazy deletion: cancelled events encountered during pop are
        silently discarded.  Returns None if the queue is empty.
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.cancelled:
                self._size = max(0, self._size - 1)
                continue
            # Check unresolved dependencies
            if event.dependencies and not event.dependencies.issubset(self._completed_events):
                # Re-insert with a small time bump to re-check later
                event.timestamp += 0.0001
                heapq.heappush(self._heap, event)
                continue
            self._size = max(0, self._size - 1)
            self._completed_events.add(event.event_id)
            return event
        return None

    def peek(self) -> Optional[SimulationEvent]:
        """Return the next event without removing it.

        Skips cancelled events by scanning forward in the heap.
        """
        for event in self._heap:
            if not event.cancelled:
                return event
        return None

    def cancel(self, event_id: int) -> bool:
        """Cancel an event by its ID.

        Uses lazy deletion — the event remains in the heap but is marked
        as cancelled and will be skipped during ``pop()``.

        Returns True if the event was found and cancelled.
        """
        event = self._event_index.get(event_id)
        if event is None:
            return False
        if event.cancelled:
            return False
        event.cancelled = True
        self._size = max(0, self._size - 1)
        return True

    def cancel_by_filter(self, filt: EventFilter) -> int:
        """Cancel all events matching a filter. Returns count cancelled."""
        count = 0
        for event in self._event_index.values():
            if not event.cancelled and filt.matches(event):
                event.cancelled = True
                self._size = max(0, self._size - 1)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def mark_completed(self, event_id: int) -> None:
        """Mark an event as completed for dependency resolution."""
        self._completed_events.add(event_id)

    def add_dependency(self, dependent_id: int, prerequisite_id: int) -> bool:
        """Add a dependency: *dependent* cannot fire until *prerequisite* completes.

        Returns True if the dependency was successfully added.
        """
        event = self._event_index.get(dependent_id)
        if event is None:
            return False
        event.dependencies.add(prerequisite_id)
        return True

    def is_dependency_satisfied(self, event_id: int) -> bool:
        """Check whether all dependencies of an event are satisfied."""
        event = self._event_index.get(event_id)
        if event is None:
            return True
        return event.dependencies.issubset(self._completed_events)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def insert_batch(self, events: List[SimulationEvent]) -> List[int]:
        """Insert multiple events at once. Returns list of event IDs."""
        return [self.insert(e) for e in events]

    def drain(self, max_events: int = -1) -> List[SimulationEvent]:
        """Pop up to *max_events* events (or all if max_events < 0)."""
        result: List[SimulationEvent] = []
        count = 0
        while True:
            if 0 <= max_events <= count:
                break
            event = self.pop()
            if event is None:
                break
            result.append(event)
            count += 1
        return result

    def drain_until(self, time_limit: float) -> List[SimulationEvent]:
        """Pop all events with timestamp <= *time_limit*."""
        result: List[SimulationEvent] = []
        while True:
            event = self.peek()
            if event is None or event.timestamp > time_limit:
                break
            event = self.pop()
            if event is not None:
                result.append(event)
        return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_event(self, event_id: int) -> Optional[SimulationEvent]:
        """Look up an event by ID."""
        return self._event_index.get(event_id)

    def pending_events(self, filt: Optional[EventFilter] = None) -> List[SimulationEvent]:
        """Return all pending (non-cancelled) events, optionally filtered."""
        events = [e for e in self._event_index.values() if not e.cancelled and e.event_id not in self._completed_events]
        if filt is not None:
            events = filt.filter_list(events)
        return sorted(events, key=lambda e: e.sort_key)

    def next_event_time(self) -> Optional[float]:
        """Return the timestamp of the next non-cancelled event, or None."""
        event = self.peek()
        return event.timestamp if event is not None else None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Approximate number of pending events (excludes cancelled)."""
        return self._size

    @property
    def is_empty(self) -> bool:
        return self.size <= 0 or self.peek() is None

    @property
    def total_inserted(self) -> int:
        return self._sequence

    @property
    def total_completed(self) -> int:
        return len(self._completed_events)

    def clear(self) -> None:
        """Remove all events."""
        self._heap.clear()
        self._event_index.clear()
        self._completed_events.clear()
        self._size = 0
        self._sequence = 0

    def snapshot(self) -> List[Dict[str, Any]]:
        """Export all pending events as a list of dicts (for checkpointing)."""
        return [e.to_dict() for e in self.pending_events()]

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return not self.is_empty

    def __repr__(self) -> str:
        return f"EventQueue(size={self.size}, total_inserted={self.total_inserted})"
