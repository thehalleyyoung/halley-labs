"""Execution trace data structures for multi-agent systems.

An *ExecutionTrace* is a totally-ordered log of events enriched with
vector-clock annotations.  A *MultiAgentTrace* merges per-agent
traces into a single joint trace that preserves the partial order
defined by the vector clocks.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from .events import (
    Event,
    EventType,
    VectorClock,
    vc_concurrent,
    vc_leq,
    vc_merge,
    vc_strictly_less,
)


# ---------------------------------------------------------------------------
# TraceSegment — a contiguous window of events
# ---------------------------------------------------------------------------
@dataclass
class TraceSegment:
    """A contiguous sub-sequence of events from a trace.

    Used for windowed analysis where only a section of the full trace
    is of interest.
    """

    events: List[Event]
    start_time: float
    end_time: float
    source_trace_id: str = ""

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    @property
    def agent_ids(self) -> Set[str]:
        return {e.agent_id for e in self.events}

    def filter_by_type(self, event_type: EventType) -> "TraceSegment":
        filtered = [e for e in self.events if e.event_type == event_type]
        return TraceSegment(
            events=filtered,
            start_time=self.start_time,
            end_time=self.end_time,
            source_trace_id=self.source_trace_id,
        )


# ---------------------------------------------------------------------------
# ExecutionTrace
# ---------------------------------------------------------------------------
class ExecutionTrace:
    """Ordered log of events annotated with vector clocks.

    Events are stored in insertion order.  The caller is responsible for
    appending them in a sensible total order (e.g. wall-clock time).
    """

    def __init__(self, trace_id: str = "", agents: Optional[Sequence[str]] = None):
        self.trace_id = trace_id
        self._events: List[Event] = []
        self._agents: Set[str] = set(agents) if agents else set()
        self._index_by_agent: Dict[str, List[int]] = defaultdict(list)
        self._index_by_type: Dict[EventType, List[int]] = defaultdict(list)
        self._id_to_index: Dict[str, int] = {}

    # ---- properties -------------------------------------------------------
    @property
    def agents(self) -> Set[str]:
        return set(self._agents)

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, idx: int) -> Event:
        return self._events[idx]

    # ---- mutation ----------------------------------------------------------
    def append_event(self, event: Event) -> None:
        """Append *event* and update internal indices."""
        idx = len(self._events)
        self._events.append(event)
        self._agents.add(event.agent_id)
        self._index_by_agent[event.agent_id].append(idx)
        self._index_by_type[event.event_type].append(idx)
        self._id_to_index[event.event_id] = idx

    def extend(self, events: Iterable[Event]) -> None:
        for e in events:
            self.append_event(e)

    # ---- queries -----------------------------------------------------------
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        idx = self._id_to_index.get(event_id)
        if idx is None:
            return None
        return self._events[idx]

    def get_agent_events(self, agent_id: str) -> List[Event]:
        """Return all events owned by *agent_id*, in trace order."""
        return [self._events[i] for i in self._index_by_agent.get(agent_id, [])]

    def filter_by_type(self, event_type: EventType) -> List[Event]:
        return [self._events[i] for i in self._index_by_type.get(event_type, [])]

    def get_events_in_window(
        self, start: float, end: float
    ) -> TraceSegment:
        """Return a TraceSegment for events with ``start <= timestamp <= end``."""
        seg = [e for e in self._events if start <= e.timestamp <= end]
        return TraceSegment(
            events=seg,
            start_time=start,
            end_time=end,
            source_trace_id=self.trace_id,
        )

    def slice_by_time(self, start: float, end: float) -> "ExecutionTrace":
        """Return a new ExecutionTrace containing only events in [start, end]."""
        sub = ExecutionTrace(trace_id=f"{self.trace_id}[{start:.4f},{end:.4f}]")
        for e in self._events:
            if start <= e.timestamp <= end:
                sub.append_event(e)
        return sub

    def get_concurrent_events(self) -> List[Tuple[Event, Event]]:
        """Return all pairs of events that are concurrent (unordered by HB).

        Complexity: O(n²) — use on moderate-length traces or segments.
        """
        pairs: List[Tuple[Event, Event]] = []
        n = len(self._events)
        for i in range(n):
            for j in range(i + 1, n):
                if vc_concurrent(
                    self._events[i].vector_clock,
                    self._events[j].vector_clock,
                ):
                    pairs.append((self._events[i], self._events[j]))
        return pairs

    def get_concurrent_events_for(self, event: Event) -> List[Event]:
        """Return all events concurrent with *event*."""
        return [
            e
            for e in self._events
            if e.event_id != event.event_id
            and vc_concurrent(e.vector_clock, event.vector_clock)
        ]

    def causal_predecessors_transitive(self, event_id: str) -> Set[str]:
        """BFS over ``causal_predecessors`` to get the transitive closure."""
        visited: Set[str] = set()
        frontier = [event_id]
        while frontier:
            eid = frontier.pop()
            ev = self.get_event_by_id(eid)
            if ev is None:
                continue
            for pred in ev.causal_predecessors:
                if pred not in visited:
                    visited.add(pred)
                    frontier.append(pred)
        return visited

    # ---- serialisation helpers ---------------------------------------------
    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events]


# ---------------------------------------------------------------------------
# MultiAgentTrace — merging per-agent traces
# ---------------------------------------------------------------------------
class MultiAgentTrace:
    """Merge independently-recorded per-agent traces into a joint trace.

    The merged trace respects the partial order induced by vector clocks:
    if ``a`` happens-before ``b`` then ``a`` precedes ``b`` in the merged
    total order.  Among concurrent events the merge is stable (preserves
    the relative order within each agent's local trace).
    """

    def __init__(self, trace_id: str = ""):
        self.trace_id = trace_id
        self._per_agent: Dict[str, ExecutionTrace] = {}
        self._merged: Optional[ExecutionTrace] = None

    def add_agent_trace(self, agent_id: str, trace: ExecutionTrace) -> None:
        self._per_agent[agent_id] = trace
        self._merged = None  # invalidate cache

    @property
    def agent_ids(self) -> Set[str]:
        return set(self._per_agent.keys())

    def get_agent_trace(self, agent_id: str) -> ExecutionTrace:
        return self._per_agent[agent_id]

    # ---- merge -------------------------------------------------------------
    def merged_trace(self) -> ExecutionTrace:
        """Return the HB-consistent merged trace (cached after first call)."""
        if self._merged is not None:
            return self._merged

        all_events: List[Event] = []
        for trace in self._per_agent.values():
            all_events.extend(trace.events)

        sorted_events = self._topological_sort(all_events)

        merged = ExecutionTrace(trace_id=self.trace_id)
        merged.extend(sorted_events)
        self._merged = merged
        return merged

    @staticmethod
    def _topological_sort(events: List[Event]) -> List[Event]:
        """Kahn's algorithm on the HB partial order.

        Ties (concurrent events) are broken by (timestamp, agent_id) to
        give a deterministic, stable total order.
        """
        id_map: Dict[str, Event] = {e.event_id: e for e in events}
        ids = set(id_map.keys())

        # Build adjacency from causal_predecessors
        in_degree: Dict[str, int] = {eid: 0 for eid in ids}
        successors: Dict[str, List[str]] = {eid: [] for eid in ids}

        for e in events:
            for pred_id in e.causal_predecessors:
                if pred_id in ids:
                    successors[pred_id].append(e.event_id)
                    in_degree[e.event_id] += 1

        # Additionally enforce HB from vector clocks for events that
        # didn't explicitly list causal_predecessors.  We only add edges
        # that aren't already implied by the explicit predecessor set.
        event_list = list(events)
        n = len(event_list)
        for i in range(n):
            for j in range(i + 1, n):
                ei, ej = event_list[i], event_list[j]
                if ei.event_id in ej.causal_predecessors:
                    continue
                if ej.event_id in ei.causal_predecessors:
                    continue
                if vc_strictly_less(ei.vector_clock, ej.vector_clock):
                    successors[ei.event_id].append(ej.event_id)
                    in_degree[ej.event_id] += 1
                elif vc_strictly_less(ej.vector_clock, ei.vector_clock):
                    successors[ej.event_id].append(ei.event_id)
                    in_degree[ei.event_id] += 1

        import heapq

        # Priority: (timestamp, agent_id, event_id) for deterministic tie-break
        ready: List[Tuple[float, str, str]] = []
        for eid, deg in in_degree.items():
            if deg == 0:
                e = id_map[eid]
                heapq.heappush(ready, (e.timestamp, e.agent_id, eid))

        result: List[Event] = []
        while ready:
            _, _, eid = heapq.heappop(ready)
            result.append(id_map[eid])
            for succ in successors[eid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    se = id_map[succ]
                    heapq.heappush(ready, (se.timestamp, se.agent_id, succ))

        if len(result) != len(events):
            raise ValueError(
                "Cycle detected in causal ordering: "
                f"sorted {len(result)} of {len(events)} events"
            )
        return result

    def interaction_pairs(self) -> List[Tuple[str, str, int]]:
        """Return (agent_a, agent_b, count) of causal interaction edges."""
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        merged = self.merged_trace()
        for e in merged:
            for pred_id in e.causal_predecessors:
                pred = merged.get_event_by_id(pred_id)
                if pred is not None and pred.agent_id != e.agent_id:
                    key = (pred.agent_id, e.agent_id)
                    counts[key] += 1
        return [(a, b, c) for (a, b), c in sorted(counts.items())]


# ---------------------------------------------------------------------------
# TraceStatistics
# ---------------------------------------------------------------------------
@dataclass
class TraceStatistics:
    """Aggregate statistics about an execution trace."""

    total_events: int = 0
    events_per_agent: Dict[str, int] = field(default_factory=dict)
    events_per_type: Dict[str, int] = field(default_factory=dict)
    concurrency_degree: float = 0.0
    interaction_frequency: float = 0.0
    trace_duration: float = 0.0
    event_density: float = 0.0  # events per second
    max_vector_clock_skew: int = 0

    @classmethod
    def compute(cls, trace: ExecutionTrace) -> "TraceStatistics":
        """Compute statistics from *trace*."""
        n = len(trace)
        if n == 0:
            return cls()

        events_per_agent: Dict[str, int] = defaultdict(int)
        events_per_type: Dict[str, int] = defaultdict(int)
        for e in trace:
            events_per_agent[e.agent_id] += 1
            events_per_type[e.event_type.name] += 1

        timestamps = [e.timestamp for e in trace]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        density = n / duration if duration > 0 else float(n)

        # Concurrency degree: fraction of all event pairs that are concurrent
        concurrent_count = 0
        total_pairs = 0
        events_list = trace.events
        # Sample if trace is large to keep O(n²) manageable
        sample_size = min(n, 500)
        if n <= sample_size:
            sample = events_list
        else:
            rng = np.random.RandomState(42)
            indices = rng.choice(n, size=sample_size, replace=False)
            sample = [events_list[i] for i in sorted(indices)]

        sn = len(sample)
        for i in range(sn):
            for j in range(i + 1, sn):
                total_pairs += 1
                if vc_concurrent(sample[i].vector_clock, sample[j].vector_clock):
                    concurrent_count += 1
        concurrency_degree = concurrent_count / total_pairs if total_pairs > 0 else 0.0

        # Interaction frequency: cross-agent causal edges per event
        cross_agent_edges = 0
        for e in trace:
            for pred_id in e.causal_predecessors:
                pred = trace.get_event_by_id(pred_id)
                if pred is not None and pred.agent_id != e.agent_id:
                    cross_agent_edges += 1
        interaction_freq = cross_agent_edges / n if n > 0 else 0.0

        # Max VC skew: largest difference between any two agents' components
        max_skew = 0
        for e in trace:
            vals = list(e.vector_clock.values())
            if len(vals) >= 2:
                skew = max(vals) - min(vals)
                if skew > max_skew:
                    max_skew = skew

        return cls(
            total_events=n,
            events_per_agent=dict(events_per_agent),
            events_per_type=dict(events_per_type),
            concurrency_degree=concurrency_degree,
            interaction_frequency=interaction_freq,
            trace_duration=duration,
            event_density=density,
            max_vector_clock_skew=max_skew,
        )


# ---------------------------------------------------------------------------
# TraceValidator
# ---------------------------------------------------------------------------
class TraceValidationResult:
    """Result of trace validation."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        return (
            f"TraceValidationResult(valid={self.is_valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


class TraceValidator:
    """Validate structural and causal consistency of an execution trace."""

    @staticmethod
    def validate(trace: ExecutionTrace) -> TraceValidationResult:
        result = TraceValidationResult()

        seen_ids: Set[str] = set()
        for e in trace:
            # Unique IDs
            if e.event_id in seen_ids:
                result.errors.append(f"Duplicate event_id: {e.event_id}")
            seen_ids.add(e.event_id)

        # Causal predecessors reference valid events
        all_ids = {e.event_id for e in trace}
        for e in trace:
            for pred in e.causal_predecessors:
                if pred not in all_ids:
                    result.warnings.append(
                        f"Event {e.event_id}: causal predecessor {pred} not in trace"
                    )

        # Vector clock consistency: if a -> b via causal_predecessors,
        # then vc(a) < vc(b) must hold
        id_map = {e.event_id: e for e in trace}
        for e in trace:
            for pred_id in e.causal_predecessors:
                pred = id_map.get(pred_id)
                if pred is None:
                    continue
                if not vc_leq(pred.vector_clock, e.vector_clock):
                    result.errors.append(
                        f"VC inconsistency: {pred_id} is predecessor of "
                        f"{e.event_id} but vc({pred_id}) ≰ vc({e.event_id})"
                    )

        # Per-agent monotonicity: within each agent, vector clock's own
        # component should be non-decreasing
        for agent_id in trace.agents:
            agent_events = trace.get_agent_events(agent_id)
            prev_component = -1
            for ae in agent_events:
                comp = ae.vector_clock.get(agent_id, 0)
                if comp < prev_component:
                    result.errors.append(
                        f"Agent {agent_id}: VC component decreased from "
                        f"{prev_component} to {comp} at event {ae.event_id}"
                    )
                prev_component = comp

        # Timestamp monotonicity within each agent
        for agent_id in trace.agents:
            agent_events = trace.get_agent_events(agent_id)
            for i in range(1, len(agent_events)):
                if agent_events[i].timestamp < agent_events[i - 1].timestamp:
                    result.warnings.append(
                        f"Agent {agent_id}: timestamp decreased between events "
                        f"{agent_events[i - 1].event_id} and {agent_events[i].event_id}"
                    )

        return result
