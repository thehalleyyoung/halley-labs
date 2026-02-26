"""
Vector clock implementation for the MARACE happens-before engine.

Provides a dictionary-based vector clock with efficient increment, merge,
comparison, serialization, and batch operations for multi-agent systems.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


class ClockComparison(Enum):
    """Result of comparing two vector clocks."""
    BEFORE = auto()       # self < other
    AFTER = auto()        # self > other
    CONCURRENT = auto()   # neither ≤ nor ≥
    EQUAL = auto()        # identical


class VectorClock:
    """Dict-based vector clock mapping agent_id -> logical timestamp.

    Each agent in the system maintains its own component in the vector.
    The clock grows lazily: only agents that have been observed appear as keys.

    Attributes:
        _clock: Internal mapping from agent identifiers to integer timestamps.
    """

    __slots__ = ("_clock",)

    def __init__(self, initial: Optional[Dict[str, int]] = None) -> None:
        self._clock: Dict[str, int] = dict(initial) if initial else {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def increment(self, agent_id: str) -> "VectorClock":
        """Increment the component for *agent_id* and return self (for chaining).

        Args:
            agent_id: The agent whose logical clock is advanced.

        Returns:
            Self, to allow fluent chaining.
        """
        self._clock[agent_id] = self._clock.get(agent_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Point-wise maximum with *other*, mutating self in place.

        After merging, every component is at least as large as the
        corresponding component in *other*.

        Args:
            other: The vector clock to merge into this one.

        Returns:
            Self, for fluent chaining.
        """
        for agent_id, ts in other._clock.items():
            if ts > self._clock.get(agent_id, 0):
                self._clock[agent_id] = ts
        return self

    def get(self, agent_id: str) -> int:
        """Return the timestamp for *agent_id* (0 if absent)."""
        return self._clock.get(agent_id, 0)

    def set(self, agent_id: str, value: int) -> None:
        """Explicitly set the clock value for an agent."""
        if value < 0:
            raise ValueError(f"Clock value must be non-negative, got {value}")
        self._clock[agent_id] = value

    @property
    def agents(self) -> FrozenSet[str]:
        """Return the set of agents represented in this clock."""
        return frozenset(self._clock.keys())

    @property
    def max_timestamp(self) -> int:
        """Return the maximum timestamp across all agents (0 if empty)."""
        return max(self._clock.values()) if self._clock else 0

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def compare(self, other: "VectorClock") -> ClockComparison:
        """Full comparison yielding one of the four ordering relations.

        Implements the standard vector-clock partial order:
        - EQUAL   if ∀ a: self[a] == other[a]
        - BEFORE  if ∀ a: self[a] ≤ other[a]  and ∃ a: self[a] < other[a]
        - AFTER   if ∀ a: self[a] ≥ other[a]  and ∃ a: self[a] > other[a]
        - CONCURRENT otherwise
        """
        all_agents = self._clock.keys() | other._clock.keys()
        has_less = False
        has_greater = False

        for agent_id in all_agents:
            s = self._clock.get(agent_id, 0)
            o = other._clock.get(agent_id, 0)
            if s < o:
                has_less = True
            elif s > o:
                has_greater = True
            if has_less and has_greater:
                return ClockComparison.CONCURRENT

        if has_less and not has_greater:
            return ClockComparison.BEFORE
        if has_greater and not has_less:
            return ClockComparison.AFTER
        return ClockComparison.EQUAL

    def happens_before(self, other: "VectorClock") -> bool:
        """Return True iff *self* causally precedes *other* (self < other)."""
        return self.compare(other) is ClockComparison.BEFORE

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Return True iff *self* and *other* are causally unordered."""
        return self.compare(other) is ClockComparison.CONCURRENT

    def __le__(self, other: "VectorClock") -> bool:
        cmp = self.compare(other)
        return cmp is ClockComparison.BEFORE or cmp is ClockComparison.EQUAL

    def __lt__(self, other: "VectorClock") -> bool:
        return self.compare(other) is ClockComparison.BEFORE

    def __ge__(self, other: "VectorClock") -> bool:
        cmp = self.compare(other)
        return cmp is ClockComparison.AFTER or cmp is ClockComparison.EQUAL

    def __gt__(self, other: "VectorClock") -> bool:
        return self.compare(other) is ClockComparison.AFTER

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.compare(other) is ClockComparison.EQUAL

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._clock.items())))

    # ------------------------------------------------------------------
    # Clock difference / distance
    # ------------------------------------------------------------------

    def difference(self, other: "VectorClock") -> Dict[str, int]:
        """Per-agent signed difference: self[a] - other[a].

        Useful for timing analysis — positive values indicate how far
        ahead *self* is for a given agent relative to *other*.

        Args:
            other: The clock to subtract.

        Returns:
            Dict mapping agent_id to (self[a] - other[a]).
        """
        all_agents = self._clock.keys() | other._clock.keys()
        return {
            a: self._clock.get(a, 0) - other._clock.get(a, 0)
            for a in all_agents
        }

    def l1_distance(self, other: "VectorClock") -> int:
        """L1 (Manhattan) distance between two clocks.

        Provides a scalar notion of "how different" two clocks are,
        useful for grouping and outlier detection.
        """
        diff = self.difference(other)
        return sum(abs(v) for v in diff.values())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, int]:
        """Return a plain-dict copy of the clock."""
        return dict(self._clock)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        """Reconstruct a VectorClock from a plain dict."""
        return cls(initial=data)

    def to_json(self) -> str:
        """Serialize to a compact JSON string."""
        return json.dumps(self._clock, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "VectorClock":
        """Deserialize from JSON."""
        return cls(initial=json.loads(raw))

    def copy(self) -> "VectorClock":
        """Return a deep copy."""
        return VectorClock(initial=dict(self._clock))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v}" for k, v in sorted(self._clock.items()))
        return f"VectorClock({{{items}}})"

    def __len__(self) -> int:
        return len(self._clock)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._clock

    def __iter__(self):
        return iter(self._clock)


# ======================================================================
# Batch comparison helpers
# ======================================================================

def _pairwise_compare(
    clocks: List[VectorClock],
) -> List[List[ClockComparison]]:
    """O(n²·d) pairwise comparison of *n* clocks with *d* total agents.

    Returns a 2-D list where result[i][j] == clocks[i].compare(clocks[j]).
    The diagonal is always EQUAL.
    """
    n = len(clocks)
    result: List[List[ClockComparison]] = [
        [ClockComparison.EQUAL] * n for _ in range(n)
    ]
    for i in range(n):
        for j in range(i + 1, n):
            cmp = clocks[i].compare(clocks[j])
            result[i][j] = cmp
            if cmp is ClockComparison.BEFORE:
                result[j][i] = ClockComparison.AFTER
            elif cmp is ClockComparison.AFTER:
                result[j][i] = ClockComparison.BEFORE
            else:  # EQUAL or CONCURRENT are symmetric
                result[j][i] = cmp
    return result


def find_concurrent_pairs(
    clocks: List[VectorClock],
) -> List[Tuple[int, int]]:
    """Return indices (i, j) with i < j where clocks are concurrent."""
    pairs: List[Tuple[int, int]] = []
    n = len(clocks)
    for i in range(n):
        for j in range(i + 1, n):
            if clocks[i].concurrent_with(clocks[j]):
                pairs.append((i, j))
    return pairs


# ======================================================================
# VectorClockManager
# ======================================================================

class VectorClockManager:
    """Manages vector clocks for all agents in a multi-agent system.

    Maintains the *current* clock for every registered agent and provides
    high-level operations for event recording and message passing.

    Typical usage::

        mgr = VectorClockManager(["a0", "a1", "a2"])
        mgr.record_local_event("a0")
        mgr.record_send("a0", "a1")    # a0 sends to a1
        mgr.record_receive("a1", "a0") # a1 receives from a0

    Attributes:
        _clocks: Mapping from agent_id to its current VectorClock.
        _history: Optional list of snapshots for later analysis.
    """

    def __init__(
        self,
        agent_ids: Iterable[str],
        *,
        keep_history: bool = False,
    ) -> None:
        self._clocks: Dict[str, VectorClock] = {
            aid: VectorClock({aid: 0}) for aid in agent_ids
        }
        self._keep_history = keep_history
        self._history: List[Dict[str, VectorClock]] = []

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    @property
    def agent_ids(self) -> List[str]:
        return list(self._clocks.keys())

    def register_agent(self, agent_id: str) -> None:
        """Add a new agent (idempotent)."""
        if agent_id not in self._clocks:
            self._clocks[agent_id] = VectorClock({agent_id: 0})

    def get_clock(self, agent_id: str) -> VectorClock:
        """Return the current clock for *agent_id*."""
        if agent_id not in self._clocks:
            raise KeyError(f"Unknown agent: {agent_id!r}")
        return self._clocks[agent_id]

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def _snapshot(self) -> None:
        if self._keep_history:
            self._history.append(
                {aid: vc.copy() for aid, vc in self._clocks.items()}
            )

    def record_local_event(self, agent_id: str) -> VectorClock:
        """Record a local event for *agent_id*: increment its own component.

        Args:
            agent_id: The agent performing the local event.

        Returns:
            The updated clock for the agent (after increment).
        """
        vc = self.get_clock(agent_id)
        vc.increment(agent_id)
        self._snapshot()
        return vc.copy()

    def record_send(self, sender: str, receiver: str) -> VectorClock:
        """Record that *sender* is sending a message to *receiver*.

        Increments the sender's clock and returns the clock that should
        be attached to the message.  The caller should later call
        :meth:`record_receive` on the receiver side with this clock.

        Args:
            sender: Agent sending the message.
            receiver: Intended recipient (used only for documentation;
                the actual merge happens in ``record_receive``).

        Returns:
            A copy of the sender's clock after incrementing.
        """
        vc = self.get_clock(sender)
        vc.increment(sender)
        self._snapshot()
        return vc.copy()

    def record_receive(
        self, receiver: str, sender: str,
        message_clock: Optional[VectorClock] = None,
    ) -> VectorClock:
        """Record that *receiver* received a message from *sender*.

        Merges the message clock (or the sender's current clock if not
        provided) into the receiver's clock, then increments the
        receiver's own component.

        Args:
            receiver: Agent receiving the message.
            sender: Agent that sent the message.
            message_clock: The clock snapshot attached to the message.
                If ``None``, uses the sender's *current* clock (less
                accurate but convenient for synchronous models).

        Returns:
            Updated clock for the receiver.
        """
        r_vc = self.get_clock(receiver)
        if message_clock is None:
            message_clock = self.get_clock(sender)
        r_vc.merge(message_clock)
        r_vc.increment(receiver)
        self._snapshot()
        return r_vc.copy()

    def record_shared_state_access(
        self, agent_id: str, shared_clock: VectorClock,
    ) -> VectorClock:
        """Record *agent_id* reading shared state carrying *shared_clock*.

        Merges the shared clock into the agent's own clock and increments.

        Args:
            agent_id: Agent accessing shared state.
            shared_clock: Clock associated with the shared state.

        Returns:
            Updated agent clock.
        """
        vc = self.get_clock(agent_id)
        vc.merge(shared_clock)
        vc.increment(agent_id)
        self._snapshot()
        return vc.copy()

    # ------------------------------------------------------------------
    # Batch queries
    # ------------------------------------------------------------------

    def pairwise_comparison(self) -> Dict[Tuple[str, str], ClockComparison]:
        """Compare all agent pairs; return dict (a_i, a_j) -> comparison."""
        ids = self.agent_ids
        result: Dict[Tuple[str, str], ClockComparison] = {}
        for i, a in enumerate(ids):
            for j in range(i + 1, len(ids)):
                b = ids[j]
                cmp = self._clocks[a].compare(self._clocks[b])
                result[(a, b)] = cmp
                # Symmetric entry
                if cmp is ClockComparison.BEFORE:
                    result[(b, a)] = ClockComparison.AFTER
                elif cmp is ClockComparison.AFTER:
                    result[(b, a)] = ClockComparison.BEFORE
                else:
                    result[(b, a)] = cmp
        return result

    def concurrent_agents(self) -> List[Tuple[str, str]]:
        """Return pairs of agents whose current clocks are concurrent."""
        pairs: List[Tuple[str, str]] = []
        ids = self.agent_ids
        for i, a in enumerate(ids):
            for j in range(i + 1, len(ids)):
                b = ids[j]
                if self._clocks[a].concurrent_with(self._clocks[b]):
                    pairs.append((a, b))
        return pairs

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Dict[str, VectorClock]]:
        """Return recorded clock history (empty if ``keep_history=False``)."""
        return list(self._history)

    def history_for_agent(self, agent_id: str) -> List[VectorClock]:
        """Return the sequence of clock snapshots for *agent_id*."""
        return [snap[agent_id] for snap in self._history if agent_id in snap]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manager state to a JSON-compatible dict."""
        return {
            "clocks": {aid: vc.to_dict() for aid, vc in self._clocks.items()},
            "keep_history": self._keep_history,
            "history": [
                {aid: vc.to_dict() for aid, vc in snap.items()}
                for snap in self._history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorClockManager":
        """Reconstruct from serialized form."""
        agent_ids = list(data["clocks"].keys())
        mgr = cls(agent_ids, keep_history=data.get("keep_history", False))
        for aid, vc_data in data["clocks"].items():
            mgr._clocks[aid] = VectorClock.from_dict(vc_data)
        for snap_data in data.get("history", []):
            mgr._history.append(
                {aid: VectorClock.from_dict(vd) for aid, vd in snap_data.items()}
            )
        return mgr

    def __repr__(self) -> str:
        agents = ", ".join(sorted(self._clocks.keys()))
        return f"VectorClockManager(agents=[{agents}])"
