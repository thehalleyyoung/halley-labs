"""Schedule space representation for importance sampling.

Defines schedules as sequences of (agent_id, action_time) pairs that
specify execution ordering, along with the space of all valid schedules
respecting happens-before constraints, distance metrics, and
neighbourhood structures for local search.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np


# ---------------------------------------------------------------------------
# Core schedule types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleEvent:
    """A single event in a schedule.

    Attributes:
        agent_id: Agent that acts.
        timestep: Discrete timestep at which the action occurs.
        action_time: Continuous time of the action (for continuous schedules).
    """

    agent_id: str
    timestep: int
    action_time: float = 0.0


@dataclass
class Schedule:
    """Sequence of (agent_id, action_time) pairs defining execution order.

    Events are ordered by ``action_time``; ties are broken by position
    in the ``events`` list.

    Attributes:
        events: Ordered list of schedule events.
        metadata: Optional metadata (e.g. sampling weight, source).
    """

    events: List[ScheduleEvent]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.events)

    @property
    def agents(self) -> FrozenSet[str]:
        return frozenset(e.agent_id for e in self.events)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def ordering(self) -> List[str]:
        """Agent IDs in execution order."""
        return [e.agent_id for e in self.events]

    def action_times(self) -> np.ndarray:
        """Array of action times."""
        return np.array([e.action_time for e in self.events])

    def to_permutation(self, agent_order: List[str]) -> np.ndarray:
        """Convert to a permutation vector over the given agent ordering.

        Returns an integer array where ``perm[i]`` is the position of
        ``agent_order[i]`` in the schedule.
        """
        pos = {e.agent_id: idx for idx, e in enumerate(self.events)}
        return np.array([pos.get(a, -1) for a in agent_order])

    def validate(self, constraints: Sequence["ScheduleConstraint"]) -> bool:
        """Check that all constraints are satisfied."""
        return all(c.is_satisfied(self) for c in constraints)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleConstraint:
    """A constraint from HB ordering that valid schedules must satisfy.

    Encodes ``before_agent`` at ``before_timestep`` must execute before
    ``after_agent`` at ``after_timestep``.
    """

    before_agent: str
    before_timestep: int
    after_agent: str
    after_timestep: int

    def is_satisfied(self, schedule: Schedule) -> bool:
        """Check whether *schedule* satisfies this constraint."""
        before_idx = None
        after_idx = None
        for idx, event in enumerate(schedule.events):
            if (
                event.agent_id == self.before_agent
                and event.timestep == self.before_timestep
            ):
                before_idx = idx
            if (
                event.agent_id == self.after_agent
                and event.timestep == self.after_timestep
            ):
                after_idx = idx

        if before_idx is None or after_idx is None:
            return True  # constraint is vacuously satisfied
        return before_idx < after_idx


# ---------------------------------------------------------------------------
# Schedule space
# ---------------------------------------------------------------------------

class ScheduleSpace:
    """The space of all valid schedules for N agents over T timesteps.

    The space is defined by the set of agents, the number of timesteps,
    and happens-before constraints that restrict which orderings are valid.
    """

    def __init__(
        self,
        agents: Sequence[str],
        num_timesteps: int,
        constraints: Sequence[ScheduleConstraint] = (),
    ) -> None:
        self._agents = list(agents)
        self._T = num_timesteps
        self._constraints = list(constraints)

    @property
    def agents(self) -> List[str]:
        return list(self._agents)

    @property
    def num_timesteps(self) -> int:
        return self._T

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def constraints(self) -> List[ScheduleConstraint]:
        return list(self._constraints)

    def total_events(self) -> int:
        """Total number of events: ``N * T``."""
        return len(self._agents) * self._T

    def unconstrained_size(self) -> int:
        """Number of orderings without constraints: ``(N*T)!``."""
        return math.factorial(self.total_events())

    def is_valid(self, schedule: Schedule) -> bool:
        """Check whether *schedule* satisfies all constraints."""
        return all(c.is_satisfied(schedule) for c in self._constraints)

    def build_precedence_graph(self) -> Dict[Tuple[str, int], Set[Tuple[str, int]]]:
        """Build a DAG of precedence relations from constraints.

        Returns a dict mapping each event ``(agent, timestep)`` to the
        set of events that must come *after* it.
        """
        graph: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {}
        for c in self._constraints:
            key = (c.before_agent, c.before_timestep)
            graph.setdefault(key, set()).add((c.after_agent, c.after_timestep))
        return graph


# ---------------------------------------------------------------------------
# Schedule generator
# ---------------------------------------------------------------------------

class ScheduleGenerator:
    """Enumerate or sample valid schedules from a :class:`ScheduleSpace`.

    Provides both exhaustive enumeration (for small spaces) and random
    sampling via topological-sort sampling of the constraint DAG.
    """

    def __init__(self, space: ScheduleSpace, rng: Optional[np.random.RandomState] = None) -> None:
        self._space = space
        self._rng = rng or np.random.RandomState(42)

    def sample_uniform(self, n_samples: int = 1) -> List[Schedule]:
        """Sample *n_samples* valid schedules uniformly at random.

        Uses randomised topological sort of the precedence DAG.
        """
        precedence = self._space.build_precedence_graph()
        all_events = [
            (a, t)
            for a in self._space.agents
            for t in range(self._space.num_timesteps)
        ]

        # Build in-degree map
        in_edges: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {
            e: set() for e in all_events
        }
        for before, afters in precedence.items():
            for after in afters:
                if after in in_edges:
                    in_edges[after].add(before)

        schedules: List[Schedule] = []
        for _ in range(n_samples):
            schedule = self._random_topo_sort(all_events, in_edges, precedence)
            schedules.append(schedule)
        return schedules

    def _random_topo_sort(
        self,
        all_events: List[Tuple[str, int]],
        in_edges: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        out_edges: Dict[Tuple[str, int], Set[Tuple[str, int]]],
    ) -> Schedule:
        """Randomised topological sort producing a valid schedule."""
        remaining_in: Dict[Tuple[str, int], int] = {
            e: len(in_edges.get(e, set())) for e in all_events
        }
        available = [e for e in all_events if remaining_in[e] == 0]
        result: List[ScheduleEvent] = []
        time_counter = 0.0

        while available:
            idx = int(self._rng.randint(0, len(available)))
            chosen = available.pop(idx)
            agent_id, timestep = chosen
            result.append(
                ScheduleEvent(
                    agent_id=agent_id,
                    timestep=timestep,
                    action_time=time_counter,
                )
            )
            time_counter += 1.0

            for successor in out_edges.get(chosen, set()):
                remaining_in[successor] -= 1
                if remaining_in[successor] == 0:
                    available.append(successor)

        return Schedule(events=result)

    def enumerate_all(self, max_count: int = 10000) -> List[Schedule]:
        """Enumerate all valid schedules (up to *max_count*).

        Only feasible for small schedule spaces.
        """
        all_events = [
            (a, t)
            for a in self._space.agents
            for t in range(self._space.num_timesteps)
        ]
        schedules: List[Schedule] = []

        for perm in itertools.islice(itertools.permutations(all_events), max_count):
            events = [
                ScheduleEvent(agent_id=a, timestep=t, action_time=float(i))
                for i, (a, t) in enumerate(perm)
            ]
            sched = Schedule(events=events)
            if self._space.is_valid(sched):
                schedules.append(sched)
                if len(schedules) >= max_count:
                    break

        return schedules


# ---------------------------------------------------------------------------
# Schedule neighbourhood
# ---------------------------------------------------------------------------

class ScheduleNeighborhood:
    """Define a neighbourhood for local search in schedule space.

    Neighbours are obtained by *adjacent transpositions* (swapping two
    consecutive events that are not constrained by HB ordering).
    """

    def __init__(self, space: ScheduleSpace) -> None:
        self._space = space

    def adjacent_swap_neighbours(self, schedule: Schedule) -> List[Schedule]:
        """All schedules obtainable by swapping one adjacent pair."""
        neighbours: List[Schedule] = []
        events = list(schedule.events)

        for i in range(len(events) - 1):
            swapped = list(events)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            # Update action times to maintain ordering
            for j, e in enumerate(swapped):
                swapped[j] = ScheduleEvent(
                    agent_id=e.agent_id,
                    timestep=e.timestep,
                    action_time=float(j),
                )
            candidate = Schedule(events=swapped)
            if self._space.is_valid(candidate):
                neighbours.append(candidate)

        return neighbours

    def k_swap_neighbours(self, schedule: Schedule, k: int = 2) -> List[Schedule]:
        """Neighbours obtained by swapping up to *k* non-adjacent pairs."""
        if k == 1:
            return self.adjacent_swap_neighbours(schedule)

        neighbours: List[Schedule] = []
        events = list(schedule.events)
        n = len(events)

        for positions in itertools.combinations(range(n), 2):
            i, j = positions
            swapped = list(events)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            for idx, e in enumerate(swapped):
                swapped[idx] = ScheduleEvent(
                    agent_id=e.agent_id,
                    timestep=e.timestep,
                    action_time=float(idx),
                )
            candidate = Schedule(events=swapped)
            if self._space.is_valid(candidate):
                neighbours.append(candidate)

        return neighbours


# ---------------------------------------------------------------------------
# Continuous schedule
# ---------------------------------------------------------------------------

class ContinuousSchedule:
    """Schedule with continuous timing (not just orderings).

    Each agent has a continuous action time in [0, T].  The discrete
    ordering is derived by sorting on these times.

    Attributes:
        agent_times: Mapping ``{agent_id: action_time}``.
        period: Total time horizon.
    """

    def __init__(
        self,
        agent_times: Dict[str, float],
        period: float = 1.0,
    ) -> None:
        self.agent_times = dict(agent_times)
        self.period = period

    def to_discrete_schedule(self) -> Schedule:
        """Convert to a discrete :class:`Schedule` by sorting on times."""
        sorted_agents = sorted(self.agent_times.items(), key=lambda x: x[1])
        events = [
            ScheduleEvent(
                agent_id=agent,
                timestep=0,
                action_time=time,
            )
            for agent, time in sorted_agents
        ]
        return Schedule(events=events)

    def to_vector(self, agent_order: List[str]) -> np.ndarray:
        """Convert to a numpy vector of action times."""
        return np.array([self.agent_times.get(a, 0.0) for a in agent_order])

    @classmethod
    def from_vector(
        cls,
        times: np.ndarray,
        agent_order: List[str],
        period: float = 1.0,
    ) -> "ContinuousSchedule":
        """Construct from a numpy vector."""
        agent_times = {a: float(t) for a, t in zip(agent_order, times)}
        return cls(agent_times, period)

    def perturb(
        self,
        sigma: float = 0.01,
        rng: Optional[np.random.RandomState] = None,
    ) -> "ContinuousSchedule":
        """Return a Gaussian-perturbed copy."""
        rng = rng or np.random.RandomState()
        new_times = {
            a: float(np.clip(t + rng.normal(0, sigma), 0, self.period))
            for a, t in self.agent_times.items()
        }
        return ContinuousSchedule(new_times, self.period)


# ---------------------------------------------------------------------------
# Schedule distance
# ---------------------------------------------------------------------------

class ScheduleDistance:
    """Distance metrics between schedules."""

    @staticmethod
    def kendall_tau(s1: Schedule, s2: Schedule) -> int:
        """Kendall tau distance (number of pairwise disagreements).

        Both schedules must cover the same set of ``(agent, timestep)``
        events.
        """
        order1 = [(e.agent_id, e.timestep) for e in s1.events]
        order2 = [(e.agent_id, e.timestep) for e in s2.events]

        pos1 = {ev: i for i, ev in enumerate(order1)}
        pos2 = {ev: i for i, ev in enumerate(order2)}

        common = set(pos1.keys()) & set(pos2.keys())
        events_list = sorted(common)
        disagreements = 0

        for i in range(len(events_list)):
            for j in range(i + 1, len(events_list)):
                a, b = events_list[i], events_list[j]
                if (pos1[a] - pos1[b]) * (pos2[a] - pos2[b]) < 0:
                    disagreements += 1

        return disagreements

    @staticmethod
    def spearman_footrule(s1: Schedule, s2: Schedule) -> float:
        """Spearman footrule distance (sum of absolute rank differences)."""
        order1 = [(e.agent_id, e.timestep) for e in s1.events]
        order2 = [(e.agent_id, e.timestep) for e in s2.events]

        pos1 = {ev: i for i, ev in enumerate(order1)}
        pos2 = {ev: i for i, ev in enumerate(order2)}

        common = set(pos1.keys()) & set(pos2.keys())
        return sum(abs(pos1[ev] - pos2[ev]) for ev in common)

    @staticmethod
    def continuous_l2(s1: ContinuousSchedule, s2: ContinuousSchedule) -> float:
        """L2 distance between continuous action-time vectors."""
        common_agents = sorted(set(s1.agent_times.keys()) & set(s2.agent_times.keys()))
        if not common_agents:
            return 0.0
        v1 = np.array([s1.agent_times[a] for a in common_agents])
        v2 = np.array([s2.agent_times[a] for a in common_agents])
        return float(np.linalg.norm(v1 - v2))
