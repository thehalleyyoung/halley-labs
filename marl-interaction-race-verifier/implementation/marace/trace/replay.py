"""Trace replay engine for MARACE.

Replays a recorded execution trace under a (possibly permuted) schedule.
The replayer checks that any proposed reordering respects the
happens-before partial order and computes the resulting state trajectory.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from .events import (
    ActionEvent,
    CommunicationEvent,
    EnvironmentEvent,
    Event,
    EventType,
    ObservationEvent,
    SyncEvent,
    VectorClock,
    vc_concurrent,
    vc_leq,
    vc_strictly_less,
)
from .trace import ExecutionTrace


# ---------------------------------------------------------------------------
# Schedule permutation
# ---------------------------------------------------------------------------
class SchedulePermutation:
    """A reordering of events identified by their event_ids.

    The permutation is stored as an ordered list of event_ids representing
    the *new* total order in which events should be replayed.
    """

    def __init__(self, event_ids: Sequence[str]):
        self._order: List[str] = list(event_ids)
        self._position: Dict[str, int] = {eid: i for i, eid in enumerate(self._order)}

    @property
    def order(self) -> List[str]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def position_of(self, event_id: str) -> int:
        return self._position[event_id]

    def swap(self, i: int, j: int) -> "SchedulePermutation":
        """Return a new permutation with positions *i* and *j* swapped."""
        new_order = list(self._order)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return SchedulePermutation(new_order)

    @classmethod
    def identity(cls, trace: ExecutionTrace) -> "SchedulePermutation":
        """The original trace order."""
        return cls([e.event_id for e in trace])

    @classmethod
    def from_trace(cls, trace: ExecutionTrace) -> "SchedulePermutation":
        return cls.identity(trace)

    def is_permutation_of(self, other: "SchedulePermutation") -> bool:
        return sorted(self._order) == sorted(other._order)


# ---------------------------------------------------------------------------
# Replay validator
# ---------------------------------------------------------------------------
class ReplayValidator:
    """Check whether a ``SchedulePermutation`` respects HB ordering."""

    def __init__(self, trace: ExecutionTrace):
        self._trace = trace
        self._id_map: Dict[str, Event] = {e.event_id: e for e in trace}

    def is_hb_consistent(self, perm: SchedulePermutation) -> bool:
        """True iff for every pair (a, b) where a → b (HB), a precedes b."""
        for eid_b, event_b in self._id_map.items():
            pos_b = perm.position_of(eid_b)
            for pred_id in event_b.causal_predecessors:
                if pred_id not in self._id_map:
                    continue
                if perm.position_of(pred_id) >= pos_b:
                    return False

        # Also check VC-implied ordering for events not linked by explicit preds
        events = list(self._id_map.values())
        n = len(events)
        for i in range(n):
            for j in range(i + 1, n):
                ei, ej = events[i], events[j]
                if vc_strictly_less(ei.vector_clock, ej.vector_clock):
                    if perm.position_of(ei.event_id) >= perm.position_of(ej.event_id):
                        return False
                elif vc_strictly_less(ej.vector_clock, ei.vector_clock):
                    if perm.position_of(ej.event_id) >= perm.position_of(ei.event_id):
                        return False
        return True

    def violation_pairs(
        self, perm: SchedulePermutation
    ) -> List[Tuple[str, str]]:
        """Return all (predecessor, successor) pairs whose order is violated."""
        violations: List[Tuple[str, str]] = []
        for eid_b, event_b in self._id_map.items():
            pos_b = perm.position_of(eid_b)
            for pred_id in event_b.causal_predecessors:
                if pred_id in self._id_map and perm.position_of(pred_id) >= pos_b:
                    violations.append((pred_id, eid_b))
        return violations

    def enumerate_valid_permutations(
        self, *, max_count: int = 1000
    ) -> List[SchedulePermutation]:
        """Enumerate HB-consistent total orders up to *max_count*.

        Uses iterative deepening over concurrent-event swaps starting from
        the identity schedule.  For traces with many concurrent events this
        can be exponential — the ``max_count`` cap prevents runaway.
        """
        identity = SchedulePermutation.identity(self._trace)
        concurrent_pairs = self._concurrent_position_pairs(identity)

        if not concurrent_pairs:
            return [identity]

        results: List[SchedulePermutation] = [identity]
        seen: Set[Tuple[str, ...]] = {tuple(identity.order)}

        frontier: List[SchedulePermutation] = [identity]
        while frontier and len(results) < max_count:
            current = frontier.pop(0)
            cp = self._concurrent_position_pairs(current)
            for i, j in cp:
                candidate = current.swap(i, j)
                key = tuple(candidate.order)
                if key in seen:
                    continue
                seen.add(key)
                if self.is_hb_consistent(candidate):
                    results.append(candidate)
                    frontier.append(candidate)
                    if len(results) >= max_count:
                        break
        return results

    def _concurrent_position_pairs(
        self, perm: SchedulePermutation
    ) -> List[Tuple[int, int]]:
        """Return (pos_i, pos_j) for adjacent concurrent events in *perm*."""
        order = perm.order
        pairs: List[Tuple[int, int]] = []
        for k in range(len(order) - 1):
            ea = self._id_map[order[k]]
            eb = self._id_map[order[k + 1]]
            if vc_concurrent(ea.vector_clock, eb.vector_clock):
                pairs.append((k, k + 1))
        return pairs


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    """Mutable state snapshot for one agent during replay."""
    agent_id: str
    state_vector: np.ndarray
    step_count: int = 0
    last_action: Optional[np.ndarray] = None
    last_observation: Optional[np.ndarray] = None
    messages_received: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReplayStepResult:
    """Outcome of replaying a single event."""
    event: Event
    step_index: int
    agent_states_after: Dict[str, np.ndarray]
    env_state_after: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    """Full result of replaying a trace under a given schedule."""
    schedule: SchedulePermutation
    steps: List[ReplayStepResult] = field(default_factory=list)
    final_agent_states: Dict[str, np.ndarray] = field(default_factory=dict)
    final_env_state: Optional[np.ndarray] = None
    is_consistent: bool = True
    total_warnings: int = 0

    @property
    def state_trajectory(self) -> Dict[str, List[np.ndarray]]:
        """Per-agent state trajectory across replay steps."""
        traj: Dict[str, List[np.ndarray]] = {}
        for step in self.steps:
            for aid, sv in step.agent_states_after.items():
                traj.setdefault(aid, []).append(sv)
        return traj


# ---------------------------------------------------------------------------
# Transition functions
# ---------------------------------------------------------------------------
# A transition function takes (current_state, event) → new_state.
TransitionFn = Callable[[np.ndarray, Event], np.ndarray]


def default_action_transition(state: np.ndarray, event: Event) -> np.ndarray:
    """Default: add scaled action vector to state."""
    if isinstance(event, ActionEvent) and event.action_vector is not None:
        av = event.action_vector
        if av.shape != state.shape:
            av = np.resize(av, state.shape)
        return state + 0.1 * av
    return state.copy()


def default_observation_transition(state: np.ndarray, event: Event) -> np.ndarray:
    """Default: blend observation into state."""
    if isinstance(event, ObservationEvent) and event.observation_vector is not None:
        ov = event.observation_vector
        if ov.shape != state.shape:
            ov = np.resize(ov, state.shape)
        return 0.9 * state + 0.1 * ov
    return state.copy()


def default_env_transition(state: np.ndarray, event: Event) -> np.ndarray:
    """Default: apply env delta."""
    if isinstance(event, EnvironmentEvent) and event.env_state_delta is not None:
        delta = event.env_state_delta
        if delta.shape != state.shape:
            delta = np.resize(delta, state.shape)
        return state + delta
    return state.copy()


# ---------------------------------------------------------------------------
# Trace replayer
# ---------------------------------------------------------------------------
class TraceReplayer:
    """Replay a trace under a given schedule, computing state trajectories.

    Parameters
    ----------
    trace : ExecutionTrace
        The recorded trace.
    state_dim : int
        Dimensionality of agent / environment state vectors.
    action_transition : TransitionFn, optional
        How action events update agent state.
    observation_transition : TransitionFn, optional
        How observation events update agent state.
    env_transition : TransitionFn, optional
        How environment events update global env state.
    """

    def __init__(
        self,
        trace: ExecutionTrace,
        state_dim: int = 8,
        *,
        action_transition: Optional[TransitionFn] = None,
        observation_transition: Optional[TransitionFn] = None,
        env_transition: Optional[TransitionFn] = None,
    ):
        self._trace = trace
        self._state_dim = state_dim
        self._action_fn = action_transition or default_action_transition
        self._obs_fn = observation_transition or default_observation_transition
        self._env_fn = env_transition or default_env_transition
        self._id_map: Dict[str, Event] = {e.event_id: e for e in trace}

    def replay(
        self,
        schedule: Optional[SchedulePermutation] = None,
        *,
        initial_agent_states: Optional[Dict[str, np.ndarray]] = None,
        initial_env_state: Optional[np.ndarray] = None,
    ) -> ReplayResult:
        """Replay the trace under *schedule* (default: original order).

        Returns a ``ReplayResult`` with the full state trajectory.
        """
        if schedule is None:
            schedule = SchedulePermutation.identity(self._trace)

        validator = ReplayValidator(self._trace)
        is_consistent = validator.is_hb_consistent(schedule)

        # Initialise states
        agent_states: Dict[str, AgentState] = {}
        for aid in self._trace.agents:
            sv = (
                initial_agent_states[aid].copy()
                if initial_agent_states and aid in initial_agent_states
                else np.zeros(self._state_dim, dtype=np.float64)
            )
            agent_states[aid] = AgentState(agent_id=aid, state_vector=sv)

        env_state = (
            initial_env_state.copy()
            if initial_env_state is not None
            else np.zeros(self._state_dim, dtype=np.float64)
        )

        steps: List[ReplayStepResult] = []
        for step_idx, eid in enumerate(schedule.order):
            event = self._id_map[eid]
            warnings: List[str] = []

            # Ensure agent state exists (for env events etc.)
            if event.agent_id not in agent_states and event.agent_id != "__env__":
                agent_states[event.agent_id] = AgentState(
                    agent_id=event.agent_id,
                    state_vector=np.zeros(self._state_dim, dtype=np.float64),
                )

            if event.event_type == EventType.ACTION:
                astate = agent_states[event.agent_id]
                astate.state_vector = self._action_fn(astate.state_vector, event)
                astate.step_count += 1
                if isinstance(event, ActionEvent):
                    astate.last_action = event.action_vector

            elif event.event_type == EventType.OBSERVATION:
                astate = agent_states[event.agent_id]
                astate.state_vector = self._obs_fn(astate.state_vector, event)
                if isinstance(event, ObservationEvent):
                    astate.last_observation = event.observation_vector

            elif event.event_type == EventType.COMMUNICATION:
                if isinstance(event, CommunicationEvent):
                    recv_id = event.receiver
                    if recv_id in agent_states:
                        agent_states[recv_id].messages_received.append(
                            event.message_payload
                        )
                    else:
                        warnings.append(
                            f"Receiver {recv_id} not in agent states"
                        )

            elif event.event_type == EventType.ENVIRONMENT:
                env_state = self._env_fn(env_state, event)
                if isinstance(event, EnvironmentEvent):
                    for aid in event.affected_agents:
                        if aid in agent_states:
                            # Lightly perturb affected agents' states
                            agent_states[aid].state_vector += (
                                0.01 * np.random.RandomState(step_idx).randn(self._state_dim)
                            )

            elif event.event_type == EventType.SYNC:
                pass  # barriers are no-ops in state; ordering is what matters

            snapshot = {
                aid: ast.state_vector.copy() for aid, ast in agent_states.items()
            }
            steps.append(
                ReplayStepResult(
                    event=event,
                    step_index=step_idx,
                    agent_states_after=snapshot,
                    env_state_after=env_state.copy(),
                    warnings=warnings,
                )
            )

        result = ReplayResult(
            schedule=schedule,
            steps=steps,
            final_agent_states={
                aid: ast.state_vector.copy() for aid, ast in agent_states.items()
            },
            final_env_state=env_state.copy(),
            is_consistent=is_consistent,
            total_warnings=sum(len(s.warnings) for s in steps),
        )
        return result

    def deterministic_replay(self) -> ReplayResult:
        """Replay under the original recorded order (identity schedule)."""
        return self.replay(SchedulePermutation.identity(self._trace))

    def compare_schedules(
        self,
        schedules: Sequence[SchedulePermutation],
    ) -> Dict[str, Any]:
        """Replay under each schedule and compare final states.

        Returns a summary dict with per-schedule final states and
        max pairwise divergence.
        """
        results: List[ReplayResult] = []
        for sched in schedules:
            results.append(self.replay(sched))

        max_divergence = 0.0
        divergence_pairs: List[Tuple[int, int, float]] = []

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                div = _state_divergence(
                    results[i].final_agent_states,
                    results[j].final_agent_states,
                )
                divergence_pairs.append((i, j, div))
                if div > max_divergence:
                    max_divergence = div

        return {
            "num_schedules": len(schedules),
            "max_divergence": max_divergence,
            "divergence_pairs": divergence_pairs,
            "all_consistent": all(r.is_consistent for r in results),
            "results": results,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _state_divergence(
    states_a: Dict[str, np.ndarray],
    states_b: Dict[str, np.ndarray],
) -> float:
    """Max L2 norm of state difference across agents."""
    all_agents = set(states_a) | set(states_b)
    max_div = 0.0
    for aid in all_agents:
        sa = states_a.get(aid, np.zeros(1))
        sb = states_b.get(aid, np.zeros(1))
        if sa.shape != sb.shape:
            sa = np.resize(sa, max(sa.shape[0], sb.shape[0]))
            sb = np.resize(sb, sa.shape[0])
        div = float(np.linalg.norm(sa - sb))
        if div > max_div:
            max_div = div
    return max_div
