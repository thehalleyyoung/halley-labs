"""Multi-agent trace construction from environment rollouts.

Provides four key components:

* **TraceConstructor** — builds a complete ``ExecutionTrace`` from a
  sequence of environment step tuples ``(actions, observations, rewards,
  dones, info)``.
* **AsyncTraceBuilder** — accumulates events from concurrent agent
  coroutines, stamping each with vector clocks.
* **TraceRecorder** — wrapper / hook that sits between agents and an
  environment, transparently recording every interaction.
* **TraceMerger** — merges independently recorded per-agent traces
  using vector clocks into a single ``MultiAgentTrace``.
* **Causal chain inference** — given environment dynamics, infer which
  agent's action causally affected another agent's observation.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
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
    make_action_event,
    make_communication_event,
    make_environment_event,
    make_observation_event,
    make_sync_event,
    vc_increment,
    vc_merge,
    vc_zero,
)
from .trace import ExecutionTrace, MultiAgentTrace


# ---------------------------------------------------------------------------
# Protocol for MARL environments
# ---------------------------------------------------------------------------
class MARLEnv(Protocol):
    """Minimal protocol for a multi-agent environment."""

    def reset(self) -> Dict[str, np.ndarray]: ...
    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # dones
        Dict[str, Any],         # infos
    ]: ...

    @property
    def agents(self) -> List[str]: ...


# ---------------------------------------------------------------------------
# TraceConstructor
# ---------------------------------------------------------------------------
class TraceConstructor:
    """Build an ``ExecutionTrace`` from a batch of environment transitions.

    Usage::

        tc = TraceConstructor(["agent_0", "agent_1"])
        tc.record_step(actions, observations, rewards, dones, infos)
        ...
        trace = tc.build()
    """

    def __init__(self, agent_ids: Sequence[str], state_dim: int = 0):
        self._agent_ids = list(agent_ids)
        self._state_dim = state_dim
        self._clocks: Dict[str, VectorClock] = {
            aid: vc_zero(self._agent_ids) for aid in self._agent_ids
        }
        self._clocks["__env__"] = vc_zero(self._agent_ids)
        self._events: List[Event] = []
        self._step_count = 0
        self._t0 = time.monotonic()
        self._last_action_ids: Dict[str, str] = {}
        self._last_obs_ids: Dict[str, str] = {}
        self._prev_states: Dict[str, Optional[np.ndarray]] = {
            aid: None for aid in self._agent_ids
        }

    def _timestamp(self) -> float:
        return time.monotonic() - self._t0

    def record_step(
        self,
        actions: Dict[str, np.ndarray],
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        infos: Optional[Dict[str, Any]] = None,
        *,
        env_state: Optional[np.ndarray] = None,
        prev_env_state: Optional[np.ndarray] = None,
    ) -> None:
        """Record one environment step (all agents act simultaneously)."""
        infos = infos or {}
        ts = self._timestamp()

        # 1. Action events
        action_ids: Dict[str, str] = {}
        for aid in self._agent_ids:
            if aid not in actions:
                continue
            self._clocks[aid] = vc_increment(self._clocks[aid], aid)
            preds: FrozenSet[str] = frozenset()
            if aid in self._last_obs_ids:
                preds = frozenset([self._last_obs_ids[aid]])
            ae = make_action_event(
                agent_id=aid,
                timestamp=ts,
                action_vector=actions[aid],
                vector_clock=dict(self._clocks[aid]),
                state_before=self._prev_states.get(aid),
                causal_predecessors=preds,
                data={"reward": rewards.get(aid, 0.0), "done": dones.get(aid, False)},
            )
            self._events.append(ae)
            action_ids[aid] = ae.event_id
            self._last_action_ids[aid] = ae.event_id

        # 2. Environment event (if state is provided)
        env_event_id: Optional[str] = None
        if env_state is not None:
            delta = (
                env_state - prev_env_state
                if prev_env_state is not None
                else env_state
            )
            env_vc = dict(self._clocks["__env__"])
            for aid in self._agent_ids:
                env_vc = vc_merge(env_vc, self._clocks[aid])
            env_preds = frozenset(action_ids.values())
            ee = make_environment_event(
                timestamp=ts + 1e-6,
                env_state_delta=delta,
                affected_agents=list(self._agent_ids),
                vector_clock=env_vc,
                causal_predecessors=env_preds,
                data={"step": self._step_count},
            )
            self._events.append(ee)
            env_event_id = ee.event_id
            self._clocks["__env__"] = env_vc

        # 3. Observation events
        for aid in self._agent_ids:
            if aid not in observations:
                continue
            # Observation depends on all actions (joint transition) and env event
            obs_preds_set: Set[str] = set(action_ids.values())
            if env_event_id is not None:
                obs_preds_set.add(env_event_id)
            self._clocks[aid] = vc_merge(self._clocks[aid], self._clocks.get("__env__", {}))
            self._clocks[aid] = vc_increment(self._clocks[aid], aid)
            source_agents = [
                a for a in self._agent_ids if a != aid and a in actions
            ]
            oe = make_observation_event(
                agent_id=aid,
                timestamp=ts + 2e-6,
                observation_vector=observations[aid],
                vector_clock=dict(self._clocks[aid]),
                source_agents=source_agents,
                causal_predecessors=frozenset(obs_preds_set),
                data={"step": self._step_count},
            )
            self._events.append(oe)
            self._last_obs_ids[aid] = oe.event_id
            self._prev_states[aid] = observations[aid].copy()

        self._step_count += 1

    def record_communication(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        *,
        channel_id: str = "default",
    ) -> str:
        """Record an inter-agent message. Returns the event id."""
        self._clocks[sender] = vc_increment(self._clocks[sender], sender)
        preds: FrozenSet[str] = frozenset()
        if sender in self._last_action_ids:
            preds = frozenset([self._last_action_ids[sender]])
        ce = make_communication_event(
            sender=sender,
            receiver=receiver,
            timestamp=self._timestamp(),
            message_payload=payload,
            vector_clock=dict(self._clocks[sender]),
            channel_id=channel_id,
            causal_predecessors=preds,
        )
        self._events.append(ce)
        # Receiver merges sender's clock
        self._clocks[receiver] = vc_merge(
            self._clocks[receiver], self._clocks[sender]
        )
        return ce.event_id

    def record_sync_barrier(self, barrier_id: str) -> None:
        """Record a synchronization barrier across all agents."""
        arrived: Dict[str, VectorClock] = {
            aid: dict(self._clocks[aid]) for aid in self._agent_ids
        }
        merged = dict(vc_zero(self._agent_ids))
        for vc in arrived.values():
            merged = vc_merge(merged, vc)

        for aid in self._agent_ids:
            self._clocks[aid] = dict(merged)
            se = make_sync_event(
                agent_id=aid,
                timestamp=self._timestamp(),
                barrier_id=barrier_id,
                barrier_agents=list(self._agent_ids),
                vector_clock=dict(merged),
                arrived_clocks=arrived,
            )
            self._events.append(se)

    def build(self, trace_id: str = "") -> ExecutionTrace:
        """Return the assembled trace."""
        trace = ExecutionTrace(
            trace_id=trace_id or uuid.uuid4().hex,
            agents=self._agent_ids,
        )
        trace.extend(self._events)
        return trace


# ---------------------------------------------------------------------------
# AsyncTraceBuilder
# ---------------------------------------------------------------------------
class AsyncTraceBuilder:
    """Thread-safe-ish builder for traces from async multi-agent execution.

    Each agent calls ``record_*`` from its own coroutine / thread.
    Vector clocks are maintained per-agent and merged at communication points.

    Note: for true thread safety wrap calls with a lock externally.
    """

    def __init__(self, agent_ids: Sequence[str]):
        self._agent_ids = list(agent_ids)
        self._clocks: Dict[str, VectorClock] = {
            aid: vc_zero(self._agent_ids) for aid in self._agent_ids
        }
        self._events: List[Event] = []
        self._t0 = time.monotonic()
        self._last_event: Dict[str, str] = {}

    def _ts(self) -> float:
        return time.monotonic() - self._t0

    def record_action(
        self,
        agent_id: str,
        action_vector: np.ndarray,
        *,
        state_before: Optional[np.ndarray] = None,
        state_after: Optional[np.ndarray] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._clocks[agent_id] = vc_increment(self._clocks[agent_id], agent_id)
        preds = frozenset([self._last_event[agent_id]]) if agent_id in self._last_event else frozenset()
        ae = make_action_event(
            agent_id=agent_id,
            timestamp=self._ts(),
            action_vector=action_vector,
            vector_clock=dict(self._clocks[agent_id]),
            state_before=state_before,
            state_after=state_after,
            causal_predecessors=preds,
            data=data,
        )
        self._events.append(ae)
        self._last_event[agent_id] = ae.event_id
        return ae.event_id

    def record_observation(
        self,
        agent_id: str,
        observation_vector: np.ndarray,
        *,
        source_agents: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._clocks[agent_id] = vc_increment(self._clocks[agent_id], agent_id)
        preds = frozenset([self._last_event[agent_id]]) if agent_id in self._last_event else frozenset()
        oe = make_observation_event(
            agent_id=agent_id,
            timestamp=self._ts(),
            observation_vector=observation_vector,
            vector_clock=dict(self._clocks[agent_id]),
            source_agents=source_agents,
            causal_predecessors=preds,
            data=data,
        )
        self._events.append(oe)
        self._last_event[agent_id] = oe.event_id
        return oe.event_id

    def record_send(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        *,
        channel_id: str = "default",
    ) -> str:
        """Record a send (sender side). Call ``record_receive`` on receiver."""
        self._clocks[sender] = vc_increment(self._clocks[sender], sender)
        preds = frozenset([self._last_event[sender]]) if sender in self._last_event else frozenset()
        ce = make_communication_event(
            sender=sender,
            receiver=receiver,
            timestamp=self._ts(),
            message_payload=payload,
            vector_clock=dict(self._clocks[sender]),
            channel_id=channel_id,
            causal_predecessors=preds,
        )
        self._events.append(ce)
        self._last_event[sender] = ce.event_id
        return ce.event_id

    def record_receive(self, receiver: str, send_event_id: str) -> None:
        """Merge sender's clock into receiver upon message delivery."""
        send_event = None
        for e in self._events:
            if e.event_id == send_event_id:
                send_event = e
                break
        if send_event is not None:
            self._clocks[receiver] = vc_merge(
                self._clocks[receiver], send_event.vector_clock
            )

    def build(self, trace_id: str = "") -> ExecutionTrace:
        trace = ExecutionTrace(
            trace_id=trace_id or uuid.uuid4().hex,
            agents=self._agent_ids,
        )
        trace.extend(self._events)
        return trace


# ---------------------------------------------------------------------------
# TraceRecorder — environment wrapper
# ---------------------------------------------------------------------------
class TraceRecorder:
    """Wraps a ``MARLEnv`` and records a trace transparently.

    Usage::

        recorder = TraceRecorder(env)
        obs = recorder.reset()
        for _ in range(100):
            actions = {aid: policy(obs[aid]) for aid in env.agents}
            obs, rewards, dones, infos = recorder.step(actions)
        trace = recorder.get_trace()
    """

    def __init__(self, env: Any, *, state_dim: int = 0):
        self._env = env
        self._state_dim = state_dim
        self._constructor: Optional[TraceConstructor] = None
        self._prev_env_state: Optional[np.ndarray] = None

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self._env.reset()
        agent_ids = list(obs.keys())
        self._constructor = TraceConstructor(agent_ids, state_dim=self._state_dim)
        self._prev_env_state = None
        return obs

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        assert self._constructor is not None, "call reset() first"

        obs, rewards, dones, infos = self._env.step(actions)

        env_state = infos.pop("__env_state__", None)
        if env_state is not None:
            env_state = np.asarray(env_state, dtype=np.float64)

        self._constructor.record_step(
            actions=actions,
            observations=obs,
            rewards=rewards,
            dones=dones,
            infos=infos,
            env_state=env_state,
            prev_env_state=self._prev_env_state,
        )

        if env_state is not None:
            self._prev_env_state = env_state.copy()

        return obs, rewards, dones, infos

    def record_communication(
        self, sender: str, receiver: str, payload: Dict[str, Any]
    ) -> str:
        assert self._constructor is not None
        return self._constructor.record_communication(sender, receiver, payload)

    def get_trace(self, trace_id: str = "") -> ExecutionTrace:
        assert self._constructor is not None
        return self._constructor.build(trace_id=trace_id)


# ---------------------------------------------------------------------------
# TraceMerger
# ---------------------------------------------------------------------------
class TraceMerger:
    """Merge independently recorded per-agent traces using vector clocks.

    Each per-agent trace must have consistent vector clock annotations.
    The merger detects shared communication events (same event_id
    appearing in both traces) and uses them as synchronisation anchors.
    """

    def __init__(self) -> None:
        self._agent_traces: Dict[str, ExecutionTrace] = {}

    def add_trace(self, agent_id: str, trace: ExecutionTrace) -> None:
        self._agent_traces[agent_id] = trace

    def merge(self, trace_id: str = "") -> MultiAgentTrace:
        mat = MultiAgentTrace(trace_id=trace_id or uuid.uuid4().hex)
        for aid, tr in self._agent_traces.items():
            mat.add_agent_trace(aid, tr)
        return mat

    def detect_shared_events(self) -> Dict[str, List[str]]:
        """Find event_ids that appear in more than one agent's trace.

        Returns ``{event_id: [agent_id, ...]}``.
        """
        id_to_agents: Dict[str, List[str]] = defaultdict(list)
        for aid, tr in self._agent_traces.items():
            for e in tr:
                id_to_agents[e.event_id].append(aid)
        return {eid: aids for eid, aids in id_to_agents.items() if len(aids) > 1}


# ---------------------------------------------------------------------------
# Causal chain inference
# ---------------------------------------------------------------------------
def infer_causal_chains(
    trace: ExecutionTrace,
    *,
    influence_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Infer which agent's action causally influenced another's observation.

    Heuristic: for each ObservationEvent *o* of agent B that lists agent A
    in ``source_agents``, check whether there is an ActionEvent of agent A
    that happens-before *o*.  If the action-observation cosine similarity
    exceeds *influence_threshold* we flag a causal chain.

    Returns a list of dicts::

        {
            "action_event_id": ...,
            "observation_event_id": ...,
            "actor": ...,
            "observer": ...,
            "similarity": ...,
        }
    """
    actions_by_agent: Dict[str, List[ActionEvent]] = defaultdict(list)
    observations: List[ObservationEvent] = []

    for e in trace:
        if isinstance(e, ActionEvent):
            actions_by_agent[e.agent_id].append(e)
        elif isinstance(e, ObservationEvent):
            observations.append(e)

    chains: List[Dict[str, Any]] = []
    for obs in observations:
        for src_agent in obs.source_agents:
            if src_agent == obs.agent_id:
                continue
            for act in actions_by_agent.get(src_agent, []):
                if not act.happens_before(obs):
                    continue
                sim = _cosine_similarity(act.action_vector, obs.observation_vector)
                if abs(sim) >= influence_threshold:
                    chains.append(
                        {
                            "action_event_id": act.event_id,
                            "observation_event_id": obs.event_id,
                            "actor": src_agent,
                            "observer": obs.agent_id,
                            "similarity": float(sim),
                        }
                    )
    return chains


def compute_influence_matrix(
    trace: ExecutionTrace,
) -> Tuple[np.ndarray, List[str]]:
    """Compute an *n × n* influence matrix where entry (i, j) measures
    how much agent *i*'s actions influenced agent *j*'s observations.

    Influence is measured as the mean absolute cosine similarity between
    agent i's action vectors and agent j's observation vectors where
    i ∈ obs.source_agents and the action happens-before the observation.

    Returns
    -------
    matrix : np.ndarray of shape (n_agents, n_agents)
    agent_order : list[str]
    """
    agents = sorted(trace.agents - {"__env__"})
    n = len(agents)
    idx = {a: i for i, a in enumerate(agents)}
    accum = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    actions_by_agent: Dict[str, List[ActionEvent]] = defaultdict(list)
    for e in trace:
        if isinstance(e, ActionEvent):
            actions_by_agent[e.agent_id].append(e)

    for e in trace:
        if not isinstance(e, ObservationEvent):
            continue
        j = idx.get(e.agent_id)
        if j is None:
            continue
        for src in e.source_agents:
            i = idx.get(src)
            if i is None or src == e.agent_id:
                continue
            for act in actions_by_agent.get(src, []):
                if act.happens_before(e):
                    sim = abs(_cosine_similarity(act.action_vector, e.observation_vector))
                    accum[i, j] += sim
                    counts[i, j] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(counts > 0, accum / counts, 0.0)
    return matrix, agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine_similarity(
    a: Optional[np.ndarray], b: Optional[np.ndarray]
) -> float:
    """Cosine similarity, returning 0.0 for degenerate inputs."""
    if a is None or b is None:
        return 0.0
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
