"""
Causal inference engine for the MARACE happens-before system.

Determines causal relationships between agent actions and observations
through multiple channels: direct observation dependencies, physics-mediated
causality, explicit communication, and shared environment state.
"""

from __future__ import annotations

import enum
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Causal chain types and evidence classification
# ======================================================================

class CausalChainType(enum.Enum):
    """Classification of causal chain mechanism."""

    OBSERVATION_DEPENDENCY = "observation_dependency"
    PHYSICS_MEDIATED = "physics_mediated"
    COMMUNICATION = "communication"
    ENVIRONMENT_MEDIATED = "environment_mediated"
    TRANSITIVE = "transitive"
    USER_ANNOTATED = "user_annotated"


class SoundnessClassification(enum.Enum):
    """Soundness classification for HB edges.

    - EXACT: the edge is precisely derived from program semantics.
    - OVER_APPROXIMATE: the edge may not exist in all executions
      but including it is sound (conservative).
    - USER_ANNOTATED: the edge was provided by the user and not
      automatically derived.
    """

    EXACT = "exact"
    OVER_APPROXIMATE = "over-approximate"
    USER_ANNOTATED = "user-annotated"


@dataclass
class CausalEvidence:
    """Structured evidence supporting a causal edge.

    Attributes:
        chain_type: The type of causal chain that produced this evidence.
        soundness: Soundness classification of the edge.
        supporting_edges: For transitive edges, the constituent edges.
        state_variable: For environment-mediated edges, the state variable.
        delay: Temporal delay between cause and effect.
        confidence: Confidence score in [0, 1].
    """

    chain_type: CausalChainType
    soundness: SoundnessClassification
    supporting_edges: List[Tuple[str, str]] = field(default_factory=list)
    state_variable: Optional[str] = None
    delay: Optional[int] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_type": self.chain_type.value,
            "soundness": self.soundness.value,
            "supporting_edges": self.supporting_edges,
            "state_variable": self.state_variable,
            "delay": self.delay,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalEvidence":
        return cls(
            chain_type=CausalChainType(d["chain_type"]),
            soundness=SoundnessClassification(d["soundness"]),
            supporting_edges=d.get("supporting_edges", []),
            state_variable=d.get("state_variable"),
            delay=d.get("delay"),
            confidence=d.get("confidence", 1.0),
        )


# ======================================================================
# Data structures for events and state
# ======================================================================

@dataclass(frozen=True)
class AgentEvent:
    """A single event in an agent's execution trace.

    Attributes:
        event_id: Globally unique identifier for this event.
        agent_id: The agent that produced the event.
        timestep: Discrete timestep when the event occurred.
        action: The action taken (may be None for observation-only events).
        observation: The observation received (may be None for action-only events).
        state_snapshot: Optional snapshot of relevant state dimensions.
    """
    event_id: str
    agent_id: str
    timestep: int
    action: Optional[np.ndarray] = None
    observation: Optional[np.ndarray] = None
    state_snapshot: Optional[Dict[str, float]] = None


@dataclass
class CausalEdge:
    """A discovered causal relationship between two events.

    Attributes:
        source: Event ID of the cause.
        target: Event ID of the effect.
        mechanism: How causality was established.
        confidence: Confidence score in [0, 1].
        metadata: Additional details about the causal link.
    """
    source: str
    target: str
    mechanism: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )


# ======================================================================
# ObservationDependencyAnalyzer
# ======================================================================

class ObservationDependencyAnalyzer:
    """Determine which agents' actions affected which agents' observations.

    Given a sequence of agent events and environment state transitions,
    identifies when agent A's action at timestep t causally influenced
    agent B's observation at timestep t' > t.

    The analysis compares the observation change to the expected change
    under a null model (no interaction) and flags significant deviations.

    Args:
        influence_threshold: Minimum observation change magnitude
            attributable to another agent's action to be considered causal.
        max_delay: Maximum timestep lag to consider for causal influence.
    """

    def __init__(
        self,
        influence_threshold: float = 0.01,
        max_delay: int = 5,
    ) -> None:
        self._threshold = influence_threshold
        self._max_delay = max_delay

    def analyze(
        self,
        events: List[AgentEvent],
        env_states: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[CausalEdge]:
        """Identify observation-dependency causal edges.

        For each pair (action_event, observation_event) where the acting
        agent differs from the observing agent and the timestep gap is
        within ``max_delay``, we check whether the action could have
        influenced the observation.

        Args:
            events: Chronologically ordered agent events.
            env_states: Optional mapping from timestep to environment state.

        Returns:
            List of :class:`CausalEdge` instances.
        """
        edges: List[CausalEdge] = []
        action_events = [e for e in events if e.action is not None]
        obs_events = [e for e in events if e.observation is not None]

        # Index observations by timestep for efficient lookup
        obs_by_step: Dict[int, List[AgentEvent]] = defaultdict(list)
        for oe in obs_events:
            obs_by_step[oe.timestep].append(oe)

        for ae in action_events:
            for dt in range(1, self._max_delay + 1):
                target_step = ae.timestep + dt
                for oe in obs_by_step.get(target_step, []):
                    if oe.agent_id == ae.agent_id:
                        continue
                    conf = self._compute_influence(ae, oe, env_states)
                    if conf >= self._threshold:
                        edges.append(CausalEdge(
                            source=ae.event_id,
                            target=oe.event_id,
                            mechanism="observation_dependency",
                            confidence=conf,
                            metadata={
                                "delay": dt,
                                "source_agent": ae.agent_id,
                                "target_agent": oe.agent_id,
                            },
                        ))
        return edges

    def _compute_influence(
        self,
        action_event: AgentEvent,
        obs_event: AgentEvent,
        env_states: Optional[Dict[int, Dict[str, float]]],
    ) -> float:
        """Estimate influence of action on observation.

        Uses state-space overlap heuristic: if the environment state
        dimensions modified by the action overlap with those observed,
        and the magnitude of change is above threshold, we assign
        confidence proportional to the magnitude.
        """
        if action_event.state_snapshot is None or obs_event.state_snapshot is None:
            # Without state snapshots, fall back to timestep proximity
            delay = obs_event.timestep - action_event.timestep
            return max(0.0, 1.0 - delay * 0.2)

        shared_dims = (
            set(action_event.state_snapshot.keys())
            & set(obs_event.state_snapshot.keys())
        )
        if not shared_dims:
            return 0.0

        deltas = [
            abs(action_event.state_snapshot[d] - obs_event.state_snapshot[d])
            for d in shared_dims
        ]
        max_delta = max(deltas)
        # Sigmoid-like mapping to [0, 1]
        return float(1.0 - np.exp(-max_delta))


# ======================================================================
# PhysicsMediatedCausalityDetector
# ======================================================================

class PhysicsMediatedCausalityDetector:
    """Detect causal chains mediated by physics simulation.

    In many MARL environments, agent A's action changes its own position
    or the physical state of the environment, which in turn changes
    agent B's sensor readings.  This detector identifies such chains by
    tracking state deltas attributable to each agent and correlating them
    with observation changes in other agents.

    Args:
        influence_radius: Maximum state-space distance within which
            physics-mediated influence is considered plausible.
        min_confidence: Minimum confidence threshold for reporting.
    """

    def __init__(
        self,
        influence_radius: float = 5.0,
        min_confidence: float = 0.1,
    ) -> None:
        self._radius = influence_radius
        self._min_confidence = min_confidence

    def detect(
        self,
        events: List[AgentEvent],
        agent_positions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    ) -> List[CausalEdge]:
        """Scan for physics-mediated causal links.

        Args:
            events: Chronologically ordered agent events.
            agent_positions: Optional mapping
                ``agent_id -> {timestep -> position_vector}``.

        Returns:
            Discovered causal edges.
        """
        edges: List[CausalEdge] = []
        if agent_positions is None:
            return edges

        agents = list(agent_positions.keys())
        timesteps = sorted(
            {t for pos in agent_positions.values() for t in pos.keys()}
        )

        for t in timesteps:
            for i, a in enumerate(agents):
                if t not in agent_positions[a]:
                    continue
                pos_a = agent_positions[a][t]
                for j in range(i + 1, len(agents)):
                    b = agents[j]
                    if t not in agent_positions[b]:
                        continue
                    pos_b = agent_positions[b][t]
                    dist = float(np.linalg.norm(pos_a - pos_b))
                    if dist > self._radius:
                        continue
                    confidence = max(0.0, 1.0 - dist / self._radius)
                    if confidence < self._min_confidence:
                        continue
                    # Find the action events for both agents at this timestep
                    a_events = [
                        e for e in events
                        if e.agent_id == a and e.timestep == t and e.action is not None
                    ]
                    b_obs = [
                        e for e in events
                        if e.agent_id == b and e.timestep == t + 1 and e.observation is not None
                    ]
                    for ae in a_events:
                        for be in b_obs:
                            edges.append(CausalEdge(
                                source=ae.event_id,
                                target=be.event_id,
                                mechanism="physics_mediated",
                                confidence=confidence,
                                metadata={
                                    "distance": dist,
                                    "radius": self._radius,
                                    "timestep": t,
                                },
                            ))
                    # Symmetric: B's action could affect A's observation
                    b_actions = [
                        e for e in events
                        if e.agent_id == b and e.timestep == t and e.action is not None
                    ]
                    a_obs = [
                        e for e in events
                        if e.agent_id == a and e.timestep == t + 1 and e.observation is not None
                    ]
                    for be in b_actions:
                        for ae in a_obs:
                            edges.append(CausalEdge(
                                source=be.event_id,
                                target=ae.event_id,
                                mechanism="physics_mediated",
                                confidence=confidence,
                                metadata={
                                    "distance": dist,
                                    "radius": self._radius,
                                    "timestep": t,
                                },
                            ))
        return edges


# ======================================================================
# CommunicationCausalityExtractor
# ======================================================================

@dataclass(frozen=True)
class CommunicationEvent:
    """A message exchanged between two agents.

    Attributes:
        message_id: Unique message identifier.
        sender: Agent that sent the message.
        receiver: Agent that received the message.
        send_timestep: Timestep when the message was sent.
        receive_timestep: Timestep when the message was received.
        channel: Optional communication channel name.
    """
    message_id: str
    sender: str
    receiver: str
    send_timestep: int
    receive_timestep: int
    channel: str = "default"


class CommunicationCausalityExtractor:
    """Extract HB edges from explicit communication events.

    Every send–receive pair establishes a happens-before edge from the
    sender's send event to the receiver's receive event.

    This is the most straightforward source of HB edges and typically
    has the highest confidence.
    """

    def extract(
        self,
        agent_events: List[AgentEvent],
        comm_events: List[CommunicationEvent],
    ) -> List[CausalEdge]:
        """Map communication events to causal edges.

        For each :class:`CommunicationEvent`, find the closest matching
        agent events for sender and receiver and emit a causal edge.

        Args:
            agent_events: All agent events in the trace.
            comm_events: Explicit communication events.

        Returns:
            Causal edges derived from communication.
        """
        # Index agent events by (agent_id, timestep) for fast lookup
        event_index: Dict[Tuple[str, int], List[AgentEvent]] = defaultdict(list)
        for ae in agent_events:
            event_index[(ae.agent_id, ae.timestep)].append(ae)

        edges: List[CausalEdge] = []
        for ce in comm_events:
            send_events = event_index.get((ce.sender, ce.send_timestep), [])
            recv_events = event_index.get((ce.receiver, ce.receive_timestep), [])

            if not send_events or not recv_events:
                logger.debug(
                    "No matching agent events for communication %s", ce.message_id
                )
                continue

            # Use the first matching event on each side
            src = send_events[0]
            tgt = recv_events[0]
            edges.append(CausalEdge(
                source=src.event_id,
                target=tgt.event_id,
                mechanism="communication",
                confidence=1.0,
                metadata={
                    "message_id": ce.message_id,
                    "channel": ce.channel,
                    "send_timestep": ce.send_timestep,
                    "receive_timestep": ce.receive_timestep,
                },
            ))
        return edges


# ======================================================================
# EnvironmentMediatedCausalChain
# ======================================================================

class EnvironmentMediatedCausalChain:
    """Track causal chains through shared environment state.

    When multiple agents interact through a shared environment (e.g.
    shared memory, a common resource), writes by one agent can affect
    subsequent reads by another.  This class models such chains by
    tracking per-dimension write/read events and constructing HB edges
    from the latest write to each subsequent read.

    Args:
        state_dimensions: Names of tracked environment state dimensions.
        staleness_limit: Maximum timestep gap between write and read
            for the causal link to be considered valid.
    """

    def __init__(
        self,
        state_dimensions: Optional[List[str]] = None,
        staleness_limit: int = 10,
    ) -> None:
        self._dims = set(state_dimensions) if state_dimensions else set()
        self._staleness = staleness_limit
        # Per-dimension: latest write event
        self._latest_write: Dict[str, Tuple[str, int]] = {}

    def register_write(
        self,
        event_id: str,
        agent_id: str,
        timestep: int,
        dimensions: Set[str],
    ) -> None:
        """Record that *agent_id* wrote to *dimensions* at *timestep*."""
        for dim in dimensions:
            self._dims.add(dim)
            prev = self._latest_write.get(dim)
            if prev is None or timestep >= prev[1]:
                self._latest_write[dim] = (event_id, timestep)

    def register_read(
        self,
        event_id: str,
        agent_id: str,
        timestep: int,
        dimensions: Set[str],
    ) -> List[CausalEdge]:
        """Record a read and return causal edges from prior writes.

        Args:
            event_id: The read event's ID.
            agent_id: Agent performing the read.
            timestep: When the read occurred.
            dimensions: State dimensions being read.

        Returns:
            Causal edges from the most recent write in each dimension.
        """
        edges: List[CausalEdge] = []
        for dim in dimensions:
            if dim not in self._latest_write:
                continue
            write_eid, write_ts = self._latest_write[dim]
            if write_eid == event_id:
                continue
            delay = timestep - write_ts
            if delay < 0 or delay > self._staleness:
                continue
            confidence = max(0.0, 1.0 - delay / (self._staleness + 1))
            edges.append(CausalEdge(
                source=write_eid,
                target=event_id,
                mechanism="environment_mediated",
                confidence=confidence,
                metadata={
                    "dimension": dim,
                    "write_timestep": write_ts,
                    "read_timestep": timestep,
                    "delay": delay,
                },
            ))
        return edges

    def process_trace(
        self,
        events: List[AgentEvent],
        write_dims_fn,
        read_dims_fn,
    ) -> List[CausalEdge]:
        """Process an entire trace and return all environment-mediated edges.

        Args:
            events: Chronologically ordered agent events.
            write_dims_fn: Callable(event) -> Set[str] returning dimensions
                written by the event (empty set if none).
            read_dims_fn: Callable(event) -> Set[str] returning dimensions
                read by the event (empty set if none).

        Returns:
            All discovered causal edges.
        """
        all_edges: List[CausalEdge] = []
        for ev in sorted(events, key=lambda e: e.timestep):
            w_dims = write_dims_fn(ev)
            if w_dims:
                self.register_write(ev.event_id, ev.agent_id, ev.timestep, w_dims)
            r_dims = read_dims_fn(ev)
            if r_dims:
                new_edges = self.register_read(
                    ev.event_id, ev.agent_id, ev.timestep, r_dims,
                )
                all_edges.extend(new_edges)
        return all_edges

    def reset(self) -> None:
        """Clear all tracked state."""
        self._latest_write.clear()


# ======================================================================
# CausalInferenceEngine
# ======================================================================

class CausalInferenceEngine:
    """Unified causal inference combining all sources.

    Merges edges from observation dependencies, physics-mediated causality,
    communication, and environment-mediated chains.  Applies configurable
    confidence thresholds and deduplication.

    Args:
        obs_analyzer: Observation dependency analyzer.
        physics_detector: Physics-mediated causality detector.
        comm_extractor: Communication causality extractor.
        env_chain: Environment-mediated causal chain tracker.
        min_confidence: Global minimum confidence threshold.
    """

    def __init__(
        self,
        obs_analyzer: Optional[ObservationDependencyAnalyzer] = None,
        physics_detector: Optional[PhysicsMediatedCausalityDetector] = None,
        comm_extractor: Optional[CommunicationCausalityExtractor] = None,
        env_chain: Optional[EnvironmentMediatedCausalChain] = None,
        min_confidence: float = 0.1,
    ) -> None:
        self._obs = obs_analyzer or ObservationDependencyAnalyzer()
        self._phys = physics_detector or PhysicsMediatedCausalityDetector()
        self._comm = comm_extractor or CommunicationCausalityExtractor()
        self._env = env_chain or EnvironmentMediatedCausalChain()
        self._min_confidence = min_confidence

    def infer(
        self,
        events: List[AgentEvent],
        comm_events: Optional[List[CommunicationEvent]] = None,
        env_states: Optional[Dict[int, Dict[str, float]]] = None,
        agent_positions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    ) -> List[CausalEdge]:
        """Run all inference sources and return merged, deduplicated edges.

        Args:
            events: All agent events in the trace.
            comm_events: Explicit communication events.
            env_states: Timestep -> environment state mapping.
            agent_positions: Agent position trajectories for physics analysis.

        Returns:
            Deduplicated causal edges above the confidence threshold.
        """
        all_edges: List[CausalEdge] = []

        # 1. Observation dependencies
        obs_edges = self._obs.analyze(events, env_states)
        all_edges.extend(obs_edges)
        logger.info("Observation analysis produced %d edges", len(obs_edges))

        # 2. Physics-mediated
        phys_edges = self._phys.detect(events, agent_positions)
        all_edges.extend(phys_edges)
        logger.info("Physics analysis produced %d edges", len(phys_edges))

        # 3. Communication
        if comm_events:
            comm_edges = self._comm.extract(events, comm_events)
            all_edges.extend(comm_edges)
            logger.info("Communication analysis produced %d edges", len(comm_edges))

        # 4. Environment-mediated (if state data available)
        if env_states:
            env_edges = self._env_analysis(events, env_states)
            all_edges.extend(env_edges)
            logger.info("Environment analysis produced %d edges", len(env_edges))

        # Filter and deduplicate
        filtered = self._filter_and_dedup(all_edges)
        logger.info(
            "Total causal edges: %d (from %d raw)", len(filtered), len(all_edges)
        )
        return filtered

    def _env_analysis(
        self,
        events: List[AgentEvent],
        env_states: Dict[int, Dict[str, float]],
    ) -> List[CausalEdge]:
        """Run environment-mediated analysis using state snapshots."""
        all_dims = set()
        for state in env_states.values():
            all_dims.update(state.keys())

        chain = EnvironmentMediatedCausalChain(
            state_dimensions=list(all_dims),
        )

        def write_dims(ev: AgentEvent) -> Set[str]:
            if ev.action is not None and ev.state_snapshot:
                return set(ev.state_snapshot.keys())
            return set()

        def read_dims(ev: AgentEvent) -> Set[str]:
            if ev.observation is not None and ev.state_snapshot:
                return set(ev.state_snapshot.keys())
            return set()

        return chain.process_trace(events, write_dims, read_dims)

    def _filter_and_dedup(
        self, edges: List[CausalEdge],
    ) -> List[CausalEdge]:
        """Filter by confidence threshold and keep the highest-confidence
        edge for each (source, target) pair."""
        best: Dict[Tuple[str, str], CausalEdge] = {}
        for edge in edges:
            if edge.confidence < self._min_confidence:
                continue
            key = (edge.source, edge.target)
            if key not in best or edge.confidence > best[key].confidence:
                best[key] = edge
        return list(best.values())

    def infer_to_dict(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Convenience: run inference and return JSON-serializable dicts."""
        edges = self.infer(**kwargs)
        return [
            {
                "source": e.source,
                "target": e.target,
                "mechanism": e.mechanism,
                "confidence": e.confidence,
                "metadata": e.metadata,
            }
            for e in edges
        ]


# ======================================================================
# TransitiveCausalityClosure
# ======================================================================

class TransitiveCausalityClosure:
    """Compute bounded-depth transitive closure over causal edges.

    Given a set of causal edges, derives transitive edges up to a
    configurable maximum depth.  Each transitive edge carries evidence
    recording the constituent path.

    Args:
        max_depth: Maximum transitive chain length (default 5).
        min_confidence: Minimum confidence for derived edges.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_confidence: float = 0.05,
    ) -> None:
        self._max_depth = max_depth
        self._min_confidence = min_confidence

    def close(self, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Compute bounded transitive closure.

        New transitive edges have confidence equal to the product of
        confidences along the path and mechanism "transitive".

        Args:
            edges: Input causal edges.

        Returns:
            Original edges plus derived transitive edges.
        """
        # Build adjacency map: source -> [(target, confidence, edge)]
        adj: Dict[str, List[Tuple[str, float, CausalEdge]]] = defaultdict(list)
        for e in edges:
            adj[e.source].append((e.target, e.confidence, e))

        existing = {(e.source, e.target) for e in edges}
        new_edges: List[CausalEdge] = []

        # BFS from each source up to max_depth
        for e in edges:
            self._explore(
                start=e.source,
                adj=adj,
                existing=existing,
                new_edges=new_edges,
            )

        return list(edges) + new_edges

    def _explore(
        self,
        start: str,
        adj: Dict[str, List[Tuple[str, float, CausalEdge]]],
        existing: Set[Tuple[str, str]],
        new_edges: List[CausalEdge],
    ) -> None:
        """BFS from start to find transitive edges."""
        # (current_node, cumulative_confidence, path_edges, depth)
        queue: List[Tuple[str, float, List[Tuple[str, str]], int]] = [
            (start, 1.0, [], 0)
        ]
        visited_at_depth: Dict[str, int] = {start: 0}

        while queue:
            node, conf, path, depth = queue.pop(0)
            if depth >= self._max_depth:
                continue

            for target, edge_conf, edge in adj.get(node, []):
                new_conf = conf * edge_conf
                if new_conf < self._min_confidence:
                    continue

                new_path = path + [(edge.source, edge.target)]
                new_depth = depth + 1

                # Only add if it's a genuinely new edge
                key = (start, target)
                if start != target and key not in existing:
                    existing.add(key)
                    new_edges.append(CausalEdge(
                        source=start,
                        target=target,
                        mechanism="transitive",
                        confidence=new_conf,
                        metadata={
                            "path": new_path,
                            "depth": new_depth,
                        },
                    ))

                if target not in visited_at_depth or visited_at_depth[target] > new_depth:
                    visited_at_depth[target] = new_depth
                    queue.append((target, new_conf, new_path, new_depth))

    def close_with_evidence(
        self, edges: List[CausalEdge],
    ) -> Tuple[List[CausalEdge], List[CausalEvidence]]:
        """Compute closure and produce CausalEvidence for each new edge."""
        closed = self.close(edges)
        evidence: List[CausalEvidence] = []
        for e in closed:
            if e.mechanism == "transitive":
                ev = CausalEvidence(
                    chain_type=CausalChainType.TRANSITIVE,
                    soundness=SoundnessClassification.OVER_APPROXIMATE,
                    supporting_edges=e.metadata.get("path", []),
                    delay=e.metadata.get("depth"),
                    confidence=e.confidence,
                )
            else:
                chain_map = {
                    "observation_dependency": CausalChainType.OBSERVATION_DEPENDENCY,
                    "physics_mediated": CausalChainType.PHYSICS_MEDIATED,
                    "communication": CausalChainType.COMMUNICATION,
                    "environment_mediated": CausalChainType.ENVIRONMENT_MEDIATED,
                }
                ct = chain_map.get(e.mechanism, CausalChainType.OBSERVATION_DEPENDENCY)
                sc = (SoundnessClassification.EXACT
                      if e.mechanism == "communication"
                      else SoundnessClassification.OVER_APPROXIMATE)
                ev = CausalEvidence(
                    chain_type=ct,
                    soundness=sc,
                    state_variable=e.metadata.get("dimension"),
                    delay=e.metadata.get("delay"),
                    confidence=e.confidence,
                )
            evidence.append(ev)
        return closed, evidence


# ======================================================================
# classify_edge_soundness
# ======================================================================

def classify_edge_soundness(edge: CausalEdge) -> SoundnessClassification:
    """Classify the soundness of a causal edge based on its mechanism.

    - Communication edges are exact (direct message passing).
    - Physics and observation edges are over-approximate.
    - User-provided edges are user-annotated.
    """
    if edge.mechanism == "communication":
        return SoundnessClassification.EXACT
    if edge.mechanism in ("physics_mediated", "observation_dependency",
                          "environment_mediated", "transitive"):
        return SoundnessClassification.OVER_APPROXIMATE
    if edge.mechanism == "user_annotated":
        return SoundnessClassification.USER_ANNOTATED
    return SoundnessClassification.OVER_APPROXIMATE
