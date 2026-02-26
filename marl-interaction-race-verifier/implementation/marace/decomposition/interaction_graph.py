"""Interaction graph construction and analysis for compositional decomposition.

Builds a weighted graph where nodes represent agents and edges represent
interactions weighted by coupling strength.  The graph is constructed from
happens-before (HB) traces and updated dynamically as new causal
dependencies are discovered during verification.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations and lightweight data containers
# ---------------------------------------------------------------------------

class InteractionType(enum.Enum):
    """Classification of causal interaction between two agents."""

    OBSERVATION = "observation"
    COMMUNICATION = "communication"
    PHYSICS = "physics"


@dataclass(frozen=True)
class InteractionEdge:
    """A directed interaction between two agents.

    Attributes:
        source_agent: Agent ID that causally influences the target.
        target_agent: Agent ID that is influenced.
        interaction_type: Nature of the causal channel.
        strength: Non-negative coupling strength in [0, 1].
        metadata: Optional dictionary of extra attributes (e.g. HB edge IDs).
    """

    source_agent: str
    target_agent: str
    interaction_type: InteractionType
    strength: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strength < 0.0 or self.strength > 1.0:
            raise ValueError(
                f"Interaction strength must be in [0, 1], got {self.strength}"
            )
        if self.source_agent == self.target_agent:
            raise ValueError("Self-loops are not allowed in the interaction graph")


# ---------------------------------------------------------------------------
# InteractionGraph
# ---------------------------------------------------------------------------

class InteractionGraph:
    """Weighted directed graph of agent interactions.

    Wraps a ``networkx.DiGraph`` where each node is an agent ID and each
    edge carries an ``InteractionEdge`` payload.  Provides convenience
    methods for querying coupling strength, connected components, and
    neighbourhood structure needed by the partitioning algorithms.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._agents: Set[str] = set()

    # -- Mutation ----------------------------------------------------------

    def add_agent(self, agent_id: str) -> None:
        """Register an agent as a node (idempotent)."""
        self._agents.add(agent_id)
        self._graph.add_node(agent_id)

    def add_interaction(self, edge: InteractionEdge) -> None:
        """Add or update an interaction edge.

        If an edge between the same pair already exists for the same
        interaction type, the maximum strength is retained.
        """
        self.add_agent(edge.source_agent)
        self.add_agent(edge.target_agent)

        key = (edge.source_agent, edge.target_agent)
        existing = self._graph.edges.get(key, {}).get("interactions", [])

        # Replace if same type already present with lower strength
        updated: List[InteractionEdge] = []
        replaced = False
        for e in existing:
            if e.interaction_type == edge.interaction_type:
                updated.append(edge if edge.strength >= e.strength else e)
                replaced = True
            else:
                updated.append(e)
        if not replaced:
            updated.append(edge)

        aggregate_strength = max(e.strength for e in updated)
        self._graph.add_edge(
            edge.source_agent,
            edge.target_agent,
            interactions=updated,
            weight=aggregate_strength,
        )

    def remove_interaction(
        self, source: str, target: str, itype: Optional[InteractionType] = None
    ) -> None:
        """Remove interaction(s) between *source* and *target*.

        If *itype* is ``None`` the entire edge is removed.  Otherwise only
        the specified interaction type is removed and the edge is deleted
        only if no interactions remain.
        """
        if not self._graph.has_edge(source, target):
            return
        if itype is None:
            self._graph.remove_edge(source, target)
            return
        data = self._graph.edges[source, target]
        remaining = [e for e in data["interactions"] if e.interaction_type != itype]
        if not remaining:
            self._graph.remove_edge(source, target)
        else:
            data["interactions"] = remaining
            data["weight"] = max(e.strength for e in remaining)

    # -- Queries -----------------------------------------------------------

    @property
    def agents(self) -> FrozenSet[str]:
        return frozenset(self._agents)

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def networkx_graph(self) -> nx.DiGraph:
        """Direct access to the underlying networkx graph (read-only use)."""
        return self._graph

    def coupling_strength(self, source: str, target: str) -> float:
        """Return aggregate coupling strength, 0 if no edge."""
        if self._graph.has_edge(source, target):
            return float(self._graph.edges[source, target]["weight"])
        return 0.0

    def interactions_between(
        self, source: str, target: str
    ) -> List[InteractionEdge]:
        """All interaction edges from *source* to *target*."""
        if not self._graph.has_edge(source, target):
            return []
        return list(self._graph.edges[source, target].get("interactions", []))

    def neighbours(self, agent_id: str) -> Set[str]:
        """Agents that *agent_id* interacts with (in either direction)."""
        preds = set(self._graph.predecessors(agent_id))
        succs = set(self._graph.successors(agent_id))
        return preds | succs

    def connected_components(self) -> List[FrozenSet[str]]:
        """Weakly-connected components of the interaction graph.

        Each component is a candidate interaction group for compositional
        decomposition.
        """
        undirected = self._graph.to_undirected()
        return [
            frozenset(comp) for comp in nx.connected_components(undirected)
        ]

    def subgraph(self, agents: Set[str]) -> "InteractionGraph":
        """Induced subgraph over a subset of agents."""
        sub = InteractionGraph()
        for a in agents:
            if a in self._agents:
                sub.add_agent(a)
        for u, v, data in self._graph.edges(data=True):
            if u in agents and v in agents:
                for edge in data.get("interactions", []):
                    sub.add_interaction(edge)
        return sub

    def adjacency_matrix(self, agent_order: Optional[List[str]] = None) -> np.ndarray:
        """Weighted adjacency matrix (symmetric max of directed weights).

        Parameters:
            agent_order: Fixed ordering of agents for matrix rows/columns.
                         Defaults to sorted agent IDs.

        Returns:
            Square numpy array of shape ``(N, N)``.
        """
        if agent_order is None:
            agent_order = sorted(self._agents)
        idx = {a: i for i, a in enumerate(agent_order)}
        n = len(agent_order)
        mat = np.zeros((n, n), dtype=np.float64)
        for u, v, data in self._graph.edges(data=True):
            if u not in idx or v not in idx:
                continue
            w = data.get("weight", 0.0)
            i, j = idx[u], idx[v]
            mat[i, j] = max(mat[i, j], w)
            mat[j, i] = max(mat[j, i], w)
        return mat

    def degree_matrix(self, agent_order: Optional[List[str]] = None) -> np.ndarray:
        """Diagonal degree matrix corresponding to :pymethod:`adjacency_matrix`."""
        adj = self.adjacency_matrix(agent_order)
        return np.diag(adj.sum(axis=1))

    def laplacian_matrix(self, agent_order: Optional[List[str]] = None) -> np.ndarray:
        """Graph Laplacian ``L = D - A``."""
        adj = self.adjacency_matrix(agent_order)
        deg = np.diag(adj.sum(axis=1))
        return deg - adj

    def total_coupling(self) -> float:
        """Sum of all edge weights."""
        return float(sum(d["weight"] for _, _, d in self._graph.edges(data=True)))

    def iter_edges(self) -> Iterator[InteractionEdge]:
        """Iterate over all interaction edges."""
        for _u, _v, data in self._graph.edges(data=True):
            yield from data.get("interactions", [])


# ---------------------------------------------------------------------------
# InteractionStrengthMetrics
# ---------------------------------------------------------------------------

class InteractionStrengthMetrics:
    """Quantify coupling strength between agent pairs.

    Metrics are computed from state trajectory data collected during
    simulation or trace replay.
    """

    @staticmethod
    def mutual_information(
        traj_a: np.ndarray,
        traj_b: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Estimate mutual information between two state trajectories.

        Uses histogram-based discretisation.  Trajectories are 1-D
        or are reduced to 1-D via PCA before binning.

        Parameters:
            traj_a: Shape ``(T,)`` or ``(T, d)`` state trajectory of agent A.
            traj_b: Shape ``(T,)`` or ``(T, d)`` state trajectory of agent B.
            n_bins: Number of histogram bins per dimension.

        Returns:
            Non-negative estimated MI in nats.
        """
        a = InteractionStrengthMetrics._reduce_to_1d(traj_a)
        b = InteractionStrengthMetrics._reduce_to_1d(traj_b)
        T = min(len(a), len(b))
        a, b = a[:T], b[:T]

        hist_ab, x_edges, y_edges = np.histogram2d(a, b, bins=n_bins)
        p_ab = hist_ab / hist_ab.sum()
        p_a = p_ab.sum(axis=1)
        p_b = p_ab.sum(axis=0)

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_ab[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                    mi += p_ab[i, j] * math.log(p_ab[i, j] / (p_a[i] * p_b[j]))
        return max(mi, 0.0)

    @staticmethod
    def hb_edge_density(
        hb_edges: Sequence[Tuple[str, str]],
        source: str,
        target: str,
        total_events: int,
    ) -> float:
        """Fraction of HB edges connecting *source* → *target*.

        Parameters:
            hb_edges: List of (from_agent, to_agent) HB edges.
            source: Agent ID.
            target: Agent ID.
            total_events: Total number of events in the trace.

        Returns:
            Density in [0, 1].
        """
        if total_events <= 0:
            return 0.0
        count = sum(1 for s, t in hb_edges if s == source and t == target)
        return min(count / total_events, 1.0)

    @staticmethod
    def observation_overlap(
        obs_a: np.ndarray, obs_b: np.ndarray, threshold: float = 0.1
    ) -> float:
        """Fraction of timesteps where agents' observations overlap.

        Two observations *overlap* at timestep *t* if at least one
        coordinate pair differs by less than *threshold*.

        Parameters:
            obs_a: Shape ``(T, d)`` observation trajectory of agent A.
            obs_b: Shape ``(T, d)`` observation trajectory of agent B.
            threshold: Distance threshold for overlap.

        Returns:
            Overlap ratio in [0, 1].
        """
        T = min(len(obs_a), len(obs_b))
        if T == 0:
            return 0.0
        diffs = np.abs(obs_a[:T] - obs_b[:T])
        overlap_mask = np.any(diffs < threshold, axis=1)
        return float(overlap_mask.mean())

    @staticmethod
    def combined_strength(
        mi: float,
        hb_density: float,
        obs_overlap: float,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> float:
        """Weighted combination of metrics, normalised to [0, 1].

        Parameters:
            mi: Mutual information (nats).  Normalised via ``tanh``.
            hb_density: HB edge density in [0, 1].
            obs_overlap: Observation overlap ratio in [0, 1].
            weights: Relative weights for (MI, HB-density, overlap).

        Returns:
            Combined strength in [0, 1].
        """
        w_mi, w_hb, w_obs = weights
        norm_mi = math.tanh(mi)  # maps [0, ∞) → [0, 1)
        total = w_mi * norm_mi + w_hb * hb_density + w_obs * obs_overlap
        return float(np.clip(total, 0.0, 1.0))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _reduce_to_1d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr
        # PCA first component
        centered = arr - arr.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            return centered.ravel()
        eigvals, eigvecs = np.linalg.eigh(cov)
        return centered @ eigvecs[:, -1]


# ---------------------------------------------------------------------------
# GraphConstructor
# ---------------------------------------------------------------------------

@dataclass
class HBEvent:
    """Lightweight representation of a happens-before event."""

    event_id: str
    agent_id: str
    timestamp: float
    state: Optional[np.ndarray] = None


@dataclass
class HBEdge:
    """Directed edge in the HB partial order."""

    source_event: str
    target_event: str
    source_agent: str
    target_agent: str
    cause: InteractionType


class GraphConstructor:
    """Build an :class:`InteractionGraph` from HB graph and trace data.

    The construction proceeds in three phases:

    1. Extract inter-agent HB edges (ignore intra-agent edges).
    2. Compute coupling-strength metrics for each agent pair.
    3. Populate the ``InteractionGraph`` with weighted edges.
    """

    def __init__(
        self,
        mi_bins: int = 20,
        strength_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        obs_overlap_threshold: float = 0.1,
    ) -> None:
        self._mi_bins = mi_bins
        self._strength_weights = strength_weights
        self._obs_threshold = obs_overlap_threshold

    def build(
        self,
        agents: Sequence[str],
        hb_edges: Sequence[HBEdge],
        trajectories: Optional[Dict[str, np.ndarray]] = None,
        observations: Optional[Dict[str, np.ndarray]] = None,
    ) -> InteractionGraph:
        """Construct the interaction graph.

        Parameters:
            agents: List of agent IDs.
            hb_edges: Directed HB edges from the partial-order engine.
            trajectories: Per-agent state trajectories ``{agent_id: (T, d)}``.
            observations: Per-agent observation trajectories.

        Returns:
            Populated :class:`InteractionGraph`.
        """
        graph = InteractionGraph()
        for a in agents:
            graph.add_agent(a)

        # Group HB edges by ordered agent pair
        inter_edges: Dict[Tuple[str, str], List[HBEdge]] = {}
        for e in hb_edges:
            if e.source_agent != e.target_agent:
                key = (e.source_agent, e.target_agent)
                inter_edges.setdefault(key, []).append(e)

        total_events = len(hb_edges) if hb_edges else 1

        for (src, tgt), edges in inter_edges.items():
            # Determine dominant interaction type
            type_counts: Dict[InteractionType, int] = {}
            for e in edges:
                type_counts[e.cause] = type_counts.get(e.cause, 0) + 1
            dominant_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

            # Compute strength
            hb_density = InteractionStrengthMetrics.hb_edge_density(
                [(e.source_agent, e.target_agent) for e in hb_edges],
                src,
                tgt,
                total_events,
            )

            mi = 0.0
            if trajectories and src in trajectories and tgt in trajectories:
                mi = InteractionStrengthMetrics.mutual_information(
                    trajectories[src], trajectories[tgt], self._mi_bins
                )

            overlap = 0.0
            if observations and src in observations and tgt in observations:
                overlap = InteractionStrengthMetrics.observation_overlap(
                    observations[src], observations[tgt], self._obs_threshold
                )

            strength = InteractionStrengthMetrics.combined_strength(
                mi, hb_density, overlap, self._strength_weights
            )

            graph.add_interaction(
                InteractionEdge(
                    source_agent=src,
                    target_agent=tgt,
                    interaction_type=dominant_type,
                    strength=strength,
                )
            )

        return graph


# ---------------------------------------------------------------------------
# DynamicGraphUpdater
# ---------------------------------------------------------------------------

class DynamicGraphUpdater:
    """Incrementally update an :class:`InteractionGraph` as new interactions
    are discovered during analysis.

    The MARACE pipeline may discover previously-unseen causal dependencies
    during abstract interpretation.  When this happens the HB graph—and
    therefore the interaction graph—must be refreshed, and the compositional
    decomposition recomputed from scratch (per the static-decomposition
    design principle).
    """

    def __init__(self, graph: InteractionGraph) -> None:
        self._graph = graph
        self._pending_edges: List[InteractionEdge] = []
        self._version: int = 0

    @property
    def version(self) -> int:
        """Monotonically increasing version counter."""
        return self._version

    @property
    def graph(self) -> InteractionGraph:
        return self._graph

    def stage_interaction(self, edge: InteractionEdge) -> None:
        """Stage a new interaction for batch application."""
        self._pending_edges.append(edge)

    def apply_pending(self) -> bool:
        """Apply all staged interactions.

        Returns:
            ``True`` if any new edge was added or any strength increased
            (indicating the decomposition should be recomputed).
        """
        changed = False
        for edge in self._pending_edges:
            old_strength = self._graph.coupling_strength(
                edge.source_agent, edge.target_agent
            )
            self._graph.add_interaction(edge)
            new_strength = self._graph.coupling_strength(
                edge.source_agent, edge.target_agent
            )
            if new_strength > old_strength + 1e-12:
                changed = True
        self._pending_edges.clear()
        if changed:
            self._version += 1
        return changed

    def add_interaction_immediate(self, edge: InteractionEdge) -> bool:
        """Add a single interaction and immediately report whether the
        graph topology changed."""
        old = self._graph.coupling_strength(edge.source_agent, edge.target_agent)
        self._graph.add_interaction(edge)
        new = self._graph.coupling_strength(edge.source_agent, edge.target_agent)
        if new > old + 1e-12:
            self._version += 1
            return True
        return False

    def recompute_strengths(
        self,
        trajectories: Dict[str, np.ndarray],
        observations: Optional[Dict[str, np.ndarray]] = None,
        hb_edges: Optional[Sequence[HBEdge]] = None,
        strength_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> bool:
        """Recompute coupling strengths from fresh trajectory data.

        Returns ``True`` if any strength changed by more than 1 %.
        """
        changed = False
        total = len(hb_edges) if hb_edges else 1
        metrics = InteractionStrengthMetrics()

        for u, v, data in list(self._graph.networkx_graph.edges(data=True)):
            mi = 0.0
            if u in trajectories and v in trajectories:
                mi = metrics.mutual_information(trajectories[u], trajectories[v])

            hb_density = 0.0
            if hb_edges is not None:
                hb_density = metrics.hb_edge_density(
                    [(e.source_agent, e.target_agent) for e in hb_edges],
                    u,
                    v,
                    total,
                )

            overlap = 0.0
            if observations and u in observations and v in observations:
                overlap = metrics.observation_overlap(observations[u], observations[v])

            new_strength = metrics.combined_strength(
                mi, hb_density, overlap, strength_weights
            )

            old_strength = data.get("weight", 0.0)
            if abs(new_strength - old_strength) > 0.01:
                changed = True

            # Update all interaction edges on this graph edge
            for ie in data.get("interactions", []):
                new_edge = InteractionEdge(
                    source_agent=ie.source_agent,
                    target_agent=ie.target_agent,
                    interaction_type=ie.interaction_type,
                    strength=new_strength,
                    metadata=ie.metadata,
                )
                self._graph.add_interaction(new_edge)

        if changed:
            self._version += 1
        return changed
