"""
Interaction group management for the MARACE happens-before engine.

Provides data structures and algorithms for grouping agents that interact
(directly or transitively), estimating interaction strength, merging /
partitioning groups, and tracking group evolution over time.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)


# ======================================================================
# Core data structure
# ======================================================================

@dataclass(frozen=True)
class InteractionGroup:
    """An immutable group of agents that interact within an HB graph.

    Attributes:
        agent_ids: Frozen set of agent identifiers in this group.
        shared_state_dims: State dimensions shared among the agents.
        interaction_strength: Scalar in [0, 1] quantifying how strongly
            the agents interact (higher = denser HB edges).
        event_ids: Frozen set of event IDs belonging to this group
            (optional; populated when extracted from an HB graph).
        metadata: Arbitrary extra information.
    """
    agent_ids: FrozenSet[str]
    shared_state_dims: FrozenSet[str] = frozenset()
    interaction_strength: float = 0.0
    event_ids: FrozenSet[str] = frozenset()
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def size(self) -> int:
        """Number of agents in the group."""
        return len(self.agent_ids)

    @property
    def num_events(self) -> int:
        """Number of events in the group."""
        return len(self.event_ids)

    def contains_agent(self, agent_id: str) -> bool:
        return agent_id in self.agent_ids

    def overlaps(self, other: "InteractionGroup") -> bool:
        """True if the two groups share at least one agent."""
        return bool(self.agent_ids & other.agent_ids)

    def merged_with(self, other: "InteractionGroup") -> "InteractionGroup":
        """Return a new group that is the union of *self* and *other*."""
        combined_agents = self.agent_ids | other.agent_ids
        combined_dims = self.shared_state_dims | other.shared_state_dims
        combined_events = self.event_ids | other.event_ids
        avg_strength = (
            (self.interaction_strength * self.size
             + other.interaction_strength * other.size)
            / max(len(combined_agents), 1)
        )
        return InteractionGroup(
            agent_ids=combined_agents,
            shared_state_dims=combined_dims,
            interaction_strength=avg_strength,
            event_ids=combined_events,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_ids": sorted(self.agent_ids),
            "shared_state_dims": sorted(self.shared_state_dims),
            "interaction_strength": self.interaction_strength,
            "event_ids": sorted(self.event_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionGroup":
        return cls(
            agent_ids=frozenset(data.get("agent_ids", [])),
            shared_state_dims=frozenset(data.get("shared_state_dims", [])),
            interaction_strength=data.get("interaction_strength", 0.0),
            event_ids=frozenset(data.get("event_ids", [])),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        agents = ", ".join(sorted(self.agent_ids))
        return (
            f"InteractionGroup(agents=[{agents}], "
            f"strength={self.interaction_strength:.3f}, "
            f"events={self.num_events})"
        )


# ======================================================================
# InteractionGroupExtractor
# ======================================================================

class InteractionGroupExtractor:
    """Extract interaction groups from HB graph connected components.

    This is a thin wrapper that delegates to the HB graph's own component
    extraction, augmented with agent-level metadata.

    Args:
        min_group_size: Minimum number of agents for a group to be kept.
    """

    def __init__(self, min_group_size: int = 1) -> None:
        self._min_size = min_group_size

    def extract(self, hb_graph: Any) -> List[InteractionGroup]:
        """Extract groups from an HBGraph.

        Args:
            hb_graph: An :class:`~marace.hb.hb_graph.HBGraph` instance.

        Returns:
            List of :class:`InteractionGroup` instances whose agent count
            meets the minimum size.
        """
        groups = hb_graph.extract_interaction_groups()
        return [g for g in groups if g.size >= self._min_size]

    def extract_with_stats(
        self, hb_graph: Any,
    ) -> List[Dict[str, Any]]:
        """Extract groups and augment with per-group statistics."""
        groups = self.extract(hb_graph)
        result: List[Dict[str, Any]] = []
        for g in groups:
            info = g.to_dict()
            info["num_agents"] = g.size
            info["num_events"] = g.num_events
            result.append(info)
        return result


# ======================================================================
# GroupMerger
# ======================================================================

class GroupMerger:
    """Merge interaction groups when new interactions are discovered.

    Maintains a list of groups and merges overlapping ones whenever new
    evidence (e.g. a new HB edge crossing group boundaries) appears.
    """

    def __init__(self) -> None:
        self._groups: List[InteractionGroup] = []

    @property
    def groups(self) -> List[InteractionGroup]:
        return list(self._groups)

    def add_group(self, group: InteractionGroup) -> List[InteractionGroup]:
        """Add a group, merging it with any existing overlapping groups.

        Returns:
            The updated list of all groups after merging.
        """
        merged = group
        remaining: List[InteractionGroup] = []
        for existing in self._groups:
            if merged.overlaps(existing):
                merged = merged.merged_with(existing)
            else:
                remaining.append(existing)
        remaining.append(merged)
        self._groups = remaining
        return list(self._groups)

    def add_groups(
        self, groups: Iterable[InteractionGroup],
    ) -> List[InteractionGroup]:
        """Add multiple groups, performing iterative merges."""
        for g in groups:
            self.add_group(g)
        return list(self._groups)

    def merge_by_edge(
        self, agent_a: str, agent_b: str,
    ) -> List[InteractionGroup]:
        """Merge the groups containing *agent_a* and *agent_b*.

        If either agent is not in any group, a singleton group is created.

        Returns:
            Updated group list.
        """
        group_a: Optional[InteractionGroup] = None
        group_b: Optional[InteractionGroup] = None
        remaining: List[InteractionGroup] = []

        for g in self._groups:
            if g.contains_agent(agent_a):
                group_a = g
            elif g.contains_agent(agent_b):
                group_b = g
            else:
                remaining.append(g)

        if group_a is None:
            group_a = InteractionGroup(agent_ids=frozenset({agent_a}))
        if group_b is None:
            group_b = InteractionGroup(agent_ids=frozenset({agent_b}))

        if group_a is group_b or group_a.agent_ids == group_b.agent_ids:
            remaining.append(group_a)
        else:
            remaining.append(group_a.merged_with(group_b))

        self._groups = remaining
        return list(self._groups)

    def reset(self) -> None:
        self._groups.clear()


# ======================================================================
# GroupPartitioner
# ======================================================================

class GroupPartitioner:
    """Partition agents into non-overlapping groups.

    Given a set of (possibly overlapping) interaction groups, produces a
    partition where each agent belongs to exactly one group.  Overlapping
    agents are assigned to the group with the highest interaction strength.
    """

    @staticmethod
    def partition(
        groups: List[InteractionGroup],
        all_agents: Optional[Set[str]] = None,
    ) -> List[InteractionGroup]:
        """Partition agents into non-overlapping groups.

        Args:
            groups: Input groups (may overlap).
            all_agents: If provided, agents not in any group get their own
                singleton group.

        Returns:
            Non-overlapping groups covering all agents.
        """
        # Sort by interaction strength descending so stronger groups
        # have priority for claiming shared agents.
        sorted_groups = sorted(
            groups, key=lambda g: g.interaction_strength, reverse=True
        )
        assigned: Set[str] = set()
        partitions: List[InteractionGroup] = []

        for g in sorted_groups:
            unassigned = g.agent_ids - assigned
            if not unassigned:
                continue
            partitions.append(InteractionGroup(
                agent_ids=frozenset(unassigned),
                shared_state_dims=g.shared_state_dims,
                interaction_strength=g.interaction_strength,
                event_ids=g.event_ids,
                metadata=g.metadata,
            ))
            assigned.update(unassigned)

        # Create singletons for uncovered agents
        if all_agents is not None:
            for a in sorted(all_agents - assigned):
                partitions.append(InteractionGroup(
                    agent_ids=frozenset({a}),
                    interaction_strength=0.0,
                ))
                assigned.add(a)

        return partitions

    @staticmethod
    def is_partition(groups: List[InteractionGroup]) -> bool:
        """Check that groups form a valid partition (no overlap)."""
        seen: Set[str] = set()
        for g in groups:
            if g.agent_ids & seen:
                return False
            seen.update(g.agent_ids)
        return True


# ======================================================================
# InteractionStrengthEstimator
# ======================================================================

class InteractionStrengthEstimator:
    """Quantify how strongly agents interact based on HB edge density.

    Interaction strength for a group of agents is defined as the number
    of cross-agent HB edges divided by the maximum possible cross-agent
    edges, optionally weighted by edge confidence.

    Args:
        weighted: If True, sum edge confidences instead of counting edges.
    """

    def __init__(self, weighted: bool = False) -> None:
        self._weighted = weighted

    def estimate(
        self,
        agent_ids: FrozenSet[str],
        hb_graph: Any,
    ) -> float:
        """Estimate interaction strength for *agent_ids* within *hb_graph*.

        Args:
            agent_ids: The set of agents to consider.
            hb_graph: An HBGraph instance.

        Returns:
            Strength in [0, 1].
        """
        if len(agent_ids) < 2:
            return 0.0

        g = hb_graph.graph
        cross_count = 0.0

        for u, v, data in g.edges(data=True):
            u_agent = g.nodes[u].get("agent_id")
            v_agent = g.nodes[v].get("agent_id")
            if u_agent in agent_ids and v_agent in agent_ids and u_agent != v_agent:
                if self._weighted:
                    cross_count += data.get("confidence", 1.0)
                else:
                    cross_count += 1.0

        n = len(agent_ids)
        # Count events per agent to compute max possible edges
        agent_events: Dict[str, int] = defaultdict(int)
        for node, data in g.nodes(data=True):
            aid = data.get("agent_id")
            if aid in agent_ids:
                agent_events[aid] += 1

        max_cross = sum(
            agent_events[a] * agent_events[b]
            for i, a in enumerate(sorted(agent_ids))
            for b in sorted(agent_ids)[i + 1:]
        )
        if max_cross == 0:
            return 0.0
        return min(1.0, cross_count / max_cross)

    def pairwise_strength(
        self,
        agent_ids: FrozenSet[str],
        hb_graph: Any,
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise interaction strengths between all agent pairs.

        Returns:
            Dict mapping (agent_a, agent_b) to strength in [0, 1].
        """
        g = hb_graph.graph
        pair_count: Dict[Tuple[str, str], float] = defaultdict(float)
        agent_events: Dict[str, int] = defaultdict(int)

        for node, data in g.nodes(data=True):
            aid = data.get("agent_id")
            if aid in agent_ids:
                agent_events[aid] += 1

        for u, v, data in g.edges(data=True):
            u_agent = g.nodes[u].get("agent_id")
            v_agent = g.nodes[v].get("agent_id")
            if (
                u_agent in agent_ids
                and v_agent in agent_ids
                and u_agent != v_agent
            ):
                key = (min(u_agent, v_agent), max(u_agent, v_agent))
                if self._weighted:
                    pair_count[key] += data.get("confidence", 1.0)
                else:
                    pair_count[key] += 1.0

        result: Dict[Tuple[str, str], float] = {}
        agents_sorted = sorted(agent_ids)
        for i, a in enumerate(agents_sorted):
            for b in agents_sorted[i + 1:]:
                key = (a, b)
                max_edges = agent_events[a] * agent_events[b]
                if max_edges == 0:
                    result[key] = 0.0
                else:
                    result[key] = min(1.0, pair_count.get(key, 0.0) / max_edges)
        return result


# ======================================================================
# GroupEvolution
# ======================================================================

@dataclass
class GroupSnapshot:
    """A snapshot of interaction groups at a specific time.

    Attributes:
        timestep: When this snapshot was taken.
        groups: The groups at this timestep.
    """
    timestep: int
    groups: List[InteractionGroup]

    def agent_to_group(self) -> Dict[str, int]:
        """Map each agent to its group index."""
        mapping: Dict[str, int] = {}
        for idx, g in enumerate(self.groups):
            for a in g.agent_ids:
                mapping[a] = idx
        return mapping


class GroupEvolution:
    """Track how interaction groups change over time in a trace.

    Records snapshots at each timestep and provides analysis of group
    stability, merges, and splits.
    """

    def __init__(self) -> None:
        self._snapshots: List[GroupSnapshot] = []

    def record(self, timestep: int, groups: List[InteractionGroup]) -> None:
        """Record a group snapshot at *timestep*."""
        self._snapshots.append(GroupSnapshot(
            timestep=timestep,
            groups=list(groups),
        ))

    @property
    def snapshots(self) -> List[GroupSnapshot]:
        return list(self._snapshots)

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    def group_count_over_time(self) -> List[Tuple[int, int]]:
        """Return (timestep, num_groups) pairs."""
        return [(s.timestep, len(s.groups)) for s in self._snapshots]

    def agent_group_trajectory(self, agent_id: str) -> List[Tuple[int, int]]:
        """Return (timestep, group_index) for a specific agent.

        Returns (-1) if the agent is not present in a snapshot.
        """
        trajectory: List[Tuple[int, int]] = []
        for snap in self._snapshots:
            mapping = snap.agent_to_group()
            gidx = mapping.get(agent_id, -1)
            trajectory.append((snap.timestep, gidx))
        return trajectory

    def detect_merges(self) -> List[Dict[str, Any]]:
        """Detect group merge events between consecutive snapshots.

        A merge is detected when two or more groups from snapshot t
        map to a single group in snapshot t+1.

        Returns:
            List of merge event dicts with keys ``timestep``,
            ``merged_groups``, ``result_group``.
        """
        merges: List[Dict[str, Any]] = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            prev_map = prev.agent_to_group()
            curr_map = curr.agent_to_group()

            # For each current group, find which previous groups contribute
            for g_idx, g in enumerate(curr.groups):
                source_groups: Set[int] = set()
                for a in g.agent_ids:
                    if a in prev_map:
                        source_groups.add(prev_map[a])
                if len(source_groups) > 1:
                    merges.append({
                        "timestep": curr.timestep,
                        "merged_groups": sorted(source_groups),
                        "result_group": g_idx,
                        "agents": sorted(g.agent_ids),
                    })
        return merges

    def detect_splits(self) -> List[Dict[str, Any]]:
        """Detect group split events between consecutive snapshots.

        A split occurs when agents from one group in snapshot t end up
        in different groups in snapshot t+1.

        Returns:
            List of split event dicts.
        """
        splits: List[Dict[str, Any]] = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            prev_map = prev.agent_to_group()
            curr_map = curr.agent_to_group()

            for g_idx, g in enumerate(prev.groups):
                dest_groups: Set[int] = set()
                for a in g.agent_ids:
                    if a in curr_map:
                        dest_groups.add(curr_map[a])
                if len(dest_groups) > 1:
                    splits.append({
                        "timestep": curr.timestep,
                        "source_group": g_idx,
                        "split_into": sorted(dest_groups),
                        "agents": sorted(g.agent_ids),
                    })
        return splits

    def stability_score(self) -> float:
        """Fraction of consecutive snapshot pairs with identical groupings.

        Returns:
            Value in [0, 1] where 1 means groups never changed.
        """
        if len(self._snapshots) < 2:
            return 1.0
        stable = 0
        for i in range(1, len(self._snapshots)):
            prev_sets = {g.agent_ids for g in self._snapshots[i - 1].groups}
            curr_sets = {g.agent_ids for g in self._snapshots[i].groups}
            if prev_sets == curr_sets:
                stable += 1
        return stable / (len(self._snapshots) - 1)

    def summary(self) -> Dict[str, Any]:
        """High-level summary of group evolution."""
        return {
            "num_snapshots": self.num_snapshots,
            "group_count_trajectory": self.group_count_over_time(),
            "num_merges": len(self.detect_merges()),
            "num_splits": len(self.detect_splits()),
            "stability_score": round(self.stability_score(), 4),
        }

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all snapshots."""
        return [
            {
                "timestep": s.timestep,
                "groups": [g.to_dict() for g in s.groups],
            }
            for s in self._snapshots
        ]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> "GroupEvolution":
        """Reconstruct from serialized form."""
        evo = cls()
        for snap_data in data:
            groups = [InteractionGroup.from_dict(gd) for gd in snap_data["groups"]]
            evo.record(snap_data["timestep"], groups)
        return evo

    def __repr__(self) -> str:
        return f"GroupEvolution(snapshots={self.num_snapshots})"
