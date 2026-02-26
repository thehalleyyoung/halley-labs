"""Agent partitioning algorithms for compositional decomposition.

Partitions the interaction graph into interaction groups that minimise
cross-group coupling while respecting tractability constraints.  Provides
spectral, min-cut, hierarchical, and constrained partitioners, along
with quality metrics and iterative refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np

from marace.decomposition.interaction_graph import InteractionGraph


# ---------------------------------------------------------------------------
# Partition representation
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    """A partitioning of agents into groups.

    Attributes:
        groups: Mapping from group ID to the set of agent IDs.
        metadata: Optional metadata (e.g., algorithm used, quality score).
    """

    groups: Dict[str, FrozenSet[str]]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def all_agents(self) -> FrozenSet[str]:
        agents: Set[str] = set()
        for g in self.groups.values():
            agents.update(g)
        return frozenset(agents)

    @property
    def max_group_size(self) -> int:
        if not self.groups:
            return 0
        return max(len(g) for g in self.groups.values())

    def group_of(self, agent_id: str) -> Optional[str]:
        """Return the group ID containing *agent_id*, or ``None``."""
        for gid, agents in self.groups.items():
            if agent_id in agents:
                return gid
        return None

    def is_valid(self, expected_agents: FrozenSet[str]) -> bool:
        """Check that the partition covers exactly *expected_agents*
        with no overlaps."""
        seen: Set[str] = set()
        for g in self.groups.values():
            if seen & g:
                return False
            seen.update(g)
        return seen == expected_agents


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

class PartitionQualityMetrics:
    """Measure quality of a partition with respect to the interaction graph."""

    @staticmethod
    def cross_group_coupling(graph: InteractionGraph, partition: Partition) -> float:
        """Total weight of edges crossing group boundaries.

        Lower is better.
        """
        total = 0.0
        for edge in graph.iter_edges():
            g_src = partition.group_of(edge.source_agent)
            g_tgt = partition.group_of(edge.target_agent)
            if g_src is not None and g_tgt is not None and g_src != g_tgt:
                total += edge.strength
        return total

    @staticmethod
    def intra_group_coupling(graph: InteractionGraph, partition: Partition) -> float:
        """Total weight of edges within groups."""
        total = 0.0
        for edge in graph.iter_edges():
            g_src = partition.group_of(edge.source_agent)
            g_tgt = partition.group_of(edge.target_agent)
            if g_src is not None and g_tgt is not None and g_src == g_tgt:
                total += edge.strength
        return total

    @staticmethod
    def balance(partition: Partition) -> float:
        """Balance score in [0, 1].  1 = perfectly balanced.

        Computed as ``min_size / max_size`` across groups.
        """
        if partition.num_groups <= 1:
            return 1.0
        sizes = [len(g) for g in partition.groups.values()]
        return min(sizes) / max(sizes) if max(sizes) > 0 else 1.0

    @staticmethod
    def normalised_cut(graph: InteractionGraph, partition: Partition) -> float:
        """Normalised cut: ``cross / total``.

        Values near 0 indicate a good partition.
        """
        total = graph.total_coupling()
        if total < 1e-12:
            return 0.0
        cross = PartitionQualityMetrics.cross_group_coupling(graph, partition)
        return cross / total

    @staticmethod
    def quality_score(
        graph: InteractionGraph,
        partition: Partition,
        balance_weight: float = 0.3,
    ) -> float:
        """Composite quality score in [0, 1].  Higher is better.

        Combines normalised cut (inverted) and balance.
        """
        ncut = PartitionQualityMetrics.normalised_cut(graph, partition)
        bal = PartitionQualityMetrics.balance(partition)
        return (1.0 - ncut) * (1.0 - balance_weight) + bal * balance_weight


# ---------------------------------------------------------------------------
# Spectral partitioner
# ---------------------------------------------------------------------------

class SpectralPartitioner:
    """Partition the interaction graph using spectral clustering.

    Computes the normalised graph Laplacian, extracts the Fiedler vector
    (second-smallest eigenvector), and bisects agents at the median.
    Recursive bisection is used for ``k > 2`` groups.
    """

    def __init__(self, num_groups: int = 2, random_state: int = 42) -> None:
        self._k = num_groups
        self._rng = np.random.RandomState(random_state)

    def partition(self, graph: InteractionGraph) -> Partition:
        """Partition *graph* into ``num_groups`` groups."""
        agents = sorted(graph.agents)
        if len(agents) <= self._k:
            groups = {f"g{i}": frozenset([a]) for i, a in enumerate(agents)}
            return Partition(groups, {"algorithm": "spectral"})

        groups = self._recursive_bisect(graph, agents, self._k)
        named = {f"g{i}": frozenset(g) for i, g in enumerate(groups)}
        return Partition(named, {"algorithm": "spectral"})

    def _recursive_bisect(
        self, graph: InteractionGraph, agents: List[str], k: int
    ) -> List[List[str]]:
        if k <= 1 or len(agents) <= 1:
            return [agents]

        left, right = self._spectral_bisect(graph, agents)
        if k == 2:
            return [left, right]

        k_left = max(1, k // 2)
        k_right = k - k_left
        result = self._recursive_bisect(graph, left, k_left)
        result.extend(self._recursive_bisect(graph, right, k_right))
        return result

    def _spectral_bisect(
        self, graph: InteractionGraph, agents: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Bisect *agents* using the Fiedler vector."""
        n = len(agents)
        if n <= 1:
            return agents, []

        L = graph.laplacian_matrix(agents)
        # Add small regularisation for numerical stability
        L += 1e-10 * np.eye(n)

        # Degree matrix for normalised Laplacian
        D = graph.degree_matrix(agents)
        d_inv_sqrt = np.zeros_like(D)
        for i in range(n):
            if D[i, i] > 1e-12:
                d_inv_sqrt[i, i] = 1.0 / math.sqrt(D[i, i])
            else:
                d_inv_sqrt[i, i] = 1.0

        L_norm = d_inv_sqrt @ L @ d_inv_sqrt
        eigvals, eigvecs = np.linalg.eigh(L_norm)

        # Fiedler vector = eigenvector of second-smallest eigenvalue
        fiedler = eigvecs[:, 1]
        median = float(np.median(fiedler))

        left = [agents[i] for i in range(n) if fiedler[i] <= median]
        right = [agents[i] for i in range(n) if fiedler[i] > median]

        # Ensure non-empty partitions
        if not left:
            left.append(right.pop(0))
        elif not right:
            right.append(left.pop())

        return left, right


# ---------------------------------------------------------------------------
# Min-cut partitioner
# ---------------------------------------------------------------------------

class MinCutPartitioner:
    """Partition the interaction graph to minimise cross-group interactions.

    Uses networkx's minimum cut algorithms on the undirected weighted
    interaction graph.
    """

    def __init__(self, num_groups: int = 2) -> None:
        self._k = num_groups

    def partition(self, graph: InteractionGraph) -> Partition:
        """Partition *graph* into ``num_groups`` groups."""
        agents = sorted(graph.agents)
        if len(agents) <= self._k:
            groups = {f"g{i}": frozenset([a]) for i, a in enumerate(agents)}
            return Partition(groups, {"algorithm": "min_cut"})

        groups = self._recursive_min_cut(graph, agents, self._k)
        named = {f"g{i}": frozenset(g) for i, g in enumerate(groups)}
        return Partition(named, {"algorithm": "min_cut"})

    def _recursive_min_cut(
        self, graph: InteractionGraph, agents: List[str], k: int
    ) -> List[List[str]]:
        if k <= 1 or len(agents) <= 1:
            return [agents]

        left, right = self._min_cut_bisect(graph, agents)
        if k == 2:
            return [left, right]

        k_left = max(1, k // 2)
        k_right = k - k_left
        result = self._recursive_min_cut(graph, left, k_left)
        result.extend(self._recursive_min_cut(graph, right, k_right))
        return result

    def _min_cut_bisect(
        self, graph: InteractionGraph, agents: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Bisect *agents* using minimum s-t cut.

        Tries all (s, t) pairs among high-degree and low-degree nodes
        to find the best cut.
        """
        if len(agents) <= 1:
            return agents, []

        sub = graph.subgraph(set(agents))
        ug = sub.networkx_graph.to_undirected()

        # Ensure all edges have capacity
        for u, v, data in ug.edges(data=True):
            if "capacity" not in data:
                data["capacity"] = data.get("weight", 1.0)

        # If graph is disconnected, use connected components
        components = list(nx.connected_components(ug))
        if len(components) >= 2:
            left_comp = list(components[0])
            right_comp = [a for comp in components[1:] for a in comp]
            return left_comp, right_comp

        # Try min-cut between degree extremes
        degrees = sorted(ug.degree(), key=lambda x: x[1])
        candidates_s = [degrees[0][0]]
        candidates_t = [degrees[-1][0]]

        best_cut_val = float("inf")
        best_partition: Tuple[List[str], List[str]] = (agents[:1], agents[1:])

        for s in candidates_s:
            for t in candidates_t:
                if s == t:
                    continue
                try:
                    cut_val, (reachable, non_reachable) = nx.minimum_cut(
                        ug, s, t, capacity="capacity"
                    )
                    if cut_val < best_cut_val and reachable and non_reachable:
                        best_cut_val = cut_val
                        best_partition = (list(reachable), list(non_reachable))
                except nx.NetworkXError:
                    continue

        return best_partition


# ---------------------------------------------------------------------------
# Hierarchical partitioner
# ---------------------------------------------------------------------------

@dataclass
class HierarchyNode:
    """Node in a hierarchical partition tree."""
    group_id: str
    agents: FrozenSet[str]
    children: List["HierarchyNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class HierarchicalPartitioner:
    """Build a hierarchy of interaction groups for multi-level analysis.

    Uses agglomerative (bottom-up) merging: start with each agent as its
    own group, then iteratively merge the pair with the strongest coupling
    until the desired number of top-level groups remains.
    """

    def __init__(
        self,
        min_groups: int = 2,
        max_group_size: int = 6,
    ) -> None:
        self._min_groups = min_groups
        self._max_group_size = max_group_size

    def partition(self, graph: InteractionGraph) -> Partition:
        """Flat partition at the top level of the hierarchy."""
        root = self.build_hierarchy(graph)
        groups: Dict[str, FrozenSet[str]] = {}
        self._collect_leaves(root, groups)
        return Partition(groups, {"algorithm": "hierarchical"})

    def build_hierarchy(self, graph: InteractionGraph) -> HierarchyNode:
        """Build the full hierarchical tree."""
        agents = sorted(graph.agents)
        # Start with singleton groups
        nodes: Dict[str, HierarchyNode] = {
            a: HierarchyNode(group_id=a, agents=frozenset([a]))
            for a in agents
        }
        active: List[str] = list(agents)
        merge_id = 0

        while len(active) > self._min_groups:
            # Find the most strongly coupled pair
            best_pair: Optional[Tuple[str, str]] = None
            best_strength = -1.0

            for i, ga in enumerate(active):
                for gb in active[i + 1:]:
                    strength = self._inter_group_coupling(
                        graph, nodes[ga].agents, nodes[gb].agents
                    )
                    combined = len(nodes[ga].agents) + len(nodes[gb].agents)
                    if strength > best_strength and combined <= self._max_group_size:
                        best_strength = strength
                        best_pair = (ga, gb)

            if best_pair is None:
                break

            ga, gb = best_pair
            merged_id = f"m{merge_id}"
            merge_id += 1
            merged = HierarchyNode(
                group_id=merged_id,
                agents=nodes[ga].agents | nodes[gb].agents,
                children=[nodes[ga], nodes[gb]],
            )
            nodes[merged_id] = merged
            active.remove(ga)
            active.remove(gb)
            active.append(merged_id)

        root_children = [nodes[a] for a in active]
        root = HierarchyNode(
            group_id="root",
            agents=frozenset().union(*(n.agents for n in root_children)),
            children=root_children,
        )
        return root

    @staticmethod
    def _inter_group_coupling(
        graph: InteractionGraph,
        agents_a: FrozenSet[str],
        agents_b: FrozenSet[str],
    ) -> float:
        total = 0.0
        for a in agents_a:
            for b in agents_b:
                total += graph.coupling_strength(a, b)
                total += graph.coupling_strength(b, a)
        return total

    def _collect_leaves(
        self, node: HierarchyNode, out: Dict[str, FrozenSet[str]]
    ) -> None:
        if node.is_leaf or not node.children:
            out[node.group_id] = node.agents
        else:
            for child in node.children:
                self._collect_leaves(child, out)


# ---------------------------------------------------------------------------
# Constrained partitioner
# ---------------------------------------------------------------------------

class ConstrainedPartitioner:
    """Partition with size constraints for tractability.

    Wraps a base partitioner and enforces that no group exceeds
    ``max_group_size``.  Oversized groups are recursively split.
    """

    def __init__(
        self,
        max_group_size: int = 4,
        base_partitioner: Optional[object] = None,
    ) -> None:
        self._max_size = max_group_size
        self._base = base_partitioner or SpectralPartitioner(num_groups=2)

    def partition(self, graph: InteractionGraph) -> Partition:
        """Partition ensuring all groups have at most ``max_group_size`` agents."""
        # Start from connected components
        components = graph.connected_components()
        groups: Dict[str, FrozenSet[str]] = {}
        gid = 0

        for comp in components:
            if len(comp) <= self._max_size:
                groups[f"g{gid}"] = comp
                gid += 1
            else:
                sub_groups = self._split(graph, sorted(comp))
                for sg in sub_groups:
                    groups[f"g{gid}"] = frozenset(sg)
                    gid += 1

        return Partition(groups, {"algorithm": "constrained"})

    def _split(self, graph: InteractionGraph, agents: List[str]) -> List[List[str]]:
        """Recursively split until all groups are within size limit."""
        if len(agents) <= self._max_size:
            return [agents]

        k = max(2, math.ceil(len(agents) / self._max_size))
        partitioner = SpectralPartitioner(num_groups=k)
        sub = graph.subgraph(set(agents))
        p = partitioner.partition(sub)

        result: List[List[str]] = []
        for group_agents in p.groups.values():
            if len(group_agents) <= self._max_size:
                result.append(sorted(group_agents))
            else:
                result.extend(self._split(graph, sorted(group_agents)))
        return result


# ---------------------------------------------------------------------------
# Partition refinement
# ---------------------------------------------------------------------------

class PartitionRefinement:
    """Iteratively improve a partition based on analysis results.

    Uses Kernighan-Lin–style swaps: for each agent, compute the gain
    of moving it to a neighbouring group and apply the best swap.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        max_group_size: int = 6,
        min_improvement: float = 1e-4,
    ) -> None:
        self._max_iter = max_iterations
        self._max_size = max_group_size
        self._min_improvement = min_improvement

    def refine(
        self, graph: InteractionGraph, partition: Partition
    ) -> Partition:
        """Iteratively refine *partition* to improve quality.

        Returns the best partition found during the refinement process.
        """
        best_partition = partition
        best_score = PartitionQualityMetrics.quality_score(graph, partition)
        current = partition

        for _ in range(self._max_iter):
            improved = self._kl_pass(graph, current)
            score = PartitionQualityMetrics.quality_score(graph, improved)

            if score > best_score + self._min_improvement:
                best_score = score
                best_partition = improved
                current = improved
            else:
                break

        return best_partition

    def _kl_pass(
        self, graph: InteractionGraph, partition: Partition
    ) -> Partition:
        """Single Kernighan-Lin refinement pass."""
        groups = {gid: set(agents) for gid, agents in partition.groups.items()}
        agents = sorted(partition.all_agents)
        improved = False

        for agent in agents:
            current_group = None
            for gid, members in groups.items():
                if agent in members:
                    current_group = gid
                    break
            if current_group is None:
                continue

            # Compute gain of moving agent to each neighbouring group
            best_gain = 0.0
            best_target: Optional[str] = None

            # Internal coupling (edges to agents in same group)
            internal = sum(
                graph.coupling_strength(agent, other) +
                graph.coupling_strength(other, agent)
                for other in groups[current_group]
                if other != agent
            )

            for target_gid, target_members in groups.items():
                if target_gid == current_group:
                    continue
                if len(target_members) >= self._max_size:
                    continue

                # External coupling (edges to agents in target group)
                external = sum(
                    graph.coupling_strength(agent, other) +
                    graph.coupling_strength(other, agent)
                    for other in target_members
                )

                gain = external - internal
                if gain > best_gain:
                    best_gain = gain
                    best_target = target_gid

            if best_target is not None and best_gain > self._min_improvement:
                groups[current_group].discard(agent)
                groups[best_target].add(agent)
                improved = True

        # Remove empty groups
        groups = {gid: frozenset(m) for gid, m in groups.items() if m}

        if not improved:
            return partition

        return Partition(dict(groups), {"algorithm": "refined"})
