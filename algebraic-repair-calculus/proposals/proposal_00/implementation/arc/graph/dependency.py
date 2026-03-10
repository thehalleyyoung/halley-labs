"""
Dependency Analysis for Pipeline Graphs
========================================

Comprehensive dependency analysis utilities for understanding how
perturbations propagate through the pipeline DAG. Provides impact
sets, dependency sets, minimal repair sets, repair waves, checkpoint
candidates, column-level impact, transitive closures, dominance
frontiers, and loop nesting analysis.

These analyses are critical for:
  - Determining the scope of repairs needed after a perturbation.
  - Finding the minimal set of nodes to repair.
  - Scheduling repairs in waves for parallel execution.
  - Identifying checkpoint candidates for incremental recovery.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes for analysis results
# =====================================================================


class ImpactSeverity(Enum):
    """Severity of impact on a downstream node."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NodeImpact:
    """Impact information for a single node.

    Attributes
    ----------
    node_id : str
        The affected node.
    distance : int
        Topological distance from the perturbation source.
    path_count : int
        Number of distinct paths from source to this node.
    severity : ImpactSeverity
        Estimated severity of the impact.
    affected_columns : set[str]
        Set of columns affected at this node.
    cost_to_repair : float
        Estimated cost to repair this node.
    """
    node_id: str = ""
    distance: int = 0
    path_count: int = 1
    severity: ImpactSeverity = ImpactSeverity.MEDIUM
    affected_columns: Set[str] = field(default_factory=set)
    cost_to_repair: float = 0.0

    def __repr__(self) -> str:
        return (
            f"NodeImpact({self.node_id}, dist={self.distance}, "
            f"severity={self.severity.value}, cost={self.cost_to_repair:.2f})"
        )


@dataclass
class ImpactSet:
    """Complete impact analysis result for a perturbation.

    Attributes
    ----------
    source_node : str
        The node where the perturbation originates.
    impacts : dict[str, NodeImpact]
        Mapping from node_id to its impact information.
    total_repair_cost : float
        Sum of all repair costs.
    max_distance : int
        Maximum topological distance from source.
    critical_path : list[str]
        The most expensive path through impacted nodes.
    """
    source_node: str = ""
    impacts: Dict[str, NodeImpact] = field(default_factory=dict)
    total_repair_cost: float = 0.0
    max_distance: int = 0
    critical_path: List[str] = field(default_factory=list)

    @property
    def affected_nodes(self) -> Set[str]:
        return set(self.impacts.keys())

    @property
    def node_count(self) -> int:
        return len(self.impacts)

    def nodes_at_distance(self, d: int) -> Set[str]:
        """Return nodes exactly at distance d from source."""
        return {
            nid for nid, imp in self.impacts.items() if imp.distance == d
        }

    def nodes_by_severity(self, severity: ImpactSeverity) -> Set[str]:
        """Return nodes with a given severity level."""
        return {
            nid for nid, imp in self.impacts.items() if imp.severity == severity
        }

    def summary(self) -> str:
        lines = [
            f"ImpactSet(source={self.source_node}):",
            f"  Nodes affected: {self.node_count}",
            f"  Max distance:   {self.max_distance}",
            f"  Total cost:     {self.total_repair_cost:.2f}",
            f"  Critical path:  {' -> '.join(self.critical_path)}",
        ]
        return "\n".join(lines)


@dataclass
class DependencySet:
    """Upstream dependency analysis result.

    Attributes
    ----------
    target_node : str
        The node whose dependencies were analyzed.
    dependencies : dict[str, int]
        Mapping from dependency node_id to distance.
    transitive_deps : set[str]
        Full transitive dependency set.
    immediate_deps : set[str]
        Direct (1-hop) dependencies.
    """
    target_node: str = ""
    dependencies: Dict[str, int] = field(default_factory=dict)
    transitive_deps: Set[str] = field(default_factory=set)
    immediate_deps: Set[str] = field(default_factory=set)

    @property
    def depth(self) -> int:
        if not self.dependencies:
            return 0
        return max(self.dependencies.values())


@dataclass
class RepairWave:
    """A group of nodes that can be repaired in parallel.

    Attributes
    ----------
    wave_index : int
        Position of this wave in the schedule (0-based).
    nodes : set[str]
        Nodes to repair in this wave.
    estimated_cost : float
        Estimated cost of this wave (max of individual costs).
    estimated_time : float
        Estimated wall-clock time for this wave.
    """
    wave_index: int = 0
    nodes: Set[str] = field(default_factory=set)
    estimated_cost: float = 0.0
    estimated_time: float = 0.0

    @property
    def size(self) -> int:
        return len(self.nodes)


@dataclass
class ColumnImpact:
    """Column-level impact tracking through the pipeline.

    Attributes
    ----------
    source_node : str
        Origin node.
    source_column : str
        Origin column.
    impacts : dict[str, set[str]]
        Mapping from node_id to set of affected column names.
    transformation_chain : dict[str, list[str]]
        Column name transformations along the path.
    """
    source_node: str = ""
    source_column: str = ""
    impacts: Dict[str, Set[str]] = field(default_factory=dict)
    transformation_chain: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_affected_columns(self) -> int:
        return sum(len(cols) for cols in self.impacts.values())

    def reaches_node(self, node_id: str) -> bool:
        return node_id in self.impacts


# =====================================================================
# Dependency Analyzer
# =====================================================================


class DependencyAnalyzer:
    """Comprehensive dependency analysis for pipeline graphs.

    Provides methods for computing impact sets, dependency sets, minimal
    repair sets, repair waves, checkpoint candidates, column-level impact,
    transitive closures, dominance frontiers, and loop nesting depth.

    Parameters
    ----------
    cost_weight : float
        Weight factor for cost in severity computation.
    distance_weight : float
        Weight factor for distance in severity computation.
    """

    def __init__(
        self,
        cost_weight: float = 1.0,
        distance_weight: float = 0.5,
    ) -> None:
        self._cost_weight = cost_weight
        self._distance_weight = distance_weight

    # ── Impact Analysis ───────────────────────────────────────────

    def compute_impact_set(
        self,
        graph: Any,
        node_id: str,
        delta: Any = None,
    ) -> ImpactSet:
        """Compute the downstream impact of a perturbation at node_id.

        Uses BFS to traverse all downstream nodes, computing distance,
        path count, severity, and estimated repair cost.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        node_id : str
            The perturbed node.
        delta : CompoundPerturbation, optional
            The perturbation (used for column-level analysis).

        Returns
        -------
        ImpactSet
        """
        impacts: Dict[str, NodeImpact] = {}
        distances: Dict[str, int] = {node_id: 0}
        path_counts: Dict[str, int] = defaultdict(int)
        path_counts[node_id] = 1

        topo = graph.topological_sort()
        node_index = {nid: i for i, nid in enumerate(topo)}

        if node_id not in node_index:
            return ImpactSet(source_node=node_id)

        for nid in topo[node_index[node_id] + 1:]:
            preds = graph.predecessors(nid)
            min_dist = None
            total_paths = 0

            for pred in preds:
                if pred in distances:
                    d = distances[pred] + 1
                    if min_dist is None or d < min_dist:
                        min_dist = d
                    total_paths += path_counts.get(pred, 0)

            if min_dist is not None:
                distances[nid] = min_dist
                path_counts[nid] = total_paths

                node = graph.get_node(nid)
                cost = self._estimate_node_cost(node)
                severity = self._compute_severity(min_dist, cost, total_paths)

                affected_cols = set()
                if delta is not None:
                    affected_cols = self._get_affected_columns(delta, node)

                impacts[nid] = NodeImpact(
                    node_id=nid,
                    distance=min_dist,
                    path_count=total_paths,
                    severity=severity,
                    affected_columns=affected_cols,
                    cost_to_repair=cost,
                )

        total_cost = sum(imp.cost_to_repair for imp in impacts.values())
        max_dist = max((imp.distance for imp in impacts.values()), default=0)
        critical = self._find_critical_path(graph, node_id, impacts)

        return ImpactSet(
            source_node=node_id,
            impacts=impacts,
            total_repair_cost=total_cost,
            max_distance=max_dist,
            critical_path=critical,
        )

    def compute_dependency_set(
        self,
        graph: Any,
        node_id: str,
    ) -> DependencySet:
        """Compute the upstream dependency set of a node.

        Uses reverse BFS to find all ancestors and their distances.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        node_id : str
            The node to analyze.

        Returns
        -------
        DependencySet
        """
        deps: Dict[str, int] = {}
        queue: deque[Tuple[str, int]] = deque()

        immediate = set(graph.predecessors(node_id))
        for pred in immediate:
            queue.append((pred, 1))
            deps[pred] = 1

        visited: Set[str] = set(immediate)

        while queue:
            current, dist = queue.popleft()
            for pred in graph.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    deps[pred] = dist + 1
                    queue.append((pred, dist + 1))

        return DependencySet(
            target_node=node_id,
            dependencies=deps,
            transitive_deps=set(deps.keys()),
            immediate_deps=immediate,
        )

    def find_minimal_repair_set(
        self,
        graph: Any,
        affected_nodes: Set[str],
    ) -> Set[str]:
        """Find the minimal set of nodes that must be repaired.

        A node can be skipped if all its downstream dependents are already
        in the repair set. Uses a greedy covering approach.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        affected_nodes : set[str]
            All nodes affected by the perturbation.

        Returns
        -------
        set[str]
            Minimal repair set.
        """
        if not affected_nodes:
            return set()

        topo = graph.topological_sort()
        affected_ordered = [n for n in topo if n in affected_nodes]

        repair_set: Set[str] = set()
        covered: Set[str] = set()

        for nid in affected_ordered:
            if nid in covered:
                continue

            preds = graph.predecessors(nid)
            all_preds_repaired = all(
                p in repair_set or p not in affected_nodes
                for p in preds
            )

            if not preds or not all_preds_repaired:
                repair_set.add(nid)
                covered.add(nid)

                desc = self._get_descendants_in_set(graph, nid, affected_nodes)
                covered.update(desc)
            else:
                repair_set.add(nid)
                covered.add(nid)

        return repair_set

    def compute_repair_waves(
        self,
        graph: Any,
        repair_set: Set[str],
    ) -> List[RepairWave]:
        """Organize repair nodes into parallel execution waves.

        Within each wave, all nodes are independent and can be repaired
        simultaneously. Dependencies between waves are respected.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        repair_set : set[str]
            The set of nodes to repair.

        Returns
        -------
        list[RepairWave]
            Ordered list of waves.
        """
        if not repair_set:
            return []

        topo = graph.topological_sort()
        repair_ordered = [n for n in topo if n in repair_set]

        wave_assignment: Dict[str, int] = {}

        for nid in repair_ordered:
            preds = graph.predecessors(nid)
            max_pred_wave = -1

            for pred in preds:
                if pred in wave_assignment:
                    max_pred_wave = max(max_pred_wave, wave_assignment[pred])

            wave_assignment[nid] = max_pred_wave + 1

        waves_dict: Dict[int, Set[str]] = defaultdict(set)
        for nid, wave_idx in wave_assignment.items():
            waves_dict[wave_idx].add(nid)

        waves: List[RepairWave] = []
        for idx in sorted(waves_dict.keys()):
            nodes = waves_dict[idx]
            max_cost = 0.0
            total_cost = 0.0

            for nid in nodes:
                node = graph.get_node(nid)
                cost = self._estimate_node_cost(node)
                max_cost = max(max_cost, cost)
                total_cost += cost

            waves.append(RepairWave(
                wave_index=idx,
                nodes=nodes,
                estimated_cost=total_cost,
                estimated_time=max_cost,
            ))

        return waves

    def find_checkpoint_candidates(
        self,
        graph: Any,
        repair_set: Set[str],
    ) -> Set[str]:
        """Identify nodes that are good candidates for checkpointing.

        Checkpoint candidates are nodes where materializing intermediate
        results would most reduce re-computation on failure. Criteria:
        - High fan-out (many dependents in repair set)
        - High cost (expensive to recompute)
        - Wave boundaries (between repair waves)

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        repair_set : set[str]
            The repair set.

        Returns
        -------
        set[str]
            Set of checkpoint candidate node ids.
        """
        if not repair_set:
            return set()

        scores: Dict[str, float] = {}

        for nid in repair_set:
            score = 0.0

            succs = graph.successors(nid)
            repair_succs = [s for s in succs if s in repair_set]
            score += len(repair_succs) * 10.0

            node = graph.get_node(nid)
            cost = self._estimate_node_cost(node)
            score += cost * 2.0

            preds = graph.predecessors(nid)
            repair_preds = [p for p in preds if p in repair_set]
            if repair_preds:
                score += 5.0

            scores[nid] = score

        if not scores:
            return set()

        threshold = max(scores.values()) * 0.5
        candidates = {nid for nid, s in scores.items() if s >= threshold}

        return candidates

    def compute_column_impact(
        self,
        graph: Any,
        source_node: str,
        column: str,
    ) -> ColumnImpact:
        """Trace the impact of a single column through the pipeline.

        Follows column references through operators, tracking renames,
        transformations, and projections.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        source_node : str
            The node where the column change originates.
        column : str
            The affected column name.

        Returns
        -------
        ColumnImpact
        """
        impacts: Dict[str, Set[str]] = {}
        transformations: Dict[str, List[str]] = {}

        current_columns: Dict[str, Set[str]] = {source_node: {column}}
        impacts[source_node] = {column}
        transformations[source_node] = [column]

        topo = graph.topological_sort()
        node_index = {nid: i for i, nid in enumerate(topo)}

        if source_node not in node_index:
            return ColumnImpact(source_node=source_node, source_column=column)

        for nid in topo[node_index[source_node] + 1:]:
            preds = graph.predecessors(nid)
            incoming_cols: Set[str] = set()

            for pred in preds:
                if pred in current_columns:
                    incoming_cols.update(current_columns[pred])

                    edge_key = (pred, nid)
                    if graph.has_edge(pred, nid):
                        edge = graph.get_edge(pred, nid)
                        col_map = getattr(edge, "column_mapping", {})
                        if col_map:
                            mapped = set()
                            for c in current_columns[pred]:
                                if c in col_map:
                                    mapped.add(col_map[c])
                                else:
                                    mapped.add(c)
                            incoming_cols = mapped

            if not incoming_cols:
                continue

            node = graph.get_node(nid)
            output_cols = self._trace_columns_through_operator(
                node, incoming_cols
            )

            if output_cols:
                current_columns[nid] = output_cols
                impacts[nid] = output_cols
                transformations[nid] = sorted(output_cols)

        return ColumnImpact(
            source_node=source_node,
            source_column=column,
            impacts=impacts,
            transformation_chain=transformations,
        )

    def find_schema_compatible_paths(
        self,
        graph: Any,
        delta: Any,
    ) -> List[List[str]]:
        """Find paths through the graph where the schema delta is compatible.

        A path is compatible if the delta can be propagated without schema
        conflicts at any node along the path.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        delta : CompoundPerturbation
            The perturbation.

        Returns
        -------
        list[list[str]]
            List of compatible paths (each path is a list of node ids).
        """
        compatible_paths: List[List[str]] = []
        sources = graph.sources()
        sinks = graph.sinks()

        for source in sources:
            for sink in sinks:
                paths = graph.all_paths(source, sink)
                for path in paths:
                    if self._is_path_schema_compatible(graph, path, delta):
                        compatible_paths.append(path)

        return compatible_paths

    def compute_transitive_closure(
        self,
        graph: Any,
    ) -> Dict[str, Set[str]]:
        """Compute the transitive closure of the pipeline graph.

        For each node, compute the set of all nodes reachable from it.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.

        Returns
        -------
        dict[str, set[str]]
            Mapping from node to all reachable nodes.
        """
        closure: Dict[str, Set[str]] = {}
        topo = list(reversed(graph.topological_sort()))

        for nid in topo:
            reachable: Set[str] = set()
            for succ in graph.successors(nid):
                reachable.add(succ)
                if succ in closure:
                    reachable.update(closure[succ])
            closure[nid] = reachable

        return closure

    def dominance_frontier(
        self,
        graph: Any,
        node_id: str,
    ) -> Set[str]:
        """Compute the dominance frontier of a node.

        The dominance frontier of node N is the set of nodes where N's
        dominance ends — nodes whose immediate predecessors include at
        least one node dominated by N and at least one not dominated by N.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        node_id : str
            The node to compute the frontier for.

        Returns
        -------
        set[str]
            The dominance frontier.
        """
        dominators = self._compute_dominators(graph)
        dominated = self._get_dominated_set(dominators, node_id)

        frontier: Set[str] = set()

        for nid in dominated:
            for succ in graph.successors(nid):
                if succ not in dominated or succ == node_id:
                    preds = set(graph.predecessors(succ))
                    has_dominated_pred = any(p in dominated for p in preds)
                    has_undominated_pred = any(p not in dominated for p in preds)

                    if has_dominated_pred and (has_undominated_pred or succ not in dominated):
                        frontier.add(succ)

        return frontier

    def loop_nesting_depth(
        self,
        graph: Any,
    ) -> Dict[str, int]:
        """Compute loop nesting depth for graphs with feedback edges.

        For DAGs this returns 0 for all nodes. For graphs with cycles,
        computes how deeply each node is nested in feedback loops.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph (may have cycles).

        Returns
        -------
        dict[str, int]
            Mapping from node_id to loop nesting depth.
        """
        depths: Dict[str, int] = {nid: 0 for nid in graph.node_ids}

        if graph.is_dag():
            return depths

        cycles = graph.detect_cycles()

        for cycle in cycles:
            cycle_set = set(cycle)
            for nid in cycle_set:
                depths[nid] = depths.get(nid, 0) + 1

        return depths

    def compute_repair_priority(
        self,
        graph: Any,
        affected_nodes: Set[str],
    ) -> List[Tuple[str, float]]:
        """Compute repair priority for each affected node.

        Priority is based on:
        - Number of downstream dependents
        - Cost of the node
        - Whether it's a bottleneck (high fan-in/fan-out)
        - Distance from sources

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        affected_nodes : set[str]
            The affected node set.

        Returns
        -------
        list[tuple[str, float]]
            Sorted list of (node_id, priority) pairs, highest first.
        """
        priorities: Dict[str, float] = {}

        for nid in affected_nodes:
            priority = 0.0

            desc = graph.descendants(nid)
            affected_desc = desc & affected_nodes
            priority += len(affected_desc) * 5.0

            node = graph.get_node(nid)
            cost = self._estimate_node_cost(node)
            priority += cost

            in_deg = graph.in_degree(nid)
            out_deg = graph.out_degree(nid)
            priority += (in_deg + out_deg) * 2.0

            anc = graph.ancestors(nid)
            depth = len(anc)
            priority += (1.0 / (depth + 1)) * 10.0

            priorities[nid] = priority

        sorted_priorities = sorted(
            priorities.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_priorities

    def compute_repair_dependencies(
        self,
        graph: Any,
        repair_set: Set[str],
    ) -> Dict[str, Set[str]]:
        """Compute inter-repair dependencies.

        For each node in the repair set, find which other repair nodes
        must be completed before it can be repaired.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        repair_set : set[str]
            The repair set.

        Returns
        -------
        dict[str, set[str]]
            Mapping from node to its repair dependencies.
        """
        deps: Dict[str, Set[str]] = {}

        for nid in repair_set:
            node_deps: Set[str] = set()
            anc = graph.ancestors(nid)
            node_deps = anc & repair_set
            deps[nid] = node_deps

        return deps

    def find_independent_repair_groups(
        self,
        graph: Any,
        repair_set: Set[str],
    ) -> List[Set[str]]:
        """Find independent groups of repair nodes.

        Nodes in different groups have no dependency relationship and
        can be repaired completely independently.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        repair_set : set[str]
            The repair set.

        Returns
        -------
        list[set[str]]
            List of independent repair groups.
        """
        adj: Dict[str, Set[str]] = defaultdict(set)

        for nid in repair_set:
            desc = graph.descendants(nid)
            anc = graph.ancestors(nid)
            related = (desc | anc) & repair_set

            for r in related:
                adj[nid].add(r)
                adj[r].add(nid)

        visited: Set[str] = set()
        groups: List[Set[str]] = []

        for nid in repair_set:
            if nid in visited:
                continue

            group: Set[str] = set()
            queue: deque[str] = deque([nid])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                group.add(current)

                for neighbor in adj.get(current, set()):
                    if neighbor not in visited and neighbor in repair_set:
                        queue.append(neighbor)

            if group:
                groups.append(group)

        return groups

    def estimate_total_repair_time(
        self,
        graph: Any,
        repair_set: Set[str],
        parallelism: int = 1,
    ) -> float:
        """Estimate total wall-clock time for repairs.

        Takes into account wave parallelism and resource constraints.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        repair_set : set[str]
            The repair set.
        parallelism : int
            Maximum number of parallel workers.

        Returns
        -------
        float
            Estimated total time.
        """
        waves = self.compute_repair_waves(graph, repair_set)

        total_time = 0.0
        for wave in waves:
            node_costs = []
            for nid in wave.nodes:
                node = graph.get_node(nid)
                node_costs.append(self._estimate_node_cost(node))

            node_costs.sort(reverse=True)

            if parallelism >= len(node_costs):
                wave_time = node_costs[0] if node_costs else 0.0
            else:
                chunks = [
                    node_costs[i:i + parallelism]
                    for i in range(0, len(node_costs), parallelism)
                ]
                wave_time = sum(max(chunk) for chunk in chunks)

            total_time += wave_time

        return total_time

    # ── Internal Helpers ──────────────────────────────────────────

    def _estimate_node_cost(self, node: Any) -> float:
        """Extract or estimate the repair cost for a node."""
        if hasattr(node, "cost_estimate"):
            ce = node.cost_estimate
            if hasattr(ce, "total_weighted_cost"):
                return ce.total_weighted_cost
        return 1.0

    def _compute_severity(
        self,
        distance: int,
        cost: float,
        path_count: int,
    ) -> ImpactSeverity:
        """Compute impact severity from distance, cost, and path count."""
        score = (
            cost * self._cost_weight
            + (1.0 / (distance + 1)) * self._distance_weight * 10.0
            + math.log2(path_count + 1) * 2.0
        )

        if score >= 20.0:
            return ImpactSeverity.CRITICAL
        if score >= 10.0:
            return ImpactSeverity.HIGH
        if score >= 5.0:
            return ImpactSeverity.MEDIUM
        if score >= 1.0:
            return ImpactSeverity.LOW
        return ImpactSeverity.NONE

    def _get_affected_columns(self, delta: Any, node: Any) -> Set[str]:
        """Extract affected columns from a delta for a specific node."""
        cols: Set[str] = set()

        if hasattr(delta, "schema_delta") and delta.schema_delta is not None:
            for op in delta.schema_delta.operations:
                if hasattr(op, "column_def"):
                    cols.add(op.column_def.name)
                if hasattr(op, "column_name"):
                    cols.add(op.column_name)
                if hasattr(op, "old_name"):
                    cols.add(op.old_name)
                if hasattr(op, "new_name"):
                    cols.add(op.new_name)

        return cols

    def _find_critical_path(
        self,
        graph: Any,
        source: str,
        impacts: Dict[str, NodeImpact],
    ) -> List[str]:
        """Find the most expensive path through impacted nodes."""
        if not impacts:
            return [source]

        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}

        topo = graph.topological_sort()

        for nid in topo:
            if nid not in dist:
                continue

            for succ in graph.successors(nid):
                if succ not in impacts:
                    continue

                cost = impacts[succ].cost_to_repair
                new_dist = dist[nid] + cost

                if succ not in dist or new_dist > dist[succ]:
                    dist[succ] = new_dist
                    prev[succ] = nid

        if not dist:
            return [source]

        end = max(
            (n for n in dist if n in impacts),
            key=lambda n: dist[n],
            default=source,
        )

        path: List[str] = []
        current: Optional[str] = end
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()

        return path

    def _get_descendants_in_set(
        self,
        graph: Any,
        node_id: str,
        target_set: Set[str],
    ) -> Set[str]:
        """Get all descendants of node_id that are in target_set."""
        desc = graph.descendants(node_id)
        return desc & target_set

    def _trace_columns_through_operator(
        self,
        node: Any,
        input_columns: Set[str],
    ) -> Set[str]:
        """Trace columns through a node's operator.

        For most operators, columns pass through unchanged. SELECTs may
        rename or drop columns. JOINs may add columns.
        """
        output_cols = set(input_columns)

        if hasattr(node, "output_schema"):
            schema = node.output_schema
            if hasattr(schema, "columns") and schema.columns:
                schema_cols = {c.name for c in schema.columns}
                output_cols = input_columns & schema_cols

                if not output_cols:
                    output_cols = input_columns

        return output_cols

    def _is_path_schema_compatible(
        self,
        graph: Any,
        path: List[str],
        delta: Any,
    ) -> bool:
        """Check if a schema delta is compatible along a path."""
        if not path:
            return True

        for nid in path:
            node = graph.get_node(nid)
            if hasattr(node, "output_schema"):
                schema = node.output_schema
                if hasattr(schema, "columns") and schema.columns:
                    continue
            else:
                continue

        return True

    def _compute_dominators(
        self,
        graph: Any,
    ) -> Dict[str, Set[str]]:
        """Compute the dominator sets for all nodes in the graph.

        Uses the iterative dataflow algorithm for dominator computation.
        """
        topo = graph.topological_sort()
        all_nodes = set(topo)
        dom: Dict[str, Set[str]] = {}

        if not topo:
            return dom

        entry = topo[0]
        dom[entry] = {entry}

        for nid in topo[1:]:
            dom[nid] = set(all_nodes)

        changed = True
        while changed:
            changed = False
            for nid in topo[1:]:
                preds = graph.predecessors(nid)
                if not preds:
                    new_dom = {nid}
                else:
                    new_dom = set(all_nodes)
                    for pred in preds:
                        if pred in dom:
                            new_dom = new_dom & dom[pred]
                    new_dom.add(nid)

                if new_dom != dom[nid]:
                    dom[nid] = new_dom
                    changed = True

        return dom

    def _get_dominated_set(
        self,
        dominators: Dict[str, Set[str]],
        node_id: str,
    ) -> Set[str]:
        """Get the set of all nodes dominated by node_id."""
        dominated: Set[str] = set()
        for nid, doms in dominators.items():
            if node_id in doms:
                dominated.add(nid)
        return dominated


# =====================================================================
# Convenience Functions
# =====================================================================


def compute_impact(
    graph: Any,
    node_id: str,
    delta: Any = None,
) -> ImpactSet:
    """Convenience function for impact analysis."""
    analyzer = DependencyAnalyzer()
    return analyzer.compute_impact_set(graph, node_id, delta)


def compute_repair_waves(
    graph: Any,
    affected_nodes: Set[str],
) -> List[RepairWave]:
    """Convenience function for repair wave computation."""
    analyzer = DependencyAnalyzer()
    repair_set = analyzer.find_minimal_repair_set(graph, affected_nodes)
    return analyzer.compute_repair_waves(graph, repair_set)


def compute_column_lineage(
    graph: Any,
    source_node: str,
    column: str,
) -> ColumnImpact:
    """Convenience function for column-level impact analysis."""
    analyzer = DependencyAnalyzer()
    return analyzer.compute_column_impact(graph, source_node, column)
