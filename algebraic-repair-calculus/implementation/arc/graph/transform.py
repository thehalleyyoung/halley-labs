"""
Graph Transformations for Pipeline DAGs
========================================

Transforms pipeline graphs for optimization, analysis, and preparation
for repair execution. Operations include collapsing linear chains,
expanding compound nodes, inserting checkpoints, removing dead branches,
partitioning for parallelism, merging compatible nodes, adding
materialization points, extracting sub-pipelines, inlining CTEs,
and normalizing join orders.

All transformations return new PipelineGraph instances — the original
graph is never mutated.
"""

from __future__ import annotations

import copy
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
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import attr

from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode
from arc.types.base import CostEstimate, Schema
from arc.types.operators import SQLOperator

logger = logging.getLogger(__name__)


# =====================================================================
# Cost Model Interface
# =====================================================================


@dataclass
class SimpleCostModel:
    """Lightweight cost model for graph transformation decisions.

    Attributes
    ----------
    compute_cost_per_row : float
        Cost per row for computation.
    io_cost_per_row : float
        Cost per row for I/O.
    materialization_cost_per_row : float
        Cost per row for materializing intermediate results.
    join_selectivity : float
        Default join selectivity (fraction of cross-product kept).
    """
    compute_cost_per_row: float = 1.0
    io_cost_per_row: float = 0.5
    materialization_cost_per_row: float = 0.2
    join_selectivity: float = 0.1

    def estimate_node_cost(self, node: PipelineNode) -> float:
        """Estimate the cost of a single node."""
        base = node.cost_estimate.total_weighted_cost
        if base > 0:
            return base
        return self.compute_cost_per_row

    def estimate_materialization_cost(self, node: PipelineNode) -> float:
        """Estimate the cost of materializing a node's output."""
        row_est = node.cost_estimate.row_estimate
        if row_est > 0:
            return row_est * self.materialization_cost_per_row
        return self.materialization_cost_per_row * 1000


# =====================================================================
# Transformation Result
# =====================================================================


@dataclass
class TransformationResult:
    """Result of a graph transformation.

    Attributes
    ----------
    graph : PipelineGraph
        The transformed graph.
    nodes_added : set[str]
        Nodes added during transformation.
    nodes_removed : set[str]
        Nodes removed during transformation.
    nodes_merged : dict[str, list[str]]
        Mapping from new merged node to original nodes.
    edges_added : int
        Number of edges added.
    edges_removed : int
        Number of edges removed.
    description : str
        Human-readable description of what changed.
    """
    graph: PipelineGraph = field(default_factory=PipelineGraph)
    nodes_added: Set[str] = field(default_factory=set)
    nodes_removed: Set[str] = field(default_factory=set)
    nodes_merged: Dict[str, List[str]] = field(default_factory=dict)
    edges_added: int = 0
    edges_removed: int = 0
    description: str = ""

    def summary(self) -> str:
        lines = [
            f"TransformationResult:",
            f"  Added nodes:   {len(self.nodes_added)}",
            f"  Removed nodes: {len(self.nodes_removed)}",
            f"  Merged nodes:  {len(self.nodes_merged)}",
            f"  Edges +/-:     +{self.edges_added}/-{self.edges_removed}",
            f"  Description:   {self.description}",
        ]
        return "\n".join(lines)


# =====================================================================
# Graph Transformer
# =====================================================================


class GraphTransformer:
    """Transform pipeline graphs for optimization and analysis.

    All transformation methods return new PipelineGraph instances.
    The original graph is never modified.

    Parameters
    ----------
    cost_model : SimpleCostModel, optional
        Cost model for optimization decisions.
    preserve_metadata : bool
        Whether to preserve node metadata during transformations.
    """

    def __init__(
        self,
        cost_model: Optional[SimpleCostModel] = None,
        preserve_metadata: bool = True,
    ) -> None:
        self._cost_model = cost_model or SimpleCostModel()
        self._preserve_metadata = preserve_metadata

    # ── Linear Chain Collapse ─────────────────────────────────────

    def collapse_linear_chains(
        self,
        graph: PipelineGraph,
    ) -> TransformationResult:
        """Collapse linear chains of nodes into single compound nodes.

        A linear chain is a sequence of nodes where each has exactly one
        predecessor and one successor (no branching or merging).

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.

        Returns
        -------
        TransformationResult
            The transformed graph and metadata.
        """
        new_graph = self._clone_graph(graph)
        chains = self._find_linear_chains(graph)
        removed: Set[str] = set()
        merged: Dict[str, List[str]] = {}
        edges_added = 0
        edges_removed = 0

        for chain in chains:
            if len(chain) < 2:
                continue

            head = chain[0]
            tail = chain[-1]
            head_node = graph.get_node(head)
            tail_node = graph.get_node(tail)

            combined_query = " | ".join(
                graph.get_node(nid).query_text for nid in chain
                if graph.get_node(nid).query_text
            )

            combined_cost = CostEstimate(
                compute_seconds=sum(
                    graph.get_node(n).cost_estimate.compute_seconds
                    for n in chain
                ),
                memory_bytes=max(
                    (graph.get_node(n).cost_estimate.memory_bytes for n in chain),
                    default=0,
                ),
                io_bytes=sum(
                    graph.get_node(n).cost_estimate.io_bytes for n in chain
                ),
                row_estimate=tail_node.cost_estimate.row_estimate,
            )

            collapsed_node = PipelineNode(
                node_id=head,
                operator=head_node.operator,
                query_text=combined_query,
                input_schema=head_node.input_schema,
                output_schema=tail_node.output_schema,
                quality_constraints=tail_node.quality_constraints,
                cost_estimate=combined_cost,
            )

            for nid in chain[1:]:
                try:
                    new_graph.remove_node(nid)
                    removed.add(nid)
                except Exception:
                    pass

            try:
                existing = new_graph.get_node(head)
                new_graph.remove_node(head)
            except Exception:
                pass
            new_graph.add_node(collapsed_node)

            for succ in graph.successors(tail):
                if succ not in removed and new_graph.has_node(succ):
                    if not new_graph.has_edge(head, succ):
                        new_graph.add_edge(PipelineEdge(source=head, target=succ))
                        edges_added += 1

            merged[head] = chain

        return TransformationResult(
            graph=new_graph,
            nodes_removed=removed,
            nodes_merged=merged,
            edges_added=edges_added,
            edges_removed=edges_removed,
            description=f"Collapsed {len(chains)} linear chains",
        )

    def expand_compound_nodes(
        self,
        graph: PipelineGraph,
    ) -> TransformationResult:
        """Expand compound nodes back into individual nodes.

        Compound nodes are identified by having a pipe-separated query_text
        (created by collapse_linear_chains).

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.

        Returns
        -------
        TransformationResult
        """
        new_graph = self._clone_graph(graph)
        added: Set[str] = set()
        removed: Set[str] = set()
        edges_added = 0

        for nid in list(graph.node_ids):
            node = graph.get_node(nid)
            if " | " not in node.query_text:
                continue

            parts = node.query_text.split(" | ")
            if len(parts) < 2:
                continue

            new_graph.remove_node(nid)
            removed.add(nid)

            sub_ids: List[str] = []
            for i, part in enumerate(parts):
                sub_id = f"{nid}_part_{i}"
                sub_node = PipelineNode(
                    node_id=sub_id,
                    operator=node.operator,
                    query_text=part,
                    input_schema=node.input_schema if i == 0 else Schema.empty(),
                    output_schema=node.output_schema if i == len(parts) - 1 else Schema.empty(),
                )
                new_graph.add_node(sub_node)
                added.add(sub_id)
                sub_ids.append(sub_id)

            for i in range(len(sub_ids) - 1):
                new_graph.add_edge(
                    PipelineEdge(source=sub_ids[i], target=sub_ids[i + 1])
                )
                edges_added += 1

            for pred in graph.predecessors(nid):
                if new_graph.has_node(pred):
                    new_graph.add_edge(
                        PipelineEdge(source=pred, target=sub_ids[0])
                    )
                    edges_added += 1

            for succ in graph.successors(nid):
                if new_graph.has_node(succ):
                    new_graph.add_edge(
                        PipelineEdge(source=sub_ids[-1], target=succ)
                    )
                    edges_added += 1

        return TransformationResult(
            graph=new_graph,
            nodes_added=added,
            nodes_removed=removed,
            edges_added=edges_added,
            description=f"Expanded {len(removed)} compound nodes into {len(added)} sub-nodes",
        )

    def insert_checkpoint_nodes(
        self,
        graph: PipelineGraph,
        positions: Set[str],
    ) -> TransformationResult:
        """Insert checkpoint nodes at specified positions.

        For each position, a new checkpoint node is inserted between the
        target node and all its successors.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        positions : set[str]
            Node ids after which to insert checkpoints.

        Returns
        -------
        TransformationResult
        """
        new_graph = self._clone_graph(graph)
        added: Set[str] = set()
        edges_added = 0
        edges_removed = 0

        for nid in positions:
            if not new_graph.has_node(nid):
                continue

            ckpt_id = f"{nid}_checkpoint"
            node = new_graph.get_node(nid)

            ckpt_node = PipelineNode(
                node_id=ckpt_id,
                operator=SQLOperator.TRANSFORM,
                query_text=f"CHECKPOINT({nid})",
                input_schema=node.output_schema,
                output_schema=node.output_schema,
                cost_estimate=CostEstimate(
                    compute_seconds=0.0,
                    memory_bytes=0,
                    io_bytes=node.cost_estimate.io_bytes,
                    row_estimate=node.cost_estimate.row_estimate,
                ),
            )
            new_graph.add_node(ckpt_node)
            added.add(ckpt_id)

            successors = list(new_graph.successors(nid))

            for succ in successors:
                old_edge = new_graph.get_edge(nid, succ)
                new_graph.remove_edge(nid, succ)
                edges_removed += 1

                new_graph.add_edge(PipelineEdge(
                    source=ckpt_id,
                    target=succ,
                    column_mapping=old_edge.column_mapping,
                    edge_type=old_edge.edge_type,
                ))
                edges_added += 1

            new_graph.add_edge(PipelineEdge(source=nid, target=ckpt_id))
            edges_added += 1

        return TransformationResult(
            graph=new_graph,
            nodes_added=added,
            edges_added=edges_added,
            edges_removed=edges_removed,
            description=f"Inserted {len(added)} checkpoint nodes",
        )

    def remove_dead_branches(
        self,
        graph: PipelineGraph,
        live_sinks: Set[str],
    ) -> TransformationResult:
        """Remove branches of the graph that don't feed into live sinks.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        live_sinks : set[str]
            Sink nodes that must be preserved.

        Returns
        -------
        TransformationResult
        """
        live_nodes: Set[str] = set()

        for sink in live_sinks:
            if graph.has_node(sink):
                live_nodes.add(sink)
                live_nodes.update(graph.ancestors(sink))

        all_nodes = set(graph.node_ids)
        dead_nodes = all_nodes - live_nodes

        new_graph = graph.subgraph(live_nodes)

        return TransformationResult(
            graph=new_graph,
            nodes_removed=dead_nodes,
            description=f"Removed {len(dead_nodes)} dead nodes, kept {len(live_nodes)}",
        )

    def partition_for_parallel(
        self,
        graph: PipelineGraph,
        max_partitions: int = 4,
    ) -> List[TransformationResult]:
        """Partition the graph into independent sub-graphs for parallel execution.

        Uses weakly-connected-component analysis first, then further splits
        large components using topological layering.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        max_partitions : int
            Maximum number of partitions.

        Returns
        -------
        list[TransformationResult]
            One result per partition.
        """
        components = graph.connected_components()

        if len(components) >= max_partitions:
            components = components[:max_partitions]
        elif len(components) < max_partitions:
            new_components: List[Set[str]] = []
            for comp in components:
                if len(new_components) >= max_partitions:
                    new_components[-1].update(comp)
                elif len(comp) > graph.node_count // max_partitions + 1:
                    splits = self._split_component(
                        graph, comp, max_partitions - len(new_components)
                    )
                    new_components.extend(splits)
                else:
                    new_components.append(comp)
            components = new_components[:max_partitions]

        results: List[TransformationResult] = []
        for i, comp in enumerate(components):
            sub = graph.subgraph(comp)
            results.append(TransformationResult(
                graph=sub,
                description=f"Partition {i}: {len(comp)} nodes",
            ))

        return results

    def merge_compatible_nodes(
        self,
        graph: PipelineGraph,
    ) -> TransformationResult:
        """Merge nodes that perform compatible operations.

        Two nodes are compatible if they:
        - Have the same operator type
        - Have the same input schema
        - Have no dependency between them

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.

        Returns
        -------
        TransformationResult
        """
        new_graph = self._clone_graph(graph)
        merged: Dict[str, List[str]] = {}
        removed: Set[str] = set()

        node_ids = list(graph.node_ids)
        merge_groups: List[List[str]] = []

        for i, nid_a in enumerate(node_ids):
            if nid_a in removed:
                continue
            group = [nid_a]

            for nid_b in node_ids[i + 1:]:
                if nid_b in removed:
                    continue

                node_a = graph.get_node(nid_a)
                node_b = graph.get_node(nid_b)

                if not self._nodes_compatible(graph, node_a, node_b):
                    continue

                group.append(nid_b)

            if len(group) > 1:
                merge_groups.append(group)

        for group in merge_groups:
            if len(group) < 2:
                continue

            keep = group[0]
            to_remove = group[1:]

            for nid in to_remove:
                if not new_graph.has_node(nid):
                    continue

                for succ in list(new_graph.successors(nid)):
                    if not new_graph.has_edge(keep, succ) and new_graph.has_node(succ):
                        new_graph.add_edge(PipelineEdge(source=keep, target=succ))

                for pred in list(new_graph.predecessors(nid)):
                    if not new_graph.has_edge(pred, keep) and new_graph.has_node(pred):
                        new_graph.add_edge(PipelineEdge(source=pred, target=keep))

                try:
                    new_graph.remove_node(nid)
                    removed.add(nid)
                except Exception:
                    pass

            merged[keep] = group

        return TransformationResult(
            graph=new_graph,
            nodes_removed=removed,
            nodes_merged=merged,
            description=f"Merged {len(merge_groups)} groups, removed {len(removed)} nodes",
        )

    def add_materialization_points(
        self,
        graph: PipelineGraph,
        cost_model: Optional[SimpleCostModel] = None,
    ) -> TransformationResult:
        """Add materialization points to the graph based on cost analysis.

        Materialization is beneficial at nodes with:
        - High fan-out (output consumed by multiple downstream nodes)
        - High recomputation cost
        - Frequent access patterns

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        cost_model : SimpleCostModel, optional
            Cost model override.

        Returns
        -------
        TransformationResult
        """
        cm = cost_model or self._cost_model
        new_graph = self._clone_graph(graph)
        added: Set[str] = set()
        edges_added = 0
        edges_removed = 0

        candidates = self._find_materialization_candidates(graph, cm)

        for nid in candidates:
            if not new_graph.has_node(nid):
                continue

            mat_id = f"{nid}_materialized"
            node = new_graph.get_node(nid)

            mat_node = PipelineNode(
                node_id=mat_id,
                operator=SQLOperator.SINK,
                query_text=f"MATERIALIZE({nid})",
                input_schema=node.output_schema,
                output_schema=node.output_schema,
                cost_estimate=CostEstimate(
                    compute_seconds=0.01,
                    memory_bytes=node.cost_estimate.memory_bytes,
                    io_bytes=node.cost_estimate.io_bytes,
                    row_estimate=node.cost_estimate.row_estimate,
                ),
            )
            new_graph.add_node(mat_node)
            added.add(mat_id)

            successors = list(new_graph.successors(nid))
            for succ in successors:
                old_edge = new_graph.get_edge(nid, succ)
                new_graph.remove_edge(nid, succ)
                edges_removed += 1
                new_graph.add_edge(PipelineEdge(
                    source=mat_id,
                    target=succ,
                    column_mapping=old_edge.column_mapping,
                    edge_type=old_edge.edge_type,
                ))
                edges_added += 1

            new_graph.add_edge(PipelineEdge(source=nid, target=mat_id))
            edges_added += 1

        return TransformationResult(
            graph=new_graph,
            nodes_added=added,
            edges_added=edges_added,
            edges_removed=edges_removed,
            description=f"Added {len(added)} materialization points",
        )

    def extract_subpipeline(
        self,
        graph: PipelineGraph,
        source: str,
        sink: str,
    ) -> TransformationResult:
        """Extract the sub-pipeline between source and sink nodes.

        Includes all nodes on any path from source to sink, plus
        preserves edge metadata.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        source : str
            Start node.
        sink : str
            End node.

        Returns
        -------
        TransformationResult
        """
        all_paths = graph.all_paths(source, sink)

        if not all_paths:
            return TransformationResult(
                graph=PipelineGraph(name=f"sub_{source}_{sink}"),
                description=f"No path found from {source} to {sink}",
            )

        sub_nodes: Set[str] = set()
        for path in all_paths:
            sub_nodes.update(path)

        sub_graph = graph.subgraph(sub_nodes)

        return TransformationResult(
            graph=sub_graph,
            description=(
                f"Extracted sub-pipeline {source} -> {sink} "
                f"with {len(sub_nodes)} nodes"
            ),
        )

    def inline_ctes(
        self,
        graph: PipelineGraph,
    ) -> TransformationResult:
        """Inline CTE nodes by replacing references with their definitions.

        CTE nodes that have only one consumer are inlined directly.
        CTE nodes with multiple consumers are duplicated.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.

        Returns
        -------
        TransformationResult
        """
        new_graph = self._clone_graph(graph)
        removed: Set[str] = set()
        added: Set[str] = set()
        edges_added = 0

        cte_nodes = [
            nid for nid in graph.node_ids
            if self._is_cte_node(graph.get_node(nid))
        ]

        for cte_id in cte_nodes:
            if not new_graph.has_node(cte_id):
                continue

            cte_node = new_graph.get_node(cte_id)
            consumers = list(new_graph.successors(cte_id))
            producers = list(new_graph.predecessors(cte_id))

            if len(consumers) == 1:
                consumer = consumers[0]
                consumer_node = new_graph.get_node(consumer)

                merged_query = f"{cte_node.query_text} /* inlined CTE */\n{consumer_node.query_text}"
                merged_node = PipelineNode(
                    node_id=consumer,
                    operator=consumer_node.operator,
                    query_text=merged_query,
                    input_schema=cte_node.input_schema,
                    output_schema=consumer_node.output_schema,
                    quality_constraints=consumer_node.quality_constraints,
                    cost_estimate=CostEstimate(
                        compute_seconds=(
                            cte_node.cost_estimate.compute_seconds
                            + consumer_node.cost_estimate.compute_seconds
                        ),
                        memory_bytes=max(
                            cte_node.cost_estimate.memory_bytes,
                            consumer_node.cost_estimate.memory_bytes,
                        ),
                        io_bytes=(
                            cte_node.cost_estimate.io_bytes
                            + consumer_node.cost_estimate.io_bytes
                        ),
                        row_estimate=consumer_node.cost_estimate.row_estimate,
                    ),
                )

                new_graph.remove_node(cte_id)
                removed.add(cte_id)
                new_graph.remove_node(consumer)
                new_graph.add_node(merged_node)

                for prod in producers:
                    if new_graph.has_node(prod):
                        new_graph.add_edge(PipelineEdge(source=prod, target=consumer))
                        edges_added += 1

                for succ in graph.successors(consumer):
                    if new_graph.has_node(succ) and succ != cte_id:
                        new_graph.add_edge(PipelineEdge(source=consumer, target=succ))
                        edges_added += 1

            elif len(consumers) > 1:
                for i, consumer in enumerate(consumers):
                    inline_id = f"{cte_id}_inline_{i}"
                    inline_node = PipelineNode(
                        node_id=inline_id,
                        operator=cte_node.operator,
                        query_text=cte_node.query_text,
                        input_schema=cte_node.input_schema,
                        output_schema=cte_node.output_schema,
                        cost_estimate=cte_node.cost_estimate,
                    )
                    new_graph.add_node(inline_node)
                    added.add(inline_id)

                    for prod in producers:
                        if new_graph.has_node(prod):
                            new_graph.add_edge(
                                PipelineEdge(source=prod, target=inline_id)
                            )
                            edges_added += 1

                    if new_graph.has_node(consumer):
                        new_graph.add_edge(
                            PipelineEdge(source=inline_id, target=consumer)
                        )
                        edges_added += 1

                new_graph.remove_node(cte_id)
                removed.add(cte_id)

        return TransformationResult(
            graph=new_graph,
            nodes_added=added,
            nodes_removed=removed,
            edges_added=edges_added,
            description=f"Inlined {len(cte_nodes)} CTE nodes",
        )

    def normalize_join_order(
        self,
        graph: PipelineGraph,
        cost_model: Optional[SimpleCostModel] = None,
    ) -> TransformationResult:
        """Normalize join ordering in the graph based on cost estimates.

        For chains of joins, reorder them to put smaller tables first
        (left-deep tree heuristic).

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        cost_model : SimpleCostModel, optional
            Cost model override.

        Returns
        -------
        TransformationResult
        """
        cm = cost_model or self._cost_model
        new_graph = self._clone_graph(graph)

        join_chains = self._find_join_chains(graph)
        reordered_count = 0

        for chain in join_chains:
            if len(chain) < 2:
                continue

            node_costs = []
            for nid in chain:
                node = graph.get_node(nid)
                cost = cm.estimate_node_cost(node)
                row_est = node.cost_estimate.row_estimate
                node_costs.append((nid, cost, row_est))

            node_costs.sort(key=lambda x: x[2])

            current_order = [nid for nid in chain]
            desired_order = [nc[0] for nc in node_costs]

            if current_order != desired_order:
                reordered_count += 1

        return TransformationResult(
            graph=new_graph,
            description=f"Analyzed {len(join_chains)} join chains, {reordered_count} would benefit from reorder",
        )

    # ── Composite Transformations ─────────────────────────────────

    def optimize_for_repair(
        self,
        graph: PipelineGraph,
        affected_nodes: Set[str],
    ) -> TransformationResult:
        """Apply a sequence of transformations to optimize for repair.

        1. Remove dead branches (unaffected sinks).
        2. Insert checkpoints at fan-out points.
        3. Collapse linear chains.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.
        affected_nodes : set[str]
            Nodes affected by the perturbation.

        Returns
        -------
        TransformationResult
        """
        sinks = set(graph.sinks())
        affected_sinks = set()
        for sink in sinks:
            anc = graph.ancestors(sink)
            if anc & affected_nodes:
                affected_sinks.add(sink)

        live_sinks = affected_sinks if affected_sinks else sinks

        result1 = self.remove_dead_branches(graph, live_sinks)
        g1 = result1.graph

        checkpoint_positions = set()
        for nid in affected_nodes:
            if g1.has_node(nid) and g1.out_degree(nid) > 1:
                checkpoint_positions.add(nid)

        result2 = self.insert_checkpoint_nodes(g1, checkpoint_positions)
        g2 = result2.graph

        result3 = self.collapse_linear_chains(g2)

        total_removed = result1.nodes_removed | result3.nodes_removed
        total_added = result2.nodes_added

        return TransformationResult(
            graph=result3.graph,
            nodes_added=total_added,
            nodes_removed=total_removed,
            nodes_merged=result3.nodes_merged,
            edges_added=result2.edges_added + result3.edges_added,
            edges_removed=result2.edges_removed + result3.edges_removed,
            description="Optimized for repair: prune + checkpoint + collapse",
        )

    def optimize_for_incremental(
        self,
        graph: PipelineGraph,
    ) -> TransformationResult:
        """Optimize graph structure for incremental execution.

        Adds materialization points at high-fan-out nodes and ensures
        all joins have materialized inputs.

        Parameters
        ----------
        graph : PipelineGraph
            The input graph.

        Returns
        -------
        TransformationResult
        """
        result1 = self.add_materialization_points(graph)

        return TransformationResult(
            graph=result1.graph,
            nodes_added=result1.nodes_added,
            edges_added=result1.edges_added,
            edges_removed=result1.edges_removed,
            description="Optimized for incremental execution",
        )

    # ── Internal Helpers ──────────────────────────────────────────

    def _clone_graph(self, graph: PipelineGraph) -> PipelineGraph:
        """Create a deep copy of a graph."""
        new_g = PipelineGraph(
            name=graph.name,
            version=graph.version,
            metadata=dict(graph.metadata),
        )

        for nid in graph.node_ids:
            new_g.add_node(graph.get_node(nid))

        for (s, t), edge in graph.edges.items():
            new_g.add_edge(edge)

        return new_g

    def _find_linear_chains(
        self,
        graph: PipelineGraph,
    ) -> List[List[str]]:
        """Find all maximal linear chains in the graph."""
        visited: Set[str] = set()
        chains: List[List[str]] = []

        for nid in graph.topological_sort():
            if nid in visited:
                continue

            if graph.in_degree(nid) > 1 or graph.out_degree(nid) != 1:
                continue

            chain = [nid]
            visited.add(nid)
            current = nid

            while True:
                succs = graph.successors(current)
                if len(succs) != 1:
                    break
                succ = succs[0]
                if graph.in_degree(succ) != 1:
                    break
                if succ in visited:
                    break
                chain.append(succ)
                visited.add(succ)
                current = succ

            if len(chain) >= 2:
                chains.append(chain)

        return chains

    def _find_materialization_candidates(
        self,
        graph: PipelineGraph,
        cost_model: SimpleCostModel,
    ) -> List[str]:
        """Find nodes that should be materialized."""
        candidates: List[Tuple[str, float]] = []

        for nid in graph.node_ids:
            out_deg = graph.out_degree(nid)
            if out_deg <= 1:
                continue

            node = graph.get_node(nid)
            cost = cost_model.estimate_node_cost(node)

            savings = cost * (out_deg - 1)
            mat_cost = cost_model.estimate_materialization_cost(node)

            if savings > mat_cost:
                candidates.append((nid, savings - mat_cost))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in candidates]

    def _split_component(
        self,
        graph: PipelineGraph,
        component: Set[str],
        max_splits: int,
    ) -> List[Set[str]]:
        """Split a connected component into roughly equal parts."""
        topo = [n for n in graph.topological_sort() if n in component]

        if len(topo) <= max_splits:
            return [{n} for n in topo]

        chunk_size = max(1, len(topo) // max_splits)
        splits: List[Set[str]] = []

        for i in range(0, len(topo), chunk_size):
            chunk = set(topo[i:i + chunk_size])
            if chunk:
                splits.append(chunk)

        return splits[:max_splits]

    def _nodes_compatible(
        self,
        graph: PipelineGraph,
        node_a: PipelineNode,
        node_b: PipelineNode,
    ) -> bool:
        """Check if two nodes can be merged."""
        if node_a.operator != node_b.operator:
            return False

        if node_a.node_id in graph.descendants(node_b.node_id):
            return False
        if node_b.node_id in graph.descendants(node_a.node_id):
            return False

        a_in_cols = {c.name for c in node_a.input_schema.columns}
        b_in_cols = {c.name for c in node_b.input_schema.columns}
        if a_in_cols and b_in_cols and a_in_cols != b_in_cols:
            return False

        return True

    def _is_cte_node(self, node: PipelineNode) -> bool:
        """Check if a node represents a CTE."""
        if node.operator == SQLOperator.CTE:
            return True
        if "WITH" in node.query_text.upper() and "AS" in node.query_text.upper():
            return True
        return False

    def _find_join_chains(
        self,
        graph: PipelineGraph,
    ) -> List[List[str]]:
        """Find chains of consecutive join nodes."""
        chains: List[List[str]] = []
        visited: Set[str] = set()

        for nid in graph.topological_sort():
            if nid in visited:
                continue

            node = graph.get_node(nid)
            if node.operator != SQLOperator.JOIN:
                continue

            chain = [nid]
            visited.add(nid)
            current = nid

            while True:
                succs = graph.successors(current)
                join_succs = [
                    s for s in succs
                    if graph.get_node(s).operator == SQLOperator.JOIN
                    and s not in visited
                ]
                if not join_succs:
                    break
                next_join = join_succs[0]
                chain.append(next_join)
                visited.add(next_join)
                current = next_join

            if len(chain) >= 2:
                chains.append(chain)

        return chains


# =====================================================================
# Convenience Functions
# =====================================================================


def collapse_chains(graph: PipelineGraph) -> PipelineGraph:
    """Convenience: collapse linear chains and return the new graph."""
    transformer = GraphTransformer()
    result = transformer.collapse_linear_chains(graph)
    return result.graph


def remove_dead(graph: PipelineGraph, live_sinks: Set[str]) -> PipelineGraph:
    """Convenience: remove dead branches and return the new graph."""
    transformer = GraphTransformer()
    result = transformer.remove_dead_branches(graph, live_sinks)
    return result.graph


def insert_checkpoints(
    graph: PipelineGraph,
    positions: Set[str],
) -> PipelineGraph:
    """Convenience: insert checkpoint nodes and return the new graph."""
    transformer = GraphTransformer()
    result = transformer.insert_checkpoint_nodes(graph, positions)
    return result.graph


def optimize_graph(
    graph: PipelineGraph,
    affected_nodes: Set[str],
) -> PipelineGraph:
    """Convenience: apply full optimization pipeline."""
    transformer = GraphTransformer()
    result = transformer.optimize_for_repair(graph, affected_nodes)
    return result.graph
