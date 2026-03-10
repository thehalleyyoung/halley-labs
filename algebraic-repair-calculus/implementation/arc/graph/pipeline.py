"""
Pipeline DAG — the typed dependency graph at the heart of ARC.

``PipelineNode`` and ``PipelineEdge`` are immutable value objects that
carry schemas, quality constraints, availability contracts and cost
estimates.  ``PipelineGraph`` is a mutable container backed by
``networkx.DiGraph`` that supports topological traversal, impact
analysis, sub-graph extraction and deep cloning.
"""

from __future__ import annotations

import copy
import itertools
from collections import deque
from typing import Any, Callable, Iterable, Iterator, Sequence

import attr
import networkx as nx
from attr import validators as v

from arc.types.base import (
    AvailabilityContract,
    CostEstimate,
    EdgeType,
    NodeMetadata,
    ParameterisedType,
    QualityConstraint,
    Schema,
)
from arc.types.errors import (
    CycleDetectedError,
    EdgeNotFoundError,
    ErrorCode,
    GraphError,
    GraphMergeConflictError,
    GraphSchemaMismatchError,
    NodeNotFoundError,
)
from arc.types.operators import (
    OperatorProperties,
    OperatorSignature,
    SQLOperator,
    get_default_properties,
)


# =====================================================================
# Pipeline Node
# =====================================================================

@attr.s(frozen=True, slots=True, repr=False, hash=True)
class PipelineNode:
    """A single transformation step in the pipeline DAG.

    Carries everything the algebra engine needs to reason about deltas:
    the operator kind, input/output schemas, quality constraints,
    availability contract and an estimated execution cost.
    """

    node_id: str = attr.ib(validator=v.instance_of(str))
    operator: SQLOperator = attr.ib(default=SQLOperator.TRANSFORM, validator=v.instance_of(SQLOperator))
    query_text: str = attr.ib(default="", validator=v.instance_of(str))
    input_schema: Schema = attr.ib(factory=Schema.empty)
    output_schema: Schema = attr.ib(factory=Schema.empty)
    quality_constraints: tuple[QualityConstraint, ...] = attr.ib(factory=tuple, converter=tuple)
    availability_contract: AvailabilityContract = attr.ib(factory=AvailabilityContract)
    cost_estimate: CostEstimate = attr.ib(factory=CostEstimate)
    properties: OperatorProperties = attr.ib(default=None)
    metadata: NodeMetadata = attr.ib(factory=NodeMetadata)

    def __attrs_post_init__(self) -> None:
        if self.properties is None:
            object.__setattr__(self, "properties", get_default_properties(self.operator))

    @property
    def in_fragment_f(self) -> bool:
        """True if this node belongs to Fragment F."""
        return self.properties.in_fragment_f

    def with_output_schema(self, schema: Schema) -> PipelineNode:
        return attr.evolve(self, output_schema=schema)

    def with_input_schema(self, schema: Schema) -> PipelineNode:
        return attr.evolve(self, input_schema=schema)

    def with_cost(self, cost: CostEstimate) -> PipelineNode:
        return attr.evolve(self, cost_estimate=cost)

    def with_quality_constraints(self, qcs: Sequence[QualityConstraint]) -> PipelineNode:
        return attr.evolve(self, quality_constraints=tuple(qcs))

    def __repr__(self) -> str:
        frag = "F" if self.in_fragment_f else "~F"
        out_cols = len(self.output_schema.columns)
        return f"PipelineNode({self.node_id!r}, op={self.operator.value}, cols={out_cols}, [{frag}])"

    def __str__(self) -> str:
        return self.node_id

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "operator": self.operator.value,
        }
        if self.query_text:
            d["query_text"] = self.query_text
        if self.input_schema.columns:
            d["input_schema"] = self.input_schema.to_dict()
        if self.output_schema.columns:
            d["output_schema"] = self.output_schema.to_dict()
        if self.quality_constraints:
            d["quality_constraints"] = [qc.to_dict() for qc in self.quality_constraints]
        d["availability_contract"] = self.availability_contract.to_dict()
        d["cost_estimate"] = self.cost_estimate.to_dict()
        d["properties"] = self.properties.to_dict()
        meta = self.metadata.to_dict()
        if meta:
            d["metadata"] = meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineNode:
        qcs = tuple(
            QualityConstraint.from_dict(qc)
            for qc in d.get("quality_constraints", [])
        )
        return cls(
            node_id=d["node_id"],
            operator=SQLOperator(d.get("operator", "TRANSFORM")),
            query_text=d.get("query_text", ""),
            input_schema=Schema.from_dict(d["input_schema"]) if "input_schema" in d else Schema.empty(),
            output_schema=Schema.from_dict(d["output_schema"]) if "output_schema" in d else Schema.empty(),
            quality_constraints=qcs,
            availability_contract=AvailabilityContract.from_dict(d.get("availability_contract", {})),
            cost_estimate=CostEstimate.from_dict(d.get("cost_estimate", {})),
            properties=OperatorProperties.from_dict(d["properties"]) if "properties" in d else None,
            metadata=NodeMetadata.from_dict(d.get("metadata", {})),
        )


# =====================================================================
# Pipeline Edge
# =====================================================================

@attr.s(frozen=True, slots=True, repr=False, hash=True)
class PipelineEdge:
    """A directed dependency between two pipeline nodes."""

    source: str = attr.ib(validator=v.instance_of(str))
    target: str = attr.ib(validator=v.instance_of(str))
    column_mapping: dict[str, str] = attr.ib(factory=dict, hash=False)
    edge_type: EdgeType = attr.ib(default=EdgeType.DATA_FLOW, validator=v.instance_of(EdgeType))
    label: str = attr.ib(default="", validator=v.instance_of(str))

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)

    def __repr__(self) -> str:
        et = self.edge_type.value
        lbl = f" ({self.label})" if self.label else ""
        return f"PipelineEdge({self.source!r} -> {self.target!r} [{et}]{lbl})"

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
        }
        if self.column_mapping:
            d["column_mapping"] = self.column_mapping
        if self.label:
            d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineEdge:
        return cls(
            source=d["source"],
            target=d["target"],
            column_mapping=d.get("column_mapping", {}),
            edge_type=EdgeType(d.get("edge_type", "data_flow")),
            label=d.get("label", ""),
        )


# =====================================================================
# Pipeline Graph
# =====================================================================

class PipelineGraph:
    """Directed acyclic graph of pipeline transformations.

    Wraps ``networkx.DiGraph`` with type-safe access, schema validation
    on edges, topological traversal, impact analysis, sub-graph extraction,
    deep cloning and merging.
    """

    __slots__ = ("_g", "_nodes", "_edges", "_name", "_version", "_metadata")

    def __init__(
        self,
        name: str = "",
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._g = nx.DiGraph()
        self._nodes: dict[str, PipelineNode] = {}
        self._edges: dict[tuple[str, str], PipelineEdge] = {}
        self._name = name
        self._version = version
        self._metadata: dict[str, Any] = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    # ── Node operations ──

    def add_node(self, node: PipelineNode) -> None:
        """Add a node (or replace an existing one with the same id)."""
        self._nodes[node.node_id] = node
        self._g.add_node(node.node_id)

    def remove_node(self, node_id: str) -> PipelineNode:
        """Remove and return a node, along with its incident edges."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        node = self._nodes.pop(node_id)
        # Remove associated edges
        edges_to_remove = [
            k for k in self._edges if k[0] == node_id or k[1] == node_id
        ]
        for k in edges_to_remove:
            del self._edges[k]
        self._g.remove_node(node_id)
        return node

    def get_node(self, node_id: str) -> PipelineNode:
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> dict[str, PipelineNode]:
        return dict(self._nodes)

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ── Edge operations ──

    def add_edge(self, edge: PipelineEdge) -> None:
        """Add an edge between two existing nodes."""
        if edge.source not in self._nodes:
            raise NodeNotFoundError(edge.source)
        if edge.target not in self._nodes:
            raise NodeNotFoundError(edge.target)
        self._edges[(edge.source, edge.target)] = edge
        self._g.add_edge(edge.source, edge.target)

    def remove_edge(self, source: str, target: str) -> PipelineEdge:
        key = (source, target)
        if key not in self._edges:
            raise EdgeNotFoundError(source, target)
        edge = self._edges.pop(key)
        self._g.remove_edge(source, target)
        return edge

    def get_edge(self, source: str, target: str) -> PipelineEdge:
        key = (source, target)
        if key not in self._edges:
            raise EdgeNotFoundError(source, target)
        return self._edges[key]

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    @property
    def edges(self) -> dict[tuple[str, str], PipelineEdge]:
        return dict(self._edges)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ── Neighbour access ──

    def predecessors(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return list(self._g.predecessors(node_id))

    def successors(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return list(self._g.successors(node_id))

    def in_degree(self, node_id: str) -> int:
        return self._g.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        return self._g.out_degree(node_id)

    def ancestors(self, node_id: str) -> set[str]:
        """All transitive predecessors."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return nx.ancestors(self._g, node_id)

    def descendants(self, node_id: str) -> set[str]:
        """All transitive successors."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return nx.descendants(self._g, node_id)

    def sources(self) -> list[str]:
        """Nodes with no incoming edges (data sources)."""
        return [n for n in self._nodes if self._g.in_degree(n) == 0]

    def sinks(self) -> list[str]:
        """Nodes with no outgoing edges (terminal outputs)."""
        return [n for n in self._nodes if self._g.out_degree(n) == 0]

    # ── Topological operations ──

    def topological_sort(self) -> list[str]:
        """Topological ordering.  Raises CycleDetectedError on cycles."""
        try:
            return list(nx.topological_sort(self._g))
        except nx.NetworkXUnfeasible:
            cycle = self._find_cycle()
            raise CycleDetectedError(cycle)

    def reverse_topological_sort(self) -> list[str]:
        return list(reversed(self.topological_sort()))

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def _find_cycle(self) -> list[str]:
        try:
            cycle = nx.find_cycle(self._g, orientation="original")
            return [u for u, _, _ in cycle]
        except nx.NetworkXNoCycle:
            return []

    def detect_cycles(self) -> list[list[str]]:
        """Return all simple cycles in the graph."""
        return list(nx.simple_cycles(self._g))

    # ── Path finding ──

    def find_path(self, source: str, target: str) -> list[str] | None:
        """Shortest path from source to target, or None."""
        try:
            return list(nx.shortest_path(self._g, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def all_paths(self, source: str, target: str) -> list[list[str]]:
        """All simple paths from source to target."""
        try:
            return list(nx.all_simple_paths(self._g, source, target))
        except nx.NodeNotFound:
            return []

    def critical_path(self) -> list[str]:
        """Longest path in the DAG by node cost (critical path).

        Returns the sequence of node ids on the most expensive path.
        """
        if not self.is_dag():
            raise GraphError("Critical path requires a DAG", code=ErrorCode.GRAPH_CYCLE_DETECTED)
        topo = self.topological_sort()
        dist: dict[str, float] = {n: 0.0 for n in topo}
        prev: dict[str, str | None] = {n: None for n in topo}
        for node_id in topo:
            node_cost = self._nodes[node_id].cost_estimate.total_weighted_cost
            for succ in self._g.successors(node_id):
                new_dist = dist[node_id] + node_cost
                if new_dist > dist[succ]:
                    dist[succ] = new_dist
                    prev[succ] = node_id
        # Find the terminal with maximum distance
        if not topo:
            return []
        end_node = max(topo, key=lambda n: dist[n])
        path: list[str] = []
        cur: str | None = end_node
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    # ── Connected components ──

    def connected_components(self) -> list[set[str]]:
        """Weakly connected components."""
        return [set(c) for c in nx.weakly_connected_components(self._g)]

    def is_connected(self) -> bool:
        return nx.is_weakly_connected(self._g) if self.node_count > 0 else True

    # ── Sub-graph extraction ──

    def subgraph(self, node_ids: Iterable[str]) -> PipelineGraph:
        """Extract a sub-graph containing only the specified nodes and
        edges between them.
        """
        nids = set(node_ids)
        sub = PipelineGraph(
            name=f"{self._name}_sub",
            version=self._version,
            metadata=dict(self._metadata),
        )
        for nid in nids:
            if nid in self._nodes:
                sub.add_node(self._nodes[nid])
        for (s, t), edge in self._edges.items():
            if s in nids and t in nids:
                sub.add_edge(edge)
        return sub

    def upstream_subgraph(self, node_id: str) -> PipelineGraph:
        """Sub-graph of all ancestors of *node_id* plus the node itself."""
        anc = self.ancestors(node_id) | {node_id}
        return self.subgraph(anc)

    def downstream_subgraph(self, node_id: str) -> PipelineGraph:
        """Sub-graph of all descendants of *node_id* plus the node itself."""
        desc = self.descendants(node_id) | {node_id}
        return self.subgraph(desc)

    # ── Schema validation ──

    def validate_edge_schemas(self) -> list[str]:
        """Check that every edge's source output schema is compatible
        with the target input schema.  Returns list of issues.
        """
        issues: list[str] = []
        for (src_id, tgt_id), edge in self._edges.items():
            src = self._nodes[src_id]
            tgt = self._nodes[tgt_id]
            if not src.output_schema.columns or not tgt.input_schema.columns:
                continue  # skip if schemas are not yet populated
            if edge.column_mapping:
                # Validate mapped columns exist in source output
                for src_col in edge.column_mapping:
                    if src_col not in src.output_schema:
                        issues.append(
                            f"Edge {src_id}->{tgt_id}: source column "
                            f"'{src_col}' not in output schema"
                        )
                # Validate mapped targets exist in target input
                for tgt_col in edge.column_mapping.values():
                    if tgt_col not in tgt.input_schema:
                        issues.append(
                            f"Edge {src_id}->{tgt_id}: target column "
                            f"'{tgt_col}' not in input schema"
                        )
            else:
                # Without explicit mapping, output columns should be a subset
                # of input columns
                mismatched = src.output_schema.compatible_with(tgt.input_schema)
                if mismatched:
                    issues.append(
                        f"Edge {src_id}->{tgt_id}: schema mismatch on "
                        f"columns {mismatched}"
                    )
        return issues

    def validate(self) -> list[str]:
        """Run all structural validations. Returns list of issues."""
        issues: list[str] = []
        # Check for cycles
        if not self.is_dag():
            cycles = self.detect_cycles()
            for cycle in cycles[:3]:
                issues.append(f"Cycle detected: {' -> '.join(cycle)}")
        # Check edge schemas
        issues.extend(self.validate_edge_schemas())
        # Check for disconnected nodes
        for nid in self._nodes:
            if self._g.in_degree(nid) == 0 and self._g.out_degree(nid) == 0 and self.node_count > 1:
                issues.append(f"Isolated node: {nid}")
        return issues

    # ── Cloning and merging ──

    def clone(self) -> PipelineGraph:
        """Deep copy of the entire graph."""
        new_graph = PipelineGraph(
            name=self._name,
            version=self._version,
            metadata=copy.deepcopy(self._metadata),
        )
        for node in self._nodes.values():
            new_graph.add_node(node)  # nodes are frozen, no need to copy
        for edge in self._edges.values():
            new_graph.add_edge(edge)
        return new_graph

    def merge(
        self,
        other: PipelineGraph,
        prefix: str = "",
        on_conflict: str = "raise",
    ) -> PipelineGraph:
        """Merge *other* into a new graph.

        Parameters
        ----------
        prefix:
            Optional prefix for other's node ids to avoid conflicts.
        on_conflict:
            ``"raise"`` (default), ``"keep_self"``, or ``"keep_other"``.
        """
        result = self.clone()
        for nid, node in other._nodes.items():
            new_id = f"{prefix}{nid}" if prefix else nid
            if new_id in result._nodes:
                if on_conflict == "raise":
                    raise GraphMergeConflictError([new_id])
                elif on_conflict == "keep_self":
                    continue
                # keep_other: fall through and overwrite
            if prefix:
                node = attr.evolve(node, node_id=new_id)
            result.add_node(node)
        for (s, t), edge in other._edges.items():
            new_s = f"{prefix}{s}" if prefix else s
            new_t = f"{prefix}{t}" if prefix else t
            if result.has_edge(new_s, new_t) and on_conflict == "keep_self":
                continue
            new_edge = PipelineEdge(
                source=new_s,
                target=new_t,
                column_mapping=edge.column_mapping,
                edge_type=edge.edge_type,
                label=edge.label,
            )
            result.add_edge(new_edge)
        return result

    # ── Fragment F classification ──

    def fragment_f_nodes(self) -> set[str]:
        """Return all node ids that belong to Fragment F."""
        return {nid for nid, node in self._nodes.items() if node.in_fragment_f}

    def non_fragment_f_nodes(self) -> set[str]:
        """Return all node ids outside Fragment F."""
        return {nid for nid, node in self._nodes.items() if not node.in_fragment_f}

    def is_in_fragment_f(self) -> bool:
        """True if the entire pipeline is in Fragment F."""
        return all(node.in_fragment_f for node in self._nodes.values())

    # ── Metrics ──

    def total_cost(self) -> CostEstimate:
        """Sum of all node cost estimates."""
        total = CostEstimate.zero()
        for node in self._nodes.values():
            total = total + node.cost_estimate
        return total

    def depth(self) -> int:
        """Length of the longest path (number of edges)."""
        if not self.is_dag() or not self._nodes:
            return 0
        return nx.dag_longest_path_length(self._g)

    def width(self) -> int:
        """Maximum number of nodes at the same topological level."""
        if not self._nodes:
            return 0
        topo = self.topological_sort()
        levels: dict[str, int] = {}
        for nid in topo:
            preds = list(self._g.predecessors(nid))
            if not preds:
                levels[nid] = 0
            else:
                levels[nid] = max(levels[p] for p in preds) + 1
        if not levels:
            return 0
        from collections import Counter as Ctr
        level_counts = Ctr(levels.values())
        return max(level_counts.values())

    # ── Serialisation ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "metadata": self._metadata,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineGraph:
        g = cls(
            name=d.get("name", ""),
            version=d.get("version", "1.0"),
            metadata=d.get("metadata", {}),
        )
        for nd in d.get("nodes", []):
            g.add_node(PipelineNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(PipelineEdge.from_dict(ed))
        return g

    # ── Representation ──

    def __repr__(self) -> str:
        return (
            f"PipelineGraph(name={self._name!r}, "
            f"nodes={self.node_count}, edges={self.edge_count})"
        )

    def __str__(self) -> str:
        lines = [repr(self)]
        for nid in self.topological_sort() if self.is_dag() else sorted(self._nodes):
            preds = self.predecessors(nid)
            pred_str = ", ".join(preds) if preds else "(source)"
            lines.append(f"  {nid} <- {pred_str}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.node_count

    def __contains__(self, node_id: str) -> bool:
        return self.has_node(node_id)

    def __iter__(self) -> Iterator[str]:
        yield from self.topological_sort() if self.is_dag() else sorted(self._nodes)

    # ── NetworkX interop ──

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Access the underlying networkx graph (read-only use recommended)."""
        return self._g

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the underlying networkx graph with node/edge data."""
        g = self._g.copy()
        for nid, node in self._nodes.items():
            g.nodes[nid]["pipeline_node"] = node
        for key, edge in self._edges.items():
            g.edges[key]["pipeline_edge"] = edge
        return g

    # ── Bulk operations ──

    def replace_node(self, node_id: str, new_node: PipelineNode) -> None:
        """Replace a node in-place, preserving edges."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        if new_node.node_id != node_id:
            # Need to rename edges
            edges_to_fix: list[tuple[PipelineEdge, bool]] = []
            for (s, t), edge in list(self._edges.items()):
                if s == node_id:
                    edges_to_fix.append((edge, True))  # source
                elif t == node_id:
                    edges_to_fix.append((edge, False))  # target
            # Remove old
            self._g.remove_node(node_id)
            del self._nodes[node_id]
            for edge, is_source in edges_to_fix:
                self._edges.pop((edge.source, edge.target), None)
            # Add new
            self._nodes[new_node.node_id] = new_node
            self._g.add_node(new_node.node_id)
            for edge, is_source in edges_to_fix:
                new_s = new_node.node_id if is_source else edge.source
                new_t = new_node.node_id if not is_source else edge.target
                new_edge = PipelineEdge(
                    source=new_s, target=new_t,
                    column_mapping=edge.column_mapping,
                    edge_type=edge.edge_type, label=edge.label,
                )
                self._edges[(new_s, new_t)] = new_edge
                self._g.add_edge(new_s, new_t)
        else:
            self._nodes[node_id] = new_node

    def update_node_schema(
        self, node_id: str, output_schema: Schema,
    ) -> None:
        """Update a node's output schema."""
        old = self.get_node(node_id)
        new_node = old.with_output_schema(output_schema)
        self._nodes[node_id] = new_node

    def update_node_cost(self, node_id: str, cost: CostEstimate) -> None:
        """Update a node's cost estimate."""
        old = self.get_node(node_id)
        self._nodes[node_id] = old.with_cost(cost)

    def filter_nodes(self, predicate: Any) -> PipelineGraph:
        """Extract a sub-graph containing only nodes matching a predicate.

        ``predicate`` is a callable ``(PipelineNode) -> bool``.
        """
        matching = {nid for nid, n in self._nodes.items() if predicate(n)}
        return self.subgraph(matching)

    def nodes_by_operator(self, operator: SQLOperator) -> list[str]:
        """Return all node ids with a specific operator type."""
        return [nid for nid, n in self._nodes.items() if n.operator == operator]

    def leaf_nodes(self) -> list[str]:
        """Nodes with no outgoing edges — equivalent to sinks()."""
        return self.sinks()

    def root_nodes(self) -> list[str]:
        """Nodes with no incoming edges — equivalent to sources()."""
        return self.sources()

    def intermediate_nodes(self) -> list[str]:
        """Nodes with both incoming and outgoing edges."""
        return [
            nid for nid in self._nodes
            if self._g.in_degree(nid) > 0 and self._g.out_degree(nid) > 0
        ]

    def edge_list(self) -> list[tuple[str, str]]:
        """Return all edges as (source, target) tuples."""
        return list(self._edges.keys())

    # ── Quality-focused queries ──

    def nodes_with_quality_constraints(self) -> dict[str, list[QualityConstraint]]:
        """Return nodes that have quality constraints attached."""
        result: dict[str, list[QualityConstraint]] = {}
        for nid, node in self._nodes.items():
            if node.quality_constraints:
                result[nid] = list(node.quality_constraints)
        return result

    def all_quality_constraints(self) -> list[tuple[str, QualityConstraint]]:
        """Return all quality constraints as (node_id, constraint) pairs."""
        result: list[tuple[str, QualityConstraint]] = []
        for nid, node in self._nodes.items():
            for qc in node.quality_constraints:
                result.append((nid, qc))
        return result

    def nodes_with_availability_sla(
        self, min_sla: float = 99.0,
    ) -> list[str]:
        """Return nodes whose availability SLA is at or above the threshold."""
        return [
            nid for nid, n in self._nodes.items()
            if n.availability_contract.sla_percentage >= min_sla
        ]

    # ── Statistics ──

    def schema_coverage(self) -> float:
        """Fraction of nodes that have non-empty output schemas."""
        if not self._nodes:
            return 0.0
        count = sum(
            1 for n in self._nodes.values()
            if n.output_schema.columns
        )
        return count / len(self._nodes)

    def cost_distribution(self) -> dict[str, float]:
        """Return a mapping of node_id -> weighted cost."""
        return {
            nid: n.cost_estimate.total_weighted_cost
            for nid, n in self._nodes.items()
        }

    def most_expensive_nodes(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-K most expensive nodes."""
        costs = self.cost_distribution()
        sorted_costs = sorted(costs.items(), key=lambda x: -x[1])
        return sorted_costs[:top_k]
