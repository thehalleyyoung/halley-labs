"""
taintflow.dag.pidag -- Core PI-DAG (Pipeline Information DAG) data structure.

The :class:`PIDAG` is the central data structure of the TaintFlow analysis
engine.  It stores the full graph of pipeline operations (nodes) and the
data-flow / dependency relationships (edges) between them, providing
efficient queries for topological traversal, column-level provenance,
leakage-path detection, and subgraph extraction.
"""

from __future__ import annotations

import copy
import math
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
)

from taintflow.core.types import (
    ColumnSchema,
    EdgeKind,
    NodeKind,
    OpType,
    Origin,
    ProvenanceInfo,
    Severity,
    ShapeMetadata,
)
from taintflow.core.errors import (
    CycleDetectedError,
    DAGConstructionError,
    MissingNodeError,
)
from taintflow.dag.node import (
    PipelineNode,
    DataSourceNode,
    PartitionNode,
    TransformNode,
    FitNode,
    PredictNode,
    PandasOpNode,
    AggregationNode,
    FeatureEngineeringNode,
    SelectionNode,
    CustomNode,
    SinkNode,
)
from taintflow.dag.edge import (
    PipelineEdge,
    EdgeSet,
    estimate_edge_capacity,
)


# ===================================================================
#  Pipeline stage grouping
# ===================================================================


@dataclass
class PipelineStage:
    """A logical stage in the pipeline, grouping related nodes."""

    stage_id: str
    stage_type: str
    description: str
    node_ids: list[str] = field(default_factory=list)


# ===================================================================
#  PIDAG
# ===================================================================


class PIDAG:
    """Pipeline Information DAG (PI-DAG).

    Central data structure for the TaintFlow analysis.  Stores nodes
    (pipeline operations) and edges (data-flow / dependency relationships)
    with efficient indexed access.

    Invariants enforced by :meth:`validate`:
      * The graph is acyclic.
      * Every edge endpoint references an existing node.
      * Source nodes have no incoming data-flow edges.
      * Sink nodes have no outgoing data-flow edges.
      * Schemas are consistent along edges (column names match).
    """

    __slots__ = ("_nodes", "_edges", "_frozen", "_topo_cache", "_meta")

    def __init__(self) -> None:
        self._nodes: dict[str, PipelineNode] = {}
        self._edges: EdgeSet = EdgeSet()
        self._frozen: bool = False
        self._topo_cache: list[str] | None = None
        self._meta: dict[str, Any] = {}

    # -- properties ----------------------------------------------------------

    @property
    def nodes(self) -> dict[str, PipelineNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> EdgeSet:
        return self._edges

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    # -- node subsets --------------------------------------------------------

    @property
    def sources(self) -> list[PipelineNode]:
        """Nodes with no incoming data-flow edges."""
        incoming = self._edges.target_ids()
        return [n for nid, n in self._nodes.items() if nid not in incoming]

    @property
    def sinks(self) -> list[PipelineNode]:
        """Nodes with no outgoing data-flow edges."""
        outgoing = self._edges.source_ids()
        return [n for nid, n in self._nodes.items() if nid not in outgoing]

    @property
    def partition_nodes(self) -> list[PipelineNode]:
        """Nodes that split data into train/test partitions."""
        return [n for n in self._nodes.values() if n.node_kind == NodeKind.SPLIT]

    @property
    def estimator_nodes(self) -> list[PipelineNode]:
        """Nodes involving sklearn estimator operations."""
        return [n for n in self._nodes.values() if n.is_estimator]

    @property
    def data_source_nodes(self) -> list[PipelineNode]:
        return [n for n in self._nodes.values() if n.node_kind == NodeKind.DATA_SOURCE]

    @property
    def sink_nodes(self) -> list[PipelineNode]:
        return [n for n in self._nodes.values() if n.node_kind == NodeKind.SINK]

    @property
    def transform_nodes(self) -> list[PipelineNode]:
        return [
            n for n in self._nodes.values()
            if n.node_kind in {NodeKind.TRANSFORM, NodeKind.ESTIMATOR_TRANSFORM}
        ]

    # -- mutation (guarded by frozen flag) -----------------------------------

    def _check_mutable(self) -> None:
        if self._frozen:
            raise DAGConstructionError("Cannot modify a frozen PIDAG")

    def add_node(self, node: PipelineNode) -> None:
        """Add a node to the DAG."""
        self._check_mutable()
        if node.node_id in self._nodes:
            raise DAGConstructionError(f"Duplicate node ID: {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._topo_cache = None

    def add_edge(self, edge: PipelineEdge) -> None:
        """Add an edge to the DAG."""
        self._check_mutable()
        if edge.source_id not in self._nodes:
            raise MissingNodeError(f"Source node not found: {edge.source_id!r}")
        if edge.target_id not in self._nodes:
            raise MissingNodeError(f"Target node not found: {edge.target_id!r}")
        self._edges.add(edge)
        self._topo_cache = None

    def remove_node(self, node_id: str) -> PipelineNode:
        """Remove a node and all its incident edges.  Returns the removed node."""
        self._check_mutable()
        if node_id not in self._nodes:
            raise MissingNodeError(f"Node not found: {node_id!r}")
        node = self._nodes.pop(node_id)
        self._edges.remove_by_node(node_id)
        self._topo_cache = None
        return node

    def remove_edge(self, edge: PipelineEdge) -> None:
        """Remove a specific edge."""
        self._check_mutable()
        self._edges.remove(edge)
        self._topo_cache = None

    def remove_edge_by_key(self, source_id: str, target_id: str, kind: EdgeKind = EdgeKind.DATA_FLOW) -> None:
        """Remove an edge identified by source, target, and kind."""
        self._check_mutable()
        edge = self._edges.get(source_id, target_id, kind)
        if edge is None:
            raise KeyError(f"Edge not found: ({source_id!r}, {target_id!r}, {kind.name})")
        self._edges.remove(edge)
        self._topo_cache = None

    def get_node(self, node_id: str) -> PipelineNode:
        """Get a node by ID, raising MissingNodeError if absent."""
        if node_id not in self._nodes:
            raise MissingNodeError(f"Node not found: {node_id!r}")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source_id: str, target_id: str, kind: EdgeKind | None = None) -> bool:
        return self._edges.contains(source_id, target_id, kind)

    # -- graph traversal queries ---------------------------------------------

    def predecessors(self, node_id: str) -> list[str]:
        """Direct predecessors of a node (via any edge kind)."""
        return [e.source_id for e in self._edges.by_target(node_id)]

    def successors(self, node_id: str) -> list[str]:
        """Direct successors of a node (via any edge kind)."""
        return [e.target_id for e in self._edges.by_source(node_id)]

    def in_edges(self, node_id: str) -> list[PipelineEdge]:
        """All edges targeting *node_id*."""
        return self._edges.by_target(node_id)

    def out_edges(self, node_id: str) -> list[PipelineEdge]:
        """All edges originating from *node_id*."""
        return self._edges.by_source(node_id)

    def in_degree(self, node_id: str) -> int:
        return len(self._edges.by_target(node_id))

    def out_degree(self, node_id: str) -> int:
        return len(self._edges.by_source(node_id))

    def ancestors(self, node_id: str) -> set[str]:
        """All transitive predecessors of *node_id*."""
        visited: set[str] = set()
        stack = list(self.predecessors(node_id))
        while stack:
            nid = stack.pop()
            if nid not in visited:
                visited.add(nid)
                stack.extend(self.predecessors(nid))
        return visited

    def descendants(self, node_id: str) -> set[str]:
        """All transitive successors of *node_id*."""
        visited: set[str] = set()
        stack = list(self.successors(node_id))
        while stack:
            nid = stack.pop()
            if nid not in visited:
                visited.add(nid)
                stack.extend(self.successors(nid))
        return visited

    # -- topological ordering ------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (Kahn's algorithm).

        Raises :class:`CycleDetectedError` if the graph has a cycle.
        """
        if self._topo_cache is not None:
            return list(self._topo_cache)

        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            if e.target_id in in_degree:
                in_degree[e.target_id] += 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for e in self._edges.by_source(nid):
                in_degree[e.target_id] -= 1
                if in_degree[e.target_id] == 0:
                    queue.append(e.target_id)

        if len(result) != len(self._nodes):
            remaining = set(self._nodes) - set(result)
            raise CycleDetectedError(
                f"Cycle detected in PI-DAG involving nodes: {remaining}"
            )

        self._topo_cache = result
        return list(result)

    def reverse_postorder(self) -> list[str]:
        """Return node IDs in reverse postorder (DFS-based).

        For a DAG this is equivalent to topological order but uses DFS.
        """
        visited: set[str] = set()
        postorder: list[str] = []

        def _dfs(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for succ in self.successors(nid):
                _dfs(succ)
            postorder.append(nid)

        for nid in self._nodes:
            if nid not in visited:
                _dfs(nid)

        return list(reversed(postorder))

    # -- column queries ------------------------------------------------------

    def get_columns_at_node(self, node_id: str) -> set[str]:
        """Columns flowing through *node_id* (union of in-edge and out-edge columns)."""
        node = self.get_node(node_id)
        cols: set[str] = set()
        cols.update(c.name for c in node.input_schema)
        cols.update(c.name for c in node.output_schema)
        for e in self._edges.by_target(node_id):
            cols.update(e.columns)
        for e in self._edges.by_source(node_id):
            cols.update(e.columns)
        return cols

    def get_provenance_at_node(self, node_id: str, column: str) -> ProvenanceInfo:
        """Get provenance for a specific column at a specific node."""
        node = self.get_node(node_id)
        if column in node.provenance:
            return node.provenance[column]
        in_provenances: list[ProvenanceInfo] = []
        for e in self._edges.by_target(node_id):
            if column in e.columns:
                src_node = self._nodes.get(e.source_id)
                if src_node and column in src_node.provenance:
                    in_provenances.append(src_node.provenance[column])
        if not in_provenances:
            return ProvenanceInfo(test_fraction=0.0, origin_set=frozenset({Origin.TRAIN}))
        result = in_provenances[0]
        for p in in_provenances[1:]:
            result = result.merge(p)
        return result

    # -- subgraph extraction -------------------------------------------------

    def get_subdag(self, node_ids: set[str]) -> "PIDAG":
        """Extract a sub-DAG containing only the specified nodes and their
        interconnecting edges."""
        sub = PIDAG()
        for nid in node_ids:
            if nid in self._nodes:
                sub._nodes[nid] = copy.deepcopy(self._nodes[nid])
        sub._edges = self._edges.restrict_to_nodes(node_ids)
        return sub

    def get_slice(self, source_id: str, sink_id: str) -> "PIDAG":
        """Extract the sub-DAG on all paths from *source_id* to *sink_id*."""
        desc_of_source = self.descendants(source_id) | {source_id}
        anc_of_sink = self.ancestors(sink_id) | {sink_id}
        slice_nodes = desc_of_source & anc_of_sink
        return self.get_subdag(slice_nodes)

    def get_column_subdag(self, column: str) -> "PIDAG":
        """Extract the sub-DAG of nodes/edges relevant to *column*."""
        relevant_nodes: set[str] = set()
        for nid, node in self._nodes.items():
            node_cols = set(c.name for c in node.input_schema) | set(c.name for c in node.output_schema)
            if column in node_cols:
                relevant_nodes.add(nid)
        for e in self._edges:
            if column in e.columns:
                relevant_nodes.add(e.source_id)
                relevant_nodes.add(e.target_id)
        return self.get_subdag(relevant_nodes)

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Check all DAG invariants, returning a list of error messages."""
        errors: list[str] = []

        # 1. Check for cycles
        try:
            self.topological_order()
        except CycleDetectedError as exc:
            errors.append(f"Cycle detected: {exc}")

        # 2. Check edge endpoint existence
        for e in self._edges:
            if e.source_id not in self._nodes:
                errors.append(f"Edge references missing source node: {e.source_id!r}")
            if e.target_id not in self._nodes:
                errors.append(f"Edge references missing target node: {e.target_id!r}")

        # 3. Validate individual edges
        for e in self._edges:
            for err in e.validate():
                errors.append(f"Edge({e.source_id}->{e.target_id}): {err}")

        # 4. Validate individual nodes
        for nid, node in self._nodes.items():
            for err in node.validate():
                errors.append(f"Node({nid}): {err}")

        # 5. Check schema consistency along data-flow edges
        for e in self._edges.data_flow_edges():
            if e.source_id in self._nodes and e.target_id in self._nodes:
                src = self._nodes[e.source_id]
                tgt = self._nodes[e.target_id]
                src_out_cols = {c.name for c in src.output_schema}
                for col in e.columns:
                    if src_out_cols and col not in src_out_cols:
                        errors.append(
                            f"Edge({e.source_id}->{e.target_id}): column {col!r} "
                            f"not in source output schema"
                        )

        # 6. Source nodes should have no incoming data-flow edges
        for n in self.data_source_nodes:
            in_data = [e for e in self._edges.by_target(n.node_id) if e.is_data_flow]
            if in_data:
                errors.append(
                    f"DataSourceNode {n.node_id!r} has {len(in_data)} incoming data-flow edges"
                )

        return errors

    # -- capacity computation ------------------------------------------------

    def compute_edge_capacities(
        self,
        capacity_catalog: Mapping[str, float] | None = None,
    ) -> None:
        """Compute/update capacity bounds for all edges.

        If *capacity_catalog* is given, it maps ``OpType.value`` to a
        manually specified capacity override.
        """
        self._check_mutable()
        catalog = dict(capacity_catalog) if capacity_catalog else {}
        updated_edges: list[PipelineEdge] = []

        for e in list(self._edges):
            src = self._nodes.get(e.source_id)
            tgt = self._nodes.get(e.target_id)
            if src is None or tgt is None:
                continue

            override_key = f"{src.op_type.value}->{tgt.op_type.value}"
            if override_key in catalog:
                cap = catalog[override_key]
            elif src.op_type.value in catalog:
                cap = catalog[src.op_type.value]
            else:
                cap = estimate_edge_capacity(
                    e,
                    source_schema=src.output_schema,
                    target_schema=tgt.input_schema,
                    n_rows=max(src.shape.n_rows, 1),
                    n_features=len(src.output_schema),
                    n_samples=src.shape.n_rows,
                    estimator_class=src.metadata.get("estimator_class", ""),
                )
            updated_edges.append(e.with_capacity(cap))

        for ue in updated_edges:
            self._edges.update_edge(ue)

    # -- leakage detection ---------------------------------------------------

    def find_leakage_paths(self) -> list[list[str]]:
        """Find all paths from test-origin sources to training sinks.

        A leakage path is any path where data with test provenance flows
        into a node that is used during training (e.g., a FitNode whose
        training data has test-tainted columns).
        """
        test_sources: list[str] = []
        for nid, node in self._nodes.items():
            if isinstance(node, DataSourceNode) and node.origin == Origin.TEST:
                test_sources.append(nid)
            elif node.max_test_fraction() > 0.0 and self.in_degree(nid) == 0:
                test_sources.append(nid)
            for e in self._edges.by_target(nid):
                if e.provenance_fraction > 0.0:
                    if nid not in test_sources:
                        pass

        train_sinks: set[str] = set()
        for nid, node in self._nodes.items():
            if node.has_fit:
                train_sinks.add(nid)
            if isinstance(node, TransformNode) and node.is_fitted:
                for pred_id in self.predecessors(nid):
                    pred = self._nodes.get(pred_id)
                    if pred and pred.has_fit:
                        train_sinks.add(pred_id)

        if not test_sources or not train_sinks:
            return []

        paths: list[list[str]] = []
        for src in test_sources:
            for sink in train_sinks:
                for path in self._all_simple_paths(src, sink):
                    paths.append(path)
        return paths

    def _all_simple_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 100,
    ) -> list[list[str]]:
        """Enumerate all simple (non-repeating) paths from source to target."""
        if source not in self._nodes or target not in self._nodes:
            return []
        paths: list[list[str]] = []
        stack: list[Tuple[str, list[str], set[str]]] = [
            (source, [source], {source})
        ]
        while stack:
            current, path, visited = stack.pop()
            if current == target and len(path) > 1:
                paths.append(list(path))
                continue
            if len(path) > max_depth:
                continue
            for succ_id in self.successors(current):
                if succ_id not in visited:
                    new_visited = visited | {succ_id}
                    stack.append((succ_id, path + [succ_id], new_visited))
        return paths

    # -- DAG merging ---------------------------------------------------------

    def merge(self, other: "PIDAG") -> None:
        """Merge *other* DAG into this DAG.

        Nodes with the same ID are kept from *self*; new nodes from
        *other* are added.  Edges are unioned.
        """
        self._check_mutable()
        for nid, node in other._nodes.items():
            if nid not in self._nodes:
                self._nodes[nid] = copy.deepcopy(node)
        for e in other._edges:
            if e.source_id in self._nodes and e.target_id in self._nodes:
                if not self._edges.has_edge(e):
                    self._edges.add(copy.deepcopy(e))
        self._topo_cache = None

    # -- statistics ----------------------------------------------------------

    @property
    def depth(self) -> int:
        """Length of the longest path in the DAG."""
        if not self._nodes:
            return 0
        topo = self.topological_order()
        dist: dict[str, int] = {nid: 0 for nid in topo}
        for nid in topo:
            for succ in self.successors(nid):
                if dist[succ] < dist[nid] + 1:
                    dist[succ] = dist[nid] + 1
        return max(dist.values()) if dist else 0

    @property
    def width(self) -> int:
        """Maximum number of nodes at any level (anti-chain width)."""
        if not self._nodes:
            return 0
        topo = self.topological_order()
        level: dict[str, int] = {nid: 0 for nid in topo}
        for nid in topo:
            for succ in self.successors(nid):
                if level[succ] < level[nid] + 1:
                    level[succ] = level[nid] + 1
        level_counts: dict[int, int] = defaultdict(int)
        for lv in level.values():
            level_counts[lv] += 1
        return max(level_counts.values()) if level_counts else 0

    def critical_path(self) -> list[str]:
        """Return the longest path through the DAG (by number of nodes)."""
        if not self._nodes:
            return []
        topo = self.topological_order()
        dist: dict[str, int] = {nid: 0 for nid in topo}
        pred_on_path: dict[str, str | None] = {nid: None for nid in topo}
        for nid in topo:
            for succ in self.successors(nid):
                if dist[succ] < dist[nid] + 1:
                    dist[succ] = dist[nid] + 1
                    pred_on_path[succ] = nid
        end_node = max(dist, key=lambda x: dist[x])
        path: list[str] = []
        current: str | None = end_node
        while current is not None:
            path.append(current)
            current = pred_on_path.get(current)
        return list(reversed(path))

    def node_levels(self) -> dict[str, int]:
        """Map each node to its level (distance from a source)."""
        topo = self.topological_order()
        level: dict[str, int] = {nid: 0 for nid in topo}
        for nid in topo:
            for succ in self.successors(nid):
                if level[succ] < level[nid] + 1:
                    level[succ] = level[nid] + 1
        return level

    # -- pipeline stages -----------------------------------------------------

    def get_pipeline_stages(self) -> list[PipelineStage]:
        """Group nodes into logical pipeline stages."""
        stages: list[PipelineStage] = []
        stage_nodes: dict[str, list[str]] = defaultdict(list)

        for nid, node in self._nodes.items():
            kind = node.node_kind
            if kind == NodeKind.DATA_SOURCE:
                stage_nodes["data_ingestion"].append(nid)
            elif kind == NodeKind.SPLIT:
                stage_nodes["data_splitting"].append(nid)
            elif kind in {NodeKind.ESTIMATOR_FIT, NodeKind.ESTIMATOR_PREDICT,
                          NodeKind.ESTIMATOR_TRANSFORM}:
                stage_nodes["model_fitting"].append(nid)
            elif kind == NodeKind.FEATURE_ENGINEERING:
                stage_nodes["feature_engineering"].append(nid)
            elif kind == NodeKind.EVALUATION:
                stage_nodes["evaluation"].append(nid)
            elif kind == NodeKind.SINK:
                stage_nodes["output"].append(nid)
            elif kind == NodeKind.MERGE:
                stage_nodes["data_merging"].append(nid)
            else:
                stage_nodes["preprocessing"].append(nid)

        stage_descriptions = {
            "data_ingestion": "Loading raw data from files or databases",
            "data_splitting": "Partitioning data into train/test/validation sets",
            "preprocessing": "Data cleaning, imputation, and transformation",
            "feature_engineering": "Creating new features from existing data",
            "data_merging": "Combining datasets from different sources",
            "model_fitting": "Training and applying machine learning models",
            "evaluation": "Evaluating model performance",
            "output": "Saving results and model artifacts",
        }

        stage_order = [
            "data_ingestion", "data_splitting", "preprocessing",
            "feature_engineering", "data_merging", "model_fitting",
            "evaluation", "output",
        ]

        for i, stage_key in enumerate(stage_order):
            if stage_key in stage_nodes:
                stages.append(PipelineStage(
                    stage_id=f"stage_{i}",
                    stage_type=stage_key,
                    description=stage_descriptions.get(stage_key, ""),
                    node_ids=stage_nodes[stage_key],
                ))

        return stages

    # -- immutability --------------------------------------------------------

    def freeze(self) -> "PIDAG":
        """Return a frozen (immutable) copy of this DAG."""
        frozen = self.clone()
        frozen._frozen = True
        return frozen

    def clone(self) -> "PIDAG":
        """Return a deep copy of this DAG."""
        new_dag = PIDAG()
        new_dag._nodes = copy.deepcopy(self._nodes)
        new_dag._edges = self._edges.clone()
        new_dag._meta = copy.deepcopy(self._meta)
        return new_dag

    # -- iteration -----------------------------------------------------------

    def iter_nodes_topo(self) -> Iterator[PipelineNode]:
        """Iterate over nodes in topological order."""
        for nid in self.topological_order():
            yield self._nodes[nid]

    def iter_edges(self) -> Iterator[PipelineEdge]:
        """Iterate over all edges."""
        return iter(self._edges)

    def iter_data_flow_edges(self) -> Iterator[PipelineEdge]:
        """Iterate over data-flow edges only."""
        return iter(self._edges.data_flow_edges())

    def iter_paths(
        self,
        source_id: str | None = None,
        sink_id: str | None = None,
        max_paths: int = 1000,
    ) -> Iterator[list[str]]:
        """Iterate over paths, optionally between specific endpoints."""
        sources = [source_id] if source_id else [n.node_id for n in self.sources]
        sinks_list = [sink_id] if sink_id else [n.node_id for n in self.sinks]
        count = 0
        for s in sources:
            for t in sinks_list:
                if s == t:
                    continue
                for path in self._all_simple_paths(s, t, max_depth=50):
                    yield path
                    count += 1
                    if count >= max_paths:
                        return

    # -- provenance propagation ----------------------------------------------

    def propagate_provenance(self) -> None:
        """Forward-propagate provenance through the DAG.

        Starting from source nodes, propagate provenance information along
        data-flow edges, merging at confluences.
        """
        self._check_mutable()
        topo = self.topological_order()

        for nid in topo:
            node = self._nodes[nid]
            in_edges = self._edges.by_target(nid)
            if not in_edges:
                continue

            merged_provenance: dict[str, list[ProvenanceInfo]] = defaultdict(list)
            for e in in_edges:
                if not e.is_data_flow:
                    continue
                src_node = self._nodes.get(e.source_id)
                if src_node is None:
                    continue
                for col in e.columns:
                    if col in src_node.provenance:
                        merged_provenance[col].append(src_node.provenance[col])

            for col, provs in merged_provenance.items():
                if provs:
                    result = provs[0]
                    for p in provs[1:]:
                        result = result.merge(p)
                    node.provenance[col] = result

            if isinstance(node, PartitionNode):
                for col in node.output_columns:
                    if col not in node.provenance:
                        node.provenance[col] = ProvenanceInfo(
                            test_fraction=node.test_size,
                            origin_set=frozenset({Origin.TRAIN, Origin.TEST}),
                            source_id=node.node_id,
                        )

    # -- edge provenance update ----------------------------------------------

    def propagate_edge_provenance(self) -> None:
        """Update provenance_fraction on all edges from node provenance."""
        self._check_mutable()
        updated: list[PipelineEdge] = []
        for e in self._edges:
            src = self._nodes.get(e.source_id)
            if src is None:
                continue
            if not e.columns:
                updated.append(e.with_provenance(src.max_test_fraction()))
                continue
            fractions = []
            for col in e.columns:
                prov = src.provenance.get(col)
                if prov is not None:
                    fractions.append(prov.test_fraction)
            if fractions:
                avg_frac = sum(fractions) / len(fractions)
                updated.append(e.with_provenance(avg_frac))
            else:
                updated.append(e.with_provenance(0.0))
        for ue in updated:
            self._edges.update_edge(ue)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full DAG to a JSON-compatible dictionary."""
        return {
            "schema_version": "1.0.0",
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "edges": self._edges.to_list(),
            "metadata": dict(self._meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PIDAG":
        """Deserialize from a dictionary."""
        dag = cls()
        for nid, ndata in data.get("nodes", {}).items():
            node = PipelineNode.from_dict(ndata)
            dag._nodes[node.node_id] = node
        for edata in data.get("edges", []):
            edge = PipelineEdge.from_dict(edata)
            if edge.source_id in dag._nodes and edge.target_id in dag._nodes:
                dag._edges.add(edge)
        dag._meta = dict(data.get("metadata", {}))
        return dag

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PIDAG(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"sources={len(self.sources)}, sinks={len(self.sinks)})"
        )

    def __str__(self) -> str:
        lines = [f"PIDAG with {self.n_nodes} nodes and {self.n_edges} edges"]
        if self._nodes:
            lines.append(f"  Depth: {self.depth}, Width: {self.width}")
            lines.append(f"  Sources: {len(self.sources)}")
            lines.append(f"  Sinks: {len(self.sinks)}")
            lines.append(f"  Partition nodes: {len(self.partition_nodes)}")
            lines.append(f"  Estimator nodes: {len(self.estimator_nodes)}")
            stages = self.get_pipeline_stages()
            if stages:
                lines.append("  Stages:")
                for s in stages:
                    lines.append(f"    {s.stage_type}: {len(s.node_ids)} nodes")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of DAG statistics."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "depth": self.depth if self._nodes else 0,
            "width": self.width if self._nodes else 0,
            "n_sources": len(self.sources),
            "n_sinks": len(self.sinks),
            "n_partition_nodes": len(self.partition_nodes),
            "n_estimator_nodes": len(self.estimator_nodes),
            "n_leakage_paths": len(self.find_leakage_paths()),
            "total_edge_capacity": self._edges.total_capacity(),
            "max_provenance_fraction": self._edges.max_provenance_fraction(),
        }

    # -- equality (structural) -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PIDAG):
            return NotImplemented
        return (
            set(self._nodes.keys()) == set(other._nodes.keys())
            and len(self._edges) == len(other._edges)
        )

    def __len__(self) -> int:
        return self.n_nodes

    def __contains__(self, node_id: object) -> bool:
        if isinstance(node_id, str):
            return node_id in self._nodes
        return False
