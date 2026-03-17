"""
taintflow.dag.pi_dag -- Pipeline Information DAG (PI-DAG) implementation.

The :class:`PipelineDAG` wraps a set of :class:`DAGNode` instances (from
``taintflow.dag.nodes``) and :class:`DAGEdge` instances (from
``taintflow.dag.edges``) into a directed acyclic graph that supports
topological traversal, subgraph extraction, path-finding, cycle detection,
and serialisation.

This module is the primary entry point for constructing and querying the
pipeline graph during TaintFlow analysis.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
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

from taintflow.core.types import EdgeKind, OpType, Origin, ShapeMetadata, ProvenanceInfo
from taintflow.core.errors import (
    CycleDetectedError,
    DAGConstructionError,
    MissingNodeError,
)
from taintflow.dag.nodes import DAGNode, SourceLocation, NodeStatus
from taintflow.dag.edges import DAGEdge, EdgeSet, ColumnMapping


# ===================================================================
#  DAG statistics
# ===================================================================


@dataclass(frozen=True)
class DAGStats:
    """Aggregate statistics for a :class:`PipelineDAG`.

    Attributes:
        node_count: Total number of nodes.
        edge_count: Total number of edges.
        source_count: Number of source nodes (in-degree 0).
        sink_count: Number of sink nodes (out-degree 0).
        depth: Length of the longest path (number of edges).
        edge_kind_counts: Number of edges per :class:`EdgeKind`.
        op_type_counts: Number of nodes per :class:`OpType`.
    """

    node_count: int = 0
    edge_count: int = 0
    source_count: int = 0
    sink_count: int = 0
    depth: int = 0
    edge_kind_counts: Dict[str, int] = field(default_factory=dict)
    op_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "source_count": self.source_count,
            "sink_count": self.sink_count,
            "depth": self.depth,
            "edge_kind_counts": dict(self.edge_kind_counts),
            "op_type_counts": dict(self.op_type_counts),
        }

    def __repr__(self) -> str:
        return (
            f"DAGStats(nodes={self.node_count}, edges={self.edge_count}, "
            f"depth={self.depth}, sources={self.source_count}, sinks={self.sink_count})"
        )


# ===================================================================
#  PipelineDAG
# ===================================================================


class PipelineDAG:
    """Directed acyclic graph of ML pipeline operations.

    The ``PipelineDAG`` stores nodes (:class:`DAGNode`) and edges
    (:class:`DAGEdge`) and provides graph-theoretic queries needed by the
    TaintFlow abstract-interpretation engine:

    * Topological ordering (Kahn's algorithm)
    * Predecessor / successor lookups
    * Path enumeration between two nodes
    * Subgraph extraction
    * Cycle detection and validation
    * Node lookup by :class:`OpType`, source location, or predicate
    * Serialisation to / from ``dict`` and JSON

    Parameters
    ----------
    name : str
        Optional descriptive name for this DAG.
    metadata : dict
        Arbitrary metadata attached to the DAG.
    """

    def __init__(
        self,
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name: str = name
        self._metadata: Dict[str, Any] = metadata or {}
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: EdgeSet = EdgeSet()
        # Adjacency caches
        self._successors: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def edges(self) -> EdgeSet:
        """The :class:`EdgeSet` backing this DAG."""
        return self._edges

    # -- node operations -----------------------------------------------------

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG.

        Raises
        ------
        ValueError
            If a node with the same ``node_id`` already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node ID: {node.node_id!r}")
        errors = node.validate()
        if errors:
            raise ValueError(
                f"Invalid node {node.node_id!r}: {'; '.join(errors)}"
            )
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> DAGNode:
        """Remove a node and all its incident edges.

        Returns the removed node.

        Raises
        ------
        MissingNodeError
            If the node does not exist.
        """
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Cannot remove node {node_id!r}: not found",
                node_id=node_id,
                available_nodes=list(self._nodes.keys()),
            )
        self._edges.remove_by_node(node_id)
        # Clean adjacency caches
        for succ in list(self._successors.get(node_id, set())):
            self._predecessors[succ].discard(node_id)
        for pred in list(self._predecessors.get(node_id, set())):
            self._successors[pred].discard(node_id)
        self._successors.pop(node_id, None)
        self._predecessors.pop(node_id, None)
        return self._nodes.pop(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> DAGNode:
        """Return a node by ID.

        Raises
        ------
        MissingNodeError
            If the node does not exist.
        """
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Node {node_id!r} not found",
                node_id=node_id,
                available_nodes=list(self._nodes.keys()),
            )
        return self._nodes[node_id]

    @property
    def nodes(self) -> Dict[str, DAGNode]:
        """Read-only view of node_id → DAGNode mapping."""
        return dict(self._nodes)

    @property
    def node_ids(self) -> List[str]:
        """List of all node IDs in insertion order."""
        return list(self._nodes.keys())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # -- edge operations -----------------------------------------------------

    def add_edge(self, edge: DAGEdge) -> None:
        """Add an edge to the DAG.

        Both the source and target nodes must already exist in the DAG.

        Raises
        ------
        MissingNodeError
            If source or target node does not exist.
        ValueError
            If the edge fails validation.
        CycleDetectedError
            If adding this edge would create a cycle.
        """
        if edge.source_id not in self._nodes:
            raise MissingNodeError(
                f"Edge source {edge.source_id!r} not in DAG",
                node_id=edge.source_id,
                available_nodes=list(self._nodes.keys()),
            )
        if edge.target_id not in self._nodes:
            raise MissingNodeError(
                f"Edge target {edge.target_id!r} not in DAG",
                node_id=edge.target_id,
                available_nodes=list(self._nodes.keys()),
            )
        # Check for cycle: would adding source→target create a path target→...→source?
        if self._would_create_cycle(edge.source_id, edge.target_id):
            raise CycleDetectedError(
                f"Adding edge {edge.source_id!r}→{edge.target_id!r} would create a cycle",
                cycle_nodes=[edge.target_id, edge.source_id],
            )
        self._edges.add(edge)
        self._successors[edge.source_id].add(edge.target_id)
        self._predecessors[edge.target_id].add(edge.source_id)

    def remove_edge(self, edge_id: str) -> DAGEdge:
        """Remove an edge by its ID.

        Raises
        ------
        KeyError
            If the edge does not exist.
        """
        edge = self._edges.remove(edge_id)
        # Only remove from adjacency if no more edges connect the pair
        if not self._edges.has_edge(edge.source_id, edge.target_id):
            self._successors[edge.source_id].discard(edge.target_id)
            self._predecessors[edge.target_id].discard(edge.source_id)
        return edge

    def has_edge(
        self,
        source_id: str,
        target_id: str,
        kind: Optional[EdgeKind] = None,
    ) -> bool:
        """Return True if an edge from *source_id* to *target_id* exists."""
        return self._edges.has_edge(source_id, target_id, kind)

    def edges_from(self, node_id: str) -> List[DAGEdge]:
        """Return all edges originating from *node_id*."""
        return self._edges.from_source(node_id)

    def edges_to(self, node_id: str) -> List[DAGEdge]:
        """Return all edges arriving at *node_id*."""
        return self._edges.to_target(node_id)

    def edges_by_kind(self, kind: EdgeKind) -> List[DAGEdge]:
        """Return all edges of a given kind."""
        return self._edges.by_kind(kind)

    # -- adjacency queries ---------------------------------------------------

    def successors(self, node_id: str) -> List[str]:
        """Return IDs of all immediate successors of *node_id*.

        Raises MissingNodeError if the node is not in the DAG.
        """
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Node {node_id!r} not found",
                node_id=node_id,
            )
        return sorted(self._successors.get(node_id, set()))

    def predecessors(self, node_id: str) -> List[str]:
        """Return IDs of all immediate predecessors of *node_id*.

        Raises MissingNodeError if the node is not in the DAG.
        """
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Node {node_id!r} not found",
                node_id=node_id,
            )
        return sorted(self._predecessors.get(node_id, set()))

    def descendants(self, node_id: str) -> Set[str]:
        """Return all transitive successors (reachable nodes) from *node_id*."""
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Node {node_id!r} not found", node_id=node_id,
            )
        visited: Set[str] = set()
        queue: Deque[str] = deque(self._successors.get(node_id, set()))
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self._successors.get(nid, set()))
        return visited

    def ancestors(self, node_id: str) -> Set[str]:
        """Return all transitive predecessors of *node_id*."""
        if node_id not in self._nodes:
            raise MissingNodeError(
                f"Node {node_id!r} not found", node_id=node_id,
            )
        visited: Set[str] = set()
        queue: Deque[str] = deque(self._predecessors.get(node_id, set()))
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self._predecessors.get(nid, set()))
        return visited

    # -- topological sort (Kahn's algorithm) ---------------------------------

    def topological_sort(self) -> List[str]:
        """Return node IDs in topological order using Kahn's algorithm.

        Raises
        ------
        CycleDetectedError
            If the graph contains a cycle.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for pred in self._predecessors.get(nid, set()):
                if pred in in_degree:
                    in_degree[nid] += 1

        queue: Deque[str] = deque(
            nid for nid, deg in sorted(in_degree.items()) if deg == 0
        )
        result: List[str] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for succ in sorted(self._successors.get(nid, set())):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        if len(result) != len(self._nodes):
            remaining = set(self._nodes.keys()) - set(result)
            raise CycleDetectedError(
                "Graph contains a cycle; topological sort is incomplete",
                cycle_nodes=sorted(remaining),
            )
        return result

    # -- cycle detection -----------------------------------------------------

    def has_cycle(self) -> bool:
        """Return True if the graph contains a cycle."""
        try:
            self.topological_sort()
            return False
        except CycleDetectedError:
            return True

    def find_cycle(self) -> Optional[List[str]]:
        """Return a list of node IDs forming a cycle, or None.

        Uses iterative DFS with a colour map (white/grey/black).
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {nid: WHITE for nid in self._nodes}
        parent: Dict[str, Optional[str]] = {nid: None for nid in self._nodes}

        for start in self._nodes:
            if colour[start] != WHITE:
                continue
            stack: List[Tuple[str, bool]] = [(start, False)]
            while stack:
                nid, returning = stack.pop()
                if returning:
                    colour[nid] = BLACK
                    continue
                if colour[nid] == GREY:
                    colour[nid] = BLACK
                    continue
                colour[nid] = GREY
                stack.append((nid, True))
                for succ in self._successors.get(nid, set()):
                    if colour[succ] == GREY:
                        # Back edge found → reconstruct cycle
                        cycle = [succ, nid]
                        cur = nid
                        while cur != succ and parent.get(cur) is not None:
                            cur = parent[cur]  # type: ignore[assignment]
                            cycle.append(cur)
                        cycle.reverse()
                        return cycle
                    if colour[succ] == WHITE:
                        parent[succ] = nid
                        stack.append((succ, False))
        return None

    def validate(self) -> List[str]:
        """Validate the entire DAG, returning a list of error messages."""
        errors: List[str] = []
        for node in self._nodes.values():
            errors.extend(node.validate())
        for edge in self._edges:
            edge_errors = edge.validate()
            errors.extend(edge_errors)
            if edge.source_id not in self._nodes:
                errors.append(
                    f"Edge {edge.edge_id} references missing source {edge.source_id!r}"
                )
            if edge.target_id not in self._nodes:
                errors.append(
                    f"Edge {edge.edge_id} references missing target {edge.target_id!r}"
                )
        cycle = self.find_cycle()
        if cycle is not None:
            errors.append(f"Graph contains a cycle through nodes: {cycle}")
        return errors

    # -- source / sink identification ----------------------------------------

    def source_nodes(self) -> List[DAGNode]:
        """Return nodes with no incoming edges (DAG sources / inputs)."""
        return [
            self._nodes[nid]
            for nid in self._nodes
            if not self._predecessors.get(nid)
        ]

    def sink_nodes(self) -> List[DAGNode]:
        """Return nodes with no outgoing edges (DAG sinks / outputs)."""
        return [
            self._nodes[nid]
            for nid in self._nodes
            if not self._successors.get(nid)
        ]

    def source_node_ids(self) -> List[str]:
        """Return IDs of source nodes."""
        return [n.node_id for n in self.source_nodes()]

    def sink_node_ids(self) -> List[str]:
        """Return IDs of sink nodes."""
        return [n.node_id for n in self.sink_nodes()]

    # -- node lookup ---------------------------------------------------------

    def nodes_by_op_type(self, op_type: OpType) -> List[DAGNode]:
        """Return all nodes matching a given :class:`OpType`."""
        return [n for n in self._nodes.values() if n.op_type == op_type]

    def nodes_by_status(self, status: NodeStatus) -> List[DAGNode]:
        """Return all nodes with a given :class:`NodeStatus`."""
        return [n for n in self._nodes.values() if n.status == status]

    def nodes_by_source_file(self, file_path: str) -> List[DAGNode]:
        """Return all nodes whose source location matches *file_path*."""
        return [
            n for n in self._nodes.values()
            if n.source_location is not None
            and n.source_location.file_path == file_path
        ]

    def nodes_by_source_location(
        self, file_path: str, line_number: int,
    ) -> List[DAGNode]:
        """Return nodes at a specific source location."""
        return [
            n for n in self._nodes.values()
            if n.source_location is not None
            and n.source_location.file_path == file_path
            and n.source_location.line_number == line_number
        ]

    def find_nodes(self, predicate: Callable[[DAGNode], bool]) -> List[DAGNode]:
        """Return all nodes satisfying *predicate*."""
        return [n for n in self._nodes.values() if predicate(n)]

    # -- path finding --------------------------------------------------------

    def all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 100,
    ) -> List[List[str]]:
        """Enumerate all simple paths from *source_id* to *target_id*.

        Parameters
        ----------
        source_id : str
            Starting node.
        target_id : str
            Ending node.
        max_depth : int
            Maximum path length to prevent combinatorial explosion.

        Returns
        -------
        list[list[str]]
            Each inner list is a sequence of node IDs from source to target.
        """
        if source_id not in self._nodes:
            raise MissingNodeError(
                f"Source {source_id!r} not found", node_id=source_id,
            )
        if target_id not in self._nodes:
            raise MissingNodeError(
                f"Target {target_id!r} not found", node_id=target_id,
            )

        paths: List[List[str]] = []

        def _dfs(current: str, target: str, visited: Set[str], path: List[str]) -> None:
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(list(path))
                return
            for succ in self._successors.get(current, set()):
                if succ not in visited:
                    visited.add(succ)
                    path.append(succ)
                    _dfs(succ, target, visited, path)
                    path.pop()
                    visited.discard(succ)

        _dfs(source_id, target_id, {source_id}, [source_id])
        return paths

    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Return the shortest path (by edge count) from *source_id* to *target_id*.

        Uses BFS. Returns None if no path exists.
        """
        if source_id not in self._nodes:
            raise MissingNodeError(
                f"Source {source_id!r} not found", node_id=source_id,
            )
        if target_id not in self._nodes:
            raise MissingNodeError(
                f"Target {target_id!r} not found", node_id=target_id,
            )
        if source_id == target_id:
            return [source_id]

        visited: Set[str] = {source_id}
        queue: Deque[List[str]] = deque([[source_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for succ in self._successors.get(current, set()):
                if succ == target_id:
                    return path + [succ]
                if succ not in visited:
                    visited.add(succ)
                    queue.append(path + [succ])
        return None

    def is_reachable(self, source_id: str, target_id: str) -> bool:
        """Return True if *target_id* is reachable from *source_id*."""
        if source_id == target_id:
            return True
        return target_id in self.descendants(source_id)

    # -- subgraph extraction -------------------------------------------------

    def subgraph(self, node_ids: Iterable[str]) -> PipelineDAG:
        """Extract an induced subgraph containing only the specified nodes.

        The returned DAG contains deep copies of the selected nodes and
        only the edges whose source *and* target are both in the subset.
        """
        keep = set(node_ids)
        sub = PipelineDAG(
            name=f"{self._name}:subgraph",
            metadata=dict(self._metadata),
        )
        for nid in keep:
            if nid in self._nodes:
                sub.add_node(copy.deepcopy(self._nodes[nid]))

        for edge in self._edges:
            if edge.source_id in keep and edge.target_id in keep:
                sub._edges.add(copy.deepcopy(edge))
                sub._successors[edge.source_id].add(edge.target_id)
                sub._predecessors[edge.target_id].add(edge.source_id)
        return sub

    def slice_forward(self, start_id: str) -> PipelineDAG:
        """Return the subgraph reachable from *start_id* (inclusive)."""
        reachable = self.descendants(start_id) | {start_id}
        return self.subgraph(reachable)

    def slice_backward(self, end_id: str) -> PipelineDAG:
        """Return the subgraph of ancestors of *end_id* (inclusive)."""
        ancs = self.ancestors(end_id) | {end_id}
        return self.subgraph(ancs)

    def slice_between(self, source_id: str, target_id: str) -> PipelineDAG:
        """Return the subgraph of nodes on any path from *source* to *target*."""
        forward = self.descendants(source_id) | {source_id}
        backward = self.ancestors(target_id) | {target_id}
        between = forward & backward
        return self.subgraph(between)

    # -- depth / statistics --------------------------------------------------

    def depth(self) -> int:
        """Return the length of the longest path (number of edges).

        Uses dynamic programming on the topological order.
        """
        if not self._nodes:
            return 0
        try:
            topo = self.topological_sort()
        except CycleDetectedError:
            return -1

        dist: Dict[str, int] = {nid: 0 for nid in topo}
        for nid in topo:
            for succ in self._successors.get(nid, set()):
                if succ in dist:
                    new_dist = dist[nid] + 1
                    if new_dist > dist[succ]:
                        dist[succ] = new_dist

        return max(dist.values()) if dist else 0

    def stats(self) -> DAGStats:
        """Compute aggregate statistics for this DAG."""
        edge_kind_counts: Dict[str, int] = defaultdict(int)
        for edge in self._edges:
            edge_kind_counts[edge.edge_kind.value] += 1

        op_type_counts: Dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            op_type_counts[node.op_type.value] += 1

        return DAGStats(
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            source_count=len(self.source_nodes()),
            sink_count=len(self.sink_nodes()),
            depth=self.depth(),
            edge_kind_counts=dict(edge_kind_counts),
            op_type_counts=dict(op_type_counts),
        )

    # -- iterator protocol ---------------------------------------------------

    def __iter__(self) -> Iterator[DAGNode]:
        """Iterate over nodes in topological order.

        Falls back to insertion order if the DAG contains a cycle.
        """
        try:
            order = self.topological_sort()
        except CycleDetectedError:
            order = list(self._nodes.keys())
        for nid in order:
            yield self._nodes[nid]

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def __bool__(self) -> bool:
        return bool(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire DAG to a JSON-compatible dictionary."""
        return {
            "name": self._name,
            "metadata": dict(self._metadata),
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": self._edges.to_list(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PipelineDAG:
        """Deserialize a DAG from a dictionary.

        Nodes are added first, then edges, preserving referential integrity.
        """
        dag = cls(
            name=d.get("name", ""),
            metadata=dict(d.get("metadata", {})),
        )
        for node_data in d.get("nodes", []):
            dag.add_node(DAGNode.from_dict(node_data))

        for edge_data in d.get("edges", []):
            edge = DAGEdge.from_dict(edge_data)
            # Only add edge if both endpoints exist
            if edge.source_id in dag._nodes and edge.target_id in dag._nodes:
                dag.add_edge(edge)
        return dag

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize the DAG to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineDAG:
        """Deserialize a DAG from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -- internal helpers ----------------------------------------------------

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Return True if adding an edge source→target would create a cycle.

        Checks whether *source_id* is reachable from *target_id* via the
        existing edges (which would mean target→…→source already exists).
        """
        if source_id == target_id:
            return True
        visited: Set[str] = set()
        queue: Deque[str] = deque([target_id])
        while queue:
            nid = queue.popleft()
            if nid == source_id:
                return True
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self._successors.get(nid, set()))
        return False

    # -- copy ----------------------------------------------------------------

    def copy(self) -> PipelineDAG:
        """Return a deep copy of this DAG."""
        return copy.deepcopy(self)

    # -- dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PipelineDAG(name={self._name!r}, "
            f"nodes={len(self._nodes)}, edges={len(self._edges)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineDAG):
            return NotImplemented
        return (
            set(self._nodes.keys()) == set(other._nodes.keys())
            and len(self._edges) == len(other._edges)
        )


# ===================================================================
#  Public API
# ===================================================================

__all__ = [
    "DAGStats",
    "PipelineDAG",
]
