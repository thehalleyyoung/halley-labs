"""Computation graph: DAG of architecture IR nodes.

Supports topological sorting, dependency tracking, subgraph extraction,
graph transformations, serialization, path enumeration, and cycle detection.
"""

from __future__ import annotations

import copy
import json
import itertools
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .types import KernelRecursionType, ScalingExponents, TensorShape
from .nodes import (
    AbstractNode,
    ActivationNode,
    AttentionNode,
    Conv1DNode,
    Conv2DNode,
    DenseNode,
    DropoutNode,
    FlattenNode,
    InputNode,
    NormNode,
    OutputNode,
    PoolingNode,
    ResidualNode,
    create_node,
)


class ComputationGraph:
    """Directed acyclic graph of computation nodes.

    The graph supports multiple inputs/outputs and arbitrary skip connections.
    Nodes are referenced by their unique ``node_id``.
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._nodes: Dict[str, AbstractNode] = {}
        self._edges: Dict[str, List[str]] = defaultdict(list)   # src -> [dst]
        self._rev_edges: Dict[str, List[str]] = defaultdict(list)  # dst -> [src]
        self._topo_cache: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: AbstractNode) -> str:
        """Add a node to the graph; return its id."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id!r} already in graph")
        self._nodes[node.node_id] = node
        self._invalidate_cache()
        return node.node_id

    def remove_node(self, node_id: str) -> AbstractNode:
        """Remove a node and all incident edges; return the removed node."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found")
        node = self._nodes.pop(node_id)
        # Remove outgoing edges
        for dst in list(self._edges.get(node_id, [])):
            self._rev_edges[dst].remove(node_id)
            self._nodes[dst].predecessors.remove(node_id)
        self._edges.pop(node_id, None)
        # Remove incoming edges
        for src in list(self._rev_edges.get(node_id, [])):
            self._edges[src].remove(node_id)
            self._nodes[src].successors.remove(node_id)
        self._rev_edges.pop(node_id, None)
        self._invalidate_cache()
        return node

    def get_node(self, node_id: str) -> AbstractNode:
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> Dict[str, AbstractNode]:
        return dict(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(self, src_id: str, dst_id: str) -> None:
        """Add a directed edge src -> dst."""
        if src_id not in self._nodes:
            raise KeyError(f"Source node {src_id!r} not found")
        if dst_id not in self._nodes:
            raise KeyError(f"Destination node {dst_id!r} not found")
        if dst_id in self._edges[src_id]:
            return  # already exists
        self._edges[src_id].append(dst_id)
        self._rev_edges[dst_id].append(src_id)
        self._nodes[src_id].successors.append(dst_id)
        self._nodes[dst_id].predecessors.append(src_id)
        self._invalidate_cache()

    def remove_edge(self, src_id: str, dst_id: str) -> None:
        if dst_id in self._edges.get(src_id, []):
            self._edges[src_id].remove(dst_id)
            self._rev_edges[dst_id].remove(src_id)
            self._nodes[src_id].successors.remove(dst_id)
            self._nodes[dst_id].predecessors.remove(src_id)
            self._invalidate_cache()

    def has_edge(self, src_id: str, dst_id: str) -> bool:
        return dst_id in self._edges.get(src_id, [])

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._edges.values())

    def successors_of(self, node_id: str) -> List[str]:
        return list(self._edges.get(node_id, []))

    def predecessors_of(self, node_id: str) -> List[str]:
        return list(self._rev_edges.get(node_id, []))

    # ------------------------------------------------------------------
    # Source / sink discovery
    # ------------------------------------------------------------------

    def input_nodes(self) -> List[str]:
        """Nodes with no predecessors."""
        return [nid for nid, n in self._nodes.items() if not self._rev_edges.get(nid)]

    def output_nodes(self) -> List[str]:
        """Nodes with no successors."""
        return [nid for nid, n in self._nodes.items() if not self._edges.get(nid)]

    # ------------------------------------------------------------------
    # Topological sorting (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return node ids in topological order. Raises on cycle."""
        if self._topo_cache is not None:
            return list(self._topo_cache)

        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for dst in self._edges.get(nid, []):
                in_degree[dst] += 1

        queue: Deque[str] = deque(nid for nid, d in in_degree.items() if d == 0)
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dst in self._edges.get(nid, []):
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)

        if len(order) != len(self._nodes):
            raise ValueError(
                f"Graph has a cycle: processed {len(order)}/{len(self._nodes)} nodes"
            )

        self._topo_cache = order
        return list(order)

    # ------------------------------------------------------------------
    # Cycle detection (DFS-based)
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """Return True if the graph contains a cycle."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in self._nodes}

        def dfs(u: str) -> bool:
            color[u] = GRAY
            for v in self._edges.get(u, []):
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(dfs(nid) for nid, c in color.items() if c == WHITE)

    def find_cycles(self) -> List[List[str]]:
        """Find all elementary cycles (Johnson's algorithm simplified)."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        stack: List[str] = []
        on_stack: Set[str] = set()

        def dfs(u: str, start: str) -> None:
            visited.add(u)
            stack.append(u)
            on_stack.add(u)
            for v in self._edges.get(u, []):
                if v == start and len(stack) > 1:
                    cycles.append(list(stack))
                elif v not in on_stack and v not in visited:
                    dfs(v, start)
            stack.pop()
            on_stack.discard(u)

        for nid in self._nodes:
            visited.clear()
            dfs(nid, nid)

        return cycles

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def all_paths(self, src: str, dst: str, max_paths: int = 1000) -> List[List[str]]:
        """Enumerate all simple paths from src to dst (DFS)."""
        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(u: str, path: List[str]) -> None:
            if len(paths) >= max_paths:
                return
            if u == dst:
                paths.append(list(path))
                return
            visited.add(u)
            for v in self._edges.get(u, []):
                if v not in visited:
                    path.append(v)
                    dfs(v, path)
                    path.pop()
            visited.discard(u)

        dfs(src, [src])
        return paths

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        """BFS shortest path (by hop count)."""
        if src == dst:
            return [src]
        visited: Set[str] = {src}
        queue: Deque[List[str]] = deque([[src]])
        while queue:
            path = queue.popleft()
            for v in self._edges.get(path[-1], []):
                if v == dst:
                    return path + [v]
                if v not in visited:
                    visited.add(v)
                    queue.append(path + [v])
        return None

    def depth(self) -> int:
        """Longest path length (number of edges) in the DAG."""
        order = self.topological_sort()
        dist: Dict[str, int] = {nid: 0 for nid in order}
        for u in order:
            for v in self._edges.get(u, []):
                dist[v] = max(dist[v], dist[u] + 1)
        return max(dist.values()) if dist else 0

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def subgraph(self, node_ids: Iterable[str]) -> "ComputationGraph":
        """Extract an induced subgraph containing only the specified nodes."""
        ids = set(node_ids)
        sg = ComputationGraph(name=f"{self.name}_sub")
        for nid in ids:
            sg.add_node(copy.deepcopy(self._nodes[nid]))
        for src in ids:
            for dst in self._edges.get(src, []):
                if dst in ids:
                    sg.add_edge(src, dst)
        return sg

    def ancestors(self, node_id: str) -> Set[str]:
        """All nodes from which node_id is reachable."""
        result: Set[str] = set()
        queue: Deque[str] = deque([node_id])
        while queue:
            u = queue.popleft()
            for v in self._rev_edges.get(u, []):
                if v not in result:
                    result.add(v)
                    queue.append(v)
        return result

    def descendants(self, node_id: str) -> Set[str]:
        """All nodes reachable from node_id."""
        result: Set[str] = set()
        queue: Deque[str] = deque([node_id])
        while queue:
            u = queue.popleft()
            for v in self._edges.get(u, []):
                if v not in result:
                    result.add(v)
                    queue.append(v)
        return result

    def between(self, src: str, dst: str) -> "ComputationGraph":
        """Subgraph of all nodes on any path from src to dst."""
        fwd = self.descendants(src) | {src}
        bwd = self.ancestors(dst) | {dst}
        return self.subgraph(fwd & bwd)

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def infer_shapes(self, input_shapes: Optional[Dict[str, TensorShape]] = None) -> None:
        """Propagate shapes through the graph in topological order.

        Parameters
        ----------
        input_shapes : dict mapping input-node ids to TensorShapes.
            If None, uses shapes already stored on InputNodes.
        """
        order = self.topological_sort()
        resolved: Dict[str, TensorShape] = {}

        if input_shapes:
            for nid, shape in input_shapes.items():
                resolved[nid] = shape
                node = self._nodes[nid]
                node._input_shape = shape
                node._output_shape = shape

        for nid in order:
            node = self._nodes[nid]
            preds = self._rev_edges.get(nid, [])

            if not preds:
                # Source node
                if nid in resolved:
                    continue
                if node._output_shape is not None:
                    resolved[nid] = node._output_shape
                continue

            # Gather predecessor output shapes
            pred_shapes = [resolved[p] for p in preds if p in resolved]
            if not pred_shapes:
                continue

            in_shape = pred_shapes[0]
            out_shape = node.resolve_shapes(in_shape)
            resolved[nid] = out_shape

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def total_parameters(self) -> int:
        return sum(n.parameter_count() for n in self._nodes.values())

    def parameter_breakdown(self) -> Dict[str, int]:
        return {nid: n.parameter_count() for nid, n in self._nodes.items()}

    def trainable_parameter_ids(self) -> List[str]:
        return [nid for nid, n in self._nodes.items() if n.parameter_count() > 0]

    # ------------------------------------------------------------------
    # Graph transformations
    # ------------------------------------------------------------------

    def fuse_linear_activation(self) -> int:
        """Fuse consecutive Dense+Activation pairs into a single metadata entry.

        Returns the number of fusions performed. The Activation node is removed
        and its type is stored in the Dense node's metadata.
        """
        order = self.topological_sort()
        fused = 0
        to_remove: List[str] = []

        for nid in order:
            node = self._nodes[nid]
            if not isinstance(node, (DenseNode, Conv1DNode, Conv2DNode)):
                continue
            succs = self._edges.get(nid, [])
            if len(succs) != 1:
                continue
            succ = self._nodes[succs[0]]
            if not isinstance(succ, ActivationNode):
                continue
            # Only fuse if the activation has a single predecessor
            if len(self._rev_edges.get(succ.node_id, [])) != 1:
                continue

            node.metadata["fused_activation"] = succ.activation.value
            # Re-wire: skip the activation node
            act_succs = list(self._edges.get(succ.node_id, []))
            to_remove.append(succ.node_id)
            for dst in act_succs:
                self.add_edge(nid, dst)
            fused += 1

        for rid in to_remove:
            self.remove_node(rid)

        return fused

    def split_residual_blocks(self) -> List["ComputationGraph"]:
        """Split the graph at residual nodes into sub-graphs per block."""
        residual_ids = [
            nid for nid, n in self._nodes.items() if isinstance(n, ResidualNode)
        ]
        if not residual_ids:
            return [self.copy()]

        blocks: List[ComputationGraph] = []
        order = self.topological_sort()
        current_block_ids: List[str] = []

        for nid in order:
            current_block_ids.append(nid)
            if nid in residual_ids:
                blocks.append(self.subgraph(current_block_ids))
                current_block_ids = [nid]  # residual node starts next block too

        if len(current_block_ids) > 1:
            blocks.append(self.subgraph(current_block_ids))

        return blocks

    def insert_node_after(self, ref_id: str, new_node: AbstractNode) -> str:
        """Insert new_node right after ref_id, inheriting its successors."""
        nid = self.add_node(new_node)
        succs = list(self._edges.get(ref_id, []))
        for dst in succs:
            self.remove_edge(ref_id, dst)
            self.add_edge(nid, dst)
        self.add_edge(ref_id, nid)
        return nid

    def insert_node_before(self, ref_id: str, new_node: AbstractNode) -> str:
        """Insert new_node right before ref_id, inheriting its predecessors."""
        nid = self.add_node(new_node)
        preds = list(self._rev_edges.get(ref_id, []))
        for src in preds:
            self.remove_edge(src, ref_id)
            self.add_edge(src, nid)
        self.add_edge(nid, ref_id)
        return nid

    def replace_node(self, old_id: str, new_node: AbstractNode) -> str:
        """Replace a node while preserving connectivity."""
        preds = list(self._rev_edges.get(old_id, []))
        succs = list(self._edges.get(old_id, []))
        self.remove_node(old_id)
        nid = self.add_node(new_node)
        for src in preds:
            if src in self._nodes:
                self.add_edge(src, nid)
        for dst in succs:
            if dst in self._nodes:
                self.add_edge(nid, dst)
        return nid

    # ------------------------------------------------------------------
    # Kernel recursion sequence
    # ------------------------------------------------------------------

    def kernel_recursion_sequence(self) -> List[Tuple[str, KernelRecursionType]]:
        """Return the sequence of kernel recursion types in topological order."""
        order = self.topological_sort()
        return [(nid, self._nodes[nid].kernel_recursion_type()) for nid in order]

    def effective_depth(self) -> int:
        """Number of layers that modify the kernel (effective depth for NTK)."""
        seq = self.kernel_recursion_sequence()
        return sum(1 for _, krt in seq if krt.modifies_kernel)

    # ------------------------------------------------------------------
    # Iteration / convenience
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[AbstractNode]:
        """Iterate nodes in topological order."""
        for nid in self.topological_sort():
            yield self._nodes[nid]

    def __len__(self) -> int:
        return self.num_nodes

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def copy(self) -> "ComputationGraph":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable model summary table."""
        lines: List[str] = []
        sep = "=" * 65
        lines.append(sep)
        lines.append(f" Graph: {self.name}")
        lines.append(sep)
        lines.append(f"{'Layer':<25s} {'Output Shape':<20s} {'Params':>10s}")
        lines.append("-" * 65)
        total = 0
        for node in self:
            pcount = node.parameter_count()
            total += pcount
            lines.append(node.summary_line())
        lines.append(sep)
        lines.append(f"Total params: {total:,d}")
        lines.append(f"Effective depth: {self.effective_depth()}")
        lines.append(sep)
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Generate Graphviz DOT representation."""
        lines = [f'digraph "{self.name}" {{', "  rankdir=TB;"]
        for nid, node in self._nodes.items():
            label = f"{node.name}\\n{node.__class__.__name__}"
            shape = "box" if node.parameter_count() > 0 else "ellipse"
            lines.append(f'  "{nid}" [label="{label}", shape={shape}];')
        for src, dsts in self._edges.items():
            for dst in dsts:
                lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return adjacency matrix and ordered list of node ids."""
        ids = self.topological_sort()
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        A = np.zeros((n, n), dtype=np.int32)
        for src, dsts in self._edges.items():
            for dst in dsts:
                A[idx[src], idx[dst]] = 1
        return A, ids

    # ------------------------------------------------------------------
    # Serialization / deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": dict(self._edges),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComputationGraph":
        g = cls(name=d.get("name", "unnamed"))
        for nid, node_dict in d["nodes"].items():
            node = create_node(node_dict)
            node.node_id = nid
            # Clear connectivity (will be rebuilt from edges)
            node.predecessors = []
            node.successors = []
            g._nodes[nid] = node
        for src, dsts in d.get("edges", {}).items():
            for dst in dsts:
                g.add_edge(src, dst)
        return g

    @classmethod
    def from_json(cls, s: str) -> "ComputationGraph":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Builder helpers (fluent API)
    # ------------------------------------------------------------------

    @classmethod
    def sequential(cls, nodes: Sequence[AbstractNode], name: str = "sequential") -> "ComputationGraph":
        """Build a simple sequential graph from a list of nodes."""
        g = cls(name=name)
        prev_id: Optional[str] = None
        for node in nodes:
            nid = g.add_node(node)
            if prev_id is not None:
                g.add_edge(prev_id, nid)
            prev_id = nid
        return g

    @classmethod
    def mlp(
        cls,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        bias: bool = True,
        name: str = "mlp",
    ) -> "ComputationGraph":
        """Build an MLP computation graph."""
        from .types import ActivationType
        act = ActivationType(activation)
        nodes: List[AbstractNode] = [
            InputNode("input", shape=TensorShape.matrix(None, input_dim)),
        ]
        prev_dim = input_dim
        for i, hdim in enumerate(hidden_dims):
            nodes.append(DenseNode(f"dense_{i}", prev_dim, hdim, bias=bias))
            nodes.append(ActivationNode(f"act_{i}", activation=act))
            prev_dim = hdim
        nodes.append(DenseNode("head", prev_dim, output_dim, bias=bias))
        nodes.append(OutputNode("output"))
        return cls.sequential(nodes, name=name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._topo_cache = None

    def __repr__(self) -> str:
        return f"ComputationGraph(name={self.name!r}, nodes={self.num_nodes}, edges={self.num_edges})"
