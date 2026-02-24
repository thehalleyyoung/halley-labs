from __future__ import annotations

import ast
import hashlib
import json
import time
import threading
import os
import re
import fnmatch
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Predicate representation
# ---------------------------------------------------------------------------

class PredicateKind(Enum):
    """Kind of a refinement predicate."""
    TYPE_CHECK = auto()
    RANGE = auto()
    NULL_CHECK = auto()
    LENGTH = auto()
    MEMBER = auto()
    CUSTOM = auto()
    EQUALITY = auto()
    ISINSTANCE = auto()
    TRUTHY = auto()
    COMPARISON = auto()
    REGEX = auto()
    CALLABLE_CHECK = auto()
    ATTRIBUTE_CHECK = auto()
    CONTAINER_CHECK = auto()


@dataclass(frozen=True)
class Predicate:
    """A single refinement predicate."""
    kind: PredicateKind
    name: str
    args: Tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash((self.kind, self.name, self.args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return NotImplemented
        return self.kind == other.kind and self.name == other.name and self.args == other.args

    def __repr__(self) -> str:
        if self.args:
            return f"Predicate({self.kind.name}, {self.name}, {self.args})"
        return f"Predicate({self.kind.name}, {self.name})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.name,
            "name": self.name,
            "args": list(self.args),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Predicate:
        return cls(
            kind=PredicateKind[data["kind"]],
            name=data["name"],
            args=tuple(data.get("args", [])),
        )


# ---------------------------------------------------------------------------
# PredicateSet
# ---------------------------------------------------------------------------

class PredicateSet:
    """Set of predicates associated with a dependency edge or function summary."""

    def __init__(self, predicates: Optional[Set[Predicate]] = None) -> None:
        self._predicates: Set[Predicate] = set(predicates) if predicates else set()

    def add(self, predicate: Predicate) -> None:
        self._predicates.add(predicate)

    def remove(self, predicate: Predicate) -> None:
        self._predicates.discard(predicate)

    def contains(self, predicate: Predicate) -> bool:
        return predicate in self._predicates

    def union(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._predicates | other._predicates)

    def intersection(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._predicates & other._predicates)

    def difference(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._predicates - other._predicates)

    def symmetric_difference(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._predicates ^ other._predicates)

    def overlaps(self, other: PredicateSet) -> bool:
        return bool(self._predicates & other._predicates)

    def is_subset(self, other: PredicateSet) -> bool:
        return self._predicates.issubset(other._predicates)

    def is_superset(self, other: PredicateSet) -> bool:
        return self._predicates.issuperset(other._predicates)

    def is_empty(self) -> bool:
        return len(self._predicates) == 0

    def size(self) -> int:
        return len(self._predicates)

    def clear(self) -> None:
        self._predicates.clear()

    def copy(self) -> PredicateSet:
        return PredicateSet(set(self._predicates))

    def predicates(self) -> FrozenSet[Predicate]:
        return frozenset(self._predicates)

    def filter_by_kind(self, kind: PredicateKind) -> PredicateSet:
        return PredicateSet({p for p in self._predicates if p.kind == kind})

    def __hash__(self) -> int:
        return hash(frozenset(self._predicates))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PredicateSet):
            return NotImplemented
        return self._predicates == other._predicates

    def __len__(self) -> int:
        return len(self._predicates)

    def __iter__(self) -> Iterator[Predicate]:
        return iter(self._predicates)

    def __repr__(self) -> str:
        return f"PredicateSet({self._predicates})"

    def to_dict(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in sorted(self._predicates, key=lambda p: (p.kind.name, p.name))]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> PredicateSet:
        return cls({Predicate.from_dict(d) for d in data})


# ---------------------------------------------------------------------------
# Dependency graph node / edge metadata
# ---------------------------------------------------------------------------

@dataclass
class NodeMetadata:
    """Metadata associated with a dependency graph node."""
    module_path: str = ""
    class_name: Optional[str] = None
    function_name: str = ""
    line_start: int = 0
    line_end: int = 0
    source_hash: str = ""
    predicate_set: PredicateSet = field(default_factory=PredicateSet)
    summary: Optional[Dict[str, Any]] = None
    last_analyzed: float = 0.0
    analysis_version: int = 0
    is_entry_point: bool = False
    is_test: bool = False
    complexity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "class_name": self.class_name,
            "function_name": self.function_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "source_hash": self.source_hash,
            "predicate_set": self.predicate_set.to_dict(),
            "last_analyzed": self.last_analyzed,
            "analysis_version": self.analysis_version,
            "is_entry_point": self.is_entry_point,
            "is_test": self.is_test,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeMetadata:
        return cls(
            module_path=data.get("module_path", ""),
            class_name=data.get("class_name"),
            function_name=data.get("function_name", ""),
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            source_hash=data.get("source_hash", ""),
            predicate_set=PredicateSet.from_dict(data.get("predicate_set", [])),
            last_analyzed=data.get("last_analyzed", 0.0),
            analysis_version=data.get("analysis_version", 0),
            is_entry_point=data.get("is_entry_point", False),
            is_test=data.get("is_test", False),
            complexity=data.get("complexity", 0),
        )


@dataclass
class EdgeMetadata:
    """Metadata associated with a dependency edge."""
    predicate_set: PredicateSet = field(default_factory=PredicateSet)
    call_sites: List[int] = field(default_factory=list)
    is_direct: bool = True
    is_conditional: bool = False
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate_set": self.predicate_set.to_dict(),
            "call_sites": self.call_sites,
            "is_direct": self.is_direct,
            "is_conditional": self.is_conditional,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EdgeMetadata:
        return cls(
            predicate_set=PredicateSet.from_dict(data.get("predicate_set", [])),
            call_sites=data.get("call_sites", []),
            is_direct=data.get("is_direct", True),
            is_conditional=data.get("is_conditional", False),
            weight=data.get("weight", 1.0),
        )


# ---------------------------------------------------------------------------
# Graph diff result
# ---------------------------------------------------------------------------

@dataclass
class GraphDiff:
    """Result of diffing two dependency graphs."""
    added_nodes: Set[str] = field(default_factory=set)
    removed_nodes: Set[str] = field(default_factory=set)
    modified_nodes: Set[str] = field(default_factory=set)
    added_edges: Set[Tuple[str, str]] = field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = field(default_factory=set)
    modified_edges: Set[Tuple[str, str]] = field(default_factory=set)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_nodes or self.removed_nodes or self.modified_nodes
            or self.added_edges or self.removed_edges or self.modified_edges
        )

    def summary(self) -> str:
        parts: List[str] = []
        if self.added_nodes:
            parts.append(f"+{len(self.added_nodes)} nodes")
        if self.removed_nodes:
            parts.append(f"-{len(self.removed_nodes)} nodes")
        if self.modified_nodes:
            parts.append(f"~{len(self.modified_nodes)} nodes")
        if self.added_edges:
            parts.append(f"+{len(self.added_edges)} edges")
        if self.removed_edges:
            parts.append(f"-{len(self.removed_edges)} edges")
        if self.modified_edges:
            parts.append(f"~{len(self.modified_edges)} edges")
        return ", ".join(parts) if parts else "no changes"


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """
    Function-level dependency graph G = (V, E) where vertices are
    function identifiers and edges represent call relationships annotated
    with predicate sets.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeMetadata] = {}
        self._forward: Dict[str, Dict[str, EdgeMetadata]] = defaultdict(dict)
        self._backward: Dict[str, Dict[str, EdgeMetadata]] = defaultdict(dict)
        self._lock = threading.RLock()

    # -- node operations ----------------------------------------------------

    def add_node(self, func_id: str, metadata: Optional[NodeMetadata] = None) -> None:
        with self._lock:
            if metadata is None:
                metadata = NodeMetadata()
            self._nodes[func_id] = metadata
            if func_id not in self._forward:
                self._forward[func_id] = {}
            if func_id not in self._backward:
                self._backward[func_id] = {}

    def remove_node(self, func_id: str) -> None:
        with self._lock:
            if func_id not in self._nodes:
                return
            for succ in list(self._forward.get(func_id, {}).keys()):
                self._backward[succ].pop(func_id, None)
            for pred in list(self._backward.get(func_id, {}).keys()):
                self._forward[pred].pop(func_id, None)
            self._forward.pop(func_id, None)
            self._backward.pop(func_id, None)
            del self._nodes[func_id]

    def has_node(self, func_id: str) -> bool:
        return func_id in self._nodes

    def get_node(self, func_id: str) -> Optional[NodeMetadata]:
        return self._nodes.get(func_id)

    def update_node(self, func_id: str, metadata: NodeMetadata) -> None:
        with self._lock:
            if func_id in self._nodes:
                self._nodes[func_id] = metadata

    def node_count(self) -> int:
        return len(self._nodes)

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def nodes_with_metadata(self) -> List[Tuple[str, NodeMetadata]]:
        return list(self._nodes.items())

    # -- edge operations ----------------------------------------------------

    def add_edge(
        self,
        caller: str,
        callee: str,
        predicate_set: Optional[PredicateSet] = None,
        metadata: Optional[EdgeMetadata] = None,
    ) -> None:
        with self._lock:
            if caller not in self._nodes:
                self.add_node(caller)
            if callee not in self._nodes:
                self.add_node(callee)
            if metadata is None:
                metadata = EdgeMetadata(
                    predicate_set=predicate_set if predicate_set else PredicateSet()
                )
            self._forward[caller][callee] = metadata
            self._backward[callee][caller] = metadata

    def remove_edge(self, caller: str, callee: str) -> None:
        with self._lock:
            self._forward.get(caller, {}).pop(callee, None)
            self._backward.get(callee, {}).pop(caller, None)

    def has_edge(self, caller: str, callee: str) -> bool:
        return callee in self._forward.get(caller, {})

    def get_edge(self, caller: str, callee: str) -> Optional[EdgeMetadata]:
        return self._forward.get(caller, {}).get(callee)

    def edge_count(self) -> int:
        return sum(len(succs) for succs in self._forward.values())

    def edges(self) -> List[Tuple[str, str, EdgeMetadata]]:
        result: List[Tuple[str, str, EdgeMetadata]] = []
        for caller, succs in self._forward.items():
            for callee, meta in succs.items():
                result.append((caller, callee, meta))
        return result

    # -- neighbours ---------------------------------------------------------

    def get_successors(self, func_id: str) -> List[str]:
        return list(self._forward.get(func_id, {}).keys())

    def get_predecessors(self, func_id: str) -> List[str]:
        return list(self._backward.get(func_id, {}).keys())

    def get_successors_with_edges(self, func_id: str) -> List[Tuple[str, EdgeMetadata]]:
        return list(self._forward.get(func_id, {}).items())

    def get_predecessors_with_edges(self, func_id: str) -> List[Tuple[str, EdgeMetadata]]:
        return list(self._backward.get(func_id, {}).items())

    # -- transitive closure -------------------------------------------------

    def get_transitive_dependents(self, func_id: str) -> Set[str]:
        visited: Set[str] = set()
        queue: Deque[str] = deque()
        for s in self.get_successors(func_id):
            if s not in visited:
                visited.add(s)
                queue.append(s)
        while queue:
            current = queue.popleft()
            for s in self.get_successors(current):
                if s not in visited:
                    visited.add(s)
                    queue.append(s)
        return visited

    def get_transitive_dependencies(self, func_id: str) -> Set[str]:
        visited: Set[str] = set()
        queue: Deque[str] = deque()
        for p in self.get_predecessors(func_id):
            if p not in visited:
                visited.add(p)
                queue.append(p)
        while queue:
            current = queue.popleft()
            for p in self.get_predecessors(current):
                if p not in visited:
                    visited.add(p)
                    queue.append(p)
        return visited

    # -- Tarjan's SCC -------------------------------------------------------

    def get_strongly_connected_components(self) -> List[List[str]]:
        index_counter = [0]
        stack: List[str] = []
        lowlink: Dict[str, int] = {}
        index: Dict[str, int] = {}
        on_stack: Set[str] = set()
        result: List[List[str]] = []

        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.get_successors(v):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                result.append(component)

        for v in self._nodes:
            if v not in index:
                strongconnect(v)

        return result

    # -- topological sort ---------------------------------------------------

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for caller, succs in self._forward.items():
            for callee in succs:
                if callee in in_degree:
                    in_degree[callee] += 1

        queue: Deque[str] = deque(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in self.get_successors(node):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        if len(order) != len(self._nodes):
            sccs = [c for c in self.get_strongly_connected_components() if len(c) > 1]
            visited_in_cycle: Set[str] = set()
            for scc in sccs:
                for n in scc:
                    visited_in_cycle.add(n)
            for n in self._nodes:
                if n not in set(order) and n not in visited_in_cycle:
                    order.append(n)
            for scc in sccs:
                order.extend(scc)

        return order

    # -- roots / leaves -----------------------------------------------------

    def get_roots(self) -> List[str]:
        return [n for n in self._nodes if not self._backward.get(n)]

    def get_leaves(self) -> List[str]:
        return [n for n in self._nodes if not self._forward.get(n)]

    # -- serialization ------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        nodes_data: Dict[str, Any] = {}
        for func_id, meta in self._nodes.items():
            nodes_data[func_id] = meta.to_dict()
        edges_data: List[Dict[str, Any]] = []
        for caller, succs in self._forward.items():
            for callee, meta in succs.items():
                edges_data.append({
                    "caller": caller,
                    "callee": callee,
                    "metadata": meta.to_dict(),
                })
        return {"nodes": nodes_data, "edges": edges_data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DependencyGraph:
        graph = cls()
        for func_id, meta_data in data.get("nodes", {}).items():
            graph.add_node(func_id, NodeMetadata.from_dict(meta_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(
                edge_data["caller"],
                edge_data["callee"],
                metadata=EdgeMetadata.from_dict(edge_data.get("metadata", {})),
            )
        return graph

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.serialize(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> DependencyGraph:
        return cls.deserialize(json.loads(json_str))

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(self.to_json())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> DependencyGraph:
        with open(path, "r") as f:
            return cls.from_json(f.read())

    # -- diff ---------------------------------------------------------------

    def diff(self, other: DependencyGraph) -> GraphDiff:
        result = GraphDiff()
        self_nodes = set(self._nodes.keys())
        other_nodes = set(other._nodes.keys())
        result.added_nodes = other_nodes - self_nodes
        result.removed_nodes = self_nodes - other_nodes
        for n in self_nodes & other_nodes:
            if self._nodes[n].source_hash != other._nodes[n].source_hash:
                result.modified_nodes.add(n)

        self_edges = {(c, s) for c in self._forward for s in self._forward[c]}
        other_edges = {(c, s) for c in other._forward for s in other._forward[c]}
        result.added_edges = other_edges - self_edges
        result.removed_edges = self_edges - other_edges
        for e in self_edges & other_edges:
            old_meta = self._forward[e[0]][e[1]]
            new_meta = other._forward[e[0]][e[1]]
            if old_meta.predicate_set != new_meta.predicate_set:
                result.modified_edges.add(e)
        return result

    # -- DOT visualization --------------------------------------------------

    def to_dot(self, title: str = "DependencyGraph") -> str:
        lines = [f'digraph "{title}" {{', "  rankdir=LR;", '  node [shape=box, style=filled, fillcolor="#E8E8E8"];']
        for func_id, meta in self._nodes.items():
            label = func_id.replace('"', '\\"')
            color = "#E8E8E8"
            if meta.is_entry_point:
                color = "#90EE90"
            elif meta.is_test:
                color = "#ADD8E6"
            lines.append(f'  "{label}" [fillcolor="{color}"];')

        for caller, succs in self._forward.items():
            for callee, edge_meta in succs.items():
                c_label = caller.replace('"', '\\"')
                s_label = callee.replace('"', '\\"')
                pred_count = edge_meta.predicate_set.size()
                edge_label = f"{pred_count}P" if pred_count > 0 else ""
                style = "dashed" if edge_meta.is_conditional else "solid"
                lines.append(
                    f'  "{c_label}" -> "{s_label}" '
                    f'[label="{edge_label}", style={style}];'
                )
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DependencyGraph(nodes={self.node_count()}, edges={self.edge_count()})"


# ---------------------------------------------------------------------------
# ChangeKind / Change types
# ---------------------------------------------------------------------------

class ChangeKind(Enum):
    UNCHANGED = auto()
    ADDED = auto()
    REMOVED = auto()
    SIGNATURE_CHANGED = auto()
    BODY_CHANGED = auto()
    PREDICATES_CHANGED = auto()
    RENAMED = auto()
    MOVED = auto()


@dataclass
class Change:
    """Represents a detected change to a function."""
    kind: ChangeKind
    func_id: str
    old_func_id: Optional[str] = None
    module_path: str = ""
    class_name: Optional[str] = None
    function_name: str = ""
    old_source: str = ""
    new_source: str = ""
    old_predicates: PredicateSet = field(default_factory=PredicateSet)
    new_predicates: PredicateSet = field(default_factory=PredicateSet)
    line_start: int = 0
    line_end: int = 0

    @property
    def delta_predicates(self) -> PredicateSet:
        return self.old_predicates.symmetric_difference(self.new_predicates)

    def __repr__(self) -> str:
        return f"Change({self.kind.name}, {self.func_id})"


# ---------------------------------------------------------------------------
# FunctionFingerprint
# ---------------------------------------------------------------------------

class FunctionFingerprint:
    """Fingerprint of a function for fast change detection."""

    def __init__(
        self,
        source_hash: str = "",
        predicate_set: Optional[PredicateSet] = None,
        signature_hash: str = "",
        body_hash: str = "",
        name: str = "",
        arg_count: int = 0,
        has_varargs: bool = False,
        has_kwargs: bool = False,
        return_annotations: str = "",
        decorators: Tuple[str, ...] = (),
        local_vars: Tuple[str, ...] = (),
        called_functions: Tuple[str, ...] = (),
        complexity: int = 0,
    ) -> None:
        self.source_hash = source_hash
        self.predicate_set = predicate_set if predicate_set else PredicateSet()
        self.signature_hash = signature_hash
        self.body_hash = body_hash
        self.name = name
        self.arg_count = arg_count
        self.has_varargs = has_varargs
        self.has_kwargs = has_kwargs
        self.return_annotations = return_annotations
        self.decorators = decorators
        self.local_vars = local_vars
        self.called_functions = called_functions
        self.complexity = complexity

    @staticmethod
    def _hash_str(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def compute_fingerprint(cls, source: str, name: str = "<unknown>") -> FunctionFingerprint:
        source_hash = cls._hash_str(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return cls(source_hash=source_hash, name=name, body_hash=cls._hash_str(source))

        func_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_defs:
            return cls(source_hash=source_hash, name=name, body_hash=cls._hash_str(source))

        func_node = func_defs[0]
        sig_parts: List[str] = [func_node.name]
        args = func_node.args
        arg_count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        has_varargs = args.vararg is not None
        has_kwargs = args.kwarg is not None
        for a in args.args:
            sig_parts.append(a.arg)
            if a.annotation:
                sig_parts.append(ast.dump(a.annotation))
        if func_node.returns:
            sig_parts.append("return:" + ast.dump(func_node.returns))
        signature_hash = cls._hash_str("|".join(sig_parts))

        body_source_lines: List[str] = []
        for node in func_node.body:
            body_source_lines.append(ast.dump(node))
        body_hash = cls._hash_str("\n".join(body_source_lines))

        decorators: List[str] = []
        for dec in func_node.decorator_list:
            decorators.append(ast.dump(dec))

        predicate_set = PredicateSet()
        called_functions: List[str] = []
        local_vars: List[str] = []
        complexity = 0

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                    called_functions.append(fn_name)
                    if fn_name == "isinstance":
                        predicate_set.add(Predicate(PredicateKind.ISINSTANCE, "isinstance"))
                    elif fn_name == "callable":
                        predicate_set.add(Predicate(PredicateKind.CALLABLE_CHECK, "callable"))
                    elif fn_name == "hasattr":
                        predicate_set.add(Predicate(PredicateKind.ATTRIBUTE_CHECK, "hasattr"))
                    elif fn_name == "len":
                        predicate_set.add(Predicate(PredicateKind.LENGTH, "len"))
                elif isinstance(node.func, ast.Attribute):
                    called_functions.append(node.func.attr)
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        predicate_set.add(Predicate(PredicateKind.NULL_CHECK, "is_none"))
                    elif isinstance(op, (ast.In, ast.NotIn)):
                        predicate_set.add(Predicate(PredicateKind.MEMBER, "in"))
                    else:
                        predicate_set.add(Predicate(PredicateKind.COMPARISON, type(op).__name__))
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_vars.append(node.id)

        return_annotation = ""
        if func_node.returns:
            return_annotation = ast.dump(func_node.returns)

        return cls(
            source_hash=source_hash,
            predicate_set=predicate_set,
            signature_hash=signature_hash,
            body_hash=body_hash,
            name=func_node.name,
            arg_count=arg_count,
            has_varargs=has_varargs,
            has_kwargs=has_kwargs,
            return_annotations=return_annotation,
            decorators=tuple(decorators),
            local_vars=tuple(sorted(set(local_vars))),
            called_functions=tuple(sorted(set(called_functions))),
            complexity=complexity,
        )

    def changed_from(self, other: FunctionFingerprint) -> ChangeKind:
        if self.source_hash == other.source_hash:
            return ChangeKind.UNCHANGED
        if self.signature_hash != other.signature_hash:
            return ChangeKind.SIGNATURE_CHANGED
        if self.predicate_set != other.predicate_set:
            return ChangeKind.PREDICATES_CHANGED
        if self.body_hash != other.body_hash:
            return ChangeKind.BODY_CHANGED
        return ChangeKind.UNCHANGED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "predicate_set": self.predicate_set.to_dict(),
            "signature_hash": self.signature_hash,
            "body_hash": self.body_hash,
            "name": self.name,
            "arg_count": self.arg_count,
            "has_varargs": self.has_varargs,
            "has_kwargs": self.has_kwargs,
            "return_annotations": self.return_annotations,
            "decorators": list(self.decorators),
            "local_vars": list(self.local_vars),
            "called_functions": list(self.called_functions),
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FunctionFingerprint:
        return cls(
            source_hash=data.get("source_hash", ""),
            predicate_set=PredicateSet.from_dict(data.get("predicate_set", [])),
            signature_hash=data.get("signature_hash", ""),
            body_hash=data.get("body_hash", ""),
            name=data.get("name", ""),
            arg_count=data.get("arg_count", 0),
            has_varargs=data.get("has_varargs", False),
            has_kwargs=data.get("has_kwargs", False),
            return_annotations=data.get("return_annotations", ""),
            decorators=tuple(data.get("decorators", [])),
            local_vars=tuple(data.get("local_vars", [])),
            called_functions=tuple(data.get("called_functions", [])),
            complexity=data.get("complexity", 0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionFingerprint):
            return NotImplemented
        return self.source_hash == other.source_hash

    def __hash__(self) -> int:
        return hash(self.source_hash)

    def __repr__(self) -> str:
        return f"FunctionFingerprint({self.name}, hash={self.source_hash[:8]})"


# ---------------------------------------------------------------------------
# AST diff helpers
# ---------------------------------------------------------------------------

@dataclass
class ASTDiffEntry:
    """One entry in an AST diff."""
    kind: str  # "added", "removed", "modified"
    node_type: str
    path: str
    old_dump: str = ""
    new_dump: str = ""
    line: int = 0


class ASTDiffer:
    """Diff two Python ASTs at the statement level."""

    @staticmethod
    def diff_sources(old_source: str, new_source: str) -> List[ASTDiffEntry]:
        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            return [ASTDiffEntry(kind="modified", node_type="module", path="<root>")]

        return ASTDiffer._diff_node_lists(old_tree.body, new_tree.body, "<module>")

    @staticmethod
    def _diff_node_lists(
        old_nodes: List[ast.AST], new_nodes: List[ast.AST], parent_path: str
    ) -> List[ASTDiffEntry]:
        entries: List[ASTDiffEntry] = []
        old_dumps = [ast.dump(n) for n in old_nodes]
        new_dumps = [ast.dump(n) for n in new_nodes]

        old_set = set(old_dumps)
        new_set = set(new_dumps)

        for i, dump in enumerate(old_dumps):
            if dump not in new_set:
                node = old_nodes[i]
                line = getattr(node, "lineno", 0)
                entries.append(ASTDiffEntry(
                    kind="removed",
                    node_type=type(node).__name__,
                    path=f"{parent_path}[{i}]",
                    old_dump=dump,
                    line=line,
                ))
        for i, dump in enumerate(new_dumps):
            if dump not in old_set:
                node = new_nodes[i]
                line = getattr(node, "lineno", 0)
                entries.append(ASTDiffEntry(
                    kind="added",
                    node_type=type(node).__name__,
                    path=f"{parent_path}[{i}]",
                    new_dump=dump,
                    line=line,
                ))
        return entries

    @staticmethod
    def extract_functions(source: str) -> Dict[str, ast.AST]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        result: Dict[str, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result[f"{node.name}.{item.name}"] = item
        return result


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Detects changes between two versions of a module."""

    def __init__(self, rename_threshold: float = 0.7) -> None:
        self._rename_threshold = rename_threshold

    def detect_changes(
        self,
        old_source: str,
        new_source: str,
        module_path: str = "<module>",
    ) -> List[Change]:
        old_funcs = ASTDiffer.extract_functions(old_source)
        new_funcs = ASTDiffer.extract_functions(new_source)

        changes: List[Change] = []
        old_names = set(old_funcs.keys())
        new_names = set(new_funcs.keys())

        # Exact matches
        common = old_names & new_names
        only_old = old_names - new_names
        only_new = new_names - old_names

        # Attempt rename detection
        rename_map: Dict[str, str] = {}
        if only_old and only_new:
            rename_map = self._detect_renames(
                {n: old_funcs[n] for n in only_old},
                {n: new_funcs[n] for n in only_new},
            )
            for old_n, new_n in rename_map.items():
                only_old.discard(old_n)
                only_new.discard(new_n)

        for name in sorted(only_old):
            old_src = self._get_func_source(old_funcs[name], old_source)
            fp = FunctionFingerprint.compute_fingerprint(old_src, name)
            changes.append(Change(
                kind=ChangeKind.REMOVED,
                func_id=f"{module_path}:{name}",
                module_path=module_path,
                function_name=name,
                old_source=old_src,
                old_predicates=fp.predicate_set,
            ))

        for name in sorted(only_new):
            new_src = self._get_func_source(new_funcs[name], new_source)
            fp = FunctionFingerprint.compute_fingerprint(new_src, name)
            changes.append(Change(
                kind=ChangeKind.ADDED,
                func_id=f"{module_path}:{name}",
                module_path=module_path,
                function_name=name,
                new_source=new_src,
                new_predicates=fp.predicate_set,
            ))

        for old_name, new_name in rename_map.items():
            old_src = self._get_func_source(old_funcs[old_name], old_source)
            new_src = self._get_func_source(new_funcs[new_name], new_source)
            old_fp = FunctionFingerprint.compute_fingerprint(old_src, old_name)
            new_fp = FunctionFingerprint.compute_fingerprint(new_src, new_name)
            changes.append(Change(
                kind=ChangeKind.RENAMED,
                func_id=f"{module_path}:{new_name}",
                old_func_id=f"{module_path}:{old_name}",
                module_path=module_path,
                function_name=new_name,
                old_source=old_src,
                new_source=new_src,
                old_predicates=old_fp.predicate_set,
                new_predicates=new_fp.predicate_set,
            ))

        for name in sorted(common):
            old_src = self._get_func_source(old_funcs[name], old_source)
            new_src = self._get_func_source(new_funcs[name], new_source)
            if old_src == new_src:
                continue
            old_fp = FunctionFingerprint.compute_fingerprint(old_src, name)
            new_fp = FunctionFingerprint.compute_fingerprint(new_src, name)
            change_kind = new_fp.changed_from(old_fp)
            if change_kind == ChangeKind.UNCHANGED:
                continue
            changes.append(Change(
                kind=change_kind,
                func_id=f"{module_path}:{name}",
                module_path=module_path,
                function_name=name,
                old_source=old_src,
                new_source=new_src,
                old_predicates=old_fp.predicate_set,
                new_predicates=new_fp.predicate_set,
            ))

        return changes

    def _detect_renames(
        self,
        old_funcs: Dict[str, ast.AST],
        new_funcs: Dict[str, ast.AST],
    ) -> Dict[str, str]:
        renames: Dict[str, str] = {}
        used_new: Set[str] = set()
        scored: List[Tuple[float, str, str]] = []

        for old_name, old_node in old_funcs.items():
            old_dump = ast.dump(old_node)
            for new_name, new_node in new_funcs.items():
                if new_name in used_new:
                    continue
                new_dump = ast.dump(new_node)
                sim = self._similarity(old_dump, new_dump)
                if sim >= self._rename_threshold:
                    scored.append((sim, old_name, new_name))

        scored.sort(key=lambda x: -x[0])
        used_old: Set[str] = set()
        for sim, old_name, new_name in scored:
            if old_name not in used_old and new_name not in used_new:
                renames[old_name] = new_name
                used_old.add(old_name)
                used_new.add(new_name)

        return renames

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def _get_func_source(node: ast.AST, full_source: str) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ast.dump(node)
        lines = full_source.splitlines()
        start = node.lineno - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])

    def detect_module_changes(
        self,
        old_sources: Dict[str, str],
        new_sources: Dict[str, str],
    ) -> List[Change]:
        all_changes: List[Change] = []
        all_modules = set(old_sources.keys()) | set(new_sources.keys())
        for mod in sorted(all_modules):
            old_src = old_sources.get(mod, "")
            new_src = new_sources.get(mod, "")
            if old_src == new_src:
                continue
            all_changes.extend(self.detect_changes(old_src, new_src, mod))
        return all_changes


# ---------------------------------------------------------------------------
# InvalidationResult
# ---------------------------------------------------------------------------

@dataclass
class InvalidationResult:
    """Result of predicate-sensitive invalidation."""
    functions_to_reanalyze: Set[str] = field(default_factory=set)
    functions_kept: Set[str] = field(default_factory=set)
    syntactic_would_invalidate: Set[str] = field(default_factory=set)
    predicates_affected: PredicateSet = field(default_factory=PredicateSet)
    savings: int = 0
    total_downstream: int = 0

    @property
    def savings_ratio(self) -> float:
        if self.total_downstream == 0:
            return 0.0
        return self.savings / self.total_downstream

    def summary(self) -> str:
        return (
            f"Reanalyze: {len(self.functions_to_reanalyze)}, "
            f"Kept: {len(self.functions_kept)}, "
            f"Savings: {self.savings}/{self.total_downstream} "
            f"({self.savings_ratio:.1%})"
        )


# ---------------------------------------------------------------------------
# PredicateSensitiveInvalidator
# ---------------------------------------------------------------------------

class PredicateSensitiveInvalidator:
    """
    The key algorithm: predicate-sensitive invalidation.

    When a function f changes, compute ΔP = old_predicates(f) △ new_predicates(f).
    For each downstream function g reachable from f, invalidate g only if the
    edge from f to g (or any intermediate edge) references a predicate in ΔP.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        summary_predicates: Optional[Dict[str, PredicateSet]] = None,
    ) -> None:
        self._graph = graph
        self._summary_predicates: Dict[str, PredicateSet] = summary_predicates or {}

    def set_summary_predicates(self, func_id: str, predicates: PredicateSet) -> None:
        self._summary_predicates[func_id] = predicates

    def get_summary_predicates(self, func_id: str) -> PredicateSet:
        return self._summary_predicates.get(func_id, PredicateSet())

    def invalidate(self, change: Change) -> InvalidationResult:
        result = InvalidationResult()
        delta_p = change.delta_predicates
        result.predicates_affected = delta_p

        all_downstream = self._graph.get_transitive_dependents(change.func_id)
        result.total_downstream = len(all_downstream)
        result.syntactic_would_invalidate = set(all_downstream)

        if change.kind in (ChangeKind.ADDED, ChangeKind.REMOVED, ChangeKind.SIGNATURE_CHANGED):
            result.functions_to_reanalyze = set(all_downstream)
            result.functions_to_reanalyze.add(change.func_id)
            result.savings = 0
            return result

        if delta_p.is_empty():
            result.functions_to_reanalyze = {change.func_id}
            result.functions_kept = set(all_downstream)
            result.savings = len(all_downstream)
            return result

        to_reanalyze: Set[str] = {change.func_id}
        visited: Set[str] = set()
        queue: Deque[str] = deque([change.func_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for succ, edge_meta in self._graph.get_successors_with_edges(current):
                if succ in to_reanalyze:
                    continue
                edge_predicates = edge_meta.predicate_set
                succ_predicates = self.get_summary_predicates(succ)
                combined = edge_predicates.union(succ_predicates)

                if combined.overlaps(delta_p):
                    to_reanalyze.add(succ)
                    queue.append(succ)

        result.functions_to_reanalyze = to_reanalyze
        result.functions_kept = all_downstream - to_reanalyze
        result.savings = len(result.functions_kept)
        return result

    def invalidate_batch(self, changes: List[Change]) -> InvalidationResult:
        combined = InvalidationResult()
        for change in changes:
            single = self.invalidate(change)
            combined.functions_to_reanalyze |= single.functions_to_reanalyze
            combined.syntactic_would_invalidate |= single.syntactic_would_invalidate
            combined.predicates_affected = combined.predicates_affected.union(
                single.predicates_affected
            )

        all_downstream: Set[str] = set()
        for change in changes:
            all_downstream |= self._graph.get_transitive_dependents(change.func_id)
        combined.total_downstream = len(all_downstream)
        combined.functions_kept = all_downstream - combined.functions_to_reanalyze
        combined.savings = len(combined.functions_kept)
        return combined

    def compare_with_syntactic(self, change: Change) -> Dict[str, Any]:
        result = self.invalidate(change)
        return {
            "predicate_sensitive_count": len(result.functions_to_reanalyze),
            "syntactic_count": len(result.syntactic_would_invalidate),
            "savings": result.savings,
            "savings_ratio": result.savings_ratio,
            "predicates_affected": result.predicates_affected.size(),
        }


# ---------------------------------------------------------------------------
# SummaryCache
# ---------------------------------------------------------------------------

@dataclass
class CachedSummary:
    """A cached function analysis summary."""
    func_id: str
    summary: Dict[str, Any]
    fingerprint: FunctionFingerprint
    timestamp: float = 0.0
    access_count: int = 0
    last_access: float = 0.0

    def touch(self) -> None:
        self.access_count += 1
        self.last_access = time.time()


class SummaryCache:
    """Caches function analysis summaries with LRU eviction."""

    def __init__(
        self,
        max_entries: int = 10000,
        max_size_bytes: int = 100 * 1024 * 1024,
        persist_path: Optional[str] = None,
    ) -> None:
        self._cache: OrderedDict[str, CachedSummary] = OrderedDict()
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._persist_path = persist_path
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def store(self, func_id: str, summary: Dict[str, Any], fingerprint: FunctionFingerprint) -> None:
        with self._lock:
            entry = CachedSummary(
                func_id=func_id,
                summary=summary,
                fingerprint=fingerprint,
                timestamp=time.time(),
            )
            if func_id in self._cache:
                self._cache.move_to_end(func_id)
            self._cache[func_id] = entry
            self._evict_if_needed()

    def retrieve(self, func_id: str) -> Optional[Tuple[Dict[str, Any], FunctionFingerprint]]:
        with self._lock:
            entry = self._cache.get(func_id)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            entry.touch()
            self._cache.move_to_end(func_id)
            return entry.summary, entry.fingerprint

    def invalidate(self, func_id: str) -> bool:
        with self._lock:
            if func_id in self._cache:
                del self._cache[func_id]
                return True
            return False

    def invalidate_all(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def invalidate_by_module(self, module_path: str) -> int:
        with self._lock:
            to_remove = [k for k in self._cache if k.startswith(module_path + ":")]
            for k in to_remove:
                del self._cache[k]
            return len(to_remove)

    def contains(self, func_id: str) -> bool:
        return func_id in self._cache

    def size(self) -> int:
        return len(self._cache)

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            self._evictions += 1

    def get_statistics(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "evictions": self._evictions,
        }

    def persist(self) -> None:
        if not self._persist_path:
            return
        with self._lock:
            data: Dict[str, Any] = {}
            for func_id, entry in self._cache.items():
                data[func_id] = {
                    "summary": entry.summary,
                    "fingerprint": entry.fingerprint.to_dict(),
                    "timestamp": entry.timestamp,
                }
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._persist_path)

    def load(self) -> int:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return 0
        with self._lock:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            count = 0
            for func_id, entry_data in data.items():
                fp = FunctionFingerprint.from_dict(entry_data["fingerprint"])
                self.store(func_id, entry_data["summary"], fp)
                count += 1
            return count

    def get_all_entries(self) -> List[CachedSummary]:
        with self._lock:
            return list(self._cache.values())

    def get_oldest_entries(self, n: int) -> List[CachedSummary]:
        with self._lock:
            entries = list(self._cache.values())
            entries.sort(key=lambda e: e.timestamp)
            return entries[:n]

    def get_least_used_entries(self, n: int) -> List[CachedSummary]:
        with self._lock:
            entries = list(self._cache.values())
            entries.sort(key=lambda e: e.access_count)
            return entries[:n]


# ---------------------------------------------------------------------------
# DeltaPropagation
# ---------------------------------------------------------------------------

@dataclass
class Delta:
    """Represents the difference between two summaries."""
    func_id: str
    added_predicates: PredicateSet = field(default_factory=PredicateSet)
    removed_predicates: PredicateSet = field(default_factory=PredicateSet)
    added_types: Dict[str, str] = field(default_factory=dict)
    removed_types: Dict[str, str] = field(default_factory=dict)
    changed_contracts: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    added_bugs: List[Dict[str, Any]] = field(default_factory=list)
    removed_bugs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            self.added_predicates.is_empty()
            and self.removed_predicates.is_empty()
            and not self.added_types
            and not self.removed_types
            and not self.changed_contracts
            and not self.added_bugs
            and not self.removed_bugs
        )

    def merge(self, other: Delta) -> Delta:
        return Delta(
            func_id=self.func_id,
            added_predicates=self.added_predicates.union(other.added_predicates),
            removed_predicates=self.removed_predicates.union(other.removed_predicates),
            added_types={**self.added_types, **other.added_types},
            removed_types={**self.removed_types, **other.removed_types},
            changed_contracts={**self.changed_contracts, **other.changed_contracts},
            added_bugs=self.added_bugs + other.added_bugs,
            removed_bugs=self.removed_bugs + other.removed_bugs,
        )


class DeltaPropagation:
    """Semi-naive delta propagation for incremental fixpoint computation."""

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph
        self._pending_deltas: Dict[str, List[Delta]] = defaultdict(list)
        self._accumulated: Dict[str, Delta] = {}

    def compute_delta(
        self,
        func_id: str,
        old_summary: Optional[Dict[str, Any]],
        new_summary: Dict[str, Any],
    ) -> Delta:
        delta = Delta(func_id=func_id)

        if old_summary is None:
            for key, val in new_summary.items():
                if key == "predicates" and isinstance(val, list):
                    for p_data in val:
                        if isinstance(p_data, dict):
                            delta.added_predicates.add(Predicate.from_dict(p_data))
                elif key == "types" and isinstance(val, dict):
                    delta.added_types.update(val)
                elif key == "bugs" and isinstance(val, list):
                    delta.added_bugs.extend(val)
            return delta

        old_preds_raw = old_summary.get("predicates", [])
        new_preds_raw = new_summary.get("predicates", [])
        old_preds = PredicateSet()
        new_preds = PredicateSet()
        for p in old_preds_raw:
            if isinstance(p, dict):
                old_preds.add(Predicate.from_dict(p))
        for p in new_preds_raw:
            if isinstance(p, dict):
                new_preds.add(Predicate.from_dict(p))
        delta.added_predicates = new_preds.difference(old_preds)
        delta.removed_predicates = old_preds.difference(new_preds)

        old_types = old_summary.get("types", {})
        new_types = new_summary.get("types", {})
        if isinstance(old_types, dict) and isinstance(new_types, dict):
            for k, v in new_types.items():
                if k not in old_types:
                    delta.added_types[k] = v
            for k, v in old_types.items():
                if k not in new_types:
                    delta.removed_types[k] = v

        old_bugs = old_summary.get("bugs", [])
        new_bugs = new_summary.get("bugs", [])
        if isinstance(old_bugs, list) and isinstance(new_bugs, list):
            old_bug_strs = {json.dumps(b, sort_keys=True) for b in old_bugs}
            new_bug_strs = {json.dumps(b, sort_keys=True) for b in new_bugs}
            for b in new_bugs:
                if json.dumps(b, sort_keys=True) not in old_bug_strs:
                    delta.added_bugs.append(b)
            for b in old_bugs:
                if json.dumps(b, sort_keys=True) not in new_bug_strs:
                    delta.removed_bugs.append(b)

        return delta

    def propagate_delta(self, delta: Delta) -> Dict[str, Delta]:
        affected: Dict[str, Delta] = {}
        if delta.is_empty:
            return affected

        queue: Deque[Tuple[str, Delta]] = deque()
        for succ in self._graph.get_successors(delta.func_id):
            queue.append((succ, delta))

        visited: Set[str] = set()
        while queue:
            func_id, incoming_delta = queue.popleft()
            if func_id in visited:
                if func_id in affected:
                    affected[func_id] = affected[func_id].merge(incoming_delta)
                continue
            visited.add(func_id)

            propagated = Delta(func_id=func_id)
            propagated.added_predicates = incoming_delta.added_predicates.copy()
            propagated.removed_predicates = incoming_delta.removed_predicates.copy()
            propagated.added_types = dict(incoming_delta.added_types)
            propagated.removed_types = dict(incoming_delta.removed_types)
            propagated.added_bugs = list(incoming_delta.added_bugs)
            propagated.removed_bugs = list(incoming_delta.removed_bugs)

            if func_id in affected:
                affected[func_id] = affected[func_id].merge(propagated)
            else:
                affected[func_id] = propagated

            for succ in self._graph.get_successors(func_id):
                if succ not in visited:
                    queue.append((succ, propagated))

        return affected

    def propagate_batch(self, deltas: List[Delta]) -> Dict[str, Delta]:
        combined: Dict[str, Delta] = {}
        for delta in deltas:
            affected = self.propagate_delta(delta)
            for func_id, d in affected.items():
                if func_id in combined:
                    combined[func_id] = combined[func_id].merge(d)
                else:
                    combined[func_id] = d
        return combined

    def accumulate(self, delta: Delta) -> None:
        if delta.func_id in self._accumulated:
            self._accumulated[delta.func_id] = self._accumulated[delta.func_id].merge(delta)
        else:
            self._accumulated[delta.func_id] = delta

    def flush_accumulated(self) -> Dict[str, Delta]:
        result = dict(self._accumulated)
        self._accumulated.clear()
        return result

    def get_accumulated(self, func_id: str) -> Optional[Delta]:
        return self._accumulated.get(func_id)


# ---------------------------------------------------------------------------
# StratifiedEvaluation for Horn clauses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HornLiteral:
    """A literal in a Horn clause."""
    predicate: str
    args: Tuple[str, ...]
    negated: bool = False

    def __repr__(self) -> str:
        neg = "¬" if self.negated else ""
        return f"{neg}{self.predicate}({', '.join(self.args)})"


@dataclass
class HornClause:
    """A Horn clause: head :- body1, body2, ..., bodyN."""
    head: HornLiteral
    body: List[HornLiteral] = field(default_factory=list)
    clause_id: str = ""

    def has_negation(self) -> bool:
        return any(lit.negated for lit in self.body)

    def positive_body(self) -> List[HornLiteral]:
        return [lit for lit in self.body if not lit.negated]

    def negative_body(self) -> List[HornLiteral]:
        return [lit for lit in self.body if lit.negated]

    def body_predicates(self) -> Set[str]:
        return {lit.predicate for lit in self.body}

    def __repr__(self) -> str:
        body_str = ", ".join(str(lit) for lit in self.body)
        return f"{self.head} :- {body_str}"


@dataclass
class Stratum:
    """A stratum in stratified evaluation."""
    level: int
    clauses: List[HornClause] = field(default_factory=list)
    predicates: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        return f"Stratum(level={self.level}, predicates={self.predicates})"


class StratifiedEvaluation:
    """Stratified evaluation for a system of Horn clauses."""

    def __init__(self) -> None:
        self._clauses: List[HornClause] = []
        self._strata: List[Stratum] = []
        self._facts: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)

    def add_clause(self, clause: HornClause) -> None:
        self._clauses.append(clause)

    def add_fact(self, predicate: str, args: Tuple[str, ...]) -> None:
        self._facts[predicate].add(args)

    def clear_facts(self) -> None:
        self._facts.clear()

    def stratify(self) -> List[Stratum]:
        pred_to_clauses: Dict[str, List[HornClause]] = defaultdict(list)
        for clause in self._clauses:
            pred_to_clauses[clause.head.predicate].append(clause)

        all_predicates: Set[str] = set()
        for clause in self._clauses:
            all_predicates.add(clause.head.predicate)
            for lit in clause.body:
                all_predicates.add(lit.predicate)

        dependency: Dict[str, Set[str]] = defaultdict(set)
        neg_dependency: Dict[str, Set[str]] = defaultdict(set)
        for clause in self._clauses:
            head_pred = clause.head.predicate
            for lit in clause.body:
                if lit.negated:
                    neg_dependency[head_pred].add(lit.predicate)
                else:
                    dependency[head_pred].add(lit.predicate)

        stratum_of: Dict[str, int] = {p: 0 for p in all_predicates}
        changed = True
        max_iter = len(all_predicates) + 1
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for p in all_predicates:
                for dep in dependency.get(p, set()):
                    if stratum_of[dep] > stratum_of[p]:
                        stratum_of[p] = stratum_of[dep]
                        changed = True
                for dep in neg_dependency.get(p, set()):
                    needed = stratum_of[dep] + 1
                    if needed > stratum_of[p]:
                        stratum_of[p] = needed
                        changed = True

        if iteration >= max_iter:
            pass  # not stratifiable; proceed with best effort

        max_stratum = max(stratum_of.values()) if stratum_of else 0
        strata: List[Stratum] = []
        for level in range(max_stratum + 1):
            preds_at_level = {p for p, s in stratum_of.items() if s == level}
            clauses_at_level = [
                c for c in self._clauses if c.head.predicate in preds_at_level
            ]
            strata.append(Stratum(
                level=level,
                clauses=clauses_at_level,
                predicates=preds_at_level,
            ))

        self._strata = strata
        return strata

    def evaluate(self) -> Dict[str, Set[Tuple[str, ...]]]:
        if not self._strata:
            self.stratify()

        derived: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
        for pred, facts in self._facts.items():
            derived[pred] = set(facts)

        for stratum in self._strata:
            self._evaluate_stratum(stratum, derived)

        return dict(derived)

    def _evaluate_stratum(
        self,
        stratum: Stratum,
        derived: Dict[str, Set[Tuple[str, ...]]],
    ) -> None:
        changed = True
        max_iter = 1000
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for clause in stratum.clauses:
                new_facts = self._evaluate_clause(clause, derived)
                for fact in new_facts:
                    if fact not in derived[clause.head.predicate]:
                        derived[clause.head.predicate].add(fact)
                        changed = True

    def _evaluate_clause(
        self,
        clause: HornClause,
        derived: Dict[str, Set[Tuple[str, ...]]],
    ) -> Set[Tuple[str, ...]]:
        if not clause.body:
            return {clause.head.args}

        bindings: List[Dict[str, str]] = [{}]
        for lit in clause.positive_body():
            new_bindings: List[Dict[str, str]] = []
            for binding in bindings:
                for fact in derived.get(lit.predicate, set()):
                    new_binding = self._unify(lit.args, fact, binding)
                    if new_binding is not None:
                        new_bindings.append(new_binding)
            bindings = new_bindings

        for lit in clause.negative_body():
            filtered: List[Dict[str, str]] = []
            for binding in bindings:
                bound_args = tuple(binding.get(a, a) for a in lit.args)
                if bound_args not in derived.get(lit.predicate, set()):
                    filtered.append(binding)
            bindings = filtered

        results: Set[Tuple[str, ...]] = set()
        for binding in bindings:
            head_args = tuple(binding.get(a, a) for a in clause.head.args)
            results.add(head_args)
        return results

    @staticmethod
    def _unify(
        pattern: Tuple[str, ...],
        fact: Tuple[str, ...],
        binding: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if len(pattern) != len(fact):
            return None
        new_binding = dict(binding)
        for p, f in zip(pattern, fact):
            if p.startswith("?"):
                if p in new_binding:
                    if new_binding[p] != f:
                        return None
                else:
                    new_binding[p] = f
            elif p != f:
                return None
        return new_binding

    def incremental_add_fact(
        self,
        predicate: str,
        args: Tuple[str, ...],
        derived: Dict[str, Set[Tuple[str, ...]]],
    ) -> Dict[str, Set[Tuple[str, ...]]]:
        new_derived: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
        if args in derived.get(predicate, set()):
            return dict(new_derived)
        derived[predicate].add(args)
        new_derived[predicate].add(args)

        changed = True
        max_iter = 100
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for stratum in self._strata:
                for clause in stratum.clauses:
                    results = self._evaluate_clause(clause, derived)
                    for fact in results:
                        if fact not in derived[clause.head.predicate]:
                            derived[clause.head.predicate].add(fact)
                            new_derived[clause.head.predicate].add(fact)
                            changed = True
        return dict(new_derived)

    def restratify_after_change(self, added_clauses: List[HornClause], removed_clause_ids: Set[str]) -> List[Stratum]:
        self._clauses = [c for c in self._clauses if c.clause_id not in removed_clause_ids]
        self._clauses.extend(added_clauses)
        return self.stratify()


# ---------------------------------------------------------------------------
# IncrementalStatistics
# ---------------------------------------------------------------------------

@dataclass
class IncrementalStatistics:
    """Statistics about an incremental analysis run."""
    total_functions: int = 0
    functions_reanalyzed: int = 0
    functions_cached: int = 0
    predicates_invalidated: int = 0
    predicates_total: int = 0
    time_incremental_ms: float = 0.0
    time_full_estimate_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    convergence_iterations: int = 0
    changes_detected: int = 0
    modules_affected: int = 0
    sccs_reanalyzed: int = 0

    @property
    def functions_saved(self) -> int:
        return self.total_functions - self.functions_reanalyzed

    @property
    def savings_ratio(self) -> float:
        if self.total_functions == 0:
            return 0.0
        return self.functions_saved / self.total_functions

    @property
    def speedup(self) -> float:
        if self.time_incremental_ms == 0:
            return 0.0
        return self.time_full_estimate_ms / self.time_incremental_ms

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def summary(self) -> str:
        return (
            f"Incremental analysis: "
            f"{self.functions_reanalyzed}/{self.total_functions} functions reanalyzed "
            f"({self.savings_ratio:.1%} saved), "
            f"{self.convergence_iterations} iterations, "
            f"cache hit rate {self.cache_hit_rate:.1%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "functions_reanalyzed": self.functions_reanalyzed,
            "functions_cached": self.functions_cached,
            "predicates_invalidated": self.predicates_invalidated,
            "predicates_total": self.predicates_total,
            "time_incremental_ms": self.time_incremental_ms,
            "time_full_estimate_ms": self.time_full_estimate_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "convergence_iterations": self.convergence_iterations,
            "changes_detected": self.changes_detected,
            "modules_affected": self.modules_affected,
            "sccs_reanalyzed": self.sccs_reanalyzed,
            "functions_saved": self.functions_saved,
            "savings_ratio": self.savings_ratio,
            "speedup": self.speedup,
            "cache_hit_rate": self.cache_hit_rate,
        }


# ---------------------------------------------------------------------------
# AnalysisPriority
# ---------------------------------------------------------------------------

class AnalysisPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class AnalysisTask:
    """A single analysis task in the priority queue."""
    func_id: str
    priority: AnalysisPriority = AnalysisPriority.NORMAL
    reason: str = ""
    dependencies_ready: bool = True
    attempt: int = 0
    max_attempts: int = 3

    def __lt__(self, other: AnalysisTask) -> bool:
        return self.priority.value < other.priority.value


# ---------------------------------------------------------------------------
# IncrementalAnalysisEngine
# ---------------------------------------------------------------------------

class IncrementalAnalysisEngine:
    """
    Orchestrates incremental re-analysis using predicate-sensitive invalidation.

    Phases:
    1. Detect changes between old and new source
    2. Invalidate affected summaries using predicate-sensitivity
    3. Schedule re-analysis in dependency order
    4. Run analysis, propagating deltas
    5. Merge results and update caches
    """

    def __init__(
        self,
        graph: DependencyGraph,
        cache: SummaryCache,
        analyze_func: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._graph = graph
        self._cache = cache
        self._analyze_func = analyze_func or self._default_analyze
        self._invalidator = PredicateSensitiveInvalidator(graph)
        self._delta_prop = DeltaPropagation(graph)
        self._detector = ChangeDetector()
        self._statistics = IncrementalStatistics()
        self._lock = threading.RLock()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._max_iterations = 50
        self._converged = False

    @staticmethod
    def _default_analyze(
        func_id: str, source: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        fp = FunctionFingerprint.compute_fingerprint(source, func_id)
        return {
            "func_id": func_id,
            "predicates": fp.predicate_set.to_dict(),
            "types": {},
            "bugs": [],
            "fingerprint": fp.to_dict(),
        }

    def analyze_incrementally(
        self,
        old_sources: Dict[str, str],
        new_sources: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        start_time = time.time()
        self._statistics = IncrementalStatistics()
        self._statistics.total_functions = self._graph.node_count()

        # Phase 1: detect changes
        changes = self._detector.detect_module_changes(old_sources, new_sources)
        self._statistics.changes_detected = len(changes)
        affected_modules: Set[str] = set()
        for c in changes:
            affected_modules.add(c.module_path)
        self._statistics.modules_affected = len(affected_modules)

        if not changes:
            self._statistics.time_incremental_ms = (time.time() - start_time) * 1000
            return dict(self._results)

        # Phase 2: invalidate
        invalidation = self._invalidator.invalidate_batch(changes)
        self._statistics.functions_reanalyzed = len(invalidation.functions_to_reanalyze)
        self._statistics.predicates_invalidated = invalidation.predicates_affected.size()
        for func_id in invalidation.functions_to_reanalyze:
            self._cache.invalidate(func_id)

        # Phase 3: schedule
        tasks = self._schedule_tasks(invalidation.functions_to_reanalyze, changes)

        # Phase 4: analyze
        self._run_analysis(tasks, new_sources)

        # Phase 5: merge
        stats = self._cache.get_statistics()
        self._statistics.cache_hits = stats["hits"]
        self._statistics.cache_misses = stats["misses"]
        self._statistics.functions_cached = stats["entries"]
        self._statistics.time_incremental_ms = (time.time() - start_time) * 1000
        self._statistics.time_full_estimate_ms = (
            self._statistics.time_incremental_ms
            * self._statistics.total_functions
            / max(self._statistics.functions_reanalyzed, 1)
        )

        return dict(self._results)

    def _schedule_tasks(
        self, to_reanalyze: Set[str], changes: List[Change]
    ) -> List[AnalysisTask]:
        changed_ids = {c.func_id for c in changes}
        topo = self._graph.topological_sort()
        topo_order = {fid: i for i, fid in enumerate(topo)}

        tasks: List[AnalysisTask] = []
        for func_id in to_reanalyze:
            dep_count = len(self._graph.get_transitive_dependents(func_id))
            if func_id in changed_ids:
                priority = AnalysisPriority.HIGH
            elif dep_count > 10:
                priority = AnalysisPriority.CRITICAL
            elif dep_count > 3:
                priority = AnalysisPriority.NORMAL
            else:
                priority = AnalysisPriority.LOW

            tasks.append(AnalysisTask(
                func_id=func_id,
                priority=priority,
                reason="changed" if func_id in changed_ids else "invalidated",
            ))

        tasks.sort(key=lambda t: (t.priority.value, topo_order.get(t.func_id, 999999)))
        return tasks

    def _run_analysis(
        self, tasks: List[AnalysisTask], sources: Dict[str, str]
    ) -> None:
        iteration = 0
        remaining = list(tasks)

        while remaining and iteration < self._max_iterations:
            iteration += 1
            next_round: List[AnalysisTask] = []

            for task in remaining:
                module_path = task.func_id.rsplit(":", 1)[0] if ":" in task.func_id else ""
                source = sources.get(module_path, "")
                callee_summaries: Dict[str, Any] = {}
                for dep in self._graph.get_predecessors(task.func_id):
                    if dep in self._results:
                        callee_summaries[dep] = self._results[dep]

                context = {
                    "callee_summaries": callee_summaries,
                    "iteration": iteration,
                }

                try:
                    old_result = self._results.get(task.func_id)
                    new_result = self._analyze_func(task.func_id, source, context)
                    self._results[task.func_id] = new_result

                    fp = FunctionFingerprint.compute_fingerprint(source, task.func_id)
                    self._cache.store(task.func_id, new_result, fp)

                    delta = self._delta_prop.compute_delta(task.func_id, old_result, new_result)
                    if not delta.is_empty:
                        affected = self._delta_prop.propagate_delta(delta)
                        for affected_id in affected:
                            if affected_id not in {t.func_id for t in next_round}:
                                next_round.append(AnalysisTask(
                                    func_id=affected_id,
                                    priority=AnalysisPriority.NORMAL,
                                    reason="delta_propagation",
                                ))
                except Exception:
                    task.attempt += 1
                    if task.attempt < task.max_attempts:
                        next_round.append(task)

            remaining = next_round

        self._statistics.convergence_iterations = iteration
        self._converged = not remaining

    def get_statistics(self) -> IncrementalStatistics:
        return self._statistics

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._results)

    @property
    def converged(self) -> bool:
        return self._converged


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------

@dataclass
class FileChangeEvent:
    """A filesystem change event."""
    path: str
    kind: str  # "created", "modified", "deleted"
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return f"FileChangeEvent({self.kind}, {self.path})"


class FileWatcher:
    """
    Watches filesystem for changes and feeds them into the incremental engine.
    Uses polling-based approach for portability.
    """

    def __init__(
        self,
        root_dir: str,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        debounce_ms: int = 500,
        poll_interval_ms: int = 1000,
    ) -> None:
        self._root = root_dir
        self._patterns = patterns or ["*.py"]
        self._exclude_patterns = exclude_patterns or [
            "__pycache__/*", "*.pyc", ".git/*", "*.egg-info/*",
            "node_modules/*", ".tox/*", ".venv/*", "venv/*",
        ]
        self._debounce_ms = debounce_ms
        self._poll_interval_ms = poll_interval_ms
        self._file_mtimes: Dict[str, float] = {}
        self._event_queue: Deque[FileChangeEvent] = deque()
        self._callbacks: List[Callable[[List[FileChangeEvent]], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending_events: List[FileChangeEvent] = []
        self._last_flush_time: float = 0.0

    def add_callback(self, callback: Callable[[List[FileChangeEvent]], None]) -> None:
        self._callbacks.append(callback)

    def _matches_pattern(self, path: str) -> bool:
        rel = os.path.relpath(path, self._root)
        for pat in self._exclude_patterns:
            if fnmatch.fnmatch(rel, pat):
                return False
        for pat in self._patterns:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _scan_files(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [
                d for d in dirnames
                if not any(fnmatch.fnmatch(d, p.rstrip("/*")) for p in self._exclude_patterns)
            ]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if self._matches_pattern(fpath):
                    try:
                        result[fpath] = os.path.getmtime(fpath)
                    except OSError:
                        pass
        return result

    def _check_changes(self) -> List[FileChangeEvent]:
        events: List[FileChangeEvent] = []
        current = self._scan_files()
        now = time.time()

        for path, mtime in current.items():
            if path not in self._file_mtimes:
                events.append(FileChangeEvent(path=path, kind="created", timestamp=now))
            elif mtime > self._file_mtimes[path]:
                events.append(FileChangeEvent(path=path, kind="modified", timestamp=now))

        for path in set(self._file_mtimes.keys()) - set(current.keys()):
            events.append(FileChangeEvent(path=path, kind="deleted", timestamp=now))

        self._file_mtimes = current
        return events

    def _poll_loop(self) -> None:
        self._file_mtimes = self._scan_files()
        while self._running:
            time.sleep(self._poll_interval_ms / 1000.0)
            events = self._check_changes()
            if events:
                with self._lock:
                    self._pending_events.extend(events)
                    for e in events:
                        self._event_queue.append(e)
                now = time.time()
                if (now - self._last_flush_time) * 1000 >= self._debounce_ms:
                    self._flush_events()

    def _flush_events(self) -> None:
        with self._lock:
            if not self._pending_events:
                return
            events = list(self._pending_events)
            self._pending_events.clear()
            self._last_flush_time = time.time()
        for cb in self._callbacks:
            try:
                cb(events)
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def poll_once(self) -> List[FileChangeEvent]:
        events = self._check_changes()
        for e in events:
            self._event_queue.append(e)
        return events

    def drain_events(self) -> List[FileChangeEvent]:
        with self._lock:
            events = list(self._event_queue)
            self._event_queue.clear()
            return events

    def get_watched_files(self) -> List[str]:
        return list(self._file_mtimes.keys())

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# IntegrationController — ties everything together for dev mode
# ---------------------------------------------------------------------------

class IntegrationController:
    """
    High-level controller that connects file watching, change detection,
    predicate-sensitive invalidation, and incremental analysis.
    """

    def __init__(
        self,
        root_dir: str,
        analyze_func: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._root = root_dir
        self._graph = DependencyGraph()
        self._cache = SummaryCache()
        self._engine = IncrementalAnalysisEngine(self._graph, self._cache, analyze_func)
        self._watcher = FileWatcher(root_dir)
        self._sources: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._run_count = 0
        self._all_statistics: List[IncrementalStatistics] = []

    def initialize(self) -> None:
        self._sources = self._load_sources()
        self._build_initial_graph()

    def _load_sources(self) -> Dict[str, str]:
        sources: Dict[str, str] = {}
        for dirpath, _, filenames in os.walk(self._root):
            for fname in filenames:
                if fname.endswith(".py"):
                    fpath = os.path.join(dirpath, fname)
                    rel = os.path.relpath(fpath, self._root)
                    try:
                        with open(fpath, "r") as f:
                            sources[rel] = f.read()
                    except OSError:
                        pass
        return sources

    def _build_initial_graph(self) -> None:
        for mod_path, source in self._sources.items():
            funcs = ASTDiffer.extract_functions(source)
            for func_name in funcs:
                func_id = f"{mod_path}:{func_name}"
                fp = FunctionFingerprint.compute_fingerprint(
                    ChangeDetector._get_func_source(funcs[func_name], source),
                    func_name,
                )
                self._graph.add_node(func_id, NodeMetadata(
                    module_path=mod_path,
                    function_name=func_name,
                    predicate_set=fp.predicate_set,
                    source_hash=fp.source_hash,
                ))

    def on_file_changes(self, events: List[FileChangeEvent]) -> Optional[IncrementalStatistics]:
        with self._lock:
            new_sources = dict(self._sources)
            for event in events:
                rel = os.path.relpath(event.path, self._root)
                if event.kind == "deleted":
                    new_sources.pop(rel, None)
                else:
                    try:
                        with open(event.path, "r") as f:
                            new_sources[rel] = f.read()
                    except OSError:
                        pass

            self._engine.analyze_incrementally(self._sources, new_sources)
            stats = self._engine.get_statistics()
            self._all_statistics.append(stats)
            self._sources = new_sources
            self._run_count += 1
            return stats

    def start_watching(self) -> None:
        self.initialize()
        self._watcher.add_callback(self.on_file_changes)
        self._watcher.start()

    def stop_watching(self) -> None:
        self._watcher.stop()

    def get_graph(self) -> DependencyGraph:
        return self._graph

    def get_cache(self) -> SummaryCache:
        return self._cache

    def get_run_count(self) -> int:
        return self._run_count

    def get_all_statistics(self) -> List[IncrementalStatistics]:
        return list(self._all_statistics)

    def get_cumulative_statistics(self) -> Dict[str, Any]:
        if not self._all_statistics:
            return {}
        total_reanalyzed = sum(s.functions_reanalyzed for s in self._all_statistics)
        total_functions = sum(s.total_functions for s in self._all_statistics)
        total_time = sum(s.time_incremental_ms for s in self._all_statistics)
        return {
            "runs": len(self._all_statistics),
            "total_reanalyzed": total_reanalyzed,
            "total_functions_across_runs": total_functions,
            "total_time_ms": total_time,
            "avg_savings_ratio": (
                sum(s.savings_ratio for s in self._all_statistics) / len(self._all_statistics)
            ),
        }


# ---------------------------------------------------------------------------
# Predicate extraction from Python source
# ---------------------------------------------------------------------------

class PredicateExtractor(ast.NodeVisitor):
    """Extract refinement predicates from Python source code."""

    def __init__(self) -> None:
        self._predicates: List[Predicate] = []
        self._current_function: Optional[str] = None

    def extract(self, source: str) -> PredicateSet:
        self._predicates = []
        try:
            tree = ast.parse(source)
            self.visit(tree)
        except SyntaxError:
            pass
        return PredicateSet(set(self._predicates))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        self._extract_from_test(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._extract_from_test(node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._extract_from_test(node.test)
        self.generic_visit(node)

    def _extract_from_test(self, node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name == "isinstance":
                    if len(node.args) >= 2:
                        type_arg = ast.dump(node.args[1])
                        self._predicates.append(
                            Predicate(PredicateKind.ISINSTANCE, "isinstance", (type_arg,))
                        )
                elif name == "callable":
                    self._predicates.append(
                        Predicate(PredicateKind.CALLABLE_CHECK, "callable")
                    )
                elif name == "hasattr":
                    if len(node.args) >= 2:
                        attr = ast.dump(node.args[1])
                        self._predicates.append(
                            Predicate(PredicateKind.ATTRIBUTE_CHECK, "hasattr", (attr,))
                        )
                elif name == "len":
                    self._predicates.append(Predicate(PredicateKind.LENGTH, "len"))
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr == "startswith":
                    self._predicates.append(
                        Predicate(PredicateKind.CUSTOM, "str.startswith")
                    )
                elif node.func.attr == "endswith":
                    self._predicates.append(
                        Predicate(PredicateKind.CUSTOM, "str.endswith")
                    )
                elif node.func.attr == "match":
                    self._predicates.append(Predicate(PredicateKind.REGEX, "re.match"))
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Is, ast.IsNot)):
                    for comp in node.comparators:
                        if isinstance(comp, ast.Constant) and comp.value is None:
                            self._predicates.append(
                                Predicate(PredicateKind.NULL_CHECK, "is_none")
                            )
                            break
                    else:
                        self._predicates.append(
                            Predicate(PredicateKind.COMPARISON, "is")
                        )
                elif isinstance(op, (ast.In, ast.NotIn)):
                    self._predicates.append(
                        Predicate(PredicateKind.MEMBER, "in")
                    )
                elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    self._predicates.append(
                        Predicate(PredicateKind.RANGE, type(op).__name__)
                    )
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    self._predicates.append(
                        Predicate(PredicateKind.EQUALITY, type(op).__name__)
                    )
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                self._extract_from_test(val)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._extract_from_test(node.operand)
        elif isinstance(node, ast.Name):
            self._predicates.append(
                Predicate(PredicateKind.TRUTHY, node.id)
            )


# ---------------------------------------------------------------------------
# Call graph builder from source
# ---------------------------------------------------------------------------

class CallGraphBuilder(ast.NodeVisitor):
    """Build a dependency graph by walking Python source ASTs."""

    def __init__(self) -> None:
        self._graph = DependencyGraph()
        self._current_module: str = ""
        self._current_function: Optional[str] = None
        self._current_class: Optional[str] = None
        self._extractor = PredicateExtractor()

    def build(self, sources: Dict[str, str]) -> DependencyGraph:
        self._graph = DependencyGraph()
        all_functions: Dict[str, str] = {}

        for mod_path, source in sources.items():
            self._current_module = mod_path
            funcs = ASTDiffer.extract_functions(source)
            for func_name, func_node in funcs.items():
                func_id = f"{mod_path}:{func_name}"
                func_source = ChangeDetector._get_func_source(func_node, source)
                fp = FunctionFingerprint.compute_fingerprint(func_source, func_name)
                pred_set = self._extractor.extract(func_source)

                self._graph.add_node(func_id, NodeMetadata(
                    module_path=mod_path,
                    function_name=func_name,
                    source_hash=fp.source_hash,
                    predicate_set=pred_set,
                    line_start=getattr(func_node, "lineno", 0),
                    line_end=getattr(func_node, "end_lineno", 0),
                    complexity=fp.complexity,
                ))
                all_functions[func_name] = func_id
                if "." in func_name:
                    short = func_name.split(".")[-1]
                    all_functions[short] = func_id

        for mod_path, source in sources.items():
            self._current_module = mod_path
            funcs = ASTDiffer.extract_functions(source)
            for func_name, func_node in funcs.items():
                caller_id = f"{mod_path}:{func_name}"
                calls = self._extract_calls(func_node)
                for callee_name in calls:
                    callee_id = all_functions.get(callee_name)
                    if callee_id and callee_id != caller_id:
                        caller_meta = self._graph.get_node(caller_id)
                        callee_meta = self._graph.get_node(callee_id)
                        edge_preds = PredicateSet()
                        if caller_meta:
                            edge_preds = caller_meta.predicate_set.copy()
                        self._graph.add_edge(caller_id, callee_id, edge_preds)

        return self._graph

    def _extract_calls(self, node: ast.AST) -> Set[str]:
        calls: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        return calls


# ---------------------------------------------------------------------------
# Batch incremental processor
# ---------------------------------------------------------------------------

class BatchIncrementalProcessor:
    """Process multiple file changes as a single batch for efficiency."""

    def __init__(
        self,
        graph: DependencyGraph,
        cache: SummaryCache,
        analyze_func: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._graph = graph
        self._cache = cache
        self._engine = IncrementalAnalysisEngine(graph, cache, analyze_func)
        self._batch: List[Tuple[Dict[str, str], Dict[str, str]]] = []
        self._lock = threading.Lock()

    def add_to_batch(self, old_sources: Dict[str, str], new_sources: Dict[str, str]) -> None:
        with self._lock:
            self._batch.append((old_sources, new_sources))

    def process_batch(self) -> List[IncrementalStatistics]:
        with self._lock:
            batch = list(self._batch)
            self._batch.clear()

        results: List[IncrementalStatistics] = []
        if not batch:
            return results

        merged_old: Dict[str, str] = {}
        merged_new: Dict[str, str] = {}
        for old_src, new_src in batch:
            merged_old.update(old_src)
            merged_new.update(new_src)

        self._engine.analyze_incrementally(merged_old, merged_new)
        results.append(self._engine.get_statistics())
        return results

    def pending_count(self) -> int:
        return len(self._batch)


# ---------------------------------------------------------------------------
# Convergence checker
# ---------------------------------------------------------------------------

class ConvergenceChecker:
    """Checks whether the incremental analysis has converged."""

    def __init__(self, tolerance: float = 0.001, max_iterations: int = 100) -> None:
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._history: List[Dict[str, Any]] = []

    def record_iteration(self, summaries: Dict[str, Dict[str, Any]]) -> None:
        snapshot: Dict[str, str] = {}
        for func_id, summary in summaries.items():
            snapshot[func_id] = hashlib.sha256(
                json.dumps(summary, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        self._history.append(snapshot)

    def has_converged(self) -> bool:
        if len(self._history) < 2:
            return False
        last = self._history[-1]
        prev = self._history[-2]
        if last == prev:
            return True
        if not last or not prev:
            return False
        total = len(last)
        changed = sum(1 for k in last if last.get(k) != prev.get(k))
        return (changed / max(total, 1)) < self._tolerance

    def iterations(self) -> int:
        return len(self._history)

    def exceeded_max(self) -> bool:
        return len(self._history) >= self._max_iterations

    def should_stop(self) -> bool:
        return self.has_converged() or self.exceeded_max()

    def reset(self) -> None:
        self._history.clear()

    def change_ratio(self) -> float:
        if len(self._history) < 2:
            return 1.0
        last = self._history[-1]
        prev = self._history[-2]
        total = max(len(last), 1)
        changed = sum(1 for k in last if last.get(k) != prev.get(k))
        return changed / total


# ---------------------------------------------------------------------------
# SCC-aware analysis scheduler
# ---------------------------------------------------------------------------

class SCCAwareScheduler:
    """
    Schedules analysis respecting strongly connected components.
    Functions in the same SCC are analyzed together until convergence.
    """

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def schedule(self, to_analyze: Set[str]) -> List[List[str]]:
        sccs = self._graph.get_strongly_connected_components()
        scc_map: Dict[str, int] = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                scc_map[node] = i

        relevant_sccs: Dict[int, List[str]] = defaultdict(list)
        for func_id in to_analyze:
            scc_idx = scc_map.get(func_id)
            if scc_idx is not None:
                relevant_sccs[scc_idx].append(func_id)

        scc_graph: Dict[int, Set[int]] = defaultdict(set)
        for func_id in to_analyze:
            src_scc = scc_map.get(func_id, -1)
            for succ in self._graph.get_successors(func_id):
                dst_scc = scc_map.get(succ, -1)
                if dst_scc != src_scc and dst_scc in relevant_sccs:
                    scc_graph[src_scc].add(dst_scc)

        in_degree: Dict[int, int] = {idx: 0 for idx in relevant_sccs}
        for src, dsts in scc_graph.items():
            for dst in dsts:
                if dst in in_degree:
                    in_degree[dst] += 1

        order: List[int] = []
        queue: Deque[int] = deque(idx for idx, deg in in_degree.items() if deg == 0)
        while queue:
            idx = queue.popleft()
            order.append(idx)
            for dst in scc_graph.get(idx, set()):
                if dst in in_degree:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)
        for idx in relevant_sccs:
            if idx not in order:
                order.append(idx)

        return [relevant_sccs[idx] for idx in order if relevant_sccs[idx]]


# ---------------------------------------------------------------------------
# Dependency graph analysis utilities
# ---------------------------------------------------------------------------

class GraphAnalyzer:
    """Various analysis operations on the dependency graph."""

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def compute_fan_in(self) -> Dict[str, int]:
        return {n: len(self._graph.get_predecessors(n)) for n in self._graph.nodes()}

    def compute_fan_out(self) -> Dict[str, int]:
        return {n: len(self._graph.get_successors(n)) for n in self._graph.nodes()}

    def find_hub_functions(self, threshold: int = 5) -> List[str]:
        fan_in = self.compute_fan_in()
        fan_out = self.compute_fan_out()
        hubs: List[str] = []
        for n in self._graph.nodes():
            if fan_in.get(n, 0) >= threshold or fan_out.get(n, 0) >= threshold:
                hubs.append(n)
        return hubs

    def find_bridge_functions(self) -> List[str]:
        bridges: List[str] = []
        for node in self._graph.nodes():
            preds = self._graph.get_predecessors(node)
            succs = self._graph.get_successors(node)
            if preds and succs:
                pred_set = set()
                for p in preds:
                    pred_set |= self._graph.get_transitive_dependencies(p)
                succ_set = set()
                for s in succs:
                    succ_set |= self._graph.get_transitive_dependents(s)
                if not pred_set & succ_set:
                    bridges.append(node)
        return bridges

    def compute_depth(self) -> Dict[str, int]:
        depth: Dict[str, int] = {}
        roots = self._graph.get_roots()
        for r in roots:
            depth[r] = 0
        queue: Deque[str] = deque(roots)
        while queue:
            node = queue.popleft()
            for succ in self._graph.get_successors(node):
                new_depth = depth[node] + 1
                if succ not in depth or depth[succ] < new_depth:
                    depth[succ] = new_depth
                    queue.append(succ)
        return depth

    def compute_predicate_coverage(self) -> Dict[str, float]:
        coverage: Dict[str, float] = {}
        for node in self._graph.nodes():
            meta = self._graph.get_node(node)
            if meta and meta.predicate_set.size() > 0:
                edge_preds = PredicateSet()
                for _, edge_meta in self._graph.get_predecessors_with_edges(node):
                    edge_preds = edge_preds.union(edge_meta.predicate_set)
                for _, edge_meta in self._graph.get_successors_with_edges(node):
                    edge_preds = edge_preds.union(edge_meta.predicate_set)
                overlap = meta.predicate_set.intersection(edge_preds)
                coverage[node] = overlap.size() / meta.predicate_set.size()
            else:
                coverage[node] = 0.0
        return coverage

    def summary(self) -> Dict[str, Any]:
        sccs = self._graph.get_strongly_connected_components()
        cyclic_sccs = [s for s in sccs if len(s) > 1]
        fan_in = self.compute_fan_in()
        fan_out = self.compute_fan_out()
        return {
            "nodes": self._graph.node_count(),
            "edges": self._graph.edge_count(),
            "roots": len(self._graph.get_roots()),
            "leaves": len(self._graph.get_leaves()),
            "sccs": len(sccs),
            "cyclic_sccs": len(cyclic_sccs),
            "max_fan_in": max(fan_in.values()) if fan_in else 0,
            "max_fan_out": max(fan_out.values()) if fan_out else 0,
            "avg_fan_in": sum(fan_in.values()) / max(len(fan_in), 1),
            "avg_fan_out": sum(fan_out.values()) / max(len(fan_out), 1),
        }


# ---------------------------------------------------------------------------
# Predicate impact analyzer
# ---------------------------------------------------------------------------

class PredicateImpactAnalyzer:
    """Analyzes the impact of individual predicates across the dependency graph."""

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def compute_predicate_frequency(self) -> Dict[Predicate, int]:
        freq: Dict[Predicate, int] = defaultdict(int)
        for node in self._graph.nodes():
            meta = self._graph.get_node(node)
            if meta:
                for pred in meta.predicate_set:
                    freq[pred] += 1
        for _, _, edge_meta in self._graph.edges():
            for pred in edge_meta.predicate_set:
                freq[pred] += 1
        return dict(freq)

    def compute_predicate_impact(self) -> Dict[Predicate, int]:
        impact: Dict[Predicate, int] = defaultdict(int)
        for node in self._graph.nodes():
            downstream = self._graph.get_transitive_dependents(node)
            meta = self._graph.get_node(node)
            if meta:
                for pred in meta.predicate_set:
                    impact[pred] += len(downstream)
        return dict(impact)

    def most_impactful_predicates(self, n: int = 10) -> List[Tuple[Predicate, int]]:
        impact = self.compute_predicate_impact()
        sorted_preds = sorted(impact.items(), key=lambda x: -x[1])
        return sorted_preds[:n]

    def predicates_for_function(self, func_id: str) -> PredicateSet:
        result = PredicateSet()
        meta = self._graph.get_node(func_id)
        if meta:
            result = result.union(meta.predicate_set)
        for _, edge_meta in self._graph.get_predecessors_with_edges(func_id):
            result = result.union(edge_meta.predicate_set)
        for _, edge_meta in self._graph.get_successors_with_edges(func_id):
            result = result.union(edge_meta.predicate_set)
        return result


# ---------------------------------------------------------------------------
# Module dependency tracker
# ---------------------------------------------------------------------------

class ModuleDependencyTracker:
    """Tracks dependencies at the module level for coarse-grained invalidation."""

    def __init__(self) -> None:
        self._imports: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)

    def add_import(self, importer: str, imported: str) -> None:
        self._imports[importer].add(imported)
        self._reverse[imported].add(importer)

    def remove_module(self, module: str) -> None:
        for imp in self._imports.get(module, set()):
            self._reverse[imp].discard(module)
        for imp in self._reverse.get(module, set()):
            self._imports[imp].discard(module)
        self._imports.pop(module, None)
        self._reverse.pop(module, None)

    def get_importers(self, module: str) -> Set[str]:
        return set(self._reverse.get(module, set()))

    def get_imports(self, module: str) -> Set[str]:
        return set(self._imports.get(module, set()))

    def get_transitive_importers(self, module: str) -> Set[str]:
        visited: Set[str] = set()
        queue: Deque[str] = deque(self.get_importers(module))
        while queue:
            m = queue.popleft()
            if m in visited:
                continue
            visited.add(m)
            for imp in self.get_importers(m):
                if imp not in visited:
                    queue.append(imp)
        return visited

    def build_from_sources(self, sources: Dict[str, str]) -> None:
        self._imports.clear()
        self._reverse.clear()
        for mod_path, source in sources.items():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        for candidate in sources:
                            if candidate.replace("/", ".").replace(".py", "") == target:
                                self.add_import(mod_path, candidate)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for candidate in sources:
                            mod_name = candidate.replace("/", ".").replace(".py", "")
                            if mod_name == node.module or mod_name.endswith("." + node.module):
                                self.add_import(mod_path, candidate)

    def modules_affected_by(self, changed_modules: Set[str]) -> Set[str]:
        affected: Set[str] = set(changed_modules)
        for mod in changed_modules:
            affected |= self.get_transitive_importers(mod)
        return affected

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: sorted(v) for k, v in self._imports.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> ModuleDependencyTracker:
        tracker = cls()
        for mod, imports in data.items():
            for imp in imports:
                tracker.add_import(mod, imp)
        return tracker


# ---------------------------------------------------------------------------
# IncrementalSession — stateful session for interactive use
# ---------------------------------------------------------------------------

class IncrementalSession:
    """
    A stateful session that maintains analysis state across multiple
    incremental updates.
    """

    def __init__(
        self,
        root_dir: str = ".",
        cache_dir: Optional[str] = None,
        analyze_func: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._root = root_dir
        self._cache_dir = cache_dir or os.path.join(root_dir, ".reftype-cache")
        self._graph = DependencyGraph()
        self._cache = SummaryCache(persist_path=os.path.join(self._cache_dir, "summaries.json"))
        self._module_tracker = ModuleDependencyTracker()
        self._engine = IncrementalAnalysisEngine(self._graph, self._cache, analyze_func)
        self._sources: Dict[str, str] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._history: List[IncrementalStatistics] = []
        self._initialized = False

    def initialize(self, sources: Dict[str, str]) -> None:
        self._sources = dict(sources)
        builder = CallGraphBuilder()
        self._graph = builder.build(sources)
        self._module_tracker.build_from_sources(sources)
        self._engine = IncrementalAnalysisEngine(self._graph, self._cache)

        if os.path.exists(os.path.join(self._cache_dir, "summaries.json")):
            self._cache.load()
        if os.path.exists(os.path.join(self._cache_dir, "graph.json")):
            try:
                self._graph = DependencyGraph.load(
                    os.path.join(self._cache_dir, "graph.json")
                )
            except (json.JSONDecodeError, KeyError):
                pass

        self._initialized = True

    def update(self, new_sources: Dict[str, str]) -> IncrementalStatistics:
        if not self._initialized:
            self.initialize(new_sources)
            return IncrementalStatistics(total_functions=self._graph.node_count())

        results = self._engine.analyze_incrementally(self._sources, new_sources)
        self._results.update(results)
        stats = self._engine.get_statistics()
        self._history.append(stats)

        new_graph = CallGraphBuilder().build(new_sources)
        self._graph = new_graph
        self._module_tracker.build_from_sources(new_sources)
        self._sources = dict(new_sources)

        return stats

    def save_state(self) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache.persist()
        self._graph.save(os.path.join(self._cache_dir, "graph.json"))
        tracker_path = os.path.join(self._cache_dir, "modules.json")
        with open(tracker_path, "w") as f:
            json.dump(self._module_tracker.to_dict(), f)

    def get_history(self) -> List[IncrementalStatistics]:
        return list(self._history)

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._results)

    def get_graph(self) -> DependencyGraph:
        return self._graph

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Validation / integrity checks
# ---------------------------------------------------------------------------

class GraphIntegrityChecker:
    """Validates the internal consistency of a dependency graph."""

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def check_all(self) -> List[str]:
        issues: List[str] = []
        issues.extend(self._check_edge_consistency())
        issues.extend(self._check_dangling_edges())
        issues.extend(self._check_self_loops())
        issues.extend(self._check_predicate_consistency())
        return issues

    def _check_edge_consistency(self) -> List[str]:
        issues: List[str] = []
        for caller, callee, _ in self._graph.edges():
            if not self._graph.has_node(caller):
                issues.append(f"Edge from non-existent node: {caller}")
            if not self._graph.has_node(callee):
                issues.append(f"Edge to non-existent node: {callee}")
            preds = self._graph.get_predecessors(callee)
            if caller not in preds:
                issues.append(f"Backward edge missing: {caller} -> {callee}")
        return issues

    def _check_dangling_edges(self) -> List[str]:
        issues: List[str] = []
        all_nodes = set(self._graph.nodes())
        for caller, callee, _ in self._graph.edges():
            if caller not in all_nodes:
                issues.append(f"Dangling caller: {caller}")
            if callee not in all_nodes:
                issues.append(f"Dangling callee: {callee}")
        return issues

    def _check_self_loops(self) -> List[str]:
        issues: List[str] = []
        for node in self._graph.nodes():
            if self._graph.has_edge(node, node):
                issues.append(f"Self-loop: {node}")
        return issues

    def _check_predicate_consistency(self) -> List[str]:
        issues: List[str] = []
        for node in self._graph.nodes():
            meta = self._graph.get_node(node)
            if meta and meta.predicate_set.size() > 100:
                issues.append(f"Unusually large predicate set ({meta.predicate_set.size()}) on {node}")
        return issues

    def is_valid(self) -> bool:
        return len(self.check_all()) == 0
