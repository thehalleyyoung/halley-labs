"""
Transition graph data structure for explicit-state model checking.

Wraps a networkx DiGraph with model-checking-specific semantics:
state nodes carry atomic propositions, edges carry action labels and
stuttering information.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateNode:
    """A node in the transition graph representing a single system state."""

    state_hash: str
    full_state: Dict[str, Any]
    atomic_propositions: FrozenSet[str] = field(default_factory=frozenset)

    # Convenience ----------------------------------------------------------

    def satisfies(self, prop: str) -> bool:
        return prop in self.atomic_propositions

    def satisfies_all(self, props: Iterable[str]) -> bool:
        return all(p in self.atomic_propositions for p in props)

    def satisfies_any(self, props: Iterable[str]) -> bool:
        return any(p in self.atomic_propositions for p in props)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "full_state": self.full_state,
            "atomic_propositions": sorted(self.atomic_propositions),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateNode":
        return cls(
            state_hash=d["state_hash"],
            full_state=d["full_state"],
            atomic_propositions=frozenset(d.get("atomic_propositions", [])),
        )


@dataclass(frozen=True)
class TransitionEdge:
    """An edge in the transition graph representing a system transition."""

    action_label: str
    source_hash: str
    target_hash: str
    is_stuttering: bool = False
    guard: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_label": self.action_label,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "is_stuttering": self.is_stuttering,
            "guard": self.guard,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransitionEdge":
        return cls(
            action_label=d["action_label"],
            source_hash=d["source_hash"],
            target_hash=d["target_hash"],
            is_stuttering=d.get("is_stuttering", False),
            guard=d.get("guard"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

@dataclass
class GraphStatistics:
    num_states: int = 0
    num_transitions: int = 0
    num_initial_states: int = 0
    avg_branching_factor: float = 0.0
    max_branching_factor: int = 0
    min_branching_factor: int = 0
    num_deadlock_states: int = 0
    num_sccs: int = 0
    largest_scc_size: int = 0
    diameter_estimate: int = 0
    num_self_loops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Transition graph
# ---------------------------------------------------------------------------

class TransitionGraph:
    """
    Directed transition graph built on top of networkx.

    Nodes are keyed by state hash.  Each node stores the corresponding
    ``StateNode`` object.  Each edge stores a ``TransitionEdge`` object.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, StateNode] = {}
        self._initial_states: Set[str] = set()
        self._action_index: Dict[str, List[TransitionEdge]] = {}
        self._prop_index: Dict[str, Set[str]] = {}

    # -- Properties --------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_transitions(self) -> int:
        return self._graph.number_of_edges()

    @property
    def initial_states(self) -> Set[str]:
        return set(self._initial_states)

    # -- Mutation ----------------------------------------------------------

    def add_state(
        self,
        node: StateNode,
        *,
        is_initial: bool = False,
    ) -> bool:
        """Add a state.  Returns True if the state was new."""
        h = node.state_hash
        if h in self._nodes:
            return False
        self._nodes[h] = node
        self._graph.add_node(h, node=node)
        if is_initial:
            self._initial_states.add(h)
        for prop in node.atomic_propositions:
            self._prop_index.setdefault(prop, set()).add(h)
        return True

    def add_transition(self, edge: TransitionEdge) -> bool:
        """Add a transition.  Returns True if it was new."""
        src, tgt = edge.source_hash, edge.target_hash
        if src not in self._nodes or tgt not in self._nodes:
            raise KeyError("Source or target state not in graph")
        key = (src, tgt, edge.action_label)
        if self._graph.has_edge(src, tgt):
            existing = self._graph[src][tgt].get("edges", [])
            for e in existing:
                if e.action_label == edge.action_label:
                    return False
            existing.append(edge)
            self._graph[src][tgt]["edges"] = existing
        else:
            self._graph.add_edge(src, tgt, edges=[edge])
        self._action_index.setdefault(edge.action_label, []).append(edge)
        return True

    def mark_initial(self, state_hash: str) -> None:
        if state_hash not in self._nodes:
            raise KeyError(f"Unknown state: {state_hash}")
        self._initial_states.add(state_hash)

    def remove_state(self, state_hash: str) -> None:
        if state_hash not in self._nodes:
            return
        node = self._nodes.pop(state_hash)
        for prop in node.atomic_propositions:
            s = self._prop_index.get(prop)
            if s:
                s.discard(state_hash)
        self._initial_states.discard(state_hash)
        self._graph.remove_node(state_hash)

    def remove_transition(self, src: str, tgt: str, action: str) -> None:
        if not self._graph.has_edge(src, tgt):
            return
        edges = self._graph[src][tgt].get("edges", [])
        edges = [e for e in edges if e.action_label != action]
        if edges:
            self._graph[src][tgt]["edges"] = edges
        else:
            self._graph.remove_edge(src, tgt)

    # -- Query -------------------------------------------------------------

    def get_state(self, state_hash: str) -> Optional[StateNode]:
        return self._nodes.get(state_hash)

    def has_state(self, state_hash: str) -> bool:
        return state_hash in self._nodes

    def all_states(self) -> Iterable[StateNode]:
        return self._nodes.values()

    def all_state_hashes(self) -> Set[str]:
        return set(self._nodes.keys())

    def get_successors(self, state_hash: str) -> List[Tuple[StateNode, TransitionEdge]]:
        result: List[Tuple[StateNode, TransitionEdge]] = []
        for succ in self._graph.successors(state_hash):
            for edge in self._graph[state_hash][succ].get("edges", []):
                result.append((self._nodes[succ], edge))
        return result

    def get_successor_hashes(self, state_hash: str) -> Set[str]:
        return set(self._graph.successors(state_hash))

    def get_predecessors(self, state_hash: str) -> List[Tuple[StateNode, TransitionEdge]]:
        result: List[Tuple[StateNode, TransitionEdge]] = []
        for pred in self._graph.predecessors(state_hash):
            for edge in self._graph[pred][state_hash].get("edges", []):
                result.append((self._nodes[pred], edge))
        return result

    def get_predecessor_hashes(self, state_hash: str) -> Set[str]:
        return set(self._graph.predecessors(state_hash))

    def get_edges(self, src: str, tgt: str) -> List[TransitionEdge]:
        if not self._graph.has_edge(src, tgt):
            return []
        return list(self._graph[src][tgt].get("edges", []))

    def get_all_edges(self) -> Iterable[TransitionEdge]:
        for u, v, data in self._graph.edges(data=True):
            yield from data.get("edges", [])

    def get_states_by_predicate(
        self, predicate: Callable[[StateNode], bool]
    ) -> List[StateNode]:
        return [n for n in self._nodes.values() if predicate(n)]

    def get_states_by_proposition(self, prop: str) -> List[StateNode]:
        hashes = self._prop_index.get(prop, set())
        return [self._nodes[h] for h in hashes if h in self._nodes]

    def get_transitions_by_action(self, action: str) -> List[TransitionEdge]:
        return list(self._action_index.get(action, []))

    def get_deadlock_states(self) -> List[StateNode]:
        return [
            self._nodes[h]
            for h in self._nodes
            if self._graph.out_degree(h) == 0
        ]

    def out_degree(self, state_hash: str) -> int:
        return self._graph.out_degree(state_hash)

    def in_degree(self, state_hash: str) -> int:
        return self._graph.in_degree(state_hash)

    # -- Structural analysis -----------------------------------------------

    def strongly_connected_components(self) -> List[Set[str]]:
        return [set(c) for c in nx.strongly_connected_components(self._graph)]

    def nontrivial_sccs(self) -> List[Set[str]]:
        """SCCs with >1 state or containing a self-loop."""
        result: List[Set[str]] = []
        for scc in self.strongly_connected_components():
            if len(scc) > 1:
                result.append(scc)
            elif len(scc) == 1:
                h = next(iter(scc))
                if self._graph.has_edge(h, h):
                    result.append(scc)
        return result

    def detect_cycles(self) -> List[List[str]]:
        """Return all simple cycles (may be expensive for large graphs)."""
        try:
            return [list(c) for c in nx.simple_cycles(self._graph)]
        except Exception:
            return []

    def has_cycle(self) -> bool:
        try:
            nx.find_cycle(self._graph)
            return True
        except nx.NetworkXNoCycle:
            return False

    def topological_sort(self) -> Optional[List[str]]:
        if self.has_cycle():
            return None
        return list(nx.topological_sort(self._graph))

    # -- Path finding ------------------------------------------------------

    def shortest_path(self, src: str, tgt: str) -> Optional[List[str]]:
        try:
            return list(nx.shortest_path(self._graph, src, tgt))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def shortest_path_length(self, src: str, tgt: str) -> Optional[int]:
        try:
            return nx.shortest_path_length(self._graph, src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def bfs_reachable(self, start: str, max_depth: Optional[int] = None) -> Set[str]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            visited.add(current)
            for succ in self._graph.successors(current):
                if succ not in visited:
                    queue.append((succ, depth + 1))
        return visited

    def all_paths(
        self, src: str, tgt: str, max_length: int = 20
    ) -> List[List[str]]:
        """Enumerate simple paths up to *max_length*."""
        return [
            list(p)
            for p in nx.all_simple_paths(self._graph, src, tgt, cutoff=max_length)
        ]

    # -- Statistics --------------------------------------------------------

    def compute_statistics(self) -> GraphStatistics:
        stats = GraphStatistics()
        stats.num_states = self.num_states
        stats.num_transitions = self.num_transitions
        stats.num_initial_states = len(self._initial_states)

        if self.num_states == 0:
            return stats

        out_degrees = [self._graph.out_degree(h) for h in self._nodes]
        stats.avg_branching_factor = statistics.mean(out_degrees)
        stats.max_branching_factor = max(out_degrees)
        stats.min_branching_factor = min(out_degrees)
        stats.num_deadlock_states = sum(1 for d in out_degrees if d == 0)
        stats.num_self_loops = sum(
            1 for h in self._nodes if self._graph.has_edge(h, h)
        )

        sccs = self.strongly_connected_components()
        stats.num_sccs = len(sccs)
        stats.largest_scc_size = max((len(s) for s in sccs), default=0)

        stats.diameter_estimate = self._estimate_diameter()
        return stats

    def _estimate_diameter(self, samples: int = 10) -> int:
        """Estimate diameter via BFS from a sample of nodes."""
        if self.num_states == 0:
            return 0
        import random

        nodes = list(self._nodes.keys())
        sample_nodes = random.sample(nodes, min(samples, len(nodes)))
        max_dist = 0
        for src in sample_nodes:
            lengths = nx.single_source_shortest_path_length(self._graph, src)
            if lengths:
                max_dist = max(max_dist, max(lengths.values()))
        return max_dist

    # -- Subgraph extraction -----------------------------------------------

    def subgraph(self, state_hashes: Set[str]) -> "TransitionGraph":
        """Return a new graph restricted to the given states."""
        sub = TransitionGraph()
        for h in state_hashes:
            node = self._nodes.get(h)
            if node is not None:
                sub.add_state(node, is_initial=(h in self._initial_states))
        for u, v, data in self._graph.edges(data=True):
            if u in state_hashes and v in state_hashes:
                for edge in data.get("edges", []):
                    sub.add_transition(edge)
        return sub

    def reachable_subgraph(self, roots: Optional[Set[str]] = None) -> "TransitionGraph":
        if roots is None:
            roots = self._initial_states
        reachable: Set[str] = set()
        for r in roots:
            reachable |= self.bfs_reachable(r)
        return self.subgraph(reachable)

    def scc_subgraph(self, scc: Set[str]) -> "TransitionGraph":
        return self.subgraph(scc)

    # -- Serialization -----------------------------------------------------

    def to_json(self) -> str:
        data: Dict[str, Any] = {
            "states": [n.to_dict() for n in self._nodes.values()],
            "transitions": [e.to_dict() for e in self.get_all_edges()],
            "initial_states": sorted(self._initial_states),
        }
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "TransitionGraph":
        data = json.loads(json_str)
        g = cls()
        initial = set(data.get("initial_states", []))
        for sd in data.get("states", []):
            node = StateNode.from_dict(sd)
            g.add_state(node, is_initial=(node.state_hash in initial))
        for ed in data.get("transitions", []):
            edge = TransitionEdge.from_dict(ed)
            g.add_transition(edge)
        return g

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, path: str) -> "TransitionGraph":
        with open(path) as f:
            return cls.from_json(f.read())

    # -- DOT export --------------------------------------------------------

    def to_dot(
        self,
        *,
        label_states: bool = True,
        highlight_initial: bool = True,
        highlight_deadlock: bool = True,
        max_label_len: int = 40,
    ) -> str:
        lines: List[str] = ["digraph TransitionGraph {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=ellipse fontsize=10];')
        lines.append('  edge [fontsize=8];')

        for h, node in self._nodes.items():
            attrs: List[str] = []
            short = h[:8]
            if label_states:
                props = ",".join(sorted(node.atomic_propositions))
                label = f"{short}\\n{{{props}}}" if props else short
                if len(label) > max_label_len:
                    label = label[: max_label_len - 3] + "..."
                attrs.append(f'label="{label}"')
            if highlight_initial and h in self._initial_states:
                attrs.append("shape=doubleoctagon")
                attrs.append("color=blue")
            if highlight_deadlock and self._graph.out_degree(h) == 0:
                attrs.append("color=red")
                attrs.append("style=filled")
                attrs.append("fillcolor=lightyellow")
            attr_str = " ".join(attrs)
            lines.append(f'  "{short}" [{attr_str}];')

        for u, v, data in self._graph.edges(data=True):
            for edge in data.get("edges", []):
                style = "dashed" if edge.is_stuttering else "solid"
                label = edge.action_label
                if len(label) > max_label_len:
                    label = label[: max_label_len - 3] + "..."
                lines.append(
                    f'  "{u[:8]}" -> "{v[:8]}" '
                    f'[label="{label}" style={style}];'
                )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, path: str, **kwargs: Any) -> None:
        with open(path, "w") as f:
            f.write(self.to_dot(**kwargs))

    # -- Utility -----------------------------------------------------------

    def copy(self) -> "TransitionGraph":
        g = TransitionGraph()
        for h, node in self._nodes.items():
            g.add_state(node, is_initial=(h in self._initial_states))
        for edge in self.get_all_edges():
            g.add_transition(edge)
        return g

    def merge(self, other: "TransitionGraph") -> None:
        for node in other.all_states():
            self.add_state(
                node,
                is_initial=(node.state_hash in other.initial_states),
            )
        for edge in other.get_all_edges():
            try:
                self.add_transition(edge)
            except KeyError:
                pass

    def clear(self) -> None:
        self._graph.clear()
        self._nodes.clear()
        self._initial_states.clear()
        self._action_index.clear()
        self._prop_index.clear()

    def __len__(self) -> int:
        return self.num_states

    def __contains__(self, state_hash: str) -> bool:
        return state_hash in self._nodes

    def __repr__(self) -> str:
        return (
            f"TransitionGraph(states={self.num_states}, "
            f"transitions={self.num_transitions}, "
            f"initial={len(self._initial_states)})"
        )
