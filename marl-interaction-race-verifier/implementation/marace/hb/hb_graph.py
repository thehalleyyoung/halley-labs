"""
Happens-before relation graph for the MARACE system.

Encapsulates a directed acyclic graph of events with HB (happens-before)
edges.  Provides transitive closure / reduction, connected-component
extraction, subgraph operations, cycle detection, and rich statistics.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from enum import Enum, auto
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

import networkx as nx  # type: ignore

from marace.hb.interaction_groups import InteractionGroup


# ======================================================================
# Public enumerations
# ======================================================================

class HBRelation(Enum):
    """Possible ordering relations between two events."""
    BEFORE = auto()      # e1 → e2
    AFTER = auto()       # e2 → e1
    CONCURRENT = auto()  # neither ordered
    EQUAL = auto()       # same event


# ======================================================================
# HBGraph
# ======================================================================

class HBGraph:
    """Directed acyclic graph encoding happens-before relations.

    Nodes are *event IDs* (any hashable, typically ``str``).  Each node
    may carry arbitrary metadata (agent_id, timestep, action, …) stored
    as node attributes.  Directed edges represent the happens-before
    relation: an edge (u, v) means *u happens-before v*.

    Internally backed by a :class:`networkx.DiGraph`.

    Parameters:
        name: Optional human-readable label for the graph.
    """

    def __init__(self, name: str = "hb_graph") -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._name = name
        self._closure_dirty = True
        self._closure: Optional[nx.DiGraph] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_events(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    @property
    def event_ids(self) -> List[str]:
        return list(self._g.nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._g.edges)

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying networkx DiGraph (read-only use)."""
        return self._g

    # ------------------------------------------------------------------
    # Node / edge manipulation
    # ------------------------------------------------------------------

    def add_event(self, event_id: str, **attrs: Any) -> None:
        """Add an event node (idempotent).  Extra keyword args become node attrs.

        Common attributes include ``agent_id``, ``timestep``, ``action``,
        ``observation``.
        """
        self._g.add_node(event_id, **attrs)
        self._closure_dirty = True

    def add_hb_edge(
        self,
        earlier: str,
        later: str,
        *,
        source: str = "explicit",
        **attrs: Any,
    ) -> None:
        """Declare that *earlier* happens-before *later*.

        Both nodes are created implicitly if they don't exist.

        Args:
            earlier: Event that causally precedes *later*.
            later: Event that causally follows *earlier*.
            source: Label describing how the edge was inferred (e.g.
                ``"program_order"``, ``"communication"``, ``"physics"``).
            **attrs: Additional edge metadata.
        """
        if earlier == later:
            raise ValueError("Self-loops are not valid HB edges")
        self._g.add_edge(earlier, later, source=source, **attrs)
        self._closure_dirty = True

    def add_hb_edges(
        self,
        edges: Iterable[Tuple[str, str]],
        *,
        source: str = "explicit",
    ) -> None:
        """Batch-add multiple HB edges sharing the same *source* tag."""
        for u, v in edges:
            self.add_hb_edge(u, v, source=source)

    def remove_event(self, event_id: str) -> None:
        """Remove an event and all its incident edges."""
        self._g.remove_node(event_id)
        self._closure_dirty = True

    def remove_hb_edge(self, earlier: str, later: str) -> None:
        """Remove a single HB edge."""
        self._g.remove_edge(earlier, later)
        self._closure_dirty = True

    def get_event_attrs(self, event_id: str) -> Dict[str, Any]:
        """Return the attribute dict for *event_id*."""
        return dict(self._g.nodes[event_id])

    def get_edge_attrs(self, earlier: str, later: str) -> Dict[str, Any]:
        """Return edge attributes for (earlier, later)."""
        return dict(self._g.edges[earlier, later])

    # ------------------------------------------------------------------
    # Querying HB relation
    # ------------------------------------------------------------------

    def query_hb(self, e1: str, e2: str) -> HBRelation:
        """Determine the happens-before relation between *e1* and *e2*.

        Uses the transitive closure for O(1) look-ups after an initial
        O(V + E) computation.

        Returns:
            One of ``BEFORE``, ``AFTER``, ``CONCURRENT``, ``EQUAL``.
        """
        if e1 == e2:
            return HBRelation.EQUAL
        closure = self._ensure_closure()
        e1_before_e2 = closure.has_edge(e1, e2)
        e2_before_e1 = closure.has_edge(e2, e1)
        if e1_before_e2 and not e2_before_e1:
            return HBRelation.BEFORE
        if e2_before_e1 and not e1_before_e2:
            return HBRelation.AFTER
        if e1_before_e2 and e2_before_e1:
            # Both directions → cycle (shouldn't happen in a valid HB graph)
            raise ValueError(
                f"Cycle detected between {e1} and {e2}; HB graph is invalid"
            )
        return HBRelation.CONCURRENT

    def happens_before(self, earlier: str, later: str) -> bool:
        """Convenience: True iff *earlier* → *later* in the HB order."""
        return self.query_hb(earlier, later) is HBRelation.BEFORE

    def are_concurrent(self, e1: str, e2: str) -> bool:
        """Convenience: True iff *e1* and *e2* are unordered."""
        return self.query_hb(e1, e2) is HBRelation.CONCURRENT

    def concurrent_pairs(self) -> List[Tuple[str, str]]:
        """Return all pairs (e1, e2) with e1 < e2 (lexicographic) that are concurrent."""
        closure = self._ensure_closure()
        nodes = sorted(self._g.nodes)
        pairs: List[Tuple[str, str]] = []
        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if not closure.has_edge(a, b) and not closure.has_edge(b, a):
                    pairs.append((a, b))
        return pairs

    # ------------------------------------------------------------------
    # Transitive closure / reduction
    # ------------------------------------------------------------------

    def _ensure_closure(self) -> nx.DiGraph:
        if self._closure_dirty or self._closure is None:
            self._closure = self.compute_transitive_closure()
            self._closure_dirty = False
        return self._closure

    def compute_transitive_closure(self) -> nx.DiGraph:
        """Compute the transitive closure using a topological-sort sweep.

        For a DAG with V nodes and E edges this runs in O(V·(V + E)).
        The result is cached until the graph is mutated.

        Returns:
            A new :class:`nx.DiGraph` containing all transitively implied edges.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        if not nx.is_directed_acyclic_graph(self._g):
            cycles = list(nx.simple_cycles(self._g))
            raise ValueError(
                f"Graph contains {len(cycles)} cycle(s); "
                f"first cycle: {cycles[0] if cycles else '?'}"
            )
        closure = nx.DiGraph()
        closure.add_nodes_from(self._g.nodes(data=True))

        # reachable[v] = set of nodes reachable from v
        reachable: Dict[str, Set[str]] = defaultdict(set)

        for node in reversed(list(nx.topological_sort(self._g))):
            for succ in self._g.successors(node):
                reachable[node].add(succ)
                reachable[node].update(reachable[succ])

        for node, targets in reachable.items():
            for t in targets:
                closure.add_edge(node, t)

        return closure

    def compute_transitive_reduction(self) -> "HBGraph":
        """Return a new HBGraph that is the transitive reduction.

        The reduction keeps the minimal set of edges such that the
        transitive closure is identical to the original's.

        Returns:
            A new :class:`HBGraph` with redundant edges removed.
        """
        reduced_nx = nx.transitive_reduction(self._g)
        result = HBGraph(name=f"{self._name}_reduced")
        for node, data in self._g.nodes(data=True):
            result.add_event(node, **data)
        for u, v in reduced_nx.edges():
            edge_data = self._g.edges.get((u, v), {})
            result._g.add_edge(u, v, **edge_data)
        result._closure_dirty = True
        return result

    # ------------------------------------------------------------------
    # Connected components / interaction groups
    # ------------------------------------------------------------------

    def connected_components(self) -> List[Set[str]]:
        """Return weakly-connected components as sets of event IDs.

        Weakly connected in a directed graph means connected when edge
        directions are ignored.
        """
        return [set(c) for c in nx.weakly_connected_components(self._g)]

    def extract_interaction_groups(self) -> List[InteractionGroup]:
        """Build :class:`InteractionGroup` objects from connected components.

        Each component becomes a group.  Agent IDs are gathered from the
        ``agent_id`` node attribute; shared state dimensions and interaction
        strength are computed from available metadata.

        Returns:
            List of :class:`InteractionGroup` instances.
        """
        groups: List[InteractionGroup] = []
        for comp in self.connected_components():
            agent_ids: Set[str] = set()
            shared_dims: Set[str] = set()
            for eid in comp:
                attrs = self._g.nodes[eid]
                if "agent_id" in attrs:
                    agent_ids.add(attrs["agent_id"])
                if "state_dims" in attrs:
                    shared_dims.update(attrs["state_dims"])

            n_agents = len(agent_ids) if agent_ids else 1
            sub = self._g.subgraph(comp)
            n_cross = sum(
                1 for u, v in sub.edges()
                if sub.nodes[u].get("agent_id") != sub.nodes[v].get("agent_id")
            )
            max_cross = n_agents * (n_agents - 1)
            strength = n_cross / max_cross if max_cross > 0 else 0.0

            groups.append(InteractionGroup(
                agent_ids=frozenset(agent_ids) if agent_ids else frozenset(comp),
                shared_state_dims=frozenset(shared_dims),
                interaction_strength=strength,
                event_ids=frozenset(comp),
            ))
        return groups

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def subgraph_for_agents(self, agent_ids: Set[str]) -> "HBGraph":
        """Extract the induced subgraph containing only events from *agent_ids*.

        Args:
            agent_ids: Set of agent identifiers to keep.

        Returns:
            A new :class:`HBGraph` restricted to the specified agents.
        """
        keep = {
            n for n, d in self._g.nodes(data=True)
            if d.get("agent_id") in agent_ids
        }
        sub = HBGraph(name=f"{self._name}_sub")
        sub_nx = self._g.subgraph(keep).copy()
        sub._g = sub_nx
        sub._closure_dirty = True
        return sub

    def subgraph_for_events(self, event_ids: Set[str]) -> "HBGraph":
        """Extract the induced subgraph for an explicit set of event IDs."""
        sub = HBGraph(name=f"{self._name}_events")
        sub._g = self._g.subgraph(event_ids).copy()
        sub._closure_dirty = True
        return sub

    def subgraph_for_timestep_range(
        self, t_min: int, t_max: int,
    ) -> "HBGraph":
        """Extract events within a timestep range [t_min, t_max]."""
        keep = {
            n for n, d in self._g.nodes(data=True)
            if t_min <= d.get("timestep", -1) <= t_max
        }
        sub = HBGraph(name=f"{self._name}_t{t_min}_{t_max}")
        sub._g = self._g.subgraph(keep).copy()
        sub._closure_dirty = True
        return sub

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def has_cycles(self) -> bool:
        """Return True if the graph contains directed cycles.

        Cycles indicate a bug in HB construction because the
        happens-before relation must be a strict partial order.
        """
        return not nx.is_directed_acyclic_graph(self._g)

    def find_cycles(self) -> List[List[str]]:
        """Return all simple directed cycles, if any."""
        return list(nx.simple_cycles(self._g))

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Length of the longest directed path (the *height* of the DAG).

        Returns 0 for an empty graph.
        """
        if self.num_events == 0:
            return 0
        return nx.dag_longest_path_length(self._g) if nx.is_directed_acyclic_graph(self._g) else -1

    def width(self) -> int:
        """Maximum anti-chain size (width of the partial order).

        Uses a greedy approximation for large graphs.
        """
        if self.num_events == 0:
            return 0
        closure = self._ensure_closure()
        # Dilworth: width = number of nodes - size of maximum matching
        # in a bipartite graph (node-split).  For moderate sizes we use
        # nx directly; for very large graphs this is approximate.
        if self.num_events > 5000:
            return self._approximate_width()
        # Build bipartite graph: split each node into (n_out, n_in)
        B = nx.Graph()
        for n in self._g.nodes:
            B.add_node(f"{n}_out", bipartite=0)
            B.add_node(f"{n}_in", bipartite=1)
        for u, v in closure.edges():
            B.add_edge(f"{u}_out", f"{v}_in")
        matching = nx.bipartite.maximum_matching(
            B, top_nodes={f"{n}_out" for n in self._g.nodes}
        )
        matching_size = len(matching) // 2
        return self.num_events - matching_size

    def _approximate_width(self) -> int:
        """Heuristic width for large graphs: count sources per topo level."""
        levels: Dict[str, int] = {}
        for node in nx.topological_sort(self._g):
            preds = list(self._g.predecessors(node))
            if not preds:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in preds) + 1
        if not levels:
            return 0
        level_counts: Dict[int, int] = defaultdict(int)
        for lv in levels.values():
            level_counts[lv] += 1
        return max(level_counts.values())

    def density(self) -> float:
        """Edge density E / (V*(V-1)/2), treating graph as undirected."""
        v = self.num_events
        if v < 2:
            return 0.0
        return self.num_edges / (v * (v - 1) / 2)

    def longest_chain(self) -> List[str]:
        """Return the longest directed path (list of event IDs)."""
        if self.num_events == 0:
            return []
        if not nx.is_directed_acyclic_graph(self._g):
            return []
        return nx.dag_longest_path(self._g)

    def statistics(self) -> Dict[str, Any]:
        """Aggregate statistics about the HB graph."""
        n_comp = len(self.connected_components())
        return {
            "num_events": self.num_events,
            "num_edges": self.num_edges,
            "num_components": n_comp,
            "depth": self.depth(),
            "density": round(self.density(), 6),
            "has_cycles": self.has_cycles(),
            "longest_chain_length": len(self.longest_chain()),
        }

    # ------------------------------------------------------------------
    # Predecessor / successor queries
    # ------------------------------------------------------------------

    def predecessors(self, event_id: str) -> Set[str]:
        """Direct predecessors of *event_id*."""
        return set(self._g.predecessors(event_id))

    def successors(self, event_id: str) -> Set[str]:
        """Direct successors of *event_id*."""
        return set(self._g.successors(event_id))

    def all_predecessors(self, event_id: str) -> Set[str]:
        """All transitive predecessors (ancestors) of *event_id*."""
        return nx.ancestors(self._g, event_id)

    def all_successors(self, event_id: str) -> Set[str]:
        """All transitive successors (descendants) of *event_id*."""
        return nx.descendants(self._g, event_id)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        edges = []
        for u, v, d in self._g.edges(data=True):
            edge_entry: Dict[str, Any] = {"from": u, "to": v}
            edge_entry["attrs"] = dict(d)
            edges.append(edge_entry)
        return {
            "name": self._name,
            "nodes": [
                {"id": n, **d} for n, d in self._g.nodes(data=True)
            ],
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HBGraph":
        """Reconstruct from serialized form."""
        g = cls(name=data.get("name", "hb_graph"))
        for node_data in data.get("nodes", []):
            nd = dict(node_data)
            nid = nd.pop("id")
            g.add_event(nid, **nd)
        for edge_data in data.get("edges", []):
            src = edge_data["from"]
            tgt = edge_data["to"]
            attrs = edge_data.get("attrs", {})
            g._g.add_edge(src, tgt, **attrs)
        g._closure_dirty = True
        return g

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: "HBGraph") -> None:
        """Merge another HBGraph into this one (union of nodes and edges)."""
        for node, data in other._g.nodes(data=True):
            if node not in self._g:
                self._g.add_node(node, **data)
        for u, v, data in other._g.edges(data=True):
            if not self._g.has_edge(u, v):
                self._g.add_edge(u, v, **data)
        self._closure_dirty = True

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HBGraph(name={self._name!r}, events={self.num_events}, "
            f"edges={self.num_edges})"
        )

    def __len__(self) -> int:
        return self.num_events

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._g
