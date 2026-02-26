"""
Markov equivalence classes for causal graphs.

Provides CPDAG (completed partially directed acyclic graph) and PAG
(partial ancestral graph) representations, MEC enumeration, equivalence
class membership testing, interventional equivalence, and essential
graph computation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np


# ====================================================================
# CPDAG
# ====================================================================

class CPDAG:
    """Completed Partially Directed Acyclic Graph.

    A CPDAG represents a Markov equivalence class: directed edges are
    *compelled* (present in every DAG in the MEC) and undirected edges
    are *reversible*.

    Parameters
    ----------
    directed_edges : list of (str, str)
        Compelled directed edges.
    undirected_edges : list of (str, str)
        Reversible edges.
    nodes : list of str, optional
        Variable names.
    """

    def __init__(
        self,
        directed_edges: Optional[List[Tuple[str, str]]] = None,
        undirected_edges: Optional[List[Tuple[str, str]]] = None,
        nodes: Optional[List[str]] = None,
    ) -> None:
        self._directed: Set[Tuple[str, str]] = set(directed_edges or [])
        self._undirected: Set[FrozenSet[str]] = set()
        for u, v in (undirected_edges or []):
            self._undirected.add(frozenset({u, v}))
        if nodes is None:
            nodes_set: Set[str] = set()
            for u, v in self._directed:
                nodes_set |= {u, v}
            for e in self._undirected:
                nodes_set |= e
            self._nodes = sorted(nodes_set)
        else:
            self._nodes = list(nodes)

    @classmethod
    def from_dag(cls, dag: nx.DiGraph) -> "CPDAG":
        """Construct the CPDAG (essential graph) of a DAG.

        An edge is compelled iff reversing it changes the Markov
        equivalence class (i.e. it participates in a v-structure or
        is forced by the Meek rules).
        """
        cpdag = cls(nodes=list(dag.nodes))
        compelled = _find_compelled_edges(dag)
        for u, v in dag.edges:
            if (u, v) in compelled:
                cpdag._directed.add((u, v))
            else:
                cpdag._undirected.add(frozenset({u, v}))
        return cpdag

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    @property
    def directed_edges(self) -> List[Tuple[str, str]]:
        return sorted(self._directed)

    @property
    def undirected_edges(self) -> List[Tuple[str, str]]:
        return sorted((min(e), max(e)) for e in self._undirected)

    @property
    def n_directed(self) -> int:
        return len(self._directed)

    @property
    def n_undirected(self) -> int:
        return len(self._undirected)

    def adjacency(self, node: str) -> Set[str]:
        """All neighbours (directed and undirected)."""
        adj: Set[str] = set()
        for u, v in self._directed:
            if u == node:
                adj.add(v)
            elif v == node:
                adj.add(u)
        for e in self._undirected:
            if node in e:
                adj |= e - {node}
        return adj

    def parents(self, node: str) -> Set[str]:
        """Nodes with a compelled directed edge into *node*."""
        return {u for u, v in self._directed if v == node}

    def children(self, node: str) -> Set[str]:
        return {v for u, v in self._directed if u == node}

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a networkx DiGraph (undirected = two directed edges)."""
        G = nx.DiGraph()
        G.add_nodes_from(self._nodes)
        for u, v in self._directed:
            G.add_edge(u, v, compelled=True)
        for e in self._undirected:
            u, v = sorted(e)
            G.add_edge(u, v, compelled=False)
            G.add_edge(v, u, compelled=False)
        return G

    def same_equivalence_class(self, dag: nx.DiGraph) -> bool:
        """Check whether *dag* belongs to this Markov equivalence class."""
        other_cpdag = CPDAG.from_dag(dag)
        return (
            self._directed == other_cpdag._directed
            and self._undirected == other_cpdag._undirected
        )

    def enumerate_dags(self, max_dags: int = 1000) -> List[nx.DiGraph]:
        """Enumerate DAGs in the MEC by orienting each undirected edge
        in all valid ways (brute-force, bounded)."""
        undirected_list = list(self._undirected)
        n_und = len(undirected_list)
        if 2 ** n_und > max_dags * 10:
            # Too many – sample instead
            return self._sample_dags(max_dags)

        dags: List[nx.DiGraph] = []
        for bits in itertools.product([0, 1], repeat=n_und):
            G = nx.DiGraph()
            G.add_nodes_from(self._nodes)
            for u, v in self._directed:
                G.add_edge(u, v)
            for idx, bit in enumerate(bits):
                u, v = sorted(undirected_list[idx])
                if bit == 0:
                    G.add_edge(u, v)
                else:
                    G.add_edge(v, u)
            if nx.is_directed_acyclic_graph(G):
                # Check it has the same skeleton and v-structures
                if CPDAG.from_dag(G)._directed == self._directed:
                    dags.append(G)
            if len(dags) >= max_dags:
                break
        return dags

    def _sample_dags(self, n_samples: int) -> List[nx.DiGraph]:
        """Randomly sample DAGs from the MEC."""
        rng = np.random.default_rng(42)
        undirected_list = list(self._undirected)
        n_und = len(undirected_list)
        dags: List[nx.DiGraph] = []
        seen: Set[FrozenSet[Tuple[str, str]]] = set()
        attempts = 0
        while len(dags) < n_samples and attempts < n_samples * 50:
            attempts += 1
            G = nx.DiGraph()
            G.add_nodes_from(self._nodes)
            for u, v in self._directed:
                G.add_edge(u, v)
            for e in undirected_list:
                u, v = sorted(e)
                if rng.random() < 0.5:
                    G.add_edge(u, v)
                else:
                    G.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(G):
                continue
            sig = frozenset(G.edges)
            if sig in seen:
                continue
            seen.add(sig)
            dags.append(G)
        return dags

    def mec_size_upper_bound(self) -> int:
        """Upper bound on MEC size: 2^(number of undirected edges)."""
        return 2 ** len(self._undirected)

    def __repr__(self) -> str:
        return (
            f"CPDAG(nodes={len(self._nodes)}, "
            f"directed={len(self._directed)}, "
            f"undirected={len(self._undirected)})"
        )


# ====================================================================
# PAG (Partial Ancestral Graph)
# ====================================================================

class PAG:
    """Partial Ancestral Graph for equivalence classes allowing latent
    confounders.

    Edge marks: '-' (tail), '>' (arrowhead), 'o' (circle/unknown).
    Stored as (i, j, mark_at_i, mark_at_j).
    """

    def __init__(self, nodes: Optional[List[str]] = None) -> None:
        self._nodes: List[str] = list(nodes or [])
        self._marks: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def add_edge(
        self, u: str, v: str, mark_u: str = "o", mark_v: str = "o"
    ) -> None:
        """Add an edge u *-* v with specified marks."""
        if u not in self._nodes:
            self._nodes.append(u)
        if v not in self._nodes:
            self._nodes.append(v)
        self._marks[(u, v)] = (mark_u, mark_v)
        self._marks[(v, u)] = (mark_v, mark_u)

    def set_marks(
        self, u: str, v: str, mark_at_u: str, mark_at_v: str
    ) -> None:
        self._marks[(u, v)] = (mark_at_u, mark_at_v)
        self._marks[(v, u)] = (mark_at_v, mark_at_u)

    def get_marks(self, u: str, v: str) -> Optional[Tuple[str, str]]:
        """Return (mark_at_u, mark_at_v) or None if not adjacent."""
        return self._marks.get((u, v))

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    @property
    def edges(self) -> List[Tuple[str, str, str, str]]:
        """All edges as (u, v, mark_at_u, mark_at_v), each pair listed once."""
        seen: Set[FrozenSet[str]] = set()
        result = []
        for (u, v), (mu, mv) in self._marks.items():
            key = frozenset({u, v})
            if key in seen:
                continue
            seen.add(key)
            result.append((u, v, mu, mv))
        return result

    def is_ancestor(self, u: str, v: str) -> Optional[bool]:
        """Check if u is a definite ancestor of v in the PAG.

        Returns True/False if determinable, None if ambiguous.
        """
        # BFS following only directed edges u → ...
        visited: Set[str] = set()
        queue = [u]
        while queue:
            current = queue.pop(0)
            if current == v and current != u:
                return True
            if current in visited:
                continue
            visited.add(current)
            for (a, b), (ma, mb) in self._marks.items():
                if a == current and mb == ">" and ma == "-":
                    queue.append(b)
        return False if len(visited) > 0 else None

    def definite_colliders(self) -> List[Tuple[str, str, str]]:
        """Return all definite collider triples (a, b, c) where a *→ b ←* c."""
        colliders = []
        for b in self._nodes:
            into_b = []
            for (u, v), (mu, mv) in self._marks.items():
                if v == b and mv == ">":
                    into_b.append(u)
            for i, a in enumerate(into_b):
                for c in into_b[i + 1:]:
                    if frozenset({a, c}) not in {
                        frozenset({u, v}) for (u, v) in self._marks
                        if u == a and v == c
                    }:
                        colliders.append((a, b, c))
        return colliders

    def possible_ancestors(self, node: str) -> Set[str]:
        """Nodes that are possible ancestors (following o→ and → edges)."""
        visited: Set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for (u, v), (mu, mv) in self._marks.items():
                if v == current and mv in (">",):
                    queue.append(u)
        visited.discard(node)
        return visited

    def to_networkx(self) -> nx.DiGraph:
        """Convert to DiGraph with mark attributes."""
        G = nx.DiGraph()
        G.add_nodes_from(self._nodes)
        for u, v, mu, mv in self.edges:
            G.add_edge(u, v, mark_source=mu, mark_target=mv)
            G.add_edge(v, u, mark_source=mv, mark_target=mu)
        return G

    @classmethod
    def from_fci_output(
        cls, pag_graph: nx.DiGraph
    ) -> "PAG":
        """Construct from an FCI-output DiGraph with mark attributes."""
        nodes = list(pag_graph.nodes)
        pag = cls(nodes=nodes)
        seen: Set[FrozenSet[str]] = set()
        for u, v, data in pag_graph.edges(data=True):
            key = frozenset({u, v})
            if key in seen:
                continue
            seen.add(key)
            mt = data.get("mark_at_target", "o")
            ms = data.get("mark_at_source", "o")
            pag.add_edge(u, v, mark_u=ms, mark_v=mt)
        return pag

    def __repr__(self) -> str:
        return f"PAG(nodes={len(self._nodes)}, edges={len(self.edges)})"


# ====================================================================
# Markov Equivalence Class utilities
# ====================================================================

class MarkovEquivalenceClass:
    """Utilities for working with Markov equivalence classes.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the equivalence class.
    """

    def __init__(self, cpdag: CPDAG) -> None:
        self.cpdag = cpdag

    @classmethod
    def from_dag(cls, dag: nx.DiGraph) -> "MarkovEquivalenceClass":
        return cls(CPDAG.from_dag(dag))

    def contains(self, dag: nx.DiGraph) -> bool:
        """Test whether *dag* is a member of this MEC."""
        return self.cpdag.same_equivalence_class(dag)

    def enumerate(self, max_dags: int = 1000) -> List[nx.DiGraph]:
        """Enumerate DAGs in the MEC."""
        return self.cpdag.enumerate_dags(max_dags=max_dags)

    def size_upper_bound(self) -> int:
        return self.cpdag.mec_size_upper_bound()

    def essential_edges(self) -> List[Tuple[str, str]]:
        """Edges that are compelled (present in every DAG in the MEC)."""
        return self.cpdag.directed_edges

    def reversible_edges(self) -> List[Tuple[str, str]]:
        """Edges that can be reversed without leaving the MEC."""
        return self.cpdag.undirected_edges

    def interventional_equivalence(
        self,
        dag1: nx.DiGraph,
        dag2: nx.DiGraph,
        intervention_targets: List[Set[str]],
    ) -> bool:
        """Test interventional equivalence: two DAGs are I-equivalent iff
        they have the same skeleton, v-structures, and the same
        interventional distributions for all given intervention targets.

        (Hauser & Bühlmann 2012)
        """
        # Same skeleton?
        skel1 = set(frozenset(e) for e in dag1.edges)
        skel2 = set(frozenset(e) for e in dag2.edges)
        if skel1 != skel2:
            return False

        # Same v-structures?
        vs1 = _v_structures(dag1)
        vs2 = _v_structures(dag2)
        if vs1 != vs2:
            return False

        # Check intervention-induced orientations
        for targets in intervention_targets:
            for target in targets:
                parents1 = set(dag1.predecessors(target))
                parents2 = set(dag2.predecessors(target))
                if parents1 != parents2:
                    return False
        return True

    def __repr__(self) -> str:
        return (
            f"MarkovEquivalenceClass(compelled={self.cpdag.n_directed}, "
            f"reversible={self.cpdag.n_undirected})"
        )


# ====================================================================
# Helper functions
# ====================================================================

def _v_structures(dag: nx.DiGraph) -> Set[Tuple[str, str, str]]:
    """Extract v-structures (colliders) from a DAG: (a, b, c) s.t.
    a → b ← c and a not adj c."""
    vs: Set[Tuple[str, str, str]] = set()
    for b in dag.nodes:
        parents_b = list(dag.predecessors(b))
        for i, a in enumerate(parents_b):
            for c in parents_b[i + 1:]:
                if not dag.has_edge(a, c) and not dag.has_edge(c, a):
                    pair = tuple(sorted([a, c]))
                    vs.add((pair[0], b, pair[1]))
    return vs


def _find_compelled_edges(dag: nx.DiGraph) -> Set[Tuple[str, str]]:
    """Find compelled (essential) edges in a DAG via Chickering's algorithm.

    An edge is compelled iff every DAG in the same MEC contains that edge
    in the same direction.
    """
    compelled: Set[Tuple[str, str]] = set()
    order_edges: List[Tuple[str, str]] = []

    # Label edges based on v-structures and Meek rules
    # Phase 1: v-structure edges are compelled
    for b in dag.nodes:
        parents_b = list(dag.predecessors(b))
        for i, a in enumerate(parents_b):
            for c in parents_b[i + 1:]:
                if not dag.has_edge(a, c) and not dag.has_edge(c, a):
                    compelled.add((a, b))
                    compelled.add((c, b))

    # Phase 2: iterative Meek-rule propagation
    changed = True
    while changed:
        changed = False
        for u, v in dag.edges:
            if (u, v) in compelled:
                continue
            # Meek R1: if w → u — v and w not adj v → orient u → v
            for w in dag.predecessors(u):
                if (w, u) in compelled and not dag.has_edge(w, v) and not dag.has_edge(v, w):
                    compelled.add((u, v))
                    changed = True
                    break
            if (u, v) in compelled:
                continue
            # Meek R2: if u → w → v and u — v → orient u → v
            for w in dag.successors(u):
                if w == v:
                    continue
                if (u, w) in compelled and dag.has_edge(w, v) and (w, v) in compelled:
                    compelled.add((u, v))
                    changed = True
                    break
            if (u, v) in compelled:
                continue
            # Meek R3: u — v with two distinct non-adjacent w1, w2
            # s.t. w1 → v and w2 → v and w1 — u and w2 — u
            parents_v = [w for w in dag.predecessors(v) if w != u and (w, v) in compelled]
            undirected_u = [
                w for w in dag.predecessors(u)
                if (u, w) not in compelled and (w, u) not in compelled
                and dag.has_edge(u, w)
            ] + [
                w for w in dag.successors(u)
                if (u, w) not in compelled and (w, u) not in compelled
                and dag.has_edge(w, u)
            ]
            for w1 in parents_v:
                if w1 in undirected_u or (dag.has_edge(w1, u) and (w1, u) not in compelled):
                    for w2 in parents_v:
                        if w2 == w1:
                            continue
                        if not dag.has_edge(w1, w2) and not dag.has_edge(w2, w1):
                            if w2 in undirected_u or (dag.has_edge(w2, u) and (w2, u) not in compelled):
                                compelled.add((u, v))
                                changed = True
                                break
                    if (u, v) in compelled:
                        break

    return compelled


def dag_to_cpdag(dag: nx.DiGraph) -> CPDAG:
    """Convert a DAG to its CPDAG (essential graph)."""
    return CPDAG.from_dag(dag)


def same_mec(dag1: nx.DiGraph, dag2: nx.DiGraph) -> bool:
    """Test if two DAGs are Markov-equivalent (same MEC)."""
    cpdag1 = CPDAG.from_dag(dag1)
    cpdag2 = CPDAG.from_dag(dag2)
    return (
        cpdag1._directed == cpdag2._directed
        and cpdag1._undirected == cpdag2._undirected
    )


def skeleton(dag: nx.DiGraph) -> nx.Graph:
    """Return the undirected skeleton of a DAG."""
    return dag.to_undirected()


def count_v_structures(dag: nx.DiGraph) -> int:
    """Count the number of v-structures (colliders) in a DAG."""
    return len(_v_structures(dag))
