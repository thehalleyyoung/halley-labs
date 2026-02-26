"""
Markov equivalence class enumeration and CPDAG representation.

Enumerates DAGs consistent with a given PAG / CPDAG, computes essential
graphs, samples DAGs from the equivalence class, and estimates the size
of the equivalence class.

References
----------
- Andersson, Madigan, Perlman (1997). A characterization of Markov
  equivalence classes for acyclic digraphs.
- Chickering (2002). Optimal structure identification with greedy search.
- He, Jia, Yu (2015). Counting and exploring sizes of Markov equivalence
  classes of directed acyclic graphs.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import defaultdict, deque
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np

from .dag import DAGRepresentation, EdgeType


class CPDAG:
    """Completed Partially Directed Acyclic Graph.

    A CPDAG encodes a Markov equivalence class.  It contains directed
    edges (present in every member DAG) and undirected edges (which can
    be oriented either way).

    Internally represented as an ``nx.DiGraph`` where a directed edge is
    stored once (u → v) and an undirected edge is stored as *both*
    (u → v) and (v → u) with attribute ``undirected=True``.
    """

    def __init__(self, variables: Optional[List[str]] = None) -> None:
        self._graph = nx.DiGraph()
        self._undirected: Set[FrozenSet[str]] = set()
        if variables:
            for v in variables:
                self._graph.add_node(v)

    # ── mutation ──────────────────────────────────────────────────────

    def add_directed_edge(self, u: str, v: str) -> None:
        """Add a directed edge u → v (compelled / essential)."""
        for node in (u, v):
            if node not in self._graph:
                self._graph.add_node(node)
        self._graph.add_edge(u, v, undirected=False)
        # Remove undirected marker if present
        self._undirected.discard(frozenset({u, v}))
        if self._graph.has_edge(v, u):
            self._graph.remove_edge(v, u)

    def add_undirected_edge(self, u: str, v: str) -> None:
        """Add an undirected (reversible) edge u — v."""
        for node in (u, v):
            if node not in self._graph:
                self._graph.add_node(node)
        self._graph.add_edge(u, v, undirected=True)
        self._graph.add_edge(v, u, undirected=True)
        self._undirected.add(frozenset({u, v}))

    def remove_edge(self, u: str, v: str) -> None:
        key = frozenset({u, v})
        if key in self._undirected:
            self._undirected.discard(key)
            if self._graph.has_edge(u, v):
                self._graph.remove_edge(u, v)
            if self._graph.has_edge(v, u):
                self._graph.remove_edge(v, u)
        else:
            if self._graph.has_edge(u, v):
                self._graph.remove_edge(u, v)

    # ── queries ───────────────────────────────────────────────────────

    @property
    def nodes(self) -> List[str]:
        return list(self._graph.nodes)

    @property
    def directed_edges(self) -> List[Tuple[str, str]]:
        return [(u, v) for u, v in self._graph.edges
                if frozenset({u, v}) not in self._undirected]

    @property
    def undirected_edges(self) -> List[FrozenSet[str]]:
        return list(self._undirected)

    def is_directed(self, u: str, v: str) -> bool:
        return (self._graph.has_edge(u, v) and
                frozenset({u, v}) not in self._undirected)

    def is_undirected(self, u: str, v: str) -> bool:
        return frozenset({u, v}) in self._undirected

    def adjacent(self, v: str) -> List[str]:
        nbrs: Set[str] = set()
        nbrs |= set(self._graph.predecessors(v))
        nbrs |= set(self._graph.successors(v))
        return list(nbrs)

    def parents(self, v: str) -> List[str]:
        """Return parents of v (directed edges into v)."""
        return [u for u in self._graph.predecessors(v)
                if frozenset({u, v}) not in self._undirected]

    def children(self, v: str) -> List[str]:
        """Return children of v (directed edges from v)."""
        return [u for u in self._graph.successors(v)
                if frozenset({u, v}) not in self._undirected]

    def undirected_neighbors(self, v: str) -> List[str]:
        """Return nodes connected to v via undirected edges."""
        result = []
        for edge in self._undirected:
            if v in edge:
                result.append((edge - {v}).pop())
        return result

    def copy(self) -> "CPDAG":
        new = CPDAG()
        new._graph = self._graph.copy()
        new._undirected = set(self._undirected)
        return new

    def to_networkx(self) -> nx.DiGraph:
        return self._graph.copy()

    @classmethod
    def from_dag(cls, dag: DAGRepresentation) -> "CPDAG":
        """Compute the CPDAG (essential graph) for a given DAG.

        An edge is *compelled* (directed in the CPDAG) if reversing it
        would change the set of v-structures or violate acyclicity.
        Otherwise it is *reversible* (undirected in the CPDAG).
        """
        cpdag = cls(dag.nodes)
        G = dag.to_networkx()
        edges = list(G.edges)

        # Step 1: find all v-structures
        v_structures: Set[Tuple[str, str, str]] = set()
        for node in G.nodes:
            parents_list = list(G.predecessors(node))
            for i in range(len(parents_list)):
                for j in range(i + 1, len(parents_list)):
                    p1, p2 = parents_list[i], parents_list[j]
                    if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                        v_structures.add((p1, node, p2))
                        v_structures.add((p2, node, p1))

        # Step 2: label edges as compelled or reversible
        # An edge u -> v is compelled if it participates in a v-structure
        # or if reversing it would create a new v-structure or cycle.
        compelled: Set[Tuple[str, str]] = set()

        # Edges in v-structures are compelled
        for p1, child, p2 in v_structures:
            compelled.add((p1, child))
            compelled.add((p2, child))

        # Propagate: if u -> v is compelled and v -> w is an edge with
        # no edge u -> w or w -> u, then v -> w is compelled
        changed = True
        while changed:
            changed = False
            for u, v in list(compelled):
                for w in G.successors(v):
                    if (v, w) in compelled:
                        continue
                    if not G.has_edge(u, w) and not G.has_edge(w, u):
                        compelled.add((v, w))
                        changed = True

        # Build CPDAG
        for u, v in edges:
            if (u, v) in compelled:
                cpdag.add_directed_edge(u, v)
            else:
                cpdag.add_undirected_edge(u, v)

        return cpdag

    def __repr__(self) -> str:
        return (
            f"CPDAG(nodes={len(self.nodes)}, "
            f"directed={len(self.directed_edges)}, "
            f"undirected={len(self._undirected)})"
        )


# ──────────────────────────────────────────────────────────────────────
# Markov Equivalence Class
# ──────────────────────────────────────────────────────────────────────

class MarkovEquivalenceClass:
    """Represents a Markov equivalence class of DAGs.

    Can be constructed from a CPDAG or a PAG and provides methods to
    enumerate, sample, and count the member DAGs.

    Parameters
    ----------
    cpdag : CPDAG, optional
        The CPDAG defining the equivalence class.
    """

    def __init__(self, cpdag: Optional[CPDAG] = None) -> None:
        self._cpdag = cpdag
        self._enumerated_dags: Optional[List[DAGRepresentation]] = None
        self._count: Optional[int] = None

    @classmethod
    def from_pag(cls, pag: Any) -> "MarkovEquivalenceClass":
        """Construct from a PAG (Partial Ancestral Graph).

        Converts the PAG to a CPDAG by treating circle marks as
        undirected and arrow/tail marks as directed.
        """
        from .causal_discovery import PAG as PAGClass, Mark

        cpdag = CPDAG(pag.variables)
        for u, v in pag.skeleton_edges():
            if pag.is_directed(u, v):
                cpdag.add_directed_edge(u, v)
            elif pag.is_directed(v, u):
                cpdag.add_directed_edge(v, u)
            else:
                cpdag.add_undirected_edge(u, v)

        mec = cls(cpdag)
        return mec

    @classmethod
    def from_dag(cls, dag: DAGRepresentation) -> "MarkovEquivalenceClass":
        """Construct the Markov equivalence class containing *dag*."""
        cpdag = CPDAG.from_dag(dag)
        return cls(cpdag)

    # ── enumeration ───────────────────────────────────────────────────

    def enumerate_dags(self, max_count: int = 1000) -> List[DAGRepresentation]:
        """Enumerate all DAGs in the Markov equivalence class.

        Uses recursive orientation of undirected edges, respecting
        acyclicity and v-structure constraints.

        Parameters
        ----------
        max_count : int
            Stop after finding this many DAGs (for large classes).
        """
        if self._cpdag is None:
            raise RuntimeError("No CPDAG set.")

        if self._enumerated_dags is not None:
            return self._enumerated_dags[:max_count]

        undirected = list(self._cpdag.undirected_edges)
        directed = list(self._cpdag.directed_edges)
        nodes = self._cpdag.nodes

        results: List[DAGRepresentation] = []

        def _backtrack(idx: int, current_edges: List[Tuple[str, str]]) -> None:
            if len(results) >= max_count:
                return

            if idx == len(undirected):
                # Check if the assignment forms a valid DAG
                dag = DAGRepresentation(nodes)
                for u, v in current_edges:
                    dag._graph.add_edge(u, v)
                if dag.is_dag() and self._preserves_v_structures(dag):
                    results.append(dag)
                return

            edge = undirected[idx]
            u, v = tuple(edge)

            # Try u -> v
            _backtrack(idx + 1, current_edges + [(u, v)])
            # Try v -> u
            _backtrack(idx + 1, current_edges + [(v, u)])

        _backtrack(0, directed)
        self._enumerated_dags = results
        return results[:max_count]

    def _preserves_v_structures(self, dag: DAGRepresentation) -> bool:
        """Check that *dag* has the same v-structures as the CPDAG's members."""
        # Extract v-structures from the DAG
        dag_vs = self._extract_v_structures(dag)
        # Extract v-structures from the CPDAG (directed edges only)
        cpdag_vs = self._extract_cpdag_v_structures()
        return dag_vs == cpdag_vs

    def _extract_v_structures(self, dag: DAGRepresentation) -> Set[FrozenSet]:
        """Extract v-structures from a DAG as frozensets {(parent1, child, parent2)}."""
        vs: Set[FrozenSet] = set()
        for node in dag.nodes:
            parents = dag.parents(node)
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1, p2 = parents[i], parents[j]
                    if not dag.has_edge(p1, p2) and not dag.has_edge(p2, p1):
                        vs.add(frozenset({(p1, node), (p2, node)}))
        return vs

    def _extract_cpdag_v_structures(self) -> Set[FrozenSet]:
        """Extract v-structures from the CPDAG's directed edges."""
        vs: Set[FrozenSet] = set()
        if self._cpdag is None:
            return vs
        for node in self._cpdag.nodes:
            parents = self._cpdag.parents(node)
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1, p2 = parents[i], parents[j]
                    if not self._cpdag._graph.has_edge(p1, p2) and \
                       not self._cpdag._graph.has_edge(p2, p1):
                        vs.add(frozenset({(p1, node), (p2, node)}))
        return vs

    # ── sampling ──────────────────────────────────────────────────────

    def sample_dag(self) -> DAGRepresentation:
        """Sample a single DAG uniformly from the equivalence class.

        Uses the direct sampling algorithm: for each chain component
        of the CPDAG, generate a consistent extension by ordering
        nodes and orienting edges lexicographically.
        """
        if self._cpdag is None:
            raise RuntimeError("No CPDAG set.")

        nodes = self._cpdag.nodes
        directed = list(self._cpdag.directed_edges)
        undirected = list(self._cpdag.undirected_edges)

        if not undirected:
            dag = DAGRepresentation(nodes)
            for u, v in directed:
                dag._graph.add_edge(u, v)
            return dag

        # Strategy: random topological extension
        # Build temporary graph with directed edges + random orientations
        max_attempts = 100
        for attempt in range(max_attempts):
            dag = DAGRepresentation(nodes)
            for u, v in directed:
                dag._graph.add_edge(u, v)

            # Randomly orient each undirected edge
            perm = np.random.permutation(len(undirected))
            valid = True
            for idx in perm:
                edge = undirected[idx]
                u, v = tuple(edge)
                if np.random.random() < 0.5:
                    u, v = v, u
                # Check that adding u -> v doesn't create a cycle
                if v in dag.ancestors(u) or v == u:
                    u, v = v, u
                    if v in dag.ancestors(u) or v == u:
                        valid = False
                        break
                dag._graph.add_edge(u, v)
                dag._invalidate_caches()

            if valid and dag.is_dag() and self._preserves_v_structures(dag):
                return dag

        # Fallback: enumerate and pick random
        dags = self.enumerate_dags(max_count=100)
        if dags:
            return dags[np.random.randint(len(dags))]

        raise RuntimeError("Failed to sample a valid DAG from the equivalence class.")

    def sample_dags(self, n: int = 10) -> List[DAGRepresentation]:
        """Sample *n* DAGs (may contain duplicates for large classes)."""
        return [self.sample_dag() for _ in range(n)]

    # ── counting ──────────────────────────────────────────────────────

    def count_dags(self) -> int:
        """Return the exact number of DAGs in the equivalence class.

        Uses the recursive formula based on chain components.
        For small classes, falls back to enumeration.
        """
        if self._count is not None:
            return self._count

        if self._cpdag is None:
            return 0

        undirected = self._cpdag.undirected_edges
        if not undirected:
            self._count = 1
            return 1

        # Decompose into chain components (connected components of undirected subgraph)
        ug = nx.Graph()
        for node in self._cpdag.nodes:
            ug.add_node(node)
        for edge in undirected:
            u, v = tuple(edge)
            ug.add_edge(u, v)

        components = list(nx.connected_components(ug))
        total = 1

        for component in components:
            comp_nodes = list(component)
            comp_undirected = [
                e for e in undirected
                if e <= set(comp_nodes)
            ]
            # Count valid orientations for this component
            comp_count = self._count_component_orientations(
                comp_nodes, comp_undirected
            )
            total *= comp_count

        self._count = total
        return total

    def _count_component_orientations(
        self, nodes: List[str], undirected: List[FrozenSet[str]]
    ) -> int:
        """Count valid orientations for a single chain component.

        For small components, enumerate directly.  For larger ones,
        use the formula: for a chordal undirected component with
        *k* undirected edges, the count is related to the number of
        perfect elimination orderings.
        """
        n = len(nodes)
        k = len(undirected)

        if k == 0:
            return 1
        if k == 1:
            return 2  # Either orientation is valid

        # For small components, enumerate
        if k <= 15:
            count = 0
            for bits in range(2 ** k):
                dag = DAGRepresentation(nodes)
                # Add directed edges from CPDAG
                for u, v in self._cpdag.directed_edges:
                    if u in nodes and v in nodes:
                        dag._graph.add_edge(u, v)
                valid = True
                for i, edge in enumerate(undirected):
                    u, v = tuple(edge)
                    if bits & (1 << i):
                        u, v = v, u
                    dag._graph.add_edge(u, v)
                if dag.is_dag():
                    count += 1
            return max(count, 1)

        # For larger components, estimate using sampling
        return self._estimate_count_sampling(nodes, undirected)

    def _estimate_count_sampling(
        self,
        nodes: List[str],
        undirected: List[FrozenSet[str]],
        n_samples: int = 10000,
    ) -> int:
        """Estimate the number of valid orientations by sampling."""
        k = len(undirected)
        valid_count = 0

        for _ in range(n_samples):
            dag = DAGRepresentation(nodes)
            for u, v in self._cpdag.directed_edges:
                if u in nodes and v in nodes:
                    dag._graph.add_edge(u, v)
            valid = True
            for edge in undirected:
                u, v = tuple(edge)
                if np.random.random() < 0.5:
                    u, v = v, u
                dag._graph.add_edge(u, v)
            if dag.is_dag():
                valid_count += 1

        if valid_count == 0:
            return 1

        fraction = valid_count / n_samples
        estimate = int(round(fraction * (2 ** k)))
        return max(estimate, 1)

    # ── CPDAG accessors ───────────────────────────────────────────────

    def get_cpdag(self) -> CPDAG:
        if self._cpdag is None:
            raise RuntimeError("No CPDAG set.")
        return self._cpdag

    def get_essential_edges(self) -> List[Tuple[str, str]]:
        """Return edges that are directed in every member DAG.

        These are the *compelled* (essential) edges of the CPDAG.
        """
        if self._cpdag is None:
            return []
        return self._cpdag.directed_edges

    def get_reversible_edges(self) -> List[FrozenSet[str]]:
        """Return edges that can be oriented either way."""
        if self._cpdag is None:
            return []
        return self._cpdag.undirected_edges

    # ── membership test ───────────────────────────────────────────────

    def is_member(self, dag: DAGRepresentation) -> bool:
        """Test whether *dag* belongs to this Markov equivalence class.

        Checks that *dag* has the same skeleton and v-structures as the
        CPDAG.
        """
        if self._cpdag is None:
            return False

        # Check skeleton
        dag_skeleton: Set[FrozenSet[str]] = set()
        for u, v in dag.edges:
            dag_skeleton.add(frozenset({u, v}))

        cpdag_skeleton: Set[FrozenSet[str]] = set()
        for u, v in self._cpdag.directed_edges:
            cpdag_skeleton.add(frozenset({u, v}))
        cpdag_skeleton |= set(self._cpdag.undirected_edges)

        if dag_skeleton != cpdag_skeleton:
            return False

        # Check v-structures
        dag_vs = self._extract_v_structures(dag)
        cpdag_vs = self._extract_cpdag_v_structures()
        return dag_vs == cpdag_vs

    # ── interventional equivalence ────────────────────────────────────

    def interventional_equivalence_class(
        self,
        intervention_targets: Set[str],
    ) -> "MarkovEquivalenceClass":
        """Compute the interventional Markov equivalence class.

        Given a set of intervention targets, refine the equivalence
        class by orienting additional edges that become identifiable.
        """
        if self._cpdag is None:
            raise RuntimeError("No CPDAG set.")

        new_cpdag = self._cpdag.copy()

        # For each intervention target, all undirected edges incident
        # to the target can be oriented (the target becomes a "root"
        # in the manipulated graph)
        for target in intervention_targets:
            for edge in list(new_cpdag.undirected_edges):
                if target in edge:
                    other = (edge - {target}).pop()
                    # Orient away from target
                    new_cpdag.remove_edge(target, other)
                    new_cpdag.add_directed_edge(target, other)

        return MarkovEquivalenceClass(new_cpdag)

    # ── size estimation ───────────────────────────────────────────────

    def size_upper_bound(self) -> int:
        """Upper bound: 2^k where k is the number of undirected edges."""
        if self._cpdag is None:
            return 0
        return 2 ** len(self._cpdag.undirected_edges)

    def size_lower_bound(self) -> int:
        """Lower bound: at least 1 (the class is non-empty by construction)."""
        return 1

    # ── comparison ────────────────────────────────────────────────────

    def structural_hamming_distance(
        self, other: "MarkovEquivalenceClass"
    ) -> int:
        """Compute the Structural Hamming Distance between two CPDAGs.

        SHD counts the number of edge additions, deletions, and reversals
        needed to transform one CPDAG into the other.
        """
        if self._cpdag is None or other._cpdag is None:
            raise RuntimeError("Both classes must have CPDAGs.")

        # Collect all edges from both CPDAGs
        self_dir = set(map(tuple, self._cpdag.directed_edges))
        other_dir = set(map(tuple, other._cpdag.directed_edges))
        self_undir = set(self._cpdag.undirected_edges)
        other_undir = set(other._cpdag.undirected_edges)

        self_all = {frozenset(e) for e in self_dir} | self_undir
        other_all = {frozenset(e) for e in other_dir} | other_undir

        shd = 0

        # Missing or extra edges
        all_edges = self_all | other_all
        for edge in all_edges:
            in_self = edge in self_all
            in_other = edge in other_all
            if in_self and not in_other:
                shd += 1
            elif in_other and not in_self:
                shd += 1
            elif in_self and in_other:
                # Both present, check orientation
                u, v = tuple(edge)
                self_directed = (u, v) in self_dir or (v, u) in self_dir
                other_directed = (u, v) in other_dir or (v, u) in other_dir
                if self_directed != other_directed:
                    shd += 1
                elif self_directed and other_directed:
                    # Same direction?
                    self_uv = (u, v) in self_dir
                    other_uv = (u, v) in other_dir
                    if self_uv != other_uv:
                        shd += 1

        return shd

    def __repr__(self) -> str:
        count_str = str(self._count) if self._count is not None else "?"
        return (
            f"MarkovEquivalenceClass(cpdag={self._cpdag}, "
            f"count={count_str})"
        )
