"""
Efficient DAG representation for causal inference.

Provides topological sort, ancestor/descendant queries, d-separation checking
via the Bayes Ball algorithm, Markov blanket computation, and interventional
graph (mutilated DAG) construction.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import (
    AbstractSet,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np


class EdgeType(Enum):
    """Types of edges that can appear in causal graphs."""

    DIRECTED = auto()       # X -> Y
    BIDIRECTED = auto()     # X <-> Y  (latent common cause)
    UNDIRECTED = auto()     # X -- Y   (unknown orientation)
    PARTIALLY = auto()      # X o-> Y  (PAG circle-arrow)
    CIRCLE_CIRCLE = auto()  # X o-o Y  (PAG circle-circle)


class DAGRepresentation:
    """Efficient DAG representation for causal inference.

    Stores a directed acyclic graph with adjacency-list representation
    augmented by pre-computed caches for ancestor / descendant queries,
    topological ordering, and Markov blankets.

    Parameters
    ----------
    variables : list[str], optional
        Initial set of variable names.  More can be added later.
    """

    def __init__(self, variables: Optional[List[str]] = None) -> None:
        self._graph = nx.DiGraph()
        self._bidirected: Set[FrozenSet[str]] = set()
        self._topo_cache: Optional[List[str]] = None
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._descendant_cache: Dict[str, Set[str]] = {}
        if variables:
            for v in variables:
                self._graph.add_node(v)

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_node(self, name: str, **attrs) -> None:
        """Add a variable node to the DAG."""
        self._graph.add_node(name, **attrs)
        self._invalidate_caches()

    def add_edge(self, u: str, v: str, edge_type: EdgeType = EdgeType.DIRECTED) -> None:
        """Add an edge *u* → *v* (or bidirected *u* ↔ *v*).

        Raises ``ValueError`` if adding the directed edge would create a cycle.
        """
        if edge_type == EdgeType.BIDIRECTED:
            self._bidirected.add(frozenset({u, v}))
            for node in (u, v):
                if node not in self._graph:
                    self._graph.add_node(node)
            self._invalidate_caches()
            return

        for node in (u, v):
            if node not in self._graph:
                self._graph.add_node(node)

        # Cycle check: v must not be an ancestor of u
        if self.is_ancestor(v, u):
            raise ValueError(
                f"Adding edge {u} -> {v} would create a cycle "
                f"({v} is already an ancestor of {u})."
            )
        self._graph.add_edge(u, v)
        self._invalidate_caches()

    def remove_edge(self, u: str, v: str) -> None:
        """Remove the directed edge *u* → *v* if present."""
        if self._graph.has_edge(u, v):
            self._graph.remove_edge(u, v)
            self._invalidate_caches()
        else:
            key = frozenset({u, v})
            if key in self._bidirected:
                self._bidirected.discard(key)
                self._invalidate_caches()

    def remove_node(self, v: str) -> None:
        """Remove node *v* and all incident edges."""
        if v in self._graph:
            self._graph.remove_node(v)
            self._bidirected = {e for e in self._bidirected if v not in e}
            self._invalidate_caches()

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[str]:
        return list(self._graph.nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._graph.edges)

    @property
    def n_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self._graph.number_of_edges()

    def has_node(self, v: str) -> bool:
        return v in self._graph

    def has_edge(self, u: str, v: str) -> bool:
        return self._graph.has_edge(u, v)

    def has_bidirected(self, u: str, v: str) -> bool:
        return frozenset({u, v}) in self._bidirected

    def parents(self, v: str) -> List[str]:
        """Return the parents of *v* (nodes with directed edges into *v*)."""
        return list(self._graph.predecessors(v))

    def children(self, v: str) -> List[str]:
        """Return the children of *v* (nodes with directed edges from *v*)."""
        return list(self._graph.successors(v))

    def neighbors(self, v: str) -> List[str]:
        """All nodes adjacent to *v* (parents + children + bidirected partners)."""
        nbrs: Set[str] = set(self._graph.predecessors(v))
        nbrs |= set(self._graph.successors(v))
        for pair in self._bidirected:
            if v in pair:
                nbrs |= pair - {v}
        return list(nbrs)

    # ------------------------------------------------------------------
    # Ancestor / descendant queries  (BFS, cached)
    # ------------------------------------------------------------------

    def ancestors(self, v: str) -> Set[str]:
        """Return all ancestors of *v* (not including *v* itself)."""
        if v in self._ancestor_cache:
            return set(self._ancestor_cache[v])
        visited: Set[str] = set()
        queue = deque(self.parents(v))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.parents(node))
        self._ancestor_cache[v] = visited
        return set(visited)

    def descendants(self, v: str) -> Set[str]:
        """Return all descendants of *v* (not including *v* itself)."""
        if v in self._descendant_cache:
            return set(self._descendant_cache[v])
        visited: Set[str] = set()
        queue = deque(self.children(v))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.children(node))
        self._descendant_cache[v] = visited
        return set(visited)

    def is_ancestor(self, u: str, v: str) -> bool:
        """Return ``True`` if *u* is an ancestor of *v*."""
        return u in self.ancestors(v)

    def is_descendant(self, u: str, v: str) -> bool:
        """Return ``True`` if *u* is a descendant of *v*."""
        return u in self.descendants(v)

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return a topological ordering of the nodes.

        Raises ``ValueError`` if the graph contains a cycle.
        """
        if self._topo_cache is not None:
            return list(self._topo_cache)
        try:
            order = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError("Graph contains a cycle.") from exc
        self._topo_cache = order
        return list(order)

    # ------------------------------------------------------------------
    # d-separation  (Bayes Ball algorithm – Shachter 1998)
    # ------------------------------------------------------------------

    def d_separated(
        self,
        x: AbstractSet[str],
        y: AbstractSet[str],
        z: AbstractSet[str],
    ) -> bool:
        """Test whether *x* ⊥ *y* | *z* using the Bayes Ball algorithm.

        Parameters
        ----------
        x, y : sets of variable names (sources / targets)
        z : conditioning set

        Returns
        -------
        bool – ``True`` if *x* and *y* are d-separated given *z*.
        """
        x = set(x) if not isinstance(x, set) else x
        y = set(y) if not isinstance(y, set) else y
        z = set(z) if not isinstance(z, set) else z

        reachable = self._bayes_ball_reachable(x, z)
        return len(reachable & y) == 0

    def _bayes_ball_reachable(
        self, sources: Set[str], conditioned: Set[str]
    ) -> Set[str]:
        """Return the set of nodes reachable from *sources* given *conditioned*.

        Implements the Bayes Ball algorithm.  Each entry in the work-queue is
        a ``(node, direction)`` pair where *direction* is ``'up'`` (entering
        from a child) or ``'down'`` (entering from a parent).
        """
        # Pre-compute ancestors of conditioned set for the collider rule
        ancestors_of_z: Set[str] = set()
        for c in conditioned:
            ancestors_of_z |= self.ancestors(c)
            ancestors_of_z.add(c)

        visited_up: Set[str] = set()
        visited_down: Set[str] = set()
        reachable: Set[str] = set()

        queue: deque[Tuple[str, str]] = deque()
        for s in sources:
            queue.append((s, "up"))
            queue.append((s, "down"))

        while queue:
            node, direction = queue.popleft()

            if direction == "up" and node not in visited_up:
                visited_up.add(node)
                reachable.add(node)

                if node not in conditioned:
                    # Pass through non-conditioned: send ball up to parents
                    for p in self.parents(node):
                        queue.append((p, "up"))
                    # and down to children
                    for c in self.children(node):
                        queue.append((c, "down"))

            elif direction == "down" and node not in visited_down:
                visited_down.add(node)
                reachable.add(node)

                if node not in conditioned:
                    # Non-conditioned: ball passes down to children
                    for c in self.children(node):
                        queue.append((c, "down"))
                else:
                    # Conditioned (or ancestor of conditioned): ball bounces up
                    for p in self.parents(node):
                        queue.append((p, "up"))

                # Collider activation: if node or any descendant is conditioned
                if node in ancestors_of_z:
                    for p in self.parents(node):
                        if (p, "up") not in visited_up:
                            queue.append((p, "up"))

        return reachable

    # ------------------------------------------------------------------
    # Markov blanket
    # ------------------------------------------------------------------

    def markov_blanket(self, v: str) -> Set[str]:
        """Return the Markov blanket of *v*.

        The Markov blanket consists of *v*'s parents, children, and the
        other parents of *v*'s children (co-parents / spouses).
        """
        blanket: Set[str] = set()
        blanket |= set(self.parents(v))
        for c in self.children(v):
            blanket.add(c)
            blanket |= set(self.parents(c))
        # Also include bidirected partners
        for pair in self._bidirected:
            if v in pair:
                blanket |= pair - {v}
        blanket.discard(v)
        return blanket

    # ------------------------------------------------------------------
    # Interventional / mutilated graph
    # ------------------------------------------------------------------

    def mutilate(self, intervention_targets: Iterable[str]) -> "DAGRepresentation":
        """Return a new DAG with incoming edges to *intervention_targets* removed.

        This is the *mutilated graph* used to represent ``do(X = x)``
        interventions in Pearl's framework.
        """
        targets = set(intervention_targets)
        new_dag = DAGRepresentation()
        for node in self._graph.nodes:
            new_dag.add_node(node, **dict(self._graph.nodes[node]))
        for u, v in self._graph.edges:
            if v not in targets:
                new_dag._graph.add_edge(u, v)
        # Preserve bidirected edges not involving targets
        for pair in self._bidirected:
            if not pair & targets:
                new_dag._bidirected.add(pair)
        new_dag._invalidate_caches()
        return new_dag

    def augmented_graph(self, intervention_targets: Iterable[str]) -> "DAGRepresentation":
        """Return augmented DAG with intervention indicator nodes.

        For each target T, add a node ``I_T`` with edge ``I_T -> T`` and
        remove all other incoming edges to T.  This is used for computing
        identification formulas.
        """
        targets = set(intervention_targets)
        aug = self.mutilate(targets)
        for t in targets:
            indicator = f"I_{t}"
            aug.add_node(indicator)
            aug._graph.add_edge(indicator, t)
        aug._invalidate_caches()
        return aug

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def get_paths(
        self,
        source: str,
        target: str,
        directed_only: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """Enumerate all simple paths from *source* to *target*.

        Parameters
        ----------
        directed_only : bool
            If ``True`` only follow directed edges in causal direction.
        max_length : int, optional
            Maximum path length (number of edges).
        """
        if directed_only:
            cutoff = max_length if max_length else self.n_nodes
            return list(nx.all_simple_paths(self._graph, source, target, cutoff=cutoff))

        # Undirected paths (treating graph as undirected)
        undirected = self._graph.to_undirected()
        for pair in self._bidirected:
            u, v = tuple(pair)
            undirected.add_edge(u, v)
        cutoff = max_length if max_length else self.n_nodes
        return list(nx.all_simple_paths(undirected, source, target, cutoff=cutoff))

    def get_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """Return all directed paths from *source* to *target*."""
        return self.get_paths(source, target, directed_only=True)

    def get_backdoor_paths(
        self,
        treatment: str,
        outcome: str,
    ) -> List[List[str]]:
        """Return all non-causal (backdoor) paths between *treatment* and *outcome*.

        A backdoor path is a path that starts with an edge *into* treatment.
        """
        all_paths = self.get_paths(treatment, outcome, directed_only=False)
        causal = set(map(tuple, self.get_causal_paths(treatment, outcome)))
        return [p for p in all_paths if tuple(p) not in causal]

    # ------------------------------------------------------------------
    # Moral graph
    # ------------------------------------------------------------------

    def moral_graph(self) -> nx.Graph:
        """Return the moral graph (marry parents of common children, drop directions)."""
        moral = self._graph.to_undirected()
        for node in self._graph.nodes:
            pars = self.parents(node)
            for i in range(len(pars)):
                for j in range(i + 1, len(pars)):
                    moral.add_edge(pars[i], pars[j])
        for pair in self._bidirected:
            u, v = tuple(pair)
            moral.add_edge(u, v)
        return moral

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the internal ``networkx.DiGraph``."""
        return self._graph.copy()

    @classmethod
    def from_networkx(cls, G: nx.DiGraph) -> "DAGRepresentation":
        """Construct a ``DAGRepresentation`` from a ``networkx.DiGraph``."""
        dag = cls()
        dag._graph = G.copy()
        if not nx.is_directed_acyclic_graph(dag._graph):
            raise ValueError("Provided graph is not a DAG.")
        return dag

    @classmethod
    def from_adjacency_matrix(
        cls, matrix: np.ndarray, labels: Optional[List[str]] = None
    ) -> "DAGRepresentation":
        """Construct from an adjacency matrix.  ``matrix[i][j] = 1`` ⇒ edge *i* → *j*."""
        n = matrix.shape[0]
        if labels is None:
            labels = [f"X{i}" for i in range(n)]
        dag = cls(variables=labels)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] != 0:
                    dag._graph.add_edge(labels[i], labels[j])
        if not nx.is_directed_acyclic_graph(dag._graph):
            raise ValueError("Provided matrix does not represent a DAG.")
        return dag

    def to_adjacency_matrix(self, order: Optional[List[str]] = None) -> np.ndarray:
        """Return adjacency matrix.  ``A[i][j] = 1`` ⇔ edge ``order[i]`` → ``order[j]``."""
        if order is None:
            order = sorted(self._graph.nodes)
        idx = {v: i for i, v in enumerate(order)}
        n = len(order)
        mat = np.zeros((n, n), dtype=int)
        for u, v in self._graph.edges:
            mat[idx[u], idx[v]] = 1
        return mat

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_dag(self) -> bool:
        """Return ``True`` if the directed component is acyclic."""
        return nx.is_directed_acyclic_graph(self._graph)

    def validate(self) -> List[str]:
        """Return a list of validation issues (empty if valid)."""
        issues: List[str] = []
        if not self.is_dag():
            issues.append("Graph contains a directed cycle.")
        isolated = list(nx.isolates(self._graph))
        if isolated:
            issues.append(f"Isolated nodes: {isolated}")
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invalidate_caches(self) -> None:
        self._topo_cache = None
        self._ancestor_cache.clear()
        self._descendant_cache.clear()

    def __repr__(self) -> str:
        return (
            f"DAGRepresentation(nodes={self.n_nodes}, "
            f"directed_edges={self.n_edges}, "
            f"bidirected_edges={len(self._bidirected)})"
        )

    def __contains__(self, v: str) -> bool:
        return v in self._graph
