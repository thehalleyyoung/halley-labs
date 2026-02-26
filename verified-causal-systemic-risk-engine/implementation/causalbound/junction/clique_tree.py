"""
Clique-tree (junction-tree) construction and manipulation.

Builds a clique tree from a triangulated moral graph using maximum
spanning-tree construction over the clique-intersection graph.
Provides clique ordering schedules for message passing and verifies
the running-intersection property.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from .potential_table import PotentialTable, multiply_potentials


# ------------------------------------------------------------------ #
#  Clique node
# ------------------------------------------------------------------ #

@dataclass
class CliqueNode:
    """A single node (clique) in the junction tree.

    Attributes
    ----------
    variables : frozenset[str]
        The variables that belong to this clique.
    potential : PotentialTable or None
        Joint potential over the clique's variables.
    index : int
        Unique sequential identifier within the tree.
    """

    variables: FrozenSet[str]
    potential: Optional[PotentialTable] = None
    index: int = 0

    # Assigned CPDs (before combination)
    _assigned_cpds: List[PotentialTable] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.variables)

    def assign_cpd(self, cpd: PotentialTable) -> None:
        """Assign a conditional-probability table to this clique."""
        self._assigned_cpds.append(cpd)

    def initialize_potential(self, cardinalities: Dict[str, int]) -> None:
        """Combine all assigned CPDs into a single clique potential."""
        if not self._assigned_cpds:
            self.potential = PotentialTable(
                sorted(self.variables), cardinalities
            )
        elif len(self._assigned_cpds) == 1:
            self.potential = self._assigned_cpds[0].copy()
            # Expand to full clique scope if needed
            missing = self.variables - set(self.potential.variables)
            if missing:
                unit = PotentialTable(sorted(missing), cardinalities)
                self.potential = self.potential.multiply(unit)
        else:
            combined = multiply_potentials(self._assigned_cpds)
            missing = self.variables - set(combined.variables)
            if missing:
                unit = PotentialTable(sorted(missing), cardinalities)
                combined = combined.multiply(unit)
            self.potential = combined

    def reset_potential(self, cardinalities: Dict[str, int]) -> None:
        """Re-initialize potential from assigned CPDs (after mutation)."""
        self.initialize_potential(cardinalities)

    def __hash__(self) -> int:
        return hash(self.variables)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CliqueNode):
            return NotImplemented
        return self.variables == other.variables

    def __repr__(self) -> str:
        return f"CliqueNode({set(self.variables)}, idx={self.index})"


# ------------------------------------------------------------------ #
#  Separator
# ------------------------------------------------------------------ #

@dataclass
class Separator:
    """Edge between two adjacent cliques, holding the separator set."""

    clique_a: CliqueNode
    clique_b: CliqueNode
    variables: FrozenSet[str]
    potential: Optional[PotentialTable] = None

    @property
    def size(self) -> int:
        return len(self.variables)

    def initialize_potential(self, cardinalities: Dict[str, int]) -> None:
        if self.variables:
            self.potential = PotentialTable(sorted(self.variables), cardinalities)
        else:
            self.potential = PotentialTable([], {}, np.array(1.0))

    def __repr__(self) -> str:
        return (
            f"Separator({set(self.clique_a.variables)} -- "
            f"{set(self.variables)} -- {set(self.clique_b.variables)})"
        )


# ------------------------------------------------------------------ #
#  Clique tree
# ------------------------------------------------------------------ #

class CliqueTree:
    """Junction tree (clique tree) data structure.

    Supports construction, clique ordering for message passing, separator
    management, and verification of the running-intersection property.

    Parameters
    ----------
    cardinalities : dict[str, int]
        Number of discrete states for every variable in the model.
    """

    def __init__(self, cardinalities: Dict[str, int]) -> None:
        self.cardinalities = dict(cardinalities)
        self._cliques: List[CliqueNode] = []
        self._adjacency: Dict[int, Dict[int, Separator]] = defaultdict(dict)
        self._var_to_cliques: Dict[str, Set[int]] = defaultdict(set)
        self._root: Optional[int] = None
        self._next_index: int = 0

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def add_clique(self, variables: Iterable[str]) -> CliqueNode:
        """Add a maximal clique to the tree (not yet connected)."""
        vs = frozenset(variables)
        node = CliqueNode(variables=vs, index=self._next_index)
        self._cliques.append(node)
        for v in vs:
            self._var_to_cliques[v].add(node.index)
        self._next_index += 1
        return node

    def connect(self, idx_a: int, idx_b: int) -> Separator:
        """Connect two cliques and create the separator between them."""
        ca = self._cliques[idx_a]
        cb = self._cliques[idx_b]
        sep_vars = ca.variables & cb.variables
        sep = Separator(clique_a=ca, clique_b=cb, variables=sep_vars)
        self._adjacency[idx_a][idx_b] = sep
        self._adjacency[idx_b][idx_a] = sep
        return sep

    def get_separator(self, idx_a: int, idx_b: int) -> Optional[Separator]:
        """Return the separator between two adjacent cliques, or None."""
        return self._adjacency.get(idx_a, {}).get(idx_b)

    @property
    def cliques(self) -> List[CliqueNode]:
        return self._cliques

    @property
    def num_cliques(self) -> int:
        return len(self._cliques)

    def neighbors(self, idx: int) -> List[int]:
        """Return indices of cliques adjacent to clique *idx*."""
        return list(self._adjacency.get(idx, {}).keys())

    # ------------------------------------------------------------------ #
    #  Tree construction from triangulated graph
    # ------------------------------------------------------------------ #

    @classmethod
    def from_triangulated_graph(
        cls,
        graph: Dict[str, Set[str]],
        cardinalities: Dict[str, int],
    ) -> "CliqueTree":
        """Build a junction tree from a triangulated (chordal) graph.

        Steps:
        1. Find all maximal cliques via perfect-elimination ordering.
        2. Build clique-intersection graph with edge weight = separator size.
        3. Compute maximum spanning tree of the intersection graph.
        """
        cliques = _find_maximal_cliques(graph)
        if not cliques:
            tree = cls(cardinalities)
            return tree

        tree = cls(cardinalities)
        nodes = [tree.add_clique(c) for c in cliques]

        if len(nodes) == 1:
            tree._root = 0
            return tree

        # Build intersection graph edges
        edges: List[Tuple[int, int, int]] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sep = nodes[i].variables & nodes[j].variables
                if sep:
                    edges.append((len(sep), i, j))

        # Maximum spanning tree via Kruskal's (sort descending by weight)
        edges.sort(key=lambda e: -e[0])
        parent = list(range(len(nodes)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        for weight, i, j in edges:
            if union(i, j):
                tree.connect(i, j)

        tree._root = 0
        return tree

    # ------------------------------------------------------------------ #
    #  Root selection & message schedules
    # ------------------------------------------------------------------ #

    def find_root(self, target_variable: Optional[str] = None) -> int:
        """Find a good root clique.

        If *target_variable* is given, pick the smallest clique that
        contains it.  Otherwise use the stored root or pick clique 0.
        """
        if target_variable is not None:
            candidates = self._var_to_cliques.get(target_variable, set())
            if candidates:
                return min(candidates, key=lambda i: self._cliques[i].size)
        if self._root is not None:
            return self._root
        return 0

    def set_root(self, idx: int) -> None:
        self._root = idx

    def get_message_schedule(
        self, root: Optional[int] = None
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return (collect_order, distribute_order) for message passing.

        ``collect_order``: leaves → root (upward pass).
        ``distribute_order``: root → leaves (downward pass).
        """
        if root is None:
            root = self.find_root()

        collect: List[Tuple[int, int]] = []
        distribute: List[Tuple[int, int]] = []

        visited: Set[int] = set()
        stack: List[Tuple[int, Optional[int]]] = [(root, None)]
        order: List[Tuple[int, Optional[int]]] = []

        # DFS to build postorder (leaves first)
        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append((node, parent))
            for nb in self.neighbors(node):
                if nb not in visited:
                    stack.append((nb, node))

        # Collect: reverse postorder, send from child to parent
        for node, parent in reversed(order):
            if parent is not None:
                collect.append((node, parent))

        # Distribute: forward order, send from parent to child
        for node, parent in order:
            if parent is not None:
                distribute.append((parent, node))

        return collect, distribute

    def get_parallel_schedule(
        self, root: Optional[int] = None
    ) -> List[List[Tuple[int, int]]]:
        """Return a parallelism-aware schedule as a list of *rounds*.

        Within each round, all messages can be computed in parallel.
        """
        if root is None:
            root = self.find_root()

        # BFS to compute depth
        depth: Dict[int, int] = {root: 0}
        parent_of: Dict[int, int] = {}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in self.neighbors(u):
                if v not in depth:
                    depth[v] = depth[u] + 1
                    parent_of[v] = u
                    queue.append(v)

        max_depth = max(depth.values()) if depth else 0

        # Collect phase: from deepest to shallowest
        collect_rounds: List[List[Tuple[int, int]]] = []
        for d in range(max_depth, 0, -1):
            round_msgs = [
                (node, parent_of[node])
                for node, dp in depth.items()
                if dp == d and node in parent_of
            ]
            if round_msgs:
                collect_rounds.append(round_msgs)

        # Distribute phase: from shallowest to deepest
        distribute_rounds: List[List[Tuple[int, int]]] = []
        for d in range(1, max_depth + 1):
            round_msgs = [
                (parent_of[node], node)
                for node, dp in depth.items()
                if dp == d and node in parent_of
            ]
            if round_msgs:
                distribute_rounds.append(round_msgs)

        return collect_rounds + distribute_rounds

    # ------------------------------------------------------------------ #
    #  Subtree extraction
    # ------------------------------------------------------------------ #

    def get_subtree(self, root_idx: int) -> List[int]:
        """Return all clique indices in the subtree rooted at *root_idx*
        (relative to the current tree root)."""
        tree_root = self._root if self._root is not None else 0
        parent_of: Dict[int, int] = {}
        queue = deque([tree_root])
        visited: Set[int] = {tree_root}
        while queue:
            u = queue.popleft()
            for v in self.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    parent_of[v] = u
                    queue.append(v)

        # Collect subtree under root_idx
        subtree = [root_idx]
        q2 = deque([root_idx])
        seen: Set[int] = {root_idx}
        while q2:
            u = q2.popleft()
            for v in self.neighbors(u):
                if v not in seen and parent_of.get(v) == u:
                    seen.add(v)
                    subtree.append(v)
                    q2.append(v)
        return subtree

    # ------------------------------------------------------------------ #
    #  Potential initialization
    # ------------------------------------------------------------------ #

    def assign_cpds(
        self, cpds: Dict[str, PotentialTable]
    ) -> None:
        """Assign each CPD to the smallest clique that contains its scope.

        Parameters
        ----------
        cpds : dict mapping variable name → CPD table.  The table's
            variables should be [child, parent1, parent2, …].
        """
        for var, cpd in cpds.items():
            cpd_scope = set(cpd.variables)
            best_clique: Optional[CliqueNode] = None
            best_size = float("inf")
            for clique in self._cliques:
                if cpd_scope <= clique.variables and clique.size < best_size:
                    best_clique = clique
                    best_size = clique.size
            if best_clique is None:
                raise ValueError(
                    f"No clique contains the scope of CPD for '{var}': "
                    f"{cpd_scope}"
                )
            best_clique.assign_cpd(cpd)

    def initialize_potentials(self) -> None:
        """Combine assigned CPDs into clique potentials."""
        for clique in self._cliques:
            clique.initialize_potential(self.cardinalities)

    def initialize_separators(self) -> None:
        """Create uniform separator potentials."""
        for idx_a, nbrs in self._adjacency.items():
            for idx_b, sep in nbrs.items():
                if idx_a < idx_b:
                    sep.initialize_potential(self.cardinalities)

    # ------------------------------------------------------------------ #
    #  Running-intersection property verification
    # ------------------------------------------------------------------ #

    def verify_running_intersection(self) -> bool:
        """Verify the running-intersection property (RIP).

        For every variable *v*, the set of cliques containing *v* must
        form a connected subtree.
        """
        for var, clique_indices in self._var_to_cliques.items():
            if len(clique_indices) <= 1:
                continue
            if not self._is_connected_subtree(clique_indices):
                return False
        return True

    def _is_connected_subtree(self, indices: Set[int]) -> bool:
        """Check that the given clique indices form a connected subgraph."""
        if not indices:
            return True
        start = next(iter(indices))
        visited: Set[int] = {start}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self.neighbors(u):
                if v in indices and v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited == indices

    # ------------------------------------------------------------------ #
    #  Graph-theory helpers
    # ------------------------------------------------------------------ #

    def treewidth(self) -> int:
        """Return the treewidth (max clique size − 1)."""
        if not self._cliques:
            return 0
        return max(c.size for c in self._cliques) - 1

    def total_table_size(self) -> int:
        """Sum of all clique-potential sizes (number of entries)."""
        total = 0
        for c in self._cliques:
            sz = 1
            for v in c.variables:
                sz *= self.cardinalities.get(v, 1)
            total += sz
        return total

    def clique_containing(self, variables: Set[str]) -> Optional[CliqueNode]:
        """Return the smallest clique containing all ``variables``."""
        best: Optional[CliqueNode] = None
        best_size = float("inf")
        for c in self._cliques:
            if variables <= c.variables and c.size < best_size:
                best = c
                best_size = c.size
        return best

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary dict."""
        return {
            "num_cliques": self.num_cliques,
            "treewidth": self.treewidth(),
            "total_table_size": self.total_table_size(),
            "rip_valid": self.verify_running_intersection(),
            "clique_sizes": [c.size for c in self._cliques],
        }

    def __repr__(self) -> str:
        return (
            f"CliqueTree(cliques={self.num_cliques}, "
            f"tw={self.treewidth()})"
        )


# ------------------------------------------------------------------ #
#  Module-level graph utilities
# ------------------------------------------------------------------ #

def _find_maximal_cliques(
    graph: Dict[str, Set[str]]
) -> List[FrozenSet[str]]:
    """Bron–Kerbosch algorithm with pivoting for maximal cliques."""
    result: List[FrozenSet[str]] = []
    nodes = set(graph.keys())

    def bron_kerbosch(R: Set[str], P: Set[str], X: Set[str]) -> None:
        if not P and not X:
            if len(R) >= 1:
                result.append(frozenset(R))
            return
        # Choose pivot with max neighbours in P
        pivot = max(P | X, key=lambda u: len(graph.get(u, set()) & P))
        for v in list(P - graph.get(pivot, set())):
            nbrs = graph.get(v, set())
            bron_kerbosch(R | {v}, P & nbrs, X & nbrs)
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(nodes), set())
    return result


def moralize(
    dag: Dict[str, List[str]]
) -> Dict[str, Set[str]]:
    """Moralize a DAG: connect co-parents and drop edge directions.

    Parameters
    ----------
    dag : adjacency list mapping each node to its list of **children**.

    Returns
    -------
    Undirected graph as adjacency sets.
    """
    nodes = set(dag.keys())
    for children in dag.values():
        nodes.update(children)

    graph: Dict[str, Set[str]] = {n: set() for n in nodes}

    # Build parent lists
    parents: Dict[str, List[str]] = defaultdict(list)
    for parent, children in dag.items():
        for child in children:
            parents[child].append(parent)

    # Add undirected edges for every directed edge
    for parent, children in dag.items():
        for child in children:
            graph[parent].add(child)
            graph[child].add(parent)

    # Marry co-parents
    for child, plist in parents.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                graph[plist[i]].add(plist[j])
                graph[plist[j]].add(plist[i])

    return graph


def triangulate(
    graph: Dict[str, Set[str]],
    elimination_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Triangulate (chordalize) an undirected graph using elimination.

    Parameters
    ----------
    graph : undirected adjacency sets.
    elimination_order : optional variable-elimination order.  If *None*
        a min-fill heuristic is used.

    Returns
    -------
    (triangulated_graph, elimination_order)
    """
    # Work on a copy
    g: Dict[str, Set[str]] = {n: set(nbrs) for n, nbrs in graph.items()}
    nodes = set(g.keys())

    if elimination_order is None:
        elimination_order = _min_fill_order(g)

    filled_graph: Dict[str, Set[str]] = {n: set(nbrs) for n, nbrs in graph.items()}

    remaining = set(nodes)
    for v in elimination_order:
        if v not in remaining:
            continue
        nbrs_in_remaining = g[v] & remaining
        # Add fill edges
        nbrs_list = sorted(nbrs_in_remaining)
        for i in range(len(nbrs_list)):
            for j in range(i + 1, len(nbrs_list)):
                a, b = nbrs_list[i], nbrs_list[j]
                g[a].add(b)
                g[b].add(a)
                filled_graph[a].add(b)
                filled_graph[b].add(a)
        remaining.discard(v)

    return filled_graph, elimination_order


def _min_fill_order(graph: Dict[str, Set[str]]) -> List[str]:
    """Compute an elimination order using the min-fill heuristic.

    Greedily picks the node whose elimination creates the fewest fill edges.
    """
    g: Dict[str, Set[str]] = {n: set(nbrs) for n, nbrs in graph.items()}
    remaining = set(g.keys())
    order: List[str] = []

    while remaining:
        best_node = None
        best_fill = float("inf")
        for v in remaining:
            nbrs = g[v] & remaining
            fill = 0
            nbrs_list = sorted(nbrs)
            for i in range(len(nbrs_list)):
                for j in range(i + 1, len(nbrs_list)):
                    if nbrs_list[j] not in g[nbrs_list[i]]:
                        fill += 1
            if fill < best_fill:
                best_fill = fill
                best_node = v

        if best_node is None:
            break

        # Add fill edges
        nbrs = g[best_node] & remaining
        nbrs_list = sorted(nbrs)
        for i in range(len(nbrs_list)):
            for j in range(i + 1, len(nbrs_list)):
                a, b = nbrs_list[i], nbrs_list[j]
                g[a].add(b)
                g[b].add(a)

        remaining.discard(best_node)
        order.append(best_node)

    return order


def build_junction_tree(
    dag: Dict[str, List[str]],
    cardinalities: Dict[str, int],
    elimination_order: Optional[List[str]] = None,
) -> CliqueTree:
    """One-shot: moralize → triangulate → build junction tree.

    Parameters
    ----------
    dag : directed adjacency list (node → list of children).
    cardinalities : variable → number of discrete states.
    elimination_order : optional variable-elimination order.

    Returns
    -------
    CliqueTree ready for CPD assignment and message passing.
    """
    moral = moralize(dag)
    tri_graph, elim = triangulate(moral, elimination_order)
    tree = CliqueTree.from_triangulated_graph(tri_graph, cardinalities)
    return tree
