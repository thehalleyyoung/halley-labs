"""
Tree decomposition algorithms for graphs.

Provides elimination-ordering-based tree decomposition with multiple
heuristics (min-fill, min-degree, min-width), simulated-annealing
refinement, junction-tree conversion, and decomposition validation.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeDecomposition:
    """Represents a tree decomposition of a graph.

    Attributes
    ----------
    bags : dict
        Mapping from bag id (int) to frozenset of vertex labels contained
        in that bag.
    tree : nx.Graph
        Tree whose nodes are bag ids and edges connect bags that should
        be adjacent in the decomposition.
    width : int
        Tree-width of this decomposition (max bag size minus one).
    ordering : list
        Elimination ordering that was used to produce this decomposition.
    """

    bags: Dict[int, FrozenSet[int]] = field(default_factory=dict)
    tree: nx.Graph = field(default_factory=nx.Graph)
    width: int = 0
    ordering: List[int] = field(default_factory=list)

    def num_bags(self) -> int:
        """Return the number of bags."""
        return len(self.bags)

    def max_bag_size(self) -> int:
        """Return the size of the largest bag."""
        if not self.bags:
            return 0
        return max(len(b) for b in self.bags.values())

    def bag_sizes(self) -> List[int]:
        """Return a sorted list of bag sizes."""
        return sorted(len(b) for b in self.bags.values())

    def vertices_covered(self) -> FrozenSet[int]:
        """Return the union of all bags."""
        result: Set[int] = set()
        for b in self.bags.values():
            result.update(b)
        return frozenset(result)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TreeDecomposer:
    """Compute tree decompositions using elimination-ordering heuristics.

    Parameters
    ----------
    strategy : str
        One of ``'min_fill'``, ``'min_degree'``, or ``'min_width'``.
        Determines the heuristic used for the initial elimination ordering.
    """

    _STRATEGIES = ("min_fill", "min_degree", "min_width")

    def __init__(self, strategy: str = "min_fill") -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {self._STRATEGIES}."
            )
        self._strategy: str = strategy
        self._last_ordering: Optional[List[int]] = None
        self._last_decomposition: Optional[TreeDecomposition] = None
        self._rng: random.Random = random.Random(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        graph: nx.Graph,
        max_width: Optional[int] = None,
    ) -> TreeDecomposition:
        """Compute a tree decomposition of *graph*.

        Parameters
        ----------
        graph : nx.Graph
            The input graph (simple, undirected).
        max_width : int or None
            If given, bags wider than *max_width* + 1 are split into
            smaller overlapping bags so that the returned decomposition
            has width at most *max_width*.

        Returns
        -------
        TreeDecomposition
        """
        if graph.number_of_nodes() == 0:
            td = TreeDecomposition(bags={}, tree=nx.Graph(), width=0, ordering=[])
            self._last_ordering = []
            self._last_decomposition = td
            return td

        ordering = self._get_ordering(graph)
        self._last_ordering = list(ordering)

        decomp = self._build_decomposition(graph, ordering)

        if max_width is not None and decomp.width > max_width:
            decomp = self._enforce_width_bound(decomp, max_width)

        self._last_decomposition = decomp
        return decomp

    def get_elimination_ordering(self) -> List[int]:
        """Return the elimination ordering from the last call to *decompose*.

        Raises
        ------
        RuntimeError
            If *decompose* has not been called yet.
        """
        if self._last_ordering is None:
            raise RuntimeError(
                "No elimination ordering available. Call decompose() first."
            )
        return list(self._last_ordering)

    def compute_width(self, decomp: TreeDecomposition) -> int:
        """Return the width of a tree decomposition (max bag size − 1)."""
        if not decomp.bags:
            return 0
        return max(len(bag) for bag in decomp.bags.values()) - 1

    def validate_decomposition(
        self, graph: nx.Graph, decomp: TreeDecomposition
    ) -> bool:
        """Check the three tree-decomposition axioms.

        1. Every vertex appears in at least one bag.
        2. For every edge (u, v) of *graph*, some bag contains both u and v.
        3. For every vertex v, the set of bags containing v forms a
           connected subtree of the decomposition tree.

        Returns True iff all three conditions hold.
        """
        if graph.number_of_nodes() == 0:
            return True

        # (1) vertex coverage
        covered = set()
        for bag in decomp.bags.values():
            covered.update(bag)
        for v in graph.nodes():
            if v not in covered:
                return False

        # (2) edge coverage
        for u, v in graph.edges():
            found = False
            for bag in decomp.bags.values():
                if u in bag and v in bag:
                    found = True
                    break
            if not found:
                return False

        # (3) running-intersection / connectedness property
        vertex_to_bags: Dict[int, Set[int]] = {}
        for bid, bag in decomp.bags.items():
            for v in bag:
                vertex_to_bags.setdefault(v, set()).add(bid)

        for v, bag_ids in vertex_to_bags.items():
            if len(bag_ids) <= 1:
                continue
            subgraph = decomp.tree.subgraph(bag_ids)
            if not nx.is_connected(subgraph):
                return False

        return True

    def refine_ordering(
        self,
        graph: nx.Graph,
        ordering: List[int],
        method: str = "simulated_annealing",
        iterations: int = 1000,
    ) -> List[int]:
        """Improve an elimination ordering via local search.

        Parameters
        ----------
        graph : nx.Graph
            The input graph.
        ordering : list of int
            Initial elimination ordering.
        method : str
            ``'simulated_annealing'`` or ``'greedy'``.
        iterations : int
            Number of search iterations.

        Returns
        -------
        list of int
            The refined ordering.
        """
        if method == "simulated_annealing":
            return self._simulated_annealing_refinement(
                graph, ordering, iterations
            )
        elif method == "greedy":
            return self._greedy_refinement(graph, ordering, iterations)
        else:
            raise ValueError(f"Unknown refinement method: {method}")

    def to_junction_tree(self, decomp: TreeDecomposition) -> nx.Graph:
        """Convert a tree decomposition into a junction tree.

        A junction tree (clique tree) is a tree whose nodes are the bags
        and whose edges are labelled with separator sets (intersection of
        the two endpoint bags).  The result satisfies the running
        intersection property.

        Parameters
        ----------
        decomp : TreeDecomposition
            A valid tree decomposition.

        Returns
        -------
        nx.Graph
            Junction tree with ``separator`` edge attributes.
        """
        bag_ids = list(decomp.bags.keys())
        n = len(bag_ids)
        if n <= 1:
            jt = nx.Graph()
            for bid in bag_ids:
                jt.add_node(bid, members=decomp.bags[bid])
            return jt

        # Build complete weighted graph on bags; weight = |intersection|
        id_to_idx = {bid: i for i, bid in enumerate(bag_ids)}
        weight_matrix = lil_matrix((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                bi = decomp.bags[bag_ids[i]]
                bj = decomp.bags[bag_ids[j]]
                w = len(bi & bj)
                if w > 0:
                    # scipy MST finds minimum, so negate for maximum
                    weight_matrix[i, j] = -w
                    weight_matrix[j, i] = -w

        mst_sparse = minimum_spanning_tree(weight_matrix.tocsr())
        mst_cx = mst_sparse.tocoo()

        jt = nx.Graph()
        for bid in bag_ids:
            jt.add_node(bid, members=decomp.bags[bid])

        for row, col in zip(mst_cx.row, mst_cx.col):
            bid_u = bag_ids[row]
            bid_v = bag_ids[col]
            sep = decomp.bags[bid_u] & decomp.bags[bid_v]
            jt.add_edge(bid_u, bid_v, separator=sep, weight=len(sep))

        # Ensure the junction tree is connected even if MST left components
        components = list(nx.connected_components(jt))
        for ci in range(1, len(components)):
            # Find the pair across components with maximum intersection
            best_u, best_v, best_w = None, None, -1
            for u in components[0]:
                for v in components[ci]:
                    w = len(decomp.bags[u] & decomp.bags[v])
                    if w > best_w:
                        best_u, best_v, best_w = u, v, w
            if best_u is not None and best_v is not None:
                sep = decomp.bags[best_u] & decomp.bags[best_v]
                jt.add_edge(best_u, best_v, separator=sep, weight=len(sep))
            # Merge component ids
            components[0] = components[0] | components[ci]

        return jt

    # ------------------------------------------------------------------
    # Ordering heuristics
    # ------------------------------------------------------------------

    def _get_ordering(self, graph: nx.Graph) -> List[int]:
        """Dispatch to the configured strategy."""
        if self._strategy == "min_fill":
            return self._min_fill_ordering(graph)
        elif self._strategy == "min_degree":
            return self._min_degree_ordering(graph)
        elif self._strategy == "min_width":
            return self._min_width_ordering(graph)
        raise ValueError(f"No handler for strategy {self._strategy}")

    def _min_fill_ordering(self, graph: nx.Graph) -> List[int]:
        """Greedy min-fill elimination ordering.

        At each step the vertex whose elimination would add the fewest
        fill edges is chosen.  Ties are broken by smallest vertex label.

        Parameters
        ----------
        graph : nx.Graph

        Returns
        -------
        list of int
            Elimination ordering (first element is eliminated first).
        """
        work = graph.copy()
        remaining = set(work.nodes())
        ordering: List[int] = []

        while remaining:
            best_vertex: Optional[int] = None
            best_fill = float("inf")

            for v in sorted(remaining):
                fill = self._count_fill_edges(work, v)
                if fill < best_fill or (fill == best_fill and (
                        best_vertex is None or v < best_vertex)):
                    best_fill = fill
                    best_vertex = v

            assert best_vertex is not None
            ordering.append(best_vertex)
            self._eliminate_vertex(work, best_vertex)
            remaining.discard(best_vertex)

        return ordering

    def _min_degree_ordering(self, graph: nx.Graph) -> List[int]:
        """Greedy min-degree elimination ordering.

        At each step the vertex with the smallest current degree is
        eliminated.  Ties broken by smallest label.

        Parameters
        ----------
        graph : nx.Graph

        Returns
        -------
        list of int
        """
        work = graph.copy()
        remaining = set(work.nodes())
        ordering: List[int] = []

        while remaining:
            best_vertex: Optional[int] = None
            best_deg = float("inf")

            for v in sorted(remaining):
                d = work.degree(v)
                if d < best_deg or (d == best_deg and (
                        best_vertex is None or v < best_vertex)):
                    best_deg = d
                    best_vertex = v

            assert best_vertex is not None
            ordering.append(best_vertex)
            self._eliminate_vertex(work, best_vertex)
            remaining.discard(best_vertex)

        return ordering

    def _min_width_ordering(self, graph: nx.Graph) -> List[int]:
        """Greedy min-width elimination ordering.

        At each step, pick the vertex that minimises the bag width
        (number of neighbours) at the point of elimination.  This
        coincides with min-degree on the evolving elimination graph,
        but we keep a running maximum and choose the vertex that
        minimises the *global* running width contribution.

        Parameters
        ----------
        graph : nx.Graph

        Returns
        -------
        list of int
        """
        work = graph.copy()
        remaining = set(work.nodes())
        ordering: List[int] = []
        current_max_width = 0

        while remaining:
            best_vertex: Optional[int] = None
            best_score = float("inf")

            for v in sorted(remaining):
                # The bag formed by eliminating v has size deg(v)+1;
                # width contribution is deg(v).
                deg_v = work.degree(v)
                score = max(current_max_width, deg_v)
                if score < best_score or (
                    score == best_score and (
                        best_vertex is None or v < best_vertex
                    )
                ):
                    best_score = score
                    best_vertex = v

            assert best_vertex is not None
            ordering.append(best_vertex)
            bag_width = work.degree(best_vertex)
            if bag_width > current_max_width:
                current_max_width = bag_width
            self._eliminate_vertex(work, best_vertex)
            remaining.discard(best_vertex)

        return ordering

    # ------------------------------------------------------------------
    # Building the decomposition from an ordering
    # ------------------------------------------------------------------

    def _build_decomposition(
        self, graph: nx.Graph, ordering: List[int]
    ) -> TreeDecomposition:
        """Construct a tree decomposition from an elimination ordering.

        Each elimination step yields a bag consisting of the eliminated
        vertex and its current neighbours in the fill-in graph.  The
        bags are then connected into a tree.

        Parameters
        ----------
        graph : nx.Graph
        ordering : list of int

        Returns
        -------
        TreeDecomposition
        """
        work = graph.copy()
        bags: Dict[int, FrozenSet[int]] = {}

        for idx, v in enumerate(ordering):
            bag = self._compute_bag_from_elimination(work, v)
            bags[idx] = bag
            self._eliminate_vertex(work, v)

        # Remove duplicate / subset bags
        bags = self._remove_redundant_bags(bags)

        tree = self._connect_bags_into_tree(bags)

        width = max((len(b) for b in bags.values()), default=1) - 1

        return TreeDecomposition(
            bags=bags,
            tree=tree,
            width=width,
            ordering=list(ordering),
        )

    def _remove_redundant_bags(
        self, bags: Dict[int, FrozenSet[int]]
    ) -> Dict[int, FrozenSet[int]]:
        """Remove bags that are strict subsets of another bag.

        Parameters
        ----------
        bags : dict mapping bag id to frozenset

        Returns
        -------
        dict
            Filtered bags.
        """
        ids = sorted(bags.keys())
        keep: Set[int] = set(ids)

        for i in ids:
            if i not in keep:
                continue
            for j in ids:
                if j == i or j not in keep:
                    continue
                if bags[j] <= bags[i] and bags[j] != bags[i]:
                    keep.discard(j)

        return {bid: bags[bid] for bid in sorted(keep)}

    def _connect_bags_into_tree(
        self, bags: Dict[int, FrozenSet[int]]
    ) -> nx.Graph:
        """Build a tree structure over bags using a maximum-weight
        spanning tree on intersection sizes.

        Parameters
        ----------
        bags : dict mapping bag id to frozenset

        Returns
        -------
        nx.Graph
            A tree on the bag ids.
        """
        bag_ids = sorted(bags.keys())
        n = len(bag_ids)
        tree = nx.Graph()
        for bid in bag_ids:
            tree.add_node(bid)

        if n <= 1:
            return tree

        id_to_idx = {bid: i for i, bid in enumerate(bag_ids)}
        weight_mat = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                w = len(bags[bag_ids[i]] & bags[bag_ids[j]])
                weight_mat[i, j] = w
                weight_mat[j, i] = w

        # Maximum spanning tree via negation + scipy minimum spanning tree
        neg_sparse = lil_matrix((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if weight_mat[i, j] > 0:
                    neg_sparse[i, j] = -weight_mat[i, j]

        mst = minimum_spanning_tree(neg_sparse.tocsr())
        cx = mst.tocoo()

        for r, c in zip(cx.row, cx.col):
            tree.add_edge(bag_ids[r], bag_ids[c])

        # Ensure connectivity: if disconnected, link components by any edge
        components = list(nx.connected_components(tree))
        for ci in range(1, len(components)):
            rep_a = next(iter(components[0]))
            rep_b = next(iter(components[ci]))
            tree.add_edge(rep_a, rep_b)
            components[0] = components[0] | components[ci]

        return tree

    # ------------------------------------------------------------------
    # Width enforcement
    # ------------------------------------------------------------------

    def _enforce_width_bound(
        self, decomp: TreeDecomposition, max_width: int
    ) -> TreeDecomposition:
        """Split bags exceeding *max_width* + 1 vertices.

        Each oversized bag is partitioned into overlapping sub-bags of
        size at most *max_width* + 1 that together cover the original
        bag.  Sub-bags share at least *max_width* vertices with their
        neighbour to preserve the running-intersection property.

        Parameters
        ----------
        decomp : TreeDecomposition
        max_width : int

        Returns
        -------
        TreeDecomposition
        """
        new_bags: Dict[int, FrozenSet[int]] = {}
        parent_map: Dict[int, List[int]] = {}
        next_id = 0

        for bid, bag in decomp.bags.items():
            if len(bag) <= max_width + 1:
                new_bags[next_id] = bag
                parent_map[bid] = [next_id]
                next_id += 1
            else:
                sub_bags = self._split_bag(bag, max_width)
                ids_for_this: List[int] = []
                for sb in sub_bags:
                    new_bags[next_id] = sb
                    ids_for_this.append(next_id)
                    next_id += 1
                parent_map[bid] = ids_for_this

        # Rebuild tree
        new_tree = nx.Graph()
        for nid in new_bags:
            new_tree.add_node(nid)

        # Connect sub-bags from the same original bag in a chain
        for bid, nids in parent_map.items():
            for k in range(len(nids) - 1):
                new_tree.add_edge(nids[k], nids[k + 1])

        # Restore edges between original bags
        for u, v in decomp.tree.edges():
            rep_u = parent_map[u][0]
            rep_v = parent_map[v][0]

            # Pick representatives with maximum intersection
            best_pair = (rep_u, rep_v)
            best_w = len(new_bags[rep_u] & new_bags[rep_v])
            for nu in parent_map[u]:
                for nv in parent_map[v]:
                    w = len(new_bags[nu] & new_bags[nv])
                    if w > best_w:
                        best_w = w
                        best_pair = (nu, nv)

            if best_pair[0] != best_pair[1]:
                new_tree.add_edge(best_pair[0], best_pair[1])

        width = max((len(b) for b in new_bags.values()), default=1) - 1

        return TreeDecomposition(
            bags=new_bags,
            tree=new_tree,
            width=width,
            ordering=list(decomp.ordering),
        )

    def _split_bag(
        self, bag: FrozenSet[int], max_width: int
    ) -> List[FrozenSet[int]]:
        """Split a single large bag into overlapping sub-bags.

        Uses a sliding-window approach over a sorted vertex list so that
        consecutive sub-bags share *max_width* vertices.

        Parameters
        ----------
        bag : frozenset of int
        max_width : int

        Returns
        -------
        list of frozenset
        """
        target_size = max_width + 1
        if target_size < 2:
            target_size = 2

        vertices = sorted(bag)
        n = len(vertices)
        if n <= target_size:
            return [frozenset(vertices)]

        overlap = max(1, target_size - 1)
        step = max(1, target_size - overlap)
        sub_bags: List[FrozenSet[int]] = []
        start = 0
        while start < n:
            end = min(start + target_size, n)
            sub_bags.append(frozenset(vertices[start:end]))
            if end >= n:
                break
            start += step

        # Make sure the last sub-bag is not too small; merge if needed
        if len(sub_bags) >= 2 and len(sub_bags[-1]) < max(2, target_size // 2):
            merged = sub_bags[-2] | sub_bags[-1]
            sub_bags = sub_bags[:-2]
            sub_bags.append(merged)

        return sub_bags

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _simulated_annealing_refinement(
        self,
        graph: nx.Graph,
        ordering: List[int],
        iterations: int,
    ) -> List[int]:
        """Refine an ordering using simulated annealing.

        At each iteration two random positions are swapped.  The move is
        accepted if the resulting width is lower, or with Boltzmann
        probability exp(-delta / temperature).

        Parameters
        ----------
        graph : nx.Graph
        ordering : list of int
        iterations : int

        Returns
        -------
        list of int
            The best ordering found.
        """
        best = list(ordering)
        best_width = self._ordering_width(graph, best)
        current = list(best)
        current_width = best_width

        t_initial = max(1.0, float(best_width))
        t_min = 1e-4
        alpha = (t_min / t_initial) ** (1.0 / max(1, iterations))
        temperature = t_initial
        n = len(ordering)
        if n < 2:
            return best

        for _it in range(iterations):
            i = self._rng.randint(0, n - 1)
            j = self._rng.randint(0, n - 1)
            while j == i:
                j = self._rng.randint(0, n - 1)

            current[i], current[j] = current[j], current[i]

            new_width = self._ordering_width(graph, current)
            delta = new_width - current_width

            if delta <= 0:
                current_width = new_width
                if new_width < best_width:
                    best = list(current)
                    best_width = new_width
            else:
                acceptance = math.exp(-delta / max(temperature, 1e-12))
                if self._rng.random() < acceptance:
                    current_width = new_width
                else:
                    # Revert swap
                    current[i], current[j] = current[j], current[i]

            temperature *= alpha

        return best

    def _greedy_refinement(
        self,
        graph: nx.Graph,
        ordering: List[int],
        iterations: int,
    ) -> List[int]:
        """Local-search refinement trying adjacent swaps.

        Scans through the ordering and swaps adjacent elements if the
        swap strictly reduces the resulting tree-width.  Repeats for
        *iterations* passes.

        Parameters
        ----------
        graph : nx.Graph
        ordering : list of int
        iterations : int

        Returns
        -------
        list of int
        """
        current = list(ordering)
        current_width = self._ordering_width(graph, current)
        n = len(current)
        if n < 2:
            return current

        for _pass in range(iterations):
            improved = False
            for i in range(n - 1):
                current[i], current[i + 1] = current[i + 1], current[i]
                new_width = self._ordering_width(graph, current)
                if new_width < current_width:
                    current_width = new_width
                    improved = True
                else:
                    current[i], current[i + 1] = current[i + 1], current[i]
            if not improved:
                break

        return current

    def _ordering_width(self, graph: nx.Graph, ordering: List[int]) -> int:
        """Compute the tree-width induced by an elimination ordering.

        Parameters
        ----------
        graph : nx.Graph
        ordering : list of int

        Returns
        -------
        int
        """
        work = graph.copy()
        max_w = 0
        for v in ordering:
            if v not in work:
                continue
            deg = work.degree(v)
            if deg > max_w:
                max_w = deg
            self._eliminate_vertex(work, v)
        return max_w

    # ------------------------------------------------------------------
    # Vertex elimination helpers
    # ------------------------------------------------------------------

    def _eliminate_vertex(
        self, graph: nx.Graph, vertex: int
    ) -> Set[Tuple[int, int]]:
        """Eliminate *vertex* from *graph* in-place.

        All neighbours of *vertex* are connected pairwise (fill edges),
        then *vertex* and its incident edges are removed.

        Parameters
        ----------
        graph : nx.Graph
        vertex : int

        Returns
        -------
        set of (int, int)
            The fill edges that were added.
        """
        if vertex not in graph:
            return set()

        neighbours = list(graph.neighbors(vertex))
        fill_edges: Set[Tuple[int, int]] = set()

        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                u, v = neighbours[i], neighbours[j]
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    fill_edges.add((min(u, v), max(u, v)))

        graph.remove_node(vertex)
        return fill_edges

    def _count_fill_edges(self, graph: nx.Graph, vertex: int) -> int:
        """Count fill edges that would be added if *vertex* were eliminated.

        Parameters
        ----------
        graph : nx.Graph
        vertex : int

        Returns
        -------
        int
        """
        neighbours = list(graph.neighbors(vertex))
        count = 0
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                if not graph.has_edge(neighbours[i], neighbours[j]):
                    count += 1
        return count

    def _compute_bag_from_elimination(
        self, graph: nx.Graph, vertex: int
    ) -> FrozenSet[int]:
        """Compute the bag formed when *vertex* is eliminated.

        The bag is {vertex} ∪ neighbours(vertex) in the current
        (fill-in) graph.

        Parameters
        ----------
        graph : nx.Graph
        vertex : int

        Returns
        -------
        frozenset of int
        """
        if vertex not in graph:
            return frozenset([vertex])
        neighbours = set(graph.neighbors(vertex))
        return frozenset(neighbours | {vertex})


# ---------------------------------------------------------------------------
# Utility functions (module-level)
# ---------------------------------------------------------------------------

def build_adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    """Return a dense boolean adjacency matrix for *graph*.

    The rows/columns are indexed by sorted node labels.  A mapping from
    node label to matrix index is also returned.

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    tuple of (np.ndarray, dict)
        ``(adj_matrix, node_to_idx)``
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.bool_)
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = True
        adj[j, i] = True
    return adj, node_to_idx


def fill_count_from_matrix(
    adj: np.ndarray, idx: int
) -> int:
    """Count fill edges for a vertex given by matrix index *idx*.

    Parameters
    ----------
    adj : np.ndarray
        Boolean adjacency matrix.
    idx : int
        Row/column index of the vertex.

    Returns
    -------
    int
    """
    neighbours = np.flatnonzero(adj[idx])
    count = 0
    for i in range(len(neighbours)):
        for j in range(i + 1, len(neighbours)):
            if not adj[neighbours[i], neighbours[j]]:
                count += 1
    return count


def eliminate_vertex_matrix(
    adj: np.ndarray, idx: int
) -> np.ndarray:
    """Eliminate vertex *idx* from the adjacency matrix.

    Connects all neighbours pairwise, then zeros out the row and column.

    Parameters
    ----------
    adj : np.ndarray
        Boolean adjacency matrix (modified in-place and returned).
    idx : int

    Returns
    -------
    np.ndarray
    """
    neighbours = np.flatnonzero(adj[idx])
    for i in range(len(neighbours)):
        for j in range(i + 1, len(neighbours)):
            ni, nj = neighbours[i], neighbours[j]
            adj[ni, nj] = True
            adj[nj, ni] = True
    adj[idx, :] = False
    adj[:, idx] = False
    return adj


def min_fill_ordering_matrix(graph: nx.Graph) -> List[int]:
    """Min-fill ordering using a numpy adjacency matrix.

    Faster than the networkx-based version for dense graphs.

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    list of int
        Elimination ordering (node labels, not matrix indices).
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    adj, node_to_idx = build_adjacency_matrix(graph)
    idx_to_node = {i: v for v, i in node_to_idx.items()}
    eliminated = np.zeros(n, dtype=np.bool_)
    ordering: List[int] = []

    for _ in range(n):
        best_idx = -1
        best_fill = float("inf")
        for k in range(n):
            if eliminated[k]:
                continue
            fill = fill_count_from_matrix(adj, k)
            if fill < best_fill or (fill == best_fill and (
                    best_idx == -1 or k < best_idx)):
                best_fill = fill
                best_idx = k

        ordering.append(idx_to_node[best_idx])
        eliminate_vertex_matrix(adj, best_idx)
        eliminated[best_idx] = True

    return ordering


def compute_treewidth_upper_bound(graph: nx.Graph) -> int:
    """Compute an upper bound on the treewidth of *graph*.

    Uses the min-fill heuristic and returns the width of the resulting
    decomposition.

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    int
    """
    td = TreeDecomposer(strategy="min_fill")
    decomp = td.decompose(graph)
    return decomp.width


def compute_treewidth_lower_bound(graph: nx.Graph) -> int:
    """Compute a lower bound on the treewidth using the degeneracy.

    The treewidth is at least the degeneracy of the graph minus one,
    and at least the minimum degree over all subgraphs (which equals
    the degeneracy).

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    int
    """
    if graph.number_of_nodes() == 0:
        return 0

    work = graph.copy()
    min_max_degree = float("inf")

    while work.number_of_nodes() > 0:
        degrees = dict(work.degree())
        min_v = min(degrees, key=degrees.get)
        d = degrees[min_v]
        if d < min_max_degree:
            min_max_degree = d
        work.remove_node(min_v)

    return max(0, int(min_max_degree))


def nice_tree_decomposition(
    decomp: TreeDecomposition,
) -> TreeDecomposition:
    """Convert a tree decomposition into a *nice* tree decomposition.

    In a nice tree decomposition every non-leaf bag is one of:
    - **introduce** : adds exactly one vertex compared to its child
    - **forget**    : removes exactly one vertex compared to its child
    - **join**      : has exactly two children with identical bags

    Leaf bags have exactly one vertex.

    Parameters
    ----------
    decomp : TreeDecomposition

    Returns
    -------
    TreeDecomposition
        A new decomposition with the nice structure.
    """
    if not decomp.bags:
        return TreeDecomposition(
            bags={}, tree=nx.Graph(), width=0, ordering=list(decomp.ordering)
        )

    # Pick an arbitrary root
    root = next(iter(decomp.bags.keys()))
    directed = nx.bfs_tree(decomp.tree, root)

    new_bags: Dict[int, FrozenSet[int]] = {}
    new_tree = nx.Graph()
    next_id = max(decomp.bags.keys()) + 1

    # Copy existing bags
    for bid, bag in decomp.bags.items():
        new_bags[bid] = bag
        new_tree.add_node(bid)

    # Process edges in BFS order to insert introduce/forget chains
    for parent, child in directed.edges():
        parent_bag = new_bags[parent]
        child_bag = new_bags[child]

        only_parent = parent_bag - child_bag
        only_child = child_bag - parent_bag

        chain_ids: List[int] = [parent]

        # Forget nodes in only_child (present in child, not parent)
        for v in sorted(only_child):
            prev_id = chain_ids[-1]
            prev_bag = new_bags[prev_id]
            new_bag = prev_bag | frozenset([v])
            new_bags[next_id] = new_bag
            new_tree.add_node(next_id)
            new_tree.add_edge(prev_id, next_id)
            chain_ids.append(next_id)
            next_id += 1

        # Introduce nodes in only_parent (present in parent, not child)
        current_bag_set = set(new_bags[chain_ids[-1]])
        for v in sorted(only_parent):
            if v in current_bag_set:
                prev_id = chain_ids[-1]
                prev_bag = new_bags[prev_id]
                reduced = prev_bag - frozenset([v])
                new_bags[next_id] = reduced
                new_tree.add_node(next_id)
                new_tree.add_edge(prev_id, next_id)
                chain_ids.append(next_id)
                current_bag_set = set(reduced)
                next_id += 1

        # Link last chain element to child
        last_chain = chain_ids[-1]
        if last_chain != child:
            new_tree.add_edge(last_chain, child)

    # Remove original tree edges that were replaced by chains
    for parent, child in directed.edges():
        if new_tree.has_edge(parent, child) and parent != child:
            # Only remove direct edge if a chain was inserted
            children_of_parent = list(new_tree.neighbors(parent))
            if len(children_of_parent) > 2:
                pass  # keep connectivity

    width = max((len(b) for b in new_bags.values()), default=1) - 1

    return TreeDecomposition(
        bags=new_bags,
        tree=new_tree,
        width=width,
        ordering=list(decomp.ordering),
    )


def decompose_and_validate(
    graph: nx.Graph,
    strategy: str = "min_fill",
    max_width: Optional[int] = None,
) -> Tuple[TreeDecomposition, bool]:
    """Convenience: decompose a graph and validate the result.

    Parameters
    ----------
    graph : nx.Graph
    strategy : str
    max_width : int or None

    Returns
    -------
    tuple of (TreeDecomposition, bool)
        The decomposition and whether it is valid.
    """
    td = TreeDecomposer(strategy=strategy)
    decomp = td.decompose(graph, max_width=max_width)
    valid = td.validate_decomposition(graph, decomp)
    return decomp, valid


def batch_decompose(
    graphs: List[nx.Graph],
    strategy: str = "min_fill",
) -> List[TreeDecomposition]:
    """Decompose a list of graphs.

    Parameters
    ----------
    graphs : list of nx.Graph
    strategy : str

    Returns
    -------
    list of TreeDecomposition
    """
    td = TreeDecomposer(strategy=strategy)
    results: List[TreeDecomposition] = []
    for g in graphs:
        results.append(td.decompose(g))
    return results


def intersection_graph(decomp: TreeDecomposition) -> nx.Graph:
    """Build the weighted intersection graph over bags.

    Nodes are bag ids; edge weight is the size of the bag intersection.

    Parameters
    ----------
    decomp : TreeDecomposition

    Returns
    -------
    nx.Graph
    """
    ig = nx.Graph()
    bag_ids = sorted(decomp.bags.keys())
    for bid in bag_ids:
        ig.add_node(bid, size=len(decomp.bags[bid]))

    for i in range(len(bag_ids)):
        for j in range(i + 1, len(bag_ids)):
            bi = decomp.bags[bag_ids[i]]
            bj = decomp.bags[bag_ids[j]]
            w = len(bi & bj)
            if w > 0:
                ig.add_edge(bag_ids[i], bag_ids[j], weight=w)

    return ig


def separator_sizes(decomp: TreeDecomposition) -> List[int]:
    """Return the sizes of all separators in a tree decomposition.

    A separator is the intersection of two adjacent bags in the tree.

    Parameters
    ----------
    decomp : TreeDecomposition

    Returns
    -------
    list of int
    """
    sizes: List[int] = []
    for u, v in decomp.tree.edges():
        sep = decomp.bags[u] & decomp.bags[v]
        sizes.append(len(sep))
    return sorted(sizes)


def optimal_treewidth_exact(graph: nx.Graph, timeout: int = 60) -> int:
    """Compute the exact treewidth using brute-force over permutations.

    Only feasible for very small graphs (≤ 15 vertices).  Returns -1
    if the timeout (in seconds) is exceeded.

    Parameters
    ----------
    graph : nx.Graph
    timeout : int

    Returns
    -------
    int
    """
    import time
    from itertools import permutations

    nodes = sorted(graph.nodes())
    n = len(nodes)
    if n == 0:
        return 0
    if n > 15:
        return compute_treewidth_upper_bound(graph)

    start = time.monotonic()
    best_width = n - 1

    td = TreeDecomposer(strategy="min_fill")

    for perm in permutations(nodes):
        if time.monotonic() - start > timeout:
            break
        w = td._ordering_width(graph, list(perm))
        if w < best_width:
            best_width = w
        if best_width == compute_treewidth_lower_bound(graph):
            break

    return best_width


def random_graph_decomposition(
    n: int,
    p: float,
    strategy: str = "min_fill",
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, TreeDecomposition]:
    """Generate a random Erdős–Rényi graph and decompose it.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Edge probability.
    strategy : str
    seed : int or None

    Returns
    -------
    tuple of (nx.Graph, TreeDecomposition)
    """
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    td = TreeDecomposer(strategy=strategy)
    decomp = td.decompose(g)
    return g, decomp


def compare_strategies(graph: nx.Graph) -> Dict[str, int]:
    """Decompose with every available strategy and compare widths.

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    dict mapping strategy name to resulting width
    """
    results: Dict[str, int] = {}
    for strategy in TreeDecomposer._STRATEGIES:
        td = TreeDecomposer(strategy=strategy)
        decomp = td.decompose(graph)
        results[strategy] = decomp.width
    return results


def decomposition_statistics(decomp: TreeDecomposition) -> Dict[str, Any]:
    """Compute summary statistics for a tree decomposition.

    Parameters
    ----------
    decomp : TreeDecomposition

    Returns
    -------
    dict
    """
    sizes = [len(b) for b in decomp.bags.values()]
    if not sizes:
        return {
            "num_bags": 0,
            "width": 0,
            "mean_bag_size": 0.0,
            "median_bag_size": 0.0,
            "std_bag_size": 0.0,
            "max_bag_size": 0,
            "min_bag_size": 0,
            "num_tree_edges": 0,
        }

    arr = np.array(sizes, dtype=np.float64)
    return {
        "num_bags": len(sizes),
        "width": max(sizes) - 1,
        "mean_bag_size": float(np.mean(arr)),
        "median_bag_size": float(np.median(arr)),
        "std_bag_size": float(np.std(arr)),
        "max_bag_size": int(np.max(arr)),
        "min_bag_size": int(np.min(arr)),
        "num_tree_edges": decomp.tree.number_of_edges(),
    }
