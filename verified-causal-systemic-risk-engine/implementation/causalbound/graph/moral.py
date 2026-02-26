"""
Moral graph construction, triangulation, and clique tree algorithms
for causal graphical models.
"""

from typing import Optional, Set, List, Tuple, Dict, FrozenSet, Deque
from collections import deque, defaultdict
import itertools

import networkx as nx
import numpy as np


class MoralGraphConstructor:
    """Constructs moral graphs, triangulations, and clique trees from DAGs."""

    def __init__(self) -> None:
        self._cache: Dict[int, nx.Graph] = {}
        self._triangulation_history: List[Tuple[int, int]] = []
        self._elimination_orderings: Dict[str, List[int]] = {}
        self._clique_cache: Dict[int, List[FrozenSet[int]]] = {}

    # ------------------------------------------------------------------
    # Moralization
    # ------------------------------------------------------------------

    def moralize(self, dag: nx.DiGraph) -> nx.Graph:
        """Construct the moral graph from a DAG.

        For every node, marry (connect) all pairs of its parents, then
        drop edge directions.  Node attributes are preserved.
        """
        moral = nx.Graph()
        for node, attrs in dag.nodes(data=True):
            moral.add_node(node, **attrs)

        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    if not moral.has_edge(parents[i], parents[j]):
                        moral.add_edge(parents[i], parents[j], moral_edge=True)

        for u, v, attrs in dag.edges(data=True):
            if moral.has_edge(u, v):
                existing = moral.edges[u, v]
                existing.update(attrs)
                existing["moral_edge"] = existing.get("moral_edge", False)
            else:
                moral.add_edge(u, v, moral_edge=False, **attrs)

        return moral

    def augmented_moral_graph(
        self, dag: nx.DiGraph, observed: Optional[Set[int]] = None
    ) -> nx.Graph:
        """Augmented moral graph that correctly handles explaining-away.

        When a collider (or its descendant) is observed, the parents of
        that collider become dependent — so we must marry them.  The
        standard moral graph only marries parents unconditionally; the
        augmented version restricts marriage to nodes whose parents are
        on *active* paths given the observed set.

        Algorithm
        ---------
        1. Find all nodes that are observed or have an observed descendant.
        2. Build the ancestral graph of the query + observed nodes.
        3. For every node in this ancestral subgraph whose corresponding
           original node (or any descendant) is observed, marry its parents.
        4. Drop directions.
        """
        if observed is None:
            return self.moralize(dag)

        observed = set(observed)

        descendants_observed: Set[int] = set()
        for obs_node in observed:
            descendants_observed.add(obs_node)
            ancestors = self._find_ancestors(dag, {obs_node})
            descendants_observed.update(ancestors)

        has_observed_descendant: Set[int] = set()
        for node in dag.nodes():
            desc = nx.descendants(dag, node)
            desc.add(node)
            if desc & observed:
                has_observed_descendant.add(node)

        moral = nx.Graph()
        for node, attrs in dag.nodes(data=True):
            moral.add_node(node, **attrs)

        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            should_marry = node in has_observed_descendant
            if should_marry and len(parents) >= 2:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        if not moral.has_edge(parents[i], parents[j]):
                            moral.add_edge(
                                parents[i],
                                parents[j],
                                moral_edge=True,
                                augmented=True,
                            )

        for u, v, attrs in dag.edges(data=True):
            if not moral.has_edge(u, v):
                moral.add_edge(u, v, moral_edge=False, **attrs)

        return moral

    # ------------------------------------------------------------------
    # Ancestral graph
    # ------------------------------------------------------------------

    def ancestral_graph(
        self, dag: nx.DiGraph, target_nodes: Set[int]
    ) -> nx.DiGraph:
        """Return the sub-DAG induced by *target_nodes* and all their ancestors."""
        ancestors = self._find_ancestors(dag, target_nodes)
        keep = ancestors | target_nodes
        subdag = dag.subgraph(keep).copy()
        return subdag

    def _find_ancestors(self, dag: nx.DiGraph, nodes: Set[int]) -> Set[int]:
        """BFS up the DAG (following edges in reverse) to collect ancestors."""
        visited: Set[int] = set()
        queue: Deque[int] = deque()
        for n in nodes:
            if n in dag:
                queue.append(n)
                visited.add(n)
        while queue:
            current = queue.popleft()
            for parent in dag.predecessors(current):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return visited - nodes

    # ------------------------------------------------------------------
    # Triangulation (chordalization)
    # ------------------------------------------------------------------

    def triangulate(
        self, graph: nx.Graph, method: str = "min_fill"
    ) -> nx.Graph:
        """Triangulate *graph* using an elimination-ordering heuristic.

        Parameters
        ----------
        method : ``'min_fill'`` or ``'min_degree'``

        Returns the triangulated (chordal) graph.
        """
        if method == "min_fill":
            tri, ordering = self._min_fill_triangulation(graph)
        elif method == "min_degree":
            tri, ordering = self._min_degree_triangulation(graph)
        else:
            raise ValueError(f"Unknown triangulation method: {method}")
        self._elimination_orderings[method] = ordering
        return tri

    def _min_fill_triangulation(
        self, graph: nx.Graph
    ) -> Tuple[nx.Graph, List[int]]:
        """Greedy min-fill elimination: at each step pick the node whose
        elimination adds the fewest fill edges, eliminate it (connect its
        remaining neighbours into a clique), and record the ordering.
        """
        g = graph.copy()
        ordering: List[int] = []
        fill_edges: List[Tuple[int, int]] = []
        remaining = set(g.nodes())

        while remaining:
            best_node = None
            best_fill = float("inf")
            best_pairs: List[Tuple[int, int]] = []

            for node in remaining:
                nbrs = [n for n in g.neighbors(node) if n in remaining]
                pairs = []
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        if not g.has_edge(nbrs[i], nbrs[j]):
                            pairs.append((nbrs[i], nbrs[j]))
                if len(pairs) < best_fill:
                    best_fill = len(pairs)
                    best_node = node
                    best_pairs = pairs
                if best_fill == 0:
                    break

            ordering.append(best_node)
            for u, v in best_pairs:
                g.add_edge(u, v, fill_edge=True)
                fill_edges.append((u, v))
            remaining.remove(best_node)

        self._triangulation_history = fill_edges
        return g, ordering

    def _min_degree_triangulation(
        self, graph: nx.Graph
    ) -> Tuple[nx.Graph, List[int]]:
        """Greedy min-degree elimination: at each step pick the node with
        the fewest remaining neighbours, add fill edges among its neighbours,
        then remove it.
        """
        g = graph.copy()
        ordering: List[int] = []
        fill_edges: List[Tuple[int, int]] = []
        remaining = set(g.nodes())

        while remaining:
            best_node = min(
                remaining,
                key=lambda n: sum(1 for nb in g.neighbors(n) if nb in remaining),
            )
            ordering.append(best_node)
            nbrs = [n for n in g.neighbors(best_node) if n in remaining]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if not g.has_edge(nbrs[i], nbrs[j]):
                        g.add_edge(nbrs[i], nbrs[j], fill_edge=True)
                        fill_edges.append((nbrs[i], nbrs[j]))
            remaining.remove(best_node)

        self._triangulation_history = fill_edges
        return g, ordering

    # ------------------------------------------------------------------
    # Minimal triangulation (LEX-M)
    # ------------------------------------------------------------------

    def minimal_triangulation(self, graph: nx.Graph) -> nx.Graph:
        """Compute a *minimal* triangulation via the LEX-M algorithm.

        A triangulation is minimal if no single fill edge can be removed
        while keeping the graph chordal.
        """
        tri, _ = self._lex_m(graph)
        return tri

    def _lex_m(self, graph: nx.Graph) -> Tuple[nx.Graph, List[int]]:
        """LEX-M: a variant of LexBFS that inserts fill edges *only* when
        necessary to maintain a perfect elimination ordering.

        Algorithm (Berry, Blair, Heggernes, Peyton 2004):
        -------------------------------------------------
        Maintain labels (lists of integers, compared lexicographically in
        descending order).  At step i (counting from n down to 1):
          1. Pick the un-numbered vertex with the lexicographically largest
             label.
          2. Number it with the current step.
          3. For every un-numbered neighbour w of the chosen vertex:
             - Append the current step number to w's label.
             - For every un-numbered vertex u that is *not* adjacent to the
               chosen vertex but IS reachable from the chosen vertex via a
               path of un-numbered vertices all with labels ≤ w's *old*
               label:  add the fill edge (chosen, u) and append the step
               to u's label as well.
        """
        g = graph.copy()
        nodes = list(g.nodes())
        n = len(nodes)
        if n == 0:
            return g, []

        label: Dict[int, List[int]] = {v: [] for v in nodes}
        numbered: Dict[int, bool] = {v: False for v in nodes}
        ordering: List[int] = []
        fill_edges: List[Tuple[int, int]] = []

        for step in range(n, 0, -1):
            unnumbered = [v for v in nodes if not numbered[v]]
            best = max(unnumbered, key=lambda v: label[v])

            numbered[best] = True
            ordering.append(best)

            unnumbered_nbrs = [
                w for w in g.neighbors(best) if not numbered[w]
            ]

            reach = self._lex_m_reach(g, best, unnumbered_nbrs, numbered, label)

            for w in reach:
                if not g.has_edge(best, w):
                    g.add_edge(best, w, fill_edge=True)
                    fill_edges.append((best, w))
                label[w].append(step)

            for w in unnumbered_nbrs:
                if step not in label[w]:
                    label[w].append(step)

        ordering.reverse()
        self._triangulation_history = fill_edges
        return g, ordering

    def _lex_m_reach(
        self,
        g: nx.Graph,
        v: int,
        initial_nbrs: List[int],
        numbered: Dict[int, bool],
        label: Dict[int, List[int]],
    ) -> Set[int]:
        """Compute the set of un-numbered vertices reachable from *v*
        through paths of un-numbered vertices whose labels are bounded,
        following the LEX-M reachability rule.

        A vertex u is reachable if there is a path v, w1, w2, …, u where
        every intermediate wi is un-numbered and label(wi) < label(u)
        (lexicographically).
        """
        reached: Set[int] = set(initial_nbrs)
        queue: Deque[int] = deque(initial_nbrs)

        while queue:
            w = queue.popleft()
            w_label = label[w]
            for u in g.neighbors(w):
                if numbered[u] or u == v or u in reached:
                    continue
                if label[u] <= w_label:
                    reached.add(u)
                    queue.append(u)

        return reached

    # ------------------------------------------------------------------
    # Chordality testing
    # ------------------------------------------------------------------

    def is_chordal(self, graph: nx.Graph) -> bool:
        """Test chordality via LexBFS + PEO verification."""
        if graph.number_of_nodes() <= 3:
            return True
        ordering = self._lexbfs(graph)
        return self._is_perfect_elimination_ordering(graph, ordering)

    def find_perfect_elimination_ordering(
        self, graph: nx.Graph
    ) -> Optional[List[int]]:
        """Return a PEO if the graph is chordal, else ``None``."""
        ordering = self._lexbfs(graph)
        if self._is_perfect_elimination_ordering(graph, ordering):
            return ordering
        return None

    def _lexbfs(self, graph: nx.Graph) -> List[int]:
        """Lexicographic BFS using partition refinement.

        Maintain an ordered list of *sets* (partitions).  Initially there
        is one partition containing all vertices.  At each step:
          1. Pick the first vertex from the first partition.
          2. Output it.
          3. Refine every partition: split each set S into
             S ∩ N(v)  and  S \\ N(v), placing the intersection *before*
             the remainder in the list.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return []

        partitions: List[List[int]] = [list(nodes)]
        ordering: List[int] = []
        in_ordering: Set[int] = set()

        for _ in range(n):
            while partitions and not partitions[0]:
                partitions.pop(0)
            if not partitions:
                break

            v = partitions[0].pop(0)
            if not partitions[0]:
                partitions.pop(0)
            ordering.append(v)
            in_ordering.add(v)

            nbrs = set(graph.neighbors(v)) - in_ordering

            new_partitions: List[List[int]] = []
            for part in partitions:
                intersection = [x for x in part if x in nbrs]
                remainder = [x for x in part if x not in nbrs]
                if intersection:
                    new_partitions.append(intersection)
                if remainder:
                    new_partitions.append(remainder)
            partitions = new_partitions

        return ordering

    def _is_perfect_elimination_ordering(
        self, graph: nx.Graph, ordering: List[int]
    ) -> bool:
        """Check the PEO property: for every vertex v at position i,
        the set of v's neighbours that appear *later* in the ordering
        must form a clique in the graph.
        """
        pos = {v: i for i, v in enumerate(ordering)}
        n = len(ordering)

        for idx, v in enumerate(ordering):
            later_nbrs = [
                w for w in graph.neighbors(v) if pos.get(w, -1) > idx
            ]
            for i in range(len(later_nbrs)):
                for j in range(i + 1, len(later_nbrs)):
                    if not graph.has_edge(later_nbrs[i], later_nbrs[j]):
                        return False
        return True

    # ------------------------------------------------------------------
    # Clique extraction & clique tree
    # ------------------------------------------------------------------

    def extract_maximal_cliques(
        self, chordal_graph: nx.Graph
    ) -> List[FrozenSet[int]]:
        """Extract maximal cliques from a chordal graph using a PEO.

        For each vertex v in the PEO, form the candidate clique
        {v} ∪ {later neighbours of v}.  Keep only those that are
        *maximal* (not a subset of any other candidate).
        """
        ordering = self.find_perfect_elimination_ordering(chordal_graph)
        if ordering is None:
            raise ValueError("Graph is not chordal; cannot extract cliques via PEO")

        pos = {v: i for i, v in enumerate(ordering)}
        candidates: List[FrozenSet[int]] = []

        for idx, v in enumerate(ordering):
            later_nbrs = frozenset(
                w for w in chordal_graph.neighbors(v) if pos[w] > idx
            )
            clique = frozenset({v}) | later_nbrs
            candidates.append(clique)

        maximal: List[FrozenSet[int]] = []
        candidates_sorted = sorted(candidates, key=len, reverse=True)
        for c in candidates_sorted:
            if not any(c < m for m in maximal):
                is_subset = False
                for m in maximal:
                    if c <= m:
                        is_subset = True
                        break
                if not is_subset:
                    maximal.append(c)

        return maximal

    def build_clique_tree(
        self, cliques: List[FrozenSet[int]]
    ) -> nx.Graph:
        """Build a clique tree (junction tree) via maximum-weight spanning
        tree on a complete graph where the weight of each edge is the size
        of the intersection of the two cliques.

        Verifies the running intersection property at the end.
        """
        k = len(cliques)
        if k == 0:
            return nx.Graph()
        if k == 1:
            tree = nx.Graph()
            tree.add_node(0, clique=cliques[0])
            return tree

        complete = nx.Graph()
        for i in range(k):
            complete.add_node(i, clique=cliques[i])
        for i in range(k):
            for j in range(i + 1, k):
                sep = cliques[i] & cliques[j]
                weight = len(sep)
                if weight > 0:
                    complete.add_edge(i, j, weight=weight, separator=sep)

        if complete.number_of_edges() == 0:
            tree = nx.Graph()
            for i in range(k):
                tree.add_node(i, clique=cliques[i])
            for i in range(1, k):
                tree.add_edge(i - 1, i, weight=0, separator=frozenset())
            return tree

        # Kruskal's maximum spanning tree
        edges = sorted(
            complete.edges(data=True), key=lambda e: e[2]["weight"], reverse=True
        )
        parent = list(range(k))
        rank = [0] * k

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        tree = nx.Graph()
        for i in range(k):
            tree.add_node(i, clique=cliques[i])

        for u, v, data in edges:
            if union(u, v):
                tree.add_edge(u, v, **data)
            if tree.number_of_edges() == k - 1:
                break

        # If the tree is still disconnected (disjoint cliques), connect
        # remaining components with zero-weight edges.
        components = list(nx.connected_components(tree))
        if len(components) > 1:
            for idx in range(1, len(components)):
                a = next(iter(components[idx - 1]))
                b = next(iter(components[idx]))
                tree.add_edge(a, b, weight=0, separator=frozenset())

        rip_ok = self._verify_running_intersection(tree, cliques)
        if not rip_ok:
            tree.graph["rip_valid"] = False
        else:
            tree.graph["rip_valid"] = True

        return tree

    def _verify_running_intersection(
        self, tree: nx.Graph, cliques: List[FrozenSet[int]]
    ) -> bool:
        """Verify the running intersection property (RIP).

        For every variable v, the set of tree nodes whose cliques contain v
        must form a connected subtree.
        """
        all_vars: Set[int] = set()
        for c in cliques:
            all_vars.update(c)

        node_to_clique: Dict[int, FrozenSet[int]] = {}
        for i, c in enumerate(cliques):
            node_to_clique[i] = c

        for var in all_vars:
            containing = [i for i, c in enumerate(cliques) if var in c]
            if len(containing) <= 1:
                continue
            subtree = tree.subgraph(containing)
            if not nx.is_connected(subtree):
                return False
        return True

    # ------------------------------------------------------------------
    # Full pipeline helpers
    # ------------------------------------------------------------------

    def moral_triangulate_cliques(
        self,
        dag: nx.DiGraph,
        method: str = "min_fill",
    ) -> Tuple[nx.Graph, List[FrozenSet[int]], nx.Graph]:
        """End-to-end: moralize → triangulate → extract cliques → build tree.

        Returns (triangulated_graph, cliques, clique_tree).
        """
        moral = self.moralize(dag)
        tri = self.triangulate(moral, method=method)
        cliques = self.extract_maximal_cliques(tri)
        tree = self.build_clique_tree(cliques)
        return tri, cliques, tree

    def dseparation_moral(
        self,
        dag: nx.DiGraph,
        x: Set[int],
        y: Set[int],
        z: Set[int],
    ) -> bool:
        """Test d-separation of *x* and *y* given *z* using the moral-graph
        criterion (ancestral graph approach).

        Steps:
        1. Build ancestral graph of x ∪ y ∪ z.
        2. Moralize the ancestral graph.
        3. Remove nodes in z.
        4. x ⊥ y | z  iff  no path between any node in x and any node in y.
        """
        target = x | y | z
        anc_dag = self.ancestral_graph(dag, target)
        moral = self.moralize(anc_dag)

        moral_reduced = moral.copy()
        moral_reduced.remove_nodes_from(z)

        for a in x:
            if a not in moral_reduced:
                continue
            for b in y:
                if b not in moral_reduced:
                    continue
                if nx.has_path(moral_reduced, a, b):
                    return False
        return True

    def treewidth_upper_bound(self, graph: nx.Graph) -> int:
        """Compute an upper bound on the treewidth via min-fill triangulation.

        The treewidth equals (max clique size in optimal triangulation) - 1.
        Since min-fill is a heuristic, this gives an upper bound.
        """
        tri = self.triangulate(graph, method="min_fill")
        cliques = self.extract_maximal_cliques(tri)
        if not cliques:
            return 0
        return max(len(c) for c in cliques) - 1

    def fill_edge_count(self, graph: nx.Graph, method: str = "min_fill") -> int:
        """Return the number of fill edges added by the given triangulation method."""
        original_edges = set(graph.edges())
        tri = self.triangulate(graph, method=method)
        tri_edges = set(tri.edges())
        new_edges = tri_edges - original_edges
        return len(new_edges)

    def compare_triangulations(
        self, graph: nx.Graph
    ) -> Dict[str, Dict[str, object]]:
        """Compare min-fill, min-degree, and minimal (LEX-M) triangulations."""
        original_edge_count = graph.number_of_edges()
        results: Dict[str, Dict[str, object]] = {}

        for method in ("min_fill", "min_degree"):
            tri = self.triangulate(graph, method=method)
            cliques = self.extract_maximal_cliques(tri)
            max_clique = max(len(c) for c in cliques) if cliques else 0
            results[method] = {
                "fill_edges": tri.number_of_edges() - original_edge_count,
                "max_clique_size": max_clique,
                "treewidth_ub": max_clique - 1 if max_clique > 0 else 0,
                "num_cliques": len(cliques),
            }

        tri_min = self.minimal_triangulation(graph)
        cliques_min = self.extract_maximal_cliques(tri_min)
        max_clique_min = max(len(c) for c in cliques_min) if cliques_min else 0
        results["lex_m"] = {
            "fill_edges": tri_min.number_of_edges() - original_edge_count,
            "max_clique_size": max_clique_min,
            "treewidth_ub": max_clique_min - 1 if max_clique_min > 0 else 0,
            "num_cliques": len(cliques_min),
        }

        return results

    def induced_width(self, graph: nx.Graph, ordering: List[int]) -> int:
        """Compute the induced width of *graph* under a specific elimination
        ordering (without mutating *graph*).

        The induced width is the maximum number of neighbours a vertex has
        among vertices later in the ordering, after processing fill edges.
        """
        g = graph.copy()
        pos = {v: i for i, v in enumerate(ordering)}
        max_width = 0

        for v in ordering:
            later_nbrs = [w for w in g.neighbors(v) if pos.get(w, -1) > pos[v]]
            max_width = max(max_width, len(later_nbrs))
            for i in range(len(later_nbrs)):
                for j in range(i + 1, len(later_nbrs)):
                    if not g.has_edge(later_nbrs[i], later_nbrs[j]):
                        g.add_edge(later_nbrs[i], later_nbrs[j])

        return max_width

    def greedy_coloring_on_moral(self, dag: nx.DiGraph) -> Dict[int, int]:
        """Greedy graph coloring on the moral graph using LexBFS ordering.

        Returns a dict mapping each node to its color (integer ≥ 0).
        For a chordal graph, this yields an optimal coloring.
        """
        moral = self.moralize(dag)
        ordering = self._lexbfs(moral)
        ordering.reverse()

        color: Dict[int, int] = {}
        for v in ordering:
            used = {color[w] for w in moral.neighbors(v) if w in color}
            c = 0
            while c in used:
                c += 1
            color[v] = c

        return color

    def neighborhood_matrix(self, graph: nx.Graph) -> np.ndarray:
        """Return the adjacency matrix of *graph* as a numpy array,
        with rows/columns ordered by sorted node list.
        """
        nodes = sorted(graph.nodes())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        mat = np.zeros((n, n), dtype=np.int32)
        for u, v in graph.edges():
            i, j = idx[u], idx[v]
            mat[i, j] = 1
            mat[j, i] = 1
        return mat

    def moral_edge_density(self, dag: nx.DiGraph) -> float:
        """Fraction of edges in the moral graph that are *moral* (marriage)
        edges, i.e. not present as directed edges in the original DAG.
        """
        moral = self.moralize(dag)
        total = moral.number_of_edges()
        if total == 0:
            return 0.0
        moral_count = sum(
            1 for _, _, d in moral.edges(data=True) if d.get("moral_edge", False)
        )
        return moral_count / total

    def separator_sizes(self, clique_tree: nx.Graph) -> List[int]:
        """Return the list of separator sizes on each edge of a clique tree."""
        sizes: List[int] = []
        for _, _, data in clique_tree.edges(data=True):
            sep = data.get("separator", frozenset())
            sizes.append(len(sep))
        return sizes

    def maximum_cardinality_search(self, graph: nx.Graph) -> List[int]:
        """Maximum cardinality search (MCS) ordering.

        At each step, pick the un-numbered vertex adjacent to the most
        already-numbered vertices.  Ties broken arbitrarily.
        Like LexBFS, MCS produces a PEO iff the graph is chordal.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return []

        numbered: Set[int] = set()
        weight: Dict[int, int] = {v: 0 for v in nodes}
        ordering: List[int] = []

        for _ in range(n):
            best = max(
                (v for v in nodes if v not in numbered),
                key=lambda v: weight[v],
            )
            ordering.append(best)
            numbered.add(best)
            for w in graph.neighbors(best):
                if w not in numbered:
                    weight[w] += 1

        return ordering

    def elimination_game(
        self, graph: nx.Graph, ordering: List[int]
    ) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
        """Simulate the elimination game on *graph* using the given ordering.

        Process vertices in order: for each vertex, connect all its
        remaining (later) neighbours, then conceptually remove it.
        Returns (triangulated_graph, list_of_fill_edges).
        """
        g = graph.copy()
        fill: List[Tuple[int, int]] = []
        pos = {v: i for i, v in enumerate(ordering)}

        for v in ordering:
            later = [w for w in g.neighbors(v) if pos.get(w, -1) > pos[v]]
            for i in range(len(later)):
                for j in range(i + 1, len(later)):
                    if not g.has_edge(later[i], later[j]):
                        g.add_edge(later[i], later[j], fill_edge=True)
                        fill.append((later[i], later[j]))

        return g, fill
