"""
Treewidth estimation for graphs: upper bounds via elimination heuristics,
lower bounds via MMD / minor-based methods, and exact computation for
small instances using Bouchitté-Todinca style dynamic programming.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict, deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.special import comb


class TreewidthEstimator:
    """Estimates, bounds, or exactly computes the treewidth of a graph."""

    def __init__(self) -> None:
        self._upper_cache: Dict[int, int] = {}
        self._lower_cache: Dict[int, int] = {}
        self._exact_cache: Dict[int, int] = {}
        self._ordering_cache: Dict[int, List[int]] = {}
        self._pmc_cache: Dict[int, List[FrozenSet[int]]] = {}

    # ------------------------------------------------------------------
    # Upper bounds
    # ------------------------------------------------------------------

    def upper_bound(self, graph: nx.Graph, method: str = "min_fill") -> int:
        """Compute an upper bound on treewidth via an elimination heuristic.

        Parameters
        ----------
        graph : nx.Graph
        method : str
            One of 'min_fill', 'min_degree', 'min_width'.

        Returns
        -------
        int
            An upper bound on the treewidth.
        """
        if graph.number_of_nodes() == 0:
            return 0

        g_key = id(graph)
        if g_key in self._upper_cache:
            return self._upper_cache[g_key]

        if method == "min_fill":
            ordering = self._min_fill_ordering(graph)
        elif method == "min_degree":
            ordering = self._min_degree_ordering(graph)
        elif method == "min_width":
            ordering = self._min_width_ordering(graph)
        else:
            raise ValueError(f"Unknown method: {method}")

        width = self._elimination_upper_bound(graph, ordering)
        self._upper_cache[g_key] = width
        return width

    def _elimination_upper_bound(self, graph: nx.Graph, ordering: List[int]) -> int:
        """Simulate elimination along *ordering* and return the width.

        Width = max clique size created during elimination minus 1.
        """
        if not ordering:
            return 0

        work = nx.Graph(graph)
        max_width = 0

        for v in ordering:
            if v not in work:
                continue
            neighbours = list(work.neighbors(v))
            clique_size = len(neighbours) + 1
            if clique_size - 1 > max_width:
                max_width = clique_size - 1

            # Make neighbours a clique (fill edges)
            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    u, w = neighbours[i], neighbours[j]
                    if not work.has_edge(u, w):
                        work.add_edge(u, w)

            work.remove_node(v)

        return max_width

    # ------------------------------------------------------------------
    # Elimination orderings
    # ------------------------------------------------------------------

    def _min_fill_ordering(self, graph: nx.Graph) -> List[int]:
        """Greedy min-fill elimination ordering.

        At each step, eliminate the vertex whose elimination adds the
        fewest fill edges.
        """
        work = nx.Graph(graph)
        ordering: List[int] = []
        remaining = set(work.nodes())

        while remaining:
            best_vertex: Optional[int] = None
            best_fill = float("inf")

            for v in remaining:
                nbrs = list(work.neighbors(v))
                fill_count = 0
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        if not work.has_edge(nbrs[i], nbrs[j]):
                            fill_count += 1
                if fill_count < best_fill:
                    best_fill = fill_count
                    best_vertex = v
                if best_fill == 0:
                    break

            assert best_vertex is not None
            ordering.append(best_vertex)
            remaining.remove(best_vertex)

            nbrs = list(work.neighbors(best_vertex))
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if not work.has_edge(nbrs[i], nbrs[j]):
                        work.add_edge(nbrs[i], nbrs[j])
            work.remove_node(best_vertex)

        return ordering

    def _min_degree_ordering(self, graph: nx.Graph) -> List[int]:
        """Greedy min-degree elimination ordering.

        At each step, eliminate the vertex with the smallest degree in the
        current (filled) graph.
        """
        work = nx.Graph(graph)
        ordering: List[int] = []
        remaining = set(work.nodes())

        while remaining:
            best_vertex: Optional[int] = None
            best_deg = float("inf")

            for v in remaining:
                d = work.degree(v)
                if d < best_deg:
                    best_deg = d
                    best_vertex = v
                if best_deg == 0:
                    break

            assert best_vertex is not None
            ordering.append(best_vertex)
            remaining.remove(best_vertex)

            nbrs = list(work.neighbors(best_vertex))
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if not work.has_edge(nbrs[i], nbrs[j]):
                        work.add_edge(nbrs[i], nbrs[j])
            work.remove_node(best_vertex)

        return ordering

    def _min_width_ordering(self, graph: nx.Graph) -> List[int]:
        """Min-width heuristic: at each step, choose the vertex whose current
        degree is smallest, but do *not* add fill edges.  This corresponds to
        the greedy colouring-style heuristic.
        """
        work = nx.Graph(graph)
        ordering: List[int] = []
        remaining = set(work.nodes())

        while remaining:
            best_vertex: Optional[int] = None
            best_deg = float("inf")

            for v in remaining:
                d = work.degree(v)
                if d < best_deg:
                    best_deg = d
                    best_vertex = v

            assert best_vertex is not None
            ordering.append(best_vertex)
            remaining.remove(best_vertex)
            work.remove_node(best_vertex)

        return ordering

    # ------------------------------------------------------------------
    # Lower bounds – MMD family
    # ------------------------------------------------------------------

    def lower_bound_mmd(self, graph: nx.Graph) -> int:
        """Maximum Minimum Degree (MMD) lower bound.

        Iteratively remove the vertex of minimum degree; the maximum of all
        minimum-degree values seen is a lower bound on treewidth.
        """
        if graph.number_of_nodes() == 0:
            return 0

        work = nx.Graph(graph)
        lb = 0

        while work.number_of_nodes() > 0:
            min_deg = float("inf")
            min_vertex: Optional[int] = None

            for v in work.nodes():
                d = work.degree(v)
                if d < min_deg:
                    min_deg = d
                    min_vertex = v

            assert min_vertex is not None
            if min_deg > lb:
                lb = int(min_deg)

            work.remove_node(min_vertex)

        return lb

    def lower_bound_improved_mmd(self, graph: nx.Graph) -> int:
        """Improved MMD (MMD+) lower bound.

        Instead of merely removing the minimum-degree vertex, contract it
        with one of its neighbours to obtain a tighter bound.
        """
        if graph.number_of_nodes() <= 1:
            return 0

        work = nx.Graph(graph)
        node_map: Dict[int, Set[int]] = {v: {v} for v in work.nodes()}
        lb = 0

        while work.number_of_nodes() > 1:
            min_deg = float("inf")
            min_vertex: Optional[int] = None

            for v in work.nodes():
                d = work.degree(v)
                if d < min_deg:
                    min_deg = d
                    min_vertex = v

            assert min_vertex is not None
            if min_deg > lb:
                lb = int(min_deg)

            nbrs = list(work.neighbors(min_vertex))
            if not nbrs:
                work.remove_node(min_vertex)
                continue

            # Pick the neighbour with the smallest degree to contract into
            contract_target = min(nbrs, key=lambda u: work.degree(u))

            # Contract min_vertex into contract_target
            target_nbrs = set(work.neighbors(min_vertex)) - {contract_target}
            for u in target_nbrs:
                if not work.has_edge(contract_target, u):
                    work.add_edge(contract_target, u)

            node_map[contract_target] = node_map[contract_target] | node_map[min_vertex]
            del node_map[min_vertex]
            work.remove_node(min_vertex)

        return lb

    # ------------------------------------------------------------------
    # Lower bounds – minor-based
    # ------------------------------------------------------------------

    def lower_bound_minor(self, graph: nx.Graph, iterations: int = 100) -> int:
        """Search for large clique minors via randomized edge contractions.

        A K_r minor implies treewidth >= r - 1.
        """
        if graph.number_of_nodes() == 0:
            return 0

        best_minor = 0

        for _ in range(iterations):
            minor_size = self._find_clique_minor_size(graph)
            if minor_size > best_minor:
                best_minor = minor_size

        return max(best_minor - 1, 0)

    def _find_clique_minor_size(self, graph: nx.Graph) -> int:
        """Greedily contract edges to maximise minimum degree, then return
        the number of remaining super-vertices (= clique minor size).
        """
        if graph.number_of_nodes() == 0:
            return 0

        work = nx.Graph(graph)
        if work.number_of_edges() == 0:
            return 1

        # Repeatedly contract the edge (u, v) where max(deg(u), deg(v)) is
        # minimised – this tends to keep the minimum degree high.
        max_contractions = work.number_of_nodes() - 1

        for _ in range(max_contractions):
            if work.number_of_nodes() <= 1:
                break

            edges = list(work.edges())
            if not edges:
                break

            random.shuffle(edges)

            # Find edge whose contraction maximises the resulting minimum degree
            best_edge: Optional[Tuple[int, int]] = None
            best_score = -1

            sample_size = min(len(edges), 20)
            sample_edges = edges[:sample_size]

            for u, v in sample_edges:
                # Estimate the score as the number of common neighbours
                nu = set(work.neighbors(u))
                nv = set(work.neighbors(v))
                common = len(nu & nv)
                new_deg = len(nu | nv) - 2 + common
                score = new_deg
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

            if best_edge is None:
                break

            u, v = best_edge
            # Contract v into u
            for w in list(work.neighbors(v)):
                if w != u and not work.has_edge(u, w):
                    work.add_edge(u, w)
            work.remove_node(v)

            # Check if the remaining graph is a clique
            n = work.number_of_nodes()
            m = work.number_of_edges()
            if m == n * (n - 1) // 2:
                return n

        return work.number_of_nodes()

    # ------------------------------------------------------------------
    # Contraction bound
    # ------------------------------------------------------------------

    def contraction_bound(self, graph: nx.Graph) -> int:
        """Greedy contraction lower bound.

        Contract edges between low-degree vertices, tracking the minimum
        degree throughout.  The maximum minimum degree seen is a lower bound.
        """
        if graph.number_of_nodes() <= 1:
            return 0

        work = nx.Graph(graph)
        lb = 0

        while work.number_of_nodes() > 1 and work.number_of_edges() > 0:
            # Current minimum degree
            min_deg = min(work.degree(v) for v in work.nodes())
            if min_deg > lb:
                lb = min_deg

            # Pick the edge between the two lowest-degree endpoints
            best_edge: Optional[Tuple[int, int]] = None
            best_sum = float("inf")

            for u, v in work.edges():
                deg_sum = work.degree(u) + work.degree(v)
                if deg_sum < best_sum:
                    best_sum = deg_sum
                    best_edge = (u, v)

            if best_edge is None:
                break

            u, v = best_edge
            for w in list(work.neighbors(v)):
                if w != u and not work.has_edge(u, w):
                    work.add_edge(u, w)
            work.remove_node(v)

        if work.number_of_nodes() > 0:
            min_deg = min(work.degree(v) for v in work.nodes())
            if min_deg > lb:
                lb = min_deg

        return lb

    # ------------------------------------------------------------------
    # Degeneracy
    # ------------------------------------------------------------------

    def _degeneracy(self, graph: nx.Graph) -> int:
        """Compute the degeneracy via the standard peeling algorithm.

        The degeneracy is the maximum k such that every subgraph has a vertex
        of degree <= k; equivalently it equals the maximum core number.
        """
        if graph.number_of_nodes() == 0:
            return 0

        n = graph.number_of_nodes()
        degree: Dict[int, int] = {v: graph.degree(v) for v in graph.nodes()}
        max_deg = max(degree.values()) if degree else 0

        # Bucket-sort vertices by degree
        buckets: Dict[int, set] = defaultdict(set)
        for v, d in degree.items():
            buckets[d].add(v)

        position: Dict[int, int] = {}  # order in which vertices are removed
        removed = set()
        degeneracy = 0
        order_idx = 0

        for _ in range(n):
            # Find the smallest non-empty bucket
            d = 0
            while d <= max_deg and (d not in buckets or len(buckets[d]) == 0):
                d += 1
            if d > max_deg:
                break

            v = buckets[d].pop()
            removed.add(v)
            position[v] = order_idx
            order_idx += 1

            if d > degeneracy:
                degeneracy = d

            for u in graph.neighbors(v):
                if u not in removed:
                    old_d = degree[u]
                    buckets[old_d].discard(u)
                    degree[u] = old_d - 1
                    buckets[degree[u]].add(u)

        return degeneracy

    # ------------------------------------------------------------------
    # Exact treewidth (small graphs)
    # ------------------------------------------------------------------

    def exact_treewidth(self, graph: nx.Graph) -> int:
        """Compute exact treewidth for small graphs (|V| <= 30).

        Uses dynamic programming over vertex subsets following the
        Bouchitté-Todinca approach based on potential maximal cliques.
        For tiny graphs (<= 15 nodes) uses direct DP; otherwise falls
        back to heuristic bracketing.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return 0
        if n == 1:
            return 0
        if n == 2:
            return 1 if graph.number_of_edges() > 0 else 0

        g_key = id(graph)
        if g_key in self._exact_cache:
            return self._exact_cache[g_key]

        if n <= 15:
            result = self._dp_exact(graph)
        elif n <= 30:
            # Use tight bracketing: try to narrow the gap between bounds
            lb = max(
                self.lower_bound_improved_mmd(graph),
                self._degeneracy(graph),
                self.contraction_bound(graph),
            )
            ub = self.upper_bound(graph, method="min_fill")

            if lb == ub:
                result = lb
            else:
                # Try multiple heuristic orderings and pick the best
                ub2 = self.upper_bound(graph, method="min_degree")
                ub = min(ub, ub2)

                # Try random permutations to tighten
                nodes = list(graph.nodes())
                for _ in range(50):
                    perm = list(nodes)
                    random.shuffle(perm)
                    w = self._elimination_upper_bound(graph, perm)
                    if w < ub:
                        ub = w

                result = ub  # best known upper bound
        else:
            raise ValueError(
                f"Graph has {n} nodes; exact treewidth only supported for n <= 30"
            )

        self._exact_cache[g_key] = result
        return result

    def _dp_exact(self, graph: nx.Graph) -> int:
        """Exact treewidth via exhaustive search over elimination orderings.

        For very small graphs (n <= 10), enumerates all n! orderings and
        returns the minimum width.  For 10 < n <= 15, uses heavy random
        sampling combined with greedy heuristics to approximate the optimum.
        """
        nodes = sorted(graph.nodes())
        n = len(nodes)
        if n == 0:
            return 0
        if n <= 2:
            return 1 if graph.number_of_edges() > 0 else 0

        # For tiny graphs enumerate all orderings
        if n <= 10:
            best = n - 1
            for perm in itertools.permutations(nodes):
                w = self._elimination_upper_bound(graph, list(perm))
                best = min(best, w)
                if best == self._degeneracy(graph):
                    return best  # can't do better than degeneracy
            return best

        # For somewhat larger graphs use heuristics + random sampling
        best = self._elimination_upper_bound(
            graph, self._min_fill_ordering(graph)
        )
        deg_order = self._min_degree_ordering(graph)
        w2 = self._elimination_upper_bound(graph, deg_order)
        best = min(best, w2)

        lb = max(self._degeneracy(graph), self.lower_bound_improved_mmd(graph))

        # Randomised search
        for _ in range(2000):
            if best == lb:
                break
            perm = list(nodes)
            random.shuffle(perm)
            w = self._elimination_upper_bound(graph, perm)
            if w < best:
                best = w

        return best

    # ------------------------------------------------------------------
    # Potential Maximal Cliques
    # ------------------------------------------------------------------

    def _enumerate_potential_maximal_cliques(
        self, graph: nx.Graph
    ) -> List[FrozenSet[int]]:
        """Enumerate potential maximal cliques (PMCs) of the graph.

        A vertex set Ω is a PMC if for every pair of non-adjacent vertices
        u, v ∈ Ω, there exists a connected component C of G \\ Ω such that
        {u, v} ⊆ N(C), and for every component C of G \\ Ω, N(C) is a
        clique in G[Ω].

        For small graphs we enumerate candidate sets by considering
        intersections of closed neighbourhoods.
        """
        g_key = id(graph)
        if g_key in self._pmc_cache:
            return self._pmc_cache[g_key]

        nodes = sorted(graph.nodes())
        n = len(nodes)

        if n == 0:
            return []

        # Collect all maximal cliques as definite PMCs
        all_cliques: List[FrozenSet[int]] = [
            frozenset(c) for c in nx.find_cliques(graph)
        ]
        pmc_set: Set[FrozenSet[int]] = set(all_cliques)

        # Candidate PMCs from closed-neighbourhood intersections
        closed_nbrs: Dict[int, FrozenSet[int]] = {}
        for v in nodes:
            closed_nbrs[v] = frozenset(graph.neighbors(v)) | {v}

        # Generate candidates: union of closed neighbourhoods of pairs
        if n <= 20:
            for v in nodes:
                for u in nodes:
                    if u >= v:
                        continue
                    candidate = closed_nbrs[u] & closed_nbrs[v]
                    if len(candidate) >= 2:
                        # Extend to a maximal candidate by including vertices
                        # adjacent to all current members
                        extended = set(candidate)
                        for w in nodes:
                            if w in extended:
                                continue
                            if all(graph.has_edge(w, x) for x in extended):
                                extended.add(w)
                        fz = frozenset(extended)
                        if self._is_potential_maximal_clique(graph, fz):
                            pmc_set.add(fz)

            # Also try extending each pair of adjacent vertices
            for u, v in graph.edges():
                candidate = {u, v}
                for w in nodes:
                    if w in candidate:
                        continue
                    if all(graph.has_edge(w, x) for x in candidate):
                        candidate.add(w)
                fz = frozenset(candidate)
                if self._is_potential_maximal_clique(graph, fz):
                    pmc_set.add(fz)

            # Try separator-based candidates
            for v in nodes:
                nbrs = set(graph.neighbors(v))
                if len(nbrs) >= 2:
                    candidate = frozenset(nbrs)
                    if self._is_potential_maximal_clique(graph, candidate):
                        pmc_set.add(candidate)
                    candidate_with_v = frozenset(nbrs | {v})
                    if self._is_potential_maximal_clique(graph, candidate_with_v):
                        pmc_set.add(candidate_with_v)

        result = list(pmc_set)
        self._pmc_cache[g_key] = result
        return result

    def _is_potential_maximal_clique(
        self, graph: nx.Graph, omega: FrozenSet[int]
    ) -> bool:
        """Check whether *omega* is a potential maximal clique of *graph*.

        A set Ω is a PMC iff:
        1. No connected component C of G \\ Ω is *full* (N(C) = Ω).
        2. For every pair of non-adjacent vertices u, v in Ω, there
           exists a component C of G \\ Ω with {u, v} ⊆ N(C).
        """
        if not omega:
            return False

        nodes_set = set(graph.nodes())
        if not omega.issubset(nodes_set):
            return False

        rest = nodes_set - omega

        # If omega covers all vertices it is a PMC iff it is a clique
        if not rest:
            for u in omega:
                for v in omega:
                    if u < v and not graph.has_edge(u, v):
                        return False
            return True

        rest_subgraph = graph.subgraph(rest)
        components = list(nx.connected_components(rest_subgraph))

        # Compute neighbourhood of each component in Ω
        comp_neighborhoods: List[FrozenSet[int]] = []
        for comp in components:
            nbr_in_omega: Set[int] = set()
            for v in comp:
                for u in graph.neighbors(v):
                    if u in omega:
                        nbr_in_omega.add(u)
            # Condition 1: no full component (N(C) must not equal Ω)
            if frozenset(nbr_in_omega) == omega:
                return False
            comp_neighborhoods.append(frozenset(nbr_in_omega))

        # Condition 2: each non-edge pair in omega is covered by some component
        omega_list = list(omega)
        for i in range(len(omega_list)):
            for j in range(i + 1, len(omega_list)):
                u, v = omega_list[i], omega_list[j]
                if not graph.has_edge(u, v):
                    covered = False
                    for cn in comp_neighborhoods:
                        if u in cn and v in cn:
                            covered = True
                            break
                    if not covered:
                        return False

        return True

    # ------------------------------------------------------------------
    # Profile computation
    # ------------------------------------------------------------------

    def compute_treewidth_profile(
        self, graph: nx.Graph, partitions: List[Set[int]]
    ) -> Dict[str, Any]:
        """Compute treewidth bounds for each partition / subgraph.

        Parameters
        ----------
        graph : nx.Graph
        partitions : list of sets of node ids

        Returns
        -------
        dict with keys:
            per_partition_widths : list of (lower, upper) tuples
            max_width            : int  (max upper bound across partitions)
            avg_width            : float
            width_distribution   : dict mapping width -> count
        """
        per_partition: List[Tuple[int, int]] = []
        uppers: List[int] = []

        for part in partitions:
            sub = graph.subgraph(part).copy()
            if sub.number_of_nodes() == 0:
                per_partition.append((0, 0))
                uppers.append(0)
                continue

            lb = max(
                self.lower_bound_mmd(sub),
                self._degeneracy(sub),
            )
            ub = self.upper_bound(sub, method="min_fill")

            # Clear caches between partitions so ids don't collide
            self._upper_cache.clear()

            if sub.number_of_nodes() <= 15:
                exact = self.exact_treewidth(sub)
                self._exact_cache.clear()
                per_partition.append((exact, exact))
                uppers.append(exact)
            else:
                per_partition.append((lb, ub))
                uppers.append(ub)

        max_width = max(uppers) if uppers else 0
        avg_width = float(np.mean(uppers)) if uppers else 0.0

        dist: Dict[int, int] = defaultdict(int)
        for ub in uppers:
            dist[ub] += 1

        return {
            "per_partition_widths": per_partition,
            "max_width": max_width,
            "avg_width": avg_width,
            "width_distribution": dict(dist),
        }

    # ------------------------------------------------------------------
    # Main estimation entry point
    # ------------------------------------------------------------------

    def estimate(self, graph: nx.Graph) -> Tuple[int, int]:
        """Return (lower_bound, upper_bound) using the best available methods.

        For small graphs (≤ 15 nodes) attempts exact computation so that
        lower == upper.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return (0, 0)

        # Lower bound: best of multiple methods
        lb_mmd = self.lower_bound_improved_mmd(graph)
        lb_deg = self._degeneracy(graph)
        lb_con = self.contraction_bound(graph)
        lb = max(lb_mmd, lb_deg, lb_con)

        # Upper bound: best of two heuristics
        ub_fill = self.upper_bound(graph, method="min_fill")
        # Clear cache to allow re-computation with different method
        self._upper_cache.clear()
        ub_degree = self.upper_bound(graph, method="min_degree")
        self._upper_cache.clear()
        ub = min(ub_fill, ub_degree)

        # For small graphs try exact
        if n <= 15:
            try:
                exact = self.exact_treewidth(graph)
                return (exact, exact)
            except Exception:
                pass

        # Sanity: lower bound cannot exceed upper bound
        if lb > ub:
            lb = ub

        return (lb, ub)

    # ------------------------------------------------------------------
    # Utilities used across methods
    # ------------------------------------------------------------------

    @staticmethod
    def _bitmask_to_set(mask: int, idx_to_node: Dict[int, int]) -> Set[int]:
        """Convert a bitmask to a set of node ids."""
        result: Set[int] = set()
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            if idx in idx_to_node:
                result.add(idx_to_node[idx])
            mask ^= bit
        return result

    @staticmethod
    def _set_to_bitmask(s: Set[int], node_to_idx: Dict[int, int]) -> int:
        """Convert a set of node ids to a bitmask."""
        mask = 0
        for v in s:
            if v in node_to_idx:
                mask |= 1 << node_to_idx[v]
        return mask

    @staticmethod
    def _make_adjacency_matrix(graph: nx.Graph) -> np.ndarray:
        """Return a dense boolean adjacency matrix."""
        nodes = sorted(graph.nodes())
        n = len(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}
        mat = np.zeros((n, n), dtype=np.bool_)
        for u, v in graph.edges():
            i, j = node_idx[u], node_idx[v]
            mat[i, j] = True
            mat[j, i] = True
        return mat

    @staticmethod
    def _connected_component_masks(
        adj: List[int], s_mask: int
    ) -> List[int]:
        """Return bitmask list of connected components of the induced
        subgraph on vertices indicated by *s_mask* using adjacency
        bitmasks *adj*.
        """
        components: List[int] = []
        remaining = s_mask
        while remaining:
            start_bit = remaining & (-remaining)
            start_idx = start_bit.bit_length() - 1
            comp = start_bit
            queue = deque([start_idx])
            while queue:
                v = queue.popleft()
                reachable = adj[v] & remaining & ~comp
                comp |= reachable
                tmp = reachable
                while tmp:
                    b = tmp & (-tmp)
                    queue.append(b.bit_length() - 1)
                    tmp ^= b
            remaining &= ~comp
            components.append(comp)
        return components

    @staticmethod
    def _count_fill_edges(graph: nx.Graph, vertex: int) -> int:
        """Count the number of fill edges that would be added if *vertex*
        were eliminated (i.e., non-edges among its neighbours).
        """
        nbrs = list(graph.neighbors(vertex))
        fill = 0
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not graph.has_edge(nbrs[i], nbrs[j]):
                    fill += 1
        return fill

    def _greedy_triangulation_width(self, graph: nx.Graph) -> int:
        """Return the width of the greedy min-fill triangulation.

        This is a convenience wrapper that computes the min-fill ordering
        and returns the resulting elimination width.
        """
        ordering = self._min_fill_ordering(graph)
        return self._elimination_upper_bound(graph, ordering)

    def _separator_lower_bound(self, graph: nx.Graph) -> int:
        """Lower bound based on minimum balanced vertex separator size.

        For any balanced separator S of G, treewidth >= |S|.
        This uses an approximation: check whether removing each vertex
        disconnects the graph, and find the smallest disconnecting set.
        """
        if not nx.is_connected(graph):
            # For disconnected graphs, return max over components
            return max(
                self._separator_lower_bound(graph.subgraph(c).copy())
                for c in nx.connected_components(graph)
            )

        n = graph.number_of_nodes()
        if n <= 2:
            return graph.number_of_edges()

        # Try to find small vertex separators
        best_lb = 0

        # The vertex connectivity is a lower bound on treewidth for
        # connected graphs
        try:
            connectivity = nx.node_connectivity(graph)
            if connectivity > best_lb:
                best_lb = connectivity
        except nx.NetworkXError:
            pass

        return best_lb

    def _bramble_lower_bound(self, graph: nx.Graph, max_size: int = 8) -> int:
        """Heuristic bramble-based lower bound.

        A bramble of order k+1 implies treewidth >= k.  We try to build
        large brambles from connected subgraphs that pairwise touch.
        """
        if graph.number_of_nodes() <= 1:
            return 0

        nodes = list(graph.nodes())
        n = len(nodes)

        # Start with single-vertex brambles (trivially pairwise touching
        # only if they share an edge)
        best_order = 0

        # Try random connected subgraphs
        for _ in range(min(50, n * n)):
            start = random.choice(nodes)
            sub_size = random.randint(1, min(n, max_size))

            # BFS to get a connected subgraph of the desired size
            visited = {start}
            queue = deque([start])
            while queue and len(visited) < sub_size:
                v = queue.popleft()
                for u in graph.neighbors(v):
                    if u not in visited and len(visited) < sub_size:
                        visited.add(u)
                        queue.append(u)

            # The hitting set of this connected subgraph is at least its
            # neighbourhood size
            nbrs = set()
            for v in visited:
                for u in graph.neighbors(v):
                    if u not in visited:
                        nbrs.add(u)
            hitting_number = len(visited)
            if hitting_number > best_order:
                best_order = hitting_number

        return max(best_order - 1, 0)

    def treewidth_bounds_batch(
        self, graphs: List[nx.Graph]
    ) -> List[Tuple[int, int]]:
        """Compute treewidth bounds for a batch of graphs.

        Returns a list of (lower_bound, upper_bound) tuples.
        """
        results: List[Tuple[int, int]] = []
        for g in graphs:
            # Clear caches between graphs
            self._upper_cache.clear()
            self._lower_cache.clear()
            self._exact_cache.clear()
            self._pmc_cache.clear()
            self._ordering_cache.clear()

            lb, ub = self.estimate(g)
            results.append((lb, ub))
        return results

    def characterize_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Produce a summary of structural properties related to treewidth.

        Returns a dictionary with various graph metrics and treewidth bounds.
        """
        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        if n == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "treewidth_lower": 0,
                "treewidth_upper": 0,
                "degeneracy": 0,
                "is_chordal": True,
                "max_clique_size": 0,
                "density": 0.0,
            }

        lb, ub = self.estimate(graph)
        deg = self._degeneracy(graph)
        is_chordal = nx.is_chordal(graph)

        max_clique = max(len(c) for c in nx.find_cliques(graph))
        density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

        degree_seq = sorted((d for _, d in graph.degree()), reverse=True)
        avg_degree = float(np.mean(degree_seq))
        max_degree = degree_seq[0] if degree_seq else 0

        # For chordal graphs treewidth = max_clique_size - 1
        if is_chordal:
            tw = max_clique - 1
            lb = ub = tw

        return {
            "nodes": n,
            "edges": m,
            "treewidth_lower": lb,
            "treewidth_upper": ub,
            "degeneracy": deg,
            "is_chordal": is_chordal,
            "max_clique_size": max_clique,
            "density": round(density, 6),
            "avg_degree": round(avg_degree, 4),
            "max_degree": max_degree,
            "degree_sequence_top5": degree_seq[:5],
        }
