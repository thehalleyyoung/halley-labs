"""
Separator extraction, enumeration, scoring, and validation for
tree-decomposition–based causal graph analysis.

Provides algorithms for minimal separators, balanced separators,
causal-preservation scoring, and Menger's-theorem validation.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
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

import networkx as nx
import numpy as np


_INF_CAP = float("inf")


class SeparatorExtractor:
    """Extract, enumerate, score, and validate vertex separators."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise internal caches used across repeated queries."""
        self._cut_cache: Dict[Tuple[int, int], FrozenSet[int]] = {}
        self._disjoint_paths_cache: Dict[Tuple[int, int], List[List[int]]] = {}
        self._centrality_cache: Dict[int, Dict[int, float]] = {}
        self._separator_score_cache: Dict[
            Tuple[FrozenSet[int], Optional[int]], Dict[str, float]
        ] = {}
        self._component_cache: Dict[
            Tuple[int, FrozenSet[int]], List[Set[int]]
        ] = {}

    # ------------------------------------------------------------------
    # 1. Extract separators from a tree decomposition
    # ------------------------------------------------------------------

    def extract_separators(
        self,
        decomposition_bags: Dict[int, FrozenSet[int]],
        decomposition_tree: nx.Graph,
    ) -> List[FrozenSet[int]]:
        """Return unique separators obtained from adjacent bags.

        For every edge (i, j) in *decomposition_tree* the separator is
        ``decomposition_bags[i] & decomposition_bags[j]``.  Only non-empty,
        unique separators are returned.
        """
        seen: Set[FrozenSet[int]] = set()
        result: List[FrozenSet[int]] = []

        for u, v in decomposition_tree.edges():
            bag_u = decomposition_bags.get(u, frozenset())
            bag_v = decomposition_bags.get(v, frozenset())
            sep = bag_u & bag_v
            if sep and sep not in seen:
                seen.add(sep)
                result.append(sep)

        result.sort(key=lambda s: (len(s), sorted(s)))
        return result

    # ------------------------------------------------------------------
    # 2. Enumerate minimal a-b separators
    # ------------------------------------------------------------------

    def enumerate_minimal_separators(
        self, graph: nx.Graph
    ) -> List[FrozenSet[int]]:
        """Enumerate all minimal vertex separators for every pair (a, b).

        For each non-adjacent pair we compute the minimum vertex cut via
        `_find_min_vertex_cut` and then verify minimality.  Verified
        separators are collected into a unique set and returned sorted by
        size.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        unique_seps: Set[FrozenSet[int]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                if a == b:
                    continue
                if graph.has_edge(a, b):
                    continue
                if not nx.has_path(graph, a, b):
                    continue

                sep = self._find_min_vertex_cut(graph, a, b)
                if not sep:
                    continue

                if self._is_minimal_separator(graph, sep, a, b):
                    unique_seps.add(sep)

                remaining = set(graph.nodes()) - sep - {a, b}
                sub = graph.subgraph(remaining | {a, b}).copy()
                if nx.has_path(sub, a, b):
                    continue

                components = self._components_after_removal(graph, sep)
                comp_a: Optional[Set[int]] = None
                comp_b: Optional[Set[int]] = None
                for comp in components:
                    if a in comp:
                        comp_a = comp
                    if b in comp:
                        comp_b = comp

                if comp_a is None or comp_b is None:
                    continue

                border_a: Set[int] = set()
                for v in comp_a:
                    for nb in graph.neighbors(v):
                        if nb in sep:
                            border_a.add(nb)
                border_b: Set[int] = set()
                for v in comp_b:
                    for nb in graph.neighbors(v):
                        if nb in sep:
                            border_b.add(nb)

                candidate = frozenset(border_a & border_b)
                if candidate and candidate not in unique_seps:
                    if self._is_minimal_separator(graph, candidate, a, b):
                        unique_seps.add(candidate)

        result = sorted(unique_seps, key=lambda s: (len(s), sorted(s)))
        return result

    # ------------------------------------------------------------------
    # 3. Min vertex cut via vertex splitting + max-flow
    # ------------------------------------------------------------------

    def _find_min_vertex_cut(
        self, graph: nx.Graph, source: int, target: int
    ) -> FrozenSet[int]:
        """Minimum vertex cut between *source* and *target*.

        Builds an auxiliary directed graph where every internal vertex *v*
        is split into ``v_in`` and ``v_out`` connected by an arc of
        capacity 1.  Original edges become arcs of infinite capacity
        between the ``_out`` of one endpoint and the ``_in`` of the other.
        After computing max-flow the minimum cut vertices are recovered.
        """
        key = (min(source, target), max(source, target))
        if key in self._cut_cache:
            return self._cut_cache[key]

        if source == target:
            self._cut_cache[key] = frozenset()
            return frozenset()
        if not nx.has_path(graph, source, target):
            self._cut_cache[key] = frozenset()
            return frozenset()

        aux = nx.DiGraph()
        node_list = list(graph.nodes())
        in_label = {}
        out_label = {}
        counter = 0
        for v in node_list:
            in_label[v] = counter
            out_label[v] = counter + 1
            counter += 2

        for v in node_list:
            if v == source or v == target:
                aux.add_edge(in_label[v], out_label[v], capacity=_INF_CAP)
            else:
                aux.add_edge(in_label[v], out_label[v], capacity=1)

        for u, v in graph.edges():
            aux.add_edge(out_label[u], in_label[v], capacity=_INF_CAP)
            aux.add_edge(out_label[v], in_label[u], capacity=_INF_CAP)

        flow_value, flow_dict = nx.maximum_flow(
            aux, out_label[source], in_label[target], capacity="capacity"
        )

        if flow_value == 0 or math.isinf(flow_value):
            self._cut_cache[key] = frozenset()
            return frozenset()

        residual = nx.DiGraph()
        for u_node in aux.nodes():
            residual.add_node(u_node)
        for u_node in flow_dict:
            for v_node, f in flow_dict[u_node].items():
                cap = aux[u_node][v_node].get("capacity", 0)
                res_forward = cap - f
                if res_forward > 1e-12:
                    residual.add_edge(u_node, v_node, capacity=res_forward)
                if f > 1e-12:
                    residual.add_edge(v_node, u_node, capacity=f)

        reachable: Set[int] = set()
        queue: deque[int] = deque([out_label[source]])
        reachable.add(out_label[source])
        while queue:
            cur = queue.popleft()
            for nbr in residual.successors(cur):
                if nbr not in reachable:
                    cap_edge = residual[cur][nbr].get("capacity", 0)
                    if cap_edge > 1e-12:
                        reachable.add(nbr)
                        queue.append(nbr)

        cut_vertices: Set[int] = set()
        for v in node_list:
            if v == source or v == target:
                continue
            if in_label[v] in reachable and out_label[v] not in reachable:
                cut_vertices.add(v)

        result = frozenset(cut_vertices)
        self._cut_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # 4. Balanced separator
    # ------------------------------------------------------------------

    def find_balanced_separator(
        self,
        graph: nx.Graph,
        weight: Optional[Dict[int, float]] = None,
    ) -> FrozenSet[int]:
        """Find a separator minimising the largest component weight.

        Uses betweenness centrality to rank vertices, then iteratively
        tries subsets of increasing size starting from the highest-
        centrality vertices.  Returns the best separator found.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n <= 1:
            return frozenset()

        if weight is None:
            weight = {v: 1.0 for v in nodes}

        total_weight = sum(weight.get(v, 1.0) for v in nodes)
        if total_weight <= 0:
            return frozenset()

        betweenness = self._cached_betweenness(graph)
        ranked = sorted(nodes, key=lambda v: betweenness.get(v, 0.0), reverse=True)

        best_sep: FrozenSet[int] = frozenset()
        best_score = 1.0

        max_sep_size = max(1, n // 3)

        for size in range(1, min(max_sep_size + 1, n)):
            top_candidates = ranked[: min(size * 3, n)]

            if size <= 5:
                subsets: Iterable[Tuple[int, ...]] = itertools.combinations(
                    top_candidates, size
                )
            else:
                subsets = self._greedy_separator_candidates(
                    graph, top_candidates, size, weight, num_candidates=10
                )

            for subset_tuple in subsets:
                sep = frozenset(subset_tuple)
                score = self._compute_balance_score(graph, sep, weight)
                if score < best_score:
                    best_score = score
                    best_sep = sep
                    if best_score <= (1.0 / 3.0) + 1e-9:
                        return best_sep

            if best_sep and best_score < 0.67:
                break

        return best_sep

    def _greedy_separator_candidates(
        self,
        graph: nx.Graph,
        candidates: List[int],
        size: int,
        weight: Dict[int, float],
        num_candidates: int = 10,
    ) -> List[Tuple[int, ...]]:
        """Generate candidate separators via greedy construction.

        Starting from each of the *num_candidates* highest-centrality
        vertices, greedily add the vertex that most improves the balance
        score until *size* vertices are selected.
        """
        results: List[Tuple[int, ...]] = []
        seeds = candidates[:num_candidates]

        for seed in seeds:
            current: List[int] = [seed]
            remaining = [v for v in candidates if v != seed]

            while len(current) < size and remaining:
                best_v: Optional[int] = None
                best_s = 2.0
                for v in remaining:
                    trial = frozenset(current + [v])
                    s = self._compute_balance_score(graph, trial, weight)
                    if s < best_s:
                        best_s = s
                        best_v = v
                if best_v is not None:
                    current.append(best_v)
                    remaining.remove(best_v)
                else:
                    break

            results.append(tuple(current))

        return results

    # ------------------------------------------------------------------
    # 5. Balance score
    # ------------------------------------------------------------------

    def _compute_balance_score(
        self,
        graph: nx.Graph,
        separator: FrozenSet[int],
        weight: Optional[Dict[int, float]],
    ) -> float:
        """Balance ratio: ``max_component_weight / total_weight``.

        Lower is better.  A perfectly balanced 2-way split yields 0.5.
        If removing the separator does not disconnect the graph the score
        is 1.0 (worst).
        """
        if weight is None:
            weight = {v: 1.0 for v in graph.nodes()}

        components = self._components_after_removal(graph, separator)

        if len(components) <= 1:
            return 1.0

        total_w = sum(weight.get(v, 1.0) for v in graph.nodes() if v not in separator)
        if total_w <= 0:
            return 1.0

        max_comp_w = 0.0
        for comp in components:
            comp_w = sum(weight.get(v, 1.0) for v in comp)
            if comp_w > max_comp_w:
                max_comp_w = comp_w

        return max_comp_w / total_w

    # ------------------------------------------------------------------
    # 6. Multi-criteria scoring
    # ------------------------------------------------------------------

    def score_separator(
        self,
        graph: nx.Graph,
        separator: FrozenSet[int],
        causal_dag: Optional[nx.DiGraph] = None,
    ) -> Dict[str, float]:
        """Score a separator on size, balance, and causal preservation.

        Returns a dict with keys ``"size_score"``, ``"balance_score"``,
        ``"causal_preservation"``, and ``"overall"``.  Each score is in
        [0, 1] where 1 is best.
        """
        cache_key = (separator, id(causal_dag) if causal_dag is not None else None)
        if cache_key in self._separator_score_cache:
            return dict(self._separator_score_cache[cache_key])

        n = graph.number_of_nodes()
        if n == 0:
            scores: Dict[str, float] = {
                "size_score": 1.0,
                "balance_score": 1.0,
                "causal_preservation": 1.0,
                "overall": 1.0,
            }
            self._separator_score_cache[cache_key] = scores
            return dict(scores)

        size_score = 1.0 - (len(separator) / n) if n > 0 else 1.0
        size_score = max(0.0, min(1.0, size_score))

        raw_balance = self._compute_balance_score(graph, separator, None)
        components = self._components_after_removal(graph, separator)
        if len(components) <= 1:
            balance_score = 0.0
        else:
            ideal = 1.0 / len(components)
            if raw_balance <= ideal + 1e-12:
                balance_score = 1.0
            else:
                balance_score = max(0.0, 1.0 - (raw_balance - ideal) / (1.0 - ideal))

        if causal_dag is not None and causal_dag.number_of_nodes() > 0:
            causal_pres = self._causal_preservation_score(causal_dag, separator)
        else:
            causal_pres = 1.0

        w_size = 0.3
        w_bal = 0.3
        w_causal = 0.4
        overall = w_size * size_score + w_bal * balance_score + w_causal * causal_pres

        scores = {
            "size_score": round(size_score, 6),
            "balance_score": round(balance_score, 6),
            "causal_preservation": round(causal_pres, 6),
            "overall": round(overall, 6),
        }
        self._separator_score_cache[cache_key] = scores
        return dict(scores)

    # ------------------------------------------------------------------
    # 7. Causal preservation
    # ------------------------------------------------------------------

    def _causal_preservation_score(
        self, dag: nx.DiGraph, separator: FrozenSet[int]
    ) -> float:
        """Fraction of source→target reachability pairs preserved.

        For every pair of nodes ``(s, t)`` with a directed path in *dag*
        we check whether the path survives after removing *separator*
        vertices.  The score is the fraction of pairs that remain
        connected.
        """
        nodes = [v for v in dag.nodes() if v not in separator]
        if len(nodes) <= 1:
            return 1.0

        sources = [v for v in dag.nodes() if dag.in_degree(v) == 0 and v not in separator]
        sinks = [v for v in dag.nodes() if dag.out_degree(v) == 0 and v not in separator]

        if not sources:
            sources = nodes[:max(1, len(nodes) // 4)]
        if not sinks:
            sinks = nodes[-(max(1, len(nodes) // 4)):]

        total_pairs = 0
        preserved = 0

        sub_nodes = set(dag.nodes()) - separator
        sub = dag.subgraph(sub_nodes)

        reachability: Dict[int, Set[int]] = {}
        for s in sources:
            if s not in sub:
                continue
            visited: Set[int] = set()
            stack: deque[int] = deque([s])
            while stack:
                cur = stack.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                for nxt in sub.successors(cur):
                    if nxt not in visited:
                        stack.append(nxt)
            reachability[s] = visited

        original_reachability: Dict[int, Set[int]] = {}
        for s in sources:
            if s not in dag:
                continue
            visited_orig: Set[int] = set()
            stack_orig: deque[int] = deque([s])
            while stack_orig:
                cur = stack_orig.popleft()
                if cur in visited_orig:
                    continue
                visited_orig.add(cur)
                for nxt in dag.successors(cur):
                    if nxt not in visited_orig:
                        stack_orig.append(nxt)
            original_reachability[s] = visited_orig

        for s in sources:
            orig_reach = original_reachability.get(s, set())
            sub_reach = reachability.get(s, set())
            for t in sinks:
                if t == s:
                    continue
                if t in orig_reach:
                    total_pairs += 1
                    if t in sub_reach:
                        preserved += 1

        if total_pairs == 0:
            return 1.0

        return preserved / total_pairs

    # ------------------------------------------------------------------
    # 8. Menger's theorem validation
    # ------------------------------------------------------------------

    def validate_separator_menger(
        self,
        graph: nx.Graph,
        separator: FrozenSet[int],
        source: int,
        target: int,
    ) -> bool:
        """Validate a separator using Menger's theorem.

        A set S is a minimum s-t separator iff
        1. Removing S disconnects s from t, **and**
        2. |S| equals the maximum number of internally vertex-disjoint
           s-t paths.

        Returns ``True`` when both conditions hold.
        """
        remaining_nodes = set(graph.nodes()) - separator
        if source not in remaining_nodes or target not in remaining_nodes:
            return False

        sub = graph.subgraph(remaining_nodes)
        if nx.has_path(sub, source, target):
            return False

        disjoint_paths = self._find_vertex_disjoint_paths(graph, source, target)
        num_disjoint = len(disjoint_paths)

        return num_disjoint == len(separator)

    # ------------------------------------------------------------------
    # 9. Safe separators (causal-aware)
    # ------------------------------------------------------------------

    def find_safe_separators(
        self,
        graph: nx.Graph,
        dag: nx.DiGraph,
        min_score: float = 0.8,
    ) -> List[Tuple[FrozenSet[int], Dict[str, float]]]:
        """Find separators scoring ≥ *min_score* on causal preservation.

        Candidates are obtained from minimal separators and from a
        betweenness-guided heuristic.  Each is scored and those above the
        threshold are returned sorted by overall quality descending.
        """
        candidates: Set[FrozenSet[int]] = set()

        minimal_seps = self.enumerate_minimal_separators(graph)
        for sep in minimal_seps:
            candidates.add(sep)

        n = graph.number_of_nodes()
        for target_size in range(1, max(2, n // 3)):
            guided = self._betweenness_guided_separator(graph, target_size)
            if guided:
                candidates.add(guided)

        scored: List[Tuple[FrozenSet[int], Dict[str, float]]] = []
        for sep in candidates:
            scores = self.score_separator(graph, sep, causal_dag=dag)
            if scores["causal_preservation"] >= min_score:
                scored.append((sep, scores))

        scored.sort(key=lambda x: x[1]["overall"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # 10. Vertex-disjoint paths (augmenting-path style)
    # ------------------------------------------------------------------

    def _find_vertex_disjoint_paths(
        self, graph: nx.Graph, source: int, target: int
    ) -> List[List[int]]:
        """Maximum set of internally vertex-disjoint s-t paths.

        Uses iterative augmentation on an auxiliary directed graph with
        vertex splitting (capacity-1 internal arcs).  Each augmenting
        path in the auxiliary graph corresponds to a new vertex-disjoint
        path (or a rerouting of existing ones).
        """
        key = (source, target)
        if key in self._disjoint_paths_cache:
            return list(self._disjoint_paths_cache[key])

        if source == target:
            self._disjoint_paths_cache[key] = [[source]]
            return [[source]]

        if source not in graph or target not in graph:
            self._disjoint_paths_cache[key] = []
            return []

        if not nx.has_path(graph, source, target):
            self._disjoint_paths_cache[key] = []
            return []

        nodes = list(graph.nodes())
        in_id: Dict[int, int] = {}
        out_id: Dict[int, int] = {}
        idx = 0
        for v in nodes:
            in_id[v] = idx
            out_id[v] = idx + 1
            idx += 2

        num_aux = idx
        capacity = np.zeros((num_aux, num_aux), dtype=np.int32)
        flow_matrix = np.zeros((num_aux, num_aux), dtype=np.int32)

        for v in nodes:
            if v == source or v == target:
                capacity[in_id[v], out_id[v]] = num_aux
            else:
                capacity[in_id[v], out_id[v]] = 1

        for u, v in graph.edges():
            capacity[out_id[u], in_id[v]] = num_aux
            capacity[out_id[v], in_id[u]] = num_aux

        s_node = out_id[source]
        t_node = in_id[target]

        while True:
            parent = [-1] * num_aux
            visited = [False] * num_aux
            visited[s_node] = True
            queue: deque[int] = deque([s_node])
            found = False

            while queue and not found:
                cur = queue.popleft()
                for nxt in range(num_aux):
                    if not visited[nxt] and capacity[cur][nxt] - flow_matrix[cur][nxt] > 0:
                        visited[nxt] = True
                        parent[nxt] = cur
                        if nxt == t_node:
                            found = True
                            break
                        queue.append(nxt)

            if not found:
                break

            v_node = t_node
            while v_node != s_node:
                u_node = parent[v_node]
                flow_matrix[u_node][v_node] += 1
                flow_matrix[v_node][u_node] -= 1
                v_node = u_node

        total_flow = int(np.sum(flow_matrix[s_node, :].clip(min=0)))

        id_to_vertex: Dict[int, int] = {}
        for v in nodes:
            id_to_vertex[out_id[v]] = v
            id_to_vertex[in_id[v]] = v

        succ: Dict[int, List[int]] = defaultdict(list)
        for i in range(num_aux):
            for j in range(num_aux):
                if flow_matrix[i][j] > 0:
                    succ[i].append(j)

        paths: List[List[int]] = []
        used_edges: Set[Tuple[int, int]] = set()

        for _ in range(total_flow):
            path_ids: List[int] = [s_node]
            cur = s_node
            visited_trace: Set[int] = {s_node}
            stuck = False

            while cur != t_node:
                moved = False
                for nxt in succ[cur]:
                    edge = (cur, nxt)
                    if edge not in used_edges and nxt not in visited_trace:
                        if flow_matrix[cur][nxt] > 0:
                            path_ids.append(nxt)
                            used_edges.add(edge)
                            visited_trace.add(nxt)
                            cur = nxt
                            moved = True
                            break
                if not moved:
                    stuck = True
                    break

            if stuck:
                continue

            path_vertices: List[int] = []
            seen_v: Set[int] = set()
            for nid in path_ids:
                v = id_to_vertex.get(nid)
                if v is not None and v not in seen_v:
                    path_vertices.append(v)
                    seen_v.add(v)

            if path_vertices and path_vertices[0] == source and path_vertices[-1] == target:
                paths.append(path_vertices)

        self._disjoint_paths_cache[key] = paths
        return list(paths)

    # ------------------------------------------------------------------
    # 11. Minimality check
    # ------------------------------------------------------------------

    def _is_minimal_separator(
        self,
        graph: nx.Graph,
        separator: FrozenSet[int],
        u: int,
        v: int,
    ) -> bool:
        """Check minimality: removing any single vertex reconnects u-v.

        A separator S between u and v is *minimal* iff for every vertex
        w in S the set S \\ {w} is **not** a u-v separator (i.e. u and v
        become connected again in G - (S \\ {w})).
        """
        if not separator:
            return False

        remaining_all = set(graph.nodes()) - separator
        if u not in remaining_all or v not in remaining_all:
            return False

        sub_all = graph.subgraph(remaining_all)
        if nx.has_path(sub_all, u, v):
            return False

        for w in separator:
            reduced = separator - {w}
            remaining = set(graph.nodes()) - reduced
            sub = graph.subgraph(remaining)
            if u in sub and v in sub and nx.has_path(sub, u, v):
                continue
            else:
                return False

        return True

    # ------------------------------------------------------------------
    # 12. Components after removal
    # ------------------------------------------------------------------

    def _components_after_removal(
        self, graph: nx.Graph, vertices: FrozenSet[int]
    ) -> List[Set[int]]:
        """Remove *vertices* and return connected components via BFS."""
        graph_id = id(graph)
        cache_key = (graph_id, vertices)
        if cache_key in self._component_cache:
            return [set(c) for c in self._component_cache[cache_key]]

        remaining = set(graph.nodes()) - vertices
        if not remaining:
            self._component_cache[cache_key] = []
            return []

        adj: Dict[int, List[int]] = defaultdict(list)
        for node in remaining:
            for nb in graph.neighbors(node):
                if nb in remaining:
                    adj[node].append(nb)

        visited: Set[int] = set()
        components: List[Set[int]] = []

        for start in remaining:
            if start in visited:
                continue
            comp: Set[int] = set()
            stack: deque[int] = deque([start])
            while stack:
                cur = stack.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        stack.append(nb)
            if comp:
                components.append(comp)

        self._component_cache[cache_key] = components
        return [set(c) for c in components]

    # ------------------------------------------------------------------
    # 13. Betweenness-guided separator
    # ------------------------------------------------------------------

    def _betweenness_guided_separator(
        self, graph: nx.Graph, target_size: int
    ) -> FrozenSet[int]:
        """Pick the *target_size* highest-betweenness vertices as separator.

        After the initial pick, iteratively try swapping each chosen
        vertex with a non-chosen neighbour if it improves the balance
        score, performing a simple local search.
        """
        if graph.number_of_nodes() == 0:
            return frozenset()

        betweenness = self._cached_betweenness(graph)
        ranked = sorted(
            graph.nodes(), key=lambda v: betweenness.get(v, 0.0), reverse=True
        )

        actual_size = min(target_size, len(ranked))
        if actual_size == 0:
            return frozenset()

        current = list(ranked[:actual_size])
        current_sep = frozenset(current)
        current_score = self._compute_balance_score(graph, current_sep, None)

        improved = True
        max_iter = actual_size * 5
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1

            for i in range(len(current)):
                v_old = current[i]
                neighbours_of_old = set(graph.neighbors(v_old)) - set(current)
                swap_candidates = list(neighbours_of_old)
                if not swap_candidates:
                    non_selected = [
                        u for u in ranked if u not in current
                    ]
                    swap_candidates = non_selected[: min(5, len(non_selected))]

                for v_new in swap_candidates:
                    trial = list(current)
                    trial[i] = v_new
                    trial_sep = frozenset(trial)
                    trial_score = self._compute_balance_score(graph, trial_sep, None)
                    if trial_score < current_score - 1e-9:
                        current = trial
                        current_sep = trial_sep
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break

        return current_sep

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cached_betweenness(self, graph: nx.Graph) -> Dict[int, float]:
        """Return betweenness centrality, caching per graph id."""
        gid = id(graph)
        if gid not in self._centrality_cache:
            if graph.number_of_nodes() == 0:
                self._centrality_cache[gid] = {}
            else:
                self._centrality_cache[gid] = nx.betweenness_centrality(graph)
        return self._centrality_cache[gid]
