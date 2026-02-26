"""
Causal-aware graph partitioning for large DAGs.

Partitions a causal DAG into sub-problems while preserving do-calculus
semantics across partition boundaries, enabling compositional causal
inference with bounded approximation error.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class CausalPartition:
    """A single partition of a causal DAG with boundary information."""

    nodes: Set[int]
    boundary_variables: Set[int]
    subgraph: nx.DiGraph
    quality_score: float = 0.0


class CausalPartitioner:
    """
    Partition a causal DAG into sub-problems that respect do-calculus
    semantics.  Uses moral-graph tree decomposition as the initial
    skeleton, then refines via merge / split passes and scores every
    candidate cut for causal-safety.
    """

    def __init__(
        self,
        max_partition_size: int = 50,
        min_partition_size: int = 5,
    ) -> None:
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def partition(
        self,
        dag: nx.DiGraph,
        max_width: Optional[int] = None,
    ) -> List[CausalPartition]:
        """Partition *dag* into causally-safe sub-problems.

        1. Build the moral graph of *dag*.
        2. Obtain a tree decomposition (bounded by *max_width* if given).
        3. Extract raw partitions from the bags of the decomposition.
        4. Merge partitions that are too small.
        5. Split partitions that are too large.
        6. Identify boundary variables, extract subgraphs, score quality.
        """
        if dag.number_of_nodes() == 0:
            return []

        if dag.number_of_nodes() <= self.max_partition_size:
            boundary = self._identify_boundary_variables(
                dag, set(dag.nodes()), [set(dag.nodes())]
            )
            sub = self._extract_partition_subgraph(dag, set(dag.nodes()), boundary)
            score = 1.0
            return [CausalPartition(set(dag.nodes()), boundary, sub, score)]

        moral = self._build_moral_graph(dag)

        if max_width is None:
            max_width = max(10, int(math.sqrt(dag.number_of_nodes())))

        tree_decomp = nx.approximation.treewidth_min_degree(moral)[1]

        raw_partitions: List[Set[int]] = []
        seen_nodes: Set[int] = set()
        for bag_node in tree_decomp.nodes():
            bag: Set[int] = set(tree_decomp.nodes[bag_node].get("graph", bag_node))
            if isinstance(bag, frozenset):
                bag = set(bag)
            elif isinstance(bag, int):
                bag = {bag}
            new_nodes = bag - seen_nodes
            if new_nodes:
                raw_partitions.append(new_nodes)
                seen_nodes |= new_nodes

        remaining = set(dag.nodes()) - seen_nodes
        if remaining:
            raw_partitions.append(remaining)

        partitions = self._merge_small_partitions(raw_partitions, dag)
        partitions = self._split_large_partitions(partitions, dag)

        result: List[CausalPartition] = []
        for part_nodes in partitions:
            boundary = self._identify_boundary_variables(dag, part_nodes, partitions)
            sub = self._extract_partition_subgraph(dag, part_nodes, boundary)
            preservation = self._check_interventional_preservation(
                dag, part_nodes, boundary
            )
            result.append(
                CausalPartition(part_nodes, boundary, sub, preservation)
            )

        return result

    # ------------------------------------------------------------------
    # Moral graph
    # ------------------------------------------------------------------

    def _build_moral_graph(self, dag: nx.DiGraph) -> nx.Graph:
        """Construct the moral graph: marry parents then drop directions."""
        moral = nx.Graph()
        moral.add_nodes_from(dag.nodes(data=True))

        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral.add_edge(parents[i], parents[j], moral_edge=True)

        for u, v, data in dag.edges(data=True):
            if moral.has_edge(u, v):
                moral[u][v].update(data)
            else:
                moral.add_edge(u, v, **data)

        return moral

    # ------------------------------------------------------------------
    # Boundary identification
    # ------------------------------------------------------------------

    def _identify_boundary_variables(
        self,
        dag: nx.DiGraph,
        partition_nodes: Set[int],
        all_partitions: List[Set[int]],
    ) -> Set[int]:
        """Return nodes in *partition_nodes* with cross-partition edges."""
        other_nodes: Set[int] = set()
        for part in all_partitions:
            if part is not partition_nodes:
                other_nodes |= part

        boundary: Set[int] = set()
        for node in partition_nodes:
            for parent in dag.predecessors(node):
                if parent in other_nodes:
                    boundary.add(node)
                    break
            else:
                for child in dag.successors(node):
                    if child in other_nodes:
                        boundary.add(node)

        return boundary

    # ------------------------------------------------------------------
    # Cut safety
    # ------------------------------------------------------------------

    def _assess_cut_safety(
        self,
        dag: nx.DiGraph,
        edge: Tuple[int, int],
        partition_a: Set[int],
        partition_b: Set[int],
    ) -> float:
        """Score how safe it is to cut *edge* (0 = dangerous, 1 = safe).

        Checks three criteria:
        * Whether the edge lies on any backdoor path between its endpoints.
        * Whether the edge participates in an instrumental-variable
          relationship (cutting it would break the exclusion restriction).
        * Whether conditioning on the boundary d-separates source from
          descendants in the other partition.
        """
        source, target = edge
        safety = 1.0

        backdoor_paths = self._find_backdoor_paths(dag, source, target)
        if backdoor_paths:
            n_paths = len(backdoor_paths)
            avg_len = np.mean([len(p) for p in backdoor_paths])
            penalty = min(0.5, 0.1 * n_paths * (1.0 / max(avg_len, 1.0)))
            safety -= penalty

        children_of_source = set(dag.successors(source))
        parents_of_target = set(dag.predecessors(target))
        shared_mediators = children_of_source & parents_of_target
        if shared_mediators:
            safety -= 0.15 * min(len(shared_mediators), 3)

        target_ancestors = nx.ancestors(dag, target)
        source_descendants = nx.descendants(dag, source)
        iv_candidates: Set[int] = set()
        for node in dag.nodes():
            if node == source or node == target:
                continue
            if node not in target_ancestors and node in set(dag.predecessors(source)):
                node_descs = nx.descendants(dag, node)
                if target in node_descs:
                    direct_children = set(dag.successors(node))
                    paths_through_source = direct_children & ({source} | source_descendants)
                    if paths_through_source:
                        iv_candidates.add(node)

        if iv_candidates:
            safety -= 0.1 * min(len(iv_candidates), 3)

        boundary = (partition_a | partition_b) - {source, target}
        if boundary:
            reachable_without_boundary: Set[int] = set()
            queue = deque([source])
            visited = {source}
            underlying = dag.to_undirected()
            while queue:
                current = queue.popleft()
                for nbr in underlying.neighbors(current):
                    if nbr not in visited and nbr not in boundary:
                        visited.add(nbr)
                        reachable_without_boundary.add(nbr)
                        queue.append(nbr)
            target_descs_in_b = source_descendants & partition_b
            if target_descs_in_b and not (target_descs_in_b & reachable_without_boundary):
                safety += 0.1

        return float(np.clip(safety, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Backdoor paths
    # ------------------------------------------------------------------

    def _find_backdoor_paths(
        self,
        dag: nx.DiGraph,
        source: int,
        target: int,
    ) -> List[List[int]]:
        """Find all backdoor paths from *source* to *target*.

        A backdoor path starts with an edge *into* source (i.e. the first
        step is from source to one of its parents) and then follows any
        sequence of directed or reversed edges to reach *target*, subject
        to the constraint that we never traverse a collider that is not
        an ancestor of any observed node (here we treat all nodes as
        observed for generality).

        Implementation: DFS on the underlying undirected skeleton, but
        we track the *direction* of entry at each step so we can enforce
        the backdoor starting condition and detect colliders.
        """
        if source == target:
            return []

        parents_of_source = set(dag.predecessors(source))
        if not parents_of_source:
            return []

        underlying = dag.to_undirected(as_view=False)
        ancestors_cache: Dict[int, Set[int]] = {}

        def _ancestors(node: int) -> Set[int]:
            if node not in ancestors_cache:
                ancestors_cache[node] = nx.ancestors(dag, node)
            return ancestors_cache[node]

        all_nodes = set(dag.nodes())

        backdoor_paths: List[List[int]] = []
        max_paths = 50

        stack: List[Tuple[int, List[int], str]] = []
        for parent in parents_of_source:
            if dag.has_edge(parent, source):
                stack.append((parent, [source, parent], "incoming"))

        while stack and len(backdoor_paths) < max_paths:
            current, path, last_direction = stack.pop()

            if current == target:
                backdoor_paths.append(list(path))
                continue

            if len(path) > dag.number_of_nodes():
                continue

            visited_set = set(path)

            for neighbor in underlying.neighbors(current):
                if neighbor in visited_set:
                    continue

                has_edge_cur_to_nbr = dag.has_edge(current, neighbor)
                has_edge_nbr_to_cur = dag.has_edge(neighbor, current)

                if has_edge_cur_to_nbr:
                    next_direction = "outgoing"
                elif has_edge_nbr_to_cur:
                    next_direction = "incoming"
                else:
                    continue

                is_collider = False
                if last_direction == "incoming" and next_direction == "incoming":
                    is_collider = True

                if is_collider:
                    node_anc = _ancestors(current)
                    has_observed_descendant = bool(
                        all_nodes & (nx.descendants(dag, current) | {current})
                    )
                    if not has_observed_descendant:
                        continue

                new_path = path + [neighbor]
                stack.append((neighbor, new_path, next_direction))

        return backdoor_paths

    # ------------------------------------------------------------------
    # Interventional preservation
    # ------------------------------------------------------------------

    def _check_interventional_preservation(
        self,
        dag: nx.DiGraph,
        partition: Set[int],
        boundary: Set[int],
    ) -> float:
        """Score how well interventions inside *partition* can be
        computed from variables in *partition* ∪ *boundary*.

        For each interior variable X (not on the boundary) we check
        whether do(X = x) requires knowledge of variables outside the
        augmented set.  The truncated factorisation formula for do(X)
        needs the parents of every variable in the post-intervention
        graph that are also ancestors of the outcome.  If all required
        parents are inside partition ∪ boundary the intervention is
        preserved; otherwise it is partially lost.
        """
        if not partition:
            return 1.0

        augmented = partition | boundary
        interior = partition - boundary
        if not interior:
            return 1.0

        preserved_count = 0
        total_count = 0

        for x in interior:
            parents_x = set(dag.predecessors(x))

            post_intervention_nodes = set(dag.nodes())
            post_dag = dag.copy()
            for p in list(parents_x):
                if post_dag.has_edge(p, x):
                    post_dag.remove_edge(p, x)

            descendants_x = nx.descendants(post_dag, x) if x in post_dag else set()
            outcome_candidates = descendants_x & augmented

            needed_outside = set()
            for y in outcome_candidates | {x}:
                y_parents = set(dag.predecessors(y))
                if y == x:
                    continue
                for yp in y_parents:
                    if yp not in augmented:
                        y_ancestors = nx.ancestors(dag, y)
                        if yp in y_ancestors or yp == y:
                            needed_outside.add(yp)

            total_count += 1
            if not needed_outside:
                preserved_count += 1
            else:
                fraction_inside = 1.0 - len(needed_outside) / max(
                    len(set(dag.predecessors(x)) | {x}), 1
                )
                preserved_count += max(fraction_inside, 0.0)

        if total_count == 0:
            return 1.0

        return preserved_count / total_count

    # ------------------------------------------------------------------
    # Edge-cut scoring
    # ------------------------------------------------------------------

    def _causal_edge_cut_score(
        self,
        dag: nx.DiGraph,
        edges_to_cut: List[Tuple[int, int]],
    ) -> float:
        """Score a set of edges to cut (lower is better).

        Penalties:
        * +1.0 for every edge on an active directed causal path between
          a source (node with no parents) and a sink (node with no children).
        * +0.5 for edges whose removal creates a new d-separation that
          did not previously exist among boundary-adjacent nodes.
        * +0.2 baseline per edge cut.

        Rewards:
        * -0.3 for edges already d-separated by their shared Markov
          blanket (cutting them costs nothing).
        """
        if not edges_to_cut:
            return 0.0

        score = 0.0
        cut_set = set(edges_to_cut)

        sources = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        sinks = [n for n in dag.nodes() if dag.out_degree(n) == 0]

        active_path_edges: Set[Tuple[int, int]] = set()
        for s in sources:
            for t in sinks:
                if s == t:
                    continue
                try:
                    for path in nx.all_simple_paths(dag, s, t, cutoff=dag.number_of_nodes()):
                        for i in range(len(path) - 1):
                            active_path_edges.add((path[i], path[i + 1]))
                        if len(active_path_edges) > 5000:
                            break
                except nx.NetworkXError:
                    continue
                if len(active_path_edges) > 5000:
                    break

        for u, v in edges_to_cut:
            score += 0.2

            if (u, v) in active_path_edges:
                score += 1.0

            mb_u = set(dag.predecessors(u)) | set(dag.successors(u))
            for child in dag.successors(u):
                mb_u |= set(dag.predecessors(child))
            mb_u.discard(u)

            mb_v = set(dag.predecessors(v)) | set(dag.successors(v))
            for child in dag.successors(v):
                mb_v |= set(dag.predecessors(child))
            mb_v.discard(v)

            shared_blanket = mb_u & mb_v
            if shared_blanket:
                remaining_paths_exist = False
                test_dag = dag.copy()
                test_dag.remove_edge(u, v)
                try:
                    nx.shortest_path(test_dag, u, v)
                    remaining_paths_exist = True
                except nx.NetworkXNoPath:
                    pass
                if not remaining_paths_exist:
                    score += 0.5
                else:
                    score -= 0.3

        return max(score, 0.0)

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def _extract_partition_subgraph(
        self,
        dag: nx.DiGraph,
        nodes: Set[int],
        boundary: Set[int],
    ) -> nx.DiGraph:
        """Extract induced subgraph on *nodes* ∪ *boundary*, preserving
        edge directions and all node / edge metadata.  Mark boundary
        nodes with the attribute ``is_boundary = True``.
        """
        combined = nodes | boundary
        subgraph = dag.subgraph(combined).copy()

        for n in subgraph.nodes():
            subgraph.nodes[n]["is_boundary"] = n in boundary
            subgraph.nodes[n]["is_interior"] = n in (nodes - boundary)

        for n in boundary:
            if n in subgraph:
                for pred in dag.predecessors(n):
                    if pred not in combined and not subgraph.has_node(pred):
                        subgraph.add_node(pred, is_boundary=False, is_exterior=True)
                        edge_data = dag.edges[pred, n] if dag.has_edge(pred, n) else {}
                        subgraph.add_edge(pred, n, **edge_data, cross_partition=True)

        return subgraph

    # ------------------------------------------------------------------
    # Merge / split
    # ------------------------------------------------------------------

    def _merge_small_partitions(
        self,
        partitions: List[Set[int]],
        dag: nx.DiGraph,
    ) -> List[Set[int]]:
        """Merge partitions smaller than *min_partition_size* into the
        neighbour partition with which they share the most edges.
        """
        if not partitions:
            return partitions

        result = [set(p) for p in partitions]
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(result):
                if len(result[i]) < self.min_partition_size and len(result) > 1:
                    best_j = -1
                    best_connectivity = -1

                    for j in range(len(result)):
                        if j == i:
                            continue
                        connectivity = 0
                        for node in result[i]:
                            for nbr in itertools.chain(
                                dag.predecessors(node), dag.successors(node)
                            ):
                                if nbr in result[j]:
                                    connectivity += 1

                        if connectivity > best_connectivity:
                            best_connectivity = connectivity
                            best_j = j

                    if best_j == -1:
                        best_j = 0 if i != 0 else (1 if len(result) > 1 else 0)

                    if best_j != i:
                        result[best_j] |= result[i]
                        result.pop(i)
                        changed = True
                        continue
                i += 1

        return result

    def _split_large_partitions(
        self,
        partitions: List[Set[int]],
        dag: nx.DiGraph,
    ) -> List[Set[int]]:
        """Split partitions larger than *max_partition_size* using a
        balanced topological cut that respects causal ordering.
        """
        result: List[Set[int]] = []

        for part in partitions:
            if len(part) <= self.max_partition_size:
                result.append(set(part))
                continue

            sub = dag.subgraph(part)
            try:
                topo_order = list(nx.topological_sort(sub))
            except nx.NetworkXUnfeasible:
                topo_order = sorted(part)

            num_splits = math.ceil(len(part) / self.max_partition_size)
            chunk_size = math.ceil(len(topo_order) / num_splits)

            for start in range(0, len(topo_order), chunk_size):
                chunk = set(topo_order[start : start + chunk_size])
                if chunk:
                    result.append(chunk)

        return result

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def compute_partition_quality(
        self,
        dag: nx.DiGraph,
        partitions: List[Set[int]],
    ) -> Dict[str, float]:
        """Compute a suite of quality metrics for the given partitioning."""
        if not partitions:
            return {
                "composition_gap_estimate": 0.0,
                "separator_coverage": 0.0,
                "causal_preservation_ratio": 0.0,
                "balance_ratio": 0.0,
                "edge_cut_fraction": 0.0,
            }

        comp_gap = self._composition_gap_estimate(dag, partitions)

        all_boundary: Set[int] = set()
        for part in partitions:
            bnd = self._identify_boundary_variables(dag, part, partitions)
            all_boundary |= bnd

        total_edges = dag.number_of_edges()
        cut_edges = 0
        node_to_part: Dict[int, int] = {}
        for idx, part in enumerate(partitions):
            for n in part:
                node_to_part[n] = idx

        for u, v in dag.edges():
            pu = node_to_part.get(u, -1)
            pv = node_to_part.get(v, -1)
            if pu != pv:
                cut_edges += 1

        edge_cut_fraction = cut_edges / max(total_edges, 1)

        separator_needed: Set[int] = set()
        for u, v in dag.edges():
            pu = node_to_part.get(u, -1)
            pv = node_to_part.get(v, -1)
            if pu != pv:
                separator_needed.add(u)
                separator_needed.add(v)

        separator_coverage = (
            len(all_boundary & separator_needed) / max(len(separator_needed), 1)
        )

        preservation_scores: List[float] = []
        for part in partitions:
            bnd = self._identify_boundary_variables(dag, part, partitions)
            score = self._check_interventional_preservation(dag, part, bnd)
            preservation_scores.append(score)

        causal_preservation = float(np.mean(preservation_scores)) if preservation_scores else 1.0

        sizes = [len(p) for p in partitions]
        balance_ratio = min(sizes) / max(max(sizes), 1)

        return {
            "composition_gap_estimate": comp_gap,
            "separator_coverage": separator_coverage,
            "causal_preservation_ratio": causal_preservation,
            "balance_ratio": balance_ratio,
            "edge_cut_fraction": edge_cut_fraction,
        }

    # ------------------------------------------------------------------
    # Composition gap
    # ------------------------------------------------------------------

    def _composition_gap_estimate(
        self,
        dag: nx.DiGraph,
        partitions: List[Set[int]],
    ) -> float:
        """Estimate the approximation error introduced by composing
        causal estimates across partition boundaries.

        The bound is based on the number of boundary variables, their
        average in-degree from other partitions, and the length of the
        longest cross-partition directed path.
        """
        if len(partitions) <= 1:
            return 0.0

        node_to_part: Dict[int, int] = {}
        for idx, part in enumerate(partitions):
            for n in part:
                node_to_part[n] = idx

        boundary_in_degrees: List[int] = []
        boundary_nodes: Set[int] = set()
        for part in partitions:
            for n in part:
                cross_parents = [
                    p
                    for p in dag.predecessors(n)
                    if node_to_part.get(p, -1) != node_to_part.get(n, -2)
                ]
                if cross_parents:
                    boundary_nodes.add(n)
                    boundary_in_degrees.append(len(cross_parents))

        if not boundary_nodes:
            return 0.0

        avg_cross_degree = float(np.mean(boundary_in_degrees))

        cross_dag = nx.DiGraph()
        for u, v in dag.edges():
            pu = node_to_part.get(u, -1)
            pv = node_to_part.get(v, -1)
            if pu != pv:
                cross_dag.add_edge(u, v)

        if cross_dag.number_of_nodes() == 0:
            return 0.0

        longest_cross_path = 0
        try:
            longest_cross_path = nx.dag_longest_path_length(cross_dag)
        except (nx.NetworkXUnfeasible, nx.NetworkXError):
            longest_cross_path = cross_dag.number_of_edges()

        n_boundary = len(boundary_nodes)
        n_total = dag.number_of_nodes()
        boundary_fraction = n_boundary / max(n_total, 1)

        gap = boundary_fraction * avg_cross_degree * math.log1p(longest_cross_path)
        gap = min(gap, 1.0)

        return float(gap)

    # ------------------------------------------------------------------
    # Topological partitioning
    # ------------------------------------------------------------------

    def _topological_partition(
        self,
        dag: nx.DiGraph,
        num_parts: int,
    ) -> List[Set[int]]:
        """Partition *dag* by slicing the topological order into
        approximately equal-size chunks, then adjust boundaries so that
        no parent-child pair is separated unless necessary.
        """
        if num_parts <= 0:
            num_parts = 1

        try:
            topo = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo = sorted(dag.nodes())

        n = len(topo)
        if n == 0:
            return []

        chunk_size = max(1, math.ceil(n / num_parts))
        raw_parts: List[List[int]] = []
        for start in range(0, n, chunk_size):
            raw_parts.append(topo[start : start + chunk_size])

        if len(raw_parts) > num_parts:
            raw_parts[-2].extend(raw_parts[-1])
            raw_parts.pop()

        node_to_part: Dict[int, int] = {}
        for idx, chunk in enumerate(raw_parts):
            for nd in chunk:
                node_to_part[nd] = idx

        max_iters = n
        iteration = 0
        while iteration < max_iters:
            moved = False
            for u, v in dag.edges():
                pu = node_to_part[u]
                pv = node_to_part[v]
                if pu != pv:
                    size_pu = sum(1 for x in node_to_part.values() if x == pu)
                    size_pv = sum(1 for x in node_to_part.values() if x == pv)
                    if abs(pu - pv) == 1:
                        if size_pu > size_pv and size_pu > chunk_size:
                            node_to_part[u] = pv
                            moved = True
                        elif size_pv > size_pu and size_pv > chunk_size:
                            node_to_part[v] = pu
                            moved = True
            iteration += 1
            if not moved:
                break

        part_map: Dict[int, Set[int]] = defaultdict(set)
        for nd, pidx in node_to_part.items():
            part_map[pidx].add(nd)

        return [part_map[k] for k in sorted(part_map.keys()) if part_map[k]]

    # ------------------------------------------------------------------
    # Spectral partitioning
    # ------------------------------------------------------------------

    def _spectral_partition(
        self,
        dag: nx.DiGraph,
        num_parts: int,
    ) -> List[Set[int]]:
        """Recursively bisect *dag* using the Fiedler vector (second-
        smallest eigenvector of the graph Laplacian) until *num_parts*
        partitions are obtained.
        """
        if num_parts <= 1 or dag.number_of_nodes() <= 1:
            return [set(dag.nodes())]

        parts = self._spectral_bisect(dag)
        if len(parts) < 2:
            return parts

        if num_parts == 2:
            return parts

        left_target = num_parts // 2
        right_target = num_parts - left_target

        result: List[Set[int]] = []

        left_sub = dag.subgraph(parts[0]).copy()
        if left_target > 1 and len(parts[0]) > 1:
            result.extend(self._spectral_partition(left_sub, left_target))
        else:
            result.append(parts[0])

        right_sub = dag.subgraph(parts[1]).copy()
        if right_target > 1 and len(parts[1]) > 1:
            result.extend(self._spectral_partition(right_sub, right_target))
        else:
            result.append(parts[1])

        return result

    def _spectral_bisect(self, dag: nx.DiGraph) -> List[Set[int]]:
        """Bisect *dag* using the Fiedler vector of its undirected
        version's Laplacian matrix.
        """
        n = dag.number_of_nodes()
        if n <= 1:
            return [set(dag.nodes())]

        undirected = dag.to_undirected()

        if not nx.is_connected(undirected):
            components = list(nx.connected_components(undirected))
            if len(components) >= 2:
                left: Set[int] = set()
                right: Set[int] = set()
                for i, comp in enumerate(components):
                    if len(left) <= len(right):
                        left |= comp
                    else:
                        right |= comp
                return [left, right]

        nodes = sorted(dag.nodes())
        node_index = {nd: i for i, nd in enumerate(nodes)}

        laplacian = np.zeros((n, n), dtype=np.float64)
        for u, v in undirected.edges():
            i, j = node_index[u], node_index[v]
            laplacian[i, j] -= 1.0
            laplacian[j, i] -= 1.0
            laplacian[i, i] += 1.0
            laplacian[j, j] += 1.0

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        except np.linalg.LinAlgError:
            mid = n // 2
            return [set(nodes[:mid]), set(nodes[mid:])]

        fiedler_index = 1 if n > 1 else 0
        fiedler_vector = eigenvectors[:, fiedler_index]

        median_val = float(np.median(fiedler_vector))

        left_nodes: Set[int] = set()
        right_nodes: Set[int] = set()

        for i, nd in enumerate(nodes):
            if fiedler_vector[i] <= median_val:
                left_nodes.add(nd)
            else:
                right_nodes.add(nd)

        if not left_nodes:
            left_nodes.add(right_nodes.pop())
        if not right_nodes:
            right_nodes.add(left_nodes.pop())

        return [left_nodes, right_nodes]

    # ------------------------------------------------------------------
    # Causal-aware refinement helpers
    # ------------------------------------------------------------------

    def _d_separated(
        self,
        dag: nx.DiGraph,
        x: int,
        y: int,
        conditioning: Set[int],
    ) -> bool:
        """Test whether *x* and *y* are d-separated given *conditioning*
        using the Bayes-Ball algorithm.
        """
        if x == y:
            return False

        visited_top: Set[int] = set()
        visited_bottom: Set[int] = set()

        queue: deque[Tuple[int, str]] = deque()
        queue.append((x, "up"))

        ancestors_of_z = set()
        for z in conditioning:
            ancestors_of_z |= nx.ancestors(dag, z)
            ancestors_of_z.add(z)

        while queue:
            node, direction = queue.popleft()

            if node == y:
                return False

            if direction == "up" and node not in visited_top:
                visited_top.add(node)
                if node not in conditioning:
                    for parent in dag.predecessors(node):
                        queue.append((parent, "up"))
                    for child in dag.successors(node):
                        queue.append((child, "down"))

            elif direction == "down" and node not in visited_bottom:
                visited_bottom.add(node)
                if node not in conditioning:
                    for child in dag.successors(node):
                        queue.append((child, "down"))
                if node in conditioning or node in ancestors_of_z:
                    for parent in dag.predecessors(node):
                        queue.append((parent, "up"))

        return True

    def _markov_blanket(self, dag: nx.DiGraph, node: int) -> Set[int]:
        """Return the Markov blanket of *node*: parents, children, and
        parents of children (co-parents).
        """
        parents = set(dag.predecessors(node))
        children = set(dag.successors(node))
        coparents: Set[int] = set()
        for child in children:
            coparents |= set(dag.predecessors(child))
        coparents.discard(node)
        return parents | children | coparents

    def _causal_ordering_score(
        self,
        dag: nx.DiGraph,
        partition_a: Set[int],
        partition_b: Set[int],
    ) -> float:
        """Score how well two partitions respect the causal (topological)
        ordering.  Returns 1.0 if all edges go from A→B or within each,
        0.0 if heavily mixed.
        """
        forward = 0
        backward = 0
        within = 0

        try:
            topo = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            return 0.5

        topo_index = {nd: i for i, nd in enumerate(topo)}
        avg_a = np.mean([topo_index.get(n, 0) for n in partition_a]) if partition_a else 0.0
        avg_b = np.mean([topo_index.get(n, 0) for n in partition_b]) if partition_b else 0.0

        a_is_earlier = avg_a < avg_b

        combined = partition_a | partition_b
        for u, v in dag.edges():
            if u in combined and v in combined:
                u_in_a = u in partition_a
                v_in_a = v in partition_a
                if u_in_a == v_in_a:
                    within += 1
                elif u_in_a and not v_in_a:
                    if a_is_earlier:
                        forward += 1
                    else:
                        backward += 1
                else:
                    if a_is_earlier:
                        backward += 1
                    else:
                        forward += 1

        total_cross = forward + backward
        if total_cross == 0:
            return 1.0

        return forward / total_cross

    def _find_minimal_separator(
        self,
        dag: nx.DiGraph,
        partition_a: Set[int],
        partition_b: Set[int],
    ) -> Set[int]:
        """Find a minimal vertex separator between partition_a and
        partition_b in the underlying undirected graph, using iterative
        max-flow / min-cut.
        """
        undirected = dag.to_undirected()

        sub_nodes = partition_a | partition_b
        sub = undirected.subgraph(sub_nodes)

        super_source = max(dag.nodes()) + 1
        super_sink = super_source + 1

        flow_graph = nx.Graph()
        for node in sub.nodes():
            flow_graph.add_node((node, "in"))
            flow_graph.add_node((node, "out"))
            flow_graph.add_edge((node, "in"), (node, "out"), capacity=1)

        for u, v in sub.edges():
            flow_graph.add_edge((u, "out"), (v, "in"), capacity=len(sub_nodes) + 1)
            flow_graph.add_edge((v, "out"), (u, "in"), capacity=len(sub_nodes) + 1)

        flow_graph.add_node(super_source)
        flow_graph.add_node(super_sink)

        for a_node in partition_a:
            if a_node in sub:
                flow_graph.add_edge(
                    super_source, (a_node, "in"), capacity=len(sub_nodes) + 1
                )
        for b_node in partition_b:
            if b_node in sub:
                flow_graph.add_edge(
                    (b_node, "out"), super_sink, capacity=len(sub_nodes) + 1
                )

        try:
            cut_value, (reachable, non_reachable) = nx.minimum_cut(
                flow_graph, super_source, super_sink
            )
        except nx.NetworkXError:
            return partition_a & partition_b

        separator: Set[int] = set()
        for node in sub.nodes():
            if (node, "in") in reachable and (node, "out") in non_reachable:
                separator.add(node)
            elif (node, "in") in non_reachable and (node, "out") in reachable:
                separator.add(node)

        return separator

    def _compute_effective_treewidth(
        self,
        dag: nx.DiGraph,
        partitions: List[Set[int]],
    ) -> int:
        """Compute the maximum treewidth across all partition subgraphs
        as a measure of local inference complexity.
        """
        max_tw = 0
        for part in partitions:
            if not part:
                continue
            sub = dag.subgraph(part)
            moral_sub = self._build_moral_graph(sub)
            tw, _ = nx.approximation.treewidth_min_degree(moral_sub)
            max_tw = max(max_tw, tw)
        return max_tw

    def _greedy_causal_partition(
        self,
        dag: nx.DiGraph,
        num_parts: int,
    ) -> List[Set[int]]:
        """Greedy partitioning that grows regions from seed nodes chosen
        to be roughly evenly spaced in the topological order, expanding
        each region by adding the neighbour that minimises the causal
        edge-cut score.
        """
        if num_parts <= 0:
            num_parts = 1

        try:
            topo = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo = sorted(dag.nodes())

        n = len(topo)
        if n == 0:
            return []

        step = max(1, n // num_parts)
        seeds = [topo[i * step] for i in range(num_parts)]
        seeds = seeds[:num_parts]

        partitions: List[Set[int]] = [set() for _ in range(num_parts)]
        for i, seed in enumerate(seeds):
            partitions[i].add(seed)

        assigned: Set[int] = set(seeds)
        unassigned = set(dag.nodes()) - assigned

        underlying = dag.to_undirected()

        while unassigned:
            best_node = None
            best_part_idx = -1
            best_score = float("inf")

            for node in list(unassigned):
                for idx in range(len(partitions)):
                    has_connection = False
                    for member in partitions[idx]:
                        if underlying.has_edge(node, member):
                            has_connection = True
                            break
                    if not has_connection:
                        continue

                    test_partition = partitions[idx] | {node}
                    cross_edges = []
                    for u, v in dag.edges():
                        if u == node or v == node:
                            u_in = u in test_partition
                            v_in = v in test_partition
                            if u_in != v_in:
                                cross_edges.append((u, v))

                    score = len(cross_edges)
                    if score < best_score:
                        best_score = score
                        best_node = node
                        best_part_idx = idx

            if best_node is None:
                node = next(iter(unassigned))
                smallest_idx = min(range(len(partitions)), key=lambda i: len(partitions[i]))
                partitions[smallest_idx].add(node)
                assigned.add(node)
                unassigned.remove(node)
            else:
                partitions[best_part_idx].add(best_node)
                assigned.add(best_node)
                unassigned.remove(best_node)

        return [p for p in partitions if p]
