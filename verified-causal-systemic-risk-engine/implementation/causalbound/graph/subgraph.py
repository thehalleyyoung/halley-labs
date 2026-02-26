"""
Subgraph extraction, merging, splitting, and validation utilities
for causal graph decomposition in systemic risk analysis.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


class SubgraphExtractor:
    """Handles extraction, manipulation, and validation of subgraphs
    within larger causal or risk-network graphs.  Provides spectral
    bisection, overlap analysis, merge/split operations, and metadata
    preservation."""

    def __init__(self) -> None:
        self._extraction_cache: Dict[frozenset, nx.Graph] = {}
        self._overlap_cache: Dict[Tuple[int, int], int] = {}
        self._node_to_subgraph_map: Dict[int, List[int]] = {}
        self._merge_history: List[Tuple[int, int]] = []
        self._statistics_dirty: bool = True

    # ------------------------------------------------------------------
    # 1. Induced subgraph with optional boundary
    # ------------------------------------------------------------------
    def extract_induced_subgraph(
        self,
        graph: nx.Graph,
        nodes: Set[int],
        include_boundary: bool = True,
    ) -> nx.Graph:
        """Extract the induced subgraph on *nodes*.  When
        *include_boundary* is True the immediate neighbours of the node
        set that are **not** already in *nodes* are added and tagged with
        ``is_boundary=True``."""
        core_nodes = nodes & set(graph.nodes())
        subgraph = graph.subgraph(core_nodes).copy()

        for node in core_nodes:
            subgraph.nodes[node]["is_boundary"] = False
            subgraph.nodes[node]["is_interface"] = False

        if include_boundary:
            boundary_nodes: Set[int] = set()
            for node in core_nodes:
                for neighbour in graph.neighbors(node):
                    if neighbour not in core_nodes:
                        boundary_nodes.add(neighbour)

            for bnode in boundary_nodes:
                subgraph.add_node(bnode, **graph.nodes[bnode])
                subgraph.nodes[bnode]["is_boundary"] = True
                subgraph.nodes[bnode]["is_interface"] = False

                for neighbour in graph.neighbors(bnode):
                    if neighbour in core_nodes:
                        edge_data = graph.edges[bnode, neighbour]
                        subgraph.add_edge(bnode, neighbour, **edge_data)

        cache_key = frozenset(subgraph.nodes())
        self._extraction_cache[cache_key] = subgraph
        self._statistics_dirty = True
        return subgraph

    # ------------------------------------------------------------------
    # 2. Extract with explicit boundary
    # ------------------------------------------------------------------
    def extract_with_boundary(
        self,
        graph: nx.Graph,
        nodes: Set[int],
        boundary: Set[int],
    ) -> nx.Graph:
        """Extract subgraph containing *nodes* (core) and *boundary*
        nodes.  Boundary nodes are marked ``is_boundary=True``.  All
        edge attributes are preserved."""
        all_nodes = (nodes | boundary) & set(graph.nodes())
        subgraph = graph.subgraph(all_nodes).copy()

        for node in subgraph.nodes():
            if node in boundary:
                subgraph.nodes[node]["is_boundary"] = True
            else:
                subgraph.nodes[node]["is_boundary"] = False
            subgraph.nodes[node]["is_interface"] = False

        for u, v in subgraph.edges():
            if graph.has_edge(u, v):
                for attr_key, attr_val in graph.edges[u, v].items():
                    subgraph.edges[u, v][attr_key] = attr_val

        self._statistics_dirty = True
        return subgraph

    # ------------------------------------------------------------------
    # 3. Add interface nodes
    # ------------------------------------------------------------------
    def add_interface_nodes(
        self,
        subgraph: nx.Graph,
        boundary_vars: Set[int],
        original_graph: nx.Graph,
    ) -> nx.Graph:
        """Add interface (boundary) variables to *subgraph* that come
        from *original_graph*.  Each interface node is connected to any
        existing subgraph node it is adjacent to in the original graph.
        The attribute ``is_interface`` is set to ``True``."""
        result = subgraph.copy()
        existing_nodes = set(result.nodes())

        for bvar in boundary_vars:
            if bvar not in original_graph:
                continue
            if bvar in existing_nodes:
                result.nodes[bvar]["is_interface"] = True
                continue

            node_attrs = dict(original_graph.nodes[bvar])
            node_attrs["is_interface"] = True
            node_attrs["is_boundary"] = True
            result.add_node(bvar, **node_attrs)

            for neighbour in original_graph.neighbors(bvar):
                if neighbour in existing_nodes:
                    edge_data = dict(original_graph.edges[bvar, neighbour])
                    result.add_edge(bvar, neighbour, **edge_data)

        self._statistics_dirty = True
        return result

    # ------------------------------------------------------------------
    # 4. Overlap graph (subgraph-level)
    # ------------------------------------------------------------------
    def compute_overlap_structure(
        self, subgraphs: List[nx.Graph]
    ) -> nx.Graph:
        """Build a meta-graph where each node represents a subgraph and
        edges are weighted by the number of vertices shared between the
        two subgraphs."""
        n = len(subgraphs)
        node_sets = [set(sg.nodes()) for sg in subgraphs]

        overlap_graph = nx.Graph()
        for i in range(n):
            overlap_graph.add_node(
                i,
                size=len(node_sets[i]),
                label=f"subgraph_{i}",
            )

        for i in range(n):
            for j in range(i + 1, n):
                shared = node_sets[i] & node_sets[j]
                overlap_size = len(shared)
                if overlap_size > 0:
                    jaccard = overlap_size / len(node_sets[i] | node_sets[j])
                    overlap_graph.add_edge(
                        i,
                        j,
                        weight=overlap_size,
                        shared_nodes=list(shared),
                        jaccard=jaccard,
                    )
                self._overlap_cache[(i, j)] = overlap_size
                self._overlap_cache[(j, i)] = overlap_size

        return overlap_graph

    # ------------------------------------------------------------------
    # 5. Pairwise overlap matrix
    # ------------------------------------------------------------------
    def compute_overlap_matrix(
        self, subgraphs: List[nx.Graph]
    ) -> np.ndarray:
        """Return an (n × n) matrix where entry (i, j) equals
        |V_i ∩ V_j|."""
        n = len(subgraphs)
        node_sets = [set(sg.nodes()) for sg in subgraphs]
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            matrix[i, i] = float(len(node_sets[i]))
            for j in range(i + 1, n):
                overlap = float(len(node_sets[i] & node_sets[j]))
                matrix[i, j] = overlap
                matrix[j, i] = overlap

        return matrix

    # ------------------------------------------------------------------
    # 6. Merge selected subgraphs
    # ------------------------------------------------------------------
    def merge_subgraphs(
        self,
        subgraphs: List[nx.Graph],
        indices: List[int],
    ) -> nx.Graph:
        """Merge the subgraphs at *indices* into a single graph.  The
        union of nodes and edges is taken; boundary nodes that appear as
        internal nodes in **any** constituent lose their boundary status."""
        merged = nx.Graph()

        internal_nodes: Set[int] = set()
        boundary_nodes: Set[int] = set()

        for idx in indices:
            sg = subgraphs[idx]
            for node, attrs in sg.nodes(data=True):
                if node not in merged:
                    merged.add_node(node, **attrs)
                else:
                    for k, v in attrs.items():
                        if k not in ("is_boundary", "is_interface"):
                            merged.nodes[node][k] = v

                if attrs.get("is_boundary", False):
                    boundary_nodes.add(node)
                else:
                    internal_nodes.add(node)

            for u, v, attrs in sg.edges(data=True):
                if merged.has_edge(u, v):
                    for k, val in attrs.items():
                        merged.edges[u, v][k] = val
                else:
                    merged.add_edge(u, v, **attrs)

        for node in internal_nodes:
            if node in merged:
                merged.nodes[node]["is_boundary"] = False

        pure_boundary = boundary_nodes - internal_nodes
        for node in pure_boundary:
            if node in merged:
                merged.nodes[node]["is_boundary"] = True

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                self._merge_history.append((indices[i], indices[j]))

        self._statistics_dirty = True
        return merged

    # ------------------------------------------------------------------
    # 7. Merge small subgraphs iteratively
    # ------------------------------------------------------------------
    def merge_small_subgraphs(
        self,
        subgraphs: List[nx.Graph],
        min_size: int,
        original_graph: nx.Graph,
    ) -> List[nx.Graph]:
        """Iteratively merge subgraphs smaller than *min_size* with their
        most-overlapping neighbour until no subgraph is below the
        threshold (or no further merges are possible)."""
        working = [sg.copy() for sg in subgraphs]
        changed = True

        while changed:
            changed = False
            sizes = [sg.number_of_nodes() for sg in working]

            small_indices = [
                i for i, s in enumerate(sizes) if s < min_size
            ]
            if not small_indices:
                break

            small_indices.sort(key=lambda i: sizes[i])

            merged_away: Set[int] = set()
            new_graphs: List[nx.Graph] = []

            for idx in small_indices:
                if idx in merged_away:
                    continue
                partner = self._find_best_merge_partner(working, idx)
                if partner == -1 or partner in merged_away:
                    continue

                merged = self.merge_subgraphs(working, [idx, partner])
                merged = self.maintain_edge_metadata(original_graph, merged)

                merged_away.add(idx)
                merged_away.add(partner)
                new_graphs.append(merged)
                changed = True

            remaining = [
                sg for i, sg in enumerate(working) if i not in merged_away
            ]
            working = remaining + new_graphs

        self._statistics_dirty = True
        return working

    # ------------------------------------------------------------------
    # 8. Split a large subgraph via spectral bisection
    # ------------------------------------------------------------------
    def split_subgraph(
        self,
        subgraph: nx.Graph,
        max_size: int,
        original_graph: nx.Graph,
    ) -> List[nx.Graph]:
        """Recursively split *subgraph* using spectral bisection until
        every resulting piece has at most *max_size* core (non-boundary)
        nodes."""
        core_nodes = set(
            n
            for n, d in subgraph.nodes(data=True)
            if not d.get("is_boundary", False)
        )
        if not core_nodes:
            core_nodes = set(subgraph.nodes())

        if len(core_nodes) <= max_size:
            return [subgraph.copy()]

        if len(core_nodes) <= 1:
            return [subgraph.copy()]

        core_sg = subgraph.subgraph(core_nodes).copy()

        components = list(nx.connected_components(core_sg))
        if len(components) > 1:
            results: List[nx.Graph] = []
            for comp_nodes in components:
                comp_all = set(comp_nodes)
                for cn in comp_nodes:
                    for nbr in subgraph.neighbors(cn):
                        if nbr not in core_nodes:
                            comp_all.add(nbr)
                piece = subgraph.subgraph(comp_all).copy()
                for n in piece.nodes():
                    if n not in comp_nodes:
                        piece.nodes[n]["is_boundary"] = True
                results.extend(
                    self.split_subgraph(piece, max_size, original_graph)
                )
            return results

        left_nodes, right_nodes = self._spectral_bisection(core_sg)

        if len(left_nodes) == 0 or len(right_nodes) == 0:
            return [subgraph.copy()]

        left_boundary: Set[int] = set()
        right_boundary: Set[int] = set()
        for u, v in core_sg.edges():
            if u in left_nodes and v in right_nodes:
                left_boundary.add(v)
                right_boundary.add(u)
            elif u in right_nodes and v in left_nodes:
                left_boundary.add(u)
                right_boundary.add(v)

        left_sg = self.extract_with_boundary(
            subgraph, left_nodes, left_boundary
        )
        right_sg = self.extract_with_boundary(
            subgraph, right_nodes, right_boundary
        )

        left_sg = self.maintain_edge_metadata(original_graph, left_sg)
        right_sg = self.maintain_edge_metadata(original_graph, right_sg)

        results = []
        results.extend(
            self.split_subgraph(left_sg, max_size, original_graph)
        )
        results.extend(
            self.split_subgraph(right_sg, max_size, original_graph)
        )
        self._statistics_dirty = True
        return results

    # ------------------------------------------------------------------
    # 9. Spectral bisection (Fiedler vector)
    # ------------------------------------------------------------------
    def _spectral_bisection(
        self, graph: nx.Graph
    ) -> Tuple[Set[int], Set[int]]:
        """Bisect *graph* using the Fiedler vector (the eigenvector
        corresponding to the second-smallest eigenvalue of the graph
        Laplacian).  Nodes with Fiedler-value < median go left,
        the rest go right."""
        nodes = list(graph.nodes())
        n = len(nodes)
        if n <= 1:
            return set(nodes), set()

        node_index = {node: i for i, node in enumerate(nodes)}

        laplacian = np.zeros((n, n), dtype=np.float64)
        for u, v in graph.edges():
            i_u = node_index[u]
            i_v = node_index[v]
            weight = graph.edges[u, v].get("weight", 1.0)
            laplacian[i_u, i_v] -= weight
            laplacian[i_v, i_u] -= weight
            laplacian[i_u, i_u] += weight
            laplacian[i_v, i_v] += weight

        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        sorted_indices = np.argsort(eigenvalues)
        if len(sorted_indices) < 2:
            mid = n // 2
            return set(nodes[:mid]), set(nodes[mid:])

        fiedler_index = sorted_indices[1]
        fiedler_vector = eigenvectors[:, fiedler_index]

        median_val = float(np.median(fiedler_vector))

        left: Set[int] = set()
        right: Set[int] = set()
        for i, node in enumerate(nodes):
            if fiedler_vector[i] < median_val:
                left.add(node)
            else:
                right.add(node)

        if len(left) == 0:
            first_node = nodes[0]
            right.discard(first_node)
            left.add(first_node)
        elif len(right) == 0:
            first_node = nodes[0]
            left.discard(first_node)
            right.add(first_node)

        return left, right

    # ------------------------------------------------------------------
    # 10. Subgraph statistics
    # ------------------------------------------------------------------
    def compute_statistics(
        self, subgraphs: List[nx.Graph]
    ) -> Dict[str, Any]:
        """Compute summary statistics across a collection of subgraphs."""
        n = len(subgraphs)
        if n == 0:
            return {
                "num_subgraphs": 0,
                "sizes": {"min": 0, "max": 0, "mean": 0.0, "std": 0.0},
                "total_nodes": 0,
                "total_boundary_nodes": 0,
                "overlap_density": 0.0,
                "edge_counts": {"min": 0, "max": 0, "mean": 0.0, "std": 0.0},
                "boundary_ratios": [],
            }

        sizes = np.array(
            [sg.number_of_nodes() for sg in subgraphs], dtype=np.float64
        )
        edge_counts = np.array(
            [sg.number_of_edges() for sg in subgraphs], dtype=np.float64
        )

        all_nodes: Set[int] = set()
        total_boundary = 0
        boundary_ratios: List[float] = []

        for sg in subgraphs:
            sg_nodes = set(sg.nodes())
            all_nodes |= sg_nodes

            boundary_count = sum(
                1
                for _, attrs in sg.nodes(data=True)
                if attrs.get("is_boundary", False)
            )
            total_boundary += boundary_count
            ratio = boundary_count / max(len(sg_nodes), 1)
            boundary_ratios.append(ratio)

        overlap_matrix = self.compute_overlap_matrix(subgraphs)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(overlap_matrix[i, j])

        if len(upper_triangle) > 0:
            max_possible_pairs = n * (n - 1) / 2
            nonzero_pairs = sum(1 for v in upper_triangle if v > 0)
            overlap_density = nonzero_pairs / max(max_possible_pairs, 1)
        else:
            overlap_density = 0.0

        self._statistics_dirty = False
        return {
            "num_subgraphs": n,
            "sizes": {
                "min": int(np.min(sizes)),
                "max": int(np.max(sizes)),
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
            },
            "total_nodes": len(all_nodes),
            "total_boundary_nodes": total_boundary,
            "overlap_density": overlap_density,
            "edge_counts": {
                "min": int(np.min(edge_counts)),
                "max": int(np.max(edge_counts)),
                "mean": float(np.mean(edge_counts)),
                "std": float(np.std(edge_counts)),
            },
            "boundary_ratios": boundary_ratios,
        }

    # ------------------------------------------------------------------
    # 11. Validation
    # ------------------------------------------------------------------
    def validate_extraction(
        self,
        original_graph: nx.Graph,
        subgraphs: List[nx.Graph],
    ) -> Dict[str, bool]:
        """Validate that the subgraph collection faithfully covers the
        original graph in terms of nodes, edges, and boundary
        consistency."""
        covered_nodes: Set[int] = set()
        covered_edges: Set[Tuple[int, int]] = set()

        boundary_consistent = True

        for sg in subgraphs:
            sg_nodes = set(sg.nodes())
            covered_nodes |= sg_nodes

            for u, v in sg.edges():
                edge = (min(u, v), max(u, v))
                covered_edges.add(edge)

            for node, attrs in sg.nodes(data=True):
                is_bnd = attrs.get("is_boundary", False)
                if is_bnd:
                    if node not in original_graph:
                        boundary_consistent = False
                        break

        original_nodes = set(original_graph.nodes())
        original_edges: Set[Tuple[int, int]] = set()
        for u, v in original_graph.edges():
            original_edges.add((min(u, v), max(u, v)))

        non_boundary_covered = set()
        for sg in subgraphs:
            for node, attrs in sg.nodes(data=True):
                if not attrs.get("is_boundary", False):
                    non_boundary_covered.add(node)

        all_nodes_covered = original_nodes.issubset(non_boundary_covered)
        all_edges_covered = original_edges.issubset(covered_edges)

        for sg in subgraphs:
            for node, attrs in sg.nodes(data=True):
                if attrs.get("is_boundary", False):
                    has_internal_neighbour = False
                    for nbr in sg.neighbors(node):
                        nbr_attrs = sg.nodes[nbr]
                        if not nbr_attrs.get("is_boundary", False):
                            has_internal_neighbour = True
                            break
                    if not has_internal_neighbour and sg.degree(node) > 0:
                        boundary_consistent = False

        return {
            "all_nodes_covered": all_nodes_covered,
            "all_edges_covered": all_edges_covered,
            "boundary_consistent": boundary_consistent,
        }

    # ------------------------------------------------------------------
    # 12. Preserve edge metadata
    # ------------------------------------------------------------------
    def maintain_edge_metadata(
        self,
        original_graph: nx.Graph,
        subgraph: nx.Graph,
    ) -> nx.Graph:
        """Ensure every edge in *subgraph* that also exists in
        *original_graph* carries all original attributes."""
        result = subgraph.copy()
        for u, v in result.edges():
            if original_graph.has_edge(u, v):
                orig_attrs = original_graph.edges[u, v]
                for key, value in orig_attrs.items():
                    result.edges[u, v][key] = value
            elif original_graph.has_edge(v, u):
                orig_attrs = original_graph.edges[v, u]
                for key, value in orig_attrs.items():
                    result.edges[u, v][key] = value
        return result

    # ------------------------------------------------------------------
    # 13. Find best merge partner
    # ------------------------------------------------------------------
    def _find_best_merge_partner(
        self,
        subgraphs: List[nx.Graph],
        idx: int,
    ) -> int:
        """Return the index of the subgraph with the greatest node
        overlap with ``subgraphs[idx]``.  Returns -1 if no overlap is
        found with any other subgraph."""
        if idx < 0 or idx >= len(subgraphs):
            return -1

        target_nodes = set(subgraphs[idx].nodes())
        best_partner = -1
        best_overlap = 0

        for j in range(len(subgraphs)):
            if j == idx:
                continue
            other_nodes = set(subgraphs[j].nodes())
            overlap = len(target_nodes & other_nodes)
            if overlap > best_overlap:
                best_overlap = overlap
                best_partner = j

        if best_partner == -1 and len(subgraphs) > 1:
            min_dist = float("inf")
            for j in range(len(subgraphs)):
                if j == idx:
                    continue
                other_nodes = set(subgraphs[j].nodes())
                combined_size = len(target_nodes | other_nodes)
                if combined_size < min_dist:
                    min_dist = combined_size
                    best_partner = j

        return best_partner

    # ------------------------------------------------------------------
    # 14. Reindex subgraph nodes
    # ------------------------------------------------------------------
    def reindex_subgraphs(
        self, subgraphs: List[nx.Graph]
    ) -> List[nx.Graph]:
        """Reindex each subgraph so nodes are labelled 0 … n-1.  The
        original node ID is stored in the ``original_id`` attribute."""
        reindexed: List[nx.Graph] = []

        for sg in subgraphs:
            sorted_nodes = sorted(sg.nodes())
            mapping = {old: new for new, old in enumerate(sorted_nodes)}
            reverse_mapping = {new: old for old, new in mapping.items()}

            new_sg = nx.Graph()
            for old_node in sorted_nodes:
                new_node = mapping[old_node]
                attrs = dict(sg.nodes[old_node])
                attrs["original_id"] = old_node
                new_sg.add_node(new_node, **attrs)

            for u, v, attrs in sg.edges(data=True):
                new_u = mapping[u]
                new_v = mapping[v]
                edge_attrs = dict(attrs)
                edge_attrs["original_source"] = u
                edge_attrs["original_target"] = v
                new_sg.add_edge(new_u, new_v, **edge_attrs)

            new_sg.graph["node_mapping"] = reverse_mapping
            new_sg.graph["inverse_mapping"] = mapping
            reindexed.append(new_sg)

        return reindexed
