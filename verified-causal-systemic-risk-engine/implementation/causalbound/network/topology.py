"""
Network Topology Analysis for Financial Networks
==================================================

Comprehensive analysis of financial network topology including degree
distributions, centrality measures, core-periphery detection, community
structure, and tiered institution classification.

References:
    - Borgatti & Everett (2000) - Core-periphery models
    - Blondel et al. (2008) - Louvain community detection
    - Newman (2006) - Modularity and community structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import optimize, stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


class CentralityMethod(Enum):
    """Available centrality computation methods."""
    BETWEENNESS = "betweenness"
    EIGENVECTOR = "eigenvector"
    KATZ = "katz"
    PAGERANK = "pagerank"
    CLOSENESS = "closeness"
    DEGREE = "degree"


@dataclass
class DegreeDistribution:
    """Degree distribution analysis results."""
    in_degrees: np.ndarray
    out_degrees: np.ndarray
    total_degrees: np.ndarray
    mean_in: float
    mean_out: float
    mean_total: float
    var_in: float
    var_out: float
    var_total: float
    max_in: int
    max_out: int
    max_total: int
    power_law_exponent: Optional[float] = None
    power_law_p_value: Optional[float] = None
    is_power_law: bool = False


@dataclass
class TopologyReport:
    """Complete topology analysis report."""
    n_nodes: int
    n_edges: int
    density: float
    reciprocity: float
    avg_clustering: float
    transitivity: float
    avg_path_length: Optional[float]
    diameter: Optional[int]
    n_components_weak: int
    n_components_strong: int
    largest_weak_component_size: int
    largest_strong_component_size: int
    degree_distribution: DegreeDistribution
    centrality_scores: Dict[str, Dict[int, float]]
    core_periphery: Optional[Dict[str, Any]] = None
    communities: Optional[Dict[str, Any]] = None
    tier_classification: Optional[Dict[int, List[int]]] = None
    concentration_metrics: Optional[Dict[str, float]] = None


class NetworkTopology:
    """Financial network topology analyser.

    Provides comprehensive analysis of interbank network structure including
    degree distributions, centrality measures, core-periphery detection,
    community detection, and systemic importance classification.

    Example:
        >>> topo = NetworkTopology()
        >>> report = topo.analyze(graph)
        >>> centrality = topo.get_centrality(graph, CentralityMethod.BETWEENNESS)
    """

    def __init__(self, weight_attr: str = "weight"):
        self.weight_attr = weight_attr

    def analyze(self, graph: nx.DiGraph) -> TopologyReport:
        """Perform full topology analysis on a financial network.

        Args:
            graph: Directed financial network graph.

        Returns:
            TopologyReport with all analysis results.
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0

        reciprocity = nx.reciprocity(graph) if n_edges > 0 else 0.0
        avg_clustering = nx.average_clustering(graph.to_undirected())
        transitivity = nx.transitivity(graph.to_undirected())

        # Connected components
        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        n_weak = len(weak_components)
        n_strong = len(strong_components)
        largest_weak = max(len(c) for c in weak_components) if weak_components else 0
        largest_strong = max(len(c) for c in strong_components) if strong_components else 0

        # Path length and diameter (on largest weakly connected component)
        avg_path = None
        diameter = None
        if largest_weak > 1:
            largest_wc = max(weak_components, key=len)
            subgraph = graph.subgraph(largest_wc)
            try:
                if nx.is_strongly_connected(subgraph):
                    avg_path = nx.average_shortest_path_length(subgraph)
                    diameter = nx.diameter(subgraph)
                else:
                    # Use undirected version for approximate measures
                    ug = subgraph.to_undirected()
                    if nx.is_connected(ug):
                        avg_path = nx.average_shortest_path_length(ug)
                        diameter = nx.diameter(ug)
            except nx.NetworkXError:
                pass

        degree_dist = self._analyze_degree_distribution(graph)

        centrality_scores = {
            "betweenness": self.get_centrality(graph, CentralityMethod.BETWEENNESS),
            "eigenvector": self.get_centrality(graph, CentralityMethod.EIGENVECTOR),
            "pagerank": self.get_centrality(graph, CentralityMethod.PAGERANK),
        }

        return TopologyReport(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            reciprocity=reciprocity,
            avg_clustering=avg_clustering,
            transitivity=transitivity,
            avg_path_length=avg_path,
            diameter=diameter,
            n_components_weak=n_weak,
            n_components_strong=n_strong,
            largest_weak_component_size=largest_weak,
            largest_strong_component_size=largest_strong,
            degree_distribution=degree_dist,
            centrality_scores=centrality_scores,
        )

    def get_centrality(
        self,
        graph: nx.DiGraph,
        method: CentralityMethod = CentralityMethod.BETWEENNESS,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Compute centrality scores for all nodes.

        Args:
            graph: Financial network graph.
            method: Centrality measure to compute.
            **kwargs: Additional parameters for specific methods.

        Returns:
            Dictionary mapping node IDs to centrality scores.
        """
        if graph.number_of_nodes() == 0:
            return {}

        if method == CentralityMethod.BETWEENNESS:
            return nx.betweenness_centrality(
                graph,
                weight=self.weight_attr,
                normalized=True,
            )
        elif method == CentralityMethod.EIGENVECTOR:
            try:
                return nx.eigenvector_centrality(
                    graph,
                    max_iter=1000,
                    weight=self.weight_attr,
                )
            except nx.PowerIterationFailedConvergence:
                return nx.eigenvector_centrality_numpy(
                    graph, weight=self.weight_attr
                )
        elif method == CentralityMethod.KATZ:
            # Compute spectral radius for alpha
            try:
                adj = nx.adjacency_matrix(graph, weight=self.weight_attr)
                eigenvalues = eigs(adj.astype(float), k=1, which="LM", return_eigenvectors=False)
                spectral_radius = float(np.abs(eigenvalues[0]))
                alpha = 0.9 / spectral_radius if spectral_radius > 0 else 0.01
            except Exception:
                alpha = 0.01
            return nx.katz_centrality(
                graph,
                alpha=alpha,
                beta=1.0,
                weight=self.weight_attr,
            )
        elif method == CentralityMethod.PAGERANK:
            damping = kwargs.get("damping", 0.85)
            return nx.pagerank(
                graph,
                alpha=damping,
                weight=self.weight_attr,
            )
        elif method == CentralityMethod.CLOSENESS:
            return nx.closeness_centrality(graph)
        elif method == CentralityMethod.DEGREE:
            n = graph.number_of_nodes()
            if n <= 1:
                return {node: 0.0 for node in graph.nodes()}
            return {
                node: graph.degree(node, weight=self.weight_attr) / (n - 1)
                for node in graph.nodes()
            }
        else:
            raise ValueError(f"Unknown centrality method: {method}")

    def detect_core_periphery(
        self,
        graph: nx.DiGraph,
        method: str = "borgatti_everett",
        n_core: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detect core-periphery structure in the network.

        Args:
            graph: Financial network.
            method: Detection algorithm ('borgatti_everett' or 'minres').
            n_core: Number of core nodes (estimated if None).

        Returns:
            Dictionary with core nodes, periphery nodes, fitness score.
        """
        adj = nx.adjacency_matrix(graph).toarray().astype(float)
        n = adj.shape[0]
        nodes = list(graph.nodes())

        if n_core is None:
            n_core = self._estimate_core_size(adj)

        if method == "borgatti_everett":
            return self._borgatti_everett(adj, nodes, n_core)
        elif method == "minres":
            return self._minres_core_periphery(adj, nodes, n_core)
        else:
            raise ValueError(f"Unknown core-periphery method: {method}")

    def _borgatti_everett(
        self, adj: np.ndarray, nodes: List[int], n_core: int
    ) -> Dict[str, Any]:
        """Borgatti-Everett core-periphery detection.

        Uses binary search to partition nodes into core and periphery by
        maximising correlation with ideal core-periphery block model.
        """
        n = adj.shape[0]
        # Normalise adjacency
        row_max = adj.max()
        if row_max > 0:
            adj_norm = adj / row_max
        else:
            adj_norm = adj.copy()

        # Initial assignment: top n_core nodes by degree
        degrees = adj_norm.sum(axis=0) + adj_norm.sum(axis=1)
        ranking = np.argsort(degrees)[::-1]
        core_mask = np.zeros(n, dtype=bool)
        core_mask[ranking[:n_core]] = True

        best_fitness = -np.inf
        best_mask = core_mask.copy()

        # Iterative improvement
        for iteration in range(200):
            improved = False
            for i in range(n):
                core_mask[i] = not core_mask[i]
                fitness = self._core_periphery_fitness(adj_norm, core_mask)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_mask = core_mask.copy()
                    improved = True
                else:
                    core_mask[i] = not core_mask[i]

            if not improved:
                break

        core_nodes = [nodes[i] for i in range(n) if best_mask[i]]
        periphery_nodes = [nodes[i] for i in range(n) if not best_mask[i]]

        return {
            "core_nodes": core_nodes,
            "periphery_nodes": periphery_nodes,
            "fitness": best_fitness,
            "core_size": len(core_nodes),
            "method": "borgatti_everett",
        }

    def _minres_core_periphery(
        self, adj: np.ndarray, nodes: List[int], n_core: int
    ) -> Dict[str, Any]:
        """MINRES algorithm for continuous core-periphery detection.

        Assigns continuous coreness scores to nodes by minimising residuals
        between the adjacency matrix and the outer product of the coreness
        vector.
        """
        n = adj.shape[0]
        row_max = adj.max()
        if row_max > 0:
            adj_norm = adj / row_max
        else:
            adj_norm = adj.copy()

        # Initialize coreness with degree
        degrees = adj_norm.sum(axis=0) + adj_norm.sum(axis=1)
        c = degrees / (degrees.max() + 1e-12)

        for iteration in range(300):
            c_old = c.copy()
            outer = np.outer(c, c)
            residuals = adj_norm - outer

            for i in range(n):
                gradient = 2 * (adj_norm[i, :] @ c + adj_norm[:, i] @ c - 2 * c[i] * (c ** 2).sum())
                c[i] = max(0.0, min(1.0, c[i] + 0.01 * gradient))

            if np.linalg.norm(c - c_old) < 1e-6:
                break

        # Threshold to binary
        threshold = np.sort(c)[::-1][min(n_core, n - 1)]
        core_nodes = [nodes[i] for i in range(n) if c[i] >= threshold]
        periphery_nodes = [nodes[i] for i in range(n) if c[i] < threshold]

        return {
            "core_nodes": core_nodes,
            "periphery_nodes": periphery_nodes,
            "coreness_scores": {nodes[i]: float(c[i]) for i in range(n)},
            "fitness": float(-np.linalg.norm(adj_norm - np.outer(c, c))),
            "core_size": len(core_nodes),
            "method": "minres",
        }

    def _core_periphery_fitness(
        self, adj: np.ndarray, core_mask: np.ndarray
    ) -> float:
        """Compute fitness of a core-periphery partition.

        Fitness is the correlation between the adjacency matrix and the ideal
        core-periphery block model (core-core=1, core-periphery=1, periphery-periphery=0).
        """
        n = adj.shape[0]
        delta = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if core_mask[i] or core_mask[j]:
                    delta[i, j] = 1.0

        # Correlation (Pearson) between flattened matrices, excluding diagonal
        mask = ~np.eye(n, dtype=bool)
        a_flat = adj[mask]
        d_flat = delta[mask]

        if np.std(a_flat) < 1e-12 or np.std(d_flat) < 1e-12:
            return 0.0

        corr = np.corrcoef(a_flat, d_flat)[0, 1]
        return float(corr)

    def _estimate_core_size(self, adj: np.ndarray) -> int:
        """Estimate optimal core size from degree distribution."""
        n = adj.shape[0]
        degrees = adj.sum(axis=0) + adj.sum(axis=1)
        sorted_deg = np.sort(degrees)[::-1]

        # Look for elbow in sorted degree sequence
        if n < 5:
            return max(1, n // 3)

        diffs = np.diff(sorted_deg)
        # Find largest drop
        elbow = int(np.argmin(diffs)) + 1
        return max(2, min(elbow, n // 2))

    def detect_communities(
        self,
        graph: nx.DiGraph,
        method: str = "louvain",
        resolution: float = 1.0,
    ) -> Dict[str, Any]:
        """Detect community structure in the network.

        Args:
            graph: Financial network.
            method: Detection method ('louvain' or 'label_propagation').
            resolution: Resolution parameter for Louvain.

        Returns:
            Dictionary with communities, modularity, and node assignments.
        """
        undirected = graph.to_undirected()

        if method == "louvain":
            return self._louvain_communities(undirected, resolution)
        elif method == "label_propagation":
            return self._label_propagation(undirected)
        else:
            raise ValueError(f"Unknown community method: {method}")

    def _louvain_communities(
        self, graph: nx.Graph, resolution: float
    ) -> Dict[str, Any]:
        """Louvain community detection with modularity optimisation."""
        communities = nx.community.louvain_communities(
            graph,
            weight=self.weight_attr,
            resolution=resolution,
            seed=42,
        )

        # Build node -> community mapping
        node_community: Dict[int, int] = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                node_community[node] = comm_id

        modularity = nx.community.modularity(
            graph, communities, weight=self.weight_attr, resolution=resolution
        )

        return {
            "communities": [list(c) for c in communities],
            "n_communities": len(communities),
            "modularity": float(modularity),
            "node_community": node_community,
            "method": "louvain",
            "resolution": resolution,
            "sizes": [len(c) for c in communities],
        }

    def _label_propagation(self, graph: nx.Graph) -> Dict[str, Any]:
        """Asynchronous label propagation community detection."""
        communities = list(nx.community.asyn_lpa_communities(
            graph, weight=self.weight_attr, seed=42
        ))

        node_community: Dict[int, int] = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                node_community[node] = comm_id

        modularity = nx.community.modularity(
            graph, communities, weight=self.weight_attr
        )

        return {
            "communities": [list(c) for c in communities],
            "n_communities": len(communities),
            "modularity": float(modularity),
            "node_community": node_community,
            "method": "label_propagation",
            "sizes": [len(c) for c in communities],
        }

    def classify_tiers(
        self, graph: nx.DiGraph
    ) -> Dict[int, List[int]]:
        """Classify institutions into systemic importance tiers.

        Tiers:
            1: G-SIB equivalent (top 5% by composite score)
            2: Large (top 15%)
            3: Medium (top 45%)
            4: Small (bottom 55%)

        The composite score combines size, interconnectedness, and centrality.

        Args:
            graph: Financial network.

        Returns:
            Mapping of tier number to list of node IDs.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return {1: [], 2: [], 3: [], 4: []}

        nodes = list(graph.nodes())
        scores = np.zeros(n)

        sizes = np.array([graph.nodes[nd].get("size", 1.0) for nd in nodes])
        if sizes.max() > 0:
            norm_sizes = sizes / sizes.max()
        else:
            norm_sizes = np.ones(n)

        # Interconnectedness from total degree (weighted)
        degrees = np.array([
            graph.degree(nd, weight=self.weight_attr)
            for nd in nodes
        ], dtype=float)
        if degrees.max() > 0:
            norm_degrees = degrees / degrees.max()
        else:
            norm_degrees = np.zeros(n)

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(graph, weight=self.weight_attr)
        bc_values = np.array([betweenness.get(nd, 0.0) for nd in nodes])
        if bc_values.max() > 0:
            norm_bc = bc_values / bc_values.max()
        else:
            norm_bc = np.zeros(n)

        # Composite score (BCBS G-SIB methodology inspired weights)
        scores = 0.35 * norm_sizes + 0.35 * norm_degrees + 0.30 * norm_bc
        ranking = np.argsort(scores)[::-1]

        tiers: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
        for rank, idx in enumerate(ranking):
            rank_pct = rank / max(n - 1, 1)
            if rank_pct < 0.05:
                tier = 1
            elif rank_pct < 0.15:
                tier = 2
            elif rank_pct < 0.45:
                tier = 3
            else:
                tier = 4
            tiers[tier].append(nodes[idx])

        return tiers

    def compute_concentration(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute concentration metrics for the financial network.

        Computes various concentration ratios (CR3, CR5, CR10) and the
        Herfindahl-Hirschman Index (HHI) based on total exposures.

        Args:
            graph: Financial network.

        Returns:
            Dictionary of concentration metrics.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return {"cr3": 0.0, "cr5": 0.0, "cr10": 0.0, "hhi": 0.0}

        nodes = list(graph.nodes())
        total_exposures = np.array([
            sum(
                graph.edges[u, v].get(self.weight_attr, 0.0)
                for u, v in graph.out_edges(nd)
            )
            + sum(
                graph.edges[u, v].get(self.weight_attr, 0.0)
                for u, v in graph.in_edges(nd)
            )
            for nd in nodes
        ])

        total_system = total_exposures.sum()
        if total_system == 0:
            return {"cr3": 0.0, "cr5": 0.0, "cr10": 0.0, "hhi": 0.0}

        shares = total_exposures / total_system
        sorted_shares = np.sort(shares)[::-1]

        cr3 = float(sorted_shares[:min(3, n)].sum())
        cr5 = float(sorted_shares[:min(5, n)].sum())
        cr10 = float(sorted_shares[:min(10, n)].sum())
        hhi = float((shares ** 2).sum())

        # Gini coefficient
        sorted_exp = np.sort(total_exposures)
        index = np.arange(1, n + 1)
        gini = float(
            (2 * (index * sorted_exp).sum()) / (n * sorted_exp.sum()) - (n + 1) / n
        ) if sorted_exp.sum() > 0 else 0.0

        return {
            "cr3": cr3,
            "cr5": cr5,
            "cr10": cr10,
            "hhi": hhi,
            "gini": gini,
            "n_institutions": n,
        }

    def _analyze_degree_distribution(
        self, graph: nx.DiGraph
    ) -> DegreeDistribution:
        """Analyse the degree distribution of the network."""
        nodes = list(graph.nodes())
        in_degrees = np.array([graph.in_degree(n) for n in nodes])
        out_degrees = np.array([graph.out_degree(n) for n in nodes])
        total_degrees = in_degrees + out_degrees

        # Fit power law to total degree distribution
        exponent = None
        p_value = None
        is_pl = False
        nonzero_degrees = total_degrees[total_degrees > 0]
        if len(nonzero_degrees) > 10:
            exponent, p_value, is_pl = self._fit_power_law(nonzero_degrees)

        return DegreeDistribution(
            in_degrees=in_degrees,
            out_degrees=out_degrees,
            total_degrees=total_degrees,
            mean_in=float(in_degrees.mean()) if len(in_degrees) > 0 else 0.0,
            mean_out=float(out_degrees.mean()) if len(out_degrees) > 0 else 0.0,
            mean_total=float(total_degrees.mean()) if len(total_degrees) > 0 else 0.0,
            var_in=float(in_degrees.var()) if len(in_degrees) > 0 else 0.0,
            var_out=float(out_degrees.var()) if len(out_degrees) > 0 else 0.0,
            var_total=float(total_degrees.var()) if len(total_degrees) > 0 else 0.0,
            max_in=int(in_degrees.max()) if len(in_degrees) > 0 else 0,
            max_out=int(out_degrees.max()) if len(out_degrees) > 0 else 0,
            max_total=int(total_degrees.max()) if len(total_degrees) > 0 else 0,
            power_law_exponent=exponent,
            power_law_p_value=p_value,
            is_power_law=is_pl,
        )

    def _fit_power_law(
        self, degrees: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """Fit power law to degree sequence using MLE.

        Uses the Clauset-Shalizi-Newman method for estimating
        the power-law exponent via maximum likelihood.

        Returns:
            (exponent, p_value, is_power_law)
        """
        x = degrees.astype(float)
        x_min = max(1, int(np.percentile(x, 10)))
        x_tail = x[x >= x_min]

        if len(x_tail) < 5:
            return None, None, False

        # MLE estimate: alpha = 1 + n / sum(ln(x / x_min))
        n = len(x_tail)
        log_sum = np.sum(np.log(x_tail / x_min))
        if log_sum == 0:
            return None, None, False

        alpha = 1.0 + n / log_sum

        # KS test against fitted power law
        cdf_empirical = np.arange(1, n + 1) / n
        x_sorted = np.sort(x_tail)
        cdf_fitted = 1.0 - (x_min / x_sorted) ** (alpha - 1)
        ks_stat = np.max(np.abs(cdf_empirical - cdf_fitted))

        # p-value via bootstrap (simplified)
        n_bootstrap = 100
        ks_samples = np.zeros(n_bootstrap)
        rng = np.random.default_rng(42)
        for b in range(n_bootstrap):
            u = rng.random(n)
            synthetic = x_min * (1 - u) ** (-1 / (alpha - 1))
            syn_sorted = np.sort(synthetic)
            cdf_syn = np.arange(1, n + 1) / n
            cdf_fit_syn = 1.0 - (x_min / syn_sorted) ** (alpha - 1)
            ks_samples[b] = np.max(np.abs(cdf_syn - cdf_fit_syn))

        p_value = float(np.mean(ks_samples >= ks_stat))
        is_power_law = p_value > 0.1

        return float(alpha), p_value, is_power_law
