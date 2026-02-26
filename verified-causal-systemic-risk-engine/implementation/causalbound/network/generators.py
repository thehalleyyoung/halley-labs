"""
Network Generators for Financial Interbank Networks
====================================================

Implements multiple random graph models adapted for financial network generation.
Each generator produces weighted, directed graphs with realistic financial attributes
(institution size, type, capital, exposure weights).

References:
    - Erdos & Renyi (1959) - Random graphs
    - Barabasi & Albert (1999) - Preferential attachment
    - Watts & Strogatz (1998) - Small-world networks
    - Craig & von Peter (2014) - Core-periphery structure in interbank networks
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import stats


class InstitutionType(Enum):
    """Types of financial institutions in the network."""
    BANK = "bank"
    CCP = "ccp"
    FUND = "fund"
    INSURANCE = "insurance"
    SOVEREIGN = "sovereign"


@dataclass
class InstitutionAttributes:
    """Financial attributes for a network node."""
    size: float  # total assets
    institution_type: InstitutionType
    capital: float  # equity capital
    leverage: float  # assets / capital
    tier: int  # 1=G-SIB, 2=large, 3=medium, 4=small
    region: str = "global"
    external_assets: float = 0.0
    external_liabilities: float = 0.0


@dataclass
class TargetStatistics:
    """Target statistics for network calibration."""
    mean_degree: float = 10.0
    degree_variance: float = 25.0
    density: float = 0.1
    concentration_cr5: float = 0.45
    avg_exposure_pct: float = 0.02
    power_law_exponent: float = 2.5
    clustering_coefficient: float = 0.15
    reciprocity: float = 0.6


@dataclass
class ExposureParams:
    """Parameters for exposure size distribution."""
    distribution: str = "pareto"  # pareto, lognormal, exponential
    location: float = 1e6
    scale: float = 5e7
    shape: float = 1.5  # Pareto alpha
    mean_log: float = 17.0  # for lognormal
    sigma_log: float = 1.5  # for lognormal
    min_exposure: float = 1e5
    max_exposure: float = 1e11


class BaseNetworkGenerator(abc.ABC):
    """Abstract base class for financial network generators.

    All generators produce weighted, directed graphs with node and edge
    financial attributes suitable for systemic risk analysis.
    """

    def __init__(
        self,
        exposure_params: Optional[ExposureParams] = None,
        seed: Optional[int] = None,
    ):
        self.exposure_params = exposure_params or ExposureParams()
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    @abc.abstractmethod
    def generate(self, n_nodes: int, **params: Any) -> nx.DiGraph:
        """Generate a financial network graph.

        Args:
            n_nodes: Number of institutions in the network.
            **params: Generator-specific parameters.

        Returns:
            Directed graph with financial attributes.
        """

    def add_exposures(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Assign exposure weights to all edges.

        Exposure sizes are drawn from the configured distribution and scaled
        relative to the borrowing institution's total assets.

        Args:
            graph: Network graph (modified in place).

        Returns:
            Graph with 'weight' and 'exposure' edge attributes.
        """
        n_edges = graph.number_of_edges()
        if n_edges == 0:
            return graph

        raw_exposures = self._draw_exposures(n_edges)

        for idx, (u, v) in enumerate(graph.edges()):
            borrower_size = graph.nodes[v].get("size", 1e9)
            lender_size = graph.nodes[u].get("size", 1e9)
            raw = raw_exposures[idx]
            # Scale exposure relative to smaller institution
            ref_size = min(borrower_size, lender_size)
            exposure = min(raw, ref_size * 0.25)
            exposure = max(exposure, self.exposure_params.min_exposure)

            graph.edges[u, v]["weight"] = exposure
            graph.edges[u, v]["exposure"] = exposure
            graph.edges[u, v]["exposure_pct_lender"] = exposure / lender_size if lender_size > 0 else 0.0
            graph.edges[u, v]["exposure_pct_borrower"] = exposure / borrower_size if borrower_size > 0 else 0.0

        return graph

    def calibrate(self, target_statistics: TargetStatistics) -> Dict[str, Any]:
        """Compute generator parameters to match target statistics.

        Args:
            target_statistics: Target network statistics to match.

        Returns:
            Dictionary of calibrated generator parameters.
        """
        params: Dict[str, Any] = {}
        params["density"] = target_statistics.density
        params["mean_degree"] = target_statistics.mean_degree
        params["clustering"] = target_statistics.clustering_coefficient
        params["reciprocity"] = target_statistics.reciprocity
        return params

    def _assign_node_attributes(
        self,
        graph: nx.DiGraph,
        size_distribution: str = "lognormal",
        type_weights: Optional[Dict[InstitutionType, float]] = None,
    ) -> nx.DiGraph:
        """Assign financial attributes to all nodes.

        Args:
            graph: Network graph.
            size_distribution: Distribution for institution sizes.
            type_weights: Probability weights for institution types.

        Returns:
            Graph with node attributes populated.
        """
        n = graph.number_of_nodes()
        if type_weights is None:
            type_weights = {
                InstitutionType.BANK: 0.65,
                InstitutionType.CCP: 0.05,
                InstitutionType.FUND: 0.20,
                InstitutionType.INSURANCE: 0.08,
                InstitutionType.SOVEREIGN: 0.02,
            }

        types = list(type_weights.keys())
        probs = np.array(list(type_weights.values()))
        probs = probs / probs.sum()
        assigned_types = self.rng.choice(len(types), size=n, p=probs)

        if size_distribution == "lognormal":
            sizes = self.rng.lognormal(mean=22.0, sigma=1.8, size=n)
        elif size_distribution == "pareto":
            sizes = (self.rng.pareto(a=1.5, size=n) + 1) * 1e9
        else:
            sizes = self.rng.exponential(scale=1e10, size=n)

        sizes = np.sort(sizes)[::-1]

        capital_ratios = {
            InstitutionType.BANK: (0.04, 0.15),
            InstitutionType.CCP: (0.02, 0.08),
            InstitutionType.FUND: (0.10, 0.50),
            InstitutionType.INSURANCE: (0.08, 0.20),
            InstitutionType.SOVEREIGN: (0.15, 0.30),
        }

        for idx, node in enumerate(sorted(graph.nodes())):
            inst_type = types[assigned_types[idx]]
            size = float(sizes[idx])
            cr_low, cr_high = capital_ratios[inst_type]
            capital_ratio = self.rng.uniform(cr_low, cr_high)
            capital = size * capital_ratio
            leverage = 1.0 / capital_ratio if capital_ratio > 0 else 20.0

            # Tier assignment based on size rank
            rank_pct = idx / max(n - 1, 1)
            if rank_pct < 0.05:
                tier = 1
            elif rank_pct < 0.15:
                tier = 2
            elif rank_pct < 0.45:
                tier = 3
            else:
                tier = 4

            ext_asset_frac = self.rng.uniform(0.3, 0.7)
            graph.nodes[node]["size"] = size
            graph.nodes[node]["institution_type"] = inst_type.value
            graph.nodes[node]["capital"] = capital
            graph.nodes[node]["leverage"] = leverage
            graph.nodes[node]["tier"] = tier
            graph.nodes[node]["capital_ratio"] = capital_ratio
            graph.nodes[node]["external_assets"] = size * ext_asset_frac
            graph.nodes[node]["external_liabilities"] = size * (1 - ext_asset_frac) * (1 - capital_ratio)

        return graph

    def _draw_exposures(self, n: int) -> np.ndarray:
        """Draw n exposure sizes from the configured distribution."""
        ep = self.exposure_params
        if ep.distribution == "pareto":
            raw = (self.rng.pareto(a=ep.shape, size=n) + 1) * ep.location
        elif ep.distribution == "lognormal":
            raw = self.rng.lognormal(mean=ep.mean_log, sigma=ep.sigma_log, size=n)
        elif ep.distribution == "exponential":
            raw = self.rng.exponential(scale=ep.scale, size=n)
        else:
            raw = self.rng.pareto(a=ep.shape, size=n) * ep.location

        return np.clip(raw, ep.min_exposure, ep.max_exposure)

    def _add_reciprocal_edges(
        self, graph: nx.DiGraph, target_reciprocity: float = 0.6
    ) -> nx.DiGraph:
        """Add reciprocal edges to reach target reciprocity.

        Financial networks exhibit high reciprocity because lending relationships
        are often bilateral.
        """
        edges = list(graph.edges())
        current_reciprocal = sum(1 for u, v in edges if graph.has_edge(v, u))
        current_reciprocity = current_reciprocal / max(len(edges), 1)

        if current_reciprocity >= target_reciprocity:
            return graph

        non_reciprocal = [(u, v) for u, v in edges if not graph.has_edge(v, u)]
        n_to_add = int((target_reciprocity - current_reciprocity) * len(edges))
        n_to_add = min(n_to_add, len(non_reciprocal))

        if n_to_add > 0:
            chosen = self.rng.choice(len(non_reciprocal), size=n_to_add, replace=False)
            for idx in chosen:
                u, v = non_reciprocal[idx]
                graph.add_edge(v, u)

        return graph


class ErdosRenyiGenerator(BaseNetworkGenerator):
    """Erdos-Renyi random interbank network generator.

    Generates G(n, p) random directed graphs where each possible edge exists
    independently with probability p. Suitable as a baseline model for
    comparison with more structured topologies.

    Parameters:
        density: Edge probability (p) for the Erdos-Renyi model.
        reciprocity: Target reciprocity level (fraction of reciprocated edges).
        self_loops: Whether to allow self-loops.

    Example:
        >>> gen = ErdosRenyiGenerator(seed=42)
        >>> G = gen.generate(50, density=0.1)
        >>> len(G.nodes), len(G.edges)
        (50, ...)
    """

    def generate(
        self,
        n_nodes: int,
        density: float = 0.1,
        reciprocity: float = 0.6,
        self_loops: bool = False,
        **params: Any,
    ) -> nx.DiGraph:
        """Generate an Erdos-Renyi random financial network.

        Args:
            n_nodes: Number of institutions.
            density: Edge probability.
            reciprocity: Target edge reciprocity.
            self_loops: Allow self-loops.

        Returns:
            Directed weighted graph with financial attributes.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))

        # Generate adjacency using Bernoulli trials
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j and not self_loops:
                    continue
                if self.rng.random() < density:
                    graph.add_edge(i, j)

        graph = self._add_reciprocal_edges(graph, target_reciprocity=reciprocity)
        graph = self._assign_node_attributes(graph)
        graph = self.add_exposures(graph)

        graph.graph["generator"] = "erdos_renyi"
        graph.graph["n_nodes"] = n_nodes
        graph.graph["density"] = density
        graph.graph["target_reciprocity"] = reciprocity

        return graph

    def calibrate(self, target_statistics: TargetStatistics) -> Dict[str, Any]:
        """Calibrate Erdos-Renyi parameters to target statistics.

        For ER graphs, density directly maps to edge probability.
        """
        params = super().calibrate(target_statistics)
        params["density"] = target_statistics.density
        params["reciprocity"] = target_statistics.reciprocity
        return params


class ScaleFreeGenerator(BaseNetworkGenerator):
    """Scale-free network generator using preferential attachment.

    Generates directed networks with power-law degree distributions via
    the Barabasi-Albert model, adapted for financial networks. Large
    institutions attract disproportionately many connections, reflecting
    the "too-interconnected-to-fail" phenomenon.

    Parameters:
        m: Number of edges each new node brings (BA parameter).
        alpha: Preferential attachment exponent.
        size_attachment: Whether larger institutions receive more connections.

    Reference:
        Barabasi, A.-L. & Albert, R. (1999). Emergence of scaling in random
        networks. Science, 286(5439), 509-512.
    """

    def generate(
        self,
        n_nodes: int,
        m: int = 3,
        alpha: float = 1.0,
        size_attachment: bool = True,
        reciprocity: float = 0.5,
        **params: Any,
    ) -> nx.DiGraph:
        """Generate a scale-free financial network.

        Args:
            n_nodes: Number of institutions.
            m: Edges per new node (controls density).
            alpha: Preferential attachment exponent.
            size_attachment: Link attachment probability to institution size.
            reciprocity: Target reciprocity.

        Returns:
            Directed weighted graph with power-law degree distribution.
        """
        # Start with a complete directed graph on m+1 nodes
        graph = nx.DiGraph()
        init_nodes = min(m + 1, n_nodes)
        graph.add_nodes_from(range(init_nodes))
        for i in range(init_nodes):
            for j in range(init_nodes):
                if i != j:
                    graph.add_edge(i, j)

        # Preferential attachment for remaining nodes
        for new_node in range(init_nodes, n_nodes):
            graph.add_node(new_node)
            existing = list(graph.nodes())
            existing.remove(new_node)

            if len(existing) == 0:
                continue

            # Compute attachment probabilities
            in_degrees = np.array([graph.in_degree(n) for n in existing], dtype=float)
            out_degrees = np.array([graph.out_degree(n) for n in existing], dtype=float)
            total_degrees = in_degrees + out_degrees + 1.0  # +1 avoids zero

            probs = np.power(total_degrees, alpha)
            probs /= probs.sum()

            n_targets = min(m, len(existing))
            targets = self.rng.choice(
                len(existing), size=n_targets, replace=False, p=probs
            )

            for t_idx in targets:
                target = existing[t_idx]
                # Randomly choose direction
                if self.rng.random() < 0.5:
                    graph.add_edge(new_node, target)
                else:
                    graph.add_edge(target, new_node)

        graph = self._add_reciprocal_edges(graph, target_reciprocity=reciprocity)
        graph = self._assign_node_attributes(graph)

        # For scale-free nets, correlate size with degree
        if size_attachment:
            self._correlate_size_degree(graph)

        graph = self.add_exposures(graph)

        graph.graph["generator"] = "scale_free"
        graph.graph["n_nodes"] = n_nodes
        graph.graph["m"] = m
        graph.graph["alpha"] = alpha

        return graph

    def _correlate_size_degree(self, graph: nx.DiGraph) -> None:
        """Reassign sizes so high-degree nodes are larger institutions."""
        nodes_by_degree = sorted(
            graph.nodes(), key=lambda n: graph.degree(n), reverse=True
        )
        sizes = sorted(
            [graph.nodes[n]["size"] for n in graph.nodes()], reverse=True
        )
        for rank, node in enumerate(nodes_by_degree):
            graph.nodes[node]["size"] = sizes[rank]
            # Recompute dependent attributes
            cr = graph.nodes[node]["capital_ratio"]
            graph.nodes[node]["capital"] = sizes[rank] * cr
            ext_frac = graph.nodes[node]["external_assets"] / (
                graph.nodes[node]["external_assets"] + graph.nodes[node]["external_liabilities"] + 1
            )
            graph.nodes[node]["external_assets"] = sizes[rank] * ext_frac
            graph.nodes[node]["external_liabilities"] = sizes[rank] * (1 - ext_frac) * (1 - cr)
            # Update tier
            rank_pct = rank / max(len(nodes_by_degree) - 1, 1)
            if rank_pct < 0.05:
                graph.nodes[node]["tier"] = 1
            elif rank_pct < 0.15:
                graph.nodes[node]["tier"] = 2
            elif rank_pct < 0.45:
                graph.nodes[node]["tier"] = 3
            else:
                graph.nodes[node]["tier"] = 4

    def calibrate(self, target_statistics: TargetStatistics) -> Dict[str, Any]:
        """Calibrate BA parameters from target statistics.

        Uses the relationship between m and mean degree: <k> ~ 2m.
        """
        params = super().calibrate(target_statistics)
        params["m"] = max(1, int(target_statistics.mean_degree / 2))
        params["alpha"] = max(0.5, min(2.0, target_statistics.power_law_exponent - 1))
        return params


class CorePeripheryGenerator(BaseNetworkGenerator):
    """Core-periphery network generator for financial systems.

    Generates networks with a dense core of highly interconnected large
    institutions and a sparse periphery of smaller institutions primarily
    connected to the core. This structure is commonly observed in empirical
    interbank networks (Craig & von Peter, 2014).

    Parameters:
        core_fraction: Fraction of nodes in the core.
        core_density: Edge density within the core.
        periphery_density: Edge density within the periphery.
        cross_density: Edge density between core and periphery.
    """

    def generate(
        self,
        n_nodes: int,
        core_fraction: float = 0.15,
        core_density: float = 0.8,
        periphery_density: float = 0.02,
        cross_density: float = 0.15,
        reciprocity: float = 0.6,
        **params: Any,
    ) -> nx.DiGraph:
        """Generate a core-periphery financial network.

        Args:
            n_nodes: Number of institutions.
            core_fraction: Fraction of nodes in the core.
            core_density: Edge probability within core.
            periphery_density: Edge probability within periphery.
            cross_density: Edge probability between core and periphery.
            reciprocity: Target reciprocity.

        Returns:
            Directed graph with core-periphery structure.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))

        n_core = max(2, int(n_nodes * core_fraction))
        core_nodes = list(range(n_core))
        periphery_nodes = list(range(n_core, n_nodes))

        # Core-core edges (dense)
        for i in core_nodes:
            for j in core_nodes:
                if i != j and self.rng.random() < core_density:
                    graph.add_edge(i, j)

        # Core-periphery and periphery-core edges
        for c in core_nodes:
            for p in periphery_nodes:
                if self.rng.random() < cross_density:
                    graph.add_edge(c, p)
                if self.rng.random() < cross_density:
                    graph.add_edge(p, c)

        # Periphery-periphery edges (sparse)
        for i in periphery_nodes:
            for j in periphery_nodes:
                if i != j and self.rng.random() < periphery_density:
                    graph.add_edge(i, j)

        graph = self._add_reciprocal_edges(graph, target_reciprocity=reciprocity)
        graph = self._assign_node_attributes(graph, size_distribution="pareto")

        # Ensure core nodes are the largest institutions
        self._assign_core_periphery_sizes(graph, core_nodes, periphery_nodes)

        graph = self.add_exposures(graph)

        # Tag nodes with core/periphery membership
        for node in graph.nodes():
            graph.nodes[node]["core"] = node in core_nodes

        graph.graph["generator"] = "core_periphery"
        graph.graph["n_nodes"] = n_nodes
        graph.graph["n_core"] = n_core
        graph.graph["core_fraction"] = core_fraction
        graph.graph["core_density"] = core_density
        graph.graph["periphery_density"] = periphery_density
        graph.graph["cross_density"] = cross_density

        return graph

    def _assign_core_periphery_sizes(
        self,
        graph: nx.DiGraph,
        core_nodes: List[int],
        periphery_nodes: List[int],
    ) -> None:
        """Assign larger sizes to core nodes, smaller to periphery."""
        all_sizes = sorted(
            [graph.nodes[n]["size"] for n in graph.nodes()], reverse=True
        )
        n_core = len(core_nodes)
        core_sizes = all_sizes[:n_core]
        periph_sizes = all_sizes[n_core:]

        self.rng.shuffle(core_sizes)
        self.rng.shuffle(periph_sizes)

        for idx, node in enumerate(core_nodes):
            size = core_sizes[idx]
            graph.nodes[node]["size"] = size
            cr = graph.nodes[node]["capital_ratio"]
            graph.nodes[node]["capital"] = size * cr
            graph.nodes[node]["tier"] = 1 if idx < max(1, n_core // 3) else 2

        for idx, node in enumerate(periphery_nodes):
            if idx < len(periph_sizes):
                size = periph_sizes[idx]
                graph.nodes[node]["size"] = size
                cr = graph.nodes[node]["capital_ratio"]
                graph.nodes[node]["capital"] = size * cr
                rank_pct = idx / max(len(periphery_nodes) - 1, 1)
                graph.nodes[node]["tier"] = 3 if rank_pct < 0.4 else 4

    def calibrate(self, target_statistics: TargetStatistics) -> Dict[str, Any]:
        """Calibrate core-periphery parameters from target statistics.

        Maps concentration ratio to core fraction and density parameters.
        """
        params = super().calibrate(target_statistics)
        cr5 = target_statistics.concentration_cr5
        params["core_fraction"] = max(0.05, min(0.3, cr5 * 0.3))
        params["core_density"] = min(0.95, target_statistics.density * 8)
        params["periphery_density"] = max(0.005, target_statistics.density * 0.2)
        params["cross_density"] = target_statistics.density
        return params


class SmallWorldGenerator(BaseNetworkGenerator):
    """Small-world network generator adapted for financial networks.

    Implements the Watts-Strogatz model with modifications for directed,
    weighted financial networks. Produces high clustering with short path
    lengths, reflecting the structure observed in payment systems and
    interbank markets.

    Parameters:
        k: Number of nearest neighbours in the initial ring lattice.
        beta: Rewiring probability.
        directed_method: How to direct edges ('random', 'size_based').

    Reference:
        Watts, D.J. & Strogatz, S.H. (1998). Collective dynamics of
        'small-world' networks. Nature, 393, 440-442.
    """

    def generate(
        self,
        n_nodes: int,
        k: int = 6,
        beta: float = 0.3,
        directed_method: str = "random",
        reciprocity: float = 0.6,
        **params: Any,
    ) -> nx.DiGraph:
        """Generate a small-world financial network.

        Args:
            n_nodes: Number of institutions.
            k: Each node connected to k nearest neighbours in ring.
            beta: Rewiring probability.
            directed_method: Edge direction assignment method.
            reciprocity: Target reciprocity.

        Returns:
            Small-world directed weighted graph.
        """
        if k >= n_nodes:
            k = max(2, n_nodes - 1)
        if k % 2 != 0:
            k = max(2, k - 1)

        # Generate undirected Watts-Strogatz graph
        ws_graph = nx.watts_strogatz_graph(
            n_nodes, k, beta, seed=self._seed
        )

        # Convert to directed graph
        graph = nx.DiGraph()
        graph.add_nodes_from(ws_graph.nodes())

        if directed_method == "size_based":
            # Temporarily assign sizes for direction
            graph = self._assign_node_attributes(graph)
            for u, v in ws_graph.edges():
                size_u = graph.nodes[u].get("size", 1.0)
                size_v = graph.nodes[v].get("size", 1.0)
                ratio = size_u / (size_u + size_v)
                # Larger nodes more likely to lend
                if self.rng.random() < ratio:
                    graph.add_edge(u, v)
                else:
                    graph.add_edge(v, u)
        else:
            for u, v in ws_graph.edges():
                if self.rng.random() < 0.5:
                    graph.add_edge(u, v)
                else:
                    graph.add_edge(v, u)
            graph = self._assign_node_attributes(graph)

        graph = self._add_reciprocal_edges(graph, target_reciprocity=reciprocity)
        graph = self.add_exposures(graph)

        graph.graph["generator"] = "small_world"
        graph.graph["n_nodes"] = n_nodes
        graph.graph["k"] = k
        graph.graph["beta"] = beta
        graph.graph["directed_method"] = directed_method

        return graph

    def calibrate(self, target_statistics: TargetStatistics) -> Dict[str, Any]:
        """Calibrate Watts-Strogatz parameters from target statistics.

        Maps mean degree to k, and clustering coefficient to beta via the
        analytical relationship C(beta) ~ C(0)(1 - beta)^3.
        """
        params = super().calibrate(target_statistics)
        k = max(2, int(target_statistics.mean_degree))
        if k % 2 != 0:
            k += 1
        params["k"] = k

        # Invert clustering-beta relationship
        c_ring = 3 * (k - 2) / (4 * (k - 1)) if k > 2 else 0.5
        target_c = target_statistics.clustering_coefficient
        if c_ring > 0 and target_c > 0:
            ratio = min(1.0, target_c / c_ring)
            beta = max(0.0, 1.0 - ratio ** (1.0 / 3.0))
        else:
            beta = 0.3
        params["beta"] = beta
        return params
