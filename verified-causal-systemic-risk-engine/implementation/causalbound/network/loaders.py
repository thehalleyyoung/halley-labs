"""
Topology Loaders for Financial Network Data
=============================================

Load empirical financial network topologies from various data formats
and reconstruct historical crisis network configurations.

Supports edge lists, adjacency matrices, and GraphML formats. Includes
preset crisis topologies for major systemic events.

References:
    - Glasserman & Young (2016) - Contagion in financial networks
    - Haldane & May (2011) - Systemic risk in banking ecosystems
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np


@dataclass
class LoaderConfig:
    """Configuration for data loading."""
    delimiter: str = ","
    has_header: bool = True
    weight_col: Optional[str] = None
    source_col: str = "source"
    target_col: str = "target"
    node_id_type: str = "int"  # int, str
    default_weight: float = 1e6
    validate_on_load: bool = True


@dataclass
class ValidationResult:
    """Results from graph validation."""
    is_valid: bool
    n_nodes: int
    n_edges: int
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, float]


# Crisis topology specifications derived from public regulatory reports
CRISIS_SPECS: Dict[str, Dict[str, Any]] = {
    "gfc_2008": {
        "description": "2008 Global Financial Crisis - US/European interbank network",
        "n_nodes": 50,
        "core_fraction": 0.12,
        "core_density": 0.85,
        "periphery_density": 0.03,
        "cross_density": 0.18,
        "reciprocity": 0.55,
        "cr5": 0.52,
        "avg_capital_ratio": 0.06,
        "leverage_mean": 25.0,
        "total_assets_bn": 45000.0,
        "institution_types": {
            "bank": 0.60, "fund": 0.25, "insurance": 0.10, "ccp": 0.05,
        },
        "regions": ["us", "uk", "eu", "ch"],
    },
    "eu_sovereign_2010": {
        "description": "2010 European Sovereign Debt Crisis",
        "n_nodes": 40,
        "core_fraction": 0.15,
        "core_density": 0.80,
        "periphery_density": 0.02,
        "cross_density": 0.15,
        "reciprocity": 0.60,
        "cr5": 0.48,
        "avg_capital_ratio": 0.065,
        "leverage_mean": 22.0,
        "total_assets_bn": 35000.0,
        "institution_types": {
            "bank": 0.70, "sovereign": 0.10, "fund": 0.10, "insurance": 0.10,
        },
        "regions": ["de", "fr", "es", "it", "gr", "pt", "ie"],
    },
    "covid_2020": {
        "description": "2020 COVID-19 Market Stress",
        "n_nodes": 60,
        "core_fraction": 0.10,
        "core_density": 0.75,
        "periphery_density": 0.04,
        "cross_density": 0.20,
        "reciprocity": 0.58,
        "cr5": 0.45,
        "avg_capital_ratio": 0.09,
        "leverage_mean": 15.0,
        "total_assets_bn": 55000.0,
        "institution_types": {
            "bank": 0.55, "fund": 0.30, "ccp": 0.08, "insurance": 0.07,
        },
        "regions": ["us", "uk", "eu", "jp", "cn"],
    },
    "uk_gilt_2023": {
        "description": "2023 UK Gilt Market Crisis (LDI funds)",
        "n_nodes": 35,
        "core_fraction": 0.14,
        "core_density": 0.70,
        "periphery_density": 0.05,
        "cross_density": 0.22,
        "reciprocity": 0.50,
        "cr5": 0.55,
        "avg_capital_ratio": 0.07,
        "leverage_mean": 20.0,
        "total_assets_bn": 8000.0,
        "institution_types": {
            "bank": 0.30, "fund": 0.50, "insurance": 0.15, "ccp": 0.05,
        },
        "regions": ["uk"],
    },
}


class TopologyLoader:
    """Load and reconstruct financial network topologies.

    Supports loading from empirical data files (edge lists, adjacency
    matrices, GraphML) and reconstructing stylised crisis network topologies
    based on public regulatory data.

    Example:
        >>> loader = TopologyLoader()
        >>> G = loader.reconstruct_crisis_topology("gfc_2008")
        >>> len(G.nodes)
        50
    """

    def __init__(
        self,
        config: Optional[LoaderConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or LoaderConfig()
        self.rng = np.random.default_rng(seed)

    def load_edgelist(
        self,
        path: Union[str, Path],
        config: Optional[LoaderConfig] = None,
    ) -> nx.DiGraph:
        """Load a network from an edge list file.

        Expects CSV/TSV with columns for source, target, and optionally weight.

        Args:
            path: Path to edge list file.
            config: Override loader configuration.

        Returns:
            Directed graph loaded from the edge list.
        """
        cfg = config or self.config
        path = Path(path)

        graph = nx.DiGraph()

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=cfg.delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Could not parse headers from {path}")

            for row in reader:
                src = self._parse_node_id(row[cfg.source_col], cfg.node_id_type)
                tgt = self._parse_node_id(row[cfg.target_col], cfg.node_id_type)

                weight = cfg.default_weight
                if cfg.weight_col and cfg.weight_col in row:
                    try:
                        weight = float(row[cfg.weight_col])
                    except (ValueError, TypeError):
                        weight = cfg.default_weight

                graph.add_edge(src, tgt, weight=weight, exposure=weight)

                # Add any extra columns as edge attributes
                for col, val in row.items():
                    if col not in (cfg.source_col, cfg.target_col, cfg.weight_col):
                        try:
                            graph.edges[src, tgt][col] = float(val)
                        except (ValueError, TypeError):
                            graph.edges[src, tgt][col] = val

        if cfg.validate_on_load:
            result = self.validate(graph)
            if not result.is_valid:
                raise ValueError(
                    f"Loaded graph failed validation: {result.errors}"
                )

        return graph

    def load_adjacency(
        self,
        path: Union[str, Path],
        delimiter: str = ",",
        has_header: bool = False,
    ) -> nx.DiGraph:
        """Load a network from an adjacency matrix file.

        Args:
            path: Path to adjacency matrix CSV.
            delimiter: Column delimiter.
            has_header: Whether the file has a header row.

        Returns:
            Directed graph from the adjacency matrix.
        """
        path = Path(path)

        with open(path, "r") as f:
            lines = f.readlines()

        start_row = 1 if has_header else 0
        node_labels = None

        if has_header:
            header = lines[0].strip().split(delimiter)
            # If first cell is empty or a label column, skip it
            if header[0] == "" or header[0].lower() in ("node", "id", "name"):
                node_labels = header[1:]
            else:
                node_labels = header

        matrix_rows = []
        row_labels = []
        for line in lines[start_row:]:
            parts = line.strip().split(delimiter)
            if not parts or all(p.strip() == "" for p in parts):
                continue
            # Check if first column is a label
            try:
                float(parts[0])
                values = [float(x) if x.strip() else 0.0 for x in parts]
            except ValueError:
                row_labels.append(parts[0])
                values = [float(x) if x.strip() else 0.0 for x in parts[1:]]
            matrix_rows.append(values)

        adj_matrix = np.array(matrix_rows)
        n = adj_matrix.shape[0]

        if node_labels is None:
            if row_labels:
                node_labels = row_labels
            else:
                node_labels = [str(i) for i in range(n)]

        graph = nx.DiGraph()
        for i in range(n):
            label = node_labels[i] if i < len(node_labels) else str(i)
            graph.add_node(label)

        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    src = node_labels[i] if i < len(node_labels) else str(i)
                    tgt = node_labels[j] if j < len(node_labels) else str(j)
                    weight = float(adj_matrix[i, j])
                    graph.add_edge(src, tgt, weight=weight, exposure=weight)

        return graph

    def load_graphml(
        self,
        path: Union[str, Path],
    ) -> nx.DiGraph:
        """Load a network from a GraphML file.

        Args:
            path: Path to GraphML file.

        Returns:
            Directed graph loaded from GraphML.
        """
        path = Path(path)
        graph = nx.read_graphml(str(path))

        # Ensure directed
        if not graph.is_directed():
            graph = graph.to_directed()

        # Ensure weight attributes exist
        for u, v in graph.edges():
            if "weight" not in graph.edges[u, v]:
                graph.edges[u, v]["weight"] = self.config.default_weight
            graph.edges[u, v]["exposure"] = graph.edges[u, v]["weight"]

        return graph

    def reconstruct_crisis_topology(
        self,
        crisis_name: str,
        seed: Optional[int] = None,
    ) -> nx.DiGraph:
        """Reconstruct a stylised crisis network topology.

        Builds a synthetic network calibrated to match published statistics
        about the financial network structure during major crises.

        Available crises:
            - 'gfc_2008': 2008 Global Financial Crisis
            - 'eu_sovereign_2010': 2010 European Sovereign Debt Crisis
            - 'covid_2020': 2020 COVID-19 Market Stress
            - 'uk_gilt_2023': 2023 UK Gilt Market Crisis

        Args:
            crisis_name: Name of the crisis to reconstruct.
            seed: Random seed for reproducibility.

        Returns:
            Directed financial network calibrated to crisis parameters.
        """
        if crisis_name not in CRISIS_SPECS:
            available = ", ".join(CRISIS_SPECS.keys())
            raise ValueError(
                f"Unknown crisis: {crisis_name}. Available: {available}"
            )

        spec = CRISIS_SPECS[crisis_name]
        rng = np.random.default_rng(seed or self.rng.integers(0, 2**31))
        n = spec["n_nodes"]
        n_core = max(2, int(n * spec["core_fraction"]))

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))

        core_nodes = list(range(n_core))
        periphery_nodes = list(range(n_core, n))

        # Core-core edges
        for i in core_nodes:
            for j in core_nodes:
                if i != j and rng.random() < spec["core_density"]:
                    graph.add_edge(i, j)

        # Core-periphery edges
        for c in core_nodes:
            for p in periphery_nodes:
                if rng.random() < spec["cross_density"]:
                    graph.add_edge(c, p)
                if rng.random() < spec["cross_density"]:
                    graph.add_edge(p, c)

        # Periphery-periphery edges
        for i in periphery_nodes:
            for j in periphery_nodes:
                if i != j and rng.random() < spec["periphery_density"]:
                    graph.add_edge(i, j)

        # Reciprocal edges
        target_recip = spec["reciprocity"]
        edges = list(graph.edges())
        non_recip = [(u, v) for u, v in edges if not graph.has_edge(v, u)]
        current_recip = 1 - len(non_recip) / max(len(edges), 1)
        if current_recip < target_recip and non_recip:
            n_add = int((target_recip - current_recip) * len(edges))
            n_add = min(n_add, len(non_recip))
            if n_add > 0:
                indices = rng.choice(len(non_recip), size=n_add, replace=False)
                for idx in indices:
                    u, v = non_recip[idx]
                    graph.add_edge(v, u)

        # Assign node attributes
        total_assets = spec["total_assets_bn"] * 1e9
        inst_types = spec["institution_types"]
        types = list(inst_types.keys())
        type_probs = np.array(list(inst_types.values()))
        type_probs /= type_probs.sum()

        # Size distribution (Pareto - heavy-tailed)
        sizes = (rng.pareto(a=1.3, size=n) + 1)
        sizes = sizes / sizes.sum() * total_assets
        sizes = np.sort(sizes)[::-1]

        # Assign core nodes the largest sizes
        assigned_types = rng.choice(len(types), size=n, p=type_probs)
        avg_cr = spec["avg_capital_ratio"]
        cr_std = avg_cr * 0.3

        for idx, node in enumerate(range(n)):
            is_core = node in core_nodes
            size = float(sizes[idx])
            inst_type = types[assigned_types[idx]]

            cr = max(0.02, rng.normal(avg_cr, cr_std))
            if is_core:
                cr = max(0.03, cr * 0.8)  # Core banks typically more leveraged

            capital = size * cr
            leverage = 1.0 / cr if cr > 0 else spec["leverage_mean"]

            if idx < max(1, n // 20):
                tier = 1
            elif idx < max(2, n // 7):
                tier = 2
            elif idx < n // 2:
                tier = 3
            else:
                tier = 4

            region = rng.choice(spec["regions"])

            graph.nodes[node]["size"] = size
            graph.nodes[node]["institution_type"] = inst_type
            graph.nodes[node]["capital"] = capital
            graph.nodes[node]["capital_ratio"] = cr
            graph.nodes[node]["leverage"] = leverage
            graph.nodes[node]["tier"] = tier
            graph.nodes[node]["core"] = is_core
            graph.nodes[node]["region"] = region
            graph.nodes[node]["external_assets"] = size * rng.uniform(0.3, 0.6)
            graph.nodes[node]["external_liabilities"] = size * rng.uniform(0.2, 0.5)

        # Assign exposure weights
        n_edges = graph.number_of_edges()
        if n_edges > 0:
            exp_sizes = (rng.pareto(a=1.5, size=n_edges) + 1) * 1e7
            for idx, (u, v) in enumerate(graph.edges()):
                lender_size = graph.nodes[u]["size"]
                borrower_size = graph.nodes[v]["size"]
                ref = min(lender_size, borrower_size)
                exposure = min(float(exp_sizes[idx]), ref * 0.20)
                exposure = max(exposure, 1e5)
                graph.edges[u, v]["weight"] = exposure
                graph.edges[u, v]["exposure"] = exposure

        graph.graph["crisis"] = crisis_name
        graph.graph["description"] = spec["description"]
        graph.graph["n_nodes"] = n
        graph.graph["total_assets_bn"] = spec["total_assets_bn"]

        return graph

    def validate(self, graph: nx.DiGraph) -> ValidationResult:
        """Validate a financial network graph.

        Checks for structural integrity, data consistency, and
        financial reasonableness.

        Args:
            graph: Network to validate.

        Returns:
            ValidationResult with errors, warnings, and statistics.
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats_dict: Dict[str, float] = {}

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes == 0:
            errors.append("Graph has no nodes")
            return ValidationResult(
                is_valid=False,
                n_nodes=0,
                n_edges=0,
                warnings=warnings,
                errors=errors,
                stats={},
            )

        # Check for self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            warnings.append(f"Graph has {len(self_loops)} self-loops")

        # Check connectivity
        if not nx.is_weakly_connected(graph):
            n_components = nx.number_weakly_connected_components(graph)
            warnings.append(f"Graph has {n_components} weakly connected components")

        # Check edge weights
        neg_weights = 0
        zero_weights = 0
        missing_weights = 0
        for u, v in graph.edges():
            w = graph.edges[u, v].get("weight", None)
            if w is None:
                missing_weights += 1
            elif w == 0:
                zero_weights += 1
            elif w < 0:
                neg_weights += 1

        if neg_weights > 0:
            errors.append(f"{neg_weights} edges have negative weights")
        if missing_weights > 0:
            warnings.append(f"{missing_weights} edges missing weight attribute")
        if zero_weights > 0:
            warnings.append(f"{zero_weights} edges have zero weight")

        # Check node attributes
        for node in graph.nodes():
            size = graph.nodes[node].get("size", None)
            if size is not None and size <= 0:
                warnings.append(f"Node {node} has non-positive size: {size}")
            capital = graph.nodes[node].get("capital", None)
            if capital is not None and capital < 0:
                errors.append(f"Node {node} has negative capital: {capital}")

        # Basic statistics
        max_edges = n_nodes * (n_nodes - 1)
        stats_dict["density"] = n_edges / max_edges if max_edges > 0 else 0.0
        stats_dict["reciprocity"] = float(nx.reciprocity(graph)) if n_edges > 0 else 0.0
        stats_dict["avg_in_degree"] = float(np.mean([graph.in_degree(n) for n in graph.nodes()]))
        stats_dict["avg_out_degree"] = float(np.mean([graph.out_degree(n) for n in graph.nodes()]))

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            n_nodes=n_nodes,
            n_edges=n_edges,
            warnings=warnings,
            errors=errors,
            stats=stats_dict,
        )

    def _parse_node_id(self, value: str, id_type: str) -> Any:
        """Parse a node ID from string."""
        if id_type == "int":
            return int(value)
        return value.strip()
