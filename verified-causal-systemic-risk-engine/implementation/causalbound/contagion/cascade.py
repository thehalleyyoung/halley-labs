"""
Default Cascade Model
======================

Sequential default propagation through financial networks. When an
institution defaults, its counterparties incur losses that may push
them into default as well, creating cascading failures.

This model captures the direct credit contagion channel and provides
tools for analysing cascade sizes, tipping points, and critical mass
thresholds.

References:
    - Eisenberg, L. & Noe, T.H. (2001). Systemic risk in financial systems.
      Management Science, 47(2), 236-249.
    - Gai, P. & Kapadia, S. (2010). Contagion in financial networks.
      Proceedings of the Royal Society A, 466(2120), 2401-2423.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import stats


@dataclass
class CascadeResult:
    """Results from a cascade simulation."""
    initial_defaults: Set[int]
    final_defaults: Set[int]
    cascade_size: int  # number of defaults beyond initial
    total_defaults: int
    cascade_rounds: int
    system_loss: float  # total loss as fraction of system value
    default_sequence: List[Set[int]]  # defaults per round
    losses_per_round: List[float]
    node_losses: Dict[int, float]  # loss incurred by each node
    survived_nodes: Set[int]
    loss_given_default: Dict[int, float]  # LGD for each defaulted node


@dataclass
class TippingPoint:
    """A tipping point in the cascade dynamics."""
    shock_size: float  # fraction of nodes initially shocked
    cascade_fraction: float  # fraction of system that defaults
    is_tipping: bool  # whether this represents a phase transition
    critical_nodes: List[int]  # nodes whose default triggers large cascades


@dataclass
class CascadePathAnalysis:
    """Analysis of contagion paths within a cascade."""
    paths: List[List[int]]  # all observed contagion paths
    path_lengths: List[int]
    mean_path_length: float
    max_path_length: int
    branching_factor: float  # average number of defaults triggered per default
    critical_edges: List[Tuple[int, int]]  # edges most frequently on contagion paths


class CascadeModel:
    """Default cascade model for financial networks.

    Simulates sequential default propagation where the failure of one
    institution imposes credit losses on its counterparties, potentially
    triggering further defaults.

    Default condition: a node defaults when cumulative losses exceed
    its available capital (equity buffer).

    Example:
        >>> model = CascadeModel()
        >>> result = model.simulate_cascade(graph, initial_defaults={0, 1})
        >>> print(f"Cascade size: {result.cascade_size}")
    """

    def __init__(
        self,
        recovery_rate: float = 0.4,
        weight_attr: str = "weight",
        capital_attr: str = "capital",
        size_attr: str = "size",
    ):
        """Initialise the cascade model.

        Args:
            recovery_rate: Recovery rate on defaulted exposures (0 to 1).
            weight_attr: Edge attribute for exposure amounts.
            capital_attr: Node attribute for available capital.
            size_attr: Node attribute for institution size.
        """
        self.recovery_rate = recovery_rate
        self.weight_attr = weight_attr
        self.capital_attr = capital_attr
        self.size_attr = size_attr

    def simulate_cascade(
        self,
        graph: nx.DiGraph,
        initial_defaults: Set[int],
        max_rounds: int = 200,
    ) -> CascadeResult:
        """Simulate a default cascade from initial defaults.

        Args:
            graph: Financial network with exposure weights and capital.
            initial_defaults: Set of initially defaulting nodes.
            max_rounds: Maximum cascade rounds.

        Returns:
            CascadeResult with full cascade details.
        """
        nodes = list(graph.nodes())
        n = len(nodes)

        # Track state
        defaulted: Set[int] = set(initial_defaults)
        losses = {nd: 0.0 for nd in nodes}
        capital_remaining = {
            nd: graph.nodes[nd].get(self.capital_attr, 1e8)
            for nd in nodes
        }
        lgd: Dict[int, float] = {}
        default_sequence: List[Set[int]] = [set(initial_defaults)]
        losses_per_round: List[float] = []

        # Compute initial loss from defaults
        round_loss = 0.0
        for d_node in initial_defaults:
            lgd[d_node] = 1.0 - self.recovery_rate
            # Losses to creditors of the defaulted node
            for pred in graph.predecessors(d_node):
                if pred in defaulted:
                    continue
                exposure = graph.edges[pred, d_node].get(self.weight_attr, 0.0)
                loss = exposure * (1.0 - self.recovery_rate)
                losses[pred] += loss
                round_loss += loss

        losses_per_round.append(round_loss)

        # Cascade propagation
        rounds = 0
        for rnd in range(max_rounds):
            new_defaults: Set[int] = set()

            for nd in nodes:
                if nd in defaulted:
                    continue
                if losses[nd] >= capital_remaining[nd]:
                    new_defaults.add(nd)
                    defaulted.add(nd)
                    lgd[nd] = min(1.0, losses[nd] / max(capital_remaining[nd], 1.0))

            if not new_defaults:
                rounds = rnd + 1
                break

            default_sequence.append(new_defaults)

            # Propagate losses from newly defaulted nodes
            round_loss = 0.0
            for d_node in new_defaults:
                for pred in graph.predecessors(d_node):
                    if pred in defaulted:
                        continue
                    exposure = graph.edges[pred, d_node].get(self.weight_attr, 0.0)
                    loss = exposure * (1.0 - self.recovery_rate)
                    losses[pred] += loss
                    round_loss += loss

            losses_per_round.append(round_loss)
            rounds = rnd + 1

        # Compute system-level metrics
        total_system_value = sum(
            graph.nodes[nd].get(self.size_attr, 1.0) for nd in nodes
        )
        system_loss = sum(
            graph.nodes[nd].get(self.size_attr, 1.0)
            for nd in defaulted
        ) / total_system_value if total_system_value > 0 else 0.0

        survived = set(nodes) - defaulted

        return CascadeResult(
            initial_defaults=set(initial_defaults),
            final_defaults=defaulted,
            cascade_size=len(defaulted) - len(initial_defaults),
            total_defaults=len(defaulted),
            cascade_rounds=rounds,
            system_loss=system_loss,
            default_sequence=default_sequence,
            losses_per_round=losses_per_round,
            node_losses=losses,
            survived_nodes=survived,
            loss_given_default=lgd,
        )

    def get_cascade_size(
        self,
        graph: nx.DiGraph,
        initial_defaults: Set[int],
    ) -> int:
        """Get the total cascade size (number of defaults beyond initial).

        Args:
            graph: Financial network.
            initial_defaults: Initially defaulting nodes.

        Returns:
            Number of additional defaults caused by the cascade.
        """
        result = self.simulate_cascade(graph, initial_defaults)
        return result.cascade_size

    def find_tipping_points(
        self,
        graph: nx.DiGraph,
        n_samples: int = 50,
        shock_fractions: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> List[TippingPoint]:
        """Find tipping points in the cascade dynamics.

        Systematically increases the fraction of initially defaulting nodes
        to identify phase transitions where small increases in initial
        defaults lead to large jumps in cascade size.

        Args:
            graph: Financial network.
            n_samples: Number of shock fraction levels to test.
            shock_fractions: Specific fractions to test (overrides n_samples).
            seed: Random seed.

        Returns:
            List of TippingPoint objects identifying phase transitions.
        """
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        n = len(nodes)

        if shock_fractions is None:
            shock_fractions = np.linspace(0.01, 0.5, n_samples)

        results: List[TippingPoint] = []
        prev_cascade_frac = 0.0

        for frac in shock_fractions:
            n_shock = max(1, int(n * frac))
            # Average over multiple random selections
            cascade_fracs = []
            n_trials = min(20, max(5, n // 5))

            for _ in range(n_trials):
                shock_nodes = set(rng.choice(nodes, size=n_shock, replace=False).tolist())
                cascade_result = self.simulate_cascade(graph, shock_nodes)
                cascade_fracs.append(cascade_result.total_defaults / n)

            avg_cascade_frac = float(np.mean(cascade_fracs))
            jump = avg_cascade_frac - prev_cascade_frac
            is_tipping = jump > 0.15  # significant phase transition

            # Find critical nodes at this shock level
            critical = self._find_critical_nodes(graph, frac, rng, nodes)

            results.append(TippingPoint(
                shock_size=float(frac),
                cascade_fraction=avg_cascade_frac,
                is_tipping=is_tipping,
                critical_nodes=critical,
            ))

            prev_cascade_frac = avg_cascade_frac

        return results

    def compute_cascade_probability(
        self,
        graph: nx.DiGraph,
        shock_distribution: str = "uniform",
        n_simulations: int = 500,
        shock_params: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute cascade probability under a given shock distribution.

        Monte Carlo simulation to estimate the probability and expected size
        of cascades under random shocks.

        Args:
            graph: Financial network.
            shock_distribution: Distribution of shocks ('uniform', 'targeted', 'correlated').
            n_simulations: Number of Monte Carlo simulations.
            shock_params: Parameters for the shock distribution.
            seed: Random seed.

        Returns:
            Dictionary with cascade probability, expected size, and distribution.
        """
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        n = len(nodes)
        params = shock_params or {}

        cascade_sizes = []
        cascade_occurred = 0
        system_losses = []

        for _ in range(n_simulations):
            initial_defaults = self._sample_shocks(
                graph, nodes, shock_distribution, params, rng
            )

            if not initial_defaults:
                cascade_sizes.append(0)
                system_losses.append(0.0)
                continue

            result = self.simulate_cascade(graph, initial_defaults)
            cascade_sizes.append(result.cascade_size)
            system_losses.append(result.system_loss)

            if result.cascade_size > 0:
                cascade_occurred += 1

        cascade_sizes_arr = np.array(cascade_sizes)
        system_losses_arr = np.array(system_losses)

        return {
            "cascade_probability": cascade_occurred / n_simulations,
            "expected_cascade_size": float(cascade_sizes_arr.mean()),
            "median_cascade_size": float(np.median(cascade_sizes_arr)),
            "max_cascade_size": int(cascade_sizes_arr.max()),
            "std_cascade_size": float(cascade_sizes_arr.std()),
            "expected_system_loss": float(system_losses_arr.mean()),
            "tail_risk_95": float(np.percentile(system_losses_arr, 95)),
            "tail_risk_99": float(np.percentile(system_losses_arr, 99)),
            "cascade_size_distribution": cascade_sizes_arr.tolist(),
            "n_simulations": n_simulations,
        }

    def analyze_cascade_paths(
        self,
        graph: nx.DiGraph,
        cascade_result: CascadeResult,
    ) -> CascadePathAnalysis:
        """Analyse the contagion paths within a completed cascade.

        Reconstructs the causal paths through which defaults propagated,
        identifying critical edges and branching structure.

        Args:
            graph: Financial network.
            cascade_result: Result from simulate_cascade.

        Returns:
            CascadePathAnalysis with path structure details.
        """
        paths: List[List[int]] = []
        edge_counts: Dict[Tuple[int, int], int] = {}

        default_rounds = cascade_result.default_sequence
        all_defaulted = cascade_result.final_defaults

        # Reconstruct paths round by round
        for rnd in range(1, len(default_rounds)):
            current_defaults = default_rounds[rnd]
            prev_defaults = set()
            for r in range(rnd):
                prev_defaults |= default_rounds[r]

            for d_node in current_defaults:
                # Find which previously defaulted node(s) caused this default
                max_loss = 0.0
                cause_node = None
                for pred in graph.predecessors(d_node):
                    if pred in prev_defaults:
                        exposure = graph.edges[pred, d_node].get(self.weight_attr, 0.0)
                        loss = exposure * (1.0 - self.recovery_rate)
                        if loss > max_loss:
                            max_loss = loss
                            cause_node = pred

                if cause_node is not None:
                    path = [cause_node, d_node]
                    paths.append(path)
                    edge = (cause_node, d_node)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

        path_lengths = [len(p) for p in paths]
        mean_length = float(np.mean(path_lengths)) if path_lengths else 0.0
        max_length = max(path_lengths) if path_lengths else 0

        # Branching factor
        trigger_counts: Dict[int, int] = {}
        for p in paths:
            src = p[0]
            trigger_counts[src] = trigger_counts.get(src, 0) + 1
        branching = float(np.mean(list(trigger_counts.values()))) if trigger_counts else 0.0

        # Critical edges (most frequently on contagion paths)
        critical_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
        top_edges = [e for e, _ in critical_edges[:20]]

        return CascadePathAnalysis(
            paths=paths,
            path_lengths=path_lengths,
            mean_path_length=mean_length,
            max_path_length=max_length,
            branching_factor=branching,
            critical_edges=top_edges,
        )

    def _sample_shocks(
        self,
        graph: nx.DiGraph,
        nodes: List[int],
        distribution: str,
        params: Dict[str, float],
        rng: np.random.Generator,
    ) -> Set[int]:
        """Sample initial defaults from a shock distribution."""
        n = len(nodes)
        n_shocks = max(1, int(n * params.get("shock_fraction", 0.05)))

        if distribution == "uniform":
            indices = rng.choice(n, size=n_shocks, replace=False)
            return {nodes[i] for i in indices}

        elif distribution == "targeted":
            # Target the largest / most connected nodes
            sizes = np.array([
                graph.nodes[nd].get(self.size_attr, 1.0) for nd in nodes
            ])
            probs = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(n) / n
            indices = rng.choice(n, size=n_shocks, replace=False, p=probs)
            return {nodes[i] for i in indices}

        elif distribution == "correlated":
            # Correlated shocks: start with one node, then its neighbours
            start = rng.choice(nodes)
            defaults = {start}
            frontier = list(graph.successors(start)) + list(graph.predecessors(start))
            rng.shuffle(frontier)
            for nd in frontier:
                if len(defaults) >= n_shocks:
                    break
                if rng.random() < params.get("correlation", 0.7):
                    defaults.add(nd)
            return defaults

        return {nodes[rng.integers(0, n)]}

    def _find_critical_nodes(
        self,
        graph: nx.DiGraph,
        shock_fraction: float,
        rng: np.random.Generator,
        nodes: List[int],
    ) -> List[int]:
        """Find nodes whose inclusion in the shock set maximises cascade size."""
        n = len(nodes)
        n_shock = max(1, int(n * shock_fraction))
        node_importance: Dict[int, float] = {}

        # Test each node's marginal contribution
        n_tests = min(n, 20)
        test_nodes = rng.choice(nodes, size=n_tests, replace=False)

        for node in test_nodes:
            # Cascade with just this node
            result = self.simulate_cascade(graph, {node})
            node_importance[node] = result.cascade_size

        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        return [nd for nd, _ in sorted_nodes[:min(5, len(sorted_nodes))]]
