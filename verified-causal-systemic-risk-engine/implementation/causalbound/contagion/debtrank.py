"""
DebtRank Contagion Model
=========================

Implementation of the DebtRank algorithm for systemic risk assessment
in financial networks. Measures the fraction of total economic value
in the system that is potentially affected by the distress or default
of individual institutions or groups.

Variants implemented:
    - Linear DebtRank (Battiston et al. 2012)
    - Threshold DebtRank (fixed default threshold)
    - Non-linear DebtRank (concave/convex impact functions)

References:
    - Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G.
      (2012). DebtRank: Too central to fail? Financial networks, the FED
      and systemic risk. Scientific Reports, 2, 541.
    - Bardoscia, M., Battiston, S., Caccioli, F., & Caldarelli, G. (2015).
      DebtRank: A microscopic foundation for shock propagation.
      PLoS ONE, 10(6), e0130406.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


class DebtRankVariant(Enum):
    """DebtRank model variants."""
    LINEAR = "linear"
    THRESHOLD = "threshold"
    NONLINEAR = "nonlinear"


class NodeState(Enum):
    """State of a node during DebtRank propagation."""
    UNDISTRESSED = 0
    DISTRESSED = 1
    INACTIVE = 2  # already propagated its distress


@dataclass
class DebtRankResult:
    """Results from a DebtRank computation."""
    system_debtrank: float
    node_debtranks: Dict[int, float]
    final_distress: Dict[int, float]
    rounds_propagated: int
    total_loss: float
    loss_fraction: float
    cascade_size: int
    contagion_paths: Optional[List[List[int]]] = None
    round_history: Optional[List[Dict[int, float]]] = None


@dataclass
class SensitivityResult:
    """Results from DebtRank sensitivity analysis."""
    node_impacts: Dict[int, float]  # Impact of each node's individual failure
    top_k_nodes: List[Tuple[int, float]]  # Nodes ranked by systemic impact
    system_vulnerability: float  # Average impact across all single-node shocks
    concentration_risk: float  # How concentrated systemic risk is


class DebtRankModel:
    """DebtRank contagion model for financial networks.

    Computes the systemic impact of distress propagation through interbank
    exposure networks. Supports partial defaults where distress levels
    are continuous between 0 (healthy) and 1 (default).

    The model tracks three node states:
        - UNDISTRESSED: not yet affected by contagion
        - DISTRESSED: currently experiencing and propagating distress
        - INACTIVE: has already propagated distress (prevents double-counting)

    Example:
        >>> model = DebtRankModel()
        >>> shocks = {0: 1.0, 1: 0.5}  # node 0 defaults, node 1 50% distressed
        >>> result = model.compute(graph, shocks)
        >>> print(f"System DebtRank: {result.system_debtrank:.4f}")
    """

    def __init__(
        self,
        variant: DebtRankVariant = DebtRankVariant.LINEAR,
        default_threshold: float = 1.0,
        nonlinear_exponent: float = 1.0,
        weight_attr: str = "weight",
    ):
        """Initialise the DebtRank model.

        Args:
            variant: Which DebtRank variant to use.
            default_threshold: Distress level triggering default (threshold variant).
            nonlinear_exponent: Exponent for non-linear impact function.
            weight_attr: Edge attribute name for exposure weights.
        """
        self.variant = variant
        self.default_threshold = default_threshold
        self.nonlinear_exponent = nonlinear_exponent
        self.weight_attr = weight_attr

    def compute(
        self,
        graph: nx.DiGraph,
        initial_shocks: Dict[int, float],
        max_rounds: int = 100,
        track_history: bool = True,
    ) -> DebtRankResult:
        """Compute DebtRank for a given set of initial shocks.

        Args:
            graph: Financial network with exposure weights and capital attributes.
            initial_shocks: Mapping of node -> initial distress level (0 to 1).
            max_rounds: Maximum propagation rounds.
            track_history: Whether to record distress history per round.

        Returns:
            DebtRankResult with system-level and node-level metrics.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        node_to_idx = {nd: i for i, nd in enumerate(nodes)}

        # Initialise distress levels and states
        distress = np.zeros(n)
        states = np.full(n, NodeState.UNDISTRESSED.value)

        for node, shock in initial_shocks.items():
            if node in node_to_idx:
                idx = node_to_idx[node]
                distress[idx] = min(max(shock, 0.0), 1.0)
                states[idx] = NodeState.DISTRESSED.value

        # Build impact matrix W[i][j] = exposure(j->i) / capital(i)
        # W[i][j] is the fraction of i's capital lost if j defaults
        impact_matrix = self._build_impact_matrix(graph, nodes, node_to_idx)

        # Economic values (sizes) for weighting
        economic_values = np.array([
            graph.nodes[nd].get("size", 1.0) for nd in nodes
        ])
        total_value = economic_values.sum()
        if total_value == 0:
            total_value = 1.0

        history: List[Dict[int, float]] = []
        if track_history:
            history.append({nodes[i]: float(distress[i]) for i in range(n)})

        rounds = 0
        for rnd in range(max_rounds):
            new_distress = distress.copy()
            new_states = states.copy()

            has_change = False
            for i in range(n):
                if states[i] == NodeState.INACTIVE.value:
                    continue

                # Compute incoming distress from neighbours
                incoming = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    if states[j] == NodeState.DISTRESSED.value:
                        delta_h = self._impact_function(distress[j])
                        incoming += impact_matrix[i, j] * delta_h

                if incoming > 0:
                    new_val = min(1.0, distress[i] + incoming)
                    if new_val > distress[i] + 1e-10:
                        new_distress[i] = new_val
                        new_states[i] = NodeState.DISTRESSED.value
                        has_change = True

            # Transition distressed nodes to inactive
            for i in range(n):
                if states[i] == NodeState.DISTRESSED.value:
                    new_states[i] = NodeState.INACTIVE.value

            # Check for new distress above threshold (threshold variant)
            if self.variant == DebtRankVariant.THRESHOLD:
                for i in range(n):
                    if new_distress[i] >= self.default_threshold and distress[i] < self.default_threshold:
                        new_distress[i] = 1.0
                        new_states[i] = NodeState.DISTRESSED.value
                        has_change = True

            distress = new_distress
            states = new_states
            rounds = rnd + 1

            if track_history:
                history.append({nodes[i]: float(distress[i]) for i in range(n)})

            if not has_change:
                break

        # Compute system DebtRank
        final_distress_dict = {nodes[i]: float(distress[i]) for i in range(n)}
        system_loss = self.compute_system_loss(distress, economic_values)
        system_debtrank = system_loss / total_value

        # Node-level DebtRank (contribution to total)
        node_drs: Dict[int, float] = {}
        for i in range(n):
            if nodes[i] in initial_shocks:
                continue  # exclude initially shocked nodes
            node_drs[nodes[i]] = float(distress[i] * economic_values[i] / total_value)

        cascade_size = int(np.sum(distress > 0.0))

        return DebtRankResult(
            system_debtrank=float(system_debtrank),
            node_debtranks=node_drs,
            final_distress=final_distress_dict,
            rounds_propagated=rounds,
            total_loss=float(system_loss),
            loss_fraction=float(system_debtrank),
            cascade_size=cascade_size,
            round_history=history if track_history else None,
        )

    def propagate_round(
        self,
        distress_levels: Dict[int, float],
        graph: nx.DiGraph,
    ) -> Dict[int, float]:
        """Propagate one round of DebtRank distress.

        This is a convenience method for step-by-step simulation.

        Args:
            distress_levels: Current distress levels per node.
            graph: Financial network.

        Returns:
            Updated distress levels after one round.
        """
        nodes = list(graph.nodes())
        node_to_idx = {nd: i for i, nd in enumerate(nodes)}
        n = len(nodes)

        impact_matrix = self._build_impact_matrix(graph, nodes, node_to_idx)

        current = np.array([distress_levels.get(nd, 0.0) for nd in nodes])
        new_distress = current.copy()

        for i in range(n):
            incoming = 0.0
            for j in range(n):
                if i == j:
                    continue
                if current[j] > 0:
                    delta_h = self._impact_function(current[j])
                    incoming += impact_matrix[i, j] * delta_h

            new_distress[i] = min(1.0, current[i] + incoming)

        return {nodes[i]: float(new_distress[i]) for i in range(n)}

    def compute_system_loss(
        self,
        distress: np.ndarray,
        economic_values: np.ndarray,
    ) -> float:
        """Compute the total economic loss from distress levels.

        Args:
            distress: Array of distress levels (0 to 1) per node.
            economic_values: Economic value (size) of each node.

        Returns:
            Total weighted loss.
        """
        return float(np.sum(distress * economic_values))

    def get_contagion_paths(
        self,
        initial_shock: int,
        graph: nx.DiGraph,
        distress_threshold: float = 0.01,
        max_rounds: int = 50,
    ) -> List[List[int]]:
        """Trace contagion paths from an initial shock.

        Identifies all paths through which distress propagates from
        the initially shocked node to other institutions.

        Args:
            initial_shock: Node that receives the initial shock.
            graph: Financial network.
            distress_threshold: Minimum distress to consider as affected.
            max_rounds: Maximum propagation rounds.

        Returns:
            List of contagion paths (each path is a list of node IDs).
        """
        result = self.compute(
            graph, {initial_shock: 1.0}, max_rounds=max_rounds, track_history=True
        )

        if result.round_history is None:
            return []

        paths: List[List[int]] = []
        affected_nodes: Set[int] = set()

        for rnd in range(1, len(result.round_history)):
            prev = result.round_history[rnd - 1]
            curr = result.round_history[rnd]

            for node in curr:
                if node == initial_shock:
                    continue
                prev_dist = prev.get(node, 0.0)
                curr_dist = curr.get(node, 0.0)

                if curr_dist - prev_dist > distress_threshold and node not in affected_nodes:
                    affected_nodes.add(node)
                    # Find the predecessor that caused the distress
                    path = self._trace_path(
                        node, initial_shock, graph, result.round_history, rnd
                    )
                    if path:
                        paths.append(path)

        return paths

    def sensitivity_analysis(
        self,
        graph: nx.DiGraph,
        shock_level: float = 1.0,
        max_rounds: int = 50,
    ) -> SensitivityResult:
        """Perform sensitivity analysis of the network to individual shocks.

        Computes the systemic impact of each node's individual failure,
        identifying the most systemically important institutions.

        Args:
            graph: Financial network.
            shock_level: Distress level applied to each node (default=1.0 full failure).
            max_rounds: Maximum propagation rounds per simulation.

        Returns:
            SensitivityResult with node impacts and system vulnerability.
        """
        nodes = list(graph.nodes())
        node_impacts: Dict[int, float] = {}

        for node in nodes:
            result = self.compute(
                graph, {node: shock_level}, max_rounds=max_rounds, track_history=False
            )
            node_impacts[node] = result.system_debtrank

        # Rank nodes by impact
        sorted_impacts = sorted(node_impacts.items(), key=lambda x: x[1], reverse=True)
        system_vulnerability = float(np.mean(list(node_impacts.values())))

        # Concentration risk: how much of total impact comes from top 5
        impacts_array = np.array(sorted([node_impacts[n] for n in nodes], reverse=True))
        total_impact = impacts_array.sum()
        if total_impact > 0:
            top5_share = impacts_array[:min(5, len(nodes))].sum() / total_impact
        else:
            top5_share = 0.0

        return SensitivityResult(
            node_impacts=node_impacts,
            top_k_nodes=sorted_impacts[:20],
            system_vulnerability=system_vulnerability,
            concentration_risk=float(top5_share),
        )

    def _build_impact_matrix(
        self,
        graph: nx.DiGraph,
        nodes: List[int],
        node_to_idx: Dict[int, int],
    ) -> np.ndarray:
        """Build the impact matrix W where W[i,j] = exposure(j->i) / capital(i).

        This represents the fraction of node i's capital that would be lost
        if node j defaults completely.
        """
        n = len(nodes)
        W = np.zeros((n, n))

        for i, node_i in enumerate(nodes):
            capital_i = graph.nodes[node_i].get("capital", 1.0)
            if capital_i <= 0:
                capital_i = graph.nodes[node_i].get("size", 1.0) * 0.05

            for pred in graph.predecessors(node_i):
                if pred not in node_to_idx:
                    continue
                j = node_to_idx[pred]
                # In DebtRank, we look at the interbank asset of i on j
                # This is the edge weight from i to j (i lent to j)
                pass

            # Interbank assets: edges from node_i to others
            for succ in graph.successors(node_i):
                if succ not in node_to_idx:
                    continue
                j = node_to_idx[succ]
                exposure = graph.edges[node_i, succ].get(self.weight_attr, 0.0)
                # i's loss if j defaults: exposure / capital_i
                W[i, j] = min(exposure / capital_i, 1.0) if capital_i > 0 else 0.0

        return W

    def _impact_function(self, distress: float) -> float:
        """Compute the impact of a given distress level.

        Args:
            distress: Current distress level (0 to 1).

        Returns:
            Impact value.
        """
        if self.variant == DebtRankVariant.LINEAR:
            return distress
        elif self.variant == DebtRankVariant.THRESHOLD:
            return 1.0 if distress >= self.default_threshold else distress
        elif self.variant == DebtRankVariant.NONLINEAR:
            return distress ** self.nonlinear_exponent
        return distress

    def _trace_path(
        self,
        target: int,
        source: int,
        graph: nx.DiGraph,
        history: List[Dict[int, float]],
        target_round: int,
    ) -> List[int]:
        """Trace the contagion path from source to target using distress history."""
        if target_round <= 0:
            return [source, target]

        path = [target]
        current = target
        for rnd in range(target_round, 0, -1):
            prev_distress = history[rnd - 1]
            curr_distress = history[rnd]

            # Find predecessor with highest distress increase in previous round
            best_pred = None
            best_delta = 0.0
            for pred in graph.predecessors(current):
                if pred == current:
                    continue
                pred_prev = prev_distress.get(pred, 0.0)
                if rnd >= 2:
                    pred_before = history[rnd - 2].get(pred, 0.0)
                else:
                    pred_before = 0.0
                delta = pred_prev - pred_before
                if delta > best_delta or pred == source:
                    best_delta = delta
                    best_pred = pred

            if best_pred is not None:
                path.append(best_pred)
                current = best_pred
                if current == source:
                    break
            else:
                break

        path.reverse()
        return path
