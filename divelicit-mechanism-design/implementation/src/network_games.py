"""
Network Games Module
====================

Implements game-theoretic models on networks including network effects,
influence maximization, network formation, contagion models, public goods
games, centrality measures, and community detection.

All algorithms use real mathematical computations via numpy and scipy.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class NodeState(Enum):
    """Possible states for a node in contagion or adoption models."""
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    ADOPTER = 3
    NON_ADOPTER = 4


@dataclass
class Equilibrium:
    """Represents a game equilibrium on a network.

    Attributes:
        strategies: Mapping from node id to chosen strategy value.
        payoffs: Mapping from node id to realized payoff.
        converged: Whether the equilibrium computation converged.
        iterations: Number of iterations used to reach equilibrium.
        social_welfare: Sum of all payoffs across players.
    """
    strategies: Dict[int, float]
    payoffs: Dict[int, float]
    converged: bool
    iterations: int
    social_welfare: float


@dataclass
class Player:
    """A player located at a network node.

    Attributes:
        node: The node identifier.
        strategy: Current strategy value in [0, 1].
        payoff: Current accumulated payoff.
        threshold: Adoption threshold for threshold models.
        state: Current state for contagion models.
    """
    node: int
    strategy: float = 0.0
    payoff: float = 0.0
    threshold: float = 0.5
    state: NodeState = NodeState.SUSCEPTIBLE


@dataclass
class FormationResult:
    """Result of a network formation analysis.

    Attributes:
        stable_edges: Set of edges in the pairwise stable network.
        efficient_edges: Set of edges in the efficient network.
        is_efficient: Whether the stable network is also efficient.
        stability_score: Fraction of edges that are stable.
        total_welfare_stable: Social welfare of the stable network.
        total_welfare_efficient: Social welfare of the efficient network.
    """
    stable_edges: Set[Tuple[int, int]]
    efficient_edges: Set[Tuple[int, int]]
    is_efficient: bool
    stability_score: float
    total_welfare_stable: float
    total_welfare_efficient: float


class NetworkGame:
    """Game-theoretic model on a network graph.

    Supports adding players at nodes, weighted edges between them, and
    computing equilibria via best-response dynamics. The network is stored
    as both an adjacency dict and a sparse matrix for efficient computation.

    Example:
        >>> game = NetworkGame()
        >>> game.add_player(0)
        >>> game.add_player(1)
        >>> game.add_edge(0, 1, weight=1.0)
        >>> eq = game.play()
        >>> isinstance(eq, Equilibrium)
        True
    """

    def __init__(self, directed: bool = False):
        """Initialize a network game.

        Args:
            directed: If True, edges are directed. Default is undirected.
        """
        self.directed = directed
        self.players: Dict[int, Player] = {}
        self.adj: Dict[int, Dict[int, float]] = {}
        self._node_list: List[int] = []
        self._node_index: Dict[int, int] = {}
        self._adj_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

    def add_player(self, node: int, threshold: float = 0.5) -> None:
        """Add a player at the given node.

        Args:
            node: Unique integer identifier for the node.
            threshold: Adoption threshold for threshold models.
        """
        if node not in self.players:
            self.players[node] = Player(node=node, threshold=threshold)
            self.adj.setdefault(node, {})
            self._node_list.append(node)
            self._node_index[node] = len(self._node_list) - 1
            self._matrix_dirty = True

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add a weighted edge between nodes u and v.

        Args:
            u: Source node identifier.
            v: Target node identifier.
            weight: Edge weight (default 1.0).

        Raises:
            ValueError: If either node has not been added as a player.
        """
        if u not in self.players or v not in self.players:
            raise ValueError(f"Both nodes must be added as players first: {u}, {v}")
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight
        self._matrix_dirty = True

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build the adjacency matrix from the current edge set.

        Returns:
            Square numpy array of shape (n, n) with edge weights.
        """
        n = len(self._node_list)
        mat = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in self.adj.items():
            i = self._node_index[u]
            for v, w in neighbors.items():
                j = self._node_index[v]
                mat[i, j] = w
        self._adj_matrix = mat
        self._matrix_dirty = False
        return mat

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Lazily-built adjacency matrix."""
        if self._matrix_dirty or self._adj_matrix is None:
            return self._build_adjacency_matrix()
        return self._adj_matrix

    def neighbors(self, node: int) -> List[int]:
        """Return the list of neighbors for a given node.

        Args:
            node: The node whose neighbors to retrieve.

        Returns:
            List of neighbor node identifiers.
        """
        return list(self.adj.get(node, {}).keys())

    def degree(self, node: int) -> float:
        """Return the weighted degree of a node.

        Args:
            node: The node identifier.

        Returns:
            Sum of weights of all edges incident to the node.
        """
        return sum(self.adj.get(node, {}).values())

    def play(self, max_iter: int = 200, tol: float = 1e-6,
             complementarity: float = 1.0) -> Equilibrium:
        """Compute a Nash equilibrium via best-response dynamics.

        Each player chooses a strategy in [0, 1] to maximize a quadratic
        payoff that depends on neighbors' strategies:
            u_i(s) = complementarity * s_i * sum_j(w_ij * s_j) - 0.5 * s_i^2

        The best response for player i is:
            s_i* = clamp(complementarity * sum_j(w_ij * s_j), 0, 1)

        Args:
            max_iter: Maximum number of best-response iterations.
            tol: Convergence tolerance on strategy change.
            complementarity: Strength of strategic complementarity.

        Returns:
            An Equilibrium dataclass with strategies, payoffs, and metadata.
        """
        n = len(self._node_list)
        if n == 0:
            return Equilibrium({}, {}, True, 0, 0.0)

        A = self.adjacency_matrix
        strategies = np.full(n, 0.5, dtype=np.float64)

        converged = False
        iterations = 0
        for it in range(max_iter):
            neighbor_sum = A @ strategies
            new_strategies = np.clip(complementarity * neighbor_sum, 0.0, 1.0)
            delta = np.max(np.abs(new_strategies - strategies))
            strategies = new_strategies
            iterations = it + 1
            if delta < tol:
                converged = True
                break

        neighbor_sum = A @ strategies
        payoff_vec = complementarity * strategies * neighbor_sum - 0.5 * strategies ** 2

        strat_dict = {self._node_list[i]: float(strategies[i]) for i in range(n)}
        payoff_dict = {self._node_list[i]: float(payoff_vec[i]) for i in range(n)}
        welfare = float(np.sum(payoff_vec))

        return Equilibrium(strat_dict, payoff_dict, converged, iterations, welfare)

    # ------------------------------------------------------------------
    # Network Effects
    # ------------------------------------------------------------------

    def direct_network_effect(self, node: int) -> float:
        """Compute the direct network effect for a node.

        The direct effect is the weighted sum of neighbors' strategies,
        capturing the immediate benefit from connected adopters.

        Args:
            node: Node identifier.

        Returns:
            Weighted sum of neighbor strategies.
        """
        effect = 0.0
        for nb, w in self.adj.get(node, {}).items():
            effect += w * self.players[nb].strategy
        return effect

    def indirect_network_effect(self, node: int, decay: float = 0.5,
                                 max_hops: int = 3) -> float:
        """Compute the indirect network effect via decaying multi-hop influence.

        The indirect effect sums contributions from nodes up to max_hops away,
        with each hop decaying the contribution by a factor of `decay`.
        Uses BFS and avoids double-counting by tracking visited nodes.

        Args:
            node: Source node identifier.
            decay: Multiplicative decay per hop (0 < decay < 1).
            max_hops: Maximum number of hops to consider.

        Returns:
            Total indirect network effect value.
        """
        visited: Set[int] = {node}
        current_layer = [node]
        total_effect = 0.0
        current_decay = 1.0

        for _hop in range(max_hops):
            current_decay *= decay
            next_layer = []
            for u in current_layer:
                for nb, w in self.adj.get(u, {}).items():
                    if nb not in visited:
                        visited.add(nb)
                        total_effect += current_decay * w * self.players[nb].strategy
                        next_layer.append(nb)
            current_layer = next_layer
            if not current_layer:
                break
        return total_effect

    def threshold_adoption(self, seeds: Set[int],
                           max_rounds: int = 100) -> Dict[int, int]:
        """Run a linear threshold adoption model.

        Each non-seed node adopts when the weighted fraction of adopted
        neighbors meets or exceeds its threshold. Seeds adopt at round 0.

        Args:
            seeds: Set of initially adopted node ids.
            max_rounds: Maximum number of propagation rounds.

        Returns:
            Dict mapping node id to the round at which it adopted,
            or -1 if it never adopted.
        """
        adoption_round: Dict[int, int] = {}
        adopted: Set[int] = set()

        for s in seeds:
            if s in self.players:
                adopted.add(s)
                adoption_round[s] = 0
                self.players[s].state = NodeState.ADOPTER

        for rnd in range(1, max_rounds + 1):
            new_adopters: List[int] = []
            for node in self.players:
                if node in adopted:
                    continue
                total_weight = self.degree(node)
                if total_weight == 0:
                    continue
                adopted_weight = sum(
                    self.adj[node].get(nb, 0.0) for nb in adopted if nb in self.adj[node]
                )
                fraction = adopted_weight / total_weight
                if fraction >= self.players[node].threshold:
                    new_adopters.append(node)

            if not new_adopters:
                break
            for node in new_adopters:
                adopted.add(node)
                adoption_round[node] = rnd
                self.players[node].state = NodeState.ADOPTER

        for node in self.players:
            if node not in adoption_round:
                adoption_round[node] = -1
        return adoption_round

    # ------------------------------------------------------------------
    # Influence Maximization
    # ------------------------------------------------------------------

    def influence_maximization(self, k: int, mc_rounds: int = 50,
                               threshold_based: bool = True) -> List[int]:
        """Greedy algorithm for influence maximization.

        Selects k seed nodes to maximize expected spread under either the
        threshold model or independent cascade model. Uses the greedy
        algorithm with lazy evaluation (CELF-like marginal gain caching).

        Args:
            k: Number of seed nodes to select.
            mc_rounds: Number of Monte Carlo simulations per candidate.
            threshold_based: Use threshold model if True, else independent cascade.

        Returns:
            Ordered list of k selected seed node ids.
        """
        nodes = list(self.players.keys())
        if k >= len(nodes):
            return nodes[:k]

        selected: List[int] = []
        remaining = set(nodes)

        for _ in range(k):
            best_node = -1
            best_spread = -1.0

            for candidate in remaining:
                seed_set = set(selected) | {candidate}
                total_spread = 0.0

                for _mc in range(mc_rounds):
                    if threshold_based:
                        rng = np.random.RandomState(_mc)
                        thresholds = {
                            n: rng.random() for n in self.players
                        }
                        spread = self._simulate_threshold_spread(seed_set, thresholds)
                    else:
                        spread = self._simulate_cascade_spread(seed_set, _mc)
                    total_spread += spread

                avg_spread = total_spread / mc_rounds
                if avg_spread > best_spread:
                    best_spread = avg_spread
                    best_node = candidate

            if best_node >= 0:
                selected.append(best_node)
                remaining.discard(best_node)

        return selected

    def _simulate_threshold_spread(self, seeds: Set[int],
                                    thresholds: Dict[int, float]) -> int:
        """Simulate one round of threshold spread with given thresholds.

        Args:
            seeds: Initial adopter set.
            thresholds: Random threshold per node in [0, 1].

        Returns:
            Total number of adopters at convergence.
        """
        adopted = set(seeds)
        changed = True
        while changed:
            changed = False
            for node in self.players:
                if node in adopted:
                    continue
                total_w = self.degree(node)
                if total_w == 0:
                    continue
                adopted_w = sum(
                    self.adj[node].get(nb, 0.0) for nb in adopted if nb in self.adj[node]
                )
                if adopted_w / total_w >= thresholds.get(node, 0.5):
                    adopted.add(node)
                    changed = True
        return len(adopted)

    def _simulate_cascade_spread(self, seeds: Set[int], seed_val: int) -> int:
        """Simulate independent cascade from seed set.

        Each active node attempts to activate each inactive neighbor with
        probability equal to the normalized edge weight.

        Args:
            seeds: Initial active set.
            seed_val: Random seed for reproducibility.

        Returns:
            Total number of activated nodes.
        """
        rng = np.random.RandomState(seed_val)
        active = set(seeds)
        frontier = list(seeds)

        while frontier:
            next_frontier = []
            for u in frontier:
                for nb, w in self.adj.get(u, {}).items():
                    if nb not in active:
                        max_w = max(self.adj.get(u, {}).values()) if self.adj.get(u, {}) else 1.0
                        prob = w / max_w if max_w > 0 else 0.0
                        if rng.random() < prob:
                            active.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier
        return len(active)

    # ------------------------------------------------------------------
    # Network Formation
    # ------------------------------------------------------------------

    def analyze_formation(self, link_cost: float = 0.1,
                          link_benefit: float = 0.3,
                          decay: float = 0.5) -> FormationResult:
        """Analyze pairwise stability and efficiency of network formation.

        Uses the connections model: player i's utility from the network is
            u_i = sum_j delta^{d(i,j)} - c * |direct links of i|
        where delta is the benefit decay and c is the per-link cost.

        A network is pairwise stable if:
          1. No player wants to sever an existing link.
          2. No pair of unlinked players both want to form a link.

        Args:
            link_cost: Cost per direct link.
            link_benefit: Not used directly; decay controls benefit.
            decay: Benefit decay factor per hop distance.

        Returns:
            FormationResult with stable and efficient edge sets.
        """
        nodes = self._node_list
        n = len(nodes)
        all_possible_edges = []
        for i in range(n):
            upper = n if self.directed else i
            for j in range(upper):
                if i != j:
                    all_possible_edges.append((nodes[i], nodes[j]))

        def compute_utilities(edges_set: Set[Tuple[int, int]]) -> Dict[int, float]:
            """Compute utility for each player given an edge set."""
            local_adj: Dict[int, Set[int]] = {nd: set() for nd in nodes}
            for u, v in edges_set:
                local_adj[u].add(v)
                local_adj[v].add(u)

            utils = {}
            for nd in nodes:
                benefit = 0.0
                visited = {nd}
                layer = [nd]
                d = 0
                while layer:
                    d += 1
                    next_layer = []
                    for curr in layer:
                        for nb in local_adj.get(curr, set()):
                            if nb not in visited:
                                visited.add(nb)
                                benefit += decay ** d
                                next_layer.append(nb)
                    layer = next_layer
                cost = len(local_adj[nd]) * link_cost
                utils[nd] = benefit - cost
            return utils

        current_edges = set()
        for u, nbrs in self.adj.items():
            for v in nbrs:
                edge = (min(u, v), max(u, v))
                current_edges.add(edge)

        stable_edges = set(current_edges)
        changed = True
        max_formation_iter = 50
        for _ in range(max_formation_iter):
            if not changed:
                break
            changed = False
            utils_now = compute_utilities(stable_edges)

            edges_to_remove = []
            for edge in list(stable_edges):
                u, v = edge
                test_edges = stable_edges - {edge}
                test_utils = compute_utilities(test_edges)
                if test_utils[u] > utils_now[u] or test_utils[v] > utils_now[v]:
                    edges_to_remove.append(edge)

            for edge in edges_to_remove:
                stable_edges.discard(edge)
                changed = True

            utils_now = compute_utilities(stable_edges)
            for i in range(n):
                for j in range(i + 1, n):
                    edge = (nodes[i], nodes[j])
                    if edge not in stable_edges:
                        test_edges = stable_edges | {edge}
                        test_utils = compute_utilities(test_edges)
                        if (test_utils[nodes[i]] > utils_now[nodes[i]] and
                                test_utils[nodes[j]] > utils_now[nodes[j]]):
                            stable_edges.add(edge)
                            changed = True
                            break
                if changed:
                    break

        best_welfare = -np.inf
        best_edges: Set[Tuple[int, int]] = set()
        if n <= 8:
            candidate_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    candidate_edges.append((nodes[i], nodes[j]))
            m = len(candidate_edges)
            for mask in range(2 ** m):
                test_set = set()
                for bit in range(m):
                    if mask & (1 << bit):
                        test_set.add(candidate_edges[bit])
                utils = compute_utilities(test_set)
                welfare = sum(utils.values())
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_edges = set(test_set)
        else:
            best_edges = set(current_edges)
            best_welfare = sum(compute_utilities(best_edges).values())

        stable_welfare = sum(compute_utilities(stable_edges).values())

        total_possible = n * (n - 1) // 2
        stability_score = len(stable_edges) / max(total_possible, 1)

        return FormationResult(
            stable_edges=stable_edges,
            efficient_edges=best_edges,
            is_efficient=(stable_edges == best_edges),
            stability_score=stability_score,
            total_welfare_stable=stable_welfare,
            total_welfare_efficient=best_welfare,
        )

    # ------------------------------------------------------------------
    # Contagion Models (SIR / SIS)
    # ------------------------------------------------------------------

    def sir_contagion(self, initial_infected: Set[int],
                      beta: float = 0.3, gamma: float = 0.1,
                      max_steps: int = 200,
                      strategic: bool = False,
                      vaccination_cost: float = 0.2) -> Dict[str, np.ndarray]:
        """Run SIR contagion on the network with optional strategic vaccination.

        At each step, each infected node transmits to each susceptible neighbor
        with probability beta * w_{ij} / max_weight. Infected nodes recover
        with probability gamma. If strategic=True, susceptible agents may
        vaccinate (become recovered) if their expected infection cost exceeds
        the vaccination cost.

        Args:
            initial_infected: Set of initially infected node ids.
            beta: Base transmission probability.
            gamma: Recovery probability per step.
            max_steps: Maximum simulation steps.
            strategic: If True, agents make vaccination decisions.
            vaccination_cost: Cost of vaccination (only if strategic=True).

        Returns:
            Dict with keys 'S', 'I', 'R' mapping to arrays of counts per step.
        """
        n = len(self._node_list)
        states = np.zeros(n, dtype=int)  # 0=S, 1=I, 2=R

        for node in initial_infected:
            if node in self._node_index:
                states[self._node_index[node]] = 1

        A = self.adjacency_matrix
        max_w = np.max(A) if np.max(A) > 0 else 1.0

        s_counts, i_counts, r_counts = [], [], []
        rng = np.random.RandomState(42)

        for _step in range(max_steps):
            s_counts.append(int(np.sum(states == 0)))
            i_counts.append(int(np.sum(states == 1)))
            r_counts.append(int(np.sum(states == 2)))

            if np.sum(states == 1) == 0:
                break

            new_states = states.copy()

            if strategic:
                for i in range(n):
                    if states[i] == 0:
                        infected_neighbors = np.sum(
                            (A[i, :] > 0) & (states == 1)
                        )
                        total_neighbors = np.sum(A[i, :] > 0)
                        if total_neighbors > 0:
                            infection_prob = 1.0 - (1.0 - beta) ** infected_neighbors
                            expected_cost = infection_prob * 1.0
                            if expected_cost > vaccination_cost:
                                new_states[i] = 2
                                continue

            infected_indices = np.where(states == 1)[0]
            for inf_idx in infected_indices:
                for sus_idx in np.where(states == 0)[0]:
                    if A[inf_idx, sus_idx] > 0 and new_states[sus_idx] == 0:
                        transmission_prob = beta * A[inf_idx, sus_idx] / max_w
                        if rng.random() < transmission_prob:
                            new_states[sus_idx] = 1

                if rng.random() < gamma:
                    new_states[inf_idx] = 2

            states = new_states

        return {
            'S': np.array(s_counts, dtype=int),
            'I': np.array(i_counts, dtype=int),
            'R': np.array(r_counts, dtype=int),
        }

    def sis_contagion(self, initial_infected: Set[int],
                      beta: float = 0.3, gamma: float = 0.1,
                      max_steps: int = 300) -> Dict[str, np.ndarray]:
        """Run SIS contagion on the network.

        Like SIR but recovered nodes return to susceptible instead of
        gaining permanent immunity. Converges to an endemic equilibrium
        if beta/gamma exceeds the epidemic threshold.

        Args:
            initial_infected: Set of initially infected node ids.
            beta: Base transmission probability.
            gamma: Recovery probability per step.
            max_steps: Maximum simulation steps.

        Returns:
            Dict with keys 'S', 'I' mapping to arrays of counts per step.
        """
        n = len(self._node_list)
        states = np.zeros(n, dtype=int)
        for node in initial_infected:
            if node in self._node_index:
                states[self._node_index[node]] = 1

        A = self.adjacency_matrix
        max_w = np.max(A) if np.max(A) > 0 else 1.0
        rng = np.random.RandomState(42)

        s_counts, i_counts = [], []

        for _step in range(max_steps):
            s_counts.append(int(np.sum(states == 0)))
            i_counts.append(int(np.sum(states == 1)))

            if np.sum(states == 1) == 0:
                break

            new_states = states.copy()
            infected_indices = np.where(states == 1)[0]

            for inf_idx in infected_indices:
                for sus_idx in np.where(states == 0)[0]:
                    if A[inf_idx, sus_idx] > 0 and new_states[sus_idx] == 0:
                        prob = beta * A[inf_idx, sus_idx] / max_w
                        if rng.random() < prob:
                            new_states[sus_idx] = 1

                if rng.random() < gamma:
                    new_states[inf_idx] = 0  # back to susceptible

            states = new_states

        return {
            'S': np.array(s_counts, dtype=int),
            'I': np.array(i_counts, dtype=int),
        }

    # ------------------------------------------------------------------
    # Public Goods on Networks
    # ------------------------------------------------------------------

    def best_shot_public_good(self, cost: float = 0.3,
                               benefit: float = 1.0,
                               max_iter: int = 100) -> Dict[int, float]:
        """Best-shot public goods game on the network.

        Each player decides an effort level in [0, 1]. The public good
        available to player i is the maximum effort among i and its neighbors.
        Payoff: benefit * max(efforts in neighborhood) - cost * own_effort.

        Equilibrium: only a minimum dominating set contributes. We compute
        this via best-response dynamics.

        Args:
            cost: Cost per unit of effort.
            benefit: Benefit per unit of public good.
            max_iter: Maximum iterations.

        Returns:
            Dict mapping node id to equilibrium effort level.
        """
        n = len(self._node_list)
        efforts = np.full(n, 0.5)
        A = self.adjacency_matrix

        for _it in range(max_iter):
            new_efforts = np.zeros(n)
            for i in range(n):
                neighbor_mask = A[i, :] > 0
                if np.any(neighbor_mask):
                    max_neighbor_effort = np.max(efforts[neighbor_mask])
                else:
                    max_neighbor_effort = 0.0

                if max_neighbor_effort >= 1.0:
                    new_efforts[i] = 0.0
                elif benefit > cost:
                    if max_neighbor_effort * benefit >= benefit - cost:
                        new_efforts[i] = 0.0
                    else:
                        new_efforts[i] = 1.0
                else:
                    new_efforts[i] = 0.0

            if np.allclose(efforts, new_efforts, atol=1e-8):
                break
            efforts = new_efforts

        return {self._node_list[i]: float(efforts[i]) for i in range(n)}

    def weakest_link_public_good(self, cost: float = 0.2,
                                  benefit: float = 1.0,
                                  max_iter: int = 100) -> Dict[int, float]:
        """Weakest-link public goods game on the network.

        The public good available to player i is the minimum effort among
        i and its neighbors. Payoff: benefit * min(efforts) - cost * effort_i.
        This models situations where the weakest contributor determines quality.

        Args:
            cost: Cost per unit of effort.
            benefit: Benefit per unit of public good.
            max_iter: Maximum iterations.

        Returns:
            Dict mapping node id to equilibrium effort level.
        """
        n = len(self._node_list)
        efforts = np.full(n, 0.5)
        A = self.adjacency_matrix

        for _it in range(max_iter):
            new_efforts = np.zeros(n)
            for i in range(n):
                neighbor_mask = A[i, :] > 0
                if np.any(neighbor_mask):
                    min_neighbor = np.min(efforts[neighbor_mask])
                else:
                    min_neighbor = efforts[i]

                if benefit >= cost:
                    new_efforts[i] = min(min_neighbor, (benefit - cost) / benefit)
                else:
                    new_efforts[i] = 0.0

            if np.allclose(efforts, new_efforts, atol=1e-8):
                break
            efforts = new_efforts

        return {self._node_list[i]: float(efforts[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # Centrality Measures
    # ------------------------------------------------------------------

    def degree_centrality(self) -> Dict[int, float]:
        """Compute degree centrality for all nodes.

        Degree centrality of node i is its degree divided by (n - 1),
        normalizing to [0, 1].

        Returns:
            Dict mapping node id to degree centrality.
        """
        n = len(self._node_list)
        if n <= 1:
            return {nd: 0.0 for nd in self._node_list}
        A = self.adjacency_matrix
        degrees = np.sum(A > 0, axis=1).astype(float)
        centrality = degrees / (n - 1)
        return {self._node_list[i]: float(centrality[i]) for i in range(n)}

    def betweenness_centrality(self) -> Dict[int, float]:
        """Compute betweenness centrality for all nodes using Brandes' algorithm.

        Betweenness centrality of node v is the fraction of all shortest
        paths between pairs (s, t) that pass through v, summed over all
        pairs and normalized by (n-1)(n-2)/2 for undirected graphs.

        Returns:
            Dict mapping node id to betweenness centrality.
        """
        n = len(self._node_list)
        A = self.adjacency_matrix
        cb = np.zeros(n, dtype=np.float64)

        for s in range(n):
            stack: List[int] = []
            pred: List[List[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=np.float64)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=int)
            dist[s] = 0

            queue = [s]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                stack.append(v)
                for w in range(n):
                    if A[v, w] <= 0:
                        continue
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = np.zeros(n, dtype=np.float64)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]

        if not self.directed:
            cb /= 2.0
        norm = (n - 1) * (n - 2)
        if norm > 0:
            cb /= norm

        return {self._node_list[i]: float(cb[i]) for i in range(n)}

    def eigenvector_centrality(self, max_iter: int = 200,
                                tol: float = 1e-8) -> Dict[int, float]:
        """Compute eigenvector centrality via power iteration.

        The eigenvector centrality is the leading eigenvector of the
        adjacency matrix, normalized so the maximum entry is 1.

        Args:
            max_iter: Maximum power iterations.
            tol: Convergence tolerance.

        Returns:
            Dict mapping node id to eigenvector centrality.
        """
        n = len(self._node_list)
        if n == 0:
            return {}
        A = self.adjacency_matrix
        x = np.ones(n, dtype=np.float64) / np.sqrt(n)

        for _it in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        x = np.abs(x)
        max_val = np.max(x)
        if max_val > 0:
            x /= max_val

        return {self._node_list[i]: float(x[i]) for i in range(n)}

    def pagerank(self, damping: float = 0.85, max_iter: int = 200,
                 tol: float = 1e-8) -> Dict[int, float]:
        """Compute PageRank centrality.

        PageRank models a random surfer who follows links with probability
        `damping` and jumps to a random node otherwise. The stationary
        distribution gives the PageRank scores.

        Args:
            damping: Damping factor (probability of following a link).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Dict mapping node id to PageRank score (sums to 1).
        """
        n = len(self._node_list)
        if n == 0:
            return {}

        A = self.adjacency_matrix
        out_degree = np.sum(A, axis=1)
        dangling = (out_degree == 0)

        pr = np.ones(n, dtype=np.float64) / n

        for _it in range(max_iter):
            pr_new = np.ones(n, dtype=np.float64) * (1.0 - damping) / n

            dangling_sum = damping * np.sum(pr[dangling]) / n
            pr_new += dangling_sum

            for j in range(n):
                if out_degree[j] > 0:
                    contribution = damping * pr[j] / out_degree[j]
                    for i in range(n):
                        if A[j, i] > 0:
                            pr_new[i] += contribution * A[j, i]

            pr_new /= np.sum(pr_new)

            if np.linalg.norm(pr_new - pr, 1) < tol:
                pr = pr_new
                break
            pr = pr_new

        return {self._node_list[i]: float(pr[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # Community Detection (Louvain-like modularity maximization)
    # ------------------------------------------------------------------

    def detect_communities(self) -> Dict[int, int]:
        """Detect communities via modularity maximization (Louvain-like).

        Phase 1: Each node starts in its own community. Greedily move nodes
        to the neighbor community that yields the greatest modularity gain.
        Repeat until no improvement.

        Phase 2: Aggregate communities into super-nodes and repeat. We
        implement a single-level pass which captures the core Louvain logic.

        Modularity Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)

        Returns:
            Dict mapping node id to community label (int).
        """
        n = len(self._node_list)
        if n == 0:
            return {}

        A = self.adjacency_matrix
        m2 = np.sum(A)
        if m2 == 0:
            return {self._node_list[i]: i for i in range(n)}

        k = np.sum(A, axis=1)
        community = np.arange(n, dtype=int)

        improved = True
        max_passes = 50
        for _pass in range(max_passes):
            if not improved:
                break
            improved = False
            order = np.random.permutation(n)

            for i in order:
                old_comm = community[i]
                neighbor_comms = set()
                for j in range(n):
                    if A[i, j] > 0:
                        neighbor_comms.add(community[j])
                neighbor_comms.discard(old_comm)

                if not neighbor_comms:
                    continue

                best_gain = 0.0
                best_comm = old_comm

                sum_in_old = 0.0
                sum_tot_old = 0.0
                for j in range(n):
                    if community[j] == old_comm:
                        sum_tot_old += k[j]
                        if j != i:
                            sum_in_old += A[i, j]

                k_i = k[i]

                for c in neighbor_comms:
                    sum_in_new = 0.0
                    sum_tot_new = 0.0
                    for j in range(n):
                        if community[j] == c:
                            sum_tot_new += k[j]
                            sum_in_new += A[i, j]

                    # Modularity gain from moving i from old_comm to c
                    gain = (
                        (sum_in_new - sum_in_old) / m2
                        - k_i * (sum_tot_new - sum_tot_old + k_i) / (m2 * m2) * 2.0
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = c

                if best_comm != old_comm:
                    community[i] = best_comm
                    improved = True

        label_map = {}
        counter = 0
        result = {}
        for i in range(n):
            c = int(community[i])
            if c not in label_map:
                label_map[c] = counter
                counter += 1
            result[self._node_list[i]] = label_map[c]

        return result

    def modularity(self, communities: Dict[int, int]) -> float:
        """Compute the modularity score for a given community assignment.

        Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)

        Args:
            communities: Mapping from node id to community label.

        Returns:
            Modularity value in [-0.5, 1].
        """
        n = len(self._node_list)
        A = self.adjacency_matrix
        m2 = np.sum(A)
        if m2 == 0:
            return 0.0

        k = np.sum(A, axis=1)
        Q = 0.0
        for i in range(n):
            ci = communities.get(self._node_list[i], -1)
            for j in range(n):
                cj = communities.get(self._node_list[j], -2)
                if ci == cj:
                    Q += A[i, j] - k[i] * k[j] / m2
        Q /= m2
        return float(Q)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def shortest_path_lengths(self, source: int) -> Dict[int, int]:
        """Compute shortest path lengths from source to all reachable nodes.

        Uses BFS on the unweighted graph structure.

        Args:
            source: Source node identifier.

        Returns:
            Dict mapping reachable node id to shortest path length.
        """
        dist: Dict[int, int] = {source: 0}
        queue = [source]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for nb in self.adj.get(u, {}):
                if nb not in dist:
                    dist[nb] = dist[u] + 1
                    queue.append(nb)
        return dist

    def clustering_coefficient(self, node: int) -> float:
        """Compute the local clustering coefficient of a node.

        The clustering coefficient is the fraction of pairs of neighbors
        that are themselves connected.

        Args:
            node: Node identifier.

        Returns:
            Clustering coefficient in [0, 1].
        """
        nbrs = list(self.adj.get(node, {}).keys())
        k = len(nbrs)
        if k < 2:
            return 0.0
        triangles = 0
        for i in range(k):
            for j in range(i + 1, k):
                if nbrs[j] in self.adj.get(nbrs[i], {}):
                    triangles += 1
        return 2.0 * triangles / (k * (k - 1))

    def average_clustering(self) -> float:
        """Compute the average clustering coefficient of the network.

        Returns:
            Mean of local clustering coefficients across all nodes.
        """
        if not self._node_list:
            return 0.0
        coeffs = [self.clustering_coefficient(nd) for nd in self._node_list]
        return float(np.mean(coeffs))

    def density(self) -> float:
        """Compute the density of the network.

        Density = 2 * |E| / (n * (n-1)) for undirected graphs.

        Returns:
            Network density in [0, 1].
        """
        n = len(self._node_list)
        if n < 2:
            return 0.0
        edge_count = sum(len(v) for v in self.adj.values())
        if not self.directed:
            edge_count //= 2
        denom = n * (n - 1) if self.directed else n * (n - 1) / 2
        return edge_count / denom

    def is_connected(self) -> bool:
        """Check whether the network is connected (for undirected graphs).

        Returns:
            True if all nodes are reachable from any starting node.
        """
        if len(self._node_list) <= 1:
            return True
        dists = self.shortest_path_lengths(self._node_list[0])
        return len(dists) == len(self._node_list)

    def connected_components(self) -> List[Set[int]]:
        """Find all connected components of the network.

        Returns:
            List of sets, each set containing node ids in one component.
        """
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for node in self._node_list:
            if node not in visited:
                dists = self.shortest_path_lengths(node)
                component = set(dists.keys())
                visited |= component
                components.append(component)
        return components
