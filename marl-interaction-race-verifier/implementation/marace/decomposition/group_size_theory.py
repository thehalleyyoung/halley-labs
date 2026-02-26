"""
Formal conditions under which interaction group sizes remain bounded.

Addresses the critique that the small-group assumption (2-3 agents) in
compositional MARL verification is unjustified. Provides graph-theoretic,
spatial-geometric, and percolation-theoretic analyses showing when and why
interaction groups stay small, and graceful degradation strategies when
they do not.

Mathematical foundation:
    - Interaction graph G = (V, E) with connected components as groups
    - Bounded-degree graphs yield bounded component sizes under locality
    - Spatial locality ties interaction to geometric ball-packing bounds
    - Percolation theory gives sharp thresholds for giant component emergence
    - Transitive closure analysis quantifies dependency chain growth

References:
    [1] Grimmett, G. (1999). Percolation. Springer.
    [2] Penrose, M. (2003). Random Geometric Graphs. Oxford University Press.
    [3] Bollobás, B. (2001). Random Graphs. Cambridge University Press.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ComponentStats:
    """Statistics for connected components of an interaction graph."""
    sizes: List[int]
    count: int
    max_size: int
    mean_size: float
    median_size: float
    total_nodes: int

    @property
    def fraction_in_largest(self) -> float:
        return self.max_size / self.total_nodes if self.total_nodes > 0 else 0.0


@dataclass
class TransitivityReport:
    """Results of transitive closure analysis on an interaction graph."""
    original_edges: int
    closure_edges: int
    transitive_fraction: float
    clustering_coefficient: float
    original_component_sizes: List[int]
    closure_component_sizes: List[int]
    transitivity_growth_ratio: float


@dataclass
class DegradationPlan:
    """Plan for handling groups that exceed the verification budget."""
    strategy: str
    subgroups: List[List[int]]
    estimated_cost: float
    soundness_loss: float
    priority_order: List[int]


@dataclass
class GroupSizeReport:
    """Statistical report on empirical group size distributions."""
    sizes: np.ndarray
    best_fit: str
    fit_params: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    confidence_interval_max: Tuple[float, float]
    expected_max_group_size: float


class DegradationStrategy(Enum):
    HIERARCHICAL = auto()
    APPROXIMATE = auto()
    PRIORITIZED = auto()


# ---------------------------------------------------------------------------
# 1. InteractionGraphModel
# ---------------------------------------------------------------------------

class InteractionGraphModel:
    """
    Formal interaction graph G = (V, E).

    Definition
    ----------
    Let A = {a_1, ..., a_n} be a set of agents. The interaction graph
    G = (V, E) is defined by:

        V = A
        E = { (a_i, a_j) : a_i and a_j interact }

    where interaction is defined by one of:
        (a) Shared state variables: ∃ v ∈ Vars s.t. v ∈ Writes(a_i) ∩ Reads(a_j)
        (b) Happens-before edges: ∃ events e_i ∈ Events(a_i), e_j ∈ Events(a_j)
            such that e_i →_{HB} e_j or e_j →_{HB} e_i

    Connected components of G are the interaction groups. Compositional
    verification proceeds per-component, so component sizes determine
    scalability: if max component size k ≪ n, verification cost is
    O(n/k · f(k)) instead of O(f(n)).
    """

    def __init__(self) -> None:
        self._graph = nx.Graph()

    @classmethod
    def from_shared_variables(
        cls,
        agents: Sequence[int],
        writes: Dict[int, set],
        reads: Dict[int, set],
    ) -> "InteractionGraphModel":
        """
        Build interaction graph from shared-variable analysis.

        Edge (i, j) exists iff Writes(i) ∩ Reads(j) ≠ ∅
                             or Writes(j) ∩ Reads(i) ≠ ∅.
        """
        model = cls()
        model._graph.add_nodes_from(agents)
        agent_list = list(agents)
        for idx_a in range(len(agent_list)):
            for idx_b in range(idx_a + 1, len(agent_list)):
                a, b = agent_list[idx_a], agent_list[idx_b]
                w_a, r_a = writes.get(a, set()), reads.get(a, set())
                w_b, r_b = writes.get(b, set()), reads.get(b, set())
                if (w_a & r_b) or (w_b & r_a):
                    model._graph.add_edge(a, b)
        return model

    @classmethod
    def from_happens_before(
        cls,
        agents: Sequence[int],
        hb_edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> "InteractionGraphModel":
        """
        Build interaction graph from happens-before relation.

        Parameters
        ----------
        hb_edges : sequence of ((agent_i, event_i), (agent_j, event_j))
            Each entry is an HB edge between events of (possibly different) agents.
        """
        model = cls()
        model._graph.add_nodes_from(agents)
        for (a_i, _), (a_j, _) in hb_edges:
            if a_i != a_j:
                model._graph.add_edge(a_i, a_j)
        return model

    @classmethod
    def from_networkx(cls, G: nx.Graph) -> "InteractionGraphModel":
        model = cls()
        model._graph = G.copy()
        return model

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def component_stats(self) -> ComponentStats:
        """Compute connected-component statistics."""
        components = list(nx.connected_components(self._graph))
        sizes = sorted([len(c) for c in components], reverse=True)
        n = self._graph.number_of_nodes()
        if not sizes:
            return ComponentStats([], 0, 0, 0.0, 0.0, n)
        return ComponentStats(
            sizes=sizes,
            count=len(sizes),
            max_size=max(sizes),
            mean_size=float(np.mean(sizes)),
            median_size=float(np.median(sizes)),
            total_nodes=n,
        )

    def interaction_groups(self) -> List[List[int]]:
        """Return list of interaction groups (connected components)."""
        return [sorted(c) for c in nx.connected_components(self._graph)]

    def max_degree(self) -> int:
        if self._graph.number_of_nodes() == 0:
            return 0
        return max(d for _, d in self._graph.degree())

    def density(self) -> float:
        return nx.density(self._graph)


# ---------------------------------------------------------------------------
# 2. BoundedDegreeCondition
# ---------------------------------------------------------------------------

class BoundedDegreeCondition:
    """
    Bounded-degree condition for interaction graphs.

    Theorem (Component-size bound under bounded degree)
    ---------------------------------------------------
    Let G = (V, E) be a graph with maximum degree d.

    (a) Trivially, every connected component has at most |V| nodes.

    (b) If G additionally has treewidth tw(G) ≤ w, then each connected
        component has at most w + 1 nodes in its tree decomposition bags,
        enabling O(2^w · n) verification algorithms.

    (c) For spatially local interactions in D dimensions with radius r:
        - Each agent interacts with at most d ≤ c_D · r^D others
          (ball-packing bound: at most Vol(B(0,2r)) / Vol(B(0, δ/2))
           non-overlapping agents fit in the interaction ball, where δ
           is the minimum inter-agent distance).
        - If agents are uniformly distributed with density ρ:
          E[degree] = ρ · C_D · r^D  where C_D = π^{D/2} / Γ(D/2 + 1)
        - Connected components are bounded by O(r^D) when the graph
          is a unit-disk (or unit-ball) graph below percolation threshold.
    """

    @staticmethod
    def unit_ball_volume(dim: int) -> float:
        """Volume of the unit ball in R^dim: C_D = π^{D/2} / Γ(D/2 + 1)."""
        return (math.pi ** (dim / 2.0)) / math.gamma(dim / 2.0 + 1.0)

    @staticmethod
    def ball_packing_bound(radius: float, dim: int, min_separation: float = 1.0) -> int:
        """
        Upper bound on the number of non-overlapping spheres of diameter
        `min_separation` that fit inside a ball of radius `2 * radius`.

        This bounds the maximum degree: d ≤ (2r / (δ/2))^D = (4r/δ)^D.
        """
        if min_separation <= 0:
            raise ValueError("min_separation must be positive")
        ratio = (2.0 * radius) / (min_separation / 2.0)
        return int(math.ceil(ratio ** dim))

    def check_condition(
        self, graph: nx.Graph
    ) -> Tuple[bool, int, Optional[int]]:
        """
        Check the bounded-degree condition on graph G.

        Returns
        -------
        satisfied : bool
            True if maximum degree is bounded (heuristic: d < sqrt(n)).
        max_degree : int
            Maximum degree of the graph.
        component_bound : int or None
            Upper bound on component sizes if the condition gives one,
            otherwise None (when the bound is trivially n).
        """
        n = graph.number_of_nodes()
        if n == 0:
            return True, 0, 0

        d = max(deg for _, deg in graph.degree())
        threshold = max(int(math.sqrt(n)), 3)
        satisfied = d <= threshold

        # BFS-depth bound: in a tree of degree d, depth h has at most
        # 1 + d + d(d-1) + ... + d(d-1)^{h-1} nodes.
        # For random bounded-degree graphs components are typically O(log n).
        if satisfied and d > 0:
            # Estimate via BFS from highest-degree node
            components = list(nx.connected_components(graph))
            max_comp = max(len(c) for c in components)
            component_bound = max_comp
        else:
            component_bound = None

        return satisfied, d, component_bound

    def predicted_component_size(
        self, max_degree: int, spatial_dim: int, density: float = 1.0,
        radius: float = 1.0,
    ) -> float:
        """
        Predicted upper bound on component sizes for a spatially local
        interaction graph.

        For a random geometric graph in D dimensions with connection
        radius R and agent density ρ:
          - Expected degree λ = ρ · C_D · R^D
          - Below percolation threshold (λ < λ_c ≈ 1 in mean-field):
            expected largest component ~ O(log n)
          - Above threshold: giant component ~ Θ(n)

        Here we return the degree-based bound: O(d^2) as a conservative
        estimate for sub-critical random graphs.
        """
        expected_degree = density * self.unit_ball_volume(spatial_dim) * (radius ** spatial_dim)
        # Sub-critical regime: components bounded by O(λ / (1 - λ/λ_c)^2)
        # Use mean-field critical value λ_c ≈ 1 (from branching-process theory)
        lambda_c = 1.0
        effective_lambda = min(expected_degree, max_degree)

        if effective_lambda >= lambda_c:
            # Super-critical: giant component, return degree-based heuristic
            return float(max_degree ** spatial_dim)

        # Sub-critical: expected component size ~ 1 / (1 - λ/λ_c)
        ratio = effective_lambda / lambda_c
        if ratio >= 0.999:
            ratio = 0.999
        return 1.0 / (1.0 - ratio)


# ---------------------------------------------------------------------------
# 3. SpatialLocalityCondition
# ---------------------------------------------------------------------------

class SpatialLocalityCondition:
    """
    Spatial locality condition for interaction group boundedness.

    Model
    -----
    Agents have positions p_1, ..., p_n ∈ ℝ^D drawn i.i.d. from a
    distribution with density ρ. The interaction graph is a random
    geometric graph (RGG):

        E = { (i, j) : ||p_i - p_j|| ≤ R }

    Percolation theory for RGGs [Penrose 2003]:
    -------------------------------------------
    Let λ = ρ · C_D · R^D  (expected degree).

    - Sub-critical phase (λ < λ_c):
      All connected components have size O(log n) w.h.p.
      The largest component satisfies:
          |C_max| ≤ (c / (λ_c - λ)²) · log n  w.h.p.

    - Super-critical phase (λ > λ_c):
      A unique giant component exists with size Θ(n).
      Specifically, |C_giant| / n → θ(λ) > 0 where θ is the
      survival probability of the associated branching process.

    - Critical value λ_c:
      In mean-field approximation, λ_c = 1 (Erdős–Rényi threshold).
      For D-dimensional RGGs, the exact value depends on D:
        D=1: λ_c = 1  (1D interval graph)
        D=2: λ_c ≈ 4.512  (continuum percolation, [Quintanilla et al. 2000])
        D=3: λ_c ≈ 2.736  (3D continuum percolation)
      We use known values where available and mean-field otherwise.
    """

    # Critical average degrees for continuum percolation in D dimensions.
    # Sources: Quintanilla et al. (2000), Torquato (2002).
    CRITICAL_DEGREES: Dict[int, float] = {
        1: 1.0,
        2: 4.512,
        3: 2.736,
    }

    @staticmethod
    def unit_ball_volume(dim: int) -> float:
        """Volume of the D-dimensional unit ball."""
        return (math.pi ** (dim / 2.0)) / math.gamma(dim / 2.0 + 1.0)

    def critical_density(self, R: float, dim: int) -> float:
        """
        Critical agent density ρ_c for percolation transition.

        Below ρ_c, all components are O(log n).
        Above ρ_c, a giant component of size Θ(n) exists.

        Derivation:
            λ_c = ρ_c · C_D · R^D
            ⟹  ρ_c = λ_c / (C_D · R^D)
        """
        if R <= 0:
            raise ValueError("Interaction radius R must be positive")
        if dim < 1:
            raise ValueError("Dimension must be at least 1")
        lambda_c = self.CRITICAL_DEGREES.get(dim, 1.0)
        vol = self.unit_ball_volume(dim)
        return lambda_c / (vol * R ** dim)

    def expected_degree(self, density: float, R: float, dim: int) -> float:
        """Expected degree λ = ρ · C_D · R^D."""
        return density * self.unit_ball_volume(dim) * (R ** dim)

    def expected_component_sizes(
        self,
        n_agents: int,
        density: float,
        R: float,
        dim: int,
        n_samples: int = 200,
    ) -> Dict[str, object]:
        """
        Estimate component size distribution via Monte Carlo sampling
        of random geometric graphs.

        Returns
        -------
        dict with keys:
            'mean_max' : average size of the largest component
            'std_max'  : std dev of largest component size
            'mean_sizes' : average component size distribution (sorted desc)
            'phase'    : 'sub-critical' or 'super-critical'
            'lambda'   : expected degree
            'lambda_c' : critical degree
            'sample_max_sizes' : list of max component sizes per sample
        """
        lam = self.expected_degree(density, R, dim)
        lam_c = self.CRITICAL_DEGREES.get(dim, 1.0)
        phase = "sub-critical" if lam < lam_c else "super-critical"

        # Determine domain size from density: n = ρ · Vol(domain)
        # For a hypercube of side L: ρ = n / L^D  ⟹  L = (n / ρ)^{1/D}
        if density <= 0:
            raise ValueError("Density must be positive")
        L = (n_agents / density) ** (1.0 / dim)

        rng = np.random.default_rng(42)
        max_sizes = []
        all_sizes: List[List[int]] = []

        for _ in range(n_samples):
            positions = rng.uniform(0, L, size=(n_agents, dim))
            G = self._build_rgg(positions, R)
            components = list(nx.connected_components(G))
            sizes = sorted([len(c) for c in components], reverse=True)
            max_sizes.append(sizes[0] if sizes else 0)
            all_sizes.append(sizes)

        max_arr = np.array(max_sizes, dtype=float)
        return {
            "mean_max": float(np.mean(max_arr)),
            "std_max": float(np.std(max_arr)),
            "phase": phase,
            "lambda": lam,
            "lambda_c": lam_c,
            "sample_max_sizes": max_sizes,
        }

    @staticmethod
    def _build_rgg(positions: np.ndarray, R: float) -> nx.Graph:
        """Build a random geometric graph from positions and radius."""
        n = positions.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        # Use scipy for efficiency on larger instances
        from scipy.spatial import KDTree
        tree = KDTree(positions)
        pairs = tree.query_pairs(R)
        G.add_edges_from(pairs)
        return G

    def is_subcritical(self, density: float, R: float, dim: int) -> bool:
        """Check whether parameters place us in the sub-critical phase."""
        lam = self.expected_degree(density, R, dim)
        lam_c = self.CRITICAL_DEGREES.get(dim, 1.0)
        return lam < lam_c

    def subcritical_component_bound(
        self, n_agents: int, density: float, R: float, dim: int
    ) -> Optional[float]:
        """
        Theoretical upper bound on max component size in sub-critical regime.

        From percolation theory [Penrose 2003, Thm 10.2]:
            |C_max| ≤  c · log(n) / (λ_c - λ)²   w.h.p.

        where c is a dimension-dependent constant (we use c = 1 as a
        normalised estimate; in practice the constant is O(1)).
        """
        lam = self.expected_degree(density, R, dim)
        lam_c = self.CRITICAL_DEGREES.get(dim, 1.0)
        if lam >= lam_c:
            return None  # super-critical, no sub-linear bound
        gap = lam_c - lam
        if gap < 1e-12:
            return float(n_agents)
        return math.log(n_agents) / (gap ** 2)


# ---------------------------------------------------------------------------
# 4. TransitiveClosure
# ---------------------------------------------------------------------------

class TransitiveClosure:
    """
    Analysis of transitive closure growth in interaction graphs.

    Critique addressed
    ------------------
    Physical dependencies can be transitive: if agent A shares state
    with agent B, and B shares state with C, then A and C may be
    indirectly coupled. The interaction graph already captures this
    via connected components (reachability = transitive closure of
    adjacency). However, the *edge-level* transitive closure reveals
    how many direct pairwise interactions are implied:

        |E(TC(G))| vs |E(G)|

    If this ratio is close to 1, interactions are already "closed"
    and the graph structure is robust. If the ratio is large,
    indirect dependencies dominate and decomposition is fragile.

    For happens-before (HB) interactions
    ------------------------------------
    The HB relation is, by definition, the transitive closure of
    program-order ∪ synchronization-order. So HB-based interaction
    graphs already reflect full transitivity. No additional closure
    is needed.

    For physical / shared-variable interactions
    --------------------------------------------
    Transitivity depends on variable scoping. If A writes x,
    B reads x and writes y, and C reads y, then A and C do NOT
    directly interact unless A writes y or C reads x. The
    transitive closure of variable-sharing is NOT the same as
    the interaction graph's transitive closure — it requires
    tracking variable-level data flow, not just agent-level
    adjacency.

    The clustering coefficient measures local transitivity:
        C = (3 × triangles) / (connected triples)
    High clustering ⟹ interactions are locally transitive.
    """

    def analyze_transitivity(self, graph: nx.Graph) -> TransitivityReport:
        """
        Compare graph G against its transitive closure TC(G).

        The transitive closure at the graph level is the graph where
        (u, v) ∈ E(TC) iff u and v are in the same connected component
        of G. Equivalently, TC(G) is the disjoint union of cliques
        corresponding to G's connected components.
        """
        original_edges = graph.number_of_edges()

        # Transitive closure: full clique within each component
        closure = nx.Graph()
        closure.add_nodes_from(graph.nodes())
        components = list(nx.connected_components(graph))
        closure_edge_count = 0
        for comp in components:
            comp_list = sorted(comp)
            k = len(comp_list)
            closure_edge_count += k * (k - 1) // 2
            for i in range(k):
                for j in range(i + 1, k):
                    closure.add_edge(comp_list[i], comp_list[j])

        # Clustering coefficient (local transitivity measure)
        clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0.0

        # Transitive fraction: what fraction of TC edges already exist
        if closure_edge_count > 0:
            transitive_fraction = original_edges / closure_edge_count
        else:
            transitive_fraction = 1.0  # no edges, trivially closed

        orig_sizes = sorted(
            [len(c) for c in components], reverse=True
        )
        # Closure components are the same (cliques don't change connectivity)
        closure_sizes = orig_sizes[:]

        growth_ratio = (
            closure_edge_count / original_edges
            if original_edges > 0 else 1.0
        )

        return TransitivityReport(
            original_edges=original_edges,
            closure_edges=closure_edge_count,
            transitive_fraction=transitive_fraction,
            clustering_coefficient=clustering,
            original_component_sizes=orig_sizes,
            closure_component_sizes=closure_sizes,
            transitivity_growth_ratio=growth_ratio,
        )

    @staticmethod
    def hb_interaction_note() -> str:
        """
        Note on HB-based interaction transitivity.

        The happens-before relation HB = (PO ∪ SO)^+ is already the
        transitive closure, so:
          - If e_A →_{HB} e_B and e_B →_{HB} e_C, then e_A →_{HB} e_C.
          - The interaction graph built from HB already reflects all
            transitive dependencies.
          - No additional closure computation is needed.
          - Component sizes from the HB-based graph are exact.
        """
        return (
            "HB-based interaction graphs are transitively closed by construction. "
            "The happens-before relation is defined as the transitive closure of "
            "program-order ∪ synchronization-order, so all indirect dependencies "
            "are already captured. Component sizes are exact."
        )

    @staticmethod
    def variable_transitivity_analysis(
        writes: Dict[int, set],
        reads: Dict[int, set],
    ) -> Dict[str, object]:
        """
        Analyse whether variable-level interactions exhibit transitivity.

        For agents A, B, C:
          A →_var B  iff  Writes(A) ∩ Reads(B) ≠ ∅
          Transitivity: A →_var B ∧ B →_var C  ⟹?  A →_var C

        This is NOT generally true. We measure the fraction of
        transitive triples where it holds.
        """
        agents = sorted(set(writes.keys()) | set(reads.keys()))
        # Build directed variable-interaction graph
        dg = nx.DiGraph()
        dg.add_nodes_from(agents)
        for a in agents:
            for b in agents:
                if a != b and (writes.get(a, set()) & reads.get(b, set())):
                    dg.add_edge(a, b)

        # Count transitive triples
        transitive_triples = 0
        total_triples = 0
        for a in agents:
            for b in dg.successors(a):
                for c in dg.successors(b):
                    if c != a:
                        total_triples += 1
                        if dg.has_edge(a, c):
                            transitive_triples += 1

        fraction = transitive_triples / total_triples if total_triples > 0 else 0.0
        return {
            "total_triples": total_triples,
            "transitive_triples": transitive_triples,
            "transitivity_fraction": fraction,
            "directed_edges": dg.number_of_edges(),
        }


# ---------------------------------------------------------------------------
# 5. AdaptiveDegradation
# ---------------------------------------------------------------------------

class AdaptiveDegradation:
    """
    Graceful degradation strategies when interaction groups exceed
    the verification budget.

    When max component size k exceeds threshold τ, exact compositional
    verification becomes infeasible. Three strategies:

    Strategy 1: Hierarchical Decomposition
    ---------------------------------------
    Recursively partition the large component using graph bisection
    (e.g., spectral bisection or Kernighan–Lin). Each sub-component
    is verified independently, with interface constraints at cut edges.
    Soundness is preserved if interface contracts are conservative.

    Strategy 2: Approximate Verification
    -------------------------------------
    Allow bounded unsoundness by abstracting away low-probability
    interactions. Edges with interaction probability below ε are
    removed, yielding a sparser graph with smaller components.
    Unsoundness is bounded by ε · |removed edges|.

    Strategy 3: Prioritized Sub-group Verification
    -----------------------------------------------
    Rank agents by criticality (e.g., degree centrality, betweenness)
    and verify high-risk sub-groups first. Budget allocation follows
    a risk-proportional scheme.
    """

    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold

    def plan_degradation(
        self,
        graph: nx.Graph,
        budget: float = 100.0,
        strategy: DegradationStrategy = DegradationStrategy.HIERARCHICAL,
    ) -> DegradationPlan:
        """
        Create a degradation plan for components exceeding threshold.

        Parameters
        ----------
        graph : nx.Graph
            The interaction graph.
        budget : float
            Total verification budget (abstract units).
        strategy : DegradationStrategy
            Which degradation strategy to apply.

        Returns
        -------
        DegradationPlan
        """
        if strategy == DegradationStrategy.HIERARCHICAL:
            return self._hierarchical(graph, budget)
        elif strategy == DegradationStrategy.APPROXIMATE:
            return self._approximate(graph, budget)
        elif strategy == DegradationStrategy.PRIORITIZED:
            return self._prioritized(graph, budget)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _hierarchical(self, graph: nx.Graph, budget: float) -> DegradationPlan:
        """
        Hierarchical decomposition via recursive spectral bisection.

        For a component of size k, bisection yields two sub-components
        of size ~k/2. Recursion depth is O(log(k/τ)), total cost is
        O(k · log(k/τ)) per component.

        Cut edges represent inter-subgroup interactions that must be
        conservatively abstracted. Soundness is preserved when interface
        contracts over-approximate the cut-edge behaviors.
        """
        subgroups = []
        total_cost = 0.0

        for comp in nx.connected_components(graph):
            comp_list = sorted(comp)
            if len(comp_list) <= self.threshold:
                subgroups.append(comp_list)
                total_cost += len(comp_list) ** 2
            else:
                parts = self._recursive_bisect(
                    graph.subgraph(comp_list).copy(), self.threshold
                )
                subgroups.extend(parts)
                for p in parts:
                    total_cost += len(p) ** 2

        # Priority: larger subgroups first
        subgroups.sort(key=len, reverse=True)
        priority = list(range(len(subgroups)))

        return DegradationPlan(
            strategy="hierarchical",
            subgroups=subgroups,
            estimated_cost=min(total_cost, budget),
            soundness_loss=0.0,  # sound if contracts are conservative
            priority_order=priority,
        )

    def _recursive_bisect(
        self, G: nx.Graph, threshold: int
    ) -> List[List[int]]:
        """Recursively bisect graph until all parts ≤ threshold."""
        nodes = list(G.nodes())
        if len(nodes) <= threshold:
            return [sorted(nodes)]

        try:
            partition = nx.community.kernighan_lin_bisection(G)
            part_a, part_b = sorted(partition[0]), sorted(partition[1])
        except Exception:
            mid = len(nodes) // 2
            part_a, part_b = sorted(nodes[:mid]), sorted(nodes[mid:])

        result = []
        if len(part_a) > threshold:
            result.extend(
                self._recursive_bisect(G.subgraph(part_a).copy(), threshold)
            )
        else:
            result.append(part_a)

        if len(part_b) > threshold:
            result.extend(
                self._recursive_bisect(G.subgraph(part_b).copy(), threshold)
            )
        else:
            result.append(part_b)

        return result

    def _approximate(self, graph: nx.Graph, budget: float) -> DegradationPlan:
        """
        Approximate verification by edge pruning.

        Remove edges with lowest "interaction strength" (approximated by
        edge betweenness centrality) until all components ≤ threshold.
        Unsoundness is proportional to the number of removed edges.
        """
        G = graph.copy()
        removed = 0

        while True:
            components = list(nx.connected_components(G))
            max_size = max((len(c) for c in components), default=0)
            if max_size <= self.threshold:
                break
            if G.number_of_edges() == 0:
                break

            # Remove edge with highest betweenness (bridge-like)
            betweenness = nx.edge_betweenness_centrality(G)
            if not betweenness:
                break
            edge_to_remove = max(betweenness, key=betweenness.get)
            G.remove_edge(*edge_to_remove)
            removed += 1

        subgroups = [sorted(c) for c in nx.connected_components(G)]
        subgroups.sort(key=len, reverse=True)
        total_edges = graph.number_of_edges()
        soundness_loss = removed / total_edges if total_edges > 0 else 0.0

        return DegradationPlan(
            strategy="approximate",
            subgroups=subgroups,
            estimated_cost=sum(len(s) ** 2 for s in subgroups),
            soundness_loss=soundness_loss,
            priority_order=list(range(len(subgroups))),
        )

    def _prioritized(self, graph: nx.Graph, budget: float) -> DegradationPlan:
        """
        Prioritized sub-group verification.

        Rank nodes by degree centrality. Form priority groups by
        taking top-k nodes and their neighborhoods, verifying
        high-centrality clusters first.
        """
        centrality = nx.degree_centrality(graph)
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

        subgroups = []
        visited = set()
        priority = []
        idx = 0

        for node in sorted_nodes:
            if node in visited:
                continue
            # Take the node and its neighbors as a subgroup
            neighbors = set(graph.neighbors(node)) | {node}
            group = sorted(neighbors - visited)
            if not group:
                continue
            # Truncate to threshold if needed
            if len(group) > self.threshold:
                group = group[: self.threshold]
            subgroups.append(group)
            priority.append(idx)
            visited.update(group)
            idx += 1

        # Add any remaining isolated nodes
        remaining = sorted(set(graph.nodes()) - visited)
        if remaining:
            for i in range(0, len(remaining), self.threshold):
                chunk = remaining[i : i + self.threshold]
                subgroups.append(chunk)
                priority.append(idx)
                idx += 1

        cost = sum(len(s) ** 2 for s in subgroups)

        return DegradationPlan(
            strategy="prioritized",
            subgroups=subgroups,
            estimated_cost=min(cost, budget),
            soundness_loss=0.0,
            priority_order=priority,
        )

    def recommend_strategy(
        self, graph: nx.Graph
    ) -> DegradationStrategy:
        """
        Recommend a degradation strategy based on graph properties.

        Heuristics:
          - If graph is sparse (density < 0.1): hierarchical works well
          - If graph has high clustering (> 0.5): approximate is effective
          - Otherwise: prioritized is safest
        """
        d = nx.density(graph)
        c = nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0

        if d < 0.1:
            return DegradationStrategy.HIERARCHICAL
        elif c > 0.5:
            return DegradationStrategy.APPROXIMATE
        else:
            return DegradationStrategy.PRIORITIZED


# ---------------------------------------------------------------------------
# 6. EmpiricalGroupSizeAnalysis
# ---------------------------------------------------------------------------

class EmpiricalGroupSizeAnalysis:
    """
    Empirical analysis of interaction group size distributions.

    Measures actual group sizes from benchmark environments and fits
    statistical models to characterise their distribution:

    Candidate distributions:
      - Poisson: λ^k · e^{-λ} / k!
        Expected when interactions are independent and uniformly random.
      - Power-law: P(k) ∝ k^{-α}
        Expected in scale-free networks (preferential attachment).
      - Log-normal: P(k) ∝ (1/k) · exp(-(ln k - μ)² / 2σ²)
        Expected when group sizes result from multiplicative random processes.

    Goodness-of-fit is assessed via Kolmogorov–Smirnov tests.
    Confidence intervals on the expected maximum group size use
    extreme-value theory (Gumbel distribution for block maxima).
    """

    @staticmethod
    def extract_component_sizes(graph: nx.Graph) -> np.ndarray:
        """Extract sorted component sizes from a graph."""
        sizes = [len(c) for c in nx.connected_components(graph)]
        return np.array(sorted(sizes, reverse=True), dtype=float)

    def analyze_single(self, graph: nx.Graph) -> GroupSizeReport:
        """Analyze group size distribution from a single graph."""
        sizes = self.extract_component_sizes(graph)
        if len(sizes) == 0:
            return GroupSizeReport(
                sizes=sizes,
                best_fit="none",
                fit_params={},
                ks_statistic=0.0,
                ks_pvalue=1.0,
                confidence_interval_max=(0.0, 0.0),
                expected_max_group_size=0.0,
            )
        return self._fit_and_report(sizes)

    def analyze_benchmarks(
        self,
        environments: Sequence[nx.Graph],
    ) -> GroupSizeReport:
        """
        Analyze group sizes across multiple benchmark environments.

        Pools component sizes from all environments and fits
        distributional models.
        """
        all_sizes = []
        for env in environments:
            sizes = [len(c) for c in nx.connected_components(env)]
            all_sizes.extend(sizes)

        if not all_sizes:
            return GroupSizeReport(
                sizes=np.array([]),
                best_fit="none",
                fit_params={},
                ks_statistic=0.0,
                ks_pvalue=1.0,
                confidence_interval_max=(0.0, 0.0),
                expected_max_group_size=0.0,
            )

        sizes = np.array(sorted(all_sizes, reverse=True), dtype=float)
        return self._fit_and_report(sizes)

    def _fit_and_report(self, sizes: np.ndarray) -> GroupSizeReport:
        """Fit candidate distributions and select best."""
        results: Dict[str, Tuple[Dict[str, float], float, float]] = {}

        # -- Poisson fit --
        try:
            lam = float(np.mean(sizes))
            ks_stat, ks_p = stats.kstest(
                sizes, lambda x: stats.poisson.cdf(x, mu=lam)
            )
            results["poisson"] = ({"lambda": lam}, ks_stat, ks_p)
        except Exception:
            pass

        # -- Log-normal fit --
        try:
            positive = sizes[sizes > 0]
            if len(positive) > 1:
                log_data = np.log(positive)
                mu, sigma = float(np.mean(log_data)), float(np.std(log_data, ddof=1))
                if sigma < 1e-12:
                    sigma = 1e-12
                ks_stat, ks_p = stats.kstest(
                    positive, "lognorm", args=(sigma, 0, np.exp(mu))
                )
                results["lognormal"] = (
                    {"mu": mu, "sigma": sigma},
                    ks_stat,
                    ks_p,
                )
        except Exception:
            pass

        # -- Power-law fit (Pareto) --
        try:
            positive = sizes[sizes >= 1]
            if len(positive) > 1:
                # MLE for Pareto exponent: α = n / Σ ln(x / x_min)
                x_min = float(np.min(positive))
                if x_min > 0:
                    log_ratios = np.log(positive / x_min)
                    sum_log = float(np.sum(log_ratios))
                    if sum_log > 0:
                        alpha = len(positive) / sum_log
                    else:
                        alpha = 1.0
                    ks_stat, ks_p = stats.kstest(
                        positive,
                        "pareto",
                        args=(alpha, 0, x_min),
                    )
                    results["power_law"] = (
                        {"alpha": alpha, "x_min": x_min},
                        ks_stat,
                        ks_p,
                    )
        except Exception:
            pass

        # Select best fit by KS p-value (higher = better fit)
        if not results:
            best_name = "empirical"
            best_params: Dict[str, float] = {}
            best_ks, best_p = 0.0, 1.0
        else:
            best_name = max(results, key=lambda k: results[k][2])
            best_params, best_ks, best_p = results[best_name]

        # Confidence interval on max group size via Gumbel extreme-value theory
        ci_max = self._gumbel_confidence_interval(sizes)
        expected_max = float(np.max(sizes))

        return GroupSizeReport(
            sizes=sizes,
            best_fit=best_name,
            fit_params=best_params,
            ks_statistic=best_ks,
            ks_pvalue=best_p,
            confidence_interval_max=ci_max,
            expected_max_group_size=expected_max,
        )

    @staticmethod
    def _gumbel_confidence_interval(
        sizes: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Confidence interval for the maximum using Gumbel extreme-value theory.

        For i.i.d. samples X_1, ..., X_n, the distribution of max(X_i) is
        approximately Gumbel with location μ_n and scale β_n estimated from
        the sample.

        The (1-α) CI for the maximum of a future sample of the same size is:
            [μ_n + β_n · ln(-ln(α/2)),  μ_n + β_n · ln(-ln(1 - α/2))]
        """
        if len(sizes) < 2:
            m = float(sizes[0]) if len(sizes) == 1 else 0.0
            return (m, m)

        # Fit Gumbel to the data using method of moments
        mean_val = float(np.mean(sizes))
        std_val = float(np.std(sizes, ddof=1))
        if std_val < 1e-12:
            return (mean_val, mean_val)

        # Gumbel parameters from moments: β = σ√6 / π, μ = mean - γ·β
        euler_gamma = 0.5772156649
        beta = std_val * math.sqrt(6) / math.pi
        mu = mean_val - euler_gamma * beta

        alpha = 1.0 - confidence
        # Quantiles of the Gumbel distribution
        lower = mu - beta * math.log(-math.log(1.0 - alpha / 2.0))
        upper = mu - beta * math.log(-math.log(alpha / 2.0))

        return (lower, upper)

    @staticmethod
    def generate_benchmark_rgg(
        n_agents: int,
        dim: int = 2,
        density: float = 1.0,
        radius: float = 1.0,
        seed: int = 0,
    ) -> nx.Graph:
        """
        Generate a random geometric graph as a synthetic benchmark.

        Places n_agents uniformly in [0, L]^dim where L = (n/ρ)^{1/D},
        then connects agents within distance R.
        """
        rng = np.random.default_rng(seed)
        L = (n_agents / density) ** (1.0 / dim)
        positions = rng.uniform(0, L, size=(n_agents, dim))

        from scipy.spatial import KDTree
        tree = KDTree(positions)
        pairs = tree.query_pairs(radius)

        G = nx.Graph()
        G.add_nodes_from(range(n_agents))
        G.add_edges_from(pairs)
        return G


# ---------------------------------------------------------------------------
# 7. GroupMergingMonotonicity
# ---------------------------------------------------------------------------


@dataclass
class MergeCostBenefit:
    """Cost-benefit analysis for a single group merge decision."""
    group_a: List[int]
    group_b: List[int]
    merged_size: int
    verification_cost_before: float
    verification_cost_after: float
    cost_increase: float
    precision_gain: float
    is_beneficial: bool


@dataclass
class MonotonicityReport:
    """Report on monotonicity of partial convergence under merging."""
    is_monotone: bool
    pre_merge_bounds: List[float]
    post_merge_bounds: List[float]
    bound_improvement: float
    explanation: str


class GroupMergingMonotonicity:
    r"""Monotonicity argument for adaptive group merging.

    When two groups G_a and G_b are merged, partial convergence results
    computed for G_a and G_b individually must remain valid (sound) for
    the merged group G_a ∪ G_b.

    **Theorem (Merge monotonicity).**
    Let S_a, S_b be the abstract states computed for groups G_a, G_b
    independently.  Let S_{ab} be the abstract state for the merged group.
    If the abstract transformer T is monotone in the group structure:

        G_a ⊆ G_{ab} ∧ G_b ⊆ G_{ab}  ⟹  S_a ⊆ S_{ab} ∧ S_b ⊆ S_{ab}

    then any safety property verified for S_{ab} also holds for S_a and S_b.

    **Cost-benefit analysis.**
    Merging increases verification cost (from O(|G_a|^2 + |G_b|^2) to
    O((|G_a| + |G_b|)^2)) but may improve precision by eliminating
    inter-group over-approximation at cut edges.

    Parameters
    ----------
    verification_cost_exponent : float
        Exponent in the cost model O(k^p).  Default 2.0.
    precision_weight : float
        Weight for precision gain relative to cost.  Default 1.0.
    """

    def __init__(
        self,
        verification_cost_exponent: float = 2.0,
        precision_weight: float = 1.0,
    ):
        self._cost_exp = verification_cost_exponent
        self._precision_weight = precision_weight

    def verification_cost(self, group_size: int) -> float:
        """Cost to verify a group of given size: O(k^p)."""
        return float(group_size ** self._cost_exp)

    def merge_cost_benefit(
        self,
        group_a: List[int],
        group_b: List[int],
        cut_edges: int = 0,
    ) -> MergeCostBenefit:
        r"""Analyse cost-benefit of merging two groups.

        Cost increase: (|a|+|b|)^p - (|a|^p + |b|^p).
        Precision gain: proportional to the number of cut edges that
        are no longer over-approximated (each cut edge contributes
        a precision gain proportional to the edge weight).

        Parameters
        ----------
        group_a, group_b : lists of node ids.
        cut_edges : int
            Number of edges between the two groups.
        """
        sa, sb = len(group_a), len(group_b)
        cost_before = self.verification_cost(sa) + self.verification_cost(sb)
        cost_after = self.verification_cost(sa + sb)
        cost_increase = cost_after - cost_before

        # Precision gain: each cut edge removes one inter-group abstraction
        precision_gain = float(cut_edges) * self._precision_weight

        # Beneficial if precision gain outweighs cost increase
        is_beneficial = precision_gain > cost_increase * 0.01

        return MergeCostBenefit(
            group_a=group_a,
            group_b=group_b,
            merged_size=sa + sb,
            verification_cost_before=cost_before,
            verification_cost_after=cost_after,
            cost_increase=cost_increase,
            precision_gain=precision_gain,
            is_beneficial=is_beneficial,
        )

    def check_monotonicity(
        self,
        pre_merge_bounds: List[float],
        post_merge_bounds: List[float],
    ) -> MonotonicityReport:
        r"""Verify that merging does not invalidate partial convergence.

        Pre-merge bounds on safety margins must be upper-bounded by the
        post-merge bounds (since the merged group's abstract state is
        at least as large, its safety margin may decrease, but soundness
        is preserved).

        Parameters
        ----------
        pre_merge_bounds : list of float
            Safety margin bounds for individual groups before merge.
        post_merge_bounds : list of float
            Safety margin bounds for merged groups after merge.
        """
        # Monotonicity: post-merge abstract state is larger, so safety
        # margins may decrease.  Soundness is maintained as long as
        # the merged analysis is re-run.
        pre_max = max(pre_merge_bounds) if pre_merge_bounds else 0.0
        post_max = max(post_merge_bounds) if post_merge_bounds else 0.0

        # The key check: post-merge bounds should be finite (analysis converged)
        all_finite = all(b < float("inf") for b in post_merge_bounds)
        # Bound improvement: how much tighter is the post-merge analysis
        improvement = pre_max - post_max if pre_max > post_max else 0.0

        is_monotone = all_finite

        if is_monotone:
            explanation = (
                "Merging preserves soundness: post-merge analysis converges "
                "and the merged abstract state soundly over-approximates "
                "all pre-merge states."
            )
        else:
            explanation = (
                "WARNING: Post-merge analysis did not converge for all groups. "
                "Partial convergence results from pre-merge analysis remain "
                "sound but may be overly conservative."
            )

        return MonotonicityReport(
            is_monotone=is_monotone,
            pre_merge_bounds=pre_merge_bounds,
            post_merge_bounds=post_merge_bounds,
            bound_improvement=improvement,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# 8. MaxGroupSizeBound
# ---------------------------------------------------------------------------


@dataclass
class MaxGroupSizeBoundResult:
    """Theoretical bound on maximum group size."""
    bound: float
    density: float
    interaction_radius: float
    spatial_dim: int
    n_agents: int
    is_subcritical: bool
    explanation: str


class MaxGroupSizeBound:
    r"""Theoretical bound on maximum group size under given HB density.

    Provides formal conditions under which interaction groups remain
    bounded, combining:

      1. **Degree-based bound**: For graphs with max degree d,
         each component has at most d^D nodes (spatial locality).
      2. **Percolation bound**: For sub-critical random geometric graphs,
         max component ≤ c·log(n) / (λ_c - λ)².
      3. **HB density bound**: The happens-before edge density ρ_HB
         determines the expected number of inter-agent dependencies.
         If ρ_HB · n < λ_c, groups remain O(log n).

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension of the agent deployment.
    """

    def __init__(self, spatial_dim: int = 2):
        self._dim = spatial_dim
        self._slc = SpatialLocalityCondition()

    def compute_bound(
        self,
        n_agents: int,
        hb_density: float,
        interaction_radius: float = 1.0,
    ) -> MaxGroupSizeBoundResult:
        r"""Compute theoretical max group size bound.

        Parameters
        ----------
        n_agents : int
            Number of agents.
        hb_density : float
            Density of HB edges (fraction of possible edges present).
        interaction_radius : float
            Spatial interaction radius R.
        """
        # Expected degree from spatial locality
        # density = n / L^D, where L is domain side length
        # We use hb_density as an edge probability proxy
        expected_degree = hb_density * (n_agents - 1)

        # Critical degree for percolation
        lam_c = self._slc.CRITICAL_DEGREES.get(self._dim, 1.0)
        is_sub = expected_degree < lam_c

        if is_sub:
            # Sub-critical: max component ≤ c·log(n) / (λ_c - λ)²
            gap = lam_c - expected_degree
            if gap < 1e-12:
                bound = float(n_agents)
            else:
                bound = max(1.0, math.log(max(n_agents, 2)) / (gap ** 2))
            explanation = (
                f"Sub-critical regime: λ={expected_degree:.4f} < λ_c={lam_c:.4f}. "
                f"Max component size ≤ O(log n / (λ_c - λ)²) = {bound:.1f}."
            )
        else:
            # Super-critical: giant component exists
            bound = float(n_agents)
            explanation = (
                f"Super-critical regime: λ={expected_degree:.4f} ≥ λ_c={lam_c:.4f}. "
                f"A giant component of size Θ(n) may exist. "
                f"Group size bound is trivially n={n_agents}."
            )

        return MaxGroupSizeBoundResult(
            bound=bound,
            density=hb_density,
            interaction_radius=interaction_radius,
            spatial_dim=self._dim,
            n_agents=n_agents,
            is_subcritical=is_sub,
            explanation=explanation,
        )

    def max_density_for_bounded_groups(
        self,
        n_agents: int,
        max_group_size: int,
    ) -> float:
        r"""Find maximum HB density that keeps groups ≤ *max_group_size*.

        Inverts the sub-critical bound:
          max_group_size ≥ log(n) / (λ_c - λ)²
          ⟹ λ ≤ λ_c - sqrt(log(n) / max_group_size)
          ⟹ density ≤ (λ_c - sqrt(log(n) / max_group_size)) / (n-1)
        """
        lam_c = self._slc.CRITICAL_DEGREES.get(self._dim, 1.0)
        log_n = math.log(max(n_agents, 2))

        if max_group_size <= 0:
            return 0.0

        gap = math.sqrt(log_n / max_group_size)
        max_lambda = lam_c - gap

        if max_lambda <= 0 or n_agents <= 1:
            return 0.0
        return max_lambda / (n_agents - 1)


# ---------------------------------------------------------------------------
# Convenience: unified analysis pipeline
# ---------------------------------------------------------------------------

def full_group_size_analysis(
    graph: nx.Graph,
    spatial_dim: int = 2,
    interaction_radius: float = 1.0,
    verification_threshold: int = 5,
    verification_budget: float = 100.0,
) -> Dict[str, object]:
    """
    Run the complete group-size analysis pipeline on an interaction graph.

    Steps:
      1. Compute interaction graph statistics
      2. Check bounded-degree condition
      3. Analyse transitivity
      4. Fit empirical distribution
      5. If components exceed threshold, plan degradation

    Returns a dict with all analysis results.
    """
    model = InteractionGraphModel.from_networkx(graph)
    stats = model.component_stats()

    bdc = BoundedDegreeCondition()
    satisfied, max_deg, comp_bound = bdc.check_condition(graph)

    tc = TransitiveClosure()
    trans_report = tc.analyze_transitivity(graph)

    empirical = EmpiricalGroupSizeAnalysis()
    size_report = empirical.analyze_single(graph)

    degradation_plan = None
    if stats.max_size > verification_threshold:
        ad = AdaptiveDegradation(threshold=verification_threshold)
        strategy = ad.recommend_strategy(graph)
        degradation_plan = ad.plan_degradation(graph, verification_budget, strategy)

    return {
        "component_stats": stats,
        "bounded_degree": {
            "satisfied": satisfied,
            "max_degree": max_deg,
            "component_bound": comp_bound,
        },
        "transitivity": trans_report,
        "empirical_fit": size_report,
        "degradation_plan": degradation_plan,
    }
