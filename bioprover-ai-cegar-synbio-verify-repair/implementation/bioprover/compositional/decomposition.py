"""Automatic module decomposition for compositional verification.

Analyzes gene-regulatory-network graph structure to partition a biological
system into loosely-coupled modules amenable to independent verification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decomposition strategy enum
# ---------------------------------------------------------------------------

class DecompositionStrategy(Enum):
    """Strategy used to decompose a GRN into modules."""
    SCC = auto()                  # Strongly connected components
    COMMUNITY = auto()            # Louvain community detection
    CASCADE = auto()              # Linear signal-flow chains
    FANOUT = auto()               # One-to-many, no feedback
    FEEDBACK_ISOLATION = auto()   # Isolate minimal feedback vertex set
    SLOW_FAST = auto()            # Time-scale separation
    HIERARCHICAL = auto()         # Recursive bisection
    OPTIMAL_ILP = auto()          # Integer programming (small instances)
    GREEDY = auto()               # Greedy heuristic (larger instances)


# ---------------------------------------------------------------------------
# Module & decomposition result
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A sub-model extracted from a larger system.

    Attributes:
        name: Identifier for the module.
        species: Set of species names internal to this module.
        input_species: Species whose values are provided by other modules.
        output_species: Species whose values are consumed by other modules.
        reactions: Indices or identifiers of reactions within this module.
        metadata: Extra information (e.g. dominant time-scale).
    """
    name: str
    species: FrozenSet[str]
    input_species: FrozenSet[str] = frozenset()
    output_species: FrozenSet[str] = frozenset()
    reactions: FrozenSet[str] = frozenset()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_species(self) -> FrozenSet[str]:
        return self.species | self.input_species | self.output_species


@dataclass
class QualityMetrics:
    """Quality metrics for a decomposition.

    Attributes:
        n_modules: Number of modules.
        size_balance: Coefficient of variation of module sizes (lower = better).
        interface_count: Total number of interface species.
        interface_minimality: Ratio of interface species to total species.
        max_coupling: Maximum coupling strength between any two modules.
        mean_coupling: Mean coupling strength across all module pairs.
    """
    n_modules: int = 0
    size_balance: float = 0.0
    interface_count: int = 0
    interface_minimality: float = 0.0
    max_coupling: float = 0.0
    mean_coupling: float = 0.0

    @property
    def score(self) -> float:
        """Composite quality score (higher is better)."""
        balance_penalty = self.size_balance
        interface_penalty = self.interface_minimality
        coupling_penalty = self.mean_coupling
        return max(0.0, 1.0 - 0.3 * balance_penalty
                   - 0.4 * interface_penalty
                   - 0.3 * coupling_penalty)


@dataclass
class DecompositionResult:
    """Output of a decomposition algorithm.

    Attributes:
        modules: Ordered list of extracted modules.
        interface_species: Species shared between at least two modules.
        coupling_strengths: Pairwise coupling strengths (module_i, module_j) → float.
        strategy: Strategy used.
        metrics: Quality metrics.
        metadata: Extra diagnostics.
    """
    modules: List[Module]
    interface_species: FrozenSet[str]
    coupling_strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)
    strategy: DecompositionStrategy = DecompositionStrategy.GREEDY
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Check invariants; return list of issues (empty = valid)."""
        issues: List[str] = []
        all_species: Set[str] = set()
        for mod in self.modules:
            overlap = all_species & mod.species
            if overlap:
                issues.append(
                    f"Species {overlap} assigned to multiple modules."
                )
            all_species |= mod.species
        return issues


# ---------------------------------------------------------------------------
# ModuleDecomposer
# ---------------------------------------------------------------------------

class ModuleDecomposer:
    """Decomposes a gene-regulatory network into verification modules.

    The decomposer works on a *networkx.DiGraph* where:
      - Nodes represent species.
      - Edges represent regulatory interactions (with optional attributes
        such as ``weight``, ``type`` = activation/repression, ``rate``).
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    # -- public API ---------------------------------------------------------

    def decompose(
        self,
        strategy: DecompositionStrategy = DecompositionStrategy.GREEDY,
        *,
        max_module_size: int = 20,
        min_module_size: int = 1,
        target_modules: Optional[int] = None,
        time_scales: Optional[Dict[str, float]] = None,
    ) -> DecompositionResult:
        """Decompose the network with the given strategy.

        Args:
            strategy: Decomposition algorithm to apply.
            max_module_size: Upper bound on species per module.
            min_module_size: Lower bound on species per module.
            target_modules: Desired number of modules (hint, not strict).
            time_scales: Map species → characteristic time-scale (for SLOW_FAST).

        Returns:
            DecompositionResult with modules, interfaces, and quality metrics.
        """
        dispatch = {
            DecompositionStrategy.SCC: self._decompose_scc,
            DecompositionStrategy.COMMUNITY: self._decompose_community,
            DecompositionStrategy.CASCADE: self._decompose_cascade,
            DecompositionStrategy.FANOUT: self._decompose_fanout,
            DecompositionStrategy.FEEDBACK_ISOLATION: self._decompose_feedback_isolation,
            DecompositionStrategy.SLOW_FAST: self._decompose_slow_fast,
            DecompositionStrategy.HIERARCHICAL: self._decompose_hierarchical,
            DecompositionStrategy.OPTIMAL_ILP: self._decompose_ilp,
            DecompositionStrategy.GREEDY: self._decompose_greedy,
        }
        fn = dispatch[strategy]
        partitions = fn(
            max_module_size=max_module_size,
            min_module_size=min_module_size,
            target_modules=target_modules,
            time_scales=time_scales,
        )
        result = self._build_result(partitions, strategy)
        logger.info(
            "Decomposition (%s): %d modules, %d interface species, score=%.3f",
            strategy.name, len(result.modules),
            len(result.interface_species), result.metrics.score,
        )
        return result

    def best_decomposition(
        self,
        strategies: Optional[Sequence[DecompositionStrategy]] = None,
        **kwargs: Any,
    ) -> DecompositionResult:
        """Try multiple strategies and return the best by quality score."""
        if strategies is None:
            strategies = [
                DecompositionStrategy.SCC,
                DecompositionStrategy.COMMUNITY,
                DecompositionStrategy.GREEDY,
            ]
        results = []
        for strat in strategies:
            try:
                results.append(self.decompose(strat, **kwargs))
            except Exception as exc:
                logger.debug("Strategy %s failed: %s", strat.name, exc)
        if not results:
            raise RuntimeError("All decomposition strategies failed.")
        return max(results, key=lambda r: r.metrics.score)

    def auto_decompose(
        self,
        model: Any = None,
        max_module_size: int = 5,
    ) -> DecompositionResult:
        """Automatically decompose a large model into verifiable modules.

        Uses graph-theoretic heuristics:
        1. Build species interaction graph (edges = reactions coupling species).
        2. Find weakly connected components.
        3. For large components, use spectral bisection to minimize coupling.
        4. Verify decomposition quality via coupling strength metric.

        If a :class:`BioModel` is provided, its reaction network is used
        to build the interaction graph.  Otherwise, the decomposer's
        existing graph is used.

        Args:
            model: Optional BioModel to extract the interaction graph from.
            max_module_size: Maximum species per module (default 5).

        Returns:
            DecompositionResult with quality-optimized modules.
        """
        if model is not None:
            self.graph = self._build_interaction_graph(model)

        n = len(self.graph)
        if n == 0:
            return DecompositionResult(
                modules=[], interface_species=frozenset(),
                strategy=DecompositionStrategy.GREEDY,
            )

        # Step 1: Find weakly connected components
        wcc = list(nx.weakly_connected_components(self.graph))
        logger.info(
            "Auto-decompose: %d species, %d weakly connected components",
            n, len(wcc),
        )

        partitions: List[Set[str]] = []
        for component in wcc:
            if len(component) <= max_module_size:
                partitions.append(set(component))
            else:
                # Step 2: Spectral bisection for large components
                sub_parts = self._spectral_bisect(
                    set(component), max_module_size,
                )
                partitions.extend(sub_parts)

        # Step 3: Try greedy merge if we have too many tiny modules
        partitions = self._merge_small(partitions, min_size=2)

        # Step 4: Build result and check quality
        result = self._build_result(
            partitions, DecompositionStrategy.GREEDY,
        )

        # If coupling is too high, try alternative strategies
        if result.metrics.max_coupling > 0.5:
            logger.info(
                "Auto-decompose: high coupling (%.3f), trying alternatives",
                result.metrics.max_coupling,
            )
            try:
                alt = self.best_decomposition(
                    max_module_size=max_module_size,
                )
                if alt.metrics.score > result.metrics.score:
                    result = alt
            except RuntimeError:
                pass

        logger.info(
            "Auto-decompose: %d modules, coupling=%.3f, score=%.3f",
            len(result.modules), result.metrics.max_coupling,
            result.metrics.score,
        )
        return result

    # -- spectral bisection helper --------------------------------------

    def _spectral_bisect(
        self, species: Set[str], max_size: int,
    ) -> List[Set[str]]:
        """Recursively bisect using the Fiedler vector of the graph Laplacian.

        Spectral bisection finds a balanced cut that minimizes the
        number of inter-partition edges, using the second-smallest
        eigenvector of the Laplacian matrix.

        Args:
            species: Set of species to partition.
            max_size: Maximum partition size.

        Returns:
            List of partitions, each with at most max_size species.
        """
        if len(species) <= max_size:
            return [species]

        sub = self.graph.subgraph(species).copy()
        undirected = sub.to_undirected()

        # If disconnected, split by components
        if not nx.is_connected(undirected):
            parts: List[Set[str]] = []
            for comp in nx.connected_components(undirected):
                parts.extend(self._spectral_bisect(set(comp), max_size))
            return parts

        nodes = sorted(undirected.nodes())
        n = len(nodes)

        if n <= 2:
            return [species]

        try:
            import numpy as np
            # Compute Laplacian and its Fiedler vector
            laplacian = nx.laplacian_matrix(undirected, nodelist=nodes)
            L_dense = laplacian.toarray().astype(float)
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            # Fiedler vector = eigenvector for second-smallest eigenvalue
            fiedler = eigenvectors[:, 1]

            # Bisect: species with fiedler < median go to partition A
            median = float(np.median(fiedler))
            part_a: Set[str] = set()
            part_b: Set[str] = set()
            for i, node in enumerate(nodes):
                if fiedler[i] <= median:
                    part_a.add(node)
                else:
                    part_b.add(node)

            # Ensure non-empty partitions
            if not part_a or not part_b:
                mid = n // 2
                part_a = set(nodes[:mid])
                part_b = set(nodes[mid:])

        except Exception:
            # Fallback to alphabetical split
            mid = n // 2
            part_a = set(nodes[:mid])
            part_b = set(nodes[mid:])

        return (self._spectral_bisect(part_a, max_size)
                + self._spectral_bisect(part_b, max_size))

    @staticmethod
    def _build_interaction_graph(model: Any) -> nx.DiGraph:
        """Build a species interaction graph from a BioModel.

        Creates a directed graph where nodes are species and edges
        represent regulatory/reaction coupling between species.
        """
        graph = nx.DiGraph()

        if not hasattr(model, 'species') or not hasattr(model, 'reactions'):
            return graph

        for sp in model.species:
            name = sp.name if hasattr(sp, 'name') else str(sp)
            graph.add_node(name)

        for rxn in model.reactions:
            reactant_names: List[str] = []
            product_names: List[str] = []

            if hasattr(rxn, 'reactants'):
                for entry in rxn.reactants:
                    name = (entry.species if hasattr(entry, 'species')
                            else str(entry))
                    reactant_names.append(name)
            if hasattr(rxn, 'products'):
                for entry in rxn.products:
                    name = (entry.species if hasattr(entry, 'species')
                            else str(entry))
                    product_names.append(name)

            # Add edges: each reactant influences each product
            for r in reactant_names:
                for p in product_names:
                    if r != p:
                        graph.add_edge(r, p, reaction=getattr(rxn, 'name', ''))

            # Add self-loops for degradation (reactant consumes itself)
            if product_names == [] and reactant_names:
                for r in reactant_names:
                    graph.add_edge(r, r, reaction=getattr(rxn, 'name', ''))

        return graph

    # -- strategy implementations -------------------------------------------

    def _decompose_scc(self, **kwargs: Any) -> List[Set[str]]:
        """Strongly connected components."""
        sccs = list(nx.strongly_connected_components(self.graph))
        # Sort by topological order of the DAG of SCCs
        condensation = nx.condensation(self.graph, scc=sccs)
        order = list(nx.topological_sort(condensation))
        return [sccs[i] for i in order]

    def _decompose_community(self, **kwargs: Any) -> List[Set[str]]:
        """Louvain community detection on the undirected view."""
        undirected = self.graph.to_undirected()
        if len(undirected) == 0:
            return []

        try:
            communities = nx.community.louvain_communities(
                undirected,
                resolution=kwargs.get("resolution", 1.0),
                seed=42,
            )
        except AttributeError:
            # Fallback for older networkx without louvain_communities
            communities = list(
                nx.community.greedy_modularity_communities(undirected)
            )
        return [set(c) for c in communities]

    def _decompose_cascade(self, **kwargs: Any) -> List[Set[str]]:
        """Cascade decomposition: identify maximal linear chains."""
        chains: List[Set[str]] = []
        visited: Set[str] = set()

        for node in nx.topological_sort(self._acyclic_view()):
            if node in visited:
                continue
            chain = self._trace_chain(node, visited)
            if chain:
                chains.append(chain)
                visited |= chain

        # Remaining nodes form their own modules
        remaining = set(self.graph.nodes) - visited
        if remaining:
            chains.append(remaining)
        return chains

    def _decompose_fanout(self, **kwargs: Any) -> List[Set[str]]:
        """Fan-out decomposition: group by shared upstream regulators."""
        partitions: List[Set[str]] = []
        visited: Set[str] = set()

        # Find fan-out hubs (nodes with out-degree > 1 and in-degree ≤ 1)
        hubs = [
            n for n in self.graph.nodes
            if self.graph.out_degree(n) > 1 and self.graph.in_degree(n) <= 1
        ]

        for hub in sorted(hubs, key=lambda n: -self.graph.out_degree(n)):
            if hub in visited:
                continue
            downstream = set(self.graph.successors(hub))
            group = {hub} | downstream
            partitions.append(group)
            visited |= group

        remaining = set(self.graph.nodes) - visited
        if remaining:
            partitions.append(remaining)
        return partitions

    def _decompose_feedback_isolation(self, **kwargs: Any) -> List[Set[str]]:
        """Identify minimal feedback vertex set and isolate feedback loops."""
        fvs = self._approximate_feedback_vertex_set()
        # Feedback nodes form one module; rest decomposes acyclically
        acyclic_graph = self.graph.copy()
        acyclic_graph.remove_nodes_from(fvs)

        acyclic_partitions: List[Set[str]] = []
        for comp in nx.weakly_connected_components(acyclic_graph):
            acyclic_partitions.append(set(comp))

        if fvs:
            acyclic_partitions.insert(0, fvs)
        return acyclic_partitions

    def _decompose_slow_fast(
        self, *, time_scales: Optional[Dict[str, float]] = None, **kwargs: Any
    ) -> List[Set[str]]:
        """Time-scale separation: group species by characteristic time-scale."""
        if time_scales is None:
            time_scales = self._estimate_time_scales()

        if not time_scales:
            logger.warning("No time-scale information; falling back to SCC.")
            return self._decompose_scc(**kwargs)

        # Cluster by log-scale
        import numpy as np
        species_list = sorted(time_scales.keys())
        log_scales = np.array([math.log10(max(time_scales[s], 1e-15))
                               for s in species_list])

        # Simple 1D k-means with 2 clusters (fast/slow)
        n_clusters = min(kwargs.get("target_modules", 2) or 2, len(species_list))
        if n_clusters < 2:
            return [set(species_list)]

        boundaries = self._cluster_1d(log_scales, n_clusters)
        partitions: List[Set[str]] = [set() for _ in range(n_clusters)]
        for sp, cluster_idx in zip(species_list, boundaries):
            partitions[cluster_idx].add(sp)

        return [p for p in partitions if p]

    def _decompose_hierarchical(self, **kwargs: Any) -> List[Set[str]]:
        """Recursive bisection via minimum vertex cut."""
        max_size = kwargs.get("max_module_size", 20)
        return self._recursive_bisect(set(self.graph.nodes), max_size)

    def _decompose_ilp(self, **kwargs: Any) -> List[Set[str]]:
        """Optimal decomposition via integer programming (small instances)."""
        n = len(self.graph)
        if n > 40:
            logger.warning(
                "ILP decomposition is expensive for n=%d; falling back to greedy.", n
            )
            return self._decompose_greedy(**kwargs)

        target_k = kwargs.get("target_modules") or max(2, n // 5)
        max_size = kwargs.get("max_module_size", 20)

        try:
            return self._solve_ilp(target_k, max_size)
        except Exception as exc:
            logger.warning("ILP solver failed (%s); falling back to greedy.", exc)
            return self._decompose_greedy(**kwargs)

    def _decompose_greedy(self, **kwargs: Any) -> List[Set[str]]:
        """Greedy decomposition minimizing inter-module edges."""
        max_size = kwargs.get("max_module_size", 20)
        min_size = kwargs.get("min_module_size", 1)
        target = kwargs.get("target_modules")

        # Start from SCCs, then merge small / split large
        partitions = self._decompose_scc(**kwargs)

        # Merge tiny modules greedily
        partitions = self._merge_small(partitions, min_size)

        # Split oversized modules
        result: List[Set[str]] = []
        for part in partitions:
            if len(part) > max_size:
                result.extend(self._recursive_bisect(part, max_size))
            else:
                result.append(part)

        # Merge to hit target count if specified
        if target and len(result) > target:
            result = self._merge_to_target(result, target)

        return result

    # -- internal helpers ---------------------------------------------------

    def _build_result(
        self, partitions: List[Set[str]], strategy: DecompositionStrategy
    ) -> DecompositionResult:
        """Convert raw partitions into a DecompositionResult with metrics."""
        modules: List[Module] = []
        all_assigned: Set[str] = set()

        for idx, part in enumerate(partitions):
            species = frozenset(part)
            # Inputs: predecessors outside this module
            inputs: Set[str] = set()
            outputs: Set[str] = set()
            for sp in part:
                for pred in self.graph.predecessors(sp):
                    if pred not in part:
                        inputs.add(pred)
                for succ in self.graph.successors(sp):
                    if succ not in part:
                        outputs.add(sp)

            modules.append(Module(
                name=f"M{idx}",
                species=species,
                input_species=frozenset(inputs),
                output_species=frozenset(outputs),
            ))
            all_assigned |= species

        # Interface species
        interface: Set[str] = set()
        for mod in modules:
            interface |= mod.input_species
            interface |= mod.output_species

        # Coupling strengths
        coupling: Dict[Tuple[str, str], float] = {}
        for i, mi in enumerate(modules):
            for j, mj in enumerate(modules):
                if i >= j:
                    continue
                strength = self._coupling_strength(mi.species, mj.species)
                coupling[(mi.name, mj.name)] = strength

        metrics = self._compute_metrics(modules, interface, coupling)

        return DecompositionResult(
            modules=modules,
            interface_species=frozenset(interface),
            coupling_strengths=coupling,
            strategy=strategy,
            metrics=metrics,
        )

    def _coupling_strength(
        self, a: FrozenSet[str], b: FrozenSet[str]
    ) -> float:
        """Count cross-edges normalised by total possible edges."""
        cross = 0
        for u, v in self.graph.edges:
            if (u in a and v in b) or (u in b and v in a):
                cross += 1
        total = len(a) * len(b)
        return cross / max(total, 1)

    def _compute_metrics(
        self,
        modules: List[Module],
        interface: Set[str],
        coupling: Dict[Tuple[str, str], float],
    ) -> QualityMetrics:
        import numpy as np

        sizes = [len(m.species) for m in modules]
        total = sum(sizes)
        mean_size = float(np.mean(sizes)) if sizes else 1.0
        std_size = float(np.std(sizes)) if len(sizes) > 1 else 0.0
        cv = std_size / mean_size if mean_size > 0 else 0.0

        coupling_vals = list(coupling.values()) if coupling else [0.0]

        return QualityMetrics(
            n_modules=len(modules),
            size_balance=cv,
            interface_count=len(interface),
            interface_minimality=len(interface) / max(total, 1),
            max_coupling=max(coupling_vals),
            mean_coupling=float(np.mean(coupling_vals)),
        )

    def _acyclic_view(self) -> nx.DiGraph:
        """Return a DAG by collapsing SCCs."""
        return nx.condensation(self.graph)

    def _trace_chain(self, start: str, visited: Set[str]) -> Set[str]:
        """Trace a linear chain from *start*."""
        chain: Set[str] = {start}
        current = start
        while True:
            succs = [s for s in self.graph.successors(current) if s not in visited]
            if len(succs) != 1:
                break
            nxt = succs[0]
            preds = list(self.graph.predecessors(nxt))
            if len(preds) != 1:
                break
            chain.add(nxt)
            visited.add(nxt)
            current = nxt
        return chain

    def _approximate_feedback_vertex_set(self) -> Set[str]:
        """Greedy approximation of minimum feedback vertex set."""
        g = self.graph.copy()
        fvs: Set[str] = set()
        while True:
            try:
                cycle = nx.find_cycle(g, orientation="original")
            except nx.NetworkXNoCycle:
                break
            # Remove node with highest degree in the cycle
            cycle_nodes = {u for u, _, _ in cycle}
            victim = max(cycle_nodes, key=lambda n: g.degree(n))
            fvs.add(victim)
            g.remove_node(victim)
        return fvs

    def _estimate_time_scales(self) -> Dict[str, float]:
        """Heuristic time-scale estimation from edge weights."""
        scales: Dict[str, float] = {}
        for node in self.graph.nodes:
            rates = []
            for _, _, data in self.graph.in_edges(node, data=True):
                r = data.get("rate") or data.get("weight", 1.0)
                rates.append(float(r))
            for _, _, data in self.graph.out_edges(node, data=True):
                r = data.get("rate") or data.get("weight", 1.0)
                rates.append(float(r))
            if rates:
                scales[node] = 1.0 / max(rates)
            else:
                scales[node] = 1.0
        return scales

    def _cluster_1d(self, values, k: int) -> List[int]:
        """Simple 1D k-means clustering. Returns cluster assignment indices."""
        import numpy as np
        n = len(values)
        if n <= k:
            return list(range(n))

        sorted_idx = np.argsort(values)
        chunk = n // k
        labels = [0] * n
        for i, idx in enumerate(sorted_idx):
            labels[idx] = min(i // chunk, k - 1)
        return labels

    def _recursive_bisect(
        self, species: Set[str], max_size: int
    ) -> List[Set[str]]:
        """Recursively bisect a species set until all parts ≤ max_size."""
        if len(species) <= max_size:
            return [species]

        sub = self.graph.subgraph(species).copy()
        undirected = sub.to_undirected()

        # Attempt minimum vertex cut
        try:
            if nx.is_connected(undirected):
                cut = nx.minimum_node_cut(undirected)
                undirected.remove_nodes_from(cut)
                components = list(nx.connected_components(undirected))
                if len(components) >= 2:
                    a = components[0] | cut
                    b = set()
                    for c in components[1:]:
                        b |= c
                    return (self._recursive_bisect(a, max_size)
                            + self._recursive_bisect(b, max_size))
        except nx.NetworkXError:
            pass

        # Fallback: split in half alphabetically
        ordered = sorted(species)
        mid = len(ordered) // 2
        return (self._recursive_bisect(set(ordered[:mid]), max_size)
                + self._recursive_bisect(set(ordered[mid:]), max_size))

    def _merge_small(
        self, partitions: List[Set[str]], min_size: int
    ) -> List[Set[str]]:
        """Merge partitions smaller than *min_size* into their best neighbour."""
        result = list(partitions)
        changed = True
        while changed:
            changed = False
            new_result: List[Set[str]] = []
            skip: Set[int] = set()
            for i, part in enumerate(result):
                if i in skip:
                    continue
                if len(part) < min_size and len(result) > 1:
                    best_j = self._best_merge_target(part, result, i)
                    if best_j is not None and best_j not in skip:
                        result[best_j] = result[best_j] | part
                        skip.add(i)
                        changed = True
                        continue
                new_result.append(part)
            result = new_result
        return result

    def _best_merge_target(
        self, part: Set[str], partitions: List[Set[str]], self_idx: int
    ) -> Optional[int]:
        """Find the partition that has the most edges to *part*."""
        best_j: Optional[int] = None
        best_score = -1
        for j, other in enumerate(partitions):
            if j == self_idx:
                continue
            score = sum(
                1 for u, v in self.graph.edges
                if (u in part and v in other) or (u in other and v in part)
            )
            if score > best_score:
                best_score = score
                best_j = j
        return best_j

    def _merge_to_target(
        self, partitions: List[Set[str]], target: int
    ) -> List[Set[str]]:
        """Greedily merge partitions until at most *target* remain."""
        result = list(partitions)
        while len(result) > target:
            # Find the pair with the strongest coupling
            best_pair: Optional[Tuple[int, int]] = None
            best_score = -1.0
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    score = self._coupling_strength(
                        frozenset(result[i]), frozenset(result[j])
                    )
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            result[i] = result[i] | result[j]
            del result[j]
        return result

    def _solve_ilp(self, k: int, max_size: int) -> List[Set[str]]:
        """Solve optimal decomposition via integer linear programming.

        Minimises the number of cross-module edges subject to a module-size
        constraint.  Uses scipy.optimize.milp when available.
        """
        from scipy.optimize import milp, Bounds, LinearConstraint
        import numpy as np

        nodes = sorted(self.graph.nodes)
        n = len(nodes)
        node_idx = {nd: i for i, nd in enumerate(nodes)}
        edges = list(self.graph.edges)

        # Decision variables: x[i, p] = 1 iff node i in partition p
        #                     y[e]    = 1 iff edge e is cross-partition
        n_x = n * k
        n_y = len(edges)
        n_vars = n_x + n_y

        # Objective: minimise sum of y_e
        c = np.zeros(n_vars)
        c[n_x:] = 1.0

        # Constraints
        A_eq_rows: List[np.ndarray] = []
        b_eq: List[float] = []

        # Each node in exactly one partition
        for i in range(n):
            row = np.zeros(n_vars)
            for p in range(k):
                row[i * k + p] = 1.0
            A_eq_rows.append(row)
            b_eq.append(1.0)

        A_ub_rows: List[np.ndarray] = []
        b_ub: List[float] = []

        # Partition size ≤ max_size
        for p in range(k):
            row = np.zeros(n_vars)
            for i in range(n):
                row[i * k + p] = 1.0
            A_ub_rows.append(row)
            b_ub.append(float(max_size))

        # Cross-edge indicator: y_e ≥ x[u,p] - x[v,p]  for each edge, partition
        for e_idx, (u, v) in enumerate(edges):
            ui = node_idx[u]
            vi = node_idx[v]
            for p in range(k):
                row = np.zeros(n_vars)
                row[ui * k + p] = 1.0
                row[vi * k + p] = -1.0
                row[n_x + e_idx] = -1.0
                A_ub_rows.append(row)
                b_ub.append(0.0)

        A_eq = np.array(A_eq_rows) if A_eq_rows else np.zeros((0, n_vars))
        b_eq_arr = np.array(b_eq) if b_eq else np.zeros(0)
        A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
        b_ub_arr = np.array(b_ub) if b_ub else np.zeros(0)

        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_vars)

        constraints = []
        if len(A_eq_rows) > 0:
            constraints.append(LinearConstraint(A_eq, b_eq_arr, b_eq_arr))
        if len(A_ub_rows) > 0:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub_arr))

        res = milp(c, integrality=integrality, bounds=bounds,
                   constraints=constraints)

        if not res.success:
            raise RuntimeError(f"ILP solver failed: {res.message}")

        x = res.x[:n_x].reshape(n, k)
        partitions: List[Set[str]] = [set() for _ in range(k)]
        for i in range(n):
            p = int(np.argmax(x[i]))
            partitions[p].add(nodes[i])

        return [p for p in partitions if p]


# ---------------------------------------------------------------------------
# Automatic decomposition heuristics
# ---------------------------------------------------------------------------

def spectral_decomposition(
    graph: nx.DiGraph,
    n_clusters: int = 2,
    jacobian: Optional[Any] = None,
) -> List[Set[str]]:
    """Use the Jacobian's eigenstructure to group tightly-coupled species.

    Computes the Laplacian of the interaction graph and uses its
    eigenvectors to identify clusters of species that are strongly
    coupled dynamically.  When a Jacobian matrix is provided, its
    eigenstructure is used directly for more accurate grouping.

    Args:
        graph: Species interaction graph (DiGraph).
        n_clusters: Desired number of clusters (default 2).
        jacobian: Optional Jacobian matrix (numpy array). If provided,
            its eigenvectors are used instead of the graph Laplacian.

    Returns:
        List of species sets, one per cluster.
    """
    import numpy as np

    nodes = sorted(graph.nodes())
    n = len(nodes)
    if n <= n_clusters:
        return [set(nodes)]

    if jacobian is not None and jacobian.shape == (n, n):
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        order = np.argsort(-np.abs(eigenvalues))
        k = min(n_clusters, n)
        features = np.real(eigenvectors[:, order[:k]])
    else:
        undirected = graph.to_undirected()
        if not nx.is_connected(undirected):
            partitions: List[Set[str]] = []
            for comp in nx.connected_components(undirected):
                partitions.append(set(comp))
            return partitions

        laplacian = nx.laplacian_matrix(undirected, nodelist=nodes)
        L_dense = laplacian.toarray().astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        k = min(n_clusters, n - 1)
        features = eigenvectors[:, 1:k + 1]

    labels = _kmeans_1d_multi(features, n_clusters)

    partitions_result: List[Set[str]] = [set() for _ in range(n_clusters)]
    for i, node in enumerate(nodes):
        partitions_result[labels[i]].add(node)

    return [p for p in partitions_result if p]


def time_scale_decomposition(
    graph: nx.DiGraph,
    time_scales: Optional[Dict[str, float]] = None,
    n_groups: int = 2,
) -> List[Set[str]]:
    """Group species by their characteristic timescales (fast vs slow).

    Species with similar timescales are grouped together, enabling
    separate verification of fast and slow subsystems.

    Args:
        graph: Species interaction graph (DiGraph).
        time_scales: Map species_name -> characteristic timescale (seconds).
            If None, timescales are estimated from edge weights.
        n_groups: Number of timescale groups (default 2: fast/slow).

    Returns:
        List of species sets, ordered from fastest to slowest.
    """
    import numpy as np

    nodes = sorted(graph.nodes())
    if not nodes:
        return []

    if time_scales is None:
        time_scales = _estimate_time_scales_from_graph(graph)

    default_scale = 1.0
    scales = np.array([
        math.log10(max(time_scales.get(sp, default_scale), 1e-15))
        for sp in nodes
    ])

    n_groups = min(n_groups, len(nodes))
    if n_groups < 2:
        return [set(nodes)]

    sorted_idx = np.argsort(scales)
    chunk = len(nodes) // n_groups
    labels = [0] * len(nodes)
    for rank, idx in enumerate(sorted_idx):
        labels[idx] = min(rank // max(chunk, 1), n_groups - 1)

    partitions: List[Set[str]] = [set() for _ in range(n_groups)]
    for i, node in enumerate(nodes):
        partitions[labels[i]].add(node)

    return [p for p in partitions if p]


class MinCutDecomposer:
    """Find minimum-cut decompositions of the species interaction graph.

    Uses graph min-cut algorithms to partition the species graph into
    modules with minimal inter-module coupling (fewest cut edges).
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    def decompose(
        self,
        n_modules: int = 2,
        max_module_size: Optional[int] = None,
    ) -> DecompositionResult:
        """Find a minimum-cut decomposition into n_modules modules.

        Args:
            n_modules: Target number of modules.
            max_module_size: Optional maximum module size constraint.

        Returns:
            DecompositionResult with minimum-cut partitions.
        """
        nodes = sorted(self.graph.nodes())
        n = len(nodes)
        if n == 0:
            return DecompositionResult(
                modules=[], interface_species=frozenset(),
                strategy=DecompositionStrategy.HIERARCHICAL,
            )

        if n_modules >= n:
            partitions = [{nd} for nd in nodes]
        else:
            partitions = self._recursive_min_cut(
                set(nodes), n_modules, max_module_size,
            )

        decomposer = ModuleDecomposer(self.graph)
        return decomposer._build_result(
            partitions, DecompositionStrategy.HIERARCHICAL,
        )

    def _recursive_min_cut(
        self,
        species: Set[str],
        target_parts: int,
        max_size: Optional[int],
    ) -> List[Set[str]]:
        """Recursively apply min-cut to partition species."""
        if target_parts <= 1 or len(species) <= 1:
            return [species]
        if max_size and len(species) <= max_size and target_parts <= 1:
            return [species]

        sub = self.graph.subgraph(species).copy()
        undirected = sub.to_undirected()

        if not nx.is_connected(undirected):
            parts: List[Set[str]] = [
                set(c) for c in nx.connected_components(undirected)
            ]
            return parts

        if len(species) <= 2:
            return [species]

        try:
            cut_value, (part_a_nodes, part_b_nodes) = nx.stoer_wagner(
                undirected
            )
            part_a = set(part_a_nodes)
            part_b = set(part_b_nodes)
        except (nx.NetworkXError, Exception):
            ordered = sorted(species)
            mid = len(ordered) // 2
            part_a = set(ordered[:mid])
            part_b = set(ordered[mid:])

        if not part_a or not part_b:
            return [species]

        left_target = max(1, target_parts // 2)
        right_target = max(1, target_parts - left_target)

        left_parts = self._recursive_min_cut(part_a, left_target, max_size)
        right_parts = self._recursive_min_cut(part_b, right_target, max_size)

        result = left_parts + right_parts

        if max_size:
            final: List[Set[str]] = []
            for part in result:
                if len(part) > max_size:
                    final.extend(self._recursive_min_cut(part, 2, max_size))
                else:
                    final.append(part)
            return final

        return result


# ---------------------------------------------------------------------------
# Private helpers for decomposition heuristics
# ---------------------------------------------------------------------------

def _kmeans_1d_multi(features: Any, k: int) -> List[int]:
    """Simple k-means on multi-dimensional feature vectors."""
    import numpy as np

    n = features.shape[0]
    if n <= k:
        return list(range(n))

    indices = np.linspace(0, n - 1, k, dtype=int)
    centroids = features[indices].copy()

    labels = [0] * n
    for _ in range(20):
        for i in range(n):
            dists = [np.sum((features[i] - centroids[c]) ** 2)
                     for c in range(k)]
            labels[i] = int(np.argmin(dists))

        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n):
            new_centroids[labels[i]] += features[i]
            counts[labels[i]] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def _estimate_time_scales_from_graph(graph: nx.DiGraph) -> Dict[str, float]:
    """Estimate characteristic timescales from graph edge weights."""
    scales: Dict[str, float] = {}
    for node in graph.nodes:
        rates: List[float] = []
        for _, _, data in graph.in_edges(node, data=True):
            r = data.get("rate") or data.get("weight", 1.0)
            rates.append(float(r))
        for _, _, data in graph.out_edges(node, data=True):
            r = data.get("rate") or data.get("weight", 1.0)
            rates.append(float(r))
        if rates:
            scales[node] = 1.0 / max(rates)
        else:
            scales[node] = 1.0
    return scales
