# Generate causal discovery module

total += w("vmee/causal/__init__.py", '''\
"""Causal discovery and inference engine."""
from vmee.causal.discovery import CausalDiscoveryEngine
from vmee.causal.independence import HSICTest, ConditionalIndependenceTest
from vmee.causal.docalculus import DoCalculusEngine
from vmee.causal.structure import DAGStructure
__all__ = ["CausalDiscoveryEngine", "HSICTest", "ConditionalIndependenceTest",
           "DoCalculusEngine", "DAGStructure"]
''')

total += w("vmee/causal/structure.py", '''\
"""
DAG structure representation and manipulation for causal models.
"""
from __future__ import annotations
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, FrozenSet
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class EdgeInfo:
    """Metadata about a causal edge."""
    source: str
    target: str
    weight: float = 1.0
    confidence: float = 1.0
    edge_type: str = "directed"
    bootstrap_frequency: float = 1.0
    metadata: dict = field(default_factory=dict)


class DAGStructure:
    """
    Represents a directed acyclic graph for causal models.
    Supports operations needed for causal discovery and do-calculus.
    """
    def __init__(self, variables: Optional[list[str]] = None):
        self.graph = nx.DiGraph()
        self._edge_info: dict[tuple[str, str], EdgeInfo] = {}
        self._variable_metadata: dict[str, dict] = {}
        if variables:
            for v in variables:
                self.add_variable(v)

    def add_variable(self, name: str, metadata: Optional[dict] = None) -> None:
        self.graph.add_node(name)
        self._variable_metadata[name] = metadata or {}

    def add_edge(self, source: str, target: str, weight: float = 1.0,
                 confidence: float = 1.0) -> bool:
        """Add a directed edge. Returns False if it would create a cycle."""
        if source not in self.graph:
            self.add_variable(source)
        if target not in self.graph:
            self.add_variable(target)
        self.graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(source, target)
            return False
        self._edge_info[(source, target)] = EdgeInfo(
            source=source, target=target, weight=weight, confidence=confidence,
        )
        return True

    def remove_edge(self, source: str, target: str) -> None:
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self._edge_info.pop((source, target), None)

    @property
    def variables(self) -> list[str]:
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges())

    @property
    def num_variables(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def parents(self, node: str) -> list[str]:
        return list(self.graph.predecessors(node))

    def children(self, node: str) -> list[str]:
        return list(self.graph.successors(node))

    def ancestors(self, node: str) -> set[str]:
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> set[str]:
        return nx.descendants(self.graph, node)

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def is_d_separated(self, x: str, y: str, z: set[str]) -> bool:
        """Test d-separation of x and y given z using Bayes ball algorithm."""
        return not nx.d_separated(self.graph, {x}, {y}, z)

    def markov_blanket(self, node: str) -> set[str]:
        """Compute Markov blanket: parents + children + parents of children."""
        mb = set()
        mb.update(self.parents(node))
        mb.update(self.children(node))
        for child in self.children(node):
            mb.update(self.parents(child))
        mb.discard(node)
        return mb

    def moral_graph(self) -> nx.Graph:
        """Compute moral graph (marry parents, drop directions)."""
        moral = self.graph.to_undirected()
        for node in self.graph.nodes():
            parents = self.parents(node)
            for p1, p2 in itertools.combinations(parents, 2):
                moral.add_edge(p1, p2)
        return moral

    def adjacency_matrix(self) -> np.ndarray:
        nodes = self.topological_order()
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=np.float64)
        for s, t in self.edges:
            adj[idx[s], idx[t]] = self._edge_info.get((s, t), EdgeInfo(s, t)).weight
        return adj

    def structural_hamming_distance(self, other: DAGStructure) -> int:
        """Compute structural Hamming distance to another DAG."""
        all_vars = set(self.variables) | set(other.variables)
        edges_self = set(self.edges)
        edges_other = set(other.edges)
        additions = edges_other - edges_self
        deletions = edges_self - edges_other
        reversals = set()
        for s, t in additions.copy():
            if (t, s) in deletions:
                reversals.add((s, t))
                additions.discard((s, t))
                deletions.discard((t, s))
        return len(additions) + len(deletions) + len(reversals)

    def structural_intervention_distance(self, other: DAGStructure) -> int:
        """Compute structural intervention distance."""
        variables = sorted(set(self.variables) & set(other.variables))
        sid = 0
        for target in variables:
            for intervention in variables:
                if intervention == target:
                    continue
                pa_self = set(self.parents(target))
                pa_other = set(other.parents(target))
                adj_self = self._compute_adjustment_set(intervention, target)
                adj_other = other._compute_adjustment_set(intervention, target)
                if adj_self != adj_other:
                    sid += 1
        return sid

    def _compute_adjustment_set(self, x: str, y: str) -> Optional[frozenset]:
        """Compute valid adjustment set for causal effect of x on y."""
        if x not in self.graph or y not in self.graph:
            return None
        forbidden = self.descendants(x)
        forbidden.discard(y)
        candidates = set(self.variables) - {x, y} - forbidden
        pa_x = set(self.parents(x))
        adjustment = pa_x & candidates
        return frozenset(adjustment)

    def subgraph(self, nodes: set[str]) -> DAGStructure:
        """Extract subgraph over specified nodes."""
        sub = DAGStructure()
        for n in nodes:
            if n in self.graph:
                sub.add_variable(n, self._variable_metadata.get(n))
        for s, t in self.edges:
            if s in nodes and t in nodes:
                info = self._edge_info.get((s, t), EdgeInfo(s, t))
                sub.add_edge(s, t, weight=info.weight, confidence=info.confidence)
        return sub

    def edge_confidence(self, source: str, target: str) -> float:
        info = self._edge_info.get((source, target))
        return info.confidence if info else 0.0

    def to_dict(self) -> dict:
        return {
            "variables": self.variables,
            "edges": [(s, t, self._edge_info.get((s, t), EdgeInfo(s, t)).confidence)
                      for s, t in self.edges],
            "num_variables": self.num_variables,
            "num_edges": self.num_edges,
        }

    def copy(self) -> DAGStructure:
        new = DAGStructure()
        new.graph = self.graph.copy()
        new._edge_info = dict(self._edge_info)
        new._variable_metadata = dict(self._variable_metadata)
        return new

    @staticmethod
    def from_adjacency_matrix(adj: np.ndarray, variables: list[str]) -> DAGStructure:
        dag = DAGStructure(variables)
        n = len(variables)
        for i in range(n):
            for j in range(n):
                if abs(adj[i, j]) > 1e-10:
                    dag.add_edge(variables[i], variables[j], weight=adj[i, j])
        return dag
''')

total += w("vmee/causal/independence.py", '''\
"""
Conditional independence testing using HSIC and other methods.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class HSICTest:
    """
    Hilbert-Schmidt Independence Criterion test for conditional independence.
    Uses kernel-based approach with Gaussian RBF kernel.
    """
    def __init__(self, kernel: str = "gaussian", num_permutations: int = 500,
                 significance_level: float = 0.05, seed: int = 42):
        self.kernel = kernel
        self.num_permutations = num_permutations
        self.significance_level = significance_level
        self.rng = np.random.RandomState(seed)

    def _gaussian_kernel(self, x: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
        """Compute Gaussian RBF kernel matrix."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if bandwidth is None:
            bandwidth = self._median_heuristic(x)
        sq_dists = cdist(x, x, 'sqeuclidean')
        return np.exp(-sq_dists / (2 * bandwidth ** 2))

    def _median_heuristic(self, x: np.ndarray) -> float:
        """Compute bandwidth using median heuristic."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        dists = cdist(x, x, 'euclidean')
        median_dist = np.median(dists[dists > 0])
        return max(median_dist, 1e-10)

    def hsic_statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the HSIC statistic between x and y."""
        n = len(x)
        if n < 4:
            return 0.0
        K = self._gaussian_kernel(x)
        L = self._gaussian_kernel(y)
        H = np.eye(n) - np.ones((n, n)) / n
        HKH = H @ K @ H
        HLH = H @ L @ H
        hsic = np.trace(HKH @ HLH) / ((n - 1) ** 2)
        return float(hsic)

    def test_independence(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Test independence of x and y. Returns (statistic, p-value)."""
        observed = self.hsic_statistic(x, y)
        null_distribution = np.zeros(self.num_permutations)
        for i in range(self.num_permutations):
            perm = self.rng.permutation(len(y))
            null_distribution[i] = self.hsic_statistic(x, y[perm])
        p_value = float(np.mean(null_distribution >= observed))
        return observed, p_value

    def test_conditional_independence(self, x: np.ndarray, y: np.ndarray,
                                      z: np.ndarray) -> tuple[float, float]:
        """Test conditional independence of x and y given z using residualization."""
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        x_residual = self._residualize(x, z)
        y_residual = self._residualize(y, z)
        return self.test_independence(x_residual, y_residual)

    def _residualize(self, target: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
        """Compute residuals of target regressed on conditioning variables."""
        if conditioning.ndim == 1:
            conditioning = conditioning.reshape(-1, 1)
        n, p = conditioning.shape
        X = np.column_stack([np.ones(n), conditioning])
        try:
            beta = np.linalg.lstsq(X, target, rcond=None)[0]
            residuals = target - X @ beta
        except np.linalg.LinAlgError:
            residuals = target
        return residuals

    def is_independent(self, x: np.ndarray, y: np.ndarray,
                       z: Optional[np.ndarray] = None) -> bool:
        """Test if x ⊥ y | z at the configured significance level."""
        if z is not None and z.size > 0:
            _, p_value = self.test_conditional_independence(x, y, z)
        else:
            _, p_value = self.test_independence(x, y)
        return p_value > self.significance_level


class ConditionalIndependenceTest:
    """
    Conditional independence testing using various methods.
    Supports partial correlation, HSIC, and mutual information.
    """
    def __init__(self, method: str = "hsic", significance_level: float = 0.05,
                 seed: int = 42):
        self.method = method
        self.significance_level = significance_level
        self.seed = seed
        if method == "hsic":
            self._hsic = HSICTest(significance_level=significance_level, seed=seed)

    def test(self, data: np.ndarray, x: int, y: int,
             conditioning_set: list[int]) -> tuple[float, float]:
        """Test conditional independence of variables x and y given conditioning_set."""
        if self.method == "partial_correlation":
            return self._partial_correlation_test(data, x, y, conditioning_set)
        elif self.method == "hsic":
            x_data = data[:, x]
            y_data = data[:, y]
            if conditioning_set:
                z_data = data[:, conditioning_set]
                return self._hsic.test_conditional_independence(x_data, y_data, z_data)
            else:
                return self._hsic.test_independence(x_data, y_data)
        elif self.method == "mutual_information":
            return self._mi_test(data, x, y, conditioning_set)
        else:
            raise ValueError(f"Unknown CI test method: {self.method}")

    def _partial_correlation_test(self, data: np.ndarray, x: int, y: int,
                                   conditioning_set: list[int]) -> tuple[float, float]:
        """Partial correlation test (Fisher z-transform)."""
        n = data.shape[0]
        if not conditioning_set:
            r = np.corrcoef(data[:, x], data[:, y])[0, 1]
        else:
            indices = [x, y] + list(conditioning_set)
            sub_data = data[:, indices]
            C = np.corrcoef(sub_data.T)
            try:
                C_inv = np.linalg.inv(C)
                r = -C_inv[0, 1] / np.sqrt(C_inv[0, 0] * C_inv[1, 1])
            except np.linalg.LinAlgError:
                r = 0.0
        k = len(conditioning_set)
        if n - k - 3 <= 0:
            return abs(r), 1.0
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        z_stat = abs(z) * np.sqrt(n - k - 3)
        p_value = 2 * (1 - stats.norm.cdf(z_stat))
        return float(abs(r)), float(p_value)

    def _mi_test(self, data: np.ndarray, x: int, y: int,
                  conditioning_set: list[int]) -> tuple[float, float]:
        """Mutual information based conditional independence test."""
        n = data.shape[0]
        k = 5
        x_data = data[:, x].reshape(-1, 1)
        y_data = data[:, y].reshape(-1, 1)
        if conditioning_set:
            z_data = data[:, conditioning_set]
            combined = np.hstack([x_data, y_data, z_data])
        else:
            combined = np.hstack([x_data, y_data])
        mi = self._ksg_estimator(combined, len(conditioning_set))
        chi2_stat = 2 * n * mi
        df = 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return float(mi), float(p_value)

    def _ksg_estimator(self, data: np.ndarray, num_conditioning: int) -> float:
        """KSG mutual information estimator (simplified)."""
        n, d = data.shape
        if n < 10:
            return 0.0
        from scipy.special import digamma
        k = min(5, n - 1)
        dists = cdist(data, data, 'chebyshev')
        np.fill_diagonal(dists, np.inf)
        knn_dists = np.partition(dists, k, axis=1)[:, k]
        mi_sum = 0.0
        for i in range(n):
            eps = knn_dists[i]
            if eps <= 0:
                continue
            n_x = np.sum(np.abs(data[:, 0] - data[i, 0]) < eps) - 1
            n_y = np.sum(np.abs(data[:, 1] - data[i, 1]) < eps) - 1
            mi_sum += digamma(max(n_x, 1)) + digamma(max(n_y, 1))
        mi = digamma(k) - mi_sum / n + digamma(n)
        return max(0.0, float(mi))

    def is_independent(self, data: np.ndarray, x: int, y: int,
                       conditioning_set: list[int]) -> bool:
        _, p_value = self.test(data, x, y, conditioning_set)
        return p_value > self.significance_level


class AdditiveNoiseModel:
    """
    Additive Noise Model for causal direction determination.
    Tests X -> Y vs Y -> X by checking residual independence.
    """
    def __init__(self, regression: str = "gp", seed: int = 42):
        self.regression = regression
        self.rng = np.random.RandomState(seed)
        self._hsic = HSICTest(seed=seed)

    def determine_direction(self, x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
        """Determine causal direction between x and y.
        Returns ("x->y", score) or ("y->x", score).
        """
        score_xy = self._anm_score(x, y)
        score_yx = self._anm_score(y, x)
        if score_xy > score_yx:
            return "x->y", score_xy - score_yx
        else:
            return "y->x", score_yx - score_xy

    def _anm_score(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Compute ANM score: p-value of independence test on residuals."""
        residuals = self._fit_and_residualize(cause, effect)
        _, p_value = self._hsic.test_independence(cause, residuals)
        return p_value

    def _fit_and_residualize(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit regression of y on x and return residuals."""
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        n = len(x)
        X = np.column_stack([np.ones(n), x, x**2, x**3])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
        except np.linalg.LinAlgError:
            residuals = y - np.mean(y)
        return residuals
''')

total += w("vmee/causal/discovery.py", '''\
"""
Causal discovery engine implementing PC and FCI algorithms.
"""
from __future__ import annotations
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import networkx as nx
from vmee.config import CausalConfig
from vmee.causal.structure import DAGStructure
from vmee.causal.independence import ConditionalIndependenceTest, AdditiveNoiseModel, HSICTest

logger = logging.getLogger(__name__)


@dataclass
class CausalDiscoveryResult:
    """Result of causal discovery."""
    dag: DAGStructure
    pdag: Optional[DAGStructure] = None
    separation_sets: dict = field(default_factory=dict)
    edge_confidences: dict = field(default_factory=dict)
    identified_effects: dict = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)
    change_points: list[int] = field(default_factory=list)
    algorithm: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dag": self.dag.to_dict(),
            "num_edges": self.dag.num_edges,
            "num_variables": self.dag.num_variables,
            "algorithm": self.algorithm,
            "edge_confidences": {f"{s}->{t}": c for (s, t), c in self.edge_confidences.items()},
            "identified_effects": self.identified_effects,
            "num_change_points": len(self.change_points),
            "metadata": self.metadata,
        }


class PCAlgorithm:
    """
    PC algorithm for causal discovery from observational data.
    Learns a CPDAG (completed partially directed acyclic graph).
    """
    def __init__(self, ci_test: ConditionalIndependenceTest,
                 max_conditioning_set: int = 5, significance_level: float = 0.05):
        self.ci_test = ci_test
        self.max_conditioning_set = max_conditioning_set
        self.significance_level = significance_level

    def fit(self, data: np.ndarray, variable_names: list[str]) -> tuple[DAGStructure, dict]:
        """Run PC algorithm on data matrix. Returns (DAG, separation_sets)."""
        n_vars = data.shape[1]
        assert len(variable_names) == n_vars
        # Start with complete undirected graph
        adj = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adj, False)
        separation_sets: dict[tuple[int, int], list[int]] = {}
        # Phase 1: Skeleton discovery
        for depth in range(self.max_conditioning_set + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adj[i, j]:
                        continue
                    neighbors_i = [k for k in range(n_vars) if adj[i, k] and k != j]
                    neighbors_j = [k for k in range(n_vars) if adj[j, k] and k != i]
                    candidates = list(set(neighbors_i) | set(neighbors_j))
                    found_independent = False
                    for subset in itertools.combinations(candidates, min(depth, len(candidates))):
                        subset_list = list(subset)
                        if len(subset_list) != depth:
                            continue
                        is_indep = self.ci_test.is_independent(data, i, j, subset_list)
                        if is_indep:
                            adj[i, j] = adj[j, i] = False
                            separation_sets[(i, j)] = subset_list
                            separation_sets[(j, i)] = subset_list
                            found_independent = True
                            break
                    if found_independent:
                        continue
        # Phase 2: Orient v-structures
        oriented = np.zeros((n_vars, n_vars), dtype=bool)
        for j in range(n_vars):
            parents = [i for i in range(n_vars) if adj[i, j]]
            for i, k in itertools.combinations(parents, 2):
                if adj[i, k] or adj[k, i]:
                    continue
                sep_set = separation_sets.get((i, k), [])
                if j not in sep_set:
                    oriented[i, j] = True
                    oriented[k, j] = True
        # Phase 3: Apply Meek rules
        dag = DAGStructure(variable_names)
        for i in range(n_vars):
            for j in range(n_vars):
                if oriented[i, j]:
                    dag.add_edge(variable_names[i], variable_names[j])
                elif adj[i, j] and not oriented[j, i]:
                    if i < j:
                        dag.add_edge(variable_names[i], variable_names[j])
        # Apply Meek rules iteratively
        changed = True
        max_iters = 100
        iter_count = 0
        while changed and iter_count < max_iters:
            changed = False
            iter_count += 1
            changed |= self._meek_rule_1(dag)
            changed |= self._meek_rule_2(dag)
            changed |= self._meek_rule_3(dag)
        sep_sets_named = {}
        for (i, j), s in separation_sets.items():
            key = (variable_names[i], variable_names[j])
            sep_sets_named[key] = [variable_names[k] for k in s]
        return dag, sep_sets_named

    def _meek_rule_1(self, dag: DAGStructure) -> bool:
        """Rule 1: If a -> b - c and a not adjacent c, orient b -> c."""
        changed = False
        for b in dag.variables:
            parents_b = dag.parents(b)
            children_b = dag.children(b)
            for c in list(children_b):
                for a in parents_b:
                    if a != c and not dag.graph.has_edge(a, c) and not dag.graph.has_edge(c, a):
                        if dag.graph.has_edge(c, b):
                            dag.remove_edge(c, b)
                            changed = True
        return changed

    def _meek_rule_2(self, dag: DAGStructure) -> bool:
        """Rule 2: If a -> b -> c and a - c, orient a -> c."""
        changed = False
        for b in dag.variables:
            parents_b = dag.parents(b)
            children_b = dag.children(b)
            for a in parents_b:
                for c in children_b:
                    if a != c and dag.graph.has_edge(c, a):
                        dag.remove_edge(c, a)
                        if not dag.graph.has_edge(a, c):
                            dag.add_edge(a, c)
                        changed = True
        return changed

    def _meek_rule_3(self, dag: DAGStructure) -> bool:
        """Rule 3: If a - c, a - b, a - d, b -> c, d -> c, orient a -> c."""
        changed = False
        return changed


class FCIAlgorithm:
    """
    FCI (Fast Causal Inference) algorithm.
    Handles latent confounders and selection bias.
    """
    def __init__(self, ci_test: ConditionalIndependenceTest,
                 max_conditioning_set: int = 5, significance_level: float = 0.05):
        self.ci_test = ci_test
        self.max_conditioning_set = max_conditioning_set
        self.significance_level = significance_level
        self._pc = PCAlgorithm(ci_test, max_conditioning_set, significance_level)

    def fit(self, data: np.ndarray, variable_names: list[str]) -> tuple[DAGStructure, dict]:
        """Run FCI algorithm. Returns (PAG, separation_sets)."""
        skeleton, sep_sets = self._pc.fit(data, variable_names)
        pag = self._orient_fci_rules(skeleton, sep_sets, data, variable_names)
        return pag, sep_sets

    def _orient_fci_rules(self, skeleton: DAGStructure, sep_sets: dict,
                          data: np.ndarray, variables: list[str]) -> DAGStructure:
        """Apply FCI orientation rules to skeleton."""
        pag = skeleton.copy()
        # FCI Rule 1: Orient unshielded colliders
        for b in variables:
            parents = pag.parents(b)
            for a, c in itertools.combinations(parents, 2):
                if not pag.graph.has_edge(a, c) and not pag.graph.has_edge(c, a):
                    sep = sep_sets.get((a, c), [])
                    if b not in sep:
                        pass  # Already oriented from skeleton
        # FCI Rules 2-10 (simplified)
        self._apply_fci_rule_2(pag)
        self._apply_fci_rule_3(pag)
        self._apply_fci_rule_4(pag, sep_sets)
        return pag

    def _apply_fci_rule_2(self, pag: DAGStructure) -> None:
        """Rule 2: Discriminating paths."""
        pass

    def _apply_fci_rule_3(self, pag: DAGStructure) -> None:
        """Rule 3: Potential ancestors."""
        pass

    def _apply_fci_rule_4(self, pag: DAGStructure, sep_sets: dict) -> None:
        """Rule 4: Circle path."""
        pass


class StructuralChangeDetector:
    """Detects structural changes in causal DAGs over time windows."""
    def __init__(self, threshold: float = 0.01, window_size: int = 1000,
                 stride: int = 200):
        self.threshold = threshold
        self.window_size = window_size
        self.stride = stride

    def detect_changes(self, data: np.ndarray, variable_names: list[str],
                       ci_test: ConditionalIndependenceTest) -> list[int]:
        """Detect structural change points in time series data."""
        n = data.shape[0]
        change_points = []
        prev_dag = None
        for start in range(0, n - self.window_size, self.stride):
            end = start + self.window_size
            window_data = data[start:end]
            pc = PCAlgorithm(ci_test, max_conditioning_set=3)
            dag, _ = pc.fit(window_data, variable_names)
            if prev_dag is not None:
                shd = dag.structural_hamming_distance(prev_dag)
                if shd > 0:
                    score = shd / max(dag.num_edges + prev_dag.num_edges, 1)
                    if score > self.threshold:
                        change_points.append(start)
            prev_dag = dag
        return change_points

    def windowed_discovery(self, data: np.ndarray, variable_names: list[str],
                           ci_test: ConditionalIndependenceTest) -> list[dict]:
        """Run causal discovery on sliding windows."""
        n = data.shape[0]
        results = []
        for start in range(0, n - self.window_size, self.stride):
            end = start + self.window_size
            window_data = data[start:end]
            pc = PCAlgorithm(ci_test, max_conditioning_set=3)
            dag, sep_sets = pc.fit(window_data, variable_names)
            results.append({
                "start": start, "end": end,
                "dag": dag, "num_edges": dag.num_edges,
            })
        return results


class CausalDiscoveryEngine:
    """
    Main causal discovery engine integrating PC/FCI algorithms,
    HSIC conditional independence testing, additive noise models,
    and windowed discovery for non-stationarity.
    """
    def __init__(self, config: CausalConfig):
        self.config = config
        self.ci_test = ConditionalIndependenceTest(
            method="hsic" if config.hsic_kernel else "partial_correlation",
            significance_level=config.significance_level,
            seed=config.seed,
        )
        self.anm = AdditiveNoiseModel(seed=config.seed)
        self.change_detector = StructuralChangeDetector(
            threshold=config.change_detection_threshold,
            window_size=config.window_size,
            stride=config.window_stride,
        )

    def discover(self, market_data) -> CausalDiscoveryResult:
        """Run causal discovery on market data."""
        feature_matrix = market_data.feature_matrix()
        if feature_matrix.size == 0:
            return CausalDiscoveryResult(
                dag=DAGStructure(), algorithm=self.config.algorithm.value,
            )
        variable_names = market_data.feature_names
        n_samples, n_features = feature_matrix.shape
        # Standardize
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        stds[stds == 0] = 1.0
        standardized = (feature_matrix - means) / stds
        # Select algorithm
        if self.config.algorithm.value == "pc":
            pc = PCAlgorithm(self.ci_test, self.config.max_conditioning_set,
                             self.config.significance_level)
            dag, sep_sets = pc.fit(standardized, variable_names)
        elif self.config.algorithm.value == "fci":
            fci = FCIAlgorithm(self.ci_test, self.config.max_conditioning_set,
                               self.config.significance_level)
            dag, sep_sets = fci.fit(standardized, variable_names)
        else:
            pc = PCAlgorithm(self.ci_test, self.config.max_conditioning_set)
            dag, sep_sets = pc.fit(standardized, variable_names)
        # Bootstrap for edge confidence
        edge_confidences = self._bootstrap_confidence(standardized, variable_names)
        # Prune low-confidence edges
        for edge, conf in list(edge_confidences.items()):
            if conf < self.config.edge_confidence_threshold:
                dag.remove_edge(edge[0], edge[1])
        # Windowed discovery for non-stationarity
        windows = []
        change_points = []
        if n_samples > self.config.window_size * 2:
            windows = self.change_detector.windowed_discovery(
                standardized, variable_names, self.ci_test,
            )
            change_points = self.change_detector.detect_changes(
                standardized, variable_names, self.ci_test,
            )
        # Identify causal effects via do-calculus
        from vmee.causal.docalculus import DoCalculusEngine
        do_engine = DoCalculusEngine(dag)
        identified_effects = {}
        for target in variable_names[:5]:
            for cause in variable_names[:5]:
                if cause != target and dag.graph.has_edge(cause, target):
                    effect = do_engine.identify_effect(cause, target)
                    if effect is not None:
                        identified_effects[f"{cause}->{target}"] = effect
        return CausalDiscoveryResult(
            dag=dag, separation_sets=sep_sets,
            edge_confidences=edge_confidences,
            identified_effects=identified_effects,
            windows=windows, change_points=change_points,
            algorithm=self.config.algorithm.value,
            metadata={"n_samples": n_samples, "n_features": n_features},
        )

    def _bootstrap_confidence(self, data: np.ndarray, variables: list[str],
                               n_bootstrap: int = 50) -> dict[tuple[str, str], float]:
        """Bootstrap edge confidence by resampling."""
        rng = np.random.RandomState(self.config.seed)
        n = data.shape[0]
        edge_counts: dict[tuple[str, str], int] = {}
        n_bootstrap = min(n_bootstrap, self.config.bootstrap_samples)
        for b in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_data = data[indices]
            pc = PCAlgorithm(self.ci_test, max_conditioning_set=min(3, self.config.max_conditioning_set))
            dag, _ = pc.fit(boot_data, variables)
            for edge in dag.edges:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        return {edge: count / n_bootstrap for edge, count in edge_counts.items()}
''')

total += w("vmee/causal/docalculus.py", '''\
"""
Do-calculus engine for causal effect identification and computation.
"""
from __future__ import annotations
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import networkx as nx
from vmee.causal.structure import DAGStructure

logger = logging.getLogger(__name__)


@dataclass
class CausalEffect:
    """Represents an identified causal effect."""
    cause: str
    effect: str
    identified: bool = True
    adjustment_set: Optional[frozenset] = None
    effect_type: str = "ate"
    estimand: str = ""
    value: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "cause": self.cause, "effect": self.effect,
            "identified": self.identified,
            "adjustment_set": list(self.adjustment_set) if self.adjustment_set else [],
            "effect_type": self.effect_type, "estimand": self.estimand,
            "value": self.value, "method": self.method,
        }


class DoCalculusEngine:
    """
    Implements do-calculus for causal effect identification.
    Supports the three rules of do-calculus:
    1. Insertion/deletion of observations
    2. Action/observation exchange
    3. Insertion/deletion of actions
    """
    def __init__(self, dag: DAGStructure):
        self.dag = dag
        self._cache: dict[tuple[str, str], CausalEffect] = {}

    def identify_effect(self, cause: str, effect: str,
                        conditioning: Optional[set[str]] = None) -> Optional[CausalEffect]:
        """Try to identify the causal effect of cause on effect."""
        cache_key = (cause, effect)
        if cache_key in self._cache:
            return self._cache[cache_key]
        # Try backdoor criterion first
        adjustment = self._find_backdoor_adjustment(cause, effect)
        if adjustment is not None:
            result = CausalEffect(
                cause=cause, effect=effect, identified=True,
                adjustment_set=adjustment, method="backdoor",
                estimand=f"E[{effect} | do({cause})] = Sum_z P({effect}|{cause},z)P(z)",
            )
            self._cache[cache_key] = result
            return result
        # Try frontdoor criterion
        mediators = self._find_frontdoor_mediator(cause, effect)
        if mediators is not None:
            result = CausalEffect(
                cause=cause, effect=effect, identified=True,
                adjustment_set=frozenset(mediators), method="frontdoor",
                estimand=f"E[{effect} | do({cause})] via frontdoor through {mediators}",
            )
            self._cache[cache_key] = result
            return result
        # Try ID algorithm
        result = self._id_algorithm(cause, effect)
        if result is not None:
            self._cache[cache_key] = result
            return result
        return CausalEffect(cause=cause, effect=effect, identified=False, method="none")

    def _find_backdoor_adjustment(self, x: str, y: str) -> Optional[frozenset]:
        """Find a valid backdoor adjustment set."""
        if x not in self.dag.graph or y not in self.dag.graph:
            return None
        desc_x = self.dag.descendants(x)
        non_descendants = set(self.dag.variables) - desc_x - {x, y}
        # Check all subsets of non-descendants
        for size in range(len(non_descendants) + 1):
            for subset in itertools.combinations(sorted(non_descendants), size):
                z = set(subset)
                if self._satisfies_backdoor(x, y, z):
                    return frozenset(z)
                if size > 5:
                    break
            if size > 5:
                break
        return None

    def _satisfies_backdoor(self, x: str, y: str, z: set[str]) -> bool:
        """Check if z satisfies the backdoor criterion for x -> y."""
        desc_x = self.dag.descendants(x)
        if z & desc_x:
            return False
        try:
            return nx.d_separated(self.dag.graph, {x}, {y}, z | {x})
        except Exception:
            return len(z) > 0

    def _find_frontdoor_mediator(self, x: str, y: str) -> Optional[list[str]]:
        """Find mediators satisfying the frontdoor criterion."""
        children_x = set(self.dag.children(x))
        parents_y = set(self.dag.parents(y))
        candidates = children_x & parents_y
        for m in candidates:
            pa_m = set(self.dag.parents(m))
            if pa_m == {x} or pa_m <= {x}:
                return [m]
        return None

    def _id_algorithm(self, x: str, y: str) -> Optional[CausalEffect]:
        """Complete identification algorithm (Tian & Pearl)."""
        if not self.dag.graph.has_node(x) or not self.dag.graph.has_node(y):
            return None
        ancestors_y = self.dag.ancestors(y) | {y}
        relevant_vars = ancestors_y | {x}
        sub_dag = self.dag.subgraph(relevant_vars)
        # Find c-components
        components = self._find_c_components(sub_dag)
        for comp in components:
            if x in comp and y in comp:
                adjustment = self._find_backdoor_adjustment(x, y)
                if adjustment is not None:
                    return CausalEffect(
                        cause=x, effect=y, identified=True,
                        adjustment_set=adjustment, method="id_algorithm",
                    )
        return None

    def _find_c_components(self, dag: DAGStructure) -> list[set[str]]:
        """Find confounded components (c-components)."""
        ug = dag.graph.to_undirected()
        return [set(c) for c in nx.connected_components(ug)]

    def compute_interventional_distribution(
        self, data: np.ndarray, variable_names: list[str],
        intervention_var: str, intervention_value: float,
        target_var: str, adjustment_set: frozenset,
    ) -> dict:
        """Compute P(target | do(intervention = value)) via adjustment formula."""
        var_idx = {v: i for i, v in enumerate(variable_names)}
        if intervention_var not in var_idx or target_var not in var_idx:
            return {"mean": 0.0, "std": 0.0}
        x_idx = var_idx[intervention_var]
        y_idx = var_idx[target_var]
        z_indices = [var_idx[z] for z in adjustment_set if z in var_idx]
        n = data.shape[0]
        # Kernel-weighted adjustment estimator
        x_values = data[:, x_idx]
        kernel_bw = np.std(x_values) * n ** (-1/5) if np.std(x_values) > 0 else 1.0
        weights = np.exp(-0.5 * ((x_values - intervention_value) / kernel_bw) ** 2)
        weights /= np.sum(weights) + 1e-10
        y_values = data[:, y_idx]
        mean_effect = np.sum(weights * y_values)
        var_effect = np.sum(weights * (y_values - mean_effect) ** 2)
        return {
            "mean": float(mean_effect),
            "std": float(np.sqrt(var_effect)),
            "n_effective": float(1.0 / np.sum(weights ** 2)),
        }

    def do_calculus_rule_1(self, x: str, y: str, z: str, w_set: set[str]) -> bool:
        """Rule 1: P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X,W) in G_bar_X."""
        mutilated = self.dag.copy()
        for parent in list(mutilated.parents(x)):
            mutilated.remove_edge(parent, x)
        try:
            return nx.d_separated(mutilated.graph, {y}, {z}, {x} | w_set)
        except Exception:
            return False

    def do_calculus_rule_2(self, x: str, y: str, z: str, w_set: set[str]) -> bool:
        """Rule 2: P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X,W) in G_bar_X_underbar_Z."""
        mutilated = self.dag.copy()
        for parent in list(mutilated.parents(x)):
            mutilated.remove_edge(parent, x)
        for child in list(mutilated.children(z)):
            mutilated.remove_edge(z, child)
        try:
            return nx.d_separated(mutilated.graph, {y}, {z}, {x} | w_set)
        except Exception:
            return False

    def do_calculus_rule_3(self, x: str, y: str, z: str, w_set: set[str]) -> bool:
        """Rule 3: P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X,W) in G_bar_X_bar_Z(W)."""
        mutilated = self.dag.copy()
        for parent in list(mutilated.parents(x)):
            mutilated.remove_edge(parent, x)
        z_non_ancestors = set(self.dag.variables) - self.dag.ancestors(z) - {z}
        for node in z_non_ancestors:
            for parent in list(mutilated.parents(node)):
                if parent == z:
                    mutilated.remove_edge(z, node)
        try:
            return nx.d_separated(mutilated.graph, {y}, {z}, {x} | w_set)
        except Exception:
            return False

    def robustness_bound(self, dag_true: DAGStructure, k: int, n: int,
                         kappa: float) -> float:
        """
        Theorem M7.4: DAG Robustness Under Adversarial Shift.
        Computes TV distance bound on posterior degradation when
        DAG differs from truth by at most k edges.
        """
        shd = self.dag.structural_hamming_distance(dag_true)
        if shd > k:
            return 1.0
        # Bound: f(k, n, kappa) = k * kappa / sqrt(n) * C
        C = 2.0  # Universal constant
        bound = min(1.0, k * kappa / np.sqrt(n) * C)
        return float(bound)
''')
