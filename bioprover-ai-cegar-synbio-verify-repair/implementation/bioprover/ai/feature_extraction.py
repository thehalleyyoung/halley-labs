"""Feature extraction utilities for the BioProver AI heuristic engine.

Extracts structural, kinetic, topological, and monotonicity features from
:class:`BioModel` instances, counterexample traces, and CEGAR abstraction
states for consumption by the GNN encoder and predicate predictor.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_DEFAULT_FEATURE_DIM = 32


# ---------------------------------------------------------------------------
# Feature normalisation helpers
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Online Welford normaliser for feature vectors.

    Maintains running mean / variance and supports transform / inverse.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._count: int = 0
        self._mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._m2: np.ndarray = np.zeros(dim, dtype=np.float64)

    # -- incremental update --------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a single sample *x* of shape ``(dim,)``."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_batch(self, X: np.ndarray) -> None:
        """Update running stats with a batch *X* of shape ``(n, dim)``."""
        for i in range(X.shape[0]):
            self.update(X[i])

    # -- statistics ----------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + _EPS)

    # -- transform -----------------------------------------------------------

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalise *x*.  Supports shapes ``(dim,)`` or ``(n, dim)``."""
        return (x - self._mean) / self.std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """Reverse Z-score normalisation."""
        return z * self.std + self._mean

    # -- serialisation -------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "FeatureNormalizer":
        obj = cls(d["dim"])
        obj._count = d["count"]
        obj._mean = np.asarray(d["mean"], dtype=np.float64)
        obj._m2 = np.asarray(d["m2"], dtype=np.float64)
        return obj


class MinMaxNormalizer:
    """Min-max normaliser mapping features to [0, 1]."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._min: np.ndarray = np.full(dim, np.inf, dtype=np.float64)
        self._max: np.ndarray = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)

    def update_batch(self, X: np.ndarray) -> None:
        self._min = np.minimum(self._min, X.min(axis=0))
        self._max = np.maximum(self._max, X.max(axis=0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        rng = self._max - self._min
        rng = np.where(rng < _EPS, 1.0, rng)
        return (x - self._min) / rng

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        rng = self._max - self._min
        rng = np.where(rng < _EPS, 1.0, rng)
        return z * rng + self._min

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "MinMaxNormalizer":
        obj = cls(d["dim"])
        obj._min = np.asarray(d["min"], dtype=np.float64)
        obj._max = np.asarray(d["max"], dtype=np.float64)
        return obj


# ---------------------------------------------------------------------------
# CircuitFeatures
# ---------------------------------------------------------------------------


@dataclass
class CircuitFeatures:
    """Feature vector extracted from a :class:`BioModel`.

    Attributes
    ----------
    structural : np.ndarray
        [species_count, reaction_count, edge_count, feedback_loop_count,
         max_in_degree, max_out_degree, density]
    kinetic : np.ndarray
        [hill_coeff_mean, hill_coeff_std, hill_coeff_max,
         rate_const_mean, rate_const_std, rate_const_max, rate_const_min]
    topological : np.ndarray
        [connectivity, diameter, avg_clustering, avg_path_length,
         num_strongly_connected, largest_scc_size]
    monotonicity : np.ndarray
        [frac_monotone_positive, frac_monotone_negative,
         frac_non_monotone, monotonicity_index]
    """

    structural: np.ndarray = field(default_factory=lambda: np.zeros(7))
    kinetic: np.ndarray = field(default_factory=lambda: np.zeros(7))
    topological: np.ndarray = field(default_factory=lambda: np.zeros(6))
    monotonicity: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def to_vector(self) -> np.ndarray:
        """Concatenate all feature groups into a single 1-D vector."""
        return np.concatenate([
            self.structural, self.kinetic, self.topological, self.monotonicity
        ])

    @property
    def dim(self) -> int:
        return (
            len(self.structural)
            + len(self.kinetic)
            + len(self.topological)
            + len(self.monotonicity)
        )


def extract_circuit_features(model: Any) -> CircuitFeatures:
    """Extract :class:`CircuitFeatures` from a *BioModel* instance.

    Parameters
    ----------
    model
        A ``BioModel`` with ``.species``, ``.reactions``, and
        ``.species_names`` attributes.

    Returns
    -------
    CircuitFeatures
    """
    species_list = model.species
    reactions_list = model.reactions
    n_sp = len(species_list)
    n_rx = len(reactions_list)

    # -- structural features -------------------------------------------------
    edge_count = 0
    in_degree: Dict[str, int] = {s.name: 0 for s in species_list}
    out_degree: Dict[str, int] = {s.name: 0 for s in species_list}

    adj: Dict[str, Set[str]] = {s.name: set() for s in species_list}

    for rxn in reactions_list:
        reactant_names = {e.species_name for e in rxn.reactants}
        product_names = {e.species_name for e in rxn.products}
        modifier_names = set(rxn.modifiers) if rxn.modifiers else set()
        sources = reactant_names | modifier_names
        for src in sources:
            for tgt in product_names:
                if src in adj and tgt in in_degree:
                    adj[src].add(tgt)
                    out_degree[src] = out_degree.get(src, 0) + 1
                    in_degree[tgt] = in_degree.get(tgt, 0) + 1
                    edge_count += 1

    feedback_loops = _count_feedback_loops(adj)
    max_in = max(in_degree.values()) if in_degree else 0
    max_out = max(out_degree.values()) if out_degree else 0
    density = edge_count / max(n_sp * (n_sp - 1), 1)

    structural = np.array([
        n_sp, n_rx, edge_count, feedback_loops,
        max_in, max_out, density,
    ], dtype=np.float64)

    # -- kinetic features ----------------------------------------------------
    hill_coeffs: List[float] = []
    rate_consts: List[float] = []
    for rxn in reactions_list:
        kl = rxn.kinetic_law
        if kl is None:
            continue
        params = kl.parameters
        for k, v in params.items():
            k_lower = k.lower()
            if "n" == k_lower or "hill" in k_lower:
                hill_coeffs.append(float(v))
            else:
                rate_consts.append(abs(float(v)))

    kinetic = np.array([
        np.mean(hill_coeffs) if hill_coeffs else 0.0,
        np.std(hill_coeffs) if hill_coeffs else 0.0,
        np.max(hill_coeffs) if hill_coeffs else 0.0,
        np.mean(rate_consts) if rate_consts else 0.0,
        np.std(rate_consts) if rate_consts else 0.0,
        np.max(rate_consts) if rate_consts else 0.0,
        np.min(rate_consts) if rate_consts else 0.0,
    ], dtype=np.float64)

    # -- topological features ------------------------------------------------
    topological = _compute_topology_features(adj, n_sp)

    # -- monotonicity features -----------------------------------------------
    monotonicity = _compute_monotonicity_features(reactions_list)

    return CircuitFeatures(
        structural=structural,
        kinetic=kinetic,
        topological=topological,
        monotonicity=monotonicity,
    )


def _count_feedback_loops(adj: Dict[str, Set[str]], max_length: int = 6) -> int:
    """Count elementary feedback loops up to *max_length* via DFS."""
    nodes = list(adj.keys())
    visited_global: Set[Tuple[str, ...]] = set()
    count = 0
    for start in nodes:
        stack: List[Tuple[str, List[str]]] = [(start, [start])]
        while stack:
            current, path = stack.pop()
            if len(path) > max_length:
                continue
            for nbr in adj.get(current, set()):
                if nbr == start and len(path) >= 2:
                    canon = _canonical_cycle(path)
                    if canon not in visited_global:
                        visited_global.add(canon)
                        count += 1
                elif nbr not in path and len(path) < max_length:
                    stack.append((nbr, path + [nbr]))
    return count


def _canonical_cycle(path: List[str]) -> Tuple[str, ...]:
    """Return a rotation-invariant canonical form for a cycle."""
    min_idx = path.index(min(path))
    rotated = path[min_idx:] + path[:min_idx]
    return tuple(rotated)


def _compute_topology_features(
    adj: Dict[str, Set[str]], n_sp: int,
) -> np.ndarray:
    """Compute topological features from an adjacency dict."""
    if n_sp == 0:
        return np.zeros(6, dtype=np.float64)

    # BFS for connectivity / diameter / avg path length
    nodes = list(adj.keys())
    total_paths = 0
    total_length = 0
    max_dist = 0

    for src in nodes:
        dist = _bfs_distances(adj, src)
        for d in dist.values():
            if d < float("inf") and d > 0:
                total_paths += 1
                total_length += d
                max_dist = max(max_dist, d)

    connectivity = total_paths / max(n_sp * (n_sp - 1), 1)
    diameter = float(max_dist)
    avg_path = total_length / max(total_paths, 1)

    # Clustering coefficient (directed)
    clustering_sum = 0.0
    for node in nodes:
        neighbours = adj.get(node, set())
        k = len(neighbours)
        if k < 2:
            continue
        links = 0
        for u in neighbours:
            for v in neighbours:
                if u != v and v in adj.get(u, set()):
                    links += 1
        clustering_sum += links / (k * (k - 1))
    avg_clustering = clustering_sum / max(n_sp, 1)

    # Strongly connected components (Tarjan-like via iterative Kosaraju)
    sccs = _kosaraju_scc(adj, nodes)
    num_scc = len(sccs)
    largest_scc = max(len(c) for c in sccs) if sccs else 0

    return np.array([
        connectivity, diameter, avg_clustering, avg_path,
        float(num_scc), float(largest_scc),
    ], dtype=np.float64)


def _bfs_distances(adj: Dict[str, Set[str]], src: str) -> Dict[str, int]:
    dist: Dict[str, int] = {src: 0}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _kosaraju_scc(
    adj: Dict[str, Set[str]], nodes: List[str],
) -> List[List[str]]:
    """Compute SCCs using Kosaraju's algorithm."""
    visited: Set[str] = set()
    order: List[str] = []

    def _dfs1(u: str) -> None:
        stack = [(u, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for v in adj.get(node, set()):
                if v not in visited:
                    stack.append((v, False))

    for n in nodes:
        if n not in visited:
            _dfs1(n)

    # Build reverse graph
    radj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for u, nbrs in adj.items():
        for v in nbrs:
            radj[v].add(u)

    visited2: Set[str] = set()
    sccs: List[List[str]] = []

    def _dfs2(u: str) -> List[str]:
        comp: List[str] = []
        stack = [u]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            comp.append(node)
            for v in radj.get(node, set()):
                if v not in visited2:
                    stack.append(v)
        return comp

    for n in reversed(order):
        if n not in visited2:
            comp = _dfs2(n)
            if comp:
                sccs.append(comp)
    return sccs


def _compute_monotonicity_features(reactions: Sequence[Any]) -> np.ndarray:
    """Compute monotonicity features from kinetic laws."""
    pos = 0
    neg = 0
    non_mono = 0

    for rxn in reactions:
        kl = rxn.kinetic_law
        if kl is None:
            continue
        klass = type(kl).__name__.lower()
        if "activation" in klass or "constitutive" in klass or "production" in klass:
            pos += 1
        elif "repression" in klass or "degradation" in klass:
            neg += 1
        else:
            non_mono += 1

    total = max(pos + neg + non_mono, 1)
    frac_pos = pos / total
    frac_neg = neg / total
    frac_non = non_mono / total
    mono_index = (pos + neg) / total  # fraction that are monotone

    return np.array([frac_pos, frac_neg, frac_non, mono_index], dtype=np.float64)


# ---------------------------------------------------------------------------
# CounterexampleFeatures
# ---------------------------------------------------------------------------


@dataclass
class CounterexampleFeatures:
    """Features extracted from a counterexample trace.

    Attributes
    ----------
    trajectory : np.ndarray
        [oscillation_score, convergence_rate, divergence_rate,
         max_amplitude, trajectory_length]
    violation : np.ndarray
        [violation_time, violation_species_idx, violation_margin,
         violation_severity, time_fraction]
    path : np.ndarray
        [path_length, branching_factor, state_space_coverage]
    """

    trajectory: np.ndarray = field(default_factory=lambda: np.zeros(5))
    violation: np.ndarray = field(default_factory=lambda: np.zeros(5))
    path: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.trajectory, self.violation, self.path])

    @property
    def dim(self) -> int:
        return len(self.trajectory) + len(self.violation) + len(self.path)


def extract_counterexample_features(
    states: np.ndarray,
    times: Optional[np.ndarray] = None,
    violation_time: Optional[float] = None,
    violation_species: Optional[int] = None,
    violation_margin: float = 0.0,
) -> CounterexampleFeatures:
    """Extract features from a counterexample state trajectory.

    Parameters
    ----------
    states
        Array of shape ``(T, n_species)`` representing the trajectory.
    times
        Array of shape ``(T,)`` with time stamps.  Auto-generated if *None*.
    violation_time
        Time at which the specification was violated.
    violation_species
        Index of species responsible for the violation.
    violation_margin
        Margin by which the specification was violated.
    """
    T, n_sp = states.shape
    if times is None:
        times = np.arange(T, dtype=np.float64)

    # -- trajectory features -------------------------------------------------
    diffs = np.diff(states, axis=0)
    sign_changes = np.sum(np.diff(np.sign(diffs), axis=0) != 0)
    oscillation = float(sign_changes) / max(T * n_sp, 1)

    norms = np.linalg.norm(diffs, axis=1)
    if T > 2:
        convergence = float(np.mean(norms[:T // 2]) - np.mean(norms[T // 2:]))
    else:
        convergence = 0.0
    divergence = float(np.max(norms)) if norms.size > 0 else 0.0
    max_amp = float(np.max(states) - np.min(states))

    trajectory_feats = np.array([
        oscillation, convergence, divergence, max_amp, float(T),
    ], dtype=np.float64)

    # -- violation features --------------------------------------------------
    v_time = violation_time if violation_time is not None else float(times[-1])
    v_species = float(violation_species) if violation_species is not None else -1.0
    v_margin = float(violation_margin)
    severity = abs(v_margin) / (max_amp + _EPS)
    time_frac = v_time / (float(times[-1]) + _EPS)

    violation_feats = np.array([
        v_time, v_species, v_margin, severity, time_frac,
    ], dtype=np.float64)

    # -- path features -------------------------------------------------------
    path_length = float(np.sum(norms))
    unique_states = len(set(map(tuple, np.round(states, decimals=4))))
    branching = float(unique_states) / max(T, 1)
    coverage = float(unique_states) / max(n_sp ** 2, 1)

    path_feats = np.array([path_length, branching, coverage], dtype=np.float64)

    return CounterexampleFeatures(
        trajectory=trajectory_feats,
        violation=violation_feats,
        path=path_feats,
    )


# ---------------------------------------------------------------------------
# AbstractionFeatures
# ---------------------------------------------------------------------------


@dataclass
class AbstractionFeatures:
    """Features characterising the current CEGAR abstraction state.

    Attributes
    ----------
    basic : np.ndarray
        [abstraction_size, depth, predicate_count]
    coverage : np.ndarray
        [state_coverage, transition_coverage, spec_coverage]
    history : np.ndarray
        [refinements_so_far, avg_predicate_effectiveness,
         predicates_added_last_step, time_since_last_progress]
    """

    basic: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coverage: np.ndarray = field(default_factory=lambda: np.zeros(3))
    history: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.basic, self.coverage, self.history])

    @property
    def dim(self) -> int:
        return len(self.basic) + len(self.coverage) + len(self.history)


def extract_abstraction_features(
    abstraction_size: int = 0,
    depth: int = 0,
    predicate_count: int = 0,
    state_coverage: float = 0.0,
    transition_coverage: float = 0.0,
    spec_coverage: float = 0.0,
    refinements_so_far: int = 0,
    avg_effectiveness: float = 0.0,
    predicates_added_last: int = 0,
    time_since_progress: float = 0.0,
) -> AbstractionFeatures:
    """Build :class:`AbstractionFeatures` from scalar quantities."""
    return AbstractionFeatures(
        basic=np.array([
            float(abstraction_size), float(depth), float(predicate_count),
        ], dtype=np.float64),
        coverage=np.array([
            state_coverage, transition_coverage, spec_coverage,
        ], dtype=np.float64),
        history=np.array([
            float(refinements_so_far), avg_effectiveness,
            float(predicates_added_last), time_since_progress,
        ], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance_permutation(
    model_fn: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Permutation-based feature importance.

    Parameters
    ----------
    model_fn
        Callable ``(X) -> predictions`` (numpy arrays).
    X
        Feature matrix ``(n_samples, n_features)``.
    y
        Target vector ``(n_samples,)``.
    n_repeats
        Number of permutation repeats per feature.
    rng
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Importance scores of shape ``(n_features,)``.  Higher = more important.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    baseline_pred = model_fn(X)
    baseline_mse = float(np.mean((baseline_pred - y) ** 2))

    n_features = X.shape[1]
    importances = np.zeros(n_features, dtype=np.float64)

    for f in range(n_features):
        deltas = np.zeros(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            perm_pred = model_fn(X_perm)
            perm_mse = float(np.mean((perm_pred - y) ** 2))
            deltas[r] = perm_mse - baseline_mse
        importances[f] = float(np.mean(deltas))

    return importances


def feature_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation between features.

    Parameters
    ----------
    X
        Feature matrix ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Correlation matrix ``(n_features, n_features)``.
    """
    if X.shape[0] < 2:
        return np.eye(X.shape[1], dtype=np.float64)
    return np.corrcoef(X, rowvar=False)


def select_features_variance(
    X: np.ndarray, threshold: float = 0.01,
) -> np.ndarray:
    """Return indices of features whose variance exceeds *threshold*."""
    variances = np.var(X, axis=0)
    return np.where(variances > threshold)[0]
