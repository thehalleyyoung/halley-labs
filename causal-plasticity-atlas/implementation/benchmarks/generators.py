"""Benchmark data generators for the CPA engine.

Implements three synthetic generator families from the CPA theory:

1. **FSVP** (Fixed Structure, Varying Parameters):
   Single DAG topology with varying regression weights across contexts.

2. **CSVM** (Changing Structure with Variable Mismatch):
   Topology changes across contexts with variable emergence.

3. **TPS** (Tipping-Point Scenario):
   Ordered contexts with abrupt transitions at known locations.

Plus a **SemiSyntheticGenerator** that perturbs real DAGs.

All generators produce reproducible results via explicit seeding.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as sp_stats


# =====================================================================
# Ground truth container
# =====================================================================


@dataclass
class GroundTruth:
    """Ground-truth information for a benchmark scenario.

    Attributes
    ----------
    adjacencies : Dict[str, np.ndarray]
        True adjacency matrix per context.
    parameters : Dict[str, np.ndarray]
        True weight matrix per context.
    variable_names : List[str]
        Ordered variable names.
    context_ids : List[str]
        Ordered context identifiers.
    classifications : Dict[str, str]
        True mechanism classification per variable.
    tipping_points : List[int]
        True tipping-point locations (context indices).
    invariant_variables : List[str]
        Variables with invariant mechanisms.
    plastic_variables : List[str]
        Variables with plastic mechanisms.
    emergent_variables : List[str]
        Variables that appear/disappear across contexts.
    metadata : Dict[str, Any]
        Additional ground-truth metadata.
    """

    adjacencies: Dict[str, np.ndarray] = field(default_factory=dict)
    parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    variable_names: List[str] = field(default_factory=list)
    context_ids: List[str] = field(default_factory=list)
    classifications: Dict[str, str] = field(default_factory=dict)
    tipping_points: List[int] = field(default_factory=list)
    invariant_variables: List[str] = field(default_factory=list)
    plastic_variables: List[str] = field(default_factory=list)
    emergent_variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_names": self.variable_names,
            "context_ids": self.context_ids,
            "classifications": self.classifications,
            "tipping_points": self.tipping_points,
            "invariant_variables": self.invariant_variables,
            "plastic_variables": self.plastic_variables,
            "emergent_variables": self.emergent_variables,
            "adjacencies": {
                k: v.tolist() for k, v in self.adjacencies.items()
            },
            "parameters": {
                k: v.tolist() for k, v in self.parameters.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Container for generated benchmark data.

    Attributes
    ----------
    context_data : Dict[str, np.ndarray]
        Generated data per context.
    ground_truth : GroundTruth
        Ground truth information.
    variable_names : List[str]
        Variable names.
    context_ids : List[str]
        Context identifiers.
    """

    context_data: Dict[str, np.ndarray] = field(default_factory=dict)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    variable_names: List[str] = field(default_factory=list)
    context_ids: List[str] = field(default_factory=list)


# =====================================================================
# DAG generation utilities
# =====================================================================


def _random_dag(
    p: int,
    density: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate a random DAG adjacency matrix (upper-triangular).

    Parameters
    ----------
    p : int
        Number of variables.
    density : float
        Edge probability.
    rng : np.random.RandomState
        Random state.

    Returns
    -------
    np.ndarray
        (p, p) binary adjacency matrix.
    """
    adj = np.zeros((p, p))
    order = rng.permutation(p)

    for idx_i in range(p):
        for idx_j in range(idx_i + 1, p):
            if rng.random() < density:
                i, j = order[idx_i], order[idx_j]
                adj[i, j] = 1.0

    return adj


def _random_weights(
    adj: np.ndarray,
    weight_range: Tuple[float, float],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Assign random weights to existing edges.

    Parameters
    ----------
    adj : np.ndarray
        Binary adjacency matrix.
    weight_range : tuple of float
        (min_abs_weight, max_abs_weight).
    rng : np.random.RandomState
        Random state.

    Returns
    -------
    np.ndarray
        Weighted adjacency matrix.
    """
    lo, hi = weight_range
    weights = adj.copy()
    edges = np.where(adj != 0)

    for i, j in zip(edges[0], edges[1]):
        sign = rng.choice([-1, 1])
        weights[i, j] = sign * rng.uniform(lo, hi)

    return weights


def _sample_linear_sem(
    weights: np.ndarray,
    n: int,
    noise_std: float,
    rng: np.random.RandomState,
    noise_type: str = "gaussian",
) -> np.ndarray:
    """Sample data from a linear SEM.

    Parameters
    ----------
    weights : np.ndarray
        (p, p) weight matrix.
    n : int
        Number of samples.
    noise_std : float
        Noise standard deviation.
    rng : np.random.RandomState
        Random state.
    noise_type : str
        'gaussian', 'uniform', or 'laplace'.

    Returns
    -------
    np.ndarray
        (n, p) data matrix.
    """
    p = weights.shape[0]

    if noise_type == "gaussian":
        noise = rng.randn(n, p) * noise_std
    elif noise_type == "uniform":
        noise = rng.uniform(-noise_std * np.sqrt(3), noise_std * np.sqrt(3), (n, p))
    elif noise_type == "laplace":
        noise = rng.laplace(0, noise_std / np.sqrt(2), (n, p))
    else:
        noise = rng.randn(n, p) * noise_std

    order = _topological_order(weights)

    data = noise.copy()
    for j in order:
        parents = np.where(weights[:, j] != 0)[0]
        for pa in parents:
            data[:, j] += weights[pa, j] * data[:, pa]

    return data


def _topological_order(adj: np.ndarray) -> List[int]:
    """Compute topological ordering using Kahn's algorithm.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix (may be weighted).

    Returns
    -------
    list of int
        Topological order (source → sink).
    """
    p = adj.shape[0]
    in_degree = np.sum(adj != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_degree[i] == 0]
    order: List[int] = []

    remaining = dict(enumerate(in_degree))
    adj_bin = (adj != 0).astype(int)

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(p):
            if adj_bin[node, child]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)

    if len(order) < p:
        remaining_nodes = [i for i in range(p) if i not in order]
        order.extend(remaining_nodes)

    return order


def _perturb_weights(
    weights: np.ndarray,
    perturbation_fraction: float,
    perturbation_scale: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Perturb a fraction of edge weights.

    Parameters
    ----------
    weights : np.ndarray
        Original weight matrix.
    perturbation_fraction : float
        Fraction of edges to perturb.
    perturbation_scale : float
        Scale of perturbation (relative to original weight).
    rng : np.random.RandomState
        Random state.

    Returns
    -------
    np.ndarray
        Perturbed weight matrix.
    """
    new_weights = weights.copy()
    edges = list(zip(*np.where(weights != 0)))

    if not edges:
        return new_weights

    n_perturb = max(1, int(len(edges) * perturbation_fraction))
    perturb_idx = rng.choice(len(edges), size=n_perturb, replace=False)

    for idx in perturb_idx:
        i, j = edges[idx]
        delta = rng.normal(0, perturbation_scale * abs(weights[i, j]))
        new_weights[i, j] += delta
        if abs(new_weights[i, j]) < 0.01:
            new_weights[i, j] = 0.01 * np.sign(weights[i, j])

    return new_weights


# =====================================================================
# Generator 1: FSVP (Fixed Structure, Varying Parameters)
# =====================================================================


class FSVPGenerator:
    """Generator 1: Fixed Structure, Varying Parameters.

    Creates a single DAG topology shared across all K contexts, with
    a controlled fraction of mechanisms having varying regression
    weights.

    Parameters
    ----------
    p : int
        Number of variables.
    K : int
        Number of contexts.
    n : int
        Samples per context.
    density : float
        DAG edge density.
    plasticity_fraction : float
        Fraction of mechanisms (edges) with varying parameters.
    weight_range : tuple of float
        (min, max) absolute edge weight.
    perturbation_scale : float
        Scale of parameter perturbation for plastic mechanisms.
    noise_std : float
        Noise standard deviation.
    noise_type : str
        Noise distribution ('gaussian', 'uniform', 'laplace').
    seed : int
        Random seed.

    Examples
    --------
    >>> gen = FSVPGenerator(p=5, K=3, n=200, seed=42)
    >>> result = gen.generate()
    >>> data, ground_truth = result.context_data, result.ground_truth
    """

    def __init__(
        self,
        p: int = 5,
        K: int = 3,
        n: int = 200,
        density: float = 0.4,
        plasticity_fraction: float = 0.5,
        weight_range: Tuple[float, float] = (0.3, 1.5),
        perturbation_scale: float = 0.5,
        noise_std: float = 1.0,
        noise_type: str = "gaussian",
        seed: int = 42,
    ) -> None:
        self.p = p
        self.K = K
        self.n = n
        self.density = density
        self.plasticity_fraction = plasticity_fraction
        self.weight_range = weight_range
        self.perturbation_scale = perturbation_scale
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.seed = seed

    def generate(self) -> BenchmarkResult:
        """Generate the benchmark scenario.

        Returns
        -------
        BenchmarkResult
            Generated data and ground truth.
        """
        rng = np.random.RandomState(self.seed)

        # Generate shared DAG structure
        adj = _random_dag(self.p, self.density, rng)
        base_weights = _random_weights(adj, self.weight_range, rng)

        variable_names = [f"X{i}" for i in range(self.p)]
        context_ids = [f"context_{k}" for k in range(self.K)]

        # Identify plastic vs invariant edges
        edges = list(zip(*np.where(adj != 0)))
        n_plastic = max(1, int(len(edges) * self.plasticity_fraction))
        plastic_edge_indices = set(
            rng.choice(len(edges), size=min(n_plastic, len(edges)), replace=False)
        )

        # Identify plastic vs invariant variables
        plastic_variables: Set[str] = set()
        for idx in plastic_edge_indices:
            i, j = edges[idx]
            plastic_variables.add(variable_names[j])

        invariant_variables = [
            v for v in variable_names if v not in plastic_variables
        ]

        # Generate per-context weights and data
        context_data: Dict[str, np.ndarray] = {}
        adjacencies: Dict[str, np.ndarray] = {}
        parameters: Dict[str, np.ndarray] = {}

        for k, cid in enumerate(context_ids):
            if k == 0:
                ctx_weights = base_weights.copy()
            else:
                ctx_weights = base_weights.copy()
                for edge_idx in plastic_edge_indices:
                    i, j = edges[edge_idx]
                    delta = rng.normal(
                        0, self.perturbation_scale * abs(base_weights[i, j])
                    )
                    ctx_weights[i, j] = base_weights[i, j] + delta
                    if abs(ctx_weights[i, j]) < 0.01:
                        ctx_weights[i, j] = 0.01 * np.sign(base_weights[i, j])

            adjacencies[cid] = adj.copy()
            parameters[cid] = ctx_weights
            context_data[cid] = _sample_linear_sem(
                ctx_weights, self.n, self.noise_std, rng, self.noise_type
            )

        # Build classifications
        classifications: Dict[str, str] = {}
        for v in variable_names:
            if v in plastic_variables:
                classifications[v] = "parametrically_plastic"
            else:
                classifications[v] = "invariant"

        ground_truth = GroundTruth(
            adjacencies=adjacencies,
            parameters=parameters,
            variable_names=variable_names,
            context_ids=context_ids,
            classifications=classifications,
            invariant_variables=invariant_variables,
            plastic_variables=list(plastic_variables),
            metadata={
                "generator": "FSVP",
                "p": self.p,
                "K": self.K,
                "n": self.n,
                "density": self.density,
                "plasticity_fraction": self.plasticity_fraction,
                "n_edges": len(edges),
                "n_plastic_edges": len(plastic_edge_indices),
                "seed": self.seed,
            },
        )

        return BenchmarkResult(
            context_data=context_data,
            ground_truth=ground_truth,
            variable_names=variable_names,
            context_ids=context_ids,
        )


# =====================================================================
# Generator 2: CSVM (Changing Structure with Variable Mismatch)
# =====================================================================


class CSVMGenerator:
    """Generator 2: Changing Structure with Variable Mismatch.

    Creates topology changes across contexts with variable
    appearance/disappearance (emergence).

    Parameters
    ----------
    p : int
        Number of variables (in the full variable set).
    K : int
        Number of contexts.
    n : int
        Samples per context.
    density : float
        Base DAG edge density.
    emergence_fraction : float
        Fraction of variables that appear/disappear.
    structural_change_fraction : float
        Fraction of edges added/removed between contexts.
    weight_range : tuple of float
        (min, max) absolute edge weight.
    noise_std : float
        Noise standard deviation.
    seed : int
        Random seed.

    Examples
    --------
    >>> gen = CSVMGenerator(p=8, K=4, n=200, emergence_fraction=0.2)
    >>> result = gen.generate()
    """

    def __init__(
        self,
        p: int = 8,
        K: int = 4,
        n: int = 200,
        density: float = 0.3,
        emergence_fraction: float = 0.2,
        structural_change_fraction: float = 0.3,
        weight_range: Tuple[float, float] = (0.3, 1.5),
        noise_std: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.p = p
        self.K = K
        self.n = n
        self.density = density
        self.emergence_fraction = emergence_fraction
        self.structural_change_fraction = structural_change_fraction
        self.weight_range = weight_range
        self.noise_std = noise_std
        self.seed = seed

    def generate(self) -> BenchmarkResult:
        """Generate the benchmark scenario.

        Returns
        -------
        BenchmarkResult
        """
        rng = np.random.RandomState(self.seed)

        variable_names = [f"X{i}" for i in range(self.p)]
        context_ids = [f"context_{k}" for k in range(self.K)]

        # Determine which variables "emerge" (appear/disappear)
        n_emergent = max(1, int(self.p * self.emergence_fraction))
        emergent_indices = set(
            rng.choice(self.p, size=n_emergent, replace=False)
        )
        emergent_variables = [variable_names[i] for i in emergent_indices]
        stable_indices = [i for i in range(self.p) if i not in emergent_indices]

        # Create emergence schedule: each emergent variable active in some contexts
        emergence_schedule: Dict[int, List[int]] = {}
        for var_idx in emergent_indices:
            n_active = rng.randint(1, max(2, self.K))
            active_contexts = sorted(
                rng.choice(self.K, size=n_active, replace=False)
            )
            emergence_schedule[var_idx] = list(active_contexts)

        # Generate base DAG
        base_adj = _random_dag(self.p, self.density, rng)
        base_weights = _random_weights(base_adj, self.weight_range, rng)

        context_data: Dict[str, np.ndarray] = {}
        adjacencies: Dict[str, np.ndarray] = {}
        parameters: Dict[str, np.ndarray] = {}

        for k, cid in enumerate(context_ids):
            ctx_adj = base_adj.copy()
            ctx_weights = base_weights.copy()

            # Apply structural changes
            edges = list(zip(*np.where(base_adj != 0)))
            if edges:
                n_change = max(1, int(
                    len(edges) * self.structural_change_fraction
                ))
                change_idx = rng.choice(
                    len(edges), size=min(n_change, len(edges)), replace=False
                )

                for idx in change_idx:
                    if rng.random() < 0.5:
                        i, j = edges[idx]
                        ctx_adj[i, j] = 0
                        ctx_weights[i, j] = 0
                    else:
                        i = rng.randint(self.p)
                        j = rng.randint(self.p)
                        if i != j and ctx_adj[i, j] == 0 and ctx_adj[j, i] == 0:
                            ctx_adj[i, j] = 1
                            sign = rng.choice([-1, 1])
                            ctx_weights[i, j] = sign * rng.uniform(*self.weight_range)

            # Apply emergence: zero out edges to/from inactive variables
            for var_idx, active_contexts in emergence_schedule.items():
                if k not in active_contexts:
                    ctx_adj[var_idx, :] = 0
                    ctx_adj[:, var_idx] = 0
                    ctx_weights[var_idx, :] = 0
                    ctx_weights[:, var_idx] = 0

            # Ensure DAG property (remove cycles)
            ctx_adj = _enforce_dag(ctx_adj, rng)
            ctx_weights = ctx_weights * (ctx_adj != 0).astype(float)

            adjacencies[cid] = ctx_adj
            parameters[cid] = ctx_weights
            context_data[cid] = _sample_linear_sem(
                ctx_weights, self.n, self.noise_std, rng
            )

        # Build classifications
        classifications: Dict[str, str] = {}
        structurally_plastic: Set[str] = set()

        for var_idx, var_name in enumerate(variable_names):
            if var_idx in emergent_indices:
                classifications[var_name] = "emergent"
                continue

            parent_sets = []
            for cid in context_ids:
                parents = frozenset(
                    j for j in range(self.p) if adjacencies[cid][j, var_idx] != 0
                )
                parent_sets.append(parents)

            if len(set(parent_sets)) > 1:
                classifications[var_name] = "structurally_plastic"
                structurally_plastic.add(var_name)
            else:
                weight_vecs = [
                    parameters[cid][:, var_idx] for cid in context_ids
                ]
                stacked = np.array(weight_vecs)
                if np.max(np.std(stacked, axis=0)) > 0.1:
                    classifications[var_name] = "parametrically_plastic"
                else:
                    classifications[var_name] = "invariant"

        invariant_vars = [
            v for v, c in classifications.items() if c == "invariant"
        ]
        plastic_vars = [
            v for v, c in classifications.items()
            if c in ("structurally_plastic", "parametrically_plastic", "fully_plastic")
        ]

        ground_truth = GroundTruth(
            adjacencies=adjacencies,
            parameters=parameters,
            variable_names=variable_names,
            context_ids=context_ids,
            classifications=classifications,
            invariant_variables=invariant_vars,
            plastic_variables=plastic_vars,
            emergent_variables=emergent_variables,
            metadata={
                "generator": "CSVM",
                "p": self.p,
                "K": self.K,
                "n": self.n,
                "emergence_fraction": self.emergence_fraction,
                "structural_change_fraction": self.structural_change_fraction,
                "emergence_schedule": {
                    str(k): v for k, v in emergence_schedule.items()
                },
                "seed": self.seed,
            },
        )

        return BenchmarkResult(
            context_data=context_data,
            ground_truth=ground_truth,
            variable_names=variable_names,
            context_ids=context_ids,
        )


# =====================================================================
# Generator 3: TPS (Tipping-Point Scenario)
# =====================================================================


class TPSGenerator:
    """Generator 3: Tipping-Point Scenario.

    Creates ordered contexts with abrupt transitions at specified
    tipping-point locations, producing both structural and parametric
    changes.

    Parameters
    ----------
    p : int
        Number of variables.
    K : int
        Number of ordered contexts.
    n : int
        Samples per context.
    density : float
        Base DAG edge density.
    n_tipping_points : int
        Number of tipping points.
    tipping_locations : list of int, optional
        Explicit tipping-point locations (context indices).
        Auto-spaced if None.
    structural_change_at_tp : float
        Fraction of edges changed at each tipping point.
    parametric_change_at_tp : float
        Scale of parametric changes at tipping points.
    gradual_drift : float
        Between-tipping-point gradual parameter drift.
    weight_range : tuple of float
        Edge weight range.
    noise_std : float
        Noise standard deviation.
    seed : int
        Random seed.

    Examples
    --------
    >>> gen = TPSGenerator(p=5, K=10, n=200, n_tipping_points=2)
    >>> result = gen.generate()
    >>> print(result.ground_truth.tipping_points)
    """

    def __init__(
        self,
        p: int = 5,
        K: int = 10,
        n: int = 200,
        density: float = 0.4,
        n_tipping_points: int = 2,
        tipping_locations: Optional[List[int]] = None,
        structural_change_at_tp: float = 0.3,
        parametric_change_at_tp: float = 1.0,
        gradual_drift: float = 0.05,
        weight_range: Tuple[float, float] = (0.3, 1.5),
        noise_std: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.p = p
        self.K = K
        self.n = n
        self.density = density
        self.n_tipping_points = n_tipping_points
        self.tipping_locations = tipping_locations
        self.structural_change_at_tp = structural_change_at_tp
        self.parametric_change_at_tp = parametric_change_at_tp
        self.gradual_drift = gradual_drift
        self.weight_range = weight_range
        self.noise_std = noise_std
        self.seed = seed

    def generate(self) -> BenchmarkResult:
        """Generate the benchmark scenario.

        Returns
        -------
        BenchmarkResult
        """
        rng = np.random.RandomState(self.seed)

        variable_names = [f"X{i}" for i in range(self.p)]
        context_ids = [f"context_{k}" for k in range(self.K)]

        # Determine tipping-point locations
        if self.tipping_locations is not None:
            tp_locs = sorted(self.tipping_locations)
        else:
            tp_locs = self._auto_tipping_locations()

        # Ensure valid locations
        tp_locs = [t for t in tp_locs if 1 <= t < self.K]

        # Generate segment DAGs
        n_segments = len(tp_locs) + 1
        segment_boundaries = [0] + tp_locs + [self.K]

        segment_adjs: List[np.ndarray] = []
        segment_weights: List[np.ndarray] = []

        base_adj = _random_dag(self.p, self.density, rng)
        base_weights = _random_weights(base_adj, self.weight_range, rng)

        segment_adjs.append(base_adj.copy())
        segment_weights.append(base_weights.copy())

        for seg in range(1, n_segments):
            prev_adj = segment_adjs[seg - 1].copy()
            prev_weights = segment_weights[seg - 1].copy()

            # Structural changes at tipping point
            new_adj = prev_adj.copy()
            new_weights = prev_weights.copy()

            edges = list(zip(*np.where(prev_adj != 0)))
            if edges:
                n_change = max(1, int(
                    len(edges) * self.structural_change_at_tp
                ))
                change_idx = rng.choice(
                    len(edges), size=min(n_change, len(edges)), replace=False
                )

                for idx in change_idx:
                    i, j = edges[idx]
                    if rng.random() < 0.5:
                        new_adj[i, j] = 0
                        new_weights[i, j] = 0
                    else:
                        delta = rng.normal(
                            0, self.parametric_change_at_tp * abs(prev_weights[i, j])
                        )
                        new_weights[i, j] = prev_weights[i, j] + delta

            # Add some new edges
            n_add = max(1, int(self.p * self.structural_change_at_tp * 0.5))
            for _ in range(n_add):
                i = rng.randint(self.p)
                j = rng.randint(self.p)
                if i != j and new_adj[i, j] == 0 and new_adj[j, i] == 0:
                    new_adj[i, j] = 1
                    sign = rng.choice([-1, 1])
                    new_weights[i, j] = sign * rng.uniform(*self.weight_range)

            new_adj = _enforce_dag(new_adj, rng)
            new_weights = new_weights * (new_adj != 0).astype(float)

            segment_adjs.append(new_adj)
            segment_weights.append(new_weights)

        # Generate data for each context
        context_data: Dict[str, np.ndarray] = {}
        adjacencies: Dict[str, np.ndarray] = {}
        parameters: Dict[str, np.ndarray] = {}

        for k, cid in enumerate(context_ids):
            # Determine which segment this context belongs to
            seg_idx = 0
            for s in range(len(tp_locs)):
                if k >= tp_locs[s]:
                    seg_idx = s + 1

            seg_adj = segment_adjs[seg_idx].copy()
            seg_weights = segment_weights[seg_idx].copy()

            # Apply gradual drift within segment
            seg_start = segment_boundaries[seg_idx]
            pos_in_seg = k - seg_start
            if pos_in_seg > 0:
                seg_weights = _perturb_weights(
                    seg_weights,
                    perturbation_fraction=1.0,
                    perturbation_scale=self.gradual_drift * pos_in_seg,
                    rng=rng,
                )

            adjacencies[cid] = seg_adj
            parameters[cid] = seg_weights
            context_data[cid] = _sample_linear_sem(
                seg_weights, self.n, self.noise_std, rng
            )

        # Build classifications
        classifications = self._build_classifications(
            variable_names, context_ids, adjacencies, parameters, tp_locs
        )

        invariant_vars = [
            v for v, c in classifications.items() if c == "invariant"
        ]
        plastic_vars = [
            v for v, c in classifications.items()
            if c != "invariant"
        ]

        ground_truth = GroundTruth(
            adjacencies=adjacencies,
            parameters=parameters,
            variable_names=variable_names,
            context_ids=context_ids,
            classifications=classifications,
            tipping_points=tp_locs,
            invariant_variables=invariant_vars,
            plastic_variables=plastic_vars,
            metadata={
                "generator": "TPS",
                "p": self.p,
                "K": self.K,
                "n": self.n,
                "n_tipping_points": len(tp_locs),
                "tipping_locations": tp_locs,
                "segment_boundaries": segment_boundaries,
                "seed": self.seed,
            },
        )

        return BenchmarkResult(
            context_data=context_data,
            ground_truth=ground_truth,
            variable_names=variable_names,
            context_ids=context_ids,
        )

    def _auto_tipping_locations(self) -> List[int]:
        """Auto-space tipping points evenly across contexts."""
        if self.n_tipping_points <= 0:
            return []
        step = self.K / (self.n_tipping_points + 1)
        return [int(round(step * (i + 1))) for i in range(self.n_tipping_points)]

    def _build_classifications(
        self,
        variable_names: List[str],
        context_ids: List[str],
        adjacencies: Dict[str, np.ndarray],
        parameters: Dict[str, np.ndarray],
        tp_locs: List[int],
    ) -> Dict[str, str]:
        """Classify mechanisms based on changes across tipping points."""
        classifications: Dict[str, str] = {}
        p = len(variable_names)

        for var_idx, var_name in enumerate(variable_names):
            parent_sets: List[FrozenSet[int]] = []
            weight_vectors: List[np.ndarray] = []

            for cid in context_ids:
                parents = frozenset(
                    j for j in range(p) if adjacencies[cid][j, var_idx] != 0
                )
                parent_sets.append(parents)
                weight_vectors.append(parameters[cid][:, var_idx].copy())

            unique_parents = set(parent_sets)
            has_structural_change = len(unique_parents) > 1

            stacked = np.array(weight_vectors)
            max_cv = 0.0
            for col in range(p):
                col_vals = stacked[:, col]
                col_mean = np.mean(np.abs(col_vals))
                if col_mean > 1e-6:
                    cv = np.std(col_vals) / col_mean
                    max_cv = max(max_cv, cv)

            has_parametric_change = max_cv > 0.3

            if has_structural_change and has_parametric_change:
                classifications[var_name] = "fully_plastic"
            elif has_structural_change:
                classifications[var_name] = "structurally_plastic"
            elif has_parametric_change:
                classifications[var_name] = "parametrically_plastic"
            else:
                classifications[var_name] = "invariant"

        return classifications


# =====================================================================
# Semi-synthetic generator
# =====================================================================


class SemiSyntheticGenerator:
    """Generate semi-synthetic benchmark data from a real DAG.

    Takes a real adjacency matrix and creates multiple synthetic
    contexts by applying controlled perturbations.

    Parameters
    ----------
    base_adjacency : np.ndarray
        Real (p, p) adjacency matrix.
    K : int
        Number of contexts to generate.
    n : int
        Samples per context.
    perturbation_types : list of str
        Types of perturbations: 'parametric', 'structural', 'emergence'.
    parametric_fraction : float
        Fraction of edges with parametric perturbation.
    structural_fraction : float
        Fraction of edges with structural perturbation.
    emergence_fraction : float
        Fraction of variables with emergence effects.
    weight_range : tuple of float
        Edge weight range for new/modified edges.
    noise_std : float
        Noise standard deviation.
    seed : int
        Random seed.

    Examples
    --------
    >>> adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> gen = SemiSyntheticGenerator(adj, K=4, n=200)
    >>> result = gen.generate()
    """

    def __init__(
        self,
        base_adjacency: np.ndarray,
        K: int = 4,
        n: int = 200,
        perturbation_types: Optional[List[str]] = None,
        parametric_fraction: float = 0.3,
        structural_fraction: float = 0.2,
        emergence_fraction: float = 0.1,
        weight_range: Tuple[float, float] = (0.3, 1.5),
        noise_std: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.base_adjacency = np.asarray(base_adjacency, dtype=float)
        self.K = K
        self.n = n
        self.perturbation_types = perturbation_types or [
            "parametric", "structural"
        ]
        self.parametric_fraction = parametric_fraction
        self.structural_fraction = structural_fraction
        self.emergence_fraction = emergence_fraction
        self.weight_range = weight_range
        self.noise_std = noise_std
        self.seed = seed

    def generate(self) -> BenchmarkResult:
        """Generate the benchmark scenario.

        Returns
        -------
        BenchmarkResult
        """
        rng = np.random.RandomState(self.seed)
        p = self.base_adjacency.shape[0]

        variable_names = [f"X{i}" for i in range(p)]
        context_ids = [f"context_{k}" for k in range(self.K)]

        base_adj = self.base_adjacency.copy()
        base_adj = _enforce_dag(base_adj, rng)

        # Assign weights to base adjacency
        base_weights = np.zeros_like(base_adj)
        for i in range(p):
            for j in range(p):
                if base_adj[i, j] != 0:
                    sign = rng.choice([-1, 1])
                    base_weights[i, j] = sign * rng.uniform(*self.weight_range)

        context_data: Dict[str, np.ndarray] = {}
        adjacencies: Dict[str, np.ndarray] = {}
        parameters: Dict[str, np.ndarray] = {}

        for k, cid in enumerate(context_ids):
            ctx_adj = base_adj.copy()
            ctx_weights = base_weights.copy()

            if k > 0:
                if "parametric" in self.perturbation_types:
                    ctx_weights = _perturb_weights(
                        ctx_weights,
                        self.parametric_fraction,
                        0.3 * k,
                        rng,
                    )

                if "structural" in self.perturbation_types:
                    ctx_adj, ctx_weights = self._structural_perturbation(
                        ctx_adj, ctx_weights, rng
                    )

                if "emergence" in self.perturbation_types:
                    ctx_adj, ctx_weights = self._emergence_perturbation(
                        ctx_adj, ctx_weights, rng
                    )

            ctx_adj = _enforce_dag(ctx_adj, rng)
            ctx_weights = ctx_weights * (ctx_adj != 0).astype(float)

            adjacencies[cid] = ctx_adj
            parameters[cid] = ctx_weights
            context_data[cid] = _sample_linear_sem(
                ctx_weights, self.n, self.noise_std, rng
            )

        # Build classifications
        classifications: Dict[str, str] = {}
        for var_idx, var_name in enumerate(variable_names):
            parent_sets = []
            weight_vecs = []
            for cid in context_ids:
                parents = frozenset(
                    j for j in range(p) if adjacencies[cid][j, var_idx] != 0
                )
                parent_sets.append(parents)
                weight_vecs.append(parameters[cid][:, var_idx])

            unique_parents = set(parent_sets)
            stacked = np.array(weight_vecs)
            max_std = np.max(np.std(stacked, axis=0))

            # Check for emergence
            has_no_parents = any(len(ps) == 0 for ps in parent_sets)
            has_parents = any(len(ps) > 0 for ps in parent_sets)

            if has_no_parents and has_parents:
                classifications[var_name] = "emergent"
            elif len(unique_parents) > 1:
                classifications[var_name] = "structurally_plastic"
            elif max_std > 0.1:
                classifications[var_name] = "parametrically_plastic"
            else:
                classifications[var_name] = "invariant"

        invariant_vars = [v for v, c in classifications.items() if c == "invariant"]
        plastic_vars = [
            v for v, c in classifications.items()
            if c in ("structurally_plastic", "parametrically_plastic", "fully_plastic")
        ]
        emergent_vars = [v for v, c in classifications.items() if c == "emergent"]

        ground_truth = GroundTruth(
            adjacencies=adjacencies,
            parameters=parameters,
            variable_names=variable_names,
            context_ids=context_ids,
            classifications=classifications,
            invariant_variables=invariant_vars,
            plastic_variables=plastic_vars,
            emergent_variables=emergent_vars,
            metadata={
                "generator": "SemiSynthetic",
                "p": p,
                "K": self.K,
                "n": self.n,
                "seed": self.seed,
            },
        )

        return BenchmarkResult(
            context_data=context_data,
            ground_truth=ground_truth,
            variable_names=variable_names,
            context_ids=context_ids,
        )

    def _structural_perturbation(
        self,
        adj: np.ndarray,
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply structural perturbation (add/remove edges)."""
        p = adj.shape[0]
        new_adj = adj.copy()
        new_weights = weights.copy()

        edges = list(zip(*np.where(adj != 0)))
        n_change = max(1, int(len(edges) * self.structural_fraction))

        # Remove some edges
        if edges:
            remove_idx = rng.choice(
                len(edges), size=min(n_change // 2 + 1, len(edges)), replace=False
            )
            for idx in remove_idx:
                i, j = edges[idx]
                new_adj[i, j] = 0
                new_weights[i, j] = 0

        # Add some edges
        for _ in range(n_change // 2 + 1):
            i = rng.randint(p)
            j = rng.randint(p)
            if i != j and new_adj[i, j] == 0 and new_adj[j, i] == 0:
                new_adj[i, j] = 1
                sign = rng.choice([-1, 1])
                new_weights[i, j] = sign * rng.uniform(*self.weight_range)

        return new_adj, new_weights

    def _emergence_perturbation(
        self,
        adj: np.ndarray,
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply emergence perturbation (disconnect some variables)."""
        p = adj.shape[0]
        new_adj = adj.copy()
        new_weights = weights.copy()

        n_emerge = max(1, int(p * self.emergence_fraction))
        emerge_vars = rng.choice(p, size=n_emerge, replace=False)

        for v in emerge_vars:
            new_adj[:, v] = 0
            new_weights[:, v] = 0

        return new_adj, new_weights


# =====================================================================
# Helper: enforce DAG property
# =====================================================================


def _enforce_dag(adj: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Remove cycles from an adjacency matrix by topological ordering.

    Parameters
    ----------
    adj : np.ndarray
        Possibly cyclic adjacency matrix.
    rng : np.random.RandomState
        Random state (for tie-breaking).

    Returns
    -------
    np.ndarray
        DAG adjacency matrix.
    """
    p = adj.shape[0]
    bin_adj = (adj != 0).astype(int)

    # Use a topological ordering heuristic:
    # compute in-degree, break ties randomly, then keep only
    # forward edges in the resulting ordering
    in_deg = np.sum(bin_adj, axis=0)
    noise = rng.random(p) * 0.01
    priority = in_deg + noise
    order = np.argsort(priority)

    rank = np.zeros(p, dtype=int)
    for pos, node in enumerate(order):
        rank[node] = pos

    dag_adj = adj.copy()
    for i in range(p):
        for j in range(p):
            if dag_adj[i, j] != 0 and rank[i] >= rank[j]:
                dag_adj[i, j] = 0

    return dag_adj
