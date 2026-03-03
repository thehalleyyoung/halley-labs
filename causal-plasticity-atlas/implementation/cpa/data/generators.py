"""Synthetic data generators (SB1–SB5) from the evaluation strategy.

Each scenario (SB1–SB5) targets a different class of causal plasticity:

* **SB1** – Fixed structure, varying parameters (parametric plasticity).
* **SB2** – Structure-varying, shared parents (structural plasticity).
* **SB3** – Emergence / disappearance of edges.
* **SB4** – Gradual drift of edge weights (tipping point detection).
* **SB5** – Mixed plasticity (combination of all types).

Every generator produces a dictionary with:

* ``"datasets"`` – list of ``np.ndarray`` of shape ``(n, p)``.
* ``"dags"`` – list of adjacency matrices.
* ``"ground_truth_plasticity"`` – dict mapping ``(i, j)`` edge tuples
  to :class:`PlasticityClass` values.
* ``"context_labels"`` – list of context label strings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cpa.core.types import PlasticityClass


# ===================================================================
# Helper functions
# ===================================================================


def _random_dag(p: int, density: float, rng: np.random.Generator) -> NDArray:
    """Generate a random DAG by sampling an upper-triangular matrix
    and applying a random permutation.

    Parameters
    ----------
    p : int
        Number of variables.
    density : float
        Edge probability in (0, 1).
    rng : Generator

    Returns
    -------
    NDArray, shape (p, p)
        Binary adjacency matrix (i→j iff adj[i,j]=1).
    """
    upper = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < density:
                upper[i, j] = 1.0
    perm = rng.permutation(p)
    adj = upper[np.ix_(perm, perm)]
    return adj


def _random_weights(
    adj: NDArray,
    low: float = 0.5,
    high: float = 2.0,
    rng: np.random.Generator = None,
) -> NDArray:
    """Assign random nonzero weights to edges in an adjacency matrix.

    Weights are uniformly sampled from ``[low, high]`` with random sign.
    """
    rng = rng or np.random.default_rng()
    weights = np.zeros_like(adj, dtype=np.float64)
    edges = np.argwhere(adj != 0)
    for i, j in edges:
        w = rng.uniform(low, high)
        if rng.random() < 0.5:
            w = -w
        weights[i, j] = w
    return weights


def _sample_linear_gaussian(
    adj_weights: NDArray,
    n: int,
    noise_std: float = 1.0,
    rng: np.random.Generator = None,
) -> NDArray:
    """Sample from a linear-Gaussian SCM with given weighted adjacency.

    Parameters
    ----------
    adj_weights : NDArray, shape (p, p)
        Weighted adjacency matrix (adj[i,j] = coefficient of i→j).
    n : int
        Number of samples.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    rng : Generator

    Returns
    -------
    NDArray, shape (n, p)
    """
    rng = rng or np.random.default_rng()
    p = adj_weights.shape[0]

    order = _topological_sort(adj_weights)
    data = np.zeros((n, p), dtype=np.float64)
    for j in order:
        parents = np.where(adj_weights[:, j] != 0)[0]
        noise = rng.normal(0, noise_std, size=n)
        if len(parents) == 0:
            data[:, j] = noise
        else:
            data[:, j] = sum(adj_weights[pa, j] * data[:, pa] for pa in parents) + noise
    return data


def _topological_sort(adj: NDArray) -> List[int]:
    """Topological sort via Kahn's algorithm."""
    p = adj.shape[0]
    in_deg = np.sum(adj != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_deg[i] == 0]
    order: List[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(p):
            if adj[node, j] != 0:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order


def _classify_edge_plasticity(
    dags: List[NDArray],
    weight_matrices: List[NDArray],
    param_tol: float = 0.1,
) -> Dict[Tuple[int, int], PlasticityClass]:
    """Classify each edge across contexts.

    Parameters
    ----------
    dags : list of binary adjacency matrices
    weight_matrices : list of weighted adjacency matrices
    param_tol : float
        Tolerance for declaring parametric invariance.

    Returns
    -------
    dict mapping (i, j) to PlasticityClass
    """
    K = len(dags)
    p = dags[0].shape[0]
    all_edges: set = set()
    for dag in dags:
        for i in range(p):
            for j in range(p):
                if dag[i, j] != 0:
                    all_edges.add((i, j))

    result: Dict[Tuple[int, int], PlasticityClass] = {}

    for edge in all_edges:
        i, j = edge
        present = [dag[i, j] != 0 for dag in dags]
        weights = [wm[i, j] for wm in weight_matrices]

        all_present = all(present)
        none_present = not any(present)
        some_present = any(present) and not all(present)

        if none_present:
            continue

        if some_present:
            n_present = sum(present)
            if n_present <= K // 2:
                result[edge] = PlasticityClass.EMERGENT
            else:
                result[edge] = PlasticityClass.STRUCTURAL_PLASTIC
        elif all_present:
            present_weights = [w for w, p in zip(weights, present) if p]
            max_diff = max(abs(a - b) for a in present_weights for b in present_weights)
            if max_diff < param_tol:
                result[edge] = PlasticityClass.INVARIANT
            else:
                result[edge] = PlasticityClass.PARAMETRIC_PLASTIC

    return result


# ===================================================================
# SyntheticGenerator class
# ===================================================================


class SyntheticGenerator:
    """Configurable multi-context synthetic data generator.

    Parameters
    ----------
    n_nodes : int
        Number of observed variables.
    n_contexts : int
        Number of distinct contexts to generate.
    n_samples_per_context : int
        Samples drawn per context.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_nodes: int,
        n_contexts: int,
        n_samples_per_context: int,
        seed: Optional[int] = None,
    ) -> None:
        self._n_nodes = n_nodes
        self._n_contexts = n_contexts
        self._n_samples = n_samples_per_context
        self._rng = np.random.default_rng(seed)
        self._dags: Dict[str, NDArray] = {}
        self._weight_matrices: Dict[str, NDArray] = {}
        self._ground_truth: Dict[Tuple[int, int], PlasticityClass] = {}

    def generate(self, scenario: str = "sb1") -> Dict[str, Any]:
        """Generate multi-context data for the given scenario.

        Parameters
        ----------
        scenario : str
            One of ``"sb1"`` … ``"sb5"``.

        Returns
        -------
        dict
            Keys: ``"datasets"``, ``"dags"``, ``"ground_truth_plasticity"``,
            ``"context_labels"``.
        """
        dispatch = {
            "sb1": self._generate_sb1,
            "sb2": self._generate_sb2,
            "sb3": self._generate_sb3,
            "sb4": self._generate_sb4,
            "sb5": self._generate_sb5,
        }
        if scenario not in dispatch:
            raise ValueError(f"Unknown scenario {scenario!r}")
        return dispatch[scenario]()

    def true_dags(self) -> Dict[str, NDArray]:
        """Return the ground-truth DAG per context."""
        return dict(self._dags)

    def true_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return ground-truth plasticity labels per edge."""
        return dict(self._ground_truth)

    # -----------------------------------------------------------------
    # SB1: Fixed Structure, Varying Parameters
    # -----------------------------------------------------------------

    def _generate_sb1(self) -> Dict[str, Any]:
        p, K, n = self._n_nodes, self._n_contexts, self._n_samples
        base_dag = _random_dag(p, density=2.0 / p, rng=self._rng)

        datasets: List[NDArray] = []
        dags: List[NDArray] = []
        weights_list: List[NDArray] = []
        labels: List[str] = []

        for k in range(K):
            w = _random_weights(base_dag, low=0.5, high=2.0, rng=self._rng)
            data = _sample_linear_gaussian(w, n, rng=self._rng)
            datasets.append(data)
            dags.append(base_dag.copy())
            weights_list.append(w)
            labels.append(f"ctx_{k}")
            self._dags[f"ctx_{k}"] = base_dag.copy()
            self._weight_matrices[f"ctx_{k}"] = w

        self._ground_truth = _classify_edge_plasticity(dags, weights_list)
        return {
            "datasets": datasets,
            "dags": dags,
            "ground_truth_plasticity": self._ground_truth,
            "context_labels": labels,
        }

    # -----------------------------------------------------------------
    # SB2: Structure-Varying, Shared Parents
    # -----------------------------------------------------------------

    def _generate_sb2(self) -> Dict[str, Any]:
        p, K, n = self._n_nodes, self._n_contexts, self._n_samples
        base_dag = _random_dag(p, density=2.0 / p, rng=self._rng)

        datasets: List[NDArray] = []
        dags: List[NDArray] = []
        weights_list: List[NDArray] = []
        labels: List[str] = []

        n_edges_to_flip = max(1, int(np.sum(base_dag != 0) * 0.3))

        for k in range(K):
            dag_k = base_dag.copy()
            if k > 0:
                edges = list(zip(*np.where(dag_k != 0)))
                non_edges = [
                    (i, j)
                    for i in range(p)
                    for j in range(p)
                    if i != j and dag_k[i, j] == 0
                ]
                flips = min(n_edges_to_flip, len(edges))
                for _ in range(flips):
                    if edges and self._rng.random() < 0.5:
                        idx = self._rng.integers(len(edges))
                        ei, ej = edges[idx]
                        dag_k[ei, ej] = 0.0
                    elif non_edges:
                        idx = self._rng.integers(len(non_edges))
                        ei, ej = non_edges[idx]
                        dag_k[ei, ej] = 1.0
                        if _has_cycle(dag_k):
                            dag_k[ei, ej] = 0.0

            w = _random_weights(dag_k, rng=self._rng)
            data = _sample_linear_gaussian(w, n, rng=self._rng)
            datasets.append(data)
            dags.append(dag_k)
            weights_list.append(w)
            labels.append(f"ctx_{k}")
            self._dags[f"ctx_{k}"] = dag_k
            self._weight_matrices[f"ctx_{k}"] = w

        self._ground_truth = _classify_edge_plasticity(dags, weights_list)
        return {
            "datasets": datasets,
            "dags": dags,
            "ground_truth_plasticity": self._ground_truth,
            "context_labels": labels,
        }

    # -----------------------------------------------------------------
    # SB3: Emergence / Disappearance
    # -----------------------------------------------------------------

    def _generate_sb3(self) -> Dict[str, Any]:
        p, K, n = self._n_nodes, self._n_contexts, self._n_samples
        base_dag = _random_dag(p, density=2.0 / p, rng=self._rng)

        datasets: List[NDArray] = []
        dags: List[NDArray] = []
        weights_list: List[NDArray] = []
        labels: List[str] = []

        for k in range(K):
            dag_k = base_dag.copy()
            edges = list(zip(*np.where(dag_k != 0)))
            n_remove = max(1, int(len(edges) * k / (2 * K)))
            if k > 0 and edges:
                remove_idx = self._rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False)
                for idx in remove_idx:
                    ei, ej = edges[idx]
                    dag_k[ei, ej] = 0.0

            if k > K // 2:
                n_add = max(1, (k - K // 2))
                for _ in range(n_add):
                    tries = 0
                    while tries < 20:
                        i_new = self._rng.integers(p)
                        j_new = self._rng.integers(p)
                        if i_new != j_new and dag_k[i_new, j_new] == 0:
                            dag_k[i_new, j_new] = 1.0
                            if _has_cycle(dag_k):
                                dag_k[i_new, j_new] = 0.0
                            else:
                                break
                        tries += 1

            w = _random_weights(dag_k, rng=self._rng)
            data = _sample_linear_gaussian(w, n, rng=self._rng)
            datasets.append(data)
            dags.append(dag_k)
            weights_list.append(w)
            labels.append(f"ctx_{k}")
            self._dags[f"ctx_{k}"] = dag_k
            self._weight_matrices[f"ctx_{k}"] = w

        self._ground_truth = _classify_edge_plasticity(dags, weights_list)
        return {
            "datasets": datasets,
            "dags": dags,
            "ground_truth_plasticity": self._ground_truth,
            "context_labels": labels,
        }

    # -----------------------------------------------------------------
    # SB4: Gradual Drift
    # -----------------------------------------------------------------

    def _generate_sb4(self) -> Dict[str, Any]:
        p, K, n = self._n_nodes, self._n_contexts, self._n_samples
        base_dag = _random_dag(p, density=2.0 / p, rng=self._rng)
        base_weights = _random_weights(base_dag, rng=self._rng)

        drift_rate = _random_weights(base_dag, low=0.02, high=0.1, rng=self._rng)

        datasets: List[NDArray] = []
        dags: List[NDArray] = []
        weights_list: List[NDArray] = []
        labels: List[str] = []

        for k in range(K):
            w = base_weights + drift_rate * k
            data = _sample_linear_gaussian(w, n, rng=self._rng)
            datasets.append(data)
            dags.append(base_dag.copy())
            weights_list.append(w.copy())
            labels.append(f"ctx_{k}")
            self._dags[f"ctx_{k}"] = base_dag.copy()
            self._weight_matrices[f"ctx_{k}"] = w.copy()

        self._ground_truth = _classify_edge_plasticity(dags, weights_list, param_tol=0.05)
        return {
            "datasets": datasets,
            "dags": dags,
            "ground_truth_plasticity": self._ground_truth,
            "context_labels": labels,
        }

    # -----------------------------------------------------------------
    # SB5: Mixed Plasticity
    # -----------------------------------------------------------------

    def _generate_sb5(self) -> Dict[str, Any]:
        p, K, n = self._n_nodes, self._n_contexts, self._n_samples
        base_dag = _random_dag(p, density=2.5 / p, rng=self._rng)
        base_weights = _random_weights(base_dag, rng=self._rng)

        edges = list(zip(*np.where(base_dag != 0)))
        n_edges = len(edges)
        if n_edges == 0:
            base_dag[0, 1] = 1.0
            base_weights[0, 1] = 1.0
            edges = [(0, 1)]
            n_edges = 1

        n_invariant = max(1, n_edges // 4)
        n_parametric = max(1, n_edges // 4)
        n_structural = max(1, n_edges // 4)

        perm = self._rng.permutation(n_edges)
        invariant_edges = set(edges[perm[i]] for i in range(n_invariant))
        param_edges = set(
            edges[perm[i]]
            for i in range(n_invariant, min(n_invariant + n_parametric, n_edges))
        )

        datasets: List[NDArray] = []
        dags: List[NDArray] = []
        weights_list: List[NDArray] = []
        labels: List[str] = []

        for k in range(K):
            dag_k = base_dag.copy()
            w_k = base_weights.copy()

            for e in param_edges:
                w_k[e[0], e[1]] = base_weights[e[0], e[1]] + self._rng.normal(0, 0.5)

            if k > 0:
                edges_k = list(zip(*np.where(dag_k != 0)))
                struct_candidates = [
                    e for e in edges_k
                    if e not in invariant_edges and e not in param_edges
                ]
                n_flip = min(n_structural, len(struct_candidates))
                if struct_candidates and n_flip > 0:
                    flip_idx = self._rng.choice(len(struct_candidates), size=n_flip, replace=False)
                    for fi in flip_idx:
                        ei, ej = struct_candidates[fi]
                        if self._rng.random() < 0.5:
                            dag_k[ei, ej] = 0.0
                            w_k[ei, ej] = 0.0
                        else:
                            w_k[ei, ej] = self._rng.uniform(0.5, 2.0) * self._rng.choice([-1, 1])

                if k >= K // 2:
                    tries = 0
                    while tries < 10:
                        i_new = self._rng.integers(p)
                        j_new = self._rng.integers(p)
                        if i_new != j_new and dag_k[i_new, j_new] == 0:
                            dag_k[i_new, j_new] = 1.0
                            if _has_cycle(dag_k):
                                dag_k[i_new, j_new] = 0.0
                            else:
                                w_k[i_new, j_new] = self._rng.uniform(0.5, 2.0) * self._rng.choice([-1, 1])
                                break
                        tries += 1

            data = _sample_linear_gaussian(w_k, n, rng=self._rng)
            datasets.append(data)
            dags.append(dag_k)
            weights_list.append(w_k)
            labels.append(f"ctx_{k}")
            self._dags[f"ctx_{k}"] = dag_k
            self._weight_matrices[f"ctx_{k}"] = w_k

        self._ground_truth = _classify_edge_plasticity(dags, weights_list)
        return {
            "datasets": datasets,
            "dags": dags,
            "ground_truth_plasticity": self._ground_truth,
            "context_labels": labels,
        }


# ===================================================================
# Cycle check helper
# ===================================================================


def _has_cycle(adj: NDArray) -> bool:
    """Check if adjacency matrix contains a cycle."""
    p = adj.shape[0]
    in_deg = np.sum(adj != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for j in range(p):
            if adj[node, j] != 0:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    return visited != p


# ===================================================================
# Module-level convenience functions
# ===================================================================


def generate_sb1(
    n_nodes: int = 20,
    n_contexts: int = 10,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate SB1 data (fixed structure, varying parameters)."""
    gen = SyntheticGenerator(n_nodes, n_contexts, n_samples, seed)
    return gen.generate("sb1")


def generate_sb2(
    n_nodes: int = 20,
    n_contexts: int = 10,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate SB2 data (structure-varying, shared parents)."""
    gen = SyntheticGenerator(n_nodes, n_contexts, n_samples, seed)
    return gen.generate("sb2")


def generate_sb3(
    n_nodes: int = 20,
    n_contexts: int = 10,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate SB3 data (emergence/disappearance)."""
    gen = SyntheticGenerator(n_nodes, n_contexts, n_samples, seed)
    return gen.generate("sb3")


def generate_sb4(
    n_nodes: int = 20,
    n_contexts: int = 20,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate SB4 data (gradual drift)."""
    gen = SyntheticGenerator(n_nodes, n_contexts, n_samples, seed)
    return gen.generate("sb4")


def generate_sb5(
    n_nodes: int = 20,
    n_contexts: int = 15,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate SB5 data (mixed plasticity)."""
    gen = SyntheticGenerator(n_nodes, n_contexts, n_samples, seed)
    return gen.generate("sb5")
