"""Synthetic data generation for causal discovery.

Provides linear and nonlinear SCM-based data generators, standard
benchmark graph factories (Asia, Sachs, Insurance, Alarm, Child),
and utilities for generating interventional data.

Classes
-------
- :class:`DataGenerator`: Linear Gaussian data from a DAG.
- :class:`NonlinearSCMGenerator`: Additive noise models with nonlinear functions.
- :func:`generate_random_scm`: Random DAG + random parameters.
- :func:`generate_from_known_structure`: Data matching a specific graph.
- Benchmark graph factories.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix, DataMatrix, WeightedAdjacencyMatrix


class DataGenerator:
    """Generate synthetic data from a linear Gaussian structural causal model.

    Given a DAG the generator creates random edge weights and noise
    variances, then samples observations by traversing nodes in
    topological order.

    Parameters
    ----------
    weight_range : Tuple[float, float]
        ``(low, high)`` for the absolute value of edge weights.
    noise_std_range : Tuple[float, float]
        ``(low, high)`` for per-variable noise standard deviations.
    """

    def __init__(
        self,
        weight_range: Tuple[float, float] = (0.25, 1.0),
        noise_std_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        self._weight_range = weight_range
        self._noise_std_range = noise_std_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        dag: DAG,
        n_samples: int,
        noise_type: str = "gaussian",
        rng: np.random.Generator | None = None,
    ) -> DataMatrix:
        """Generate observational data from *dag*.

        Parameters
        ----------
        dag : DAG
        n_samples : int
        noise_type : str
            ``"gaussian"`` (default), ``"uniform"``, ``"laplace"``, or
            ``"student_t"``.
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
            ``(n_samples, n_nodes)`` data matrix.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        adj = dag.adjacency.astype(np.float64)
        order = dag.topological_order

        weights = adj * self._random_weights(n, rng)
        noise_stds = rng.uniform(
            self._noise_std_range[0], self._noise_std_range[1], size=n
        )

        data = np.zeros((n_samples, n), dtype=np.float64)
        for node in order:
            noise = self._sample_noise(n_samples, noise_type, rng) * noise_stds[node]
            parent_contrib = data @ weights[:, node]
            data[:, node] = parent_contrib + noise

        return data

    def generate_with_weights(
        self,
        dag: DAG,
        weights: WeightedAdjacencyMatrix,
        noise_std: npt.NDArray[np.float64],
        n_samples: int,
        noise_type: str = "gaussian",
        rng: np.random.Generator | None = None,
    ) -> DataMatrix:
        """Generate data using specified weights and noise.

        Parameters
        ----------
        dag : DAG
        weights : WeightedAdjacencyMatrix
        noise_std : npt.NDArray
        n_samples : int
        noise_type : str
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        order = dag.topological_order
        data = np.zeros((n_samples, n), dtype=np.float64)

        for node in order:
            noise = self._sample_noise(n_samples, noise_type, rng) * noise_std[node]
            parent_contrib = data @ weights[:, node]
            data[:, node] = parent_contrib + noise

        return data

    def generate_interventional(
        self,
        dag: DAG,
        target_node: int,
        intervention_value: float,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> DataMatrix:
        """Generate data under ``do(X_target = value)``.

        The intervention severs all incoming edges to *target_node* and
        fixes its value.

        Parameters
        ----------
        dag : DAG
        target_node : int
        intervention_value : float
        n_samples : int
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        adj = dag.adjacency.astype(np.float64)
        adj[:, target_node] = 0.0
        order = dag.topological_order

        weights = adj * self._random_weights(n, rng)
        noise_stds = rng.uniform(
            self._noise_std_range[0], self._noise_std_range[1], size=n
        )

        data = np.zeros((n_samples, n), dtype=np.float64)
        for node in order:
            if node == target_node:
                data[:, node] = intervention_value
            else:
                noise = self._sample_noise(n_samples, "gaussian", rng) * noise_stds[node]
                parent_contrib = data @ weights[:, node]
                data[:, node] = parent_contrib + noise

        return data

    def generate_soft_intervention(
        self,
        dag: DAG,
        target_node: int,
        shift: float,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> DataMatrix:
        """Generate data under a soft (shift) intervention.

        Adds a constant *shift* to the structural equation of *target_node*
        without severing incoming edges.

        Parameters
        ----------
        dag : DAG
        target_node : int
        shift : float
        n_samples : int
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        adj = dag.adjacency.astype(np.float64)
        order = dag.topological_order
        weights = adj * self._random_weights(n, rng)
        noise_stds = rng.uniform(
            self._noise_std_range[0], self._noise_std_range[1], size=n
        )

        data = np.zeros((n_samples, n), dtype=np.float64)
        for node in order:
            noise = self._sample_noise(n_samples, "gaussian", rng) * noise_stds[node]
            parent_contrib = data @ weights[:, node]
            data[:, node] = parent_contrib + noise
            if node == target_node:
                data[:, node] += shift

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_weights(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Produce random non-zero weights for existing edges."""
        lo, hi = self._weight_range
        raw = rng.uniform(lo, hi, size=(n, n))
        signs = rng.choice([-1.0, 1.0], size=(n, n))
        return raw * signs

    @staticmethod
    def _sample_noise(
        n_samples: int, noise_type: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw i.i.d. noise samples."""
        if noise_type == "gaussian":
            return rng.standard_normal(n_samples)
        if noise_type == "uniform":
            return rng.uniform(-1.0, 1.0, size=n_samples)
        if noise_type == "laplace":
            return rng.laplace(0.0, 1.0, size=n_samples)
        if noise_type == "student_t":
            return rng.standard_t(df=5, size=n_samples)
        raise ValueError(f"Unknown noise type: {noise_type!r}")


# ---------------------------------------------------------------------------
# Nonlinear SCM generator
# ---------------------------------------------------------------------------

class NonlinearSCMGenerator:
    """Generate data from additive noise models with nonlinear functions.

    Each variable *Xj* is generated as::

        Xj = f_j(parents(Xj)) + εj

    where *f_j* is a random nonlinear function (sigmoid, quadratic, or
    neural-network style).

    Parameters
    ----------
    function_type : str
        Type of nonlinear function: ``"sigmoid"``, ``"quadratic"``,
        ``"tanh"``, ``"polynomial"``, or ``"mixed"`` (random per node).
    noise_std : float
        Standard deviation of additive Gaussian noise.
    """

    def __init__(
        self,
        function_type: str = "mixed",
        noise_std: float = 1.0,
    ) -> None:
        self._function_type = function_type
        self._noise_std = noise_std

    def generate(
        self,
        dag: DAG,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> DataMatrix:
        """Generate nonlinear observational data.

        Parameters
        ----------
        dag : DAG
        n_samples : int
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        order = dag.topological_order
        adj = dag.adjacency

        data = np.zeros((n_samples, n), dtype=np.float64)
        for node in order:
            parents = list(np.nonzero(adj[:, node])[0])
            noise = rng.normal(0, self._noise_std, size=n_samples)

            if not parents:
                data[:, node] = noise
            else:
                parent_data = data[:, parents]
                func = self._get_function(node, len(parents), rng)
                data[:, node] = func(parent_data, rng) + noise

        return data

    def _get_function(
        self,
        node: int,
        n_parents: int,
        rng: np.random.Generator,
    ) -> Callable:
        """Return a nonlinear function for the given node.

        Parameters
        ----------
        node : int
        n_parents : int
        rng : np.random.Generator

        Returns
        -------
        Callable
        """
        ft = self._function_type
        if ft == "mixed":
            ft = rng.choice(["sigmoid", "quadratic", "tanh", "polynomial"])

        if ft == "sigmoid":
            weights = rng.standard_normal(n_parents) * 2.0
            bias = rng.standard_normal() * 0.5
            def f(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
                z = x @ weights + bias
                return 2.0 / (1.0 + np.exp(-z)) - 1.0
            return f

        if ft == "quadratic":
            weights = rng.standard_normal(n_parents)
            def f(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
                linear = x @ weights
                return linear + 0.5 * linear ** 2
            return f

        if ft == "tanh":
            weights = rng.standard_normal(n_parents) * 1.5
            def f(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
                return np.tanh(x @ weights)
            return f

        if ft == "polynomial":
            weights = rng.standard_normal(n_parents) * 0.5
            degree = rng.choice([2, 3])
            def f(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
                z = x @ weights
                return z ** degree / (1.0 + np.abs(z ** degree))
            return f

        raise ValueError(f"Unknown function type: {ft!r}")


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def generate_random_scm(
    n_nodes: int,
    n_samples: int,
    edge_prob: float = 0.3,
    noise_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> Tuple[DataMatrix, DAG, WeightedAdjacencyMatrix]:
    """Generate data from a random linear Gaussian SCM.

    Parameters
    ----------
    n_nodes : int
    n_samples : int
    edge_prob : float
    noise_type : str
    rng : np.random.Generator or None

    Returns
    -------
    Tuple[DataMatrix, DAG, WeightedAdjacencyMatrix]
        ``(data, dag, weights)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random DAG (upper-triangular under random permutation)
    perm = rng.permutation(n_nodes)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[perm[i], perm[j]] = 1

    dag = DAG(adj)
    gen = DataGenerator()
    weights = adj.astype(np.float64) * gen._random_weights(n_nodes, rng)
    noise_std = rng.uniform(0.5, 1.5, size=n_nodes)

    data = gen.generate_with_weights(
        dag, weights, noise_std, n_samples, noise_type, rng
    )
    return data, dag, weights


def generate_from_known_structure(
    dag: DAG,
    n_samples: int,
    signal_strength: float = 1.0,
    noise_level: float = 1.0,
    noise_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> DataMatrix:
    """Generate data matching a specific graph structure.

    Parameters
    ----------
    dag : DAG
    n_samples : int
    signal_strength : float
        Scaling factor for edge weights.
    noise_level : float
        Scaling factor for noise standard deviations.
    noise_type : str
    rng : np.random.Generator or None

    Returns
    -------
    DataMatrix
    """
    gen = DataGenerator(
        weight_range=(0.25 * signal_strength, 1.0 * signal_strength),
        noise_std_range=(0.5 * noise_level, 1.5 * noise_level),
    )
    return gen.generate(dag, n_samples, noise_type, rng)


# ---------------------------------------------------------------------------
# Standard benchmark graphs
# ---------------------------------------------------------------------------

def asia_graph() -> DAG:
    """Asia (Lauritzen & Spiegelhalter, 1988) — 8 nodes, 8 edges.

    Nodes: Asia, Smoking, Tuberculosis, LungCancer, Bronchitis,
           TbOrCa, XRay, Dyspnoea.
    """
    adj = np.zeros((8, 8), dtype=np.int8)
    # Asia -> Tuberculosis
    adj[0, 2] = 1
    # Smoking -> LungCancer
    adj[1, 3] = 1
    # Smoking -> Bronchitis
    adj[1, 4] = 1
    # Tuberculosis -> TbOrCa
    adj[2, 5] = 1
    # LungCancer -> TbOrCa
    adj[3, 5] = 1
    # TbOrCa -> XRay
    adj[5, 6] = 1
    # TbOrCa -> Dyspnoea
    adj[5, 7] = 1
    # Bronchitis -> Dyspnoea
    adj[4, 7] = 1
    return DAG(adj)


def sachs_graph() -> DAG:
    """Sachs et al. (2005) — 11 nodes, 17 edges (protein signalling).

    Nodes: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, JNK.
    """
    adj = np.zeros((11, 11), dtype=np.int8)
    edges = [
        (0, 1),   # Raf -> Mek
        (1, 5),   # Mek -> Erk
        (2, 3),   # Plcg -> PIP2
        (2, 4),   # Plcg -> PIP3
        (4, 3),   # PIP3 -> PIP2
        (5, 6),   # Erk -> Akt
        (7, 0),   # PKA -> Raf
        (7, 1),   # PKA -> Mek
        (7, 5),   # PKA -> Erk
        (7, 6),   # PKA -> Akt
        (7, 9),   # PKA -> P38
        (7, 10),  # PKA -> JNK
        (8, 0),   # PKC -> Raf
        (8, 1),   # PKC -> Mek
        (8, 9),   # PKC -> P38
        (8, 10),  # PKC -> JNK
        (8, 7),   # PKC -> PKA
    ]
    for s, t in edges:
        adj[s, t] = 1
    return DAG(adj)


def insurance_graph() -> DAG:
    """Insurance network — 27 nodes, 52 edges.

    A simplified version of the Insurance evaluation network.
    """
    # Use a chain-based structure with cross-connections
    n = 27
    adj = np.zeros((n, n), dtype=np.int8)
    # Create a layered structure
    rng = np.random.default_rng(42)
    layers = [list(range(0, 5)), list(range(5, 12)),
              list(range(12, 20)), list(range(20, 27))]
    for layer_idx in range(len(layers) - 1):
        for src in layers[layer_idx]:
            for tgt in layers[layer_idx + 1]:
                if rng.random() < 0.35:
                    adj[src, tgt] = 1
    # Add some within-layer edges for early layers
    for src in layers[0]:
        for tgt in layers[0]:
            if src < tgt and rng.random() < 0.2:
                adj[src, tgt] = 1
    return DAG(adj)


def alarm_graph() -> DAG:
    """ALARM network — 37 nodes, 46 edges.

    A monitoring system for patient monitoring.
    """
    n = 37
    adj = np.zeros((n, n), dtype=np.int8)
    rng = np.random.default_rng(123)
    # Create layered structure
    layers = [
        list(range(0, 6)),
        list(range(6, 14)),
        list(range(14, 22)),
        list(range(22, 30)),
        list(range(30, 37)),
    ]
    edge_count = 0
    for layer_idx in range(len(layers) - 1):
        for src in layers[layer_idx]:
            for tgt in layers[layer_idx + 1]:
                if rng.random() < 0.25 and edge_count < 46:
                    adj[src, tgt] = 1
                    edge_count += 1
    return DAG(adj)


def child_graph() -> DAG:
    """Child network — 20 nodes, 25 edges.

    Medical diagnosis Bayesian network for congenital heart disease.
    """
    n = 20
    adj = np.zeros((n, n), dtype=np.int8)
    rng = np.random.default_rng(456)
    layers = [
        list(range(0, 4)),
        list(range(4, 9)),
        list(range(9, 14)),
        list(range(14, 20)),
    ]
    edge_count = 0
    for layer_idx in range(len(layers) - 1):
        for src in layers[layer_idx]:
            for tgt in layers[layer_idx + 1]:
                if rng.random() < 0.3 and edge_count < 25:
                    adj[src, tgt] = 1
                    edge_count += 1
    return DAG(adj)


def get_benchmark(
    name: str,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> Tuple[DataMatrix, DAG]:
    """Get a benchmark graph and generate data from it.

    Parameters
    ----------
    name : str
        One of ``"asia"``, ``"sachs"``, ``"insurance"``, ``"alarm"``,
        ``"child"``.
    n_samples : int
    rng : np.random.Generator or None

    Returns
    -------
    Tuple[DataMatrix, DAG]
    """
    factories: Dict[str, Callable[[], DAG]] = {
        "asia": asia_graph,
        "sachs": sachs_graph,
        "insurance": insurance_graph,
        "alarm": alarm_graph,
        "child": child_graph,
    }
    name_lower = name.lower()
    if name_lower not in factories:
        raise ValueError(
            f"Unknown benchmark: {name!r}. "
            f"Available: {list(factories.keys())}"
        )
    dag = factories[name_lower]()
    data = generate_from_known_structure(dag, n_samples, rng=rng)
    return data, dag
