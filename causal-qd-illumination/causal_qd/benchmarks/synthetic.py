"""Synthetic benchmark generation for scalability and robustness testing.

Generates synthetic causal graphs with controlled properties for
systematic evaluation of causal discovery algorithms.

Classes
-------
* :class:`RandomDAGBenchmark` – random DAGs with controlled properties
* :class:`ScalabilityBenchmark` – series of increasing sizes
* :class:`FaithfulnessViolationBenchmark` – controlled faithfulness violations
* :class:`SparsityBenchmark` – varying edge densities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix

__all__ = [
    "RandomDAGBenchmark",
    "ScalabilityBenchmark",
    "FaithfulnessViolationBenchmark",
    "SparsityBenchmark",
    "SyntheticBenchmark",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class SyntheticBenchmarkSpec:
    """Specification for a synthetic benchmark instance."""

    name: str
    n_nodes: int
    true_dag: AdjacencyMatrix
    data: DataMatrix
    n_samples: int
    n_edges: int
    density: float
    params: Dict[str, Any] = field(default_factory=dict)


class SyntheticBenchmark:
    """Base class for synthetic benchmarks.

    Provides data generation from linear-Gaussian models and evaluation
    metrics.
    """

    @staticmethod
    def generate_random_dag(
        n_nodes: int,
        edge_prob: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> AdjacencyMatrix:
        """Generate a random DAG by sampling edges in a random topological order.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        edge_prob : float
            Probability of including each possible edge.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            ``(n_nodes, n_nodes)`` adjacency matrix.
        """
        if rng is None:
            rng = np.random.default_rng()

        perm = rng.permutation(n_nodes)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    adj[perm[i], perm[j]] = 1

        return adj

    @staticmethod
    def generate_data_from_dag(
        adj: AdjacencyMatrix,
        n_samples: int = 1000,
        noise_std: float = 1.0,
        weight_range: Tuple[float, float] = (0.5, 2.0),
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[DataMatrix, np.ndarray]:
        """Generate data from a linear-Gaussian model.

        Parameters
        ----------
        adj : np.ndarray
            DAG adjacency matrix.
        n_samples : int
            Number of observations.
        noise_std : float
            Noise standard deviation.
        weight_range : tuple of float
            Range for absolute edge weight values.
        rng : np.random.Generator, optional
            RNG.

        Returns
        -------
        tuple of (DataMatrix, weights)
            Data matrix and edge weight matrix.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = adj.shape[0]

        # Generate weights
        weights = np.zeros((n, n), dtype=np.float64)
        rows, cols = np.nonzero(adj)
        for i, j in zip(rows, cols):
            sign = rng.choice([-1, 1])
            weights[i, j] = sign * rng.uniform(weight_range[0], weight_range[1])

        # Topological sort
        in_deg = adj.sum(axis=0).astype(np.int64).copy()
        queue = list(np.nonzero(in_deg == 0)[0])
        order: List[int] = []

        while queue:
            u = queue.pop(0)
            order.append(int(u))
            for v in np.nonzero(adj[u])[0]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(int(v))

        # Ancestral sampling
        data = np.zeros((n_samples, n), dtype=np.float64)
        for j in order:
            parents = np.nonzero(adj[:, j])[0]
            noise = rng.normal(0, noise_std, n_samples)
            if len(parents) > 0:
                data[:, j] = data[:, parents] @ weights[parents, j] + noise
            else:
                data[:, j] = noise

        return data, weights

    @staticmethod
    def shd(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> int:
        """Structural Hamming Distance."""
        return int(np.sum(adj1.astype(np.int8) != adj2.astype(np.int8)))


# ---------------------------------------------------------------------------
# RandomDAGBenchmark
# ---------------------------------------------------------------------------


class RandomDAGBenchmark(SyntheticBenchmark):
    """Random DAGs with controlled properties.

    Generates random Erdos-Renyi DAGs with specified number of nodes,
    edge probability, and maximum in-degree.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Edge probability.
    max_in_degree : int, optional
        Maximum in-degree constraint. -1 for no limit.
    n_samples : int
        Number of data samples.
    n_instances : int
        Number of random instances to generate.
    seed : int
        Random seed.

    Examples
    --------
    >>> bench = RandomDAGBenchmark(n_nodes=10, n_instances=5)
    >>> instances = bench.generate()
    >>> len(instances)
    5
    """

    def __init__(
        self,
        n_nodes: int = 10,
        edge_prob: float = 0.3,
        max_in_degree: int = -1,
        n_samples: int = 1000,
        n_instances: int = 10,
        seed: int = 42,
    ) -> None:
        self._n_nodes = n_nodes
        self._edge_prob = edge_prob
        self._max_in_degree = max_in_degree
        self._n_samples = n_samples
        self._n_instances = n_instances
        self._seed = seed

    def generate(self) -> List[SyntheticBenchmarkSpec]:
        """Generate benchmark instances.

        Returns
        -------
        list of SyntheticBenchmarkSpec
            Benchmark instances with DAGs and data.
        """
        rng = np.random.default_rng(self._seed)
        instances: List[SyntheticBenchmarkSpec] = []

        for i in range(self._n_instances):
            adj = self.generate_random_dag(
                self._n_nodes, self._edge_prob, rng
            )

            # Enforce max in-degree
            if self._max_in_degree > 0:
                for j in range(self._n_nodes):
                    parents = np.nonzero(adj[:, j])[0]
                    if len(parents) > self._max_in_degree:
                        keep = rng.choice(
                            parents, size=self._max_in_degree, replace=False
                        )
                        adj[:, j] = 0
                        adj[keep, j] = 1

            data, _ = self.generate_data_from_dag(
                adj, self._n_samples, rng=rng
            )

            n_edges = int(np.sum(adj))
            max_edges = self._n_nodes * (self._n_nodes - 1)
            density = n_edges / max_edges if max_edges > 0 else 0.0

            instances.append(
                SyntheticBenchmarkSpec(
                    name=f"random_{self._n_nodes}_{i}",
                    n_nodes=self._n_nodes,
                    true_dag=adj,
                    data=data,
                    n_samples=self._n_samples,
                    n_edges=n_edges,
                    density=density,
                    params={
                        "edge_prob": self._edge_prob,
                        "max_in_degree": self._max_in_degree,
                        "instance": i,
                    },
                )
            )

        return instances


# ---------------------------------------------------------------------------
# ScalabilityBenchmark
# ---------------------------------------------------------------------------


class ScalabilityBenchmark(SyntheticBenchmark):
    """Scalability benchmark: series of increasing problem sizes.

    Generates benchmarks with increasing numbers of nodes to test
    how algorithms scale.

    Parameters
    ----------
    sizes : list of int
        Node counts to generate.
    edge_prob : float
        Edge probability (constant across sizes).
    n_samples : int
        Number of data samples.
    seed : int
        Random seed.

    Examples
    --------
    >>> bench = ScalabilityBenchmark(sizes=[5, 10, 20, 50, 100])
    >>> instances = bench.generate()
    """

    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        edge_prob: float = 0.2,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        self._sizes = sizes or [5, 10, 20, 50, 100]
        self._edge_prob = edge_prob
        self._n_samples = n_samples
        self._seed = seed

    def generate(self) -> List[SyntheticBenchmarkSpec]:
        """Generate scalability benchmark instances."""
        rng = np.random.default_rng(self._seed)
        instances: List[SyntheticBenchmarkSpec] = []

        for n in self._sizes:
            adj = self.generate_random_dag(n, self._edge_prob, rng)
            data, _ = self.generate_data_from_dag(
                adj, self._n_samples, rng=rng
            )

            n_edges = int(np.sum(adj))
            max_edges = n * (n - 1)
            density = n_edges / max_edges if max_edges > 0 else 0.0

            instances.append(
                SyntheticBenchmarkSpec(
                    name=f"scale_{n}",
                    n_nodes=n,
                    true_dag=adj,
                    data=data,
                    n_samples=self._n_samples,
                    n_edges=n_edges,
                    density=density,
                    params={"size": n, "edge_prob": self._edge_prob},
                )
            )

        return instances


# ---------------------------------------------------------------------------
# FaithfulnessViolationBenchmark
# ---------------------------------------------------------------------------


class FaithfulnessViolationBenchmark(SyntheticBenchmark):
    """Benchmark with controlled faithfulness violations.

    Generates DAGs where some conditional independencies implied by
    the structure are violated due to specific parameter choices
    (near-cancellation paths).

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_violations : int
        Number of faithfulness violations to introduce.
    n_samples : int
        Number of data samples.
    n_instances : int
        Number of instances.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_nodes: int = 10,
        n_violations: int = 2,
        n_samples: int = 1000,
        n_instances: int = 5,
        seed: int = 42,
    ) -> None:
        self._n_nodes = n_nodes
        self._n_violations = n_violations
        self._n_samples = n_samples
        self._n_instances = n_instances
        self._seed = seed

    def generate(self) -> List[SyntheticBenchmarkSpec]:
        """Generate benchmarks with faithfulness violations."""
        rng = np.random.default_rng(self._seed)
        instances: List[SyntheticBenchmarkSpec] = []

        for inst in range(self._n_instances):
            # Generate DAG with enough structure for cancellation paths
            adj = self.generate_random_dag(
                self._n_nodes, 0.3, rng
            )

            # Generate weights
            n = self._n_nodes
            weights = np.zeros((n, n), dtype=np.float64)
            rows, cols = np.nonzero(adj)
            for i, j in zip(rows, cols):
                sign = rng.choice([-1, 1])
                weights[i, j] = sign * rng.uniform(0.5, 2.0)

            # Introduce faithfulness violations by creating cancellation paths
            # Find nodes with multiple parents and adjust weights so that
            # the total effect approximately cancels
            violations_created = 0
            for j in range(n):
                if violations_created >= self._n_violations:
                    break
                parents = np.nonzero(adj[:, j])[0]
                if len(parents) >= 2:
                    # Make first two parent effects cancel
                    w0 = weights[parents[0], j]
                    weights[parents[1], j] = -w0 * (1 + rng.normal(0, 0.01))
                    violations_created += 1

            # Generate data with modified weights
            in_deg = adj.sum(axis=0).astype(np.int64).copy()
            queue = list(np.nonzero(in_deg == 0)[0])
            order: List[int] = []
            while queue:
                u = queue.pop(0)
                order.append(int(u))
                for v in np.nonzero(adj[u])[0]:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        queue.append(int(v))

            data = np.zeros((self._n_samples, n), dtype=np.float64)
            for j in order:
                parents_j = np.nonzero(adj[:, j])[0]
                noise = rng.normal(0, 1.0, self._n_samples)
                if len(parents_j) > 0:
                    data[:, j] = data[:, parents_j] @ weights[parents_j, j] + noise
                else:
                    data[:, j] = noise

            n_edges = int(np.sum(adj))
            max_edges = n * (n - 1)
            density = n_edges / max_edges if max_edges > 0 else 0.0

            instances.append(
                SyntheticBenchmarkSpec(
                    name=f"faithfulness_{n}_{inst}",
                    n_nodes=n,
                    true_dag=adj,
                    data=data,
                    n_samples=self._n_samples,
                    n_edges=n_edges,
                    density=density,
                    params={
                        "n_violations": violations_created,
                        "instance": inst,
                    },
                )
            )

        return instances


# ---------------------------------------------------------------------------
# SparsityBenchmark
# ---------------------------------------------------------------------------


class SparsityBenchmark(SyntheticBenchmark):
    """Benchmark with varying edge densities.

    Generates DAGs with the same number of nodes but different edge
    probabilities to test robustness across sparsity levels.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    densities : list of float
        Edge probabilities to test.
    n_samples : int
        Number of data samples.
    n_instances_per_density : int
        Instances per density level.
    seed : int
        Random seed.

    Examples
    --------
    >>> bench = SparsityBenchmark(
    ...     n_nodes=15,
    ...     densities=[0.1, 0.2, 0.3, 0.5, 0.7]
    ... )
    >>> instances = bench.generate()
    """

    def __init__(
        self,
        n_nodes: int = 15,
        densities: Optional[List[float]] = None,
        n_samples: int = 1000,
        n_instances_per_density: int = 3,
        seed: int = 42,
    ) -> None:
        self._n_nodes = n_nodes
        self._densities = densities or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        self._n_samples = n_samples
        self._n_instances = n_instances_per_density
        self._seed = seed

    def generate(self) -> List[SyntheticBenchmarkSpec]:
        """Generate sparsity benchmark instances."""
        rng = np.random.default_rng(self._seed)
        instances: List[SyntheticBenchmarkSpec] = []

        for edge_prob in self._densities:
            for inst in range(self._n_instances):
                adj = self.generate_random_dag(
                    self._n_nodes, edge_prob, rng
                )
                data, _ = self.generate_data_from_dag(
                    adj, self._n_samples, rng=rng
                )

                n_edges = int(np.sum(adj))
                max_edges = self._n_nodes * (self._n_nodes - 1)
                actual_density = (
                    n_edges / max_edges if max_edges > 0 else 0.0
                )

                instances.append(
                    SyntheticBenchmarkSpec(
                        name=f"sparsity_{edge_prob:.2f}_{inst}",
                        n_nodes=self._n_nodes,
                        true_dag=adj,
                        data=data,
                        n_samples=self._n_samples,
                        n_edges=n_edges,
                        density=actual_density,
                        params={
                            "target_edge_prob": edge_prob,
                            "actual_density": actual_density,
                            "instance": inst,
                        },
                    )
                )

        return instances
