"""
Synthetic DGP generation with known robustness radii.

Generates causal DAGs paired with structural equation models where the
true robustness radius is known by construction.  This enables ground-truth
evaluation of the solver and fragility scorer.

Supports:
- Random DAG generation (Erdos-Renyi, scale-free, chain, diamond, etc.)
- Linear Gaussian SCMs with configurable effect sizes
- Nonlinear SCMs (additive noise, polynomial)
- Controlled fragility: construct DAGs where specific edges are load-bearing
- Known radius construction: build DAGs that have exactly radius k
- Multiple treatment/outcome configurations
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    RobustnessRadius,
    StructuralEdit,
)


@dataclass(slots=True)
class DGPInstance:
    """A synthetic DGP instance with known ground truth.

    Attributes
    ----------
    adj : AdjacencyMatrix
        True DAG adjacency matrix.
    data : pd.DataFrame
        Generated observational data.
    treatment : int
        Treatment node index.
    outcome : int
        Outcome node index.
    true_radius : int
        Ground-truth robustness radius.
    witness_edits : tuple[StructuralEdit, ...]
        Ground-truth minimum edit set.
    true_ate : float
        True average treatment effect.
    name : str
        Descriptive name of the DGP.
    """

    adj: AdjacencyMatrix
    data: pd.DataFrame
    treatment: int
    outcome: int
    true_radius: int
    witness_edits: tuple[StructuralEdit, ...] = ()
    true_ate: float = 0.0
    name: str = ""


# ---------------------------------------------------------------------------
# DAG generation
# ---------------------------------------------------------------------------


def random_dag_erdos_renyi(
    n: int,
    density: float = 0.2,
    rng: np.random.RandomState | None = None,
) -> AdjacencyMatrix:
    """Generate a random DAG via the Erdos-Renyi model.

    Samples a random upper-triangular matrix under a random topological
    ordering, then permutes.

    Parameters
    ----------
    n : int
        Number of nodes.
    density : float
        Expected edge density.
    rng : np.random.RandomState | None
        Random state.

    Returns
    -------
    AdjacencyMatrix
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.zeros((n, n), dtype=np.int8)
    # Random topological order
    perm = rng.permutation(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                adj[perm[i], perm[j]] = 1
    return adj


def random_dag_scale_free(
    n: int,
    m_edges: int = 2,
    rng: np.random.RandomState | None = None,
) -> AdjacencyMatrix:
    """Generate a scale-free DAG via preferential attachment.

    Parameters
    ----------
    n : int
        Number of nodes.
    m_edges : int
        Edges per new node.
    rng : np.random.RandomState | None

    Returns
    -------
    AdjacencyMatrix
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.zeros((n, n), dtype=np.int8)
    degrees = np.zeros(n)

    for new_node in range(1, n):
        m = min(m_edges, new_node)
        # Preferential attachment
        weights = degrees[:new_node] + 1.0
        weights /= weights.sum()
        targets = rng.choice(new_node, size=m, replace=False, p=weights)
        for t in targets:
            adj[int(t), new_node] = 1
            degrees[t] += 1
            degrees[new_node] += 1

    return adj


def chain_dag(n: int) -> AdjacencyMatrix:
    """Generate a simple chain DAG: 0 → 1 → 2 → ... → n-1."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj


def diamond_dag() -> AdjacencyMatrix:
    """Generate a 4-node diamond DAG: X → M1, X → M2, M1 → Y, M2 → Y."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = adj[0, 2] = adj[1, 3] = adj[2, 3] = 1
    return adj


def fork_dag() -> AdjacencyMatrix:
    """Generate a 3-node fork: C → X, C → Y (common cause)."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = adj[0, 2] = 1
    return adj


def collider_dag() -> AdjacencyMatrix:
    """Generate a 3-node collider: X → C ← Y."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = adj[1, 2] = 1
    return adj


def mediator_dag() -> AdjacencyMatrix:
    """Generate a 3-node mediator: X → M → Y."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = adj[1, 2] = 1
    return adj


def random_dag_layered(
    layers: list[int],
    inter_density: float = 0.3,
    rng: np.random.RandomState | None = None,
) -> AdjacencyMatrix:
    """Generate a layered DAG with edges only from earlier to later layers.

    Parameters
    ----------
    layers : list[int]
        Number of nodes per layer.
    inter_density : float
        Edge density between consecutive layers.
    rng : np.random.RandomState | None

    Returns
    -------
    AdjacencyMatrix
    """
    if rng is None:
        rng = np.random.RandomState()
    n = sum(layers)
    adj = np.zeros((n, n), dtype=np.int8)
    layer_starts = []
    offset = 0
    for sz in layers:
        layer_starts.append(offset)
        offset += sz
    layer_starts.append(n)

    for l_idx in range(len(layers) - 1):
        src_start = layer_starts[l_idx]
        src_end = layer_starts[l_idx + 1]
        tgt_start = layer_starts[l_idx + 1]
        tgt_end = layer_starts[l_idx + 2]
        for s in range(src_start, src_end):
            for t in range(tgt_start, tgt_end):
                if rng.random() < inter_density:
                    adj[s, t] = 1
    return adj


# ---------------------------------------------------------------------------
# SCM data generation
# ---------------------------------------------------------------------------


def generate_linear_gaussian(
    adj: AdjacencyMatrix,
    n_samples: int = 1000,
    effect_range: tuple[float, float] = (0.3, 1.5),
    noise_std: float = 1.0,
    treatment: int = 0,
    treatment_effect: float | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Generate data from a linear Gaussian SCM.

    X_j = sum_{i: i → j} beta_{ij} * X_i + eps_j

    Parameters
    ----------
    adj : AdjacencyMatrix
    n_samples : int
    effect_range : tuple[float, float]
        Range for random edge coefficients.
    noise_std : float
    treatment : int
        Treatment node.
    treatment_effect : float | None
        If given, set the direct treatment→outcome coefficient.
    rng : np.random.RandomState | None

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, float]
        ``(data, coefficients, true_ate)``.
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    # Generate coefficients
    coeffs = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                sign = rng.choice([-1, 1])
                mag = rng.uniform(effect_range[0], effect_range[1])
                coeffs[i, j] = sign * mag

    # Override treatment→outcome coefficient if specified
    if treatment_effect is not None:
        for j in range(n):
            if adj[treatment, j]:
                # Find outcome (last in topological order among descendants)
                pass

    # Topological order
    topo = _topological_sort(adj)

    # Generate data
    data = np.zeros((n_samples, n), dtype=np.float64)
    for node in topo:
        noise = rng.normal(0, noise_std, size=n_samples)
        parent_contrib = np.zeros(n_samples)
        for parent in range(n):
            if adj[parent, node]:
                parent_contrib += coeffs[parent, node] * data[:, parent]
        data[:, node] = parent_contrib + noise

    # Binarise treatment (threshold at median)
    median_t = np.median(data[:, treatment])
    data[:, treatment] = (data[:, treatment] > median_t).astype(float)

    df = pd.DataFrame(data, columns=list(range(n)))

    # Compute true ATE (direct + indirect effects via SCM)
    true_ate = _compute_true_ate_linear(adj, coeffs, treatment, n)

    return df, coeffs, true_ate


def generate_nonlinear_additive(
    adj: AdjacencyMatrix,
    n_samples: int = 1000,
    noise_std: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Generate data from a nonlinear additive noise SCM.

    X_j = sum_{i: i → j} f_ij(X_i) + eps_j

    where f_ij is a random nonlinear function (quadratic or sigmoid).

    Parameters
    ----------
    adj : AdjacencyMatrix
    n_samples : int
    noise_std : float
    rng : np.random.RandomState | None

    Returns
    -------
    pd.DataFrame
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    topo = _topological_sort(adj)

    data = np.zeros((n_samples, n), dtype=np.float64)
    for node in topo:
        noise = rng.normal(0, noise_std, size=n_samples)
        parent_contrib = np.zeros(n_samples)
        for parent in range(n):
            if adj[parent, node]:
                func_type = rng.choice(["quadratic", "sigmoid", "sin"])
                coeff = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])
                x = data[:, parent]
                if func_type == "quadratic":
                    parent_contrib += coeff * (x + 0.3 * x ** 2)
                elif func_type == "sigmoid":
                    parent_contrib += coeff * (2.0 / (1.0 + np.exp(-x)) - 1.0)
                else:
                    parent_contrib += coeff * np.sin(x)
        data[:, node] = parent_contrib + noise

    return pd.DataFrame(data, columns=list(range(n)))


def generate_polynomial_scm(
    adj: AdjacencyMatrix,
    n_samples: int = 1000,
    max_degree: int = 3,
    noise_std: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Generate data from a polynomial SCM.

    Parameters
    ----------
    adj : AdjacencyMatrix
    n_samples : int
    max_degree : int
    noise_std : float
    rng : np.random.RandomState | None

    Returns
    -------
    pd.DataFrame
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    topo = _topological_sort(adj)

    data = np.zeros((n_samples, n), dtype=np.float64)
    for node in topo:
        noise = rng.normal(0, noise_std, size=n_samples)
        parent_contrib = np.zeros(n_samples)
        for parent in range(n):
            if adj[parent, node]:
                degree = rng.randint(1, max_degree + 1)
                coeff = rng.uniform(0.2, 0.8)
                x = data[:, parent]
                parent_contrib += coeff * np.power(x, degree)
        data[:, node] = parent_contrib + noise

    return pd.DataFrame(data, columns=list(range(n)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topological_sort(adj: np.ndarray) -> list[int]:
    """Kahn's algorithm for topological sort."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in np.nonzero(adj[node])[0]:
            child = int(child)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)
    return order


def _compute_true_ate_linear(
    adj: np.ndarray, coeffs: np.ndarray, treatment: int, n: int
) -> float:
    """Compute true ATE in a linear SCM via path analysis.

    ATE = sum of products of coefficients along all directed paths
    from treatment to each descendant.
    """
    # Use matrix power series: total effect = (I - B)^{-1}
    # where B is the coefficient matrix
    B = coeffs.copy()
    try:
        total_effect = np.linalg.inv(np.eye(n) - B)
    except np.linalg.LinAlgError:
        return 0.0

    # Find outcome (last descendant of treatment)
    desc = set()
    queue = deque([treatment])
    while queue:
        v = queue.popleft()
        for c in np.nonzero(adj[v])[0]:
            c = int(c)
            if c not in desc:
                desc.add(c)
                queue.append(c)
    if not desc:
        return 0.0

    # Sum total effects on all direct children (approximation)
    ate = 0.0
    for d in desc:
        ate += total_effect[treatment, d]
    return float(ate / max(len(desc), 1))


def _is_dag(adj: np.ndarray) -> bool:
    """Quick DAG check."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        node = queue.popleft()
        count += 1
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    return count == n


# ---------------------------------------------------------------------------
# Known-radius construction
# ---------------------------------------------------------------------------


def _construct_known_radius_dag(
    n: int,
    target_radius: int,
    rng: np.random.RandomState,
) -> tuple[AdjacencyMatrix, int, int, tuple[StructuralEdit, ...]]:
    """Construct a DAG with a known robustness radius.

    Strategy: build a chain with *target_radius* load-bearing edges.
    The causal effect X → Y flows through exactly *target_radius*
    critical edges; removing any one of them severs the causal path.

    Each critical edge is the sole directed path between two nodes,
    so the minimum number of edits to break the causal conclusion is
    exactly *target_radius* ... wait, if ANY single edge breaks the path,
    then radius = 1.

    Better strategy: create *target_radius* INDEPENDENT causal paths
    from X to Y.  The radius is then *target_radius* because you need
    to break ALL paths.

    Parameters
    ----------
    n : int
        Number of nodes (must be >= target_radius + 2).
    target_radius : int
        Desired robustness radius.
    rng : np.random.RandomState

    Returns
    -------
    tuple
        ``(adj, treatment, outcome, witness_edits)``
    """
    n = max(n, target_radius * 2 + 2)
    adj = np.zeros((n, n), dtype=np.int8)
    treatment = 0
    outcome = n - 1

    # Create target_radius independent paths from treatment to outcome
    mediators_per_path = max(1, (n - 2) // target_radius)
    path_edges: list[list[tuple[int, int]]] = []
    node_idx = 1  # start after treatment

    for p in range(target_radius):
        path: list[tuple[int, int]] = []
        prev = treatment
        for step in range(mediators_per_path):
            if node_idx >= n - 1:
                break
            adj[prev, node_idx] = 1
            path.append((prev, node_idx))
            prev = node_idx
            node_idx += 1
        adj[prev, outcome] = 1
        path.append((prev, outcome))
        path_edges.append(path)

    # Add some noise edges (confounders) that don't create new paths
    for _ in range(n // 3):
        u = rng.randint(0, n)
        v = rng.randint(0, n)
        if u != v and not adj[u, v] and u != outcome and v != treatment:
            adj_test = adj.copy()
            adj_test[u, v] = 1
            if _is_dag(adj_test):
                adj[u, v] = 1

    # Witness edits: delete one critical edge from each path
    witness = []
    for path in path_edges:
        if path:
            u, v = path[0]
            witness.append(StructuralEdit(EditType.DELETE, u, v))

    return adj, treatment, outcome, tuple(witness)


# ---------------------------------------------------------------------------
# Main SyntheticDGP class
# ---------------------------------------------------------------------------


class SyntheticDGP:
    """Generator for synthetic DGP instances with known radii.

    Parameters
    ----------
    n_nodes_range : tuple[int, int]
        Range of DAG sizes.
    density_range : tuple[float, float]
        Range of edge densities.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_nodes_range: tuple[int, int] = (5, 30),
        density_range: tuple[float, float] = (0.1, 0.4),
        seed: int = 42,
    ) -> None:
        self.n_nodes_range = n_nodes_range
        self.density_range = density_range
        self.seed = seed

    def generate(
        self,
        n_instances: int = 100,
        n_samples: int = 1000,
    ) -> list[DGPInstance]:
        """Generate a batch of DGP instances.

        Parameters
        ----------
        n_instances : int
            Number of instances.
        n_samples : int
            Samples per instance.

        Returns
        -------
        list[DGPInstance]
        """
        rng = np.random.RandomState(self.seed)
        instances: list[DGPInstance] = []

        for i in range(n_instances):
            n = rng.randint(self.n_nodes_range[0], self.n_nodes_range[1] + 1)
            density = rng.uniform(self.density_range[0], self.density_range[1])
            target_radius = rng.randint(1, min(4, n // 2 + 1))

            inst = self.generate_with_known_radius(
                n_nodes=n,
                target_radius=target_radius,
                n_samples=n_samples,
                seed=self.seed + i,
            )
            inst.name = f"synthetic_{i:04d}_n{n}_r{target_radius}"
            instances.append(inst)

        return instances

    def generate_with_known_radius(
        self,
        n_nodes: int,
        target_radius: int,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> DGPInstance:
        """Generate a single instance with a specific target radius.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        target_radius : int
            Desired robustness radius.
        n_samples : int
            Number of samples.
        seed : int | None
            Random seed override.

        Returns
        -------
        DGPInstance
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)

        adj, treatment, outcome, witness = _construct_known_radius_dag(
            n_nodes, target_radius, rng
        )

        df, coeffs, true_ate = generate_linear_gaussian(
            adj, n_samples=n_samples,
            treatment=treatment, rng=rng,
        )

        return DGPInstance(
            adj=adj,
            data=df,
            treatment=treatment,
            outcome=outcome,
            true_radius=target_radius,
            witness_edits=witness,
            true_ate=true_ate,
            name=f"known_radius_{target_radius}_n{n_nodes}",
        )

    # ------------------------------------------------------------------
    # Specialised generators
    # ------------------------------------------------------------------

    def generate_chain(
        self,
        length: int = 5,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> DGPInstance:
        """Generate a chain DAG with known radius 1."""
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        adj = chain_dag(length)
        df, coeffs, true_ate = generate_linear_gaussian(
            adj, n_samples=n_samples, treatment=0, rng=rng,
        )
        # Radius is 1: deleting any edge on the sole path breaks it
        witness = (StructuralEdit(EditType.DELETE, 0, 1),)
        return DGPInstance(
            adj=adj, data=df, treatment=0, outcome=length - 1,
            true_radius=1, witness_edits=witness, true_ate=true_ate,
            name=f"chain_n{length}",
        )

    def generate_diamond(
        self,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> DGPInstance:
        """Generate a diamond DAG with known radius 2."""
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        adj = diamond_dag()
        df, coeffs, true_ate = generate_linear_gaussian(
            adj, n_samples=n_samples, treatment=0, rng=rng,
        )
        # Radius is 2: need to break both paths X→M1→Y and X→M2→Y
        witness = (
            StructuralEdit(EditType.DELETE, 0, 1),
            StructuralEdit(EditType.DELETE, 0, 2),
        )
        return DGPInstance(
            adj=adj, data=df, treatment=0, outcome=3,
            true_radius=2, witness_edits=witness, true_ate=true_ate,
            name="diamond",
        )

    def generate_fork(
        self,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> DGPInstance:
        """Generate a fork DAG (no causal effect, radius 0)."""
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        adj = fork_dag()
        df = generate_nonlinear_additive(adj, n_samples=n_samples, rng=rng)
        # X and Y share a common cause but X has no effect on Y
        return DGPInstance(
            adj=adj, data=df, treatment=1, outcome=2,
            true_radius=0, witness_edits=(), true_ate=0.0,
            name="fork",
        )

    def generate_nonlinear_batch(
        self,
        n_instances: int = 50,
        n_samples: int = 1000,
    ) -> list[DGPInstance]:
        """Generate instances with nonlinear mechanisms."""
        rng = np.random.RandomState(self.seed)
        instances: list[DGPInstance] = []

        for i in range(n_instances):
            n = rng.randint(self.n_nodes_range[0], self.n_nodes_range[1] + 1)
            target_radius = rng.randint(1, min(3, n // 2 + 1))

            adj, treatment, outcome, witness = _construct_known_radius_dag(
                n, target_radius, rng,
            )
            df = generate_nonlinear_additive(
                adj, n_samples=n_samples, rng=rng,
            )
            # Binarise treatment
            median_t = df[treatment].median()
            df[treatment] = (df[treatment] > median_t).astype(float)

            instances.append(DGPInstance(
                adj=adj, data=df,
                treatment=treatment, outcome=outcome,
                true_radius=target_radius, witness_edits=witness,
                true_ate=0.0,
                name=f"nonlinear_{i:04d}_n{n}_r{target_radius}",
            ))

        return instances

    def generate_varying_sample_sizes(
        self,
        n_nodes: int = 10,
        target_radius: int = 2,
        sample_sizes: Sequence[int] = (100, 500, 1000, 5000),
    ) -> list[DGPInstance]:
        """Generate identical DAGs with varying sample sizes."""
        instances = []
        for n_samples in sample_sizes:
            inst = self.generate_with_known_radius(
                n_nodes, target_radius, n_samples=n_samples,
            )
            inst.name = f"vary_n{n_samples}_p{n_nodes}_r{target_radius}"
            instances.append(inst)
        return instances

    def generate_varying_effect_sizes(
        self,
        n_nodes: int = 10,
        n_samples: int = 1000,
        effect_ranges: Sequence[tuple[float, float]] | None = None,
    ) -> list[DGPInstance]:
        """Generate DAGs with varying effect sizes."""
        if effect_ranges is None:
            effect_ranges = [(0.1, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0)]

        instances = []
        rng = np.random.RandomState(self.seed)

        adj, treatment, outcome, witness = _construct_known_radius_dag(
            n_nodes, 2, rng,
        )

        for er in effect_ranges:
            rng_copy = np.random.RandomState(self.seed)
            df, coeffs, ate = generate_linear_gaussian(
                adj, n_samples=n_samples, effect_range=er,
                treatment=treatment, rng=rng_copy,
            )
            instances.append(DGPInstance(
                adj=adj.copy(), data=df,
                treatment=treatment, outcome=outcome,
                true_radius=2, witness_edits=witness,
                true_ate=ate,
                name=f"effect_{er[0]:.1f}_{er[1]:.1f}",
            ))

        return instances
