"""
Synthetic data generation from causal DAGs.

Generates observational datasets by sampling from linear Gaussian or
non-linear structural equation models (SEMs) parameterised by a given DAG.
Useful for testing, evaluation, and power analysis.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import pandas as pd

from causalcert.types import AdjacencyMatrix


# ============================================================================
# Internal helpers
# ============================================================================


def _topological_order(adj: np.ndarray) -> list[int]:
    """Kahn's algorithm topological sort."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    if len(order) != n:
        raise ValueError("Graph contains a cycle; cannot generate data.")
    return order


def _parents(adj: np.ndarray, v: int) -> list[int]:
    """Return parent list of node v."""
    return [int(p) for p in np.nonzero(adj[:, v])[0]]


def _descendants(adj: np.ndarray, sources: set[int]) -> set[int]:
    """BFS descendants (inclusive)."""
    visited: set[int] = set()
    queue = deque(sources)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for c in np.nonzero(adj[v])[0]:
            if int(c) not in visited:
                queue.append(int(c))
    return visited


# ============================================================================
# Linear Gaussian SCM
# ============================================================================


def generate_linear_gaussian(
    adj: AdjacencyMatrix,
    n: int = 1000,
    noise_scale: float = 1.0,
    edge_weight_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
    intercepts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data from a linear Gaussian SEM.

    Each node X_i = Σ_{j∈pa(i)} w_ji X_j + intercept_i + ε_i,
    where ε_i ~ N(0, noise_scale²).

    Edge weights are sampled uniformly from ``[-high, -low] ∪ [low, high]``
    where ``low, high = edge_weight_range``.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    n : int
        Number of samples.
    noise_scale : float
        Standard deviation of exogenous noise.
    edge_weight_range : tuple[float, float]
        Range for uniform edge weight sampling (absolute values).
    seed : int
        Random seed.
    intercepts : np.ndarray | None
        Per-node intercepts.  ``None`` for all zeros.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        ``(data, weight_matrix)`` — the generated dataset and the true
        weight matrix.
    """
    adj = np.asarray(adj, dtype=np.int8)
    p = adj.shape[0]
    rng = np.random.default_rng(seed)

    # Sample edge weights
    low, high = edge_weight_range
    W = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            if adj[i, j]:
                abs_w = rng.uniform(low, high)
                sign = rng.choice([-1, 1])
                W[i, j] = sign * abs_w

    if intercepts is None:
        intercepts = np.zeros(p)
    else:
        intercepts = np.asarray(intercepts, dtype=np.float64)

    order = _topological_order(adj)
    data = np.zeros((n, p), dtype=np.float64)

    for v in order:
        noise = rng.normal(0, noise_scale, size=n)
        pa = _parents(adj, v)
        signal = data[:, pa] @ W[pa, v] if pa else 0.0
        data[:, v] = signal + intercepts[v] + noise

    columns = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=columns), W


def generate_linear_gaussian_with_treatment(
    adj: AdjacencyMatrix,
    treatment: int,
    outcome: int,
    n: int = 1000,
    noise_scale: float = 1.0,
    true_ate: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Generate data with a specified true ATE.

    Sets the weight from treatment to outcome to *true_ate* and samples
    remaining weights randomly.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix (must have treatment → outcome edge).
    treatment : int
        Treatment node index.
    outcome : int
        Outcome node index.
    n : int
        Number of samples.
    noise_scale : float
        Noise scale.
    true_ate : float
        Desired true causal effect of treatment on outcome.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, float]
        ``(data, weight_matrix, true_ate)``
    """
    adj = np.asarray(adj, dtype=np.int8)
    df, W = generate_linear_gaussian(
        adj, n=n, noise_scale=noise_scale, seed=seed
    )

    # Override with specified ATE
    W[treatment, outcome] = true_ate

    # Re-generate with fixed weights
    p = adj.shape[0]
    rng = np.random.default_rng(seed)
    order = _topological_order(adj)
    data = np.zeros((n, p), dtype=np.float64)

    for v in order:
        noise = rng.normal(0, noise_scale, size=n)
        pa = _parents(adj, v)
        signal = data[:, pa] @ W[pa, v] if pa else 0.0
        data[:, v] = signal + noise

    # Make treatment binary via median split
    median_t = np.median(data[:, treatment])
    data[:, treatment] = (data[:, treatment] > median_t).astype(float)

    # Recompute outcome given binary treatment
    for v in order:
        if v == treatment:
            continue
        noise = rng.normal(0, noise_scale, size=n)
        pa = _parents(adj, v)
        signal = data[:, pa] @ W[pa, v] if pa else 0.0
        data[:, v] = signal + noise

    columns = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=columns), W, true_ate


# ============================================================================
# Nonlinear SCM
# ============================================================================


def generate_nonlinear(
    adj: AdjacencyMatrix,
    n: int = 1000,
    mechanism: str = "gp",
    noise_scale: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data from a non-linear additive noise SEM.

    X_i = f_i(pa(X_i)) + ε_i

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    n : int
        Number of samples.
    mechanism : str
        Non-linear mechanism: ``"gp"`` (Gaussian process-like),
        ``"mlp"`` (random MLP), ``"polynomial"``.
    noise_scale : float
        Noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
    """
    adj = np.asarray(adj, dtype=np.int8)
    p = adj.shape[0]
    rng = np.random.default_rng(seed)
    order = _topological_order(adj)

    mechanism_fns = _build_mechanisms(adj, mechanism, rng)

    data = np.zeros((n, p), dtype=np.float64)
    for v in order:
        noise = rng.normal(0, noise_scale, size=n)
        pa = _parents(adj, v)
        if pa:
            parent_data = data[:, pa]
            data[:, v] = mechanism_fns[v](parent_data) + noise
        else:
            data[:, v] = noise

    columns = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=columns)


def _build_mechanisms(
    adj: np.ndarray,
    mechanism: str,
    rng: np.random.Generator,
) -> dict[int, Callable]:
    """Build nonlinear mechanisms for each node."""
    p = adj.shape[0]
    fns: dict[int, Callable] = {}

    for v in range(p):
        pa = _parents(adj, v)
        if not pa:
            fns[v] = lambda x: 0.0
            continue

        n_pa = len(pa)

        if mechanism == "polynomial":
            # Random polynomial: sum of w_j * x_j + w_jj * x_j^2
            w_lin = rng.uniform(-1.5, 1.5, size=n_pa)
            w_quad = rng.uniform(-0.5, 0.5, size=n_pa)
            w_cross = rng.uniform(-0.3, 0.3) if n_pa >= 2 else 0.0

            def _poly(x: np.ndarray, wl: np.ndarray = w_lin, wq: np.ndarray = w_quad, wc: float = w_cross) -> np.ndarray:
                result = x @ wl + (x ** 2) @ wq
                if x.shape[1] >= 2:
                    result += wc * x[:, 0] * x[:, 1]
                return result

            fns[v] = _poly

        elif mechanism == "mlp":
            # Random single hidden layer MLP
            hidden = max(5, n_pa * 2)
            W1 = rng.standard_normal((n_pa, hidden)) * 0.5
            b1 = rng.standard_normal(hidden) * 0.1
            W2 = rng.standard_normal(hidden) * 0.5

            def _mlp(x: np.ndarray, w1: np.ndarray = W1, bb: np.ndarray = b1, w2: np.ndarray = W2) -> np.ndarray:
                h = np.tanh(x @ w1 + bb)
                return h @ w2

            fns[v] = _mlp

        else:  # "gp" — GP-like (random Fourier features)
            n_features = 20
            Omega = rng.standard_normal((n_pa, n_features))
            bias = rng.uniform(0, 2 * np.pi, size=n_features)
            weights = rng.standard_normal(n_features)

            def _gp(x: np.ndarray, om: np.ndarray = Omega, b: np.ndarray = bias, w: np.ndarray = weights) -> np.ndarray:
                features = np.cos(x @ om + b) * np.sqrt(2.0 / om.shape[1])
                return features @ w

            fns[v] = _gp

    return fns


# ============================================================================
# Binary / categorical generation
# ============================================================================


def generate_with_binary_treatment(
    adj: AdjacencyMatrix,
    treatment: int,
    n: int = 1000,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data where the treatment node is binary.

    Other nodes follow a linear Gaussian SEM.  The treatment is generated
    via a logistic link from its parents.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : int
        Treatment node index.
    n : int
        Number of samples.
    noise_scale : float
        Noise scale for non-treatment nodes.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
    """
    adj = np.asarray(adj, dtype=np.int8)
    p = adj.shape[0]
    rng = np.random.default_rng(seed)
    order = _topological_order(adj)

    # Random weights
    W = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            if adj[i, j]:
                W[i, j] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)

    data = np.zeros((n, p), dtype=np.float64)

    for v in order:
        pa = _parents(adj, v)
        if v == treatment:
            if pa:
                logit = data[:, pa] @ W[pa, v]
            else:
                logit = np.zeros(n)
            prob = 1.0 / (1.0 + np.exp(-logit))
            data[:, v] = rng.binomial(1, prob).astype(float)
        else:
            noise = rng.normal(0, noise_scale, size=n)
            signal = data[:, pa] @ W[pa, v] if pa else 0.0
            data[:, v] = signal + noise

    columns = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=columns)


def generate_categorical_node(
    parent_data: np.ndarray,
    n_categories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a categorical node from parent values via softmax.

    Parameters
    ----------
    parent_data : np.ndarray
        Parent values, shape ``(n, n_parents)``.
    n_categories : int
        Number of categories.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Category assignments, shape ``(n,)``.
    """
    n = parent_data.shape[0]
    n_pa = parent_data.shape[1] if parent_data.ndim == 2 else 1
    if parent_data.ndim == 1:
        parent_data = parent_data.reshape(-1, 1)

    W = rng.standard_normal((n_pa, n_categories))
    logits = parent_data @ W
    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    categories = np.array([
        rng.choice(n_categories, p=probs[i])
        for i in range(n)
    ])
    return categories


# ============================================================================
# Interventional data (do-calculus)
# ============================================================================


def generate_interventional(
    adj: AdjacencyMatrix,
    intervention_node: int,
    intervention_value: float,
    n: int = 1000,
    noise_scale: float = 1.0,
    edge_weight_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
    weight_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data under a do-intervention: do(X_k = v).

    Cuts all incoming edges to ``intervention_node`` and sets it to the
    fixed value.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    intervention_node : int
        Node to intervene on.
    intervention_value : float
        Value to set the intervened node to.
    n : int
        Number of samples.
    noise_scale : float
        Noise scale.
    edge_weight_range : tuple[float, float]
        Edge weight sampling range (used only if weight_matrix is None).
    seed : int
        Random seed.
    weight_matrix : np.ndarray | None
        Pre-specified weight matrix. If None, generated randomly.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        ``(data, weight_matrix)``
    """
    adj = np.asarray(adj, dtype=np.int8)
    p = adj.shape[0]
    rng = np.random.default_rng(seed)

    if weight_matrix is None:
        low, high = edge_weight_range
        W = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                if adj[i, j]:
                    W[i, j] = rng.choice([-1, 1]) * rng.uniform(low, high)
    else:
        W = np.asarray(weight_matrix, dtype=np.float64)

    # Mutilated graph: remove edges into intervention node
    adj_do = adj.copy()
    adj_do[:, intervention_node] = 0

    order = _topological_order(adj_do)
    data = np.zeros((n, p), dtype=np.float64)

    for v in order:
        if v == intervention_node:
            data[:, v] = intervention_value
        else:
            noise = rng.normal(0, noise_scale, size=n)
            pa = _parents(adj_do, v)
            # Use original weights (not mutilated) for non-intervened edges
            signal = data[:, pa] @ W[pa, v] if pa else 0.0
            data[:, v] = signal + noise

    columns = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=columns), W


# ============================================================================
# Counterfactual data
# ============================================================================


def generate_counterfactual(
    adj: AdjacencyMatrix,
    treatment: int,
    factual_data: pd.DataFrame,
    weight_matrix: np.ndarray,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate counterfactual data by flipping the treatment.

    For each observation, computes what would have happened had the
    treatment been the opposite value (assuming known linear SCM).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : int
        Treatment node index.
    factual_data : pd.DataFrame
        Observed data.
    weight_matrix : np.ndarray
        True SCM weight matrix.
    noise_scale : float
        Noise scale (used to infer exogenous noise).
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(cf_data_t0, cf_data_t1)`` — counterfactual data under T=0 and T=1.
    """
    adj = np.asarray(adj, dtype=np.int8)
    W = np.asarray(weight_matrix, dtype=np.float64)
    p = adj.shape[0]
    n = len(factual_data)
    order = _topological_order(adj)

    factual = factual_data.values.astype(np.float64)

    # Infer exogenous noise: ε_i = X_i - Σ_{j∈pa(i)} W_{j,i} X_j
    noise = np.zeros_like(factual)
    for v in order:
        pa = _parents(adj, v)
        if pa:
            signal = factual[:, pa] @ W[pa, v]
        else:
            signal = 0.0
        noise[:, v] = factual[:, v] - signal

    results = []
    for t_val in [0.0, 1.0]:
        cf = np.zeros((n, p), dtype=np.float64)
        for v in order:
            if v == treatment:
                cf[:, v] = t_val
            else:
                pa = _parents(adj, v)
                signal = cf[:, pa] @ W[pa, v] if pa else 0.0
                cf[:, v] = signal + noise[:, v]
        results.append(pd.DataFrame(cf, columns=factual_data.columns))

    return results[0], results[1]


# ============================================================================
# Ground truth effects
# ============================================================================


def compute_true_ate_linear(
    adj: AdjacencyMatrix,
    weight_matrix: np.ndarray,
    treatment: int,
    outcome: int,
) -> float:
    """Compute the true ATE for a linear Gaussian SCM with binary treatment.

    In a linear SCM, the causal effect of X on Y equals the sum of all
    directed-path products from X to Y.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    weight_matrix : np.ndarray
        Weight matrix.
    treatment : int
        Treatment node.
    outcome : int
        Outcome node.

    Returns
    -------
    float
        True ATE.
    """
    adj = np.asarray(adj, dtype=np.int8)
    W = np.asarray(weight_matrix, dtype=np.float64)
    p = adj.shape[0]

    # Sum over all directed paths from treatment to outcome
    # Use (I - W^T)^{-1} to get total effects
    # Total effect = entry [treatment, outcome] of (I - W)^{-1} ... no,
    # for linear SCM: X = W^T X + ε, so X = (I - W^T)^{-1} ε
    # Causal effect of do(X_t = x_t + 1) on X_o:
    # = sum of all directed-path products
    return _sum_directed_paths(adj, W, treatment, outcome)


def _sum_directed_paths(
    adj: np.ndarray,
    W: np.ndarray,
    source: int,
    target: int,
) -> float:
    """Sum products of edge weights along all directed paths."""
    if source == target:
        return 1.0

    total = 0.0
    # BFS/DFS over directed paths
    stack: list[tuple[int, float]] = [(source, 1.0)]
    while stack:
        node, product = stack.pop()
        for child in np.nonzero(adj[node])[0]:
            child = int(child)
            new_product = product * W[node, child]
            if child == target:
                total += new_product
            else:
                stack.append((child, new_product))
    return total


# ============================================================================
# Confounder generation
# ============================================================================


def generate_with_confounders(
    n_obs_variables: int,
    n_confounders: int,
    treatment: int,
    outcome: int,
    n: int = 1000,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate data with explicit confounders between treatment and outcome.

    Creates a DAG where ``n_confounders`` variables are parents of both
    the treatment and the outcome.

    Parameters
    ----------
    n_obs_variables : int
        Total number of observed variables.
    n_confounders : int
        Number of confounders (must be ≤ n_obs_variables - 2).
    treatment : int
        Treatment index.
    outcome : int
        Outcome index.
    n : int
        Number of samples.
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        ``(data, adjacency_matrix, weight_matrix)``
    """
    p = n_obs_variables
    if n_confounders > p - 2:
        raise ValueError("Too many confounders for the number of variables.")

    adj = np.zeros((p, p), dtype=np.int8)

    # Treatment → Outcome
    adj[treatment, outcome] = 1

    # Assign confounders: first n_confounders nodes (excluding treatment/outcome)
    available = [i for i in range(p) if i != treatment and i != outcome]
    confounders = available[:n_confounders]

    for c in confounders:
        adj[c, treatment] = 1
        adj[c, outcome] = 1

    df, W = generate_linear_gaussian(
        adj, n=n, noise_scale=noise_scale, seed=seed
    )
    return df, adj, W


# ============================================================================
# Random DAG generation
# ============================================================================


def random_dag(
    n_nodes: int,
    edge_prob: float = 0.3,
    seed: int = 42,
) -> AdjacencyMatrix:
    """Generate a random DAG via Erdős-Rényi over a fixed topological order.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Probability of each forward edge.
    seed : int
        Random seed.

    Returns
    -------
    AdjacencyMatrix
        Random DAG adjacency matrix (guaranteed acyclic).
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    # Edges only go from lower to higher index (topological order)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1

    return adj
