"""Phase diagrams for Graph Neural Networks.

Extends the finite-width phase diagram framework to GNNs, accounting for
message-passing structure, graph topology, over-smoothing, and over-squashing.
The NTK for GNNs is modified by the graph adjacency/Laplacian spectrum,
leading to topology-dependent phase boundaries.

Example
-------
>>> from phase_diagrams.graph_neural_networks import gnn_phase_diagram
>>> diagram = gnn_phase_diagram(model, graph_data, lr_range=(1e-4, 1.0))
>>> print(diagram.metadata["architecture"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .api import PhaseDiagram, PhasePoint, Regime, _compute_gamma, _predict_gamma_star


# ======================================================================
# Enums and data classes
# ======================================================================

class AggregationType(str, Enum):
    """Message-passing aggregation strategy."""
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    ATTENTION = "attention"


class GNNVariant(str, Enum):
    """Graph neural network architecture variant."""
    GCN = "gcn"
    GAT = "gat"
    GIN = "gin"
    GRAPHSAGE = "graphsage"
    MPNN = "mpnn"


@dataclass
class NTKResult:
    """Result of message-passing NTK computation.

    Attributes
    ----------
    kernel_matrix : NDArray
        NTK Gram matrix of shape (n_nodes, n_nodes).
    eigenvalues : NDArray
        Eigenvalues of the NTK Gram matrix, sorted descending.
    spectral_gap : float
        Gap between first and second eigenvalues.
    effective_rank : float
        Effective rank of the kernel (trace / max eigenvalue).
    graph_coupling : float
        Measure of how strongly graph structure modifies the NTK
        relative to a fully-connected equivalent.
    per_layer_contributions : Dict[int, NDArray]
        Per-layer NTK eigenspectra.
    """
    kernel_matrix: NDArray = field(default_factory=lambda: np.array([]))
    eigenvalues: NDArray = field(default_factory=lambda: np.array([]))
    spectral_gap: float = 0.0
    effective_rank: float = 0.0
    graph_coupling: float = 0.0
    per_layer_contributions: Dict[int, NDArray] = field(default_factory=dict)


@dataclass
class SmoothingCurve:
    """Over-smoothing prediction as a function of depth.

    Attributes
    ----------
    depths : NDArray
        Array of layer depths evaluated.
    dirichlet_energies : NDArray
        Dirichlet energy of node features at each depth.
    smoothing_rate : float
        Exponential decay rate of Dirichlet energy per layer.
    critical_depth : int
        Depth at which Dirichlet energy falls below threshold.
    regime_at_depth : Dict[int, Regime]
        Phase regime at each evaluated depth.
    feature_entropy : NDArray
        Shannon entropy of node feature distributions at each depth.
    """
    depths: NDArray = field(default_factory=lambda: np.array([]))
    dirichlet_energies: NDArray = field(default_factory=lambda: np.array([]))
    smoothing_rate: float = 0.0
    critical_depth: int = 0
    regime_at_depth: Dict[int, Regime] = field(default_factory=dict)
    feature_entropy: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class SquashingReport:
    """Over-squashing analysis report.

    Attributes
    ----------
    bottleneck_nodes : List[int]
        Node indices that act as information bottlenecks.
    jacobian_norms : NDArray
        Norms of the Jacobian of each node's representation w.r.t. distant nodes.
    effective_resistance : NDArray
        Effective resistance matrix (n_nodes x n_nodes).
    squashing_severity : float
        Scalar summary of squashing (0 = none, 1 = severe).
    cheeger_constant : float
        Cheeger constant of the graph (expansion quality).
    recommended_rewiring : Dict[str, Any]
        Suggested graph rewiring operations to alleviate squashing.
    sensitivity_by_distance : NDArray
        Average Jacobian norm as function of graph distance.
    """
    bottleneck_nodes: List[int] = field(default_factory=list)
    jacobian_norms: NDArray = field(default_factory=lambda: np.array([]))
    effective_resistance: NDArray = field(default_factory=lambda: np.array([]))
    squashing_severity: float = 0.0
    cheeger_constant: float = 0.0
    recommended_rewiring: Dict[str, Any] = field(default_factory=dict)
    sensitivity_by_distance: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class ScalingCurve:
    """How phase boundaries scale with graph size.

    Attributes
    ----------
    graph_sizes : NDArray
        Number of nodes in each graph evaluated.
    critical_lrs : NDArray
        Critical learning rate at each graph size.
    critical_gammas : NDArray
        Critical coupling at each graph size.
    scaling_exponent : float
        Fitted exponent alpha in gamma_star ~ n^{-alpha}.
    scaling_r_squared : float
        R^2 of the power-law fit.
    per_size_diagrams : Dict[int, PhaseDiagram]
        Full phase diagram at each graph size.
    """
    graph_sizes: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    critical_gammas: NDArray = field(default_factory=lambda: np.array([]))
    scaling_exponent: float = 0.0
    scaling_r_squared: float = 0.0
    per_size_diagrams: Dict[int, PhaseDiagram] = field(default_factory=dict)


@dataclass
class SpectralReport:
    """Spectral analysis of graph structure and its effect on training dynamics.

    Attributes
    ----------
    adjacency_eigenvalues : NDArray
        Eigenvalues of the adjacency matrix.
    laplacian_eigenvalues : NDArray
        Eigenvalues of the normalized Laplacian.
    spectral_gap : float
        Lambda_2 - lambda_1 of the Laplacian.
    mixing_time : float
        Estimated mixing time of random walk on the graph.
    ntk_spectral_alignment : float
        Alignment between NTK and Laplacian eigenspaces.
    phase_boundary_from_spectrum : float
        Predicted critical LR from graph spectral properties.
    community_structure : Dict[str, Any]
        Detected community structure and its effect on phases.
    """
    adjacency_eigenvalues: NDArray = field(default_factory=lambda: np.array([]))
    laplacian_eigenvalues: NDArray = field(default_factory=lambda: np.array([]))
    spectral_gap: float = 0.0
    mixing_time: float = 0.0
    ntk_spectral_alignment: float = 0.0
    phase_boundary_from_spectrum: float = 0.0
    community_structure: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _validate_adjacency(adj: NDArray) -> NDArray:
    """Validate and normalize adjacency matrix."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj.shape}")
    adj = adj.astype(np.float64)
    # symmetrize if needed
    if not np.allclose(adj, adj.T, atol=1e-10):
        adj = (adj + adj.T) / 2.0
    return adj


def _normalized_laplacian(adj: NDArray) -> NDArray:
    """Compute the symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}."""
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    n = adj.shape[0]
    return np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt


def _gcn_propagation(features: NDArray, adj: NDArray, weight: NDArray) -> NDArray:
    """Single GCN layer: H' = sigma(D^{-1/2} A D^{-1/2} H W)."""
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = adj + np.eye(adj.shape[0])
    deg_hat = A_hat.sum(axis=1)
    deg_hat_inv_sqrt = np.where(deg_hat > 0, 1.0 / np.sqrt(deg_hat), 0.0)
    D_hat_inv_sqrt = np.diag(deg_hat_inv_sqrt)
    norm_adj = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
    return np.maximum(norm_adj @ features @ weight, 0)


def _compute_dirichlet_energy(features: NDArray, laplacian: NDArray) -> float:
    """Dirichlet energy E(H) = tr(H^T L H) / tr(H^T H)."""
    numerator = np.trace(features.T @ laplacian @ features)
    denominator = np.trace(features.T @ features)
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _effective_resistance_matrix(laplacian: NDArray) -> NDArray:
    """Compute effective resistance between all node pairs using pseudoinverse."""
    n = laplacian.shape[0]
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    # pseudoinverse of Laplacian (skip zero eigenvalue)
    mask = eigvals > 1e-10
    L_pinv = eigvecs[:, mask] @ np.diag(1.0 / eigvals[mask]) @ eigvecs[:, mask].T
    R = np.zeros((n, n))
    diag = np.diag(L_pinv)
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = diag[i] + diag[j] - 2 * L_pinv[i, j]
            R[j, i] = R[i, j]
    return R


def _cheeger_constant_approx(laplacian: NDArray) -> float:
    """Approximate the Cheeger constant via the Cheeger inequality: h >= lambda_2 / 2."""
    eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    if len(eigvals) < 2:
        return 0.0
    lambda_2 = eigvals[1] if eigvals[1] > 1e-10 else eigvals[min(2, len(eigvals) - 1)]
    return float(lambda_2 / 2.0)


def _build_random_graph(n_nodes: int, edge_prob: float, seed: int = 42) -> NDArray:
    """Build an Erdos-Renyi random graph adjacency matrix."""
    rng = np.random.RandomState(seed)
    adj = (rng.rand(n_nodes, n_nodes) < edge_prob).astype(float)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T
    np.fill_diagonal(adj, 0)
    return adj


def _graph_distance_matrix(adj: NDArray) -> NDArray:
    """BFS-based shortest path distances between all node pairs."""
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    for source in range(n):
        visited = {source}
        frontier = [source]
        d = 0
        while frontier:
            d += 1
            next_frontier = []
            for node in frontier:
                neighbors = np.where(adj[node] > 0)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        dist[source, nb] = d
                        next_frontier.append(nb)
            frontier = next_frontier
    return dist


def _message_passing_ntk_layer(
    features: NDArray,
    adj_norm: NDArray,
    weight: NDArray,
    activation_derivative: NDArray,
) -> NDArray:
    """Compute per-layer NTK contribution for a single GNN layer.

    For a GCN layer h' = sigma(A_norm @ h @ W), the NTK contribution is:
    K_l(i,j) = (A_norm @ features)[i] . (A_norm @ features)[j] * sigma'(pre_i) * sigma'(pre_j)
    summed over output dimensions.
    """
    propagated = adj_norm @ features
    pre_act = propagated @ weight
    # Jacobian w.r.t. W: dh_i / dW = sigma'(pre_i) * kron(e_k, (A_norm @ h)_i)
    # NTK_l(i,j) = sigma'(pre_i) . sigma'(pre_j) * (A h)_i . (A h)_j * I_out
    n_nodes = features.shape[0]
    K_l = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            dot = np.dot(propagated[i], propagated[j])
            act_prod = np.dot(activation_derivative[i], activation_derivative[j])
            K_l[i, j] = dot * act_prod
            K_l[j, i] = K_l[i, j]
    return K_l


def _compute_gnn_ntk(
    features: NDArray,
    adj: NDArray,
    widths: List[int],
    depth: int,
    seed: int = 42,
) -> Tuple[NDArray, Dict[int, NDArray]]:
    """Full GNN NTK computation via layer-wise accumulation.

    Returns the NTK Gram matrix and per-layer eigenspectra.
    """
    rng = np.random.RandomState(seed)
    n_nodes = features.shape[0]

    # Normalized adjacency with self-loops
    A_hat = adj + np.eye(n_nodes)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    K_total = np.zeros((n_nodes, n_nodes))
    per_layer = {}
    h = features.copy()

    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = widths[l] if l < len(widths) else widths[-1]
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)

        propagated = adj_norm @ h
        pre_act = propagated @ W

        if l < depth - 1:
            act_out = np.maximum(pre_act, 0)
            act_deriv = (pre_act > 0).astype(float)
        else:
            act_out = pre_act
            act_deriv = np.ones_like(pre_act)

        K_l = _message_passing_ntk_layer(h, adj_norm, W, act_deriv)
        K_total += K_l

        eigvals_l = np.sort(np.linalg.eigvalsh(K_l))[::-1]
        per_layer[l] = eigvals_l

        h = act_out

    return K_total, per_layer


def _compute_gnn_mu_max(
    features: NDArray,
    adj: NDArray,
    widths: List[int],
    depth: int,
    seed: int = 42,
) -> float:
    """Effective perturbation eigenvalue mu_max for GNN bifurcation analysis."""
    K, _ = _compute_gnn_ntk(features, adj, widths, depth, seed)
    eigvals = np.linalg.eigvalsh(K)
    mu_max = float(np.max(eigvals))
    width = widths[0] if widths else 64
    return mu_max / width


def _extract_gnn_params(model: Any) -> Tuple[List[int], int, float, GNNVariant]:
    """Extract architecture parameters from a GNN model or config dict.

    Supports:
    - dict with keys 'widths', 'depth', 'init_scale', 'variant'
    - list of weight matrices (infers architecture)
    """
    if isinstance(model, dict):
        widths = model.get("widths", [64, 64, 64])
        depth = model.get("depth", len(widths))
        init_scale = model.get("init_scale", 1.0)
        variant = GNNVariant(model.get("variant", "gcn"))
        return widths, depth, init_scale, variant

    if isinstance(model, (list, tuple)):
        widths = [w.shape[0] if w.ndim == 2 else w.shape[1] for w in model]
        depth = len(model)
        init_scale = float(np.std(model[0]) * math.sqrt(widths[0]))
        return widths, depth, init_scale, GNNVariant.GCN

    raise TypeError(f"Unsupported model type: {type(model)}")


def _extract_graph_data(
    graph_data: Any,
) -> Tuple[NDArray, NDArray]:
    """Extract node features and adjacency from graph data.

    Supports:
    - tuple of (features, adjacency)
    - dict with 'features' and 'adjacency' keys
    """
    if isinstance(graph_data, tuple) and len(graph_data) == 2:
        features, adj = graph_data
        return np.asarray(features, dtype=np.float64), _validate_adjacency(np.asarray(adj))

    if isinstance(graph_data, dict):
        features = np.asarray(graph_data["features"], dtype=np.float64)
        adj = _validate_adjacency(np.asarray(graph_data["adjacency"]))
        return features, adj

    raise TypeError(f"Unsupported graph_data type: {type(graph_data)}")


# ======================================================================
# Public API
# ======================================================================

def gnn_phase_diagram(
    model: Any,
    graph_data: Any,
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    n_lr_steps: int = 30,
    n_width_steps: int = 10,
    training_steps: int = 100,
    width_range: Optional[Tuple[int, int]] = None,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for a GNN over learning rate and width.

    The phase boundary accounts for graph topology via the message-passing
    NTK. The adjacency spectrum modulates the effective coupling, shifting
    boundaries compared to standard MLPs.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification. Either a config dict with keys
        ``{'widths', 'depth', 'init_scale', 'variant'}`` or a list of
        weight matrices.
    graph_data : tuple or dict
        Node features and adjacency. Either ``(features, adjacency)``
        or ``{'features': ..., 'adjacency': ...}``.
    lr_range : (float, float)
        Min and max learning rate (log-spaced scan).
    n_lr_steps : int
        Number of LR grid points.
    n_width_steps : int
        Number of width grid points.
    training_steps : int
        Assumed training duration T.
    width_range : (int, int) or None
        Min/max width. If None, inferred from model.
    seed : int
        Random seed for NTK approximation.

    Returns
    -------
    PhaseDiagram
        Complete phase diagram with boundary curve.
    """
    widths, depth, init_scale, variant = _extract_gnn_params(model)
    features, adj = _extract_graph_data(graph_data)

    w0 = widths[0]
    if width_range is None:
        width_range = (max(16, w0 // 4), w0 * 4)

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    width_vals = np.unique(
        np.logspace(
            math.log10(width_range[0]),
            math.log10(width_range[1]),
            n_width_steps,
        ).astype(int)
    )

    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    # Precompute graph spectral properties for topology correction
    laplacian = _normalized_laplacian(adj)
    lap_eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    spectral_gap = float(lap_eigvals[1]) if len(lap_eigvals) > 1 else 0.0
    # Topology correction factor: denser graphs push boundary higher
    topology_factor = 1.0 + 0.5 * spectral_gap

    for w in width_vals:
        w_int = int(w)
        scaled_widths = [w_int] * depth
        mu_max = _compute_gnn_mu_max(features, adj, scaled_widths, depth, seed)
        mu_max_corrected = mu_max * topology_factor
        g_star = _predict_gamma_star(mu_max_corrected, training_steps)

        prev_regime = None
        for lr in lrs:
            gamma = _compute_gamma(lr, init_scale, w_int)
            if gamma < g_star * 0.8:
                regime = Regime.LAZY
                confidence = min(1.0, (g_star - gamma) / g_star)
            elif gamma > g_star * 1.2:
                regime = Regime.RICH
                confidence = min(1.0, (gamma - g_star) / g_star)
            else:
                regime = Regime.CRITICAL
                confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

            ntk_drift = gamma * mu_max_corrected * training_steps
            points.append(PhasePoint(
                lr=float(lr),
                width=w_int,
                regime=regime,
                gamma=gamma,
                gamma_star=g_star,
                confidence=max(0.0, min(1.0, confidence)),
                ntk_drift_predicted=ntk_drift,
            ))

            if prev_regime is not None and prev_regime != regime:
                boundary_pts.append((float(lr), w_int))
            prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None

    timescale_constants = []
    for bp_lr, bp_w in boundary_pts:
        g = _compute_gamma(bp_lr, init_scale, bp_w)
        timescale_constants.append(training_steps * g)
    tc = float(np.mean(timescale_constants)) if timescale_constants else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(int(width_vals[0]), int(width_vals[-1])),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": f"GNN-{variant.value}",
            "depth": depth,
            "widths": widths,
            "n_nodes": features.shape[0],
            "spectral_gap": spectral_gap,
            "topology_factor": topology_factor,
        },
    )


def message_passing_ntk(
    model: Any,
    graph: Any,
    seed: int = 42,
) -> NTKResult:
    """Compute the message-passing NTK for a GNN on a given graph.

    The message-passing NTK differs from the standard NTK by incorporating
    the graph adjacency structure into the kernel. For a GCN with L layers:

        K_GNN(i,j) = sum_l (A^l X)_i^T (A^l X)_j * prod_{l'<=l} sigma'(...)

    where A is the normalized adjacency with self-loops.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification.
    graph : tuple or dict
        Node features and adjacency matrix.
    seed : int
        Random seed.

    Returns
    -------
    NTKResult
        NTK Gram matrix, eigenvalues, and per-layer contributions.
    """
    widths, depth, init_scale, variant = _extract_gnn_params(model)
    features, adj = _extract_graph_data(graph)

    K, per_layer = _compute_gnn_ntk(features, adj, widths, depth, seed)

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    spectral_gap = float(eigvals[0] - eigvals[1]) if len(eigvals) > 1 else 0.0
    eff_rank = float(np.sum(eigvals) / (eigvals[0] + 1e-12)) if len(eigvals) > 0 else 0.0

    # Compute graph coupling: ratio of GNN NTK spectral spread to MLP NTK
    rng = np.random.RandomState(seed)
    h = features.copy()
    K_mlp = np.zeros_like(K)
    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = widths[l] if l < len(widths) else widths[-1]
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        pre = h @ W
        act_deriv = (pre > 0).astype(float) if l < depth - 1 else np.ones_like(pre)
        for i in range(features.shape[0]):
            for j in range(i, features.shape[0]):
                val = np.dot(h[i], h[j]) * np.dot(act_deriv[i], act_deriv[j])
                K_mlp[i, j] += val
                K_mlp[j, i] = K_mlp[i, j]
        h = np.maximum(pre, 0) if l < depth - 1 else pre

    mlp_eigvals = np.sort(np.linalg.eigvalsh(K_mlp))[::-1]
    gnn_spread = float(eigvals[0] - eigvals[-1]) if len(eigvals) > 1 else 1.0
    mlp_spread = float(mlp_eigvals[0] - mlp_eigvals[-1]) if len(mlp_eigvals) > 1 else 1.0
    graph_coupling = gnn_spread / (mlp_spread + 1e-12)

    return NTKResult(
        kernel_matrix=K,
        eigenvalues=eigvals,
        spectral_gap=spectral_gap,
        effective_rank=eff_rank,
        graph_coupling=graph_coupling,
        per_layer_contributions=per_layer,
    )


def over_smoothing_prediction(
    model: Any,
    n_layers: int,
    graph_data: Optional[Any] = None,
    n_nodes: int = 100,
    feature_dim: int = 16,
    edge_prob: float = 0.1,
    smoothing_threshold: float = 0.05,
    seed: int = 42,
) -> SmoothingCurve:
    """Predict over-smoothing behavior as a function of GNN depth.

    Over-smoothing occurs when repeated message passing causes all node
    representations to converge, destroying distinguishing information.
    The Dirichlet energy E(H) = tr(H^T L H) / tr(H^T H) measures
    representation smoothness and decays exponentially with depth for
    most graph topologies.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification.
    n_layers : int
        Maximum number of layers to evaluate.
    graph_data : tuple or dict or None
        If provided, use this graph. Otherwise generate random graph.
    n_nodes : int
        Nodes in generated graph (ignored if graph_data given).
    feature_dim : int
        Feature dimension for generated data.
    edge_prob : float
        Edge probability for generated graph.
    smoothing_threshold : float
        Dirichlet energy below which we declare over-smoothing.
    seed : int
        Random seed.

    Returns
    -------
    SmoothingCurve
        Dirichlet energies and regime predictions at each depth.
    """
    rng = np.random.RandomState(seed)

    if graph_data is not None:
        features, adj = _extract_graph_data(graph_data)
    else:
        adj = _build_random_graph(n_nodes, edge_prob, seed)
        features = rng.randn(n_nodes, feature_dim)

    widths, _, init_scale, _ = _extract_gnn_params(model)
    laplacian = _normalized_laplacian(adj)

    # Normalized adjacency with self-loops
    n = adj.shape[0]
    A_hat = adj + np.eye(n)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    depths = np.arange(1, n_layers + 1)
    dirichlet_energies = np.zeros(n_layers)
    feature_entropy = np.zeros(n_layers)
    regime_at_depth: Dict[int, Regime] = {}

    h = features.copy()
    critical_depth = n_layers

    for l_idx, d in enumerate(depths):
        fan_in = h.shape[1]
        fan_out = widths[min(l_idx, len(widths) - 1)]
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        h = _gcn_propagation(h, adj, W)

        # Normalize features for stable Dirichlet energy computation
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)

        de = _compute_dirichlet_energy(h_norm, laplacian)
        dirichlet_energies[l_idx] = de

        # Feature entropy: discretize and compute Shannon entropy
        h_flat = h_norm.flatten()
        hist, _ = np.histogram(h_flat, bins=50, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        feature_entropy[l_idx] = -np.sum(hist * np.log(hist + 1e-12))

        if de < smoothing_threshold and critical_depth == n_layers:
            critical_depth = int(d)

        # Phase regime depends on whether features are still distinctive
        if de > 0.3:
            regime_at_depth[int(d)] = Regime.RICH
        elif de > smoothing_threshold:
            regime_at_depth[int(d)] = Regime.CRITICAL
        else:
            regime_at_depth[int(d)] = Regime.LAZY

    # Fit exponential decay rate: E(l) ~ E(0) * exp(-rate * l)
    valid = dirichlet_energies > 1e-12
    if np.sum(valid) > 2:
        log_de = np.log(dirichlet_energies[valid] + 1e-12)
        depths_valid = depths[valid].astype(float)
        coeffs = np.polyfit(depths_valid, log_de, 1)
        smoothing_rate = -float(coeffs[0])
    else:
        smoothing_rate = 0.0

    return SmoothingCurve(
        depths=depths.astype(float),
        dirichlet_energies=dirichlet_energies,
        smoothing_rate=smoothing_rate,
        critical_depth=critical_depth,
        regime_at_depth=regime_at_depth,
        feature_entropy=feature_entropy,
    )


def over_squashing_analysis(
    model: Any,
    graph: Any,
    max_distance: int = 10,
    seed: int = 42,
) -> SquashingReport:
    """Analyze over-squashing in a GNN on a given graph.

    Over-squashing occurs when information from distant nodes is
    exponentially attenuated through message passing. This is quantified
    by the Jacobian of node representations w.r.t. input features of
    distant nodes, which decays as O(lambda_1^d) where lambda_1 is the
    largest eigenvalue of the normalized adjacency and d is graph distance.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification.
    graph : tuple or dict
        Node features and adjacency.
    max_distance : int
        Maximum graph distance to analyze.
    seed : int
        Random seed.

    Returns
    -------
    SquashingReport
        Bottleneck nodes, Jacobian norms, effective resistance, and
        recommendations.
    """
    widths, depth, init_scale, _ = _extract_gnn_params(model)
    features, adj = _extract_graph_data(graph)
    n_nodes = features.shape[0]

    laplacian = _normalized_laplacian(adj)
    eff_resistance = _effective_resistance_matrix(laplacian)
    cheeger = _cheeger_constant_approx(laplacian)
    dist_matrix = _graph_distance_matrix(adj)

    # Approximate Jacobian norms via power iteration on the normalized adjacency
    A_hat = adj + np.eye(n_nodes)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # Compute Jacobian norm: ||dh_i^L / dx_j|| via L-step propagation
    jacobian_norms = np.zeros((n_nodes, n_nodes))
    adj_power = np.eye(n_nodes)
    for l in range(depth):
        adj_power = adj_norm @ adj_power
        # Contraction factor from ReLU (approximately 0.5 per layer)
        contraction = 0.5 ** (l + 1)
        jacobian_norms += adj_power * contraction

    # Sensitivity by graph distance
    unique_dists = np.unique(dist_matrix[np.isfinite(dist_matrix)])
    unique_dists = unique_dists[unique_dists > 0]
    unique_dists = unique_dists[unique_dists <= max_distance]
    sensitivity_by_distance = np.zeros(len(unique_dists))
    for idx, d in enumerate(unique_dists):
        mask = (dist_matrix == d)
        if np.any(mask):
            sensitivity_by_distance[idx] = float(np.mean(jacobian_norms[mask]))

    # Identify bottleneck nodes: high betweenness approximated by
    # how much effective resistance they contribute
    node_importance = np.zeros(n_nodes)
    for i in range(n_nodes):
        # Sum of inverse effective resistances to all other nodes
        for j in range(n_nodes):
            if i != j and eff_resistance[i, j] > 1e-12:
                node_importance[i] += 1.0 / eff_resistance[i, j]

    # Top 10% as bottlenecks
    threshold = np.percentile(node_importance, 90)
    bottleneck_nodes = [int(i) for i in range(n_nodes) if node_importance[i] >= threshold]

    # Overall squashing severity: ratio of sensitivity at max distance to distance 1
    if len(sensitivity_by_distance) >= 2 and sensitivity_by_distance[0] > 1e-12:
        squashing_severity = 1.0 - float(
            sensitivity_by_distance[-1] / sensitivity_by_distance[0]
        )
    else:
        squashing_severity = 0.0
    squashing_severity = max(0.0, min(1.0, squashing_severity))

    # Recommendations based on analysis
    recommendations: Dict[str, Any] = {}
    if squashing_severity > 0.7:
        recommendations["strategy"] = "graph_rewiring"
        recommendations["add_edges"] = len(bottleneck_nodes) * 2
        recommendations["description"] = (
            "Severe over-squashing detected. Consider adding skip connections "
            "between distant nodes or using graph transformers."
        )
    elif squashing_severity > 0.4:
        recommendations["strategy"] = "multi_scale"
        recommendations["description"] = (
            "Moderate over-squashing. Consider multi-scale message passing "
            "or reducing model depth."
        )
    else:
        recommendations["strategy"] = "none"
        recommendations["description"] = "Over-squashing is not a significant issue."

    if cheeger < 0.1:
        recommendations["expansion_warning"] = (
            "Low Cheeger constant indicates poor graph expansion. "
            "Consider adding virtual nodes or spectral rewiring."
        )

    return SquashingReport(
        bottleneck_nodes=bottleneck_nodes,
        jacobian_norms=jacobian_norms,
        effective_resistance=eff_resistance,
        squashing_severity=squashing_severity,
        cheeger_constant=cheeger,
        recommended_rewiring=recommendations,
        sensitivity_by_distance=sensitivity_by_distance,
    )


def graph_size_scaling(
    model: Any,
    graph_sizes: Sequence[int],
    feature_dim: int = 16,
    edge_prob: float = 0.1,
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    n_lr_steps: int = 20,
    training_steps: int = 100,
    seed: int = 42,
) -> ScalingCurve:
    """Analyze how phase boundaries scale with graph size.

    For each graph size n, generates a random graph and computes the
    phase diagram. The critical learning rate and coupling are then
    fit to power laws in n to characterize the scaling behavior.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification.
    graph_sizes : sequence of int
        Graph sizes (number of nodes) to evaluate.
    feature_dim : int
        Feature dimensionality.
    edge_prob : float
        Edge probability for Erdos-Renyi graphs.
    lr_range : (float, float)
        Learning rate range.
    n_lr_steps : int
        LR grid resolution.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    ScalingCurve
        Critical LRs and gammas at each size, with fitted scaling exponent.
    """
    widths, depth, init_scale, variant = _extract_gnn_params(model)

    sizes = np.array(sorted(graph_sizes))
    critical_lrs = np.zeros(len(sizes))
    critical_gammas = np.zeros(len(sizes))
    per_size_diagrams: Dict[int, PhaseDiagram] = {}

    for idx, n_nodes in enumerate(sizes):
        rng = np.random.RandomState(seed + idx)
        adj = _build_random_graph(int(n_nodes), edge_prob, seed + idx)
        features = rng.randn(int(n_nodes), feature_dim)
        graph_data = (features, adj)

        diagram = gnn_phase_diagram(
            model, graph_data,
            lr_range=lr_range,
            n_lr_steps=n_lr_steps,
            n_width_steps=1,
            training_steps=training_steps,
            seed=seed + idx,
        )
        per_size_diagrams[int(n_nodes)] = diagram

        # Extract critical LR from boundary points
        boundary_gammas = [p.gamma_star for p in diagram.points if p.regime == Regime.CRITICAL]
        boundary_lrs = [p.lr for p in diagram.points if p.regime == Regime.CRITICAL]

        if boundary_gammas:
            critical_gammas[idx] = float(np.mean(boundary_gammas))
            critical_lrs[idx] = float(np.mean(boundary_lrs))
        else:
            # Estimate from transition between lazy and rich
            lazy_lrs = [p.lr for p in diagram.points if p.regime == Regime.LAZY]
            rich_lrs = [p.lr for p in diagram.points if p.regime == Regime.RICH]
            if lazy_lrs and rich_lrs:
                critical_lrs[idx] = (max(lazy_lrs) + min(rich_lrs)) / 2.0
                w = widths[0]
                critical_gammas[idx] = _compute_gamma(critical_lrs[idx], init_scale, w)
            else:
                critical_lrs[idx] = float(np.sqrt(lr_range[0] * lr_range[1]))
                critical_gammas[idx] = _compute_gamma(critical_lrs[idx], init_scale, widths[0])

    # Fit power law: gamma_star ~ n^{-alpha}
    valid = critical_gammas > 1e-12
    if np.sum(valid) > 2:
        log_n = np.log(sizes[valid].astype(float))
        log_g = np.log(critical_gammas[valid])
        coeffs = np.polyfit(log_n, log_g, 1)
        scaling_exponent = -float(coeffs[0])

        # R-squared
        predicted = np.polyval(coeffs, log_n)
        ss_res = np.sum((log_g - predicted) ** 2)
        ss_tot = np.sum((log_g - np.mean(log_g)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    else:
        scaling_exponent = 0.0
        r_squared = 0.0

    return ScalingCurve(
        graph_sizes=sizes.astype(float),
        critical_lrs=critical_lrs,
        critical_gammas=critical_gammas,
        scaling_exponent=scaling_exponent,
        scaling_r_squared=r_squared,
        per_size_diagrams=per_size_diagrams,
    )


def spectral_analysis(
    model: Any,
    graph: Any,
    n_communities: int = 0,
    seed: int = 42,
) -> SpectralReport:
    """Perform spectral analysis of graph structure and its NTK implications.

    Computes adjacency and Laplacian spectra, mixing time estimates, and
    the alignment between graph spectral structure and the NTK. Community
    structure detection uses spectral clustering on the Laplacian.

    Parameters
    ----------
    model : dict or list of NDArray
        GNN specification.
    graph : tuple or dict
        Node features and adjacency.
    n_communities : int
        Number of communities to detect. If 0, auto-detect via eigengap.
    seed : int
        Random seed.

    Returns
    -------
    SpectralReport
        Full spectral decomposition with phase boundary predictions.
    """
    widths, depth, init_scale, _ = _extract_gnn_params(model)
    features, adj = _extract_graph_data(graph)
    n_nodes = features.shape[0]

    # Adjacency spectrum
    adj_eigvals = np.sort(np.linalg.eigvalsh(adj))[::-1]

    # Laplacian spectrum
    laplacian = _normalized_laplacian(adj)
    lap_eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    spectral_gap = float(lap_eigvals[1]) if len(lap_eigvals) > 1 else 0.0

    # Mixing time estimate: t_mix ~ 1 / spectral_gap
    mixing_time = 1.0 / (spectral_gap + 1e-12) if spectral_gap > 0 else float("inf")

    # NTK computation for alignment
    K, _ = _compute_gnn_ntk(features, adj, widths, depth, seed)
    ntk_eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    _, ntk_eigvecs = np.linalg.eigh(K)
    _, lap_eigvecs = np.linalg.eigh(laplacian)

    # Spectral alignment: average squared overlap between top-k eigenvectors
    k = min(10, n_nodes)
    alignment = 0.0
    for i in range(k):
        for j in range(k):
            overlap = float(np.abs(np.dot(ntk_eigvecs[:, -(i + 1)], lap_eigvecs[:, j])))
            alignment += overlap ** 2
    alignment /= k ** 2

    # Phase boundary prediction from spectral properties
    # gamma_star ~ C / (T * mu_max), where mu_max is modulated by spectral gap
    mu_max = float(ntk_eigvals[0]) / widths[0] if len(ntk_eigvals) > 0 else 1.0
    phase_boundary = _predict_gamma_star(mu_max * (1 + 0.5 * spectral_gap), 100)

    # Community detection via spectral clustering
    community_info: Dict[str, Any] = {}
    if n_communities == 0:
        # Auto-detect via largest eigengap
        eiggaps = np.diff(lap_eigvals[:min(20, len(lap_eigvals))])
        if len(eiggaps) > 0:
            n_communities = int(np.argmax(eiggaps)) + 1
            n_communities = max(1, min(n_communities, n_nodes // 2))

    if n_communities > 1:
        # Spectral clustering on Laplacian eigenvectors
        _, eigvecs = np.linalg.eigh(laplacian)
        embedding = eigvecs[:, :n_communities]

        # K-means-like assignment (simplified)
        rng = np.random.RandomState(seed)
        centroids = embedding[rng.choice(n_nodes, n_communities, replace=False)]
        assignments = np.zeros(n_nodes, dtype=int)

        for iteration in range(20):
            # Assign nodes to nearest centroid
            for i in range(n_nodes):
                dists = [np.linalg.norm(embedding[i] - centroids[c]) for c in range(n_communities)]
                assignments[i] = int(np.argmin(dists))

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(n_communities):
                members = embedding[assignments == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids, atol=1e-8):
                break
            centroids = new_centroids

        community_sizes = [int(np.sum(assignments == c)) for c in range(n_communities)]
        community_info["n_communities"] = n_communities
        community_info["sizes"] = community_sizes
        community_info["assignments"] = assignments.tolist()

        # Inter- vs intra-community edge ratio
        intra_edges = 0
        inter_edges = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j] > 0:
                    if assignments[i] == assignments[j]:
                        intra_edges += 1
                    else:
                        inter_edges += 1
        total_edges = intra_edges + inter_edges
        community_info["modularity"] = float(intra_edges - inter_edges) / (total_edges + 1e-12)
        community_info["phase_effect"] = (
            "Strong community structure may cause different phase behavior "
            "within vs. across communities"
            if community_info["modularity"] > 0.3
            else "Weak community structure; uniform phase behavior expected"
        )

    return SpectralReport(
        adjacency_eigenvalues=adj_eigvals,
        laplacian_eigenvalues=lap_eigvals,
        spectral_gap=spectral_gap,
        mixing_time=mixing_time,
        ntk_spectral_alignment=alignment,
        phase_boundary_from_spectrum=phase_boundary,
        community_structure=community_info,
    )
