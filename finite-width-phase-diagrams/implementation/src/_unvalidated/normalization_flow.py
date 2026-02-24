"""Phase analysis for normalizing flows.

Extends the phase diagram framework to normalizing flow models, analyzing
the interplay between invertibility, Jacobian conditioning, coupling layer
structure, and training dynamics. Normalizing flows have unique phase
behavior tied to the log-determinant penalty and volume preservation.

Example
-------
>>> from phase_diagrams.normalization_flow import flow_phase_diagram
>>> diagram = flow_phase_diagram(model, dataset)
>>> print(diagram.metadata["flow_type"])
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

class FlowType(str, Enum):
    """Type of normalizing flow architecture."""
    REALNVP = "realnvp"
    GLOW = "glow"
    NSF = "neural_spline_flow"
    RESIDUAL = "residual"
    CONTINUOUS = "continuous"


class StabilityClass(str, Enum):
    """Stability classification for invertibility."""
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"


@dataclass
class CouplingReport:
    """Analysis of coupling layer phase behavior.

    Attributes
    ----------
    n_coupling_layers : int
        Number of coupling layers in the flow.
    per_layer_jacobian_norm : NDArray
        Frobenius norm of Jacobian for each coupling layer.
    per_layer_condition : NDArray
        Condition number of Jacobian for each coupling layer.
    per_layer_regime : Dict[int, Regime]
        Phase regime at each coupling layer.
    log_det_mean : float
        Average log-determinant across coupling layers.
    log_det_std : float
        Std of log-determinant (measures training stability).
    volume_preservation_score : float
        How close to volume-preserving the flow is (1 = perfect).
    partition_analysis : Dict[str, Any]
        Analysis of the variable partition strategy.
    """
    n_coupling_layers: int = 0
    per_layer_jacobian_norm: NDArray = field(default_factory=lambda: np.array([]))
    per_layer_condition: NDArray = field(default_factory=lambda: np.array([]))
    per_layer_regime: Dict[int, Regime] = field(default_factory=dict)
    log_det_mean: float = 0.0
    log_det_std: float = 0.0
    volume_preservation_score: float = 0.0
    partition_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvertibilityReport:
    """Analysis of forward-inverse stability.

    Attributes
    ----------
    forward_condition : float
        Condition number of the forward Jacobian.
    inverse_condition : float
        Condition number of the inverse Jacobian.
    stability_class : StabilityClass
        Overall stability classification.
    reconstruction_error : float
        Mean ||x - f^{-1}(f(x))|| / ||x||.
    lipschitz_constant : float
        Estimated Lipschitz constant of the flow.
    inverse_lipschitz : float
        Estimated Lipschitz constant of the inverse.
    singular_value_spectrum : NDArray
        Singular values of the Jacobian.
    stability_margin : float
        Distance from instability boundary in parameter space.
    """
    forward_condition: float = 1.0
    inverse_condition: float = 1.0
    stability_class: StabilityClass = StabilityClass.STABLE
    reconstruction_error: float = 0.0
    lipschitz_constant: float = 1.0
    inverse_lipschitz: float = 1.0
    singular_value_spectrum: NDArray = field(default_factory=lambda: np.array([]))
    stability_margin: float = 0.0


@dataclass
class ConditioningReport:
    """Jacobian conditioning analysis for the full flow.

    Attributes
    ----------
    mean_condition : float
        Average condition number across data points.
    max_condition : float
        Worst-case condition number.
    condition_distribution : NDArray
        Histogram of condition numbers across data.
    log_det_distribution : NDArray
        Distribution of log-determinants across data.
    ill_conditioned_fraction : float
        Fraction of data points with condition > threshold.
    regime : Regime
        Phase regime from conditioning analysis.
    gradient_variance : float
        Variance of gradient norms (high = unstable training).
    optimal_regularization : float
        Suggested Jacobian regularization strength.
    """
    mean_condition: float = 1.0
    max_condition: float = 1.0
    condition_distribution: NDArray = field(default_factory=lambda: np.array([]))
    log_det_distribution: NDArray = field(default_factory=lambda: np.array([]))
    ill_conditioned_fraction: float = 0.0
    regime: Regime = Regime.LAZY
    gradient_variance: float = 0.0
    optimal_regularization: float = 0.0


@dataclass
class DepthScaling:
    """How phase behavior scales with flow depth.

    Attributes
    ----------
    depths : NDArray
        Array of depths evaluated.
    critical_lrs : NDArray
        Critical LR at each depth.
    jacobian_norms : NDArray
        Mean Jacobian norm at each depth.
    condition_numbers : NDArray
        Mean condition number at each depth.
    log_det_means : NDArray
        Mean log-determinant at each depth.
    scaling_exponent : float
        How critical LR scales with depth: eta* ~ L^{-alpha}.
    optimal_depth : int
        Depth that maximizes expressiveness while maintaining stability.
    per_depth_diagrams : Dict[int, PhaseDiagram]
        Full phase diagram at each depth.
    """
    depths: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    jacobian_norms: NDArray = field(default_factory=lambda: np.array([]))
    condition_numbers: NDArray = field(default_factory=lambda: np.array([]))
    log_det_means: NDArray = field(default_factory=lambda: np.array([]))
    scaling_exponent: float = 0.0
    optimal_depth: int = 1
    per_depth_diagrams: Dict[int, PhaseDiagram] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_flow_params(model: Any) -> Dict[str, Any]:
    """Extract normalizing flow parameters from model or config dict."""
    if isinstance(model, dict):
        return {
            "dim": model.get("dim", 16),
            "n_layers": model.get("n_layers", 8),
            "hidden_dim": model.get("hidden_dim", 64),
            "flow_type": FlowType(model.get("flow_type", "realnvp")),
            "init_scale": model.get("init_scale", 1.0),
        }
    if isinstance(model, (list, tuple)):
        dim = model[0].shape[-1] if model[0].ndim >= 2 else model[0].shape[0]
        return {
            "dim": dim,
            "n_layers": len(model),
            "hidden_dim": dim * 2,
            "flow_type": FlowType.REALNVP,
            "init_scale": float(np.std(model[0]) * math.sqrt(dim)),
        }
    raise TypeError(f"Unsupported model type: {type(model)}")


def _coupling_layer_forward(
    x: NDArray,
    mask: NDArray,
    scale_net_w: NDArray,
    shift_net_w: NDArray,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Forward pass through a single affine coupling layer.

    y_A = x_A
    y_B = x_B * exp(s(x_A)) + t(x_A)

    Returns (output, log_det_jacobian_per_sample).
    """
    x_A = x * mask
    x_B = x * (1 - mask)

    # Scale and shift networks (single hidden layer)
    rng = np.random.RandomState(seed)
    h_dim = scale_net_w.shape[0]
    h = np.maximum(x_A @ scale_net_w.T, 0)

    scale = h @ shift_net_w  # reuse weight for simplicity
    scale = np.tanh(scale) * 2  # clamp scale for stability
    shift = h @ rng.randn(h_dim, x.shape[1]) / math.sqrt(h_dim)

    y_B = x_B * np.exp(scale * (1 - mask)) + shift * (1 - mask)
    y = x_A + y_B

    log_det = np.sum(scale * (1 - mask), axis=1)
    return y, log_det


def _coupling_jacobian(
    x: NDArray,
    mask: NDArray,
    scale_net_w: NDArray,
    shift_net_w: NDArray,
) -> NDArray:
    """Compute Jacobian of a coupling layer at a single data point.

    Returns Jacobian matrix of shape (dim, dim).
    """
    dim = x.shape[0]
    J = np.eye(dim)
    eps = 1e-5

    x_batch = np.tile(x, (dim, 1))
    for i in range(dim):
        x_plus = x_batch[i].copy()
        x_plus[i] += eps
        x_minus = x_batch[i].copy()
        x_minus[i] -= eps
        y_plus, _ = _coupling_layer_forward(
            x_plus.reshape(1, -1), mask, scale_net_w, shift_net_w
        )
        y_minus, _ = _coupling_layer_forward(
            x_minus.reshape(1, -1), mask, scale_net_w, shift_net_w
        )
        J[:, i] = (y_plus.flatten() - y_minus.flatten()) / (2 * eps)

    return J


def _flow_ntk_eigenspectrum(
    dataset: NDArray,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    seed: int = 42,
) -> NDArray:
    """Approximate NTK eigenspectrum for a normalizing flow.

    The NTK for flows has contributions from both the scale and shift
    networks, modulated by the Jacobian structure. The log-det term
    adds a second-order contribution.
    """
    rng = np.random.RandomState(seed)
    n_samples = min(dataset.shape[0], 100)
    X = dataset[:n_samples]

    # Random feature approximation of the flow NTK
    h = X.copy()
    K = np.zeros((n_samples, n_samples))

    for l in range(n_layers):
        W_scale = rng.randn(hidden_dim, dim) / math.sqrt(dim)
        W_shift = rng.randn(hidden_dim, dim) / math.sqrt(hidden_dim)

        # Scale network contribution
        h_scale = np.maximum(h @ W_scale.T, 0)
        K += h_scale @ h_scale.T / hidden_dim

        # Shift network contribution
        h_shift = np.maximum(h @ W_shift.T, 0)
        K += h_shift @ h_shift.T / hidden_dim

        # Update features through coupling
        mask = np.zeros(dim)
        mask[:dim // 2] = 1.0 if l % 2 == 0 else 0.0
        mask[dim // 2:] = 0.0 if l % 2 == 0 else 1.0

        scale = np.tanh(h_scale @ W_shift) * 2
        h = h * mask + h * (1 - mask) * np.exp(scale * (1 - mask))

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    return eigvals


def _compute_flow_mu_max(
    dataset: NDArray,
    params: Dict[str, Any],
    seed: int = 42,
) -> float:
    """Effective mu_max for flow bifurcation analysis."""
    eigvals = _flow_ntk_eigenspectrum(
        dataset, params["dim"], params["n_layers"], params["hidden_dim"], seed
    )
    return float(eigvals[0]) / params["hidden_dim"] if len(eigvals) > 0 else 1.0


# ======================================================================
# Public API
# ======================================================================

def flow_phase_diagram(
    model: Any,
    dataset: NDArray,
    lr_range: Tuple[float, float] = (1e-5, 0.1),
    n_lr_steps: int = 30,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for a normalizing flow.

    Normalizing flows have unique phase behavior due to the log-determinant
    term in the loss, which penalizes large Jacobians and creates an
    additional contribution to the effective coupling. The phase boundary
    depends on the balance between likelihood fitting (pushes toward rich)
    and volume preservation (stabilizes lazy).

    Parameters
    ----------
    model : dict or list of NDArray
        Flow specification with keys ``{'dim', 'n_layers', 'hidden_dim',
        'flow_type', 'init_scale'}``.
    dataset : NDArray
        Training data of shape ``(n_samples, dim)``.
    lr_range : (float, float)
        Learning rate scan range.
    n_lr_steps : int
        Number of LR grid points.
    training_steps : int
        Assumed training duration T.
    seed : int
        Random seed.

    Returns
    -------
    PhaseDiagram
        Phase diagram with boundary curve.
    """
    params = _extract_flow_params(model)
    dim = params["dim"]
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    n_layers = params["n_layers"]

    mu_max = _compute_flow_mu_max(dataset, params, seed)

    # Log-det penalty correction: flows have an additional stabilizing term
    # that effectively reduces the coupling by a factor related to the
    # Jacobian spectral norm
    rng = np.random.RandomState(seed)
    sample = dataset[0] if dataset.shape[0] > 0 else rng.randn(dim)
    W_s = rng.randn(hidden_dim, dim) / math.sqrt(dim)
    W_t = rng.randn(hidden_dim, dim) / math.sqrt(hidden_dim)
    mask = np.zeros(dim)
    mask[:dim // 2] = 1.0
    J = _coupling_jacobian(sample, mask, W_s, W_t)
    spectral_norm = float(np.linalg.norm(J, 2))
    logdet_correction = 1.0 / (1.0 + 0.1 * spectral_norm)

    g_star = _predict_gamma_star(mu_max * logdet_correction, training_steps)

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    prev_regime = None
    for lr in lrs:
        gamma = _compute_gamma(lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            regime = Regime.LAZY
            confidence = min(1.0, (g_star - gamma) / g_star)
        elif gamma > g_star * 1.2:
            regime = Regime.RICH
            confidence = min(1.0, (gamma - g_star) / g_star)
        else:
            regime = Regime.CRITICAL
            confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

        ntk_drift = gamma * mu_max * logdet_correction * training_steps
        points.append(PhasePoint(
            lr=float(lr),
            width=hidden_dim,
            regime=regime,
            gamma=gamma,
            gamma_star=g_star,
            confidence=max(0.0, min(1.0, confidence)),
            ntk_drift_predicted=ntk_drift,
        ))

        if prev_regime is not None and prev_regime != regime:
            boundary_pts.append((float(lr), hidden_dim))
        prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    tc_vals = [training_steps * _compute_gamma(bp[0], init_scale, hidden_dim) for bp in boundary_pts]
    tc = float(np.mean(tc_vals)) if tc_vals else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(hidden_dim, hidden_dim),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": f"NormFlow-{params['flow_type'].value}",
            "dim": dim,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "logdet_correction": logdet_correction,
            "spectral_norm": spectral_norm,
        },
    )


def coupling_layer_analysis(
    model: Any,
    dataset: NDArray,
    training_steps: int = 100,
    seed: int = 42,
) -> CouplingReport:
    """Analyze phase behavior of individual coupling layers.

    Each coupling layer has its own Jacobian structure and contributes
    differently to the overall flow. Layers closer to the data tend to
    have larger Jacobians (more expressive) while deeper layers are
    closer to the base distribution (more regularized).

    Parameters
    ----------
    model : dict or list of NDArray
        Flow specification.
    dataset : NDArray
        Training data.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    CouplingReport
        Per-layer Jacobian analysis and regime classification.
    """
    params = _extract_flow_params(model)
    dim = params["dim"]
    n_layers = params["n_layers"]
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    rng = np.random.RandomState(seed)

    n_eval = min(20, dataset.shape[0])
    X = dataset[:n_eval]

    per_layer_jac_norm = np.zeros(n_layers)
    per_layer_cond = np.zeros(n_layers)
    per_layer_regime: Dict[int, Regime] = {}
    log_dets_all = []

    h = X.copy()
    for l in range(n_layers):
        mask = np.zeros(dim)
        if l % 2 == 0:
            mask[:dim // 2] = 1.0
        else:
            mask[dim // 2:] = 1.0

        W_s = rng.randn(hidden_dim, dim) / math.sqrt(dim)
        W_t = rng.randn(hidden_dim, dim) / math.sqrt(hidden_dim)

        # Compute Jacobian for each sample
        jac_norms = []
        cond_numbers = []
        log_dets_layer = []

        for i in range(n_eval):
            J = _coupling_jacobian(h[i], mask, W_s, W_t)
            jac_norms.append(float(np.linalg.norm(J, "fro")))
            svs = np.linalg.svd(J, compute_uv=False)
            cond = float(svs[0] / (svs[-1] + 1e-12))
            cond_numbers.append(cond)
            sign, logabsdet = np.linalg.slogdet(J)
            log_dets_layer.append(float(logabsdet) if sign > 0 else float(-logabsdet))

        per_layer_jac_norm[l] = float(np.mean(jac_norms))
        per_layer_cond[l] = float(np.mean(cond_numbers))
        log_dets_all.extend(log_dets_layer)

        # Regime classification based on Jacobian norm
        mean_jac = per_layer_jac_norm[l]
        if mean_jac < 1.2:
            per_layer_regime[l] = Regime.LAZY
        elif mean_jac > 3.0:
            per_layer_regime[l] = Regime.RICH
        else:
            per_layer_regime[l] = Regime.CRITICAL

        # Forward through coupling layer
        h, _ = _coupling_layer_forward(h, mask, W_s, W_t, seed + l)

    log_det_mean = float(np.mean(log_dets_all)) if log_dets_all else 0.0
    log_det_std = float(np.std(log_dets_all)) if log_dets_all else 0.0

    # Volume preservation: |mean(log_det)| close to 0 means volume-preserving
    vol_score = math.exp(-abs(log_det_mean))

    # Partition analysis
    partition_info: Dict[str, Any] = {
        "strategy": "alternating_halves",
        "n_partitions": 2,
        "partition_sizes": [dim // 2, dim - dim // 2],
        "effectiveness": min(1.0, 1.0 / (per_layer_cond.mean() + 1e-12)),
    }

    return CouplingReport(
        n_coupling_layers=n_layers,
        per_layer_jacobian_norm=per_layer_jac_norm,
        per_layer_condition=per_layer_cond,
        per_layer_regime=per_layer_regime,
        log_det_mean=log_det_mean,
        log_det_std=log_det_std,
        volume_preservation_score=vol_score,
        partition_analysis=partition_info,
    )


def invertibility_stability(
    model: Any,
    dataset: NDArray,
    n_eval: int = 50,
    seed: int = 42,
) -> InvertibilityReport:
    """Assess invertibility and numerical stability of the flow.

    A normalizing flow must be invertible, but numerical errors and
    ill-conditioning can make the inverse unreliable. This function
    measures the reconstruction error ||x - f^{-1}(f(x))||, Jacobian
    conditioning, and Lipschitz constants.

    Parameters
    ----------
    model : dict or list of NDArray
        Flow specification.
    dataset : NDArray
        Training data.
    n_eval : int
        Number of data points to evaluate.
    seed : int
        Random seed.

    Returns
    -------
    InvertibilityReport
        Stability classification and metrics.
    """
    params = _extract_flow_params(model)
    dim = params["dim"]
    n_layers = params["n_layers"]
    hidden_dim = params["hidden_dim"]
    rng = np.random.RandomState(seed)

    n_eval = min(n_eval, dataset.shape[0])
    X = dataset[:n_eval]

    # Forward pass through all layers
    h = X.copy()
    all_jacobians = []
    layer_params = []

    for l in range(n_layers):
        mask = np.zeros(dim)
        if l % 2 == 0:
            mask[:dim // 2] = 1.0
        else:
            mask[dim // 2:] = 1.0

        W_s = rng.randn(hidden_dim, dim) / math.sqrt(dim)
        W_t = rng.randn(hidden_dim, dim) / math.sqrt(hidden_dim)
        layer_params.append((mask, W_s, W_t))

        # Compute full Jacobian at first sample for spectrum
        J = _coupling_jacobian(h[0], mask, W_s, W_t)
        all_jacobians.append(J)

        h, _ = _coupling_layer_forward(h, mask, W_s, W_t, seed + l)

    z = h  # latent representation

    # Inverse pass: coupling layers are exactly invertible
    h_inv = z.copy()
    for l in range(n_layers - 1, -1, -1):
        mask, W_s, W_t = layer_params[l]
        # Inverse: x_B = (y_B - t(y_A)) * exp(-s(y_A))
        h_A = h_inv * mask
        h_scale = np.maximum(h_A @ W_s.T, 0)
        scale = np.tanh(h_scale @ W_t) * 2
        shift = h_scale @ rng.randn(hidden_dim, dim) / math.sqrt(hidden_dim)
        h_B = (h_inv * (1 - mask) - shift * (1 - mask)) * np.exp(-scale * (1 - mask))
        h_inv = h_A + h_B

    # Reconstruction error
    recon_error = float(np.mean(np.linalg.norm(X - h_inv, axis=1) / (np.linalg.norm(X, axis=1) + 1e-12)))

    # Compose Jacobians for full flow
    J_full = np.eye(dim)
    for J in all_jacobians:
        J_full = J @ J_full

    svs = np.linalg.svd(J_full, compute_uv=False)
    forward_cond = float(svs[0] / (svs[-1] + 1e-12))
    lipschitz = float(svs[0])
    inv_lipschitz = 1.0 / (svs[-1] + 1e-12)

    # Inverse Jacobian condition
    try:
        J_inv = np.linalg.inv(J_full)
        svs_inv = np.linalg.svd(J_inv, compute_uv=False)
        inverse_cond = float(svs_inv[0] / (svs_inv[-1] + 1e-12))
    except np.linalg.LinAlgError:
        inverse_cond = float("inf")

    # Stability classification
    if forward_cond < 100 and recon_error < 0.01:
        stability = StabilityClass.STABLE
    elif forward_cond < 1e4 and recon_error < 0.1:
        stability = StabilityClass.MARGINALLY_STABLE
    else:
        stability = StabilityClass.UNSTABLE

    # Stability margin: how far from instability
    stability_margin = max(0.0, 1.0 - math.log10(forward_cond + 1) / 6.0)

    return InvertibilityReport(
        forward_condition=forward_cond,
        inverse_condition=inverse_cond,
        stability_class=stability,
        reconstruction_error=recon_error,
        lipschitz_constant=lipschitz,
        inverse_lipschitz=inv_lipschitz,
        singular_value_spectrum=svs,
        stability_margin=stability_margin,
    )


def jacobian_conditioning(
    model: Any,
    dataset: NDArray,
    condition_threshold: float = 1e4,
    n_eval: int = 100,
    seed: int = 42,
) -> ConditioningReport:
    """Analyze Jacobian conditioning across the dataset.

    The Jacobian condition number determines both training stability
    and the quality of density estimation. Ill-conditioned Jacobians
    lead to high-variance gradient estimates and poor log-likelihood.

    Parameters
    ----------
    model : dict or list of NDArray
        Flow specification.
    dataset : NDArray
        Training data.
    condition_threshold : float
        Threshold above which a point is considered ill-conditioned.
    n_eval : int
        Number of data points to evaluate.
    seed : int
        Random seed.

    Returns
    -------
    ConditioningReport
        Distribution of condition numbers and regime analysis.
    """
    params = _extract_flow_params(model)
    dim = params["dim"]
    n_layers = params["n_layers"]
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    rng = np.random.RandomState(seed)

    n_eval = min(n_eval, dataset.shape[0])
    X = dataset[:n_eval]

    conditions = np.zeros(n_eval)
    log_dets = np.zeros(n_eval)
    gradient_norms = np.zeros(n_eval)

    for i in range(n_eval):
        x = X[i]
        J_full = np.eye(dim)
        rng_l = np.random.RandomState(seed)

        for l in range(n_layers):
            mask = np.zeros(dim)
            if l % 2 == 0:
                mask[:dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0

            W_s = rng_l.randn(hidden_dim, dim) / math.sqrt(dim)
            W_t = rng_l.randn(hidden_dim, dim) / math.sqrt(hidden_dim)

            J_l = _coupling_jacobian(x, mask, W_s, W_t)
            J_full = J_l @ J_full
            x_arr, _ = _coupling_layer_forward(x.reshape(1, -1), mask, W_s, W_t, seed + l)
            x = x_arr.flatten()

        svs = np.linalg.svd(J_full, compute_uv=False)
        conditions[i] = float(svs[0] / (svs[-1] + 1e-12))
        sign, logabsdet = np.linalg.slogdet(J_full)
        log_dets[i] = float(logabsdet)
        gradient_norms[i] = float(np.linalg.norm(J_full, "fro"))

    mean_cond = float(np.mean(conditions))
    max_cond = float(np.max(conditions))
    ill_frac = float(np.mean(conditions > condition_threshold))
    grad_var = float(np.var(gradient_norms))

    # Condition distribution (histogram)
    log_conditions = np.log10(conditions + 1)
    cond_hist, _ = np.histogram(log_conditions, bins=50)
    logdet_hist, _ = np.histogram(log_dets, bins=50)

    # Regime from conditioning
    if mean_cond < 10:
        regime = Regime.LAZY
    elif mean_cond > 1000:
        regime = Regime.RICH
    else:
        regime = Regime.CRITICAL

    # Optimal regularization: lambda ~ 1/condition_threshold
    optimal_reg = 1.0 / (mean_cond + 1e-12)

    return ConditioningReport(
        mean_condition=mean_cond,
        max_condition=max_cond,
        condition_distribution=cond_hist.astype(float),
        log_det_distribution=logdet_hist.astype(float),
        ill_conditioned_fraction=ill_frac,
        regime=regime,
        gradient_variance=grad_var,
        optimal_regularization=optimal_reg,
    )


def flow_depth_scaling(
    model: Any,
    dataset: NDArray,
    depths: Optional[Sequence[int]] = None,
    lr_range: Tuple[float, float] = (1e-5, 0.1),
    n_lr_steps: int = 20,
    training_steps: int = 100,
    seed: int = 42,
) -> DepthScaling:
    """Analyze how phase boundaries scale with flow depth.

    Deeper flows are more expressive but may become harder to train
    due to Jacobian accumulation. The critical LR typically decreases
    with depth, following a power law eta* ~ L^{-alpha}.

    Parameters
    ----------
    model : dict or list of NDArray
        Flow specification (depth will be varied).
    dataset : NDArray
        Training data.
    depths : sequence of int or None
        Depths to evaluate. If None, uses [2, 4, 8, 16, 32].
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
    DepthScaling
        Scaling behavior of phase boundaries with depth.
    """
    params = _extract_flow_params(model)
    dim = params["dim"]
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]

    if depths is None:
        depths = [2, 4, 8, 16, 32]
    depths_arr = np.array(sorted(depths))

    critical_lrs = np.zeros(len(depths_arr))
    jac_norms = np.zeros(len(depths_arr))
    cond_nums = np.zeros(len(depths_arr))
    logdet_means = np.zeros(len(depths_arr))
    per_depth_diagrams: Dict[int, PhaseDiagram] = {}

    rng = np.random.RandomState(seed)
    n_eval = min(10, dataset.shape[0])

    for idx, d in enumerate(depths_arr):
        d_int = int(d)
        model_d = {
            "dim": dim,
            "n_layers": d_int,
            "hidden_dim": hidden_dim,
            "flow_type": params["flow_type"].value,
            "init_scale": init_scale,
        }

        diagram = flow_phase_diagram(
            model_d, dataset, lr_range=lr_range,
            n_lr_steps=n_lr_steps, training_steps=training_steps, seed=seed,
        )
        per_depth_diagrams[d_int] = diagram

        # Extract critical LR
        critical_pts = [p for p in diagram.points if p.regime == Regime.CRITICAL]
        if critical_pts:
            critical_lrs[idx] = float(np.mean([p.lr for p in critical_pts]))
        else:
            lazy_lrs = [p.lr for p in diagram.points if p.regime == Regime.LAZY]
            rich_lrs = [p.lr for p in diagram.points if p.regime == Regime.RICH]
            if lazy_lrs and rich_lrs:
                critical_lrs[idx] = (max(lazy_lrs) + min(rich_lrs)) / 2.0
            else:
                critical_lrs[idx] = float(np.sqrt(lr_range[0] * lr_range[1]))

        # Jacobian analysis at this depth
        X = dataset[:n_eval]
        layer_jacs = []
        rng_d = np.random.RandomState(seed)
        x = X[0]
        J_full = np.eye(dim)

        for l in range(d_int):
            mask = np.zeros(dim)
            if l % 2 == 0:
                mask[:dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0
            W_s = rng_d.randn(hidden_dim, dim) / math.sqrt(dim)
            W_t = rng_d.randn(hidden_dim, dim) / math.sqrt(hidden_dim)
            J_l = _coupling_jacobian(x, mask, W_s, W_t)
            J_full = J_l @ J_full
            x_arr, _ = _coupling_layer_forward(x.reshape(1, -1), mask, W_s, W_t, seed + l)
            x = x_arr.flatten()

        jac_norms[idx] = float(np.linalg.norm(J_full, "fro"))
        svs = np.linalg.svd(J_full, compute_uv=False)
        cond_nums[idx] = float(svs[0] / (svs[-1] + 1e-12))
        sign, logabsdet = np.linalg.slogdet(J_full)
        logdet_means[idx] = float(logabsdet)

    # Fit power law: eta* ~ L^{-alpha}
    valid = critical_lrs > 1e-12
    if np.sum(valid) > 2:
        log_d = np.log(depths_arr[valid].astype(float))
        log_lr = np.log(critical_lrs[valid])
        coeffs = np.polyfit(log_d, log_lr, 1)
        scaling_exp = -float(coeffs[0])
    else:
        scaling_exp = 0.0

    # Optimal depth: maximize expressiveness (Jacobian norm) while
    # keeping condition number manageable
    score = jac_norms / (np.log(cond_nums + 1) + 1)
    optimal_idx = int(np.argmax(score))
    optimal_depth = int(depths_arr[optimal_idx])

    return DepthScaling(
        depths=depths_arr.astype(float),
        critical_lrs=critical_lrs,
        jacobian_norms=jac_norms,
        condition_numbers=cond_nums,
        log_det_means=logdet_means,
        scaling_exponent=scaling_exp,
        optimal_depth=optimal_depth,
        per_depth_diagrams=per_depth_diagrams,
    )
