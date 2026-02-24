"""Phase diagrams for self-supervised learning.

Extends the phase diagram framework to contrastive and non-contrastive SSL,
analyzing the role of temperature, projection head dimensionality,
augmentation strength, and representation collapse. SSL methods exhibit
unique phase behavior due to their reliance on data augmentation symmetries
rather than labels.

Example
-------
>>> from phase_diagrams.self_supervised import contrastive_phase_diagram
>>> diagram = contrastive_phase_diagram(model, dataset)
>>> print(diagram.metadata["temperature"])
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

class SSLMethod(str, Enum):
    """Self-supervised learning method."""
    SIMCLR = "simclr"
    BYOL = "byol"
    BARLOW_TWINS = "barlow_twins"
    VICREG = "vicreg"
    DINO = "dino"


class CollapseType(str, Enum):
    """Type of representation collapse."""
    NONE = "none"
    COMPLETE = "complete"
    DIMENSIONAL = "dimensional"
    CLUSTER = "cluster"


@dataclass
class TemperaturePhase:
    """Phase diagram over temperature parameter.

    Attributes
    ----------
    temperatures : NDArray
        Temperature values evaluated.
    critical_lrs : NDArray
        Critical LR at each temperature.
    regimes : Dict[float, Regime]
        Regime at each temperature (at default LR).
    optimal_temperature : float
        Temperature maximizing representation quality.
    collapse_temperature : float
        Temperature below which collapse occurs.
    uniformity : NDArray
        Uniformity of representations at each temperature.
    alignment : NDArray
        Alignment of positive pairs at each temperature.
    """
    temperatures: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[float, Regime] = field(default_factory=dict)
    optimal_temperature: float = 0.1
    collapse_temperature: float = 0.0
    uniformity: NDArray = field(default_factory=lambda: np.array([]))
    alignment: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class ProjectionPhase:
    """Phase analysis of projection head dimensionality.

    Attributes
    ----------
    dims : NDArray
        Projection dimensions evaluated.
    critical_lrs : NDArray
        Critical LR at each dimension.
    regimes : Dict[int, Regime]
        Regime at each dimension.
    optimal_dim : int
        Optimal projection dimension.
    collapse_dim : int
        Dimension below which collapse occurs.
    effective_rank : NDArray
        Effective rank of representations at each dimension.
    information_retained : NDArray
        Information retained from backbone at each dimension.
    """
    dims: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[int, Regime] = field(default_factory=dict)
    optimal_dim: int = 128
    collapse_dim: int = 1
    effective_rank: NDArray = field(default_factory=lambda: np.array([]))
    information_retained: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class AugPhase:
    """Phase analysis of augmentation strength.

    Attributes
    ----------
    strengths : NDArray
        Augmentation strengths evaluated (0 = no aug, 1 = max aug).
    critical_lrs : NDArray
        Critical LR at each strength.
    regimes : Dict[float, Regime]
        Regime at each strength.
    optimal_strength : float
        Augmentation strength maximizing feature learning.
    invariance_scores : NDArray
        How invariant representations are to augmentation.
    covariance_spectrum : NDArray
        Eigenvalues of representation covariance at each strength.
    """
    strengths: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[float, Regime] = field(default_factory=dict)
    optimal_strength: float = 0.5
    invariance_scores: NDArray = field(default_factory=lambda: np.array([]))
    covariance_spectrum: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class CollapsePrediction:
    """Prediction of representation collapse.

    Attributes
    ----------
    collapse_type : CollapseType
        Type of collapse predicted.
    collapse_probability : float
        Probability of collapse (0-1).
    effective_dimension : float
        Effective dimensionality of representations.
    uniformity_score : float
        Uniformity of representations on unit sphere.
    alignment_score : float
        Alignment between positive pairs.
    covariance_eigenvalues : NDArray
        Eigenvalues of the representation covariance matrix.
    critical_lr_for_collapse : float
        LR above which collapse occurs.
    prevention_recommendations : Dict[str, Any]
        Strategies to prevent collapse.
    """
    collapse_type: CollapseType = CollapseType.NONE
    collapse_probability: float = 0.0
    effective_dimension: float = 0.0
    uniformity_score: float = 0.0
    alignment_score: float = 0.0
    covariance_eigenvalues: NDArray = field(default_factory=lambda: np.array([]))
    critical_lr_for_collapse: float = float("inf")
    prevention_recommendations: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_ssl_params(model: Any) -> Dict[str, Any]:
    """Extract SSL model parameters."""
    if isinstance(model, dict):
        return {
            "backbone_dim": model.get("backbone_dim", 512),
            "projection_dim": model.get("projection_dim", 128),
            "hidden_dim": model.get("hidden_dim", 256),
            "depth": model.get("depth", 3),
            "init_scale": model.get("init_scale", 1.0),
            "temperature": model.get("temperature", 0.1),
            "method": SSLMethod(model.get("method", "simclr")),
        }
    if isinstance(model, (list, tuple)):
        backbone_dim = model[0].shape[1] if model[0].ndim == 2 else 512
        return {
            "backbone_dim": backbone_dim,
            "projection_dim": model[-1].shape[1] if model[-1].ndim == 2 else 128,
            "hidden_dim": backbone_dim,
            "depth": len(model),
            "init_scale": float(np.std(model[0]) * math.sqrt(backbone_dim)),
            "temperature": 0.1,
            "method": SSLMethod.SIMCLR,
        }
    raise TypeError(f"Unsupported model type: {type(model)}")


def _augment_data(
    data: NDArray,
    strength: float,
    seed: int = 42,
) -> NDArray:
    """Apply random augmentation to data.

    For feature vectors, augmentation = additive noise + masking.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(*data.shape) * strength * np.std(data)
    mask = (rng.rand(*data.shape) > strength * 0.5).astype(float)
    return data * mask + noise


def _contrastive_loss_gradient_norm(
    features: NDArray,
    temperature: float,
) -> float:
    """Compute gradient norm of InfoNCE loss at initialization.

    The gradient magnitude determines the effective coupling for SSL.
    """
    n = features.shape[0]
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    z = features / norms

    # Similarity matrix
    sim = z @ z.T / temperature
    sim -= np.max(sim, axis=1, keepdims=True)
    exp_sim = np.exp(sim)
    np.fill_diagonal(exp_sim, 0)

    # Gradient: softmax probabilities minus positive mask
    probs = exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-12)
    positive_mask = np.eye(n)  # simplified: self-supervised positive pairs
    grad = (probs - positive_mask) / (temperature * n)

    return float(np.linalg.norm(grad, "fro"))


def _compute_uniformity(features: NDArray) -> float:
    """Compute uniformity of features on the unit sphere.

    Uniformity = log(E[exp(-2||z_i - z_j||^2)])
    Lower is better (more uniform).
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    z = features / norms
    n = z.shape[0]

    sq_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((z[i] - z[j]) ** 2)
            sq_dists[i, j] = d
            sq_dists[j, i] = d

    triu_dists = sq_dists[np.triu_indices(n, k=1)]
    uniformity = float(np.log(np.mean(np.exp(-2 * triu_dists)) + 1e-12))
    return uniformity


def _compute_alignment(
    features1: NDArray,
    features2: NDArray,
) -> float:
    """Compute alignment between positive pairs.

    Alignment = E[||z_i - z_i^+||^2]
    Lower is better (positive pairs are closer).
    """
    n1 = np.linalg.norm(features1, axis=1, keepdims=True) + 1e-12
    n2 = np.linalg.norm(features2, axis=1, keepdims=True) + 1e-12
    z1 = features1 / n1
    z2 = features2 / n2
    return float(np.mean(np.sum((z1 - z2) ** 2, axis=1)))


def _ssl_ntk_eigenspectrum(
    features: NDArray,
    backbone_dim: int,
    projection_dim: int,
    depth: int,
    temperature: float,
    seed: int = 42,
) -> NDArray:
    """Approximate NTK eigenspectrum for SSL.

    The SSL NTK includes contributions from the contrastive loss gradient,
    which depends on temperature and the similarity structure of the batch.
    """
    rng = np.random.RandomState(seed)
    n = min(features.shape[0], 100)
    X = features[:n]

    K = np.zeros((n, n))
    h = X.copy()

    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = backbone_dim if l < depth - 1 else projection_dim
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        pre = h @ W
        if l < depth - 1:
            act_deriv = (pre > 0).astype(float)
            for i in range(n):
                for j in range(i, n):
                    val = np.dot(h[i], h[j]) * np.dot(act_deriv[i], act_deriv[j])
                    K[i, j] += val
                    K[j, i] = K[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K[i, j] += np.dot(h[i], h[j])
                    K[j, i] = K[i, j]
            h = pre

    # Temperature modifies the effective NTK by scaling gradients
    K *= 1.0 / (temperature + 1e-12)

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    return eigvals


def _compute_ssl_mu_max(
    features: NDArray,
    params: Dict[str, Any],
    seed: int = 42,
) -> float:
    """Effective mu_max for SSL bifurcation analysis."""
    eigvals = _ssl_ntk_eigenspectrum(
        features, params["backbone_dim"], params["projection_dim"],
        params["depth"], params["temperature"], seed,
    )
    return float(eigvals[0]) / params["backbone_dim"] if len(eigvals) > 0 else 1.0


def _backbone_forward(
    data: NDArray,
    backbone_dim: int,
    depth: int,
    seed: int = 42,
) -> NDArray:
    """Forward pass through backbone network."""
    rng = np.random.RandomState(seed)
    h = data
    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = backbone_dim
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        h = np.maximum(h @ W, 0) if l < depth - 1 else h @ W
    return h


# ======================================================================
# Public API
# ======================================================================

def contrastive_phase_diagram(
    model: Any,
    dataset: NDArray,
    lr_range: Tuple[float, float] = (1e-4, 0.1),
    n_lr_steps: int = 25,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for contrastive self-supervised learning.

    The SSL phase boundary depends on temperature, batch size, and
    augmentation strength. The effective coupling is modulated by the
    contrastive loss gradient, which scales inversely with temperature.

    Parameters
    ----------
    model : dict or list of NDArray
        SSL model specification with keys ``{'backbone_dim',
        'projection_dim', 'depth', 'temperature', 'method'}``.
    dataset : NDArray
        Unlabeled data of shape (n_samples, input_dim).
    lr_range : (float, float)
        Learning rate scan range.
    n_lr_steps : int
        Number of LR grid points.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    PhaseDiagram
        Phase diagram for SSL training.
    """
    params = _extract_ssl_params(model)
    backbone_dim = params["backbone_dim"]
    init_scale = params["init_scale"]
    temperature = params["temperature"]

    mu_max = _compute_ssl_mu_max(dataset, params, seed)

    # Temperature correction: lower temperature → stronger gradients → higher coupling
    temp_correction = 1.0 / (temperature + 1e-12)
    g_star = _predict_gamma_star(mu_max * min(temp_correction, 100.0), training_steps)

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    prev_regime = None
    for lr in lrs:
        gamma = _compute_gamma(lr, init_scale, backbone_dim)
        if gamma < g_star * 0.8:
            regime = Regime.LAZY
            confidence = min(1.0, (g_star - gamma) / g_star)
        elif gamma > g_star * 1.2:
            regime = Regime.RICH
            confidence = min(1.0, (gamma - g_star) / g_star)
        else:
            regime = Regime.CRITICAL
            confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

        ntk_drift = gamma * mu_max * min(temp_correction, 100.0) * training_steps
        points.append(PhasePoint(
            lr=float(lr),
            width=backbone_dim,
            regime=regime,
            gamma=gamma,
            gamma_star=g_star,
            confidence=max(0.0, min(1.0, confidence)),
            ntk_drift_predicted=ntk_drift,
        ))

        if prev_regime is not None and prev_regime != regime:
            boundary_pts.append((float(lr), backbone_dim))
        prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    tc_vals = [training_steps * _compute_gamma(bp[0], init_scale, backbone_dim)
               for bp in boundary_pts]
    tc = float(np.mean(tc_vals)) if tc_vals else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(backbone_dim, backbone_dim),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": f"SSL-{params['method'].value}",
            "backbone_dim": backbone_dim,
            "projection_dim": params["projection_dim"],
            "temperature": temperature,
            "depth": params["depth"],
        },
    )


def temperature_phase(
    model: Any,
    dataset: NDArray,
    temps: Optional[Sequence[float]] = None,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> TemperaturePhase:
    """Analyze the phase diagram as a function of temperature.

    Temperature controls the sharpness of the contrastive loss. Low
    temperature makes the loss focus on hard negatives (feature learning),
    while high temperature treats all negatives equally (lazy-like).

    Parameters
    ----------
    model : dict or list of NDArray
        SSL model specification.
    dataset : NDArray
        Unlabeled data.
    temps : sequence of float or None
        Temperature values. If None, uses logarithmic spacing.
    lr : float
        Reference learning rate.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    TemperaturePhase
        Phase analysis over temperature.
    """
    params = _extract_ssl_params(model)
    backbone_dim = params["backbone_dim"]
    init_scale = params["init_scale"]

    if temps is None:
        temps = np.logspace(-2, 1, 15)
    temps_arr = np.array(sorted(temps))

    critical_lrs = np.zeros(len(temps_arr))
    regimes: Dict[float, Regime] = {}
    uniformity_scores = np.zeros(len(temps_arr))
    alignment_scores = np.zeros(len(temps_arr))

    rng = np.random.RandomState(seed)
    optimal_temp = temps_arr[0]
    best_balance = -float("inf")
    collapse_temp = 0.0

    for idx, tau in enumerate(temps_arr):
        tau_float = float(tau)
        params_t = {**params, "temperature": tau_float}

        mu_max = _compute_ssl_mu_max(dataset, params_t, seed)
        temp_correction = min(1.0 / (tau_float + 1e-12), 100.0)
        g_star = _predict_gamma_star(mu_max * temp_correction, training_steps)
        critical_lr = g_star * backbone_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(lr, init_scale, backbone_dim)
        if gamma < g_star * 0.8:
            regimes[tau_float] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[tau_float] = Regime.RICH
        else:
            regimes[tau_float] = Regime.CRITICAL

        # Compute representation properties at this temperature
        features = _backbone_forward(dataset[:min(50, len(dataset))], backbone_dim, params["depth"], seed)
        uniformity_scores[idx] = _compute_uniformity(features)

        aug_features = _augment_data(dataset[:min(50, len(dataset))], 0.3, seed + 1)
        aug_rep = _backbone_forward(aug_features, backbone_dim, params["depth"], seed)
        alignment_scores[idx] = _compute_alignment(features, aug_rep)

        # Track optimal temperature (best uniformity-alignment trade-off)
        balance = -uniformity_scores[idx] - alignment_scores[idx]
        if balance > best_balance:
            best_balance = balance
            optimal_temp = tau_float

        # Collapse detection: very low alignment + very low uniformity
        if alignment_scores[idx] < 0.01 and uniformity_scores[idx] > -0.1 and collapse_temp == 0:
            collapse_temp = tau_float

    return TemperaturePhase(
        temperatures=temps_arr,
        critical_lrs=critical_lrs,
        regimes=regimes,
        optimal_temperature=optimal_temp,
        collapse_temperature=collapse_temp,
        uniformity=uniformity_scores,
        alignment=alignment_scores,
    )


def projection_head_phase(
    model: Any,
    dataset: NDArray,
    dims: Optional[Sequence[int]] = None,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> ProjectionPhase:
    """Analyze how projection head dimensionality affects phases.

    The projection head maps backbone representations to a space where
    contrastive learning occurs. Its dimensionality affects the effective
    coupling and can determine whether the backbone features are learned
    in the lazy or rich regime.

    Parameters
    ----------
    model : dict or list of NDArray
        SSL model specification.
    dataset : NDArray
        Unlabeled data.
    dims : sequence of int or None
        Projection dimensions to evaluate.
    lr : float
        Reference learning rate.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    ProjectionPhase
        Analysis of projection dimension effects.
    """
    params = _extract_ssl_params(model)
    backbone_dim = params["backbone_dim"]
    init_scale = params["init_scale"]

    if dims is None:
        dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    dims_arr = np.array(sorted(dims))

    critical_lrs = np.zeros(len(dims_arr))
    regimes: Dict[int, Regime] = {}
    eff_ranks = np.zeros(len(dims_arr))
    info_retained = np.zeros(len(dims_arr))

    optimal_dim = int(dims_arr[0])
    collapse_dim = 1
    best_score = -float("inf")

    features = _backbone_forward(dataset[:min(50, len(dataset))], backbone_dim, params["depth"], seed)
    rng = np.random.RandomState(seed)

    for idx, d in enumerate(dims_arr):
        d_int = int(d)
        params_d = {**params, "projection_dim": d_int}

        mu_max = _compute_ssl_mu_max(dataset, params_d, seed)
        # Projection dimension modulates coupling: larger projection → more parameters → higher coupling
        dim_correction = math.sqrt(d_int / backbone_dim)
        g_star = _predict_gamma_star(mu_max * dim_correction, training_steps)
        critical_lr = g_star * backbone_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(lr, init_scale, backbone_dim)
        if gamma < g_star * 0.8:
            regimes[d_int] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[d_int] = Regime.RICH
        else:
            regimes[d_int] = Regime.CRITICAL

        # Project features to evaluate rank and information retention
        W_proj = rng.randn(backbone_dim, d_int) / math.sqrt(backbone_dim)
        projected = features @ W_proj

        # Effective rank
        cov = projected.T @ projected / projected.shape[0]
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 0)
        total = np.sum(eigvals) + 1e-12
        eff_ranks[idx] = float(total / (eigvals[0] + 1e-12))

        # Information retained: fraction of backbone variance preserved
        backbone_var = float(np.sum(np.var(features, axis=0)))
        proj_var = float(np.sum(np.var(projected, axis=0)))
        info_retained[idx] = min(1.0, proj_var / (backbone_var + 1e-12))

        # Track best dimension
        score = eff_ranks[idx] * info_retained[idx]
        if score > best_score:
            best_score = score
            optimal_dim = d_int

        # Collapse detection
        if eff_ranks[idx] < 2.0 and collapse_dim == 1:
            collapse_dim = d_int

    return ProjectionPhase(
        dims=dims_arr.astype(float),
        critical_lrs=critical_lrs,
        regimes=regimes,
        optimal_dim=optimal_dim,
        collapse_dim=collapse_dim,
        effective_rank=eff_ranks,
        information_retained=info_retained,
    )


def augmentation_strength_phase(
    model: Any,
    dataset: NDArray,
    strengths: Optional[Sequence[float]] = None,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> AugPhase:
    """Analyze how augmentation strength affects the phase diagram.

    Augmentation creates the positive pairs that drive SSL learning.
    Too-weak augmentation provides no learning signal (lazy), while
    too-strong augmentation destroys information (collapse). The
    optimal strength maximizes feature learning.

    Parameters
    ----------
    model : dict or list of NDArray
        SSL model specification.
    dataset : NDArray
        Unlabeled data.
    strengths : sequence of float or None
        Augmentation strengths (0 to 1).
    lr : float
        Reference learning rate.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    AugPhase
        Phase analysis over augmentation strength.
    """
    params = _extract_ssl_params(model)
    backbone_dim = params["backbone_dim"]
    init_scale = params["init_scale"]

    if strengths is None:
        strengths = np.linspace(0.0, 1.0, 15)
    strengths_arr = np.array(sorted(strengths))

    critical_lrs = np.zeros(len(strengths_arr))
    regimes: Dict[float, Regime] = {}
    invariance = np.zeros(len(strengths_arr))
    cov_spectra = []

    optimal_strength = 0.5
    best_score = -float("inf")

    for idx, s in enumerate(strengths_arr):
        s_float = float(s)

        # Augmented data modifies the effective NTK
        aug1 = _augment_data(dataset, s_float, seed)
        aug2 = _augment_data(dataset, s_float, seed + 1)

        # The contrastive gradient depends on similarity between augmented views
        features1 = _backbone_forward(aug1[:min(50, len(aug1))], backbone_dim, params["depth"], seed)
        features2 = _backbone_forward(aug2[:min(50, len(aug2))], backbone_dim, params["depth"], seed)

        # Invariance: how similar are representations of augmented views
        align = _compute_alignment(features1, features2)
        invariance[idx] = 1.0 / (1.0 + align)  # higher = more invariant

        # Augmentation modifies effective coupling:
        # stronger aug → more gradient signal → higher effective coupling
        aug_correction = 1.0 + s_float * 3.0

        mu_max = _compute_ssl_mu_max(aug1, params, seed)
        g_star = _predict_gamma_star(mu_max * aug_correction, training_steps)
        critical_lr = g_star * backbone_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(lr, init_scale, backbone_dim)
        if gamma < g_star * 0.8:
            regimes[s_float] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[s_float] = Regime.RICH
        else:
            regimes[s_float] = Regime.CRITICAL

        # Covariance spectrum of representations
        cov = features1.T @ features1 / features1.shape[0]
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        cov_spectra.append(eigvals)

        # Track optimal strength (maximize invariance while maintaining rank)
        eff_rank = float(np.sum(eigvals) / (eigvals[0] + 1e-12))
        score = invariance[idx] * min(1.0, eff_rank / 10.0)
        if score > best_score:
            best_score = score
            optimal_strength = s_float

    # Pad spectra to same length
    max_len = max(len(s) for s in cov_spectra) if cov_spectra else 0
    cov_arr = np.zeros((len(strengths_arr), max_len))
    for i, spec in enumerate(cov_spectra):
        cov_arr[i, :len(spec)] = spec

    return AugPhase(
        strengths=strengths_arr,
        critical_lrs=critical_lrs,
        regimes=regimes,
        optimal_strength=optimal_strength,
        invariance_scores=invariance,
        covariance_spectrum=cov_arr,
    )


def collapse_prediction(
    model: Any,
    dataset: NDArray,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> CollapsePrediction:
    """Predict whether SSL training will suffer from representation collapse.

    Collapse occurs when representations become constant or low-rank.
    This is related to the phase diagram: complete collapse corresponds
    to a degenerate lazy regime where no useful features are learned.

    Parameters
    ----------
    model : dict or list of NDArray
        SSL model specification.
    dataset : NDArray
        Unlabeled data.
    lr : float
        Training learning rate.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    CollapsePrediction
        Collapse analysis and prevention recommendations.
    """
    params = _extract_ssl_params(model)
    backbone_dim = params["backbone_dim"]
    projection_dim = params["projection_dim"]
    init_scale = params["init_scale"]
    temperature = params["temperature"]

    # Compute representations
    n = min(100, dataset.shape[0])
    features = _backbone_forward(dataset[:n], backbone_dim, params["depth"], seed)

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    z = features / norms

    # Uniformity
    uniformity = _compute_uniformity(z)

    # Alignment with augmented views
    aug_data = _augment_data(dataset[:n], 0.3, seed + 1)
    aug_features = _backbone_forward(aug_data, backbone_dim, params["depth"], seed)
    alignment = _compute_alignment(features, aug_features)

    # Covariance analysis
    cov = features.T @ features / n
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0)

    # Effective dimension
    total_var = np.sum(eigvals)
    if total_var > 1e-12:
        normalized_eigvals = eigvals / total_var
        eff_dim = float(np.exp(-np.sum(
            normalized_eigvals * np.log(normalized_eigvals + 1e-12)
        )))
    else:
        eff_dim = 0.0

    # Collapse classification
    if eff_dim < 2.0:
        collapse_type = CollapseType.COMPLETE
        collapse_prob = 0.9
    elif eff_dim < backbone_dim * 0.1:
        collapse_type = CollapseType.DIMENSIONAL
        collapse_prob = 0.6
    elif uniformity > -0.5:
        # Poor uniformity suggests cluster collapse
        collapse_type = CollapseType.CLUSTER
        collapse_prob = 0.4
    else:
        collapse_type = CollapseType.NONE
        collapse_prob = 0.1

    # Critical LR for collapse
    mu_max = _compute_ssl_mu_max(dataset, params, seed)
    temp_correction = min(1.0 / (temperature + 1e-12), 100.0)
    g_star = _predict_gamma_star(mu_max * temp_correction, training_steps)
    # Collapse occurs when the training is too aggressive (very rich regime)
    # or too weak (very lazy regime with no momentum to escape)
    critical_collapse_lr = g_star * backbone_dim * 5.0 / (init_scale ** 2 + 1e-12)

    # Prevention recommendations
    recommendations: Dict[str, Any] = {}
    if collapse_type != CollapseType.NONE:
        if temperature > 0.5:
            recommendations["temperature"] = {
                "action": "decrease",
                "target": 0.1,
                "reason": "High temperature weakens contrastive signal",
            }
        if projection_dim < backbone_dim // 4:
            recommendations["projection_dim"] = {
                "action": "increase",
                "target": backbone_dim // 2,
                "reason": "Small projection dimension limits representation capacity",
            }
        if params["method"] == SSLMethod.SIMCLR:
            recommendations["method"] = {
                "action": "consider_alternatives",
                "options": ["barlow_twins", "vicreg"],
                "reason": "Non-contrastive methods have explicit anti-collapse terms",
            }
        recommendations["regularization"] = {
            "action": "add",
            "options": ["variance_regularization", "covariance_regularization"],
            "reason": "Explicit regularization prevents dimensional collapse",
        }
        if lr > critical_collapse_lr:
            recommendations["learning_rate"] = {
                "action": "decrease",
                "target": critical_collapse_lr * 0.5,
                "reason": "LR too high, causing training instability",
            }
    else:
        recommendations["status"] = "No collapse risk detected"

    return CollapsePrediction(
        collapse_type=collapse_type,
        collapse_probability=collapse_prob,
        effective_dimension=eff_dim,
        uniformity_score=uniformity,
        alignment_score=alignment,
        covariance_eigenvalues=eigvals,
        critical_lr_for_collapse=critical_collapse_lr,
        prevention_recommendations=recommendations,
    )
