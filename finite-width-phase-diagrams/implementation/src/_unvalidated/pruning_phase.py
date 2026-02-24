"""Phase transitions in neural network pruning.

Analyzes phase transitions that occur during pruning: there exists a
critical sparsity level below which pruning preserves the training regime
(lazy or rich) and above which the network undergoes a phase transition
to a qualitatively different regime. This is connected to the lottery
ticket hypothesis and structured vs. unstructured pruning strategies.

Example
-------
>>> from phase_diagrams.pruning_phase import pruning_phase_diagram
>>> diagram = pruning_phase_diagram(model, dataset, sparsities=[0.1, 0.5, 0.9])
>>> print(diagram.metadata["critical_sparsity"])
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

class PruningStrategy(str, Enum):
    """Pruning strategy type."""
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    GRADIENT = "gradient"
    SNIP = "snip"
    GRASP = "grasp"
    SYNFLOW = "synflow"


@dataclass
class LotteryPhase:
    """Phase analysis of lottery ticket behavior.

    Attributes
    ----------
    winning_ticket_sparsity : float
        Maximum sparsity at which a winning ticket exists.
    ticket_quality : NDArray
        Quality score of tickets at each sparsity level.
    sparsities : NDArray
        Sparsity levels evaluated.
    regime_before_pruning : Regime
        Phase regime of the unpruned model.
    regime_at_winning_sparsity : Regime
        Phase regime at the winning ticket sparsity.
    ntk_alignment : NDArray
        Alignment between pruned and unpruned NTK at each sparsity.
    rewind_quality : NDArray
        Quality of weight rewinding at each sparsity.
    iterative_vs_oneshot : Dict[str, float]
        Comparison of iterative vs. one-shot pruning.
    """
    winning_ticket_sparsity: float = 0.0
    ticket_quality: NDArray = field(default_factory=lambda: np.array([]))
    sparsities: NDArray = field(default_factory=lambda: np.array([]))
    regime_before_pruning: Regime = Regime.LAZY
    regime_at_winning_sparsity: Regime = Regime.LAZY
    ntk_alignment: NDArray = field(default_factory=lambda: np.array([]))
    rewind_quality: NDArray = field(default_factory=lambda: np.array([]))
    iterative_vs_oneshot: Dict[str, float] = field(default_factory=dict)


@dataclass
class StructuredPhase:
    """Comparison of structured vs. unstructured pruning phases.

    Attributes
    ----------
    unstructured_critical_sparsity : float
        Critical sparsity for unstructured (weight-level) pruning.
    structured_critical_sparsity : float
        Critical sparsity for structured (filter/channel) pruning.
    unstructured_regimes : Dict[float, Regime]
        Regime at each sparsity for unstructured pruning.
    structured_regimes : Dict[float, Regime]
        Regime at each sparsity for structured pruning.
    effective_width_reduction : NDArray
        Effective width after structured pruning at each sparsity.
    ntk_preservation : Dict[str, NDArray]
        How well each strategy preserves the NTK.
    recommended_strategy : PruningStrategy
        Recommended pruning strategy for phase preservation.
    explanation : str
        Human-readable comparison.
    """
    unstructured_critical_sparsity: float = 0.0
    structured_critical_sparsity: float = 0.0
    unstructured_regimes: Dict[float, Regime] = field(default_factory=dict)
    structured_regimes: Dict[float, Regime] = field(default_factory=dict)
    effective_width_reduction: NDArray = field(default_factory=lambda: np.array([]))
    ntk_preservation: Dict[str, NDArray] = field(default_factory=dict)
    recommended_strategy: PruningStrategy = PruningStrategy.MAGNITUDE
    explanation: str = ""


@dataclass
class InitPruningPhase:
    """Phase analysis of pruning at initialization.

    Attributes
    ----------
    strategy_rankings : Dict[str, float]
        Score for each pruning-at-init strategy.
    best_strategy : PruningStrategy
        Best strategy for preserving training dynamics.
    critical_sparsities : Dict[str, float]
        Critical sparsity for each strategy.
    ntk_preservation_scores : Dict[str, NDArray]
        NTK preservation at each sparsity per strategy.
    gradient_flow_scores : Dict[str, NDArray]
        Gradient flow quality at each sparsity per strategy.
    phase_regime_map : Dict[str, Dict[float, Regime]]
        Phase regime map for each strategy.
    """
    strategy_rankings: Dict[str, float] = field(default_factory=dict)
    best_strategy: PruningStrategy = PruningStrategy.SYNFLOW
    critical_sparsities: Dict[str, float] = field(default_factory=dict)
    ntk_preservation_scores: Dict[str, NDArray] = field(default_factory=dict)
    gradient_flow_scores: Dict[str, NDArray] = field(default_factory=dict)
    phase_regime_map: Dict[str, Dict[float, Regime]] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_pruning_params(model: Any) -> Tuple[List[NDArray], Dict[str, Any]]:
    """Extract weights and architecture params for pruning analysis."""
    if isinstance(model, dict):
        input_dim = model.get("input_dim", 784)
        hidden_dim = model.get("hidden_dim", 256)
        output_dim = model.get("output_dim", 10)
        depth = model.get("depth", 3)
        init_scale = model.get("init_scale", 1.0)

        rng = np.random.RandomState(model.get("seed", 42))
        weights = []
        dims = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        for l in range(depth):
            W = rng.randn(dims[l], dims[l + 1]) * init_scale / math.sqrt(dims[l])
            weights.append(W)

        params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "depth": depth,
            "init_scale": init_scale,
        }
        return weights, params

    if isinstance(model, (list, tuple)):
        weights = [np.asarray(w, dtype=np.float64) for w in model]
        input_dim = weights[0].shape[0] if weights[0].ndim == 2 else weights[0].shape[1]
        hidden_dim = weights[0].shape[1] if weights[0].ndim == 2 else weights[0].shape[0]
        output_dim = weights[-1].shape[1] if weights[-1].ndim == 2 else weights[-1].shape[0]
        init_scale = float(np.std(weights[0]) * math.sqrt(hidden_dim))
        params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "depth": len(weights),
            "init_scale": init_scale,
        }
        return weights, params

    raise TypeError(f"Unsupported model type: {type(model)}")


def _prune_weights_magnitude(
    weights: List[NDArray],
    sparsity: float,
) -> Tuple[List[NDArray], List[NDArray]]:
    """Prune weights by magnitude (global threshold).

    Returns (pruned_weights, masks) where masks are binary.
    """
    all_params = np.concatenate([w.flatten() for w in weights])
    threshold = np.percentile(np.abs(all_params), sparsity * 100)

    pruned = []
    masks = []
    for w in weights:
        mask = (np.abs(w) >= threshold).astype(float)
        pruned.append(w * mask)
        masks.append(mask)

    return pruned, masks


def _prune_weights_random(
    weights: List[NDArray],
    sparsity: float,
    seed: int = 42,
) -> Tuple[List[NDArray], List[NDArray]]:
    """Prune weights randomly at the specified sparsity."""
    rng = np.random.RandomState(seed)
    pruned = []
    masks = []
    for w in weights:
        mask = (rng.rand(*w.shape) >= sparsity).astype(float)
        pruned.append(w * mask)
        masks.append(mask)
    return pruned, masks


def _prune_structured(
    weights: List[NDArray],
    sparsity: float,
) -> Tuple[List[NDArray], NDArray]:
    """Structured pruning: remove entire columns (neurons) by norm.

    Returns (pruned_weights, effective_widths_per_layer).
    """
    pruned = []
    effective_widths = np.zeros(len(weights))

    for l, w in enumerate(weights):
        if w.ndim < 2:
            pruned.append(w)
            effective_widths[l] = w.shape[0]
            continue

        # Column norms (each column = one output neuron)
        col_norms = np.linalg.norm(w, axis=0)
        n_keep = max(1, int(w.shape[1] * (1 - sparsity)))
        keep_indices = np.argsort(col_norms)[-n_keep:]

        pruned_w = np.zeros_like(w)
        pruned_w[:, keep_indices] = w[:, keep_indices]
        pruned.append(pruned_w)
        effective_widths[l] = n_keep

    return pruned, effective_widths


def _pruned_ntk_eigenspectrum(
    weights: List[NDArray],
    masks: List[NDArray],
    dataset: NDArray,
    n_samples: int = 50,
    seed: int = 42,
) -> NDArray:
    """Compute NTK eigenspectrum of the pruned network.

    The pruned NTK is K_pruned(x,x') = sum_l (mask_l * J_l(x))^T (mask_l * J_l(x'))
    where J_l is the per-layer Jacobian.
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, dataset.shape[0])
    X = dataset[:n]

    depth = len(weights)
    K = np.zeros((n, n))

    h = X.copy()
    for l in range(depth):
        w = weights[l] * masks[l]
        pre = h @ w
        if l < depth - 1:
            act_deriv = (pre > 0).astype(float)
            # Per-layer NTK: K_l(i,j) = h_i^T h_j * act_deriv_i . act_deriv_j
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

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    return eigvals


def _ntk_alignment(eigvals_a: NDArray, eigvals_b: NDArray) -> float:
    """Compute alignment between two NTK spectra (cosine similarity)."""
    min_len = min(len(eigvals_a), len(eigvals_b))
    a = eigvals_a[:min_len]
    b = eigvals_b[:min_len]
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (norm_a * norm_b))


def _gradient_flow_score(
    weights: List[NDArray],
    masks: Optional[List[NDArray]] = None,
) -> float:
    """Measure gradient flow quality through the network.

    Uses the product of spectral norms of weight matrices (with masks applied)
    as a proxy for gradient flow. Values close to 1 indicate healthy flow.
    """
    product = 1.0
    for l, w in enumerate(weights):
        if masks is not None:
            w = w * masks[l]
        if w.ndim >= 2:
            s = np.linalg.svd(w, compute_uv=False)
            product *= float(s[0]) if len(s) > 0 else 0.0
        else:
            product *= float(np.max(np.abs(w))) if w.size > 0 else 0.0

    # Normalize by depth to get per-layer average
    depth = len(weights)
    flow = product ** (1.0 / depth) if depth > 0 else 0.0
    return flow


def _snip_scores(
    weights: List[NDArray],
    dataset: NDArray,
    seed: int = 42,
) -> List[NDArray]:
    """Compute SNIP (connection sensitivity) scores.

    SNIP scores approximate |dL/dw * w| for each weight, measuring
    how important each connection is for the loss at initialization.
    """
    rng = np.random.RandomState(seed)
    n = min(32, dataset.shape[0])
    X = dataset[:n]
    depth = len(weights)

    # Forward pass to get activations
    activations = [X]
    h = X
    for l in range(depth):
        pre = h @ weights[l]
        h = np.maximum(pre, 0) if l < depth - 1 else pre
        activations.append(h)

    # Random targets for gradient
    Y = rng.randn(*activations[-1].shape)
    grad_output = 2 * (activations[-1] - Y) / n

    # Backward pass
    scores = []
    delta = grad_output
    for l in range(depth - 1, -1, -1):
        grad_w = activations[l].T @ delta
        # SNIP score: |grad_w * w|
        score = np.abs(grad_w * weights[l])
        scores.insert(0, score)
        if l > 0:
            delta = delta @ weights[l].T
            delta = delta * (activations[l] > 0).astype(float)

    return scores


def _synflow_scores(weights: List[NDArray]) -> List[NDArray]:
    """Compute SynFlow (synaptic flow) scores.

    SynFlow iteratively prunes based on the product of absolute
    weight values along paths through the network, which preserves
    gradient flow.
    """
    # SynFlow score: product of |w| along each path
    # Approximate by |w| * (product of spectral norms of adjacent layers)
    depth = len(weights)
    spectral_norms = []
    for w in weights:
        if w.ndim >= 2:
            s = np.linalg.svd(w, compute_uv=False)
            spectral_norms.append(float(s[0]))
        else:
            spectral_norms.append(float(np.max(np.abs(w))))

    scores = []
    for l in range(depth):
        # Score proportional to weight magnitude times adjacent layer norms
        path_product = 1.0
        for k in range(depth):
            if k != l:
                path_product *= spectral_norms[k]
        score = np.abs(weights[l]) * path_product
        scores.append(score)

    return scores


# ======================================================================
# Public API
# ======================================================================

def pruning_phase_diagram(
    model: Any,
    dataset: NDArray,
    sparsities: Sequence[float],
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    n_lr_steps: int = 25,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram over learning rate and sparsity.

    Pruning modifies the effective width and NTK structure of the
    network. At a critical sparsity, the NTK spectrum changes
    qualitatively, shifting the phase boundary.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification or weight matrices.
    dataset : NDArray
        Input data of shape (n_samples, input_dim).
    sparsities : sequence of float
        Sparsity levels to evaluate (0 to 1).
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
        Phase diagram with sparsity as the width axis.
    """
    weights, params = _extract_pruning_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    depth = params["depth"]

    sparsities_arr = np.array(sorted(sparsities))
    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)

    # Unpruned NTK for reference
    identity_masks = [np.ones_like(w) for w in weights]
    unpruned_eigvals = _pruned_ntk_eigenspectrum(weights, identity_masks, dataset, seed=seed)

    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []
    crit_sparsity = 0.0

    for sparsity in sparsities_arr:
        pruned_w, masks = _prune_weights_magnitude(weights, sparsity)

        # Effective width after pruning
        remaining_frac = 1.0 - sparsity
        effective_width = max(1, int(hidden_dim * remaining_frac))

        eigvals = _pruned_ntk_eigenspectrum(pruned_w, masks, dataset, seed=seed)
        mu_max = float(eigvals[0]) / effective_width if len(eigvals) > 0 else 1.0

        # Sparsity correction: pruning concentrates the NTK, which can
        # either increase or decrease mu_max depending on which connections remain
        alignment = _ntk_alignment(unpruned_eigvals, eigvals)
        sparsity_correction = alignment * (1.0 + sparsity * 0.5)

        g_star = _predict_gamma_star(mu_max * sparsity_correction, training_steps)

        prev_regime = None
        for lr in lrs:
            gamma = _compute_gamma(lr, init_scale, effective_width)
            if gamma < g_star * 0.8:
                regime = Regime.LAZY
                confidence = min(1.0, (g_star - gamma) / g_star)
            elif gamma > g_star * 1.2:
                regime = Regime.RICH
                confidence = min(1.0, (gamma - g_star) / g_star)
            else:
                regime = Regime.CRITICAL
                confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

            ntk_drift = gamma * mu_max * sparsity_correction * training_steps
            points.append(PhasePoint(
                lr=float(lr),
                width=effective_width,
                regime=regime,
                gamma=gamma,
                gamma_star=g_star,
                confidence=max(0.0, min(1.0, confidence)),
                ntk_drift_predicted=ntk_drift,
            ))

            if prev_regime is not None and prev_regime != regime:
                boundary_pts.append((float(lr), effective_width))
            prev_regime = regime

    # Find critical sparsity where the boundary shifts significantly
    for i in range(len(sparsities_arr) - 1):
        s1 = sparsities_arr[i]
        s2 = sparsities_arr[i + 1]
        ew1 = max(1, int(hidden_dim * (1 - s1)))
        ew2 = max(1, int(hidden_dim * (1 - s2)))
        pts1 = [p for p in points if p.width == ew1]
        pts2 = [p for p in points if p.width == ew2]
        if pts1 and pts2:
            g1 = np.mean([p.gamma_star for p in pts1])
            g2 = np.mean([p.gamma_star for p in pts2])
            if abs(g2 - g1) / (g1 + 1e-12) > 0.5:
                crit_sparsity = float(s1 + s2) / 2
                break

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    tc_vals = []
    for bp in boundary_pts:
        g = _compute_gamma(bp[0], init_scale, bp[1])
        tc_vals.append(training_steps * g)
    tc = float(np.mean(tc_vals)) if tc_vals else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(
            max(1, int(hidden_dim * (1 - max(sparsities_arr)))),
            hidden_dim,
        ),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": "PrunedMLP",
            "hidden_dim": hidden_dim,
            "depth": depth,
            "sparsities": sparsities_arr.tolist(),
            "critical_sparsity": crit_sparsity,
        },
    )


def critical_sparsity(
    model: Any,
    dataset: NDArray,
    sparsity_resolution: int = 50,
    training_steps: int = 100,
    seed: int = 42,
) -> float:
    """Find the critical sparsity level where a phase transition occurs.

    Binary-searches for the sparsity at which the NTK alignment with the
    unpruned network drops below a threshold, indicating a qualitative
    change in training dynamics.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    sparsity_resolution : int
        Number of sparsity levels to scan initially.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    float
        Critical sparsity level (0 to 1).
    """
    weights, params = _extract_pruning_params(model)
    hidden_dim = params["hidden_dim"]

    identity_masks = [np.ones_like(w) for w in weights]
    unpruned_eigvals = _pruned_ntk_eigenspectrum(weights, identity_masks, dataset, seed=seed)

    # Coarse scan
    sparsities = np.linspace(0.0, 0.99, sparsity_resolution)
    alignments = np.zeros(len(sparsities))

    for idx, s in enumerate(sparsities):
        pruned_w, masks = _prune_weights_magnitude(weights, s)
        eigvals = _pruned_ntk_eigenspectrum(pruned_w, masks, dataset, seed=seed)
        alignments[idx] = _ntk_alignment(unpruned_eigvals, eigvals)

    # Find where alignment drops below threshold (0.5)
    threshold = 0.5
    for idx in range(len(alignments) - 1):
        if alignments[idx] >= threshold and alignments[idx + 1] < threshold:
            # Linear interpolation
            t = (threshold - alignments[idx + 1]) / (alignments[idx] - alignments[idx + 1] + 1e-12)
            crit = float(sparsities[idx + 1] * (1 - t) + sparsities[idx] * t)
            return crit

    # No clear transition found
    # Return sparsity at steepest alignment drop
    diffs = np.diff(alignments)
    steepest = int(np.argmin(diffs))
    return float(sparsities[steepest])


def lottery_ticket_phase(
    model: Any,
    dataset: NDArray,
    sparsities: Optional[Sequence[float]] = None,
    n_iterations: int = 3,
    training_steps: int = 100,
    seed: int = 42,
) -> LotteryPhase:
    """Analyze lottery ticket behavior through the phase lens.

    A winning lottery ticket is a sparse subnetwork that, when rewound
    to initialization, trains to the same quality as the full network.
    The phase framework predicts that winning tickets preserve the NTK
    structure (and hence the training regime) of the full network.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    sparsities : sequence of float or None
        Sparsity levels. If None, uses logarithmic spacing.
    n_iterations : int
        Number of iterative pruning rounds.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    LotteryPhase
        Lottery ticket analysis including winning ticket sparsity.
    """
    weights, params = _extract_pruning_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]

    if sparsities is None:
        sparsities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    sparsities_arr = np.array(sorted(sparsities))

    identity_masks = [np.ones_like(w) for w in weights]
    unpruned_eigvals = _pruned_ntk_eigenspectrum(weights, identity_masks, dataset, seed=seed)
    unpruned_mu = float(unpruned_eigvals[0]) / hidden_dim if len(unpruned_eigvals) > 0 else 1.0
    unpruned_g_star = _predict_gamma_star(unpruned_mu, training_steps)

    # Determine unpruned regime at default LR
    default_lr = 0.01
    gamma_default = _compute_gamma(default_lr, init_scale, hidden_dim)
    if gamma_default < unpruned_g_star:
        regime_before = Regime.LAZY
    elif gamma_default > unpruned_g_star * 1.2:
        regime_before = Regime.RICH
    else:
        regime_before = Regime.CRITICAL

    ticket_quality = np.zeros(len(sparsities_arr))
    ntk_alignments = np.zeros(len(sparsities_arr))
    rewind_quality_arr = np.zeros(len(sparsities_arr))
    winning_sparsity = 0.0

    for idx, s in enumerate(sparsities_arr):
        pruned_w, masks = _prune_weights_magnitude(weights, s)
        eigvals = _pruned_ntk_eigenspectrum(pruned_w, masks, dataset, seed=seed)

        alignment = _ntk_alignment(unpruned_eigvals, eigvals)
        ntk_alignments[idx] = alignment

        # Ticket quality: based on NTK alignment and gradient flow
        flow = _gradient_flow_score(pruned_w, masks)
        ticket_quality[idx] = alignment * min(1.0, flow)

        # Rewind quality: how well the pruned network at init weights preserves dynamics
        # Approximate by checking if the phase regime is preserved
        effective_width = max(1, int(hidden_dim * (1 - s)))
        mu_pruned = float(eigvals[0]) / effective_width if len(eigvals) > 0 else 1.0
        g_star_pruned = _predict_gamma_star(mu_pruned, training_steps)
        gamma_pruned = _compute_gamma(default_lr, init_scale, effective_width)

        regime_match = 1.0
        if gamma_pruned < g_star_pruned and regime_before != Regime.LAZY:
            regime_match = 0.5
        elif gamma_pruned > g_star_pruned * 1.2 and regime_before != Regime.RICH:
            regime_match = 0.5

        rewind_quality_arr[idx] = regime_match * alignment

        if ticket_quality[idx] > 0.5:
            winning_sparsity = float(s)

    # Regime at winning sparsity
    if winning_sparsity > 0:
        w_idx = int(np.argmin(np.abs(sparsities_arr - winning_sparsity)))
        ew = max(1, int(hidden_dim * (1 - winning_sparsity)))
        pw, pm = _prune_weights_magnitude(weights, winning_sparsity)
        ev = _pruned_ntk_eigenspectrum(pw, pm, dataset, seed=seed)
        mu_w = float(ev[0]) / ew if len(ev) > 0 else 1.0
        gs_w = _predict_gamma_star(mu_w, training_steps)
        g_w = _compute_gamma(default_lr, init_scale, ew)
        if g_w < gs_w:
            regime_at_win = Regime.LAZY
        elif g_w > gs_w * 1.2:
            regime_at_win = Regime.RICH
        else:
            regime_at_win = Regime.CRITICAL
    else:
        regime_at_win = regime_before

    # Iterative vs one-shot comparison
    # One-shot: prune to final sparsity directly
    oneshot_quality = float(ticket_quality[len(sparsities_arr) // 2])

    # Iterative: prune progressively
    iter_masks = [np.ones_like(w) for w in weights]
    iter_sparsity = sparsities_arr[len(sparsities_arr) // 2]
    per_round_sparsity = 1.0 - (1.0 - iter_sparsity) ** (1.0 / n_iterations)

    current_weights = [w.copy() for w in weights]
    for it in range(n_iterations):
        current_weights, iter_masks = _prune_weights_magnitude(current_weights, per_round_sparsity)

    iter_eigvals = _pruned_ntk_eigenspectrum(current_weights, iter_masks, dataset, seed=seed)
    iter_alignment = _ntk_alignment(unpruned_eigvals, iter_eigvals)
    iterative_quality = float(iter_alignment * _gradient_flow_score(current_weights, iter_masks))

    return LotteryPhase(
        winning_ticket_sparsity=winning_sparsity,
        ticket_quality=ticket_quality,
        sparsities=sparsities_arr,
        regime_before_pruning=regime_before,
        regime_at_winning_sparsity=regime_at_win,
        ntk_alignment=ntk_alignments,
        rewind_quality=rewind_quality_arr,
        iterative_vs_oneshot={
            "iterative_quality": iterative_quality,
            "oneshot_quality": oneshot_quality,
            "iterative_better": iterative_quality > oneshot_quality,
        },
    )


def structured_vs_unstructured_phase(
    model: Any,
    dataset: NDArray,
    sparsities: Optional[Sequence[float]] = None,
    training_steps: int = 100,
    seed: int = 42,
) -> StructuredPhase:
    """Compare structured and unstructured pruning in phase space.

    Structured pruning removes entire neurons/filters, reducing the
    effective width directly. Unstructured pruning removes individual
    weights, which modifies the NTK without directly changing width.
    These strategies can lead to different phase transitions.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    sparsities : sequence of float or None
        Sparsity levels to compare.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    StructuredPhase
        Comparison of structured vs. unstructured pruning.
    """
    weights, params = _extract_pruning_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]

    if sparsities is None:
        sparsities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

    sparsities_arr = np.array(sorted(sparsities))
    default_lr = 0.01

    identity_masks = [np.ones_like(w) for w in weights]
    unpruned_eigvals = _pruned_ntk_eigenspectrum(weights, identity_masks, dataset, seed=seed)

    unstructured_regimes: Dict[float, Regime] = {}
    structured_regimes: Dict[float, Regime] = {}
    effective_widths = np.zeros(len(sparsities_arr))
    ntk_unstruct = np.zeros(len(sparsities_arr))
    ntk_struct = np.zeros(len(sparsities_arr))

    unstruct_crit = 0.0
    struct_crit = 0.0

    for idx, s in enumerate(sparsities_arr):
        s_float = float(s)

        # Unstructured pruning
        u_weights, u_masks = _prune_weights_magnitude(weights, s)
        u_eigvals = _pruned_ntk_eigenspectrum(u_weights, u_masks, dataset, seed=seed)
        u_eff_width = max(1, int(hidden_dim * (1 - s)))
        u_mu = float(u_eigvals[0]) / u_eff_width if len(u_eigvals) > 0 else 1.0
        u_g_star = _predict_gamma_star(u_mu, training_steps)
        u_gamma = _compute_gamma(default_lr, init_scale, u_eff_width)
        ntk_unstruct[idx] = _ntk_alignment(unpruned_eigvals, u_eigvals)

        if u_gamma < u_g_star:
            unstructured_regimes[s_float] = Regime.LAZY
        elif u_gamma > u_g_star * 1.2:
            unstructured_regimes[s_float] = Regime.RICH
        else:
            unstructured_regimes[s_float] = Regime.CRITICAL

        # Structured pruning
        s_weights, eff_ws = _prune_structured(weights, s)
        s_masks = [np.ones_like(w) for w in s_weights]  # mask is implicit in zeroed columns
        s_eigvals = _pruned_ntk_eigenspectrum(s_weights, s_masks, dataset, seed=seed)
        s_eff_width = max(1, int(eff_ws[0]))
        effective_widths[idx] = s_eff_width
        s_mu = float(s_eigvals[0]) / s_eff_width if len(s_eigvals) > 0 else 1.0
        s_g_star = _predict_gamma_star(s_mu, training_steps)
        s_gamma = _compute_gamma(default_lr, init_scale, s_eff_width)
        ntk_struct[idx] = _ntk_alignment(unpruned_eigvals, s_eigvals)

        if s_gamma < s_g_star:
            structured_regimes[s_float] = Regime.LAZY
        elif s_gamma > s_g_star * 1.2:
            structured_regimes[s_float] = Regime.RICH
        else:
            structured_regimes[s_float] = Regime.CRITICAL

        # Detect critical sparsities
        if idx > 0:
            prev_s = float(sparsities_arr[idx - 1])
            if (unstructured_regimes.get(prev_s) != unstructured_regimes.get(s_float)
                    and unstruct_crit == 0.0):
                unstruct_crit = (prev_s + s_float) / 2
            if (structured_regimes.get(prev_s) != structured_regimes.get(s_float)
                    and struct_crit == 0.0):
                struct_crit = (prev_s + s_float) / 2

    # Recommendation
    if unstruct_crit > struct_crit and unstruct_crit > 0:
        rec_strategy = PruningStrategy.MAGNITUDE
        explanation = (
            f"Unstructured pruning maintains the training regime up to "
            f"{unstruct_crit:.0%} sparsity (vs {struct_crit:.0%} for structured). "
            f"The NTK is better preserved because individual weight removal "
            f"has a gentler effect on the kernel than removing entire neurons."
        )
    elif struct_crit > 0:
        rec_strategy = PruningStrategy.MAGNITUDE
        explanation = (
            f"Structured pruning transitions at {struct_crit:.0%} sparsity "
            f"(unstructured at {unstruct_crit:.0%}). Structured pruning directly "
            f"reduces effective width, causing sharper phase transitions."
        )
    else:
        rec_strategy = PruningStrategy.MAGNITUDE
        explanation = (
            "No clear phase transition detected at evaluated sparsities. "
            "Both strategies appear stable across the tested range."
        )

    return StructuredPhase(
        unstructured_critical_sparsity=unstruct_crit,
        structured_critical_sparsity=struct_crit,
        unstructured_regimes=unstructured_regimes,
        structured_regimes=structured_regimes,
        effective_width_reduction=effective_widths,
        ntk_preservation={
            "unstructured": ntk_unstruct,
            "structured": ntk_struct,
        },
        recommended_strategy=rec_strategy,
        explanation=explanation,
    )


def pruning_at_initialization_phase(
    model: Any,
    dataset: Optional[NDArray] = None,
    sparsities: Optional[Sequence[float]] = None,
    training_steps: int = 100,
    seed: int = 42,
) -> InitPruningPhase:
    """Analyze phase behavior of pruning-at-initialization methods.

    Compares SNIP, GraSP, SynFlow, and random pruning in terms of
    their effect on the phase diagram. Each method selects different
    weights to prune, leading to different NTK structures.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray or None
        Input data (required for SNIP, optional for SynFlow).
    sparsities : sequence of float or None
        Sparsity levels to evaluate.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    InitPruningPhase
        Comparison of pruning-at-init strategies.
    """
    weights, params = _extract_pruning_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    input_dim = params["input_dim"]

    if dataset is None:
        rng = np.random.RandomState(seed)
        dataset = rng.randn(100, input_dim)

    if sparsities is None:
        sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    sparsities_arr = np.array(sorted(sparsities))

    strategies = {
        "magnitude": PruningStrategy.MAGNITUDE,
        "random": PruningStrategy.RANDOM,
        "snip": PruningStrategy.SNIP,
        "synflow": PruningStrategy.SYNFLOW,
    }

    strategy_rankings: Dict[str, float] = {}
    critical_sparsities: Dict[str, float] = {}
    ntk_scores: Dict[str, NDArray] = {}
    flow_scores: Dict[str, NDArray] = {}
    regime_maps: Dict[str, Dict[float, Regime]] = {}

    identity_masks = [np.ones_like(w) for w in weights]
    unpruned_eigvals = _pruned_ntk_eigenspectrum(weights, identity_masks, dataset, seed=seed)

    # Compute strategy-specific scores
    snip_sc = _snip_scores(weights, dataset, seed)
    synflow_sc = _synflow_scores(weights)

    for strat_name, strat_enum in strategies.items():
        ntk_pres = np.zeros(len(sparsities_arr))
        flow_pres = np.zeros(len(sparsities_arr))
        regimes: Dict[float, Regime] = {}
        crit_s = 0.0

        for idx, s in enumerate(sparsities_arr):
            s_float = float(s)

            if strat_name == "magnitude":
                pw, pm = _prune_weights_magnitude(weights, s)
            elif strat_name == "random":
                pw, pm = _prune_weights_random(weights, s, seed)
            elif strat_name == "snip":
                # Prune based on SNIP scores
                all_scores = np.concatenate([sc.flatten() for sc in snip_sc])
                threshold = np.percentile(all_scores, s * 100)
                pm = [(sc >= threshold).astype(float) for sc in snip_sc]
                pw = [w * m for w, m in zip(weights, pm)]
            elif strat_name == "synflow":
                all_scores = np.concatenate([sc.flatten() for sc in synflow_sc])
                threshold = np.percentile(all_scores, s * 100)
                pm = [(sc >= threshold).astype(float) for sc in synflow_sc]
                pw = [w * m for w, m in zip(weights, pm)]
            else:
                pw, pm = _prune_weights_magnitude(weights, s)

            eigvals = _pruned_ntk_eigenspectrum(pw, pm, dataset, seed=seed)
            ntk_pres[idx] = _ntk_alignment(unpruned_eigvals, eigvals)
            flow_pres[idx] = _gradient_flow_score(pw, pm)

            eff_w = max(1, int(hidden_dim * (1 - s)))
            mu = float(eigvals[0]) / eff_w if len(eigvals) > 0 else 1.0
            gs = _predict_gamma_star(mu, training_steps)
            g = _compute_gamma(0.01, init_scale, eff_w)

            if g < gs:
                regimes[s_float] = Regime.LAZY
            elif g > gs * 1.2:
                regimes[s_float] = Regime.RICH
            else:
                regimes[s_float] = Regime.CRITICAL

            if idx > 0 and crit_s == 0.0:
                prev_s = float(sparsities_arr[idx - 1])
                if regimes.get(prev_s) != regimes.get(s_float):
                    crit_s = (prev_s + s_float) / 2

        ntk_scores[strat_name] = ntk_pres
        flow_scores[strat_name] = flow_pres
        regime_maps[strat_name] = regimes
        critical_sparsities[strat_name] = crit_s

        # Overall ranking: average NTK preservation weighted by gradient flow
        strategy_rankings[strat_name] = float(np.mean(ntk_pres * flow_pres))

    best_name = max(strategy_rankings, key=lambda k: strategy_rankings[k])
    best_strat = strategies[best_name]

    return InitPruningPhase(
        strategy_rankings=strategy_rankings,
        best_strategy=best_strat,
        critical_sparsities=critical_sparsities,
        ntk_preservation_scores=ntk_scores,
        gradient_flow_scores=flow_scores,
        phase_regime_map=regime_maps,
    )
