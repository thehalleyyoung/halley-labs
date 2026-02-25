"""Phase transitions in neural network quantization.

Analyzes phase transitions that occur during quantization: reducing
numerical precision introduces quantization noise that modifies the
effective NTK and shifts phase boundaries. There exists a critical
bit-width below which the network undergoes a phase transition from
feature-learning to a noise-dominated regime.

Example
-------
>>> from phase_diagrams.quantization_phase import quantization_phase_diagram
>>> diagram = quantization_phase_diagram(model, dataset, bits=[2, 4, 8, 16, 32])
>>> print(diagram.metadata["critical_bits"])
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

class QuantizationMode(str, Enum):
    """Quantization approach."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DYNAMIC = "dynamic"


@dataclass
class MixedPrecisionPhase:
    """Phase analysis of mixed-precision quantization.

    Attributes
    ----------
    layer_bits : Dict[int, int]
        Optimal bit-width assignment per layer.
    per_layer_sensitivity : NDArray
        Quantization sensitivity score per layer.
    per_layer_regime : Dict[int, Regime]
        Phase regime per layer at assigned precision.
    total_model_bits : float
        Average bits per parameter.
    memory_reduction : float
        Memory reduction factor vs. FP32.
    accuracy_preservation : float
        Estimated accuracy preservation (0-1).
    phase_boundary_shift : NDArray
        How much the phase boundary shifts per layer.
    recommendation : str
        Human-readable mixed-precision recommendation.
    """
    layer_bits: Dict[int, int] = field(default_factory=dict)
    per_layer_sensitivity: NDArray = field(default_factory=lambda: np.array([]))
    per_layer_regime: Dict[int, Regime] = field(default_factory=dict)
    total_model_bits: float = 32.0
    memory_reduction: float = 1.0
    accuracy_preservation: float = 1.0
    phase_boundary_shift: NDArray = field(default_factory=lambda: np.array([]))
    recommendation: str = ""


@dataclass
class QATPhase:
    """Phase analysis of quantization-aware training.

    Attributes
    ----------
    pre_qat_regime : Regime
        Phase regime before QAT.
    post_qat_regime : Regime
        Phase regime after QAT.
    qat_critical_lr : float
        Critical LR during QAT.
    straight_through_error : float
        Error introduced by straight-through estimator.
    ntk_with_ste : NDArray
        NTK eigenspectrum with STE gradients.
    convergence_rate : float
        Relative convergence rate of QAT vs. full-precision.
    bits_schedule : Dict[int, int]
        Recommended bit-width schedule during training.
    regime_trajectory : List[Regime]
        How the regime evolves during QAT.
    """
    pre_qat_regime: Regime = Regime.LAZY
    post_qat_regime: Regime = Regime.LAZY
    qat_critical_lr: float = 0.0
    straight_through_error: float = 0.0
    ntk_with_ste: NDArray = field(default_factory=lambda: np.array([]))
    convergence_rate: float = 1.0
    bits_schedule: Dict[int, int] = field(default_factory=dict)
    regime_trajectory: List[Regime] = field(default_factory=list)


@dataclass
class PTQPhase:
    """Phase analysis of post-training quantization.

    Attributes
    ----------
    pre_quantization_regime : Regime
        Regime of the full-precision trained model.
    post_quantization_regime : Regime
        Regime after quantization.
    regime_preserved : bool
        Whether quantization preserves the training regime.
    optimal_bits : int
        Minimum bits that preserve the regime.
    weight_quantization_error : float
        MSE of quantized vs. original weights.
    activation_quantization_error : float
        MSE of quantized vs. original activations.
    calibration_quality : float
        Quality of calibration data for quantization ranges.
    ntk_drift_from_quantization : float
        NTK drift caused by quantization (analogous to training drift).
    per_layer_bits : Dict[int, int]
        Optimal per-layer bit allocation.
    """
    pre_quantization_regime: Regime = Regime.LAZY
    post_quantization_regime: Regime = Regime.LAZY
    regime_preserved: bool = True
    optimal_bits: int = 8
    weight_quantization_error: float = 0.0
    activation_quantization_error: float = 0.0
    calibration_quality: float = 1.0
    ntk_drift_from_quantization: float = 0.0
    per_layer_bits: Dict[int, int] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_quant_params(model: Any) -> Tuple[List[NDArray], Dict[str, Any]]:
    """Extract weights and architecture params for quantization analysis."""
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
            "input_dim": input_dim, "hidden_dim": hidden_dim,
            "output_dim": output_dim, "depth": depth, "init_scale": init_scale,
        }
        return weights, params

    if isinstance(model, (list, tuple)):
        weights = [np.asarray(w, dtype=np.float64) for w in model]
        input_dim = weights[0].shape[0] if weights[0].ndim == 2 else weights[0].shape[1]
        hidden_dim = weights[0].shape[1] if weights[0].ndim == 2 else weights[0].shape[0]
        output_dim = weights[-1].shape[1] if weights[-1].ndim == 2 else weights[-1].shape[0]
        init_scale = float(np.std(weights[0]) * math.sqrt(hidden_dim))
        params = {
            "input_dim": input_dim, "hidden_dim": hidden_dim,
            "output_dim": output_dim, "depth": len(weights), "init_scale": init_scale,
        }
        return weights, params

    raise TypeError(f"Unsupported model type: {type(model)}")


def _quantize_tensor(
    x: NDArray,
    bits: int,
    mode: QuantizationMode = QuantizationMode.SYMMETRIC,
) -> Tuple[NDArray, float]:
    """Quantize a tensor to the specified bit width.

    Returns (quantized_tensor, quantization_error).
    """
    if bits >= 32:
        return x.copy(), 0.0

    n_levels = 2 ** bits
    if mode == QuantizationMode.SYMMETRIC:
        abs_max = np.max(np.abs(x)) + 1e-12
        scale = abs_max / (n_levels // 2)
        quantized = np.round(x / scale) * scale
        quantized = np.clip(quantized, -abs_max, abs_max)
    elif mode == QuantizationMode.ASYMMETRIC:
        x_min, x_max = np.min(x), np.max(x)
        scale = (x_max - x_min + 1e-12) / (n_levels - 1)
        quantized = np.round((x - x_min) / scale) * scale + x_min
    else:
        # Dynamic: per-channel quantization (along axis 0)
        if x.ndim >= 2:
            abs_max = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
        else:
            abs_max = np.max(np.abs(x)) + 1e-12
        scale = abs_max / (n_levels // 2)
        quantized = np.round(x / scale) * scale

    error = float(np.mean((x - quantized) ** 2))
    return quantized, error


def _quantized_ntk_eigenspectrum(
    weights: List[NDArray],
    dataset: NDArray,
    bits: int,
    n_samples: int = 50,
    seed: int = 42,
) -> Tuple[NDArray, float]:
    """Compute NTK eigenspectrum of the quantized network.

    Returns (eigenvalues, total_quantization_error).
    """
    n = min(n_samples, dataset.shape[0])
    X = dataset[:n]
    depth = len(weights)

    # Quantize weights
    q_weights = []
    total_error = 0.0
    for w in weights:
        qw, err = _quantize_tensor(w, bits)
        q_weights.append(qw)
        total_error += err

    K = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        w = q_weights[l]
        pre = h @ w
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

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    return eigvals, total_error


def _quantization_noise_model(bits: int, weight_range: float) -> float:
    """Theoretical quantization noise variance.

    For uniform quantization with b bits over range [-R, R],
    the quantization noise variance is R^2 / (3 * 2^{2b}).
    """
    if bits >= 32:
        return 0.0
    return weight_range ** 2 / (3.0 * (2 ** (2 * bits)))


def _layer_sensitivity(
    weights: List[NDArray],
    layer_idx: int,
    dataset: NDArray,
    seed: int = 42,
) -> float:
    """Compute quantization sensitivity of a specific layer.

    Sensitivity is measured as the relative change in NTK when
    only this layer is quantized to low precision.
    """
    n = min(30, dataset.shape[0])
    X = dataset[:n]
    depth = len(weights)

    # Full-precision NTK
    K_fp = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        pre = h @ weights[l]
        if l < depth - 1:
            for i in range(n):
                for j in range(i, n):
                    val = np.dot(h[i], h[j])
                    K_fp[i, j] += val
                    K_fp[j, i] = K_fp[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K_fp[i, j] += np.dot(h[i], h[j])
                    K_fp[j, i] = K_fp[i, j]

    # Quantized-layer NTK (quantize only target layer to 4 bits)
    q_weights = [w.copy() for w in weights]
    q_weights[layer_idx], _ = _quantize_tensor(weights[layer_idx], 4)

    K_q = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        pre = h @ q_weights[l]
        if l < depth - 1:
            for i in range(n):
                for j in range(i, n):
                    val = np.dot(h[i], h[j])
                    K_q[i, j] += val
                    K_q[j, i] = K_q[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K_q[i, j] += np.dot(h[i], h[j])
                    K_q[j, i] = K_q[i, j]

    fp_norm = np.linalg.norm(K_fp, "fro") + 1e-12
    diff_norm = np.linalg.norm(K_fp - K_q, "fro")
    return float(diff_norm / fp_norm)


# ======================================================================
# Public API
# ======================================================================

def quantization_phase_diagram(
    model: Any,
    dataset: NDArray,
    bits: Sequence[int],
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    n_lr_steps: int = 25,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram over learning rate and bit width.

    Quantization introduces noise that modifies the NTK and shifts
    the phase boundary. At very low bit widths, the quantization noise
    dominates and the model enters a noise-dominated regime.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification or weight matrices.
    dataset : NDArray
        Input data of shape (n_samples, input_dim).
    bits : sequence of int
        Bit widths to evaluate (e.g., [2, 4, 8, 16, 32]).
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
        Phase diagram with bit width as the width axis.
    """
    weights, params = _extract_quant_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]

    bits_arr = np.array(sorted(bits))
    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)

    # Full-precision reference
    fp_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, 32, seed=seed)
    fp_mu_max = float(fp_eigvals[0]) / hidden_dim if len(fp_eigvals) > 0 else 1.0

    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []
    crit_bits = int(bits_arr[0])

    for b in bits_arr:
        b_int = int(b)
        eigvals, q_error = _quantized_ntk_eigenspectrum(weights, dataset, b_int, seed=seed)
        mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0

        # Quantization noise acts as an effective regularizer,
        # reducing the coupling by adding noise to the NTK
        weight_range = float(np.max(np.abs(np.concatenate([w.flatten() for w in weights]))))
        q_noise = _quantization_noise_model(b_int, weight_range)
        noise_correction = 1.0 / (1.0 + q_noise * hidden_dim)

        g_star = _predict_gamma_star(mu_max * noise_correction, training_steps)

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

            ntk_drift = gamma * mu_max * noise_correction * training_steps
            points.append(PhasePoint(
                lr=float(lr),
                width=b_int,
                regime=regime,
                gamma=gamma,
                gamma_star=g_star,
                confidence=max(0.0, min(1.0, confidence)),
                ntk_drift_predicted=ntk_drift,
            ))

            if prev_regime is not None and prev_regime != regime:
                boundary_pts.append((float(lr), b_int))
            prev_regime = regime

    # Find critical bits
    for i in range(len(bits_arr) - 1):
        b1 = int(bits_arr[i])
        b2 = int(bits_arr[i + 1])
        pts1 = [p for p in points if p.width == b1]
        pts2 = [p for p in points if p.width == b2]
        if pts1 and pts2:
            g1 = np.mean([p.gamma_star for p in pts1])
            g2 = np.mean([p.gamma_star for p in pts2])
            if abs(g2 - g1) / (g1 + 1e-12) > 0.3:
                crit_bits = b2
                break

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    tc_vals = [training_steps * _compute_gamma(bp[0], init_scale, hidden_dim) for bp in boundary_pts]
    tc = float(np.mean(tc_vals)) if tc_vals else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(int(bits_arr[0]), int(bits_arr[-1])),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": "QuantizedMLP",
            "hidden_dim": hidden_dim,
            "depth": params["depth"],
            "bits_evaluated": bits_arr.tolist(),
            "critical_bits": crit_bits,
        },
    )


def critical_bits(
    model: Any,
    dataset: NDArray,
    max_bits: int = 32,
    training_steps: int = 100,
    seed: int = 42,
) -> int:
    """Find the minimum bit width before phase collapse.

    Binary-searches for the bit width at which the NTK spectrum changes
    qualitatively, indicating a phase transition from the feature-learning
    regime to a noise-dominated regime.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    max_bits : int
        Maximum bit width (reference precision).
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    int
        Minimum bit width that preserves the training regime.
    """
    weights, params = _extract_quant_params(model)
    hidden_dim = params["hidden_dim"]

    fp_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, max_bits, seed=seed)

    # Binary search between 1 and max_bits
    lo, hi = 1, max_bits
    threshold = 0.5  # NTK alignment threshold

    while lo < hi:
        mid = (lo + hi) // 2
        q_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, mid, seed=seed)

        # Alignment between quantized and full-precision NTK
        min_len = min(len(fp_eigvals), len(q_eigvals))
        a = fp_eigvals[:min_len]
        b = q_eigvals[:min_len]
        alignment = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        if alignment >= threshold:
            hi = mid
        else:
            lo = mid + 1

    return lo


def mixed_precision_phase(
    model: Any,
    dataset: NDArray,
    bit_options: Optional[Sequence[int]] = None,
    training_steps: int = 100,
    seed: int = 42,
) -> MixedPrecisionPhase:
    """Determine optimal mixed-precision quantization for phase preservation.

    Different layers have different sensitivity to quantization. Early
    layers and the final layer are typically more sensitive because they
    directly affect the NTK spectrum. This function assigns optimal
    bit widths per layer.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    bit_options : sequence of int or None
        Available bit widths. If None, uses [4, 8, 16, 32].
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    MixedPrecisionPhase
        Optimal per-layer bit assignment and analysis.
    """
    weights, params = _extract_quant_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    depth = params["depth"]

    if bit_options is None:
        bit_options = [4, 8, 16, 32]
    bit_options = sorted(bit_options)

    # Compute per-layer sensitivity
    sensitivities = np.zeros(depth)
    for l in range(depth):
        sensitivities[l] = _layer_sensitivity(weights, l, dataset, seed)

    # Assign bits based on sensitivity ranking
    # Most sensitive layers get most bits
    sensitivity_rank = np.argsort(sensitivities)[::-1]  # highest sensitivity first
    layer_bits: Dict[int, int] = {}
    per_layer_regime: Dict[int, Regime] = {}
    phase_shifts = np.zeros(depth)

    # Strategy: assign higher precision to more sensitive layers
    n_bit_levels = len(bit_options)
    for rank, l in enumerate(sensitivity_rank):
        # Map rank to bit level
        level = min(rank * n_bit_levels // depth, n_bit_levels - 1)
        layer_bits[l] = bit_options[-(level + 1)]  # highest bits for most sensitive

    # Compute phase boundary with mixed precision
    q_weights = []
    total_q_error = 0.0
    for l, w in enumerate(weights):
        qw, err = _quantize_tensor(w, layer_bits[l])
        q_weights.append(qw)
        total_q_error += err

    # NTK with mixed precision
    n = min(30, dataset.shape[0])
    X = dataset[:n]
    K_mp = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        pre = h @ q_weights[l]
        if l < depth - 1:
            for i in range(n):
                for j in range(i, n):
                    K_mp[i, j] += np.dot(h[i], h[j])
                    K_mp[j, i] = K_mp[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K_mp[i, j] += np.dot(h[i], h[j])
                    K_mp[j, i] = K_mp[i, j]

    mp_eigvals = np.sort(np.linalg.eigvalsh(K_mp))[::-1]
    mu_max = float(mp_eigvals[0]) / hidden_dim if len(mp_eigvals) > 0 else 1.0
    g_star = _predict_gamma_star(mu_max, training_steps)

    for l in range(depth):
        gamma_l = _compute_gamma(0.01, init_scale, hidden_dim)
        if gamma_l < g_star:
            per_layer_regime[l] = Regime.LAZY
        elif gamma_l > g_star * 1.2:
            per_layer_regime[l] = Regime.RICH
        else:
            per_layer_regime[l] = Regime.CRITICAL
        phase_shifts[l] = sensitivities[l] * (32 - layer_bits[l]) / 32.0

    avg_bits = float(np.mean(list(layer_bits.values())))
    memory_reduction = 32.0 / avg_bits
    accuracy_preservation = 1.0 - total_q_error * depth

    # Recommendation
    high_bit_layers = [l for l, b in layer_bits.items() if b >= 16]
    low_bit_layers = [l for l, b in layer_bits.items() if b <= 8]
    recommendation = (
        f"Use {max(bit_options)} bits for layers {high_bit_layers} (most sensitive). "
        f"Use {min(bit_options)} bits for layers {low_bit_layers} (least sensitive). "
        f"Average {avg_bits:.1f} bits/param, {memory_reduction:.1f}x memory reduction."
    )

    return MixedPrecisionPhase(
        layer_bits=layer_bits,
        per_layer_sensitivity=sensitivities,
        per_layer_regime=per_layer_regime,
        total_model_bits=avg_bits,
        memory_reduction=memory_reduction,
        accuracy_preservation=max(0.0, min(1.0, accuracy_preservation)),
        phase_boundary_shift=phase_shifts,
        recommendation=recommendation,
    )


def quantization_aware_training_phase(
    model: Any,
    dataset: NDArray,
    target_bits: int = 8,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> QATPhase:
    """Analyze phase behavior during quantization-aware training.

    QAT inserts fake quantization operations during training, using
    the straight-through estimator (STE) for backpropagation. The STE
    introduces a systematic bias in gradients that modifies the NTK.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    dataset : NDArray
        Input data.
    target_bits : int
        Target quantization bit width.
    lr : float
        Training learning rate.
    training_steps : int
        Number of QAT steps.
    seed : int
        Random seed.

    Returns
    -------
    QATPhase
        Phase analysis of QAT training dynamics.
    """
    weights, params = _extract_quant_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    depth = params["depth"]

    # Pre-QAT regime
    fp_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, 32, seed=seed)
    fp_mu = float(fp_eigvals[0]) / hidden_dim if len(fp_eigvals) > 0 else 1.0
    fp_g_star = _predict_gamma_star(fp_mu, training_steps)
    gamma = _compute_gamma(lr, init_scale, hidden_dim)

    if gamma < fp_g_star:
        pre_regime = Regime.LAZY
    elif gamma > fp_g_star * 1.2:
        pre_regime = Regime.RICH
    else:
        pre_regime = Regime.CRITICAL

    # QAT NTK: the STE modifies the effective gradient
    q_eigvals, q_error = _quantized_ntk_eigenspectrum(weights, dataset, target_bits, seed=seed)
    q_mu = float(q_eigvals[0]) / hidden_dim if len(q_eigvals) > 0 else 1.0

    # STE error: gradient approximation error from straight-through
    ste_error = 0.0
    for w in weights:
        qw, _ = _quantize_tensor(w, target_bits)
        # STE uses quantized forward but real gradient
        # Error is proportional to quantization noise * gradient magnitude
        ste_error += float(np.mean((w - qw) ** 2)) * float(np.mean(w ** 2))

    # QAT effectively uses a modified NTK: K_qat = K_fp + noise * I
    noise_level = ste_error * hidden_dim
    qat_mu = fp_mu / (1.0 + noise_level)
    qat_g_star = _predict_gamma_star(qat_mu, training_steps)
    qat_critical_lr = qat_g_star * hidden_dim / (init_scale ** 2 + 1e-12)

    if gamma < qat_g_star:
        post_regime = Regime.LAZY
    elif gamma > qat_g_star * 1.2:
        post_regime = Regime.RICH
    else:
        post_regime = Regime.CRITICAL

    # Convergence rate relative to full precision
    convergence_rate = fp_mu / (qat_mu + 1e-12) if qat_mu > 0 else 0.0
    convergence_rate = min(2.0, max(0.1, convergence_rate))

    # Bits schedule: start with higher precision, gradually reduce
    bits_schedule: Dict[int, int] = {}
    steps_per_stage = max(1, training_steps // 4)
    bits_schedule[0] = 32  # full precision warmup
    bits_schedule[steps_per_stage] = min(32, target_bits * 4)
    bits_schedule[2 * steps_per_stage] = min(32, target_bits * 2)
    bits_schedule[3 * steps_per_stage] = target_bits

    # Regime trajectory during QAT
    trajectory = []
    for step_bits in [32, target_bits * 4, target_bits * 2, target_bits]:
        step_bits = min(32, max(1, step_bits))
        s_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, step_bits, seed=seed)
        s_mu = float(s_eigvals[0]) / hidden_dim if len(s_eigvals) > 0 else 1.0
        s_noise = _quantization_noise_model(step_bits, float(np.max(np.abs(
            np.concatenate([w.flatten() for w in weights])))))
        s_g_star = _predict_gamma_star(s_mu / (1 + s_noise * hidden_dim), training_steps)
        if gamma < s_g_star:
            trajectory.append(Regime.LAZY)
        elif gamma > s_g_star * 1.2:
            trajectory.append(Regime.RICH)
        else:
            trajectory.append(Regime.CRITICAL)

    return QATPhase(
        pre_qat_regime=pre_regime,
        post_qat_regime=post_regime,
        qat_critical_lr=qat_critical_lr,
        straight_through_error=ste_error,
        ntk_with_ste=q_eigvals,
        convergence_rate=convergence_rate,
        bits_schedule=bits_schedule,
        regime_trajectory=trajectory,
    )


def post_training_quantization_phase(
    model: Any,
    dataset: NDArray,
    target_bits: int = 8,
    lr: float = 0.001,
    training_steps: int = 100,
    calibration_samples: int = 100,
    seed: int = 42,
) -> PTQPhase:
    """Analyze how post-training quantization affects the phase regime.

    PTQ quantizes a trained model without retraining. The key question
    is whether quantization preserves the training regime: if the model
    trained in the rich regime, does quantization push it back to lazy?

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification (represents trained model).
    dataset : NDArray
        Calibration data.
    target_bits : int
        Target quantization bit width.
    lr : float
        Original training learning rate (for regime classification).
    training_steps : int
        Original training duration.
    calibration_samples : int
        Number of calibration samples.
    seed : int
        Random seed.

    Returns
    -------
    PTQPhase
        Analysis of PTQ's effect on training regime.
    """
    weights, params = _extract_quant_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    depth = params["depth"]

    n_cal = min(calibration_samples, dataset.shape[0])
    X_cal = dataset[:n_cal]

    # Pre-quantization regime
    fp_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, 32, seed=seed)
    fp_mu = float(fp_eigvals[0]) / hidden_dim if len(fp_eigvals) > 0 else 1.0
    fp_g_star = _predict_gamma_star(fp_mu, training_steps)
    gamma = _compute_gamma(lr, init_scale, hidden_dim)

    if gamma < fp_g_star:
        pre_regime = Regime.LAZY
    elif gamma > fp_g_star * 1.2:
        pre_regime = Regime.RICH
    else:
        pre_regime = Regime.CRITICAL

    # Quantize weights
    q_weights = []
    weight_q_error = 0.0
    for w in weights:
        qw, err = _quantize_tensor(w, target_bits)
        q_weights.append(qw)
        weight_q_error += err

    # Activation quantization error (approximate)
    h_fp = X_cal.copy()
    h_q = X_cal.copy()
    act_q_error = 0.0
    for l in range(depth):
        pre_fp = h_fp @ weights[l]
        pre_q = h_q @ q_weights[l]
        if l < depth - 1:
            h_fp = np.maximum(pre_fp, 0)
            h_q_raw = np.maximum(pre_q, 0)
            h_q, a_err = _quantize_tensor(h_q_raw, target_bits)
            act_q_error += a_err
        else:
            h_fp = pre_fp
            h_q = pre_q
    act_q_error /= max(1, depth - 1)

    # Post-quantization NTK
    q_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, target_bits, seed=seed)
    q_mu = float(q_eigvals[0]) / hidden_dim if len(q_eigvals) > 0 else 1.0
    q_g_star = _predict_gamma_star(q_mu, training_steps)

    if gamma < q_g_star:
        post_regime = Regime.LAZY
    elif gamma > q_g_star * 1.2:
        post_regime = Regime.RICH
    else:
        post_regime = Regime.CRITICAL

    regime_preserved = (pre_regime == post_regime)

    # NTK drift from quantization
    min_len = min(len(fp_eigvals), len(q_eigvals))
    ntk_drift = float(np.linalg.norm(fp_eigvals[:min_len] - q_eigvals[:min_len]) / (
        np.linalg.norm(fp_eigvals[:min_len]) + 1e-12))

    # Find optimal bits for regime preservation
    optimal = target_bits
    for b in range(1, 33):
        b_eigvals, _ = _quantized_ntk_eigenspectrum(weights, dataset, b, seed=seed)
        b_mu = float(b_eigvals[0]) / hidden_dim if len(b_eigvals) > 0 else 1.0
        b_g_star = _predict_gamma_star(b_mu, training_steps)
        if gamma < b_g_star:
            b_regime = Regime.LAZY
        elif gamma > b_g_star * 1.2:
            b_regime = Regime.RICH
        else:
            b_regime = Regime.CRITICAL
        if b_regime == pre_regime:
            optimal = b
            break

    # Per-layer optimal bits
    per_layer_bits: Dict[int, int] = {}
    for l in range(depth):
        sensitivity = _layer_sensitivity(weights, l, dataset, seed)
        if sensitivity > 0.1:
            per_layer_bits[l] = max(target_bits, 16)
        elif sensitivity > 0.01:
            per_layer_bits[l] = max(target_bits, 8)
        else:
            per_layer_bits[l] = target_bits

    # Calibration quality: how well calibration data represents the true distribution
    cal_eigvals = np.sort(np.linalg.eigvalsh(X_cal @ X_cal.T))[::-1]
    eff_rank = float(np.sum(cal_eigvals) / (cal_eigvals[0] + 1e-12))
    cal_quality = min(1.0, eff_rank / params["input_dim"])

    return PTQPhase(
        pre_quantization_regime=pre_regime,
        post_quantization_regime=post_regime,
        regime_preserved=regime_preserved,
        optimal_bits=optimal,
        weight_quantization_error=weight_q_error,
        activation_quantization_error=act_q_error,
        calibration_quality=cal_quality,
        ntk_drift_from_quantization=ntk_drift,
        per_layer_bits=per_layer_bits,
    )
