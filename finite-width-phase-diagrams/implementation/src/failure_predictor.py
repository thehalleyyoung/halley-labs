"""Predict training failures before they happen using NTK/phase theory.

Uses spectral properties of the Neural Tangent Kernel, mean-field gradient
propagation analysis, and phase-diagram theory to identify likely failure
modes (NaN divergence, non-convergence, mode collapse, catastrophic
forgetting, gradient pathologies) *before* training begins.

Example
-------
>>> from phase_diagrams.failure_predictor import predict_failures
>>> model = {"input_dim": 784, "width": 256, "depth": 4,
...          "init_scale": 1.0, "activation": "relu"}
>>> config = {"lr": 0.01, "epochs": 100, "batch_size": 64,
...           "optimizer": "sgd", "task_type": "classification"}
>>> failures = predict_failures(model, config)
>>> for f in failures:
...     print(f.failure_type, f.probability, f.severity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .api import Regime


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class PredictedFailure:
    """A predicted training failure mode.

    Attributes
    ----------
    failure_type : str
        Short identifier for the failure mode (e.g. ``"nan_divergence"``).
    probability : float
        Estimated probability in [0, 1].
    severity : str
        One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    description : str
        Human-readable description of the failure.
    mitigation : str
        Suggested action to avoid the failure.
    affected_epoch_range : Tuple[int, int]
        Estimated (start, end) epoch window where the failure is likely.
    """

    failure_type: str = ""
    probability: float = 0.0
    severity: str = "low"
    description: str = ""
    mitigation: str = ""
    affected_epoch_range: Tuple[int, int] = (0, 0)


@dataclass
class PathologyReport:
    """Gradient pathology diagnosis for a network architecture.

    Attributes
    ----------
    has_vanishing_gradients : bool
        True when cumulative gradient gain across layers < 0.01.
    has_exploding_gradients : bool
        True when cumulative gradient gain across layers > 100.
    gradient_norm_estimate : float
        Estimated ratio of output-to-input gradient norm.
    effective_depth : float
        Number of layers that meaningfully contribute to learning.
    dead_neuron_fraction : float
        Estimated fraction of permanently inactive neurons (ReLU).
    spectral_gap : float
        Ratio lambda_max / lambda_2 of the NTK spectrum.
    issues : List[str]
        Detected problems.
    recommendations : List[str]
        Actionable fixes.
    """

    has_vanishing_gradients: bool = False
    has_exploding_gradients: bool = False
    gradient_norm_estimate: float = 1.0
    effective_depth: float = 1.0
    dead_neuron_fraction: float = 0.0
    spectral_gap: float = 1.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ======================================================================
# Helper utilities
# ======================================================================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _lambda_max(width: int, init_scale: float) -> float:
    """Approximate largest NTK eigenvalue for a dense layer."""
    return float(width) * init_scale ** 2


def _lambda_min(width: int, depth: int, init_scale: float) -> float:
    """Approximate smallest NTK eigenvalue for an MLP."""
    return float(width) * init_scale ** 2 / (float(width) * float(depth))


def _severity_from_probability(prob: float) -> str:
    """Map a failure probability to a severity label."""
    if prob >= 0.8:
        return "critical"
    if prob >= 0.5:
        return "high"
    if prob >= 0.25:
        return "medium"
    return "low"


# ======================================================================
# Failure-mode estimators
# ======================================================================

def nan_risk(model: Dict[str, Any], lr: float) -> float:
    """Estimate the probability of NaN divergence during training.

    Parameters
    ----------
    model : Dict
        Architecture descriptor with keys ``width``, ``init_scale``,
        ``activation``, and ``depth``.
    lr : float
        Learning rate.

    Returns
    -------
    float
        Probability in [0, 1].

    Notes
    -----
    Stability requires ``lr * lambda_max < 2`` for gradient descent on a
    quadratic.  We model the transition with a sharp sigmoid centred near
    the instability boundary (1.8 rather than 2.0 to account for
    stochastic effects).  Adam effectively reduces the learning rate by
    approximately ``1 / sqrt(variance)``, so we dampen the risk
    accordingly.
    """
    width = int(model.get("width", 256))
    init_scale = float(model.get("init_scale", 1.0))

    lam_max = _lambda_max(width, init_scale)
    product = lr * lam_max

    # Sharp sigmoid transition near the instability boundary
    raw_risk = _sigmoid(10.0 * (product - 1.8))

    # Optimizer adjustment
    optimizer = model.get("optimizer", "sgd")
    if optimizer == "adam":
        # Adam's adaptive scaling reduces effective lr
        raw_risk = _sigmoid(10.0 * (product / 3.0 - 1.8))

    return _clamp(raw_risk)


def convergence_probability(model: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Estimate the probability of convergence within the given budget.

    Parameters
    ----------
    model : Dict
        Architecture descriptor.
    config : Dict
        Training configuration with ``lr``, ``epochs``, ``optimizer``.

    Returns
    -------
    float
        Probability of convergence in [0, 1].
    """
    width = int(model.get("width", 256))
    depth = int(model.get("depth", 3))
    init_scale = float(model.get("init_scale", 1.0))

    lam_max = _lambda_max(width, init_scale)
    lam_min = _lambda_min(width, depth, init_scale)
    lam_min = max(lam_min, 1e-12)  # avoid division by zero

    kappa = lam_max / lam_min  # condition number
    epsilon = 1e-3
    required_epochs = kappa * math.log(1.0 / epsilon)

    epochs = int(config.get("epochs", 100))
    ratio = float(epochs) / max(required_epochs, 1.0)
    prob = _sigmoid(2.0 * (ratio - 0.5))

    # Penalise learning rate far from optimal
    lr = float(config.get("lr", 0.01))
    lr_optimal = 2.0 / (lam_max + lam_min)
    lr_log_ratio = abs(math.log(max(lr, 1e-15) / max(lr_optimal, 1e-15)))
    if lr_log_ratio > 2.0:
        penalty = _sigmoid(lr_log_ratio - 2.0)
        prob *= 1.0 - 0.5 * penalty

    return _clamp(prob)


def mode_collapse_risk(model: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Estimate the risk of mode collapse for generative tasks.

    Parameters
    ----------
    model : Dict
        Architecture descriptor.
    config : Dict
        Training configuration with ``task_type``, ``lr``, ``batch_size``.

    Returns
    -------
    float
        Risk probability in [0, 1]; 0 for non-generative tasks.
    """
    task_type = str(config.get("task_type", "classification"))
    if task_type not in ("generation", "gan"):
        return 0.0

    width = int(model.get("width", 256))
    lr = float(config.get("lr", 0.01))
    batch_size = int(config.get("batch_size", 64))

    risk_factors = 0.0
    # Small width increases risk
    if width < 128:
        risk_factors += (128 - width) / 128.0 * 3.0
    # Large learning rate
    if lr > 0.001:
        risk_factors += math.log10(lr / 0.001) * 2.0
    # Small batch size
    if batch_size < 32:
        risk_factors += (32 - batch_size) / 32.0 * 2.0

    return _clamp(_sigmoid(risk_factors - 2.0))


def catastrophic_forgetting_risk(
    model: Dict[str, Any], config: Dict[str, Any]
) -> float:
    """Estimate the risk of catastrophic forgetting during fine-tuning.

    Parameters
    ----------
    model : Dict
        Architecture descriptor.  May include ``pretrain_lr`` and
        ``finetune_fraction`` (fraction of parameters updated).
    config : Dict
        Training configuration with ``task_type`` and ``lr``.

    Returns
    -------
    float
        Risk probability in [0, 1]; 0 for non-fine-tuning tasks.
    """
    task_type = str(config.get("task_type", "classification"))
    if task_type != "fine_tuning":
        return 0.0

    lr = float(config.get("lr", 0.01))
    pretrain_lr = float(model.get("pretrain_lr", 1e-3))
    finetune_frac = float(model.get("finetune_fraction", 1.0))
    width = int(model.get("width", 256))

    # Higher lr ratio ⇒ more forgetting
    lr_ratio = lr / max(pretrain_lr, 1e-15)
    lr_risk = math.log10(max(lr_ratio, 1.0))

    # More parameters updated ⇒ more forgetting
    frac_risk = finetune_frac * 3.0

    # Wider networks have more stable representations
    width_bonus = -math.log2(max(width, 1) / 128.0)

    total = lr_risk + frac_risk + width_bonus - 2.0
    return _clamp(_sigmoid(total))


# ======================================================================
# Gradient pathology analysis
# ======================================================================

def gradient_pathology_check(model: Dict[str, Any]) -> PathologyReport:
    """Diagnose gradient propagation pathologies for *model*.

    Parameters
    ----------
    model : Dict
        Architecture descriptor with ``width``, ``depth``,
        ``init_scale``, ``activation``.

    Returns
    -------
    PathologyReport
        Comprehensive gradient health report.
    """
    width = int(model.get("width", 256))
    depth = int(model.get("depth", 3))
    init_scale = float(model.get("init_scale", 1.0))
    activation = str(model.get("activation", "relu"))

    issues: List[str] = []
    recommendations: List[str] = []

    # ------------------------------------------------------------------
    # Per-layer gain
    # ------------------------------------------------------------------
    if activation == "relu":
        gain_per_layer = math.sqrt(2.0 / max(width, 1)) * init_scale
    elif activation == "tanh":
        expected_saturation = min(0.1 * depth, 0.8)
        gain_per_layer = init_scale * (1.0 - expected_saturation)
    else:
        # Generic linear-like activation
        gain_per_layer = init_scale * math.sqrt(1.0 / max(width, 1))

    cumulative_gain = gain_per_layer ** depth

    # ------------------------------------------------------------------
    # Vanishing / exploding classification
    # ------------------------------------------------------------------
    has_vanishing = cumulative_gain < 0.01
    has_exploding = cumulative_gain > 100.0

    if has_vanishing:
        issues.append(
            f"Vanishing gradients: cumulative gain = {cumulative_gain:.2e} "
            f"across {depth} layers."
        )
        recommendations.append(
            "Use residual / skip connections or reduce depth."
        )
        recommendations.append(
            "Increase init_scale or switch to He initialisation."
        )
    if has_exploding:
        issues.append(
            f"Exploding gradients: cumulative gain = {cumulative_gain:.2e} "
            f"across {depth} layers."
        )
        recommendations.append("Decrease init_scale or add gradient clipping.")

    # ------------------------------------------------------------------
    # Dead neuron fraction (ReLU specific)
    # ------------------------------------------------------------------
    if activation == "relu":
        # Approximate fraction of permanently dead ReLU units
        survival_per_neuron = 1.0 - 0.5 ** (1.0 / max(width, 1))
        alive_fraction = survival_per_neuron ** depth
        dead_frac = _clamp(1.0 - alive_fraction)
        if dead_frac > 0.3:
            issues.append(
                f"High dead neuron fraction: ~{dead_frac:.1%} estimated dead."
            )
            recommendations.append(
                "Consider Leaky ReLU or ELU to reduce dead neurons."
            )
    else:
        dead_frac = 0.0

    # ------------------------------------------------------------------
    # Spectral gap
    # ------------------------------------------------------------------
    lam_max = _lambda_max(width, init_scale)
    lam_min = _lambda_min(width, depth, init_scale)
    lam_min = max(lam_min, 1e-12)
    spectral_gap = lam_max / max(lam_min, 1e-12)

    if spectral_gap > 1e4:
        issues.append(
            f"Large spectral gap ({spectral_gap:.1e}): slow convergence "
            "along smallest eigendirections."
        )
        recommendations.append(
            "Use adaptive optimiser (Adam) or preconditioned SGD."
        )

    # ------------------------------------------------------------------
    # Effective depth
    # ------------------------------------------------------------------
    if gain_per_layer < 1.0 and gain_per_layer > 0.0:
        effective_depth = -1.0 / math.log(gain_per_layer)
    else:
        effective_depth = float(depth)

    if effective_depth < depth * 0.5:
        issues.append(
            f"Effective depth ({effective_depth:.1f}) much less than "
            f"actual depth ({depth})."
        )
        recommendations.append(
            "Deeper layers contribute little; consider reducing depth or "
            "adding skip connections."
        )

    if not issues:
        recommendations.append("No gradient pathologies detected.")

    return PathologyReport(
        has_vanishing_gradients=has_vanishing,
        has_exploding_gradients=has_exploding,
        gradient_norm_estimate=cumulative_gain,
        effective_depth=effective_depth,
        dead_neuron_fraction=dead_frac,
        spectral_gap=spectral_gap,
        issues=issues,
        recommendations=recommendations,
    )


# ======================================================================
# Aggregate predictor
# ======================================================================

def predict_failures(
    model: Dict[str, Any], config: Dict[str, Any]
) -> List[PredictedFailure]:
    """Run all failure-mode checks and return significant predictions.

    Parameters
    ----------
    model : Dict
        Architecture descriptor with keys ``input_dim``, ``width``,
        ``depth``, ``init_scale``, ``activation``.
    config : Dict
        Training configuration with keys ``lr``, ``epochs``,
        ``batch_size``, ``optimizer``, ``task_type``.

    Returns
    -------
    List[PredictedFailure]
        Failures with probability > 0.1, sorted by probability descending.
    """
    lr = float(config.get("lr", 0.01))
    epochs = int(config.get("epochs", 100))
    failures: List[PredictedFailure] = []

    # -- NaN divergence ------------------------------------------------
    p_nan = nan_risk(model, lr)
    if p_nan > 0.1:
        failures.append(PredictedFailure(
            failure_type="nan_divergence",
            probability=p_nan,
            severity=_severity_from_probability(p_nan),
            description=(
                "Training is likely to diverge to NaN due to the learning "
                "rate exceeding the NTK stability threshold."
            ),
            mitigation="Reduce learning rate or use gradient clipping.",
            affected_epoch_range=(0, min(10, epochs)),
        ))

    # -- Non-convergence -----------------------------------------------
    p_conv = convergence_probability(model, config)
    p_noconv = 1.0 - p_conv
    if p_noconv > 0.1:
        failures.append(PredictedFailure(
            failure_type="non_convergence",
            probability=p_noconv,
            severity=_severity_from_probability(p_noconv),
            description=(
                "The model is unlikely to converge within the given epoch "
                "budget due to a high condition number."
            ),
            mitigation=(
                "Increase epochs, reduce depth, or use an adaptive optimiser."
            ),
            affected_epoch_range=(epochs // 2, epochs),
        ))

    # -- Mode collapse -------------------------------------------------
    p_mc = mode_collapse_risk(model, config)
    if p_mc > 0.1:
        failures.append(PredictedFailure(
            failure_type="mode_collapse",
            probability=p_mc,
            severity=_severity_from_probability(p_mc),
            description=(
                "Generator may collapse to producing limited output "
                "diversity."
            ),
            mitigation=(
                "Increase width, decrease learning rate, or add spectral "
                "normalisation."
            ),
            affected_epoch_range=(epochs // 4, 3 * epochs // 4),
        ))

    # -- Catastrophic forgetting ---------------------------------------
    p_cf = catastrophic_forgetting_risk(model, config)
    if p_cf > 0.1:
        failures.append(PredictedFailure(
            failure_type="catastrophic_forgetting",
            probability=p_cf,
            severity=_severity_from_probability(p_cf),
            description=(
                "Fine-tuning may overwrite pre-trained representations."
            ),
            mitigation=(
                "Lower learning rate, freeze early layers, or use elastic "
                "weight consolidation."
            ),
            affected_epoch_range=(0, epochs // 3),
        ))

    # -- Gradient pathologies ------------------------------------------
    report = gradient_pathology_check(model)
    if report.has_vanishing_gradients:
        failures.append(PredictedFailure(
            failure_type="vanishing_gradients",
            probability=_clamp(1.0 - report.gradient_norm_estimate * 10.0),
            severity="high" if report.gradient_norm_estimate < 1e-4 else "medium",
            description=(
                "Gradients vanish across layers, preventing deep layer "
                "updates."
            ),
            mitigation="Add skip connections or use He initialisation.",
            affected_epoch_range=(0, epochs),
        ))

    if report.has_exploding_gradients:
        failures.append(PredictedFailure(
            failure_type="exploding_gradients",
            probability=_clamp(1.0 - 1.0 / report.gradient_norm_estimate),
            severity="critical" if report.gradient_norm_estimate > 1e4 else "high",
            description="Gradient norms grow exponentially across layers.",
            mitigation="Reduce init_scale or add gradient clipping.",
            affected_epoch_range=(0, min(5, epochs)),
        ))

    # Keep only failures above threshold and sort by probability
    failures = [f for f in failures if f.probability > 0.1]
    failures.sort(key=lambda f: f.probability, reverse=True)
    return failures
