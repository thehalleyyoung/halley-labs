"""Optimize weight initialization using phase-diagram theory.

Computes per-layer initialization scales that place a network in a
desired training regime, detects common initialisation pathologies,
and provides comparative evaluation of standard init schemes.

Example
-------
>>> from phase_diagrams.initialization_optimizer import optimal_init_scale
>>> scales = optimal_init_scale(model, dataset_loader)
>>> print(scales)  # {'layer1.weight': 0.023, 'layer2.weight': 0.018, ...}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .api import Regime

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# ======================================================================
# Data classes
# ======================================================================

class InitMethod(str, Enum):
    KAIMING = "kaiming"
    XAVIER = "xavier"
    ORTHOGONAL = "orthogonal"
    LECUN = "lecun"
    CUSTOM = "custom"


class ProblemSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass
class InitProblem:
    """A detected initialization problem."""
    layer_name: str
    problem: str
    severity: ProblemSeverity
    current_scale: float
    recommended_scale: float
    explanation: str


@dataclass
class InitComparison:
    """Result of comparing initialization strategies."""
    method: str
    per_layer_scales: Dict[str, float]
    predicted_regime: Regime
    gamma: float
    gradient_norm_ratio: float
    score: float
    notes: str = ""


# ======================================================================
# Internal helpers
# ======================================================================

def _fan_in_out(param: Any) -> Tuple[int, int]:
    """Compute fan-in and fan-out for a parameter tensor."""
    shape = param.shape
    if len(shape) < 2:
        return shape[0], shape[0]
    fan_in = shape[1]
    fan_out = shape[0]
    if len(shape) > 2:
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in *= receptive
        fan_out *= receptive
    return fan_in, fan_out


def _compute_scale(fan_in: int, fan_out: int, method: str) -> float:
    """Compute the standard deviation for a given init method."""
    if method == "kaiming":
        return math.sqrt(2.0 / max(fan_in, 1))
    elif method == "xavier":
        return math.sqrt(2.0 / max(fan_in + fan_out, 1))
    elif method == "lecun":
        return math.sqrt(1.0 / max(fan_in, 1))
    elif method == "orthogonal":
        return 1.0  # orthogonal init has unit-norm rows
    return math.sqrt(2.0 / max(fan_in, 1))


def _layer_is_weight(name: str) -> bool:
    return "weight" in name and "norm" not in name.lower() and "bn" not in name.lower()


def _estimate_depth_from_model(model: Any) -> int:
    depth = 0
    for m in model.modules():
        cname = type(m).__name__.lower()
        if any(kw in cname for kw in ("linear", "conv2d", "conv1d")):
            depth += 1
    return max(depth, 1)


def _collect_weight_params(model: Any) -> List[Tuple[str, Any]]:
    """Return list of (name, param) for weight matrices."""
    return [
        (name, p) for name, p in model.named_parameters()
        if _layer_is_weight(name) and p.dim() >= 2
    ]


# ======================================================================
# Public API
# ======================================================================

def optimal_init_scale(
    model: Any,
    dataset: Any = None,
    target_regime: str = "rich",
    lr: float = 0.01,
) -> Dict[str, float]:
    """Compute per-layer optimal initialization scales.

    Uses the phase-diagram relation γ = η·σ²/N to set σ per layer so
    that the effective coupling sits in the desired regime.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    dataset : optional
        Dataset or loader (used to estimate input scale; unused currently).
    target_regime : str
        ``"rich"``, ``"lazy"``, or ``"critical"``.
    lr : float
        Planned learning rate.

    Returns
    -------
    Dict[str, float]
        Mapping from parameter name to recommended init std.
    """
    depth = _estimate_depth_from_model(model)
    weight_params = _collect_weight_params(model)

    scales: Dict[str, float] = {}
    for idx, (name, param) in enumerate(weight_params):
        fan_in, fan_out = _fan_in_out(param)
        width = max(fan_in, fan_out)

        # Target: gamma = lr * sigma^2 / width
        # Rich:    gamma > gamma_star  → sigma^2 > gamma_star * width / lr
        # Lazy:    gamma < gamma_star  → sigma^2 < gamma_star * width / lr
        # Approximate gamma_star ≈ 1 / sqrt(depth)
        gamma_star = 1.0 / math.sqrt(depth)

        if target_regime.lower() == "rich":
            target_gamma = gamma_star * 1.5
        elif target_regime.lower() == "lazy":
            target_gamma = gamma_star * 0.5
        else:
            target_gamma = gamma_star

        sigma_sq = target_gamma * width / max(lr, 1e-10)
        sigma = math.sqrt(max(sigma_sq, 1e-8))

        # Apply depth correction: scale down deeper layers slightly
        depth_factor = 1.0 / (1.0 + 0.1 * idx)
        sigma *= depth_factor

        scales[name] = sigma

    return scales


def compare_initializations(
    model: Any,
    dataset: Any = None,
    inits: Optional[List[str]] = None,
    lr: float = 0.01,
) -> Dict[str, InitComparison]:
    """Compare standard initialization strategies.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    dataset : optional
        Dataset (for future use).
    inits : list of str
        Methods to compare. Defaults to ``["kaiming", "xavier", "orthogonal"]``.
    lr : float
        Planned learning rate.

    Returns
    -------
    Dict[str, InitComparison]
        One entry per init method with predicted regime and score.
    """
    if inits is None:
        inits = ["kaiming", "xavier", "orthogonal"]

    depth = _estimate_depth_from_model(model)
    weight_params = _collect_weight_params(model)
    gamma_star = 1.0 / math.sqrt(max(depth, 1))

    results: Dict[str, InitComparison] = {}
    for method in inits:
        per_layer: Dict[str, float] = {}
        gammas: List[float] = []
        grad_norms: List[float] = []

        for name, param in weight_params:
            fan_in, fan_out = _fan_in_out(param)
            sigma = _compute_scale(fan_in, fan_out, method)
            per_layer[name] = sigma

            width = max(fan_in, fan_out)
            gamma = lr * sigma ** 2 / max(width, 1)
            gammas.append(gamma)

            # Estimate gradient norm ratio (forward signal propagation)
            if method == "orthogonal":
                grad_norms.append(1.0)
            else:
                # sigma^2 * fan_in gives variance of layer output
                output_var = sigma ** 2 * fan_in
                grad_norms.append(output_var)

        avg_gamma = float(np.mean(gammas)) if gammas else 0.0
        ratio = avg_gamma / max(gamma_star, 1e-10)

        if ratio < 0.8:
            regime = Regime.LAZY
        elif ratio > 1.2:
            regime = Regime.RICH
        else:
            regime = Regime.CRITICAL

        # Score: closeness to critical + gradient stability
        grad_ratio = float(np.mean(grad_norms)) if grad_norms else 1.0
        grad_stability = 1.0 / (1.0 + abs(math.log(max(grad_ratio, 1e-10))))
        critical_closeness = 1.0 / (1.0 + abs(ratio - 1.0))
        score = 0.6 * critical_closeness + 0.4 * grad_stability

        notes_parts = []
        if grad_ratio > 2.0:
            notes_parts.append("Gradient explosion risk.")
        if grad_ratio < 0.3:
            notes_parts.append("Gradient vanishing risk.")
        if regime == Regime.LAZY:
            notes_parts.append("Network may underfit (lazy regime).")

        results[method] = InitComparison(
            method=method,
            per_layer_scales=per_layer,
            predicted_regime=regime,
            gamma=avg_gamma,
            gradient_norm_ratio=grad_ratio,
            score=score,
            notes=" ".join(notes_parts),
        )

    return results


def detect_init_problems(model: Any, lr: float = 0.01) -> List[InitProblem]:
    """Detect common initialization pathologies.

    Checks for variance blow-up / collapse, regime mismatch between
    layers, and excessively large or small scales.

    Parameters
    ----------
    model : nn.Module
        PyTorch model (with current weights).
    lr : float
        Planned learning rate.

    Returns
    -------
    List[InitProblem]
    """
    depth = _estimate_depth_from_model(model)
    gamma_star = 1.0 / math.sqrt(max(depth, 1))
    problems: List[InitProblem] = []

    for name, param in model.named_parameters():
        if not _layer_is_weight(name) or param.dim() < 2:
            continue

        if HAS_TORCH:
            current_std = float(param.data.std())
        else:
            current_std = 1.0

        fan_in, fan_out = _fan_in_out(param)
        expected_std = _compute_scale(fan_in, fan_out, "kaiming")
        width = max(fan_in, fan_out)

        # Check variance ratio
        ratio = current_std / max(expected_std, 1e-10)

        if ratio > 5.0:
            problems.append(InitProblem(
                layer_name=name,
                problem="excessive_variance",
                severity=ProblemSeverity.ERROR,
                current_scale=current_std,
                recommended_scale=expected_std,
                explanation=(
                    f"Init std {current_std:.4f} is {ratio:.1f}x the Kaiming "
                    f"recommendation ({expected_std:.4f}). Risk of gradient explosion."
                ),
            ))
        elif ratio > 2.5:
            problems.append(InitProblem(
                layer_name=name,
                problem="high_variance",
                severity=ProblemSeverity.WARNING,
                current_scale=current_std,
                recommended_scale=expected_std,
                explanation=(
                    f"Init std {current_std:.4f} is {ratio:.1f}x normal. "
                    f"May cause unstable early training."
                ),
            ))
        elif ratio < 0.1:
            problems.append(InitProblem(
                layer_name=name,
                problem="near_zero_init",
                severity=ProblemSeverity.ERROR,
                current_scale=current_std,
                recommended_scale=expected_std,
                explanation=(
                    f"Init std {current_std:.6f} is extremely small. "
                    f"Network will behave like a constant function initially."
                ),
            ))
        elif ratio < 0.3:
            problems.append(InitProblem(
                layer_name=name,
                problem="low_variance",
                severity=ProblemSeverity.WARNING,
                current_scale=current_std,
                recommended_scale=expected_std,
                explanation=(
                    f"Init std {current_std:.4f} is {ratio:.1f}x normal. "
                    f"Gradient vanishing likely."
                ),
            ))

        # Check if layer is in a different regime than expected
        gamma = lr * current_std ** 2 / max(width, 1)
        gamma_ratio = gamma / max(gamma_star, 1e-10)
        if gamma_ratio > 3.0:
            problems.append(InitProblem(
                layer_name=name,
                problem="strong_rich_regime",
                severity=ProblemSeverity.WARNING,
                current_scale=current_std,
                recommended_scale=expected_std,
                explanation=(
                    f"Layer γ/γ*={gamma_ratio:.1f} >> 1: deeply in the rich regime. "
                    f"Feature learning may be chaotic at this LR."
                ),
            ))

    return problems


def custom_init(
    model: Any,
    regime: str = "rich",
    lr: float = 0.01,
    inplace: bool = True,
) -> Any:
    """Initialize model weights to target a specific training regime.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    regime : str
        Target regime: ``"rich"``, ``"lazy"``, or ``"critical"``.
    lr : float
        Planned learning rate.
    inplace : bool
        If True modify the model in place, otherwise clone first.

    Returns
    -------
    nn.Module
        The initialised model.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for custom_init")

    if not inplace:
        import copy
        model = copy.deepcopy(model)

    scales = optimal_init_scale(model, target_regime=regime, lr=lr)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in scales:
                sigma = scales[name]
                param.normal_(0.0, sigma)
            elif "bias" in name:
                param.zero_()

    return model
