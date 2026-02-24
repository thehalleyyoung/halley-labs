"""Phase analysis for transfer learning and fine-tuning.

Determines what training regime a pretrained model sits in when
adapted to a new dataset, recommends fine-tuning hyperparameters
based on phase boundaries, and provides per-layer regime maps.

Example
-------
>>> from phase_diagrams.transfer_phase import pretrained_regime
>>> regime = pretrained_regime(model, new_dataset_loader)
>>> print(regime)  # "lazy" / "rich" / "critical"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

@dataclass
class PhaseShift:
    """Describes how a model's phase changes under distribution shift."""
    old_regime: Regime
    new_regime: Regime
    gamma_old: float
    gamma_new: float
    gamma_star: float
    shift_magnitude: float
    per_layer_shifts: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class LayerRegimeInfo:
    """Regime classification for a single layer."""
    name: str
    regime: Regime
    gamma: float
    gamma_star: float
    width: int
    effective_lr: float
    confidence: float


# ======================================================================
# Internal helpers
# ======================================================================

def _weight_layers(model: Any) -> List[Tuple[str, Any]]:
    """Extract named weight parameters (≥ 2-D, excluding norms)."""
    results = []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "weight" in name:
            low = name.lower()
            if "norm" not in low and "bn" not in low and "ln" not in low:
                results.append((name, p))
    return results


def _fan_in(param: Any) -> int:
    shape = param.shape
    fan = shape[1] if len(shape) >= 2 else shape[0]
    if len(shape) > 2:
        for s in shape[2:]:
            fan *= s
    return int(fan)


def _param_std(param: Any) -> float:
    if HAS_TORCH:
        return float(param.data.std())
    return 1.0


def _estimate_gamma(std: float, lr: float, width: int) -> float:
    return lr * std ** 2 / max(width, 1)


def _gamma_star_for_depth(depth: int) -> float:
    return 1.0 / math.sqrt(max(depth, 1))


def _classify(gamma: float, gamma_star: float) -> Tuple[Regime, float]:
    ratio = gamma / max(gamma_star, 1e-10)
    if ratio < 0.8:
        return Regime.LAZY, min(1.0, (0.8 - ratio) / 0.8)
    elif ratio > 1.2:
        return Regime.RICH, min(1.0, (ratio - 1.2) / ratio)
    return Regime.CRITICAL, 1.0 - abs(ratio - 1.0) / 0.2


def _model_depth(model: Any) -> int:
    d = 0
    for m in model.modules():
        cn = type(m).__name__.lower()
        if any(k in cn for k in ("linear", "conv2d", "conv1d")):
            d += 1
    return max(d, 1)


def _dataset_scale(dataset: Any) -> float:
    """Estimate a rough input-variance scale from a dataset or loader."""
    try:
        # If it's a DataLoader, grab one batch
        batch = next(iter(dataset))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if HAS_TORCH:
            return float(x.float().std())
        return float(np.std(np.array(x)))
    except Exception:
        return 1.0


# ======================================================================
# Public API
# ======================================================================

def pretrained_regime(
    model: Any,
    new_dataset: Any = None,
    lr: float = 0.001,
) -> str:
    """Determine the training regime of a pretrained model on new data.

    Parameters
    ----------
    model : nn.Module
        Pretrained PyTorch model.
    new_dataset : optional
        New dataset or data loader (used to estimate input scale).
    lr : float
        Fine-tuning learning rate.

    Returns
    -------
    str
        ``"lazy"``, ``"rich"``, or ``"critical"``.
    """
    depth = _model_depth(model)
    gamma_star = _gamma_star_for_depth(depth)
    data_scale = _dataset_scale(new_dataset) if new_dataset is not None else 1.0

    layers = _weight_layers(model)
    if not layers:
        return Regime.LAZY.value

    gammas = []
    for name, param in layers:
        std = _param_std(param) * data_scale
        width = _fan_in(param)
        gammas.append(_estimate_gamma(std, lr, width))

    avg_gamma = float(np.mean(gammas))
    regime, _ = _classify(avg_gamma, gamma_star)
    return regime.value


def fine_tuning_phase_shift(
    model: Any,
    old_data: Any,
    new_data: Any,
    lr: float = 0.001,
) -> PhaseShift:
    """Analyse how the model's phase changes between two datasets.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    old_data, new_data
        Data loaders or datasets for the source and target domains.
    lr : float
        Fine-tuning learning rate.

    Returns
    -------
    PhaseShift
    """
    depth = _model_depth(model)
    gamma_star = _gamma_star_for_depth(depth)

    scale_old = _dataset_scale(old_data)
    scale_new = _dataset_scale(new_data)

    layers = _weight_layers(model)
    gammas_old, gammas_new = [], []
    per_layer_shifts: Dict[str, float] = {}

    for name, param in layers:
        std = _param_std(param)
        width = _fan_in(param)
        g_old = _estimate_gamma(std * scale_old, lr, width)
        g_new = _estimate_gamma(std * scale_new, lr, width)
        gammas_old.append(g_old)
        gammas_new.append(g_new)
        per_layer_shifts[name] = g_new - g_old

    avg_old = float(np.mean(gammas_old)) if gammas_old else 0.0
    avg_new = float(np.mean(gammas_new)) if gammas_new else 0.0

    old_regime, _ = _classify(avg_old, gamma_star)
    new_regime, _ = _classify(avg_new, gamma_star)

    shift_mag = abs(avg_new - avg_old) / max(gamma_star, 1e-10)

    rec_parts = []
    if old_regime != new_regime:
        rec_parts.append(
            f"Phase shift detected: {old_regime.value} → {new_regime.value}."
        )
        if new_regime == Regime.RICH:
            rec_parts.append("Reduce LR or increase width to stabilise.")
        elif new_regime == Regime.LAZY:
            rec_parts.append("Increase LR or decrease width for richer features.")
    else:
        rec_parts.append(f"Regime stays {old_regime.value}; fine-tuning should be stable.")

    return PhaseShift(
        old_regime=old_regime,
        new_regime=new_regime,
        gamma_old=avg_old,
        gamma_new=avg_new,
        gamma_star=gamma_star,
        shift_magnitude=shift_mag,
        per_layer_shifts=per_layer_shifts,
        recommendation=" ".join(rec_parts),
    )


def optimal_fine_tuning_lr(
    model: Any,
    new_data: Any = None,
    target_regime: str = "critical",
) -> float:
    """Recommend a fine-tuning learning rate based on phase analysis.

    Finds the LR that places the network at the boundary between
    lazy and rich regimes (or in the specified regime).

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    new_data : optional
        Target dataset or loader.
    target_regime : str
        Desired regime.

    Returns
    -------
    float
        Recommended learning rate.
    """
    depth = _model_depth(model)
    gamma_star = _gamma_star_for_depth(depth)
    data_scale = _dataset_scale(new_data) if new_data is not None else 1.0

    layers = _weight_layers(model)
    if not layers:
        return 1e-3

    # For each layer solve: lr * (std * data_scale)^2 / width = target_gamma
    if target_regime.lower() == "rich":
        target_gamma = gamma_star * 1.5
    elif target_regime.lower() == "lazy":
        target_gamma = gamma_star * 0.5
    else:
        target_gamma = gamma_star

    lr_per_layer = []
    for name, param in layers:
        std = _param_std(param) * data_scale
        width = _fan_in(param)
        # lr = target_gamma * width / std^2
        sigma_sq = std ** 2
        if sigma_sq > 1e-12:
            lr_per_layer.append(target_gamma * width / sigma_sq)

    if not lr_per_layer:
        return 1e-3

    # Use geometric mean to balance across layers
    log_lrs = [math.log(max(lr, 1e-12)) for lr in lr_per_layer]
    recommended = math.exp(float(np.mean(log_lrs)))

    # Clip to a sane range
    return float(np.clip(recommended, 1e-6, 1.0))


def layer_wise_regime(
    model: Any,
    dataset: Any = None,
    lr: float = 0.001,
) -> Dict[str, str]:
    """Classify each layer of a model into its training regime.

    Parameters
    ----------
    model : nn.Module
        PyTorch model with current weights.
    dataset : optional
        Dataset or loader for input scale estimation.
    lr : float
        Learning rate.

    Returns
    -------
    Dict[str, str]
        Mapping from layer name to regime string.
    """
    depth = _model_depth(model)
    gamma_star = _gamma_star_for_depth(depth)
    data_scale = _dataset_scale(dataset) if dataset is not None else 1.0

    regimes: Dict[str, str] = {}
    for name, param in _weight_layers(model):
        std = _param_std(param) * data_scale
        width = _fan_in(param)
        gamma = _estimate_gamma(std, lr, width)
        regime, _ = _classify(gamma, gamma_star)
        regimes[name] = regime.value

    return regimes
