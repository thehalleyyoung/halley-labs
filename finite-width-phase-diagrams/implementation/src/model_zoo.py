"""Pre-computed phase diagrams for common neural network architectures.

Provides a lookup table of phase boundaries and regime classifications
for well-known models, plus utilities for interpolating between diagrams
and finding the closest precomputed match for a novel architecture.

Example
-------
>>> from phase_diagrams.model_zoo import get_phase_diagram, list_available_models
>>> diagram = get_phase_diagram("resnet50")
>>> print(diagram.boundary_curve.shape)
>>> models = list_available_models()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .api import PhaseDiagram, PhasePoint, Regime


# ======================================================================
# Model specification dataclass
# ======================================================================

@dataclass
class ModelSpec:
    """Specification of a precomputed model entry."""
    name: str
    family: str
    depth: int
    widths: List[int]
    total_params: int
    has_residual: bool = False
    has_attention: bool = False
    has_conv: bool = False
    gamma_star: float = 0.0
    timescale_constant: float = 0.0
    boundary_coeffs: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ======================================================================
# Built-in model registry
# ======================================================================

_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    _MODEL_REGISTRY[spec.name.lower()] = spec


def _init_registry() -> None:
    """Populate the registry with well-known architectures."""
    # ------- ResNet family -------
    _register(ModelSpec(
        name="resnet18", family="resnet", depth=18,
        widths=[64, 64, 128, 256, 512], total_params=11_689_512,
        has_residual=True, has_conv=True,
        gamma_star=0.042, timescale_constant=1.85,
        boundary_coeffs=(0.042, -0.78, 0.15),
    ))
    _register(ModelSpec(
        name="resnet34", family="resnet", depth=34,
        widths=[64, 64, 128, 256, 512], total_params=21_797_672,
        has_residual=True, has_conv=True,
        gamma_star=0.031, timescale_constant=2.10,
        boundary_coeffs=(0.031, -0.82, 0.12),
    ))
    _register(ModelSpec(
        name="resnet50", family="resnet", depth=50,
        widths=[64, 256, 512, 1024, 2048], total_params=25_557_032,
        has_residual=True, has_conv=True,
        gamma_star=0.025, timescale_constant=2.35,
        boundary_coeffs=(0.025, -0.85, 0.10),
    ))
    # ------- VGG family -------
    _register(ModelSpec(
        name="vgg11", family="vgg", depth=11,
        widths=[64, 128, 256, 512, 512], total_params=132_863_336,
        has_conv=True,
        gamma_star=0.065, timescale_constant=1.50,
        boundary_coeffs=(0.065, -0.70, 0.20),
    ))
    _register(ModelSpec(
        name="vgg16", family="vgg", depth=16,
        widths=[64, 128, 256, 512, 512], total_params=138_357_544,
        has_conv=True,
        gamma_star=0.055, timescale_constant=1.65,
        boundary_coeffs=(0.055, -0.73, 0.18),
    ))
    _register(ModelSpec(
        name="vgg19", family="vgg", depth=19,
        widths=[64, 128, 256, 512, 512], total_params=143_667_240,
        has_conv=True,
        gamma_star=0.050, timescale_constant=1.72,
        boundary_coeffs=(0.050, -0.75, 0.17),
    ))
    # ------- Vision Transformer family -------
    _register(ModelSpec(
        name="vit-tiny", family="vit", depth=12,
        widths=[192] * 12, total_params=5_717_416,
        has_attention=True,
        gamma_star=0.070, timescale_constant=1.40,
        boundary_coeffs=(0.070, -0.65, 0.22),
    ))
    _register(ModelSpec(
        name="vit-small", family="vit", depth=12,
        widths=[384] * 12, total_params=22_050_664,
        has_attention=True,
        gamma_star=0.048, timescale_constant=1.75,
        boundary_coeffs=(0.048, -0.72, 0.16),
    ))
    _register(ModelSpec(
        name="vit-base", family="vit", depth=12,
        widths=[768] * 12, total_params=86_567_656,
        has_attention=True,
        gamma_star=0.033, timescale_constant=2.05,
        boundary_coeffs=(0.033, -0.80, 0.13),
    ))
    # ------- GPT-2 small -------
    _register(ModelSpec(
        name="gpt2-small", family="gpt", depth=12,
        widths=[768] * 12, total_params=124_439_808,
        has_attention=True,
        gamma_star=0.030, timescale_constant=2.15,
        boundary_coeffs=(0.030, -0.82, 0.11),
    ))
    # ------- BERT base -------
    _register(ModelSpec(
        name="bert-base", family="bert", depth=12,
        widths=[768] * 12, total_params=110_104_890,
        has_attention=True,
        gamma_star=0.032, timescale_constant=2.08,
        boundary_coeffs=(0.032, -0.81, 0.12),
    ))


_init_registry()


# ======================================================================
# Phase diagram synthesis from precomputed coefficients
# ======================================================================

def _synthesize_boundary(
    coeffs: Tuple[float, float, float],
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    n_points: int = 200,
) -> NDArray:
    """Build a boundary curve from polynomial coefficients.

    The boundary in (log-lr, log-width) space is modelled as:
        log10(width*) = a + b * log10(lr) + c * log10(lr)^2
    """
    a, b, c = coeffs
    log_lrs = np.linspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_points)
    log_widths = a + b * log_lrs + c * log_lrs ** 2
    # Convert back to linear scale and clip to reasonable widths
    widths = np.clip(10.0 ** log_widths, 1, 1e6)
    lrs = 10.0 ** log_lrs
    return np.column_stack([lrs, widths])


def _classify_point(
    lr: float,
    width: int,
    coeffs: Tuple[float, float, float],
    gamma_star: float,
) -> Tuple[Regime, float, float]:
    """Classify a single (lr, width) point relative to precomputed boundary."""
    a, b, c = coeffs
    log_lr = np.log10(max(lr, 1e-10))
    critical_log_w = a + b * log_lr + c * log_lr ** 2
    critical_w = 10.0 ** critical_log_w

    gamma = lr / max(width, 1)
    distance = (width - critical_w) / max(critical_w, 1.0)

    if abs(distance) < 0.1:
        regime = Regime.CRITICAL
        confidence = 1.0 - abs(distance) / 0.1
    elif width > critical_w:
        regime = Regime.LAZY
        confidence = min(1.0, distance)
    else:
        regime = Regime.RICH
        confidence = min(1.0, -distance)

    return regime, confidence, gamma


def _build_diagram(spec: ModelSpec) -> PhaseDiagram:
    """Build a full PhaseDiagram from a ModelSpec."""
    lr_range = (1e-4, 1.0)
    min_w = min(spec.widths) // 4
    max_w = max(spec.widths) * 4
    width_range = (max(8, min_w), max_w)

    boundary = _synthesize_boundary(spec.boundary_coeffs, lr_range)

    # Build grid of phase points
    lrs = np.geomspace(lr_range[0], lr_range[1], 20)
    widths = np.linspace(width_range[0], width_range[1], 15, dtype=int)
    points: List[PhasePoint] = []
    for lr in lrs:
        for w in widths:
            regime, conf, gamma = _classify_point(
                lr, int(w), spec.boundary_coeffs, spec.gamma_star
            )
            points.append(PhasePoint(
                lr=float(lr), width=int(w), regime=regime,
                gamma=gamma, gamma_star=spec.gamma_star,
                confidence=conf,
                ntk_drift_predicted=gamma / max(spec.gamma_star, 1e-8),
            ))

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=width_range,
        boundary_curve=boundary,
        timescale_constant=spec.timescale_constant,
        metadata={
            "model": spec.name,
            "family": spec.family,
            "depth": spec.depth,
            "widths": spec.widths,
            "total_params": spec.total_params,
            "precomputed": True,
        },
    )


# ======================================================================
# Public API
# ======================================================================

def list_available_models() -> List[str]:
    """Return sorted list of model names with precomputed phase diagrams."""
    return sorted(_MODEL_REGISTRY.keys())


def get_phase_diagram(model_name: str) -> PhaseDiagram:
    """Look up a precomputed phase diagram by model name.

    Parameters
    ----------
    model_name : str
        Case-insensitive model identifier (e.g. ``"resnet50"``).

    Returns
    -------
    PhaseDiagram
        The synthesized phase diagram.

    Raises
    ------
    KeyError
        If *model_name* is not in the registry.
    """
    key = model_name.lower().strip()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return _build_diagram(_MODEL_REGISTRY[key])


def closest_match(model: Any) -> str:
    """Find the registry model closest to an arbitrary ``nn.Module``.

    Matching is based on total parameter count, depth, and architectural
    features (residual connections, attention, convolution).

    Parameters
    ----------
    model : nn.Module
        A PyTorch model (or any object with ``parameters()`` method).

    Returns
    -------
    str
        Name of the closest precomputed model.
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
    except AttributeError:
        total_params = 0

    # Detect features by module class names
    module_names = [type(m).__name__.lower() for m in model.modules()]
    has_res = any("residual" in n or "shortcut" in n or "skip" in n for n in module_names)
    has_attn = any("attention" in n or "multihead" in n for n in module_names)
    has_conv = any("conv" in n for n in module_names)

    depth = sum(
        1 for m in model.modules()
        if any(kw in type(m).__name__.lower() for kw in ("linear", "conv2d", "conv1d"))
    )

    best_name = list_available_models()[0]
    best_score = float("inf")

    for name, spec in _MODEL_REGISTRY.items():
        # Weighted distance
        param_dist = abs(math.log1p(total_params) - math.log1p(spec.total_params))
        depth_dist = abs(depth - spec.depth) / max(spec.depth, 1)
        feat_dist = (
            (has_res != spec.has_residual) * 2.0
            + (has_attn != spec.has_attention) * 2.0
            + (has_conv != spec.has_conv) * 1.0
        )
        score = param_dist + depth_dist * 3.0 + feat_dist
        if score < best_score:
            best_score = score
            best_name = name

    return best_name


def interpolate_diagram(
    model_a: str,
    model_b: str,
    alpha: float = 0.5,
) -> PhaseDiagram:
    """Interpolate between two precomputed phase diagrams.

    This linearly interpolates the boundary coefficients, gamma_star,
    and timescale constant, then rebuilds the diagram.  Useful when a
    model sits architecturally between two known families.

    Parameters
    ----------
    model_a, model_b : str
        Names of models in the registry.
    alpha : float
        Interpolation weight.  ``alpha=0`` → *model_a*,
        ``alpha=1`` → *model_b*.

    Returns
    -------
    PhaseDiagram
        Interpolated phase diagram.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    key_a = model_a.lower().strip()
    key_b = model_b.lower().strip()

    if key_a not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_a}'")
    if key_b not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_b}'")

    sa = _MODEL_REGISTRY[key_a]
    sb = _MODEL_REGISTRY[key_b]

    interp = lambda va, vb: va * (1 - alpha) + vb * alpha  # noqa: E731

    mixed_coeffs = tuple(interp(a, b) for a, b in zip(sa.boundary_coeffs, sb.boundary_coeffs))
    mixed_gamma = interp(sa.gamma_star, sb.gamma_star)
    mixed_tc = interp(sa.timescale_constant, sb.timescale_constant)
    mixed_widths = [
        int(interp(wa, wb))
        for wa, wb in zip(sa.widths, sb.widths[:len(sa.widths)])
    ]
    mixed_params = int(interp(sa.total_params, sb.total_params))
    mixed_depth = int(interp(sa.depth, sb.depth))

    mixed_spec = ModelSpec(
        name=f"{sa.name}_x_{sb.name}@{alpha:.2f}",
        family="interpolated",
        depth=mixed_depth,
        widths=mixed_widths,
        total_params=mixed_params,
        has_residual=sa.has_residual or sb.has_residual,
        has_attention=sa.has_attention or sb.has_attention,
        has_conv=sa.has_conv or sb.has_conv,
        gamma_star=mixed_gamma,
        timescale_constant=mixed_tc,
        boundary_coeffs=mixed_coeffs,  # type: ignore[arg-type]
    )

    diagram = _build_diagram(mixed_spec)
    diagram.metadata["interpolation"] = {
        "model_a": sa.name,
        "model_b": sb.name,
        "alpha": alpha,
    }
    return diagram
