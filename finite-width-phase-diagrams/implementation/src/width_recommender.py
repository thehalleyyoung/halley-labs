"""Recommend model widths based on phase-diagram theory.

Uses finite-width phase boundaries, scaling laws, and compute-budget
constraints to suggest optimal widths, depth-vs-width trade-offs,
and diminishing-returns thresholds.

Example
-------
>>> from phase_diagrams.width_recommender import recommend_width
>>> rec = recommend_width("image_classification", dataset_size=50000,
...                       compute_budget=1e15)
>>> print(rec.width, rec.regime, rec.explanation)
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
class WidthRecommendation:
    """Result of a width recommendation."""
    width: int
    regime: Regime
    min_width: int
    max_useful_width: int
    compute_flops: float
    explanation: str
    confidence: float = 0.0


@dataclass
class TradeoffPoint:
    """Single point on a width-vs-depth trade-off curve."""
    width: int
    depth: int
    estimated_accuracy: float
    flops: float
    regime: Regime


@dataclass
class TradeoffCurve:
    """Width-vs-depth Pareto frontier."""
    points: List[TradeoffPoint] = field(default_factory=list)
    optimal_idx: int = 0
    arch_family: str = ""
    dataset: str = ""
    target_accuracy: float = 0.0

    @property
    def optimal(self) -> TradeoffPoint:
        return self.points[self.optimal_idx]


# ======================================================================
# Task complexity estimation
# ======================================================================

_TASK_COMPLEXITY: Dict[str, float] = {
    "image_classification": 1.0,
    "object_detection": 2.5,
    "semantic_segmentation": 2.0,
    "text_classification": 0.8,
    "language_modelling": 3.0,
    "machine_translation": 3.5,
    "speech_recognition": 2.2,
    "tabular": 0.4,
    "regression": 0.3,
}

_FAMILY_BASE_FLOPS: Dict[str, float] = {
    "mlp": 2.0,
    "cnn": 4.0,
    "resnet": 5.0,
    "vgg": 6.0,
    "vit": 8.0,
    "gpt": 8.0,
    "bert": 7.5,
}


def _task_complexity(task: str) -> float:
    key = task.lower().strip().replace("-", "_").replace(" ", "_")
    return _TASK_COMPLEXITY.get(key, 1.0)


def _estimate_flops(width: int, depth: int, family: str) -> float:
    """Rough FLOPs estimate: base_factor * width^2 * depth."""
    base = _FAMILY_BASE_FLOPS.get(family.lower(), 4.0)
    return base * (width ** 2) * depth


def _gamma_at(lr: float, width: int) -> float:
    return lr / max(width, 1)


def _regime_at(gamma: float, gamma_star: float) -> Regime:
    ratio = gamma / max(gamma_star, 1e-10)
    if ratio < 0.8:
        return Regime.LAZY
    elif ratio > 1.2:
        return Regime.RICH
    return Regime.CRITICAL


# ======================================================================
# Core estimation: critical width from scaling law
# ======================================================================

def _critical_width(
    task_complexity: float,
    dataset_size: int,
    lr: float = 0.01,
) -> int:
    """Estimate the critical width separating lazy from rich regimes.

    Derived from the scaling relation  N* ∝ η · σ² · sqrt(n) / C_task
    where n = dataset size and C_task is a task complexity factor.
    """
    # Empirical fit: N* ~ lr * sqrt(dataset_size) / task_complexity * scale
    scale = 500.0
    n_star = lr * math.sqrt(dataset_size) / max(task_complexity, 0.1) * scale
    return max(16, int(round(n_star)))


def _diminishing_width(
    task_complexity: float,
    dataset_size: int,
) -> int:
    """Width beyond which additional neurons yield negligible gains.

    Uses the empirical law  N_dim ≈ 4 · N* · log(n / 1000).
    """
    n_star = _critical_width(task_complexity, dataset_size)
    log_factor = max(1.0, math.log(max(dataset_size, 1000) / 1000.0))
    return max(n_star, int(round(4.0 * n_star * log_factor)))


# ======================================================================
# Public API
# ======================================================================

def recommend_width(
    task: str,
    dataset_size: int,
    compute_budget: float = 1e16,
    lr: float = 0.01,
    arch_family: str = "resnet",
    target_regime: str = "rich",
) -> WidthRecommendation:
    """Recommend network width for a given task and budget.

    Parameters
    ----------
    task : str
        Task name (e.g. ``"image_classification"``).
    dataset_size : int
        Number of training samples.
    compute_budget : float
        Maximum training FLOPs.
    lr : float
        Planned learning rate.
    arch_family : str
        Architecture family for FLOPs estimate.
    target_regime : str
        Desired regime: ``"rich"``, ``"lazy"``, or ``"critical"``.

    Returns
    -------
    WidthRecommendation
    """
    tc = _task_complexity(task)
    n_star = _critical_width(tc, dataset_size, lr)
    n_dim = _diminishing_width(tc, dataset_size)

    # Choose width based on target regime
    if target_regime.lower() == "lazy":
        width = int(n_star * 2.0)
    elif target_regime.lower() == "critical":
        width = n_star
    else:
        width = max(16, int(n_star * 0.6))

    # Enforce compute budget: width^2 * depth * base < budget
    depth_guess = 20
    max_width_budget = int(math.sqrt(compute_budget / max(_estimate_flops(1, depth_guess, arch_family), 1.0)))
    width = min(width, max_width_budget)
    width = max(16, width)

    gamma = _gamma_at(lr, width)
    gamma_star = lr / max(n_star, 1)
    regime = _regime_at(gamma, gamma_star)
    flops = _estimate_flops(width, depth_guess, arch_family)

    explanation_parts = [
        f"Critical width N*≈{n_star} for task_complexity={tc:.1f}, n={dataset_size}.",
        f"Diminishing-returns width ≈{n_dim}.",
        f"Recommended width={width} targets '{target_regime}' regime (actual: {regime.value}).",
    ]
    if width >= n_dim:
        explanation_parts.append("Warning: width exceeds diminishing-returns threshold.")
    if flops > compute_budget:
        explanation_parts.append("Warning: estimated FLOPs exceed budget; consider reducing depth.")

    return WidthRecommendation(
        width=width,
        regime=regime,
        min_width=max(16, n_star // 2),
        max_useful_width=n_dim,
        compute_flops=flops,
        explanation=" ".join(explanation_parts),
        confidence=min(1.0, dataset_size / 10000.0),
    )


def width_vs_depth_tradeoff(
    arch_family: str,
    dataset: str,
    target_acc: float = 0.90,
    dataset_size: int = 50_000,
    lr: float = 0.01,
    max_flops: float = 1e16,
) -> TradeoffCurve:
    """Compute a Pareto-optimal width-vs-depth trade-off curve.

    Parameters
    ----------
    arch_family : str
        Architecture family name.
    dataset : str
        Dataset identifier (used for metadata).
    target_acc : float
        Target accuracy (0–1).
    dataset_size : int
        Training set size.
    lr : float
        Learning rate.
    max_flops : float
        Maximum allowed FLOPs.

    Returns
    -------
    TradeoffCurve
    """
    tc = _task_complexity(dataset)
    n_star = _critical_width(tc, dataset_size, lr)

    # Scan a grid of (width, depth) and estimate accuracy via a simple
    # scaling model: acc ~ 1 - C / (width * depth)^alpha
    alpha = 0.3
    c_scale = target_acc * (n_star * 20) ** alpha

    depths = [4, 8, 12, 16, 20, 24, 32, 48, 64]
    widths = [32, 64, 128, 256, 512, 768, 1024, 1536, 2048]

    candidates: List[TradeoffPoint] = []
    for d in depths:
        for w in widths:
            flops = _estimate_flops(w, d, arch_family)
            if flops > max_flops:
                continue
            wd = max(w * d, 1)
            acc = 1.0 - c_scale / (wd ** alpha)
            acc = float(np.clip(acc, 0.0, 1.0))

            gamma = _gamma_at(lr, w)
            gamma_star = lr / max(n_star, 1)
            regime = _regime_at(gamma, gamma_star)

            candidates.append(TradeoffPoint(
                width=w, depth=d,
                estimated_accuracy=acc,
                flops=flops, regime=regime,
            ))

    # Pareto filter: keep points not dominated in both accuracy and flops
    pareto: List[TradeoffPoint] = []
    for pt in sorted(candidates, key=lambda p: p.flops):
        if not pareto or pt.estimated_accuracy > pareto[-1].estimated_accuracy:
            pareto.append(pt)

    # Pick the cheapest point meeting target accuracy
    optimal_idx = 0
    for i, pt in enumerate(pareto):
        if pt.estimated_accuracy >= target_acc:
            optimal_idx = i
            break
    else:
        optimal_idx = len(pareto) - 1 if pareto else 0

    return TradeoffCurve(
        points=pareto,
        optimal_idx=optimal_idx,
        arch_family=arch_family,
        dataset=dataset,
        target_accuracy=target_acc,
    )


def minimum_width_for_task(
    task_complexity: float,
    dataset_size: int,
    lr: float = 0.01,
) -> int:
    """Return the minimum width for meaningful learning on a task.

    Below this width the network is in a heavily under-parameterised
    regime and is unlikely to fit the data.

    Parameters
    ----------
    task_complexity : float
        Scalar complexity (use ``_task_complexity(task_name)`` for lookup).
    dataset_size : int
        Training set size.
    lr : float
        Planned learning rate.

    Returns
    -------
    int
        Minimum recommended width.
    """
    n_star = _critical_width(task_complexity, dataset_size, lr)
    # Minimum useful width is roughly N* / 3
    return max(16, n_star // 3)


def diminishing_returns_width(
    model_family: str,
    dataset: str,
    dataset_size: int = 50_000,
) -> int:
    """Return the width beyond which added neurons yield negligible gains.

    Parameters
    ----------
    model_family : str
        Architecture family (e.g. ``"resnet"``).
    dataset : str
        Dataset / task name.
    dataset_size : int
        Training set size.

    Returns
    -------
    int
        Width threshold.
    """
    tc = _task_complexity(dataset)
    return _diminishing_width(tc, dataset_size)
