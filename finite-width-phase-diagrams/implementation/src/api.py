"""High-level API for finite-width phase diagram prediction.

Provides the primary user-facing interface for predicting lazy-to-rich
learning transitions without training. Wraps the internal kernel engine,
correction fitting, and bifurcation analysis into simple function calls.

Example
-------
>>> from phase_diagrams.api import detect_regime, recommend_training_regime
>>> regime = detect_regime(model, dataset, lr=0.01)
>>> print(regime)  # "lazy" or "rich"
>>> rec = recommend_training_regime(model, dataset)
>>> print(rec.recommended_lr, rec.regime, rec.explanation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Data classes
# ======================================================================

class Regime(str, Enum):
    """Training regime classification."""
    LAZY = "lazy"
    RICH = "rich"
    CRITICAL = "critical"


@dataclass
class PhasePoint:
    """A single point in phase-diagram parameter space.

    Attributes
    ----------
    lr : float
        Learning rate at this point.
    width : int
        Network width at this point.
    regime : Regime
        Predicted regime (lazy / rich / critical).
    gamma : float
        Effective coupling γ = η·σ²/N (or equivalent parameterisation).
    gamma_star : float
        Critical coupling at this width.
    confidence : float
        Confidence in the regime classification (0–1).
    ntk_drift_predicted : float
        Predicted NTK drift δ(T) at this configuration.
    """
    lr: float
    width: int
    regime: Regime
    gamma: float = 0.0
    gamma_star: float = 0.0
    confidence: float = 0.0
    ntk_drift_predicted: float = 0.0


@dataclass
class PhaseDiagram:
    """Complete phase diagram over a grid of (lr, width) values.

    Attributes
    ----------
    points : list of PhasePoint
        All evaluated points in the diagram.
    lr_range : tuple of float
        (min_lr, max_lr) scanned.
    width_range : tuple of int
        (min_width, max_width) scanned.
    boundary_curve : NDArray
        Array of shape (K, 2) giving (lr, width) pairs on the phase boundary.
    timescale_constant : float
        Fitted T·γ* constant from the timescale law.
    metadata : dict
        Arbitrary metadata (architecture, dataset, etc.).
    """
    points: List[PhasePoint] = field(default_factory=list)
    lr_range: Tuple[float, float] = (1e-4, 1.0)
    width_range: Tuple[int, int] = (32, 2048)
    boundary_curve: Optional[NDArray] = None
    timescale_constant: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ---------------------------------------------------

    def query(self, lr: float, width: int) -> PhasePoint:
        """Find the nearest evaluated point to (lr, width)."""
        best, best_dist = None, float("inf")
        for p in self.points:
            d = (math.log(p.lr) - math.log(lr)) ** 2 + (
                math.log(p.width) - math.log(width)
            ) ** 2
            if d < best_dist:
                best, best_dist = p, d
        if best is None:
            raise ValueError("Phase diagram has no evaluated points.")
        return best

    @property
    def lazy_points(self) -> List[PhasePoint]:
        return [p for p in self.points if p.regime == Regime.LAZY]

    @property
    def rich_points(self) -> List[PhasePoint]:
        return [p for p in self.points if p.regime == Regime.RICH]

    def boundary_at_width(self, width: int) -> float:
        """Interpolate critical LR at a given width from the boundary curve."""
        if self.boundary_curve is None or len(self.boundary_curve) == 0:
            raise ValueError("No boundary curve computed.")
        lrs = self.boundary_curve[:, 0]
        widths = self.boundary_curve[:, 1]
        return float(np.interp(width, widths, lrs))


@dataclass
class TrainingRecommendation:
    """Actionable training recommendation from phase analysis.

    Attributes
    ----------
    recommended_lr : float
        Suggested learning rate.
    regime : Regime
        Regime the model will enter at recommended_lr.
    critical_lr : float
        The phase boundary learning rate.
    explanation : str
        Human-readable explanation.
    warmup_steps : int
        Recommended LR warmup steps (0 if none needed).
    init_scale : float
        Recommended initialisation scale σ.
    """
    recommended_lr: float
    regime: Regime
    critical_lr: float
    explanation: str = ""
    warmup_steps: int = 0
    init_scale: float = 1.0


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_mlp_params(
    weights: List[NDArray],
) -> Tuple[int, int, int, float]:
    """Extract (input_dim, width, depth, init_scale) from weight matrices."""
    depth = len(weights)
    input_dim = weights[0].shape[1] if weights[0].ndim == 2 else weights[0].shape[0]
    width = weights[0].shape[0] if weights[0].ndim == 2 else weights[0].shape[1]
    # estimate init scale from first layer
    init_scale = float(np.std(weights[0]) * math.sqrt(width))
    return input_dim, width, depth, init_scale


def _compute_gamma(lr: float, init_scale: float, width: int) -> float:
    """Effective coupling γ = η·σ²/N in standard parameterisation."""
    return lr * init_scale ** 2 / width


def _predict_gamma_star(
    mu_max_eff: float,
    training_steps: int,
    drift_threshold: float = 0.1,
    drift_floor: float = 1e-3,
) -> float:
    """Critical coupling from bifurcation analysis.

    γ*(T) = log(δ_thresh / δ_floor) / (T · μ_max_eff)
    """
    if mu_max_eff <= 0:
        return float("inf")
    c = math.log(drift_threshold / drift_floor)
    return c / (training_steps * mu_max_eff)


def _ntk_eigenspectrum_mlp(
    input_dim: int,
    width: int,
    depth: int,
    n_samples: int = 50,
    seed: int = 42,
) -> NDArray:
    """Approximate NTK eigenspectrum for an MLP via random feature model.

    Returns eigenvalues of the empirical NTK Gram matrix.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    # Build random feature Jacobian for an MLP
    h = X.copy()
    jacobian_blocks = []
    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = width if l < depth - 1 else 1
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        pre = h @ W
        jacobian_blocks.append(
            np.kron(np.eye(fan_out), h) if fan_out <= 16
            else h  # memory-efficient fallback
        )
        h = np.maximum(pre, 0) if l < depth - 1 else pre

    # Gram matrix as proxy
    K = h @ h.T
    eigvals = np.linalg.eigvalsh(K)
    return np.sort(eigvals)[::-1]


def _compute_mu_max_eff(
    input_dim: int,
    width: int,
    depth: int,
    n_samples: int = 50,
    seed: int = 42,
) -> float:
    """Effective perturbation eigenvalue μ_max for bifurcation analysis."""
    eigs = _ntk_eigenspectrum_mlp(input_dim, width, depth, n_samples, seed)
    # 1/N perturbation spectral radius approximation
    mu_max = float(eigs[0]) if len(eigs) > 0 else 1.0
    return mu_max / width


# ======================================================================
# Public API
# ======================================================================

def compute_phase_diagram(
    weights: List[NDArray],
    dataset: NDArray,
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    width_range: Optional[Tuple[int, int]] = None,
    n_lr_steps: int = 30,
    n_width_steps: int = 10,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram over learning rate and width.

    Parameters
    ----------
    weights : list of NDArray
        Weight matrices of the network (one per layer). Used to infer
        architecture parameters and initialisation scale.
    dataset : NDArray of shape (n_samples, input_dim)
        Input data (labels not required).
    lr_range : (float, float)
        Min and max learning rate (log-spaced scan).
    width_range : (int, int) or None
        Min and max width to scan.  If None, inferred from weights.
    n_lr_steps : int
        Number of LR grid points.
    n_width_steps : int
        Number of width grid points.
    training_steps : int
        Assumed training duration T.
    seed : int
        Random seed for NTK approximation.

    Returns
    -------
    PhaseDiagram
        Complete phase diagram with boundary curve.
    """
    input_dim, w0, depth, init_scale = _extract_mlp_params(weights)

    if width_range is None:
        width_range = (max(16, w0 // 4), w0 * 4)

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    widths = np.unique(
        np.logspace(
            math.log10(width_range[0]),
            math.log10(width_range[1]),
            n_width_steps,
        ).astype(int)
    )

    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    for w in widths:
        mu_max = _compute_mu_max_eff(input_dim, int(w), depth, len(dataset), seed)
        g_star = _predict_gamma_star(mu_max, training_steps)

        prev_regime = None
        for lr in lrs:
            g = _compute_gamma(lr, init_scale, int(w))
            if g < g_star * 0.8:
                regime = Regime.LAZY
                conf = min(1.0, (g_star - g) / g_star)
            elif g > g_star * 1.2:
                regime = Regime.RICH
                conf = min(1.0, (g - g_star) / g_star)
            else:
                regime = Regime.CRITICAL
                conf = 0.5

            drift = g * mu_max * training_steps
            pt = PhasePoint(
                lr=float(lr),
                width=int(w),
                regime=regime,
                gamma=g,
                gamma_star=g_star,
                confidence=conf,
                ntk_drift_predicted=drift,
            )
            points.append(pt)

            if prev_regime is not None and prev_regime != regime:
                boundary_pts.append((float(lr), int(w)))
            prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None

    # Fit timescale constant T·γ*
    gamma_stars = [_predict_gamma_star(
        _compute_mu_max_eff(input_dim, int(w), depth, len(dataset), seed),
        training_steps
    ) for w in widths]
    ts_consts = [training_steps * gs for gs in gamma_stars if np.isfinite(gs)]
    ts_constant = float(np.mean(ts_consts)) if ts_consts else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(int(widths[0]), int(widths[-1])),
        boundary_curve=boundary_curve,
        timescale_constant=ts_constant,
        metadata={
            "depth": depth,
            "init_scale": init_scale,
            "training_steps": training_steps,
            "n_samples": len(dataset),
        },
    )


def detect_regime(
    weights: List[NDArray],
    dataset: NDArray,
    lr: float,
    training_steps: int = 100,
    seed: int = 42,
) -> str:
    """Detect whether a model will train in the lazy or rich regime.

    Parameters
    ----------
    weights : list of NDArray
        Network weight matrices.
    dataset : NDArray of shape (n_samples, input_dim)
        Input data.
    lr : float
        Learning rate.
    training_steps : int
        Training duration.
    seed : int
        Random seed.

    Returns
    -------
    str
        ``"lazy"``, ``"rich"``, or ``"critical"``.
    """
    input_dim, width, depth, init_scale = _extract_mlp_params(weights)
    mu_max = _compute_mu_max_eff(input_dim, width, depth, len(dataset), seed)
    g_star = _predict_gamma_star(mu_max, training_steps)
    g = _compute_gamma(lr, init_scale, width)

    if g < g_star * 0.8:
        return "lazy"
    elif g > g_star * 1.2:
        return "rich"
    return "critical"


def predict_phase_boundary(
    input_dim: int,
    width: int,
    depth: int,
    init_scale: float = 1.0,
    training_steps: int = 100,
    n_samples: int = 50,
    seed: int = 42,
) -> float:
    """Predict the critical learning rate separating lazy from rich.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    width : int
        Network width.
    depth : int
        Network depth (number of weight matrices).
    init_scale : float
        Initialisation scale σ.
    training_steps : int
        Training duration T.
    n_samples : int
        Number of data samples (affects NTK spectrum).
    seed : int
        Random seed.

    Returns
    -------
    float
        Critical learning rate η*.
    """
    mu_max = _compute_mu_max_eff(input_dim, width, depth, n_samples, seed)
    g_star = _predict_gamma_star(mu_max, training_steps)
    # γ = η·σ²/N  =>  η* = γ*·N/σ²
    critical_lr = g_star * width / (init_scale ** 2)
    return critical_lr


def recommend_training_regime(
    weights: List[NDArray],
    dataset: NDArray,
    training_steps: int = 100,
    prefer_rich: bool = True,
    seed: int = 42,
) -> TrainingRecommendation:
    """Generate an actionable training recommendation.

    Analyzes the network architecture and dataset to recommend a learning
    rate that places training in the desired regime.

    Parameters
    ----------
    weights : list of NDArray
        Network weight matrices.
    dataset : NDArray of shape (n_samples, input_dim)
        Input data.
    training_steps : int
        Planned training duration.
    prefer_rich : bool
        If True, recommend LR above the phase boundary (feature learning).
        If False, recommend LR below (kernel regime).
    seed : int
        Random seed.

    Returns
    -------
    TrainingRecommendation
        Complete recommendation with explanation.
    """
    input_dim, width, depth, init_scale = _extract_mlp_params(weights)
    mu_max = _compute_mu_max_eff(input_dim, width, depth, len(dataset), seed)
    g_star = _predict_gamma_star(mu_max, training_steps)
    critical_lr = g_star * width / (init_scale ** 2)

    if prefer_rich:
        rec_lr = critical_lr * 3.0  # well into rich regime
        regime = Regime.RICH
        explanation = (
            f"LR={rec_lr:.2e} places training in the RICH (feature-learning) "
            f"regime. Critical LR={critical_lr:.2e} at width={width}, "
            f"depth={depth}. The network will learn task-relevant features."
        )
        warmup = max(10, training_steps // 10)
    else:
        rec_lr = critical_lr * 0.3  # well into lazy regime
        regime = Regime.LAZY
        explanation = (
            f"LR={rec_lr:.2e} places training in the LAZY (kernel) "
            f"regime. Critical LR={critical_lr:.2e} at width={width}, "
            f"depth={depth}. Training dynamics follow the NTK prediction."
        )
        warmup = 0

    return TrainingRecommendation(
        recommended_lr=rec_lr,
        regime=regime,
        critical_lr=critical_lr,
        explanation=explanation,
        warmup_steps=warmup,
        init_scale=init_scale,
    )
