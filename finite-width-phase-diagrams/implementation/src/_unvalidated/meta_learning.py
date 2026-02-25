"""Phase diagrams for meta-learning algorithms.

Extends the phase diagram framework to meta-learning, analyzing the
interplay between inner-loop and outer-loop learning rates, task diversity,
few-shot boundaries, and meta-overfitting. MAML and related algorithms
exhibit a double phase structure arising from nested optimization.

Example
-------
>>> from phase_diagrams.meta_learning import maml_phase_diagram
>>> diagram = maml_phase_diagram(model, task_distribution)
>>> print(diagram.metadata["inner_lr"])
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

class MetaRegime(str, Enum):
    """Meta-learning regime classification."""
    FEATURE_REUSE = "feature_reuse"
    RAPID_LEARNING = "rapid_learning"
    MEMORIZATION = "memorization"
    COLLAPSE = "collapse"


@dataclass
class InnerOuterPhase:
    """Phase diagram over inner and outer learning rates.

    Attributes
    ----------
    inner_lrs : NDArray
        Inner-loop learning rates evaluated.
    outer_lrs : NDArray
        Outer-loop learning rates evaluated.
    regime_map : NDArray
        2D array of regime classifications (encoded as ints).
    regime_labels : Dict[int, str]
        Mapping from int codes to regime names.
    optimal_inner_lr : float
        Best inner LR for feature learning.
    optimal_outer_lr : float
        Best outer LR for feature learning.
    phase_boundaries : List[NDArray]
        Boundary curves between regions.
    inner_outer_ratio : float
        Optimal inner/outer LR ratio.
    hessian_condition : float
        Condition number of the meta-gradient Hessian.
    """
    inner_lrs: NDArray = field(default_factory=lambda: np.array([]))
    outer_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regime_map: NDArray = field(default_factory=lambda: np.array([]))
    regime_labels: Dict[int, str] = field(default_factory=dict)
    optimal_inner_lr: float = 0.0
    optimal_outer_lr: float = 0.0
    phase_boundaries: List[NDArray] = field(default_factory=list)
    inner_outer_ratio: float = 1.0
    hessian_condition: float = 1.0


@dataclass
class PhaseBoundary:
    """Phase boundary in few-shot regime.

    Attributes
    ----------
    shots : NDArray
        Number of shots evaluated.
    critical_lrs : NDArray
        Critical LR at each shot count.
    regimes : Dict[int, Regime]
        Regime at each shot count (at default LR).
    min_shots_for_rich : int
        Minimum shots needed for feature learning.
    scaling_exponent : float
        How critical LR scales with shots: eta* ~ K^{alpha}.
    generalization_gap : NDArray
        Train-test gap at each shot count.
    """
    shots: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[int, Regime] = field(default_factory=dict)
    min_shots_for_rich: int = 1
    scaling_exponent: float = 0.0
    generalization_gap: NDArray = field(default_factory=lambda: np.array([]))


@dataclass
class TaskDiversityPhase:
    """How task diversity affects phase behavior.

    Attributes
    ----------
    diversity_levels : NDArray
        Task diversity levels evaluated (0 = identical tasks, 1 = maximally diverse).
    critical_lrs : NDArray
        Critical LR at each diversity level.
    regimes : Dict[float, Regime]
        Regime at each diversity level.
    optimal_diversity : float
        Diversity level that best promotes feature learning.
    diversity_collapse_threshold : float
        Diversity above which meta-learning collapses.
    task_similarity_spectrum : NDArray
        Eigenvalues of the task similarity matrix.
    effective_task_rank : float
        Effective number of distinct tasks.
    """
    diversity_levels: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[float, Regime] = field(default_factory=dict)
    optimal_diversity: float = 0.5
    diversity_collapse_threshold: float = 1.0
    task_similarity_spectrum: NDArray = field(default_factory=lambda: np.array([]))
    effective_task_rank: float = 1.0


@dataclass
class OverfitPrediction:
    """Prediction of meta-overfitting behavior.

    Attributes
    ----------
    n_tasks_evaluated : NDArray
        Number of training tasks evaluated.
    train_meta_loss : NDArray
        Meta-training loss at each task count.
    val_meta_loss : NDArray
        Meta-validation loss at each task count.
    overfit_onset : int
        Number of tasks at which overfitting begins.
    critical_task_count : int
        Minimum tasks to avoid overfitting.
    generalization_bound : float
        PAC-Bayes-style generalization bound.
    regime_at_task_count : Dict[int, Regime]
        Phase regime as function of task count.
    recommendation : str
        Actionable advice on task count.
    """
    n_tasks_evaluated: NDArray = field(default_factory=lambda: np.array([]))
    train_meta_loss: NDArray = field(default_factory=lambda: np.array([]))
    val_meta_loss: NDArray = field(default_factory=lambda: np.array([]))
    overfit_onset: int = 0
    critical_task_count: int = 1
    generalization_bound: float = 0.0
    regime_at_task_count: Dict[int, Regime] = field(default_factory=dict)
    recommendation: str = ""


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_meta_params(model: Any) -> Dict[str, Any]:
    """Extract meta-learning model parameters."""
    if isinstance(model, dict):
        return {
            "input_dim": model.get("input_dim", 64),
            "hidden_dim": model.get("hidden_dim", 64),
            "output_dim": model.get("output_dim", 5),
            "depth": model.get("depth", 3),
            "init_scale": model.get("init_scale", 1.0),
            "inner_lr": model.get("inner_lr", 0.01),
            "inner_steps": model.get("inner_steps", 5),
        }
    if isinstance(model, (list, tuple)):
        input_dim = model[0].shape[1] if model[0].ndim == 2 else model[0].shape[0]
        hidden_dim = model[0].shape[0] if model[0].ndim == 2 else 64
        return {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": model[-1].shape[0] if model[-1].ndim == 2 else 5,
            "depth": len(model),
            "init_scale": float(np.std(model[0]) * math.sqrt(hidden_dim)),
            "inner_lr": 0.01,
            "inner_steps": 5,
        }
    raise TypeError(f"Unsupported model type: {type(model)}")


def _generate_task(
    input_dim: int,
    output_dim: int,
    n_samples: int,
    task_seed: int,
    diversity: float = 0.5,
) -> Tuple[NDArray, NDArray]:
    """Generate a synthetic meta-learning task.

    Returns (X, Y) where X is (n_samples, input_dim) and
    Y is (n_samples, output_dim).
    """
    rng = np.random.RandomState(task_seed)
    X = rng.randn(n_samples, input_dim)

    # Task-specific linear + nonlinear transformation
    W_task = rng.randn(input_dim, output_dim) * diversity
    bias = rng.randn(output_dim) * diversity * 0.5
    Y = np.tanh(X @ W_task + bias)
    return X, Y


def _maml_inner_loop(
    weights: List[NDArray],
    X: NDArray,
    Y: NDArray,
    inner_lr: float,
    inner_steps: int,
) -> Tuple[List[NDArray], float]:
    """Simulate MAML inner loop (gradient descent on task).

    Returns (adapted_weights, final_loss).
    """
    adapted = [w.copy() for w in weights]
    n_samples = X.shape[0]

    for step in range(inner_steps):
        # Forward pass
        h = X
        activations = [h]
        for l, w in enumerate(adapted):
            pre = h @ w
            if l < len(adapted) - 1:
                h = np.maximum(pre, 0)
            else:
                h = pre
            activations.append(h)

        # Loss: MSE
        loss = float(np.mean((h - Y) ** 2))

        # Backward pass (simplified gradient computation)
        grad_output = 2 * (h - Y) / n_samples
        grads = []
        delta = grad_output

        for l in range(len(adapted) - 1, -1, -1):
            grad_w = activations[l].T @ delta
            grads.insert(0, grad_w)
            if l > 0:
                delta = delta @ adapted[l].T
                delta = delta * (activations[l] > 0).astype(float)

        # Update
        for l in range(len(adapted)):
            adapted[l] = adapted[l] - inner_lr * grads[l]

    # Final loss
    h = X
    for l, w in enumerate(adapted):
        pre = h @ w
        h = np.maximum(pre, 0) if l < len(adapted) - 1 else pre
    final_loss = float(np.mean((h - Y) ** 2))

    return adapted, final_loss


def _meta_ntk_eigenspectrum(
    weights: List[NDArray],
    task_data: List[Tuple[NDArray, NDArray]],
    inner_lr: float,
    inner_steps: int,
    seed: int = 42,
) -> NDArray:
    """Approximate meta-NTK eigenspectrum.

    The meta-NTK captures how the meta-gradient (gradient of outer loss
    w.r.t. initialization) depends on the model parameters. It has
    contributions from both the inner-loop Jacobian and the outer-loop
    Jacobian, coupled through the Hessian.
    """
    n_tasks = len(task_data)
    n_params = sum(w.size for w in weights)
    n_eval = min(n_tasks, 30)

    # Meta-gradient features: compute adapted parameters for each task
    meta_features = np.zeros((n_eval, n_params))
    for t in range(n_eval):
        X, Y = task_data[t]
        adapted, _ = _maml_inner_loop(weights, X, Y, inner_lr, inner_steps)
        # Meta-gradient direction: difference between adapted and initial
        param_diff = np.concatenate([
            (adapted[l] - weights[l]).flatten() for l in range(len(weights))
        ])
        meta_features[t] = param_diff

    # Meta-NTK Gram matrix
    K_meta = meta_features @ meta_features.T
    eigvals = np.sort(np.linalg.eigvalsh(K_meta))[::-1]
    return eigvals


def _compute_meta_mu_max(
    weights: List[NDArray],
    task_data: List[Tuple[NDArray, NDArray]],
    inner_lr: float,
    inner_steps: int,
    hidden_dim: int,
    seed: int = 42,
) -> float:
    """Effective mu_max for meta-learning bifurcation analysis."""
    eigvals = _meta_ntk_eigenspectrum(weights, task_data, inner_lr, inner_steps, seed)
    return float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0


def _initialize_weights(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    init_scale: float,
    seed: int = 42,
) -> List[NDArray]:
    """Initialize MLP weights for meta-learning."""
    rng = np.random.RandomState(seed)
    weights = []
    dims = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
    for l in range(depth):
        fan_in = dims[l]
        fan_out = dims[l + 1]
        W = rng.randn(fan_in, fan_out) * init_scale / math.sqrt(fan_in)
        weights.append(W)
    return weights


def _task_similarity_matrix(
    task_data: List[Tuple[NDArray, NDArray]],
    weights: List[NDArray],
    inner_lr: float,
    inner_steps: int,
) -> NDArray:
    """Compute pairwise task similarity from adapted weights."""
    n_tasks = len(task_data)
    adapted_params = []

    for t in range(n_tasks):
        X, Y = task_data[t]
        adapted, _ = _maml_inner_loop(weights, X, Y, inner_lr, inner_steps)
        params = np.concatenate([w.flatten() for w in adapted])
        adapted_params.append(params)

    adapted_params = np.array(adapted_params)
    # Cosine similarity
    norms = np.linalg.norm(adapted_params, axis=1, keepdims=True) + 1e-12
    normalized = adapted_params / norms
    similarity = normalized @ normalized.T
    return similarity


# ======================================================================
# Public API
# ======================================================================

def maml_phase_diagram(
    model: Any,
    task_distribution: Any,
    lr_range: Tuple[float, float] = (1e-4, 0.1),
    n_lr_steps: int = 25,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for MAML over learning rate and width.

    MAML has a double phase structure: the inner loop determines
    task-specific adaptation, while the outer loop determines
    meta-learning. The effective coupling depends on both the inner
    and outer learning rates, mediated by the meta-Hessian.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification with keys ``{'input_dim', 'hidden_dim',
        'output_dim', 'depth', 'init_scale', 'inner_lr', 'inner_steps'}``.
    task_distribution : list of (NDArray, NDArray) or dict
        Task distribution. List of (X, Y) pairs, or dict with
        ``{'n_tasks', 'n_samples', 'diversity'}``.
    lr_range : (float, float)
        Outer learning rate scan range.
    n_lr_steps : int
        Number of outer LR grid points.
    training_steps : int
        Number of outer-loop steps.
    seed : int
        Random seed.

    Returns
    -------
    PhaseDiagram
        Phase diagram with meta-learning-aware boundary.
    """
    params = _extract_meta_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    inner_lr = params["inner_lr"]
    inner_steps = params["inner_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    # Generate or use task data
    if isinstance(task_distribution, dict):
        n_tasks = task_distribution.get("n_tasks", 100)
        n_samples = task_distribution.get("n_samples", 10)
        diversity = task_distribution.get("diversity", 0.5)
        task_data = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_samples, seed + t, diversity)
            for t in range(n_tasks)
        ]
    else:
        task_data = list(task_distribution)

    mu_max = _compute_meta_mu_max(
        weights, task_data, inner_lr, inner_steps, hidden_dim, seed
    )

    # Inner-loop correction: inner LR amplifies the effective coupling
    inner_amplification = 1.0 + inner_lr * inner_steps * mu_max
    g_star = _predict_gamma_star(mu_max * inner_amplification, training_steps)

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

        ntk_drift = gamma * mu_max * inner_amplification * training_steps
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
            "architecture": "MAML",
            "hidden_dim": hidden_dim,
            "depth": params["depth"],
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "n_tasks": len(task_data),
            "inner_amplification": inner_amplification,
        },
    )


def inner_outer_lr_phase(
    model: Any,
    tasks: Any,
    inner_lr_range: Tuple[float, float] = (1e-4, 1.0),
    outer_lr_range: Tuple[float, float] = (1e-4, 0.1),
    n_inner_steps: int = 15,
    n_outer_steps: int = 15,
    training_steps: int = 100,
    seed: int = 42,
) -> InnerOuterPhase:
    """Map the phase diagram over inner and outer learning rates.

    The inner-outer LR space has four distinct regions:
    1. Feature reuse: low inner LR, moderate outer LR (rich meta-learning)
    2. Rapid learning: high inner LR, low outer LR (fast adaptation)
    3. Memorization: high both (overfits to training tasks)
    4. Collapse: extreme values (training diverges)

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    tasks : list or dict
        Task distribution.
    inner_lr_range : (float, float)
        Inner LR scan range.
    outer_lr_range : (float, float)
        Outer LR scan range.
    n_inner_steps : int
        Grid resolution for inner LR.
    n_outer_steps : int
        Grid resolution for outer LR.
    training_steps : int
        Number of outer-loop steps.
    seed : int
        Random seed.

    Returns
    -------
    InnerOuterPhase
        2D phase map over inner/outer LR space.
    """
    params = _extract_meta_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    inner_steps_cfg = params["inner_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    if isinstance(tasks, dict):
        n_tasks = tasks.get("n_tasks", 50)
        n_samples = tasks.get("n_samples", 10)
        diversity = tasks.get("diversity", 0.5)
        task_data = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_samples, seed + t, diversity)
            for t in range(n_tasks)
        ]
    else:
        task_data = list(tasks)

    inner_lrs = np.logspace(
        math.log10(inner_lr_range[0]), math.log10(inner_lr_range[1]), n_inner_steps
    )
    outer_lrs = np.logspace(
        math.log10(outer_lr_range[0]), math.log10(outer_lr_range[1]), n_outer_steps
    )

    # Regime encoding: 0=feature_reuse, 1=rapid_learning, 2=memorization, 3=collapse
    regime_labels = {
        0: "feature_reuse",
        1: "rapid_learning",
        2: "memorization",
        3: "collapse",
    }
    regime_map = np.zeros((n_inner_steps, n_outer_steps), dtype=int)
    boundary_points_list: List[NDArray] = []

    best_score = -float("inf")
    best_inner = inner_lrs[0]
    best_outer = outer_lrs[0]

    for i, inner_lr in enumerate(inner_lrs):
        mu_max = _compute_meta_mu_max(
            weights, task_data[:min(20, len(task_data))],
            inner_lr, inner_steps_cfg, hidden_dim, seed,
        )
        inner_amp = 1.0 + inner_lr * inner_steps_cfg * mu_max

        for j, outer_lr in enumerate(outer_lrs):
            gamma_outer = _compute_gamma(outer_lr, init_scale, hidden_dim)
            gamma_inner = inner_lr * init_scale ** 2 / hidden_dim
            effective_gamma = gamma_outer * inner_amp

            g_star = _predict_gamma_star(mu_max * inner_amp, training_steps)

            # Classify region based on inner/outer balance
            if gamma_inner > 10 and gamma_outer > 10:
                # Both very high → collapse
                regime_map[i, j] = 3
            elif gamma_inner > 5 * gamma_outer:
                # Inner dominates → rapid learning
                regime_map[i, j] = 1
            elif effective_gamma > g_star * 1.5:
                if gamma_inner > gamma_outer:
                    regime_map[i, j] = 2  # memorization
                else:
                    regime_map[i, j] = 0  # feature_reuse (rich)
            else:
                regime_map[i, j] = 0  # feature_reuse (lazy/critical)

            # Track best for feature learning
            if regime_map[i, j] == 0:
                score = -abs(math.log(effective_gamma + 1e-12) - math.log(g_star + 1e-12))
                if score > best_score:
                    best_score = score
                    best_inner = float(inner_lr)
                    best_outer = float(outer_lr)

    # Extract boundaries
    for i in range(n_inner_steps - 1):
        for j in range(n_outer_steps - 1):
            if regime_map[i, j] != regime_map[i + 1, j] or regime_map[i, j] != regime_map[i, j + 1]:
                boundary_points_list.append(np.array([inner_lrs[i], outer_lrs[j]]))

    # Hessian condition number approximation
    # Meta-Hessian condition ~ inner_amplification * cond(NTK)
    meta_eigvals = _meta_ntk_eigenspectrum(
        weights, task_data[:min(20, len(task_data))],
        best_inner, inner_steps_cfg, seed,
    )
    hessian_cond = float(meta_eigvals[0] / (meta_eigvals[-1] + 1e-12)) if len(meta_eigvals) > 1 else 1.0

    return InnerOuterPhase(
        inner_lrs=inner_lrs,
        outer_lrs=outer_lrs,
        regime_map=regime_map,
        regime_labels=regime_labels,
        optimal_inner_lr=best_inner,
        optimal_outer_lr=best_outer,
        phase_boundaries=boundary_points_list,
        inner_outer_ratio=best_inner / (best_outer + 1e-12),
        hessian_condition=hessian_cond,
    )


def few_shot_phase_boundary(
    model: Any,
    shots: Sequence[int],
    n_ways: int = 5,
    n_query: int = 15,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseBoundary:
    """Find the phase boundary as a function of shots per class.

    More shots provide more gradient signal per task, shifting the
    effective coupling. Below a critical number of shots, the model
    cannot enter the feature-learning regime regardless of LR.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    shots : sequence of int
        Shot counts to evaluate (e.g., [1, 2, 5, 10, 20]).
    n_ways : int
        Number of classes per task.
    n_query : int
        Number of query samples per class.
    training_steps : int
        Number of outer-loop steps.
    seed : int
        Random seed.

    Returns
    -------
    PhaseBoundary
        Phase boundary as function of shot count.
    """
    params = _extract_meta_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    inner_lr = params["inner_lr"]
    inner_steps = params["inner_steps"]

    shots_arr = np.array(sorted(shots))
    critical_lrs = np.zeros(len(shots_arr))
    regimes: Dict[int, Regime] = {}
    gen_gaps = np.zeros(len(shots_arr))

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    min_shots_rich = int(shots_arr[-1])
    default_lr = 0.001

    for idx, k_shot in enumerate(shots_arr):
        k = int(k_shot)
        n_support = k * n_ways
        n_total = n_support + n_query * n_ways

        # Generate tasks with this shot count
        task_data = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_total, seed + t, 0.5)
            for t in range(50)
        ]

        # The effective coupling scales with sqrt(n_support) due to
        # gradient averaging in the inner loop
        support_factor = math.sqrt(n_support) / math.sqrt(n_ways)

        mu_max = _compute_meta_mu_max(
            weights, task_data[:20], inner_lr, inner_steps, hidden_dim, seed,
        )
        inner_amp = 1.0 + inner_lr * inner_steps * mu_max * support_factor
        g_star = _predict_gamma_star(mu_max * inner_amp, training_steps)
        critical_lr = g_star * hidden_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(default_lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            regimes[k] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[k] = Regime.RICH
            if k < min_shots_rich:
                min_shots_rich = k
        else:
            regimes[k] = Regime.CRITICAL

        # Generalization gap approximation: scales as 1/sqrt(k)
        gen_gaps[idx] = 1.0 / math.sqrt(k + 1)

    # Fit scaling: eta* ~ K^{alpha}
    valid = critical_lrs > 1e-12
    if np.sum(valid) > 2:
        log_k = np.log(shots_arr[valid].astype(float))
        log_lr = np.log(critical_lrs[valid])
        coeffs = np.polyfit(log_k, log_lr, 1)
        scaling_exp = float(coeffs[0])
    else:
        scaling_exp = 0.0

    return PhaseBoundary(
        shots=shots_arr.astype(float),
        critical_lrs=critical_lrs,
        regimes=regimes,
        min_shots_for_rich=min_shots_rich,
        scaling_exponent=scaling_exp,
        generalization_gap=gen_gaps,
    )


def task_diversity_phase(
    model: Any,
    task_distributions: Optional[Sequence[float]] = None,
    n_tasks: int = 100,
    n_samples_per_task: int = 20,
    training_steps: int = 100,
    seed: int = 42,
) -> TaskDiversityPhase:
    """Analyze how task diversity affects the phase diagram.

    Task diversity controls the effective dimensionality of the
    meta-learning problem. Low diversity (similar tasks) leads to
    feature reuse in a low-dimensional subspace, while high diversity
    requires learning more general representations.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    task_distributions : sequence of float or None
        Diversity levels to evaluate (0 to 1). If None, uses linspace.
    n_tasks : int
        Number of tasks per diversity level.
    n_samples_per_task : int
        Samples per task.
    training_steps : int
        Number of outer-loop steps.
    seed : int
        Random seed.

    Returns
    -------
    TaskDiversityPhase
        Analysis of diversity's effect on phase behavior.
    """
    params = _extract_meta_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    inner_lr = params["inner_lr"]
    inner_steps = params["inner_steps"]

    if task_distributions is None:
        task_distributions = np.linspace(0.05, 1.0, 10)
    else:
        task_distributions = np.array(sorted(task_distributions))

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    critical_lrs = np.zeros(len(task_distributions))
    regimes: Dict[float, Regime] = {}
    default_lr = 0.001

    collapse_threshold = 1.0
    optimal_diversity = 0.5
    best_gap = float("inf")

    all_similarities = []

    for idx, div in enumerate(task_distributions):
        div_float = float(div)
        task_data = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_samples_per_task, seed + t, div_float)
            for t in range(min(n_tasks, 50))
        ]

        # Task similarity matrix
        sim = _task_similarity_matrix(task_data[:20], weights, inner_lr, inner_steps)
        sim_eigvals = np.sort(np.linalg.eigvalsh(sim))[::-1]
        all_similarities.append(sim_eigvals)

        mu_max = _compute_meta_mu_max(
            weights, task_data[:20], inner_lr, inner_steps, hidden_dim, seed,
        )

        # Diversity modulates the effective coupling:
        # High diversity → more diverse gradients → lower effective mu_max
        diversity_correction = 1.0 / (1.0 + div_float * 2.0)
        g_star = _predict_gamma_star(mu_max * diversity_correction, training_steps)
        critical_lr = g_star * hidden_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(default_lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            regimes[div_float] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[div_float] = Regime.RICH
        else:
            regimes[div_float] = Regime.CRITICAL

        # Check for collapse: if task similarity becomes too low
        mean_sim = float(np.mean(sim[np.triu_indices_from(sim, k=1)]))
        if mean_sim < 0.1 and collapse_threshold == 1.0:
            collapse_threshold = div_float

        # Track best diversity for feature learning
        gap = abs(math.log(gamma + 1e-12) - math.log(g_star + 1e-12))
        if gap < best_gap:
            best_gap = gap
            optimal_diversity = div_float

    # Combine similarity spectra
    if all_similarities:
        max_len = max(len(s) for s in all_similarities)
        # Use the spectrum at optimal diversity
        opt_idx = int(np.argmin(np.abs(task_distributions - optimal_diversity)))
        task_spectrum = all_similarities[opt_idx]
        eff_rank = float(np.sum(task_spectrum) / (task_spectrum[0] + 1e-12))
    else:
        task_spectrum = np.array([1.0])
        eff_rank = 1.0

    return TaskDiversityPhase(
        diversity_levels=task_distributions,
        critical_lrs=critical_lrs,
        regimes=regimes,
        optimal_diversity=optimal_diversity,
        diversity_collapse_threshold=collapse_threshold,
        task_similarity_spectrum=task_spectrum,
        effective_task_rank=eff_rank,
    )


def meta_overfitting_prediction(
    model: Any,
    n_tasks: Sequence[int],
    n_samples_per_task: int = 20,
    task_diversity: float = 0.5,
    training_steps: int = 100,
    seed: int = 42,
) -> OverfitPrediction:
    """Predict meta-overfitting as a function of training task count.

    Meta-overfitting occurs when the model memorizes the training task
    distribution rather than learning generalizable features. The onset
    depends on the ratio of model capacity to task diversity.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    n_tasks : sequence of int
        Number of training tasks to evaluate.
    n_samples_per_task : int
        Samples per task.
    task_diversity : float
        Diversity level for task generation.
    training_steps : int
        Number of outer-loop steps.
    seed : int
        Random seed.

    Returns
    -------
    OverfitPrediction
        Predicted meta-overfitting behavior.
    """
    params = _extract_meta_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    inner_lr = params["inner_lr"]
    inner_steps = params["inner_steps"]

    n_tasks_arr = np.array(sorted(n_tasks))
    train_losses = np.zeros(len(n_tasks_arr))
    val_losses = np.zeros(len(n_tasks_arr))
    regime_at_count: Dict[int, Regime] = {}

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    # Model capacity (number of parameters)
    n_params = sum(w.size for w in weights)
    overfit_onset = int(n_tasks_arr[-1])
    found_onset = False

    for idx, nt in enumerate(n_tasks_arr):
        nt_int = int(nt)

        # Generate training tasks
        train_tasks = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_samples_per_task, seed + t, task_diversity)
            for t in range(nt_int)
        ]

        # Generate validation tasks (different seeds)
        val_tasks = [
            _generate_task(params["input_dim"], params["output_dim"],
                          n_samples_per_task, seed + 10000 + t, task_diversity)
            for t in range(min(20, nt_int))
        ]

        # Simulate meta-training loss
        train_task_losses = []
        for X, Y in train_tasks[:20]:
            _, loss = _maml_inner_loop(weights, X, Y, inner_lr, inner_steps)
            train_task_losses.append(loss)
        train_losses[idx] = float(np.mean(train_task_losses))

        # Simulate meta-validation loss
        val_task_losses = []
        for X, Y in val_tasks:
            _, loss = _maml_inner_loop(weights, X, Y, inner_lr, inner_steps)
            val_task_losses.append(loss)
        val_losses[idx] = float(np.mean(val_task_losses))

        # Overfitting detection: val loss increases while train decreases
        # Approximate: overfit when effective capacity > task diversity
        effective_capacity = n_params / (nt_int * n_samples_per_task)
        if effective_capacity > 1.0 and not found_onset:
            # More parameters than data → overfitting likely
            regime_at_count[nt_int] = Regime.RICH  # overfitting in rich regime
        elif effective_capacity > 0.1:
            regime_at_count[nt_int] = Regime.CRITICAL
        else:
            regime_at_count[nt_int] = Regime.LAZY

        # Detect overfit onset: where val loss starts being > train loss significantly
        if idx > 0 and val_losses[idx] > train_losses[idx] * 1.5 and not found_onset:
            overfit_onset = nt_int
            found_onset = True

    # Critical task count: minimum tasks to avoid overfitting
    # Heuristic: need at least n_params / n_samples_per_task tasks
    critical_count = max(1, int(math.ceil(n_params / (n_samples_per_task * 10))))
    critical_count = min(critical_count, int(n_tasks_arr[-1]))

    # PAC-Bayes generalization bound approximation
    # Bound ~ sqrt(KL(posterior || prior) / n_tasks) + sqrt(log(1/delta) / n_tasks)
    kl_div = n_params * math.log(2)  # rough estimate
    n_total = int(n_tasks_arr[-1]) * n_samples_per_task
    gen_bound = math.sqrt(kl_div / (n_total + 1)) + math.sqrt(math.log(100) / (n_total + 1))

    # Recommendation
    if overfit_onset < int(n_tasks_arr[-1]):
        recommendation = (
            f"Meta-overfitting detected at {overfit_onset} tasks. "
            f"Use at least {critical_count} tasks to avoid overfitting. "
            f"Consider reducing model capacity (hidden_dim={hidden_dim}) "
            f"or increasing task diversity."
        )
    else:
        recommendation = (
            f"No meta-overfitting detected up to {int(n_tasks_arr[-1])} tasks. "
            f"Model capacity ({n_params} params) is well-matched to the "
            f"task distribution at diversity={task_diversity}."
        )

    return OverfitPrediction(
        n_tasks_evaluated=n_tasks_arr.astype(float),
        train_meta_loss=train_losses,
        val_meta_loss=val_losses,
        overfit_onset=overfit_onset,
        critical_task_count=critical_count,
        generalization_bound=gen_bound,
        regime_at_task_count=regime_at_count,
        recommendation=recommendation,
    )
