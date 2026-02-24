"""Phase-aware neural architecture search.

Scores architectures by their phase-regime characteristics: whether they
will feature-learn at practical learning rates, what the minimal width is
for rich training, and the estimated training cost in each regime.

Example
-------
>>> from phase_diagrams.architecture_search import PhaseAwareNAS
>>> nas = PhaseAwareNAS(input_dim=784, output_dim=10, n_samples=100)
>>> result = nas.search(width_range=(64, 2048), depth_range=(2, 6))
>>> print(result.best.width, result.best.depth, result.best.regime)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ArchScore:
    """Phase-regime score for a single architecture.

    Attributes
    ----------
    width : int
        Network width.
    depth : int
        Network depth.
    regime_at_default_lr : str
        Regime at LR=0.01 (or specified default).
    critical_lr : float
        Phase boundary LR.
    gamma_star : float
        Critical coupling.
    feature_learning_score : float
        0–1 score: how easily the arch enters the rich regime.
    cost_estimate : float
        Relative training cost (FLOPs proxy).
    params : int
        Total parameter count.
    richness_margin : float
        How far into rich regime at the evaluated LR (>0 = rich).
    """
    width: int = 256
    depth: int = 2
    regime_at_default_lr: str = "unknown"
    critical_lr: float = 0.0
    gamma_star: float = 0.0
    feature_learning_score: float = 0.0
    cost_estimate: float = 0.0
    params: int = 0
    richness_margin: float = 0.0


@dataclass
class CostEstimate:
    """Training cost estimate for an architecture.

    Attributes
    ----------
    flops_per_step : float
        Approximate FLOPs per training step.
    total_flops : float
        Total FLOPs for the planned training.
    memory_bytes : float
        Estimated peak memory (bytes).
    wall_time_estimate : str
        Human-readable wall time estimate.
    regime : str
        Predicted regime (affects convergence speed).
    convergence_steps : int
        Estimated steps to convergence in the predicted regime.
    """
    flops_per_step: float = 0.0
    total_flops: float = 0.0
    memory_bytes: float = 0.0
    wall_time_estimate: str = ""
    regime: str = "unknown"
    convergence_steps: int = 0


@dataclass
class ComparisonResult:
    """Comparison of architecture families.

    Attributes
    ----------
    families : dict
        {family_name: list[ArchScore]} for each family evaluated.
    best_per_family : dict
        {family_name: ArchScore} — best architecture in each family.
    overall_best : ArchScore
        Best architecture across all families.
    summary : str
        Human-readable comparison summary.
    """
    families: Dict[str, List[ArchScore]] = field(default_factory=dict)
    best_per_family: Dict[str, ArchScore] = field(default_factory=dict)
    overall_best: Optional[ArchScore] = None
    summary: str = ""


@dataclass
class NASResult:
    """Result of a phase-aware architecture search.

    Attributes
    ----------
    candidates : list of ArchScore
        All evaluated architectures, sorted by desirability.
    best : ArchScore
        Best architecture found.
    pareto_front : list of ArchScore
        Pareto-optimal architectures (feature learning vs. cost).
    search_config : dict
        Configuration used for the search.
    summary : str
        Human-readable summary.
    """
    candidates: List[ArchScore] = field(default_factory=list)
    best: Optional[ArchScore] = None
    pareto_front: List[ArchScore] = field(default_factory=list)
    search_config: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


# ======================================================================
# Internal helpers
# ======================================================================

def _compute_mu_max_eff(
    input_dim: int, width: int, depth: int, n_samples: int = 50, seed: int = 42
) -> float:
    """Approximate effective perturbation eigenvalue."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    h = X.copy()
    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = width if l < depth - 1 else 1
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        pre = h @ W
        h = np.maximum(pre, 0) if l < depth - 1 else pre

    K = h @ h.T
    eigvals = np.linalg.eigvalsh(K)
    mu_max = float(eigvals[-1]) if len(eigvals) > 0 else 1.0
    return mu_max / width


def _predict_gamma_star(
    mu_max_eff: float, training_steps: int,
    drift_threshold: float = 0.1, drift_floor: float = 1e-3,
) -> float:
    if mu_max_eff <= 0:
        return float("inf")
    c = math.log(drift_threshold / drift_floor)
    return c / (training_steps * mu_max_eff)


def _compute_gamma(lr: float, init_scale: float, width: int) -> float:
    return lr * init_scale ** 2 / width


def _count_params(input_dim: int, width: int, depth: int, output_dim: int = 1) -> int:
    """Count parameters for an MLP: input→width→...→width→output."""
    total = input_dim * width + width  # first layer + bias
    total += (depth - 2) * (width * width + width)  # hidden layers
    total += width * output_dim + output_dim  # output layer
    return max(total, 0)


def _estimate_flops(params: int, n_samples: int) -> float:
    """Approximate FLOPs per step: ~6 × params × batch_size (fwd+bwd)."""
    return 6.0 * params * n_samples


# ======================================================================
# PhaseAwareNAS
# ======================================================================

class PhaseAwareNAS:
    """Phase-aware neural architecture search.

    Evaluates architectures based on their phase-regime characteristics,
    finding the best architecture for the desired training regime while
    considering computational cost.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    output_dim : int
        Output dimensionality.
    n_samples : int
        Dataset size (affects NTK spectrum and cost).
    training_steps : int
        Planned training duration.
    default_lr : float
        Default learning rate for regime evaluation.
    init_scale : float
        Initialisation scale σ.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_samples: int = 100,
        training_steps: int = 100,
        default_lr: float = 0.01,
        init_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        self.training_steps = training_steps
        self.default_lr = default_lr
        self.init_scale = init_scale
        self.seed = seed

    # ------------------------------------------------------------------
    # Score a single architecture
    # ------------------------------------------------------------------

    def score_architecture(
        self, width: int, depth: int, lr: Optional[float] = None
    ) -> ArchScore:
        """Score a single architecture for phase-regime desirability.

        Parameters
        ----------
        width : int
            Network width.
        depth : int
            Network depth.
        lr : float or None
            Learning rate (default: self.default_lr).

        Returns
        -------
        ArchScore
        """
        if lr is None:
            lr = self.default_lr

        mu_max = _compute_mu_max_eff(
            self.input_dim, width, depth, self.n_samples, self.seed
        )
        gamma_star = _predict_gamma_star(mu_max, self.training_steps)
        gamma = _compute_gamma(lr, self.init_scale, width)
        critical_lr = gamma_star * width / (self.init_scale ** 2) if self.init_scale > 0 else float("inf")

        # Regime classification
        if gamma < gamma_star * 0.8:
            regime = "lazy"
        elif gamma > gamma_star * 1.2:
            regime = "rich"
        else:
            regime = "critical"

        # Feature learning score: how easily the arch enters rich
        # Lower critical_lr = easier to enter rich = higher score
        # Normalise to [0, 1] using a sigmoid-like function
        if np.isfinite(critical_lr) and critical_lr > 0:
            fl_score = 1.0 / (1.0 + critical_lr / self.default_lr)
        else:
            fl_score = 0.0

        # Richness margin: γ/γ* - 1 (positive = rich)
        richness_margin = (gamma / gamma_star - 1.0) if gamma_star > 0 else 0.0

        params = _count_params(self.input_dim, width, depth, self.output_dim)
        cost = _estimate_flops(params, self.n_samples) * self.training_steps

        return ArchScore(
            width=width,
            depth=depth,
            regime_at_default_lr=regime,
            critical_lr=critical_lr,
            gamma_star=gamma_star,
            feature_learning_score=fl_score,
            cost_estimate=cost,
            params=params,
            richness_margin=richness_margin,
        )

    # ------------------------------------------------------------------
    # Find minimal width for rich training
    # ------------------------------------------------------------------

    def minimal_width_for_rich(
        self, depth: int, lr: Optional[float] = None
    ) -> int:
        """Find the smallest width at which rich training occurs.

        Since γ = η·σ²/N and γ* depends on N, we scan widths
        to find the boundary.

        Parameters
        ----------
        depth : int
            Network depth.
        lr : float or None
            Learning rate.

        Returns
        -------
        int
            Minimal width for rich regime (or -1 if not found).
        """
        if lr is None:
            lr = self.default_lr

        # γ = η·σ²/N decreases with width, so smaller widths are richer
        # but γ* also depends on width
        for w in range(8, 8192, 8):
            score = self.score_architecture(w, depth, lr)
            if score.regime_at_default_lr == "rich":
                return w
        return -1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        width_range: Tuple[int, int] = (32, 2048),
        depth_range: Tuple[int, int] = (2, 6),
        prefer_rich: bool = True,
        cost_weight: float = 0.3,
        n_width_samples: int = 12,
        n_depth_samples: int = 5,
    ) -> NASResult:
        """Run a phase-aware architecture search.

        Evaluates a grid of (width, depth) combinations and ranks them
        by a combined score of regime desirability and training cost.

        Parameters
        ----------
        width_range : (int, int)
            Min and max width.
        depth_range : (int, int)
            Min and max depth.
        prefer_rich : bool
            If True, prefer architectures that feature-learn.
        cost_weight : float
            Weight of cost penalty relative to feature-learning score (0–1).
        n_width_samples : int
            Number of widths to evaluate.
        n_depth_samples : int
            Number of depths to evaluate.

        Returns
        -------
        NASResult
        """
        widths = np.unique(
            np.logspace(
                math.log10(width_range[0]),
                math.log10(width_range[1]),
                n_width_samples,
            ).astype(int)
        )
        depths = np.unique(
            np.linspace(depth_range[0], depth_range[1], n_depth_samples).astype(int)
        )

        candidates: List[ArchScore] = []
        for w in widths:
            for d in depths:
                score = self.score_architecture(int(w), int(d))
                candidates.append(score)

        # Normalise costs for scoring
        costs = np.array([c.cost_estimate for c in candidates])
        max_cost = costs.max() if costs.max() > 0 else 1.0
        norm_costs = costs / max_cost

        # Combined score
        def combined_score(arch: ArchScore, norm_cost: float) -> float:
            if prefer_rich:
                regime_score = arch.feature_learning_score
            else:
                regime_score = 1.0 - arch.feature_learning_score
            return regime_score * (1 - cost_weight) - norm_cost * cost_weight

        scored = sorted(
            zip(candidates, norm_costs),
            key=lambda x: combined_score(x[0], x[1]),
            reverse=True,
        )
        candidates_sorted = [s[0] for s in scored]

        # Pareto front: feature_learning_score vs cost
        pareto = []
        min_cost_seen = float("inf")
        for arch in sorted(candidates, key=lambda a: -a.feature_learning_score):
            if arch.cost_estimate < min_cost_seen:
                pareto.append(arch)
                min_cost_seen = arch.cost_estimate

        best = candidates_sorted[0] if candidates_sorted else None

        summary = f"Searched {len(candidates)} architectures. "
        if best:
            summary += (
                f"Best: width={best.width}, depth={best.depth}, "
                f"regime={best.regime_at_default_lr}, "
                f"FL score={best.feature_learning_score:.3f}, "
                f"params={best.params:,}."
            )

        return NASResult(
            candidates=candidates_sorted,
            best=best,
            pareto_front=pareto,
            search_config={
                "width_range": width_range,
                "depth_range": depth_range,
                "prefer_rich": prefer_rich,
                "cost_weight": cost_weight,
                "default_lr": self.default_lr,
            },
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Compare architecture families
    # ------------------------------------------------------------------

    def compare_families(
        self,
        families: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> ComparisonResult:
        """Compare architecture families by phase-regime characteristics.

        Parameters
        ----------
        families : dict or None
            ``{family_name: [(width, depth), ...]}``
            If None, uses default MLP families (narrow-deep vs wide-shallow).

        Returns
        -------
        ComparisonResult
        """
        if families is None:
            families = {
                "wide-shallow": [(w, 2) for w in [128, 256, 512, 1024, 2048]],
                "narrow-deep": [(64, d) for d in [2, 3, 4, 5, 6, 8]],
                "balanced": [(w, max(2, int(math.log2(w)))) for w in [64, 128, 256, 512]],
            }

        family_scores: Dict[str, List[ArchScore]] = {}
        best_per: Dict[str, ArchScore] = {}
        overall_best: Optional[ArchScore] = None

        for name, configs in families.items():
            scores = [self.score_architecture(w, d) for w, d in configs]
            family_scores[name] = scores
            best = max(scores, key=lambda s: s.feature_learning_score)
            best_per[name] = best
            if overall_best is None or best.feature_learning_score > overall_best.feature_learning_score:
                overall_best = best

        # Summary
        lines = ["Architecture family comparison:"]
        for name in sorted(family_scores.keys()):
            b = best_per[name]
            avg_fl = np.mean([s.feature_learning_score for s in family_scores[name]])
            lines.append(
                f"  {name}: best=(w={b.width},d={b.depth}), "
                f"avg FL score={avg_fl:.3f}, "
                f"regime={b.regime_at_default_lr}"
            )
        if overall_best:
            lines.append(
                f"Overall best: w={overall_best.width}, d={overall_best.depth} "
                f"(FL={overall_best.feature_learning_score:.3f})"
            )
        summary = "\n".join(lines)

        return ComparisonResult(
            families=family_scores,
            best_per_family=best_per,
            overall_best=overall_best,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        width: int,
        depth: int,
        dataset_size: Optional[int] = None,
        gpu_tflops: float = 10.0,
    ) -> CostEstimate:
        """Estimate training cost for an architecture.

        Parameters
        ----------
        width : int
            Network width.
        depth : int
            Network depth.
        dataset_size : int or None
            Dataset size (default: self.n_samples).
        gpu_tflops : float
            GPU throughput in TFLOPS (for wall-time estimate).

        Returns
        -------
        CostEstimate
        """
        if dataset_size is None:
            dataset_size = self.n_samples

        params = _count_params(self.input_dim, width, depth, self.output_dim)
        flops_per_step = _estimate_flops(params, dataset_size)
        total_flops = flops_per_step * self.training_steps

        # Memory: params (4 bytes) + gradients (4 bytes) + optimizer state (8 bytes)
        memory = params * 16

        # Wall time estimate
        seconds = total_flops / (gpu_tflops * 1e12)
        if seconds < 60:
            wall = f"{seconds:.1f}s"
        elif seconds < 3600:
            wall = f"{seconds / 60:.1f}min"
        else:
            wall = f"{seconds / 3600:.1f}h"

        # Regime affects convergence speed
        score = self.score_architecture(width, depth)
        regime = score.regime_at_default_lr

        # Rich regime typically needs more steps but learns better features
        if regime == "rich":
            convergence_steps = int(self.training_steps * 1.5)
        elif regime == "lazy":
            convergence_steps = self.training_steps
        else:
            convergence_steps = int(self.training_steps * 1.2)

        return CostEstimate(
            flops_per_step=flops_per_step,
            total_flops=total_flops,
            memory_bytes=float(memory),
            wall_time_estimate=wall,
            regime=regime,
            convergence_steps=convergence_steps,
        )
