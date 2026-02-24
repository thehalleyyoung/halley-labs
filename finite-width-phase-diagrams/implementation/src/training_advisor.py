"""Training advisor: actionable recommendations from phase analysis.

Takes the theoretical phase-diagram machinery and produces concrete,
human-readable advice:

- Optimal learning rate for desired regime (lazy or rich)
- Learning rate warmup schedule recommendation
- Width recommendation for a target regime
- Initialisation scale tuning

Example
-------
>>> from phase_diagrams.training_advisor import TrainingAdvisor
>>> advisor = TrainingAdvisor.from_arch(input_dim=784, width=512, depth=3)
>>> rec = advisor.full_recommendation(prefer_rich=True)
>>> print(rec.summary)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class LRRecommendation:
    """Learning rate recommendation.

    Attributes
    ----------
    lr : float
        Recommended learning rate.
    regime : str
        Expected regime at this LR.
    critical_lr : float
        Phase boundary learning rate.
    margin : float
        Distance from boundary (in units of γ/γ*).
    explanation : str
        Human-readable rationale.
    """
    lr: float = 0.0
    regime: str = "rich"
    critical_lr: float = 0.0
    margin: float = 0.0
    explanation: str = ""


@dataclass
class WarmupSchedule:
    """Learning rate warmup schedule.

    Attributes
    ----------
    warmup_steps : int
        Number of warmup steps.
    warmup_type : str
        ``"linear"``, ``"cosine"``, or ``"none"``.
    initial_lr : float
        Starting LR (during warmup).
    target_lr : float
        LR after warmup completes.
    schedule : list of float
        LR at each step (for the warmup phase only).
    explanation : str
        Why this schedule was chosen.
    """
    warmup_steps: int = 0
    warmup_type: str = "linear"
    initial_lr: float = 0.0
    target_lr: float = 0.0
    schedule: List[float] = field(default_factory=list)
    explanation: str = ""


@dataclass
class WidthRecommendation:
    """Width recommendation for a target regime.

    Attributes
    ----------
    width : int
        Recommended width.
    target_regime : str
        Desired regime.
    min_width_for_rich : int
        Smallest width at which rich training is possible (at given LR).
    max_width_for_lazy : int
        Largest width at which lazy training persists.
    explanation : str
        Rationale.
    """
    width: int = 256
    target_regime: str = "rich"
    min_width_for_rich: int = 0
    max_width_for_lazy: int = 0
    explanation: str = ""


@dataclass
class InitScaleRecommendation:
    """Initialisation scale recommendation.

    Attributes
    ----------
    init_scale : float
        Recommended σ.
    current_scale : float
        Current estimated σ.
    target_regime : str
        Desired regime.
    explanation : str
    """
    init_scale: float = 1.0
    current_scale: float = 1.0
    target_regime: str = "rich"
    explanation: str = ""


@dataclass
class FullRecommendation:
    """Complete training recommendation.

    Combines LR, warmup, width, and init-scale advice.

    Attributes
    ----------
    lr : LRRecommendation
    warmup : WarmupSchedule
    width : WidthRecommendation
    init_scale : InitScaleRecommendation
    summary : str
        One-paragraph summary of all recommendations.
    config : dict
        Machine-readable config dict for direct use.
    """
    lr: Optional[LRRecommendation] = None
    warmup: Optional[WarmupSchedule] = None
    width: Optional[WidthRecommendation] = None
    init_scale: Optional[InitScaleRecommendation] = None
    summary: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Internal helpers
# ======================================================================

def _compute_gamma(lr: float, init_scale: float, width: int) -> float:
    """Effective coupling γ = η·σ²/N."""
    return lr * init_scale ** 2 / width


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
    mu_max_eff: float,
    training_steps: int,
    drift_threshold: float = 0.1,
    drift_floor: float = 1e-3,
) -> float:
    """Critical coupling from bifurcation analysis."""
    if mu_max_eff <= 0:
        return float("inf")
    c = math.log(drift_threshold / drift_floor)
    return c / (training_steps * mu_max_eff)


# ======================================================================
# TrainingAdvisor
# ======================================================================

class TrainingAdvisor:
    """Generate actionable training recommendations from phase analysis.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    width : int
        Network width.
    depth : int
        Network depth.
    init_scale : float
        Current initialisation scale σ.
    n_samples : int
        Number of data samples (affects NTK spectrum).
    training_steps : int
        Planned training duration.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        init_scale: float = 1.0,
        n_samples: int = 50,
        training_steps: int = 100,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.init_scale = init_scale
        self.n_samples = n_samples
        self.training_steps = training_steps
        self.seed = seed

        self._mu_max = _compute_mu_max_eff(
            input_dim, width, depth, n_samples, seed
        )
        self._gamma_star = _predict_gamma_star(self._mu_max, training_steps)
        self._critical_lr = (
            self._gamma_star * width / (init_scale ** 2)
            if init_scale > 0
            else float("inf")
        )

    @classmethod
    def from_arch(
        cls,
        input_dim: int,
        width: int,
        depth: int,
        init_scale: float = 1.0,
        n_samples: int = 50,
        training_steps: int = 100,
        seed: int = 42,
    ) -> "TrainingAdvisor":
        """Create an advisor from architecture specification."""
        return cls(input_dim, width, depth, init_scale, n_samples, training_steps, seed)

    @property
    def critical_lr(self) -> float:
        """The phase boundary learning rate."""
        return self._critical_lr

    @property
    def gamma_star(self) -> float:
        """The critical coupling γ*."""
        return self._gamma_star

    # ------------------------------------------------------------------
    # Learning rate recommendation
    # ------------------------------------------------------------------

    def recommend_lr(self, prefer_rich: bool = True, margin: float = 3.0) -> LRRecommendation:
        """Recommend a learning rate for the desired regime.

        Parameters
        ----------
        prefer_rich : bool
            If True, recommend LR above the phase boundary.
        margin : float
            How far from the boundary (multiplicative factor).

        Returns
        -------
        LRRecommendation
        """
        if prefer_rich:
            lr = self._critical_lr * margin
            regime = "rich"
            explanation = (
                f"Use LR={lr:.2e} to train in the rich (feature-learning) regime. "
                f"This is {margin:.1f}× above the critical LR={self._critical_lr:.2e}. "
                f"The model (width={self.width}, depth={self.depth}) will learn "
                f"task-relevant representations rather than staying near initialisation."
            )
        else:
            lr = self._critical_lr / margin
            regime = "lazy"
            explanation = (
                f"Use LR={lr:.2e} to train in the lazy (kernel) regime. "
                f"This is {margin:.1f}× below the critical LR={self._critical_lr:.2e}. "
                f"Training dynamics will follow the NTK prediction, which is "
                f"analytically tractable and gives predictable convergence."
            )

        return LRRecommendation(
            lr=lr,
            regime=regime,
            critical_lr=self._critical_lr,
            margin=margin,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Warmup schedule
    # ------------------------------------------------------------------

    def recommend_warmup(
        self, target_lr: float, warmup_fraction: float = 0.1
    ) -> WarmupSchedule:
        """Recommend a learning rate warmup schedule.

        For rich-regime training, warmup helps avoid early instability
        by starting in the lazy regime and gradually crossing the boundary.

        Parameters
        ----------
        target_lr : float
            Final (post-warmup) learning rate.
        warmup_fraction : float
            Fraction of training steps used for warmup.

        Returns
        -------
        WarmupSchedule
        """
        gamma_target = _compute_gamma(target_lr, self.init_scale, self.width)
        is_rich = gamma_target > self._gamma_star

        if not is_rich:
            # No warmup needed in lazy regime
            return WarmupSchedule(
                warmup_steps=0,
                warmup_type="none",
                initial_lr=target_lr,
                target_lr=target_lr,
                schedule=[target_lr],
                explanation=(
                    f"No warmup needed: LR={target_lr:.2e} is in the lazy regime "
                    f"(γ={gamma_target:.4e} < γ*={self._gamma_star:.4e}). "
                    f"Training is stable from the start."
                ),
            )

        warmup_steps = max(1, int(self.training_steps * warmup_fraction))
        # Start at 10% of critical LR (safely lazy), ramp to target
        initial_lr = self._critical_lr * 0.1

        # Use cosine warmup for smooth boundary crossing
        schedule = []
        for step in range(warmup_steps):
            t = step / max(1, warmup_steps - 1)
            # Cosine schedule from initial_lr to target_lr
            lr_t = initial_lr + (target_lr - initial_lr) * 0.5 * (1 - math.cos(math.pi * t))
            schedule.append(lr_t)

        # Find the step where we cross the boundary
        crossing_step = 0
        for i, lr_t in enumerate(schedule):
            g = _compute_gamma(lr_t, self.init_scale, self.width)
            if g >= self._gamma_star:
                crossing_step = i
                break

        explanation = (
            f"Cosine warmup over {warmup_steps} steps: "
            f"LR {initial_lr:.2e} → {target_lr:.2e}. "
            f"Boundary crossing at step ~{crossing_step} "
            f"(LR≈{self._critical_lr:.2e}). "
            f"This avoids instability from starting directly in the rich regime."
        )

        return WarmupSchedule(
            warmup_steps=warmup_steps,
            warmup_type="cosine",
            initial_lr=initial_lr,
            target_lr=target_lr,
            schedule=schedule,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Width recommendation
    # ------------------------------------------------------------------

    def recommend_width(
        self, target_regime: str = "rich", lr: Optional[float] = None
    ) -> WidthRecommendation:
        """Recommend a width for the desired regime at a given LR.

        Parameters
        ----------
        target_regime : str
            ``"lazy"`` or ``"rich"``.
        lr : float or None
            Learning rate. If None, uses the current critical LR.

        Returns
        -------
        WidthRecommendation
        """
        if lr is None:
            lr = self._critical_lr

        # Scan widths to find boundary
        widths = [2 ** k for k in range(4, 16)]  # 16 to 32768
        min_rich = None
        max_lazy = None

        for w in widths:
            mu = _compute_mu_max_eff(
                self.input_dim, w, self.depth, self.n_samples, self.seed
            )
            gs = _predict_gamma_star(mu, self.training_steps)
            g = _compute_gamma(lr, self.init_scale, w)

            if g > gs * 1.2 and min_rich is None:
                min_rich = w
            if g < gs * 0.8:
                max_lazy = w

        if min_rich is None:
            min_rich = widths[-1]
        if max_lazy is None:
            max_lazy = widths[0]

        if target_regime == "rich":
            rec_width = min_rich
            explanation = (
                f"Width={min_rich} is the minimum width for rich training "
                f"at LR={lr:.2e}. Increasing width further deepens the lazy "
                f"regime (γ∝1/N), so narrow networks are richer."
            )
        else:
            rec_width = max_lazy * 2  # some margin
            explanation = (
                f"Width={rec_width} ensures lazy training at LR={lr:.2e}. "
                f"The max width still in lazy regime is ~{max_lazy}; "
                f"using 2× that for safety margin."
            )

        return WidthRecommendation(
            width=rec_width,
            target_regime=target_regime,
            min_width_for_rich=min_rich,
            max_width_for_lazy=max_lazy,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Init scale recommendation
    # ------------------------------------------------------------------

    def recommend_init_scale(
        self, target_regime: str = "rich", lr: Optional[float] = None
    ) -> InitScaleRecommendation:
        """Recommend an initialisation scale for the desired regime.

        Since γ = η·σ²/N, adjusting σ shifts the effective coupling.

        Parameters
        ----------
        target_regime : str
            ``"lazy"`` or ``"rich"``.
        lr : float or None
            Learning rate. If None, uses current critical LR.

        Returns
        -------
        InitScaleRecommendation
        """
        if lr is None:
            lr = self._critical_lr

        # γ = η·σ²/N => σ² = γ·N/η
        if target_regime == "rich":
            target_gamma = self._gamma_star * 3.0
            new_scale = math.sqrt(target_gamma * self.width / lr)
            explanation = (
                f"Set σ={new_scale:.4f} (currently {self.init_scale:.4f}) "
                f"to enter the rich regime at LR={lr:.2e}. "
                f"Larger init scale ⇒ larger γ ⇒ more feature learning."
            )
        else:
            target_gamma = self._gamma_star * 0.3
            new_scale = math.sqrt(target_gamma * self.width / lr)
            explanation = (
                f"Set σ={new_scale:.4f} (currently {self.init_scale:.4f}) "
                f"to stay in the lazy regime at LR={lr:.2e}. "
                f"Smaller init scale ⇒ smaller γ ⇒ kernel-like behaviour."
            )

        return InitScaleRecommendation(
            init_scale=new_scale,
            current_scale=self.init_scale,
            target_regime=target_regime,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Full recommendation
    # ------------------------------------------------------------------

    def full_recommendation(
        self,
        prefer_rich: bool = True,
        margin: float = 3.0,
        warmup_fraction: float = 0.1,
    ) -> FullRecommendation:
        """Generate a complete training recommendation.

        Combines LR, warmup, width, and init-scale advice into a
        single actionable recommendation.

        Parameters
        ----------
        prefer_rich : bool
            If True, target the rich (feature-learning) regime.
        margin : float
            How far from boundary for LR (multiplicative factor).
        warmup_fraction : float
            Fraction of steps for warmup.

        Returns
        -------
        FullRecommendation
        """
        regime_str = "rich" if prefer_rich else "lazy"

        lr_rec = self.recommend_lr(prefer_rich=prefer_rich, margin=margin)
        warmup_rec = self.recommend_warmup(lr_rec.lr, warmup_fraction)
        width_rec = self.recommend_width(target_regime=regime_str, lr=lr_rec.lr)
        scale_rec = self.recommend_init_scale(target_regime=regime_str, lr=lr_rec.lr)

        summary = (
            f"For {regime_str} training with a "
            f"(d={self.input_dim}, w={self.width}, L={self.depth}) MLP: "
            f"Use LR={lr_rec.lr:.2e} "
            f"(critical={self._critical_lr:.2e}), "
        )
        if warmup_rec.warmup_steps > 0:
            summary += (
                f"{warmup_rec.warmup_type} warmup for {warmup_rec.warmup_steps} steps, "
            )
        summary += (
            f"σ={scale_rec.init_scale:.4f}. "
            f"Min width for rich={width_rec.min_width_for_rich}, "
            f"max width for lazy={width_rec.max_width_for_lazy}."
        )

        config = {
            "lr": lr_rec.lr,
            "init_scale": scale_rec.init_scale,
            "width": self.width,
            "depth": self.depth,
            "warmup_steps": warmup_rec.warmup_steps,
            "warmup_type": warmup_rec.warmup_type,
            "warmup_initial_lr": warmup_rec.initial_lr,
            "training_steps": self.training_steps,
            "predicted_regime": regime_str,
            "critical_lr": self._critical_lr,
            "gamma_star": self._gamma_star,
        }

        return FullRecommendation(
            lr=lr_rec,
            warmup=warmup_rec,
            width=width_rec,
            init_scale=scale_rec,
            summary=summary,
            config=config,
        )
