"""
Epsilon-race formulation with iterative calibration.

Defines epsilon-neighbourhood races and provides an iterative calibration
algorithm that refines epsilon using Lipschitz-based safety margins,
together with false-positive estimation and sensitivity analysis.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.race.definition import (
    InteractionRace,
    RaceAbsence,
    RaceClassification,
    RaceCondition,
)


# ---------------------------------------------------------------------------
# Epsilon-race
# ---------------------------------------------------------------------------

@dataclass
class EpsilonRace:
    """An interaction race within an ε-ball of a joint state.

    The epsilon-race captures the idea that a race exists not just at a
    single point but within a neighbourhood of radius ε around a joint
    state.  This accounts for the fact that agent states are known only
    approximately.

    Attributes:
        center: Centre of the ε-ball (flat joint state vector).
        epsilon: Radius of the neighbourhood.
        race: The underlying interaction race.
        lipschitz_constant: Lipschitz constant of the safety predicate
            within the neighbourhood.
        safety_margin: Minimum robustness within the ε-ball.
        false_positive_volume: Estimated volume of the region that is
            flagged as a race but is actually safe.
    """
    center: np.ndarray
    epsilon: float
    race: Optional[InteractionRace] = None
    lipschitz_constant: float = 1.0
    safety_margin: float = 0.0
    false_positive_volume: float = 0.0

    @property
    def dimension(self) -> int:
        return len(self.center)

    @property
    def ball_volume(self) -> float:
        """Volume of the ε-ball (n-dimensional hypersphere)."""
        n = self.dimension
        return (math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * self.epsilon ** n

    @property
    def false_positive_rate(self) -> float:
        """Fraction of the ε-ball that is a false positive."""
        vol = self.ball_volume
        if vol < 1e-30:
            return 0.0
        return min(1.0, self.false_positive_volume / vol)

    def contains(self, state: np.ndarray) -> bool:
        """Check if *state* is within the ε-ball."""
        return bool(np.linalg.norm(state - self.center) <= self.epsilon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "epsilon": self.epsilon,
            "lipschitz": self.lipschitz_constant,
            "safety_margin": self.safety_margin,
            "fp_rate": self.false_positive_rate,
            "dimension": self.dimension,
        }

    def __repr__(self) -> str:
        return (
            f"EpsilonRace(ε={self.epsilon:.6f}, dim={self.dimension}, "
            f"L={self.lipschitz_constant:.4f}, margin={self.safety_margin:.6f})"
        )


# ---------------------------------------------------------------------------
# Calibration record
# ---------------------------------------------------------------------------

@dataclass
class CalibrationStep:
    """Record of a single calibration iteration."""
    iteration: int
    epsilon: float
    safety_margin: float
    lipschitz_estimate: float
    false_positive_estimate: float
    converged: bool = False


# ---------------------------------------------------------------------------
# Epsilon calibrator
# ---------------------------------------------------------------------------

class EpsilonCalibrator:
    """Iteratively calibrate ε for epsilon-race detection.

    Algorithm:
        1. Initialize: ε₀ = L⁻¹ · δ_global
        2. Run abstract interpretation at current ε to compute refined
           safety margin δ.
        3. Update: ε_{k+1} = L⁻¹ · δ_k
        4. Convergence: |ε_{k+1} − ε_k| < threshold

    Args:
        lipschitz_constant: Initial Lipschitz constant estimate.
        global_safety_margin: Global safety margin δ_global.
        convergence_threshold: Convergence criterion on |Δε|.
        max_iterations: Maximum number of iterations.
        safety_margin_fn: Optional callable ``(center, epsilon) -> margin``
            that computes the refined safety margin via abstract
            interpretation.  If ``None``, uses a simple contraction.
    """

    def __init__(
        self,
        lipschitz_constant: float = 1.0,
        global_safety_margin: float = 1.0,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 50,
        safety_margin_fn: Optional[Callable[[np.ndarray, float], float]] = None,
    ) -> None:
        self._L = lipschitz_constant
        self._delta_global = global_safety_margin
        self._threshold = convergence_threshold
        self._max_iter = max_iterations
        self._margin_fn = safety_margin_fn
        self._history: List[CalibrationStep] = []

    def calibrate(
        self,
        center: np.ndarray,
        initial_epsilon: Optional[float] = None,
    ) -> EpsilonRace:
        """Run the iterative calibration and return the calibrated ``EpsilonRace``.

        Args:
            center: Centre of the neighbourhood.
            initial_epsilon: Starting ε (defaults to L⁻¹ · δ_global).

        Returns:
            Calibrated ``EpsilonRace``.
        """
        self._history.clear()
        eps = initial_epsilon if initial_epsilon is not None else self._delta_global / self._L

        for k in range(self._max_iter):
            margin = self._compute_margin(center, eps)
            new_eps = margin / self._L if self._L > 0 else eps
            new_eps = max(new_eps, 1e-12)  # prevent collapse

            fp_est = self._estimate_fp(center, new_eps, margin)

            step = CalibrationStep(
                iteration=k,
                epsilon=new_eps,
                safety_margin=margin,
                lipschitz_estimate=self._L,
                false_positive_estimate=fp_est,
                converged=abs(new_eps - eps) < self._threshold,
            )
            self._history.append(step)

            if step.converged:
                return self._build_result(center, new_eps, margin, fp_est)

            eps = new_eps

        # Did not converge — return last iterate
        return self._build_result(center, eps, margin, fp_est)

    def _compute_margin(self, center: np.ndarray, eps: float) -> float:
        if self._margin_fn is not None:
            return self._margin_fn(center, eps)
        # Default: contraction  δ_{k+1} = δ_global − L·ε_k
        return max(0.0, self._delta_global - self._L * eps)

    def _estimate_fp(
        self, center: np.ndarray, eps: float, margin: float
    ) -> float:
        """Estimate false-positive volume."""
        n = len(center)
        ball_vol = (math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * eps ** n
        if margin <= 0 or self._L <= 0:
            return ball_vol
        safe_radius = margin / self._L
        if safe_radius >= eps:
            return 0.0
        safe_vol = (math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * safe_radius ** n
        return max(0.0, ball_vol - safe_vol)

    def _build_result(
        self,
        center: np.ndarray,
        eps: float,
        margin: float,
        fp_vol: float,
    ) -> EpsilonRace:
        return EpsilonRace(
            center=center.copy(),
            epsilon=eps,
            lipschitz_constant=self._L,
            safety_margin=margin,
            false_positive_volume=fp_vol,
        )

    @property
    def history(self) -> List[CalibrationStep]:
        return list(self._history)

    @property
    def converged(self) -> bool:
        return len(self._history) > 0 and self._history[-1].converged

    @property
    def num_iterations(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"EpsilonCalibrator(L={self._L}, δ={self._delta_global}, "
            f"iterations={self.num_iterations}, converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# False-positive estimator
# ---------------------------------------------------------------------------

class FalsePositiveEstimator:
    """Estimate false-positive volume as a function of ε.

    Uses the Lipschitz constant and safety margin to bound the volume
    of the ε-ball that is flagged as a race but is actually safe.
    """

    def __init__(self, lipschitz_constant: float = 1.0) -> None:
        self._L = lipschitz_constant

    def estimate(
        self,
        center: np.ndarray,
        epsilon: float,
        safety_margin: float,
    ) -> float:
        """Return estimated false-positive volume."""
        n = len(center)
        ball_vol = self._hypersphere_volume(n, epsilon)
        safe_radius = safety_margin / self._L if self._L > 0 else 0.0
        if safe_radius >= epsilon:
            return 0.0
        safe_vol = self._hypersphere_volume(n, safe_radius)
        return max(0.0, ball_vol - safe_vol)

    def false_positive_rate(
        self,
        center: np.ndarray,
        epsilon: float,
        safety_margin: float,
    ) -> float:
        vol = self._hypersphere_volume(len(center), epsilon)
        if vol < 1e-30:
            return 0.0
        return min(1.0, self.estimate(center, epsilon, safety_margin) / vol)

    def sweep(
        self,
        center: np.ndarray,
        epsilons: Sequence[float],
        safety_margin: float,
    ) -> List[Tuple[float, float]]:
        """Compute (ε, fp_rate) for a range of ε values."""
        return [
            (eps, self.false_positive_rate(center, eps, safety_margin))
            for eps in epsilons
        ]

    @staticmethod
    def _hypersphere_volume(n: int, r: float) -> float:
        return (math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * r ** n


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class EpsilonSensitivityAnalysis:
    """Analyse how race detection results change with ε.

    Given a detection function ``detect(center, epsilon) -> bool``,
    computes the sensitivity of detection to perturbations of ε.
    """

    def __init__(
        self,
        detect_fn: Callable[[np.ndarray, float], bool],
    ) -> None:
        self._detect = detect_fn

    def sensitivity_profile(
        self,
        center: np.ndarray,
        epsilons: Sequence[float],
    ) -> List[Tuple[float, bool]]:
        """Return ``(ε, detected)`` pairs."""
        return [(eps, self._detect(center, eps)) for eps in epsilons]

    def critical_epsilon(
        self,
        center: np.ndarray,
        eps_low: float = 1e-8,
        eps_high: float = 1.0,
        tolerance: float = 1e-6,
        max_iter: int = 60,
    ) -> float:
        """Binary search for the critical ε at which detection transitions.

        Returns the smallest ε at which a race is detected.
        """
        lo, hi = eps_low, eps_high
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if self._detect(center, mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < tolerance:
                break
        return (lo + hi) / 2

    def gradient_estimate(
        self,
        center: np.ndarray,
        epsilon: float,
        robustness_fn: Callable[[np.ndarray, float], float],
        delta: float = 1e-5,
    ) -> float:
        """Finite-difference estimate of ∂robustness/∂ε."""
        r_plus = robustness_fn(center, epsilon + delta)
        r_minus = robustness_fn(center, epsilon - delta)
        return (r_plus - r_minus) / (2 * delta)


# ---------------------------------------------------------------------------
# Monotone convergence proof
# ---------------------------------------------------------------------------

class MonotoneConvergenceProof:
    """Verify that iterative calibration is monotone.

    Checks that the sequence of ε values produced by ``EpsilonCalibrator``
    is monotonically decreasing (or increasing) and converges.

    Attributes:
        history: List of ``CalibrationStep`` records.
    """

    def __init__(self, history: List[CalibrationStep]) -> None:
        self.history = list(history)

    @property
    def is_monotone_decreasing(self) -> bool:
        for i in range(1, len(self.history)):
            if self.history[i].epsilon > self.history[i - 1].epsilon + 1e-12:
                return False
        return True

    @property
    def is_monotone_increasing(self) -> bool:
        for i in range(1, len(self.history)):
            if self.history[i].epsilon < self.history[i - 1].epsilon - 1e-12:
                return False
        return True

    @property
    def is_monotone(self) -> bool:
        return self.is_monotone_decreasing or self.is_monotone_increasing

    @property
    def is_converged(self) -> bool:
        return len(self.history) > 0 and self.history[-1].converged

    @property
    def convergence_rate(self) -> float:
        """Estimate asymptotic convergence rate (ratio of successive deltas)."""
        if len(self.history) < 3:
            return 0.0
        deltas = [
            abs(self.history[i].epsilon - self.history[i - 1].epsilon)
            for i in range(1, len(self.history))
        ]
        rates = []
        for i in range(1, len(deltas)):
            if deltas[i - 1] > 1e-15:
                rates.append(deltas[i] / deltas[i - 1])
        return float(np.mean(rates)) if rates else 0.0

    def verify(self) -> Dict[str, Any]:
        """Return a verification report."""
        return {
            "num_iterations": len(self.history),
            "monotone": self.is_monotone,
            "monotone_decreasing": self.is_monotone_decreasing,
            "converged": self.is_converged,
            "convergence_rate": self.convergence_rate,
            "final_epsilon": self.history[-1].epsilon if self.history else None,
            "final_margin": self.history[-1].safety_margin if self.history else None,
        }

    def __repr__(self) -> str:
        return (
            f"MonotoneConvergenceProof(iters={len(self.history)}, "
            f"monotone={self.is_monotone}, converged={self.is_converged})"
        )
