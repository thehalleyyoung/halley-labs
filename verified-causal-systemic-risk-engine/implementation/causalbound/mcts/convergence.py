"""
PAC convergence monitor for MCTS search.

Tracks value estimates and confidence intervals per arm, computes
Hoeffding and Bernstein bounds, and provides stopping criteria and
convergence diagnostics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class ArmTracker:
    """Tracks observations for a single arm."""

    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    @property
    def variance(self) -> float:
        if len(self.values) < 2:
            return float("inf")
        return float(np.var(self.values, ddof=1))

    @property
    def std_error(self) -> float:
        if len(self.values) < 2:
            return float("inf")
        return float(np.std(self.values, ddof=1) / math.sqrt(len(self.values)))

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        self.values.append(value)
        self.timestamps.append(timestamp if timestamp is not None else time.time())


@dataclass
class ConvergenceSnapshot:
    """Point-in-time snapshot of convergence state."""

    total_rollouts: int
    best_arm: Any
    best_mean: float
    gap: float  # gap between best and second-best
    max_ci_width: float  # widest confidence interval
    is_converged: bool
    epsilon: float
    delta: float
    timestamp: float = field(default_factory=time.time)


# -----------------------------------------------------------------------
# ConvergenceMonitor
# -----------------------------------------------------------------------

class ConvergenceMonitor:
    """
    Monitor PAC convergence of MCTS search.

    Provides Hoeffding and empirical Bernstein bounds, stopping criteria,
    and convergence diagnostics.

    Parameters
    ----------
    value_range : float
        Range of possible reward values (e.g. max - min).
    default_epsilon : float
        Default accuracy parameter for convergence checks.
    default_delta : float
        Default confidence parameter for convergence checks.
    """

    def __init__(
        self,
        value_range: float = 1.0,
        default_epsilon: float = 0.05,
        default_delta: float = 0.05,
    ) -> None:
        self.value_range = value_range
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta

        self._arms: Dict[Any, ArmTracker] = defaultdict(ArmTracker)
        self._total_rollouts: int = 0
        self._convergence_curve: List[ConvergenceSnapshot] = []
        self._start_time: float = time.time()
        self._global_values: List[float] = []

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, arm: Any, value: float) -> None:
        """
        Record a new observation for the given arm.

        Parameters
        ----------
        arm : hashable
            Arm identifier.
        value : float
            Observed reward/loss value.
        """
        self._arms[arm].add(value)
        self._total_rollouts += 1
        self._global_values.append(value)

        # Update value range adaptively
        if len(self._global_values) > 1:
            observed_range = max(self._global_values) - min(self._global_values)
            if observed_range > self.value_range:
                self.value_range = observed_range

    def batch_update(self, observations: List[Tuple[Any, float]]) -> None:
        """
        Record a batch of observations.

        Parameters
        ----------
        observations : list of (arm, value) tuples
        """
        for arm, value in observations:
            self.update(arm, value)

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def get_confidence_interval(
        self,
        arm: Any,
        delta: Optional[float] = None,
        bound_type: str = "auto",
    ) -> Tuple[float, float]:
        """
        Compute a confidence interval for the arm's mean.

        Parameters
        ----------
        arm : hashable
            Arm identifier.
        delta : float or None
            Per-arm confidence level. Default uses self.default_delta / n_arms.
        bound_type : str
            "hoeffding", "bernstein", or "auto" (picks tighter bound).

        Returns
        -------
        tuple of float
            (lower_bound, upper_bound)
        """
        tracker = self._arms[arm]

        if tracker.n == 0:
            return (float("-inf"), float("inf"))

        n_arms = max(1, len(self._arms))
        if delta is None:
            delta = self.default_delta / n_arms

        mean = tracker.mean
        n = tracker.n
        R = self.value_range

        # Hoeffding bound
        hoeffding_width = R * math.sqrt(math.log(2.0 / delta) / (2.0 * n))

        if bound_type == "hoeffding":
            return (mean - hoeffding_width, mean + hoeffding_width)

        # Empirical Bernstein bound
        if n >= 5:
            var = tracker.variance
            var = max(0.0, min(var, R * R))
            log_term = math.log(3.0 / delta)
            bernstein_width = (
                math.sqrt(2.0 * var * log_term / n)
                + 3.0 * R * log_term / n
            )
        else:
            bernstein_width = hoeffding_width

        if bound_type == "bernstein":
            return (mean - bernstein_width, mean + bernstein_width)

        # Auto: use tighter bound
        width = min(hoeffding_width, bernstein_width)
        return (mean - width, mean + width)

    def get_all_confidence_intervals(
        self, delta: Optional[float] = None
    ) -> Dict[Any, Tuple[float, float]]:
        """Compute confidence intervals for all arms."""
        result = {}
        for arm in self._arms:
            result[arm] = self.get_confidence_interval(arm, delta)
        return result

    # ------------------------------------------------------------------
    # Convergence checking
    # ------------------------------------------------------------------

    def is_converged(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> bool:
        """
        Check if the search has converged to an (epsilon, delta)-PAC
        optimal arm.

        Convergence holds when the confidence interval of the best arm
        does not overlap with any other arm's interval shifted by epsilon.

        Parameters
        ----------
        epsilon : float or None
            Accuracy parameter.
        delta : float or None
            Confidence parameter.

        Returns
        -------
        bool
            True if converged.
        """
        if epsilon is None:
            epsilon = self.default_epsilon
        if delta is None:
            delta = self.default_delta

        if len(self._arms) < 2:
            # If only one arm, check if we have enough samples
            if len(self._arms) == 1:
                arm = list(self._arms.keys())[0]
                ci = self.get_confidence_interval(arm, delta)
                return (ci[1] - ci[0]) <= 2 * epsilon
            return False

        # Find the arm with the highest mean
        best_arm = self.get_best_arm()
        if best_arm is None:
            return False

        best_ci = self.get_confidence_interval(best_arm, delta)

        # Check that best arm is epsilon-separated from all others
        for arm in self._arms:
            if arm == best_arm:
                continue
            ci = self.get_confidence_interval(arm, delta)

            # If any other arm's upper CI + epsilon > best arm's lower CI,
            # we cannot distinguish them
            if ci[1] + epsilon > best_ci[0]:
                return False

        # Record convergence snapshot
        self._record_snapshot(epsilon, delta, True)
        return True

    def is_converged_successive_elimination(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Tuple[bool, List[Any]]:
        """
        Successive elimination convergence check.

        Eliminates arms whose upper confidence bound is below the lower
        confidence bound of the current best arm minus epsilon.

        Parameters
        ----------
        epsilon : float or None
            Accuracy parameter.
        delta : float or None
            Confidence parameter.

        Returns
        -------
        tuple
            (converged: bool, eliminated_arms: list)
        """
        if epsilon is None:
            epsilon = self.default_epsilon
        if delta is None:
            delta = self.default_delta

        if len(self._arms) < 2:
            return (len(self._arms) == 1, [])

        intervals = self.get_all_confidence_intervals(delta)

        # Find arm with highest lower confidence bound
        best_lcb = float("-inf")
        best_arm = None
        for arm, (lo, hi) in intervals.items():
            if lo > best_lcb:
                best_lcb = lo
                best_arm = arm

        eliminated = []
        for arm, (lo, hi) in intervals.items():
            if arm == best_arm:
                continue
            if hi + epsilon < best_lcb:
                eliminated.append(arm)

        # Converged if only one arm remains
        remaining = len(self._arms) - len(eliminated)
        return (remaining <= 1, eliminated)

    # ------------------------------------------------------------------
    # Best arm identification
    # ------------------------------------------------------------------

    def get_best_arm(self, maximize: bool = True) -> Any:
        """
        Return the arm with the best empirical mean.

        Parameters
        ----------
        maximize : bool
            If True, return arm with highest mean.

        Returns
        -------
        hashable or None
            Best arm identifier.
        """
        if not self._arms:
            return None

        if maximize:
            return max(
                self._arms.keys(),
                key=lambda a: self._arms[a].mean if self._arms[a].n > 0 else float("-inf"),
            )
        else:
            return min(
                self._arms.keys(),
                key=lambda a: self._arms[a].mean if self._arms[a].n > 0 else float("inf"),
            )

    def get_arm_ranking(
        self, maximize: bool = True, top_k: Optional[int] = None
    ) -> List[Tuple[Any, float, int]]:
        """
        Rank all arms by mean value.

        Returns
        -------
        list of (arm, mean, visit_count) tuples, sorted best-first.
        """
        items = [
            (arm, tracker.mean, tracker.n)
            for arm, tracker in self._arms.items()
            if tracker.n > 0
        ]

        items.sort(key=lambda x: x[1], reverse=maximize)

        if top_k is not None:
            items = items[:top_k]

        return items

    # ------------------------------------------------------------------
    # Remaining rollouts estimation
    # ------------------------------------------------------------------

    def estimate_remaining_rollouts(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> int:
        """
        Estimate the number of additional rollouts needed for convergence.

        Uses the current empirical variances and gap between the best
        arm and the nearest competitor.

        Parameters
        ----------
        epsilon : float or None
            Accuracy parameter.
        delta : float or None
            Confidence parameter.

        Returns
        -------
        int
            Estimated remaining rollouts (0 if already converged).
        """
        if epsilon is None:
            epsilon = self.default_epsilon
        if delta is None:
            delta = self.default_delta

        if self.is_converged(epsilon, delta):
            return 0

        n_arms = max(1, len(self._arms))
        per_arm_delta = delta / n_arms

        # Find max variance across arms
        max_var = 0.0
        for tracker in self._arms.values():
            if tracker.n >= 2:
                max_var = max(max_var, tracker.variance)

        if max_var < 1e-12:
            max_var = (self.value_range ** 2) / 4.0  # worst case

        # Bernstein-based estimate: n >= (2 * sigma^2 + 2*R*eps/3) * log(3/delta_a) / eps^2
        log_term = math.log(3.0 / per_arm_delta)
        n_per_arm = math.ceil(
            (2.0 * max_var + 2.0 * self.value_range * epsilon / 3.0) * log_term
            / (epsilon * epsilon)
        )

        total_needed = n_per_arm * n_arms

        # Subtract rollouts already done
        remaining = max(0, total_needed - self._total_rollouts)
        return remaining

    def estimate_time_remaining(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> float:
        """
        Estimate time remaining based on current rollout rate.

        Returns
        -------
        float
            Estimated seconds remaining.
        """
        remaining_rollouts = self.estimate_remaining_rollouts(epsilon, delta)
        elapsed = time.time() - self._start_time

        if self._total_rollouts == 0:
            return float("inf")

        rate = self._total_rollouts / elapsed  # rollouts per second
        return remaining_rollouts / rate

    # ------------------------------------------------------------------
    # Convergence curve
    # ------------------------------------------------------------------

    def get_convergence_curve(self) -> List[Dict[str, Any]]:
        """
        Return the convergence curve as a list of snapshots.

        Each snapshot contains total_rollouts, best_mean, gap,
        max_ci_width, and whether convergence was achieved.
        """
        return [
            {
                "total_rollouts": s.total_rollouts,
                "best_arm": s.best_arm,
                "best_mean": s.best_mean,
                "gap": s.gap,
                "max_ci_width": s.max_ci_width,
                "is_converged": s.is_converged,
                "epsilon": s.epsilon,
                "delta": s.delta,
                "timestamp": s.timestamp,
            }
            for s in self._convergence_curve
        ]

    def record_snapshot(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> ConvergenceSnapshot:
        """
        Explicitly record a convergence snapshot at the current state.

        Returns
        -------
        ConvergenceSnapshot
        """
        if epsilon is None:
            epsilon = self.default_epsilon
        if delta is None:
            delta = self.default_delta
        return self._record_snapshot(epsilon, delta, self.is_converged(epsilon, delta))

    def _record_snapshot(
        self,
        epsilon: float,
        delta: float,
        converged: bool,
    ) -> ConvergenceSnapshot:
        """Internal snapshot recording."""
        best_arm = self.get_best_arm()
        best_mean = self._arms[best_arm].mean if best_arm is not None and self._arms[best_arm].n > 0 else 0.0

        # Compute gap
        gap = float("inf")
        if best_arm is not None and len(self._arms) >= 2:
            second_best = float("-inf")
            for arm, tracker in self._arms.items():
                if arm == best_arm or tracker.n == 0:
                    continue
                second_best = max(second_best, tracker.mean)
            if second_best > float("-inf"):
                gap = best_mean - second_best

        # Max CI width
        max_width = 0.0
        for arm in self._arms:
            ci = self.get_confidence_interval(arm, delta)
            width = ci[1] - ci[0]
            if width < float("inf"):
                max_width = max(max_width, width)

        snapshot = ConvergenceSnapshot(
            total_rollouts=self._total_rollouts,
            best_arm=best_arm,
            best_mean=best_mean,
            gap=gap,
            max_ci_width=max_width,
            is_converged=converged,
            epsilon=epsilon,
            delta=delta,
        )

        self._convergence_curve.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_convergence_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive convergence statistics.

        Returns
        -------
        dict
            Contains total_rollouts, n_arms, best_arm, best_mean,
            convergence_rate, estimated_remaining, elapsed_time, etc.
        """
        elapsed = time.time() - self._start_time
        best_arm = self.get_best_arm()

        # Convergence rate: how quickly the best mean is stabilizing
        convergence_rate = self._compute_convergence_rate()

        stats: Dict[str, Any] = {
            "total_rollouts": self._total_rollouts,
            "n_arms": len(self._arms),
            "elapsed_seconds": elapsed,
            "rollouts_per_second": self._total_rollouts / elapsed if elapsed > 0 else 0.0,
            "best_arm": best_arm,
            "best_mean": self._arms[best_arm].mean if best_arm and self._arms[best_arm].n > 0 else None,
            "convergence_rate": convergence_rate,
            "estimated_remaining_rollouts": self.estimate_remaining_rollouts(),
            "estimated_remaining_seconds": self.estimate_time_remaining(),
            "is_converged": self.is_converged(),
        }

        # Per-arm stats
        arm_stats = {}
        for arm, tracker in self._arms.items():
            ci = self.get_confidence_interval(arm)
            arm_stats[str(arm)] = {
                "n": tracker.n,
                "mean": tracker.mean,
                "variance": tracker.variance if tracker.n >= 2 else None,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "ci_width": ci[1] - ci[0] if ci[1] < float("inf") else None,
            }
        stats["arm_details"] = arm_stats

        return stats

    def _compute_convergence_rate(self) -> float:
        """
        Estimate the convergence rate from the convergence curve.

        Uses the reduction in max CI width over the last few snapshots.
        """
        if len(self._convergence_curve) < 3:
            return 0.0

        recent = self._convergence_curve[-10:]
        widths = [s.max_ci_width for s in recent if s.max_ci_width < float("inf")]

        if len(widths) < 2:
            return 0.0

        # Rate = (first_width - last_width) / first_width
        first = widths[0]
        last = widths[-1]

        if first < 1e-12:
            return 1.0

        rate = (first - last) / first
        return max(0.0, min(1.0, rate))

    def get_global_statistics(self) -> Dict[str, Any]:
        """Return statistics across all observations."""
        if not self._global_values:
            return {"n": 0}

        values = np.array(self._global_values)
        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracking state."""
        self._arms.clear()
        self._total_rollouts = 0
        self._convergence_curve.clear()
        self._start_time = time.time()
        self._global_values.clear()
